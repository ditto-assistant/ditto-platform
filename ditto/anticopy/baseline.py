"""Shared-scaffolding baseline corpus for novelty-aware anti-copy comparison.

Nearly every SN118 submission is derived from the public ``dittobench-starter-kit``
reference harness, so any two honest submissions share almost all of their
tarball text: the kit alone contributes ~107k distinct shingles, while a
competitive edit touches a handful of lines. Whole-tarball similarity therefore
saturates at ~1.0 between *independent* miners — the anti-copy gate's dominant
false-positive source — and a bottom-k sketch of the whole tarball is
statistically blind to the few shingles that are actually the miner's own work.

This module loads the baseline corpus: the full shingle-hash set of the public
starter kit (optionally several released versions, unioned). The upload path
subtracts it *before* sketching (:func:`ditto.api_server.fingerprint.
compute_content_fingerprint` with ``exclude=``), so the stored sketch describes
only the submission's **novel** content — what a copier would actually have to
steal — at full resolution.

The corpus file is gzipped text, one 16-hex shingle hash per line, generated
with::

    uv run python -m ditto.anticopy.baseline generate \
        --git /path/to/dittobench-starter-kit \
        --out ditto/anticopy/data/starter-kit-baseline.txt.gz

Regenerate (unioning the previous file via ``--merge``) whenever a new starter
kit version is published, or honest submissions tracking the new kit will carry
the kit delta as "novelty" shared with every other up-to-date miner.

``ANTICOPY_BASELINE_FILE`` overrides the packaged corpus path; a missing file
disables subtraction (legacy whole-tarball fingerprints, version 1).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import logging
import os
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"
_DEFAULT_BASELINE_FILE = _DATA_DIR / "starter-kit-baseline.txt.gz"
_ENV_VAR = "ANTICOPY_BASELINE_FILE"
# Identity is the sha256 of the decompressed, sorted hash lines: two deploys
# comparing sketches must have subtracted the same corpus or the comparison is
# meaningless (content_similarity refuses mismatched baseline ids).
_BASELINE_ID_HEX = 12


@dataclass(frozen=True)
class BaselineCorpus:
    """A loaded shared-scaffolding corpus: shingle hashes + stable identity."""

    shingles: frozenset[str]
    baseline_id: str


def _baseline_path() -> Path:
    override = os.environ.get(_ENV_VAR)
    return Path(override) if override else _DEFAULT_BASELINE_FILE


@lru_cache(maxsize=1)
def _load_cached(path_str: str, _mtime_ns: int) -> BaselineCorpus:
    raw = gzip.decompress(Path(path_str).read_bytes())
    lines = sorted({line for line in raw.decode().split("\n") if line})
    canonical = "\n".join(lines).encode()
    return BaselineCorpus(
        shingles=frozenset(lines),
        baseline_id=hashlib.sha256(canonical).hexdigest()[:_BASELINE_ID_HEX],
    )


def load_baseline() -> BaselineCorpus | None:
    """Load the baseline corpus, or ``None`` when no corpus file exists.

    Cached per (path, mtime), so a redeploy that ships a new corpus file is
    picked up without a restart-order dependency, while steady-state calls are
    dictionary lookups.
    """
    path = _baseline_path()
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        logger.warning("anticopy baseline corpus missing at %s", path)
        return None
    return _load_cached(str(path), mtime_ns)


# --- generator CLI ----------------------------------------------------------


def _tarball_shingles(tar_gz_bytes: bytes) -> set[str]:
    # Local import: fingerprint imports nothing from here, keeping the
    # dependency one-directional (api_server -> anticopy would be a cycle).
    from ditto.api_server.fingerprint import _file_shingles

    shingles: set[str] = set()
    with tarfile.open(fileobj=io.BytesIO(tar_gz_bytes), mode="r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            shingles.update(_file_shingles(extracted.read()))
    return shingles


def _git_archive(repo: Path) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar.gz", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="build a baseline corpus file")
    gen.add_argument(
        "tarballs", nargs="*", type=Path, help="starter-kit .tar.gz files to union"
    )
    gen.add_argument(
        "--git",
        action="append",
        default=[],
        type=Path,
        help="starter-kit git checkout; HEAD is archived and unioned",
    )
    gen.add_argument(
        "--merge",
        action="append",
        default=[],
        type=Path,
        help="existing baseline .txt.gz file(s) to union in",
    )
    gen.add_argument("--out", type=Path, required=True, help="output .txt.gz path")
    args = parser.parse_args(argv)

    shingles: set[str] = set()
    for tarball in args.tarballs:
        shingles |= _tarball_shingles(tarball.read_bytes())
    for repo in args.git:
        shingles |= _tarball_shingles(_git_archive(repo))
    for merge in args.merge:
        shingles |= set(gzip.decompress(merge.read_bytes()).decode().split("\n")) - {""}
    if not shingles:
        parser.error("no shingles collected; pass tarballs, --git, or --merge")

    canonical = "\n".join(sorted(shingles)).encode()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 and an empty embedded filename so regenerating identical
    # content yields byte-identical output regardless of the output path.
    with (
        args.out.open("wb") as handle,
        gzip.GzipFile(filename="", fileobj=handle, mode="wb", mtime=0) as gz,
    ):
        gz.write(canonical)
    baseline_id = hashlib.sha256(canonical).hexdigest()[:_BASELINE_ID_HEX]
    print(f"wrote {args.out}: {len(shingles)} shingles, baseline_id={baseline_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
