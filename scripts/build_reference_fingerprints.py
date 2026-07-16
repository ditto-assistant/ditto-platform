"""Build reference-shingle bundles from an official starter-kit clone.

Usage::

    uv run python scripts/build_reference_fingerprints.py /path/to/starter-kit

The clone must contain the complete official history.  Output is deterministic:
three sorted big-endian uint64 streams plus a manifest that pins every included
commit.  Generated bundles are committed with the fingerprint algorithm so upload
processing never needs network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from ditto.api_server.fingerprint import (
    _file_shingles,
    _normalized_source_shingles,
    _prompt_shingles,
)

OUTPUT = Path(__file__).parents[1] / "ditto" / "anticopy"


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def _write_bundle(name: str, shingles: set[str]) -> None:
    (OUTPUT / name).write_bytes(
        b"".join(int(shingle, 16).to_bytes(8, "big") for shingle in sorted(shingles))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("starter_repo", type=Path)
    parser.add_argument(
        "--revision",
        default="origin/main",
        help="authoritative default-branch lineage to include (default: origin/main)",
    )
    args = parser.parse_args()

    resolved_revision = (
        _git(args.starter_repo, "rev-parse", args.revision).decode().strip()
    )
    commits = sorted(
        _git(args.starter_repo, "rev-list", args.revision).decode().splitlines()
    )
    blobs: set[str] = set()
    for commit in commits:
        for line in (
            _git(args.starter_repo, "ls-tree", "-r", commit).decode().splitlines()
        ):
            metadata = line.split("\t", 1)[0].split()
            if len(metadata) >= 3 and metadata[1] == "blob":
                blobs.add(metadata[2])

    lexical: set[str] = set()
    normalized: set[str] = set()
    prompt: set[str] = set()
    for oid in sorted(blobs):
        raw = _git(args.starter_repo, "cat-file", "blob", oid)
        lexical.update(_file_shingles(raw))
        normalized.update(_normalized_source_shingles(raw))
        prompt.update(_prompt_shingles(raw))

    bundles = {
        "reference_lexical_v2.bin": lexical,
        "reference_normalized_v2.bin": normalized,
        "reference_prompt_v2.bin": prompt,
    }
    for name, shingles in bundles.items():
        _write_bundle(name, shingles)

    manifest = {
        "format": "sorted-big-endian-uint64-v1",
        "source": "https://github.com/ditto-assistant/dittobench-starter-kit",
        "revision": resolved_revision,
        "requested_revision": args.revision,
        "commits": commits,
        "commit_set_sha256": hashlib.sha256("\n".join(commits).encode()).hexdigest(),
        "unique_blobs": len(blobs),
        "bundles": {name: len(shingles) for name, shingles in bundles.items()},
    }
    (OUTPUT / "reference_manifest_v2.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
