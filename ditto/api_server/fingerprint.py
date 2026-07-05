"""Content fingerprint for the anti-copy gate.

The anti-copy gate in :mod:`ditto.api_server.scoring_gate` starts from two cheap
signals: an exact ``sha256`` match and a size+score-proximity heuristic. Both are
*byte*-level — a copier who re-indents the source or renames files moves the
tarball size past the heuristic's tolerance and dodges it. This module adds a
*content*-level signal that survives those edits.

The fingerprint is the **set of normalized per-file content hashes** of the
regular files in the tarball:

- The filename is not part of a file's hash, so renaming files does not change
  the fingerprint (only the *set* of contents matters, and the set is order-free
  so reordering files is invisible too).
- Each file's bytes are normalized before hashing — decoded leniently, every
  line stripped of leading/trailing whitespace, blank lines dropped — so
  re-indentation and trailing-whitespace churn collapse to the same hash.

Two fingerprints are compared with Jaccard similarity (:func:`content_jaccard`):
``|A ∩ B| / |A ∪ B|``. A lightly-tweaked copy shares almost every file with its
source and scores near 1.0; two genuinely different harnesses share only the
common scaffolding and score far lower.

This is a **similarity signal, not an AST/semantic detector**: it does not defeat
identifier renaming or logic reordering *within* a file — the doc routes that
heavier analysis to the screener/dittobench where the tree is already unpacked.
It is computed here because ``/upload/agent`` already holds the whole tarball in
memory (streamed for the size cap + sha256), so the platform gets the signal for
free without a second unpack. The functions are pure + deterministic so the same
tarball always yields the same fingerprint (and the gate the same verdict).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import logging
import tarfile

logger = logging.getLogger(__name__)

# Decompression-bomb guards. The upload cap bounds the *compressed* tarball at
# 2 MiB, but gzip/tar can inflate far past that, so fingerprinting reads through
# its own independent limits. When either is tripped the tarball is treated as
# unfingerprintable (returns ``None``) rather than partially hashed — a partial
# fingerprint could spuriously match or mask a copy, and an abusive tarball is
# not something the moderation signal should try to reason about.
_MAX_TOTAL_BYTES = 64 * 1024 * 1024  # matches the dittobench sandbox extract cap
_MAX_FILES = 5000
# Per-file read cap: no single member may pull more than this from the stream.
_MAX_FILE_BYTES = 8 * 1024 * 1024


def compute_content_fingerprint(tar_gz_bytes: bytes) -> list[str] | None:
    """Return the sorted set of normalized per-file content hashes, or ``None``.

    Opens ``tar_gz_bytes`` as a gzipped tar in memory and hashes each regular
    file's normalized contents (see module docstring). Returns the hex hashes as
    a sorted, de-duplicated list (stable across runs, JSON-serializable for the
    ``agents.content_fingerprint`` column).

    Returns ``None`` — meaning "no usable fingerprint", which the gate treats as
    *no content match* — when the bytes are not a readable tar.gz, contain no
    regular files, or trip the decompression-bomb guards. Fingerprinting is a
    best-effort moderation signal layered on top of the already-verified upload,
    so it never raises into the upload path: a hostile or corrupt tarball that
    defeats it simply gets no content signal (the sha256 + size signals still
    apply, and the validator/screener still reject a broken harness downstream).
    """
    hashes: set[str] = set()
    total = 0
    count = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_gz_bytes), mode="r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                count += 1
                if count > _MAX_FILES:
                    logger.warning("fingerprint: >%d files, skipping", _MAX_FILES)
                    return None
                extracted = tar.extractfile(member)
                if extracted is None:  # e.g. a hardlink with no data
                    continue
                # Bounded read: +1 so an over-cap file is detected, not silently
                # truncated into a hash that a smaller honest file could collide.
                raw = extracted.read(_MAX_FILE_BYTES + 1)
                if len(raw) > _MAX_FILE_BYTES:
                    logger.warning("fingerprint: file exceeds per-file cap")
                    return None
                total += len(raw)
                if total > _MAX_TOTAL_BYTES:
                    logger.warning("fingerprint: >%d bytes, skipping", _MAX_TOTAL_BYTES)
                    return None
                digest = _normalized_hash(raw)
                if digest is not None:
                    hashes.add(digest)
    except (tarfile.TarError, gzip.BadGzipFile, EOFError, OSError) as e:
        # Not a readable tar.gz. The upload already matched the miner-signed
        # sha256, so a junk body here is the miner's problem downstream, not a
        # reason to fail intake — fall through to "no fingerprint".
        logger.info("fingerprint: unreadable tarball (%s)", type(e).__name__)
        return None

    if not hashes:
        return None
    return sorted(hashes)


def _normalized_hash(raw: bytes) -> str | None:
    """Hash file bytes after normalizing away indentation + whitespace churn.

    Decodes leniently (``errors="replace"`` so binary blobs still hash stably),
    strips each line, and drops blank lines, so a re-indented or
    trailing-whitespace-edited copy of a file collapses to the same digest as its
    original. Case and token identity are preserved — this defeats reformatting,
    not renaming-of-identifiers (that heavier analysis is deferred to the
    screener/dittobench).

    Returns ``None`` when the file is empty after normalization (blank or
    whitespace-only): such a file carries no content, so folding the hash of the
    empty string into every such submission's fingerprint would only add a shared
    token that inflates Jaccard between unrelated harnesses.
    """
    text = raw.decode("utf-8", errors="replace")
    lines = [stripped for line in text.splitlines() if (stripped := line.strip())]
    if not lines:
        return None
    normalized = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def content_jaccard(a: list[str] | None, b: list[str] | None) -> float:
    """Return the Jaccard similarity of two content fingerprints in ``[0, 1]``.

    ``|A ∩ B| / |A ∪ B|`` over the two hash sets. Returns ``0.0`` when either
    fingerprint is missing or empty (nothing to compare ⇒ no content match), so a
    caller can gate on the score without special-casing ``None``.
    """
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)
