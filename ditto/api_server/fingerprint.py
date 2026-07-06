"""Content fingerprint for the anti-copy gate.

The anti-copy gate in :mod:`ditto.api_server.scoring_gate` starts from two cheap
byte-level signals — an exact ``sha256`` match and a size+score-proximity
heuristic — both of which a copier defeats by re-indenting, renaming files, or
nudging the tarball size. This module adds a *content*-level signal that survives
reformatting and localized edits.

**How it works.** Each regular file's text is normalized (decoded leniently, all
intra-line whitespace removed, blank lines dropped) so indentation,
tabs-vs-spaces, line-endings, and operator/keyword-spacing reformatting all
wash out. The normalized lines are cut into overlapping **k-line shingles**
(:data:`_SHINGLE_LINES`); every shingle is hashed, and the shingles from all files
are unioned into one set. Shingling is per-file then unioned, so renaming or
reordering files is invisible and a *localized* edit only disturbs the handful of
shingles that span it — unlike a whole-file hash, which any single edit voids.

**Storage — a MinHash (bottom-k) sketch.** Keeping the full shingle set would grow
with harness size; instead the fingerprint stores the ``k`` smallest shingle
hashes (a KMV / bottom-k MinHash sketch) plus the true set cardinality. That is a
*fixed*-size summary (:data:`_MINHASH_K` hashes) from which :func:`content_similarity`
estimates both Jaccard and containment. For a harness whose shingle set is smaller
than ``k`` (the common case) the sketch is the whole set, so the estimate is
*exact*; the approximation only engages for unusually large trees.

**What it catches / misses.** Jaccard catches sprinkled small edits; containment
(the symmetric overlap coefficient) catches a copy that pads itself with junk
files to dilute Jaccard. It is still a *lexical* signal: it does **not** defeat
identifier renaming or statement reordering *within* a shingle window — that needs
language-aware AST/token analysis, which belongs in the screener/dittobench where
the crate is already unpacked and its Rust toolchain is available (the platform
has no Rust parser). See ``docs`` handoff for that layer.

Computed here because ``/upload/agent`` already holds the whole tarball in memory
(streamed for the size cap + sha256), so the platform gets the signal without a
second unpack. Everything is pure + deterministic: the same tarball always yields
the same sketch, and the gate the same verdict.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import logging
import tarfile

logger = logging.getLogger(__name__)

# Bump when the normalization / shingling / sketch format changes so stored
# fingerprints from an older algorithm are not silently compared against new ones
# (a cross-version Jaccard is meaningless). :func:`content_similarity` returns no
# match unless both sketches carry the same version.
_FP_VERSION = 1

# A shingle is this many consecutive normalized lines. Small enough that a
# one-line edit disturbs only a few shingles (robust to sprinkled edits), large
# enough that a shingle is distinctive (common single lines like a lone ``}`` do
# not collide across unrelated crates).
_SHINGLE_LINES = 4
# Bottom-k sketch size. 256 keeps the stored fingerprint ~4 KB and, because most
# harness shingle sets are smaller than this, makes the similarity estimate exact
# for the common case (the sketch is then the whole set).
_MINHASH_K = 256
# Width of each shingle hash: 64 bits as zero-padded hex. Fixed width so the hex
# strings sort in the same order as the integers they encode (bottom-k == the k
# lexicographically-smallest), and JSON-safe (no >2^53 bigints on the wire).
_HASH_HEX = 16

# Decompression-bomb + work guards. The upload cap bounds the *compressed* tarball
# at 2 MiB, but gzip/tar inflate far past that, so fingerprinting reads through its
# own limits. Tripping any of them yields ``None`` ("unfingerprintable"), which the
# gate reads as no content match (the sha256 + size signals still apply).
_MAX_TOTAL_BYTES = 64 * 1024 * 1024  # matches the dittobench sandbox extract cap
_MAX_FILE_BYTES = 8 * 1024 * 1024
# Bound on *every* iterated tar member, not just regular files: a tar of hundreds
# of thousands of directory/symlink headers is a cheap CPU-DoS otherwise (the
# member walk is O(headers) regardless of file content).
_MAX_MEMBERS = 20000
# Hard ceiling on distinct shingles before we give up (paired with the byte caps).
_MAX_SHINGLES = 500_000


def compute_content_fingerprint(tar_gz_bytes: bytes) -> dict | None:
    """Return a MinHash shingle sketch of the tarball's source, or ``None``.

    The returned dict is ``{"v", "k", "card", "m"}`` — algorithm version, sketch
    budget, true shingle-set cardinality, and the sorted bottom-``k`` shingle
    hashes — JSON-serializable for the ``agents.content_fingerprint`` column and
    consumed by :func:`content_similarity`.

    Returns ``None`` — "no usable fingerprint", which the gate treats as no
    content match — when the bytes are not a readable tar.gz, contain no source
    lines, or trip the bomb/work guards. Fingerprinting is a best-effort
    moderation signal layered on an already-verified upload, so it never raises
    into the upload path: a hostile or corrupt tarball simply gets no content
    signal (the validator/screener still reject a broken harness downstream).
    """
    shingles: set[str] = set()
    total = 0
    members = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_gz_bytes), mode="r:gz") as tar:
            for member in tar:
                members += 1
                if members > _MAX_MEMBERS:
                    logger.warning("fingerprint: >%d members, skipping", _MAX_MEMBERS)
                    return None
                if not member.isfile():
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:  # e.g. a hardlink carrying no data
                    continue
                # Bounded read: +1 so an over-cap file is detected rather than
                # silently truncated into a hash a smaller honest file could hit.
                raw = extracted.read(_MAX_FILE_BYTES + 1)
                if len(raw) > _MAX_FILE_BYTES:
                    logger.warning("fingerprint: file exceeds per-file cap")
                    return None
                total += len(raw)
                if total > _MAX_TOTAL_BYTES:
                    logger.warning("fingerprint: >%d bytes, skipping", _MAX_TOTAL_BYTES)
                    return None
                for shingle in _file_shingles(raw):
                    shingles.add(shingle)
                    if len(shingles) > _MAX_SHINGLES:
                        logger.warning("fingerprint: >%d shingles", _MAX_SHINGLES)
                        return None
    except (tarfile.TarError, gzip.BadGzipFile, EOFError, OSError) as e:
        logger.info("fingerprint: unreadable tarball (%s)", type(e).__name__)
        return None

    if not shingles:
        return None
    return {
        "v": _FP_VERSION,
        "k": _MINHASH_K,
        "card": len(shingles),
        "m": sorted(shingles)[:_MINHASH_K],
    }


def _file_shingles(raw: bytes) -> list[str]:
    """Return the hashed k-line shingles of one file's normalized text.

    Normalization decodes leniently (``errors="replace"`` so binary blobs still
    hash stably), removes *all* intra-line whitespace, and drops blank lines — so
    indentation, tabs-vs-spaces, line-endings, and operator/keyword spacing
    reformatting (what a formatter like rustfmt churns) all normalize away. Token
    identity, ordering, and case within a line are preserved (defeating those is
    the AST layer's job). Two distinct files would have to share whitespace-free
    4-line windows to collide, which in practice means the same code. A file with
    fewer than :data:`_SHINGLE_LINES` non-blank lines yields one whole-file shingle.
    """
    text = raw.decode("utf-8", errors="replace")
    lines = [norm for line in text.splitlines() if (norm := "".join(line.split()))]
    if not lines:
        return []
    k = _SHINGLE_LINES
    if len(lines) <= k:
        return [_hash_shingle("\n".join(lines))]
    return [
        _hash_shingle("\n".join(lines[i : i + k])) for i in range(len(lines) - k + 1)
    ]


def _hash_shingle(shingle: str) -> str:
    """Hash one shingle to a fixed-width hex string (top 64 bits of sha256)."""
    return hashlib.sha256(shingle.encode("utf-8")).hexdigest()[:_HASH_HEX]


def content_similarity(a: dict | None, b: dict | None) -> tuple[float, float]:
    """Estimate ``(jaccard, containment)`` of two fingerprint sketches in ``[0, 1]``.

    ``jaccard`` = ``|A ∩ B| / |A ∪ B|`` (dilutes when a copy pads itself with junk
    files). ``containment`` = ``|A ∩ B| / min(|A|, |B|)`` (the symmetric overlap
    coefficient — ~1.0 when the smaller shingle set is essentially a subset of the
    larger, so it still fires on a padded copy). Both are estimated from the
    bottom-k sketches and are exact when both sets are small enough to be fully
    retained (``card <= k``).

    Returns ``(0.0, 0.0)`` when either sketch is missing, empty, or carries a
    different sketch-format version (a cross-version comparison is meaningless), so
    the gate can threshold without special-casing ``None``. The two channels
    (lexical / structural) are isolated by storage column, and each compares only
    within its own version — hence the equality check on ``v`` rather than a
    hard-coded constant, so a channel can version its format independently.
    """
    if not a or not b:
        return (0.0, 0.0)
    va = a.get("v")
    if va is None or va != b.get("v"):
        return (0.0, 0.0)
    ma, mb = set(a.get("m", ())), set(b.get("m", ()))
    if not ma or not mb:
        return (0.0, 0.0)
    ka, kb = int(a.get("k", _MINHASH_K)), int(b.get("k", _MINHASH_K))
    card_a, card_b = int(a.get("card", 0)), int(b.get("card", 0))

    # Jaccard (KMV): over the k smallest hashes of the union (each guaranteed to sit
    # in one of the two bottom-k sketches), the fraction lying in *both* sketches —
    # i.e. in the intersection — estimates the Jaccard index.
    sample = sorted(ma | mb)[: min(ka, kb)]
    if not sample:
        return (0.0, 0.0)
    jaccard = sum(1 for v in sample if v in ma and v in mb) / len(sample)

    # Containment estimated DIRECTLY, not derived from Jaccard: the algebraic
    # I = J*(|A|+|B|)/(1+J) is exact for the true J but multiplies the KMV sample
    # error by (|A|+|B|), which is huge and one-sided exactly in the asymmetric
    # (padded-copy) regime — biasing containment upward into false holds. Instead
    # condition on the SMALLER set's min-hashes restricted to the range the larger
    # set's sketch actually observes: for a min-hash x of the smaller set with
    # x <= the larger set's k-th smallest hash, membership in the larger set is
    # fully determined (x in larger  iff  x in larger's sketch). The shared fraction
    # of those observable min-hashes estimates containment unbiasedly, and is exact
    # when the larger set is fully retained.
    if card_a <= card_b:
        small, large, large_k, large_card = ma, mb, kb, card_b
    else:
        small, large, large_k, large_card = mb, ma, ka, card_a
    if large_card <= large_k:
        observable = small  # larger set fully retained: every membership is known
    else:
        tau = max(large)  # the larger set's k-th smallest hash
        observable = {x for x in small if x <= tau}
    if not observable:
        return (jaccard, 0.0)
    containment = sum(1 for x in observable if x in large) / len(observable)
    return (jaccard, containment)
