"""Unit tests for the content fingerprint :mod:`ditto.api_server.fingerprint`."""

from __future__ import annotations

import gzip
import hashlib
import io
import tarfile

from ditto.api_server.fingerprint import (
    _FP_VERSION,
    _MINHASH_K,
    compute_content_fingerprint,
    compute_normalized_source_hash,
    content_similarity,
)


def _tar_gz(files: dict[str, bytes]) -> bytes:
    """Pack ``{name: bytes}`` into a gzipped tar and return the bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _rust_file(n_fns: int, prefix: str = "f") -> bytes:
    """A plausibly-sized Rust source file with ``n_fns`` small functions."""
    body = "\n".join(
        f"fn {prefix}{i}(x: i64) -> i64 {{\n    let y = x + {i};\n    y * 2\n}}"
        for i in range(n_fns)
    )
    return body.encode()


def _jaccard(a: dict | None, b: dict | None) -> float:
    return content_similarity(a, b)[0]


def _containment(a: dict | None, b: dict | None) -> float:
    return content_similarity(a, b)[1]


class TestNormalizedSourceHash:
    """L3a exact-repack hash: cosmetic repackaging normalizes to the same hash,
    genuinely different source does not, and string literals are preserved."""

    def test_shape_and_determinism(self) -> None:
        tar = _tar_gz({"src/lib.rs": _rust_file(5)})
        h1 = compute_normalized_source_hash(tar)
        h2 = compute_normalized_source_hash(tar)
        assert isinstance(h1, str) and len(h1) == 64  # sha256 hex
        assert h1 == h2

    def test_comments_stripped(self) -> None:
        plain = b"fn a(x: i64) -> i64 {\n    x + 1\n}\n"
        commented = (
            b"// a doc comment\n"
            b"fn a(x: i64) -> i64 {\n"
            b"    /* inline */ x + 1  // trailing\n"
            b"}\n"
            b"/* trailing\n   block */\n"
        )
        assert compute_normalized_source_hash(
            _tar_gz({"lib.rs": plain})
        ) == compute_normalized_source_hash(_tar_gz({"lib.rs": commented}))

    def test_reindent_and_reformat_absorbed(self) -> None:
        a = b"fn a(x:i64)->i64{\nx+1\n}\n"
        b = b"fn  a( x : i64 ) -> i64 {\n\n        x  +  1\n\n}\n"
        assert compute_normalized_source_hash(
            _tar_gz({"lib.rs": a})
        ) == compute_normalized_source_hash(_tar_gz({"lib.rs": b}))

    def test_rename_and_reorder_files_invisible(self) -> None:
        f1, f2 = _rust_file(3, "a"), _rust_file(4, "b")
        original = _tar_gz({"src/one.rs": f1, "src/two.rs": f2})
        # different file names, reverse insertion order — same source content
        renamed = _tar_gz({"src/zzz.rs": f2, "src/aaa.rs": f1})
        assert compute_normalized_source_hash(
            original
        ) == compute_normalized_source_hash(renamed)

    def test_string_literal_double_slash_preserved(self) -> None:
        # A `//` inside a string is NOT a comment; changing the URL must change
        # the hash (proving the string body is kept, not stripped as a comment).
        a = b'fn u() -> &\'static str { "http://a.example/x" }\n'
        b = b'fn u() -> &\'static str { "http://b.example/y" }\n'
        ha = compute_normalized_source_hash(_tar_gz({"lib.rs": a}))
        hb = compute_normalized_source_hash(_tar_gz({"lib.rs": b}))
        assert ha is not None and hb is not None and ha != hb

    def test_distinct_source_differs(self) -> None:
        ha = compute_normalized_source_hash(_tar_gz({"lib.rs": _rust_file(5, "a")}))
        hb = compute_normalized_source_hash(_tar_gz({"lib.rs": _rust_file(5, "b")}))
        assert ha != hb

    def test_only_regular_files_and_empty_none(self) -> None:
        assert compute_normalized_source_hash(b"not a tarball") is None
        assert compute_normalized_source_hash(_tar_gz({})) is None
        # a tar of only comments/whitespace normalizes to empty -> None
        only_comment = _tar_gz({"x.rs": b"// just a comment\n"})
        assert compute_normalized_source_hash(only_comment) is None


class TestComputeContentFingerprint:
    def test_shape_and_determinism(self) -> None:
        fp = compute_content_fingerprint(_tar_gz({"src/lib.rs": _rust_file(10)}))
        assert fp is not None
        assert fp["v"] == _FP_VERSION and fp["k"] == _MINHASH_K
        assert fp["card"] >= 1 and fp["m"] == sorted(fp["m"])
        assert len(fp["m"]) <= _MINHASH_K
        # Deterministic.
        assert fp == compute_content_fingerprint(
            _tar_gz({"src/lib.rs": _rust_file(10)})
        )

    def test_self_similarity_is_one(self) -> None:
        fp = compute_content_fingerprint(_tar_gz({"a.rs": _rust_file(12)}))
        assert _jaccard(fp, fp) == 1.0
        assert _containment(fp, fp) == 1.0

    def test_reindent_and_reformat_absorbed(self) -> None:
        # Leading indent change, tabs, CRLF, AND operator-spacing reformat all wash
        # out — the whole-file whitespace churn a formatter (rustfmt) produces.
        orig = b"fn f(x: i64) -> i64 {\n    let y = x + 1;\n    y * 2\n}\n"
        reformatted = b"fn f(x:i64)->i64{\r\n\t\tlet  y = x+1 ;\r\n\t\ty*2\r\n}\r\n"
        a = compute_content_fingerprint(_tar_gz({"m.rs": orig}))
        b = compute_content_fingerprint(_tar_gz({"m.rs": reformatted}))
        assert _jaccard(a, b) == 1.0

    def test_rename_and_reorder_files_invisible(self) -> None:
        base = _tar_gz({"a.rs": _rust_file(8, "a"), "b.rs": _rust_file(8, "b")})
        # Same contents, renamed files, packed in the other order.
        shuffled = _tar_gz({"z.rs": _rust_file(8, "b"), "y.rs": _rust_file(8, "a")})
        assert (
            _jaccard(
                compute_content_fingerprint(base), compute_content_fingerprint(shuffled)
            )
            == 1.0
        )

    def test_sprinkled_edit_stays_high(self) -> None:
        # Add a comment line to EVERY file — the evasion that zeroed the old
        # whole-file-hash approach. Shingling keeps similarity high because only
        # the few shingles spanning each new line change.
        files = {f"m{i}.rs": _rust_file(20, f"m{i}") for i in range(4)}
        edited = {name: data + b"\n// tweaked note\n" for name, data in files.items()}
        j = _jaccard(
            compute_content_fingerprint(_tar_gz(files)),
            compute_content_fingerprint(_tar_gz(edited)),
        )
        assert j > 0.85, j

    def test_padding_caught_by_containment(self) -> None:
        # A verbatim copy that pads with junk files dilutes Jaccard but stays fully
        # contained in the (larger) copy => containment ~1.0.
        original = {f"m{i}.rs": _rust_file(15, f"m{i}") for i in range(3)}
        copy = dict(original)
        copy.update(
            {f"pad{i}.txt": (f"lorem ipsum {i}\n" * 40).encode() for i in range(20)}
        )
        a = compute_content_fingerprint(_tar_gz(original))
        b = compute_content_fingerprint(_tar_gz(copy))
        jac, con = content_similarity(a, b)
        assert jac < 0.9, jac  # padding did dilute Jaccard
        assert con > 0.95, con  # but containment still flags it

    def test_distinct_harnesses_low(self) -> None:
        a = compute_content_fingerprint(_tar_gz({"a.rs": _rust_file(20, "alpha")}))
        b = compute_content_fingerprint(_tar_gz({"b.rs": _rust_file(20, "beta")}))
        jac, con = content_similarity(a, b)
        assert jac < 0.3 and con < 0.3

    def test_only_regular_files_counted(self) -> None:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            d = tarfile.TarInfo(name="pkg")
            d.type = tarfile.DIRTYPE
            tar.addfile(d)
            data = _rust_file(5)
            info = tarfile.TarInfo(name="pkg/x.rs")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        assert compute_content_fingerprint(buf.getvalue()) is not None

    def test_member_flood_is_bounded(self) -> None:
        # Hundreds of thousands of directory headers must not be walked to the end
        # (the CPU-DoS the per-file cap missed). Returns None fast once the member
        # cap trips — we assert on the result, the timeout guards the runtime.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for i in range(100000):
                di = tarfile.TarInfo(name=f"d{i}/")
                di.type = tarfile.DIRTYPE
                tar.addfile(di)
        assert compute_content_fingerprint(buf.getvalue()) is None

    def test_unreadable_or_empty_returns_none(self) -> None:
        assert compute_content_fingerprint(b"not a tarball") is None
        assert compute_content_fingerprint(gzip.compress(b"plain gzip, no tar")) is None
        assert compute_content_fingerprint(_tar_gz({})) is None
        assert compute_content_fingerprint(_tar_gz({"blank": b"\n  \n\t\n"})) is None


class TestContentSimilarity:
    def test_missing_or_version_mismatch_scores_zero(self) -> None:
        fp = compute_content_fingerprint(_tar_gz({"a.rs": _rust_file(6)}))
        assert content_similarity(None, fp) == (0.0, 0.0)
        assert content_similarity(fp, None) == (0.0, 0.0)
        assert content_similarity({"v": 999, "k": 1, "card": 1, "m": ["x"]}, fp) == (
            0.0,
            0.0,
        )

    def test_exact_when_sets_small(self) -> None:
        # Both shingle sets fit inside k, so the bottom-k sketch is the whole set
        # and Jaccard is exact, not estimated. Hand-build two overlapping sets.
        def sk(vals: set[str]) -> dict:
            return {
                "v": _FP_VERSION,
                "k": _MINHASH_K,
                "card": len(vals),
                "m": sorted(vals)[:_MINHASH_K],
            }

        a = {f"{i:016x}" for i in range(10)}
        b = {f"{i:016x}" for i in range(5, 15)}
        # |A∩B|=5, |A∪B|=15 => J=1/3; min card=10 => containment=5/10=0.5.
        jac, con = content_similarity(sk(a), sk(b))
        assert abs(jac - 1 / 3) < 1e-9
        assert abs(con - 0.5) < 1e-9

    def _sk(self, vals: set[str]) -> dict:
        return {
            "v": _FP_VERSION,
            "k": _MINHASH_K,
            "card": len(vals),
            "m": sorted(vals)[:_MINHASH_K],
        }

    def test_containment_unbiased_on_asymmetric_sets(self) -> None:
        # Regression: deriving containment from a noisy Jaccard used to blow up for
        # very asymmetric cardinalities (a lean set sharing scaffolding with a fat
        # one estimated ~0.955 vs true 0.50 -> false holds). The direct estimator
        # must track the true value, well clear of the 0.95 gate tolerance.
        def h(tag: str, i: int) -> str:
            return hashlib.sha256(f"{tag}{i}".encode()).hexdigest()[:16]

        shared = {h("s", i) for i in range(750)}
        lean = shared | {h("a", i) for i in range(750)}  # 1500, 50% shared
        fat = shared | {h("b", i) for i in range(45000)}  # 45750
        _, containment = content_similarity(self._sk(lean), self._sk(fat))
        assert containment < 0.85, containment  # true is 0.50; nowhere near a hold

    def test_containment_still_catches_padding(self) -> None:
        # The asymmetric case the estimator MUST keep flagging: a verbatim copy
        # padded to dilute Jaccard is still fully contained => containment ~1.0.
        def h(tag: str, i: int) -> str:
            return hashlib.sha256(f"{tag}{i}".encode()).hexdigest()[:16]

        incumbent = {h("x", i) for i in range(1500)}
        padded_copy = incumbent | {h("pad", i) for i in range(44000)}
        _, containment = content_similarity(self._sk(incumbent), self._sk(padded_copy))
        assert containment > 0.95, containment

    def test_minhash_estimator_accurate_on_large_sets(self) -> None:
        # Sets far larger than k so the KMV approximation actually engages; the
        # estimate must track the true Jaccard within sampling tolerance.
        def h(tag: str, i: int) -> str:
            return hashlib.sha256(f"{tag}{i}".encode()).hexdigest()[:16]

        common = {h("c", i) for i in range(6000)}
        a_vals = common | {h("a", i) for i in range(2000)}
        b_vals = common | {h("b", i) for i in range(2000)}
        true_j = len(a_vals & b_vals) / len(a_vals | b_vals)  # 6000/10000 = 0.6

        def sk(vals: set[str]) -> dict:
            return {
                "v": _FP_VERSION,
                "k": _MINHASH_K,
                "card": len(vals),
                "m": sorted(vals)[:_MINHASH_K],
            }

        jac, con = content_similarity(sk(a_vals), sk(b_vals))
        assert abs(jac - true_j) < 0.1, (jac, true_j)
        # true containment = 6000/8000 = 0.75.
        assert abs(con - 0.75) < 0.12, con
