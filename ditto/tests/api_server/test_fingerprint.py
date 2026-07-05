"""Unit tests for the content fingerprint :mod:`ditto.api_server.fingerprint`."""

from __future__ import annotations

import gzip
import io
import tarfile

from ditto.api_server.fingerprint import (
    compute_content_fingerprint,
    content_jaccard,
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


class TestComputeContentFingerprint:
    def test_stable_and_sorted(self) -> None:
        fp = compute_content_fingerprint(
            _tar_gz({"a.py": b"print(1)\n", "b.py": b"print(2)\n"})
        )
        assert fp is not None
        assert fp == sorted(fp)
        # Deterministic: same input, same fingerprint.
        assert fp == compute_content_fingerprint(
            _tar_gz({"a.py": b"print(1)\n", "b.py": b"print(2)\n"})
        )

    def test_reindent_is_invisible(self) -> None:
        # Same logic, re-indented + trailing whitespace: identical fingerprint.
        original = _tar_gz({"main.py": b"def f():\n    return 1\n"})
        reindented = _tar_gz({"main.py": b"def f():\n        return 1  \n\n"})
        assert compute_content_fingerprint(original) == compute_content_fingerprint(
            reindented
        )

    def test_rename_and_reorder_are_invisible(self) -> None:
        # Filenames are not hashed and the fingerprint is a set: renaming the
        # files and packing them in a different order yields the same result.
        base = _tar_gz({"a.py": b"AAA\n", "b.py": b"BBB\n"})
        renamed = _tar_gz({"z_renamed.py": b"BBB\n", "y_renamed.py": b"AAA\n"})
        assert compute_content_fingerprint(base) == compute_content_fingerprint(renamed)

    def test_different_content_differs(self) -> None:
        one = compute_content_fingerprint(_tar_gz({"a.py": b"alpha\n"}))
        two = compute_content_fingerprint(_tar_gz({"a.py": b"beta\n"}))
        assert one != two

    def test_only_regular_files_counted(self) -> None:
        # A directory member carries no content and must not become a hash.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            dir_info = tarfile.TarInfo(name="pkg")
            dir_info.type = tarfile.DIRTYPE
            tar.addfile(dir_info)  # directory entry, not a regular file
            data = b"content\n"
            info = tarfile.TarInfo(name="pkg/x.py")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        fp = compute_content_fingerprint(buf.getvalue())
        assert fp is not None
        assert len(fp) == 1

    def test_unreadable_tarball_returns_none(self) -> None:
        assert compute_content_fingerprint(b"not a tarball") is None
        assert compute_content_fingerprint(gzip.compress(b"plain gzip, no tar")) is None

    def test_empty_tarball_returns_none(self) -> None:
        assert compute_content_fingerprint(_tar_gz({})) is None
        # A tarball whose only file is blank-after-normalization has no hashes.
        assert compute_content_fingerprint(_tar_gz({"blank.txt": b"\n  \n\n"})) is None


class TestContentJaccard:
    def test_identical_sets(self) -> None:
        fp = ["a", "b", "c"]
        assert content_jaccard(fp, fp) == 1.0

    def test_disjoint_sets(self) -> None:
        assert content_jaccard(["a", "b"], ["c", "d"]) == 0.0

    def test_partial_overlap(self) -> None:
        # {a,b,c} vs {b,c,d}: intersection 2, union 4 => 0.5.
        assert content_jaccard(["a", "b", "c"], ["b", "c", "d"]) == 0.5

    def test_one_edited_file_of_ten_is_high(self) -> None:
        # A copy that changes exactly one of ten files: intersection 9, union 11.
        original = [f"h{i}" for i in range(10)]
        copy = [f"h{i}" for i in range(9)] + ["edited"]
        assert content_jaccard(original, copy) == 9 / 11  # ~0.818

    def test_missing_fingerprints_score_zero(self) -> None:
        assert content_jaccard(None, ["a"]) == 0.0
        assert content_jaccard(["a"], None) == 0.0
        assert content_jaccard([], ["a"]) == 0.0
