"""Bounded, read-only inspection of an uploaded submission tarball.

Serves the operator quarantine-review console: the screener's source-review
finding flags ``path:line`` evidence, and these helpers let an authenticated
operator read exactly those bounded excerpts server-side without downloading
and unpacking the artifact locally.

The tarball is UNTRUSTED miner input, so the reader mirrors the screener's
defensive posture (:mod:`ditto_screener.source_review`): only regular members
with safe relative paths are visible, reads are line- and size-bounded, and
non-UTF-8 or oversized files are reported as opaque blobs instead of being
silently invisible.
"""

from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass

MAX_LISTING_FILES = 512
MAX_OPAQUE_BLOBS = 128
MAX_READ_LINES = 400
TEXT_SIZE_LIMIT = 2 * 1024 * 1024
# Uploads are capped well below this (20 MiB by default); the bound only
# protects the API process from a mis-sized stored object.
MAX_TARBALL_BYTES = 64 * 1024 * 1024


class SourceInspectError(Exception):
    """Raised when the archive or a requested member cannot be inspected."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _Member:
    name: str
    archive_name: str
    size: int


class TarSourceInspector:
    """A read-only, size-bounded view over regular files in a tarball."""

    def __init__(self, tar_bytes: bytes) -> None:
        self._tar_bytes = tar_bytes
        members: list[_Member] = []
        try:
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as archive:
                for member in archive.getmembers():
                    normalized = member.name.removeprefix("./")
                    parts = normalized.split("/")
                    if (
                        not normalized
                        or normalized.startswith("/")
                        or ".." in parts
                        or "\\" in normalized
                        or not member.isfile()
                    ):
                        continue
                    members.append(_Member(normalized, member.name, member.size))
        except (tarfile.TarError, OSError, EOFError) as error:
            raise SourceInspectError(
                "artifact-unreadable", f"artifact is not a readable tarball: {error}"
            ) from error
        self._members = {member.name: member for member in members}

    def listing(self) -> dict[str, object]:
        """Bounded inventory: every path with its size, opaque blobs called out."""
        ordered = sorted(self._members.values(), key=lambda item: item.name)
        rows = [
            {"path": item.name, "bytes": item.size}
            for item in ordered[:MAX_LISTING_FILES]
        ]
        return {
            "file_count": len(self._members),
            "files": rows,
            "opaque_blobs": self._opaque_blobs(),
            "truncated": len(ordered) > len(rows),
        }

    def read(self, path: str, start_line: int, end_line: int) -> dict[str, object]:
        """Read a bounded line range from one UTF-8 text member."""
        normalized = path.removeprefix("./")
        member = self._members.get(normalized)
        if member is None:
            raise SourceInspectError("file-not-found", f"no file at {normalized!r}")
        text = self._read_text(normalized)
        if text is None:
            raise SourceInspectError(
                "file-is-not-utf8-text",
                f"{normalized!r} is binary or exceeds the {TEXT_SIZE_LIMIT} byte "
                "text bound",
            )
        lines = text.splitlines()
        start = max(1, start_line)
        end = max(start, min(end_line, start + MAX_READ_LINES - 1, len(lines)))
        return {
            "path": normalized,
            "total_lines": len(lines),
            "start_line": start,
            "end_line": min(end, len(lines)),
            "lines": [
                {"line": index, "text": lines[index - 1][:500]}
                for index in range(start, min(end, len(lines)) + 1)
            ],
        }

    def _opaque_blobs(self) -> list[dict[str, object]]:
        blobs: list[dict[str, object]] = []
        for name in sorted(self._members):
            info = self._members[name]
            if info.size > TEXT_SIZE_LIMIT:
                reason = "oversized"
            elif self._read_text(name) is not None:
                continue
            else:
                reason = "non_utf8"
            blobs.append({"path": name, "bytes": info.size, "reason": reason})
            if len(blobs) >= MAX_OPAQUE_BLOBS:
                break
        return blobs

    def _read_text(self, path: str) -> str | None:
        member_info = self._members[path]
        if member_info.size > TEXT_SIZE_LIMIT:
            return None
        with tarfile.open(fileobj=io.BytesIO(self._tar_bytes), mode="r:gz") as archive:
            member = archive.getmember(member_info.archive_name)
            extracted = archive.extractfile(member)
            if extracted is None:
                return None
            raw = extracted.read(TEXT_SIZE_LIMIT + 1)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return None


__all__ = [
    "MAX_LISTING_FILES",
    "MAX_READ_LINES",
    "MAX_TARBALL_BYTES",
    "SourceInspectError",
    "TarSourceInspector",
]
