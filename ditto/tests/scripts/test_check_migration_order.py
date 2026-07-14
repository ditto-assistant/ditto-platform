from __future__ import annotations

import pytest

from scripts.check_migration_order import (
    Migration,
    MigrationError,
    parse_migration,
    validate_linear_history,
)


def migration(revision: str, down_revision: str | None) -> Migration:
    return Migration(f"{revision}.py", revision, down_revision)


def test_linear_history_returns_single_head() -> None:
    migrations = [migration("one", None), migration("two", "one")]

    assert validate_linear_history(migrations, "test") == "two"


def test_duplicate_revision_is_rejected() -> None:
    migrations = [migration("one", None), migration("one", None)]

    with pytest.raises(MigrationError, match="duplicate revision one"):
        validate_linear_history(migrations, "test")


def test_parallel_heads_are_rejected() -> None:
    migrations = [
        migration("one", None),
        migration("two", "one"),
        migration("three", "one"),
    ]

    with pytest.raises(MigrationError, match="migration history branches"):
        validate_linear_history(migrations, "test")


def test_merge_revision_is_rejected() -> None:
    source = """
revision: str = "merge"
down_revision: tuple[str, str] = ("one", "two")
"""

    with pytest.raises(MigrationError, match="merge revisions are not allowed"):
        parse_migration("merge.py", source)
