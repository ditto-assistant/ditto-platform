from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_migration_order import (
    Migration,
    MigrationError,
    parse_migration,
    resolve_working_tree_head,
    validate_linear_history,
)

ROOT = Path(__file__).parents[3]

MIGRATION = """\
revision: str = "{revision}"
down_revision: str | None = {down!r}
"""


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

    with pytest.raises(MigrationError, match="expected one head revision"):
        validate_linear_history(migrations, "test")


def test_merge_revision_resolves_parallel_heads() -> None:
    source = """
revision: str = "merge"
down_revision: tuple[str, str] = ("one", "two")
"""
    merge = parse_migration("merge.py", source)
    migrations = [
        migration("root", None),
        migration("one", "root"),
        migration("two", "root"),
        merge,
    ]

    assert validate_linear_history(migrations, "test") == "merge"


def test_merge_revision_rejects_duplicate_parents() -> None:
    source = """
revision: str = "merge"
down_revision: tuple[str, str] = ("one", "one")
"""

    with pytest.raises(MigrationError, match="down_revision contains duplicates"):
        parse_migration("merge.py", source)


def test_repository_migration_history_has_one_head() -> None:
    migrations = [
        parse_migration(str(path), path.read_text())
        for path in sorted(Path("alembic/versions").glob("*.py"))
    ]

    assert validate_linear_history(migrations, "repository")


def test_working_tree_head_resolves_for_this_repository() -> None:
    """The deploy preflight has to agree with the CI check on this checkout."""
    assert resolve_working_tree_head()


def _run_head_mode(cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the deploy-side `--head` mode the way update.sh does.

    Deliberately the system interpreter with no venv and no database: the
    preflight must be able to run before `uv sync`.
    """
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_migration_order.py"), "--head"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_history(tmp_path: Path, *, diverged: bool) -> None:
    versions = tmp_path / "alembic" / "versions"
    versions.mkdir(parents=True)
    (versions / "2026_07_01_root.py").write_text(
        MIGRATION.format(revision="root", down=None)
    )
    (versions / "2026_07_02_first.py").write_text(
        MIGRATION.format(revision="e7b4c02a5d18", down="root")
    )
    if diverged:
        (versions / "2026_07_02_second.py").write_text(
            MIGRATION.format(revision="e5b8c31d47af", down="root")
        )


def test_head_mode_prints_the_single_head(tmp_path: Path) -> None:
    _write_history(tmp_path, diverged=False)

    result = _run_head_mode(tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "e7b4c02a5d18"


def test_head_mode_names_every_head_and_the_merge_that_fixes_it(
    tmp_path: Path,
) -> None:
    """The 2026-07-25 shape: two PRs extended the same parent and both merged.

    Each passed its own migration-order check against `main` at the time, and
    the divergence only existed once both had landed -- so the deploy is the
    first place it can be caught. The message has to carry the revisions and
    the remedy, because alembic's own error carries neither.
    """
    _write_history(tmp_path, diverged=True)

    result = _run_head_mode(tmp_path)

    assert result.returncode == 1
    assert "2 head revisions are present" in result.stderr
    assert "e5b8c31d47af" in result.stderr
    assert "e7b4c02a5d18" in result.stderr
    assert (
        'uv run alembic merge -m "merge heads" e5b8c31d47af e7b4c02a5d18'
        in result.stderr
    )
