#!/usr/bin/env python3
"""Validate that Alembic migrations remain a safe, linear history."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

MIGRATIONS_DIR = Path("alembic/versions")
MIGRATION_NAME = re.compile(r"^(?P<date>\d{4}_\d{2}_\d{2})_.+\.py$")


@dataclass(frozen=True)
class Migration:
    path: str
    revision: str
    down_revision: str | None


class MigrationError(ValueError):
    """Raised when a migration history violates repository policy."""


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _assignment(tree: ast.Module, name: str, path: str) -> object:
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise MigrationError(f"{path}: missing {name}")


def parse_migration(path: str, source: str) -> Migration:
    """Parse the revision relationship without importing migration code."""
    try:
        tree = ast.parse(source, filename=path)
        revision = _assignment(tree, "revision", path)
        down_revision = _assignment(tree, "down_revision", path)
    except (SyntaxError, ValueError) as exc:
        raise MigrationError(f"{path}: cannot parse migration metadata: {exc}") from exc

    if not isinstance(revision, str) or not revision:
        raise MigrationError(f"{path}: revision must be a non-empty string")
    if down_revision is not None and not isinstance(down_revision, str):
        raise MigrationError(
            f"{path}: down_revision must be one revision string; "
            "merge revisions are not allowed"
        )
    return Migration(path=path, revision=revision, down_revision=down_revision)


def validate_linear_history(migrations: list[Migration], label: str) -> str:
    """Return the sole head after validating a connected, non-branching chain."""
    by_revision: dict[str, Migration] = {}
    children: defaultdict[str, list[str]] = defaultdict(list)
    roots: list[str] = []

    for migration in migrations:
        previous = by_revision.get(migration.revision)
        if previous is not None:
            raise MigrationError(
                f"{label}: duplicate revision {migration.revision}: "
                f"{previous.path}, {migration.path}"
            )
        by_revision[migration.revision] = migration

    for migration in migrations:
        if migration.down_revision is None:
            roots.append(migration.revision)
            continue
        if migration.down_revision not in by_revision:
            raise MigrationError(
                f"{migration.path}: unknown down_revision {migration.down_revision}"
            )
        children[migration.down_revision].append(migration.revision)

    if len(roots) != 1:
        raise MigrationError(f"{label}: expected one root revision, found {len(roots)}")

    forks = {
        parent: child_ids
        for parent, child_ids in children.items()
        if len(child_ids) > 1
    }
    if forks:
        details = ", ".join(
            f"{parent} -> {', '.join(ids)}" for parent, ids in forks.items()
        )
        raise MigrationError(f"{label}: migration history branches: {details}")

    visited: set[str] = set()
    current = roots[0]
    while True:
        if current in visited:
            raise MigrationError(f"{label}: cycle detected at revision {current}")
        visited.add(current)
        next_revisions = children.get(current, [])
        if not next_revisions:
            head = current
            break
        current = next_revisions[0]

    if len(visited) != len(migrations):
        raise MigrationError(f"{label}: migration history is disconnected")
    return head


def _paths_at(ref: str) -> list[str]:
    output = _git("ls-tree", "-r", "--name-only", ref, "--", str(MIGRATIONS_DIR))
    return sorted(path for path in output.splitlines() if path.endswith(".py"))


def _migrations_at(ref: str) -> list[Migration]:
    return [
        parse_migration(path, _git("show", f"{ref}:{path}")) for path in _paths_at(ref)
    ]


def _head_migrations() -> list[Migration]:
    paths = sorted(MIGRATIONS_DIR.glob("*.py"))
    return [parse_migration(str(path), path.read_text()) for path in paths]


def check(base_ref: str) -> tuple[int, str, str]:
    """Validate HEAD against the immutable migration history on *base_ref*."""
    base_paths = set(_paths_at(base_ref))
    head_paths = {str(path) for path in MIGRATIONS_DIR.glob("*.py")}

    removed = sorted(base_paths - head_paths)
    if removed:
        raise MigrationError("existing migrations were removed: " + ", ".join(removed))

    changed = _git(
        "diff",
        "--name-only",
        "--diff-filter=M",
        f"{base_ref}...HEAD",
        "--",
        str(MIGRATIONS_DIR),
    ).splitlines()
    if changed:
        raise MigrationError("existing migrations are immutable: " + ", ".join(changed))

    base_migrations = _migrations_at(base_ref)
    head_migrations = _head_migrations()
    base_head = validate_linear_history(base_migrations, base_ref)
    head_revision = validate_linear_history(head_migrations, "HEAD")

    new_paths = sorted(head_paths - base_paths)
    base_dates = [MIGRATION_NAME.match(Path(path).name) for path in base_paths]
    if any(match is None for match in base_dates):
        raise MigrationError(
            f"{base_ref}: migration filename does not use YYYY_MM_DD_name.py"
        )
    latest_base_date = max(
        match.group("date") for match in base_dates if match is not None
    )

    for path in new_paths:
        match = MIGRATION_NAME.match(Path(path).name)
        if match is None:
            raise MigrationError(
                f"{path}: migration filename must use YYYY_MM_DD_name.py"
            )
        if match.group("date") < latest_base_date:
            raise MigrationError(
                f"{path}: date precedes {base_ref}'s latest migration date "
                f"{latest_base_date}"
            )

    if new_paths and head_revision == base_head:
        raise MigrationError("new migrations do not extend the base migration head")

    return len(new_paths), base_head, head_revision


def main() -> int:
    base_ref = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    try:
        count, base_head, head = check(base_ref)
    except (MigrationError, subprocess.CalledProcessError) as exc:
        print(f"migration-order: {exc}", file=sys.stderr)
        return 1
    print(
        f"migration-order: ok ({count} new migration(s); "
        f"{base_ref} head {base_head}; HEAD head {head})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
