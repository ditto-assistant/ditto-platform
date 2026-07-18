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
    down_revision: str | tuple[str, ...] | None


def _parents(migration: Migration) -> tuple[str, ...]:
    if migration.down_revision is None:
        return ()
    if isinstance(migration.down_revision, str):
        return (migration.down_revision,)
    return migration.down_revision


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
    if isinstance(down_revision, tuple):
        if not down_revision or not all(
            isinstance(parent, str) and parent for parent in down_revision
        ):
            raise MigrationError(
                f"{path}: down_revision must contain non-empty revision strings"
            )
        if len(set(down_revision)) != len(down_revision):
            raise MigrationError(f"{path}: down_revision contains duplicates")
    elif down_revision is not None and not isinstance(down_revision, str):
        raise MigrationError(
            f"{path}: down_revision must be a revision string or tuple of strings"
        )
    return Migration(path=path, revision=revision, down_revision=down_revision)


def _history_heads(migrations: list[Migration], label: str) -> set[str]:
    """Return every head after validating a connected, acyclic migration DAG."""
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
        parents = _parents(migration)
        if not parents:
            roots.append(migration.revision)
            continue
        for parent in parents:
            if parent not in by_revision:
                raise MigrationError(
                    f"{migration.path}: unknown down_revision {parent}"
                )
            children[parent].append(migration.revision)

    if len(roots) != 1:
        raise MigrationError(f"{label}: expected one root revision, found {len(roots)}")

    visited: set[str] = set()
    ready = list(roots)
    remaining_parents = {
        revision: len(_parents(migration))
        for revision, migration in by_revision.items()
    }
    while ready:
        current = ready.pop()
        visited.add(current)
        for child in children.get(current, []):
            remaining_parents[child] -= 1
            if remaining_parents[child] == 0:
                ready.append(child)

    if len(visited) != len(migrations):
        raise MigrationError(f"{label}: migration history has a cycle")
    return set(by_revision) - set(children)


def validate_linear_history(migrations: list[Migration], label: str) -> str:
    """Return the sole head after validating a resolved migration DAG."""
    heads = _history_heads(migrations, label)
    if len(heads) != 1:
        raise MigrationError(
            f"{label}: expected one head revision, found {len(heads)}: "
            + ", ".join(sorted(heads))
        )
    return next(iter(heads))


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
    base_heads = _history_heads(base_migrations, base_ref)
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

    if new_paths and head_revision in base_heads:
        raise MigrationError("new migrations do not extend the base migration head")

    return len(new_paths), ", ".join(sorted(base_heads)), head_revision


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
