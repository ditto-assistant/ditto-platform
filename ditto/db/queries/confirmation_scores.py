"""Reads + append-only writes for the top-5 shared-seed confirmation ledger.

The continual top-5 rescore lane (``docs/top5-rescore-lane.md``) accumulates one
immutable :class:`~ditto.db.models.ConfirmationScore` row per
``(agent_id, validator_hotkey, bench_version, seed)``. Writes are
INSERT-idempotent (``ON CONFLICT DO NOTHING``) and never UPDATE/delete, so the
record grows monotonically over a champion's reign and stays fully auditable.

The KOTH fold reads paired evidence from this history: per agent, the per-seed
composite is the **median across validators** (N-agnostic, like the k=3
composite), and the fold pairs a challenger against the champion on their shared
seeds. This module returns those per-seed aggregates plus the shared-seed depth
surfaced on the leaderboard.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from ditto.db.models import ConfirmationScore

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class ConfirmationSeedScore:
    """One validator's immutable composite for one champion-anchored seed."""

    agent_id: UUID
    validator_hotkey: str
    seed: int
    composite: float
    run_id: str
    signature: str | None


@dataclass(frozen=True)
class ConfirmationHistoryRow:
    """One append-only confirmation record as exposed on the ledger read.

    Raw per-``(validator_hotkey, seed)`` rows (NOT pre-aggregated): the KOTH fold
    groups them by seed and medians the composite across validators itself.
    """

    agent_id: UUID
    seed: int
    composite: float
    validator_hotkey: str
    bench_version: int
    signature: str | None


def completed_confirmation_wave_seeds(
    *,
    member_ids: Iterable[UUID],
    seeds_by_agent: Mapping[UUID, Iterable[int]],
) -> frozenset[int]:
    """Seeds with accepted confirmation evidence for every cohort member.

    Continual retests are cohort waves, not independent per-agent samples.  A
    partially completed wave stays append-only and visible for audit, but must
    not enter the KOTH fold until every current top-five member has one result
    for the same seed.  Otherwise the first report can change the champion and
    invalidate the still-running leases for the rest of the wave.
    """
    members = tuple(dict.fromkeys(member_ids))
    if not members:
        return frozenset()
    common: set[int] | None = None
    for member_id in members:
        member_seeds = set(seeds_by_agent.get(member_id, ()))
        common = member_seeds if common is None else common & member_seeds
        if not common:
            return frozenset()
    return frozenset(common or ())


async def append_confirmation_scores(
    session: AsyncSession,
    *,
    rows: Sequence[ConfirmationSeedScore],
    bench_version: int,
    created_at: datetime,
) -> int:
    """Append confirmation rows idempotently; return the count actually inserted.

    ``ON CONFLICT DO NOTHING`` on the ``(agent_id, bench_version,
    validator_hotkey, seed)`` key: a re-submitted seed (the validator resends the
    whole champion-anchored union each round) is a no-op, and the first-written
    composite for a ``(validator, seed)`` is immutable. Because dittobench is
    deterministic per ``(agent, seed)`` a re-score would produce the identical
    composite, so idempotency is consensus-safe.
    """
    if not rows:
        return 0
    dialect = session.get_bind().dialect.name
    values = [
        {
            "agent_id": row.agent_id,
            "validator_hotkey": row.validator_hotkey,
            "bench_version": bench_version,
            "seed": row.seed,
            "composite": row.composite,
            "run_id": row.run_id,
            "signature": row.signature,
            "created_at": created_at,
        }
        for row in rows
    ]
    insert = pg_insert if dialect == "postgresql" else sqlite_insert
    statement: Any = (
        insert(ConfirmationScore)
        .values(values)
        .on_conflict_do_nothing(
            index_elements=["agent_id", "bench_version", "validator_hotkey", "seed"]
        )
    )
    result = await session.execute(statement)
    return int(getattr(result, "rowcount", 0) or 0)


async def confirmation_composites_by_seed(
    session: AsyncSession,
    *,
    agent_ids: Iterable[UUID],
    bench_version: int,
) -> dict[UUID, dict[int, float]]:
    """Per agent, ``{seed: median composite across validators}`` for one version.

    The median across validators mirrors the k=3 composite selection (no single
    validator decides a seed), and pairs the fold uses shared seeds against the
    champion. Absent agents / versions map to an empty dict.
    """
    ids = list(dict.fromkeys(agent_ids))
    if not ids:
        return {}
    rows = await session.execute(
        select(
            ConfirmationScore.agent_id,
            ConfirmationScore.seed,
            ConfirmationScore.composite,
        ).where(
            ConfirmationScore.agent_id.in_(ids),
            ConfirmationScore.bench_version == bench_version,
        )
    )
    grouped: dict[UUID, dict[int, list[float]]] = {}
    for agent_id, seed, composite in rows:
        grouped.setdefault(agent_id, {}).setdefault(seed, []).append(composite)
    return {
        agent_id: {
            seed: statistics.median(composites) for seed, composites in seeds.items()
        }
        for agent_id, seeds in grouped.items()
    }


async def confirmation_history_by_agent(
    session: AsyncSession,
    *,
    agent_ids: Iterable[UUID],
    bench_version: int,
) -> dict[UUID, list[ConfirmationHistoryRow]]:
    """Per agent, the raw append-only confirmation rows for one version.

    Ordered ``(seed, validator_hotkey)`` for a deterministic wire order. Raw
    per-``(validator, seed)`` records so the fold does its own group-by-seed
    median; the platform does not pre-aggregate the exposed history.
    """
    ids = list(dict.fromkeys(agent_ids))
    if not ids:
        return {}
    rows = await session.execute(
        select(
            ConfirmationScore.agent_id,
            ConfirmationScore.seed,
            ConfirmationScore.composite,
            ConfirmationScore.validator_hotkey,
            ConfirmationScore.bench_version,
            ConfirmationScore.signature,
        )
        .where(
            ConfirmationScore.agent_id.in_(ids),
            ConfirmationScore.bench_version == bench_version,
        )
        .order_by(ConfirmationScore.seed, ConfirmationScore.validator_hotkey)
    )
    history: dict[UUID, list[ConfirmationHistoryRow]] = {}
    for agent_id, seed, composite, validator_hotkey, version, signature in rows:
        history.setdefault(agent_id, []).append(
            ConfirmationHistoryRow(
                agent_id=agent_id,
                seed=seed,
                composite=composite,
                validator_hotkey=validator_hotkey,
                bench_version=version,
                signature=signature,
            )
        )
    return history


async def confirmation_depths(
    session: AsyncSession,
    *,
    agent_ids: Iterable[UUID],
    bench_version: int,
) -> dict[UUID, int]:
    """Per agent, the shared-seed confirmation depth = number of distinct seeds.

    This is the "N shared-seed confirmations" count surfaced on the leaderboard;
    it grows while an agent holds its emission-set spot.
    """
    ids = list(dict.fromkeys(agent_ids))
    if not ids:
        return {}
    rows = await session.execute(
        select(
            ConfirmationScore.agent_id,
            func.count(func.distinct(ConfirmationScore.seed)),
        )
        .where(
            ConfirmationScore.agent_id.in_(ids),
            ConfirmationScore.bench_version == bench_version,
        )
        .group_by(ConfirmationScore.agent_id)
    )
    return {agent_id: int(depth) for agent_id, depth in rows}
