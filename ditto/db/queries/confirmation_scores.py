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

from ditto.api_models.continual_retest_settings import (
    DEFAULT_WAVE_MEMBERSHIP,
    WaveMembership,
)
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

    This is the ``strict`` half of :func:`fold_eligible_seeds_by_agent`; see
    there for why "every current member" is not the same predicate as "every
    member the wave was actually issued to".
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


def fold_eligible_seeds_by_agent(
    *,
    member_ids: Iterable[UUID],
    seeds_by_agent: Mapping[UUID, Iterable[int]],
    mode: WaveMembership = DEFAULT_WAVE_MEMBERSHIP,
) -> dict[UUID, frozenset[int]]:
    """Per agent, which confirmation seeds may enter the fold's aggregate.

    Three policies, widening in order. All three are pure functions of the same
    two inputs; the mode is an operator setting because the choice changes what
    ``effective_composite`` averages and therefore what validators weight.

    ``strict`` -- the historical behaviour, and now the ROLLBACK PATH rather
        than the default. One global intersection over EVERY current
        emission-set member, handed to every agent.

        The invariant it buys is real: ``effective_composite`` averages the three
        quorum scores together with one score per wave, and it is handed a bare
        tuple of composites with the seed identity already discarded. Comparing
        two such means is only sound when the two agents were measured on the
        SAME seeds, because seed difficulty is the dominant variance term --
        that is the whole reason the lane uses common random numbers.

        Its defect is that "every current member" is evaluated against the live
        emission set rather than against the membership the wave was issued to.
        A newly finalized agent entering the top five with no retests of its own
        empties the intersection, and every other agent's accumulated depth
        silently stops counting. This is not display-only: ``official_composite``
        falls back from the continual mean to the three-score quorum median, so
        the aggregate feeding validator weights reverts too. Every top-five
        membership change currently discards the fold's accumulated evidence.

    ``participants`` -- **the shipped default.** The same intersection, taken
        over members that hold at
        least one confirmation row. An agent that has never reported a single
        seed is not "still running" any wave, so it cannot be protecting a lease;
        including it can only erase evidence, never validate any. Equal sample
        composition is fully preserved among every agent that receives a
        continual mean, because they still share one intersected seed set. A
        member at depth zero simply keeps its quorum median until it starts
        reporting, exactly as every agent outside the emission set already does.

        This is a strictly smaller change than it looks: the board ALREADY mixes
        the two estimators. ``official_composite`` is computed for every
        finalized row and sorted into one list, so an agent with completed waves
        is already ranked directly against an agent on its quorum median at the
        rank-five boundary. ``participants`` does not introduce that mixing; it
        stops a membership change from flipping every agent onto the noisier
        estimator at once.

    ``per_agent`` -- each agent aggregates over its own completed seeds, with no
        intersection at all. This is the most responsive and the least
        comparable: two agents' means are then taken over different seed sets, so
        the difference between them carries a seed-composition term that the
        shared-seed design exists to cancel. Under exchangeable CRN draws it
        stays unbiased, but it adds variance, and variance is what the whole lane
        was built to suppress -- at a ``KOTH_MARGIN`` of 0.007 the noise is the
        same size as the decision. Offered because it is what "retests should
        count for an agent even when others lack waves" literally asks for, and
        gated because it changes what validators weight.

    Note that the paired dethrone test is untouched by all three.
    ``koth._paired_statistic`` computes its OWN pairwise seed intersection
    between challenger and champion, so the crown comparison stays paired
    whatever this returns.
    """
    members = tuple(dict.fromkeys(member_ids))
    own = {agent_id: frozenset(seeds) for agent_id, seeds in seeds_by_agent.items()}
    if mode == "per_agent":
        return own
    if mode == "participants":
        members = tuple(member_id for member_id in members if own.get(member_id))
    shared = completed_confirmation_wave_seeds(
        member_ids=members, seeds_by_agent=seeds_by_agent
    )
    # Intersected with what the agent actually holds, so the result is that
    # agent's fold-eligible seeds rather than a filter to be applied later. For
    # every member this is the shared set unchanged (the intersection could not
    # contain a seed the member is missing); it only narrows for a non-member
    # that happens to carry rows, which the widened retest cohort makes routine.
    return {agent_id: seeds & shared for agent_id, seeds in own.items()}


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
