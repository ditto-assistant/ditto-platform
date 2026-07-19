"""Application service for convergent rolling benchmark qualification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_server.datapipeline import DatasetGenerator
from ditto.db.models import (
    Agent,
    BenchmarkDataset,
    BenchmarkRollout,
    BenchmarkRolloutMember,
    Score,
    ValidatorHeartbeat,
)
from ditto.db.queries.benchmark_rollout import (
    CANARY_BENCH_VERSION,
    COHORT_SIZE,
    DatasetPin,
    RolloutSnapshotMember,
    append_rollout_member,
    create_rollout_snapshot,
    heartbeat_supports_v3,
    maybe_activate_rollout,
    open_rollout,
    rolling_top_five,
    rollout_for_transition,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PendingQualification:
    member: RolloutSnapshotMember
    seed: int
    run_size: str
    seed_block: int | None
    seed_block_hash: str | None
    seed_source: Literal["legacy_pin", "v2_scores", "versioned_pin"]
    existing_sha256: str | None = None


async def qualification_candidate(
    session: AsyncSession,
    *,
    source_bench_version: int,
    target_bench_version: int,
    member: RolloutSnapshotMember,
    generator_run_size: str | None,
) -> tuple[PendingQualification | None, str | None]:
    """Resolve a v3 input without rewriting an older agent-level dataset pin.

    Pre-pin agents may have null compatibility columns even though every
    accepted v2 score binds the same signed seed. One consistent score seed is
    safe to reuse for the separately versioned v3 dataset; ambiguity fails
    closed and is surfaced to operators.
    """
    agent = await session.get(Agent, member.agent_id)
    assert agent is not None
    versioned = await session.get(
        BenchmarkDataset, (member.agent_id, target_bench_version)
    )
    if versioned is not None:
        return (
            PendingQualification(
                member=member,
                seed=versioned.seed,
                run_size=versioned.run_size,
                seed_block=versioned.seed_block,
                seed_block_hash=versioned.seed_block_hash,
                seed_source="versioned_pin",
                existing_sha256=versioned.sha256,
            ),
            None,
        )
    if (
        agent.dataset_seed is not None
        and agent.dataset_sha256 is not None
        and agent.dataset_run_size is not None
    ):
        return (
            PendingQualification(
                member=member,
                seed=agent.dataset_seed,
                run_size=agent.dataset_run_size,
                seed_block=agent.dataset_seed_block,
                seed_block_hash=agent.dataset_seed_block_hash,
                seed_source="legacy_pin",
            ),
            None,
        )

    score_seeds = set(
        await session.scalars(
            select(Score.seed)
            .where(
                Score.agent_id == member.agent_id,
                Score.bench_version == source_bench_version,
            )
            .distinct()
        )
    )
    if len(score_seeds) > 1:
        return None, "ambiguous_source_score_seeds"
    if score_seeds:
        seed = score_seeds.pop()
        seed_source: Literal["legacy_pin", "v2_scores"] = "v2_scores"
        seed_block = None
        seed_block_hash = None
    elif agent.dataset_seed is not None:
        seed = agent.dataset_seed
        seed_source = "legacy_pin"
        seed_block = agent.dataset_seed_block
        seed_block_hash = agent.dataset_seed_block_hash
    else:
        return None, "missing_dataset_seed"
    run_size = agent.dataset_run_size or generator_run_size
    if run_size is None:
        return None, "dataset_generator_disabled"
    return (
        PendingQualification(
            member=member,
            seed=seed,
            run_size=run_size,
            seed_block=seed_block,
            seed_block_hash=seed_block_hash,
            seed_source=seed_source,
        ),
        None,
    )


async def rolling_qualification_blockers(
    session: AsyncSession, *, generator_run_size: str | None
) -> list[dict[str, str]]:
    """Describe hybrid-top-five agents that automatic qualification cannot add."""
    rollout = await open_rollout(session)
    if rollout is None:
        return []
    blockers: list[dict[str, str]] = []
    for member in await rolling_top_five(session):
        if (
            await session.get(
                BenchmarkRolloutMember, (rollout.rollout_id, member.agent_id)
            )
            is not None
        ):
            continue
        candidate, reason = await qualification_candidate(
            session,
            source_bench_version=rollout.from_version,
            target_bench_version=rollout.desired_version,
            member=member,
            generator_run_size=generator_run_size,
        )
        if candidate is None:
            assert reason is not None
            blockers.append({"agent_id": str(member.agent_id), "reason": reason})
    return blockers


async def ensure_rolling_qualification(
    session: AsyncSession, *, generator: DatasetGenerator, now: datetime
) -> bool:
    """Lazily seed v3 qualification once one verified validator requests work.

    Benchmark v3 is a shipped contract, not an operator-started feature flag.
    Rendering happens outside a transaction and snapshot creation remains
    idempotent under the query layer's advisory lock.
    """
    async with session.begin():
        existing = await rollout_for_transition(
            session, from_version=2, desired_version=CANARY_BENCH_VERSION
        )
        if existing is not None:
            return False
        heartbeats = list(await session.scalars(select(ValidatorHeartbeat)))
        if not any(heartbeat_supports_v3(item, now=now) for item in heartbeats):
            return False
        members = await rolling_top_five(session)
        if len(members) != COHORT_SIZE:
            return False
        pending: list[PendingQualification] = []
        for member in members:
            candidate, reason = await qualification_candidate(
                session,
                source_bench_version=2,
                target_bench_version=CANARY_BENCH_VERSION,
                member=member,
                generator_run_size=generator.run_size,
            )
            if candidate is None:
                logger.warning(
                    "benchmark qualification bootstrap blocked agent_id=%s reason=%s",
                    member.agent_id,
                    reason,
                )
                return False
            pending.append(candidate)

    datasets: dict[UUID, DatasetPin] = {}
    for candidate in pending:
        datasets[candidate.member.agent_id] = DatasetPin(
            seed=candidate.seed,
            sha256=await generator.generate(
                candidate.seed, bench_version=CANARY_BENCH_VERSION
            )
            if candidate.existing_sha256 is None
            else candidate.existing_sha256,
            run_size=candidate.run_size,
            seed_block=candidate.seed_block,
            seed_block_hash=candidate.seed_block_hash,
        )
    async with session.begin():
        current_members = await rolling_top_five(session)
        if current_members != members:
            return False
        for expected, current_member in zip(pending, current_members, strict=True):
            current, _reason = await qualification_candidate(
                session,
                source_bench_version=2,
                target_bench_version=CANARY_BENCH_VERSION,
                member=current_member,
                generator_run_size=generator.run_size,
            )
            if current != expected:
                return False
        await create_rollout_snapshot(
            session, members=members, datasets=datasets, now=now
        )
    return True


async def refresh_rolling_qualification(
    session: AsyncSession, *, generator: DatasetGenerator, now: datetime
) -> int:
    """Append every newly risen hybrid-top-five agent and try activation.

    Dataset rendering deliberately happens between transactions: the generator
    is a network service and must never run while holding rollout/agent locks.
    The second transaction rechecks membership, so concurrent refreshes are
    idempotent.
    """
    pending: list[PendingQualification] = []
    async with session.begin():
        rollout = await open_rollout(session)
        if rollout is None:
            return 0
        top_five = await rolling_top_five(session)
        for member in top_five:
            existing = await session.get(
                BenchmarkRolloutMember, (rollout.rollout_id, member.agent_id)
            )
            if existing is not None:
                continue
            candidate, reason = await qualification_candidate(
                session,
                source_bench_version=rollout.from_version,
                target_bench_version=rollout.desired_version,
                member=member,
                generator_run_size=generator.run_size,
            )
            if candidate is None:
                logger.warning(
                    "benchmark qualification blocked agent_id=%s reason=%s",
                    member.agent_id,
                    reason,
                )
                continue
            pending.append(candidate)
        rollout_id: UUID = rollout.rollout_id
        desired_version = rollout.desired_version

    rendered: list[tuple[PendingQualification, str]] = []
    for candidate in pending:
        rendered.append(
            (
                candidate,
                candidate.existing_sha256
                or await generator.generate(
                    candidate.seed, bench_version=desired_version
                ),
            )
        )

    appended = 0
    async with session.begin():
        rollout = await session.get(BenchmarkRollout, rollout_id, with_for_update=True)
        if rollout is None or rollout.status not in (
            "collecting",
            "blocked_ineligible",
        ):
            return 0
        current_top = {
            member.agent_id: member for member in await rolling_top_five(session)
        }
        for candidate, sha256 in rendered:
            current_member = current_top.get(candidate.member.agent_id)
            if current_member is None:
                continue
            current_candidate, _reason = await qualification_candidate(
                session,
                source_bench_version=rollout.from_version,
                target_bench_version=rollout.desired_version,
                member=current_member,
                generator_run_size=generator.run_size,
            )
            if current_candidate != candidate:
                logger.warning(
                    "benchmark qualification changed during render agent_id=%s",
                    candidate.member.agent_id,
                )
                continue
            appended += await append_rollout_member(
                session,
                rollout=rollout,
                member=candidate.member,
                dataset=DatasetPin(
                    seed=candidate.seed,
                    sha256=sha256,
                    run_size=candidate.run_size,
                    seed_block=candidate.seed_block,
                    seed_block_hash=candidate.seed_block_hash,
                ),
                now=now,
            )
        await maybe_activate_rollout(session, rollout, now=now)
    return appended
