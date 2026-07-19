"""Application service for convergent rolling benchmark qualification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_server.datapipeline import DatasetGenerator
from ditto.db.models import (
    Agent,
    BenchmarkRollout,
    BenchmarkRolloutMember,
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


@dataclass(frozen=True)
class _PendingQualification:
    member: RolloutSnapshotMember
    seed: int
    run_size: str
    seed_block: int | None
    seed_block_hash: str | None


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
        pending: list[_PendingQualification] = []
        for member in members:
            agent = await session.get(Agent, member.agent_id)
            assert agent is not None
            if (
                agent.dataset_seed is None
                or agent.dataset_sha256 is None
                or agent.dataset_run_size is None
            ):
                return False
            pending.append(
                _PendingQualification(
                    member=member,
                    seed=agent.dataset_seed,
                    run_size=agent.dataset_run_size,
                    seed_block=agent.dataset_seed_block,
                    seed_block_hash=agent.dataset_seed_block_hash,
                )
            )

    datasets: dict[UUID, DatasetPin] = {}
    for candidate in pending:
        datasets[candidate.member.agent_id] = DatasetPin(
            seed=candidate.seed,
            sha256=await generator.generate(
                candidate.seed, bench_version=CANARY_BENCH_VERSION
            ),
            run_size=candidate.run_size,
            seed_block=candidate.seed_block,
            seed_block_hash=candidate.seed_block_hash,
        )
    async with session.begin():
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
    pending: list[_PendingQualification] = []
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
            agent = await session.get(Agent, member.agent_id)
            assert agent is not None
            if (
                agent.dataset_seed is None
                or agent.dataset_run_size is None
                or agent.dataset_sha256 is None
            ):
                continue
            pending.append(
                _PendingQualification(
                    member=member,
                    seed=agent.dataset_seed,
                    run_size=agent.dataset_run_size,
                    seed_block=agent.dataset_seed_block,
                    seed_block_hash=agent.dataset_seed_block_hash,
                )
            )
        rollout_id: UUID = rollout.rollout_id
        desired_version = rollout.desired_version

    rendered: list[tuple[_PendingQualification, str]] = []
    for candidate in pending:
        rendered.append(
            (
                candidate,
                await generator.generate(candidate.seed, bench_version=desired_version),
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
        for candidate, sha256 in rendered:
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
