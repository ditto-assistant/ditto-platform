"""Application service for convergent rolling benchmark qualification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_server.datapipeline import DatasetGenerator
from ditto.db.models import Agent, BenchmarkRollout, BenchmarkRolloutMember
from ditto.db.queries.benchmark_rollout import (
    DatasetPin,
    RolloutSnapshotMember,
    append_rollout_member,
    maybe_activate_rollout,
    open_rollout,
    rolling_top_five,
)


@dataclass(frozen=True)
class _PendingQualification:
    member: RolloutSnapshotMember
    seed: int
    run_size: str
    seed_block: int | None
    seed_block_hash: str | None


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
