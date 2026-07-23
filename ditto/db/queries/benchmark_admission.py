"""One authoritative admission boundary for an activated benchmark era."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ColumnElement, or_, select
from sqlalchemy.orm.util import AliasedClass

from ditto.db.models import (
    Agent,
    BenchmarkRollout,
    BenchmarkRolloutMember,
    ScoreAuditEntry,
    ValidatorTicket,
)
from ditto.db.queries.audit import benchmark_contract_refresh_event

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def activated_rollout_for_version(
    session: AsyncSession, *, bench_version: int
) -> BenchmarkRollout | None:
    """Return the newest activated rollout that established ``bench_version``."""

    return await session.scalar(
        select(BenchmarkRollout)
        .where(
            BenchmarkRollout.desired_version == bench_version,
            BenchmarkRollout.status == "activated",
        )
        .order_by(BenchmarkRollout.activated_at.desc())
        .limit(1)
    )


def benchmark_admission_predicate(
    *,
    rollout: BenchmarkRollout,
    bench_version: int,
    agent: type[Agent] | AliasedClass[Agent] = Agent,
) -> ColumnElement[bool]:
    """SQL predicate for agents allowed to consume this benchmark's capacity.

    A generated dataset is deliberately not admission evidence: routine policy
    rescreens can regenerate one for historical submissions. Only submissions
    received in the new era, frozen rollout members, and submissions carrying a
    version-scoped audited benchmark-contract refresh may enter the active queue.
    An ordinary retry grant is not a benchmark-era admission credential.
    """

    rollout_member = (
        select(BenchmarkRolloutMember.agent_id)
        .where(
            BenchmarkRolloutMember.rollout_id == rollout.rollout_id,
            BenchmarkRolloutMember.agent_id == agent.agent_id,
        )
        .correlate(agent)
        .exists()
    )
    contract_refresh = (
        select(ScoreAuditEntry.agent_id)
        .where(
            ScoreAuditEntry.agent_id == agent.agent_id,
            ScoreAuditEntry.event == benchmark_contract_refresh_event(bench_version),
        )
        .correlate(agent)
        .exists()
    )
    refresh_retry_grant = (
        select(ValidatorTicket.agent_id)
        .where(
            ValidatorTicket.agent_id == agent.agent_id,
            ValidatorTicket.bench_version == bench_version,
            ValidatorTicket.manual_retry_grants > 0,
        )
        .correlate(agent)
        .exists()
    )
    return or_(
        agent.created_at >= rollout.created_at,
        rollout_member,
        contract_refresh & refresh_retry_grant,
    )


async def admitted_agent_ids(
    session: AsyncSession,
    *,
    bench_version: int,
    agent_ids: Sequence[UUID],
) -> set[UUID]:
    """Return which requested agents are admitted to the active-era queue."""

    requested = set(agent_ids)
    if not requested:
        return set()
    rollout = await activated_rollout_for_version(session, bench_version=bench_version)
    if rollout is None:
        return requested
    return set(
        await session.scalars(
            select(Agent.agent_id).where(
                Agent.agent_id.in_(requested),
                benchmark_admission_predicate(
                    rollout=rollout, bench_version=bench_version
                ),
            )
        )
    )


async def agent_is_admitted(
    session: AsyncSession, *, bench_version: int, agent_id: UUID
) -> bool:
    """Whether one exact agent may consume the benchmark's validator capacity."""

    return agent_id in await admitted_agent_ids(
        session, bench_version=bench_version, agent_ids=[agent_id]
    )
