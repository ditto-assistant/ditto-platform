"""Leased screening attempts and their append-only public history."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import exists, func, or_, select
from sqlalchemy.sql.selectable import ScalarSelect

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.db.models import Agent, Score, ScreeningAttempt

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def screening_score_count() -> ScalarSelect[int]:
    """Return the accepted-score count correlated to the current agent."""
    return (
        select(func.count())
        .where(Score.agent_id == Agent.agent_id)
        .correlate(Agent)
        .scalar_subquery()
    )


async def expire_screening_attempts(session: AsyncSession, *, now: datetime) -> int:
    """Expire overdue leases and return their submissions to the retry pool."""
    attempts = list(
        await session.scalars(
            select(ScreeningAttempt)
            .where(
                ScreeningAttempt.status == "running",
                ScreeningAttempt.deadline < now,
            )
            .with_for_update()
        )
    )
    for attempt in attempts:
        attempt.status = "expired"
        attempt.finished_at = now
        attempt.public_reason = "Screening lease expired"
        agent = await session.get(Agent, attempt.agent_id)
        if agent is not None and agent.status == AgentStatus.SCREENING:
            agent.status = AgentStatus.SCREENING_FAILED
            agent.screening_reason = "Screening lease expired"
    return len(attempts)


async def claim_screening_attempts(
    session: AsyncSession,
    *,
    screener_hotkey: str,
    now: datetime,
    ttl: timedelta,
    limit: int,
) -> list[tuple[Agent, ScreeningAttempt]]:
    """Claim least-scored eligible submissions, then oldest within that bucket."""
    await expire_screening_attempts(session, now=now)
    has_running = exists(
        select(ScreeningAttempt.attempt_id).where(
            ScreeningAttempt.agent_id == Agent.agent_id,
            ScreeningAttempt.status == "running",
        )
    )
    eligible = or_(
        Agent.status == AgentStatus.UPLOADED,
        Agent.status == AgentStatus.SCREENING_FAILED,
        (
            Agent.status.in_(
                (
                    AgentStatus.EVALUATING,
                    AgentStatus.REJECTED,
                )
            )
            & (Agent.screening_policy_version < SCREENING_POLICY_VERSION)
        ),
    )
    agents = list(
        await session.scalars(
            select(Agent)
            .where(eligible, ~has_running)
            .order_by(
                screening_score_count().asc(),
                Agent.created_at.asc(),
                Agent.agent_id.asc(),
            )
            .limit(limit)
            .with_for_update(of=Agent, skip_locked=True)
        )
    )
    claimed: list[tuple[Agent, ScreeningAttempt]] = []
    for agent in agents:
        has_history = await session.scalar(
            select(exists().where(ScreeningAttempt.agent_id == agent.agent_id))
        )
        if not has_history and agent.screening_policy_version > 0:
            legacy_status = {
                AgentStatus.EVALUATING: "passed",
                AgentStatus.REJECTED: "rejected",
                AgentStatus.SCREENING_FAILED: "failed",
            }.get(agent.status)
            if legacy_status is not None:
                session.add(
                    ScreeningAttempt(
                        attempt_id=uuid4(),
                        agent_id=agent.agent_id,
                        screener_hotkey=screener_hotkey,
                        policy_version=agent.screening_policy_version,
                        status=legacy_status,
                        started_at=agent.created_at,
                        deadline=agent.created_at,
                        finished_at=agent.created_at,
                        public_reason=agent.screening_reason,
                    )
                )
        attempt = ScreeningAttempt(
            attempt_id=uuid4(),
            agent_id=agent.agent_id,
            screener_hotkey=screener_hotkey,
            policy_version=SCREENING_POLICY_VERSION,
            status="running",
            started_at=now,
            deadline=now + ttl,
        )
        session.add(attempt)
        agent.status = AgentStatus.SCREENING
        agent.screening_reason = None
        claimed.append((agent, attempt))
    await session.flush()
    return claimed


async def get_screening_attempt(
    session: AsyncSession,
    *,
    attempt_id: UUID,
    for_update: bool = False,
) -> ScreeningAttempt | None:
    stmt = select(ScreeningAttempt).where(ScreeningAttempt.attempt_id == attempt_id)
    if for_update:
        stmt = stmt.with_for_update()
    return (await session.scalars(stmt)).one_or_none()


async def list_screening_attempts(
    session: AsyncSession, *, agent_id: UUID
) -> list[ScreeningAttempt]:
    return list(
        await session.scalars(
            select(ScreeningAttempt)
            .where(ScreeningAttempt.agent_id == agent_id)
            .order_by(
                ScreeningAttempt.started_at.desc(),
                ScreeningAttempt.attempt_id.desc(),
            )
        )
    )


async def get_running_screening_attempts(
    session: AsyncSession, *, agent_ids: list[UUID]
) -> dict[UUID, ScreeningAttempt]:
    if not agent_ids:
        return {}
    attempts = await session.scalars(
        select(ScreeningAttempt).where(
            ScreeningAttempt.agent_id.in_(agent_ids),
            ScreeningAttempt.status == "running",
        )
    )
    return {attempt.agent_id: attempt for attempt in attempts}
