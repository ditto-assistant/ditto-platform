"""Leased screening attempts and their append-only public history."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import ColumnElement, case, exists, func, or_, select
from sqlalchemy.orm import aliased
from sqlalchemy.sql.selectable import ScalarSelect

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.db.models import Agent, Score, ScreeningAttempt
from ditto.db.queries.scores import SCORING_QUORUM

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


def screening_priority_order() -> tuple[ColumnElement[Any], ...]:
    """Prioritize likely finalists, then preserve least-scored fairness.

    A policy bump can return the whole scored field to screening. Submissions
    already one score from quorum should not lose their chance to finalize
    behind the rescreen backlog, so that completion lane drains by provisional
    score. Everything else keeps the existing least-scored, oldest-first order.
    """
    score_count = screening_score_count()
    provisional_composite = (
        select(func.avg(Score.composite))
        .where(Score.agent_id == Agent.agent_id)
        .correlate(Agent)
        .scalar_subquery()
    )
    in_completion_lane = case(
        (score_count >= SCORING_QUORUM - 1, 1),
        else_=0,
    )
    completion_lane_score = case(
        (score_count >= SCORING_QUORUM - 1, provisional_composite),
        else_=0.0,
    )
    return (
        in_completion_lane.desc(),
        completion_lane_score.desc(),
        score_count.asc(),
        Agent.created_at.asc(),
        Agent.agent_id.asc(),
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
) -> list[tuple[Agent, ScreeningAttempt, UUID | None]]:
    """Claim completion-lane contenders, then least-scored eligible work."""
    # Claiming is already a short transaction. Serialize it in Postgres so two
    # workers cannot skip-lock sibling rows with the same hash and admit both.
    # SQLite serializes writes itself and does not provide advisory locks.
    bind = session.get_bind()
    if bind.dialect.name == "postgresql":
        await session.execute(select(func.pg_advisory_xact_lock(0x445554544F534352)))
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
    earlier = aliased(Agent)
    earlier_pending = exists(
        select(earlier.agent_id).where(
            earlier.sha256 == Agent.sha256,
            earlier.miner_hotkey != Agent.miner_hotkey,
            (earlier.created_at < Agent.created_at)
            | (
                (earlier.created_at == Agent.created_at)
                & (earlier.agent_id < Agent.agent_id)
            ),
            earlier.status.in_(
                (
                    AgentStatus.UPLOADED,
                    AgentStatus.SCREENING,
                    AgentStatus.SCREENING_FAILED,
                )
            ),
        )
    )
    agents = list(
        await session.scalars(
            select(Agent)
            .where(eligible, ~has_running, ~earlier_pending)
            .order_by(*screening_priority_order())
            .limit(limit)
            .with_for_update(of=Agent, skip_locked=True)
        )
    )
    claimed: list[tuple[Agent, ScreeningAttempt, UUID | None]] = []
    for agent in agents:
        owner = aliased(Agent)
        duplicate_of = await session.scalar(
            select(owner.agent_id)
            .where(
                owner.sha256 == agent.sha256,
                owner.miner_hotkey != agent.miner_hotkey,
                owner.agent_id != agent.agent_id,
                owner.status.in_(
                    (
                        AgentStatus.EVALUATING,
                        AgentStatus.SCORED,
                        AgentStatus.LIVE,
                        AgentStatus.ATH_PENDING_REVIEW,
                    )
                ),
            )
            .order_by(owner.created_at.asc(), owner.agent_id.asc())
            .limit(1)
        )
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
            reason_code=(
                "exact-cross-miner-duplicate" if duplicate_of is not None else None
            ),
            duplicate_of=duplicate_of,
        )
        session.add(attempt)
        agent.status = AgentStatus.SCREENING
        agent.screening_reason = None
        agent.screening_reason_code = None
        claimed.append((agent, attempt, duplicate_of))
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
