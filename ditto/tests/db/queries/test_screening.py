"""Tests for the cap that parks repeatedly-inconclusive agents for review.

Exercises the real ORM + SQLite-in-memory engine (same as the sibling query
tests) so the attempt/quarantine rows and the agent transition are real.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.db.models import Agent, AgentStatus, ScreeningAttempt, ScreeningQuarantine
from ditto.db.queries.screening import (
    MAX_SCREENING_EXPIRIES,
    claim_screening_attempts,
)

_SCREENER = "5GScreenerHotkeyForClaimTests000000000000000000000"


async def _seed_failed_agent(session: AsyncSession) -> Agent:
    agent = Agent(
        agent_id=uuid4(),
        miner_hotkey="5HKMinerHotkey",
        name="inconclusive-agent",
        sha256="de" * 32,
        status=AgentStatus.SCREENING_FAILED,
    )
    agent.screening_policy_version = SCREENING_POLICY_VERSION
    async with session.begin():
        session.add(agent)
    return agent


async def _add_expired_attempts(
    session: AsyncSession, agent: Agent, count: int
) -> None:
    base = datetime.now(UTC) - timedelta(hours=6)
    async with session.begin():
        for index in range(count):
            started = base + timedelta(minutes=45 * index)
            session.add(
                ScreeningAttempt(
                    attempt_id=uuid4(),
                    agent_id=agent.agent_id,
                    screener_hotkey=_SCREENER,
                    policy_version=SCREENING_POLICY_VERSION,
                    status="expired",
                    started_at=started,
                    deadline=started + timedelta(minutes=45),
                    finished_at=started + timedelta(minutes=45),
                    public_reason="Screening lease expired",
                )
            )


async def _claim(session: AsyncSession) -> list:
    async with session.begin():
        return await claim_screening_attempts(
            session,
            screener_hotkey=_SCREENER,
            now=datetime.now(UTC),
            ttl=timedelta(minutes=45),
            limit=10,
        )


async def test_agent_parked_for_review_after_expiry_cap(session: AsyncSession):
    agent = await _seed_failed_agent(session)
    await _add_expired_attempts(session, agent, MAX_SCREENING_EXPIRIES)

    claimed = await _claim(session)

    # It must not be leased out again...
    claimed_ids = {claimed_agent.agent_id for claimed_agent, _, _ in claimed}
    assert agent.agent_id not in claimed_ids
    # ...it is quarantined for operator review...
    refreshed = await session.get(Agent, agent.agent_id)
    assert refreshed is not None
    assert refreshed.status == AgentStatus.QUARANTINED
    assert refreshed.screening_reason_code == "repeatedly-inconclusive"
    # ...with an active quarantine row (what the operator console lists).
    quarantine = await session.scalar(
        select(ScreeningQuarantine).where(
            ScreeningQuarantine.agent_id == agent.agent_id
        )
    )
    assert quarantine is not None
    assert quarantine.status == "active"
    assert quarantine.reason_code == "repeatedly-inconclusive"
    assert len(quarantine.manifest_digest) == 64


async def test_agent_still_claimed_below_the_cap(session: AsyncSession):
    agent = await _seed_failed_agent(session)
    await _add_expired_attempts(session, agent, MAX_SCREENING_EXPIRIES - 1)

    claimed = await _claim(session)

    claimed_ids = {claimed_agent.agent_id for claimed_agent, _, _ in claimed}
    assert agent.agent_id in claimed_ids
    refreshed = await session.get(Agent, agent.agent_id)
    assert refreshed is not None
    assert refreshed.status == AgentStatus.SCREENING
    quarantine = await session.scalar(
        select(ScreeningQuarantine).where(
            ScreeningQuarantine.agent_id == agent.agent_id
        )
    )
    assert quarantine is None
