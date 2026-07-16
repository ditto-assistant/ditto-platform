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


async def _seed_failed_agent_with_age(
    session: AsyncSession, *, name: str, age: timedelta
) -> Agent:
    agent = Agent(
        agent_id=uuid4(),
        miner_hotkey=f"5HK-{name}",
        name=name,
        sha256=uuid4().hex * 2,
        status=AgentStatus.SCREENING_FAILED,
        created_at=datetime.now(UTC) - age,
    )
    agent.screening_policy_version = SCREENING_POLICY_VERSION
    async with session.begin():
        session.add(agent)
    return agent


async def _add_expired_attempts(
    session: AsyncSession,
    agent: Agent,
    count: int,
    *,
    policy_version: int = SCREENING_POLICY_VERSION,
    base: datetime | None = None,
) -> None:
    base = base or datetime.now(UTC) - timedelta(hours=6)
    async with session.begin():
        for index in range(count):
            started = base + timedelta(minutes=45 * index)
            session.add(
                ScreeningAttempt(
                    attempt_id=uuid4(),
                    agent_id=agent.agent_id,
                    screener_hotkey=_SCREENER,
                    policy_version=policy_version,
                    status="expired",
                    started_at=started,
                    deadline=started + timedelta(minutes=45),
                    finished_at=started + timedelta(minutes=45),
                    public_reason="Screening lease expired",
                )
            )


async def _claim(session: AsyncSession, *, limit: int = 10) -> list:
    async with session.begin():
        return await claim_screening_attempts(
            session,
            screener_hotkey=_SCREENER,
            now=datetime.now(UTC),
            ttl=timedelta(minutes=45),
            limit=limit,
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


async def test_fresh_agent_runs_before_older_retry(session: AsyncSession):
    retry = await _seed_failed_agent_with_age(
        session, name="older-retry", age=timedelta(days=2)
    )
    fresh = await _seed_failed_agent_with_age(
        session, name="fresh-work", age=timedelta(days=1)
    )
    await _add_expired_attempts(session, retry, 1)

    claimed = await _claim(session, limit=1)

    assert [agent.agent_id for agent, _, _ in claimed] == [fresh.agent_id]


async def test_retry_runs_before_a_later_arriving_fresh_agent(
    session: AsyncSession,
):
    retry = await _seed_failed_agent_with_age(
        session, name="retry", age=timedelta(days=2)
    )
    await _seed_failed_agent_with_age(
        session, name="later-fresh", age=timedelta(hours=1)
    )
    await _add_expired_attempts(session, retry, 1)

    claimed = await _claim(session, limit=1)

    assert [agent.agent_id for agent, _, _ in claimed] == [retry.agent_id]


async def test_previous_policy_attempt_does_not_defer_current_policy_work(
    session: AsyncSession,
):
    older = await _seed_failed_agent_with_age(
        session, name="older-policy-history", age=timedelta(days=2)
    )
    await _seed_failed_agent_with_age(
        session, name="newer-current-work", age=timedelta(days=1)
    )
    await _add_expired_attempts(
        session,
        older,
        1,
        policy_version=SCREENING_POLICY_VERSION - 1,
        base=datetime.now(UTC) - timedelta(minutes=45),
    )

    claimed = await _claim(session, limit=1)

    assert [agent.agent_id for agent, _, _ in claimed] == [older.agent_id]


async def test_operator_rescreen_resets_the_expiry_budget(session: AsyncSession):
    """A quarantine resolved with ``rescreen`` grants a fresh attempt budget.

    Regression for 2026-07-16: agents whose expiries came from a screener
    fleet outage were instantly re-parked on the next claim after an operator
    rescreen, because the expiry count ignored the rescreen entirely.
    """
    agent = await _seed_failed_agent(session)
    base = datetime.now(UTC) - timedelta(hours=6)
    await _add_expired_attempts(session, agent, MAX_SCREENING_EXPIRIES, base=base)
    # The exhaustion park + the operator's rescreen, AFTER the expiries.
    async with session.begin():
        park_attempt = ScreeningAttempt(
            attempt_id=uuid4(),
            agent_id=agent.agent_id,
            screener_hotkey=_SCREENER,
            policy_version=SCREENING_POLICY_VERSION,
            status="quarantined",
            started_at=base + timedelta(hours=5),
            deadline=base + timedelta(hours=5),
            finished_at=base + timedelta(hours=5),
            public_reason="Screening was inconclusive repeatedly",
            reason_code="repeatedly-inconclusive",
        )
        session.add(park_attempt)
        await session.flush()
        session.add(
            ScreeningQuarantine(
                quarantine_id=uuid4(),
                agent_id=agent.agent_id,
                attempt_id=park_attempt.attempt_id,
                screener_hotkey=_SCREENER,
                policy_version=SCREENING_POLICY_VERSION,
                manifest_digest="d" * 64,
                finding_digest=None,
                reason_code="repeatedly-inconclusive",
                evidence=None,
                finding=None,
                status="resolved",
                resolved_at=datetime.now(UTC) - timedelta(minutes=30),
                resolved_by="operator",
                resolution="rescreen",
                resolution_reason="fleet outage, not agent behavior",
            )
        )

    claimed = await _claim(session)

    # The rescreen zeroed the budget: the agent is leased out for a REAL run,
    # not instantly re-parked.
    claimed_ids = {claimed_agent.agent_id for claimed_agent, _, _ in claimed}
    assert agent.agent_id in claimed_ids
    refreshed = await session.get(Agent, agent.agent_id)
    assert refreshed is not None
    assert refreshed.status == AgentStatus.SCREENING


async def test_expiries_after_a_rescreen_still_exhaust(session: AsyncSession):
    """Only pre-rescreen expiries are forgiven; the cap still protects the pool."""
    agent = await _seed_failed_agent(session)
    rescreened_at = datetime.now(UTC) - timedelta(hours=5)
    async with session.begin():
        anchor = ScreeningAttempt(
            attempt_id=uuid4(),
            agent_id=agent.agent_id,
            screener_hotkey=_SCREENER,
            policy_version=SCREENING_POLICY_VERSION,
            status="quarantined",
            started_at=rescreened_at - timedelta(minutes=1),
            deadline=rescreened_at - timedelta(minutes=1),
            finished_at=rescreened_at - timedelta(minutes=1),
            public_reason="parked",
            reason_code="repeatedly-inconclusive",
        )
        session.add(anchor)
        await session.flush()
        session.add(
            ScreeningQuarantine(
                quarantine_id=uuid4(),
                agent_id=agent.agent_id,
                attempt_id=anchor.attempt_id,
                screener_hotkey=_SCREENER,
                policy_version=SCREENING_POLICY_VERSION,
                manifest_digest="d" * 64,
                finding_digest=None,
                reason_code="repeatedly-inconclusive",
                evidence=None,
                finding=None,
                status="resolved",
                resolved_at=rescreened_at,
                resolved_by="operator",
                resolution="rescreen",
                resolution_reason="grant a fresh budget",
            )
        )
    # A fresh cap's worth of expiries AFTER the rescreen…
    await _add_expired_attempts(
        session,
        agent,
        MAX_SCREENING_EXPIRIES,
        base=rescreened_at + timedelta(minutes=5),
    )

    claimed = await _claim(session)

    # …parks it again: the reset is not a permanent exemption.
    claimed_ids = {claimed_agent.agent_id for claimed_agent, _, _ in claimed}
    assert agent.agent_id not in claimed_ids
    refreshed = await session.get(Agent, agent.agent_id)
    assert refreshed is not None
    assert refreshed.status == AgentStatus.QUARANTINED
