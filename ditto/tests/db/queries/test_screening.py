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
    _EXHAUSTED_REASON_CODE,
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


async def _seed_owner_and_duplicate(
    session: AsyncSession,
    *,
    owner_status: AgentStatus,
) -> tuple[Agent, Agent]:
    """Seed an earlier owner and a later, different-miner copy of the SAME bytes.

    The copy is a fresh UPLOADED submission, so it is claimable and the claim
    runs the cross-miner duplicate precheck against the owner.
    """
    sha256 = uuid4().hex * 2
    created = datetime.now(UTC) - timedelta(days=2)
    owner = Agent(
        agent_id=uuid4(),
        miner_hotkey="5HK-owner",
        name="original",
        sha256=sha256,
        status=owner_status,
        created_at=created,
    )
    owner.screening_policy_version = SCREENING_POLICY_VERSION
    duplicate = Agent(
        agent_id=uuid4(),
        miner_hotkey="5HK-copycat",
        name="copy",
        sha256=sha256,
        status=AgentStatus.UPLOADED,
        created_at=created + timedelta(days=1),
    )
    async with session.begin():
        session.add(owner)
        session.add(duplicate)
    return owner, duplicate


# A real screener finding, as opposed to the platform-raised exhaustion
# sentinel. The two are NOT interchangeable: only a screener finding is "for
# cause", so every for-cause test must state which one it seeds.
_SCREENER_FINDING_REASON_CODE = "source-review"


async def _add_quarantine(
    session: AsyncSession,
    agent: Agent,
    *,
    status: str,
    resolution: str | None = None,
    reason_code: str = _SCREENER_FINDING_REASON_CODE,
) -> None:
    """Attach one quarantine row (plus the attempt its FK requires) to an agent."""
    at = datetime.now(UTC) - timedelta(days=1)
    async with session.begin():
        attempt = ScreeningAttempt(
            attempt_id=uuid4(),
            agent_id=agent.agent_id,
            screener_hotkey=_SCREENER,
            policy_version=SCREENING_POLICY_VERSION,
            status="quarantined",
            started_at=at,
            deadline=at,
            finished_at=at,
            public_reason="held for review",
            reason_code=reason_code,
        )
        session.add(attempt)
        await session.flush()
        session.add(
            ScreeningQuarantine(
                quarantine_id=uuid4(),
                agent_id=agent.agent_id,
                attempt_id=attempt.attempt_id,
                screener_hotkey=_SCREENER,
                policy_version=SCREENING_POLICY_VERSION,
                manifest_digest="a" * 64,
                finding_digest=None,
                reason_code=reason_code,
                evidence=None,
                finding=None,
                status=status,
                resolved_at=None if resolution is None else at,
                resolved_by=None if resolution is None else "operator",
                resolution=resolution,
                resolution_reason=None if resolution is None else "operator decision",
            )
        )


def _claimed_duplicate(
    claimed: list[tuple[Agent, ScreeningAttempt, object]], agent: Agent
) -> tuple[ScreeningAttempt, object]:
    """Return the (attempt, duplicate_of) the claim produced for ``agent``."""
    for claimed_agent, attempt, duplicate_of in claimed:
        if claimed_agent.agent_id == agent.agent_id:
            return attempt, duplicate_of
    raise AssertionError("agent was not claimed")


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


async def test_duplicate_flagged_when_owner_rejected_with_reject_resolution(
    session: AsyncSession,
):
    """A copy of an artifact whose original was rejected FOR CAUSE is flagged.

    The 716ditto case: refusing the original used to remove it from the owner
    set, so the very act of adjudicating the first copy disarmed the detector
    for every later identical submission.
    """
    owner, duplicate = await _seed_owner_and_duplicate(
        session, owner_status=AgentStatus.REJECTED
    )
    await _add_quarantine(session, owner, status="resolved", resolution="reject")

    claimed = await _claim(session)

    attempt, duplicate_of = _claimed_duplicate(claimed, duplicate)
    assert duplicate_of == owner.agent_id
    assert attempt.reason_code == "exact-cross-miner-duplicate"
    assert attempt.duplicate_of == owner.agent_id


async def test_duplicate_flagged_when_owner_banned_without_quarantine(
    session: AsyncSession,
):
    """BANNED is for-cause on its own; a ban may be issued with no quarantine row."""
    owner, duplicate = await _seed_owner_and_duplicate(
        session, owner_status=AgentStatus.BANNED
    )

    claimed = await _claim(session)

    attempt, duplicate_of = _claimed_duplicate(claimed, duplicate)
    assert duplicate_of == owner.agent_id
    assert attempt.reason_code == "exact-cross-miner-duplicate"


async def test_duplicate_flagged_when_owner_has_active_quarantine(
    session: AsyncSession,
):
    """An outstanding SCREENER FINDING counts as for-cause while the operator decides.

    The finding reason_code is asserted explicitly: an active quarantine alone
    is not enough, since the platform raises active quarantines of its own for
    exhausted attempts (see the exhaustion-sentinel test below). If this test
    silently seeded the sentinel it would be passing for the wrong reason.
    """
    owner, duplicate = await _seed_owner_and_duplicate(
        session, owner_status=AgentStatus.QUARANTINED
    )
    await _add_quarantine(
        session, owner, status="active", reason_code=_SCREENER_FINDING_REASON_CODE
    )
    assert _SCREENER_FINDING_REASON_CODE != _EXHAUSTED_REASON_CODE

    claimed = await _claim(session)

    attempt, duplicate_of = _claimed_duplicate(claimed, duplicate)
    assert duplicate_of == owner.agent_id
    assert attempt.reason_code == "exact-cross-miner-duplicate"


async def test_duplicate_not_flagged_when_owner_parked_as_inconclusive(
    session: AsyncSession,
):
    """False-positive guard: a platform-raised exhaustion park is not for cause.

    ``_park_repeatedly_inconclusive`` writes an ACTIVE quarantine carrying the
    ``repeatedly-inconclusive`` sentinel when an agent keeps expiring its lease.
    That is an infrastructure outcome, not a provenance finding — the 2026-07-16
    incident parked 12 agents purely from a screener-fleet outage. Treating such
    a park as for-cause would let an outage condemn every later identical
    cross-miner submission, which is the same false positive the build/infra
    rejection guard above exists to prevent.
    """
    owner, duplicate = await _seed_owner_and_duplicate(
        session, owner_status=AgentStatus.QUARANTINED
    )
    await _add_quarantine(
        session, owner, status="active", reason_code=_EXHAUSTED_REASON_CODE
    )

    claimed = await _claim(session)

    attempt, duplicate_of = _claimed_duplicate(claimed, duplicate)
    assert duplicate_of is None
    assert attempt.reason_code is None


async def test_duplicate_flagged_when_operator_rejects_an_inconclusive_park(
    session: AsyncSession,
):
    """Human judgement overrides the infra origin of the park.

    The sentinel only excuses the park itself. Once an operator reviewed it and
    resolved ``reject``, that IS an adjudicated refusal for cause, so the owner
    is a valid duplicate owner again regardless of how the hold started.
    """
    owner, duplicate = await _seed_owner_and_duplicate(
        session, owner_status=AgentStatus.REJECTED
    )
    await _add_quarantine(
        session,
        owner,
        status="resolved",
        resolution="reject",
        reason_code=_EXHAUSTED_REASON_CODE,
    )

    claimed = await _claim(session)

    attempt, duplicate_of = _claimed_duplicate(claimed, duplicate)
    assert duplicate_of == owner.agent_id
    assert attempt.reason_code == "exact-cross-miner-duplicate"


async def test_duplicate_not_flagged_when_owner_rejected_without_quarantine(
    session: AsyncSession,
):
    """False-positive guard: a build/infra rejection must not condemn a copy.

    Such a rejection writes no quarantine row, so nothing was ever adjudicated
    about the artifact's provenance. Flagging here would punish an honest
    resubmission for the platform's own build failure.
    """
    owner, duplicate = await _seed_owner_and_duplicate(
        session, owner_status=AgentStatus.REJECTED
    )

    claimed = await _claim(session)

    attempt, duplicate_of = _claimed_duplicate(claimed, duplicate)
    assert duplicate_of is None
    assert attempt.reason_code is None


async def test_duplicate_not_flagged_when_owner_quarantine_released(
    session: AsyncSession,
):
    """An operator ``release`` deliberately clears the finding.

    The agent row's screening_reason_code is wiped by a re-screen, so the
    for-cause test reads quarantine history; a released hold must read as
    "cleared", not as a standing finding that condemns later copies.
    """
    owner, duplicate = await _seed_owner_and_duplicate(
        session, owner_status=AgentStatus.REJECTED
    )
    await _add_quarantine(session, owner, status="resolved", resolution="release")

    claimed = await _claim(session)

    attempt, duplicate_of = _claimed_duplicate(claimed, duplicate)
    assert duplicate_of is None
    assert attempt.reason_code is None


async def test_duplicate_not_flagged_when_owner_quarantine_rescreened(
    session: AsyncSession,
):
    """``rescreen`` is likewise an operator clearing the hold, not a finding."""
    owner, duplicate = await _seed_owner_and_duplicate(
        session, owner_status=AgentStatus.REJECTED
    )
    await _add_quarantine(session, owner, status="resolved", resolution="rescreen")

    claimed = await _claim(session)

    attempt, duplicate_of = _claimed_duplicate(claimed, duplicate)
    assert duplicate_of is None
    assert attempt.reason_code is None


async def test_duplicate_flagged_when_owner_is_usable(session: AsyncSession):
    """Unchanged behavior: live work being copied is still flagged."""
    owner, duplicate = await _seed_owner_and_duplicate(
        session, owner_status=AgentStatus.EVALUATING
    )

    claimed = await _claim(session)

    attempt, duplicate_of = _claimed_duplicate(claimed, duplicate)
    assert duplicate_of == owner.agent_id
    assert attempt.reason_code == "exact-cross-miner-duplicate"


async def _seed_agent_at_policy(
    session: AsyncSession, *, status: AgentStatus, policy_version: int
) -> Agent:
    agent = Agent(
        agent_id=uuid4(),
        miner_hotkey="5HK-requeue",
        name="stale-policy-agent",
        sha256=uuid4().hex * 2,
        status=status,
    )
    agent.screening_policy_version = policy_version
    async with session.begin():
        session.add(agent)
    return agent


async def test_rejected_agent_is_not_auto_requeued_on_policy_bump(
    session: AsyncSession,
):
    """A refused artifact must not return just because the policy version moved.

    Auto-requeue resurrected every past rejection fleet-wide and cleared the
    operator's stated reason, letting a refused artifact back in under a policy
    that never re-derived the original finding.
    """
    agent = await _seed_agent_at_policy(
        session,
        status=AgentStatus.REJECTED,
        policy_version=SCREENING_POLICY_VERSION - 1,
    )

    claimed = await _claim(session)

    assert agent.agent_id not in {
        claimed_agent.agent_id for claimed_agent, _, _ in claimed
    }
    refreshed = await session.get(Agent, agent.agent_id)
    assert refreshed is not None
    assert refreshed.status == AgentStatus.REJECTED


async def test_appealed_agent_in_screening_failed_is_claimable(session: AsyncSession):
    """The operator appeal endpoint moves REJECTED -> SCREENING_FAILED.

    That is the ONLY re-entry path now, so it must still be claimable.
    """
    agent = await _seed_agent_at_policy(
        session,
        status=AgentStatus.SCREENING_FAILED,
        policy_version=SCREENING_POLICY_VERSION - 1,
    )

    claimed = await _claim(session)

    assert agent.agent_id in {claimed_agent.agent_id for claimed_agent, _, _ in claimed}
    refreshed = await session.get(Agent, agent.agent_id)
    assert refreshed is not None
    assert refreshed.status == AgentStatus.SCREENING
