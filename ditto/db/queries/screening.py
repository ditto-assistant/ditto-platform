"""Leased screening attempts and their append-only public history."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import ColumnElement, case, exists, func, or_, select
from sqlalchemy.orm import aliased
from sqlalchemy.sql.selectable import ScalarSelect

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.benchmark_contract import benchmark_contracts
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.db.models import (
    Agent,
    BenchmarkDataset,
    BenchmarkRollout,
    BenchmarkRolloutMember,
    Score,
    ScreeningAttempt,
    ScreeningQuarantine,
)
from ditto.db.queries.scores import SCORING_QUORUM

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Expired attempts under the current policy after which an agent is parked for
# operator review instead of re-queued forever. An inconclusive screen (the
# harness never completes the private audit) submits no verdict, so its lease
# expires; a permanently-inconclusive agent would otherwise re-attempt every
# lease indefinitely. Only "expired" attempts count -- infrastructure "failed"
# attempts are usually screener-side, so a screener outage must not mass-park
# every in-flight agent.
MAX_SCREENING_EXPIRIES = 5

# Duplicate-owner statuses. A later cross-miner submission of the SAME bytes is
# flagged against the earliest owner in either set.
#
# "Usable" owners are live work being copied. "Adjudicated-negative" owners were
# refused, and only count when the refusal was for cause (see refused_for_cause
# in claim_screening_attempts) -- a build failure or an infrastructure reject
# says nothing about the artifact's provenance, so it must not condemn a later
# identical submission.
#
# In-flight statuses (UPLOADED / SCREENING / SCREENING_FAILED) belong to neither
# set: an owner still being screened is handled by deferring the claim
# (earlier_pending), not by flagging, so the race resolves before either is
# judged.
_USABLE_OWNER_STATUSES = (
    AgentStatus.EVALUATING,
    AgentStatus.SCORED,
    AgentStatus.LIVE,
    AgentStatus.ATH_PENDING_REVIEW,
)
_ADJUDICATED_NEGATIVE_OWNER_STATUSES = (
    AgentStatus.REJECTED,
    AgentStatus.QUARANTINED,
    AgentStatus.BANNED,
)

# A platform-raised quarantine has no screener finding, but the row's
# manifest_digest is NOT NULL and shown verbatim in the operator console. This
# stable sentinel marks the origin as "platform, attempts exhausted".
_EXHAUSTED_REASON_CODE = "repeatedly-inconclusive"
_EXHAUSTED_PUBLIC_REASON = (
    "Screening was inconclusive repeatedly; held for operator review"
)
_EXHAUSTED_MANIFEST_DIGEST = hashlib.sha256(
    b"ditto:repeatedly-inconclusive:v1"
).hexdigest()


def screening_score_count() -> ScalarSelect[int]:
    """Return the accepted-score count correlated to the current agent."""
    return (
        select(func.count())
        .where(Score.agent_id == Agent.agent_id)
        .correlate(Agent)
        .scalar_subquery()
    )


def screening_last_served_at() -> ColumnElement[Any]:
    """Return when current-policy screening last consumed a queue turn."""
    latest_attempt = (
        select(
            func.max(
                func.coalesce(
                    ScreeningAttempt.finished_at,
                    ScreeningAttempt.deadline,
                    ScreeningAttempt.started_at,
                )
            )
        )
        .where(
            ScreeningAttempt.agent_id == Agent.agent_id,
            ScreeningAttempt.policy_version == SCREENING_POLICY_VERSION,
        )
        .correlate(Agent)
        .scalar_subquery()
    )
    return func.coalesce(latest_attempt, Agent.created_at)


def screening_priority_order() -> tuple[ColumnElement[Any], ...]:
    """Prioritize finalists while interleaving bounded screening retries.

    A policy bump can return the whole scored field to screening. Submissions
    already one score from quorum should not lose their chance to finalize
    behind the rescreen backlog. Within each lane, the least recently served
    submission goes first: an expired lease moves an item behind the untouched
    backlog, but it remains ahead of submissions arriving later. This prevents
    either retries or fresh arrivals from monopolizing the worker while
    preserving the existing score and age tie-breakers.
    """
    score_count = screening_score_count()
    last_served_at = screening_last_served_at()
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
        last_served_at.asc(),
        Agent.created_at.asc(),
        Agent.agent_id.asc(),
    )


def missing_active_benchmark_dataset() -> ColumnElement[bool]:
    """Whether the current agent lacks the activated benchmark-version pin."""
    active_version = (
        select(BenchmarkRollout.desired_version)
        .where(BenchmarkRollout.status == "activated")
        .order_by(BenchmarkRollout.activated_at.desc())
        .limit(1)
        .scalar_subquery()
    )
    activation_exists = exists(
        select(BenchmarkRollout.rollout_id).where(
            BenchmarkRollout.status == "activated"
        )
    )
    versioned_dataset_exists = exists(
        select(BenchmarkDataset.agent_id).where(
            BenchmarkDataset.agent_id == Agent.agent_id,
            BenchmarkDataset.bench_version == active_version,
        )
    )
    return activation_exists & ~versioned_dataset_exists


def missing_active_screened_image() -> ColumnElement[bool]:
    """Whether the current agent lacks a complete screened image the activated
    benchmark version requires (v3+).

    Validators only lease an agent for a screened-image benchmark once every
    ``screened_image_*`` field is set (see ``eligible_screened_image`` in
    ``db/queries/tickets.py``). An agent quarantined mid-screen — before its
    image was uploaded and verified — and then RELEASED to ``evaluating`` on the
    current policy has an incomplete image: validators skip it, and without this
    predicate no screener re-claims it, so it is stuck for good. Re-screening it
    rebuilds and verifies the image (and its dataset)."""
    active_version = (
        select(BenchmarkRollout.desired_version)
        .where(BenchmarkRollout.status == "activated")
        .order_by(BenchmarkRollout.activated_at.desc())
        .limit(1)
        .scalar_subquery()
    )
    screened_versions = [
        contract.version
        for contract in benchmark_contracts()
        if contract.requires_screened_image
    ]
    incomplete_image = (
        Agent.screened_image_sha256.is_(None)
        | Agent.screened_image_size_bytes.is_(None)
        | Agent.screened_image_id.is_(None)
        | Agent.screened_image_ref.is_(None)
        | Agent.screened_image_upload_id.is_(None)
        | Agent.screened_image_verified_at.is_(None)
    )
    return active_version.in_(screened_versions) & incomplete_image


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


async def _expired_attempt_count(session: AsyncSession, *, agent_id: UUID) -> int:
    """Count expired screening leases under the current policy **since the
    agent's most recent operator rescreen**.

    An operator resolving a quarantine with ``rescreen`` explicitly grants a
    fresh attempt budget. Without the lower bound, an agent whose expiries
    came from a screener-fleet outage carries them forever: its next claim is
    instantly re-parked as ``repeatedly-inconclusive`` (started_at ==
    finished_at, no screening ever runs) and no number of operator rescreens
    can break the loop — exactly what happened on 2026-07-16 when 12 freshly
    rescreened agents were re-parked within seconds of each other.
    """
    last_rescreen = (
        select(func.max(ScreeningQuarantine.resolved_at))
        .where(
            ScreeningQuarantine.agent_id == agent_id,
            ScreeningQuarantine.resolution == "rescreen",
        )
        .scalar_subquery()
    )
    count = await session.scalar(
        select(func.count())
        .select_from(ScreeningAttempt)
        .where(
            ScreeningAttempt.agent_id == agent_id,
            ScreeningAttempt.policy_version == SCREENING_POLICY_VERSION,
            ScreeningAttempt.status == "expired",
            ScreeningAttempt.started_at
            > func.coalesce(last_rescreen, datetime(1970, 1, 1, tzinfo=UTC)),
        )
    )
    return int(count or 0)


async def _park_repeatedly_inconclusive(
    session: AsyncSession,
    agent: Agent,
    *,
    screener_hotkey: str,
    now: datetime,
) -> None:
    """Quarantine an agent that keeps expiring its lease, for operator review.

    Records a terminal ``quarantined`` attempt plus an active quarantine row
    (the operator console is driven entirely by ``ScreeningQuarantine``) so the
    agent leaves the retry pool and a human decides its fate instead of the
    screener re-attempting it every lease forever.
    """
    attempt = ScreeningAttempt(
        attempt_id=uuid4(),
        agent_id=agent.agent_id,
        screener_hotkey=screener_hotkey,
        policy_version=SCREENING_POLICY_VERSION,
        status="quarantined",
        started_at=now,
        deadline=now,
        finished_at=now,
        public_reason=_EXHAUSTED_PUBLIC_REASON,
        reason_code=_EXHAUSTED_REASON_CODE,
    )
    session.add(attempt)
    # Flush so the attempt row exists before the quarantine's FK references it
    # (no ORM relationship links them to order the inserts automatically).
    await session.flush()
    session.add(
        ScreeningQuarantine(
            quarantine_id=uuid4(),
            agent_id=agent.agent_id,
            attempt_id=attempt.attempt_id,
            screener_hotkey=screener_hotkey,
            policy_version=SCREENING_POLICY_VERSION,
            manifest_digest=_EXHAUSTED_MANIFEST_DIGEST,
            finding_digest=None,
            reason_code=_EXHAUSTED_REASON_CODE,
            evidence=None,
            finding=None,
            status="active",
        )
    )
    agent.status = AgentStatus.QUARANTINED
    agent.screening_reason = _EXHAUSTED_PUBLIC_REASON
    agent.screening_reason_code = _EXHAUSTED_REASON_CODE


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
    # A REJECTED agent is deliberately absent above. It re-enters screening only
    # through the operator appeal (POST /screening-submissions/{id}/rescreen),
    # which moves it to SCREENING_FAILED. Re-queueing it on a policy bump instead
    # resurrected every past rejection fleet-wide and cleared the operator's
    # stated reason, so a refused artifact could return under a newer policy that
    # never re-derived the original finding.
    await expire_screening_attempts(session, now=now)
    has_running = exists(
        select(ScreeningAttempt.attempt_id).where(
            ScreeningAttempt.agent_id == Agent.agent_id,
            ScreeningAttempt.status == "running",
        )
    )
    rolling_qualified = exists(
        select(BenchmarkRolloutMember.agent_id)
        .join(
            BenchmarkRollout,
            BenchmarkRollout.rollout_id == BenchmarkRolloutMember.rollout_id,
        )
        .where(
            BenchmarkRolloutMember.agent_id == Agent.agent_id,
            BenchmarkRollout.status.in_(("collecting", "blocked_ineligible")),
        )
    )
    missing_v3_screen = (
        (Agent.screening_policy_version < SCREENING_POLICY_VERSION)
        | Agent.screened_image_sha256.is_(None)
        | Agent.screened_image_size_bytes.is_(None)
        | Agent.screened_image_id.is_(None)
        | Agent.screened_image_ref.is_(None)
        | Agent.screened_image_upload_id.is_(None)
        | Agent.screened_image_verified_at.is_(None)
    )
    eligible = or_(
        Agent.status == AgentStatus.UPLOADED,
        Agent.status == AgentStatus.SCREENING_FAILED,
        (
            (Agent.status == AgentStatus.EVALUATING)
            & (Agent.screening_policy_version < SCREENING_POLICY_VERSION)
        ),
        (
            Agent.status.in_((AgentStatus.SCORED, AgentStatus.LIVE))
            & rolling_qualified
            & missing_v3_screen
        ),
        ((Agent.status == AgentStatus.EVALUATING) & missing_active_benchmark_dataset()),
        # A submission released from an anti-cheat quarantine back to EVALUATING
        # but without a complete screened image the active version needs is
        # otherwise stuck forever — validators skip it and nothing re-screens it.
        ((Agent.status == AgentStatus.EVALUATING) & missing_active_screened_image()),
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
        # An agent that keeps coming back inconclusive expires its lease every
        # cycle; after the cap, park it for operator review instead of leasing
        # it out again to loop forever.
        if (
            await _expired_attempt_count(session, agent_id=agent.agent_id)
            >= MAX_SCREENING_EXPIRIES
        ):
            await _park_repeatedly_inconclusive(
                session, agent, screener_hotkey=screener_hotkey, now=now
            )
            continue
        owner = aliased(Agent)
        # An artifact refused FOR CAUSE stays a valid duplicate owner. Scoping
        # owners to live statuses alone meant banning an original disarmed this
        # check for its clones: the very act of refusing the first copy removed
        # the row that would flag the next one.
        #
        # "For cause" is read from quarantine history rather than the agent row,
        # because a re-screen clears screening_reason_code. An active hold counts
        # (a finding is outstanding); a hold an operator resolved as release or
        # rescreen does not -- they cleared it deliberately.
        #
        # A platform-raised _EXHAUSTED_REASON_CODE park is the exception: it is an
        # infrastructure outcome (the screen never concluded), not a finding about
        # the artifact, so a screener outage must not turn every parked original
        # into grounds for condemning a later identical submission. Once an
        # operator reviews that park and resolves it "reject", the rejection
        # branch below picks it up -- that IS a human judgement for cause.
        refused_for_cause = or_(
            owner.status == AgentStatus.BANNED,
            exists(
                select(ScreeningQuarantine.quarantine_id).where(
                    ScreeningQuarantine.agent_id == owner.agent_id,
                    or_(
                        (ScreeningQuarantine.status == "active")
                        & (ScreeningQuarantine.reason_code != _EXHAUSTED_REASON_CODE),
                        ScreeningQuarantine.resolution == "reject",
                    ),
                )
            ),
        )
        duplicate_of = await session.scalar(
            select(owner.agent_id)
            .where(
                owner.sha256 == agent.sha256,
                owner.miner_hotkey != agent.miner_hotkey,
                owner.agent_id != agent.agent_id,
                (owner.created_at < agent.created_at)
                | (
                    (owner.created_at == agent.created_at)
                    & (owner.agent_id < agent.agent_id)
                ),
                or_(
                    owner.status.in_(_USABLE_OWNER_STATUSES),
                    owner.status.in_(_ADJUDICATED_NEGATIVE_OWNER_STATUSES)
                    & refused_for_cause,
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
        if agent.status not in (AgentStatus.SCORED, AgentStatus.LIVE):
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
