"""Mutations + reads against the ``validator_tickets`` table (the k=3 pool).

A submission (agent) is scored by at most :data:`SCORING_QUORUM` validators. A
ticket is issued on demand to a validator that does not already hold one for the
agent, expires if unscored by its deadline (freeing the slot), and is marked
``scored`` when the validator posts a valid score in time.

Issuance locks the candidate agent row and then recounts its occupied slots in a
fresh statement. Concurrent platform replicas therefore serialize allocation
for a given agent and cannot over-issue a fourth slot. The ``(agent_id,
validator_hotkey)`` primary key separately guarantees a validator can hold only
one ticket per agent. A partial unique index plus a per-validator transaction
lock guarantees one validator cannot hold two live assignments across agents.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from sqlalchemy import and_, case, func, literal, or_, select
from sqlalchemy.orm import aliased

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.benchmark_contract import benchmark_contract
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.api_models.ticket_status import TicketPurpose, TicketStatus
from ditto.db.models import (
    Agent,
    BenchmarkDataset,
    BenchmarkRollout,
    EvaluationPayment,
    Score,
    ValidatorTicket,
)
from ditto.db.queries.audit import (
    EVENT_SCORE_RETEST_REQUESTED,
    get_latest_score_retest_event,
)
from ditto.db.queries.benchmark_admission import (
    activated_rollout_for_version,
    benchmark_admission_predicate,
)
from ditto.db.queries.scores import SCORING_QUORUM, list_eligible_ledger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _as_utc(dt: datetime) -> datetime:
    """Coerce a DB-read datetime to UTC-aware. The SQLite test path round-trips
    timestamps tz-naive; Postgres preserves the zone. Keeps ``deadline``
    comparisons from mixing naive and aware datetimes."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


# Statuses that occupy a slot: an issued (live) or already-scored ticket. An
# expired ticket does not count, so its slot re-opens.
_LIVE_TICKET_STATUSES = (TicketStatus.ISSUED, TicketStatus.SCORED)

# A timed-out artifact must not monopolize one validator. Give transient
# failures one delayed retry, while allowing other validators to make an
# independent attempt immediately.
RETRY_COOLDOWN = timedelta(hours=6)
MAX_ATTEMPTS_PER_VERSION = 2

# An infrastructure failure (a signed ``fail_job`` with reason
# ``infrastructure``) is never the agent's fault, so it earns a compensating
# grant that offsets the attempt the reissue consumes. Bounded so a persistent
# validator-side outage cannot re-lease one agent forever.
MAX_INFRA_RETRY_GRANTS = 8

# Infrastructure retries reissue quickly (no 6h agent-failure cooldown) so a
# transient blip recovers fast, but back-to-back re-leases during a *sustained*
# provider/relay outage would hammer the failing provider (an inference burst).
# The cooldown before the next infra retry therefore doubles per grant already
# earned, capped, so the agent is still retried to success while the failing
# provider gets breathing room.
INFRA_RETRY_BACKOFF_BASE = timedelta(minutes=2)
INFRA_RETRY_BACKOFF_CAP = timedelta(minutes=30)


def infra_retry_backoff(infra_retry_grants: int) -> timedelta:
    """Cooldown before an infrastructure-failed lease may be re-leased.

    ``infra_retry_grants`` is the count *after* this failure bumped it (so the
    first infra failure passes ``1``). Doubles per prior grant, capped at
    :data:`INFRA_RETRY_BACKOFF_CAP`.
    """
    if infra_retry_grants <= 1:
        return INFRA_RETRY_BACKOFF_BASE
    # Clamp the exponent so a large count can't overflow the timedelta multiply;
    # anything past the cap is clamped to it anyway (real inputs are <= 8).
    steps = min(infra_retry_grants - 1, 20)
    scaled = INFRA_RETRY_BACKOFF_BASE * (2**steps)
    return min(scaled, INFRA_RETRY_BACKOFF_CAP)


def ticket_attempt_cap(ticket: ValidatorTicket) -> int:
    """Total leases a validator may spend on this agent+version.

    The base per-version budget, plus audited operator grants, plus
    infrastructure-failure compensation. ``attempt_count`` is compared against
    this everywhere issuance, exhaustion, and natural-retry eligibility are
    decided, so the three surfaces agree on one budget.
    """
    return (
        MAX_ATTEMPTS_PER_VERSION
        + ticket.manual_retry_grants
        + ticket.infra_retry_grants
    )


# The KOTH champion plus four participation-tail miners receive emissions.
# Ticket continuation uses the current fifth finalized score as a dynamic floor,
# but only after two scores: with median-of-three, the highest final score still
# reachable from two observations is their maximum. A single low first score is
# never sufficient to eliminate a submission because two later high scores can
# make that first score irrelevant to the median.
EMISSION_CONTENDER_COUNT = 5

# Advance a bounded set of likely leaderboard contenders before the ordinary
# coverage rounds. Keeping one best submission per miner in this small lane
# lets a strong 1-of-3 result reach 2-of-3 quickly and still finishes strong
# 2-of-3 submissions, without allowing the whole scored backlog to starve new
# miners.
PROVISIONAL_CONTENDER_LANE_SIZE = 10


async def get_score_priority_floors(
    session: AsyncSession, *, bench_version: int | None = None
) -> tuple[float | None, float | None]:
    """Return finalized fifth-place and tenth-place floors for one benchmark era."""
    eligible = [
        row
        for row in await list_eligible_ledger(
            session, include_fingerprints=False, bench_version=bench_version
        )
        if row.eligible
    ]
    continuation = (
        eligible[EMISSION_CONTENDER_COUNT - 1].composite
        if len(eligible) >= EMISSION_CONTENDER_COUNT
        else None
    )
    provisional = (
        eligible[PROVISIONAL_CONTENDER_LANE_SIZE - 1].composite
        if len(eligible) >= PROVISIONAL_CONTENDER_LANE_SIZE
        else None
    )
    return continuation, provisional


async def get_score_continuation_floor(
    session: AsyncSession, *, bench_version: int | None = None
) -> float | None:
    """Return the finalized fifth-place score for one benchmark era, if five exist.

    The ledger is already best-agent-per-miner and ordered by the same
    eligibility, composite, age, and UUID rules used for emissions. Provisional
    rows remain visible in that ledger, so filter them before selecting fifth.

    ``bench_version`` pins the era the floor is drawn from. Composites are only
    comparable within one benchmark version, so a floor must never be blended
    across versions: left unset, :func:`list_eligible_ledger` pools the
    canonical version together with an open rollout's desired version, and the
    resulting fifth place belongs to no single era. Callers that compare a
    version-scoped composite against this floor must pass that same version.
    Returns ``None`` when the era does not yet have five eligible agents, which
    correctly disables the floor for a benchmark version still filling up.
    """
    continuation, _ = await get_score_priority_floors(
        session, bench_version=bench_version
    )
    return continuation


async def get_provisional_contender_floor(
    session: AsyncSession, *, bench_version: int | None = None
) -> float | None:
    """Return the finalized tenth-place score for provisional fast-lane admission.

    A provisional submission is only a likely top-ten contender when its first
    accepted score reaches the current finalized top ten.  Ranking the ten best
    provisional rows against one another is not sufficient: when the whole
    provisional pool is weak, that interpretation starves untouched submissions
    without advancing a plausible leaderboard contender.

    ``None`` means fewer than ten finalized owners exist in this benchmark era,
    so there is not yet a meaningful top-ten boundary and the bounded lane keeps
    its bootstrap behavior.
    """
    _, provisional = await get_score_priority_floors(
        session, bench_version=bench_version
    )
    return provisional


async def expire_overdue_tickets(session: AsyncSession, *, now: datetime) -> int:
    """Flip every overdue ``issued`` ticket to ``expired``; return the count.

    Frees the slots of validators that took a ticket and never scored in time,
    so another validator can pick the agent up. Runs inside the caller's
    transaction. Idempotent: a second call over the same window flips nothing.
    """
    overdue = (
        (
            await session.execute(
                select(ValidatorTicket).where(
                    ValidatorTicket.status == TicketStatus.ISSUED,
                    ValidatorTicket.deadline <= now,
                )
            )
        )
        .scalars()
        .all()
    )
    for ticket in overdue:
        ticket.status = TicketStatus.EXPIRED
        # Cooldown begins at the lease deadline, not whenever a later sweep
        # happens to notice it.
        ticket.retry_after = _as_utc(ticket.deadline) + RETRY_COOLDOWN
    return len(overdue)


async def issue_ticket(
    session: AsyncSession,
    *,
    validator_hotkey: str,
    now: datetime,
    ttl: timedelta,
    bench_version: int | None = 2,
    artifact_mode: Literal["legacy", "prefer_screened", "screened_only"] = "legacy",
    validator_running_benchmark: bool = False,
    submitted_at_or_after: datetime | None = None,
    fifo_start_at: datetime | None = None,
    completion_first: bool = False,
    slot_id: str = "slot-0",
) -> ValidatorTicket | None:
    """Issue a ticket to ``validator_hotkey`` for the next eligible agent.

    Sweeps overdue tickets first, then picks an ``evaluating`` agent that (a)
    has fewer than :data:`SCORING_QUORUM` live tickets and (b) this validator
    does not already hold a live or scored ticket for. Candidates in the
    bounded set whose first score can reach the finalized top ten comes first.
    Other provisional rows do not outrank untouched submissions merely because
    they already have a score. ``completion_first`` instead makes benchmark-era
    FIFO primary so the oldest submission reaches quorum before the next
    submission is opened. A
    2-of-3 submission that can no longer reach this era's emission set sorts
    behind every other candidate rather than being withheld, so it still
    finalizes once the queue drains. A prior expired row is reissued only after
    its cooldown and only once for the same benchmark version. Returns the
    ticket, or ``None`` when there is no work for this validator ("no job for
    you"). Runs inside the caller's transaction.
    """
    # No row exists to lock before a validator's first claim. Serialize that
    # gap explicitly on Postgres; the unique partial index remains the durable
    # backstop and SQLite test transactions are already single-writer.
    if session.get_bind().dialect.name == "postgresql":
        await session.execute(
            select(
                func.pg_advisory_xact_lock(
                    func.hashtextextended(f"{validator_hotkey}:{slot_id}", 0)
                )
            )
        )
    await expire_overdue_tickets(session, now=now)
    if bench_version is None:
        raise ValueError("benchmark version is required for ticket issuance")
    activated_rollout = await activated_rollout_for_version(
        session, bench_version=bench_version
    )
    if fifo_start_at is None:
        fifo_start_at = (
            activated_rollout.created_at
            if activated_rollout is not None
            else await session.scalar(
                select(BenchmarkRollout.created_at)
                .where(BenchmarkRollout.desired_version == bench_version)
                .order_by(BenchmarkRollout.created_at.desc())
                .limit(1)
            )
        )
    contract = benchmark_contract(bench_version)
    requires_screened = (
        contract.requires_screened_image or artifact_mode == "screened_only"
    )

    # A validator slot executes one benchmark at a time. Polling the same slot
    # again (including after a process restart) must resume that still-live
    # lease instead of allocating unrelated work and leaving it stranded.
    complete_screened_image = (
        Agent.screened_image_sha256.is_not(None)
        & Agent.screened_image_size_bytes.is_not(None)
        & Agent.screened_image_id.is_not(None)
        & Agent.screened_image_ref.is_not(None)
        & Agent.screened_image_upload_id.is_not(None)
        & Agent.screened_image_verified_at.is_not(None)
    )
    eligible_screened_image = complete_screened_image & (
        Agent.screening_policy_version >= contract.minimum_screening_policy_version
    )
    rollout_admitted = None
    if activated_rollout is not None:
        rollout_admitted = benchmark_admission_predicate(
            rollout=activated_rollout, bench_version=bench_version
        )
    existing_statement = (
        select(ValidatorTicket)
        .join(Agent, Agent.agent_id == ValidatorTicket.agent_id)
        .where(
            ValidatorTicket.validator_hotkey == validator_hotkey,
            ValidatorTicket.slot_id == slot_id,
            ValidatorTicket.bench_version == bench_version,
            ValidatorTicket.status == TicketStatus.ISSUED,
            ValidatorTicket.purpose == TicketPurpose.CANONICAL_QUORUM,
            ValidatorTicket.purpose_revision > 0,
            ValidatorTicket.deadline > now,
        )
        .order_by(ValidatorTicket.issued_at.asc(), ValidatorTicket.agent_id.asc())
        .limit(1)
        .with_for_update()
    )
    if requires_screened:
        existing_statement = existing_statement.where(eligible_screened_image)
    if rollout_admitted is not None:
        existing_statement = existing_statement.where(rollout_admitted)
    existing = await session.scalar(existing_statement)
    if existing is not None:
        return existing
    incompatible_existing = await session.scalar(
        select(ValidatorTicket)
        .where(
            ValidatorTicket.validator_hotkey == validator_hotkey,
            ValidatorTicket.slot_id == slot_id,
            ValidatorTicket.status == TicketStatus.ISSUED,
            ValidatorTicket.deadline > now,
        )
        .limit(1)
        .with_for_update()
    )
    if incompatible_existing is not None:
        if (
            incompatible_existing.purpose != TicketPurpose.CANONICAL_QUORUM
            or incompatible_existing.purpose_revision <= 0
        ):
            # Continual and deployment-transition leases own this slot until
            # their deadline. A canonical claim must neither serve nor cancel
            # work from another authorization lane.
            return None
        if validator_running_benchmark:
            # Never revoke work a fresh signed heartbeat says is active.
            return None
        # A validator may only resume a lease from the requested benchmark era
        # and artifact contract. Release an idle incompatible assignment so a
        # retired benchmark cannot leak into the active queue after activation.
        incompatible_existing.status = TicketStatus.EXPIRED
        incompatible_existing.deadline = now
        incompatible_existing.retry_after = now
        await session.flush()

    # Scoped to the era this ticket is for: a v2 fifth place says nothing about
    # whether a v4 two-score maximum is still in contention.
    score_continuation_floor, provisional_contender_floor = (
        (None, None)
        if completion_first
        else await get_score_priority_floors(session, bench_version=bench_version)
    )

    # Agents this validator must not receive right now: live/scored tickets,
    # same-version tickets cooling down after expiry, and same-version tickets
    # that already consumed the two-attempt budget. A benchmark-version bump
    # resets the budget so repaired scoring software can revisit the artifact.
    already_mine = select(ValidatorTicket.agent_id).where(
        ValidatorTicket.validator_hotkey == validator_hotkey,
        ValidatorTicket.bench_version == bench_version,
        (
            ValidatorTicket.status.in_(_LIVE_TICKET_STATUSES)
            | (
                (ValidatorTicket.status == TicketStatus.EXPIRED)
                & (
                    (ValidatorTicket.retry_after > now)
                    | (
                        ValidatorTicket.attempt_count
                        >= (
                            MAX_ATTEMPTS_PER_VERSION
                            + ValidatorTicket.manual_retry_grants
                            + ValidatorTicket.infra_retry_grants
                        )
                    )
                )
            )
        ),
    )
    had_prior_ticket = (
        select(ValidatorTicket.agent_id)
        .where(
            ValidatorTicket.agent_id == Agent.agent_id,
            ValidatorTicket.validator_hotkey == validator_hotkey,
        )
        .correlate(Agent)
        .exists()
    )
    live_assignment_count = (
        select(func.count())
        .where(
            ValidatorTicket.agent_id == Agent.agent_id,
            ValidatorTicket.bench_version == bench_version,
            ValidatorTicket.status == TicketStatus.ISSUED,
            ValidatorTicket.deadline > now,
        )
        .correlate(Agent)
        .scalar_subquery()
    )
    provisional_composite = func.coalesce(
        (
            select(func.avg(Score.composite))
            .where(
                Score.agent_id == Agent.agent_id,
                Score.bench_version == bench_version,
            )
            .correlate(Agent)
            .scalar_subquery()
        ),
        0.0,
    )
    recorded_score_count = (
        select(func.count(Score.validator_hotkey))
        .where(
            Score.agent_id == Agent.agent_id,
            Score.bench_version == bench_version,
        )
        .correlate(Agent)
        .scalar_subquery()
    )
    highest_recorded_score = func.coalesce(
        (
            select(func.max(Score.composite))
            .where(
                Score.agent_id == Agent.agent_id,
                Score.bench_version == bench_version,
            )
            .correlate(Agent)
            .scalar_subquery()
        ),
        0.0,
    )
    # A median-of-three cannot be bounded safely after one score. Once two
    # scores exist, their maximum is the best final median the third score can
    # produce, so a submission whose strict upper bound sits below this era's
    # finalized fifth place cannot reach the emission set. That earns it last
    # place in the queue, not removal: the third score still finalizes the
    # submission for the public record, and deferring rather than dropping it
    # means a later floor move (or a new benchmark era, where the old floor
    # never applied) cannot strand it at 2-of-3 forever. When the era has no
    # floor yet, every candidate shares lane 0 and ordering is unchanged.
    below_floor_lane = (
        case(
            (
                (recorded_score_count == SCORING_QUORUM - 1)
                & (highest_recorded_score < score_continuation_floor),
                1,
            ),
            else_=0,
        )
        if score_continuation_floor is not None
        else literal(0)
    )
    contender = aliased(Agent)
    contender_accepted_score_count = (
        select(func.count())
        .where(
            ValidatorTicket.agent_id == contender.agent_id,
            ValidatorTicket.bench_version == bench_version,
            ValidatorTicket.status == TicketStatus.SCORED,
        )
        .correlate(contender)
        .scalar_subquery()
    )
    contender_recorded_score_count = (
        select(func.count(Score.validator_hotkey))
        .where(
            Score.agent_id == contender.agent_id,
            Score.bench_version == bench_version,
        )
        .correlate(contender)
        .scalar_subquery()
    )
    contender_first_score = (
        select(Score.composite)
        .where(
            Score.agent_id == contender.agent_id,
            Score.bench_version == bench_version,
        )
        .order_by(Score.created_at.asc(), Score.validator_hotkey.asc())
        .limit(1)
        .correlate(contender)
        .scalar_subquery()
    )
    contender_provisional_composite = (
        select(func.avg(Score.composite))
        .where(
            Score.agent_id == contender.agent_id,
            Score.bench_version == bench_version,
        )
        .correlate(contender)
        .scalar_subquery()
    )
    contender_payment = aliased(EvaluationPayment)
    contender_owner = case(
        (
            contender_payment.miner_coldkey.is_not(None),
            literal("coldkey:") + contender_payment.miner_coldkey,
        ),
        else_=literal("hotkey:") + contender.miner_hotkey,
    )
    contender_per_miner = (
        select(
            contender.agent_id.label("agent_id"),
            contender.created_at.label("created_at"),
            contender_provisional_composite.label("provisional_composite"),
            func.row_number()
            .over(
                partition_by=contender_owner,
                order_by=(
                    contender_provisional_composite.desc(),
                    contender.created_at.asc(),
                    contender.agent_id.asc(),
                ),
            )
            .label("miner_rank"),
        )
        .outerjoin(
            contender_payment,
            contender_payment.agent_id == contender.agent_id,
        )
        .where(
            contender.status == AgentStatus.EVALUATING,
            contender.screening_policy_version >= SCREENING_POLICY_VERSION,
            contender_accepted_score_count.between(1, SCORING_QUORUM - 1),
            contender_recorded_score_count >= contender_accepted_score_count,
            (
                contender_first_score >= provisional_contender_floor
                if provisional_contender_floor is not None
                else literal(True)
            ),
        )
        .subquery()
    )
    top_provisional_contenders = (
        select(contender_per_miner.c.agent_id)
        .where(contender_per_miner.c.miner_rank == 1)
        .order_by(
            contender_per_miner.c.provisional_composite.desc(),
            contender_per_miner.c.created_at.asc(),
            contender_per_miner.c.agent_id.asc(),
        )
        .limit(PROVISIONAL_CONTENDER_LANE_SIZE)
    )
    contender_lane = case(
        (Agent.agent_id.in_(top_provisional_contenders), 0),
        else_=1,
    )
    contender_lane_score = case(
        (Agent.agent_id.in_(top_provisional_contenders), provisional_composite),
        else_=0.0,
    )
    overflow_two_score_lane = case(
        (recorded_score_count >= SCORING_QUORUM - 1, 1),
        else_=0,
    )
    # Lock one candidate Agent row before counting its tickets. The recount is a
    # separate statement after the lock is acquired, so under Postgres READ
    # COMMITTED it sees any ticket committed by the previous lock holder.
    # SKIP LOCKED lets unrelated agents continue allocating concurrently.
    skipped: list[UUID] = []
    while True:
        candidate = select(Agent.agent_id).where(
            Agent.status == AgentStatus.EVALUATING,
            Agent.screening_policy_version >= SCREENING_POLICY_VERSION,
        )
        if not completion_first:
            candidate = candidate.where(Agent.agent_id.not_in(already_mine))
        if bench_version != 2:
            versioned_dataset = (
                select(BenchmarkDataset.agent_id)
                .where(
                    BenchmarkDataset.agent_id == Agent.agent_id,
                    BenchmarkDataset.bench_version == bench_version,
                )
                .exists()
            )
            candidate = candidate.where(versioned_dataset)
        if requires_screened:
            candidate = candidate.where(eligible_screened_image)
        if rollout_admitted is not None:
            candidate = candidate.where(rollout_admitted)
        if submitted_at_or_after is not None:
            candidate = candidate.where(Agent.created_at >= submitted_at_or_after)
        if skipped:
            candidate = candidate.where(Agent.agent_id.not_in(skipped))
        fifo_age = (
            case(
                (Agent.created_at < fifo_start_at, fifo_start_at),
                else_=Agent.created_at,
            )
            if fifo_start_at is not None
            else Agent.created_at
        )
        queue_order = (
            (
                # Keep the fresh-submission lane independent of the ordinary
                # queue's contender, coverage, artifact, and continuation-floor
                # priorities. Age is the contract; UUID is only a stable tie.
                fifo_age.asc(),
                Agent.agent_id.asc(),
            )
            if completion_first
            else (
                below_floor_lane.asc(),
                case(
                    (complete_screened_image, 0),
                    else_=(0 if artifact_mode == "legacy" else 1),
                ).asc(),
                contender_lane.asc(),
                contender_lane_score.desc(),
                # Keep the existing bounded-contender guarantee: a two-score
                # row outside the top contender set must not turn the whole
                # backlog into an unbounded completion lane.
                overflow_two_score_lane.asc(),
                live_assignment_count.asc(),
                had_prior_ticket.asc(),
                fifo_age.asc(),
                Agent.agent_id.asc(),
            )
        )
        candidate = (
            # The ordinary queue first advances the bounded set of strongest
            # scored provisional contenders, one best submission per miner. A
            # stronger 1-of-3 candidate can therefore receive its second score
            # before a weaker 2-of-3 candidate receives its third. The remaining
            # queue gives untouched work a coverage opportunity before weak
            # provisional rows. The fresh lane uses queue_order's FIFO-first
            # alternative.
            candidate.order_by(*queue_order)
            .limit(1)
            .with_for_update(of=Agent, skip_locked=not completion_first)
        )
        agent_id = (await session.execute(candidate)).scalar_one_or_none()
        if agent_id is None:
            return None

        # One paid owner may have many generations, but only one generation may
        # occupy validator capacity at a time. Serialize by the immutable
        # payment-time coldkey (legacy rows fall back to hotkey), then re-check
        # for an issued sibling after taking the lock. Accepted scores are
        # history, not occupied capacity, but they keep the first progressing
        # generation selected until it settles. The post-lock checks close the
        # race where two platform replicas select different agents for the same
        # owner before either ticket exists.
        owner_row = (
            await session.execute(
                select(Agent.miner_hotkey, EvaluationPayment.miner_coldkey)
                .outerjoin(
                    EvaluationPayment,
                    EvaluationPayment.agent_id == Agent.agent_id,
                )
                .where(Agent.agent_id == agent_id)
            )
        ).one()
        owner_hotkey, owner_coldkey = owner_row
        linked_coldkeys = {
            coldkey
            for coldkey in (
                await session.scalars(
                    select(EvaluationPayment.miner_coldkey)
                    .where(
                        EvaluationPayment.miner_hotkey == owner_hotkey,
                        EvaluationPayment.miner_coldkey.is_not(None),
                    )
                    .distinct()
                )
            ).all()
            if coldkey is not None
        }
        if owner_coldkey is not None:
            linked_coldkeys.add(owner_coldkey)
        linked_hotkeys = {owner_hotkey}
        if linked_coldkeys:
            linked_hotkeys.update(
                (
                    await session.scalars(
                        select(EvaluationPayment.miner_hotkey)
                        .where(EvaluationPayment.miner_coldkey.in_(linked_coldkeys))
                        .distinct()
                    )
                ).all()
            )
        if session.get_bind().dialect.name == "postgresql":
            # A legacy row inherits every payment-time coldkey previously
            # observed for its hotkey. Lock those identities in sorted order,
            # so it also serializes with a paid generation submitted after a
            # hotkey rotation. A truly unlinked legacy row falls back to its
            # hotkey. The canonical ordering prevents multi-key deadlocks.
            owner_lock_keys = (
                [f"coldkey:{coldkey}" for coldkey in sorted(linked_coldkeys)]
                if linked_coldkeys
                else [f"hotkey:{owner_hotkey}"]
            )
            for owner_lock_key in owner_lock_keys:
                await session.execute(
                    select(
                        func.pg_advisory_xact_lock(
                            func.hashtextextended(owner_lock_key, 0)
                        )
                    )
                )
        sibling_agent = aliased(Agent)
        sibling_payment = aliased(EvaluationPayment)
        same_owner = (
            or_(
                sibling_payment.miner_coldkey.in_(linked_coldkeys),
                and_(
                    sibling_payment.miner_coldkey.is_(None),
                    sibling_agent.miner_hotkey.in_(linked_hotkeys),
                ),
            )
            if linked_coldkeys
            else and_(
                sibling_payment.miner_coldkey.is_(None),
                sibling_agent.miner_hotkey == owner_hotkey,
            )
        )
        live_sibling_count = await session.scalar(
            select(func.count())
            .select_from(ValidatorTicket)
            .join(sibling_agent, sibling_agent.agent_id == ValidatorTicket.agent_id)
            .outerjoin(
                sibling_payment,
                sibling_payment.agent_id == sibling_agent.agent_id,
            )
            .where(
                sibling_agent.agent_id != agent_id,
                ValidatorTicket.status == TicketStatus.ISSUED,
                ValidatorTicket.deadline > now,
                same_owner,
            )
        )
        if (live_sibling_count or 0) > 0:
            skipped.append(agent_id)
            continue

        # Keep one current-era generation selected across the gaps between its
        # leases. Otherwise a validator that already scored the selected row
        # can open a sibling after the last lease becomes SCORED, and that new
        # lease diverts every eligible validator away from finishing the first
        # generation. Historical overlaps converge deterministically on the
        # generation whose accepted/live progress began first. Expired-only
        # attempts do not pin an owner, so failed work can still drain.
        owner_progress_started_at = (
            select(func.min(ValidatorTicket.issued_at))
            .where(
                ValidatorTicket.agent_id == sibling_agent.agent_id,
                ValidatorTicket.bench_version == bench_version,
                (
                    (ValidatorTicket.status == TicketStatus.SCORED)
                    | (
                        (ValidatorTicket.status == TicketStatus.ISSUED)
                        & (ValidatorTicket.deadline > now)
                    )
                ),
            )
            .correlate(sibling_agent)
            .scalar_subquery()
        )
        owner_first_score = (
            select(Score.composite)
            .where(
                Score.agent_id == sibling_agent.agent_id,
                Score.bench_version == bench_version,
            )
            .order_by(Score.created_at.asc(), Score.validator_hotkey.asc())
            .limit(1)
            .correlate(sibling_agent)
            .scalar_subquery()
        )
        selected_owner_agent_id = await session.scalar(
            select(sibling_agent.agent_id)
            .outerjoin(
                sibling_payment,
                sibling_payment.agent_id == sibling_agent.agent_id,
            )
            .where(
                sibling_agent.status == AgentStatus.EVALUATING,
                same_owner,
                owner_progress_started_at.is_not(None),
                (
                    owner_first_score >= provisional_contender_floor
                    if provisional_contender_floor is not None
                    else literal(True)
                ),
                (
                    benchmark_admission_predicate(
                        rollout=activated_rollout,
                        bench_version=bench_version,
                        agent=sibling_agent,
                    )
                    if activated_rollout is not None
                    else literal(True)
                ),
            )
            .order_by(
                owner_progress_started_at.asc(),
                sibling_agent.created_at.asc(),
                sibling_agent.agent_id.asc(),
            )
            .limit(1)
        )
        if selected_owner_agent_id is not None and selected_owner_agent_id != agent_id:
            skipped.append(agent_id)
            continue
        occupied = await session.scalar(
            select(func.count()).where(
                ValidatorTicket.agent_id == agent_id,
                ValidatorTicket.bench_version == bench_version,
                ValidatorTicket.status.in_(_LIVE_TICKET_STATUSES),
            )
        )
        if (occupied or 0) >= SCORING_QUORUM:
            skipped.append(agent_id)
            continue
        if completion_first:
            # Completion-first admission is global, not per validator slot.
            # Every slot waits on the same FIFO head. Once the row lock is
            # acquired, re-check this validator against fresh committed state.
            # A sibling slot waits while this validator owns the head; a
            # validator that can no longer score the head advances to the next
            # FIFO candidate instead of idling behind impossible work.
            same_validator_blocking_status = await session.scalar(
                select(ValidatorTicket.status)
                .where(
                    ValidatorTicket.agent_id == agent_id,
                    ValidatorTicket.validator_hotkey == validator_hotkey,
                    ValidatorTicket.bench_version == bench_version,
                    (
                        ValidatorTicket.status.in_(_LIVE_TICKET_STATUSES)
                        | (
                            (ValidatorTicket.status == TicketStatus.EXPIRED)
                            & (
                                (ValidatorTicket.retry_after > now)
                                | (
                                    ValidatorTicket.attempt_count
                                    >= (
                                        MAX_ATTEMPTS_PER_VERSION
                                        + ValidatorTicket.manual_retry_grants
                                        + ValidatorTicket.infra_retry_grants
                                    )
                                )
                            )
                        )
                    ),
                )
                .limit(1)
            )
            if same_validator_blocking_status == TicketStatus.ISSUED:
                # A sibling slot must not advance while this validator already
                # owns the FIFO head. Let another validator fill the remaining
                # quorum slots first.
                return None
            if same_validator_blocking_status is not None:
                # This validator cannot contribute another score to the FIFO
                # head (it already scored it, is cooling down, or exhausted its
                # retry budget). Keeping it parked here can idle the entire
                # fleet when every remaining scorer is similarly ineligible.
                # Preserve FIFO among work this validator can actually claim.
                skipped.append(agent_id)
                continue
        break

    ticket = await session.get(
        ValidatorTicket, (agent_id, bench_version, validator_hotkey)
    )
    if ticket is None:
        ticket = ValidatorTicket(
            agent_id=agent_id,
            validator_hotkey=validator_hotkey,
            slot_id=slot_id,
            status=TicketStatus.ISSUED,
            purpose=TicketPurpose.CANONICAL_QUORUM,
            purpose_revision=1,
            issued_at=now,
            deadline=now + ttl,
            bench_version=bench_version,
            attempt_count=1,
            manual_retry_grants=0,
            retry_after=None,
        )
        session.add(ticket)
    else:
        # The composite PK preserves one validator slot per agent. Reuse the
        # expired row with a fresh lease rather than inserting a duplicate.
        ticket.status = TicketStatus.ISSUED
        ticket.purpose = TicketPurpose.CANONICAL_QUORUM
        ticket.purpose_revision += 1
        ticket.legacy_completion_allowed = False
        ticket.slot_id = slot_id
        ticket.issued_at = now
        ticket.deadline = now + ttl
        ticket.attempt_count += 1
        ticket.retry_after = None
    await session.flush()
    return ticket


async def issue_confirmation_ticket(
    session: AsyncSession,
    *,
    agent_id: UUID,
    validator_hotkey: str,
    now: datetime,
    ttl: timedelta,
    bench_version: int,
    seed: int | None = None,
    dataset_sha256: str | None = None,
) -> ValidatorTicket | None:
    """Reissue this validator's existing quorum slot for top-five maintenance.

    The caller has already proven that ``agent_id`` is the one bounded KOTH
    confirmation target.  Reusing the existing composite-key row keeps score
    submission on the dedicated append-only endpoint. A validator with unrelated
    live work receives no confirmation
    ticket, so this maintenance run cannot interrupt queue scoring.
    """
    if session.get_bind().dialect.name == "postgresql":
        await session.execute(
            select(
                func.pg_advisory_xact_lock(func.hashtextextended(validator_hotkey, 0))
            )
        )
    await expire_overdue_tickets(session, now=now)
    existing_live = await session.scalar(
        select(ValidatorTicket)
        .where(
            ValidatorTicket.validator_hotkey == validator_hotkey,
            ValidatorTicket.status == TicketStatus.ISSUED,
            ValidatorTicket.deadline > now,
        )
        .limit(1)
        .with_for_update()
    )
    if existing_live is not None:
        return (
            existing_live
            if existing_live.agent_id == agent_id
            and existing_live.bench_version == bench_version
            and existing_live.purpose == TicketPurpose.CONTINUAL_RETEST
            and existing_live.purpose_revision > 0
            and (seed is None or existing_live.seed == seed)
            and (
                dataset_sha256 is None or existing_live.dataset_sha256 == dataset_sha256
            )
            else None
        )

    agent = await session.scalar(
        select(Agent).where(Agent.agent_id == agent_id).with_for_update()
    )
    if agent is None or agent.status not in {AgentStatus.SCORED, AgentStatus.LIVE}:
        return None
    # Confirmation evidence is append-only and never changes the canonical k=3
    # Score rows, so any permitted validator may contribute the shared wave.
    # Restricting this to the original three scorers strands healthy validators
    # and can make a five-member wave impossible to finish during fleet churn.
    latest_retest = await get_latest_score_retest_event(
        session,
        agent_id=agent_id,
        validator_hotkey=validator_hotkey,
    )
    if (
        latest_retest is not None
        and latest_retest.event == EVENT_SCORE_RETEST_REQUESTED
    ):
        # An operator-authorized canonical replacement owns this validator/agent
        # lifecycle until it is completed or released. Continual maintenance
        # must never repurpose its expired mutable row.
        return None

    ticket = await session.get(
        ValidatorTicket, (agent_id, bench_version, validator_hotkey)
    )
    if ticket is not None and ticket.retry_after is not None:
        retry_after = ticket.retry_after
        if retry_after.tzinfo is None:
            retry_after = retry_after.replace(tzinfo=UTC)
        if retry_after > now:
            return None
    if ticket is None:
        ticket = ValidatorTicket(
            agent_id=agent_id,
            validator_hotkey=validator_hotkey,
            status=TicketStatus.ISSUED,
            purpose=TicketPurpose.CONTINUAL_RETEST,
            purpose_revision=1,
            issued_at=now,
            deadline=now + ttl,
            bench_version=bench_version,
            seed=seed,
            dataset_sha256=dataset_sha256,
            attempt_count=1,
            manual_retry_grants=0,
            retry_after=None,
        )
        session.add(ticket)
    else:
        same_version = ticket.bench_version == bench_version
        ticket.status = TicketStatus.ISSUED
        ticket.purpose = TicketPurpose.CONTINUAL_RETEST
        ticket.purpose_revision += 1
        ticket.legacy_completion_allowed = False
        ticket.issued_at = now
        ticket.deadline = now + ttl
        ticket.bench_version = bench_version
        ticket.seed = seed
        ticket.dataset_sha256 = dataset_sha256
        ticket.seed_block = None
        ticket.seed_block_hash = None
        ticket.attempt_count = ticket.attempt_count + 1 if same_version else 1
        ticket.manual_retry_grants = ticket.manual_retry_grants if same_version else 0
        ticket.retry_after = None
    await session.flush()
    return ticket


async def get_open_ticket(
    session: AsyncSession,
    *,
    agent_id: UUID,
    validator_hotkey: str,
    now: datetime,
    deadline: datetime,
    bench_version: int | None = 2,
    slot_id: str | None = None,
    for_update: bool = False,
) -> ValidatorTicket | None:
    """Return the validator's live ticket matching the signed lease.

    ``bench_version=None`` is reserved for signed heartbeat progress, where the
    exact lease deadline identifies work across benchmark versions. The
    one-issued-ticket-per-validator index keeps that cross-version lookup
    unambiguous.
    """
    statement = select(ValidatorTicket).where(
        ValidatorTicket.agent_id == agent_id,
        ValidatorTicket.validator_hotkey == validator_hotkey,
        ValidatorTicket.status == TicketStatus.ISSUED,
    )
    if bench_version is not None:
        statement = statement.where(ValidatorTicket.bench_version == bench_version)
    if slot_id is not None:
        statement = statement.where(ValidatorTicket.slot_id == slot_id)
    if for_update:
        statement = statement.with_for_update()
    ticket = await session.scalar(statement)
    if (
        ticket is None
        or _as_utc(ticket.deadline) <= now
        or _as_utc(ticket.deadline) != _as_utc(deadline)
    ):
        return None
    return ticket


async def mark_ticket_scored(
    session: AsyncSession,
    *,
    agent_id: UUID,
    validator_hotkey: str,
    bench_version: int = 2,
) -> None:
    """Mark the validator's ticket for the agent ``scored`` (slot spent). No-op
    if there is no ticket row (e.g. a legacy score predating ticketing)."""
    ticket = await session.get(
        ValidatorTicket, (agent_id, bench_version, validator_hotkey)
    )
    if ticket is not None:
        ticket.status = TicketStatus.SCORED
