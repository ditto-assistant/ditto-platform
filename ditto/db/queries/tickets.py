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
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import case, func, select

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.api_models.ticket_status import TicketStatus
from ditto.db.models import Agent, Score, ValidatorTicket
from ditto.db.queries.scores import SCORING_QUORUM

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
    bench_version: int = 2,
) -> ValidatorTicket | None:
    """Issue a ticket to ``validator_hotkey`` for the next eligible agent.

    Sweeps overdue tickets first, then picks an ``evaluating`` agent that (a)
    has fewer than :data:`SCORING_QUORUM` live tickets and (b) this validator
    does not already hold a live or scored ticket for. Candidates with the
    least total coverage (accepted scores plus live assignments) come first,
    then never-attempted work, then submission age. A prior expired row is
    reissued only after its cooldown and only once for the same benchmark
    version. Returns the ticket, or ``None`` when there is no work for this
    validator ("no job for you"). Runs inside the caller's transaction.
    """
    # No row exists to lock before a validator's first claim. Serialize that
    # gap explicitly on Postgres; the unique partial index remains the durable
    # backstop and SQLite test transactions are already single-writer.
    if session.get_bind().dialect.name == "postgresql":
        await session.execute(
            select(
                func.pg_advisory_xact_lock(func.hashtextextended(validator_hotkey, 0))
            )
        )
    await expire_overdue_tickets(session, now=now)

    # A validator executes one benchmark at a time. Polling again (including
    # after a process restart) must resume that still-live lease instead of
    # allocating unrelated work and leaving the first ticket stranded.
    existing = await session.scalar(
        select(ValidatorTicket)
        .where(
            ValidatorTicket.validator_hotkey == validator_hotkey,
            ValidatorTicket.status == TicketStatus.ISSUED,
            ValidatorTicket.deadline > now,
        )
        .order_by(ValidatorTicket.issued_at.asc(), ValidatorTicket.agent_id.asc())
        .limit(1)
        .with_for_update()
    )
    if existing is not None:
        return existing

    # Agents this validator must not receive right now: live/scored tickets,
    # same-version tickets cooling down after expiry, and same-version tickets
    # that already consumed the two-attempt budget. A benchmark-version bump
    # resets the budget so repaired scoring software can revisit the artifact.
    already_mine = select(ValidatorTicket.agent_id).where(
        ValidatorTicket.validator_hotkey == validator_hotkey,
        (
            ValidatorTicket.status.in_(_LIVE_TICKET_STATUSES)
            | (
                (ValidatorTicket.status == TicketStatus.EXPIRED)
                & (ValidatorTicket.bench_version == bench_version)
                & (
                    (ValidatorTicket.retry_after > now)
                    | (ValidatorTicket.attempt_count >= MAX_ATTEMPTS_PER_VERSION)
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
    accepted_score_count = (
        select(func.count())
        .where(
            ValidatorTicket.agent_id == Agent.agent_id,
            ValidatorTicket.status == TicketStatus.SCORED,
        )
        .correlate(Agent)
        .scalar_subquery()
    )
    live_assignment_count = (
        select(func.count())
        .where(
            ValidatorTicket.agent_id == Agent.agent_id,
            ValidatorTicket.status == TicketStatus.ISSUED,
            ValidatorTicket.deadline > now,
        )
        .correlate(Agent)
        .scalar_subquery()
    )
    total_coverage = accepted_score_count + live_assignment_count
    provisional_composite = func.coalesce(
        (
            select(func.avg(Score.composite))
            .where(Score.agent_id == Agent.agent_id)
            .correlate(Agent)
            .scalar_subquery()
        ),
        0.0,
    )
    completion_lane_score = case(
        (
            accepted_score_count >= SCORING_QUORUM - 1,
            provisional_composite,
        ),
        else_=0.0,
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
            Agent.agent_id.not_in(already_mine),
        )
        if skipped:
            candidate = candidate.where(Agent.agent_id.not_in(skipped))
        candidate = (
            # Round-robin the scoreable backlog. A live evaluator counts as one
            # unit of coverage just like an accepted score, so an agent cannot
            # jump from zero coverage to three concurrent validators while
            # another eligible agent remains uncovered. Within one coverage
            # round, never-attempted work precedes this validator's cooled-down
            # retry. Once candidates reach the two-score completion lane, prefer
            # the highest provisional composite so the likely emission winner
            # reaches quorum first. Submission age and UUID remain stable ties.
            candidate.order_by(
                total_coverage.asc(),
                had_prior_ticket.asc(),
                completion_lane_score.desc(),
                Agent.created_at.asc(),
                Agent.agent_id.asc(),
            )
            .limit(1)
            .with_for_update(of=Agent, skip_locked=True)
        )
        agent_id = (await session.execute(candidate)).scalar_one_or_none()
        if agent_id is None:
            return None
        occupied = await session.scalar(
            select(func.count()).where(
                ValidatorTicket.agent_id == agent_id,
                ValidatorTicket.status.in_(_LIVE_TICKET_STATUSES),
            )
        )
        if (occupied or 0) < SCORING_QUORUM:
            break
        skipped.append(agent_id)

    ticket = await session.get(ValidatorTicket, (agent_id, validator_hotkey))
    if ticket is None:
        ticket = ValidatorTicket(
            agent_id=agent_id,
            validator_hotkey=validator_hotkey,
            status=TicketStatus.ISSUED,
            issued_at=now,
            deadline=now + ttl,
            bench_version=bench_version,
            attempt_count=1,
            retry_after=None,
        )
        session.add(ticket)
    else:
        # The composite PK preserves one validator slot per agent. Reuse the
        # expired row with a fresh lease rather than inserting a duplicate.
        same_version = ticket.bench_version == bench_version
        ticket.status = TicketStatus.ISSUED
        ticket.issued_at = now
        ticket.deadline = now + ttl
        ticket.bench_version = bench_version
        ticket.attempt_count = ticket.attempt_count + 1 if same_version else 1
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
    for_update: bool = False,
) -> ValidatorTicket | None:
    """Return the validator's live (``issued``, not-yet-past-deadline) ticket for
    the agent, or ``None`` if it has none, it is already spent, or it expired."""
    statement = select(ValidatorTicket).where(
        ValidatorTicket.agent_id == agent_id,
        ValidatorTicket.validator_hotkey == validator_hotkey,
    )
    if for_update:
        statement = statement.with_for_update()
    ticket = await session.scalar(statement)
    if (
        ticket is None
        or ticket.status != TicketStatus.ISSUED
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
) -> None:
    """Mark the validator's ticket for the agent ``scored`` (slot spent). No-op
    if there is no ticket row (e.g. a legacy score predating ticketing)."""
    ticket = await session.get(ValidatorTicket, (agent_id, validator_hotkey))
    if ticket is not None:
        ticket.status = TicketStatus.SCORED
