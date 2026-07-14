"""Mutations + reads against the ``validator_tickets`` table (the k=3 pool).

A submission (agent) is scored by at most :data:`SCORING_QUORUM` validators. A
ticket is issued on demand to a validator that does not already hold one for the
agent, expires if unscored by its deadline (freeing the slot), and is marked
``scored`` when the validator posts a valid score in time.

Issuance locks the candidate agent row and then recounts its occupied slots in a
fresh statement. Concurrent platform replicas therefore serialize allocation
for a given agent and cannot over-issue a fourth slot. The ``(agent_id,
validator_hotkey)`` primary key separately guarantees a validator can hold only
one ticket per agent, so no single validator can take two slots.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import func, select, update

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.api_models.ticket_status import TicketStatus
from ditto.db.models import Agent, ValidatorTicket
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


async def expire_overdue_tickets(session: AsyncSession, *, now: datetime) -> int:
    """Flip every overdue ``issued`` ticket to ``expired``; return the count.

    Frees the slots of validators that took a ticket and never scored in time,
    so another validator can pick the agent up. Runs inside the caller's
    transaction. Idempotent: a second call over the same window flips nothing.
    """
    overdue = (
        await session.execute(
            select(ValidatorTicket.agent_id, ValidatorTicket.validator_hotkey).where(
                ValidatorTicket.status == TicketStatus.ISSUED,
                ValidatorTicket.deadline < now,
            )
        )
    ).all()
    if overdue:
        await session.execute(
            update(ValidatorTicket)
            .where(
                ValidatorTicket.status == TicketStatus.ISSUED,
                ValidatorTicket.deadline < now,
            )
            .values(status=TicketStatus.EXPIRED)
        )
    return len(overdue)


async def issue_ticket(
    session: AsyncSession,
    *,
    validator_hotkey: str,
    now: datetime,
    ttl: timedelta,
) -> ValidatorTicket | None:
    """Issue a ticket to ``validator_hotkey`` for the next eligible agent.

    Sweeps overdue tickets first, then picks the oldest ``evaluating`` agent
    that (a) has fewer than :data:`SCORING_QUORUM` live tickets and (b) this
    validator does not already hold a live or scored ticket for. A prior expired
    row is reissued with a ``now + ttl`` deadline. Returns the ticket, or ``None``
    when there is no work for this validator ("no job for you"). Runs inside the
    caller's transaction.
    """
    await expire_overdue_tickets(session, now=now)

    # Agents this validator already has a live or scored ticket for. Expired
    # tickets are intentionally excluded: validators are stateless workers, so
    # a repaired/restarted validator must be able to retry unfinished work.
    already_mine = select(ValidatorTicket.agent_id).where(
        ValidatorTicket.validator_hotkey == validator_hotkey,
        ValidatorTicket.status.in_(_LIVE_TICKET_STATUSES),
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
            candidate.order_by(Agent.created_at.asc())
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
        )
        session.add(ticket)
    else:
        # The composite PK preserves one validator slot per agent. Reuse the
        # expired row with a fresh lease rather than inserting a duplicate.
        ticket.status = TicketStatus.ISSUED
        ticket.issued_at = now
        ticket.deadline = now + ttl
    await session.flush()
    return ticket


async def get_open_ticket(
    session: AsyncSession,
    *,
    agent_id: UUID,
    validator_hotkey: str,
    now: datetime,
    deadline: datetime,
) -> ValidatorTicket | None:
    """Return the validator's live (``issued``, not-yet-past-deadline) ticket for
    the agent, or ``None`` if it has none, it is already spent, or it expired."""
    ticket = await session.get(ValidatorTicket, (agent_id, validator_hotkey))
    if (
        ticket is None
        or ticket.status != TicketStatus.ISSUED
        or _as_utc(ticket.deadline) < now
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
