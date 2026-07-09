"""Mutations + reads against the ``validator_tickets`` table (the k=3 pool).

A submission (agent) is scored by at most :data:`SCORING_QUORUM` validators. A
ticket is issued on demand to a validator that does not already hold one for the
agent, expires if unscored by its deadline (freeing the slot), and is marked
``scored`` when the validator posts a valid score in time.

The per-agent cap is enforced by a count-then-insert, which is best-effort under
concurrency: a rare race could seat a fourth validator, but that is harmless
because finalization triggers at exactly the quorum (the extra score simply
joins the median pool and is never decisive). The ``(agent_id,
validator_hotkey)`` primary key still hard-guarantees a validator can hold only
one ticket per agent, so no single validator can take two slots.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import func, select, update

from ditto.api_models.agent_status import AgentStatus
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
    result = await session.execute(
        update(ValidatorTicket)
        .where(
            ValidatorTicket.status == TicketStatus.ISSUED,
            ValidatorTicket.deadline < now,
        )
        .values(status=TicketStatus.EXPIRED)
    )
    return result.rowcount or 0


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
    validator does not already hold any ticket for, and seats a fresh ``issued``
    ticket with a ``now + ttl`` deadline. Returns the ticket, or ``None`` when
    there is no work for this validator ("no job for you"). Runs inside the
    caller's transaction.
    """
    await expire_overdue_tickets(session, now=now)

    # Live ticket count per agent (issued + scored; expired freed their slot).
    live_counts = (
        select(
            ValidatorTicket.agent_id.label("agent_id"),
            func.count().label("n"),
        )
        .where(ValidatorTicket.status.in_(_LIVE_TICKET_STATUSES))
        .group_by(ValidatorTicket.agent_id)
        .subquery()
    )
    # Agents this validator already has any ticket for (never re-seat it).
    already_mine = select(ValidatorTicket.agent_id).where(
        ValidatorTicket.validator_hotkey == validator_hotkey
    )
    candidate = (
        select(Agent.agent_id)
        .outerjoin(live_counts, live_counts.c.agent_id == Agent.agent_id)
        .where(
            Agent.status == AgentStatus.EVALUATING,
            func.coalesce(live_counts.c.n, 0) < SCORING_QUORUM,
            Agent.agent_id.not_in(already_mine),
        )
        .order_by(Agent.created_at.asc())
        .limit(1)
    )
    agent_id = (await session.execute(candidate)).scalar_one_or_none()
    if agent_id is None:
        return None

    ticket = ValidatorTicket(
        agent_id=agent_id,
        validator_hotkey=validator_hotkey,
        status=TicketStatus.ISSUED,
        issued_at=now,
        deadline=now + ttl,
    )
    session.add(ticket)
    await session.flush()
    return ticket


async def get_open_ticket(
    session: AsyncSession,
    *,
    agent_id: UUID,
    validator_hotkey: str,
    now: datetime,
) -> ValidatorTicket | None:
    """Return the validator's live (``issued``, not-yet-past-deadline) ticket for
    the agent, or ``None`` if it has none, it is already spent, or it expired."""
    ticket = await session.get(ValidatorTicket, (agent_id, validator_hotkey))
    if (
        ticket is None
        or ticket.status != TicketStatus.ISSUED
        or _as_utc(ticket.deadline) < now
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
