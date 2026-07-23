"""Serialized, append-only score re-tests for one validator.

The score stays canonical while an operator request waits in the audit log.
Only the queue head owns an issued ticket; completion or release promotes the
next compatible request atomically in the same transaction.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.ticket_status import TicketPurpose, TicketStatus
from ditto.db.models import Agent, Score, ScoreAuditEntry, ValidatorTicket
from ditto.db.queries.audit import (
    EVENT_SCORE_INVALIDATED,
    EVENT_SCORE_RETEST_QUEUED,
    EVENT_SCORE_RETEST_RELEASED,
    EVENT_SCORE_RETEST_REQUESTED,
    SCORE_RETEST_EVENTS,
    append_audit_entry,
)

REPLACEMENT_TICKET_TTL = timedelta(minutes=90)
_FINALIZED_STATUSES = (AgentStatus.SCORED, AgentStatus.LIVE)


async def lock_validator(session: AsyncSession, validator_hotkey: str) -> None:
    """Serialize ticket ownership changes for one validator."""
    if session.get_bind().dialect.name == "postgresql":
        await session.execute(
            select(
                func.pg_advisory_xact_lock(func.hashtextextended(validator_hotkey, 0))
            )
        )


async def latest_retest_events_for_validator(
    session: AsyncSession, *, validator_hotkey: str
) -> dict[UUID, ScoreAuditEntry]:
    """Return the latest lifecycle entry for every queued/recent agent."""
    entries = list(
        (
            await session.scalars(
                select(ScoreAuditEntry)
                .where(
                    ScoreAuditEntry.validator_hotkey == validator_hotkey,
                    ScoreAuditEntry.event.in_(SCORE_RETEST_EVENTS),
                )
                .order_by(ScoreAuditEntry.seq.asc())
            )
        ).all()
    )
    return {entry.agent_id: entry for entry in entries}


async def score_retest_queue_positions(
    session: AsyncSession, *, validator_hotkey: str
) -> dict[UUID, int]:
    latest = await latest_retest_events_for_validator(
        session, validator_hotkey=validator_hotkey
    )
    queued = sorted(
        (
            entry
            for entry in latest.values()
            if entry.event == EVENT_SCORE_RETEST_QUEUED
        ),
        key=lambda entry: entry.seq,
    )
    return {entry.agent_id: index for index, entry in enumerate(queued, start=1)}


async def _close_unserviceable(
    session: AsyncSession,
    *,
    entry: ScoreAuditEntry,
    now: datetime,
    reason: str,
) -> None:
    await append_audit_entry(
        session,
        agent_id=entry.agent_id,
        validator_hotkey=entry.validator_hotkey,
        event=EVENT_SCORE_RETEST_RELEASED,
        payload={
            "request_id": entry.payload.get("request_id"),
            "retest_request_id": entry.payload.get("request_id"),
            "actor": "platform:score-retest-queue",
            "reason": reason,
            "bench_version": entry.payload.get("bench_version"),
            "preserved_run_id": entry.payload.get("run_id"),
            "automatic": True,
        },
        recorded_at=now,
    )


async def activate_next_score_retest(
    session: AsyncSession,
    *,
    validator_hotkey: str,
    now: datetime,
    supports_version: Callable[[int], bool],
    validator_running_benchmark: bool = False,
    slot_id: str = "slot-0",
) -> ValidatorTicket | None:
    """Resume the active re-test or promote the oldest runnable queued item.

    Must be called inside a transaction. Stale requests close append-only and
    never mutate the accepted score. An unrelated live assignment keeps all
    queued requests waiting.
    """
    await lock_validator(session, validator_hotkey)
    latest = await latest_retest_events_for_validator(
        session, validator_hotkey=validator_hotkey
    )

    issued_rows = list(
        (
            await session.scalars(
                select(ValidatorTicket)
                .where(
                    ValidatorTicket.validator_hotkey == validator_hotkey,
                    ValidatorTicket.status == TicketStatus.ISSUED,
                )
                .with_for_update()
            )
        ).all()
    )
    issued = next(
        (
            ticket
            for ticket in issued_rows
            if ticket.purpose == TicketPurpose.CANONICAL_QUORUM
            and ticket.purpose_revision > 0
            if (lifecycle := latest.get(ticket.agent_id)) is not None
            and lifecycle.event == EVENT_SCORE_RETEST_REQUESTED
        ),
        None,
    )
    # Retests remain serialized behind every live ticket for this validator.
    # Parallel ordinary capacity must not let a replacement jump the existing
    # recovery queue or displace the public canonical score early.
    if issued_rows and issued is None:
        return None
    if issued is not None:
        if issued.slot_id != slot_id:
            return None
        lifecycle = latest.get(issued.agent_id)
        if (
            lifecycle is not None
            and lifecycle.event == EVENT_SCORE_RETEST_REQUESTED
            and int(lifecycle.payload.get("bench_version", -1)) == issued.bench_version
        ):
            deadline = issued.deadline
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=UTC)
            if deadline > now and supports_version(issued.bench_version):
                return issued
            if validator_running_benchmark and deadline > now:
                return None
            issued.status = TicketStatus.SCORED
            issued.retry_after = None
            await _close_unserviceable(
                session,
                entry=lifecycle,
                now=now,
                reason=(
                    "replacement ticket expired before completion"
                    if deadline <= now
                    else "validator no longer advertises this benchmark version"
                ),
            )
            await session.flush()
        else:
            return None

    queued = sorted(
        (
            entry
            for entry in latest.values()
            if entry.event == EVENT_SCORE_RETEST_QUEUED
        ),
        key=lambda entry: entry.seq,
    )
    for entry in queued:
        bench_version = int(entry.payload.get("bench_version", -1))
        if not supports_version(bench_version):
            continue
        agent = await session.get(Agent, entry.agent_id)
        ticket = await session.get(
            ValidatorTicket,
            (entry.agent_id, bench_version, validator_hotkey),
            with_for_update=True,
        )
        score = await session.get(
            Score, (entry.agent_id, bench_version, validator_hotkey)
        )
        stale_reason = None
        if agent is None or agent.status not in _FINALIZED_STATUSES:
            stale_reason = "submission is no longer finalized"
        elif ticket is None or ticket.status != TicketStatus.SCORED:
            stale_reason = "accepted score ticket is no longer reusable"
        elif score is None or score.run_id != entry.payload.get("run_id"):
            stale_reason = "accepted score changed while the request was queued"
        if stale_reason is not None:
            await _close_unserviceable(
                session, entry=entry, now=now, reason=stale_reason
            )
            continue

        assert ticket is not None
        deadline = now + REPLACEMENT_TICKET_TTL
        ticket.status = TicketStatus.ISSUED
        ticket.purpose = TicketPurpose.CANONICAL_QUORUM
        ticket.purpose_revision += 1
        ticket.legacy_completion_allowed = False
        ticket.slot_id = slot_id
        ticket.issued_at = now
        ticket.deadline = deadline
        ticket.attempt_count += 1
        ticket.retry_after = None
        payload = dict(entry.payload)
        payload["replacement_deadline"] = deadline.isoformat()
        await append_audit_entry(
            session,
            agent_id=entry.agent_id,
            validator_hotkey=validator_hotkey,
            event=EVENT_SCORE_RETEST_REQUESTED,
            payload=payload,
            recorded_at=now,
        )
        await session.flush()
        return ticket
    await session.flush()
    return None


def retest_is_open(entry: ScoreAuditEntry | None) -> bool:
    return entry is not None and entry.event in {
        EVENT_SCORE_RETEST_QUEUED,
        EVENT_SCORE_RETEST_REQUESTED,
    }


def retest_is_active(entry: ScoreAuditEntry | None) -> bool:
    return entry is not None and entry.event == EVENT_SCORE_RETEST_REQUESTED


def retest_is_queued(entry: ScoreAuditEntry | None) -> bool:
    return entry is not None and entry.event == EVENT_SCORE_RETEST_QUEUED


__all__ = [
    "EVENT_SCORE_INVALIDATED",
    "REPLACEMENT_TICKET_TTL",
    "activate_next_score_retest",
    "lock_validator",
    "retest_is_active",
    "retest_is_open",
    "retest_is_queued",
    "score_retest_queue_positions",
]
