"""Audited recovery for submissions stranded by validator infrastructure.

The validator protocol currently reports lease progress but not a signed,
platform-verifiable terminal failure classification. Automatic infrastructure
retry would therefore guess from expiry and risk retrying deterministic miner
failures. This route requires an operator decision until that contract exists.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.admin_validation_retry import (
    AdminValidationRecovery,
    AdminValidationRetryDetail,
    AdminValidationRetryRequest,
    AdminValidationRetryResponse,
    AdminValidationTicket,
)
from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.api_models.ticket_status import TicketStatus
from ditto.api_server.dependencies import get_session
from ditto.api_server.endpoints.admin_quarantine import require_admin
from ditto.db.models import Agent, Score, ValidatorRetryRecovery, ValidatorTicket
from ditto.db.queries.benchmark_rollout import active_bench_version
from ditto.db.queries.scores import SCORING_QUORUM
from ditto.db.queries.tickets import MAX_ATTEMPTS_PER_VERSION

router = APIRouter(prefix="/admin", tags=["admin"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
AdminDep = Annotated[None, Depends(require_admin)]

MAX_OPERATOR_RECOVERIES_PER_AGENT = 3


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _ticket_wire(ticket: ValidatorTicket) -> dict[str, object]:
    return {
        "validator_hotkey": ticket.validator_hotkey,
        "status": ticket.status.value,
        "issued_at": _aware(ticket.issued_at).isoformat(timespec="microseconds"),
        "deadline": _aware(ticket.deadline).isoformat(timespec="microseconds"),
        "bench_version": ticket.bench_version,
        "attempt_count": ticket.attempt_count,
        "manual_retry_grants": ticket.manual_retry_grants,
        "retry_after": (
            _aware(ticket.retry_after).isoformat(timespec="microseconds")
            if ticket.retry_after is not None
            else None
        ),
    }


def _snapshot(
    *, agent: Agent, scores: list[Score], tickets: list[ValidatorTicket]
) -> str:
    payload = {
        "agent_id": str(agent.agent_id),
        "status": agent.status.value,
        "artifact_sha256": agent.sha256,
        "screening_policy_version": agent.screening_policy_version,
        "scores": [
            {
                "validator_hotkey": score.validator_hotkey,
                "run_id": score.run_id,
                "composite": score.composite,
                "signature": score.signature,
                "generated_at": _aware(score.generated_at).isoformat(
                    timespec="microseconds"
                ),
                "updated_at": _aware(score.updated_at).isoformat(
                    timespec="microseconds"
                ),
            }
            for score in sorted(
                scores, key=lambda item: (item.validator_hotkey, item.run_id)
            )
        ],
        "tickets": [_ticket_wire(ticket) for ticket in tickets],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _recovery_item(row: ValidatorRetryRecovery) -> AdminValidationRecovery:
    return AdminValidationRecovery(
        recovery_id=row.recovery_id,
        agent_id=row.agent_id,
        actor=row.actor,
        reason=row.reason,
        score_count=row.score_count,
        bench_version=row.bench_version,
        expected_snapshot=row.expected_snapshot,
        granted_validator_hotkeys=list(row.granted_validator_hotkeys),
        created_at=row.created_at,
    )


def _ticket_item(ticket: ValidatorTicket) -> AdminValidationTicket:
    return AdminValidationTicket(
        validator_hotkey=ticket.validator_hotkey,
        status=ticket.status.value,  # type: ignore[arg-type]
        issued_at=ticket.issued_at,
        deadline=ticket.deadline,
        bench_version=ticket.bench_version,
        attempt_count=ticket.attempt_count,
        manual_retry_grants=ticket.manual_retry_grants,
        retry_after=ticket.retry_after,
        retry_budget_exhausted=(
            ticket.status == TicketStatus.EXPIRED
            and ticket.attempt_count
            >= MAX_ATTEMPTS_PER_VERSION + ticket.manual_retry_grants
        ),
    )


def _recovery_gate(
    *,
    agent: Agent,
    scores: list[Score],
    tickets: list[ValidatorTicket],
    recovery_count: int,
    now: datetime,
    bench_version: int,
) -> tuple[bool, bool, str | None, list[ValidatorTicket]]:
    score_count = len(scores)
    if agent.status != AgentStatus.EVALUATING:
        return False, False, "submission is not waiting for validator scores", []
    if agent.screening_policy_version < SCREENING_POLICY_VERSION:
        return False, False, "submission is not on the current screening policy", []
    if score_count >= SCORING_QUORUM:
        return False, False, "submission already reached scoring quorum", []
    if any(ticket.status == TicketStatus.ISSUED for ticket in tickets):
        return False, False, "a validator ticket is still active", []
    if recovery_count >= MAX_OPERATOR_RECOVERIES_PER_AGENT:
        return False, False, "operator retry limit reached", []

    score_hotkeys = {score.validator_hotkey for score in scores}
    non_scored = [
        ticket
        for ticket in tickets
        if ticket.status != TicketStatus.SCORED
        and ticket.validator_hotkey not in score_hotkeys
        and ticket.bench_version == bench_version
    ]
    naturally_retryable = [
        ticket
        for ticket in non_scored
        if ticket.status == TicketStatus.EXPIRED
        and ticket.attempt_count < MAX_ATTEMPTS_PER_VERSION + ticket.manual_retry_grants
    ]
    if naturally_retryable:
        available = any(
            ticket.retry_after is None or _aware(ticket.retry_after) <= now
            for ticket in naturally_retryable
        )
        reason = (
            "automatic validator retry is already available"
            if available
            else "automatic validator retry is still cooling down"
        )
        return available, False, reason, []

    needed = SCORING_QUORUM - score_count
    exhausted = sorted(
        (
            ticket
            for ticket in non_scored
            if ticket.status == TicketStatus.EXPIRED
            and ticket.attempt_count
            >= MAX_ATTEMPTS_PER_VERSION + ticket.manual_retry_grants
        ),
        key=lambda ticket: (_aware(ticket.deadline), ticket.validator_hotkey),
    )
    if len(exhausted) < needed:
        return False, False, "not enough expired tickets to restore quorum", []
    return False, True, None, exhausted[:needed]


async def _load(
    session: AsyncSession, *, agent_id: UUID, for_update: bool
) -> tuple[
    Agent | None,
    int,
    list[Score],
    list[ValidatorTicket],
    list[ValidatorRetryRecovery],
]:
    agent_query = select(Agent).where(Agent.agent_id == agent_id)
    if for_update:
        agent_query = agent_query.with_for_update()
    agent = await session.scalar(agent_query)
    if agent is None:
        return None, 2, [], [], []
    all_scores = list(
        (
            await session.scalars(
                select(Score)
                .where(Score.agent_id == agent_id)
                .order_by(Score.validator_hotkey.asc(), Score.run_id.asc())
            )
        ).all()
    )
    ticket_query = (
        select(ValidatorTicket)
        .where(ValidatorTicket.agent_id == agent_id)
        .order_by(
            ValidatorTicket.deadline.asc(), ValidatorTicket.validator_hotkey.asc()
        )
    )
    if for_update:
        ticket_query = ticket_query.with_for_update()
    all_tickets = list((await session.scalars(ticket_query)).all())
    canonical_version = await active_bench_version(session)
    work_tickets = [
        ticket
        for ticket in all_tickets
        if ticket.status in (TicketStatus.ISSUED, TicketStatus.EXPIRED)
    ]
    if work_tickets:
        bench_version = max(
            work_tickets,
            key=lambda ticket: (_aware(ticket.issued_at), ticket.bench_version),
        ).bench_version
    elif all_scores:
        bench_version = max(
            all_scores,
            key=lambda score: (_aware(score.generated_at), score.bench_version),
        ).bench_version
    else:
        bench_version = canonical_version
    scores = [score for score in all_scores if score.bench_version == bench_version]
    tickets = [
        ticket for ticket in all_tickets if ticket.bench_version == bench_version
    ]
    recoveries = list(
        (
            await session.scalars(
                select(ValidatorRetryRecovery)
                .where(
                    ValidatorRetryRecovery.agent_id == agent_id,
                    ValidatorRetryRecovery.bench_version == bench_version,
                )
                .order_by(
                    ValidatorRetryRecovery.created_at.asc(),
                    ValidatorRetryRecovery.recovery_id.asc(),
                )
            )
        ).all()
    )
    return agent, bench_version, scores, tickets, recoveries


@router.get("/validation-retries/{agent_id}", response_model=AdminValidationRetryDetail)
async def get_validation_retry(
    agent_id: UUID, _admin: AdminDep, session: SessionDep
) -> AdminValidationRetryDetail:
    agent, bench_version, scores, tickets, recoveries = await _load(
        session, agent_id=agent_id, for_update=False
    )
    if agent is None:
        raise HTTPException(status_code=404, detail="agent not found")
    score_count = len(scores)
    automatic, allowed, reason, _ = _recovery_gate(
        agent=agent,
        scores=scores,
        tickets=tickets,
        recovery_count=len(recoveries),
        now=datetime.now(UTC),
        bench_version=bench_version,
    )
    return AdminValidationRetryDetail(
        agent_id=agent.agent_id,
        miner_hotkey=agent.miner_hotkey,
        agent_name=agent.name,
        agent_version=agent.version,
        agent_status=agent.status.value,
        score_count=score_count,
        quorum=SCORING_QUORUM,
        snapshot=_snapshot(agent=agent, scores=scores, tickets=tickets),
        automatic_retry_available=automatic,
        recovery_allowed=allowed,
        blocking_reason=reason,
        tickets=[_ticket_item(ticket) for ticket in tickets],
        recoveries=[_recovery_item(row) for row in recoveries],
    )


@router.post(
    "/validation-retries/{agent_id}/retry",
    response_model=AdminValidationRetryResponse,
)
async def retry_validation_after_infrastructure_failure(
    agent_id: UUID,
    payload: AdminValidationRetryRequest,
    _admin: AdminDep,
    session: SessionDep,
    x_admin_actor: Annotated[str | None, Header()] = None,
) -> AdminValidationRetryResponse:
    actor = x_admin_actor.strip() if x_admin_actor is not None else ""
    if not 1 <= len(actor) <= 120:
        raise HTTPException(status_code=422, detail="X-Admin-Actor is required")
    async with session.begin():
        agent, bench_version, scores, tickets, recoveries = await _load(
            session, agent_id=agent_id, for_update=True
        )
        if agent is None:
            raise HTTPException(status_code=404, detail="agent not found")
        score_count = len(scores)

        existing = await session.get(ValidatorRetryRecovery, payload.request_id)
        if existing is not None:
            if (
                existing.agent_id != agent_id
                or existing.actor != actor
                or existing.reason != payload.reason
                or existing.expected_snapshot != payload.expected_snapshot
            ):
                raise HTTPException(status_code=409, detail="request id already used")
            return AdminValidationRetryResponse(
                recovery=_recovery_item(existing), idempotent=True
            )

        current_snapshot = _snapshot(agent=agent, scores=scores, tickets=tickets)
        if current_snapshot != payload.expected_snapshot:
            raise HTTPException(status_code=409, detail="validation state changed")
        _, allowed, reason, selected = _recovery_gate(
            agent=agent,
            scores=scores,
            tickets=tickets,
            recovery_count=len(recoveries),
            now=datetime.now(UTC),
            bench_version=bench_version,
        )
        if not allowed:
            raise HTTPException(status_code=409, detail=reason or "retry unavailable")

        ticket_snapshot = [_ticket_wire(ticket) for ticket in tickets]
        now = datetime.now(UTC)
        for ticket in selected:
            ticket.manual_retry_grants += 1
            ticket.retry_after = now
        recovery = ValidatorRetryRecovery(
            recovery_id=payload.request_id,
            agent_id=agent_id,
            actor=actor,
            reason=payload.reason,
            expected_snapshot=current_snapshot,
            score_count=score_count,
            bench_version=bench_version,
            ticket_snapshot=ticket_snapshot,
            granted_validator_hotkeys=[ticket.validator_hotkey for ticket in selected],
            created_at=now,
        )
        session.add(recovery)
        await session.flush()
    return AdminValidationRetryResponse(
        recovery=_recovery_item(recovery), idempotent=False
    )
