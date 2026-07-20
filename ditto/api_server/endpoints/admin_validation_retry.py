"""Audited recovery for submissions stranded by validator infrastructure.

The validator protocol currently reports lease progress but not a signed,
platform-verifiable terminal failure classification. Automatic infrastructure
retry would therefore guess from expiry and risk retrying deterministic miner
failures. This route requires an operator decision until that contract exists.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
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
    AdminValidatorScoreReplacementDetail,
    AdminValidatorScoreReplacementRequest,
    AdminValidatorScoreReplacementResponse,
)
from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.api_models.ticket_status import TicketStatus
from ditto.api_server.dependencies import get_session
from ditto.api_server.endpoints.admin_quarantine import require_admin
from ditto.db.models import (
    Agent,
    Score,
    ScoreAuditEntry,
    ValidatorRetryRecovery,
    ValidatorTicket,
)
from ditto.db.queries.audit import EVENT_SCORE_INVALIDATED, append_audit_entry
from ditto.db.queries.benchmark_rollout import active_bench_version
from ditto.db.queries.scores import SCORING_QUORUM
from ditto.db.queries.tickets import MAX_ATTEMPTS_PER_VERSION

router = APIRouter(prefix="/admin", tags=["admin"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
AdminDep = Annotated[None, Depends(require_admin)]

MAX_OPERATOR_RECOVERIES_PER_AGENT = 3
_REPLACEMENT_TICKET_TTL = timedelta(minutes=90)
_REPLACEABLE_STATUSES = {
    AgentStatus.EVALUATING,
    AgentStatus.SCORED,
    AgentStatus.LIVE,
}


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


async def _replacement_state(
    session: AsyncSession,
    *,
    agent_id: UUID,
    validator_hotkey: str,
    for_update: bool,
) -> tuple[Agent | None, int, list[Score], list[ValidatorTicket], Score | None]:
    agent, bench_version, scores, tickets, _ = await _load(
        session, agent_id=agent_id, for_update=for_update
    )
    target = next(
        (score for score in scores if score.validator_hotkey == validator_hotkey),
        None,
    )
    return agent, bench_version, scores, tickets, target


async def _validator_busy_elsewhere(
    session: AsyncSession,
    *,
    agent_id: UUID,
    validator_hotkey: str,
) -> bool:
    return (
        await session.scalar(
            select(ValidatorTicket.agent_id).where(
                ValidatorTicket.validator_hotkey == validator_hotkey,
                ValidatorTicket.status == TicketStatus.ISSUED,
                ValidatorTicket.agent_id != agent_id,
            )
        )
        is not None
    )


def _replacement_gate(
    *,
    agent: Agent,
    target: Score | None,
    ticket: ValidatorTicket | None,
    validator_busy: bool,
) -> str | None:
    if agent.status not in _REPLACEABLE_STATUSES:
        return "submission is not in a scoreable state"
    if target is None:
        return "validator has no accepted score to replace"
    if ticket is None or ticket.status != TicketStatus.SCORED:
        return "accepted score is not backed by a consumed validator ticket"
    if validator_busy:
        return "validator is currently assigned to another submission"
    return None


@router.get(
    "/validation-retries/{agent_id}/validators/{validator_hotkey}",
    response_model=AdminValidatorScoreReplacementDetail,
)
async def inspect_validator_score_replacement(
    agent_id: UUID,
    validator_hotkey: str,
    _admin: AdminDep,
    session: SessionDep,
) -> AdminValidatorScoreReplacementDetail:
    agent, bench_version, scores, tickets, target = await _replacement_state(
        session,
        agent_id=agent_id,
        validator_hotkey=validator_hotkey,
        for_update=False,
    )
    if agent is None:
        raise HTTPException(status_code=404, detail="agent not found")
    ticket = next(
        (
            item
            for item in tickets
            if item.validator_hotkey == validator_hotkey
            and item.bench_version == bench_version
        ),
        None,
    )
    reason = _replacement_gate(
        agent=agent,
        target=target,
        ticket=ticket,
        validator_busy=await _validator_busy_elsewhere(
            session, agent_id=agent_id, validator_hotkey=validator_hotkey
        ),
    )
    return AdminValidatorScoreReplacementDetail(
        agent_id=agent_id,
        validator_hotkey=validator_hotkey,
        agent_status=agent.status.value,
        bench_version=bench_version,
        score_count=len(scores),
        quorum=SCORING_QUORUM,
        snapshot=_snapshot(agent=agent, scores=scores, tickets=tickets),
        run_id=target.run_id if target is not None else None,
        composite=target.composite if target is not None else None,
        ticket_status=ticket.status.value if ticket is not None else None,
        ticket_deadline=ticket.deadline if ticket is not None else None,
        replacement_allowed=reason is None,
        blocking_reason=reason,
    )


@router.post(
    "/validation-retries/{agent_id}/validators/{validator_hotkey}/replace-score",
    response_model=AdminValidatorScoreReplacementResponse,
)
async def replace_validator_score_after_infrastructure_failure(
    agent_id: UUID,
    validator_hotkey: str,
    payload: AdminValidatorScoreReplacementRequest,
    _admin: AdminDep,
    session: SessionDep,
    x_admin_actor: Annotated[str | None, Header()] = None,
) -> AdminValidatorScoreReplacementResponse:
    actor = x_admin_actor.strip() if x_admin_actor is not None else ""
    if not 1 <= len(actor) <= 120:
        raise HTTPException(status_code=422, detail="X-Admin-Actor is required")
    async with session.begin():
        prior_entries = list(
            (
                await session.scalars(
                    select(ScoreAuditEntry).where(
                        ScoreAuditEntry.agent_id == agent_id,
                        ScoreAuditEntry.validator_hotkey == validator_hotkey,
                        ScoreAuditEntry.event == EVENT_SCORE_INVALIDATED,
                    )
                )
            ).all()
        )
        prior = next(
            (
                entry
                for entry in prior_entries
                if entry.payload.get("request_id") == str(payload.request_id)
            ),
            None,
        )
        if prior is not None:
            if (
                prior.payload.get("actor") != actor
                or prior.payload.get("reason") != payload.reason
                or prior.payload.get("run_id") != payload.expected_run_id
                or prior.payload.get("expected_snapshot") != payload.expected_snapshot
            ):
                raise HTTPException(status_code=409, detail="request id already used")
            return AdminValidatorScoreReplacementResponse(
                request_id=payload.request_id,
                agent_id=agent_id,
                validator_hotkey=validator_hotkey,
                invalidated_run_id=payload.expected_run_id,
                bench_version=int(prior.payload["bench_version"]),
                replacement_deadline=datetime.fromisoformat(
                    str(prior.payload["replacement_deadline"])
                ),
                remaining_score_count=int(prior.payload["remaining_score_count"]),
                idempotent=True,
            )

        agent, bench_version, scores, tickets, target = await _replacement_state(
            session,
            agent_id=agent_id,
            validator_hotkey=validator_hotkey,
            for_update=True,
        )
        if agent is None:
            raise HTTPException(status_code=404, detail="agent not found")
        current_snapshot = _snapshot(agent=agent, scores=scores, tickets=tickets)
        if current_snapshot != payload.expected_snapshot:
            raise HTTPException(status_code=409, detail="validation state changed")
        if target is not None and target.run_id != payload.expected_run_id:
            raise HTTPException(status_code=409, detail="accepted score run changed")
        ticket = next(
            (
                item
                for item in tickets
                if item.validator_hotkey == validator_hotkey
                and item.bench_version == bench_version
            ),
            None,
        )
        reason = _replacement_gate(
            agent=agent,
            target=target,
            ticket=ticket,
            validator_busy=await _validator_busy_elsewhere(
                session, agent_id=agent_id, validator_hotkey=validator_hotkey
            ),
        )
        if reason is not None:
            raise HTTPException(status_code=409, detail=reason)
        assert target is not None and ticket is not None
        now = datetime.now(UTC)
        deadline = now + _REPLACEMENT_TICKET_TTL
        invalidated = {
            "run_id": target.run_id,
            "seed": target.seed,
            "composite": target.composite,
            "tool_mean": target.tool_mean,
            "memory_mean": target.memory_mean,
            "median_ms": target.median_ms,
            "n": target.n,
            "bench_version": target.bench_version,
            "ticket_deadline": (
                target.details.get("ticket_deadline")
                if isinstance(target.details, dict)
                else None
            ),
            "signature": target.signature,
            "generated_at": _aware(target.generated_at).isoformat(),
        }
        await session.delete(target)
        ticket.status = TicketStatus.ISSUED
        ticket.issued_at = now
        ticket.deadline = deadline
        ticket.attempt_count += 1
        ticket.retry_after = None
        if agent.status in {AgentStatus.SCORED, AgentStatus.LIVE}:
            agent.status = AgentStatus.EVALUATING
        remaining = len(scores) - 1
        await append_audit_entry(
            session,
            agent_id=agent_id,
            validator_hotkey=validator_hotkey,
            event=EVENT_SCORE_INVALIDATED,
            payload={
                "request_id": str(payload.request_id),
                "actor": actor,
                "reason": payload.reason,
                "expected_snapshot": payload.expected_snapshot,
                "bench_version": bench_version,
                "run_id": payload.expected_run_id,
                "invalidated_score": invalidated,
                "replacement_deadline": deadline.isoformat(),
                "remaining_score_count": remaining,
            },
            recorded_at=now,
        )
        await session.flush()
    return AdminValidatorScoreReplacementResponse(
        request_id=payload.request_id,
        agent_id=agent_id,
        validator_hotkey=validator_hotkey,
        invalidated_run_id=payload.expected_run_id,
        bench_version=bench_version,
        replacement_deadline=deadline,
        remaining_score_count=remaining,
        idempotent=False,
    )
