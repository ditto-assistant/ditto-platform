"""Audited recovery for submissions stranded by validator infrastructure.

The validator protocol currently reports lease progress but not a signed,
platform-verifiable terminal failure classification. Automatic infrastructure
retry would therefore guess from expiry and risk retrying deterministic miner
failures. This route requires an operator decision until that contract exists.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from datetime import UTC, datetime, timedelta
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.admin_validation_retry import (
    AdminScoreOutlier,
    AdminScoreOutlierList,
    AdminScoreOutlierScore,
    AdminStuckSubmission,
    AdminStuckSubmissionsResponse,
    AdminValidationRecovery,
    AdminValidationRetryDetail,
    AdminValidationRetryRequest,
    AdminValidationRetryResponse,
    AdminValidationTicket,
    AdminValidatorScoreReplacementDetail,
    AdminValidatorScoreReplacementRequest,
    AdminValidatorScoreReplacementResponse,
    AdminValidatorScoreRetestReleaseRequest,
    AdminValidatorScoreRetestReleaseResponse,
)
from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.retry_state import RETRY_STATE_ORDER, RetryState
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
from ditto.db.queries.agents import list_agents_by_status
from ditto.db.queries.audit import (
    EVENT_SCORE_RETEST_RELEASED,
    EVENT_SCORE_RETEST_REQUESTED,
    append_audit_entry,
    get_latest_score_retest_event,
)
from ditto.db.queries.benchmark_rollout import active_bench_version
from ditto.db.queries.retry_state import (
    classify_agent_retry_states,
    is_exhausted,
    recovery_gate,
    resolve_bench_version,
)
from ditto.db.queries.scores import SCORING_QUORUM
from ditto.db.queries.tickets import MAX_ATTEMPTS_PER_VERSION

router = APIRouter(prefix="/admin", tags=["admin"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
AdminDep = Annotated[None, Depends(require_admin)]

# How many EVALUATING agents to classify in one fleet sweep. The evaluating
# backlog is bounded (a few hundred at most); this caps a pathological scan.
_STUCK_SCAN_LIMIT = 2000

_REPLACEMENT_TICKET_TTL = timedelta(minutes=90)
_REPLACEABLE_STATUSES = {
    AgentStatus.EVALUATING,
    AgentStatus.SCORED,
    AgentStatus.LIVE,
}
_FINALIZED_STATUSES = {AgentStatus.SCORED, AgentStatus.LIVE}
OUTLIER_MIN_DEVIATION = 0.15
OUTLIER_GAP_RATIO = 2.0


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


def _score_evidence(score: Score) -> dict[str, object]:
    return {
        "run_id": score.run_id,
        "seed": score.seed,
        "composite": score.composite,
        "tool_mean": score.tool_mean,
        "memory_mean": score.memory_mean,
        "median_ms": score.median_ms,
        "n": score.n,
        "bench_version": score.bench_version,
        "ticket_deadline": (
            score.details.get("ticket_deadline")
            if isinstance(score.details, dict)
            else None
        ),
        "signature": score.signature,
        "generated_at": _aware(score.generated_at).isoformat(),
    }


def _detect_outlier(scores: list[Score]) -> tuple[Score, str, float, float] | None:
    """Find one unambiguous extreme in a three-score median quorum.

    The extreme-to-median gap must be at least 0.15 and at least twice the
    other adjacent gap. This catches one validator far above or below a tight
    peer pair while declining to guess when all three scores are simply broad.
    """
    if len(scores) != SCORING_QUORUM:
        return None
    ordered = sorted(
        scores, key=lambda score: (score.composite, score.validator_hotkey)
    )
    low_gap = ordered[1].composite - ordered[0].composite
    high_gap = ordered[2].composite - ordered[1].composite
    if low_gap == high_gap:
        return None
    if low_gap > high_gap:
        deviation, peer_spread = low_gap, high_gap
        candidate, direction = ordered[0], "low"
    else:
        deviation, peer_spread = high_gap, low_gap
        candidate, direction = ordered[2], "high"
    if deviation < OUTLIER_MIN_DEVIATION:
        return None
    if deviation < OUTLIER_GAP_RATIO * peer_spread:
        return None
    return candidate, direction, deviation, peer_spread


def _outlier_score(score: Score) -> AdminScoreOutlierScore:
    return AdminScoreOutlierScore(
        validator_hotkey=score.validator_hotkey,
        run_id=score.run_id,
        composite=score.composite,
    )


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
    bench_version = resolve_bench_version(
        all_tickets=all_tickets,
        all_scores=all_scores,
        canonical_version=canonical_version,
    )
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


@router.get("/validation-retries", response_model=AdminStuckSubmissionsResponse)
async def list_validation_retries(
    _admin: AdminDep,
    session: SessionDep,
    state: Annotated[list[str] | None, Query()] = None,
) -> AdminStuckSubmissionsResponse:
    """Fleet-wide triage of every below-quorum submission and why it is stuck.

    The single-agent detail route answers "why is *this* one stuck?"; this
    answers "which ones need me?" without a per-agent sweep. Filter with one or
    more ``state`` query params (e.g. ``?state=exhausted``); ``counts`` always
    reflects the whole fleet so a filtered view still shows the totals.
    """
    now = datetime.now(UTC)
    requested_states = set(state or [])
    unknown = requested_states - set(RETRY_STATE_ORDER)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail="unknown retry state: " + ", ".join(sorted(unknown)),
        )

    agents = await list_agents_by_status(
        session, statuses=[AgentStatus.EVALUATING], limit=_STUCK_SCAN_LIMIT
    )
    classified = await classify_agent_retry_states(session, agents=agents, now=now)
    agents_by_id = {agent.agent_id: agent for agent in agents}

    submissions: list[AdminStuckSubmission] = []
    counts: dict[RetryState, int] = {}
    for agent_id, retry in classified.items():
        counts[retry.state] = counts.get(retry.state, 0) + 1
        if requested_states and retry.state not in requested_states:
            continue
        agent = agents_by_id[agent_id]
        scored_hotkeys = {s.validator_hotkey for s in retry.scores}
        submissions.append(
            AdminStuckSubmission(
                agent_id=agent.agent_id,
                miner_hotkey=agent.miner_hotkey,
                agent_name=agent.name,
                agent_version=agent.version,
                bench_version=retry.bench_version,
                score_count=retry.score_count,
                quorum=SCORING_QUORUM,
                retry_state=retry.state,
                automatic_retry_available=retry.automatic_retry_available,
                recovery_allowed=retry.recovery_allowed,
                blocking_reason=retry.blocking_reason,
                earliest_retry_after=retry.earliest_retry_after,
                attempts_used=max((t.attempt_count for t in retry.tickets), default=0),
                exhausted_validator_count=sum(
                    1
                    for t in retry.tickets
                    if is_exhausted(t) and t.validator_hotkey not in scored_hotkeys
                ),
                snapshot=_snapshot(
                    agent=agent, scores=retry.scores, tickets=retry.tickets
                ),
                tickets=[_ticket_item(t) for t in retry.tickets],
            )
        )

    submissions.sort(
        key=lambda item: (
            RETRY_STATE_ORDER[item.retry_state],
            item.earliest_retry_after or datetime.max.replace(tzinfo=UTC),
            item.agent_id,
        )
    )
    return AdminStuckSubmissionsResponse(
        generated_at=now,
        quorum=SCORING_QUORUM,
        counts=counts,
        submissions=submissions,
    )


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
    automatic, allowed, reason, _ = recovery_gate(
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
        _, allowed, reason, selected = recovery_gate(
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
    replacement_pending: bool = False,
) -> str | None:
    if agent.status not in _REPLACEABLE_STATUSES:
        return "submission is not in a scoreable state"
    if target is None:
        return "validator has no accepted score to replace"
    if replacement_pending:
        return "replacement score is already pending"
    if ticket is None or ticket.status != TicketStatus.SCORED:
        return "accepted score is not backed by a consumed validator ticket"
    if validator_busy:
        return "validator is currently assigned to another submission"
    return None


@router.get("/score-outliers", response_model=AdminScoreOutlierList)
async def list_score_outliers(
    _admin: AdminDep,
    session: SessionDep,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> AdminScoreOutlierList:
    """List finalized three-score quorums with one unambiguous extreme."""
    agents = list(
        (
            await session.scalars(
                select(Agent)
                .where(Agent.status.in_(_FINALIZED_STATUSES))
                .order_by(Agent.agent_id.asc())
            )
        ).all()
    )
    detected: list[AdminScoreOutlier] = []
    for listed_agent in agents:
        agent, bench_version, scores, tickets, _ = await _load(
            session, agent_id=listed_agent.agent_id, for_update=False
        )
        assert agent is not None
        result = _detect_outlier(scores)
        if result is None:
            continue
        target, direction, deviation, peer_spread = result
        ticket = next(
            (
                item
                for item in tickets
                if item.validator_hotkey == target.validator_hotkey
            ),
            None,
        )
        latest = await get_latest_score_retest_event(
            session,
            agent_id=agent.agent_id,
            validator_hotkey=target.validator_hotkey,
        )
        pending = latest is not None and latest.event == EVENT_SCORE_RETEST_REQUESTED
        busy = await _validator_busy_elsewhere(
            session,
            agent_id=agent.agent_id,
            validator_hotkey=target.validator_hotkey,
        )
        blocking = _replacement_gate(
            agent=agent,
            target=target,
            ticket=ticket,
            validator_busy=busy,
            replacement_pending=pending,
        )
        detected.append(
            AdminScoreOutlier(
                agent_id=agent.agent_id,
                agent_name=agent.name,
                miner_hotkey=agent.miner_hotkey,
                agent_status=agent.status.value,
                bench_version=bench_version,
                snapshot=_snapshot(agent=agent, scores=scores, tickets=tickets),
                median_composite=float(
                    statistics.median(score.composite for score in scores)
                ),
                direction=direction,  # type: ignore[arg-type]
                outlier=_outlier_score(target),
                peers=[
                    _outlier_score(score) for score in scores if score is not target
                ],
                deviation=deviation,
                peer_spread=peer_spread,
                ticket_status=ticket.status.value if ticket is not None else None,
                replacement_pending=pending,
                replacement_deadline=(
                    ticket.deadline if pending and ticket is not None else None
                ),
                replacement_allowed=blocking is None,
                blocking_reason=blocking,
            )
        )

    detected.sort(key=lambda item: (-item.deviation, str(item.agent_id)))
    return AdminScoreOutlierList(
        items=detected[offset : offset + limit],
        count=len(detected),
        limit=limit,
        offset=offset,
    )


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
    latest = await get_latest_score_retest_event(
        session, agent_id=agent_id, validator_hotkey=validator_hotkey
    )
    pending = latest is not None and latest.event == EVENT_SCORE_RETEST_REQUESTED
    reason = _replacement_gate(
        agent=agent,
        target=target,
        ticket=ticket,
        validator_busy=await _validator_busy_elsewhere(
            session, agent_id=agent_id, validator_hotkey=validator_hotkey
        ),
        replacement_pending=pending,
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
        replacement_pending=pending,
        replacement_request_id=(
            UUID(str(latest.payload["request_id"]))
            if pending and latest is not None
            else None
        ),
        replacement_reason=(
            str(latest.payload["reason"]) if pending and latest is not None else None
        ),
        replacement_actor=(
            str(latest.payload["actor"]) if pending and latest is not None else None
        ),
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
                        ScoreAuditEntry.event == EVENT_SCORE_RETEST_REQUESTED,
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
                original_run_id=payload.expected_run_id,
                bench_version=int(prior.payload["bench_version"]),
                replacement_deadline=datetime.fromisoformat(
                    str(prior.payload["replacement_deadline"])
                ),
                preserved_score_count=int(prior.payload["preserved_score_count"]),
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
            replacement_pending=False,
        )
        if reason is not None:
            raise HTTPException(status_code=409, detail=reason)
        assert target is not None and ticket is not None
        now = datetime.now(UTC)
        deadline = now + _REPLACEMENT_TICKET_TTL
        ticket.status = TicketStatus.ISSUED
        ticket.issued_at = now
        ticket.deadline = deadline
        ticket.attempt_count += 1
        ticket.retry_after = None
        await append_audit_entry(
            session,
            agent_id=agent_id,
            validator_hotkey=validator_hotkey,
            event=EVENT_SCORE_RETEST_REQUESTED,
            payload={
                "request_id": str(payload.request_id),
                "actor": actor,
                "reason": payload.reason,
                "expected_snapshot": payload.expected_snapshot,
                "bench_version": bench_version,
                "run_id": payload.expected_run_id,
                "preserved_score": _score_evidence(target),
                "replacement_deadline": deadline.isoformat(),
                "preserved_score_count": len(scores),
            },
            recorded_at=now,
        )
        await session.flush()
    return AdminValidatorScoreReplacementResponse(
        request_id=payload.request_id,
        agent_id=agent_id,
        validator_hotkey=validator_hotkey,
        original_run_id=payload.expected_run_id,
        bench_version=bench_version,
        replacement_deadline=deadline,
        preserved_score_count=len(scores),
        idempotent=False,
    )


@router.post(
    "/validation-retries/{agent_id}/validators/{validator_hotkey}/release-ticket",
    response_model=AdminValidatorScoreRetestReleaseResponse,
)
async def release_validator_score_retest_ticket(
    agent_id: UUID,
    validator_hotkey: str,
    payload: AdminValidatorScoreRetestReleaseRequest,
    _admin: AdminDep,
    session: SessionDep,
    x_admin_actor: Annotated[str | None, Header()] = None,
) -> AdminValidatorScoreRetestReleaseResponse:
    actor = x_admin_actor.strip() if x_admin_actor is not None else ""
    if not 1 <= len(actor) <= 120:
        raise HTTPException(status_code=422, detail="X-Admin-Actor is required")
    async with session.begin():
        release_entries = list(
            (
                await session.scalars(
                    select(ScoreAuditEntry).where(
                        ScoreAuditEntry.agent_id == agent_id,
                        ScoreAuditEntry.validator_hotkey == validator_hotkey,
                        ScoreAuditEntry.event == EVENT_SCORE_RETEST_RELEASED,
                    )
                )
            ).all()
        )
        prior = next(
            (
                entry
                for entry in release_entries
                if entry.payload.get("request_id") == str(payload.request_id)
            ),
            None,
        )
        if prior is not None:
            if (
                prior.payload.get("actor") != actor
                or prior.payload.get("reason") != payload.reason
                or prior.payload.get("expected_snapshot") != payload.expected_snapshot
                or prior.payload.get("expected_deadline")
                != _aware(payload.expected_deadline).isoformat()
            ):
                raise HTTPException(status_code=409, detail="request id already used")
            return AdminValidatorScoreRetestReleaseResponse(
                request_id=payload.request_id,
                agent_id=agent_id,
                validator_hotkey=validator_hotkey,
                status="scored",
                preserved_run_id=str(prior.payload["preserved_run_id"]),
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
        latest = await get_latest_score_retest_event(
            session, agent_id=agent_id, validator_hotkey=validator_hotkey
        )
        if latest is None or latest.event != EVENT_SCORE_RETEST_REQUESTED:
            raise HTTPException(
                status_code=409, detail="no replacement ticket is pending"
            )
        ticket = next(
            (
                item
                for item in tickets
                if item.validator_hotkey == validator_hotkey
                and item.bench_version == bench_version
            ),
            None,
        )
        if target is None or ticket is None:
            raise HTTPException(
                status_code=409, detail="replacement state is incomplete"
            )
        if _aware(ticket.deadline) != _aware(payload.expected_deadline):
            raise HTTPException(status_code=409, detail="replacement ticket changed")
        if ticket.status not in {TicketStatus.ISSUED, TicketStatus.EXPIRED}:
            raise HTTPException(
                status_code=409, detail="replacement ticket is not releasable"
            )
        ticket.status = TicketStatus.SCORED
        ticket.retry_after = None
        now = datetime.now(UTC)
        await append_audit_entry(
            session,
            agent_id=agent_id,
            validator_hotkey=validator_hotkey,
            event=EVENT_SCORE_RETEST_RELEASED,
            payload={
                "request_id": str(payload.request_id),
                "retest_request_id": latest.payload.get("request_id"),
                "actor": actor,
                "reason": payload.reason,
                "expected_snapshot": payload.expected_snapshot,
                "expected_deadline": _aware(payload.expected_deadline).isoformat(),
                "bench_version": bench_version,
                "preserved_run_id": target.run_id,
            },
            recorded_at=now,
        )
        await session.flush()
    return AdminValidatorScoreRetestReleaseResponse(
        request_id=payload.request_id,
        agent_id=agent_id,
        validator_hotkey=validator_hotkey,
        status="scored",
        preserved_run_id=target.run_id,
        idempotent=False,
    )
