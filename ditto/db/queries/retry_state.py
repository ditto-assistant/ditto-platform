"""Shared validator-retry classification.

Both the operator triage routes (``endpoints/admin_validation_retry.py``) and the
public operations feed (``endpoints/public.py``) must agree on why a below-quorum
submission is or is not advancing. That verdict lives here once — a pure gate
plus a bulk loader — so the two surfaces can never drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.retry_state import RetryState
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.api_models.ticket_status import TicketStatus
from ditto.db.models import Agent, Score, ValidatorRetryRecovery, ValidatorTicket
from ditto.db.queries.benchmark_rollout import active_bench_version
from ditto.db.queries.scores import SCORING_QUORUM
from ditto.db.queries.tickets import MAX_ATTEMPTS_PER_VERSION

# Operators may hand-grant at most this many recoveries to one agent before the
# stuck submission needs a harder look than another retry.
MAX_OPERATOR_RECOVERIES_PER_AGENT = 3


def aware(value: datetime) -> datetime:
    """Coerce a DB-read datetime to UTC-aware (SQLite round-trips tz-naive)."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def resolve_bench_version(
    *,
    all_tickets: list[ValidatorTicket],
    all_scores: list[Score],
    canonical_version: int,
) -> int:
    """Pick the benchmark era an agent's retry state belongs to.

    Live/expired work is the strongest signal (the era it is being scored on
    right now); otherwise its newest recorded score; otherwise the canonical
    active version for an agent with no ticket or score history yet.
    """
    work_tickets = [
        ticket
        for ticket in all_tickets
        if ticket.status in (TicketStatus.ISSUED, TicketStatus.EXPIRED)
    ]
    if work_tickets:
        return max(
            work_tickets,
            key=lambda ticket: (aware(ticket.issued_at), ticket.bench_version),
        ).bench_version
    if all_scores:
        return max(
            all_scores,
            key=lambda score: (aware(score.generated_at), score.bench_version),
        ).bench_version
    return canonical_version


def is_exhausted(ticket: ValidatorTicket) -> bool:
    """An expired ticket whose validator has spent its whole attempt budget."""
    return (
        ticket.status == TicketStatus.EXPIRED
        and ticket.attempt_count
        >= MAX_ATTEMPTS_PER_VERSION + ticket.manual_retry_grants
    )


def recovery_gate(
    *,
    agent: Agent,
    scores: list[Score],
    tickets: list[ValidatorTicket],
    recovery_count: int,
    now: datetime,
    bench_version: int,
) -> tuple[bool, bool, str | None, list[ValidatorTicket]]:
    """Decide whether an automatic or operator retry is possible.

    Returns ``(automatic_retry_available, recovery_allowed, blocking_reason,
    tickets_to_grant)``. ``tickets_to_grant`` is the minimum set of expired,
    budget-exhausted tickets an operator grant would revive to restore quorum.
    """
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
            ticket.retry_after is None or aware(ticket.retry_after) <= now
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
        (ticket for ticket in non_scored if is_exhausted(ticket)),
        key=lambda ticket: (aware(ticket.deadline), ticket.validator_hotkey),
    )
    if len(exhausted) < needed:
        return False, False, "not enough expired tickets to restore quorum", []
    return False, True, None, exhausted[:needed]


def classify_retry_state(
    *,
    automatic: bool,
    allowed: bool,
    reason: str | None,
    scores: list[Score],
    tickets: list[ValidatorTicket],
) -> RetryState | None:
    """Fold the gate's verdict into a single triage label.

    Returns ``None`` for a submission that is not actually below quorum and
    therefore not stuck. ``exhausted`` is the only label that needs an operator.
    """
    score_hotkeys = {score.validator_hotkey for score in scores}
    if len(scores) >= SCORING_QUORUM:
        return None
    if any(ticket.status == TicketStatus.ISSUED for ticket in tickets):
        return "running"
    if automatic:
        return "retry_available"
    if reason == "automatic validator retry is still cooling down":
        return "cooling_down"
    if allowed:
        return "exhausted"
    # No natural or operator-grantable retry is pending. Only call it exhausted
    # (needs a human) when enough validators have burned their budget that the
    # submission cannot reach quorum on its own — at least as many exhausted
    # non-scored tickets as scores still needed. Fewer than that means the
    # remaining slots were simply never leased and fresh validators will fill
    # them, so it is still queued, not stuck.
    needed = SCORING_QUORUM - len(scores)
    exhausted_unscored = sum(
        1
        for ticket in tickets
        if is_exhausted(ticket) and ticket.validator_hotkey not in score_hotkeys
    )
    if exhausted_unscored >= needed:
        return "exhausted"
    return "queued"


@dataclass(frozen=True)
class AgentRetryState:
    """One agent's resolved retry classification and the evidence behind it."""

    state: RetryState
    bench_version: int
    score_count: int
    automatic_retry_available: bool
    recovery_allowed: bool
    blocking_reason: str | None
    earliest_retry_after: datetime | None
    scores: list[Score]
    tickets: list[ValidatorTicket]


async def classify_agent_retry_states(
    session: AsyncSession,
    *,
    agents: list[Agent],
    now: datetime,
) -> dict[UUID, AgentRetryState]:
    """Classify a batch of agents' retry state in three bulk statements.

    Only ``EVALUATING`` agents below quorum get an entry; a finished, rejected,
    uploaded, or otherwise-not-waiting agent is omitted (its retry state is
    meaningless). The public activity feed passes agents of every status, so the
    non-evaluating ones are filtered out up front — before the bulk load — which
    also keeps the id list bounded to the (small) evaluating backlog rather than
    every submission ever made.
    """
    agent_by_id = {
        agent.agent_id: agent
        for agent in agents
        if agent.status == AgentStatus.EVALUATING
    }
    if not agent_by_id:
        return {}
    id_subq = (
        select(Agent.agent_id)
        .where(Agent.agent_id.in_(list(agent_by_id)))
        .scalar_subquery()
    )
    tickets_by_agent: dict[UUID, list[ValidatorTicket]] = {}
    for ticket in (
        await session.scalars(
            select(ValidatorTicket).where(ValidatorTicket.agent_id.in_(id_subq))
        )
    ).all():
        tickets_by_agent.setdefault(ticket.agent_id, []).append(ticket)
    scores_by_agent: dict[UUID, list[Score]] = {}
    for score in (
        await session.scalars(select(Score).where(Score.agent_id.in_(id_subq)))
    ).all():
        scores_by_agent.setdefault(score.agent_id, []).append(score)
    recovery_counts: dict[tuple[UUID, int], int] = {
        (agent_id, version): count
        for agent_id, version, count in (
            await session.execute(
                select(
                    ValidatorRetryRecovery.agent_id,
                    ValidatorRetryRecovery.bench_version,
                    func.count(),
                )
                .where(ValidatorRetryRecovery.agent_id.in_(id_subq))
                .group_by(
                    ValidatorRetryRecovery.agent_id,
                    ValidatorRetryRecovery.bench_version,
                )
            )
        ).all()
    }
    canonical_version = await active_bench_version(session)

    result: dict[UUID, AgentRetryState] = {}
    for agent_id, agent in agent_by_id.items():
        all_tickets = sorted(
            tickets_by_agent.get(agent_id, []),
            key=lambda ticket: (aware(ticket.deadline), ticket.validator_hotkey),
        )
        all_scores = scores_by_agent.get(agent_id, [])
        bench_version = resolve_bench_version(
            all_tickets=all_tickets,
            all_scores=all_scores,
            canonical_version=canonical_version,
        )
        v_scores = [s for s in all_scores if s.bench_version == bench_version]
        v_tickets = [t for t in all_tickets if t.bench_version == bench_version]
        automatic, allowed, reason, _ = recovery_gate(
            agent=agent,
            scores=v_scores,
            tickets=v_tickets,
            recovery_count=recovery_counts.get((agent_id, bench_version), 0),
            now=now,
            bench_version=bench_version,
        )
        state = classify_retry_state(
            automatic=automatic,
            allowed=allowed,
            reason=reason,
            scores=v_scores,
            tickets=v_tickets,
        )
        if state is None:
            continue
        scored_hotkeys = {s.validator_hotkey for s in v_scores}
        result[agent_id] = AgentRetryState(
            state=state,
            bench_version=bench_version,
            score_count=len(v_scores),
            automatic_retry_available=automatic,
            recovery_allowed=allowed,
            blocking_reason=reason,
            # Only a ticket that can still retry has a meaningful "retry at"
            # time; an exhausted ticket's stale cooldown must not read as
            # "coming back soon".
            earliest_retry_after=min(
                (
                    aware(t.retry_after)
                    for t in v_tickets
                    if t.status == TicketStatus.EXPIRED
                    and t.retry_after is not None
                    and t.validator_hotkey not in scored_hotkeys
                    and not is_exhausted(t)
                ),
                default=None,
            ),
            scores=v_scores,
            tickets=v_tickets,
        )
    return result
