"""Admin (Backroom) endpoints for anti-copy ``ath_pending_review`` holds.

Read side: list every held agent with its original stored hold AND a fresh,
side-effect-free re-run of the anti-copy gate against the current eligible
ledger (``recomputed`` / ``would_release``), plus current per-channel
similarity against the originally matched agent. The recompute is pure
(:func:`ditto.api_server.scoring_gate.evaluate_duplicate_signals` is
deterministic) and never mutates a row, so listing is always safe.

Write side: resolve ONE hold per call — ``release`` (cleared; the agent
returns to ``scored`` and re-enters the ledger) or ``ban`` (confirmed copy) —
via the same :func:`ditto.db.queries.agents.resolve_review` transition as the
owner CLI. Every resolution requires ``X-Admin-Actor`` and is logged with the
actor, decision, and the operator's reason; there is deliberately no
bulk-resolution endpoint (a console can issue individual audited calls).
"""

from __future__ import annotations

import logging
import statistics
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.admin_copy_review import (
    AdminCopyReviewItem,
    AdminCopyReviewList,
    AdminCopyReviewPairSimilarity,
    AdminCopyReviewRecomputed,
    AdminCopyReviewResolveRequest,
    AdminCopyReviewResolveResponse,
)
from ditto.api_server.dependencies import get_session
from ditto.api_server.endpoints.admin_quarantine import require_admin
from ditto.api_server.fingerprint import content_similarity
from ditto.api_server.scoring_gate import evaluate_duplicate_signals
from ditto.db.models import Agent, AgentStatus
from ditto.db.queries.agents import resolve_review
from ditto.db.queries.scores import list_eligible_ledger, list_scores_for_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
AdminDep = Annotated[None, Depends(require_admin)]

_RESOLUTION_STATUS = {
    "release": AgentStatus.SCORED,
    "ban": AgentStatus.BANNED,
}


def _pair_similarity(
    agent: Agent, matched: Agent | None
) -> AdminCopyReviewPairSimilarity | None:
    if matched is None:
        return None

    def channel(a: dict | None, b: dict | None) -> tuple[float | None, float | None]:
        if not a or not b:
            return (None, None)
        return content_similarity(a, b)

    lex = channel(agent.content_fingerprint, matched.content_fingerprint)
    struct = channel(agent.structural_fingerprint, matched.structural_fingerprint)
    prompt = channel(agent.prompt_fingerprint, matched.prompt_fingerprint)
    return AdminCopyReviewPairSimilarity(
        lexical_jaccard=lex[0],
        lexical_containment=lex[1],
        structural_jaccard=struct[0],
        structural_containment=struct[1],
        prompt_jaccard=prompt[0],
        prompt_containment=prompt[1],
    )


@router.get("/copy-reviews", response_model=AdminCopyReviewList)
async def list_copy_reviews(
    _admin: AdminDep,
    session: SessionDep,
) -> AdminCopyReviewList:
    held = (
        (
            await session.execute(
                select(Agent)
                .where(Agent.status == AgentStatus.ATH_PENDING_REVIEW)
                .order_by(Agent.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    eligible = await list_eligible_ledger(session)
    matched_ids = {a.duplicate_of for a in held if a.duplicate_of is not None}
    matched_rows: dict[UUID, Agent] = {}
    if matched_ids:
        for row in (
            (
                await session.execute(
                    select(Agent).where(Agent.agent_id.in_(matched_ids))
                )
            )
            .scalars()
            .all()
        ):
            matched_rows[row.agent_id] = row

    items: list[AdminCopyReviewItem] = []
    for agent in held:
        scores = await list_scores_for_agent(session, agent_id=agent.agent_id)
        median = (
            statistics.median(float(s.composite) for s in scores) if scores else None
        )
        if median is None:
            # A hold with no scores cannot be re-evaluated (the gate keys on
            # score proximity); surface it as still-held for human judgment.
            recomputed = AdminCopyReviewRecomputed(
                held=True,
                duplicate_of=agent.duplicate_of,
                reason="hold has no recorded scores; manual review required",
            )
        else:
            decision = evaluate_duplicate_signals(
                agent_id=agent.agent_id,
                submitted_at=agent.created_at,
                miner_hotkey=agent.miner_hotkey,
                sha256=agent.sha256,
                composite=median,
                size_bytes=agent.size_bytes,
                eligible=eligible,
                normalized_source_hash=agent.normalized_source_hash,
                content_fingerprint=agent.content_fingerprint,
                structural_fingerprint=agent.structural_fingerprint,
                prompt_fingerprint=agent.prompt_fingerprint,
            )
            recomputed = AdminCopyReviewRecomputed(
                held=decision.held,
                duplicate_of=decision.duplicate_of,
                reason=decision.reason,
            )
        items.append(
            AdminCopyReviewItem(
                agent_id=agent.agent_id,
                miner_hotkey=agent.miner_hotkey,
                agent_name=agent.name,
                agent_version=agent.version,
                sha256=agent.sha256,
                size_bytes=agent.size_bytes,
                submitted_at=agent.created_at,
                median_composite=median,
                stored_duplicate_of=agent.duplicate_of,
                stored_reason=agent.review_reason,
                recomputed=recomputed,
                would_release=not recomputed.held,
                pair_similarity=_pair_similarity(
                    agent,
                    matched_rows.get(agent.duplicate_of)
                    if agent.duplicate_of is not None
                    else None,
                ),
            )
        )
    return AdminCopyReviewList(
        items=items,
        count=len(items),
        would_release_count=sum(1 for item in items if item.would_release),
    )


@router.post(
    "/copy-reviews/{agent_id}/resolve",
    response_model=AdminCopyReviewResolveResponse,
)
async def resolve_copy_review(
    agent_id: UUID,
    payload: AdminCopyReviewResolveRequest,
    _admin: AdminDep,
    session: SessionDep,
    x_admin_actor: Annotated[str | None, Header()] = None,
) -> AdminCopyReviewResolveResponse:
    if x_admin_actor is None or not 1 <= len(x_admin_actor) <= 120:
        raise HTTPException(status_code=422, detail="X-Admin-Actor is required")
    decision = _RESOLUTION_STATUS[payload.resolution]
    try:
        async with session.begin():
            agent = await resolve_review(session, agent_id=agent_id, decision=decision)
    except ValueError as e:
        # resolve_review rejects an agent that is not (or no longer) held —
        # e.g. a concurrent operator resolved it first.
        raise HTTPException(status_code=409, detail=str(e)) from e
    if agent is None:
        raise HTTPException(status_code=404, detail="agent not found")
    logger.info(
        "copy review resolved: agent_id=%s resolution=%s actor=%s reason=%r",
        agent_id,
        payload.resolution,
        x_admin_actor,
        payload.reason,
    )
    return AdminCopyReviewResolveResponse(
        agent_id=agent_id,
        agent_status=agent.status.value,
        resolution=payload.resolution,
    )
