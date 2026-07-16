"""Durable admin review surface for ``ath_pending_review`` holds."""

import asyncio
from collections import defaultdict
from datetime import UTC, datetime
from typing import Annotated, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.admin_copy_review import (
    AdminCopyReviewComparisonUnavailable,
    AdminCopyReviewCurrentComparison,
    AdminCopyReviewEvidence,
    AdminCopyReviewItem,
    AdminCopyReviewList,
    AdminCopyReviewResolveRequest,
    AdminCopyReviewResolveResponse,
)
from ditto.api_server.anti_copy_comparison import compare_anti_copy_pair
from ditto.api_server.dependencies import get_session
from ditto.api_server.endpoints.admin_quarantine import require_admin
from ditto.db.models import Agent, AgentStatus, AthReview, Score
from ditto.db.queries.scores import (
    MIN_ELIGIBLE_CASES,
    LedgerRow,
    list_scores_for_agent,
)

router = APIRouter(prefix="/admin", tags=["admin"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
AdminDep = Annotated[None, Depends(require_admin)]


def _fingerprint_versions(evidence: dict) -> dict[str, int | str | None]:
    return {
        "lexical": evidence.get("content_fingerprint_version"),
        "structural": evidence.get("structural_fingerprint_version"),
        "prompt": evidence.get("prompt_fingerprint_version"),
    }


def _item(
    review: AthReview,
    agent: Agent,
    matched: Agent | None = None,
    comparison: (
        AdminCopyReviewCurrentComparison | AdminCopyReviewComparisonUnavailable | None
    ) = None,
) -> AdminCopyReviewItem:
    provenance = review.algorithm_provenance
    return AdminCopyReviewItem(
        review_id=review.review_id,
        agent_id=agent.agent_id,
        miner_hotkey=agent.miner_hotkey,
        agent_name=agent.name,
        agent_version=agent.version,
        submitted_at=agent.created_at,
        status=cast(Literal["pending", "resolved"], review.status),
        opened_at=review.opened_at,
        resolved_at=review.resolved_at,
        resolved_by=review.resolved_by,
        resolution=cast(Literal["clear", "reject"] | None, review.resolution),
        resolution_reason=review.resolution_reason,
        original=AdminCopyReviewEvidence(
            duplicate_of=review.original_duplicate_of,
            reason=review.original_reason,
            policy_version=review.original_policy_version,
            fingerprint_versions=_fingerprint_versions(review.original_evidence),
            reference_provenance=str(
                provenance.get("reference_corpus_id")
                or provenance.get("reference_provenance", "unknown")
            ),
            backfilled=bool(provenance.get("backfilled", False)),
            duplicate_of_name=matched.name if matched else None,
            duplicate_of_version=matched.version if matched else None,
            duplicate_of_hotkey=matched.miner_hotkey if matched else None,
            duplicate_of_submitted_at=matched.created_at if matched else None,
        ),
        current_comparison=comparison,
    )


async def _matched_agents(
    session: AsyncSession, reviews: list[AthReview]
) -> dict[UUID, Agent]:
    """Batch-load the originally matched agents for a page of reviews."""
    ids = {r.original_duplicate_of for r in reviews if r.original_duplicate_of}
    if not ids:
        return {}
    rows = (
        (await session.execute(select(Agent).where(Agent.agent_id.in_(ids))))
        .scalars()
        .all()
    )
    return {row.agent_id: row for row in rows}


_UNAVAILABLE = "current comparison unavailable"


async def _batch_comparisons(
    session: AsyncSession,
    rows: list[tuple[AthReview, Agent]],
    matched: dict[UUID, Agent],
) -> dict[
    UUID, AdminCopyReviewCurrentComparison | AdminCopyReviewComparisonUnavailable
]:
    """Recompute the pair comparison for a whole page of reviews.

    Consumers previously fanned out one ``/current-comparison`` request per
    row (~4 queries each). This loads every involved agent's scores with ONE
    ``IN`` query and runs the pure per-pair compares in a worker thread so
    the event loop stays responsive. Rows the dedicated endpoint would 409
    embed the same fail-closed unavailable state instead.
    """
    involved: set[UUID] = set()
    for review, agent in rows:
        if (
            review.status == "pending"
            and agent.status == AgentStatus.ATH_PENDING_REVIEW
            and review.original_duplicate_of is not None
            and review.original_duplicate_of in matched
        ):
            involved.add(agent.agent_id)
            involved.add(review.original_duplicate_of)
    scores_by_agent: defaultdict[UUID, list[Score]] = defaultdict(list)
    if involved:
        score_rows = (
            (await session.execute(select(Score).where(Score.agent_id.in_(involved))))
            .scalars()
            .all()
        )
        for score in score_rows:
            scores_by_agent[score.agent_id].append(score)

    pairs: list[tuple[UUID, LedgerRow, LedgerRow]] = []
    out: dict[
        UUID, AdminCopyReviewCurrentComparison | AdminCopyReviewComparisonUnavailable
    ] = {}
    for review, agent in rows:
        reference = (
            matched.get(review.original_duplicate_of)
            if review.original_duplicate_of
            else None
        )
        if (
            review.status != "pending"
            or agent.status != AgentStatus.ATH_PENDING_REVIEW
            or reference is None
        ):
            out[review.review_id] = AdminCopyReviewComparisonUnavailable(
                reason=_UNAVAILABLE
            )
            continue
        candidate = _canonical_ledger_row(
            agent, scores_by_agent.get(agent.agent_id, [])
        )
        reference_row = _canonical_ledger_row(
            reference, scores_by_agent.get(reference.agent_id, [])
        )
        if candidate is None or reference_row is None:
            out[review.review_id] = AdminCopyReviewComparisonUnavailable(
                reason=_UNAVAILABLE
            )
            continue
        pairs.append((review.review_id, candidate, reference_row))

    def _compute() -> dict[UUID, dict[str, object]]:
        return {
            review_id: compare_anti_copy_pair(
                candidate=candidate, reference=reference_row
            ).to_wire()
            for review_id, candidate, reference_row in pairs
        }

    for review_id, wire in (await asyncio.to_thread(_compute)).items():
        out[review_id] = AdminCopyReviewCurrentComparison.model_validate(wire)
    return out


async def _get_review(
    session: AsyncSession, agent_id: UUID, *, lock: bool = False
) -> tuple[AthReview, Agent] | None:
    stmt = (
        select(AthReview, Agent)
        .join(Agent, Agent.agent_id == AthReview.agent_id)
        .where(AthReview.agent_id == agent_id)
    )
    if lock:
        stmt = stmt.with_for_update()
    row = (await session.execute(stmt)).one_or_none()
    return None if row is None else (row[0], row[1])


def _canonical_ledger_row(agent: Agent, scores: list[Score]) -> LedgerRow | None:
    """Build the same median-score value object used by the anti-copy gate."""
    if not scores:
        return None
    ordered = sorted(
        scores, key=lambda score: (score.composite, score.validator_hotkey)
    )
    canonical = ordered[(len(ordered) - 1) // 2]
    return LedgerRow(
        miner_hotkey=agent.miner_hotkey,
        agent_id=agent.agent_id,
        composite=canonical.composite,
        tool_mean=canonical.tool_mean,
        memory_mean=canonical.memory_mean,
        first_seen=agent.created_at,
        sha256=agent.sha256,
        size_bytes=agent.size_bytes,
        run_id=canonical.run_id,
        seed=canonical.seed,
        validator_hotkey=canonical.validator_hotkey,
        signature=canonical.signature,
        status=agent.status,
        content_fingerprint=agent.content_fingerprint,
        structural_fingerprint=agent.structural_fingerprint,
        normalized_source_hash=agent.normalized_source_hash,
        prompt_fingerprint=agent.prompt_fingerprint,
        code_embedding=agent.code_embedding,
        code_embed_model=agent.code_embed_model,
        median_ms=canonical.median_ms,
        n=canonical.n,
        eligible=canonical.n >= MIN_ELIGIBLE_CASES and canonical.composite > 0.0,
        details=canonical.details,
    )


@router.get("/copy-reviews", response_model=AdminCopyReviewList)
async def list_copy_reviews(
    _admin: AdminDep,
    session: SessionDep,
    status: Literal["pending", "resolved", "all"] = "pending",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    include: Literal["current_comparison"] | None = None,
) -> AdminCopyReviewList:
    where = [] if status == "all" else [AthReview.status == status]
    count = await session.scalar(
        select(func.count()).select_from(AthReview).where(*where)
    )
    rows = (
        await session.execute(
            select(AthReview, Agent)
            .join(Agent, Agent.agent_id == AthReview.agent_id)
            .where(*where)
            .order_by(AthReview.opened_at.asc(), AthReview.review_id.asc())
            .limit(limit)
            .offset(offset)
        )
    ).all()
    row_pairs = [(review, agent) for review, agent in rows]
    matched = await _matched_agents(session, [review for review, _ in row_pairs])
    comparisons: dict[
        UUID, AdminCopyReviewCurrentComparison | AdminCopyReviewComparisonUnavailable
    ] = {}
    if include == "current_comparison":
        comparisons = await _batch_comparisons(session, row_pairs, matched)
    return AdminCopyReviewList(
        items=[
            _item(
                review,
                agent,
                matched.get(review.original_duplicate_of),
                comparison=comparisons.get(review.review_id),
            )
            for review, agent in row_pairs
        ],
        count=count or 0,
        limit=limit,
        offset=offset,
    )


@router.get("/copy-reviews/{agent_id}", response_model=AdminCopyReviewItem)
async def get_copy_review(
    agent_id: UUID, _admin: AdminDep, session: SessionDep
) -> AdminCopyReviewItem:
    row = await _get_review(session, agent_id)
    if row is None:
        raise HTTPException(status_code=404, detail="copy review not found")
    review, agent = row
    matched = (
        await session.get(Agent, review.original_duplicate_of)
        if review.original_duplicate_of
        else None
    )
    return _item(review, agent, matched)


@router.get(
    "/copy-reviews/{agent_id}/current-comparison",
    response_model=AdminCopyReviewCurrentComparison,
)
async def get_copy_review_current_comparison(
    agent_id: UUID, _admin: AdminDep, session: SessionDep
) -> dict[str, object]:
    row = await _get_review(session, agent_id)
    if row is None:
        raise HTTPException(status_code=404, detail="copy review not found")
    review, candidate_agent = row
    if (
        review.status != "pending"
        or candidate_agent.status != AgentStatus.ATH_PENDING_REVIEW
        or review.original_duplicate_of is None
    ):
        raise HTTPException(status_code=409, detail="current comparison unavailable")
    reference_agent = await session.get(Agent, review.original_duplicate_of)
    if reference_agent is None:
        raise HTTPException(status_code=409, detail="current comparison unavailable")
    candidate_scores = await list_scores_for_agent(
        session, agent_id=candidate_agent.agent_id
    )
    reference_scores = await list_scores_for_agent(
        session, agent_id=reference_agent.agent_id
    )
    candidate = _canonical_ledger_row(candidate_agent, candidate_scores)
    reference = _canonical_ledger_row(reference_agent, reference_scores)
    if candidate is None or reference is None:
        raise HTTPException(status_code=409, detail="current comparison unavailable")
    comparison = compare_anti_copy_pair(candidate=candidate, reference=reference)
    return comparison.to_wire()


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
    actor = x_admin_actor.strip() if x_admin_actor is not None else ""
    if not 1 <= len(actor) <= 120:
        raise HTTPException(status_code=422, detail="X-Admin-Actor is required")
    canonical = {"release": "clear", "ban": "reject"}.get(
        payload.resolution, payload.resolution
    )
    async with session.begin():
        row = await _get_review(session, agent_id, lock=True)
        if row is None:
            raise HTTPException(status_code=404, detail="copy review not found")
        review, agent = row
        matched = (
            await session.get(Agent, review.original_duplicate_of)
            if review.original_duplicate_of
            else None
        )
        if review.status == "resolved":
            if review.resolution != canonical:
                raise HTTPException(status_code=409, detail="review already resolved")
            return AdminCopyReviewResolveResponse(
                review=_item(review, agent, matched),
                agent_status=agent.status.value,
                idempotent=True,
            )
        if agent.status != AgentStatus.ATH_PENDING_REVIEW:
            raise HTTPException(status_code=409, detail="agent is no longer held")
        if agent.duplicate_of != review.original_duplicate_of:
            raise HTTPException(
                status_code=409, detail="agent hold evidence no longer matches review"
            )
        if agent.review_reason != review.original_reason:
            raise HTTPException(
                status_code=409, detail="agent hold reason no longer matches review"
            )
        agent.status = (
            AgentStatus.SCORED if canonical == "clear" else AgentStatus.BANNED
        )
        review.status = "resolved"
        review.resolved_at = datetime.now(UTC)
        review.resolved_by = actor
        review.resolution = canonical
        review.resolution_reason = payload.reason
        await session.flush()
    return AdminCopyReviewResolveResponse(
        review=_item(review, agent, matched),
        agent_status=agent.status.value,
        idempotent=False,
    )
