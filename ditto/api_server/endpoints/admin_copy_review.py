"""Durable admin review surface for ``ath_pending_review`` holds."""

from datetime import UTC, datetime
from typing import Annotated, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.admin_copy_review import (
    AdminCopyReviewCurrentComparison,
    AdminCopyReviewEvidence,
    AdminCopyReviewItem,
    AdminCopyReviewList,
    AdminCopyReviewResolveRequest,
    AdminCopyReviewResolveResponse,
)
from ditto.api_server.dependencies import get_session
from ditto.api_server.endpoints.admin_quarantine import require_admin
from ditto.db.models import Agent, AgentStatus, AthReview

router = APIRouter(prefix="/admin", tags=["admin"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
AdminDep = Annotated[None, Depends(require_admin)]


def _fingerprint_versions(evidence: dict) -> dict[str, int | str | None]:
    return {
        "lexical": evidence.get("content_fingerprint_version"),
        "structural": evidence.get("structural_fingerprint_version"),
        "prompt": evidence.get("prompt_fingerprint_version"),
    }


def _item(review: AthReview, agent: Agent) -> AdminCopyReviewItem:
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
            reference_provenance=str(provenance.get("reference_provenance", "unknown")),
            backfilled=bool(provenance.get("backfilled", False)),
        ),
    )


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


@router.get("/copy-reviews", response_model=AdminCopyReviewList)
async def list_copy_reviews(
    _admin: AdminDep,
    session: SessionDep,
    status: Literal["pending", "resolved", "all"] = "pending",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
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
    return AdminCopyReviewList(
        items=[_item(review, agent) for review, agent in rows],
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
    return _item(*row)


@router.get(
    "/copy-reviews/{agent_id}/current-comparison",
    response_model=AdminCopyReviewCurrentComparison,
)
async def get_copy_review_current_comparison(
    agent_id: UUID, _admin: AdminDep, session: SessionDep
) -> AdminCopyReviewCurrentComparison:
    if await _get_review(session, agent_id) is None:
        raise HTTPException(status_code=404, detail="copy review not found")
    return AdminCopyReviewCurrentComparison(
        reason="corrected reference-aware comparison is not deployed",
        algorithm_provenance={
            "adapter": "unavailable",
            "reference_aware": False,
        },
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
        if review.status == "resolved":
            if review.resolution != canonical:
                raise HTTPException(status_code=409, detail="review already resolved")
            return AdminCopyReviewResolveResponse(
                review=_item(review, agent),
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
        review=_item(review, agent),
        agent_status=agent.status.value,
        idempotent=False,
    )
