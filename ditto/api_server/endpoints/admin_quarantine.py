"""Authenticated operator API for durable screening quarantines."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.admin_quarantine import (
    AdminArtifactDuplicate,
    AdminDuplicateSummary,
    AdminMinerContext,
    AdminMinerQuarantineSummary,
    AdminQuarantineAgentContext,
    AdminQuarantineContext,
    AdminQuarantineItem,
    AdminQuarantineList,
    AdminQuarantineResolutionEvent,
    AdminQuarantineResolveRequest,
    AdminQuarantineResolveResponse,
    AdminScreeningAttempt,
    AdminScreeningDisputeItem,
    AdminScreeningDisputeList,
    AdminScreeningDisputeResolveRequest,
    AdminScreeningDisputeResolveResponse,
    AdminScreeningRescreenRequest,
    AdminScreeningRescreenResponse,
    AdminScreeningSubmission,
    AdminScreeningSubmissionList,
    AdminSourceExcerpt,
    AdminSourceListing,
    AdminValidatorAssignment,
    AdminValidatorAssignmentList,
    AdminValidatorAssignmentReleaseRequest,
    AdminValidatorAssignmentReleaseResponse,
)
from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.screener import ScreenEvidenceItem, SourceReviewFinding
from ditto.api_models.ticket_status import TicketStatus
from ditto.api_models.validator import ArtifactResponse
from ditto.api_server.datapipeline import DatasetGenerator
from ditto.api_server.dependencies import (
    get_dataset_generator,
    get_session,
    get_storage_client,
)
from ditto.api_server.endpoints.screener import _derive_dataset_seed
from ditto.api_server.endpoints.validator import ChainDep
from ditto.api_server.source_inspect import (
    MAX_READ_LINES,
    MAX_TARBALL_BYTES,
    SourceInspectError,
    TarSourceInspector,
)
from ditto.api_server.storage import ObjectDownloadFailedError, S3StorageClient
from ditto.db.models import (
    Agent,
    Score,
    ScreeningAttempt,
    ScreeningDispute,
    ScreeningQuarantine,
    ScreeningQuarantineResolution,
    ValidatorTicket,
)
from ditto.db.queries.tickets import RETRY_COOLDOWN

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
GeneratorDep = Annotated[DatasetGenerator, Depends(get_dataset_generator)]
StorageDep = Annotated[S3StorageClient, Depends(get_storage_client)]
DatasetPin = tuple[int, str, str, int | None, str | None]


async def require_admin(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    expected = request.app.state.config.admin_api_token
    if expected is None:
        raise HTTPException(status_code=503, detail="admin API is not configured")
    prefix = "Bearer "
    if authorization is None or not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="missing admin bearer token")
    if not secrets.compare_digest(authorization[len(prefix) :], expected):
        raise HTTPException(status_code=401, detail="invalid admin bearer token")


AdminDep = Annotated[None, Depends(require_admin)]


def _review_payloads(
    row: ScreeningQuarantine, agent: Agent
) -> tuple[list[ScreenEvidenceItem] | None, SourceReviewFinding | None, bool]:
    """Parse the stored review payloads, tolerating legacy/foreign shapes.

    Rows written before the payloads landed have nulls; a row whose JSON no
    longer parses (schema drift) degrades to null rather than breaking the
    whole listing. ``finding_verified`` re-derives the digest binding at read
    time — and requires the finding to name THIS agent's artifact digest — so
    the console never has to trust a stored boolean and a finding copied from
    another submission can never present as verified.
    """
    evidence: list[ScreenEvidenceItem] | None = None
    if isinstance(row.evidence, list):
        try:
            evidence = [
                ScreenEvidenceItem.model_validate(item) for item in row.evidence[:16]
            ]
        except ValueError:
            evidence = None
    finding: SourceReviewFinding | None = None
    if isinstance(row.finding, dict):
        try:
            finding = SourceReviewFinding.model_validate(row.finding)
        except ValueError:
            finding = None
    verified = (
        finding is not None
        and row.finding_digest is not None
        and finding.canonical_digest() == row.finding_digest
        and finding.artifact_sha256 == agent.sha256
    )
    return evidence, finding, verified


def _item(
    row: ScreeningQuarantine,
    agent: Agent,
    history: list[ScreeningQuarantineResolution] | None = None,
) -> AdminQuarantineItem:
    evidence, finding, finding_verified = _review_payloads(row, agent)


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return AdminQuarantineItem(
        quarantine_id=row.quarantine_id,
        agent_id=row.agent_id,
        attempt_id=row.attempt_id,
        miner_hotkey=agent.miner_hotkey,
        agent_name=agent.name,
        agent_version=agent.version,
        artifact_sha256=agent.sha256,
        policy_version=row.policy_version,
        manifest_digest=row.manifest_digest,
        finding_digest=row.finding_digest,
        reason_code=row.reason_code,
        evidence=evidence,
        finding=finding,
        finding_verified=finding_verified,
        status=row.status,  # type: ignore[arg-type]
        created_at=row.created_at,
        resolved_at=row.resolved_at,
        resolved_by=row.resolved_by,
        resolution=row.resolution,  # type: ignore[arg-type]
        resolution_reason=row.resolution_reason,
        resolution_history=[
            AdminQuarantineResolutionEvent(
                resolution=event.resolution,  # type: ignore[arg-type]
                reason=event.reason,
                actor=event.actor,
                created_at=event.created_at,
            )
            for event in history or []
        ],
    )


async def _resolution_history(
    session: AsyncSession, quarantine_ids: list[UUID]
) -> dict[UUID, list[ScreeningQuarantineResolution]]:
    history: dict[UUID, list[ScreeningQuarantineResolution]] = defaultdict(list)
    if not quarantine_ids:
        return history
    events = await session.scalars(
        select(ScreeningQuarantineResolution)
        .where(ScreeningQuarantineResolution.quarantine_id.in_(quarantine_ids))
        .order_by(
            ScreeningQuarantineResolution.created_at,
            ScreeningQuarantineResolution.resolution_id,
        )
    )
    for event in events:
        history[event.quarantine_id].append(event)
    return history


async def _prepare_release_dataset(
    session: AsyncSession,
    chain: ChainDep,
    generator: DatasetGenerator,
    quarantine_id: UUID,
) -> DatasetPin | None:
    if generator.run_size is None:
        return None
    existing = await session.scalar(
        select(Agent)
        .join(ScreeningQuarantine, ScreeningQuarantine.agent_id == Agent.agent_id)
        .where(ScreeningQuarantine.quarantine_id == quarantine_id)
    )
    await session.rollback()
    if existing is None:
        raise HTTPException(status_code=404, detail="quarantine not found")
    if existing.dataset_seed is not None:
        return None
    seed, block_number, block_hash = await _derive_dataset_seed(
        chain, existing.agent_id
    )
    dataset_sha256 = await generator.generate(seed)
    return seed, dataset_sha256, generator.run_size, block_number, block_hash


def _apply_dataset(agent: Agent, dataset: DatasetPin | None) -> None:
    if dataset is None or agent.dataset_seed is not None:
        return
    (
        agent.dataset_seed,
        agent.dataset_sha256,
        agent.dataset_run_size,
        agent.dataset_seed_block,
        agent.dataset_seed_block_hash,
    ) = dataset


def _dispute_item(
    dispute: ScreeningDispute,
    agent: Agent,
    quarantine: ScreeningQuarantine,
    history: list[ScreeningQuarantineResolution],
) -> AdminScreeningDisputeItem:
    original_reason = next(
        (event.reason for event in history if event.resolution == "reject"),
        quarantine.resolution_reason,
    )
    return AdminScreeningDisputeItem(
        dispute_id=dispute.dispute_id,
        agent_id=dispute.agent_id,
        quarantine_id=dispute.quarantine_id,
        miner_hotkey=dispute.miner_hotkey,
        agent_name=agent.name,
        agent_version=agent.version,
        artifact_sha256=agent.sha256,
        message=dispute.message,
        status=dispute.status,  # type: ignore[arg-type]
        created_at=dispute.created_at,
        original_reason=original_reason,
        resolved_at=dispute.resolved_at,
        resolved_by=dispute.resolved_by,
        resolution=dispute.resolution,  # type: ignore[arg-type]
        resolution_reason=dispute.resolution_reason,
    )


@router.get("/validator-assignments", response_model=AdminValidatorAssignmentList)
async def list_validator_assignments(
    _admin: AdminDep,
    session: SessionDep,
) -> AdminValidatorAssignmentList:
    """List live scoring leases for operator recovery tooling."""
    now = datetime.now(UTC)
    rows = (
        await session.execute(
            select(
                ValidatorTicket,
                Agent,
                func.count(Score.validator_hotkey),
                func.avg(Score.composite),
            )
            .join(Agent, Agent.agent_id == ValidatorTicket.agent_id)
            .outerjoin(Score, Score.agent_id == ValidatorTicket.agent_id)
            .where(
                ValidatorTicket.status == TicketStatus.ISSUED,
                ValidatorTicket.deadline > now,
            )
            .group_by(ValidatorTicket, Agent)
            .order_by(ValidatorTicket.deadline.asc(), ValidatorTicket.agent_id.asc())
        )
    ).all()
    items = [
        AdminValidatorAssignment(
            agent_id=ticket.agent_id,
            agent_name=agent.name,
            miner_hotkey=agent.miner_hotkey,
            validator_hotkey=ticket.validator_hotkey,
            issued_at=ticket.issued_at,
            deadline=ticket.deadline,
            bench_version=ticket.bench_version,
            attempt_count=ticket.attempt_count,
            score_count=int(score_count),
            provisional_composite=(
                float(provisional_composite)
                if provisional_composite is not None
                else None
            ),
        )
        for ticket, agent, score_count, provisional_composite in rows
    ]
    return AdminValidatorAssignmentList(items=items, count=len(items))


@router.post(
    "/validator-assignments/{agent_id}/{validator_hotkey}/release",
    response_model=AdminValidatorAssignmentReleaseResponse,
)
async def release_validator_assignment(
    agent_id: UUID,
    validator_hotkey: str,
    payload: AdminValidatorAssignmentReleaseRequest,
    _admin: AdminDep,
    session: SessionDep,
    x_admin_actor: Annotated[str | None, Header()] = None,
) -> AdminValidatorAssignmentReleaseResponse:
    """Expire one exact live lease without deleting its submission or scores."""
    if x_admin_actor is None or not 1 <= len(x_admin_actor) <= 120:
        raise HTTPException(status_code=422, detail="X-Admin-Actor is required")

    now = datetime.now(UTC)
    async with session.begin():
        ticket = await session.scalar(
            select(ValidatorTicket)
            .where(
                ValidatorTicket.agent_id == agent_id,
                ValidatorTicket.validator_hotkey == validator_hotkey,
            )
            .with_for_update()
        )
        if ticket is None:
            raise HTTPException(
                status_code=404, detail="validator assignment not found"
            )
        if (
            ticket.status != TicketStatus.ISSUED
            or _as_utc(ticket.deadline) <= now
            or _as_utc(ticket.deadline) != _as_utc(payload.expected_deadline)
        ):
            raise HTTPException(
                status_code=409,
                detail="validator assignment changed or is no longer active",
            )
        ticket.status = TicketStatus.EXPIRED
        ticket.retry_after = now + RETRY_COOLDOWN

    logger.warning(
        "admin released validator assignment actor=%s agent_id=%s validator=%s "
        "deadline=%s retry_after=%s reason=%r",
        x_admin_actor,
        agent_id,
        validator_hotkey,
        payload.expected_deadline.isoformat(),
        ticket.retry_after.isoformat(),
        payload.reason,
    )
    return AdminValidatorAssignmentReleaseResponse(
        agent_id=agent_id,
        validator_hotkey=validator_hotkey,
        status="expired",
        retry_after=ticket.retry_after,
    )


@router.get("/screening-quarantines", response_model=AdminQuarantineList)
async def list_quarantines(
    _admin: AdminDep,
    session: SessionDep,
    status: Annotated[Literal["active", "resolved", "all"], Query()] = "active",
    sort: Annotated[Literal["oldest", "newest"], Query()] = "oldest",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AdminQuarantineList:
    order = (
        (
            ScreeningQuarantine.created_at.asc(),
            ScreeningQuarantine.quarantine_id.asc(),
        )
        if sort == "oldest"
        else (
            ScreeningQuarantine.created_at.desc(),
            ScreeningQuarantine.quarantine_id.desc(),
        )
    )
    stmt = (
        select(ScreeningQuarantine, Agent)
        .join(Agent, Agent.agent_id == ScreeningQuarantine.agent_id)
        .order_by(*order)
        .offset(offset)
        .limit(limit)
    )
    if status != "all":
        stmt = stmt.where(ScreeningQuarantine.status == status)
    count_stmt = select(func.count()).select_from(ScreeningQuarantine)
    if status != "all":
        count_stmt = count_stmt.where(ScreeningQuarantine.status == status)
    total = int((await session.scalar(count_stmt)) or 0)
    rows = (await session.execute(stmt)).all()
    history = await _resolution_history(
        session, [quarantine.quarantine_id for quarantine, _agent in rows]
    )
    items = [
        _item(quarantine, agent, history[quarantine.quarantine_id])
        for quarantine, agent in rows
    ]
    return AdminQuarantineList(items=items, count=total)


@router.get(
    "/screening-quarantines/{quarantine_id}", response_model=AdminQuarantineItem
)
async def get_quarantine(
    quarantine_id: UUID, _admin: AdminDep, session: SessionDep
) -> AdminQuarantineItem:
    result = (
        await session.execute(
            select(ScreeningQuarantine, Agent)
            .join(Agent, Agent.agent_id == ScreeningQuarantine.agent_id)
            .where(ScreeningQuarantine.quarantine_id == quarantine_id)
        )
    ).one_or_none()
    if result is None:
        raise HTTPException(status_code=404, detail="quarantine not found")
    quarantine, agent = result
    history = await _resolution_history(session, [quarantine.quarantine_id])
    return _item(quarantine, agent, history[quarantine.quarantine_id])


@router.get(
    "/screening-quarantines/{quarantine_id}/context",
    response_model=AdminQuarantineContext,
)
async def get_quarantine_context(
    quarantine_id: UUID, _admin: AdminDep, session: SessionDep
) -> AdminQuarantineContext:
    """One-stop review context: finding, attempts, miner history, duplicates."""
    result = (
        await session.execute(
            select(ScreeningQuarantine, Agent)
            .join(Agent, Agent.agent_id == ScreeningQuarantine.agent_id)
            .where(ScreeningQuarantine.quarantine_id == quarantine_id)
        )
    ).one_or_none()
    if result is None:
        raise HTTPException(status_code=404, detail="quarantine not found")
    quarantine, agent = result

    attempts = (
        (
            await session.execute(
                select(ScreeningAttempt)
                .where(ScreeningAttempt.agent_id == agent.agent_id)
                .order_by(
                    ScreeningAttempt.started_at.desc(),
                    ScreeningAttempt.attempt_id.desc(),
                )
            )
        )
        .scalars()
        .all()
    )

    total_submissions = int(
        (
            await session.scalar(
                select(func.count())
                .select_from(Agent)
                .where(Agent.miner_hotkey == agent.miner_hotkey)
            )
        )
        or 0
    )
    # Aggregate in SQL: a prolific miner must not make the console
    # materialize their entire quarantine history per request.
    resolution_rows = (
        await session.execute(
            select(ScreeningQuarantine.resolution, func.count())
            .join(Agent, Agent.agent_id == ScreeningQuarantine.agent_id)
            .where(Agent.miner_hotkey == agent.miner_hotkey)
            .group_by(ScreeningQuarantine.resolution)
        )
    ).all()
    resolution_counts = {
        resolution: int(count) for resolution, count in resolution_rows
    }
    quarantine_count = sum(resolution_counts.values())
    miner_rows = (
        await session.execute(
            select(ScreeningQuarantine, Agent)
            .join(Agent, Agent.agent_id == ScreeningQuarantine.agent_id)
            .where(
                Agent.miner_hotkey == agent.miner_hotkey,
                ScreeningQuarantine.quarantine_id != quarantine_id,
            )
            .order_by(
                ScreeningQuarantine.created_at.desc(),
                ScreeningQuarantine.quarantine_id.desc(),
            )
            .limit(10)
        )
    ).all()
    recent = [
        AdminMinerQuarantineSummary(
            quarantine_id=row.quarantine_id,
            agent_id=row.agent_id,
            agent_name=other.name,
            reason_code=row.reason_code,
            status=row.status,  # type: ignore[arg-type]
            resolution=row.resolution,  # type: ignore[arg-type]
            resolution_reason=row.resolution_reason,
            created_at=row.created_at,
            resolved_at=row.resolved_at,
        )
        for row, other in miner_rows
    ]

    # Exact-duplicate signals only: identical tarball bytes, or identical
    # canonicalized source (reformat/re-comment/reorder repack). Fuzzy MinHash
    # similarity stays in the scoring gate; here a hit must be self-evident.
    duplicate_conditions = [Agent.sha256 == agent.sha256]
    if agent.normalized_source_hash is not None:
        duplicate_conditions.append(
            Agent.normalized_source_hash == agent.normalized_source_hash
        )
    duplicate_filter = (
        Agent.agent_id != agent.agent_id,
        or_(*duplicate_conditions),
    )
    # Authoritative aggregate counts, independent of the bounded sample below:
    # attribution claims ("another miner submitted this exact code") must
    # never be derived from whether a 20-row sample happened to include one.
    cross_miner_count = int(
        (
            await session.scalar(
                select(func.count())
                .select_from(Agent)
                .where(*duplicate_filter, Agent.miner_hotkey != agent.miner_hotkey)
            )
        )
        or 0
    )
    same_miner_count = int(
        (
            await session.scalar(
                select(func.count())
                .select_from(Agent)
                .where(*duplicate_filter, Agent.miner_hotkey == agent.miner_hotkey)
            )
        )
        or 0
    )
    duplicate_rows = (
        (
            await session.execute(
                select(Agent)
                .where(*duplicate_filter)
                .order_by(Agent.created_at.desc(), Agent.agent_id.desc())
                .limit(20)
            )
        )
        .scalars()
        .all()
    )
    duplicates = [
        AdminArtifactDuplicate(
            agent_id=other.agent_id,
            miner_hotkey=other.miner_hotkey,
            agent_name=other.name,
            agent_status=other.status,
            submitted_at=other.created_at,
            match=(
                "identical_artifact"
                if other.sha256 == agent.sha256
                else "identical_normalized_source"
            ),
        )
        for other in duplicate_rows
    ]

    return AdminQuarantineContext(
        quarantine=_item(quarantine, agent),
        agent=AdminQuarantineAgentContext(
            agent_id=agent.agent_id,
            miner_hotkey=agent.miner_hotkey,
            agent_name=agent.name,
            artifact_sha256=agent.sha256,
            agent_status=agent.status,
            size_bytes=agent.size_bytes,
            submitted_at=agent.created_at,
            screening_policy_version=agent.screening_policy_version,
            screening_reason=agent.screening_reason,
        ),
        attempts=[
            AdminScreeningAttempt(
                attempt_id=attempt.attempt_id,
                policy_version=attempt.policy_version,
                status=attempt.status,  # type: ignore[arg-type]
                screener_hotkey=attempt.screener_hotkey,
                started_at=attempt.started_at,
                deadline=attempt.deadline,
                finished_at=attempt.finished_at,
                reason=attempt.public_reason,
                reason_code=attempt.reason_code,
                duplicate_of=attempt.duplicate_of,
            )
            for attempt in attempts
        ],
        miner=AdminMinerContext(
            miner_hotkey=agent.miner_hotkey,
            total_submissions=total_submissions,
            quarantine_count=quarantine_count,
            released_count=resolution_counts.get("release", 0),
            rescreened_count=resolution_counts.get("rescreen", 0),
            rejected_count=resolution_counts.get("reject", 0),
            recent_quarantines=recent,
        ),
        duplicates=duplicates,
        duplicate_summary=AdminDuplicateSummary(
            total=cross_miner_count + same_miner_count,
            cross_miner=cross_miner_count,
            same_miner=same_miner_count,
            sample_truncated=cross_miner_count + same_miner_count > len(duplicate_rows),
        ),
    )


@router.post(
    "/screening-quarantines/{quarantine_id}/resolve",
    response_model=AdminQuarantineResolveResponse,
)
async def resolve_quarantine(
    quarantine_id: UUID,
    payload: AdminQuarantineResolveRequest,
    _admin: AdminDep,
    session: SessionDep,
    chain: ChainDep,
    generator: GeneratorDep,
    x_admin_actor: Annotated[str | None, Header()] = None,
) -> AdminQuarantineResolveResponse:
    if x_admin_actor is None or not 1 <= len(x_admin_actor) <= 120:
        raise HTTPException(status_code=422, detail="X-Admin-Actor is required")

    new_dataset = (
        await _prepare_release_dataset(session, chain, generator, quarantine_id)
        if payload.resolution == "release"
        else None
    )

    async with session.begin():
        quarantine = await session.scalar(
            select(ScreeningQuarantine)
            .where(ScreeningQuarantine.quarantine_id == quarantine_id)
            .with_for_update()
        )
        if quarantine is None:
            raise HTTPException(status_code=404, detail="quarantine not found")
        agent = await session.scalar(
            select(Agent).where(Agent.agent_id == quarantine.agent_id).with_for_update()
        )
        if agent is None:
            raise HTTPException(status_code=404, detail="agent not found")
        is_initial_resolution = (
            quarantine.status == "active" and agent.status == AgentStatus.QUARANTINED
        )
        is_rejection_correction = (
            quarantine.status == "resolved"
            and quarantine.resolution == "reject"
            and agent.status == AgentStatus.REJECTED
            and payload.resolution == "release"
        )
        if not is_initial_resolution and not is_rejection_correction:
            raise HTTPException(
                status_code=409,
                detail="quarantine is not active or a correctable rejection",
            )

        target = {
            "release": AgentStatus.EVALUATING,
            "rescreen": AgentStatus.SCREENING_FAILED,
            "reject": AgentStatus.REJECTED,
        }[payload.resolution]
        agent.status = target
        agent.screening_reason = payload.reason
        _apply_dataset(agent, new_dataset)
        quarantine.status = "resolved"
        quarantine.resolved_at = datetime.now(UTC)
        quarantine.resolved_by = x_admin_actor
        quarantine.resolution = payload.resolution
        quarantine.resolution_reason = payload.reason
        session.add(
            ScreeningQuarantineResolution(
                resolution_id=uuid4(),
                quarantine_id=quarantine.quarantine_id,
                resolution=payload.resolution,
                reason=payload.reason,
                actor=x_admin_actor,
                created_at=quarantine.resolved_at,
            )
        )

    history = await _resolution_history(session, [quarantine.quarantine_id])
    return AdminQuarantineResolveResponse(
        quarantine=_item(quarantine, agent, history[quarantine.quarantine_id]),
        agent_status=agent.status,
    )


@router.get("/screening-disputes", response_model=AdminScreeningDisputeList)
async def list_screening_disputes(
    _admin: AdminDep,
    session: SessionDep,
    status: Annotated[Literal["pending", "resolved", "all"], Query()] = "pending",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AdminScreeningDisputeList:
    stmt = (
        select(ScreeningDispute, Agent, ScreeningQuarantine)
        .join(Agent, Agent.agent_id == ScreeningDispute.agent_id)
        .join(
            ScreeningQuarantine,
            ScreeningQuarantine.quarantine_id == ScreeningDispute.quarantine_id,
        )
        .order_by(ScreeningDispute.created_at, ScreeningDispute.dispute_id)
        .offset(offset)
        .limit(limit)
    )
    count_stmt = select(func.count()).select_from(ScreeningDispute)
    if status != "all":
        stmt = stmt.where(ScreeningDispute.status == status)
        count_stmt = count_stmt.where(ScreeningDispute.status == status)
    rows = (await session.execute(stmt)).all()
    history = await _resolution_history(
        session, [dispute.quarantine_id for dispute, _agent, _quarantine in rows]
    )
    return AdminScreeningDisputeList(
        items=[
            _dispute_item(
                dispute,
                agent,
                quarantine,
                history[dispute.quarantine_id],
            )
            for dispute, agent, quarantine in rows
        ],
        count=int((await session.scalar(count_stmt)) or 0),
    )


@router.post(
    "/screening-disputes/{dispute_id}/resolve",
    response_model=AdminScreeningDisputeResolveResponse,
)
async def resolve_screening_dispute(
    dispute_id: UUID,
    payload: AdminScreeningDisputeResolveRequest,
    _admin: AdminDep,
    session: SessionDep,
    chain: ChainDep,
    generator: GeneratorDep,
    x_admin_actor: Annotated[str | None, Header()] = None,
) -> AdminScreeningDisputeResolveResponse:
    if x_admin_actor is None or not 1 <= len(x_admin_actor) <= 120:
        raise HTTPException(status_code=422, detail="X-Admin-Actor is required")

    new_dataset: DatasetPin | None = None
    if payload.resolution == "release":
        existing = await session.get(ScreeningDispute, dispute_id)
        quarantine_id = existing.quarantine_id if existing is not None else None
        await session.rollback()
        if quarantine_id is None:
            raise HTTPException(status_code=404, detail="dispute not found")
        new_dataset = await _prepare_release_dataset(
            session, chain, generator, quarantine_id
        )

    async with session.begin():
        dispute = await session.scalar(
            select(ScreeningDispute)
            .where(ScreeningDispute.dispute_id == dispute_id)
            .with_for_update()
        )
        if dispute is None:
            raise HTTPException(status_code=404, detail="dispute not found")
        quarantine = await session.scalar(
            select(ScreeningQuarantine)
            .where(ScreeningQuarantine.quarantine_id == dispute.quarantine_id)
            .with_for_update()
        )
        agent = await session.scalar(
            select(Agent).where(Agent.agent_id == dispute.agent_id).with_for_update()
        )
        if quarantine is None or agent is None:
            raise HTTPException(status_code=404, detail="disputed submission not found")
        if dispute.status != "pending":
            raise HTTPException(status_code=409, detail="dispute is already resolved")
        if (
            agent.status != AgentStatus.REJECTED
            or quarantine.status != "resolved"
            or quarantine.resolution != "reject"
        ):
            raise HTTPException(
                status_code=409,
                detail="the disputed rejection is no longer current",
            )

        now = datetime.now(UTC)
        if payload.resolution == "release":
            agent.status = AgentStatus.EVALUATING
            agent.screening_reason = payload.reason
            _apply_dataset(agent, new_dataset)
            quarantine.resolved_at = now
            quarantine.resolved_by = x_admin_actor
            quarantine.resolution = "release"
            quarantine.resolution_reason = payload.reason
            session.add(
                ScreeningQuarantineResolution(
                    resolution_id=uuid4(),
                    quarantine_id=quarantine.quarantine_id,
                    resolution="release",
                    reason=payload.reason,
                    actor=x_admin_actor,
                    created_at=now,
                )
            )
        dispute.status = "resolved"
        dispute.resolved_at = now
        dispute.resolved_by = x_admin_actor
        dispute.resolution = payload.resolution
        dispute.resolution_reason = payload.reason

    history = await _resolution_history(session, [dispute.quarantine_id])
    return AdminScreeningDisputeResolveResponse(
        dispute=_dispute_item(
            dispute,
            agent,
            quarantine,
            history[dispute.quarantine_id],
        ),
        agent_status=agent.status,
    )


@router.get("/screening-submissions", response_model=AdminScreeningSubmissionList)
async def list_screening_submissions(
    _admin: AdminDep,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AdminScreeningSubmissionList:
    """Return private screening history without source or artifact URLs."""
    total = int((await session.scalar(select(func.count()).select_from(Agent))) or 0)
    agents = (
        (
            await session.execute(
                select(Agent)
                .order_by(Agent.created_at.desc(), Agent.agent_id.desc())
                .offset(offset)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    agent_ids = [agent.agent_id for agent in agents]
    attempts_by_agent: dict[UUID, list[AdminScreeningAttempt]] = defaultdict(list)
    if agent_ids:
        attempts = (
            (
                await session.execute(
                    select(ScreeningAttempt)
                    .where(ScreeningAttempt.agent_id.in_(agent_ids))
                    .order_by(
                        ScreeningAttempt.started_at.desc(),
                        ScreeningAttempt.attempt_id.desc(),
                    )
                )
            )
            .scalars()
            .all()
        )
        duplicate_ids = {
            attempt.duplicate_of
            for attempt in attempts
            if attempt.duplicate_of is not None
        }
        duplicate_agents = {
            duplicate.agent_id: duplicate
            for duplicate in await session.scalars(
                select(Agent).where(Agent.agent_id.in_(duplicate_ids))
            )
        }
        for attempt in attempts:
            duplicate = (
                duplicate_agents.get(attempt.duplicate_of)
                if attempt.duplicate_of is not None
                else None
            )
            attempts_by_agent[attempt.agent_id].append(
                AdminScreeningAttempt(
                    attempt_id=attempt.attempt_id,
                    policy_version=attempt.policy_version,
                    status=attempt.status,  # type: ignore[arg-type]
                    screener_hotkey=attempt.screener_hotkey,
                    started_at=attempt.started_at,
                    deadline=attempt.deadline,
                    finished_at=attempt.finished_at,
                    reason=attempt.public_reason,
                    reason_code=attempt.reason_code,
                    duplicate_of=attempt.duplicate_of,
                    duplicate_name=duplicate.name if duplicate is not None else None,
                    duplicate_version=(
                        duplicate.version if duplicate is not None else None
                    ),
                )
            )
    return AdminScreeningSubmissionList(
        count=total,
        items=[
            AdminScreeningSubmission(
                agent_id=agent.agent_id,
                miner_hotkey=agent.miner_hotkey,
                agent_name=agent.name,
                agent_version=agent.version,
                artifact_sha256=agent.sha256,
                agent_status=agent.status,
                screening_policy_version=agent.screening_policy_version,
                screening_reason=agent.screening_reason,
                screening_reason_code=agent.screening_reason_code,
                submitted_at=agent.created_at,
                attempts=attempts_by_agent[agent.agent_id],
            )
            for agent in agents
        ],
    )


@router.post(
    "/screening-submissions/{agent_id}/rescreen",
    response_model=AdminScreeningRescreenResponse,
)
async def rescreen_rejected_submission(
    agent_id: UUID,
    payload: AdminScreeningRescreenRequest,
    _admin: AdminDep,
    session: SessionDep,
    x_admin_actor: Annotated[str | None, Header()] = None,
) -> AdminScreeningRescreenResponse:
    """Return one rejected submission to the queue without rewriting history."""
    if x_admin_actor is None or not 1 <= len(x_admin_actor) <= 120:
        raise HTTPException(status_code=422, detail="X-Admin-Actor is required")
    async with session.begin():
        agent = await session.scalar(
            select(Agent).where(Agent.agent_id == agent_id).with_for_update()
        )
        if agent is None:
            raise HTTPException(status_code=404, detail="agent not found")
        if agent.sha256 != payload.expected_sha256:
            raise HTTPException(status_code=409, detail="artifact identity changed")
        score_count = int(
            await session.scalar(
                select(func.count())
                .select_from(Score)
                .where(Score.agent_id == agent_id)
            )
            or 0
        )
        if score_count != payload.expected_score_count:
            raise HTTPException(status_code=409, detail="score count changed")
        running_attempt = await session.scalar(
            select(ScreeningAttempt.attempt_id).where(
                ScreeningAttempt.agent_id == agent_id,
                ScreeningAttempt.status == "running",
            )
        )
        if running_attempt is not None:
            raise HTTPException(status_code=409, detail="screening attempt is active")
        if agent.status != AgentStatus.REJECTED:
            raise HTTPException(status_code=409, detail="submission is not rejected")
        agent.status = AgentStatus.SCREENING_FAILED
        agent.screening_reason = "Operator requested a screening retry"
    logger.info(
        "admin_actor=%s requested rescreen agent_id=%s reason=%s",
        x_admin_actor,
        agent_id,
        payload.reason,
    )
    return AdminScreeningRescreenResponse(
        agent_id=agent_id, agent_status=AgentStatus.SCREENING_FAILED
    )


@router.get(
    "/screening-submissions/{agent_id}/artifact",
    response_model=ArtifactResponse,
)
async def get_screening_artifact(
    agent_id: UUID,
    _admin: AdminDep,
    session: SessionDep,
    storage: StorageDep,
    x_admin_actor: Annotated[str | None, Header()] = None,
) -> ArtifactResponse:
    """Issue an audited five-minute artifact URL to an authenticated operator."""
    if x_admin_actor is None or not 1 <= len(x_admin_actor) <= 120:
        raise HTTPException(status_code=422, detail="X-Admin-Actor is required")
    agent = await session.get(Agent, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="agent not found")
    expires_in = 300
    url = await storage.presigned_get_url(
        key=f"{agent_id}/agent.tar.gz", expires_in=expires_in
    )
    logger.info(
        "admin_actor=%s issued screening artifact url for agent_id=%s",
        x_admin_actor,
        agent_id,
    )
    return ArtifactResponse(
        agent_id=agent_id,
        sha256=agent.sha256,
        download_url=url,
        expires_at=datetime.now(UTC) + timedelta(seconds=expires_in),
    )


async def _load_inspector(
    agent_id: UUID, session: AsyncSession, storage: S3StorageClient
) -> tuple[Agent, TarSourceInspector]:
    """Fetch the stored tarball, verify its digest, and open a bounded reader."""
    agent = await session.get(Agent, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="agent not found")
    try:
        tar_bytes = await storage.get_object(
            key=f"{agent_id}/agent.tar.gz", max_bytes=MAX_TARBALL_BYTES
        )
    except ObjectDownloadFailedError as error:
        raise HTTPException(
            status_code=502, detail="artifact is unavailable in storage"
        ) from error

    def _verify_and_open() -> TarSourceInspector:
        # Digest verification and archive characterization are CPU-bound over
        # attacker-supplied bytes; keep them off the event loop.
        if hashlib.sha256(tar_bytes).hexdigest() != agent.sha256:
            raise HTTPException(
                status_code=502, detail="stored artifact does not match its digest"
            )
        return TarSourceInspector(tar_bytes)

    try:
        return agent, await asyncio.to_thread(_verify_and_open)
    except SourceInspectError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get(
    "/screening-submissions/{agent_id}/source-files",
    response_model=AdminSourceListing,
)
async def list_screening_source_files(
    agent_id: UUID,
    _admin: AdminDep,
    session: SessionDep,
    storage: StorageDep,
    x_admin_actor: Annotated[str | None, Header()] = None,
) -> AdminSourceListing:
    """Audited, bounded file inventory of one submission tarball."""
    if x_admin_actor is None or not 1 <= len(x_admin_actor) <= 120:
        raise HTTPException(status_code=422, detail="X-Admin-Actor is required")
    agent, inspector = await _load_inspector(agent_id, session, storage)
    logger.info(
        "admin_actor=%s listed screening source for agent_id=%s",
        x_admin_actor,
        agent_id,
    )
    listing = await asyncio.to_thread(inspector.listing)
    return AdminSourceListing(
        agent_id=agent_id,
        artifact_sha256=agent.sha256,
        **listing,  # type: ignore[arg-type]
    )


@router.get(
    "/screening-submissions/{agent_id}/source-file",
    response_model=AdminSourceExcerpt,
)
async def read_screening_source_file(
    agent_id: UUID,
    _admin: AdminDep,
    session: SessionDep,
    storage: StorageDep,
    path: Annotated[str, Query(min_length=1, max_length=240)],
    start_line: Annotated[int, Query(ge=1)] = 1,
    end_line: Annotated[int, Query(ge=1)] = MAX_READ_LINES,
    x_admin_actor: Annotated[str | None, Header()] = None,
) -> AdminSourceExcerpt:
    """Audited, bounded line excerpt from one submission source file.

    Pairs with the source-review finding's ``path:line`` evidence so the
    operator can see exactly the flagged code without a full download.
    """
    if x_admin_actor is None or not 1 <= len(x_admin_actor) <= 120:
        raise HTTPException(status_code=422, detail="X-Admin-Actor is required")
    _agent, inspector = await _load_inspector(agent_id, session, storage)
    try:
        excerpt = await asyncio.to_thread(inspector.read, path, start_line, end_line)
    except SourceInspectError as error:
        status = 404 if error.code == "file-not-found" else 422
        raise HTTPException(status_code=status, detail=str(error)) from error
    logger.info(
        "admin_actor=%s read screening source agent_id=%s path=%s lines=%s-%s",
        x_admin_actor,
        agent_id,
        path,
        excerpt["start_line"],
        excerpt["end_line"],
    )
    return AdminSourceExcerpt(agent_id=agent_id, **excerpt)  # type: ignore[arg-type]


__all__ = ["router"]
