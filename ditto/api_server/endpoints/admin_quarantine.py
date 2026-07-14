"""Authenticated operator API for durable screening quarantines."""

from __future__ import annotations

import logging
import secrets
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.admin_quarantine import (
    AdminQuarantineItem,
    AdminQuarantineList,
    AdminQuarantineResolveRequest,
    AdminQuarantineResolveResponse,
    AdminScreeningAttempt,
    AdminScreeningSubmission,
    AdminScreeningSubmissionList,
)
from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.validator import ArtifactResponse
from ditto.api_server.datapipeline import DatasetGenerator
from ditto.api_server.dependencies import (
    get_dataset_generator,
    get_session,
    get_storage_client,
)
from ditto.api_server.endpoints.screener import _derive_dataset_seed
from ditto.api_server.endpoints.validator import ChainDep
from ditto.api_server.storage import S3StorageClient
from ditto.db.models import Agent, ScreeningAttempt, ScreeningQuarantine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
GeneratorDep = Annotated[DatasetGenerator, Depends(get_dataset_generator)]
StorageDep = Annotated[S3StorageClient, Depends(get_storage_client)]


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


def _item(row: ScreeningQuarantine, agent: Agent) -> AdminQuarantineItem:
    return AdminQuarantineItem(
        quarantine_id=row.quarantine_id,
        agent_id=row.agent_id,
        attempt_id=row.attempt_id,
        miner_hotkey=agent.miner_hotkey,
        agent_name=agent.name,
        artifact_sha256=agent.sha256,
        policy_version=row.policy_version,
        manifest_digest=row.manifest_digest,
        finding_digest=row.finding_digest,
        reason_code=row.reason_code,
        status=row.status,  # type: ignore[arg-type]
        created_at=row.created_at,
        resolved_at=row.resolved_at,
        resolved_by=row.resolved_by,
        resolution=row.resolution,  # type: ignore[arg-type]
        resolution_reason=row.resolution_reason,
    )


@router.get("/screening-quarantines", response_model=AdminQuarantineList)
async def list_quarantines(
    _admin: AdminDep,
    session: SessionDep,
    status: Annotated[Literal["active", "resolved", "all"], Query()] = "active",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AdminQuarantineList:
    stmt = (
        select(ScreeningQuarantine, Agent)
        .join(Agent, Agent.agent_id == ScreeningQuarantine.agent_id)
        .order_by(
            ScreeningQuarantine.created_at.desc(),
            ScreeningQuarantine.quarantine_id.desc(),
        )
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
    items = [_item(quarantine, agent) for quarantine, agent in rows]
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
    return _item(*result)


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

    new_dataset: tuple[int, str, str, int | None, str | None] | None = None
    if payload.resolution == "release" and generator.run_size is not None:
        existing = await session.scalar(
            select(Agent)
            .join(
                ScreeningQuarantine,
                ScreeningQuarantine.agent_id == Agent.agent_id,
            )
            .where(ScreeningQuarantine.quarantine_id == quarantine_id)
        )
        await session.rollback()
        if existing is None:
            raise HTTPException(status_code=404, detail="quarantine not found")
        if existing.dataset_seed is None:
            seed, block_number, block_hash = await _derive_dataset_seed(
                chain, existing.agent_id
            )
            dataset_sha256 = await generator.generate(seed)
            new_dataset = (
                seed,
                dataset_sha256,
                generator.run_size,
                block_number,
                block_hash,
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
        if quarantine.status != "active" or agent.status != AgentStatus.QUARANTINED:
            raise HTTPException(status_code=409, detail="quarantine is not active")

        target = {
            "release": AgentStatus.EVALUATING,
            "rescreen": AgentStatus.SCREENING_FAILED,
            "reject": AgentStatus.REJECTED,
        }[payload.resolution]
        agent.status = target
        agent.screening_reason = payload.reason
        if new_dataset is not None and agent.dataset_seed is None:
            (
                agent.dataset_seed,
                agent.dataset_sha256,
                agent.dataset_run_size,
                agent.dataset_seed_block,
                agent.dataset_seed_block_hash,
            ) = new_dataset
        quarantine.status = "resolved"
        quarantine.resolved_at = datetime.now(UTC)
        quarantine.resolved_by = x_admin_actor
        quarantine.resolution = payload.resolution
        quarantine.resolution_reason = payload.reason

    return AdminQuarantineResolveResponse(
        quarantine=_item(quarantine, agent), agent_status=agent.status
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
        for attempt in attempts:
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
                )
            )
    return AdminScreeningSubmissionList(
        count=total,
        items=[
            AdminScreeningSubmission(
                agent_id=agent.agent_id,
                miner_hotkey=agent.miner_hotkey,
                agent_name=agent.name,
                artifact_sha256=agent.sha256,
                agent_status=agent.status,
                screening_policy_version=agent.screening_policy_version,
                screening_reason=agent.screening_reason,
                submitted_at=agent.created_at,
                attempts=attempts_by_agent[agent.agent_id],
            )
            for agent in agents
        ],
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


__all__ = ["router"]
