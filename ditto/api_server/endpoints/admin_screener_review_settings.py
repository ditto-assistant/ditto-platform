"""Audited operator control for per-instance L2/L3 reviewer settings."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.screener_review_settings import (
    AdminScreenerReviewSettingsRequest,
    AdminScreenerReviewSettingsResponse,
    ScreenerReviewSettings,
)
from ditto.api_models.screener_review_settings import (
    ScreenerReviewSettingsRevision as RevisionModel,
)
from ditto.api_server.dependencies import get_session
from ditto.api_server.endpoints.admin_quarantine import require_admin
from ditto.db.models import ScreenerHeartbeat, ScreenerReviewSettingsRevision

router = APIRouter(prefix="/admin/screener-review-settings", tags=["admin"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
AdminDep = Annotated[None, Depends(require_admin)]
_SCOPE_RE = re.compile(r"^(?:\*|[a-zA-Z0-9._-]{1,63})$")


def _checksum(settings: ScreenerReviewSettings) -> str:
    encoded = json.dumps(
        settings.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _revision(row: ScreenerReviewSettingsRevision) -> RevisionModel:
    return RevisionModel(
        revision=row.revision,
        parent_revision=row.parent_revision,
        scope=row.scope,
        settings=ScreenerReviewSettings.model_validate_json(json.dumps(row.settings)),
        reason=row.reason,
        actor=row.actor,
        created_at=row.created_at,
        checksum=row.checksum,
    )


async def _rows(session: AsyncSession) -> list[ScreenerReviewSettingsRevision]:
    return list(
        await session.scalars(
            select(ScreenerReviewSettingsRevision).order_by(
                ScreenerReviewSettingsRevision.revision.desc()
            )
        )
    )


@router.get("", response_model=AdminScreenerReviewSettingsResponse)
async def get_settings(
    _admin: AdminDep,
    session: SessionDep,
) -> AdminScreenerReviewSettingsResponse:
    """Return current settings, append-only history, and addressable instances."""
    rows = await _rows(session)
    current_by_scope: dict[str, ScreenerReviewSettingsRevision] = {}
    for row in rows:
        current_by_scope.setdefault(row.scope, row)
    instances = sorted(
        set(await session.scalars(select(ScreenerHeartbeat.instance_id)))
        | {scope for scope in current_by_scope if scope != "*"}
    )
    return AdminScreenerReviewSettingsResponse(
        current=[_revision(row) for row in current_by_scope.values()],
        history=[_revision(row) for row in rows[:200]],
        known_instances=instances,
    )


@router.post("", response_model=RevisionModel)
async def create_settings_revision(
    payload: AdminScreenerReviewSettingsRequest,
    _admin: AdminDep,
    session: SessionDep,
) -> RevisionModel:
    """Append one optimistic, idempotency-safe settings revision."""
    if not _SCOPE_RE.fullmatch(payload.scope):
        raise HTTPException(status_code=422, detail="invalid screener settings scope")
    expected_confirmation = (
        f"APPLY SCREENER REVIEW {payload.scope} {payload.settings.mode.upper()}"
    )
    if payload.confirmation != expected_confirmation:
        raise HTTPException(
            status_code=409,
            detail=f"confirmation must be exactly {expected_confirmation}",
        )
    latest = await session.scalar(
        select(ScreenerReviewSettingsRevision)
        .where(ScreenerReviewSettingsRevision.scope == payload.scope)
        .order_by(ScreenerReviewSettingsRevision.revision.desc())
        .limit(1)
    )
    actual_revision = latest.revision if latest is not None else 0
    if payload.expected_revision != actual_revision:
        raise HTTPException(
            status_code=409,
            detail=(
                "screener review settings changed; refresh before applying "
                f"(expected {payload.expected_revision}, current {actual_revision})"
            ),
        )
    row = ScreenerReviewSettingsRevision(
        parent_revision=actual_revision,
        scope=payload.scope,
        settings=payload.settings.model_dump(mode="json"),
        checksum=_checksum(payload.settings),
        reason=payload.reason.strip(),
        actor=payload.actor.strip(),
    )
    session.add(row)
    try:
        await session.commit()
    except IntegrityError as error:
        await session.rollback()
        raise HTTPException(
            status_code=409,
            detail=(
                "screener review settings changed concurrently; refresh before applying"
            ),
        ) from error
    await session.refresh(row)
    return _revision(row)
