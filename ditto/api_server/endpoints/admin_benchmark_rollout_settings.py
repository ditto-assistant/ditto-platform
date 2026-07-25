"""Audited operator control for benchmark-rollout sizing.

Append-only revisions of the rollout policy -- today one knob,
``rescore_cohort_size``, how many inherited agents a new benchmark transition
re-scores. It has always been a hard-coded ten; as the subnet scales an operator
raises it toward the persisted ceiling of twenty-five so more agents carry a
target-version score into the new era, with no redeploy.

Deliberately a **separate router prefix** from ``admin_benchmark_rollout``:
that router owns ``POST /admin/benchmark-rollout/{desired_version}``, so a
``/settings`` sub-path there would sit in the same namespace as a version
segment and depend on declaration order to not be parsed as one. It is also a
different kind of control -- rollout *activation* is a typed-confirmation UI
action, while this is subnet policy that backroom may set over MCP.

The read is intentionally not cached: the value is consulted exactly once per
rollout, at start, so a cache could only hide a fresh revision from the very
next start.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.benchmark_rollout_settings import (
    AdminBenchmarkRolloutSettingsRequest,
    AdminBenchmarkRolloutSettingsResponse,
    BenchmarkRolloutSettings,
    BenchmarkRolloutSettingsRevision,
    EffectiveBenchmarkRolloutSettings,
)
from ditto.api_server.benchmark_rollout_settings import (
    DEFAULT_SETTINGS,
    settings_from_row,
)
from ditto.api_server.dependencies import get_session
from ditto.api_server.endpoints.admin_quarantine import require_admin
from ditto.db.models import BenchmarkRolloutSettingsRevision as RevisionRow
from ditto.db.queries.benchmark_rollout import open_rollout
from ditto.db.queries.benchmark_rollout_settings import (
    GLOBAL_SCOPE,
    insert_benchmark_rollout_settings_revision,
    latest_benchmark_rollout_settings_revision,
    list_benchmark_rollout_settings_revisions,
)

router = APIRouter(prefix="/admin/benchmark-rollout-settings", tags=["admin"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
AdminDep = Annotated[None, Depends(require_admin)]
CONFIRMATION = "APPLY BENCHMARK ROLLOUT SETTINGS"


def _checksum(settings: BenchmarkRolloutSettings) -> str:
    encoded = json.dumps(
        settings.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _revision(row: RevisionRow) -> BenchmarkRolloutSettingsRevision:
    return BenchmarkRolloutSettingsRevision(
        revision=row.revision,
        parent_revision=row.parent_revision,
        scope=row.scope,
        settings=BenchmarkRolloutSettings.model_validate(row.settings),
        reason=row.reason,
        actor=row.actor,
        created_at=row.created_at,
        checksum=row.checksum,
    )


async def _effective(
    session: AsyncSession, latest: RevisionRow | None
) -> EffectiveBenchmarkRolloutSettings:
    """What the next start will freeze, plus what the open rollout already did.

    Reporting the open rollout's frozen target beside the configured one is how
    an operator sees, without reading the source, that raising the policy does
    not resize the transition currently in flight.
    """
    settings = settings_from_row(latest)
    rollout = await open_rollout(session)
    frozen = rollout.rescore_cohort_target if rollout is not None else None
    return EffectiveBenchmarkRolloutSettings(
        revision=latest.revision if latest is not None else 0,
        scope=latest.scope if latest is not None else GLOBAL_SCOPE,
        settings=settings,
        checksum=latest.checksum if latest is not None else "",
        source="revision" if latest is not None else "default",
        open_rollout_desired_version=(
            rollout.desired_version if rollout is not None else None
        ),
        open_rollout_rescore_cohort_target=frozen,
        open_rollout_overrides_setting=(
            frozen is not None and frozen != settings.rescore_cohort_size
        ),
    )


@router.get("", response_model=AdminBenchmarkRolloutSettingsResponse)
async def get_settings(
    _admin: AdminDep, session: SessionDep
) -> AdminBenchmarkRolloutSettingsResponse:
    """Current policy, append-only history, the default, and what is in force."""
    latest = await latest_benchmark_rollout_settings_revision(session)
    history = await list_benchmark_rollout_settings_revisions(session)
    return AdminBenchmarkRolloutSettingsResponse(
        current=[_revision(latest)] if latest is not None else [],
        history=[_revision(row) for row in history],
        default=DEFAULT_SETTINGS,
        effective=await _effective(session, latest),
    )


@router.post("", response_model=BenchmarkRolloutSettingsRevision)
async def create_settings_revision(
    payload: AdminBenchmarkRolloutSettingsRequest,
    _admin: AdminDep,
    session: SessionDep,
) -> BenchmarkRolloutSettingsRevision:
    """Append one optimistic, confirmation-gated revision.

    Takes effect at the NEXT ``POST /admin/benchmark-rollout/{version}``. An
    already-open rollout keeps the target it froze at start.
    """
    if payload.scope != GLOBAL_SCOPE:
        raise HTTPException(
            status_code=422,
            detail="benchmark rollout policy is subnet-global; scope must be '*'",
        )
    if payload.confirmation != CONFIRMATION:
        raise HTTPException(
            status_code=409,
            detail=f"confirmation must be exactly {CONFIRMATION}",
        )
    latest = await latest_benchmark_rollout_settings_revision(
        session, scope=payload.scope
    )
    actual_revision = latest.revision if latest is not None else 0
    if payload.expected_revision != actual_revision:
        raise HTTPException(
            status_code=409,
            detail=(
                "benchmark rollout settings changed; refresh before applying "
                f"(expected {payload.expected_revision}, current {actual_revision})"
            ),
        )
    try:
        row = await insert_benchmark_rollout_settings_revision(
            session,
            parent_revision=actual_revision,
            scope=payload.scope,
            settings=payload.settings.model_dump(mode="json"),
            checksum=_checksum(payload.settings),
            reason=payload.reason.strip(),
            actor=payload.actor.strip(),
        )
        await session.commit()
    except IntegrityError as error:
        await session.rollback()
        raise HTTPException(
            status_code=409,
            detail=(
                "benchmark rollout settings changed concurrently; refresh and retry"
            ),
        ) from error
    await session.refresh(row)
    return _revision(row)
