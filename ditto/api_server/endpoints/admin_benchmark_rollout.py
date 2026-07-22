"""Audited operator control for bounded benchmark cohort qualification."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, StringConstraints
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.benchmark_contract import (
    benchmark_contract,
    benchmark_contracts,
)
from ditto.api_server.benchmark_rollout import (
    qualification_candidate,
    refresh_rolling_qualification,
)
from ditto.api_server.datapipeline import DataPipelineError, DatasetGenerator
from ditto.api_server.dependencies import get_dataset_generator, get_session
from ditto.api_server.endpoints.admin_quarantine import require_admin
from ditto.db.queries.benchmark_rollout import (
    DatasetPin,
    RolloutConflictError,
    active_bench_version,
    authority_selection_state,
    create_rollout_snapshot,
    historical_rescore_cohort,
    rollout_for_desired_version,
    rollout_state,
    select_active_bench_version,
    supersede_open_rollout,
)

router = APIRouter(prefix="/admin/benchmark-rollout", tags=["admin"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
GeneratorDep = Annotated[DatasetGenerator, Depends(get_dataset_generator)]
AdminDep = Annotated[None, Depends(require_admin)]
MINIMUM_ROLLOUT_START_VALIDATORS = 1

_Reason = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)
]
_Actor = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)
]


class AdminRolloutSupersedeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: _Reason
    actor: _Actor = "admin_api"
    confirmation: str


class AdminRolloutStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: _Reason
    actor: _Actor = "admin_api"
    confirmation: str
    expected_active_version: int


class AdminActiveContractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: _Reason
    actor: _Actor = "admin_api"
    confirmation: str
    expected_active_version: int


def _parse_desired_version(desired_version: str) -> int:
    """Accept both the legacy ``/v3`` path and a bare ``/4``.

    Keeping the ``v`` prefix parseable is what preserves every already-deployed
    caller pinned to ``/admin/benchmark-rollout/v3``.
    """
    text = (
        desired_version[1:] if desired_version[:1].lower() == "v" else desired_version
    )
    if not text.isdigit():
        raise HTTPException(status_code=404, detail="unknown benchmark rollout version")
    version = int(text)
    try:
        benchmark_contract(version)
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"benchmark version {version} has no shipped contract",
        ) from exc
    return version


def _generator_unavailable(target: int, exc: DataPipelineError) -> HTTPException:
    """502 for a generate-service that cannot render the target benchmark.

    The generator is deployed separately and pinned to a datagen release, so it
    lags this API whenever a new benchmark ships. Left unhandled this reached the
    operator as a bare 500 naming nothing -- not the dependency, not the version,
    not the fix.
    """
    return HTTPException(
        status_code=502,
        detail=(
            f"dataset generator could not render benchmark v{target} ({exc}). "
            f"Deploy the generate-service at a datagen release that ships "
            f"v{target} before starting this rollout."
        ),
    )


async def _require_rollout_start_capacity(
    session: AsyncSession, *, now: datetime, desired_version: int
) -> dict[str, object]:
    """Fail closed until one independently verified target scorer is online."""
    state = await rollout_state(session, now=now, capability_version=desired_version)
    capable = int(state["canary_capable_validator_count"])
    if capable < MINIMUM_ROLLOUT_START_VALIDATORS:
        raise HTTPException(
            status_code=409,
            detail=(
                f"benchmark v{desired_version} rollout requires at least "
                f"{MINIMUM_ROLLOUT_START_VALIDATORS} fresh, identity-matched "
                "v8 scorer validators"
            ),
        )
    return state


def _require_confirmation(actual: str, *, action: str, version: int) -> None:
    expected = f"{action} BENCHMARK V{version}"
    if actual != expected:
        raise HTTPException(
            status_code=409,
            detail=f'type "{expected}" exactly to confirm this operation',
        )


@router.get("")
async def get_rollout_control(
    _: AdminDep,
    session: SessionDep,
) -> dict[str, object]:
    """Advertise shipped targets and durable state without changing either."""
    state = await rollout_state(session)
    active = int(state["active_version"])
    contracts = benchmark_contracts()
    capability_counts = {
        contract.version: int(
            (await rollout_state(session, capability_version=contract.version))[
                "canary_capable_validator_count"
            ]
        )
        for contract in contracts
    }
    return {
        **state,
        "contracts": [
            {
                "version": contract.version,
                "minimum_screening_policy_version": (
                    contract.minimum_screening_policy_version
                ),
                "requires_screened_image": contract.requires_screened_image,
                "capable_validator_count": capability_counts[contract.version],
            }
            for contract in contracts
        ],
        "available_target_versions": [
            contract.version for contract in contracts if contract.version > active
        ],
        "active_contract_candidates": [
            await authority_selection_state(session, bench_version=contract.version)
            for contract in contracts
            if contract.version > active
        ],
    }


@router.get("/{desired_version}")
async def get_rollout(
    _: AdminDep,
    session: SessionDep,
    desired_version: str,
) -> dict[str, object]:
    """Return rollout telemetry without starting or refreshing qualification."""
    target = _parse_desired_version(desired_version)
    return await rollout_state(session, capability_version=target)


@router.post("/{desired_version}/supersede")
async def supersede_rollout(
    _: AdminDep,
    session: SessionDep,
    desired_version: str,
    payload: AdminRolloutSupersedeRequest,
) -> dict[str, object]:
    """Terminally abandon the open rollout so a newer one can be opened.

    Refuses an activated rollout: activation already moved chain weights and
    released the superseded corpus, so it is history, not state.
    """
    target = _parse_desired_version(desired_version)
    _require_confirmation(payload.confirmation, action="SUPERSEDE", version=target)
    now = datetime.now(UTC)
    try:
        rollout = await supersede_open_rollout(
            session, actor=payload.actor, reason=payload.reason, now=now
        )
    except RolloutConflictError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if rollout is None:
        await session.rollback()
        raise HTTPException(
            status_code=409, detail="no open benchmark rollout to supersede"
        )
    if rollout.desired_version != target:
        await session.rollback()
        raise HTTPException(
            status_code=409,
            detail=(
                f"the open rollout targets v{rollout.desired_version}, not v{target}"
            ),
        )
    await session.commit()
    return await rollout_state(session, capability_version=target)


@router.post("/{desired_version}/select-active")
async def select_active_contract(
    _: AdminDep,
    session: SessionDep,
    desired_version: str,
    payload: AdminActiveContractRequest,
) -> dict[str, object]:
    """Select a fully qualified superseded contract as weight authority."""
    target = _parse_desired_version(desired_version)
    _require_confirmation(payload.confirmation, action="ACTIVATE", version=target)
    current = await active_bench_version(session)
    if payload.expected_active_version != current:
        raise HTTPException(
            status_code=409,
            detail=(
                "active benchmark changed: expected "
                f"v{payload.expected_active_version}, found v{current}"
            ),
        )
    try:
        await select_active_bench_version(
            session,
            bench_version=target,
            actor=payload.actor,
            reason=payload.reason,
            now=datetime.now(UTC),
        )
    except RolloutConflictError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    await session.commit()
    return await get_rollout_control(None, session)


@router.post("/{desired_version}")
async def start_rollout(
    _: AdminDep,
    session: SessionDep,
    generator: GeneratorDep,
    desired_version: str,
    payload: AdminRolloutStartRequest,
) -> dict[str, object]:
    """Seed the bounded historical rescore cohort and target datasets."""
    target = _parse_desired_version(desired_version)
    _require_confirmation(payload.confirmation, action="START", version=target)
    from_version = await active_bench_version(session)
    if payload.expected_active_version != from_version:
        raise HTTPException(
            status_code=409,
            detail=(
                "active benchmark changed: expected "
                f"v{payload.expected_active_version}, found v{from_version}"
            ),
        )
    # Idempotence first: a rollout already targeting this version is returned
    # as-is, whatever it started from. Only then does the forward-only guard
    # apply, so re-POSTing an already-activated version is not a 409.
    existing = await rollout_for_desired_version(session, desired_version=target)
    if (
        existing is not None
        and existing.status == "superseded"
        and existing.from_version != from_version
    ):
        # A recovered authority creates a new transition lineage. The terminal
        # superseded row remains immutable history; it must not make the fresh
        # active->target transition look idempotently complete.
        existing = None
    if existing is None and target <= from_version:
        raise HTTPException(
            status_code=409,
            detail=(
                f"cannot start benchmark {target} rollout while "
                f"active benchmark is {from_version}"
            ),
        )
    if existing is not None:
        await session.rollback()  # close the read-only autobegin before the service
        # This refresh renders datasets too, so a lagging generate-service fails
        # here exactly as it does on the fresh-start path below. Re-POSTing a
        # version whose rollout already exists is the natural operator retry, so
        # it must not be the one route that still answers with an opaque 500.
        #
        # Unlike validator/screener ingest -- which deliberately swallow this so a
        # renderer outage cannot fail an already-committed score or verdict -- an
        # admin POST exists to report whether the action succeeded, so it surfaces.
        try:
            await refresh_rolling_qualification(
                session, generator=generator, now=datetime.now(UTC)
            )
        except DataPipelineError as exc:
            raise _generator_unavailable(target, exc) from exc
        return await rollout_state(session, capability_version=target)
    now = datetime.now(UTC)
    await _require_rollout_start_capacity(session, now=now, desired_version=target)
    members = await historical_rescore_cohort(session, source_version=from_version)
    if len(members) < 5:
        raise HTTPException(
            status_code=409,
            detail=(
                f"benchmark v{target} rollout requires five eligible distinct miners"
            ),
        )
    pins: dict = {}
    seed_sources: dict[str, str] = {}
    for member in members:
        candidate, blocker = await qualification_candidate(
            session,
            source_bench_version=from_version,
            target_bench_version=target,
            member=member,
            generator_run_size=generator.run_size,
        )
        if candidate is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"cohort agent {member.agent_id} cannot qualify for "
                    f"v{target}: {blocker}"
                ),
            )
        try:
            sha256 = candidate.existing_sha256 or await generator.generate(
                candidate.seed, bench_version=target
            )
        except DataPipelineError as exc:
            raise _generator_unavailable(target, exc) from exc
        pins[member.agent_id] = DatasetPin(
            seed=candidate.seed,
            sha256=sha256,
            run_size=candidate.run_size,
            seed_block=candidate.seed_block,
            seed_block_hash=candidate.seed_block_hash,
        )
        seed_sources[str(member.agent_id)] = candidate.seed_source
    try:
        await create_rollout_snapshot(
            session,
            members=members,
            datasets=pins,
            now=now,
            from_version=from_version,
            desired_version=target,
            audit_context={
                "origin": "admin",
                "actor": payload.actor,
                "reason": payload.reason,
                "from_version": from_version,
                "desired_version": target,
                "seed_sources": seed_sources,
            },
        )
    except RolloutConflictError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=409,
            detail=(
                f"{exc}. Supersede it first with "
                f"POST /admin/benchmark-rollout/<version>/supersede."
            ),
        ) from exc
    await session.commit()
    return await rollout_state(session, capability_version=target)
