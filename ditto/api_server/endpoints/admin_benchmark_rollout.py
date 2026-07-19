"""Audited operator control for rolling benchmark top-five qualification."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, StringConstraints
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.benchmark_contract import benchmark_contract
from ditto.api_server.benchmark_rollout import refresh_rolling_qualification
from ditto.api_server.datapipeline import DatasetGenerator
from ditto.api_server.dependencies import get_dataset_generator, get_session
from ditto.api_server.endpoints.admin_quarantine import require_admin
from ditto.db.models import Agent
from ditto.db.queries.benchmark_rollout import (
    DatasetPin,
    RolloutConflictError,
    RolloutSnapshotMember,
    active_bench_version,
    create_rollout_snapshot,
    rollout_for_desired_version,
    rollout_state,
    supersede_open_rollout,
)
from ditto.db.queries.scores import list_eligible_ledger

router = APIRouter(prefix="/admin/benchmark-rollout", tags=["admin"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
GeneratorDep = Annotated[DatasetGenerator, Depends(get_dataset_generator)]
AdminDep = Annotated[None, Depends(require_admin)]
MINIMUM_ROLLOUT_START_VALIDATORS = 2

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


async def _require_rollout_start_capacity(
    session: AsyncSession, *, now: datetime, desired_version: int
) -> dict[str, object]:
    """Fail closed until two independently verified target scorers are online."""
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


@router.post("/{desired_version}")
async def start_rollout(
    _: AdminDep,
    session: SessionDep,
    generator: GeneratorDep,
    desired_version: str,
) -> dict[str, object]:
    """Seed rolling qualification with the current top five and target datasets."""
    target = _parse_desired_version(desired_version)
    from_version = await active_bench_version(session)
    # Idempotence first: a rollout already targeting this version is returned
    # as-is, whatever it started from. Only then does the forward-only guard
    # apply, so re-POSTing an already-activated version is not a 409.
    existing = await rollout_for_desired_version(session, desired_version=target)
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
        await refresh_rolling_qualification(
            session, generator=generator, now=datetime.now(UTC)
        )
        return await rollout_state(session, capability_version=target)
    now = datetime.now(UTC)
    await _require_rollout_start_capacity(session, now=now, desired_version=target)
    ledger = [row for row in await list_eligible_ledger(session) if row.eligible][:5]
    if len(ledger) != 5:
        raise HTTPException(
            status_code=409,
            detail=(
                f"benchmark v{target} rollout requires five eligible distinct miners"
            ),
        )
    members: list[RolloutSnapshotMember] = []
    pins: dict = {}
    for row in ledger:
        agent = await session.get(Agent, row.agent_id)
        assert agent is not None
        if (
            agent.dataset_seed is None
            or agent.dataset_sha256 is None
            or agent.dataset_run_size is None
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"cohort agent {agent.agent_id} has no pinned "
                    f"v{from_version} dataset"
                ),
            )
        sha256 = await generator.generate(agent.dataset_seed, bench_version=target)
        members.append(
            RolloutSnapshotMember(
                agent_id=agent.agent_id,
                miner_hotkey=agent.miner_hotkey,
                composite=row.composite,
            )
        )
        pins[agent.agent_id] = DatasetPin(
            seed=agent.dataset_seed,
            sha256=sha256,
            run_size=agent.dataset_run_size,
            seed_block=agent.dataset_seed_block,
            seed_block_hash=agent.dataset_seed_block_hash,
        )
    try:
        await create_rollout_snapshot(
            session,
            members=members,
            datasets=pins,
            now=now,
            from_version=from_version,
            desired_version=target,
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
