"""Audited operator start for rolling benchmark-v3 top-five qualification."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_server.benchmark_rollout import refresh_rolling_qualification
from ditto.api_server.datapipeline import DatasetGenerator
from ditto.api_server.dependencies import get_dataset_generator, get_session
from ditto.api_server.endpoints.admin_quarantine import require_admin
from ditto.db.models import Agent
from ditto.db.queries.benchmark_rollout import (
    CANARY_BENCH_VERSION,
    DEFAULT_BENCH_VERSION,
    DatasetPin,
    RolloutSnapshotMember,
    active_bench_version,
    create_rollout_snapshot,
    rollout_for_transition,
    rollout_state,
)
from ditto.db.queries.scores import list_eligible_ledger

router = APIRouter(prefix="/admin/benchmark-rollout", tags=["admin"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
GeneratorDep = Annotated[DatasetGenerator, Depends(get_dataset_generator)]
AdminDep = Annotated[None, Depends(require_admin)]
MINIMUM_V3_START_VALIDATORS = 1


async def _require_v3_start_capacity(
    session: AsyncSession, *, now: datetime
) -> dict[str, object]:
    """Fail closed until one independently verified v3 scorer is online."""
    state = await rollout_state(session, now=now)
    capable = int(state["v3_capable_validator_count"])
    if capable < MINIMUM_V3_START_VALIDATORS:
        raise HTTPException(
            status_code=409,
            detail=(
                "benchmark v3 rollout requires at least one fresh, "
                "identity-matched v8 scorer validator"
            ),
        )
    return state


@router.post("/v3")
async def start_v3_rollout(
    _: AdminDep,
    session: SessionDep,
    generator: GeneratorDep,
) -> dict[str, object]:
    """Seed rolling qualification with the current top five and v3 datasets."""
    existing = await rollout_for_transition(
        session,
        from_version=DEFAULT_BENCH_VERSION,
        desired_version=CANARY_BENCH_VERSION,
    )
    if existing is not None:
        await session.rollback()  # close the read-only autobegin before the service
        await refresh_rolling_qualification(
            session, generator=generator, now=datetime.now(UTC)
        )
        return await rollout_state(session)
    active_version = await active_bench_version(session)
    if active_version != DEFAULT_BENCH_VERSION:
        raise HTTPException(
            status_code=409,
            detail=(
                f"cannot start benchmark {CANARY_BENCH_VERSION} rollout while "
                f"active benchmark is {active_version}"
            ),
        )
    now = datetime.now(UTC)
    await _require_v3_start_capacity(session, now=now)
    ledger = [row for row in await list_eligible_ledger(session) if row.eligible][:5]
    if len(ledger) != 5:
        raise HTTPException(
            status_code=409,
            detail="benchmark v3 rollout requires five eligible distinct miners",
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
                detail=f"cohort agent {agent.agent_id} has no pinned v2 dataset",
            )
        sha256 = await generator.generate(
            agent.dataset_seed, bench_version=CANARY_BENCH_VERSION
        )
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
    await create_rollout_snapshot(
        session,
        members=members,
        datasets=pins,
        now=now,
    )
    await session.commit()
    return await rollout_state(session)
