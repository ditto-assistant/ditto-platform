"""Admin: explain why a submission is (or isn't) leaseable for scoring.

Mirrors the prerequisite checks in ``issue_ticket`` (dataset + screened image +
screening policy + evaluating status) so an operator can see, without DB access,
why an agent sits below quorum with no validator ever picking it up — the
classic "0/3 on v4 but never leased" case, usually a missing v4 dataset or an
unbuilt screened image.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.admin_scoring_readiness import (
    AgentScoringReadiness,
    ScreenedImageReadiness,
)
from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.benchmark_contract import benchmark_contract
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.api_server.dependencies import get_session
from ditto.api_server.endpoints.admin_quarantine import require_admin
from ditto.db.models import Agent, BenchmarkDataset
from ditto.db.queries.benchmark_admission import agent_is_admitted
from ditto.db.queries.benchmark_rollout import active_bench_version

router = APIRouter(prefix="/admin", tags=["admin"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
AdminDep = Annotated[None, Depends(require_admin)]

# The screened-image columns issue_ticket requires all-present before a
# screened-only benchmark will lease the agent (see tickets.py
# ``complete_screened_image``).
_SCREENED_IMAGE_FIELDS = (
    "screened_image_sha256",
    "screened_image_size_bytes",
    "screened_image_id",
    "screened_image_ref",
    "screened_image_upload_id",
    "screened_image_verified_at",
)


@router.get(
    "/agents/{agent_id}/scoring-readiness", response_model=AgentScoringReadiness
)
async def scoring_readiness(
    agent_id: UUID, _admin: AdminDep, session: SessionDep
) -> AgentScoringReadiness:
    agent = await session.scalar(select(Agent).where(Agent.agent_id == agent_id))
    if agent is None:
        raise HTTPException(status_code=404, detail="agent not found")

    active_version = await active_bench_version(session)
    contract = benchmark_contract(active_version)

    missing_fields = [
        field for field in _SCREENED_IMAGE_FIELDS if getattr(agent, field) is None
    ]
    image_complete = not missing_fields
    image_verified = agent.screened_image_verified_at is not None
    policy_ok = (
        agent.screening_policy_version >= contract.minimum_screening_policy_version
    )
    image_eligible = image_complete and policy_ok

    # A versioned dataset is only a prerequisite off the legacy v2 path (see the
    # ``bench_version != 2`` guard in issue_ticket).
    has_versioned_dataset = bool(
        await session.scalar(
            select(
                exists().where(
                    BenchmarkDataset.agent_id == agent_id,
                    BenchmarkDataset.bench_version == active_version,
                )
            )
        )
    )
    benchmark_admitted = await agent_is_admitted(
        session, bench_version=active_version, agent_id=agent_id
    )

    blocking: list[str] = []
    if agent.status != AgentStatus.EVALUATING:
        blocking.append(
            f"agent status is '{agent.status.value}', not evaluating "
            "(only evaluating submissions are leased for scoring)"
        )
    if agent.screening_policy_version < SCREENING_POLICY_VERSION:
        blocking.append(
            f"screening policy v{agent.screening_policy_version} is below the "
            f"required v{SCREENING_POLICY_VERSION} — needs a re-screen"
        )
    if contract.requires_screened_image and not image_eligible:
        if not image_complete:
            blocking.append(
                "screened image is not built yet (missing: "
                + ", ".join(missing_fields)
                + ") — needs a build-only re-screen"
            )
        else:
            blocking.append(
                f"screened image was built under a policy below the v{active_version} "
                f"contract minimum v{contract.minimum_screening_policy_version}"
            )
    if active_version != 2 and not has_versioned_dataset:
        blocking.append(
            f"no v{active_version} benchmark dataset has been generated "
            "for this agent yet"
        )
    if not benchmark_admitted:
        blocking.append(
            f"historical submission is not admitted to the active v{active_version} "
            "validator queue"
        )

    return AgentScoringReadiness(
        agent_id=agent.agent_id,
        agent_name=agent.name,
        miner_hotkey=agent.miner_hotkey,
        status=agent.status.value,
        active_bench_version=active_version,
        screening_policy_version=agent.screening_policy_version,
        required_screening_policy_version=SCREENING_POLICY_VERSION,
        requires_screened_image=contract.requires_screened_image,
        has_versioned_dataset=has_versioned_dataset,
        screened_image=ScreenedImageReadiness(
            complete=image_complete,
            verified=image_verified,
            policy_ok=policy_ok,
            missing_fields=missing_fields,
        ),
        leaseable=not blocking,
        blocking_reasons=blocking,
    )
