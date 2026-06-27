"""Validator-facing endpoints — the daemon's epoch loop against the platform.

The platform is intentionally *thin*: the validator daemon owns the chain
identity and drives the scoring engine (``dittobench-api``) itself. These
endpoints let it (1) pull agents awaiting evaluation, (2) fetch the uploaded
tarball, and (3) report a DittoBench :class:`ScoreReport` back. Weight-setting
stays on the daemon (``ChainClient.put_weights``); the platform never touches
the chain identity.

STATUS: STUBBED. Every handler here returns deterministic in-memory data and
does **not** touch Postgres or object storage yet — this unblocks the validator
daemon author wiring the ``miner -> API -> vali -> chain`` cycle against stable
wire shapes. The request bodies *are* validated (Pydantic), so the contract is
enforced today. The real pass wires: a ``scores`` table + migration, a
list-by-status query, real S3 pre-signing, real ``uploaded -> evaluating ->
scored`` transitions, and real validator-hotkey signature auth. Each stub is
marked ``# STUB`` at the point real wiring lands.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Annotated
from uuid import UUID, uuid5

from fastapi import APIRouter, Depends, Header, Query, Response

from ditto.api_models import (
    ArtifactResponse,
    SubmitScoreRequest,
    SubmitScoreResponse,
    ValidatorQueueItem,
    ValidatorQueueResponse,
)
from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.upload import _SS58_PATTERN
from ditto.api_server.endpoints.retrieval import AgentNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/validator", tags=["validator"])

# How long a pre-signed artifact URL stays valid. Mirrors the planned real
# S3 pre-sign TTL so the daemon's retry budget is correct against the stub.
_ARTIFACT_URL_TTL = timedelta(minutes=5)

# Deterministic namespace for synthetic stub agent ids, so repeated /queue
# reads return stable ids the client can hard-code in integration wiring.
_STUB_NAMESPACE = UUID("11118888-1111-8888-1111-888811118888")

# A sentinel id the stub treats as "unknown" so callers can exercise the
# 404/not-found envelope on the artifact + score paths without a live DB.
_STUB_UNKNOWN_AGENT = UUID("00000000-0000-0000-0000-000000000000")


class ValidatorAuthError(Exception):
    """Raised when a validator request carries malformed auth.

    STUB: today this only fires on a malformed ``X-Validator-Hotkey``
    header. Real token + signature verification (validator permit, stake,
    replay protection) lands with the real-auth pass and will raise this
    same type so the envelope mapping (code 4000) does not change.
    """


class AgentNotEvaluatableError(Exception):
    """Raised when a score is submitted for an agent not in an eval state.

    A score is only accepted while an agent is mid-evaluation. Reporting
    against a ``banned`` / ``live`` / already-``scored`` agent is a no-op
    the daemon should not retry, so it maps to HTTP 409 (code 4001).
    """


async def require_validator(
    x_validator_hotkey: Annotated[str | None, Header()] = None,
) -> str | None:
    """Auth seam for validator endpoints.

    STUB: returns the supplied ``X-Validator-Hotkey`` (validated for SS58
    shape) or ``None`` when absent. It does **not** yet verify a session
    token, the validator permit, or a request signature — that is the
    real-auth pass. Kept as a dependency now so every validator route
    already depends on it and the swap is a one-file change.
    """
    if x_validator_hotkey is None:
        return None
    if not re.fullmatch(_SS58_PATTERN, x_validator_hotkey):
        raise ValidatorAuthError(
            f"malformed X-Validator-Hotkey: {x_validator_hotkey!r}"
        )
    return x_validator_hotkey


ValidatorDep = Annotated[str | None, Depends(require_validator)]


def _stub_queue_items() -> list[ValidatorQueueItem]:
    """Deterministic synthetic work queue (STUB).

    Two agents in ``screening_passed`` — the state the real list-by-status
    query will filter on. Fixed ids + timestamps so the client gets stable
    data to wire against.
    """
    base = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
    fixtures = [
        ("5DhaT8U7LVwnnJNUU8VL1XEipicatoaDVVq7cHo227gogVZm", "alpha-agent", "11"),
        ("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", "beta-agent", "22"),
    ]
    items: list[ValidatorQueueItem] = []
    for i, (hotkey, name, sha_seed) in enumerate(fixtures):
        items.append(
            ValidatorQueueItem(
                agent_id=uuid5(_STUB_NAMESPACE, name),
                miner_hotkey=hotkey,
                name=name,
                sha256=(sha_seed * 32),  # 64 hex chars
                status=AgentStatus.SCREENING_PASSED,
                created_at=base + timedelta(minutes=i),
            )
        )
    return items


@router.get(
    "/queue",
    response_model=ValidatorQueueResponse,
    responses={
        401: {"description": "Malformed validator auth header."},
        422: {"description": "Malformed query parameter."},
    },
)
async def queue(
    response: Response,
    _validator: ValidatorDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> ValidatorQueueResponse:
    """List agents awaiting evaluation, oldest first.

    STUB: returns a fixed synthetic set. Real pass queries the ``agents``
    table for rows in an evaluatable status (``screening_passed``), ordered
    by ``created_at``.
    """
    response.headers["Cache-Control"] = "no-store"
    items = _stub_queue_items()[:limit]  # STUB: real list-by-status query
    return ValidatorQueueResponse(items=items, count=len(items))


@router.get(
    "/agent/{agent_id}/artifact",
    response_model=ArtifactResponse,
    responses={
        401: {"description": "Malformed validator auth header."},
        404: {"description": "No agent with the given id."},
        422: {"description": "Malformed UUID path parameter."},
    },
)
async def agent_artifact(
    agent_id: UUID,
    response: Response,
    _validator: ValidatorDep,
) -> ArtifactResponse:
    """Return a (pre-signed) download URL for the agent's tarball.

    STUB: synthesises a URL + expiry without touching object storage. Real
    pass looks up the agent, then asks the storage client to pre-sign a
    short-lived GET against the stored tarball key.
    """
    response.headers["Cache-Control"] = "no-store"
    if agent_id == _STUB_UNKNOWN_AGENT:  # STUB: real lookup -> 404 when absent
        raise AgentNotFoundError(f"no agent with id={agent_id}")
    return ArtifactResponse(
        agent_id=agent_id,
        sha256="ab" * 32,  # STUB: real sha256 from the agents row
        download_url=(  # STUB: real S3 pre-signed GET
            f"https://stub.minio.local/ditto-agents/{agent_id}.tar.gz?X-Amz-Stub=1"
        ),
        expires_at=datetime.now(UTC) + _ARTIFACT_URL_TTL,
    )


@router.post(
    "/agent/{agent_id}/score",
    response_model=SubmitScoreResponse,
    responses={
        401: {"description": "Malformed validator auth header."},
        404: {"description": "No agent with the given id."},
        409: {"description": "Agent is not in an evaluatable state."},
        422: {"description": "Malformed request body or UUID path parameter."},
    },
)
async def submit_score(
    agent_id: UUID,
    payload: SubmitScoreRequest,
    response: Response,
    _validator: ValidatorDep,
) -> SubmitScoreResponse:
    """Record a DittoBench score report for an agent.

    STUB: validates + accepts the report and echoes ``scored`` without
    persisting. Real pass verifies the validator signature, inserts a
    ``scores`` row, and transitions the agent ``evaluating -> scored``
    inside one transaction.
    """
    response.headers["Cache-Control"] = "no-store"
    if agent_id == _STUB_UNKNOWN_AGENT:  # STUB: real lookup -> 404 when absent
        raise AgentNotFoundError(f"no agent with id={agent_id}")
    logger.info(
        "stub score recorded agent_id=%s validator=%s run_id=%s composite=%.3f",
        agent_id,
        payload.validator_hotkey,
        payload.report.run_id,
        payload.report.composite,
    )
    return SubmitScoreResponse(
        agent_id=agent_id,
        status=AgentStatus.SCORED,  # STUB: real evaluating -> scored transition
        accepted=True,
    )
