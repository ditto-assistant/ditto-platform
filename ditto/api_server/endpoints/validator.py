"""Validator-facing endpoints — the daemon's epoch loop against the platform.

The platform is intentionally *thin*: the validator daemon owns the chain
identity and drives the scoring engine (``dittobench-api``) itself. These
endpoints let it (1) pull agents awaiting evaluation, (2) fetch the uploaded
tarball, and (3) report a DittoBench :class:`ScoreReport` back. Weight-setting
stays on the daemon (``ChainClient.put_weights``); the platform never touches
the chain identity.

Lifecycle + scope decisions (documented so they're easy to revisit):

- **Queue = agents in ``evaluating``.** Honors the partial index
  ``agents_status_evaluating_idx``. The screener (deferred) promotes
  ``screening_passed -> evaluating``; until it lands, agents must be advanced
  into ``evaluating`` for them to appear here.
- **Scoring is single-validator-MVP.** A score POST records one row per
  ``(agent, validator)`` and transitions ``evaluating -> scored``. A
  multi-validator subnet wants every validator to score before finalizing;
  that consensus/promotion step is a documented follow-up. The transition
  lives in one place (:data:`_SCOREABLE_STATUSES` + the handler) so widening
  it is a small change.
- **Auth.** Only chain-registered hotkeys holding a ``validator_permit`` may
  call these. The score POST additionally verifies an sr25519 signature over
  ``f"{validator_hotkey}:{run_id}"``. The GET endpoints authenticate via the
  ``X-Validator-Hotkey`` header + the on-chain permit check; binding those
  reads to a per-request signature (nonce/timestamp) is a known gap.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import bittensor
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models import (
    ArtifactResponse,
    SubmitScoreRequest,
    SubmitScoreResponse,
    ValidatorQueueItem,
    ValidatorQueueResponse,
)
from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.upload import _SS58_PATTERN
from ditto.api_server.dependencies import (
    get_chain_client,
    get_session,
    get_storage_client,
)
from ditto.api_server.endpoints.retrieval import AgentNotFoundError
from ditto.api_server.storage import S3StorageClient
from ditto.chain import ChainError
from ditto.db.queries.agents import get_agent_by_id, list_agents_by_status
from ditto.db.queries.scores import upsert_score

if TYPE_CHECKING:
    from ditto.chain import ChainClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/validator", tags=["validator"])

# How long a pre-signed artifact URL stays valid.
_ARTIFACT_URL_TTL = timedelta(minutes=5)


# Object-store key the upload pipeline writes the tarball under.
def _artifact_key(agent_id: UUID) -> str:
    return f"{agent_id}/agent.tar.gz"


# Agents the validator may pull as work. The partial index covers exactly
# this set; widening it means widening the index too.
_QUEUE_STATUSES = (AgentStatus.EVALUATING,)

# Agents a score may be reported against. ``scored`` / ``live`` are included
# so a validator can re-score across epochs without a 409.
_SCOREABLE_STATUSES = (
    AgentStatus.EVALUATING,
    AgentStatus.SCORED,
    AgentStatus.LIVE,
)


class ValidatorAuthError(Exception):
    """Raised when a validator request fails authentication/authorization.

    Covers a missing/malformed ``X-Validator-Hotkey`` header, a hotkey not
    registered on the netuid, a hotkey without a ``validator_permit``, and
    a score whose signature does not verify. The envelope handler maps all
    of these to HTTP 401 + code 4000.
    """


class AgentNotEvaluatableError(Exception):
    """Raised when a score is submitted for an agent not in a scoreable state.

    A score is only accepted once an agent has reached evaluation
    (``evaluating`` / ``scored`` / ``live``). Reporting against an
    ``uploaded`` / ``screening*`` / ``banned`` agent is a no-op the daemon
    should not retry, so it maps to HTTP 409 (code 4001).
    """


ChainDep = Annotated["ChainClient", Depends(get_chain_client)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
StorageDep = Annotated[S3StorageClient, Depends(get_storage_client)]


async def _assert_validator_permitted(
    chain: ChainClient, netuid: int, hotkey: str
) -> None:
    """Raise unless ``hotkey`` is a permitted validator on ``netuid``.

    A chain outage surfaces as 503 (matching the upload endpoints) rather
    than a silent allow/deny; a registered-but-unpermitted or unregistered
    hotkey is a :class:`ValidatorAuthError`.
    """
    try:
        neurons = await chain.get_recent_neurons(netuid)
    except ChainError as e:
        logger.warning(f"chain unreachable during validator authz: {e}")
        raise HTTPException(
            status_code=503, detail="chain unavailable; retry shortly"
        ) from e
    for neuron in neurons:
        if neuron.hotkey == hotkey:
            if neuron.validator_permit:
                return
            raise ValidatorAuthError(
                f"hotkey {hotkey} is registered but lacks a validator permit"
            )
    raise ValidatorAuthError(f"hotkey {hotkey} is not registered on netuid {netuid}")


async def require_validator(
    request: Request,
    chain: ChainDep,
    x_validator_hotkey: Annotated[str | None, Header()] = None,
) -> str:
    """Authenticate a validator GET via the ``X-Validator-Hotkey`` header.

    Verifies the header is a well-formed SS58 hotkey and that it is a
    permitted validator on the configured netuid. Returns the hotkey for
    logging/audit by the route.
    """
    if x_validator_hotkey is None or not re.fullmatch(
        _SS58_PATTERN, x_validator_hotkey
    ):
        raise ValidatorAuthError("missing or malformed X-Validator-Hotkey header")
    netuid = request.app.state.config.chain.netuid
    await _assert_validator_permitted(chain, netuid, x_validator_hotkey)
    return x_validator_hotkey


ValidatorDep = Annotated[str, Depends(require_validator)]


def _verify_signature(hotkey: str, payload: bytes, signature_hex: str) -> bool:
    """Return True iff ``signature_hex`` is a valid sr25519 sig over ``payload``.

    Mirrors the upload endpoint's verification: a narrow ``(ValueError,
    TypeError)`` catch covers malformed hex / SS58 / wrong-shape inputs;
    anything else is a programming bug that should surface as a 500.
    """
    try:
        keypair = bittensor.Keypair(ss58_address=hotkey)
        return bool(keypair.verify(payload, bytes.fromhex(signature_hex)))
    except (ValueError, TypeError):
        return False


@router.get(
    "/queue",
    response_model=ValidatorQueueResponse,
    responses={
        401: {"description": "Missing/invalid validator auth."},
        422: {"description": "Malformed query parameter."},
        503: {"description": "Chain unavailable for the permit check."},
    },
)
async def queue(
    response: Response,
    validator_hotkey: ValidatorDep,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> ValidatorQueueResponse:
    """List agents awaiting evaluation (status ``evaluating``), oldest first."""
    response.headers["Cache-Control"] = "no-store"
    agents = await list_agents_by_status(session, statuses=_QUEUE_STATUSES, limit=limit)
    items = [
        ValidatorQueueItem(
            agent_id=a.agent_id,
            miner_hotkey=a.miner_hotkey,
            name=a.name,
            sha256=a.sha256,
            status=a.status,
            created_at=a.created_at,
        )
        for a in agents
    ]
    logger.info("validator=%s polled queue: %d item(s)", validator_hotkey, len(items))
    return ValidatorQueueResponse(items=items, count=len(items))


@router.get(
    "/agent/{agent_id}/artifact",
    response_model=ArtifactResponse,
    responses={
        401: {"description": "Missing/invalid validator auth."},
        404: {"description": "No agent with the given id."},
        422: {"description": "Malformed UUID path parameter."},
        503: {"description": "Chain unavailable for the permit check."},
    },
)
async def agent_artifact(
    agent_id: UUID,
    response: Response,
    validator_hotkey: ValidatorDep,
    session: SessionDep,
    storage: StorageDep,
) -> ArtifactResponse:
    """Return a short-lived pre-signed download URL for the agent's tarball."""
    response.headers["Cache-Control"] = "no-store"
    agent = await get_agent_by_id(session, agent_id=agent_id)
    if agent is None:
        raise AgentNotFoundError(f"no agent with id={agent_id}")
    url = await storage.presigned_get_url(
        key=_artifact_key(agent_id),
        expires_in=int(_ARTIFACT_URL_TTL.total_seconds()),
    )
    logger.info(
        "validator=%s fetched artifact url for agent_id=%s",
        validator_hotkey,
        agent_id,
    )
    return ArtifactResponse(
        agent_id=agent_id,
        sha256=agent.sha256,
        download_url=url,
        expires_at=datetime.now(UTC) + _ARTIFACT_URL_TTL,
    )


@router.post(
    "/agent/{agent_id}/score",
    response_model=SubmitScoreResponse,
    responses={
        401: {"description": "Signature did not verify / not a permitted validator."},
        404: {"description": "No agent with the given id."},
        409: {"description": "Agent is not in a scoreable state."},
        422: {"description": "Malformed request body or UUID path parameter."},
        503: {"description": "Chain unavailable for the permit check."},
    },
)
async def submit_score(
    agent_id: UUID,
    payload: SubmitScoreRequest,
    request: Request,
    response: Response,
    chain: ChainDep,
    session: SessionDep,
) -> SubmitScoreResponse:
    """Record a DittoBench score report and advance the agent's lifecycle.

    Ordering is cheap-before-expensive and no DB write happens until every
    check passes: (1) signature over ``{validator_hotkey}:{run_id}``,
    (2) on-chain validator-permit check, (3) one transaction that upserts
    the score and transitions ``evaluating -> scored``.
    """
    response.headers["Cache-Control"] = "no-store"
    report = payload.report

    # 1. Signature proves the reporting validator owns the hotkey + bound
    #    this run. CPU-only, no I/O.
    signed = f"{payload.validator_hotkey}:{report.run_id}".encode()
    if not _verify_signature(payload.validator_hotkey, signed, payload.signature):
        raise ValidatorAuthError(
            f"score signature did not verify for hotkey {payload.validator_hotkey}"
        )

    # 2. The hotkey must be a permitted validator on this subnet.
    netuid = request.app.state.config.chain.netuid
    await _assert_validator_permitted(chain, netuid, payload.validator_hotkey)

    # 3. Atomic: record the score + advance status together.
    async with session.begin():
        agent = await get_agent_by_id(session, agent_id=agent_id)
        if agent is None:
            raise AgentNotFoundError(f"no agent with id={agent_id}")
        if agent.status not in _SCOREABLE_STATUSES:
            raise AgentNotEvaluatableError(
                f"agent {agent_id} is {agent.status}, not in {_SCOREABLE_STATUSES}"
            )
        await upsert_score(
            session,
            agent_id=agent_id,
            validator_hotkey=payload.validator_hotkey,
            run_id=report.run_id,
            seed=report.seed,
            composite=report.composite,
            tool_mean=report.tool_mean,
            memory_mean=report.memory_mean,
            median_ms=report.median_ms,
            n=report.n,
            generated_at=report.generated_at,
            details={"per_case": [c.model_dump(mode="json") for c in report.per_case]}
            if report.per_case
            else None,
        )
        # Single-validator-MVP finalize: first score moves evaluating -> scored.
        if agent.status == AgentStatus.EVALUATING:
            agent.status = AgentStatus.SCORED
        result_status = agent.status

    logger.info(
        "score recorded agent_id=%s validator=%s run_id=%s composite=%.3f status=%s",
        agent_id,
        payload.validator_hotkey,
        report.run_id,
        report.composite,
        result_status,
    )
    return SubmitScoreResponse(agent_id=agent_id, status=result_status, accepted=True)
