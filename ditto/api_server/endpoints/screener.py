"""Screener-facing endpoints — the cheap pre-evaluation gate.

The screener worker (in ``ditto-subnet``) drains freshly ``uploaded`` agents,
does a lint + compile + build check on each tarball, and reports a verdict.
A pass promotes the agent ``uploaded -> evaluating`` so the validator queue
picks it up; a fail moves it ``uploaded -> screening_failed`` and it never costs
a full DittoBench run. This is the promotion path that today is manual.

The platform stays thin: it owns the state machine + the queue only. The build
check lives in the worker. These endpoints mirror ``/validator/*`` so the two
workers look identical to an operator.

Lifecycle + scope decisions (documented so they're easy to revisit):

- **Queue = agents in ``uploaded``.** Backed by the partial index
  ``agents_status_uploaded_idx``. Oldest-first drains in arrival order.
- **Verdict is a direct promotion.** A pass sets ``evaluating`` (not
  ``screening_passed``) so nothing else has to promote it; a fail sets
  ``screening_failed``. Re-reporting the same verdict is idempotent; a
  conflicting or late verdict (agent already past screening) is a 409.
- **Auth mirrors the validator.** Only a chain-registered hotkey holding a
  ``validator_permit`` may call these (a distinct ``screener_permit`` is a
  future refinement); the result POST additionally verifies an sr25519
  signature over ``f"{screener_hotkey}:{agent_id}"``.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models import (
    ArtifactResponse,
    ScreenerQueueItem,
    ScreenerQueueResponse,
    ScreenResultRequest,
    ScreenResultResponse,
)
from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.upload import _SS58_PATTERN
from ditto.api_server.dependencies import (
    get_session,
    get_storage_client,
)
from ditto.api_server.endpoints.retrieval import AgentNotFoundError

# Reuse the validator's proven chain-auth + signature helpers so the screener
# and validator share one implementation (same permit set, same sr25519 check).
from ditto.api_server.endpoints.validator import (
    ChainDep,
    ValidatorAuthError,
    _assert_validator_permitted,
    _verify_signature,
)
from ditto.api_server.storage import S3StorageClient
from ditto.db.queries.agents import get_agent_by_id, list_agents_by_status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/screener", tags=["screener"])

# How long a pre-signed artifact URL stays valid (mirrors the validator's).
_ARTIFACT_URL_TTL = timedelta(minutes=5)


def _artifact_key(agent_id: UUID) -> str:
    return f"{agent_id}/agent.tar.gz"


# Agents the screener may pull as work. Backed by ``agents_status_uploaded_idx``.
_QUEUE_STATUSES = (AgentStatus.UPLOADED,)

# Agents a verdict may act on. ``screening`` is included for forward-compat with
# a future claim step; the terminal targets are handled separately (idempotency).
_SCREENABLE_STATUSES = (AgentStatus.UPLOADED, AgentStatus.SCREENING)


class ScreenerAuthError(Exception):
    """Raised when a screener request fails authentication/authorization.

    Covers a missing/malformed ``X-Screener-Hotkey`` header, a hotkey not
    registered on the netuid, a hotkey without a ``validator_permit``, and a
    verdict whose signature does not verify. The envelope handler maps all of
    these to HTTP 401 + code 5000.
    """


class AgentNotScreenableError(Exception):
    """Raised when a verdict targets an agent past the screening stage.

    A verdict only applies to a pre-evaluation agent (``uploaded`` /
    ``screening``), or is an idempotent no-op when the agent already holds the
    verdict's target status. Reporting against an already-``evaluating`` /
    ``scored`` / ``live`` / ``banned`` agent (or flipping a decided verdict) is
    a conflict the worker should not retry: HTTP 409 (code 5001).
    """


SessionDep = Annotated[AsyncSession, Depends(get_session)]
StorageDep = Annotated[S3StorageClient, Depends(get_storage_client)]


async def _assert_screener_permitted(chain: ChainDep, netuid: int, hotkey: str) -> None:
    """Permit check reusing the validator's, re-flavoured as a screener error.

    A chain outage still surfaces as the validator helper's 503; a
    registered-but-unpermitted / unregistered hotkey becomes a
    :class:`ScreenerAuthError` (code 5000) rather than a validator one.
    """
    try:
        await _assert_validator_permitted(chain, netuid, hotkey)
    except ValidatorAuthError as e:
        raise ScreenerAuthError(str(e)) from e


async def require_screener(
    request: Request,
    chain: ChainDep,
    x_screener_hotkey: Annotated[str | None, Header()] = None,
) -> str:
    """Authenticate a screener request via the ``X-Screener-Hotkey`` header.

    Verifies the header is a well-formed SS58 hotkey and a permitted validator
    on the configured netuid. Returns the hotkey for logging/audit.
    """
    if x_screener_hotkey is None or not re.fullmatch(_SS58_PATTERN, x_screener_hotkey):
        raise ScreenerAuthError("missing or malformed X-Screener-Hotkey header")
    netuid = request.app.state.config.chain.netuid
    await _assert_screener_permitted(chain, netuid, x_screener_hotkey)
    return x_screener_hotkey


ScreenerDep = Annotated[str, Depends(require_screener)]


@router.get(
    "/queue",
    response_model=ScreenerQueueResponse,
    responses={
        401: {"description": "Missing/invalid screener auth."},
        422: {"description": "Malformed query parameter."},
        503: {"description": "Chain unavailable for the permit check."},
    },
)
async def queue(
    response: Response,
    screener_hotkey: ScreenerDep,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> ScreenerQueueResponse:
    """List agents awaiting screening (status ``uploaded``), oldest first."""
    response.headers["Cache-Control"] = "no-store"
    agents = await list_agents_by_status(session, statuses=_QUEUE_STATUSES, limit=limit)
    items = [
        ScreenerQueueItem(
            agent_id=a.agent_id,
            miner_hotkey=a.miner_hotkey,
            name=a.name,
            sha256=a.sha256,
            status=a.status,
            created_at=a.created_at,
        )
        for a in agents
    ]
    logger.info("screener=%s polled queue: %d item(s)", screener_hotkey, len(items))
    return ScreenerQueueResponse(items=items, count=len(items))


@router.get(
    "/agent/{agent_id}/artifact",
    response_model=ArtifactResponse,
    responses={
        401: {"description": "Missing/invalid screener auth."},
        404: {"description": "No agent with the given id."},
        422: {"description": "Malformed UUID path parameter."},
        503: {"description": "Chain unavailable for the permit check."},
    },
)
async def agent_artifact(
    agent_id: UUID,
    response: Response,
    screener_hotkey: ScreenerDep,
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
        "screener=%s fetched artifact url for agent_id=%s", screener_hotkey, agent_id
    )
    return ArtifactResponse(
        agent_id=agent_id,
        sha256=agent.sha256,
        download_url=url,
        expires_at=datetime.now(UTC) + _ARTIFACT_URL_TTL,
    )


@router.post(
    "/agent/{agent_id}/result",
    response_model=ScreenResultResponse,
    responses={
        401: {"description": "Signature did not verify / not a permitted screener."},
        404: {"description": "No agent with the given id."},
        409: {"description": "Agent is past the screening stage."},
        422: {"description": "Malformed request body or UUID path parameter."},
        503: {"description": "Chain unavailable for the permit check."},
    },
)
async def submit_result(
    agent_id: UUID,
    payload: ScreenResultRequest,
    request: Request,
    response: Response,
    chain: ChainDep,
    session: SessionDep,
) -> ScreenResultResponse:
    """Record the screener's verdict and advance the agent's lifecycle.

    Ordering is cheap-before-expensive; no DB write happens until every check
    passes: (1) signature over ``{screener_hotkey}:{agent_id}``, (2) on-chain
    permit check, (3) one transaction that promotes ``uploaded -> evaluating``
    (pass) or ``uploaded -> screening_failed`` (fail).
    """
    response.headers["Cache-Control"] = "no-store"

    # 1. Signature proves the screener owns the hotkey + bound this verdict.
    signed = f"{payload.screener_hotkey}:{agent_id}".encode()
    if not _verify_signature(payload.screener_hotkey, signed, payload.signature):
        raise ScreenerAuthError(
            f"verdict signature did not verify for hotkey {payload.screener_hotkey}"
        )

    # 2. The hotkey must be a permitted validator on this subnet.
    netuid = request.app.state.config.chain.netuid
    await _assert_screener_permitted(chain, netuid, payload.screener_hotkey)

    target = AgentStatus.EVALUATING if payload.passed else AgentStatus.SCREENING_FAILED

    # 3. Atomic: apply the verdict.
    async with session.begin():
        agent = await get_agent_by_id(session, agent_id=agent_id)
        if agent is None:
            raise AgentNotFoundError(f"no agent with id={agent_id}")
        if agent.status in _SCREENABLE_STATUSES:
            agent.status = target
        elif agent.status == target:
            pass  # idempotent re-report of the same verdict
        else:
            raise AgentNotScreenableError(
                f"agent {agent_id} is {agent.status}, cannot apply verdict "
                f"passed={payload.passed} (target {target})"
            )
        result_status = agent.status

    logger.info(
        "screen verdict agent_id=%s screener=%s passed=%s status=%s%s",
        agent_id,
        payload.screener_hotkey,
        payload.passed,
        result_status,
        f" detail={payload.detail!r}" if payload.detail else "",
    )
    return ScreenResultResponse(agent_id=agent_id, status=result_status, accepted=True)
