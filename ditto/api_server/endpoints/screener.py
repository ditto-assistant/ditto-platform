"""Screener-facing endpoints — the cheap pre-evaluation gate.

The screener worker (in ``ditto-subnet``) drains freshly ``uploaded`` agents,
does a lint + compile + build check on each tarball, and reports a verdict.
A pass promotes the agent ``uploaded -> evaluating`` so the validator queue
picks it up. A deterministic submission failure becomes ``rejected``; a
retryable infrastructure failure becomes ``screening_failed``.

The platform stays thin: it owns the state machine + the queue only. The build
check lives in the worker. These endpoints mirror ``/validator/*`` so the two
workers look identical to an operator.

Lifecycle + scope decisions (documented so they're easy to revisit):

- **Queue = new uploads, retryable failures, and stale-policy results.**
  Oldest-first drains in arrival order and re-screens in place.
- **Verdict is a direct promotion.** A pass sets ``evaluating`` (not
  ``screening_passed``). A deterministic fail sets ``rejected``; an
  infrastructure fail remains retryable as ``screening_failed``. Re-reporting
  the same verdict is idempotent; a conflicting or late verdict is a 409.
- **Dedicated auth.** Every request carries a bearer token and the configured
  screener hotkey. Result POSTs additionally verify the hotkey's sr25519
  signature over the verdict and its policy version.
"""

from __future__ import annotations

import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models import (
    ArtifactResponse,
    ScreenerQueueItem,
    ScreenerQueueResponse,
    ScreenResultRequest,
    ScreenResultResponse,
)
from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.api_server.datapipeline import DatasetGenerator
from ditto.api_server.dependencies import (
    get_dataset_generator,
    get_session,
    get_storage_client,
)
from ditto.api_server.endpoints.retrieval import AgentNotFoundError
from ditto.api_server.endpoints.validator import (
    ChainDep,
    _verify_signature,
)
from ditto.api_server.onchain_seed import derive_seed
from ditto.api_server.storage import S3StorageClient
from ditto.chain import ChainError
from ditto.db.models import Agent, ScreeningAttempt
from ditto.db.queries.agents import get_agent_by_id
from ditto.db.queries.screening import (
    claim_screening_attempts,
    get_screening_attempt,
)
from ditto_screening_protocol import verdict_signing_message

if TYPE_CHECKING:
    from ditto.chain import ChainClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/screener", tags=["screener"])

# How long a pre-signed artifact URL stays valid (mirrors the validator's).
_ARTIFACT_URL_TTL = timedelta(minutes=5)
_SCREENING_LEASE_TTL = timedelta(minutes=30)

# Policy v1 used the legacy three-field signature. Every policy from v2 onward
# binds its version, including an older worker reporting a failure during a
# future rolling policy upgrade.
_FIRST_VERSIONED_POLICY = 2


def _artifact_key(agent_id: UUID) -> str:
    return f"{agent_id}/agent.tar.gz"


# Agents a verdict may act on. ``screening`` is included for forward-compat with
# a future claim step; the terminal targets are handled separately (idempotency).
_SCREENABLE_STATUSES = (AgentStatus.UPLOADED, AgentStatus.SCREENING)


def _fresh_dataset_seed() -> int:
    """Fallback local-CSPRNG seed, used only when chain derivation is unavailable.

    Cryptographically random so a miner cannot anticipate their dataset; bounded
    to the signed 64-bit range the ``scores.seed`` / ``agents.dataset_seed``
    columns store (``[0, 2**63)``). Mirrors dittobench-api's ``FreshSeed``. The
    preferred path is :func:`_derive_dataset_seed` (verifiable on-chain); a seed
    from here is flagged by null ``dataset_seed_block`` columns so an observer can
    see it was not chain-derived.
    """
    return secrets.randbits(63)


async def _derive_dataset_seed(
    chain: ChainClient, agent_id: UUID
) -> tuple[int, int | None, str | None]:
    """Derive the dataset seed from the latest on-chain block (verifiable).

    Returns ``(seed, block_number, block_hash)``. On a chain-read failure it falls
    back to a local CSPRNG seed with ``(None, None)`` block reference, so a chain
    blip never halts submissions but the (non-verifiable) provenance is explicit.
    The block is read at job-ready, which is causally after the miner committed
    their submission, so they could not have anticipated the seed.
    """
    try:
        block = await chain.get_latest_block()
    except ChainError as e:
        logger.warning(
            "on-chain seed derivation unavailable (%s); falling back to a local "
            "CSPRNG seed for agent %s (block provenance will be null)",
            e,
            agent_id,
        )
        return _fresh_dataset_seed(), None, None
    return derive_seed(block.hash, agent_id), block.number, block.hash


class ScreenerAuthError(Exception):
    """Raised when a screener request fails authentication/authorization.

    Covers a missing/invalid bearer token, a hotkey other than the configured
    dedicated screener, and a verdict whose signature does not verify. The
    envelope handler maps these to HTTP 401 + code 5000.
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
GeneratorDep = Annotated[DatasetGenerator, Depends(get_dataset_generator)]


async def require_screener(
    request: Request,
    x_screener_hotkey: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """Authenticate the dedicated screener by hotkey and bearer token."""
    auth = request.app.state.config.screener_auth
    expected_hotkey = auth.hotkey
    expected_token = auth.api_token
    if expected_hotkey is None or expected_token is None:
        raise ScreenerAuthError("screener authentication is not configured")
    if x_screener_hotkey != expected_hotkey:
        raise ScreenerAuthError("X-Screener-Hotkey is not authorized")
    prefix = "Bearer "
    if authorization is None or not authorization.startswith(prefix):
        raise ScreenerAuthError("missing screener bearer token")
    if not secrets.compare_digest(authorization[len(prefix) :], expected_token):
        raise ScreenerAuthError("invalid screener bearer token")
    return expected_hotkey


ScreenerDep = Annotated[str, Depends(require_screener)]


@router.get(
    "/queue",
    response_model=ScreenerQueueResponse,
    responses={
        401: {"description": "Missing/invalid screener auth."},
        422: {"description": "Malformed query parameter."},
    },
)
async def queue(
    response: Response,
    screener_hotkey: ScreenerDep,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> ScreenerQueueResponse:
    """List new and stale-policy agents awaiting screening, oldest first."""
    response.headers["Cache-Control"] = "no-store"
    agents = (
        await session.scalars(
            select(Agent)
            .where(
                or_(
                    Agent.status == AgentStatus.UPLOADED,
                    Agent.status == AgentStatus.SCREENING_FAILED,
                    (
                        Agent.status.in_(
                            (
                                AgentStatus.EVALUATING,
                                AgentStatus.REJECTED,
                            )
                        )
                        & (Agent.screening_policy_version < SCREENING_POLICY_VERSION)
                    ),
                )
            )
            .order_by(Agent.created_at.asc())
            .limit(limit)
        )
    ).all()
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
    return ScreenerQueueResponse(
        items=items,
        count=len(items),
        required_policy_version=SCREENING_POLICY_VERSION,
    )


@router.post(
    "/claim",
    response_model=ScreenerQueueResponse,
    responses={
        401: {"description": "Missing/invalid screener auth."},
        422: {"description": "Malformed query parameter."},
    },
)
async def claim(
    response: Response,
    screener_hotkey: ScreenerDep,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=20)] = 1,
) -> ScreenerQueueResponse:
    """Lease pending work and make its active screening state public."""
    response.headers["Cache-Control"] = "no-store"
    now = datetime.now(UTC)
    async with session.begin():
        claimed = await claim_screening_attempts(
            session,
            screener_hotkey=screener_hotkey,
            now=now,
            ttl=_SCREENING_LEASE_TTL,
            limit=limit,
        )
    items = [
        ScreenerQueueItem(
            agent_id=agent.agent_id,
            miner_hotkey=agent.miner_hotkey,
            name=agent.name,
            sha256=agent.sha256,
            status=AgentStatus.SCREENING,
            created_at=agent.created_at,
            attempt_id=attempt.attempt_id,
            lease_deadline=attempt.deadline,
        )
        for agent, attempt in claimed
    ]
    logger.info("screener=%s claimed %d item(s)", screener_hotkey, len(items))
    return ScreenerQueueResponse(
        items=items,
        count=len(items),
        required_policy_version=SCREENING_POLICY_VERSION,
    )


@router.get(
    "/agent/{agent_id}/artifact",
    response_model=ArtifactResponse,
    responses={
        401: {"description": "Missing/invalid screener auth."},
        404: {"description": "No agent with the given id."},
        422: {"description": "Malformed UUID path parameter."},
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


def _public_screening_reason(detail: str) -> str:
    """Map untrusted screener detail to a stable, public-safe failure category.

    ``detail`` can include a Docker build-log tail produced by miner-controlled
    code. Never persist or return it verbatim: a malicious Dockerfile could print
    the BuildKit secret mounted for private dependency access.
    """
    normalized = detail.strip().casefold()
    if "no dockerfile at tarball root" in normalized:
        return "Dockerfile missing from archive root"
    if normalized.startswith("build failed"):
        return "Docker image build failed"
    if normalized.startswith("serve check failed"):
        return "Container failed the health check"
    if "tarball exceeds" in normalized:
        return "Submission archive exceeded the size limit"
    if "sha256 mismatch" in normalized:
        return "Submission artifact failed integrity verification"
    if normalized.startswith("artifact download"):
        return "Submission artifact could not be downloaded"
    if normalized.startswith("screener error"):
        return "Screening infrastructure error"
    if normalized.startswith("policy failed"):
        return "Submission failed anti-cheat screening"
    if normalized.startswith("contract failed"):
        return "Submission does not satisfy the Rust harness contract"
    if normalized.startswith("model canary"):
        return "Harness did not use the validator model gateway"
    return "Screening failed"


def _failed_screening_target(detail: str) -> AgentStatus:
    """Separate submission rejection from retryable screening infrastructure."""
    normalized = detail.strip().casefold()
    infrastructure_markers = (
        "artifact download",
        "screener error",
        "could not resolve published port",
        "cannot connect to the docker daemon",
        "docker daemon",
    )
    if any(marker in normalized for marker in infrastructure_markers):
        return AgentStatus.SCREENING_FAILED
    return AgentStatus.REJECTED


@router.post(
    "/agent/{agent_id}/result",
    response_model=ScreenResultResponse,
    responses={
        401: {"description": "Invalid screener credentials or signature."},
        404: {"description": "No agent with the given id."},
        409: {"description": "Agent is past the screening stage."},
        422: {"description": "Malformed request body or UUID path parameter."},
    },
)
async def submit_result(
    agent_id: UUID,
    payload: ScreenResultRequest,
    screener_hotkey: ScreenerDep,
    response: Response,
    chain: ChainDep,
    session: SessionDep,
    generator: GeneratorDep,
) -> ScreenResultResponse:
    """Record the screener's verdict and advance the agent's lifecycle.

    Ordering is cheap-before-expensive; no DB write happens until every check
    passes: (1) dedicated screener bearer authentication, (2) signature over
    the versioned verdict, (3) generate the per-submission
    dataset (pass + generation enabled), (4) one transaction that promotes
    ``uploaded -> evaluating`` (pass, pinning the dataset) or ``uploaded ->
    screening_failed``.

    The dataset generation (3) is a network call to the private generate service;
    it runs BEFORE the row-lock transaction (never hold a lock across I/O) and
    only when the agent isn't already pinned, so a re-reported verdict doesn't
    regenerate. If generation fails it raises and the agent is NOT promoted — the
    verdict can be retried — so an evaluating agent always has a scoreable dataset.
    """
    response.headers["Cache-Control"] = "no-store"

    if payload.screener_hotkey != screener_hotkey:
        raise ScreenerAuthError("payload hotkey does not match authenticated screener")

    # Signature proves the screener owns the hotkey and binds THIS verdict:
    #    ``passed`` is signed, so a captured result can't be replayed with the
    #    boolean flipped to grief (or unfairly promote) a miner.
    if payload.attempt_id is not None:
        signed = verdict_signing_message(
            screener_hotkey=payload.screener_hotkey,
            agent_id=agent_id,
            attempt_id=payload.attempt_id,
            passed=payload.passed,
            policy_version=payload.policy_version,
        )
    elif payload.policy_version >= _FIRST_VERSIONED_POLICY:
        signed = verdict_signing_message(
            screener_hotkey=payload.screener_hotkey,
            agent_id=agent_id,
            passed=payload.passed,
            policy_version=payload.policy_version,
        )
    else:
        signed = f"{payload.screener_hotkey}:{agent_id}:{payload.passed}".encode()
    if not _verify_signature(payload.screener_hotkey, signed, payload.signature):
        raise ScreenerAuthError(
            f"verdict signature did not verify for hotkey {payload.screener_hotkey}"
        )

    # A legacy worker may still report a failure during a rolling deploy, but it
    # can never promote a submission without attesting the current policy.
    if payload.passed and payload.policy_version != SCREENING_POLICY_VERSION:
        raise AgentNotScreenableError(
            f"passing verdict requires screening policy {SCREENING_POLICY_VERSION}"
        )

    target = (
        AgentStatus.EVALUATING
        if payload.passed
        else _failed_screening_target(payload.detail)
    )
    public_reason = None if payload.passed else _public_screening_reason(payload.detail)

    # 3. Generate the per-submission dataset (outside the row lock). Only on a pass,
    #    when generation is enabled, and when the agent is not already pinned (a
    #    cheap pre-read guards a re-reported verdict from regenerating).
    new_dataset: tuple[int, str, str, int | None, str | None] | None = None
    if payload.passed and generator.run_size is not None:
        # Own transaction so the read commits/closes before the write txn below
        # (a bare SELECT autobegins a transaction that would collide with it).
        async with session.begin():
            existing = await get_agent_by_id(session, agent_id=agent_id)
            if existing is None:
                raise AgentNotFoundError(f"no agent with id={agent_id}")
            needs_dataset = existing.dataset_seed is None
        if needs_dataset:
            seed, block_number, block_hash = await _derive_dataset_seed(chain, agent_id)
            dataset_sha256 = await generator.generate(seed)
            new_dataset = (
                seed,
                dataset_sha256,
                generator.run_size,
                block_number,
                block_hash,
            )

    # 4. Atomic: apply the verdict + pin the dataset. The row lock serializes
    #    concurrent verdicts so the status guard + transition can't be lost-updated.
    async with session.begin():
        agent = await get_agent_by_id(session, agent_id=agent_id, for_update=True)
        if agent is None:
            raise AgentNotFoundError(f"no agent with id={agent_id}")
        attempt: ScreeningAttempt | None = None
        attempt_status = (
            "passed"
            if payload.passed
            else ("rejected" if target == AgentStatus.REJECTED else "failed")
        )
        if payload.attempt_id is not None:
            attempt = await get_screening_attempt(
                session, attempt_id=payload.attempt_id, for_update=True
            )
            if (
                attempt is None
                or attempt.agent_id != agent_id
                or attempt.screener_hotkey != screener_hotkey
                or attempt.policy_version != payload.policy_version
            ):
                raise AgentNotScreenableError(
                    "verdict does not match the claimed screening attempt"
                )
            deadline = attempt.deadline
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=UTC)
            if attempt.status == attempt_status and agent.status == target:
                result_status = agent.status
                return ScreenResultResponse(
                    agent_id=agent_id, status=result_status, accepted=True
                )
            if attempt.status != "running" or datetime.now(UTC) > deadline:
                raise AgentNotScreenableError(
                    "screening attempt is expired or already completed"
                )
        rescreening = (
            agent.status
            in (
                AgentStatus.EVALUATING,
                AgentStatus.SCREENING_FAILED,
                AgentStatus.REJECTED,
            )
            and agent.screening_policy_version < SCREENING_POLICY_VERSION
        )
        idempotent = agent.status == target
        if agent.status in _SCREENABLE_STATUSES or rescreening:
            agent.status = target
        elif agent.status == target:
            pass  # idempotent re-report of the same verdict
        else:
            raise AgentNotScreenableError(
                f"agent {agent_id} is {agent.status}, cannot apply verdict "
                f"passed={payload.passed} (target {target})"
            )
        agent.screening_reason = public_reason
        # Persist the policy that produced either terminal verdict. Rejected
        # submissions retry only after a policy bump; infrastructure failures
        # remain retryable under the same policy.
        if payload.policy_version == SCREENING_POLICY_VERSION:
            agent.screening_policy_version = payload.policy_version
        if attempt is None and not idempotent:
            now = datetime.now(UTC)
            attempt = ScreeningAttempt(
                attempt_id=payload.attempt_id or UUID(int=secrets.randbits(128)),
                agent_id=agent_id,
                screener_hotkey=screener_hotkey,
                policy_version=payload.policy_version,
                status=attempt_status,
                started_at=now,
                deadline=now,
                finished_at=now,
                public_reason=public_reason,
            )
            session.add(attempt)
        elif attempt is not None:
            attempt.status = attempt_status
            attempt.finished_at = datetime.now(UTC)
            attempt.public_reason = public_reason
        # Pin the generated dataset once, when evaluating and not yet set (the
        # `is None` guard keeps a concurrent/duplicate verdict from overwriting).
        if (
            new_dataset is not None
            and agent.status == AgentStatus.EVALUATING
            and agent.dataset_seed is None
        ):
            (
                agent.dataset_seed,
                agent.dataset_sha256,
                agent.dataset_run_size,
                agent.dataset_seed_block,
                agent.dataset_seed_block_hash,
            ) = new_dataset
        result_status = agent.status

    logger.info(
        "screen verdict agent_id=%s screener=%s passed=%s status=%s dataset=%s "
        "reason=%r",
        agent_id,
        payload.screener_hotkey,
        payload.passed,
        result_status,
        "pinned" if new_dataset is not None else "none",
        agent.screening_reason,
    )
    return ScreenResultResponse(agent_id=agent_id, status=result_status, accepted=True)
