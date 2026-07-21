"""Validator-facing endpoints — the daemon's epoch loop against the platform.

The platform is intentionally *thin*: the validator daemon owns the chain
identity and drives the scoring engine (``dittobench-api``) itself. These
endpoints let it (1) pull agents awaiting evaluation, (2) fetch the uploaded
tarball, and (3) report a DittoBench :class:`ScoreReport` back. Weight-setting
stays on the daemon (``ChainClient.put_weights``); the platform never touches
the chain identity.

Lifecycle + scope decisions (documented so they're easy to revisit):

- **Queue = agents in ``evaluating``.** Honors the partial index
  ``agents_status_evaluating_idx``. The screener promotes ``uploaded ->
  evaluating`` (see ``endpoints/screener.py``); a submission that hasn't been
  screened yet won't appear here.
- **Scoring is k=3 multi-validator consensus.** Up to
  :data:`~ditto.db.queries.scores.SCORING_QUORUM` distinct validators each score
  an agent, gated by leased tickets (:mod:`ditto.db.queries.tickets`), one row
  per ``(agent, validator)``. The agent stays ``evaluating`` until the
  quorum-th score, then the handler finalizes it on the **median** composite and
  transitions ``evaluating -> scored`` (or ``ath_pending_review`` if the copy
  gate holds it). No single validator is decisive; the transition lives in one
  place (:data:`_SCOREABLE_STATUSES` + the handler).
- **Auth.** Only chain-registered hotkeys holding a ``validator_permit`` may
  call these. Job claims additionally carry a fresh, one-time signed nonce so a
  caller cannot reserve work by merely naming somebody else's permitted
  hotkey. The score POST verifies an sr25519 signature over a
  **canonical payload** binding the agent id and the reported
  ``run_id`` / ``composite`` / ``seed`` (see :func:`_score_signing_message`), so
  a captured signature can neither be replayed against a different agent nor
  cover an altered composite. The remaining GET endpoints are read-only and
  authenticate via the ``X-Validator-Hotkey`` header + on-chain permit check;
  they cannot allocate a quorum slot or submit a score.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import statistics
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Annotated, Any, Literal
from uuid import UUID, uuid4

import bittensor
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models import (
    ArtifactResponse,
    BenchmarkProgress,
    FailJobRequest,
    FailJobResponse,
    JobRequest,
    JobResponse,
    ScoreReport,
    SubmitScoreRequest,
    SubmitScoreResponse,
    SubmitTranscriptResponse,
    ValidatorHeartbeatRequest,
    ValidatorHeartbeatResponse,
)
from ditto.api_models.agent_status import SCOREABLE_AGENT_STATUSES, AgentStatus
from ditto.api_models.benchmark_contract import benchmark_contract
from ditto.api_models.benchmark_progress import benchmark_progress_signing_token
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.api_models.stack_health import (
    ValidatorStackHealth,
    validator_stack_health_signing_token,
)
from ditto.api_models.system_health import (
    SystemMetrics,
    system_metrics_signing_token,
)
from ditto.api_models.ticket_status import TicketStatus
from ditto.api_models.upload import _SS58_PATTERN
from ditto.api_models.validator_capabilities import (
    ValidatorCapabilities,
    ValidatorStackIdentity,
    validator_artifact_mode,
    validator_identity_signing_token,
)
from ditto.api_server.anti_copy_comparison import ANTI_COPY_ALGORITHM_VERSION
from ditto.api_server.benchmark_rollout import (
    refresh_rolling_qualification,
)
from ditto.api_server.config import ValidatorCompatibilityConfig
from ditto.api_server.datapipeline import DatasetGenerator
from ditto.api_server.dependencies import (
    get_chain_client,
    get_dataset_generator,
    get_session,
    get_storage_client,
)
from ditto.api_server.endpoints.retrieval import AgentNotFoundError
from ditto.api_server.fingerprint import reference_corpus_provenance
from ditto.api_server.scoring_gate import evaluate_duplicate_signals
from ditto.api_server.storage import S3StorageClient
from ditto.chain import ChainError
from ditto.db.models import (
    Agent,
    AthReview,
    BenchmarkDataset,
    Score,
    ValidatorHeartbeat,
    ValidatorTicket,
)
from ditto.db.queries.agents import get_agent_by_id
from ditto.db.queries.audit import (
    EVENT_AUDIT,
    EVENT_FINALIZED,
    EVENT_SCORE,
    EVENT_SCORE_INVALIDATED,
    EVENT_SCORE_RETEST_REQUESTED,
    append_audit_entry,
    get_latest_score_retest_event,
)
from ditto.db.queries.benchmark_rollout import (
    LEGACY_BENCH_VERSION,
    active_bench_version,
    heartbeat_supports_version,
    issue_rollout_ticket,
    open_rollout,
)
from ditto.db.queries.heartbeats import (
    upsert_validator_heartbeat,
)
from ditto.db.queries.scores import (
    SCORING_QUORUM,
    get_score_for_validator,
    list_eligible_ledger,
    list_scores_for_agent,
    upsert_score,
)
from ditto.db.queries.tickets import (
    MAX_INFRA_RETRY_GRANTS,
    get_open_ticket,
    infra_retry_backoff,
    issue_ticket,
    mark_ticket_scored,
)
from ditto.db.queries.validator_auth import (
    ValidatorRequestReplayError,
    consume_validator_nonce,
)

if TYPE_CHECKING:
    from ditto.chain import ChainClient

logger = logging.getLogger(__name__)

# Reproduce-under-transform audit (v3 Part A). These mirror the validator's
# constants in ditto-subnet ``ditto/validator/transform_audit.py``, which in turn
# mirror dittobench-datagen ``persona/transform.go``. They are part of a PUBLIC
# derivation contract, not tunables: the whole point is that any third party can
# recompute a verdict from the published seed and get the same answer.
AUDIT_BPS = 2500

# The brittleness verdict is a one-sided exact BINOMIAL TEST on discordant audit
# pairs, mirroring ditto-subnet ``ditto/validator/transform_audit.py``.
#
# A pair answered correctly in the base phrasing and incorrectly under the
# post-commit transform is the brittleness event; the mirror image is not. The
# null is that discordant pairs fall either way equally, which is what an honest
# nondeterministic model does. The 2026-07-18 calibration measured honest at 5
# base-only vs 6 transform-only (symmetric) and a surface-gated harness at 6 vs
# 0; the previous ratio threshold could not tell those apart.
#
# ALPHA *is* the false-positive rate on honest miners, by construction. The
# ratio floor it replaces had an unknown error rate, measured at 16% of honest
# runs.
AUDIT_ALPHA = 0.01
# Fewest discordant pairs that can produce a verdict: below this the exact test
# cannot reach ALPHA even on a perfect one-directional run.
AUDIT_MIN_DISCORDANT = 6
TRANSFORM_AUDIT_REVIEW_REASON = "transform_audit_brittleness"

# Enforcement stays OFF by default. The metric now discriminates in principle
# (see the calibration in dittobench-api docs/BASELINES.md Run 3), but the floor
# has not been re-validated end to end against the population it judges --
# champion/tail agents, which are more accurate than the stock reference harness
# every number above came from. Turn this on only with that evidence.
TRANSFORM_AUDIT_ENFORCE = os.environ.get(
    "DITTO_TRANSFORM_AUDIT_ENFORCE", "false"
).strip().lower() in {"1", "true", "yes", "on"}


def _binomial_tail(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) for X ~ Binomial(n, p). Exact, no dependencies."""
    if n <= 0:
        return 1.0
    k = max(0, k)
    total = 0.0
    coeff = 1.0
    for i in range(0, n + 1):
        if i >= k:
            total += coeff * (p**i) * ((1 - p) ** (n - i))
        coeff = coeff * (n - i) / (i + 1)
    return min(1.0, total)


def _pool_audit_pairs(agent_scores: Sequence[Any]) -> dict[str, int]:
    """Sum the audit 2x2 counts across an agent's finalized scores.

    Each validator already pooled its own confirmation runs; this pools across
    the k=3 validators, so the verdict rests on all the evidence rather than on
    any one validator's handful of pairs. Same reasoning as finalizing the
    composite on the median: no single validator decides an agent's fate.
    """
    pooled = {"both_correct": 0, "base_only": 0, "transform_only": 0, "both_wrong": 0}
    for score in agent_scores:
        details = score.details if isinstance(score.details, dict) else {}
        raw = details.get("audit_pairs_pooled") or details.get("audit_pairs")
        if not isinstance(raw, dict):
            continue
        for key in pooled:
            v = raw.get(key, 0)
            if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
                pooled[key] += v
    return pooled


def _transform_audit_verdict(
    agent_scores: Sequence[Any],
) -> tuple[float | None, dict[str, int], bool]:
    """Pooled brittleness verdict across an agent's finalized scores.

    Returns ``(p_value, pooled_counts, failed)``. ``failed`` is False whenever
    the evidence is thin -- no score carried the counts (an older scoring
    engine), or too few discordant pairs to reach ALPHA. Absence of evidence is
    not a failed audit, and the cost of getting that backwards is paid by a
    legitimate miner.
    """
    pooled = _pool_audit_pairs(agent_scores)
    discordant = pooled["base_only"] + pooled["transform_only"]
    if sum(pooled.values()) == 0:
        return None, pooled, False
    if discordant < AUDIT_MIN_DISCORDANT:
        return None, pooled, False
    pvalue = _binomial_tail(pooled["base_only"], discordant)
    return pvalue, pooled, pvalue <= AUDIT_ALPHA


router = APIRouter(prefix="/validator", tags=["validator"])

# How long a pre-signed artifact URL stays valid.
_ARTIFACT_URL_TTL = timedelta(minutes=5)

# How long a validator has to redeem a ticket with a score before it lapses and
# the slot re-opens for another validator.
# Keep the lease longer than the validator's locked 75-minute benchmark cap.
# The 15-minute margin leaves enough time to fetch, sign, and submit a
# completed run.
_TICKET_TTL = timedelta(minutes=90)

# Signed job claims outside this window are stale. A consumed nonce remains in
# the database for the same window, making replay rejection consistent across
# every API replica without introducing another secret.
_JOB_REQUEST_MAX_AGE = timedelta(minutes=2)
_QUALIFICATION_REFRESH_INTERVAL_SECONDS = 30.0
_qualification_refresh_due = 0.0

# Reject captured heartbeats outside a short clock-skew/retry window. Workers
# report every two minutes, so five minutes tolerates normal transient outages.
_HEARTBEAT_MAX_SKEW_SECONDS = 300
_HEARTBEAT_MAX_BYTES = 4096


# Object-store key the upload pipeline writes the tarball under.
def _artifact_key(agent_id: UUID) -> str:
    return f"{agent_id}/agent.tar.gz"


def _screened_image_key(agent_id: UUID, image_upload_id: UUID) -> str:
    """Return the immutable accepted screener image object key."""
    return f"{agent_id}/screened-images/{image_upload_id}.tar"


# Agents the validator may pull as work. The partial index covers exactly
# Agents a score may be reported against. ``scored`` / ``live`` are included
# so a validator can re-score across epochs without a 409;
# ``ath_pending_review`` is included so a re-score of a held agent updates its
# score row (feeding the eventual review) without un-holding it.
_SCOREABLE_STATUSES = SCOREABLE_AGENT_STATUSES


async def _refresh_qualification_if_due(
    session: AsyncSession, *, generator: DatasetGenerator, now: datetime
) -> None:
    """Single-flight best-effort convergence for authenticated idle pollers."""
    global _qualification_refresh_due
    monotonic_now = time.monotonic()
    if monotonic_now < _qualification_refresh_due:
        return
    # Set before the first await so concurrent requests in this process collapse
    # into one refresh. Score/verdict triggers remain the immediate primary path.
    _qualification_refresh_due = monotonic_now + _QUALIFICATION_REFRESH_INTERVAL_SECONDS
    try:
        await refresh_rolling_qualification(session, generator=generator, now=now)
    except Exception:
        logger.exception("automatic benchmark qualification refresh failed")


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
GeneratorDep = Annotated[DatasetGenerator, Depends(get_dataset_generator)]


def _dev_bypass_permit(network: str) -> bool:
    """Whether the dev "skip the validator permit check" escape hatch is active.

    Only when ``DITTO_DEV_ALLOW_UNPERMITTED_VALIDATOR`` is explicitly truthy AND
    the process is not pointed at mainnet. On ``finney`` the flag is refused
    outright (logged at ERROR) so a stray dev env var can never open the
    validator surface on the production chain, defence-in-depth beyond keeping it
    unset in prod."""
    if os.environ.get("DITTO_DEV_ALLOW_UNPERMITTED_VALIDATOR", "").lower() not in {
        "1",
        "true",
        "yes",
    }:
        return False
    net = network.lower()
    if net.startswith("finney") or net == "mainnet":
        logger.error(
            "refusing DITTO_DEV_ALLOW_UNPERMITTED_VALIDATOR on production network=%s;"
            " enforcing the validator permit check",
            network,
        )
        return False
    return True


async def _assert_validator_permitted(
    chain: ChainClient, netuid: int, hotkey: str, *, network: str
) -> None:
    """Raise unless ``hotkey`` is a permitted validator on ``netuid``.

    A chain outage surfaces as 503 (matching the upload endpoints) rather
    than a silent allow/deny; a registered-but-unpermitted or unregistered
    hotkey is a :class:`ValidatorAuthError`. ``network`` is the resolved
    subtensor network, so the dev bypass can be refused on mainnet.
    """
    if _dev_bypass_permit(network):
        logger.warning(
            "DEV: allowing validator request without permit hotkey=%s netuid=%d",
            hotkey,
            netuid,
        )
        return
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
    network = request.app.state.config.chain.subtensor_network
    await _assert_validator_permitted(
        chain, netuid, x_validator_hotkey, network=network
    )
    return x_validator_hotkey


ValidatorDep = Annotated[str, Depends(require_validator)]


def _lease_token(deadline: datetime) -> str:
    """Canonical UTC token that binds a score to one ticket lease."""
    return deadline.astimezone(UTC).isoformat(timespec="microseconds")


def _reported_transcript_sha256(report: ScoreReport) -> str | None:
    """The transcript digest a report declares, or ``None``.

    The scoring engine content-addresses the run's transcript artifact (the
    graded per-case inputs) and the validator forwards the digest under
    ``details["transcript_sha256"]``. ``details`` is otherwise opaque; this is
    the one key the platform reads back out of it at ingest, because the digest
    is bound into the signed payload (offline reproducibility, v3 review
    finding 3).
    """
    details = report.details if isinstance(report.details, dict) else {}
    value = details.get("transcript_sha256")
    if isinstance(value, str) and value:
        return value
    return None


def _score_signing_message(
    validator_hotkey: str,
    agent_id: UUID,
    ticket_deadline: datetime | None,
    report: ScoreReport,
) -> bytes:
    """Canonical bytes a score signature is verified against.

    Must match the validator's ``sign_score`` byte-for-byte:
    ``{validator_hotkey}:{agent_id}:{ticket_deadline}:{run_id}:``
    ``{composite!r}:{seed}`` — and, when the report declares a transcript
    digest, ``:{transcript_sha256}`` is appended (both sides derive presence
    from the same report field, so old validators that publish no transcript
    keep the previous format). Binding the exact lease means a response from an
    expired attempt cannot be replayed after the ticket is reissued; binding
    the transcript digest means the published artifact cannot be swapped after
    the fact without breaking the signature.
    """
    lease = _lease_token(ticket_deadline) if ticket_deadline is not None else ""
    # CANONICAL FIELD ORDER, mirrored byte-for-byte by ditto-subnet
    # ditto/validator/signing.py. Two independent changes each append a
    # conditional suffix here, so the order is fixed deliberately rather than
    # left to whichever merged first:
    #
    #   base : bench_version? : transcript_sha256?
    #
    # bench_version sits next to seed because it QUALIFIES the seed -- the same
    # seed is a different dataset under a different contract -- so the "what was
    # scored" tuple stays contiguous. transcript_sha256 binds the artifact the
    # run PRODUCED, so it is outermost. A validator that sends neither produces
    # the pre-existing bytes, which is what keeps old validators verifiable.
    msg = (
        f"{validator_hotkey}:{agent_id}:{lease}:{report.run_id}:"
        f"{report.composite!r}:{report.seed}"
    )
    if report.bench_version is not None:
        msg += f":{report.bench_version}"
    transcript = _reported_transcript_sha256(report)
    if transcript:
        msg += f":{transcript}"
    return msg.encode()


def _job_signing_message(
    validator_hotkey: str, nonce: UUID, requested_at: datetime
) -> bytes:
    """Canonical bytes proving possession of a hotkey for one job claim."""
    requested = requested_at.astimezone(UTC).isoformat(timespec="microseconds")
    return f"validator-job:{validator_hotkey}:{nonce}:{requested}".encode()


def _artifact_signing_message(
    validator_hotkey: str,
    agent_id: UUID,
    nonce: UUID,
    requested_at: datetime,
) -> bytes:
    """Canonical proof-of-possession bytes for one artifact URL request."""
    requested = requested_at.astimezone(UTC).isoformat(timespec="microseconds")
    return (
        f"validator-artifact:v1:{validator_hotkey}:{agent_id}:{nonce}:{requested}"
    ).encode()


def _job_fail_signing_message(
    validator_hotkey: str,
    agent_id: UUID,
    ticket_deadline: datetime,
    nonce: UUID,
    requested_at: datetime,
) -> bytes:
    """Canonical proof-of-possession bytes for one ticket-fail request.

    Mirrored byte-for-byte by ditto-subnet ``ditto/validator/signing.py``. The
    lease ``ticket_deadline`` is bound so a captured fail request cannot close a
    later reissued ticket, and both timestamps use the same canonical UTC
    microsecond form as every other validator write.
    """
    deadline = ticket_deadline.astimezone(UTC).isoformat(timespec="microseconds")
    requested = requested_at.astimezone(UTC).isoformat(timespec="microseconds")
    return (
        f"validator-job-fail:v1:{validator_hotkey}:{agent_id}:{deadline}:"
        f"{nonce}:{requested}"
    ).encode()


def _heartbeat_signing_message(
    *,
    validator_hotkey: str,
    software_version: str,
    protocol_version: int,
    code_digest: str,
    state: str,
    timestamp: int,
    active_agent_id: UUID | None = None,
    system_metrics: SystemMetrics | None = None,
    benchmark_progress: BenchmarkProgress | None = None,
    capabilities: ValidatorCapabilities | None = None,
    stack: ValidatorStackIdentity | None = None,
    stack_health: ValidatorStackHealth | None = None,
) -> bytes:
    """Canonical heartbeat payload, mirrored by ``ditto-subnet``."""
    if stack_health is not None and protocol_version < 9:
        raise ValueError("per-component stack health requires heartbeat protocol v9")
    if protocol_version >= 9:
        if capabilities is None or stack is None:
            raise ValueError("heartbeat protocol v9 requires capabilities and stack")
        if stack_health is None:
            raise ValueError("heartbeat protocol v9 requires stack health")
        active = str(active_agent_id) if active_agent_id is not None else ""
        return (
            "ditto-validator-heartbeat:v9:"
            f"{validator_hotkey}:{software_version}:{protocol_version}:"
            f"{code_digest}:{state}:{active}:"
            f"{system_metrics_signing_token(system_metrics)}:"
            f"{benchmark_progress_signing_token(benchmark_progress)}:"
            f"{validator_identity_signing_token(capabilities, stack)}:"
            f"{validator_stack_health_signing_token(stack_health)}:{timestamp}"
        ).encode()
    if protocol_version >= 8:
        if capabilities is None or stack is None:
            raise ValueError("heartbeat protocol v8 requires capabilities and stack")
        active = str(active_agent_id) if active_agent_id is not None else ""
        return (
            "ditto-validator-heartbeat:v8:"
            f"{validator_hotkey}:{software_version}:{protocol_version}:"
            f"{code_digest}:{state}:{active}:"
            f"{system_metrics_signing_token(system_metrics)}:"
            f"{benchmark_progress_signing_token(benchmark_progress)}:"
            f"{validator_identity_signing_token(capabilities, stack)}:{timestamp}"
        ).encode()
    if protocol_version >= 7:
        if capabilities is None or stack is None:
            raise ValueError("heartbeat protocol v7 requires capabilities and stack")
        active = str(active_agent_id) if active_agent_id is not None else ""
        return (
            "ditto-validator-heartbeat:v7:"
            f"{validator_hotkey}:{software_version}:{protocol_version}:"
            f"{code_digest}:{state}:{active}:"
            f"{system_metrics_signing_token(system_metrics)}:"
            f"{benchmark_progress_signing_token(benchmark_progress)}:"
            f"{validator_identity_signing_token(capabilities, stack)}:{timestamp}"
        ).encode()
    if protocol_version >= 4:
        active = str(active_agent_id) if active_agent_id is not None else ""
        return (
            "ditto-validator-heartbeat:v4:"
            f"{validator_hotkey}:{software_version}:{protocol_version}:"
            f"{code_digest}:{state}:{active}:"
            f"{system_metrics_signing_token(system_metrics)}:"
            f"{benchmark_progress_signing_token(benchmark_progress)}:{timestamp}"
        ).encode()
    if protocol_version >= 3:
        active = str(active_agent_id) if active_agent_id is not None else ""
        return (
            "ditto-validator-heartbeat:v3:"
            f"{validator_hotkey}:{software_version}:{protocol_version}:"
            f"{code_digest}:{state}:{active}:"
            f"{system_metrics_signing_token(system_metrics)}:{timestamp}"
        ).encode()
    if protocol_version >= 2:
        active = str(active_agent_id) if active_agent_id is not None else ""
        return (
            "ditto-validator-heartbeat:v2:"
            f"{validator_hotkey}:{software_version}:{protocol_version}:"
            f"{code_digest}:{state}:{active}:{timestamp}"
        ).encode()
    return (
        "ditto-validator-heartbeat:v1:"
        f"{validator_hotkey}:{software_version}:{protocol_version}:"
        f"{code_digest}:{state}:{timestamp}"
    ).encode()


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


@router.post(
    "/heartbeat",
    response_model=ValidatorHeartbeatResponse,
    responses={
        401: {"description": "Invalid permit, identity, signature, or timestamp."},
        503: {"description": "Chain unavailable for the permit check."},
    },
)
async def heartbeat(
    request: Request,
    request_body: ValidatorHeartbeatRequest,
    validator_hotkey: ValidatorDep,
    session: SessionDep,
) -> ValidatorHeartbeatResponse:
    """Record a fresh, signed proof of the worker bytes serving this hotkey."""
    content_length = request.headers.get("content-length")
    try:
        claimed_bytes = int(content_length) if content_length is not None else 0
    except ValueError as error:
        raise HTTPException(status_code=400, detail="invalid Content-Length") from error
    if (
        claimed_bytes > _HEARTBEAT_MAX_BYTES
        or len(await request.body()) > _HEARTBEAT_MAX_BYTES
    ):
        raise HTTPException(status_code=413, detail="heartbeat payload too large")
    if request_body.validator_hotkey != validator_hotkey:
        raise ValidatorAuthError("heartbeat body hotkey does not match header")

    now = datetime.now(UTC)
    if abs(int(now.timestamp()) - request_body.timestamp) > _HEARTBEAT_MAX_SKEW_SECONDS:
        raise ValidatorAuthError(
            "heartbeat timestamp is stale or too far in the future"
        )
    if request_body.protocol_version < 2 and request_body.active_agent_id is not None:
        raise ValidatorAuthError("heartbeat protocol v1 cannot report active work")
    if request_body.protocol_version < 3 and request_body.system_metrics is not None:
        raise ValidatorAuthError("system metrics require heartbeat protocol v3")
    if (
        request_body.protocol_version < 4
        and request_body.benchmark_progress is not None
    ):
        raise ValidatorAuthError("benchmark progress requires heartbeat protocol v4")
    if request_body.benchmark_progress is not None and (
        request_body.active_agent_id is None
        or request_body.state != "running_benchmark"
    ):
        raise ValidatorAuthError(
            "benchmark progress requires active running_benchmark work"
        )
    if (
        request_body.system_metrics is not None
        and abs(request_body.timestamp - request_body.system_metrics.collected_at)
        > _HEARTBEAT_MAX_SKEW_SECONDS
    ):
        raise ValidatorAuthError(
            "system metrics timestamp is outside the heartbeat window"
        )
    if (
        request_body.active_agent_id is not None
        and request_body.state != "running_benchmark"
    ):
        raise ValidatorAuthError("active agent requires running_benchmark state")
    payload = _heartbeat_signing_message(
        validator_hotkey=validator_hotkey,
        software_version=request_body.software_version,
        protocol_version=request_body.protocol_version,
        code_digest=request_body.code_digest,
        state=request_body.state,
        timestamp=request_body.timestamp,
        active_agent_id=request_body.active_agent_id,
        system_metrics=request_body.system_metrics,
        benchmark_progress=request_body.benchmark_progress,
        capabilities=request_body.capabilities,
        stack=request_body.stack,
        stack_health=request_body.stack_health,
    )
    if not _verify_signature(validator_hotkey, payload, request_body.signature):
        raise ValidatorAuthError("heartbeat signature verification failed")

    reported_at = datetime.fromtimestamp(request_body.timestamp, tz=UTC)
    async with session.begin():
        stored_active_agent_id = request_body.active_agent_id
        stored_benchmark_progress = (
            request_body.benchmark_progress.model_dump(mode="json")
            if request_body.benchmark_progress is not None
            else None
        )
        if request_body.benchmark_progress is not None:
            assert request_body.active_agent_id is not None
            agent = await get_agent_by_id(
                session, agent_id=request_body.active_agent_id, for_update=True
            )
            ticket = await get_open_ticket(
                session,
                agent_id=request_body.active_agent_id,
                validator_hotkey=validator_hotkey,
                now=now,
                deadline=request_body.benchmark_progress.ticket_deadline,
                bench_version=None,
                for_update=True,
            )
            if (
                ticket is None
                or agent is None
                or agent.status not in _SCOREABLE_STATUSES
            ):
                # Ticket-bound progress is optional decoration. A benchmark can
                # outlive or lose its lease, but that must not discard an
                # otherwise valid signed liveness/health report. Persist the
                # authenticated fail-open projection without stale work context;
                # tickets, submissions, benchmarks, and scores are untouched.
                stored_active_agent_id = None
                stored_benchmark_progress = None
                logger.info(
                    "validator heartbeat dropped stale ticket-bound progress "
                    "validator=%s",
                    validator_hotkey,
                )
        # Progress monotonicity is enforced fail-open inside the query: a
        # genuine same-run regression keeps the previously stored progress
        # (never moving the public display backward) but never rejects an
        # authenticated liveness report, and a fresh run_token rebaselines.
        row, accepted = await upsert_validator_heartbeat(
            session,
            validator_hotkey=validator_hotkey,
            software_version=request_body.software_version,
            protocol_version=request_body.protocol_version,
            code_digest=request_body.code_digest,
            state=request_body.state,
            active_agent_id=stored_active_agent_id,
            system_metrics=(
                request_body.system_metrics.model_dump(mode="json")
                if request_body.system_metrics is not None
                else None
            ),
            benchmark_progress=stored_benchmark_progress,
            capabilities=(
                request_body.capabilities.model_dump(mode="json", exclude_none=True)
                if request_body.capabilities is not None
                else None
            ),
            stack=(
                request_body.stack.model_dump(mode="json")
                if request_body.stack is not None
                else None
            ),
            stack_health=(
                request_body.stack_health.model_dump(mode="json", exclude_none=True)
                if request_body.stack_health is not None
                else None
            ),
            reported_at=reported_at,
            seen_at=now,
            signature=request_body.signature,
        )
    seen_at = row.seen_at
    if seen_at.tzinfo is None:
        seen_at = seen_at.replace(tzinfo=UTC)
    return ValidatorHeartbeatResponse(accepted=accepted, seen_at=seen_at)


@router.post(
    "/job",
    response_model=JobResponse,
    responses={
        204: {"description": "No agent needs this validator right now."},
        401: {"description": "Missing/invalid validator auth."},
        426: {"description": "Validator software or protocol must be upgraded."},
        428: {"description": "A fresh signed validator heartbeat is required."},
        409: {"description": "Stale or replayed signed job claim."},
        503: {"description": "Chain unavailable for the permit check."},
    },
)
async def request_job(
    payload: JobRequest,
    request: Request,
    response: Response,
    chain: ChainDep,
    session: SessionDep,
    generator: GeneratorDep,
    x_validator_hotkey: Annotated[str | None, Header()] = None,
) -> JobResponse | Response:
    """Issue this validator a scoring ticket for the next eligible agent.

    The k=3 pull: at most :data:`SCORING_QUORUM` tickets per agent go to that
    many distinct validators, so most requests get **204 No Content** ("no job
    for you"). An issued ticket must be redeemed with a score before its
    deadline, or it lapses and the slot re-opens for another validator. The
    ticket write (and the overdue-ticket sweep it runs) commit together.
    """
    # Prove the caller owns the hotkey before it can reserve a scarce quorum
    # slot. The header remains for consistent routing/audit but must match the
    # signed body exactly.
    if x_validator_hotkey != payload.validator_hotkey:
        raise ValidatorAuthError("job claim header does not match signed hotkey")
    signed = _job_signing_message(
        payload.validator_hotkey, payload.nonce, payload.requested_at
    )
    if not _verify_signature(payload.validator_hotkey, signed, payload.signature):
        raise ValidatorAuthError(
            f"job claim signature did not verify for hotkey {payload.validator_hotkey}"
        )
    now = datetime.now(UTC)
    requested_at = payload.requested_at.astimezone(UTC)
    if abs(now - requested_at) > _JOB_REQUEST_MAX_AGE:
        raise HTTPException(status_code=409, detail="job claim timestamp is stale")

    netuid = request.app.state.config.chain.netuid
    network = request.app.state.config.chain.subtensor_network
    await _assert_validator_permitted(
        chain, netuid, payload.validator_hotkey, network=network
    )

    job: JobResponse | None = None
    async with session.begin():
        await _assert_validator_compatible(
            session,
            validator_hotkey=payload.validator_hotkey,
            now=now,
            config=request.app.state.config.validator_compatibility,
        )
        artifact_mode, validator_state = await _validator_artifact_routing(
            session,
            validator_hotkey=payload.validator_hotkey,
            now=now,
            heartbeat_max_age_seconds=(
                request.app.state.config.validator_compatibility.heartbeat_max_age_seconds
            ),
        )
        try:
            await consume_validator_nonce(
                session,
                nonce=payload.nonce,
                validator_hotkey=payload.validator_hotkey,
                now=now,
                expires_at=now + _JOB_REQUEST_MAX_AGE,
            )
        except ValidatorRequestReplayError as exc:
            raise HTTPException(
                status_code=409, detail="job claim nonce has already been used"
            ) from exc
        ticket = await issue_rollout_ticket(
            session,
            validator_hotkey=payload.validator_hotkey,
            now=now,
            ttl=_TICKET_TTL,
            artifact_mode=artifact_mode,
            validator_running_benchmark=validator_state == "running_benchmark",
        )
        canonical_version = await active_bench_version(session)
        rollout = await open_rollout(session)
        if ticket is None:
            # During an open rollout, a source-version validator may resume a
            # source-version lease. Once activation completes, only the active
            # benchmark era is resumable; retired tickets must never leak back
            # into the queue ahead of the capability gate below.
            live_ticket_statement = (
                select(ValidatorTicket)
                .join(Agent, Agent.agent_id == ValidatorTicket.agent_id)
                .where(
                    ValidatorTicket.validator_hotkey == payload.validator_hotkey,
                    ValidatorTicket.bench_version == canonical_version,
                    ValidatorTicket.status == TicketStatus.ISSUED,
                    ValidatorTicket.deadline > now,
                )
                .order_by(ValidatorTicket.issued_at.asc())
                .limit(1)
                .with_for_update()
            )
            if artifact_mode == "screened_only":
                live_ticket_statement = live_ticket_statement.where(
                    Agent.screened_image_sha256.is_not(None),
                    Agent.screened_image_size_bytes.is_not(None),
                    Agent.screened_image_id.is_not(None),
                    Agent.screened_image_ref.is_not(None),
                    Agent.screened_image_upload_id.is_not(None),
                    Agent.screened_image_verified_at.is_not(None),
                )
            if rollout is not None:
                ticket = await session.scalar(live_ticket_statement)
        if ticket is None:
            if rollout is None:
                stale_ticket = await session.scalar(
                    select(ValidatorTicket)
                    .where(
                        ValidatorTicket.validator_hotkey == payload.validator_hotkey,
                        ValidatorTicket.bench_version != canonical_version,
                        ValidatorTicket.status == TicketStatus.ISSUED,
                        ValidatorTicket.deadline > now,
                    )
                    .limit(1)
                    .with_for_update()
                )
                if stale_ticket is not None:
                    if validator_state == "running_benchmark":
                        # The signed heartbeat says this exact worker is still
                        # occupied; let it finish, but issue nothing else.
                        return Response(status_code=204)
                    stale_ticket.status = TicketStatus.EXPIRED
                    stale_ticket.deadline = now
                    stale_ticket.retry_after = now
                    await session.flush()
            heartbeat = await session.get(ValidatorHeartbeat, payload.validator_hotkey)
            if (
                rollout is not None
                and heartbeat is not None
                and heartbeat_supports_version(
                    heartbeat, now=now, version=rollout.desired_version
                )
            ):
                # Guarded admin migrations pin a canary dataset without adding
                # the submission to the top-five activation cohort. Give those
                # explicitly pinned submissions a canary-capable lane.
                ticket = await issue_ticket(
                    session,
                    validator_hotkey=payload.validator_hotkey,
                    now=now,
                    ttl=_TICKET_TTL,
                    bench_version=rollout.desired_version,
                    artifact_mode="screened_only",
                    validator_running_benchmark=validator_state == "running_benchmark",
                )
            if ticket is None:
                # Any post-legacy benchmark needs a fresh, identity-matched
                # scorer for THAT version. Keyed on the legacy floor, not on the
                # canary: an activated v3 still gates a v3-incapable validator
                # out once the canary has moved on to v4.
                gated = canonical_version > LEGACY_BENCH_VERSION and (
                    heartbeat is None
                    or not heartbeat_supports_version(
                        heartbeat, now=now, version=canonical_version
                    )
                )
                ticket = (
                    None
                    if gated
                    else await issue_ticket(
                        session,
                        validator_hotkey=payload.validator_hotkey,
                        now=now,
                        ttl=_TICKET_TTL,
                        bench_version=canonical_version,
                        artifact_mode=artifact_mode,
                        validator_running_benchmark=validator_state
                        == "running_benchmark",
                    )
                )
        if ticket is not None:
            agent = await get_agent_by_id(session, agent_id=ticket.agent_id)
            # issue_ticket selected this agent from ``agents``, so it exists.
            assert agent is not None
            dataset = await session.get(
                BenchmarkDataset, (agent.agent_id, ticket.bench_version)
            )
            contract = benchmark_contract(ticket.bench_version)
            job = JobResponse(
                agent_id=agent.agent_id,
                miner_hotkey=agent.miner_hotkey,
                sha256=agent.sha256,
                deadline=ticket.deadline,
                seed=dataset.seed if dataset is not None else agent.dataset_seed,
                dataset_sha256=(
                    dataset.sha256 if dataset is not None else agent.dataset_sha256
                ),
                run_size=(
                    dataset.run_size if dataset is not None else agent.dataset_run_size
                ),
                dataset_seed_block=(
                    dataset.seed_block
                    if dataset is not None
                    else agent.dataset_seed_block
                ),
                dataset_seed_block_hash=(
                    dataset.seed_block_hash
                    if dataset is not None
                    else agent.dataset_seed_block_hash
                ),
                bench_version=ticket.bench_version,
                minimum_screening_policy_version=(
                    contract.minimum_screening_policy_version
                ),
                requires_screened_image=contract.requires_screened_image,
            )
    if job is None:
        # Only a fully authenticated, compatible, replay-checked idle poll can
        # trigger bounded convergence. The next poll sees any newly queued work.
        await _refresh_qualification_if_due(session, generator=generator, now=now)
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
    response.headers["Cache-Control"] = "no-store"
    logger.info(
        "issued job agent=%s validator=%s deadline=%s",
        job.agent_id,
        payload.validator_hotkey,
        job.deadline.isoformat(),
    )
    return job


@router.post(
    "/job/fail",
    response_model=FailJobResponse,
    responses={
        401: {"description": "Missing/invalid validator auth or signature."},
        409: {"description": "Stale or replayed signed fail request."},
        503: {"description": "Chain unavailable for the permit check."},
    },
)
async def fail_job(
    payload: FailJobRequest,
    request: Request,
    response: Response,
    chain: ChainDep,
    session: SessionDep,
    x_validator_hotkey: Annotated[str | None, Header()] = None,
) -> FailJobResponse:
    """Hand a failed but still-leased ticket back for immediate reissue.

    A validator whose scoring attempt failed calls this so the platform closes
    the live ticket now (status ``expired``, ``deadline`` now, ``retry_after``
    now) instead of leaving the lease idle until its own deadline. The next
    ``request_job`` then mints a **fresh** ticket (new deadline) rather than
    resuming the failed lease. Additive and best-effort: an old validator that
    never calls this behaves exactly as today (the ticket expires on its own via
    the overdue sweep).

    Auth mirrors the job claim: the header must match the signed hotkey, the
    signature proves possession, ``requested_at`` is freshness-bounded, the
    nonce is consumed once, and the caller must actually hold the live ticket
    named by ``(agent_id, ticket_deadline)``.
    """
    response.headers["Cache-Control"] = "no-store"
    if x_validator_hotkey != payload.validator_hotkey:
        raise ValidatorAuthError("job-fail header does not match signed hotkey")
    signed = _job_fail_signing_message(
        payload.validator_hotkey,
        payload.agent_id,
        payload.ticket_deadline,
        payload.nonce,
        payload.requested_at,
    )
    if not _verify_signature(payload.validator_hotkey, signed, payload.signature):
        raise ValidatorAuthError(
            f"job-fail signature did not verify for hotkey {payload.validator_hotkey}"
        )
    now = datetime.now(UTC)
    requested_at = payload.requested_at.astimezone(UTC)
    if abs(now - requested_at) > _JOB_REQUEST_MAX_AGE:
        raise HTTPException(status_code=409, detail="job-fail timestamp is stale")

    netuid = request.app.state.config.chain.netuid
    network = request.app.state.config.chain.subtensor_network
    await _assert_validator_permitted(
        chain, netuid, payload.validator_hotkey, network=network
    )

    reopened = False
    async with session.begin():
        try:
            await consume_validator_nonce(
                session,
                nonce=payload.nonce,
                validator_hotkey=payload.validator_hotkey,
                now=now,
                expires_at=now + _JOB_REQUEST_MAX_AGE,
            )
        except ValidatorRequestReplayError as exc:
            raise HTTPException(
                status_code=409, detail="job-fail nonce has already been used"
            ) from exc
        # Authorize off the live ticket the caller holds (cross-version lookup on
        # the exact lease, same as the heartbeat progress path), never a
        # standalone nonce grant. A missing/expired/spent lease is a safe no-op.
        ticket = await get_open_ticket(
            session,
            agent_id=payload.agent_id,
            validator_hotkey=payload.validator_hotkey,
            now=now,
            deadline=payload.ticket_deadline,
            bench_version=None,
            for_update=True,
        )
        if ticket is not None:
            # Close for reissue without the 6h agent-failure cooldown so the
            # next request_job mints a fresh lease instead of resuming this one.
            ticket.status = TicketStatus.EXPIRED
            ticket.deadline = now
            if payload.reason == "infrastructure":
                # Not the agent's fault: bump the (bounded) infra grant that
                # offsets the coming attempt_count++, so an outage never spends
                # the agent's genuine per-version budget. Then apply an
                # escalating cooldown so a *sustained* outage isn't hammered by
                # immediate back-to-back re-leases of the same agent.
                if ticket.infra_retry_grants < MAX_INFRA_RETRY_GRANTS:
                    ticket.infra_retry_grants += 1
                ticket.retry_after = now + infra_retry_backoff(
                    ticket.infra_retry_grants
                )
            else:
                # A scoring_error is the agent's own failure: consume the budget
                # and reissue immediately for another validator/attempt.
                ticket.retry_after = now
            await session.flush()
            reopened = True
    logger.info(
        "validator=%s reported job failure agent=%s reason=%s reopened=%s",
        payload.validator_hotkey,
        payload.agent_id,
        payload.reason,
        reopened,
    )
    return FailJobResponse(agent_id=payload.agent_id, reopened=reopened)


def _stable_version(value: str) -> tuple[int, int, int] | None:
    """Parse the stable release format validators publish in heartbeats."""
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", value.strip())
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


async def _assert_validator_compatible(
    session: AsyncSession,
    *,
    validator_hotkey: str,
    now: datetime,
    config: ValidatorCompatibilityConfig,
) -> None:
    """Reject scoring work until a fresh, supported heartbeat is observed."""
    if config.minimum_software_version is None:
        return
    heartbeat = await session.get(ValidatorHeartbeat, validator_hotkey)
    if heartbeat is None:
        raise HTTPException(
            status_code=428,
            detail=(
                "validator heartbeat required before requesting work; "
                "update and restart ditto-subnet"
            ),
        )
    seen_at = heartbeat.seen_at
    if seen_at.tzinfo is None:
        seen_at = seen_at.replace(tzinfo=UTC)
    if now - seen_at > timedelta(seconds=config.heartbeat_max_age_seconds):
        raise HTTPException(
            status_code=428,
            detail=(
                "validator heartbeat is stale; confirm the current validator "
                "release is running before requesting work"
            ),
        )
    if heartbeat.protocol_version < config.minimum_protocol_version:
        raise HTTPException(
            status_code=426,
            detail=(
                f"validator protocol {heartbeat.protocol_version} is below required "
                f"{config.minimum_protocol_version}; update ditto-subnet"
            ),
        )
    current = _stable_version(heartbeat.software_version)
    minimum = _stable_version(config.minimum_software_version)
    assert minimum is not None  # validated at process boot
    if current is None or current < minimum:
        raise HTTPException(
            status_code=426,
            detail=(
                f"validator software {heartbeat.software_version!r} is below required "
                f"{config.minimum_software_version}; update ditto-subnet"
            ),
        )


async def _validator_artifact_routing(
    session: AsyncSession,
    *,
    validator_hotkey: str,
    now: datetime,
    heartbeat_max_age_seconds: int,
) -> tuple[Literal["legacy", "prefer_screened", "screened_only"], str | None]:
    """Return signed routing mode/state; pre-v7 reporters remain legacy."""
    heartbeat = await session.get(ValidatorHeartbeat, validator_hotkey)
    if heartbeat is None or heartbeat.protocol_version < 7:
        return "legacy", None
    seen_at = heartbeat.seen_at
    if seen_at.tzinfo is None:
        seen_at = seen_at.replace(tzinfo=UTC)
    if now - seen_at > timedelta(seconds=heartbeat_max_age_seconds):
        raise HTTPException(
            status_code=428,
            detail="validator heartbeat v7 is stale; report a fresh heartbeat",
        )
    try:
        capabilities = ValidatorCapabilities.model_validate_json(
            json.dumps(heartbeat.capabilities)
        )
        stack = ValidatorStackIdentity.model_validate_json(json.dumps(heartbeat.stack))
    except ValidationError as error:
        raise HTTPException(
            status_code=428,
            detail=(
                "validator heartbeat v7 capabilities are malformed; "
                "report a fresh heartbeat"
            ),
        ) from error
    if capabilities.full_stack_managed != (stack.mode == "managed"):
        raise HTTPException(
            status_code=428,
            detail="validator heartbeat v7 capabilities contradict stack identity",
        )
    return validator_artifact_mode(capabilities), heartbeat.state


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
    request: Request,
    response: Response,
    chain: ChainDep,
    session: SessionDep,
    storage: StorageDep,
    x_validator_hotkey: Annotated[str | None, Header()] = None,
    x_validator_artifact_nonce: Annotated[UUID | None, Header()] = None,
    x_validator_artifact_requested_at: Annotated[datetime | None, Header()] = None,
    x_validator_artifact_signature: Annotated[str | None, Header()] = None,
) -> ArtifactResponse:
    """Return an artifact URL after fresh proof of validator-key possession."""
    response.headers["Cache-Control"] = "no-store"
    if (
        x_validator_hotkey is None
        or not re.fullmatch(_SS58_PATTERN, x_validator_hotkey)
        or x_validator_artifact_nonce is None
        or x_validator_artifact_requested_at is None
        or x_validator_artifact_signature is None
    ):
        raise ValidatorAuthError("artifact request proof is missing or malformed")
    if x_validator_artifact_requested_at.tzinfo is None:
        raise ValidatorAuthError("artifact request timestamp must include a timezone")
    signed = _artifact_signing_message(
        x_validator_hotkey,
        agent_id,
        x_validator_artifact_nonce,
        x_validator_artifact_requested_at,
    )
    if not _verify_signature(
        x_validator_hotkey, signed, x_validator_artifact_signature
    ):
        raise ValidatorAuthError("artifact request signature did not verify")
    now = datetime.now(UTC)
    if (
        abs(now - x_validator_artifact_requested_at.astimezone(UTC))
        > _JOB_REQUEST_MAX_AGE
    ):
        raise HTTPException(
            status_code=409, detail="artifact request timestamp is stale"
        )
    await _assert_validator_permitted(
        chain,
        request.app.state.config.chain.netuid,
        x_validator_hotkey,
        network=request.app.state.config.chain.subtensor_network,
    )
    async with session.begin():
        try:
            await consume_validator_nonce(
                session,
                nonce=x_validator_artifact_nonce,
                validator_hotkey=x_validator_hotkey,
                now=now,
                expires_at=now + _JOB_REQUEST_MAX_AGE,
            )
        except ValidatorRequestReplayError as exc:
            raise HTTPException(
                status_code=409,
                detail="artifact request nonce has already been used",
            ) from exc
        agent = await get_agent_by_id(session, agent_id=agent_id)
        if agent is None:
            raise AgentNotFoundError(f"no agent with id={agent_id}")
        ticket = await session.scalar(
            select(ValidatorTicket).where(
                ValidatorTicket.agent_id == agent_id,
                ValidatorTicket.validator_hotkey == x_validator_hotkey,
                ValidatorTicket.status == TicketStatus.ISSUED,
                ValidatorTicket.deadline > now,
            )
        )
    url = await storage.presigned_get_url(
        key=_artifact_key(agent_id),
        expires_in=int(_ARTIFACT_URL_TTL.total_seconds()),
    )
    image_url = None
    if (
        agent.screened_image_sha256 is not None
        and agent.screened_image_upload_id is not None
    ):
        image_url = await storage.presigned_get_url(
            key=_screened_image_key(agent_id, agent.screened_image_upload_id),
            expires_in=int(_ARTIFACT_URL_TTL.total_seconds()),
        )
    logger.info(
        "validator=%s fetched artifact url for agent_id=%s",
        x_validator_hotkey,
        agent_id,
    )
    return ArtifactResponse(
        agent_id=agent_id,
        sha256=agent.sha256,
        download_url=url,
        expires_at=datetime.now(UTC) + _ARTIFACT_URL_TTL,
        screened_image_url=image_url,
        screened_image_sha256=agent.screened_image_sha256,
        screened_image_size_bytes=agent.screened_image_size_bytes,
        screened_image_id=agent.screened_image_id,
        screened_image_ref=agent.screened_image_ref,
        bench_version=ticket.bench_version if ticket is not None else None,
        screening_policy_version=agent.screening_policy_version,
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
    storage: StorageDep,
    generator: GeneratorDep,
) -> SubmitScoreResponse:
    """Record a DittoBench score report and advance the agent's lifecycle.

    Ordering is cheap-before-expensive and no DB write happens until every
    check passes: (1) signature over ``{validator_hotkey}:{run_id}``,
    (2) on-chain validator-permit check, (3) one transaction that upserts
    the score and, once the k=3 quorum has reported, finalizes the agent
    ``evaluating -> scored`` on the median composite. Below quorum the score
    is recorded and the agent stays provisional (``evaluating``).
    """
    response.headers["Cache-Control"] = "no-store"
    report = payload.report

    # 1. Signature proves the reporting validator owns the hotkey and binds the
    #    agent + score contents (anti-replay / anti-tamper). CPU-only, no I/O.
    signed = _score_signing_message(
        payload.validator_hotkey, agent_id, payload.ticket_deadline, report
    )
    if not _verify_signature(payload.validator_hotkey, signed, payload.signature):
        raise ValidatorAuthError(
            f"score signature did not verify for hotkey {payload.validator_hotkey}"
        )

    # 2. The hotkey must be a permitted validator on this subnet.
    netuid = request.app.state.config.chain.netuid
    network = request.app.state.config.chain.subtensor_network
    await _assert_validator_permitted(
        chain, netuid, payload.validator_hotkey, network=network
    )

    # 3. Atomic: record the score + advance status together. The row lock
    #    serializes concurrent scorers so the status guard + transition below
    #    can't be lost-updated.
    async with session.begin():
        agent = await get_agent_by_id(
            session, agent_id=agent_id, for_update=True, include_anticopy=True
        )
        if agent is None:
            raise AgentNotFoundError(f"no agent with id={agent_id}")
        if agent.status not in _SCOREABLE_STATUSES:
            raise AgentNotEvaluatableError(
                f"agent {agent_id} is {agent.status}, not in {_SCOREABLE_STATUSES}"
            )
        if agent.screening_policy_version < SCREENING_POLICY_VERSION:
            raise AgentNotEvaluatableError(
                f"agent {agent_id} has not passed screening policy "
                f"{SCREENING_POLICY_VERSION}"
            )
        if payload.ticket_deadline is None:
            raise HTTPException(
                status_code=409,
                detail="score submission is missing its ticket lease deadline",
            )
        # k=3 gate: a score is only accepted against a live ticket this validator
        # holds for the agent. No ticket (never issued, expired, or already
        # spent) means the score is unsolicited or late, so it is rejected and
        # the slot is left for a validator that will score in time. One ticket,
        # one score: the ticket is consumed below, so a re-score needs a new one.
        ticket = await get_open_ticket(
            session,
            agent_id=agent_id,
            validator_hotkey=payload.validator_hotkey,
            now=datetime.now(UTC),
            deadline=payload.ticket_deadline,
            # A validator on the old protocol omits bench_version. Falling back
            # to CURRENT would send every legacy submission hunting a ticket
            # for whatever version is current, find none, and 409. A version-less
            # report means v2 by definition, so pin the frozen legacy version --
            # NOT the rollout's from_version, which moves.
            bench_version=(report.bench_version or LEGACY_BENCH_VERSION),
            for_update=True,
        )
        if ticket is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "no open scoring ticket for this validator and agent "
                    "(never issued, expired, or already scored)"
                ),
            )
        # Every post-legacy benchmark must be bound EXPLICITLY, not just the
        # current canary: a v3 ticket keeps this requirement after the canary
        # moves to v4, instead of silently falling through to the lenient branch.
        if ticket.bench_version > LEGACY_BENCH_VERSION:
            if report.bench_version != ticket.bench_version:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"benchmark v{ticket.bench_version} score must explicitly "
                        f"bind bench_version={ticket.bench_version}"
                    ),
                )
        elif report.bench_version not in (None, ticket.bench_version):
            raise HTTPException(
                status_code=409,
                detail="score benchmark version does not match its ticket lease",
            )
        existing_score = await session.get(
            Score, (agent_id, ticket.bench_version, payload.validator_hotkey)
        )
        replacement_event = None
        if existing_score is not None:
            latest_retest = await get_latest_score_retest_event(
                session,
                agent_id=agent_id,
                validator_hotkey=payload.validator_hotkey,
            )
            if (
                latest_retest is None
                or latest_retest.event != EVENT_SCORE_RETEST_REQUESTED
            ):
                if agent.status not in {AgentStatus.SCORED, AgentStatus.LIVE}:
                    latest_retest = None
                else:
                    raise HTTPException(
                        status_code=409,
                        detail="accepted score has no operator-authorized re-test",
                    )
            if latest_retest is None:
                replacement_event = None
            else:
                if (
                    int(latest_retest.payload.get("bench_version", -1))
                    != ticket.bench_version
                ):
                    raise HTTPException(
                        status_code=409,
                        detail="replacement request benchmark version changed",
                    )
                if latest_retest.payload.get("run_id") != existing_score.run_id:
                    raise HTTPException(
                        status_code=409,
                        detail="accepted score changed after replacement request",
                    )
                replacement_event = latest_retest
        # Persist the scoring engine's opaque telemetry (models used,
        # bench_version, dataset_sha256, per-category means, token spend, …) plus
        # the per-case breakdown, all under scores.details. The public leaderboard
        # surfaces a safe subset of this; the full blob (incl. per_case answer-key
        # fields) is only ever read back through validator-gated endpoints.
        score_details: dict[str, Any] = dict(report.details or {})
        # Persist the exact lease identity alongside the signature so public
        # records remain independently verifiable. Existing scores have no such
        # key and intentionally remain valid legacy records.
        score_details["ticket_deadline"] = _lease_token(payload.ticket_deadline)
        # Stamp the current benchmark version when the scorer omitted it, so no
        # run scored from now on is ever recorded as "legacy" (null version).
        # An explicit version in the report is left as-is (honest provenance).
        score_details["bench_version"] = ticket.bench_version
        # Stash the composite standard error into details so the ledger can
        # surface it (mirroring bench_version; no schema migration). The
        # validator reads it back for the KOTH indifference band.
        if report.composite_stderr is not None:
            score_details["composite_stderr"] = report.composite_stderr
        # Same for the P4 per-seed confirmation composites: the validator submits
        # one median-run score carrying the K per-seed composites, and the ledger
        # surfaces them so the KOTH fold dethrones on the median over seeds.
        if report.confirmation_composites is not None:
            score_details["confirmation_composites"] = report.confirmation_composites
        # The K CRN seeds aligned 1:1 with those composites, so the fold can pair
        # a challenger against the champion on shared seeds (paired-difference
        # variance) instead of the wider independent-sum band.
        if report.confirmation_seeds is not None:
            score_details["confirmation_seeds"] = report.confirmation_seeds
        if report.per_case:
            score_details["per_case"] = [
                c.model_dump(mode="json") for c in report.per_case
            ]
        audit_now = datetime.now(UTC)
        if replacement_event is not None and agent.status in {
            AgentStatus.SCORED,
            AgentStatus.LIVE,
        }:
            assert existing_score is not None
            await append_audit_entry(
                session,
                agent_id=agent_id,
                validator_hotkey=payload.validator_hotkey,
                event=EVENT_SCORE_INVALIDATED,
                payload={
                    "request_id": replacement_event.payload["request_id"],
                    "actor": replacement_event.payload["actor"],
                    "reason": replacement_event.payload["reason"],
                    "bench_version": ticket.bench_version,
                    "run_id": existing_score.run_id,
                    "invalidated_score": {
                        "run_id": existing_score.run_id,
                        "seed": existing_score.seed,
                        "composite": existing_score.composite,
                        "tool_mean": existing_score.tool_mean,
                        "memory_mean": existing_score.memory_mean,
                        "median_ms": existing_score.median_ms,
                        "n": existing_score.n,
                        "bench_version": existing_score.bench_version,
                        "ticket_deadline": (
                            existing_score.details.get("ticket_deadline")
                            if isinstance(existing_score.details, dict)
                            else None
                        ),
                        "signature": existing_score.signature,
                        "generated_at": existing_score.generated_at.isoformat(),
                    },
                    "replacement_run_id": report.run_id,
                    "replacement_composite": report.composite,
                },
                recorded_at=audit_now,
            )
        await upsert_score(
            session,
            agent_id=agent_id,
            validator_hotkey=payload.validator_hotkey,
            bench_version=ticket.bench_version,
            run_id=report.run_id,
            seed=report.seed,
            composite=report.composite,
            tool_mean=report.tool_mean,
            memory_mean=report.memory_mean,
            median_ms=report.median_ms,
            n=report.n,
            generated_at=report.generated_at,
            signature=payload.signature,
            details=score_details or None,
        )
        # Append the immutable, hash-chained audit entry for this score in the
        # same transaction (durable iff the score is). Records the full signed
        # tuple + signature so the entry is independently verifiable off the
        # public audit feed, never any per-case answer-key content.
        await append_audit_entry(
            session,
            agent_id=agent_id,
            validator_hotkey=payload.validator_hotkey,
            event=EVENT_SCORE,
            payload={
                "run_id": report.run_id,
                "seed": report.seed,
                "composite": report.composite,
                "tool_mean": report.tool_mean,
                "memory_mean": report.memory_mean,
                "median_ms": report.median_ms,
                "n": report.n,
                "bench_version": ticket.bench_version,
                "ticket_deadline": _lease_token(payload.ticket_deadline),
                "signature": payload.signature,
                "generated_at": report.generated_at.isoformat(),
            },
            recorded_at=audit_now,
        )
        if replacement_event is not None:
            replacement_scores = await list_scores_for_agent(
                session, agent_id=agent_id, bench_version=ticket.bench_version
            )
            replacement_median = statistics.median(
                score.composite for score in replacement_scores
            )
            await append_audit_entry(
                session,
                agent_id=agent_id,
                validator_hotkey=None,
                event=EVENT_FINALIZED,
                payload={
                    "miner_hotkey": agent.miner_hotkey,
                    "median_composite": replacement_median,
                    "quorum": SCORING_QUORUM,
                    "score_count": len(replacement_scores),
                    "validator_hotkeys": sorted(
                        score.validator_hotkey for score in replacement_scores
                    ),
                    "dataset_seed": agent.dataset_seed,
                    "dataset_sha256": agent.dataset_sha256,
                    "dataset_seed_block": agent.dataset_seed_block,
                    "dataset_seed_block_hash": agent.dataset_seed_block_hash,
                    "status": agent.status.value,
                    "replacement_request_id": replacement_event.payload["request_id"],
                    "replaced_run_id": replacement_event.payload["run_id"],
                },
                recorded_at=audit_now,
            )
            await _publish_finalized_run(
                storage,
                agent=agent,
                scores=replacement_scores,
                median=replacement_median,
            )
        # Persist the crate's structural (AST) fingerprint from the report, so it
        # is available for the gate here and for future cross-miner comparison.
        # Advisory + unsigned: only overwrite when the report actually carries one,
        # so a re-score by a scorer that omits it never wipes a stored sketch.
        if report.structural_fingerprint is not None:
            agent.structural_fingerprint = report.structural_fingerprint.model_dump()
        # Finalize at quorum (k=3): an agent stays provisional (``evaluating``)
        # until :data:`SCORING_QUORUM` validators have scored it; only the
        # quorum-th score moves it ``evaluating -> scored``, unless the anti-copy
        # gate holds a suspected copy in ``ath_pending_review``. Both the gate
        # and the transition run on the **median** composite, so no single
        # validator's score decides an agent's fate. The gate runs only on this
        # one transition; a re-score of an already-scored (or held) agent leaves
        # its status put so re-reporting never thrashes the ledger. The agent is
        # still ``evaluating`` here, so it is not yet in the eligible ledger (no
        # self-match). A below-quorum score just records the row and waits.
        if agent.status == AgentStatus.EVALUATING:
            agent_scores = await list_scores_for_agent(
                session, agent_id=agent_id, bench_version=ticket.bench_version
            )
            if len(agent_scores) >= SCORING_QUORUM:
                median_composite = statistics.median(s.composite for s in agent_scores)
                eligible = await list_eligible_ledger(session)
                decision = evaluate_duplicate_signals(
                    agent_id=agent_id,
                    miner_hotkey=agent.miner_hotkey,
                    submitted_at=agent.created_at,
                    sha256=agent.sha256,
                    composite=median_composite,
                    size_bytes=agent.size_bytes,
                    normalized_source_hash=agent.normalized_source_hash,
                    content_fingerprint=agent.content_fingerprint,
                    structural_fingerprint=agent.structural_fingerprint,
                    prompt_fingerprint=agent.prompt_fingerprint,
                    eligible=eligible,
                )
                if decision.held:
                    reference_provenance = reference_corpus_provenance()
                    agent.status = AgentStatus.ATH_PENDING_REVIEW
                    agent.duplicate_of = decision.duplicate_of
                    agent.review_reason = decision.reason
                    session.add(
                        AthReview(
                            review_id=uuid4(),
                            agent_id=agent.agent_id,
                            status="pending",
                            opened_at=audit_now,
                            original_duplicate_of=decision.duplicate_of,
                            original_reason=decision.reason,
                            original_policy_version=agent.screening_policy_version,
                            original_evidence={
                                "content_fingerprint_version": (
                                    agent.content_fingerprint or {}
                                ).get("v"),
                                "structural_fingerprint_version": (
                                    agent.structural_fingerprint or {}
                                ).get("v"),
                                "prompt_fingerprint_version": (
                                    agent.prompt_fingerprint or {}
                                ).get("v"),
                            },
                            algorithm_provenance={
                                "snapshot": "score-finalization",
                                "algorithm_version": ANTI_COPY_ALGORITHM_VERSION,
                                "canonical_reference_revision": (
                                    reference_provenance["revision"]
                                ),
                                "reference_corpus_id": reference_provenance[
                                    "corpus_id"
                                ],
                                "reference_exclusion_mode": reference_provenance[
                                    "exclusion_mode"
                                ],
                                "backfilled": False,
                                "opened_at_source": "agent_finalized_audit",
                            },
                        )
                    )
                    logger.warning(
                        "agent %s held for copy review: %s", agent_id, decision.reason
                    )
                else:
                    agent.status = AgentStatus.SCORED
                # Reproduce-under-transform audit (v3 Part A). A share of every
                # run's cases is re-asked under a transform derived from the
                # block-hash-seeded dataset seed, which postdates the commit; the
                # validator reports the median robustness over its confirmation
                # runs. Below the public floor, the agent goes to review instead
                # of scored, so it is excluded from emissions until an operator
                # resolves it -- exactly like the copy-review hold.
                #
                # Quarantine-then-review, never an auto-ban: a low value is the
                # surface-brittleness or memorization signature, and it is NOT
                # evidence about a harness that genuinely recomputes its answers
                # (that one scores the same under the transform). It reuses
                # ATH_PENDING_REVIEW with a distinct review_reason rather than
                # adding a sibling status, which would force a
                # ditto-screening-protocol pin bump across every consumer for a
                # distinction the reason string already carries.
                audit_pvalue, audit_pairs, audit_failed = _transform_audit_verdict(
                    agent_scores
                )
                if audit_failed and not TRANSFORM_AUDIT_ENFORCE:
                    # Observational mode: record what the verdict WOULD have been
                    # (the EVENT_AUDIT entry below carries `failed`) without
                    # touching the agent's status. This is what accumulates the
                    # real-world distribution a future threshold can be set from.
                    logger.info(
                        "agent %s: transform-audit brittleness signature "
                        "(%d base-only vs %d transform-only, p=%.4f <= %.3f) "
                        "— NOT enforced pending champion-population validation",
                        agent_id,
                        audit_pairs["base_only"],
                        audit_pairs["transform_only"],
                        audit_pvalue if audit_pvalue is not None else 1.0,
                        AUDIT_ALPHA,
                    )
                if (
                    audit_failed
                    and TRANSFORM_AUDIT_ENFORCE
                    and agent.status == AgentStatus.SCORED
                ):
                    agent.status = AgentStatus.ATH_PENDING_REVIEW
                    agent.review_reason = TRANSFORM_AUDIT_REVIEW_REASON
                    session.add(
                        AthReview(
                            review_id=uuid4(),
                            agent_id=agent.agent_id,
                            status="pending",
                            opened_at=audit_now,
                            original_reason=TRANSFORM_AUDIT_REVIEW_REASON,
                            original_policy_version=agent.screening_policy_version,
                            original_evidence={
                                "audit_pairs": audit_pairs,
                                "transform_audit_pvalue": audit_pvalue,
                                "audit_alpha": AUDIT_ALPHA,
                            },
                            algorithm_provenance={
                                "snapshot": "score-finalization",
                                "opened_at_source": "transform_audit",
                            },
                        )
                    )
                    logger.warning(
                        "agent %s held for transform-audit review: %d base-only "
                        "vs %d transform-only discordant pairs, p=%.4f <= %.3f",
                        agent_id,
                        audit_pairs["base_only"],
                        audit_pairs["transform_only"],
                        audit_pvalue if audit_pvalue is not None else 1.0,
                        AUDIT_ALPHA,
                    )
                if sum(audit_pairs.values()) > 0:
                    # Recorded whether or not it held, so the public feed shows
                    # the audit ran and what it found -- not only its failures.
                    # PUBLIC INPUTS ONLY: never a transformed expected answer or
                    # any other answer-key material, the same redaction rule the
                    # score entry follows. Everything here is either already
                    # published or re-derivable from the published seed, so a
                    # third party can recompute this verdict independently.
                    await append_audit_entry(
                        session,
                        agent_id=agent_id,
                        validator_hotkey=None,
                        event=EVENT_AUDIT,
                        payload={
                            "miner_hotkey": agent.miner_hotkey,
                            "audit_pairs": audit_pairs,
                            "transform_audit_pvalue": audit_pvalue,
                            "audit_alpha": AUDIT_ALPHA,
                            "audit_bps": AUDIT_BPS,
                            "failed": audit_failed,
                            # Whether the verdict was allowed to affect status.
                            # Published so the feed is unambiguous about which
                            # entries were observational.
                            "enforced": TRANSFORM_AUDIT_ENFORCE,
                            "dataset_seed": agent.dataset_seed,
                            "dataset_sha256": agent.dataset_sha256,
                            "dataset_seed_block": agent.dataset_seed_block,
                            "dataset_seed_block_hash": agent.dataset_seed_block_hash,
                            "score_count": len(agent_scores),
                        },
                        recorded_at=audit_now,
                    )
                # Append the finalize audit entry: quorum reached, the median the
                # platform finalized on, and which validators scored it. The
                # moderation detail (why held / duplicate_of) is deliberately kept
                # out of the public chain — only the neutral outcome status.
                await append_audit_entry(
                    session,
                    agent_id=agent_id,
                    validator_hotkey=None,
                    event=EVENT_FINALIZED,
                    payload={
                        "miner_hotkey": agent.miner_hotkey,
                        "median_composite": median_composite,
                        "quorum": SCORING_QUORUM,
                        "score_count": len(agent_scores),
                        "validator_hotkeys": sorted(
                            s.validator_hotkey for s in agent_scores
                        ),
                        "dataset_seed": agent.dataset_seed,
                        "dataset_sha256": agent.dataset_sha256,
                        "dataset_seed_block": agent.dataset_seed_block,
                        "dataset_seed_block_hash": agent.dataset_seed_block_hash,
                        "status": agent.status.value,
                    },
                    recorded_at=audit_now,
                )
                # Transparency mirror: publish the finalized run record to the
                # public bucket so third parties can verify signatures and
                # re-grade offline without touching the API. Additive and
                # fail-open: the canonical record is Postgres; a publish
                # failure logs and never fails the score write. Idempotent by
                # key, so a retried request republishes identical content.
                await _publish_finalized_run(
                    storage, agent=agent, scores=agent_scores, median=median_composite
                )
        # Consume the ticket (one ticket, one score); the slot stays occupied.
        await mark_ticket_scored(
            session,
            agent_id=agent_id,
            validator_hotkey=payload.validator_hotkey,
            bench_version=ticket.bench_version,
        )
        result_status = agent.status

    # Both a completed v3 quorum and a newly finalized v2 contender can change
    # the hybrid top five. This is a cheap no-op when no rollout is open.
    try:
        await refresh_rolling_qualification(session, generator=generator, now=audit_now)
    except Exception:
        # The score is already committed and remains canonical. Do not report a
        # false score failure because the independent v3 dataset renderer is
        # temporarily unavailable; the next score/verdict/admin retry converges.
        logger.exception("rolling benchmark qualification refresh failed")

    logger.info(
        "score recorded agent_id=%s validator=%s run_id=%s composite=%.3f status=%s",
        agent_id,
        payload.validator_hotkey,
        report.run_id,
        report.composite,
        result_status,
    )
    return SubmitScoreResponse(agent_id=agent_id, status=result_status, accepted=True)


async def _publish_finalized_run(
    storage: S3StorageClient,
    *,
    agent: Agent,
    scores: Sequence[Score],
    median: float,
) -> None:
    """Mirror a finalized run to the public bucket (``scored/{agent_id}.json``).

    The record carries everything an offline verifier needs: the dataset pin
    (seed, sha256, seed block), the k=3 signed scores with their full details
    (per-case breakdown included), and the median the platform finalized on.
    Current signatures cover
    ``{hotkey}:{agent_id}:{ticket_deadline}:{run_id}:{composite!r}:{seed}``;
    legacy scores have no ``ticket_deadline`` detail and retain the previous
    payload format. The record therefore carries the lease identity needed to
    verify either generation against the validator's on-chain hotkey.
    No-op when ``STORAGE_PUBLIC_BUCKET`` is unset; failures log only.
    """
    if storage.public_bucket is None:
        return
    record = {
        "agent_id": str(agent.agent_id),
        "miner_hotkey": agent.miner_hotkey,
        "status": agent.status.value,
        "median_composite": median,
        "dataset_seed": agent.dataset_seed,
        "dataset_sha256": agent.dataset_sha256,
        "dataset_run_size": agent.dataset_run_size,
        "dataset_seed_block": agent.dataset_seed_block,
        "dataset_seed_block_hash": agent.dataset_seed_block_hash,
        "scores": [
            {
                "validator_hotkey": sc.validator_hotkey,
                "run_id": sc.run_id,
                "ticket_deadline": (
                    sc.details.get("ticket_deadline")
                    if isinstance(sc.details, dict)
                    else None
                ),
                "seed": sc.seed,
                "composite": sc.composite,
                "tool_mean": sc.tool_mean,
                "memory_mean": sc.memory_mean,
                "median_ms": sc.median_ms,
                "n": sc.n,
                "generated_at": sc.generated_at.isoformat()
                if sc.generated_at
                else None,
                "signature": sc.signature,
                # Where the validator's transcript artifact lives (finding 3):
                # the digest is inside the signed payload; the key is derived
                # from it, so the record always names immutable bytes. Null for
                # scores whose validator published no transcript.
                "transcript_sha256": digest,
                "transcript_key": transcript_object_key(digest) if digest else None,
                "details": sc.details,
            }
            for sc, digest in (
                (sc, _score_transcript_sha256(sc))
                for sc in sorted(scores, key=lambda sc: sc.validator_hotkey)
            )
        ],
    }
    body = json.dumps(record, sort_keys=True, default=str).encode()
    try:
        await storage.put_object(
            key=f"scored/{agent.agent_id}.json",
            body=body,
            content_type="application/json",
            bucket=storage.public_bucket,
        )
    except Exception:  # noqa: BLE001 - additive mirror, never fail the write
        logger.exception("public mirror publish failed for agent %s", agent.agent_id)


# Transcript artifacts are content-addressed in the public bucket so a record
# referencing a digest always names immutable bytes.
_TRANSCRIPT_KEY_TEMPLATE = "transcripts/{sha256}.json"

# A transcript carries every graded final_text for a full run; cap well above
# any legitimate size while bounding a hostile body.
_TRANSCRIPT_MAX_BYTES = 32 << 20

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def transcript_object_key(sha256_hex: str) -> str:
    """Public-bucket key for a transcript digest."""
    return _TRANSCRIPT_KEY_TEMPLATE.format(sha256=sha256_hex)


def _score_transcript_sha256(score: Score) -> str | None:
    """The well-formed transcript digest a stored score declares, or ``None``."""
    details = score.details if isinstance(score.details, dict) else {}
    value = details.get("transcript_sha256")
    if isinstance(value, str) and _SHA256_HEX.fullmatch(value):
        return value
    return None


@router.put(
    "/agent/{agent_id}/transcript/{run_id}",
    response_model=SubmitTranscriptResponse,
)
async def submit_transcript(
    agent_id: UUID,
    run_id: str,
    request: Request,
    response: Response,
    session: SessionDep,
    validator: ValidatorDep,
    storage: StorageDep,
) -> SubmitTranscriptResponse:
    """Publish the transcript artifact behind a signed score (finding 3).

    The body is the scoring engine's canonical transcript for ``run_id`` — the
    graded per-case inputs whose digest the validator declared under
    ``details["transcript_sha256"]`` and bound into its score signature. The
    platform accepts the bytes only when their SHA-256 equals that declared
    digest, then stores them content-addressed in the public bucket. Because
    the binding is *content* equality against an already-signed digest, a
    caller spoofing another validator's hotkey can only ever upload the exact
    bytes that validator attested — so the header + permit check is sufficient
    auth here. Idempotent: re-uploading an existing digest is a no-op.
    """
    response.headers["Cache-Control"] = "no-store"
    body = await request.body()
    if len(body) > _TRANSCRIPT_MAX_BYTES:
        raise HTTPException(status_code=413, detail="transcript exceeds size cap")
    if not body:
        raise HTTPException(status_code=400, detail="empty transcript body")
    digest = hashlib.sha256(body).hexdigest()

    score = await get_score_for_validator(
        session, agent_id=agent_id, validator_hotkey=validator
    )
    if score is None or score.run_id != run_id:
        raise HTTPException(
            status_code=409,
            detail=(
                "no recorded score by this validator for this agent and run; "
                "submit the score (with details.transcript_sha256) first"
            ),
        )
    declared = (
        score.details.get("transcript_sha256")
        if isinstance(score.details, dict)
        else None
    )
    if not isinstance(declared, str) or not _SHA256_HEX.fullmatch(declared):
        raise HTTPException(
            status_code=409,
            detail="the recorded score declares no transcript_sha256",
        )
    if digest != declared:
        raise HTTPException(
            status_code=409,
            detail=(
                f"transcript bytes hash to {digest} but the signed score "
                f"declared {declared}"
            ),
        )

    if storage.public_bucket is None:
        logger.warning(
            "transcript %s for agent %s accepted but STORAGE_PUBLIC_BUCKET is "
            "unset; nothing published",
            digest,
            agent_id,
        )
        return SubmitTranscriptResponse(
            agent_id=agent_id, run_id=run_id, transcript_sha256=digest, stored=False
        )
    key = transcript_object_key(digest)
    if not await storage.object_exists(key=key, bucket=storage.public_bucket):
        await storage.put_object(
            key=key,
            body=body,
            content_type="application/json",
            bucket=storage.public_bucket,
        )
        logger.info(
            "transcript published agent_id=%s run_id=%s sha256=%s bytes=%d",
            agent_id,
            run_id,
            digest,
            len(body),
        )
    return SubmitTranscriptResponse(
        agent_id=agent_id, run_id=run_id, transcript_sha256=digest, stored=True
    )
