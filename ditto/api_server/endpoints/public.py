"""Public, unauthenticated read endpoints for the subnet dashboard.

Three surfaces, all open (no credentials) and fronting the same DB the
validator-gated ``/scoring/scores`` reads:

* **Aggregate leaderboard / health** (``/leaderboard``, ``/health``): best score
  per miner, composite plus tool/memory means and rank, never exposing per-case
  answer-key detail. This half stays aggregate-only.
* **Submission lifecycle** (``/activity``, ``/agent/{id}/pipeline``): recent
  uploads, public pipeline stage, safe screening evidence, and accepted numeric
  scores as they arrive. In-progress score rows carry reproducibility inputs but
  omit validator identity, signatures, ticket leases, and scorer internals.
* **Per-submission transparency** (``/submissions``, ``/agent/{id}/scores``): the
  k=3 record for a finalized agent — *which* validators scored it, each one's
  exact numbers + signature, the median the platform finalized on, and the pinned
  dataset (seed + sha256). This deliberately exposes ``validator_hotkey`` (a
  public on-chain identity) and the raw ``seed`` so anyone can reproduce and audit
  a score; because the platform draws the seed after screening, publishing it
  post-hoc never lets a miner pre-overfit. It still omits the per-case answer key.
  See ``docs/public-telemetry.md``.

Responses are cacheable (``max-age=30``) so a CDN / the dashboard can front this
cheaply; the underlying rows only change when a sweep records a new score. The
leaderboard includes a read-only projection of the validator's frozen KOTH fold
so raw score rank is never mistaken for the emissions champion. Validators remain
the authority that independently computes and submits the real weight vector.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import math
import os
import re
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal, cast
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models import (
    BenchDatasetConfig,
    BenchGradingConfig,
    BenchHarnessConfig,
    CreateScreeningDisputeRequest,
    CreateScreeningDisputeResponse,
    PublicActivityEntry,
    PublicActivityResponse,
    PublicAuditEntry,
    PublicAuditResponse,
    PublicBenchConfigResponse,
    PublicBenchCorpusEntry,
    PublicBenchCorpusResponse,
    PublicBenchIntegrity,
    PublicBenchmarkProgress,
    PublicCaseResult,
    PublicCategoryStat,
    PublicChainWeight,
    PublicChainWeightsResponse,
    PublicDatasetReveal,
    PublicDethroneDecision,
    PublicEmissionRecipient,
    PublicHealthResponse,
    PublicKothEmissions,
    PublicLeaderboardEntry,
    PublicLeaderboardResponse,
    PublicOperationsResponse,
    PublicProvisionalScore,
    PublicRunModels,
    PublicScreenerHeartbeat,
    PublicScreenerHeartbeatsResponse,
    PublicScreenerProgress,
    PublicScreeningAttempt,
    PublicScreeningDispute,
    PublicSubmissionPipeline,
    PublicSubmissionScores,
    PublicSubmissionsResponse,
    PublicSubmissionSummary,
    PublicSystemMetrics,
    PublicValidationAttempt,
    PublicValidatorHeartbeat,
    PublicValidatorHeartbeatsResponse,
    PublicValidatorName,
    PublicValidatorNamesResponse,
    PublicValidatorScore,
    PublicValidatorWeightVector,
)
from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.public import (
    FleetAvailability,
    FleetHealth,
    ValidatorAssignmentState,
)
from ditto.api_models.screener import (
    SCREENING_POLICY_VERSION,
    ScreenerProgress,
    ScreenerRuntimeState,
)
from ditto.api_models.stack_health import ValidatorStackHealth
from ditto.api_models.system_health import SystemMetrics
from ditto.api_models.ticket_status import TicketStatus
from ditto.api_models.validator import ValidatorRuntimeState
from ditto.api_models.validator_capabilities import (
    ValidatorCapabilities,
    ValidatorStackIdentity,
)
from ditto.api_server.bench import CURRENT_BENCH_VERSION, is_bench_version_retired
from ditto.api_server.datapipeline import DataPipelineError
from ditto.api_server.endpoints.scoring import (
    _confirmation_composites,
    _confirmation_seeds,
    _ledger_stderr,
)
from ditto.api_server.endpoints.screener import GeneratorDep
from ditto.api_server.endpoints.upload import _verify_signature
from ditto.api_server.endpoints.validator import SessionDep
from ditto.api_server.koth import (
    KOTH_CHAMPION_SHARE,
    KOTH_DETHRONE_Z,
    KOTH_MARGIN,
    KOTH_TAIL_SIZE,
    KothEntry,
    project_koth,
)
from ditto.chain import ChainError
from ditto.db.models import (
    Agent,
    AthReview,
    Score,
    ScreeningDispute,
    ScreeningQuarantine,
    ValidatorTicket,
)
from ditto.db.queries.agents import list_public_activity
from ditto.db.queries.audit import GENESIS_HASH, list_audit_entries
from ditto.db.queries.benchmark_rollout import active_bench_version, rollout_state
from ditto.db.queries.heartbeats import (
    ActiveValidatorAssignment,
    ActiveValidatorWork,
    list_active_validator_assignments,
    list_active_validator_work,
    list_screener_heartbeats,
    list_validator_heartbeats,
)
from ditto.db.queries.scores import (
    SCORING_QUORUM,
    LedgerRow,
    SubmissionRow,
    get_public_health,
    get_score_counts,
    get_submission_scores,
    list_eligible_ledger,
    list_miner_composite_history,
    list_provisional_ledger,
    list_public_submissions,
    list_scores_for_bench_version,
    quorum_composites,
)
from ditto.db.queries.screening import (
    get_running_screening_attempts,
    list_screening_attempts,
)
from ditto.db.queries.tickets import (
    PROVISIONAL_CONTENDER_LANE_SIZE,
    get_score_continuation_floor,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/public", tags=["public"])

# The ledger only moves when a sweep records a new best score, so a short shared
# cache is safe and shields the DB from dashboard/CDN traffic.
_CACHE_CONTROL = "public, max-age=30"
_REGISTRATION_LOOKUP_TIMEOUT_SECONDS = 1.0
_REGISTRATION_CACHE_TTL_SECONDS = 15.0
_REGISTRATION_FAILURE_CACHE_TTL_SECONDS = 5.0
_CHAIN_WEIGHTS_TIMEOUT_SECONDS = 4.0
# Historical reproduction must fail closed: only benchmark epochs whose exact
# generator release is known get a copyable command. Add a mapping deliberately
# when a future epoch pins its generator; never point an old score at ``latest``.

# The exact generator release each benchmark version's reproduction commands
# pin. v0.8.0 is the tag cut from dittobench-datagen's anti-gaming branch at
# the v3 release (see dittobench-api docs/v3-release.md for the merge order).
_DATAGEN_VERSION_BY_BENCH_VERSION = {2: "v0.7.0", 3: "v0.8.0"}
_DATAGEN_RUN_SIZES = frozenset({"small", "medium", "full"})
_VALIDATOR_ONLINE_WINDOW = timedelta(minutes=5)
_VALIDATOR_STALE_WINDOW = timedelta(minutes=15)
_PUBLIC_ACTIVITY_STATUSES = frozenset(
    {
        "waiting_screening",
        "screening",
        "waiting_validator",
        "evaluating",
        "below_score_floor",
        "under_review",
        "rejected",
        "scored",
        "live",
    }
)


@dataclass(frozen=True)
class _RegistrationSnapshot:
    expires_at: float
    uids_by_hotkey: dict[str, int] | None


@router.get("/weights", response_model=PublicChainWeightsResponse)
async def chain_weights(
    request: Request, response: Response
) -> PublicChainWeightsResponse:
    """Return the latest publicly revealed SN118 validator weight matrix.

    This reads Subtensor storage directly. With commit-reveal enabled the matrix
    is necessarily the last revealed state and may lag encrypted commitments;
    it is evidence of what is public on chain, not a substitute for Yuma's
    stake-weighted emissions calculation.
    """
    response.headers["Cache-Control"] = _CACHE_CONTROL
    chain = getattr(request.app.state, "chain", None)
    config = getattr(request.app.state, "config", None)
    get_weights = getattr(chain, "get_weights", None)
    if chain is None or config is None or not callable(get_weights):
        raise HTTPException(status_code=503, detail="chain weights unavailable")
    try:
        snapshot = await asyncio.wait_for(
            get_weights(config.chain.netuid), timeout=_CHAIN_WEIGHTS_TIMEOUT_SECONDS
        )
    except (ChainError, TimeoutError) as error:
        logger.warning("public chain weights unavailable: %s", error)
        raise HTTPException(
            status_code=503, detail="chain weights unavailable"
        ) from error
    return PublicChainWeightsResponse(
        generated_at=datetime.now(UTC),
        netuid=snapshot.netuid,
        block=snapshot.block,
        block_hash=snapshot.block_hash,
        owner_hotkey=snapshot.owner_hotkey,
        vectors=[
            PublicValidatorWeightVector(
                validator_uid=vector.validator_uid,
                validator_hotkey=vector.validator_hotkey,
                weights=[
                    PublicChainWeight(
                        uid=weight.uid, hotkey=weight.hotkey, value=weight.value
                    )
                    for weight in vector.weights
                ],
            )
            for vector in snapshot.vectors
        ],
    )


def screening_dispute_signing_message(agent_id: UUID, message: str) -> bytes:
    """Return the stable payload a miner signs to authorize one dispute."""

    digest = hashlib.sha256(message.encode()).hexdigest()
    return f"ditto-dispute-v1:{agent_id}:{digest}".encode()


def _public_dispute(dispute: ScreeningDispute) -> PublicScreeningDispute:
    return PublicScreeningDispute(
        status=dispute.status,  # type: ignore[arg-type]
        submitted_at=dispute.created_at,
        resolved_at=dispute.resolved_at,
        resolution=dispute.resolution,  # type: ignore[arg-type]
    )


def _aware(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


def _public_system_metrics(raw: dict | None) -> PublicSystemMetrics | None:
    """Validate stored telemetry again and expose only the fixed public allowlist."""
    if not isinstance(raw, dict):
        return None
    try:
        metrics = SystemMetrics.model_validate(raw)
    except Exception:  # noqa: BLE001 - malformed historical rows stay private
        return None
    return PublicSystemMetrics(
        cpu_percent=metrics.cpu_percent,
        memory_percent=metrics.memory_percent,
        disk_percent=metrics.disk_percent,
        docker_status=metrics.docker.status,
        running_containers=metrics.docker.running_containers,
        unhealthy_containers=metrics.docker.unhealthy_containers,
    )


def _screener_system_metrics(raw: dict | None) -> PublicSystemMetrics | None:
    """Read legacy raw metrics or the private v2 telemetry envelope."""
    if isinstance(raw, dict) and "screening_progress" in raw:
        nested = raw.get("system_metrics")
        return _public_system_metrics(nested if isinstance(nested, dict) else None)
    return _public_system_metrics(raw)


def _stored_screener_progress(raw: dict | None) -> ScreenerProgress | None:
    """Revalidate only the signed progress pair from a v2 storage envelope."""
    if not isinstance(raw, dict):
        return None
    value = raw.get("screening_progress")
    if not isinstance(value, dict):
        return None
    try:
        return ScreenerProgress.model_validate(value)
    except Exception:  # noqa: BLE001 - malformed historical rows stay private
        return None


def _public_benchmark_progress(work: ActiveValidatorWork) -> PublicBenchmarkProgress:
    """Coarsen private signed counts into the fixed public allowlist."""
    progress = work.progress
    if progress is None:
        return PublicBenchmarkProgress(
            agent_id=work.agent.agent_id,
            agent_name=work.agent.name,
            bench_version=work.ticket.bench_version,
            started_at=cast(datetime, _aware(work.ticket.issued_at)),
        )
    percent: int | None = None
    completed_checks: int | None = None
    total_checks: int | None = None
    if progress.completed is not None and progress.total is not None:
        # Nearest 5% is useful without exposing high-resolution timing. Even
        # 114/114 remains 95% while finalization/signing is still in progress.
        percent = min(
            95,
            ((progress.completed * 200 + progress.total * 5) // (progress.total * 10))
            * 5,
        )
        total_checks = progress.total
        completed_checks = (
            progress.total
            if progress.stage in {"finalizing", "submitting_result"}
            else progress.completed
        )
    return PublicBenchmarkProgress(
        agent_id=work.agent.agent_id,
        agent_name=work.agent.name,
        bench_version=work.ticket.bench_version,
        started_at=cast(datetime, _aware(work.ticket.issued_at)),
        stage=progress.stage,
        completed_checks=completed_checks,
        total_checks=total_checks,
        percent=percent,
    )


def _fleet_classification(
    *, state: str, seen_at: datetime, now: datetime, metrics: PublicSystemMetrics | None
) -> tuple[bool, FleetAvailability, FleetHealth]:
    """Return online, availability, and health without treating omission as outage."""
    online = seen_at >= now - _VALIDATOR_ONLINE_WINDOW
    availability: FleetAvailability
    if online and state == "paused":
        availability = "paused"
    elif online:
        availability = "available"
    elif seen_at >= now - _VALIDATOR_STALE_WINDOW:
        availability = "stale"
    else:
        availability = "offline"

    health: FleetHealth
    if state == "error":
        health = "warning"
    elif metrics is None:
        health = "unknown"
    elif (
        metrics.memory_percent >= 90
        or metrics.disk_percent >= 95
        or metrics.docker_status == "degraded"
    ):
        health = "warning"
    elif metrics.docker_status == "unavailable":
        health = "unknown"
    else:
        health = "healthy"
    return online, availability, health


def _safe_models(details: dict) -> PublicRunModels | None:
    """Pull the run's models from the details blob, tolerating a malformed shape."""
    raw = details.get("models")
    if not isinstance(raw, dict):
        return None
    try:
        return PublicRunModels.model_validate(raw)
    except Exception:  # noqa: BLE001 - a bad blob must not break the leaderboard
        return None


_TRANSCRIPT_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def _safe_transcript_sha256(details: dict) -> str | None:
    """Pull the score's declared transcript digest, tolerating malformed blobs."""
    raw = details.get("transcript_sha256")
    if isinstance(raw, str) and _TRANSCRIPT_SHA256_HEX.fullmatch(raw):
        return raw
    return None


def _safe_transform_robustness(details: dict) -> tuple[float | None, int | None]:
    """Pull the reproduce-under-transform audit result, tolerating bad blobs.

    Returns ``(None, None)`` for a run that carried no audit pairs or predates
    the audit, so an absent metric is never published as a failing one.
    """
    raw = details.get("transform_robustness")
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        return None, None
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        return None, None
    pairs = details.get("audit_case_count")
    if not isinstance(pairs, int) or isinstance(pairs, bool) or pairs < 0:
        pairs = None
    return value, pairs


def _safe_categories(details: dict) -> list[PublicCategoryStat] | None:
    """Pull the per-category breakdown, dropping any malformed entries."""
    raw = details.get("per_category")
    if not isinstance(raw, list):
        return None
    out: list[PublicCategoryStat] = []
    for c in raw:
        try:
            out.append(PublicCategoryStat.model_validate(c))
        except Exception:  # noqa: BLE001 - skip a bad category, keep the rest
            continue
    return out or None


def _safe_integrity(details: dict) -> PublicBenchIntegrity | None:
    """Assemble the anti-overfit / integrity telemetry from the details blob.

    The scoring engine nests these under ``paraphrase`` / ``lexical_gap`` sub-dicts
    plus flat ``capped_tool_cases`` / ``seeding_waves``; flatten defensively so a
    partial or malformed shape yields ``None`` fields, never an error."""
    para = details.get("paraphrase")
    para = para if isinstance(para, dict) else {}
    lex = details.get("lexical_gap")
    lex = lex if isinstance(lex, dict) else {}

    def _i(v: object) -> int | None:
        return v if isinstance(v, int) and not isinstance(v, bool) else None

    def _f(v: object) -> float | None:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        return float(v)

    try:
        model = PublicBenchIntegrity(
            paraphrase_applied=_i(para.get("applied")),
            paraphrase_attempted=_i(para.get("attempted")),
            paraphrase_fallback=_i(para.get("fallback")),
            lexical_gap_rewritten=_i(lex.get("rewritten")),
            lexical_gap_questions=_i(lex.get("questions")),
            lexical_gap_mean_before=_f(lex.get("mean_before")),
            lexical_gap_mean_after=_f(lex.get("mean_after")),
            capped_tool_cases=_i(details.get("capped_tool_cases")),
            seeding_waves=_i(details.get("seeding_waves")),
        )
    except Exception:  # noqa: BLE001 - a bad blob must not break the leaderboard
        return None
    if all(v is None for v in model.model_dump().values()):
        return None
    return model


def _safe_case_results(details: dict) -> list[PublicCaseResult] | None:
    """Redact ``details.per_case`` down to the publishable per-case view.

    Whitelists only ``category / kind / score / correct / latency_ms / notes`` —
    the answer-key fields (``expected``, the agent's ``called`` tools, the
    seed-derived ``case_id``, and any other key) are dropped by construction, not
    filtered out, so a new per-case field can never leak by default. ``None`` when
    there is no usable per-case data.
    """
    per_case = details.get("per_case")
    if not isinstance(per_case, list):
        return None
    out: list[PublicCaseResult] = []
    for c in per_case:
        if not isinstance(c, dict):
            continue
        score = c.get("score")
        category = c.get("category")
        if not isinstance(category, str):
            continue
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        kind = c.get("kind")
        latency = c.get("latency_ms")
        correct = c.get("correct")
        notes = c.get("notes")
        clean_notes = (
            [str(n) for n in notes] if isinstance(notes, list) and notes else None
        )
        try:
            out.append(
                PublicCaseResult(
                    category=category,
                    kind=str(kind) if isinstance(kind, str) else "",
                    score=float(score),
                    correct=correct if isinstance(correct, bool) else None,
                    latency_ms=(
                        latency
                        if isinstance(latency, int) and not isinstance(latency, bool)
                        else None
                    ),
                    notes=clean_notes,
                )
            )
        except Exception:  # noqa: BLE001 - skip a bad case, keep the rest
            continue
    return out or None


def _safe_calibration(details: dict) -> tuple[float | None, int | None]:
    """Pull the advisory calibration telemetry (prod hardening P5): the mean
    Brier score over confidence-reporting cases and its sample size. Tolerates
    a malformed blob — anything out of range degrades to ``(None, None)``.
    Never scored; surfacing it costs nothing to harnesses that omit confidence.
    """
    brier = details.get("calibration_brier")
    n = details.get("calibration_n")
    if isinstance(brier, bool) or not isinstance(brier, (int, float)):
        return None, None
    b = float(brier)
    if not 0.0 <= b <= 1.0:
        return None, None
    count = n if isinstance(n, int) and not isinstance(n, bool) and n > 0 else None
    return b, count


def _public_entry(
    rank: int,
    r: LedgerRow,
    agent_name: str,
    agent_version: int | None,
    history: list[float] | None = None,
    *,
    finalized: bool = True,
    score_count: int = SCORING_QUORUM,
    registered: bool | None = None,
    miner_uid: int | None = None,
    fold_stderr: float | None = None,
    active_version: int,
) -> PublicLeaderboardEntry:
    """Map a ledger row to the public entry, exposing only the safe subset of
    ``details`` (never ``per_case``, which carries the answer key)."""
    details = r.details if isinstance(r.details, dict) else {}
    bench_version = details.get("bench_version")
    dataset_sha256 = details.get("dataset_sha256")
    raw_tokens = details.get("tokens")
    tokens = (
        raw_tokens
        if isinstance(raw_tokens, int) and not isinstance(raw_tokens, bool)
        else None
    )
    # A length-1 history is just the current score — not a trend; drop it so the
    # dashboard shows a sparkline only when there's an actual trajectory.
    trend = history if history and len(history) >= 2 else None
    calibration_brier, calibration_n = _safe_calibration(details)
    return PublicLeaderboardEntry(
        rank=rank,
        finalized=finalized,
        score_count=score_count,
        score_quorum=SCORING_QUORUM,
        agent_id=r.agent_id,
        agent_name=agent_name,
        agent_version=agent_version,
        miner_hotkey=r.miner_hotkey,
        miner_uid=miner_uid,
        registered=registered,
        emission_eligible=(
            finalized and r.eligible and registered and bench_version == active_version
            if registered is not None
            else None
        ),
        composite=r.composite,
        # Use the exact uncertainty value sent to validators: a stashed re-score
        # SE when present, otherwise the k=3 quorum SEM. This keeps the displayed
        # band and the KOTH projection aligned with the real fold.
        composite_stderr=fold_stderr,
        calibration_brier=calibration_brier,
        calibration_n=calibration_n,
        tool_mean=r.tool_mean,
        memory_mean=r.memory_mean,
        first_seen=r.first_seen,
        median_ms=r.median_ms,
        n=r.n,
        eligible=r.eligible,
        bench_version=bench_version if isinstance(bench_version, int) else None,
        dataset_sha256=dataset_sha256 if isinstance(dataset_sha256, str) else None,
        models=_safe_models(details),
        per_category=_safe_categories(details),
        integrity=_safe_integrity(details),
        tokens=tokens,
        history=trend,
        case_results=_safe_case_results(details),
    )


def _public_koth_emissions(
    rows: list[LedgerRow],
    *,
    stderrs: dict[UUID, float | None],
    active_version: int,
) -> PublicKothEmissions | None:
    """Project the finalized score pool through the validator's pure fold."""
    candidates: list[LedgerRow] = []
    for row in rows:
        details = row.details if isinstance(row.details, dict) else {}
        if (
            row.eligible
            and row.composite > 0.0
            and details.get("bench_version") == active_version
        ):
            candidates.append(row)
    candidates.sort(key=lambda row: (-row.composite, row.first_seen, row.agent_id))

    fold_entries = []
    for raw_rank, row in enumerate(candidates, start=1):
        details = row.details if isinstance(row.details, dict) else {}
        fold_entries.append(
            KothEntry(
                miner_hotkey=row.miner_hotkey,
                agent_id=row.agent_id,
                composite=row.composite,
                first_seen=row.first_seen,
                raw_rank=raw_rank,
                composite_stderr=stderrs.get(row.agent_id),
                confirmation_composites=(
                    tuple(values)
                    if (values := _confirmation_composites(details)) is not None
                    else None
                ),
                confirmation_seeds=(
                    tuple(seeds)
                    if (seeds := _confirmation_seeds(details)) is not None
                    else None
                ),
            )
        )

    projection = project_koth(fold_entries)
    if projection is None:
        return None
    tail_share = (
        (1.0 - KOTH_CHAMPION_SHARE) / len(projection.tail) if projection.tail else 0.0
    )
    recipients = [
        PublicEmissionRecipient(
            role="champion",
            agent_id=projection.champion.agent_id,
            miner_hotkey=projection.champion.miner_hotkey,
            raw_rank=projection.champion.raw_rank,
            share_of_miner_pool=KOTH_CHAMPION_SHARE,
        )
    ]
    recipients.extend(
        PublicEmissionRecipient(
            role="tail",
            agent_id=entry.agent_id,
            miner_hotkey=entry.miner_hotkey,
            raw_rank=entry.raw_rank,
            share_of_miner_pool=tail_share,
        )
        for entry in projection.tail
    )
    decision = projection.raw_leader_decision
    return PublicKothEmissions(
        margin=KOTH_MARGIN,
        dethrone_z=KOTH_DETHRONE_Z,
        champion_share=KOTH_CHAMPION_SHARE,
        tail_size=KOTH_TAIL_SIZE,
        champion_agent_id=projection.champion.agent_id,
        champion_miner_hotkey=projection.champion.miner_hotkey,
        raw_leader_agent_id=projection.raw_leader.agent_id,
        raw_leader_miner_hotkey=projection.raw_leader.miner_hotkey,
        raw_leader_decision=(
            PublicDethroneDecision(
                challenger_lead=decision.challenger_lead,
                required_lead=decision.required_lead,
                margin_lead=decision.margin_lead,
                statistical_lead=decision.statistical_lead,
                method=decision.method,
                dethrones=decision.dethrones,
            )
            if decision is not None
            else None
        ),
        recipients=recipients,
    )


@router.get("/leaderboard", response_model=PublicLeaderboardResponse)
async def leaderboard(
    request: Request,
    response: Response,
    session: SessionDep,
) -> PublicLeaderboardResponse:
    """Best score per miner, with quorum and current registration eligibility."""
    response.headers["Cache-Control"] = _CACHE_CONTROL
    canonical_version = await active_bench_version(session)
    ledger_rows = await list_eligible_ledger(session, include_fingerprints=False)
    registered_uids = await _current_registered_uids(request)
    quorum = await quorum_composites(session, [row.agent_id for row in ledger_rows])
    fold_stderrs = {
        row.agent_id: _ledger_stderr(
            row.details if isinstance(row.details, dict) else None,
            quorum.get(row.agent_id, []),
        )
        for row in ledger_rows
    }
    score_counts = await get_score_counts(
        session, [row.agent_id for row in ledger_rows]
    )
    finalized_rows = [
        row
        for row in ledger_rows
        if score_counts.get(row.agent_id, 0) >= SCORING_QUORUM
    ]
    finalized_miners = {row.miner_hotkey for row in finalized_rows}
    provisional_candidates = [
        (row, score_counts.get(row.agent_id, 0))
        for row in ledger_rows
        if score_counts.get(row.agent_id, 0) < SCORING_QUORUM
    ] + list(await list_provisional_ledger(session))
    provisional_candidates.sort(
        key=lambda candidate: (
            not candidate[0].eligible,
            -candidate[0].composite,
            candidate[0].first_seen,
            str(candidate[0].agent_id),
        )
    )
    provisional_by_miner: dict[str, tuple[LedgerRow, int]] = {}
    for candidate in provisional_candidates:
        if candidate[0].miner_hotkey not in finalized_miners:
            provisional_by_miner.setdefault(candidate[0].miner_hotkey, candidate)
    provisional_rows = list(provisional_by_miner.values())
    rows = finalized_rows + [row for row, _count in provisional_rows]
    agent_metadata = {
        agent_id: (name, version)
        for agent_id, name, version in (
            await session.execute(
                select(Agent.agent_id, Agent.name, Agent.version).where(
                    Agent.agent_id.in_([row.agent_id for row in rows])
                )
            )
        )
        .tuples()
        .all()
    }
    histories = await list_miner_composite_history(
        session, [r.miner_hotkey for r in rows]
    )
    entries = []
    for i, row in enumerate(finalized_rows, start=1):
        entries.append(
            _public_entry(
                i,
                row,
                *agent_metadata[row.agent_id],
                histories.get(row.miner_hotkey),
                finalized=True,
                score_count=score_counts.get(row.agent_id, SCORING_QUORUM),
                registered=(
                    row.miner_hotkey in registered_uids
                    if registered_uids is not None
                    else None
                ),
                miner_uid=(
                    registered_uids.get(row.miner_hotkey)
                    if registered_uids is not None
                    else None
                ),
                fold_stderr=fold_stderrs.get(row.agent_id),
                active_version=canonical_version,
            )
        )
    for row, count in provisional_rows:
        entries.append(
            _public_entry(
                len(entries) + 1,
                row,
                *agent_metadata[row.agent_id],
                histories.get(row.miner_hotkey),
                finalized=False,
                score_count=count,
                registered=(
                    row.miner_hotkey in registered_uids
                    if registered_uids is not None
                    else None
                ),
                miner_uid=(
                    registered_uids.get(row.miner_hotkey)
                    if registered_uids is not None
                    else None
                ),
                fold_stderr=fold_stderrs.get(row.agent_id),
                active_version=canonical_version,
            )
        )
    return PublicLeaderboardResponse(
        generated_at=datetime.now(UTC),
        count=len(entries),
        current_bench_version=canonical_version,
        entries=entries,
        emissions=_public_koth_emissions(
            finalized_rows,
            stderrs=fold_stderrs,
            active_version=canonical_version,
        ),
    )


async def _current_registered_uids(request: Request) -> dict[str, int] | None:
    """Current subnet hotkeys and UIDs, or ``None`` when chain data is unavailable.

    Registration decorates the durable score ledger; it never deletes or changes
    a submission. Public reads therefore degrade to an explicit unknown state
    instead of failing or pretending a stale registration result is current.
    """
    chain = getattr(request.app.state, "chain", None)
    config = getattr(request.app.state, "config", None)
    if chain is None or config is None:
        return None
    now = time.monotonic()
    cached = getattr(request.app.state, "public_registration_snapshot", None)
    if isinstance(cached, _RegistrationSnapshot) and cached.expires_at > now:
        return cached.uids_by_hotkey
    try:
        async with asyncio.timeout(_REGISTRATION_LOOKUP_TIMEOUT_SECONDS):
            neurons = await chain.get_recent_neurons(config.chain.netuid)
    except (ChainError, TimeoutError) as e:
        logger.warning("public leaderboard registration read failed: %s", e)
        request.app.state.public_registration_snapshot = _RegistrationSnapshot(
            expires_at=now + _REGISTRATION_FAILURE_CACHE_TTL_SECONDS,
            uids_by_hotkey=None,
        )
        return None
    uids_by_hotkey = {neuron.hotkey: int(neuron.uid) for neuron in neurons}
    request.app.state.public_registration_snapshot = _RegistrationSnapshot(
        expires_at=now + _REGISTRATION_CACHE_TTL_SECONDS,
        uids_by_hotkey=uids_by_hotkey,
    )
    return uids_by_hotkey


@router.get("/health", response_model=PublicHealthResponse)
async def health(
    response: Response,
    session: SessionDep,
) -> PublicHealthResponse:
    """Aggregate subnet-health rollup (submissions + reported scores).

    Aggregate-only, like the leaderboard: miner/agent counts, last-scored time,
    24h scoring throughput, and average latency. Failure/latency-of-weights
    telemetry lives in wandb — the platform only sees successful scores.
    """
    response.headers["Cache-Control"] = _CACHE_CONTROL
    now = datetime.now(UTC)
    roll = await get_public_health(session, now=now)
    return PublicHealthResponse(
        generated_at=now,
        miners=roll.miners,
        scored_miners=roll.scored_miners,
        scored_agents=roll.scored_agents,
        last_scored_at=roll.last_scored_at,
        total_scores=roll.total_scores,
        scores_24h=roll.scores_24h,
        avg_latency_ms=roll.avg_latency_ms,
    )


def _validator_heartbeats_response(
    *,
    rows: list[Any],
    assignments: list[ActiveValidatorAssignment],
    active_work: list[ActiveValidatorWork],
    now: datetime,
) -> PublicValidatorHeartbeatsResponse:
    """Reconcile platform leases and signed heartbeat claims without conflating them."""
    assignment_by_hotkey = {
        assignment.ticket.validator_hotkey: assignment for assignment in assignments
    }
    active_by_hotkey = {work.heartbeat.validator_hotkey: work for work in active_work}
    entries = []
    for row in rows:
        seen_at = cast(datetime, _aware(row.seen_at))
        metrics = _public_system_metrics(row.system_metrics)
        online, availability, health = _fleet_classification(
            state=row.state, seen_at=seen_at, now=now, metrics=metrics
        )
        assignment = assignment_by_hotkey.get(row.validator_hotkey)
        synchronized_work = active_by_hotkey.get(row.validator_hotkey)
        capabilities = None
        stack = None
        if row.protocol_version >= 7:
            try:
                capabilities = ValidatorCapabilities.model_validate(row.capabilities)
                stack = ValidatorStackIdentity.model_validate(row.stack)
            except ValidationError:
                # Stored telemetry is not trusted merely because it is JSON.
                # Malformed v7 data is omitted publicly and rejected for routing.
                pass
        stack_health = None
        if row.protocol_version >= 9:
            # Same posture as v7 identity: publish only what re-validates
            # against the closed schema, never raw stored JSON.
            with contextlib.suppress(ValidationError):
                stack_health = ValidatorStackHealth.model_validate(row.stack_health)
        assignment_state: ValidatorAssignmentState
        if assignment is None:
            assignment_state = (
                "heartbeat_mismatch"
                if row.active_agent_id is not None
                else "unassigned"
            )
        elif seen_at < now - _VALIDATOR_ONLINE_WINDOW:
            assignment_state = "heartbeat_stale"
        elif synchronized_work is not None:
            assignment_state = "synchronized"
        else:
            assignment_state = "heartbeat_mismatch"
        entries.append(
            PublicValidatorHeartbeat(
                validator_hotkey=row.validator_hotkey,
                software_version=row.software_version,
                protocol_version=row.protocol_version,
                state=cast(ValidatorRuntimeState, row.state),
                assigned_agent_id=(
                    assignment.agent.agent_id if assignment is not None else None
                ),
                assigned_agent_name=(
                    assignment.agent.name if assignment is not None else None
                ),
                reported_agent_id=row.active_agent_id,
                assignment_state=assignment_state,
                active_agent_id=(
                    synchronized_work.agent.agent_id
                    if synchronized_work is not None
                    else None
                ),
                active_benchmark=(
                    _public_benchmark_progress(synchronized_work)
                    if synchronized_work is not None
                    else None
                ),
                first_seen_at=_aware(row.first_seen_at),
                reported_at=cast(datetime, _aware(row.reported_at)),
                seen_at=seen_at,
                online=online,
                availability=availability,
                health=health,
                system_metrics=metrics,
                capabilities=capabilities,
                stack=stack,
                stack_health=stack_health,
            )
        )
    return PublicValidatorHeartbeatsResponse(
        generated_at=now,
        online_window_seconds=int(_VALIDATOR_ONLINE_WINDOW.total_seconds()),
        stale_window_seconds=int(_VALIDATOR_STALE_WINDOW.total_seconds()),
        reported_count=len(entries),
        online_count=sum(entry.online for entry in entries),
        validators=entries,
    )


@router.get("/validators", response_model=PublicValidatorHeartbeatsResponse)
async def validators(
    response: Response,
    session: SessionDep,
) -> PublicValidatorHeartbeatsResponse:
    """Signed reports reconciled with the platform's current assignment truth."""
    response.headers["Cache-Control"] = _CACHE_CONTROL
    now = datetime.now(UTC)
    return _validator_heartbeats_response(
        rows=await list_validator_heartbeats(session),
        assignments=await list_active_validator_assignments(session, now=now),
        active_work=await list_active_validator_work(
            session, now=now, cutoff=now - _VALIDATOR_ONLINE_WINDOW
        ),
        now=now,
    )


@router.get("/validator-names", response_model=PublicValidatorNamesResponse)
async def validator_names(
    request: Request,
    response: Response,
    session: SessionDep,
) -> PublicValidatorNamesResponse:
    """Cached optional Taostats labels; this route never performs external I/O."""
    response.headers["Cache-Control"] = _CACHE_CONTROL
    rows = await list_validator_heartbeats(session)
    reporter_hotkeys = {row.validator_hotkey for row in rows}
    snapshot = request.app.state.validator_names.snapshot(sorted(reporter_hotkeys))
    return PublicValidatorNamesResponse(
        generated_at=datetime.now(UTC),
        status=snapshot.status,
        refreshed_at=snapshot.refreshed_at,
        validators=[
            PublicValidatorName(
                validator_hotkey=hotkey,
                display_name=snapshot.names.get(hotkey),
                stake_weight=snapshot.stake_weights.get(hotkey),
            )
            for hotkey in sorted(snapshot.names.keys() | snapshot.stake_weights.keys())
            if hotkey in reporter_hotkeys
        ],
    )


@router.get("/screeners", response_model=PublicScreenerHeartbeatsResponse)
async def screeners(
    response: Response,
    session: SessionDep,
) -> PublicScreenerHeartbeatsResponse:
    """Authenticated screener fleet reports with a strict public allowlist."""
    response.headers["Cache-Control"] = _CACHE_CONTROL
    now = datetime.now(UTC)
    rows = await list_screener_heartbeats(session)
    active_ids = [
        row.active_agent_id
        for row in rows
        if row.state == "screening" and row.active_agent_id is not None
    ]
    attempts = await get_running_screening_attempts(session, agent_ids=active_ids)
    agents = {
        agent.agent_id: agent
        for agent in await session.scalars(
            select(Agent).where(Agent.agent_id.in_(active_ids))
        )
    }
    entries = []
    for row in rows:
        seen_at = cast(datetime, _aware(row.seen_at))
        metrics = _screener_system_metrics(row.system_metrics)
        online, availability, health = _fleet_classification(
            state=row.state, seen_at=seen_at, now=now, metrics=metrics
        )
        active_agent_id = row.active_agent_id
        active_agent = (
            agents.get(active_agent_id) if active_agent_id is not None else None
        )
        active_attempt = (
            attempts.get(active_agent_id) if active_agent_id is not None else None
        )
        active_work = bool(
            online
            and row.state == "screening"
            and active_agent is not None
            and active_agent.status == AgentStatus.SCREENING
            and active_attempt is not None
            and active_attempt.screener_hotkey == row.screener_hotkey
            and cast(datetime, _aware(active_attempt.deadline)) >= now
        )
        progress = (
            _stored_screener_progress(row.system_metrics) if active_work else None
        )
        public_progress = None
        if progress is not None and active_attempt is not None:
            progress_started = datetime.fromtimestamp(progress.started_at, tz=UTC)
            attempt_started = cast(datetime, _aware(active_attempt.started_at))
            if (
                attempt_started - _VALIDATOR_ONLINE_WINDOW
                <= progress_started
                <= seen_at + _VALIDATOR_ONLINE_WINDOW
            ):
                public_progress = PublicScreenerProgress(
                    stage=progress.stage,
                    started_at=progress_started,
                )
        entries.append(
            PublicScreenerHeartbeat(
                instance_id=row.instance_id,
                screener_hotkey=row.screener_hotkey,
                software_version=row.software_version,
                protocol_version=row.protocol_version,
                policy_version=row.policy_version,
                state=cast(ScreenerRuntimeState, row.state),
                active_agent_id=active_agent_id if active_work else None,
                active_agent_name=(
                    active_agent.name
                    if active_work and active_agent is not None
                    else None
                ),
                screening_progress=public_progress,
                first_seen_at=_aware(row.first_seen_at),
                reported_at=cast(datetime, _aware(row.reported_at)),
                seen_at=seen_at,
                online=online,
                availability=availability,
                health=health,
                system_metrics=metrics,
            )
        )
    return PublicScreenerHeartbeatsResponse(
        generated_at=now,
        online_window_seconds=int(_VALIDATOR_ONLINE_WINDOW.total_seconds()),
        stale_window_seconds=int(_VALIDATOR_STALE_WINDOW.total_seconds()),
        reported_count=len(entries),
        online_count=sum(entry.online for entry in entries),
        screeners=entries,
    )


def _median_composite(row: SubmissionRow) -> float | None:
    """Median of the reported composites — the canonical score, or None if unscored."""
    if not row.scores:
        return None
    return statistics.median(s.composite for s in row.scores)


def _ticket_deadline(score: Score) -> datetime | None:
    """Read the signed lease identity from score details; null means legacy."""
    if not isinstance(score.details, dict):
        return None
    value = score.details.get("ticket_deadline")
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _score_bench_version(score: Score) -> int | None:
    """Read the persisted benchmark epoch without guessing for legacy rows."""
    if not isinstance(score.details, dict):
        return None
    value = score.details.get("bench_version")
    return value if isinstance(value, int) and value > 0 else None


def _datagen_version(bench_version: int | None) -> str | None:
    """Resolve a benchmark epoch only when its exact generator pin is known."""
    if bench_version is None:
        return None
    return _DATAGEN_VERSION_BY_BENCH_VERSION.get(bench_version)


def _dataset_command(
    *, seed: int, run_size: str | None, bench_version: int | None, sha_only: bool
) -> str | None:
    """Return the documented deterministic generator command for a score."""
    datagen_version = _datagen_version(bench_version)
    if run_size not in _DATAGEN_RUN_SIZES or datagen_version is None:
        return None
    command = (
        "go run github.com/ditto-assistant/dittobench-datagen/cmd/generate@"
        f"{datagen_version} -seed {seed} -run-size {run_size}"
    )
    return f"{command} -sha" if sha_only else f"{command} -out dataset.json"


def _submission_scores(row: SubmissionRow) -> PublicSubmissionScores:
    """Map a submission row to the full public k=3 record."""
    return PublicSubmissionScores(
        agent_id=row.agent_id,
        miner_hotkey=row.miner_hotkey,
        status=row.status.value,
        quorum=SCORING_QUORUM,
        score_count=len(row.scores),
        median_composite=_median_composite(row),
        dataset_seed=row.dataset_seed,
        dataset_sha256=row.dataset_sha256,
        dataset_run_size=row.dataset_run_size,
        dataset_seed_block=row.dataset_seed_block,
        dataset_seed_block_hash=row.dataset_seed_block_hash,
        scores=[_public_validator_score(s) for s in row.scores],
        generated_at=datetime.now(UTC),
    )


def _public_validator_score(s) -> PublicValidatorScore:
    """Map one stored score row to its published, redacted form."""
    details = s.details if isinstance(s.details, dict) else {}
    robustness, audit_pairs = _safe_transform_robustness(details)
    return PublicValidatorScore(
        validator_hotkey=s.validator_hotkey,
        composite=s.composite,
        tool_mean=s.tool_mean,
        memory_mean=s.memory_mean,
        median_ms=s.median_ms,
        n=s.n,
        seed=s.seed,
        run_id=s.run_id,
        ticket_deadline=_ticket_deadline(s),
        signature=s.signature,
        generated_at=s.generated_at,
        case_results=_safe_case_results(details),
        transcript_sha256=_safe_transcript_sha256(details),
        transform_robustness=robustness,
        audit_case_count=audit_pairs,
    )


def _submission_summary(row: SubmissionRow) -> PublicSubmissionSummary:
    """Map a submission row to the compact index entry."""
    return PublicSubmissionSummary(
        agent_id=row.agent_id,
        miner_hotkey=row.miner_hotkey,
        status=row.status.value,
        score_count=len(row.scores),
        median_composite=_median_composite(row),
        dataset_seed=row.dataset_seed,
        dataset_sha256=row.dataset_sha256,
        last_scored_at=row.last_scored_at,
    )


def _public_activity_status(
    status: AgentStatus,
    *,
    screening_policy_version: int,
    has_active_attempt: bool,
    has_active_validation: bool,
    has_live_assignment: bool = False,
    score_count: int = 0,
    highest_composite: float | None = None,
    score_continuation_floor: float | None = None,
) -> str:
    """Collapse internal moderation detail into stable public lifecycle labels."""
    needs_rescreen = (
        status
        in (
            AgentStatus.EVALUATING,
            AgentStatus.REJECTED,
        )
        and screening_policy_version < SCREENING_POLICY_VERSION
    )
    if has_active_attempt or status == AgentStatus.SCREENING:
        return AgentStatus.SCREENING.value
    if status in (AgentStatus.UPLOADED, AgentStatus.SCREENING_FAILED) or needs_rescreen:
        return "waiting_screening"
    if status in (AgentStatus.SCREENING_PASSED, AgentStatus.EVALUATING):
        if (
            status == AgentStatus.EVALUATING
            and not has_live_assignment
            and score_count == SCORING_QUORUM - 1
            and highest_composite is not None
            and score_continuation_floor is not None
            and highest_composite < score_continuation_floor
        ):
            return "below_score_floor"
        return "evaluating" if has_active_validation else "waiting_validator"
    if status in (AgentStatus.ATH_PENDING_REVIEW, AgentStatus.QUARANTINED):
        return "under_review"
    if status == AgentStatus.BANNED:
        return "rejected"
    return status.value


def _public_activity_response(
    *,
    rows: list[Any],
    active_work: list[ActiveValidatorWork],
    now: datetime,
    page: int,
    limit: int,
    requested_statuses: set[str],
    query: str | None,
    score_continuation_floor: float | None,
    active_assignment_agent_ids: set[UUID],
    duplicate_metadata: dict[UUID, tuple[str, int | None]] | None = None,
    ath_review_opened_at: dict[UUID, datetime] | None = None,
    ath_review_composite: dict[UUID, float] | None = None,
    ath_only: bool = False,
) -> PublicActivityResponse:
    """Project activity from the same validated work set used by fleet health."""
    active_by_agent: dict[UUID, list[PublicBenchmarkProgress]] = {}
    for work in active_work:
        active_by_agent.setdefault(work.agent.agent_id, []).append(
            _public_benchmark_progress(work)
        )
    active_agent_ids = set(active_by_agent)

    def public_status(row: Any) -> str:
        return _public_activity_status(
            row.agent.status,
            screening_policy_version=row.agent.screening_policy_version,
            has_active_attempt=row.screening_attempt is not None,
            has_active_validation=row.agent.agent_id in active_agent_ids,
            has_live_assignment=row.agent.agent_id in active_assignment_agent_ids,
            score_count=row.score_count,
            highest_composite=row.highest_composite,
            score_continuation_floor=score_continuation_floor,
        )

    projected = [(row, public_status(row)) for row in rows]
    if ath_only:
        projected = [
            (row, row_status)
            for row, row_status in projected
            if row.agent.status == AgentStatus.ATH_PENDING_REVIEW
        ]
    # Mirror the validator ticket queue's global ordering. The ticket query adds
    # validator-specific retry and eligibility checks that can still skip a row.
    waiting_candidates = [
        row for row, row_status in projected if row_status == "waiting_validator"
    ]
    provisional_candidates = sorted(
        (
            row
            for row in waiting_candidates
            if row.score_count == SCORING_QUORUM - 1
            and row.provisional_composite is not None
        ),
        key=lambda row: (
            -(row.provisional_composite or 0.0),
            row.agent.created_at,
            str(row.agent.agent_id),
        ),
    )
    best_provisional_by_miner: dict[str, Any] = {}
    for row in provisional_candidates:
        best_provisional_by_miner.setdefault(row.agent.miner_hotkey, row)
    provisional_contender_ids = {
        row.agent.agent_id
        for row in list(best_provisional_by_miner.values())[
            :PROVISIONAL_CONTENDER_LANE_SIZE
        ]
    }
    waiting_rows = sorted(
        waiting_candidates,
        key=lambda row: (
            0 if row.agent.agent_id in provisional_contender_ids else 1,
            row.score_count,
            -(row.provisional_composite or 0.0),
            row.agent.created_at,
            str(row.agent.agent_id),
        ),
    )
    validator_queue_ranks = {
        row.agent.agent_id: rank for rank, row in enumerate(waiting_rows, start=1)
    }
    normalized_query = query.strip().casefold() if query else ""
    if normalized_query:
        projected = [
            (row, row_status)
            for row, row_status in projected
            if normalized_query
            in " ".join(
                (
                    row.agent.name,
                    str(row.agent.agent_id),
                    row.agent.miner_hotkey,
                    row_status,
                )
            ).casefold()
        ]

    status_counts: dict[str, int] = {}
    for _, row_status in projected:
        status_counts[row_status] = status_counts.get(row_status, 0) + 1
    if requested_statuses:
        projected = [
            (row, row_status)
            for row, row_status in projected
            if row_status in requested_statuses
        ]

    total = len(projected)
    page_rows = projected[(page - 1) * limit : page * limit]
    return PublicActivityResponse(
        generated_at=now,
        count=len(page_rows),
        total=total,
        status_counts=status_counts,
        page=page,
        page_size=limit,
        total_pages=max(1, math.ceil(total / limit)),
        entries=[
            PublicActivityEntry(
                agent_id=row.agent.agent_id,
                miner_hotkey=row.agent.miner_hotkey,
                name=row.agent.name,
                version=row.agent.version,
                status=row_status,
                submitted_at=row.agent.created_at,
                last_scored_at=_aware(row.last_scored_at),
                screening_reason=(
                    None
                    if row_status in ("waiting_screening", "screening")
                    else row.agent.screening_reason
                ),
                duplicate_of=row.agent.duplicate_of,
                duplicate_name=(duplicate_metadata or {}).get(
                    row.agent.duplicate_of, (None, None)
                )[0],
                duplicate_version=(duplicate_metadata or {}).get(
                    row.agent.duplicate_of, (None, None)
                )[1],
                review_reason=row.agent.review_reason,
                review_opened_at=(ath_review_opened_at or {}).get(row.agent.agent_id),
                preserved_composite=(ath_review_composite or {}).get(
                    row.agent.agent_id
                ),
                score_count=row.score_count,
                provisional_composite=row.provisional_composite,
                validator_queue_rank=validator_queue_ranks.get(row.agent.agent_id),
                quorum=SCORING_QUORUM,
                screening_policy_version=row.agent.screening_policy_version,
                required_screening_policy_version=SCREENING_POLICY_VERSION,
                screening_attempt_id=(
                    row.screening_attempt.attempt_id
                    if row.screening_attempt is not None
                    else None
                ),
                screening_started_at=(
                    row.screening_attempt.started_at
                    if row.screening_attempt is not None
                    else None
                ),
                screening_deadline=(
                    row.screening_attempt.deadline
                    if row.screening_attempt is not None
                    else None
                ),
                active_benchmarks=active_by_agent.get(row.agent.agent_id, []),
            )
            for row, row_status in page_rows
        ],
    )


async def _duplicate_submission_metadata(
    session: AsyncSession, rows: list[Any]
) -> dict[UUID, tuple[str, int | None]]:
    """Resolve safe display metadata for copy-review comparison targets."""
    duplicate_ids = {
        row.agent.duplicate_of for row in rows if row.agent.duplicate_of is not None
    }
    if not duplicate_ids:
        return {}
    return {
        agent_id: (name, version)
        for agent_id, name, version in (
            await session.execute(
                select(Agent.agent_id, Agent.name, Agent.version).where(
                    Agent.agent_id.in_(duplicate_ids)
                )
            )
        )
        .tuples()
        .all()
    }


async def _ath_review_public_snapshot(
    session: AsyncSession, rows: list[Any]
) -> tuple[dict[UUID, datetime], dict[UUID, float]]:
    """Load public-safe hold times and canonical composites for active reviews."""
    agent_ids = {
        row.agent.agent_id
        for row in rows
        if row.agent.status == AgentStatus.ATH_PENDING_REVIEW
    }
    if not agent_ids:
        return {}, {}
    opened_at = dict(
        (
            await session.execute(
                select(AthReview.agent_id, AthReview.opened_at).where(
                    AthReview.agent_id.in_(agent_ids), AthReview.status == "pending"
                )
            )
        )
        .tuples()
        .all()
    )
    composites: dict[UUID, list[float]] = {}
    for agent_id, composite in (
        await session.execute(
            select(Score.agent_id, Score.composite).where(Score.agent_id.in_(agent_ids))
        )
    ).tuples():
        composites.setdefault(agent_id, []).append(float(composite))
    # Match the canonical median used when the score quorum finalized.
    medians = {
        agent_id: float(statistics.median(values))
        for agent_id, values in composites.items()
    }
    return opened_at, medians


@router.get("/activity", response_model=PublicActivityResponse)
async def activity(
    response: Response,
    session: SessionDep,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
    status: Annotated[list[str] | None, Query()] = None,
    review: Literal["ath"] | None = Query(default=None),
    q: str | None = Query(default=None, min_length=1, max_length=200),
) -> PublicActivityResponse:
    """Recent submissions and their safe public pipeline stage, newest first.

    This exposes the evidence a miner needs to understand a failure or review:
    a safe screening category plus the duplicate reference and anti-copy signal
    summary. Artifact locations, hashes, payments, and raw build logs remain
    private.
    """
    response.headers["Cache-Control"] = "public, max-age=10"
    requested_statuses = set(status or [])
    unknown_statuses = requested_statuses - _PUBLIC_ACTIVITY_STATUSES
    if unknown_statuses:
        raise HTTPException(
            status_code=422,
            detail="unknown public activity status: "
            + ", ".join(sorted(unknown_statuses)),
        )

    now = datetime.now(UTC)
    rows, _ = await list_public_activity(session)
    ath_opened_at: dict[UUID, datetime] = {}
    ath_composite: dict[UUID, float] = {}
    if review == "ath":
        ath_opened_at, ath_composite = await _ath_review_public_snapshot(session, rows)
    assignments = await list_active_validator_assignments(session, now=now)
    return _public_activity_response(
        rows=rows,
        active_work=await list_active_validator_work(
            session, now=now, cutoff=now - _VALIDATOR_ONLINE_WINDOW
        ),
        now=now,
        page=page,
        limit=limit,
        requested_statuses=requested_statuses,
        query=q,
        score_continuation_floor=await get_score_continuation_floor(session),
        active_assignment_agent_ids={
            assignment.agent.agent_id for assignment in assignments
        },
        duplicate_metadata=await _duplicate_submission_metadata(session, rows),
        ath_review_opened_at=ath_opened_at,
        ath_review_composite=ath_composite,
        ath_only=review == "ath",
    )


@router.get("/operations", response_model=PublicOperationsResponse)
async def operations(
    response: Response,
    session: SessionDep,
) -> PublicOperationsResponse:
    """Atomic dashboard snapshot for submission pipeline and validator fleet health."""
    response.headers["Cache-Control"] = "public, max-age=10"
    now = datetime.now(UTC)
    activity_rows, _ = await list_public_activity(session)
    heartbeat_rows = await list_validator_heartbeats(session)
    assignments = await list_active_validator_assignments(session, now=now)
    active_work = await list_active_validator_work(
        session, now=now, cutoff=now - _VALIDATOR_ONLINE_WINDOW
    )
    activity_snapshot = _public_activity_response(
        rows=activity_rows,
        active_work=active_work,
        now=now,
        page=1,
        limit=max(1, len(activity_rows)),
        requested_statuses=set(),
        query=None,
        score_continuation_floor=await get_score_continuation_floor(session),
        active_assignment_agent_ids={
            assignment.agent.agent_id for assignment in assignments
        },
        duplicate_metadata=await _duplicate_submission_metadata(session, activity_rows),
    )
    validator_snapshot = _validator_heartbeats_response(
        rows=heartbeat_rows,
        assignments=assignments,
        active_work=active_work,
        now=now,
    )
    benchmark_rollout = await rollout_state(session, now=now)
    return PublicOperationsResponse(
        generated_at=now,
        active_bench_version=cast(int, benchmark_rollout["active_version"]),
        desired_bench_version=cast(int, benchmark_rollout["desired_version"]),
        benchmark_rollout_status=cast(
            Literal["inactive", "collecting", "blocked_ineligible", "activated"],
            benchmark_rollout["status"],
        ),
        activity=activity_snapshot,
        validators=validator_snapshot,
    )


@router.post(
    "/agent/{agent_id}/dispute",
    response_model=CreateScreeningDisputeResponse,
    status_code=201,
)
async def create_screening_dispute(
    response: Response,
    session: SessionDep,
    agent_id: UUID,
    payload: CreateScreeningDisputeRequest,
) -> CreateScreeningDisputeResponse:
    """Record the submitting hotkey's single appeal of a quarantine rejection."""

    response.headers["Cache-Control"] = "no-store"
    dispute: ScreeningDispute | None = None
    try:
        async with session.begin():
            agent = await session.scalar(
                select(Agent).where(Agent.agent_id == agent_id).with_for_update()
            )
            if agent is None:
                raise HTTPException(status_code=404, detail="submission not found")
            if not _verify_signature(
                agent.miner_hotkey,
                screening_dispute_signing_message(agent_id, payload.message),
                payload.signature,
            ):
                raise HTTPException(
                    status_code=401,
                    detail="signature did not verify against the submitting hotkey",
                )
            if await session.scalar(
                select(ScreeningDispute).where(ScreeningDispute.agent_id == agent_id)
            ):
                raise HTTPException(
                    status_code=409,
                    detail="this submission has already used its one dispute",
                )
            quarantine = await session.scalar(
                select(ScreeningQuarantine)
                .where(
                    ScreeningQuarantine.agent_id == agent_id,
                    ScreeningQuarantine.status == "resolved",
                    ScreeningQuarantine.resolution == "reject",
                )
                .order_by(ScreeningQuarantine.resolved_at.desc())
                .with_for_update()
            )
            if agent.status != AgentStatus.REJECTED or quarantine is None:
                raise HTTPException(
                    status_code=409,
                    detail="only a rejected quarantine decision can be disputed",
                )
            dispute = ScreeningDispute(
                dispute_id=uuid4(),
                agent_id=agent.agent_id,
                quarantine_id=quarantine.quarantine_id,
                miner_hotkey=agent.miner_hotkey,
                message=payload.message,
                status="pending",
                created_at=datetime.now(UTC),
            )
            session.add(dispute)
    except IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail="this submission has already used its one dispute",
        ) from exc

    assert dispute is not None
    return CreateScreeningDisputeResponse(dispute=_public_dispute(dispute))


@router.get("/agent/{agent_id}/pipeline", response_model=PublicSubmissionPipeline)
async def agent_pipeline(
    response: Response,
    session: SessionDep,
    agent_id: UUID,
) -> PublicSubmissionPipeline:
    """Screening history, validator progress, and accepted scores for a submission.

    Accepted scores are visible before quorum with the seed and version-pinned
    dataset command needed to reproduce them. The canonical aggregate remains
    null until the independent-score quorum is reached.
    """
    response.headers["Cache-Control"] = "public, max-age=10"
    agent = await session.get(Agent, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="submission not found")

    attempts = await list_screening_attempts(session, agent_id=agent_id)
    resolved_quarantines_by_attempt: dict[
        UUID,
        tuple[Literal["release", "rescreen", "reject"] | None, datetime | None],
    ] = {
        quarantine.attempt_id: (
            cast(Literal["release", "rescreen", "reject"], quarantine.resolution),
            quarantine.resolved_at,
        )
        for quarantine in await session.scalars(
            select(ScreeningQuarantine).where(ScreeningQuarantine.agent_id == agent_id)
        )
        if quarantine.status == "resolved"
        and quarantine.resolution in {"release", "rescreen", "reject"}
    }
    tickets = list(
        await session.scalars(
            select(ValidatorTicket)
            .where(ValidatorTicket.agent_id == agent_id)
            .order_by(
                ValidatorTicket.issued_at.desc(),
                ValidatorTicket.validator_hotkey,
            )
        )
    )
    now = datetime.now(UTC)
    active_work = [
        work
        for work in await list_active_validator_work(
            session, now=now, cutoff=now - _VALIDATOR_ONLINE_WINDOW
        )
        if work.agent.agent_id == agent_id
    ]
    active_by_hotkey = {work.heartbeat.validator_hotkey: work for work in active_work}
    accepted_scores = list(
        await session.scalars(
            select(Score)
            .where(Score.agent_id == agent_id)
            .order_by(Score.created_at, Score.validator_hotkey)
        )
    )
    canonical_version = await active_bench_version(session)
    canonical_scores = [
        score for score in accepted_scores if score.bench_version == canonical_version
    ]
    running_attempt = next(
        (attempt for attempt in attempts if attempt.status == "running"), None
    )
    score_continuation_floor = await get_score_continuation_floor(session)
    dispute = await session.scalar(
        select(ScreeningDispute).where(ScreeningDispute.agent_id == agent_id)
    )
    return PublicSubmissionPipeline(
        generated_at=now,
        agent_id=agent_id,
        status=_public_activity_status(
            agent.status,
            screening_policy_version=agent.screening_policy_version,
            has_active_attempt=running_attempt is not None,
            has_active_validation=bool(active_work),
            has_live_assignment=any(
                ticket.status == TicketStatus.ISSUED
                and cast(datetime, _aware(ticket.deadline)) > now
                for ticket in tickets
            ),
            score_count=len(canonical_scores),
            highest_composite=(
                max(score.composite for score in canonical_scores)
                if canonical_scores
                else None
            ),
            score_continuation_floor=score_continuation_floor,
        ),
        active_bench_version=canonical_version,
        score_count=len(canonical_scores),
        quorum=SCORING_QUORUM,
        score_floor=score_continuation_floor or 0.0,
        provisional_scores=[
            PublicProvisionalScore(
                composite=score.composite,
                seed=str(score.seed),
                run_size=agent.dataset_run_size,
                bench_version=_score_bench_version(score),
                datagen_version=_datagen_version(_score_bench_version(score)),
                seed_source=(
                    # No pinned dataset (generation disabled when this agent was
                    # screened): the platform never derived a seed, so the one on
                    # the score is the validator's own benchmark seed.
                    "validator_local"
                    if agent.dataset_seed is None
                    else "on_chain"
                    if agent.dataset_seed_block is not None
                    and agent.dataset_seed_block_hash is not None
                    else "random_fallback"
                ),
                dataset_sha256=agent.dataset_sha256,
                accepted_at=score.created_at,
                reproduction_command=_dataset_command(
                    seed=score.seed,
                    run_size=agent.dataset_run_size,
                    bench_version=_score_bench_version(score),
                    sha_only=False,
                ),
                verification_command=_dataset_command(
                    seed=score.seed,
                    run_size=agent.dataset_run_size,
                    bench_version=_score_bench_version(score),
                    sha_only=True,
                ),
                case_results=_safe_case_results(
                    score.details if isinstance(score.details, dict) else {}
                ),
                transcript_sha256=_safe_transcript_sha256(
                    score.details if isinstance(score.details, dict) else {}
                ),
            )
            for score in accepted_scores
        ],
        final_composite=(
            statistics.median(score.composite for score in canonical_scores)
            if len(canonical_scores) >= SCORING_QUORUM
            and agent.status in (AgentStatus.SCORED, AgentStatus.LIVE)
            else None
        ),
        screening_attempts=[
            PublicScreeningAttempt(
                attempt_id=attempt.attempt_id,
                policy_version=attempt.policy_version,
                status=attempt.status,
                screener_hotkey=attempt.screener_hotkey,
                started_at=attempt.started_at,
                deadline=attempt.deadline,
                finished_at=attempt.finished_at,
                reason=attempt.public_reason,
                quarantine_resolution=resolved_quarantines_by_attempt.get(
                    attempt.attempt_id, (None, None)
                )[0],
                quarantine_resolved_at=resolved_quarantines_by_attempt.get(
                    attempt.attempt_id, (None, None)
                )[1],
            )
            for attempt in attempts
        ],
        validation_attempts=[
            PublicValidationAttempt(
                validator_hotkey=ticket.validator_hotkey,
                status=ticket.status.value,
                issued_at=ticket.issued_at,
                deadline=ticket.deadline,
                bench_version=ticket.bench_version,
                actively_running=ticket.validator_hotkey in active_by_hotkey,
                benchmark_progress=(
                    _public_benchmark_progress(
                        active_by_hotkey[ticket.validator_hotkey]
                    )
                    if ticket.validator_hotkey in active_by_hotkey
                    else None
                ),
            )
            for ticket in tickets
        ],
        dispute=_public_dispute(dispute) if dispute is not None else None,
    )


@router.get("/submissions", response_model=PublicSubmissionsResponse)
async def submissions(
    response: Response,
    session: SessionDep,
    limit: int = Query(default=50, ge=1, le=200),
) -> PublicSubmissionsResponse:
    """Recent finalized submissions, most recently scored first.

    The index over the k=3 transparency records: each entry carries the median
    composite, how many validators scored it, and the dataset pin (seed +
    sha256); drill into ``/public/agent/{agent_id}/scores`` for the full
    per-validator breakdown. Held-for-review and still-evaluating agents are
    excluded — only settled public scores appear.
    """
    response.headers["Cache-Control"] = _CACHE_CONTROL
    rows = await list_public_submissions(session, limit=limit)
    return PublicSubmissionsResponse(
        generated_at=datetime.now(UTC),
        count=len(rows),
        quorum=SCORING_QUORUM,
        submissions=[_submission_summary(r) for r in rows],
    )


@router.get("/agent/{agent_id}/scores", response_model=PublicSubmissionScores)
async def agent_scores(
    response: Response,
    session: SessionDep,
    agent_id: UUID,
) -> PublicSubmissionScores:
    """The full k=3 scoring record for one finalized agent.

    Publishes which validators scored the agent, each one's exact numbers +
    signature (self-verifying against the published validator key), the median
    composite the platform finalized on, and the pinned dataset (seed + sha256)
    so anyone can reproduce and audit the score. 404 for an agent that does not
    exist or has not settled into a public status (still evaluating, or held for
    copy review) — a provisional agent's partial scores are never exposed.
    """
    response.headers["Cache-Control"] = _CACHE_CONTROL
    row = await get_submission_scores(session, agent_id=agent_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no public scores for this agent")
    return _submission_scores(row)


@router.get("/agent/{agent_id}/dataset", response_model=PublicDatasetReveal)
async def agent_dataset(
    response: Response,
    session: SessionDep,
    generator: GeneratorDep,
    agent_id: UUID,
) -> PublicDatasetReveal:
    """The FULL labeled dataset a finalized submission was scored against.

    Regenerated from the submission's published (on-chain-derived) seed so anyone
    can independently re-grade its k=3 scores. The regenerated artifact's SHA-256
    is re-verified against the hash pinned at scoring, so the revealed bytes
    provably are the scored dataset. 404 for an unknown / not-yet-finalized agent
    (a provisional agent's answers are never revealed); 502 if the generator drifts
    from the pinned hash; 503 if the generate service is unavailable.

    Safe despite carrying the answer key: the seed is one-time and was
    unpredictable at submission, so a past dataset's answers cannot help overfit a
    future (differently-seeded) run.
    """
    # A finalized dataset never changes (fixed seed), so it is immutable + highly
    # cacheable.
    response.headers["Cache-Control"] = "public, max-age=3600, immutable"
    row = await get_submission_scores(session, agent_id=agent_id)
    if row is None or row.dataset_seed is None or row.dataset_run_size is None:
        raise HTTPException(
            status_code=404, detail="no revealable dataset for this agent"
        )
    try:
        artifact, sha = await generator.fetch_dataset(
            row.dataset_seed, row.dataset_run_size
        )
    except DataPipelineError as e:
        raise HTTPException(
            status_code=503, detail="dataset generate service unavailable"
        ) from e
    if row.dataset_sha256 and sha.lower() != row.dataset_sha256.lower():
        # The regenerated dataset does not hash to what was pinned at scoring —
        # generator drift. Refuse rather than serve a dataset that is not the one
        # that was scored.
        raise HTTPException(
            status_code=502,
            detail="regenerated dataset does not match the pinned hash",
        )
    bench_version = artifact.get("bench_version")
    return PublicDatasetReveal(
        agent_id=row.agent_id,
        miner_hotkey=row.miner_hotkey,
        seed=row.dataset_seed,
        run_size=row.dataset_run_size,
        dataset_sha256=sha,
        bench_version=bench_version if isinstance(bench_version, int) else None,
        dataset_seed_block=row.dataset_seed_block,
        dataset_seed_block_hash=row.dataset_seed_block_hash,
        artifact=artifact,
    )


@router.get("/audit", response_model=PublicAuditResponse)
async def audit(
    response: Response,
    session: SessionDep,
    since_seq: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
) -> PublicAuditResponse:
    """A page of the append-only, hash-chained score audit log, oldest first.

    The tamper-evident public projection of every scoring event: each validator's
    signed ``score`` and each ``agent_finalized`` (quorum reached, the median +
    scoring validators), in append order. Replay from ``since_seq=0`` and
    re-request with the last ``seq`` seen to stream new entries; recompute each
    ``entry_hash`` and check it links to the prior ``prev_hash`` (rooted at
    ``genesis_hash``) to prove nothing was reordered, edited, or dropped. Every
    ``score`` entry also carries the validator's sr25519 signature, so a consumer
    can verify authenticity against the published validator key. Never carries
    per-case answer-key content.
    """
    response.headers["Cache-Control"] = _CACHE_CONTROL
    entries = await list_audit_entries(session, since_seq=since_seq, limit=limit)
    return PublicAuditResponse(
        generated_at=datetime.now(UTC),
        count=len(entries),
        genesis_hash=GENESIS_HASH,
        head_hash=entries[-1].entry_hash if entries else None,
        entries=[
            PublicAuditEntry(
                seq=e.seq,
                agent_id=e.agent_id,
                validator_hotkey=e.validator_hotkey,
                event=e.event,
                payload=e.payload,
                prev_hash=e.prev_hash,
                entry_hash=e.entry_hash,
                recorded_at=e.recorded_at,
            )
            for e in entries
        ],
    )


def _corpus_per_case(details: object) -> list[dict[str, Any]]:
    """Extract the full UNREDACTED per-case list from a score's details blob."""
    if not isinstance(details, dict):
        return []
    per_case = details.get("per_case")
    if not isinstance(per_case, list):
        return []
    return [c for c in per_case if isinstance(c, dict)]


@router.get("/bench/{version}/corpus", response_model=PublicBenchCorpusResponse)
async def bench_corpus(
    response: Response,
    session: SessionDep,
    version: int,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> PublicBenchCorpusResponse:
    """The FULL labeled corpus of a RETIRED benchmark version (answer keys included).

    Once a benchmark is superseded it is never scored again, so its per-case answer
    keys carry zero anti-overfit cost and are released verbatim from the stored
    scores for research + audit. Refused with 409 for the current (live) version or
    any unknown future version, so no live answer key is ever exposed here.
    Paginate with ``limit`` / ``offset`` up to ``total``.
    """
    # Retirement follows the ACTIVATED epoch, not the shipped constant: releasing
    # answer keys is irreversible and must not happen before miners are notified.
    active = await active_bench_version(session)
    if not is_bench_version_retired(version, active):
        raise HTTPException(
            status_code=409,
            detail=(
                f"bench_version {version} is not retired (active is "
                f"{active}); its answer keys are not released"
            ),
        )
    response.headers["Cache-Control"] = "public, max-age=3600, immutable"
    rows, total = await list_scores_for_bench_version(
        session, version=version, limit=limit, offset=offset
    )
    return PublicBenchCorpusResponse(
        bench_version=version,
        generated_at=datetime.now(UTC),
        count=len(rows),
        total=total,
        limit=limit,
        offset=offset,
        entries=[
            PublicBenchCorpusEntry(
                agent_id=score.agent_id,
                miner_hotkey=miner,
                validator_hotkey=score.validator_hotkey,
                seed=score.seed,
                run_id=score.run_id,
                composite=score.composite,
                per_case=_corpus_per_case(score.details),
            )
            for score, miner in rows
        ],
    )


@router.get("/bench/config", response_model=PublicBenchConfigResponse)
async def bench_config(response: Response) -> PublicBenchConfigResponse:
    """The current benchmark setup: frozen model, judge-free grading, seeds.

    The harness model is a consensus parameter: every scoring validator runs
    the same frozen open-weight artifact through a model-pinning gateway, so
    model choice is not a miner lever and k=3 scores are comparable. The
    ``BENCH_*`` env overrides exist for coordinated fleet bumps only.
    """
    response.headers["Cache-Control"] = "public, max-age=300"
    public_bucket = os.environ.get("STORAGE_PUBLIC_BUCKET", "")
    mirror = (
        f"https://storage.googleapis.com/{public_bucket}/scored/{{agent_id}}.json"
        if public_bucket
        else None
    )
    transcript_template = (
        f"https://storage.googleapis.com/{public_bucket}/transcripts/{{sha256}}.json"
        if public_bucket
        else None
    )
    return PublicBenchConfigResponse(
        bench_version=CURRENT_BENCH_VERSION,
        harness=BenchHarnessConfig(
            locked=True,
            canonical_id=os.environ.get("BENCH_HARNESS_MODEL_ID", "qwen/qwen3-32b"),
            serving=os.environ.get("BENCH_HARNESS_SERVING", "Qwen/Qwen3-32B-TEE"),
            thinking=os.environ.get("BENCH_HARNESS_THINKING", "false") == "true",
            enforcement=(
                "model-pinning relay forces the model field and holds the "
                "upstream key outside the sandbox; sandbox egress is deny-all "
                "(no other model is reachable)"
            ),
        ),
        grading=BenchGradingConfig(
            judge_free=True,
            grader="github.com/ditto-assistant/dittobench-datagen/grade",
            description=(
                "deterministic per-answer_kind checks with distractor and "
                "forbidden-value zeroing; a score is a pure function of "
                "(dataset, transcript)"
            ),
        ),
        dataset=BenchDatasetConfig(
            generator="github.com/ditto-assistant/dittobench-datagen",
            seed_derivation=(
                "derived from an on-chain block hash fixed AFTER the miner "
                "commits; unpredictable, one fresh dataset per submission"
            ),
            reproduce=(
                "generate -seed <seed> -run-size full -sha reproduces any "
                "scored run's exact bytes and dataset_sha256"
            ),
        ),
        public_mirror_url_template=mirror,
        public_transcript_url_template=transcript_template,
        ledger_path="/api/v1/scoring/scores",
        generated_at=datetime.now(UTC),
    )


@router.get("/bench/rollout")
async def benchmark_rollout_state(
    response: Response, session: SessionDep
) -> dict[str, object]:
    """Expose desired/active versions and the frozen cohort's exact progress."""
    response.headers["Cache-Control"] = "public, max-age=30"
    return await rollout_state(session)
