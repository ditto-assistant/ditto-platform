"""Public, unauthenticated read endpoints for the subnet dashboard.

Two surfaces, both open (no credentials) and both fronting the same DB the
validator-gated ``/scoring/scores`` reads:

* **Aggregate leaderboard / health** (``/leaderboard``, ``/health``): best score
  per miner, composite plus tool/memory means and rank, never exposing per-case
  answer-key detail. This half stays aggregate-only.
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
KOTH champion / weight vector is deliberately **not** served here — that is
validator-side (see the scoring endpoint's boundary note); the dashboard reads
weights from wandb or the chain.
"""

from __future__ import annotations

import logging
import math
import os
import statistics
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response

from ditto.api_models import (
    BenchDatasetConfig,
    BenchGradingConfig,
    BenchHarnessConfig,
    PublicAuditEntry,
    PublicAuditResponse,
    PublicBenchConfigResponse,
    PublicBenchCorpusEntry,
    PublicBenchCorpusResponse,
    PublicBenchIntegrity,
    PublicCaseResult,
    PublicCategoryStat,
    PublicDatasetReveal,
    PublicHealthResponse,
    PublicLeaderboardEntry,
    PublicLeaderboardResponse,
    PublicRunModels,
    PublicSubmissionScores,
    PublicSubmissionsResponse,
    PublicSubmissionSummary,
    PublicValidatorScore,
)
from ditto.api_server.bench import CURRENT_BENCH_VERSION, is_bench_version_retired
from ditto.api_server.datapipeline import DataPipelineError
from ditto.api_server.endpoints.screener import GeneratorDep
from ditto.api_server.endpoints.validator import SessionDep
from ditto.db.queries.audit import GENESIS_HASH, list_audit_entries
from ditto.db.queries.scores import (
    SCORING_QUORUM,
    LedgerRow,
    SubmissionRow,
    get_public_health,
    get_submission_scores,
    list_eligible_ledger,
    list_miner_composite_history,
    list_public_submissions,
    list_scores_for_bench_version,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/public", tags=["public"])

# The ledger only moves when a sweep records a new best score, so a short shared
# cache is safe and shields the DB from dashboard/CDN traffic.
_CACHE_CONTROL = "public, max-age=30"


def _safe_models(details: dict) -> PublicRunModels | None:
    """Pull the run's models from the details blob, tolerating a malformed shape."""
    raw = details.get("models")
    if not isinstance(raw, dict):
        return None
    try:
        return PublicRunModels.model_validate(raw)
    except Exception:  # noqa: BLE001 - a bad blob must not break the leaderboard
        return None


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


def _safe_stderr(details: dict) -> float | None:
    """Estimate the composite's standard error from the per-case score spread.

    ``composite = 0.5 * tool_mean + 0.5 * memory_mean`` (B6 weighting). Treating
    the tool and memory case sets as independent samples, the SE of that weighted
    sum is ``sqrt(0.25*se_tool^2 + 0.25*se_memory^2)`` where each ``se`` is the
    standard error of the mean of its kind's per-case scores. Derived from the
    stored ``details.per_case`` (never exposed itself) — so the leaderboard can
    show error bars / a statistical-tie band without a re-score. ``None`` when
    there is no usable per-case data; a kind with <2 cases contributes SE 0.
    """
    per_case = details.get("per_case")
    if not isinstance(per_case, list):
        return None
    tool: list[float] = []
    memory: list[float] = []
    for c in per_case:
        if not isinstance(c, dict):
            continue
        score = c.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        kind = c.get("kind")
        if kind == "tool":
            tool.append(float(score))
        elif kind == "memory":
            memory.append(float(score))
    if not tool and not memory:
        return None

    def _sem(xs: list[float]) -> float:
        n = len(xs)
        if n < 2:
            return 0.0
        mean = sum(xs) / n
        var = sum((x - mean) ** 2 for x in xs) / (n - 1)
        return math.sqrt(var / n)

    se_t = _sem(tool)
    se_m = _sem(memory)
    return math.sqrt(0.25 * se_t * se_t + 0.25 * se_m * se_m)


def _public_entry(
    rank: int, r: LedgerRow, history: list[float] | None = None
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
        agent_id=r.agent_id,
        miner_hotkey=r.miner_hotkey,
        composite=r.composite,
        composite_stderr=_safe_stderr(details),
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


@router.get("/leaderboard", response_model=PublicLeaderboardResponse)
async def leaderboard(
    response: Response,
    session: SessionDep,
) -> PublicLeaderboardResponse:
    """Best eligible score per miner, aggregate-only, highest composite first."""
    response.headers["Cache-Control"] = _CACHE_CONTROL
    rows = await list_eligible_ledger(session)
    histories = await list_miner_composite_history(
        session, [r.miner_hotkey for r in rows]
    )
    entries = [
        _public_entry(i, r, histories.get(r.miner_hotkey))
        for i, r in enumerate(rows, start=1)
    ]
    return PublicLeaderboardResponse(
        generated_at=datetime.now(UTC),
        count=len(entries),
        current_bench_version=CURRENT_BENCH_VERSION,
        entries=entries,
    )


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
        scores_24h=roll.scores_24h,
        avg_latency_ms=roll.avg_latency_ms,
    )


def _median_composite(row: SubmissionRow) -> float | None:
    """Median of the reported composites — the canonical score, or None if unscored."""
    if not row.scores:
        return None
    return statistics.median(s.composite for s in row.scores)


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
        scores=[
            PublicValidatorScore(
                validator_hotkey=s.validator_hotkey,
                composite=s.composite,
                tool_mean=s.tool_mean,
                memory_mean=s.memory_mean,
                median_ms=s.median_ms,
                n=s.n,
                seed=s.seed,
                run_id=s.run_id,
                signature=s.signature,
                generated_at=s.generated_at,
                case_results=_safe_case_results(
                    s.details if isinstance(s.details, dict) else {}
                ),
            )
            for s in row.scores
        ],
        generated_at=datetime.now(UTC),
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
    if not is_bench_version_retired(version):
        raise HTTPException(
            status_code=409,
            detail=(
                f"bench_version {version} is not retired (current is "
                f"{CURRENT_BENCH_VERSION}); its answer keys are not released"
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
        ledger_path="/api/v1/scoring/scores",
        generated_at=datetime.now(UTC),
    )
