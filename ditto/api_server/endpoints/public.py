"""Public, unauthenticated read endpoints for the subnet dashboard.

Unlike ``/scoring/scores`` (validator-hotkey gated, full signed rows), this
surface is open and **aggregate-only**: composite plus tool/memory means and
rank, so a public leaderboard / dashboard can read scores with no credentials
while never exposing per-case detail (the benchmark's answer key) or
integrity-internal fields. See ``docs/public-telemetry.md``.

Responses are cacheable (``max-age=30``) so a CDN / the dashboard can front this
cheaply; the underlying ledger only changes when a sweep records a new best.
The KOTH champion / weight vector is deliberately **not** served here — that is
validator-side (see the scoring endpoint's boundary note); the dashboard reads
weights from wandb or the chain.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

from fastapi import APIRouter, Response

from ditto.api_models import (
    PublicBenchIntegrity,
    PublicCaseResult,
    PublicCategoryStat,
    PublicHealthResponse,
    PublicLeaderboardEntry,
    PublicLeaderboardResponse,
    PublicRunModels,
)
from ditto.api_server.bench import CURRENT_BENCH_VERSION
from ditto.api_server.endpoints.validator import SessionDep
from ditto.db.queries.scores import (
    LedgerRow,
    get_public_health,
    list_eligible_ledger,
    list_miner_composite_history,
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
    return PublicLeaderboardEntry(
        rank=rank,
        miner_hotkey=r.miner_hotkey,
        composite=r.composite,
        composite_stderr=_safe_stderr(details),
        tool_mean=r.tool_mean,
        memory_mean=r.memory_mean,
        first_seen=r.first_seen,
        median_ms=r.median_ms,
        n=r.n,
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
