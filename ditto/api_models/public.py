"""Public, unauthenticated read models for the subnet dashboard.

These expose the **aggregate** shape only — composite plus tool/memory means and
rank — and deliberately omit the fields on :class:`LedgerEntry` that are either
integrity-internal (``sha256``, ``signature``, ``validator_hotkey``) or would
hand a miner the benchmark's answer key (per-case ``expected``/``called``). See
``docs/public-telemetry.md`` for the transparency policy this encodes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

_SS58_PATTERN = r"^[1-9A-HJ-NP-Za-km-z]{47,48}$"


class PublicCategoryStat(BaseModel):
    """One category's mean in a run's per-category breakdown (public)."""

    category: Annotated[
        str, Field(description="Category name (tool name / memory type).")
    ]
    count: Annotated[int, Field(ge=0, description="Cases scored in this category.")]
    mean: Annotated[float, Field(ge=0.0, le=1.0, description="Mean score in [0,1].")]


class PublicBenchIntegrity(BaseModel):
    """Anti-overfit / scoring-integrity telemetry for a scored run (public).

    These describe *how the dataset resists gaming*, not the miner's answers:
    the paraphrase pass (reword-or-fallback), the NoLiMa lexical-gap rewrite
    (questions reworded to share fewer content words with the stored fact), how
    many tool cases were capped because the harness self-reported instead of
    calling the observable endpoint, and the memory seeding-wave count. They are
    uniform across miners scored on the same seed/version and exist so the
    community can audit the benchmark's anti-overfit posture.
    """

    paraphrase_applied: Annotated[
        int | None,
        Field(default=None, ge=0, description="Cases whose text was paraphrased."),
    ]
    paraphrase_attempted: Annotated[
        int | None,
        Field(default=None, ge=0, description="Cases the paraphraser was run on."),
    ]
    paraphrase_fallback: Annotated[
        int | None,
        Field(
            default=None,
            ge=0,
            description="Paraphrases that failed verify and fell back to template.",
        ),
    ]
    lexical_gap_rewritten: Annotated[
        int | None,
        Field(default=None, ge=0, description="Questions reworded to drop a word."),
    ]
    lexical_gap_questions: Annotated[
        int | None,
        Field(default=None, ge=0, description="Questions considered for lexical gap."),
    ]
    lexical_gap_mean_before: Annotated[
        float | None,
        Field(default=None, ge=0.0, description="Mean shared-content overlap before."),
    ]
    lexical_gap_mean_after: Annotated[
        float | None,
        Field(default=None, ge=0.0, description="Mean shared-content overlap after."),
    ]
    capped_tool_cases: Annotated[
        int | None,
        Field(
            default=None,
            ge=0,
            description="Tool cases capped (self-report untrusted, not via endpoint).",
        ),
    ]
    seeding_waves: Annotated[
        int | None,
        Field(default=None, ge=0, description="Memory seeding waves in this run."),
    ]


class PublicCaseResult(BaseModel):
    """One scored case, **redacted** for public per-case analysis.

    Carries only *how the agent did* on the case — its category, kind, score,
    pass/fail, latency, and the scorer's mechanical notes (e.g. "1 extra tool
    call", "capped: self-report untrusted"). It deliberately **omits the answer
    key**: the ``expected`` tools/answer, the agent's ``called`` tools (which on a
    correct case would reveal ``expected``), and the seed-derived ``case_id``.
    Combined with per-submission seed rotation, this lets anyone inspect a run's
    per-case strengths/weaknesses without learning anything that helps overfit.
    """

    category: Annotated[
        str, Field(description="Case category (tool name / memory question type).")
    ]
    kind: Annotated[str, Field(description='"tool" or "memory".')]
    score: Annotated[float, Field(ge=0.0, le=1.0, description="Case score in [0,1].")]
    correct: Annotated[
        bool | None, Field(default=None, description="Whether the case passed.")
    ]
    latency_ms: Annotated[
        int | None, Field(default=None, ge=0, description="Case latency (ms).")
    ]
    notes: Annotated[
        list[str] | None,
        Field(default=None, description="Scorer's mechanical notes (no answers)."),
    ]


class PublicRunModels(BaseModel):
    """The LLM models a scored run was produced with (public transparency)."""

    generator: Annotated[
        str | None, Field(default=None, description="Datagen model id.")
    ]
    judge: Annotated[
        str | None, Field(default=None, description="Judge/scorer model id.")
    ]
    judge_audit: Annotated[
        str | None,
        Field(default=None, description="Second (audit) judge model id, if any."),
    ]
    harness: Annotated[
        str | None,
        Field(
            default=None,
            description="Miner harness chat model id, when the operator pinned it.",
        ),
    ]


class PublicLeaderboardEntry(BaseModel):
    """One miner's best score, aggregate-only, for public display.

    Beyond the headline composite + tool/memory means, this carries the
    benchmark provenance a transparent leaderboard needs — the models that
    generated + graded the run, the ``bench_version`` and ``dataset_sha256``
    (which pins the exact scored artifact for a dispute re-score), latency, case
    count, and a per-category breakdown. All are advisory and deliberately
    exclude the raw ``seed`` (anti-overfit) and any per-case answer-key content
    (``expected`` / ``called``).
    """

    rank: Annotated[int, Field(ge=1, description="1-based rank by composite.")]
    miner_hotkey: Annotated[
        str, Field(pattern=_SS58_PATTERN, description="Miner's SS58 hotkey.")
    ]
    composite: Annotated[
        float, Field(ge=0.0, le=1.0, description="Best composite in [0,1].")
    ]
    composite_stderr: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            description=(
                "Standard error of the composite, estimated from the per-case "
                "score spread — the measurement uncertainty behind the headline "
                "number. Lets a consumer draw error bars and judge whether two "
                "miners are a statistical tie (the same signal the validator's "
                "indifference-band dethroning uses). None when the run carries no "
                "per-case data to estimate from."
            ),
        ),
    ]
    tool_mean: Annotated[
        float, Field(ge=0.0, le=1.0, description="Mean tool accuracy in [0,1].")
    ]
    memory_mean: Annotated[
        float, Field(ge=0.0, le=1.0, description="Mean memory recall in [0,1].")
    ]
    first_seen: Annotated[
        datetime, Field(description="When the winning agent was first uploaded (UTC).")
    ]
    median_ms: Annotated[
        int | None,
        Field(default=None, ge=0, description="Median per-case latency (ms)."),
    ]
    n: Annotated[
        int | None, Field(default=None, ge=0, description="Number of cases scored.")
    ]
    bench_version: Annotated[
        int | None, Field(default=None, description="Benchmark scoring version.")
    ]
    dataset_sha256: Annotated[
        str | None,
        Field(default=None, description="SHA-256 of the scored dataset artifact."),
    ]
    models: Annotated[
        PublicRunModels | None,
        Field(default=None, description="LLM models that produced + graded the run."),
    ]
    per_category: Annotated[
        list[PublicCategoryStat] | None,
        Field(default=None, description="Per-category (per tool / memory type) means."),
    ]
    integrity: Annotated[
        PublicBenchIntegrity | None,
        Field(default=None, description="Anti-overfit / scoring-integrity telemetry."),
    ]
    tokens: Annotated[
        int | None,
        Field(default=None, ge=0, description="LLM tokens spent generating+judging."),
    ]
    history: Annotated[
        list[float] | None,
        Field(
            default=None,
            description=(
                "This miner's recent composite scores, oldest→newest (across their "
                "submissions / re-scores), for a trend sparkline. Aggregate only — "
                "no seeds, no per-case content. None / omitted when there is no "
                "history beyond the current score."
            ),
        ),
    ]
    case_results: Annotated[
        list[PublicCaseResult] | None,
        Field(
            default=None,
            description=(
                "Redacted per-case results for detailed analysis — each case's "
                "category / kind / score / pass / latency / mechanical notes, but "
                "never the answer key (``expected`` / ``called`` / ``case_id``). "
                "None when the run carries no per-case data."
            ),
        ),
    ]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rank": 1,
                "miner_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
                "composite": 0.587,
                "composite_stderr": 0.014,
                "tool_mean": 0.867,
                "memory_mean": 0.167,
                "first_seen": "2026-07-03T20:00:00Z",
                "median_ms": 2720,
                "n": 12,
                "bench_version": 4,
                "dataset_sha256": "9f2c…",
                "models": {
                    "generator": "google/gemini-3.1-flash-lite",
                    "judge": "google/gemini-3.1-flash-lite",
                    "harness": "google/gemini-3.1-flash-lite",
                },
                "per_category": [
                    {"category": "memory_lookup", "count": 6, "mean": 1.0},
                    {"category": "web_search", "count": 1, "mean": 0.5},
                ],
                "integrity": {
                    "paraphrase_applied": 20,
                    "paraphrase_attempted": 20,
                    "paraphrase_fallback": 0,
                    "lexical_gap_rewritten": 2,
                    "lexical_gap_questions": 5,
                    "lexical_gap_mean_before": 0.45,
                    "lexical_gap_mean_after": 0.2,
                    "capped_tool_cases": 4,
                    "seeding_waves": 1,
                },
                "tokens": 7622,
                "history": [0.502, 0.548, 0.571, 0.587],
                "case_results": [
                    {
                        "category": "web_search",
                        "kind": "tool",
                        "score": 0.6,
                        "correct": False,
                        "latency_ms": 3382,
                        "notes": ["1 extra/unexpected tool call(s)"],
                    },
                    {
                        "category": "preference",
                        "kind": "memory",
                        "score": 1.0,
                        "correct": True,
                        "latency_ms": 1333,
                        "notes": ["deterministic answer match (no judge call)"],
                    },
                ],
            }
        }
    )


class PublicLeaderboardResponse(BaseModel):
    """The public best-score-per-miner leaderboard, highest composite first."""

    generated_at: Annotated[
        datetime, Field(description="When this snapshot was read (UTC).")
    ]
    count: Annotated[int, Field(ge=0, description="Number of entries.")]
    current_bench_version: Annotated[
        int,
        Field(
            description=(
                "The latest DittoBench benchmark version. Entries whose "
                "bench_version is below this were scored on a previous benchmark "
                "and are not directly comparable; the UI marks them as such."
            )
        ),
    ]
    entries: Annotated[
        list[PublicLeaderboardEntry],
        Field(default_factory=list, description="Ranked miners, best composite first."),
    ]


class PublicHealthResponse(BaseModel):
    """Aggregate subnet-health rollup for the public dashboard.

    Derived only from what the platform records (submissions + reported scores).
    Run started/failed counts, set-weights latency and per-stage timings are
    validator-side telemetry (wandb), not served here — the platform only ever
    sees a *successful* score, so it deliberately reports no "success rate".
    """

    generated_at: Annotated[
        datetime, Field(description="When this snapshot was read (UTC).")
    ]
    miners: Annotated[
        int, Field(ge=0, description="Distinct miners who have ever submitted.")
    ]
    scored_miners: Annotated[
        int, Field(ge=0, description="Distinct miners on the leaderboard (scored).")
    ]
    scored_agents: Annotated[
        int, Field(ge=0, description="Agents currently eligible (scored).")
    ]
    last_scored_at: Annotated[
        datetime | None,
        Field(default=None, description="When a validator last scored anything (UTC)."),
    ]
    scores_24h: Annotated[
        int, Field(ge=0, description="Scores generated in the last 24h.")
    ]
    avg_latency_ms: Annotated[
        int | None,
        Field(
            default=None, ge=0, description="Mean per-score median case latency (ms)."
        ),
    ]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "generated_at": "2026-07-04T12:00:00Z",
                "miners": 12,
                "scored_miners": 5,
                "scored_agents": 7,
                "last_scored_at": "2026-07-04T11:52:00Z",
                "scores_24h": 9,
                "avg_latency_ms": 812,
            }
        }
    )
