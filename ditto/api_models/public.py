"""Public, unauthenticated read models for the subnet dashboard.

These expose the **aggregate** shape only — composite plus tool/memory means and
rank — and deliberately omit the fields on :class:`LedgerEntry` that are either
integrity-internal (``sha256``, ``signature``, ``validator_hotkey``) or would
hand a miner the benchmark's answer key (per-case ``expected``/``called``). See
``docs/public-telemetry.md`` for the transparency policy this encodes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ditto.api_models.benchmark_progress import BenchmarkProgressStage
from ditto.api_models.screener import ScreenerProgressStage, ScreenerRuntimeState
from ditto.api_models.stack_health import ValidatorStackHealth
from ditto.api_models.validator import ValidatorRuntimeState
from ditto.api_models.validator_capabilities import (
    ValidatorCapabilities,
    ValidatorStackIdentity,
)

_SS58_PATTERN = r"^[1-9A-HJ-NP-Za-km-z]{47,48}$"
_SIGNATURE_HEX_PATTERN = r"^[0-9a-fA-F]{128}$"


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
    finalized: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Whether the submission reached the three-validator quorum. "
                "False entries are provisional feedback and never drive weights."
            ),
        ),
    ]
    score_count: Annotated[
        int,
        Field(
            default=3,
            ge=1,
            description="Accepted independent validator scores currently available.",
        ),
    ]
    score_quorum: Annotated[
        int,
        Field(default=3, ge=1, description="Scores required for finalization."),
    ]
    agent_id: Annotated[
        UUID,
        Field(
            description=(
                "The scored agent's id, to drill into its k=3 record at "
                "/public/agent/{id}/scores. Already public via "
                "/public/submissions."
            )
        ),
    ]
    agent_name: Annotated[
        str,
        Field(description="Human-friendly name of the miner's winning agent."),
    ]
    agent_version: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Winning submission's version; null for legacy uploads.",
        ),
    ] = None
    miner_hotkey: Annotated[
        str, Field(pattern=_SS58_PATTERN, description="Miner's SS58 hotkey.")
    ]
    miner_uid: Annotated[
        int | None,
        Field(
            default=None,
            ge=0,
            description=(
                "Miner's current UID on this subnet; null when the hotkey is "
                "not registered or the chain snapshot is unavailable."
            ),
        ),
    ] = None
    registered: Annotated[
        bool | None,
        Field(
            default=None,
            description=(
                "Whether the miner hotkey currently has a UID on this subnet. "
                "False pauses weight and emission eligibility without deleting "
                "the submission or score; null means the chain snapshot was "
                "temporarily unavailable."
            ),
        ),
    ]
    emission_eligible: Annotated[
        bool | None,
        Field(
            default=None,
            description=(
                "Whether this entry is finalized on the current benchmark, "
                "full-benchmark eligible, and currently registered, so validators "
                "may include it in the active weight fold. Null when registration "
                "could not be read."
            ),
        ),
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
                "The exact standard error surfaced to the validator's KOTH fold: "
                "a stashed confirmation re-score SE when present, otherwise the "
                "between-validator SEM of the finalized k=3 quorum. This is the "
                "measurement uncertainty used by the public dethrone decision and "
                "the validator's indifference band. None when neither estimate is "
                "available."
            ),
        ),
    ]
    settled_composite: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            le=1.0,
            description=(
                "The agent's finalized median on the settled (active) benchmark "
                "version. Only populated in authoritative mode while a rollout is "
                "collecting the next version; null when there is no open rollout "
                "or the agent never reached quorum on the active version. This is "
                "the comparable baseline the dashboard ranks by mid-rollout, even "
                "for agents whose headline composite already flipped to the "
                "desired version."
            ),
        ),
    ] = None
    rollout_composite: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            le=1.0,
            description=(
                "Median of the agent's accepted scores on the desired (rolling "
                "out) benchmark version so far. Preliminary until "
                "rollout_score_count reaches score_quorum; null when there is no "
                "open rollout or no accepted score on the desired version yet."
            ),
        ),
    ] = None
    rollout_score_count: Annotated[
        int | None,
        Field(
            default=None,
            ge=0,
            description=(
                "Accepted validator scores on the desired benchmark version so "
                "far (the settlement state of rollout_composite, out of "
                "score_quorum). Null when there is no open rollout."
            ),
        ),
    ] = None
    calibration_brier: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            le=1.0,
            description=(
                "Mean Brier score over cases where the harness self-reported a "
                "confidence: mean((confidence - correct)^2), lower is better. "
                "Honest confidence minimizes it; always-100% does not. Advisory "
                "only — never folded into the composite, so a harness that omits "
                "confidence is unaffected. None when no case carried a confidence."
            ),
        ),
    ]
    calibration_n: Annotated[
        int | None,
        Field(
            default=None,
            ge=0,
            description=(
                "How many cases carried a self-reported confidence (the sample "
                "behind calibration_brier). None when zero."
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
    eligible: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Whether this run administered the full benchmark and is therefore "
                "score-rank eligible. Current weight and emission eligibility also "
                "requires finalized=true and registered=true. False marks a "
                "provisional smoke/practice "
                "run (a smaller run-size profile that omits the hard memory "
                "categories): it is shown for transparency but is not ranked and "
                "never earns emissions. The rank field is only meaningful for "
                "eligible entries."
            ),
        ),
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
                "finalized": True,
                "score_count": 3,
                "score_quorum": 3,
                "miner_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
                "miner_uid": 42,
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


class PublicEmissionRecipient(BaseModel):
    """One miner projected to receive a non-zero share of the KOTH miner pool."""

    role: Annotated[
        Literal["champion", "tail"],
        Field(description="Champion or participation-tail recipient."),
    ]
    agent_id: Annotated[UUID, Field(description="The recipient's folded agent id.")]
    miner_hotkey: Annotated[str, Field(pattern=_SS58_PATTERN)]
    raw_rank: Annotated[
        int,
        Field(
            ge=1,
            description="This entry's independent rank by finalized composite.",
        ),
    ]
    share_of_miner_pool: Annotated[
        float,
        Field(
            gt=0.0,
            le=1.0,
            description=(
                "Relative KOTH weight within the miner pool, before the subnet's "
                "separate miner-emission cap."
            ),
        ),
    ]


class PublicDethroneDecision(BaseModel):
    """The raw leader's comparison against the incumbent KOTH champion."""

    challenger_lead: float
    required_lead: Annotated[float, Field(ge=0.0)]
    margin_lead: Annotated[float, Field(ge=0.0)]
    statistical_lead: Annotated[float | None, Field(default=None, ge=0.0)]
    method: Literal["flat", "unpaired", "paired"]
    dethrones: bool


class PublicKothEmissions(BaseModel):
    """Current read-only projection of the validator's KOTH weight fold."""

    margin: Annotated[float, Field(ge=0.0, le=1.0)]
    dethrone_z: Annotated[float, Field(ge=0.0)]
    champion_share: Annotated[float, Field(gt=0.0, le=1.0)]
    tail_size: Annotated[int, Field(ge=0)]
    champion_agent_id: UUID
    champion_miner_hotkey: Annotated[str, Field(pattern=_SS58_PATTERN)]
    raw_leader_agent_id: UUID
    raw_leader_miner_hotkey: Annotated[str, Field(pattern=_SS58_PATTERN)]
    raw_leader_decision: PublicDethroneDecision | None = None
    recipients: list[PublicEmissionRecipient] = Field(default_factory=list)


class PublicLeaderboardResponse(BaseModel):
    """Raw score standings plus the current KOTH emissions projection."""

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
    active_bench_version: Annotated[
        int,
        Field(description="Globally activated benchmark version."),
    ]
    desired_bench_version: Annotated[
        int,
        Field(
            description=(
                "Version currently being collected, or the active version when "
                "there is no open rollout."
            )
        ),
    ]
    selection_mode: Annotated[
        Literal["authoritative", "historical"],
        Field(
            description=(
                "authoritative selects v3 per agent at quorum and otherwise its "
                "active-version fallback; historical is a requested single version."
            )
        ),
    ]
    entries: Annotated[
        list[PublicLeaderboardEntry],
        Field(default_factory=list, description="Ranked miners, best composite first."),
    ]
    emissions: Annotated[
        PublicKothEmissions | None,
        Field(
            default=None,
            description=(
                "Current KOTH fold over finalized, full-benchmark entries on the "
                "current benchmark. Null when no entry can receive emissions."
            ),
        ),
    ] = None


class PublicChainWeight(BaseModel):
    """One non-zero destination in a validator's revealed chain vector."""

    uid: Annotated[int, Field(ge=0)]
    hotkey: Annotated[str, Field(pattern=_SS58_PATTERN)]
    value: Annotated[int, Field(gt=0, le=65535)]


class PublicValidatorWeightVector(BaseModel):
    """One validator's latest publicly revealed on-chain weights."""

    validator_uid: Annotated[int, Field(ge=0)]
    validator_hotkey: Annotated[str, Field(pattern=_SS58_PATTERN)]
    weights: list[PublicChainWeight] = Field(default_factory=list)


class PublicChainWeightsResponse(BaseModel):
    """Block-consistent SN118 weight matrix read from Subtensor storage."""

    generated_at: datetime
    netuid: Annotated[int, Field(ge=0)]
    block: Annotated[int, Field(ge=0)]
    block_hash: Annotated[str, Field(pattern=r"^0x[0-9a-fA-F]{64}$")]
    owner_hotkey: Annotated[str | None, Field(default=None, pattern=_SS58_PATTERN)]
    vectors: list[PublicValidatorWeightVector] = Field(default_factory=list)


class PublicValidatorScore(BaseModel):
    """One validator's score for a submission, published verbatim (public).

    The per-validator half of the k=3 transparency record: *which* validator
    scored the agent and the exact numbers it reported, including its sr25519
    ``signature`` so the row is independently verifiable against the published
    validator public key. Unlike the aggregate leaderboard this deliberately
    exposes ``validator_hotkey`` (a public on-chain identity) and the raw
    ``seed`` — the whole point of the record is to show *who* scored an agent on
    *which* dataset, so an observer can reproduce and audit the number.
    """

    validator_hotkey: Annotated[
        str, Field(pattern=_SS58_PATTERN, description="Scoring validator's hotkey.")
    ]
    composite: Annotated[
        float, Field(ge=0.0, le=1.0, description="Composite this validator reported.")
    ]
    tool_mean: Annotated[
        float, Field(ge=0.0, le=1.0, description="Mean tool accuracy in [0,1].")
    ]
    memory_mean: Annotated[
        float, Field(ge=0.0, le=1.0, description="Mean memory recall in [0,1].")
    ]
    median_ms: Annotated[int, Field(ge=0, description="Median per-case latency (ms).")]
    n: Annotated[int, Field(ge=0, description="Number of cases scored.")]
    seed: Annotated[
        int,
        Field(
            description=(
                "Dataset seed this validator scored on. The platform draws it "
                "after screening (the miner never sees it before submitting), so "
                "publishing it post-hoc enables reproduction/audit without letting "
                "anyone pre-overfit a future submission."
            )
        ),
    ]
    run_id: Annotated[
        str, Field(description="Scoring-engine run id the signature is bound to.")
    ]
    ticket_deadline: Annotated[
        datetime | None,
        Field(
            default=None,
            description=(
                "Exact ticket lease bound into current score signatures. Null "
                "identifies a legacy score recorded before lease-bound signing."
            ),
        ),
    ]
    signature: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "sr25519 signature over the score payload, hex. Current signatures "
                "include ticket_deadline; legacy rows with a null deadline use the "
                "pre-lease payload and remain valid."
            ),
        ),
    ]
    generated_at: Annotated[
        datetime, Field(description="When the scoring engine produced the score (UTC).")
    ]
    transform_robustness: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            le=1.0,
            description=(
                "Reproduce-under-transform audit result: the fraction of audit "
                "pairs this run answered consistently. A share of every run's "
                "cases is re-asked under a rephrasing (or a shift that moves the "
                "answer) derived from the block-hash-seeded dataset seed, which "
                "postdates the submission's commit -- so the miner could not have "
                "pre-handled it. What a low value measures is SURFACE "
                "BRITTLENESS (right on the phrasing the harness was built for, "
                "wrong on one it was not) or MEMORIZATION; it is not evidence "
                "about a harness that genuinely recomputes the answer, which "
                "scores the same under the transform. Both the selection and the "
                "transforms are pure functions of the published seed, so anyone "
                "can regenerate the audit set and recheck this number. Null for "
                "a run that carried no audit pairs or predates the audit."
            ),
        ),
    ]
    audit_case_count: Annotated[
        int | None,
        Field(
            default=None,
            ge=0,
            description=(
                "How many audit pairs backed ``transform_robustness``, so a value "
                "backed by many pairs is distinguishable from one backed by two."
            ),
        ),
    ]
    case_results: Annotated[
        list[PublicCaseResult] | None,
        Field(
            default=None,
            description=(
                "Redacted per-case breakdown of this validator's run — each case's "
                "category / kind / score / pass / latency / mechanical notes, so an "
                "observer can audit exactly where the agent gained or lost points. "
                "Never the answer key (expected / called / case_id). None when the "
                "run carries no per-case data."
            ),
        ),
    ]
    transcript_sha256: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "SHA-256 of this validator's published transcript artifact (the "
                "graded per-case inputs), bound into the score signature. The "
                "bytes live content-addressed in the public bucket at "
                "``transcripts/{sha256}.json``; regenerating the dataset from "
                "the seed and re-running the public grader over the transcript "
                "reproduces this score offline. Null for scores whose validator "
                "published no transcript."
            ),
        ),
    ]


class PublicSubmissionScores(BaseModel):
    """The full k=3 scoring record for one submission (public transparency).

    Publishes, per agent: which validators scored it, each validator's exact
    numbers + signature, and the ``median_composite`` the platform finalized on
    (the canonical score no single validator controls). ``score_count`` reaching
    ``quorum`` is what finalized the agent; a re-scored agent may carry more than
    ``quorum`` rows (older + current runs). The dataset pin (``dataset_seed`` +
    ``dataset_sha256``) identifies the exact bytes all validators scored.
    """

    agent_id: Annotated[UUID, Field(description="The scored agent's id.")]
    miner_hotkey: Annotated[
        str, Field(pattern=_SS58_PATTERN, description="Submitting miner's SS58 hotkey.")
    ]
    status: Annotated[str, Field(description='Public status ("scored" or "live").')]
    quorum: Annotated[
        int, Field(ge=1, description="Validators required to finalize (k=3).")
    ]
    score_count: Annotated[
        int, Field(ge=0, description="Score rows recorded for this agent.")
    ]
    median_composite: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            le=1.0,
            description="Median of the reported composites — the canonical score.",
        ),
    ]
    dataset_seed: Annotated[
        int | None,
        Field(default=None, description="Platform-pinned dataset seed (regenerable)."),
    ]
    dataset_sha256: Annotated[
        str | None,
        Field(default=None, description="SHA-256 of the pinned dataset artifact."),
    ]
    dataset_run_size: Annotated[
        str | None,
        Field(default=None, description="Generator profile (small|medium|full)."),
    ]
    dataset_seed_block: Annotated[
        int | None,
        Field(
            default=None,
            description=(
                "On-chain block number the seed was derived from. Fetch this "
                "block's hash and recompute derive_seed(hash, agent_id) to verify "
                "the seed was not platform-chosen. Null on the CSPRNG fallback "
                "(chain was unavailable at job-ready)."
            ),
        ),
    ]
    dataset_seed_block_hash: Annotated[
        str | None,
        Field(
            default=None,
            description="Hash of dataset_seed_block; the seed's verification input.",
        ),
    ]
    scores: Annotated[
        list[PublicValidatorScore],
        Field(default_factory=list, description="Per-validator scores, by hotkey."),
    ]
    generated_at: Annotated[
        datetime, Field(description="When this snapshot was read (UTC).")
    ]


class PublicSubmissionSummary(BaseModel):
    """One row of the public recent-submissions index (drill into the detail)."""

    agent_id: Annotated[UUID, Field(description="The scored agent's id.")]
    miner_hotkey: Annotated[
        str, Field(pattern=_SS58_PATTERN, description="Submitting miner's SS58 hotkey.")
    ]
    status: Annotated[str, Field(description='Public status ("scored" or "live").')]
    score_count: Annotated[
        int, Field(ge=0, description="Score rows recorded for this agent.")
    ]
    median_composite: Annotated[
        float | None,
        Field(default=None, ge=0.0, le=1.0, description="Median canonical composite."),
    ]
    dataset_seed: Annotated[
        int | None, Field(default=None, description="Platform-pinned dataset seed.")
    ]
    dataset_sha256: Annotated[
        str | None, Field(default=None, description="SHA-256 of the pinned dataset.")
    ]
    last_scored_at: Annotated[
        datetime | None,
        Field(default=None, description="Most recent score time for this agent (UTC)."),
    ]


class PublicSubmissionsResponse(BaseModel):
    """The public recent-submissions index, most recently scored first."""

    generated_at: Annotated[
        datetime, Field(description="When this snapshot was read (UTC).")
    ]
    count: Annotated[int, Field(ge=0, description="Number of submissions returned.")]
    quorum: Annotated[
        int, Field(ge=1, description="Validators required to finalize (k=3).")
    ]
    submissions: Annotated[
        list[PublicSubmissionSummary],
        Field(default_factory=list, description="Recent finalized submissions."),
    ]


class PublicBenchmarkProgress(BaseModel):
    """Ticket-validated and coarsened public benchmark progress allowlist."""

    agent_id: UUID
    agent_name: str
    bench_version: Annotated[
        int, Field(ge=1, description="DittoBench contract bound to this ticket.")
    ]
    started_at: Annotated[
        datetime, Field(description="When the validator ticket was issued (UTC).")
    ]
    stage: BenchmarkProgressStage | None = None
    completed_checks: Annotated[int | None, Field(default=None, ge=0)] = None
    total_checks: Annotated[int | None, Field(default=None, ge=1)] = None
    percent: Annotated[int | None, Field(default=None, ge=0, le=95, multiple_of=5)] = (
        None
    )


class PublicActivityEntry(BaseModel):
    """One submission's safe, public lifecycle state."""

    agent_id: Annotated[UUID, Field(description="The submitted agent's id.")]
    miner_hotkey: Annotated[
        str, Field(pattern=_SS58_PATTERN, description="Submitting miner's SS58 hotkey.")
    ]
    name: Annotated[str, Field(description="Miner-provided agent display name.")]
    version: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Submission version within this named agent; null for legacy uploads."
            ),
        ),
    ] = None
    status: Annotated[
        str,
        Field(
            description=(
                "Public lifecycle stage. Internal review and enforcement states are "
                "collapsed to under_review or rejected."
            )
        ),
    ]
    submitted_at: Annotated[
        datetime, Field(description="When the platform accepted the upload (UTC).")
    ]
    last_scored_at: Annotated[
        datetime | None,
        Field(
            default=None,
            description="When the platform most recently recorded a score (UTC).",
        ),
    ]
    screening_reason: Annotated[
        str | None,
        Field(default=None, description="Public-safe screening failure category."),
    ]
    duplicate_of: Annotated[
        UUID | None,
        Field(default=None, description="Earlier agent this submission may duplicate."),
    ]
    duplicate_name: Annotated[
        str | None,
        Field(default=None, description="Name of the matched submission."),
    ]
    duplicate_version: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Version of the matched submission; null when it is legacy.",
        ),
    ]
    review_reason: Annotated[
        str | None,
        Field(
            default=None,
            description="Anti-copy signals that routed this submission to review.",
        ),
    ]
    review_opened_at: Annotated[
        datetime | None,
        Field(
            default=None,
            description="When the active ATH review hold began (UTC).",
        ),
    ]
    preserved_composite: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            le=1.0,
            description=("Median composite preserved while an ATH review is active."),
        ),
    ]
    score_count: Annotated[
        int,
        Field(ge=0, description="Independent validator scores recorded so far."),
    ]
    provisional_composite: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            le=1.0,
            description="Mean composite across accepted validator scores so far.",
        ),
    ]
    validator_queue_rank: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Current global validator-assignment priority for a waiting "
                "submission. Validator-specific eligibility may skip a row."
            ),
        ),
    ]
    quorum: Annotated[
        int,
        Field(ge=1, description="Independent validator scores required to finalize."),
    ]
    screening_policy_version: Annotated[
        int, Field(ge=0, description="Latest completed screening policy version.")
    ]
    required_screening_policy_version: Annotated[
        int, Field(ge=1, description="Policy currently required by the platform.")
    ]
    screening_attempt_id: Annotated[
        UUID | None, Field(default=None, description="Active screening lease, if any.")
    ]
    screening_started_at: Annotated[
        datetime | None, Field(default=None, description="Active attempt start time.")
    ]
    screening_deadline: Annotated[
        datetime | None, Field(default=None, description="Active attempt deadline.")
    ]
    active_benchmarks: list[PublicBenchmarkProgress] = Field(default_factory=list)


class PublicActivityResponse(BaseModel):
    """Recent submission activity, newest first."""

    generated_at: Annotated[
        datetime, Field(description="When this snapshot was read (UTC).")
    ]
    count: Annotated[int, Field(ge=0, description="Number of submissions returned.")]
    total: Annotated[int, Field(ge=0, description="Total number of submissions.")]
    status_counts: Annotated[
        dict[str, int],
        Field(
            default_factory=dict,
            description=(
                "Counts by canonical public lifecycle stage before status filters "
                "are applied. Search filtering is reflected when present."
            ),
        ),
    ]
    page: Annotated[int, Field(ge=1, description="Current one-based page number.")]
    page_size: Annotated[int, Field(ge=1, description="Maximum entries per page.")]
    total_pages: Annotated[
        int, Field(ge=1, description="Total pages, or one when there are no entries.")
    ]
    entries: Annotated[
        list[PublicActivityEntry],
        Field(default_factory=list, description="Recent submissions, newest first."),
    ]


class PublicScreeningAttempt(BaseModel):
    """One append-only screening attempt shown in submission details."""

    attempt_id: UUID
    policy_version: Annotated[int, Field(ge=1)]
    status: Annotated[
        str,
        Field(pattern=r"^(running|passed|rejected|failed|expired|quarantined)$"),
    ]
    screener_hotkey: Annotated[str, Field(pattern=_SS58_PATTERN)]
    started_at: datetime
    deadline: datetime
    finished_at: datetime | None = None
    reason: str | None = None
    quarantine_resolution: Literal["release", "rescreen", "reject"] | None = None
    quarantine_resolved_at: datetime | None = None


class PublicScreeningDispute(BaseModel):
    """Public-safe appeal state; the miner's private message is never exposed."""

    status: Literal["pending", "resolved"]
    submitted_at: datetime
    resolved_at: datetime | None = None
    resolution: Literal["release", "uphold"] | None = None


class CreateScreeningDisputeRequest(BaseModel):
    """One signed appeal of a rejected screening decision."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: Annotated[str, Field(min_length=20, max_length=1000)]
    signature: Annotated[str, Field(pattern=_SIGNATURE_HEX_PATTERN)]


class CreateScreeningDisputeResponse(BaseModel):
    dispute: PublicScreeningDispute


class PublicValidationAttempt(BaseModel):
    """One validator ticket contributing toward the three-score quorum."""

    validator_hotkey: Annotated[str, Field(pattern=_SS58_PATTERN)]
    status: Annotated[str, Field(pattern=r"^(issued|scored|expired)$")]
    issued_at: datetime
    deadline: datetime
    bench_version: Annotated[int, Field(ge=1)]
    actively_running: bool = False
    benchmark_progress: PublicBenchmarkProgress | None = None


class PublicProvisionalScore(BaseModel):
    """One score the platform accepted toward a submission's quorum.

    This deliberately exposes only the numeric composite, the deterministic
    dataset inputs needed to reproduce it, and the same redacted per-question
    outcomes shown for finalized scores. Validator identity, signatures, ticket
    leases, answer keys, and scorer internals remain outside the public
    in-progress surface.
    """

    composite: Annotated[
        float, Field(ge=0.0, le=1.0, description="Accepted composite in [0,1].")
    ]
    seed: Annotated[
        str,
        Field(
            pattern=r"^\d+$",
            description=(
                "Exact decimal dataset seed fixed after the miner committed the "
                "submission. Encoded as a string to avoid JavaScript integer rounding."
            ),
        ),
    ]
    run_size: Annotated[
        str | None,
        Field(
            default=None,
            pattern=r"^(small|medium|full)$",
            description="Generator profile used for the score, when recorded.",
        ),
    ]
    bench_version: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="DittoBench version recorded with the score.",
        ),
    ]
    datagen_version: Annotated[
        str | None,
        Field(
            default=None,
            pattern=r"^v\d+\.\d+\.\d+$",
            description="Pinned dittobench-datagen module release for reproduction.",
        ),
    ]
    seed_source: Annotated[
        str,
        Field(
            pattern=r"^(on_chain|random_fallback|validator_local)$",
            description=(
                "Whether the post-commit seed was derived from an on-chain block, "
                "an unpredictable platform fallback, or chosen by the scoring "
                "validator because no per-submission dataset was pinned."
            ),
        ),
    ]
    dataset_sha256: Annotated[
        str | None,
        Field(
            default=None,
            pattern=r"^[0-9a-f]{64}$",
            description="Pinned hash of the exact generated dataset, when recorded.",
        ),
    ]
    accepted_at: Annotated[
        datetime, Field(description="When the platform accepted this score (UTC).")
    ]
    reproduction_command: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Copyable dittobench-datagen command pinned to the generator "
                "release used by the current benchmark."
            ),
        ),
    ]
    verification_command: Annotated[
        str | None,
        Field(
            default=None,
            description="Copyable command that prints the regenerated dataset hash.",
        ),
    ]
    case_results: Annotated[
        list[PublicCaseResult] | None,
        Field(
            default=None,
            description=(
                "Redacted per-question outcomes; answer keys and raw responses "
                "are never included."
            ),
        ),
    ]
    transcript_sha256: Annotated[
        str | None,
        Field(
            default=None,
            pattern=r"^[0-9a-f]{64}$",
            description=(
                "SHA-256 of this run's published transcript artifact (the "
                "graded per-case inputs), bound into the validator's score "
                "signature. The bytes live content-addressed in the public "
                "bucket at ``transcripts/{sha256}.json``; regenerating the "
                "dataset from the seed and re-running the public grader over "
                "the transcript reproduces this composite offline. Null when "
                "the validator published no transcript."
            ),
        ),
    ] = None


class PublicSubmissionPipeline(BaseModel):
    """Full public execution history for one submitted agent."""

    generated_at: datetime
    agent_id: UUID
    status: str
    active_bench_version: Annotated[int, Field(ge=1)]
    score_count: Annotated[int, Field(ge=0)]
    quorum: Annotated[int, Field(ge=1)]
    score_floor: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "Current finalized fifth-place score used for safe continuation "
                "after two scores; 0 when fewer than five ranked miners exist."
            ),
        ),
    ]
    provisional_scores: list[PublicProvisionalScore] = Field(default_factory=list)
    final_composite: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            le=1.0,
            description=(
                "Canonical median once quorum is reached; null while scores are "
                "still provisional."
            ),
        ),
    ]
    screening_attempts: list[PublicScreeningAttempt] = Field(default_factory=list)
    validation_attempts: list[PublicValidationAttempt] = Field(default_factory=list)
    dispute: PublicScreeningDispute | None = None


class PublicDatasetReveal(BaseModel):
    """The full labeled dataset a finalized submission was scored against.

    Regenerated from the submission's published (on-chain-derived) seed, so anyone
    can **independently re-grade** the k=3 scores: the ``artifact`` carries the
    complete DatasetArtifact including the answer keys (expected tools/answers).
    Safe to publish because the seed is one-time and unpredictable, so revealing a
    past submission's answers cannot help overfit any future (differently-seeded)
    run. ``dataset_sha256`` is re-verified to match what was pinned at scoring, so
    the revealed bytes provably are the scored dataset.
    """

    agent_id: Annotated[UUID, Field(description="The scored agent's id.")]
    miner_hotkey: Annotated[
        str, Field(pattern=_SS58_PATTERN, description="Submitting miner's SS58 hotkey.")
    ]
    seed: Annotated[int, Field(description="Dataset seed (on-chain derived).")]
    run_size: Annotated[
        str, Field(description="Generator profile (small|medium|full).")
    ]
    dataset_sha256: Annotated[
        str, Field(description="SHA-256 of the artifact, verified against the pin.")
    ]
    bench_version: Annotated[
        int | None,
        Field(default=None, description="Benchmark version of the artifact."),
    ]
    dataset_seed_block: Annotated[
        int | None,
        Field(default=None, description="On-chain block the seed was derived from."),
    ]
    dataset_seed_block_hash: Annotated[
        str | None, Field(default=None, description="Hash of the seed block.")
    ]
    artifact: Annotated[
        dict[str, Any],
        Field(
            description=(
                "The full labeled DatasetArtifact (tool + memory cases, seeding "
                "waves, fixtures, AND the answer keys) so the score is "
                "independently reproducible."
            )
        ),
    ]


class PublicBenchCorpusEntry(BaseModel):
    """One scored run of a retired benchmark, with its FULL answer key.

    Part of the retired-version corpus release: because a retired benchmark is
    never scored again, its per-case answer keys (``expected`` tools/answers,
    ``called``, ``case_id``) carry zero anti-overfit cost and are published
    verbatim from ``scores.details`` so researchers get the complete labeled
    benchmark.
    """

    agent_id: Annotated[UUID, Field(description="The scored agent's id.")]
    miner_hotkey: Annotated[
        str, Field(pattern=_SS58_PATTERN, description="Submitting miner's hotkey.")
    ]
    validator_hotkey: Annotated[
        str, Field(pattern=_SS58_PATTERN, description="Scoring validator's hotkey.")
    ]
    seed: Annotated[int, Field(description="Dataset seed for the run.")]
    run_id: Annotated[str, Field(description="Scoring-engine run id.")]
    composite: Annotated[
        float, Field(ge=0.0, le=1.0, description="Composite this validator reported.")
    ]
    per_case: Annotated[
        list[dict[str, Any]],
        Field(
            default_factory=list,
            description=(
                "Full UNREDACTED per-case records, answer keys included (retired "
                "version, so safe). Empty when the run stored no per-case data."
            ),
        ),
    ]


class PublicBenchCorpusResponse(BaseModel):
    """A page of a retired benchmark's full labeled corpus.

    Served only for a retired ``bench_version`` (``< current``); the live version
    is refused (409) since exposing its answer keys would be an overfit vector.
    Paginate with ``limit`` / ``offset`` up to ``total``.
    """

    bench_version: Annotated[int, Field(description="The retired benchmark version.")]
    generated_at: Annotated[
        datetime, Field(description="When this page was read (UTC).")
    ]
    count: Annotated[int, Field(ge=0, description="Entries in this page.")]
    total: Annotated[int, Field(ge=0, description="Total runs for this version.")]
    limit: Annotated[int, Field(ge=1, description="Page size.")]
    offset: Annotated[int, Field(ge=0, description="Page offset.")]
    entries: Annotated[
        list[PublicBenchCorpusEntry],
        Field(default_factory=list, description="Scored runs with full answer keys."),
    ]


class PublicAuditEntry(BaseModel):
    """One entry of the append-only, hash-chained public score audit log.

    Each entry records a scoring event verbatim: a validator's signed ``score``
    or an ``agent_finalized`` (quorum reached, the median + scoring validators).
    ``entry_hash`` is the SHA-256 of the entry's canonical content (which embeds
    ``prev_hash``); ``prev_hash`` links to the previous entry's ``entry_hash``.
    A consumer replays the feed and recomputes each hash to prove the sequence
    was never reordered, edited, or truncated.
    """

    seq: Annotated[int, Field(ge=1, description="Monotonic append order.")]
    agent_id: Annotated[UUID, Field(description="Agent the event is about.")]
    validator_hotkey: Annotated[
        str | None,
        Field(default=None, description="Scoring validator (null on finalize)."),
    ]
    event: Annotated[str, Field(description='"score" or "agent_finalized".')]
    payload: Annotated[
        dict[str, Any],
        Field(description="Event content (the hash preimage's payload field)."),
    ]
    prev_hash: Annotated[
        str, Field(description="Previous entry's entry_hash (hex); genesis = 64 zeros.")
    ]
    entry_hash: Annotated[
        str, Field(description="SHA-256 (hex) of this entry's canonical content.")
    ]
    recorded_at: Annotated[
        datetime, Field(description="When the platform appended the entry (UTC).")
    ]


class PublicAuditResponse(BaseModel):
    """A page of the public audit feed, oldest first, with the chain root.

    Paginate by ``seq``: replay from ``since_seq=0`` and re-request with the last
    ``seq`` seen to stream new entries. ``genesis_hash`` is the ``prev_hash`` of
    the very first entry, so a consumer can verify the chain from the root.
    """

    generated_at: Annotated[
        datetime, Field(description="When this page was read (UTC).")
    ]
    count: Annotated[int, Field(ge=0, description="Entries in this page.")]
    genesis_hash: Annotated[
        str, Field(description="The chain root (first entry's prev_hash).")
    ]
    head_hash: Annotated[
        str | None,
        Field(default=None, description="entry_hash of the last entry in this page."),
    ]
    entries: Annotated[
        list[PublicAuditEntry],
        Field(default_factory=list, description="Entries with seq > since_seq."),
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
    total_scores: Annotated[
        int, Field(ge=0, description="All validator score records ever recorded.")
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
                "total_scores": 18,
                "scores_24h": 9,
                "avg_latency_ms": 812,
            }
        }
    )


FleetAvailability = Literal["available", "stale", "offline", "paused", "unknown"]
FleetHealth = Literal["healthy", "warning", "unknown"]
ValidatorAssignmentState = Literal[
    "synchronized",
    "heartbeat_stale",
    "heartbeat_mismatch",
    "unassigned",
]


class PublicSystemMetrics(BaseModel):
    """Coarse allowlisted metrics; collector timestamps stay private."""

    cpu_percent: Annotated[int, Field(ge=0, le=100, multiple_of=5)]
    memory_percent: Annotated[int, Field(ge=0, le=100, multiple_of=5)]
    disk_percent: Annotated[int, Field(ge=0, le=100, multiple_of=5)]
    docker_status: Literal["healthy", "degraded", "unavailable"]
    running_containers: Annotated[int, Field(ge=0, le=1000)]
    unhealthy_containers: Annotated[int, Field(ge=0, le=1000)]


class PublicValidatorHeartbeat(BaseModel):
    """Latest signed software report from one permitted validator."""

    validator_hotkey: Annotated[
        str, Field(pattern=_SS58_PATTERN, description="Validator's public hotkey.")
    ]
    software_version: str
    protocol_version: Annotated[int, Field(ge=1)]
    state: ValidatorRuntimeState
    assigned_agent_id: UUID | None = None
    assigned_agent_name: str | None = None
    reported_agent_id: UUID | None = None
    assignment_state: ValidatorAssignmentState
    active_agent_id: UUID | None = None
    active_benchmark: PublicBenchmarkProgress | None = None
    first_seen_at: datetime | None = None
    reported_at: datetime
    seen_at: datetime
    online: bool
    availability: FleetAvailability
    health: FleetHealth
    system_metrics: PublicSystemMetrics | None = None
    capabilities: ValidatorCapabilities | None = None
    stack: ValidatorStackIdentity | None = None
    stack_health: ValidatorStackHealth | None = None


class PublicValidatorHeartbeatsResponse(BaseModel):
    """Public view of validators that run heartbeat-capable software."""

    generated_at: datetime
    online_window_seconds: Annotated[int, Field(ge=1)]
    stale_window_seconds: Annotated[int, Field(ge=1)]
    reported_count: Annotated[int, Field(ge=0)]
    online_count: Annotated[int, Field(ge=0)]
    validators: list[PublicValidatorHeartbeat] = Field(default_factory=list)


class PublicOperationsResponse(BaseModel):
    """One cacheable operations snapshot shared by pipeline and fleet views."""

    generated_at: datetime
    active_bench_version: Annotated[int, Field(ge=1)]
    desired_bench_version: Annotated[int, Field(ge=1)]
    benchmark_rollout_status: Literal[
        "inactive", "collecting", "blocked_ineligible", "activated"
    ]
    activity: PublicActivityResponse
    validators: PublicValidatorHeartbeatsResponse


class PublicValidatorName(BaseModel):
    """Optional public chain metadata paired with a validator identity."""

    validator_hotkey: Annotated[
        str, Field(pattern=_SS58_PATTERN, description="Validator's public hotkey.")
    ]
    display_name: Annotated[str, Field(min_length=1, max_length=80)] | None = None
    stake_weight: Annotated[float, Field(ge=0)] | None = None


class PublicValidatorNamesResponse(BaseModel):
    """Non-blocking snapshot of optional Taostats display-name decoration."""

    generated_at: datetime
    source: Literal["taostats"] = "taostats"
    status: Literal["disabled", "fresh", "stale", "unavailable"]
    refreshed_at: datetime | None = None
    validators: list[PublicValidatorName] = Field(default_factory=list)


class PublicScreenerProgress(BaseModel):
    """Allowlisted stage and signed start time for one current job."""

    stage: ScreenerProgressStage
    started_at: datetime


class PublicScreenerHeartbeat(BaseModel):
    """Latest public-safe report from one authenticated screener instance."""

    instance_id: Annotated[
        str,
        Field(
            description="Per-worker instance id (fleet shares one hotkey).",
        ),
    ]
    screener_hotkey: Annotated[
        str, Field(pattern=_SS58_PATTERN, description="Screener's public hotkey.")
    ]
    software_version: str
    protocol_version: Annotated[int, Field(ge=1)]
    policy_version: Annotated[int, Field(ge=1)]
    state: ScreenerRuntimeState
    active_agent_id: UUID | None = None
    active_agent_name: str | None = None
    screening_progress: PublicScreenerProgress | None = None
    first_seen_at: datetime | None = None
    reported_at: datetime
    seen_at: datetime
    online: bool
    availability: FleetAvailability
    health: FleetHealth
    system_metrics: PublicSystemMetrics | None = None


class PublicScreenerHeartbeatsResponse(BaseModel):
    """Public view of authenticated platform-operated screeners."""

    generated_at: datetime
    online_window_seconds: Annotated[int, Field(ge=1)]
    stale_window_seconds: Annotated[int, Field(ge=1)]
    reported_count: Annotated[int, Field(ge=0)]
    online_count: Annotated[int, Field(ge=0)]
    screeners: list[PublicScreenerHeartbeat] = Field(default_factory=list)


class BenchHarnessConfig(BaseModel):
    """How the harness model is frozen for the current benchmark version."""

    locked: bool = Field(description="Every harness is scored against ONE model.")
    canonical_id: str = Field(
        description="Canonical locked model id (docs + score reports)."
    )
    serving: str = Field(
        description="The exact served artifact (fleet standard: Chutes TEE)."
    )
    thinking: bool = Field(
        description="Locked hybrid-reasoning mode; false fleet-wide."
    )
    enforcement: str = Field(description="How the lock is enforced around the sandbox.")


class BenchGradingConfig(BaseModel):
    """How runs are graded."""

    judge_free: bool = Field(description="No LLM judge anywhere in scoring.")
    grader: str = Field(description="The public grader module.")
    description: str = Field(description="One-line grading summary.")


class BenchDatasetConfig(BaseModel):
    """How datasets are generated and pinned."""

    generator: str = Field(description="The public generator module.")
    seed_derivation: str = Field(description="Where a scored run's seed comes from.")
    reproduce: str = Field(
        description="The command reproducing any scored dataset byte-for-byte."
    )


class PublicBenchConfigResponse(BaseModel):
    """The current benchmark setup (``GET /public/bench/config``).

    Everything here is a consensus parameter or a public fact: the frozen
    harness model, the judge-free grading rules, and the seed/dataset
    reproducibility story. Values change only with coordinated fleet bumps
    (and a bench_version change when scoring-affecting).
    """

    bench_version: int
    harness: BenchHarnessConfig
    grading: BenchGradingConfig
    dataset: BenchDatasetConfig
    public_mirror_url_template: str | None = Field(
        description=(
            "Anonymous-read URL template for finalized run records "
            "(dataset pin + k=3 signed scores), or null when mirroring is off."
        )
    )
    public_transcript_url_template: str | None = Field(
        default=None,
        description=(
            "Anonymous-read URL template for content-addressed run transcripts "
            "(``{sha256}`` = a score's signature-bound ``transcript_sha256``), "
            "or null when mirroring is off."
        ),
    )
    ledger_path: str = Field(description="The self-verifying signed score ledger.")
    generated_at: datetime
