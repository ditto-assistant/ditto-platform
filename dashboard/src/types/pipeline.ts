// Submission activity, agent pipeline detail, ATH reviews, and run-telemetry
// wire shapes (/public/activity, /public/agent/{id}/pipeline, and the
// digest-verified transcript telemetry sidecar).

import type { CaseResult } from "./leaderboard";

// ── Activity / submissions (/public/activity) ────────────────

export interface ActivityEntry {
  agent_id?: string;
  name?: string | null;
  version?: number | null;
  miner_hotkey?: string;
  /** Submission status slug, e.g. "waiting_screening" | "scored" | "rejected". */
  status?: string;
  submitted_at?: string;
  score_count?: number | null;
  quorum?: number | null;
  score_floor?: number | null;
  review_reason?: string | null;
  screening_reason?: string | null;
  duplicate_of?: string | null;
  duplicate_name?: string | null;
  duplicate_version?: number | null;
  artifact_sha256?: string | null;
  screening_policy_version?: number | null;
  required_screening_policy_version?: number | null;
}

interface ActivityPayloadBase<E> {
  entries?: E[];
  status_counts?: Record<string, number>;
  page?: number;
  total_pages?: number;
  total?: number;
  count?: number;
  generated_at?: string;
}

export type ActivityPayload = ActivityPayloadBase<ActivityEntry>;

/** One held high-score submission from the ATH review queue
 * (/public/activity?review=ath&status=under_review). */
export interface AthReview extends ActivityEntry {
  preserved_composite?: number | null;
  review_opened_at?: string | null;
}

/** The stitched multi-page ATH review snapshot. */
export type AthSnapshot = ActivityPayloadBase<AthReview>;

// ── Operations pipeline feed + agent pipeline detail ─────────

/** Live progress of one benchmark run (validator slot or pipeline card). */
export interface BenchmarkProgress {
  /** "preparing" | "building_harness" | … | "failed_retrying". */
  stage?: string | null;
  percent?: number | null;
  stalled?: boolean;
  started_at?: string | null;
  completed_checks?: number | null;
  total_checks?: number | null;
  bench_version?: number | null;
  agent_id?: string | null;
  agent_name?: string | null;
  slot_id?: string | null;
}

/** One row of the operations activity feed (superset of a submission row). */
export interface PipelineEntry extends ActivityEntry {
  last_scored_at?: string | null;
  validator_queue_rank?: number | null;
  /** "exhausted" | "cooling_down" | others advance on their own. */
  retry_state?: string | null;
  retry_after?: string | null;
  provisional_composite?: number | null;
  active_benchmarks?: BenchmarkProgress[];
  active_bench_version?: number | null;
}

/** The `activity` slice of the operations payload (and what the pipeline
 * board re-renders from cache). */
export interface PipelineFeed {
  entries?: PipelineEntry[];
}

export interface ScreeningReviewLocation {
  path?: string;
  line?: number | string;
  category?: string;
}

export interface ScreeningReviewFinding {
  summary?: string;
  confidence?: number | null;
  categories?: string[];
  locations?: ScreeningReviewLocation[];
  reviewer_revision?: string;
}

export interface ScreeningReviewEvidence {
  code?: string;
  summary?: string;
}

export interface ScreeningAttempt {
  /** "running" | "passed" | "rejected" | "failed" | "expired" | "quarantined" | … */
  status?: string;
  policy_version?: number | null;
  screener_hotkey?: string;
  reason?: string | null;
  deadline?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  /** "release" | "rescreen" | "reject"; overrides status display when quarantined. */
  quarantine_resolution?: string | null;
  quarantine_resolution_reason?: string | null;
  quarantine_resolved_at?: string | null;
  review_finding?: ScreeningReviewFinding | null;
  review_evidence?: ScreeningReviewEvidence[] | null;
}

export interface ValidationAttempt {
  /** "issued" | "expired" | "scored" | … */
  status?: string;
  actively_running?: boolean;
  /** "canonical_quorum" | "continual_retest" | "legacy_unclassified". */
  purpose?: string;
  validator_hotkey?: string;
  /** "sandbox_oom" | "infrastructure" | "scoring_error". */
  failure_reason?: string | null;
  deadline?: string | null;
  benchmark_progress?: BenchmarkProgress | null;
  bench_version?: number | null;
  issued_at?: string | null;
  failed_at?: string | null;
}

/** An accepted (provisional/quorum) validator score with its reproducibility
 * evidence. */
export interface AcceptedScore {
  composite: number;
  bench_version?: number | null;
  accepted_at?: string | null;
  seed?: string | number;
  /** "on_chain" | "validator_local" | anything else reads as legacy fallback. */
  seed_source?: string | null;
  reproduction_command?: string | null;
  verification_command?: string | null;
  dataset_sha256?: string | null;
  transcript_sha256?: string | null;
  case_results?: CaseResult[];
}

/** A shared-seed continual top-five retest result. */
export interface ConfirmationScore {
  composite: number;
  bench_version?: number | null;
  seed?: string | number;
  validator_hotkey?: string;
  accepted_at?: string | null;
}

export interface Dispute {
  /** "pending" or resolved. */
  status?: string;
  /** "release" means accepted; anything else reads as upheld. */
  resolution?: string | null;
  submitted_at?: string | null;
}

/** /public/agent/{id}/pipeline — the drawer's full history. */
export interface PipelinePayload {
  status?: string;
  quorum?: number | null;
  score_count?: number | null;
  active_bench_version?: number | null;
  score_floor?: number | null;
  provisional_scores?: AcceptedScore[];
  confirmation_scores?: ConfirmationScore[];
  validation_attempts?: ValidationAttempt[];
  screening_attempts?: ScreeningAttempt[];
  dispute?: Dispute | null;
  active_benchmarks?: BenchmarkProgress[];
}

// ── Run telemetry (digest-verified transcript sidecar) ───────

export interface TelemetryAttempt {
  attempt?: number | null;
  outcome?: string | null;
  duration_ms?: number | null;
  http_status?: number | null;
}

export interface TelemetryCaseExecution {
  terminal_outcome?: string | null;
  total_duration_ms?: number | null;
  attempts?: TelemetryAttempt[] | null;
}

export interface TelemetryCase {
  position?: number | string | null;
  execution?: TelemetryCaseExecution | null;
}

export interface TelemetryExecution {
  cases?: number | null;
  succeeded?: number | null;
  median_duration_ms?: number | null;
  p95_duration_ms?: number | null;
  max_duration_ms?: number | null;
  retried?: number | null;
  timed_out?: number | null;
  cancelled?: number | null;
  total_attempts?: number | null;
}

export interface TelemetryModelRelay {
  successes?: number | null;
  requests?: number | null;
  retries?: number | null;
  caller_cancellations?: number | null;
  infrastructure_failures?: number | null;
  upstream_attempts?: number | null;
}

export interface RunTelemetry {
  /** Must equal the requested transcript digest or the payload is rejected. */
  source_sha256?: string;
  execution?: TelemetryExecution | null;
  model_relay?: TelemetryModelRelay | null;
  cases?: TelemetryCase[] | null;
}
