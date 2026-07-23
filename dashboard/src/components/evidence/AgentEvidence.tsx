import { For, Show } from "solid-js";
import type { JSX } from "solid-js";

import type { PipelinePayload } from "../../types";
import { athDate, fx, relTime, shortKey } from "../../lib/format";
import { StatusChip } from "../ui/StatusChip";
import { DisputeForm } from "./DisputeForm";

export function AgentEvidence(props: {
  agentId: string;
  detail: PipelinePayload;
  refresh: () => void;
}): JSX.Element {
  return (
    <div class="pipeline-detail">
      <dl class="detail-grid">
        <div>
          <dt>Status</dt>
          <dd>
            <StatusChip status={props.detail.status} />
          </dd>
        </div>
        <div>
          <dt>Canonical quorum</dt>
          <dd>
            {props.detail.score_count ?? 0}/{props.detail.quorum ?? 3}
          </dd>
        </div>
        <div>
          <dt>Active benchmark</dt>
          <dd>v{props.detail.active_bench_version ?? "—"}</dd>
        </div>
        <div>
          <dt>Score floor</dt>
          <dd>{props.detail.score_floor == null ? "—" : fx(props.detail.score_floor)}</dd>
        </div>
      </dl>
      <section class="pipeline-section">
        <div class="pipeline-section-heading">
          <h4>Current work</h4>
        </div>
        <For
          each={props.detail.active_benchmarks || []}
          fallback={<p class="pipeline-detail-state">No benchmark is actively running.</p>}
        >
          {(run) => (
            <article class="benchmark-progress">
              <strong>{run.stage?.replaceAll("_", " ") || "Preparing"}</strong>
              <span>
                v{run.bench_version ?? "—"} ·{" "}
                {run.percent == null ? "progress pending" : `${run.percent}%`} · started{" "}
                {relTime(run.started_at)}
              </span>
              <progress value={run.percent ?? 0} max="100" />
            </article>
          )}
        </For>
      </section>
      <section class="pipeline-section">
        <div class="pipeline-section-heading">
          <h4>Accepted validator scores</h4>
        </div>
        <For
          each={props.detail.provisional_scores || []}
          fallback={<p class="pipeline-detail-state">No validator score has been accepted yet.</p>}
        >
          {(score, index) => (
            <details class="accepted-score">
              <summary>
                <span class="accepted-score-value">{fx(score.composite)}</span>
                <span>
                  Score {index() + 1} · Bench v{score.bench_version ?? "—"} ·{" "}
                  {relTime(score.accepted_at)}
                </span>
              </summary>
              <dl>
                <div>
                  <dt>Seed</dt>
                  <dd>
                    <code>{score.seed ?? "—"}</code>
                  </dd>
                </div>
                <div>
                  <dt>Dataset digest</dt>
                  <dd>
                    <code>{score.dataset_sha256 || "Not published"}</code>
                  </dd>
                </div>
                <div>
                  <dt>Transcript digest</dt>
                  <dd>
                    <code>{score.transcript_sha256 || "Not published"}</code>
                  </dd>
                </div>
              </dl>
              <Show when={score.reproduction_command}>
                <pre>
                  <code>{score.reproduction_command}</code>
                </pre>
              </Show>
              <Show when={score.verification_command}>
                <pre>
                  <code>{score.verification_command}</code>
                </pre>
              </Show>
              <p>{score.case_results?.length ?? 0} redacted case results published.</p>
            </details>
          )}
        </For>
      </section>
      <Show when={(props.detail.confirmation_scores || []).length}>
        <section class="pipeline-section">
          <div class="pipeline-section-heading">
            <h4>Continual top-five retests</h4>
          </div>
          <For each={props.detail.confirmation_scores || []}>
            {(score) => (
              <div class="accepted-score">
                <strong>{fx(score.composite)}</strong>
                <span>
                  Shared seed {score.seed} · Validator {shortKey(score.validator_hotkey)} ·{" "}
                  {relTime(score.accepted_at)}
                </span>
              </div>
            )}
          </For>
        </section>
      </Show>
      <section class="pipeline-section">
        <div class="pipeline-section-heading">
          <h4>Screening history</h4>
        </div>
        <For
          each={props.detail.screening_attempts || []}
          fallback={<p class="pipeline-detail-state">No screening attempt recorded.</p>}
        >
          {(attempt) => (
            <article class="attempt-row">
              <StatusChip status={attempt.quarantine_resolution || attempt.status} />
              <div>
                <strong>Policy v{attempt.policy_version ?? "—"}</strong>
                <span>
                  {attempt.reason ||
                    attempt.review_finding?.summary ||
                    "No public-safe reason recorded."}
                </span>
                <small>
                  {athDate(attempt.started_at)} → {athDate(attempt.finished_at)}
                </small>
              </div>
            </article>
          )}
        </For>
      </section>
      <section class="pipeline-section">
        <div class="pipeline-section-heading">
          <h4>Validation history</h4>
        </div>
        <For
          each={props.detail.validation_attempts || []}
          fallback={<p class="pipeline-detail-state">No validator assignment recorded.</p>}
        >
          {(attempt) => (
            <article class="attempt-row">
              <StatusChip status={attempt.failure_reason || attempt.status} />
              <div>
                <strong>
                  {shortKey(attempt.validator_hotkey) || "Validator pending"} · v
                  {attempt.bench_version ?? "—"}
                </strong>
                <span>{attempt.purpose?.replaceAll("_", " ") || "Canonical validation"}</span>
                <small>
                  Issued {athDate(attempt.issued_at)}
                  <Show when={attempt.failed_at}> · failed {athDate(attempt.failed_at)}</Show>
                </small>
              </div>
            </article>
          )}
        </For>
      </section>
      <Show when={props.detail.dispute}>
        <section class="pipeline-section screening-dispute">
          <div class="pipeline-section-heading">
            <h4>Screening dispute</h4>
          </div>
          <StatusChip status={props.detail.dispute?.resolution || props.detail.dispute?.status} />
          <p>Submitted {athDate(props.detail.dispute?.submitted_at)}.</p>
        </section>
      </Show>
      <Show when={props.detail.status === "rejected" && !props.detail.dispute}>
        <DisputeForm agentId={props.agentId} onSubmitted={props.refresh} />
      </Show>
    </div>
  );
}
