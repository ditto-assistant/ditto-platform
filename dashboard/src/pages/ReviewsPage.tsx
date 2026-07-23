import { For, Show } from "solid-js";
import type { JSX } from "solid-js";

import { EntityButton } from "../components/ui/EntityButton";
import { EmptyState, ErrorState } from "../components/ui/States";
import type { ResourceState } from "../data/useEndpoint";
import { athDate, fx, relTime, shortKey } from "../lib/format";
import type { AthReview, AthSnapshot } from "../types";

export function ReviewsPage(props: { resource: ResourceState<AthSnapshot> }): JSX.Element {
  const entries = () => props.resource.data()?.entries || [];
  const oldest = () =>
    entries()
      .slice()
      .sort(
        (a, b) => Date.parse(a.review_opened_at || "") - Date.parse(b.review_opened_at || ""),
      )[0];
  return (
    <>
      <section class="ath-explainer">
        <div class="review-intro">
          <strong>
            High-score review protects the public leaderboard without erasing evidence.
          </strong>
          <span>
            The original score remains preserved while source and run evidence receive an audited
            decision.
          </span>
        </div>
        <div class="ath-outcomes">
          <article>
            <strong>Clear</strong>
            <p>Evidence supports the score. The hold is removed and normal eligibility resumes.</p>
          </article>
          <article>
            <strong>Reject</strong>
            <p>
              Evidence shows the score is not eligible. The decision and reason remain auditable.
            </p>
          </article>
          <article>
            <strong>Rerun</strong>
            <p>
              The evidence is inconclusive or operationally compromised. A controlled re-evaluation
              is requested.
            </p>
          </article>
        </div>
      </section>
      <section>
        <div class="section-head">
          <div>
            <h2>Active public holds</h2>
            <p>Scores remain visible but do not influence emissions until review closes.</p>
          </div>
          <span class="hint">Snapshot {relTime(props.resource.data()?.generated_at)}</span>
        </div>
        <div class="ath-metrics">
          <div>
            <span>Active reviews</span>
            <strong>{props.resource.data()?.total ?? entries().length}</strong>
          </div>
          <div>
            <span>Oldest open</span>
            <strong>{oldest() ? relTime(oldest()?.review_opened_at) : "—"}</strong>
          </div>
          <div>
            <span>Preserved scores</span>
            <strong>{entries().filter((entry) => entry.preserved_composite != null).length}</strong>
          </div>
        </div>
        <Show when={props.resource.error()}>
          <ErrorState error={props.resource.error()} retry={props.resource.refresh} />
        </Show>
        <div class="review-list">
          <For
            each={entries()}
            fallback={
              <EmptyState
                title="The review queue is clear"
                detail="No high-scoring submissions are currently held for ATH review."
              />
            }
          >
            {(entry: AthReview) => (
              <article class="review-row">
                <div>
                  <EntityButton kind="agent" id={entry.agent_id} class="row-title">
                    {entry.name || "Unnamed agent"}
                  </EntityButton>
                  <span>{entry.review_reason || "Review evidence is being prepared."}</span>
                  <small>
                    Submission {shortKey(entry.agent_id)} · Miner {shortKey(entry.miner_hotkey)} ·
                    submitted {athDate(entry.submitted_at)}
                  </small>
                </div>
                <div>
                  <strong>
                    {entry.preserved_composite == null ? "—" : fx(entry.preserved_composite)}
                  </strong>
                  <small>
                    {entry.score_count ?? 0}/{entry.quorum ?? 3} scores preserved
                  </small>
                  <small>Opened {athDate(entry.review_opened_at)}</small>
                </div>
              </article>
            )}
          </For>
        </div>
      </section>
    </>
  );
}
