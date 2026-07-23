import { For, Show } from "solid-js";
import type { JSX } from "solid-js";

import { EntityButton } from "../components/ui/EntityButton";
import { EmptyState, ErrorState } from "../components/ui/States";
import type { ResourceState } from "../data/useEndpoint";
import { athDate, fx } from "../lib/format";
import type { AthReview, AthSnapshot } from "../types";

export function ReviewsPage(props: { resource: ResourceState<AthSnapshot> }): JSX.Element {
  const entries = () => props.resource.data()?.entries || [];
  return (
    <section>
      <div class="review-intro">
        <strong>Scores stay preserved while review is open.</strong>
        <span>
          These submissions are excluded from emissions until an audited decision is recorded.
        </span>
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
              </div>
              <div>
                <strong>
                  {entry.preserved_composite == null ? "—" : fx(entry.preserved_composite)}
                </strong>
                <small>Opened {athDate(entry.review_opened_at)}</small>
              </div>
            </article>
          )}
        </For>
      </div>
    </section>
  );
}
