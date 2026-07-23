import { Show, createSignal } from "solid-js";
import type { JSX } from "solid-js";

import { PipelineTable } from "../components/PipelineTable";
import { EmptyState, ErrorState } from "../components/ui/States";
import type { ResourceState } from "../data/useEndpoint";
import { parseHashRoute } from "../lib/router";
import type { ActivityPayload } from "../types";

export function SubmissionsPage(props: { resource: ResourceState<ActivityPayload> }): JSX.Element {
  const entries = () => props.resource.data()?.entries || [];
  const [query, setQuery] = createSignal(parseHashRoute().query.get("q") || "");
  const filtered = () => {
    const needle = query().trim().toLowerCase();
    return needle
      ? entries().filter((entry) =>
          `${entry.name || ""} ${entry.agent_id || ""} ${entry.miner_hotkey || ""}`
            .toLowerCase()
            .includes(needle),
        )
      : entries();
  };
  return (
    <section>
      <div class="toolbar">
        <label class="search-field">
          <span class="visually-hidden">Filter submissions</span>
          <input
            value={query()}
            onInput={(event) => setQuery(event.currentTarget.value)}
            placeholder="Filter by agent, ID, or miner"
          />
        </label>
        <span>{props.resource.data()?.total ?? 0} submissions</span>
      </div>
      <Show when={props.resource.error()}>
        <ErrorState error={props.resource.error()} retry={props.resource.refresh} />
      </Show>
      <Show when={!props.resource.error()}>
        <PipelineTable entries={filtered()} loading={props.resource.loading()} />
      </Show>
      <Show when={!props.resource.loading() && filtered().length === 0}>
        <EmptyState
          title="No matching submissions"
          detail="Clear the filter or wait for a new public submission."
        />
      </Show>
    </section>
  );
}
