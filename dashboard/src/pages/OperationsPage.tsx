import { For, Show } from "solid-js";
import type { JSX } from "solid-js";

import { PipelineTable } from "../components/PipelineTable";
import { EntityButton } from "../components/ui/EntityButton";
import { EmptyState, ErrorState } from "../components/ui/States";
import { StatusChip } from "../components/ui/StatusChip";
import type { ResourceState } from "../data/useEndpoint";
import { pct, shortKey } from "../lib/format";
import type { FleetEntry, OperationsPayload } from "../types";

function FleetList(props: {
  title: string;
  entries: FleetEntry[];
  kind: "validator" | "screener";
}): JSX.Element {
  return (
    <section class="fleet-section">
      <div class="section-head">
        <h2>{props.title}</h2>
        <span class="hint">{props.entries.length} reporting</span>
      </div>
      <div class="fleet-list">
        <For
          each={props.entries}
          fallback={
            <EmptyState
              title={`No ${props.kind}s reporting`}
              detail="Workers appear when their signed heartbeat is fresh."
            />
          }
        >
          {(entry) => {
            const key = () =>
              props.kind === "validator" ? entry.validator_hotkey : entry.screener_hotkey;
            return (
              <article class="fleet-row">
                <div>
                  <EntityButton kind={props.kind} id={key()} class="row-title">
                    <span title={key()}>{shortKey(key()) || "Unknown worker"}</span>
                  </EntityButton>
                  <small>{entry.software_version || "Version not reported"}</small>
                </div>
                <div class="fleet-state">
                  <StatusChip status={entry.availability || entry.state} />
                  <span>
                    {entry.active_agent_name ||
                      entry.active_benchmark?.agent_name ||
                      "No active assignment"}
                  </span>
                </div>
                <div class="fleet-metrics">
                  <span>
                    CPU {entry.system_metrics ? pct(entry.system_metrics.cpu_percent / 100) : "—"}
                  </span>
                  <span>
                    Memory{" "}
                    {entry.system_metrics ? pct(entry.system_metrics.memory_percent / 100) : "—"}
                  </span>
                </div>
              </article>
            );
          }}
        </For>
      </div>
    </section>
  );
}

export function OperationsPage(props: { resource: ResourceState<OperationsPayload> }): JSX.Element {
  const validators = () => props.resource.data()?.validators.validators || [];
  const screeners = () => props.resource.data()?.validators.screeners || [];
  const activity = () => props.resource.data()?.activity?.entries || [];
  return (
    <>
      <Show when={props.resource.error()}>
        <ErrorState error={props.resource.error()} retry={props.resource.refresh} />
      </Show>
      <Show when={!props.resource.error()}>
        <section>
          <div class="section-head">
            <h2>Live pipeline</h2>
            <span class="hint">Refreshes every 30 seconds</span>
          </div>
          <PipelineTable entries={activity()} loading={props.resource.loading()} />
        </section>
        <FleetList title="Validator fleet" entries={validators()} kind="validator" />
        <FleetList title="Screener fleet" entries={screeners()} kind="screener" />
      </Show>
    </>
  );
}
