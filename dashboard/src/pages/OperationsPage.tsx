import { For, Show, createMemo, createSignal } from "solid-js";
import type { JSX } from "solid-js";

import { FleetTable } from "../components/operations/FleetTable";
import { PipelineAtlas } from "../components/operations/PipelineAtlas";
import { EmptyState, ErrorState } from "../components/ui/States";
import type { ResourceState } from "../data/useEndpoint";
import { relTime } from "../lib/format";
import type { FleetEntry, OperationsPayload } from "../types";

type FleetKind = "validator" | "screener";

export function OperationsPage(props: { resource: ResourceState<OperationsPayload> }): JSX.Element {
  const [kind, setKind] = createSignal<FleetKind>("validator");
  const validators = () => props.resource.data()?.validators.validators || [];
  const screeners = () => props.resource.data()?.validators.screeners || [];
  const fleet = () => (kind() === "validator" ? validators() : screeners());
  const activity = () => props.resource.data()?.activity?.entries || [];
  const statusCounts = createMemo(() => {
    const result: Record<string, number> = {
      healthy: 0,
      warning: 0,
      stale: 0,
      offline: 0,
      paused: 0,
      unknown: 0,
    };
    fleet().forEach((entry: FleetEntry) => {
      const state = entry.availability || entry.health || "unknown";
      result[state] = (result[state] || 0) + 1;
    });
    return result;
  });
  return (
    <>
      <Show when={props.resource.error()}>
        <ErrorState error={props.resource.error()} retry={props.resource.refresh} />
      </Show>
      <Show when={!props.resource.error()}>
        <section class="operations-snapshot">
          <div>
            <span>Active benchmark</span>
            <strong>v{props.resource.data()?.active_bench_version ?? "—"}</strong>
          </div>
          <div>
            <span>Rollout target</span>
            <strong>v{props.resource.data()?.desired_bench_version ?? "—"}</strong>
          </div>
          <div>
            <span>Pipeline records</span>
            <strong>{activity().length}</strong>
          </div>
          <div>
            <span>Snapshot</span>
            <strong>{relTime(props.resource.data()?.generated_at)}</strong>
          </div>
        </section>
        <section>
          <div class="section-head">
            <div>
              <h2>Live pipeline</h2>
              <p>
                Reconciled view from upload through screening, validator assignment, evaluation, and
                accepted scores.
              </p>
            </div>
            <span class="hint">Refreshes every 30 seconds</span>
          </div>
          <PipelineAtlas entries={activity()} />
        </section>
        <section class="fleet-section">
          <div class="section-head">
            <div>
              <h2>Worker fleet</h2>
              <p>
                Signed heartbeat, assignment, stack identity, capability, and system-health
                evidence.
              </p>
            </div>
            <div class="activity-filter-list">
              <button
                class="activity-filter"
                aria-pressed={kind() === "validator"}
                onClick={() => setKind("validator")}
              >
                Validators {validators().length}
              </button>
              <button
                class="activity-filter"
                aria-pressed={kind() === "screener"}
                onClick={() => setKind("screener")}
              >
                Screeners {screeners().length}
              </button>
            </div>
          </div>
          <div class="fleet-summary">
            <For each={Object.entries(statusCounts())}>
              {([status, count]) => (
                <div>
                  <span>{status}</span>
                  <strong>{count}</strong>
                </div>
              )}
            </For>
          </div>
          <Show
            when={fleet().length}
            fallback={
              <EmptyState
                title={`No ${kind()}s reporting`}
                detail="Workers appear once their signed heartbeat is fresh."
              />
            }
          >
            <FleetTable entries={fleet()} kind={kind()} />
          </Show>
        </section>
      </Show>
    </>
  );
}
