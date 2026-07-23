import { Show, createResource } from "solid-js";
import type { JSX } from "solid-js";

import { getJSON } from "../lib/api";
import { fx, shortKey } from "../lib/format";
import { closeEntityRoute, entityRoute } from "../stores/routeStore";
import type { LeaderboardPayload, OperationsPayload, PipelinePayload } from "../types";
import type { FleetEntry } from "../types";
import { AgentEvidence } from "./evidence/AgentEvidence";
import { ErrorState } from "./ui/States";
import { StatusChip } from "./ui/StatusChip";

export function EntityPanel(): JSX.Element {
  const path = () => {
    const current = entityRoute();
    if (!current) return "";
    if (current.kind === "agent") return `/public/agent/${encodeURIComponent(current.id)}/pipeline`;
    if (current.kind === "miner") return "/public/leaderboard";
    return "/public/operations";
  };
  const [payload, { refetch }] = createResource(
    () => entityRoute()?.key || "",
    async () => getJSON<PipelinePayload | LeaderboardPayload | OperationsPayload>(path()),
  );
  const agentPipeline = () => (entityRoute()?.kind === "agent" ? payload() : undefined);
  const minerEntry = () =>
    entityRoute()?.kind === "miner"
      ? (payload() as LeaderboardPayload | undefined)?.entries?.find(
          (entry) => entry.miner_hotkey === entityRoute()?.id,
        )
      : undefined;
  const workerEntry = (): FleetEntry | undefined => {
    const current = entityRoute();
    if (!current || (current.kind !== "validator" && current.kind !== "screener")) return undefined;
    const operations = payload() as OperationsPayload | undefined;
    const entries =
      current.kind === "validator"
        ? operations?.validators?.validators
        : operations?.validators?.screeners;
    return entries?.find(
      (entry) =>
        (current.kind === "validator" ? entry.validator_hotkey : entry.screener_hotkey) ===
        current.id,
    );
  };
  return (
    <Show when={entityRoute()}>
      {(current) => (
        <div
          class="drawer-backdrop"
          onClick={(event) => event.target === event.currentTarget && closeEntityRoute()}
        >
          <aside
            class="entity-drawer"
            role="dialog"
            aria-modal="true"
            aria-labelledby="entity-title"
          >
            <button
              class="drawer-close"
              type="button"
              aria-label="Close details"
              onClick={closeEntityRoute}
            >
              ×
            </button>
            <span class="drawer-kicker">{current().kind}</span>
            <h2 id="entity-title">{shortKey(current().id)}</h2>
            <code>{current().id}</code>
            <Show when={payload.loading}>
              <div class="drawer-loading">Loading public history…</div>
            </Show>
            <Show when={payload.error}>
              <ErrorState error={payload.error} retry={() => void refetch()} />
            </Show>
            <Show when={agentPipeline()}>
              {(detail) => (
                <AgentEvidence
                  agentId={current().id}
                  detail={detail()}
                  refresh={() => void refetch()}
                />
              )}
            </Show>
            <Show when={minerEntry()}>
              {(entry) => (
                <dl class="detail-grid">
                  <div>
                    <dt>Best agent</dt>
                    <dd>{entry().agent_name || entry().agent_id}</dd>
                  </div>
                  <div>
                    <dt>Composite</dt>
                    <dd>{fx(entry().composite)}</dd>
                  </div>
                  <div>
                    <dt>Registration</dt>
                    <dd>
                      {entry().registered === true
                        ? "Registered"
                        : entry().registered === false
                          ? "Inactive"
                          : "Unknown"}
                    </dd>
                  </div>
                </dl>
              )}
            </Show>
            <Show when={workerEntry()}>
              {(entry) => (
                <div class="pipeline-detail">
                  <dl class="detail-grid">
                    <div>
                      <dt>Availability</dt>
                      <dd>
                        <StatusChip status={entry().availability || entry().health} />
                      </dd>
                    </div>
                    <div>
                      <dt>State</dt>
                      <dd>{entry().state || "—"}</dd>
                    </div>
                    <div>
                      <dt>Software</dt>
                      <dd>{entry().software_version || "—"}</dd>
                    </div>
                    <div>
                      <dt>Protocol</dt>
                      <dd>{entry().protocol_version ?? "—"}</dd>
                    </div>
                    <div>
                      <dt>Assignment</dt>
                      <dd>{entry().active_agent_name || entry().assigned_agent_name || "None"}</dd>
                    </div>
                    <div>
                      <dt>Admission</dt>
                      <dd>
                        <StatusChip status={entry().admission || entry().assignment_state} />
                      </dd>
                    </div>
                  </dl>
                  <section class="pipeline-section">
                    <h4>Managed stack</h4>
                    <pre>
                      <code>
                        {JSON.stringify(entry().stack || entry().stack_health || {}, null, 2)}
                      </code>
                    </pre>
                  </section>
                  <section class="pipeline-section">
                    <h4>Capabilities</h4>
                    <pre>
                      <code>{JSON.stringify(entry().capabilities || {}, null, 2)}</code>
                    </pre>
                  </section>
                </div>
              )}
            </Show>
            <Show
              when={
                !payload.loading &&
                !payload.error &&
                current().kind !== "agent" &&
                current().kind !== "miner"
              }
            >
              <p class="drawer-note">Worker state is available on Network operations.</p>
            </Show>
          </aside>
        </div>
      )}
    </Show>
  );
}
