import { Show, createResource } from "solid-js";
import type { JSX } from "solid-js";

import { getJSON } from "../lib/api";
import { fx, shortKey } from "../lib/format";
import { closeEntityRoute, entityRoute } from "../stores/routeStore";
import type { LeaderboardPayload, OperationsPayload, PipelinePayload } from "../types";
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
  const agentPipeline = () =>
    entityRoute()?.kind === "agent" ? (payload() as PipelinePayload | undefined) : undefined;
  const minerEntry = () =>
    entityRoute()?.kind === "miner"
      ? (payload() as LeaderboardPayload | undefined)?.entries?.find(
          (entry) => entry.miner_hotkey === entityRoute()?.id,
        )
      : undefined;
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
                <dl class="detail-grid">
                  <div>
                    <dt>Status</dt>
                    <dd>
                      <StatusChip status={detail().status} />
                    </dd>
                  </div>
                  <div>
                    <dt>Score quorum</dt>
                    <dd>
                      {detail().score_count ?? 0}/{detail().quorum ?? 3}
                    </dd>
                  </div>
                  <div>
                    <dt>Bench version</dt>
                    <dd>v{detail().active_bench_version ?? "—"}</dd>
                  </div>
                  <div>
                    <dt>Attempts</dt>
                    <dd>
                      {detail().validation_attempts?.length ?? 0} validation ·{" "}
                      {detail().screening_attempts?.length ?? 0} screening
                    </dd>
                  </div>
                </dl>
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
