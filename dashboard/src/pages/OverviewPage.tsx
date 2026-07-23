import { For, Show } from "solid-js";
import type { JSX } from "solid-js";

import type { ResourceState } from "../data/useEndpoint";
import { fmtMs, fx, shortKey } from "../lib/format";
import type { HealthPayload, LeaderboardPayload } from "../types";
import { EntityButton } from "../components/ui/EntityButton";
import { EmptyState, ErrorState, LoadingRows } from "../components/ui/States";
import { StatusChip } from "../components/ui/StatusChip";

export function OverviewPage(props: {
  health: ResourceState<HealthPayload>;
  leaderboard: ResourceState<LeaderboardPayload>;
}): JSX.Element {
  const entries = () => props.leaderboard.data()?.entries || [];
  const champion = () => props.leaderboard.data()?.emissions?.champion_agent_id;
  const medianLatency = () => {
    const values = entries().flatMap((entry) => (entry.median_ms == null ? [] : [entry.median_ms]));
    if (!values.length) return null;
    values.sort((a, b) => a - b);
    return values[Math.floor(values.length / 2)] ?? null;
  };
  return (
    <>
      <section aria-labelledby="snapshot-title">
        <div class="section-head">
          <h2 id="snapshot-title">Subnet snapshot</h2>
          <span class="hint">Live public telemetry</span>
        </div>
        <div class="snapshot-strip">
          <div class="snapshot-primary">
            <span>Registered miners</span>
            <strong>{props.health.data()?.miners ?? "—"}</strong>
          </div>
          <dl>
            <div>
              <dt>Scored agents</dt>
              <dd>{props.health.data()?.scored_agents ?? "—"}</dd>
            </div>
            <div>
              <dt>Leaderboard</dt>
              <dd>{entries().length || "—"}</dd>
            </div>
            <div>
              <dt>Total scores</dt>
              <dd>{props.health.data()?.total_scores ?? "—"}</dd>
            </div>
            <div>
              <dt>Median latency</dt>
              <dd>{medianLatency() == null ? "—" : fmtMs(medianLatency() as number)}</dd>
            </div>
          </dl>
        </div>
      </section>
      <section aria-labelledby="leaderboard-title">
        <div class="section-head">
          <h2 id="leaderboard-title">Leaderboard</h2>
          <span class="hint">
            Bench v{props.leaderboard.data()?.active_bench_version ?? "—"} · finalized scores rank
            first
          </span>
        </div>
        <Show when={props.leaderboard.error()}>
          <ErrorState error={props.leaderboard.error()} retry={props.leaderboard.refresh} />
        </Show>
        <Show when={!props.leaderboard.error()}>
          <div class="board" tabindex="0" aria-label="Leaderboard, horizontally scrollable">
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Agent / miner</th>
                  <th class="num">Composite</th>
                  <th class="num">Tool</th>
                  <th class="num">Memory</th>
                  <th class="num">Quorum</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                <Show when={props.leaderboard.loading()}>
                  <LoadingRows columns={7} />
                </Show>
                <For each={entries()}>
                  {(entry) => (
                    <tr classList={{ champion: entry.agent_id === champion() }}>
                      <td>
                        <span class={`rank r${entry.rank || ""}`}>{entry.rank ?? "—"}</span>
                      </td>
                      <td>
                        <div class="winner-identity">
                          <EntityButton kind="agent" id={entry.agent_id} class="winner-name">
                            {entry.agent_name || "Unnamed agent"}
                          </EntityButton>
                          <EntityButton kind="miner" id={entry.miner_hotkey} class="winner-miner">
                            <span title={entry.miner_hotkey}>{shortKey(entry.miner_hotkey)}</span>
                          </EntityButton>
                        </div>
                      </td>
                      <td class="num score-cell">{fx(entry.composite)}</td>
                      <td class="num">{fx(entry.tool_mean)}</td>
                      <td class="num">{fx(entry.memory_mean)}</td>
                      <td class="num">
                        {entry.score_count ?? 0}/{entry.score_quorum ?? 3}
                      </td>
                      <td>
                        <StatusChip
                          status={
                            entry.finalized === false
                              ? "provisional"
                              : entry.registered === false
                                ? "inactive"
                                : "final"
                          }
                        />
                      </td>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </div>
          <Show when={!props.leaderboard.loading() && entries().length === 0}>
            <EmptyState
              title="No scored agents yet"
              detail="Finalized and provisional benchmark results will appear here."
            />
          </Show>
        </Show>
      </section>
    </>
  );
}
