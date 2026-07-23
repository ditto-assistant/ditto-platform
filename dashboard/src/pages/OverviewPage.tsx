import { For, Show, createMemo, createSignal } from "solid-js";
import type { JSX } from "solid-js";

import { EmissionsPanel } from "../components/overview/EmissionsPanel";
import { RolloutPanel } from "../components/overview/RolloutPanel";
import { EntityButton } from "../components/ui/EntityButton";
import { Pager } from "../components/ui/Pager";
import { Sparkline } from "../components/ui/Sparkline";
import { EmptyState, ErrorState, LoadingRows } from "../components/ui/States";
import type { ResourceState } from "../data/useEndpoint";
import { useEndpoint } from "../data/useEndpoint";
import { fmtMs, fx, median, relTime, shortKey, shortModel } from "../lib/format";
import type { HealthPayload, LeaderboardEntry, LeaderboardPayload, RolloutState } from "../types";

type BoardTab = "all" | "scored" | "provisional";
type SortKey = "rank" | "name" | "composite" | "tool" | "memory" | "latency" | "first_seen";

function isFinalized(entry: LeaderboardEntry): boolean {
  return entry.finalized !== false;
}

export function OverviewPage(props: {
  health: ResourceState<HealthPayload>;
  rollout: ResourceState<RolloutState>;
}): JSX.Element {
  const [version, setVersion] = createSignal<string>("current");
  const [tab, setTab] = createSignal<BoardTab>("scored");
  const [sort, setSort] = createSignal<SortKey>("rank");
  const [direction, setDirection] = createSignal(1);
  const [page, setPage] = createSignal(1);
  const leaderboard = useEndpoint<LeaderboardPayload>(() =>
    version() === "current"
      ? "/public/leaderboard"
      : `/public/leaderboard?bench_version=${version()}`,
  );
  const entries = () => leaderboard.data()?.entries || [];
  const visible = createMemo(() => {
    const source = entries().filter((entry) =>
      tab() === "all" ? true : tab() === "scored" ? isFinalized(entry) : !isFinalized(entry),
    );
    const key = sort();
    return source.slice().sort((a, b) => {
      const value = (entry: LeaderboardEntry): string | number | null | undefined => {
        if (key === "name") return (entry.agent_name || entry.miner_hotkey || "").toLowerCase();
        if (key === "first_seen") return entry.first_seen ? Date.parse(entry.first_seen) : null;
        if (key === "latency") return entry.median_ms;
        return key === "rank"
          ? entry.rank
          : entry[key === "tool" ? "tool_mean" : key === "memory" ? "memory_mean" : "composite"];
      };
      const av = value(a);
      const bv = value(b);
      if (av == null || bv == null) return av == null ? 1 : -1;
      return (av < bv ? -1 : av > bv ? 1 : 0) * direction();
    });
  });
  const pageSize = 25;
  const pages = () => Math.max(1, Math.ceil(visible().length / pageSize));
  const rows = () => visible().slice((page() - 1) * pageSize, page() * pageSize);
  const scored = () => entries().filter(isFinalized);
  const provisional = () => entries().filter((entry) => !isFinalized(entry));
  const composites = () =>
    scored()
      .map((entry) => entry.composite)
      .filter(Number.isFinite);
  const top = () => Math.max(...composites(), 0);
  const middle = () => median(composites());
  const benchmarkVersions = () => leaderboard.data()?.available_bench_versions || [];
  const champion = () => leaderboard.data()?.emissions?.champion_agent_id;

  const changeSort = (key: SortKey) => {
    if (sort() === key) setDirection((value) => -value);
    else {
      setSort(key);
      setDirection(key === "rank" || key === "name" ? 1 : -1);
    }
    setPage(1);
  };

  return (
    <>
      <RolloutPanel rollout={props.rollout.data()} />
      <section aria-labelledby="snapshot-title">
        <div class="section-head">
          <h2 id="snapshot-title">Subnet snapshot</h2>
          <span class="hint">Live public telemetry</span>
        </div>
        <div class="cards metric-cards">
          <article class="card">
            <span>Registered miners</span>
            <strong>{props.health.data()?.miners ?? "—"}</strong>
          </article>
          <article class="card">
            <span>Scored miners</span>
            <strong>{props.health.data()?.scored_miners ?? "—"}</strong>
          </article>
          <article class="card">
            <span>Scored agents</span>
            <strong>{props.health.data()?.scored_agents ?? "—"}</strong>
          </article>
          <article class="card">
            <span>Top composite</span>
            <strong>{top() ? fx(top()) : "—"}</strong>
          </article>
          <article class="card">
            <span>Median composite</span>
            <strong>{composites().length ? fx(middle()) : "—"}</strong>
          </article>
          <article class="card">
            <span>Scores · 24h</span>
            <strong>{props.health.data()?.scores_24h ?? "—"}</strong>
          </article>
          <article class="card">
            <span>Total scores</span>
            <strong>{props.health.data()?.total_scores ?? "—"}</strong>
          </article>
          <article class="card">
            <span>Median latency</span>
            <strong>
              {props.health.data()?.avg_latency_ms == null
                ? "—"
                : fmtMs(props.health.data()!.avg_latency_ms!)}
            </strong>
          </article>
          <article class="card">
            <span>Current benchmark</span>
            <strong>
              v
              {leaderboard.data()?.current_bench_version ??
                leaderboard.data()?.active_bench_version ??
                "—"}
            </strong>
          </article>
          <article class="card">
            <span>Last score</span>
            <strong>{relTime(props.health.data()?.last_scored_at)}</strong>
          </article>
        </div>
      </section>
      <EmissionsPanel emissions={leaderboard.data()?.emissions} />
      <section aria-labelledby="leaderboard-title">
        <div class="section-head">
          <div>
            <h2 id="leaderboard-title">Leaderboard</h2>
            <p>
              Finalized active-benchmark scores rank first. Provisional runs remain visible for
              transparency.
            </p>
          </div>
          <span class="hint">{entries().length} runs</span>
        </div>
        <div class="leaderboard-version-switch">
          <div class="activity-filter-list" aria-label="Benchmark version">
            <button
              class="activity-filter"
              aria-pressed={version() === "current"}
              onClick={() => {
                setVersion("current");
                setPage(1);
              }}
            >
              Current rollout
            </button>
            <For each={benchmarkVersions()}>
              {(item) => (
                <button
                  class="activity-filter"
                  aria-pressed={version() === String(item)}
                  onClick={() => {
                    setVersion(String(item));
                    setPage(1);
                  }}
                >
                  Bench v{item}
                </button>
              )}
            </For>
          </div>
          <div class="activity-filter-list" aria-label="Leaderboard state">
            <button
              class="activity-filter"
              aria-pressed={tab() === "all"}
              onClick={() => {
                setTab("all");
                setPage(1);
              }}
            >
              All <b>{entries().length}</b>
            </button>
            <button
              class="activity-filter"
              aria-pressed={tab() === "scored"}
              onClick={() => {
                setTab("scored");
                setPage(1);
              }}
            >
              Scored <b>{scored().length}</b>
            </button>
            <button
              class="activity-filter"
              aria-pressed={tab() === "provisional"}
              onClick={() => {
                setTab("provisional");
                setPage(1);
              }}
            >
              Provisional <b>{provisional().length}</b>
            </button>
          </div>
        </div>
        <Show when={leaderboard.error()}>
          <ErrorState error={leaderboard.error()} retry={leaderboard.refresh} />
        </Show>
        <Show when={!leaderboard.error()}>
          <div class="board" tabindex="0" aria-label="Leaderboard, horizontally scrollable">
            <table>
              <thead>
                <tr>
                  <th>
                    <button class="sort-button" onClick={() => changeSort("rank")}>
                      Rank
                    </button>
                  </th>
                  <th>
                    <button class="sort-button" onClick={() => changeSort("name")}>
                      Agent / miner
                    </button>
                  </th>
                  <th>Emissions</th>
                  <th>Model</th>
                  <th class="num">
                    <button class="sort-button" onClick={() => changeSort("composite")}>
                      Composite
                    </button>
                  </th>
                  <th class="num">
                    <button class="sort-button" onClick={() => changeSort("tool")}>
                      Tool
                    </button>
                  </th>
                  <th class="num">
                    <button class="sort-button" onClick={() => changeSort("memory")}>
                      Memory
                    </button>
                  </th>
                  <th class="num">
                    <button class="sort-button" onClick={() => changeSort("latency")}>
                      Latency
                    </button>
                  </th>
                  <th>
                    <button class="sort-button" onClick={() => changeSort("first_seen")}>
                      First seen
                    </button>
                  </th>
                </tr>
              </thead>
              <tbody>
                <Show when={leaderboard.loading()}>
                  <LoadingRows columns={9} />
                </Show>
                <For each={rows()}>
                  {(entry) => (
                    <tr classList={{ champion: entry.agent_id === champion() }}>
                      <td>
                        <span class={`rank r${entry.rank || ""}`}>
                          {isFinalized(entry) ? (entry.rank ?? "—") : `P${entry.rank ?? "—"}`}
                        </span>
                      </td>
                      <td>
                        <div class="winner-identity">
                          <EntityButton kind="agent" id={entry.agent_id} class="winner-name">
                            {entry.agent_name || "Unnamed agent"}
                          </EntityButton>
                          <span>v{entry.agent_version ?? "?"}</span>
                          <EntityButton kind="miner" id={entry.miner_hotkey} class="winner-miner">
                            UID {entry.miner_uid ?? "—"} · {shortKey(entry.miner_hotkey)}
                          </EntityButton>
                          <Show when={!isFinalized(entry)}>
                            <span class="quorum-badge">
                              {entry.score_count ?? 0} of {entry.score_quorum ?? 3} · provisional
                            </span>
                          </Show>
                        </div>
                      </td>
                      <td>
                        <Show when={entry["_emission"]} fallback={<span class="muted">—</span>}>
                          {(emission) => (
                            <span class={`emission-badge ${emission().role}`}>
                              {emission().role} ·{" "}
                              {emission().share_of_miner_pool == null
                                ? "—"
                                : `${(emission().share_of_miner_pool! * 100).toFixed(1)}%`}
                            </span>
                          )}
                        </Show>
                      </td>
                      <td>
                        <span>{shortModel(entry.models?.harness || "Not reported")}</span>
                      </td>
                      <td class="num score-cell">
                        <strong>{fx(entry.settled_composite ?? entry.composite)}</strong>
                        <Sparkline
                          values={entry.history}
                          label={`${entry.agent_name || "Agent"} composite trend`}
                        />
                      </td>
                      <td class="num">{fx(entry.tool_mean)}</td>
                      <td class="num">{fx(entry.memory_mean)}</td>
                      <td class="num">{entry.median_ms == null ? "—" : fmtMs(entry.median_ms)}</td>
                      <td>{relTime(entry.first_seen)}</td>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </div>
          <Pager page={page()} pages={pages()} total={visible().length} onPage={setPage} />
          <Show when={!leaderboard.loading() && visible().length === 0}>
            <EmptyState
              title="No runs in this view"
              detail="Choose another benchmark version or status filter."
            />
          </Show>
        </Show>
      </section>
    </>
  );
}
