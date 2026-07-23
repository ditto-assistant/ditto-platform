import {
  For,
  Match,
  Show,
  Switch,
  createEffect,
  createResource,
  createSignal,
  onCleanup,
  onMount,
} from "solid-js";
import type { Accessor, JSX } from "solid-js";

import brandLogo from "./assets/brand-logo.png";
import { getJSON } from "./lib/api";
import { REFRESH_MS, WANDB_URL } from "./lib/config";
import { athDate, fmtMs, fx, pct, relTime, shortKey } from "./lib/format";
import { PAGES, parseHashRoute } from "./lib/router";
import type { EntityKind, PageName } from "./lib/router";
import {
  closeEntityRoute,
  currentPage,
  entityRoute,
  initRouteListeners,
  navigateToPage,
  pushEntityRoute,
  syncFromLocation,
} from "./stores/routeStore";
import type {
  ActivityEntry,
  ActivityPayload,
  AthReview,
  AthSnapshot,
  BenchConfigPayload,
  FleetEntry,
  GlossaryPayload,
  HealthPayload,
  LeaderboardPayload,
  OperationsPayload,
  PipelinePayload,
} from "./types";

type ResourceState<T> = {
  data: Accessor<T | undefined>;
  loading: Accessor<boolean>;
  error: Accessor<unknown>;
  refresh: () => void;
};

const NAV: Array<{ page: PageName; icon: string; description: string }> = [
  { page: "overview", icon: "◫", description: "Scores and emissions" },
  { page: "operations", icon: "⌁", description: "Pipeline and fleet" },
  { page: "submissions", icon: "↳", description: "Screening history" },
  { page: "reviews", icon: "◇", description: "Held high scores" },
  { page: "benchmark", icon: "◎", description: "Scoring contract" },
];

const STATUS_LABELS: Record<string, string> = {
  waiting_screening: "Waiting for screening",
  screening: "Screening",
  waiting_validation: "Waiting for validator",
  validating: "Benchmarking",
  scored: "Scored",
  rejected: "Rejected",
  under_review: "Under review",
  not_queued: "Not queued",
};

function useEndpoint<T>(path: Accessor<string> | string): ResourceState<T> {
  const source = typeof path === "string" ? () => path : path;
  const [data, { refetch }] = createResource(source, (next) => getJSON<T>(next));
  return {
    data,
    loading: () => data.loading,
    error: () => data.error,
    refresh: () => void refetch(),
  };
}

function ErrorState(props: { error: unknown; retry: () => void }): JSX.Element {
  const message = () => (props.error instanceof Error ? props.error.message : "Unknown error");
  return (
    <div class="state-panel error-state" role="alert">
      <strong>Live data is unavailable.</strong>
      <span>{message()}</span>
      <button class="btn" type="button" onClick={props.retry}>
        Try again
      </button>
    </div>
  );
}

function LoadingRows(props: { columns: number }): JSX.Element {
  return (
    <For each={[0, 1, 2, 3]}>
      {() => (
        <tr class="skeleton-row" aria-hidden="true">
          <td colSpan={props.columns}>
            <span />
          </td>
        </tr>
      )}
    </For>
  );
}

function EmptyState(props: { title: string; detail: string }): JSX.Element {
  return (
    <div class="state-panel empty-state">
      <strong>{props.title}</strong>
      <span>{props.detail}</span>
    </div>
  );
}

function StatusChip(props: { status?: string | null }): JSX.Element {
  const status = () => props.status || "unknown";
  return (
    <span class={`status-chip status-${status()}`}>
      {STATUS_LABELS[status()] || status().replaceAll("_", " ")}
    </span>
  );
}

function EntityButton(props: {
  kind: EntityKind;
  id?: string | null;
  children: JSX.Element;
  class?: string;
}): JSX.Element {
  return (
    <Show when={props.id} fallback={<span class={props.class}>{props.children}</span>}>
      {(id) => (
        <button
          type="button"
          class={`entity-button ${props.class || ""}`}
          onClick={() => pushEntityRoute(props.kind, id())}
        >
          {props.children}
        </button>
      )}
    </Show>
  );
}

function Snapshot(props: {
  health: ResourceState<HealthPayload>;
  leaderboard: ResourceState<LeaderboardPayload>;
}): JSX.Element {
  const entries = () => props.leaderboard.data()?.entries || [];
  const medianLatency = () => {
    const values = entries().flatMap((entry) => (entry.median_ms == null ? [] : [entry.median_ms]));
    if (!values.length) return null;
    values.sort((a, b) => a - b);
    return values[Math.floor(values.length / 2)] ?? null;
  };
  return (
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
  );
}

function Leaderboard(props: { resource: ResourceState<LeaderboardPayload> }): JSX.Element {
  const entries = () => props.resource.data()?.entries || [];
  const champion = () => props.resource.data()?.emissions?.champion_agent_id;
  return (
    <section aria-labelledby="leaderboard-title">
      <div class="section-head">
        <h2 id="leaderboard-title">Leaderboard</h2>
        <span class="hint">
          Bench v{props.resource.data()?.active_bench_version ?? "—"} · finalized scores rank first
        </span>
      </div>
      <Show when={props.resource.error()}>
        <ErrorState error={props.resource.error()} retry={props.resource.refresh} />
      </Show>
      <Show when={!props.resource.error()}>
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
              <Show when={props.resource.loading()}>
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
        <Show when={!props.resource.loading() && entries().length === 0}>
          <EmptyState
            title="No scored agents yet"
            detail="Finalized and provisional benchmark results will appear here."
          />
        </Show>
      </Show>
    </section>
  );
}

function OverviewPage(props: {
  health: ResourceState<HealthPayload>;
  leaderboard: ResourceState<LeaderboardPayload>;
}): JSX.Element {
  return (
    <>
      <Snapshot health={props.health} leaderboard={props.leaderboard} />
      <Leaderboard resource={props.leaderboard} />
    </>
  );
}

function PipelineTable(props: {
  entries: PipelinePayload["active_benchmarks"] extends never ? never[] : ActivityEntry[];
  loading?: boolean;
}): JSX.Element {
  return (
    <div class="board" tabindex="0" aria-label="Submissions, horizontally scrollable">
      <table>
        <thead>
          <tr>
            <th>Submission</th>
            <th>Miner</th>
            <th>Status</th>
            <th class="num">Scores</th>
            <th>Submitted</th>
          </tr>
        </thead>
        <tbody>
          <Show when={props.loading}>
            <LoadingRows columns={5} />
          </Show>
          <For each={props.entries}>
            {(entry) => (
              <tr>
                <td>
                  <EntityButton kind="agent" id={entry.agent_id} class="row-title">
                    {entry.name || "Unnamed agent"}
                  </EntityButton>
                </td>
                <td>
                  <EntityButton kind="miner" id={entry.miner_hotkey}>
                    <span title={entry.miner_hotkey}>{shortKey(entry.miner_hotkey)}</span>
                  </EntityButton>
                </td>
                <td>
                  <StatusChip status={entry.status} />
                </td>
                <td class="num">
                  {entry.score_count ?? 0}/{entry.quorum ?? 3}
                </td>
                <td>{relTime(entry.submitted_at)}</td>
              </tr>
            )}
          </For>
        </tbody>
      </table>
    </div>
  );
}

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

function OperationsPage(props: { resource: ResourceState<OperationsPayload> }): JSX.Element {
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

function SubmissionsPage(props: { resource: ResourceState<ActivityPayload> }): JSX.Element {
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

function ReviewsPage(props: { resource: ResourceState<AthSnapshot> }): JSX.Element {
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

function BenchmarkPage(props: {
  glossary: ResourceState<GlossaryPayload>;
  config: ResourceState<BenchConfigPayload>;
}): JSX.Element {
  const categories = () => props.glossary.data()?.categories || [];
  return (
    <>
      <section class="benchmark-contract">
        <div>
          <span>Active public contract</span>
          <strong>DittoBench v{props.config.data()?.bench_version ?? "—"}</strong>
        </div>
        <dl>
          <div>
            <dt>Canonical harness</dt>
            <dd>{props.config.data()?.harness?.canonical_id || "Not reported"}</dd>
          </div>
          <div>
            <dt>Serving model</dt>
            <dd>{props.config.data()?.harness?.serving || "Not reported"}</dd>
          </div>
          <div>
            <dt>Reasoning</dt>
            <dd>{props.config.data()?.harness?.thinking ? "Enabled" : "Disabled"}</dd>
          </div>
        </dl>
      </section>
      <Show when={props.glossary.error() || props.config.error()}>
        <ErrorState
          error={props.glossary.error() || props.config.error()}
          retry={() => {
            props.glossary.refresh();
            props.config.refresh();
          }}
        />
      </Show>
      <section>
        <div class="section-head">
          <h2>What the benchmark measures</h2>
          <span class="hint">{categories().length} public categories</span>
        </div>
        <div class="category-list">
          <For each={categories()}>
            {(category) => (
              <article>
                <div>
                  <StatusChip status={category.kind} />
                  <h3>{category.label || category.key}</h3>
                </div>
                <p>{category.purpose}</p>
                <Show when={category.example}>
                  <blockquote>{category.example}</blockquote>
                </Show>
              </article>
            )}
          </For>
        </div>
      </section>
    </>
  );
}

function EntityPanel(): JSX.Element {
  const route = entityRoute;
  const path = () => {
    const current = route();
    if (!current) return "";
    if (current.kind === "agent") return `/public/agent/${encodeURIComponent(current.id)}/pipeline`;
    if (current.kind === "miner") return `/public/leaderboard`;
    return "/public/operations";
  };
  const [payload, { refetch }] = createResource(
    () => route()?.key || "",
    async () => getJSON<PipelinePayload | LeaderboardPayload | OperationsPayload>(path()),
  );
  const agentPipeline = () =>
    route()?.kind === "agent" ? (payload() as PipelinePayload | undefined) : undefined;
  const minerEntry = () =>
    route()?.kind === "miner"
      ? (payload() as LeaderboardPayload | undefined)?.entries?.find(
          (entry) => entry.miner_hotkey === route()?.id,
        )
      : undefined;
  return (
    <Show when={route()}>
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

function ThemeControl(): JSX.Element {
  const modes = ["system", "light", "dark", "time"] as const;
  const [mode, setMode] = createSignal(document.documentElement.dataset.theme || "system");
  const apply = (next: string) => {
    setMode(next);
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("ditto:dashboard-theme", next);
    } catch {
      /* storage is optional */
    }
  };
  return (
    <div class="theme-switch" aria-label="Color theme">
      <For each={modes}>
        {(item) => (
          <button
            type="button"
            class="theme-option"
            classList={{ active: mode() === item }}
            onClick={() => apply(item)}
          >
            {item}
          </button>
        )}
      </For>
    </div>
  );
}

export default function App(): JSX.Element {
  const health = useEndpoint<HealthPayload>("/public/health");
  const leaderboard = useEndpoint<LeaderboardPayload>("/public/leaderboard");
  const operations = useEndpoint<OperationsPayload>("/public/operations");
  const activity = useEndpoint<ActivityPayload>("/public/activity?limit=100&page=1");
  const reviews = useEndpoint<AthSnapshot>(
    "/public/activity?review=ath&status=under_review&limit=100&page=1",
  );
  const glossary = useEndpoint<GlossaryPayload>("/public/bench/glossary");
  const benchConfig = useEndpoint<BenchConfigPayload>("/public/bench/config");
  const [lastRefresh, setLastRefresh] = createSignal(new Date());

  const refreshAll = () => {
    [health, leaderboard, operations, activity, reviews, glossary, benchConfig].forEach(
      (resource) => resource.refresh(),
    );
    setLastRefresh(new Date());
  };

  onMount(() => {
    syncFromLocation();
    initRouteListeners(() => undefined);
    const timer = window.setInterval(refreshAll, REFRESH_MS);
    onCleanup(() => window.clearInterval(timer));
  });

  createEffect(() => {
    const page = currentPage();
    document.title = `${PAGES[page].title} · Ditto SN118`;
  });

  return (
    <div class="layout">
      <a class="skip-link" href="#main-content">
        Skip to content
      </a>
      <aside class="sidebar">
        <div class="brand">
          <span class="mark">
            <img src={brandLogo} alt="" />
          </span>
          <div>
            <div class="brand-name">Ditto SN118</div>
            <div class="sub">Public transparency</div>
          </div>
        </div>
        <nav class="nav" aria-label="Dashboard pages">
          <For each={NAV}>
            {(item) => (
              <button
                type="button"
                class="nav-item"
                classList={{ active: currentPage() === item.page }}
                aria-current={currentPage() === item.page ? "page" : undefined}
                onClick={() => navigateToPage(item.page)}
              >
                <span class="ni-icon" aria-hidden="true">
                  {item.icon}
                </span>
                <span class="ni-text">
                  <span class="ni-label">{PAGES[item.page].title}</span>
                  <span class="ni-desc">{item.description}</span>
                </span>
              </button>
            )}
          </For>
        </nav>
        <div class="side-theme">
          <ThemeControl />
        </div>
        <div class="side-foot">
          <a class="btn" href={WANDB_URL} target="_blank" rel="noreferrer">
            Open W&amp;B ↗
          </a>
          <button class="btn ghost" type="button" onClick={refreshAll}>
            Refresh data
          </button>
          <small>
            Updated {lastRefresh().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </small>
        </div>
      </aside>
      <main class="main" id="main-content">
        <header class="page-header">
          <div>
            <h1>{PAGES[currentPage()].title}</h1>
            <p>{PAGES[currentPage()].sub}</p>
          </div>
          <span class="live-indicator">
            <i /> Live data
          </span>
        </header>
        <div class="page-content">
          <Switch>
            <Match when={currentPage() === "overview"}>
              <OverviewPage health={health} leaderboard={leaderboard} />
            </Match>
            <Match when={currentPage() === "operations"}>
              <OperationsPage resource={operations} />
            </Match>
            <Match when={currentPage() === "submissions"}>
              <SubmissionsPage resource={activity} />
            </Match>
            <Match when={currentPage() === "reviews"}>
              <ReviewsPage resource={reviews} />
            </Match>
            <Match when={currentPage() === "benchmark"}>
              <BenchmarkPage glossary={glossary} config={benchConfig} />
            </Match>
          </Switch>
        </div>
      </main>
      <EntityPanel />
      <div id="copy-status" class="visually-hidden" aria-live="polite" />
    </div>
  );
}
