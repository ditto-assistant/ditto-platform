import { Match, Switch, createEffect, createSignal, onCleanup, onMount } from "solid-js";
import type { JSX } from "solid-js";

import { EntityPanel } from "./components/EntityPanel";
import { GlobalSearch } from "./components/search/GlobalSearch";
import { Sidebar } from "./components/shell/Sidebar";
import { useEndpoint } from "./data/useEndpoint";
import { REFRESH_MS } from "./lib/config";
import { PAGES } from "./lib/router";
import { BenchmarkPage } from "./pages/BenchmarkPage";
import { OperationsPage } from "./pages/OperationsPage";
import { OverviewPage } from "./pages/OverviewPage";
import { ReviewsPage } from "./pages/ReviewsPage";
import { SubmissionsPage } from "./pages/SubmissionsPage";
import { currentPage, initRouteListeners, syncFromLocation } from "./stores/routeStore";
import type {
  AthSnapshot,
  BenchConfigPayload,
  GlossaryPayload,
  HealthPayload,
  OperationsPayload,
  RolloutState,
  TimelinePayload,
} from "./types";

export default function App(): JSX.Element {
  const health = useEndpoint<HealthPayload>("/public/health");
  const operations = useEndpoint<OperationsPayload>("/public/operations");
  const reviews = useEndpoint<AthSnapshot>(
    "/public/activity?review=ath&status=under_review&limit=100&page=1",
  );
  const glossary = useEndpoint<GlossaryPayload>("/public/bench/glossary");
  const benchConfig = useEndpoint<BenchConfigPayload>("/public/bench/config");
  const rollout = useEndpoint<RolloutState>("/public/bench/rollout");
  const timeline = useEndpoint<TimelinePayload>("/public/bench/timeline");
  const [lastRefresh, setLastRefresh] = createSignal(new Date());
  const resources = [health, operations, reviews, glossary, benchConfig, rollout, timeline];

  const refreshAll = () => {
    resources.forEach((resource) => resource.refresh());
    setLastRefresh(new Date());
  };

  onMount(() => {
    syncFromLocation();
    initRouteListeners(() => undefined);
    const timer = window.setInterval(refreshAll, REFRESH_MS);
    onCleanup(() => window.clearInterval(timer));
  });

  createEffect(() => {
    document.title = `${PAGES[currentPage()].title} · Ditto SN118`;
  });

  return (
    <div class="layout">
      <a class="skip-link" href="#main-content">
        Skip to content
      </a>
      <Sidebar lastRefresh={lastRefresh} refresh={refreshAll} />
      <main class="main" id="main-content">
        <header class="page-header">
          <div>
            <h1>{PAGES[currentPage()].title}</h1>
            <p>{PAGES[currentPage()].sub}</p>
          </div>
          <div class="page-actions">
            <GlobalSearch />
            <span class="live-indicator">
              <i /> Live data
            </span>
          </div>
        </header>
        <div class="page-content">
          <Switch>
            <Match when={currentPage() === "overview"}>
              <OverviewPage health={health} rollout={rollout} />
            </Match>
            <Match when={currentPage() === "operations"}>
              <OperationsPage resource={operations} />
            </Match>
            <Match when={currentPage() === "submissions"}>
              <SubmissionsPage />
            </Match>
            <Match when={currentPage() === "reviews"}>
              <ReviewsPage resource={reviews} />
            </Match>
            <Match when={currentPage() === "benchmark"}>
              <BenchmarkPage glossary={glossary} config={benchConfig} timeline={timeline} />
            </Match>
          </Switch>
        </div>
      </main>
      <EntityPanel />
      <div id="copy-status" class="visually-hidden" aria-live="polite" />
    </div>
  );
}
