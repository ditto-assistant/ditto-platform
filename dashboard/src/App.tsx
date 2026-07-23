import { Match, Switch, createEffect, createSignal, onCleanup, onMount } from "solid-js";
import type { JSX } from "solid-js";

import { EntityPanel } from "./components/EntityPanel";
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
  ActivityPayload,
  AthSnapshot,
  BenchConfigPayload,
  GlossaryPayload,
  HealthPayload,
  LeaderboardPayload,
  OperationsPayload,
} from "./types";

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
  const resources = [health, leaderboard, operations, activity, reviews, glossary, benchConfig];

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
