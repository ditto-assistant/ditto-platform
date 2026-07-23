import { For } from "solid-js";
import type { Accessor, JSX } from "solid-js";

import brandLogo from "../../assets/brand-logo.png";
import { WANDB_URL } from "../../lib/config";
import { PAGES } from "../../lib/router";
import type { PageName } from "../../lib/router";
import { currentPage, navigateToPage } from "../../stores/routeStore";
import { ThemeControl } from "./ThemeControl";

const NAV: Array<{ page: PageName; description: string }> = [
  { page: "overview", description: "Snapshot & leaderboard" },
  { page: "operations", description: "Pipeline & fleet health" },
  { page: "submissions", description: "Recent uploads" },
  { page: "reviews", description: "Active public holds" },
  { page: "benchmark", description: "Scoring benchmark" },
];

function NavIcon(props: { page: PageName }): JSX.Element {
  if (props.page === "overview")
    return (
      <svg class="ic" viewBox="0 0 24 24">
        <rect x="3" y="3" width="7" height="9" rx="1" />
        <rect x="14" y="3" width="7" height="5" rx="1" />
        <rect x="14" y="12" width="7" height="9" rx="1" />
        <rect x="3" y="16" width="7" height="5" rx="1" />
      </svg>
    );
  if (props.page === "operations")
    return (
      <svg class="ic" viewBox="0 0 24 24">
        <rect width="8" height="8" x="3" y="3" rx="2" />
        <path d="M7 11v4a2 2 0 0 0 2 2h4" />
        <rect width="8" height="8" x="13" y="13" rx="2" />
      </svg>
    );
  if (props.page === "submissions")
    return (
      <svg class="ic" viewBox="0 0 24 24">
        <path d="M22 12h-6l-2 3h-4l-2-3H2" />
        <path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z" />
      </svg>
    );
  if (props.page === "reviews")
    return (
      <svg class="ic" viewBox="0 0 24 24">
        <path d="M12 3 3.6 7.2v5.6c0 4.7 3.6 7.2 8.4 8.2 4.8-1 8.4-3.5 8.4-8.2V7.2Z" />
        <path d="M9 12h6M12 9v6" />
      </svg>
    );
  return (
    <svg class="ic" viewBox="0 0 24 24">
      <path d="M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.41 2.41 0 0 1 0-3.4l2.6-2.6a2.41 2.41 0 0 1 3.4 0Z" />
      <path d="m14.5 12.5 2-2M11.5 9.5l2-2M8.5 6.5l2-2M17.5 15.5l2-2" />
    </svg>
  );
}

export function Sidebar(props: { lastRefresh: Accessor<Date>; refresh: () => void }): JSX.Element {
  return (
    <aside class="sidebar">
      <div class="brand">
        <span class="mark">
          <img src={brandLogo} alt="" />
        </span>
        <div>
          <div class="brand-name">Ditto · Subnet 118</div>
          <div class="sub">Public agent-memory scoring leaderboard</div>
        </div>
      </div>
      <nav class="nav" aria-label="Dashboard pages">
        <For each={NAV}>
          {(item) => (
            <a
              href={`#/${item.page}`}
              class="nav-item"
              classList={{ active: currentPage() === item.page }}
              aria-current={currentPage() === item.page ? "page" : undefined}
              onClick={(event) => {
                event.preventDefault();
                navigateToPage(item.page);
              }}
            >
              <span class="ni-icon" aria-hidden="true">
                <NavIcon page={item.page} />
              </span>
              <span class="ni-text">
                <span class="ni-label">{PAGES[item.page].title}</span>
                <span class="ni-desc">{item.description}</span>
              </span>
            </a>
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
        <button class="btn ghost" type="button" onClick={props.refresh}>
          Refresh data
        </button>
        <small>
          Updated{" "}
          {props.lastRefresh().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </small>
      </div>
    </aside>
  );
}
