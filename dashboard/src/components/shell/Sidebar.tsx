import { For } from "solid-js";
import type { Accessor, JSX } from "solid-js";

import brandLogo from "../../assets/brand-logo.png";
import { WANDB_URL } from "../../lib/config";
import { PAGES } from "../../lib/router";
import type { PageName } from "../../lib/router";
import { currentPage, navigateToPage } from "../../stores/routeStore";
import { ThemeControl } from "./ThemeControl";

const NAV: Array<{ page: PageName; icon: string; description: string }> = [
  { page: "overview", icon: "◫", description: "Scores and emissions" },
  { page: "operations", icon: "⌁", description: "Pipeline and fleet" },
  { page: "submissions", icon: "↳", description: "Screening history" },
  { page: "reviews", icon: "◇", description: "Held high scores" },
  { page: "benchmark", icon: "◎", description: "Scoring contract" },
];

export function Sidebar(props: { lastRefresh: Accessor<Date>; refresh: () => void }): JSX.Element {
  return (
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
