import { For, Show, createEffect, onMount } from "solid-js";
import type { JSX } from "solid-js";

import { PAGES } from "./lib/router";
import type { PageName } from "./lib/router";
import { currentPage, initRouteListeners, syncFromLocation } from "./stores/routeStore";

const PAGE_NAMES = Object.keys(PAGES) as PageName[];

// Composition root. The shell (sidebar, global search, entity panel) and the
// real pages land with their ports; until then each page renders its heading
// so routing is exercisable end to end.
export default function App(): JSX.Element {
  onMount(() => {
    syncFromLocation();
    initRouteListeners(() => undefined);
  });

  createEffect(() => {
    document.title = `${PAGES[currentPage()].title} · Ditto SN118`;
  });

  return (
    <div class="layout">
      <a class="skip-link" href="#main-content">
        Skip to content
      </a>
      <main class="main" id="main-content">
        <For each={PAGE_NAMES}>
          {(name) => (
            <Show when={currentPage() === name}>
              <section class="page" data-page={name}>
                <h1>{PAGES[name].title}</h1>
                <p>{PAGES[name].sub}</p>
              </section>
            </Show>
          )}
        </For>
      </main>
    </div>
  );
}
