import { For, Show, createEffect, createSignal, onCleanup, onMount } from "solid-js";
import type { JSX } from "solid-js";

import { useEndpoint } from "../../data/useEndpoint";
import { navigateToPage, pushEntityRoute } from "../../stores/routeStore";
import type { ActivityPayload } from "../../types";
import { shortKey } from "../../lib/format";

export function GlobalSearch(): JSX.Element {
  const [query, setQuery] = createSignal("");
  const [open, setOpen] = createSignal(false);
  const path = () =>
    query().trim().length >= 2
      ? `/public/activity?page=1&limit=8&q=${encodeURIComponent(query().trim())}`
      : "";
  const results = useEndpoint<ActivityPayload>(path);
  let input: HTMLInputElement | undefined;

  onMount(() => {
    const shortcut = (event: KeyboardEvent) => {
      if (event.key === "/" && !event.metaKey && !event.ctrlKey && !event.altKey) {
        const target = event.target as HTMLElement | null;
        if (target?.matches("input, textarea, select, [contenteditable=true]")) return;
        event.preventDefault();
        input?.focus();
        setOpen(true);
      }
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", shortcut);
    onCleanup(() => window.removeEventListener("keydown", shortcut));
  });

  createEffect(() => setOpen(query().trim().length >= 2));

  return (
    <div class="global-search" role="search">
      <label>
        <span class="visually-hidden">Search agents, submissions, and miners</span>
        <input
          ref={(element) => {
            input = element;
          }}
          value={query()}
          onInput={(event) => setQuery(event.currentTarget.value)}
          onFocus={() => query().trim().length >= 2 && setOpen(true)}
          placeholder="Search miner, agent, or submission"
          aria-expanded={open()}
          aria-controls="global-search-results"
        />
        <kbd>/</kbd>
      </label>
      <Show when={open()}>
        <div class="search-popover" id="global-search-results">
          <Show when={results.loading()}>
            <p>Searching public records…</p>
          </Show>
          <For
            each={results.data()?.entries || []}
            fallback={
              <Show when={!results.loading()}>
                <p>No matching public records.</p>
              </Show>
            }
          >
            {(entry) => (
              <button
                type="button"
                onClick={() => {
                  if (entry.agent_id) pushEntityRoute("agent", entry.agent_id);
                  else navigateToPage("submissions");
                  setOpen(false);
                }}
              >
                <strong>{entry.name || "Unnamed agent"}</strong>
                <span>
                  {entry.status || "unknown"} · {shortKey(entry.miner_hotkey)}
                </span>
              </button>
            )}
          </For>
          <button class="search-all" type="button" onClick={() => navigateToPage("submissions")}>
            Open submissions →
          </button>
        </div>
      </Show>
    </div>
  );
}
