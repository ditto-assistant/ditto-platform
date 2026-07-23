import { For, Show, createMemo, createSignal } from "solid-js";
import type { JSX } from "solid-js";

import { PipelineTable } from "../components/PipelineTable";
import { Pager } from "../components/ui/Pager";
import { EmptyState, ErrorState } from "../components/ui/States";
import { useEndpoint } from "../data/useEndpoint";
import type { ActivityPayload } from "../types";

const STATUS_LABELS: Record<string, string> = {
  waiting_screening: "Waiting screening",
  screening: "Screening",
  waiting_validator: "Waiting scores",
  scoring: "Scoring",
  under_review: "Under review",
  scored: "Scored",
  rejected: "Rejected",
  failed: "Failed",
};

export function SubmissionsPage(): JSX.Element {
  const [query, setQuery] = createSignal("");
  const [statuses, setStatuses] = createSignal<string[]>([]);
  const [page, setPage] = createSignal(1);
  const requestPath = createMemo(() => {
    const params = new URLSearchParams({ page: String(page()), limit: "10" });
    if (query().trim()) params.set("q", query().trim());
    statuses().forEach((status) => params.append("status", status));
    return `/public/activity?${params}`;
  });
  const resource = useEndpoint<ActivityPayload>(requestPath);
  const toggle = (status: string) => {
    setStatuses((current) =>
      current.includes(status) ? current.filter((item) => item !== status) : [...current, status],
    );
    setPage(1);
  };
  return (
    <section>
      <div class="section-head">
        <div>
          <h2>Public submission activity</h2>
          <p>
            Track screening, validation, scoring, and review outcomes without leaving the public
            audit trail.
          </p>
        </div>
        <span class="hint">{resource.data()?.total ?? 0} submissions</span>
      </div>
      <div class="toolbar activity-toolbar">
        <label class="search-field">
          <span class="visually-hidden">Search submissions</span>
          <input
            value={query()}
            onInput={(event) => {
              setQuery(event.currentTarget.value);
              setPage(1);
            }}
            placeholder="Agent name, submission ID, or miner hotkey"
          />
        </label>
        <Show when={statuses().length}>
          <button
            class="btn ghost"
            type="button"
            onClick={() => {
              setStatuses([]);
              setPage(1);
            }}
          >
            Clear filters
          </button>
        </Show>
      </div>
      <div class="activity-filter-list" aria-label="Submission status filters">
        <For each={Object.entries(resource.data()?.status_counts || {})}>
          {([status, count]) => (
            <button
              class="activity-filter"
              type="button"
              aria-pressed={statuses().includes(status)}
              onClick={() => toggle(status)}
            >
              {STATUS_LABELS[status] || status.replaceAll("_", " ")} <b>{count}</b>
            </button>
          )}
        </For>
      </div>
      <Pager
        page={resource.data()?.page ?? page()}
        pages={resource.data()?.total_pages ?? 1}
        total={resource.data()?.total}
        onPage={setPage}
      />
      <Show when={resource.error()}>
        <ErrorState error={resource.error()} retry={resource.refresh} />
      </Show>
      <Show when={!resource.error()}>
        <PipelineTable entries={resource.data()?.entries || []} loading={resource.loading()} />
      </Show>
      <Show when={!resource.loading() && (resource.data()?.entries || []).length === 0}>
        <EmptyState
          title="No matching submissions"
          detail="Clear a filter or search term to widen the public activity view."
        />
      </Show>
      <Pager
        page={resource.data()?.page ?? page()}
        pages={resource.data()?.total_pages ?? 1}
        total={resource.data()?.total}
        onPage={setPage}
      />
    </section>
  );
}
