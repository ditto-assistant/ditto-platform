import { Show } from "solid-js";
import type { JSX } from "solid-js";

export function Pager(props: {
  page: number;
  pages: number;
  total?: number;
  onPage: (page: number) => void;
}): JSX.Element {
  return (
    <Show when={props.pages > 1}>
      <nav class="pager" aria-label="Results pages">
        <button
          class="btn ghost"
          type="button"
          disabled={props.page <= 1}
          onClick={() => props.onPage(props.page - 1)}
        >
          ← Previous
        </button>
        <span>
          Page <strong>{props.page}</strong> of <strong>{props.pages}</strong>
          <Show when={props.total != null}> · {props.total} results</Show>
        </span>
        <button
          class="btn ghost"
          type="button"
          disabled={props.page >= props.pages}
          onClick={() => props.onPage(props.page + 1)}
        >
          Next →
        </button>
      </nav>
    </Show>
  );
}
