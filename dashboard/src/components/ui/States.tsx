import { For } from "solid-js";
import type { JSX } from "solid-js";

export function ErrorState(props: { error: unknown; retry: () => void }): JSX.Element {
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

export function LoadingRows(props: { columns: number }): JSX.Element {
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

export function EmptyState(props: { title: string; detail: string }): JSX.Element {
  return (
    <div class="state-panel empty-state">
      <strong>{props.title}</strong>
      <span>{props.detail}</span>
    </div>
  );
}
