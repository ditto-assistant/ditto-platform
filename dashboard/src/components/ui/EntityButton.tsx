import { Show } from "solid-js";
import type { JSX } from "solid-js";

import type { EntityKind } from "../../lib/router";
import { pushEntityRoute } from "../../stores/routeStore";

export function EntityButton(props: {
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
