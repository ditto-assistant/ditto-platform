import { For, createSignal } from "solid-js";
import type { JSX } from "solid-js";

export function ThemeControl(): JSX.Element {
  const modes = ["system", "light", "dark", "time"] as const;
  const [mode, setMode] = createSignal(document.documentElement.dataset.theme || "system");
  const apply = (next: string) => {
    setMode(next);
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("ditto:dashboard-theme", next);
    } catch {
      // Storage is optional; the theme still applies for this page view.
    }
  };
  return (
    <div class="theme-switch" aria-label="Color theme">
      <For each={modes}>
        {(item) => (
          <button
            type="button"
            class="theme-option"
            classList={{ active: mode() === item }}
            onClick={() => apply(item)}
          >
            {item}
          </button>
        )}
      </For>
    </div>
  );
}
