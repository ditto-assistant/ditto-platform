import { For, Show } from "solid-js";
import type { JSX } from "solid-js";

import { ErrorState } from "../components/ui/States";
import { StatusChip } from "../components/ui/StatusChip";
import type { ResourceState } from "../data/useEndpoint";
import type { BenchConfigPayload, GlossaryPayload } from "../types";

export function BenchmarkPage(props: {
  glossary: ResourceState<GlossaryPayload>;
  config: ResourceState<BenchConfigPayload>;
}): JSX.Element {
  const categories = () => props.glossary.data()?.categories || [];
  return (
    <>
      <section class="benchmark-contract">
        <div>
          <span>Active public contract</span>
          <strong>DittoBench v{props.config.data()?.bench_version ?? "—"}</strong>
        </div>
        <dl>
          <div>
            <dt>Canonical harness</dt>
            <dd>{props.config.data()?.harness?.canonical_id || "Not reported"}</dd>
          </div>
          <div>
            <dt>Serving model</dt>
            <dd>{props.config.data()?.harness?.serving || "Not reported"}</dd>
          </div>
          <div>
            <dt>Reasoning</dt>
            <dd>{props.config.data()?.harness?.thinking ? "Enabled" : "Disabled"}</dd>
          </div>
        </dl>
      </section>
      <Show when={props.glossary.error() || props.config.error()}>
        <ErrorState
          error={props.glossary.error() || props.config.error()}
          retry={() => {
            props.glossary.refresh();
            props.config.refresh();
          }}
        />
      </Show>
      <section>
        <div class="section-head">
          <h2>What the benchmark measures</h2>
          <span class="hint">{categories().length} public categories</span>
        </div>
        <div class="category-list">
          <For each={categories()}>
            {(category) => (
              <article>
                <div>
                  <StatusChip status={category.kind} />
                  <h3>{category.label || category.key}</h3>
                </div>
                <p>{category.purpose}</p>
                <Show when={category.example}>
                  <blockquote>{category.example}</blockquote>
                </Show>
              </article>
            )}
          </For>
        </div>
      </section>
    </>
  );
}
