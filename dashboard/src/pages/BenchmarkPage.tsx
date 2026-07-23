import { For, Show } from "solid-js";
import type { JSX } from "solid-js";

import { MemoryTimeline } from "../components/benchmark/MemoryTimeline";
import { ErrorState } from "../components/ui/States";
import { StatusChip } from "../components/ui/StatusChip";
import type { ResourceState } from "../data/useEndpoint";
import type { BenchConfigPayload, GlossaryPayload, TimelinePayload } from "../types";

function configRows(value: Record<string, unknown> | undefined): Array<[string, string]> {
  return Object.entries(value || {}).map(([key, item]) => [
    key.replaceAll("_", " "),
    Array.isArray(item)
      ? item.join(", ")
      : item !== null && typeof item === "object"
        ? JSON.stringify(item)
        : typeof item === "string" || typeof item === "number" || typeof item === "boolean"
          ? String(item)
          : "—",
  ]);
}

function harnessRows(config: BenchConfigPayload | undefined): Array<[string, string]> {
  const harness = config?.harness;
  if (!harness) return [];
  return configRows({
    canonical_id: harness.canonical_id,
    serving: harness.serving,
    reasoning_effort: harness.reasoning_effort,
    thinking: harness.thinking,
  });
}

export function BenchmarkPage(props: {
  glossary: ResourceState<GlossaryPayload>;
  config: ResourceState<BenchConfigPayload>;
  timeline: ResourceState<TimelinePayload>;
}): JSX.Element {
  const categories = () => props.glossary.data()?.categories || [];
  const metrics = () => props.glossary.data()?.metrics || [];
  const versions = () => props.glossary.data()?.versions || [];
  return (
    <>
      <section class="benchmark-hero">
        <div>
          <span class="status-chip status-final">Active public contract</span>
          <h2>DittoBench v{props.config.data()?.bench_version ?? "—"}</h2>
          <p>
            One frozen, reproducible contract for evaluating tool use and memory. A version change
            creates a new comparison era; historical scores are never silently reinterpreted.
          </p>
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
            <dd>
              {props.config.data()?.harness?.reasoning_effort ||
                (props.config.data()?.harness?.thinking ? "Enabled" : "Disabled")}
            </dd>
          </div>
          <div>
            <dt>Score quorum</dt>
            <dd>{props.timeline.data()?.score_quorum ?? 3} validators</dd>
          </div>
        </dl>
      </section>
      <Show when={props.glossary.error() || props.config.error() || props.timeline.error()}>
        <ErrorState
          error={props.glossary.error() || props.config.error() || props.timeline.error()}
          retry={() => {
            props.glossary.refresh();
            props.config.refresh();
            props.timeline.refresh();
          }}
        />
      </Show>
      <section aria-labelledby="pillars-title">
        <div class="section-head">
          <div>
            <h2 id="pillars-title">Two scoring pillars, equal weight</h2>
            <p>
              The headline composite balances reliable action with durable, instruction-safe memory.
            </p>
          </div>
        </div>
        <div class="pillar-grid">
          <article>
            <span>50%</span>
            <h3>Tool use</h3>
            <p>
              Chooses the right tool, supplies grounded arguments, follows multi-step dependencies,
              and avoids unnecessary external work when memory already contains the answer.
            </p>
          </article>
          <article>
            <span>50%</span>
            <h3>Memory</h3>
            <p>
              Recalls and reasons over facts across sessions and time while resisting hidden
              instructions, stale claims, and unsupported synthesis.
            </p>
          </article>
        </div>
      </section>
      <MemoryTimeline timeline={props.timeline.data()} />
      <section aria-labelledby="emissions-flow-title">
        <div class="section-head">
          <div>
            <h2 id="emissions-flow-title">How a score becomes emissions</h2>
            <p>
              Every stage is public and bounded; no single provisional run becomes an authoritative
              standing.
            </p>
          </div>
        </div>
        <ol class="score-flow">
          <li>
            <strong>Commit</strong>
            <span>The miner uploads an immutable artifact before its seed exists.</span>
          </li>
          <li>
            <strong>Screen</strong>
            <span>Policy checks admit or reject the source with public-safe reasons.</span>
          </li>
          <li>
            <strong>Evaluate</strong>
            <span>
              Independent validators run the frozen harness against unpredictable datasets.
            </span>
          </li>
          <li>
            <strong>Reach quorum</strong>
            <span>The canonical median forms after the required accepted scores arrive.</span>
          </li>
          <li>
            <strong>Apply KOTH</strong>
            <span>
              Eligibility, registration, and the dethrone margin determine emission recipients.
            </span>
          </li>
        </ol>
      </section>
      <section aria-labelledby="setup-title">
        <div class="section-head">
          <div>
            <h2 id="setup-title">Frozen scoring setup</h2>
            <p>
              Dataset generation, harness identity, model serving, grading, and published evidence
              are versioned together.
            </p>
          </div>
        </div>
        <div class="setup-grid">
          <article>
            <h3>Harness</h3>
            <dl>
              <For each={harnessRows(props.config.data())}>
                {([key, value]) => (
                  <div>
                    <dt>{key}</dt>
                    <dd>{value}</dd>
                  </div>
                )}
              </For>
            </dl>
          </article>
          <article>
            <h3>Dataset</h3>
            <dl>
              <For each={configRows(props.config.data()?.dataset)}>
                {([key, value]) => (
                  <div>
                    <dt>{key}</dt>
                    <dd>{value}</dd>
                  </div>
                )}
              </For>
            </dl>
          </article>
          <article>
            <h3>Grading</h3>
            <dl>
              <For each={configRows(props.config.data()?.grading)}>
                {([key, value]) => (
                  <div>
                    <dt>{key}</dt>
                    <dd>{value}</dd>
                  </div>
                )}
              </For>
            </dl>
          </article>
        </div>
      </section>
      <section aria-labelledby="versions-title">
        <div class="section-head">
          <div>
            <h2 id="versions-title">Benchmark version history</h2>
            <p>
              Scores compare only within the same version. Each entry records why a new immutable
              contract was released.
            </p>
          </div>
        </div>
        <div class="version-history">
          <For each={versions()}>
            {(version) => (
              <details open={version.version === props.config.data()?.bench_version}>
                <summary>
                  <span>v{version.version}</span>
                  <strong>{version.title}</strong>
                  <time>{version.epoch}</time>
                </summary>
                <p>{version.summary}</p>
                <ul>
                  <For each={version.highlights || []}>{(highlight) => <li>{highlight}</li>}</For>
                </ul>
              </details>
            )}
          </For>
        </div>
      </section>
      <section aria-labelledby="glossary-title">
        <div class="section-head">
          <div>
            <h2 id="glossary-title">Scoring glossary</h2>
            <p>Definitions are served by the same versioned public API as the dashboard.</p>
          </div>
          <span class="hint">
            {metrics().length} metrics · {categories().length} categories
          </span>
        </div>
        <div class="glossary-grid">
          <For each={metrics()}>
            {(metric) => (
              <article>
                <h3>{metric.label || metric.key}</h3>
                <p>{metric.description}</p>
              </article>
            )}
          </For>
        </div>
        <details class="category-details">
          <summary>Browse all {categories().length} benchmark categories</summary>
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
        </details>
      </section>
      <section class="open-source-stack">
        <div>
          <h2>Audit the open stack</h2>
          <p>
            Reproduce the dataset, inspect the deterministic grader, and verify published run
            evidence independently.
          </p>
        </div>
        <div>
          <a
            class="btn"
            href="https://github.com/ditto-assistant/dittobench"
            target="_blank"
            rel="noreferrer"
          >
            DittoBench ↗
          </a>
          <a
            class="btn ghost"
            href="https://github.com/ditto-assistant/dittobench-datagen"
            target="_blank"
            rel="noreferrer"
          >
            Generator & grader ↗
          </a>
          <a
            class="btn ghost"
            href="https://github.com/ditto-assistant/ditto-platform"
            target="_blank"
            rel="noreferrer"
          >
            Platform ↗
          </a>
          <a
            class="btn ghost"
            href="https://github.com/ditto-assistant/ditto-subnet"
            target="_blank"
            rel="noreferrer"
          >
            Subnet ↗
          </a>
        </div>
      </section>
    </>
  );
}
