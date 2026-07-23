import { For, Show, createMemo } from "solid-js";
import type { JSX } from "solid-js";

import type { TimelinePayload } from "../../types";
import { timelineDate } from "../../lib/format";

const EVIDENCE_URL =
  "https://github.com/ditto-assistant/dittobench-api/tree/beb8e1a5fb5a2c35f5c34ce33b422978504d611e/docs/third-party-benchmark-timeline";
const REFERENCES = [
  {
    id: "hermes",
    label: "Hermes Agent 0.19.0",
    profile: "Native SessionDB session_search",
    points: [
      [2, 0.1111111111],
      [3, 0.1279661017],
      [4, 0.1779661017],
      [5, 0.1690705104],
      [6, 0.1979166667],
    ] as Array<[number, number]>,
  },
  {
    id: "openclaw",
    label: "OpenClaw 2026.7.1",
    profile: "Native memory-core FTS · 20-result recall",
    points: [
      [2, 0.3333333333],
      [3, 0.3771186441],
      [4, 0.3559322034],
      [5, 0.421875],
      [6, 0.3333333333],
    ] as Array<[number, number]>,
  },
];

export function MemoryTimeline(props: { timeline?: TimelinePayload }): JSX.Element {
  const points = () => props.timeline?.points || [];
  const values = () =>
    [
      ...points().map((point) => Number(point.memory_mean)),
      ...REFERENCES.flatMap((series) => series.points.map(([, value]) => value)),
    ].filter(Number.isFinite);
  const dates = () =>
    points()
      .map((point) => Date.parse(point.recorded_at || ""))
      .filter(Number.isFinite);
  const x = (date: string | undefined) => {
    const parsed = Date.parse(date || "");
    const domain = dates();
    if (!Number.isFinite(parsed) || !domain.length) return 40;
    return (
      40 + ((parsed - Math.min(...domain)) / (Math.max(...domain) - Math.min(...domain) || 1)) * 700
    );
  };
  const y = (value: number | string | undefined) => {
    const numeric = Number(value);
    const domain = values();
    if (!Number.isFinite(numeric) || !domain.length) return 180;
    return (
      180 -
      ((numeric - Math.min(...domain)) / (Math.max(...domain) - Math.min(...domain) || 1)) * 130
    );
  };
  const path = createMemo(() =>
    points()
      .map(
        (point, index) =>
          `${index ? "L" : "M"}${x(point.recorded_at).toFixed(1)},${y(point.memory_mean).toFixed(1)}`,
      )
      .join(" "),
  );
  const releaseDate = (version: number) =>
    props.timeline?.releases?.find((release) => Number(release.bench_version) === version)
      ?.released_at;
  const referencePath = (series: (typeof REFERENCES)[number]) =>
    series.points
      .map(
        ([version, value], index) =>
          `${index ? "L" : "M"}${x(releaseDate(version)).toFixed(1)},${y(value).toFixed(1)}`,
      )
      .join(" ");
  return (
    <section aria-labelledby="memory-timeline-title">
      <div class="section-head">
        <div>
          <h2 id="memory-timeline-title">How far miners have taken memory</h2>
          <p>
            Observed best miner memory scores alongside independent reference-harness bounds.
            Version markers separate non-comparable benchmark eras.
          </p>
        </div>
        <span class="hint">Observational · not ranking input</span>
      </div>
      <div class="memory-timeline-legend">
        <span>
          <i /> Ditto miner
        </span>
        <span class="hermes">
          <i /> Hermes reference
        </span>
        <span class="openclaw">
          <i /> OpenClaw reference
        </span>
      </div>
      <div class="memory-timeline-frame">
        <svg
          class="memory-timeline-svg"
          viewBox="0 0 780 220"
          role="img"
          aria-label="Memory scores across benchmark releases"
        >
          <For each={[0, 0.25, 0.5, 0.75, 1]}>
            {(value) => (
              <>
                <line
                  class="timeline-grid"
                  x1="40"
                  x2="740"
                  y1={180 - value * 130}
                  y2={180 - value * 130}
                />
                <text class="timeline-axis-label" x="6" y={184 - value * 130}>
                  {value.toFixed(2)}
                </text>
              </>
            )}
          </For>
          <For each={props.timeline?.releases || []}>
            {(release) => (
              <>
                <line
                  class="timeline-release"
                  x1={x(release.released_at)}
                  x2={x(release.released_at)}
                  y1="35"
                  y2="186"
                />
                <text class="timeline-release-label" x={x(release.released_at) + 4} y="28">
                  v{release.bench_version}
                </text>
              </>
            )}
          </For>
          <path class="timeline-path miner" d={path()} />
          <For each={REFERENCES}>
            {(series) => (
              <>
                <path class={`timeline-path ${series.id}`} d={referencePath(series)} />
                <For each={series.points}>
                  {([version, value]) => (
                    <circle
                      class={`timeline-point ${series.id}`}
                      cx={x(releaseDate(version))}
                      cy={y(value)}
                      r="4"
                    >
                      <title>
                        {series.label} · Bench v{version} · {value.toFixed(3)} · measured 2026-07-23
                      </title>
                    </circle>
                  )}
                </For>
              </>
            )}
          </For>
          <For each={points()}>
            {(point) => (
              <circle
                class="timeline-point miner"
                cx={x(point.recorded_at)}
                cy={y(point.memory_mean)}
                r="4"
              >
                <title>
                  {point.agent_name || point.agent_id || "Miner"} ·{" "}
                  {Number(point.memory_mean).toFixed(3)} · {timelineDate(point.recorded_at || "")}
                </title>
              </circle>
            )}
          </For>
        </svg>
      </div>
      <details class="harness-comparison-method">
        <summary>Method and comparability</summary>
        <div class="harness-comparison-method-body">
          <p>
            The miner line uses finalized three-validator memory medians within each benchmark era.
            Hermes and OpenClaw are single-seed off-network practice runs measured retrospectively
            on immutable v2-v6 contracts with the same Qwen3-32B model, pinned OpenRouter/Nebius
            route, and seed. They never enter rankings, KOTH, weights, or payouts.
          </p>
          <a href={EVIDENCE_URL} target="_blank" rel="noreferrer">
            Inspect retained run evidence ↗
          </a>
        </div>
      </details>
      <details class="timeline-data-details">
        <summary>View the underlying public observations</summary>
        <div class="timeline-data-table-wrap">
          <table class="timeline-data-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Benchmark</th>
                <th>Agent</th>
                <th>Memory</th>
              </tr>
            </thead>
            <tbody>
              <For each={points()}>
                {(point) => (
                  <tr>
                    <td>{timelineDate(point.recorded_at || "")}</td>
                    <td>v{point.bench_version ?? "—"}</td>
                    <td>{point.agent_name || point.agent_id || "—"}</td>
                    <td>{Number(point.memory_mean).toFixed(3)}</td>
                  </tr>
                )}
              </For>
            </tbody>
          </table>
        </div>
      </details>
      <Show when={!points().length}>
        <p class="memory-timeline-note">No public timeline observations are available yet.</p>
      </Show>
    </section>
  );
}
