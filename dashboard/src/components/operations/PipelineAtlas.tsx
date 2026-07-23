import { For } from "solid-js";
import type { JSX } from "solid-js";

import type { PipelineEntry } from "../../types";
import { EntityButton } from "../ui/EntityButton";
import { StatusChip } from "../ui/StatusChip";
import { relTime } from "../../lib/format";

const STAGES = [
  {
    key: "waiting-screen",
    title: "Waiting screening",
    statuses: ["waiting_screening", "uploaded"],
  },
  { key: "screening", title: "Screening", statuses: ["screening", "under_review"] },
  {
    key: "waiting-validator",
    title: "Waiting scores",
    statuses: ["waiting_validator", "waiting_scoring"],
  },
  { key: "evaluating", title: "Evaluating", statuses: ["scoring", "evaluating", "benchmarking"] },
  {
    key: "scored",
    title: "Recent scores",
    statuses: ["scored", "below_score_floor", "rejected", "failed"],
  },
];

export function PipelineAtlas(props: { entries: PipelineEntry[] }): JSX.Element {
  return (
    <div class="pipeline-atlas" aria-label="Submission pipeline stages">
      <For each={STAGES}>
        {(stage, index) => {
          const entries = () =>
            props.entries.filter((entry) => stage.statuses.includes(entry.status || ""));
          return (
            <section class="pipeline-stage">
              <header>
                <span>{index() + 1}</span>
                <div>
                  <h3>{stage.title}</h3>
                  <small>{entries().length} submissions</small>
                </div>
              </header>
              <div class="pipeline-stage-rows">
                <For
                  each={entries().slice(0, 6)}
                  fallback={<p class="pipeline-empty">No submissions</p>}
                >
                  {(entry) => (
                    <article>
                      <div>
                        <EntityButton kind="agent" id={entry.agent_id} class="row-title">
                          {entry.name || "Unnamed agent"}
                        </EntityButton>
                        <small>{relTime(entry.submitted_at)}</small>
                      </div>
                      <StatusChip status={entry.status} />
                    </article>
                  )}
                </For>
              </div>
            </section>
          );
        }}
      </For>
    </div>
  );
}
