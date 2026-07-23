import { Show } from "solid-js";
import type { JSX } from "solid-js";

import type { RolloutState } from "../../types";

export function RolloutPanel(props: { rollout?: RolloutState }): JSX.Element {
  const complete = () => props.rollout?.cohort_ready_count ?? 0;
  const total = () => props.rollout?.cohort_size ?? 0;
  const progress = () => (total() > 0 ? Math.min(100, (complete() / total()) * 100) : 0);
  return (
    <Show when={props.rollout && props.rollout.desired_version !== props.rollout.active_version}>
      <section class="rollout-strip" aria-labelledby="rollout-title">
        <div>
          <span class="status-chip status-provisional">Benchmark rollout</span>
          <h2 id="rollout-title">
            v{props.rollout?.active_version ?? "—"} → v{props.rollout?.desired_version ?? "—"}
          </h2>
          <p>
            {props.rollout?.blocked_reason ||
              "Qualification scores are collecting before activation."}
          </p>
        </div>
        <div class="rollout-progress">
          <strong>
            {complete()} / {total()} cohort ready
          </strong>
          <progress value={progress()} max="100">
            {progress()}%
          </progress>
          <span>
            {props.rollout?.ranked_quorum_agents ?? 0} /{" "}
            {props.rollout?.min_ranked_quorum_agents ?? 0} ranked agents at quorum
          </span>
        </div>
      </section>
    </Show>
  );
}
