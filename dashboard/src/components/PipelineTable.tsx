import { For, Show } from "solid-js";
import type { JSX } from "solid-js";

import { relTime, shortKey } from "../lib/format";
import type { ActivityEntry } from "../types";
import { EntityButton } from "./ui/EntityButton";
import { LoadingRows } from "./ui/States";
import { StatusChip } from "./ui/StatusChip";

export function PipelineTable(props: { entries: ActivityEntry[]; loading?: boolean }): JSX.Element {
  return (
    <div class="board" tabindex="0" aria-label="Submissions, horizontally scrollable">
      <table>
        <thead>
          <tr>
            <th>Submission</th>
            <th>Miner</th>
            <th>Status</th>
            <th class="num">Scores</th>
            <th>Submitted</th>
          </tr>
        </thead>
        <tbody>
          <Show when={props.loading}>
            <LoadingRows columns={5} />
          </Show>
          <For each={props.entries}>
            {(entry) => (
              <tr>
                <td>
                  <EntityButton kind="agent" id={entry.agent_id} class="row-title">
                    {entry.name || "Unnamed agent"}
                  </EntityButton>
                </td>
                <td>
                  <EntityButton kind="miner" id={entry.miner_hotkey}>
                    <span title={entry.miner_hotkey}>{shortKey(entry.miner_hotkey)}</span>
                  </EntityButton>
                </td>
                <td>
                  <StatusChip status={entry.status} />
                </td>
                <td class="num">
                  {entry.score_count ?? 0}/{entry.quorum ?? 3}
                </td>
                <td>{relTime(entry.submitted_at)}</td>
              </tr>
            )}
          </For>
        </tbody>
      </table>
    </div>
  );
}
