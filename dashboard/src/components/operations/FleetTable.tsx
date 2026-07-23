import { For, Show } from "solid-js";
import type { JSX } from "solid-js";

import type { FleetEntry } from "../../types";
import { pct, relTime, shortKey } from "../../lib/format";
import { EntityButton } from "../ui/EntityButton";
import { StatusChip } from "../ui/StatusChip";

export function FleetTable(props: {
  entries: FleetEntry[];
  kind: "validator" | "screener";
}): JSX.Element {
  const key = (entry: FleetEntry) =>
    props.kind === "validator" ? entry.validator_hotkey : entry.screener_hotkey;
  return (
    <div
      class="board fleet-table-wrap"
      tabindex="0"
      aria-label={`${props.kind} fleet, horizontally scrollable`}
    >
      <table class="fleet-table">
        <thead>
          <tr>
            <th>{props.kind === "validator" ? "Validator" : "Screener"}</th>
            <th>Availability</th>
            <th>Work</th>
            <th>Version / protocol</th>
            <th>Admission</th>
            <th>Stack</th>
            <th class="num">CPU</th>
            <th class="num">Memory</th>
            <th>Last seen</th>
          </tr>
        </thead>
        <tbody>
          <For each={props.entries}>
            {(entry) => (
              <tr>
                <td>
                  <EntityButton kind={props.kind} id={key(entry)} class="row-title">
                    {shortKey(key(entry)) || "Unknown worker"}
                  </EntityButton>
                  <Show when={entry.instance_id}>
                    <small>{entry.instance_id}</small>
                  </Show>
                </td>
                <td>
                  <StatusChip status={entry.availability || entry.health || entry.state} />
                </td>
                <td>
                  <strong>
                    {entry.active_agent_name ||
                      entry.active_benchmark?.agent_name ||
                      entry.state ||
                      "Idle"}
                  </strong>
                  <small>
                    {entry.active_benchmark?.stage ||
                      entry.screening_progress?.stage ||
                      "No active assignment"}
                  </small>
                </td>
                <td>
                  {entry.software_version || "—"}
                  <small>
                    Protocol {entry.protocol_version ?? "—"}
                    <Show when={entry.policy_version != null}>
                      {" "}
                      · Policy {entry.policy_version}
                    </Show>
                  </small>
                </td>
                <td>
                  <StatusChip status={entry.admission || entry.assignment_state || "unknown"} />
                </td>
                <td>
                  {entry.stack?.mode || "—"}
                  <small>
                    {entry.stack?.components
                      ? `${Object.keys(entry.stack.components).length} reported components`
                      : "Identity not reported"}
                  </small>
                </td>
                <td class="num">
                  {entry.system_metrics ? pct(entry.system_metrics.cpu_percent / 100) : "—"}
                </td>
                <td class="num">
                  {entry.system_metrics ? pct(entry.system_metrics.memory_percent / 100) : "—"}
                </td>
                <td>{relTime(entry.reported_at || entry.seen_at)}</td>
              </tr>
            )}
          </For>
        </tbody>
      </table>
    </div>
  );
}
