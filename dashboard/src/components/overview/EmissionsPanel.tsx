import { For, Show } from "solid-js";
import type { JSX } from "solid-js";

import type { EmissionsFold } from "../../types";
import { pct, shortKey } from "../../lib/format";

export function EmissionsPanel(props: { emissions?: EmissionsFold | null }): JSX.Element {
  return (
    <Show when={props.emissions}>
      {(fold) => (
        <section class="emissions-strip" aria-labelledby="emissions-title">
          <div class="section-head">
            <div>
              <h2 id="emissions-title">King of the Hill emissions</h2>
              <p>Finalized scores nominate the champion; a statistical margin prevents churn.</p>
            </div>
            <span class="emission-badge champion">
              Champion {shortKey(fold().champion_miner_hotkey)}
            </span>
          </div>
          <dl class="emissions-metrics">
            <div>
              <dt>Champion pool</dt>
              <dd>{fold().champion_share == null ? "—" : pct(fold().champion_share as number)}</dd>
            </div>
            <div>
              <dt>Dethrone margin</dt>
              <dd>{fold().margin == null ? "—" : pct(fold().margin as number)}</dd>
            </div>
            <div>
              <dt>Confidence z</dt>
              <dd>{fold().dethrone_z ?? "—"}</dd>
            </div>
            <div>
              <dt>Tail recipients</dt>
              <dd>{fold().tail_size ?? fold().rank_shares?.length ?? "—"}</dd>
            </div>
          </dl>
          <div class="recipient-list" aria-label="Emission recipients">
            <For each={fold().recipients || []}>
              {(recipient) => (
                <span class={`recipient ${recipient.role || "tail"}`}>
                  {recipient.role === "champion" ? "Crown" : "Tail"} ·{" "}
                  {shortKey(recipient.miner_hotkey)} ·{" "}
                  {recipient.share_of_miner_pool == null ? "—" : pct(recipient.share_of_miner_pool)}
                </span>
              )}
            </For>
          </div>
          <Show
            when={
              fold().raw_leader_agent_id && fold().raw_leader_agent_id !== fold().champion_agent_id
            }
          >
            <p class="leaderboard-notice show">
              The raw leader has not cleared the configured dethrone band, so the incumbent remains
              champion.
            </p>
          </Show>
        </section>
      )}
    </Show>
  );
}
