// The entity modal shell (monolith 3013–3037) and its open/close/focus logic
// (showModal 5980–6004, trapFocus 6008–6017, closeModal 6338–6367), driven by
// the routeStore's entityRoute. Three tenants share the one dialog:
//   miner   — leaderboard run summary (openModal 5845–5978, summarized);
//   validator — signed heartbeat report (renderValidatorDetail 8687–8814,
//               summarized; the operations port supplies the deep body);
//   agent   — the submission drawer; this file renders the section skeleton
//             (AgentEvidenceSlot) that the submissions/reviews port fills.
// Screener routes never open the modal: the monolith highlights the fleet
// row on the operations page instead (resolveEntityRoute 9303–9399), which
// is that page's job.
import {
  Match,
  Show,
  Switch,
  createEffect,
  createSignal,
  onCleanup,
  onMount,
  untrack,
} from "solid-js";
import type { JSX } from "solid-js";

import { getJSON } from "../lib/api";
import { entityActions } from "../lib/entity-links";
import { agentName, agentVersionLabel, fx, pct, relTime } from "../lib/format";
import { entityHref } from "../lib/router";
import type { EntityKind, EntityRoute } from "../lib/router";
import {
  COMPOSITE_CALC_HEADING,
  COMPOSITE_CALC_NOTE,
  compositeCalculationRows,
  displayComposite,
  isEligible,
  isFinalized,
  unrankedKind,
} from "../lib/scoring";
import type { ContinualAggregate } from "../lib/scoring";
import { closeEntityRoute, currentPage, entityRoute, syncFromLocation } from "../stores/routeStore";
import type { FleetEntry, OperationsPayload } from "../types/fleet";
import type { LeaderboardEntry } from "../types/leaderboard";
import type { ActivityEntry, ActivityPayload } from "../types/pipeline";
import { CopyButton } from "./shell/CopyButton";
import { EntityButton } from "./ui/EntityButton";
import { StatusChip } from "./ui/StatusChip";

type RankedEntry = LeaderboardEntry & { rank?: number | null };

// Heartbeat fields the fleet status reads that are beyond the base wire type.
interface FleetStatusFields {
  bench_serviceability?: string | null;
  scorer_liveness?: string | null;
  allowed_slots?: number | null;
}
type ValidatorEntry = FleetEntry & FleetStatusFields;

// ── Fleet status (port of fleetStatus 8305–8328 + offlineAware 8334–8338) ──
// Exported so the operations page port can share one verdict per validator
// instead of growing a second copy.

export function fleetStatus(entry: ValidatorEntry): [string, string] {
  // Software that cannot describe the benchmark being scored comes first;
  // then the scorer that is down (the cause) before the bench gate (the
  // consequence). All three mean no lease completes.
  if (entry.bench_serviceability === "software_obsolete") return ["Obsolete build", "bad"];
  if (entry.scorer_liveness === "not_serving") return ["Scorer down", "bad"];
  if (entry.bench_serviceability === "scorer_unverified") return ["Bench unsupported", "bad"];
  if (entry.health === "critical") return ["Critical", "bad"];
  if (entry.assignment_state === "assignment_mismatch") return ["Mismatch", "bad"];
  if (entry.assignment_state === "heartbeat_stale") return ["Heartbeat stale", "warn"];
  if (entry.availability === "stale") return ["Stale", "warn"];
  if (entry.availability === "offline") return ["Offline", "bad"];
  if (entry.availability === "paused") return ["Paused", "paused"];
  if (entry.health === "warning") return ["Warning", "warn"];
  if (entry.health === "healthy") return ["Healthy", "good"];
  return ["Not reported", "unknown"];
}

/** For a node that is already offline a liveness badge only restates that it
 * is gone in the gentler warn tone; name it offline. A real fault keeps its
 * own label. */
export function offlineAwareFleetStatus(entry: ValidatorEntry): [string, string] {
  const status = fleetStatus(entry);
  if (entry.availability === "offline" && status[1] !== "bad") return ["Offline", "bad"];
  return status;
}

// ── Shared row / section helpers ────────────────────────────────────────────

function Stat(props: { k: string; v: JSX.Element; mono?: boolean }): JSX.Element {
  return (
    <div class="stat-row">
      <span class="k">{props.k}</span>
      <span class={"v" + (props.mono ? " mono" : "")}>{props.v}</span>
    </div>
  );
}

function FleetTime(props: { iso: string | null | undefined }): JSX.Element {
  return (
    <Show when={props.iso} fallback={<>–</>}>
      {(iso) => (
        <span class="fleet-time" title={iso()}>
          {relTime(iso())}
        </span>
      )}
    </Show>
  );
}

// ── Agent tenant: the deep-evidence container the submissions port fills ────

/**
 * Placeholder body for the agent drawer: the pipeline-summary /
 * pipeline-history section skeleton with the ledger ids
 * (pipeline-current-title, pipeline-meta-title, pipeline-screening-history,
 * pipeline-accepted-scores, pipeline-confirmation-scores,
 * pipeline-validator-history). The submissions/reviews port replaces each
 * section's state paragraph with the real evidence renderers.
 */
export function AgentEvidenceSlot(props: { entry: ActivityEntry }): JSX.Element {
  return (
    <div class="pipeline-detail" data-agent-evidence={props.entry.agent_id}>
      <div class="pipeline-summary">
        <section class="pipeline-current" aria-labelledby="pipeline-current-title">
          <h4 id="pipeline-current-title">Current progress</h4>
          <p class="pipeline-detail-state loading" role="status">
            Loading screening and validation history…
          </p>
        </section>
        <section class="pipeline-meta" aria-labelledby="pipeline-meta-title">
          <h4 id="pipeline-meta-title">Submission details</h4>
          <dl class="pipeline-meta-list">
            <div>
              <dt>Agent</dt>
              <dd>{agentName(props.entry.name)}</dd>
            </div>
            <div>
              <dt>Submission</dt>
              <dd>{agentVersionLabel(props.entry.version)}</dd>
            </div>
            <div>
              <dt>Agent ID</dt>
              <dd>
                <span class="copyable">
                  <code>
                    <EntityButton
                      kind="agent"
                      id={props.entry.agent_id}
                      label={String(props.entry.agent_id || "")}
                    />
                  </code>
                  <CopyButton value={props.entry.agent_id} label="agent ID" />
                </span>
              </dd>
            </div>
          </dl>
        </section>
      </div>
      <div class="pipeline-history">
        <section class="pipeline-section" aria-labelledby="pipeline-screening-history">
          <div class="pipeline-section-heading">
            <h4 id="pipeline-screening-history">Screener result</h4>
          </div>
          <div class="attempt-list" />
        </section>
        <section class="pipeline-section" aria-labelledby="pipeline-accepted-scores">
          <div class="pipeline-section-heading">
            <h4 id="pipeline-accepted-scores">Accepted validator scores</h4>
          </div>
        </section>
        <section class="pipeline-section" aria-labelledby="pipeline-confirmation-scores">
          <div class="pipeline-section-heading">
            <h4 id="pipeline-confirmation-scores">Continual top-five retests</h4>
          </div>
        </section>
        <section class="pipeline-section" aria-labelledby="pipeline-validator-history">
          <div class="pipeline-section-heading">
            <h4 id="pipeline-validator-history">Validator progress</h4>
          </div>
          <div class="benchmark-cohort-list" />
        </section>
      </div>
    </div>
  );
}

// Stage labels for the agent tenant's header chip (the submissions port owns
// the full vocabulary; the shell needs the [label, tone] pair only).
const AGENT_STAGE: Record<string, [string, string]> = {
  uploaded: ["Waiting for screening", "progress"],
  waiting_screening: ["Waiting for screening", "progress"],
  screening: ["Screening", "progress"],
  screening_passed: ["Screening passed", "good"],
  screening_failed: ["Screening interrupted", "warn"],
  waiting_validator: ["Waiting for scores", "progress"],
  evaluating: ["Evaluating", "progress"],
  below_score_floor: ["Low-priority completion", "warn"],
  not_queued: ["Historical · not queued", ""],
  retired: ["Retired · earlier benchmark", ""],
  scored: ["Scored", "good"],
  live: ["Live", "good"],
  under_review: ["Operator review", "warn"],
  rejected: ["Rejected", "bad"],
};

// ── The panel itself ─────────────────────────────────────────────────────────

type PanelView =
  | { tenant: "miner"; key: string; entry: RankedEntry }
  | { tenant: "validator"; key: string; hotkey: string; entry: ValidatorEntry }
  | { tenant: "agent"; key: string; entry: ActivityEntry }
  | { tenant: "agent-state"; key: string; id: string; message: string; state: "loading" | "error" };

export interface EntityPanelProps {
  /** Ranked leaderboard entries (last successful payload, display order). */
  entries: () => RankedEntry[];
  operations: () => OperationsPayload | undefined;
  /** Optional display names keyed by validator hotkey — untrusted decoration
   * from a separate feed; the hotkey stays the anchor identity. */
  validatorNames: () => Record<string, string>;
  /** The settled/current bench version, for the miner bench chip. */
  currentBench: () => number | null;
  /** Mid-rollout settled view (affects the displayed composite). */
  settledView?: () => boolean;
}

// The URL half of closing the overlay. Dedicated entity pages (/agent/{id})
// own their URL, so closing is a no-op there (closeModal 6341–6342).
function close(): void {
  const entity = entityRoute();
  if (entity && entity.full) return;
  closeEntityRoute();
}

export function EntityPanel(props: EntityPanelProps): JSX.Element {
  const [view, setView] = createSignal<PanelView | null>(null);
  const [full, setFull] = createSignal(false);

  let modalEl: HTMLElement | undefined;
  let lastFocused: Element | null = null;
  let resolvingKey = "";

  const isOpen = () => view() !== null;
  const settled = () => (props.settledView ? props.settledView() : false);

  function resolveAgent(route: EntityRoute): void {
    const key = route.key;
    const current = untrack(view);
    if (current && current.key === key && current.tenant === "agent") return;
    if (resolvingKey === key) return;
    resolvingKey = key;
    if (route.full) {
      setView({
        tenant: "agent-state",
        key,
        id: route.id,
        message: "Loading submission details…",
        state: "loading",
      });
    }
    getJSON<ActivityPayload>("/public/activity?page=1&limit=1&q=" + encodeURIComponent(route.id))
      .then((data) => {
        const now = entityRoute();
        if (!now || now.key !== key) return;
        const entry = (data.entries || []).find((item) => String(item.agent_id) === route.id);
        if (entry) {
          setView({ tenant: "agent", key, entry });
        } else if (route.full) {
          setView({
            tenant: "agent-state",
            key,
            id: route.id,
            message: "This submission could not be found.",
            state: "error",
          });
        }
      })
      .catch(() => {
        const now = entityRoute();
        if (route.full && now && now.key === key) {
          setView({
            tenant: "agent-state",
            key,
            id: route.id,
            message: "Submission details are temporarily unavailable. Try refreshing in a moment.",
            state: "error",
          });
        }
      })
      .finally(() => {
        if (resolvingKey === key) resolvingKey = "";
      });
  }

  createEffect(() => {
    const route = entityRoute();
    if (!route) {
      setView(null);
      setFull(false);
      return;
    }
    // Legacy URL forms (real-query params, plural hash/path routes) are
    // recognized once and normalized to the canonical hash-query form.
    if (route.legacy) {
      history.replaceState((history.state as unknown) ?? {}, "", entityHref(route.kind, route.id));
      syncFromLocation();
      return;
    }
    setFull(route.full);
    if (route.kind === "validator" || route.kind === "screener") {
      // Fleet-row targets live on the operations page only; normalize there.
      // currentPage is deliberately tracked: the store updates the entity
      // and page signals in sequence, so this effect may observe the entity
      // before the page has caught up and must re-run when it does.
      if (currentPage() !== "operations") {
        history.replaceState(
          (history.state as unknown) ?? {},
          "",
          entityHref(route.kind, route.id, "operations"),
        );
        syncFromLocation();
        return;
      }
      if (route.kind === "validator") {
        const report = props.operations()?.validators;
        const entry = (report?.validators || []).find(
          (item) => item.validator_hotkey === route.id,
        ) as ValidatorEntry | undefined;
        if (entry) setView({ tenant: "validator", key: route.key, hotkey: route.id, entry });
      }
      // Screeners: the operations page highlights the fleet row; no modal.
      return;
    }
    if (route.kind === "miner") {
      const entry = props.entries().find((item) => item.miner_hotkey === route.id);
      if (entry) setView({ tenant: "miner", key: route.key, entry });
      return;
    }
    resolveAgent(route);
  });

  // Open/close side effects: focus capture + restore, background inert, the
  // full-page body mode, scroll reset.
  createEffect(() => {
    const current = view();
    const fullPage = full();
    const wrap = document.querySelector(".wrap") as (HTMLElement & { inert?: boolean }) | null;
    if (current) {
      document.body.classList.toggle("entity-page", fullPage);
      if (!lastFocused) lastFocused = document.activeElement;
      if (wrap) wrap.inert = !fullPage;
      if (fullPage) {
        try {
          window.scrollTo(0, 0);
        } catch {
          // jsdom has no layout; scroll reset is cosmetic there.
        }
      }
      if (modalEl) modalEl.scrollTop = 0;
      const target = fullPage
        ? document.getElementById("d-back-dashboard")
        : document.getElementById("modal-close");
      target?.focus();
    } else {
      document.body.classList.remove("entity-page");
      if (wrap) wrap.inert = false;
      if (lastFocused instanceof HTMLElement) lastFocused.focus();
      lastFocused = null;
    }
  });

  // Keep Tab focus inside the open dialog (a lightweight focus trap).
  function trapFocus(ev: KeyboardEvent): void {
    if (ev.key !== "Tab" || !modalEl || !isOpen()) return;
    const nodes = modalEl.querySelectorAll<HTMLElement>(
      'a[href], button, [tabindex]:not([tabindex="-1"]), summary, input, textarea, [contenteditable]',
    );
    const focusable = Array.from(nodes).filter(
      (el) => el.offsetParent !== null || el === document.activeElement,
    );
    if (!focusable.length) return;
    const first = focusable[0] as HTMLElement;
    const last = focusable[focusable.length - 1] as HTMLElement;
    if (ev.shiftKey && document.activeElement === first) {
      ev.preventDefault();
      last.focus();
    } else if (!ev.shiftKey && document.activeElement === last) {
      ev.preventDefault();
      first.focus();
    }
  }

  onMount(() => {
    const onKeyDown = (ev: KeyboardEvent): void => {
      if (ev.key === "Escape") close();
      else trapFocus(ev);
    };
    document.addEventListener("keydown", onKeyDown);
    onCleanup(() => document.removeEventListener("keydown", onKeyDown));
  });

  // ── Header derivations per tenant ─────────────────────────────────────────

  const actionKind = (): EntityKind | null => {
    const current = view();
    if (!current) return null;
    if (current.tenant === "miner") return "miner";
    if (current.tenant === "validator") return "validator";
    return "agent";
  };
  const actionId = (): string => {
    const current = view();
    if (!current) return "";
    if (current.tenant === "miner") return current.entry.miner_hotkey;
    if (current.tenant === "validator") return current.hotkey;
    if (current.tenant === "agent") return String(current.entry.agent_id || "");
    return current.id;
  };
  const actions = () => {
    const kind = actionKind();
    return kind ? entityActions(kind, actionId()) : null;
  };

  interface HeaderState {
    title: string;
    chip: { text: string; class: string; title: string };
    dkLabel: string;
    hotkey: string | null;
    hotkeyKind: EntityKind;
  }

  const header = (): HeaderState | null => {
    const current = view();
    if (!current) return null;
    if (current.tenant === "miner") {
      const e = current.entry;
      const title = isFinalized(e)
        ? "Raw score rank #" +
          e.rank +
          (e._emission && e._emission.role === "champion" ? " · KOTH champion" : "")
        : "Provisional rank P" +
          e.rank +
          " · " +
          (e.score_count || 0) +
          " of " +
          (e.score_quorum || 3) +
          " scores";
      return {
        title,
        chip: minerBenchChip(e, props.currentBench()),
        dkLabel: "Miner",
        hotkey: e.miner_hotkey,
        hotkeyKind: "miner",
      };
    }
    if (current.tenant === "validator") {
      const status = offlineAwareFleetStatus(current.entry);
      // Display names are optional untrusted decoration from a separate
      // feed; the hotkey below stays the anchor identity.
      return {
        title: props.validatorNames()[current.hotkey] || "Validator",
        chip: { text: status[0], class: status[1], title: "Current fleet status" },
        dkLabel: "Validator",
        hotkey: current.hotkey,
        hotkeyKind: "validator",
      };
    }
    if (current.tenant === "agent") {
      const e = current.entry;
      const stage = (e.status !== undefined && AGENT_STAGE[e.status]) || ["Pending", ""];
      return {
        title: agentName(e.name),
        chip: {
          text: stage[0] as string,
          class: stage[1] as string,
          title: "Current submission stage",
        },
        dkLabel: "Miner",
        hotkey: e.miner_hotkey || null,
        hotkeyKind: "miner",
      };
    }
    return {
      title: "Agent submission",
      chip:
        current.state === "error"
          ? { text: "Unavailable", class: "error", title: "Submission unavailable" }
          : { text: "Loading", class: "loading", title: "Loading submission" },
      dkLabel: "Miner",
      hotkey: null,
      hotkeyKind: "miner",
    };
  };

  const minerEntry = () => {
    const current = view();
    return current && current.tenant === "miner" ? current.entry : null;
  };
  const validatorView = () => {
    const current = view();
    return current && current.tenant === "validator" ? current : null;
  };
  const agentView = () => {
    const current = view();
    return current && current.tenant === "agent" ? current : null;
  };
  const agentStateView = () => {
    const current = view();
    return current && current.tenant === "agent-state" ? current : null;
  };
  const toolMean = () => minerEntry()?.tool_mean ?? 0;
  const memoryMean = () => minerEntry()?.memory_mean ?? 0;
  const toolShare = () => {
    const e = minerEntry();
    if (!e) return 50;
    const sum = e.tool_mean + e.memory_mean || 1;
    return (e.tool_mean / sum) * 100;
  };

  return (
    <>
      <div
        id="modal-back"
        class="modal-back"
        classList={{ open: isOpen() && !full() }}
        onClick={() => close()}
      />
      <aside
        id="modal"
        class="modal"
        classList={{ open: isOpen(), "full-page": full() }}
        role={full() ? "main" : "dialog"}
        aria-modal={full() ? "false" : "true"}
        aria-hidden={isOpen() ? "false" : "true"}
        aria-labelledby="d-title"
        tabindex="-1"
        ref={(el) => {
          modalEl = el;
        }}
      >
        <div class="modal-actions">
          <a
            class="btn ghost back-dashboard"
            id="d-back-dashboard"
            href={actions()?.backHref ?? "/#/overview"}
          >
            ← Dashboard
          </a>
          <a
            class="btn ghost open-full"
            id="d-open-full"
            href={actions()?.openFullHref ?? "/"}
            style={{ display: actions()?.openFullHref ? "" : "none" }}
          >
            Open full page
          </a>
          <button
            class="btn ghost close"
            id="modal-close"
            aria-label="Close detail"
            onClick={() => close()}
          >
            <span aria-hidden="true">✕</span>
          </button>
        </div>
        <div class="dhead">
          <h3 id="d-title">{header()?.title ?? "Miner"}</h3>
          <span id="d-bench" class={header()?.chip.class ?? ""} title={header()?.chip.title ?? ""}>
            {header()?.chip.text ?? ""}
          </span>
        </div>
        <div class="dk">
          <span class="dk-label">{header()?.dkLabel ?? "Miner"}</span>
          <span class="copyable">
            <span id="d-hotkey">
              <Show when={header()?.hotkey}>
                {(hotkey) => (
                  <EntityButton
                    kind={header()?.hotkeyKind ?? "miner"}
                    id={hotkey()}
                    label={hotkey()}
                  />
                )}
              </Show>
            </span>
            <CopyButton id="d-hotkey-copy" value={header()?.hotkey || null} label="miner hotkey" />
          </span>
        </div>
        <div class="split" style={{ display: minerEntry() ? "" : "none" }}>
          <div class="seg">
            <div id="d-tool-seg" style={{ background: "var(--tool)", width: toolShare() + "%" }}>
              {toolShare() > 14 ? fx(toolMean()) : ""}
            </div>
            <div
              id="d-mem-seg"
              style={{ background: "var(--memory)", width: 100 - toolShare() + "%" }}
            >
              {100 - toolShare() > 14 ? fx(memoryMean()) : ""}
            </div>
          </div>
          <div class="legend">
            <span>
              <i style={{ background: "var(--tool)" }} />
              Tool{" "}
              <span id="d-tool-pct" class="muted">
                {minerEntry() ? fx(toolMean()) : ""}
              </span>
            </span>
            <span>
              <i style={{ background: "var(--memory)" }} />
              Memory{" "}
              <span id="d-mem-pct" class="muted">
                {minerEntry() ? fx(memoryMean()) : ""}
              </span>
            </span>
          </div>
        </div>
        <div id="d-stats" classList={{ "pipeline-mode": view()?.tenant !== "miner" }}>
          <Switch>
            <Match when={minerEntry()}>
              {(entry) => (
                <MinerSummary
                  entry={entry()}
                  settled={settled()}
                  total={props.entries().filter(isEligible).length}
                />
              )}
            </Match>
            <Match when={validatorView()}>
              {(v) => <ValidatorSummary entry={v().entry} activeBench={props.currentBench()} />}
            </Match>
            <Match when={agentView()}>{(v) => <AgentEvidenceSlot entry={v().entry} />}</Match>
            <Match when={agentStateView()}>
              {(v) => (
                <div class="pipeline-detail">
                  <p class={"pipeline-detail-state " + v().state} role="status">
                    {v().message}
                  </p>
                </div>
              )}
            </Match>
          </Switch>
        </div>
      </aside>
    </>
  );
}

// ── Miner tenant summary ─────────────────────────────────────────────────────

function minerBenchChip(
  e: RankedEntry,
  currentBench: number | null,
): { text: string; class: string; title: string } {
  if (e.bench_version == null) {
    return isFinalized(e)
      ? {
          text: "legacy",
          class: "prev",
          title:
            "Scored before benchmark versioning. A legacy run, not comparable to " +
            (currentBench ? "the current DittoBench v" + currentBench : "the current benchmark") +
            ".",
        }
      : {
          text: "pending quorum",
          class: "",
          title: "Run provenance appears after the three-validator aggregate is final.",
        };
  }
  const settledVersion = currentBench;
  const old = settledVersion !== null && e.bench_version < settledVersion;
  return {
    text: "DittoBench v" + e.bench_version + (old ? " · old" : ""),
    class: old ? "prev" : "",
    title: old
      ? "Scored on DittoBench v" +
        e.bench_version +
        ", a previous benchmark. Not directly comparable to the settled v" +
        settledVersion +
        "."
      : "Scored on DittoBench v" + e.bench_version + ".",
  };
}

function MinerSummary(props: { entry: RankedEntry; settled: boolean; total: number }): JSX.Element {
  const e = () => props.entry;
  const agg = () => e() as RankedEntry & ContinualAggregate;
  const official = () => displayComposite(e(), props.settled);
  const rolling = () => agg().aggregate_method === "continual_mean";
  const kind = () => unrankedKind(e());
  const calcRows = () => compositeCalculationRows(e());
  return (
    <>
      <div class="stat-cols">
        <div class="stat-group">
          <div class="stat-head">Overview</div>
          <Stat k="Best-scoring agent" v={agentName(e().agent_name)} />
          <Stat k="Submission" v={agentVersionLabel(e().agent_version)} />
          <Stat
            k="Current leaderboard score"
            v={
              fx(official()) +
              (rolling()
                ? "  · mean of " + agg().aggregate_sample_count + " scores"
                : (e().composite_stderr != null
                    ? "  ± " + fx(e().composite_stderr as number) + " SE"
                    : "") + "  · canonical quorum median")
            }
          />
          <Stat k="Tool mean" v={fx(e().tool_mean) + "  (" + pct(e().tool_mean) + ")"} />
          <Stat k="Memory mean" v={fx(e().memory_mean) + "  (" + pct(e().memory_mean) + ")"} />
          <Show when={e().first_seen}>
            {(seen) => <Stat k="First seen" v={new Date(seen()).toLocaleString()} />}
          </Show>
          <Stat
            k="Rank"
            v={
              isEligible(e())
                ? "#" + e().rank + " of " + props.total
                : kind() === "zero"
                  ? "unranked (scored 0.000)"
                  : "unranked (provisional)"
            }
          />
        </div>
        <Show when={calcRows()}>
          {(rows) => (
            <div class="stat-group">
              <div class="stat-head">{COMPOSITE_CALC_HEADING}</div>
              {rows().map((row) => (
                <Stat k={row.k} v={row.v} />
              ))}
              <p class="calc-note">{COMPOSITE_CALC_NOTE}</p>
            </div>
          )}
        </Show>
      </div>
      <div id="d-consensus" />
      <div class="gloss-link">
        <a href="#/benchmark">What each category and metric means →</a>
      </div>
    </>
  );
}

// ── Validator tenant summary ────────────────────────────────────────────────

function ValidatorSummary(props: {
  entry: ValidatorEntry;
  activeBench: number | null;
}): JSX.Element {
  const e = () => props.entry;
  const status = () => offlineAwareFleetStatus(e());
  const scoredLabel = () =>
    props.activeBench ? "bench v" + props.activeBench : "the scored benchmark";
  const slotSummary = () => {
    // Advertised, healthy and funded are three different numbers, and the
    // one that decides whether a slot gets work is the last of them.
    let summary =
      String((e().healthy_slots || []).length) + " healthy of " + String(e().configured_slots || 1);
    if (isFinite(Number(e().allowed_slots))) {
      summary += " · " + String(e().allowed_slots) + " funded by the operator cap";
    }
    return summary + " · " + String(e().admission || "accepting");
  };
  return (
    <div class="vdetail">
      <details class="cgroup" open>
        <summary class="cgsum">Signed report</summary>
        <div style={{ padding: "2px 0 10px 22px" }}>
          <Stat k="Fleet status" v={<StatusChip label={status()[0]} tone={status()[1]} />} />
          <Show when={e().bench_serviceability && e().bench_serviceability !== "serving"}>
            <Stat
              k="Benchmark eligibility"
              v={
                <StatusChip
                  tone="bad"
                  label={
                    e().bench_serviceability === "software_obsolete"
                      ? "Cannot serve " + scoredLabel() + " · needs a software upgrade"
                      : "Scorer is not advertising " + scoredLabel()
                  }
                />
              }
            />
          </Show>
          <Stat k="Worker state" v={e().state || "unknown"} />
          <Stat k="Software version" v={e().software_version || "Unknown"} />
          <Stat k="Heartbeat protocol" v={String(e().protocol_version)} />
          <Stat k="First seen" v={<FleetTime iso={e().first_seen_at} />} />
          <Stat k="Validator reported" v={<FleetTime iso={e().reported_at} />} />
          <Stat k="Platform received" v={<FleetTime iso={e().seen_at} />} />
          <Stat k="Slots" v={slotSummary()} />
        </div>
      </details>
      <details class="cgroup">
        <summary class="cgsum">Host metrics</summary>
        <div style={{ padding: "2px 0 10px 22px" }}>
          <Show
            when={e().system_metrics}
            fallback={<Stat k="Host metrics" v={<span class="muted">Not reported</span>} />}
          >
            {(m) => (
              <>
                <Stat k="CPU" v={m().cpu_percent + "%"} />
                <Stat k="Memory" v={m().memory_percent + "%"} />
                <Stat k="Disk" v={m().disk_percent + "%"} />
                <Stat
                  k="Docker"
                  v={
                    m().docker_status +
                    " · " +
                    m().running_containers +
                    " running, " +
                    m().unhealthy_containers +
                    " unhealthy"
                  }
                />
              </>
            )}
          </Show>
        </div>
      </details>
    </div>
  );
}
