// Parity tests for the validator modal's three stack sections — Capabilities
// (monolith 9101–9114), Stack identity (9116–9126) and Component health
// (9128–9139 rendered at 9160–9162), with the helpers they lean on:
// renderStackComponent 8959–8990, renderScorerBenchmarks 8992–9011,
// renderScorerProbe 9013–9035, identityRows 8933–8940 and
// identityComparisonNote 8942–8957.
//
// These sections escaped BOTH gates of the SPA port: the modal only opens on
// interaction, so the per-page DOM goldens never contained it, and the old
// Python suites barely asserted it. Nothing here is incidental — every row is
// an operator answering "which half of this validator is lying to me", so the
// tests pin the copy, the collapsed/open state and the order, not just
// presence.
//
// Frozen clock 2026-07-31T14:00:00Z, the golden renderer's instant, so the
// relative probe times are stable.
import { cleanup, render } from "@solidjs/testing-library";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { rankEntries } from "../lib/scoring";
import { syncFromLocation } from "../stores/routeStore";
import { installFixtureFetch, loadFixture } from "../test-fixtures";
import type { FleetEntry, OperationsPayload } from "../types/fleet";
import type { LeaderboardPayload } from "../types/leaderboard";
import { EntityPanel } from "./EntityPanel";

const leaderboard = loadFixture<LeaderboardPayload>("leaderboard");
const entries = rankEntries(leaderboard.entries ?? []);
const operations = loadFixture<OperationsPayload>("operations");
const validatorRows = operations.validators.validators ?? [];

const hotkeyOf = (prefix: string): string =>
  String(
    validatorRows.find((v) => String(v.validator_hotkey).startsWith(prefix))?.validator_hotkey,
  );

/** Protocol 18, managed signed release, fresh identity-verified scorer, five
 * of six components observed (model_relay and ollama absent on both sides). */
const MANAGED = hotkeyOf("5HmP9732");
/** Protocol 18 source build: no release descriptor digest. */
const SOURCE = hotkeyOf("5CqJAjSj");
/** Protocol 15: legacy v2-only scorer, a probe that never served, and all six
 * components observed (ollama reports an embedding model). */
const LEGACY = hotkeyOf("5FU3YKmv");
/** Protocol 6: no capabilities, no stack, no per-component health at all. */
const ANCIENT = hotkeyOf("5HKpbkeL");

const COMPONENT_LABELS = [
  "Validator worker",
  "Scorer · dittobench-api",
  "Sandbox Docker",
  "Model relay",
  "Pylon",
  "Ollama",
];

let restoreFetch: (() => void) | null = null;

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(new Date("2026-07-31T14:00:00Z"));
  history.replaceState(null, "", "/#/operations");
  syncFromLocation();
  restoreFetch = installFixtureFetch();
});

afterEach(() => {
  cleanup();
  restoreFetch?.();
  restoreFetch = null;
  vi.useRealTimers();
  document.body.classList.remove("entity-page");
});

/** A deep copy of the recorded snapshot with one validator patched, so a
 * synthetic shape never leaks into the next test. */
function patched(hotkey: string, patch: Partial<FleetEntry>): OperationsPayload {
  const payload = structuredClone(operations);
  const rows = payload.validators.validators ?? [];
  const index = rows.findIndex((row) => row.validator_hotkey === hotkey);
  rows[index] = { ...(rows[index] as FleetEntry), ...patch };
  return payload;
}

function open(hotkey: string, payload: OperationsPayload = operations): void {
  render(() => (
    <EntityPanel
      entries={() => entries}
      operations={() => payload}
      validatorNames={() => ({})}
      currentBench={() => 7}
      settledView={() => false}
    />
  ));
  history.replaceState(null, "", "/#/operations?validator=" + hotkey);
  syncFromLocation();
}

/** Top-level sections of the validator body, in render order. */
function sections(): HTMLElement[] {
  return Array.from(document.querySelectorAll<HTMLElement>("#d-stats .vdetail > details.cgroup"));
}

function sectionTitles(): string[] {
  return sections().map((el) => el.firstElementChild?.textContent ?? "");
}

function section(title: string): HTMLElement {
  const found = sections().find((el) => el.firstElementChild?.textContent === title);
  if (!found) throw new Error("missing section: " + title);
  return found;
}

/** One component's collapsible group inside Component health. */
function component(label: string): HTMLElement {
  const groups = Array.from(
    section("Component health").querySelectorAll<HTMLElement>("details.cgroup"),
  );
  const found = groups.find(
    (el) => el.querySelector("summary.cgsum > span")?.textContent === label,
  );
  if (!found) throw new Error("missing component: " + label);
  return found;
}

/** Stat rows directly inside a scope (never a nested component group). */
function rows(scope: ParentNode): HTMLElement[] {
  return Array.from(scope.querySelectorAll<HTMLElement>(".stat-row"));
}

function rowKeys(scope: ParentNode): string[] {
  return rows(scope).map((el) => el.querySelector(".k")?.textContent ?? "");
}

function row(scope: ParentNode, key: string): HTMLElement {
  const found = rows(scope).find((el) => el.querySelector(".k")?.textContent === key);
  if (!found) throw new Error("missing row: " + key);
  return found;
}

function value(scope: ParentNode, key: string): string {
  return row(scope, key).querySelector(".v")?.textContent ?? "";
}

function chipOf(scope: ParentNode, key: string): [string, string] {
  const chip = row(scope, key).querySelector(".stage");
  return [chip?.textContent ?? "", chip?.className ?? ""];
}

function subheads(scope: ParentNode): string[] {
  return Array.from(scope.querySelectorAll<HTMLElement>(".subhead")).map(
    (el) => el.textContent ?? "",
  );
}

describe("validator modal section order", () => {
  it("renders all five signed-report sections, Component health among them", () => {
    open(MANAGED);
    expect(sectionTitles()).toEqual([
      "Signed report",
      "Capabilities",
      "Stack identity",
      "Component health",
      "Host metrics",
    ]);
    // Signed report and Component health answer the two questions an operator
    // opens this modal with; the other three are one click away.
    expect(section("Signed report").hasAttribute("open")).toBe(true);
    expect(section("Component health").hasAttribute("open")).toBe(true);
    expect(section("Capabilities").hasAttribute("open")).toBe(false);
    expect(section("Stack identity").hasAttribute("open")).toBe(false);
    expect(section("Host metrics").hasAttribute("open")).toBe(false);
  });
});

describe("validator modal · Capabilities (9101–9114)", () => {
  it("reports every capability flag plus the scorer's own probe", () => {
    open(MANAGED);
    const caps = section("Capabilities");
    expect(value(caps, "Screened images")).toBe("Yes");
    expect(value(caps, "Requires screened image")).toBe("Yes");
    expect(value(caps, "Source-build fallback")).toBe("No");
    expect(value(caps, "Managed full stack")).toBe("Yes");
    expect(value(caps, "Stack auto-updater")).toBe("Yes");
    expect(value(caps, "Sandbox egress restricted")).toBe("Yes");
    expect(value(caps, "Executor isolation")).toBe("privileged_dind");
    // The scorer block, in the monolith's order: verdict, then the probe that
    // produced it, then what the scorer says it can serve.
    expect(rowKeys(caps)).toEqual([
      "Screened images",
      "Requires screened image",
      "Source-build fallback",
      "Managed full stack",
      "Stack auto-updater",
      "Sandbox egress restricted",
      "Executor isolation",
      "Scorer status",
      "Scorer probe",
      "Probe observed",
      "Last served",
      "Supported benchmarks",
      "Capability observed",
      "Scorer version",
      "Scorer revision",
    ]);
    expect(chipOf(caps, "Scorer status")).toEqual(["Fresh, identity-verified", "stage good"]);
    expect(chipOf(caps, "Scorer probe")).toEqual(["Serving", "stage good"]);
    expect(value(caps, "Scorer probe")).toBe("Serving · HTTP 200");
    expect(value(caps, "Supported benchmarks")).toBe("v7");
    expect(value(caps, "Scorer version")).toBe("0.41.0");
    // Probe times are epoch seconds on the wire: relative text, exact instant
    // in the title.
    const observed = row(caps, "Probe observed").querySelector(".fleet-time");
    expect(observed).toHaveAttribute("title", "2026-07-31T11:48:41.000Z");
    expect(observed?.textContent).toBe("2h ago");
    expect(row(caps, "Last served").querySelector(".fleet-time")).toHaveAttribute(
      "title",
      "2026-07-31T11:48:41.000Z",
    );
  });

  it("gives the scorer revision the mono/copy treatment, elided but copyable", () => {
    open(MANAGED);
    const revision = row(section("Capabilities"), "Scorer revision");
    expect(revision.querySelector(".v")?.className).toBe("v mono");
    const copyable = revision.querySelector(".copyable");
    expect(copyable).toHaveAttribute("title", "25e8f296a573673ea47610c7a133e50660f2f416");
    expect(copyable?.querySelector("span")?.textContent).toBe("25e8f296a573…0660f2f416");
    const copy = revision.querySelector("button.copy");
    expect(copy).toHaveAttribute("data-key", "25e8f296a573673ea47610c7a133e50660f2f416");
    expect(copy).toHaveAttribute("data-copy-label", "scorer source revision");
  });

  it("reads a failing probe as evidence, not just a verdict", () => {
    open(LEGACY);
    const caps = section("Capabilities");
    expect(chipOf(caps, "Scorer status")).toEqual(["Legacy scorer (v2 only)", "stage warn"]);
    expect(chipOf(caps, "Scorer probe")).toEqual(["No usable answer", "stage bad"]);
    expect(value(caps, "Scorer probe")).toBe("No usable answer · HTTP 404 · 3284 in a row");
    // Never served is a different claim from "served a while ago".
    expect(value(caps, "Last served")).toBe("Not since this validator started");
    expect(value(caps, "Supported benchmarks")).toBe("v2");
    // Rows the payload has nothing for are absent, not blank or guessed.
    expect(rowKeys(caps)).not.toContain("Capability observed");
    expect(rowKeys(caps)).not.toContain("Scorer version");
    expect(rowKeys(caps)).not.toContain("Scorer revision");
  });

  it("names the missing heartbeat protocol instead of implying an unequipped validator", () => {
    open(ANCIENT);
    const caps = section("Capabilities");
    expect(rowKeys(caps)).toEqual(["Capabilities"]);
    expect(value(caps, "Capabilities")).toBe("Not reported (requires heartbeat protocol 7)");
    expect(row(caps, "Capabilities").querySelector(".muted")).toBeTruthy();
  });

  it("separates an unreported scorer (protocol 8) from an unreported probe (15)", () => {
    const capabilities = {
      ...validatorRows.find((v) => v.validator_hotkey === MANAGED)?.capabilities,
    };
    open(MANAGED, patched(MANAGED, { capabilities: { ...capabilities, scorer_benchmarks: null } }));
    expect(value(section("Capabilities"), "Scorer benchmarks")).toBe(
      "Not reported (requires heartbeat protocol 8)",
    );
    cleanup();

    const scorer = { ...capabilities.scorer_benchmarks, probe: null };
    open(
      MANAGED,
      patched(MANAGED, { capabilities: { ...capabilities, scorer_benchmarks: scorer } }),
    );
    const caps = section("Capabilities");
    expect(chipOf(caps, "Scorer status")).toEqual(["Fresh, identity-verified", "stage good"]);
    expect(value(caps, "Scorer probe")).toBe("Not reported (requires heartbeat protocol 15)");
    expect(rowKeys(caps)).not.toContain("Probe observed");
  });

  it("says nothing rather than No for a capability the heartbeat omitted", () => {
    open(MANAGED, patched(MANAGED, { capabilities: { screened_images: null } }));
    const caps = section("Capabilities");
    expect(value(caps, "Screened images")).toBe("Not reported");
    expect(value(caps, "Requires screened image")).toBe("Not reported");
    expect(value(caps, "Executor isolation")).toBe("unknown");
  });
});

describe("validator modal · Stack identity (9116–9126)", () => {
  it("names the signed managed release and pins the descriptor digest", () => {
    open(MANAGED);
    const stack = section("Stack identity");
    expect(rowKeys(stack)).toEqual(["Stack mode", "Compose schema", "Release descriptor"]);
    expect(value(stack, "Stack mode")).toBe("Managed (signed GHCR release)");
    expect(value(stack, "Compose schema")).toBe("1");
    const descriptor = row(stack, "Release descriptor");
    expect(descriptor.querySelector(".v")?.className).toBe("v mono");
    expect(descriptor.querySelector(".copyable")).toHaveAttribute(
      "title",
      "sha256:181dca5089981df874790b424123d59b65374fdf1cae49ea21a54ee47301e30a",
    );
    expect(descriptor.querySelector(".copyable > span")?.textContent).toBe(
      "sha256:181dc…e47301e30a",
    );
    expect(descriptor.querySelector("button.copy")).toHaveAttribute(
      "data-copy-label",
      "release descriptor digest",
    );
  });

  it("reads anything that is not the managed release as a source build", () => {
    open(SOURCE);
    const stack = section("Stack identity");
    expect(value(stack, "Stack mode")).toBe("Source build");
    // No signed descriptor exists for a source build; the row is omitted.
    expect(rowKeys(stack)).toEqual(["Stack mode", "Compose schema"]);
  });

  it("names the missing heartbeat protocol when no stack is reported", () => {
    open(ANCIENT);
    const stack = section("Stack identity");
    expect(rowKeys(stack)).toEqual(["Stack identity"]);
    expect(value(stack, "Stack identity")).toBe("Not reported (requires heartbeat protocol 7)");
  });
});

describe("validator modal · Component health (9128–9139, 9160–9162)", () => {
  it("lists the six components in a fixed order, whatever the payload holds", () => {
    open(MANAGED);
    const labels = Array.from(
      section("Component health").querySelectorAll<HTMLElement>("summary.cgsum > span:first-child"),
    ).map((el) => el.textContent ?? "");
    expect(labels).toEqual(COMPONENT_LABELS);
  });

  it("explains configured versus observed identity when health is present", () => {
    open(MANAGED);
    const intro = section("Component health").querySelector("p.muted") as HTMLElement;
    // Explanatory, not a row: the intro stays smaller than the stats it heads.
    expect(intro.style.fontSize).toBe("12px");
    expect(intro?.textContent).toBe(
      "Configured identity is what Compose intends to run; observed identity is what a live " +
        "probe independently verified; readiness is a real request answered just now. " +
        "Per-component probe times are independent of heartbeat freshness.",
    );
  });

  it("explains the protocol gap instead, when health is absent", () => {
    open(ANCIENT);
    const health = section("Component health");
    expect(health.querySelector("p.muted")?.textContent).toBe(
      "This validator reports heartbeat protocol 6. Per-component runtime health arrives with " +
        "protocol 9.",
    );
    // Still six groups: the components are named even when nothing observed
    // them, so the absence is legible rather than an empty section.
    expect(health.querySelectorAll("details.cgroup").length).toBe(6);
    for (const label of COMPONENT_LABELS) {
      expect(value(component(label), "Health")).toBe(
        "Not reported (requires heartbeat protocol 9)",
      );
      expect(component(label).querySelector("summary .stage")?.textContent).toBe("Unknown");
      expect(subheads(component(label))).toEqual([]);
    }
  });

  it("shows an observed component's readiness, probe time and both identities", () => {
    open(MANAGED);
    const worker = component("Validator worker");
    expect(worker.hasAttribute("open")).toBe(false);
    expect(worker.querySelector("summary .stage")?.textContent).toBe("Healthy");
    expect(worker.querySelector("summary .stage")?.className).toBe("stage good");
    // Probe freshness is per component and independent of the heartbeat, so
    // the summary carries its own time.
    const probed = worker.querySelector("summary .probe-time");
    expect(probed?.textContent).toBe("probed 2h ago");
    expect(probed?.querySelector(".fleet-time")).toHaveAttribute(
      "title",
      "2026-07-31T11:48:41.000Z",
    );
    // Observed identity first (a version only — the probe verified no digest),
    // then the configured pin, which is the whole point of the pairing.
    expect(rowKeys(worker)).toEqual([
      "Health",
      "Required component",
      "Endpoint ready",
      "Last probe",
      "Version",
      "Provenance",
      "Image digest",
      "Source revision",
      "Version",
    ]);
    expect(chipOf(worker, "Health")).toEqual(["Healthy", "stage good"]);
    expect(value(worker, "Required component")).toBe("Yes");
    expect(value(worker, "Endpoint ready")).toBe("Yes");
    expect(subheads(worker)).toEqual(["Observed identity", "Configured identity"]);
    expect(value(worker, "Version")).toBe("0.41.0");
    expect(value(worker, "Provenance")).toBe("signed_descriptor");
    expect(row(worker, "Image digest").querySelector("button.copy")).toHaveAttribute(
      "data-copy-label",
      "Validator worker configured image digest",
    );
    expect(row(worker, "Image digest").querySelector(".copyable > span")?.textContent).toBe(
      "sha256:a4d1e…7bb1a92a07",
    );
  });

  it("distinguishes a component nobody probed from one nobody reported", () => {
    open(MANAGED);
    // Absent on both sides: neither configured nor observed.
    const relay = component("Model relay");
    expect(relay.querySelector("summary .stage")?.textContent).toBe("Unknown");
    expect(relay.querySelector("summary .probe-time")).toBeNull();
    expect(rowKeys(relay)).toEqual(["Health"]);
    expect(value(relay, "Health")).toBe("Not reported (requires heartbeat protocol 9)");
    // Observed and healthy, but the probe could not verify an identity.
    const sandbox = component("Sandbox Docker");
    expect(value(sandbox, "Observed identity")).toBe("Not independently observed");
    expect(subheads(sandbox)).toEqual(["Observed identity", "Configured identity"]);
  });

  it("reads a self-declared unknown as 'Not observed', never as unreported", () => {
    // The validator answered; it just has not probed. Collapsing the two would
    // let a live component that stopped reporting hide behind a protocol gap.
    open(
      MANAGED,
      patched(MANAGED, {
        stack_health: { model_relay: { health: "unknown", required: false } },
      }),
    );
    const relay = component("Model relay");
    expect(relay.querySelector("summary .stage")?.textContent).toBe("Not observed");
    expect(chipOf(relay, "Health")).toEqual(["Not observed", "stage unknown"]);
    expect(value(relay, "Required component")).toBe("No");
    expect(component("Validator worker").querySelector("summary .stage")?.textContent).toBe(
      "Unknown",
    );
  });

  it("names Ollama's readiness as an embedding model, other components as a route", () => {
    open(LEGACY);
    expect(value(component("Ollama"), "Embedding model ready")).toBe("Yes");
    expect(rowKeys(component("Model relay"))).not.toContain("Model route ready");
    cleanup();
    open(
      LEGACY,
      patched(LEGACY, {
        stack_health: { model_relay: { health: "degraded", required: true, model_ready: false } },
      }),
    );
    expect(value(component("Model relay"), "Model route ready")).toBe("No");
    expect(rowKeys(component("Model relay"))).not.toContain("Embedding model ready");
  });

  it("confirms an observed identity that matches the configured pin", () => {
    open(MANAGED);
    // dittobench_api pins a source revision the live probe independently
    // reported; the image digest is unobserved, so the revision is compared.
    const note = component("Scorer · dittobench-api").querySelector(".identity-note");
    expect(note?.className).toBe("identity-note good");
    expect(note?.textContent).toBe("Observed source revision matches the configured pin.");
  });

  it("flags an observed identity that differs from the configured pin", () => {
    open(
      MANAGED,
      patched(MANAGED, {
        stack_health: {
          ditto_subnet: {
            health: "identity_mismatch",
            required: true,
            observed_identity: { image_digest: "sha256:deadbeef" },
          },
        },
      }),
    );
    const worker = component("Validator worker");
    expect(worker.querySelector("summary .stage")?.textContent).toBe("Identity mismatch");
    const note = worker.querySelector(".identity-note");
    expect(note?.className).toBe("identity-note warn");
    expect(note?.textContent).toBe("Observed image digest differs from the configured pin.");
  });

  it("stays silent when only one side has an identity to compare", () => {
    open(MANAGED);
    // Configured pins a digest and a revision; the probe reported a version
    // only, so there is nothing to compare and no note is invented.
    expect(component("Validator worker").querySelector(".identity-note")).toBeNull();
    expect(component("Sandbox Docker").querySelector(".identity-note")).toBeNull();
  });

  it("says a configured component pins nothing rather than showing empty rows", () => {
    open(
      MANAGED,
      patched(MANAGED, {
        stack: { mode: "managed", compose_schema: 1, components: { pylon: { provenance: null } } },
      }),
    );
    const pylon = component("Pylon");
    expect(value(pylon, "Provenance")).toBe("unknown");
    expect(value(pylon, "Identity")).toBe("None pinned");
  });
});
