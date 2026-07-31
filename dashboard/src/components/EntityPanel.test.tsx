// Tests for the entity modal shell: opens from the routeStore's entityRoute,
// overlay vs dedicated /agent/{id} page modes (role swap + body.entity-page,
// showModal 5980–6004), close semantics via closeEntityRoute + Escape
// (closeModal 6338–6367), focus moved into the dialog, and the three tenant
// bodies (miner summary, validator summary, agent evidence skeleton).
// Also guards the shell slice of assert-inventory row 26
// (test_validator_names_remain_optional_untrusted_decoration): the validator
// display name is optional untrusted decoration — rendered as text, never
// markup — and the hotkey stays the anchor identity.
import { cleanup, render, waitFor } from "@solidjs/testing-library";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { rankEntries } from "../lib/scoring";
import { syncFromLocation } from "../stores/routeStore";
import { FIXTURE_TOP_AGENT_ID, installFixtureFetch, loadFixture } from "../test-fixtures";
import type { OperationsPayload } from "../types/fleet";
import type { LeaderboardPayload } from "../types/leaderboard";
import { EntityPanel } from "./EntityPanel";

const leaderboard = loadFixture<LeaderboardPayload>("leaderboard");
const operations = loadFixture<OperationsPayload>("operations");
const entries = rankEntries(leaderboard.entries ?? []);
const topEntry = entries[0] as (typeof entries)[number];
const validatorHotkey = String(operations.validators.validators?.[0]?.validator_hotkey);

let restoreFetch: (() => void) | null = null;

beforeEach(() => {
  history.replaceState(null, "", "/#/overview");
  syncFromLocation();
  restoreFetch = installFixtureFetch();
});

afterEach(() => {
  cleanup();
  restoreFetch?.();
  restoreFetch = null;
  document.body.classList.remove("entity-page");
});

function renderPanel(names: Record<string, string> = {}): void {
  render(() => (
    <EntityPanel
      entries={() => entries}
      operations={() => operations}
      validatorNames={() => names}
      currentBench={() => 7}
      settledView={() => false}
    />
  ));
}

function visit(url: string): void {
  history.replaceState(null, "", url);
  syncFromLocation();
}

function modal(): HTMLElement {
  const el = document.getElementById("modal");
  if (!el) throw new Error("missing modal");
  return el;
}

describe("EntityPanel miner tenant", () => {
  it("opens the overlay dialog from a hash-query miner route", () => {
    renderPanel();
    visit("/#/overview?miner=" + topEntry.miner_hotkey);
    expect(modal().classList.contains("open")).toBe(true);
    expect(modal()).toHaveAttribute("role", "dialog");
    expect(modal()).toHaveAttribute("aria-modal", "true");
    expect(modal()).toHaveAttribute("aria-hidden", "false");
    expect(document.getElementById("modal-back")?.classList.contains("open")).toBe(true);
    expect(document.getElementById("d-title")).toHaveTextContent("Raw score rank #1");
    // The hotkey is an entity anchor plus the copy control.
    const anchor = document.querySelector('#d-hotkey a[data-entity-link="miner"]');
    expect(anchor).toHaveTextContent(topEntry.miner_hotkey);
    expect(document.getElementById("d-hotkey-copy")).toHaveAttribute(
      "data-key",
      topEntry.miner_hotkey,
    );
    // Focus lands on the close control for keyboard/AT users.
    expect(document.activeElement).toBe(document.getElementById("modal-close"));
  });

  it("summarizes the run and links the full page", () => {
    renderPanel();
    visit("/#/overview?miner=" + topEntry.miner_hotkey);
    expect(document.getElementById("d-bench")).toHaveTextContent("DittoBench v7");
    expect(document.getElementById("d-stats")?.textContent).toContain("Best-scoring agent");
    expect(document.getElementById("d-stats")?.textContent).toContain("Current leaderboard score");
    const openFull = document.getElementById("d-open-full");
    expect(openFull).toHaveAttribute("href", "/miner/" + topEntry.miner_hotkey);
    expect(document.getElementById("d-stats")?.classList.contains("pipeline-mode")).toBe(false);
  });

  it("closes on Escape, returning to the page under the overlay", async () => {
    renderPanel();
    visit("/#/overview?miner=" + topEntry.miner_hotkey);
    expect(modal().classList.contains("open")).toBe(true);
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    await waitFor(() => expect(modal().classList.contains("open")).toBe(false));
    expect(location.hash).toBe("#/overview");
    expect(modal()).toHaveAttribute("aria-hidden", "true");
  });
});

describe("EntityPanel validator tenant (row 26 shell slice)", () => {
  it("titles the dialog with the hotkey identity when no display name exists", () => {
    renderPanel();
    visit("/#/operations?validator=" + validatorHotkey);
    expect(modal().classList.contains("open")).toBe(true);
    expect(document.getElementById("d-title")).toHaveTextContent("Validator");
    const anchor = document.querySelector('#d-hotkey a[data-entity-link="validator"]');
    expect(anchor).toHaveTextContent(validatorHotkey);
    // Validators have no dedicated full page; the action is hidden.
    const openFull = document.getElementById("d-open-full") as HTMLElement;
    expect(openFull.style.display).toBe("none");
  });

  it("treats display names as escaped, optional decoration over the hotkey identity", () => {
    const hostile = '<img src=x onerror="alert(1)"> & "Validator" <b>Name</b>';
    renderPanel({ [validatorHotkey]: hostile });
    visit("/#/operations?validator=" + validatorHotkey);
    const title = document.getElementById("d-title");
    // Rendered as text, byte for byte — never parsed as markup.
    expect(title?.textContent).toBe(hostile);
    expect(title?.querySelector("img")).toBeNull();
    expect(title?.querySelector("b")).toBeNull();
    // The hotkey stays the anchor identity regardless of the name.
    const anchor = document.querySelector('#d-hotkey a[data-entity-link="validator"]');
    expect(anchor).toHaveTextContent(validatorHotkey);
  });

  it("renders the signed-report summary from the heartbeat", () => {
    renderPanel();
    visit("/#/operations?validator=" + validatorHotkey);
    const stats = document.getElementById("d-stats");
    expect(stats?.textContent).toContain("Fleet status");
    expect(stats?.textContent).toContain("Worker state");
    expect(stats?.textContent).toContain("Heartbeat protocol");
    expect(stats?.textContent).toContain("Slots");
    expect(stats?.classList.contains("pipeline-mode")).toBe(true);
  });

  it("normalizes a validator route onto the operations page", async () => {
    renderPanel();
    visit("/#/overview?validator=" + validatorHotkey);
    await waitFor(() => expect(location.hash).toBe("#/operations?validator=" + validatorHotkey));
    expect(modal().classList.contains("open")).toBe(true);
  });
});

describe("EntityPanel agent tenant", () => {
  it("resolves an agent overlay via the public activity lookup and mounts the evidence slot", async () => {
    renderPanel();
    visit("/#/submissions?agent=" + FIXTURE_TOP_AGENT_ID);
    await waitFor(() => expect(modal().classList.contains("open")).toBe(true));
    expect(document.getElementById("d-title")).toHaveTextContent("bolt-v7-top1");
    // The deep-evidence container the submissions/reviews port fills, with
    // the ledger section ids.
    for (const id of [
      "pipeline-current-title",
      "pipeline-meta-title",
      "pipeline-screening-history",
      "pipeline-accepted-scores",
      "pipeline-confirmation-scores",
      "pipeline-validator-history",
    ]) {
      expect(document.getElementById(id), id).toBeTruthy();
    }
    expect(document.querySelector("[data-agent-evidence]")).toHaveAttribute(
      "data-agent-evidence",
      FIXTURE_TOP_AGENT_ID,
    );
  });

  it("renders the dedicated /agent/{id} page as a main region, not a dialog", async () => {
    renderPanel();
    visit("/agent/" + FIXTURE_TOP_AGENT_ID);
    await waitFor(() => expect(modal().classList.contains("open")).toBe(true));
    await waitFor(() =>
      expect(document.getElementById("d-title")).toHaveTextContent("bolt-v7-top1"),
    );
    expect(modal()).toHaveAttribute("role", "main");
    expect(modal()).toHaveAttribute("aria-modal", "false");
    expect(modal().classList.contains("full-page")).toBe(true);
    expect(document.body.classList.contains("entity-page")).toBe(true);
    // No backdrop in full-page mode; Escape must not tear down the page.
    expect(document.getElementById("modal-back")?.classList.contains("open")).toBe(false);
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    expect(modal().classList.contains("open")).toBe(true);
  });

  it("states plainly when a full-page submission cannot be found", async () => {
    renderPanel();
    visit("/agent/ffffffff-0000-0000-0000-000000000000");
    await waitFor(() =>
      expect(document.getElementById("d-stats")?.textContent).toContain(
        "This submission could not be found.",
      ),
    );
    expect(document.getElementById("d-bench")).toHaveTextContent("Unavailable");
  });
});
