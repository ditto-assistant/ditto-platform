# Public telemetry: wandb + dashboard + public API

Status: **approach decided 2026-07-04** (Nick). This doc is the contract for what
SN118 publishes publicly and how. Implementation tracked per section below.

## Decisions

1. **wandb transparency = aggregate + per-category.** Publish per-agent
   composite, tool/memory means, and **per-category** means, plus
   leaderboard / weights / health. **Do not** publish raw per-case
   `expected`/`called` tools or the haystack contents — that is the benchmark's
   answer key and would let miners fit to specific cases.
2. **Dashboard = both.** A custom public "front door" (leaderboard + health) for
   humans, backed by a public read API + Prometheus; wandb linked alongside for
   researchers who want full per-epoch telemetry.
3. **Public read API = yes.** Add a rate-limited, no-auth read endpoint so the
   dashboard (and anyone) can read the leaderboard without a validator hotkey.

## Anti-gaming posture (the load-bearing rule)

Everything public is an **aggregate**. The seed is published only *after* a run
is scored (reproducibility, not pre-disclosure); seeds already rotate per
submission, so a past seed does not unlock a future run. Per-case rows stay
private. If we ever want research-grade per-case release, do it on a **delay**
(e.g. after that dataset generation is retired) — never live.

## Surface 1 — wandb (validator → public project)

The validator logs one wandb run per validator hotkey; each epoch/sweep appends.
New module `ditto/validator/telemetry.py` in ditto-subnet, called from the worker
after each sweep + weight set. Config (env): `WANDB_PROJECT`, `WANDB_ENTITY`,
`WANDB_API_KEY`, `WANDB_MODE` (`online`|`disabled`, default `disabled` so it is
opt-in). No secrets or keys are ever logged.

**Time-series scalars** (per epoch): `sweep_duration_s`, `queue_depth`,
`runs_started`, `runs_failed`, `runs_failed_frac`, `champion_composite`,
`positive_miner_count`, `openrouter_cost_delta`, `set_weights_latency_s`,
`set_weights_ok` (0/1), `chain_block`, and per-stage dittobench durations
(`build_s`, `seed_s`, `run_s`, `judge_s`).

**Tables** (snapshot per epoch):
- `scores` — one row per agent scored this sweep: `uid`/`miner` (short),
  `agent_id[:8]`, `composite`, `tool_mean`, `memory_mean`, one column per
  category mean (`cat.link_read`, `cat.web_search`, `cat.memory_lookup`,
  `cat.single-session-user`, `cat.temporal-reasoning`, `cat.multi-session`, …),
  `n`, `median_ms`, `seed`, `run_id`.
- `leaderboard` — best-per-miner ledger: `rank`, `miner`, `composite`,
  `is_champion`, `ath` (all-time-high composite for that miner).
- `weights` — `miner`, `uid`, `weight` (normalized), `role` (champion|tail),
  plus scalars `koth_margin`, `champion_share`.
- `integrity` — counters: `held_for_copy_review`, `dedup_rejected`,
  `banned_hotkey_rejected`, `seed_rotations`.

## Surface 2 — public read API (platform)

New router `endpoints/public.py`, mounted at `/api/v1/public`, **no auth**,
rate-limited, `Cache-Control: public, max-age=30`. Read-only, aggregate-only.

- `GET /api/v1/public/leaderboard` → `{ generated_at, count, entries: [
  { rank, miner_hotkey, composite, tool_mean, memory_mean, is_champion,
    first_seen, n } ] }`. Best-per-miner, ranked by composite; `is_champion`
  from the KOTH fold. **No** agent_id/sha/signature/per_case.
- `GET /api/v1/public/weights` → the last-published normalized weight vector
  (champion + tail) — mirrors what the validator set on-chain.
- `GET /api/v1/public/health` → subnet rollup: `miners`, `scored_miners`,
  `last_sweep_at`, `runs_24h`, `success_rate_24h`, `avg_latency_ms`. (Detailed
  ops stay on the existing Prometheus `/metrics`.)
- Per-category means: derive from the stored `scores.details.per_case` at read
  time (aggregate only) or persist a `per_category` blob on submit — either way
  the public shape exposes category means, never per-case rows.

## Surface 3 — dashboard (custom front door)

Static SPA (no server-side secrets) pulling the public API above; wandb linked
for the deep dive. Sections:
- **Leaderboard** — rank, miner, composite, category radar sparkline, weight %,
  trend arrow; champion highlighted.
- **Miner drill-down** — composite history, category radar, ATH badge, best run
  (aggregate only).
- **Weights** — current on-chain allocation (pie/bar), champion callout.
- **Health** — validators online, sweep cadence, runs/day, success rate, avg
  latency; "anti-overfit: seeds rotate every submission" assurance line.

Hosting: static build → object storage + CDN (matches the earlier static-serve
idea); no server needed since all data comes from the public API + wandb.

## Build order

1. Public API (`/api/v1/public/leaderboard` + `/weights` + `/health`) — unblocks
   both the dashboard and external transparency. **← start here**
2. wandb `telemetry.py` in the validator — aggregate + per-category tables.
3. Dashboard SPA against the public API + embedded wandb.
4. (Optional) persist `per_category` on submit for cheaper category reads.
