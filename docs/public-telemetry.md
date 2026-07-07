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
  { rank, miner_hotkey, composite, tool_mean, memory_mean, first_seen, n,
    median_ms, bench_version, dataset_sha256, models, per_category,
    integrity, tokens } ] }`.
  Best-per-miner, ranked by composite. The provenance block (`models` =
  generator/judge/judge_audit/harness, `bench_version`, `dataset_sha256`,
  `per_category` means, `median_ms`, `n`) is the **transparency payload**: it
  lets anyone see *what model produced a run and how it was scored* and pins the
  exact scored artifact (`dataset_sha256`) for a dispute re-score. All of it is
  advisory (not signed) and lifted from the safe subset of `scores.details` —
  extracted defensively so a malformed blob can never break the endpoint.
  `integrity` (paraphrase applied/attempted/fallback, NoLiMa lexical-gap
  rewrites + overlap before→after, capped tool cases, seeding waves) and
  `tokens` (LLM spend to generate+judge) publish the benchmark's **anti-overfit
  posture** so the community can audit *how gaming is resisted*, not just the
  scores.
  **Never** included: `seed` (anti-overfit), `per_case` `expected`/`called` (the
  answer key), agent_id/sha256/signature/validator_hotkey (integrity-internal).
  `is_champion`/weights stay validator-side (KOTH fold), not served here.
- `GET /api/v1/public/weights` → the last-published normalized weight vector
  (champion + tail) — mirrors what the validator set on-chain.
- `GET /api/v1/public/health` → subnet rollup **from what the platform records**:
  `miners`, `scored_miners`, `scored_agents`, `last_scored_at`, `scores_24h`,
  `avg_latency_ms`. Note: no `success_rate` — the platform only ever sees a
  *successful* score, so run started/failed counts and set-weights latency are
  validator-side telemetry (wandb), not fabricated here. Detailed ops stay on the
  existing Prometheus `/metrics`.
- Per-category means + run provenance: the scoring engine (dittobench-api) emits
  `models` + `per_category` (alongside `bench_version`, `dataset_sha256`,
  `lexical_gap`, `paraphrase`, `seeding_waves`, `tokens`) in `RunDetails`; the
  validator forwards the whole blob unsigned as `ScoreReport.details`, the
  platform persists it verbatim to `scores.details` (merged with `per_case`, not
  overwritten), and the public endpoint surfaces only the safe subset. Category
  means come straight from `details.per_category`, never re-derived from
  `per_case` at read time.

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

1. ✅ Public API — `/api/v1/public/leaderboard` + `/api/v1/public/health`.
   `/weights` is intentionally **not** served: the KOTH weight vector is
   validator-side (see the scoring endpoint boundary); the dashboard links wandb
   / the chain for weights.
2. ✅ wandb `telemetry.py` in the validator (ditto-subnet #27) — aggregate +
   per-category tables, opt-in and off by default.
3. ✅ Dashboard SPA (`dashboard/index.html`) against the public API + wandb link.
   Now renders a per-row model chip (harness/generator + `bench v{N}`) and a
   drawer "Benchmark run" section (cases, median latency, bench version,
   harness/judge/audit/generator models, dataset SHA-256, per-category
   breakdown).
4. ✅ Run provenance persisted end-to-end (2026-07-07): dittobench-api
   `RunDetails.{Models,PerCategory}` → `ScoreReport.details` → `scores.details`
   → public leaderboard `models`/`bench_version`/`dataset_sha256`/`per_category`.
   Verified live on localnet against a real Ollama-backed harness run. The
   `ScoreReport.details` field is unsigned and additive — the signed tuple
   (`run_id, seed, composite, tool_mean, memory_mean, median_ms, n`) is
   unchanged, so this never touches the score or the signature.
