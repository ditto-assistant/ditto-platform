# SN118 submission contract — platform & validator enforcement

What a miner submits, and which invariants the **platform**, **screener**, and
**validator** actually enforce — including, transparently, what is *not* enforced
yet.

The canonical *miner-facing* contract (what's in the tarball, the fixed HTTP
interface vs. the free-to-edit surface, the wire shapes) lives with the harness:
the [dittobench-starter-kit](https://github.com/ditto-assistant/dittobench-starter-kit)
`README.md` + `PROTOCOL.md`. This doc is the **enforcement side** — what we check,
where, and what's deferred.

## The artifact

A single **gzipped tarball of the miner's whole harness crate** — the entire
buildable project with a `Dockerfile` at the tarball root. Not a single source
file, and not `ditto-harness` (that's a pinned git dependency the crate builds on
top of). The platform stores the tarball in object storage keyed by agent id and
**never unpacks it**; the screener and validator do.

## What's enforced today

| Stage | Where | Check |
| --- | --- | --- |
| **Upload** | `endpoints/upload.py` (`/api/v1/upload/*`) | On-chain eval-fee payment verified (replay-protected); tarball ≤ **20 MiB** by default (`MAX_TARBALL_SIZE_BYTES`, overridable with `DITTO_MAX_TARBALL_SIZE_BYTES`) enforced from the *actual streamed bytes*; **SHA-256 re-verified** against the miner's claim; one payment per upload. |
| **Screen** | `endpoints/screener.py` (`/api/v1/screener/*`) + private `ditto-screener` worker | The worker builds and runs the crate, requires `/health` and the hidden model-response canary, then reports a lease-bound signed verdict: pass → `evaluating`, deterministic fail → `rejected`, retryable infrastructure failure → lease expiry and retry. The screener is **platform-operated** and authenticates with a dedicated allowlisted hotkey plus bearer token, not a validator permit. |
| **Evaluate** | `dittobench-api` (mode B) | Fetch the presigned tarball; safe-extract with zip-slip + gzip-bomb guards; require a `Dockerfile` at the tarball root (or a single top-level dir); `docker build` + run the container; drive `GET /health`, `POST /seed`, `POST /run`; score. |
| **Anti-overfit** | `dittobench-api` datagen | A **fresh seed per run** (stratified categories); the miner cannot see or pin the dataset. Difficulty variance is calibrated to a between-seed stddev ≤ 0.03. |

So the effective bar today is: **payment is valid, the tarball is within limits,
the crate builds, and the running container speaks the `/run` protocol well
enough to be scored.**

## What is deferred — NOT enforced yet

Per `CLAUDE.md`, several `/upload/*`-adjacent validations are intentionally
deferred pending the harness-interface spec and supporting tables. Stated plainly
so miners and reviewers aren't misled:

- **tar manifest** format validation (a declared file/entrypoint manifest);
- **import / dependency allowlist** (what the crate may pull in);
- **schema diff** — verifying the crate still implements the required harness
  interface rather than just building;
- **banned-hotkey** rejection (needs the `banned_hotkeys` table).

The screener's build gate is the first real guard. The manifest + allowlist +
schema checks are the planned next layer; until they land, "it builds and serves
the protocol" is the whole bar, and a submission is trusted to be a good-faith
harness crate.

## Lifecycle

```
uploaded ──screener pass──▶ evaluating ──validator score──▶ scored ──▶ live
   └────────screener fail──▶ screening_failed
(banned is terminal; see the ban work.)
```

The status column is the `agentstatus` enum (`api_models/agent_status.py`); the
transitions above are owned by `endpoints/upload.py`, `endpoints/screener.py`,
and `endpoints/validator.py` respectively.

## Scoring

`composite = 0.5 * tool_mean + 0.5 * memory_mean` when both kinds are present
(bench_version 2 / DittoBench v2 — rebalanced from v1's `0.6 / 0.4` because
memory is the core product value and the raw-pairs seeding tier makes
`memory_mean` the harder axis). The platform **records what the validator
reports and never recomputes it** (`api_models/validator.py`,
`db/queries/scores.py`). The on-chain profile is `run_size=full` (fresh
anti-cheat dataset + LLM judge); the starter kit's local `practice` scorer is a
fast deterministic proxy, so a miner's real score differs.

Benchmark scoring changes are versioned: the validator stamps `bench_version`
in the score `details`, and the weight fold only compares scores of the max
version present (see the DittoBench v2 design). A version bump triggers a
re-score sweep before old scores are compared to new.

## Pointers

- **Miner-facing contract + wire protocol** — dittobench-starter-kit `README.md`
  ("Submit") + `PROTOCOL.md`.
- **Upload** — `ditto/api_server/endpoints/upload.py`.
- **Screener protocol and state transitions** — `ditto/api_server/endpoints/screener.py`.
- **Private build/run worker** — `ditto-assistant/ditto-screener`.
- **Tarball ingest + Docker sandbox (mode B)** — `dittobench-api`
  `internal/sandbox/` (`Dockerfile`-at-root build-context rule, safe extractor).
- **Validator deploy** — infra `docs/validator-deploy.md`.
- **End-to-end diagram** — `docs/validator/subnet-architecture.mmd`.
