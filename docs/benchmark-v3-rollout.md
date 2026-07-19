# Benchmark v3 activation

Benchmark v3 uses a durable five-agent collection cohort. A median never mixes
score rows from different versions, and the ledger is on exactly one version at
a time.

> **Threshold-gated authority.** The desired version takes over ledger
> authority for the whole pool only once at least
> `MIN_DESIRED_AUTHORITY_AGENTS` (= 5, the KOTH champion + `KOTH_TAIL_SIZE`)
> agents hold a complete, ranked desired-version quorum
> (`ditto/db/queries/benchmark_rollout.py`, applied in
> `list_eligible_ledger`). Below that count the active version stays
> authoritative for every agent — for the public leaderboard, the validator
> weight fold, and KOTH — while desired-version quorums are collected and shown
> as per-row rollout progress. The threshold exists because the flip drops
> agents without a desired-version quorum: crossing it guarantees the emission
> set still has its full complement of recipients. Authority is never mixed
> per-agent, because composites from different benchmark versions are not on a
> comparable scale within one KOTH fold.

## Compatibility

- Heartbeat protocols v1 through v7 are v2-only.
- Protocol v8 signs `capabilities.scorer_benchmarks`. A validator is v3-capable
  only while its `fresh_verified` observation is at most five minutes old and
  its scorer source revision and software version match the signed stack
  identity.
- Job and artifact responses add `bench_version`. Old validators ignore it.
- A v2 score may omit `bench_version` and keeps the legacy signature bytes. A
  v3 score must explicitly report and sign `bench_version=3`.

## Cohort and activation

An authenticated operator starts the rollout once:

```text
POST /api/v1/admin/benchmark-rollout/v3
X-Admin-Key: ...
```

The transaction freezes the current top five eligible agents and miners. Before
the transaction, the platform explicitly renders a v3 dataset for every frozen
agent. The frozen membership and positions are never silently reshuffled.

Only fresh, identity-matched v8 validators receive v3 cohort tickets. Scores,
tickets, retry budgets, datasets, leases, and uniqueness are keyed by benchmark
version. At one or two distinct v3 scores, that agent's v2 result remains
authoritative. Its third v3 score atomically replaces only that agent's v2 median
in the leaderboard and validator ledger. This hybrid pool lets work begin with
one compatible validator while incomplete agents retain stable v2 scores. The
same locked transaction that observes exactly 3/3 on all five changes the global
active version to v3 and appends an audit event. Canonical reads then exclude
v2-only agents; no median combines versions.

The authenticated ledger adds optional `bench_version` provenance. Old
validators ignore that additive field and fold the full platform-selected pool;
new validators retain it for audit and re-score scheduling but deliberately fold
the same full pool. This keeps on-chain weights identical during asynchronous
validator upgrades.

If a frozen member becomes banned, held, or otherwise ineligible, the rollout
changes to `blocked_ineligible`. It neither replaces nor drops that member. It
resumes with the same snapshot only if eligibility is restored; any different
cohort requires a future explicit audited rollout mechanism.

## Observation and recovery

`GET /api/v1/public/bench/rollout` reports desired and active versions, rollout
status, the conservative count of currently capable validators, each frozen
agent and position, and its v3 score count. State is database-backed and
idempotent across API restarts.

Before activation, newly screened submissions and every non-cohort ticket remain
v2. The public leaderboard defaults to the authoritative hybrid pool and offers
`?bench_version=2` as a historical view that never projects current emissions.
After activation, v3 is the canonical ledger version and v2-only results are
excluded.
