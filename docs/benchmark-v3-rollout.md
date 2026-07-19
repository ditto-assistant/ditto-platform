# Benchmark rollout activation

Every benchmark transition uses a durable five-agent collection cohort. A median never mixes
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

- Heartbeat protocols before v8 cannot advertise benchmark capabilities.
- Protocol v8 and newer sign `capabilities.scorer_benchmarks`. A validator is
  capable of a version only while its `fresh_verified` observation is at most
  five minutes old and its scorer source revision and software version match the
  signed stack identity.
- Job and artifact responses carry `bench_version`; old validators ignore the
  additive field.
- A legacy v2 score may omit `bench_version` and keeps the legacy signature
  bytes. Newer scores must explicitly report and sign their version.

## Operator-controlled start

Shipping a contract makes it available but never opens a rollout. Validator
heartbeats and job polls may refresh an existing rollout, but cannot create one.
Backroom discovers the shipped choices from the read-only control endpoint:

```text
GET /api/v1/admin/benchmark-rollout
Authorization: Bearer ...
```

Starting a selected target is an authenticated, audited UI action. It requires
a reason, an expected active-version compare-and-swap value, and the exact typed
confirmation `START BENCHMARK V{target}`. Superseding an unactivated rollout
similarly requires `SUPERSEDE BENCHMARK V{target}` and a reason. These controls
are intentionally not exposed over MCP.

The start freezes the current top five eligible agents and miners and renders a
target-version dataset for every member. The frozen membership and positions
are never silently reshuffled. At least one fresh, identity-matched validator
must advertise the selected target before it can start; additional compatible
validators can join asynchronously.

Only capable validators receive target-version cohort tickets. Scores, tickets,
retry budgets, datasets, leases, and uniqueness are keyed by benchmark version.
At one or two distinct target scores, the active version remains authoritative
for the whole pool. This lets work begin with one compatible validator while
the canonical ranking remains stable. Once the full five-agent emission set has
target-version quorum, the desired version takes authority for the whole pool;
the same locked transaction activates the target and appends an audit event.
Canonical reads then exclude source-only agents; no median or weight fold mixes
versions.

The authenticated ledger adds optional `bench_version` provenance. Old
validators ignore that additive field and fold the full platform-selected pool;
new validators retain it for audit and re-score scheduling but deliberately fold
the same pool. This keeps on-chain weights identical during asynchronous
validator upgrades.

If a frozen member becomes banned, held, or otherwise ineligible, the rollout
changes to `blocked_ineligible`. It neither replaces nor drops that member. It
resumes with the same snapshot only if eligibility is restored. An operator may
instead explicitly supersede the unactivated rollout with a recorded reason.

## Observation and recovery

`GET /api/v1/public/bench/rollout` reports desired and active versions, rollout
status, the conservative count of currently capable validators, each qualified
agent and position, and its target-version score count. State is database-backed
and idempotent across API restarts.

Before activation, the public leaderboard defaults to the active-version pool
and retains explicit historical version views that never project current
emissions. After activation, the target is canonical and source-only results
are excluded.
