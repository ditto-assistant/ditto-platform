# Benchmark v3 activation

Benchmark v3 uses a durable five-agent collection cohort plus per-agent quorum
authority. A median never mixes score rows from different versions: each agent
is represented by its v3 median after 3/3, otherwise by its v2 median.

Per-agent quorum authority can be temporarily suspended with
`DESIRED_AUTHORITY_AT_QUORUM = False` in `ditto/db/queries/benchmark_rollout.py`
(used 2026-07-19 while v3 scoring was stabilized): every agent with an
active-version median keeps it until activation, while desired-version quorums
are still collected and shown as rollout progress.

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
