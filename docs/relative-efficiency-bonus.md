# Relative token-efficiency bonus (bench_version >= 7)

Platform-side implementation of the v7 efficiency contract. The other half of
the contract already shipped validator-side: for `bench_version >= 7` the
deterministic scorer is **quality-only** — it reports audited token usage
(`details.token_usage`, relay-metered through the trusted broker) and a neutral
`token_efficiency` record (`formula_version: "v7-quality-only-v1"`, multiplier
1) as in-band proof that usage never moved the composite. Efficiency
incentives live here, as a bounded, epoch-frozen, strictly-upside bonus over
frozen cohorts. The validator scorer is never touched: same artifact, same
validator score, forever.

Code map:

* `ditto/api_server/efficiency.py` — the pure math (cohort build, dedupe,
  robust reference, bonus curve) plus the DB-backed materializer.
* `ditto/db/models.py` — `EfficiencyCohortSnapshot` + `EfficiencyBonus`
  (migration `2026_07_24_add_efficiency_bonus_tables.py`).
* `ditto/db/queries/efficiency.py` — append-only reads/writes.
* `ditto/api_server/endpoints/public.py` — leaderboard exposure +
  `GET /public/efficiency/snapshots/{snapshot_id}`.
* `ditto/api_server/endpoints/scoring.py` — validator-ledger exposure behind
  the default-off fold flag.

## Algorithm

One **efficiency epoch** is a fixed UTC window (`epoch_hours`, default 24 h,
windows counted from the Unix epoch). Once per epoch, per
`(bench_version, run_size)` board — ranked boards are always `run_size =
"full"`, since smaller profiles never rank — the platform freezes a cohort:

1. **Candidates.** Every finalized (k=3 quorum), ranked
   (`n >= MIN_ELIGIBLE_CASES`, composite > 0) entry of the authoritative
   ledger (`list_eligible_ledger`, one entry per payment-time coldkey).
2. **Audited cost.** Per agent: the median of `details.token_usage.total_tokens`
   over its quorum score rows whose accounting is `status == "complete"` with
   `usage_unavailable == 0`. Chat tokens only — embedding load is
   validator-fixed per dataset, not a harness skill. Miner-reported numbers
   never reach this blob; the validator writes it from the trusted relay. No
   complete audited row → the submission is unqualified (bonus 0, never a
   penalty).
3. **Quality gate.** `composite >= Q_min` AND `memory_mean >= M_min`. The
   memory floor exists so a harness cannot buy efficiency by gutting the
   memory half. First epoch: the static config floors. Later epochs ratchet
   from the previous **active** cohort: `Q_min = max(config, median composite
   of previous cohort)`, `M_min = max(config, 0.8 x median memory_mean)`.
4. **Lineage dedupe.** Qualified submissions collapse to one entry per
   lineage key — `normalized_source_hash` (canonicalized source; survives
   repack/reformat) when present, else the artifact `sha256`. The
   best-scoring entry survives (composite desc, first_seen asc, agent_id
   asc); the collapsed ids are recorded in the snapshot. Grounded in a real
   observation: 3 of the top-5 harnesses ship byte-identical model fixtures
   under different hotkeys — one lineage must not define the frontier.
5. **Cohort.** The top-`cohort_size` (default 25) deduped entries by
   composite.
6. **Activation gate.** If the deduped cohort is smaller than `min_cohort`
   (`N_min`, default 8 — the spec's value), the snapshot is frozen **inactive**:
   the observation is recorded, no bonus is awarded, the API reports
   `active: false`, and no bonus rows are written (so activation in a later
   epoch can still freeze those agents at their first active epoch).
7. **Robust reference** (never the mean, never the single minimum):
   * `P25` = nearest-rank 25th percentile of cohort token totals — the
     efficient-quartile full-bonus frontier;
   * `median` = cohort median — the zero-bonus point.
8. **Bonus curve.** For a qualified submission with audited cost `C`:

   ```
   bonus = cap                                  if C <= P25
   bonus = cap * (median - C) / (median - P25)  if P25 < C < median
   bonus = 0                                    if C >= median
   ```

   `cap` = `B_max`, default 5%, hard ceiling 10% (boot check + DB CHECK).
   Degenerate cohorts (`median == P25`) collapse to a step at the frontier.

   *Reconciliation note:* the validator-side spec draft tapered to zero at
   `4 x P25`. The operator decision (implemented here) anchors the zero point
   at the cohort **median** instead: both are robust, but the median ties the
   taper to the cohort's actual dispersion, so a uniformly lean cohort does
   not hand near-full bonuses to its own laggards, and an outlier-heavy tail
   cannot stretch the paying range. Full-bonus frontier (P25) and the 5–10%
   cap are identical to the spec.
9. **Strictly upside.** `bonus >= 0` always. Unqualified, unaudited, or
   expensive runs keep their unmodified composite. There is no path where
   fewer tokens raise a score that quality did not already earn.
10. **Effective score.** `effective_composite = composite * (1 + bonus)` —
    multiplicative, matching the spec's `bonus_multiplier` form; a scale
    factor preserves the fold's composite-comparison semantics and a zero
    composite can never buy weight from cheapness. The validator's signed
    composite is never modified; the effective value is derived at read time
    from the frozen bonus.

### Freezing semantics

* The **snapshot** (membership, floors, `P25`/`median`, cap) is computed once
  per epoch and persisted append-only. Recomputing in a later epoch inserts a
  new row; historical rows never change.
* A submission's **bonus row** (`efficiency_bonuses`) is inserted exactly
  once per `(agent_id, bench_version)`, against the frozen snapshot of the
  epoch in which it is first seen finalized while a snapshot is ACTIVE —
  including explicit zero rows, so "no bonus" is as frozen as "5%". Published
  effective scores never drift when later submissions arrive.
* A submission finalizing mid-epoch is scored against the epoch's **frozen**
  reference (it does not join the cohort until the next epoch's freeze).
* Materialization is lazy and idempotent: the first authoritative-board
  leaderboard read of an epoch freezes the snapshot and assigns missing
  rows (the dashboard polls continuously, so in practice this is prompt).
  Concurrent materializers race safely — unique keys make the loser retry and
  adopt the winner's frozen rows.

## Storage schema

`efficiency_cohort_snapshots` (append-only, immutable):

| column | meaning |
|---|---|
| `snapshot_id` (PK, UUID) | provenance pointer used by bonus rows and the API |
| `bench_version`, `run_size`, `epoch_index` | unique cohort key |
| `active` | whether `n_min` was met after dedupe |
| `cohort_limit`, `n_min`, `bonus_cap`, `quality_floor`, `memory_floor` | frozen policy |
| `reference_p25_tokens`, `reference_median_tokens` | frozen robust reference (null while inactive) |
| `members` (JSON) | `[{agent_id, miner_hotkey, lineage_key, composite, memory_mean, token_total, collapsed_agent_ids}]` |
| `computed_at` | freeze time (UTC) |

`efficiency_bonuses` (insert-once per `(agent_id, bench_version)`):

| column | meaning |
|---|---|
| `agent_id`, `bench_version` (PK) | the bonused submission at its board version |
| `snapshot_id` (FK) | the frozen cohort the bonus was computed against |
| `token_total` | audited cost used (median of complete quorum totals; null → bonus 0) |
| `bonus` | frozen fraction in `[0, 0.1]` (DB CHECK) |
| `created_at` | assignment time (UTC) |

Every bonus is reproducible from stored data alone:
`reference_from_snapshot(snapshot)` + the row's `token_total` re-derive it.

## API changes

Public (`GET /api/v1/public/leaderboard`):

* Per finalized entry (v7+, feature enabled): `efficiency_bonus`,
  `effective_composite`, `efficiency_snapshot_id` — base composite, bonus,
  and effective score are exposed **distinctly** so the UI can show
  provenance. Null below v7, while disabled/inactive, or before assignment.
* Response-level `efficiency` status: `active`, `epoch_index`,
  `snapshot_id`, `cohort_size`, `n_min`, `bonus_cap`,
  `reference_p25_tokens`, `reference_median_tokens`. Before activation the
  API says the bonus is inactive (`active: false`); below v7 or disabled it
  is null.
* `GET /api/v1/public/efficiency/snapshots/{snapshot_id}`: the full immutable
  audit record. Lineage digests are moderation-adjacent and never exposed —
  members carry opaque `lineage_group` ordinals plus `collapsed_agent_ids`.
* Leaderboard **ranking still follows the base composite**; the effective
  score is display/provenance until the fold consumes it (below).

Validator ledger (`GET /api/v1/scoring/scores`): `LedgerEntry` gains
additive-optional `efficiency_bonus` and `effective_composite`, populated
**only** when `DITTO_EFFICIENCY_BONUS_FOLD_ENABLED=true`. With the flag off
(default) the ledger is byte-identical to the pre-bonus wire shape. The
weight fold itself lives in ditto-subnet and must ship its own consensus
change before consuming these fields; a validator must never fold them
unilaterally. (The `LedgerEntry` contract golden was regenerated —
`ditto/tests/contract/validator_contract.json` — and the ditto-subnet copy
must be synced when that repo picks up the fields.)

## Config knobs (env)

| env var | default | meaning |
|---|---|---|
| `DITTO_EFFICIENCY_BONUS_ENABLED` | `false` | master switch: snapshots, bonus assignment, public exposure |
| `DITTO_EFFICIENCY_BONUS_FOLD_ENABLED` | `false` | expose effective_composite on the validator ledger (requires enabled) |
| `DITTO_EFFICIENCY_BONUS_CAP` | `0.05` | `B_max`; boot-validated to `(0, 0.10]` |
| `DITTO_EFFICIENCY_BONUS_COHORT_SIZE` | `25` | top-N cohort membership cap |
| `DITTO_EFFICIENCY_BONUS_MIN_COHORT` | `8` | `N_min` activation gate (after dedupe) |
| `DITTO_EFFICIENCY_BONUS_EPOCH_HOURS` | `24` | efficiency epoch length (fixed UTC windows) |
| `DITTO_EFFICIENCY_BONUS_QUALITY_FLOOR` | `0` | static `Q_min` fallback (first epoch / lower bound) |
| `DITTO_EFFICIENCY_BONUS_MEMORY_FLOOR` | `0` | static `M_min` fallback (first epoch / lower bound) |

All are validated at boot (`check_config`); the process refuses to start with
an out-of-range cap, `cohort_size < min_cohort`, `min_cohort < 2`, or fold
exposure without the master switch.

## Activation lifecycle

1. **Dark (today).** `DITTO_EFFICIENCY_BONUS_ENABLED=false`. Nothing is
   computed, written, or exposed. All boards byte-identical to today.
2. **Observing.** Enable the master switch once v7 is the authoritative
   board. Each epoch freezes a snapshot; while the deduped qualified cohort
   is below `N_min` the snapshots are inactive, every bonus is 0, and the API
   reports `active: false`.
3. **Active.** The first epoch whose deduped cohort reaches `N_min` freezes
   an active snapshot; every finalized qualified agent gets its insert-once
   bonus row; the leaderboard shows base / bonus / effective distinctly.
4. **Fold (later, cross-repo).** After ditto-subnet ships a weight-fold
   consensus change that consumes `effective_composite`, set
   `DITTO_EFFICIENCY_BONUS_FOLD_ENABLED=true` so the ledger carries the
   fields. Until both halves are done, emissions remain a pure function of
   the base composite.

### Operational runbook: enabling on v7

1. Deploy this build; run `make migrate` (adds the two tables; no existing
   table is touched).
2. Confirm v7 is the authoritative board (`/public/leaderboard` →
   `active_bench_version >= 7`) and that scores carry
   `token_efficiency.formula_version == "v7-quality-only-v1"` with complete
   `token_usage` blocks.
3. Pick the epoch's policy: leave defaults (`cap 5%`, `N=25`, `N_min 8`,
   24 h epochs) or set the env knobs. Set static floors if the first epoch
   should already gate quality (e.g. `QUALITY_FLOOR=0.3`).
4. Set `DITTO_EFFICIENCY_BONUS_ENABLED=true` and restart. The next
   leaderboard read freezes the first snapshot.
5. Verify: `/public/leaderboard` → `efficiency.active`; when true, spot-check
   one entry's bonus against its snapshot via
   `/public/efficiency/snapshots/{id}` (P25/median + the entry's
   `token_total`).
6. Leave `DITTO_EFFICIENCY_BONUS_FOLD_ENABLED=false` until the subnet-side
   fold change is reviewed, shipped, and coordinated.

Changing `cap` / floors / `N` mid-flight only affects **future** epochs'
snapshots; frozen snapshots and assigned bonuses never move.

## Anti-gaming properties and residual risks

Blocked by construction:

* **Sandbagging** (cheap-but-bad runs): the quality gate — composite floor
  AND memory floor — plus the ratchet from the previous cohort's medians.
  Returning nothing or gutting memory can only *lose* the bonus.
* **Outlier / lowball bombing:** the reference is nearest-rank P25 and
  median — one adversarially cheap run cannot move everyone's frontier, and
  the single minimum is never used.
* **Sybil / lineage stuffing:** near-identical submissions collapse to one
  cohort entry before the reference is computed (normalized-source hash,
  else artifact sha256), so cloning a lean harness across hotkeys does not
  widen its influence on the frontier; the coldkey-deduped ledger already
  limits one entry per paying owner.
* **Blast radius:** the cap (5%, hard max 10%) keeps the bonus a tiebreaker
  among comparable-quality agents; quality dominates by construction.
* **Retroactive drift:** epoch-frozen snapshots + insert-once bonus rows —
  published scores never move when new submissions arrive.
* **Cheap-model substitution / miner-reported usage:** only relay-metered
  usage from the validator's trusted broker is read; runs without complete
  audited accounting are simply unqualified.

Residual risks (honest list):

* **Lineage detection is best-effort.** The keys are the exact artifact
  digest and the canonicalized-source hash; a refactored or re-generated
  copy of the same harness evades both and occupies its own cohort slot. The
  shadow anti-copy signals (AST fingerprints, code embeddings) exist but are
  not yet fused into this dedupe; when they graduate from shadow mode the
  lineage key should adopt them.
* **Route-identity pinning is validator-side.** The platform trusts the
  scorer's `status == "complete"` accounting and does not re-verify the
  model/route identity block against the locked contract; a scorer bug that
  marked a partial or off-route run complete would leak into the frontier.
  (The broker enforces the allowlist upstream, so this requires a validator
  fault, not a miner action.)
* **Cohort timing:** the snapshot reflects the board at the first
  leaderboard read of the epoch; a lean harness finalizing minutes later
  waits one epoch (<= 24 h) to influence the frontier. Deliberate — the
  alternative is a moving reference.
* **Insert-once vs. re-scores:** a submission re-scored later (retest lane)
  keeps its originally frozen bonus even if its composite moved; the frozen
  contract was chosen over recomputation. The base composite (which
  dominates) always reflects the latest accepted scores.
* **Median-zero curve is cohort-relative:** if the whole cohort converges to
  near-identical usage, bonuses degenerate toward a step at P25 — bounded by
  the cap, and exactly the "field has converged" signal.

## Test coverage

* `ditto/tests/api_server/test_efficiency.py` — audited-total parsing,
  lineage keys/dedupe, nearest-rank + median reference, interpolation
  boundaries (at/below P25, at/above median, midpoint, degenerate), quality
  gate, N_min inactivity, strictly-upside, floor ratchet, epoch arithmetic.
* `ditto/tests/db/queries/test_efficiency.py` — snapshot roundtrip
  (members JSON), unique epoch key, old-snapshot immutability under new
  epochs, bonus insert-once.
* `ditto/tests/api_server/endpoints/test_public_efficiency.py` — end-to-end:
  active cohort awards frozen bonuses; within-epoch freezing (newcomer scored
  against the frozen frontier, published rows unmoved); N_min inactive board;
  lineage dedupe visible via the snapshot endpoint (no raw digests); v<7 and
  disabled boards write and expose nothing; new-epoch snapshot without
  mutating the old; floor ratchet; validator-ledger fold flag off/on.
* `ditto/tests/api_server/test_config.py` — env parsing + boot validation.
* `ditto/tests/contract/` — regenerated `LedgerEntry` golden (additive
  fields).
