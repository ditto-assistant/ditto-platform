# Top-5 continual shared-seed rescore lane

**Status:** design / RFC.
**Depends on:** ditto-platform #195 + ditto-subnet #161 (`confirm uncertain KOTH
challengers`) — this generalizes them.

## Goal

Continually re-score the **emission set (champion + 4 tail = top 5)** on fresh,
**uniform-but-random** benchmark seeds so their medians keep widening over more
data. Every top-5 agent is scored on the *same* dataset each round (random yet
identical across the five), which cancels dataset-difficulty and keeps
**intra-top-5 ranking fair** — the champion holds the crown on paired evidence,
not on a lucky seed. Runs on a **tempo** (once every *T* ∈ 2–8 tempos, a tempo =
360 blocks ≈ 72 min) so it is negligible under normal evaluation load. Membership
**follows the set**: a new agent entering the top 5 automatically joins; one that
drops out stops. A fresh entrant gets **catch-up** (extra seeds/round) until its
confirmation depth matches the incumbents.

## TL;DR — most of this already exists

This is **~80% wiring, not a redesign.** The consensus-critical machinery is
already built and deployed on the subnet side:

| Piece you need | Already exists | Where |
| --- | --- | --- |
| Deterministic uniform-but-random shared seed across a set of agents | **CRN** `crn_seed(agent_ids, version, k) = SHA256(sorted(ids)‖version‖k)` — every validator derives the identical seed | `ditto-subnet ditto/validator/crn.py:32-62` |
| A loop that continually re-scores champion + tail on those seeds | `worker._rescore_stale_champions` (every sweep) → `_rescore_stale_champion_and_tail` / `_confirm_contested_dethrone` | `ditto-subnet ditto/validator/worker.py:787-962` |
| Median that widens over more seeds without more Score rows | multi-seed data lives **in-row** in `Score.details.confirmation_composites`; the median (`_effective_composite`) and lower-median ledger read are **N-agnostic** | `ditto-subnet weights.py`; `ditto-platform scores.py:665-753` |
| The KOTH fold consuming paired shared-seed evidence | `_paired_dethrone` / `_beats` / `contested_confirmation_set` | `ditto-subnet weights.py:266-494`; platform mirror `koth.py:99-198` |
| A platform confirmation-ticket lane (for **one** challenger) | #195/#161 | (open PRs) |

**The only gap:** the subnet's rescore submissions carry **no `ticket_deadline`**
(`worker.py:1298` default `None`), and the platform `submit_score` **rejects any
submission without a live ticket** (`endpoints/validator.py:1411-1415,
1435-1442`). So the machinery is inert against the platform. #195 opens that lane
for the single uncertain leader; this doc opens it for the **whole top 5,
continually**.

## The design

### 1. The lane = a parallel issuer, not a new ticket column

There is **no ticket type/kind column** today; `ValidatorTicket`'s only
discriminator is `bench_version` (`db/models.py:1220-1224`). The established
precedent for a second lane is **a separate issuer keyed off a durable table**,
wired into `request_job`: `issue_rollout_ticket` (`benchmark_rollout.py:579`)
already sits beside `issue_ticket` (`tickets.py:125`) in
`endpoints/validator.py:840-973`. The top-5 lane is a **third issuer**
(`issue_top5_confirmation_ticket`) selected there, ahead of normal evaluation
only when the tempo says so. It grants a **confirmation ticket** the existing
`submit_score` path can bind to (the #195 mechanism, generalized), so it **never
consumes an agent's k=3 evaluation budget** and adds **no fourth scorer** to the
authoritative quorum.

### 2. Champion-anchored shared seeds (recommended over set-anchored)

CRN can key the seed on any set of agent ids. Two options:

- **Set-anchored** (literal "seed derived from the top-5 set"): `crn_seed({all 5},
  …)`. Problem: the moment membership changes, the seed set changes, the shared
  baseline re-randomizes, and accumulated paired history is stranded.
- **Champion-anchored** (recommended, = #195's choice, `worker.py:933-937`): seeds
  are anchored to the **champion's** agent_id; the champion mints one fresh CRN
  seed per tempo and **all five score that same seed**. This still gives "all
  top-5 on one uniform-but-random seed each round," but the baseline is **stable**
  as tail members churn, a newcomer simply **starts sharing the champion's
  existing seeds** (O(1), and catch-up = give it more of them), and history
  accumulates. When the champion is dethroned the anchor moves to the new
  champion — rare and meaningful. **Recommend champion-anchored.**

Consensus safety: the seed is a pure function of `(champion_id, version,
tempo_index)` — every validator derives it identically; no wall-clock, no RNG.

### 3. Where the scores land — in-row, no new rows

Each rescore does **not** add a Score row. The validator folds its K per-seed
composites into **one** median-run row and writes the per-seed arrays into
`Score.details.confirmation_composites` / `confirmation_seeds` (+ `composite_stderr`)
(`worker.py:1463-1533`; surfaced by `endpoints/scoring.py:274-276`). The ledger's
lower-median (`scores.py:665-753`) and the KOTH fold already median over those
in-row seeds. **Consequence:** none of the "exactly-3" invariants are touched — no
4th ticket (`tickets.py:471-479` untouched), no 4th `Score` PK row
(`db/models.py:829-831`), no change to k=3 finalization. "More seeds" = a longer
`confirmation_composites` array, which the fold already handles.

### 4. Membership, tempo, catch-up

- **Membership** = the current emission set (champion via `_champion`, tail via
  `_tail`, `weights.py:346-367`) recomputed each round. New entrant in → gets the
  lane; drop out → doesn't. No manual list.
- **Tempo** = gate the issuer to fire the confirmation round once per *T* tempos
  (config `TOP5_RESCORE_TEMPO`, default e.g. 4). Under normal load one extra
  shared-seed round every ~5 h is negligible.
- **Catch-up** = a tail agent whose `confirmation_seeds` count is below the
  incumbents' gets **2× seeds/round** (score the two most-recent champion seeds it
  is missing) until its depth converges. Bounded so it can never exceed the
  champion's own seed count.

## What does NOT change (and why that's the point)

Because rescores land in `details` and not as new rows/validators, all of these
stay exactly as-is: `SCORING_QUORUM = 3`, the k=3 ticket cap, the
one-ticket-per-validator index, `submit_score` finalization, and the
`get_score_continuation_floor` two-score bound. The lane is **additive** to the
authoritative k=3 record; it only enriches the top 5's `details` with more shared
seeds, which the fold already medians. That is what keeps it consensus-safe.

## PR breakdown

Ordering: **land #195 + #161 first** (they establish the confirmation-ticket
endpoint + the subnet confirmation submit path this generalizes). Then:

1. **ditto-platform — top-5 confirmation issuer + tempo** (build on #195's
   endpoint): generalize the single-leader confirmation ticket to the emission set
   (champion + 4 tail), gate issuance on `TOP5_RESCORE_TEMPO`, select the set from
   `list_eligible_ledger`/`project_koth`, derive champion-anchored CRN seeds, wire
   into `request_job`. Leaderboard shows "top-5 shared-seed depth."
2. **ditto-subnet — point the existing rescore loop at the lane + catch-up**: have
   `_rescore_stale_champion_and_tail` claim the platform confirmation ticket and
   submit **with** the granted `ticket_deadline` (the one missing field), and
   implement the 2× catch-up seed selection for new tail entrants.
3. **(optional) dittobench-api** — only if datagen needs a set-keyed entry point;
   likely none (CRN seed → existing generator).

Each platform/subnet PR opens as a **draft** with this doc linked, because the
change feeds dethroning → emissions → chain weights and must be reviewed against
the consensus fold before merge.

## Open decisions (need a call before implementation)

1. **Champion-anchored vs set-anchored seeds** — recommend champion-anchored
   (§2). Confirm.
2. **Extend #195 vs stack on it** — recommend a **stacked** PR (keep #195's
   reviewed single-leader path intact; add the top-5 generalization on top) rather
   than rewriting #195 in place.
3. **Tempo default `T`** — 2–8 range given; recommend **4** (~one round / 5 h) as
   the default, tunable by config.
4. **Catch-up rate** — recommend **2×** until depth-converged, hard-capped at the
   champion's seed count.
