# Validator retry & exhaustion model

How a submission gets scored, how many times a validator may re-attempt it, and
how an operator recovers a submission stranded by validator-side infrastructure.

## Scoring quorum

A submission (`agent`) is finalized once **`SCORING_QUORUM = 3`** distinct
validators each post a valid score. Each validator leases a `ValidatorTicket`
for the agent (one ticket per `(agent, bench_version, validator)`), runs the
benchmark, and either posts a score (ticket → `scored`) or lets the lease lapse.

## Per-validator attempt budget

Each validator gets a bounded number of attempts **per benchmark version**:

| Constant | Value | Meaning |
| --- | --- | --- |
| `MAX_ATTEMPTS_PER_VERSION` | `2` | Base attempts a validator may spend on one agent+version. |
| `manual_retry_grants` | `0`+ | Per-ticket operator extension; raises the cap for that ticket. |
| `infra_retry_grants` | `0`–`8` | Per-ticket automatic extension earned when a lease fails on validator-side infrastructure; raises the cap so an outage doesn't spend the agent's budget. |
| `RETRY_COOLDOWN` | `6h` | Delay before the **same** validator may re-lease after a timeout. |

The issuance cap for a ticket is:

```
attempt_count  >=  MAX_ATTEMPTS_PER_VERSION + manual_retry_grants + infra_retry_grants
                                                                  →  no more reissue
```

`attempt_count` increments each time the expired ticket is **re-leased**
(`issue_ticket`), not when it fails. Key consequences:

- **Another validator may pick the agent up immediately** — the cap and cooldown
  are per-validator, so a timeout on one validator never blocks the other two.
- **A benchmark-version bump resets the budget.** Tickets are keyed by
  `bench_version`, so repaired scoring software revisits the artifact with a
  fresh 2-attempt budget on the new version.
- **Infrastructure failures don't consume the agent's budget.** The validator
  reports a signed `fail_job` with `reason` (`infrastructure` vs
  `scoring_error`). On `infrastructure` the platform bumps `infra_retry_grants`
  (bounded at `8`), which offsets the `attempt_count` the reissue adds — so a
  validator-side outage (e.g. a model-relay/upstream failure) never spends the
  agent's genuine `MAX_ATTEMPTS_PER_VERSION` budget. A `scoring_error` is the
  agent's own failure and consumes an attempt normally.
- **Infrastructure retries back off; scoring failures reissue immediately.** A
  `scoring_error` sets `retry_after = now` (immediate reissue for another
  validator/attempt). An `infrastructure` failure instead sets an **escalating
  cooldown** — `infra_retry_backoff(infra_retry_grants)`, doubling from 2m up to
  a 30m cap — so a *sustained* provider/relay outage is retried to success
  without immediate back-to-back re-leases hammering the failing provider (an
  inference burst). Both are well short of the 6h agent-failure timeout cooldown.

## Timeout vs. explicit failure

- **Timeout** (`expire_overdue_tickets`): a lease passes its deadline unscored →
  ticket `expired`, `retry_after = deadline + RETRY_COOLDOWN` (6h).
- **Explicit fail** (`fail_job`): the validator reports terminal failure →
  ticket `expired`, `retry_after = now` (immediate reissue, no 6h wait).

## When is a submission "stuck"?

A below-quorum submission is one of these retry states (surfaced per agent and
fleet-wide, see below):

| State | Meaning | Needs an operator? |
| --- | --- | --- |
| `running` | A validator holds a live ticket right now. | No |
| `retry_available` | An expired ticket is off cooldown, budget to spare; re-leases next sweep. | No |
| `cooling_down` | Expired ticket has budget but is waiting out `RETRY_COOLDOWN`. | No |
| `exhausted` | Every remaining validator burned its attempt budget; cannot advance without a grant. | **Yes** |
| `queued` | Below quorum with slots simply never leased yet. | No |

Only **`exhausted`** needs a human. The most common cause is a validator-side
infrastructure outage (e.g. a model-relay/upstream outage) that burned attempts
on failures that were not the agent's fault.

## Visibility

- **Per agent:** `GET /api/v1/admin/validation-retries/{agent_id}` — full ticket
  ledger, `automatic_retry_available`, `recovery_allowed`, `blocking_reason`.
- **Fleet-wide:** `GET /api/v1/admin/validation-retries` — every below-quorum
  submission with its `retry_state`, sorted most-urgent first, plus fleet
  `counts` per state. Filter with `?state=exhausted`. This is the triage view;
  it replaces sweeping the per-agent route one agent at a time.

## Operator recovery

For an `exhausted` submission after verifying the failure was validator-side
infrastructure:

1. Read the fleet list (or per-agent detail) to confirm `recovery_allowed:true`
   and capture the `snapshot`.
2. `POST /api/v1/admin/validation-retries/{agent_id}/retry` with the `snapshot`,
   an idempotency `request_id`, and a `reason`. This raises `manual_retry_grants`
   on exactly the minimum number of expired tickets needed to restore quorum and
   clears their cooldown. Accepted scores, screening verdicts, and ticket history
   are preserved; it is **not** a rescreen.

To recover several stranded submissions at once (e.g. the batch left exhausted
by one outage), `POST /api/v1/admin/validation-retries/batch-retry` with a shared
`reason` and one `{agent_id, request_id, expected_snapshot}` item per agent. Each
item is gated and snapshot-checked exactly like the single route; an item whose
state moved is **skipped** with a reason rather than force-granted, and all grants
commit together.

Recoveries are bounded (`MAX_OPERATOR_RECOVERIES_PER_AGENT = 3`) and audited in
`ValidatorRetryRecovery`.
