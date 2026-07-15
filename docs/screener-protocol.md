# Screening protocol dependency

The API owns queue state, leases, verdict acceptance, screening history, and
public status projection. The public `ditto-screener` repository owns the
build/run worker. They share only `ditto-screening-protocol`, pinned in
`pyproject.toml` and `uv.lock` to an exact public-repository commit.

The protocol package contains request/response models, `AgentStatus`, artifact
metadata, `SCREENING_POLICY_VERSION`, and the canonical signing function. The
API never imports worker application code.

The pin remains policy 6. Moving the package changes neither public statuses nor
canonical verdict bytes. The platform accepts only the exact current policy and
lease-bound attempt, and continues rejecting expired, late, conflicting,
wrong-agent, wrong-policy, and wrong-signer verdicts. Private screener outcomes
do not widen the API: pass, deterministic reject, and retryable infrastructure
use the existing signed boolean verdict; quarantine and inconclusive submit no
public verdict and leave the lease authoritative.

Merge `ditto-screener` first, deploy its core-only v6 worker alongside the old
v6 worker, then merge/deploy this pin. Stop the old worker only after equivalent
signed pass/reject/retry behavior is observed. Merge the subnet runtime-removal
PR last. This order requires no migration and preserves screening history,
active leases, waiting-validator submissions, evaluations, and score receipts.
## Quarantine management

A current worker can return a signed, attempt-bound `quarantine` outcome with
only bounded reason and evidence digests. The platform completes that exact
lease, moves the submission to the non-scoreable `quarantined` state, and
appends a `screening_quarantines` row. Raw source, model transcripts, private
prompts, and challenge contents are never stored in the platform database.

Backroom and other operator clients use the bearer-protected endpoints below:

- `GET /api/v1/admin/screening-quarantines`
- `GET /api/v1/admin/screening-quarantines/{quarantine_id}`
- `POST /api/v1/admin/screening-quarantines/{quarantine_id}/resolve`

Resolution actions are append-only in `resolution_history`. A resolved rejection may
be corrected to `release` while the agent is still rejected; other second resolutions
remain conflicts. This narrow correction path preserves the original actor, reason,
and timestamp while allowing a reviewed false positive to resume evaluation.

Resolution requires `X-Admin-Actor` and one of `release`, `rescreen`, or
`reject`. A row lock makes resolution single-writer. Release pins a dataset if
needed and promotes to evaluation; rescreen returns the preserved submission to
the screener queue; reject retains the submission and prior scores but prevents
evaluation until a future policy-version rescreen.
