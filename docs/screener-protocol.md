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

## Miner disputes

A miner may dispute a resolved quarantine rejection exactly once per submission.
The request is accepted only while the submission remains rejected and only when
its sr25519 signature verifies against the hotkey recorded at upload. The miner
signs the following canonical UTF-8 payload, where `message` is trimmed before
hashing:

```text
ditto-dispute-v1:{agent_id}:{sha256(message)}
```

The submission dashboard generates that payload and a ready-to-run command after
the miner enters the local wallet and hotkey names:

```bash
btcli wallet sign --wallet-name '<wallet-name>' --wallet-hotkey '<hotkey-name>' \
  --use-hotkey --message 'ditto-dispute-v1:<agent_id>:<sha256>' --json-output
```

`--use-hotkey` prevents an accidental coldkey signature. The miner pastes the
128-character `signed_message` value from the command output into the dispute
form. Wallet and hotkey names are used only to construct the command in the
browser and are not included in the dispute request.

`POST /api/v1/public/agent/{agent_id}/dispute` accepts a 20–1000 character
message and a 128-character hexadecimal signature. Database uniqueness on both
`agent_id` and `quarantine_id` enforces the one-dispute limit under concurrent
requests. The public submission pipeline exposes only dispute status, timestamps,
and the final `release` or `uphold` result; the miner's message remains private.

Operators use the same admin bearer-token boundary as quarantine review:

- `GET /api/v1/admin/screening-disputes`
- `POST /api/v1/admin/screening-disputes/{dispute_id}/resolve`

Resolution requires `X-Admin-Actor`. `release` atomically records the accepted
dispute, changes the effective quarantine resolution to release, and returns the
submission to evaluation. `uphold` records a final review while leaving the
submission rejected. The original rejection and its operator reason remain in
append-only quarantine history in either case.
