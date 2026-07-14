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
