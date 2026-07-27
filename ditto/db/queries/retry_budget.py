"""What a failure that was not the agent's fault costs the miner: nothing.

A ticket carries a bounded per-version attempt budget, and every reissue spends
one. Anything that ends a lease for reasons outside the agent's control has to
hand that attempt back, or a miner pays for faults that were never theirs. Two
things qualify: a validator-side infrastructure failure, and the platform
revoking a live lease itself.

The compensation raises the *cap* rather than rewriting ``attempt_count``. The
ledger keeps saying how many leases this agent actually consumed; the grant says
how many of those the miner should not be billed for. Those are different facts
and collapsing them loses the audit trail.

This lives in its own module because the two places that grant it sit on opposite
sides of an import edge: :mod:`ditto.db.queries.tickets` owns issuance and the
budget arithmetic and already imports :mod:`ditto.db.queries.lease_liveness`, the
revocation gate, so the gate cannot import back. ditto-platform#460 established
the rule -- "do not bill a miner for a lease the platform itself revoked" -- but
implemented it in exactly one call site, the signed ``fail_job`` handler, and the
platform's own revocation path went on billing. One definition imported by both
is what stops that from recurring.
"""

from __future__ import annotations

from datetime import timedelta

from ditto.db.models import ValidatorTicket

# An infrastructure failure, or a lease the platform revoked, is never the
# agent's fault, so it earns a compensating grant that offsets the attempt the
# reissue consumes. Bounded so a persistent validator-side outage cannot re-lease
# one agent forever.
MAX_INFRA_RETRY_GRANTS = 8

# Infrastructure retries reissue quickly (no 6h agent-failure cooldown) so a
# transient blip recovers fast, but back-to-back re-leases during a *sustained*
# provider/relay outage would hammer the failing provider (an inference burst).
# The cooldown before the next infra retry therefore doubles per grant already
# earned, capped, so the agent is still retried to success while the failing
# provider gets breathing room.
INFRA_RETRY_BACKOFF_BASE = timedelta(minutes=2)
INFRA_RETRY_BACKOFF_CAP = timedelta(minutes=30)


def infra_retry_backoff(infra_retry_grants: int) -> timedelta:
    """Cooldown before an infrastructure-failed lease may be re-leased.

    ``infra_retry_grants`` is the count *after* this failure bumped it (so the
    first infra failure passes ``1``). Doubles per prior grant, capped at
    :data:`INFRA_RETRY_BACKOFF_CAP`.
    """
    if infra_retry_grants <= 1:
        return INFRA_RETRY_BACKOFF_BASE
    # Clamp the exponent so a large count can't overflow the timedelta multiply;
    # anything past the cap is clamped to it anyway (real inputs are <= 8).
    steps = min(infra_retry_grants - 1, 20)
    scaled = INFRA_RETRY_BACKOFF_BASE * (2**steps)
    return min(scaled, INFRA_RETRY_BACKOFF_CAP)


def grant_no_fault_retry(ticket: ValidatorTicket) -> bool:
    """Offset the attempt the coming reissue will charge. Returns whether it did.

    Returns ``False`` once :data:`MAX_INFRA_RETRY_GRANTS` is reached, so a
    persistently sick lease cannot mint attempts forever -- the bound is what
    makes an automatic grant safe to hand out without an operator in the loop.
    """
    if ticket.infra_retry_grants >= MAX_INFRA_RETRY_GRANTS:
        return False
    ticket.infra_retry_grants += 1
    return True
