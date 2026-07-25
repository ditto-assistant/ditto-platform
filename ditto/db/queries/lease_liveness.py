"""The one gate every platform-initiated validator lease revocation goes through.

A validator slot runs one benchmark at a time, so when a validator claims work
for a slot that still holds an unfinished lease the platform has to decide
between two irreversible-in-practice options: resume the old lease, or revoke it
and hand the slot something else. Revoking rewrites ``deadline = now``, which
destroys an in-flight benchmark run *and* burns one of the agent's bounded
same-version retries. A ~19-minute run and a retry are both expensive; an idle
slot held to its deadline is cheap. The asymmetry is the whole design.

**Absence of evidence is not evidence of idleness.** The revocation used to be
guarded by a single boolean derived from the last stored
:class:`~ditto.api_models.benchmark_capacity.BenchmarkCapacity` blob: "this slot
is not in ``capacity.active``, therefore nothing is running there". That blob is
a cache of the last heartbeat the platform successfully *ingested*, and it can
silently freeze while the run underneath it is perfectly healthy -- a 500 in the
heartbeat handler rolls the ingest transaction back (``seen_at`` and
``benchmark_capacity`` revert together), a 502 at the edge drops the beat, an
ingest slow enough to be retried loses the write, a deploy restart interrupts the
stream. In every one of those cases the blob keeps answering "slot free" for a
run that is still scoring. That is how three healthy v7 runs were destroyed with
no log line explaining it.

So this module inverts the burden of proof. A lease is revocable only on
**positive, fresh evidence of idleness that postdates the lease**:

1. The heartbeat row exists and its ``seen_at`` is within
   :data:`IDLE_EVIDENCE_MAX_AGE`. Missing, unreadable, or older than that is
   *unknown*, and unknown reads as running.
2. That observation is newer than ``issued_at + LEASE_REPORTING_GRACE``. A blob
   captured before (or in the first moments of) the lease cannot testify about
   it; a validator needs a beat or two to start a run and advertise the slot.
3. The evidence itself says idle: a parseable capacity blob that lists neither
   this slot nor this lease's agent as active (protocol v10+), or a signed
   ``state`` that is not ``running_benchmark`` (pre-v10 reporters, which have no
   per-slot capacity to consult).

Every other outcome -- and every parse failure, missing row, or stale sample --
returns :attr:`LeaseLiveness.idle` ``False``, meaning *assume the run is alive,
do not revoke*. The cost of that conservatism is bounded and known: the lease
still expires at its deadline via
:func:`~ditto.db.queries.tickets.expire_overdue_tickets`, so a genuinely dead
validator's slot is always reclaimed, just on the deadline rather than on the
next poll. A validator that merely restarted keeps heartbeating, so it satisfies
all three conditions within one heartbeat interval and reclaims its own slot
immediately -- which is the case the revocation was actually built for, since a
crashed validator does not poll at all and therefore never reaches this code.

Revocations are also no longer silent: :func:`force_expire_lease` writes a
:class:`~ditto.db.models.ValidatorLeaseAudit` row and a WARNING log carrying the
evidence it acted on, and both outcomes increment a Prometheus counter labelled
by reason, so the near-misses are as visible as the revocations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from prometheus_client import Counter
from pydantic import ValidationError

from ditto.api_models.benchmark_capacity import BenchmarkCapacity
from ditto.api_models.ticket_status import TicketStatus
from ditto.db.models import ValidatorHeartbeat, ValidatorLeaseAudit, ValidatorTicket

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


logger = logging.getLogger(__name__)


# How old the idle observation may be and still count as evidence. Validators
# beat far more often than this, so a sample older than one missed beat plus
# slack is not "the slot is free", it is "the platform has not heard lately" --
# which is exactly the state that destroyed the v7 runs. Deliberately much
# tighter than ``DITTO_VALIDATOR_HEARTBEAT_MAX_AGE_SECONDS`` (300s): that gate
# decides whether a validator may be *given* work, and 300 seconds of blindness
# is fine for handing out a job but is five minutes of licence to destroy one.
IDLE_EVIDENCE_MAX_AGE = timedelta(seconds=120)

# How long after issuance a lease is unconditionally protected. The validator
# has to pull the screened image, generate the dataset, and start the harness
# before the slot shows up as active, and the first heartbeat carrying the new
# slot has to be ingested. Until an observation is newer than this, "the slot is
# not active" is indistinguishable from "the run has not announced itself yet".
LEASE_REPORTING_GRACE = timedelta(minutes=5)


# Reason codes. Everything except IDLE_* means "assume running, do not revoke".
REASON_RUNNING_REPORTED = "running_benchmark_reported"
REASON_HEARTBEAT_MISSING = "heartbeat_missing"
REASON_HEARTBEAT_STALE = "heartbeat_stale"
REASON_EVIDENCE_PREDATES_LEASE = "evidence_predates_lease"
REASON_CAPACITY_UNREADABLE = "capacity_unreadable"
REASON_SLOT_ACTIVE = "slot_active"
REASON_AGENT_ACTIVE_ELSEWHERE = "agent_active_on_another_slot"
REASON_IDLE_CAPACITY = "idle_capacity_reports_slot_free"
REASON_IDLE_STATE = "idle_state_not_running_benchmark"

LEASE_FORCE_EXPIRY_TOTAL = Counter(
    "ditto_validator_lease_force_expiry_total",
    "Validator leases revoked before their deadline by the platform.",
    ("context", "reason"),
)
LEASE_FORCE_EXPIRY_DECLINED_TOTAL = Counter(
    "ditto_validator_lease_force_expiry_declined_total",
    "Revocations declined because the lease was not proven idle.",
    ("context", "reason"),
)


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


@dataclass(frozen=True)
class LeaseLiveness:
    """Whether one lease is provably idle, and the evidence behind the verdict."""

    idle: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        return {"idle": self.idle, "reason": self.reason, **self.evidence}


def _assume_running(reason: str, **evidence: Any) -> LeaseLiveness:
    return LeaseLiveness(idle=False, reason=reason, evidence=evidence)


async def lease_liveness(
    session: AsyncSession,
    *,
    ticket: ValidatorTicket,
    validator_hotkey: str,
    slot_id: str,
    now: datetime,
    running_benchmark_reported: bool = False,
) -> LeaseLiveness:
    """Return whether ``ticket`` is provably idle and may be force-expired.

    Fails safe in every direction: only a fresh, post-issuance, positively idle
    observation returns ``idle=True``. Runs inside the caller's transaction and
    mutates nothing.
    """
    if running_benchmark_reported:
        # The caller already holds a signed report that this slot is running.
        return _assume_running(REASON_RUNNING_REPORTED)

    heartbeat = await session.get(ValidatorHeartbeat, validator_hotkey)
    if heartbeat is None:
        return _assume_running(REASON_HEARTBEAT_MISSING)

    seen_at = _as_utc(heartbeat.seen_at)
    age = now - seen_at
    age_seconds = round(age.total_seconds(), 3)
    if age > IDLE_EVIDENCE_MAX_AGE:
        # The blob may be describing a world several minutes gone. Treating that
        # as proof of an idle slot is exactly the bug this module exists to stop.
        return _assume_running(
            REASON_HEARTBEAT_STALE,
            heartbeat_age_seconds=age_seconds,
            max_age_seconds=IDLE_EVIDENCE_MAX_AGE.total_seconds(),
        )

    issued_at = _as_utc(ticket.issued_at)
    if seen_at <= issued_at + LEASE_REPORTING_GRACE:
        return _assume_running(
            REASON_EVIDENCE_PREDATES_LEASE,
            heartbeat_age_seconds=age_seconds,
            lease_age_seconds=round((now - issued_at).total_seconds(), 3),
            grace_seconds=LEASE_REPORTING_GRACE.total_seconds(),
        )

    if heartbeat.protocol_version >= 10:
        if heartbeat.benchmark_capacity is None:
            return _assume_running(
                REASON_CAPACITY_UNREADABLE, heartbeat_age_seconds=age_seconds
            )
        try:
            capacity = BenchmarkCapacity.model_validate(heartbeat.benchmark_capacity)
        except ValidationError:
            return _assume_running(
                REASON_CAPACITY_UNREADABLE, heartbeat_age_seconds=age_seconds
            )
        for slot in capacity.active:
            if slot.slot_id == slot_id:
                return _assume_running(
                    REASON_SLOT_ACTIVE, heartbeat_age_seconds=age_seconds
                )
            if slot.agent_id == ticket.agent_id:
                # The validator moved this agent's run to a different slot. The
                # lease is still doing work; releasing it kills that run.
                return _assume_running(
                    REASON_AGENT_ACTIVE_ELSEWHERE,
                    heartbeat_age_seconds=age_seconds,
                    active_slot_id=slot.slot_id,
                )
        return LeaseLiveness(
            idle=True,
            reason=REASON_IDLE_CAPACITY,
            evidence={
                "heartbeat_age_seconds": age_seconds,
                "protocol_version": heartbeat.protocol_version,
                "active_slot_ids": [slot.slot_id for slot in capacity.active],
                "admission": capacity.admission,
            },
        )

    # Pre-v10 reporters carry no per-slot capacity, so the signed whole-validator
    # state is the only idleness evidence there is. It is fresh and it postdates
    # the lease, so it is admissible.
    if heartbeat.state == "running_benchmark":
        return _assume_running(
            REASON_RUNNING_REPORTED, heartbeat_age_seconds=age_seconds
        )
    return LeaseLiveness(
        idle=True,
        reason=REASON_IDLE_STATE,
        evidence={
            "heartbeat_age_seconds": age_seconds,
            "protocol_version": heartbeat.protocol_version,
            "state": heartbeat.state,
        },
    )


async def record_lease_revocation(
    session: AsyncSession,
    *,
    ticket: ValidatorTicket,
    now: datetime,
    liveness: LeaseLiveness,
    context: str,
    action: str,
    requested_bench_version: int | None = None,
) -> None:
    """Log, count, and durably record one platform-initiated lease revocation.

    Refuses outright unless ``liveness`` carries an idle verdict, so no call site
    can reintroduce a revocation that acts on absence of evidence. Separate from
    the mutation because the lanes end a lease differently (expired here, closed
    unserviceable in the re-test lane) but must all be equally visible.
    """
    if not liveness.idle:
        raise ValueError(
            f"refusing to revoke a lease that was not proven idle: {liveness.reason}"
        )
    evidence = {
        **liveness.as_payload(),
        "context": context,
        "slot_id": ticket.slot_id,
        "ticket_bench_version": ticket.bench_version,
        "requested_bench_version": requested_bench_version,
        "purpose": str(ticket.purpose),
        "issued_at": _as_utc(ticket.issued_at).isoformat(),
        "original_deadline": _as_utc(ticket.deadline).isoformat(),
        "lease_age_seconds": round(
            (now - _as_utc(ticket.issued_at)).total_seconds(), 3
        ),
        "attempt_count": ticket.attempt_count,
    }
    evidence["action"] = action
    logger.warning(
        "revoking validator lease action=%s agent=%s validator=%s slot=%s "
        "bench_version=%s lease_age_s=%s reason=%s evidence=%s",
        action,
        ticket.agent_id,
        ticket.validator_hotkey,
        ticket.slot_id,
        ticket.bench_version,
        evidence["lease_age_seconds"],
        liveness.reason,
        evidence,
    )
    LEASE_FORCE_EXPIRY_TOTAL.labels(context=context, reason=liveness.reason).inc()
    session.add(
        ValidatorLeaseAudit(
            audit_id=uuid4(),
            agent_id=ticket.agent_id,
            validator_hotkey=ticket.validator_hotkey,
            slot_id=ticket.slot_id,
            bench_version=ticket.bench_version,
            action=action,
            reason=liveness.reason,
            context=context,
            evidence=evidence,
            recorded_at=now,
        )
    )


async def force_expire_lease(
    session: AsyncSession,
    *,
    ticket: ValidatorTicket,
    now: datetime,
    liveness: LeaseLiveness,
    context: str,
    requested_bench_version: int | None = None,
) -> None:
    """Expire a lease proven idle, leaving a log line and an audit row behind."""
    await record_lease_revocation(
        session,
        ticket=ticket,
        now=now,
        liveness=liveness,
        context=context,
        action="force_expired",
        requested_bench_version=requested_bench_version,
    )
    ticket.status = TicketStatus.EXPIRED
    ticket.deadline = now
    ticket.retry_after = now
    await session.flush()


def record_declined_force_expiry(
    *,
    ticket: ValidatorTicket,
    liveness: LeaseLiveness,
    context: str,
) -> None:
    """Log + count a revocation the liveness gate refused.

    Deliberately not an audit row: a declined revocation is the *safe* outcome
    and repeats on every poll of a busy slot, so it belongs in metrics and logs
    rather than in an append-only table.
    """
    LEASE_FORCE_EXPIRY_DECLINED_TOTAL.labels(
        context=context, reason=liveness.reason
    ).inc()
    logger.info(
        "declining to force-expire live validator lease agent=%s validator=%s "
        "slot=%s bench_version=%s reason=%s evidence=%s",
        ticket.agent_id,
        ticket.validator_hotkey,
        ticket.slot_id,
        ticket.bench_version,
        liveness.reason,
        liveness.as_payload(),
    )


async def maybe_force_expire_lease(
    session: AsyncSession,
    *,
    ticket: ValidatorTicket,
    validator_hotkey: str,
    slot_id: str,
    now: datetime,
    context: str,
    running_benchmark_reported: bool = False,
    requested_bench_version: int | None = None,
) -> bool:
    """Force-expire ``ticket`` iff it is provably idle. Returns whether it did.

    The single entry point every revocation site uses, so the evidence rule, the
    log line, the audit row, and the metrics cannot drift apart between them.
    """
    liveness = await lease_liveness(
        session,
        ticket=ticket,
        validator_hotkey=validator_hotkey,
        slot_id=slot_id,
        now=now,
        running_benchmark_reported=running_benchmark_reported,
    )
    if not liveness.idle:
        record_declined_force_expiry(ticket=ticket, liveness=liveness, context=context)
        return False
    await force_expire_lease(
        session,
        ticket=ticket,
        now=now,
        liveness=liveness,
        context=context,
        requested_bench_version=requested_bench_version,
    )
    return True


__all__ = [
    "IDLE_EVIDENCE_MAX_AGE",
    "LEASE_FORCE_EXPIRY_DECLINED_TOTAL",
    "LEASE_FORCE_EXPIRY_TOTAL",
    "LEASE_REPORTING_GRACE",
    "LeaseLiveness",
    "force_expire_lease",
    "lease_liveness",
    "maybe_force_expire_lease",
    "record_declined_force_expiry",
    "record_lease_revocation",
]
