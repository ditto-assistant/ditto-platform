"""Reads and mutations against the ``owner_attestations`` table.

This is the *copy-screening* owner signal. It is stronger than the
payment-record inference in :mod:`ditto.db.queries.ownership` -- a signature
proves control of a key, where a shared coldkey only suggests a shared payer --
so where both are surfaced, this one outranks it.

Evidence grading does not gate the exemption
--------------------------------------------
Each half of a link is proved by either a hotkey or a coldkey signature, and
the pair is graded ``coldkey-coldkey`` / ``mixed`` / ``hotkey-hotkey``. That
grade is **reviewer context, not a gate**: any two valid halves establish the
link, and screening treats all three grades identically.

Sufficiency is binary because the question the exemption turns on -- "are these
two hotkeys the same operator?" -- is answered by either key type. A hotkey
signature is not a weaker *answer*; it is a weaker *key*. Gating on the grade
would mean the weaker grade did something, and the only somethings available
are "hold anyway" or "flag for an operator" -- both of which reintroduce the
operator dependency this feature exists to remove, and both of which would
penalise exactly the recovery cases that motivated accepting two key types (a
miner who lost the old hotkey's key but kept the coldkey, or the reverse).

The grade earns its place elsewhere: it is what a reviewer reads when
adjudicating a dispute, and it is the handle an operator would use to find and
revoke links resting on a key type later found to be compromised. Both are real
value without gating behaviour.

Not an emission input
---------------------
Deliberately **not** wired into
:func:`ditto.db.queries.scores.emission_owner_key`. That expression is the
single authority for one-slot-per-owner partitioning, and an attestation must
not be able to move a submission into or out of an emission position. See the
module docstring of :mod:`ditto.api_server.attestation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import desc, or_, select

from ditto.db.models import Agent, EvaluationPayment, OwnerAttestation

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession


class AttestationReplayedError(Exception):
    """The nonce on this attestation has already been recorded."""


@dataclass(frozen=True)
class LinkedHotkey:
    """One hotkey proven to be the same operator as the one asked about."""

    hotkey: str
    """SS58 of the linked hotkey."""

    attestation_id: UUID
    """The link that proves it, so a reviewer can go straight to the evidence."""

    grade: str
    """``coldkey-coldkey`` / ``mixed`` / ``hotkey-hotkey``. Context, not a gate."""


async def get_bound_coldkey_for_hotkey(
    session: AsyncSession, *, hotkey: str
) -> str | None:
    """The coldkey the platform independently associates with ``hotkey``.

    The most recent payment-time coldkey across that hotkey's submissions, or
    ``None`` when it has never funded one. This is what a coldkey-signed half
    is checked against -- a binding learned from on-chain payment proofs, never
    from the attestation itself, which is what makes a coldkey proof mean
    anything.

    Most-recent rather than any-ever on purpose: if a hotkey changed hands, the
    previous owner's stale payment rows must not keep authorising them to sign
    for it.
    """
    return await session.scalar(
        select(EvaluationPayment.miner_coldkey)
        .join(Agent, Agent.agent_id == EvaluationPayment.agent_id)
        .where(Agent.miner_hotkey == hotkey)
        .order_by(desc(EvaluationPayment.timestamp))
        .limit(1)
    )


async def record_attestation(
    session: AsyncSession,
    *,
    netuid: int,
    hotkey_lo: str,
    hotkey_hi: str,
    nonce: UUID,
    issued_at: datetime,
    lo_key_kind: str,
    lo_signer: str,
    lo_signature: str,
    hi_key_kind: str,
    hi_signer: str,
    hi_signature: str,
) -> OwnerAttestation:
    """Persist a verified owner link.

    The caller is responsible for having verified both halves first
    (:func:`ditto.api_server.attestation.verify_link`); this function enforces
    only the relational invariants.

    Raises:
        AttestationReplayedError: When ``nonce`` was already used, or when an
            active link already joins this pair.
    """
    existing_nonce = await session.scalar(
        select(OwnerAttestation.attestation_id).where(OwnerAttestation.nonce == nonce)
    )
    if existing_nonce is not None:
        raise AttestationReplayedError(
            "this attestation nonce has already been submitted"
        )

    active = await session.scalar(
        select(OwnerAttestation.attestation_id).where(
            OwnerAttestation.netuid == netuid,
            OwnerAttestation.hotkey_lo == hotkey_lo,
            OwnerAttestation.hotkey_hi == hotkey_hi,
            OwnerAttestation.revoked_at.is_(None),
        )
    )
    if active is not None:
        raise AttestationReplayedError(
            "an active attestation already links these two hotkeys"
        )

    row = OwnerAttestation(
        netuid=netuid,
        hotkey_lo=hotkey_lo,
        hotkey_hi=hotkey_hi,
        nonce=nonce,
        issued_at=issued_at,
        lo_key_kind=lo_key_kind,
        lo_signer=lo_signer,
        lo_signature=lo_signature,
        hi_key_kind=hi_key_kind,
        hi_signer=hi_signer,
        hi_signature=hi_signature,
    )
    session.add(row)
    await session.flush()
    return row


async def revoke_attestation(
    session: AsyncSession,
    *,
    attestation_id: UUID,
    revoked_by: str,
    reason: str,
    now: datetime | None = None,
) -> OwnerAttestation | None:
    """Mark an attestation inactive from now on.

    Prospective by design: screening decisions already taken under the link
    stand. Returns ``None`` when the id is unknown or already revoked.

    Creating a link needs both endpoints; withdrawing one needs only one of
    them (or an operator). Staying linked to someone is an ongoing claim about
    a shared identity, and either party should be able to stop asserting it
    without needing the other's cooperation -- which they may well not get if
    the link has gone bad. The asymmetry is deliberate.
    """
    row = await session.scalar(
        select(OwnerAttestation).where(
            OwnerAttestation.attestation_id == attestation_id,
            OwnerAttestation.revoked_at.is_(None),
        )
    )
    if row is None:
        return None
    row.revoked_at = now or datetime.now(UTC)
    row.revoked_by = revoked_by
    row.revoked_reason = reason
    await session.flush()
    return row


async def list_linked_hotkeys(
    session: AsyncSession, *, hotkey: str, netuid: int
) -> list[LinkedHotkey]:
    """Hotkeys proven to be the same operator as ``hotkey``.

    **Direct links only** -- see
    :data:`ditto.api_server.attestation.MAX_LINK_DEPTH` for why the transitive
    walk was dropped when the link became symmetric.

    Ordered by hotkey so callers and tests see a stable list. Empty for a miner
    who has never attested, which is the overwhelming majority -- callers
    should treat the empty case as the fast path.
    """
    from ditto.api_server.attestation import evidence_grade

    rows = await session.execute(
        select(OwnerAttestation).where(
            OwnerAttestation.netuid == netuid,
            OwnerAttestation.revoked_at.is_(None),
            or_(
                OwnerAttestation.hotkey_lo == hotkey,
                OwnerAttestation.hotkey_hi == hotkey,
            ),
        )
    )
    linked = [
        LinkedHotkey(
            hotkey=row.hotkey_hi if row.hotkey_lo == hotkey else row.hotkey_lo,
            attestation_id=row.attestation_id,
            grade=evidence_grade(row.lo_key_kind, row.hi_key_kind),  # type: ignore[arg-type]
        )
        for row in rows.scalars().all()
    ]
    return sorted(linked, key=lambda link: link.hotkey)


async def list_attestations_for_hotkey(
    session: AsyncSession,
    *,
    hotkey: str,
    netuid: int,
    include_revoked: bool = True,
) -> Sequence[OwnerAttestation]:
    """Every attestation row naming ``hotkey`` on either side.

    The reviewer-facing read: revoked links are included by default, because
    "this link was live last Tuesday" is exactly the question a dispute turns
    on.
    """
    stmt = select(OwnerAttestation).where(
        OwnerAttestation.netuid == netuid,
        or_(
            OwnerAttestation.hotkey_lo == hotkey,
            OwnerAttestation.hotkey_hi == hotkey,
        ),
    )
    if not include_revoked:
        stmt = stmt.where(OwnerAttestation.revoked_at.is_(None))
    rows = await session.execute(stmt.order_by(OwnerAttestation.created_at))
    return rows.scalars().all()
