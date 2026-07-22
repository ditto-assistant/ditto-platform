"""Common Random Numbers (CRN) seed derivation — platform mirror.

The **authoritative** implementation lives in ``ditto-subnet`` at
``ditto/validator/crn.py``; validators derive the seeds they benchmark on from
it. This module is a **byte-for-byte mirror** so the platform can independently
re-derive the champion-anchored seed set and *validate* a submitted confirmation
seed against it (anti-grind: a validator cannot cherry-pick a favourable seed).
Keep it exactly aligned with the subnet — the digest encoding is consensus.

    crn_seed(agent_ids, version, k) = sha256(
        sorted(agent_ids) each ‖ b"\\x00", then str(version) ascii,
        then (if k>0) b"\\x00k" ‖ str(k) ascii
    ) → first 8 bytes little-endian unsigned & (2**63 - 1)

The int63 masking mirrors dittobench-api's ``gen.FreshSeed`` (``int64(uint64 >>
1)``) so the value round-trips through the wire unchanged.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from uuid import UUID

# Mask to a non-negative signed-63-bit integer, matching dittobench-api's
# FreshSeed (``int64(uint64 >> 1)``): JSON-clean and never negative.
_INT63_MASK = (1 << 63) - 1


def crn_seed(agent_ids: Iterable[str], *, version: int, k: int = 0) -> int:
    """Deterministic dataset seed for a CRN comparison over ``agent_ids`` at
    ``version``. Order-independent (the *set* of compared agents determines the
    seed) and pure, so every validator computes the same value.

    ``k`` indexes a confirmation replicate: a dethrone-grade comparison runs the
    compared agents on ``K`` common seeds (k = 0..K-1). ``k=0`` is byte-identical
    to the original single-seed derivation.

    Byte-for-byte aligned with ``ditto-subnet/ditto/validator/crn.py::crn_seed``.
    """
    h = hashlib.sha256()
    for aid in sorted(agent_ids):
        h.update(aid.encode("utf-8"))
        h.update(b"\x00")
    h.update(str(int(version)).encode("ascii"))
    if k > 0:
        h.update(b"\x00k")
        h.update(str(int(k)).encode("ascii"))
    return int.from_bytes(h.digest()[:8], "little", signed=False) & _INT63_MASK


def confirmation_seeds(
    agent_ids: Iterable[str], *, version: int, count: int
) -> list[int]:
    """The ``count`` common confirmation seeds for one comparison, k = 0..count-1.

    ``count <= 1`` degrades to the single classic CRN seed. Byte-for-byte aligned
    with ``ditto-subnet/ditto/validator/crn.py::confirmation_seeds``.
    """
    ids = list(agent_ids)
    n = max(1, int(count))
    return [crn_seed(ids, version=version, k=k) for k in range(n)]


def champion_anchored_seeds(
    champion_agent_id: UUID, *, version: int, max_seeds: int
) -> list[int]:
    """The champion-anchored CRN seed set for the top-5 shared-seed rescore lane.

    The seed set keys only on the *champion's* agent id (LOCKED, cf.
    ``docs/top5-rescore-lane.md`` §2), so the shared baseline stays stable as the
    tail churns and moves only when the champion is dethroned. ``version`` is the
    **major** benchmark version, so successive versions get distinct seeds. This
    is exactly what the subnet's ``top5_confirmation_set`` derives via
    ``confirmation_seeds([str(champion_id)], version=current_version, count)`` —
    the platform mirrors it to bound the anti-grind check to the first
    ``max_seeds`` (``TOP5_MAX_CONFIRMATION_SEEDS``) replicate indices.
    """
    return confirmation_seeds(
        [str(champion_agent_id)], version=version, count=max_seeds
    )
