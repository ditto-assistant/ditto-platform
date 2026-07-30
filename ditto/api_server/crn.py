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
from collections.abc import Iterable, Mapping
from uuid import UUID

from ditto.api_server.koth import TOP5_MAX_CONFIRMATION_SEEDS

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


def continual_anchor_horizon(
    seeds_by_agent: Mapping[UUID, Iterable[int]],
    *,
    minimum: int = TOP5_MAX_CONFIRMATION_SEEDS,
) -> int:
    """Enough deterministic anchor seeds to include one unseen growth seed.

    The historical fixed horizon of sixteen exhausted permanently. A finite
    durable history can contain at most ``N`` distinct seeds, so generating
    ``N + 1`` anchor candidates guarantees at least one fresh seed without an
    arbitrary new ceiling.
    """
    observed: set[int] = set()
    for seeds in seeds_by_agent.values():
        observed.update(int(seed) for seed in seeds)
    return max(1, int(minimum), len(observed) + 1)


def fold_seed_bound(
    *,
    champion_agent_id: UUID,
    anchor_version: int,
    seeds_by_agent: Mapping[UUID, Iterable[int]],
    max_seeds: int = TOP5_MAX_CONFIRMATION_SEEDS,
) -> tuple[int, ...]:
    """The seeds the fold may consider: the live anchor plus everything recorded.

    The fold has to be bounded by the same set issuance leases against, or the
    lane pays for runs the fold then ignores. ditto-platform#547 widened issuance
    to ``anchor | universe`` -- catch-up sends every cohort member to any seed a
    peer holds, including seeds from an earlier reign -- but left the fold on the
    champion's anchor alone. That is the worst of both: validators converge the
    cohort onto a seed and the fold refuses to count it.

    It also silently narrowed what already worked. The anchor is keyed on the
    champion's agent id, so it describes ONE reign of at most ``max_seeds``
    seeds. Scoping the fold to it discards every cross-reign seed the cohort
    genuinely shares: on the 2026-07-28 board the fold fell from ten shared seeds
    to four while the shallowest member of the raw top five held sixteen. Those
    six were not noise -- they were datasets every member had been scored on.

    Cross-reign seeds are valid paired evidence. A seed IS a dataset, and two
    agents holding it were measured on the same one whatever anchored it. The
    dethrone test has always relied on exactly this, pairing over shared seeds
    with no anchor filter at all (``koth._paired_statistic``). The anchor's job
    is to introduce NEW unpredictable seeds, which it still does -- it is the
    growth frontier, not the window.

    The anchor stays in the bound even though the universe is derived from
    recorded rows: a freshly crowned champion's seeds are in nobody's history
    yet, and the first one to land has to be foldable immediately.
    """
    universe: set[int] = set()
    for seeds in seeds_by_agent.values():
        universe.update(seeds)
    anchor = champion_anchored_seeds(
        champion_agent_id, version=anchor_version, max_seeds=max_seeds
    )
    return (*anchor, *sorted(universe.difference(anchor)))
