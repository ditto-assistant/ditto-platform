"""Read-only projection of the validator's frozen KOTH emissions fold.

The canonical consensus implementation lives in ``ditto-subnet`` at
``ditto/validator/weights.py``.  The platform uses this small, pure projection
only to explain that fold on the public leaderboard; validators still compute
and submit their own weights.  Keep the constants and comparison semantics
byte-for-byte aligned with the subnet implementation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

# Frozen consensus constants from ditto-subnet/ditto/validator/config.py.
KOTH_MARGIN = 0.007
KOTH_TAIL_SIZE = 4
KOTH_RANK_SHARES = (0.65, 0.14, 0.10, 0.07, 0.04)
KOTH_CHAMPION_SHARE = KOTH_RANK_SHARES[0]
KOTH_DETHRONE_Z = 1.64

# One tempo = 360 blocks (~72 min at 12 s/block); mirrors the subnet worker's
# rescore cadence.  The top-5 continual shared-seed rescore lane opens rounds on
# a reign-backoff over the champion's crown (see ``top5_round_is_due``).
BLOCKS_PER_TEMPO = 360

# Ceiling on the champion-anchored confirmation-seed depth, mirroring the subnet's
# ``TOP5_MAX_CONFIRMATION_SEEDS`` (ditto/validator/config.py).  The platform
# derives ``crn_seed([champion], version, k)`` for ``k in range(this)`` to bound
# the anti-grind check: a submitted confirmation seed only counts as
# champion-anchored top-5 evidence if it lands in this set.  Must be >= the
# subnet's cap so a legitimately-deep champion's newest seed is still recognised.
TOP5_MAX_CONFIRMATION_SEEDS = 16


@dataclass(frozen=True)
class KothEntry:
    """The public-safe subset of one active ledger row used by the fold."""

    miner_hotkey: str
    agent_id: UUID
    composite: float
    first_seen: datetime
    raw_rank: int
    composite_stderr: float | None = None
    confirmation_composites: tuple[float, ...] | None = None
    confirmation_seeds: tuple[int, ...] | None = None


@dataclass(frozen=True)
class DethroneDecision:
    """Why the raw score leader did or did not clear the incumbent."""

    challenger_lead: float
    required_lead: float
    margin_lead: float
    statistical_lead: float | None
    method: Literal["flat", "unpaired", "paired"]
    dethrones: bool


@dataclass(frozen=True)
class KothProjection:
    champion: KothEntry
    tail: tuple[KothEntry, ...]
    raw_leader: KothEntry
    raw_leader_decision: DethroneDecision | None


def emission_set(projection: KothProjection | None) -> tuple[KothEntry, ...]:
    """Return the emission set (champion + up to 4 distinct-miner tail = top 5).

    This is the membership of the continual top-5 shared-seed rescore lane.  It
    reuses the frozen KOTH fold (:func:`project_koth`): the champion via the
    paired dethrone chain, the tail via ``project_koth``'s
    ``KOTH_TAIL_SIZE``-capped, distinct-miner ``-composite`` ordering.  The
    champion is always first (the anchor), followed by the tail in fold order.
    A newcomer that enters the top 5 automatically joins the set; one that drops
    out stops -- membership follows the set, no manual list.

    The result contains no duplicate ``agent_id`` (``project_koth`` already
    excludes the champion's miner from the tail), so it is at most five entries.
    """
    if projection is None:
        return ()
    seen = {projection.champion.agent_id}
    members = [projection.champion]
    for entry in projection.tail:
        if entry.agent_id in seen:
            continue
        seen.add(entry.agent_id)
        members.append(entry)
    return tuple(members)


def tempo_index(block_number: int) -> int:
    """The tempo ordinal a chain block falls in (``block // BLOCKS_PER_TEMPO``)."""
    return block_number // BLOCKS_PER_TEMPO


def top5_round_is_due(
    current_block: int,
    crown_block: int,
    *,
    base: int,
    doubling_k: int,
    cap: int,
) -> bool:
    """Whether a top-5 shared-seed rescore round is due at ``current_block``.

    The interval between rounds is an **exponential backoff over the champion's
    reign** (``docs/top5-rescore-lane.md`` §4): dense while a fresh or contested
    king must prove its crown on many seeds, sparse once the reign settles ---
    saving tokens on a stable leader. Measured in tempos since the champion's
    ``crown_block`` (a deterministic ledger fact that changes on any king
    change, so churn re-enters the dense regime and stagnation tapers)::

        interval(reign_tempos) = min(base * 2**floor(reign_tempos / K), cap)

    A round is due exactly when the current reign-tempo lands on a scheduled
    point of that growing schedule (offset 0 = the crown tempo, then repeatedly
    advancing by the interval at each reached point). ``base`` holds for the
    first ``doubling_k`` reign-tempos, front-loading the densest rounds across
    the ~24 h king-source-reveal window (#277/#278) before doubling begins. The
    interval is capped, so the rate never reaches zero -- a champion flatlining
    at ``cap`` is itself the "field has gone stagnant" signal.

    Pure and deterministic: a function only of the two block numbers and the
    consensus constants, so every validator hitting the platform at the same
    height gets the same decision. ``base <= 0`` disables the lane.
    """
    if base <= 0:
        return False
    step_cap = max(base, cap)
    span = max(1, doubling_k)
    reign_tempo = max(0, current_block - crown_block) // BLOCKS_PER_TEMPO
    scheduled = 0
    while scheduled < reign_tempo:
        interval = min(base * (2 ** (scheduled // span)), step_cap)
        scheduled += interval
    return scheduled == reign_tempo


def project_koth(entries: Sequence[KothEntry]) -> KothProjection | None:
    """Return the champion and participation tail for an eligible score pool."""
    scored = [entry for entry in entries if entry.composite > 0.0]
    if not scored:
        return None

    ordered = sorted(scored, key=lambda entry: (entry.first_seen, entry.agent_id))
    champion = ordered[0]
    for challenger in ordered[1:]:
        if _dethrone_decision(challenger, champion).dethrones:
            champion = challenger

    tail = tuple(
        sorted(
            (entry for entry in scored if entry.miner_hotkey != champion.miner_hotkey),
            key=lambda entry: (
                -entry.composite,
                entry.first_seen,
                entry.agent_id,
            ),
        )[:KOTH_TAIL_SIZE]
    )
    raw_leader = sorted(
        scored,
        key=lambda entry: (-entry.composite, entry.first_seen, entry.agent_id),
    )[0]
    decision = (
        None
        if raw_leader.agent_id == champion.agent_id
        else _dethrone_decision(raw_leader, champion)
    )
    return KothProjection(
        champion=champion,
        tail=tail,
        raw_leader=raw_leader,
        raw_leader_decision=decision,
    )


def _confirmations(entry: KothEntry) -> tuple[float, ...] | None:
    values = entry.confirmation_composites
    if values is None or len(values) < 2:
        return None
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        return None
    return values


def _effective_composite(entry: KothEntry) -> float:
    values = _confirmations(entry)
    if values is None:
        return entry.composite
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _stderr(entry: KothEntry) -> float | None:
    value = entry.composite_stderr
    if value is not None and math.isfinite(value) and value >= 0.0:
        return value
    return None


def _seed_composites(entry: KothEntry) -> dict[int, float] | None:
    composites = _confirmations(entry)
    seeds = entry.confirmation_seeds
    if composites is None or seeds is None or len(seeds) != len(composites):
        return None
    out: dict[int, float] = {}
    for seed, composite in zip(seeds, composites, strict=True):
        if seed < 0 or seed in out:
            return None
        out[seed] = composite
    return out


def _paired_statistic(
    challenger: KothEntry, champion: KothEntry
) -> tuple[float, float, float] | None:
    challenger_by_seed = _seed_composites(challenger)
    champion_by_seed = _seed_composites(champion)
    if challenger_by_seed is None or champion_by_seed is None:
        return None
    shared = sorted(challenger_by_seed.keys() & champion_by_seed.keys())
    if len(shared) < 2:
        return None
    differences = [challenger_by_seed[seed] - champion_by_seed[seed] for seed in shared]
    champion_reference = sum(champion_by_seed[seed] for seed in shared) / len(shared)
    mean_difference = sum(differences) / len(differences)
    variance = sum(
        (difference - mean_difference) ** 2 for difference in differences
    ) / (len(differences) - 1)
    return mean_difference, champion_reference, math.sqrt(variance / len(differences))


def _dethrone_decision(challenger: KothEntry, champion: KothEntry) -> DethroneDecision:
    paired = _paired_statistic(challenger, champion)
    if paired is not None:
        lead, champion_reference, standard_error = paired
        margin_lead = KOTH_MARGIN
        paired_statistical_lead = KOTH_DETHRONE_Z * standard_error
        required = max(margin_lead, paired_statistical_lead)
        return DethroneDecision(
            challenger_lead=lead,
            required_lead=required,
            margin_lead=margin_lead,
            statistical_lead=paired_statistical_lead,
            method="paired",
            dethrones=(champion_reference + lead > champion_reference + required),
        )

    challenger_composite = _effective_composite(challenger)
    champion_composite = _effective_composite(champion)
    lead = challenger_composite - champion_composite
    margin_lead = KOTH_MARGIN
    challenger_stderr = _stderr(challenger)
    champion_stderr = _stderr(champion)
    statistical_lead: float | None = None
    method: Literal["flat", "unpaired", "paired"] = "flat"
    if challenger_stderr is not None and champion_stderr is not None:
        statistical_lead = KOTH_DETHRONE_Z * math.sqrt(
            challenger_stderr**2 + champion_stderr**2
        )
        method = "unpaired"
    required = max(
        margin_lead,
        statistical_lead if statistical_lead is not None else margin_lead,
    )
    return DethroneDecision(
        challenger_lead=lead,
        required_lead=required,
        margin_lead=margin_lead,
        statistical_lead=statistical_lead,
        method=method,
        # Mirror the validator's threshold comparison. Subtracting first can
        # round an exact decimal boundary infinitesimally upward.
        dethrones=challenger_composite > champion_composite + required,
    )
