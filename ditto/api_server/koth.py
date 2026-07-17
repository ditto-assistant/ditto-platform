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
KOTH_MARGIN = 0.02
KOTH_TAIL_SIZE = 4
KOTH_CHAMPION_SHARE = 0.9
KOTH_DETHRONE_Z = 1.64


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
        margin_lead = champion_reference * KOTH_MARGIN
        paired_statistical_lead = KOTH_DETHRONE_Z * standard_error
        required = max(margin_lead, paired_statistical_lead)
        return DethroneDecision(
            challenger_lead=lead,
            required_lead=required,
            margin_lead=margin_lead,
            statistical_lead=paired_statistical_lead,
            method="paired",
            dethrones=lead > required,
        )

    challenger_composite = _effective_composite(challenger)
    champion_composite = _effective_composite(champion)
    lead = challenger_composite - champion_composite
    margin_lead = champion_composite * KOTH_MARGIN
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
        dethrones=lead > required,
    )
