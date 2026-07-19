"""Contract tests for the public projection of ditto-subnet's KOTH fold."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from ditto.api_server.koth import KothEntry, confirmation_pair, project_koth

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _entry(
    marker: int,
    composite: float,
    *,
    minutes: int,
    stderr: float | None = None,
    confirmations: tuple[float, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
) -> KothEntry:
    return KothEntry(
        miner_hotkey="5" + str(marker) * 47,
        agent_id=UUID(int=marker),
        composite=composite,
        first_seen=_T0 + timedelta(minutes=minutes),
        raw_rank=marker,
        composite_stderr=stderr,
        confirmation_composites=confirmations,
        confirmation_seeds=seeds,
    )


def test_older_incumbent_survives_a_sub_margin_raw_leader() -> None:
    incumbent = _entry(2, 0.800, minutes=0)
    raw_leader = _entry(1, 0.815, minutes=1)

    projection = project_koth([raw_leader, incumbent])

    assert projection is not None
    assert projection.champion == incumbent
    assert projection.raw_leader == raw_leader
    assert projection.raw_leader_decision is not None
    assert projection.raw_leader_decision.challenger_lead == pytest.approx(0.015)
    assert projection.raw_leader_decision.required_lead == pytest.approx(0.016)
    assert projection.raw_leader_decision.method == "flat"
    assert projection.raw_leader_decision.dethrones is False


def test_statistical_band_matches_validator_unpaired_rule() -> None:
    incumbent = _entry(2, 0.80, minutes=0, stderr=0.03)
    raw_leader = _entry(1, 0.85, minutes=1, stderr=0.03)

    projection = project_koth([raw_leader, incumbent])

    assert projection is not None
    decision = projection.raw_leader_decision
    assert decision is not None
    assert projection.champion == incumbent
    assert decision.margin_lead == pytest.approx(0.016)
    assert decision.statistical_lead == pytest.approx(1.64 * (0.03**2 + 0.03**2) ** 0.5)
    assert decision.required_lead == decision.statistical_lead
    assert decision.method == "unpaired"


def test_clear_challenger_dethrones_and_tail_uses_raw_composite_order() -> None:
    incumbent = _entry(3, 0.80, minutes=0, stderr=0.01)
    challenger = _entry(1, 0.90, minutes=2, stderr=0.01)
    runner_up = _entry(2, 0.85, minutes=1)

    projection = project_koth([runner_up, challenger, incumbent])

    assert projection is not None
    assert projection.champion == challenger
    assert projection.raw_leader_decision is None
    assert projection.tail == (runner_up, incumbent)


def test_confirmation_median_and_paired_seed_band_match_validator_fold() -> None:
    incumbent = _entry(
        2,
        0.80,
        minutes=0,
        confirmations=(0.80, 0.82, 0.78),
        seeds=(10, 20, 30),
    )
    lucky_raw_leader = _entry(
        1,
        0.90,
        minutes=1,
        confirmations=(0.81, 0.83, 0.79),
        seeds=(10, 20, 30),
    )

    projection = project_koth([lucky_raw_leader, incumbent])

    assert projection is not None
    assert projection.champion == incumbent
    decision = projection.raw_leader_decision
    assert decision is not None
    assert decision.method == "paired"
    assert decision.challenger_lead == pytest.approx(0.01)
    assert decision.margin_lead == pytest.approx(0.016)
    assert decision.dethrones is False


def test_empty_or_non_positive_pool_has_no_projection() -> None:
    assert project_koth([]) is None
    assert project_koth([_entry(1, 0.0, minutes=0)]) is None


def test_confirmation_pair_requires_flat_win_inside_unpaired_band() -> None:
    incumbent = _entry(
        2,
        0.88,
        minutes=0,
        stderr=0.03,
        confirmations=(0.87, 0.88, 0.89),
        seeds=(7, 8, 9),
    )
    challenger = _entry(1, 0.93, minutes=1, stderr=0.03)

    projection = project_koth([incumbent, challenger])

    assert projection is not None
    assert projection.champion == incumbent
    assert confirmation_pair(projection) == (incumbent, challenger)


def test_confirmation_pair_stops_after_shared_seed_evidence() -> None:
    incumbent = _entry(
        2,
        0.88,
        minutes=0,
        stderr=0.03,
        confirmations=(0.87, 0.88, 0.89),
        seeds=(7, 8, 9),
    )
    challenger = _entry(
        1,
        0.93,
        minutes=1,
        stderr=0.03,
        confirmations=(0.92, 0.93, 0.94),
        seeds=(7, 8, 9),
    )

    assert confirmation_pair(project_koth([incumbent, challenger])) is None
