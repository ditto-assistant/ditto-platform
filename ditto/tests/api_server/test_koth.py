"""Contract tests for the public projection of ditto-subnet's KOTH fold."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from ditto.api_server.koth import (
    BLOCKS_PER_TEMPO,
    KothEntry,
    emission_set,
    project_koth,
    tempo_index,
    top5_round_is_due,
)

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
    raw_leader = _entry(1, 0.804, minutes=1)

    projection = project_koth([raw_leader, incumbent])

    assert projection is not None
    assert projection.champion == incumbent
    assert projection.raw_leader == raw_leader
    assert projection.raw_leader_decision is not None
    assert projection.raw_leader_decision.challenger_lead == pytest.approx(0.004)
    assert projection.raw_leader_decision.required_lead == pytest.approx(0.007)
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
    assert decision.margin_lead == pytest.approx(0.007)
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


def test_fixed_margin_does_not_grow_into_a_ceiling_lock() -> None:
    incumbent = _entry(2, 0.930, minutes=0)
    challenger = _entry(1, 0.938, minutes=1)

    projection = project_koth([challenger, incumbent])

    assert projection is not None
    assert projection.champion == challenger

    exact_boundary = project_koth([_entry(3, 0.937, minutes=1), incumbent])
    assert exact_boundary is not None
    assert exact_boundary.champion == incumbent


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
        confirmations=(0.804, 0.824, 0.784),
        seeds=(10, 20, 30),
    )

    projection = project_koth([lucky_raw_leader, incumbent])

    assert projection is not None
    assert projection.champion == incumbent
    decision = projection.raw_leader_decision
    assert decision is not None
    assert decision.method == "paired"
    assert decision.challenger_lead == pytest.approx(0.004)
    assert decision.margin_lead == pytest.approx(0.007)
    assert decision.dethrones is False


def test_empty_or_non_positive_pool_has_no_projection() -> None:
    assert project_koth([]) is None
    assert project_koth([_entry(1, 0.0, minutes=0)]) is None


def test_emission_set_is_champion_plus_four_distinct_miner_tail() -> None:
    # Oldest + highest composite is the champion; five others trail it.
    champion = _entry(1, 0.90, minutes=0)
    tail = [
        _entry(2, 0.88, minutes=1),
        _entry(3, 0.86, minutes=2),
        _entry(4, 0.84, minutes=3),
        _entry(5, 0.82, minutes=4),
    ]
    sixth = _entry(6, 0.80, minutes=5)

    members = emission_set(project_koth([champion, *tail, sixth]))

    assert len(members) == 5
    assert members[0].agent_id == champion.agent_id
    assert {m.agent_id for m in members} == {UUID(int=i) for i in range(1, 6)}
    # The set follows the top five: the sixth-place agent is not in the lane.
    assert sixth.agent_id not in {m.agent_id for m in members}


def test_emission_set_admits_a_new_top_five_entrant() -> None:
    champion = _entry(1, 0.90, minutes=0)
    incumbents = [
        _entry(2, 0.88, minutes=1),
        _entry(3, 0.86, minutes=2),
        _entry(4, 0.84, minutes=3),
        _entry(5, 0.82, minutes=4),
    ]
    before = emission_set(project_koth([champion, *incumbents]))
    assert {m.agent_id for m in before} == {UUID(int=i) for i in range(1, 6)}

    # A fresh entrant scoring above the weakest tail member joins automatically
    # and evicts agent 5 (0.82); membership follows the set with no manual list.
    newcomer = _entry(6, 0.85, minutes=5)
    after = emission_set(project_koth([champion, *incumbents, newcomer]))

    assert newcomer.agent_id in {m.agent_id for m in after}
    assert UUID(int=5) not in {m.agent_id for m in after}
    assert len(after) == 5


def test_emission_set_empty_pool_is_empty() -> None:
    assert emission_set(None) == ()
    assert emission_set(project_koth([])) == ()


def _due_reign_tempos(
    max_tempo: int, *, base: int, doubling_k: int, cap: int, crown_block: int = 0
) -> list[int]:
    """Reign-tempos (from the crown) at which a round is due, for assertions."""
    return [
        t
        for t in range(max_tempo + 1)
        if top5_round_is_due(
            crown_block + t * BLOCKS_PER_TEMPO,
            crown_block,
            base=base,
            doubling_k=doubling_k,
            cap=cap,
        )
    ]


def test_backoff_is_deterministic_across_validators() -> None:
    # Two independent "validators" reading the same chain height + crown block
    # get byte-identical due decisions -- pure function, no clock, no RNG.
    for block in (0, 720, 5000, 100_000):
        a = top5_round_is_due(block, 0, base=2, doubling_k=20, cap=8)
        b = top5_round_is_due(block, 0, base=2, doubling_k=20, cap=8)
        assert a == b


def test_backoff_is_dense_early_and_sparse_late() -> None:
    # base=2, K=20, cap=8: interval holds at 2 for the first 20 reign-tempos
    # (front-loading the ~24h reveal window), then doubles to 4, then caps at 8.
    due = _due_reign_tempos(80, base=2, doubling_k=20, cap=8)
    # Dense early: every 2 tempos through the first ~20.
    assert due[:11] == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    # After 20 reign-tempos the interval doubles to 4.
    assert 24 in due and 22 not in due
    # Gaps only grow, and never exceed the cap of 8 tempos (never zero-rate).
    gaps = [b - a for a, b in zip(due, due[1:], strict=False)]
    assert gaps == sorted(gaps)
    assert max(gaps) == 8
    assert min(gaps) == 2


def test_backoff_caps_the_interval() -> None:
    # Far into a long reign the interval flatlines at the cap; rounds keep firing.
    due = _due_reign_tempos(400, base=2, doubling_k=20, cap=8)
    tail_gaps = [b - a for a, b in zip(due, due[1:], strict=False)][-10:]
    assert set(tail_gaps) == {8}


def test_backoff_resets_on_king_change() -> None:
    # A new champion (new crown block) re-enters the dense regime: block that was
    # sparse-late under the old crown is due-at-offset-0 under the new one.
    old_crown = 0
    new_crown = 900 * BLOCKS_PER_TEMPO  # a fresh coronation far later
    # At exactly the new crown block, reign-tempo 0 -> due.
    assert top5_round_is_due(new_crown, new_crown, base=2, doubling_k=20, cap=8)
    # The same height under the old, long crown is on the sparse cap schedule and
    # is (generally) not a scheduled point -- the reset changes the answer.
    old_due = top5_round_is_due(new_crown, old_crown, base=2, doubling_k=20, cap=8)
    new_due = top5_round_is_due(new_crown, new_crown, base=2, doubling_k=20, cap=8)
    assert new_due is True
    assert old_due is False


def test_backoff_disabled_when_base_non_positive() -> None:
    assert top5_round_is_due(0, 0, base=0, doubling_k=20, cap=8) is False
    assert top5_round_is_due(1440, 0, base=-1, doubling_k=20, cap=8) is False


def test_tempo_index_counts_360_block_windows() -> None:
    assert tempo_index(0) == 0
    assert tempo_index(359) == 0
    assert tempo_index(360) == 1
    assert tempo_index(1440) == 4
