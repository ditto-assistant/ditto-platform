"""Unit tests for the pure half of :mod:`ditto.api_server.efficiency`.

The relative token-efficiency bonus math must be deterministic and robust:
lineage dedupe, nearest-rank quartile + median reference, linear
interpolation boundaries, the N_min activation gate, and the strictly-upside
guarantee are all covered here without a database.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from ditto.api_server.efficiency import (
    BONUS_RUN_SIZE,
    MIN_BONUS_BENCH_VERSION,
    CohortReference,
    EfficiencyCandidate,
    audited_token_total,
    bonus_for_submission,
    bonus_fraction,
    build_cohort_snapshot,
    dedupe_lineages,
    effective_composite,
    epoch_index_for,
    floors_from_previous,
    lineage_key,
    nearest_rank_percentile,
    qualifies,
)

_T0 = datetime(2026, 7, 24, 12, 0, 0, tzinfo=UTC)


def _uuid(n: int) -> UUID:
    return UUID(int=n)


def _candidate(
    n: int,
    *,
    composite: float = 0.8,
    memory_mean: float = 0.7,
    token_total: float | None = 100_000.0,
    lineage: str | None = None,
    first_seen: datetime | None = None,
) -> EfficiencyCandidate:
    return EfficiencyCandidate(
        agent_id=_uuid(n),
        miner_hotkey=f"5Miner{n}",
        lineage_key=lineage or f"sha:{n:064x}",
        composite=composite,
        memory_mean=memory_mean,
        token_total=token_total,
        first_seen=first_seen or _T0,
    )


def _usage(total: int, *, status: str = "complete", unavailable: int = 0) -> dict:
    return {
        "token_usage": {
            "status": status,
            "total_tokens": total,
            "usage_unavailable": unavailable,
        }
    }


class TestAuditedTokenTotal:
    def test_median_over_complete_rows(self) -> None:
        blobs = [_usage(100), _usage(300), _usage(200)]
        assert audited_token_total(blobs) == 200.0

    def test_even_count_averages_middle_pair(self) -> None:
        assert audited_token_total([_usage(100), _usage(200)]) == 150.0

    def test_ignores_incomplete_and_malformed_rows(self) -> None:
        blobs = [
            _usage(100),
            _usage(999_999, status="unavailable"),
            _usage(999_999, unavailable=3),
            {"token_usage": {"status": "complete", "total_tokens": "1"}},
            {"token_usage": {"status": "complete", "total_tokens": -5}},
            {"token_usage": {"status": "complete", "total_tokens": True}},
            {"token_usage": "nope"},
            {},
            None,
        ]
        assert audited_token_total(blobs) == 100.0

    def test_none_when_no_complete_row(self) -> None:
        assert audited_token_total([_usage(5, status="unavailable"), None]) is None
        assert audited_token_total([]) is None


class TestLineageKey:
    def test_prefers_normalized_source_hash(self) -> None:
        assert lineage_key("aa" * 32, "bb" * 32) == "nsh:" + "aa" * 32

    def test_falls_back_to_artifact_sha(self) -> None:
        assert lineage_key(None, "bb" * 32) == "sha:" + "bb" * 32
        assert lineage_key("", "bb" * 32) == "sha:" + "bb" * 32

    def test_channels_never_collide(self) -> None:
        assert lineage_key("ab" * 32, "xx" * 32) != lineage_key(None, "ab" * 32)


class TestDedupeLineages:
    def test_collapses_same_lineage_to_best_composite(self) -> None:
        winner = _candidate(1, composite=0.9, lineage="sha:dup")
        loser = _candidate(2, composite=0.7, lineage="sha:dup")
        other = _candidate(3, composite=0.5)

        members = dedupe_lineages([loser, winner, other])

        assert len(members) == 2
        by_key = {member.lineage_key: member for member in members}
        assert by_key["sha:dup"].agent_id == winner.agent_id
        assert by_key["sha:dup"].collapsed_agent_ids == (loser.agent_id,)
        assert by_key[other.lineage_key].collapsed_agent_ids == ()

    def test_tie_breaks_on_first_seen_then_agent_id(self) -> None:
        earlier = _candidate(5, composite=0.8, lineage="sha:dup", first_seen=_T0)
        later = _candidate(
            4,
            composite=0.8,
            lineage="sha:dup",
            first_seen=_T0 + timedelta(minutes=1),
        )
        members = dedupe_lineages([later, earlier])
        assert members[0].agent_id == earlier.agent_id


class TestNearestRankPercentile:
    def test_quartile_over_eight_values(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        # rank = ceil(0.25 * 8) = 2 -> the 2nd smallest.
        assert nearest_rank_percentile(values, 0.25) == 20.0

    def test_single_value(self) -> None:
        assert nearest_rank_percentile([42.0], 0.25) == 42.0

    def test_unsorted_input(self) -> None:
        assert nearest_rank_percentile([30.0, 10.0, 20.0], 0.25) == 10.0


class TestQualityGate:
    def test_requires_audited_usage(self) -> None:
        assert not qualifies(0.9, 0.9, None, quality_floor=0.0, memory_floor=0.0)

    def test_requires_both_floors(self) -> None:
        assert qualifies(0.6, 0.5, 1.0, quality_floor=0.6, memory_floor=0.5)
        assert not qualifies(0.59, 0.5, 1.0, quality_floor=0.6, memory_floor=0.5)
        assert not qualifies(0.6, 0.49, 1.0, quality_floor=0.6, memory_floor=0.5)


class TestBuildCohortSnapshot:
    def _snapshot(self, candidates, **overrides) -> CohortReference:
        from typing import Any

        kwargs: dict[str, Any] = {
            "bench_version": MIN_BONUS_BENCH_VERSION,
            "run_size": BONUS_RUN_SIZE,
            "epoch_index": 1000,
            "cohort_limit": 25,
            "n_min": 4,
            "bonus_cap": 0.05,
            "quality_floor": 0.0,
            "memory_floor": 0.0,
        }
        kwargs.update(overrides)
        return build_cohort_snapshot(candidates, **kwargs)

    def test_reference_math(self) -> None:
        candidates = [
            _candidate(n, token_total=float(total))
            for n, total in enumerate([100, 200, 300, 400], start=1)
        ]
        snapshot = self._snapshot(candidates)
        assert snapshot.active
        assert snapshot.reference_p25_tokens == 100.0  # ceil(0.25*4)=1st
        assert snapshot.reference_median_tokens == 250.0

    def test_quality_gate_excludes_low_quality_and_unaudited(self) -> None:
        good = [_candidate(n, token_total=100.0) for n in range(1, 5)]
        sandbagged = _candidate(10, composite=0.2, token_total=1.0)
        memory_gutted = _candidate(11, memory_mean=0.1, token_total=1.0)
        unaudited = _candidate(12, token_total=None)
        snapshot = self._snapshot(
            [*good, sandbagged, memory_gutted, unaudited],
            quality_floor=0.5,
            memory_floor=0.5,
        )
        member_ids = {member.agent_id for member in snapshot.members}
        assert member_ids == {candidate.agent_id for candidate in good}

    def test_top_n_cap_keeps_best_composites(self) -> None:
        candidates = [
            _candidate(n, composite=0.5 + n * 0.01, token_total=100.0)
            for n in range(1, 11)
        ]
        snapshot = self._snapshot(candidates, cohort_limit=5, n_min=5)
        assert len(snapshot.members) == 5
        assert min(member.composite for member in snapshot.members) >= 0.56

    def test_n_min_gate_inactive_snapshot(self) -> None:
        candidates = [_candidate(n, token_total=100.0) for n in range(1, 4)]
        snapshot = self._snapshot(candidates, n_min=4)
        assert not snapshot.active
        assert snapshot.reference_p25_tokens is None
        assert snapshot.reference_median_tokens is None
        # Membership is still frozen for observability; it awards nothing.
        assert len(snapshot.members) == 3

    def test_dedupe_applies_before_n_min(self) -> None:
        # Four submissions but only three lineages: the gate must see 3.
        candidates = [
            _candidate(1, lineage="sha:dup", token_total=100.0),
            _candidate(2, lineage="sha:dup", token_total=100.0),
            _candidate(3, token_total=100.0),
            _candidate(4, token_total=100.0),
        ]
        snapshot = self._snapshot(candidates, n_min=4)
        assert not snapshot.active
        assert len(snapshot.members) == 3


class TestBonusFraction:
    def test_full_bonus_at_or_below_frontier(self) -> None:
        assert bonus_fraction(99.0, reference_p25=100, reference_median=200, cap=0.05)
        assert (
            bonus_fraction(100.0, reference_p25=100, reference_median=200, cap=0.05)
            == 0.05
        )

    def test_zero_at_or_above_median(self) -> None:
        assert (
            bonus_fraction(200.0, reference_p25=100, reference_median=200, cap=0.05)
            == 0.0
        )
        assert (
            bonus_fraction(1e12, reference_p25=100, reference_median=200, cap=0.05)
            == 0.0
        )

    def test_linear_between(self) -> None:
        mid = bonus_fraction(150.0, reference_p25=100, reference_median=200, cap=0.05)
        assert mid == 0.025
        q3 = bonus_fraction(175.0, reference_p25=100, reference_median=200, cap=0.05)
        assert abs(q3 - 0.0125) < 1e-12

    def test_degenerate_reference_is_a_step(self) -> None:
        assert (
            bonus_fraction(100.0, reference_p25=100, reference_median=100, cap=0.05)
            == 0.05
        )
        assert (
            bonus_fraction(101.0, reference_p25=100, reference_median=100, cap=0.05)
            == 0.0
        )

    def test_never_negative(self) -> None:
        for tokens in (0.0, 100.0, 150.0, 200.0, 1e9):
            assert (
                bonus_fraction(
                    tokens, reference_p25=100, reference_median=200, cap=0.05
                )
                >= 0.0
            )


class TestBonusForSubmission:
    def _reference(self, *, active: bool = True) -> CohortReference:
        return CohortReference(
            bench_version=7,
            run_size=BONUS_RUN_SIZE,
            epoch_index=1000,
            active=active,
            cohort_limit=25,
            n_min=4,
            bonus_cap=0.05,
            quality_floor=0.5,
            memory_floor=0.4,
            reference_p25_tokens=100.0 if active else None,
            reference_median_tokens=200.0 if active else None,
            members=(),
        )

    def test_inactive_snapshot_awards_nothing(self) -> None:
        assert bonus_for_submission(0.9, 0.9, 50.0, self._reference(active=False)) == 0

    def test_unqualified_submission_awards_nothing(self) -> None:
        reference = self._reference()
        assert bonus_for_submission(0.4, 0.9, 50.0, reference) == 0.0  # sandbag
        assert bonus_for_submission(0.9, 0.3, 50.0, reference) == 0.0  # memory gut
        assert bonus_for_submission(0.9, 0.9, None, reference) == 0.0  # unaudited

    def test_qualified_submission_gets_curve_value(self) -> None:
        reference = self._reference()
        assert bonus_for_submission(0.9, 0.9, 50.0, reference) == 0.05
        assert bonus_for_submission(0.9, 0.9, 150.0, reference) == 0.025


class TestEffectiveComposite:
    def test_multiplicative_and_bounded(self) -> None:
        assert effective_composite(0.8, 0.05) == 0.8 * 1.05
        assert effective_composite(0.8, 0.0) == 0.8
        assert effective_composite(1.0, 0.1) <= 1.1


class TestEpochIndex:
    def test_fixed_utc_windows(self) -> None:
        base = datetime(2026, 7, 24, 0, 0, 1, tzinfo=UTC)
        assert epoch_index_for(base, 24) == epoch_index_for(
            base + timedelta(hours=23), 24
        )
        assert (
            epoch_index_for(base + timedelta(hours=24), 24)
            == epoch_index_for(base, 24) + 1
        )

    def test_epoch_hours_config(self) -> None:
        base = datetime(2026, 7, 24, 0, 0, 1, tzinfo=UTC)
        assert (
            epoch_index_for(base + timedelta(hours=6), 6)
            == epoch_index_for(base, 6) + 1
        )


class TestFloorsFromPrevious:
    def test_static_floors_without_previous_cohort(self) -> None:
        assert floors_from_previous(None, quality_floor=0.3, memory_floor=0.2) == (
            0.3,
            0.2,
        )
        assert floors_from_previous([], quality_floor=0.3, memory_floor=0.2) == (
            0.3,
            0.2,
        )

    def test_derives_from_previous_cohort_medians(self) -> None:
        members = [
            {"composite": 0.6, "memory_mean": 0.5},
            {"composite": 0.8, "memory_mean": 0.7},
            {"composite": 0.7, "memory_mean": 0.6},
        ]
        quality, memory = floors_from_previous(
            members, quality_floor=0.0, memory_floor=0.0
        )
        assert quality == 0.7
        assert abs(memory - 0.8 * 0.6) < 1e-12

    def test_static_floors_win_when_higher(self) -> None:
        members = [{"composite": 0.1, "memory_mean": 0.1}]
        quality, memory = floors_from_previous(
            members, quality_floor=0.5, memory_floor=0.4
        )
        assert (quality, memory) == (0.5, 0.4)
