"""Unit tests for the append-only top-5 confirmation-score ledger queries."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.db.models import Agent, ConfirmationScore
from ditto.db.queries.confirmation_scores import (
    ConfirmationSeedScore,
    append_confirmation_scores,
    completed_confirmation_wave_seeds,
    confirmation_composites_by_seed,
    confirmation_depths,
    confirmation_history_by_agent,
    fold_eligible_seeds_by_agent,
)

_NOW = datetime(2026, 7, 21, 12, 0, 0, tzinfo=UTC)


async def _seed_agent(session: AsyncSession, name: str = "a") -> UUID:
    aid = uuid4()
    async with session.begin():
        session.add(
            Agent(
                agent_id=aid,
                miner_hotkey="5Miner",
                name=name,
                sha256="ab" * 32,
                status=AgentStatus.SCORED,
                screening_policy_version=SCREENING_POLICY_VERSION,
                created_at=_NOW,
            )
        )
    return aid


def _row(
    agent_id: UUID, validator: str, seed: int, composite: float
) -> ConfirmationSeedScore:
    return ConfirmationSeedScore(
        agent_id=agent_id,
        validator_hotkey=validator,
        seed=seed,
        composite=composite,
        run_id=f"run-{validator}-{seed}",
        signature="ab" * 64,
    )


class TestAppendConfirmationScores:
    async def test_append_is_insert_idempotent_on_the_unique_key(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_agent(session)
        async with session.begin():
            n = await append_confirmation_scores(
                session,
                rows=[_row(aid, "5V1", 100, 0.80), _row(aid, "5V1", 200, 0.82)],
                bench_version=2,
                created_at=_NOW,
            )
        assert n == 2
        # Re-submitting the whole union (incumbent resends every round) is a no-op
        # on the already-present seeds; only the genuinely new seed is inserted.
        async with session.begin():
            n2 = await append_confirmation_scores(
                session,
                rows=[
                    _row(aid, "5V1", 100, 0.99),  # same key -> ignored (immutable)
                    _row(aid, "5V1", 200, 0.99),  # same key -> ignored
                    _row(aid, "5V1", 300, 0.85),  # new seed -> inserted
                ],
                bench_version=2,
                created_at=_NOW,
            )
        assert n2 == 1
        async with session.begin():
            total = await session.scalar(
                select(func.count()).select_from(ConfirmationScore)
            )
            first = await session.get(ConfirmationScore, (aid, 2, "5V1", 100))
        assert total == 3
        # The first-written composite wins; a later resend never overwrites it.
        assert first is not None and first.composite == 0.80

    async def test_distinct_validators_and_versions_coexist(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_agent(session)
        async with session.begin():
            await append_confirmation_scores(
                session,
                rows=[_row(aid, "5V1", 100, 0.80), _row(aid, "5V2", 100, 0.82)],
                bench_version=2,
                created_at=_NOW,
            )
            await append_confirmation_scores(
                session,
                rows=[_row(aid, "5V1", 100, 0.90)],
                bench_version=3,
                created_at=_NOW,
            )
        async with session.begin():
            total = await session.scalar(
                select(func.count()).select_from(ConfirmationScore)
            )
        assert total == 3


class TestConfirmationAggregates:
    def test_only_complete_cohort_waves_are_fold_eligible(self) -> None:
        first, second, third = uuid4(), uuid4(), uuid4()
        assert completed_confirmation_wave_seeds(
            member_ids=[first, second, third],
            seeds_by_agent={
                first: [100, 200, 300],
                second: [100, 200],
                third: [100],
            },
        ) == frozenset({100})

    def test_a_zero_depth_entrant_empties_the_strict_intersection(self) -> None:
        """The 03:56Z incident, as a unit test.

        ``dittoLife-v1`` finalized, displaced ``banblackycat`` from the top five,
        and brought no confirmation rows with it. Under ``strict`` that single
        arrival erases eight waves of accumulated evidence for every other agent
        -- and not only on the display: ``official_composite`` reverts to the
        three-score quorum median, so validator weights revert with it.
        """
        deep, mid, entrant = uuid4(), uuid4(), uuid4()
        seeds = {deep: [100, 200, 300], mid: [100, 200, 300], entrant: []}

        strict = fold_eligible_seeds_by_agent(
            member_ids=[deep, mid, entrant], seeds_by_agent=seeds, mode="strict"
        )

        assert strict[deep] == frozenset()
        assert strict[mid] == frozenset()

    def test_participants_keeps_the_evidence_the_entrant_never_ran(self) -> None:
        """The recommended fix, on the same input.

        An agent at depth zero has never been leased any of these seeds, so it is
        not protecting a running lease. Excluding it preserves every completed
        wave, and the two agents that DO get a continual mean still share one
        identical seed set -- comparability is untouched.
        """
        deep, mid, entrant = uuid4(), uuid4(), uuid4()
        seeds = {deep: [100, 200, 300], mid: [100, 200, 300], entrant: []}

        participants = fold_eligible_seeds_by_agent(
            member_ids=[deep, mid, entrant],
            seeds_by_agent=seeds,
            mode="participants",
        )

        assert participants[deep] == frozenset({100, 200, 300})
        assert participants[mid] == frozenset({100, 200, 300})
        # Equal composition among everyone who receives a continual mean.
        assert participants[deep] == participants[mid]
        assert participants[entrant] == frozenset()

    def test_participants_still_waits_on_a_member_that_has_started(self) -> None:
        """Catch-up is preserved: a partial member still narrows the wave.

        This is the half of the strict rule that is genuinely protecting a
        running lease, and ``participants`` keeps it. Only depth ZERO is treated
        as "not in the lane yet".
        """
        deep, catching_up = uuid4(), uuid4()

        participants = fold_eligible_seeds_by_agent(
            member_ids=[deep, catching_up],
            seeds_by_agent={deep: [100, 200, 300], catching_up: [100]},
            mode="participants",
        )

        assert participants[deep] == frozenset({100})
        assert participants[catching_up] == frozenset({100})

    def test_per_agent_gives_every_agent_its_own_depth(self) -> None:
        """Peyton's literal ask, and the comparability it costs.

        Each agent keeps everything it ran. The means are then taken over
        different seed sets, which is exactly the seed-composition confound the
        shared-seed design exists to cancel -- hence the operator gate.
        """
        deep, mid, entrant = uuid4(), uuid4(), uuid4()

        per_agent = fold_eligible_seeds_by_agent(
            member_ids=[deep, mid, entrant],
            seeds_by_agent={deep: [100, 200, 300], mid: [100], entrant: []},
            mode="per_agent",
        )

        assert per_agent[deep] == frozenset({100, 200, 300})
        assert per_agent[mid] == frozenset({100})
        assert per_agent[entrant] == frozenset()

    def test_strict_is_the_default_and_matches_the_legacy_helper(self) -> None:
        """Merging this must not move a single composite."""
        first, second, third = uuid4(), uuid4(), uuid4()
        seeds = {first: [100, 200, 300], second: [100, 200], third: [100]}

        by_agent = fold_eligible_seeds_by_agent(
            member_ids=[first, second, third], seeds_by_agent=seeds
        )
        legacy = completed_confirmation_wave_seeds(
            member_ids=[first, second, third], seeds_by_agent=seeds
        )

        # Every member resolves to the same set the single shared value held,
        # so no composite moves when this ships.
        assert set(by_agent.values()) == {legacy}
        assert legacy == frozenset({100})

    async def test_composites_by_seed_medians_across_validators(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_agent(session)
        async with session.begin():
            await append_confirmation_scores(
                session,
                rows=[
                    _row(aid, "5V1", 100, 0.80),
                    _row(aid, "5V2", 100, 0.84),
                    _row(aid, "5V3", 100, 0.82),
                    _row(aid, "5V1", 200, 0.70),
                ],
                bench_version=2,
                created_at=_NOW,
            )
        by_seed = await confirmation_composites_by_seed(
            session, agent_ids=[aid], bench_version=2
        )
        assert by_seed[aid][100] == 0.82  # median of 0.80/0.82/0.84
        assert by_seed[aid][200] == 0.70

    async def test_depth_counts_distinct_seeds(self, session: AsyncSession) -> None:
        aid = await _seed_agent(session)
        async with session.begin():
            await append_confirmation_scores(
                session,
                rows=[
                    _row(aid, "5V1", 100, 0.80),
                    _row(aid, "5V2", 100, 0.81),  # same seed, another validator
                    _row(aid, "5V1", 200, 0.82),
                    _row(aid, "5V1", 300, 0.83),
                ],
                bench_version=2,
                created_at=_NOW,
            )
        depths = await confirmation_depths(session, agent_ids=[aid], bench_version=2)
        assert depths[aid] == 3  # three distinct seeds

    async def test_history_returns_raw_unaggregated_records(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_agent(session)
        async with session.begin():
            await append_confirmation_scores(
                session,
                rows=[_row(aid, "5V1", 100, 0.80), _row(aid, "5V2", 100, 0.84)],
                bench_version=2,
                created_at=_NOW,
            )
        history = await confirmation_history_by_agent(
            session, agent_ids=[aid], bench_version=2
        )
        rows = history[aid]
        # Raw per-(validator, seed) rows, NOT medianed: two rows for seed 100.
        assert len(rows) == 2
        assert {r.composite for r in rows} == {0.80, 0.84}
        assert all(r.bench_version == 2 for r in rows)

    async def test_absent_agents_map_to_empty(self, session: AsyncSession) -> None:
        assert (
            await confirmation_composites_by_seed(
                session, agent_ids=[], bench_version=2
            )
            == {}
        )
        assert (
            await confirmation_depths(session, agent_ids=[uuid4()], bench_version=2)
            == {}
        )
