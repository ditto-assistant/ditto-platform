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
    confirmation_composites_by_seed,
    confirmation_depths,
    confirmation_history_by_agent,
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
