"""Unit tests for :mod:`ditto.db.queries.efficiency` against SQLite-in-memory.

The two tables are append-only by contract: snapshots are unique per
``(bench_version, run_size, epoch)`` and bonus rows insert-once per
``(agent_id, bench_version)`` — a duplicate insert must fail loudly rather
than silently mutate frozen history.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy.exc import IntegrityError as SAIntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.agent_status import AgentStatus
from ditto.api_server.efficiency import CohortMember, CohortReference
from ditto.db.models import Agent
from ditto.db.queries.efficiency import (
    get_bonus_rows,
    get_snapshot,
    get_snapshot_by_id,
    insert_bonus,
    insert_snapshot,
    latest_snapshot,
)

_MINER = "5DhaT8U7LVwnnJNUU8VL1XEipicatoaDVVq7cHo227gogVZm"


def _reference(
    *,
    epoch_index: int = 1000,
    active: bool = True,
    members: tuple[CohortMember, ...] = (),
) -> CohortReference:
    return CohortReference(
        bench_version=7,
        run_size="full",
        epoch_index=epoch_index,
        active=active,
        cohort_limit=25,
        n_min=8,
        bonus_cap=0.05,
        quality_floor=0.5,
        memory_floor=0.4,
        reference_p25_tokens=100.0 if active else None,
        reference_median_tokens=200.0 if active else None,
        members=members,
    )


def _member(n: int, *, collapsed: tuple[UUID, ...] = ()) -> CohortMember:
    return CohortMember(
        agent_id=UUID(int=n),
        miner_hotkey=_MINER,
        lineage_key=f"sha:{n:064x}",
        composite=0.8,
        memory_mean=0.7,
        token_total=100.0 + n,
        collapsed_agent_ids=collapsed,
    )


async def _seed_agent(session: AsyncSession) -> Agent:
    agent = Agent(
        agent_id=uuid4(),
        miner_hotkey=_MINER,
        name="alpha",
        sha256="ab" * 32,
        status=AgentStatus.SCORED,
        created_at=datetime.now(UTC),
    )
    async with session.begin():
        session.add(agent)
    return agent


class TestSnapshots:
    async def test_roundtrip_including_members_json(self, session: AsyncSession):
        reference = _reference(
            members=(_member(1, collapsed=(UUID(int=9),)), _member(2))
        )
        async with session.begin():
            inserted = await insert_snapshot(session, reference)

        read = await get_snapshot(
            session, bench_version=7, run_size="full", epoch_index=1000
        )
        assert read is not None
        assert read.snapshot_id == inserted.snapshot_id
        assert read.active is True
        assert read.reference_p25_tokens == 100.0
        assert read.reference_median_tokens == 200.0
        assert read.members is not None and len(read.members) == 2
        assert read.members[0]["agent_id"] == str(UUID(int=1))
        assert read.members[0]["collapsed_agent_ids"] == [str(UUID(int=9))]
        assert await get_snapshot_by_id(session, inserted.snapshot_id) is not None

    async def test_epoch_key_is_unique(self, session: AsyncSession):
        async with session.begin():
            await insert_snapshot(session, _reference())
        with pytest.raises(SAIntegrityError):
            async with session.begin():
                await insert_snapshot(session, _reference())

    async def test_new_epoch_never_mutates_the_old_snapshot(
        self, session: AsyncSession
    ):
        async with session.begin():
            first = await insert_snapshot(
                session, _reference(epoch_index=1000, members=(_member(1),))
            )
        original = (
            first.snapshot_id,
            first.reference_p25_tokens,
            first.reference_median_tokens,
            list(first.members or []),
        )

        async with session.begin():
            await insert_snapshot(
                session,
                CohortReference(
                    bench_version=7,
                    run_size="full",
                    epoch_index=1001,
                    active=True,
                    cohort_limit=25,
                    n_min=8,
                    bonus_cap=0.05,
                    quality_floor=0.6,
                    memory_floor=0.5,
                    reference_p25_tokens=50.0,
                    reference_median_tokens=75.0,
                    members=(_member(2), _member(3)),
                ),
            )

        reread = await get_snapshot(
            session, bench_version=7, run_size="full", epoch_index=1000
        )
        assert reread is not None
        assert (
            reread.snapshot_id,
            reread.reference_p25_tokens,
            reread.reference_median_tokens,
            list(reread.members or []),
        ) == original

    async def test_latest_snapshot_honors_bounds_and_active_filter(
        self, session: AsyncSession
    ):
        async with session.begin():
            await insert_snapshot(session, _reference(epoch_index=1000, active=False))
            newest = await insert_snapshot(session, _reference(epoch_index=1001))

        found = await latest_snapshot(
            session,
            bench_version=7,
            run_size="full",
            max_epoch_index=1001,
        )
        assert found is not None and found.snapshot_id == newest.snapshot_id

        bounded = await latest_snapshot(
            session,
            bench_version=7,
            run_size="full",
            max_epoch_index=1000,
        )
        assert bounded is not None and bounded.epoch_index == 1000

        active_only = await latest_snapshot(
            session,
            bench_version=7,
            run_size="full",
            max_epoch_index=1000,
            active_only=True,
        )
        assert active_only is None

    async def test_missing_snapshot_reads_return_none(self, session: AsyncSession):
        assert (
            await get_snapshot(session, bench_version=7, run_size="full", epoch_index=1)
            is None
        )
        assert (
            await latest_snapshot(
                session, bench_version=7, run_size="full", max_epoch_index=10
            )
            is None
        )


class TestBonuses:
    async def test_insert_once_and_read_back(self, session: AsyncSession):
        agent = await _seed_agent(session)
        async with session.begin():
            snapshot = await insert_snapshot(session, _reference())
            await insert_bonus(
                session,
                agent_id=agent.agent_id,
                bench_version=7,
                snapshot_id=snapshot.snapshot_id,
                token_total=90.0,
                bonus=0.05,
            )

        rows = await get_bonus_rows(
            session, [agent.agent_id], bench_versions={agent.agent_id: 7}
        )
        assert rows[agent.agent_id].bonus == 0.05
        assert rows[agent.agent_id].snapshot_id == snapshot.snapshot_id
        assert rows[agent.agent_id].token_total == 90.0

    async def test_duplicate_assignment_is_rejected(self, session: AsyncSession):
        agent = await _seed_agent(session)
        # Primitives captured up front: the failed transaction's rollback
        # expires every ORM object, so attribute access after it would lazy-load.
        agent_id = agent.agent_id
        async with session.begin():
            snapshot = await insert_snapshot(session, _reference())
            snapshot_id = snapshot.snapshot_id
            await insert_bonus(
                session,
                agent_id=agent_id,
                bench_version=7,
                snapshot_id=snapshot_id,
                token_total=90.0,
                bonus=0.05,
            )
        with pytest.raises(SAIntegrityError):
            async with session.begin():
                await insert_bonus(
                    session,
                    agent_id=agent_id,
                    bench_version=7,
                    snapshot_id=snapshot_id,
                    token_total=90.0,
                    bonus=0.01,
                )
        rows = await get_bonus_rows(session, [agent_id], bench_versions={agent_id: 7})
        assert rows[agent_id].bonus == 0.05

    async def test_version_scoped_read(self, session: AsyncSession):
        agent = await _seed_agent(session)
        async with session.begin():
            snapshot = await insert_snapshot(session, _reference())
            await insert_bonus(
                session,
                agent_id=agent.agent_id,
                bench_version=7,
                snapshot_id=snapshot.snapshot_id,
                token_total=90.0,
                bonus=0.02,
            )
        assert (
            await get_bonus_rows(
                session, [agent.agent_id], bench_versions={agent.agent_id: 8}
            )
            == {}
        )
        assert await get_bonus_rows(session, [], bench_versions={}) == {}
