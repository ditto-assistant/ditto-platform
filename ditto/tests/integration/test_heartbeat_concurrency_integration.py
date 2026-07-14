"""Real-Postgres concurrency coverage for signed validator heartbeats."""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.ticket_status import TicketStatus
from ditto.db import create_db_engine, create_session_maker
from ditto.db.models import Agent, ValidatorHeartbeat, ValidatorTicket
from ditto.db.queries.agents import get_agent_by_id
from ditto.db.queries.heartbeats import upsert_validator_heartbeat
from ditto.db.queries.tickets import get_open_ticket

pytestmark = pytest.mark.integration

_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


@pytest.fixture(scope="module", autouse=True)
def _alembic_upgrade_head() -> None:
    subprocess.run(
        ["uv", "run", "alembic", "upgrade", "head"],
        check=True,
        env=os.environ.copy(),
        capture_output=True,
    )


@pytest.fixture
async def session_maker() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_db_engine()
    async with engine.begin() as connection:
        await connection.execute(
            text("TRUNCATE TABLE validator_heartbeats, agents CASCADE")
        )
    try:
        yield create_session_maker(engine)
    finally:
        await engine.dispose()


async def _upsert_idle(
    session: AsyncSession, *, reported_at: datetime
) -> tuple[ValidatorHeartbeat, bool]:
    return await upsert_validator_heartbeat(
        session,
        validator_hotkey=_HOTKEY,
        software_version="1.2.3",
        protocol_version=4,
        code_digest="ab" * 32,
        state="idle",
        active_agent_id=None,
        system_metrics=None,
        benchmark_progress=None,
        reported_at=reported_at,
        seen_at=reported_at,
        signature="cd" * 64,
    )


async def test_concurrent_first_heartbeat_uses_on_conflict_loser_path(
    session_maker: async_sessionmaker[AsyncSession],
) -> None:
    """Two missing-row writers serialize without a PK error or lost update."""
    first_at = datetime.now(UTC)
    second_at = first_at + timedelta(seconds=1)
    second_started = asyncio.Event()

    async def second_writer() -> tuple[ValidatorHeartbeat, bool]:
        async with session_maker() as session, session.begin():
            second_started.set()
            return await _upsert_idle(session, reported_at=second_at)

    async with session_maker() as first_session:
        transaction = await first_session.begin()
        _, first_accepted = await _upsert_idle(first_session, reported_at=first_at)
        assert first_accepted is True

        second_task = asyncio.create_task(second_writer())
        await second_started.wait()
        await asyncio.sleep(0.05)
        assert not second_task.done(), "second INSERT should wait on the PK conflict"
        await transaction.commit()

    second_row, second_accepted = await asyncio.wait_for(second_task, timeout=2)
    assert second_accepted is True
    assert second_row.reported_at == second_at

    async with session_maker() as session:
        count = await session.scalar(
            select(func.count()).select_from(ValidatorHeartbeat)
        )
        row = await session.get(ValidatorHeartbeat, _HOTKEY)
    assert count == 1
    assert row is not None and row.reported_at == second_at


async def test_progress_waiting_behind_score_lock_rechecks_consumed_ticket(
    session_maker: async_sessionmaker[AsyncSession],
) -> None:
    """A score that wins the Agent lock prevents late progress resurrection.

    This exercises the same Agent-then-ticket lock order used by both endpoints.
    The delayed progress transaction cannot inspect the ticket until the scoring
    transaction commits it as spent, then its mandatory in-transaction recheck
    returns no open ticket and it never writes a heartbeat.
    """
    now = datetime.now(UTC)
    deadline = now + timedelta(minutes=30)
    agent_id = uuid4()
    async with session_maker() as session, session.begin():
        session.add(
            Agent(
                agent_id=agent_id,
                miner_hotkey="5DhaT8U7LVwnnJNUU8VL1XEipicatoaDVVq7cHo227gogVZm",
                name="concurrency-agent",
                sha256="ab" * 32,
                size_bytes=524288,
                status=AgentStatus.EVALUATING,
                created_at=now,
            )
        )
        session.add(
            ValidatorTicket(
                agent_id=agent_id,
                validator_hotkey=_HOTKEY,
                status=TicketStatus.ISSUED,
                issued_at=now,
                deadline=deadline,
            )
        )

    progress_started = asyncio.Event()

    async def delayed_progress() -> bool:
        async with session_maker() as session, session.begin():
            progress_started.set()
            agent = await get_agent_by_id(session, agent_id=agent_id, for_update=True)
            assert agent is not None
            ticket = await get_open_ticket(
                session,
                agent_id=agent_id,
                validator_hotkey=_HOTKEY,
                now=now,
                deadline=deadline,
                for_update=True,
            )
            if agent.status != AgentStatus.EVALUATING or ticket is None:
                return False
            await upsert_validator_heartbeat(
                session,
                validator_hotkey=_HOTKEY,
                software_version="1.2.3",
                protocol_version=4,
                code_digest="ab" * 32,
                state="running_benchmark",
                active_agent_id=agent_id,
                system_metrics=None,
                benchmark_progress={
                    "stage": "running_benchmark",
                    "completed": 1,
                    "total": 114,
                    "ticket_deadline": deadline.isoformat(),
                },
                reported_at=now,
                seen_at=now,
                signature="cd" * 64,
            )
            return True

    async with session_maker() as score_session:
        score_transaction = await score_session.begin()
        agent = await get_agent_by_id(score_session, agent_id=agent_id, for_update=True)
        assert agent is not None
        ticket = await get_open_ticket(
            score_session,
            agent_id=agent_id,
            validator_hotkey=_HOTKEY,
            now=now,
            deadline=deadline,
            for_update=True,
        )
        assert ticket is not None

        progress_task = asyncio.create_task(delayed_progress())
        await progress_started.wait()
        await asyncio.sleep(0.05)
        assert not progress_task.done(), "progress should wait behind the Agent lock"

        ticket.status = TicketStatus.SCORED
        await score_session.flush()
        await score_transaction.commit()

    assert await asyncio.wait_for(progress_task, timeout=2) is False
    async with session_maker() as session:
        heartbeat = await session.get(ValidatorHeartbeat, _HOTKEY)
        spent_ticket = await session.get(ValidatorTicket, (agent_id, _HOTKEY))
    assert heartbeat is None
    assert spent_ticket is not None and spent_ticket.status == TicketStatus.SCORED
