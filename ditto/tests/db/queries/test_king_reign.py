"""Unit tests for the write-once king-reign ledger.

Exercises the real ORM + SQLite-in-memory engine so the write-once invariant
(a later re-coronation never moves ``first_crowned_at``) is enforced by the
same code path production runs, not a mock.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.db.models import Agent, AgentStatus
from ditto.db.queries.king_reign import get_first_crowned, record_first_crowned

pytestmark = pytest.mark.asyncio


async def _agent(session: AsyncSession, agent_id: object) -> None:
    session.add(
        Agent(
            agent_id=agent_id,
            miner_hotkey="5HKAlphaHotkey",
            name="alpha-agent",
            sha256="deadbeef" * 8,
            size_bytes=524288,
            status=AgentStatus.SCORED,
        )
    )
    await session.flush()


async def test_first_coronation_is_write_once(session: AsyncSession) -> None:
    agent_id = uuid4()
    await _agent(session, agent_id)
    first = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
    later = first + timedelta(hours=5)

    await record_first_crowned(session, agent_id=agent_id, now=first)
    # A later re-coronation must not move the anchored release deadline: a brief
    # early reign is what arms the window.
    await record_first_crowned(session, agent_id=agent_id, now=later)

    crowned = await get_first_crowned(session, agent_ids=[agent_id])
    assert crowned[agent_id] == first


async def test_get_first_crowned_only_returns_kings(session: AsyncSession) -> None:
    king_id = uuid4()
    commoner_id = uuid4()
    await _agent(session, king_id)
    await _agent(session, commoner_id)
    crowned_at = datetime(2026, 7, 21, 0, 0, tzinfo=UTC)
    await record_first_crowned(session, agent_id=king_id, now=crowned_at)

    result = await get_first_crowned(session, agent_ids=[king_id, commoner_id])
    assert result == {king_id: crowned_at}
    assert await get_first_crowned(session, agent_ids=[]) == {}
