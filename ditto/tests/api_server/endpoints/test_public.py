"""Unit tests for :mod:`ditto.api_server.endpoints.public`.

``GET /api/v1/public/leaderboard`` is open (no validator auth) and aggregate-only:
it must rank miners by composite, expose tool/memory means, and NEVER leak the
integrity-internal fields (``signature``, ``sha256``, ``validator_hotkey``) or
per-case detail.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ditto.api_models.agent_status import AgentStatus
from ditto.api_server.dependencies import get_session
from ditto.db.models import Agent, Base
from ditto.db.queries.scores import upsert_score

_MINER_A = "5DhaT8U7LVwnnJNUU8VL1XEipicatoaDVVq7cHo227gogVZm"
_MINER_B = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(eng.sync_engine, "connect")
    def _enable_fk(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest.fixture
def session_maker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


def _install_db(app: FastAPI, maker: async_sessionmaker[AsyncSession]) -> None:
    async def _session() -> AsyncIterator[AsyncSession]:
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = _session


async def _seed_scored(
    maker: async_sessionmaker[AsyncSession],
    *,
    miner: str,
    composite: float,
    tool_mean: float,
    memory_mean: float,
    status: AgentStatus = AgentStatus.SCORED,
) -> None:
    async with maker() as s, s.begin():
        agent = Agent(
            agent_id=uuid4(),
            miner_hotkey=miner,
            name="agent",
            sha256="ab" * 32,
            size_bytes=524288,
            status=status,
            created_at=datetime.now(UTC),
        )
        s.add(agent)
        await s.flush()
        await upsert_score(
            s,
            agent_id=agent.agent_id,
            validator_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            run_id="run_1",
            seed=42,
            composite=composite,
            tool_mean=tool_mean,
            memory_mean=memory_mean,
            median_ms=500,
            n=20,
            generated_at=datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC),
            signature="ab" * 64,
        )


class TestPublicLeaderboard:
    async def test_ranks_by_composite_and_exposes_aggregates(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_scored(
            session_maker, miner=_MINER_A, composite=0.4, tool_mean=0.5, memory_mean=0.3
        )
        await _seed_scored(
            session_maker,
            miner=_MINER_B,
            composite=0.9,
            tool_mean=0.95,
            memory_mean=0.8,
        )
        # Held (suspected copy) must not surface.
        await _seed_scored(
            session_maker,
            miner="5HeldMinerXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            composite=0.99,
            tool_mean=0.99,
            memory_mean=0.99,
            status=AgentStatus.ATH_PENDING_REVIEW,
        )
        _install_db(app, session_maker)

        resp = await client.get("/api/v1/public/leaderboard")
        assert resp.status_code == 200
        assert resp.headers["Cache-Control"] == "public, max-age=30"
        body = resp.json()
        assert body["count"] == 2
        assert [e["rank"] for e in body["entries"]] == [1, 2]
        assert [e["miner_hotkey"] for e in body["entries"]] == [_MINER_B, _MINER_A]
        top = body["entries"][0]
        assert top["composite"] == pytest.approx(0.9)
        assert top["tool_mean"] == pytest.approx(0.95)
        assert top["memory_mean"] == pytest.approx(0.8)

    async def test_never_leaks_integrity_fields(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_scored(
            session_maker, miner=_MINER_A, composite=0.4, tool_mean=0.5, memory_mean=0.3
        )
        _install_db(app, session_maker)

        resp = await client.get("/api/v1/public/leaderboard")
        entry = resp.json()["entries"][0]
        for leaked in ("signature", "sha256", "validator_hotkey", "agent_id", "seed"):
            assert leaked not in entry

    async def test_empty_ledger(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        resp = await client.get("/api/v1/public/leaderboard")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 0
        assert body["entries"] == []

    async def test_no_auth_required(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        # No X-Validator-Hotkey header, no chain override — must still succeed.
        resp = await client.get("/api/v1/public/leaderboard")
        assert resp.status_code == 200
