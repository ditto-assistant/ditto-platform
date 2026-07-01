"""Unit tests for :mod:`ditto.api_server.endpoints.screener`.

Exercise the real endpoints end to end against in-memory SQLite (real queries,
real status transitions) with chain + storage mocked. Signatures use a real
sr25519 dev keypair so the verification path runs for real.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import bittensor
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
from ditto.api_server.dependencies import (
    get_chain_client,
    get_session,
    get_storage_client,
)
from ditto.api_server.middleware.error_envelope import (
    ERROR_CODE_AGENT_NOT_FOUND,
    ERROR_CODE_AGENT_NOT_SCREENABLE,
    ERROR_CODE_SCREENER_AUTH,
    ERROR_CODE_VALIDATION,
)
from ditto.chain.models import NeuronInfo
from ditto.db.models import Agent, Base

_KEYPAIR = bittensor.Keypair.create_from_uri("//Alice")
_SCREENER_HOTKEY = _KEYPAIR.ss58_address
_MINER_HOTKEY = "5DhaT8U7LVwnnJNUU8VL1XEipicatoaDVVq7cHo227gogVZm"
_SHA256 = "ab" * 32


def _sign(message: str) -> str:
    return _KEYPAIR.sign(message.encode()).hex()


def _result_payload(
    agent_id: UUID, *, passed: bool = True, **overrides: object
) -> dict:
    body = {
        "screener_hotkey": _SCREENER_HOTKEY,
        "signature": _sign(f"{_SCREENER_HOTKEY}:{agent_id}"),
        "passed": passed,
        "detail": "",
    }
    body.update(overrides)
    return body


# --- DB + dependency wiring ------------------------------------------------


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


def _install_chain(
    app: FastAPI, *, permitted: bool = True, registered: bool = True
) -> None:
    neurons = []
    if registered:
        neurons.append(
            NeuronInfo(
                hotkey=_SCREENER_HOTKEY,
                coldkey="5GReceiverColdkeyPlaceholderXXXXXXXXXXXXXXXXXXX",
                uid=1,
                stake=1000.0,
                validator_permit=permitted,
            )
        )

    async def _chain() -> MagicMock:
        c = MagicMock()
        c.get_recent_neurons = AsyncMock(return_value=neurons)
        return c

    app.dependency_overrides[get_chain_client] = _chain


def _install_storage(app: FastAPI) -> MagicMock:
    storage = MagicMock()
    storage.presigned_get_url = AsyncMock(
        return_value="https://signed.example/ditto-agents/x.tar.gz?sig=1"
    )

    async def _storage() -> MagicMock:
        return storage

    app.dependency_overrides[get_storage_client] = _storage
    return storage


async def _seed_agent(
    maker: async_sessionmaker[AsyncSession],
    *,
    status: AgentStatus,
    name: str = "alpha-agent",
    created_at: datetime | None = None,
    agent_id: UUID | None = None,
) -> UUID:
    aid = agent_id or uuid4()
    async with maker() as s, s.begin():
        s.add(
            Agent(
                agent_id=aid,
                miner_hotkey=_MINER_HOTKEY,
                name=name,
                sha256=_SHA256,
                status=status,
                created_at=created_at or datetime.now(UTC),
            )
        )
    return aid


_AUTH_HEADER = {"X-Screener-Hotkey": _SCREENER_HOTKEY}


# --- Queue -----------------------------------------------------------------


class TestQueue:
    async def test_lists_only_uploaded_oldest_first(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        base = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        await _seed_agent(
            session_maker,
            status=AgentStatus.UPLOADED,
            name="younger",
            created_at=base + timedelta(minutes=5),
        )
        await _seed_agent(
            session_maker, status=AgentStatus.UPLOADED, name="older", created_at=base
        )
        # Already promoted -> excluded from the screener queue.
        await _seed_agent(
            session_maker, status=AgentStatus.EVALUATING, name="promoted"
        )
        _install_db(app, session_maker)
        _install_chain(app)

        response = await client.get("/api/v1/screener/queue", headers=_AUTH_HEADER)
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        body = response.json()
        assert body["count"] == 2
        assert [i["name"] for i in body["items"]] == ["older", "younger"]
        assert all(i["status"] == AgentStatus.UPLOADED for i in body["items"])

    async def test_limit_caps_results(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        for i in range(3):
            await _seed_agent(session_maker, status=AgentStatus.UPLOADED, name=f"a{i}")
        _install_db(app, session_maker)
        _install_chain(app)

        response = await client.get(
            "/api/v1/screener/queue?limit=2", headers=_AUTH_HEADER
        )
        assert response.status_code == 200
        assert response.json()["count"] == 2

    async def test_missing_auth_header_returns_401(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        response = await client.get("/api/v1/screener/queue")
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_SCREENER_AUTH

    async def test_unpermitted_returns_401(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app, permitted=False)
        response = await client.get("/api/v1/screener/queue", headers=_AUTH_HEADER)
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_SCREENER_AUTH

    async def test_limit_out_of_range_returns_422(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        response = await client.get(
            "/api/v1/screener/queue?limit=0", headers=_AUTH_HEADER
        )
        assert response.status_code == 422
        assert response.json()["error_code"] == ERROR_CODE_VALIDATION


# --- Artifact --------------------------------------------------------------


class TestArtifact:
    async def test_returns_presigned_url_and_sha(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)
        storage = _install_storage(app)

        response = await client.get(
            f"/api/v1/screener/agent/{agent_id}/artifact", headers=_AUTH_HEADER
        )
        assert response.status_code == 200
        body = response.json()
        assert body["agent_id"] == str(agent_id)
        assert body["sha256"] == _SHA256
        assert body["download_url"].startswith("https://")
        assert (
            storage.presigned_get_url.await_args.kwargs["key"]
            == f"{agent_id}/agent.tar.gz"
        )

    async def test_unknown_agent_returns_404(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        _install_storage(app)
        response = await client.get(
            f"/api/v1/screener/agent/{uuid4()}/artifact", headers=_AUTH_HEADER
        )
        assert response.status_code == 404
        assert response.json()["error_code"] == ERROR_CODE_AGENT_NOT_FOUND


# --- Submit result ---------------------------------------------------------


class TestSubmitResult:
    async def test_pass_promotes_to_evaluating(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)

        response = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result",
            json=_result_payload(agent_id, passed=True),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == AgentStatus.EVALUATING
        assert body["accepted"] is True

        async with session_maker() as s:
            agent = await s.get(Agent, agent_id)
            assert agent is not None
            assert agent.status == AgentStatus.EVALUATING

    async def test_fail_moves_to_screening_failed(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)

        response = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result",
            json=_result_payload(agent_id, passed=False, detail="cargo build failed"),
        )
        assert response.status_code == 200
        assert response.json()["status"] == AgentStatus.SCREENING_FAILED

    async def test_pass_is_idempotent(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)

        first = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result",
            json=_result_payload(agent_id, passed=True),
        )
        second = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result",
            json=_result_payload(agent_id, passed=True),
        )
        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json()["status"] == AgentStatus.EVALUATING

    async def test_promotes_from_screening_state(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.SCREENING)
        _install_db(app, session_maker)
        _install_chain(app)
        response = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result",
            json=_result_payload(agent_id, passed=True),
        )
        assert response.status_code == 200
        assert response.json()["status"] == AgentStatus.EVALUATING

    async def test_conflicting_verdict_on_promoted_agent_returns_409(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # Agent already promoted; a fail verdict now must not demote it.
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        response = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result",
            json=_result_payload(agent_id, passed=False),
        )
        assert response.status_code == 409
        assert response.json()["error_code"] == ERROR_CODE_AGENT_NOT_SCREENABLE

    async def test_verdict_on_scored_agent_returns_409(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.SCORED)
        _install_db(app, session_maker)
        _install_chain(app)
        response = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result",
            json=_result_payload(agent_id, passed=True),
        )
        assert response.status_code == 409
        assert response.json()["error_code"] == ERROR_CODE_AGENT_NOT_SCREENABLE

    async def test_bad_signature_returns_401(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)
        payload = _result_payload(agent_id)
        payload["signature"] = "ab" * 64  # well-formed but wrong
        response = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result", json=payload
        )
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_SCREENER_AUTH

    async def test_unpermitted_returns_401(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app, permitted=False)
        response = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result",
            json=_result_payload(agent_id),
        )
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_SCREENER_AUTH

    async def test_unknown_agent_returns_404(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        aid = uuid4()
        response = await client.post(
            f"/api/v1/screener/agent/{aid}/result", json=_result_payload(aid)
        )
        assert response.status_code == 404
        assert response.json()["error_code"] == ERROR_CODE_AGENT_NOT_FOUND
