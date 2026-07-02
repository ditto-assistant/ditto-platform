"""Unit tests for :mod:`ditto.api_server.endpoints.validator`.

These exercise the real endpoints end to end against an in-memory SQLite
database (real queries, real status transitions) with the chain + storage
dependencies mocked. Signatures are produced with a real sr25519 dev
keypair so the signature-verification path runs for real.
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
    ERROR_CODE_AGENT_NOT_EVALUATABLE,
    ERROR_CODE_AGENT_NOT_FOUND,
    ERROR_CODE_VALIDATION,
    ERROR_CODE_VALIDATOR_AUTH,
)
from ditto.chain.models import NeuronInfo
from ditto.db.models import Agent, Base, Score

# Real dev keypair: signs for real so _verify_signature runs end to end.
_KEYPAIR = bittensor.Keypair.create_from_uri("//Alice")
_VALIDATOR_HOTKEY = _KEYPAIR.ss58_address
_MINER_HOTKEY = "5DhaT8U7LVwnnJNUU8VL1XEipicatoaDVVq7cHo227gogVZm"
_SHA256 = "ab" * 32


def _sign(message: str) -> str:
    return _KEYPAIR.sign(message.encode()).hex()


def _score_payload(
    agent_id: UUID, run_id: str = "run_test_1", **overrides: object
) -> dict:
    report = {
        "run_id": run_id,
        "seed": 8675309,
        "composite": 0.82,
        "tool_mean": 0.88,
        "memory_mean": 0.73,
        "median_ms": 812,
        "n": 30,
        "generated_at": "2026-06-08T12:04:30Z",
        "per_case": [],
    }
    report.update(overrides)
    signed = (
        f"{_VALIDATOR_HOTKEY}:{agent_id}:{run_id}:"
        f"{report['composite']!r}:{report['seed']}"
    )
    return {
        "validator_hotkey": _VALIDATOR_HOTKEY,
        "signature": _sign(signed),
        "report": report,
    }


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
                hotkey=_VALIDATOR_HOTKEY,
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
    miner_hotkey: str = _MINER_HOTKEY,
    sha256: str = _SHA256,
    size_bytes: int = 524288,
) -> UUID:
    aid = agent_id or uuid4()
    async with maker() as s, s.begin():
        s.add(
            Agent(
                agent_id=aid,
                miner_hotkey=miner_hotkey,
                name=name,
                sha256=sha256,
                size_bytes=size_bytes,
                status=status,
                created_at=created_at or datetime.now(UTC),
            )
        )
    return aid


_AUTH_HEADER = {"X-Validator-Hotkey": _VALIDATOR_HOTKEY}


# --- Queue -----------------------------------------------------------------


class TestQueue:
    async def test_lists_only_evaluating_oldest_first(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        base = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            name="younger",
            created_at=base + timedelta(minutes=5),
        )
        await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            name="older",
            created_at=base,
        )
        # Not in the evaluating state -> excluded from the queue.
        await _seed_agent(session_maker, status=AgentStatus.UPLOADED, name="pending")
        _install_db(app, session_maker)
        _install_chain(app)

        response = await client.get("/api/v1/validator/queue", headers=_AUTH_HEADER)
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        body = response.json()
        assert body["count"] == 2
        assert [i["name"] for i in body["items"]] == ["older", "younger"]
        assert all(i["status"] == AgentStatus.EVALUATING for i in body["items"])

    async def test_limit_caps_results(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        for i in range(3):
            await _seed_agent(
                session_maker, status=AgentStatus.EVALUATING, name=f"a{i}"
            )
        _install_db(app, session_maker)
        _install_chain(app)

        response = await client.get(
            "/api/v1/validator/queue?limit=2", headers=_AUTH_HEADER
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
        response = await client.get("/api/v1/validator/queue")
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_VALIDATOR_AUTH

    async def test_unpermitted_validator_returns_401(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app, permitted=False)
        response = await client.get("/api/v1/validator/queue", headers=_AUTH_HEADER)
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_VALIDATOR_AUTH

    async def test_limit_out_of_range_returns_422(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        response = await client.get(
            "/api/v1/validator/queue?limit=0", headers=_AUTH_HEADER
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
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        storage = _install_storage(app)

        response = await client.get(
            f"/api/v1/validator/agent/{agent_id}/artifact", headers=_AUTH_HEADER
        )
        assert response.status_code == 200
        body = response.json()
        assert body["agent_id"] == str(agent_id)
        assert body["sha256"] == _SHA256
        assert body["download_url"].startswith("https://")
        storage.presigned_get_url.assert_awaited_once()
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
            f"/api/v1/validator/agent/{uuid4()}/artifact", headers=_AUTH_HEADER
        )
        assert response.status_code == 404
        assert response.json()["error_code"] == ERROR_CODE_AGENT_NOT_FOUND


# --- Submit score ----------------------------------------------------------


class TestSubmitScore:
    async def test_records_score_and_finalizes(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)

        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["agent_id"] == str(agent_id)
        assert body["status"] == AgentStatus.SCORED
        assert body["accepted"] is True

        # A scores row landed and the agent transitioned.
        async with session_maker() as s:
            score = await s.get(Score, (agent_id, _VALIDATOR_HOTKEY))
            assert score is not None
            assert score.composite == pytest.approx(0.82)
            agent = await s.get(Agent, agent_id)
            assert agent is not None
            assert agent.status == AgentStatus.SCORED

    async def test_re_score_upserts_single_row(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)

        await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id, run_id="run_a", composite=0.5),
        )
        r2 = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id, run_id="run_b", composite=0.9),
        )
        assert r2.status_code == 200

        async with session_maker() as s:
            from ditto.db.queries.scores import list_scores_for_agent

            scores = await list_scores_for_agent(s, agent_id=agent_id)
            assert len(scores) == 1
            assert scores[0].run_id == "run_b"
            assert scores[0].composite == pytest.approx(0.9)

    async def test_bad_signature_returns_401(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)

        payload = _score_payload(agent_id)
        payload["signature"] = "ab" * 64  # well-formed but wrong
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score", json=payload
        )
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_VALIDATOR_AUTH

    async def test_unpermitted_validator_returns_401(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app, permitted=False)
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id),
        )
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_VALIDATOR_AUTH

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
            f"/api/v1/validator/agent/{aid}/score", json=_score_payload(aid)
        )
        assert response.status_code == 404
        assert response.json()["error_code"] == ERROR_CODE_AGENT_NOT_FOUND

    async def test_non_scoreable_status_returns_409(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id),
        )
        assert response.status_code == 409
        assert response.json()["error_code"] == ERROR_CODE_AGENT_NOT_EVALUATABLE

    async def test_re_score_live_agent_keeps_live(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.LIVE)
        _install_db(app, session_maker)
        _install_chain(app)
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id),
        )
        assert response.status_code == 200
        assert response.json()["status"] == AgentStatus.LIVE

    async def test_out_of_range_composite_returns_422(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id, composite=1.5),
        )
        assert response.status_code == 422
        assert response.json()["error_code"] == ERROR_CODE_VALIDATION

    async def test_cross_agent_replay_rejected(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # A signature valid for agent A must not be accepted for agent B: the
        # signed payload binds the agent id.
        agent_a = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        agent_b = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        payload = _score_payload(agent_a)  # signed for A
        response = await client.post(
            f"/api/v1/validator/agent/{agent_b}/score", json=payload
        )
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_VALIDATOR_AUTH

    async def test_tampered_composite_rejected(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # The composite is signed: altering it after signing invalidates the sig.
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        payload = _score_payload(agent_id, composite=0.50)
        payload["report"]["composite"] = 0.99  # tamper post-signing
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score", json=payload
        )
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_VALIDATOR_AUTH


_MINER_B = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


class TestAntiCopyGate:
    """The score-write path holds a suspected copy in ath_pending_review."""

    async def _score(
        self,
        client: httpx.AsyncClient,
        agent_id: UUID,
        *,
        run_id: str,
        composite: float,
    ) -> httpx.Response:
        return await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id, run_id=run_id, composite=composite),
        )

    async def test_exact_copy_is_held(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        # Incumbent scores + becomes eligible.
        incumbent = await _seed_agent(
            session_maker, status=AgentStatus.EVALUATING, sha256="cc" * 32
        )
        await self._score(client, incumbent, run_id="run_inc", composite=0.80)
        # A byte-identical resubmission from another miner.
        copy = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            miner_hotkey=_MINER_B,
            sha256="cc" * 32,
        )
        resp = await self._score(client, copy, run_id="run_copy", composite=0.80)
        assert resp.status_code == 200
        assert resp.json()["status"] == AgentStatus.ATH_PENDING_REVIEW

        async with session_maker() as s:
            held = await s.get(Agent, copy)
            assert held is not None
            assert held.status == AgentStatus.ATH_PENDING_REVIEW
            assert held.duplicate_of == incumbent
            assert "sha256" in (held.review_reason or "")

    async def test_near_dup_dethroner_is_held(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        incumbent = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            sha256="aa" * 32,
            size_bytes=500000,
        )
        await self._score(client, incumbent, run_id="run_inc", composite=0.80)
        # Different bytes, near-identical size, beats incumbent by a hair.
        tweaked = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            miner_hotkey=_MINER_B,
            sha256="bb" * 32,
            size_bytes=500100,
        )
        resp = await self._score(client, tweaked, run_id="run_tweak", composite=0.805)
        assert resp.status_code == 200
        assert resp.json()["status"] == AgentStatus.ATH_PENDING_REVIEW

    async def test_genuine_improvement_not_held(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        incumbent = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            sha256="aa" * 32,
            size_bytes=500000,
        )
        await self._score(client, incumbent, run_id="run_inc", composite=0.80)
        better = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            miner_hotkey=_MINER_B,
            sha256="bb" * 32,
            size_bytes=700000,
        )
        resp = await self._score(client, better, run_id="run_better", composite=0.92)
        assert resp.status_code == 200
        assert resp.json()["status"] == AgentStatus.SCORED

    async def test_rescore_of_held_agent_stays_held_no_409(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        incumbent = await _seed_agent(
            session_maker, status=AgentStatus.EVALUATING, sha256="cc" * 32
        )
        await self._score(client, incumbent, run_id="run_inc", composite=0.80)
        copy = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            miner_hotkey=_MINER_B,
            sha256="cc" * 32,
        )
        await self._score(client, copy, run_id="run_copy", composite=0.80)
        # Re-scoring a held agent must not 409 and must not un-hold it.
        resp = await self._score(client, copy, run_id="run_copy2", composite=0.81)
        assert resp.status_code == 200
        assert resp.json()["status"] == AgentStatus.ATH_PENDING_REVIEW
