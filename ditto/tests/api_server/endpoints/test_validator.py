"""Unit tests for :mod:`ditto.api_server.endpoints.validator`.

These exercise the real endpoints end to end against an in-memory SQLite
database (real queries, real status transitions) with the chain + storage
dependencies mocked. Signatures are produced with a real sr25519 dev
keypair so the signature-verification path runs for real.
"""

from __future__ import annotations

import json
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
from ditto.api_models.ticket_status import TicketStatus
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
from ditto.db.models import Agent, Base, Score, ValidatorTicket

# Real dev keypairs: sign for real so _verify_signature runs end to end. The k=3
# quorum needs three distinct permitted validators before an agent finalizes.
_KEYPAIRS = [
    bittensor.Keypair.create_from_uri(uri) for uri in ("//Alice", "//Bob", "//Charlie")
]
_KEYPAIR = _KEYPAIRS[0]
_VALIDATOR_HOTKEY = _KEYPAIR.ss58_address
# A fourth validator, used only to prove an expired ticket re-opens a slot for a
# validator that was shut out when the k=3 pool was full.
_DAVE = bittensor.Keypair.create_from_uri("//Dave")
_MINER_HOTKEY = "5DhaT8U7LVwnnJNUU8VL1XEipicatoaDVVq7cHo227gogVZm"
_SHA256 = "ab" * 32
_TICKET_DEADLINE = datetime(2030, 1, 1, tzinfo=UTC)


def _sign(message: str) -> str:
    return _KEYPAIR.sign(message.encode()).hex()


def _score_payload(
    agent_id: UUID,
    run_id: str = "run_test_1",
    *,
    keypair: bittensor.Keypair = _KEYPAIR,
    **overrides: object,
) -> dict:
    ticket_deadline = overrides.pop("ticket_deadline", _TICKET_DEADLINE)
    assert isinstance(ticket_deadline, datetime)
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
    hotkey = keypair.ss58_address
    lease = ticket_deadline.astimezone(UTC).isoformat(timespec="microseconds")
    signed = (
        f"{hotkey}:{agent_id}:{lease}:{run_id}:{report['composite']!r}:{report['seed']}"
    )
    return {
        "validator_hotkey": hotkey,
        "ticket_deadline": ticket_deadline.isoformat(),
        "signature": keypair.sign(signed.encode()).hex(),
        "report": report,
    }


async def _score_to_quorum(
    client: httpx.AsyncClient,
    agent_id: UUID,
    *,
    maker: async_sessionmaker[AsyncSession],
    run_id: str = "run_q",
    composite: float = 0.82,
    **overrides: object,
) -> httpx.Response:
    """Seed a ticket for each quorum validator and post one score each (all at
    ``composite``, so the median is ``composite``); return the final response,
    finalized on the last."""
    resp: httpx.Response | None = None
    for i, kp in enumerate(_KEYPAIRS):
        await _seed_ticket(maker, agent_id, keypair=kp)
        resp = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(
                agent_id,
                run_id=f"{run_id}_{i}",
                keypair=kp,
                composite=composite,
                **overrides,
            ),
        )
        assert resp.status_code == 200, resp.text
    assert resp is not None
    return resp


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
    app: FastAPI,
    *,
    permitted: bool = True,
    registered: bool = True,
    extra_keypairs: tuple[bittensor.Keypair, ...] = (),
) -> None:
    neurons = []
    if registered:
        for uid, kp in enumerate((*_KEYPAIRS, *extra_keypairs), start=1):
            neurons.append(
                NeuronInfo(
                    hotkey=kp.ss58_address,
                    coldkey="5GReceiverColdkeyPlaceholderXXXXXXXXXXXXXXXXXXX",
                    uid=uid,
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


async def _seed_ticket(
    maker: async_sessionmaker[AsyncSession],
    agent_id: UUID,
    *,
    keypair: bittensor.Keypair = _KEYPAIR,
    deadline: datetime = _TICKET_DEADLINE,
) -> None:
    """Seat (or re-open) an issued ticket for a specific (agent, validator) so a
    score against that agent is accepted by the k=3 gate. Upserts so a test can
    simulate the platform re-issuing a ticket to the same validator."""
    async with maker() as s, s.begin():
        existing = await s.get(ValidatorTicket, (agent_id, keypair.ss58_address))
        if existing is None:
            s.add(
                ValidatorTicket(
                    agent_id=agent_id,
                    validator_hotkey=keypair.ss58_address,
                    status=TicketStatus.ISSUED,
                    issued_at=datetime.now(UTC),
                    deadline=deadline,
                )
            )
        else:
            existing.status = TicketStatus.ISSUED
            existing.deadline = deadline


_AUTH_HEADER = {"X-Validator-Hotkey": _VALIDATOR_HOTKEY}


# --- Queue -----------------------------------------------------------------


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


class TestRequestJob:
    async def test_issues_ticket_for_evaluating_agent(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        resp = await client.post("/api/v1/validator/job", headers=_AUTH_HEADER)
        assert resp.status_code == 200
        body = resp.json()
        assert body["agent_id"] == str(agent_id)
        assert "deadline" in body

    async def test_no_work_returns_204(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        resp = await client.post("/api/v1/validator/job", headers=_AUTH_HEADER)
        assert resp.status_code == 204

    async def test_caps_at_quorum_across_validators(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        # Three distinct validators each get the single agent (fills the pool).
        for kp in _KEYPAIRS:
            r = await client.post(
                "/api/v1/validator/job",
                headers={"X-Validator-Hotkey": kp.ss58_address},
            )
            assert r.status_code == 200
        # A further request finds no open slot -> no job.
        r = await client.post("/api/v1/validator/job", headers=_AUTH_HEADER)
        assert r.status_code == 204

    async def test_unpermitted_validator_returns_401(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app, permitted=False)
        resp = await client.post("/api/v1/validator/job", headers=_AUTH_HEADER)
        assert resp.status_code == 401


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

        # A single below-quorum score records the row but keeps the agent
        # provisional (evaluating) — no finalization until the k=3 quorum.
        await _seed_ticket(session_maker, agent_id, keypair=_KEYPAIRS[0])
        first = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id, keypair=_KEYPAIRS[0]),
        )
        assert first.status_code == 200
        assert first.json()["status"] == AgentStatus.EVALUATING

        # The quorum-th score finalizes it on the median composite.
        response = await _score_to_quorum(
            client, agent_id, maker=session_maker, composite=0.82
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

    async def test_finalize_writes_verifiable_audit_chain(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        from ditto.db.queries.audit import (
            EVENT_FINALIZED,
            EVENT_SCORE,
            list_audit_entries,
            verify_audit_chain,
        )

        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)

        # One below-quorum score, then the k=3 quorum (which re-scores validator 0
        # with a fresh ticket): 4 append-only score events + 1 finalize.
        await _seed_ticket(session_maker, agent_id, keypair=_KEYPAIRS[0])
        await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id, keypair=_KEYPAIRS[0]),
        )
        await _score_to_quorum(client, agent_id, maker=session_maker, composite=0.82)

        async with session_maker() as s:
            entries = await list_audit_entries(s, limit=1000)
        # Append-only: the re-score is its own entry even though the table upserts.
        score_entries = [e for e in entries if e.event == EVENT_SCORE]
        finalized = [e for e in entries if e.event == EVENT_FINALIZED]
        assert len(score_entries) == 4
        assert len(finalized) == 1
        assert entries[-1].event == EVENT_FINALIZED
        # The finalize entry carries the median + quorum + scoring validators.
        fin = finalized[0].payload
        assert fin["median_composite"] == pytest.approx(0.82)
        assert fin["quorum"] == 3
        assert fin["score_count"] == 3
        assert len(fin["validator_hotkeys"]) == 3
        assert fin["status"] == AgentStatus.SCORED.value
        # The whole chain replays and verifies (tamper-evident end to end).
        assert verify_audit_chain(entries) is True

    async def test_stamps_current_bench_version_when_omitted(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # A report that omits bench_version (as the default payload does) must be
        # stamped with the current version so it is never recorded as legacy.
        from ditto.api_server.bench import CURRENT_BENCH_VERSION

        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)

        await _seed_ticket(session_maker, agent_id)
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id),
        )
        assert response.status_code == 200

        async with session_maker() as s:
            score = await s.get(Score, (agent_id, _VALIDATOR_HOTKEY))
            assert score is not None
            assert score.details is not None
            assert score.details["bench_version"] == CURRENT_BENCH_VERSION

    async def test_preserves_explicit_bench_version(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # An explicit (older) version in the report is honest provenance and must
        # not be bumped — only a missing version gets defaulted.
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)

        await _seed_ticket(session_maker, agent_id)
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id, details={"bench_version": 1}),
        )
        assert response.status_code == 200

        async with session_maker() as s:
            score = await s.get(Score, (agent_id, _VALIDATOR_HOTKEY))
            assert score is not None
            assert score.details is not None
            assert score.details["bench_version"] == 1

    async def test_one_ticket_one_score_no_rescore(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # One ticket, one score: a validator's first score is accepted and
        # consumes its ticket; a second score without a fresh ticket is rejected
        # (409), so a validator cannot re-roll for a better number.
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)

        await _seed_ticket(session_maker, agent_id)
        r1 = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id, run_id="run_a", composite=0.5),
        )
        assert r1.status_code == 200
        # Ticket spent: the re-score has no open ticket.
        r2 = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id, run_id="run_b", composite=0.9),
        )
        assert r2.status_code == 409

        async with session_maker() as s:
            from ditto.db.queries.scores import list_scores_for_agent

            scores = await list_scores_for_agent(s, agent_id=agent_id)
            assert len(scores) == 1
            assert scores[0].run_id == "run_a"  # the first (only) score stands
            assert scores[0].composite == pytest.approx(0.5)

    async def test_superseded_ticket_lease_rejects_late_score(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        old_deadline = _TICKET_DEADLINE
        new_deadline = old_deadline + timedelta(hours=1)
        await _seed_ticket(session_maker, agent_id, deadline=new_deadline)

        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id, ticket_deadline=old_deadline),
        )

        assert response.status_code == 409
        assert "no open scoring ticket" in response.json()["message"]

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
        await _seed_ticket(session_maker, agent_id)
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

    async def test_out_of_range_seed_returns_422(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # A seed outside signed int64 would 500 at the BigInteger insert; it must
        # be a clean 422 before signing/DB work.
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id, seed=2**63),
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
        maker: async_sessionmaker[AsyncSession],
        run_id: str,
        composite: float,
    ) -> httpx.Response:
        # Score to the k=3 quorum so the agent finalizes and the gate runs on
        # the median (= composite, since all three validators post the same).
        return await _score_to_quorum(
            client, agent_id, maker=maker, run_id=run_id, composite=composite
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
        await self._score(
            client, incumbent, maker=session_maker, run_id="run_inc", composite=0.80
        )
        # A byte-identical resubmission from another miner.
        copy = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            miner_hotkey=_MINER_B,
            sha256="cc" * 32,
        )
        resp = await self._score(
            client, copy, maker=session_maker, run_id="run_copy", composite=0.80
        )
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
        await self._score(
            client, incumbent, maker=session_maker, run_id="run_inc", composite=0.80
        )
        # Different bytes, near-identical size, beats incumbent by a hair.
        tweaked = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            miner_hotkey=_MINER_B,
            sha256="bb" * 32,
            size_bytes=500100,
        )
        resp = await self._score(
            client, tweaked, maker=session_maker, run_id="run_tweak", composite=0.805
        )
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
        await self._score(
            client, incumbent, maker=session_maker, run_id="run_inc", composite=0.80
        )
        better = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            miner_hotkey=_MINER_B,
            sha256="bb" * 32,
            size_bytes=700000,
        )
        resp = await self._score(
            client, better, maker=session_maker, run_id="run_better", composite=0.92
        )
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
        await self._score(
            client, incumbent, maker=session_maker, run_id="run_inc", composite=0.80
        )
        copy = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            miner_hotkey=_MINER_B,
            sha256="cc" * 32,
        )
        await self._score(
            client, copy, maker=session_maker, run_id="run_copy", composite=0.80
        )
        # Re-scoring a held agent must not 409 and must not un-hold it.
        resp = await self._score(
            client, copy, maker=session_maker, run_id="run_copy2", composite=0.81
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == AgentStatus.ATH_PENDING_REVIEW


class TestPublicMirror:
    """The finalize hook mirrors the run record to the public bucket."""

    async def test_finalize_publishes_when_public_bucket_configured(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        storage = _install_storage(app)
        storage.public_bucket = "ditto-public"
        storage.put_object = AsyncMock()
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await _score_to_quorum(
            client, agent_id, maker=session_maker, run_id="run_pub", composite=0.5
        )
        storage.put_object.assert_awaited_once()
        kwargs = storage.put_object.await_args.kwargs
        assert kwargs["bucket"] == "ditto-public"
        assert kwargs["key"] == f"scored/{agent_id}.json"
        record = json.loads(kwargs["body"])
        assert record["median_composite"] == 0.5
        assert len(record["scores"]) == 3
        assert all(sc["signature"] for sc in record["scores"])
        assert record["status"] == AgentStatus.SCORED.value

    async def test_finalize_skips_publish_when_unconfigured(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        storage = _install_storage(app)
        storage.public_bucket = None
        storage.put_object = AsyncMock()
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await _score_to_quorum(
            client, agent_id, maker=session_maker, run_id="run_nopub", composite=0.5
        )
        storage.put_object.assert_not_awaited()


class TestMultiValidatorConsensus:
    """The k=3 consensus semantics the decentralized design promises: the
    canonical score is the MEDIAN of the (differing) independent validator
    composites, the full per-validator record is exposed publicly, and an
    expired ticket re-opens the slot so a shut-out validator can pick the agent
    up. These exercise consensus correctness end to end, complementing the
    all-equal-composite quorum tests above."""

    async def test_finalizes_on_median_of_differing_scores(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # Three independent validators disagree. The platform must finalize on the
        # MEDIAN (0.82), never the mean (0.7067) or any single validator's number.
        composites = {_KEYPAIRS[0]: 0.40, _KEYPAIRS[1]: 0.82, _KEYPAIRS[2]: 0.90}
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)

        last: httpx.Response | None = None
        for i, (kp, comp) in enumerate(composites.items()):
            await _seed_ticket(session_maker, agent_id, keypair=kp)
            last = await client.post(
                f"/api/v1/validator/agent/{agent_id}/score",
                json=_score_payload(
                    agent_id, run_id=f"run_med_{i}", keypair=kp, composite=comp
                ),
            )
            assert last.status_code == 200, last.text
        assert last is not None
        assert last.json()["status"] == AgentStatus.SCORED

        # Public transparency record (the diagram's "which validators / all 3
        # scores + median"): all three validators, their exact composites +
        # signatures, and the median the platform finalized on.
        record = await client.get(f"/api/v1/public/agent/{agent_id}/scores")
        assert record.status_code == 200, record.text
        body = record.json()
        assert body["score_count"] == 3
        assert body["quorum"] == 3
        assert body["median_composite"] == pytest.approx(0.82)
        by_hotkey = {s["validator_hotkey"]: s["composite"] for s in body["scores"]}
        assert by_hotkey == {
            kp.ss58_address: pytest.approx(comp) for kp, comp in composites.items()
        }
        assert all(s["signature"] for s in body["scores"])

    async def test_expired_ticket_reopens_slot_for_a_new_validator(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app, extra_keypairs=(_DAVE,))

        # Three distinct validators claim the k=3 slots via the job endpoint.
        for kp in _KEYPAIRS:
            r = await client.post(
                "/api/v1/validator/job",
                headers={"X-Validator-Hotkey": kp.ss58_address},
            )
            assert r.status_code == 200, r.text
        # A fourth, never-assigned validator is shut out (pool full, not
        # already-mine): "no job for you".
        dave_hdr = {"X-Validator-Hotkey": _DAVE.ss58_address}
        assert (
            await client.post("/api/v1/validator/job", headers=dave_hdr)
        ).status_code == 204

        # One validator's ticket lapses past its deadline, re-opening its slot.
        async with session_maker() as s, s.begin():
            lapsed = await s.get(ValidatorTicket, (agent_id, _KEYPAIRS[0].ss58_address))
            assert lapsed is not None
            lapsed.deadline = datetime.now(UTC) - timedelta(minutes=1)

        # The fourth validator now picks up the re-opened slot.
        reopened = await client.post("/api/v1/validator/job", headers=dave_hdr)
        assert reopened.status_code == 200, reopened.text
        assert reopened.json()["agent_id"] == str(agent_id)

    async def test_expired_ticket_reopens_for_same_validator(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        keypair = _KEYPAIRS[0]
        headers = {"X-Validator-Hotkey": keypair.ss58_address}

        claimed = await client.post("/api/v1/validator/job", headers=headers)
        assert claimed.status_code == 200, claimed.text

        async with session_maker() as s, s.begin():
            lapsed = await s.get(ValidatorTicket, (agent_id, keypair.ss58_address))
            assert lapsed is not None
            lapsed.deadline = datetime.now(UTC) - timedelta(minutes=1)

        retried = await client.post("/api/v1/validator/job", headers=headers)
        assert retried.status_code == 200, retried.text
        assert retried.json()["agent_id"] == str(agent_id)


def test_dev_bypass_permit_refused_on_mainnet(monkeypatch) -> None:
    """The dev permit-bypass flag is honored off mainnet but refused on finney,
    so a stray env var can never open the validator surface on production."""
    from ditto.api_server.endpoints.validator import _dev_bypass_permit

    # Unset: never bypass, regardless of network.
    monkeypatch.delenv("DITTO_DEV_ALLOW_UNPERMITTED_VALIDATOR", raising=False)
    assert _dev_bypass_permit("finney") is False
    assert _dev_bypass_permit("ws://localhost:9944") is False

    # Set: honored on a dev/local network...
    monkeypatch.setenv("DITTO_DEV_ALLOW_UNPERMITTED_VALIDATOR", "true")
    assert _dev_bypass_permit("ws://localhost:9944") is True
    assert _dev_bypass_permit("test") is True
    # ...but refused on mainnet even when explicitly set.
    assert _dev_bypass_permit("finney") is False
    assert _dev_bypass_permit("Finney") is False
    assert _dev_bypass_permit("mainnet") is False
