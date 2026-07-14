"""Unit tests for :mod:`ditto.api_server.endpoints.screener`.

Exercise the real endpoints end to end against in-memory SQLite (real queries,
real status transitions) with chain + storage mocked. Signatures use a real
sr25519 dev keypair so the verification path runs for real.
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
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.api_models.system_health import (
    SystemMetrics,
    system_metrics_signing_token,
)
from ditto.api_server.datapipeline import DataPipelineError, NullGenerator
from ditto.api_server.dependencies import (
    get_chain_client,
    get_dataset_generator,
    get_session,
    get_storage_client,
)
from ditto.api_server.middleware.error_envelope import (
    ERROR_CODE_AGENT_NOT_FOUND,
    ERROR_CODE_AGENT_NOT_SCREENABLE,
    ERROR_CODE_SCREENER_AUTH,
    ERROR_CODE_VALIDATION,
)
from ditto.chain import ChainError
from ditto.chain.models import BlockInfo, NeuronInfo
from ditto.db.models import Agent, Base, ScreenerHeartbeat, ScreeningAttempt
from ditto_screening_protocol import verdict_signing_message

_KEYPAIR = bittensor.Keypair.create_from_uri("//Alice")
_SCREENER_HOTKEY = _KEYPAIR.ss58_address
_MINER_HOTKEY = "5DhaT8U7LVwnnJNUU8VL1XEipicatoaDVVq7cHo227gogVZm"
_SHA256 = "ab" * 32
# A fixed block the mocked chain returns for on-chain seed derivation.
_BLOCK = BlockInfo(number=4321, hash="0x" + "9f" * 32, timestamp=0)


def _sign(message: str | bytes) -> str:
    return _KEYPAIR.sign(
        message.encode() if isinstance(message, str) else message
    ).hex()


def _result_payload(
    agent_id: UUID,
    *,
    passed: bool = True,
    policy_version: int = SCREENING_POLICY_VERSION,
    **overrides: object,
) -> dict:
    attempt_id = overrides.get("attempt_id")
    signed = (
        verdict_signing_message(
            screener_hotkey=_SCREENER_HOTKEY,
            agent_id=agent_id,
            attempt_id=attempt_id,
            passed=passed,
            policy_version=policy_version,
        )
        if isinstance(attempt_id, UUID)
        else f"{_SCREENER_HOTKEY}:{agent_id}:{passed}:{policy_version}"
    )
    body = {
        "screener_hotkey": _SCREENER_HOTKEY,
        "signature": _sign(signed),
        "passed": passed,
        "policy_version": policy_version,
        "detail": "",
    }
    body.update(overrides)
    if isinstance(body.get("attempt_id"), UUID):
        body["attempt_id"] = str(body["attempt_id"])
    return body


def _heartbeat_payload(
    *,
    timestamp: int | None = None,
    state: str = "polling",
    active_agent_id: UUID | None = None,
    system_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    ts = timestamp if timestamp is not None else int(datetime.now(UTC).timestamp())
    metrics = (
        SystemMetrics.model_validate(system_metrics)
        if system_metrics is not None
        else None
    )
    message = (
        "ditto-screener-heartbeat:v1:"
        f"{_SCREENER_HOTKEY}:0.4.2:1:{SCREENING_POLICY_VERSION}:{state}:"
        f"{active_agent_id or ''}:{system_metrics_signing_token(metrics)}:{ts}"
    ).encode()
    payload: dict[str, object] = {
        "screener_hotkey": _SCREENER_HOTKEY,
        "software_version": "0.4.2",
        "protocol_version": 1,
        "policy_version": SCREENING_POLICY_VERSION,
        "state": state,
        "timestamp": ts,
        "signature": _sign(message),
    }
    if active_agent_id is not None:
        payload["active_agent_id"] = str(active_agent_id)
    if system_metrics is not None:
        payload["system_metrics"] = system_metrics
    return payload


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


class _FakeGenerator:
    """Test double for the dataset generator: pins a fixed hash, or raises."""

    def __init__(
        self, *, run_size: str = "full", sha: str = "ca" * 32, fail: bool = False
    ):
        self.run_size: str | None = run_size
        self._sha = sha
        self._fail = fail
        self.calls = 0

    async def generate(self, _seed: int) -> str:
        self.calls += 1
        if self._fail:
            raise DataPipelineError("generate service unavailable (test)")
        return self._sha

    async def aclose(self) -> None:
        return None


def _install_db(app: FastAPI, maker: async_sessionmaker[AsyncSession]) -> None:
    async def _session() -> AsyncIterator[AsyncSession]:
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = _session
    # Default: generation disabled (NullGenerator) so the existing verdict tests
    # promote without pinning a dataset. Tests that exercise the pinned path call
    # _install_generator afterward to override.
    app.dependency_overrides.setdefault(get_dataset_generator, lambda: NullGenerator())


def _install_generator(app: FastAPI, generator: object) -> None:
    app.dependency_overrides[get_dataset_generator] = lambda: generator


def _install_chain(
    app: FastAPI,
    *,
    permitted: bool = True,
    registered: bool = True,
    block: BlockInfo | None = _BLOCK,
    block_error: bool = False,
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
        if block_error:
            c.get_latest_block = AsyncMock(side_effect=ChainError("pylon down"))
        else:
            c.get_latest_block = AsyncMock(return_value=block)
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
    screening_policy_version: int | None = None,
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
                screening_policy_version=(
                    SCREENING_POLICY_VERSION
                    if screening_policy_version is None
                    and status == AgentStatus.EVALUATING
                    else (screening_policy_version or 0)
                ),
                created_at=created_at or datetime.now(UTC),
            )
        )
    return aid


_AUTH_HEADER = {
    "Authorization": "Bearer test-screener-token-at-least-32-characters",
    "X-Screener-Hotkey": _SCREENER_HOTKEY,
}
_CLAIM_URL = f"/api/v1/screener/claim?policy_version={SCREENING_POLICY_VERSION}"


@pytest.fixture(autouse=True)
def _authenticate_screener_client(client: httpx.AsyncClient) -> None:
    client.headers.update(_AUTH_HEADER)


# --- Queue -----------------------------------------------------------------


class TestHeartbeat:
    async def test_records_signed_metrics_and_is_publicly_visible(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        timestamp = int(datetime.now(UTC).timestamp())
        metrics = {
            "collected_at": timestamp,
            "cpu_percent": 20,
            "memory_percent": 35,
            "disk_percent": 50,
            "docker": {
                "status": "healthy",
                "running_containers": 3,
                "unhealthy_containers": 0,
            },
        }
        payload = _heartbeat_payload(timestamp=timestamp, system_metrics=metrics)
        response = await client.post("/api/v1/screener/heartbeat", json=payload)
        assert response.status_code == 200, response.text
        assert response.json()["accepted"] is True

        async with session_maker() as session:
            stored = await session.get(ScreenerHeartbeat, _SCREENER_HOTKEY)
            assert stored is not None
            assert stored.first_seen_at is not None
            assert stored.system_metrics is not None
            assert stored.system_metrics["docker"]["running_containers"] == 3

        public = (await client.get("/api/v1/public/screeners")).json()
        assert public["reported_count"] == 1
        entry = public["screeners"][0]
        assert entry["screener_hotkey"] == _SCREENER_HOTKEY
        assert entry["availability"] == "available"
        assert entry["health"] == "healthy"
        assert entry["system_metrics"]["docker_status"] == "healthy"
        assert "signature" not in entry

        replay = await client.post("/api/v1/screener/heartbeat", json=payload)
        assert replay.status_code == 200
        assert replay.json()["accepted"] is False

    async def test_rejects_tampering_arbitrary_metrics_and_wrong_auth(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        timestamp = int(datetime.now(UTC).timestamp())
        metrics = {
            "collected_at": timestamp,
            "cpu_percent": 20,
            "memory_percent": 35,
            "disk_percent": 50,
            "docker": {
                "status": "healthy",
                "running_containers": 3,
                "unhealthy_containers": 0,
            },
        }
        tampered = _heartbeat_payload(timestamp=timestamp, system_metrics=metrics)
        tampered["system_metrics"]["disk_percent"] = 90  # type: ignore[index]
        response = await client.post("/api/v1/screener/heartbeat", json=tampered)
        assert response.status_code == 401

        malformed = _heartbeat_payload(timestamp=timestamp, system_metrics=metrics)
        malformed["system_metrics"]["container_names"] = ["secret"]  # type: ignore[index]
        response = await client.post("/api/v1/screener/heartbeat", json=malformed)
        assert response.status_code == 422

        response = await client.post(
            "/api/v1/screener/heartbeat",
            headers={**_AUTH_HEADER, "Authorization": "Bearer wrong-token"},
            json=_heartbeat_payload(),
        )
        assert response.status_code == 401

    async def test_heartbeat_payload_size_is_bounded(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        response = await client.post(
            "/api/v1/screener/heartbeat",
            headers={"Content-Length": "4097"},
            json=_heartbeat_payload(),
        )
        assert response.status_code == 413

        payload = json.dumps(_heartbeat_payload())
        response = await client.post(
            "/api/v1/screener/heartbeat",
            headers={"Content-Type": "application/json"},
            content=(" " * 4097) + payload,
        )
        assert response.status_code == 413


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
        await _seed_agent(session_maker, status=AgentStatus.EVALUATING, name="promoted")
        _install_db(app, session_maker)
        _install_chain(app)

        response = await client.get("/api/v1/screener/queue", headers=_AUTH_HEADER)
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        body = response.json()
        assert body["count"] == 2
        assert [i["name"] for i in body["items"]] == ["older", "younger"]
        assert all(i["status"] == AgentStatus.UPLOADED for i in body["items"])
        assert body["required_policy_version"] == SCREENING_POLICY_VERSION

    async def test_requeues_legacy_evaluating_submission(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            screening_policy_version=0,
        )
        _install_db(app, session_maker)
        response = await client.get("/api/v1/screener/queue")
        assert response.status_code == 200
        assert response.json()["count"] == 1

    async def test_requeues_retryable_failures_regardless_of_policy(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        stale_id = await _seed_agent(
            session_maker,
            status=AgentStatus.SCREENING_FAILED,
            screening_policy_version=SCREENING_POLICY_VERSION - 1,
        )
        current_id = await _seed_agent(
            session_maker,
            status=AgentStatus.SCREENING_FAILED,
            screening_policy_version=SCREENING_POLICY_VERSION,
        )
        _install_db(app, session_maker)

        response = await client.get("/api/v1/screener/queue")

        assert response.status_code == 200
        assert {item["agent_id"] for item in response.json()["items"]} == {
            str(stale_id),
            str(current_id),
        }

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
        client.headers.clear()
        response = await client.get("/api/v1/screener/queue")
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_SCREENER_AUTH

    async def test_invalid_bearer_token_returns_401(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        response = await client.get(
            "/api/v1/screener/queue",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_SCREENER_AUTH

    async def test_unapproved_hotkey_returns_401(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        response = await client.get(
            "/api/v1/screener/queue",
            headers={
                "X-Screener-Hotkey": (
                    "5DhaT8U7LVwnnJNUU8VL1XEipicatoaDVVq7cHo227gogVZm"
                )
            },
        )
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_SCREENER_AUTH

    async def test_dedicated_screener_needs_no_validator_permit(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app, permitted=False, registered=False)
        response = await client.get("/api/v1/screener/queue")
        assert response.status_code == 200

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


# --- Leased claims ---------------------------------------------------------


class TestClaim:
    async def test_claim_is_exclusive_and_lease_bound_verdict_is_idempotent(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)

        claimed = await client.post(_CLAIM_URL, headers=_AUTH_HEADER)
        assert claimed.status_code == 200
        item = claimed.json()["items"][0]
        assert item["agent_id"] == str(agent_id)
        assert item["status"] == AgentStatus.SCREENING
        assert item["attempt_id"]
        assert item["lease_deadline"]

        duplicate = await client.post(_CLAIM_URL, headers=_AUTH_HEADER)
        assert duplicate.status_code == 200
        assert duplicate.json()["count"] == 0

        payload = _result_payload(
            agent_id,
            passed=True,
            attempt_id=UUID(item["attempt_id"]),
        )
        first = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result",
            headers=_AUTH_HEADER,
            json=payload,
        )
        replay = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result",
            headers=_AUTH_HEADER,
            json=payload,
        )
        assert first.status_code == 200
        assert replay.status_code == 200
        assert replay.json()["status"] == AgentStatus.EVALUATING

    async def test_policy_mismatch_does_not_create_lease(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)

        mismatch = await client.post(
            f"/api/v1/screener/claim?policy_version={SCREENING_POLICY_VERSION - 1}",
            headers=_AUTH_HEADER,
        )

        assert mismatch.status_code == 409
        assert mismatch.json()["error_code"] == ERROR_CODE_AGENT_NOT_SCREENABLE
        async with session_maker() as session:
            agent = await session.get(Agent, agent_id)
            assert agent is not None
            assert agent.status == AgentStatus.UPLOADED
            attempts = (await session.scalars(select(ScreeningAttempt))).all()
            assert attempts == []

    async def test_expired_lease_rejects_late_verdict(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)
        claimed = await client.post(_CLAIM_URL, headers=_AUTH_HEADER)
        attempt_id = UUID(claimed.json()["items"][0]["attempt_id"])
        async with session_maker() as session, session.begin():
            attempt = await session.get(ScreeningAttempt, attempt_id)
            assert attempt is not None
            attempt.started_at = datetime.now(UTC) - timedelta(minutes=2)
            attempt.deadline = datetime.now(UTC) - timedelta(minutes=1)

        response = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result",
            headers=_AUTH_HEADER,
            json=_result_payload(agent_id, attempt_id=attempt_id),
        )

        assert response.status_code == 409
        assert response.json()["error_code"] == ERROR_CODE_AGENT_NOT_SCREENABLE

    async def test_attempt_cannot_be_replayed_for_another_agent(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        claimed_agent = await _seed_agent(
            session_maker,
            status=AgentStatus.UPLOADED,
            created_at=datetime.now(UTC) - timedelta(minutes=1),
        )
        other_agent = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)
        claimed = await client.post(_CLAIM_URL, headers=_AUTH_HEADER)
        item = next(
            row
            for row in claimed.json()["items"]
            if row["agent_id"] == str(claimed_agent)
        )
        attempt_id = UUID(item["attempt_id"])

        response = await client.post(
            f"/api/v1/screener/agent/{other_agent}/result",
            headers=_AUTH_HEADER,
            json=_result_payload(other_agent, attempt_id=attempt_id),
        )

        assert response.status_code == 409
        assert response.json()["error_code"] == ERROR_CODE_AGENT_NOT_SCREENABLE

    async def test_attempt_rejects_wrong_policy_even_for_signed_failure(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)
        claimed = await client.post(_CLAIM_URL, headers=_AUTH_HEADER)
        attempt_id = UUID(claimed.json()["items"][0]["attempt_id"])

        response = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result",
            headers=_AUTH_HEADER,
            json=_result_payload(
                agent_id,
                attempt_id=attempt_id,
                passed=False,
                policy_version=SCREENING_POLICY_VERSION - 1,
            ),
        )

        assert response.status_code == 409
        assert response.json()["error_code"] == ERROR_CODE_AGENT_NOT_SCREENABLE


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
    async def test_legacy_pass_cannot_promote(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)
        payload = _result_payload(agent_id, policy_version=1)
        payload["signature"] = _sign(f"{_SCREENER_HOTKEY}:{agent_id}:True")
        response = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result", json=payload
        )
        assert response.status_code == 409

    async def test_v2_pass_rescreens_in_place_and_preserves_dataset(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            screening_policy_version=0,
        )
        async with session_maker() as s, s.begin():
            agent = await s.get(Agent, agent_id)
            assert agent is not None
            agent.dataset_seed = 42
            agent.dataset_sha256 = "cd" * 32
            agent.dataset_run_size = "full"
        _install_db(app, session_maker)
        _install_chain(app)
        response = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result",
            json=_result_payload(agent_id),
        )
        assert response.status_code == 200
        async with session_maker() as s:
            agent = await s.get(Agent, agent_id)
            assert agent is not None
            assert agent.status == AgentStatus.EVALUATING
            assert agent.screening_policy_version == SCREENING_POLICY_VERSION
            assert agent.dataset_seed == 42

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

    async def test_deterministic_fail_moves_to_rejected(
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
            json=_result_payload(
                agent_id,
                passed=False,
                detail="build failed: cargo error SECRET_FROM_BUILD",
            ),
        )
        assert response.status_code == 200
        assert response.json()["status"] == AgentStatus.REJECTED

        async with session_maker() as s:
            agent = await s.get(Agent, agent_id)
            assert agent is not None
            assert agent.screening_reason == "Docker image build failed"
            assert agent.screening_policy_version == SCREENING_POLICY_VERSION
            assert "SECRET_FROM_BUILD" not in agent.screening_reason

    async def test_current_pass_recovers_stale_screening_failure(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(
            session_maker,
            status=AgentStatus.SCREENING_FAILED,
            screening_policy_version=SCREENING_POLICY_VERSION - 1,
        )
        _install_db(app, session_maker)
        _install_chain(app)

        response = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result",
            json=_result_payload(agent_id, passed=True),
        )

        assert response.status_code == 200
        assert response.json()["status"] == AgentStatus.EVALUATING
        async with session_maker() as s:
            agent = await s.get(Agent, agent_id)
            assert agent is not None
            assert agent.screening_policy_version == SCREENING_POLICY_VERSION

    async def test_infrastructure_failure_is_retryable_not_rejected(
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
            json=_result_payload(
                agent_id,
                passed=False,
                detail="screener error: Docker daemon unavailable SECRET",
            ),
        )
        assert response.status_code == 200
        assert response.json()["status"] == AgentStatus.SCREENING_FAILED

        retry = await client.post(_CLAIM_URL, headers=_AUTH_HEADER)
        assert retry.status_code == 200
        assert retry.json()["items"][0]["agent_id"] == str(agent_id)

    async def test_model_canary_failure_has_public_safe_reason(
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
            json=_result_payload(
                agent_id,
                passed=False,
                detail="model canary observed no model call",
            ),
        )
        assert response.status_code == 200
        async with session_maker() as session:
            agent = await session.get(Agent, agent_id)
            assert agent is not None
            assert (
                agent.screening_reason
                == "Harness did not use the validator model gateway"
            )

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

    async def test_pass_pins_dataset_when_enabled(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)
        gen = _FakeGenerator(run_size="full", sha="be" * 32)
        _install_generator(app, gen)

        response = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result",
            json=_result_payload(agent_id, passed=True),
        )
        assert response.status_code == 200
        assert response.json()["status"] == AgentStatus.EVALUATING
        assert gen.calls == 1

        async with session_maker() as s:
            agent = await s.get(Agent, agent_id)
            assert agent is not None
            assert agent.status == AgentStatus.EVALUATING
            assert agent.dataset_seed is not None and agent.dataset_seed >= 0
            assert agent.dataset_sha256 == "be" * 32
            assert agent.dataset_run_size == "full"
            # The seed is derived from the on-chain block and pinned with its
            # provenance, so anyone can recompute + verify it.
            from ditto.api_server.onchain_seed import derive_seed

            assert agent.dataset_seed_block == _BLOCK.number
            assert agent.dataset_seed_block_hash == _BLOCK.hash
            assert agent.dataset_seed == derive_seed(_BLOCK.hash, agent_id)

    async def test_seed_falls_back_when_chain_unavailable(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # A chain outage must not halt submissions: the seed falls back to a local
        # CSPRNG value, with null block provenance flagging it as not chain-derived.
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app, block_error=True)
        _install_generator(app, _FakeGenerator(run_size="full", sha="be" * 32))

        response = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result",
            json=_result_payload(agent_id, passed=True),
        )
        assert response.status_code == 200
        async with session_maker() as s:
            agent = await s.get(Agent, agent_id)
            assert agent is not None
            assert agent.status == AgentStatus.EVALUATING
            assert agent.dataset_seed is not None and agent.dataset_seed >= 0
            # Fallback provenance: no block reference.
            assert agent.dataset_seed_block is None
            assert agent.dataset_seed_block_hash is None

    async def test_generation_failure_does_not_promote(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)
        _install_generator(app, _FakeGenerator(fail=True))

        response = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result",
            json=_result_payload(agent_id, passed=True),
        )
        # Required dataset failed to generate: the verdict must NOT have promoted
        # the agent (it can be retried).
        assert response.status_code == 500
        async with session_maker() as s:
            agent = await s.get(Agent, agent_id)
            assert agent is not None
            assert agent.status == AgentStatus.UPLOADED
            assert agent.dataset_seed is None

    async def test_idempotent_repeat_does_not_regenerate(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)
        gen = _FakeGenerator(sha="ab" * 32)
        _install_generator(app, gen)

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
        # The dataset was pinned once; the re-report did not call the generator
        # again (the pre-read guard sees dataset_seed already set).
        assert gen.calls == 1
        async with session_maker() as s:
            agent = await s.get(Agent, agent_id)
            assert agent is not None
            assert agent.dataset_sha256 == "ab" * 32

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

    async def test_flipped_verdict_rejected(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # A pass signed by the screener must not be replayable as a fail: the
        # signature binds the ``passed`` flag, so flipping it 401s.
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)
        payload = _result_payload(agent_id, passed=True)
        payload["passed"] = False  # grief attempt: replay the pass sig as a fail
        response = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result", json=payload
        )
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_SCREENER_AUTH

    async def test_payload_hotkey_must_match_authenticated_hotkey(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)
        other = "5DhaT8U7LVwnnJNUU8VL1XEipicatoaDVVq7cHo227gogVZm"
        response = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result",
            json=_result_payload(agent_id, screener_hotkey=other),
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
