"""Unit tests for :mod:`ditto.api_server.endpoints.screener`.

Exercise the real endpoints end to end against in-memory SQLite (real queries,
real status transitions) with chain + storage mocked. Signatures use a real
sr25519 dev keypair so the verification path runs for real.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import replace
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
from ditto.api_models.screener import (
    SCREENING_POLICY_VERSION,
    ScreenerHeartbeatRequest,
    SourceReviewEvidenceItem,
    SourceReviewFinding,
)
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
from ditto.api_server.endpoints.screener import _heartbeat_signing_message
from ditto.api_server.middleware.error_envelope import (
    ERROR_CODE_AGENT_NOT_FOUND,
    ERROR_CODE_AGENT_NOT_SCREENABLE,
    ERROR_CODE_SCREENER_AUTH,
    ERROR_CODE_VALIDATION,
)
from ditto.chain import ChainError
from ditto.chain.models import BlockInfo, NeuronInfo
from ditto.db.models import (
    Agent,
    Base,
    Score,
    ScreenerHeartbeat,
    ScreeningAttempt,
    ScreeningQuarantine,
)
from ditto_screening_protocol import ScreenResultOutcome, verdict_signing_message

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
    outcome_raw = overrides.get("outcome")
    outcome = ScreenResultOutcome(outcome_raw) if isinstance(outcome_raw, str) else None
    signed = (
        verdict_signing_message(
            screener_hotkey=_SCREENER_HOTKEY,
            agent_id=agent_id,
            attempt_id=attempt_id,
            passed=passed,
            policy_version=policy_version,
            outcome=outcome,
            manifest_digest=overrides.get("manifest_digest")
            if isinstance(overrides.get("manifest_digest"), str)
            else None,
            finding_digest=overrides.get("finding_digest")
            if isinstance(overrides.get("finding_digest"), str)
            else None,
            reason_code=overrides.get("reason_code")
            if isinstance(overrides.get("reason_code"), str)
            else None,
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
    protocol_version: int = 1,
    progress: dict[str, object] | None = None,
    system_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    ts = timestamp if timestamp is not None else int(datetime.now(UTC).timestamp())
    metrics = (
        SystemMetrics.model_validate(system_metrics)
        if system_metrics is not None
        else None
    )
    if protocol_version == 1:
        message = (
            "ditto-screener-heartbeat:v1:"
            f"{_SCREENER_HOTKEY}:0.4.2:1:{SCREENING_POLICY_VERSION}:{state}:"
            f"{active_agent_id or ''}:{system_metrics_signing_token(metrics)}:{ts}"
        ).encode()
    else:
        progress_token = (
            f"{progress['stage']},{progress['started_at']}" if progress else "-"
        )
        message = (
            "ditto-screener-heartbeat:v2:"
            f"{_SCREENER_HOTKEY}:0.4.2:{protocol_version}:"
            f"{SCREENING_POLICY_VERSION}:{state}:{active_agent_id or ''}:"
            f"{progress_token}:{system_metrics_signing_token(metrics)}:{ts}"
        ).encode()
    payload: dict[str, object] = {
        "screener_hotkey": _SCREENER_HOTKEY,
        "software_version": "0.4.2",
        "protocol_version": protocol_version,
        "policy_version": SCREENING_POLICY_VERSION,
        "state": state,
        "timestamp": ts,
        "signature": _sign(message),
    }
    if active_agent_id is not None:
        payload["active_agent_id"] = str(active_agent_id)
    if progress is not None:
        payload["progress"] = progress
    if system_metrics is not None:
        payload["system_metrics"] = system_metrics
    return payload


@pytest.mark.parametrize(
    "stage",
    [
        "preparing",
        "downloading",
        "validating",
        "building",
        "starting",
        "health_check",
        "submitting",
    ],
)
def test_v2_canonical_signing_matches_screener_contract(stage: str) -> None:
    payload = _heartbeat_payload(
        timestamp=456,
        state="screening",
        active_agent_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
        protocol_version=2,
        progress={"stage": stage, "started_at": 400},
    )
    request = ScreenerHeartbeatRequest.model_validate(payload)
    assert (
        _heartbeat_signing_message(request)
        == (
            "ditto-screener-heartbeat:v2:"
            f"{_SCREENER_HOTKEY}:0.4.2:2:{SCREENING_POLICY_VERSION}:screening:"
            f"550e8400-e29b-41d4-a716-446655440000:{stage},400:-:456"
        ).encode()
    )


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
    miner_hotkey: str = _MINER_HOTKEY,
    sha256: str = _SHA256,
) -> UUID:
    aid = agent_id or uuid4()
    async with maker() as s, s.begin():
        s.add(
            Agent(
                agent_id=aid,
                miner_hotkey=miner_hotkey,
                name=name,
                sha256=sha256,
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


async def _seed_score(
    maker: async_sessionmaker[AsyncSession], *, agent_id: UUID
) -> None:
    async with maker() as session, session.begin():
        session.add(
            Score(
                agent_id=agent_id,
                validator_hotkey="5ScoreValidatorHotkeyXXXXXXXXXXXXXXXXXXXXXXXXXX",
                run_id=str(uuid4()),
                signature=None,
                seed=1,
                composite=0.5,
                tool_mean=0.5,
                memory_mean=0.5,
                median_ms=100,
                n=1,
                details=None,
                generated_at=datetime.now(UTC),
            )
        )


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
    async def test_v2_progress_is_public_and_clears_on_idle_and_terminal(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        started = datetime.now(UTC).replace(microsecond=0) - timedelta(minutes=2)
        agent_id = await _seed_agent(
            session_maker, status=AgentStatus.SCREENING, name="steady-agent"
        )
        attempt_id = uuid4()
        async with session_maker() as session, session.begin():
            session.add(
                ScreeningAttempt(
                    attempt_id=attempt_id,
                    agent_id=agent_id,
                    screener_hotkey=_SCREENER_HOTKEY,
                    policy_version=SCREENING_POLICY_VERSION,
                    status="running",
                    started_at=started,
                    deadline=started + timedelta(minutes=30),
                )
            )

        timestamp = int(datetime.now(UTC).timestamp())
        progress = {"stage": "building", "started_at": int(started.timestamp())}
        response = await client.post(
            "/api/v1/screener/heartbeat",
            json=_heartbeat_payload(
                timestamp=timestamp,
                state="screening",
                active_agent_id=agent_id,
                protocol_version=2,
                progress=progress,
            ),
        )
        assert response.status_code == 200, response.text
        entry = (await client.get("/api/v1/public/screeners")).json()["screeners"][0]
        assert entry["active_agent_id"] == str(agent_id)
        assert entry["active_agent_name"] == "steady-agent"
        assert entry["screening_progress"]["stage"] == "building"
        assert entry["screening_progress"]["started_at"].startswith(
            started.isoformat().replace("+00:00", "")
        )

        review = await client.post(
            "/api/v1/screener/heartbeat",
            json=_heartbeat_payload(
                timestamp=timestamp + 1,
                state="screening",
                active_agent_id=agent_id,
                protocol_version=2,
                progress={
                    "stage": "source_review_30",
                    "started_at": int(started.timestamp()),
                },
            ),
        )
        assert review.status_code == 200, review.text
        review_entry = (await client.get("/api/v1/public/screeners")).json()[
            "screeners"
        ][0]
        assert review_entry["screening_progress"]["stage"] == "source_review_30"

        idle = await client.post(
            "/api/v1/screener/heartbeat",
            json=_heartbeat_payload(timestamp=timestamp + 2, protocol_version=2),
        )
        assert idle.status_code == 200
        idle_entry = (await client.get("/api/v1/public/screeners")).json()["screeners"][
            0
        ]
        assert idle_entry["active_agent_id"] is None
        assert idle_entry["active_agent_name"] is None
        assert idle_entry["screening_progress"] is None

        legacy = await client.post(
            "/api/v1/screener/heartbeat",
            json=_heartbeat_payload(
                timestamp=timestamp + 3,
                state="screening",
                active_agent_id=agent_id,
                protocol_version=1,
            ),
        )
        assert legacy.status_code == 200
        legacy_entry = (await client.get("/api/v1/public/screeners")).json()[
            "screeners"
        ][0]
        assert legacy_entry["active_agent_name"] == "steady-agent"
        assert legacy_entry["screening_progress"] is None

        active = await client.post(
            "/api/v1/screener/heartbeat",
            json=_heartbeat_payload(
                timestamp=timestamp + 4,
                state="screening",
                active_agent_id=agent_id,
                protocol_version=2,
                progress=progress,
            ),
        )
        assert active.status_code == 200
        async with session_maker() as session, session.begin():
            attempt = await session.get(ScreeningAttempt, attempt_id)
            agent = await session.get(Agent, agent_id)
            assert attempt is not None and agent is not None
            attempt.status = "passed"
            attempt.finished_at = datetime.now(UTC)
            agent.status = AgentStatus.EVALUATING
        terminal_entry = (await client.get("/api/v1/public/screeners")).json()[
            "screeners"
        ][0]
        assert terminal_entry["active_agent_id"] is None
        assert terminal_entry["screening_progress"] is None

    async def test_stale_progress_is_offline_and_not_projected(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        started = datetime.now(UTC).replace(microsecond=0) - timedelta(minutes=2)
        agent_id = await _seed_agent(session_maker, status=AgentStatus.SCREENING)
        async with session_maker() as session, session.begin():
            session.add(
                ScreeningAttempt(
                    attempt_id=uuid4(),
                    agent_id=agent_id,
                    screener_hotkey=_SCREENER_HOTKEY,
                    policy_version=SCREENING_POLICY_VERSION,
                    status="running",
                    started_at=started,
                    deadline=started + timedelta(minutes=30),
                )
            )
        timestamp = int(datetime.now(UTC).timestamp())
        response = await client.post(
            "/api/v1/screener/heartbeat",
            json=_heartbeat_payload(
                timestamp=timestamp,
                state="screening",
                active_agent_id=agent_id,
                protocol_version=2,
                progress={
                    "stage": "health_check",
                    "started_at": int(started.timestamp()),
                },
            ),
        )
        assert response.status_code == 200
        async with session_maker() as session, session.begin():
            heartbeat = await session.get(ScreenerHeartbeat, _SCREENER_HOTKEY)
            assert heartbeat is not None
            heartbeat.seen_at = datetime.now(UTC) - timedelta(minutes=10)
        entry = (await client.get("/api/v1/public/screeners")).json()["screeners"][0]
        assert entry["online"] is False
        assert entry["active_agent_id"] is None
        assert entry["screening_progress"] is None

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

        now = int(datetime.now(UTC).timestamp())
        progress = {"stage": "building", "started_at": now - 30}
        tampered_progress = _heartbeat_payload(
            timestamp=now,
            state="screening",
            active_agent_id=uuid4(),
            protocol_version=2,
            progress=progress,
        )
        tampered_progress["progress"]["stage"] = "submitting"  # type: ignore[index]
        response = await client.post(
            "/api/v1/screener/heartbeat", json=tampered_progress
        )
        assert response.status_code == 401

        private_field = _heartbeat_payload(
            timestamp=now,
            state="screening",
            active_agent_id=uuid4(),
            protocol_version=2,
            progress={"stage": "building", "started_at": now - 30},
        )
        private_field["progress"]["dependency"] = "private-package"  # type: ignore[index]
        response = await client.post("/api/v1/screener/heartbeat", json=private_field)
        assert response.status_code == 422

        invalid_stage = _heartbeat_payload(
            timestamp=now,
            state="screening",
            active_agent_id=uuid4(),
            protocol_version=2,
            progress={"stage": "docker_layer", "started_at": now - 30},
        )
        response = await client.post("/api/v1/screener/heartbeat", json=invalid_stage)
        assert response.status_code == 422

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

    async def test_prioritizes_zero_score_submission_before_older_scored_one(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        base = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        scored = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            name="older-scored",
            created_at=base,
            screening_policy_version=SCREENING_POLICY_VERSION - 1,
        )
        await _seed_score(session_maker, agent_id=scored)
        await _seed_agent(
            session_maker,
            status=AgentStatus.UPLOADED,
            name="younger-unscored",
            created_at=base + timedelta(minutes=5),
        )
        _install_db(app, session_maker)

        response = await client.get("/api/v1/screener/queue")

        assert response.status_code == 200
        assert [item["name"] for item in response.json()["items"]] == [
            "younger-unscored",
            "older-scored",
        ]

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
    async def test_claim_prioritizes_zero_score_submission(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        base = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        scored = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            name="older-scored",
            created_at=base,
            screening_policy_version=SCREENING_POLICY_VERSION - 1,
        )
        await _seed_score(session_maker, agent_id=scored)
        unscored = await _seed_agent(
            session_maker,
            status=AgentStatus.UPLOADED,
            name="younger-unscored",
            created_at=base + timedelta(minutes=5),
        )
        _install_db(app, session_maker)

        response = await client.post(_CLAIM_URL)

        assert response.status_code == 200
        assert response.json()["items"][0]["agent_id"] == str(unscored)

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

    async def test_exact_duplicate_waits_for_usable_owner_then_rejects_before_screen(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        """Three hotkeys share one hash: a failed build claims nothing durable.

        Concurrent claims cannot admit both later uploads. The first later upload
        must pass the build gate before the other receives an exact-duplicate
        precheck. Replaying that signed rejection remains idempotent.
        """
        now = datetime.now(UTC)
        first = await _seed_agent(
            session_maker,
            status=AgentStatus.UPLOADED,
            miner_hotkey="5FirstMinerHotkeyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            created_at=now - timedelta(minutes=3),
        )
        _install_db(app, session_maker)
        _install_chain(app)

        first_claim = (await client.post(_CLAIM_URL)).json()["items"][0]
        first_failure = await client.post(
            f"/api/v1/screener/agent/{first}/result",
            json=_result_payload(
                first,
                passed=False,
                attempt_id=UUID(first_claim["attempt_id"]),
                outcome="deterministic_reject",
                detail="build failed: synthetic compiler error",
                reason_code="docker-build",
            ),
        )
        assert first_failure.status_code == 200

        second = await _seed_agent(
            session_maker,
            status=AgentStatus.UPLOADED,
            miner_hotkey="5SecondMinerHotkeyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            created_at=now - timedelta(minutes=2),
        )
        third = await _seed_agent(
            session_maker,
            status=AgentStatus.UPLOADED,
            miner_hotkey="5ThirdMinerHotkeyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            created_at=now - timedelta(minutes=1),
        )

        simultaneous = await asyncio.gather(
            client.post(_CLAIM_URL), client.post(_CLAIM_URL)
        )
        admitted = [
            item for response in simultaneous for item in response.json()["items"]
        ]
        assert [item["agent_id"] for item in admitted] == [str(second)]
        assert admitted[0]["precheck_reason_code"] is None

        second_pass = await client.post(
            f"/api/v1/screener/agent/{second}/result",
            json=_result_payload(
                second,
                attempt_id=UUID(admitted[0]["attempt_id"]),
                outcome="pass",
            ),
        )
        assert second_pass.status_code == 200

        duplicate_claim = (await client.post(_CLAIM_URL)).json()["items"][0]
        assert duplicate_claim["agent_id"] == str(third)
        assert duplicate_claim["precheck_reason_code"] == (
            "exact-cross-miner-duplicate"
        )
        assert duplicate_claim["duplicate_of"] == str(second)
        conflicting_pass = await client.post(
            f"/api/v1/screener/agent/{third}/result",
            json=_result_payload(
                third,
                attempt_id=UUID(duplicate_claim["attempt_id"]),
                outcome="pass",
            ),
        )
        assert conflicting_pass.status_code == 409
        duplicate_payload = _result_payload(
            third,
            passed=False,
            attempt_id=UUID(duplicate_claim["attempt_id"]),
            outcome="deterministic_reject",
            detail="exact cross-miner duplicate",
            reason_code="exact-cross-miner-duplicate",
        )
        rejected = await client.post(
            f"/api/v1/screener/agent/{third}/result", json=duplicate_payload
        )
        replay = await client.post(
            f"/api/v1/screener/agent/{third}/result", json=duplicate_payload
        )
        assert rejected.status_code == replay.status_code == 200
        assert replay.json()["status"] == AgentStatus.REJECTED

        async with session_maker() as session:
            failed = await session.get(Agent, first)
            owner = await session.get(Agent, second)
            duplicate = await session.get(Agent, third)
            attempt = await session.get(
                ScreeningAttempt, UUID(duplicate_claim["attempt_id"])
            )
            assert failed is not None and failed.status == AgentStatus.REJECTED
            assert owner is not None and owner.status == AgentStatus.EVALUATING
            assert duplicate is not None and duplicate.duplicate_of == second
            assert duplicate.screening_reason_code == "exact-cross-miner-duplicate"
            assert duplicate.screening_reason == (
                "Artifact is an exact duplicate of another miner submission"
            )
            assert attempt is not None
            assert attempt.reason_code == "exact-cross-miner-duplicate"
            assert attempt.duplicate_of == second

        status = await client.get(f"/api/v1/retrieval/agent/{third}/status")
        assert status.json()["screening_reason_code"] == ("exact-cross-miner-duplicate")

    async def test_same_miner_exact_hash_retry_is_not_prechecked_as_duplicate(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        retry = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)

        claimed = (await client.post(_CLAIM_URL)).json()["items"][0]

        assert claimed["agent_id"] == str(retry)
        assert claimed["precheck_reason_code"] is None
        assert claimed["duplicate_of"] is None

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

    async def test_attempt_bound_quarantine_is_durable_and_idempotent(
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
        payload = _result_payload(
            agent_id,
            passed=False,
            attempt_id=attempt_id,
            outcome="quarantine",
            manifest_digest="12" * 32,
            finding_digest="34" * 32,
            reason_code="agentic-source-review-tripwire",
        )
        first = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result", json=payload
        )
        replay = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result", json=payload
        )
        assert first.status_code == replay.status_code == 200
        assert replay.json()["status"] == AgentStatus.QUARANTINED
        async with session_maker() as session:
            attempt = await session.get(ScreeningAttempt, attempt_id)
            quarantines = (await session.scalars(select(ScreeningQuarantine))).all()
            assert attempt is not None and attempt.status == "quarantined"
            assert len(quarantines) == 1


class TestQuarantineAdmin:
    @pytest.mark.parametrize(
        ("resolution", "expected_status", "expected_reason"),
        [
            (
                "release",
                AgentStatus.EVALUATING,
                "Remove the bundled credential and resubmit",
            ),
            (
                "rescreen",
                AgentStatus.SCREENING_FAILED,
                "Remove the bundled credential and resubmit",
            ),
            (
                "reject",
                AgentStatus.REJECTED,
                "Remove the bundled credential and resubmit",
            ),
        ],
    )
    async def test_list_resolution_reason_and_conflicting_second_resolution(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
        resolution: str,
        expected_status: AgentStatus,
        expected_reason: str | None,
    ) -> None:
        app.state.config = replace(
            app.state.config,
            admin_api_token="test-admin-token-at-least-32-characters",
        )
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)
        claimed = await client.post(_CLAIM_URL)
        attempt_id = UUID(claimed.json()["items"][0]["attempt_id"])
        quarantine_payload = _result_payload(
            agent_id,
            passed=False,
            attempt_id=attempt_id,
            outcome="quarantine",
            manifest_digest="56" * 32,
            finding_digest="78" * 32,
            reason_code="agentic-source-review-tripwire",
        )
        held = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result", json=quarantine_payload
        )
        assert held.status_code == 200

        admin_headers = {
            "Authorization": "Bearer test-admin-token-at-least-32-characters",
            "X-Admin-Actor": "backroom:test-user",
        }
        listing = await client.get(
            "/api/v1/admin/screening-quarantines", headers=admin_headers
        )
        assert listing.status_code == 200
        item = listing.json()["items"][0]
        assert item["agent_id"] == str(agent_id)
        assert item["reason_code"] == "agentic-source-review-tripwire"
        assert "source" not in item

        resolved = await client.post(
            f"/api/v1/admin/screening-quarantines/{item['quarantine_id']}/resolve",
            headers=admin_headers,
            json={
                "resolution": resolution,
                "reason": "Remove the bundled credential and resubmit",
            },
        )
        conflict = await client.post(
            f"/api/v1/admin/screening-quarantines/{item['quarantine_id']}/resolve",
            headers=admin_headers,
            json={"resolution": "reject", "reason": "Conflicting action"},
        )
        assert resolved.status_code == 200
        assert resolved.json()["agent_status"] == expected_status
        assert conflict.status_code == 409
        async with session_maker() as session:
            agent = await session.get(Agent, agent_id)
            assert agent is not None
            assert agent.screening_reason == expected_reason

    async def test_admin_auth_is_required(
        self, app: FastAPI, client: httpx.AsyncClient
    ) -> None:
        app.state.config = replace(
            app.state.config,
            admin_api_token="test-admin-token-at-least-32-characters",
        )
        response = await client.get(
            "/api/v1/admin/screening-quarantines",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401

    async def test_lists_all_screening_outcomes_and_issues_audited_artifact_url(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        app.state.config = replace(
            app.state.config,
            admin_api_token="test-admin-token-at-least-32-characters",
        )
        agent_id = await _seed_agent(session_maker, status=AgentStatus.REJECTED)
        now = datetime.now(UTC)
        async with session_maker() as session, session.begin():
            session.add(
                ScreeningAttempt(
                    attempt_id=uuid4(),
                    agent_id=agent_id,
                    screener_hotkey=_SCREENER_HOTKEY,
                    policy_version=SCREENING_POLICY_VERSION,
                    status="rejected",
                    started_at=now - timedelta(minutes=2),
                    deadline=now + timedelta(minutes=28),
                    finished_at=now,
                    public_reason="Docker image build failed",
                )
            )
        _install_db(app, session_maker)
        storage = _install_storage(app)
        headers = {
            "Authorization": "Bearer test-admin-token-at-least-32-characters",
            "X-Admin-Actor": "backroom:test-user",
        }

        listing = await client.get(
            "/api/v1/admin/screening-submissions", headers=headers
        )
        artifact = await client.get(
            f"/api/v1/admin/screening-submissions/{agent_id}/artifact",
            headers=headers,
        )

        assert listing.status_code == 200
        item = listing.json()["items"][0]
        assert item["agent_id"] == str(agent_id)
        assert item["attempts"][0]["status"] == "rejected"
        assert item["attempts"][0]["reason"] == "Docker image build failed"
        assert artifact.status_code == 200
        assert artifact.json()["sha256"] == _SHA256
        assert storage.presigned_get_url.await_args.kwargs == {
            "key": f"{agent_id}/agent.tar.gz",
            "expires_in": 300,
        }

    async def test_rejected_rescreen_preserves_score_and_attempt_history(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        app.state.config = replace(
            app.state.config,
            admin_api_token="test-admin-token-at-least-32-characters",
        )
        agent_id = await _seed_agent(
            session_maker,
            status=AgentStatus.REJECTED,
            screening_policy_version=SCREENING_POLICY_VERSION,
        )
        await _seed_score(session_maker, agent_id=agent_id)
        attempt_id = uuid4()
        now = datetime.now(UTC)
        async with session_maker() as session, session.begin():
            session.add(
                ScreeningAttempt(
                    attempt_id=attempt_id,
                    agent_id=agent_id,
                    screener_hotkey=_SCREENER_HOTKEY,
                    policy_version=SCREENING_POLICY_VERSION,
                    status="rejected",
                    started_at=now - timedelta(minutes=2),
                    deadline=now + timedelta(minutes=28),
                    finished_at=now,
                    public_reason="Docker image build failed",
                )
            )
        _install_db(app, session_maker)
        response = await client.post(
            f"/api/v1/admin/screening-submissions/{agent_id}/rescreen",
            headers={
                "Authorization": "Bearer test-admin-token-at-least-32-characters",
                "X-Admin-Actor": "backroom:test-user",
            },
            json={
                "reason": "Build was interrupted by a worker deployment",
                "expected_sha256": _SHA256,
                "expected_score_count": 1,
            },
        )
        assert response.status_code == 200
        assert response.json()["agent_status"] == AgentStatus.SCREENING_FAILED
        async with session_maker() as session:
            agent = await session.get(Agent, agent_id)
            attempts = list(
                await session.scalars(
                    select(ScreeningAttempt).where(
                        ScreeningAttempt.agent_id == agent_id
                    )
                )
            )
            scores = list(
                await session.scalars(select(Score).where(Score.agent_id == agent_id))
            )
            assert agent is not None
            assert agent.status == AgentStatus.SCREENING_FAILED
            assert agent.screening_policy_version == SCREENING_POLICY_VERSION
            assert [attempt.attempt_id for attempt in attempts] == [attempt_id]
            assert len(scores) == 1


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

    @pytest.mark.parametrize(
        ("outcome", "detail", "expected"),
        [
            (
                "retryable_infra",
                "build failed: dependency fetch returned 503",
                AgentStatus.SCREENING_FAILED,
            ),
            (
                "deterministic_reject",
                "screener error: deliberately misleading legacy detail",
                AgentStatus.REJECTED,
            ),
        ],
    )
    async def test_typed_failure_outcome_is_authoritative(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
        outcome: str,
        detail: str,
        expected: AgentStatus,
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
                outcome=outcome,
                detail=detail,
            ),
        )
        assert response.status_code == 200
        assert response.json()["status"] == expected

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


_ADMIN_HEADERS = {
    "Authorization": "Bearer test-admin-token-at-least-32-characters",
    "X-Admin-Actor": "backroom:test-user",
}


def _review_finding(artifact_sha256: str = _SHA256) -> SourceReviewFinding:
    return SourceReviewFinding(
        artifact_sha256=artifact_sha256,
        prompt_revision="source-review-v2",
        risk_level="high",
        confidence=0.97,
        categories=["benchmark_emulation"],
        evidence=[
            SourceReviewEvidenceItem(
                path="src/main.rs", line=2, category="benchmark_emulation"
            )
        ],
        summary="Deterministic shortcut bypasses the general provider path.",
    )


def _review_evidence(digest: str) -> list[dict[str, object]]:
    return [
        {
            "module_id": "luna-source-review",
            "code": "agentic-source-review-tripwire",
            "summary": "private source analysis selected a behavioral audit",
            "digest": digest,
        }
    ]


def _source_tarball() -> tuple[bytes, str]:
    import hashlib
    import io
    import tarfile

    files = {
        "Cargo.toml": b'[package]\nname="agent"\nversion="0.1.0"\n',
        "src/main.rs": b"fn main() {\n    fast_path();\n}\n",
        "assets/table.bin": b"\xff\xfe\x00binary-table" * 4,
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, raw in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    body = buffer.getvalue()
    return body, hashlib.sha256(body).hexdigest()


class TestQuarantineReviewContext:
    async def _quarantine(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
        finding_model: SourceReviewFinding | None = None,
        **payload_overrides: object,
    ) -> tuple[UUID, dict]:
        app.state.config = replace(
            app.state.config,
            admin_api_token="test-admin-token-at-least-32-characters",
        )
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)
        claimed = await client.post(_CLAIM_URL)
        attempt_id = UUID(claimed.json()["items"][0]["attempt_id"])
        finding = finding_model or _review_finding()
        digest = finding.canonical_digest()
        payload = _result_payload(
            agent_id,
            passed=False,
            attempt_id=attempt_id,
            outcome="quarantine",
            manifest_digest="56" * 32,
            finding_digest=digest,
            reason_code="agentic-source-review-tripwire",
            evidence=_review_evidence(digest),
            finding=finding.model_dump(mode="json"),
        )
        payload.update(payload_overrides)
        response = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result", json=payload
        )
        return agent_id, {"response": response, "finding": finding}

    async def test_review_payloads_are_stored_listed_and_digest_verified(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id, ctx = await self._quarantine(app, client, session_maker)
        assert ctx["response"].status_code == 200

        listing = await client.get(
            "/api/v1/admin/screening-quarantines", headers=_ADMIN_HEADERS
        )
        item = listing.json()["items"][0]
        assert item["agent_id"] == str(agent_id)
        assert item["finding_verified"] is True
        assert item["finding"]["risk_level"] == "high"
        assert item["finding"]["summary"] == ctx["finding"].summary
        assert item["finding"]["evidence"] == [
            {"path": "src/main.rs", "line": 2, "category": "benchmark_emulation"}
        ]
        assert [entry["code"] for entry in item["evidence"]] == [
            "agentic-source-review-tripwire"
        ]

    async def test_finding_that_does_not_match_signed_digest_is_rejected(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        tampered = _review_finding().model_dump(mode="json")
        tampered["summary"] = "tampered summary"
        _agent_id, ctx = await self._quarantine(
            app, client, session_maker, finding=tampered
        )
        assert ctx["response"].status_code == 422

    async def test_context_reports_miner_history_and_duplicates(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id, ctx = await self._quarantine(app, client, session_maker)
        assert ctx["response"].status_code == 200
        now = datetime.now(UTC)
        other_miner = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        prior_agent = uuid4()
        prior_attempt = uuid4()
        duplicate_agent = uuid4()
        async with session_maker() as session, session.begin():
            # An earlier, already-resolved quarantine from the same miner.
            session.add(
                Agent(
                    agent_id=prior_agent,
                    miner_hotkey=_MINER_HOTKEY,
                    name="alpha-agent-v1",
                    sha256="99" * 32,
                    status=AgentStatus.REJECTED,
                    screening_policy_version=SCREENING_POLICY_VERSION,
                    created_at=now - timedelta(days=2),
                )
            )
            session.add(
                ScreeningAttempt(
                    attempt_id=prior_attempt,
                    agent_id=prior_agent,
                    screener_hotkey=_SCREENER_HOTKEY,
                    policy_version=SCREENING_POLICY_VERSION,
                    status="quarantined",
                    started_at=now - timedelta(days=2),
                    deadline=now - timedelta(days=2, minutes=-30),
                    finished_at=now - timedelta(days=2),
                )
            )
            session.add(
                ScreeningQuarantine(
                    quarantine_id=uuid4(),
                    agent_id=prior_agent,
                    attempt_id=prior_attempt,
                    screener_hotkey=_SCREENER_HOTKEY,
                    policy_version=SCREENING_POLICY_VERSION,
                    manifest_digest="11" * 32,
                    reason_code="behavioral-oracle-wrong-answer",
                    status="resolved",
                    created_at=now - timedelta(days=2),
                    resolved_at=now - timedelta(days=1),
                    resolved_by="backroom:test-user",
                    resolution="reject",
                    resolution_reason="Static table confirmed",
                )
            )
            # A byte-identical artifact submitted by a different miner.
            session.add(
                Agent(
                    agent_id=duplicate_agent,
                    miner_hotkey=other_miner,
                    name="copycat-agent",
                    sha256=_SHA256,
                    status=AgentStatus.UPLOADED,
                    screening_policy_version=0,
                    created_at=now - timedelta(hours=3),
                )
            )

        listing = await client.get(
            "/api/v1/admin/screening-quarantines", headers=_ADMIN_HEADERS
        )
        quarantine_id = listing.json()["items"][0]["quarantine_id"]
        context = await client.get(
            f"/api/v1/admin/screening-quarantines/{quarantine_id}/context",
            headers=_ADMIN_HEADERS,
        )
        assert context.status_code == 200
        body = context.json()
        assert body["quarantine"]["quarantine_id"] == quarantine_id
        assert body["agent"]["agent_id"] == str(agent_id)
        assert body["agent"]["agent_status"] == AgentStatus.QUARANTINED
        assert [a["status"] for a in body["attempts"]] == ["quarantined"]
        assert body["miner"]["total_submissions"] == 2
        assert body["miner"]["quarantine_count"] == 2
        assert body["miner"]["rejected_count"] == 1
        assert [q["agent_name"] for q in body["miner"]["recent_quarantines"]] == [
            "alpha-agent-v1"
        ]
        assert body["duplicates"] == [
            {
                "agent_id": str(duplicate_agent),
                "miner_hotkey": other_miner,
                "agent_name": "copycat-agent",
                "agent_status": AgentStatus.UPLOADED,
                "submitted_at": body["duplicates"][0]["submitted_at"],
                "match": "identical_artifact",
            }
        ]
        # Attribution comes from authoritative SQL aggregates, not the sample.
        assert body["duplicate_summary"] == {
            "total": 1,
            "cross_miner": 1,
            "same_miner": 0,
            "sample_truncated": False,
        }

    async def test_finding_for_a_different_artifact_is_not_verified(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        """A digest-consistent finding about ANOTHER artifact must not verify."""
        foreign = _review_finding(artifact_sha256="ee" * 32)
        agent_id, ctx = await self._quarantine(
            app, client, session_maker, finding_model=foreign
        )
        assert ctx["response"].status_code == 200
        listing = await client.get(
            "/api/v1/admin/screening-quarantines", headers=_ADMIN_HEADERS
        )
        item = listing.json()["items"][0]
        assert item["agent_id"] == str(agent_id)
        assert item["finding_verified"] is False
        assert item["finding"]["artifact_sha256"] == "ee" * 32

    async def test_idempotent_replay_backfills_missing_review_payloads(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        """A retry can restore payloads the first report did not carry."""
        app.state.config = replace(
            app.state.config,
            admin_api_token="test-admin-token-at-least-32-characters",
        )
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)
        claimed = await client.post(_CLAIM_URL)
        attempt_id = UUID(claimed.json()["items"][0]["attempt_id"])
        finding = _review_finding()
        digest = finding.canonical_digest()
        bare = _result_payload(
            agent_id,
            passed=False,
            attempt_id=attempt_id,
            outcome="quarantine",
            manifest_digest="56" * 32,
            finding_digest=digest,
            reason_code="agentic-source-review-tripwire",
        )
        first = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result", json=bare
        )
        assert first.status_code == 200

        enriched = dict(bare)
        enriched["evidence"] = _review_evidence(digest)
        enriched["finding"] = finding.model_dump(mode="json")
        replay = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result", json=enriched
        )
        assert replay.status_code == 200

        listing = await client.get(
            "/api/v1/admin/screening-quarantines", headers=_ADMIN_HEADERS
        )
        item = listing.json()["items"][0]
        assert item["finding_verified"] is True
        assert item["finding"]["summary"] == finding.summary
        assert [entry["code"] for entry in item["evidence"]] == [
            "agentic-source-review-tripwire"
        ]

    async def test_posted_inconclusive_outcome_is_rejected_not_a_rejection(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)
        claimed = await client.post(_CLAIM_URL)
        attempt_id = UUID(claimed.json()["items"][0]["attempt_id"])
        response = await client.post(
            f"/api/v1/screener/agent/{agent_id}/result",
            json=_result_payload(
                agent_id,
                passed=False,
                attempt_id=attempt_id,
                outcome="inconclusive",
            ),
        )
        assert response.status_code == 409
        assert response.json()["error_code"] == ERROR_CODE_AGENT_NOT_SCREENABLE
        async with session_maker() as session:
            refreshed = await session.get(Agent, agent_id)
            assert refreshed is not None
            # The claim moved it to screening; the rejected non-verdict
            # must not advance or reject it.
            assert refreshed.status == AgentStatus.SCREENING

    async def test_missing_context_is_404(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        app.state.config = replace(
            app.state.config,
            admin_api_token="test-admin-token-at-least-32-characters",
        )
        _install_db(app, session_maker)
        response = await client.get(
            f"/api/v1/admin/screening-quarantines/{uuid4()}/context",
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 404


class TestQuarantineSourceInspection:
    async def _seed_with_tarball(
        self,
        app: FastAPI,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> tuple[UUID, MagicMock]:
        app.state.config = replace(
            app.state.config,
            admin_api_token="test-admin-token-at-least-32-characters",
        )
        body, sha256 = _source_tarball()
        agent_id = uuid4()
        async with session_maker() as session, session.begin():
            session.add(
                Agent(
                    agent_id=agent_id,
                    miner_hotkey=_MINER_HOTKEY,
                    name="alpha-agent",
                    sha256=sha256,
                    status=AgentStatus.QUARANTINED,
                    screening_policy_version=SCREENING_POLICY_VERSION,
                    created_at=datetime.now(UTC),
                )
            )
        _install_db(app, session_maker)
        storage = _install_storage(app)
        storage.get_object = AsyncMock(return_value=body)
        return agent_id, storage

    async def test_listing_surfaces_files_and_opaque_blobs(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id, storage = await self._seed_with_tarball(app, session_maker)
        response = await client.get(
            f"/api/v1/admin/screening-submissions/{agent_id}/source-files",
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["file_count"] == 3
        assert {entry["path"] for entry in body["files"]} == {
            "Cargo.toml",
            "src/main.rs",
            "assets/table.bin",
        }
        assert body["opaque_blobs"] == [
            {
                "path": "assets/table.bin",
                "bytes": body["opaque_blobs"][0]["bytes"],
                "reason": "non_utf8",
            }
        ]
        storage.get_object.assert_awaited_once()

    async def test_excerpt_reads_bounded_flagged_lines(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id, _storage = await self._seed_with_tarball(app, session_maker)
        response = await client.get(
            f"/api/v1/admin/screening-submissions/{agent_id}/source-file",
            params={"path": "src/main.rs", "start_line": 1, "end_line": 999},
            headers=_ADMIN_HEADERS,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["path"] == "src/main.rs"
        assert body["total_lines"] == 3
        assert body["lines"][1] == {"line": 2, "text": "    fast_path();"}

        missing = await client.get(
            f"/api/v1/admin/screening-submissions/{agent_id}/source-file",
            params={"path": "src/nope.rs"},
            headers=_ADMIN_HEADERS,
        )
        assert missing.status_code == 404

        binary = await client.get(
            f"/api/v1/admin/screening-submissions/{agent_id}/source-file",
            params={"path": "assets/table.bin"},
            headers=_ADMIN_HEADERS,
        )
        assert binary.status_code == 422

    async def test_source_reads_require_admin_actor_and_matching_digest(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id, storage = await self._seed_with_tarball(app, session_maker)
        headers = dict(_ADMIN_HEADERS)
        headers.pop("X-Admin-Actor")
        anonymous = await client.get(
            f"/api/v1/admin/screening-submissions/{agent_id}/source-files",
            headers=headers,
        )
        assert anonymous.status_code == 422

        storage.get_object = AsyncMock(return_value=b"not the stored artifact")
        tampered = await client.get(
            f"/api/v1/admin/screening-submissions/{agent_id}/source-files",
            headers=_ADMIN_HEADERS,
        )
        assert tampered.status_code == 502
