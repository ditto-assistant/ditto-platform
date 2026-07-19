"""HTTP contract tests for the guarded benchmark rollout control plane."""

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ditto.api_models.agent_status import AgentStatus
from ditto.api_server.datapipeline import DataPipelineError
from ditto.api_server.dependencies import get_session
from ditto.api_server.endpoints.admin_benchmark_rollout import (
    MINIMUM_ROLLOUT_START_VALIDATORS,
)
from ditto.api_server.middleware.error_envelope import (
    ERROR_CODE_HTTP_EXCEPTION,
    ERROR_CODE_UNHANDLED,
)
from ditto.db.models import (
    Agent,
    Base,
    BenchmarkRollout,
    Score,
    ValidatorHeartbeat,
)
from ditto.db.queries.benchmark_rollout import (
    DatasetPin,
    RolloutSnapshotMember,
    create_rollout_snapshot,
)

pytestmark = pytest.mark.asyncio

_TOKEN = "test-admin-token-at-least-32-characters"
_HEADERS = {"Authorization": f"Bearer {_TOKEN}"}
_TARGET = 4
# The message the deployed generate-service actually returned when the v4
# rollout was started from the operator console against a datagen release
# that only ships v2 and v3.
_LAGGING = "bench_version query param required (supported: 2, 3)"


@pytest.fixture
async def rollout_maker() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


def _install(app: FastAPI, maker: async_sessionmaker[AsyncSession]) -> None:
    app.state.config = replace(app.state.config, admin_api_token=_TOKEN)

    async def _session() -> AsyncIterator[AsyncSession]:
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = _session


class _StubGenerator:
    """The separately deployed generate-service, present or lagging.

    ``run_size`` must be set: a ``None`` run size is read as "generation
    disabled" and blocks qualification before any call is attempted, which
    would make these tests pass for the wrong reason.
    """

    run_size = "full"

    def __init__(self, error: DataPipelineError | None = None) -> None:
        self._error = error
        self.calls: list[tuple[int, int]] = []

    async def generate(self, seed: int, bench_version: int = 2) -> str:
        self.calls.append((seed, bench_version))
        if self._error is not None:
            raise self._error
        return f"{seed:064x}"

    async def aclose(self) -> None:
        return None


def _capabilities(now: datetime) -> tuple[dict[str, Any], dict[str, Any]]:
    revision = "a" * 40
    capabilities = {
        "screened_images": True,
        "require_screened_image": False,
        "source_build_fallback": True,
        "full_stack_managed": False,
        "stack_updater": False,
        "sandbox_egress_restricted": True,
        "executor_isolation": "privileged_dind",
        "scorer_benchmarks": {
            "status": "fresh_verified",
            "supported_bench_versions": [2, 3, _TARGET],
            "observed_at": int(now.timestamp()),
            "software_version": "1.3.0",
            "source_revision": revision,
        },
    }
    stack = {
        "mode": "source",
        "compose_schema": 1,
        "release_descriptor_digest": None,
        "components": {
            name: {
                "source_revision": revision if name == "dittobench_api" else "b" * 40,
                "version": "1.3.0" if name == "dittobench_api" else "1.2.0",
                "provenance": "committed_pin",
            }
            for name in (
                "ditto_subnet",
                "dittobench_api",
                "sandbox_docker",
                "model_relay",
                "pylon",
                "ollama",
            )
        },
    }
    return capabilities, stack


def _add_cohort_agent(
    session: AsyncSession, *, position: int, composite: float, now: datetime
) -> RolloutSnapshotMember:
    """Add one SCORED agent with a full v2 quorum, top-five eligible."""
    agent_id = uuid4()
    miner = f"miner-{position}"
    digest = f"{position:x}" * 64
    session.add(
        Agent(
            agent_id=agent_id,
            miner_hotkey=miner,
            name=f"agent-{position}",
            sha256=digest,
            status=AgentStatus.SCORED,
            screening_policy_version=9,
            screened_image_sha256=digest,
            screened_image_size_bytes=1024,
            screened_image_id=f"sha256:{digest}",
            screened_image_ref=f"ditto-screen/{agent_id}:latest",
            screened_image_upload_id=uuid4(),
            screened_image_verified_at=now,
            created_at=now + timedelta(seconds=position),
        )
    )
    for validator in range(3):
        session.add(
            Score(
                agent_id=agent_id,
                bench_version=2,
                validator_hotkey=f"legacy-{validator}",
                run_id=f"v2-{position}-{validator}",
                signature="aa",
                seed=position,
                composite=composite,
                tool_mean=0.5,
                memory_mean=0.5,
                median_ms=1,
                n=114,
                details={"bench_version": 2},
                generated_at=now,
            )
        )
    return RolloutSnapshotMember(
        agent_id=agent_id, miner_hotkey=miner, composite=composite
    )


async def _seed_start_ready(
    maker: async_sessionmaker[AsyncSession], now: datetime
) -> list[RolloutSnapshotMember]:
    """Seed the five-miner cohort and just enough capable validators to start."""
    capabilities, stack = _capabilities(now)
    async with maker() as session, session.begin():
        members = [
            _add_cohort_agent(
                session, position=position, composite=0.5 + position / 100, now=now
            )
            for position in range(1, 6)
        ]
        for index in range(MINIMUM_ROLLOUT_START_VALIDATORS):
            session.add(
                ValidatorHeartbeat(
                    validator_hotkey=f"validator-{index}",
                    software_version="1.0.0",
                    protocol_version=8,
                    code_digest="d" * 64,
                    state="polling",
                    first_seen_at=now,
                    reported_at=now,
                    seen_at=now,
                    signature="ab" * 64,
                    capabilities=capabilities,
                    stack=stack,
                )
            )
    return members


async def _rollout_count(maker: async_sessionmaker[AsyncSession]) -> int:
    async with maker() as session:
        return int(
            await session.scalar(select(func.count(BenchmarkRollout.rollout_id))) or 0
        )


def _start_payload() -> dict[str, Any]:
    return {
        "reason": f"start the v{_TARGET} rollout",
        "actor": "backroom:test",
        "confirmation": f"START BENCHMARK V{_TARGET}",
        "expected_active_version": 2,
    }


async def test_control_discovery_is_authenticated_read_only_and_dynamic(
    app: FastAPI,
    client: httpx.AsyncClient,
    rollout_maker: async_sessionmaker[AsyncSession],
) -> None:
    _install(app, rollout_maker)

    denied = await client.get("/api/v1/admin/benchmark-rollout")
    assert denied.status_code == 401

    response = await client.get("/api/v1/admin/benchmark-rollout", headers=_HEADERS)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["active_version"] == 2
    assert body["status"] == "inactive"
    assert body["available_target_versions"] == [3, 4]
    assert [contract["version"] for contract in body["contracts"]] == [2, 3, 4]
    assert all(
        contract["capable_validator_count"] == 0 for contract in body["contracts"]
    )


async def test_start_requires_full_guard_payload_and_exact_confirmation(
    app: FastAPI,
    client: httpx.AsyncClient,
    rollout_maker: async_sessionmaker[AsyncSession],
) -> None:
    _install(app, rollout_maker)

    missing = await client.post("/api/v1/admin/benchmark-rollout/4", headers=_HEADERS)
    assert missing.status_code == 422

    wrong = await client.post(
        "/api/v1/admin/benchmark-rollout/4",
        headers=_HEADERS,
        json={
            "reason": "prepare the v4 rollout",
            "actor": "backroom:test",
            "confirmation": "START BENCHMARK V3",
            "expected_active_version": 2,
        },
    )
    assert wrong.status_code == 409
    assert "START BENCHMARK V4" in wrong.json()["message"]

    unsupported = await client.post(
        "/api/v1/admin/benchmark-rollout/5",
        headers=_HEADERS,
        json={
            "reason": "attempt an unshipped contract",
            "actor": "backroom:test",
            "confirmation": "START BENCHMARK V5",
            "expected_active_version": 2,
        },
    )
    assert unsupported.status_code == 409
    assert "no shipped contract" in unsupported.json()["message"]


async def test_start_reports_a_lagging_generate_service_as_a_named_502(
    app: FastAPI,
    client: httpx.AsyncClient,
    rollout_maker: async_sessionmaker[AsyncSession],
) -> None:
    """The generator lags this API by a deploy; that is a 502, never a bare 500."""
    _install(app, rollout_maker)
    now = datetime.now(UTC).replace(microsecond=0)
    await _seed_start_ready(rollout_maker, now)
    generator = _StubGenerator(DataPipelineError(_LAGGING))
    app.state.dataset_generator = generator

    response = await client.post(
        f"/api/v1/admin/benchmark-rollout/{_TARGET}",
        headers=_HEADERS,
        json=_start_payload(),
    )

    assert response.status_code == 502, response.text
    body = response.json()
    # A handled HTTPException, not the unhandled-exception envelope the
    # operator got in production.
    assert body["error_code"] == ERROR_CODE_HTTP_EXCEPTION
    assert body["error_code"] != ERROR_CODE_UNHANDLED
    message = body["message"]
    assert f"v{_TARGET}" in message
    assert _LAGGING in message
    assert "generate-service" in message
    # The lag is version-specific, so the call must have asked for the target.
    assert [version for _seed, version in generator.calls] == [_TARGET]
    # Nothing half-started: the failed attempt leaves no rollout behind, so a
    # retry after the generator is deployed is a clean start, not a 409.
    assert await _rollout_count(rollout_maker) == 0


async def test_reposting_an_existing_rollout_also_502s_on_a_lagging_generator(
    app: FastAPI,
    client: httpx.AsyncClient,
    rollout_maker: async_sessionmaker[AsyncSession],
) -> None:
    """Re-POSTing is the natural operator retry, so it must not be the 500 route."""
    _install(app, rollout_maker)
    now = datetime.now(UTC).replace(microsecond=0)
    members = await _seed_start_ready(rollout_maker, now)
    async with rollout_maker() as session, session.begin():
        await create_rollout_snapshot(
            session,
            members=members,
            datasets={
                member.agent_id: DatasetPin(
                    seed=index, sha256="c" * 64, run_size="full"
                )
                for index, member in enumerate(members, start=1)
            },
            now=now,
            from_version=2,
            desired_version=_TARGET,
        )
        # A newly risen agent outranks the frozen cohort, so the idempotent
        # refresh has something to render and reaches the generator.
        _add_cohort_agent(session, position=6, composite=0.99, now=now)
    generator = _StubGenerator(DataPipelineError(_LAGGING))
    app.state.dataset_generator = generator

    response = await client.post(
        f"/api/v1/admin/benchmark-rollout/{_TARGET}",
        headers=_HEADERS,
        json=_start_payload(),
    )

    assert response.status_code == 502, response.text
    body = response.json()
    assert body["error_code"] == ERROR_CODE_HTTP_EXCEPTION
    assert body["error_code"] != ERROR_CODE_UNHANDLED
    assert _LAGGING in body["message"]
    assert f"v{_TARGET}" in body["message"]
    assert "generate-service" in body["message"]
    assert [version for _seed, version in generator.calls] == [_TARGET]


async def test_start_still_succeeds_when_the_generator_ships_the_target(
    app: FastAPI,
    client: httpx.AsyncClient,
    rollout_maker: async_sessionmaker[AsyncSession],
) -> None:
    """Anti-no-op guard: the 502 wrapper swallows nothing on the happy path."""
    _install(app, rollout_maker)
    now = datetime.now(UTC).replace(microsecond=0)
    members = await _seed_start_ready(rollout_maker, now)
    generator = _StubGenerator()
    app.state.dataset_generator = generator

    response = await client.post(
        f"/api/v1/admin/benchmark-rollout/{_TARGET}",
        headers=_HEADERS,
        json=_start_payload(),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "collecting"
    assert body["desired_version"] == _TARGET
    assert body["active_version"] == 2
    assert [version for _seed, version in generator.calls] == [_TARGET] * len(members)
    assert {UUID(member["agent_id"]) for member in body["members"]} == {
        member.agent_id for member in members
    }
    assert await _rollout_count(rollout_maker) == 1
