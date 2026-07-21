"""Coverage for the admin scoring-readiness inspection endpoint."""

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

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
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.api_server.dependencies import get_session
from ditto.db.models import Agent, Base, BenchmarkDataset, BenchmarkRollout

_TOKEN = "test-admin-token-at-least-32-characters"
_HEADERS = {"Authorization": f"Bearer {_TOKEN}", "X-Admin-Actor": "operator"}
_T0 = datetime(2026, 7, 21, 4, tzinfo=UTC)


@pytest.fixture
async def sr_engine() -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def _fk(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def sr_maker(sr_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(sr_engine, expire_on_commit=False)


def _install(app: FastAPI, maker: async_sessionmaker[AsyncSession]) -> None:
    app.state.config = replace(app.state.config, admin_api_token=_TOKEN)

    async def _session() -> AsyncIterator[AsyncSession]:
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = _session


async def _seed(
    maker: async_sessionmaker[AsyncSession],
    *,
    active_version: int = 4,
    status: AgentStatus = AgentStatus.EVALUATING,
    policy_version: int = SCREENING_POLICY_VERSION,
    with_image: bool = True,
    with_dataset: bool = True,
) -> UUID:
    agent_id = uuid4()
    async with maker() as session, session.begin():
        session.add(
            BenchmarkRollout(
                rollout_id=uuid4(),
                from_version=2,
                desired_version=active_version,
                status="activated",
                activated_at=_T0 - timedelta(hours=1),
            )
        )
        image = (
            {
                "screened_image_sha256": "a" * 64,
                "screened_image_size_bytes": 1024,
                "screened_image_id": "sha256:" + "b" * 64,
                "screened_image_ref": f"ditto-screen/{agent_id}:latest",
                "screened_image_upload_id": uuid4(),
                "screened_image_verified_at": _T0 - timedelta(minutes=30),
            }
            if with_image
            else {}
        )
        session.add(
            Agent(
                agent_id=agent_id,
                miner_hotkey="5Miner",
                name="candidate",
                version=1,
                sha256=agent_id.hex * 2,
                status=status,
                screening_policy_version=policy_version,
                created_at=_T0 - timedelta(days=1),
                **image,
            )
        )
        if with_dataset:
            session.add(
                BenchmarkDataset(
                    agent_id=agent_id,
                    bench_version=active_version,
                    seed=7,
                    sha256="d" * 64,
                    run_size="full",
                )
            )
    return agent_id


async def _get(client: httpx.AsyncClient, agent_id: UUID) -> dict:
    resp = await client.get(
        f"/api/v1/admin/agents/{agent_id}/scoring-readiness", headers=_HEADERS
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


async def test_fully_ready_v4_agent_is_leaseable(
    app: FastAPI, client: httpx.AsyncClient, sr_maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id = await _seed(sr_maker)
    _install(app, sr_maker)
    body = await _get(client, agent_id)
    assert body["leaseable"] is True
    assert body["blocking_reasons"] == []
    assert body["active_bench_version"] == 4
    assert body["has_versioned_dataset"] is True
    assert body["screened_image"]["complete"] is True


async def test_missing_screened_image_blocks_and_names_fields(
    app: FastAPI, client: httpx.AsyncClient, sr_maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id = await _seed(sr_maker, with_image=False)
    _install(app, sr_maker)
    body = await _get(client, agent_id)
    assert body["leaseable"] is False
    assert body["screened_image"]["complete"] is False
    assert body["screened_image"]["missing_fields"]
    assert any("not built yet" in r for r in body["blocking_reasons"])


async def test_missing_v4_dataset_blocks(
    app: FastAPI, client: httpx.AsyncClient, sr_maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id = await _seed(sr_maker, with_dataset=False)
    _install(app, sr_maker)
    body = await _get(client, agent_id)
    assert body["leaseable"] is False
    assert body["has_versioned_dataset"] is False
    assert any("benchmark dataset" in r for r in body["blocking_reasons"])


async def test_stale_screening_policy_blocks(
    app: FastAPI, client: httpx.AsyncClient, sr_maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id = await _seed(sr_maker, policy_version=SCREENING_POLICY_VERSION - 1)
    _install(app, sr_maker)
    body = await _get(client, agent_id)
    assert body["leaseable"] is False
    assert any("re-screen" in r for r in body["blocking_reasons"])


async def test_non_evaluating_agent_blocks(
    app: FastAPI, client: httpx.AsyncClient, sr_maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id = await _seed(sr_maker, status=AgentStatus.QUARANTINED)
    _install(app, sr_maker)
    body = await _get(client, agent_id)
    assert body["leaseable"] is False
    assert any("not evaluating" in r for r in body["blocking_reasons"])


async def test_unknown_agent_is_404(
    app: FastAPI, client: httpx.AsyncClient, sr_maker: async_sessionmaker[AsyncSession]
) -> None:
    _install(app, sr_maker)
    resp = await client.get(
        f"/api/v1/admin/agents/{uuid4()}/scoring-readiness", headers=_HEADERS
    )
    assert resp.status_code == 404
