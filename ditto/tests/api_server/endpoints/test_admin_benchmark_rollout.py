"""HTTP contract tests for the guarded benchmark rollout control plane."""

from collections.abc import AsyncIterator
from dataclasses import replace

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ditto.api_server.dependencies import get_session
from ditto.db.models import Base

pytestmark = pytest.mark.asyncio

_TOKEN = "test-admin-token-at-least-32-characters"
_HEADERS = {"Authorization": f"Bearer {_TOKEN}"}


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
