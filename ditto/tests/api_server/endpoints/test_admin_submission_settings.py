"""Contract tests for platform-owned submission cooldown settings."""

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

_ADMIN_TOKEN = "test-admin-token-at-least-32-characters"
_HEADERS = {"Authorization": f"Bearer {_ADMIN_TOKEN}"}


@pytest.fixture
async def settings_maker() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


def _install(app: FastAPI, maker: async_sessionmaker[AsyncSession]) -> None:
    app.state.config = replace(app.state.config, admin_api_token=_ADMIN_TOKEN)

    async def _session() -> AsyncIterator[AsyncSession]:
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = _session


def _payload(seconds: int, expected: int = 0) -> dict[str, object]:
    return {
        "expected_revision": expected,
        "cooldown_seconds": seconds,
        "reason": f"set miner submission cooldown to {seconds} seconds",
        "actor": "operator@example.com",
        "confirmation": f"SET SUBMISSION COOLDOWN {seconds} SECONDS",
    }


async def test_defaults_to_one_hour_and_appends_audited_revision(
    app: FastAPI,
    client: httpx.AsyncClient,
    settings_maker: async_sessionmaker[AsyncSession],
) -> None:
    _install(app, settings_maker)
    initial = await client.get("/api/v1/admin/submission-settings", headers=_HEADERS)
    assert initial.status_code == 200
    assert initial.json()["current"]["cooldown_seconds"] == 3600

    updated = await client.post(
        "/api/v1/admin/submission-settings",
        headers=_HEADERS,
        json=_payload(1800),
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["cooldown_seconds"] == 1800
    assert updated.json()["actor"] == "operator@example.com"


async def test_rejects_stale_revision_wrong_confirmation_and_unsafe_bounds(
    app: FastAPI,
    client: httpx.AsyncClient,
    settings_maker: async_sessionmaker[AsyncSession],
) -> None:
    _install(app, settings_maker)
    first = await client.post(
        "/api/v1/admin/submission-settings",
        headers=_HEADERS,
        json=_payload(1800),
    )
    assert first.status_code == 200

    stale = await client.post(
        "/api/v1/admin/submission-settings",
        headers=_HEADERS,
        json=_payload(1200, expected=0),
    )
    assert stale.status_code == 409

    wrong = _payload(1200, expected=first.json()["revision"])
    wrong["confirmation"] = "SET SUBMISSION COOLDOWN 3600 SECONDS"
    confirmation = await client.post(
        "/api/v1/admin/submission-settings", headers=_HEADERS, json=wrong
    )
    assert confirmation.status_code == 409

    too_short = await client.post(
        "/api/v1/admin/submission-settings",
        headers=_HEADERS,
        json=_payload(59, expected=first.json()["revision"]),
    )
    assert too_short.status_code == 422
