"""Contract and concurrency tests for public source-release settings."""

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


def _payload(hours: int, expected: int = 0) -> dict[str, object]:
    return {
        "expected_revision": expected,
        "embargo_hours": hours,
        "reason": f"stage public source release at {hours} hours",
        "actor": "operator@example.com",
        "confirmation": f"SET SOURCE EMBARGO {hours} HOURS",
    }


async def test_defaults_to_24_then_shortens_with_audited_revisions(
    app: FastAPI,
    client: httpx.AsyncClient,
    settings_maker: async_sessionmaker[AsyncSession],
) -> None:
    _install(app, settings_maker)
    initial = await client.get(
        "/api/v1/admin/artifact-release-settings", headers=_HEADERS
    )
    assert initial.status_code == 200
    assert initial.json()["current"] == {
        "revision": 0,
        "parent_revision": 0,
        "embargo_hours": 24,
        "reason": "Built-in privacy-first default",
        "actor": "platform",
        "created_at": None,
    }

    twelve = await client.post(
        "/api/v1/admin/artifact-release-settings",
        headers=_HEADERS,
        json=_payload(12),
    )
    assert twelve.status_code == 200, twelve.text
    revision = twelve.json()["revision"]

    six = await client.post(
        "/api/v1/admin/artifact-release-settings",
        headers=_HEADERS,
        json=_payload(6, expected=revision),
    )
    assert six.status_code == 200, six.text
    current = await client.get(
        "/api/v1/admin/artifact-release-settings", headers=_HEADERS
    )
    assert current.json()["current"]["embargo_hours"] == 6
    assert [row["embargo_hours"] for row in current.json()["history"]] == [6, 12]


async def test_rejects_increases_stale_writes_and_wrong_confirmation(
    app: FastAPI,
    client: httpx.AsyncClient,
    settings_maker: async_sessionmaker[AsyncSession],
) -> None:
    _install(app, settings_maker)
    first = await client.post(
        "/api/v1/admin/artifact-release-settings",
        headers=_HEADERS,
        json=_payload(12),
    )
    revision = first.json()["revision"]

    increase = await client.post(
        "/api/v1/admin/artifact-release-settings",
        headers=_HEADERS,
        json=_payload(24, expected=revision),
    )
    assert increase.status_code == 409
    assert "only be shortened" in increase.text

    stale = await client.post(
        "/api/v1/admin/artifact-release-settings",
        headers=_HEADERS,
        json=_payload(6, expected=0),
    )
    assert stale.status_code == 409
    assert "refresh before applying" in stale.text

    wrong = _payload(6, expected=revision)
    wrong["confirmation"] = "SET SOURCE EMBARGO 12 HOURS"
    confirmation = await client.post(
        "/api/v1/admin/artifact-release-settings",
        headers=_HEADERS,
        json=wrong,
    )
    assert confirmation.status_code == 409
    assert "must be exactly" in confirmation.text
