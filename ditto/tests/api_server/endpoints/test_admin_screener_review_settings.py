"""Contract and concurrency tests for screener review settings."""

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
_ADMIN_HEADERS = {"Authorization": f"Bearer {_ADMIN_TOKEN}"}
_SCREENER_HEADERS = {
    "Authorization": "Bearer test-screener-token-at-least-32-characters",
    "X-Screener-Hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
}


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


def _payload(scope: str, mode: str, expected: int = 0) -> dict[str, object]:
    return {
        "scope": scope,
        "expected_revision": expected,
        "settings": {"mode": mode},
        "reason": f"exercise {mode} settings safely",
        "actor": "backroom:test",
        "confirmation": f"APPLY SCREENER REVIEW {scope} {mode.upper()}",
    }


async def test_builtin_off_then_global_shadow_and_instance_override(
    app: FastAPI,
    client: httpx.AsyncClient,
    settings_maker: async_sessionmaker[AsyncSession],
) -> None:
    _install(app, settings_maker)
    denied = await client.get(
        "/api/v1/screener/review-settings?instance_id=ditto-screener-prod"
    )
    assert denied.status_code == 401

    initial = await client.get(
        "/api/v1/screener/review-settings?instance_id=ditto-screener-prod",
        headers=_SCREENER_HEADERS,
    )
    assert initial.status_code == 200
    assert initial.json()["revision"] == 0
    assert initial.json()["settings"]["mode"] == "off"
    assert initial.headers["etag"]

    global_write = await client.post(
        "/api/v1/admin/screener-review-settings",
        headers=_ADMIN_HEADERS,
        json=_payload("*", "shadow"),
    )
    assert global_write.status_code == 200, global_write.text
    global_revision = global_write.json()["revision"]

    fleet = await client.get(
        "/api/v1/screener/review-settings?instance_id=ditto-screener-fleet-abc",
        headers=_SCREENER_HEADERS,
    )
    assert fleet.json()["revision"] == global_revision
    assert fleet.json()["scope"] == "*"
    assert fleet.json()["settings"]["mode"] == "shadow"

    override = await client.post(
        "/api/v1/admin/screener-review-settings",
        headers=_ADMIN_HEADERS,
        json=_payload("ditto-screener-prod", "off"),
    )
    assert override.status_code == 200, override.text
    pet = await client.get(
        "/api/v1/screener/review-settings?instance_id=ditto-screener-prod",
        headers=_SCREENER_HEADERS,
    )
    assert pet.json()["revision"] == override.json()["revision"]
    assert pet.json()["scope"] == "ditto-screener-prod"
    assert pet.json()["settings"]["mode"] == "off"


async def test_stale_parent_and_duplicate_model_chain_are_rejected(
    app: FastAPI,
    client: httpx.AsyncClient,
    settings_maker: async_sessionmaker[AsyncSession],
) -> None:
    _install(app, settings_maker)
    first = await client.post(
        "/api/v1/admin/screener-review-settings",
        headers=_ADMIN_HEADERS,
        json=_payload("ditto-screener-prod", "shadow"),
    )
    assert first.status_code == 200

    stale = await client.post(
        "/api/v1/admin/screener-review-settings",
        headers=_ADMIN_HEADERS,
        json=_payload("ditto-screener-prod", "off", expected=0),
    )
    assert stale.status_code == 409

    invalid = _payload("*", "shadow")
    invalid["settings"] = {
        "mode": "shadow",
        "l2_model": "moonshotai/kimi-k3",
        "l2_fallback_models": ["moonshotai/kimi-k3"],
    }
    duplicate = await client.post(
        "/api/v1/admin/screener-review-settings",
        headers=_ADMIN_HEADERS,
        json=invalid,
    )
    assert duplicate.status_code == 422


async def test_admin_read_is_authenticated_and_history_is_append_only(
    app: FastAPI,
    client: httpx.AsyncClient,
    settings_maker: async_sessionmaker[AsyncSession],
) -> None:
    _install(app, settings_maker)
    denied = await client.get("/api/v1/admin/screener-review-settings")
    assert denied.status_code == 401
    first = await client.post(
        "/api/v1/admin/screener-review-settings",
        headers=_ADMIN_HEADERS,
        json=_payload("*", "off"),
    )
    second = await client.post(
        "/api/v1/admin/screener-review-settings",
        headers=_ADMIN_HEADERS,
        json=_payload("*", "shadow", expected=first.json()["revision"]),
    )
    assert second.status_code == 200
    state = await client.get(
        "/api/v1/admin/screener-review-settings", headers=_ADMIN_HEADERS
    )
    assert state.status_code == 200
    assert len(state.json()["current"]) == 1
    assert [item["revision"] for item in state.json()["history"]] == [
        second.json()["revision"],
        first.json()["revision"],
    ]
