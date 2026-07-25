"""Contract and concurrency tests for public source-release settings."""

from collections.abc import AsyncIterator
from dataclasses import replace

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from ditto.api_server.dependencies import get_session

pytestmark = pytest.mark.asyncio

_ADMIN_TOKEN = "test-admin-token-at-least-32-characters"
_HEADERS = {"Authorization": f"Bearer {_ADMIN_TOKEN}"}


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


async def test_defaults_to_48_then_shortens_with_audited_revisions(
    app: FastAPI,
    client: httpx.AsyncClient,
    session_maker: async_sessionmaker[AsyncSession],
) -> None:
    _install(app, session_maker)
    initial = await client.get(
        "/api/v1/admin/artifact-release-settings", headers=_HEADERS
    )
    assert initial.status_code == 200
    # The 48-hour default is served from the migration-seeded audit chain, not
    # from the built-in fallback: the first artifact-release migration seeds a
    # row and the table is append-only, so `revision 0 / actor "platform"` is
    # a state production is never in.
    baseline = initial.json()
    assert baseline["current"]["embargo_hours"] == 48
    assert baseline["current"]["actor"] == "migration"
    assert baseline["current"]["created_at"] is not None
    head = baseline["current"]["revision"]
    seeded_hours = [row["embargo_hours"] for row in baseline["history"]]

    twelve = await client.post(
        "/api/v1/admin/artifact-release-settings",
        headers=_HEADERS,
        json=_payload(12, expected=head),
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
    assert [row["embargo_hours"] for row in current.json()["history"]] == [
        6,
        12,
        *seeded_hours,
    ]


async def test_lengthens_shortens_and_rejects_stale_or_wrong_confirmation(
    app: FastAPI,
    client: httpx.AsyncClient,
    session_maker: async_sessionmaker[AsyncSession],
) -> None:
    _install(app, session_maker)
    head = (
        await client.get("/api/v1/admin/artifact-release-settings", headers=_HEADERS)
    ).json()["current"]["revision"]
    first = await client.post(
        "/api/v1/admin/artifact-release-settings",
        headers=_HEADERS,
        json=_payload(12, expected=head),
    )
    assert first.status_code == 200, first.text
    revision = first.json()["revision"]

    # Lengthening is now allowed, up to the 48-hour ceiling.
    increase = await client.post(
        "/api/v1/admin/artifact-release-settings",
        headers=_HEADERS,
        json=_payload(24, expected=revision),
    )
    assert increase.status_code == 200, increase.text
    revision = increase.json()["revision"]

    ceiling = await client.post(
        "/api/v1/admin/artifact-release-settings",
        headers=_HEADERS,
        json=_payload(48, expected=revision),
    )
    assert ceiling.status_code == 200, ceiling.text
    assert ceiling.json()["embargo_hours"] == 48

    # One hour past the ceiling is rejected by the request contract.
    over = await client.post(
        "/api/v1/admin/artifact-release-settings",
        headers=_HEADERS,
        json=_payload(49, expected=ceiling.json()["revision"]),
    )
    assert over.status_code == 422

    stale = await client.post(
        "/api/v1/admin/artifact-release-settings",
        headers=_HEADERS,
        json=_payload(6, expected=0),
    )
    assert stale.status_code == 409
    assert "refresh before applying" in stale.text

    wrong = _payload(6, expected=ceiling.json()["revision"])
    wrong["confirmation"] = "SET SOURCE EMBARGO 12 HOURS"
    confirmation = await client.post(
        "/api/v1/admin/artifact-release-settings",
        headers=_HEADERS,
        json=wrong,
    )
    assert confirmation.status_code == 409
    assert "must be exactly" in confirmation.text
