"""Contract tests for hot-swappable continual-retest settings."""

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

from ditto.api_server.continual_retest_settings import aggregate_is_active
from ditto.api_server.dependencies import get_session
from ditto.db.models import Base

pytestmark = pytest.mark.asyncio

_ADMIN_TOKEN = "test-admin-token-at-least-32-characters"
_HEADERS = {"Authorization": f"Bearer {_ADMIN_TOKEN}"}
_URL = "/api/v1/admin/continual-retest-settings"


@pytest.fixture
async def settings_maker() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


def _install(app: FastAPI, maker: async_sessionmaker[AsyncSession]) -> None:
    app.state.config = replace(app.state.config, admin_api_token=_ADMIN_TOKEN)
    app.state.session_maker = maker

    async def _session() -> AsyncIterator[AsyncSession]:
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = _session
    app.state.continual_retest_settings.invalidate()


def _payload(
    *,
    expected_revision: int = 0,
    aggregate_mode: str = "fleet_ready",
    idle_retests_enabled: bool = False,
    rollout_standdown: str = "capable_validators",
) -> dict[str, object]:
    return {
        "expected_revision": expected_revision,
        "scope": "*",
        "settings": {
            "aggregate_mode": aggregate_mode,
            "idle_retests_enabled": idle_retests_enabled,
            "rollout_standdown": rollout_standdown,
        },
        "reason": "operator-approved continual retest policy change",
        "actor": "operator@example.com",
        "confirmation": "APPLY CONTINUAL RETEST SETTINGS",
    }


async def test_defaults_are_safe_and_revision_is_audited(
    app: FastAPI,
    client: httpx.AsyncClient,
    settings_maker: async_sessionmaker[AsyncSession],
) -> None:
    _install(app, settings_maker)

    initial = await client.get(_URL, headers=_HEADERS)
    assert initial.status_code == 200, initial.text
    assert initial.json()["effective"]["settings"] == {
        "aggregate_mode": "fleet_ready",
        "idle_retests_enabled": False,
        "rollout_standdown": "capable_validators",
    }
    assert initial.json()["effective"]["open_rollout_desired_version"] is None
    assert initial.json()["effective"]["rollout_standdown_active"] is False

    updated = await client.post(
        _URL,
        headers=_HEADERS,
        json=_payload(aggregate_mode="enabled", idle_retests_enabled=True),
    )
    assert updated.status_code == 200, updated.text
    body = updated.json()
    assert body["revision"] == 1
    assert body["parent_revision"] == 0
    assert body["settings"]["aggregate_mode"] == "enabled"
    assert body["settings"]["idle_retests_enabled"] is True
    assert body["actor"] == "operator@example.com"

    refreshed = await client.get(_URL, headers=_HEADERS)
    assert refreshed.status_code == 200
    assert refreshed.json()["effective"]["revision"] == 1
    assert refreshed.json()["effective"]["aggregate_active"] is True


async def test_rejects_stale_revision_and_wrong_confirmation(
    app: FastAPI,
    client: httpx.AsyncClient,
    settings_maker: async_sessionmaker[AsyncSession],
) -> None:
    _install(app, settings_maker)
    first = await client.post(_URL, headers=_HEADERS, json=_payload())
    assert first.status_code == 200

    stale = await client.post(_URL, headers=_HEADERS, json=_payload())
    assert stale.status_code == 409

    wrong = _payload(expected_revision=1)
    wrong["confirmation"] = "ENABLE RETESTS"
    confirmation = await client.post(_URL, headers=_HEADERS, json=wrong)
    assert confirmation.status_code == 409


async def test_aggregate_modes_are_explicit() -> None:
    from ditto.api_models.continual_retest_settings import ContinualRetestSettings

    assert aggregate_is_active(
        ContinualRetestSettings(aggregate_mode="fleet_ready"),
        fleet_protocol_ready=True,
    )
    assert not aggregate_is_active(
        ContinualRetestSettings(aggregate_mode="fleet_ready"),
        fleet_protocol_ready=False,
    )
    assert aggregate_is_active(
        ContinualRetestSettings(aggregate_mode="enabled"),
        fleet_protocol_ready=False,
    )
    assert not aggregate_is_active(
        ContinualRetestSettings(aggregate_mode="disabled"),
        fleet_protocol_ready=True,
    )


async def test_open_rollout_surfaces_the_standdown_state(
    app: FastAPI,
    client: httpx.AsyncClient,
    settings_maker: async_sessionmaker[AsyncSession],
) -> None:
    """An operator must be able to see why retests went quiet."""
    from datetime import UTC, datetime, timedelta
    from uuid import uuid4

    from ditto.db.models import BenchmarkRollout

    _install(app, settings_maker)
    async with settings_maker() as session, session.begin():
        session.add(
            BenchmarkRollout(
                rollout_id=uuid4(),
                from_version=6,
                desired_version=7,
                status="collecting",
                cohort_size=5,
                created_at=datetime.now(UTC) - timedelta(hours=1),
            )
        )

    collecting = await client.get(_URL, headers=_HEADERS)
    assert collecting.status_code == 200, collecting.text
    effective = collecting.json()["effective"]
    assert effective["open_rollout_desired_version"] == 7
    assert effective["rollout_standdown_active"] is True

    override = _payload(rollout_standdown="off")
    applied = await client.post(_URL, headers=_HEADERS, json=override)
    assert applied.status_code == 200, applied.text
    assert applied.json()["settings"]["rollout_standdown"] == "off"

    forced = await client.get(_URL, headers=_HEADERS)
    assert forced.json()["effective"]["open_rollout_desired_version"] == 7
    assert forced.json()["effective"]["rollout_standdown_active"] is False


async def test_rollout_standdown_modes_are_explicit() -> None:
    from ditto.api_models.continual_retest_settings import ContinualRetestSettings
    from ditto.api_server.continual_retest_settings import rollout_standdown_reason

    default = ContinualRetestSettings()
    assert default.rollout_standdown == "capable_validators"
    # No open rollout is the only state in which the lane is unconditionally on.
    assert (
        rollout_standdown_reason(
            default,
            open_rollout_desired_version=None,
            validator_supports_desired_version=True,
        )
        is None
    )
    reason = rollout_standdown_reason(
        default,
        open_rollout_desired_version=7,
        validator_supports_desired_version=True,
    )
    assert reason is not None
    assert "resume automatically" in reason
    # Capacity the rollout cannot consume keeps confirming the active era.
    assert (
        rollout_standdown_reason(
            default,
            open_rollout_desired_version=7,
            validator_supports_desired_version=False,
        )
        is None
    )
    assert (
        rollout_standdown_reason(
            ContinualRetestSettings(rollout_standdown="all"),
            open_rollout_desired_version=7,
            validator_supports_desired_version=False,
        )
        is not None
    )
    assert (
        rollout_standdown_reason(
            ContinualRetestSettings(rollout_standdown="off"),
            open_rollout_desired_version=7,
            validator_supports_desired_version=True,
        )
        is None
    )
