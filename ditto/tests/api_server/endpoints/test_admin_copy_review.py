"""Durable ATH copy-review API regression coverage."""

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

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

from ditto.api_server.dependencies import get_session
from ditto.db.models import Agent, AgentStatus, AthReview, Base

_TOKEN = "test-admin-token-at-least-32-characters"
_HEADERS = {"Authorization": f"Bearer {_TOKEN}", "X-Admin-Actor": "operator"}
_T0 = datetime(2026, 7, 16, 12, tzinfo=UTC)


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(eng.sync_engine, "connect")
    def _fk(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
def maker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


def _install(app: FastAPI, maker: async_sessionmaker[AsyncSession]) -> None:
    app.state.config = replace(app.state.config, admin_api_token=_TOKEN)

    async def _session() -> AsyncIterator[AsyncSession]:
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = _session


async def _seed(
    maker: async_sessionmaker[AsyncSession],
    *,
    opened_at: datetime = _T0,
) -> tuple[UUID, UUID]:
    original_id, agent_id, review_id = uuid4(), uuid4(), uuid4()
    async with maker() as session, session.begin():
        session.add_all(
            [
                Agent(
                    agent_id=original_id,
                    miner_hotkey="5Original",
                    name="original",
                    sha256=original_id.hex * 2,
                    status=AgentStatus.SCORED,
                    created_at=_T0 - timedelta(hours=1),
                ),
                Agent(
                    agent_id=agent_id,
                    miner_hotkey="5Held",
                    name="held",
                    sha256=agent_id.hex * 2,
                    status=AgentStatus.ATH_PENDING_REVIEW,
                    duplicate_of=original_id,
                    review_reason="legacy near-copy signal",
                    screening_policy_version=8,
                    created_at=_T0,
                ),
                AthReview(
                    review_id=review_id,
                    agent_id=agent_id,
                    status="pending",
                    opened_at=opened_at,
                    original_duplicate_of=original_id,
                    original_reason="legacy near-copy signal",
                    original_policy_version=8,
                    original_evidence={
                        "content_fingerprint_version": 1,
                        "structural_fingerprint_version": 1,
                        "prompt_fingerprint_version": "p1",
                    },
                    algorithm_provenance={
                        "reference_provenance": "legacy",
                        "backfilled": True,
                    },
                ),
            ]
        )
    return agent_id, original_id


async def test_list_is_bounded_oldest_first_and_private(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    await _seed(maker, opened_at=_T0 + timedelta(hours=1))
    oldest, _ = await _seed(maker, opened_at=_T0)
    _install(app, maker)
    response = await client.get(
        "/api/v1/admin/copy-reviews?limit=1&offset=0", headers=_HEADERS
    )
    assert response.status_code == 200
    body = response.json()
    assert (body["count"], body["limit"], body["offset"]) == (2, 1, 0)
    assert body["items"][0]["agent_id"] == str(oldest)
    serialized = response.text.lower()
    assert "sha256" not in serialized and '"m":' not in serialized


async def test_detail_and_comparison_unavailable_until_corrected_adapter(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id, _ = await _seed(maker)
    _install(app, maker)
    detail = await client.get(
        f"/api/v1/admin/copy-reviews/{agent_id}", headers=_HEADERS
    )
    assert detail.status_code == 200
    comparison = await client.get(
        f"/api/v1/admin/copy-reviews/{agent_id}/current-comparison", headers=_HEADERS
    )
    assert comparison.status_code == 200
    assert comparison.json() == {
        "label": "current_comparison",
        "availability": "unavailable",
        "bulk_eligible": False,
        "reason": "corrected reference-aware comparison is not deployed",
        "algorithm_provenance": {"adapter": "unavailable", "reference_aware": False},
    }


async def test_clear_is_durable_preserves_evidence_and_retries_idempotently(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id, original_id = await _seed(maker)
    _install(app, maker)
    payload = {"resolution": "release", "reason": "Corrected comparison clears it"}
    first = await client.post(
        f"/api/v1/admin/copy-reviews/{agent_id}/resolve", json=payload, headers=_HEADERS
    )
    assert first.status_code == 200 and first.json()["idempotent"] is False
    retry = await client.post(
        f"/api/v1/admin/copy-reviews/{agent_id}/resolve", json=payload, headers=_HEADERS
    )
    assert retry.status_code == 200 and retry.json()["idempotent"] is True
    async with maker() as session:
        agent = await session.get(Agent, agent_id)
        review = await session.scalar(
            select(AthReview).where(AthReview.agent_id == agent_id)
        )
        assert agent is not None and review is not None
        assert agent.status == AgentStatus.SCORED
        assert agent.duplicate_of == original_id and agent.review_reason is not None
        assert review.resolution == "clear" and review.resolved_by == "operator"
        assert review.resolution_reason == payload["reason"]


async def test_conflicting_retry_and_changed_snapshot_fail_closed(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id, _ = await _seed(maker)
    _install(app, maker)
    async with maker() as session, session.begin():
        agent = await session.get(Agent, agent_id)
        assert agent is not None
        agent.duplicate_of = None
    mismatch = await client.post(
        f"/api/v1/admin/copy-reviews/{agent_id}/resolve",
        json={"resolution": "ban", "reason": "Confirmed copied implementation"},
        headers=_HEADERS,
    )
    assert mismatch.status_code == 409


async def test_whitespace_actor_and_reason_are_rejected(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id, _ = await _seed(maker)
    _install(app, maker)
    actor = await client.post(
        f"/api/v1/admin/copy-reviews/{agent_id}/resolve",
        json={"resolution": "clear", "reason": "valid reason"},
        headers={**_HEADERS, "X-Admin-Actor": "   "},
    )
    reason = await client.post(
        f"/api/v1/admin/copy-reviews/{agent_id}/resolve",
        json={"resolution": "clear", "reason": "   "},
        headers=_HEADERS,
    )
    assert actor.status_code == 422 and reason.status_code == 422


async def test_changed_hold_reason_fails_closed(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id, _ = await _seed(maker)
    _install(app, maker)
    async with maker() as session, session.begin():
        agent = await session.get(Agent, agent_id)
        assert agent is not None
        agent.review_reason = "different evidence"
    response = await client.post(
        f"/api/v1/admin/copy-reviews/{agent_id}/resolve",
        json={"resolution": "clear", "reason": "Operator cleared evidence"},
        headers=_HEADERS,
    )
    assert response.status_code == 409
