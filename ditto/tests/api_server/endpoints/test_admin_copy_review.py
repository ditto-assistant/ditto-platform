"""Admin copy-review endpoints: list ath_pending_review holds and resolve them."""

from __future__ import annotations

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

from ditto.api_server.dependencies import get_session
from ditto.api_server.fingerprint import _MINHASH_K
from ditto.db.models import Agent, AgentStatus, Base, Score

_ADMIN_TOKEN = "test-admin-token-at-least-32-characters"
_HEADERS = {
    "Authorization": f"Bearer {_ADMIN_TOKEN}",
    "X-Admin-Actor": "backroom:test-operator",
}
_T0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)


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


def _install(app: FastAPI, maker: async_sessionmaker[AsyncSession]) -> None:
    app.state.config = replace(app.state.config, admin_api_token=_ADMIN_TOKEN)

    async def _session() -> AsyncIterator[AsyncSession]:
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = _session


def _sketch(shingles: set[str]) -> dict:
    return {
        "v": 2,
        "k": _MINHASH_K,
        "card": len(shingles),
        "m": sorted(shingles)[:_MINHASH_K],
    }


async def _seed(
    maker: async_sessionmaker[AsyncSession],
    *,
    status: AgentStatus,
    miner: str,
    composite: float | None,
    created_at: datetime,
    content_fingerprint: dict | None = None,
    duplicate_of: UUID | None = None,
    review_reason: str | None = None,
) -> UUID:
    agent_id = uuid4()
    async with maker() as s, s.begin():
        s.add(
            Agent(
                agent_id=agent_id,
                miner_hotkey=miner,
                name=f"agent-{miner[-4:]}",
                sha256=agent_id.hex * 2,
                size_bytes=524288,
                status=status,
                screening_policy_version=8,
                created_at=created_at,
                content_fingerprint=content_fingerprint,
                duplicate_of=duplicate_of,
                review_reason=review_reason,
            )
        )
        if composite is not None:
            s.add(
                Score(
                    agent_id=agent_id,
                    validator_hotkey="5Validator",
                    run_id=f"run-{agent_id.hex[:8]}",
                    seed=42,
                    composite=composite,
                    tool_mean=composite,
                    memory_mean=composite,
                    median_ms=500,
                    n=200,
                    generated_at=created_at,
                )
            )
    return agent_id


class TestListCopyReviews:
    async def test_requires_admin_token(
        self, app: FastAPI, client: httpx.AsyncClient
    ) -> None:
        del app  # fixture builds the application under test
        response = await client.get("/api/v1/admin/copy-reviews")
        assert response.status_code in (401, 503)

    async def test_stale_hold_is_marked_releasable(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # Incumbent and held agent carry DISJOINT novelty fingerprints: the
        # reference-aware gate no longer fires, so the hold (created by the
        # old whole-tarball gate) reads as releasable.
        incumbent = await _seed(
            session_maker,
            status=AgentStatus.SCORED,
            miner="5Incumbent",
            composite=0.80,
            created_at=_T0,
            content_fingerprint=_sketch({f"a{i:015x}" for i in range(30)}),
        )
        held = await _seed(
            session_maker,
            status=AgentStatus.ATH_PENDING_REVIEW,
            miner="5Challenger",
            composite=0.801,
            created_at=_T0 + timedelta(hours=1),
            content_fingerprint=_sketch({f"b{i:015x}" for i in range(30)}),
            duplicate_of=incumbent,
            review_reason="content near-duplicate of agent ... (legacy hold)",
        )
        _install(app, session_maker)
        response = await client.get("/api/v1/admin/copy-reviews", headers=_HEADERS)
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 1 and body["would_release_count"] == 1
        (item,) = body["items"]
        assert item["agent_id"] == str(held)
        assert item["stored_duplicate_of"] == str(incumbent)
        assert item["would_release"] is True
        assert item["recomputed"]["held"] is False
        assert item["median_composite"] == pytest.approx(0.801)
        # Pair similarity against the originally matched agent is surfaced.
        assert item["pair_similarity"]["lexical_jaccard"] == pytest.approx(0.0)

    async def test_refiring_hold_stays_held(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        stolen = {f"a{i:015x}" for i in range(30)}
        incumbent = await _seed(
            session_maker,
            status=AgentStatus.SCORED,
            miner="5Incumbent",
            composite=0.80,
            created_at=_T0,
            content_fingerprint=_sketch(stolen),
        )
        await _seed(
            session_maker,
            status=AgentStatus.ATH_PENDING_REVIEW,
            miner="5Copier",
            composite=0.802,
            created_at=_T0 + timedelta(hours=1),
            content_fingerprint=_sketch(stolen),
            duplicate_of=incumbent,
            review_reason="content near-duplicate",
        )
        _install(app, session_maker)
        response = await client.get("/api/v1/admin/copy-reviews", headers=_HEADERS)
        assert response.status_code == 200
        body = response.json()
        assert body["would_release_count"] == 0
        (item,) = body["items"]
        assert item["would_release"] is False
        assert item["recomputed"]["held"] is True
        assert item["recomputed"]["duplicate_of"] == str(incumbent)

    async def test_hold_without_scores_requires_manual_review(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed(
            session_maker,
            status=AgentStatus.ATH_PENDING_REVIEW,
            miner="5Challenger",
            composite=None,
            created_at=_T0,
        )
        _install(app, session_maker)
        response = await client.get("/api/v1/admin/copy-reviews", headers=_HEADERS)
        assert response.status_code == 200
        (item,) = response.json()["items"]
        assert item["would_release"] is False
        assert "manual review" in item["recomputed"]["reason"]


class TestResolveCopyReview:
    async def test_release_returns_agent_to_scored(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        held = await _seed(
            session_maker,
            status=AgentStatus.ATH_PENDING_REVIEW,
            miner="5Challenger",
            composite=0.8,
            created_at=_T0,
            duplicate_of=None,
            review_reason="legacy hold",
        )
        _install(app, session_maker)
        response = await client.post(
            f"/api/v1/admin/copy-reviews/{held}/resolve",
            headers=_HEADERS,
            json={"resolution": "release", "reason": "reference-aware gate clears"},
        )
        assert response.status_code == 200
        assert response.json()["agent_status"] == "scored"
        async with session_maker() as s:
            agent = await s.get(Agent, held)
            assert agent is not None
            assert agent.status == AgentStatus.SCORED
            assert agent.review_reason is None

        # A second resolution of the same agent conflicts.
        again = await client.post(
            f"/api/v1/admin/copy-reviews/{held}/resolve",
            headers=_HEADERS,
            json={"resolution": "release", "reason": "double submit"},
        )
        assert again.status_code == 409

    async def test_ban_preserves_the_moderation_record(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        held = await _seed(
            session_maker,
            status=AgentStatus.ATH_PENDING_REVIEW,
            miner="5Copier",
            composite=0.8,
            created_at=_T0,
            review_reason="content near-duplicate of agent X",
        )
        _install(app, session_maker)
        response = await client.post(
            f"/api/v1/admin/copy-reviews/{held}/resolve",
            headers=_HEADERS,
            json={"resolution": "ban", "reason": "confirmed copy on review"},
        )
        assert response.status_code == 200
        assert response.json()["agent_status"] == "banned"
        async with session_maker() as s:
            agent = await s.get(Agent, held)
            assert agent is not None
            assert agent.status == AgentStatus.BANNED
            assert agent.review_reason == "content near-duplicate of agent X"

    async def test_actor_header_is_required(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        held = await _seed(
            session_maker,
            status=AgentStatus.ATH_PENDING_REVIEW,
            miner="5Challenger",
            composite=0.8,
            created_at=_T0,
        )
        _install(app, session_maker)
        response = await client.post(
            f"/api/v1/admin/copy-reviews/{held}/resolve",
            headers={"Authorization": f"Bearer {_ADMIN_TOKEN}"},
            json={"resolution": "release", "reason": "no actor supplied"},
        )
        assert response.status_code == 422

    async def test_unknown_agent_is_404(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install(app, session_maker)
        response = await client.post(
            f"/api/v1/admin/copy-reviews/{uuid4()}/resolve",
            headers=_HEADERS,
            json={"resolution": "release", "reason": "missing row"},
        )
        assert response.status_code == 404
