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
from ditto.api_server.fingerprint import reference_corpus_provenance
from ditto.db.models import Agent, AgentStatus, AthReview, Base, Score

_TOKEN = "test-admin-token-at-least-32-characters"
_HEADERS = {"Authorization": f"Bearer {_TOKEN}", "X-Admin-Actor": "operator"}
_T0 = datetime(2026, 7, 16, 12, tzinfo=UTC)
_CORPUS_ID = reference_corpus_provenance()["corpus_id"]


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


def _fingerprint(prefix: str, *, corpus: str = _CORPUS_ID) -> dict:
    values = [f"{prefix}{i:015x}" for i in range(12)]
    return {"v": 2, "corpus": corpus, "k": 256, "card": 12, "m": values}


async def _add_finalized_scores(
    maker: async_sessionmaker[AsyncSession], *, agent_ids: tuple[UUID, UUID]
) -> None:
    async with maker() as session, session.begin():
        for agent_id in agent_ids:
            for index, composite in enumerate((0.79, 0.80, 0.81)):
                session.add(
                    Score(
                        agent_id=agent_id,
                        validator_hotkey=f"validator-{index}",
                        run_id=f"run-{index}",
                        signature=None,
                        seed=7,
                        composite=composite,
                        tool_mean=composite,
                        memory_mean=composite,
                        median_ms=100 + index,
                        n=114,
                        details={"bench_version": 2},
                        generated_at=_T0 + timedelta(minutes=index),
                    )
                )


async def _seed_current_comparison(
    maker: async_sessionmaker[AsyncSession],
    *,
    reference_corpus: str = _CORPUS_ID,
) -> tuple[UUID, UUID]:
    agent_id, original_id = await _seed(maker)
    async with maker() as session, session.begin():
        candidate = await session.get(Agent, agent_id)
        reference = await session.get(Agent, original_id)
        assert candidate is not None and reference is not None
        candidate.content_fingerprint = _fingerprint("c")
        reference.content_fingerprint = _fingerprint("r", corpus=reference_corpus)
        candidate.size_bytes = 500_001
        reference.size_bytes = 500_000
        # The endpoint must use AthReview.original_duplicate_of, not this mutable
        # field, when it reconstructs current comparison evidence.
        candidate.duplicate_of = None
    await _add_finalized_scores(maker, agent_ids=(agent_id, original_id))
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


async def test_current_comparison_is_unavailable_without_finalized_scores(
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
    assert comparison.status_code == 409
    assert "current comparison unavailable" in comparison.text


async def test_current_comparison_returns_only_corrected_aggregate_wire(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id, _ = await _seed_current_comparison(maker)
    _install(app, maker)
    response = await client.get(
        f"/api/v1/admin/copy-reviews/{agent_id}/current-comparison", headers=_HEADERS
    )

    assert response.status_code == 200
    body = response.json()
    assert body["availability"] == "available"
    assert body["bulk_eligible"] is True
    assert body["current_decision"] == "clear"
    assert body["chronology_direction"] == "reference_earlier"
    assert body["lexical"]["candidate_cardinality"] == 12
    serialized = response.text.lower()
    for forbidden in (
        "sha256",
        "normalized_source_hash",
        "artifact",
        "source_path",
        '"m"',
        "credential",
    ):
        assert forbidden not in serialized
    async with maker() as session:
        agent = await session.get(Agent, agent_id)
        review = await session.scalar(
            select(AthReview).where(AthReview.agent_id == agent_id)
        )
        assert agent is not None and review is not None
        assert agent.status == AgentStatus.ATH_PENDING_REVIEW
        assert agent.duplicate_of is None
        assert review.algorithm_provenance == {
            "reference_provenance": "legacy",
            "backfilled": True,
        }


async def test_incompatible_current_comparison_is_never_bulk_eligible(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id, _ = await _seed_current_comparison(maker, reference_corpus="older-corpus")
    _install(app, maker)
    response = await client.get(
        f"/api/v1/admin/copy-reviews/{agent_id}/current-comparison", headers=_HEADERS
    )

    assert response.status_code == 200
    assert response.json()["current_decision"] == "inconclusive_review"
    assert response.json()["bulk_eligible"] is False


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
