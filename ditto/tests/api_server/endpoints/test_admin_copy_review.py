"""Durable ATH copy-review API regression coverage."""

import gzip
import hashlib
import io
import tarfile
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
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

from ditto.api_server.dependencies import get_session, get_storage_client
from ditto.api_server.fingerprint import reference_corpus_provenance
from ditto.api_server.storage import ObjectDownloadFailedError
from ditto.db.models import Agent, AgentStatus, AthReview, AthReviewAction, Base, Score
from ditto.db.queries.scores import list_eligible_ledger

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


async def _seed_scored_agent(
    maker: async_sessionmaker[AsyncSession],
    *,
    score_count: int = 3,
    status: AgentStatus = AgentStatus.SCORED,
) -> tuple[UUID, str]:
    agent_id = uuid4()
    sha256 = agent_id.hex * 2
    async with maker() as session, session.begin():
        session.add(
            Agent(
                agent_id=agent_id,
                miner_hotkey="5Benchmax",
                name="benchmax",
                sha256=sha256,
                status=status,
                screening_policy_version=8,
                created_at=_T0,
            )
        )
        for index in range(score_count):
            session.add(
                Score(
                    agent_id=agent_id,
                    validator_hotkey=f"validator-{index}",
                    run_id=f"manual-hold-run-{index}",
                    signature=None,
                    seed=7,
                    composite=0.97,
                    tool_mean=0.97,
                    memory_mean=0.90,
                    median_ms=100,
                    n=114,
                    details={"bench_version": 2},
                    generated_at=_T0 + timedelta(minutes=index),
                )
            )
    return agent_id, sha256


async def test_manual_open_holds_exact_scored_artifact_and_removes_it_from_ledger(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id, sha256 = await _seed_scored_agent(maker)
    _install(app, maker)
    payload = {
        "expected_sha256": sha256,
        "expected_score_count": 3,
        "reason": "Deterministic benchmark-family routing requires ATH review",
    }

    response = await client.post(
        f"/api/v1/admin/copy-reviews/{agent_id}/open",
        json=payload,
        headers=_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["agent_status"] == AgentStatus.ATH_PENDING_REVIEW
    assert body["idempotent"] is False
    assert body["review"]["original"]["review_kind"] == "benchmark_overfit"
    assert body["review"]["original"]["duplicate_of"] is None
    retry = await client.post(
        f"/api/v1/admin/copy-reviews/{agent_id}/open",
        json=payload,
        headers=_HEADERS,
    )
    assert retry.status_code == 200 and retry.json()["idempotent"] is True

    audit = await client.get(
        f"/api/v1/admin/copy-reviews/{agent_id}/audit", headers=_HEADERS
    )
    assert audit.status_code == 200
    audit_body = audit.json()
    assert audit_body["review"]["review_id"] == body["review"]["review_id"]
    assert audit_body["review"]["original"]["review_kind"] == "benchmark_overfit"
    assert audit_body["review"]["original"]["reason"] == payload["reason"]
    assert {key: value for key, value in audit_body.items() if key != "review"} == {
        "agent_status": AgentStatus.ATH_PENDING_REVIEW,
        "held_artifact_sha256": sha256,
        "held_score_count": 3,
        "previous_status": AgentStatus.SCORED,
        "opened_by": "operator",
        "action_history": [],
    }

    async with maker() as session:
        agent = await session.get(Agent, agent_id)
        review = await session.scalar(
            select(AthReview).where(AthReview.agent_id == agent_id)
        )
        ledger = await list_eligible_ledger(session)
        assert agent is not None and review is not None
        assert agent.status == AgentStatus.ATH_PENDING_REVIEW
        assert agent.review_reason == payload["reason"]
        assert review.original_evidence["sha256"] == sha256
        assert review.original_evidence["score_count"] == 3
        assert review.algorithm_provenance["opened_by"] == "operator"
        assert all(row.agent_id != agent_id for row in ledger)


@pytest.mark.parametrize(
    ("field", "value", "detail"),
    [
        ("expected_sha256", "0" * 64, "artifact sha256 changed"),
        ("expected_score_count", 2, "score count changed"),
    ],
)
async def test_manual_open_rejects_stale_identity_guards(
    app: FastAPI,
    client: httpx.AsyncClient,
    maker: async_sessionmaker[AsyncSession],
    field: str,
    value: str | int,
    detail: str,
) -> None:
    agent_id, sha256 = await _seed_scored_agent(maker)
    _install(app, maker)
    payload: dict[str, object] = {
        "expected_sha256": sha256,
        "expected_score_count": 3,
        "reason": "Manual benchmark-overfit review",
    }
    payload[field] = value

    response = await client.post(
        f"/api/v1/admin/copy-reviews/{agent_id}/open",
        json=payload,
        headers=_HEADERS,
    )

    assert response.status_code == 409
    assert detail in response.text
    async with maker() as session:
        agent = await session.get(Agent, agent_id)
        assert agent is not None and agent.status == AgentStatus.SCORED
        assert (
            await session.scalar(
                select(AthReview).where(AthReview.agent_id == agent_id)
            )
            is None
        )


async def test_clearing_manual_hold_restores_live_status(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id, sha256 = await _seed_scored_agent(maker, status=AgentStatus.LIVE)
    _install(app, maker)
    opened = await client.post(
        f"/api/v1/admin/copy-reviews/{agent_id}/open",
        json={
            "expected_sha256": sha256,
            "expected_score_count": 3,
            "reason": "Manual benchmark-overfit review",
        },
        headers=_HEADERS,
    )
    assert opened.status_code == 200

    resolved = await client.post(
        f"/api/v1/admin/copy-reviews/{agent_id}/resolve",
        json={"resolution": "clear", "reason": "General behavior confirmed"},
        headers=_HEADERS,
    )

    assert resolved.status_code == 200
    assert resolved.json()["agent_status"] == AgentStatus.LIVE


async def test_resolved_review_reopens_without_rewriting_original_evidence(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id, sha256 = await _seed_scored_agent(maker, status=AgentStatus.LIVE)
    _install(app, maker)
    first_reason = "Deterministic benchmark-family routing"
    opened = await client.post(
        f"/api/v1/admin/copy-reviews/{agent_id}/open",
        json={
            "expected_sha256": sha256,
            "expected_score_count": 3,
            "reason": first_reason,
        },
        headers=_HEADERS,
    )
    assert opened.status_code == 200
    review_id = UUID(opened.json()["review"]["review_id"])
    initial_opened_at = datetime.fromisoformat(opened.json()["review"]["opened_at"])
    assert opened.json()["reopened"] is False
    cleared = await client.post(
        f"/api/v1/admin/copy-reviews/{agent_id}/resolve",
        json={"resolution": "clear", "reason": "Initial source review cleared it"},
        headers=_HEADERS,
    )
    assert cleared.status_code == 200

    reopen_payload = {
        "expected_sha256": sha256,
        "expected_score_count": 3,
        "reason": "New benchmark-overfit evidence requires another review",
    }
    reopened = await client.post(
        f"/api/v1/admin/copy-reviews/{agent_id}/open",
        json=reopen_payload,
        headers=_HEADERS,
    )
    assert reopened.status_code == 200
    assert reopened.json()["reopened"] is True
    assert reopened.json()["idempotent"] is False
    assert reopened.json()["review"]["review_id"] == str(review_id)
    assert reopened.json()["review"]["original"]["reason"] == first_reason
    retry = await client.post(
        f"/api/v1/admin/copy-reviews/{agent_id}/open",
        json=reopen_payload,
        headers=_HEADERS,
    )
    assert retry.status_code == 200
    assert retry.json()["idempotent"] is True
    assert retry.json()["reopened"] is True

    audit = await client.get(
        f"/api/v1/admin/copy-reviews/{agent_id}/audit", headers=_HEADERS
    )
    assert audit.status_code == 200
    assert [event["action"] for event in audit.json()["action_history"]] == [
        "clear",
        "reopen",
    ]
    assert audit.json()["action_history"][1] == {
        "action": "reopen",
        "reason": reopen_payload["reason"],
        "actor": "operator",
        "created_at": audit.json()["action_history"][1]["created_at"],
        "previous_status": "live",
        "artifact_sha256": sha256,
        "score_count": 3,
    }

    recleared = await client.post(
        f"/api/v1/admin/copy-reviews/{agent_id}/resolve",
        json={"resolution": "clear", "reason": "Second review also cleared it"},
        headers=_HEADERS,
    )
    assert recleared.status_code == 200
    assert recleared.json()["agent_status"] == AgentStatus.LIVE
    async with maker() as session:
        review = await session.scalar(
            select(AthReview).where(AthReview.agent_id == agent_id)
        )
        actions = list(
            await session.scalars(
                select(AthReviewAction).where(AthReviewAction.review_id == review_id)
            )
        )
        assert review is not None and review.original_reason == first_reason
        assert review.opened_at.replace(tzinfo=UTC) == initial_opened_at
        assert review.reopened_at is not None and review.reopened_at > review.opened_at
        assert len(actions) == 3


async def test_reopen_still_fails_closed_on_changed_score_count(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id, sha256 = await _seed_scored_agent(maker)
    _install(app, maker)
    opened = await client.post(
        f"/api/v1/admin/copy-reviews/{agent_id}/open",
        json={
            "expected_sha256": sha256,
            "expected_score_count": 3,
            "reason": "Manual benchmark-overfit review",
        },
        headers=_HEADERS,
    )
    assert opened.status_code == 200
    cleared = await client.post(
        f"/api/v1/admin/copy-reviews/{agent_id}/resolve",
        json={"resolution": "clear", "reason": "Initial evidence was clear"},
        headers=_HEADERS,
    )
    assert cleared.status_code == 200
    response = await client.post(
        f"/api/v1/admin/copy-reviews/{agent_id}/open",
        json={
            "expected_sha256": sha256,
            "expected_score_count": 2,
            "reason": "New evidence requires another review",
        },
        headers=_HEADERS,
    )
    assert response.status_code == 409
    assert "score count changed" in response.text


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


async def test_original_evidence_names_the_matched_submission(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id, original_id = await _seed(maker)
    _install(app, maker)
    listing = await client.get("/api/v1/admin/copy-reviews", headers=_HEADERS)
    assert listing.status_code == 200
    original = listing.json()["items"][0]["original"]
    assert original["duplicate_of"] == str(original_id)
    assert original["duplicate_of_name"] == "original"
    assert original["duplicate_of_hotkey"] == "5Original"
    assert original["duplicate_of_submitted_at"] is not None
    detail = await client.get(
        f"/api/v1/admin/copy-reviews/{agent_id}", headers=_HEADERS
    )
    assert detail.status_code == 200
    assert detail.json()["original"]["duplicate_of_name"] == "original"


async def test_matched_identity_is_null_when_reference_row_is_gone(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id, _ = await _seed(maker)
    async with maker() as session, session.begin():
        review = (
            await session.execute(
                select(AthReview).where(AthReview.agent_id == agent_id)
            )
        ).scalar_one()
        review.original_duplicate_of = None
    _install(app, maker)
    detail = await client.get(
        f"/api/v1/admin/copy-reviews/{agent_id}", headers=_HEADERS
    )
    assert detail.status_code == 200
    original = detail.json()["original"]
    assert original["duplicate_of"] is None
    assert original["duplicate_of_name"] is None
    assert original["duplicate_of_hotkey"] is None


async def test_list_can_embed_current_comparisons_in_one_request(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    comparable, _ = await _seed_current_comparison(maker)
    scoreless, _ = await _seed(maker, opened_at=_T0 + timedelta(hours=1))
    _install(app, maker)
    response = await client.get(
        "/api/v1/admin/copy-reviews?include=current_comparison", headers=_HEADERS
    )
    assert response.status_code == 200
    by_agent = {item["agent_id"]: item for item in response.json()["items"]}
    embedded = by_agent[str(comparable)]["current_comparison"]
    assert embedded["availability"] == "available"
    assert embedded["algorithm_version"] == "reference-aware-v2"
    assert embedded["current_decision"] in {"clear", "hold", "inconclusive_review"}
    # A row the dedicated endpoint would 409 embeds the fail-closed state.
    failed_closed = by_agent[str(scoreless)]["current_comparison"]
    assert failed_closed == {
        "availability": "unavailable",
        "bulk_eligible": False,
        "reason": "current comparison unavailable",
    }
    # No fingerprint material leaks through the embedded form either.
    serialized = response.text.lower()
    assert "sha256" not in serialized and '"m":' not in serialized


async def test_list_without_include_omits_comparisons(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    await _seed_current_comparison(maker)
    _install(app, maker)
    response = await client.get("/api/v1/admin/copy-reviews", headers=_HEADERS)
    assert response.status_code == 200
    assert response.json()["items"][0]["current_comparison"] is None


async def test_embedded_comparison_matches_the_dedicated_endpoint(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id, _ = await _seed_current_comparison(maker)
    _install(app, maker)
    dedicated = await client.get(
        f"/api/v1/admin/copy-reviews/{agent_id}/current-comparison", headers=_HEADERS
    )
    listing = await client.get(
        "/api/v1/admin/copy-reviews?include=current_comparison", headers=_HEADERS
    )
    assert dedicated.status_code == 200 and listing.status_code == 200
    embedded = listing.json()["items"][0]["current_comparison"]
    assert embedded == dedicated.json()


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


def _tarball(files: dict[str, str]) -> bytes:
    """Build a gzip tarball of ``path -> text`` files, like an agent artifact."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for path, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name=path)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return gzip.compress(raw.getvalue())


async def _seed_diff_pair(
    maker: async_sessionmaker[AsyncSession],
    candidate_files: dict[str, str],
    reference_files: dict[str, str],
) -> tuple[UUID, UUID, dict[str, bytes]]:
    """Seed a held/reference pair whose sha256 match real tarball bytes.

    Returns the ids plus the ``key -> tar bytes`` map the storage stub serves.
    """
    candidate_tar = _tarball(candidate_files)
    reference_tar = _tarball(reference_files)
    reference_id, candidate_id, review_id = uuid4(), uuid4(), uuid4()
    objects = {
        f"{candidate_id}/agent.tar.gz": candidate_tar,
        f"{reference_id}/agent.tar.gz": reference_tar,
    }
    async with maker() as session, session.begin():
        session.add_all(
            [
                Agent(
                    agent_id=reference_id,
                    miner_hotkey="5Original",
                    name="original",
                    sha256=hashlib.sha256(reference_tar).hexdigest(),
                    status=AgentStatus.SCORED,
                    created_at=_T0 - timedelta(hours=1),
                ),
                Agent(
                    agent_id=candidate_id,
                    miner_hotkey="5Held",
                    name="held",
                    sha256=hashlib.sha256(candidate_tar).hexdigest(),
                    status=AgentStatus.ATH_PENDING_REVIEW,
                    duplicate_of=reference_id,
                    review_reason="near-copy signal",
                    screening_policy_version=8,
                    created_at=_T0,
                ),
                AthReview(
                    review_id=review_id,
                    agent_id=candidate_id,
                    status="pending",
                    opened_at=_T0,
                    original_duplicate_of=reference_id,
                    original_reason="near-copy signal",
                    original_policy_version=8,
                    original_evidence={},
                    algorithm_provenance={},
                ),
            ]
        )
    return candidate_id, reference_id, objects


def _install_storage(app: FastAPI, objects: dict[str, bytes]) -> None:
    storage = MagicMock()

    async def _get_object(*, key: str, max_bytes: int) -> bytes:
        del max_bytes
        if key not in objects:
            raise ObjectDownloadFailedError(key)
        return objects[key]

    storage.get_object = AsyncMock(side_effect=_get_object)

    async def _fake_storage() -> MagicMock:
        return storage

    app.dependency_overrides[get_storage_client] = _fake_storage


async def test_source_diff_manifest_classifies_files(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    candidate_id, reference_id, objects = await _seed_diff_pair(
        maker,
        candidate_files={
            "src/main.rs": "fn main() {}\n",
            "src/util.rs": "fn util() -> i32 { 1 }\n",
            "src/new.rs": "fn extra() {}\n",
        },
        reference_files={
            "src/main.rs": "fn main() {}\n",
            "src/util.rs": "fn util() -> i32 { 2 }\n",
            "src/gone.rs": "fn gone() {}\n",
        },
    )
    _install(app, maker)
    _install_storage(app, objects)
    response = await client.get(
        f"/api/v1/admin/copy-reviews/{candidate_id}/source-diff", headers=_HEADERS
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reference_agent_id"] == str(reference_id)
    assert (body["identical_count"], body["modified_count"]) == (1, 1)
    assert (body["added_count"], body["removed_count"]) == (1, 1)
    by_path = {entry["path"]: entry for entry in body["files"]}
    assert by_path["src/main.rs"]["status"] == "identical"
    assert by_path["src/util.rs"]["status"] == "modified"
    assert by_path["src/new.rs"]["status"] == "added"
    assert by_path["src/gone.rs"]["status"] == "removed"


async def test_source_diff_file_returns_unified_body(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    candidate_id, _, objects = await _seed_diff_pair(
        maker,
        candidate_files={"src/util.rs": "fn util() -> i32 { 1 }\n"},
        reference_files={"src/util.rs": "fn util() -> i32 { 2 }\n"},
    )
    _install(app, maker)
    _install_storage(app, objects)
    response = await client.get(
        f"/api/v1/admin/copy-reviews/{candidate_id}/source-diff/file",
        params={"path": "src/util.rs"},
        headers=_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["candidate_present"] and body["reference_present"]
    joined = "\n".join(body["diff_lines"])
    assert "{ 2 }" in joined and "{ 1 }" in joined


async def test_source_diff_requires_admin_actor(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    candidate_id, _, objects = await _seed_diff_pair(
        maker, {"a.rs": "x\n"}, {"a.rs": "y\n"}
    )
    _install(app, maker)
    _install_storage(app, objects)
    response = await client.get(
        f"/api/v1/admin/copy-reviews/{candidate_id}/source-diff",
        headers={"Authorization": f"Bearer {_TOKEN}"},
    )
    assert response.status_code == 422


async def test_source_diff_missing_file_is_404(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    candidate_id, _, objects = await _seed_diff_pair(
        maker, {"a.rs": "x\n"}, {"a.rs": "y\n"}
    )
    _install(app, maker)
    _install_storage(app, objects)
    response = await client.get(
        f"/api/v1/admin/copy-reviews/{candidate_id}/source-diff/file",
        params={"path": "ghost.rs"},
        headers=_HEADERS,
    )
    assert response.status_code == 404


async def test_source_diff_digest_mismatch_is_502(
    app: FastAPI, client: httpx.AsyncClient, maker: async_sessionmaker[AsyncSession]
) -> None:
    candidate_id, _, objects = await _seed_diff_pair(
        maker, {"a.rs": "x\n"}, {"a.rs": "y\n"}
    )
    # Corrupt the stored candidate bytes so they no longer match agent.sha256.
    objects[f"{candidate_id}/agent.tar.gz"] = _tarball({"a.rs": "tampered\n"})
    _install(app, maker)
    _install_storage(app, objects)
    response = await client.get(
        f"/api/v1/admin/copy-reviews/{candidate_id}/source-diff", headers=_HEADERS
    )
    assert response.status_code == 502
