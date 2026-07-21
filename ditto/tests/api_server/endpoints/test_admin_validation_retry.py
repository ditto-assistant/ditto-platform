"""Regression coverage for audited validator-infrastructure recovery."""

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

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.api_models.ticket_status import TicketStatus
from ditto.api_server.dependencies import get_session
from ditto.db.models import (
    Agent,
    Base,
    Score,
    ScoreAuditEntry,
    ValidatorRetryRecovery,
    ValidatorTicket,
)
from ditto.db.queries.audit import (
    EVENT_SCORE_RETEST_RELEASED,
    EVENT_SCORE_RETEST_REQUESTED,
)
from ditto.db.queries.benchmark_rollout import DEFAULT_BENCH_VERSION
from ditto.db.queries.tickets import issue_ticket

_TOKEN = "test-admin-token-at-least-32-characters"
_HEADERS = {"Authorization": f"Bearer {_TOKEN}", "X-Admin-Actor": "operator"}
_T0 = datetime(2026, 7, 18, 12, tzinfo=UTC)
# Robust against the CI wall clock the endpoint reads via datetime.now(UTC):
# _PAST is always behind it, _FUTURE always ahead.
_PAST = _T0 - timedelta(hours=1)
_FUTURE = datetime(2099, 1, 1, tzinfo=UTC)


@pytest.fixture
async def retry_engine() -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def _fk(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def retry_maker(
    retry_engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(retry_engine, expire_on_commit=False)


def _install(app: FastAPI, maker: async_sessionmaker[AsyncSession]) -> None:
    app.state.config = replace(app.state.config, admin_api_token=_TOKEN)

    async def _session() -> AsyncIterator[AsyncSession]:
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = _session


async def _seed(
    maker: async_sessionmaker[AsyncSession],
    *,
    score_count: int = 0,
    bench_version: int = DEFAULT_BENCH_VERSION,
    composites: list[float] | None = None,
) -> UUID:
    agent_id = uuid4()
    async with maker() as session, session.begin():
        session.add(
            Agent(
                agent_id=agent_id,
                miner_hotkey="5Miner",
                name="valid-agent",
                version=1,
                sha256=agent_id.hex * 2,
                status=AgentStatus.EVALUATING,
                screening_policy_version=SCREENING_POLICY_VERSION,
                created_at=_T0 - timedelta(days=1),
            )
        )
        for index in range(4):
            hotkey = f"validator-{index}"
            status = (
                TicketStatus.SCORED if index < score_count else TicketStatus.EXPIRED
            )
            session.add(
                ValidatorTicket(
                    agent_id=agent_id,
                    validator_hotkey=hotkey,
                    status=status,
                    issued_at=_T0 - timedelta(hours=3 - index / 10),
                    deadline=_T0 - timedelta(hours=2 - index / 10),
                    bench_version=bench_version,
                    attempt_count=1 if status == TicketStatus.SCORED else 2,
                    manual_retry_grants=0,
                    retry_after=_T0 - timedelta(hours=1),
                )
            )
            if status == TicketStatus.SCORED:
                session.add(
                    Score(
                        agent_id=agent_id,
                        bench_version=bench_version,
                        validator_hotkey=hotkey,
                        run_id=f"run-{index}",
                        seed=7,
                        composite=(composites or [0.7] * score_count)[index],
                        tool_mean=0.7,
                        memory_mean=0.7,
                        median_ms=100,
                        n=114,
                        generated_at=_T0 - timedelta(hours=1),
                    )
                )
    return agent_id


async def test_retry_is_bound_to_v3_ticket_and_score_epoch(
    app: FastAPI,
    client: httpx.AsyncClient,
    retry_maker: async_sessionmaker[AsyncSession],
) -> None:
    agent_id = await _seed(retry_maker, score_count=1, bench_version=3)
    _install(app, retry_maker)

    detail = await client.get(
        f"/api/v1/admin/validation-retries/{agent_id}", headers=_HEADERS
    )
    assert detail.status_code == 200, detail.text
    assert {ticket["bench_version"] for ticket in detail.json()["tickets"]} == {3}

    response = await client.post(
        f"/api/v1/admin/validation-retries/{agent_id}/retry",
        headers=_HEADERS,
        json={
            "request_id": str(uuid4()),
            "expected_snapshot": detail.json()["snapshot"],
            "reason": "v3 validator infrastructure recovery",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["recovery"]["bench_version"] == 3


async def test_retry_grants_only_minimum_quorum_slots_and_preserves_history(
    app: FastAPI,
    client: httpx.AsyncClient,
    retry_maker: async_sessionmaker[AsyncSession],
) -> None:
    agent_id = await _seed(retry_maker)
    _install(app, retry_maker)

    detail = await client.get(
        f"/api/v1/admin/validation-retries/{agent_id}", headers=_HEADERS
    )
    assert detail.status_code == 200
    assert detail.json()["recovery_allowed"] is True
    assert all(item["retry_budget_exhausted"] for item in detail.json()["tickets"])

    request_id = uuid4()
    payload = {
        "request_id": str(request_id),
        "expected_snapshot": detail.json()["snapshot"],
        "reason": "Sandbox OOM and writable-storage exhaustion verified by operator",
    }
    response = await client.post(
        f"/api/v1/admin/validation-retries/{agent_id}/retry",
        json=payload,
        headers=_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["idempotent"] is False
    assert body["recovery"]["granted_validator_hotkeys"] == [
        "validator-0",
        "validator-1",
        "validator-2",
    ]
    retry = await client.post(
        f"/api/v1/admin/validation-retries/{agent_id}/retry",
        json=payload,
        headers=_HEADERS,
    )
    assert retry.status_code == 200 and retry.json()["idempotent"] is True

    async with retry_maker() as session:
        agent = await session.get(Agent, agent_id)
        tickets = list(
            (
                await session.scalars(
                    select(ValidatorTicket)
                    .where(ValidatorTicket.agent_id == agent_id)
                    .order_by(ValidatorTicket.validator_hotkey)
                )
            ).all()
        )
        actions = list(
            (
                await session.scalars(
                    select(ValidatorRetryRecovery).where(
                        ValidatorRetryRecovery.agent_id == agent_id
                    )
                )
            ).all()
        )
    assert agent is not None and agent.status == AgentStatus.EVALUATING
    assert [ticket.attempt_count for ticket in tickets] == [2, 2, 2, 2]
    assert [ticket.manual_retry_grants for ticket in tickets] == [1, 1, 1, 0]
    assert len(actions) == 1
    assert len(actions[0].ticket_snapshot) == 4


async def test_one_score_grants_only_two_more_attempts_and_keeps_score(
    app: FastAPI,
    client: httpx.AsyncClient,
    retry_maker: async_sessionmaker[AsyncSession],
) -> None:
    agent_id = await _seed(retry_maker, score_count=1)
    _install(app, retry_maker)
    detail = await client.get(
        f"/api/v1/admin/validation-retries/{agent_id}", headers=_HEADERS
    )
    response = await client.post(
        f"/api/v1/admin/validation-retries/{agent_id}/retry",
        json={
            "request_id": str(uuid4()),
            "expected_snapshot": detail.json()["snapshot"],
            "reason": "Validator container loss corroborated by runtime exit evidence",
        },
        headers=_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["recovery"]["granted_validator_hotkeys"] == [
        "validator-1",
        "validator-2",
    ]
    async with retry_maker() as session:
        score_total = await session.scalar(
            select(Score).where(Score.agent_id == agent_id)
        )
    assert score_total is not None


async def test_stale_snapshot_and_active_or_natural_retry_fail_closed(
    app: FastAPI,
    client: httpx.AsyncClient,
    retry_maker: async_sessionmaker[AsyncSession],
) -> None:
    agent_id = await _seed(retry_maker)
    _install(app, retry_maker)
    await client.get(f"/api/v1/admin/validation-retries/{agent_id}", headers=_HEADERS)
    stale = await client.post(
        f"/api/v1/admin/validation-retries/{agent_id}/retry",
        json={
            "request_id": str(uuid4()),
            "expected_snapshot": "0" * 64,
            "reason": "Verified validator infrastructure failure",
        },
        headers=_HEADERS,
    )
    assert stale.status_code == 409

    async with retry_maker() as session, session.begin():
        ticket = await session.get(ValidatorTicket, (agent_id, 2, "validator-0"))
        assert ticket is not None
        ticket.attempt_count = 1
    changed = await client.get(
        f"/api/v1/admin/validation-retries/{agent_id}", headers=_HEADERS
    )
    assert changed.json()["automatic_retry_available"] is True
    assert changed.json()["recovery_allowed"] is False
    denied = await client.post(
        f"/api/v1/admin/validation-retries/{agent_id}/retry",
        json={
            "request_id": str(uuid4()),
            "expected_snapshot": changed.json()["snapshot"],
            "reason": "Verified validator infrastructure failure",
        },
        headers=_HEADERS,
    )
    assert denied.status_code == 409


async def test_manual_grant_allows_exactly_one_more_same_version_issue(
    app: FastAPI,
    client: httpx.AsyncClient,
    retry_maker: async_sessionmaker[AsyncSession],
) -> None:
    agent_id = await _seed(retry_maker)
    _install(app, retry_maker)
    detail = await client.get(
        f"/api/v1/admin/validation-retries/{agent_id}", headers=_HEADERS
    )
    await client.post(
        f"/api/v1/admin/validation-retries/{agent_id}/retry",
        json={
            "request_id": str(uuid4()),
            "expected_snapshot": detail.json()["snapshot"],
            "reason": "Verified validator infrastructure failure",
        },
        headers=_HEADERS,
    )
    async with retry_maker() as session, session.begin():
        ticket = await issue_ticket(
            session,
            validator_hotkey="validator-0",
            now=datetime.now(UTC) + timedelta(seconds=1),
            ttl=timedelta(minutes=90),
            bench_version=DEFAULT_BENCH_VERSION,
        )
    assert ticket is not None and ticket.agent_id == agent_id
    assert ticket.attempt_count == 3
    assert ticket.manual_retry_grants == 1


async def test_replace_one_validators_score_and_reissue_same_ticket(
    app: FastAPI,
    client: httpx.AsyncClient,
    retry_maker: async_sessionmaker[AsyncSession],
) -> None:
    agent_id = await _seed(retry_maker, score_count=3)
    _install(app, retry_maker)
    async with retry_maker() as session, session.begin():
        agent = await session.get(Agent, agent_id)
        assert agent is not None
        agent.status = AgentStatus.SCORED

    detail = await client.get(
        f"/api/v1/admin/validation-retries/{agent_id}/validators/validator-1",
        headers=_HEADERS,
    )
    assert detail.status_code == 200, detail.text
    assert detail.json()["replacement_allowed"] is True
    request_id = uuid4()
    payload = {
        "request_id": str(request_id),
        "expected_snapshot": detail.json()["snapshot"],
        "expected_run_id": "run-1",
        "reason": "Validator relay failure made this accepted score untrustworthy",
    }
    response = await client.post(
        f"/api/v1/admin/validation-retries/{agent_id}/validators/validator-1/replace-score",
        headers=_HEADERS,
        json=payload,
    )
    assert response.status_code == 200, response.text
    assert response.json()["preserved_score_count"] == 3
    assert response.json()["original_run_id"] == "run-1"

    replay = await client.post(
        f"/api/v1/admin/validation-retries/{agent_id}/validators/validator-1/replace-score",
        headers=_HEADERS,
        json=payload,
    )
    assert replay.status_code == 200
    assert replay.json()["idempotent"] is True

    async with retry_maker() as session:
        agent = await session.get(Agent, agent_id)
        scores = list(
            (
                await session.scalars(select(Score).where(Score.agent_id == agent_id))
            ).all()
        )
        ticket = await session.get(
            ValidatorTicket, (agent_id, DEFAULT_BENCH_VERSION, "validator-1")
        )
        audit = await session.scalar(
            select(ScoreAuditEntry).where(
                ScoreAuditEntry.agent_id == agent_id,
                ScoreAuditEntry.event == EVENT_SCORE_RETEST_REQUESTED,
            )
        )
    assert agent is not None and agent.status == AgentStatus.SCORED
    assert {score.validator_hotkey for score in scores} == {
        "validator-0",
        "validator-1",
        "validator-2",
    }
    assert ticket is not None and ticket.status == TicketStatus.ISSUED
    assert ticket.attempt_count == 2
    assert audit is not None
    assert audit.payload["actor"] == "operator"
    assert audit.payload["reason"] == payload["reason"]
    assert audit.payload["run_id"] == "run-1"
    assert audit.payload["preserved_score"]["composite"] == 0.7
    assert audit.payload["preserved_score"]["bench_version"] == (DEFAULT_BENCH_VERSION)

    pending = await client.get(
        f"/api/v1/admin/validation-retries/{agent_id}/validators/validator-1",
        headers=_HEADERS,
    )
    assert pending.json()["replacement_pending"] is True
    assert pending.json()["replacement_allowed"] is False

    release = await client.post(
        f"/api/v1/admin/validation-retries/{agent_id}/validators/validator-1/release-ticket",
        headers=_HEADERS,
        json={
            "request_id": str(uuid4()),
            "expected_snapshot": pending.json()["snapshot"],
            "expected_deadline": pending.json()["ticket_deadline"],
            "reason": "Operator released the re-test after validator evidence cleared",
        },
    )
    assert release.status_code == 200, release.text
    assert release.json()["status"] == "scored"
    async with retry_maker() as session:
        ticket = await session.get(
            ValidatorTicket, (agent_id, DEFAULT_BENCH_VERSION, "validator-1")
        )
        released = await session.scalar(
            select(ScoreAuditEntry).where(
                ScoreAuditEntry.agent_id == agent_id,
                ScoreAuditEntry.event == EVENT_SCORE_RETEST_RELEASED,
            )
        )
    assert ticket is not None and ticket.status == TicketStatus.SCORED
    assert released is not None


async def test_replace_score_fails_closed_on_run_change_or_busy_validator(
    app: FastAPI,
    client: httpx.AsyncClient,
    retry_maker: async_sessionmaker[AsyncSession],
) -> None:
    agent_id = await _seed(retry_maker, score_count=1)
    other_agent_id = await _seed(retry_maker)
    _install(app, retry_maker)
    detail = await client.get(
        f"/api/v1/admin/validation-retries/{agent_id}/validators/validator-0",
        headers=_HEADERS,
    )
    wrong_run = await client.post(
        f"/api/v1/admin/validation-retries/{agent_id}/validators/validator-0/replace-score",
        headers=_HEADERS,
        json={
            "request_id": str(uuid4()),
            "expected_snapshot": detail.json()["snapshot"],
            "expected_run_id": "different-run",
            "reason": "Verified validator infrastructure failure changed the result",
        },
    )
    assert wrong_run.status_code == 409

    async with retry_maker() as session, session.begin():
        other = await session.get(
            ValidatorTicket,
            (other_agent_id, DEFAULT_BENCH_VERSION, "validator-0"),
        )
        assert other is not None
        other.status = TicketStatus.ISSUED
        other.deadline = datetime.now(UTC) + timedelta(minutes=30)
    blocked = await client.get(
        f"/api/v1/admin/validation-retries/{agent_id}/validators/validator-0",
        headers=_HEADERS,
    )
    assert blocked.status_code == 200
    assert blocked.json()["replacement_allowed"] is False
    assert blocked.json()["blocking_reason"] == (
        "validator is currently assigned to another submission"
    )


async def test_lists_only_unambiguous_finalized_score_outliers(
    app: FastAPI,
    client: httpx.AsyncClient,
    retry_maker: async_sessionmaker[AsyncSession],
) -> None:
    low_id = await _seed(retry_maker, score_count=3, composites=[0.12, 0.81, 0.83])
    high_id = await _seed(retry_maker, score_count=3, composites=[0.68, 0.70, 0.96])
    broad_id = await _seed(retry_maker, score_count=3, composites=[0.20, 0.50, 0.80])
    async with retry_maker() as session, session.begin():
        for agent_id in (low_id, high_id, broad_id):
            agent = await session.get(Agent, agent_id)
            assert agent is not None
            agent.status = AgentStatus.SCORED
    _install(app, retry_maker)

    response = await client.get("/api/v1/admin/score-outliers", headers=_HEADERS)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 2
    by_agent = {item["agent_id"]: item for item in body["items"]}
    assert by_agent[str(low_id)]["direction"] == "low"
    assert by_agent[str(low_id)]["outlier"]["validator_hotkey"] == "validator-0"
    assert by_agent[str(high_id)]["direction"] == "high"
    assert by_agent[str(high_id)]["outlier"]["validator_hotkey"] == "validator-2"
    assert str(broad_id) not in by_agent


# --- fleet-wide stuck list ------------------------------------------------


async def _seed_states(
    maker: async_sessionmaker[AsyncSession],
    *,
    name: str,
    tickets: list[tuple[str, TicketStatus, int, datetime | None]],
    created_offset_hours: float = 24.0,
    agent_status: AgentStatus = AgentStatus.EVALUATING,
    bench_version: int = DEFAULT_BENCH_VERSION,
) -> UUID:
    """Seed one agent with explicit (hotkey, status, attempt_count, retry_after)
    tickets; a SCORED ticket gets a matching score row."""
    agent_id = uuid4()
    async with maker() as session, session.begin():
        session.add(
            Agent(
                agent_id=agent_id,
                miner_hotkey="5Miner",
                name=name,
                version=1,
                sha256=agent_id.hex * 2,
                status=agent_status,
                screening_policy_version=SCREENING_POLICY_VERSION,
                created_at=_T0 - timedelta(hours=created_offset_hours),
            )
        )
        for index, (hotkey, status, attempt, retry_after) in enumerate(tickets):
            deadline = (
                _T0 + timedelta(hours=1)
                if status == TicketStatus.ISSUED
                else _T0 - timedelta(hours=2, minutes=index)
            )
            session.add(
                ValidatorTicket(
                    agent_id=agent_id,
                    validator_hotkey=hotkey,
                    status=status,
                    issued_at=_T0 - timedelta(hours=3, minutes=index),
                    deadline=deadline,
                    bench_version=bench_version,
                    attempt_count=attempt,
                    manual_retry_grants=0,
                    retry_after=retry_after,
                )
            )
            if status == TicketStatus.SCORED:
                session.add(
                    Score(
                        agent_id=agent_id,
                        bench_version=bench_version,
                        validator_hotkey=hotkey,
                        run_id=f"run-{hotkey}",
                        seed=7,
                        composite=0.7,
                        tool_mean=0.7,
                        memory_mean=0.7,
                        median_ms=100,
                        n=114,
                        generated_at=_T0 - timedelta(hours=1),
                    )
                )
    return agent_id


async def test_list_classifies_every_retry_state(
    app: FastAPI,
    client: httpx.AsyncClient,
    retry_maker: async_sessionmaker[AsyncSession],
) -> None:
    await _seed_states(
        retry_maker,
        name="exhausted-agent",
        tickets=[
            ("val-0", TicketStatus.EXPIRED, 2, _PAST),
            ("val-1", TicketStatus.EXPIRED, 2, _PAST),
            ("val-2", TicketStatus.EXPIRED, 2, _PAST),
        ],
    )
    await _seed_states(
        retry_maker,
        name="cooling-agent",
        tickets=[("val-0", TicketStatus.EXPIRED, 1, _FUTURE)],
    )
    await _seed_states(
        retry_maker,
        name="available-agent",
        tickets=[("val-0", TicketStatus.EXPIRED, 1, _PAST)],
    )
    await _seed_states(
        retry_maker,
        name="running-agent",
        tickets=[("val-0", TicketStatus.ISSUED, 1, None)],
    )
    await _seed_states(retry_maker, name="queued-agent", tickets=[])
    _install(app, retry_maker)

    response = await client.get("/api/v1/admin/validation-retries", headers=_HEADERS)
    assert response.status_code == 200, response.text
    body = response.json()
    by_name = {item["agent_name"]: item["retry_state"] for item in body["submissions"]}
    assert by_name == {
        "exhausted-agent": "exhausted",
        "cooling-agent": "cooling_down",
        "available-agent": "retry_available",
        "running-agent": "running",
        "queued-agent": "queued",
    }
    assert body["counts"] == {
        "exhausted": 1,
        "cooling_down": 1,
        "retry_available": 1,
        "running": 1,
        "queued": 1,
    }
    assert body["quorum"] == 3


async def test_list_excludes_agents_at_quorum(
    app: FastAPI,
    client: httpx.AsyncClient,
    retry_maker: async_sessionmaker[AsyncSession],
) -> None:
    await _seed_states(
        retry_maker,
        name="finished-agent",
        tickets=[
            ("val-0", TicketStatus.SCORED, 1, None),
            ("val-1", TicketStatus.SCORED, 1, None),
            ("val-2", TicketStatus.SCORED, 1, None),
        ],
    )
    _install(app, retry_maker)

    response = await client.get("/api/v1/admin/validation-retries", headers=_HEADERS)
    assert response.status_code == 200, response.text
    assert response.json()["submissions"] == []
    assert response.json()["counts"] == {}


async def test_list_state_filter_keeps_fleetwide_counts(
    app: FastAPI,
    client: httpx.AsyncClient,
    retry_maker: async_sessionmaker[AsyncSession],
) -> None:
    await _seed_states(
        retry_maker,
        name="exhausted-agent",
        tickets=[
            ("val-0", TicketStatus.EXPIRED, 2, _PAST),
            ("val-1", TicketStatus.EXPIRED, 2, _PAST),
            ("val-2", TicketStatus.EXPIRED, 2, _PAST),
        ],
    )
    await _seed_states(retry_maker, name="queued-agent", tickets=[])
    _install(app, retry_maker)

    response = await client.get(
        "/api/v1/admin/validation-retries",
        params={"state": "exhausted"},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [item["agent_name"] for item in body["submissions"]] == ["exhausted-agent"]
    # counts stay fleet-wide even though the list is filtered.
    assert body["counts"] == {"exhausted": 1, "queued": 1}


async def test_list_rejects_unknown_state(
    app: FastAPI,
    client: httpx.AsyncClient,
    retry_maker: async_sessionmaker[AsyncSession],
) -> None:
    _install(app, retry_maker)
    response = await client.get(
        "/api/v1/admin/validation-retries",
        params={"state": "wedged"},
        headers=_HEADERS,
    )
    assert response.status_code == 422
    assert "unknown retry state: wedged" in response.text


async def test_list_snapshot_matches_single_agent_detail(
    app: FastAPI,
    client: httpx.AsyncClient,
    retry_maker: async_sessionmaker[AsyncSession],
) -> None:
    # The snapshot an operator reads from the list must drive the per-agent
    # retry endpoint unchanged, so the two routes must agree byte for byte.
    agent_id = await _seed_states(
        retry_maker,
        name="exhausted-agent",
        tickets=[
            ("val-0", TicketStatus.EXPIRED, 2, _PAST),
            ("val-1", TicketStatus.EXPIRED, 2, _PAST),
            ("val-2", TicketStatus.EXPIRED, 2, _PAST),
        ],
    )
    _install(app, retry_maker)

    listing = await client.get("/api/v1/admin/validation-retries", headers=_HEADERS)
    detail = await client.get(
        f"/api/v1/admin/validation-retries/{agent_id}", headers=_HEADERS
    )
    assert listing.status_code == 200 and detail.status_code == 200
    item = listing.json()["submissions"][0]
    assert item["snapshot"] == detail.json()["snapshot"]
    assert item["recovery_allowed"] is True

    accepted = await client.post(
        f"/api/v1/admin/validation-retries/{agent_id}/retry",
        headers=_HEADERS,
        json={
            "request_id": str(uuid4()),
            "expected_snapshot": item["snapshot"],
            "reason": "chutes infrastructure outage burned the attempt budget",
        },
    )
    assert accepted.status_code == 200, accepted.text


async def test_list_sorts_exhausted_before_queued(
    app: FastAPI,
    client: httpx.AsyncClient,
    retry_maker: async_sessionmaker[AsyncSession],
) -> None:
    # An older queued agent must still sort behind a newer exhausted one:
    # severity, not age, drives the operator's attention.
    await _seed_states(
        retry_maker, name="old-queued", tickets=[], created_offset_hours=100.0
    )
    await _seed_states(
        retry_maker,
        name="new-exhausted",
        tickets=[
            ("val-0", TicketStatus.EXPIRED, 2, _PAST),
            ("val-1", TicketStatus.EXPIRED, 2, _PAST),
            ("val-2", TicketStatus.EXPIRED, 2, _PAST),
        ],
        created_offset_hours=1.0,
    )
    _install(app, retry_maker)

    response = await client.get("/api/v1/admin/validation-retries", headers=_HEADERS)
    assert response.status_code == 200, response.text
    names = [item["agent_name"] for item in response.json()["submissions"]]
    assert names == ["new-exhausted", "old-queued"]


async def test_partial_exhaustion_stays_queued_not_exhausted(
    app: FastAPI,
    client: httpx.AsyncClient,
    retry_maker: async_sessionmaker[AsyncSession],
) -> None:
    # One validator burned its budget but two slots were never leased: fresh
    # validators can still reach quorum, so this is queued (self-heals), not
    # exhausted (needs an operator).
    await _seed_states(
        retry_maker,
        name="one-exhausted",
        tickets=[("val-0", TicketStatus.EXPIRED, 2, _PAST)],
    )
    # One accepted score + two exhausted validators: only one slot remains
    # fillable, so a grant IS required → exhausted.
    await _seed_states(
        retry_maker,
        name="two-exhausted-one-scored",
        tickets=[
            ("val-0", TicketStatus.SCORED, 1, None),
            ("val-1", TicketStatus.EXPIRED, 2, _PAST),
            ("val-2", TicketStatus.EXPIRED, 2, _PAST),
        ],
    )
    _install(app, retry_maker)

    response = await client.get("/api/v1/admin/validation-retries", headers=_HEADERS)
    assert response.status_code == 200, response.text
    by_name = {
        item["agent_name"]: item["retry_state"]
        for item in response.json()["submissions"]
    }
    assert by_name["one-exhausted"] == "queued"
    assert by_name["two-exhausted-one-scored"] == "exhausted"


async def test_infra_grant_lifts_a_would_be_exhausted_ticket(
    app: FastAPI,
    client: httpx.AsyncClient,
    retry_maker: async_sessionmaker[AsyncSession],
) -> None:
    # attempt_count sits at the base cap, but an infrastructure grant raised the
    # cap — so the ticket still has budget and reads as retry_available, not
    # exhausted. This is the whole point of infra_retry_grants: an outage does
    # not spend the agent's genuine attempt budget.
    agent_id = uuid4()
    async with retry_maker() as session, session.begin():
        session.add(
            Agent(
                agent_id=agent_id,
                miner_hotkey="5Miner",
                name="infra-compensated",
                version=1,
                sha256=agent_id.hex * 2,
                status=AgentStatus.EVALUATING,
                screening_policy_version=SCREENING_POLICY_VERSION,
                created_at=_T0 - timedelta(days=1),
            )
        )
        session.add(
            ValidatorTicket(
                agent_id=agent_id,
                validator_hotkey="val-0",
                status=TicketStatus.EXPIRED,
                issued_at=_T0 - timedelta(hours=3),
                deadline=_T0 - timedelta(hours=2),
                bench_version=DEFAULT_BENCH_VERSION,
                attempt_count=2,
                manual_retry_grants=0,
                infra_retry_grants=1,
                retry_after=_PAST,
            )
        )
    _install(app, retry_maker)

    response = await client.get("/api/v1/admin/validation-retries", headers=_HEADERS)
    assert response.status_code == 200, response.text
    item = next(
        entry
        for entry in response.json()["submissions"]
        if entry["agent_id"] == str(agent_id)
    )
    assert item["retry_state"] == "retry_available"
    assert item["tickets"][0]["infra_retry_grants"] == 1
    assert item["tickets"][0]["retry_budget_exhausted"] is False


# --- batch retry-grant ----------------------------------------------------


async def _snapshot_of(client: httpx.AsyncClient, agent_id: UUID) -> str:
    detail = await client.get(
        f"/api/v1/admin/validation-retries/{agent_id}", headers=_HEADERS
    )
    return detail.json()["snapshot"]


async def test_batch_retry_grants_recoverable_and_skips_the_rest(
    app: FastAPI,
    client: httpx.AsyncClient,
    retry_maker: async_sessionmaker[AsyncSession],
) -> None:
    recoverable_a = await _seed(retry_maker)
    recoverable_b = await _seed(retry_maker)
    queued = await _seed_states(retry_maker, name="queued", tickets=[])
    stale = await _seed(retry_maker)
    _install(app, retry_maker)

    payload = {
        "reason": "chutes infrastructure outage recovery batch",
        "items": [
            {
                "agent_id": str(recoverable_a),
                "request_id": str(uuid4()),
                "expected_snapshot": await _snapshot_of(client, recoverable_a),
            },
            {
                "agent_id": str(recoverable_b),
                "request_id": str(uuid4()),
                "expected_snapshot": await _snapshot_of(client, recoverable_b),
            },
            {
                "agent_id": str(queued),
                "request_id": str(uuid4()),
                "expected_snapshot": await _snapshot_of(client, queued),
            },
            {
                "agent_id": str(stale),
                "request_id": str(uuid4()),
                "expected_snapshot": "0" * 64,
            },
        ],
    }
    resp = await client.post(
        "/api/v1/admin/validation-retries/batch-retry", headers=_HEADERS, json=payload
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["granted"] == 2
    by_id = {result["agent_id"]: result for result in body["results"]}
    assert by_id[str(recoverable_a)]["status"] == "granted"
    assert by_id[str(recoverable_b)]["status"] == "granted"
    assert by_id[str(queued)]["status"] == "skipped"
    assert (
        by_id[str(queued)]["detail"] == "not enough expired tickets to restore quorum"
    )
    assert by_id[str(stale)]["status"] == "skipped"
    assert by_id[str(stale)]["detail"] == "validation state changed"

    # A granted item actually raised the cap on its tickets.
    async with retry_maker() as session:
        tickets = list(
            (
                await session.scalars(
                    select(ValidatorTicket).where(
                        ValidatorTicket.agent_id == recoverable_a
                    )
                )
            ).all()
        )
        # The gate grants exactly the quorum (3) minimum slots needed.
        assert sum(t.manual_retry_grants for t in tickets) == 3


async def test_batch_retry_is_idempotent(
    app: FastAPI,
    client: httpx.AsyncClient,
    retry_maker: async_sessionmaker[AsyncSession],
) -> None:
    agent_id = await _seed(retry_maker)
    _install(app, retry_maker)
    payload = {
        "reason": "outage recovery, replayed",
        "items": [
            {
                "agent_id": str(agent_id),
                "request_id": str(uuid4()),
                "expected_snapshot": await _snapshot_of(client, agent_id),
            }
        ],
    }
    first = await client.post(
        "/api/v1/admin/validation-retries/batch-retry", headers=_HEADERS, json=payload
    )
    assert first.json()["results"][0]["status"] == "granted"

    second = await client.post(
        "/api/v1/admin/validation-retries/batch-retry", headers=_HEADERS, json=payload
    )
    assert second.status_code == 200, second.text
    assert second.json()["granted"] == 0
    assert second.json()["results"][0]["status"] == "idempotent"


async def test_batch_retry_rejects_duplicate_agent_ids(
    app: FastAPI,
    client: httpx.AsyncClient,
    retry_maker: async_sessionmaker[AsyncSession],
) -> None:
    agent_id = await _seed(retry_maker)
    _install(app, retry_maker)
    snapshot = await _snapshot_of(client, agent_id)
    payload = {
        "reason": "duplicate agents in one batch",
        "items": [
            {
                "agent_id": str(agent_id),
                "request_id": str(uuid4()),
                "expected_snapshot": snapshot,
            },
            {
                "agent_id": str(agent_id),
                "request_id": str(uuid4()),
                "expected_snapshot": snapshot,
            },
        ],
    }
    resp = await client.post(
        "/api/v1/admin/validation-retries/batch-retry", headers=_HEADERS, json=payload
    )
    assert resp.status_code == 422
