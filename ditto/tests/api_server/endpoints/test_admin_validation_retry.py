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
from ditto.db.models import Agent, Base, Score, ValidatorRetryRecovery, ValidatorTicket
from ditto.db.queries.tickets import issue_ticket

_TOKEN = "test-admin-token-at-least-32-characters"
_HEADERS = {"Authorization": f"Bearer {_TOKEN}", "X-Admin-Actor": "operator"}
_T0 = datetime(2026, 7, 18, 12, tzinfo=UTC)


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
    bench_version: int = 2,
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
                        composite=0.7,
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
            bench_version=2,
        )
    assert ticket is not None and ticket.agent_id == agent_id
    assert ticket.attempt_count == 3
    assert ticket.manual_retry_grants == 1
