"""Unit tests for :mod:`ditto.db.queries.tickets` against SQLite-in-memory."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.ticket_status import TicketStatus
from ditto.db.models import Agent
from ditto.db.queries.scores import SCORING_QUORUM
from ditto.db.queries.tickets import (
    expire_overdue_tickets,
    get_open_ticket,
    issue_ticket,
    mark_ticket_scored,
)

_NOW = datetime(2026, 7, 9, 12, 0, 0, tzinfo=UTC)
_TTL = timedelta(minutes=30)
_LATER = _NOW + timedelta(hours=1)


async def _seed_evaluating(
    session: AsyncSession, *, created_at: datetime = _NOW, name: str = "a"
) -> object:
    aid = uuid4()
    async with session.begin():
        session.add(
            Agent(
                agent_id=aid,
                miner_hotkey="5Miner",
                name=name,
                sha256="ab" * 32,
                status=AgentStatus.EVALUATING,
                created_at=created_at,
            )
        )
    return aid


class TestIssueTicket:
    async def test_seats_ticket_for_evaluating_agent(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            t = await issue_ticket(
                session, validator_hotkey="5V1", now=_NOW, ttl=_TTL
            )
        assert t is not None
        assert t.agent_id == aid
        assert t.status == TicketStatus.ISSUED
        assert t.deadline == _NOW + _TTL

    async def test_no_evaluating_agent_returns_none(
        self, session: AsyncSession
    ) -> None:
        async with session.begin():
            t = await issue_ticket(
                session, validator_hotkey="5V1", now=_NOW, ttl=_TTL
            )
        assert t is None

    async def test_caps_at_quorum(self, session: AsyncSession) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            for i in range(SCORING_QUORUM):
                t = await issue_ticket(
                    session, validator_hotkey=f"5V{i}", now=_NOW, ttl=_TTL
                )
                assert t is not None and t.agent_id == aid
            # Quorum reached: a further distinct validator gets no job.
            extra = await issue_ticket(
                session, validator_hotkey="5Vx", now=_NOW, ttl=_TTL
            )
        assert extra is None

    async def test_same_validator_not_seated_twice_on_one_agent(
        self, session: AsyncSession
    ) -> None:
        await _seed_evaluating(session)
        async with session.begin():
            t1 = await issue_ticket(
                session, validator_hotkey="5V1", now=_NOW, ttl=_TTL
            )
            t2 = await issue_ticket(
                session, validator_hotkey="5V1", now=_NOW, ttl=_TTL
            )
        assert t1 is not None
        assert t2 is None  # only one agent, and this validator already holds it

    async def test_validator_can_hold_tickets_for_distinct_agents(
        self, session: AsyncSession
    ) -> None:
        a1 = await _seed_evaluating(session, created_at=_NOW, name="old")
        a2 = await _seed_evaluating(
            session, created_at=_NOW + timedelta(minutes=1), name="new"
        )
        async with session.begin():
            t1 = await issue_ticket(
                session, validator_hotkey="5V1", now=_NOW, ttl=_TTL
            )
            t2 = await issue_ticket(
                session, validator_hotkey="5V1", now=_NOW, ttl=_TTL
            )
        assert t1 is not None and t2 is not None
        assert t1.agent_id == a1  # oldest first
        assert {t1.agent_id, t2.agent_id} == {a1, a2}


class TestExpiry:
    async def test_expired_ticket_frees_slot(self, session: AsyncSession) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            for i in range(SCORING_QUORUM):
                await issue_ticket(
                    session, validator_hotkey=f"5V{i}", now=_NOW, ttl=_TTL
                )
        # After the deadline the three lapse, so a new validator can seat.
        async with session.begin():
            t = await issue_ticket(
                session, validator_hotkey="5Vnew", now=_LATER, ttl=_TTL
            )
        assert t is not None and t.agent_id == aid

    async def test_expire_overdue_returns_count(
        self, session: AsyncSession
    ) -> None:
        await _seed_evaluating(session)
        async with session.begin():
            await issue_ticket(
                session, validator_hotkey="5V1", now=_NOW, ttl=_TTL
            )
        async with session.begin():
            n = await expire_overdue_tickets(session, now=_LATER)
        assert n == 1


class TestTicketLifecycle:
    async def test_get_open_ticket_live(self, session: AsyncSession) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            await issue_ticket(
                session, validator_hotkey="5V1", now=_NOW, ttl=_TTL
            )
        async with session.begin():
            t = await get_open_ticket(
                session, agent_id=aid, validator_hotkey="5V1", now=_NOW
            )
        assert t is not None

    async def test_get_open_ticket_expired_is_none(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            await issue_ticket(
                session, validator_hotkey="5V1", now=_NOW, ttl=_TTL
            )
        async with session.begin():
            t = await get_open_ticket(
                session, agent_id=aid, validator_hotkey="5V1", now=_LATER
            )
        assert t is None

    async def test_get_open_ticket_absent_is_none(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            t = await get_open_ticket(
                session, agent_id=aid, validator_hotkey="5Vx", now=_NOW
            )
        assert t is None

    async def test_mark_scored_makes_ticket_not_open(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            await issue_ticket(
                session, validator_hotkey="5V1", now=_NOW, ttl=_TTL
            )
        async with session.begin():
            await mark_ticket_scored(session, agent_id=aid, validator_hotkey="5V1")
        async with session.begin():
            t = await get_open_ticket(
                session, agent_id=aid, validator_hotkey="5V1", now=_NOW
            )
        assert t is None  # spent, no longer open
