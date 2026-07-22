"""Real-Postgres proof that parallel slots preserve completion-first FIFO."""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.db import create_db_engine
from ditto.db.models import Agent, ValidatorTicket
from ditto.db.queries.tickets import issue_ticket

pytestmark = pytest.mark.integration


async def test_same_validator_slots_do_not_advance_past_fifo_head() -> None:
    engine = create_db_engine()
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    oldest = uuid4()
    newer = uuid4()
    async with maker() as session, session.begin():
        await session.execute(text("TRUNCATE TABLE agents CASCADE"))
        session.add_all(
            [
                Agent(
                    agent_id=oldest,
                    miner_hotkey="completion-first-oldest",
                    name="completion-first-oldest",
                    sha256="a" * 64,
                    status=AgentStatus.EVALUATING,
                    screening_policy_version=SCREENING_POLICY_VERSION,
                    created_at=now,
                ),
                Agent(
                    agent_id=newer,
                    miner_hotkey="completion-first-newer",
                    name="completion-first-newer",
                    sha256="b" * 64,
                    status=AgentStatus.EVALUATING,
                    screening_policy_version=SCREENING_POLICY_VERSION,
                    created_at=now + timedelta(minutes=1),
                ),
            ]
        )

    async def claim(slot_id: str):
        async with maker() as session, session.begin():
            ticket = await issue_ticket(
                session,
                validator_hotkey="5ConcurrentCompletionFirst",
                slot_id=slot_id,
                now=now,
                ttl=timedelta(minutes=30),
                completion_first=True,
            )
            return ticket.agent_id if ticket is not None else None

    outcomes = await asyncio.gather(claim("slot-0"), claim("slot-1"))
    assert outcomes.count(oldest) == 1
    assert outcomes.count(None) == 1
    assert newer not in outcomes

    async with maker() as session:
        newer_tickets = await session.scalar(
            select(func.count()).where(ValidatorTicket.agent_id == newer)
        )
    assert newer_tickets == 0
    await engine.dispose()
