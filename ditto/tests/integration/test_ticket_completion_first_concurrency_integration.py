"""Real-Postgres proof that parallel slots preserve completion-first FIFO."""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.api_models.ticket_status import TicketStatus
from ditto.db import create_db_engine
from ditto.db.models import Agent, EvaluationPayment, Score, ValidatorTicket
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


@pytest.mark.parametrize("completion_first", [False, True])
@pytest.mark.parametrize(
    "identity_mode", ["paid-coldkey", "mixed-legacy", "rotated-legacy-bridge"]
)
async def test_same_owner_partial_scores_converge_on_one_generation(
    completion_first: bool, identity_mode: str
) -> None:
    engine = create_db_engine()
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    agents = [uuid4(), uuid4()]
    async with maker() as session, session.begin():
        await session.execute(text("TRUNCATE TABLE agents CASCADE"))
        for index, agent_id in enumerate(agents):
            miner_hotkey = (
                "mixed-owner-hotkey"
                if identity_mode == "mixed-legacy"
                else f"rotated-owner-hotkey-{index}"
                if identity_mode == "rotated-legacy-bridge"
                else f"same-owner-partial-{index}"
            )
            session.add_all(
                [
                    Agent(
                        agent_id=agent_id,
                        miner_hotkey=miner_hotkey,
                        name=f"same-owner-partial-{index}",
                        sha256=f"{index + 1:x}" * 64,
                        status=AgentStatus.EVALUATING,
                        screening_policy_version=SCREENING_POLICY_VERSION,
                        created_at=now + timedelta(minutes=index),
                    ),
                    ValidatorTicket(
                        agent_id=agent_id,
                        validator_hotkey=f"prior-validator-{index}",
                        status=TicketStatus.SCORED,
                        issued_at=now,
                        deadline=now + timedelta(minutes=30),
                        bench_version=2,
                        attempt_count=1,
                    ),
                    Score(
                        agent_id=agent_id,
                        validator_hotkey=f"prior-validator-{index}",
                        run_id=f"prior-run-{index}",
                        signature=None,
                        seed=index,
                        composite=0.5 + index / 10,
                        tool_mean=0.5,
                        memory_mean=0.5,
                        median_ms=100,
                        n=206,
                        details=None,
                        bench_version=2,
                        generated_at=now,
                    ),
                ]
            )
            should_add_payment = (
                identity_mode == "paid-coldkey"
                or (identity_mode == "mixed-legacy" and index == 0)
                or (identity_mode == "rotated-legacy-bridge" and index == 1)
            )
            if should_add_payment:
                session.add(
                    EvaluationPayment(
                        block_hash=f"0xsame-owner-partial-{index}",
                        extrinsic_index=index,
                        agent_id=agent_id,
                        miner_hotkey=miner_hotkey,
                        miner_coldkey="same-owner-coldkey",
                        amount_rao=1,
                        tao_usd_rate=Decimal("1"),
                        dest_address="payment-destination",
                        timestamp=now,
                    )
                )
        if identity_mode == "rotated-legacy-bridge":
            bridge_id = uuid4()
            session.add_all(
                [
                    Agent(
                        agent_id=bridge_id,
                        miner_hotkey="rotated-owner-hotkey-0",
                        name="settled-owner-identity-bridge",
                        sha256="f" * 64,
                        status=AgentStatus.SCORED,
                        screening_policy_version=SCREENING_POLICY_VERSION,
                        created_at=now - timedelta(days=1),
                    ),
                    EvaluationPayment(
                        block_hash="0xsettled-owner-identity-bridge",
                        extrinsic_index=99,
                        agent_id=bridge_id,
                        miner_hotkey="rotated-owner-hotkey-0",
                        miner_coldkey="same-owner-coldkey",
                        amount_rao=1,
                        tao_usd_rate=Decimal("1"),
                        dest_address="payment-destination",
                        timestamp=now - timedelta(days=1),
                    ),
                ]
            )

    async def claim(validator_hotkey: str):
        async with maker() as session, session.begin():
            ticket = await issue_ticket(
                session,
                validator_hotkey=validator_hotkey,
                now=now,
                ttl=timedelta(minutes=30),
                completion_first=completion_first,
            )
            return ticket.agent_id if ticket is not None else None

    outcomes = await asyncio.gather(claim("recovery-a"), claim("recovery-b"))
    successful = [outcome for outcome in outcomes if outcome is not None]
    assert successful
    assert len(set(successful)) == 1

    # SKIP LOCKED may make one simultaneous poll yield while its sibling
    # transaction owns the selected Agent row. The next ordinary poll must
    # fill that free slot with the same generation rather than remain stuck or
    # open the other one.
    settled = [await claim("recovery-a"), await claim("recovery-b")]
    assert None not in settled
    assert len(set(settled)) == 1

    async with maker() as session:
        issued_agents = (
            await session.scalars(
                select(ValidatorTicket.agent_id).where(
                    ValidatorTicket.status == TicketStatus.ISSUED
                )
            )
        ).all()
    assert len(issued_agents) == 2
    assert len(set(issued_agents)) == 1
    await engine.dispose()
