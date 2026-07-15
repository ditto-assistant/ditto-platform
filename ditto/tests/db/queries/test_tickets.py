"""Unit tests for :mod:`ditto.db.queries.tickets` against SQLite-in-memory."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.api_models.ticket_status import TicketStatus
from ditto.db.models import Agent, Score, ValidatorTicket
from ditto.db.queries.scores import SCORING_QUORUM
from ditto.db.queries.tickets import (
    FIRST_SCORE_CONTINUATION_FLOOR,
    expire_overdue_tickets,
    get_open_ticket,
    issue_ticket,
    mark_ticket_scored,
)

_NOW = datetime(2026, 7, 9, 12, 0, 0, tzinfo=UTC)
_TTL = timedelta(minutes=30)
_LATER = _NOW + timedelta(hours=1)
_AFTER_COOLDOWN = _NOW + timedelta(hours=7)


async def _seed_evaluating(
    session: AsyncSession, *, created_at: datetime = _NOW, name: str = "a"
) -> UUID:
    aid = uuid4()
    async with session.begin():
        session.add(
            Agent(
                agent_id=aid,
                miner_hotkey="5Miner",
                name=name,
                sha256="ab" * 32,
                status=AgentStatus.EVALUATING,
                screening_policy_version=SCREENING_POLICY_VERSION,
                created_at=created_at,
            )
        )
    return aid


class TestIssueTicket:
    async def test_skips_agent_that_needs_rescreening(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            agent = await session.get(Agent, aid)
            assert agent is not None
            agent.screening_policy_version = 0
        async with session.begin():
            ticket = await issue_ticket(
                session, validator_hotkey="5V1", now=_NOW, ttl=_TTL
            )
        assert ticket is None

    async def test_seats_ticket_for_evaluating_agent(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            t = await issue_ticket(session, validator_hotkey="5V1", now=_NOW, ttl=_TTL)
        assert t is not None
        assert t.agent_id == aid
        assert t.status == TicketStatus.ISSUED
        assert t.deadline == _NOW + _TTL

    async def test_score_below_floor_does_not_receive_another_ticket(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            session.add(
                ValidatorTicket(
                    agent_id=aid,
                    validator_hotkey="5Scored",
                    status=TicketStatus.SCORED,
                    issued_at=_NOW,
                    deadline=_NOW + _TTL,
                    bench_version=2,
                    attempt_count=1,
                )
            )
            session.add(
                Score(
                    agent_id=aid,
                    validator_hotkey="5Scored",
                    run_id="below-floor",
                    signature=None,
                    seed=123,
                    composite=FIRST_SCORE_CONTINUATION_FLOOR - 0.01,
                    tool_mean=0.24,
                    memory_mean=0.24,
                    median_ms=100,
                    n=114,
                    details=None,
                    generated_at=_NOW,
                )
            )

        async with session.begin():
            ticket = await issue_ticket(
                session, validator_hotkey="5Next", now=_NOW, ttl=_TTL
            )

        assert ticket is None

    async def test_score_at_floor_receives_another_ticket(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            session.add(
                ValidatorTicket(
                    agent_id=aid,
                    validator_hotkey="5Scored",
                    status=TicketStatus.SCORED,
                    issued_at=_NOW,
                    deadline=_NOW + _TTL,
                    bench_version=2,
                    attempt_count=1,
                )
            )
            session.add(
                Score(
                    agent_id=aid,
                    validator_hotkey="5Scored",
                    run_id="at-floor",
                    signature=None,
                    seed=123,
                    composite=FIRST_SCORE_CONTINUATION_FLOOR,
                    tool_mean=FIRST_SCORE_CONTINUATION_FLOOR,
                    memory_mean=FIRST_SCORE_CONTINUATION_FLOOR,
                    median_ms=100,
                    n=114,
                    details=None,
                    generated_at=_NOW,
                )
            )

        async with session.begin():
            ticket = await issue_ticket(
                session, validator_hotkey="5Next", now=_NOW, ttl=_TTL
            )

        assert ticket is not None
        assert ticket.agent_id == aid

    async def test_floor_does_not_interrupt_submission_after_second_score(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            for validator, composite in (("5First", 0.30), ("5Second", 0.10)):
                session.add(
                    Score(
                        agent_id=aid,
                        validator_hotkey=validator,
                        run_id=f"run-{validator}",
                        signature=None,
                        seed=123,
                        composite=composite,
                        tool_mean=composite,
                        memory_mean=composite,
                        median_ms=100,
                        n=114,
                        details=None,
                        generated_at=_NOW,
                    )
                )

        async with session.begin():
            ticket = await issue_ticket(
                session, validator_hotkey="5Third", now=_NOW, ttl=_TTL
            )

        assert ticket is not None
        assert ticket.agent_id == aid

    async def test_no_evaluating_agent_returns_none(
        self, session: AsyncSession
    ) -> None:
        async with session.begin():
            t = await issue_ticket(session, validator_hotkey="5V1", now=_NOW, ttl=_TTL)
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

    async def test_same_validator_resumes_its_live_ticket(
        self, session: AsyncSession
    ) -> None:
        await _seed_evaluating(session)
        async with session.begin():
            t1 = await issue_ticket(session, validator_hotkey="5V1", now=_NOW, ttl=_TTL)
            t2 = await issue_ticket(session, validator_hotkey="5V1", now=_NOW, ttl=_TTL)
        assert t1 is not None
        assert t2 is not None
        assert t2.agent_id == t1.agent_id
        assert t2.deadline == t1.deadline

    async def test_validator_cannot_hold_live_tickets_for_distinct_agents(
        self, session: AsyncSession
    ) -> None:
        a1 = await _seed_evaluating(session, created_at=_NOW, name="old")
        a2 = await _seed_evaluating(
            session, created_at=_NOW + timedelta(minutes=1), name="new"
        )
        async with session.begin():
            t1 = await issue_ticket(session, validator_hotkey="5V1", now=_NOW, ttl=_TTL)
            t2 = await issue_ticket(session, validator_hotkey="5V1", now=_NOW, ttl=_TTL)
        assert t1 is not None and t2 is not None
        assert t1.agent_id == a1  # oldest first
        assert t2.agent_id == a1
        assert t2.agent_id != a2

    async def test_prioritizes_fewest_accepted_scores_before_age(
        self, session: AsyncSession
    ) -> None:
        two_scores = await _seed_evaluating(
            session, created_at=_NOW - timedelta(hours=2), name="two-scores"
        )
        one_score = await _seed_evaluating(
            session, created_at=_NOW - timedelta(hours=1), name="one-score"
        )
        zero_scores = await _seed_evaluating(
            session, created_at=_NOW, name="zero-scores"
        )
        async with session.begin():
            for agent_id, validators in (
                (two_scores, ("5A", "5B")),
                (one_score, ("5C",)),
            ):
                for validator in validators:
                    session.add(
                        ValidatorTicket(
                            agent_id=agent_id,
                            validator_hotkey=validator,
                            status=TicketStatus.SCORED,
                            issued_at=_NOW,
                            deadline=_NOW + _TTL,
                            bench_version=2,
                            attempt_count=1,
                        )
                    )

        claimed: list[UUID] = []
        async with session.begin():
            for _ in range(3):
                ticket = await issue_ticket(
                    session, validator_hotkey="5New", now=_NOW, ttl=_TTL
                )
                assert ticket is not None
                ticket.status = TicketStatus.SCORED
                claimed.append(ticket.agent_id)

        assert claimed == [zero_scores, one_score, two_scores]

    async def test_completion_lane_prioritizes_highest_provisional_score(
        self, session: AsyncSession
    ) -> None:
        low = await _seed_evaluating(
            session, created_at=_NOW - timedelta(hours=2), name="low"
        )
        high = await _seed_evaluating(
            session, created_at=_NOW - timedelta(hours=1), name="high"
        )
        medium = await _seed_evaluating(session, created_at=_NOW, name="medium")
        async with session.begin():
            for agent_id, composites in (
                (low, (0.20, 0.30)),
                (high, (0.90, 0.80)),
                (medium, (0.60, 0.70)),
            ):
                for index, composite in enumerate(composites):
                    validator = f"5Scored-{agent_id}-{index}"
                    session.add(
                        ValidatorTicket(
                            agent_id=agent_id,
                            validator_hotkey=validator,
                            status=TicketStatus.SCORED,
                            issued_at=_NOW,
                            deadline=_NOW + _TTL,
                            bench_version=2,
                            attempt_count=1,
                        )
                    )
                    session.add(
                        Score(
                            agent_id=agent_id,
                            validator_hotkey=validator,
                            run_id=f"run-{agent_id}-{index}",
                            signature=None,
                            seed=123,
                            composite=composite,
                            tool_mean=composite,
                            memory_mean=composite,
                            median_ms=100,
                            n=114,
                            details=None,
                            generated_at=_NOW,
                        )
                    )

        async with session.begin():
            ticket = await issue_ticket(
                session, validator_hotkey="5Completion", now=_NOW, ttl=_TTL
            )

        assert ticket is not None
        assert ticket.agent_id == high

    async def test_round_robins_live_assignments_across_zero_score_agents(
        self, session: AsyncSession
    ) -> None:
        agents = [
            await _seed_evaluating(
                session,
                created_at=_NOW + timedelta(minutes=index),
                name=f"agent-{index}",
            )
            for index in range(3)
        ]

        claimed: list[UUID] = []
        async with session.begin():
            for index in range(3):
                ticket = await issue_ticket(
                    session,
                    validator_hotkey=f"5V{index}",
                    now=_NOW,
                    ttl=_TTL,
                )
                assert ticket is not None
                claimed.append(ticket.agent_id)

        assert claimed == agents

    async def test_live_assignment_and_accepted_score_are_equal_coverage(
        self, session: AsyncSession
    ) -> None:
        one_score = await _seed_evaluating(session, name="one-score")
        zero_scores = await _seed_evaluating(
            session,
            created_at=_NOW + timedelta(minutes=1),
            name="zero-scores",
        )
        async with session.begin():
            session.add(
                ValidatorTicket(
                    agent_id=one_score,
                    validator_hotkey="5Scored",
                    status=TicketStatus.SCORED,
                    issued_at=_NOW,
                    deadline=_NOW + _TTL,
                    bench_version=2,
                    attempt_count=1,
                )
            )

        async with session.begin():
            first = await issue_ticket(
                session, validator_hotkey="5NewA", now=_NOW, ttl=_TTL
            )
        async with session.begin():
            second = await issue_ticket(
                session, validator_hotkey="5NewB", now=_NOW, ttl=_TTL
            )

        assert first is not None and first.agent_id == zero_scores
        assert second is not None and second.agent_id == one_score


class TestExpiry:
    async def test_deadline_instant_is_expired(self, session: AsyncSession) -> None:
        await _seed_evaluating(session)
        async with session.begin():
            await issue_ticket(session, validator_hotkey="5V1", now=_NOW, ttl=_TTL)
        deadline = _NOW + _TTL
        async with session.begin():
            assert await expire_overdue_tickets(session, now=deadline) == 1

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

    async def test_expired_ticket_cools_down_and_next_agent_moves_ahead(
        self, session: AsyncSession
    ) -> None:
        slow = await _seed_evaluating(session, name="slow")
        next_agent = await _seed_evaluating(
            session,
            created_at=_NOW + timedelta(minutes=1),
            name="next",
        )
        async with session.begin():
            first = await issue_ticket(
                session, validator_hotkey="5V1", now=_NOW, ttl=_TTL
            )
        assert first is not None and first.agent_id == slow

        async with session.begin():
            claimed = await issue_ticket(
                session, validator_hotkey="5V1", now=_LATER, ttl=_TTL
            )

        assert claimed is not None
        assert claimed.agent_id == next_agent

    async def test_expired_ticket_gets_one_retry_after_cooldown(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            await issue_ticket(session, validator_hotkey="5V1", now=_NOW, ttl=_TTL)
        async with session.begin():
            retried = await issue_ticket(
                session,
                validator_hotkey="5V1",
                now=_AFTER_COOLDOWN,
                ttl=_TTL,
            )

        assert retried is not None and retried.agent_id == aid
        assert retried.attempt_count == 2
        assert retried.issued_at == _AFTER_COOLDOWN

    async def test_never_attempted_agent_precedes_eligible_retry(
        self, session: AsyncSession
    ) -> None:
        slow = await _seed_evaluating(session, name="slow")
        untouched = await _seed_evaluating(
            session,
            created_at=_NOW + timedelta(minutes=1),
            name="untouched",
        )
        async with session.begin():
            first = await issue_ticket(
                session, validator_hotkey="5V1", now=_NOW, ttl=_TTL
            )
        assert first is not None and first.agent_id == slow

        async with session.begin():
            claimed = await issue_ticket(
                session,
                validator_hotkey="5V1",
                now=_AFTER_COOLDOWN,
                ttl=_TTL,
            )

        assert claimed is not None
        assert claimed.agent_id == untouched

    async def test_second_expiry_exhausts_same_version_retry_budget(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            await issue_ticket(session, validator_hotkey="5V1", now=_NOW, ttl=_TTL)
        async with session.begin():
            await issue_ticket(
                session,
                validator_hotkey="5V1",
                now=_AFTER_COOLDOWN,
                ttl=_TTL,
            )
        after_second_expiry = _AFTER_COOLDOWN + timedelta(hours=7)
        async with session.begin():
            third = await issue_ticket(
                session,
                validator_hotkey="5V1",
                now=after_second_expiry,
                ttl=_TTL,
            )

        assert third is None
        async with session.begin():
            ticket = await session.get(ValidatorTicket, (aid, "5V1"))
        assert ticket is not None
        assert ticket.status == TicketStatus.EXPIRED
        assert ticket.attempt_count == 2

    async def test_benchmark_version_change_resets_retry_budget(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            ticket = await issue_ticket(
                session,
                validator_hotkey="5V1",
                now=_NOW,
                ttl=_TTL,
                bench_version=2,
            )
        assert ticket is not None
        async with session.begin():
            ticket = await session.get(ValidatorTicket, (aid, "5V1"))
            assert ticket is not None
            ticket.status = TicketStatus.EXPIRED
            ticket.attempt_count = 2
            ticket.retry_after = _NOW + timedelta(days=1)
        async with session.begin():
            reset = await issue_ticket(
                session,
                validator_hotkey="5V1",
                now=_LATER,
                ttl=_TTL,
                bench_version=3,
            )

        assert reset is not None
        assert reset.bench_version == 3
        assert reset.attempt_count == 1
        assert reset.retry_after is None

    async def test_expire_overdue_returns_count(self, session: AsyncSession) -> None:
        await _seed_evaluating(session)
        async with session.begin():
            await issue_ticket(session, validator_hotkey="5V1", now=_NOW, ttl=_TTL)
        async with session.begin():
            n = await expire_overdue_tickets(session, now=_LATER)
        assert n == 1


class TestTicketLifecycle:
    async def test_get_open_ticket_live(self, session: AsyncSession) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            await issue_ticket(session, validator_hotkey="5V1", now=_NOW, ttl=_TTL)
        async with session.begin():
            t = await get_open_ticket(
                session,
                agent_id=aid,
                validator_hotkey="5V1",
                now=_NOW,
                deadline=_NOW + _TTL,
            )
        assert t is not None

    async def test_get_open_ticket_expired_is_none(self, session: AsyncSession) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            await issue_ticket(session, validator_hotkey="5V1", now=_NOW, ttl=_TTL)
        async with session.begin():
            t = await get_open_ticket(
                session,
                agent_id=aid,
                validator_hotkey="5V1",
                now=_LATER,
                deadline=_NOW + _TTL,
            )
        assert t is None

    async def test_get_open_ticket_at_exact_deadline_is_none(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_evaluating(session)
        deadline = _NOW + _TTL
        async with session.begin():
            await issue_ticket(session, validator_hotkey="5V1", now=_NOW, ttl=_TTL)
        async with session.begin():
            ticket = await get_open_ticket(
                session,
                agent_id=aid,
                validator_hotkey="5V1",
                now=deadline,
                deadline=deadline,
            )
        assert ticket is None

    async def test_get_open_ticket_absent_is_none(self, session: AsyncSession) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            t = await get_open_ticket(
                session,
                agent_id=aid,
                validator_hotkey="5Vx",
                now=_NOW,
                deadline=_NOW + _TTL,
            )
        assert t is None

    async def test_mark_scored_makes_ticket_not_open(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            await issue_ticket(session, validator_hotkey="5V1", now=_NOW, ttl=_TTL)
        async with session.begin():
            await mark_ticket_scored(session, agent_id=aid, validator_hotkey="5V1")
        async with session.begin():
            t = await get_open_ticket(
                session,
                agent_id=aid,
                validator_hotkey="5V1",
                now=_NOW,
                deadline=_NOW + _TTL,
            )
        assert t is None  # spent, no longer open
