"""Unit tests for :mod:`ditto.db.queries.tickets` against SQLite-in-memory."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.api_models.ticket_status import TicketStatus
from ditto.db.models import (
    Agent,
    BenchmarkDataset,
    BenchmarkRollout,
    Score,
    ValidatorTicket,
)
from ditto.db.queries.scores import SCORING_QUORUM
from ditto.db.queries.tickets import (
    EMISSION_CONTENDER_COUNT,
    MAX_ATTEMPTS_PER_VERSION,
    PROVISIONAL_CONTENDER_LANE_SIZE,
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
    session: AsyncSession,
    *,
    created_at: datetime = _NOW,
    name: str = "a",
    screened: bool = False,
) -> UUID:
    aid = uuid4()
    async with session.begin():
        agent = Agent(
            agent_id=aid,
            miner_hotkey="5Miner",
            name=name,
            sha256="ab" * 32,
            status=AgentStatus.EVALUATING,
            screening_policy_version=SCREENING_POLICY_VERSION,
            created_at=created_at,
        )
        if screened:
            agent.screened_image_sha256 = "12" * 32
            agent.screened_image_size_bytes = 123
            agent.screened_image_id = "sha256:" + "34" * 32
            agent.screened_image_ref = f"ditto-screen/{aid}:latest"
            agent.screened_image_upload_id = uuid4()
            agent.screened_image_verified_at = _NOW
        session.add(agent)
    return aid


async def _seed_finalized_top_five(
    session: AsyncSession, *, fifth_place: float = 0.80
) -> None:
    """Establish five ranked miners with ``fifth_place`` as the live floor."""
    async with session.begin():
        for rank in range(EMISSION_CONTENDER_COUNT):
            agent_id = uuid4()
            composite = fifth_place + (EMISSION_CONTENDER_COUNT - rank - 1) * 0.01
            session.add(
                Agent(
                    agent_id=agent_id,
                    miner_hotkey=f"5Ranked-{rank}",
                    name=f"ranked-{rank}",
                    sha256=f"{rank + 100:064x}",
                    status=AgentStatus.SCORED,
                    screening_policy_version=SCREENING_POLICY_VERSION,
                    created_at=_NOW - timedelta(days=1, minutes=rank),
                )
            )
            for validator_index in range(SCORING_QUORUM):
                validator = f"5Ranked-{rank}-{validator_index}"
                session.add(
                    Score(
                        agent_id=agent_id,
                        validator_hotkey=validator,
                        run_id=f"ranked-{rank}-{validator_index}",
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


async def _seed_two_scores_below_floor(
    session: AsyncSession, *, bench_version: int = 2
) -> UUID:
    """An ``evaluating`` agent whose best-case median cannot reach the floor."""
    aid = await _seed_evaluating(session)
    async with session.begin():
        for index, composite in enumerate((0.10, 0.20)):
            validator = f"5Scored-{index}"
            session.add(
                ValidatorTicket(
                    agent_id=aid,
                    validator_hotkey=validator,
                    status=TicketStatus.SCORED,
                    issued_at=_NOW,
                    deadline=_NOW + _TTL,
                    bench_version=bench_version,
                    attempt_count=1,
                )
            )
            session.add(
                Score(
                    agent_id=aid,
                    validator_hotkey=validator,
                    run_id=f"below-top-five-{index}",
                    signature=None,
                    seed=123,
                    composite=composite,
                    tool_mean=composite,
                    memory_mean=composite,
                    median_ms=100,
                    n=114,
                    details=None,
                    bench_version=bench_version,
                    generated_at=_NOW,
                )
            )
    return aid


class TestIssueTicket:
    async def test_fresh_lane_excludes_pre_rollout_backlog(
        self, session: AsyncSession
    ) -> None:
        rollout_started = _NOW - timedelta(minutes=5)
        old = await _seed_evaluating(
            session,
            created_at=rollout_started - timedelta(days=1),
            name="old",
            screened=True,
        )
        fresh = await _seed_evaluating(
            session,
            created_at=rollout_started + timedelta(minutes=1),
            name="fresh",
            screened=True,
        )
        async with session.begin():
            for agent_id in (old, fresh):
                session.add(
                    BenchmarkDataset(
                        agent_id=agent_id,
                        bench_version=3,
                        seed=123,
                        sha256="cd" * 32,
                        run_size="full",
                    )
                )

        async with session.begin():
            ticket = await issue_ticket(
                session,
                validator_hotkey="5Fresh",
                now=_NOW,
                ttl=_TTL,
                bench_version=3,
                submitted_at_or_after=rollout_started,
                fifo_start_at=rollout_started,
            )

        assert ticket is not None
        assert ticket.agent_id == fresh
        assert ticket.agent_id != old

    async def test_new_benchmark_resets_fifo_age_to_rollout_start(
        self, session: AsyncSession
    ) -> None:
        rollout_started = _NOW - timedelta(minutes=5)
        lower_id = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
        higher_id = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
        async with session.begin():
            session.add(
                BenchmarkRollout(
                    rollout_id=uuid4(),
                    from_version=2,
                    desired_version=3,
                    status="activated",
                    cohort_size=5,
                    created_at=rollout_started,
                    activated_at=rollout_started,
                )
            )
            for agent_id, created_at in (
                (higher_id, rollout_started - timedelta(days=2)),
                (lower_id, rollout_started - timedelta(days=1)),
            ):
                session.add(
                    Agent(
                        agent_id=agent_id,
                        miner_hotkey=f"5Miner-{agent_id}",
                        name=str(agent_id),
                        sha256=f"{agent_id.int:064x}",
                        status=AgentStatus.EVALUATING,
                        screening_policy_version=9,
                        screened_image_sha256="12" * 32,
                        screened_image_size_bytes=123,
                        screened_image_id="sha256:" + "34" * 32,
                        screened_image_ref=f"ditto-screen/{agent_id}:latest",
                        screened_image_upload_id=uuid4(),
                        screened_image_verified_at=_NOW,
                        created_at=created_at,
                    )
                )
                session.add(
                    BenchmarkDataset(
                        agent_id=agent_id,
                        bench_version=3,
                        seed=123,
                        sha256="ef" * 32,
                        run_size="full",
                    )
                )

        async with session.begin():
            ticket = await issue_ticket(
                session,
                validator_hotkey="5EraFIFO",
                now=_NOW,
                ttl=_TTL,
                bench_version=3,
            )

        assert ticket is not None
        assert ticket.agent_id == lower_id

    async def test_screened_only_skips_source_only_agent(
        self, session: AsyncSession
    ) -> None:
        source_id = await _seed_evaluating(session, created_at=_NOW)
        image_id = await _seed_evaluating(
            session, created_at=_NOW + timedelta(seconds=1), screened=True
        )
        async with session.begin():
            ticket = await issue_ticket(
                session,
                validator_hotkey="5ImageOnly",
                now=_NOW,
                ttl=_TTL,
                artifact_mode="screened_only",
            )
        assert ticket is not None
        assert ticket.agent_id == image_id
        assert ticket.agent_id != source_id

    async def test_prefer_screened_falls_back_to_source(
        self, session: AsyncSession
    ) -> None:
        source_id = await _seed_evaluating(session)
        async with session.begin():
            ticket = await issue_ticket(
                session,
                validator_hotkey="5Prefer",
                now=_NOW,
                ttl=_TTL,
                artifact_mode="prefer_screened",
            )
        assert ticket is not None
        assert ticket.agent_id == source_id

    async def test_prefer_screened_prioritizes_complete_verified_tuple(
        self, session: AsyncSession
    ) -> None:
        await _seed_evaluating(session, created_at=_NOW)
        image_id = await _seed_evaluating(
            session, created_at=_NOW + timedelta(seconds=1), screened=True
        )
        async with session.begin():
            ticket = await issue_ticket(
                session,
                validator_hotkey="5Prefer",
                now=_NOW,
                ttl=_TTL,
                artifact_mode="prefer_screened",
            )
        assert ticket is not None
        assert ticket.agent_id == image_id

    async def test_screened_only_releases_unstarted_incompatible_lease(
        self, session: AsyncSession
    ) -> None:
        source_id = await _seed_evaluating(session)
        async with session.begin():
            issued = await issue_ticket(
                session, validator_hotkey="5Transition", now=_NOW, ttl=_TTL
            )
        assert issued is not None

        async with session.begin():
            replacement = await issue_ticket(
                session,
                validator_hotkey="5Transition",
                now=_NOW + timedelta(seconds=1),
                ttl=_TTL,
                artifact_mode="screened_only",
                validator_running_benchmark=False,
            )
        assert replacement is None
        async with session.begin():
            released = await session.get(ValidatorTicket, (source_id, 2, "5Transition"))
            assert released is not None
            assert released.status == TicketStatus.EXPIRED

    async def test_screened_only_preserves_actively_running_incompatible_lease(
        self, session: AsyncSession
    ) -> None:
        source_id = await _seed_evaluating(session)
        async with session.begin():
            await issue_ticket(session, validator_hotkey="5Running", now=_NOW, ttl=_TTL)
        async with session.begin():
            replacement = await issue_ticket(
                session,
                validator_hotkey="5Running",
                now=_NOW + timedelta(seconds=1),
                ttl=_TTL,
                artifact_mode="screened_only",
                validator_running_benchmark=True,
            )
        assert replacement is None
        async with session.begin():
            preserved = await session.get(ValidatorTicket, (source_id, 2, "5Running"))
            assert preserved is not None
            assert preserved.status == TicketStatus.ISSUED

    @pytest.mark.parametrize(
        "status",
        (
            AgentStatus.ATH_PENDING_REVIEW,
            AgentStatus.QUARANTINED,
            AgentStatus.REJECTED,
        ),
    )
    async def test_terminal_review_states_do_not_receive_new_tickets(
        self, session: AsyncSession, status: AgentStatus
    ) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            agent = await session.get(Agent, aid)
            assert agent is not None
            agent.status = status

        async with session.begin():
            ticket = await issue_ticket(
                session, validator_hotkey="5V1", now=_NOW, ttl=_TTL
            )

        assert ticket is None

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

    async def test_low_first_score_still_receives_a_second_ticket(
        self, session: AsyncSession
    ) -> None:
        await _seed_finalized_top_five(session)
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
                    composite=0.10,
                    tool_mean=0.10,
                    memory_mean=0.10,
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

    async def test_two_scores_below_top_five_bound_defer_behind_other_work(
        self, session: AsyncSession
    ) -> None:
        """An eliminated 2-of-3 submission yields to every other candidate."""
        await _seed_finalized_top_five(session, fifth_place=0.80)
        below_floor = await _seed_two_scores_below_floor(session)
        # Newer than the eliminated submission, so arrival order alone would
        # still hand the eliminated one out first.
        fresh = await _seed_evaluating(
            session, created_at=_NOW + timedelta(minutes=5), name="fresh"
        )

        async with session.begin():
            ticket = await issue_ticket(
                session, validator_hotkey="5Next", now=_NOW, ttl=_TTL
            )

        assert ticket is not None
        assert ticket.agent_id == fresh
        assert ticket.agent_id != below_floor

    async def test_two_scores_below_top_five_bound_finalize_once_queue_drains(
        self, session: AsyncSession
    ) -> None:
        """Deferred, not withheld: the third score still lands eventually."""
        await _seed_finalized_top_five(session, fifth_place=0.80)
        below_floor = await _seed_two_scores_below_floor(session)

        async with session.begin():
            ticket = await issue_ticket(
                session, validator_hotkey="5Next", now=_NOW, ttl=_TTL
            )

        assert ticket is not None
        assert ticket.agent_id == below_floor

    async def test_score_floor_does_not_cross_benchmark_eras(
        self, session: AsyncSession
    ) -> None:
        """A v2 fifth place must not eliminate a v4 two-score submission.

        Composites only compare within one benchmark version, so a new era with
        fewer than five ranked agents has no floor at all and nothing in it can
        be pre-emptively eliminated.
        """
        await _seed_finalized_top_five(session, fifth_place=0.80)
        aid = await _seed_evaluating(session, screened=True)
        async with session.begin():
            session.add(
                BenchmarkDataset(
                    agent_id=aid,
                    bench_version=4,
                    seed=42,
                    sha256="cd" * 32,
                    run_size="full",
                )
            )
            # Two v4 scores that would sit below the v2-era floor of 0.80.
            for index, composite in enumerate((0.10, 0.20)):
                validator = f"5V4-{index}"
                session.add(
                    ValidatorTicket(
                        agent_id=aid,
                        validator_hotkey=validator,
                        status=TicketStatus.SCORED,
                        issued_at=_NOW,
                        deadline=_NOW + _TTL,
                        bench_version=4,
                        attempt_count=1,
                    )
                )
                session.add(
                    Score(
                        agent_id=aid,
                        validator_hotkey=validator,
                        run_id=f"v4-below-v2-floor-{index}",
                        signature=None,
                        seed=42,
                        composite=composite,
                        tool_mean=composite,
                        memory_mean=composite,
                        median_ms=100,
                        n=119,
                        details=None,
                        bench_version=4,
                        generated_at=_NOW,
                    )
                )

        async with session.begin():
            ticket = await issue_ticket(
                session,
                validator_hotkey="5NextV4",
                now=_NOW,
                ttl=_TTL,
                bench_version=4,
            )

        assert ticket is not None
        assert ticket.agent_id == aid
        assert ticket.bench_version == 4

    async def test_high_variance_two_score_candidate_can_still_reach_top_five(
        self, session: AsyncSession
    ) -> None:
        await _seed_finalized_top_five(session, fifth_place=0.80)
        aid = await _seed_evaluating(session)
        async with session.begin():
            for validator, composite in (("5First", 0.10), ("5Second", 0.90)):
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

    async def test_exact_top_five_bound_continues_for_oldest_first_tie_break(
        self, session: AsyncSession
    ) -> None:
        await _seed_finalized_top_five(session, fifth_place=0.80)
        aid = await _seed_evaluating(session)
        async with session.begin():
            for validator, composite in (("5First", 0.20), ("5Second", 0.80)):
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

    async def test_distinct_slots_receive_distinct_agents(
        self, session: AsyncSession
    ) -> None:
        first = await _seed_evaluating(session, name="parallel-a", created_at=_NOW)
        second = await _seed_evaluating(
            session, name="parallel-b", created_at=_NOW + timedelta(seconds=1)
        )
        async with session.begin():
            slot0 = await issue_ticket(
                session,
                validator_hotkey="5Parallel",
                slot_id="slot-0",
                now=_NOW,
                ttl=_TTL,
            )
            slot1 = await issue_ticket(
                session,
                validator_hotkey="5Parallel",
                slot_id="slot-1",
                now=_NOW,
                ttl=_TTL,
            )
        assert slot0 is not None and slot1 is not None
        assert {slot0.agent_id, slot1.agent_id} == {first, second}
        assert slot0.slot_id == "slot-0"
        assert slot1.slot_id == "slot-1"

    async def test_second_slot_never_duplicates_same_agent(
        self, session: AsyncSession
    ) -> None:
        await _seed_evaluating(session, name="only-agent")
        async with session.begin():
            first = await issue_ticket(
                session,
                validator_hotkey="5Parallel",
                slot_id="slot-0",
                now=_NOW,
                ttl=_TTL,
            )
            duplicate = await issue_ticket(
                session,
                validator_hotkey="5Parallel",
                slot_id="slot-1",
                now=_NOW,
                ttl=_TTL,
            )
        assert first is not None
        assert duplicate is None

    async def test_new_benchmark_era_expires_idle_legacy_ticket(
        self, session: AsyncSession
    ) -> None:
        legacy_agent = await _seed_evaluating(
            session, name="legacy", created_at=_NOW - timedelta(minutes=1)
        )
        current_agent = await _seed_evaluating(
            session, name="current", created_at=_NOW, screened=True
        )
        async with session.begin():
            session.add(
                BenchmarkDataset(
                    agent_id=current_agent,
                    bench_version=3,
                    seed=42,
                    sha256="cd" * 32,
                    run_size="full",
                )
            )
            legacy = await issue_ticket(
                session,
                validator_hotkey="5V1",
                now=_NOW,
                ttl=_TTL,
                bench_version=2,
            )
            assert legacy is not None
            assert legacy.agent_id == legacy_agent

        async with session.begin():
            current = await issue_ticket(
                session,
                validator_hotkey="5V1",
                now=_NOW + timedelta(minutes=1),
                ttl=_TTL,
                bench_version=3,
                artifact_mode="screened_only",
            )

        assert current is not None
        assert current.agent_id == current_agent
        assert current.bench_version == 3
        await session.refresh(legacy)
        assert legacy.status == TicketStatus.EXPIRED

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

    async def test_one_score_round_prioritizes_highest_provisional_score(
        self, session: AsyncSession
    ) -> None:
        low = await _seed_evaluating(
            session, created_at=_NOW - timedelta(hours=1), name="low"
        )
        high = await _seed_evaluating(session, created_at=_NOW, name="high")
        async with session.begin():
            for agent_id, composite in ((low, 0.40), (high, 0.80)):
                validator = f"5Scored-{agent_id}"
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
                        run_id=f"run-{agent_id}",
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
                session, validator_hotkey="5Next", now=_NOW, ttl=_TTL
            )

        assert ticket is not None
        assert ticket.agent_id == high

    async def test_top_provisional_contender_precedes_uncovered_work(
        self, session: AsyncSession
    ) -> None:
        uncovered = await _seed_evaluating(
            session, created_at=_NOW - timedelta(hours=2), name="uncovered"
        )
        contender = await _seed_evaluating(session, created_at=_NOW, name="contender")
        async with session.begin():
            for index, composite in enumerate((0.80, 0.90)):
                validator = f"5Contender-{index}"
                session.add(
                    ValidatorTicket(
                        agent_id=contender,
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
                        agent_id=contender,
                        validator_hotkey=validator,
                        run_id=f"run-contender-{index}",
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
        assert ticket.agent_id == contender
        assert ticket.agent_id != uncovered

    async def test_contender_lane_is_bounded(self, session: AsyncSession) -> None:
        uncovered = await _seed_evaluating(
            session, created_at=_NOW - timedelta(hours=2), name="uncovered"
        )
        async with session.begin():
            for rank in range(PROVISIONAL_CONTENDER_LANE_SIZE + 1):
                contender = Agent(
                    agent_id=uuid4(),
                    miner_hotkey=f"5Miner-{rank}",
                    name=f"contender-{rank}",
                    sha256=f"{rank + 1:064x}",
                    status=AgentStatus.EVALUATING,
                    screening_policy_version=SCREENING_POLICY_VERSION,
                    created_at=_NOW + timedelta(minutes=rank),
                )
                session.add(contender)
                for index in range(2):
                    validator = f"5Scored-{rank}-{index}"
                    composite = 1.0 - rank / 100
                    session.add(
                        ValidatorTicket(
                            agent_id=contender.agent_id,
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
                            agent_id=contender.agent_id,
                            validator_hotkey=validator,
                            run_id=f"run-{rank}-{index}",
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
                if rank < PROVISIONAL_CONTENDER_LANE_SIZE:
                    session.add(
                        ValidatorTicket(
                            agent_id=contender.agent_id,
                            validator_hotkey="5Completion",
                            status=TicketStatus.EXPIRED,
                            issued_at=_NOW - _TTL,
                            deadline=_NOW,
                            bench_version=2,
                            attempt_count=MAX_ATTEMPTS_PER_VERSION,
                            retry_after=_NOW + timedelta(days=1),
                        )
                    )

        async with session.begin():
            ticket = await issue_ticket(
                session, validator_hotkey="5Completion", now=_NOW, ttl=_TTL
            )

        assert ticket is not None
        assert ticket.agent_id == uncovered

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

    async def test_completion_first_finishes_oldest_before_opening_next(
        self, session: AsyncSession
    ) -> None:
        oldest = await _seed_evaluating(session, created_at=_NOW, name="oldest")
        newer = await _seed_evaluating(
            session,
            created_at=_NOW + timedelta(minutes=1),
            name="newer",
        )

        claimed: list[UUID] = []
        async with session.begin():
            for index in range(SCORING_QUORUM):
                ticket = await issue_ticket(
                    session,
                    validator_hotkey=f"5Finish-{index}",
                    now=_NOW,
                    ttl=_TTL,
                    completion_first=True,
                )
                assert ticket is not None
                claimed.append(ticket.agent_id)
            next_ticket = await issue_ticket(
                session,
                validator_hotkey="5Finish-next",
                now=_NOW,
                ttl=_TTL,
                completion_first=True,
            )

        assert claimed == [oldest] * SCORING_QUORUM
        assert next_ticket is not None
        assert next_ticket.agent_id == newer

    async def test_completion_first_does_not_demote_oldest_below_floor(
        self, session: AsyncSession
    ) -> None:
        await _seed_finalized_top_five(session, fifth_place=0.80)
        oldest = await _seed_two_scores_below_floor(session)
        newer = await _seed_evaluating(
            session,
            created_at=_NOW + timedelta(minutes=1),
            name="newer",
        )

        async with session.begin():
            ticket = await issue_ticket(
                session,
                validator_hotkey="5Finish-oldest",
                now=_NOW,
                ttl=_TTL,
                completion_first=True,
            )

        assert ticket is not None
        assert ticket.agent_id == oldest
        assert ticket.agent_id != newer

    async def test_completion_first_second_slot_waits_for_global_oldest(
        self, session: AsyncSession
    ) -> None:
        oldest = await _seed_evaluating(session, created_at=_NOW, name="oldest")
        newer = await _seed_evaluating(
            session,
            created_at=_NOW + timedelta(minutes=1),
            name="newer",
        )

        async with session.begin():
            slot0 = await issue_ticket(
                session,
                validator_hotkey="5ParallelFinish",
                slot_id="slot-0",
                now=_NOW,
                ttl=_TTL,
                completion_first=True,
            )
            slot1 = await issue_ticket(
                session,
                validator_hotkey="5ParallelFinish",
                slot_id="slot-1",
                now=_NOW,
                ttl=_TTL,
                completion_first=True,
            )

        assert slot0 is not None
        assert slot0.agent_id == oldest
        assert slot1 is None
        assert await session.get(ValidatorTicket, (newer, 2, "5ParallelFinish")) is None

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
            ticket = await session.get(ValidatorTicket, (aid, 2, "5V1"))
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
            ticket = await session.get(ValidatorTicket, (aid, 2, "5V1"))
            assert ticket is not None
            ticket.status = TicketStatus.EXPIRED
            ticket.attempt_count = 2
            ticket.retry_after = _NOW + timedelta(days=1)
            agent = await session.get(Agent, aid)
            assert agent is not None
            agent.screened_image_sha256 = "12" * 32
            agent.screened_image_size_bytes = 123
            agent.screened_image_id = "sha256:" + "34" * 32
            agent.screened_image_ref = f"ditto-screen/{aid}:latest"
            agent.screened_image_upload_id = uuid4()
            agent.screened_image_verified_at = _NOW
            session.add(
                BenchmarkDataset(
                    agent_id=aid,
                    bench_version=3,
                    seed=42,
                    sha256="cd" * 32,
                    run_size="full",
                )
            )
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

    async def test_v3_requires_image_while_v2_keeps_source_fallback(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            session.add(
                BenchmarkDataset(
                    agent_id=aid,
                    bench_version=3,
                    seed=42,
                    sha256="cd" * 32,
                    run_size="full",
                )
            )
        async with session.begin():
            assert (
                await issue_ticket(
                    session,
                    validator_hotkey="5V3NoImage",
                    now=_NOW,
                    ttl=_TTL,
                    bench_version=3,
                )
                is None
            )
            v2 = await issue_ticket(
                session,
                validator_hotkey="5V2Fallback",
                now=_NOW,
                ttl=_TTL,
                bench_version=2,
            )
            assert v2 is not None
            assert v2.agent_id == aid

    async def test_prior_scored_version_does_not_block_new_version(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            prior = await issue_ticket(
                session,
                validator_hotkey="5V1",
                now=_NOW,
                ttl=_TTL,
                bench_version=2,
            )
            assert prior is not None
            prior.status = TicketStatus.SCORED
            agent = await session.get(Agent, aid)
            assert agent is not None
            agent.screened_image_sha256 = "12" * 32
            agent.screened_image_size_bytes = 123
            agent.screened_image_id = "sha256:" + "34" * 32
            agent.screened_image_ref = f"ditto-screen/{aid}:latest"
            agent.screened_image_upload_id = uuid4()
            agent.screened_image_verified_at = _NOW
            session.add(
                BenchmarkDataset(
                    agent_id=aid,
                    bench_version=3,
                    seed=42,
                    sha256="cd" * 32,
                    run_size="full",
                )
            )

        async with session.begin():
            current = await issue_ticket(
                session,
                validator_hotkey="5V1",
                now=_LATER,
                ttl=_TTL,
                bench_version=3,
            )

        assert current is not None
        assert current.agent_id == aid
        assert current.bench_version == 3
        assert current.attempt_count == 1

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

    async def test_open_ticket_selects_explicit_version_with_dual_rows(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_evaluating(session)
        async with session.begin():
            session.add_all(
                [
                    ValidatorTicket(
                        agent_id=aid,
                        bench_version=2,
                        validator_hotkey="5V1",
                        status=TicketStatus.SCORED,
                        issued_at=_NOW,
                        deadline=_NOW + _TTL,
                    ),
                    ValidatorTicket(
                        agent_id=aid,
                        bench_version=3,
                        validator_hotkey="5V1",
                        status=TicketStatus.ISSUED,
                        issued_at=_NOW,
                        deadline=_NOW + _TTL,
                    ),
                ]
            )
        async with session.begin():
            ticket = await get_open_ticket(
                session,
                agent_id=aid,
                validator_hotkey="5V1",
                now=_NOW,
                deadline=_NOW + _TTL,
                bench_version=3,
            )
        assert ticket is not None
        assert ticket.bench_version == 3

    async def test_open_ticket_selects_signed_lease_across_versions(
        self, session: AsyncSession
    ) -> None:
        aid = await _seed_evaluating(session)
        v3_deadline = _NOW + _TTL + timedelta(minutes=1)
        async with session.begin():
            session.add_all(
                [
                    ValidatorTicket(
                        agent_id=aid,
                        bench_version=2,
                        validator_hotkey="5V1",
                        status=TicketStatus.SCORED,
                        issued_at=_NOW,
                        deadline=_NOW + _TTL,
                    ),
                    ValidatorTicket(
                        agent_id=aid,
                        bench_version=3,
                        validator_hotkey="5V1",
                        status=TicketStatus.ISSUED,
                        issued_at=_NOW,
                        deadline=v3_deadline,
                    ),
                ]
            )
        async with session.begin():
            ticket = await get_open_ticket(
                session,
                agent_id=aid,
                validator_hotkey="5V1",
                now=_NOW,
                deadline=v3_deadline,
                bench_version=None,
            )
        assert ticket is not None
        assert ticket.bench_version == 3
