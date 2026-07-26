"""The preview and the allocator must rank the queue identically.

Three miner-visible divergences landed in one evening because the public queue
preview restated the allocator's ordering in Python instead of sharing it. The
tests here are the standing guarantee that a fourth cannot: over generated
fixture sets, the row :func:`ditto.db.queries.queue_order.preview_queue_order`
puts first is the row :func:`ditto.db.queries.tickets.issue_ticket` actually
leases.

Each historical divergence also has its own named regression below, seeded so
that the specific confusion that caused it (a coldkey that differs from its
hotkey; a stranded pre-rollout backlog) is present in the data.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.api_models.ticket_status import TicketPurpose, TicketStatus
from ditto.db.models import (
    Agent,
    BenchmarkRollout,
    EvaluationPayment,
    Score,
    SubmissionRetirement,
    ValidatorTicket,
)
from ditto.db.queries.benchmark_admission import activated_rollout_for_version
from ditto.db.queries.queue_order import (
    QueuePreviewEntry,
    preview_artifact_mode,
    preview_queue_order,
    resolve_owner_linkage,
    resolve_owner_linkage_batch,
)
from ditto.db.queries.tickets import get_score_priority_floors, issue_ticket

pytestmark = pytest.mark.asyncio

_NOW = datetime(2026, 7, 25, 12, 0, 0, tzinfo=UTC)
_TTL = timedelta(minutes=90)
_BENCH = 2
_FRESH_VALIDATOR = "5FreshValidatorWithNoPriorTickets"


async def _seed_agent(
    session: AsyncSession,
    *,
    name: str,
    hotkey: str,
    coldkey: str | None,
    created_at: datetime,
    scores: tuple[float, ...] = (),
    live_lease_validator: str | None = None,
) -> UUID:
    """One waiting submission, with the ticket rows its scores imply.

    A recorded score always has an accepted ticket behind it in production --
    a validator cannot post one without holding a lease -- and the allocator's
    contender lane counts accepted tickets, so seeding scores alone would
    quietly disable the lane under test.
    """
    agent_id = uuid4()
    async with session.begin():
        session.add(
            Agent(
                agent_id=agent_id,
                miner_hotkey=hotkey,
                name=name,
                sha256="ab" * 32,
                status=AgentStatus.EVALUATING,
                screening_policy_version=SCREENING_POLICY_VERSION,
                created_at=created_at,
            )
        )
        await session.flush()
        if coldkey is not None:
            session.add(
                EvaluationPayment(
                    block_hash=f"0x{uuid4().hex}",
                    extrinsic_index=0,
                    agent_id=agent_id,
                    miner_hotkey=hotkey,
                    miner_coldkey=coldkey,
                    amount_rao=1,
                    tao_usd_rate=Decimal("1"),
                    dest_address="5Destination",
                    timestamp=created_at,
                )
            )
        for index, composite in enumerate(scores):
            scorer = f"5Scorer{index}"
            session.add(
                Score(
                    agent_id=agent_id,
                    validator_hotkey=scorer,
                    run_id=f"run-{index}",
                    signature=None,
                    seed=index,
                    composite=composite,
                    tool_mean=composite,
                    memory_mean=composite,
                    median_ms=100,
                    n=114,
                    details={"bench_version": _BENCH},
                    generated_at=created_at + timedelta(minutes=index),
                )
            )
            session.add(
                ValidatorTicket(
                    agent_id=agent_id,
                    bench_version=_BENCH,
                    validator_hotkey=scorer,
                    slot_id="slot-0",
                    status=TicketStatus.SCORED,
                    purpose=TicketPurpose.CANONICAL_QUORUM,
                    purpose_revision=1,
                    issued_at=created_at + timedelta(minutes=index),
                    deadline=created_at + timedelta(minutes=index) + _TTL,
                    attempt_count=1,
                )
            )
        if live_lease_validator is not None:
            session.add(
                ValidatorTicket(
                    agent_id=agent_id,
                    bench_version=_BENCH,
                    validator_hotkey=live_lease_validator,
                    slot_id="slot-0",
                    status=TicketStatus.ISSUED,
                    purpose=TicketPurpose.CANONICAL_QUORUM,
                    purpose_revision=1,
                    issued_at=_NOW - timedelta(minutes=5),
                    deadline=_NOW + _TTL,
                    attempt_count=1,
                )
            )
    return agent_id


async def _preview_entries(
    session: AsyncSession,
    *,
    agent_ids: list[UUID],
    previous_generation: set[UUID] | None = None,
) -> dict[UUID, QueuePreviewEntry]:
    """The preview's verdict on ``agent_ids``: rank plus gate."""
    continuation, provisional = await get_score_priority_floors(
        session, bench_version=_BENCH
    )
    return await preview_queue_order(
        session,
        bench_version=_BENCH,
        now=_NOW,
        agent_ids=agent_ids,
        score_continuation_floor=continuation,
        provisional_contender_floor=provisional,
        rollout=await activated_rollout_for_version(session, bench_version=_BENCH),
        previous_generation_agent_ids=previous_generation or set(),
    )


async def _preview(
    session: AsyncSession,
    *,
    agent_ids: list[UUID],
    previous_generation: set[UUID] | None = None,
) -> list[UUID]:
    """The preview's ranking of ``agent_ids``, best first."""
    entries = await _preview_entries(
        session, agent_ids=agent_ids, previous_generation=previous_generation
    )
    return [entry.agent_id for entry in sorted(entries.values(), key=lambda e: e.rank)]


async def _allocator_pick(
    session: AsyncSession, *, validator_hotkey: str = _FRESH_VALIDATOR
) -> UUID | None:
    """What the allocator hands a validator holding no tickets at all.

    A fresh validator is deliberate: the per-validator terms the shared
    ordering cannot carry (``had_prior_ticket``, ``already_mine``, retry
    cooldowns) are all inert for it, so any disagreement with the preview is a
    real divergence rather than a rule the preview never claimed to model.
    """
    # The preview's reads leave an implicit transaction open on this session.
    await session.rollback()
    async with session.begin():
        ticket = await issue_ticket(
            session,
            validator_hotkey=validator_hotkey,
            now=_NOW,
            ttl=_TTL,
            bench_version=_BENCH,
            artifact_mode=preview_artifact_mode(_BENCH),
        )
        return None if ticket is None else ticket.agent_id


class TestPreviewMatchesAllocator:
    """The head of the preview is the row the allocator leases."""

    @pytest.mark.integration
    @pytest.mark.parametrize("seed", range(12))
    async def test_generated_worlds_agree_on_the_next_submission(
        self, session: AsyncSession, seed: int
    ) -> None:
        """Property: over random queues, preview rank 1 == the allocator's pick.

        The generator deliberately produces the shapes the three incidents
        turned on: several hotkeys funded by one coldkey, submissions at every
        score count from zero to quorum-minus-one, live leases, and owners with
        multiple generations.
        """
        rng = random.Random(seed)
        coldkeys = [f"5Coldkey{index}" for index in range(3)]
        agent_ids: list[UUID] = []
        for index in range(rng.randint(4, 9)):
            coldkey = rng.choice(coldkeys)
            score_count = rng.randint(0, 2)
            agent_ids.append(
                await _seed_agent(
                    session,
                    name=f"agent-{index}",
                    # Several hotkeys per coldkey: the #435 shape.
                    hotkey=f"5Hotkey{index % 5}",
                    coldkey=coldkey,
                    created_at=_NOW - timedelta(hours=rng.randint(1, 200)),
                    scores=tuple(
                        round(rng.uniform(0.1, 0.99), 3) for _ in range(score_count)
                    ),
                    live_lease_validator=(
                        f"5Busy{index}" if rng.random() < 0.2 else None
                    ),
                )
            )

        ranked = await _preview(session, agent_ids=agent_ids)
        entries = await _preview_entries(session, agent_ids=agent_ids)
        leasable = [agent_id for agent_id in ranked if entries[agent_id].gate is None]
        picked = await _allocator_pick(session)

        assert picked == (leasable[0] if leasable else None), (
            "the queue preview and the ticket allocator disagree about which "
            "submission is next; they are supposed to share one ordering"
        )

    @pytest.mark.integration
    async def test_owner_serialization_is_visible_rather_than_silent(
        self, session: AsyncSession
    ) -> None:
        """A miner's parked submission is ranked last *and* says why.

        ``issue_ticket`` pins an owner's capacity to whichever generation
        started progressing first, so the sibling never moves. The preview used
        to model none of this and ranked the sibling as though it were next --
        the single largest source of "why isn't mine moving".
        """
        progressing = await _seed_agent(
            session,
            name="progressing",
            hotkey="5OwnerHotkeyOne",
            coldkey="5SharedOwnerColdkey",
            created_at=_NOW - timedelta(hours=10),
            scores=(0.7,),
        )
        # Same owner, different hotkey: rotating hotkeys buys no second slot.
        parked = await _seed_agent(
            session,
            name="parked",
            hotkey="5OwnerHotkeyTwo",
            coldkey="5SharedOwnerColdkey",
            created_at=_NOW - timedelta(hours=9),
        )
        other = await _seed_agent(
            session,
            name="other-miner",
            hotkey="5UnrelatedHotkey",
            coldkey="5UnrelatedColdkey",
            created_at=_NOW - timedelta(hours=1),
        )

        entries = await _preview_entries(
            session, agent_ids=[progressing, parked, other]
        )
        assert entries[parked].gate == "owner_serialized"
        assert entries[other].gate is None
        assert entries[parked].rank > entries[other].rank
        assert await _allocator_pick(session) != parked

    @pytest.mark.integration
    async def test_a_retired_submission_is_excluded_from_both(
        self, session: AsyncSession
    ) -> None:
        """Retirement is a real queue exclusion, not just a public label.

        The exclusion lives in ``queue_candidate_predicate`` rather than at
        ``issue_ticket``'s call site precisely so it cannot hold for one side
        and not the other: a row the allocator will never lease again must not
        be ranked by the preview as though it were next.
        """
        retired = await _seed_agent(
            session,
            name="retired",
            hotkey="5RetiredHotkey",
            coldkey="5RetiredColdkey",
            created_at=_NOW - timedelta(hours=10),
        )
        live = await _seed_agent(
            session,
            name="live",
            hotkey="5LiveHotkey",
            coldkey="5LiveColdkey",
            created_at=_NOW - timedelta(hours=1),
        )
        async with session.begin():
            session.add(
                SubmissionRetirement(
                    retirement_id=uuid4(),
                    agent_id=retired,
                    bench_version=_BENCH,
                    superseded_by_version=_BENCH + 1,
                    actor="operator",
                    reason="the generation this was queued against has closed",
                    expected_snapshot="waiting_validator",
                    score_count=0,
                    ticket_snapshot=[],
                )
            )

        entries = await _preview_entries(session, agent_ids=[retired, live])
        # The preview gates rather than drops, so the row still carries a
        # reason. ``/activity`` never even sends it here -- a retired row is not
        # in the waiting population -- but the two layers must agree if it does.
        assert entries[retired].gate == "not_leasable"
        assert entries[live].gate is None
        assert entries[retired].rank > entries[live].rank
        # The older row would otherwise be leased first; retirement is the only
        # reason the allocator skips it.
        assert await _allocator_pick(session) == live


class TestHistoricalDivergences:
    """One test per divergence that reached a miner."""

    @pytest.mark.integration
    async def test_contender_lane_is_one_slot_per_coldkey_not_per_hotkey(
        self, session: AsyncSession
    ) -> None:
        """#435: the preview deduped contenders by hotkey, the allocator by coldkey.

        The fixture is built so the two groupings produce *different* orders,
        which is the part the original review missed: when every hotkey has its
        own coldkey the two rules are indistinguishable, and the divergence
        shipped. Here one coldkey funds two hotkeys, and the owner's slot is
        pinned to the weaker of them (its progress started first), so the
        surviving row's lane is decided purely by how contenders are grouped.

        By coldkey -- the allocator's rule -- the owner already spent its one
        contender slot on ``strong``, so ``weaker`` drops to the ordinary queue
        and the independent miner's contender outranks it. By hotkey, ``weaker``
        would keep a contender slot of its own and jump the independent miner.
        """
        owner_coldkey = "5OneColdkeyTwoHotkeys"
        weaker = await _seed_agent(
            session,
            name="weaker",
            hotkey="5HotkeyB",
            coldkey=owner_coldkey,
            created_at=_NOW - timedelta(hours=9),
            scores=(0.85,),
        )
        strong = await _seed_agent(
            session,
            name="strong",
            hotkey="5HotkeyA",
            coldkey=owner_coldkey,
            created_at=_NOW - timedelta(hours=5),
            scores=(0.90,),
        )
        independent = await _seed_agent(
            session,
            name="independent",
            hotkey="5HotkeyC",
            coldkey="5IndependentColdkey",
            created_at=_NOW - timedelta(hours=3),
            scores=(0.50,),
        )

        entries = await _preview_entries(
            session, agent_ids=[strong, weaker, independent]
        )

        # The owner's slot is pinned to whichever generation started first.
        assert entries[strong].gate == "owner_serialized"
        assert entries[weaker].gate is None
        # ``strong`` took the coldkey's single contender slot, so ``weaker`` is
        # not a contender and the independent miner's 0.50 outranks its 0.85.
        assert entries[independent].rank < entries[weaker].rank
        assert await _allocator_pick(session) == independent

    @pytest.mark.integration
    async def test_previous_generation_never_holds_the_head_of_the_queue(
        self, session: AsyncSession
    ) -> None:
        """#448: stranded pre-rollout rows outranked every fresh submission.

        They are served only by the carryover and source-backfill lanes, which
        the operator policy holds strictly behind the whole desired era, so a
        preview that ranks them by arrival tells miners the opposite of what
        the fleet will do -- the report that named v6 work as being graded
        ahead of v7.
        """
        rollout_started = _NOW - timedelta(days=2)
        stranded = await _seed_agent(
            session,
            name="stranded",
            hotkey="5StrandedHotkey",
            coldkey="5ColdStranded",
            created_at=rollout_started - timedelta(days=5),
        )
        fresh = await _seed_agent(
            session,
            name="fresh",
            hotkey="5FreshHotkey",
            coldkey="5ColdFresh",
            created_at=rollout_started + timedelta(hours=6),
        )
        async with session.begin():
            session.add(
                BenchmarkRollout(
                    rollout_id=uuid4(),
                    from_version=1,
                    desired_version=_BENCH,
                    status="collecting",
                    cohort_size=5,
                    created_at=rollout_started,
                )
            )

        entries = await _preview_entries(
            session,
            agent_ids=[stranded, fresh],
            previous_generation={stranded},
        )
        assert entries[fresh].rank == 1
        assert entries[stranded].rank == 2
        # And it must not be presentable as imminent at any rank.
        assert entries[stranded].gate == "previous_generation"
        assert entries[fresh].gate is None


class TestOwnerLinkage:
    """The batch resolver the preview uses must match the allocator's."""

    @pytest.mark.integration
    async def test_batch_linkage_matches_the_per_candidate_resolver(
        self, session: AsyncSession
    ) -> None:
        """Two hops, resolved two ways, over a deliberately tangled ledger."""
        agent_ids = [
            await _seed_agent(
                session,
                name="one",
                hotkey="5Rotating",
                coldkey="5ColdOne",
                created_at=_NOW - timedelta(hours=3),
            ),
            await _seed_agent(
                session,
                name="two",
                hotkey="5Rotating",
                coldkey="5ColdTwo",
                created_at=_NOW - timedelta(hours=2),
            ),
            await _seed_agent(
                session,
                name="three",
                hotkey="5Other",
                coldkey="5ColdTwo",
                created_at=_NOW - timedelta(hours=1),
            ),
            # A legacy row with no payment at all.
            await _seed_agent(
                session,
                name="legacy",
                hotkey="5LegacyOnly",
                coldkey=None,
                created_at=_NOW,
            ),
        ]

        batch = await resolve_owner_linkage_batch(session, agent_ids=agent_ids)
        for agent_id in agent_ids:
            assert batch[agent_id] == await resolve_owner_linkage(
                session, agent_id=agent_id
            )
