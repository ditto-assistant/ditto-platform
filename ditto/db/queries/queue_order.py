"""The one definition of validator-queue order.

The miner-facing queue preview and the allocator that actually issues tickets
were two independent implementations of the same ranking: one a hand-rolled
Python ``sorted()`` key in ``ditto/api_server/endpoints/public.py``, the other a
SQL ``ORDER BY`` in :func:`ditto.db.queries.tickets.issue_ticket`. They drifted
three times in one evening -- the preview grouped contenders by hotkey while the
allocator partitioned on the payment-time coldkey (#435), the preview ranked a
stranded previous-generation backlog above every fresh submission (#448), and
the preview never modelled owner serialization at all, which is the single
largest reason a miner's submission does not move.

This module owns the ordering so there is nothing left to sync:

* :func:`queue_order_terms` builds the ``ORDER BY`` expression list. The
  allocator composes it with ``limit(1)`` and a row lock; the preview composes
  the identical list with no limit and an id projection. Same SQL, different
  limit -- not a Python restatement of it.
* :func:`queue_candidate_predicate` builds the validator-independent half of the
  eligibility filter, so "would the allocator even consider this row" has one
  answer too.
* :class:`OwnerLinkage` and the owner-serialization helpers are extracted
  verbatim from the allocator's post-selection loop, so the preview can say
  *why* a submission is parked instead of silently ranking it as if it were
  next.

What a global preview structurally cannot predict
=================================================

``issue_ticket`` answers a *per-validator* question. Three of its rules have no
global truth value and are deliberately absent from the shared ordering:

``had_prior_ticket``
    A coverage tiebreak scoped to one validator: rows this validator has never
    held sort ahead of rows it has. Validator A and validator B genuinely
    disagree about it, so a single global list cannot carry it.
    :func:`queue_order_terms` includes the term only when a ``validator_hotkey``
    is supplied, and the preview supplies none.

Per-validator retry cooldowns and the one-score-per-validator rule
    ``already_mine`` excludes agents this validator holds, has scored, is
    cooling down on, or has exhausted its attempt budget for. Every validator
    has a different exclusion set. A row can therefore be rank 1 globally and
    still be skipped by the next validator to poll.

``artifact_mode``
    A validator running ``prefer_screened`` ranks complete screened images
    ahead of legacy artifacts; a ``legacy`` validator does not. The preview
    derives the mode from the benchmark contract, which makes the lane inert
    whenever the contract requires screened images (every candidate has one) --
    but under a permissive contract the fleet genuinely disagrees.

So this module shares an *ordering*, not a prediction of issuance. The queue
preview is honest about the difference: see ``QueuePreviewEntry.gate`` for the
rows it knows are not leasable at all, and the API docs for the rest.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from sqlalchemy import ColumnElement, and_, case, func, literal, or_, select
from sqlalchemy.orm import aliased

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.benchmark_contract import benchmark_contract
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.api_models.ticket_status import TicketStatus
from ditto.db.models import (
    Agent,
    BenchmarkDataset,
    BenchmarkRollout,
    EvaluationPayment,
    Score,
    ValidatorHeartbeat,
    ValidatorTicket,
)
from ditto.db.queries.benchmark_admission import (
    benchmark_admission_predicate,
    validator_queue_admission_predicate,
)
from ditto.db.queries.retirement import retirement_admission_predicate
from ditto.db.queries.scores import SCORING_QUORUM, emission_owner_key

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

ArtifactMode = Literal["legacy", "prefer_screened", "screened_only"]

AgentEntity = Any
"""``type[Agent] | AliasedClass[Agent]``, spelled loosely for mypy's benefit.

SQLAlchemy's aliased entities are not statically an ``Agent`` subclass, and the
column expressions built from them are what this module actually needs.
"""

# The KOTH champion plus four participation-tail miners receive emissions.
EMISSION_CONTENDER_COUNT = 5

# Advance a bounded set of likely leaderboard contenders before the ordinary
# coverage rounds. Keeping one best submission per miner in this small lane
# lets a strong 1-of-3 result reach 2-of-3 quickly and still finishes strong
# 2-of-3 submissions, without allowing the whole scored backlog to starve new
# miners.
PROVISIONAL_CONTENDER_LANE_SIZE = 10


def preview_artifact_mode(bench_version: int) -> ArtifactMode:
    """The artifact mode a global preview may honestly assume.

    A contract that requires screened images makes the screened-image lane inert
    -- ``queue_candidate_predicate`` has already filtered every candidate that
    lacks one -- so the strictest mode is also the safest one to display. Under
    a permissive contract the mode is genuinely per-validator, and ``legacy``
    (lane constant zero) is the neutral choice rather than a guess about which
    half of the fleet a miner will be served by.
    """
    return (
        "screened_only"
        if benchmark_contract(bench_version).requires_screened_image
        else "legacy"
    )


async def resolve_fifo_start_at(
    session: AsyncSession,
    *,
    bench_version: int,
    rollout: BenchmarkRollout | None,
) -> datetime | None:
    """When this era's FIFO clock starts, for the age tiebreak.

    Arrival times before the era began are clamped to the era's start so a
    stranded backlog cannot claim seniority over the submissions the era is
    actually about. The allocator and the preview must clamp against the same
    instant or their age tiebreak disagrees on every carried-over row, so this
    is the one place that resolves it: the activated rollout's creation time,
    or the latest rollout aiming at this version while it is still settling.
    """
    if rollout is not None:
        return rollout.created_at
    return await session.scalar(
        select(BenchmarkRollout.created_at)
        .where(BenchmarkRollout.desired_version == bench_version)
        .order_by(BenchmarkRollout.created_at.desc())
        .limit(1)
    )


def complete_screened_image_predicate(
    *, agent: AgentEntity = Agent
) -> ColumnElement[bool]:
    """Whether the platform holds a fully verified screened image for ``agent``."""
    return (
        agent.screened_image_sha256.is_not(None)
        & agent.screened_image_size_bytes.is_not(None)
        & agent.screened_image_id.is_not(None)
        & agent.screened_image_ref.is_not(None)
        & agent.screened_image_upload_id.is_not(None)
        & agent.screened_image_verified_at.is_not(None)
    )


def eligible_screened_image_predicate(
    *, bench_version: int, agent: AgentEntity = Agent
) -> ColumnElement[bool]:
    """Complete screened image *and* a screening policy this contract accepts."""
    contract = benchmark_contract(bench_version)
    return complete_screened_image_predicate(agent=agent) & (
        agent.screening_policy_version >= contract.minimum_screening_policy_version
    )


def queue_candidate_predicate(
    *,
    bench_version: int,
    artifact_mode: ArtifactMode,
    rollout: BenchmarkRollout | None,
    submitted_at_or_after: datetime | None = None,
    agent: AgentEntity = Agent,
) -> list[ColumnElement[bool]]:
    """The validator-independent half of ``issue_ticket``'s candidate filter.

    Everything here is true or false for the whole fleet at once: submission
    status, screening policy, queue withdrawal, the versioned dataset pin, the
    screened-artifact contract, and era admission. The per-validator half
    (``already_mine`` -- live leases, recorded scores, retry cooldowns, spent
    attempt budgets) is deliberately excluded; see the module docstring.

    A row the allocator would not even consider must not be ranked as though it
    were next, which is what ``QueuePreviewEntry.gate`` reports.
    """
    contract = benchmark_contract(bench_version)
    predicates: list[ColumnElement[bool]] = [
        agent.status == AgentStatus.EVALUATING,
        agent.screening_policy_version >= SCREENING_POLICY_VERSION,
        validator_queue_admission_predicate(bench_version=bench_version, agent=agent),
        # A retirement can only be written against an era older than the active
        # one, so this is a no-op for current-era work. It is here so that
        # "retired" is a real queue exclusion rather than a label the allocator
        # could disagree with, exactly like a withdrawal -- and because it is
        # fleet-wide, the preview must gate on it too or it would rank a retired
        # row as though it were next.
        retirement_admission_predicate(bench_version=bench_version, agent=agent),
    ]
    if bench_version != 2:
        predicates.append(
            select(BenchmarkDataset.agent_id)
            .where(
                BenchmarkDataset.agent_id == agent.agent_id,
                BenchmarkDataset.bench_version == bench_version,
            )
            .correlate(agent)
            .exists()
        )
    if contract.requires_screened_image or artifact_mode == "screened_only":
        predicates.append(
            eligible_screened_image_predicate(bench_version=bench_version, agent=agent)
        )
    if rollout is not None:
        predicates.append(
            benchmark_admission_predicate(
                rollout=rollout, bench_version=bench_version, agent=agent
            )
        )
    if submitted_at_or_after is not None:
        predicates.append(agent.created_at >= submitted_at_or_after)
    return predicates


def _top_provisional_contenders(
    *, bench_version: int, provisional_contender_floor: float | None
) -> Any:
    """The bounded contender lane: one best submission per emission owner.

    Partitioned on :func:`ditto.db.queries.scores.emission_owner_key` -- the
    payment-time coldkey -- so one coldkey funding several hotkeys holds ONE
    contender slot. Keying this on the hotkey is exactly the divergence #435
    fixed in the preview; there is now one expression to key wrongly.
    """
    contender = aliased(Agent)
    contender_accepted_score_count = (
        select(func.count())
        .where(
            ValidatorTicket.agent_id == contender.agent_id,
            ValidatorTicket.bench_version == bench_version,
            ValidatorTicket.status == TicketStatus.SCORED,
        )
        .correlate(contender)
        .scalar_subquery()
    )
    contender_recorded_score_count = (
        select(func.count(Score.validator_hotkey))
        .where(
            Score.agent_id == contender.agent_id,
            Score.bench_version == bench_version,
        )
        .correlate(contender)
        .scalar_subquery()
    )
    contender_first_score = (
        select(Score.composite)
        .where(
            Score.agent_id == contender.agent_id,
            Score.bench_version == bench_version,
        )
        .order_by(Score.created_at.asc(), Score.validator_hotkey.asc())
        .limit(1)
        .correlate(contender)
        .scalar_subquery()
    )
    contender_provisional_composite = (
        select(func.avg(Score.composite))
        .where(
            Score.agent_id == contender.agent_id,
            Score.bench_version == bench_version,
        )
        .correlate(contender)
        .scalar_subquery()
    )
    contender_payment = aliased(EvaluationPayment)
    contender_owner = emission_owner_key(agent=contender, payment=contender_payment)
    contender_per_miner = (
        select(
            contender.agent_id.label("agent_id"),
            contender.created_at.label("created_at"),
            contender_provisional_composite.label("provisional_composite"),
            func.row_number()
            .over(
                partition_by=contender_owner,
                order_by=(
                    contender_provisional_composite.desc(),
                    contender.created_at.asc(),
                    contender.agent_id.asc(),
                ),
            )
            .label("miner_rank"),
        )
        .outerjoin(
            contender_payment,
            contender_payment.agent_id == contender.agent_id,
        )
        .where(
            contender.status == AgentStatus.EVALUATING,
            contender.screening_policy_version >= SCREENING_POLICY_VERSION,
            contender_accepted_score_count.between(1, SCORING_QUORUM - 1),
            contender_recorded_score_count >= contender_accepted_score_count,
            (
                contender_first_score >= provisional_contender_floor
                if provisional_contender_floor is not None
                else literal(True)
            ),
        )
        .subquery()
    )
    return (
        select(contender_per_miner.c.agent_id)
        .where(contender_per_miner.c.miner_rank == 1)
        .order_by(
            contender_per_miner.c.provisional_composite.desc(),
            contender_per_miner.c.created_at.asc(),
            contender_per_miner.c.agent_id.asc(),
        )
        .limit(PROVISIONAL_CONTENDER_LANE_SIZE)
    )


def queue_order_terms(
    *,
    bench_version: int,
    now: datetime,
    artifact_mode: ArtifactMode,
    fifo_start_at: datetime | None,
    score_continuation_floor: float | None,
    provisional_contender_floor: float | None,
    completion_first: bool = False,
    validator_hotkey: str | None,
    agent: AgentEntity = Agent,
) -> tuple[ColumnElement[Any], ...]:
    """The queue's ``ORDER BY``, for whoever is reading it.

    The allocator composes this with ``.limit(1)`` and a row lock to pick the
    next submission; the miner-facing preview composes the same list with no
    limit to show the order. There is no second implementation to keep in sync.

    ``validator_hotkey=None`` means "rank this for the whole fleet", which drops
    the per-validator ``had_prior_ticket`` coverage tiebreak because no global
    answer to it exists. Every other term is fleet-wide by construction.
    """
    fifo_age = (
        case(
            (agent.created_at < fifo_start_at, fifo_start_at),
            else_=agent.created_at,
        )
        if fifo_start_at is not None
        else agent.created_at
    )
    if completion_first:
        # Keep the fresh-submission lane independent of the ordinary queue's
        # contender, coverage, artifact, and continuation-floor priorities. Age
        # is the contract; UUID is only a stable tie.
        return (fifo_age.asc(), agent.agent_id.asc())

    recorded_score_count = (
        select(func.count(Score.validator_hotkey))
        .where(
            Score.agent_id == agent.agent_id,
            Score.bench_version == bench_version,
        )
        .correlate(agent)
        .scalar_subquery()
    )
    highest_recorded_score = func.coalesce(
        (
            select(func.max(Score.composite))
            .where(
                Score.agent_id == agent.agent_id,
                Score.bench_version == bench_version,
            )
            .correlate(agent)
            .scalar_subquery()
        ),
        0.0,
    )
    provisional_composite = func.coalesce(
        (
            select(func.avg(Score.composite))
            .where(
                Score.agent_id == agent.agent_id,
                Score.bench_version == bench_version,
            )
            .correlate(agent)
            .scalar_subquery()
        ),
        0.0,
    )
    live_assignment_count = (
        select(func.count())
        .where(
            ValidatorTicket.agent_id == agent.agent_id,
            ValidatorTicket.bench_version == bench_version,
            ValidatorTicket.status == TicketStatus.ISSUED,
            ValidatorTicket.deadline > now,
        )
        .correlate(agent)
        .scalar_subquery()
    )
    # A median-of-three cannot be bounded safely after one score. Once two
    # scores exist, their maximum is the best final median the third score can
    # produce, so a submission whose strict upper bound sits below this era's
    # finalized fifth place cannot reach the emission set. That earns it last
    # place in the queue, not removal: the third score still finalizes the
    # submission for the public record, and deferring rather than dropping it
    # means a later floor move (or a new benchmark era, where the old floor
    # never applied) cannot strand it at 2-of-3 forever. When the era has no
    # floor yet, every candidate shares lane 0 and ordering is unchanged.
    below_floor_lane = (
        case(
            (
                (recorded_score_count == SCORING_QUORUM - 1)
                & (highest_recorded_score < score_continuation_floor),
                1,
            ),
            else_=0,
        )
        if score_continuation_floor is not None
        else literal(0)
    )
    top_provisional_contenders = _top_provisional_contenders(
        bench_version=bench_version,
        provisional_contender_floor=provisional_contender_floor,
    )
    contender_lane = case(
        (agent.agent_id.in_(top_provisional_contenders), 0),
        else_=1,
    )
    contender_lane_score = case(
        (agent.agent_id.in_(top_provisional_contenders), provisional_composite),
        else_=0.0,
    )
    overflow_two_score_lane = case(
        (recorded_score_count >= SCORING_QUORUM - 1, 1),
        else_=0,
    )
    screened_image_lane = case(
        (complete_screened_image_predicate(agent=agent), 0),
        else_=(0 if artifact_mode == "legacy" else 1),
    )
    terms: list[ColumnElement[Any]] = [
        below_floor_lane.asc(),
        screened_image_lane.asc(),
        contender_lane.asc(),
        contender_lane_score.desc(),
        # Keep the existing bounded-contender guarantee: a two-score row
        # outside the top contender set must not turn the whole backlog into an
        # unbounded completion lane.
        overflow_two_score_lane.asc(),
        live_assignment_count.asc(),
    ]
    if validator_hotkey is not None:
        had_prior_ticket = (
            select(ValidatorTicket.agent_id)
            .where(
                ValidatorTicket.agent_id == agent.agent_id,
                ValidatorTicket.validator_hotkey == validator_hotkey,
            )
            .correlate(agent)
            .exists()
        )
        terms.append(had_prior_ticket.asc())
    terms.extend((fifo_age.asc(), agent.agent_id.asc()))
    return tuple(terms)


@dataclass(frozen=True)
class OwnerLinkage:
    """Every key that counts as "the same miner" for capacity serialization.

    ``issue_ticket`` enforces one live lease per owner fleet-wide, so rotating
    hotkeys buys no second slot. The rule is a two-hop expansion, not a full
    transitive closure: the payment-time coldkeys ever observed for this
    submission's hotkey (plus its own), and then every hotkey those coldkeys
    have funded. Reproduced exactly rather than approximated, because the
    preview reports "your other submission is holding your slot" on the back of
    it.
    """

    hotkey: str
    coldkeys: frozenset[str]
    hotkeys: frozenset[str]

    @property
    def advisory_lock_keys(self) -> tuple[str, ...]:
        """Owner identities to lock, in the canonical order that avoids deadlock.

        A truly unlinked legacy row falls back to its hotkey.
        """
        if self.coldkeys:
            return tuple(f"coldkey:{coldkey}" for coldkey in sorted(self.coldkeys))
        return (f"hotkey:{self.hotkey}",)

    def same_owner_predicate(
        self, *, agent: AgentEntity, payment: Any
    ) -> ColumnElement[bool]:
        """Match sibling submissions belonging to this owner.

        The caller must have outer-joined ``payment`` to ``agent``.
        """
        if self.coldkeys:
            return or_(
                payment.miner_coldkey.in_(self.coldkeys),
                and_(
                    payment.miner_coldkey.is_(None),
                    agent.miner_hotkey.in_(self.hotkeys),
                ),
            )
        return and_(
            payment.miner_coldkey.is_(None),
            agent.miner_hotkey == self.hotkey,
        )


async def resolve_owner_linkage(
    session: AsyncSession, *, agent_id: UUID
) -> OwnerLinkage:
    """Resolve one submission's owner identity set.

    The allocator's authority, called once per candidate inside the issuing
    transaction. :func:`resolve_owner_linkage_batch` is the preview's
    equivalent over many submissions at once, and a test asserts the two agree.
    """
    owner_hotkey, owner_coldkey = (
        await session.execute(
            select(Agent.miner_hotkey, EvaluationPayment.miner_coldkey)
            .outerjoin(
                EvaluationPayment,
                EvaluationPayment.agent_id == Agent.agent_id,
            )
            .where(Agent.agent_id == agent_id)
        )
    ).one()
    linked_coldkeys = {
        coldkey
        for coldkey in (
            await session.scalars(
                select(EvaluationPayment.miner_coldkey)
                .where(
                    EvaluationPayment.miner_hotkey == owner_hotkey,
                    EvaluationPayment.miner_coldkey.is_not(None),
                )
                .distinct()
            )
        ).all()
        if coldkey is not None
    }
    if owner_coldkey is not None:
        linked_coldkeys.add(owner_coldkey)
    linked_hotkeys = {owner_hotkey}
    if linked_coldkeys:
        linked_hotkeys.update(
            (
                await session.scalars(
                    select(EvaluationPayment.miner_hotkey)
                    .where(EvaluationPayment.miner_coldkey.in_(linked_coldkeys))
                    .distinct()
                )
            ).all()
        )
    return OwnerLinkage(
        hotkey=owner_hotkey,
        coldkeys=frozenset(linked_coldkeys),
        hotkeys=frozenset(linked_hotkeys),
    )


async def resolve_owner_linkage_batch(
    session: AsyncSession, *, agent_ids: Collection[UUID]
) -> dict[UUID, OwnerLinkage]:
    """Resolve owner identity sets for a whole page in three bounded queries.

    Reproduces :func:`resolve_owner_linkage`'s two-hop expansion over the same
    payment ledger, so a preview cannot drift from the allocator's
    per-candidate answer;
    ``test_queue_order_shared.py::test_batch_linkage_matches_the_per_candidate_resolver``
    asserts they agree row for row. Every query is keyed on the requested
    submissions, so the cost tracks the page, not the size of the ledger.
    """
    requested = set(agent_ids)
    if not requested:
        return {}
    agent_rows = (
        await session.execute(
            select(Agent.agent_id, Agent.miner_hotkey, EvaluationPayment.miner_coldkey)
            .outerjoin(
                EvaluationPayment,
                EvaluationPayment.agent_id == Agent.agent_id,
            )
            .where(Agent.agent_id.in_(requested))
        )
    ).all()
    owner_hotkeys = {hotkey for _, hotkey, _ in agent_rows}
    # Hop one: every payment-time coldkey ever observed for these hotkeys.
    coldkeys_by_hotkey: dict[str, set[str]] = {}
    for hotkey, coldkey in (
        await session.execute(
            select(EvaluationPayment.miner_hotkey, EvaluationPayment.miner_coldkey)
            .where(
                EvaluationPayment.miner_hotkey.in_(owner_hotkeys),
                EvaluationPayment.miner_coldkey.is_not(None),
            )
            .distinct()
        )
    ).all():
        coldkeys_by_hotkey.setdefault(hotkey, set()).add(coldkey)
    reachable_coldkeys = {
        coldkey for coldkeys in coldkeys_by_hotkey.values() for coldkey in coldkeys
    }
    reachable_coldkeys.update(
        coldkey for _, _, coldkey in agent_rows if coldkey is not None
    )
    # Hop two: every hotkey those coldkeys have funded. The expansion stops
    # here, exactly as the per-candidate resolver's does -- this is a two-hop
    # rule, not a transitive closure, and widening it would serialize owners the
    # allocator does not.
    hotkeys_by_coldkey: dict[str, set[str]] = {}
    if reachable_coldkeys:
        for hotkey, coldkey in (
            await session.execute(
                select(EvaluationPayment.miner_hotkey, EvaluationPayment.miner_coldkey)
                .where(EvaluationPayment.miner_coldkey.in_(reachable_coldkeys))
                .distinct()
            )
        ).all():
            hotkeys_by_coldkey.setdefault(coldkey, set()).add(hotkey)
    linkage: dict[UUID, OwnerLinkage] = {}
    for agent_id, owner_hotkey, owner_coldkey in agent_rows:
        linked_coldkeys = set(coldkeys_by_hotkey.get(owner_hotkey, ()))
        if owner_coldkey is not None:
            linked_coldkeys.add(owner_coldkey)
        linked_hotkeys = {owner_hotkey}
        for coldkey in linked_coldkeys:
            linked_hotkeys.update(hotkeys_by_coldkey.get(coldkey, ()))
        linkage[agent_id] = OwnerLinkage(
            hotkey=owner_hotkey,
            coldkeys=frozenset(linked_coldkeys),
            hotkeys=frozenset(linked_hotkeys),
        )
    return linkage


async def owner_live_lease_agent_ids(
    session: AsyncSession,
    *,
    linkage: OwnerLinkage,
    now: datetime,
) -> set[UUID]:
    """Which of this owner's submissions currently hold a live lease.

    One paid owner may have many generations, but only one generation may
    occupy validator capacity at a time, so a non-empty result minus the
    candidate itself is what blocks issuance. Returning the set rather than a
    count lets the preview answer the question once per owner instead of once
    per row, from the same statement the allocator uses.
    """
    sibling_agent = aliased(Agent)
    sibling_payment = aliased(EvaluationPayment)
    return set(
        await session.scalars(
            select(sibling_agent.agent_id)
            .select_from(ValidatorTicket)
            .join(sibling_agent, sibling_agent.agent_id == ValidatorTicket.agent_id)
            .outerjoin(
                sibling_payment,
                sibling_payment.agent_id == sibling_agent.agent_id,
            )
            .where(
                ValidatorTicket.status == TicketStatus.ISSUED,
                ValidatorTicket.deadline > now,
                linkage.same_owner_predicate(
                    agent=sibling_agent, payment=sibling_payment
                ),
            )
            .distinct()
        )
    )


async def quorum_capable_validator_hotkeys(
    session: AsyncSession, *, bench_version: int, now: datetime
) -> set[str]:
    """Validators whose fresh signed heartbeat advertises ``bench_version``.

    The fleet that could actually contribute a score at this version, which is
    the denominator for asking whether a submission can still reach quorum.
    Same notion of "capable" the previous-generation gate feeds into
    :func:`~ditto.db.queries.desired_era_backlog.desired_era_work_outstanding`.

    Read once per issuance or preview and passed down, so the owner loop in
    :func:`preview_queue_order` does not re-read the heartbeat table per owner.
    """
    from ditto.db.queries.benchmark_rollout import heartbeat_supports_version

    return {
        heartbeat.validator_hotkey
        for heartbeat in (await session.scalars(select(ValidatorHeartbeat))).all()
        if heartbeat_supports_version(heartbeat, now=now, version=bench_version)
    }


async def selected_owner_agent_id(
    session: AsyncSession,
    *,
    linkage: OwnerLinkage,
    bench_version: int,
    now: datetime,
    provisional_contender_floor: float | None,
    rollout: BenchmarkRollout | None,
    capable_validator_hotkeys: Collection[str],
) -> UUID | None:
    """This owner's pinned generation, or ``None`` when none has started.

    Keeps one current-era generation selected across the gaps between its
    leases. Otherwise a validator that already scored the selected row can open
    a sibling after the last lease becomes SCORED, and that new lease diverts
    every eligible validator away from finishing the first generation.
    Historical overlaps converge deterministically on the generation whose
    accepted/live progress began first. Expired-only attempts do not pin an
    owner, so failed work can still drain -- and neither does a generation that
    can no longer reach quorum, see ``owner_quorum_reachable`` below.
    """
    sibling_agent = aliased(Agent)
    sibling_payment = aliased(EvaluationPayment)
    owner_progress_started_at = (
        select(func.min(ValidatorTicket.issued_at))
        .where(
            ValidatorTicket.agent_id == sibling_agent.agent_id,
            ValidatorTicket.bench_version == bench_version,
            (
                (ValidatorTicket.status == TicketStatus.SCORED)
                | (
                    (ValidatorTicket.status == TicketStatus.ISSUED)
                    & (ValidatorTicket.deadline > now)
                )
            ),
        )
        .correlate(sibling_agent)
        .scalar_subquery()
    )
    owner_first_score = (
        select(Score.composite)
        .where(
            Score.agent_id == sibling_agent.agent_id,
            Score.bench_version == bench_version,
        )
        .order_by(Score.created_at.asc(), Score.validator_hotkey.asc())
        .limit(1)
        .correlate(sibling_agent)
        .scalar_subquery()
    )
    # Pinning asks "did this generation start progress?" -- but the slot it
    # holds is only worth holding if the generation can still *finish*. A
    # capable validator that has spent its retry budget on a submission is a
    # quorum slot nobody can ever fill, so once enough of them are gone the
    # submission is structurally dead: it can never reach SCORING_QUORUM, yet
    # "it started first" kept it pinned and left the owner's healthy siblings
    # unleasable while fleet slots sat idle. With three v7-capable validators
    # and a quorum of three, one exhausted validator is already fatal.
    #
    # A live lease, a recorded score, and a retry cooldown all still lead to a
    # score, so none of them subtract from the ceiling -- only a spent budget
    # does. That is deliberately narrower than
    # ``desired_era_work_outstanding``'s ``blocked_validators``, which asks
    # "could a validator take this agent *right now*"; reusing that here would
    # unpin a generation that is merely mid-lease or cooling down, which is
    # exactly the mid-quorum diversion pinning exists to prevent. Both share
    # one definition of a spent budget in :func:`retry_budget_spent`.
    #
    # Imported lazily because :mod:`ditto.db.queries.tickets` imports this
    # module; the budget constant belongs beside the retry rules it caps.
    from ditto.db.queries.tickets import retry_budget_spent

    owner_scored_count = (
        select(func.count(func.distinct(ValidatorTicket.validator_hotkey)))
        .where(
            ValidatorTicket.agent_id == sibling_agent.agent_id,
            ValidatorTicket.bench_version == bench_version,
            ValidatorTicket.status == TicketStatus.SCORED,
        )
        .correlate(sibling_agent)
        .scalar_subquery()
    )
    # Capable validators already counted in ``owner_scored_count`` or gone for
    # good. Subtracting them from the capable fleet leaves the validators that
    # have yet to contribute and still can, so the sum below is a true ceiling
    # with nothing double-counted.
    owner_spent_capable = (
        select(func.count(func.distinct(ValidatorTicket.validator_hotkey)))
        .where(
            ValidatorTicket.agent_id == sibling_agent.agent_id,
            ValidatorTicket.bench_version == bench_version,
            ValidatorTicket.validator_hotkey.in_(set(capable_validator_hotkeys)),
            or_(
                ValidatorTicket.status == TicketStatus.SCORED,
                and_(
                    ValidatorTicket.status == TicketStatus.EXPIRED,
                    retry_budget_spent(),
                ),
            ),
        )
        .correlate(sibling_agent)
        .scalar_subquery()
    )
    capable_count = len(set(capable_validator_hotkeys))
    # Only provable when the visible capable fleet is itself at least
    # quorum-sized. Below that the fleet is the constraint, not the submission,
    # and the term would condemn every generation at once -- so it switches off
    # and the pin behaves exactly as it did before.
    owner_quorum_reachable = (
        (owner_scored_count + (capable_count - owner_spent_capable)) >= SCORING_QUORUM
        if capable_count >= SCORING_QUORUM
        else literal(True)
    )
    return await session.scalar(
        select(sibling_agent.agent_id)
        .outerjoin(
            sibling_payment,
            sibling_payment.agent_id == sibling_agent.agent_id,
        )
        .where(
            sibling_agent.status == AgentStatus.EVALUATING,
            linkage.same_owner_predicate(agent=sibling_agent, payment=sibling_payment),
            owner_progress_started_at.is_not(None),
            owner_quorum_reachable,
            (
                owner_first_score >= provisional_contender_floor
                if provisional_contender_floor is not None
                else literal(True)
            ),
            (
                benchmark_admission_predicate(
                    rollout=rollout,
                    bench_version=bench_version,
                    agent=sibling_agent,
                )
                if rollout is not None
                else literal(True)
            ),
            validator_queue_admission_predicate(
                bench_version=bench_version,
                agent=sibling_agent,
            ),
            # A retired generation must not hold the owner's one slot. It can
            # never be leased again, so pinning it would strand every healthy
            # sibling behind a submission that can no longer finish -- the same
            # failure ``owner_quorum_reachable`` above exists to prevent.
            retirement_admission_predicate(
                bench_version=bench_version,
                agent=sibling_agent,
            ),
        )
        .order_by(
            owner_progress_started_at.asc(),
            sibling_agent.created_at.asc(),
            sibling_agent.agent_id.asc(),
        )
        .limit(1)
    )


QueueGate = Literal["previous_generation", "owner_serialized", "not_leasable"]
"""Why a ranked submission cannot be leased on the next poll, if it cannot.

``previous_generation``
    Served only by the carryover and source-backfill lanes, which the operator
    policy holds strictly behind every desired-era submission.
``owner_serialized``
    Another generation from the same paid owner occupies the owner's one slot.
``not_leasable``
    The allocator's validator-independent candidate filter excludes it: no
    versioned dataset, no eligible screened image, withdrawn from the queue, or
    not admitted to this era.

``None`` means the row is genuinely leasable right now, subject only to the
per-validator checks a global preview cannot represent.
"""

# Gates sort behind un-gated work, in the order the fleet will reach them.
# ``owner_serialized`` outranks ``not_leasable`` because the owner's slot frees
# on its own; ``previous_generation`` is last because the policy gate holds it
# behind the entire desired era.
_GATE_RANK: dict[QueueGate | None, int] = {
    None: 0,
    "owner_serialized": 1,
    "not_leasable": 2,
    "previous_generation": 3,
}


@dataclass(frozen=True)
class QueuePreviewEntry:
    """One submission's place in the fleet-wide queue preview."""

    agent_id: UUID
    rank: int
    gate: QueueGate | None

    @property
    def leasable(self) -> bool:
        """Whether any validator could take this row on its next poll."""
        return self.gate is None


async def preview_queue_order(
    session: AsyncSession,
    *,
    bench_version: int,
    now: datetime,
    agent_ids: Sequence[UUID],
    score_continuation_floor: float | None,
    provisional_contender_floor: float | None,
    rollout: BenchmarkRollout | None,
    previous_generation_agent_ids: Collection[UUID] = (),
) -> dict[UUID, QueuePreviewEntry]:
    """Rank ``agent_ids`` the way the allocator would, and say what is gated.

    The ranking is :func:`queue_order_terms` executed as SQL -- the identical
    expression list ``issue_ticket`` orders by, without its ``limit(1)`` -- so
    the preview cannot drift from the allocator by construction. Gate lanes are
    then applied ahead of it, which is equivalent to prefixing them to the
    ``ORDER BY`` and keeps Python-derived sets out of the shared expression.

    Ranks are advisory. See the module docstring for the per-validator rules no
    global list can carry.
    """
    requested = list(dict.fromkeys(agent_ids))
    if not requested:
        return {}
    artifact_mode = preview_artifact_mode(bench_version)
    ordered = list(
        await session.scalars(
            select(Agent.agent_id)
            .where(Agent.agent_id.in_(requested))
            .order_by(
                *queue_order_terms(
                    bench_version=bench_version,
                    now=now,
                    artifact_mode=artifact_mode,
                    fifo_start_at=await resolve_fifo_start_at(
                        session, bench_version=bench_version, rollout=rollout
                    ),
                    score_continuation_floor=score_continuation_floor,
                    provisional_contender_floor=provisional_contender_floor,
                    validator_hotkey=None,
                )
            )
        )
    )
    leasable = set(
        await session.scalars(
            select(Agent.agent_id).where(
                Agent.agent_id.in_(requested),
                *queue_candidate_predicate(
                    bench_version=bench_version,
                    artifact_mode=artifact_mode,
                    rollout=rollout,
                ),
            )
        )
    )
    previous_generation = set(previous_generation_agent_ids)
    linkage = await resolve_owner_linkage_batch(session, agent_ids=requested)
    # Owner serialization is a property of the owner, not of the row, so both
    # questions are asked once per distinct owner rather than once per rank.
    selected_by_owner: dict[tuple[str, ...], UUID | None] = {}
    leased_by_owner: dict[tuple[str, ...], set[UUID]] = {}
    # Read once for the whole preview: the pin's reachability term needs the
    # capable fleet, and it is the same fleet for every owner in this pass.
    capable_hotkeys = await quorum_capable_validator_hotkeys(
        session, bench_version=bench_version, now=now
    )
    gates: dict[UUID, QueueGate | None] = {}
    for agent_id in ordered:
        if agent_id in previous_generation:
            gates[agent_id] = "previous_generation"
            continue
        if agent_id not in leasable:
            gates[agent_id] = "not_leasable"
            continue
        owner = linkage.get(agent_id)
        if owner is None:
            gates[agent_id] = None
            continue
        owner_key = owner.advisory_lock_keys
        if owner_key not in selected_by_owner:
            selected_by_owner[owner_key] = await selected_owner_agent_id(
                session,
                linkage=owner,
                bench_version=bench_version,
                now=now,
                provisional_contender_floor=provisional_contender_floor,
                rollout=rollout,
                capable_validator_hotkeys=capable_hotkeys,
            )
            leased_by_owner[owner_key] = await owner_live_lease_agent_ids(
                session, linkage=owner, now=now
            )
        selected = selected_by_owner[owner_key]
        if selected is not None and selected != agent_id:
            gates[agent_id] = "owner_serialized"
            continue
        if leased_by_owner[owner_key] - {agent_id}:
            gates[agent_id] = "owner_serialized"
            continue
        gates[agent_id] = None
    ranked = sorted(
        enumerate(ordered),
        key=lambda item: (_GATE_RANK[gates[item[1]]], item[0]),
    )
    return {
        agent_id: QueuePreviewEntry(agent_id=agent_id, rank=rank, gate=gates[agent_id])
        for rank, (_, agent_id) in enumerate(ranked, start=1)
    }
