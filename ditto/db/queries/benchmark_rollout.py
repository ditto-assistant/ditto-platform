"""Durable, version-separated benchmark activation state machine."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from pydantic import ValidationError
from sqlalchemy import func, select

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.benchmark_contract import (
    benchmark_contract,
    latest_benchmark_contract,
)
from ditto.api_models.ticket_status import TicketStatus
from ditto.api_models.validator_capabilities import (
    ValidatorCapabilities,
    ValidatorStackIdentity,
)
from ditto.db.models import (
    Agent,
    BenchmarkDataset,
    BenchmarkRollout,
    BenchmarkRolloutAudit,
    BenchmarkRolloutMember,
    Score,
    ValidatorHeartbeat,
    ValidatorTicket,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


# The version a rollout starts FROM when no rollout has ever activated. This
# moves forward as benchmarks activate.
DEFAULT_BENCH_VERSION = 2
# What a version-less report from a pre-bench_version validator actually ran.
# This is a statement about history and is frozen at 2 forever: it must NOT
# follow DEFAULT_BENCH_VERSION, or a future rollout-from bump would silently
# reinterpret every legacy submission as a newer benchmark.
LEGACY_BENCH_VERSION = 2
# Compatibility name for callers/tests that need the newest *shipped* contract.
# This is discovery metadata only: it no longer opens or selects a rollout.
CANARY_BENCH_VERSION = latest_benchmark_contract().version
COHORT_SIZE = 5
SCORING_QUORUM = 3
# How many agents must hold a COMPLETE, ranked desired-version quorum before
# the desired version may take over. Two gates enforce it against the same count
# (``ditto.db.queries.scores.count_ranked_quorum_agents``): the ledger's
# authority switch (``list_eligible_ledger``) and rollout activation
# (``maybe_activate_rollout``), which is where the ledger gate stops applying.
#
# Derived, not guessed: it is exactly the size of the KOTH emission set — the
# champion plus the participation tail. Below that count, flipping the ledger to
# the desired version would drop agents that have no desired-version quorum yet,
# and the fold would have fewer recipients than the emission split expects, so
# emissions would go sparse mid-rollout. Deriving it from KOTH_TAIL_SIZE keeps
# the two from drifting apart if the tail is ever resized.
#
# The value is ``1 (champion) + KOTH_TAIL_SIZE`` from ``ditto.api_server.koth``
# (which mirrors the frozen consensus constants of ditto-subnet's
# ``ditto/validator/weights.py`` / ``config.py``). It is spelled out rather than
# imported because ``ditto.api_server`` imports this module, so importing back
# from it here is a cycle; ``test_min_desired_authority_matches_koth_recipients``
# asserts the equality, so resizing the tail without resizing this fails CI.
MIN_DESIRED_AUTHORITY_AGENTS = 5


class RolloutConflictError(RuntimeError):
    """Raised when a rollout cannot be opened because another one is open.

    ``benchmark_rollouts_one_open_idx`` enforces this in the database; catching
    the condition first turns an opaque IntegrityError 500 into a clean 409.
    """


@dataclass(frozen=True)
class RolloutSnapshotMember:
    agent_id: UUID
    miner_hotkey: str
    composite: float


@dataclass(frozen=True)
class DatasetPin:
    seed: int
    sha256: str
    run_size: str
    seed_block: int | None = None
    seed_block_hash: str | None = None


async def rolling_top_five(session: AsyncSession) -> list[RolloutSnapshotMember]:
    """Return the hybrid top five for the durable rollout transition.

    While a rollout is open, an agent remains ranked by the rollout's source
    median until it has a complete target-version quorum. At quorum its target
    median atomically replaces the source median for this ranking. With no open
    rollout, only the active version is authoritative. No compiled "canary"
    constant is allowed to select or open a benchmark transition.
    """
    rollout = await open_rollout(session)
    source_version = rollout.from_version if rollout is not None else None
    if source_version is None:
        source_version = await active_bench_version(session)
    target_version = rollout.desired_version if rollout is not None else None
    agents = list(
        await session.scalars(select(Agent).where(Agent.status == AgentStatus.SCORED))
    )
    if not agents:
        return []
    scores = list(
        await session.scalars(
            select(Score).where(
                Score.agent_id.in_([agent.agent_id for agent in agents]),
                Score.bench_version.in_(
                    (source_version, target_version)
                    if target_version is not None
                    else (source_version,)
                ),
            )
        )
    )
    by_agent: dict[UUID, dict[int, list[Score]]] = {}
    for score in scores:
        by_agent.setdefault(score.agent_id, {}).setdefault(
            score.bench_version, []
        ).append(score)

    candidates: list[tuple[Agent, float]] = []
    for agent in agents:
        versions = by_agent.get(agent.agent_id, {})
        selected = (
            versions.get(target_version, []) if target_version is not None else []
        )
        if target_version is None or len(selected) < SCORING_QUORUM:
            selected = versions.get(source_version, [])
        if not selected:
            continue
        middle = sorted(
            selected, key=lambda row: (row.composite, row.validator_hotkey)
        )[(len(selected) - 1) // 2]
        if middle.n < 100 or middle.composite <= 0:
            continue
        candidates.append((agent, float(middle.composite)))

    # Keep one best agent per miner before taking the network-wide top five.
    candidates.sort(key=lambda item: (-item[1], item[0].created_at, item[0].agent_id))
    unique: list[RolloutSnapshotMember] = []
    seen_miners: set[str] = set()
    for agent, composite in candidates:
        if agent.miner_hotkey in seen_miners:
            continue
        seen_miners.add(agent.miner_hotkey)
        unique.append(
            RolloutSnapshotMember(agent.agent_id, agent.miner_hotkey, composite)
        )
        if len(unique) == COHORT_SIZE:
            break
    return unique


async def append_rollout_member(
    session: AsyncSession,
    *,
    rollout: BenchmarkRollout,
    member: RolloutSnapshotMember,
    dataset: DatasetPin,
    now: datetime,
    audit_context: dict[str, Any] | None = None,
) -> bool:
    """Permanently qualify one newly risen hybrid-top-five agent."""
    locked = await session.get(
        BenchmarkRollout, rollout.rollout_id, with_for_update=True
    )
    assert locked is not None
    existing = await session.get(
        BenchmarkRolloutMember, (rollout.rollout_id, member.agent_id)
    )
    if existing is not None or locked.status not in (
        "collecting",
        "blocked_ineligible",
    ):
        return False
    if locked.status == "blocked_ineligible":
        locked.status = "collecting"
        locked.blocked_reason = None
    position = (
        int(
            await session.scalar(
                select(
                    func.coalesce(func.max(BenchmarkRolloutMember.position), 0)
                ).where(BenchmarkRolloutMember.rollout_id == rollout.rollout_id)
            )
        )
        + 1
    )
    session.add(
        BenchmarkRolloutMember(
            rollout_id=rollout.rollout_id,
            agent_id=member.agent_id,
            position=position,
            frozen_miner_hotkey=member.miner_hotkey,
            frozen_composite=member.composite,
        )
    )
    existing_dataset = await session.get(
        BenchmarkDataset, (member.agent_id, locked.desired_version)
    )
    if existing_dataset is None:
        session.add(
            BenchmarkDataset(
                agent_id=member.agent_id,
                bench_version=locked.desired_version,
                seed=dataset.seed,
                sha256=dataset.sha256,
                run_size=dataset.run_size,
                seed_block=dataset.seed_block,
                seed_block_hash=dataset.seed_block_hash,
                created_at=now,
            )
        )
    elif (
        existing_dataset.seed,
        existing_dataset.sha256,
        existing_dataset.run_size,
        existing_dataset.seed_block,
        existing_dataset.seed_block_hash,
    ) != (
        dataset.seed,
        dataset.sha256,
        dataset.run_size,
        dataset.seed_block,
        dataset.seed_block_hash,
    ):
        raise ValueError("existing benchmark dataset does not match qualification")
    audit_payload: dict[str, Any] = {
        "agent_id": str(member.agent_id),
        "position": position,
        "hybrid_composite": member.composite,
        "dataset_seed": dataset.seed,
        "dataset_sha256": dataset.sha256,
        "origin": "automatic",
    }
    if audit_context is not None:
        audit_payload.update(audit_context)
    await _audit(
        session,
        locked,
        "member_qualified",
        audit_payload,
        now=now,
    )
    await session.flush()
    return True


async def active_bench_version(session: AsyncSession) -> int:
    version = await session.scalar(
        select(BenchmarkRollout.desired_version)
        .where(BenchmarkRollout.status == "activated")
        .order_by(BenchmarkRollout.activated_at.desc())
        .limit(1)
    )
    return int(version or DEFAULT_BENCH_VERSION)


async def open_rollout(
    session: AsyncSession, *, for_update: bool = False
) -> BenchmarkRollout | None:
    statement = (
        select(BenchmarkRollout)
        .where(BenchmarkRollout.status.in_(("collecting", "blocked_ineligible")))
        .limit(1)
    )
    if for_update:
        statement = statement.with_for_update()
    return await session.scalar(statement)


async def rollout_for_transition(
    session: AsyncSession,
    *,
    from_version: int,
    desired_version: int = CANARY_BENCH_VERSION,
    for_update: bool = False,
) -> BenchmarkRollout | None:
    """Return the durable row for one transition, including after activation."""
    statement = (
        select(BenchmarkRollout)
        .where(
            BenchmarkRollout.from_version == from_version,
            BenchmarkRollout.desired_version == desired_version,
        )
        .limit(1)
    )
    if for_update:
        statement = statement.with_for_update()
    return await session.scalar(statement)


async def rollout_for_desired_version(
    session: AsyncSession, *, desired_version: int
) -> BenchmarkRollout | None:
    """Return the durable row targeting ``desired_version``, whatever it came from."""
    return await session.scalar(
        select(BenchmarkRollout)
        .where(BenchmarkRollout.desired_version == desired_version)
        .order_by(BenchmarkRollout.created_at.desc())
        .limit(1)
    )


async def _audit(
    session: AsyncSession,
    rollout: BenchmarkRollout,
    event: str,
    payload: dict[str, Any],
    *,
    now: datetime,
) -> None:
    session.add(
        BenchmarkRolloutAudit(
            audit_id=uuid4(),
            rollout_id=rollout.rollout_id,
            event=event,
            payload=payload,
            recorded_at=now,
        )
    )


async def supersede_open_rollout(
    session: AsyncSession,
    *,
    actor: str,
    reason: str,
    now: datetime,
) -> BenchmarkRollout | None:
    """Terminally abandon the open rollout so the next one can be opened.

    Returns ``None`` when no rollout is open. Refuses to touch an ``activated``
    rollout: activation already moved chain weights and published the retired
    corpus, so rewriting it would be rewriting history. The partial open index
    excludes ``superseded``, so the single open slot is freed immediately.
    """
    # Deliberately the LATEST row rather than open_rollout(): an activated
    # rollout must produce an explicit refusal, not a misleading "nothing open".
    rollout = await session.scalar(
        select(BenchmarkRollout)
        .order_by(BenchmarkRollout.created_at.desc())
        .limit(1)
        .with_for_update()
    )
    if rollout is None or rollout.status == "superseded":
        return None
    if rollout.status == "activated":
        raise RolloutConflictError(
            "an activated benchmark rollout cannot be superseded"
        )
    previous_status = rollout.status
    rollout.status = "superseded"
    rollout.blocked_reason = None
    await _audit(
        session,
        rollout,
        "superseded",
        {
            "actor": actor,
            "reason": reason,
            "previous_status": previous_status,
            "from_version": rollout.from_version,
            "desired_version": rollout.desired_version,
        },
        now=now,
    )
    await session.flush()
    return rollout


async def create_rollout_snapshot(
    session: AsyncSession,
    *,
    members: Sequence[RolloutSnapshotMember],
    datasets: dict[UUID, DatasetPin],
    now: datetime,
    from_version: int = DEFAULT_BENCH_VERSION,
    desired_version: int = CANARY_BENCH_VERSION,
    audit_context: dict[str, Any] | None = None,
) -> BenchmarkRollout:
    """Freeze exactly five ranked agents and their target dataset pins, idempotently."""
    if desired_version <= from_version:
        raise ValueError("a benchmark rollout must move the version forward")
    if session.get_bind().dialect.name == "postgresql":
        # One global rollout lock name, deliberately NOT keyed on the version:
        # only one rollout may be open at a time, so every transition must
        # serialise against every other. The legacy literal is kept so a
        # mid-deploy mix of old and new code still shares the same lock.
        await session.execute(
            select(
                func.pg_advisory_xact_lock(
                    func.hashtextextended("benchmark-v3-rollout", 0)
                )
            )
        )
    existing = await rollout_for_transition(
        session,
        from_version=from_version,
        desired_version=desired_version,
        for_update=True,
    )
    if existing is not None:
        return existing
    conflicting = await open_rollout(session, for_update=True)
    if conflicting is not None:
        raise RolloutConflictError(
            f"benchmark rollout {conflicting.from_version}->"
            f"{conflicting.desired_version} is still {conflicting.status}; only one "
            "benchmark rollout may be open at a time"
        )
    if len(members) != COHORT_SIZE:
        raise ValueError("a benchmark rollout requires exactly five members")
    if len({m.agent_id for m in members}) != COHORT_SIZE:
        raise ValueError("benchmark rollout agents must be distinct")
    if len({m.miner_hotkey for m in members}) != COHORT_SIZE:
        raise ValueError("benchmark rollout miners must be distinct")
    if set(datasets) != {m.agent_id for m in members}:
        raise ValueError("every frozen rollout member requires one target dataset pin")

    rollout = BenchmarkRollout(
        rollout_id=uuid4(),
        from_version=from_version,
        desired_version=desired_version,
        status="collecting",
        cohort_size=COHORT_SIZE,
        created_at=now,
    )
    session.add(rollout)
    for position, member in enumerate(members, start=1):
        session.add(
            BenchmarkRolloutMember(
                rollout_id=rollout.rollout_id,
                agent_id=member.agent_id,
                position=position,
                frozen_miner_hotkey=member.miner_hotkey,
                frozen_composite=member.composite,
            )
        )
        pin = datasets[member.agent_id]
        existing_dataset = await session.get(
            BenchmarkDataset, (member.agent_id, desired_version)
        )
        if existing_dataset is None:
            session.add(
                BenchmarkDataset(
                    agent_id=member.agent_id,
                    bench_version=desired_version,
                    seed=pin.seed,
                    sha256=pin.sha256,
                    run_size=pin.run_size,
                    seed_block=pin.seed_block,
                    seed_block_hash=pin.seed_block_hash,
                    created_at=now,
                )
            )
        elif (
            existing_dataset.seed,
            existing_dataset.sha256,
            existing_dataset.run_size,
            existing_dataset.seed_block,
            existing_dataset.seed_block_hash,
        ) != (
            pin.seed,
            pin.sha256,
            pin.run_size,
            pin.seed_block,
            pin.seed_block_hash,
        ):
            raise ValueError("existing benchmark dataset does not match snapshot")
    audit_payload: dict[str, Any] = {
        "agent_ids": [str(member.agent_id) for member in members]
    }
    if audit_context:
        audit_payload.update(audit_context)
    await _audit(session, rollout, "cohort_frozen", audit_payload, now=now)
    await session.flush()
    return rollout


async def _validate_frozen_members(
    session: AsyncSession, rollout: BenchmarkRollout, *, now: datetime
) -> bool:
    rows = (
        await session.execute(
            select(BenchmarkRolloutMember, Agent)
            .join(Agent, Agent.agent_id == BenchmarkRolloutMember.agent_id)
            .where(BenchmarkRolloutMember.rollout_id == rollout.rollout_id)
            .order_by(BenchmarkRolloutMember.position)
        )
    ).all()
    invalid = [
        str(member.agent_id)
        for member, agent in rows
        if agent.status not in (AgentStatus.SCORED, AgentStatus.LIVE)
    ]
    if len(rows) != COHORT_SIZE or invalid:
        reason = (
            "frozen cohort is incomplete"
            if len(rows) != COHORT_SIZE
            else "ineligible frozen members: " + ",".join(invalid)
        )
        if rollout.status != "blocked_ineligible" or rollout.blocked_reason != reason:
            rollout.status = "blocked_ineligible"
            rollout.blocked_reason = reason
            await _audit(
                session, rollout, "cohort_blocked", {"reason": reason}, now=now
            )
        return False
    if rollout.status == "blocked_ineligible":
        rollout.status = "collecting"
        rollout.blocked_reason = None
        await _audit(session, rollout, "cohort_unblocked", {}, now=now)
    return True


def heartbeat_supports_version(
    heartbeat: ValidatorHeartbeat,
    *,
    now: datetime,
    version: int = CANARY_BENCH_VERSION,
) -> bool:
    """Accept ``version`` only from a fresh scorer report matching its identity."""
    # Protocol 8 is the floor at which a heartbeat carries a SIGNED capability
    # and stack-identity payload at all -- it is a wire-format floor, not a
    # per-benchmark one, so it stays fixed as the benchmark version moves. Any
    # validator that can advertise a post-v2 benchmark is already >= 8.
    if heartbeat.protocol_version < 8:
        return False
    seen_at = (
        heartbeat.seen_at.replace(tzinfo=UTC)
        if heartbeat.seen_at.tzinfo is None
        else heartbeat.seen_at
    )
    if now - seen_at > timedelta(minutes=5):
        return False
    try:
        capabilities = ValidatorCapabilities.model_validate_json(
            json.dumps(heartbeat.capabilities)
        )
        stack = ValidatorStackIdentity.model_validate_json(json.dumps(heartbeat.stack))
    except ValidationError:
        return False
    scorer = capabilities.scorer_benchmarks
    if (
        scorer is None
        or scorer.status != "fresh_verified"
        or version not in scorer.supported_bench_versions
    ):
        return False
    if (
        scorer.observed_at is None
        or abs(int(now.timestamp()) - scorer.observed_at) > 300
    ):
        return False
    component = stack.components.dittobench_api
    return (
        capabilities.screened_images
        and component.source_revision == scorer.source_revision
        and (component.version is None or component.version == scorer.software_version)
    )


async def issue_rollout_ticket(
    session: AsyncSession,
    *,
    validator_hotkey: str,
    now: datetime,
    ttl: timedelta,
    artifact_mode: Literal["legacy", "prefer_screened", "screened_only"] = "legacy",
    validator_running_benchmark: bool = False,
) -> ValidatorTicket | None:
    """Issue one cohort lease, balanced one score per agent per coverage round."""
    # Retained as a keyword-compatible parameter for mixed platform callers;
    # the version contract, not an operator-wide routing flag, governs this.
    _ = artifact_mode
    rollout = await open_rollout(session, for_update=True)
    if rollout is None:
        return None
    heartbeat = await session.get(ValidatorHeartbeat, validator_hotkey)
    if heartbeat is None or not heartbeat_supports_version(
        heartbeat, now=now, version=rollout.desired_version
    ):
        return None
    contract = benchmark_contract(rollout.desired_version)
    complete_screened_image = (
        Agent.screened_image_sha256.is_not(None)
        & Agent.screened_image_size_bytes.is_not(None)
        & Agent.screened_image_id.is_not(None)
        & Agent.screened_image_ref.is_not(None)
        & Agent.screened_image_upload_id.is_not(None)
        & Agent.screened_image_verified_at.is_not(None)
    )
    existing_statement = (
        select(ValidatorTicket)
        .join(
            BenchmarkRolloutMember,
            BenchmarkRolloutMember.agent_id == ValidatorTicket.agent_id,
        )
        .join(Agent, Agent.agent_id == ValidatorTicket.agent_id)
        .where(
            BenchmarkRolloutMember.rollout_id == rollout.rollout_id,
            ValidatorTicket.validator_hotkey == validator_hotkey,
            ValidatorTicket.bench_version == rollout.desired_version,
            ValidatorTicket.status == TicketStatus.ISSUED,
            ValidatorTicket.deadline > now,
        )
        .limit(1)
        .with_for_update()
    )
    existing_statement = existing_statement.where(
        Agent.status.in_((AgentStatus.SCORED, AgentStatus.LIVE)),
        Agent.screening_policy_version >= contract.minimum_screening_policy_version,
        complete_screened_image,
    )
    existing = await session.scalar(existing_statement)
    if existing is not None:
        return existing
    score_count = (
        select(func.count(Score.validator_hotkey))
        .where(
            Score.agent_id == BenchmarkRolloutMember.agent_id,
            Score.bench_version == rollout.desired_version,
        )
        .correlate(BenchmarkRolloutMember)
        .scalar_subquery()
    )
    occupied_count = (
        select(func.count(ValidatorTicket.validator_hotkey))
        .where(
            ValidatorTicket.agent_id == BenchmarkRolloutMember.agent_id,
            ValidatorTicket.bench_version == rollout.desired_version,
            ValidatorTicket.status == TicketStatus.ISSUED,
            ValidatorTicket.deadline > now,
        )
        .correlate(BenchmarkRolloutMember)
        .scalar_subquery()
    )
    already_scored = (
        select(Score.agent_id)
        .where(
            Score.agent_id == BenchmarkRolloutMember.agent_id,
            Score.bench_version == rollout.desired_version,
            Score.validator_hotkey == validator_hotkey,
        )
        .exists()
    )
    member_statement = (
        select(BenchmarkRolloutMember)
        .join(Agent, Agent.agent_id == BenchmarkRolloutMember.agent_id)
        .where(
            BenchmarkRolloutMember.rollout_id == rollout.rollout_id,
            Agent.status.in_((AgentStatus.SCORED, AgentStatus.LIVE)),
            Agent.screening_policy_version >= contract.minimum_screening_policy_version,
            complete_screened_image,
            ~already_scored,
            score_count + occupied_count < SCORING_QUORUM,
        )
    )
    member = await session.scalar(
        member_statement.order_by(
            (score_count + occupied_count).asc(), BenchmarkRolloutMember.position
        )
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    if member is None:
        return None
    # A rollout can open while this validator still owns an ordinary source-
    # version lease. Preserve genuinely running work, but an idle/polling
    # validator must not keep resuming that lower-priority lease ahead of the
    # target-version cohort. The database allows only one issued ticket per
    # validator across all benchmark versions, so release any non-resumable
    # lease only after proving that eligible rollout work exists.
    competing_ticket = await session.scalar(
        select(ValidatorTicket)
        .where(
            ValidatorTicket.validator_hotkey == validator_hotkey,
            ValidatorTicket.status == TicketStatus.ISSUED,
        )
        .limit(1)
        .with_for_update()
    )
    if competing_ticket is not None:
        if validator_running_benchmark:
            return None
        competing_ticket.status = TicketStatus.EXPIRED
        competing_ticket.deadline = now
        competing_ticket.retry_after = now
        await session.flush()
    ticket = await session.get(
        ValidatorTicket,
        (member.agent_id, rollout.desired_version, validator_hotkey),
    )
    if ticket is None:
        ticket = ValidatorTicket(
            agent_id=member.agent_id,
            bench_version=rollout.desired_version,
            validator_hotkey=validator_hotkey,
            status=TicketStatus.ISSUED,
            issued_at=now,
            deadline=now + ttl,
            attempt_count=1,
            manual_retry_grants=0,
        )
        session.add(ticket)
    else:
        ticket.status = TicketStatus.ISSUED
        ticket.issued_at = now
        ticket.deadline = now + ttl
        ticket.attempt_count += 1
        ticket.retry_after = None
    await session.flush()
    return ticket


async def maybe_activate_rollout(
    session: AsyncSession, rollout: BenchmarkRollout, *, now: datetime
) -> bool:
    """Activate after the rolling qualification set converges at desired quorum."""
    # A superseded (or already activated) rollout is terminal and must never be
    # revived by a refresh sweep that still holds a stale reference to it.
    if rollout.status not in ("collecting", "blocked_ineligible"):
        return False
    count_rows = (
        await session.execute(
            select(BenchmarkRolloutMember.agent_id, func.count(Score.validator_hotkey))
            .outerjoin(
                Score,
                (Score.agent_id == BenchmarkRolloutMember.agent_id)
                & (Score.bench_version == rollout.desired_version),
            )
            .where(BenchmarkRolloutMember.rollout_id == rollout.rollout_id)
            .group_by(BenchmarkRolloutMember.agent_id)
        )
    ).all()
    counts: dict[UUID, int] = {agent_id: int(count) for agent_id, count in count_rows}
    # Fewer than quorum is "not ready yet"; MORE than quorum is still ready. An
    # equality test deadlocks the rollout permanently the first time any member
    # picks up a 4th score (a retry grant, an admin recovery, or a lost race),
    # with no operator escape hatch.
    top_five = await rolling_top_five(session)
    if len(top_five) != COHORT_SIZE:
        return False
    member_rows = (
        await session.execute(
            select(BenchmarkRolloutMember, Agent)
            .join(Agent, Agent.agent_id == BenchmarkRolloutMember.agent_id)
            .where(BenchmarkRolloutMember.rollout_id == rollout.rollout_id)
        )
    ).all()
    member_ids = {member.agent_id for member, _agent in member_rows}
    if not {member.agent_id for member in top_five}.issubset(member_ids):
        return False
    eligible_ids = {
        member.agent_id
        for member, agent in member_rows
        if agent.status in (AgentStatus.SCORED, AgentStatus.LIVE)
    }
    if any(counts.get(agent_id, 0) < SCORING_QUORUM for agent_id in eligible_ids):
        return False
    # Activation is the LAST point at which the full-emission-set guarantee can
    # be enforced. Before activation, list_eligible_ledger's own threshold holds
    # the ledger on the active version until MIN_DESIRED_AUTHORITY_AGENTS agents
    # hold a ranked desired-version quorum. After activation open_rollout()
    # returns None, so desired_version is None, the ledger reads the desired
    # version unconditionally, and that threshold is bypassed entirely — an agent
    # without desired-version scores simply drops out.
    #
    # The checks above do not imply the threshold. ``eligible_ids`` skips cohort
    # members that went banned / ath_pending_review, so it can be satisfied by
    # fewer than five agents; and ``counts`` is a raw count(scores) that a
    # smoke-profile 3/3 satisfies without ever being rankable. Either path would
    # activate into a pool too small to fill the champion + tail. Re-check with
    # the ledger's own definition.
    from ditto.db.queries.scores import count_ranked_quorum_agents

    ranked_quorum_agents = await count_ranked_quorum_agents(
        session, bench_version=rollout.desired_version
    )
    if ranked_quorum_agents < MIN_DESIRED_AUTHORITY_AGENTS:
        return False
    rollout.status = "activated"
    rollout.activated_at = now
    rollout.blocked_reason = None
    await _audit(
        session,
        rollout,
        "activated",
        {
            "bench_version": rollout.desired_version,
            "score_counts": {str(k): v for k, v in counts.items()},
            "ranked_quorum_agents": ranked_quorum_agents,
        },
        now=now,
    )
    await session.flush()
    return True


async def rollout_state(
    session: AsyncSession,
    *,
    now: datetime | None = None,
    capability_version: int | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    rollout = await session.scalar(
        select(BenchmarkRollout).order_by(BenchmarkRollout.created_at.desc()).limit(1)
    )
    active_version = (
        DEFAULT_BENCH_VERSION
        if rollout is None
        else rollout.desired_version
        if rollout.status == "activated"
        else rollout.from_version
    )
    version = capability_version or (
        rollout.desired_version if rollout is not None else active_version
    )
    heartbeats = (await session.execute(select(ValidatorHeartbeat))).scalars().all()
    capable_count = sum(
        heartbeat_supports_version(heartbeat, now=now, version=version)
        for heartbeat in heartbeats
    )
    if rollout is None:
        return {
            "active_version": active_version,
            "desired_version": active_version,
            "status": "inactive",
            "capability_bench_version": version,
            "canary_capable_validator_count": capable_count,
            # DEPRECATED alias of canary_capable_validator_count. It counts
            # validators capable of capability_bench_version, which is no longer
            # always 3. Kept because it is public API; read the new key.
            "v3_capable_validator_count": capable_count,
            "current_hybrid_top_five": [],
            "qualification_converged": False,
            "members": [],
        }
    count_rows = (
        await session.execute(
            select(BenchmarkRolloutMember.agent_id, func.count(Score.validator_hotkey))
            .outerjoin(
                Score,
                (Score.agent_id == BenchmarkRolloutMember.agent_id)
                & (Score.bench_version == rollout.desired_version),
            )
            .where(BenchmarkRolloutMember.rollout_id == rollout.rollout_id)
            .group_by(BenchmarkRolloutMember.agent_id)
        )
    ).all()
    counts: dict[UUID, int] = {agent_id: int(count) for agent_id, count in count_rows}
    members = (
        (
            await session.execute(
                select(BenchmarkRolloutMember)
                .where(BenchmarkRolloutMember.rollout_id == rollout.rollout_id)
                .order_by(BenchmarkRolloutMember.position)
            )
        )
        .scalars()
        .all()
    )
    current_top = await rolling_top_five(session)
    current_top_ids = {member.agent_id for member in current_top}
    qualified_ids = {member.agent_id for member in members}
    return {
        "active_version": rollout.desired_version
        if rollout.status == "activated"
        else rollout.from_version,
        "desired_version": rollout.desired_version,
        "status": rollout.status,
        "blocked_reason": rollout.blocked_reason,
        "capability_bench_version": version,
        "canary_capable_validator_count": capable_count,
        # DEPRECATED alias of canary_capable_validator_count; see above.
        "v3_capable_validator_count": capable_count,
        "current_hybrid_top_five": [str(member.agent_id) for member in current_top],
        "qualification_converged": len(current_top) == COHORT_SIZE
        and current_top_ids.issubset(qualified_ids),
        "members": [
            {
                "agent_id": str(member.agent_id),
                "position": member.position,
                "score_count": int(counts.get(member.agent_id, 0)),
                "currently_top_five": member.agent_id in current_top_ids,
            }
            for member in members
        ],
    }
