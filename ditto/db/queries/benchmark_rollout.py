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
from ditto.api_models.benchmark_contract import benchmark_contract
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


DEFAULT_BENCH_VERSION = 2
CANARY_BENCH_VERSION = 3
COHORT_SIZE = 5
SCORING_QUORUM = 3
# Whether a complete desired-version quorum takes over an agent's ledger
# authority mid-rollout (per-agent rolling authority). Set to False to pin
# authority to the active version while a desired version stabilizes — every
# agent with an active-version median then keeps it (leaderboard, validator
# weight fold, KOTH) until the rollout activates, while desired-version medians
# stay visible as per-row rollout progress. Pinned 2026-07-19 → unpinned once
# the v3 scoring fixes landed.
DESIRED_AUTHORITY_AT_QUORUM = True


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
    """Return the hybrid top five used while v3 is collecting.

    An agent remains ranked by its v2 median until it has a complete v3 quorum.
    At quorum its v3 median atomically replaces v2 for this ranking. Recomputing
    this after every accepted score/verdict is what lets ranks 6-10 rise and
    become newly qualified without making one- or two-validator samples churn
    the cohort.
    """
    agents = list(
        await session.scalars(select(Agent).where(Agent.status == AgentStatus.SCORED))
    )
    if not agents:
        return []
    scores = list(
        await session.scalars(
            select(Score).where(
                Score.agent_id.in_([agent.agent_id for agent in agents]),
                Score.bench_version.in_((DEFAULT_BENCH_VERSION, CANARY_BENCH_VERSION)),
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
        selected = versions.get(CANARY_BENCH_VERSION, [])
        if len(selected) < SCORING_QUORUM:
            selected = versions.get(DEFAULT_BENCH_VERSION, [])
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
    await _audit(
        session,
        locked,
        "member_qualified",
        {
            "agent_id": str(member.agent_id),
            "position": position,
            "hybrid_composite": member.composite,
        },
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
    desired_version: int,
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


async def create_rollout_snapshot(
    session: AsyncSession,
    *,
    members: Sequence[RolloutSnapshotMember],
    datasets: dict[UUID, DatasetPin],
    now: datetime,
) -> BenchmarkRollout:
    """Freeze exactly five ranked agents and their v3 dataset pins, idempotently."""
    if session.get_bind().dialect.name == "postgresql":
        await session.execute(
            select(
                func.pg_advisory_xact_lock(
                    func.hashtextextended("benchmark-v3-rollout", 0)
                )
            )
        )
    existing = await rollout_for_transition(
        session,
        from_version=DEFAULT_BENCH_VERSION,
        desired_version=CANARY_BENCH_VERSION,
        for_update=True,
    )
    if existing is not None:
        return existing
    if len(members) != COHORT_SIZE:
        raise ValueError("benchmark v3 rollout requires exactly five members")
    if len({m.agent_id for m in members}) != COHORT_SIZE:
        raise ValueError("benchmark rollout agents must be distinct")
    if len({m.miner_hotkey for m in members}) != COHORT_SIZE:
        raise ValueError("benchmark rollout miners must be distinct")
    if set(datasets) != {m.agent_id for m in members}:
        raise ValueError("every frozen rollout member requires one v3 dataset pin")

    rollout = BenchmarkRollout(
        rollout_id=uuid4(),
        from_version=DEFAULT_BENCH_VERSION,
        desired_version=CANARY_BENCH_VERSION,
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
        session.add(
            BenchmarkDataset(
                agent_id=member.agent_id,
                bench_version=CANARY_BENCH_VERSION,
                seed=pin.seed,
                sha256=pin.sha256,
                run_size=pin.run_size,
                seed_block=pin.seed_block,
                seed_block_hash=pin.seed_block_hash,
                created_at=now,
            )
        )
    await _audit(
        session,
        rollout,
        "cohort_frozen",
        {"agent_ids": [str(member.agent_id) for member in members]},
        now=now,
    )
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


def heartbeat_supports_v3(heartbeat: ValidatorHeartbeat, *, now: datetime) -> bool:
    """Accept v3 only from a fresh scorer report matching the signed stack identity."""
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
        or 3 not in scorer.supported_bench_versions
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
    """Issue one v3 cohort lease, balanced one score per agent per coverage round."""
    # Retained as a keyword-compatible parameter for mixed platform callers;
    # the version contract, not an operator-wide routing flag, governs v3.
    _ = artifact_mode
    heartbeat = await session.get(ValidatorHeartbeat, validator_hotkey)
    if heartbeat is None or not heartbeat_supports_v3(heartbeat, now=now):
        return None
    rollout = await open_rollout(session, for_update=True)
    if rollout is None:
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
    incompatible_existing = await session.scalar(
        select(ValidatorTicket)
        .join(
            BenchmarkRolloutMember,
            BenchmarkRolloutMember.agent_id == ValidatorTicket.agent_id,
        )
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
    if incompatible_existing is not None:
        if validator_running_benchmark:
            return None
        incompatible_existing.status = TicketStatus.EXPIRED
        incompatible_existing.deadline = now
        incompatible_existing.retry_after = now
        await session.flush()

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
    """Activate after the rolling qualification set converges at v3 quorum."""
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
        },
        now=now,
    )
    await session.flush()
    return True


async def rollout_state(
    session: AsyncSession, *, now: datetime | None = None
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    heartbeats = (await session.execute(select(ValidatorHeartbeat))).scalars().all()
    capable_count = sum(
        heartbeat_supports_v3(heartbeat, now=now) for heartbeat in heartbeats
    )
    rollout = await session.scalar(
        select(BenchmarkRollout).order_by(BenchmarkRollout.created_at.desc()).limit(1)
    )
    if rollout is None:
        return {
            "active_version": DEFAULT_BENCH_VERSION,
            "desired_version": DEFAULT_BENCH_VERSION,
            "status": "inactive",
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
