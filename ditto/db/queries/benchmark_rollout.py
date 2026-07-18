"""Durable, version-separated benchmark activation state machine."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from pydantic import ValidationError
from sqlalchemy import func, select

from ditto.api_models.agent_status import AgentStatus
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
    existing = await open_rollout(session, for_update=True)
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
    return component.source_revision == scorer.source_revision and (
        component.version is None or component.version == scorer.software_version
    )


async def issue_rollout_ticket(
    session: AsyncSession,
    *,
    validator_hotkey: str,
    now: datetime,
    ttl: timedelta,
) -> ValidatorTicket | None:
    """Issue one v3 cohort lease, balanced one score per agent per coverage round."""
    heartbeat = await session.get(ValidatorHeartbeat, validator_hotkey)
    if heartbeat is None or not heartbeat_supports_v3(heartbeat, now=now):
        return None
    rollout = await open_rollout(session, for_update=True)
    if rollout is None or not await _validate_frozen_members(session, rollout, now=now):
        return None
    existing = await session.scalar(
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
    member = await session.scalar(
        select(BenchmarkRolloutMember)
        .where(
            BenchmarkRolloutMember.rollout_id == rollout.rollout_id,
            ~already_scored,
            score_count + occupied_count < SCORING_QUORUM,
        )
        .order_by((score_count + occupied_count).asc(), BenchmarkRolloutMember.position)
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
    """Atomically activate only when every frozen member has exactly three v3 scores."""
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
    if len(counts) != COHORT_SIZE or any(
        count != SCORING_QUORUM for count in counts.values()
    ):
        return False
    if not await _validate_frozen_members(session, rollout, now=now):
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
    return {
        "active_version": rollout.desired_version
        if rollout.status == "activated"
        else rollout.from_version,
        "desired_version": rollout.desired_version,
        "status": rollout.status,
        "blocked_reason": rollout.blocked_reason,
        "v3_capable_validator_count": capable_count,
        "members": [
            {
                "agent_id": str(member.agent_id),
                "position": member.position,
                "score_count": int(counts.get(member.agent_id, 0)),
            }
            for member in members
        ],
    }
