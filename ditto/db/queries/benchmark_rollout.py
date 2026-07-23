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
    InferenceProviderRoute,
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
PRIORITY_COHORT_SIZE = 5
# New benchmark transitions rescore only the inherited top ten. The database
# still permits older 11-25-member rollout snapshots so historical audit rows
# remain readable and an in-flight rollout created by an older deployment can
# finish without destructive member deletion.
RESCORE_COHORT_SIZE = 10
MAX_PERSISTED_RESCORE_COHORT_SIZE = 25
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

    from ditto.db.queries.payments import get_miner_coldkeys_for_agents

    coldkeys = await get_miner_coldkeys_for_agents(
        session, agent_ids={agent.agent_id for agent, _ in candidates}
    )
    # Keep one best generation per payment-time coldkey before taking the
    # network-wide top five. Legacy rows without payment provenance remain
    # isolated by hotkey.
    candidates.sort(key=lambda item: (-item[1], item[0].created_at, item[0].agent_id))
    unique: list[RolloutSnapshotMember] = []
    seen_owners: set[str] = set()
    for agent, composite in candidates:
        owner = (
            f"coldkey:{coldkeys[agent.agent_id]}"
            if agent.agent_id in coldkeys
            else f"hotkey:{agent.miner_hotkey}"
        )
        if owner in seen_owners:
            continue
        seen_owners.add(owner)
        unique.append(
            RolloutSnapshotMember(agent.agent_id, agent.miner_hotkey, composite)
        )
        if len(unique) == PRIORITY_COHORT_SIZE:
            break
    return unique


async def historical_rescore_cohort(
    session: AsyncSession,
    *,
    source_version: int,
    limit: int = RESCORE_COHORT_SIZE,
) -> list[RolloutSnapshotMember]:
    """Freeze the prior-era rescore cohort without admitting the whole ledger.

    The immediately previous benchmark owns the cohort. If it has fewer than
    ``limit`` finalized distinct miners, the next older scored benchmark fills
    the remaining positions. No third historical era is consulted: this is the
    explicit "combine two previous benchmark iterations" fallback, not an
    unbounded backfill of every legacy submission.
    """
    if limit < PRIORITY_COHORT_SIZE or limit > RESCORE_COHORT_SIZE:
        raise ValueError(
            f"rollout cohort limit must be between {PRIORITY_COHORT_SIZE} "
            f"and {RESCORE_COHORT_SIZE}"
        )
    versions = list(
        await session.scalars(
            select(Score.bench_version)
            .where(Score.bench_version <= source_version)
            .distinct()
            .order_by(Score.bench_version.desc())
            .limit(2)
        )
    )
    if not versions:
        return []
    agents = list(
        await session.scalars(
            select(Agent).where(
                Agent.status.in_((AgentStatus.SCORED, AgentStatus.LIVE))
            )
        )
    )
    if not agents:
        return []
    scores = list(
        await session.scalars(
            select(Score).where(
                Score.agent_id.in_([agent.agent_id for agent in agents]),
                Score.bench_version.in_(versions),
            )
        )
    )
    by_version_agent: dict[int, dict[UUID, list[Score]]] = {}
    for score in scores:
        by_version_agent.setdefault(score.bench_version, {}).setdefault(
            score.agent_id, []
        ).append(score)

    from ditto.db.queries.payments import get_miner_coldkeys_for_agents

    coldkeys = await get_miner_coldkeys_for_agents(
        session, agent_ids={agent.agent_id for agent in agents}
    )
    agent_by_id = {agent.agent_id: agent for agent in agents}
    selected: list[RolloutSnapshotMember] = []
    seen_agents: set[UUID] = set()
    seen_owners: set[str] = set()
    for version in versions:
        ranked: list[tuple[Agent, float]] = []
        for agent_id, version_scores in by_version_agent.get(version, {}).items():
            # A partial score set is provisional, not a finalized historical
            # standing, and must not consume one of the bounded rescore slots.
            if len(version_scores) < SCORING_QUORUM:
                continue
            middle = sorted(
                version_scores,
                key=lambda row: (row.composite, row.validator_hotkey),
            )[(len(version_scores) - 1) // 2]
            if middle.n < 100 or middle.composite <= 0:
                continue
            ranked.append((agent_by_id[agent_id], float(middle.composite)))
        ranked.sort(key=lambda item: (-item[1], item[0].created_at, item[0].agent_id))
        for agent, composite in ranked:
            owner = (
                f"coldkey:{coldkeys[agent.agent_id]}"
                if agent.agent_id in coldkeys
                else f"hotkey:{agent.miner_hotkey}"
            )
            if agent.agent_id in seen_agents or owner in seen_owners:
                continue
            seen_agents.add(agent.agent_id)
            seen_owners.add(owner)
            selected.append(
                RolloutSnapshotMember(agent.agent_id, agent.miner_hotkey, composite)
            )
            if len(selected) == limit:
                return selected
    return selected


async def append_rollout_member(
    session: AsyncSession,
    *,
    rollout: BenchmarkRollout,
    member: RolloutSnapshotMember,
    dataset: DatasetPin,
    now: datetime,
    audit_context: dict[str, Any] | None = None,
) -> bool:
    """Permanently qualify one member of the frozen historical cohort."""
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
    open_transition = await open_rollout(session)
    if open_transition is not None:
        from ditto.db.queries.scores import count_ranked_quorum_agents

        priority_ids = set(
            await session.scalars(
                select(BenchmarkRolloutMember.agent_id).where(
                    BenchmarkRolloutMember.rollout_id == open_transition.rollout_id,
                    BenchmarkRolloutMember.position <= PRIORITY_COHORT_SIZE,
                )
            )
        )
        ready = await count_ranked_quorum_agents(
            session,
            bench_version=open_transition.desired_version,
            agent_ids=priority_ids,
        )
        if (
            len(priority_ids) == PRIORITY_COHORT_SIZE
            and ready >= MIN_DESIRED_AUTHORITY_AGENTS
        ):
            return open_transition.desired_version
    return await persisted_active_bench_version(session)


async def persisted_active_bench_version(session: AsyncSession) -> int:
    """Return the latest durable benchmark-authority decision.

    Normal rollout activation records authority on the rollout row. Recovery from
    an already-superseded, fully qualified rollout records an append-only
    ``authority_selected`` audit event instead of rewriting terminal history.
    Comparing both timestamps keeps the newest durable authority decision
    authoritative without adding a second mutable state table.
    """
    activated = (
        await session.execute(
            select(BenchmarkRollout.desired_version, BenchmarkRollout.activated_at)
            .where(
                BenchmarkRollout.status == "activated",
                BenchmarkRollout.activated_at.is_not(None),
            )
            .order_by(BenchmarkRollout.activated_at.desc())
            .limit(1)
        )
    ).first()
    selected = (
        await session.execute(
            select(BenchmarkRollout.desired_version, BenchmarkRolloutAudit.recorded_at)
            .join(
                BenchmarkRolloutAudit,
                BenchmarkRolloutAudit.rollout_id == BenchmarkRollout.rollout_id,
            )
            .where(BenchmarkRolloutAudit.event == "authority_selected")
            .order_by(
                BenchmarkRolloutAudit.recorded_at.desc(),
                BenchmarkRolloutAudit.audit_id.desc(),
            )
            .limit(1)
        )
    ).first()
    if selected is not None and (
        activated is None or selected.recorded_at >= activated.activated_at
    ):
        return int(selected.desired_version)
    if activated is not None:
        return int(activated.desired_version)
    return DEFAULT_BENCH_VERSION


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
    if await active_bench_version(session) == rollout.desired_version:
        raise RolloutConflictError(
            "a benchmark rollout that already owns active authority cannot be "
            "superseded; select another qualified active contract first"
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


async def authority_selection_state(
    session: AsyncSession, *, bench_version: int
) -> dict[str, Any]:
    """Describe whether a historical contract can safely own weight authority."""
    rollout = await rollout_for_desired_version(session, desired_version=bench_version)
    if rollout is None:
        return {
            "version": bench_version,
            "ready": False,
            "ranked_quorum_agents": 0,
            "min_ranked_quorum_agents": MIN_DESIRED_AUTHORITY_AGENTS,
            "blocked_reason": "no rollout history exists for this contract",
        }
    priority_ids = set(
        await session.scalars(
            select(BenchmarkRolloutMember.agent_id).where(
                BenchmarkRolloutMember.rollout_id == rollout.rollout_id,
                BenchmarkRolloutMember.position <= PRIORITY_COHORT_SIZE,
            )
        )
    )
    if len(priority_ids) != PRIORITY_COHORT_SIZE:
        return {
            "version": bench_version,
            "ready": False,
            "ranked_quorum_agents": 0,
            "min_ranked_quorum_agents": MIN_DESIRED_AUTHORITY_AGENTS,
            "blocked_reason": "the rollout does not contain a complete priority cohort",
        }
    eligible_ids = set(
        await session.scalars(
            select(Agent.agent_id).where(
                Agent.agent_id.in_(priority_ids),
                Agent.status.in_((AgentStatus.SCORED, AgentStatus.LIVE)),
            )
        )
    )
    if eligible_ids != priority_ids:
        return {
            "version": bench_version,
            "ready": False,
            "ranked_quorum_agents": 0,
            "min_ranked_quorum_agents": MIN_DESIRED_AUTHORITY_AGENTS,
            "blocked_reason": "one or more priority agents are no longer eligible",
        }
    from ditto.db.queries.scores import count_ranked_quorum_agents

    ranked = await count_ranked_quorum_agents(
        session, bench_version=bench_version, agent_ids=priority_ids
    )
    ready = ranked >= MIN_DESIRED_AUTHORITY_AGENTS
    return {
        "version": bench_version,
        "ready": ready,
        "ranked_quorum_agents": ranked,
        "min_ranked_quorum_agents": MIN_DESIRED_AUTHORITY_AGENTS,
        "blocked_reason": None
        if ready
        else "the priority cohort does not yet have five ranked quorums",
    }


async def select_active_bench_version(
    session: AsyncSession,
    *,
    bench_version: int,
    actor: str,
    reason: str,
    now: datetime,
) -> BenchmarkRollout:
    """Select a fully qualified historical contract as active authority.

    This is a recovery/control-plane action, not an arbitrary version setter.
    It is forward-only, requires the rollout target to be terminal, and refuses
    to race an open rollout. The append-only audit event becomes the durable
    authority decision while the superseded rollout row remains immutable.
    """
    rows = list(
        (
            await session.execute(
                select(BenchmarkRollout)
                .order_by(BenchmarkRollout.created_at)
                .with_for_update()
            )
        ).scalars()
    )
    if any(row.status in ("collecting", "blocked_ineligible") for row in rows):
        raise RolloutConflictError(
            "supersede the open benchmark rollout before changing active authority"
        )
    current = await persisted_active_bench_version(session)
    if bench_version <= current:
        raise RolloutConflictError(
            f"active benchmark selection is forward-only: current v{current}, "
            f"requested v{bench_version}"
        )
    rollout = next(
        (row for row in reversed(rows) if row.desired_version == bench_version),
        None,
    )
    if rollout is None or rollout.status != "superseded":
        raise RolloutConflictError(
            "only a fully qualified superseded rollout can be selected for recovery"
        )
    readiness = await authority_selection_state(session, bench_version=bench_version)
    if not readiness["ready"]:
        raise RolloutConflictError(str(readiness["blocked_reason"]))
    await _audit(
        session,
        rollout,
        "authority_selected",
        {
            "actor": actor,
            "reason": reason,
            "previous_active_version": current,
            "bench_version": bench_version,
            "ranked_quorum_agents": readiness["ranked_quorum_agents"],
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
    """Freeze a bounded prior-era cohort and target dataset pins, idempotently."""
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
    if not PRIORITY_COHORT_SIZE <= len(members) <= RESCORE_COHORT_SIZE:
        raise ValueError("a benchmark rollout requires between five and ten members")
    if len({m.agent_id for m in members}) != len(members):
        raise ValueError("benchmark rollout agents must be distinct")
    if len({m.miner_hotkey for m in members}) != len(members):
        raise ValueError("benchmark rollout miners must be distinct")
    if set(datasets) != {m.agent_id for m in members}:
        raise ValueError("every frozen rollout member requires one target dataset pin")

    rollout = BenchmarkRollout(
        rollout_id=uuid4(),
        from_version=from_version,
        desired_version=desired_version,
        status="collecting",
        cohort_size=len(members),
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
    if len(rows) != rollout.cohort_size or invalid:
        reason = (
            "frozen cohort is incomplete"
            if len(rows) != rollout.cohort_size
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
    if version >= 7 and heartbeat.protocol_version < 11:
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
    if version >= 7 and (
        not capabilities.ticket_inference or scorer.v7_calibration is None
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
    slot_id: str = "slot-0",
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
            ValidatorTicket.slot_id == slot_id,
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
    already_ticketed = (
        select(ValidatorTicket.agent_id)
        .where(
            ValidatorTicket.agent_id == BenchmarkRolloutMember.agent_id,
            ValidatorTicket.bench_version == rollout.desired_version,
            ValidatorTicket.validator_hotkey == validator_hotkey,
            ValidatorTicket.status.in_((TicketStatus.ISSUED, TicketStatus.SCORED)),
        )
        .exists()
    )
    priority_count_rows = (
        await session.execute(
            select(
                BenchmarkRolloutMember.agent_id,
                func.count(Score.validator_hotkey),
            )
            .outerjoin(
                Score,
                (Score.agent_id == BenchmarkRolloutMember.agent_id)
                & (Score.bench_version == rollout.desired_version),
            )
            .where(
                BenchmarkRolloutMember.rollout_id == rollout.rollout_id,
                BenchmarkRolloutMember.position <= PRIORITY_COHORT_SIZE,
            )
            .group_by(BenchmarkRolloutMember.agent_id)
        )
    ).all()
    priority_counts: dict[UUID, int] = {
        agent_id: int(count) for agent_id, count in priority_count_rows
    }
    priority_complete = len(priority_counts) == PRIORITY_COHORT_SIZE and all(
        count >= SCORING_QUORUM for count in priority_counts.values()
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
            ~already_ticketed,
            score_count + occupied_count < SCORING_QUORUM,
        )
    )
    if not priority_complete:
        # This is deliberately fleet-wide, not validator-local. A validator
        # that has already scored every incomplete priority member idles until
        # another validator closes those quorums; it must not skip to rank 6.
        member_statement = member_statement.where(
            BenchmarkRolloutMember.position <= PRIORITY_COHORT_SIZE
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
    # validator slot across all benchmark versions, so release any non-resumable
    # lease only after proving that eligible rollout work exists.
    competing_ticket = await session.scalar(
        select(ValidatorTicket)
        .where(
            ValidatorTicket.validator_hotkey == validator_hotkey,
            ValidatorTicket.slot_id == slot_id,
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
            slot_id=slot_id,
            status=TicketStatus.ISSUED,
            issued_at=now,
            deadline=now + ttl,
            attempt_count=1,
            manual_retry_grants=0,
        )
        session.add(ticket)
    else:
        ticket.status = TicketStatus.ISSUED
        ticket.slot_id = slot_id
        ticket.issued_at = now
        ticket.deadline = now + ttl
        ticket.attempt_count += 1
        ticket.retry_after = None
    await session.flush()
    return ticket


async def maybe_activate_rollout(
    session: AsyncSession, rollout: BenchmarkRollout, *, now: datetime
) -> bool:
    """Activate after every frozen cohort member reaches desired quorum."""
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
    member_rows = (
        await session.execute(
            select(BenchmarkRolloutMember, Agent)
            .join(Agent, Agent.agent_id == BenchmarkRolloutMember.agent_id)
            .where(BenchmarkRolloutMember.rollout_id == rollout.rollout_id)
        )
    ).all()
    if len(member_rows) != rollout.cohort_size:
        return False
    if any(
        agent.status not in (AgentStatus.SCORED, AgentStatus.LIVE)
        for _member, agent in member_rows
    ):
        return False
    member_ids = {member.agent_id for member, _agent in member_rows}
    if any(counts.get(agent_id, 0) < SCORING_QUORUM for agent_id in member_ids):
        return False
    # Activation is the LAST point at which the full-emission-set guarantee can
    # be enforced. Before activation, list_eligible_ledger's own threshold holds
    # the ledger on the active version until MIN_DESIRED_AUTHORITY_AGENTS agents
    # hold a ranked desired-version quorum. After activation open_rollout()
    # returns None, so desired_version is None, the ledger reads the desired
    # version unconditionally, and that threshold is bypassed entirely — an agent
    # without desired-version scores simply drops out.
    #
    # The raw counts above do not imply rankability: a smoke-profile 3/3 can
    # satisfy them without ever being eligible for weights. Require every
    # frozen cohort member to hold a ranked quorum before closing the rollout.
    from ditto.db.queries.scores import count_ranked_quorum_agents

    ranked_cohort_agents = await count_ranked_quorum_agents(
        session,
        bench_version=rollout.desired_version,
        agent_ids=member_ids,
    )
    if ranked_cohort_agents != len(member_ids):
        return False
    if rollout.desired_version >= 7:
        routes = list(
            await session.scalars(
                select(InferenceProviderRoute).where(
                    InferenceProviderRoute.model == "openai/gpt-oss-20b",
                    InferenceProviderRoute.status.in_(("discovered", "healthy")),
                    InferenceProviderRoute.calibration_status == "eligible",
                    InferenceProviderRoute.calibration_manifest_sha256.is_not(None),
                )
            )
        )
        if not routes:
            return False
        route_identities = {
            (
                route.provider,
                route.profile_revision,
                route.calibration_manifest_sha256,
            )
            for route in routes
        }
        heartbeats = list(await session.scalars(select(ValidatorHeartbeat)))
        inference_ready = False
        for heartbeat in heartbeats:
            if heartbeat.protocol_version < 11 or not heartbeat_supports_version(
                heartbeat, now=now, version=rollout.desired_version
            ):
                continue
            try:
                capabilities = ValidatorCapabilities.model_validate_json(
                    json.dumps(heartbeat.capabilities)
                )
            except ValidationError:
                continue
            scorer = capabilities.scorer_benchmarks
            calibration = scorer.v7_calibration if scorer is not None else None
            if calibration is None or not capabilities.ticket_inference:
                continue
            if any(
                route.model == "openai/gpt-oss-20b"
                and (
                    route.provider,
                    route.profile_revision,
                    calibration.manifest_sha256,
                )
                in route_identities
                for route in calibration.supported_routes
            ):
                inference_ready = True
                break
        if not inference_ready:
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
            "ranked_quorum_agents": ranked_cohort_agents,
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
    # Single source of truth for the active version: whatever the weight-setting
    # guard (`active_bench_version`) resolves. This endpoint's `active_version` is
    # what operators read and echo back as `expected_active_version` when starting
    # a rollout, so deriving it from the same authority the start guard checks means
    # the two can never disagree and spuriously 409 ("active benchmark changed").
    # In the normal open-rollout case this is identical to the row-derived value
    # (the flip predicates are equivalent when MIN_DESIRED_AUTHORITY_AGENTS ==
    # PRIORITY_COHORT_SIZE); it only reconciles the terminal/edge cases where the
    # most-recent row and the latest activated row differ.
    active_version = await active_bench_version(session)
    version = capability_version or (
        rollout.desired_version if rollout is not None else active_version
    )
    heartbeats = (await session.execute(select(ValidatorHeartbeat))).scalars().all()
    capable_count = sum(
        heartbeat_supports_version(heartbeat, now=now, version=version)
        for heartbeat in heartbeats
    )
    # The authority switch is gated on this count, not on the cohort's raw score
    # counts: the whole ledger flips to the desired version only once at least
    # MIN_DESIRED_AUTHORITY_AGENTS agents hold a complete RANKED quorum there, so
    # the emission set (champion + tail) is never short. Exposing it is the only
    # way a reader can answer "when do weights switch?" without re-deriving it.
    from ditto.db.queries.scores import count_ranked_quorum_agents

    if rollout is None:
        return {
            "active_version": active_version,
            "desired_version": active_version,
            "status": "inactive",
            "capability_bench_version": version,
            "ranked_quorum_agents": await count_ranked_quorum_agents(
                session, bench_version=DEFAULT_BENCH_VERSION
            ),
            "min_ranked_quorum_agents": MIN_DESIRED_AUTHORITY_AGENTS,
            "canary_capable_validator_count": capable_count,
            # DEPRECATED alias of canary_capable_validator_count. It counts
            # validators capable of capability_bench_version, which is no longer
            # always 3. Kept because it is public API; read the new key.
            "v3_capable_validator_count": capable_count,
            "current_hybrid_top_five": [],
            "qualification_converged": False,
            "cohort_size": 0,
            "cohort_ready_count": 0,
            "priority_cohort_size": PRIORITY_COHORT_SIZE,
            "priority_complete": False,
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
    priority_member_ids = {
        member.agent_id for member in members if member.position <= PRIORITY_COHORT_SIZE
    }
    ranked_quorum_agents = await count_ranked_quorum_agents(
        session,
        bench_version=rollout.desired_version,
        agent_ids=priority_member_ids,
    )
    cohort_ready_count = sum(
        counts.get(member.agent_id, 0) >= SCORING_QUORUM for member in members
    )
    priority_members = [
        member for member in members if member.position <= PRIORITY_COHORT_SIZE
    ]
    priority_complete = len(priority_members) == PRIORITY_COHORT_SIZE and all(
        counts.get(member.agent_id, 0) >= SCORING_QUORUM for member in priority_members
    )
    return {
        "active_version": active_version,
        "desired_version": rollout.desired_version,
        "status": rollout.status,
        "blocked_reason": rollout.blocked_reason,
        "capability_bench_version": version,
        "ranked_quorum_agents": ranked_quorum_agents,
        "min_ranked_quorum_agents": MIN_DESIRED_AUTHORITY_AGENTS,
        "canary_capable_validator_count": capable_count,
        # DEPRECATED alias of canary_capable_validator_count; see above.
        "v3_capable_validator_count": capable_count,
        "current_hybrid_top_five": [str(member.agent_id) for member in current_top],
        "qualification_converged": len(current_top) == PRIORITY_COHORT_SIZE
        and current_top_ids.issubset(qualified_ids),
        "cohort_size": rollout.cohort_size,
        "cohort_ready_count": cohort_ready_count,
        "priority_cohort_size": PRIORITY_COHORT_SIZE,
        "priority_complete": priority_complete,
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
