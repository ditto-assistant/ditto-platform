"""Mutations + reads against the ``scores`` table.

A score is upserted per ``(agent_id, validator_hotkey)``: a validator
re-scoring an agent overwrites its prior row rather than appending. The
upsert is a read-then-write inside the caller's transaction (portable
across the Postgres runtime and the SQLite unit-test fallback) rather than
a dialect-specific ``ON CONFLICT``; at MVP single-validator concurrency the
PK still guarantees one row per ``(agent, validator)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError as SAIntegrityError

from ditto.api_models.agent_status import AgentStatus
from ditto.db.errors import IntegrityError as DbIntegrityError
from ditto.db.models import Agent, Score

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class LedgerRow:
    """One entry of the best-eligible-score-per-miner ledger.

    The immutable value object :func:`list_eligible_ledger` returns and the
    ``GET /scoring/scores`` endpoint maps onto the ``LedgerEntry`` wire model.
    ``first_seen`` is the agent's upload time — the KOTH tie-break that lets the
    original beat a later copy of the same score.
    """

    miner_hotkey: str
    agent_id: UUID
    composite: float
    first_seen: datetime
    sha256: str
    size_bytes: int | None
    seed: int
    validator_hotkey: str
    signature: str | None
    status: AgentStatus


async def upsert_score(
    session: AsyncSession,
    *,
    agent_id: UUID,
    validator_hotkey: str,
    run_id: str,
    seed: int,
    composite: float,
    tool_mean: float,
    memory_mean: float,
    median_ms: int,
    n: int,
    generated_at: datetime,
    signature: str | None = None,
    details: dict | None = None,
) -> None:
    """Insert or update the score for ``(agent_id, validator_hotkey)``.

    Runs inside the caller-owned transaction (``async with
    session.begin():``) so the score write and the agent status transition
    commit atomically. Re-reporting the same ``run_id`` is idempotent; a new
    ``run_id`` overwrites the validator's prior score for this agent.

    Raises:
        DbIntegrityError: Any constraint violation on ``scores`` (the FK to
            ``agents`` when ``agent_id`` is unknown, or a CHECK on a value
            outside its declared range). These indicate a caller bug — the
            handler validates ranges + agent existence first — so the
            envelope catch-all maps them to HTTP 500.
    """
    existing = await session.get(Score, (agent_id, validator_hotkey))
    if existing is None:
        session.add(
            Score(
                agent_id=agent_id,
                validator_hotkey=validator_hotkey,
                run_id=run_id,
                seed=seed,
                composite=composite,
                tool_mean=tool_mean,
                memory_mean=memory_mean,
                median_ms=median_ms,
                n=n,
                generated_at=generated_at,
                signature=signature,
                details=details,
            )
        )
    else:
        existing.run_id = run_id
        existing.seed = seed
        existing.composite = composite
        existing.tool_mean = tool_mean
        existing.memory_mean = memory_mean
        existing.median_ms = median_ms
        existing.n = n
        existing.generated_at = generated_at
        existing.signature = signature
        existing.details = details
    try:
        await session.flush()
    except SAIntegrityError as e:
        raise DbIntegrityError(f"scores upsert violated constraint: {e.orig}") from e


async def list_scores_for_agent(
    session: AsyncSession,
    *,
    agent_id: UUID,
) -> list[Score]:
    """Return every validator's score for ``agent_id`` (unordered).

    Used by weight computation / leaderboard reads that aggregate across
    the validator set. Returns an empty list when no validator has scored
    the agent yet.
    """
    stmt = select(Score).where(Score.agent_id == agent_id)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_eligible_ledger(session: AsyncSession) -> list[LedgerRow]:
    """Return the best eligible score per miner, highest composite first.

    The persistent ledger the validator folds into KOTH+ATH weights (via
    ``GET /scoring/scores``). "Eligible" = agents in ``scored`` — this excludes
    ``ath_pending_review`` holds (suspected copies) and ``banned`` agents, and
    (because scoring flips ``evaluating -> scored``) is served by the partial
    index ``agents_status_scored_idx``.

    Two levels: an inner per-agent aggregate collapses each agent's score
    row(s), then a ``ROW_NUMBER`` window keeps only each miner's single best
    agent. Ordering (``composite DESC, first_seen ASC, agent_id ASC``) matches
    the validator fold's champion/tail tie-breaks so the exposed order and the
    computed weights agree by construction.

    v1 is single-validator, so exactly one ``scores`` row backs each eligible
    agent and the inner ``MAX`` aggregates are a no-op. When the D3 k=3 design
    lands, ``MAX(composite)`` becomes ``median`` and ``seed`` / ``validator_hotkey``
    / ``signature`` move to a representative-of-the-median selection — a
    localized change to this one query.
    """
    per_agent = (
        select(
            Agent.agent_id.label("agent_id"),
            Agent.miner_hotkey.label("miner_hotkey"),
            Agent.sha256.label("sha256"),
            Agent.size_bytes.label("size_bytes"),
            Agent.created_at.label("first_seen"),
            Agent.status.label("status"),
            func.max(Score.composite).label("composite"),
            func.max(Score.seed).label("seed"),
            func.max(Score.validator_hotkey).label("validator_hotkey"),
            func.max(Score.signature).label("signature"),
        )
        .join(Score, Score.agent_id == Agent.agent_id)
        .where(Agent.status == AgentStatus.SCORED)
        .group_by(
            Agent.agent_id,
            Agent.miner_hotkey,
            Agent.sha256,
            Agent.size_bytes,
            Agent.created_at,
            Agent.status,
        )
        .subquery()
    )
    rn = (
        func.row_number()
        .over(
            partition_by=per_agent.c.miner_hotkey,
            order_by=(
                per_agent.c.composite.desc(),
                per_agent.c.first_seen.asc(),
                per_agent.c.agent_id.asc(),
            ),
        )
        .label("rn")
    )
    ranked = select(per_agent, rn).subquery()
    stmt = (
        select(ranked)
        .where(ranked.c.rn == 1)
        .order_by(
            ranked.c.composite.desc(),
            ranked.c.first_seen.asc(),
            ranked.c.agent_id.asc(),
        )
    )
    result = await session.execute(stmt)
    return [
        LedgerRow(
            miner_hotkey=row.miner_hotkey,
            agent_id=row.agent_id,
            composite=row.composite,
            first_seen=row.first_seen,
            sha256=row.sha256,
            size_bytes=row.size_bytes,
            seed=row.seed,
            validator_hotkey=row.validator_hotkey,
            signature=row.signature,
            status=AgentStatus(row.status),
        )
        for row in result
    ]
