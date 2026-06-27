"""Mutations + reads against the ``scores`` table.

A score is upserted per ``(agent_id, validator_hotkey)``: a validator
re-scoring an agent overwrites its prior row rather than appending. The
upsert is a read-then-write inside the caller's transaction (portable
across the Postgres runtime and the SQLite unit-test fallback) rather than
a dialect-specific ``ON CONFLICT``; at MVP single-validator concurrency the
PK still guarantees one row per ``(agent, validator)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy.exc import IntegrityError as SAIntegrityError

from ditto.db.errors import IntegrityError as DbIntegrityError
from ditto.db.models import Score

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession


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
    from sqlalchemy import select

    stmt = select(Score).where(Score.agent_id == agent_id)
    result = await session.execute(stmt)
    return list(result.scalars().all())
