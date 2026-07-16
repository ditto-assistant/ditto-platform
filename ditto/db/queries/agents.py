"""Queries against the ``agents`` table.

Writes (``insert_agent``) and reads (``get_latest_agent_by_hotkey``,
``get_agent_by_id``) sit together because the table is small and the
two surfaces share their dispatch on the ORM model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError as SAIntegrityError

from ditto.api_models.agent_status import AgentStatus
from ditto.db.errors import IntegrityError as DbIntegrityError
from ditto.db.models import Agent, Score, ScreeningAttempt

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession


async def insert_agent(
    session: AsyncSession,
    *,
    agent_id: UUID,
    miner_hotkey: str,
    name: str,
    sha256: str,
    size_bytes: int,
    content_fingerprint: dict | None = None,
    normalized_source_hash: str | None = None,
    prompt_fingerprint: dict | None = None,
    code_embedding: list | None = None,
    code_embed_model: str | None = None,
) -> int:
    """Insert one ``agents`` row inside the caller-owned transaction.

    Status is omitted so the schema default ``'uploaded'`` applies; the
    screener PR moves it forward through the state machine. The caller
    runs this together with :func:`insert_evaluation_payment` inside one
    ``async with session.begin():`` block so both rows commit atomically
    (a PK violation on the payment insert rolls the agent insert back).

    ``size_bytes`` is the actual streamed tarball size; it feeds the
    anti-copy near-dup signal (a lightly-tweaked copy has a near-identical
    size + score) and is surfaced in the validator ledger.
    ``content_fingerprint`` (:mod:`ditto.api_server.fingerprint`) is the
    shingle MinHash sketch feeding the gate's content-level signal; ``None``
    when the tarball was unreadable/empty at upload.
    ``normalized_source_hash`` (same module) is the exact-repack hash of the
    canonicalized source feeding the gate's equality signal; ``None`` on the same
    unreadable/empty condition. ``prompt_fingerprint`` (same module) is the
    prompt-surface sketch, stored in shadow mode; ``None`` when the crate carries no
    prompt-length literal or the tarball is unreadable. ``code_embedding`` /
    ``code_embed_model`` are the code-embedding vector and its ``model@revision`` tag
    (see
    :mod:`ditto.api_server.embedding`), stored in shadow mode; ``None`` when the
    embedder is disabled or the embed failed.

    Raises:
        DbIntegrityError: Any constraint violation on ``agents``
            (UNIQUE ``(agent_id, miner_hotkey)``, NOT NULL violations,
            invalid enum value, etc.). No agents-level constraint is a
            miner-facing action, so the envelope catch-all maps every
            case to HTTP 500.

    Returns:
        The immutable version assigned within this hotkey-and-name series.
    """
    # Serialize one logical agent series on Postgres so concurrent uploads with
    # the same hotkey + name cannot claim the same revision. SQLite tests run
    # single-process; the UNIQUE constraint remains the final invariant on both.
    if session.get_bind().dialect.name == "postgresql":
        series_key = f"{miner_hotkey}:{len(name)}:{name}"
        await session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:series_key, 0))"),
            {"series_key": series_key},
        )
    version = int(
        (
            await session.scalar(
                select(func.coalesce(func.max(Agent.version), 0) + 1).where(
                    Agent.miner_hotkey == miner_hotkey,
                    Agent.name == name,
                )
            )
        )
        or 1
    )

    row = Agent(
        agent_id=agent_id,
        miner_hotkey=miner_hotkey,
        name=name,
        version=version,
        sha256=sha256,
        size_bytes=size_bytes,
        content_fingerprint=content_fingerprint,
        normalized_source_hash=normalized_source_hash,
        prompt_fingerprint=prompt_fingerprint,
        code_embedding=code_embedding,
        code_embed_model=code_embed_model,
    )
    session.add(row)
    try:
        await session.flush()
    except SAIntegrityError as e:
        raise DbIntegrityError(f"agents insert violated constraint: {e.orig}") from e
    return version


async def resolve_review(
    session: AsyncSession,
    *,
    agent_id: UUID,
    decision: AgentStatus,
) -> Agent | None:
    """Resolve an ``ath_pending_review`` hold, returning the updated agent.

    The anti-copy gate parks a suspicious high-scorer in
    ``ath_pending_review`` (see :mod:`ditto.api_server.scoring_gate`); this
    is the human-review exit that un-holds it. ``decision`` is either
    :attr:`AgentStatus.SCORED` (cleared — the agent re-enters the ledger and
    the validator fold can crown it) or :attr:`AgentStatus.BANNED` (rejected
    — a confirmed copy). Clearing wipes the ``duplicate_of`` / ``review_reason``
    moderation record; banning preserves it as the audit trail.

    Winner-take-all makes a false-positive hold catastrophic (a legitimate
    miner earns nothing while held), so this exit is deliberately manual —
    there is intentionally no auto-timeout release, which a real copy could
    simply wait out.

    Returns ``None`` if no agent has that id. Raises ``ValueError`` if the
    agent is not currently held, or ``decision`` is not one of the two
    allowed targets. Runs inside the caller-owned transaction.
    """
    if decision not in (AgentStatus.SCORED, AgentStatus.BANNED):
        raise ValueError(
            f"resolve_review decision must be scored or banned, got {decision}"
        )
    agent = await session.get(Agent, agent_id)
    if agent is None:
        return None
    if agent.status != AgentStatus.ATH_PENDING_REVIEW:
        raise ValueError(f"agent {agent_id} is {agent.status}, not ath_pending_review")
    agent.status = decision
    if decision == AgentStatus.SCORED:
        agent.duplicate_of = None
        agent.review_reason = None
    await session.flush()
    return agent


async def get_latest_agent_by_hotkey(
    session: AsyncSession,
    *,
    miner_hotkey: str,
) -> Agent | None:
    """Return the most recent ``agents`` row for the given hotkey, or ``None``.

    Orders by ``created_at DESC`` and takes one. Status is unfiltered;
    callers see banned or failed rows if they are the most recent. The
    ``/retrieval/agent-by-hotkey`` endpoint additionally consults
    :func:`ditto.db.queries.bans.is_hotkey_banned` to surface a hotkey-level
    ban (distinct from a per-agent ``banned`` status).
    """
    stmt = (
        select(Agent)
        .where(Agent.miner_hotkey == miner_hotkey)
        .order_by(Agent.created_at.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_agent_by_id(
    session: AsyncSession,
    *,
    agent_id: UUID,
    for_update: bool = False,
) -> Agent | None:
    """Return the ``agents`` row for the given id, or ``None``.

    ``for_update=True`` takes a row lock (``SELECT ... FOR UPDATE``) so a
    read-then-conditional-write transition (screener promotion, score finalize)
    serializes against a concurrent writer instead of last-writer-wins. The lock
    is a no-op on the SQLite unit-test fallback and a real row lock on Postgres.
    """
    return await session.get(
        Agent, agent_id, with_for_update=True if for_update else None
    )


async def list_agents_by_status(
    session: AsyncSession,
    *,
    statuses: Sequence[AgentStatus],
    limit: int,
) -> list[Agent]:
    """Return agents whose status is in ``statuses``, oldest first.

    Backs the validator work queue. The default caller passes
    ``[AgentStatus.EVALUATING]``, which is served by the partial index
    ``agents_status_evaluating_idx`` (``WHERE status = 'evaluating'``).
    Ordering by ``created_at`` ascending drains the queue in arrival order.
    """
    stmt = (
        select(Agent)
        .where(Agent.status.in_(statuses))
        .order_by(Agent.created_at.asc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


@dataclass(frozen=True)
class PublicActivityRow:
    """One public activity entry plus its safe aggregate score progress."""

    agent: Agent
    score_count: int
    provisional_composite: float | None
    highest_composite: float | None
    last_scored_at: datetime | None
    screening_attempt: ScreeningAttempt | None


async def list_public_activity(
    session: AsyncSession,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> tuple[list[PublicActivityRow], int]:
    """Return recent submissions and the total row count.

    ``limit=None`` lets callers that must filter on the derived public status
    inspect the complete dataset before applying their own pagination.
    """
    total = int(await session.scalar(select(func.count(Agent.agent_id))) or 0)
    score_counts = (
        select(
            Score.agent_id,
            func.count(Score.validator_hotkey).label("score_count"),
            func.avg(Score.composite).label("provisional_composite"),
            func.max(Score.composite).label("highest_composite"),
            func.max(Score.updated_at).label("last_scored_at"),
        )
        .group_by(Score.agent_id)
        .subquery()
    )
    stmt = (
        select(
            Agent,
            func.coalesce(score_counts.c.score_count, 0),
            score_counts.c.provisional_composite,
            score_counts.c.highest_composite,
            score_counts.c.last_scored_at,
            ScreeningAttempt,
        )
        .outerjoin(score_counts, score_counts.c.agent_id == Agent.agent_id)
        .outerjoin(
            ScreeningAttempt,
            (ScreeningAttempt.agent_id == Agent.agent_id)
            & (ScreeningAttempt.status == "running"),
        )
        .order_by(Agent.created_at.desc(), Agent.agent_id.desc())
    )
    if offset:
        stmt = stmt.offset(offset)
    if limit is not None:
        stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return (
        [
            PublicActivityRow(
                agent=agent,
                score_count=int(score_count),
                provisional_composite=(
                    float(provisional_composite)
                    if provisional_composite is not None
                    else None
                ),
                highest_composite=(
                    float(highest_composite) if highest_composite is not None else None
                ),
                last_scored_at=last_scored_at,
                screening_attempt=screening_attempt,
            )
            for (
                agent,
                score_count,
                provisional_composite,
                highest_composite,
                last_scored_at,
                screening_attempt,
            ) in result.all()
        ],
        total,
    )
