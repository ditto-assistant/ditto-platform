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
from datetime import UTC, datetime, timedelta
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
    tool_mean: float
    memory_mean: float
    first_seen: datetime
    sha256: str
    size_bytes: int | None
    run_id: str
    seed: int
    validator_hotkey: str
    signature: str | None
    status: AgentStatus
    content_fingerprint: dict | None = None
    """Shingle MinHash sketch of the tarball source (see
    :mod:`ditto.api_server.fingerprint`); the gate's content-level anti-copy
    signal. ``None`` for rows uploaded before fingerprinting or with an
    unreadable tarball. Defaulted so it need not be threaded through the
    validator ledger *wire* model — it is moderation-only, never exposed."""
    structural_fingerprint: dict | None = None
    """AST-level structural sketch of the crate (computed by dittobench, written at
    score time); the gate's rename-resistant anti-copy signal. Same ``{v,k,card,m}``
    shape as ``content_fingerprint``. ``None`` before this landed / no parseable
    Rust. Moderation-only, never exposed on the wire."""
    normalized_source_hash: str | None = None
    """L3a exact-repack hash of the canonicalized source (see
    :func:`ditto.api_server.fingerprint.compute_normalized_source_hash`); the gate's
    equality anti-copy signal, held unconditionally on a match like exact
    ``sha256``. ``None`` before this landed or for an unreadable tarball.
    Moderation-only, never exposed on the wire."""
    median_ms: int = 0
    """Median per-case latency (ms) of the winning run — public benchmark telemetry."""
    n: int = 0
    """Number of cases scored in the winning run — public benchmark telemetry."""
    details: dict | None = None
    """The winning run's opaque telemetry blob (``scores.details``): models used,
    bench_version, dataset_sha256, per-category means, token spend, and the
    per-case breakdown. The public leaderboard exposes a **safe subset** (never
    ``per_case``, which carries the answer key); validator-gated endpoints may
    read it whole. ``None`` for rows scored before details were persisted."""


@dataclass(frozen=True)
class HealthRollup:
    """Aggregate subnet-health counters for the public dashboard.

    Everything here is derived from what the platform *actually* records —
    submissions and the scores validators report back. Run started/failed counts
    and set-weights latency are validator-side telemetry (wandb), not stored
    here, so this rollup deliberately omits a "success rate": the platform only
    ever sees a *successful* score, so it cannot honestly report failures.
    """

    miners: int
    """Distinct miners who have ever submitted an agent."""
    scored_miners: int
    """Distinct miners with a ``scored`` agent that has a score row (==
    leaderboard size — a stray scored-status row with no score is excluded)."""
    scored_agents: int
    """``scored`` agents that actually carry a score row (eligible submissions)."""
    last_scored_at: datetime | None
    """Newest score ``generated_at`` — when a validator last scored anything."""
    scores_24h: int
    """Scores generated in the last 24h — scoring throughput."""
    avg_latency_ms: int | None
    """Mean of the per-score median case latency (ms), across all scores."""


_HEALTH_WINDOW = timedelta(hours=24)


def _as_utc(dt: datetime) -> datetime:
    """Coerce a possibly-naive DB timestamp to aware UTC.

    ``TIMESTAMP(timezone=True)`` round-trips as aware on Postgres but can come
    back naive on the SQLite unit-test fallback; treat naive as UTC so the 24h
    window comparison is dialect-independent.
    """
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


async def get_public_health(session: AsyncSession, *, now: datetime) -> HealthRollup:
    """Compute the public subnet-health rollup as of ``now`` (aware UTC).

    Two cheap reads: a conditional-aggregate over ``agents`` for the miner/agent
    counts, and a scan of ``(generated_at, median_ms)`` over ``scores`` reduced
    in Python. The Python reduction (rather than a SQL time filter + window)
    keeps the 24h cutoff and the naive/aware timestamp handling identical across
    Postgres and the SQLite test fallback; the ``scores`` table is small at
    subnet scale (one row per agent per validator).
    """
    total_miners = (
        await session.execute(select(func.count(func.distinct(Agent.miner_hotkey))))
    ).scalar_one()

    # Scored counts require an actual ``scores`` row (INNER JOIN), so they can
    # never contradict the leaderboard: a stray ``scored``-status agent with no
    # score row is not "scored" for public purposes. Distinct-count both because
    # the join fans out one row per validator that scored the agent.
    scored = (
        await session.execute(
            select(
                func.count(func.distinct(Agent.miner_hotkey)),
                func.count(func.distinct(Agent.agent_id)),
            )
            .select_from(Agent)
            .join(Score, Score.agent_id == Agent.agent_id)
            .where(Agent.status == AgentStatus.SCORED)
        )
    ).one()

    rows = (await session.execute(select(Score.generated_at, Score.median_ms))).all()
    cutoff = now - _HEALTH_WINDOW
    generated = [_as_utc(r[0]) for r in rows]
    return HealthRollup(
        miners=int(total_miners),
        scored_miners=int(scored[0]),
        scored_agents=int(scored[1]),
        last_scored_at=max(generated) if generated else None,
        scores_24h=sum(1 for g in generated if g >= cutoff),
        avg_latency_ms=(round(sum(r[1] for r in rows) / len(rows)) if rows else None),
    )


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


async def list_miner_composite_history(
    session: AsyncSession,
    hotkeys: list[str],
    *,
    limit_per: int = 12,
) -> dict[str, list[float]]:
    """Per-miner composite trajectory (oldest→newest) for the trend sparkline.

    Returns ``{miner_hotkey: [composite, ...]}`` — every score row for the
    miner's agents, chronological, capped to the most recent ``limit_per``. This
    is the miner's score *over time* (across submissions + re-scores), which is
    aggregate-only (a composite series — no seeds, no per-case content). Empty
    dict for no hotkeys; a miner with a single score simply gets a length-1 list.

    One join + one pass; the ``scores`` table is small at subnet scale.
    """
    if not hotkeys:
        return {}
    stmt = (
        select(Agent.miner_hotkey, Score.composite, Score.generated_at)
        .select_from(Score)
        .join(Agent, Agent.agent_id == Score.agent_id)
        .where(Agent.miner_hotkey.in_(hotkeys))
        .order_by(Agent.miner_hotkey, Score.generated_at.asc())
    )
    rows = (await session.execute(stmt)).all()
    out: dict[str, list[float]] = {}
    for hotkey, composite, _generated_at in rows:
        out.setdefault(hotkey, []).append(float(composite))
    return {hotkey: series[-limit_per:] for hotkey, series in out.items()}


async def list_eligible_ledger(session: AsyncSession) -> list[LedgerRow]:
    """Return the best eligible score per miner, highest composite first.

    The persistent ledger the validator folds into KOTH+ATH weights (via
    ``GET /scoring/scores``). "Eligible" = agents in ``scored`` — this excludes
    ``ath_pending_review`` holds (suspected copies) and ``banned`` agents, and
    (because scoring flips ``evaluating -> scored``) is served by the partial
    index ``agents_status_scored_idx``.

    Three levels, all deterministic:

    1. ``agent_best`` ranks each agent's ``scores`` rows and keeps the single
       best **whole row** — so ``composite`` / ``seed`` / ``run_id`` /
       ``validator_hotkey`` / ``signature`` always come from the *same* physical
       row and the exposed signature verifies against the exposed composite.
       (Picking each column independently with ``MAX`` would stitch a mismatched
       tuple the moment an agent has >1 score row — e.g. after a validator
       hotkey rotation inserts a second ``(agent_id, validator_hotkey)`` row.)
    2. join to ``agents`` filtered to ``scored``.
    3. a ``ROW_NUMBER`` window keeps each miner's single best agent.

    Ordering (``composite DESC, first_seen ASC, agent_id ASC``) matches the
    validator fold's champion/tail tie-breaks. When the D3 k=3 design lands,
    ``agent_best`` becomes a median-of-3 selection — a localized change here.
    """
    agent_best = select(
        Score.agent_id.label("agent_id"),
        Score.composite.label("composite"),
        Score.tool_mean.label("tool_mean"),
        Score.memory_mean.label("memory_mean"),
        Score.seed.label("seed"),
        Score.run_id.label("run_id"),
        Score.median_ms.label("median_ms"),
        Score.n.label("n"),
        Score.details.label("details"),
        Score.validator_hotkey.label("validator_hotkey"),
        Score.signature.label("signature"),
        func.row_number()
        .over(
            partition_by=Score.agent_id,
            order_by=(Score.composite.desc(), Score.validator_hotkey.asc()),
        )
        .label("srn"),
    ).subquery()
    per_agent = (
        select(
            Agent.agent_id.label("agent_id"),
            Agent.miner_hotkey.label("miner_hotkey"),
            Agent.sha256.label("sha256"),
            Agent.size_bytes.label("size_bytes"),
            Agent.content_fingerprint.label("content_fingerprint"),
            Agent.structural_fingerprint.label("structural_fingerprint"),
            Agent.normalized_source_hash.label("normalized_source_hash"),
            Agent.created_at.label("first_seen"),
            Agent.status.label("status"),
            agent_best.c.composite,
            agent_best.c.tool_mean,
            agent_best.c.memory_mean,
            agent_best.c.seed,
            agent_best.c.run_id,
            agent_best.c.median_ms,
            agent_best.c.n,
            agent_best.c.details,
            agent_best.c.validator_hotkey,
            agent_best.c.signature,
        )
        .join(agent_best, agent_best.c.agent_id == Agent.agent_id)
        .where(Agent.status == AgentStatus.SCORED, agent_best.c.srn == 1)
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
            tool_mean=row.tool_mean,
            memory_mean=row.memory_mean,
            first_seen=row.first_seen,
            sha256=row.sha256,
            size_bytes=row.size_bytes,
            run_id=row.run_id,
            seed=row.seed,
            validator_hotkey=row.validator_hotkey,
            signature=row.signature,
            status=AgentStatus(row.status),
            content_fingerprint=row.content_fingerprint,
            structural_fingerprint=row.structural_fingerprint,
            normalized_source_hash=row.normalized_source_hash,
            median_ms=row.median_ms,
            n=row.n,
            details=row.details,
        )
        for row in result
    ]
