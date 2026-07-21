"""Mutations + reads against the ``scores`` table.

A score is upserted per ``(agent_id, validator_hotkey)``: a validator
re-scoring an agent overwrites its prior row rather than appending. The
upsert is a read-then-write inside the caller's transaction (portable
across the Postgres runtime and the SQLite unit-test fallback) rather than
a dialect-specific ``ON CONFLICT``. Under the k=3 quorum several validators
score the same agent, so the ``(agent_id, validator_hotkey)`` PK is what holds
each validator to one row; finalization takes the median across the rows
(:func:`list_eligible_ledger`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from statistics import median
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import ColumnElement, and_, case, func, literal, null, or_, select
from sqlalchemy.exc import IntegrityError as SAIntegrityError

from ditto.api_models.agent_status import AgentStatus
from ditto.db.errors import IntegrityError as DbIntegrityError
from ditto.db.models import (
    Agent,
    BenchmarkRolloutMember,
    EvaluationPayment,
    Score,
)
from ditto.db.queries.agents import get_agent_by_id

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession


# A run must administer the *full* benchmark to be ranked on the leaderboard and
# to earn emissions. The dittobench-api run-size profiles are small = 6 tool + 6
# memory = 12 cases, medium ~= 42, full = 60 tool + 50 memory + 4 isolation ~=
# 114 (dittobench-api internal/gen/gen.go Profiles). A smaller profile omits the
# hard anti-overfit memory categories (injection-resistance, aggregation-count,
# assistant-recall) entirely — its 6-case memory suite is trivially aced (a
# "100% memory" is a small-sample artifact), so the composite is neither
# comparable across miners nor discriminative. This floor cleanly separates full
# (~114) from the smoke/practice profiles (small 12, medium ~42); runs below it
# are surfaced as "provisional" (eligible=False) but never ranked or folded into
# weights. Keep in sync with the validator's MIN_ELIGIBLE_CASES
# (ditto-subnet ditto/validator/weights.py).
MIN_ELIGIBLE_CASES = 100

# The validator pool size: a submission is scored by this many independent
# validators (the k=3 model), and its canonical score is the MEDIAN of their
# composites, so a single generous or harsh validator cannot move it. The
# platform issues at most this many tickets per agent (``validator_tickets``)
# and finalizes an agent (``evaluating -> scored``) once it has this many
# scores. Keep in sync with the ticket-issue cap and the validator quorum.
SCORING_QUORUM = 3


def _is_ranked() -> ColumnElement[bool]:
    """SQL predicate for a *ranked* run: it administered the full benchmark AND
    scored a positive composite. Both gates mirror the validator's weight fold:
    ``filter_eligible`` drops sub-floor runs and ``compute_weights`` then drops
    ``composite <= 0`` (a failed/zero run earns nothing). Without the second gate
    a full run that scored 0.000 is crowned #1 over a higher provisional run,
    which is exactly what the fold refuses to pay. Keep the two in lockstep.
    """
    return and_(Score.n >= MIN_ELIGIBLE_CASES, Score.composite > 0.0)


@dataclass(frozen=True)
class LedgerRow:
    """One entry of the best-eligible-score-per-payment-coldkey ledger.

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
    miner_coldkey: str | None = None
    """Payment-time owner identity used to enforce one emission position.

    This is moderation-only metadata and is not exposed on the public scoring
    wire model. ``None`` is retained for legacy rows that predate payments.
    """
    bench_version: int = 1
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
    """exact-repack hash of the canonicalized source (see
    :func:`ditto.api_server.fingerprint.compute_normalized_source_hash`); the gate's
    equality anti-copy signal, held unconditionally on a match like exact
    ``sha256``. ``None`` before this landed or for an unreadable tarball.
    Moderation-only, never exposed on the wire."""
    prompt_fingerprint: dict | None = None
    """Prompt-surface sketch (see
    :func:`ditto.api_server.fingerprint.compute_prompt_fingerprint`). Shadow signal:
    surfaced so the gate can note a corroborating prompt overlap in a hold's audit
    reason, but not a hold trigger on its own (honest agents share harness
    scaffolding prompts). ``None`` before this landed / no prompt-length literal.
    Moderation-only, never exposed on the wire."""
    code_embedding: list | None = None
    """Code-embedding vector (see :mod:`ditto.api_server.embedding`). Shadow
    signal: surfaced so the gate can cosine-compare it against a candidate (only
    when ``code_embed_model`` matches — a cross-model cosine is meaningless), but not
    yet a hold trigger. ``None`` before this landed / embedder disabled / embed
    failed. Moderation-only, never exposed on the wire."""
    code_embed_model: str | None = None
    """``model@revision`` provenance tag of :attr:`code_embedding`. Gates cosine
    comparisons to same-model vectors and drives re-embed sweeps on a model bump."""
    median_ms: int = 0
    """Median per-case latency (ms) of the winning run — public benchmark telemetry."""
    n: int = 0
    """Number of cases scored in the winning run — public benchmark telemetry."""
    eligible: bool = False
    """Whether this run is *ranked*: it administered the full benchmark
    (``n >= MIN_ELIGIBLE_CASES``) **and** scored a positive composite
    (:func:`_is_ranked`). ``False`` marks either a smoke/practice run (small/medium
    profile) or a full run that scored 0.000; both are surfaced for transparency
    but never ranked or folded into weights, matching the validator's two-gate
    fold — see :data:`MIN_ELIGIBLE_CASES`."""
    details: dict | None = None
    """The winning run's opaque telemetry blob (``scores.details``): models used,
    bench_version, dataset_sha256, per-category means, token spend, and the
    per-case breakdown. The public leaderboard exposes a **safe subset** (never
    ``per_case``, which carries the answer key); validator-gated endpoints may
    read it whole. ``None`` for rows scored before details were persisted."""


def _emission_owner_key() -> ColumnElement[str]:
    """Stable owner key for one-emission-position selection.

    New uploads always carry an immutable payment-time coldkey. The hotkey
    fallback preserves legacy/test rows without accidentally collapsing every
    missing-payment row into one owner.
    """
    return case(
        (
            EvaluationPayment.miner_coldkey.is_not(None),
            literal("coldkey:") + EvaluationPayment.miner_coldkey,
        ),
        else_=literal("hotkey:") + Agent.miner_hotkey,
    )


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
    """Newest platform score-write time — when a validator last scored anything."""
    total_scores: int
    """All validator score records currently stored by the platform."""
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
    counts, and a scan of ``(updated_at, median_ms)`` over ``scores`` reduced in
    Python. Public activity uses the platform-controlled write timestamp rather
    than the validator-supplied report ``generated_at`` provenance field. The
    Python reduction (rather than a SQL time filter + window) keeps the 24h
    cutoff and the naive/aware timestamp handling identical across Postgres and
    the SQLite test fallback; the ``scores`` table is small at subnet scale (one
    row per agent per validator).
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

    rows = (await session.execute(select(Score.updated_at, Score.median_ms))).all()
    cutoff = now - _HEALTH_WINDOW
    generated = [_as_utc(r[0]) for r in rows]
    return HealthRollup(
        miners=int(total_miners),
        scored_miners=int(scored[0]),
        scored_agents=int(scored[1]),
        last_scored_at=max(generated) if generated else None,
        total_scores=len(rows),
        scores_24h=sum(1 for g in generated if g >= cutoff),
        avg_latency_ms=(round(sum(r[1] for r in rows) / len(rows)) if rows else None),
    )


async def upsert_score(
    session: AsyncSession,
    *,
    agent_id: UUID,
    validator_hotkey: str,
    bench_version: int = 2,
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
    existing = await session.get(Score, (agent_id, bench_version, validator_hotkey))
    if existing is None:
        session.add(
            Score(
                agent_id=agent_id,
                validator_hotkey=validator_hotkey,
                bench_version=bench_version,
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
    bench_version: int | None = None,
) -> list[Score]:
    """Return every validator's score for ``agent_id`` (unordered).

    Used by weight computation / leaderboard reads that aggregate across
    the validator set. Returns an empty list when no validator has scored
    the agent yet.
    """
    if bench_version is None:
        from ditto.db.queries.benchmark_rollout import active_bench_version

        bench_version = await active_bench_version(session)
    stmt = select(Score).where(
        Score.agent_id == agent_id, Score.bench_version == bench_version
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_score_for_validator(
    session: AsyncSession,
    *,
    agent_id: UUID,
    validator_hotkey: str,
) -> Score | None:
    """One validator's recorded score row for an agent, or ``None``.

    Backs the transcript upload path: the declared ``transcript_sha256`` in
    this row's details is what an uploaded artifact's bytes must hash to.
    """
    stmt = select(Score).where(
        Score.agent_id == agent_id,
        Score.validator_hotkey == validator_hotkey,
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


# Agents whose scoring has settled into a public, non-provisional state. A held
# copy (``ath_pending_review``) is deliberately excluded — its score is not yet
# public and surfacing "under review" before resolution would be premature.
_PUBLIC_SUBMISSION_STATUSES = (AgentStatus.SCORED, AgentStatus.LIVE)


@dataclass(frozen=True)
class SubmissionRow:
    """One finalized submission plus every validator's score for it.

    The value object behind the public per-submission transparency surface: the
    agent's dataset pin (``dataset_seed`` / ``dataset_sha256`` / ``run_size``)
    and the full set of per-validator :class:`Score` rows the median finalized
    on. ``last_scored_at`` is the most recent score time (the recency key the
    index sorts by).
    """

    agent_id: UUID
    miner_hotkey: str
    status: AgentStatus
    dataset_seed: int | None
    dataset_sha256: str | None
    dataset_run_size: str | None
    dataset_seed_block: int | None
    dataset_seed_block_hash: str | None
    last_scored_at: datetime | None
    scores: list[Score]


async def get_submission_scores(
    session: AsyncSession, *, agent_id: UUID
) -> SubmissionRow | None:
    """The full k=3 scoring record for one finalized agent, or ``None``.

    ``None`` when the agent does not exist or has not settled into a public
    status (still evaluating, or held for copy review) — callers map that to 404
    so a provisional agent's partial scores are never exposed.
    """
    agent = await get_agent_by_id(session, agent_id=agent_id)
    if agent is None or agent.status not in _PUBLIC_SUBMISSION_STATUSES:
        return None
    # Every version's rows, not just the active bench version: the public record
    # covers older + current runs (each row is version-labeled downstream), and
    # the submissions index batch-fetch is unfiltered the same way.
    scores = list(
        (await session.execute(select(Score).where(Score.agent_id == agent_id)))
        .scalars()
        .all()
    )
    last_scored_at = _as_utc(max(s.generated_at for s in scores)) if scores else None
    return SubmissionRow(
        agent_id=agent.agent_id,
        miner_hotkey=agent.miner_hotkey,
        status=agent.status,
        dataset_seed=agent.dataset_seed,
        dataset_sha256=agent.dataset_sha256,
        dataset_run_size=agent.dataset_run_size,
        dataset_seed_block=agent.dataset_seed_block,
        dataset_seed_block_hash=agent.dataset_seed_block_hash,
        last_scored_at=last_scored_at,
        scores=sorted(scores, key=lambda s: s.validator_hotkey),
    )


async def list_public_submissions(
    session: AsyncSession, *, limit: int = 50
) -> list[SubmissionRow]:
    """Recent finalized submissions, most recently scored first.

    Two queries: pick the finalized agents by latest-score recency (capped to
    ``limit``), then batch-fetch their score rows and group in Python. The
    ``scores`` table is small at subnet scale, so this stays cheap.
    """
    recency = (
        select(
            Score.agent_id.label("agent_id"),
            func.max(Score.generated_at).label("last_scored"),
        )
        .group_by(Score.agent_id)
        .subquery()
    )
    stmt = (
        select(Agent, recency.c.last_scored)
        .join(recency, recency.c.agent_id == Agent.agent_id)
        .where(Agent.status.in_(_PUBLIC_SUBMISSION_STATUSES))
        .order_by(recency.c.last_scored.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    if not rows:
        return []
    agent_ids = [agent.agent_id for agent, _ in rows]
    score_rows = (
        (await session.execute(select(Score).where(Score.agent_id.in_(agent_ids))))
        .scalars()
        .all()
    )
    by_agent: dict[UUID, list[Score]] = {}
    for s in score_rows:
        by_agent.setdefault(s.agent_id, []).append(s)
    return [
        SubmissionRow(
            agent_id=agent.agent_id,
            miner_hotkey=agent.miner_hotkey,
            status=agent.status,
            dataset_seed=agent.dataset_seed,
            dataset_sha256=agent.dataset_sha256,
            dataset_run_size=agent.dataset_run_size,
            dataset_seed_block=agent.dataset_seed_block,
            dataset_seed_block_hash=agent.dataset_seed_block_hash,
            last_scored_at=_as_utc(last_scored) if last_scored else None,
            scores=sorted(
                by_agent.get(agent.agent_id, []), key=lambda s: s.validator_hotkey
            ),
        )
        for agent, last_scored in rows
    ]


async def list_scores_for_bench_version(
    session: AsyncSession, *, version: int, limit: int = 100, offset: int = 0
) -> tuple[list[tuple[Score, str]], int]:
    """Score rows scored under ``bench_version == version``, plus the total count.

    Returns ``([(score, miner_hotkey), ...], total)``, ordered deterministically
    (generated_at, agent_id, validator_hotkey) and paginated. Powers the retired-
    version corpus release, which serves the FULL unredacted per-case answer keys
    stored in ``scores.details`` — the caller must gate this to retired versions.
    Filters on the first-class version column bound to the ticket and signature;
    the JSON detail is advisory compatibility telemetry only.
    """
    base = (
        select(Score, Agent.miner_hotkey)
        .join(Agent, Agent.agent_id == Score.agent_id)
        .where(Score.bench_version == version)
    )
    total = (
        await session.execute(select(func.count()).select_from(base.subquery()))
    ).scalar_one()
    rows = (
        await session.execute(
            base.order_by(
                Score.generated_at.asc(),
                Score.agent_id.asc(),
                Score.validator_hotkey.asc(),
            )
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return [(score, miner) for score, miner in rows], int(total)


async def list_miner_composite_history(
    session: AsyncSession,
    hotkeys: list[str],
    *,
    limit_per: int = 12,
    bench_versions: dict[str, int] | None = None,
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
        select(
            Agent.miner_hotkey,
            Score.composite,
            Score.generated_at,
            Score.bench_version,
        )
        .select_from(Score)
        .join(Agent, Agent.agent_id == Score.agent_id)
        .where(Agent.miner_hotkey.in_(hotkeys))
        .order_by(Agent.miner_hotkey, Score.generated_at.asc())
    )
    rows = (await session.execute(stmt)).all()
    out: dict[str, list[float]] = {}
    for hotkey, composite, _generated_at, score_version in rows:
        if bench_versions is None or bench_versions.get(hotkey) == score_version:
            out.setdefault(hotkey, []).append(float(composite))
    return {hotkey: series[-limit_per:] for hotkey, series in out.items()}


async def count_ranked_quorum_agents(
    session: AsyncSession,
    *,
    bench_version: int,
    agent_ids: set[UUID] | None = None,
) -> int:
    """How many eligible agents hold a complete, RANKED quorum at ``bench_version``.

    One definition, two consumers — they must agree or the guarantee they
    implement has a hole:

    * :func:`list_eligible_ledger` gates the whole-pool authority switch on this
      count (:data:`~ditto.db.queries.benchmark_rollout.MIN_DESIRED_AUTHORITY_AGENTS`).
    * :func:`~ditto.db.queries.benchmark_rollout.maybe_activate_rollout` gates
      activation on it.

    "Eligible" and "ranked" mean exactly what they mean on the ledger: the agent
    is in ``scored`` (so a ``banned`` / ``ath_pending_review`` agent never
    counts), it has at least :data:`SCORING_QUORUM` score rows at this version,
    and its **median** row at this version passes :func:`_is_ranked` — the full
    benchmark (``n >= MIN_ELIGIBLE_CASES``) with a positive composite. A raw
    ``count(scores)`` is NOT a substitute: three smoke-profile rows are a quorum
    by row count but can never rank or earn emissions.

    Deterministic and derived only from committed rows, because it feeds a
    consensus-critical decision: the same DB state must give every reader the
    same answer.
    """
    score_scope = select(
        Score.agent_id.label("agent_id"),
        _is_ranked().label("eligible"),
        func.count(Score.agent_id).over(partition_by=Score.agent_id).label("cnt"),
        func.row_number()
        .over(
            partition_by=Score.agent_id,
            order_by=(Score.composite.asc(), Score.validator_hotkey.asc()),
        )
        .label("srn"),
    ).where(Score.bench_version == bench_version)
    if agent_ids is not None:
        if not agent_ids:
            return 0
        score_scope = score_scope.where(Score.agent_id.in_(agent_ids))
    per_version = score_scope.subquery()
    # Same median arithmetic as the ledger (see list_eligible_ledger): the
    # lower-middle row by ascending composite represents the agent.
    qualifying = (
        select(
            per_version.c.agent_id,
            _emission_owner_key().label("emission_owner"),
        )
        .join(Agent, Agent.agent_id == per_version.c.agent_id)
        .outerjoin(
            EvaluationPayment,
            EvaluationPayment.agent_id == per_version.c.agent_id,
        )
        .where(
            Agent.status == AgentStatus.SCORED,
            per_version.c.cnt >= SCORING_QUORUM,
            per_version.c.eligible,
            or_(
                per_version.c.srn * 2 == per_version.c.cnt,
                per_version.c.srn * 2 == per_version.c.cnt + 1,
            ),
        )
        .subquery()
    )
    return int(
        await session.scalar(
            select(func.count(func.distinct(qualifying.c.emission_owner)))
        )
        or 0
    )


async def list_eligible_ledger(
    session: AsyncSession,
    *,
    include_fingerprints: bool = True,
    bench_version: int | None = None,
) -> list[LedgerRow]:
    """Return the best eligible score per payment-time coldkey.

    ``include_fingerprints=False`` selects NULL for the anti-copy sketch
    columns (fingerprints + code embedding, several hundred KB per row) —
    the right call for every consumer except the scoring gate, which is the
    only reader that compares them. The public leaderboard, validator ledger
    read, and ticket-eligibility paths were paying that serialization cost on
    every poll for data they never used.

    The persistent ledger the validator folds into KOTH+ATH weights (via
    ``GET /scoring/scores``). "Eligible" = agents in ``scored`` — this excludes
    ``ath_pending_review`` holds (suspected copies) and ``banned`` agents, and
    (because scoring flips ``evaluating -> scored``) is served by the partial
    index ``agents_status_scored_idx``.

    Three levels, all deterministic:

    1. ``agent_best`` takes each agent's **median** score row: it orders the
       agent's ``scores`` rows by composite and keeps the middle one (position
       ``(count+1)/2``), so the k=3 pool's canonical score is the median of its
       validators' composites and a single generous or harsh validator cannot
       move it. It keeps the whole **row** (not a per-column ``MEDIAN``), so
       ``composite`` / ``seed`` / ``run_id`` / ``validator_hotkey`` /
       ``signature`` all come from the same physical row and the exposed
       signature still verifies against the exposed composite. An agent with a
       single score is its own median, so pre-quorum agents degrade cleanly.
    2. join to ``agents`` filtered to ``scored``.
    3. a ``ROW_NUMBER`` window keeps each payment-time coldkey's single best
       agent, even when that owner submitted through multiple hotkeys or names.

    Two senses of "eligible" apply. *Pool* eligibility = ``status == scored``
    (excludes ``ath_pending_review`` holds and ``banned`` agents). *Ranking*
    eligibility = the run administered the full benchmark AND scored a positive
    composite (:func:`_is_ranked`), exposed per-row as ``LedgerRow.eligible``: a
    smoke/practice run *or* a full run that scored 0.000 stays in the pool
    (surfaced as *provisional* / unranked) but is ordered **below** every ranked
    run and dropped by the validator's weight fold, so it can never rank or earn
    emissions. Both the per-agent and per-owner selections prefer a ranked
    row/agent, so neither an inflated small run nor a zero-scoring full run
    shadows a miner's real ranked run.

    Ordering (``eligible DESC, composite DESC, first_seen ASC, agent_id ASC``)
    matches the validator fold's eligibility gate + champion/tail tie-breaks.
    During an open rollout the whole ledger sits on ONE benchmark version,
    chosen by the threshold rule below
    (:data:`~ditto.db.queries.benchmark_rollout.MIN_DESIRED_AUTHORITY_AGENTS`);
    a single read never returns a mix of authoritative versions.

    The per-agent selection is median-of-quorum (:data:`SCORING_QUORUM`); the
    per-agent ``n`` is uniform across the pool because all validators score the
    same platform-generated dataset, so the median row's ``eligible`` flag
    represents the agent.
    """
    from ditto.db.queries.benchmark_rollout import (
        MIN_DESIRED_AUTHORITY_AGENTS,
        SCORING_QUORUM,
        active_bench_version,
        open_rollout,
    )

    canonical_version = await active_bench_version(session)
    rollout = None if bench_version is not None else await open_rollout(session)
    desired_version = rollout.desired_version if rollout is not None else None
    candidate_versions = (
        (bench_version,)
        if bench_version is not None
        else tuple({canonical_version, desired_version} - {None})
    )
    agent_best = (
        select(
            Score.agent_id.label("agent_id"),
            Score.bench_version.label("bench_version"),
            Score.composite.label("composite"),
            Score.tool_mean.label("tool_mean"),
            Score.memory_mean.label("memory_mean"),
            Score.seed.label("seed"),
            Score.run_id.label("run_id"),
            Score.median_ms.label("median_ms"),
            Score.n.label("n"),
            _is_ranked().label("eligible"),
            Score.details.label("details"),
            Score.validator_hotkey.label("validator_hotkey"),
            Score.signature.label("signature"),
            # Row count in the agent's pool, so the median position is (cnt+1)/2.
            func.count(Score.agent_id)
            .over(partition_by=(Score.agent_id, Score.bench_version))
            .label("cnt"),
            # Ascending composite so the middle row (by srn) is the median; the
            # validator_hotkey tie-break keeps the pick deterministic.
            func.row_number()
            .over(
                partition_by=(Score.agent_id, Score.bench_version),
                order_by=(
                    Score.composite.asc(),
                    Score.validator_hotkey.asc(),
                ),
            )
            .label("srn"),
        )
        .where(Score.bench_version.in_(candidate_versions))
        .subquery()
    )
    sketch_columns: tuple[ColumnElement[Any], ...]
    if include_fingerprints:
        sketch_columns = (
            Agent.content_fingerprint.label("content_fingerprint"),
            Agent.structural_fingerprint.label("structural_fingerprint"),
            Agent.prompt_fingerprint.label("prompt_fingerprint"),
            Agent.code_embedding.label("code_embedding"),
        )
    else:
        sketch_columns = (
            null().label("content_fingerprint"),
            null().label("structural_fingerprint"),
            null().label("prompt_fingerprint"),
            null().label("code_embedding"),
        )
    per_agent = (
        select(
            Agent.agent_id.label("agent_id"),
            Agent.miner_hotkey.label("miner_hotkey"),
            EvaluationPayment.miner_coldkey.label("miner_coldkey"),
            _emission_owner_key().label("emission_owner"),
            Agent.sha256.label("sha256"),
            Agent.size_bytes.label("size_bytes"),
            *sketch_columns,
            Agent.normalized_source_hash.label("normalized_source_hash"),
            Agent.code_embed_model.label("code_embed_model"),
            Agent.created_at.label("first_seen"),
            Agent.status.label("status"),
            agent_best.c.composite,
            agent_best.c.tool_mean,
            agent_best.c.memory_mean,
            agent_best.c.seed,
            agent_best.c.run_id,
            agent_best.c.median_ms,
            agent_best.c.n,
            agent_best.c.eligible,
            agent_best.c.details,
            agent_best.c.validator_hotkey,
            agent_best.c.signature,
            agent_best.c.bench_version,
            agent_best.c.cnt,
        )
        .join(agent_best, agent_best.c.agent_id == Agent.agent_id)
        .outerjoin(
            EvaluationPayment,
            EvaluationPayment.agent_id == Agent.agent_id,
        )
        # The median row: the middle by ascending composite. Expressed as
        # integer arithmetic (no division, so it is exact + portable across
        # Postgres and the SQLite test path): the lower-middle index m has
        # 2m == cnt (even) or 2m == cnt+1 (odd). A lone score picks itself; a
        # quorum of 3 picks the 2nd (true median).
        .where(
            Agent.status == AgentStatus.SCORED,
            or_(
                agent_best.c.srn * 2 == agent_best.c.cnt,
                agent_best.c.srn * 2 == agent_best.c.cnt + 1,
            ),
        )
        .subquery()
    )
    # During a collecting rollout the ledger is on exactly ONE benchmark version
    # at a time, chosen by a threshold rule evaluated on each read:
    #
    #   fewer than MIN_DESIRED_AUTHORITY_AGENTS frozen priority members hold a
    #   complete, ranked desired-version quorum  ->  the ACTIVE version stays
    #   authoritative for every agent (desired-version medians are collected
    #   and visible as rollout progress only);
    #   all priority members ready  ->  the DESIRED version becomes authoritative
    #   for the whole pool, and an agent without a desired-version quorum has no
    #   authoritative row at all and drops out. That drop-out is the point of the
    #   threshold: it only happens once enough agents have crossed to still fill
    #   the KOTH emission set (see MIN_DESIRED_AUTHORITY_AGENTS).
    #
    # Deliberately NOT a per-agent switch. Ranking a v_next composite against a
    # v_active composite inside one KOTH fold compares incomparable scales — a
    # newer benchmark applies gates the older one does not, so an already-
    # migrated agent would be systematically penalised against a not-yet-migrated
    # peer. Keeping the whole ledger on one version avoids that entirely.
    #
    # The threshold is computed here, inside the read, from committed rows only:
    # every validator folds the ledger this query serves, so a given DB state
    # must yield the same authority decision for every reader. It must never
    # become time-based or config-drifty.
    #
    # Either way partial desired-version samples never affect ranks or weights,
    # and an explicit ``bench_version`` request stays a historical single-version
    # view (``desired_version`` is None for those, so this whole block is skipped).
    authority_filter: ColumnElement[bool] | None = None
    if desired_version is None:
        version_priority: ColumnElement[Any] = per_agent.c.bench_version
    else:
        assert rollout is not None
        desired_at_quorum = and_(
            per_agent.c.bench_version == desired_version,
            per_agent.c.cnt >= SCORING_QUORUM,
        )
        on_canonical = per_agent.c.bench_version == canonical_version
        # Only genuine, ranked 3/3 desired-version PRIORITY members count, so a
        # rank-6 leak, smoke/practice run, or zero-scoring full run cannot push
        # the ledger over. Shared with the rollout activation gate, which must
        # apply the identical definition — see
        # :func:`count_ranked_quorum_agents`.
        priority_ids = set(
            await session.scalars(
                select(BenchmarkRolloutMember.agent_id).where(
                    BenchmarkRolloutMember.rollout_id == rollout.rollout_id,
                    BenchmarkRolloutMember.position <= MIN_DESIRED_AUTHORITY_AGENTS,
                )
            )
        )
        desired_ready_agents = await count_ranked_quorum_agents(
            session,
            bench_version=desired_version,
            agent_ids=priority_ids,
        )
        if desired_ready_agents >= MIN_DESIRED_AUTHORITY_AGENTS:
            # Whole-pool flip: drop every row that is not a desired-version
            # quorum, so the read cannot return a mix of versions.
            authority_filter = desired_at_quorum
            version_priority = per_agent.c.bench_version
        else:
            version_priority = case((on_canonical, 0), (desired_at_quorum, 1), else_=2)
    selected_version_stmt = select(
        per_agent,
        func.row_number()
        .over(
            partition_by=per_agent.c.agent_id,
            order_by=(version_priority, per_agent.c.bench_version.desc()),
        )
        .label("version_rn"),
    )
    if authority_filter is not None:
        # WHERE is applied before the window function, so the surviving rows are
        # exactly the desired-version medians.
        selected_version_stmt = selected_version_stmt.where(authority_filter)
    selected_version = selected_version_stmt.subquery()
    authoritative = (
        select(selected_version).where(selected_version.c.version_rn == 1).subquery()
    )
    rn = (
        func.row_number()
        .over(
            partition_by=authoritative.c.emission_owner,
            # Eligible-first so a miner is represented by their best full-benchmark
            # agent, not an inflated smoke run; composite breaks ties within a tier.
            order_by=(
                authoritative.c.eligible.desc(),
                authoritative.c.composite.desc(),
                authoritative.c.first_seen.asc(),
                authoritative.c.agent_id.asc(),
            ),
        )
        .label("rn")
    )
    ranked = select(authoritative, rn).subquery()
    stmt = (
        select(ranked)
        .where(ranked.c.rn == 1)
        # Eligible (ranked) entries first, then provisional ones; the public rank
        # and the validator fold both read this order.
        .order_by(
            ranked.c.eligible.desc(),
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
            miner_coldkey=row.miner_coldkey,
            bench_version=row.bench_version,
            content_fingerprint=row.content_fingerprint,
            structural_fingerprint=row.structural_fingerprint,
            normalized_source_hash=row.normalized_source_hash,
            prompt_fingerprint=row.prompt_fingerprint,
            code_embedding=row.code_embedding,
            code_embed_model=row.code_embed_model,
            median_ms=row.median_ms,
            n=row.n,
            eligible=bool(row.eligible),
            details=row.details,
        )
        for row in result
    ]


async def quorum_composites(
    session: AsyncSession,
    agent_ids: Sequence[UUID],
    *,
    bench_versions: dict[UUID, int] | None = None,
) -> dict[UUID, list[float]]:
    """Every accepted composite per agent for the given ids.

    Lets the ledger report the between-validator spread — the k=3 quorum's
    standard error — as the composite's ``composite_stderr`` from data already
    collected, no re-score. One flat read (no aggregation): the SEM is computed
    in Python (:func:`ditto.api_server.endpoints.scoring._quorum_stderr`), so this
    stays portable across Postgres and the SQLite test path (no ``stddev``). The
    row set matches :func:`list_eligible_ledger`'s median (all of the agent's
    score rows), so the SE describes the same quorum the composite came from.
    """
    if not agent_ids:
        return {}
    from ditto.db.queries.benchmark_rollout import active_bench_version

    canonical_version = await active_bench_version(session)
    versions = bench_versions or dict.fromkeys(agent_ids, canonical_version)
    result = await session.execute(
        select(Score.agent_id, Score.composite, Score.bench_version).where(
            Score.agent_id.in_(agent_ids),
            Score.bench_version.in_(set(versions.values())),
        )
    )
    out: dict[UUID, list[float]] = {}
    for agent_id, composite, score_version in result:
        if versions.get(agent_id) == score_version:
            out.setdefault(agent_id, []).append(composite)
    return out


async def list_provisional_ledger(
    session: AsyncSession,
    *,
    bench_version: int | None = None,
) -> list[tuple[LedgerRow, int]]:
    """Return each unfinalized miner's best partially scored submission.

    This is a public-feedback read only. It deliberately considers only agents
    still in ``evaluating`` with at least one accepted score; validator weights
    continue to use :func:`list_eligible_ledger` and therefore remain gated on
    the finalized ``scored`` status. Numeric fields are medians of the accepted
    reports available so far. Opaque run details are omitted because, before
    quorum, no single validator row is the canonical result.
    """
    from ditto.db.queries.benchmark_rollout import active_bench_version

    canonical_version = bench_version or await active_bench_version(session)
    rows = (
        await session.execute(
            select(Agent, Score)
            .join(Score, Score.agent_id == Agent.agent_id)
            .where(
                Agent.status == AgentStatus.EVALUATING,
                Score.bench_version == canonical_version,
            )
            .order_by(
                Agent.created_at.asc(),
                Agent.agent_id.asc(),
                Score.generated_at.asc(),
                Score.validator_hotkey.asc(),
            )
        )
    ).all()

    by_agent: dict[UUID, tuple[Agent, list[Score]]] = {}
    for agent, score in rows:
        if agent.agent_id not in by_agent:
            by_agent[agent.agent_id] = (agent, [])
        by_agent[agent.agent_id][1].append(score)

    candidates: list[tuple[LedgerRow, int]] = []
    for agent, scores in by_agent.values():
        if not scores:
            continue
        representative = sorted(
            scores, key=lambda score: (score.composite, score.validator_hotkey)
        )[(len(scores) - 1) // 2]
        composite = float(median(score.composite for score in scores))
        tool_mean = float(median(score.tool_mean for score in scores))
        memory_mean = float(median(score.memory_mean for score in scores))
        median_ms = int(median(score.median_ms for score in scores))
        n = int(median(score.n for score in scores))
        candidates.append(
            (
                LedgerRow(
                    miner_hotkey=agent.miner_hotkey,
                    agent_id=agent.agent_id,
                    composite=composite,
                    tool_mean=tool_mean,
                    memory_mean=memory_mean,
                    first_seen=agent.created_at,
                    sha256=agent.sha256,
                    size_bytes=agent.size_bytes,
                    run_id=representative.run_id,
                    seed=representative.seed,
                    validator_hotkey=representative.validator_hotkey,
                    signature=representative.signature,
                    status=AgentStatus.EVALUATING,
                    bench_version=representative.bench_version,
                    median_ms=median_ms,
                    n=n,
                    eligible=n >= MIN_ELIGIBLE_CASES and composite > 0.0,
                    details=None,
                ),
                len(scores),
            )
        )

    candidates.sort(
        key=lambda candidate: (
            not candidate[0].eligible,
            -candidate[0].composite,
            candidate[0].first_seen,
            str(candidate[0].agent_id),
        ),
    )
    best_by_miner: dict[str, tuple[LedgerRow, int]] = {}
    for candidate in candidates:
        best_by_miner.setdefault(candidate[0].miner_hotkey, candidate)
    return list(best_by_miner.values())


async def list_scored_bench_versions(session: AsyncSession) -> list[int]:
    """Every benchmark version with at least one accepted score, newest first.

    Backs the dashboard's per-version history pills: a version earns a pill as
    soon as its first score lands and keeps it as a historical view forever.
    ``NULL`` (pre-versioning legacy) rows carry no version to browse by and are
    excluded.
    """
    rows = await session.scalars(
        select(Score.bench_version)
        .where(Score.bench_version.is_not(None))
        .distinct()
        .order_by(Score.bench_version.desc())
    )
    return [int(version) for version in rows]


async def get_score_counts(
    session: AsyncSession,
    agent_ids: list[UUID],
    *,
    bench_versions: dict[UUID, int] | None = None,
) -> dict[UUID, int]:
    """Return accepted validator-score counts for the requested agents."""
    if not agent_ids:
        return {}
    from ditto.db.queries.benchmark_rollout import active_bench_version

    canonical_version = await active_bench_version(session)
    versions = bench_versions or dict.fromkeys(agent_ids, canonical_version)
    rows = (
        await session.execute(
            select(
                Score.agent_id,
                Score.bench_version,
                func.count(Score.validator_hotkey),
            )
            .where(
                Score.agent_id.in_(agent_ids),
                Score.bench_version.in_(set(versions.values())),
            )
            .group_by(Score.agent_id, Score.bench_version)
        )
    ).all()
    return {
        agent_id: int(count)
        for agent_id, score_version, count in rows
        if versions.get(agent_id) == score_version
    }
