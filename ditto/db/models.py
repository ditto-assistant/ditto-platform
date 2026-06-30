"""SQLAlchemy 2.0 declarative models for the Ditto data layer.

Alembic migrations under :file:`alembic/versions/` own the schema;
these models describe it in Python so :class:`AsyncSession` queries
hydrate into typed objects. Models and migrations must stay in sync.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    Enum,
    Float,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy import (
    UUID as SaUUID,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TIMESTAMP

from ditto.api_models.agent_status import AgentStatus

# Per-case detail is a JSON blob: JSONB on Postgres (indexable, compact),
# plain JSON on the SQLite unit-test fallback. The variant keeps one model
# working across both dialects.
_JSON_VARIANT = JSON().with_variant(JSONB(), "postgresql")

# Naming convention so alembic autogenerate produces deterministic constraint
# names instead of random SHA suffixes.
_NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base for every Ditto ORM model."""

    metadata = MetaData(naming_convention=_NAMING_CONVENTION)


class Agent(Base):
    """One row of the ``agents`` table.

    Represents a single miner submission. Lifecycle is tracked through
    :class:`AgentStatus`; transitions are owned by the upload, evaluator,
    and scoring modules.
    """

    __tablename__ = "agents"

    agent_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), primary_key=True)
    """Primary key. Client-generated UUID supplied at INSERT time."""

    miner_hotkey: Mapped[str] = mapped_column(Text, nullable=False)
    """SS58 hotkey of the submitting miner."""

    name: Mapped[str] = mapped_column(Text, nullable=False)
    """Human-friendly agent name supplied by the miner."""

    sha256: Mapped[str] = mapped_column(Text, nullable=False)
    """SHA-256 of the uploaded tarball, hex encoded."""

    status: Mapped[AgentStatus] = mapped_column(
        Enum(
            AgentStatus,
            name="agentstatus",
            values_callable=lambda enum_cls: [m.value for m in enum_cls],
            create_constraint=True,
        ),
        nullable=False,
        server_default=text("'uploaded'"),
    )
    """Current state in the submission state machine."""

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    """Upload timestamp (UTC)."""

    __table_args__ = (
        UniqueConstraint(
            "agent_id", "miner_hotkey", name="agents_agent_id_miner_hotkey_key"
        ),
        Index("agents_miner_hotkey_idx", "miner_hotkey"),
        Index(
            "agents_status_evaluating_idx",
            "status",
            postgresql_where=text("status = 'evaluating'"),
        ),
    )


class EvaluationPayment(Base):
    """One row of the ``evaluation_payments`` table.

    The composite primary key ``(block_hash, extrinsic_index)`` is the
    replay-protection mechanism: the same on-chain payment proof cannot
    be inserted twice. ``UNIQUE (agent_id)`` enforces the 1:1 invariant
    (one upload = one payment). The composite FK on
    ``(agent_id, miner_hotkey)`` documents the ownership invariant in
    DDL so future endpoints can't silently break it.
    """

    __tablename__ = "evaluation_payments"

    block_hash: Mapped[str] = mapped_column(Text, nullable=False)
    """Hash of the block containing the payment extrinsic. PK part 1."""

    extrinsic_index: Mapped[int] = mapped_column(Integer, nullable=False)
    """Zero-based index of the extrinsic within the block. PK part 2."""

    agent_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    """FK to ``agents.agent_id``. The agent this payment funds. ``UNIQUE``."""

    miner_hotkey: Mapped[str] = mapped_column(Text, nullable=False)
    """Signer hotkey on the payment extrinsic. FK-bound to ``agents.miner_hotkey``."""

    miner_coldkey: Mapped[str] = mapped_column(Text, nullable=False)
    """Coldkey that owns the hotkey at payment time. Snapshot for audit."""

    amount_rao: Mapped[int] = mapped_column(BigInteger, nullable=False)
    """Payment amount in rao (1 TAO = 1e9 rao)."""

    dest_address: Mapped[str] = mapped_column(Text, nullable=False)
    """SS58 address that received the payment."""

    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    """On-chain block timestamp."""

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    """Row insertion timestamp."""

    __table_args__ = (
        PrimaryKeyConstraint(
            "block_hash", "extrinsic_index", name="evaluation_payments_pkey"
        ),
        UniqueConstraint("agent_id", name="evaluation_payments_agent_id_key"),
        ForeignKeyConstraint(
            ["agent_id", "miner_hotkey"],
            ["agents.agent_id", "agents.miner_hotkey"],
            ondelete="RESTRICT",
            name="evaluation_payments_agent_id_miner_hotkey_fkey",
        ),
        CheckConstraint("amount_rao > 0", name="evaluation_payments_amount_rao_check"),
        CheckConstraint(
            "extrinsic_index >= 0",
            name="evaluation_payments_extrinsic_index_check",
        ),
        Index("evaluation_payments_miner_hotkey_idx", "miner_hotkey"),
    )


class Score(Base):
    """One validator's DittoBench score for one agent.

    The composite primary key ``(agent_id, validator_hotkey)`` keeps a
    single current score per validator per agent: a validator re-scoring
    the same agent (new ``run_id``) upserts this row rather than appending
    history. ``agent_id`` is a single-column FK to ``agents.agent_id`` with
    ``ON DELETE CASCADE`` because a score is derived data — deleting the
    agent discards its scores.

    Aggregates (``composite`` / ``tool_mean`` / ``memory_mean`` /
    ``median_ms`` / ``n``) are first-class columns so weight computation and
    leaderboards query them directly; ``details`` carries the optional
    per-case breakdown verbatim for audit/dispute. The platform records what
    the validator reports and never recomputes ``composite``.
    """

    __tablename__ = "scores"

    agent_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    """FK to ``agents.agent_id``. PK part 1."""

    validator_hotkey: Mapped[str] = mapped_column(Text, nullable=False)
    """SS58 hotkey of the reporting validator. PK part 2."""

    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    """Scoring-engine run identifier (the value the signature is bound to)."""

    seed: Mapped[int] = mapped_column(BigInteger, nullable=False)
    """Dataset seed used for the run (anti-overfit reproducibility)."""

    composite: Mapped[float] = mapped_column(Float, nullable=False)
    """Aggregate score in [0, 1] as reported (not recomputed)."""

    tool_mean: Mapped[float] = mapped_column(Float, nullable=False)
    """Mean tool accuracy in [0, 1]."""

    memory_mean: Mapped[float] = mapped_column(Float, nullable=False)
    """Mean memory recall in [0, 1]."""

    median_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    """Median per-case latency in milliseconds."""

    n: Mapped[int] = mapped_column(Integer, nullable=False)
    """Number of cases scored."""

    details: Mapped[dict | None] = mapped_column(_JSON_VARIANT, nullable=True)
    """Optional per-case breakdown ``{"per_case": [...]}`` for audit."""

    generated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    """When the scoring engine produced the report (UTC)."""

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    """When this row was first inserted (UTC)."""

    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    """When this row was last upserted (UTC)."""

    __table_args__ = (
        PrimaryKeyConstraint("agent_id", "validator_hotkey", name="scores_pkey"),
        ForeignKeyConstraint(
            ["agent_id"],
            ["agents.agent_id"],
            ondelete="CASCADE",
            name="scores_agent_id_fkey",
        ),
        CheckConstraint(
            "composite >= 0 AND composite <= 1", name="scores_composite_range_check"
        ),
        CheckConstraint(
            "tool_mean >= 0 AND tool_mean <= 1", name="scores_tool_mean_range_check"
        ),
        CheckConstraint(
            "memory_mean >= 0 AND memory_mean <= 1",
            name="scores_memory_mean_range_check",
        ),
        CheckConstraint("n >= 0", name="scores_n_check"),
        CheckConstraint("median_ms >= 0", name="scores_median_ms_check"),
        Index("scores_agent_id_idx", "agent_id"),
    )
