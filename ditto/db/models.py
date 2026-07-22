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
    Boolean,
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
from ditto.api_models.ticket_status import TicketStatus

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

    version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    """Immutable revision within ``(miner_hotkey, name)``; null for legacy rows."""

    sha256: Mapped[str] = mapped_column(Text, nullable=False)
    """SHA-256 of the uploaded tarball, hex encoded."""

    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    """Uploaded tarball size in bytes. Nullable for rows written before the
    ledger migration; a cheap near-dup signal (a copy has a near-identical size)."""

    # The four anti-copy sketch columns below hold multi-hundred-KB JSON blobs
    # (k=256 minhash arrays, embedding vectors). They are deferred behind the
    # "anticopy" group: the dominant DB cost in production was serializing them
    # on every Agent read (leaderboard/ticket/screener paths that never look at
    # them). Readers that DO need them — the scoring gate, the admin copy-review
    # comparison, the fingerprint backfill — load with
    # ``undefer_group("anticopy")`` / ``include_anticopy=True``; under the async
    # session a forgotten undefer fails loudly (MissingGreenlet) rather than
    # silently re-fetching.
    content_fingerprint: Mapped[dict | None] = mapped_column(
        _JSON_VARIANT, nullable=True, deferred=True, deferred_group="anticopy"
    )

    """Shingle MinHash sketch of the tarball source (see
    :mod:`ditto.api_server.fingerprint`). Feeds the anti-copy gate's content-level
    signal: a reindented/renamed/reformatted or locally-edited copy keeps a
    near-identical fingerprint even when its byte size drifts past the
    ``size_bytes`` tolerance. Nullable for rows written before this landed and for
    tarballs that were unreadable/empty at upload (the gate reads null as "no
    content match")."""

    structural_fingerprint: Mapped[dict | None] = mapped_column(
        _JSON_VARIANT, nullable=True, deferred=True, deferred_group="anticopy"
    )
    """AST-level shingle MinHash sketch of the crate, same ``{v,k,card,m}`` shape as
    :attr:`content_fingerprint` (see the dittobench ``astfp`` package). Computed by
    dittobench where the crate is unpacked and a Rust parser exists, it hashes only
    the parse-tree *shape*, so it survives identifier renaming that the lexical
    channel misses. Arrives (unsigned, advisory) on the score report and is written
    at score time. Nullable: null before this landed, on the local harness_url
    path, or when the crate has no parseable Rust."""

    normalized_source_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    """SHA-256 (hex) of the crate's canonicalized source — comments stripped,
    whitespace removed, files sorted (see
    :func:`ditto.api_server.fingerprint.compute_normalized_source_hash`). Feeds the
    anti-copy gate's "exact-repack" signal: an *equality* match means the same
    source repackaged (reformat / re-comment / file rename+reorder), which the
    ``sha256`` and shingle sketches miss. Computed at upload. Nullable for rows
    written before this landed and for tarballs unreadable/empty at upload (the
    gate reads null as "no repack match")."""

    dataset_seed: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    """The per-submission dataset seed the platform generated at job-ready
    (``uploaded -> evaluating``). All k=3 validators score against THIS seed, so
    their scores are comparable (the median-of-3 is over one dataset). Fresh and
    unpredictable per submission, so a published run's answer key does not help the
    miner's next (differently-seeded) submission. Null until the agent is promoted
    to ``evaluating``. Bounded to the signed 64-bit range ``scores.seed`` stores."""

    dataset_sha256: Mapped[str | None] = mapped_column(Text, nullable=True)
    """SHA-256 (hex) of the fully-rendered dataset the generate service produced
    for ``dataset_seed`` (the DatasetArtifact digest). Issued to every validator in
    the ticket; the validator's scoring call regenerates the dataset from the seed
    and the scoring API fails if it does not hash to this — tamper-evidence that
    all three validators scored the exact dataset the platform pinned. Null until
    job-ready."""

    dataset_run_size: Mapped[str | None] = mapped_column(Text, nullable=True)
    """The generator profile the dataset was built with (``small|medium|full``),
    issued with the ticket so the validator's scoring call uses the same profile.
    Null until job-ready."""

    dataset_seed_block: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    """Number of the on-chain block whose hash the ``dataset_seed`` was derived
    from (see :mod:`ditto.api_server.onchain_seed`). Pinned at job-ready so anyone
    can fetch that block, recompute ``derive_seed(block_hash, agent_id)``, and
    verify the seed the platform published — the seed is not platform-chosen. Null
    until job-ready, or when chain-derivation is unavailable (fallback path)."""

    dataset_seed_block_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    """Hex hash of :attr:`dataset_seed_block`, stored alongside the number so a
    verifier need not trust the platform's block lookup: it recomputes the seed
    directly from this hash + the agent id. Null until job-ready / on fallback."""

    code_embedding: Mapped[list | None] = mapped_column(
        _JSON_VARIANT, nullable=True, deferred=True, deferred_group="anticopy"
    )
    """Unit-norm code-embedding vector (JSON float array) of the crate's canonical
    source, from the self-hosted the code-embedding signal embedding service (see
    :mod:`ditto.api_server.embedding`). The rename/refactor-robust anti-copy signal:
    a code embedder scores a renamed+refactored copy high but a genuinely different
    agent low, so it is orthogonal to same-harness convergence. Stored in shadow
    mode — computed for every agent (calibration + retroactive) but not yet a hold
    trigger. Nullable before this landed, when the embedder is disabled
    (``CODE_EMBEDDER_URL`` unset), or on any best-effort embed failure."""

    code_embed_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    """``model@revision`` provenance tag of :attr:`code_embedding` (see
    :attr:`ditto.api_server.embedding.EmbeddingConfig.model_tag`). Lets a model
    change drive a re-embed sweep, and lets the gate compare only same-model vectors
    (a cross-model cosine is meaningless). Nullable alongside the vector."""

    prompt_fingerprint: Mapped[dict | None] = mapped_column(
        _JSON_VARIANT, nullable=True, deferred=True, deferred_group="anticopy"
    )
    """Word-shingle MinHash sketch of the crate's prompt-length string literals
    (see :func:`ditto.api_server.fingerprint.compute_prompt_fingerprint`), same
    ``{v,k,card,m}`` shape as :attr:`content_fingerprint` but with a string ``v`` so
    it never compares against the lexical/structural channels. The prompt
    surface: because it hashes string *contents*, it survives identifier renaming
    that defeats the lexical + normalized-source channels. Computed at upload.
    Stored in shadow mode — captured for every agent (calibration + retroactive
    analysis) but not yet a hold trigger on its own, since honest agents on the same
    reference harness share scaffolding prompts and the orthogonal-to-convergence
    signals it must fuse with are not built yet. Nullable before this landed / no
    prompt-length literal / unreadable tarball."""

    duplicate_of: Mapped[UUID | None] = mapped_column(
        SaUUID(as_uuid=True), nullable=True
    )
    """Set when the anti-copy gate holds this agent in ``ath_pending_review``:
    the ``agent_id`` of the earlier submission it appears to duplicate."""

    review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    """Human-readable reason a held agent was routed to review (audit trail)."""

    screening_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    """Public-safe reason category for a failed build/serve screen.

    The screener's raw ``detail`` may contain an untrusted Docker build-log tail,
    so the endpoint maps it to a fixed category before persisting it here.
    """

    screening_reason_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    """Stable public/operator-safe machine code for the screening outcome."""

    screening_policy_version: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    """Latest anti-cheat screening policy this submission passed.

    Zero marks submissions screened before policy attestation was introduced.
    Validators may score only submissions at the platform's required version.
    """

    screened_image_sha256: Mapped[str | None] = mapped_column(Text, nullable=True)
    """SHA-256 of the screener-exported Docker image archive."""

    screened_image_size_bytes: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True
    )
    """Exact byte size of the screener-exported Docker image archive."""

    screened_image_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    """Docker content ID verified by the screener and each validator."""

    screened_image_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    """Screener-owned tag expected inside the Docker save archive."""

    screened_image_upload_id: Mapped[UUID | None] = mapped_column(
        SaUUID(as_uuid=True), nullable=True
    )
    """Platform-minted immutable multipart object identity."""

    screened_image_verified_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    """When the platform streamed and verified the complete archive bytes."""

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
        UniqueConstraint(
            "miner_hotkey",
            "name",
            "version",
            name="agents_hotkey_name_version_key",
        ).ddl_if(dialect="postgresql"),
        CheckConstraint(
            "version IS NULL OR version > 0", name="agents_version_positive_check"
        ),
        CheckConstraint(
            "(screened_image_sha256 IS NULL AND screened_image_size_bytes IS NULL "
            "AND screened_image_id IS NULL AND screened_image_ref IS NULL "
            "AND screened_image_upload_id IS NULL "
            "AND screened_image_verified_at IS NULL) OR "
            "(length(screened_image_sha256) = 64 AND screened_image_size_bytes > 0 "
            "AND length(screened_image_id) = 71 AND length(screened_image_ref) > 0 "
            "AND screened_image_upload_id IS NOT NULL "
            "AND screened_image_verified_at IS NOT NULL)",
            name="agents_screened_image_fields_check",
        ),
        Index("agents_miner_hotkey_idx", "miner_hotkey"),
        Index("agents_sha256_idx", "sha256"),
        # Exact-repack duplicate lookups for the quarantine review console.
        Index("agents_normalized_source_hash_idx", "normalized_source_hash"),
        Index(
            "agents_status_evaluating_idx",
            "status",
            postgresql_where=text("status = 'evaluating'"),
        ),
        # Screener polls for agents in the 'uploaded' state; a partial index
        # keeps that lookup cheap (mirrors the evaluating index).
        Index(
            "agents_status_uploaded_idx",
            "status",
            postgresql_where=text("status = 'uploaded'"),
        ),
        # The validator's ledger read (GET /scoring/scores) selects agents in
        # 'scored'; a partial index keeps that scan cheap (mirrors the two above).
        Index(
            "agents_status_scored_idx",
            "status",
            postgresql_where=text("status = 'scored'"),
        ),
        # ``duplicate_of`` points at the earlier submission a held agent copies.
        ForeignKeyConstraint(
            ["duplicate_of"],
            ["agents.agent_id"],
            ondelete="SET NULL",
            name="agents_duplicate_of_fkey",
        ),
    )


class ScreenedImageUpload(Base):
    """Attempt-bound multipart upload verified before a passing verdict."""

    __tablename__ = "screened_image_uploads"

    image_upload_id: Mapped[UUID] = mapped_column(
        SaUUID(as_uuid=True), primary_key=True
    )
    agent_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    attempt_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    screener_hotkey: Mapped[str] = mapped_column(Text, nullable=False)
    storage_upload_id: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    image_id: Mapped[str] = mapped_column(Text, nullable=False)
    image_ref: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="initiated"
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    verified_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["agent_id"],
            ["agents.agent_id"],
            ondelete="CASCADE",
            name="screened_image_uploads_agent_id_fkey",
        ),
        ForeignKeyConstraint(
            ["attempt_id"],
            ["screening_attempts.attempt_id"],
            ondelete="CASCADE",
            name="screened_image_uploads_attempt_id_fkey",
        ),
        CheckConstraint(
            "status IN ('initiated', 'verified', 'aborted')",
            name="screened_image_uploads_status_check",
        ),
        CheckConstraint("size_bytes > 0", name="screened_image_uploads_size_check"),
        Index("screened_image_uploads_attempt_idx", "attempt_id"),
        Index("screened_image_uploads_status_expires_idx", "status", "expires_at"),
    )


class ScreeningAttempt(Base):
    """One claimed, versioned screening lease for a submission."""

    __tablename__ = "screening_attempts"

    attempt_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), primary_key=True)
    agent_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    screener_hotkey: Mapped[str] = mapped_column(Text, nullable=False)
    policy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    deadline: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    public_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    duplicate_of: Mapped[UUID | None] = mapped_column(
        SaUUID(as_uuid=True), nullable=True
    )
    build_only: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    """This attempt only rebuilds an already-adjudicated submission's missing
    prerequisites (screened image / dataset); the screener must NOT re-run the
    anti-cheat source review and cannot quarantine. Set when an EVALUATING agent
    on the current policy is re-claimed — its review was already cleared, so a
    re-screen would wrongly re-judge an approved artifact."""

    __table_args__ = (
        ForeignKeyConstraint(
            ["agent_id"],
            ["agents.agent_id"],
            ondelete="CASCADE",
            name="screening_attempts_agent_id_fkey",
        ),
        ForeignKeyConstraint(
            ["duplicate_of"],
            ["agents.agent_id"],
            ondelete="SET NULL",
            name="screening_attempts_duplicate_of_fkey",
        ),
        CheckConstraint(
            "policy_version > 0",
            name="screening_attempts_policy_version_check",
        ),
        CheckConstraint(
            "status IN ('running', 'passed', 'rejected', 'failed', 'expired', "
            "'quarantined')",
            name="screening_attempts_status_check",
        ),
        CheckConstraint(
            "deadline >= started_at",
            name="screening_attempts_deadline_check",
        ),
        CheckConstraint(
            "finished_at IS NULL OR finished_at >= started_at",
            name="screening_attempts_finished_check",
        ),
        CheckConstraint(
            "reason_code IS NULL OR length(reason_code) BETWEEN 1 AND 64",
            name="screening_attempts_reason_code_check",
        ),
        Index("screening_attempts_agent_started_idx", "agent_id", "started_at"),
        Index(
            "screening_attempts_one_running_idx",
            "agent_id",
            unique=True,
            postgresql_where=text("status = 'running'"),
            sqlite_where=text("status = 'running'"),
        ),
    )


class AthReview(Base):
    """Durable, immutable-evidence audit record for an ATH copy hold."""

    __tablename__ = "ath_reviews"

    review_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), primary_key=True)
    agent_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    opened_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    reopened_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    resolved_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    resolved_by: Mapped[str | None] = mapped_column(Text)
    resolution: Mapped[str | None] = mapped_column(Text)
    resolution_reason: Mapped[str | None] = mapped_column(Text)
    original_duplicate_of: Mapped[UUID | None] = mapped_column(SaUUID(as_uuid=True))
    original_reason: Mapped[str | None] = mapped_column(Text)
    original_policy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    original_evidence: Mapped[dict] = mapped_column(_JSON_VARIANT, nullable=False)
    algorithm_provenance: Mapped[dict] = mapped_column(_JSON_VARIANT, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(["agent_id"], ["agents.agent_id"], ondelete="RESTRICT"),
        ForeignKeyConstraint(
            ["original_duplicate_of"], ["agents.agent_id"], ondelete="RESTRICT"
        ),
        UniqueConstraint("agent_id", name="ath_reviews_agent_id_key"),
        CheckConstraint(
            "status IN ('pending', 'resolved')", name="ath_reviews_status_check"
        ),
        CheckConstraint(
            "resolution IS NULL OR resolution IN ('clear', 'reject')",
            name="ath_reviews_resolution_check",
        ),
        CheckConstraint(
            "(status = 'pending' AND resolved_at IS NULL AND resolved_by IS NULL "
            "AND resolution IS NULL AND resolution_reason IS NULL) OR "
            "(status = 'resolved' AND resolved_at IS NOT NULL "
            "AND resolved_by IS NOT NULL "
            "AND length(trim(resolved_by)) BETWEEN 1 AND 120 "
            "AND resolution IS NOT NULL "
            "AND resolution IN ('clear', 'reject') "
            "AND resolution_reason IS NOT NULL "
            "AND length(trim(resolution_reason)) BETWEEN 3 AND 500)",
            name="ath_reviews_lifecycle_check",
        ),
        Index("ath_reviews_status_opened_idx", "status", "opened_at", "review_id"),
    )


class AthReviewAction(Base):
    """Append-only operator lifecycle history for an ATH review."""

    __tablename__ = "ath_review_actions"

    action_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), primary_key=True)
    review_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    actor: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[dict] = mapped_column(_JSON_VARIANT, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["review_id"], ["ath_reviews.review_id"], ondelete="CASCADE"
        ),
        CheckConstraint(
            "action IN ('reopen', 'clear', 'reject')",
            name="ath_review_actions_action_check",
        ),
        CheckConstraint(
            "length(trim(reason)) BETWEEN 3 AND 500",
            name="ath_review_actions_reason_check",
        ),
        CheckConstraint(
            "length(trim(actor)) BETWEEN 1 AND 120",
            name="ath_review_actions_actor_check",
        ),
        Index(
            "ath_review_actions_review_created_idx",
            "review_id",
            "created_at",
            "action_id",
        ),
    )


class ScreeningQuarantine(Base):
    """Append-only quarantine decision plus its operator resolution."""

    __tablename__ = "screening_quarantines"

    quarantine_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), primary_key=True)
    agent_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    attempt_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    screener_hotkey: Mapped[str] = mapped_column(Text, nullable=False)
    policy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    manifest_digest: Mapped[str] = mapped_column(Text, nullable=False)
    finding_digest: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason_code: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[list | None] = mapped_column(_JSON_VARIANT, nullable=True)
    """Bounded public-safe policy evidence trail (module, code, summary,
    digest) shipped by the screener on quarantine. Display data for the
    operator console; the signed verdict binds only the digests. Null for
    rows written before the review payloads landed."""

    finding: Mapped[dict | None] = mapped_column(_JSON_VARIANT, nullable=True)
    """Bounded source-review finding (risk, confidence, categories, flagged
    path/line evidence, summary). Its canonical JSON hashes to
    ``finding_digest``, which the verdict signature covers, so this payload is
    verifiable end to end. Null before the review payloads landed and for
    quarantines with no source-review finding."""

    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="active")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    resolved_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolution: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolution_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        ForeignKeyConstraint(["agent_id"], ["agents.agent_id"], ondelete="CASCADE"),
        ForeignKeyConstraint(
            ["attempt_id"],
            ["screening_attempts.attempt_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("attempt_id", name="screening_quarantines_attempt_id_key"),
        CheckConstraint(
            "policy_version > 0", name="screening_quarantines_policy_check"
        ),
        CheckConstraint(
            "status IN ('active', 'resolved')",
            name="screening_quarantines_status_check",
        ),
        CheckConstraint(
            "resolution IS NULL OR resolution IN ('release', 'rescreen', 'reject')",
            name="screening_quarantines_resolution_check",
        ),
        Index(
            "screening_quarantines_one_active_agent_idx",
            "agent_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
            sqlite_where=text("status = 'active'"),
        ),
        Index("screening_quarantines_created_idx", "created_at"),
        # Miner-history lookups (all quarantines for one agent, any status).
        Index("screening_quarantines_agent_idx", "agent_id"),
    )


class ScreeningQuarantineResolution(Base):
    """Append-only operator action history for a screening quarantine."""

    __tablename__ = "screening_quarantine_resolutions"

    resolution_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), primary_key=True)
    quarantine_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    resolution: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    actor: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["quarantine_id"],
            ["screening_quarantines.quarantine_id"],
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "resolution IN ('release', 'rescreen', 'reject')",
            name="screening_quarantine_resolutions_resolution_check",
        ),
        Index(
            "screening_quarantine_resolutions_quarantine_created_idx",
            "quarantine_id",
            "created_at",
        ),
    )


class ScreeningDispute(Base):
    """One miner-authenticated appeal of a rejected screening decision."""

    __tablename__ = "screening_disputes"

    dispute_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), primary_key=True)
    agent_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    quarantine_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    miner_hotkey: Mapped[str] = mapped_column(Text, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    resolved_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolution: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolution_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        ForeignKeyConstraint(["agent_id"], ["agents.agent_id"], ondelete="CASCADE"),
        ForeignKeyConstraint(
            ["quarantine_id"],
            ["screening_quarantines.quarantine_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("agent_id", name="screening_disputes_agent_id_key"),
        UniqueConstraint("quarantine_id", name="screening_disputes_quarantine_id_key"),
        CheckConstraint(
            "length(message) BETWEEN 20 AND 1000",
            name="screening_disputes_message_length_check",
        ),
        CheckConstraint(
            "status IN ('pending', 'resolved')",
            name="screening_disputes_status_check",
        ),
        CheckConstraint(
            "resolution IS NULL OR resolution IN ('release', 'uphold')",
            name="screening_disputes_resolution_check",
        ),
        Index("screening_disputes_status_created_idx", "status", "created_at"),
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

    bench_version: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    """Benchmark semantics this signed score and its dataset were produced under."""

    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    """Scoring-engine run identifier (part of the value the signature is bound to)."""

    signature: Mapped[str | None] = mapped_column(Text, nullable=True)
    """The reporting validator's sr25519 signature over the score payload, hex
    encoded. Persisted so the exposed ledger is self-verifying. Nullable for rows
    written before the ledger migration."""

    seed: Mapped[int] = mapped_column(BigInteger, nullable=False)
    """Dataset seed used for the run (anti-overfit reproducibility)."""

    composite: Mapped[float] = mapped_column(Float, nullable=False)
    """Aggregate benchmark score in [0, 1] (not recomputed by the platform)."""

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
        PrimaryKeyConstraint(
            "agent_id", "bench_version", "validator_hotkey", name="scores_pkey"
        ),
        ForeignKeyConstraint(
            ["agent_id"],
            ["agents.agent_id"],
            ondelete="CASCADE",
            name="scores_agent_id_fkey",
        ),
        CheckConstraint(
            "composite >= 0 AND composite <= 1",
            name="scores_composite_range_check",
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
        CheckConstraint("bench_version > 0", name="scores_bench_version_positive"),
        Index("scores_agent_id_idx", "agent_id"),
        # Dashboard/ledger reads select one benchmark era before grouping or
        # ranking scores.  Keep the aggregate columns in the index so the
        # frequently-polled activity snapshot can stay index-only as older
        # benchmark eras accumulate.
        Index(
            "scores_bench_version_agent_composite_idx",
            "bench_version",
            "agent_id",
            "composite",
            "validator_hotkey",
            postgresql_include=["updated_at"],
        ),
    )


class ConfirmationScore(Base):
    """One append-only shared-seed rescore result: a top-5 confirmation ledger row.

    The continual top-5 shared-seed rescore lane
    (``docs/top5-rescore-lane.md``) re-scores each emission-set member on the
    **champion-anchored** CRN seed set. Unlike :class:`Score` (the authoritative
    k=3 record, one upserted row per validator), this ledger is **append-only**:
    one immutable row per ``(agent_id, validator_hotkey, bench_version, seed)``,
    inserted with ``ON CONFLICT DO NOTHING`` and **never UPDATEd or deleted**. The
    longer a champion reigns, the more rows accumulate — the record grows
    monotonically and every seed's score is kept forever (auditable, no
    destructive read-modify-write).

    The KOTH fold (and its ``koth.py`` platform mirror) read the paired evidence
    from this history — grouped by seed, lower-median across validators — so
    "more seeds" is just "more rows to median over". The authoritative k=3
    ``scores`` table is untouched: no 4th ticket, no 4th ``Score`` PK row, no
    change to finalization. This ledger is a separate, additive store.
    """

    __tablename__ = "confirmation_scores"

    agent_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    """FK to ``agents.agent_id``. Unique-key part 1."""

    validator_hotkey: Mapped[str] = mapped_column(Text, nullable=False)
    """SS58 hotkey of the reporting validator. Unique-key part 2."""

    bench_version: Mapped[int] = mapped_column(Integer, nullable=False)
    """Benchmark semantics this signed rescore was produced under. Unique-key
    part 3 — the champion-anchored seed set is keyed on the major version."""

    seed: Mapped[int] = mapped_column(BigInteger, nullable=False)
    """The champion-anchored CRN seed the member was re-scored on. Unique-key
    part 4 — a validator contributes at most one composite per seed, immutably."""

    composite: Mapped[float] = mapped_column(Float, nullable=False)
    """Aggregate score in [0, 1] on this seed's dataset (as reported)."""

    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    """Scoring-engine run identifier for this seed's benchmark run."""

    signature: Mapped[str | None] = mapped_column(Text, nullable=True)
    """The reporting validator's sr25519 signature over the parent score payload,
    hex encoded, so the ledger row is self-verifying alongside the k=3 receipt."""

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    """When this immutable row was appended (UTC). Never updated."""

    __table_args__ = (
        PrimaryKeyConstraint(
            "agent_id",
            "bench_version",
            "validator_hotkey",
            "seed",
            name="confirmation_scores_pkey",
        ),
        ForeignKeyConstraint(
            ["agent_id"],
            ["agents.agent_id"],
            ondelete="CASCADE",
            name="confirmation_scores_agent_id_fkey",
        ),
        CheckConstraint(
            "composite >= 0 AND composite <= 1",
            name="confirmation_scores_composite_range_check",
        ),
        CheckConstraint(
            "bench_version > 0", name="confirmation_scores_bench_version_positive"
        ),
        CheckConstraint("seed >= 0", name="confirmation_scores_seed_check"),
        Index("confirmation_scores_agent_version_idx", "agent_id", "bench_version"),
    )


class BenchmarkDataset(Base):
    """Immutable dataset pin for one agent and benchmark version."""

    __tablename__ = "benchmark_datasets"

    agent_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    bench_version: Mapped[int] = mapped_column(Integer, nullable=False)
    seed: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(Text, nullable=False)
    run_size: Mapped[str] = mapped_column(Text, nullable=False)
    seed_block: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    seed_block_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        PrimaryKeyConstraint(
            "agent_id", "bench_version", name="benchmark_datasets_pkey"
        ),
        ForeignKeyConstraint(["agent_id"], ["agents.agent_id"], ondelete="CASCADE"),
        CheckConstraint("bench_version > 0", name="benchmark_dataset_version_positive"),
        CheckConstraint("length(sha256) = 64", name="benchmark_dataset_sha_length"),
    )


class BenchmarkRollout(Base):
    """Durable benchmark transition snapshot; there is at most one open row."""

    __tablename__ = "benchmark_rollouts"

    rollout_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), primary_key=True)
    from_version: Mapped[int] = mapped_column(Integer, nullable=False)
    desired_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    cohort_size: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    blocked_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    activated_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    __table_args__ = (
        CheckConstraint("from_version > 0", name="benchmark_rollout_from_positive"),
        CheckConstraint(
            "desired_version > from_version", name="benchmark_rollout_forward"
        ),
        CheckConstraint(
            # Retain the historical storage bound for rollout snapshots created
            # before new transitions were narrowed to ten members.
            "cohort_size BETWEEN 5 AND 25",
            name="benchmark_rollout_bounded_members",
        ),
        CheckConstraint(
            # 'superseded' is terminal: an operator abandoned the rollout before
            # activation. The partial open index below excludes it, so it frees
            # the single open slot.
            "status IN ('collecting', 'blocked_ineligible', 'activated', 'superseded')",
            name="benchmark_rollout_status",
        ),
        Index(
            "benchmark_rollouts_one_open_idx",
            text("(1)"),
            unique=True,
            postgresql_where=text("status IN ('collecting', 'blocked_ineligible')"),
            sqlite_where=text("status IN ('collecting', 'blocked_ineligible')"),
        ),
        Index(
            "benchmark_rollouts_transition_idx",
            "from_version",
            "desired_version",
            unique=True,
        ),
    )


class BenchmarkRolloutMember(Base):
    """An agent qualified during a rolling benchmark activation."""

    __tablename__ = "benchmark_rollout_members"

    rollout_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    agent_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    frozen_miner_hotkey: Mapped[str] = mapped_column(Text, nullable=False)
    frozen_composite: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("rollout_id", "agent_id"),
        UniqueConstraint("rollout_id", "position"),
        ForeignKeyConstraint(
            ["rollout_id"], ["benchmark_rollouts.rollout_id"], ondelete="CASCADE"
        ),
        ForeignKeyConstraint(["agent_id"], ["agents.agent_id"], ondelete="RESTRICT"),
        CheckConstraint("position > 0", name="benchmark_member_position"),
    )


class BenchmarkRolloutAudit(Base):
    """Append-only operator/public-safe history for benchmark transitions."""

    __tablename__ = "benchmark_rollout_audit"

    audit_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), primary_key=True)
    rollout_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    event: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict] = mapped_column(_JSON_VARIANT, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["rollout_id"], ["benchmark_rollouts.rollout_id"], ondelete="CASCADE"
        ),
        Index("benchmark_rollout_audit_history_idx", "rollout_id", "recorded_at"),
    )


class ValidatorHeartbeat(Base):
    """Latest signed build and runtime heartbeat for one validator hotkey."""

    __tablename__ = "validator_heartbeats"

    validator_hotkey: Mapped[str] = mapped_column(Text, primary_key=True)
    software_version: Mapped[str] = mapped_column(Text, nullable=False)
    protocol_version: Mapped[int] = mapped_column(Integer, nullable=False)
    code_digest: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(Text, nullable=False)
    active_agent_id: Mapped[UUID | None] = mapped_column(
        SaUUID(as_uuid=True), nullable=True
    )
    first_seen_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    system_metrics: Mapped[dict | None] = mapped_column(_JSON_VARIANT, nullable=True)
    benchmark_progress: Mapped[dict | None] = mapped_column(
        _JSON_VARIANT, nullable=True
    )
    benchmark_progress_reported: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    benchmark_progress_agent_id: Mapped[UUID | None] = mapped_column(
        SaUUID(as_uuid=True), nullable=True
    )
    capabilities: Mapped[dict | None] = mapped_column(_JSON_VARIANT, nullable=True)
    stack: Mapped[dict | None] = mapped_column(_JSON_VARIANT, nullable=True)
    stack_health: Mapped[dict | None] = mapped_column(_JSON_VARIANT, nullable=True)
    benchmark_capacity: Mapped[dict | None] = mapped_column(
        _JSON_VARIANT, nullable=True
    )
    reported_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    seen_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    signature: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        CheckConstraint(
            "length(software_version) BETWEEN 1 AND 64",
            name="validator_heartbeats_software_version_length_check",
        ),
        CheckConstraint(
            "protocol_version > 0",
            name="validator_heartbeats_protocol_version_check",
        ),
        CheckConstraint(
            "length(code_digest) = 64",
            name="validator_heartbeats_code_digest_length_check",
        ),
        CheckConstraint(
            "state IN ('polling', 'running_benchmark', 'updating_weights', "
            "'idle', 'error', 'paused')",
            name="validator_heartbeats_state_check",
        ),
        CheckConstraint(
            "length(signature) = 128",
            name="validator_heartbeats_signature_length_check",
        ),
        ForeignKeyConstraint(
            ["active_agent_id"],
            ["agents.agent_id"],
            ondelete="SET NULL",
            name="validator_heartbeats_active_agent_id_fkey",
        ),
        ForeignKeyConstraint(
            ["benchmark_progress_agent_id"],
            ["agents.agent_id"],
            ondelete="SET NULL",
            name="validator_heartbeats_benchmark_progress_agent_id_fkey",
        ),
        Index("validator_heartbeats_seen_at_idx", "seen_at"),
        Index(
            "validator_heartbeats_active_agent_idx",
            "active_agent_id",
            postgresql_where=text("active_agent_id IS NOT NULL"),
        ),
    )


class ScreenerHeartbeat(Base):
    """Latest signed runtime and host-health report for one screener instance.

    Keyed by (screener_hotkey, instance_id): the prod fleet shares one hotkey,
    so instance_id (the worker's GCE instance name) is what keeps each worker a
    distinct row instead of collapsing the fleet into one. Pre-v3 workers that
    send no instance_id are stored under the ``"legacy"`` sentinel.
    """

    __tablename__ = "screener_heartbeats"

    screener_hotkey: Mapped[str] = mapped_column(Text, primary_key=True)
    instance_id: Mapped[str] = mapped_column(
        Text, primary_key=True, server_default="legacy"
    )
    software_version: Mapped[str] = mapped_column(Text, nullable=False)
    protocol_version: Mapped[int] = mapped_column(Integer, nullable=False)
    policy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[str] = mapped_column(Text, nullable=False)
    active_agent_id: Mapped[UUID | None] = mapped_column(
        SaUUID(as_uuid=True), nullable=True
    )
    first_seen_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    system_metrics: Mapped[dict | None] = mapped_column(_JSON_VARIANT, nullable=True)
    reported_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    seen_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    signature: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        CheckConstraint(
            "length(software_version) BETWEEN 1 AND 64",
            name="screener_heartbeats_software_version_length_check",
        ),
        CheckConstraint(
            "length(instance_id) BETWEEN 1 AND 63",
            name="screener_heartbeats_instance_id_length_check",
        ),
        CheckConstraint(
            "protocol_version > 0",
            name="screener_heartbeats_protocol_version_check",
        ),
        CheckConstraint(
            "policy_version > 0",
            name="screener_heartbeats_policy_version_check",
        ),
        CheckConstraint(
            "state IN ('polling', 'screening', 'error', 'paused')",
            name="screener_heartbeats_state_check",
        ),
        CheckConstraint(
            "length(signature) = 128",
            name="screener_heartbeats_signature_length_check",
        ),
        ForeignKeyConstraint(
            ["active_agent_id"],
            ["agents.agent_id"],
            ondelete="SET NULL",
            name="screener_heartbeats_active_agent_id_fkey",
        ),
        Index("screener_heartbeats_seen_at_idx", "seen_at"),
        Index(
            "screener_heartbeats_active_agent_idx",
            "active_agent_id",
            postgresql_where=text("active_agent_id IS NOT NULL"),
        ),
    )


class ValidatorTicket(Base):
    """One validator's evaluation ticket for one agent (a k=3 scoring grant).

    A submission is scored by exactly three validators. The platform issues at
    most three tickets per agent, each to a *distinct* validator hotkey, and
    refuses further requests ("no job for you"). A ticket is the right to score:
    the validator loads the agent + the platform-generated dataset, scores it,
    and must post the signed score back before ``deadline`` or the ticket
    expires and its slot re-opens for another validator.

    The composite primary key ``(agent_id, validator_hotkey)`` enforces
    distinctness — a validator can hold at most one ticket per agent, so it can
    never occupy two of the three slots and skew the median. ``agent_id`` is a
    single-column FK to ``agents.agent_id`` with ``ON DELETE CASCADE`` because a
    ticket is derived from a live submission.

    The three composites themselves live in :class:`Score` (one row per
    ``(agent, validator)``); a ticket tracks only issuance, the deadline, and
    lifecycle. An agent finalizes (median-of-three) once it has three
    ``scored`` tickets.
    """

    __tablename__ = "validator_tickets"

    agent_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    """FK to ``agents.agent_id``. PK part 1."""

    validator_hotkey: Mapped[str] = mapped_column(Text, nullable=False)
    """SS58 hotkey of the ticket-holding validator. PK part 2 (distinctness)."""

    slot_id: Mapped[str] = mapped_column(
        Text, nullable=False, default="slot-0", server_default="slot-0"
    )
    """Heartbeat-v10 execution slot holding this live lease."""

    status: Mapped[TicketStatus] = mapped_column(
        Enum(
            TicketStatus,
            name="ticketstatus",
            values_callable=lambda enum_cls: [m.value for m in enum_cls],
            create_constraint=True,
        ),
        nullable=False,
        server_default=text("'issued'"),
    )
    """Current state: ``issued`` -> ``scored`` | ``expired``."""

    issued_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    """When the ticket was granted (UTC)."""

    deadline: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    """When an unscored ticket expires and its slot re-opens (UTC)."""

    bench_version: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    """Benchmark version whose retry budget this ticket consumes."""

    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    """Number of leases issued to this validator for this agent/version."""

    manual_retry_grants: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    """Audited operator grants that each allow one additional lease after the
    automatic same-version retry budget is exhausted. Grants never reduce or
    rewrite :attr:`attempt_count`; the append-only recovery row records why the
    extra eligibility was created."""

    infra_retry_grants: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    """Automatic cap extensions earned when this lease failed on validator-side
    infrastructure (a signed ``fail_job`` with reason ``infrastructure``) rather
    than the agent. Each one offsets the :attr:`attempt_count` increment the
    reissue will add, so an infrastructure outage never spends the agent's
    genuine attempt budget. Like :attr:`manual_retry_grants` it only raises the
    cap; it never rewrites :attr:`attempt_count`."""

    retry_after: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    """Earliest time this validator may retry an expired ticket."""

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
    """When this row was last updated (UTC)."""

    __table_args__ = (
        PrimaryKeyConstraint(
            "agent_id",
            "bench_version",
            "validator_hotkey",
            name="validator_tickets_pkey",
        ),
        ForeignKeyConstraint(
            ["agent_id"],
            ["agents.agent_id"],
            ondelete="CASCADE",
            name="validator_tickets_agent_id_fkey",
        ),
        Index("validator_tickets_agent_id_idx", "agent_id"),
        CheckConstraint(
            "bench_version > 0",
            name="validator_tickets_bench_version_positive",
        ),
        CheckConstraint(
            "attempt_count > 0",
            name="validator_tickets_attempt_count_positive",
        ),
        CheckConstraint(
            "manual_retry_grants >= 0",
            name="validator_tickets_manual_retry_grants_nonnegative",
        ),
        CheckConstraint(
            "infra_retry_grants >= 0",
            name="validator_tickets_infra_retry_grants_nonnegative",
        ),
        CheckConstraint(
            "slot_id IN ('slot-0', 'slot-1', 'slot-2', 'slot-3', "
            "'slot-4', 'slot-5', 'slot-6', 'slot-7')",
            name="validator_tickets_slot_id",
        ),
        # The expiry sweep and the live-slot count both scan open tickets only;
        # a partial index keeps those hot paths off the full table.
        Index(
            "validator_tickets_open_idx",
            "deadline",
            postgresql_where=text("status = 'issued'"),
        ),
        Index(
            "validator_tickets_one_issued_per_validator_slot_idx",
            "validator_hotkey",
            "slot_id",
            unique=True,
            postgresql_where=text("status = 'issued'"),
            sqlite_where=text("status = 'issued'"),
        ),
    )


class InferenceGrant(Base):
    """One ticket-scoped platform inference capability."""

    __tablename__ = "inference_grants"

    grant_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), primary_key=True)
    agent_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    bench_version: Mapped[int] = mapped_column(Integer, nullable=False)
    validator_hotkey: Mapped[str] = mapped_column(Text, nullable=False)
    slot_id: Mapped[str] = mapped_column(Text, nullable=False)
    ticket_deadline: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    bearer_digest: Mapped[str | None] = mapped_column(Text, nullable=True)
    broker_public_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    allowed_models: Mapped[list[str]] = mapped_column(_JSON_VARIANT, nullable=False)
    request_budget: Mapped[int] = mapped_column(Integer, nullable=False)
    token_budget: Mapped[int] = mapped_column(BigInteger, nullable=False)
    request_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    prompt_tokens: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    completion_tokens: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    cost_microusd: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    active_requests: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    expires_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["agent_id", "bench_version", "validator_hotkey"],
            [
                "validator_tickets.agent_id",
                "validator_tickets.bench_version",
                "validator_tickets.validator_hotkey",
            ],
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "agent_id",
            "bench_version",
            "validator_hotkey",
            "ticket_deadline",
            name="inference_grants_ticket_lease",
        ),
        CheckConstraint(
            "status IN ('pending', 'active', 'revoked', 'exhausted')",
            name="inference_grants_status",
        ),
        CheckConstraint("request_budget > 0", name="inference_grants_request_budget"),
        CheckConstraint("token_budget > 0", name="inference_grants_token_budget"),
        CheckConstraint(
            "active_requests >= 0", name="inference_grants_active_requests"
        ),
        Index("inference_grants_expiry_idx", "expires_at"),
    )


class InferenceRequest(Base):
    """Replay ledger and bounded accounting for one proxy request."""

    __tablename__ = "inference_requests"

    grant_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    nonce: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="started")
    model: Mapped[str] = mapped_column(Text, nullable=False)
    reserved_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0
    )
    cost_microusd: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    started_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    __table_args__ = (
        PrimaryKeyConstraint("grant_id", "nonce"),
        ForeignKeyConstraint(
            ["grant_id"], ["inference_grants.grant_id"], ondelete="CASCADE"
        ),
        CheckConstraint(
            "status IN ('started', 'completed', 'failed', 'canceled')",
            name="inference_requests_status",
        ),
        CheckConstraint(
            "reserved_tokens > 0", name="inference_requests_reserved_tokens"
        ),
        CheckConstraint("generation > 0", name="inference_requests_generation"),
        Index("inference_requests_started_idx", "started_at"),
    )


class ValidatorRetryRecovery(Base):
    """One immutable operator action that restores bounded validation eligibility.

    A recovery snapshots every ticket before changing only the selected expired
    rows' retry grant counters. It is separate from the public score audit log:
    no score or miner verdict is being created, replaced, or removed.
    """

    __tablename__ = "validator_retry_recoveries"

    recovery_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), primary_key=True)
    agent_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    actor: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    expected_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    score_count: Mapped[int] = mapped_column(Integer, nullable=False)
    bench_version: Mapped[int] = mapped_column(Integer, nullable=False)
    ticket_snapshot: Mapped[list[dict]] = mapped_column(_JSON_VARIANT, nullable=False)
    granted_validator_hotkeys: Mapped[list[str]] = mapped_column(
        _JSON_VARIANT, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["agent_id"],
            ["agents.agent_id"],
            ondelete="RESTRICT",
            name="validator_retry_recoveries_agent_id_fkey",
        ),
        CheckConstraint(
            "length(trim(actor)) BETWEEN 1 AND 120",
            name="validator_retry_recoveries_actor_length",
        ),
        CheckConstraint(
            "length(trim(reason)) BETWEEN 3 AND 500",
            name="validator_retry_recoveries_reason_length",
        ),
        CheckConstraint(
            "score_count >= 0",
            name="validator_retry_recoveries_score_count_nonnegative",
        ),
        CheckConstraint(
            "bench_version > 0",
            name="validator_retry_recoveries_bench_version_positive",
        ),
        Index(
            "validator_retry_recoveries_agent_created_idx",
            "agent_id",
            "bench_version",
            "created_at",
            "recovery_id",
        ),
        UniqueConstraint(
            "agent_id",
            "bench_version",
            "expected_snapshot",
            name="validator_retry_recoveries_agent_snapshot_key",
        ),
    )


class ValidatorRequestNonce(Base):
    """A recently consumed signed validator request nonce.

    Rows are short-lived replay guards. The UUID primary key makes consuming a
    nonce atomic across every platform replica; expired rows are pruned during
    subsequent claims.
    """

    __tablename__ = "validator_request_nonces"

    nonce: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), primary_key=True)
    validator_hotkey: Mapped[str] = mapped_column(Text, nullable=False)
    used_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )

    __table_args__ = (Index("validator_request_nonces_expires_at_idx", "expires_at"),)


class BannedHotkey(Base):
    """One row of the ``banned_hotkeys`` table.

    A hotkey-level ban, distinct from the per-agent :attr:`AgentStatus.BANNED`
    status: it blocks the *miner* (all future uploads) rather than a single
    submission. Enforced at upload (``/upload/agent`` rejects a banned hotkey
    before any expensive chain/payment work) and surfaced on the read path
    (``/retrieval/agent-by-hotkey``). Populated out-of-band by the owner via
    ``scripts/ban_hotkey.py`` — there is no public write surface.
    """

    __tablename__ = "banned_hotkeys"

    hotkey: Mapped[str] = mapped_column(Text, primary_key=True)
    """SS58 hotkey of the banned miner. Primary key (a hotkey is banned once)."""

    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    """Free-text audit note (e.g. "confirmed copy of agent <id>"). Nullable."""

    banned_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    """When the ban was recorded (UTC)."""


class ScoreAuditEntry(Base):
    """One append-only, hash-chained entry in the public score audit log.

    Every scoring *event* (a validator recording a score, and an agent
    finalizing at the k=3 median) appends one immutable row here, in the same
    transaction as the score write. Unlike ``scores`` (which is UPSERTed per
    ``(agent, validator)`` and so reflects only the *current* score), this table
    is never updated or deleted — it is the durable, ordered history.

    Tamper-evidence is a hash chain: ``entry_hash`` = SHA-256 over the entry's
    canonical JSON (which embeds ``prev_hash``), and ``prev_hash`` is the
    previous entry's ``entry_hash`` (genesis = 64 zeros). Editing or removing any
    historical entry breaks every subsequent link, so a public consumer that
    replays the chain can prove nothing was silently rewritten. Each ``score``
    entry also carries the validator's sr25519 ``signature`` verbatim, so an
    entry is independently authenticatable against the published validator key —
    the log adds ordering + immutability on top of the already-signed payload.
    """

    __tablename__ = "score_audit_log"

    seq: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    """Monotonic append order (BIGSERIAL on Postgres, INTEGER rowid on SQLite)."""

    agent_id: Mapped[UUID] = mapped_column(SaUUID(as_uuid=True), nullable=False)
    """Agent the event is about. Not FK-bound — the log outlives the agent row
    (agents may be pruned; the audit history must not cascade away)."""

    validator_hotkey: Mapped[str | None] = mapped_column(Text, nullable=True)
    """Reporting validator for a ``score`` event; null for an ``agent_finalized``
    event (which is the platform's own median computation, not one validator's)."""

    event: Mapped[str] = mapped_column(Text, nullable=False)
    """Event kind: ``score`` (one validator's signed score) or ``agent_finalized``
    (quorum reached; the median + participating validators)."""

    payload: Mapped[dict] = mapped_column(_JSON_VARIANT, nullable=False)
    """The event's immutable content, JSON. For ``score``: the full signed tuple
    (run_id, seed, composite, tool/memory means, median_ms, n, signature,
    generated_at). For ``agent_finalized``: median_composite, quorum, the
    scoring validators, and the pinned dataset. Hashed into ``entry_hash``."""

    prev_hash: Mapped[str] = mapped_column(Text, nullable=False)
    """The previous entry's ``entry_hash`` (hex); ``"0" * 64`` for the genesis."""

    entry_hash: Mapped[str] = mapped_column(Text, nullable=False)
    """SHA-256 (hex) over this entry's canonical JSON, which embeds ``prev_hash``.
    The chain link a public verifier recomputes to detect tampering."""

    recorded_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    """When the platform appended the entry (UTC). Part of the hashed content."""

    __table_args__ = (
        UniqueConstraint("entry_hash", name="score_audit_log_entry_hash_key"),
        Index("score_audit_log_agent_id_idx", "agent_id"),
    )
