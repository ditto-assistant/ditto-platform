"""Admin (Backroom) surface for anti-copy ``ath_pending_review`` holds.

The anti-copy gate (:mod:`ditto.api_server.scoring_gate`) parks a suspicious
high-scorer in ``ath_pending_review``; until now the only exit was the
owner-run ``scripts/resolve_review.py``. These models back the operator
console's review queue: each held agent is listed with its ORIGINAL stored
hold (reason + attribution, exactly as recorded at hold time) side by side
with a freshly RECOMPUTED gate decision against the current eligible ledger,
so a hold created by a since-fixed gate (e.g. pre-reference-aware
fingerprints) is visibly stale. Resolution stays an individual, audited,
per-agent action — release restores ``scored``; ban confirms the copy — the
console merely makes issuing those individual actions fast.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AdminCopyReviewRecomputed(BaseModel):
    """A fresh, side-effect-free gate run for a held agent.

    ``held`` false means the current gate no longer finds copy evidence for
    this agent (``would_release``): the hold predates a gate fix or the
    matched agent left the ledger. ``reason``/``duplicate_of`` describe the
    NEW decision when still held; they may differ from the stored ones.
    """

    held: bool
    duplicate_of: UUID | None = None
    reason: str | None = None


class AdminCopyReviewPairSimilarity(BaseModel):
    """Current per-channel similarity against the ORIGINALLY matched agent.

    Operator context for judging the stored attribution: values are computed
    from the fingerprints as stored today (post-backfill they are
    reference-aware), so a pair whose lexical numbers collapsed after
    reference subtraction reads as scaffolding convergence, not copying.
    ``None`` when the originally matched agent no longer exists or either
    side lacks that channel's sketch.
    """

    lexical_jaccard: float | None = None
    lexical_containment: float | None = None
    structural_jaccard: float | None = None
    structural_containment: float | None = None
    prompt_jaccard: float | None = None
    prompt_containment: float | None = None


class AdminCopyReviewItem(BaseModel):
    agent_id: UUID
    miner_hotkey: str
    agent_name: str
    agent_version: int | None = None
    sha256: str
    size_bytes: int | None
    submitted_at: datetime
    median_composite: float | None
    """Median of the quorum scores the hold was finalized on (``None`` only
    for a hold that somehow has no scores; such rows are never auto-cleared)."""

    stored_duplicate_of: UUID | None
    stored_reason: str | None
    recomputed: AdminCopyReviewRecomputed
    would_release: bool
    """``recomputed.held is False`` — the current gate clears this agent."""

    pair_similarity: AdminCopyReviewPairSimilarity | None


class AdminCopyReviewList(BaseModel):
    items: list[AdminCopyReviewItem]
    count: int
    would_release_count: int


class AdminCopyReviewResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolution: Literal["release", "ban"]
    reason: Annotated[str, Field(min_length=3, max_length=500)]


class AdminCopyReviewResolveResponse(BaseModel):
    agent_id: UUID
    agent_status: str
    resolution: Literal["release", "ban"]
