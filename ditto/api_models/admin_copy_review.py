"""Admin contracts for durable ATH copy-review records."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StringConstraints


class AdminCopyReviewEvidence(BaseModel):
    duplicate_of: UUID | None
    reason: str | None
    policy_version: int
    fingerprint_versions: dict[str, int | str | None]
    reference_provenance: str
    backfilled: bool = False
    # Identity of the originally matched agent, so operators see WHICH
    # submission triggered the hold instead of a bare UUID. Null when the
    # matched agent row no longer exists.
    duplicate_of_name: str | None = None
    duplicate_of_version: int | None = None
    duplicate_of_hotkey: str | None = None
    duplicate_of_submitted_at: datetime | None = None


class AdminCopySimilarityEvidence(BaseModel):
    candidate_version: int | str | None
    reference_version: int | str | None
    compatible: bool
    applicable: bool
    candidate_cardinality: int | None
    reference_cardinality: int | None
    jaccard: float | None
    containment: float | None
    above_threshold: bool
    decision_role: str


class AdminCopyReviewCurrentComparison(BaseModel):
    availability: Literal["available"]
    bulk_eligible: bool
    algorithm_version: str
    lexical_fingerprint_version: int
    normalized_source_fingerprint_version: str
    prompt_fingerprint_version: str
    canonical_reference_revision: str
    reference_corpus_id: str
    reference_exclusion_mode: str
    miner_exclusion_mode: str
    same_miner_excluded: bool
    chronology_direction: str
    chronology_eligible: bool
    exact_byte_match: bool
    normalized_source_match: bool
    lexical: AdminCopySimilarityEvidence
    structural: AdminCopySimilarityEvidence
    prompt: AdminCopySimilarityEvidence
    triggered: bool
    triggered_signal: str | None
    current_decision: str


class AdminCopyReviewItem(BaseModel):
    review_id: UUID
    agent_id: UUID
    miner_hotkey: str
    agent_name: str
    agent_version: int | None = None
    submitted_at: datetime
    status: Literal["pending", "resolved"]
    opened_at: datetime
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    resolution: Literal["clear", "reject"] | None = None
    resolution_reason: str | None = None
    original: AdminCopyReviewEvidence


class AdminCopyReviewList(BaseModel):
    items: list[AdminCopyReviewItem]
    count: int
    limit: int
    offset: int


class AdminCopyReviewResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # release/ban remain accepted for Backroom #20 wire compatibility.
    resolution: Literal["clear", "reject", "release", "ban"]
    reason: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)
    ]


class AdminCopyReviewResolveResponse(BaseModel):
    review: AdminCopyReviewItem
    agent_status: str
    idempotent: bool
