"""Private Backroom/operator models for screening quarantine management."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ditto.api_models.screener import ScreenEvidenceItem, SourceReviewFinding

QuarantineResolution = Literal["release", "rescreen", "reject"]


class AdminQuarantineResolutionEvent(BaseModel):
    resolution: QuarantineResolution
    reason: str
    actor: str
    created_at: datetime


class AdminQuarantineItem(BaseModel):
    quarantine_id: UUID
    agent_id: UUID
    attempt_id: UUID
    miner_hotkey: str
    agent_name: str
    agent_version: int | None = None
    artifact_sha256: str
    policy_version: int
    manifest_digest: str
    finding_digest: str | None
    reason_code: str
    evidence: list[ScreenEvidenceItem] | None
    finding: SourceReviewFinding | None
    finding_verified: bool
    """True iff ``finding`` is present and its canonical digest equals the
    ``finding_digest`` bound into the screener's signed verdict."""

    status: Literal["active", "resolved"]
    created_at: datetime
    resolved_at: datetime | None
    resolved_by: str | None
    resolution: QuarantineResolution | None
    resolution_reason: str | None
    resolution_history: list[AdminQuarantineResolutionEvent] = Field(
        default_factory=list
    )


class AdminQuarantineList(BaseModel):
    items: list[AdminQuarantineItem]
    count: int


class AdminQuarantineResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolution: QuarantineResolution
    reason: Annotated[str, Field(min_length=3, max_length=500)]


class AdminQuarantineResolveResponse(BaseModel):
    quarantine: AdminQuarantineItem
    agent_status: str


class AdminScreeningAttempt(BaseModel):
    attempt_id: UUID
    policy_version: int
    status: Literal["running", "passed", "rejected", "failed", "expired", "quarantined"]
    screener_hotkey: str
    started_at: datetime
    deadline: datetime
    finished_at: datetime | None
    reason: str | None
    reason_code: str | None
    duplicate_of: UUID | None
    duplicate_name: str | None = None
    duplicate_version: int | None = None


class AdminScreeningSubmission(BaseModel):
    agent_id: UUID
    miner_hotkey: str
    agent_name: str
    agent_version: int | None = None
    artifact_sha256: str
    agent_status: str
    screening_policy_version: int
    screening_reason: str | None
    screening_reason_code: str | None
    submitted_at: datetime
    attempts: list[AdminScreeningAttempt]


class AdminScreeningSubmissionList(BaseModel):
    items: list[AdminScreeningSubmission]
    count: int


class AdminScreeningRescreenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: Annotated[str, Field(min_length=3, max_length=500)]
    expected_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    expected_score_count: Annotated[int, Field(ge=0)]


class AdminScreeningRescreenResponse(BaseModel):
    agent_id: UUID
    agent_status: str


class AdminQuarantineAgentContext(BaseModel):
    """Submission metadata an operator needs while judging a quarantine."""

    agent_id: UUID
    miner_hotkey: str
    agent_name: str
    artifact_sha256: str
    agent_status: str
    size_bytes: int | None
    submitted_at: datetime
    screening_policy_version: int
    screening_reason: str | None


class AdminMinerQuarantineSummary(BaseModel):
    """One prior quarantine from the same miner, with its resolution."""

    quarantine_id: UUID
    agent_id: UUID
    agent_name: str
    reason_code: str
    status: Literal["active", "resolved"]
    resolution: QuarantineResolution | None
    resolution_reason: str | None
    created_at: datetime
    resolved_at: datetime | None


class AdminMinerContext(BaseModel):
    """The submitting miner's track record across all submissions."""

    miner_hotkey: str
    total_submissions: int
    quarantine_count: int
    released_count: int
    rescreened_count: int
    rejected_count: int
    recent_quarantines: list[AdminMinerQuarantineSummary]


class AdminArtifactDuplicate(BaseModel):
    """Another submission whose artifact matches this one."""

    agent_id: UUID
    miner_hotkey: str
    agent_name: str
    agent_status: str
    submitted_at: datetime
    match: Literal["identical_artifact", "identical_normalized_source"]


class AdminDuplicateSummary(BaseModel):
    """Authoritative duplicate counts, independent of the bounded sample."""

    total: int
    cross_miner: int
    same_miner: int
    sample_truncated: bool


class AdminQuarantineContext(BaseModel):
    """Everything the review console shows for one quarantine decision."""

    quarantine: AdminQuarantineItem
    agent: AdminQuarantineAgentContext
    attempts: list[AdminScreeningAttempt]
    miner: AdminMinerContext
    duplicates: list[AdminArtifactDuplicate]
    """A bounded sample (at most 20); use ``duplicate_summary`` for counts."""

    duplicate_summary: AdminDuplicateSummary


class AdminSourceFileEntry(BaseModel):
    path: str
    bytes: int


class AdminOpaqueBlobEntry(BaseModel):
    """A member the text reader cannot show; a natural hiding place."""

    path: str
    bytes: int
    reason: Literal["oversized", "non_utf8"]


class AdminSourceListing(BaseModel):
    agent_id: UUID
    artifact_sha256: str
    file_count: int
    files: list[AdminSourceFileEntry]
    opaque_blobs: list[AdminOpaqueBlobEntry]
    opaque_total: int
    """Total unreadable members found; ``opaque_blobs`` shows at most 128."""

    truncated: bool


class AdminSourceLine(BaseModel):
    line: int
    text: str


class AdminSourceExcerpt(BaseModel):
    agent_id: UUID
    path: str
    total_lines: int
    start_line: int
    end_line: int
    lines: list[AdminSourceLine]


__all__ = [
    "AdminArtifactDuplicate",
    "AdminDuplicateSummary",
    "AdminMinerContext",
    "AdminMinerQuarantineSummary",
    "AdminOpaqueBlobEntry",
    "AdminQuarantineAgentContext",
    "AdminQuarantineContext",
    "AdminQuarantineItem",
    "AdminQuarantineList",
    "AdminQuarantineResolutionEvent",
    "AdminQuarantineResolveRequest",
    "AdminQuarantineResolveResponse",
    "AdminScreeningAttempt",
    "AdminScreeningSubmission",
    "AdminScreeningSubmissionList",
    "AdminScreeningRescreenRequest",
    "AdminScreeningRescreenResponse",
    "AdminSourceExcerpt",
    "AdminSourceFileEntry",
    "AdminSourceLine",
    "AdminSourceListing",
]
