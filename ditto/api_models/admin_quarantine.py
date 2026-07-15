"""Private Backroom/operator models for screening quarantine management."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

QuarantineResolution = Literal["release", "rescreen", "reject"]


class AdminQuarantineItem(BaseModel):
    quarantine_id: UUID
    agent_id: UUID
    attempt_id: UUID
    miner_hotkey: str
    agent_name: str
    artifact_sha256: str
    policy_version: int
    manifest_digest: str
    finding_digest: str | None
    reason_code: str
    status: Literal["active", "resolved"]
    created_at: datetime
    resolved_at: datetime | None
    resolved_by: str | None
    resolution: QuarantineResolution | None
    resolution_reason: str | None


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


class AdminScreeningSubmission(BaseModel):
    agent_id: UUID
    miner_hotkey: str
    agent_name: str
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


__all__ = [
    "AdminQuarantineItem",
    "AdminQuarantineList",
    "AdminQuarantineResolveRequest",
    "AdminQuarantineResolveResponse",
    "AdminScreeningAttempt",
    "AdminScreeningSubmission",
    "AdminScreeningSubmissionList",
    "AdminScreeningRescreenRequest",
    "AdminScreeningRescreenResponse",
]
