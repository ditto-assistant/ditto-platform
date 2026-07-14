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


__all__ = [
    "AdminQuarantineItem",
    "AdminQuarantineList",
    "AdminQuarantineResolveRequest",
    "AdminQuarantineResolveResponse",
]
