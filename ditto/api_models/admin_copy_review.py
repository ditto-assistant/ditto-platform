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


class AdminCopyReviewCurrentComparison(BaseModel):
    label: Literal["current_comparison"] = "current_comparison"
    availability: Literal["unavailable"] = "unavailable"
    bulk_eligible: Literal[False] = False
    reason: str
    algorithm_provenance: dict[str, str | int | bool | None]


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
