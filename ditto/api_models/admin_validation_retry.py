"""Private operator contract for bounded validator-infrastructure retries."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class AdminValidationTicket(BaseModel):
    validator_hotkey: str
    status: Literal["issued", "scored", "expired"]
    issued_at: datetime
    deadline: datetime
    bench_version: int
    attempt_count: int
    manual_retry_grants: int
    retry_after: datetime | None
    retry_budget_exhausted: bool


class AdminValidationRecovery(BaseModel):
    recovery_id: UUID
    agent_id: UUID
    actor: str
    reason: str
    score_count: int
    bench_version: int
    expected_snapshot: str
    granted_validator_hotkeys: list[str]
    created_at: datetime


class AdminValidationRetryDetail(BaseModel):
    agent_id: UUID
    miner_hotkey: str
    agent_name: str
    agent_version: int | None
    agent_status: str
    score_count: int
    quorum: int
    snapshot: str
    automatic_retry_available: bool
    recovery_allowed: bool
    blocking_reason: str | None
    tickets: list[AdminValidationTicket]
    recoveries: list[AdminValidationRecovery]


class AdminValidationRetryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    expected_snapshot: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    reason: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=3, max_length=500),
    ]


class AdminValidationRetryResponse(BaseModel):
    recovery: AdminValidationRecovery
    idempotent: bool


class AdminValidatorScoreReplacementDetail(BaseModel):
    agent_id: UUID
    validator_hotkey: str
    agent_status: str
    bench_version: int
    score_count: int
    quorum: int
    snapshot: str
    run_id: str | None
    composite: float | None
    ticket_status: Literal["issued", "scored", "expired"] | None
    ticket_deadline: datetime | None
    replacement_pending: bool
    replacement_request_id: UUID | None
    replacement_reason: str | None
    replacement_actor: str | None
    replacement_allowed: bool
    blocking_reason: str | None


class AdminValidatorScoreReplacementRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    expected_snapshot: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    expected_run_id: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)
    ]
    reason: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=8, max_length=500),
    ]


class AdminValidatorScoreReplacementResponse(BaseModel):
    request_id: UUID
    agent_id: UUID
    validator_hotkey: str
    original_run_id: str
    bench_version: int
    replacement_deadline: datetime
    preserved_score_count: int
    idempotent: bool


class AdminValidatorScoreRetestReleaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    expected_snapshot: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    expected_deadline: datetime
    reason: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=8, max_length=500),
    ]


class AdminValidatorScoreRetestReleaseResponse(BaseModel):
    request_id: UUID
    agent_id: UUID
    validator_hotkey: str
    status: Literal["scored"]
    preserved_run_id: str
    idempotent: bool


class AdminScoreOutlierScore(BaseModel):
    validator_hotkey: str
    run_id: str
    composite: float


class AdminScoreOutlier(BaseModel):
    agent_id: UUID
    agent_name: str
    miner_hotkey: str
    agent_status: str
    bench_version: int
    snapshot: str
    median_composite: float
    direction: Literal["high", "low"]
    outlier: AdminScoreOutlierScore
    peers: list[AdminScoreOutlierScore]
    deviation: float
    peer_spread: float
    ticket_status: Literal["issued", "scored", "expired"] | None
    replacement_pending: bool
    replacement_deadline: datetime | None
    replacement_allowed: bool
    blocking_reason: str | None


class AdminScoreOutlierList(BaseModel):
    items: list[AdminScoreOutlier]
    count: int
    limit: int
    offset: int
