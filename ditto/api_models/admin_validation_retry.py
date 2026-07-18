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
