"""Private operator contract for bounded validator-infrastructure retries."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from ditto.api_models.retry_state import RetryState


class AdminValidationTicket(BaseModel):
    validator_hotkey: str
    status: Literal["issued", "scored", "expired"]
    issued_at: datetime
    deadline: datetime
    bench_version: int
    attempt_count: int
    manual_retry_grants: int
    infra_retry_grants: int
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


class AdminStuckSubmission(BaseModel):
    """One below-quorum submission plus why it is (or is not) advancing.

    ``retry_state`` is the operator-facing triage label:

    * ``running`` — a validator holds a live ticket right now.
    * ``retry_available`` — an expired ticket is off cooldown and will be
      re-leased on the next sweep with budget to spare.
    * ``cooling_down`` — an expired ticket still has budget but is waiting out
      its retry cooldown.
    * ``exhausted`` — no ticket can advance without an operator grant (every
      remaining validator burned its attempt budget). This is the only state
      that needs a human.
    * ``queued`` — below quorum with slots that have simply never been leased
      yet; it will advance on its own.
    """

    agent_id: UUID
    miner_hotkey: str
    agent_name: str
    agent_version: int | None
    bench_version: int
    score_count: int
    quorum: int
    retry_state: RetryState
    automatic_retry_available: bool
    recovery_allowed: bool
    blocking_reason: str | None
    earliest_retry_after: datetime | None
    attempts_used: int
    exhausted_validator_count: int
    snapshot: str
    tickets: list[AdminValidationTicket]


class AdminStuckSubmissionsResponse(BaseModel):
    generated_at: datetime
    quorum: int
    counts: dict[RetryState, int]
    submissions: list[AdminStuckSubmission]


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


class AdminBatchRetryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: UUID
    request_id: UUID
    expected_snapshot: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class AdminBatchRetryRequest(BaseModel):
    """Grant recoveries to several stranded submissions in one operator action.

    Each item carries its own ``expected_snapshot`` and idempotency
    ``request_id`` so the batch is exactly as safe as N single-agent retries:
    an item whose state moved is skipped, never force-granted.
    """

    model_config = ConfigDict(extra="forbid")

    reason: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=3, max_length=500),
    ]
    items: Annotated[list[AdminBatchRetryItem], Field(min_length=1, max_length=100)]

    @field_validator("items")
    @classmethod
    def _unique(cls, items: list[AdminBatchRetryItem]) -> list[AdminBatchRetryItem]:
        if len({item.agent_id for item in items}) != len(items):
            raise ValueError("duplicate agent_id in batch")
        if len({item.request_id for item in items}) != len(items):
            raise ValueError("duplicate request_id in batch")
        return items


class AdminBatchRetryResult(BaseModel):
    agent_id: UUID
    status: Literal["granted", "idempotent", "skipped"]
    detail: str | None
    recovery: AdminValidationRecovery | None


class AdminBatchRetryResponse(BaseModel):
    granted: int
    results: list[AdminBatchRetryResult]


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


class AdminValidatorScoreRetestQueueItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: UUID
    request_id: UUID
    expected_snapshot: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    expected_run_id: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)
    ]


class AdminValidatorScoreRetestQueueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=8, max_length=500),
    ]
    items: Annotated[
        list[AdminValidatorScoreRetestQueueItem], Field(min_length=1, max_length=100)
    ]

    @field_validator("items")
    @classmethod
    def _unique(
        cls, items: list[AdminValidatorScoreRetestQueueItem]
    ) -> list[AdminValidatorScoreRetestQueueItem]:
        if len({item.agent_id for item in items}) != len(items):
            raise ValueError("duplicate agent_id in queue")
        if len({item.request_id for item in items}) != len(items):
            raise ValueError("duplicate request_id in queue")
        return items


class AdminValidatorScoreRetestQueueResult(BaseModel):
    agent_id: UUID
    request_id: UUID
    status: Literal["activated", "queued", "idempotent", "skipped"]
    detail: str | None
    queue_position: int | None


class AdminValidatorScoreRetestQueueResponse(BaseModel):
    validator_hotkey: str
    activated: int
    queued: int
    idempotent: int
    skipped: int
    results: list[AdminValidatorScoreRetestQueueResult]


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
    replacement_queued: bool
    queue_position: int | None
    replacement_deadline: datetime | None
    replacement_allowed: bool
    blocking_reason: str | None
    queue_allowed: bool
    queue_blocking_reason: str | None


class AdminScoreOutlierList(BaseModel):
    items: list[AdminScoreOutlier]
    count: int
    limit: int
    offset: int
