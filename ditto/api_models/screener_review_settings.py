"""Versioned operator settings for private L2/L3 source review."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ReviewMode = Literal["off", "shadow", "enforce"]
ReviewModel = Literal[
    "moonshotai/kimi-k3",
    "z-ai/glm-5.2",
    "openai/gpt-5.6-sol",
]
ReasoningEffort = Literal["low", "medium"]


class ScreenerReviewSettings(BaseModel):
    """Strict, secret-free settings applied between screening leases."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    mode: ReviewMode = "off"
    l2_model: ReviewModel = "moonshotai/kimi-k3"
    l2_fallback_models: tuple[ReviewModel, ...] = (
        "z-ai/glm-5.2",
        "openai/gpt-5.6-sol",
    )
    l3_model: Literal["openai/gpt-5.6-sol"] = "openai/gpt-5.6-sol"
    timeout_seconds: Annotated[int, Field(ge=30, le=900)] = 900
    max_steps: Annotated[int, Field(ge=1, le=20)] = 18
    max_input_tokens: Annotated[int, Field(ge=1, le=1_000_000)] = 425_000
    max_output_tokens: Annotated[int, Field(ge=1, le=128_000)] = 20_000
    max_completion_tokens: Annotated[int, Field(ge=1, le=128_000)] = 2_400
    max_cost_usd: Annotated[float, Field(gt=0, le=10)] = 2.0
    critic_reasoning_effort: ReasoningEffort = "medium"
    cache_ttl_seconds: Annotated[int, Field(ge=60, le=2_592_000)] = 604_800
    audit_retention_days: Annotated[int, Field(ge=1, le=365)] = 30

    @model_validator(mode="after")
    def validate_model_chain(self) -> ScreenerReviewSettings:
        chain = (self.l2_model, *self.l2_fallback_models)
        if len(chain) != len(set(chain)):
            raise ValueError("L2 model chain must not contain duplicates")
        if self.max_completion_tokens > self.max_output_tokens:
            raise ValueError("completion budget must not exceed output budget")
        return self


class ScreenerReviewSettingsRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int
    parent_revision: int
    scope: str
    settings: ScreenerReviewSettings
    reason: str
    actor: str
    created_at: datetime
    checksum: str


class EffectiveScreenerReviewSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int
    scope: str
    settings: ScreenerReviewSettings
    checksum: str
    max_age_seconds: int = 60


class AdminScreenerReviewSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scope: str
    expected_revision: Annotated[int, Field(ge=0)]
    settings: ScreenerReviewSettings
    reason: Annotated[str, Field(min_length=8, max_length=500)]
    actor: Annotated[str, Field(min_length=1, max_length=120)] = "admin_api"
    confirmation: str


class AdminScreenerReviewSettingsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    current: list[ScreenerReviewSettingsRevision]
    history: list[ScreenerReviewSettingsRevision]
    known_instances: list[str]
