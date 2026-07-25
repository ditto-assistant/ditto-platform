"""Hot-swappable operator policy for continual top-five retests."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class ContinualRetestSettings(BaseModel):
    """Complete subnet-global continual-retest policy stored per revision."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    aggregate_mode: Literal["disabled", "fleet_ready", "enabled"] = "fleet_ready"
    idle_retests_enabled: bool = False


class ContinualRetestSettingsRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int
    parent_revision: int
    scope: str
    settings: ContinualRetestSettings
    reason: str
    actor: str
    created_at: datetime
    checksum: str


class EffectiveContinualRetestSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int
    scope: str
    settings: ContinualRetestSettings
    checksum: str
    source: Literal["revision", "default"]
    fleet_protocol_ready: bool
    aggregate_active: bool
    max_age_seconds: float


class AdminContinualRetestSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scope: str = "*"
    expected_revision: Annotated[int, Field(ge=0)]
    settings: ContinualRetestSettings
    reason: Annotated[str, Field(min_length=8, max_length=500)]
    actor: Annotated[str, Field(min_length=1, max_length=120)] = "admin_api"
    confirmation: str


class AdminContinualRetestSettingsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    current: list[ContinualRetestSettingsRevision]
    history: list[ContinualRetestSettingsRevision]
    default: ContinualRetestSettings
    effective: EffectiveContinualRetestSettings
