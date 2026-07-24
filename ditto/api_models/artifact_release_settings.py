"""Audited operator settings for public source release timing."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class ArtifactReleaseSettingsRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int
    parent_revision: int
    embargo_hours: int
    reason: str
    actor: str
    created_at: datetime | None


class AdminArtifactReleaseSettingsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    current: ArtifactReleaseSettingsRevision
    history: list[ArtifactReleaseSettingsRevision]


class AdminArtifactReleaseSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    expected_revision: Annotated[int, Field(ge=0)]
    embargo_hours: Annotated[int, Field(ge=6, le=24)]
    reason: Annotated[str, Field(min_length=8, max_length=500)]
    actor: Annotated[str, Field(min_length=1, max_length=120)] = "admin_api"
    confirmation: str
