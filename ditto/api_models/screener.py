"""Screener wire shapes layered over the shared verdict protocol."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ditto.api_models.system_health import SystemMetrics
from ditto_screening_protocol import (
    SCREENING_POLICY_VERSION,
    ScreenerQueueItem,
    ScreenerQueueResponse,
    ScreenResultOutcome,
    ScreenResultResponse,
)
from ditto_screening_protocol import (
    ScreenResultRequest as ProtocolScreenResultRequest,
)

_SS58_PATTERN = r"^[1-9A-HJ-NP-Za-km-z]{47,48}$"
_SIGNATURE_HEX_PATTERN = r"^[0-9a-fA-F]{128}$"
_SOFTWARE_VERSION_PATTERN = r"^[0-9A-Za-z][0-9A-Za-z._+-]{0,63}$"

ScreenerRuntimeState = Literal["polling", "screening", "error", "paused"]
ScreenerProgressStage = Literal[
    "preparing",
    "downloading",
    "validating",
    "building",
    "starting",
    "health_check",
    "source_review_0",
    "source_review_10",
    "source_review_20",
    "source_review_30",
    "source_review_40",
    "source_review_50",
    "source_review_60",
    "source_review_70",
    "source_review_80",
    "source_review_90",
    "source_review_100",
    "submitting",
]


class ScreenerProgress(BaseModel):
    """Signed, public-safe progress for one active screening job."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    stage: ScreenerProgressStage
    started_at: Annotated[int, Field(ge=0)]


class ScreenerHeartbeatRequest(BaseModel):
    """Dedicated screener identity, work, and optional host-health report."""

    model_config = ConfigDict(extra="forbid")

    screener_hotkey: Annotated[str, Field(pattern=_SS58_PATTERN)]
    software_version: Annotated[str, Field(pattern=_SOFTWARE_VERSION_PATTERN)]
    protocol_version: Annotated[int, Field(ge=1, le=2**31 - 1)]
    policy_version: Annotated[int, Field(ge=1, le=2**31 - 1)]
    state: ScreenerRuntimeState
    active_agent_id: UUID | None = None
    progress: ScreenerProgress | None = None
    system_metrics: SystemMetrics | None = None
    timestamp: Annotated[int, Field(ge=0)]
    signature: Annotated[str, Field(pattern=_SIGNATURE_HEX_PATTERN)]

    @model_validator(mode="after")
    def validate_progress(self) -> ScreenerHeartbeatRequest:
        if self.progress is None:
            return self
        if self.protocol_version < 2:
            raise ValueError("progress requires heartbeat protocol v2")
        if self.state != "screening" or self.active_agent_id is None:
            raise ValueError("progress requires active screening work")
        if self.progress.started_at > self.timestamp:
            raise ValueError("progress start cannot be after the heartbeat")
        if self.timestamp - self.progress.started_at > 6 * 60 * 60:
            raise ValueError("progress start is outside the bounded job window")
        return self


class ScreenerHeartbeatResponse(BaseModel):
    """Acknowledgement that a signed screener heartbeat was persisted."""

    accepted: bool
    seen_at: datetime


# --- quarantine review payloads -------------------------------------------
#
# These mirror ditto-screening-protocol 0.9.0. The platform still pins 0.8.0
# (a git rev of the screener repo), so the extended request is declared here
# and shadows the protocol re-export; drop the local copies once the pin
# reaches >= 0.9.0.


class ScreenEvidenceItem(BaseModel):
    """One bounded, public-safe policy evidence summary carried on a verdict."""

    model_config = ConfigDict(extra="forbid")

    module_id: Annotated[str, Field(min_length=1, max_length=64)]
    code: Annotated[str, Field(min_length=1, max_length=64)]
    summary: Annotated[str, Field(min_length=1, max_length=240)]
    digest: Annotated[str | None, Field(pattern=r"^[0-9a-f]{64}$")] = None


class SourceReviewEvidenceItem(BaseModel):
    """One flagged source location from the read-only source review."""

    model_config = ConfigDict(extra="forbid")

    path: Annotated[str, Field(min_length=1, max_length=240)]
    line: Annotated[int, Field(ge=1)]
    category: Annotated[str, Field(min_length=1, max_length=64)]


class SourceReviewFinding(BaseModel):
    """Bounded source-review finding whose canonical JSON is digest-bound."""

    model_config = ConfigDict(extra="forbid")

    artifact_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    prompt_revision: Annotated[str, Field(min_length=1, max_length=64)]
    risk_level: Literal["low", "medium", "high"]
    confidence: Annotated[float, Field(ge=0, le=1)]
    categories: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=64)]],
        Field(min_length=1, max_length=8),
    ]
    evidence: Annotated[
        list[SourceReviewEvidenceItem], Field(default_factory=list, max_length=16)
    ]
    summary: Annotated[str, Field(min_length=1, max_length=240)]

    def canonical_digest(self) -> str:
        """SHA-256 over the canonical JSON encoding of this finding.

        Must stay byte-identical to
        ``ditto_screening_protocol.SourceReviewFinding.canonical_digest`` so
        the platform verifies exactly what the screener signed.
        """
        canonical = json.dumps(
            {
                "artifact_sha256": self.artifact_sha256,
                "prompt_revision": self.prompt_revision,
                "risk_level": self.risk_level,
                "confidence": self.confidence,
                "categories": sorted(set(self.categories)),
                "evidence": [
                    {
                        "path": item.path,
                        "line": item.line,
                        "category": item.category,
                    }
                    for item in self.evidence
                ],
                "summary": self.summary,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


class ScreenResultRequest(ProtocolScreenResultRequest):
    """Verdict request extended with bounded operator-review payloads."""

    evidence: Annotated[list[ScreenEvidenceItem] | None, Field(max_length=16)] = None
    finding: SourceReviewFinding | None = None

    @model_validator(mode="after")
    def validate_review_payloads(self) -> ScreenResultRequest:
        if (self.evidence is not None or self.finding is not None) and (
            self.outcome
            not in {ScreenResultOutcome.QUARANTINE, ScreenResultOutcome.INCONCLUSIVE}
        ):
            raise ValueError("evidence and finding require a review outcome")
        if self.finding is not None:
            if self.finding_digest is None:
                raise ValueError("finding requires finding_digest")
            if self.finding.canonical_digest() != self.finding_digest:
                raise ValueError("finding does not match finding_digest")
        return self


__all__ = [
    "SCREENING_POLICY_VERSION",
    "ScreenerQueueItem",
    "ScreenerQueueResponse",
    "ScreenerHeartbeatRequest",
    "ScreenerHeartbeatResponse",
    "ScreenerProgress",
    "ScreenerProgressStage",
    "ScreenerRuntimeState",
    "ScreenEvidenceItem",
    "ScreenResultRequest",
    "ScreenResultResponse",
    "SourceReviewEvidenceItem",
    "SourceReviewFinding",
]
