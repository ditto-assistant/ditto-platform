"""Pydantic wire shapes shared across HTTP boundary.

Every model in this package describes the JSON payload of a request or
response served by :mod:`ditto.api_server` (and consumed by the miner
CLI + validator daemon). Pydantic lives here and nowhere else: internal
configs / value objects / results use ``@dataclass(frozen=True)`` per the
code quality standards.

Models are organised per concern in submodules; this ``__init__`` only
re-exports.

Usage:
    from ditto.api_models import HealthResponse

    payload = HealthResponse(status="ok", db="ok", chain="ok", commit="abc...")
"""

from __future__ import annotations

from ditto.api_models.benchmark_progress import BenchmarkProgress
from ditto.api_models.health import HealthResponse
from ditto.api_models.public import (
    BenchDatasetConfig,
    BenchGradingConfig,
    BenchHarnessConfig,
    PublicActivityEntry,
    PublicActivityResponse,
    PublicAuditEntry,
    PublicAuditResponse,
    PublicBenchConfigResponse,
    PublicBenchCorpusEntry,
    PublicBenchCorpusResponse,
    PublicBenchIntegrity,
    PublicBenchmarkProgress,
    PublicCaseResult,
    PublicCategoryStat,
    PublicDatasetReveal,
    PublicHealthResponse,
    PublicLeaderboardEntry,
    PublicLeaderboardResponse,
    PublicOperationsResponse,
    PublicProvisionalScore,
    PublicRunModels,
    PublicScreenerHeartbeat,
    PublicScreenerHeartbeatsResponse,
    PublicScreenerProgress,
    PublicScreeningAttempt,
    PublicSubmissionPipeline,
    PublicSubmissionScores,
    PublicSubmissionsResponse,
    PublicSubmissionSummary,
    PublicSystemMetrics,
    PublicValidationAttempt,
    PublicValidatorHeartbeat,
    PublicValidatorHeartbeatsResponse,
    PublicValidatorName,
    PublicValidatorNamesResponse,
    PublicValidatorScore,
)
from ditto.api_models.retrieval import AgentResponse, AgentStatusResponse
from ditto.api_models.screener import (
    ScreenerHeartbeatRequest,
    ScreenerHeartbeatResponse,
    ScreenerQueueItem,
    ScreenerQueueResponse,
    ScreenEvidenceItem,
    ScreenResultRequest,
    ScreenResultResponse,
    SourceReviewEvidenceItem,
    SourceReviewFinding,
)
from ditto.api_models.upload import (
    EvalPricingResponse,
    UploadAgentResponse,
    UploadCheckRequest,
    UploadCheckResponse,
)
from ditto.api_models.validator import (
    ArtifactResponse,
    CaseScore,
    JobRequest,
    JobResponse,
    LedgerEntry,
    LedgerResponse,
    ScoreReport,
    SubmitScoreRequest,
    SubmitScoreResponse,
    ValidatorHeartbeatRequest,
    ValidatorHeartbeatResponse,
)

__all__ = [
    "AgentResponse",
    "AgentStatusResponse",
    "ArtifactResponse",
    "BenchmarkProgress",
    "CaseScore",
    "EvalPricingResponse",
    "HealthResponse",
    "JobRequest",
    "JobResponse",
    "LedgerEntry",
    "LedgerResponse",
    "PublicAuditEntry",
    "PublicAuditResponse",
    "PublicActivityEntry",
    "PublicActivityResponse",
    "PublicBenchCorpusEntry",
    "BenchDatasetConfig",
    "BenchGradingConfig",
    "BenchHarnessConfig",
    "PublicBenchConfigResponse",
    "PublicBenchCorpusResponse",
    "PublicBenchIntegrity",
    "PublicBenchmarkProgress",
    "PublicCaseResult",
    "PublicCategoryStat",
    "PublicDatasetReveal",
    "PublicHealthResponse",
    "PublicLeaderboardEntry",
    "PublicLeaderboardResponse",
    "PublicOperationsResponse",
    "PublicProvisionalScore",
    "PublicRunModels",
    "PublicScreenerHeartbeat",
    "PublicScreenerHeartbeatsResponse",
    "PublicScreenerProgress",
    "PublicSubmissionScores",
    "PublicSystemMetrics",
    "PublicSubmissionPipeline",
    "PublicSubmissionSummary",
    "PublicSubmissionsResponse",
    "PublicValidatorScore",
    "PublicScreeningAttempt",
    "PublicValidationAttempt",
    "PublicValidatorHeartbeat",
    "PublicValidatorHeartbeatsResponse",
    "PublicValidatorName",
    "PublicValidatorNamesResponse",
    "ScoreReport",
    "ScreenEvidenceItem",
    "ScreenResultRequest",
    "ScreenResultResponse",
    "SourceReviewEvidenceItem",
    "SourceReviewFinding",
    "ScreenerHeartbeatRequest",
    "ScreenerHeartbeatResponse",
    "ScreenerQueueItem",
    "ScreenerQueueResponse",
    "SubmitScoreRequest",
    "SubmitScoreResponse",
    "ValidatorHeartbeatRequest",
    "ValidatorHeartbeatResponse",
    "UploadAgentResponse",
    "UploadCheckRequest",
    "UploadCheckResponse",
]
