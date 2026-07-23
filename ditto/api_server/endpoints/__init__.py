"""HTTP routers grouped by domain."""

from __future__ import annotations

from ditto.api_server.endpoints.admin_benchmark_rollout import (
    router as admin_benchmark_rollout_router,
)
from ditto.api_server.endpoints.admin_copy_review import (
    router as admin_copy_review_router,
)
from ditto.api_server.endpoints.admin_inference_routes import (
    router as admin_inference_routes_router,
)
from ditto.api_server.endpoints.admin_miner_fees import (
    router as admin_miner_fees_router,
)
from ditto.api_server.endpoints.admin_quarantine import (
    router as admin_quarantine_router,
)
from ditto.api_server.endpoints.admin_scoring_readiness import (
    router as admin_scoring_readiness_router,
)
from ditto.api_server.endpoints.admin_screener_review_settings import (
    router as admin_screener_review_settings_router,
)
from ditto.api_server.endpoints.admin_validation_retry import (
    router as admin_validation_retry_router,
)
from ditto.api_server.endpoints.health import router as health_router
from ditto.api_server.endpoints.inference import router as inference_router
from ditto.api_server.endpoints.metrics import router as metrics_router
from ditto.api_server.endpoints.public import router as public_router
from ditto.api_server.endpoints.retrieval import router as retrieval_router
from ditto.api_server.endpoints.scoring import router as scoring_router
from ditto.api_server.endpoints.screener import router as screener_router
from ditto.api_server.endpoints.upload import router as upload_router
from ditto.api_server.endpoints.validator import router as validator_router

__all__ = [
    "health_router",
    "inference_router",
    "admin_benchmark_rollout_router",
    "admin_inference_routes_router",
    "admin_copy_review_router",
    "admin_miner_fees_router",
    "admin_quarantine_router",
    "admin_scoring_readiness_router",
    "admin_screener_review_settings_router",
    "admin_validation_retry_router",
    "metrics_router",
    "public_router",
    "retrieval_router",
    "scoring_router",
    "screener_router",
    "upload_router",
    "validator_router",
]
