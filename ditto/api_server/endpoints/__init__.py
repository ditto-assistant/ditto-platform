"""HTTP routers grouped by domain."""

from __future__ import annotations

from ditto.api_server.endpoints.health import router as health_router
from ditto.api_server.endpoints.metrics import router as metrics_router
from ditto.api_server.endpoints.retrieval import router as retrieval_router
from ditto.api_server.endpoints.scoring import router as scoring_router
from ditto.api_server.endpoints.screener import router as screener_router
from ditto.api_server.endpoints.upload import router as upload_router
from ditto.api_server.endpoints.validator import router as validator_router

__all__ = [
    "health_router",
    "metrics_router",
    "retrieval_router",
    "scoring_router",
    "screener_router",
    "upload_router",
    "validator_router",
]
