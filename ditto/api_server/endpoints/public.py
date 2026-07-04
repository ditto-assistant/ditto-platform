"""Public, unauthenticated read endpoints for the subnet dashboard.

Unlike ``/scoring/scores`` (validator-hotkey gated, full signed rows), this
surface is open and **aggregate-only**: composite plus tool/memory means and
rank, so a public leaderboard / dashboard can read scores with no credentials
while never exposing per-case detail (the benchmark's answer key) or
integrity-internal fields. See ``docs/public-telemetry.md``.

Responses are cacheable (``max-age=30``) so a CDN / the dashboard can front this
cheaply; the underlying ledger only changes when a sweep records a new best.
The KOTH champion / weight vector is deliberately **not** served here — that is
validator-side (see the scoring endpoint's boundary note); the dashboard reads
weights from wandb or the chain.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Response

from ditto.api_models import PublicLeaderboardEntry, PublicLeaderboardResponse
from ditto.api_server.endpoints.validator import SessionDep
from ditto.db.queries.scores import list_eligible_ledger

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/public", tags=["public"])

# The ledger only moves when a sweep records a new best score, so a short shared
# cache is safe and shields the DB from dashboard/CDN traffic.
_CACHE_CONTROL = "public, max-age=30"


@router.get("/leaderboard", response_model=PublicLeaderboardResponse)
async def leaderboard(
    response: Response,
    session: SessionDep,
) -> PublicLeaderboardResponse:
    """Best eligible score per miner, aggregate-only, highest composite first."""
    response.headers["Cache-Control"] = _CACHE_CONTROL
    rows = await list_eligible_ledger(session)
    entries = [
        PublicLeaderboardEntry(
            rank=i,
            miner_hotkey=r.miner_hotkey,
            composite=r.composite,
            tool_mean=r.tool_mean,
            memory_mean=r.memory_mean,
            first_seen=r.first_seen,
        )
        for i, r in enumerate(rows, start=1)
    ]
    return PublicLeaderboardResponse(
        generated_at=datetime.now(UTC), count=len(entries), entries=entries
    )
