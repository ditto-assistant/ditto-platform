"""Scoring-pool endpoint — the public best-score ledger the validator folds.

``GET /scoring/scores`` returns one entry per active miner: that miner's best
*eligible* score (status ``scored``), highest composite first. This is the
persistent ledger that fixes the one-epoch-weight bug — the validator computes
KOTH+ATH weights from this pool every epoch instead of from the transient
``evaluating`` sweep, so a scored agent keeps its weight until genuinely
dethroned rather than being zeroed the moment it leaves the queue.

**The weight fold is NOT here.** Per the repo boundary (CLAUDE.md) and PROJECT.md
D3, weights are computed validator-side: every validator reads this identical
pool and runs the same deterministic function, so Yuma consensus clips any
deviator. The platform only exposes the raw, signed scores.

Auth mirrors the validator queue: an ``X-Validator-Hotkey`` header for a
chain-registered, validator-permitted hotkey. D3's end state makes this pool
fully public (self-verifying via the stored signatures); gating it to permitted
validators is the v1 stance.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Response

from ditto.api_models import LedgerEntry, LedgerResponse
from ditto.api_server.endpoints.validator import SessionDep, ValidatorDep
from ditto.db.queries.scores import list_eligible_ledger

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scoring", tags=["scoring"])


@router.get(
    "/scores",
    response_model=LedgerResponse,
    responses={
        401: {"description": "Missing/invalid validator auth."},
        503: {"description": "Chain unavailable for the permit check."},
    },
)
async def scores(
    response: Response,
    validator_hotkey: ValidatorDep,
    session: SessionDep,
) -> LedgerResponse:
    """Return the best eligible score per miner, highest composite first."""
    response.headers["Cache-Control"] = "no-store"
    rows = await list_eligible_ledger(session)
    entries = [
        LedgerEntry(
            miner_hotkey=r.miner_hotkey,
            agent_id=r.agent_id,
            composite=r.composite,
            first_seen=r.first_seen,
            sha256=r.sha256,
            size_bytes=r.size_bytes,
            seed=r.seed,
            validator_hotkey=r.validator_hotkey,
            signature=r.signature,
            status=r.status,
        )
        for r in rows
    ]
    logger.info(
        "validator=%s read scoring ledger: %d miner(s)", validator_hotkey, len(entries)
    )
    return LedgerResponse(entries=entries, count=len(entries))
