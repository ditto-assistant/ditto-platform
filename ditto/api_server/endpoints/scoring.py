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
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request, Response
from sqlalchemy.exc import SQLAlchemyError

from ditto.api_models import LedgerEntry, LedgerResponse
from ditto.api_server.endpoints.validator import SessionDep, ValidatorDep
from ditto.db.queries.scores import list_eligible_ledger

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scoring", tags=["scoring"])

# Serve-last-known staleness policy: how long the cached ledger may be served
# after a live DB read fails before it is refused as too stale (503). The ledger
# moves slowly (only when a sweep records a new best) and the validator's fold is
# resilient to a missed epoch, so a few minutes of last-known-good is safe and
# far better than zeroing every miner on a transient DB blip. Beyond this the
# snapshot could hide a genuine change, so we stop vouching for it.
_MAX_STALE_SECONDS = 300


@dataclass
class _LedgerSnapshot:
    """The last successfully-read ledger, kept per-process for serve-last-known."""

    entries: list[LedgerEntry]
    generated_at: datetime


def _composite_stderr(details: dict | None) -> float | None:
    """Read the composite standard error stashed in a score's details blob
    (mirroring how bench_version is surfaced). Returns None when absent or not a
    finite non-negative number, so a malformed value degrades to the flat-margin
    fold rather than corrupting the indifference band."""
    if not isinstance(details, dict):
        return None
    v = details.get("composite_stderr")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        f = float(v)
        if f >= 0.0 and f == f and f != float("inf"):
            return f
    return None


def _confirmation_composites(details: dict | None) -> list[float] | None:
    """Read the P4 per-seed confirmation composites stashed in a score's details
    blob (mirroring :func:`_composite_stderr`). Returns None unless the value is a
    list of finite floats in [0, 1], so a malformed value degrades to the raw-
    composite fold rather than corrupting the dethrone comparison. Surfaced
    as-is (any length); the validator ignores lists shorter than two."""
    if not isinstance(details, dict):
        return None
    v = details.get("confirmation_composites")
    if not isinstance(v, list) or not v:
        return None
    out: list[float] = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        f = float(x)
        if not (0.0 <= f <= 1.0) or f != f:
            return None
        out.append(f)
    return out


def _cached_snapshot(request: Request) -> _LedgerSnapshot | None:
    return getattr(request.app.state, "ledger_snapshot", None)


def _store_snapshot(request: Request, snapshot: _LedgerSnapshot) -> None:
    request.app.state.ledger_snapshot = snapshot


@router.get(
    "/scores",
    response_model=LedgerResponse,
    responses={
        401: {"description": "Missing/invalid validator auth."},
        503: {"description": "Chain unavailable, or no fresh/recent ledger to serve."},
    },
)
async def scores(
    request: Request,
    response: Response,
    validator_hotkey: ValidatorDep,
    session: SessionDep,
) -> LedgerResponse:
    """Return the best eligible score per miner, highest composite first.

    Serve-last-known: on a transient DB failure the last successfully-read ledger
    is served (flagged ``stale`` with its ``age_seconds``) rather than erroring,
    since zeroing every miner on a blip is the worse failure. Past
    :data:`_MAX_STALE_SECONDS` the cache is refused with 503 so a consumer never
    silently folds a dangerously old pool.
    """
    response.headers["Cache-Control"] = "no-store"
    try:
        rows = await list_eligible_ledger(session)
    except SQLAlchemyError as e:
        return _serve_last_known(request, validator_hotkey, e)

    now = datetime.now(UTC)
    entries = [
        LedgerEntry(
            miner_hotkey=r.miner_hotkey,
            agent_id=r.agent_id,
            composite=r.composite,
            n=r.n,
            first_seen=r.first_seen,
            sha256=r.sha256,
            size_bytes=r.size_bytes,
            run_id=r.run_id,
            seed=r.seed,
            validator_hotkey=r.validator_hotkey,
            signature=r.signature,
            composite_stderr=_composite_stderr(r.details),
            confirmation_composites=_confirmation_composites(r.details),
            status=r.status,
        )
        for r in rows
    ]
    _store_snapshot(request, _LedgerSnapshot(entries=entries, generated_at=now))
    logger.info(
        "validator=%s read scoring ledger: %d miner(s)", validator_hotkey, len(entries)
    )
    return LedgerResponse(
        entries=entries,
        count=len(entries),
        generated_at=now,
        stale=False,
        age_seconds=0,
    )


def _serve_last_known(
    request: Request, validator_hotkey: str, error: Exception
) -> LedgerResponse:
    """Serve the cached ledger on a DB failure, or 503 if there is none / too old."""
    snapshot = _cached_snapshot(request)
    if snapshot is None:
        logger.warning(
            "validator=%s ledger read failed and no cached snapshot to serve: %s",
            validator_hotkey,
            error,
        )
        raise HTTPException(
            status_code=503, detail="scoring ledger temporarily unavailable"
        ) from error
    age = int((datetime.now(UTC) - snapshot.generated_at).total_seconds())
    if age > _MAX_STALE_SECONDS:
        logger.warning(
            "validator=%s ledger read failed; cached snapshot is %ds old "
            "(> %ds max), refusing to serve stale: %s",
            validator_hotkey,
            age,
            _MAX_STALE_SECONDS,
            error,
        )
        raise HTTPException(
            status_code=503,
            detail=f"scoring ledger unavailable; last snapshot {age}s old exceeds "
            f"the {_MAX_STALE_SECONDS}s staleness limit",
        ) from error
    logger.warning(
        "validator=%s ledger read failed; serving last-known-good (%ds old, "
        "%d miner(s)): %s",
        validator_hotkey,
        age,
        len(snapshot.entries),
        error,
    )
    return LedgerResponse(
        entries=snapshot.entries,
        count=len(snapshot.entries),
        generated_at=snapshot.generated_at,
        stale=True,
        age_seconds=max(0, age),
    )
