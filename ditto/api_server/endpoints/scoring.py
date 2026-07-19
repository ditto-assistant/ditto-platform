"""Scoring-pool endpoint — the validator best-score ledger used for weights.

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

The raw pool remains validator-only even though public projections expose a safe
subset of benchmark results. Reads require a fresh signature from a
chain-registered, validator-permitted hotkey; a public hotkey is an identity, not
proof that the caller controls it.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Request, Response
from sqlalchemy.exc import SQLAlchemyError

from ditto.api_models import LedgerEntry, LedgerResponse
from ditto.api_models.upload import _SS58_PATTERN
from ditto.api_server.endpoints.validator import (
    ChainDep,
    SessionDep,
    ValidatorAuthError,
    _assert_validator_permitted,
    _verify_signature,
)
from ditto.db.queries.scores import list_eligible_ledger, quorum_composites
from ditto.db.queries.validator_auth import (
    ValidatorRequestReplayError,
    consume_validator_nonce,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scoring", tags=["scoring"])

# Serve-last-known staleness policy: how long the cached ledger may be served
# after a live DB read fails before it is refused as too stale (503). The ledger
# moves slowly (only when a sweep records a new best) and the validator's fold is
# resilient to a missed epoch, so a few minutes of last-known-good is safe and
# far better than zeroing every miner on a transient DB blip. Beyond this the
# snapshot could hide a genuine change, so we stop vouching for it.
_MAX_STALE_SECONDS = 300
_LEDGER_REQUEST_MAX_AGE = timedelta(minutes=2)


def _ledger_signing_message(
    validator_hotkey: str, nonce: UUID, requested_at: datetime
) -> bytes:
    """Canonical proof-of-possession bytes for one scoring-ledger read."""
    requested = requested_at.astimezone(UTC).isoformat(timespec="microseconds")
    return f"validator-ledger:v1:{validator_hotkey}:{nonce}:{requested}".encode()


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


def _confirmation_seeds(details: dict | None) -> list[int] | None:
    """Read the P4 confirmation CRN seeds stashed in a score's details blob,
    aligned 1:1 with :func:`_confirmation_composites`. Returns None unless the
    value is a non-empty list of non-negative ints, so a malformed value degrades
    to the unpaired dethrone band rather than corrupting the paired comparison.
    Surfaced as-is (any length); the validator pairs only over shared seeds."""
    if not isinstance(details, dict):
        return None
    v = details.get("confirmation_seeds")
    if not isinstance(v, list) or not v:
        return None
    out: list[int] = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, int) or x < 0:
            return None
        out.append(x)
    return out


def _quorum_stderr(composites: list[float]) -> float | None:
    """The standard error of an agent's composite from its k=3 quorum spread:
    ``stdev(composites) / sqrt(n)`` over the validators' composites, or None with
    fewer than two scores. This turns data the platform ALREADY collects (the
    quorum) into the KOTH z-band's measurement-uncertainty estimate with no
    re-score, so a plain quorum-scored champion still gets a noise-aware dethrone
    band. Only finite composites contribute; a degenerate (identical) quorum
    yields 0.0 (the band collapses to the flat margin, which is correct)."""
    xs = [c for c in composites if isinstance(c, (int, float)) and c == c]
    n = len(xs)
    if n < 2:
        return None
    mean = sum(xs) / n
    var = sum((c - mean) ** 2 for c in xs) / (n - 1)
    return math.sqrt(var / n)


def _ledger_stderr(details: dict | None, quorum: list[float]) -> float | None:
    """The composite standard error to surface on a ledger entry. Prefer the
    run's own stashed ``composite_stderr`` (e.g. a confirmation re-score's pooled
    between-seed SE); else fall back to the between-validator SEM of the k=3
    quorum (:func:`_quorum_stderr`), so the KOTH z-band has an uncertainty
    estimate even for a plain quorum-scored agent, from data already collected."""
    stashed = _composite_stderr(details)
    if stashed is not None:
        return stashed
    return _quorum_stderr(quorum)


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
    chain: ChainDep,
    session: SessionDep,
    x_validator_hotkey: Annotated[str | None, Header()] = None,
    x_validator_ledger_nonce: Annotated[UUID | None, Header()] = None,
    x_validator_ledger_requested_at: Annotated[datetime | None, Header()] = None,
    x_validator_ledger_signature: Annotated[str | None, Header()] = None,
) -> LedgerResponse:
    """Return the best eligible score per miner, highest composite first.

    Serve-last-known: on a transient DB failure the last successfully-read ledger
    is served (flagged ``stale`` with its ``age_seconds``) rather than erroring,
    since zeroing every miner on a blip is the worse failure. Past
    :data:`_MAX_STALE_SECONDS` the cache is refused with 503 so a consumer never
    silently folds a dangerously old pool.
    """
    response.headers["Cache-Control"] = "no-store"
    if (
        x_validator_hotkey is None
        or not re.fullmatch(_SS58_PATTERN, x_validator_hotkey)
        or x_validator_ledger_nonce is None
        or x_validator_ledger_requested_at is None
        or x_validator_ledger_signature is None
        or x_validator_ledger_requested_at.tzinfo is None
    ):
        raise ValidatorAuthError("ledger request proof is missing or malformed")
    signed = _ledger_signing_message(
        x_validator_hotkey,
        x_validator_ledger_nonce,
        x_validator_ledger_requested_at,
    )
    if not _verify_signature(x_validator_hotkey, signed, x_validator_ledger_signature):
        raise ValidatorAuthError("ledger request signature did not verify")
    auth_now = datetime.now(UTC)
    if (
        abs(auth_now - x_validator_ledger_requested_at.astimezone(UTC))
        > _LEDGER_REQUEST_MAX_AGE
    ):
        raise HTTPException(status_code=409, detail="ledger request timestamp is stale")
    await _assert_validator_permitted(
        chain,
        request.app.state.config.chain.netuid,
        x_validator_hotkey,
        network=request.app.state.config.chain.subtensor_network,
    )
    async with session.begin():
        try:
            await consume_validator_nonce(
                session,
                nonce=x_validator_ledger_nonce,
                validator_hotkey=x_validator_hotkey,
                now=auth_now,
                expires_at=auth_now + _LEDGER_REQUEST_MAX_AGE,
            )
        except ValidatorRequestReplayError as exc:
            raise HTTPException(
                status_code=409, detail="ledger request nonce has already been used"
            ) from exc
        except SQLAlchemyError as exc:
            # Never serve cached ledger data when replay protection cannot record
            # this proof. Failing closed preserves one-time nonce semantics.
            raise HTTPException(
                status_code=503,
                detail="scoring ledger authorization temporarily unavailable",
            ) from exc
    try:
        rows = await list_eligible_ledger(session, include_fingerprints=False)
        # The k=3 quorum spread per agent -> composite_stderr when the run itself
        # did not stash one, so the KOTH z-band is noise-aware with no re-score.
        quorum = await quorum_composites(
            session,
            [r.agent_id for r in rows],
            bench_versions={r.agent_id: r.bench_version for r in rows},
        )
    except SQLAlchemyError as e:
        return _serve_last_known(request, x_validator_hotkey, e)

    generated_at = datetime.now(UTC)
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
            bench_version=r.bench_version,
            signature=r.signature,
            composite_stderr=_ledger_stderr(r.details, quorum.get(r.agent_id, [])),
            confirmation_composites=_confirmation_composites(r.details),
            confirmation_seeds=_confirmation_seeds(r.details),
            status=r.status,
        )
        for r in rows
    ]
    _store_snapshot(
        request, _LedgerSnapshot(entries=entries, generated_at=generated_at)
    )
    logger.info(
        "validator=%s read scoring ledger: %d miner(s)",
        x_validator_hotkey,
        len(entries),
    )
    return LedgerResponse(
        entries=entries,
        count=len(entries),
        generated_at=generated_at,
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
