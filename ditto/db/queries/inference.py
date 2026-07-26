"""Ticket-scoped inference grant lifecycle."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from ditto.api_models.ticket_status import TicketStatus
from ditto.api_server.inference_routing import benchmark_model, select_route
from ditto.db.models import InferenceGrant, InferenceRequest, ValidatorTicket

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from ditto.api_server.config import InferenceProxyConfig


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def bearer_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


async def ensure_inference_grant(
    session: AsyncSession,
    *,
    ticket: ValidatorTicket,
    config: InferenceProxyConfig,
    supported_profiles: tuple[str, ...] | None = None,
    calibration_manifest_sha256: str | None = None,
) -> InferenceGrant | None:
    """Create or return the one grant bound to this exact live lease.

    Creation is race-free at the database level, not by convention. ``SELECT
    ... FOR UPDATE`` locks rows that exist; when the grant has not been created
    yet there is nothing to lock, so two callers racing on the same lease -- two
    concurrent offer/heartbeat calls for one ticket -- can both miss and both
    insert. The ``inference_grants_ticket_lease`` unique constraint has always
    made the dangerous outcome impossible: a single ticket could never actually
    obtain two grants and therefore never double its request or token budget.
    What was missing was handling the losing side, which surfaced the conflict
    as an unhandled IntegrityError and a 500 on an otherwise valid offer. The
    loser now adopts the winner's row, which is the answer it wanted anyway.

    The savepoint spans route selection as well as the insert, so a loser also
    rolls back the ``selected_ticket_count`` increment ``select_route`` applies:
    it never held a ticket, so it must not be counted as having been offered
    one.
    """
    if not config.enabled or ticket.status != TicketStatus.ISSUED:
        return None
    deadline = _aware(ticket.deadline)
    lease = (
        select(InferenceGrant)
        .where(
            InferenceGrant.agent_id == ticket.agent_id,
            InferenceGrant.bench_version == ticket.bench_version,
            InferenceGrant.validator_hotkey == ticket.validator_hotkey,
            InferenceGrant.ticket_deadline == deadline,
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    grant = await session.scalar(lease)
    if grant is None:
        model = benchmark_model(ticket.bench_version)
        if model not in config.allowed_models:
            return None
        route_provider: str | None = config.provider
        route_profile: str | None = f"legacy-config-{config.provider}"
        route_quantization: str | None = None
        route_prompt_price_per_token: float | None = None
        route_completion_price_per_token: float | None = None
        try:
            async with session.begin_nested():
                if ticket.bench_version >= 7:
                    route = await select_route(
                        session,
                        model=model,
                        now=datetime.now(UTC),
                        supported_profiles=supported_profiles,
                        calibration_manifest_sha256=calibration_manifest_sha256,
                        routing_mode=config.routing_mode,
                    )
                    if route is None:
                        return None
                    route_provider = route.provider
                    route_profile = route.profile_revision
                    route_quantization = route.quantization
                    route_prompt_price_per_token = route.prompt_price_per_token
                    route_completion_price_per_token = route.completion_price_per_token
                grant = InferenceGrant(
                    grant_id=uuid4(),
                    agent_id=ticket.agent_id,
                    bench_version=ticket.bench_version,
                    validator_hotkey=ticket.validator_hotkey,
                    slot_id=ticket.slot_id,
                    ticket_deadline=deadline,
                    status="pending",
                    bearer_digest=None,
                    broker_public_key=None,
                    generation=0,
                    allowed_models=[model],
                    route_provider=route_provider,
                    route_profile=route_profile,
                    route_quantization=route_quantization,
                    route_prompt_price_per_token=route_prompt_price_per_token,
                    route_completion_price_per_token=route_completion_price_per_token,
                    request_budget=config.request_budget,
                    token_budget=config.token_budget,
                    embedding_model=config.embedding_model,
                    embedding_profile=config.embedding_profile,
                    embedding_provider=config.embedding_provider,
                    embedding_dimensions=config.embedding_dimensions,
                    embedding_request_budget=config.embedding_request_budget,
                    embedding_token_budget=config.embedding_token_budget,
                    embedding_request_count=0,
                    embedding_tokens=0,
                    embedding_cost_microusd=0,
                    embedding_active_requests=0,
                    request_count=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    cost_microusd=0,
                    active_requests=0,
                    expires_at=deadline,
                )
                session.add(grant)
                await session.flush()
        except IntegrityError:
            # Another caller created this lease's grant first. Its row is the
            # one grant for this ticket; adopt it instead of inserting a second.
            return await session.scalar(lease)
    return grant


async def activate_inference_grant(
    session: AsyncSession,
    *,
    grant_id: UUID,
    validator_hotkey: str,
    broker_public_key: str,
    now: datetime,
    config: InferenceProxyConfig,
) -> tuple[InferenceGrant, str] | None:
    """Rotate the broker binding and return a fresh opaque bearer.

    Rotation is restart-safe: the prior bearer becomes invalid immediately and
    a fresh validator signature is required for every exchange.
    """
    snapshot = await session.get(InferenceGrant, grant_id)
    if snapshot is None or snapshot.validator_hotkey != validator_hotkey:
        return None
    ticket = await session.get(
        ValidatorTicket,
        (snapshot.agent_id, snapshot.bench_version, snapshot.validator_hotkey),
        with_for_update=True,
    )
    grant = await session.scalar(
        select(InferenceGrant)
        .where(InferenceGrant.grant_id == grant_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if grant is None:
        return None
    if (
        grant.validator_hotkey != validator_hotkey
        or ticket is None
        or ticket.status != TicketStatus.ISSUED
        or _aware(ticket.deadline) != _aware(grant.ticket_deadline)
        or _aware(ticket.deadline) <= now
        or grant.status in {"revoked", "exhausted"}
    ):
        grant.status = "revoked"
        return None
    started = list(
        (
            await session.scalars(
                select(InferenceRequest)
                .where(
                    InferenceRequest.grant_id == grant.grant_id,
                    InferenceRequest.status == "started",
                )
                .with_for_update()
            )
        ).all()
    )
    stale_cutoff = now - timedelta(seconds=config.timeout_seconds * 2)
    if any(_aware(request.started_at) >= stale_cutoff for request in started):
        # A restart may rotate only after every previous generation call has
        # either settled or crossed the provider timeout recovery window.
        return None
    for request in started:
        request.status = "canceled"
        request.prompt_tokens = request.reserved_tokens
        request.completed_at = now
        if request.request_kind == "chat":
            grant.prompt_tokens += request.reserved_tokens
        else:
            grant.embedding_tokens += request.reserved_tokens
    bearer = secrets.token_urlsafe(32)
    grant.bearer_digest = bearer_digest(bearer)
    grant.broker_public_key = broker_public_key.rstrip("=")
    grant.generation += 1
    grant.status = "active"
    grant.slot_id = ticket.slot_id
    grant.expires_at = _aware(ticket.deadline)
    grant.active_requests = 0
    grant.embedding_active_requests = 0
    grant.updated_at = now
    await session.flush()
    return grant, bearer


async def revoke_ticket_inference(
    session: AsyncSession,
    *,
    ticket: ValidatorTicket,
    now: datetime,
) -> None:
    grants = list(
        (
            await session.scalars(
                select(InferenceGrant)
                .where(
                    InferenceGrant.agent_id == ticket.agent_id,
                    InferenceGrant.bench_version == ticket.bench_version,
                    InferenceGrant.validator_hotkey == ticket.validator_hotkey,
                    InferenceGrant.status.in_(("pending", "active")),
                )
                .with_for_update()
                .execution_options(populate_existing=True)
            )
        ).all()
    )
    for grant in grants:
        requests = list(
            (
                await session.scalars(
                    select(InferenceRequest)
                    .where(
                        InferenceRequest.grant_id == grant.grant_id,
                        InferenceRequest.status == "started",
                    )
                    .with_for_update()
                )
            ).all()
        )
        for request in requests:
            request.status = "canceled"
            request.prompt_tokens = request.reserved_tokens
            request.completed_at = now
            if request.request_kind == "chat":
                grant.prompt_tokens += request.reserved_tokens
            else:
                grant.embedding_tokens += request.reserved_tokens
        grant.status = "revoked"
        grant.active_requests = 0
        grant.embedding_active_requests = 0
        grant.updated_at = now


class InferenceDecline(StrEnum):
    """Why an admission was refused, when the reason is worth naming.

    Historically :func:`begin_inference_request` returned ``None`` for every
    refusal and the endpoint mapped all of them to ``429``. That collapsed three
    unrelated events into one status code, and dittobench-api #103 documents the
    damage: on the ticket path the broker reads *any* ``429`` as "the lease is
    gone" and discards the whole run.

    The three events, now each with a name:

    * :attr:`GRANT_REVOKED` — the lease really is dead (the ticket expired, was
      reassigned, or the deadline moved). Fatal, and correctly so.
    * :attr:`BUDGET_EXHAUSTED` — the lease is alive but its request allowance is
      spent. Also terminal for chat, but for a completely different reason, and
      a harness that can tell the difference can wind down and submit the work
      it already has instead of dying mid-check.
    * :attr:`AT_CAPACITY` — nothing is wrong at all; the lane was momentarily
      full. The caller should back off and come back.

    Conflating the third with the first is what killed ``banblackycat``: 17
    capacity declines, read as 17 dead leases. It is also why the status code is
    no longer the discriminator. The endpoint answers ``AT_CAPACITY`` with
    ``503 + Retry-After`` and the two terminal declines with ``429``, but the
    *authoritative* signal is the numeric ``error_code`` in the error body
    (``ditto/api_server/middleware/error_envelope.py``). A status code carries
    about two bits; application semantics need more, and every attempt to
    encode them in the status has cost a run.
    """

    GRANT_REVOKED = "grant_revoked"
    BUDGET_EXHAUSTED = "budget_exhausted"
    AT_CAPACITY = "at_capacity"


# Both lanes return AT_CAPACITY now. Chat used to be excluded on the grounds
# that its limits are boot-time constants which cannot move under a live ticket
# -- true, and still true, but it was never the whole argument. A chat lane can
# hit its rate or concurrency ceiling under perfectly ordinary load with nothing
# whatsoever wrong with the lease, and answering that with the same ``429`` that
# means "your lease is dead" is how a healthy run gets thrown away.


async def begin_inference_request(
    session: AsyncSession,
    *,
    grant_id: UUID,
    nonce: UUID,
    bearer: str,
    model: str,
    token_reservation: int,
    now: datetime,
    config: InferenceProxyConfig,
    request_kind: str = "chat",
) -> tuple[InferenceGrant, InferenceRequest] | InferenceDecline | None:
    """Atomically consume one nonce and reserve bounded proxy capacity.

    Returns the reservation on success, an :class:`InferenceDecline` when the
    refusal has a name worth telling the caller, and ``None`` for the remaining
    fail-closed class.

    ``None`` now means strictly "refused, and saying why would either help an
    attacker or mean nothing to the caller": a bad bearer, a grant that is not
    this caller's, a model the grant does not permit, a replayed nonce, an
    expired clock, a grant minted but never exchanged, or a token budget spent.
    Everything an honest broker can act on differently -- the lease is dead, the
    allowance is spent, the lane is full -- is a named decline. The caller maps
    all three to a status code *and* a stable numeric error code; see
    :class:`InferenceDecline`.

    Locking model: the grant row taken ``FOR UPDATE`` below is the only
    serialization point, and every invariant that spends a budget is scoped to
    that one grant -- reserved tokens, request count, per-ticket concurrency,
    per-ticket rate, stale reclamation, and the nonce replay guard all filter on
    ``grant_id``. Postgres serializes writers of one row by construction, so
    two reservations against the same grant cannot both pass a budget check,
    while reservations against different grants proceed fully in parallel.

    ``populate_existing`` on that locking read is load-bearing, not decoration.
    The unlocked ``session.get`` above puts the row in the identity map, and by
    default SQLAlchemy will hand a later query the object it already has
    without overwriting its attributes from the new result. The FOR UPDATE
    select would then block correctly, wait its turn correctly, and still
    evaluate every budget check against the values it read *before* acquiring
    the lock -- so concurrent reservations would each see a stale
    ``request_count`` and collectively overspend the grant. The old global
    advisory lock hid this: it was taken before the unlocked read, so nothing
    could commit between them and the stale value was always current anyway.
    Removing the lock without this is silent accounting corruption, and
    ``test_reservations_on_one_grant_serialize_and_respect_the_budget`` fails
    against real Postgres if it is dropped.

    This previously also took ``pg_advisory_xact_lock(hashtextextended(
    'inference', 0))`` -- one lock, with a constant key, for every reservation
    on the platform. It was held for the whole transaction (roughly eight
    statements), so it serialized the entire fleet's reservation path and put a
    hard ceiling on horizontal scaling. It was never what protected the
    money-critical invariants; the grant row lock already did, and no other
    caller in this module ever took the advisory lock, so it also provided no
    mutual exclusion with grant creation, activation, revocation, or finish.

    What it did cover is the cross-grant admission rails below (per-validator
    and global in-flight counts and per-minute rates). Those aggregate across
    every grant, so no row lock can make them exact, and making them exact
    would require reintroducing exactly the global barrier being removed. They
    are therefore best-effort: a burst of simultaneous reservations can
    overshoot a rail by at most the number of racers, which is acceptable for
    operational load-shedding backstops with headroom. The per-ticket rails
    directly above them stay exact, and those are the ones a miner can target.
    Best-effort does not change what a refusal *means*: a cross-grant rail that
    does trip still answers :attr:`InferenceDecline.AT_CAPACITY`, so the caller
    reports a healthy-but-full lane rather than a dead lease.
    """
    if request_kind not in {"chat", "embedding"}:
        return None
    snapshot = await session.get(InferenceGrant, grant_id)
    if snapshot is None:
        return None
    ticket = await session.get(
        ValidatorTicket,
        (snapshot.agent_id, snapshot.bench_version, snapshot.validator_hotkey),
        with_for_update=True,
    )
    grant = await session.scalar(
        select(InferenceGrant)
        .where(InferenceGrant.grant_id == grant_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if (
        grant is None
        or grant.bearer_digest is None
        or not secrets.compare_digest(grant.bearer_digest, bearer_digest(bearer))
        or _aware(grant.expires_at) <= now
        or (
            model not in grant.allowed_models
            if request_kind == "chat"
            else grant.bench_version < 7 or model != grant.embedding_model
        )
    ):
        return None
    # Status is checked *after* the bearer comparison, never before. The reason
    # a grant is unusable is information about someone else's lease, so it is
    # only ever disclosed to a caller that has already proved it holds this
    # grant's bearer. Ordering, not an extra check, is what enforces that.
    #
    # This gate is also what makes the terminal declines *persistent*. The first
    # refusal below sets the status; every subsequent call in the run lands
    # here, and without this branch the whole tail of the run would decay back
    # to an unnamed refusal -- which is precisely the window in which a harness
    # needs to know whether to retry, wind down, or give up.
    if grant.status != "active":
        if grant.status == "exhausted":
            return InferenceDecline.BUDGET_EXHAUSTED
        if grant.status == "revoked":
            return InferenceDecline.GRANT_REVOKED
        # "pending" -- minted but never exchanged. Not a named decline: there is
        # no live lease here to have an opinion about.
        return None
    stale_cutoff = now - timedelta(seconds=config.timeout_seconds * 2)
    stale_requests = list(
        (
            await session.scalars(
                select(InferenceRequest)
                .where(
                    InferenceRequest.grant_id == grant.grant_id,
                    InferenceRequest.status == "started",
                    InferenceRequest.request_kind == request_kind,
                    InferenceRequest.started_at < stale_cutoff,
                )
                .with_for_update()
            )
        ).all()
    )
    for stale in stale_requests:
        stale.status = "canceled"
        stale.prompt_tokens = stale.reserved_tokens
        stale.completed_at = now
        if request_kind == "chat":
            grant.prompt_tokens += stale.reserved_tokens
        else:
            grant.embedding_tokens += stale.reserved_tokens
    if stale_requests:
        await session.flush()
        active_count = int(
            await session.scalar(
                select(func.count()).where(
                    InferenceRequest.grant_id == grant.grant_id,
                    InferenceRequest.status == "started",
                    InferenceRequest.request_kind == request_kind,
                )
            )
            or 0
        )
        if request_kind == "chat":
            grant.active_requests = active_count
        else:
            grant.embedding_active_requests = active_count
    if (
        ticket is None
        or ticket.status != TicketStatus.ISSUED
        or _aware(ticket.deadline) != _aware(grant.ticket_deadline)
        or _aware(ticket.deadline) <= now
    ):
        grant.status = "revoked"
        return InferenceDecline.GRANT_REVOKED
    if request_kind == "chat" and grant.request_count >= grant.request_budget:
        # Still terminal: the allowance is spent and no amount of waiting brings
        # it back, so the broker must not retry. What changes is that the caller
        # can now say *which* terminal this is. "Your lease died" and "you spent
        # your budget" call for opposite reactions from a harness -- discard
        # versus wind down and submit -- and until now they were the same byte.
        grant.status = "exhausted"
        return InferenceDecline.BUDGET_EXHAUSTED
    if (
        request_kind == "embedding"
        and grant.embedding_request_count >= grant.embedding_request_budget
    ):
        # Deliberately not a status change. The embedding allowance is 100,000
        # against ~671 used per run, so reaching it means something pathological
        # rather than a strategy being thorough, and killing the grant outright
        # would also take the chat lane down with it.
        return InferenceDecline.BUDGET_EXHAUSTED
    active_reserved = await session.scalar(
        select(func.coalesce(func.sum(InferenceRequest.reserved_tokens), 0)).where(
            InferenceRequest.grant_id == grant.grant_id,
            InferenceRequest.status == "started",
            InferenceRequest.request_kind == request_kind,
        )
    )
    if token_reservation < 1 or (
        grant.prompt_tokens + grant.completion_tokens
        if request_kind == "chat"
        else grant.embedding_tokens
    ) + int(active_reserved or 0) + token_reservation > (
        grant.token_budget if request_kind == "chat" else grant.embedding_token_budget
    ):
        return None
    active_requests = (
        grant.active_requests
        if request_kind == "chat"
        else grant.embedding_active_requests
    )
    per_ticket_concurrency = (
        config.per_ticket_concurrency
        if request_kind == "chat"
        else config.embedding_per_ticket_concurrency
    )
    if active_requests >= per_ticket_concurrency:
        # Healthy lease, lane momentarily full. This is the limit an operator
        # tunes from backroom, so it is also the one most likely to move under a
        # live run -- it must degrade to backpressure, never to a lost run.
        return InferenceDecline.AT_CAPACITY

    # Fast replay path avoids an ORM identity collision in the common case;
    # the composite primary key and nested transaction remain authoritative
    # for concurrent attempts on different platform workers.
    if await session.get(InferenceRequest, (grant.grant_id, nonce)) is not None:
        return None

    active_column = (
        InferenceGrant.active_requests
        if request_kind == "chat"
        else InferenceGrant.embedding_active_requests
    )
    validator_active = await session.scalar(
        select(func.coalesce(func.sum(active_column), 0)).where(
            InferenceGrant.validator_hotkey == grant.validator_hotkey,
            InferenceGrant.status == "active",
        )
    )
    global_active = await session.scalar(
        select(func.coalesce(func.sum(active_column), 0)).where(
            InferenceGrant.status == "active"
        )
    )
    minute_start = now - timedelta(minutes=1)
    validator_recent = await session.scalar(
        select(func.count())
        .select_from(InferenceRequest)
        .join(InferenceGrant, InferenceGrant.grant_id == InferenceRequest.grant_id)
        .where(
            InferenceGrant.validator_hotkey == grant.validator_hotkey,
            InferenceRequest.started_at >= minute_start,
            InferenceRequest.request_kind == request_kind,
        )
    )
    ticket_recent = await session.scalar(
        select(func.count()).where(
            InferenceRequest.grant_id == grant.grant_id,
            InferenceRequest.started_at >= minute_start,
            InferenceRequest.request_kind == request_kind,
        )
    )
    global_recent = await session.scalar(
        select(func.count()).where(
            InferenceRequest.started_at >= minute_start,
            InferenceRequest.request_kind == request_kind,
        )
    )
    per_validator_concurrency = (
        config.per_validator_concurrency
        if request_kind == "chat"
        else config.embedding_per_validator_concurrency
    )
    global_concurrency = (
        config.global_concurrency
        if request_kind == "chat"
        else config.embedding_global_concurrency
    )
    per_ticket_rpm = (
        config.per_ticket_requests_per_minute
        if request_kind == "chat"
        else config.embedding_per_ticket_requests_per_minute
    )
    per_validator_rpm = (
        config.per_validator_requests_per_minute
        if request_kind == "chat"
        else config.embedding_per_validator_requests_per_minute
    )
    global_rpm = (
        config.global_requests_per_minute
        if request_kind == "chat"
        else config.embedding_global_requests_per_minute
    )
    if (
        int(validator_active or 0) >= per_validator_concurrency
        or int(global_active or 0) >= global_concurrency
        or int(ticket_recent or 0) >= per_ticket_rpm
        or int(validator_recent or 0) >= per_validator_rpm
        or int(global_recent or 0) >= global_rpm
    ):
        return InferenceDecline.AT_CAPACITY

    request = InferenceRequest(
        grant_id=grant.grant_id,
        nonce=nonce,
        generation=grant.generation,
        status="started",
        request_kind=request_kind,
        model=model,
        reserved_tokens=token_reservation,
        started_at=now,
    )
    try:
        async with session.begin_nested():
            session.add(request)
            await session.flush()
    except IntegrityError:
        # The composite primary key is the distributed replay guard.
        return None
    if request_kind == "chat":
        grant.request_count += 1
        grant.active_requests += 1
    else:
        grant.embedding_request_count += 1
        grant.embedding_active_requests += 1
    grant.updated_at = now
    await session.flush()
    return grant, request


async def finish_inference_request(
    session: AsyncSession,
    *,
    grant_id: UUID,
    nonce: UUID,
    generation: int,
    status: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_microusd: int,
    usage_available: bool,
    now: datetime,
    upstream_provider: str | None = None,
    timed_out: bool = False,
    latency_ms: int | None = None,
    upstream_attempts: int = 0,
) -> bool:
    snapshot = await session.get(InferenceGrant, grant_id)
    if snapshot is None:
        return False
    ticket = await session.get(
        ValidatorTicket,
        (snapshot.agent_id, snapshot.bench_version, snapshot.validator_hotkey),
        with_for_update=True,
    )
    grant = await session.scalar(
        select(InferenceGrant)
        .where(InferenceGrant.grant_id == grant_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    request = await session.get(
        InferenceRequest, (grant_id, nonce), with_for_update=True
    )
    if (
        grant is None
        or request is None
        or request.status not in {"started", "canceled"}
        or request.generation != generation
    ):
        return False
    was_started = request.status == "started"
    if not was_started and (
        request.prompt_tokens > 0
        or request.completion_tokens > 0
        or request.cost_microusd > 0
    ):
        return False
    deliverable = (
        status == "completed"
        and usage_available
        and grant.status == "active"
        and grant.generation == generation
        and was_started
        and _aware(grant.expires_at) > now
        and ticket is not None
        and ticket.status == TicketStatus.ISSUED
        and _aware(ticket.deadline) == _aware(grant.ticket_deadline)
        and _aware(ticket.deadline) > now
    )
    prompt_tokens = max(0, prompt_tokens)
    completion_tokens = max(0, completion_tokens)
    cost_microusd = max(0, cost_microusd)
    if not usage_available:
        # Every provider outcome without trusted usage is conservatively
        # charged to its reservation, including timeout and transport failure.
        prompt_tokens = request.reserved_tokens
        completion_tokens = 0
    elif prompt_tokens + completion_tokens > request.reserved_tokens:
        # Untrusted provider accounting cannot exceed the atomically reserved
        # budget or overflow the grant's integer counters.
        prompt_tokens = request.reserved_tokens
        completion_tokens = 0
        deliverable = False
    request.status = (
        status if was_started and (deliverable or status != "completed") else "canceled"
    )
    request.prompt_tokens = prompt_tokens
    request.completion_tokens = completion_tokens
    request.cost_microusd = cost_microusd
    request.upstream_provider = upstream_provider
    request.upstream_attempts = max(0, upstream_attempts)
    request.timed_out = timed_out
    request.latency_ms = latency_ms
    request.completed_at = now
    if request.request_kind == "chat":
        if was_started:
            grant.active_requests = max(0, grant.active_requests - 1)
        grant.prompt_tokens += prompt_tokens
        grant.completion_tokens += completion_tokens
        grant.cost_microusd += cost_microusd
    else:
        if was_started:
            grant.embedding_active_requests = max(
                0, grant.embedding_active_requests - 1
            )
        grant.embedding_tokens += prompt_tokens
        grant.embedding_cost_microusd += cost_microusd
    grant.updated_at = now
    if (
        request.request_kind == "chat"
        and grant.prompt_tokens + grant.completion_tokens >= grant.token_budget
    ):
        grant.status = "exhausted"
    return deliverable


__all__ = [
    "activate_inference_grant",
    "bearer_digest",
    "InferenceDecline",
    "begin_inference_request",
    "ensure_inference_grant",
    "finish_inference_request",
    "revoke_ticket_inference",
]
