"""Ticket-scoped inference grant lifecycle."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
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
    """Create or return the one grant bound to this exact live lease."""
    if not config.enabled or ticket.status != TicketStatus.ISSUED:
        return None
    deadline = _aware(ticket.deadline)
    grant = await session.scalar(
        select(InferenceGrant)
        .where(
            InferenceGrant.agent_id == ticket.agent_id,
            InferenceGrant.bench_version == ticket.bench_version,
            InferenceGrant.validator_hotkey == ticket.validator_hotkey,
            InferenceGrant.ticket_deadline == deadline,
        )
        .with_for_update()
    )
    if grant is None:
        model = benchmark_model(ticket.bench_version)
        if model not in config.allowed_models:
            return None
        route_provider: str | None = config.provider
        route_profile: str | None = f"legacy-config-{config.provider}"
        route_quantization: str | None = None
        route_prompt_price_per_token: float | None = None
        route_completion_price_per_token: float | None = None
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
            request_count=0,
            prompt_tokens=0,
            completion_tokens=0,
            cost_microusd=0,
            active_requests=0,
            expires_at=deadline,
        )
        session.add(grant)
        await session.flush()
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
        grant.prompt_tokens += request.reserved_tokens
    bearer = secrets.token_urlsafe(32)
    grant.bearer_digest = bearer_digest(bearer)
    grant.broker_public_key = broker_public_key.rstrip("=")
    grant.generation += 1
    grant.status = "active"
    grant.slot_id = ticket.slot_id
    grant.expires_at = _aware(ticket.deadline)
    grant.active_requests = 0
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
            grant.prompt_tokens += request.reserved_tokens
        grant.status = "revoked"
        grant.active_requests = 0
        grant.updated_at = now


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
) -> tuple[InferenceGrant, InferenceRequest] | None:
    """Atomically consume one nonce and reserve bounded proxy capacity."""
    if session.get_bind().dialect.name == "postgresql":
        await session.execute(
            select(func.pg_advisory_xact_lock(func.hashtextextended("inference", 0)))
        )
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
    )
    if (
        grant is None
        or grant.status != "active"
        or grant.bearer_digest is None
        or not secrets.compare_digest(grant.bearer_digest, bearer_digest(bearer))
        or _aware(grant.expires_at) <= now
        or model not in grant.allowed_models
    ):
        return None
    stale_cutoff = now - timedelta(seconds=config.timeout_seconds * 2)
    stale_requests = list(
        (
            await session.scalars(
                select(InferenceRequest)
                .where(
                    InferenceRequest.grant_id == grant.grant_id,
                    InferenceRequest.status == "started",
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
        grant.prompt_tokens += stale.reserved_tokens
    if stale_requests:
        await session.flush()
        grant.active_requests = int(
            await session.scalar(
                select(func.count()).where(
                    InferenceRequest.grant_id == grant.grant_id,
                    InferenceRequest.status == "started",
                )
            )
            or 0
        )
    if (
        ticket is None
        or ticket.status != TicketStatus.ISSUED
        or _aware(ticket.deadline) != _aware(grant.ticket_deadline)
        or _aware(ticket.deadline) <= now
    ):
        grant.status = "revoked"
        return None
    if grant.request_count >= grant.request_budget:
        grant.status = "exhausted"
        return None
    active_reserved = await session.scalar(
        select(func.coalesce(func.sum(InferenceRequest.reserved_tokens), 0)).where(
            InferenceRequest.grant_id == grant.grant_id,
            InferenceRequest.status == "started",
        )
    )
    if (
        token_reservation < 1
        or grant.prompt_tokens
        + grant.completion_tokens
        + int(active_reserved or 0)
        + token_reservation
        > grant.token_budget
    ):
        return None
    if grant.active_requests >= config.per_ticket_concurrency:
        return None

    # Fast replay path avoids an ORM identity collision in the common case;
    # the composite primary key and nested transaction remain authoritative
    # for concurrent attempts on different platform workers.
    if await session.get(InferenceRequest, (grant.grant_id, nonce)) is not None:
        return None

    validator_active = await session.scalar(
        select(func.coalesce(func.sum(InferenceGrant.active_requests), 0)).where(
            InferenceGrant.validator_hotkey == grant.validator_hotkey,
            InferenceGrant.status == "active",
        )
    )
    global_active = await session.scalar(
        select(func.coalesce(func.sum(InferenceGrant.active_requests), 0)).where(
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
        )
    )
    ticket_recent = await session.scalar(
        select(func.count()).where(
            InferenceRequest.grant_id == grant.grant_id,
            InferenceRequest.started_at >= minute_start,
        )
    )
    global_recent = await session.scalar(
        select(func.count()).where(InferenceRequest.started_at >= minute_start)
    )
    if (
        int(validator_active or 0) >= config.per_validator_concurrency
        or int(global_active or 0) >= config.global_concurrency
        or int(ticket_recent or 0) >= config.per_ticket_requests_per_minute
        or int(validator_recent or 0) >= config.per_validator_requests_per_minute
        or int(global_recent or 0) >= config.global_requests_per_minute
    ):
        return None

    request = InferenceRequest(
        grant_id=grant.grant_id,
        nonce=nonce,
        generation=grant.generation,
        status="started",
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
    grant.request_count += 1
    grant.active_requests += 1
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
    request.timed_out = timed_out
    request.latency_ms = latency_ms
    request.completed_at = now
    if was_started:
        grant.active_requests = max(0, grant.active_requests - 1)
    grant.prompt_tokens += prompt_tokens
    grant.completion_tokens += completion_tokens
    grant.cost_microusd += cost_microusd
    grant.updated_at = now
    if grant.prompt_tokens + grant.completion_tokens >= grant.token_budget:
        grant.status = "exhausted"
    return deliverable


__all__ = [
    "activate_inference_grant",
    "bearer_digest",
    "begin_inference_request",
    "ensure_inference_grant",
    "finish_inference_request",
    "revoke_ticket_inference",
]
