from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.ticket_status import TicketStatus
from ditto.api_server.config import InferenceProxyConfig
from ditto.db.models import (
    Agent,
    AgentStatus,
    InferenceProviderRoute,
    InferenceRoutingPolicy,
    ValidatorTicket,
)
from ditto.db.queries.inference import (
    activate_inference_grant,
    begin_inference_request,
    ensure_inference_grant,
    finish_inference_request,
    revoke_ticket_inference,
)


def _config() -> InferenceProxyConfig:
    return InferenceProxyConfig(
        enabled=True,
        required=False,
        public_base_url="https://platform.example",
        openrouter_api_key="test-key",
        upstream_url="https://openrouter.ai/api/v1/chat/completions",
        allowed_models=("qwen/qwen3-32b",),
        provider="nebius",
        routing_mode="adaptive",
        request_budget=2,
        token_budget=100,
        embedding_upstream_url="https://openrouter.ai/api/v1/embeddings",
        embedding_model="perplexity/pplx-embed-v1-0.6b",
        embedding_profile="dittobench-v7-openrouter-pplx-embed-v1-0.6b-768-v1",
        embedding_provider="Perplexity",
        embedding_dimensions=768,
        embedding_request_budget=100_000,
        embedding_token_budget=1_000_000_000,
        embedding_per_ticket_concurrency=1,
        embedding_per_validator_concurrency=8,
        embedding_global_concurrency=32,
        embedding_per_ticket_requests_per_minute=10_000,
        embedding_per_validator_requests_per_minute=40_000,
        embedding_global_requests_per_minute=100_000,
        embedding_request_body_bytes=1 << 20,
        embedding_response_body_bytes=16 << 20,
        per_ticket_concurrency=1,
        per_validator_concurrency=1,
        global_concurrency=1,
        per_ticket_requests_per_minute=2,
        per_validator_requests_per_minute=2,
        global_requests_per_minute=2,
        request_body_bytes=1024,
        response_body_bytes=1024,
        timeout_seconds=10,
        max_output_tokens=32,
    )


async def _live_grant(session: AsyncSession):
    now = datetime.now(UTC)
    agent = Agent(
        agent_id=uuid4(),
        miner_hotkey="miner",
        name="parallel-inference",
        sha256="ab" * 32,
        status=AgentStatus.EVALUATING,
        created_at=now,
    )
    ticket = ValidatorTicket(
        agent_id=agent.agent_id,
        validator_hotkey="validator",
        slot_id="slot-0",
        status=TicketStatus.ISSUED,
        issued_at=now,
        deadline=now + timedelta(minutes=20),
        bench_version=5,
        attempt_count=1,
    )
    session.add_all([agent, ticket])
    await session.flush()
    grant = await ensure_inference_grant(session, ticket=ticket, config=_config())
    assert grant is not None
    assert (
        await activate_inference_grant(
            session,
            grant_id=grant.grant_id,
            validator_hotkey="wrong-validator",
            broker_public_key="broker-key",
            now=now,
            config=_config(),
        )
        is None
    )
    activated = await activate_inference_grant(
        session,
        grant_id=grant.grant_id,
        validator_hotkey="validator",
        broker_public_key="broker-key",
        now=now,
        config=_config(),
    )
    assert activated is not None
    return ticket, activated[0], activated[1], now


@pytest.mark.asyncio
async def test_v7_grant_requires_and_binds_one_calibrated_dynamic_route(
    session: AsyncSession,
) -> None:
    now = datetime.now(UTC)
    config = replace(_config(), allowed_models=("qwen/qwen3-32b", "openai/gpt-oss-20b"))
    async with session.begin():
        agent = Agent(
            agent_id=uuid4(),
            miner_hotkey="miner-v7",
            name="adaptive-route",
            sha256="cd" * 32,
            status=AgentStatus.EVALUATING,
            created_at=now,
        )
        ticket = ValidatorTicket(
            agent_id=agent.agent_id,
            validator_hotkey="validator-v7",
            slot_id="slot-0",
            status=TicketStatus.ISSUED,
            issued_at=now,
            deadline=now + timedelta(minutes=20),
            bench_version=7,
            attempt_count=1,
        )
        session.add_all([agent, ticket])
        await session.flush()
        assert (
            await ensure_inference_grant(session, ticket=ticket, config=config) is None
        )

        route = InferenceProviderRoute(
            model="openai/gpt-oss-20b",
            provider="discovered-provider",
            profile_revision="openrouter-route-test-v1",
            status="healthy",
            calibration_status="eligible",
            prompt_price_per_token=0.00000003,
            completion_price_per_token=0.00000013,
            ewma_tokens_per_second=150,
            ewma_latency_ms=900,
            ewma_error_rate=0,
            ewma_timeout_rate=0,
            calibration_tool_accuracy=0.65,
            calibration_composite=0.20,
            calibration_sample_count=60,
            calibration_manifest_sha256="ab" * 32,
            sample_count=20,
            discovered_at=now,
        )
        session.add(route)
        session.add(
            InferenceRoutingPolicy(
                model="openai/gpt-oss-20b",
                enabled=True,
                speed_weight=0.65,
                cost_weight=0.25,
                exploration_weight=0.10,
                exploration_ticket_budget=3,
                min_tool_accuracy=0.55,
                min_composite=0.15,
                min_calibration_samples=20,
                max_error_rate=0.25,
                max_timeout_rate=0.15,
                cooldown_seconds=30,
                ewma_alpha=0.20,
                updated_at=now,
            )
        )
        await session.flush()
        grant = await ensure_inference_grant(session, ticket=ticket, config=config)
        assert grant is not None
        assert grant.allowed_models == ["openai/gpt-oss-20b"]
        assert grant.route_provider == "discovered-provider"
        assert grant.route_profile == "openrouter-route-test-v1"
        assert route.selected_ticket_count == 1
        assert route.exploration_ticket_count == 1
        activated = await activate_inference_grant(
            session,
            grant_id=grant.grant_id,
            validator_hotkey=ticket.validator_hotkey,
            broker_public_key="broker-key",
            now=now,
            config=config,
        )
        assert activated is not None
        bearer = activated[1]
        nonce = uuid4()
        started = await begin_inference_request(
            session,
            grant_id=grant.grant_id,
            nonce=nonce,
            bearer=bearer,
            model=config.embedding_model,
            token_reservation=1_000_000,
            now=now,
            config=config,
            request_kind="embedding",
        )
        assert started
        request = started[1]
        assert await finish_inference_request(
            session,
            grant_id=grant.grant_id,
            nonce=nonce,
            generation=grant.generation,
            status="completed",
            prompt_tokens=250_000,
            completion_tokens=0,
            cost_microusd=1_000,
            usage_available=True,
            now=now,
            upstream_provider=config.embedding_provider,
            upstream_attempts=3,
        )
        assert grant.embedding_request_count == 1
        assert grant.embedding_tokens == 250_000
        assert grant.embedding_cost_microusd == 1_000
        assert grant.request_count == 0
        assert grant.prompt_tokens == 0
        assert request.upstream_attempts == 3


@pytest.mark.asyncio
async def test_grant_rejects_wrong_bearer_model_budget_and_replay(
    session: AsyncSession,
) -> None:
    async with session.begin():
        _ticket, grant, bearer, now = await _live_grant(session)
        assert (
            await begin_inference_request(
                session,
                grant_id=grant.grant_id,
                nonce=uuid4(),
                bearer="stolen-sibling-bearer",
                model="qwen/qwen3-32b",
                token_reservation=10,
                now=now,
                config=_config(),
            )
            is None
        )
        assert (
            await begin_inference_request(
                session,
                grant_id=grant.grant_id,
                nonce=uuid4(),
                bearer=bearer,
                model="not-allowed",
                token_reservation=10,
                now=now,
                config=_config(),
            )
            is None
        )
        assert (
            await begin_inference_request(
                session,
                grant_id=grant.grant_id,
                nonce=uuid4(),
                bearer=bearer,
                model="qwen/qwen3-32b",
                token_reservation=101,
                now=now,
                config=_config(),
            )
            is None
        )
        nonce = uuid4()
        accepted = await begin_inference_request(
            session,
            grant_id=grant.grant_id,
            nonce=nonce,
            bearer=bearer,
            model="qwen/qwen3-32b",
            token_reservation=10,
            now=now,
            config=_config(),
        )
        assert accepted is not None
        await finish_inference_request(
            session,
            grant_id=grant.grant_id,
            nonce=nonce,
            generation=grant.generation,
            status="completed",
            prompt_tokens=3,
            completion_tokens=4,
            cost_microusd=5,
            usage_available=True,
            now=now,
        )
        assert (
            await begin_inference_request(
                session,
                grant_id=grant.grant_id,
                nonce=nonce,
                bearer=bearer,
                model="qwen/qwen3-32b",
                token_reservation=10,
                now=now,
                config=_config(),
            )
            is None
        )


@pytest.mark.asyncio
async def test_canceled_or_expired_ticket_revokes_capability(
    session: AsyncSession,
) -> None:
    async with session.begin():
        ticket, grant, bearer, now = await _live_grant(session)
        await revoke_ticket_inference(session, ticket=ticket, now=now)
        ticket.status = TicketStatus.EXPIRED
        assert (
            await begin_inference_request(
                session,
                grant_id=grant.grant_id,
                nonce=uuid4(),
                bearer=bearer,
                model="qwen/qwen3-32b",
                token_reservation=10,
                now=now,
                config=_config(),
            )
            is None
        )


@pytest.mark.asyncio
async def test_revocation_cancels_inflight_and_missing_usage_charges_reservation(
    session: AsyncSession,
) -> None:
    async with session.begin():
        ticket, grant, bearer, now = await _live_grant(session)
        nonce = uuid4()
        assert await begin_inference_request(
            session,
            grant_id=grant.grant_id,
            nonce=nonce,
            bearer=bearer,
            model="qwen/qwen3-32b",
            token_reservation=10,
            now=now,
            config=_config(),
        )
        await revoke_ticket_inference(session, ticket=ticket, now=now)
        ticket.status = TicketStatus.EXPIRED
        assert not await finish_inference_request(
            session,
            grant_id=grant.grant_id,
            nonce=nonce,
            generation=grant.generation,
            status="completed",
            prompt_tokens=0,
            completion_tokens=0,
            cost_microusd=0,
            usage_available=False,
            now=now,
        )

    async with session.begin():
        _ticket, grant, bearer, now = await _live_grant(session)
        nonce = uuid4()
        assert await begin_inference_request(
            session,
            grant_id=grant.grant_id,
            nonce=nonce,
            bearer=bearer,
            model="qwen/qwen3-32b",
            token_reservation=10,
            now=now,
            config=_config(),
        )
        assert not await finish_inference_request(
            session,
            grant_id=grant.grant_id,
            nonce=nonce,
            generation=grant.generation,
            status="completed",
            prompt_tokens=0,
            completion_tokens=0,
            cost_microusd=0,
            usage_available=False,
            now=now,
        )
        assert grant.prompt_tokens == 10


@pytest.mark.asyncio
async def test_ticket_request_rate_is_bounded_after_requests_finish(
    session: AsyncSession,
) -> None:
    config = replace(
        _config(),
        request_budget=10,
        per_ticket_requests_per_minute=1,
        per_validator_requests_per_minute=10,
        global_requests_per_minute=10,
    )
    async with session.begin():
        _ticket, grant, bearer, now = await _live_grant(session)
        first_nonce = uuid4()
        assert await begin_inference_request(
            session,
            grant_id=grant.grant_id,
            nonce=first_nonce,
            bearer=bearer,
            model="qwen/qwen3-32b",
            token_reservation=10,
            now=now,
            config=config,
        )
        assert await finish_inference_request(
            session,
            grant_id=grant.grant_id,
            nonce=first_nonce,
            generation=grant.generation,
            status="completed",
            prompt_tokens=2,
            completion_tokens=1,
            cost_microusd=0,
            usage_available=True,
            now=now,
        )
        assert (
            await begin_inference_request(
                session,
                grant_id=grant.grant_id,
                nonce=uuid4(),
                bearer=bearer,
                model="qwen/qwen3-32b",
                token_reservation=10,
                now=now,
                config=config,
            )
            is None
        )
