"""HTTP contracts for adaptive inference policy and route admission."""

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ditto.api_server.dependencies import get_session
from ditto.api_server.inference_routing import aggregate_profile_revision
from ditto.db.models import (
    Base,
    InferenceProviderRoute,
    InferenceRequest,
    InferenceRoutingPolicy,
)

pytestmark = pytest.mark.asyncio

_TOKEN = "test-admin-token-at-least-32-characters"
_HEADERS = {"Authorization": f"Bearer {_TOKEN}", "X-Admin-Actor": "operator"}
_MODEL = "openai/gpt-oss-20b"
_PROFILE = "openrouter-route-test-v1"


@pytest.fixture
async def route_maker() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def _install(
    app: FastAPI,
    maker: async_sessionmaker[AsyncSession],
    *,
    routing_mode: str = "adaptive",
) -> None:
    app.state.config = replace(
        app.state.config,
        admin_api_token=_TOKEN,
        inference_proxy=replace(
            app.state.config.inference_proxy,
            routing_mode=routing_mode,
            reviewed_calibration_manifest_sha256="a" * 64,
        ),
    )

    async def _session() -> AsyncIterator[AsyncSession]:
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = _session
    now = datetime.now(UTC)
    async with maker() as session, session.begin():
        session.add(
            InferenceRoutingPolicy(
                model=_MODEL,
                enabled=False,
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
        session.add(
            InferenceProviderRoute(
                model=_MODEL,
                provider="Groq",
                profile_revision=_PROFILE,
                status="healthy",
                calibration_status="shadow",
                quantization="fp8",
                prompt_price_per_token=0.000000075,
                completion_price_per_token=0.0000003,
                ewma_error_rate=0,
                ewma_timeout_rate=0,
                sample_count=0,
                selected_ticket_count=0,
                exploration_ticket_count=0,
                discovered_at=now,
                updated_at=now,
            )
        )


def _policy_payload() -> dict[str, object]:
    return {
        "enabled": True,
        "expected_revision": 0,
        "speed_weight": 0.65,
        "cost_weight": 0.25,
        "exploration_weight": 0.10,
        "exploration_ticket_budget": 3,
        "min_tool_accuracy": 0.55,
        "min_composite": 0.15,
        "min_calibration_samples": 20,
        "max_error_rate": 0.25,
        "max_timeout_rate": 0.15,
        "cooldown_seconds": 30,
        "ewma_alpha": 0.20,
        "confirmation": f"UPDATE INFERENCE POLICY {_MODEL}",
    }


async def test_lists_and_updates_complete_model_policy(
    app: FastAPI,
    client: httpx.AsyncClient,
    route_maker: async_sessionmaker[AsyncSession],
) -> None:
    await _install(app, route_maker)
    listing = await client.get("/api/v1/admin/inference-routes", headers=_HEADERS)
    assert listing.status_code == 200
    assert listing.headers["Cache-Control"] == "no-store"
    assert listing.json()["policies"][0]["enabled"] is False
    assert listing.json()["routes"][0]["profile_revision"] == _PROFILE

    response = await client.put(
        f"/api/v1/admin/inference-routes/policy/{_MODEL}",
        headers=_HEADERS,
        json=_policy_payload(),
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"model": _MODEL, "enabled": True, "revision": 1}
    stale = await client.put(
        f"/api/v1/admin/inference-routes/policy/{_MODEL}",
        headers=_HEADERS,
        json=_policy_payload(),
    )
    assert stale.status_code == 409
    audited = await client.get("/api/v1/admin/inference-routes", headers=_HEADERS)
    assert audited.json()["audits"][0]["action"] == "policy_updated"
    assert audited.json()["audits"][0]["actor"] == "operator"


async def test_route_admission_requires_exact_confirmation_and_quality_floor(
    app: FastAPI,
    client: httpx.AsyncClient,
    route_maker: async_sessionmaker[AsyncSession],
) -> None:
    await _install(app, route_maker)
    payload = {
        "model": _MODEL,
        "provider": "Groq",
        "expected_revision": 0,
        "action": "eligible",
        "manifest_sha256": "a" * 64,
        "tool_accuracy": 0.65,
        "composite": 0.20,
        "sample_count": 60,
        "confirmation": "wrong",
    }
    rejected = await client.post(
        f"/api/v1/admin/inference-routes/{_PROFILE}/calibration",
        headers=_HEADERS,
        json=payload,
    )
    assert rejected.status_code == 409

    payload["confirmation"] = f"ELIGIBLE INFERENCE ROUTE {_PROFILE}"
    payload["manifest_sha256"] = "b" * 64
    unreviewed = await client.post(
        f"/api/v1/admin/inference-routes/{_PROFILE}/calibration",
        headers=_HEADERS,
        json=payload,
    )
    assert unreviewed.status_code == 409
    assert "deployed reviewed artifact" in unreviewed.json()["message"]

    payload["manifest_sha256"] = "a" * 64
    accepted = await client.post(
        f"/api/v1/admin/inference-routes/{_PROFILE}/calibration",
        headers=_HEADERS,
        json=payload,
    )
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["calibration_status"] == "eligible"
    assert accepted.json()["calibration_revision"] == 1
    audited = await client.get("/api/v1/admin/inference-routes", headers=_HEADERS)
    assert audited.json()["audits"][0]["action"] == "route_eligible"


async def test_aggregate_mode_blocks_adaptive_controls_but_allows_logical_route(
    app: FastAPI,
    client: httpx.AsyncClient,
    route_maker: async_sessionmaker[AsyncSession],
) -> None:
    await _install(app, route_maker, routing_mode="aggregate_throughput")
    profile = aggregate_profile_revision(_MODEL)
    async with route_maker() as session, session.begin():
        session.add(
            InferenceProviderRoute(
                model=_MODEL,
                provider="openrouter",
                profile_revision=profile,
                status="healthy",
                calibration_status="shadow",
                ewma_error_rate=0,
                ewma_timeout_rate=0,
                sample_count=0,
                selected_ticket_count=0,
                exploration_ticket_count=0,
                discovered_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
        session.add(
            InferenceRequest(
                grant_id=uuid4(),
                nonce=uuid4(),
                generation=1,
                status="completed",
                model=_MODEL,
                reserved_tokens=100,
                prompt_tokens=80,
                completion_tokens=20,
                cost_microusd=123,
                upstream_provider="WandB",
                timed_out=False,
                latency_ms=250,
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
            )
        )
    listing = await client.get("/api/v1/admin/inference-routes", headers=_HEADERS)
    assert listing.json()["routing_mode"] == "aggregate_throughput"
    assert listing.json()["aggregate_route"] == {
        "model": _MODEL,
        "provider": "openrouter",
        "profile_revision": profile,
        "provider_sort": "throughput",
        "allow_fallbacks": True,
    }
    assert listing.json()["provider_telemetry"] == [
        {
            "provider": "WandB",
            "request_count": 1,
            "completed_count": 1,
            "timeout_count": 0,
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "cost_microusd": 123,
            "average_latency_ms": 250.0,
        }
    ]
    blocked = await client.put(
        f"/api/v1/admin/inference-routes/policy/{_MODEL}",
        headers=_HEADERS,
        json=_policy_payload(),
    )
    assert blocked.status_code == 409
    provider_payload = {
        "model": _MODEL,
        "provider": "Groq",
        "expected_revision": 0,
        "action": "eligible",
        "manifest_sha256": "a" * 64,
        "tool_accuracy": 0.65,
        "composite": 0.20,
        "sample_count": 60,
        "confirmation": f"ELIGIBLE INFERENCE ROUTE {_PROFILE}",
    }
    blocked_provider = await client.post(
        f"/api/v1/admin/inference-routes/{_PROFILE}/calibration",
        headers=_HEADERS,
        json=provider_payload,
    )
    assert blocked_provider.status_code == 409
    provider_payload.update(
        {
            "provider": "openrouter",
            "confirmation": f"ELIGIBLE INFERENCE ROUTE {profile}",
        }
    )
    provider_payload["sample_count"] = 20
    incomplete = await client.post(
        f"/api/v1/admin/inference-routes/{profile}/calibration",
        headers=_HEADERS,
        json=provider_payload,
    )
    assert incomplete.status_code == 409
    provider_payload["sample_count"] = 60
    admitted = await client.post(
        f"/api/v1/admin/inference-routes/{profile}/calibration",
        headers=_HEADERS,
        json=provider_payload,
    )
    assert admitted.status_code == 200, admitted.text
