"""Unit tests for :mod:`ditto.api_server.endpoints.validator`.

These cover the STUBBED validator endpoints: the wire contract (shapes,
status codes, error envelopes) the validator daemon connects against. No
database / storage is touched — the handlers return synthetic data — so the
tests assert the contract, not persistence. Persistence + real auth land in
the real-next pass and get their own (integration) coverage.
"""

from __future__ import annotations

import httpx

from ditto.api_models.agent_status import AgentStatus
from ditto.api_server.middleware.error_envelope import (
    ERROR_CODE_AGENT_NOT_FOUND,
    ERROR_CODE_VALIDATION,
    ERROR_CODE_VALIDATOR_AUTH,
)

_VALIDATOR_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
_MINER_HOTKEY = "5DhaT8U7LVwnnJNUU8VL1XEipicatoaDVVq7cHo227gogVZm"
_UNKNOWN_AGENT = "00000000-0000-0000-0000-000000000000"


def _score_payload(run_id: str = "run_test_1") -> dict:
    return {
        "validator_hotkey": _VALIDATOR_HOTKEY,
        "signature": "ab" * 64,
        "report": {
            "run_id": run_id,
            "seed": 8675309,
            "composite": 0.82,
            "tool_mean": 0.88,
            "memory_mean": 0.73,
            "median_ms": 812,
            "n": 30,
            "generated_at": "2026-06-08T12:04:30Z",
            "per_case": [],
        },
    }


class TestQueue:
    async def test_returns_oldest_first_items(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/validator/queue")
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        body = response.json()
        assert body["count"] == len(body["items"]) >= 1
        first = body["items"][0]
        assert first["miner_hotkey"] == _MINER_HOTKEY
        assert first["status"] == AgentStatus.SCREENING_PASSED
        # Synthetic sha256 must satisfy the documented 64-hex shape.
        assert len(first["sha256"]) == 64

    async def test_limit_caps_item_count(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/validator/queue?limit=1")
        assert response.status_code == 200
        assert response.json()["count"] == 1

    async def test_limit_out_of_range_returns_422(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.get("/api/v1/validator/queue?limit=0")
        assert response.status_code == 422
        assert response.json()["error_code"] == ERROR_CODE_VALIDATION

    async def test_malformed_validator_hotkey_header_returns_401(
        self, client: httpx.AsyncClient
    ) -> None:
        """The auth seam rejects a present-but-malformed hotkey header."""
        response = await client.get(
            "/api/v1/validator/queue",
            headers={"X-Validator-Hotkey": "not-an-ss58"},
        )
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_VALIDATOR_AUTH

    async def test_well_formed_validator_hotkey_header_passes(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.get(
            "/api/v1/validator/queue",
            headers={"X-Validator-Hotkey": _VALIDATOR_HOTKEY},
        )
        assert response.status_code == 200


class TestArtifact:
    async def test_returns_download_url(self, client: httpx.AsyncClient) -> None:
        agent_id = "550e8400-e29b-41d4-a716-446655440000"
        response = await client.get(f"/api/v1/validator/agent/{agent_id}/artifact")
        assert response.status_code == 200
        body = response.json()
        assert body["agent_id"] == agent_id
        assert body["download_url"].startswith("https://")
        assert "expires_at" in body

    async def test_unknown_agent_returns_404(self, client: httpx.AsyncClient) -> None:
        response = await client.get(
            f"/api/v1/validator/agent/{_UNKNOWN_AGENT}/artifact"
        )
        assert response.status_code == 404
        assert response.json()["error_code"] == ERROR_CODE_AGENT_NOT_FOUND

    async def test_malformed_uuid_returns_422(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/validator/agent/not-a-uuid/artifact")
        assert response.status_code == 422
        assert response.json()["error_code"] == ERROR_CODE_VALIDATION


class TestSubmitScore:
    async def test_accepts_and_echoes_scored(self, client: httpx.AsyncClient) -> None:
        agent_id = "550e8400-e29b-41d4-a716-446655440000"
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["agent_id"] == agent_id
        assert body["status"] == AgentStatus.SCORED
        assert body["accepted"] is True

    async def test_unknown_agent_returns_404(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            f"/api/v1/validator/agent/{_UNKNOWN_AGENT}/score",
            json=_score_payload(),
        )
        assert response.status_code == 404
        assert response.json()["error_code"] == ERROR_CODE_AGENT_NOT_FOUND

    async def test_out_of_range_composite_returns_422(
        self, client: httpx.AsyncClient
    ) -> None:
        """Field bounds (composite in [0,1]) are enforced even in stub mode."""
        agent_id = "550e8400-e29b-41d4-a716-446655440000"
        payload = _score_payload()
        payload["report"]["composite"] = 1.5
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=payload,
        )
        assert response.status_code == 422
        assert response.json()["error_code"] == ERROR_CODE_VALIDATION

    async def test_malformed_validator_hotkey_in_body_returns_422(
        self, client: httpx.AsyncClient
    ) -> None:
        agent_id = "550e8400-e29b-41d4-a716-446655440000"
        payload = _score_payload()
        payload["validator_hotkey"] = "not-an-ss58"
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=payload,
        )
        assert response.status_code == 422
        assert response.json()["error_code"] == ERROR_CODE_VALIDATION
