from __future__ import annotations

import json

import httpx
import pytest

from ditto.api_server.endpoints import public

_VALIDATOR = "5HmP9732JFjnut2RY9yg4Gz2qJ38vF8xFwZb5dQVPF7FsmZz"


async def test_latest_wandb_logs_url_filters_by_full_hotkey() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url == public._WANDB_GRAPHQL_URL
        assert payload["variables"] == {
            "entity": "heyditto",
            "project": "ditto-sn118",
            "filters": f'{{"config.validator_hotkey":"{_VALIDATOR}"}}',
        }
        return httpx.Response(
            200,
            json={
                "data": {
                    "project": {"runs": {"edges": [{"node": {"name": "h0j4iiss"}}]}}
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as wandb:
        url = await public._latest_wandb_logs_url(
            "https://wandb.ai/heyditto/ditto-sn118", _VALIDATOR, client=wandb
        )

    assert url == "https://wandb.ai/heyditto/ditto-sn118/runs/h0j4iiss/logs"


async def test_validator_wandb_logs_redirects_to_latest_run(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def lookup(project_url: str, validator_hotkey: str) -> str:
        assert project_url == "https://wandb.ai/"
        assert validator_hotkey == _VALIDATOR
        return "https://wandb.ai/heyditto/ditto-sn118/runs/h0j4iiss/logs"

    monkeypatch.setattr(public, "_latest_wandb_logs_url", lookup)
    response = await client.get(
        f"/api/v1/public/validators/{_VALIDATOR}/wandb-logs",
        follow_redirects=False,
    )

    assert response.status_code == 307
    assert response.headers["location"].endswith("/runs/h0j4iiss/logs")
    assert response.headers["cache-control"] == "no-store"


async def test_validator_wandb_logs_rejects_invalid_hotkey(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get(
        "/api/v1/public/validators/not-a-hotkey/wandb-logs",
        follow_redirects=False,
    )

    assert response.status_code == 404
