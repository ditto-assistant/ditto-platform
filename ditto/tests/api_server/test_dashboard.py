"""Tests for the same-origin dashboard SPA served at ``/``.

The platform doubles as the transparency front door: it serves
``dashboard/index.html`` at ``/`` so the SPA's ``/api/v1/public/*`` calls are
same-origin (no CORS). The served HTML must carry the injected wandb project URL
and be suppressible via config.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from ditto.api_server.factory import create_api_server

from .conftest import make_api_server_config


async def _get(app: FastAPI, path: str) -> httpx.Response:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


class TestDashboard:
    async def test_served_at_root_with_injected_wandb_url(self) -> None:
        url = "https://wandb.ai/ditto/ditto-sn118"
        app = create_api_server(
            make_api_server_config(dashboard_enabled=True, dashboard_wandb_url=url)
        )
        resp = await _get(app, "/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert resp.headers["Cache-Control"] == "public, max-age=300"
        body = resp.text
        # The wandb link is injected into the meta tag the SPA reads.
        assert f'content="{url}"' in body
        assert 'name="ditto:wandb-url"' in body
        # api-base stays empty so the SPA uses its same-origin /api/v1 default.
        assert 'name="ditto:api-base" content=""' in body

    async def test_wandb_url_is_html_escaped(self) -> None:
        # A stray quote in the configured URL must not break out of the attribute.
        app = create_api_server(
            make_api_server_config(
                dashboard_enabled=True,
                dashboard_wandb_url='https://wandb.ai/"><script>x',
            )
        )
        body = (await _get(app, "/")).text
        assert "<script>x" not in body
        assert "&lt;script&gt;x" in body

    async def test_disabled_returns_404(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=False))
        resp = await _get(app, "/")
        assert resp.status_code == 404

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/public/leaderboard",
            "/api/v1/public/activity",
            "/api/v1/public/validators",
        ],
    )
    async def test_api_still_mounted_alongside_dashboard(self, path: str) -> None:
        # Serving HTML at / must not shadow the API routes.
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        # 200 requires a DB; here we only assert the route exists (not 404).
        assert any(getattr(r, "path", None) == path for r in app.routes)

    async def test_includes_submission_pipeline(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert "Submission pipeline" in body
        assert body.index("<h2>Leaderboard</h2>") < body.index(
            "<h2>Submission pipeline</h2>"
        )
        assert 'getJSON("/public/activity?page="' in body
        assert 'id="activity-rows"' in body
        assert 'id="activity-pager"' in body
        assert 'id="pipeline-overview"' in body
        assert "Waiting for screening" in body
        assert "Waiting for validator" in body
        assert 'getJSON("/public/activity?page=1&limit=200")' in body
        assert "Validation" in body
        assert "openActivityModal" in body
        assert "validators scored this submission" in body
        assert "Copy review:" in body
        assert "screening_reason" in body

    async def test_includes_validator_fleet_status(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert "Validator fleet" in body
        assert 'id="validator-summary"' in body
        assert 'id="validator-rows"' in body
        assert 'getJSON("/public/validators")' in body
        assert "running_benchmark" in body
