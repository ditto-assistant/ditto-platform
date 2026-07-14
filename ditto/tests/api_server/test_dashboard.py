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

    async def test_includes_social_preview_metadata(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        image_url = "https://platform-api.heyditto.ai/assets/paperditto-512.png"
        assert '<meta property="og:type" content="website"' in body
        assert (
            '<meta property="og:title" content="Ditto SN118 · Subnet Leaderboard"'
            in body
        )
        assert f'<meta property="og:image" content="{image_url}"' in body
        assert '<meta property="og:image:width" content="512"' in body
        assert '<meta name="twitter:card" content="summary"' in body
        assert f'<meta name="twitter:image" content="{image_url}"' in body
        assert '<link rel="canonical" href="https://platform-api.heyditto.ai/"' in body

    async def test_serves_social_preview_image(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        resp = await _get(app, "/assets/paperditto-512.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.headers["Cache-Control"] == "public, max-age=86400"
        assert resp.content.startswith(b"\x89PNG\r\n\x1a\n")

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
            "/api/v1/public/screeners",
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
        assert body.index("<h2>Leaderboard</h2>") < body.index('class="operations"')
        assert 'getJSON("/public/activity?page="' in body
        assert 'id="activity-rows"' in body
        assert 'id="activity-pager"' in body
        assert 'id="pipeline-overview"' in body
        assert "Network operations" in body
        assert "Waiting for screening" in body
        assert "Waiting for scores" in body
        assert "Evaluating" in body
        assert 'id="pipeline-scored"' in body
        assert 'data-pipeline-stage="scored"' in body
        assert 'getJSON("/public/activity?page=1&limit=200")' in body
        assert body.count('type="button" data-activity-page="prev"') == 2
        assert body.count('type="button" data-activity-page="next"') == 2
        assert 'aria-label="Submission pages, bottom"' in body
        assert 'class="activity-table-frame"' in body
        assert "lockActivityFrameHeight" in body
        assert "anchor.getBoundingClientRect().top - anchorTop" in body
        assert "Validation" in body
        assert "openActivityModal" in body
        assert "scores received" in body
        assert "validator is assigned; its score is pending" in body
        assert "Score pending" in body
        assert "has this assignment" in body
        assert "Copy review:" in body
        assert "screening_reason" in body
        assert '<details class="old-screeners">' in body
        assert "Old screener results" in body
        assert "Screener result" in body
        assert "Lease expired" not in body
        assert "System failure" not in body
        assert 'expired: ["Assignment expired", "warn"]' in body
        assert 'failed: ["Could not complete", "warn"]' in body
        assert 'class="pipeline-summary"' in body
        assert 'class="pipeline-key-facts"' in body
        assert 'class="pipeline-meta-list"' in body
        assert 'class="pipeline-history"' in body
        assert 'class="pipeline-detail-state"' in body
        assert 'style="margin-top:18px">Validator progress' not in body

    async def test_includes_accessible_fleet_status(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert "Fleet health" in body
        assert 'id="fleet-summary"' in body
        assert 'id="fleet-rows"' in body
        assert 'id="show-screeners"' in body
        assert 'type="checkbox"' in body
        assert '<label class="fleet-toggle" for="show-screeners">' in body
        assert '<table class="fleet-table"' in body
        assert '<th scope="col" style="width:105px">First seen</th>' in body
        assert '<th scope="col" style="width:110px">Last heartbeat</th>' in body
        assert '<th scope="col" style="width:118px">Status</th>' in body
        assert '<th scope="col" style="width:108px">CPU</th>' in body
        assert '<th scope="col" style="width:120px">Containers</th>' in body
        assert "Missing optional telemetry is not an outage." in body
        assert "allowlisted" not in body
        assert 'id="fleet-count-unknown"' in body
        assert 'getJSON("/public/validators")' in body
        assert 'getJSON("/public/screeners")' in body
        assert 'getElementById("show-screeners").addEventListener' in body
        assert 'showScreeners ? "Screener" : "Validator"' in body
        assert "running_benchmark" in body
        assert "metrics.cpu_percent >= 95" not in body
        assert 'fleetMeter(metrics.cpu_percent, "")' in body
        assert "privacy-note" not in body
        assert "fleet-health-note" not in body
        assert '" reporting " + kind' not in body

    async def test_includes_accessible_benchmark_progress(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert "benchmarkStageLabel" in body
        assert "active_benchmarks" in body
        assert "active_benchmark" in body
        assert '<progress max="100" value="' in body
        assert "aria-label" in body
        assert "Benchmark progress not reported" in body
        assert "failed_retrying" in body
        assert "Scoring and finalizing" in body
        assert "Signing and submitting result" in body
        assert "prefers-reduced-motion" in body
        assert "@media (forced-colors: active)" in body
        assert "@media (max-width: 720px)" in body
        assert 'class="fleet-work-col"' in body
        assert "Current work" in body
        assert "screenerStageLabel" in body
        assert "screening_progress" in body
        assert "Building image" in body
        assert "elapsedDuration" in body
        assert "data-started-at" in body
        assert "active_agent_name" in body
        assert "setInterval(updateScreenerElapsed, 1000)" in body

    async def test_includes_system_and_time_aware_theme_switcher(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert 'data-theme-choice="system"' in body
        assert 'data-theme-choice="light"' in body
        assert 'data-theme-choice="dark"' in body
        assert 'data-theme-choice="time"' in body
        assert 'var STORAGE_KEY = "ditto:dashboard-theme"' in body
        assert 'return MODES[saved] ? saved : "system"' in body
        assert 'window.matchMedia("(prefers-color-scheme: dark)")' in body
        assert "root.dataset.systemTheme" in body
        assert "root.dataset.timePhase = fromHour(new Date().getHours())" in body
        assert 'if (hour >= 5 && hour < 8) return "dawn"' in body
        assert ".side-theme { flex: 1 0 100%; width: 100%; }" in body
        assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in body
        assert ".side-theme .theme-option { min-height: 44px;" in body

    async def test_sidebar_shell_routes_every_section(self) -> None:
        # The dashboard is a sidebar shell with hash-routed pages; the theme
        # switcher moved into the sidebar and the leaderboard is consolidated
        # onto the Overview page (no separate Leaderboard tab).
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert '<aside class="sidebar"' in body
        for page in ("overview", "operations", "submissions", "benchmark"):
            assert f'href="#/{page}"' in body
            assert f'data-page="{page}"' in body
        assert 'href="#/leaderboard"' not in body
        assert "<h2>Leaderboard</h2>" in body  # folded into Overview
        assert 'data-theme-choice="system"' in body  # switcher still wired

    async def test_benchmark_badge_omits_latest_suffix(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert 'badge.textContent = "DittoBench v" + currentBench +' in body
        assert 'currentBench + " · latest"' not in body
