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
        assert resp.headers["Cache-Control"] == "public, max-age=60, must-revalidate"
        body = resp.text
        # The wandb link is injected into the meta tag the SPA reads.
        assert f'content="{url}"' in body
        assert 'name="ditto:wandb-url"' in body
        # api-base stays empty so the SPA uses its same-origin /api/v1 default.
        assert 'name="ditto:api-base" content=""' in body

    @pytest.mark.parametrize(
        "path",
        [
            "/agent/6c10d0df-fc93-4903-a939-147d51cea1cc",
            "/miner/5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            "/agents/6c10d0df-fc93-4903-a939-147d51cea1cc",
            "/miners/5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            "/validators/5DhaT8U7LVwnnJNUU8VL1XEipicatoaDVVq7cHo227gogVZm",
            "/screeners/5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        ],
    )
    async def test_serves_dashboard_at_entity_paths(self, path: str) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        resp = await _get(app, path)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert resp.headers["Cache-Control"] == "public, max-age=60, must-revalidate"
        assert '<h1 id="page-title">Overview</h1>' in resp.text

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

    async def test_dashboard_html_is_gzip_encoded(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Accept-Encoding": "gzip"})
        assert resp.status_code == 200
        assert resp.headers["content-encoding"] == "gzip"
        # httpx transparently decodes; the decoded HTML is still the SPA shell.
        assert '<h1 id="page-title">Overview</h1>' in resp.text

    async def test_dashboard_serves_strong_etag(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        resp = await _get(app, "/")
        etag = resp.headers.get("etag")
        assert etag is not None
        assert etag.startswith('"') and etag.endswith('"')

    async def test_dashboard_if_none_match_returns_304_empty_body(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            first = await client.get("/")
            etag = first.headers["etag"]
            second = await client.get("/", headers={"If-None-Match": etag})
        assert second.status_code == 304
        assert second.content == b""
        assert second.headers["etag"] == etag
        assert second.headers["Cache-Control"] == "public, max-age=60, must-revalidate"
        # RFC 9110 §15.4.5: the 304 repeats the 200's Vary: Accept-Encoding.
        assert second.headers["Vary"] == "Accept-Encoding"

    async def test_dashboard_entity_path_if_none_match_returns_304(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        path = "/agent/6c10d0df-fc93-4903-a939-147d51cea1cc"
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            first = await client.get(path)
            etag = first.headers["etag"]
            second = await client.get(path, headers={"If-None-Match": etag})
        assert first.status_code == 200
        assert second.status_code == 304
        assert second.content == b""

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/public/leaderboard",
            "/api/v1/public/weights",
            "/api/v1/public/activity",
            "/api/v1/public/operations",
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
        assert '<h1 id="page-title">Overview</h1>' in body
        assert '<h2 id="operations-title">Network operations</h2>' not in body
        assert "<h2>Recent submissions</h2>" not in body
        assert "<h2>What DittoBench v2 measures</h2>" not in body
        assert "Submission pipeline" in body
        assert body.index('id="leaderboard-title">Leaderboard</h2>') < body.index(
            'class="operations"'
        )
        assert 'id="leaderboard-title">Leaderboard' in body
        assert 'id="leaderboard-notice" role="status" aria-live="polite"' in body
        assert "Provisional standings." in body
        assert "not the required 3 of 3 scores" in body
        assert "only final results drive emissions" in body
        assert "Registration unavailable." in body
        assert (
            "function isRegistered(e) { return !!e && e.registered === true; }" in body
        )
        assert "e.emission_eligible === true" in body
        assert ">registration unknown</span>" in body
        assert 'class="quorum-badge"' in body
        assert ">Best-scoring agent</span>" in body
        assert 'class="miner-uid" title="Current SN118 UID">UID ' in body
        assert ">Total scores</span>" in body
        assert ">Validators</span>" in body
        assert "Scoring Spend" not in body
        assert "Avg latency" not in body
        assert "Scores · 24h" not in body
        assert ">Score rank</span>" in body
        assert ">Emissions</span>" in body
        assert 'id="emissions-strip" role="status" aria-live="polite"' in body
        assert 'id="chain-observation"' in body
        assert 'getJSON("/public/weights")' in body
        assert "Commit-reveal can make this lag active commitments" in body
        assert "Yuma combines validator inputs stake-weightedly" in body
        assert 'return "Validator top choice · "' in body
        assert 'return "Validator support · "' in body
        assert "Chain · champion" not in body
        assert "Chain · weighted" not in body
        assert "is the validator top choice in" in body
        assert "Revealed validator support" in body
        assert "Top choice means the miner received" in body
        assert "KOTH champion and recipients shown separately" in body
        assert "Raw score rank #" in body
        assert 'emission.role === "champion"' in body
        assert "must lead by more than" in body
        assert "2% protection margin" in body
        assert 'class="winner-identity"' in body
        assert 'entityAnchor("agent", e.agent_id, displayAgentName)' in body
        assert "agentVersionBadge(e.agent_version)" in body
        assert "Legacy submission" in body
        assert "<b>Compared with:</b>" in body
        assert "function isFinalized(e)" in body
        assert '"Provisional leaderboard"' in body
        assert '"P" + e.rank' in body
        assert "getJSON(activityRequestPath(page))" in body
        assert 'id="activity-rows"' in body
        assert 'id="activity-pager"' in body
        assert 'id="pipeline-overview"' in body
        assert "Network operations" in body
        assert "Waiting for screening" in body
        assert "Waiting for scores" in body
        assert "function validatorQueueCompare(a, b)" in body
        assert "indexed.sort(validatorQueueCompare)" in body
        assert "validator_queue_rank" in body
        assert "entry.provisional_composite" in body
        assert '"Provisional " + fx(Number(entry.provisional_composite))' in body
        assert "Highest current priority; validator eligibility can vary" in body
        assert ">Up next</span>" in body
        assert "Evaluating" in body
        assert 'id="pipeline-scored"' in body
        assert 'data-pipeline-stage="scored"' in body
        assert "Recent scores" in body
        assert 'statuses: ["scored", "live", "below_score_floor"]' in body
        assert 'getJSON("/public/operations")' in body
        assert "max-height: 390px" in body
        assert "indexed.slice(0, 5)" not in body
        assert body.count('type="button" data-activity-page="prev"') == 2
        assert body.count('type="button" data-activity-page="next"') == 2
        assert 'aria-label="Submission pages, bottom"' in body
        assert 'class="activity-table-frame"' in body
        assert "lockActivityFrameHeight" in body
        assert "anchor.getBoundingClientRect().top - anchorTop" in body
        assert "Validation" in body
        assert "openActivityModal" in body
        assert "scores received" in body
        assert "renderAcceptedScores" in body
        assert "Accepted scores" in body
        assert '"Bench v" + score.bench_version' in body
        assert '"Bench v" + a.bench_version' in body
        assert 'class="bench-version-badge"' in body
        assert "function benchmarkCohorts(pipeline)" in body
        assert "function cohortProgressSummary(cohort, quorum)" in body
        assert '" · " + running + " running"' in body
        assert '" · " + pending + " pending"' in body
        assert "benchmarkVersionKey(pipeline.active_bench_version)" in body
        assert "cohortMedian(cohort.scores)" in body
        assert "pipeline.score_count) + ' of ' + esc(pipeline.quorum)" not in body
        assert "pipeline.final_composite" not in body
        assert "Per-question results" in body
        assert "casesSection(score)" in body
        assert "casesSection(s)" in body
        assert "Provisional score " in body
        assert "Provisional scores may change" in body
        assert "final median is authoritative" in body
        assert "No validator score has been accepted yet." in body
        assert "esc(benchmarkVersionLabel(cohort.key)) + ' aggregate: '" in body
        assert "median of " in body
        assert "score.reproduction_command" in body
        assert "score.verification_command" in body
        assert "score.dataset_sha256" in body
        assert 'copyButton(score.seed, "benchmark seed")' in body
        assert 'copyButton(score.reproduction_command, "dataset command")' in body
        assert "esc(score.reproduction_command)" in body
        assert "esc(score.verification_command)" in body
        assert (
            "Derived from an on-chain block hash after submission commitment." in body
        )
        assert "random fallback after submission commitment" in body
        assert "per-submission dataset pinning was not enabled" in body
        assert "before per-submission dataset pinning was enabled" in body
        assert "already-submitted artifact" in body
        assert "validator is assigned; its score is pending" in body
        assert "Score pending" in body
        assert "has this assignment" in body
        assert "Copy review:" in body
        assert "screening_reason" in body
        assert '<details class="old-screeners">' in body
        assert "Old screener results" in body
        assert "Screener result" in body
        assert "Released from quarantine" in body
        assert "Sent for rescreening" in body
        assert "Rejected after quarantine" in body
        assert 'return ["Quarantined", "warn"]' in body
        assert "a.quarantine_resolved_at || a.finished_at || a.started_at" in body
        assert "Lease expired" not in body
        assert "System failure" not in body
        assert 'role === "validator" ? "Retrying" : "Expired"' in body
        assert "Validator took too long to post a score." in body
        assert "Another validator will score you soon." in body
        assert 'class="retry-info" role="img" tabindex="0"' in body
        assert 'data-tooltip="' in body
        assert "validatorRetryInfo(a.actively_running" in body
        assert "Assignment expired" not in body
        assert 'failed: ["Could not complete", "warn"]' in body
        assert 'class="pipeline-summary"' in body
        assert 'class="pipeline-key-facts"' in body
        assert 'class="pipeline-meta-list"' in body
        assert 'class="pipeline-history"' in body
        assert 'class="pipeline-detail-state"' in body
        assert 'style="margin-top:18px">Validator progress' not in body
        assert "Dispute screening decision" in body
        assert "one private dispute" in body
        assert "cannot be edited or replaced" in body
        assert "ditto-dispute-v1:" in body
        assert 'id="screening-dispute-wallet"' in body
        assert 'id="screening-dispute-hotkey"' in body
        assert "Ready-to-run btcli command" in body
        assert "btcli wallet sign --wallet-name" in body
        assert '" --wallet-hotkey "' in body
        assert '" --use-hotkey --message "' in body
        assert '" --json-output"' in body
        assert 'data-copy-label="btcli signing command"' in body
        assert "Wallet details stay in this browser and are not submitted." in body
        assert 'maxlength="1000"' in body
        assert 'maxlength="128"' in body
        assert 'pattern="[0-9a-fA-F]{128}"' in body
        assert 'postJSON("/public/agent/"' in body
        assert "renderScreeningDispute(pipeline)" in body
        assert "Your one dispute was submitted" in body
        assert "private message" in body

    async def test_api_failures_do_not_render_sample_data(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text

        assert "var SAMPLE" not in body
        assert "SAMPLE_HEALTH" not in body
        assert "render(SAMPLE" not in body
        assert "function renderLeaderboardUnavailable()" in body
        assert "function renderHealthUnavailable()" in body
        assert "<b>Live data unavailable.</b>" in body
        assert "No example data is shown." in body
        assert (
            'Promise.allSettled([getJSON("/public/leaderboard"), '
            'getJSON("/public/weights")])' in body
        )
        assert "lastLeaderboardData = null;" in body
        assert 'setStatus("error", "Data unavailable");' in body
        assert "renderLeaderboardUnavailable();" in body
        assert ".catch(function () { renderHealthUnavailable(); });" in body

    async def test_includes_public_miner_facing_ath_review_queue(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text

        assert 'href="#/reviews" data-page="reviews"' in body
        assert '<section class="page" data-page="reviews">' in body
        assert "High scores get a second look." in body
        assert "Recorded scores stay preserved" in body
        assert "emission eligibility pauses" in body
        assert "A clear restores eligibility" in body
        assert "A fresh evaluation replaces the held result" in body
        assert 'id="ath-review-list" class="ath-review-list" aria-live="polite"' in body
        assert 'id="ath-review-state" class="ath-state"' in body
        assert "No active ATH reviews." in body
        assert "Could not load active reviews." in body
        assert "Cached snapshot" in body
        assert "Refresh failed · showing last public snapshot" in body
        assert 'var path = "/public/activity?review=ath&status=' in body
        assert 'under_review&limit=200&page=1"' in body
        assert "return poolMap(pageNumbers, 4, function (pageNumber) {" in body
        assert 'return getJSON(path.replace("page=1", "page=" + pageNumber));' in body
        assert "entry.review_opened_at" in body
        assert "entry.preserved_composite" in body
        assert 'copyButton(hotkey, "miner hotkey")' in body
        assert 'entityAnchor("agent", agentId' in body
        assert 'entityAnchor("miner", hotkey' in body
        assert "Authorization" not in body
        assert "/admin/copy-reviews" not in body

    async def test_includes_server_backed_submission_quick_filters(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert 'aria-label="Quick submission filters"' in body
        assert 'data-activity-filter="all" aria-pressed="true"' in body
        assert 'data-activity-filter="rejected" aria-pressed="false"' in body
        assert 'data-activity-filter="under_review" aria-pressed="false"' in body
        assert "Operator review" in body
        assert 'data-activity-filter="waiting_validator" aria-pressed="false"' in body
        assert 'data-activity-filter="queued" aria-pressed="false"' in body
        assert 'waiting_screening", "screening", "waiting_validator' in body
        assert 'below_score_floor: ["Below score floor", "warn"]' in body
        assert 'under_review: ["Operator review", "warn"]' in body
        assert '"below_score_floor", "under_review"' in body
        assert "var provisionalScores = e.provisional_scores || [];" in body
        assert "if (!provisionalScores.length || !Number.isFinite(scoreFloor))" in body
        assert (
            "Evaluation stopped after two accepted scores below the current score "
            "floor." in body
        )
        assert "No further validator tickets will be issued." in body
        assert (
            "Automated processing is paused while an operator reviews this submission."
            in body
        )
        assert "No screener or validator is currently working on it." in body
        assert 'id="activity-clear" type="button" hidden' in body
        assert 'id="activity-filter-summary" role="status" aria-live="polite"' in body
        assert 'query.append("status", status)' in body
        assert 'if (activityQuery) query.set("q", activityQuery)' in body
        assert "getJSON(activityRequestPath(page))" in body
        assert "activityPage = 1;" in body
        assert "No submissions match these filters." in body
        assert "Could not load submissions. Try again." in body

    async def test_submission_filters_and_page_restore_and_sanitize_the_url(
        self,
    ) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert "function restoreActivityUrl()" in body
        assert 'url.searchParams.getAll("status")' in body
        assert "ACTIVITY_STATUSES.indexOf(value) >= 0" in body
        assert "function writeActivityUrl(push)" in body
        assert 'url.searchParams.append("status", status)' in body
        assert 'url.searchParams.set("q", activityQuery)' in body
        assert 'url.searchParams.get("page")' in body
        assert "/^[1-9][0-9]*$/.test(requestedPage)" in body
        assert "Number.isSafeInteger(parsedPage)" in body
        assert 'url.searchParams.set("page", String(activityPage))' in body
        assert 'url.searchParams.delete("page")' in body
        assert (
            "function navigateActivityPage(page, anchor, push, userInitiated)" in body
        )
        assert "navigateActivityPage(activityPage - 1, control, true, true)" in body
        assert "navigateActivityPage(activityPage + 1, control, true, true)" in body
        assert (
            "navigateActivityPage(Math.max(1, data.total_pages), anchor, false, "
            "userInitiated === true)" in body
        )
        assert 'history[push ? "pushState" : "replaceState"]' in body
        assert 'window.addEventListener("popstate"' in body
        assert "if (activityUrlWasSanitized) writeActivityUrl(false)" in body
        assert "searchInput.value = activityQuery" in body
        assert "loadActivity(activityPage, null, true)" in body

    async def test_submission_filters_are_mobile_and_keyboard_accessible(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert '.activity-filter[aria-pressed="true"]' in body
        assert ".activity-filter { min-height: 44px; }" in body
        assert ".activity-filter-list { width: 100%; }" in body
        assert ".activity-table-frame { min-width: 680px; }" in body
        assert (
            'button.setAttribute("aria-pressed", selected ? "true" : "false")' in body
        )
        assert "[data-activity-filter]" in body
        assert ":focus-visible { outline: 2px solid var(--focus)" in body

    async def test_explains_policy_rescreen_from_public_activity_state(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert 'id="rescreen-notice"' in body
        assert 'id="rescreen-count"' in body
        assert 'id="rescreen-scored"' in body
        assert "function renderPolicyRescreenNotice(entries, unavailable)" in body
        assert (
            'entry.status === "waiting_screening" || entry.status === "screening"'
            in body
        )
        assert "completed < required" in body
        assert "Number(entry.screening_policy_version) > 0" in body
        assert "Prior scores remain preserved" in body
        assert "validators may intentionally idle" in body
        assert "lower-score submissions clear screening" in body
        assert "This is not data loss" in body
        assert "policyScreeningLabel(entry)" in body
        assert 'return "Rescreen · policy v" + completed + " → v" + required' in body

    async def test_includes_accessible_fleet_status(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert "Fleet health" in body
        assert 'id="fleet-summary"' in body
        assert 'id="fleet-rows"' in body
        assert 'id="fleet-retired" hidden' in body
        assert 'id="fleet-retired-rows"' in body
        assert "Recently offline" in body
        assert "Heartbeat history remains visible for 24 hours" in body
        assert 'id="show-screeners"' in body
        assert 'type="checkbox"' in body
        assert '<label class="fleet-toggle" for="show-screeners">' in body
        assert '<table class="fleet-table"' in body
        assert (
            '<th scope="col" id="fleet-node-heading" style="width:240px">Validator</th>'
            in body
        )
        assert '<th scope="col" style="width:88px">First seen</th>' in body
        assert '<th scope="col" style="width:96px">Last heartbeat</th>' in body
        assert '<th scope="col" style="width:100px">Status</th>' in body
        assert '<th scope="col" style="width:88px">CPU</th>' in body
        assert '<th scope="col" style="width:78px">Containers</th>' in body
        assert 'showScreeners ? "Screener" : "Validator"' in body
        assert "Missing optional telemetry is not an outage." in body
        assert "allowlisted" not in body
        assert 'id="fleet-count-unknown"' in body
        assert 'getJSON("/public/operations")' in body
        assert 'getJSON("/public/validator-names")' in body
        assert 'getJSON("/public/screeners")' in body
        assert 'getElementById("show-screeners").addEventListener' in body
        assert 'showScreeners ? "Screener" : "Validator"' in body
        assert "running_benchmark" in body
        assert "metrics.cpu_percent >= 95" not in body
        assert 'fleetMeter(metrics.cpu_percent, "")' in body
        assert 'metrics.disk_percent >= 95 ? "warn" : ""' in body
        assert "metrics.disk_percent >= 85" not in body
        assert "privacy-note" not in body
        assert "fleet-health-note" not in body
        assert '" reporting " + kind' not in body
        assert 'available + " of " + entries.length + " active " + kind' in body
        assert 'entry.availability === "offline"' in body
        assert "retired.hidden = !showScreeners || !retiredEntries.length" in body
        assert '" · " + retiredEntries.length + " recently offline"' in body

    async def test_operations_panels_share_one_snapshot_and_show_skew(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert body.count('getJSON("/public/operations")') == 1
        assert 'getJSON("/public/validators")' not in body
        assert 'getJSON("/public/activity?page=1&limit=200")' not in body
        assert 'id="operations-snapshot" aria-live="polite"' in body
        assert "Pipeline and fleet reconciled" in body
        assert 'entry.assignment_state === "heartbeat_mismatch"' in body
        assert 'entry.assignment_state === "heartbeat_stale"' in body
        assert 'return ["Mismatch", "bad"]' in body
        assert "counts.warning++" in body
        assert "Assignment mismatch" in body
        assert "Heartbeat stale" in body
        assert "<b>Platform</b>" in body
        assert "<b>Heartbeat</b>" in body

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
        assert "LLM review" in body
        assert "source_review_" in body
        assert "elapsedDuration" in body
        assert "benchmark-progress-time" in body
        assert "progress.started_at" in body
        assert "renderPipelineScreenerProgress" in body
        assert "activeScreenerFor" in body
        assert "pipelineScreenerStage" in body
        assert "renderPipelineBoard({ entries: pipelineEntries }, false)" in body
        assert "data-started-at" in body
        assert "active_agent_name" in body
        assert "setInterval(updateElapsedTimes, 1000)" in body

    async def test_includes_copy_controls_for_operational_identifiers(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert 'id="d-hotkey-copy"' in body
        assert 'copyButton(e.dataset_sha256, "dataset SHA-256")' in body
        assert 'copyButton(e.agent_id, "agent ID")' in body
        assert 'copyButton(e.miner_hotkey, "miner hotkey")' in body
        assert 'copyButton(hotkey, singular + " hotkey")' in body
        assert 'copyButton(s.validator_hotkey, "validator hotkey")' in body
        assert 'id="copy-status"' in body
        assert 'role="status"' in body
        assert 'aria-live="polite"' in body
        assert 'aria-describedby="copy-status"' in body
        assert 'type="button" class="copy"' in body
        assert 'document.addEventListener("click"' in body
        assert 'document.addEventListener("keydown"' in body
        assert 'ev.key !== "Enter" && ev.key !== " "' in body
        assert 'document.execCommand("copy")' in body
        assert "navigator.clipboard.writeText(value).catch" in body
        assert 'btn.classList.add("failed")' in body
        assert "Could not copy " in body
        assert "Select the full value and copy it manually." in body

    async def test_includes_miner_facing_review_details_copy(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text

        assert "function reviewPacket(entry)" in body
        assert '"Please review agent " + agentId' in body
        assert '"Name: " + name + " (" + agentVersionLabel(version) + ")"' in body
        assert '"Miner hotkey: " + reviewPacketLine(entry.miner_hotkey)' in body
        assert 'lines.push("Status: " + reviewPacketLine(entry.status))' in body
        assert 'lines.push("Artifact SHA-256: "' in body
        assert 'canonicalEntityUrl("agent", agentId)' in body
        assert 'class="copy review-copy"' in body
        assert body.count("reviewPacketButton(e)") == 2
        assert 'aria-label="Copy review details"' in body
        assert ".review-copy { width: 100%;" in body

    async def test_validator_names_remain_optional_untrusted_decoration(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert "var validatorNames = {};" in body
        assert "var validatorStakeWeights = {};" in body
        assert "validatorNames = {};" in body
        assert "validatorStakeWeights = {};" in body
        assert "(data.validators || []).forEach" in body
        assert "validatorNames[entry.validator_hotkey] = entry.display_name" in body
        assert (
            "validatorStakeWeights[entry.validator_hotkey] = entry.stake_weight" in body
        )
        assert "function sortFleetEntries(entries, singular)" in body
        assert "return rightStake - leftStake" in body
        assert "if (leftHotkey < rightHotkey) return -1" in body
        assert "if (leftHotkey > rightHotkey) return 1" in body
        assert "sortFleetEntries(data[kind] || [], singular)" in body
        assert (
            'var displayName = singular === "validator" ? validatorNames[hotkey]'
            in body
        )
        assert "esc(displayName)" in body
        assert "entityAnchor(singular, hotkey, shortKey(hotkey))" in body
        assert "fleet-node-key copyable" in body
        assert "title=\"' + esc(hotkey)" in body
        assert 'copyButton(hotkey, singular + " hotkey")' in body
        assert "fleetUnavailable.validators = true" in body

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
        assert (
            ".side-theme { order: 3; flex: 1 0 100%; width: 100%; min-width: 0; "
            "margin-top: 0; }" in body
        )
        assert (
            ".side-theme .theme-switch { display: grid; "
            "grid-template-columns: repeat(4, minmax(0, 1fr)); width: 100%; }" in body
        )
        assert ".side-theme .theme-option { min-height: 36px;" in body

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
        assert 'id="leaderboard-title">Leaderboard</h2>' in body  # folded into Overview
        assert 'data-theme-choice="system"' in body  # switcher still wired

    async def test_dashboard_entities_use_query_popovers_and_pages(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert 'agents: "submissions"' in body
        assert 'miners: "overview"' in body
        assert 'validators: "operations"' in body
        assert 'screeners: "operations"' in body
        assert "url.searchParams.set(ENTITY_PARAMS[plural], String(identifier))" in body
        # Drilldowns are overlays over the current page; ENTITY_PAGES is only the
        # cold-link fallback when no page route is present in the hash.
        assert (
            'url.hash = "#/" + (page || currentPageName() || ENTITY_PAGES[plural])'
            in body
        )
        assert (
            'return "/" + singular + "/" + encodeURIComponent(String(identifier))'
            in body
        )
        assert 'id="d-open-full"' in body
        assert 'id="d-back-dashboard"' in body
        assert r"/^\/(agent|miner)\/([^/]+)\/?$/" in body
        assert "query.has(ENTITY_PARAMS[kind])" in body
        assert r"/^#\/(agents|miners|validators|screeners)\/([^/?#]+)\/?$/" in body
        assert "if (entity.legacy)" in body
        assert 'history.replaceState(history.state || {}, "", entityHref(' in body
        assert 'data-entity-link="agent"' in body
        assert 'entityAnchor("validator", a.validator_hotkey' in body
        assert 'entityAnchor("screener", a.screener_hotkey' in body
        assert 'data-entity-kind="' in body
        assert 'history.pushState({ entity: true }, "", href)' in body
        assert 'window.addEventListener("popstate"' in body
        assert "if (entity.full) showAgentRouteState(" in body
        assert '"Loading submission details…", "loading"' in body
        assert '"This submission could not be found.", "error"' in body
        assert '"Submission details are temporarily unavailable.' in body
        assert 'getJSON("/public/activity?page=1&limit=1&q="' in body
        assert 'target.setAttribute("aria-current", "true")' in body

    async def test_mobile_sidebar_stays_below_modal(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert ".modal-back { position: fixed;" in body
        assert "z-index: 40;" in body
        assert "position: fixed; top: 50%; left: 50%; z-index: 50;" in body
        assert ".sidebar { position: sticky; top: 0; z-index: 30;" in body

    async def test_includes_accessible_global_search(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert 'id="global-search"' in body
        assert 'id="search-input" type="search" role="combobox"' in body
        assert 'aria-controls="search-results"' in body
        assert 'id="search-results" role="listbox"' in body
        assert 'role="option" aria-selected="false"' in body
        assert "function searchCorpus()" in body
        assert "pipelineEntries.forEach" in body
        assert 'history.pushState({}, "", dashboardHref("overview"))' in body
        assert 'history.pushState({}, "", dashboardHref("submissions"))' in body
        assert 'event.key === "/"' in body
        assert 'event.key.toLowerCase() === "k"' in body
        assert 'event.key === "ArrowDown"' in body
        assert 'event.key === "Escape"' in body

    async def test_benchmark_badge_omits_latest_suffix(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert 'badge.textContent = "DittoBench v" + currentBench +' in body
        assert 'currentBench + " · latest"' not in body

    async def test_leaderboard_omits_tie_labels(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert "≈ tie" not in body
        assert "function tieChip(" not in body
