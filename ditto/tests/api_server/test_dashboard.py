"""Tests for the same-origin dashboard SPA served at ``/``.

The platform doubles as the transparency front door: it serves
``dashboard/index.html`` at ``/`` so the SPA's ``/api/v1/public/*`` calls are
same-origin (no CORS). The served HTML must carry the injected wandb project URL
and be suppressible via config.
"""

from __future__ import annotations

import json
import re
import subprocess

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
            "/api/v1/public/bench/timeline",
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
        assert "function artifactReleaseCopy(release)" in body
        assert "function artifactReleaseNote(release)" in body
        assert "function renderArtifactRelease(release, agentId)" in body
        assert "function downloadArtifact(agentId, button)" in body
        assert '"/public/agent/" + encodeURIComponent(agentId) + "/artifact"' in body
        assert 'cache: "no-store"' in body
        assert "Download submitted source" in body
        assert "The 3/3 score quorum" in body
        assert "Number(release.embargo_hours) || 24" in body
        assert "privacy window" in body
        assert "the original 3/3 timestamp still applies" in body
        assert "function relTimeUntil(iso)" in body
        assert "relTimeUntil(release.available_at)" in body
        assert "relTime(release.available_at)" not in body
        assert "window.location.assign(data.download_url)" in body
        assert "Download started" in body
        assert "Download opened" not in body
        assert "public/king/artifact" not in body
        assert "continuous 24-hour reign" not in body
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
        assert '"ranked by settled v" + activeBench + " composite during the v"' in body
        assert (
            '" rollout · v" + data.desired_bench_version + " progress shown per row"'
            in body
        )
        assert 'data-leaderboard-version="current"' in body
        assert 'data-leaderboard-version="2"' in body
        assert '"?bench_version=" + encodeURIComponent(leaderboardVersionView)' in body
        assert "Raw score rank #" in body
        assert 'emission.role === "champion"' in body
        assert "must lead by more than" in body
        # The margin is read from emissions.margin, not written into the copy.
        assert '" protection margin and the " + method' in body
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
        assert "function queueRelevantBenchmark(progress)" in body
        assert "Number(activeBench) || Number(currentBench)" in body
        assert "version >= activeVersion" in body
        assert "function pipelineBoardStage(entry)" in body
        assert "(entry.active_benchmarks || []).some(queueRelevantBenchmark)" in body
        assert ".filter(queueRelevantBenchmark)" in body
        assert "column.statuses.indexOf(pipelineBoardStage(entry))" in body
        assert '"Bench v" + rescore.targetVersion + " rescore"' in body
        assert "validator_queue_rank" in body
        assert "entry.provisional_composite" in body
        assert '"Provisional " + fx(Number(entry.provisional_composite))' in body
        assert "Highest current priority; validator eligibility can vary" in body
        assert ">Up next</span>" in body
        assert "Evaluating" in body
        assert 'id="pipeline-scored"' in body
        assert 'data-pipeline-stage="scored"' in body
        assert "Recent scores" in body
        assert 'statuses: ["waiting_validator", "below_score_floor"]' in body
        assert 'statuses: ["scored", "live"]' in body

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
        assert "Accepted validator scores" in body
        assert '"Bench v" + score.bench_version' in body
        assert '"Bench v" + a.bench_version' in body
        assert 'class="bench-version-badge"' in body
        assert "function benchmarkCohorts(pipeline)" in body
        assert "function pipelineDisplayState(entry, pipeline)" in body
        assert "function renderPreliminaryFact(cohort, quorum)" in body
        assert " preliminary</dt><dd>" in body
        assert 'class="pipeline-preliminary-value"' in body
        assert "function cohortProgressSummary(cohort, quorum)" in body
        assert "function retestAttemptCounts(attempts)" in body
        assert 'attempt.purpose === "continual_retest"' in body
        assert '" of " + quorum + " quorum inputs"' in body
        assert '" legacy lease draining"' in body
        assert '"Legacy lease unclassified · "' in body
        assert 'retestState(retests.running, "running")' in body
        assert 'retestState(retests.assigned, "assigned")' in body
        assert "function confirmationCohorts(pipeline)" in body
        assert "function renderConfirmationScores(pipeline)" in body
        assert "Continual top-five retests" in body
        assert "Accepted validator scores" in body
        assert "Canonical quorum" in body
        assert "canonical median of" in body
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
        assert (
            "esc(benchmarkVersionLabel(cohort.key)) + ' official aggregate: '" in body
        )
        assert "median of " in body
        assert "score.reproduction_command" in body
        assert "score.verification_command" in body
        assert "score.dataset_sha256" in body
        assert 'copyButton(score.seed, "benchmark seed")' in body
        assert 'copyButton(score.reproduction_command, "dataset command")' in body
        assert "esc(score.reproduction_command)" in body
        assert "esc(score.verification_command)" in body
        assert 'class="run-telemetry-load"' in body
        assert "function loadRunTelemetry(button)" in body
        assert "function renderRunTelemetry(transcript)" in body
        assert "TRANSCRIPT_TELEMETRY_URL_TEMPLATE" in body
        assert "telemetry.source_sha256 !== sha256" in body
        assert "Run telemetry could not be verified or loaded." in body
        assert "Per-question execution" in body
        assert "Model relay:" in body
        assert "caller_cancellations" in body
        assert "infrastructure_failures" in body
        assert "caseEntry.response" not in body
        assert "caseEntry.error" not in body
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
        assert "a.quarantine_resolution_reason" in body
        assert "esc(a.quarantine_resolution_reason)" in body
        assert "Operator reason:" in body
        assert "Sent for rescreening" in body
        assert "Rejected after quarantine" in body
        assert 'return ["Quarantined", "warn"]' in body
        assert "a.quarantine_resolved_at || a.finished_at || a.started_at" in body
        assert "Lease expired" not in body
        assert "System failure" not in body
        assert 'role === "validator" ? "Assignment expired" : "Expired"' in body
        assert "Validator took too long to post a score." in body
        assert "Another validator will score you soon." in body
        assert 'class="retry-info" role="img" tabindex="0"' in body
        assert 'data-tooltip="' in body
        assert "validatorRetryInfo(a.actively_running" in body
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

    async def test_includes_off_network_harness_memory_comparison(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text

        overview_start = body.index(
            '<section class="page active" data-page="overview">'
        )
        benchmark_start = body.index('<section class="page" data-page="benchmark">')
        comparison_start = body.index('<section class="harness-comparison"')
        # The timeline leads the overview now; it is no longer on the benchmark
        # page it used to sit at the bottom of.
        assert overview_start < comparison_start < benchmark_start
        assert 'id="harness-comparison-title">How far miners have taken memory' in body
        assert 'data-tooltip="This isolates memory performance.' in body
        assert 'id="third-party-harness-filter"' not in body
        assert '<details class="harness-comparison-method">' in body
        assert "Method and comparability caveats" in body
        assert "Hermes Agent and OpenClaw measured retrospectively" in body
        assert "var THIRD_PARTY_HARNESSES = [{" in body
        assert 'profile: "Native SessionDB session_search"' in body
        assert 'model: "qwen/qwen3-32b"' in body
        assert 'route: "OpenRouter · Nebius pinned"' in body
        assert 'seed: "3058240546919425205"' in body
        assert 'getJSON("/public/bench/timeline")' in body
        assert 'class="memory-timeline-svg"' in body
        assert 'class="timeline-path miner"' in body
        assert 'class="timeline-release"' in body
        assert 'class="timeline-data-details"' in body
        assert "Exact timeline data" in body
        assert "Their points are positioned in each immutable contract's band" in body
        assert "v4 corrects v3 false positives" in body
        assert "Third-party harnesses never enter score rank, KOTH" in body
        assert "validator weights, or payouts." in body
        assert 'subject: "OpenClaw 2026.7.1"' in body
        assert 'profile: "Native memory-core FTS · 20-result recall"' in body
        assert "THIRD_PARTY_HARNESSES.map(function (evidence)" in body
        assert "Hermes Agent evidence ↗" not in body
        assert "esc(evidence.label) + ' evidence ↗</a>'" in body
        assert "memory-chart-row" not in body
        # The kicker above the title was dropped.
        assert "Reference only · no emissions" not in body

    async def test_memory_timeline_plots_the_field_and_crowns_the_champion(
        self,
    ) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text

        # Every finalized run per contract, from the existing per-version board.
        assert 'getJSON("/public/leaderboard?bench_version=" + version)' in body
        assert 'class="timeline-field' in body
        # Settled contracts are immutable, so only the newest board refetches.
        assert "!memoryFieldByVersion[version] || version === activeVersion" in body
        # The champion is the state the chart exists to show.
        assert 'class="timeline-champion-plate"' in body
        assert 'class="timeline-champion-halo"' in body
        assert "lastEmissions ? lastEmissions.champion_miner_hotkey : null" in body
        # Contracts are banded, not spaced by wall clock.
        assert "var bandWidth = plotWidth / eras.length;" in body
        assert "Each contract gets an equal band, not equal clock time" in body
        # The ordinal generation ramp, interpolated between two theme-aware ends.
        assert "color-mix(in oklch, var(--era-to) " in body
        assert "--era-from: oklch(" in body
        # The viewBox is measured so type keeps its real size on a phone.
        assert "var measured = Math.round((target.clientWidth || 960) - 2);" in body
        assert "new ResizeObserver(function (entries) {" in body
        # A reveal must enhance an already-visible default.
        assert "@keyframes timeline-dot-in { from { opacity: 0;" in body
        assert ".timeline-champion-pulse { display: none; }" in body

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
            "Promise.allSettled([getJSON(leaderboardPath), "
            'getJSON("/public/weights"), getJSON("/public/bench/rollout")])' in body
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
        assert (
            'waiting_screening", "screening", "waiting_validator", "below_score_floor'
            in body
        )
        assert 'below_score_floor: ["Low-priority completion", "warn"]' in body
        assert 'not_queued: ["Historical · not queued", ""]' in body
        assert 'under_review: ["Operator review", "warn"]' in body
        assert '"below_score_floor", "not_queued", "under_review"' in body
        assert "var provisionalScores = e.provisional_scores || [];" in body
        assert "if (!provisionalScores.length || !Number.isFinite(scoreFloor))" in body
        assert (
            "Two accepted scores are below the current same-benchmark score floor."
            in body
        )
        assert "The final score is still queued" in body
        assert "The third score is still queued at low priority" in body
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
        # Filters live in the hash query; legacy real-query filters are honored
        # once and normalized into the hash form.
        assert "var hashQuery = parseHashRoute().query" in body
        assert "var legacy = !hasIn(hashQuery) && hasIn(searchQuery)" in body
        assert 'source.getAll("status")' in body
        assert "ACTIVITY_STATUSES.indexOf(value) >= 0" in body
        assert "function writeActivityUrl(push)" in body
        assert 'query.append("status", status)' in body
        assert 'query.set("q", activityQuery)' in body
        assert 'source.get("page")' in body
        assert "/^[1-9][0-9]*$/.test(requestedPage)" in body
        assert "Number.isSafeInteger(parsedPage)" in body
        assert 'query.set("page", String(activityPage))' in body
        assert 'query.delete("page")' in body
        assert "spaHref(page, query)" in body
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
        assert "data.active_bench_version" in body
        assert "data.desired_bench_version" in body
        assert "data.benchmark_rollout_status" in body
        assert "function renderBenchBadge()" in body
        assert "activeBench || currentBench" in body
        assert "return Number(activeBench) || Number(currentBench) || null" in body
        assert "Math.max(Number(currentBench)" not in body
        assert '" → v" + desired + " rollout"' in body
        assert 'getJSON("/public/validators")' not in body
        assert 'getJSON("/public/activity?page=1&limit=200")' not in body
        assert 'id="operations-snapshot" aria-live="polite"' in body
        assert "Pipeline and fleet reconciled" in body
        assert 'entry.assignment_state === "assignment_mismatch"' in body
        assert 'entry.assignment_state === "assigning"' in body
        assert 'entry.assignment_state === "heartbeat_stale"' in body
        assert 'return ["Mismatch", "bad"]' in body
        assert "counts.warning++" in body
        assert "Assignment mismatch" in body
        assert "Heartbeat stale" in body
        assert "<b>Platform</b>" in body
        assert "<b>Heartbeat</b>" in body

    async def test_benchmark_authority_state_never_promotes_rollout_target(
        self,
    ) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        match = re.search(
            r"(function benchmarkAuthorityState\(.*?\n    \})"
            r"\n\n    function renderBenchBadge",
            body,
            re.DOTALL,
        )
        assert match is not None
        cases = [
            [6, 7, "collecting"],
            [6, 7, "blocked_ineligible"],
            [6, 7, "activated"],
            [6, None, "inactive"],
        ]
        script = (
            match.group(1)
            + "\nconsole.log(JSON.stringify("
            + json.dumps(cases)
            + ".map(function (args) { "
            + "return benchmarkAuthorityState.apply(null, args); })));"
        )
        result = subprocess.run(
            ["node", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(result.stdout) == [
            {"active": 6, "desired": 7, "rolling": True},
            {"active": 6, "desired": 7, "rolling": True},
            {"active": 6, "desired": 7, "rolling": False},
            {"active": 6, "desired": 6, "rolling": False},
        ]

    async def test_leaderboard_state_separates_active_desired_and_history(
        self,
    ) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        match = re.search(
            r"(function leaderboardBenchState\(.*?\n    \})"
            r"\n\n    function renderBenchBadge",
            body,
            re.DOTALL,
        )
        assert match is not None
        cases = [
            ["authoritative", 7, 6, 7, None],
            ["authoritative", 7, 6, 7, 7],
            ["historical", 5, 6, 7, None],
            ["authoritative", 6, None, None, 6],
        ]
        script = (
            match.group(1)
            + "\nconsole.log(JSON.stringify("
            + json.dumps(cases)
            + ".map(function (args) { "
            + "return leaderboardBenchState.apply(null, args); })));"
        )
        result = subprocess.run(
            ["node", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(result.stdout) == [
            {"active": 6, "desired": 7, "selected": 6},
            {"active": 6, "desired": 7, "selected": 6},
            {"active": 6, "desired": 7, "selected": 5},
            {"active": None, "desired": None, "selected": 6},
        ]
        assert "currentBench = data.desired_bench_version" not in body

    async def test_includes_accessible_benchmark_progress(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert "benchmarkStageLabel" in body
        assert "active_benchmarks" in body
        assert "active_benchmark" in body
        assert "progress.bench_version" in body
        assert 'class="benchmark-version-chip"' in body
        assert "function pipelineRescoreState(entry)" in body
        assert " score stays live until v" in body
        assert 'class="pipeline-qualification-badge"' in body
        assert "Cohort → v" in body
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
        assert "L1 source review" in body
        assert "L2/L3 deep review" in body
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

    async def test_includes_public_terminal_screening_review_cards(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert "function renderScreeningReview(attempt)" in body
        assert "Why this submission was rejected" in body
        assert "Source locations in the served path" in body
        assert "Policy observations" in body
        assert "Digest-verified public review" in body
        assert "no source text or private challenge data" in body
        assert "finding.reviewer_revision" in body
        assert 'aria-label="Detailed screening rejection"' in body
        assert ".screening-review-location code" in body
        assert "grid-column: 1 / -1" in body

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

    async def test_advertises_public_source_repositories(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert 'aria-label="Platform source on GitHub"' in body
        assert body.count("https://github.com/ditto-assistant/ditto-platform") == 2
        assert "https://github.com/ditto-assistant/ditto-subnet" in body
        assert "https://github.com/ditto-assistant/ditto-screener" in body
        assert 'aria-label="Open-source Ditto repositories"' in body

    async def test_dashboard_entities_use_query_popovers_and_pages(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert 'agents: "submissions"' in body
        assert 'miners: "overview"' in body
        assert 'validators: "operations"' in body
        assert 'screeners: "operations"' in body
        # Entity params live in the hash query; the real query carries config
        # knobs only. Drilldowns are overlays over the current page and
        # ENTITY_PAGES is only the cold-link fallback when no page route exists.
        assert "query.set(ENTITY_PARAMS[plural], String(identifier))" in body
        assert (
            "return spaHref(page || currentPageName() || ENTITY_PAGES[plural], query)"
            in body
        )
        assert "function parseHashRoute()" in body
        assert "function configSearch()" in body
        # Legacy real-query entity links are recognized and normalized.
        assert "searchEntity.legacy = true" in body
        assert (
            'return "/" + singular + "/" + encodeURIComponent(String(identifier))'
            in body
        )
        assert 'id="d-open-full"' in body
        assert 'id="d-back-dashboard"' in body
        assert r"/^\/(agent|miner)\/([^/]+)\/?$/" in body
        assert "query.has(ENTITY_PARAMS[candidate])" in body
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

    async def test_benchmark_badge_communicates_rollout_transition(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert '"DittoBench v" + active + " → v" + desired + " rollout"' in body
        assert '"DittoBench v" + active + (benchHasOlderRuns' in body
        assert 'currentBench + " · latest"' not in body

    async def test_leaderboard_omits_tie_labels(self) -> None:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        body = (await _get(app, "/")).text
        assert "≈ tie" not in body
        assert "function tieChip(" not in body


class TestDashboardScoringTransparency:
    """The SPA must not restate consensus parameters as literals.

    Every number here (the incumbent margin, the champion share, the tail size,
    the authority-switch threshold, the benchmark version) is served by the API
    and can change without touching this file. A literal in the markup is a
    claim that silently stops being true, which is worse than no claim at all:
    a miner reads it as the rule they are being judged by.
    """

    async def _body(self) -> str:
        app = create_api_server(make_api_server_config(dashboard_enabled=True))
        return (await _get(app, "/")).text

    async def test_no_hardcoded_fold_constants(self) -> None:
        body = await self._body()
        for literal in (
            "2% protection margin",
            "2% incumbent margin",
            "receives 90% of the miner pool",
            "up to four participation-tail recipients",
            "up to 25 miners",
        ):
            assert literal not in body, f"hardcoded fold constant: {literal}"

    async def test_renders_the_dethrone_floor_and_rollout_state(self) -> None:
        body = await self._body()
        # The score to beat is its own element, so it can be shown whether or
        # not there is an active contest.
        assert 'id="emissions-threshold"' in body
        assert "Beat this to contend" in body
        assert "var floor = champComposite + effectiveMargin" in body
        assert "dethroneBandScale" in body
        assert "champComposite * (1 + margin)" not in body
        # And it is published as a floor, never as a sufficient score.
        assert "this is a floor, not a guarantee" in body
        # Rollout / authority state, with the threshold read from the API.
        assert 'id="rollout-strip"' in body
        assert "/public/bench/rollout" in body
        assert "ranked_quorum_agents" in body
        assert "min_ranked_quorum_agents" in body
        assert "This rollout's bounded inherited cohort has " in body
        assert "Number(state.cohort_size)" in body

    async def test_explainer_covers_scoring_emissions_and_koth(self) -> None:
        body = await self._body()
        assert '<details class="bench-disclosure" id="scoring-explainer">' in body
        assert '<details class="bench-disclosure" id="bench-setup">' in body
        assert '<details class="bench-disclosure" id="bench-versions">' in body
        assert '<details class="bench-disclosure" id="bench-glossary">' in body
        assert "the active version is highlighted" in body
        assert "var activeVersion = Number(activeBench)" in body
        assert "var rolloutVersion = Number(desiredBench)" in body
        assert '<span class="ver-now">active</span>' in body
        assert '<span class="ver-next">rollout</span>' in body
        assert "Memory cases contribute half of the unadjusted composite." in body
        assert "Tool-use cases contribute the other half" in body
        for heading in (
            "What a score is.",
            "Which runs rank.",
            "Scores compare only within one benchmark version.",
            "How emissions work.",
            "How the crown changes hands.",
            "When weights move to a new version.",
        ):
            assert heading in body, f"explainer is missing: {heading}"
        assert "0.5 × tool mean + 0.5 × memory mean" in body
        assert "token efficiency, which can remove <b>at most 10%</b>" in body

    async def test_composite_detail_separates_quality_and_token_adjustments(
        self,
    ) -> None:
        body = await self._body()
        assert "Composite calculation" in body
        assert "Tool/memory base" in body
        assert "Benchmark quality gates" in body
        assert "Pre-token composite" in body
        assert "Token efficiency" in body
        assert "token no penalty" in body
        assert "token −" in body
        assert "Token efficiency is separate and can never remove more than 10%" in body

    async def test_benchmark_version_is_never_a_literal(self) -> None:
        body = await self._body()
        # The frozen-setup tag and the version-specific copy are both filled
        # from the API; the static markup carries only a placeholder.
        assert '<span class="tag" id="bs-version">v–</span>' in body
        assert 'class="bv-desired"' in body

    async def test_reference_baseline_is_keyed_by_benchmark_version(self) -> None:
        """Every measured version keeps its baseline; unmeasured ones say so.

        A baseline is a real run of the stock harness on the locked model, so it
        cannot be carried across versions -- composites only compare within one.
        Holding a single constant meant the card went blank the moment the board
        moved to a version it was not measured on, which reads as a broken widget
        rather than as the honest "we have not run this yet" it actually is.
        """
        body = await self._body()
        assert "var REFERENCE_BASELINES = {" in body
        # The published runs, both from dittobench-api docs/BASELINES.md.
        assert "2: { composite: 0.492" in body
        assert "3: { composite: 0.445" in body
        assert "4: { composite: 0.429" in body
        # Unmeasured versions must be stated, not rendered as a bare dash.
        assert "not yet measured" in body
        assert "No reference baseline has been measured on bench_version" in body
        # And the card must not claim a baseline is the winning score.
        assert "it is not the score that wins" in body
