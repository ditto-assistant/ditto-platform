"""Unit tests for :mod:`ditto.api_server.factory`."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

from ditto.api_models.inference_concurrency_settings import (
    MAX_EMBEDDING_GLOBAL_CONCURRENCY,
)
from ditto.api_server import create_api_server
from ditto.api_server.errors import ApiServerConfigError, ApiServerLifespanError
from ditto.api_server.middleware import (
    AuthPassThroughMiddleware,
    RequestIDMiddleware,
)
from ditto.tests.api_server.conftest import make_api_server_config


class TestCreateApiServer:
    """Sanity-checks the wiring."""

    def test_returns_fastapi_instance(self):
        app = create_api_server(make_api_server_config())
        assert isinstance(app, FastAPI)

    def test_state_carries_config(self):
        config = make_api_server_config()
        app = create_api_server(config)
        assert app.state.config is config
        assert app.state.commit_hash == "test-commit"

    def test_middleware_order_request_id_outermost(self):
        """Starlette inserts each middleware at position 0, so the LAST
        ``add_middleware`` call ends up outermost. RequestIDMiddleware
        must be outermost or future auth that short-circuits with 401
        before ``call_next`` would skip request-id setup entirely."""
        app = create_api_server(make_api_server_config())
        classes = [m.cls for m in app.user_middleware]
        assert classes[0] is RequestIDMiddleware
        assert AuthPassThroughMiddleware in classes
        assert classes.index(RequestIDMiddleware) < classes.index(
            AuthPassThroughMiddleware
        )

    def test_redoc_disabled(self):
        app = create_api_server(make_api_server_config())
        assert app.redoc_url is None
        assert app.docs_url == "/docs"
        assert app.openapi_url == "/openapi.json"

    def test_health_and_metrics_excluded_from_schema(self):
        app = create_api_server(make_api_server_config())
        schema = app.openapi()
        # Ops routes have include_in_schema=False so they should not appear.
        assert "/health" not in schema["paths"]
        assert "/metrics" not in schema["paths"]

    def test_health_and_metrics_routes_registered(self):
        app = create_api_server(make_api_server_config())
        paths = {route.path for route in app.routes if hasattr(route, "path")}
        assert "/health" in paths
        assert "/metrics" in paths


class TestLifespanFailureCleanup:
    """``AsyncExitStack`` must dispose the engine if chain open fails."""

    async def test_engine_disposed_when_chain_open_raises(self):
        """Regression net for the ordering in ``_make_lifespan``: the
        engine must be registered as a stack callback BEFORE the chain
        client enters, so a chain-open failure unwinds the engine cleanly
        instead of leaking pooled Postgres connections."""
        engine = MagicMock()
        engine.dispose = AsyncMock()

        # The chain client's ``__aenter__`` raises during lifespan startup.
        chain_ctx = MagicMock()
        chain_ctx.__aenter__ = AsyncMock(side_effect=RuntimeError("pylon down"))
        chain_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("ditto.api_server.factory.create_db_engine", return_value=engine),
            patch(
                "ditto.api_server.factory.create_session_maker",
                return_value=MagicMock(),
            ),
            patch(
                "ditto.api_server.factory.create_chain_client",
                return_value=chain_ctx,
            ),
        ):
            app = create_api_server(make_api_server_config())
            with pytest.raises(ApiServerLifespanError, match="pylon down"):
                async with app.router.lifespan_context(app):
                    pass

        engine.dispose.assert_awaited_once()


class TestRouteDiscoveryIsSingleton:
    """Only the platform role may run the provider-route discovery loop.

    ``ProviderRouteRefresher.refresh`` upserts ``InferenceRoutingPolicy`` and
    ``InferenceProviderRoute`` rows by hand -- ``session.get`` then
    ``session.add`` -- with no ON CONFLICT, no row lock and no savepoint, all
    inside one transaction per model. Two processes running it against the one
    shared database collide on the primary key, and because the insert shares
    that transaction the whole model's refresh cycle is discarded; the loop's
    handler logs only the exception type, so it degrades silently.

    The relay does not need it: route selection reads Postgres per request
    under ``FOR UPDATE``, so there is no in-memory route cache to keep warm.
    """

    @staticmethod
    async def _refresher_for_role(role: str | None, monkeypatch) -> MagicMock:
        if role is None:
            monkeypatch.delenv("DITTO_ROLE", raising=False)
        else:
            monkeypatch.setenv("DITTO_ROLE", role)

        refresher = MagicMock()
        refresher.start = AsyncMock()
        refresher.aclose = AsyncMock()
        engine = MagicMock()
        engine.dispose = AsyncMock()
        chain_ctx = MagicMock()
        chain_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        chain_ctx.__aexit__ = AsyncMock(return_value=False)
        storage_ctx = MagicMock()
        storage_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        storage_ctx.__aexit__ = AsyncMock(return_value=False)
        closeable = MagicMock()
        closeable.aclose = AsyncMock()

        with (
            patch("ditto.api_server.factory.create_db_engine", return_value=engine),
            patch(
                "ditto.api_server.factory.create_session_maker",
                return_value=MagicMock(),
            ),
            patch(
                "ditto.api_server.factory.create_chain_client", return_value=chain_ctx
            ),
            patch(
                "ditto.api_server.factory.create_price_oracle", return_value=closeable
            ),
            patch("ditto.api_server.factory.create_payment_verifier"),
            patch(
                "ditto.api_server.factory.create_storage_client",
                return_value=storage_ctx,
            ),
            patch("ditto.api_server.factory.create_embedder", return_value=closeable),
            patch("ditto.api_server.factory.create_generator", return_value=closeable),
            patch(
                "ditto.api_server.factory.ProviderRouteRefresher",
                return_value=refresher,
            ),
        ):
            app = create_api_server(make_api_server_config())
            app.state.validator_names.start = AsyncMock()
            app.state.validator_names.aclose = AsyncMock()
            async with app.router.lifespan_context(app):
                pass
        return refresher

    async def test_relay_does_not_start_route_discovery(self, monkeypatch):
        refresher = await self._refresher_for_role("relay", monkeypatch)
        refresher.start.assert_not_awaited()
        # Still constructed and still registered for cleanup, so shutdown is
        # symmetric with the platform role.
        refresher.aclose.assert_awaited_once()

    @pytest.mark.parametrize("role", [None, "platform"])
    async def test_platform_role_still_starts_route_discovery(
        self, monkeypatch, role: str | None
    ):
        refresher = await self._refresher_for_role(role, monkeypatch)
        refresher.start.assert_awaited_once()


class TestProcessRole:
    """``DITTO_ROLE=relay`` serves the inference plane and nothing else.

    The relay exists to get the inference hot path off the event loop that
    ingests validator heartbeats, because proxy load delaying that ingest is
    what lets the platform force-expire live leases and destroy in-flight runs.
    It is the same codebase deployed twice, so these tests pin the only thing
    that actually differs between the two roles: the mounted surface.
    """

    @staticmethod
    def _paths(app: FastAPI) -> set[str]:
        return {getattr(route, "path", "") for route in app.routes}

    def test_relay_serves_inference_health_and_metrics_only(self, monkeypatch):
        monkeypatch.setenv("DITTO_ROLE", "relay")
        paths = self._paths(create_api_server(make_api_server_config()))
        assert "/api/v1/inference/chat/completions" in paths
        assert "/api/v1/inference/embeddings" in paths
        assert "/api/v1/inference/exchange" in paths
        assert "/health" in paths
        assert "/metrics" in paths

    def test_relay_does_not_mount_the_platform_surface(self, monkeypatch):
        """A relay host must not serve uploads, scoring, validator, or admin.

        This is the containment property: even if the relay is reachable and
        holds valid credentials, there is no route on it to drive the rest of
        the platform.
        """
        monkeypatch.setenv("DITTO_ROLE", "relay")
        paths = self._paths(create_api_server(make_api_server_config()))
        forbidden = [
            path
            for path in paths
            if path.startswith("/api/v1/") and not path.startswith("/api/v1/inference/")
        ]
        assert forbidden == []

    def test_platform_role_is_the_default_and_is_unchanged(self, monkeypatch):
        monkeypatch.delenv("DITTO_ROLE", raising=False)
        default = self._paths(create_api_server(make_api_server_config()))
        monkeypatch.setenv("DITTO_ROLE", "platform")
        explicit = self._paths(create_api_server(make_api_server_config()))
        assert default == explicit
        assert "/api/v1/upload/agent" in default
        assert "/api/v1/inference/chat/completions" in default

    def test_relay_is_a_strict_subset_of_the_platform_surface(self, monkeypatch):
        """The relay must never expose a route the platform does not.

        Same code, same routers -- if this ever fails, the two roles have
        diverged into two implementations, which is exactly what this design
        exists to prevent.
        """
        monkeypatch.setenv("DITTO_ROLE", "platform")
        platform = self._paths(create_api_server(make_api_server_config()))
        monkeypatch.setenv("DITTO_ROLE", "relay")
        relay = self._paths(create_api_server(make_api_server_config()))
        assert relay <= platform

    @pytest.mark.parametrize("value", ["proxy", "RELAY ", "", "platfrom", "true"])
    def test_unknown_role_fails_boot_instead_of_defaulting(self, monkeypatch, value):
        """A typo must not silently serve the full admin surface from a relay.

        Only the empty string and exact casings of the two known roles are
        tolerated; everything else is a configuration error at boot.
        """
        monkeypatch.setenv("DITTO_ROLE", value)
        if value.strip().lower() in {"", "relay", "platform"}:
            create_api_server(make_api_server_config())
            return
        with pytest.raises(ApiServerConfigError):
            create_api_server(make_api_server_config())


class TestInferenceLanes:
    """``DITTO_INFERENCE_LANES`` puts chat and embeddings on separate processes.

    The two lanes are opposite-shaped traffic sharing one single-threaded event
    loop: a chat completion is ~913 ms of wall clock, an embedding ~75 ms. The
    slow lane's response handling is head-of-line blocking for the fast one, so
    a chat burst adds embedding latency that has nothing to do with the
    embedding provider. Restricting a relay to one lane is what lets the two
    land on different cores.

    Every test here pins the same safety property from a different side: a
    process must never silently serve less than the operator asked for.
    """

    @staticmethod
    def _paths(app: FastAPI) -> set[str]:
        return {getattr(route, "path", "") for route in app.routes}

    def test_default_is_every_lane_so_an_unset_variable_changes_nothing(
        self, monkeypatch
    ):
        """The knob must be inert until someone reaches for it.

        Every existing deployment has this variable unset, and the Caddy path
        split that a lane-restricted relay requires has not landed. An unset
        default that dropped a route would 404 live traffic.
        """
        monkeypatch.setenv("DITTO_ROLE", "relay")
        monkeypatch.delenv("DITTO_INFERENCE_LANES", raising=False)
        unset = self._paths(create_api_server(make_api_server_config()))
        monkeypatch.setenv("DITTO_INFERENCE_LANES", "chat,embedding")
        explicit = self._paths(create_api_server(make_api_server_config()))
        assert unset == explicit
        assert "/api/v1/inference/chat/completions" in unset
        assert "/api/v1/inference/embeddings" in unset

    def test_an_embedding_relay_does_not_mount_the_chat_lane(self, monkeypatch):
        monkeypatch.setenv("DITTO_ROLE", "relay")
        monkeypatch.setenv("DITTO_INFERENCE_LANES", "embedding")
        paths = self._paths(create_api_server(make_api_server_config()))
        assert "/api/v1/inference/embeddings" in paths
        assert "/api/v1/inference/chat/completions" not in paths

    def test_a_chat_relay_does_not_mount_the_embedding_lane(self, monkeypatch):
        monkeypatch.setenv("DITTO_ROLE", "relay")
        monkeypatch.setenv("DITTO_INFERENCE_LANES", "chat")
        paths = self._paths(create_api_server(make_api_server_config()))
        assert "/api/v1/inference/chat/completions" in paths
        assert "/api/v1/inference/embeddings" not in paths

    def test_exchange_is_mounted_on_every_lane(self, monkeypatch):
        """A lane-restricted relay must still be a drop-in behind a path split.

        ``/exchange`` mints the grant BOTH lanes spend against and is called
        once per grant rather than per request. Keeping it everywhere is what
        lets Caddy route only ``/inference/embeddings`` and send *everything
        else* to the chat relay with no special case.
        """
        monkeypatch.setenv("DITTO_ROLE", "relay")
        for lanes in ("chat", "embedding", "chat,embedding"):
            monkeypatch.setenv("DITTO_INFERENCE_LANES", lanes)
            paths = self._paths(create_api_server(make_api_server_config()))
            assert "/api/v1/inference/exchange" in paths, lanes

    def test_the_platform_role_keeps_every_lane_regardless(self, monkeypatch):
        """Narrowing the fallback would turn a relay outage into a full outage."""
        monkeypatch.setenv("DITTO_ROLE", "platform")
        monkeypatch.setenv("DITTO_INFERENCE_LANES", "embedding")
        paths = self._paths(create_api_server(make_api_server_config()))
        assert "/api/v1/inference/chat/completions" in paths
        assert "/api/v1/inference/embeddings" in paths

    @pytest.mark.parametrize("value", ["embeddings", "chat,embeddigs", "llm"])
    def test_an_unknown_lane_fails_boot(self, monkeypatch, value: str):
        """A typo must not produce a healthy process that serves no traffic.

        That failure presents as a load-balancer fault and is considerably
        harder to find than a crash-loop naming the bad value. Note that
        ``embeddings`` (plural) is rejected: it is the route's name, so it is
        the most likely thing an operator types.
        """
        monkeypatch.setenv("DITTO_ROLE", "relay")
        monkeypatch.setenv("DITTO_INFERENCE_LANES", value)
        with pytest.raises(ApiServerConfigError, match="DITTO_INFERENCE_LANES"):
            create_api_server(make_api_server_config())

    def test_a_lane_restricted_relay_is_still_a_subset_of_the_platform(
        self, monkeypatch
    ):
        monkeypatch.setenv("DITTO_ROLE", "platform")
        monkeypatch.delenv("DITTO_INFERENCE_LANES", raising=False)
        platform = self._paths(create_api_server(make_api_server_config()))
        monkeypatch.setenv("DITTO_ROLE", "relay")
        monkeypatch.setenv("DITTO_INFERENCE_LANES", "embedding")
        relay = self._paths(create_api_server(make_api_server_config()))
        assert relay <= platform


class TestEmbeddingConnectionPoolIsSeparate:
    """The embedding lane gets its own httpx pool, and that is a bug fix.

    Both lanes used to share one client whose ``max_connections`` is
    ``global_concurrency`` -- the CHAT limit, 72 in production. Embedding
    admission was independently willing to admit up to 128, so any
    ``embedding_global_concurrency`` above 72 was unreachable by construction:
    admission let the request through and it then blocked acquiring a
    connection, invisibly, inside the window ``latency_ms`` measures. Raising
    the board past 72 was therefore partly a no-op, and the part that did land
    surfaced as latency rather than as an honest decline.
    """

    @staticmethod
    async def _clients(config) -> tuple[object, object]:
        engine = MagicMock()
        engine.dispose = AsyncMock()
        chain_ctx = MagicMock()
        chain_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        chain_ctx.__aexit__ = AsyncMock(return_value=False)
        storage_ctx = MagicMock()
        storage_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        storage_ctx.__aexit__ = AsyncMock(return_value=False)
        closeable = MagicMock()
        closeable.aclose = AsyncMock()
        refresher = MagicMock()
        refresher.start = AsyncMock()
        refresher.aclose = AsyncMock()

        with (
            patch("ditto.api_server.factory.create_db_engine", return_value=engine),
            patch(
                "ditto.api_server.factory.create_session_maker",
                return_value=MagicMock(),
            ),
            patch(
                "ditto.api_server.factory.create_chain_client", return_value=chain_ctx
            ),
            patch(
                "ditto.api_server.factory.create_price_oracle", return_value=closeable
            ),
            patch("ditto.api_server.factory.create_payment_verifier"),
            patch(
                "ditto.api_server.factory.create_storage_client",
                return_value=storage_ctx,
            ),
            patch("ditto.api_server.factory.create_embedder", return_value=closeable),
            patch("ditto.api_server.factory.create_generator", return_value=closeable),
            patch(
                "ditto.api_server.factory.ProviderRouteRefresher",
                return_value=refresher,
            ),
        ):
            app = create_api_server(config)
            app.state.validator_names.start = AsyncMock()
            app.state.validator_names.aclose = AsyncMock()
            async with app.router.lifespan_context(app):
                captured = (
                    app.state.inference_client,
                    app.state.inference_embedding_client,
                )
        return captured

    async def test_the_two_lanes_do_not_share_a_client(self):
        chat, embedding = await self._clients(make_api_server_config())
        assert chat is not embedding

    async def test_the_embedding_pool_can_carry_the_whole_ceiling(self):
        """Sized from the CEILING, not from the configured value.

        ``embedding_global_concurrency`` is hot-swappable from Backroom while
        the pool is fixed at boot. Sizing the pool from the live setting would
        recreate the original bug the instant an operator raised the board
        without a restart -- the ceiling is the only number guaranteed to still
        be true after the next revision.
        """
        _chat, embedding = await self._clients(make_api_server_config())
        limits = embedding._transport._pool._max_connections
        assert limits >= MAX_EMBEDDING_GLOBAL_CONCURRENCY

    async def test_a_chat_burst_cannot_consume_the_embedding_lanes_sockets(self):
        """The isolation property, stated as the thing an operator cares about.

        The chat pool's ceiling is its own lane's limit, so chat saturating
        itself leaves the embedding pool entirely untouched.
        """
        config = make_api_server_config()
        chat, embedding = await self._clients(config)
        chat_max = chat._transport._pool._max_connections
        embedding_max = embedding._transport._pool._max_connections
        assert chat_max == config.inference_proxy.global_concurrency
        assert embedding_max == MAX_EMBEDDING_GLOBAL_CONCURRENCY
