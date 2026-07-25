"""Unit tests for :mod:`ditto.api_server.factory`."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

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
