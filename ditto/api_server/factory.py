"""FastAPI app factory.

Per-test instantiation (no module-level ``app =`` global) keeps
``dependency_overrides`` isolated across tests.
"""

from __future__ import annotations

import html
import logging
import re
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse

import ditto
from ditto.api_server.config import (
    ApiServerConfig,
    parse_api_server_config_from_env,
)
from ditto.api_server.datapipeline import create_generator
from ditto.api_server.embedding import create_embedder
from ditto.api_server.endpoints import (
    admin_copy_review_router,
    admin_quarantine_router,
    health_router,
    metrics_router,
    public_router,
    retrieval_router,
    scoring_router,
    screener_router,
    upload_router,
    validator_router,
)
from ditto.api_server.errors import ApiServerLifespanError
from ditto.api_server.middleware import (
    AuthPassThroughMiddleware,
    PublicCacheMiddleware,
    RequestIDMiddleware,
    register_exception_handlers,
)
from ditto.api_server.payment_verifier import create_payment_verifier
from ditto.api_server.pricing import create_price_oracle
from ditto.api_server.storage import create_storage_client
from ditto.api_server.validator_names import create_validator_names
from ditto.chain import create_chain_client
from ditto.db import create_db_engine, create_session_maker

logger = logging.getLogger(__name__)

# The dashboard SPA lives at the repo root (source checkout on the deployed VM);
# it is not packaged into the wheel, so a missing file just disables the route.
_DASHBOARD_FILE = Path(__file__).resolve().parents[2] / "dashboard" / "index.html"
_DASHBOARD_IMAGE = (
    Path(__file__).resolve().parents[2] / "dashboard" / "assets" / "paperditto-512.png"
)
_WANDB_META_RE = re.compile(r'(<meta name="ditto:wandb-url" content=")[^"]*(")')


def _render_dashboard(wandb_url: str) -> str | None:
    """Read the dashboard SPA and inject the public wandb project URL.

    Returns ``None`` (route is skipped) when the file is absent — e.g. a
    packaged/wheel install or a checkout without the ``dashboard/`` dir. The
    ``api-base`` meta is left empty on purpose so the SPA falls back to its
    same-origin ``/api/v1`` default when the platform serves it.
    """
    try:
        page = _DASHBOARD_FILE.read_text(encoding="utf-8")
    except OSError:
        return None
    return _WANDB_META_RE.sub(
        lambda m: m.group(1) + html.escape(wandb_url, quote=True) + m.group(2),
        page,
        count=1,
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    config: ApiServerConfig = app.state.config
    async with AsyncExitStack() as stack:
        try:
            engine = create_db_engine(config.postgres)
            stack.push_async_callback(engine.dispose)
            app.state.engine = engine
            app.state.session_maker = create_session_maker(engine)

            chain = await stack.enter_async_context(create_chain_client(config.chain))
            app.state.chain = chain

            price_oracle = create_price_oracle(config.pricing)
            stack.push_async_callback(price_oracle.aclose)
            app.state.price_oracle = price_oracle

            payment_verifier = create_payment_verifier(
                chain=chain,
                oracle=price_oracle,
                pricing_config=config.pricing,
                send_address=config.upload_payment_address,
            )
            app.state.payment_verifier = payment_verifier

            storage = await stack.enter_async_context(
                create_storage_client(config.storage)
            )
            app.state.storage = storage

            embedder = create_embedder(config.embedding)
            stack.push_async_callback(embedder.aclose)
            app.state.embedder = embedder

            generator = create_generator(config.data_pipeline)
            stack.push_async_callback(generator.aclose)
            app.state.dataset_generator = generator

            validator_names = app.state.validator_names
            stack.push_async_callback(validator_names.aclose)
            await validator_names.start()
        except Exception as e:
            raise ApiServerLifespanError(
                f"failed to open dependencies during startup: {e}"
            ) from e

        logger.info(
            f"api server ready on {config.host}:{config.port} "
            f"commit={config.commit_hash}"
        )
        yield
        logger.info("api server shutting down")


def create_api_server(config: ApiServerConfig | None = None) -> FastAPI:
    """Build the FastAPI app, lifespan, middleware, handlers, and routers.

    When ``config`` is ``None``, falls back to
    :func:`parse_api_server_config_from_env` with ``commit_hash`` set to
    ``"unknown"`` so tests that do not exercise the git-rev path can
    skip resolving it.
    """
    if config is None:
        config = parse_api_server_config_from_env(commit_hash="unknown")

    app = FastAPI(
        title="Ditto API",
        version=ditto.__version__,
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url=None,
        lifespan=_lifespan,
    )
    app.state.config = config
    app.state.commit_hash = config.commit_hash
    # The object exists even when lifespan is skipped in unit tests. Its
    # snapshot path is synchronous and disabled by default; production lifespan
    # starts the optional background refresher without blocking API startup.
    app.state.validator_names = create_validator_names(config.validator_names)

    # Starlette inserts each middleware at position 0, so the LAST
    # add_middleware call ends up outermost on the wire. RequestIDMiddleware
    # must be outermost so its contextvar is live for every downstream
    # middleware + handler + log line, including any future auth that
    # short-circuits before reaching the app.
    # PublicCacheMiddleware is innermost: cache hits skip the endpoint (and
    # its database work) while request-id logging still records every hit.
    app.add_middleware(PublicCacheMiddleware)
    app.add_middleware(AuthPassThroughMiddleware)
    app.add_middleware(RequestIDMiddleware)

    register_exception_handlers(app)

    app.include_router(health_router)
    app.include_router(metrics_router)
    app.include_router(upload_router, prefix="/api/v1")
    app.include_router(retrieval_router, prefix="/api/v1")
    app.include_router(validator_router, prefix="/api/v1")
    app.include_router(screener_router, prefix="/api/v1")
    app.include_router(scoring_router, prefix="/api/v1")
    app.include_router(public_router, prefix="/api/v1")
    app.include_router(admin_quarantine_router, prefix="/api/v1")
    app.include_router(admin_copy_review_router, prefix="/api/v1")

    # Serve the public dashboard SPA same-origin at ``/`` so the platform is the
    # transparency front door (its ``/api/v1/public/*`` calls need no CORS). The
    # HTML is rendered once at boot with the wandb link injected; a missing file
    # just skips the route.
    if config.dashboard_enabled:
        dashboard_html = _render_dashboard(config.dashboard_wandb_url)
        if dashboard_html is None:
            logger.info(
                "dashboard SPA not found at %s; serving API only", _DASHBOARD_FILE
            )
        else:

            async def dashboard_response() -> HTMLResponse:
                return HTMLResponse(
                    content=dashboard_html,
                    headers={"Cache-Control": "public, max-age=300"},
                )

            @app.get("/", include_in_schema=False, response_class=HTMLResponse)
            async def dashboard() -> HTMLResponse:
                return await dashboard_response()

            # Stable public URLs for dashboard objects. These routes all serve the
            # SPA shell; the client resolves the identifier and opens the matching
            # agent, miner, validator, or screener view after its public data loads.
            for entity_kind in ("agents", "miners", "validators", "screeners"):
                app.add_api_route(
                    f"/{entity_kind}/{{entity_id}}",
                    dashboard_response,
                    methods=["GET"],
                    include_in_schema=False,
                    response_class=HTMLResponse,
                    name=f"dashboard_{entity_kind}",
                )

            if _DASHBOARD_IMAGE.is_file():

                @app.get(
                    "/assets/paperditto-512.png",
                    include_in_schema=False,
                    response_class=FileResponse,
                )
                async def dashboard_image() -> FileResponse:
                    return FileResponse(
                        _DASHBOARD_IMAGE,
                        media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"},
                    )

    return app
