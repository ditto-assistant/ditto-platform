"""Resolved configuration for the API server process."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from ditto.api_server.datapipeline import (
    DataPipelineConfig,
    parse_data_pipeline_config_from_env,
)
from ditto.api_server.embedding import (
    EmbeddingConfig,
    parse_embedding_config_from_env,
)
from ditto.api_server.errors import ApiServerConfigError
from ditto.api_server.pricing import PricingConfig, parse_pricing_config_from_env
from ditto.api_server.storage import StorageConfig, parse_storage_config_from_env
from ditto.api_server.validator_names import (
    ValidatorNamesConfig,
    parse_validator_names_config_from_env,
)
from ditto.chain import ChainConfig, parse_chain_config_from_env
from ditto.db import PostgresConfig, parse_postgres_config_from_env

# Substrate SS58 base58 alphabet, 47-48 chars. Same shape Pydantic
# enforces on the wire; mirrored here so a bad payment address fails
# boot instead of running with a placeholder.
_SS58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{47,48}$")


@dataclass(frozen=True)
class ScreenerAuthConfig:
    """Credentials accepted by the platform-operated screener endpoints."""

    hotkey: str | None
    """Dedicated screener SS58 hotkey. It does not need an on-chain permit."""

    api_token: str | None
    """Bearer token required on every screener request."""

    @property
    def enabled(self) -> bool:
        return self.hotkey is not None and self.api_token is not None


@dataclass(frozen=True)
class ApiServerConfig:
    """Resolved configuration for the API server process.

    Composition over flattening: ``postgres``, ``chain``, ``pricing``,
    and ``storage`` carry their own typed dataclasses so the same
    sub-configs feed validator daemon + smoke scripts unchanged.
    """

    host: str
    """Interface to bind. ``0.0.0.0`` for compose / cloud, ``127.0.0.1`` locally."""

    port: int
    """TCP port. Defaults to 8000; Pylon shifts to 8001 in compose."""

    log_level: str
    """Root logger level. One of the stdlib level names (``DEBUG``, ``INFO``,
    ``WARNING``, ``ERROR``, ``CRITICAL``)."""

    commit_hash: str
    """Git revision the process was built from, or ``"unknown"`` outside a checkout.

    Resolved by :mod:`ditto.api_server.__main__` via ``git rev-parse HEAD``
    before the FastAPI app is built, so :func:`create_api_server` can stash
    it on ``app.state.commit_hash`` for the ``/health`` endpoint.
    """

    upload_payment_address: str
    """Ditto-controlled SS58 receive address for upload fees
    (``DITTO_UPLOAD_PAYMENT_ADDRESS``). Required at boot."""

    postgres: PostgresConfig
    """Connection parameters for the platform database."""

    chain: ChainConfig
    """Pylon + subtensor settings for chain reads."""

    pricing: PricingConfig
    """CoinGecko oracle + upload-fee parameters."""

    storage: StorageConfig
    """S3-compatible object store parameters for uploaded tarballs."""

    embedding: EmbeddingConfig
    """Code-embedding client parameters. Disabled by default (no
    ``CODE_EMBEDDER_URL``), so the platform runs unchanged until an operator points
    it at a self-hosted TEI service."""

    data_pipeline: DataPipelineConfig
    """Generate-service client parameters. Disabled by default (no
    ``DATA_PIPELINE_URL``); when set, the platform generates one dataset per
    submission at job-ready and pins (seed, dataset_sha256, run_size) on the
    agent."""

    screener_auth: ScreenerAuthConfig
    """Dedicated signer and bearer token for the platform-operated screener."""

    validator_names: ValidatorNamesConfig
    """Optional, background-only Taostats display-name decoration."""

    admin_api_token: str | None = None
    """Bearer token for private Backroom/operator administration endpoints."""

    dashboard_enabled: bool = True
    """Serve the public dashboard SPA (``dashboard/index.html``) at ``/``.

    On by default so the platform doubles as the transparency front door
    (same-origin with the public read API — no CORS / separate host needed).
    Set ``DITTO_DASHBOARD_ENABLED=false`` to run headless (API only)."""

    dashboard_wandb_url: str = "https://wandb.ai/"
    """Public wandb project URL injected into the served dashboard's
    ``ditto:wandb-url`` meta tag (``DITTO_DASHBOARD_WANDB_URL``), e.g.
    ``https://wandb.ai/<entity>/ditto-sn118``. The committed HTML ships a bare
    default so the link still resolves before the project exists."""


_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

_TRUTHY = {"1", "true", "yes", "on"}


def parse_api_server_config_from_env(commit_hash: str) -> ApiServerConfig:
    """Build an :class:`ApiServerConfig` from ``API_*`` env vars plus
    the postgres + chain + pricing + storage sub-config parsers. Call
    :func:`check_config` after to validate ranges + set membership.

    Raises:
        ApiServerConfigError: When ``API_PORT`` is not parseable as int
            or ``DITTO_UPLOAD_PAYMENT_ADDRESS`` is unset.
    """
    host = os.environ.get("API_HOST", "0.0.0.0")
    raw_port = os.environ.get("API_PORT", "8000")
    log_level = os.environ.get("API_LOG_LEVEL", "INFO").upper()

    try:
        port = int(raw_port)
    except ValueError as e:
        raise ApiServerConfigError(
            f"API_PORT must be an integer, got {raw_port!r}"
        ) from e

    upload_payment_address = os.environ.get("DITTO_UPLOAD_PAYMENT_ADDRESS")
    if not upload_payment_address:
        raise ApiServerConfigError(
            "DITTO_UPLOAD_PAYMENT_ADDRESS must be set to a Ditto-controlled "
            "SS58 receive address"
        )
    if not _SS58_RE.fullmatch(upload_payment_address):
        raise ApiServerConfigError(
            "DITTO_UPLOAD_PAYMENT_ADDRESS does not look like an SS58 address; "
            "got a value that fails the base58 / length check"
        )

    dashboard_enabled = (
        os.environ.get("DITTO_DASHBOARD_ENABLED", "true").strip().lower() in _TRUTHY
    )
    dashboard_wandb_url = os.environ.get(
        "DITTO_DASHBOARD_WANDB_URL", "https://wandb.ai/"
    )
    screener_hotkey = os.environ.get("SCREENER_HOTKEY") or None
    screener_api_token = os.environ.get("SCREENER_API_TOKEN") or None

    return ApiServerConfig(
        host=host,
        port=port,
        log_level=log_level,
        commit_hash=commit_hash,
        upload_payment_address=upload_payment_address,
        postgres=parse_postgres_config_from_env(),
        chain=parse_chain_config_from_env(),
        pricing=parse_pricing_config_from_env(),
        storage=parse_storage_config_from_env(),
        embedding=parse_embedding_config_from_env(),
        data_pipeline=parse_data_pipeline_config_from_env(),
        screener_auth=ScreenerAuthConfig(
            hotkey=screener_hotkey,
            api_token=screener_api_token,
        ),
        validator_names=parse_validator_names_config_from_env(),
        admin_api_token=os.environ.get("DITTO_ADMIN_API_TOKEN") or None,
        dashboard_enabled=dashboard_enabled,
        dashboard_wandb_url=dashboard_wandb_url,
    )


def check_config(config: ApiServerConfig) -> None:
    """Validate port range + log-level set membership.

    Raises:
        ApiServerConfigError: When ``port`` is outside ``1..65535`` or
            ``log_level`` is not a stdlib level name.
    """
    if not 1 <= config.port <= 65535:
        raise ApiServerConfigError(f"port out of range: {config.port}")
    if config.log_level not in _VALID_LOG_LEVELS:
        raise ApiServerConfigError(
            f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}; "
            f"got {config.log_level!r}"
        )
    auth = config.screener_auth
    if (auth.hotkey is None) != (auth.api_token is None):
        raise ApiServerConfigError(
            "SCREENER_HOTKEY and SCREENER_API_TOKEN must be set together"
        )
    if auth.hotkey is not None and not _SS58_RE.fullmatch(auth.hotkey):
        raise ApiServerConfigError("SCREENER_HOTKEY is not a valid SS58 address")
    if auth.api_token is not None and len(auth.api_token) < 32:
        raise ApiServerConfigError("SCREENER_API_TOKEN must be at least 32 characters")
    if config.admin_api_token is not None and len(config.admin_api_token) < 32:
        raise ApiServerConfigError(
            "DITTO_ADMIN_API_TOKEN must be at least 32 characters"
        )
    names = config.validator_names
    if (names.url is None) != (names.api_key is None):
        raise ApiServerConfigError(
            "DITTO_TAOSTATS_VALIDATOR_NAMES_URL and DITTO_TAOSTATS_API_KEY "
            "must be set together"
        )
    if names.url is not None:
        parsed = urlparse(names.url)
        query = parse_qs(parsed.query)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "api.taostats.io"
            or parsed.port not in {None, 443}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path != "/api/dtao/validator/available/v1"
            or query.get("netuid") != ["118"]
        ):
            raise ApiServerConfigError(
                "DITTO_TAOSTATS_VALIDATOR_NAMES_URL must use the documented "
                "https://api.taostats.io/api/dtao/validator/available/v1"
                "?netuid=118 endpoint"
            )
    if not 0.1 <= names.timeout_seconds <= 5.0:
        raise ApiServerConfigError(
            "DITTO_TAOSTATS_TIMEOUT_SECONDS must be between 0.1 and 5"
        )
    if names.retry_seconds < 60:
        raise ApiServerConfigError("DITTO_TAOSTATS_RETRY_SECONDS must be at least 60")
    if names.refresh_seconds < names.retry_seconds:
        raise ApiServerConfigError(
            "DITTO_TAOSTATS_REFRESH_SECONDS must be at least the retry interval"
        )
    if names.max_stale_seconds < names.refresh_seconds:
        raise ApiServerConfigError(
            "DITTO_TAOSTATS_MAX_STALE_SECONDS must be at least the refresh interval"
        )
