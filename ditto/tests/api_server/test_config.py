"""Unit tests for :mod:`ditto.api_server.config`."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ditto.api_server.config import check_config, parse_api_server_config_from_env
from ditto.api_server.errors import ApiServerConfigError
from ditto.api_server.validator_names import ValidatorNamesConfig
from ditto.tests.api_server.conftest import make_api_server_config


def _set_minimum_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the minimum env vars to make every sub-config parser succeed."""
    monkeypatch.setenv("POSTGRES_USER", "u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p")
    monkeypatch.setenv("POSTGRES_DB", "d")
    monkeypatch.setenv("PYLON_OPEN_ACCESS_TOKEN", "tok")
    monkeypatch.setenv(
        "DITTO_UPLOAD_PAYMENT_ADDRESS",
        "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
    )
    monkeypatch.setenv("STORAGE_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("STORAGE_BUCKET", "ditto-agents")
    monkeypatch.setenv("STORAGE_ACCESS_KEY", "minio")
    monkeypatch.setenv("STORAGE_SECRET_KEY", "miniominio")
    monkeypatch.setenv(
        "SCREENER_HOTKEY",
        "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
    )
    monkeypatch.setenv(
        "SCREENER_API_TOKEN", "test-screener-token-at-least-32-characters"
    )
    # Override unset by default; tested explicitly elsewhere.
    monkeypatch.delenv("TAO_PRICE_OVERRIDE_USD", raising=False)
    monkeypatch.delenv("DITTO_TAOSTATS_VALIDATOR_NAMES_URL", raising=False)
    monkeypatch.delenv("DITTO_TAOSTATS_API_KEY", raising=False)


class TestParseApiServerConfigFromEnv:
    """Tests for the env-var builder."""

    def test_defaults_apply_when_unset(self, monkeypatch: pytest.MonkeyPatch):
        _set_minimum_env(monkeypatch)
        monkeypatch.delenv("API_HOST", raising=False)
        monkeypatch.delenv("API_PORT", raising=False)
        monkeypatch.delenv("API_LOG_LEVEL", raising=False)

        config = parse_api_server_config_from_env(commit_hash="abc")

        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.log_level == "INFO"
        assert config.commit_hash == "abc"
        assert config.validator_names.url is None
        assert config.validator_names.api_key is None
        assert config.validator_compatibility.minimum_software_version == "0.7.0"
        assert config.validator_compatibility.minimum_protocol_version == 4
        assert config.validator_compatibility.heartbeat_max_age_seconds == 300

    def test_free_taostats_key_config_is_optional(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _set_minimum_env(monkeypatch)
        url = "https://api.taostats.io/api/dtao/validator/available/v1?netuid=118"
        monkeypatch.setenv("DITTO_TAOSTATS_VALIDATOR_NAMES_URL", url)
        monkeypatch.setenv("DITTO_TAOSTATS_API_KEY", "free-api-key")

        config = parse_api_server_config_from_env(commit_hash="abc")

        assert config.validator_names.url == url
        assert config.validator_names.api_key == "free-api-key"
        assert config.validator_names.enabled is True

    def test_overrides_picked_up(self, monkeypatch: pytest.MonkeyPatch):
        _set_minimum_env(monkeypatch)
        monkeypatch.setenv("API_HOST", "127.0.0.1")
        monkeypatch.setenv("API_PORT", "9000")
        monkeypatch.setenv("API_LOG_LEVEL", "debug")

        config = parse_api_server_config_from_env(commit_hash="abc")

        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.log_level == "DEBUG"

    def test_composition_with_sub_configs(self, monkeypatch: pytest.MonkeyPatch):
        _set_minimum_env(monkeypatch)
        monkeypatch.setenv("POSTGRES_USER", "alice")
        monkeypatch.setenv("PYLON_OPEN_ACCESS_TOKEN", "tok-xyz")

        config = parse_api_server_config_from_env(commit_hash="abc")

        assert config.postgres.user == "alice"
        assert config.chain.open_access_token == "tok-xyz"

    def test_non_integer_port_raises(self, monkeypatch: pytest.MonkeyPatch):
        """Parse-time failure: the value is not coercible to int."""
        _set_minimum_env(monkeypatch)
        monkeypatch.setenv("API_PORT", "not-a-port")

        with pytest.raises(ApiServerConfigError, match="API_PORT"):
            parse_api_server_config_from_env(commit_hash="abc")

    def test_missing_payment_address_raises(self, monkeypatch: pytest.MonkeyPatch):
        _set_minimum_env(monkeypatch)
        monkeypatch.delenv("DITTO_UPLOAD_PAYMENT_ADDRESS", raising=False)

        with pytest.raises(ApiServerConfigError, match="DITTO_UPLOAD_PAYMENT_ADDRESS"):
            parse_api_server_config_from_env(commit_hash="abc")

    @pytest.mark.parametrize(
        "bad",
        [
            "REPLACE_WITH_DITTO_SS58_ADDRESS",  # the .env.example placeholder
            "not-an-ss58",
            "5short",
            "0OIl" * 12 + "1234567890ab",  # SS58 forbidden chars 0/O/I/l
        ],
    )
    def test_malformed_payment_address_raises(
        self, monkeypatch: pytest.MonkeyPatch, bad: str
    ):
        _set_minimum_env(monkeypatch)
        monkeypatch.setenv("DITTO_UPLOAD_PAYMENT_ADDRESS", bad)

        with pytest.raises(ApiServerConfigError, match="SS58"):
            parse_api_server_config_from_env(commit_hash="abc")

    def test_pricing_sub_config_picked_up(self, monkeypatch: pytest.MonkeyPatch):
        _set_minimum_env(monkeypatch)
        monkeypatch.setenv("DITTO_UPLOAD_FEE_USD", "7.50")
        monkeypatch.setenv("DITTO_UPLOAD_FEE_BUFFER", "1.2")
        monkeypatch.setenv("PRICING_CACHE_TTL_SECONDS", "60")

        config = parse_api_server_config_from_env(commit_hash="abc")

        from decimal import Decimal

        assert config.pricing.fee_usd == Decimal("7.50")
        assert config.pricing.fee_buffer == Decimal("1.2")
        assert config.pricing.cache_ttl_seconds == 60

    def test_storage_sub_config_picked_up(self, monkeypatch: pytest.MonkeyPatch):
        _set_minimum_env(monkeypatch)
        monkeypatch.setenv("STORAGE_ENDPOINT_URL", "https://s3.example.com")
        monkeypatch.setenv("STORAGE_BUCKET", "custom-bucket")
        monkeypatch.setenv("STORAGE_REGION", "eu-west-1")
        monkeypatch.setenv("STORAGE_USE_TLS", "true")

        config = parse_api_server_config_from_env(commit_hash="abc")

        assert config.storage.endpoint_url == "https://s3.example.com"
        assert config.storage.bucket == "custom-bucket"
        assert config.storage.region == "eu-west-1"
        assert config.storage.use_tls is True


class TestCheckConfig:
    """Validation gates that the dataclass type system cannot enforce."""

    def test_valid_config_passes(self):
        check_config(make_api_server_config())

    def test_port_out_of_range_raises(self):
        config = replace(make_api_server_config(), port=0)
        with pytest.raises(ApiServerConfigError, match="port out of range"):
            check_config(config)

    def test_port_above_max_raises(self):
        config = replace(make_api_server_config(), port=70000)
        with pytest.raises(ApiServerConfigError, match="port out of range"):
            check_config(config)

    def test_unknown_log_level_raises(self):
        config = replace(make_api_server_config(), log_level="loud")
        with pytest.raises(ApiServerConfigError, match="log_level"):
            check_config(config)

    @pytest.mark.parametrize("version", ["latest", "v0.7.0", "0.7", ""])
    def test_validator_minimum_requires_stable_version(self, version: str):
        from ditto.api_server import ValidatorCompatibilityConfig

        config = replace(
            make_api_server_config(),
            validator_compatibility=ValidatorCompatibilityConfig(
                minimum_software_version=version,
                minimum_protocol_version=4,
                heartbeat_max_age_seconds=300,
            ),
        )
        with pytest.raises(ApiServerConfigError, match="stable X.Y.Z"):
            check_config(config)

    def test_screener_auth_may_be_disabled(self):
        from ditto.api_server import ScreenerAuthConfig

        config = replace(
            make_api_server_config(),
            screener_auth=ScreenerAuthConfig(hotkey=None, api_token=None),
        )
        check_config(config)

    def test_screener_auth_rejects_partial_config(self):
        from ditto.api_server import ScreenerAuthConfig

        config = replace(
            make_api_server_config(),
            screener_auth=ScreenerAuthConfig(
                hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                api_token=None,
            ),
        )
        with pytest.raises(ApiServerConfigError, match="must be set together"):
            check_config(config)

    def test_screener_auth_rejects_short_token(self):
        from ditto.api_server import ScreenerAuthConfig

        config = replace(
            make_api_server_config(),
            screener_auth=ScreenerAuthConfig(
                hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                api_token="too-short",
            ),
        )
        with pytest.raises(ApiServerConfigError, match="at least 32"):
            check_config(config)

    def test_validator_names_accept_only_taostats_https(self):
        config = replace(
            make_api_server_config(),
            validator_names=ValidatorNamesConfig(
                url="https://api.taostats.io/api/dtao/validator/available/v1?netuid=118",
                api_key="free-api-key",
            ),
        )
        check_config(config)

        for url in (
            "http://api.taostats.io/api/dtao/validator/available/v1?netuid=118",
            "https://taostats.io/subnets/118/metagraph",
            "https://example.com/taostats",
            "https://api.taostats.io/api/price/latest/v1?netuid=118",
            "https://api.taostats.io/api/dtao/validator/available/v1?netuid=13",
            "https://api.taostats.io:8443/api/dtao/validator/available/v1?netuid=118",
        ):
            rejected = replace(
                config,
                validator_names=replace(config.validator_names, url=url),
            )
            with pytest.raises(ApiServerConfigError, match="documented"):
                check_config(rejected)

    @pytest.mark.parametrize(
        "names",
        [
            ValidatorNamesConfig(
                url="https://api.taostats.io/api/dtao/validator/available/v1?netuid=118"
            ),
            ValidatorNamesConfig(api_key="free-api-key"),
        ],
    )
    def test_validator_names_reject_partial_auth_config(
        self, names: ValidatorNamesConfig
    ) -> None:
        config = replace(make_api_server_config(), validator_names=names)

        with pytest.raises(ApiServerConfigError, match="must be set together"):
            check_config(config)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("timeout_seconds", 5.1, "between 0.1 and 5"),
            ("retry_seconds", 59, "at least 60"),
            ("refresh_seconds", 299, "at least the retry"),
            ("max_stale_seconds", 3599, "at least the refresh"),
        ],
    )
    def test_validator_name_cache_bounds(
        self, field: str, value: float, message: str
    ) -> None:
        names = ValidatorNamesConfig(
            url="https://api.taostats.io/api/dtao/validator/available/v1?netuid=118",
            api_key="free-api-key",
        )
        if field == "timeout_seconds":
            names = replace(names, timeout_seconds=value)
        elif field == "retry_seconds":
            names = replace(names, retry_seconds=int(value))
        elif field == "refresh_seconds":
            names = replace(names, refresh_seconds=int(value))
        else:
            names = replace(names, max_stale_seconds=int(value))
        config = replace(
            make_api_server_config(),
            validator_names=names,
        )

        with pytest.raises(ApiServerConfigError, match=message):
            check_config(config)
