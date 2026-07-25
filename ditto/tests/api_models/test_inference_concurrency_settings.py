"""The hosted-embedding concurrency board's contract.

Two things are pinned here that a future edit could quietly undo:

* the shipped defaults are a **raise**, not the old serialised values, so the
  improvement cannot be reduced to an opt-in knob by editing one number; and
* the board's ceilings are exactly the ceiling ``check_config`` enforces at
  boot, so the two validators can never disagree about what is acceptable.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ditto.api_models.inference_concurrency_settings import (
    DEFAULT_EMBEDDING_GLOBAL_CONCURRENCY,
    DEFAULT_EMBEDDING_PER_TICKET_CONCURRENCY,
    DEFAULT_EMBEDDING_PER_VALIDATOR_CONCURRENCY,
    MAX_EMBEDDING_GLOBAL_CONCURRENCY,
    AdminInferenceConcurrencySettingsRequest,
    InferenceConcurrencySettings,
)

# The values the hosted v7 lane ran at while it was still sized for a local
# Ollama container. Named here so the assertion below reads as the claim it is.
VESTIGIAL_OLLAMA_ERA_LIMITS = (1, 8, 32)


class TestDefaults:
    def test_defaults_are_a_raise_not_the_old_serialised_values(self) -> None:
        settings = InferenceConcurrencySettings()
        assert (
            settings.embedding_per_ticket_concurrency,
            settings.embedding_per_validator_concurrency,
            settings.embedding_global_concurrency,
        ) != VESTIGIAL_OLLAMA_ERA_LIMITS
        # Each one strictly above what it replaced: shipping a board whose
        # default reproduces the old behaviour would make the improvement
        # opt-in, which is the failure mode this change exists to avoid.
        assert settings.embedding_per_ticket_concurrency > 1
        assert settings.embedding_per_validator_concurrency > 8
        assert settings.embedding_global_concurrency > 32

    def test_defaults_match_the_documented_constants(self) -> None:
        settings = InferenceConcurrencySettings()
        assert (
            settings.embedding_per_ticket_concurrency
            == DEFAULT_EMBEDDING_PER_TICKET_CONCURRENCY
        )
        assert (
            settings.embedding_per_validator_concurrency
            == DEFAULT_EMBEDDING_PER_VALIDATOR_CONCURRENCY
        )
        assert (
            settings.embedding_global_concurrency
            == DEFAULT_EMBEDDING_GLOBAL_CONCURRENCY
        )

    def test_board_ceiling_matches_the_boot_time_check(self) -> None:
        """A revision must never be accepted that ``check_config`` would reject.

        ``check_config`` bounds the boot-time embedding limits at 128. If the
        board allowed more, an operator could apply a revision that works until
        the next restart and then refuses to boot.
        """
        assert MAX_EMBEDDING_GLOBAL_CONCURRENCY == 128


class TestHierarchy:
    def test_ticket_may_not_exceed_validator(self) -> None:
        with pytest.raises(ValidationError, match="may not exceed"):
            InferenceConcurrencySettings(
                embedding_per_ticket_concurrency=16,
                embedding_per_validator_concurrency=8,
                embedding_global_concurrency=64,
            )

    def test_validator_may_not_exceed_global(self) -> None:
        with pytest.raises(ValidationError, match="may not exceed"):
            InferenceConcurrencySettings(
                embedding_per_ticket_concurrency=4,
                embedding_per_validator_concurrency=64,
                embedding_global_concurrency=16,
            )

    def test_equal_limits_are_allowed(self) -> None:
        settings = InferenceConcurrencySettings(
            embedding_per_ticket_concurrency=8,
            embedding_per_validator_concurrency=8,
            embedding_global_concurrency=8,
        )
        assert settings.embedding_global_concurrency == 8

    def test_zero_is_refused(self) -> None:
        """A lane of zero would stall every v7 run rather than slow it."""
        with pytest.raises(ValidationError):
            InferenceConcurrencySettings(embedding_per_ticket_concurrency=0)

    def test_above_ceiling_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            InferenceConcurrencySettings(
                embedding_per_ticket_concurrency=129,
                embedding_per_validator_concurrency=129,
                embedding_global_concurrency=129,
            )


class TestWriteContract:
    def _request(self, **settings: int) -> AdminInferenceConcurrencySettingsRequest:
        return AdminInferenceConcurrencySettingsRequest(
            expected_revision=0,
            settings=InferenceConcurrencySettings(**settings),
            reason="widen the hosted embedding lane",
            confirmation="APPLY INFERENCE CONCURRENCY SETTINGS",
        )

    def test_partial_policy_is_refused_with_the_missing_fields_named(self) -> None:
        with pytest.raises(ValidationError, match="embedding_per_ticket_concurrency"):
            self._request(
                embedding_per_validator_concurrency=48,
                embedding_global_concurrency=96,
            )

    def test_complete_policy_is_accepted(self) -> None:
        request = self._request(
            embedding_per_ticket_concurrency=16,
            embedding_per_validator_concurrency=64,
            embedding_global_concurrency=128,
        )
        assert request.settings.embedding_per_ticket_concurrency == 16
