"""Focused validation for safe public score-progress models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ditto.api_models.public import PublicProvisionalScore


def test_provisional_score_accepts_reproducible_safe_fields() -> None:
    score = PublicProvisionalScore(
        composite=0.625,
        seed="5585512758338063316",
        run_size="full",
        bench_version=2,
        datagen_version="v0.7.0",
        seed_source="on_chain",
        dataset_sha256="ab" * 32,
        accepted_at=datetime(2026, 7, 14, tzinfo=UTC),
        reproduction_command="generate -seed 123456789 -run-size full",
        verification_command="generate -seed 123456789 -run-size full -sha",
    )

    assert score.composite == pytest.approx(0.625)
    assert score.seed == "5585512758338063316"
    assert score.seed_source == "on_chain"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_size", "custom; rm -rf /"),
        ("datagen_version", "latest"),
        ("seed_source", "miner_supplied"),
        ("dataset_sha256", "not-a-hash"),
        ("seed", "1e9"),
    ],
)
def test_provisional_score_rejects_untrusted_command_inputs(
    field: str, value: str
) -> None:
    payload = {
        "composite": 0.625,
        "seed": "123456789",
        "run_size": "full",
        "bench_version": 2,
        "datagen_version": "v0.7.0",
        "seed_source": "on_chain",
        "dataset_sha256": "ab" * 32,
        "accepted_at": datetime(2026, 7, 14, tzinfo=UTC),
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        PublicProvisionalScore.model_validate(payload)
