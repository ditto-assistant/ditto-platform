"""Reproduce-under-transform audit verdict at score finalization (v3 Part A4)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ditto.api_server.endpoints.public import _safe_transform_robustness
from ditto.api_server.endpoints.validator import (
    AUDIT_MIN_PAIRS,
    AUDIT_MIN_ROBUSTNESS,
    _transform_audit_verdict,
)


def score(robustness: float | None, pairs: int = AUDIT_MIN_PAIRS):
    details: dict = {}
    if robustness is not None:
        details["transform_robustness_median"] = robustness
        details["audit_case_count"] = pairs
    return SimpleNamespace(details=details)


def test_verdict_uses_the_median_across_validators() -> None:
    """No single validator's number decides an agent's fate, matching how the
    platform already finalizes on the median composite."""
    median, pairs, failed = _transform_audit_verdict(
        [score(0.1), score(0.2), score(0.9)]
    )
    assert median == 0.2
    assert failed is True
    assert pairs == 3 * AUDIT_MIN_PAIRS

    # One low validator among three does not trip the hold.
    median, _, failed = _transform_audit_verdict([score(0.1), score(0.9), score(0.9)])
    assert median == 0.9
    assert failed is False


def test_absent_metric_is_not_a_failed_audit() -> None:
    """An older scoring engine reports nothing. Holding an agent on that would
    quarantine a miner for the platform's own upgrade lag."""
    median, pairs, failed = _transform_audit_verdict([score(None), score(None)])
    assert (median, pairs, failed) == (None, 0, False)


def test_thin_evidence_is_not_judged() -> None:
    """Too few audit pairs behind a value: one split would swing the rate."""
    thin = score(0.0, pairs=AUDIT_MIN_PAIRS - 1)
    assert _transform_audit_verdict([thin, thin, thin]) == (None, 0, False)


def test_honest_robustness_clears_the_floor() -> None:
    _, _, failed = _transform_audit_verdict([score(1.0), score(1.0), score(1.0)])
    assert failed is False
    assert AUDIT_MIN_ROBUSTNESS < 1.0


@pytest.mark.parametrize(
    "details",
    [
        {},
        {"transform_robustness": "nope"},
        {"transform_robustness": True},
        {"transform_robustness": 1.5},
        {"transform_robustness": -0.1},
    ],
)
def test_public_surfacing_rejects_malformed_values(details: dict) -> None:
    """A malformed blob must publish nothing rather than a bogus audit number."""
    assert _safe_transform_robustness(details) == (None, None)


def test_public_surfacing_reads_a_good_value() -> None:
    assert _safe_transform_robustness(
        {"transform_robustness": 0.75, "audit_case_count": 8}
    ) == (0.75, 8)


def test_public_surfacing_tolerates_a_missing_pair_count() -> None:
    """The robustness value still publishes; only the count is unknown."""
    assert _safe_transform_robustness({"transform_robustness": 0.75}) == (0.75, None)


def test_enforcement_is_off_by_default() -> None:
    """The transform-audit hold must not affect an agent's status by default.

    The 2026-07-18 calibration measured the brittle attacker at 0.863 and the
    honest model at 0.910 (sd 0.148, min 0.60): the cheater sits inside the
    honest spread, so at the 0.70 floor 16% of honest runs would be quarantined
    while almost no brittle ones are caught. Until a calibration shows real
    separation, the verdict is observational. If this flips to True without that
    evidence, legitimate miners pay for it.
    """
    from ditto.api_server.endpoints import validator as v

    assert v.TRANSFORM_AUDIT_ENFORCE is False
