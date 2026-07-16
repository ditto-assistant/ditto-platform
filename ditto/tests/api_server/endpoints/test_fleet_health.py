"""Classification and privacy tests for public fleet-health reporting."""

from datetime import UTC, datetime, timedelta

import pytest

from ditto.api_models.public import PublicSystemMetrics
from ditto.api_server.endpoints.public import (
    _fleet_classification,
    _public_system_metrics,
)


@pytest.fixture
def healthy_metrics() -> PublicSystemMetrics:
    return PublicSystemMetrics(
        cpu_percent=20,
        memory_percent=40,
        disk_percent=55,
        docker_status="healthy",
        running_containers=3,
        unhealthy_containers=0,
    )


@pytest.mark.parametrize(
    ("state", "age", "metrics_kind", "expected"),
    [
        ("idle", timedelta(seconds=30), "healthy", (True, "available", "healthy")),
        ("idle", timedelta(seconds=30), "warning", (True, "available", "warning")),
        ("idle", timedelta(minutes=6), "healthy", (False, "stale", "healthy")),
        ("idle", timedelta(minutes=16), "healthy", (False, "offline", "healthy")),
        ("paused", timedelta(seconds=30), "healthy", (True, "paused", "healthy")),
        ("idle", timedelta(seconds=30), "missing", (True, "available", "unknown")),
        ("idle", timedelta(seconds=30), "partial", (True, "available", "unknown")),
    ],
)
def test_classifies_availability_without_turning_missing_metrics_into_outage(
    healthy_metrics: PublicSystemMetrics,
    state: str,
    age: timedelta,
    metrics_kind: str,
    expected: tuple[bool, str, str],
) -> None:
    now = datetime.now(UTC)
    metrics: PublicSystemMetrics | None = healthy_metrics
    if metrics_kind == "warning":
        metrics = healthy_metrics.model_copy(update={"disk_percent": 95})
    elif metrics_kind == "missing":
        metrics = None
    elif metrics_kind == "partial":
        metrics = healthy_metrics.model_copy(update={"docker_status": "unavailable"})
    assert (
        _fleet_classification(
            state=state,
            seen_at=now - age,
            now=now,
            metrics=metrics,
        )
        == expected
    )


def test_disk_usage_below_warning_threshold_is_healthy(
    healthy_metrics: PublicSystemMetrics,
) -> None:
    now = datetime.now(UTC)
    metrics = healthy_metrics.model_copy(update={"disk_percent": 90})

    assert _fleet_classification(
        state="idle",
        seen_at=now - timedelta(seconds=30),
        now=now,
        metrics=metrics,
    ) == (True, "available", "healthy")


def test_saturated_cpu_is_healthy_workload(
    healthy_metrics: PublicSystemMetrics,
) -> None:
    now = datetime.now(UTC)
    busy_metrics = healthy_metrics.model_copy(update={"cpu_percent": 100})

    assert _fleet_classification(
        state="running_benchmark",
        seen_at=now - timedelta(seconds=30),
        now=now,
        metrics=busy_metrics,
    ) == (True, "available", "healthy")


def test_malformed_stored_metrics_are_not_partially_exposed() -> None:
    raw = {
        "collected_at": int(datetime.now(UTC).timestamp()),
        "cpu_percent": 20,
        "memory_percent": 40,
        "disk_percent": 55,
        "docker": {
            "status": "healthy",
            "running_containers": 3,
            "unhealthy_containers": 0,
        },
        "hostname": "private-host",
    }
    assert _public_system_metrics(raw) is None
