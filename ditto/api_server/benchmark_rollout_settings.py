"""Resolver for the operator-controlled benchmark-rollout policy.

Deliberately **not** cached behind a TTL the way
``ditto.api_server.continual_retest_settings`` is. That policy is consulted on
every validator job poll, so a short-TTL cache saves real work. This one is
consulted exactly once per rollout — at ``POST /admin/benchmark-rollout/{v}`` —
and the value is then frozen onto the rollout row. A cache would only add a
window in which an operator's fresh revision is silently ignored by the very
next start, which is the one moment it must not be.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ditto.api_models.benchmark_rollout_settings import BenchmarkRolloutSettings
from ditto.db.queries.benchmark_rollout_settings import (
    latest_benchmark_rollout_settings_revision,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from ditto.db.models import BenchmarkRolloutSettingsRevision

logger = logging.getLogger(__name__)
DEFAULT_SETTINGS = BenchmarkRolloutSettings()


def settings_from_row(
    row: BenchmarkRolloutSettingsRevision | None,
) -> BenchmarkRolloutSettings:
    """Decode a revision, falling back to the defaults on a corrupt payload.

    Fails **closed** onto the historical top-ten default rather than raising:
    an unreadable row must not be able to block an operator from starting a
    rollout, and the default is the behavior every prior rollout had.
    """
    if row is None:
        return DEFAULT_SETTINGS
    try:
        return BenchmarkRolloutSettings.model_validate(row.settings)
    except ValidationError:
        logger.warning(
            "benchmark rollout settings revision %s is invalid; using defaults",
            getattr(row, "revision", "?"),
            exc_info=True,
        )
        return DEFAULT_SETTINGS


async def resolve_benchmark_rollout_settings(
    session: AsyncSession,
) -> BenchmarkRolloutSettings:
    """The policy the *next* rollout start will freeze."""
    return settings_from_row(await latest_benchmark_rollout_settings_revision(session))
