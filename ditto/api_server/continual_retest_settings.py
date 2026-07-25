"""Short-TTL resolver for the operator-controlled continual-retest policy."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ditto.api_models.continual_retest_settings import ContinualRetestSettings

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from ditto.db.models import ContinualRetestSettingsRevision

logger = logging.getLogger(__name__)
DEFAULT_SETTINGS = ContinualRetestSettings()
DEFAULT_SETTINGS_TTL_SECONDS = 5.0


def settings_from_row(
    row: ContinualRetestSettingsRevision | None,
) -> ContinualRetestSettings:
    if row is None:
        return DEFAULT_SETTINGS
    try:
        return ContinualRetestSettings.model_validate(row.settings)
    except ValidationError:
        logger.warning(
            "continual retest settings revision %s is invalid; using defaults",
            getattr(row, "revision", "?"),
            exc_info=True,
        )
        return DEFAULT_SETTINGS


def aggregate_is_active(
    settings: ContinualRetestSettings, *, fleet_protocol_ready: bool
) -> bool:
    if settings.aggregate_mode == "enabled":
        return True
    if settings.aggregate_mode == "disabled":
        return False
    return fleet_protocol_ready


@dataclass
class _CacheEntry:
    settings: ContinualRetestSettings
    loaded_at: float


class ContinualRetestSettingsResolver:
    def __init__(self, *, ttl_seconds: float = DEFAULT_SETTINGS_TTL_SECONDS) -> None:
        self._ttl = max(0.0, ttl_seconds)
        self._cache: _CacheEntry | None = None
        self._lock = asyncio.Lock()

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    def invalidate(self) -> None:
        self._cache = None

    async def resolve(
        self, session_maker: async_sessionmaker | None
    ) -> ContinualRetestSettings:
        if session_maker is None:
            return DEFAULT_SETTINGS
        now = time.monotonic()
        if self._cache is not None and now - self._cache.loaded_at < self._ttl:
            return self._cache.settings
        async with self._lock:
            now = time.monotonic()
            if self._cache is not None and now - self._cache.loaded_at < self._ttl:
                return self._cache.settings
            from ditto.db.queries.continual_retest_settings import (
                latest_continual_retest_settings_revision,
            )

            async with session_maker() as session:
                row = await latest_continual_retest_settings_revision(session)
            settings = settings_from_row(row)
            self._cache = _CacheEntry(settings=settings, loaded_at=time.monotonic())
            return settings
