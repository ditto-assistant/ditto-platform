"""Effective public artifact-release settings."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.db.models import ArtifactReleaseSettingsRevision

DEFAULT_ARTIFACT_RELEASE_EMBARGO_HOURS = 48
MIN_ARTIFACT_RELEASE_EMBARGO_HOURS = 6
# The ceiling is a range bound, not a recommendation: 48 hours stays the
# community-agreed operative default, and the extra headroom exists so an
# operator can hold the king's source private for a bounded stretch (a
# disclosure window, an unresolved dispute) without a code change.
MAX_ARTIFACT_RELEASE_EMBARGO_HOURS = 720


async def latest_artifact_release_settings(
    session: AsyncSession,
) -> ArtifactReleaseSettingsRevision | None:
    return await session.scalar(
        select(ArtifactReleaseSettingsRevision)
        .order_by(ArtifactReleaseSettingsRevision.revision.desc())
        .limit(1)
    )


async def artifact_release_embargo_hours(session: AsyncSession) -> int:
    latest = await latest_artifact_release_settings(session)
    return (
        latest.embargo_hours
        if latest is not None
        else DEFAULT_ARTIFACT_RELEASE_EMBARGO_HOURS
    )
