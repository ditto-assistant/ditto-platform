"""Effective public artifact-release settings."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.db.models import ArtifactReleaseSettingsRevision

DEFAULT_ARTIFACT_RELEASE_EMBARGO_HOURS = 48
MIN_ARTIFACT_RELEASE_EMBARGO_HOURS = 6
MAX_ARTIFACT_RELEASE_EMBARGO_HOURS = 48


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
