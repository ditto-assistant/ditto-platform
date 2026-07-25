"""Effective miner submission settings and pre-payment reservations."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.db.models import SubmissionSettingsRevision, UploadAdmissionReservation
from ditto.db.queries.agents import SubmissionCooldownError, get_submission_retry_at

DEFAULT_SUBMISSION_COOLDOWN_SECONDS = 3600
MIN_SUBMISSION_COOLDOWN_SECONDS = 60
MAX_SUBMISSION_COOLDOWN_SECONDS = 86400
UPLOAD_ADMISSION_TTL = timedelta(minutes=30)


@dataclass(frozen=True)
class EffectiveSubmissionSettings:
    revision: int
    cooldown_seconds: int


@dataclass(frozen=True)
class UploadAdmission:
    token: uuid.UUID
    expires_at: datetime
    cooldown_seconds: int


async def latest_submission_settings(
    session: AsyncSession,
) -> SubmissionSettingsRevision | None:
    return await session.scalar(
        select(SubmissionSettingsRevision)
        .order_by(SubmissionSettingsRevision.revision.desc())
        .limit(1)
    )


async def effective_submission_settings(
    session: AsyncSession,
) -> EffectiveSubmissionSettings:
    latest = await latest_submission_settings(session)
    if latest is None:
        return EffectiveSubmissionSettings(
            revision=0, cooldown_seconds=DEFAULT_SUBMISSION_COOLDOWN_SECONDS
        )
    return EffectiveSubmissionSettings(
        revision=latest.revision, cooldown_seconds=latest.cooldown_seconds
    )


async def _lock_coldkey(session: AsyncSession, miner_coldkey: str) -> None:
    if session.get_bind().dialect.name == "postgresql":
        await session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:coldkey, 0))"),
            {"coldkey": miner_coldkey},
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


async def reserve_upload_admission(
    session: AsyncSession,
    *,
    miner_coldkey: str,
    miner_hotkey: str,
    sha256: str,
    settings: EffectiveSubmissionSettings,
    now: datetime | None = None,
) -> UploadAdmission:
    """Reserve one eligible coldkey slot so payment cannot lose a later race."""
    current = _utc(now or datetime.now(UTC))
    await _lock_coldkey(session, miner_coldkey)
    existing = await session.get(
        UploadAdmissionReservation, miner_coldkey, with_for_update=True
    )
    if existing is not None and _utc(existing.expires_at) <= current:
        await session.delete(existing)
        await session.flush()
        existing = None
    if existing is not None:
        if existing.miner_hotkey == miner_hotkey and existing.sha256 == sha256:
            return UploadAdmission(
                token=existing.token,
                expires_at=_utc(existing.expires_at),
                cooldown_seconds=existing.cooldown_seconds,
            )
        raise SubmissionCooldownError(_utc(existing.expires_at))

    retry_at = await get_submission_retry_at(
        session,
        miner_coldkey=miner_coldkey,
        cooldown=timedelta(seconds=settings.cooldown_seconds),
        now=current,
    )
    if retry_at is not None:
        raise SubmissionCooldownError(retry_at)

    row = UploadAdmissionReservation(
        miner_coldkey=miner_coldkey,
        token=uuid.uuid4(),
        miner_hotkey=miner_hotkey,
        sha256=sha256,
        settings_revision=settings.revision,
        cooldown_seconds=settings.cooldown_seconds,
        expires_at=current + UPLOAD_ADMISSION_TTL,
    )
    session.add(row)
    await session.flush()
    return UploadAdmission(
        token=row.token,
        expires_at=row.expires_at,
        cooldown_seconds=row.cooldown_seconds,
    )


async def consume_or_enforce_upload_admission(
    session: AsyncSession,
    *,
    miner_coldkey: str,
    miner_hotkey: str,
    sha256: str,
    admission_token: uuid.UUID | None,
    settings: EffectiveSubmissionSettings,
    now: datetime | None = None,
) -> None:
    """Consume a matching reservation, or enforce cooldown for a legacy client."""
    current = _utc(now or datetime.now(UTC))
    await _lock_coldkey(session, miner_coldkey)
    existing = await session.get(
        UploadAdmissionReservation, miner_coldkey, with_for_update=True
    )
    if existing is not None and _utc(existing.expires_at) <= current:
        await session.delete(existing)
        await session.flush()
        existing = None

    if admission_token is not None:
        if (
            existing is None
            or existing.token != admission_token
            or existing.miner_hotkey != miner_hotkey
            or existing.sha256 != sha256
        ):
            raise SubmissionCooldownError(
                _utc(existing.expires_at)
                if existing is not None
                else current + timedelta(seconds=60)
            )
        await session.delete(existing)
        await session.flush()
        return

    if existing is not None:
        raise SubmissionCooldownError(_utc(existing.expires_at))

    retry_at = await get_submission_retry_at(
        session,
        miner_coldkey=miner_coldkey,
        cooldown=timedelta(seconds=settings.cooldown_seconds),
        now=current,
    )
    if retry_at is not None:
        raise SubmissionCooldownError(retry_at)
