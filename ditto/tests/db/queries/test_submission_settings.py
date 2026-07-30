"""Race and lifecycle tests for pre-payment upload admission."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.db.queries.agents import SubmissionCooldownError
from ditto.db.queries.submission_settings import (
    EffectiveSubmissionSettings,
    consume_or_enforce_upload_admission,
    reserve_upload_admission,
)

pytestmark = pytest.mark.asyncio


async def test_admission_remains_valid_during_extended_paid_upload_recovery(
    session: AsyncSession,
) -> None:
    now = datetime(2026, 7, 24, 20, 0, tzinfo=UTC)
    settings = EffectiveSubmissionSettings(revision=1, cooldown_seconds=3600)
    async with session.begin():
        admission = await reserve_upload_admission(
            session,
            miner_coldkey="coldkey",
            miner_hotkey="hotkey",
            sha256="a" * 64,
            settings=settings,
            now=now,
        )
    async with session.begin():
        await consume_or_enforce_upload_admission(
            session,
            miner_coldkey="coldkey",
            miner_hotkey="hotkey",
            sha256="a" * 64,
            admission_token=admission.token,
            settings=settings,
            now=now + timedelta(hours=23, minutes=59),
        )


async def test_reservation_is_idempotent_and_blocks_competing_series(
    session: AsyncSession,
) -> None:
    now = datetime(2026, 7, 24, 20, 0, tzinfo=UTC)
    settings = EffectiveSubmissionSettings(revision=4, cooldown_seconds=3600)
    async with session.begin():
        first = await reserve_upload_admission(
            session,
            miner_coldkey="coldkey",
            miner_hotkey="hotkey-a",
            sha256="a" * 64,
            settings=settings,
            now=now,
        )
    async with session.begin():
        repeated = await reserve_upload_admission(
            session,
            miner_coldkey="coldkey",
            miner_hotkey="hotkey-a",
            sha256="a" * 64,
            settings=settings,
            now=now + timedelta(seconds=5),
        )
    assert repeated.token == first.token

    with pytest.raises(SubmissionCooldownError):
        async with session.begin():
            await reserve_upload_admission(
                session,
                miner_coldkey="coldkey",
                miner_hotkey="hotkey-b",
                sha256="b" * 64,
                settings=settings,
                now=now + timedelta(seconds=10),
            )


async def test_matching_token_is_consumed_and_legacy_upload_cannot_steal_slot(
    session: AsyncSession,
) -> None:
    now = datetime(2026, 7, 24, 20, 0, tzinfo=UTC)
    settings = EffectiveSubmissionSettings(revision=1, cooldown_seconds=3600)
    async with session.begin():
        admission = await reserve_upload_admission(
            session,
            miner_coldkey="coldkey",
            miner_hotkey="hotkey-a",
            sha256="a" * 64,
            settings=settings,
            now=now,
        )
    with pytest.raises(SubmissionCooldownError):
        async with session.begin():
            await consume_or_enforce_upload_admission(
                session,
                miner_coldkey="coldkey",
                miner_hotkey="hotkey-a",
                sha256="a" * 64,
                admission_token=None,
                settings=settings,
                now=now + timedelta(seconds=1),
            )
    async with session.begin():
        await consume_or_enforce_upload_admission(
            session,
            miner_coldkey="coldkey",
            miner_hotkey="hotkey-a",
            sha256="a" * 64,
            admission_token=admission.token,
            settings=settings,
            now=now + timedelta(seconds=2),
        )


async def test_verified_payment_rotates_reservation_to_new_archive(
    session: AsyncSession,
) -> None:
    now = datetime(2026, 7, 24, 20, 0, tzinfo=UTC)
    settings = EffectiveSubmissionSettings(revision=4, cooldown_seconds=3600)
    async with session.begin():
        original = await reserve_upload_admission(
            session,
            miner_coldkey="coldkey",
            miner_hotkey="hotkey",
            sha256="a" * 64,
            settings=settings,
            now=now,
        )
    async with session.begin():
        replacement = await reserve_upload_admission(
            session,
            miner_coldkey="coldkey",
            miner_hotkey="hotkey",
            sha256="b" * 64,
            settings=settings,
            replace_existing=True,
            now=now + timedelta(minutes=5),
        )

    assert replacement.token != original.token
    with pytest.raises(SubmissionCooldownError):
        async with session.begin():
            await consume_or_enforce_upload_admission(
                session,
                miner_coldkey="coldkey",
                miner_hotkey="hotkey",
                sha256="a" * 64,
                admission_token=original.token,
                settings=settings,
                now=now + timedelta(minutes=6),
            )
    async with session.begin():
        await consume_or_enforce_upload_admission(
            session,
            miner_coldkey="coldkey",
            miner_hotkey="hotkey",
            sha256="b" * 64,
            admission_token=replacement.token,
            settings=settings,
            now=now + timedelta(minutes=6),
        )


async def test_verified_payment_cannot_move_reservation_to_different_hotkey(
    session: AsyncSession,
) -> None:
    now = datetime(2026, 7, 24, 20, 0, tzinfo=UTC)
    settings = EffectiveSubmissionSettings(revision=4, cooldown_seconds=3600)
    async with session.begin():
        await reserve_upload_admission(
            session,
            miner_coldkey="coldkey",
            miner_hotkey="hotkey-a",
            sha256="a" * 64,
            settings=settings,
            now=now,
        )
    with pytest.raises(SubmissionCooldownError):
        async with session.begin():
            await reserve_upload_admission(
                session,
                miner_coldkey="coldkey",
                miner_hotkey="hotkey-b",
                sha256="b" * 64,
                settings=settings,
                replace_existing=True,
                now=now + timedelta(minutes=5),
            )
