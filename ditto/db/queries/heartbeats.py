"""Persistence for signed validator software heartbeats."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select

from ditto.db.models import ValidatorHeartbeat

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def upsert_validator_heartbeat(
    session: AsyncSession,
    *,
    validator_hotkey: str,
    software_version: str,
    protocol_version: int,
    code_digest: str,
    state: str,
    active_agent_id: UUID | None,
    reported_at: datetime,
    seen_at: datetime,
    signature: str,
) -> tuple[ValidatorHeartbeat, bool]:
    """Persist only a strictly newer heartbeat; return ``(row, accepted)``."""
    row = await session.get(ValidatorHeartbeat, validator_hotkey)
    if row is None:
        row = ValidatorHeartbeat(validator_hotkey=validator_hotkey)
        session.add(row)
    else:
        existing_reported_at = row.reported_at
        if existing_reported_at.tzinfo is None:
            existing_reported_at = existing_reported_at.replace(tzinfo=UTC)
        if reported_at <= existing_reported_at:
            return row, False
    row.software_version = software_version
    row.protocol_version = protocol_version
    row.code_digest = code_digest
    row.state = state
    row.active_agent_id = active_agent_id
    row.reported_at = reported_at
    row.seen_at = seen_at
    row.signature = signature
    await session.flush()
    return row, True


async def list_validator_heartbeats(
    session: AsyncSession,
) -> list[ValidatorHeartbeat]:
    """Return every reporting validator, newest heartbeat first."""
    result = await session.scalars(
        select(ValidatorHeartbeat).order_by(
            ValidatorHeartbeat.seen_at.desc(), ValidatorHeartbeat.validator_hotkey
        )
    )
    return list(result)
