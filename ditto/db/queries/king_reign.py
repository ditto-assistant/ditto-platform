"""Write-once record of when each agent first became the KOTH champion.

The public source-release embargo is king-only. An agent's source is revealed
only if it has held the crown, and the window is anchored to the FIRST time it
took the throne, so a brief early reign still releases that agent's source one
window later. Recording happens on the validator score path; the public gate
here only reads it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.db.models import AgentKingship


def _as_utc(value: datetime) -> datetime:
    """Normalize SQLite's naive timestamps to the Postgres UTC contract."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


async def record_first_crowned(
    session: AsyncSession, *, agent_id: UUID, now: datetime
) -> None:
    """Record an agent's first coronation. Idempotent and write-once.

    A later re-coronation must NOT move ``first_crowned_at``: the public release
    deadline is anchored to the first reign, so a brief early stint on the throne
    still releases that agent's source one window later. Callers run this in a
    best-effort, isolated transaction, so a duplicate race (two validators
    crowning the same champion at once) is a harmless no-op.
    """
    if await session.get(AgentKingship, agent_id) is not None:
        return
    session.add(AgentKingship(agent_id=agent_id, first_crowned_at=now))


async def get_first_crowned(
    session: AsyncSession,
    *,
    agent_ids: list[UUID] | set[UUID] | tuple[UUID, ...],
) -> dict[UUID, datetime]:
    """Return ``agent_id -> first_crowned_at`` for agents that have held the crown."""
    if not agent_ids:
        return {}
    rows = (
        await session.execute(
            select(AgentKingship.agent_id, AgentKingship.first_crowned_at).where(
                AgentKingship.agent_id.in_(agent_ids)
            )
        )
    ).all()
    return {agent_id: _as_utc(crowned_at) for agent_id, crowned_at in rows}
