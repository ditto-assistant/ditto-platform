"""Canary: the test database is real, migrated, isolated, and drift-free.

Every assertion here protects a property the rest of the suite silently
assumes. If one of these fails, other failures elsewhere are probably
symptoms rather than causes.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from ditto.db.models import Base


async def test_tests_run_on_postgres_not_sqlite(session: AsyncSession) -> None:
    """The whole point. A regression here re-hides the #438 bug class."""
    assert session.get_bind().dialect.name == "postgresql"


async def test_schema_came_from_alembic_not_create_all(
    session: AsyncSession,
) -> None:
    """``alembic_version`` exists only if the real migration chain ran.

    ``Base.metadata.create_all`` builds the schema the models *claim*;
    only Alembic builds the schema production has.
    """
    head = await session.scalar(text("SELECT version_num FROM alembic_version"))
    assert head, "no alembic_version row: the template was not migrated"


async def test_models_declare_every_constraint_the_migrations_create(
    session: AsyncSession,
) -> None:
    """Guard against models-vs-migrations drift.

    Drift is invisible while tests build their schema from the models: the
    database enforces a rule production has and the tests never see it.
    That is not hypothetical -- ``screening_quarantines`` has three CHECKs
    (``manifest_digest``, ``finding_digest``, ``reason_code`` formats) and
    ``validator_tickets`` has one (``seed >= 0``) that ``models.py`` does
    not declare, so no test had ever exercised them.

    Ratchet, not a hard equality: tighten ``allowed`` as the gaps close,
    never loosen it.
    """
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateTable

    migrated = await session.scalar(
        text(
            "SELECT count(*) FROM pg_constraint con "
            "JOIN pg_class c ON c.oid = con.conrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE con.contype = 'c' AND n.nspname = 'public'"
        )
    )
    declared = sum(
        str(CreateTable(table).compile(dialect=postgresql.dialect())).count("CHECK ")
        for table in Base.metadata.sorted_tables
    )
    allowed = 4
    assert migrated is not None
    assert migrated - declared <= allowed, (
        f"models.py is missing {migrated - declared} CHECK constraints that the "
        f"migrations create (was {allowed}). Declare them on the model, or the "
        f"suite tests a schema production does not have."
    )


async def test_reset_restores_migration_seeded_rows(session: AsyncSession) -> None:
    """The migration chain plants real defaults; the reset must keep them.

    Under SQLite's ``create_all`` these tables were empty, so the suite ran
    against a baseline production never has.
    """
    seeded = await session.scalar(
        text("SELECT count(*) FROM artifact_release_settings_revisions")
    )
    assert seeded == 2, "migration-seeded artifact release revisions were lost"


async def test_reset_leaves_no_rows_from_the_previous_test(
    session: AsyncSession,
) -> None:
    """Paired with the writer below; order-independent by construction."""
    count = await session.scalar(text("SELECT count(*) FROM banned_hotkeys"))
    assert count == 0


async def test_reset_leaves_no_rows_from_the_previous_test_writer(
    session: AsyncSession,
) -> None:
    """Commits a row the sibling test above must never observe."""
    async with session.begin():
        await session.execute(
            text(
                "INSERT INTO banned_hotkeys (hotkey, reason, banned_at) "
                "VALUES ('canary', 'harness isolation canary', now())"
            )
        )
    assert await session.scalar(text("SELECT count(*) FROM banned_hotkeys")) == 1


async def test_select_for_update_actually_locks(
    session_maker: async_sessionmaker[AsyncSession],
) -> None:
    """``FOR UPDATE`` is emitted and honoured.

    SQLAlchemy's SQLite dialect emits no ``FOR UPDATE`` at all, so the 77
    ``with_for_update`` sites in production code were decoration under test.
    Two sessions contend for one row here; the second must block until the
    first commits, which is only observable with real concurrent writers.
    """
    async with session_maker() as setup, setup.begin():
        await setup.execute(
            text(
                "INSERT INTO banned_hotkeys (hotkey, reason, banned_at) "
                "VALUES ('lock-canary', 'row lock canary', now())"
            )
        )

    holder_locked = asyncio.Event()
    contender_acquired = asyncio.Event()

    async def holder() -> None:
        async with session_maker() as s, s.begin():
            await s.execute(
                text(
                    "SELECT 1 FROM banned_hotkeys WHERE hotkey = 'lock-canary' "
                    "FOR UPDATE"
                )
            )
            holder_locked.set()
            # If FOR UPDATE were a no-op the contender would finish here.
            await asyncio.sleep(0.25)
            assert not contender_acquired.is_set(), "FOR UPDATE did not block"

    async def contender() -> None:
        await holder_locked.wait()
        async with session_maker() as s, s.begin():
            await s.execute(
                text(
                    "SELECT 1 FROM banned_hotkeys WHERE hotkey = 'lock-canary' "
                    "FOR UPDATE"
                )
            )
            contender_acquired.set()

    async with asyncio.timeout(10):
        await asyncio.gather(holder(), contender())
    assert contender_acquired.is_set()


async def test_advisory_locks_exist(session: AsyncSession) -> None:
    """Fifteen production call sites branch on this being available.

    Under SQLite all fifteen took the no-op branch, so every advisory lock
    in the codebase had zero test coverage.
    """
    assert await session.scalar(text("SELECT pg_try_advisory_xact_lock(42)")) is True


async def test_agent_uniqueness_constraint_is_enforced(engine: AsyncEngine) -> None:
    """``agents_hotkey_name_version_key`` is ``ddl_if(dialect='postgresql')``.

    It did not exist under SQLite, so two agents sharing
    ``(miner_hotkey, name, version)`` committed happily -- while
    ``ditto/db/queries/agents.py:139`` comments that "the UNIQUE constraint
    remains the final invariant on both". It is now genuinely final.
    """
    from datetime import UTC, datetime
    from uuid import uuid4

    from ditto.api_models.agent_status import AgentStatus
    from ditto.db.models import Agent

    def _agent() -> Agent:
        return Agent(
            agent_id=uuid4(),
            miner_hotkey="dup-hotkey",
            name="dup-name",
            version=1,
            sha256=uuid4().hex + uuid4().hex,
            status=AgentStatus.UPLOADED,
            created_at=datetime.now(UTC),
        )

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s, s.begin():
        s.add(_agent())
    with pytest.raises(IntegrityError):
        async with maker() as s, s.begin():
            s.add(_agent())


async def test_timestamps_round_trip_timezone_aware(session: AsyncSession) -> None:
    """SQLite dropped ``tzinfo`` on the way out; asyncpg does not.

    Thirteen ``_as_utc`` helpers and 39 raw ``.replace(tzinfo=UTC)``
    coercions exist in production code to paper over the SQLite behaviour.
    They are now provably unnecessary on the read path.
    """
    from datetime import UTC, datetime

    async with session.begin():
        await session.execute(
            text(
                "INSERT INTO banned_hotkeys (hotkey, reason, banned_at) "
                "VALUES ('tz-canary', 'timezone canary', :ts)"
            ),
            {"ts": datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)},
        )
    from ditto.db.models import BannedHotkey

    row = await session.scalar(
        select(BannedHotkey).where(BannedHotkey.hotkey == "tz-canary")
    )
    assert row is not None
    assert row.banned_at.tzinfo is not None, "timestamp came back naive"
    assert row.banned_at == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
