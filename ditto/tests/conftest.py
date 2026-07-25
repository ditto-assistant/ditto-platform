"""Suite-wide test configuration.

Registers explicit ``sqlite3`` date/datetime adapters so the test suite
does not depend on the implicit ones CPython deprecated in 3.12.

Production runs on Postgres via asyncpg; SQLite is a test/dev-only
backend (in-memory ``sqlite+aiosqlite`` engines), so this registration
belongs here rather than in ``ditto.db``.
"""

from __future__ import annotations

import datetime as _dt
import sqlite3


def _adapt_date_iso(val: _dt.date) -> str:
    """Adapt :class:`datetime.date` to an ISO-8601 date string."""
    return val.isoformat()


def _adapt_datetime_iso(val: _dt.datetime) -> str:
    """Adapt :class:`datetime.datetime` to an ISO-8601 timestamp string.

    The separator is a space, not the ``"T"`` used by the bare
    ``isoformat()`` recipe in the sqlite3 docs. Two reasons, both
    load-bearing:

    * It reproduces CPython's deprecated default adapter byte for byte
      (``val.isoformat(" ")``), so stored text is unchanged by this fix.
    * SQLAlchemy's own SQLite ``DATETIME`` storage format is space
      separated. A ``"T"`` here would make timestamps written through
      raw ``text()`` binds sort and compare differently from those
      written through mapped columns, since SQLite compares them as
      text.

    ``isoformat`` keeps the UTC offset suffix on aware datetimes, so
    timezone information survives into the stored value exactly as it
    did before.
    """
    return val.isoformat(" ")


# Registration is process-global and idempotent. pytest-xdist runs each
# worker in its own process, and every worker imports this module.
sqlite3.register_adapter(_dt.date, _adapt_date_iso)
sqlite3.register_adapter(_dt.datetime, _adapt_datetime_iso)

# No matching converter is registered on purpose. Converters only run
# when the connection is opened with ``detect_types``, and SQLAlchemy
# never enables it (its docs actively discourage it) -- the read path is
# handled by the dialect's own result processors instead. A converter
# here would be unreachable code that misleads readers about how rows
# are decoded.
