"""Canonical ``AgentStatus`` lifecycle enum.

Lives in :mod:`ditto.api_models` (not ``ditto.db``) because it is a wire
value: it appears in HTTP response bodies *and* in the database column.
Keeping the source of truth here lets the wire models stay free of any
database dependency, so the same ``api_models`` package can be vendored
into the miner/validator side without pulling in SQLAlchemy.

:mod:`ditto.db.models` re-imports this enum for its ``agents.status``
column, and binds it to the Postgres ENUM type ``agentstatus``.
"""

from __future__ import annotations

from enum import StrEnum


class AgentStatus(StrEnum):
    """Lifecycle state machine values for an agent submission.

    Matches the Postgres ENUM type ``agentstatus``. :class:`enum.StrEnum`
    (Python 3.11+) makes values usable as plain strings so they round-trip
    through both the native PG ENUM type and the SQLite CHECK-constraint
    fallback used in unit tests, and serialize directly to JSON.
    """

    UPLOADED = "uploaded"
    SCREENING = "screening"
    SCREENING_PASSED = "screening_passed"
    SCREENING_FAILED = "screening_failed"
    EVALUATING = "evaluating"
    SCORED = "scored"
    LIVE = "live"
    ATH_PENDING_REVIEW = "ath_pending_review"
    BANNED = "banned"
