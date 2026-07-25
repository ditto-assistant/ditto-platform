"""Point the integration suite at this worker's own database.

Every file here calls :func:`ditto.db.create_db_engine` inline, with no
fixture, reading ambient ``POSTGRES_*``. Requesting ``worker_database``
autouse rewrites those variables to this worker's private clone, which is
what makes ``-n auto`` safe: the ``TRUNCATE ... CASCADE`` these tests run
now takes ``ACCESS EXCLUSIVE`` on tables nobody else can be holding row
locks on, so the lock-order inversion behind the intermittent
``DeadlockDetectedError`` cannot form.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _integration_database(worker_database: object) -> object:
    return worker_database
