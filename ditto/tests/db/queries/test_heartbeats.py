"""Unit tests for per-run keying + fail-open in the validator heartbeat upsert.

These exercise :func:`ditto.db.queries.heartbeats.upsert_validator_heartbeat`
directly against SQLite-in-memory, focusing on the run_token rebaseline and the
fail-open regression behaviour that the ``/validator/heartbeat`` endpoint relies
on (it no longer maps a regression to HTTP 409).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.benchmark_progress import BenchmarkProgress
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.db.models import Agent, ValidatorHeartbeat
from ditto.db.queries.heartbeats import upsert_validator_heartbeat

_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
_DEADLINE = datetime(2030, 1, 1, tzinfo=UTC)


def _progress(
    stage: str,
    *,
    completed: int | None = None,
    total: int | None = None,
    run_token: str | None = None,
) -> dict:
    return BenchmarkProgress(
        stage=stage,  # type: ignore[arg-type]
        completed=completed,
        total=total,
        ticket_deadline=_DEADLINE,
        run_token=run_token,
    ).model_dump(mode="json")


async def _seed_agent(session: AsyncSession) -> UUID:
    aid = uuid4()
    async with session.begin():
        session.add(
            Agent(
                agent_id=aid,
                miner_hotkey="5Miner",
                name="a",
                sha256="ab" * 32,
                status=AgentStatus.EVALUATING,
                screening_policy_version=SCREENING_POLICY_VERSION,
                created_at=datetime.now(UTC),
            )
        )
    return aid


async def _upsert(
    session: AsyncSession,
    agent_id: UUID,
    progress: dict | None,
    *,
    reported_at: datetime,
) -> tuple[ValidatorHeartbeat, bool]:
    async with session.begin():
        return await upsert_validator_heartbeat(
            session,
            validator_hotkey=_HOTKEY,
            software_version="0.1.0",
            protocol_version=4,
            code_digest="ab" * 32,
            state="running_benchmark",
            active_agent_id=agent_id,
            system_metrics=None,
            benchmark_progress=progress,
            reported_at=reported_at,
            seen_at=reported_at,
            signature="ab" * 64,
        )


def _stored(row: ValidatorHeartbeat) -> BenchmarkProgress:
    assert row.benchmark_progress is not None
    return BenchmarkProgress.model_validate_json(json.dumps(row.benchmark_progress))


async def test_same_run_regression_is_fail_open_and_keeps_previous(
    session: AsyncSession,
) -> None:
    agent_id = await _seed_agent(session)
    base = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
    _, accepted = await _upsert(
        session,
        agent_id,
        _progress("running_benchmark", completed=51, total=114, run_token="a" * 16),
        reported_at=base,
    )
    assert accepted

    row, accepted = await _upsert(
        session,
        agent_id,
        _progress("running_benchmark", completed=40, total=114, run_token="a" * 16),
        reported_at=base + timedelta(seconds=1),
    )
    # The heartbeat is accepted (never rejected) but the stored progress floor is
    # kept — the public display must not move backward.
    assert accepted
    assert row.benchmark_progress_reported is True
    assert _stored(row).completed == 51
    # Liveness still advances so the validator does not read as stale.
    assert row.seen_at == base + timedelta(seconds=1)


async def test_new_run_token_rebaselines_instead_of_regressing(
    session: AsyncSession,
) -> None:
    agent_id = await _seed_agent(session)
    base = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
    await _upsert(
        session,
        agent_id,
        _progress("running_benchmark", completed=51, total=114, run_token="a" * 16),
        reported_at=base,
    )

    # A fresh run_token means a new run (retry / next seed); its lower count is a
    # legitimate restart, not a regression.
    row, accepted = await _upsert(
        session,
        agent_id,
        _progress("running_benchmark", completed=1, total=114, run_token="b" * 16),
        reported_at=base + timedelta(seconds=1),
    )
    assert accepted
    assert _stored(row).completed == 1
    assert _stored(row).run_token == "b" * 16


async def test_same_run_monotonic_progress_is_accepted(
    session: AsyncSession,
) -> None:
    agent_id = await _seed_agent(session)
    base = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
    await _upsert(
        session,
        agent_id,
        _progress("running_benchmark", completed=51, total=114, run_token="a" * 16),
        reported_at=base,
    )
    row, accepted = await _upsert(
        session,
        agent_id,
        _progress("running_benchmark", completed=90, total=114, run_token="a" * 16),
        reported_at=base + timedelta(seconds=1),
    )
    assert accepted
    assert _stored(row).completed == 90
