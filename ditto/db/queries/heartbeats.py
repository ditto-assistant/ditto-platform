"""Persistence for signed validator software heartbeats."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.benchmark_progress import (
    BenchmarkProgress,
    BenchmarkProgressStage,
)
from ditto.api_models.ticket_status import TicketStatus
from ditto.db.models import (
    Agent,
    ScreenerHeartbeat,
    ValidatorHeartbeat,
    ValidatorTicket,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_STAGE_ORDER: dict[BenchmarkProgressStage, int] = {
    "preparing": 0,
    "building_harness": 1,
    "starting_harness": 2,
    "running_benchmark": 3,
    "finalizing": 4,
    "submitting_result": 5,
    "failed_retrying": 6,
}


class HeartbeatProgressRegressionError(ValueError):
    """Raised when a newer heartbeat regresses progress for the same lease."""


@dataclass(frozen=True)
class ActiveValidatorWork:
    """One ticket-validated active heartbeat used by every public projection."""

    heartbeat: ValidatorHeartbeat
    ticket: ValidatorTicket
    agent: Agent
    progress: BenchmarkProgress | None


@dataclass(frozen=True)
class ActiveValidatorAssignment:
    """One live platform-issued validator assignment, independent of heartbeat."""

    ticket: ValidatorTicket
    agent: Agent


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _raw_percent(progress: BenchmarkProgress) -> int | None:
    if progress.completed is None or progress.total is None:
        return None
    return progress.completed * 100 // progress.total


def _parse_progress(value: dict) -> BenchmarkProgress:
    """Validate a JSON-column value through Pydantic's strict JSON path."""
    return BenchmarkProgress.model_validate_json(json.dumps(value))


def _validate_same_lease_progress(
    previous: BenchmarkProgress, current: BenchmarkProgress
) -> None:
    if previous.ticket_deadline != current.ticket_deadline:
        return
    # The validator's retry loop explicitly emits preparing before restarting
    # the same ticketed job. This one signed transition is the reset marker;
    # failed_retrying -> any other earlier stage remains a regression.
    if previous.stage == "failed_retrying" and current.stage == "preparing":
        return
    if _STAGE_ORDER[current.stage] < _STAGE_ORDER[previous.stage]:
        raise HeartbeatProgressRegressionError(
            "benchmark stage cannot regress for the same ticket lease"
        )
    if previous.total is not None and current.total != previous.total:
        raise HeartbeatProgressRegressionError(
            "benchmark total cannot change for the same ticket lease"
        )
    if previous.completed is not None and (
        current.completed is None or current.completed < previous.completed
    ):
        raise HeartbeatProgressRegressionError(
            "benchmark completed count cannot regress for the same ticket lease"
        )
    previous_percent = _raw_percent(previous)
    current_percent = _raw_percent(current)
    if previous_percent is not None and (
        current_percent is None or current_percent < previous_percent
    ):
        raise HeartbeatProgressRegressionError(
            "benchmark percent cannot regress for the same ticket lease"
        )


async def upsert_validator_heartbeat(
    session: AsyncSession,
    *,
    validator_hotkey: str,
    software_version: str,
    protocol_version: int,
    code_digest: str,
    state: str,
    active_agent_id: UUID | None,
    system_metrics: dict | None,
    benchmark_progress: dict | None,
    reported_at: datetime,
    seen_at: datetime,
    signature: str,
) -> tuple[ValidatorHeartbeat, bool]:
    """Persist only a strictly newer heartbeat; return ``(row, accepted)``."""
    row = await session.scalar(
        select(ValidatorHeartbeat)
        .where(ValidatorHeartbeat.validator_hotkey == validator_hotkey)
        .with_for_update()
    )
    is_new = False
    if row is None:
        values = {
            "validator_hotkey": validator_hotkey,
            "software_version": software_version,
            "protocol_version": protocol_version,
            "code_digest": code_digest,
            "state": state,
            "active_agent_id": active_agent_id,
            "first_seen_at": seen_at,
            "system_metrics": system_metrics,
            "benchmark_progress": benchmark_progress,
            "benchmark_progress_reported": benchmark_progress is not None,
            "benchmark_progress_agent_id": (
                active_agent_id if benchmark_progress is not None else None
            ),
            "reported_at": reported_at,
            "seen_at": seen_at,
            "signature": signature,
        }
        dialect_name = session.get_bind().dialect.name
        inserted: str | None = None
        if dialect_name == "postgresql":
            statement = (
                postgresql_insert(ValidatorHeartbeat)
                .values(**values)
                .on_conflict_do_nothing(index_elements=["validator_hotkey"])
                .returning(ValidatorHeartbeat.validator_hotkey)
            )
            inserted = await session.scalar(statement)
        elif dialect_name == "sqlite":
            sqlite_statement = (
                sqlite_insert(ValidatorHeartbeat)
                .values(**values)
                .on_conflict_do_nothing(index_elements=["validator_hotkey"])
                .returning(ValidatorHeartbeat.validator_hotkey)
            )
            inserted = await session.scalar(sqlite_statement)
        if dialect_name in {"postgresql", "sqlite"}:
            if inserted is not None:
                row = await session.get(ValidatorHeartbeat, validator_hotkey)
                assert row is not None
                return row, True
            row = await session.scalar(
                select(ValidatorHeartbeat)
                .where(ValidatorHeartbeat.validator_hotkey == validator_hotkey)
                .with_for_update()
            )
        if row is None:
            row = ValidatorHeartbeat(
                validator_hotkey=validator_hotkey, first_seen_at=seen_at
            )
            session.add(row)
            is_new = True
    if not is_new:
        assert row is not None
        existing_reported_at = row.reported_at
        if existing_reported_at.tzinfo is None:
            existing_reported_at = existing_reported_at.replace(tzinfo=UTC)
        if reported_at <= existing_reported_at:
            return row, False
        if row.benchmark_progress is not None and benchmark_progress is not None:
            try:
                previous_progress = _parse_progress(row.benchmark_progress)
                current_progress = _parse_progress(benchmark_progress)
            except ValidationError as error:
                raise HeartbeatProgressRegressionError(
                    "stored benchmark progress is malformed"
                ) from error
            if row.benchmark_progress_agent_id == active_agent_id:
                _validate_same_lease_progress(previous_progress, current_progress)
    row.software_version = software_version
    row.protocol_version = protocol_version
    row.code_digest = code_digest
    row.state = state
    row.active_agent_id = active_agent_id
    row.system_metrics = system_metrics
    if benchmark_progress is not None:
        row.benchmark_progress = benchmark_progress
        row.benchmark_progress_reported = True
        row.benchmark_progress_agent_id = active_agent_id
    else:
        # Retain the last signed progress and its separate agent binding as a
        # private monotonic floor across idle/polling/downgrade heartbeats. The
        # public view follows this flag and therefore clears immediately.
        row.benchmark_progress_reported = False
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


async def list_active_validator_work(
    session: AsyncSession,
    *,
    now: datetime,
    cutoff: datetime,
) -> list[ActiveValidatorWork]:
    """Return only fresh heartbeats still bound to the exact live ticket.

    Protocol v4 binds the deadline cryptographically. Legacy v2/v3 rows remain
    compatible, but only when their signed report is newer than the current ticket
    issuance; this prevents a stale row from reviving after a same-agent requeue.
    """
    rows = (
        await session.execute(
            select(ValidatorHeartbeat, ValidatorTicket, Agent)
            .join(
                ValidatorTicket,
                (ValidatorTicket.agent_id == ValidatorHeartbeat.active_agent_id)
                & (
                    ValidatorTicket.validator_hotkey
                    == ValidatorHeartbeat.validator_hotkey
                ),
            )
            .join(Agent, Agent.agent_id == ValidatorHeartbeat.active_agent_id)
            .where(
                ValidatorHeartbeat.state == "running_benchmark",
                ValidatorHeartbeat.active_agent_id.is_not(None),
                ValidatorHeartbeat.seen_at >= cutoff,
                ValidatorTicket.status == TicketStatus.ISSUED,
                ValidatorTicket.deadline > now,
                Agent.status == AgentStatus.EVALUATING,
            )
            .order_by(ValidatorHeartbeat.validator_hotkey)
        )
    ).all()
    active: list[ActiveValidatorWork] = []
    for heartbeat, ticket, agent in rows:
        progress: BenchmarkProgress | None = None
        if heartbeat.protocol_version >= 4:
            if not heartbeat.benchmark_progress_reported:
                if _aware(heartbeat.reported_at) <= _aware(ticket.issued_at).replace(
                    microsecond=0
                ):
                    continue
            else:
                if heartbeat.benchmark_progress is None:
                    continue
                try:
                    progress = _parse_progress(heartbeat.benchmark_progress)
                except ValidationError:
                    continue
                if progress.ticket_deadline != _aware(ticket.deadline):
                    continue
        elif _aware(heartbeat.reported_at) <= _aware(ticket.issued_at).replace(
            microsecond=0
        ):
            continue
        active.append(
            ActiveValidatorWork(
                heartbeat=heartbeat,
                ticket=ticket,
                agent=agent,
                progress=progress,
            )
        )
    return active


async def list_active_validator_assignments(
    session: AsyncSession,
    *,
    now: datetime,
) -> list[ActiveValidatorAssignment]:
    """Return platform assignment truth without inferring validator liveness."""
    rows = (
        await session.execute(
            select(ValidatorTicket, Agent)
            .join(Agent, Agent.agent_id == ValidatorTicket.agent_id)
            .where(
                ValidatorTicket.status == TicketStatus.ISSUED,
                ValidatorTicket.deadline > now,
            )
            .order_by(ValidatorTicket.validator_hotkey)
        )
    ).all()
    return [
        ActiveValidatorAssignment(ticket=ticket, agent=agent) for ticket, agent in rows
    ]


async def upsert_screener_heartbeat(
    session: AsyncSession,
    *,
    screener_hotkey: str,
    instance_id: str,
    software_version: str,
    protocol_version: int,
    policy_version: int,
    state: str,
    active_agent_id: UUID | None,
    screening_progress: dict | None,
    system_metrics: dict | None,
    reported_at: datetime,
    seen_at: datetime,
    signature: str,
) -> tuple[ScreenerHeartbeat, bool]:
    """Persist only a strictly newer heartbeat for one (hotkey, instance)."""
    row = await session.get(ScreenerHeartbeat, (screener_hotkey, instance_id))
    if row is None:
        row = ScreenerHeartbeat(
            screener_hotkey=screener_hotkey,
            instance_id=instance_id,
            first_seen_at=seen_at,
        )
        session.add(row)
    else:
        existing_reported_at = row.reported_at
        if existing_reported_at.tzinfo is None:
            existing_reported_at = existing_reported_at.replace(tzinfo=UTC)
        if reported_at <= existing_reported_at:
            return row, False
    row.software_version = software_version
    row.protocol_version = protocol_version
    row.policy_version = policy_version
    row.state = state
    row.active_agent_id = active_agent_id
    # Reuse the existing JSON telemetry column. Legacy rows contain the raw
    # metrics object; active v2 rows use this private envelope so no migration is
    # needed and public projection still reconstructs fields from an allowlist.
    row.system_metrics = (
        {
            "system_metrics": system_metrics,
            "screening_progress": screening_progress,
        }
        if screening_progress is not None
        else system_metrics
    )
    row.reported_at = reported_at
    row.seen_at = seen_at
    row.signature = signature
    await session.flush()
    return row, True


async def list_screener_heartbeats(
    session: AsyncSession,
) -> list[ScreenerHeartbeat]:
    """Return every reporting screener instance, newest heartbeat first."""
    result = await session.scalars(
        select(ScreenerHeartbeat).order_by(
            ScreenerHeartbeat.seen_at.desc(),
            ScreenerHeartbeat.screener_hotkey,
            ScreenerHeartbeat.instance_id,
        )
    )
    return list(result)


async def prune_stale_screener_heartbeats(
    session: AsyncSession,
    *,
    before: datetime,
) -> None:
    """Delete heartbeat rows last seen before ``before``.

    Bounds the per-instance table: a scaled-in fleet instance (unique name)
    stops reporting and would otherwise leave a permanent dead row.
    """
    await session.execute(
        delete(ScreenerHeartbeat).where(ScreenerHeartbeat.seen_at < before),
        execution_options={"synchronize_session": False},
    )
