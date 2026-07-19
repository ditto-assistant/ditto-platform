from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.ticket_status import TicketStatus
from ditto.api_server.benchmark_rollout import (
    ensure_rolling_qualification,
    refresh_rolling_qualification,
)
from ditto.api_server.endpoints.admin_benchmark_rollout import (
    _require_v3_start_capacity,
    get_v3_rollout,
    start_v3_rollout,
)
from ditto.db.models import (
    Agent,
    Base,
    BenchmarkRollout,
    BenchmarkRolloutMember,
    Score,
    ValidatorHeartbeat,
    ValidatorTicket,
)
from ditto.db.queries.benchmark_rollout import (
    DatasetPin,
    RolloutSnapshotMember,
    active_bench_version,
    create_rollout_snapshot,
    heartbeat_supports_v3,
    issue_rollout_ticket,
    maybe_activate_rollout,
    open_rollout,
    rollout_state,
)
from ditto.db.queries.scores import list_eligible_ledger
from ditto.db.queries.screening import claim_screening_attempts

pytestmark = pytest.mark.asyncio


async def test_admin_status_read_does_not_start_rollout() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        state = await get_v3_rollout(None, session)
        assert state == {
            "active_version": 2,
            "desired_version": 2,
            "status": "inactive",
            "v3_capable_validator_count": 0,
            "current_hybrid_top_five": [],
            "qualification_converged": False,
            "members": [],
        }
        count = await session.scalar(select(func.count(BenchmarkRollout.rollout_id)))
        assert count == 0
    await engine.dispose()


def _capabilities(now: datetime) -> tuple[dict, dict]:
    revision = "a" * 40
    capabilities = {
        "screened_images": True,
        "require_screened_image": False,
        "source_build_fallback": True,
        "full_stack_managed": False,
        "stack_updater": False,
        "sandbox_egress_restricted": True,
        "executor_isolation": "privileged_dind",
        "scorer_benchmarks": {
            "status": "fresh_verified",
            "supported_bench_versions": [2, 3],
            "observed_at": int(now.timestamp()),
            "software_version": "1.3.0",
            "source_revision": revision,
        },
    }
    components = {
        name: {
            "source_revision": revision if name == "dittobench_api" else "b" * 40,
            "version": "1.3.0" if name == "dittobench_api" else "1.2.0",
            "provenance": "committed_pin",
        }
        for name in (
            "ditto_subnet",
            "dittobench_api",
            "sandbox_docker",
            "model_relay",
            "pylon",
            "ollama",
        )
    }
    stack = {
        "mode": "source",
        "compose_schema": 1,
        "release_descriptor_digest": None,
        "components": components,
    }
    return capabilities, stack


async def _seed_rollout(session, now: datetime) -> tuple[list[UUID], BenchmarkRollout]:
    agent_ids = [uuid4() for _ in range(5)]
    members = []
    pins = {}
    for position, agent_id in enumerate(agent_ids, start=1):
        miner = f"miner-{position}"
        session.add(
            Agent(
                agent_id=agent_id,
                miner_hotkey=miner,
                name=f"agent-{position}",
                sha256=f"{position:x}" * 64,
                status=AgentStatus.SCORED,
                screening_policy_version=9,
                screened_image_sha256=f"{position:x}" * 64,
                screened_image_size_bytes=1024,
                screened_image_id="sha256:" + f"{position:x}" * 64,
                screened_image_ref=f"ditto-screen/{agent_id}:latest",
                screened_image_upload_id=uuid4(),
                screened_image_verified_at=now,
                created_at=now + timedelta(seconds=position),
            )
        )
        members.append(
            RolloutSnapshotMember(
                agent_id=agent_id,
                miner_hotkey=miner,
                composite=1 - position / 100,
            )
        )
        pins[agent_id] = DatasetPin(
            seed=position,
            sha256="c" * 64,
            run_size="full",
        )
        for validator in range(3):
            session.add(
                Score(
                    agent_id=agent_id,
                    bench_version=2,
                    validator_hotkey=f"legacy-{validator}",
                    run_id=f"v2-{position}-{validator}",
                    signature="aa",
                    seed=position,
                    composite=0.5 + position / 100,
                    tool_mean=0.5,
                    memory_mean=0.5,
                    median_ms=1,
                    n=114,
                    details={"bench_version": 2},
                    generated_at=now,
                )
            )
    await session.flush()
    rollout = await create_rollout_snapshot(
        session, members=members, datasets=pins, now=now
    )
    capabilities, stack = _capabilities(now)
    for hotkey in ("validator-a", "validator-b", "validator-c"):
        session.add(
            ValidatorHeartbeat(
                validator_hotkey=hotkey,
                software_version="1.0.0",
                protocol_version=8,
                code_digest="d" * 64,
                state="polling",
                first_seen_at=now,
                reported_at=now,
                seen_at=now,
                signature="ab" * 64,
                capabilities=capabilities,
                stack=stack,
            )
        )
    await session.flush()
    return agent_ids, rollout


async def test_five_agents_remain_v2_at_two_of_three_then_activate_atomically() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    async with maker() as session, session.begin():
        agent_ids, rollout = await _seed_rollout(session, now)
        v2_only_id = uuid4()
        session.add(
            Agent(
                agent_id=v2_only_id,
                miner_hotkey="miner-v2-only",
                name="v2-only",
                sha256="e" * 64,
                status=AgentStatus.SCORED,
                screening_policy_version=9,
                created_at=now + timedelta(minutes=1),
            )
        )
        for validator in range(3):
            session.add(
                Score(
                    agent_id=v2_only_id,
                    bench_version=2,
                    validator_hotkey=f"legacy-{validator}",
                    run_id=f"v2-only-{validator}",
                    signature="dd",
                    seed=99,
                    composite=0.4,
                    tool_mean=0.4,
                    memory_mean=0.4,
                    median_ms=1,
                    n=114,
                    details={"bench_version": 2},
                    generated_at=now,
                )
            )
        await session.flush()
        heartbeat = await session.get(ValidatorHeartbeat, "validator-a")
        assert heartbeat is not None
        assert heartbeat_supports_v3(heartbeat, now=now)
        heartbeat.protocol_version = 7
        assert not heartbeat_supports_v3(heartbeat, now=now)
        heartbeat.protocol_version = 8

        for validator_index, hotkey in enumerate(("validator-a", "validator-b")):
            for agent_index in range(5):
                ticket = await issue_rollout_ticket(
                    session,
                    validator_hotkey=hotkey,
                    now=now,
                    ttl=timedelta(minutes=90),
                )
                assert ticket is not None
                assert ticket.agent_id == agent_ids[agent_index]
                session.add(
                    Score(
                        agent_id=ticket.agent_id,
                        bench_version=3,
                        validator_hotkey=hotkey,
                        run_id=f"v3-{validator_index}-{agent_index}",
                        signature="bb",
                        seed=agent_index + 1,
                        composite=0.7 + agent_index / 100,
                        tool_mean=0.7,
                        memory_mean=0.7,
                        median_ms=1,
                        n=114,
                        details={"bench_version": 3},
                        generated_at=now,
                    )
                )
                ticket.status = TicketStatus.SCORED
                await session.flush()

        state = await rollout_state(session)
        assert state["active_version"] == 2
        assert state["v3_capable_validator_count"] == 3
        assert [member["score_count"] for member in state["members"]] == [2] * 5
        assert await active_bench_version(session) == 2
        collecting_ledger = await list_eligible_ledger(session)
        assert {row.agent_id for row in collecting_ledger} == {
            *agent_ids,
            v2_only_id,
        }
        assert all(row.bench_version == 2 for row in collecting_ledger)

        activations = []
        for agent_index in range(5):
            ticket = await issue_rollout_ticket(
                session,
                validator_hotkey="validator-c",
                now=now,
                ttl=timedelta(minutes=90),
            )
            assert ticket is not None
            assert ticket.agent_id == agent_ids[agent_index]
            session.add(
                Score(
                    agent_id=ticket.agent_id,
                    bench_version=3,
                    validator_hotkey="validator-c",
                    run_id=f"v3-2-{agent_index}",
                    signature="cc",
                    seed=agent_index + 1,
                    composite=0.8 + agent_index / 100,
                    tool_mean=0.8,
                    memory_mean=0.8,
                    median_ms=1,
                    n=114,
                    details={"bench_version": 3},
                    generated_at=now,
                )
            )
            ticket.status = TicketStatus.SCORED
            await session.flush()
            activations.append(await maybe_activate_rollout(session, rollout, now=now))
            if agent_index == 0:
                # Agent 0 has a complete v3 quorum, but the temporary authority
                # pin (DESIRED_AUTHORITY_AT_QUORUM = False) keeps every agent on
                # its settled v2 median until the rollout activates.
                pinned = await list_eligible_ledger(session)
                by_agent = {row.agent_id: row for row in pinned}
                assert by_agent[agent_ids[0]].bench_version == 2
                assert by_agent[agent_ids[0]].composite == pytest.approx(0.51)
                assert all(
                    by_agent[agent_id].bench_version == 2 for agent_id in agent_ids[1:]
                )
                assert by_agent[v2_only_id].bench_version == 2

        assert activations == [False, False, False, False, True]
        assert await active_bench_version(session) == 3
        state = await rollout_state(session)
        assert state["status"] == "activated"
        assert [member["score_count"] for member in state["members"]] == [3] * 5
        v3_ledger = await list_eligible_ledger(session)
        assert len(v3_ledger) == 5
        assert v2_only_id not in {row.agent_id for row in v3_ledger}
        assert all(
            row.bench_version == 3
            and row.details is not None
            and row.details["bench_version"] == 3
            for row in v3_ledger
        )
    await engine.dispose()


async def test_ineligible_qualified_member_does_not_block_remaining_work() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    async with maker() as session, session.begin():
        agent_ids, _ = await _seed_rollout(session, now)
        agent = await session.get(Agent, agent_ids[2])
        assert agent is not None
        agent.status = AgentStatus.BANNED
        ticket = await issue_rollout_ticket(
            session,
            validator_hotkey="validator-a",
            now=now,
            ttl=timedelta(minutes=90),
        )
        assert ticket is not None
        assert ticket.agent_id != agent_ids[2]
        state = await rollout_state(session)
        assert state["status"] == "collecting"
        assert [UUID(member["agent_id"]) for member in state["members"]] == agent_ids
        agent.status = AgentStatus.SCORED
        ticket = await issue_rollout_ticket(
            session,
            validator_hotkey="validator-a",
            now=now,
            ttl=timedelta(minutes=90),
        )
        assert ticket is not None
        assert [
            UUID(member["agent_id"])
            for member in (await rollout_state(session))["members"]
        ] == agent_ids
    await engine.dispose()


async def test_rollout_screened_only_skips_and_releases_source_only_work() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    async with maker() as session, session.begin():
        agent_ids, rollout = await _seed_rollout(session, now)
        for agent_id in agent_ids:
            agent = await session.get(Agent, agent_id)
            assert agent is not None
            agent.screened_image_sha256 = None
            agent.screened_image_size_bytes = None
            agent.screened_image_id = None
            agent.screened_image_ref = None
            agent.screened_image_upload_id = None
            agent.screened_image_verified_at = None
        screened = await session.get(Agent, agent_ids[1])
        assert screened is not None
        screened.screened_image_sha256 = "12" * 32
        screened.screened_image_size_bytes = 123
        screened.screened_image_id = "sha256:" + "34" * 32
        screened.screened_image_ref = f"ditto-screen/{screened.agent_id}:latest"
        screened.screened_image_upload_id = uuid4()
        screened.screened_image_verified_at = now

        first = await issue_rollout_ticket(
            session,
            validator_hotkey="validator-a",
            now=now,
            ttl=timedelta(minutes=90),
            artifact_mode="screened_only",
        )
        assert first is not None
        assert first.agent_id == screened.agent_id

        incompatible = ValidatorTicket(
            agent_id=agent_ids[0],
            bench_version=rollout.desired_version,
            validator_hotkey="validator-b",
            status=TicketStatus.ISSUED,
            issued_at=now,
            deadline=now + timedelta(minutes=90),
            attempt_count=1,
            manual_retry_grants=0,
        )
        session.add(incompatible)
        await session.flush()
        replacement = await issue_rollout_ticket(
            session,
            validator_hotkey="validator-b",
            now=now,
            ttl=timedelta(minutes=90),
            artifact_mode="screened_only",
        )
        assert replacement is not None
        assert replacement.agent_id == screened.agent_id
        assert incompatible.status == TicketStatus.EXPIRED

        running = ValidatorTicket(
            agent_id=agent_ids[0],
            bench_version=rollout.desired_version,
            validator_hotkey="validator-c",
            status=TicketStatus.ISSUED,
            issued_at=now,
            deadline=now + timedelta(minutes=90),
            attempt_count=1,
            manual_retry_grants=0,
        )
        session.add(running)
        await session.flush()
        assert (
            await issue_rollout_ticket(
                session,
                validator_hotkey="validator-c",
                now=now,
                ttl=timedelta(minutes=90),
                artifact_mode="screened_only",
                validator_running_benchmark=True,
            )
            is None
        )
        assert running.status == TicketStatus.ISSUED
    await engine.dispose()


async def test_v3_score_drop_qualifies_and_rescreens_new_top_five_agent() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    rising_id = uuid4()
    async with maker() as session:
        async with session.begin():
            initial_ids, rollout = await _seed_rollout(session, now)
            session.add(
                Agent(
                    agent_id=rising_id,
                    miner_hotkey="miner-6",
                    name="rising-sixth",
                    sha256="f" * 64,
                    status=AgentStatus.SCORED,
                    screening_policy_version=8,
                    dataset_seed=66,
                    dataset_sha256="d" * 64,
                    dataset_run_size="full",
                    created_at=now + timedelta(minutes=1),
                )
            )
            for validator in range(3):
                session.add(
                    Score(
                        agent_id=rising_id,
                        bench_version=2,
                        validator_hotkey=f"legacy-{validator}",
                        run_id=f"v2-rising-{validator}",
                        signature="aa",
                        seed=66,
                        composite=0.505,
                        tool_mean=0.5,
                        memory_mean=0.5,
                        median_ms=1,
                        n=114,
                        details={"bench_version": 2},
                        generated_at=now,
                    )
                )
                session.add(
                    Score(
                        agent_id=initial_ids[0],
                        bench_version=3,
                        validator_hotkey=f"v3-{validator}",
                        run_id=f"v3-drop-{validator}",
                        signature="bb",
                        seed=1,
                        composite=0.1,
                        tool_mean=0.1,
                        memory_mean=0.1,
                        median_ms=1,
                        n=114,
                        details={"bench_version": 3},
                        generated_at=now,
                    )
                )

        generator = AsyncMock()
        generator.generate.return_value = "e" * 64
        assert (
            await refresh_rolling_qualification(
                session, generator=generator, now=now + timedelta(seconds=1)
            )
            == 1
        )
        async with session.begin():
            member = await session.get(
                BenchmarkRolloutMember, (rollout.rollout_id, rising_id)
            )
            assert member is not None
            assert member.position == 6
            claimed = await claim_screening_attempts(
                session,
                screener_hotkey="screener-1",
                now=now + timedelta(seconds=2),
                ttl=timedelta(minutes=70),
                limit=20,
            )
            assert rising_id in {agent.agent_id for agent, _attempt, _dup in claimed}
            rising = await session.get(Agent, rising_id)
            assert rising is not None
            assert rising.status == AgentStatus.SCORED
    await engine.dispose()


async def test_only_one_open_rollout_across_collecting_and_blocked_states() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        session.add_all(
            [
                BenchmarkRollout(
                    rollout_id=uuid4(),
                    from_version=2,
                    desired_version=3,
                    status="collecting",
                    cohort_size=5,
                ),
                BenchmarkRollout(
                    rollout_id=uuid4(),
                    from_version=2,
                    desired_version=3,
                    status="blocked_ineligible",
                    cohort_size=5,
                ),
            ]
        )
        with pytest.raises(IntegrityError):
            await session.flush()
        await session.rollback()
    await engine.dispose()


async def test_first_capable_validator_automatically_seeds_v3_work() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    async with maker() as session, session.begin():
        for position in range(1, 6):
            agent_id = uuid4()
            session.add(
                Agent(
                    agent_id=agent_id,
                    miner_hotkey=f"miner-auto-{position}",
                    name=f"auto-{position}",
                    sha256=f"{position:x}" * 64,
                    status=AgentStatus.SCORED,
                    screening_policy_version=9,
                    screened_image_sha256=f"{position:x}" * 64,
                    screened_image_size_bytes=1024,
                    screened_image_id="sha256:" + f"{position:x}" * 64,
                    screened_image_ref=f"ditto-screen/{agent_id}:latest",
                    screened_image_upload_id=uuid4(),
                    screened_image_verified_at=now,
                    dataset_seed=position,
                    dataset_sha256="c" * 64,
                    dataset_run_size="full",
                    created_at=now + timedelta(seconds=position),
                )
            )
            for validator in range(3):
                session.add(
                    Score(
                        agent_id=agent_id,
                        bench_version=2,
                        validator_hotkey=f"legacy-{validator}",
                        run_id=f"auto-v2-{position}-{validator}",
                        signature="aa",
                        seed=position,
                        composite=0.5 + position / 100,
                        tool_mean=0.5,
                        memory_mean=0.5,
                        median_ms=1,
                        n=114,
                        details={"bench_version": 2},
                        generated_at=now,
                    )
                )
        capabilities, stack = _capabilities(now)
        session.add(
            ValidatorHeartbeat(
                validator_hotkey="validator-auto",
                software_version="1.0.0",
                protocol_version=8,
                code_digest="d" * 64,
                state="polling",
                first_seen_at=now,
                reported_at=now,
                seen_at=now,
                signature="ab" * 64,
                capabilities=capabilities,
                stack=stack,
            )
        )

    generator = AsyncMock()
    generator.generate.return_value = "e" * 64
    assert await ensure_rolling_qualification(session, generator=generator, now=now)
    assert generator.generate.await_count == 5
    async with session.begin():
        rollout = await open_rollout(session)
        assert rollout is not None
        ticket = await issue_rollout_ticket(
            session,
            validator_hotkey="validator-auto",
            now=now,
            ttl=timedelta(minutes=90),
        )
        assert ticket is not None
        assert ticket.bench_version == 3

    # Repeated job polls are idempotent and do not render another dataset set.
    assert not await ensure_rolling_qualification(session, generator=generator, now=now)
    assert generator.generate.await_count == 5
    await engine.dispose()


async def test_admin_start_is_idempotent_after_unique_transition_activation() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    rollout_id = uuid4()
    async with maker() as session, session.begin():
        session.add(
            BenchmarkRollout(
                rollout_id=rollout_id,
                from_version=2,
                desired_version=3,
                status="activated",
                cohort_size=5,
                created_at=now,
                activated_at=now,
            )
        )
    async with maker() as session:
        state = await start_v3_rollout(None, session, object())  # type: ignore[arg-type]
        assert state["active_version"] == 3
        assert state["desired_version"] == 3
        assert state["status"] == "activated"
        count = await session.scalar(select(func.count(BenchmarkRollout.rollout_id)))
        assert count == 1
        session.add(
            BenchmarkRollout(
                rollout_id=uuid4(),
                from_version=2,
                desired_version=3,
                status="activated",
                cohort_size=5,
                created_at=now + timedelta(seconds=1),
                activated_at=now + timedelta(seconds=1),
            )
        )
        with pytest.raises(IntegrityError):
            await session.flush()
        await session.rollback()
    await engine.dispose()


@pytest.mark.parametrize("capable_count", [0, 1, 2])
async def test_v3_start_requires_two_capable_validators_and_matches_telemetry(
    capable_count: int,
) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    capabilities, stack = _capabilities(now)
    async with maker() as session, session.begin():
        for index in range(capable_count):
            session.add(
                ValidatorHeartbeat(
                    validator_hotkey=f"validator-{index}",
                    software_version="1.0.0",
                    protocol_version=8,
                    code_digest="d" * 64,
                    state="polling",
                    first_seen_at=now,
                    reported_at=now,
                    seen_at=now,
                    signature="ab" * 64,
                    capabilities=capabilities,
                    stack=stack,
                )
            )
        await session.flush()

        telemetry = await rollout_state(session, now=now)
        assert telemetry["v3_capable_validator_count"] == capable_count
        if capable_count < 2:
            with pytest.raises(HTTPException) as exc_info:
                await _require_v3_start_capacity(session, now=now)
            assert exc_info.value.status_code == 409
            assert "at least two" in str(exc_info.value.detail)
            assert await open_rollout(session) is None
        else:
            guarded = await _require_v3_start_capacity(session, now=now)
            assert guarded["v3_capable_validator_count"] == capable_count
    await engine.dispose()
