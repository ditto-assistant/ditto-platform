from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ditto.api_models.admin_quarantine import AdminBenchmarkQualificationRequest
from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.ticket_status import TicketStatus
from ditto.api_server.benchmark_rollout import (
    ensure_rolling_qualification,
    refresh_rolling_qualification,
    rolling_qualification_blockers,
)
from ditto.api_server.endpoints.admin_benchmark_rollout import (
    AdminRolloutStartRequest,
    AdminRolloutSupersedeRequest,
    _require_rollout_start_capacity,
    get_rollout,
    get_rollout_control,
    start_rollout,
    supersede_rollout,
)
from ditto.api_server.endpoints.admin_quarantine import (
    inspect_benchmark_qualification,
    qualify_benchmark_rollout,
)
from ditto.db.models import (
    Agent,
    Base,
    BenchmarkDataset,
    BenchmarkRollout,
    BenchmarkRolloutAudit,
    BenchmarkRolloutMember,
    EvaluationPayment,
    Score,
    ValidatorHeartbeat,
    ValidatorTicket,
)
from ditto.db.queries.benchmark_rollout import (
    CANARY_BENCH_VERSION,
    MIN_DESIRED_AUTHORITY_AGENTS,
    DatasetPin,
    RolloutConflictError,
    RolloutSnapshotMember,
    active_bench_version,
    append_rollout_member,
    create_rollout_snapshot,
    heartbeat_supports_version,
    historical_rescore_cohort,
    issue_rollout_ticket,
    maybe_activate_rollout,
    open_rollout,
    rolling_top_five,
    rollout_state,
    select_active_bench_version,
    supersede_open_rollout,
)
from ditto.db.queries.scores import count_ranked_quorum_agents, list_eligible_ledger
from ditto.db.queries.screening import claim_screening_attempts

pytestmark = pytest.mark.asyncio


async def test_admin_status_read_does_not_start_rollout() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        state = await get_rollout(None, session, "v3")
        assert state == {
            "active_version": 2,
            "desired_version": 2,
            "status": "inactive",
            "capability_bench_version": 3,
            "ranked_quorum_agents": 0,
            "min_ranked_quorum_agents": 5,
            "canary_capable_validator_count": 0,
            "v3_capable_validator_count": 0,
            "current_hybrid_top_five": [],
            "qualification_converged": False,
            "cohort_size": 0,
            "cohort_ready_count": 0,
            "priority_cohort_size": 5,
            "priority_complete": False,
            "members": [],
        }
        count = await session.scalar(select(func.count(BenchmarkRollout.rollout_id)))
        assert count == 0

        control = await get_rollout_control(None, session)
        assert control["available_target_versions"] == [3, 4, 5, 6]
        contracts = control["contracts"]
        assert isinstance(contracts, list)
        assert [item["version"] for item in contracts] == [2, 3, 4, 5, 6]
        assert control["status"] == "inactive"
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
            "supported_bench_versions": [2, CANARY_BENCH_VERSION],
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


async def test_historical_rescore_cohort_fills_from_exactly_two_prior_eras() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    expected_v4: list[UUID] = []
    expected_v3: list[UUID] = []
    async with maker() as session, session.begin():
        for version, count in ((4, 6), (3, 10), (2, 5)):
            for rank in range(count):
                agent_id = uuid4()
                session.add(
                    Agent(
                        agent_id=agent_id,
                        miner_hotkey=f"miner-v{version}-{rank}",
                        name=f"agent-v{version}-{rank}",
                        sha256=f"{version:x}" * 64,
                        status=AgentStatus.SCORED,
                        screening_policy_version=9,
                        created_at=now + timedelta(seconds=version * 100 + rank),
                    )
                )
                for validator in range(3):
                    session.add(
                        Score(
                            agent_id=agent_id,
                            bench_version=version,
                            validator_hotkey=f"validator-{version}-{validator}",
                            run_id=f"run-{version}-{rank}-{validator}",
                            signature="aa",
                            seed=rank,
                            composite=1 - rank / 100,
                            tool_mean=0.5,
                            memory_mean=0.5,
                            median_ms=1,
                            n=114,
                            details={"bench_version": version},
                            generated_at=now,
                        )
                    )
                if version == 4:
                    expected_v4.append(agent_id)
                elif version == 3 and rank < 4:
                    expected_v3.append(agent_id)
        await session.flush()

        cohort = await historical_rescore_cohort(session, source_version=4)
        assert [member.agent_id for member in cohort] == [
            *expected_v4,
            *expected_v3,
        ]
        assert len(cohort) == 10
        assert not any("v2" in member.miner_hotkey for member in cohort)
    await engine.dispose()


async def test_rollout_idles_validator_until_fleet_finishes_priority_five() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    async with maker() as session, session.begin():
        priority_ids, rollout = await _seed_rollout(session, now)
        sixth_id = uuid4()
        session.add(
            Agent(
                agent_id=sixth_id,
                miner_hotkey="miner-sixth",
                name="sixth",
                sha256="e" * 64,
                status=AgentStatus.SCORED,
                screening_policy_version=9,
                screened_image_sha256="e" * 64,
                screened_image_size_bytes=1024,
                screened_image_id="sha256:" + "e" * 64,
                screened_image_ref=f"ditto-screen/{sixth_id}:latest",
                screened_image_upload_id=uuid4(),
                screened_image_verified_at=now,
                created_at=now + timedelta(minutes=1),
            )
        )
        rollout.cohort_size = 6
        assert await append_rollout_member(
            session,
            rollout=rollout,
            member=RolloutSnapshotMember(sixth_id, "miner-sixth", 0.4),
            dataset=DatasetPin(seed=6, sha256="e" * 64, run_size="full"),
            now=now,
        )

        for index, priority_id in enumerate(priority_ids):
            ticket = await issue_rollout_ticket(
                session,
                validator_hotkey="validator-a",
                now=now,
                ttl=timedelta(minutes=90),
            )
            assert ticket is not None and ticket.agent_id == priority_id
            ticket.status = TicketStatus.SCORED
            session.add(
                Score(
                    agent_id=priority_id,
                    bench_version=CANARY_BENCH_VERSION,
                    validator_hotkey="validator-a",
                    run_id=f"priority-a-{index}",
                    signature="aa",
                    seed=index,
                    composite=0.8,
                    tool_mean=0.8,
                    memory_mean=0.8,
                    median_ms=1,
                    n=114,
                    details={"bench_version": CANARY_BENCH_VERSION},
                    generated_at=now,
                )
            )
            await session.flush()

        leaked_outsider_id = uuid4()
        session.add(
            Agent(
                agent_id=leaked_outsider_id,
                miner_hotkey="miner-leaked-outsider",
                name="leaked-outsider",
                sha256="d" * 64,
                status=AgentStatus.SCORED,
                screening_policy_version=9,
                created_at=now + timedelta(minutes=2),
            )
        )
        for validator in range(3):
            session.add(
                Score(
                    agent_id=leaked_outsider_id,
                    bench_version=CANARY_BENCH_VERSION,
                    validator_hotkey=f"outsider-{validator}",
                    run_id=f"outsider-{validator}",
                    signature="cc",
                    seed=1,
                    composite=0.9,
                    tool_mean=0.9,
                    memory_mean=0.9,
                    median_ms=1,
                    n=114,
                    details={"bench_version": CANARY_BENCH_VERSION},
                    generated_at=now,
                )
            )
        await session.flush()
        # An out-of-cohort v5 quorum left by the old fallback cannot count as a
        # substitute for an unfinished inherited leader.
        assert await active_bench_version(session) == 2

        # Validator A has exhausted its legal top-five work, but the fleet has
        # not completed those quorums. Rank six must not leak through.
        assert (
            await issue_rollout_ticket(
                session,
                validator_hotkey="validator-a",
                now=now,
                ttl=timedelta(minutes=90),
            )
            is None
        )

        for priority_id in priority_ids:
            for hotkey in ("validator-b", "validator-c"):
                session.add(
                    Score(
                        agent_id=priority_id,
                        bench_version=CANARY_BENCH_VERSION,
                        validator_hotkey=hotkey,
                        run_id=f"priority-{priority_id}-{hotkey}",
                        signature="bb",
                        seed=1,
                        composite=0.8,
                        tool_mean=0.8,
                        memory_mean=0.8,
                        median_ms=1,
                        n=114,
                        details={"bench_version": CANARY_BENCH_VERSION},
                        generated_at=now,
                    )
                )
        await session.flush()
        sixth_ticket = await issue_rollout_ticket(
            session,
            validator_hotkey="validator-a",
            now=now,
            ttl=timedelta(minutes=90),
        )
        assert sixth_ticket is not None and sixth_ticket.agent_id == sixth_id
        assert await active_bench_version(session) == CANARY_BENCH_VERSION
        assert not await maybe_activate_rollout(session, rollout, now=now)
    await engine.dispose()


async def test_parallel_rollout_slots_stay_distinct_inside_frozen_priority_five() -> (
    None
):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    async with maker() as session, session.begin():
        priority_ids, _rollout = await _seed_rollout(session, now)
        slot0 = await issue_rollout_ticket(
            session,
            validator_hotkey="validator-a",
            slot_id="slot-0",
            now=now,
            ttl=timedelta(minutes=90),
        )
        slot1 = await issue_rollout_ticket(
            session,
            validator_hotkey="validator-a",
            slot_id="slot-1",
            now=now,
            ttl=timedelta(minutes=90),
        )
        assert slot0 is not None and slot1 is not None
        assert slot0.agent_id != slot1.agent_id
        assert {slot0.agent_id, slot1.agent_id}.issubset(set(priority_ids))
    await engine.dispose()


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
        assert heartbeat_supports_version(heartbeat, now=now)
        heartbeat.protocol_version = 7
        assert not heartbeat_supports_version(heartbeat, now=now)
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
                        bench_version=CANARY_BENCH_VERSION,
                        validator_hotkey=hotkey,
                        run_id=f"v3-{validator_index}-{agent_index}",
                        signature="bb",
                        seed=agent_index + 1,
                        composite=0.7 + agent_index / 100,
                        tool_mean=0.7,
                        memory_mean=0.7,
                        median_ms=1,
                        n=114,
                        details={"bench_version": CANARY_BENCH_VERSION},
                        generated_at=now,
                    )
                )
                ticket.status = TicketStatus.SCORED
                await session.flush()

        state = await rollout_state(session)
        assert state["active_version"] == 2
        assert state["v3_capable_validator_count"] == 3
        assert [member["score_count"] for member in state["members"]] == [2] * 5
        # The authority-switch threshold is public: at 2/3 scores per member no
        # agent holds a ranked quorum yet, and the client reads the bar rather
        # than hardcoding it.
        assert state["ranked_quorum_agents"] == 0
        assert state["min_ranked_quorum_agents"] == MIN_DESIRED_AUTHORITY_AGENTS
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
                    bench_version=CANARY_BENCH_VERSION,
                    validator_hotkey="validator-c",
                    run_id=f"v3-2-{agent_index}",
                    signature="cc",
                    seed=agent_index + 1,
                    composite=0.8 + agent_index / 100,
                    tool_mean=0.8,
                    memory_mean=0.8,
                    median_ms=1,
                    n=114,
                    details={"bench_version": CANARY_BENCH_VERSION},
                    generated_at=now,
                )
            )
            ticket.status = TicketStatus.SCORED
            await session.flush()
            activations.append(await maybe_activate_rollout(session, rollout, now=now))
            if agent_index == 0:
                # Agent 0 has a complete desired-version quorum, but it is one
                # of MIN_DESIRED_AUTHORITY_AGENTS, so the threshold gate keeps
                # the whole ledger on its settled v2 medians.
                pinned = await list_eligible_ledger(session)
                by_agent = {row.agent_id: row for row in pinned}
                assert by_agent[agent_ids[0]].bench_version == 2
                assert by_agent[agent_ids[0]].composite == pytest.approx(0.51)
                assert all(
                    by_agent[agent_id].bench_version == 2 for agent_id in agent_ids[1:]
                )
                assert by_agent[v2_only_id].bench_version == 2

        assert activations == [False, False, False, False, True]
        assert await active_bench_version(session) == CANARY_BENCH_VERSION
        state = await rollout_state(session)
        assert state["status"] == "activated"
        assert [member["score_count"] for member in state["members"]] == [3] * 5
        v3_ledger = await list_eligible_ledger(session)
        assert len(v3_ledger) == 5
        assert v2_only_id not in {row.agent_id for row in v3_ledger}
        assert all(
            row.bench_version == CANARY_BENCH_VERSION
            and row.details is not None
            and row.details["bench_version"] == CANARY_BENCH_VERSION
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


async def test_rollout_preempts_idle_source_lease_only_when_target_work_exists() -> (
    None
):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    async with maker() as session, session.begin():
        agent_ids, rollout = await _seed_rollout(session, now)
        ordinary_id = uuid4()
        session.add(
            Agent(
                agent_id=ordinary_id,
                miner_hotkey="ordinary-miner",
                name="ordinary-v2-work",
                sha256="ab" * 32,
                status=AgentStatus.EVALUATING,
                screening_policy_version=9,
                created_at=now + timedelta(minutes=1),
            )
        )
        idle_source_ticket = ValidatorTicket(
            agent_id=ordinary_id,
            bench_version=2,
            validator_hotkey="validator-a",
            status=TicketStatus.ISSUED,
            issued_at=now,
            deadline=now + timedelta(minutes=90),
            attempt_count=1,
            manual_retry_grants=0,
        )
        running_source_ticket = ValidatorTicket(
            agent_id=ordinary_id,
            bench_version=2,
            validator_hotkey="validator-b",
            status=TicketStatus.ISSUED,
            issued_at=now,
            deadline=now + timedelta(minutes=90),
            attempt_count=1,
            manual_retry_grants=0,
        )
        no_target_source_ticket = ValidatorTicket(
            agent_id=ordinary_id,
            bench_version=2,
            validator_hotkey="validator-c",
            status=TicketStatus.ISSUED,
            issued_at=now,
            deadline=now + timedelta(minutes=90),
            attempt_count=1,
            manual_retry_grants=0,
        )
        session.add_all(
            [idle_source_ticket, running_source_ticket, no_target_source_ticket]
        )
        await session.flush()

        replacement = await issue_rollout_ticket(
            session,
            validator_hotkey="validator-a",
            now=now,
            ttl=timedelta(minutes=90),
        )
        assert replacement is not None
        assert replacement.agent_id == agent_ids[0]
        assert replacement.bench_version == rollout.desired_version
        assert idle_source_ticket.status == TicketStatus.EXPIRED

        expired_deadline_id = uuid4()
        session.add(
            Agent(
                agent_id=expired_deadline_id,
                miner_hotkey="expired-deadline-miner",
                name="expired-deadline-v2-work",
                sha256="bc" * 32,
                status=AgentStatus.EVALUATING,
                screening_policy_version=9,
                created_at=now + timedelta(minutes=2),
            )
        )
        stale_issued_ticket = ValidatorTicket(
            agent_id=expired_deadline_id,
            bench_version=2,
            validator_hotkey="validator-d",
            status=TicketStatus.ISSUED,
            issued_at=now - timedelta(minutes=90),
            deadline=now - timedelta(seconds=1),
            attempt_count=1,
            manual_retry_grants=0,
        )
        session.add(stale_issued_ticket)
        capabilities, stack = _capabilities(now)
        session.add(
            ValidatorHeartbeat(
                validator_hotkey="validator-d",
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
        after_stale_deadline = await issue_rollout_ticket(
            session,
            validator_hotkey="validator-d",
            now=now,
            ttl=timedelta(minutes=90),
        )
        assert after_stale_deadline is not None
        assert after_stale_deadline.bench_version == rollout.desired_version
        assert stale_issued_ticket.status == TicketStatus.EXPIRED

        assert (
            await issue_rollout_ticket(
                session,
                validator_hotkey="validator-b",
                now=now,
                ttl=timedelta(minutes=90),
                validator_running_benchmark=True,
            )
            is None
        )
        assert running_source_ticket.status == TicketStatus.ISSUED

        for agent_id in agent_ids:
            agent = await session.get(Agent, agent_id)
            assert agent is not None
            agent.screened_image_sha256 = None
            agent.screened_image_size_bytes = None
            agent.screened_image_id = None
            agent.screened_image_ref = None
            agent.screened_image_upload_id = None
            agent.screened_image_verified_at = None
        await session.flush()
        assert (
            await issue_rollout_ticket(
                session,
                validator_hotkey="validator-c",
                now=now,
                ttl=timedelta(minutes=90),
            )
            is None
        )
        assert no_target_source_ticket.status == TicketStatus.ISSUED
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
                        bench_version=CANARY_BENCH_VERSION,
                        validator_hotkey=f"v3-{validator}",
                        run_id=f"v3-drop-{validator}",
                        signature="bb",
                        seed=1,
                        composite=0.1,
                        tool_mean=0.1,
                        memory_mean=0.1,
                        median_ms=1,
                        n=114,
                        details={"bench_version": CANARY_BENCH_VERSION},
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


async def test_legacy_scored_top_five_recovers_seed_and_converges_idempotently() -> (
    None
):
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
                    miner_hotkey="miner-legacy",
                    name="legacy-riser",
                    sha256="f" * 64,
                    status=AgentStatus.SCORED,
                    screening_policy_version=8,
                    created_at=now + timedelta(minutes=1),
                )
            )
            for validator in range(3):
                session.add(
                    Score(
                        agent_id=rising_id,
                        bench_version=2,
                        validator_hotkey=f"legacy-riser-{validator}",
                        run_id=f"legacy-riser-{validator}",
                        signature="aa",
                        seed=8675309,
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
                        bench_version=CANARY_BENCH_VERSION,
                        validator_hotkey=f"drop-{validator}",
                        run_id=f"drop-{validator}",
                        signature="bb",
                        seed=1,
                        composite=0.1,
                        tool_mean=0.1,
                        memory_mean=0.1,
                        median_ms=1,
                        n=114,
                        details={"bench_version": CANARY_BENCH_VERSION},
                        generated_at=now,
                    )
                )

        async def generate(seed: int, *, bench_version: int) -> str:
            assert not session.in_transaction()
            assert seed == 8675309
            assert bench_version == CANARY_BENCH_VERSION
            return "e" * 64

        generator = AsyncMock()
        generator.run_size = "full"
        generator.generate.side_effect = generate
        assert (
            await refresh_rolling_qualification(
                session, generator=generator, now=now + timedelta(seconds=1)
            )
            == 1
        )
        assert (
            await refresh_rolling_qualification(
                session, generator=generator, now=now + timedelta(seconds=2)
            )
            == 0
        )
        assert generator.generate.await_count == 1
        async with session.begin():
            member = await session.get(
                BenchmarkRolloutMember, (rollout.rollout_id, rising_id)
            )
            dataset = await session.get(
                BenchmarkDataset, (rising_id, CANARY_BENCH_VERSION)
            )
            legacy = await session.get(Agent, rising_id)
            assert member is not None
            assert dataset is not None
            assert dataset.seed == 8675309
            assert dataset.sha256 == "e" * 64
            assert dataset.run_size == "full"
            assert dataset.seed_block is None
            assert legacy is not None
            assert legacy.dataset_seed is None
            assert legacy.dataset_sha256 is None
            claimed = await claim_screening_attempts(
                session,
                screener_hotkey="screener-legacy",
                now=now + timedelta(seconds=3),
                ttl=timedelta(minutes=70),
                limit=20,
            )
            assert rising_id in {agent.agent_id for agent, _attempt, _dup in claimed}
            assert legacy.status == AgentStatus.SCORED
            rising_attempt = next(
                attempt
                for agent, attempt, _dup in claimed
                if agent.agent_id == rising_id
            )
            rising_attempt.status = "passed"
            rising_attempt.finished_at = now + timedelta(seconds=4)
            legacy.screening_policy_version = 9
            legacy.screened_image_sha256 = "1" * 64
            legacy.screened_image_size_bytes = 1024
            legacy.screened_image_id = "sha256:" + "2" * 64
            legacy.screened_image_ref = f"ditto-screen/{rising_id}:latest"
            legacy.screened_image_upload_id = uuid4()
            legacy.screened_image_verified_at = now + timedelta(seconds=4)
            for initial_id in initial_ids[1:]:
                for validator in range(3):
                    session.add(
                        Score(
                            agent_id=initial_id,
                            bench_version=CANARY_BENCH_VERSION,
                            validator_hotkey=f"filled-{initial_id}-{validator}",
                            run_id=f"filled-{initial_id}-{validator}",
                            signature="cc",
                            seed=1,
                            composite=0.1,
                            tool_mean=0.1,
                            memory_mean=0.1,
                            median_ms=1,
                            n=114,
                            details={"bench_version": CANARY_BENCH_VERSION},
                            generated_at=now,
                        )
                    )
            await session.flush()
            ticket = await issue_rollout_ticket(
                session,
                validator_hotkey="validator-a",
                now=now + timedelta(seconds=5),
                ttl=timedelta(minutes=90),
            )
            assert ticket is not None
            assert ticket.agent_id == rising_id
            assert ticket.bench_version == CANARY_BENCH_VERSION
    await engine.dispose()


async def test_admin_qualifies_scored_top_five_with_compare_and_swap_guards() -> None:
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
                    miner_hotkey="miner-admin-riser",
                    name="admin-riser",
                    sha256="f" * 64,
                    status=AgentStatus.SCORED,
                    screening_policy_version=8,
                    created_at=now + timedelta(minutes=1),
                )
            )
            for validator, seed in enumerate((43, 41, 42)):
                session.add(
                    Score(
                        agent_id=rising_id,
                        bench_version=2,
                        validator_hotkey=f"admin-riser-{validator}",
                        run_id=f"admin-riser-{validator}",
                        signature="aa",
                        seed=seed,
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
                        bench_version=CANARY_BENCH_VERSION,
                        validator_hotkey=f"admin-drop-{validator}",
                        run_id=f"admin-drop-{validator}",
                        signature="bb",
                        seed=1,
                        composite=0.1,
                        tool_mean=0.1,
                        memory_mean=0.1,
                        median_ms=1,
                        n=114,
                        details={"bench_version": CANARY_BENCH_VERSION},
                        generated_at=now,
                    )
                )

        rollout_id = rollout.rollout_id
        generator = AsyncMock()
        generator.run_size = "full"
        generator.generate.return_value = "e" * 64
        detail = await inspect_benchmark_qualification(
            rising_id, None, session, generator
        )
        assert detail.qualification_allowed
        assert detail.currently_top_five
        assert not detail.rollout_member
        assert detail.total_score_count == 3
        assert detail.source_score_count == 3
        assert detail.target_score_count == 0

        await session.rollback()
        async with session.begin():
            issued = ValidatorTicket(
                agent_id=rising_id,
                bench_version=2,
                validator_hotkey="validator-issued-before-heartbeat",
                status=TicketStatus.ISSUED,
                issued_at=now,
                deadline=now + timedelta(minutes=30),
                attempt_count=1,
                manual_retry_grants=0,
            )
            session.add(issued)
        blocked = await inspect_benchmark_qualification(
            rising_id, None, session, generator
        )
        assert blocked.validator_run_active
        assert blocked.blocking_reason == "validator benchmark is active"
        await session.rollback()
        async with session.begin():
            locked_issued = await session.get(
                ValidatorTicket,
                (rising_id, 2, "validator-issued-before-heartbeat"),
            )
            assert locked_issued is not None
            locked_issued.deadline = now - timedelta(seconds=1)

        with pytest.raises(HTTPException, match="score count changed"):
            await qualify_benchmark_rollout(
                rising_id,
                AdminBenchmarkQualificationRequest(
                    reason="recover legacy top-five qualification",
                    expected_sha256="f" * 64,
                    expected_rollout_id=rollout_id,
                    expected_total_score_count=4,
                    expected_source_score_count=3,
                    expected_target_score_count=0,
                ),
                None,
                session,
                generator,
                "backroom:test",
            )

        response = await qualify_benchmark_rollout(
            rising_id,
            AdminBenchmarkQualificationRequest(
                reason="recover legacy top-five qualification",
                expected_sha256="f" * 64,
                expected_rollout_id=rollout_id,
                expected_total_score_count=3,
                expected_source_score_count=3,
                expected_target_score_count=0,
            ),
            None,
            session,
            generator,
            "backroom:test",
        )
        assert response.agent_status == AgentStatus.SCORED
        assert response.rollout_member
        assert response.screening_queued
        assert response.target_dataset_sha256 == "e" * 64
        async with session.begin():
            scores = list(
                await session.scalars(select(Score).where(Score.agent_id == rising_id))
            )
            agent = await session.get(Agent, rising_id)
            dataset = await session.get(
                BenchmarkDataset, (rising_id, CANARY_BENCH_VERSION)
            )
            audit = await session.scalar(
                select(BenchmarkRolloutAudit).where(
                    BenchmarkRolloutAudit.rollout_id == rollout_id,
                    BenchmarkRolloutAudit.event == "member_qualified",
                )
            )
            assert len(scores) == 3
            assert agent is not None and agent.status == AgentStatus.SCORED
            assert dataset is not None and dataset.seed == 41
            assert audit is not None
            assert audit.payload["origin"] == "manual"
            assert audit.payload["actor"] == "backroom:test"
            assert audit.payload["reason"] == ("recover legacy top-five qualification")
            assert audit.payload["seed_source"] == "source_scores_canonical_min"
            assert audit.payload["dataset_seed"] == 41
            assert audit.payload["dataset_sha256"] == "e" * 64
    await engine.dispose()


async def test_multiple_legacy_score_seeds_use_canonical_minimum() -> None:
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
                    miner_hotkey="miner-ambiguous",
                    name="ambiguous-riser",
                    sha256="f" * 64,
                    status=AgentStatus.SCORED,
                    screening_policy_version=8,
                    created_at=now + timedelta(minutes=1),
                )
            )
            for validator, seed in enumerate((41, 42, 42)):
                session.add(
                    Score(
                        agent_id=rising_id,
                        bench_version=2,
                        validator_hotkey=f"ambiguous-{validator}",
                        run_id=f"ambiguous-{validator}",
                        signature="aa",
                        seed=seed,
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
                        bench_version=CANARY_BENCH_VERSION,
                        validator_hotkey=f"ambiguous-drop-{validator}",
                        run_id=f"ambiguous-drop-{validator}",
                        signature="bb",
                        seed=1,
                        composite=0.1,
                        tool_mean=0.1,
                        memory_mean=0.1,
                        median_ms=1,
                        n=114,
                        details={"bench_version": CANARY_BENCH_VERSION},
                        generated_at=now,
                    )
                )

        generator = AsyncMock()
        generator.run_size = "full"
        generator.generate.return_value = "e" * 64
        assert (
            await refresh_rolling_qualification(
                session, generator=generator, now=now + timedelta(seconds=1)
            )
            == 1
        )
        generator.generate.assert_awaited_once_with(
            41, bench_version=CANARY_BENCH_VERSION
        )
        blockers = await rolling_qualification_blockers(
            session, generator_run_size="full"
        )
        assert blockers == []
        member = await session.get(
            BenchmarkRolloutMember, (rollout.rollout_id, rising_id)
        )
        dataset = await session.get(BenchmarkDataset, (rising_id, CANARY_BENCH_VERSION))
        audit = await session.scalar(
            select(BenchmarkRolloutAudit).where(
                BenchmarkRolloutAudit.rollout_id == rollout.rollout_id,
                BenchmarkRolloutAudit.event == "member_qualified",
                BenchmarkRolloutAudit.payload["agent_id"].as_string() == str(rising_id),
            )
        )
        assert member is not None
        assert dataset is not None and dataset.seed == 41
        assert audit is not None
        assert audit.payload["origin"] == "automatic"
        assert audit.payload["seed_source"] == "source_scores_canonical_min"
        assert audit.payload["dataset_seed"] == 41
        assert audit.payload["dataset_sha256"] == "e" * 64
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
                    desired_version=CANARY_BENCH_VERSION,
                    status="collecting",
                    cohort_size=5,
                ),
                BenchmarkRollout(
                    rollout_id=uuid4(),
                    from_version=2,
                    desired_version=CANARY_BENCH_VERSION,
                    status="blocked_ineligible",
                    cohort_size=5,
                ),
            ]
        )
        with pytest.raises(IntegrityError):
            await session.flush()
        await session.rollback()
    await engine.dispose()


async def test_capable_validator_cannot_automatically_seed_rollout_work() -> None:
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
    assert not await ensure_rolling_qualification(session, generator=generator, now=now)
    generator.generate.assert_not_awaited()
    async with session.begin():
        rollout = await open_rollout(session)
        assert rollout is None
        ticket = await issue_rollout_ticket(
            session,
            validator_hotkey="validator-auto",
            now=now,
            ttl=timedelta(minutes=90),
        )
        assert ticket is None

    # Repeated job polls remain fail-closed and cannot render or open a rollout.
    assert not await ensure_rolling_qualification(session, generator=generator, now=now)
    generator.generate.assert_not_awaited()

    state = await start_rollout(
        None,
        session,
        generator,
        str(CANARY_BENCH_VERSION),
        AdminRolloutStartRequest(
            reason="operator opens shipped benchmark",
            actor="backroom:test",
            confirmation=f"START BENCHMARK V{CANARY_BENCH_VERSION}",
            expected_active_version=2,
        ),
    )
    assert state["status"] == "collecting"
    assert state["active_version"] == 2
    assert state["desired_version"] == CANARY_BENCH_VERSION
    assert generator.generate.await_count == 5
    await session.rollback()
    async with session.begin():
        rollout = await open_rollout(session)
        assert rollout is not None
        audit = await session.scalar(
            select(BenchmarkRolloutAudit).where(
                BenchmarkRolloutAudit.rollout_id == rollout.rollout_id,
                BenchmarkRolloutAudit.event == "cohort_frozen",
            )
        )
        assert audit is not None
        assert audit.payload["actor"] == "backroom:test"
        assert audit.payload["reason"] == "operator opens shipped benchmark"
        assert set(audit.payload["seed_sources"].values()) == {"legacy_pin"}
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
                desired_version=CANARY_BENCH_VERSION,
                status="activated",
                cohort_size=5,
                created_at=now,
                activated_at=now,
            )
        )
    async with maker() as session:
        with pytest.raises(HTTPException, match="type .* exactly"):
            await start_rollout(
                None,
                session,
                object(),  # type: ignore[arg-type]
                str(CANARY_BENCH_VERSION),
                AdminRolloutStartRequest(
                    reason="confirmation guard",
                    actor="test",
                    confirmation="START BENCHMARK V3",
                    expected_active_version=CANARY_BENCH_VERSION,
                ),
            )
        state = await start_rollout(
            None,
            session,
            object(),  # type: ignore[arg-type]
            str(CANARY_BENCH_VERSION),
            AdminRolloutStartRequest(
                reason="idempotence check",
                actor="test",
                confirmation=f"START BENCHMARK V{CANARY_BENCH_VERSION}",
                expected_active_version=CANARY_BENCH_VERSION,
            ),
        )
        assert state["active_version"] == CANARY_BENCH_VERSION
        assert state["desired_version"] == CANARY_BENCH_VERSION
        assert state["status"] == "activated"
        count = await session.scalar(select(func.count(BenchmarkRollout.rollout_id)))
        assert count == 1
        session.add(
            BenchmarkRollout(
                rollout_id=uuid4(),
                from_version=2,
                desired_version=CANARY_BENCH_VERSION,
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
async def test_rollout_start_requires_one_capable_validator_and_matches_telemetry(
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
        if capable_count < 1:
            with pytest.raises(HTTPException) as exc_info:
                await _require_rollout_start_capacity(
                    session, now=now, desired_version=CANARY_BENCH_VERSION
                )
            assert exc_info.value.status_code == 409
            assert "requires at least 1" in str(exc_info.value.detail)
            assert await open_rollout(session) is None
        else:
            guarded = await _require_rollout_start_capacity(
                session, now=now, desired_version=CANARY_BENCH_VERSION
            )
            assert guarded["v3_capable_validator_count"] == capable_count
    await engine.dispose()


def _heartbeat(
    hotkey: str, now: datetime, *, versions: list[int], protocol_version: int = 8
) -> ValidatorHeartbeat:
    capabilities, stack = _capabilities(now)
    capabilities["scorer_benchmarks"]["supported_bench_versions"] = versions
    return ValidatorHeartbeat(
        validator_hotkey=hotkey,
        software_version="1.0.0",
        protocol_version=protocol_version,
        code_digest="d" * 64,
        state="polling",
        first_seen_at=now,
        reported_at=now,
        seen_at=now,
        signature="ab" * 64,
        capabilities=capabilities,
        stack=stack,
    )


async def test_capability_gate_is_parameterised_per_bench_version() -> None:
    """A v4-capable heartbeat gates v4 in and v3 out, and vice versa."""
    now = datetime.now(UTC).replace(microsecond=0)
    v4_only = _heartbeat("v4-only", now, versions=[2, 4])
    v3_only = _heartbeat("v3-only", now, versions=[2, 3])

    assert heartbeat_supports_version(v4_only, now=now, version=4)
    assert not heartbeat_supports_version(v4_only, now=now, version=3)
    assert heartbeat_supports_version(v3_only, now=now, version=3)
    assert not heartbeat_supports_version(v3_only, now=now, version=4)
    # Both are gated by the same fixed protocol-8 wire floor.
    stale = _heartbeat("old", now, versions=[2, 4], protocol_version=7)
    assert not heartbeat_supports_version(stale, now=now, version=4)


async def test_second_rollout_while_one_is_open_raises_conflict_not_integrity() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    async with maker() as session, session.begin():
        members, pins = await _seed_members(session, now)
        await create_rollout_snapshot(
            session,
            members=members,
            datasets=pins,
            now=now,
            from_version=2,
            desired_version=3,
        )
        with pytest.raises(RolloutConflictError) as exc_info:
            await create_rollout_snapshot(
                session,
                members=members,
                datasets=pins,
                now=now,
                from_version=2,
                desired_version=4,
            )
        assert "only one benchmark rollout may be open" in str(exc_info.value)
    await engine.dispose()


async def _seed_members(
    session, now: datetime
) -> tuple[list[RolloutSnapshotMember], dict[UUID, DatasetPin]]:
    members: list[RolloutSnapshotMember] = []
    pins: dict[UUID, DatasetPin] = {}
    for position in range(1, 6):
        agent_id = uuid4()
        session.add(
            Agent(
                agent_id=agent_id,
                miner_hotkey=f"miner-{position}",
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
                miner_hotkey=f"miner-{position}",
                composite=1 - position / 100,
            )
        )
        pins[agent_id] = DatasetPin(seed=position, sha256="c" * 64, run_size="full")
    await session.flush()
    return members, pins


async def test_supersede_frees_the_open_slot_for_the_next_rollout() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    async with maker() as session, session.begin():
        members, pins = await _seed_members(session, now)
        stale = await create_rollout_snapshot(
            session,
            members=members,
            datasets=pins,
            now=now,
            from_version=2,
            desired_version=3,
        )
        assert await open_rollout(session) is not None

        superseded = await supersede_open_rollout(
            session, actor="nick", reason="v3 gate had false positives", now=now
        )
        assert superseded is not None
        assert superseded.rollout_id == stale.rollout_id
        assert superseded.status == "superseded"
        # The partial unique index excludes 'superseded', so the slot is free.
        assert await open_rollout(session) is None

        audit = (
            await session.scalars(
                select(BenchmarkRolloutAudit).where(
                    BenchmarkRolloutAudit.event == "superseded"
                )
            )
        ).all()
        assert len(audit) == 1
        assert audit[0].payload["actor"] == "nick"
        assert audit[0].payload["reason"] == "v3 gate had false positives"
        assert audit[0].payload["previous_status"] == "collecting"
        assert audit[0].payload["desired_version"] == 3

        # 2 -> 4 now inserts cleanly rather than tripping the unique index.
        fresh = await create_rollout_snapshot(
            session,
            members=members,
            datasets=pins,
            now=now + timedelta(seconds=1),
            from_version=2,
            desired_version=4,
        )
        assert fresh.desired_version == 4
        assert fresh.status == "collecting"
        await session.flush()
    await engine.dispose()


async def test_superseded_rollout_issues_no_tickets_and_never_activates() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    async with maker() as session, session.begin():
        _agent_ids, rollout = await _seed_rollout(session, now)
        assert await supersede_open_rollout(
            session, actor="nick", reason="abandoned", now=now
        )
        assert (
            await issue_rollout_ticket(
                session,
                validator_hotkey="validator-a",
                now=now,
                ttl=timedelta(minutes=90),
            )
            is None
        )
        assert not await maybe_activate_rollout(session, rollout, now=now)
        assert rollout.status == "superseded"
        assert await active_bench_version(session) == 2
    await engine.dispose()


async def test_supersede_refuses_rollout_after_priority_cohort_owns_authority() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    async with maker() as session, session.begin():
        _agent_ids, rollout = await _seed_desired_quorum_cohort(session, now)
        assert await active_bench_version(session) == CANARY_BENCH_VERSION

        with pytest.raises(RolloutConflictError, match="already owns active authority"):
            await supersede_open_rollout(
                session,
                actor="operator",
                reason="must not roll authority backward",
                now=now,
            )

        assert rollout.status == "collecting"
    await engine.dispose()


async def test_operator_can_select_fully_qualified_superseded_authority() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    async with maker() as session, session.begin():
        _agent_ids, rollout = await _seed_desired_quorum_cohort(session, now)
        rollout.status = "superseded"
        await session.flush()
        assert await active_bench_version(session) == 2

        selected = await select_active_bench_version(
            session,
            bench_version=CANARY_BENCH_VERSION,
            actor="operator",
            reason="restore the completed contract",
            now=now + timedelta(minutes=1),
        )

        assert selected.rollout_id == rollout.rollout_id
        assert selected.status == "superseded"
        assert await active_bench_version(session) == CANARY_BENCH_VERSION
        audit = await session.scalar(
            select(BenchmarkRolloutAudit).where(
                BenchmarkRolloutAudit.rollout_id == rollout.rollout_id,
                BenchmarkRolloutAudit.event == "authority_selected",
            )
        )
        assert audit is not None
        assert audit.payload["previous_active_version"] == 2
        assert audit.payload["bench_version"] == CANARY_BENCH_VERSION
    await engine.dispose()


async def test_activated_rollout_cannot_be_superseded() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    async with maker() as session, session.begin():
        session.add(
            BenchmarkRollout(
                rollout_id=uuid4(),
                from_version=2,
                desired_version=3,
                status="activated",
                cohort_size=5,
                created_at=now,
                activated_at=now,
            )
        )
        await session.flush()
        with pytest.raises(RolloutConflictError) as exc_info:
            await supersede_open_rollout(session, actor="nick", reason="oops", now=now)
        assert "activated" in str(exc_info.value)
    await engine.dispose()


async def test_admin_supersede_endpoint_audits_and_refuses_activated() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    async with maker() as session:
        async with session.begin():
            members, pins = await _seed_members(session, now)
            await create_rollout_snapshot(
                session,
                members=members,
                datasets=pins,
                now=now,
                from_version=2,
                desired_version=3,
            )
        # The path accepts the legacy "v3" spelling as well as a bare "3".
        state = await supersede_rollout(
            None,
            session,
            "v3",
            AdminRolloutSupersedeRequest(
                reason="false positives",
                actor="nick",
                confirmation="SUPERSEDE BENCHMARK V3",
            ),
        )
        assert state["status"] == "superseded"
        assert await open_rollout(session) is None

        # A second call has nothing open left to supersede.
        with pytest.raises(HTTPException) as exc_info:
            await supersede_rollout(
                None,
                session,
                "v3",
                AdminRolloutSupersedeRequest(
                    reason="again",
                    actor="nick",
                    confirmation="SUPERSEDE BENCHMARK V3",
                ),
            )
        assert exc_info.value.status_code == 409
    await engine.dispose()


async def test_admin_start_route_is_parameterised_by_version() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    async with maker() as session:
        async with session.begin():
            for index in range(2):
                session.add(_heartbeat(f"validator-{index}", now, versions=[2, 4]))
            await session.flush()
        # Telemetry is counted against the REQUESTED version, not a constant.
        assert (await get_rollout(None, session, "4"))[
            "canary_capable_validator_count"
        ] == 2
        assert (await get_rollout(None, session, "3"))[
            "canary_capable_validator_count"
        ] == 0

        # An unshipped version fails closed rather than opening a bad rollout.
        with pytest.raises(HTTPException) as exc_info:
            await get_rollout(None, session, "9")
        assert exc_info.value.status_code == 409
        with pytest.raises(HTTPException) as not_found:
            await get_rollout(None, session, "banana")
        assert not_found.value.status_code == 404
    await engine.dispose()


async def _seed_desired_quorum_cohort(
    session,
    now: datetime,
    *,
    smoke_indices: tuple[int, ...] = (),
    held_indices: tuple[int, ...] = (),
) -> tuple[list[UUID], BenchmarkRollout]:
    """The five-agent cohort, each member carrying a full raw desired quorum.

    ``smoke_indices`` gives those members a 3/3 quorum of sub-floor (smoke
    profile) runs — a quorum by row count that can never rank. ``held_indices``
    moves members out of the eligible pool.
    """
    agent_ids, rollout = await _seed_rollout(session, now)
    for position, agent_id in enumerate(agent_ids, start=1):
        smoke = (position - 1) in smoke_indices
        for validator in range(3):
            session.add(
                Score(
                    agent_id=agent_id,
                    bench_version=CANARY_BENCH_VERSION,
                    validator_hotkey=f"validator-{validator}",
                    run_id=f"v4-{position}-{validator}",
                    signature="bb",
                    seed=position,
                    composite=0.7 + position / 100,
                    tool_mean=0.7,
                    memory_mean=0.7,
                    median_ms=1,
                    n=50 if smoke else 114,
                    details={"bench_version": CANARY_BENCH_VERSION},
                    generated_at=now,
                )
            )
    for index in held_indices:
        agent = await session.get(Agent, agent_ids[index])
        assert agent is not None
        agent.status = AgentStatus.ATH_PENDING_REVIEW
    await session.flush()
    return agent_ids, rollout


async def test_activation_requires_five_ranked_desired_quorum_agents() -> None:
    # Activation is the last point the full-emission-set guarantee can be
    # enforced: afterwards open_rollout() is None, so list_eligible_ledger reads
    # the desired version unconditionally and its own threshold no longer
    # applies. rolling_top_five happens to refuse both degraded cohorts below on
    # its own today (COHORT_SIZE == MIN_DESIRED_AUTHORITY_AGENTS), so these
    # assert the guarantee, not which gate fired; the isolating case is
    # test_activation_refused_when_only_ranked_quorum_count_is_short.
    for smoke_indices, held_indices in (((0,), ()), ((), (0,))):
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        maker = async_sessionmaker(engine, expire_on_commit=False)
        now = datetime.now(UTC).replace(microsecond=0)
        async with maker() as session, session.begin():
            _agent_ids, rollout = await _seed_desired_quorum_cohort(
                session,
                now,
                smoke_indices=smoke_indices,
                held_indices=held_indices,
            )
            # Raw row counts look like a complete cohort quorum.
            raw_counts = (
                await session.execute(
                    select(Score.agent_id, func.count(Score.validator_hotkey))
                    .where(Score.bench_version == CANARY_BENCH_VERSION)
                    .group_by(Score.agent_id)
                )
            ).all()
            assert [count for _agent_id, count in raw_counts] == [3] * 5
            # Ranked, eligible quorums are what actually matter, and are short.
            assert (
                await count_ranked_quorum_agents(
                    session, bench_version=CANARY_BENCH_VERSION
                )
                == MIN_DESIRED_AUTHORITY_AGENTS - 1
            )
            assert await maybe_activate_rollout(session, rollout, now=now) is False
            assert rollout.status == "collecting"
            assert await active_bench_version(session) == 2
        await engine.dispose()


async def test_same_coldkey_generations_fill_one_rollout_position() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    async with maker() as session, session.begin():
        agent_ids, _rollout = await _seed_desired_quorum_cohort(session, now)
        for index, agent_id in enumerate(agent_ids):
            agent = await session.get(Agent, agent_id)
            assert agent is not None
            session.add(
                EvaluationPayment(
                    block_hash=f"0x{agent_id.hex}",
                    extrinsic_index=0,
                    agent_id=agent_id,
                    miner_hotkey=agent.miner_hotkey,
                    miner_coldkey=(
                        "5SharedColdkey" if index < 2 else f"5Coldkey{index:048d}"
                    ),
                    amount_rao=1,
                    dest_address="5Destination",
                    timestamp=now,
                )
            )
        await session.flush()

        top = await rolling_top_five(session)

        assert len(top) == MIN_DESIRED_AUTHORITY_AGENTS - 1
        assert (
            await count_ranked_quorum_agents(
                session, bench_version=CANARY_BENCH_VERSION
            )
            == MIN_DESIRED_AUTHORITY_AGENTS - 1
        )
    await engine.dispose()


async def test_activation_refused_when_only_ranked_quorum_count_is_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Isolates the new precondition: every legacy activation check is satisfied
    # (a converged five-member top five, a full raw quorum for every eligible
    # member) and only the ranked-quorum count is short. Without the
    # precondition this cohort would activate into a four-agent pool and the
    # KOTH tail would go short.
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    async with maker() as session, session.begin():
        agent_ids, rollout = await _seed_desired_quorum_cohort(
            session, now, smoke_indices=(0,)
        )
        converged = [
            RolloutSnapshotMember(
                agent_id=agent_id, miner_hotkey=f"miner-{index + 1}", composite=0.9
            )
            for index, agent_id in enumerate(agent_ids)
        ]

        async def _converged_top_five(_session: object) -> list[RolloutSnapshotMember]:
            return converged

        monkeypatch.setattr(
            "ditto.db.queries.benchmark_rollout.rolling_top_five",
            _converged_top_five,
        )
        assert len(await rolling_top_five(session)) == MIN_DESIRED_AUTHORITY_AGENTS - 1
        assert await maybe_activate_rollout(session, rollout, now=now) is False
        assert rollout.status == "collecting"
    await engine.dispose()


async def test_activation_at_five_ranked_quorums_keeps_a_full_emission_set() -> None:
    # The invariant that actually matters: it holds ACROSS the activation
    # boundary. Before, the ledger is pinned to v2 with five entries; after, it
    # is wholly on v4 and still has five.
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC).replace(microsecond=0)
    async with maker() as session, session.begin():
        agent_ids, rollout = await _seed_desired_quorum_cohort(session, now)
        assert (
            await count_ranked_quorum_agents(
                session, bench_version=CANARY_BENCH_VERSION
            )
            == MIN_DESIRED_AUTHORITY_AGENTS
        )
        assert await maybe_activate_rollout(session, rollout, now=now) is True
        assert rollout.status == "activated"
        assert await active_bench_version(session) == CANARY_BENCH_VERSION
        assert await open_rollout(session) is None

        ledger = await list_eligible_ledger(session)
        assert len(ledger) == MIN_DESIRED_AUTHORITY_AGENTS
        assert {row.agent_id for row in ledger} == set(agent_ids)
        assert {row.bench_version for row in ledger} == {CANARY_BENCH_VERSION}
        assert all(row.eligible for row in ledger)
    await engine.dispose()


async def test_rollout_state_active_version_matches_start_guard_authority() -> None:
    """rollout_state's active_version must equal active_bench_version.

    Regression for the spurious "active benchmark changed: expected v5, found v4"
    409 on rollout start. The start guard compares the operator-supplied
    expected_active_version against active_bench_version(), while the operator UI
    reads it from rollout_state()["active_version"]. When those two derive the
    active version differently they disagree and start_rollout 409s even though
    nothing changed.

    The divergent state: an activated older transition plus a newer, terminally
    superseded transition that never activated (a real sequence -- a v5->v6 rollout
    opened while a converging v4->v5 briefly read as active, then v4->v5 was
    reverted, leaving v5->v6 dangling as the most-recent row). The most-recent row
    (superseded, from=5) and the latest activated row (desired=4) disagree; both
    reports must nonetheless agree with each other.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        base = datetime(2026, 7, 1, tzinfo=UTC)
        # Older transition, activated: this is what the weight-setting guard treats
        # as authoritative (latest activated desired_version == 4).
        session.add(
            BenchmarkRollout(
                rollout_id=uuid4(),
                from_version=3,
                desired_version=4,
                status="activated",
                cohort_size=5,
                created_at=base,
                activated_at=base + timedelta(hours=1),
            )
        )
        # Newer transition, terminally superseded (never activated). Its from_version
        # is 5, so the pre-fix most-recent-row derivation reported active_version == 5.
        session.add(
            BenchmarkRollout(
                rollout_id=uuid4(),
                from_version=5,
                desired_version=6,
                status="superseded",
                cohort_size=5,
                created_at=base + timedelta(hours=2),
            )
        )
        await session.flush()

        guard = await active_bench_version(session)
        state = await rollout_state(session)
        assert guard == 4
        # The invariant the fix guarantees: the value the UI echoes back as
        # expected_active_version is exactly what the start guard checks.
        assert state["active_version"] == guard
    await engine.dispose()
