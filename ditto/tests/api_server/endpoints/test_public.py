"""Unit tests for :mod:`ditto.api_server.endpoints.public`.

``GET /api/v1/public/leaderboard`` is open (no validator auth) and aggregate-only:
it must rank miners by composite, expose tool/memory means, and NEVER leak the
integrity-internal fields (``signature``, ``sha256``, ``validator_hotkey``) or
per-case detail.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.public import PublicSystemMetrics
from ditto.api_models.screener import (
    SCREENING_POLICY_VERSION,
    SourceReviewEvidenceItem,
    SourceReviewFinding,
)
from ditto.api_models.stack_health import (
    ComponentHealthState,
    ValidatorComponentHealth,
    ValidatorStackHealth,
)
from ditto.api_models.ticket_status import TicketPurpose, TicketStatus
from ditto.api_server.bench import CURRENT_BENCH_VERSION
from ditto.api_server.datapipeline import DataPipelineError
from ditto.api_server.dependencies import (
    get_dataset_generator,
    get_session,
    get_storage_client,
)
from ditto.api_server.endpoints import public as public_endpoint
from ditto.api_server.endpoints.public import _fleet_classification
from ditto.api_server.storage import ObjectDownloadFailedError
from ditto.api_server.validator_names import ValidatorNamesSnapshot
from ditto.chain import ChainError
from ditto.chain.models import (
    ChainWeight,
    ChainWeightsSnapshot,
    ChainWeightVector,
)
from ditto.db.models import (
    Agent,
    AthReview,
    Base,
    BenchmarkDataset,
    BenchmarkRollout,
    Score,
    ScreeningAttempt,
    ScreeningQuarantine,
    ValidatorHeartbeat,
    ValidatorTicket,
)
from ditto.db.queries.audit import (
    EVENT_SCORE,
    GENESIS_HASH,
    append_audit_entry,
)
from ditto.db.queries.benchmark_rollout import DEFAULT_BENCH_VERSION
from ditto.db.queries.scores import upsert_score

_MINER_A = "5DhaT8U7LVwnnJNUU8VL1XEipicatoaDVVq7cHo227gogVZm"
_MINER_B = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
_VALIDATOR_C = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


def test_v5_token_telemetry_public_parser_is_typed_and_fail_closed() -> None:
    details = {
        "token_usage": {
            "accounting_version": 2,
            "status": "complete",
            "source": "model_proxy_provider_response",
            "provider": "openrouter",
            "profile_revision": "profile-v1",
            "model": "qwen/qwen3-32b",
            "prompt_tokens": 1800,
            "prompt_bytes": 7200,
            "completion_tokens": 200,
            "total_tokens": 2000,
            "requests": 10,
            "successes": 10,
            "usage_available": 10,
            "usage_unavailable": 0,
            "provider_latency_ms": 2500,
            "ttft_status": "unavailable_non_streaming",
        },
        "token_efficiency": {
            "formula_version": "v5-relay-token-waste-p90-v1",
            "baseline_id": "v5-baseline",
            "baseline_prompt_tokens": 900,
            "baseline_completion_tokens": 100,
            "baseline_total_tokens": 1000,
            "budget_percentile": 0.9,
            "observed_prompt_tokens": 1800,
            "observed_completion_tokens": 200,
            "observed_total_tokens": 2000,
            "excess_ratio": 1.0,
            "maximum_penalty": 0.1,
            "minimum_multiplier": 0.9,
            "multiplier": 0.95,
            "raw_composite": 0.9,
            "adjusted_composite": 0.855,
            "penalty_applied": True,
            "decision_reason": "above_budget",
        },
    }
    usage = public_endpoint._safe_token_usage(details)
    decision = public_endpoint._safe_token_efficiency(details)
    assert usage is not None and usage.total_tokens == 2000
    assert decision is not None and decision.adjusted_composite == 0.855
    assert decision.penalty_applied is True

    details["token_efficiency"]["multiplier"] = 1.001
    assert public_endpoint._safe_token_efficiency(details) is None


def test_composite_breakdown_separates_quality_gates_from_token_penalty() -> None:
    details = {
        "raw_composite": 0.372854,
        "token_efficiency": {
            "formula_version": "v5-relay-token-waste-p90-v1",
            "baseline_id": "v5-baseline",
            "baseline_prompt_tokens": 1_200_000,
            "baseline_completion_tokens": 291_793,
            "baseline_total_tokens": 1_491_793,
            "budget_percentile": 0.9,
            "observed_prompt_tokens": 1_500_000,
            "observed_completion_tokens": 364_699,
            "observed_total_tokens": 1_864_699,
            "excess_ratio": 0.25,
            "maximum_penalty": 0.1,
            "minimum_multiplier": 0.9,
            "multiplier": 0.9800018,
            "raw_composite": 0.372854,
            "adjusted_composite": 0.365398,
            "penalty_applied": True,
            "decision_reason": "above_budget",
        },
    }

    breakdown = public_endpoint._composite_breakdown(
        tool_mean=0.9278788,
        memory_mean=0.5729167,
        final_composite=0.365398,
        details=details,
    )

    assert breakdown is not None
    assert breakdown.base_accuracy == pytest.approx(0.75039775)
    assert breakdown.benchmark_quality_multiplier == pytest.approx(
        0.372854 / 0.75039775
    )
    assert breakdown.pre_token_composite == 0.372854
    assert breakdown.token_efficiency_multiplier == pytest.approx(0.9800018)
    assert breakdown.token_penalty == pytest.approx(0.0199982)
    assert breakdown.maximum_token_penalty == 0.1
    assert breakdown.final_composite == 0.365398


def test_composite_breakdown_shows_no_token_penalty_when_within_budget() -> None:
    details = {
        "raw_composite": 0.493952,
        "token_efficiency": {
            "formula_version": "v5-relay-token-waste-p90-v1",
            "baseline_id": "v5-baseline",
            "baseline_prompt_tokens": 1_200_000,
            "baseline_completion_tokens": 291_793,
            "baseline_total_tokens": 1_491_793,
            "budget_percentile": 0.9,
            "observed_prompt_tokens": 1_000_000,
            "observed_completion_tokens": 283_639,
            "observed_total_tokens": 1_283_639,
            "excess_ratio": 0.0,
            "maximum_penalty": 0.1,
            "minimum_multiplier": 0.9,
            "multiplier": 1.0,
            "raw_composite": 0.493952,
            "adjusted_composite": 0.493952,
            "penalty_applied": False,
            "decision_reason": "within_budget",
        },
    }

    breakdown = public_endpoint._composite_breakdown(
        tool_mean=0.8018181818,
        memory_mean=0.8333333333,
        final_composite=0.493952,
        details=details,
    )

    assert breakdown is not None
    assert breakdown.base_accuracy == pytest.approx(0.81757575755)
    assert breakdown.benchmark_quality_multiplier == pytest.approx(
        0.493952 / 0.81757575755
    )
    assert breakdown.token_efficiency_multiplier == 1.0
    assert breakdown.token_penalty == 0.0


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(eng.sync_engine, "connect")
    def _enable_fk(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest.fixture
def session_maker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


def _install_db(app: FastAPI, maker: async_sessionmaker[AsyncSession]) -> None:
    async def _session() -> AsyncIterator[AsyncSession]:
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = _session


async def _seed_scored(
    maker: async_sessionmaker[AsyncSession],
    *,
    miner: str,
    composite: float,
    tool_mean: float,
    memory_mean: float,
    status: AgentStatus = AgentStatus.SCORED,
    median_ms: int = 500,
    generated_at: datetime = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC),
    recorded_at: datetime | None = None,
    details: dict | None = None,
) -> None:
    async with maker() as s, s.begin():
        agent = Agent(
            agent_id=uuid4(),
            miner_hotkey=miner,
            name="agent",
            sha256="ab" * 32,
            size_bytes=524288,
            status=status,
            created_at=datetime.now(UTC),
        )
        s.add(agent)
        await s.flush()
        await upsert_score(
            s,
            agent_id=agent.agent_id,
            validator_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            run_id="run_1",
            seed=42,
            composite=composite,
            tool_mean=tool_mean,
            memory_mean=memory_mean,
            median_ms=median_ms,
            n=20,
            generated_at=generated_at,
            signature="ab" * 64,
            details=details,
        )
        if recorded_at is not None:
            score = await s.get(
                Score,
                (
                    agent.agent_id,
                    2,
                    "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                ),
            )
            assert score is not None
            score.created_at = recorded_at
            score.updated_at = recorded_at


async def _seed_k3(
    maker: async_sessionmaker[AsyncSession],
    *,
    miner: str,
    composites: list[float],
    status: AgentStatus = AgentStatus.SCORED,
    dataset_seed: int | None = 987654321,
    dataset_sha256: str | None = "cd" * 32,
    dataset_run_size: str | None = "full",
    dataset_seed_block: int | None = 4321,
    dataset_seed_block_hash: str | None = "0x" + "9f" * 32,
    details: dict | None = None,
    base_time: datetime = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC),
    created_at: datetime | None = None,
) -> str:
    """Seed one agent scored by ``len(composites)`` distinct validators.

    Returns the agent_id (hex str) so a test can hit the detail endpoint.
    """
    validators = [
        "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        "5DhaT8U7LVwnnJNUU8VL1XEipicatoaDVVq7cHo227gogVZm",
        "5CZq6MdanxF3j8ACp8oVtiaphTeyrA7QFPU92ke2jEFzK1mp",
    ]
    agent_id = uuid4()
    bench_version = int(details.get("bench_version", 2)) if details else 2
    async with maker() as s, s.begin():
        agent = Agent(
            agent_id=agent_id,
            miner_hotkey=miner,
            name="agent",
            sha256="ab" * 32,
            size_bytes=524288,
            status=status,
            dataset_seed=dataset_seed,
            dataset_sha256=dataset_sha256,
            dataset_run_size=dataset_run_size,
            dataset_seed_block=dataset_seed_block,
            dataset_seed_block_hash=dataset_seed_block_hash,
            created_at=created_at or datetime.now(UTC),
        )
        s.add(agent)
        await s.flush()
        for i, composite in enumerate(composites):
            await upsert_score(
                s,
                agent_id=agent_id,
                validator_hotkey=validators[i],
                run_id=f"run_{i}",
                seed=dataset_seed or 0,
                composite=composite,
                tool_mean=composite,
                memory_mean=composite,
                median_ms=500,
                n=110,
                generated_at=base_time + timedelta(minutes=i),
                signature="ab" * 64,
                details=details,
                bench_version=bench_version,
            )
    return str(agent_id)


async def _seed_top_five_floor(
    maker: async_sessionmaker[AsyncSession],
    *,
    fifth_place: float = 0.80,
    bench_version: int = DEFAULT_BENCH_VERSION,
) -> None:
    for rank, marker in enumerate(("A", "B", "C", "D", "E")):
        composite = fifth_place + (4 - rank) * 0.01
        await _seed_k3(
            maker,
            miner="5" + marker * 47,
            composites=[composite, composite, composite],
            details={"bench_version": bench_version},
        )


async def _seed_agent(
    maker: async_sessionmaker[AsyncSession],
    *,
    miner: str,
    status: AgentStatus = AgentStatus.UPLOADED,
    name: str = "agent",
    created_at: datetime | None = None,
    screening_reason: str | None = None,
    duplicate_of: UUID | None = None,
    review_reason: str | None = None,
    screening_policy_version: int = 0,
) -> str:
    """Seed a submission with no score (e.g. still uploaded/evaluating)."""
    agent_id = uuid4()
    async with maker() as s, s.begin():
        s.add(
            Agent(
                agent_id=agent_id,
                miner_hotkey=miner,
                name=name,
                sha256="cd" * 32,
                size_bytes=524288,
                status=status,
                created_at=created_at or datetime.now(UTC),
                screening_reason=screening_reason,
                duplicate_of=duplicate_of,
                review_reason=review_reason,
                screening_policy_version=screening_policy_version,
            )
        )
    return str(agent_id)


class TestPublicChainWeights:
    async def test_returns_native_revealed_weight_matrix(
        self, app: FastAPI, client: httpx.AsyncClient
    ) -> None:
        snapshot = ChainWeightsSnapshot(
            netuid=118,
            block=8_639_503,
            block_hash="0x" + "ab" * 32,
            owner_hotkey=_MINER_B,
            vectors=(
                ChainWeightVector(
                    validator_uid=25,
                    validator_hotkey=_VALIDATOR_C,
                    weights=(ChainWeight(uid=169, hotkey=_MINER_A, value=14745),),
                ),
            ),
        )
        app.state.chain = SimpleNamespace(get_weights=AsyncMock(return_value=snapshot))

        response = await client.get("/api/v1/public/weights")

        assert response.status_code == 200
        assert response.headers["cache-control"] == "public, max-age=30"
        body = response.json()
        assert body["netuid"] == 118
        assert body["block"] == 8_639_503
        assert body["block_hash"] == "0x" + "ab" * 32
        assert body["owner_hotkey"] == _MINER_B
        assert body["vectors"] == [
            {
                "validator_uid": 25,
                "validator_hotkey": _VALIDATOR_C,
                "weights": [{"uid": 169, "hotkey": _MINER_A, "value": 14745}],
            }
        ]
        app.state.chain.get_weights.assert_awaited_once_with(118)

    async def test_returns_503_when_chain_read_is_unavailable(
        self, app: FastAPI, client: httpx.AsyncClient
    ) -> None:
        app.state.chain = SimpleNamespace(
            get_weights=AsyncMock(side_effect=ChainError("rpc unavailable"))
        )

        response = await client.get("/api/v1/public/weights")

        assert response.status_code == 503
        assert response.json()["message"] == "chain weights unavailable"

    async def test_returns_503_when_chain_client_lacks_weight_read(
        self, app: FastAPI, client: httpx.AsyncClient
    ) -> None:
        app.state.chain = SimpleNamespace()

        response = await client.get("/api/v1/public/weights")

        assert response.status_code == 503


class TestPublicBenchmarkTimeline:
    async def test_returns_release_events_and_finalized_memory_highs(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        first_id = await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.41, 0.42, 0.43],
            details={"bench_version": 2},
            base_time=datetime(2026, 7, 8, tzinfo=UTC),
        )
        second_id = await _seed_k3(
            session_maker,
            miner=_MINER_B,
            composites=[0.71, 0.72, 0.73],
            details={"bench_version": 2},
            base_time=datetime(2026, 7, 9, tzinfo=UTC),
        )
        async with session_maker() as session, session.begin():
            session.add(
                BenchmarkRollout(
                    rollout_id=uuid4(),
                    from_version=2,
                    desired_version=3,
                    status="activated",
                    cohort_size=5,
                    created_at=datetime(2026, 7, 18, 14, 30, tzinfo=UTC),
                    activated_at=datetime(2026, 7, 18, 16, 0, tzinfo=UTC),
                )
            )
            for agent_id, recorded_at in (
                (UUID(first_id), datetime(2026, 7, 8, tzinfo=UTC)),
                (UUID(second_id), datetime(2026, 7, 9, tzinfo=UTC)),
            ):
                scores = list(
                    await session.scalars(
                        select(Score).where(Score.agent_id == agent_id)
                    )
                )
                for index, score in enumerate(scores):
                    score.created_at = recorded_at + timedelta(minutes=index)
                    score.updated_at = recorded_at + timedelta(minutes=index)
        await _seed_k3(
            session_maker,
            miner="5" + "A" * 47,
            composites=[0.51, 0.52],
            details={"bench_version": 3},
            base_time=datetime(2026, 7, 19, tzinfo=UTC),
        )
        _install_db(app, session_maker)

        response = await client.get("/api/v1/public/bench/timeline")

        assert response.status_code == 200
        assert response.headers["cache-control"] == "public, max-age=300"
        body = response.json()
        assert body["metric"] == "memory_mean"
        assert body["score_quorum"] == 3
        assert [release["bench_version"] for release in body["releases"]] == [
            2,
            3,
            4,
            5,
            6,
        ]
        assert body["releases"][0]["released_at"] == "2026-07-07T00:00:00Z"
        assert body["releases"][1]["released_at"] == "2026-07-18T14:30:00Z"
        assert body["releases"][1]["activated_at"] == "2026-07-18T16:00:00Z"
        assert [point["agent_id"] for point in body["points"]] == [
            first_id,
            second_id,
        ]
        assert [point["memory_mean"] for point in body["points"]] == [0.42, 0.72]


class TestPublicLeaderboard:
    async def test_distinguishes_raw_rank_one_from_koth_emissions_champion(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        details = {"bench_version": DEFAULT_BENCH_VERSION, "composite_stderr": 0.03}
        incumbent_id = await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.80, 0.80, 0.80],
            details=details,
            created_at=datetime(2026, 7, 15, tzinfo=UTC),
        )
        raw_leader_id = await _seed_k3(
            session_maker,
            miner=_MINER_B,
            composites=[0.85, 0.85, 0.85],
            details=details,
            created_at=datetime(2026, 7, 16, tzinfo=UTC),
        )
        _install_db(app, session_maker)
        app.state.chain = SimpleNamespace(
            get_recent_neurons=AsyncMock(
                return_value=[
                    SimpleNamespace(hotkey=_MINER_A, uid=41),
                    SimpleNamespace(hotkey=_MINER_B, uid=42),
                ]
            )
        )

        body = (await client.get("/api/v1/public/leaderboard")).json()

        assert body["entries"][0]["agent_id"] == raw_leader_id
        assert body["entries"][0]["rank"] == 1
        assert body["emissions"]["raw_leader_agent_id"] == raw_leader_id
        assert body["emissions"]["champion_agent_id"] == incumbent_id
        assert body["emissions"]["margin"] == pytest.approx(0.007)
        assert body["emissions"]["dethrone_z"] == pytest.approx(1.64)
        assert body["emissions"]["rank_shares"] == pytest.approx(
            [0.65, 0.14, 0.10, 0.07, 0.04]
        )
        decision = body["emissions"]["raw_leader_decision"]
        assert decision["challenger_lead"] == pytest.approx(0.05)
        assert decision["required_lead"] == pytest.approx(
            1.64 * (0.03**2 + 0.03**2) ** 0.5
        )
        assert decision["method"] == "unpaired"
        assert decision["dethrones"] is False
        assert body["emissions"]["recipients"] == [
            {
                "role": "champion",
                "agent_id": incumbent_id,
                "miner_hotkey": _MINER_A,
                "raw_rank": 2,
                "share_of_miner_pool": pytest.approx(0.65 / 0.79),
                "shared_seed_confirmations": 0,
            },
            {
                "role": "tail",
                "agent_id": raw_leader_id,
                "miner_hotkey": _MINER_B,
                "raw_rank": 1,
                "share_of_miner_pool": pytest.approx(0.14 / 0.79),
                "shared_seed_confirmations": 0,
            },
        ]

    async def test_leaderboard_surfaces_shared_seed_confirmation_depth(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        from ditto.db.queries.confirmation_scores import (
            ConfirmationSeedScore,
            append_confirmation_scores,
        )

        champion_id = await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.90, 0.90, 0.90],
            details={"bench_version": DEFAULT_BENCH_VERSION},
            created_at=datetime(2026, 6, 1, tzinfo=UTC),
        )
        await _seed_k3(
            session_maker,
            miner=_MINER_B,
            composites=[0.80, 0.80, 0.80],
            details={"bench_version": DEFAULT_BENCH_VERSION},
            created_at=datetime(2026, 6, 2, tzinfo=UTC),
        )
        # The champion accumulated three champion-anchored shared-seed rescores.
        async with session_maker() as s, s.begin():
            await append_confirmation_scores(
                s,
                rows=[
                    ConfirmationSeedScore(
                        UUID(champion_id), "5V1", seed, 0.90, f"r{seed}", None
                    )
                    for seed in (100, 200, 300)
                ],
                bench_version=DEFAULT_BENCH_VERSION,
                created_at=datetime.now(UTC),
            )
        _install_db(app, session_maker)

        body = (await client.get("/api/v1/public/leaderboard")).json()
        recipients = {r["agent_id"]: r for r in body["emissions"]["recipients"]}
        assert recipients[champion_id]["shared_seed_confirmations"] == 3

    async def test_marks_deregistered_scores_retained_but_emission_ineligible(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.7, 0.8, 0.9],
            details={"bench_version": DEFAULT_BENCH_VERSION},
        )
        await _seed_k3(
            session_maker,
            miner=_MINER_B,
            composites=[0.6, 0.7, 0.8],
            details={"bench_version": DEFAULT_BENCH_VERSION},
        )
        _install_db(app, session_maker)
        app.state.chain = SimpleNamespace(
            get_recent_neurons=AsyncMock(
                return_value=[SimpleNamespace(hotkey=_MINER_B, uid=42)]
            )
        )

        body = (await client.get("/api/v1/public/leaderboard")).json()

        by_miner = {e["miner_hotkey"]: e for e in body["entries"]}
        assert by_miner[_MINER_A]["registered"] is False
        assert by_miner[_MINER_A]["miner_uid"] is None
        assert by_miner[_MINER_A]["emission_eligible"] is False
        assert by_miner[_MINER_A]["finalized"] is True
        assert by_miner[_MINER_A]["score_count"] == 3
        assert by_miner[_MINER_B]["registered"] is True
        assert by_miner[_MINER_B]["miner_uid"] == 42
        assert by_miner[_MINER_B]["emission_eligible"] is True

    async def test_chain_error_keeps_leaderboard_available_with_unknown_registration(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.7, 0.8, 0.9],
            details={"bench_version": 2},
        )
        _install_db(app, session_maker)
        app.state.chain = SimpleNamespace(
            get_recent_neurons=AsyncMock(side_effect=ChainError("pylon unavailable"))
        )

        response = await client.get("/api/v1/public/leaderboard")

        assert response.status_code == 200
        entry = response.json()["entries"][0]
        assert entry["registered"] is None
        assert entry["emission_eligible"] is None

    async def test_chain_timeout_keeps_leaderboard_available_with_unknown_registration(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.7, 0.8, 0.9],
            details={"bench_version": 2},
        )
        _install_db(app, session_maker)

        async def _never_returns(_netuid: int) -> list[object]:
            await asyncio.Event().wait()
            return []

        app.state.chain = SimpleNamespace(get_recent_neurons=_never_returns)
        monkeypatch.setattr(
            public_endpoint, "_REGISTRATION_LOOKUP_TIMEOUT_SECONDS", 0.001
        )

        response = await client.get("/api/v1/public/leaderboard")

        assert response.status_code == 200
        entry = response.json()["entries"][0]
        assert entry["registered"] is None
        assert entry["emission_eligible"] is None

    async def test_includes_pre_quorum_scores_as_provisional_feedback(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.6, 0.8],
            status=AgentStatus.EVALUATING,
        )
        _install_db(app, session_maker)

        body = (await client.get("/api/v1/public/leaderboard")).json()

        assert body["count"] == 1
        entry = body["entries"][0]
        assert entry["miner_hotkey"] == _MINER_A
        assert entry["composite"] == pytest.approx(0.7)
        assert entry["tool_mean"] == pytest.approx(0.7)
        assert entry["memory_mean"] == pytest.approx(0.7)
        assert entry["finalized"] is False
        assert entry["score_count"] == 2
        assert entry["score_quorum"] == 3
        assert entry["bench_version"] == DEFAULT_BENCH_VERSION

    async def test_open_rollout_exposes_settled_and_rollout_state_per_entry(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        """Mid-rollout, every entry carries the settled v2 median plus the v3
        settlement state (median so far + score count). With the temporary
        authority pin, even a complete v3 quorum stays on its v2 median until
        the rollout activates."""
        flipped_id = await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.80, 0.80, 0.80],
        )
        partial_id = await _seed_k3(
            session_maker,
            miner=_MINER_B,
            composites=[0.85, 0.85, 0.85],
        )
        async with session_maker() as s, s.begin():
            s.add(
                BenchmarkRollout(
                    rollout_id=uuid4(),
                    from_version=2,
                    desired_version=3,
                    status="collecting",
                    cohort_size=5,
                    created_at=datetime.now(UTC),
                )
            )
            for i, composite in enumerate([0.90, 0.92, 0.94]):
                await upsert_score(
                    s,
                    agent_id=UUID(flipped_id),
                    validator_hotkey=f"5Validator{i}Flipped",
                    bench_version=3,
                    run_id=f"v3_run_{i}",
                    seed=1,
                    composite=composite,
                    tool_mean=composite,
                    memory_mean=composite,
                    median_ms=500,
                    n=110,
                    generated_at=datetime(2026, 7, 18, 12, i, tzinfo=UTC),
                    signature="ab" * 64,
                )
            await upsert_score(
                s,
                agent_id=UUID(partial_id),
                validator_hotkey="5Validator0Partial",
                bench_version=3,
                run_id="v3_run_partial",
                seed=1,
                composite=0.5,
                tool_mean=0.5,
                memory_mean=0.5,
                median_ms=500,
                n=110,
                generated_at=datetime(2026, 7, 18, 12, 0, tzinfo=UTC),
                signature="ab" * 64,
            )
        _install_db(app, session_maker)

        body = (await client.get("/api/v1/public/leaderboard")).json()

        assert body["active_bench_version"] == 2
        assert body["desired_bench_version"] == 3
        assert body["available_bench_versions"] == [3, 2]
        by_agent = {e["agent_id"]: e for e in body["entries"]}
        flipped = by_agent[flipped_id]
        assert flipped["bench_version"] == DEFAULT_BENCH_VERSION
        assert flipped["composite"] == pytest.approx(0.80)
        assert flipped["settled_composite"] == pytest.approx(0.80)
        assert flipped["rollout_composite"] == pytest.approx(0.92)
        assert flipped["rollout_score_count"] == 3
        partial = by_agent[partial_id]
        assert partial["bench_version"] == DEFAULT_BENCH_VERSION
        assert partial["composite"] == pytest.approx(0.85)
        assert partial["settled_composite"] == pytest.approx(0.85)
        assert partial["rollout_composite"] == pytest.approx(0.5)
        assert partial["rollout_score_count"] == 1

    async def test_rollout_state_is_null_without_an_open_rollout(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.7, 0.8, 0.9],
        )
        _install_db(app, session_maker)

        body = (await client.get("/api/v1/public/leaderboard")).json()

        entry = body["entries"][0]
        assert entry["settled_composite"] is None
        assert entry["rollout_composite"] is None
        assert entry["rollout_score_count"] is None

    async def test_finalized_miner_supersedes_partial_submission(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.4, 0.5, 0.6],
        )
        await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.99],
            status=AgentStatus.EVALUATING,
        )
        _install_db(app, session_maker)

        body = (await client.get("/api/v1/public/leaderboard")).json()

        assert body["count"] == 1
        entry = body["entries"][0]
        assert entry["composite"] == pytest.approx(0.5)
        assert entry["finalized"] is True
        assert entry["score_count"] == 3

    async def test_ranks_by_composite_and_exposes_aggregates(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_scored(
            session_maker, miner=_MINER_A, composite=0.4, tool_mean=0.5, memory_mean=0.3
        )
        await _seed_scored(
            session_maker,
            miner=_MINER_B,
            composite=0.9,
            tool_mean=0.95,
            memory_mean=0.8,
        )
        # Held (suspected copy) must not surface.
        await _seed_scored(
            session_maker,
            miner="5HeldMinerXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            composite=0.99,
            tool_mean=0.99,
            memory_mean=0.99,
            status=AgentStatus.ATH_PENDING_REVIEW,
        )
        _install_db(app, session_maker)

        resp = await client.get("/api/v1/public/leaderboard")
        assert resp.status_code == 200
        assert resp.headers["Cache-Control"] == "public, max-age=30"
        body = resp.json()
        assert body["selection_mode"] == "authoritative"
        assert body["active_bench_version"] == DEFAULT_BENCH_VERSION
        assert body["desired_bench_version"] == DEFAULT_BENCH_VERSION
        assert body["current_bench_version"] == DEFAULT_BENCH_VERSION
        assert body["available_bench_versions"] == [DEFAULT_BENCH_VERSION]
        assert body["count"] == 2
        assert [e["rank"] for e in body["entries"]] == [1, 2]
        assert [e["miner_hotkey"] for e in body["entries"]] == [_MINER_B, _MINER_A]
        assert all(e["finalized"] is False for e in body["entries"])
        assert all(e["score_count"] == 1 for e in body["entries"])
        top = body["entries"][0]
        assert top["agent_name"] == "agent"
        assert top["agent_version"] is None
        assert top["composite"] == pytest.approx(0.9)
        assert top["tool_mean"] == pytest.approx(0.95)
        assert top["memory_mean"] == pytest.approx(0.8)

        historical = (
            await client.get(
                f"/api/v1/public/leaderboard?bench_version={DEFAULT_BENCH_VERSION}"
            )
        ).json()
        assert historical["selection_mode"] == "historical"
        assert historical["entries"] == body["entries"]
        assert historical["emissions"] is None
        assert historical["available_bench_versions"] == [DEFAULT_BENCH_VERSION]

    async def test_exposes_advisory_calibration(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # P5: the advisory Brier calibration telemetry surfaces as an unscored
        # column; a run without it (or with a malformed value) shows null.
        await _seed_scored(
            session_maker,
            miner=_MINER_A,
            composite=0.7,
            tool_mean=0.7,
            memory_mean=0.7,
            details={"calibration_brier": 0.12, "calibration_n": 34},
        )
        await _seed_scored(
            session_maker,
            miner=_MINER_B,
            composite=0.6,
            tool_mean=0.6,
            memory_mean=0.6,
            details={"calibration_brier": 7.5},  # out of range → dropped
        )
        _install_db(app, session_maker)

        body = (await client.get("/api/v1/public/leaderboard")).json()
        by_miner = {e["miner_hotkey"]: e for e in body["entries"]}
        assert by_miner[_MINER_A]["calibration_brier"] == pytest.approx(0.12)
        assert by_miner[_MINER_A]["calibration_n"] == 34
        assert by_miner[_MINER_B]["calibration_brier"] is None
        assert by_miner[_MINER_B]["calibration_n"] is None

    async def test_never_leaks_integrity_fields(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # Seed a run whose details carry the raw per-case answer key so we can
        # assert it is redacted out, not merely absent because it was never set.
        details = {
            "bench_version": 2,
            "per_case": [
                {
                    "kind": "tool",
                    "category": "web_search",
                    "score": 0.6,
                    "correct": False,
                    "latency_ms": 3382,
                    "notes": ["1 extra/unexpected tool call(s)"],
                    "expected": ["search_web"],
                    "called": ["search_web", "search_web"],
                    "case_id": "web_search-8860569897825046057-0001",
                },
            ],
        }
        await _seed_scored(
            session_maker,
            miner=_MINER_A,
            composite=0.4,
            tool_mean=0.5,
            memory_mean=0.3,
            details=details,
        )
        _install_db(app, session_maker)

        resp = await client.get("/api/v1/public/leaderboard")
        entry = resp.json()["entries"][0]
        # agent_id is deliberately exposed (already public via /submissions and the
        # per-agent drill-in endpoints) so the dashboard can link a row to its k=3
        # record; the seed and the per-validator/artifact identifiers stay hidden.
        assert "agent_id" in entry
        for leaked in ("signature", "sha256", "validator_hotkey", "seed"):
            assert leaked not in entry
        # The answer key must appear NOWHERE in the whole response, even nested
        # inside the redacted per-case results. Check the quoted JSON keys (so a
        # note like "unexpected tool call" doesn't false-match "expected") plus
        # the expected/called tool token itself.
        raw = resp.text
        for answer_key in ('"expected"', '"called"', '"case_id"', "search_web"):
            assert answer_key not in raw
        # …but the safe, redacted per-case view IS surfaced for analysis.
        cases = entry["case_results"]
        assert cases and cases[0]["category"] == "web_search"
        assert cases[0]["score"] == pytest.approx(0.6)
        assert cases[0]["correct"] is False
        assert set(cases[0]).issubset(
            {"category", "kind", "score", "correct", "latency_ms", "notes"}
        )

    async def test_empty_ledger(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        resp = await client.get("/api/v1/public/leaderboard")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 0
        assert body["entries"] == []

    async def test_no_auth_required(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        # No X-Validator-Hotkey header, no chain override — must still succeed.
        resp = await client.get("/api/v1/public/leaderboard")
        assert resp.status_code == 200


class TestPublicHealth:
    async def test_counts_latency_and_window(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        now = datetime.now(UTC)
        # Two scored miners, latencies 400 + 800 => avg 600. The signed report
        # timestamps are deliberately stale: public activity must use when the
        # platform recorded each score, not validator-controlled provenance.
        await _seed_scored(
            session_maker,
            miner=_MINER_A,
            composite=0.4,
            tool_mean=0.5,
            memory_mean=0.3,
            median_ms=400,
            generated_at=datetime(2026, 1, 1, tzinfo=UTC),
            recorded_at=now - timedelta(minutes=5),
        )
        await _seed_scored(
            session_maker,
            miner=_MINER_B,
            composite=0.9,
            tool_mean=0.95,
            memory_mean=0.8,
            median_ms=800,
            generated_at=datetime(2026, 1, 1, tzinfo=UTC),
            recorded_at=now - timedelta(days=2),  # outside the 24h window
        )
        # A third miner who submitted but has not been scored yet.
        await _seed_agent(
            session_maker,
            miner="5CFn5zVKp6taKY8T39M92cWWpsCXBQym37waFAtiKmZmznu9",
            status=AgentStatus.UPLOADED,
        )
        _install_db(app, session_maker)

        resp = await client.get("/api/v1/public/health")
        assert resp.status_code == 200
        assert resp.headers["Cache-Control"] == "public, max-age=30"
        body = resp.json()
        assert body["miners"] == 3
        assert body["scored_miners"] == 2
        assert body["scored_agents"] == 2
        assert body["total_scores"] == 2
        assert body["scores_24h"] == 1  # only MINER_A is within 24h
        assert body["avg_latency_ms"] == 600
        # last_scored_at is the newest platform write (MINER_A, ~5 min ago).
        last = datetime.fromisoformat(body["last_scored_at"])
        assert abs((now - last).total_seconds()) < 3600

    async def test_orphan_scored_agent_not_counted(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # A scored-STATUS agent with no score row (a stray/hand-edited state)
        # must not inflate the scored counts — they require a real score row so
        # health can never contradict the leaderboard.
        await _seed_scored(
            session_maker,
            miner=_MINER_A,
            composite=0.5,
            tool_mean=0.6,
            memory_mean=0.4,
            generated_at=datetime.now(UTC),
        )
        await _seed_agent(
            session_maker, miner=_MINER_B, status=AgentStatus.SCORED
        )  # scored status, but no score row
        _install_db(app, session_maker)

        body = (await client.get("/api/v1/public/health")).json()
        assert body["miners"] == 2  # both submitted
        assert body["scored_miners"] == 1  # only MINER_A is score-backed
        assert body["scored_agents"] == 1

    async def test_held_agent_not_counted_as_scored(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # A held (ATH review) agent has a score but is not eligible: it counts
        # toward total miners but not scored_miners/scored_agents.
        await _seed_scored(
            session_maker,
            miner=_MINER_A,
            composite=0.99,
            tool_mean=0.99,
            memory_mean=0.99,
            status=AgentStatus.ATH_PENDING_REVIEW,
            generated_at=datetime.now(UTC),
        )
        _install_db(app, session_maker)

        body = (await client.get("/api/v1/public/health")).json()
        assert body["miners"] == 1
        assert body["scored_miners"] == 0
        assert body["scored_agents"] == 0

    async def test_empty(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        resp = await client.get("/api/v1/public/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {
            "generated_at": body["generated_at"],
            "miners": 0,
            "scored_miners": 0,
            "scored_agents": 0,
            "last_scored_at": None,
            "total_scores": 0,
            "scores_24h": 0,
            "avg_latency_ms": None,
        }


class TestPublicFleet:
    def test_stale_boundaries_and_recovery_after_delayed_heartbeat(self) -> None:
        now = datetime(2026, 7, 14, 20, 0, tzinfo=UTC)

        assert _fleet_classification(
            state="idle",
            seen_at=now - timedelta(minutes=5),
            now=now,
            metrics=None,
        )[:2] == (True, "available")
        assert _fleet_classification(
            state="running_benchmark",
            seen_at=now - timedelta(minutes=5, microseconds=1),
            now=now,
            metrics=None,
        )[:2] == (False, "stale")
        assert _fleet_classification(
            state="running_benchmark",
            seen_at=now - timedelta(minutes=15),
            now=now,
            metrics=None,
        )[:2] == (False, "stale")
        assert _fleet_classification(
            state="running_benchmark",
            seen_at=now - timedelta(minutes=15, microseconds=1),
            now=now,
            metrics=None,
        )[:2] == (False, "offline")
        assert _fleet_classification(
            state="running_benchmark", seen_at=now, now=now, metrics=None
        )[:2] == (True, "available")

    def test_stack_health_rolls_required_degraded_components_into_warning(
        self,
    ) -> None:
        def _component(
            health: ComponentHealthState, required: bool = True
        ) -> ValidatorComponentHealth:
            observed = None if health == "unknown" else 1_784_000_000
            ready = None if health in ("unknown", "unreachable") else True
            return ValidatorComponentHealth(
                health=health, required=required, observed_at=observed, ready=ready
            )

        def _stack(**overrides: ValidatorComponentHealth) -> ValidatorStackHealth:
            base = {
                name: _component("healthy")
                for name in ValidatorStackHealth.model_fields
            }
            base.update(overrides)
            return ValidatorStackHealth(**base)

        assert public_endpoint._stack_component_issues(None) == []
        assert public_endpoint._stack_component_issues(_stack()) == []
        # A reachable-but-degraded required scorer (its relay path is down) is
        # named with its exact state, not collapsed into a bare flag.
        assert public_endpoint._stack_component_issues(
            _stack(dittobench_api=_component("degraded"))
        ) == ["dittobench_api: degraded"]
        assert public_endpoint._stack_component_issues(
            _stack(model_relay=_component("unreachable"))
        ) == ["model_relay: unreachable"]
        # "unknown" is not-observed and must never raise a false warning.
        assert (
            public_endpoint._stack_component_issues(
                _stack(model_relay=_component("unknown"))
            )
            == []
        )
        # A non-required component in a bad state does not warn the fleet.
        assert (
            public_endpoint._stack_component_issues(
                _stack(pylon=_component("degraded", required=False))
            )
            == []
        )

    def test_health_reasons_name_every_cause_for_the_badge(self) -> None:
        def _component(
            health: ComponentHealthState, required: bool = True
        ) -> ValidatorComponentHealth:
            observed = None if health == "unknown" else 1_784_000_000
            ready = None if health in ("unknown", "unreachable") else True
            return ValidatorComponentHealth(
                health=health, required=required, observed_at=observed, ready=ready
            )

        def _stack(**overrides: ValidatorComponentHealth) -> ValidatorStackHealth:
            base = {
                name: _component("healthy")
                for name in ValidatorStackHealth.model_fields
            }
            base.update(overrides)
            return ValidatorStackHealth(**base)

        # A fully healthy validator carries no reasons.
        assert (
            public_endpoint._health_reasons(
                state="idle",
                metrics=PublicSystemMetrics(
                    cpu_percent=0,
                    memory_percent=10,
                    disk_percent=10,
                    docker_status="healthy",
                    running_containers=1,
                    unhealthy_containers=0,
                ),
                active_benchmark=None,
                stack_health=_stack(),
            )
            == []
        )
        # Every distinct cause is named; the stack cause carries the component.
        reasons = public_endpoint._health_reasons(
            state="idle",
            metrics=PublicSystemMetrics(
                cpu_percent=0,
                memory_percent=95,
                disk_percent=10,
                docker_status="healthy",
                running_containers=1,
                unhealthy_containers=0,
            ),
            active_benchmark=None,
            stack_health=_stack(dittobench_api=_component("degraded")),
        )
        assert reasons == ["memory 95%", "dittobench_api: degraded"]
        # No metrics reported explains an otherwise-unknown badge.
        assert public_endpoint._health_reasons(
            state="idle", metrics=None, active_benchmark=None, stack_health=None
        ) == ["host metrics not reported"]

    async def test_validator_name_response_is_allowlisted_to_reporters(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        now = datetime.now(UTC)
        async with session_maker() as session, session.begin():
            session.add(
                ValidatorHeartbeat(
                    validator_hotkey=_MINER_A,
                    software_version="1.2.3",
                    protocol_version=4,
                    code_digest="ab" * 32,
                    state="idle",
                    reported_at=now,
                    seen_at=now,
                    signature="cd" * 64,
                )
            )
        _install_db(app, session_maker)

        class Names:
            calls = 0

            def snapshot(self, hotkeys: list[str]) -> ValidatorNamesSnapshot:
                self.calls += 1
                assert hotkeys == [_MINER_A]
                return ValidatorNamesSnapshot(
                    status="fresh",
                    refreshed_at=now,
                    names={_MINER_A: "Rizzo", _MINER_B: "Not a reporter"},
                    stake_weights={_MINER_A: 123.5, _MINER_B: 456.0},
                )

        names = Names()
        app.state.validator_names = names
        response = await client.get("/api/v1/public/validator-names")

        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "public, max-age=30"
        body = response.json()
        assert set(body) == {
            "generated_at",
            "source",
            "status",
            "refreshed_at",
            "validators",
        }
        assert body["source"] == "taostats"
        assert body["status"] == "fresh"
        assert body["validators"] == [
            {
                "validator_hotkey": _MINER_A,
                "display_name": "Rizzo",
                "stake_weight": 123.5,
            }
        ]
        assert names.calls == 1

    async def test_core_fleet_endpoint_never_reads_external_name_cache(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)

        class ExplodingNames:
            def snapshot(self, hotkeys: list[str]) -> ValidatorNamesSnapshot:
                raise AssertionError(f"unexpected name lookup for {hotkeys}")

        app.state.validator_names = ExplodingNames()
        response = await client.get("/api/v1/public/validators")

        assert response.status_code == 200
        assert response.json()["validators"] == []


class TestPublicActivity:
    async def test_lists_all_stages_newest_first_without_sensitive_fields(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        older_id = await _seed_agent(
            session_maker,
            miner=_MINER_A,
            status=AgentStatus.UPLOADED,
            name="memory-v1",
            created_at=datetime(2026, 7, 13, 10, 0, 0, tzinfo=UTC),
        )
        await _seed_agent(
            session_maker,
            miner=_MINER_B,
            status=AgentStatus.ATH_PENDING_REVIEW,
            name="memory-v2",
            created_at=datetime(2026, 7, 13, 11, 0, 0, tzinfo=UTC),
            duplicate_of=UUID(older_id),
            review_reason=(
                f"content near-duplicate of agent {older_id}: "
                "composite delta 0.0010, jaccard 0.950"
            ),
        )
        await _seed_agent(
            session_maker,
            miner=_MINER_A,
            status=AgentStatus.BANNED,
            name="memory-v3",
            created_at=datetime(2026, 7, 13, 12, 0, 0, tzinfo=UTC),
            screening_reason="Docker image build failed",
        )
        _install_db(app, session_maker)

        resp = await client.get("/api/v1/public/activity")
        assert resp.status_code == 200
        assert resp.headers["Cache-Control"] == "public, max-age=10"
        body = resp.json()
        assert body["count"] == 3
        assert body["total"] == 3
        assert body["page"] == 1
        assert body["page_size"] == 50
        assert body["total_pages"] == 1
        assert [entry["name"] for entry in body["entries"]] == [
            "memory-v3",
            "memory-v2",
            "memory-v1",
        ]
        assert [entry["status"] for entry in body["entries"]] == [
            "rejected",
            "under_review",
            "waiting_screening",
        ]
        assert body["entries"][2]["agent_id"] == older_id
        assert body["entries"][0]["screening_reason"] == "Docker image build failed"
        assert body["entries"][1]["duplicate_of"] == older_id
        assert body["entries"][1]["duplicate_name"] == "memory-v1"
        assert body["entries"][1]["duplicate_version"] is None
        assert "jaccard 0.950" in body["entries"][1]["review_reason"]
        assert set(body["entries"][0]) == {
            "agent_id",
            "miner_hotkey",
            "name",
            "version",
            "status",
            "submitted_at",
            "last_scored_at",
            "screening_reason",
            "duplicate_of",
            "duplicate_name",
            "duplicate_version",
            "review_reason",
            "review_opened_at",
            "preserved_composite",
            "score_count",
            "provisional_composite",
            "validator_queue_rank",
            "quorum",
            "retry_state",
            "retry_after",
            "screening_policy_version",
            "required_screening_policy_version",
            "screening_attempt_id",
            "screening_started_at",
            "screening_deadline",
            "active_benchmarks",
        }
        serialized = resp.text
        for private_field in (
            "sha256",
            "artifact",
            "payment",
            "SECRET_FROM_BUILD",
        ):
            assert private_field not in serialized

    async def test_ath_review_filter_is_public_safe_and_includes_hold_snapshot(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        held_id = UUID(
            await _seed_k3(
                session_maker,
                miner=_MINER_A,
                composites=[0.4, 0.8, 0.9],
                status=AgentStatus.ATH_PENDING_REVIEW,
            )
        )
        await _seed_agent(
            session_maker,
            miner=_MINER_B,
            status=AgentStatus.QUARANTINED,
            name="screening-review",
        )
        opened_at = datetime(2026, 7, 16, 15, 30, tzinfo=UTC)
        async with session_maker() as session, session.begin():
            held = await session.get(Agent, held_id)
            assert held is not None
            held.name = "memory-harness"
            held.version = 4
            held.review_reason = "Submission requires ATH similarity review"
            session.add(
                AthReview(
                    review_id=uuid4(),
                    agent_id=held_id,
                    status="pending",
                    opened_at=opened_at,
                    original_duplicate_of=None,
                    original_reason=held.review_reason,
                    original_policy_version=8,
                    original_evidence={
                        "sha256": held.sha256,
                        "challenge_value": "private-challenge",
                        "answer_key": "private-answer-key",
                        "source_path": "/private/source.rs",
                    },
                    algorithm_provenance={
                        "opened_by": "private-operator",
                        "credential": "private-credential",
                    },
                )
            )
        _install_db(app, session_maker)

        response = await client.get(
            "/api/v1/public/activity?review=ath&status=under_review&limit=200"
        )

        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "public, max-age=10"
        body = response.json()
        assert body["total"] == body["count"] == 1
        entry = body["entries"][0]
        assert entry["agent_id"] == str(held_id)
        assert entry["name"] == "memory-harness"
        assert entry["version"] == 4
        assert entry["miner_hotkey"] == _MINER_A
        assert entry["status"] == "under_review"
        assert datetime.fromisoformat(entry["review_opened_at"]) == opened_at.replace(
            tzinfo=None
        )
        assert entry["review_reason"] == "Submission requires ATH similarity review"
        assert entry["score_count"] == 3
        assert entry["provisional_composite"] == pytest.approx(0.7)
        assert entry["preserved_composite"] == pytest.approx(0.8)
        serialized = response.text.lower()
        for private_value in (
            "sha256",
            "private-challenge",
            "private-answer-key",
            "private/source.rs",
            "private-operator",
            "private-credential",
            "opened_by",
        ):
            assert private_value not in serialized

    async def test_exposes_queue_priority_with_provisional_composites(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_top_five_floor(session_maker, fifth_place=0.60)
        zero_id = await _seed_agent(
            session_maker,
            miner=_MINER_A,
            status=AgentStatus.EVALUATING,
            name="zero",
            screening_policy_version=SCREENING_POLICY_VERSION,
        )
        one_id = await _seed_k3(
            session_maker,
            miner=_MINER_B,
            composites=[0.5],
            status=AgentStatus.EVALUATING,
        )
        one_high_id = await _seed_k3(
            session_maker,
            miner=_VALIDATOR_C,
            composites=[0.7],
            status=AgentStatus.EVALUATING,
        )
        low_id = await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.2, 0.3],
            status=AgentStatus.EVALUATING,
        )
        high_id = await _seed_k3(
            session_maker,
            miner=_MINER_B,
            composites=[0.8, 0.9],
            status=AgentStatus.EVALUATING,
        )
        async with session_maker() as session, session.begin():
            for agent_id in (one_id, one_high_id, low_id, high_id):
                agent = await session.get(Agent, UUID(agent_id))
                assert agent is not None
                agent.screening_policy_version = SCREENING_POLICY_VERSION
        _install_db(app, session_maker)

        response = await client.get("/api/v1/public/activity")
        by_id = {entry["agent_id"]: entry for entry in response.json()["entries"]}

        assert by_id[high_id]["validator_queue_rank"] == 1
        assert by_id[zero_id]["validator_queue_rank"] == 2
        assert by_id[one_high_id]["validator_queue_rank"] == 3
        assert by_id[one_id]["validator_queue_rank"] == 4
        assert by_id[low_id]["validator_queue_rank"] == 5
        assert by_id[low_id]["status"] == "below_score_floor"
        assert by_id[zero_id]["provisional_composite"] is None
        assert by_id[one_id]["provisional_composite"] == pytest.approx(0.5)
        assert by_id[one_high_id]["provisional_composite"] == pytest.approx(0.7)
        assert by_id[high_id]["provisional_composite"] == pytest.approx(0.85)
        assert by_id[low_id]["provisional_composite"] == pytest.approx(0.25)

    async def test_filters_complete_dataset_before_paginating_with_counts(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        for index in range(12):
            await _seed_agent(
                session_maker,
                miner=_MINER_A,
                status=AgentStatus.UPLOADED,
                name=f"queued-{index}",
                created_at=datetime(2026, 7, 13, 10, index, tzinfo=UTC),
            )
        await _seed_agent(
            session_maker,
            miner=_MINER_B,
            status=AgentStatus.BANNED,
            name="rejected-late",
            created_at=datetime(2026, 7, 13, 9, 0, tzinfo=UTC),
        )
        _install_db(app, session_maker)

        body = (
            await client.get("/api/v1/public/activity?status=rejected&page=1&limit=10")
        ).json()

        assert body["total"] == 1
        assert body["count"] == 1
        assert body["entries"][0]["name"] == "rejected-late"
        assert body["status_counts"]["waiting_screening"] == 12
        assert body["status_counts"]["rejected"] == 1

    async def test_combines_states_and_composes_with_search(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_agent(
            session_maker,
            miner=_MINER_A,
            status=AgentStatus.UPLOADED,
            name="alpha queued",
        )
        await _seed_agent(
            session_maker,
            miner=_MINER_B,
            status=AgentStatus.SCREENING,
            name="alpha screening",
        )
        await _seed_agent(
            session_maker,
            miner=_MINER_A,
            status=AgentStatus.BANNED,
            name="alpha rejected",
        )
        await _seed_agent(
            session_maker,
            miner=_MINER_B,
            status=AgentStatus.UPLOADED,
            name="beta queued",
        )
        _install_db(app, session_maker)

        response = await client.get(
            "/api/v1/public/activity",
            params=[
                ("status", "waiting_screening"),
                ("status", "screening"),
                ("q", "alpha"),
            ],
        )

        assert response.status_code == 200
        body = response.json()
        assert {entry["name"] for entry in body["entries"]} == {
            "alpha queued",
            "alpha screening",
        }
        assert body["total"] == 2
        assert body["status_counts"] == {
            "waiting_screening": 1,
            "screening": 1,
            "rejected": 1,
        }

    async def test_rejects_unknown_public_status_filter(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)

        response = await client.get("/api/v1/public/activity?status=obsolete")

        assert response.status_code == 422
        assert "unknown public activity status: obsolete" in response.text

    async def test_exposes_latest_platform_score_time_for_finalized_agents(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        recorded_at = datetime(2026, 7, 14, 9, 30, 0, tzinfo=UTC)
        agent_id = UUID(
            await _seed_k3(
                session_maker,
                miner=_MINER_A,
                composites=[0.61, 0.64, 0.67],
                status=AgentStatus.LIVE,
                # Validator provenance may be stale or inaccurate and must not drive
                # the public dashboard's relative score age.
                base_time=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )
        async with session_maker() as session, session.begin():
            scores = (
                (await session.execute(select(Score).where(Score.agent_id == agent_id)))
                .scalars()
                .all()
            )
            for index, score in enumerate(scores):
                score.created_at = recorded_at - timedelta(minutes=index)
                score.updated_at = recorded_at - timedelta(minutes=index)

        _install_db(app, session_maker)

        entry = (await client.get("/api/v1/public/activity")).json()["entries"][0]

        assert entry["status"] == "live"
        assert entry["score_count"] == 3
        assert datetime.fromisoformat(entry["last_scored_at"]) == recorded_at

    async def test_active_rescreen_projects_yellow_and_exposes_version_history(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = UUID(
            await _seed_agent(
                session_maker,
                miner=_MINER_A,
                status=AgentStatus.SCREENING,
                screening_reason="Container failed the health check",
                screening_policy_version=SCREENING_POLICY_VERSION - 1,
            )
        )
        now = datetime.now(UTC)
        old_attempt_id = uuid4()
        active_attempt_id = uuid4()
        async with session_maker() as session, session.begin():
            session.add_all(
                [
                    ScreeningAttempt(
                        attempt_id=old_attempt_id,
                        agent_id=agent_id,
                        screener_hotkey=_MINER_B,
                        policy_version=SCREENING_POLICY_VERSION - 1,
                        status="rejected",
                        started_at=now - timedelta(hours=1),
                        deadline=now - timedelta(minutes=40),
                        finished_at=now - timedelta(minutes=45),
                        public_reason="Container failed the health check",
                    ),
                    ScreeningAttempt(
                        attempt_id=active_attempt_id,
                        agent_id=agent_id,
                        screener_hotkey=_MINER_B,
                        policy_version=SCREENING_POLICY_VERSION,
                        status="running",
                        started_at=now,
                        deadline=now + timedelta(minutes=30),
                    ),
                ]
            )
        _install_db(app, session_maker)

        activity = (await client.get("/api/v1/public/activity")).json()["entries"][0]
        assert activity["status"] == "screening"
        assert activity["screening_reason"] is None
        assert activity["screening_policy_version"] == SCREENING_POLICY_VERSION - 1
        assert activity["required_screening_policy_version"] == SCREENING_POLICY_VERSION
        assert activity["screening_attempt_id"] == str(active_attempt_id)

        response = await client.get(f"/api/v1/public/agent/{agent_id}/pipeline")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "screening"
        assert [
            attempt["policy_version"] for attempt in body["screening_attempts"]
        ] == [
            SCREENING_POLICY_VERSION,
            SCREENING_POLICY_VERSION - 1,
        ]
        assert [attempt["status"] for attempt in body["screening_attempts"]] == [
            "running",
            "rejected",
        ]

    async def test_each_score_carries_its_own_bench_versions_dataset_digest(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        """A v3 score must not be published with the v2 dataset digest.

        Dataset provenance is per bench version, but the agent row carries only
        the version it was first pinned at. Pairing every score with that column
        advertised the v2 digest next to a verification_command naming v3, so a
        verifier would render v3 and get a mismatch on a perfectly good score.
        """
        agent_id = uuid4()
        v2_sha, v3_sha = "a1" * 32, "b2" * 32
        async with session_maker() as session, session.begin():
            session.add(
                Agent(
                    agent_id=agent_id,
                    miner_hotkey=_MINER_A,
                    name="agent",
                    sha256="ab" * 32,
                    size_bytes=524288,
                    status=AgentStatus.SCORED,
                    dataset_seed=42,
                    dataset_sha256=v2_sha,
                    dataset_run_size="full",
                    created_at=datetime.now(UTC),
                )
            )
            await session.flush()
            # Only v3 is pinned; v2 predates versioned pins and falls back to the
            # agent column, which is exactly the mixed state production is in.
            session.add(
                BenchmarkDataset(
                    agent_id=agent_id,
                    bench_version=3,
                    seed=42,
                    sha256=v3_sha,
                    run_size="full",
                )
            )
            for bench_version, hotkey in ((2, _VALIDATOR_C), (3, _MINER_B)):
                await upsert_score(
                    session,
                    agent_id=agent_id,
                    validator_hotkey=hotkey,
                    bench_version=bench_version,
                    run_id=f"run_{bench_version}",
                    seed=42,
                    composite=0.9,
                    tool_mean=0.9,
                    memory_mean=0.9,
                    median_ms=500,
                    n=20,
                    generated_at=datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC),
                    signature="ab" * 64,
                    details={"bench_version": bench_version},
                )
        _install_db(app, session_maker)

        response = await client.get(f"/api/v1/public/agent/{agent_id}/pipeline")

        assert response.status_code == 200
        by_version = {
            score["bench_version"]: score["dataset_sha256"]
            for score in response.json()["provisional_scores"]
        }
        assert by_version == {2: v2_sha, 3: v3_sha}

    async def test_stale_rejection_projects_as_waiting_for_rescreen(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_agent(
            session_maker,
            miner=_MINER_A,
            status=AgentStatus.REJECTED,
            screening_reason="Container failed the health check",
            screening_policy_version=SCREENING_POLICY_VERSION - 1,
        )
        _install_db(app, session_maker)

        entry = (await client.get("/api/v1/public/activity")).json()["entries"][0]
        assert entry["status"] == "waiting_screening"
        assert entry["screening_reason"] is None

    async def test_quarantined_attempt_history_is_publicly_serializable(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = UUID(
            await _seed_agent(
                session_maker,
                miner=_MINER_A,
                status=AgentStatus.QUARANTINED,
                screening_reason="Submission held for anti-cheat review",
                screening_policy_version=SCREENING_POLICY_VERSION,
            )
        )
        now = datetime.now(UTC)
        attempt_id = uuid4()
        private_finding = SourceReviewFinding(
            artifact_sha256="cd" * 32,
            prompt_revision="private-pending-review-v1",
            risk_level="high",
            confidence=0.99,
            categories=["answer_mutation"],
            evidence=[
                SourceReviewEvidenceItem(
                    path="src/private_innovation.rs",
                    line=41,
                    category="answer_mutation",
                )
            ],
            summary="Pending finding must remain private until a terminal rejection.",
        )
        async with session_maker() as session, session.begin():
            session.add_all(
                [
                    ScreeningAttempt(
                        attempt_id=attempt_id,
                        agent_id=agent_id,
                        screener_hotkey=_MINER_B,
                        policy_version=SCREENING_POLICY_VERSION,
                        status="quarantined",
                        started_at=now - timedelta(minutes=2),
                        deadline=now + timedelta(minutes=28),
                        finished_at=now,
                        public_reason="Submission held for anti-cheat review",
                    ),
                    ScreeningQuarantine(
                        quarantine_id=uuid4(),
                        agent_id=agent_id,
                        attempt_id=attempt_id,
                        screener_hotkey=_MINER_B,
                        policy_version=SCREENING_POLICY_VERSION,
                        manifest_digest="ab" * 32,
                        finding_digest=private_finding.canonical_digest(),
                        reason_code="suspicious_source",
                        evidence=[
                            {
                                "module_id": "agentic-source-review",
                                "code": "pending-private-review",
                                "summary": "Pending evidence remains private.",
                                "digest": None,
                            }
                        ],
                        finding=private_finding.model_dump(mode="json"),
                        status="active",
                    ),
                ]
            )
        _install_db(app, session_maker)

        response = await client.get(f"/api/v1/public/agent/{agent_id}/pipeline")

        assert response.status_code == 200
        attempt = response.json()["screening_attempts"][0]
        assert attempt["status"] == "quarantined"
        assert attempt["quarantine_resolution"] is None
        assert attempt["quarantine_resolved_at"] is None
        assert attempt["quarantine_resolution_reason"] is None
        assert attempt["review_evidence"] == []
        assert attempt["review_finding"] is None

    async def test_released_quarantine_resolution_is_public_in_attempt_history(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = UUID(
            await _seed_agent(
                session_maker,
                miner=_MINER_A,
                status=AgentStatus.EVALUATING,
                screening_reason="Manual review found no prohibited behavior",
                screening_policy_version=SCREENING_POLICY_VERSION,
            )
        )
        now = datetime.now(UTC)
        attempt_id = uuid4()
        async with session_maker() as session, session.begin():
            session.add_all(
                [
                    ScreeningAttempt(
                        attempt_id=attempt_id,
                        agent_id=agent_id,
                        screener_hotkey=_MINER_B,
                        policy_version=SCREENING_POLICY_VERSION,
                        status="quarantined",
                        started_at=now - timedelta(minutes=12),
                        deadline=now + timedelta(minutes=18),
                        finished_at=now - timedelta(minutes=10),
                        public_reason="Submission held for anti-cheat review",
                    ),
                    ScreeningQuarantine(
                        quarantine_id=uuid4(),
                        agent_id=agent_id,
                        attempt_id=attempt_id,
                        screener_hotkey=_MINER_B,
                        policy_version=SCREENING_POLICY_VERSION,
                        manifest_digest="ab" * 32,
                        reason_code="suspicious_source",
                        status="resolved",
                        resolved_at=now,
                        resolved_by="admin@example.com",
                        resolution="release",
                        resolution_reason=(
                            "Manual review found no prohibited behavior"
                        ),
                    ),
                ]
            )
        _install_db(app, session_maker)

        response = await client.get(f"/api/v1/public/agent/{agent_id}/pipeline")

        assert response.status_code == 200
        attempt = response.json()["screening_attempts"][0]
        assert attempt["status"] == "quarantined"
        assert attempt["quarantine_resolution"] == "release"
        assert datetime.fromisoformat(attempt["quarantine_resolved_at"]) == now.replace(
            tzinfo=None
        )
        assert attempt["quarantine_resolution_reason"] == (
            "Manual review found no prohibited behavior"
        )
        assert "resolved_by" not in attempt
        assert attempt["review_evidence"] == []
        assert attempt["review_finding"] is None

    async def test_rejected_quarantine_publishes_only_digest_verified_review(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = UUID(
            await _seed_agent(
                session_maker,
                miner=_MINER_A,
                status=AgentStatus.REJECTED,
                screening_reason="Submission violated the anti-cheat policy",
                screening_policy_version=SCREENING_POLICY_VERSION,
            )
        )
        now = datetime.now(UTC)
        attempt_id = uuid4()
        finding = SourceReviewFinding(
            artifact_sha256="cd" * 32,
            prompt_revision="public-safe-review-v1",
            risk_level="high",
            confidence=0.99,
            categories=["answer_mutation"],
            evidence=[
                SourceReviewEvidenceItem(
                    path="src/response.rs",
                    line=73,
                    category="answer_mutation",
                )
            ],
            summary=(
                "A reachable policy-controlled branch replaces the authoritative "
                "model answer before the response is returned."
            ),
        )
        async with session_maker() as session, session.begin():
            session.add_all(
                [
                    ScreeningAttempt(
                        attempt_id=attempt_id,
                        agent_id=agent_id,
                        screener_hotkey=_MINER_B,
                        policy_version=SCREENING_POLICY_VERSION,
                        status="quarantined",
                        started_at=now - timedelta(minutes=12),
                        deadline=now + timedelta(minutes=18),
                        finished_at=now - timedelta(minutes=10),
                        public_reason="Submission held for anti-cheat review",
                    ),
                    ScreeningQuarantine(
                        quarantine_id=uuid4(),
                        agent_id=agent_id,
                        attempt_id=attempt_id,
                        screener_hotkey=_MINER_B,
                        policy_version=SCREENING_POLICY_VERSION,
                        manifest_digest="ab" * 32,
                        finding_digest=finding.canonical_digest(),
                        reason_code="suspicious_source",
                        evidence=[
                            {
                                "module_id": "agentic-source-review",
                                "code": "answer-authority-violation",
                                "summary": (
                                    "The served response path replaces a model-"
                                    "authored answer with policy-controlled output."
                                ),
                                "digest": "ef" * 32,
                            }
                        ],
                        finding=finding.model_dump(mode="json"),
                        status="resolved",
                        resolved_at=now,
                        resolved_by="automation:screening-policy-v9",
                        resolution="reject",
                        resolution_reason="Verified prohibited answer replacement",
                    ),
                ]
            )
        _install_db(app, session_maker)

        response = await client.get(f"/api/v1/public/agent/{agent_id}/pipeline")

        assert response.status_code == 200
        attempt = response.json()["screening_attempts"][0]
        assert attempt["review_evidence"] == [
            {
                "module": "agentic-source-review",
                "code": "answer-authority-violation",
                "summary": (
                    "The served response path replaces a model-authored answer with "
                    "policy-controlled output."
                ),
            }
        ]
        assert attempt["review_finding"] == {
            "reviewer_revision": "public-safe-review-v1",
            "risk_level": "high",
            "confidence": 0.99,
            "categories": ["answer_mutation"],
            "locations": [
                {
                    "path": "src/response.rs",
                    "line": 73,
                    "category": "answer_mutation",
                }
            ],
            "summary": finding.summary,
        }
        assert "artifact_sha256" not in attempt["review_finding"]
        assert "digest" not in attempt["review_evidence"][0]

    async def test_evaluation_projects_live_work_from_validator_heartbeat(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = UUID(
            await _seed_agent(
                session_maker,
                miner=_MINER_A,
                status=AgentStatus.EVALUATING,
                screening_policy_version=SCREENING_POLICY_VERSION,
            )
        )
        _install_db(app, session_maker)

        waiting = (await client.get("/api/v1/public/activity")).json()["entries"][0]
        assert waiting["status"] == "waiting_validator"

        now = datetime.now(UTC)
        async with session_maker() as session, session.begin():
            session.add(
                ValidatorHeartbeat(
                    validator_hotkey=_MINER_B,
                    software_version="1.2.3",
                    protocol_version=2,
                    code_digest="ab" * 32,
                    state="running_benchmark",
                    active_agent_id=agent_id,
                    reported_at=now,
                    seen_at=now,
                    signature="cd" * 64,
                )
            )
            session.add(
                ValidatorTicket(
                    agent_id=agent_id,
                    validator_hotkey=_MINER_B,
                    status=TicketStatus.ISSUED,
                    issued_at=now - timedelta(seconds=1),
                    deadline=now + timedelta(minutes=30),
                )
            )

        evaluating = (await client.get("/api/v1/public/activity")).json()["entries"][0]
        assert evaluating["status"] == "evaluating"

    async def test_two_scores_below_top_five_bound_are_queued_for_completion(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_top_five_floor(session_maker, fifth_place=0.80)
        agent_id = UUID(
            await _seed_agent(
                session_maker,
                miner=_MINER_A,
                status=AgentStatus.EVALUATING,
                screening_policy_version=SCREENING_POLICY_VERSION,
            )
        )
        async with session_maker() as session, session.begin():
            for index, (validator, composite) in enumerate(
                ((_VALIDATOR_C, 0.10), (_MINER_B, 0.20))
            ):
                await upsert_score(
                    session,
                    agent_id=agent_id,
                    validator_hotkey=validator,
                    run_id=f"below-floor-{index}",
                    seed=42,
                    composite=composite,
                    tool_mean=composite,
                    memory_mean=composite,
                    median_ms=500,
                    n=114,
                    generated_at=datetime.now(UTC),
                    signature="ab" * 64,
                    details={
                        "per_case": [
                            {
                                "kind": "memory",
                                "category": "temporal_reasoning",
                                "score": composite,
                                "correct": False,
                                "latency_ms": 500,
                                "notes": ["answer did not match"],
                                "expected": "private answer key",
                                "called": ["private tool trace"],
                                "case_id": f"private-{index}",
                                "raw_response": "private response",
                            }
                        ]
                    },
                )
        _install_db(app, session_maker)

        entries = (await client.get("/api/v1/public/activity")).json()["entries"]
        activity = next(
            entry for entry in entries if entry["agent_id"] == str(agent_id)
        )
        assert activity["status"] == "below_score_floor"
        assert activity["score_count"] == 2
        assert activity["validator_queue_rank"] == 1

        pipeline = (
            await client.get(f"/api/v1/public/agent/{agent_id}/pipeline")
        ).json()
        assert pipeline["status"] == "below_score_floor"
        assert pipeline["score_count"] == 2
        assert pipeline["score_floor"] == pytest.approx(0.80)
        assert len(pipeline["provisional_scores"]) == 2
        case_results = [
            score["case_results"][0] for score in pipeline["provisional_scores"]
        ]
        assert {case["score"] for case in case_results} == {0.10, 0.20}
        for case in case_results:
            assert set(case) == {
                "category",
                "kind",
                "score",
                "correct",
                "latency_ms",
                "notes",
            }
            assert case["category"] == "temporal_reasoning"
            assert case["kind"] == "memory"
            assert case["correct"] is False
            assert case["latency_ms"] == 500
            assert case["notes"] == ["answer did not match"]
        for leaked in (
            '"expected"',
            '"called"',
            '"case_id"',
            '"raw_response"',
            "private answer key",
            "private tool trace",
            "private response",
        ):
            assert leaked not in json.dumps(pipeline)

        now = datetime.now(UTC)
        async with session_maker() as session, session.begin():
            session.add(
                ValidatorTicket(
                    agent_id=agent_id,
                    validator_hotkey=_MINER_A,
                    status=TicketStatus.ISSUED,
                    issued_at=now,
                    deadline=now + timedelta(minutes=30),
                )
            )

        entries = (await client.get("/api/v1/public/activity")).json()["entries"]
        activity = next(
            entry for entry in entries if entry["agent_id"] == str(agent_id)
        )
        assert activity["status"] == "waiting_validator"
        pipeline = (
            await client.get(f"/api/v1/public/agent/{agent_id}/pipeline")
        ).json()
        assert pipeline["status"] == "waiting_validator"

    async def test_public_progress_never_combines_benchmark_eras(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        """love-v8's v3 and v4 scores are one score in each era, not two."""
        await _seed_top_five_floor(session_maker, fifth_place=0.80, bench_version=4)
        agent_id = UUID(
            await _seed_agent(
                session_maker,
                miner=_MINER_A,
                status=AgentStatus.EVALUATING,
                name="love-v8",
                screening_policy_version=SCREENING_POLICY_VERSION,
            )
        )
        now = datetime.now(UTC)
        async with session_maker() as session, session.begin():
            session.add(
                BenchmarkRollout(
                    rollout_id=uuid4(),
                    from_version=2,
                    desired_version=4,
                    status="activated",
                    cohort_size=5,
                    created_at=now - timedelta(hours=1),
                    activated_at=now,
                )
            )
            for bench_version, validator, composite in (
                (3, _VALIDATOR_C, 0.391235),
                (4, _MINER_B, 0.391897),
            ):
                await upsert_score(
                    session,
                    agent_id=agent_id,
                    validator_hotkey=validator,
                    bench_version=bench_version,
                    run_id=f"love-v8-v{bench_version}",
                    seed=42,
                    composite=composite,
                    tool_mean=composite,
                    memory_mean=composite,
                    median_ms=500,
                    n=119,
                    generated_at=now,
                    signature="ab" * 64,
                    details={"bench_version": bench_version},
                )
        _install_db(app, session_maker)

        activity_body = (await client.get("/api/v1/public/activity")).json()
        activity = next(
            entry
            for entry in activity_body["entries"]
            if entry["agent_id"] == str(agent_id)
        )
        assert activity["status"] == "waiting_validator"
        assert activity["score_count"] == 1
        assert activity["provisional_composite"] == pytest.approx(0.391897)

        operations = (await client.get("/api/v1/public/operations")).json()
        operations_entry = next(
            entry
            for entry in operations["activity"]["entries"]
            if entry["agent_id"] == str(agent_id)
        )
        assert operations["active_bench_version"] == 4
        assert operations_entry["status"] == "waiting_validator"
        assert operations_entry["score_count"] == 1
        assert operations_entry["provisional_composite"] == pytest.approx(0.391897)

        pipeline = (
            await client.get(f"/api/v1/public/agent/{agent_id}/pipeline")
        ).json()
        assert pipeline["active_bench_version"] == 4
        assert pipeline["status"] == "waiting_validator"
        assert pipeline["score_count"] == 1
        assert pipeline["score_floor"] == pytest.approx(0.80)
        scores_by_version = {
            score["bench_version"]: score["composite"]
            for score in pipeline["provisional_scores"]
        }
        assert scores_by_version[3] == pytest.approx(0.391235)
        assert scores_by_version[4] == pytest.approx(0.391897)

    async def test_retry_state_surfaces_exhausted_and_cooling_submissions(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        """The public feed labels why a below-quorum submission is (not) advancing."""
        now = datetime.now(UTC)
        exhausted_id = UUID(
            await _seed_agent(
                session_maker,
                miner=_MINER_A,
                status=AgentStatus.EVALUATING,
                name="exhausted",
                screening_policy_version=SCREENING_POLICY_VERSION,
            )
        )
        cooling_id = UUID(
            await _seed_agent(
                session_maker,
                miner=_MINER_B,
                status=AgentStatus.EVALUATING,
                name="cooling",
                screening_policy_version=SCREENING_POLICY_VERSION,
            )
        )
        # A rejected submission with the exact same exhausted tickets must NOT be
        # labelled: retry_state is only meaningful while EVALUATING. (Regression
        # guard: the classifier once labelled every status.)
        rejected_id = UUID(
            await _seed_agent(
                session_maker,
                miner=_MINER_A,
                status=AgentStatus.REJECTED,
                name="rejected",
                screening_policy_version=SCREENING_POLICY_VERSION,
            )
        )
        cooldown_until = now + timedelta(hours=6)
        async with session_maker() as session, session.begin():
            for agent_id in (exhausted_id, rejected_id):
                for index in range(3):
                    session.add(
                        ValidatorTicket(
                            agent_id=agent_id,
                            validator_hotkey=f"validator-{index}",
                            status=TicketStatus.EXPIRED,
                            issued_at=now - timedelta(hours=3),
                            deadline=now - timedelta(hours=2, minutes=index),
                            bench_version=2,
                            attempt_count=2,
                            manual_retry_grants=0,
                            retry_after=now - timedelta(hours=1),
                        )
                    )
            session.add(
                ValidatorTicket(
                    agent_id=cooling_id,
                    validator_hotkey="validator-0",
                    status=TicketStatus.EXPIRED,
                    issued_at=now - timedelta(hours=1),
                    deadline=now - timedelta(minutes=30),
                    bench_version=2,
                    attempt_count=1,
                    manual_retry_grants=0,
                    retry_after=cooldown_until,
                )
            )
        _install_db(app, session_maker)

        by_id = {
            entry["agent_id"]: entry
            for entry in (await client.get("/api/v1/public/operations")).json()[
                "activity"
            ]["entries"]
        }
        assert by_id[str(exhausted_id)]["retry_state"] == "exhausted"
        assert by_id[str(exhausted_id)]["retry_after"] is None
        assert by_id[str(cooling_id)]["retry_state"] == "cooling_down"
        assert (
            datetime.fromisoformat(by_id[str(cooling_id)]["retry_after"])
            == cooldown_until
        )
        # Rejected (not EVALUATING): no retry_state, despite exhausted tickets.
        assert by_id[str(rejected_id)]["retry_state"] is None
        assert by_id[str(rejected_id)]["retry_after"] is None

    async def test_progress_is_multi_validator_allowlisted_and_recursively_redacted(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = UUID(
            await _seed_agent(
                session_maker,
                miner=_MINER_A,
                status=AgentStatus.EVALUATING,
                name="privacy-safe-agent",
                screening_policy_version=SCREENING_POLICY_VERSION,
            )
        )
        now = datetime.now(UTC)
        deadline = now + timedelta(minutes=30)
        safe_progress = {
            "stage": "running_benchmark",
            "completed": 51,
            "total": 114,
            "ticket_deadline": deadline.isoformat(),
        }
        sentinel = "PRIVATE_PROMPT_CANARY_DO_NOT_PUBLISH"
        async with session_maker() as session, session.begin():
            for hotkey, progress in (
                (_MINER_A, safe_progress),
                (_MINER_B, {**safe_progress, "completed": 3, "total": 8}),
                (_VALIDATOR_C, {**safe_progress, "prompt": sentinel}),
            ):
                session.add(
                    ValidatorHeartbeat(
                        validator_hotkey=hotkey,
                        software_version="1.2.3",
                        protocol_version=4,
                        code_digest="ab" * 32,
                        state="running_benchmark",
                        active_agent_id=agent_id,
                        benchmark_progress=progress,
                        benchmark_progress_reported=True,
                        reported_at=now,
                        seen_at=now,
                        signature="cd" * 64,
                    )
                )
                session.add(
                    ValidatorTicket(
                        agent_id=agent_id,
                        validator_hotkey=hotkey,
                        status=TicketStatus.ISSUED,
                        issued_at=now - timedelta(seconds=1),
                        deadline=deadline,
                    )
                )
        _install_db(app, session_maker)

        responses = [
            await client.get("/api/v1/public/validators"),
            await client.get("/api/v1/public/activity"),
            await client.get(f"/api/v1/public/agent/{agent_id}/pipeline"),
        ]
        assert all(response.status_code == 200 for response in responses)
        public_progress_keys = {
            "slot_id",
            "agent_id",
            "agent_name",
            "bench_version",
            "started_at",
            "stage",
            "completed_checks",
            "total_checks",
            "percent",
            "stalled",
        }
        fleet = responses[0].json()
        shown = [
            row["active_benchmark"]
            for row in fleet["validators"]
            if row["active_benchmark"] is not None
        ]
        assert len(shown) == 2
        assert all(set(progress) == public_progress_keys for progress in shown)
        first = next(
            progress for progress in shown if progress["completed_checks"] == 51
        )
        assert first["percent"] == 45
        assert first["bench_version"] == 2
        assert first["total_checks"] == 114
        assert datetime.fromisoformat(first["started_at"].replace("Z", "+00:00")) == (
            now - timedelta(seconds=1)
        )
        threshold = next(
            progress for progress in shown if progress["completed_checks"] == 3
        )
        assert threshold["percent"] == 40  # 3/8 = 37.5%, rounded half-up.
        activity = responses[1].json()["entries"][0]
        assert len(activity["active_benchmarks"]) == 2
        pipeline = responses[2].json()
        assert sum(a["actively_running"] for a in pipeline["validation_attempts"]) == 2
        assert all(a["bench_version"] == 2 for a in pipeline["validation_attempts"])

        forbidden_keys = {
            "case_id",
            "case_category",
            "prompt",
            "expected",
            "called",
            "tool_names",
            "memory_contents",
            "dataset",
            "dataset_sha256",
            "seed",
            "canary",
            "partial_score",
            "latency_ms",
            "model_output",
            "harness_logs",
            "tarball_logs",
            "run_id",
            "container_id",
            "filesystem_path",
            "ip_address",
            "error_body",
            "ticket_deadline",
        }

        def assert_redacted(value: object) -> None:
            if isinstance(value, dict):
                assert forbidden_keys.isdisjoint(value)
                for nested in value.values():
                    assert_redacted(nested)
            elif isinstance(value, list):
                for nested in value:
                    assert_redacted(nested)
            elif isinstance(value, str):
                assert sentinel not in value

        for response in responses:
            assert_redacted(response.json())

    async def test_live_work_marks_only_its_own_bench_version_attempt(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = UUID(
            await _seed_agent(
                session_maker,
                miner=_MINER_A,
                status=AgentStatus.EVALUATING,
                screening_policy_version=SCREENING_POLICY_VERSION,
            )
        )
        now = datetime.now(UTC)
        deadline = now + timedelta(minutes=30)
        async with session_maker() as session, session.begin():
            session.add(
                ValidatorHeartbeat(
                    validator_hotkey=_MINER_A,
                    software_version="1.2.3",
                    protocol_version=4,
                    code_digest="ab" * 32,
                    state="running_benchmark",
                    active_agent_id=agent_id,
                    benchmark_progress={
                        "stage": "running_benchmark",
                        "completed": 8,
                        "total": 119,
                        "ticket_deadline": deadline.isoformat(),
                    },
                    benchmark_progress_reported=True,
                    reported_at=now,
                    seen_at=now,
                    signature="cd" * 64,
                )
            )
            # The same validator already finished this agent on v2 and v3; only
            # the v4 ticket is live.
            for bench_version, status in (
                (2, TicketStatus.SCORED),
                (3, TicketStatus.SCORED),
                (4, TicketStatus.ISSUED),
            ):
                session.add(
                    ValidatorTicket(
                        agent_id=agent_id,
                        validator_hotkey=_MINER_A,
                        bench_version=bench_version,
                        status=status,
                        issued_at=now - timedelta(seconds=1),
                        deadline=deadline,
                    )
                )
        _install_db(app, session_maker)

        pipeline = (
            await client.get(f"/api/v1/public/agent/{agent_id}/pipeline")
        ).json()
        running = [
            attempt
            for attempt in pipeline["validation_attempts"]
            if attempt["actively_running"]
        ]
        assert [attempt["bench_version"] for attempt in running] == [4]
        assert running[0]["benchmark_progress"]["bench_version"] == 4
        assert all(
            attempt["benchmark_progress"] is None
            for attempt in pipeline["validation_attempts"]
            if attempt["bench_version"] != 4
        )

    async def test_delayed_legacy_or_omitted_progress_cannot_revive_reissued_work(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = UUID(
            await _seed_agent(
                session_maker,
                miner=_MINER_A,
                status=AgentStatus.EVALUATING,
                screening_policy_version=SCREENING_POLICY_VERSION,
            )
        )
        now = datetime.now(UTC)
        issued_at = now - timedelta(seconds=5)
        old_signed_at = now - timedelta(seconds=10)
        deadline = now + timedelta(minutes=30)
        async with session_maker() as session, session.begin():
            for hotkey, protocol_version in ((_MINER_A, 3), (_MINER_B, 4)):
                session.add(
                    ValidatorHeartbeat(
                        validator_hotkey=hotkey,
                        software_version="1.2.3",
                        protocol_version=protocol_version,
                        code_digest="ab" * 32,
                        state="running_benchmark",
                        active_agent_id=agent_id,
                        benchmark_progress=None,
                        benchmark_progress_reported=False,
                        reported_at=old_signed_at,
                        # Receipt after reissue must not make the old signature fresh.
                        seen_at=now,
                        signature="cd" * 64,
                    )
                )
                session.add(
                    ValidatorTicket(
                        agent_id=agent_id,
                        validator_hotkey=hotkey,
                        status=TicketStatus.ISSUED,
                        issued_at=issued_at,
                        deadline=deadline,
                    )
                )
        _install_db(app, session_maker)

        fleet = (await client.get("/api/v1/public/validators")).json()
        assert all(row["active_agent_id"] is None for row in fleet["validators"])
        activity = (await client.get("/api/v1/public/activity")).json()
        assert activity["entries"][0]["status"] == "waiting_validator"
        assert activity["entries"][0]["active_benchmarks"] == []

    async def test_respects_limit(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_agent(session_maker, miner=_MINER_A)
        await _seed_agent(session_maker, miner=_MINER_B)
        _install_db(app, session_maker)
        body = (await client.get("/api/v1/public/activity?limit=1")).json()
        assert body["count"] == 1
        assert body["total"] == 2
        assert body["total_pages"] == 2

    async def test_paginates_newest_first_without_overlap(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        for hour, name in ((10, "oldest"), (11, "middle"), (12, "newest")):
            await _seed_agent(
                session_maker,
                miner=_MINER_A,
                name=name,
                created_at=datetime(2026, 7, 13, hour, tzinfo=UTC),
            )
        _install_db(app, session_maker)

        first = (await client.get("/api/v1/public/activity?limit=2&page=1")).json()
        second = (await client.get("/api/v1/public/activity?limit=2&page=2")).json()

        assert [entry["name"] for entry in first["entries"]] == ["newest", "middle"]
        assert [entry["name"] for entry in second["entries"]] == ["oldest"]
        assert first["total"] == second["total"] == 3
        assert first["total_pages"] == second["total_pages"] == 2
        assert first["page"] == 1
        assert second["page"] == 2

    async def test_exposes_progress_count_with_partial_score(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.42],
            status=AgentStatus.EVALUATING,
        )
        _install_db(app, session_maker)

        resp = await client.get("/api/v1/public/activity")
        entry = resp.json()["entries"][0]
        assert entry["score_count"] == 1
        assert entry["quorum"] == 3
        assert entry["provisional_composite"] == pytest.approx(0.42)
        assert "signature" not in resp.text

    @pytest.mark.parametrize("score_count", [0, 1, 2, 3])
    async def test_pipeline_exposes_only_safe_accepted_scores_before_quorum(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
        score_count: int,
    ) -> None:
        composites = [0.41, 0.58, 0.73][:score_count]
        transcript_sha256 = "ef" * 32
        if score_count:
            agent_id = await _seed_k3(
                session_maker,
                miner=_MINER_A,
                composites=composites,
                status=(
                    AgentStatus.SCORED if score_count == 3 else AgentStatus.EVALUATING
                ),
                details={
                    "bench_version": 2,
                    "transcript_sha256": transcript_sha256,
                },
            )
        else:
            agent_id = await _seed_agent(
                session_maker,
                miner=_MINER_A,
                status=AgentStatus.EVALUATING,
                screening_policy_version=SCREENING_POLICY_VERSION,
            )
        _install_db(app, session_maker)

        response = await client.get(f"/api/v1/public/agent/{agent_id}/pipeline")

        assert response.status_code == 200
        body = response.json()
        assert body["score_count"] == score_count
        assert body["quorum"] == 3
        assert len(body["provisional_scores"]) == score_count
        assert body["final_composite"] == (
            pytest.approx(0.58) if score_count == 3 else None
        )
        assert sorted(score["composite"] for score in body["provisional_scores"]) == (
            composites
        )
        for score in body["provisional_scores"]:
            assert score["seed"] == "987654321"
            assert score["run_size"] == "full"
            assert score["bench_version"] == 2
            assert score["datagen_version"] == "v0.7.0"
            assert score["seed_source"] == "on_chain"
            assert score["dataset_sha256"] == "cd" * 32
            assert score["reproduction_command"] == (
                "go run github.com/ditto-assistant/dittobench-datagen/cmd/"
                "generate@v0.7.0 -seed 987654321 -run-size full -out dataset.json"
            )
            assert score["verification_command"].endswith(
                "-seed 987654321 -run-size full -sha"
            )
            # The signature-bound transcript digest is public; the offline
            # verification path depends on it.
            assert score["transcript_sha256"] == transcript_sha256
            assert "validator_hotkey" not in score
            assert "signature" not in score
            assert "ticket_deadline" not in score
            assert "run_id" not in score

    async def test_pipeline_labels_random_seed_fallback_without_block_provenance(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.52],
            status=AgentStatus.EVALUATING,
            dataset_seed_block=None,
            dataset_seed_block_hash=None,
            details={"bench_version": 2},
        )
        _install_db(app, session_maker)

        body = (await client.get(f"/api/v1/public/agent/{agent_id}/pipeline")).json()

        assert body["provisional_scores"][0]["seed_source"] == "random_fallback"

    async def test_pipeline_labels_validator_local_seed_without_pinned_dataset(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        """No pinned dataset at all (generation disabled when screened)."""
        agent_id = await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.52],
            status=AgentStatus.EVALUATING,
            dataset_seed=None,
            dataset_sha256=None,
            dataset_run_size=None,
            dataset_seed_block=None,
            dataset_seed_block_hash=None,
            details={"bench_version": 2},
        )
        _install_db(app, session_maker)

        body = (await client.get(f"/api/v1/public/agent/{agent_id}/pipeline")).json()

        score = body["provisional_scores"][0]
        assert score["seed_source"] == "validator_local"
        assert score["run_size"] is None
        assert score["dataset_sha256"] is None
        assert score["reproduction_command"] is None
        assert score["verification_command"] is None

    async def test_pipeline_keeps_accepted_score_visible_during_retry(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = UUID(
            await _seed_k3(
                session_maker,
                miner=_MINER_A,
                composites=[0.52],
                status=AgentStatus.EVALUATING,
                details={"bench_version": 2},
            )
        )
        now = datetime.now(UTC)
        async with session_maker() as session, session.begin():
            session.add(
                ValidatorTicket(
                    agent_id=agent_id,
                    validator_hotkey=_MINER_B,
                    status=TicketStatus.EXPIRED,
                    purpose=TicketPurpose.CANONICAL_QUORUM,
                    issued_at=now - timedelta(hours=2),
                    deadline=now - timedelta(hours=1),
                    failure_reason="sandbox_oom",
                    failed_at=now - timedelta(hours=1),
                )
            )
        _install_db(app, session_maker)

        body = (await client.get(f"/api/v1/public/agent/{agent_id}/pipeline")).json()

        assert body["score_count"] == 1
        assert body["provisional_scores"][0]["composite"] == pytest.approx(0.52)
        assert body["validation_attempts"][0]["status"] == "expired"
        assert body["validation_attempts"][0]["bench_version"] == 2
        assert body["validation_attempts"][0]["purpose"] == "canonical_quorum"
        assert body["validation_attempts"][0]["failure_reason"] == "sandbox_oom"
        assert body["validation_attempts"][0]["failed_at"] is not None

    async def test_pipeline_separates_canonical_quorum_from_continual_retests(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        from ditto.db.queries.confirmation_scores import (
            ConfirmationSeedScore,
            append_confirmation_scores,
        )

        agent_id = UUID(
            await _seed_k3(
                session_maker,
                miner=_MINER_A,
                composites=[0.91, 0.92, 0.93],
                status=AgentStatus.SCORED,
                details={"bench_version": 2},
            )
        )
        now = datetime.now(UTC)
        async with session_maker() as session, session.begin():
            canonical = list(
                await session.scalars(
                    select(Score)
                    .where(Score.agent_id == agent_id)
                    .order_by(Score.validator_hotkey)
                )
            )
            for score in canonical:
                score.created_at = now - timedelta(hours=1)
            completed_validator = canonical[0].validator_hotkey
            pending_validator = canonical[1].validator_hotkey
            replacement_validator = canonical[2].validator_hotkey
            await append_confirmation_scores(
                session,
                rows=[
                    ConfirmationSeedScore(
                        agent_id=agent_id,
                        validator_hotkey=completed_validator,
                        seed=111,
                        composite=0.94,
                        run_id="confirmation-run",
                        signature="ab" * 64,
                    ),
                    ConfirmationSeedScore(
                        agent_id=agent_id,
                        validator_hotkey=completed_validator,
                        seed=222,
                        composite=0.95,
                        run_id="confirmation-run",
                        signature="ab" * 64,
                    ),
                ],
                bench_version=2,
                created_at=now,
            )
            session.add_all(
                [
                    ValidatorTicket(
                        agent_id=agent_id,
                        validator_hotkey=completed_validator,
                        status=TicketStatus.SCORED,
                        purpose=TicketPurpose.CONTINUAL_RETEST,
                        issued_at=now - timedelta(minutes=10),
                        deadline=now - timedelta(minutes=5),
                        bench_version=2,
                    ),
                    ValidatorTicket(
                        agent_id=agent_id,
                        validator_hotkey=pending_validator,
                        status=TicketStatus.ISSUED,
                        purpose=TicketPurpose.CONTINUAL_RETEST,
                        issued_at=now,
                        deadline=now + timedelta(minutes=30),
                        bench_version=2,
                    ),
                    ValidatorTicket(
                        agent_id=agent_id,
                        validator_hotkey=replacement_validator,
                        status=TicketStatus.ISSUED,
                        purpose=TicketPurpose.CANONICAL_QUORUM,
                        issued_at=now,
                        deadline=now + timedelta(minutes=30),
                        bench_version=2,
                    ),
                ]
            )
        _install_db(app, session_maker)

        body = (await client.get(f"/api/v1/public/agent/{agent_id}/pipeline")).json()

        assert body["score_count"] == body["quorum"] == 3
        assert len(body["provisional_scores"]) == 3
        assert [
            (score["seed"], score["composite"]) for score in body["confirmation_scores"]
        ] == [
            ("111", pytest.approx(0.94)),
            ("222", pytest.approx(0.95)),
        ]
        assert all("run_id" not in score for score in body["confirmation_scores"])
        assert {
            attempt["validator_hotkey"]: attempt["purpose"]
            for attempt in body["validation_attempts"]
        } == {
            completed_validator: "continual_retest",
            pending_validator: "continual_retest",
            replacement_validator: "canonical_quorum",
        }

    async def test_pipeline_keeps_mixed_benchmark_quorums_separate(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = UUID(
            await _seed_k3(
                session_maker,
                miner=_MINER_A,
                composites=[0.41, 0.58, 0.73],
                status=AgentStatus.SCORED,
                details={"bench_version": 2},
            )
        )
        now = datetime.now(UTC)
        async with session_maker() as session, session.begin():
            await upsert_score(
                session,
                agent_id=agent_id,
                validator_hotkey=_VALIDATOR_C,
                run_id="v3-run",
                seed=123,
                composite=0.91,
                tool_mean=0.91,
                memory_mean=0.91,
                median_ms=400,
                n=114,
                generated_at=now,
                signature="ab" * 64,
                details={"bench_version": 3},
                bench_version=3,
            )
            session.add(
                ValidatorTicket(
                    agent_id=agent_id,
                    validator_hotkey=_MINER_A,
                    status=TicketStatus.ISSUED,
                    issued_at=now,
                    deadline=now + timedelta(hours=1),
                    bench_version=3,
                )
            )
        _install_db(app, session_maker)

        response = await client.get(f"/api/v1/public/agent/{agent_id}/pipeline")

        assert response.status_code == 200
        body = response.json()
        assert body["active_bench_version"] == 2
        assert body["score_count"] == 3
        assert body["final_composite"] == pytest.approx(0.58)
        assert [score["bench_version"] for score in body["provisional_scores"]].count(
            2
        ) == 3
        assert [score["bench_version"] for score in body["provisional_scores"]].count(
            3
        ) == 1
        assert body["validation_attempts"][0]["bench_version"] == 3

    @pytest.mark.parametrize(
        "status",
        [AgentStatus.SCREENING, AgentStatus.QUARANTINED, AgentStatus.REJECTED],
    )
    async def test_pipeline_preserves_scores_without_finalizing_screening_states(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
        status: AgentStatus,
    ) -> None:
        agent_id = await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.41, 0.58, 0.73],
            status=status,
            details={"bench_version": 2},
        )
        _install_db(app, session_maker)

        body = (await client.get(f"/api/v1/public/agent/{agent_id}/pipeline")).json()

        assert body["score_count"] == 3
        assert len(body["provisional_scores"]) == 3
        assert body["final_composite"] is None


class TestPublicSubmissionScores:
    async def test_detail_exposes_k3_breakdown_and_median(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_k3(
            session_maker, miner=_MINER_A, composites=[0.40, 0.70, 0.55]
        )
        _install_db(app, session_maker)

        resp = await client.get(f"/api/v1/public/agent/{agent_id}/scores")
        assert resp.status_code == 200
        assert resp.headers["Cache-Control"] == "public, max-age=30"
        body = resp.json()
        assert body["agent_id"] == agent_id
        assert body["miner_hotkey"] == _MINER_A
        assert body["status"] == "scored"
        assert body["quorum"] == 3
        assert body["score_count"] == 3
        # Median of {0.40, 0.55, 0.70} is 0.55 — no single validator controls it.
        assert body["median_composite"] == pytest.approx(0.55)
        # The dataset pin + raw seed are published for reproduction/audit.
        assert body["dataset_seed"] == 987654321
        assert body["dataset_sha256"] == "cd" * 32
        assert body["dataset_run_size"] == "full"
        # The on-chain seed provenance lets anyone verify the seed was not
        # platform-chosen (recompute derive_seed(block_hash, agent_id)).
        assert body["dataset_seed_block"] == 4321
        assert body["dataset_seed_block_hash"] == "0x" + "9f" * 32
        # All three validators, each with hotkey + signature (self-verifying).
        assert len(body["scores"]) == 3
        hotkeys = {s["validator_hotkey"] for s in body["scores"]}
        assert len(hotkeys) == 3
        for s in body["scores"]:
            assert s["signature"] == "ab" * 64
            assert s["seed"] == 987654321
            assert "run_id" in s
            # Scores recorded before lease-bound signing remain public and
            # continue counting; null identifies their legacy signature format.
            assert s["ticket_deadline"] is None
            # No bench_version in details → published as null (legacy), never
            # guessed from the column default.
            assert s["bench_version"] is None

    async def test_detail_labels_each_score_with_its_bench_version(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # A re-scored agent carries rows from more than one benchmark version;
        # each published row names the version it was scored under so its
        # incomparable composites cannot be read as one pool.
        agent_id = await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.40, 0.70, 0.55],
            details={"bench_version": 2},
        )
        async with session_maker() as s, s.begin():
            await upsert_score(
                s,
                agent_id=UUID(agent_id),
                validator_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                run_id="run_v3",
                seed=987654321,
                composite=0.61,
                tool_mean=0.61,
                memory_mean=0.61,
                median_ms=500,
                n=110,
                generated_at=datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC),
                signature="ab" * 64,
                details={"bench_version": 3},
                bench_version=3,
            )
        _install_db(app, session_maker)

        body = (await client.get(f"/api/v1/public/agent/{agent_id}/scores")).json()

        assert body["score_count"] == 4
        assert sorted(s["bench_version"] for s in body["scores"]) == [2, 2, 2, 3]

    async def test_detail_exposes_redacted_per_case_breakdown(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # Per-validator per-case breakdown (where points were won/lost) is served,
        # redacted: category/kind/score/pass/latency/notes but never the answer key.
        details = {
            "per_case": [
                {
                    "kind": "tool",
                    "category": "web_search",
                    "score": 0.6,
                    "correct": False,
                    "latency_ms": 3382,
                    "notes": ["1 extra/unexpected tool call(s)"],
                    "expected": ["search_web"],
                    "called": ["search_web", "search_web"],
                    "case_id": "web_search-8860569897825046057-0001",
                },
            ],
        }
        agent_id = await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.4, 0.5, 0.6],
            details=details,
        )
        _install_db(app, session_maker)
        resp = await client.get(f"/api/v1/public/agent/{agent_id}/scores")
        body = resp.json()
        cases = body["scores"][0]["case_results"]
        assert cases and cases[0]["category"] == "web_search"
        assert cases[0]["score"] == pytest.approx(0.6)
        assert cases[0]["correct"] is False
        assert set(cases[0]).issubset(
            {"category", "kind", "score", "correct", "latency_ms", "notes"}
        )
        # The answer key never appears anywhere in the response.
        for leaked in ('"expected"', '"called"', '"case_id"'):
            assert leaked not in resp.text

    async def test_detail_omits_per_case_answer_key(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_k3(
            session_maker, miner=_MINER_A, composites=[0.4, 0.5, 0.6]
        )
        _install_db(app, session_maker)
        raw = (await client.get(f"/api/v1/public/agent/{agent_id}/scores")).text
        # The per-submission record publishes validators + seed by design, but
        # still never the per-case answer key.
        for answer_key in ('"expected"', '"called"', '"case_id"', '"per_case"'):
            assert answer_key not in raw

    async def test_detail_404_for_unknown_agent(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        resp = await client.get(f"/api/v1/public/agent/{uuid4()}/scores")
        assert resp.status_code == 404

    async def test_detail_404_for_provisional_agent(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # A still-evaluating agent's partial scores must not be exposed.
        agent_id = await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.4],
            status=AgentStatus.EVALUATING,
        )
        # ...nor a held (suspected-copy) agent's.
        held_id = await _seed_k3(
            session_maker,
            miner=_MINER_B,
            composites=[0.9, 0.9, 0.9],
            status=AgentStatus.ATH_PENDING_REVIEW,
        )
        _install_db(app, session_maker)
        assert (
            await client.get(f"/api/v1/public/agent/{agent_id}/scores")
        ).status_code == 404
        assert (
            await client.get(f"/api/v1/public/agent/{held_id}/scores")
        ).status_code == 404

    async def test_index_lists_recent_finalized_only(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.4, 0.5, 0.6],
            base_time=datetime(2026, 6, 8, 10, 0, 0, tzinfo=UTC),
        )
        await _seed_k3(
            session_maker,
            miner=_MINER_B,
            composites=[0.7, 0.8, 0.9],
            base_time=datetime(2026, 6, 8, 14, 0, 0, tzinfo=UTC),
        )
        # Held + still-evaluating must be excluded from the index.
        await _seed_k3(
            session_maker,
            miner="5HeldMinerXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            composites=[0.99, 0.99, 0.99],
            status=AgentStatus.ATH_PENDING_REVIEW,
        )
        _install_db(app, session_maker)

        body = (await client.get("/api/v1/public/submissions")).json()
        assert body["count"] == 2
        assert body["quorum"] == 3
        # Most recently scored first: MINER_B (14:00) before MINER_A (10:00).
        assert [s["miner_hotkey"] for s in body["submissions"]] == [_MINER_B, _MINER_A]
        top = body["submissions"][0]
        assert top["median_composite"] == pytest.approx(0.8)
        assert top["score_count"] == 3
        assert top["dataset_seed"] == 987654321
        assert top["last_scored_at"] is not None

    async def test_index_respects_limit(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        for i in range(3):
            await _seed_k3(
                session_maker,
                miner=_MINER_A,
                composites=[0.4, 0.5, 0.6],
                base_time=datetime(2026, 6, 8, 10 + i, 0, 0, tzinfo=UTC),
            )
        _install_db(app, session_maker)
        body = (await client.get("/api/v1/public/submissions?limit=2")).json()
        assert body["count"] == 2

    async def test_index_empty(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        resp = await client.get("/api/v1/public/submissions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 0
        assert body["submissions"] == []


async def _seed_audit(maker: async_sessionmaker[AsyncSession], *, n: int) -> None:
    """Append ``n`` chained score entries to the audit log."""
    async with maker() as s, s.begin():
        for i in range(n):
            await append_audit_entry(
                s,
                agent_id=uuid4(),
                validator_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                event=EVENT_SCORE,
                payload={"run_id": f"run_{i}", "composite": 0.5, "seed": 42},
                recorded_at=datetime(2026, 6, 8, 12, i, 0, tzinfo=UTC),
            )


class _FakeRevealGenerator:
    """Stands in for the data-pipeline generate service on the reveal path."""

    def __init__(
        self,
        *,
        artifact: dict | None = None,
        sha: str = "cd" * 32,
        fail: bool = False,
    ) -> None:
        self._artifact = artifact if artifact is not None else {"bench_version": 2}
        self._sha = sha
        self._fail = fail
        self.calls = 0

    async def fetch_dataset(self, seed: int, run_size: str) -> tuple[dict, str]:
        self.calls += 1
        if self._fail:
            raise DataPipelineError("generate service down")
        return {**self._artifact, "seed": seed, "run_size": run_size}, self._sha


def _install_generator(app: FastAPI, generator: object) -> None:
    async def _gen() -> object:
        return generator

    app.dependency_overrides[get_dataset_generator] = _gen


class TestPublicDatasetReveal:
    async def test_reveals_full_labeled_dataset_for_finalized_agent(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_k3(
            session_maker, miner=_MINER_A, composites=[0.4, 0.5, 0.6]
        )
        _install_db(app, session_maker)
        # The generator returns a dataset whose sha matches the pinned "cd"*32.
        artifact = {"bench_version": 2, "tool_cases": [{"expected_tools": ["x"]}]}
        gen = _FakeRevealGenerator(artifact=artifact, sha="cd" * 32)
        _install_generator(app, gen)

        resp = await client.get(f"/api/v1/public/agent/{agent_id}/dataset")
        assert resp.status_code == 200
        body = resp.json()
        assert body["agent_id"] == agent_id
        assert body["seed"] == 987654321
        assert body["run_size"] == "full"
        assert body["dataset_sha256"] == "cd" * 32
        assert body["bench_version"] == 2
        # The FULL labeled artifact (answer keys included) is served.
        assert body["artifact"]["tool_cases"][0]["expected_tools"] == ["x"]
        assert gen.calls == 1

    async def test_404_for_unfinalized_agent(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.4],
            status=AgentStatus.EVALUATING,
        )
        _install_db(app, session_maker)
        _install_generator(app, _FakeRevealGenerator())
        resp = await client.get(f"/api/v1/public/agent/{agent_id}/dataset")
        assert resp.status_code == 404

    async def test_502_on_generator_hash_drift(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_k3(
            session_maker, miner=_MINER_A, composites=[0.4, 0.5, 0.6]
        )
        _install_db(app, session_maker)
        # Generator returns a DIFFERENT sha than the pinned "cd"*32.
        _install_generator(app, _FakeRevealGenerator(sha="ab" * 32))
        resp = await client.get(f"/api/v1/public/agent/{agent_id}/dataset")
        assert resp.status_code == 502

    async def test_503_when_generator_unavailable(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_k3(
            session_maker, miner=_MINER_A, composites=[0.4, 0.5, 0.6]
        )
        _install_db(app, session_maker)
        _install_generator(app, _FakeRevealGenerator(fail=True))
        resp = await client.get(f"/api/v1/public/agent/{agent_id}/dataset")
        assert resp.status_code == 503


class TestPublicBenchCorpus:
    async def test_retired_version_serves_full_answer_keys(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # A run scored under the retired v1 (current is 2). Its full per-case
        # answer keys are released verbatim.
        details = {
            "bench_version": 1,
            "per_case": [
                {
                    "category": "web_search",
                    "score": 0.6,
                    "expected": ["search_web"],
                    "called": ["search_web"],
                    "case_id": "web_search-1-0001",
                }
            ],
        }
        await _seed_k3(
            session_maker, miner=_MINER_A, composites=[0.4, 0.5, 0.6], details=details
        )
        _install_db(app, session_maker)

        resp = await client.get("/api/v1/public/bench/1/corpus")
        assert resp.status_code == 200
        body = resp.json()
        assert body["bench_version"] == 1
        assert body["total"] == 3  # three validator rows
        entry = body["entries"][0]
        # The FULL answer key is present (retired = safe).
        assert entry["per_case"][0]["expected"] == ["search_web"]
        assert entry["per_case"][0]["case_id"] == "web_search-1-0001"

    async def test_live_version_is_refused(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # The current (live) version: its answer keys must never be released.
        await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.4, 0.5, 0.6],
            details={
                "bench_version": CURRENT_BENCH_VERSION,
                "per_case": [{"expected": ["x"]}],
            },
        )
        _install_db(app, session_maker)
        resp = await client.get(f"/api/v1/public/bench/{CURRENT_BENCH_VERSION}/corpus")
        assert resp.status_code == 409
        # ...and the live answer key is not in the refusal body.
        assert '"expected"' not in resp.text

    async def test_v2_corpus_remains_private_before_v3_activation(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.4, 0.5, 0.6],
            details={
                "bench_version": 2,
                "per_case": [{"expected": ["still-live"]}],
            },
        )
        _install_db(app, session_maker)

        response = await client.get("/api/v1/public/bench/2/corpus")

        assert response.status_code == 409
        assert '"expected"' not in response.text

    async def test_retired_version_paginates(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.4, 0.5, 0.6],
            details={"bench_version": 1, "per_case": []},
        )
        _install_db(app, session_maker)
        page = (await client.get("/api/v1/public/bench/1/corpus?limit=2")).json()
        assert page["count"] == 2
        assert page["total"] == 3
        page2 = (
            await client.get("/api/v1/public/bench/1/corpus?limit=2&offset=2")
        ).json()
        assert page2["count"] == 1

    async def test_retired_version_with_no_runs_is_empty(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        body = (await client.get("/api/v1/public/bench/1/corpus")).json()
        assert body["total"] == 0
        assert body["entries"] == []


class TestPublicAudit:
    async def test_feed_returns_chained_entries(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_audit(session_maker, n=3)
        _install_db(app, session_maker)

        resp = await client.get("/api/v1/public/audit")
        assert resp.status_code == 200
        assert resp.headers["Cache-Control"] == "public, max-age=30"
        body = resp.json()
        assert body["count"] == 3
        assert body["genesis_hash"] == GENESIS_HASH
        entries = body["entries"]
        # Oldest first, contiguous seqs, and each links to the prior entry_hash.
        assert [e["seq"] for e in entries] == sorted(e["seq"] for e in entries)
        assert entries[0]["prev_hash"] == GENESIS_HASH
        for prev, cur in zip(entries, entries[1:], strict=False):
            assert cur["prev_hash"] == prev["entry_hash"]
        assert body["head_hash"] == entries[-1]["entry_hash"]
        # The signed-tuple payload is present; no per-case answer key ever is.
        assert entries[0]["payload"]["run_id"] == "run_0"
        assert '"per_case"' not in resp.text

    async def test_feed_pages_by_since_seq(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_audit(session_maker, n=5)
        _install_db(app, session_maker)

        first = (await client.get("/api/v1/public/audit?limit=2")).json()
        assert first["count"] == 2
        last_seq = first["entries"][-1]["seq"]
        nxt = (await client.get(f"/api/v1/public/audit?since_seq={last_seq}")).json()
        assert nxt["count"] == 3
        assert nxt["entries"][0]["seq"] > last_seq
        # The page still links onto the first page's head.
        assert nxt["entries"][0]["prev_hash"] == first["head_hash"]

    async def test_feed_empty(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        body = (await client.get("/api/v1/public/audit")).json()
        assert body["count"] == 0
        assert body["entries"] == []
        assert body["head_hash"] is None
        assert body["genesis_hash"] == GENESIS_HASH


class TestBenchConfig:
    """GET /public/bench/config exposes the frozen-model + grading setup."""

    async def test_config_shape_and_defaults(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
        monkeypatch,
    ) -> None:
        _install_db(app, session_maker)
        monkeypatch.delenv("STORAGE_PUBLIC_BUCKET", raising=False)
        resp = await client.get("/api/v1/public/bench/config")
        assert resp.status_code == 200
        assert "max-age=300" in resp.headers["Cache-Control"]
        body = resp.json()
        assert body["bench_version"] == DEFAULT_BENCH_VERSION
        h = body["harness"]
        assert h["locked"] is True
        assert h["canonical_id"] == "qwen/qwen3-32b"
        assert h["serving"] == "Qwen/Qwen3-32B-TEE"
        assert h["thinking"] is False
        assert h["reasoning_effort"] is None
        assert body["grading"]["judge_free"] is True
        assert "dittobench-datagen" in body["grading"]["grader"]
        assert "dataset_sha256" in body["dataset"]["reproduce"]
        assert body["public_mirror_url_template"] is None
        assert body["public_transcript_url_template"] is None
        assert body["public_transcript_telemetry_url_template"] == (
            "/api/v1/public/bench/transcript/{sha256}/telemetry"
        )
        assert body["ledger_path"] == "/api/v1/scoring/scores"

    async def test_open_v7_rollout_keeps_active_v6_harness_authoritative(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
        monkeypatch,
    ) -> None:
        _install_db(app, session_maker)
        monkeypatch.delenv("BENCH_HARNESS_MODEL_ID", raising=False)
        monkeypatch.delenv("BENCH_HARNESS_SERVING", raising=False)
        async with session_maker() as session, session.begin():
            session.add(
                BenchmarkRollout(
                    rollout_id=uuid4(),
                    from_version=DEFAULT_BENCH_VERSION,
                    desired_version=7,
                    status="collecting",
                    cohort_size=5,
                    created_at=datetime.now(UTC),
                )
            )

        body = (await client.get("/api/v1/public/bench/config")).json()

        assert body["bench_version"] == DEFAULT_BENCH_VERSION
        assert body["desired_bench_version"] == 7
        assert body["harness"]["canonical_id"] == "qwen/qwen3-32b"
        assert body["harness"]["serving"] == "Qwen/Qwen3-32B-TEE"
        assert body["harness"]["thinking"] is False
        assert body["harness"]["reasoning_effort"] is None

    async def test_mirror_template_from_env(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
        monkeypatch,
    ) -> None:
        _install_db(app, session_maker)
        monkeypatch.setenv("STORAGE_PUBLIC_BUCKET", "ditto-platform-public-dev")
        body = (await client.get("/api/v1/public/bench/config")).json()
        assert body["public_mirror_url_template"] == (
            "https://storage.googleapis.com/ditto-platform-public-dev/scored/{agent_id}.json"
        )
        assert body["public_transcript_url_template"] == (
            "https://storage.googleapis.com/ditto-platform-public-dev/transcripts/{sha256}.json"
        )
        assert body["public_transcript_telemetry_url_template"] == (
            "/api/v1/public/bench/transcript/{sha256}/telemetry"
        )

    async def test_transcript_telemetry_is_verified_allowlisted_and_immutable(
        self, app: FastAPI, client: httpx.AsyncClient
    ) -> None:
        body = json.dumps(
            {
                "execution": {"cases": 1, "succeeded": 1, "max_duration_ms": 25},
                "model_relay": {"requests": 2, "successes": 1},
                "cases": [
                    {
                        "prompt": "private question",
                        "response": "private answer",
                        "execution": {
                            "total_duration_ms": 25,
                            "terminal_outcome": "success",
                            "attempts": [
                                {
                                    "attempt": 1,
                                    "duration_ms": 25,
                                    "outcome": "success",
                                    "http_status": 200,
                                    "error": "private raw error",
                                }
                            ],
                        },
                    }
                ],
            },
            separators=(",", ":"),
        ).encode()
        digest = hashlib.sha256(body).hexdigest()
        storage = AsyncMock()
        storage.get_object.return_value = body

        async def _storage():
            return storage

        app.dependency_overrides[get_storage_client] = _storage
        response = await client.get(
            f"/api/v1/public/bench/transcript/{digest}/telemetry"
        )

        assert response.status_code == 200
        assert response.json() == {
            "source_sha256": digest,
            "execution": {
                "cases": 1,
                "succeeded": 1,
                "timed_out": 0,
                "cancelled": 0,
                "retried": 0,
                "total_attempts": 0,
                "median_duration_ms": None,
                "p95_duration_ms": None,
                "max_duration_ms": 25,
            },
            "model_relay": {
                "requests": 2,
                "successes": 1,
                "infrastructure_failures": 0,
                "caller_cancellations": 0,
                "upstream_attempts": 0,
                "retries": 0,
            },
            "cases": [
                {
                    "position": 1,
                    "total_duration_ms": 25,
                    "terminal_outcome": "success",
                    "timed_out": False,
                    "cancelled": False,
                    "attempts": [
                        {
                            "attempt": 1,
                            "duration_ms": 25,
                            "outcome": "success",
                            "http_status": 200,
                        }
                    ],
                }
            ],
        }
        assert "private" not in response.text
        assert "immutable" in response.headers["cache-control"]
        storage.get_object.assert_awaited_once_with(
            key=f"transcripts/{digest}.json", max_bytes=32 << 20
        )

    async def test_transcript_rejects_bad_address_and_stored_digest(
        self, app: FastAPI, client: httpx.AsyncClient
    ) -> None:
        storage = AsyncMock()

        async def _storage():
            return storage

        app.dependency_overrides[get_storage_client] = _storage
        assert (
            await client.get("/api/v1/public/bench/transcript/not-a-digest/telemetry")
        ).status_code == 404
        storage.get_object.assert_not_awaited()

        expected = "0" * 64
        storage.get_object.return_value = b"{}"
        response = await client.get(
            f"/api/v1/public/bench/transcript/{expected}/telemetry"
        )
        assert response.status_code == 502

    async def test_transcript_missing_is_not_publicly_distinguishable(
        self, app: FastAPI, client: httpx.AsyncClient
    ) -> None:
        storage = AsyncMock()
        storage.get_object.side_effect = ObjectDownloadFailedError("missing")

        async def _storage():
            return storage

        app.dependency_overrides[get_storage_client] = _storage
        response = await client.get(
            "/api/v1/public/bench/transcript/" + "a" * 64 + "/telemetry"
        )
        assert response.status_code == 404


def test_bench_glossary_explains_every_v5_category_and_metric() -> None:
    from ditto.api_models import bench_glossary as bg

    cats = {c["key"]: c for c in bg.category_entries()}
    # The v5 families the composite quality gate hinges on must be documented.
    for key in (
        "conversational-chitchat",
        "conversational-declarative",
        "declarative-write",
        "declarative-write-read",
        "declarative-behavior",
        "multi-hop-relational",
        "temporal-depth",
        "canary",
        # bench_version 6 complexity classes
        "injection-stored-instruction",
        "stored-instruction-benign",
        "multi-query-recall",
        "nonverbatim-computed",
        "passive-consolidation",
    ):
        assert key in cats, f"undocumented category: {key}"
    # Every entry is complete and public-safe (a purpose, a known kind, no blanks),
    # and carries a concrete illustrative example so the glossary shows what each
    # case actually looks like, not just what it probes.
    kinds = {"memory", "conversational", "tool", "multi_step", "integrity"}
    for c in cats.values():
        assert c["label"] and c["purpose"]
        assert c["kind"] in kinds
        assert c["example"], f"category missing example: {c['key']}"
    # The metrics / quality factors that pull the composite below the halves.
    metrics = {m["key"] for m in bg.metric_entries()}
    # bench_version changelog is present, newest first, complete per version.
    versions = bg.version_entries()
    assert [v["version"] for v in versions] == [7, 6, 5, 4, 3, 2]
    for v in versions:
        assert v["title"] and v["summary"] and v["epoch"]

    v7 = versions[0]
    assert v7["title"] == "GPT-OSS inference contract"
    assert "openai/gpt-oss-20b" in v7["summary"]
    assert "medium" in v7["summary"]
    assert any("Same generated questions" in item for item in v7["highlights"])

    for key in (
        "composite",
        "conversational_sanity",
        "metamorphic_consistency",
        "tool_efficiency",
        "token_efficiency",
    ):
        assert key in metrics, f"undocumented metric: {key}"
