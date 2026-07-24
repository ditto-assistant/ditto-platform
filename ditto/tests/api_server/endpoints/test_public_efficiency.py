"""End-to-end tests for the relative token-efficiency bonus (bench_version 7).

Covers the platform-layer contract from ``docs/relative-efficiency-bonus.md``:
the leaderboard materializes an epoch-frozen cohort snapshot, assigns
insert-once bonuses against its robust reference, exposes base composite /
bonus / effective composite distinctly, honors the N_min activation gate, and
leaves every bench_version < 7 board byte-identical (no snapshot, no bonus
row, null fields). The validator-facing ledger exposes effective_composite
only behind the default-off fold flag.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import bittensor
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
from ditto.api_server import EfficiencyBonusConfig, create_api_server
from ditto.api_server.dependencies import get_chain_client, get_session
from ditto.api_server.efficiency import ensure_efficiency_state
from ditto.chain.models import NeuronInfo
from ditto.db.models import (
    Agent,
    Base,
    BenchmarkRollout,
    EfficiencyBonus,
    EfficiencyCohortSnapshot,
)
from ditto.db.queries.scores import upsert_score
from ditto.tests.api_server.conftest import make_api_server_config

_VALIDATORS = [
    "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
    "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
    "5DhaT8U7LVwnnJNUU8VL1XEipicatoaDVVq7cHo227gogVZm",
]
_KEYPAIR = bittensor.Keypair.create_from_uri("//Alice")
_VALIDATOR_HOTKEY = _KEYPAIR.ss58_address
_T0 = datetime(2026, 7, 24, 12, 0, 0, tzinfo=UTC)

_MINERS = [
    "5DhaT8U7LVwnnJNUU8VL1XEipicatoaDVVq7cHo227gogVZm",
    "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
    "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
    "5CZq6MdanxF3j8ACp8oVtiaphTeyrA7QFPU92ke2jEFzK1mp",
    "5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw",
]


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


def _make_app(
    maker: async_sessionmaker[AsyncSession],
    *,
    efficiency: EfficiencyBonusConfig,
) -> FastAPI:
    import os

    # These tests assert mutate-then-refetch sequences; the public TTL cache
    # would serve the stale first body (it has its own middleware coverage).
    os.environ["PUBLIC_CACHE_DISABLED"] = "1"
    app = create_api_server(make_api_server_config(efficiency_bonus=efficiency))
    app.state.commit_hash = "test-commit"

    async def _session() -> AsyncIterator[AsyncSession]:
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = _session
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )


_ENABLED = EfficiencyBonusConfig(
    enabled=True,
    cap=0.05,
    cohort_size=25,
    min_cohort=3,
    epoch_hours=24,
)


async def _activate_bench_version(
    maker: async_sessionmaker[AsyncSession], version: int
) -> None:
    async with maker() as s, s.begin():
        s.add(
            BenchmarkRollout(
                rollout_id=uuid4(),
                from_version=2,
                desired_version=version,
                status="activated",
                cohort_size=5,
                activated_at=_T0,
            )
        )


def _details(total_tokens: int, *, bench_version: int = 7) -> dict:
    return {
        "bench_version": bench_version,
        "token_usage": {
            "status": "complete",
            "total_tokens": total_tokens,
            "usage_unavailable": 0,
        },
        "token_efficiency": {"formula_version": "v7-quality-only-v1"},
    }


async def _seed_finalized(
    maker: async_sessionmaker[AsyncSession],
    *,
    miner: str,
    composite: float,
    total_tokens: int,
    bench_version: int = 7,
    memory_mean: float | None = None,
    sha256: str | None = None,
    normalized_source_hash: str | None = None,
    created_at: datetime | None = None,
) -> UUID:
    """One agent with a full k=3 quorum of identical v7 scores + audited usage."""
    agent_id = uuid4()
    async with maker() as s, s.begin():
        s.add(
            Agent(
                agent_id=agent_id,
                miner_hotkey=miner,
                name="agent",
                sha256=sha256 or agent_id.hex * 2,
                normalized_source_hash=normalized_source_hash,
                size_bytes=524288,
                status=AgentStatus.SCORED,
                dataset_run_size="full",
                created_at=created_at or _T0,
            )
        )
        await s.flush()
        for i, validator in enumerate(_VALIDATORS):
            await upsert_score(
                s,
                agent_id=agent_id,
                validator_hotkey=validator,
                bench_version=bench_version,
                run_id=f"run_{agent_id.hex[:8]}_{i}",
                seed=42,
                composite=composite,
                tool_mean=composite,
                memory_mean=memory_mean if memory_mean is not None else composite,
                median_ms=500,
                n=110,
                generated_at=_T0 + timedelta(minutes=i),
                signature="ab" * 64,
                details=_details(total_tokens, bench_version=bench_version),
            )
    return agent_id


async def _seed_v7_board(
    maker: async_sessionmaker[AsyncSession],
) -> dict[str, UUID]:
    """Three distinct-lineage finalized v7 agents: lean, median, heavy."""
    await _activate_bench_version(maker, 7)
    lean = await _seed_finalized(
        maker, miner=_MINERS[0], composite=0.80, total_tokens=100_000
    )
    mid = await _seed_finalized(
        maker, miner=_MINERS[1], composite=0.70, total_tokens=200_000
    )
    heavy = await _seed_finalized(
        maker, miner=_MINERS[2], composite=0.60, total_tokens=400_000
    )
    return {"lean": lean, "mid": mid, "heavy": heavy}


def _entry(payload: dict, agent_id: UUID) -> dict:
    for entry in payload["entries"]:
        if entry["agent_id"] == str(agent_id):
            return entry
    raise AssertionError(f"agent {agent_id} not on the board")


class TestLeaderboardBonusExposure:
    async def test_active_cohort_awards_frozen_bonuses(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        agents = await _seed_v7_board(session_maker)
        app = _make_app(session_maker, efficiency=_ENABLED)
        async with _client(app) as client:
            response = await client.get("/api/v1/public/leaderboard")
        assert response.status_code == 200
        payload = response.json()

        status = payload["efficiency"]
        assert status is not None
        assert status["active"] is True
        assert status["bench_version"] == 7
        assert status["run_size"] == "full"
        assert status["n_min"] == 3
        assert status["cohort_size"] == 3
        assert status["bonus_cap"] == 0.05
        # 3 members at 100k/200k/400k: nearest-rank P25 = 100k, median = 200k.
        assert status["reference_p25_tokens"] == 100_000.0
        assert status["reference_median_tokens"] == 200_000.0

        lean = _entry(payload, agents["lean"])
        assert lean["efficiency_bonus"] == 0.05
        assert lean["effective_composite"] == pytest.approx(0.80 * 1.05)
        assert lean["efficiency_snapshot_id"] == status["snapshot_id"]
        assert lean["composite"] == 0.80  # base composite is never modified

        mid = _entry(payload, agents["mid"])
        assert mid["efficiency_bonus"] == 0.0
        assert mid["effective_composite"] == pytest.approx(0.70)

        heavy = _entry(payload, agents["heavy"])
        assert heavy["efficiency_bonus"] == 0.0
        assert heavy["effective_composite"] == pytest.approx(0.60)

        # Ranking still follows the base composite (fold wiring is flag-off).
        assert [e["agent_id"] for e in payload["entries"]] == [
            str(agents["lean"]),
            str(agents["mid"]),
            str(agents["heavy"]),
        ]

    async def test_bonuses_and_reference_are_frozen_within_an_epoch(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        agents = await _seed_v7_board(session_maker)
        app = _make_app(session_maker, efficiency=_ENABLED)
        async with _client(app) as client:
            first = (await client.get("/api/v1/public/leaderboard")).json()

            # A cheaper newcomer finalizes mid-epoch. It must be scored against
            # the FROZEN reference; nothing already published may move.
            newcomer = await _seed_finalized(
                session_maker,
                miner=_MINERS[3],
                composite=0.75,
                total_tokens=50_000,
            )
            second = (await client.get("/api/v1/public/leaderboard")).json()

        assert second["efficiency"]["snapshot_id"] == first["efficiency"]["snapshot_id"]
        assert second["efficiency"]["reference_p25_tokens"] == 100_000.0
        assert second["efficiency"]["reference_median_tokens"] == 200_000.0
        for name in ("lean", "mid", "heavy"):
            assert (
                _entry(second, agents[name])["efficiency_bonus"]
                == _entry(first, agents[name])["efficiency_bonus"]
            )
        # 50k <= frozen P25 (100k) -> full bonus against the frozen frontier.
        assert _entry(second, newcomer)["efficiency_bonus"] == 0.05

    async def test_below_n_min_is_inactive_and_awards_nothing(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        await _activate_bench_version(session_maker, 7)
        a = await _seed_finalized(
            session_maker, miner=_MINERS[0], composite=0.8, total_tokens=100_000
        )
        b = await _seed_finalized(
            session_maker, miner=_MINERS[1], composite=0.7, total_tokens=200_000
        )
        app = _make_app(session_maker, efficiency=_ENABLED)
        async with _client(app) as client:
            payload = (await client.get("/api/v1/public/leaderboard")).json()

        status = payload["efficiency"]
        assert status is not None
        assert status["active"] is False
        assert status["cohort_size"] == 2
        assert status["reference_p25_tokens"] is None
        assert status["reference_median_tokens"] is None
        for agent_id in (a, b):
            entry = _entry(payload, agent_id)
            assert entry["efficiency_bonus"] is None
            assert entry["effective_composite"] is None
            assert entry["efficiency_snapshot_id"] is None
        # Inactive epochs assign no rows at all — activation later must be
        # able to freeze these agents at their first ACTIVE epoch.
        async with session_maker() as s:
            assert (await s.scalars(select(EfficiencyBonus))).all() == []

    async def test_lineage_dedupe_collapses_copies_before_the_frontier(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        await _activate_bench_version(session_maker, 7)
        original = await _seed_finalized(
            session_maker,
            miner=_MINERS[0],
            composite=0.85,
            total_tokens=100_000,
            sha256="cc" * 32,
            created_at=_T0,
        )
        copycat = await _seed_finalized(
            session_maker,
            miner=_MINERS[1],
            composite=0.80,
            total_tokens=100_000,
            sha256="cc" * 32,  # byte-identical artifact under another hotkey
            created_at=_T0 + timedelta(hours=1),
        )
        others = [
            await _seed_finalized(
                session_maker,
                miner=miner,
                composite=0.7 - i * 0.05,
                total_tokens=200_000 + i * 100_000,
            )
            for i, miner in enumerate(_MINERS[2:4])
        ]
        app = _make_app(session_maker, efficiency=_ENABLED)
        async with _client(app) as client:
            board = (await client.get("/api/v1/public/leaderboard")).json()
            snapshot_id = board["efficiency"]["snapshot_id"]
            snapshot = (
                await client.get(f"/api/v1/public/efficiency/snapshots/{snapshot_id}")
            ).json()

        # 4 submissions, 3 lineages: the duplicate collapsed into the original.
        assert board["efficiency"]["cohort_size"] == 3
        members = {member["agent_id"]: member for member in snapshot["members"]}
        assert str(original) in members
        assert str(copycat) not in members
        assert members[str(original)]["collapsed_agent_ids"] == [str(copycat)]
        for other in others:
            assert str(other) in members
        # No raw lineage digests on the public wire — opaque ordinals only.
        assert "lineage_key" not in next(iter(members.values()))

    async def test_pre_v7_board_is_untouched(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        # Default active version (2): flag ON but the bonus must never engage.
        for i, miner in enumerate(_MINERS[:3]):
            await _seed_finalized(
                session_maker,
                miner=miner,
                composite=0.8 - i * 0.1,
                total_tokens=100_000 * (i + 1),
                bench_version=2,
            )
        app = _make_app(session_maker, efficiency=_ENABLED)
        async with _client(app) as client:
            payload = (await client.get("/api/v1/public/leaderboard")).json()

        assert payload["efficiency"] is None
        for entry in payload["entries"]:
            assert entry["efficiency_bonus"] is None
            assert entry["effective_composite"] is None
            assert entry["efficiency_snapshot_id"] is None
        async with session_maker() as s:
            assert (await s.scalars(select(EfficiencyCohortSnapshot))).all() == []
            assert (await s.scalars(select(EfficiencyBonus))).all() == []

    async def test_disabled_flag_writes_and_exposes_nothing(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        agents = await _seed_v7_board(session_maker)
        app = _make_app(session_maker, efficiency=EfficiencyBonusConfig())
        async with _client(app) as client:
            payload = (await client.get("/api/v1/public/leaderboard")).json()

        assert payload["efficiency"] is None
        assert _entry(payload, agents["lean"])["efficiency_bonus"] is None
        async with session_maker() as s:
            assert (await s.scalars(select(EfficiencyCohortSnapshot))).all() == []
            assert (await s.scalars(select(EfficiencyBonus))).all() == []

    async def test_snapshot_endpoint_404_for_unknown_id(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        app = _make_app(session_maker, efficiency=_ENABLED)
        async with _client(app) as client:
            response = await client.get(
                f"/api/v1/public/efficiency/snapshots/{uuid4()}"
            )
        assert response.status_code == 404


class TestEpochFreezing:
    async def test_new_epoch_freezes_a_new_snapshot_without_mutating_the_old(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        agents = await _seed_v7_board(session_maker)
        async with session_maker() as s:
            await ensure_efficiency_state(s, _ENABLED, now=_T0)
        async with session_maker() as s:
            snapshots = (await s.scalars(select(EfficiencyCohortSnapshot))).all()
            assert len(snapshots) == 1
            first_id = snapshots[0].snapshot_id
            first_epoch = snapshots[0].epoch_index
            first_members = list(snapshots[0].members or [])
            first_p25 = snapshots[0].reference_p25_tokens

        # A leaner agent lands in the NEXT epoch: fresh cohort, fresh frontier.
        newcomer = await _seed_finalized(
            session_maker,
            miner=_MINERS[3],
            composite=0.75,
            total_tokens=50_000,
        )
        async with session_maker() as s:
            await ensure_efficiency_state(s, _ENABLED, now=_T0 + timedelta(hours=25))

        async with session_maker() as s:
            snapshots = (
                await s.scalars(
                    select(EfficiencyCohortSnapshot).order_by(
                        EfficiencyCohortSnapshot.epoch_index
                    )
                )
            ).all()
            assert len(snapshots) == 2
            old, new = snapshots
            # The historical snapshot did not move.
            assert old.snapshot_id == first_id
            assert old.epoch_index == first_epoch
            assert list(old.members or []) == first_members
            assert old.reference_p25_tokens == first_p25
            # The new one reflects the new population under the ratcheted
            # quality floor (median of the previous cohort = 0.7, so the 0.6
            # agent drops out and the 0.75 newcomer joins).
            assert new.epoch_index == first_epoch + 1
            assert new.quality_floor == 0.7
            assert len(new.members or []) == 3
            assert new.reference_p25_tokens == 50_000.0

            bonuses = {
                row.agent_id: row
                for row in (await s.scalars(select(EfficiencyBonus))).all()
            }
        # Epoch-1 agents stay frozen against snapshot 1; only the newcomer is
        # assigned in epoch 2, against snapshot 2.
        for agent_id in agents.values():
            assert bonuses[agent_id].snapshot_id == first_id
        assert bonuses[newcomer].snapshot_id == new.snapshot_id
        assert bonuses[newcomer].bonus == 0.05

    async def test_quality_floors_ratchet_from_previous_active_cohort(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        await _seed_v7_board(session_maker)  # composites 0.8 / 0.7 / 0.6
        async with session_maker() as s:
            await ensure_efficiency_state(s, _ENABLED, now=_T0)
        async with session_maker() as s:
            await ensure_efficiency_state(s, _ENABLED, now=_T0 + timedelta(hours=25))
        async with session_maker() as s:
            snapshots = (
                await s.scalars(
                    select(EfficiencyCohortSnapshot).order_by(
                        EfficiencyCohortSnapshot.epoch_index
                    )
                )
            ).all()
        assert snapshots[0].quality_floor == 0.0
        # Epoch 2: Q_min = previous cohort median composite (0.7),
        # M_min = 0.8 x previous median memory_mean (0.8 x 0.7).
        assert snapshots[1].quality_floor == 0.7
        assert snapshots[1].memory_floor == pytest.approx(0.8 * 0.7)
        # The 0.6 agent no longer qualifies; cohort shrinks below n_min=3.
        assert snapshots[1].active is False


class TestValidatorLedgerFoldFlag:
    def _headers(self) -> dict[str, str]:
        nonce = uuid4()
        requested_at = datetime.now(UTC)
        requested = requested_at.astimezone(UTC).isoformat(timespec="microseconds")
        signed = (
            f"validator-ledger:v1:{_VALIDATOR_HOTKEY}:{nonce}:{requested}"
        ).encode()
        return {
            "X-Validator-Hotkey": _VALIDATOR_HOTKEY,
            "X-Validator-Ledger-Nonce": str(nonce),
            "X-Validator-Ledger-Requested-At": requested_at.isoformat(),
            "X-Validator-Ledger-Signature": _KEYPAIR.sign(signed).hex(),
        }

    def _install_chain(self, app: FastAPI) -> None:
        from unittest.mock import AsyncMock, MagicMock

        async def _chain() -> MagicMock:
            c = MagicMock()
            c.get_recent_neurons = AsyncMock(
                return_value=[
                    NeuronInfo(
                        hotkey=_VALIDATOR_HOTKEY,
                        coldkey="5GReceiverColdkeyPlaceholderXXXXXXXXXXXXXXXXXXX",
                        uid=1,
                        stake=1000.0,
                        validator_permit=True,
                    )
                ]
            )
            return c

        app.dependency_overrides[get_chain_client] = _chain

    async def _ledger(
        self,
        maker: async_sessionmaker[AsyncSession],
        efficiency: EfficiencyBonusConfig,
    ) -> dict:
        app = _make_app(maker, efficiency=efficiency)
        self._install_chain(app)
        async with _client(app) as client:
            response = await client.get(
                "/api/v1/scoring/scores", headers=self._headers()
            )
        assert response.status_code == 200
        return response.json()

    async def test_fold_flag_off_keeps_ledger_fields_null(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        await _seed_v7_board(session_maker)
        async with session_maker() as s:
            await ensure_efficiency_state(s, _ENABLED, now=_T0)

        payload = await self._ledger(session_maker, _ENABLED)
        assert payload["count"] == 3
        for entry in payload["entries"]:
            assert entry["efficiency_bonus"] is None
            assert entry["effective_composite"] is None

    async def test_fold_flag_on_exposes_frozen_effective_composite(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        agents = await _seed_v7_board(session_maker)
        fold_on = EfficiencyBonusConfig(enabled=True, fold_enabled=True, min_cohort=3)
        async with session_maker() as s:
            await ensure_efficiency_state(s, fold_on, now=_T0)

        payload = await self._ledger(session_maker, fold_on)
        by_agent = {entry["agent_id"]: entry for entry in payload["entries"]}
        lean = by_agent[str(agents["lean"])]
        assert lean["efficiency_bonus"] == 0.05
        assert lean["composite"] == 0.80
        assert lean["effective_composite"] == pytest.approx(0.80 * 1.05)
        heavy = by_agent[str(agents["heavy"])]
        assert heavy["efficiency_bonus"] == 0.0
        assert heavy["effective_composite"] == pytest.approx(0.60)
