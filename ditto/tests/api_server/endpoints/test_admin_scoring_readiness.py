"""Coverage for the admin scoring-readiness inspection endpoint."""

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.api_server.dependencies import get_session
from ditto.db.models import (
    Agent,
    Base,
    BenchmarkDataset,
    BenchmarkRollout,
    BenchmarkRolloutMember,
)

_TOKEN = "test-admin-token-at-least-32-characters"
_HEADERS = {"Authorization": f"Bearer {_TOKEN}", "X-Admin-Actor": "operator"}
_T0 = datetime(2026, 7, 21, 4, tzinfo=UTC)


@pytest.fixture
async def sr_engine() -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def _fk(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def sr_maker(sr_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(sr_engine, expire_on_commit=False)


def _install(app: FastAPI, maker: async_sessionmaker[AsyncSession]) -> None:
    app.state.config = replace(app.state.config, admin_api_token=_TOKEN)

    async def _session() -> AsyncIterator[AsyncSession]:
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = _session


async def _seed(
    maker: async_sessionmaker[AsyncSession],
    *,
    active_version: int = 4,
    status: AgentStatus = AgentStatus.EVALUATING,
    policy_version: int = SCREENING_POLICY_VERSION,
    with_image: bool = True,
    with_dataset: bool = True,
    historical: bool = False,
) -> UUID:
    agent_id = uuid4()
    async with maker() as session, session.begin():
        session.add(
            BenchmarkRollout(
                rollout_id=uuid4(),
                from_version=2,
                desired_version=active_version,
                status="activated",
                created_at=_T0 - timedelta(hours=2),
                activated_at=_T0 - timedelta(hours=1),
            )
        )
        image = (
            {
                "screened_image_sha256": "a" * 64,
                "screened_image_size_bytes": 1024,
                "screened_image_id": "sha256:" + "b" * 64,
                "screened_image_ref": f"ditto-screen/{agent_id}:latest",
                "screened_image_upload_id": uuid4(),
                "screened_image_verified_at": _T0 - timedelta(minutes=30),
            }
            if with_image
            else {}
        )
        session.add(
            Agent(
                agent_id=agent_id,
                miner_hotkey="5Miner",
                name="candidate",
                version=1,
                sha256=agent_id.hex * 2,
                status=status,
                screening_policy_version=policy_version,
                created_at=(_T0 - timedelta(days=1) if historical else _T0),
                **image,
            )
        )
        if with_dataset:
            session.add(
                BenchmarkDataset(
                    agent_id=agent_id,
                    bench_version=active_version,
                    seed=7,
                    sha256="d" * 64,
                    run_size="full",
                )
            )
    return agent_id


async def _get(client: httpx.AsyncClient, agent_id: UUID) -> dict:
    resp = await client.get(
        f"/api/v1/admin/agents/{agent_id}/scoring-readiness", headers=_HEADERS
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


async def test_fully_ready_v4_agent_is_leaseable(
    app: FastAPI, client: httpx.AsyncClient, sr_maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id = await _seed(sr_maker)
    _install(app, sr_maker)
    body = await _get(client, agent_id)
    assert body["leaseable"] is True
    assert body["blocking_reasons"] == []
    assert body["active_bench_version"] == 4
    assert body["has_versioned_dataset"] is True
    assert body["screened_image"]["complete"] is True


async def test_historical_nonmember_is_not_leaseable_even_with_dataset(
    app: FastAPI, client: httpx.AsyncClient, sr_maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id = await _seed(sr_maker, historical=True)
    _install(app, sr_maker)

    body = await _get(client, agent_id)

    assert body["leaseable"] is False
    assert body["has_versioned_dataset"] is True
    assert any("historical submission" in reason for reason in body["blocking_reasons"])


async def test_missing_screened_image_blocks_and_names_fields(
    app: FastAPI, client: httpx.AsyncClient, sr_maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id = await _seed(sr_maker, with_image=False)
    _install(app, sr_maker)
    body = await _get(client, agent_id)
    assert body["leaseable"] is False
    assert body["screened_image"]["complete"] is False
    assert body["screened_image"]["missing_fields"]
    assert any("not built yet" in r for r in body["blocking_reasons"])


async def test_missing_v4_dataset_blocks(
    app: FastAPI, client: httpx.AsyncClient, sr_maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id = await _seed(sr_maker, with_dataset=False)
    _install(app, sr_maker)
    body = await _get(client, agent_id)
    assert body["leaseable"] is False
    assert body["has_versioned_dataset"] is False
    assert any("benchmark dataset" in r for r in body["blocking_reasons"])


async def test_stale_screening_policy_blocks(
    app: FastAPI, client: httpx.AsyncClient, sr_maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id = await _seed(sr_maker, policy_version=SCREENING_POLICY_VERSION - 1)
    _install(app, sr_maker)
    body = await _get(client, agent_id)
    assert body["leaseable"] is False
    assert any("re-screen" in r for r in body["blocking_reasons"])


async def test_non_evaluating_agent_blocks(
    app: FastAPI, client: httpx.AsyncClient, sr_maker: async_sessionmaker[AsyncSession]
) -> None:
    agent_id = await _seed(sr_maker, status=AgentStatus.QUARANTINED)
    _install(app, sr_maker)
    body = await _get(client, agent_id)
    assert body["leaseable"] is False
    assert any("not evaluating" in r for r in body["blocking_reasons"])


async def test_unknown_agent_is_404(
    app: FastAPI, client: httpx.AsyncClient, sr_maker: async_sessionmaker[AsyncSession]
) -> None:
    _install(app, sr_maker)
    resp = await client.get(
        f"/api/v1/admin/agents/{uuid4()}/scoring-readiness", headers=_HEADERS
    )
    assert resp.status_code == 404


async def _seed_open_rollout(
    maker: async_sessionmaker[AsyncSession],
    *,
    agent_created_at: datetime,
    agent_status: AgentStatus,
    dataset_versions: tuple[int, ...],
    cohort_member: bool = False,
) -> UUID:
    """Seed an active v6 ledger with a still-collecting v6 -> v7 rollout."""
    rollout_started = _T0 - timedelta(hours=1)
    agent_id = uuid4()
    open_rollout_id = uuid4()
    async with maker() as session, session.begin():
        session.add_all(
            [
                BenchmarkRollout(
                    rollout_id=uuid4(),
                    from_version=2,
                    desired_version=6,
                    status="activated",
                    created_at=_T0 - timedelta(days=7),
                    activated_at=_T0 - timedelta(days=6),
                ),
                BenchmarkRollout(
                    rollout_id=open_rollout_id,
                    from_version=6,
                    desired_version=7,
                    status="collecting",
                    cohort_size=5,
                    created_at=rollout_started,
                ),
                Agent(
                    agent_id=agent_id,
                    miner_hotkey="5Miner",
                    name="cool-v1",
                    version=1,
                    sha256=agent_id.hex * 2,
                    status=agent_status,
                    screening_policy_version=SCREENING_POLICY_VERSION,
                    created_at=agent_created_at,
                    screened_image_sha256="a" * 64,
                    screened_image_size_bytes=1024,
                    screened_image_id="sha256:" + "b" * 64,
                    screened_image_ref=f"ditto-screen/{agent_id}:latest",
                    screened_image_upload_id=uuid4(),
                    screened_image_verified_at=_T0 - timedelta(minutes=30),
                ),
            ]
        )
        await session.flush()
        for version in dataset_versions:
            session.add(
                BenchmarkDataset(
                    agent_id=agent_id,
                    bench_version=version,
                    seed=7,
                    sha256="d" * 64,
                    run_size="full",
                )
            )
        if cohort_member:
            session.add(
                BenchmarkRolloutMember(
                    rollout_id=open_rollout_id,
                    agent_id=agent_id,
                    position=1,
                    frozen_miner_hotkey="5Miner",
                    frozen_composite=0.97,
                )
            )
    return agent_id


async def test_submission_that_arrived_during_rollout_reports_the_desired_era(
    app: FastAPI, client: httpx.AsyncClient, sr_maker: async_sessionmaker[AsyncSession]
) -> None:
    """A post-rollout arrival is queued at v7, so v6 must not be the answer.

    Regression for a false "not leaseable" that cost an operator investigation:
    the endpoint answered against the active version and reported a missing v6
    dataset for a submission that was admitted, queued, and actively leased in
    the v7 fresh-submission lane.
    """
    agent_id = await _seed_open_rollout(
        sr_maker,
        agent_created_at=_T0,
        agent_status=AgentStatus.EVALUATING,
        dataset_versions=(7,),
    )
    _install(app, sr_maker)

    body = await _get(client, agent_id)

    assert body["active_bench_version"] == 6
    assert body["scoring_bench_version"] == 7
    assert body["scoring_lane"] == "fresh_submission"
    assert body["has_versioned_dataset"] is True
    assert body["leaseable"] is True, body["blocking_reasons"]
    assert not any("v6 benchmark dataset" in r for r in body["blocking_reasons"])


async def test_rollout_cohort_member_reports_the_desired_era_from_scored(
    app: FastAPI, client: httpx.AsyncClient, sr_maker: async_sessionmaker[AsyncSession]
) -> None:
    """The cohort lane rescores frozen members straight out of ``scored``."""
    agent_id = await _seed_open_rollout(
        sr_maker,
        agent_created_at=_T0 - timedelta(days=3),
        agent_status=AgentStatus.SCORED,
        dataset_versions=(6, 7),
        cohort_member=True,
    )
    _install(app, sr_maker)

    body = await _get(client, agent_id)

    assert body["active_bench_version"] == 6
    assert body["scoring_bench_version"] == 7
    assert body["scoring_lane"] == "rollout_cohort"
    assert body["leaseable"] is True, body["blocking_reasons"]


async def test_pre_rollout_nonmember_still_reports_the_active_era(
    app: FastAPI, client: httpx.AsyncClient, sr_maker: async_sessionmaker[AsyncSession]
) -> None:
    """Older non-cohort submissions stay on the active version during a rollout."""
    agent_id = await _seed_open_rollout(
        sr_maker,
        agent_created_at=_T0 - timedelta(days=3),
        agent_status=AgentStatus.EVALUATING,
        dataset_versions=(6,),
    )
    _install(app, sr_maker)

    body = await _get(client, agent_id)

    assert body["active_bench_version"] == 6
    assert body["scoring_bench_version"] == 6
    assert body["scoring_lane"] == "ordinary"
    assert body["has_versioned_dataset"] is True
