"""Unit tests for :mod:`ditto.api_server.endpoints.public`.

``GET /api/v1/public/leaderboard`` is open (no validator auth) and aggregate-only:
it must rank miners by composite, expose tool/memory means, and NEVER leak the
integrity-internal fields (``signature``, ``sha256``, ``validator_hotkey``) or
per-case detail.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

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
from ditto.api_server.datapipeline import DataPipelineError
from ditto.api_server.dependencies import get_dataset_generator, get_session
from ditto.db.models import Agent, Base
from ditto.db.queries.audit import (
    EVENT_SCORE,
    GENESIS_HASH,
    append_audit_entry,
)
from ditto.db.queries.scores import upsert_score

_MINER_A = "5DhaT8U7LVwnnJNUU8VL1XEipicatoaDVVq7cHo227gogVZm"
_MINER_B = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


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
            created_at=datetime.now(UTC),
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
            )
    return str(agent_id)


async def _seed_agent(
    maker: async_sessionmaker[AsyncSession],
    *,
    miner: str,
    status: AgentStatus = AgentStatus.UPLOADED,
) -> None:
    """Seed a submission with no score (e.g. still uploaded/evaluating)."""
    async with maker() as s, s.begin():
        s.add(
            Agent(
                agent_id=uuid4(),
                miner_hotkey=miner,
                name="agent",
                sha256="cd" * 32,
                size_bytes=524288,
                status=status,
                created_at=datetime.now(UTC),
            )
        )


class TestPublicLeaderboard:
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
        assert body["count"] == 2
        assert [e["rank"] for e in body["entries"]] == [1, 2]
        assert [e["miner_hotkey"] for e in body["entries"]] == [_MINER_B, _MINER_A]
        top = body["entries"][0]
        assert top["composite"] == pytest.approx(0.9)
        assert top["tool_mean"] == pytest.approx(0.95)
        assert top["memory_mean"] == pytest.approx(0.8)

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
        for leaked in ("signature", "sha256", "validator_hotkey", "agent_id", "seed"):
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
        # Two scored miners (recent), latencies 400 + 800 => avg 600.
        await _seed_scored(
            session_maker,
            miner=_MINER_A,
            composite=0.4,
            tool_mean=0.5,
            memory_mean=0.3,
            median_ms=400,
            generated_at=now - timedelta(minutes=5),
        )
        await _seed_scored(
            session_maker,
            miner=_MINER_B,
            composite=0.9,
            tool_mean=0.95,
            memory_mean=0.8,
            median_ms=800,
            generated_at=now - timedelta(days=2),  # outside the 24h window
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
        assert body["scores_24h"] == 1  # only MINER_A is within 24h
        assert body["avg_latency_ms"] == 600
        # last_scored_at is the newest generated_at (MINER_A, ~5 min ago).
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
            "scores_24h": 0,
            "avg_latency_ms": None,
        }


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
        # v2 is the current (live) version: its answer keys must never be released.
        await _seed_k3(
            session_maker,
            miner=_MINER_A,
            composites=[0.4, 0.5, 0.6],
            details={"bench_version": 2, "per_case": [{"expected": ["x"]}]},
        )
        _install_db(app, session_maker)
        resp = await client.get("/api/v1/public/bench/2/corpus")
        assert resp.status_code == 409
        # ...and the live answer key is not in the refusal body.
        assert '"expected"' not in resp.text

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
        self, client: httpx.AsyncClient, monkeypatch
    ) -> None:
        monkeypatch.delenv("STORAGE_PUBLIC_BUCKET", raising=False)
        resp = await client.get("/api/v1/public/bench/config")
        assert resp.status_code == 200
        assert "max-age=300" in resp.headers["Cache-Control"]
        body = resp.json()
        assert body["bench_version"] >= 2
        h = body["harness"]
        assert h["locked"] is True
        assert h["canonical_id"] == "qwen/qwen3-32b"
        assert h["serving"] == "Qwen/Qwen3-32B-TEE"
        assert h["thinking"] is False
        assert body["grading"]["judge_free"] is True
        assert "dittobench-datagen" in body["grading"]["grader"]
        assert "dataset_sha256" in body["dataset"]["reproduce"]
        assert body["public_mirror_url_template"] is None
        assert body["ledger_path"] == "/api/v1/scoring/scores"

    async def test_mirror_template_from_env(
        self, client: httpx.AsyncClient, monkeypatch
    ) -> None:
        monkeypatch.setenv("STORAGE_PUBLIC_BUCKET", "ditto-platform-public-dev")
        body = (await client.get("/api/v1/public/bench/config")).json()
        assert body["public_mirror_url_template"] == (
            "https://storage.googleapis.com/ditto-platform-public-dev/scored/{agent_id}.json"
        )
