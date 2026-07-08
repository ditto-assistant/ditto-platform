"""Unit tests for :mod:`ditto.db.queries.scores` against SQLite-in-memory."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.agent_status import AgentStatus
from ditto.db.errors import IntegrityError as DbIntegrityError
from ditto.db.models import Agent
from ditto.db.queries.scores import (
    list_eligible_ledger,
    list_scores_for_agent,
    upsert_score,
)

_MINER = "5DhaT8U7LVwnnJNUU8VL1XEipicatoaDVVq7cHo227gogVZm"
_MINER_B = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
_VALIDATOR = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
_GEN_AT = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)


async def _seed_agent(session: AsyncSession) -> Agent:
    agent = Agent(
        agent_id=uuid4(),
        miner_hotkey=_MINER,
        name="alpha",
        sha256="ab" * 32,
        status=AgentStatus.EVALUATING,
        created_at=datetime.now(UTC),
    )
    async with session.begin():
        session.add(agent)
    return agent


async def _upsert(session: AsyncSession, agent_id: object, **overrides: object) -> None:
    kwargs: dict = {
        "agent_id": agent_id,
        "validator_hotkey": _VALIDATOR,
        "run_id": "run_1",
        "seed": 42,
        "composite": 0.7,
        "tool_mean": 0.8,
        "memory_mean": 0.6,
        "median_ms": 500,
        "n": 20,
        "generated_at": _GEN_AT,
        "details": None,
    }
    kwargs.update(overrides)
    async with session.begin():
        await upsert_score(session, **kwargs)


class TestUpsertScore:
    async def test_inserts_new_row(self, session: AsyncSession) -> None:
        agent = await _seed_agent(session)
        await _upsert(session, agent.agent_id, details={"per_case": [{"x": 1}]})

        scores = await list_scores_for_agent(session, agent_id=agent.agent_id)
        assert len(scores) == 1
        assert scores[0].run_id == "run_1"
        assert scores[0].details == {"per_case": [{"x": 1}]}

    async def test_second_upsert_overwrites_same_row(
        self, session: AsyncSession
    ) -> None:
        agent = await _seed_agent(session)
        await _upsert(session, agent.agent_id, run_id="run_1", composite=0.4)
        await _upsert(session, agent.agent_id, run_id="run_2", composite=0.95)

        scores = await list_scores_for_agent(session, agent_id=agent.agent_id)
        assert len(scores) == 1
        assert scores[0].run_id == "run_2"
        assert scores[0].composite == pytest.approx(0.95)

    async def test_unknown_agent_raises_integrity(self, session: AsyncSession) -> None:
        with pytest.raises(DbIntegrityError):
            await _upsert(session, uuid4())

    async def test_list_empty_when_unscored(self, session: AsyncSession) -> None:
        agent = await _seed_agent(session)
        assert await list_scores_for_agent(session, agent_id=agent.agent_id) == []


async def _seed_scored(
    session: AsyncSession,
    *,
    miner: str,
    composite: float,
    created_at: datetime,
    size_bytes: int = 524288,
    status: AgentStatus = AgentStatus.SCORED,
    normalized_source_hash: str | None = None,
    prompt_fingerprint: dict | None = None,
    code_embedding: list | None = None,
    code_embed_model: str | None = None,
) -> Agent:
    """Seed one agent + its score row, in the given lifecycle state."""
    agent = Agent(
        agent_id=uuid4(),
        miner_hotkey=miner,
        name="agent",
        sha256="ab" * 32,
        size_bytes=size_bytes,
        status=status,
        created_at=created_at,
        normalized_source_hash=normalized_source_hash,
        prompt_fingerprint=prompt_fingerprint,
        code_embedding=code_embedding,
        code_embed_model=code_embed_model,
    )
    async with session.begin():
        session.add(agent)
        await session.flush()
        await upsert_score(
            session,
            agent_id=agent.agent_id,
            validator_hotkey=_VALIDATOR,
            run_id="run_1",
            seed=42,
            composite=composite,
            tool_mean=composite,
            memory_mean=composite,
            median_ms=500,
            n=20,
            generated_at=_GEN_AT,
            signature="ab" * 64,
        )
    return agent


class TestListEligibleLedger:
    async def test_empty_when_nothing_scored(self, session: AsyncSession) -> None:
        assert await list_eligible_ledger(session) == []

    async def test_best_agent_per_miner_only(self, session: AsyncSession) -> None:
        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        # Same miner, two scored agents; only the higher composite should appear.
        await _seed_scored(session, miner=_MINER, composite=0.5, created_at=t0)
        best = await _seed_scored(
            session, miner=_MINER, composite=0.8, created_at=t0.replace(hour=13)
        )
        ledger = await list_eligible_ledger(session)
        assert len(ledger) == 1
        assert ledger[0].agent_id == best.agent_id
        assert ledger[0].composite == pytest.approx(0.8)
        assert ledger[0].miner_hotkey == _MINER
        assert ledger[0].signature == "ab" * 64

    async def test_normalized_source_hash_flows_through(
        self, session: AsyncSession
    ) -> None:
        # The exact-repack hash must reach the ledger so the gate can read it.
        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        await _seed_scored(
            session,
            miner=_MINER,
            composite=0.7,
            created_at=t0,
            normalized_source_hash="ns" * 32,
        )
        ledger = await list_eligible_ledger(session)
        assert len(ledger) == 1
        assert ledger[0].normalized_source_hash == "ns" * 32

    async def test_prompt_fingerprint_flows_through(
        self, session: AsyncSession
    ) -> None:
        # The prompt sketch (shadow signal) must reach the ledger for the gate.
        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        sketch = {"v": "p1", "k": 256, "card": 2, "m": ["aa", "bb"]}
        await _seed_scored(
            session,
            miner=_MINER,
            composite=0.7,
            created_at=t0,
            prompt_fingerprint=sketch,
        )
        ledger = await list_eligible_ledger(session)
        assert len(ledger) == 1
        assert ledger[0].prompt_fingerprint == sketch

    async def test_code_embedding_flows_through(self, session: AsyncSession) -> None:
        # The code-embedding vector + its model tag must reach the ledger for the gate
        # cosine.
        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        vector = [0.1, 0.2, 0.3]
        await _seed_scored(
            session,
            miner=_MINER,
            composite=0.7,
            created_at=t0,
            code_embedding=vector,
            code_embed_model="Qwen/Qwen3-Embedding-0.6B@main",
        )
        ledger = await list_eligible_ledger(session)
        assert len(ledger) == 1
        assert ledger[0].code_embedding == vector
        assert ledger[0].code_embed_model == "Qwen/Qwen3-Embedding-0.6B@main"

    async def test_ordered_by_composite_desc(self, session: AsyncSession) -> None:
        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        await _seed_scored(session, miner=_MINER, composite=0.4, created_at=t0)
        await _seed_scored(session, miner=_MINER_B, composite=0.9, created_at=t0)
        ledger = await list_eligible_ledger(session)
        assert [e.miner_hotkey for e in ledger] == [_MINER_B, _MINER]

    async def test_excludes_non_scored_states(self, session: AsyncSession) -> None:
        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        # Held / evaluating agents must not leak into the eligible ledger.
        await _seed_scored(
            session,
            miner=_MINER,
            composite=0.95,
            created_at=t0,
            status=AgentStatus.ATH_PENDING_REVIEW,
        )
        await _seed_scored(
            session,
            miner=_MINER_B,
            composite=0.6,
            created_at=t0,
            status=AgentStatus.SCORED,
        )
        ledger = await list_eligible_ledger(session)
        assert [e.miner_hotkey for e in ledger] == [_MINER_B]

    async def test_multiple_score_rows_returns_consistent_row(
        self, session: AsyncSession
    ) -> None:
        # An agent scored by two validators (e.g. after a hotkey rotation) has two
        # scores rows. The ledger must return the BEST row whole — composite, seed,
        # run_id, signature, validator_hotkey all from the same physical row — not
        # a per-column MAX that stitches a mismatched (composite, signature) tuple.
        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        agent = Agent(
            agent_id=uuid4(),
            miner_hotkey=_MINER,
            name="agent",
            sha256="ab" * 32,
            size_bytes=524288,
            status=AgentStatus.SCORED,
            created_at=t0,
        )
        async with session.begin():
            session.add(agent)
            await session.flush()
            # Low composite from an alphabetically-later validator hotkey.
            await upsert_score(
                session,
                agent_id=agent.agent_id,
                validator_hotkey="5Zzz_validator",
                run_id="run_low",
                seed=111,
                composite=0.70,
                tool_mean=0.7,
                memory_mean=0.7,
                median_ms=500,
                n=20,
                generated_at=_GEN_AT,
                signature="11" * 64,
            )
            # High composite from an alphabetically-earlier validator hotkey.
            await upsert_score(
                session,
                agent_id=agent.agent_id,
                validator_hotkey="5Aaa_validator",
                run_id="run_high",
                seed=222,
                composite=0.90,
                tool_mean=0.9,
                memory_mean=0.9,
                median_ms=500,
                n=20,
                generated_at=_GEN_AT,
                signature="99" * 64,
            )
        ledger = await list_eligible_ledger(session)
        assert len(ledger) == 1
        e = ledger[0]
        # Every field must come from the high-composite row, not be stitched.
        assert e.composite == pytest.approx(0.90)
        assert e.seed == 222
        assert e.run_id == "run_high"
        assert e.signature == "99" * 64
        assert e.validator_hotkey == "5Aaa_validator"
