"""Unit tests for :mod:`ditto.db.queries.scores` against SQLite-in-memory."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models.agent_status import AgentStatus
from ditto.db.errors import IntegrityError as DbIntegrityError
from ditto.db.models import Agent, EvaluationPayment
from ditto.db.queries.scores import (
    MIN_ELIGIBLE_CASES,
    list_eligible_ledger,
    list_scores_for_agent,
    quorum_composites,
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

    async def test_database_rejects_above_one_for_v5(
        self, session: AsyncSession
    ) -> None:
        v5_agent = await _seed_agent(session)
        with pytest.raises(DbIntegrityError):
            await _upsert(session, v5_agent.agent_id, bench_version=5, composite=1.001)

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
    n: int = 20,
    normalized_source_hash: str | None = None,
    prompt_fingerprint: dict | None = None,
    code_embedding: list | None = None,
    code_embed_model: str | None = None,
    coldkey: str | None = None,
    name: str = "agent",
) -> Agent:
    """Seed one agent + its score row, in the given lifecycle state."""
    agent = Agent(
        agent_id=uuid4(),
        miner_hotkey=miner,
        name=name,
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
        if coldkey is not None:
            session.add(
                EvaluationPayment(
                    block_hash=f"0x{agent.agent_id.hex}",
                    extrinsic_index=0,
                    agent_id=agent.agent_id,
                    miner_hotkey=miner,
                    miner_coldkey=coldkey,
                    amount_rao=1,
                    dest_address="5Destination",
                    timestamp=created_at,
                )
            )
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
            n=n,
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

    async def test_best_agent_per_coldkey_across_hotkeys_and_names(
        self, session: AsyncSession
    ) -> None:
        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        coldkey = "5ColdkeyOwner"
        await _seed_scored(
            session,
            miner=_MINER,
            coldkey=coldkey,
            name="mnemo-v4",
            composite=0.91,
            created_at=t0,
            n=MIN_ELIGIBLE_CASES,
        )
        best = await _seed_scored(
            session,
            miner=_MINER_B,
            coldkey=coldkey,
            name="mnemo-v5",
            composite=0.95,
            created_at=t0.replace(hour=13),
            n=MIN_ELIGIBLE_CASES,
        )

        ledger = await list_eligible_ledger(session)

        assert len(ledger) == 1
        assert ledger[0].agent_id == best.agent_id
        assert ledger[0].miner_hotkey == _MINER_B
        assert ledger[0].miner_coldkey == coldkey

    async def test_newer_lower_score_does_not_shadow_coldkey_best(
        self, session: AsyncSession
    ) -> None:
        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        coldkey = "5ColdkeyOwner"
        best = await _seed_scored(
            session,
            miner=_MINER,
            coldkey=coldkey,
            composite=0.95,
            created_at=t0,
            n=MIN_ELIGIBLE_CASES,
        )
        await _seed_scored(
            session,
            miner=_MINER_B,
            coldkey=coldkey,
            composite=0.90,
            created_at=t0.replace(hour=13),
            n=MIN_ELIGIBLE_CASES,
        )

        ledger = await list_eligible_ledger(session)

        assert [row.agent_id for row in ledger] == [best.agent_id]

    async def test_different_coldkeys_keep_separate_positions(
        self, session: AsyncSession
    ) -> None:
        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        await _seed_scored(
            session,
            miner=_MINER,
            coldkey="5ColdkeyA",
            composite=0.95,
            created_at=t0,
            n=MIN_ELIGIBLE_CASES,
        )
        await _seed_scored(
            session,
            miner=_MINER_B,
            coldkey="5ColdkeyB",
            composite=0.90,
            created_at=t0,
            n=MIN_ELIGIBLE_CASES,
        )

        assert len(await list_eligible_ledger(session)) == 2

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

    async def test_eligible_flag_reflects_case_count(
        self, session: AsyncSession
    ) -> None:
        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        await _seed_scored(
            session, miner=_MINER, composite=0.5, created_at=t0, n=MIN_ELIGIBLE_CASES
        )
        await _seed_scored(session, miner=_MINER_B, composite=0.5, created_at=t0, n=12)
        by_miner = {e.miner_hotkey: e for e in await list_eligible_ledger(session)}
        assert by_miner[_MINER].eligible is True
        assert by_miner[_MINER_B].eligible is False

    async def test_eligible_ranked_above_higher_provisional(
        self, session: AsyncSession
    ) -> None:
        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        # A full run at a *lower* composite must outrank a provisional smoke run
        # at a higher composite — the whole point of the gate.
        await _seed_scored(
            session, miner=_MINER, composite=0.55, created_at=t0, n=MIN_ELIGIBLE_CASES
        )
        await _seed_scored(session, miner=_MINER_B, composite=0.90, created_at=t0, n=12)
        ledger = await list_eligible_ledger(session)
        assert [e.miner_hotkey for e in ledger] == [_MINER, _MINER_B]
        assert ledger[0].eligible is True and ledger[1].eligible is False

    async def test_full_run_not_shadowed_by_inflated_small(
        self, session: AsyncSession
    ) -> None:
        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        # Same miner: a real full run (lower composite) and a lucky small run
        # (higher composite). The miner must be represented by the *full* run so
        # the small one cannot both hide the full result and be dropped by the
        # emission gate.
        await _seed_scored(
            session, miner=_MINER, composite=0.52, created_at=t0, n=MIN_ELIGIBLE_CASES
        )
        await _seed_scored(
            session,
            miner=_MINER,
            composite=0.95,
            created_at=t0.replace(hour=13),
            n=12,
        )
        ledger = await list_eligible_ledger(session)
        assert len(ledger) == 1
        assert ledger[0].eligible is True
        assert ledger[0].composite == 0.52

    async def test_zero_composite_full_run_is_not_ranked(
        self, session: AsyncSession
    ) -> None:
        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        # A full run that scored 0.000 must NOT be ranked or crowned: the validator
        # drops composite <= 0 from the fold, so a failed full run earns nothing and
        # must never outrank a positive provisional run on the board.
        await _seed_scored(
            session, miner=_MINER, composite=0.0, created_at=t0, n=MIN_ELIGIBLE_CASES
        )
        await _seed_scored(session, miner=_MINER_B, composite=0.9, created_at=t0, n=12)
        ledger = await list_eligible_ledger(session)
        by_miner = {e.miner_hotkey: e for e in ledger}
        assert by_miner[_MINER].eligible is False  # full but zero-scoring
        assert by_miner[_MINER_B].eligible is False  # provisional smoke run
        # Neither ranks; the positive provisional sorts above the zero full run.
        assert [e.miner_hotkey for e in ledger] == [_MINER_B, _MINER]

    async def test_positive_full_run_ranked_above_zero_full_run(
        self, session: AsyncSession
    ) -> None:
        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        # Two full runs, one positive and one zero: only the positive one is ranked.
        await _seed_scored(
            session, miner=_MINER, composite=0.0, created_at=t0, n=MIN_ELIGIBLE_CASES
        )
        await _seed_scored(
            session, miner=_MINER_B, composite=0.4, created_at=t0, n=MIN_ELIGIBLE_CASES
        )
        ledger = await list_eligible_ledger(session)
        by_miner = {e.miner_hotkey: e for e in ledger}
        assert by_miner[_MINER_B].eligible is True
        assert by_miner[_MINER].eligible is False
        assert ledger[0].miner_hotkey == _MINER_B

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
        # An agent scored by the k=3 validator pool has three score rows. The
        # ledger must return the MEDIAN row whole (composite, seed, run_id,
        # signature, validator_hotkey all from the same physical row), not a
        # per-column aggregate that stitches a mismatched (composite, signature).
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
            # Median composite from a third validator: this is the row the
            # ledger must surface whole under median-of-3.
            await upsert_score(
                session,
                agent_id=agent.agent_id,
                validator_hotkey="5Mmm_validator",
                run_id="run_mid",
                seed=333,
                composite=0.80,
                tool_mean=0.8,
                memory_mean=0.8,
                median_ms=500,
                n=20,
                generated_at=_GEN_AT,
                signature="55" * 64,
            )
        ledger = await list_eligible_ledger(session)
        assert len(ledger) == 1
        e = ledger[0]
        # Every field must come from the MEDIAN row (composite 0.80), whole.
        assert e.composite == pytest.approx(0.80)
        assert e.seed == 333
        assert e.run_id == "run_mid"
        assert e.signature == "55" * 64
        assert e.validator_hotkey == "5Mmm_validator"

    async def test_median_of_three_ignores_outlier_validator(
        self, session: AsyncSession
    ) -> None:
        # Three validators score the same agent. The canonical score is the
        # MEDIAN (0.55), so a single generous (0.99) or harsh (0.10) validator
        # cannot move it — and it is the median, not the mean (~0.547).
        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        agent = Agent(
            agent_id=uuid4(),
            miner_hotkey=_MINER,
            name="agent",
            sha256="cd" * 32,
            status=AgentStatus.SCORED,
            created_at=t0,
        )
        async with session.begin():
            session.add(agent)
            await session.flush()
            for vh, comp in (("5A_v", 0.10), ("5B_v", 0.55), ("5C_v", 0.99)):
                await upsert_score(
                    session,
                    agent_id=agent.agent_id,
                    validator_hotkey=vh,
                    run_id=f"run_{vh}",
                    seed=1,
                    composite=comp,
                    tool_mean=comp,
                    memory_mean=comp,
                    median_ms=1,
                    n=MIN_ELIGIBLE_CASES,
                    generated_at=_GEN_AT,
                )
        ledger = await list_eligible_ledger(session)
        assert len(ledger) == 1
        assert ledger[0].composite == pytest.approx(0.55)
        assert ledger[0].eligible is True


class TestListScoresForBenchVersion:
    async def test_filters_by_first_class_version_not_advisory_details(
        self, session: AsyncSession
    ) -> None:
        from ditto.db.queries.scores import list_scores_for_bench_version

        agent = await _seed_agent(session)
        # First-class v1 wins even if other rows' advisory details say v2/null.
        await _upsert(
            session,
            agent.agent_id,
            validator_hotkey="5V1",
            run_id="r1",
            bench_version=1,
            details={"bench_version": 1, "per_case": [{"expected": ["x"]}]},
        )
        await _upsert(
            session,
            agent.agent_id,
            validator_hotkey="5V2",
            run_id="r2",
            details={"bench_version": 2},
        )
        await _upsert(
            session, agent.agent_id, validator_hotkey="5V3", run_id="r3", details=None
        )

        rows, total = await list_scores_for_bench_version(session, version=1)
        assert total == 1
        assert len(rows) == 1
        score, miner = rows[0]
        assert score.run_id == "r1"
        assert miner == _MINER
        # The unredacted answer key rides along for the (retired) corpus.
        assert score.details is not None
        assert score.details["per_case"][0]["expected"] == ["x"]


_ROLLOUT_FROM = 2
_ROLLOUT_DESIRED = 4
_QUORUM_VALIDATORS = ("5Va", "5Vb", "5Vc")


async def _open_rollout(session: AsyncSession) -> None:
    """A collecting v2 -> v4 rollout, the state the threshold rule governs."""
    from ditto.db.models import BenchmarkRollout

    async with session.begin():
        session.add(
            BenchmarkRollout(
                rollout_id=uuid4(),
                from_version=_ROLLOUT_FROM,
                desired_version=_ROLLOUT_DESIRED,
                status="collecting",
                cohort_size=5,
                created_at=datetime.now(UTC),
            )
        )


async def _seed_versioned_agent(
    session: AsyncSession,
    *,
    miner: str,
    created_at: datetime,
    v2_composite: float,
    desired_composite: float | None = None,
    desired_samples: int = 3,
    desired_n: int = MIN_ELIGIBLE_CASES,
    status: AgentStatus = AgentStatus.SCORED,
    coldkey: str | None = None,
) -> Agent:
    """One agent with a full v2 quorum and an optional partial/full v4 quorum."""
    agent = Agent(
        agent_id=uuid4(),
        miner_hotkey=miner,
        name="agent",
        sha256="ab" * 32,
        size_bytes=524288,
        status=status,
        created_at=created_at,
    )
    async with session.begin():
        session.add(agent)
        await session.flush()
        if coldkey is not None:
            session.add(
                EvaluationPayment(
                    block_hash=f"0x{agent.agent_id.hex}",
                    extrinsic_index=0,
                    agent_id=agent.agent_id,
                    miner_hotkey=miner,
                    miner_coldkey=coldkey,
                    amount_rao=1,
                    dest_address="5Destination",
                    timestamp=created_at,
                )
            )
        for index, validator in enumerate(_QUORUM_VALIDATORS):
            await upsert_score(
                session,
                agent_id=agent.agent_id,
                validator_hotkey=validator,
                bench_version=_ROLLOUT_FROM,
                run_id=f"v2-{miner}-{index}",
                seed=index,
                composite=v2_composite,
                tool_mean=v2_composite,
                memory_mean=v2_composite,
                median_ms=500,
                n=MIN_ELIGIBLE_CASES,
                generated_at=_GEN_AT,
                signature="ab" * 64,
            )
        if desired_composite is not None:
            for index, validator in enumerate(_QUORUM_VALIDATORS[:desired_samples]):
                await upsert_score(
                    session,
                    agent_id=agent.agent_id,
                    validator_hotkey=validator,
                    bench_version=_ROLLOUT_DESIRED,
                    run_id=f"v4-{miner}-{index}",
                    seed=index,
                    composite=desired_composite,
                    tool_mean=desired_composite,
                    memory_mean=desired_composite,
                    median_ms=500,
                    n=desired_n,
                    generated_at=_GEN_AT,
                    signature="cd" * 64,
                )
    return agent


def _miner(index: int) -> str:
    return f"5Miner{index:048d}"


class TestThresholdGatedAuthority:
    """Ledger authority flips to the desired version only past the threshold.

    The whole ledger is on ONE version at a time; the flip point is
    ``MIN_DESIRED_AUTHORITY_AGENTS`` (the KOTH champion + tail), so the emission
    set never loses recipients mid-rollout.
    """

    def test_min_desired_authority_matches_koth_recipients(self) -> None:
        # Drift guard: the threshold IS the KOTH recipient count. The constant is
        # spelled out in benchmark_rollout (importing api_server there is a
        # cycle), so this asserts the derivation it documents.
        from ditto.api_server.koth import KOTH_TAIL_SIZE
        from ditto.db.queries.benchmark_rollout import MIN_DESIRED_AUTHORITY_AGENTS

        assert MIN_DESIRED_AUTHORITY_AGENTS == 1 + KOTH_TAIL_SIZE

    @staticmethod
    def _assert_single_version(ledger: list, expected: int) -> None:
        versions = {row.bench_version for row in ledger}
        assert versions == {expected}, f"mixed authority versions: {versions}"

    async def test_below_threshold_keeps_active_version_for_every_agent(
        self, session: AsyncSession
    ) -> None:
        from ditto.db.queries.benchmark_rollout import MIN_DESIRED_AUTHORITY_AGENTS

        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        await _open_rollout(session)
        # Five agents; only four hold a full v4 quorum — one short of the flip.
        for index in range(MIN_DESIRED_AUTHORITY_AGENTS):
            await _seed_versioned_agent(
                session,
                miner=_miner(index),
                created_at=t0,
                v2_composite=0.50 + index / 100,
                desired_composite=(
                    0.80 if index < MIN_DESIRED_AUTHORITY_AGENTS - 1 else None
                ),
            )
        ledger = await list_eligible_ledger(session)
        # Every agent still ranked on v2, and the emission set is still full.
        self._assert_single_version(ledger, _ROLLOUT_FROM)
        assert len(ledger) == MIN_DESIRED_AUTHORITY_AGENTS
        assert all(row.eligible for row in ledger)
        assert sorted(row.composite for row in ledger) == pytest.approx(
            [0.50 + index / 100 for index in range(MIN_DESIRED_AUTHORITY_AGENTS)]
        )

    async def test_at_threshold_flips_whole_ledger_to_desired_version(
        self, session: AsyncSession
    ) -> None:
        from ditto.db.queries.benchmark_rollout import MIN_DESIRED_AUTHORITY_AGENTS

        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        await _open_rollout(session)
        for index in range(MIN_DESIRED_AUTHORITY_AGENTS):
            await _seed_versioned_agent(
                session,
                miner=_miner(index),
                created_at=t0,
                v2_composite=0.50 + index / 100,
                desired_composite=0.70 + index / 100,
            )
        ledger = await list_eligible_ledger(session)
        self._assert_single_version(ledger, _ROLLOUT_DESIRED)
        assert len(ledger) == MIN_DESIRED_AUTHORITY_AGENTS
        assert all(row.eligible for row in ledger)
        assert ledger[0].composite == pytest.approx(
            0.70 + (MIN_DESIRED_AUTHORITY_AGENTS - 1) / 100
        )

    async def test_duplicate_coldkey_cannot_tip_rollout_threshold(
        self, session: AsyncSession
    ) -> None:
        from ditto.db.queries.benchmark_rollout import MIN_DESIRED_AUTHORITY_AGENTS

        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        await _open_rollout(session)
        for index in range(MIN_DESIRED_AUTHORITY_AGENTS):
            await _seed_versioned_agent(
                session,
                miner=_miner(index),
                coldkey=("5SharedColdkey" if index < 2 else f"5Coldkey{index:048d}"),
                created_at=t0,
                v2_composite=0.50 + index / 100,
                desired_composite=0.70 + index / 100,
            )

        ledger = await list_eligible_ledger(session)

        self._assert_single_version(ledger, _ROLLOUT_FROM)
        assert len(ledger) == MIN_DESIRED_AUTHORITY_AGENTS - 1

    async def test_agent_without_desired_quorum_drops_out_after_flip(
        self, session: AsyncSession
    ) -> None:
        from ditto.db.queries.benchmark_rollout import MIN_DESIRED_AUTHORITY_AGENTS

        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        await _open_rollout(session)
        for index in range(MIN_DESIRED_AUTHORITY_AGENTS):
            await _seed_versioned_agent(
                session,
                miner=_miner(index),
                created_at=t0,
                v2_composite=0.50,
                desired_composite=0.70,
            )
        laggard = await _seed_versioned_agent(
            session,
            miner=_miner(99),
            created_at=t0,
            v2_composite=0.99,
            desired_composite=None,
        )
        ledger = await list_eligible_ledger(session)
        # Intended: a v2-only agent has no authoritative row once the pool is on
        # v4, however high its v2 composite was. That is why the threshold waits
        # for a full emission set first.
        self._assert_single_version(ledger, _ROLLOUT_DESIRED)
        assert laggard.agent_id not in {row.agent_id for row in ledger}
        assert len(ledger) == MIN_DESIRED_AUTHORITY_AGENTS

    @pytest.mark.parametrize("samples", [1, 2])
    async def test_partial_desired_samples_never_count_toward_threshold(
        self, session: AsyncSession, samples: int
    ) -> None:
        from ditto.db.queries.benchmark_rollout import MIN_DESIRED_AUTHORITY_AGENTS

        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        await _open_rollout(session)
        # Every agent has an incomplete v4 sample: 1/3 or 2/3 is not a quorum.
        for index in range(MIN_DESIRED_AUTHORITY_AGENTS + 2):
            await _seed_versioned_agent(
                session,
                miner=_miner(index),
                created_at=t0,
                v2_composite=0.50,
                desired_composite=0.90,
                desired_samples=samples,
            )
        ledger = await list_eligible_ledger(session)
        self._assert_single_version(ledger, _ROLLOUT_FROM)
        assert all(row.composite == pytest.approx(0.50) for row in ledger)

    async def test_ineligible_desired_run_does_not_count_toward_threshold(
        self, session: AsyncSession
    ) -> None:
        from ditto.db.queries.benchmark_rollout import MIN_DESIRED_AUTHORITY_AGENTS

        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        await _open_rollout(session)
        for index in range(MIN_DESIRED_AUTHORITY_AGENTS - 1):
            await _seed_versioned_agent(
                session,
                miner=_miner(index),
                created_at=t0,
                v2_composite=0.50,
                desired_composite=0.70,
            )
        # A full 3/3 v4 quorum on a SMOKE run (below the full-benchmark floor).
        # It is unranked, so it must not be the agent that tips the ledger over.
        await _seed_versioned_agent(
            session,
            miner=_miner(50),
            created_at=t0,
            v2_composite=0.50,
            desired_composite=0.95,
            desired_n=MIN_ELIGIBLE_CASES - 1,
        )
        # Same for a held agent: it is outside the eligible pool entirely.
        await _seed_versioned_agent(
            session,
            miner=_miner(51),
            created_at=t0,
            v2_composite=0.50,
            desired_composite=0.95,
            status=AgentStatus.ATH_PENDING_REVIEW,
        )
        ledger = await list_eligible_ledger(session)
        self._assert_single_version(ledger, _ROLLOUT_FROM)
        assert len(ledger) == MIN_DESIRED_AUTHORITY_AGENTS

    async def test_explicit_bench_version_request_is_unaffected(
        self, session: AsyncSession
    ) -> None:
        from ditto.db.queries.benchmark_rollout import MIN_DESIRED_AUTHORITY_AGENTS

        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        await _open_rollout(session)
        # Below the threshold, so the default read is pinned to v2 — but an
        # explicit historical view must still return exactly what it asked for.
        for index in range(MIN_DESIRED_AUTHORITY_AGENTS - 1):
            await _seed_versioned_agent(
                session,
                miner=_miner(index),
                created_at=t0,
                v2_composite=0.50,
                desired_composite=0.70,
            )
        default_ledger = await list_eligible_ledger(session)
        self._assert_single_version(default_ledger, _ROLLOUT_FROM)

        desired_view = await list_eligible_ledger(
            session, bench_version=_ROLLOUT_DESIRED
        )
        self._assert_single_version(desired_view, _ROLLOUT_DESIRED)
        assert len(desired_view) == MIN_DESIRED_AUTHORITY_AGENTS - 1
        assert all(row.composite == pytest.approx(0.70) for row in desired_view)

        active_view = await list_eligible_ledger(session, bench_version=_ROLLOUT_FROM)
        self._assert_single_version(active_view, _ROLLOUT_FROM)

    async def test_no_open_rollout_keeps_plain_highest_version_behaviour(
        self, session: AsyncSession
    ) -> None:
        t0 = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
        # No rollout row at all: desired_version is None and nothing changes.
        await _seed_versioned_agent(
            session, miner=_miner(0), created_at=t0, v2_composite=0.50
        )
        ledger = await list_eligible_ledger(session)
        self._assert_single_version(ledger, _ROLLOUT_FROM)
        assert ledger[0].composite == pytest.approx(0.50)


class TestQuorumComposites:
    async def test_returns_every_composite_per_agent(
        self, session: AsyncSession
    ) -> None:
        # Three validators score one agent on three different seeds; the query
        # returns all three composites so the ledger can compute their SEM.
        agent = await _seed_agent(session)
        for vh, comp in (("5V1", 0.80), ("5V2", 0.85), ("5V3", 0.90)):
            await _upsert(
                session, agent.agent_id, validator_hotkey=vh, run_id=vh, composite=comp
            )
        out = await quorum_composites(session, [agent.agent_id])
        assert sorted(out[agent.agent_id]) == pytest.approx([0.80, 0.85, 0.90])

    async def test_empty_ids_returns_empty(self, session: AsyncSession) -> None:
        assert await quorum_composites(session, []) == {}

    async def test_unknown_agent_absent(self, session: AsyncSession) -> None:
        assert await quorum_composites(session, [uuid4()]) == {}
