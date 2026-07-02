"""Unit tests for the pure anti-copy gate :mod:`ditto.api_server.scoring_gate`."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from ditto.api_models.agent_status import AgentStatus
from ditto.api_server.scoring_gate import evaluate_antidup
from ditto.db.queries.scores import LedgerRow

_FIRST_SEEN = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)


def _entry(
    *,
    composite: float,
    miner: str = "5Incumbent",
    sha256: str = "aa" * 32,
    size_bytes: int | None = 524288,
) -> LedgerRow:
    return LedgerRow(
        miner_hotkey=miner,
        agent_id=uuid4(),
        composite=composite,
        first_seen=_FIRST_SEEN,
        sha256=sha256,
        size_bytes=size_bytes,
        run_id="run_1",
        seed=42,
        validator_hotkey="5Validator",
        signature="ab" * 64,
        status=AgentStatus.SCORED,
    )


class TestEvaluateAntidup:
    def test_clean_submission_not_held(self) -> None:
        incumbent = _entry(composite=0.70, sha256="aa" * 32, size_bytes=500000)
        # A genuine improvement from another miner: far higher, different size+hash.
        decision = evaluate_antidup(
            agent_id=uuid4(),
            miner_hotkey="5Challenger",
            sha256="bb" * 32,
            composite=0.85,
            size_bytes=700000,
            eligible=[incumbent],
        )
        assert decision.held is False
        assert decision.duplicate_of is None

    def test_exact_sha256_copy_is_held(self) -> None:
        incumbent = _entry(composite=0.70, sha256="cc" * 32, size_bytes=500000)
        decision = evaluate_antidup(
            agent_id=uuid4(),
            miner_hotkey="5Copier",
            sha256="cc" * 32,  # byte-identical resubmission
            composite=0.70,
            size_bytes=500000,
            eligible=[incumbent],
        )
        assert decision.held is True
        assert decision.duplicate_of == incumbent.agent_id
        assert "sha256" in (decision.reason or "")

    def test_near_dup_from_other_miner_is_held(self) -> None:
        incumbent = _entry(composite=0.80, sha256="aa" * 32, size_bytes=500000)
        # Another miner, different bytes, near-identical size + score.
        decision = evaluate_antidup(
            agent_id=uuid4(),
            miner_hotkey="5Copier",
            sha256="bb" * 32,
            composite=0.805,
            size_bytes=500100,
            eligible=[incumbent],
        )
        assert decision.held is True
        assert decision.duplicate_of == incumbent.agent_id
        assert "near-duplicate" in (decision.reason or "")

    def test_large_improvement_not_held(self) -> None:
        incumbent = _entry(composite=0.80, sha256="aa" * 32, size_bytes=500000)
        # Same size but a big score jump => a real improvement, not a copy.
        decision = evaluate_antidup(
            agent_id=uuid4(),
            miner_hotkey="5Challenger",
            sha256="bb" * 32,
            composite=0.90,
            size_bytes=500000,
            eligible=[incumbent],
        )
        assert decision.held is False

    def test_distant_score_not_held(self) -> None:
        incumbent = _entry(composite=0.80, sha256="aa" * 32, size_bytes=500000)
        # Similar size but score gap > tol => not a near-dup.
        decision = evaluate_antidup(
            agent_id=uuid4(),
            miner_hotkey="5Challenger",
            sha256="bb" * 32,
            composite=0.75,
            size_bytes=500050,
            eligible=[incumbent],
        )
        assert decision.held is False

    def test_same_miner_improvement_not_held(self) -> None:
        # A miner iterating on THEIR OWN eligible agent (near-identical size, small
        # score bump, different bytes) is not a copier — must not be held.
        incumbent = _entry(
            composite=0.80, miner="5Mine", sha256="aa" * 32, size_bytes=500000
        )
        decision = evaluate_antidup(
            agent_id=uuid4(),
            miner_hotkey="5Mine",  # same miner
            sha256="bb" * 32,
            composite=0.81,
            size_bytes=500200,
            eligible=[incumbent],
        )
        assert decision.held is False

    def test_copy_not_masked_by_unrelated_midscorer(self) -> None:
        # A genuine unrelated agent scoring between the copied agent and the copy
        # must not let the copy escape the gate (the false-negative A2).
        original = _entry(
            composite=0.80, miner="5Orig", sha256="aa" * 32, size_bytes=500000
        )
        midscorer = _entry(
            composite=0.804, miner="5Mid", sha256="dd" * 32, size_bytes=900000
        )
        decision = evaluate_antidup(
            agent_id=uuid4(),
            miner_hotkey="5Copier",
            sha256="bb" * 32,
            composite=0.805,
            size_bytes=500050,  # matches the original's size, not the midscorer's
            eligible=[midscorer, original],
        )
        assert decision.held is True
        assert decision.duplicate_of == original.agent_id

    def test_missing_sizes_skip_near_dup(self) -> None:
        incumbent = _entry(composite=0.80, sha256="aa" * 32, size_bytes=None)
        decision = evaluate_antidup(
            agent_id=uuid4(),
            miner_hotkey="5Copier",
            sha256="bb" * 32,
            composite=0.805,
            size_bytes=None,
            eligible=[incumbent],
        )
        assert decision.held is False

    def test_deterministic_and_self_excluded(self) -> None:
        # An entry with the same agent_id is the agent itself (re-score): neither
        # rule may match it. Verdict is repeatable.
        me = uuid4()
        entry = LedgerRow(
            miner_hotkey="5Me",
            agent_id=me,
            composite=0.80,
            first_seen=_FIRST_SEEN,
            sha256="dd" * 32,
            size_bytes=500000,
            run_id="run_1",
            seed=42,
            validator_hotkey="5Validator",
            signature=None,
            status=AgentStatus.SCORED,
        )
        d1 = evaluate_antidup(
            agent_id=me,
            miner_hotkey="5Me",
            sha256="dd" * 32,
            composite=0.80,
            size_bytes=500000,
            eligible=[entry],
        )
        d2 = evaluate_antidup(
            agent_id=me,
            miner_hotkey="5Me",
            sha256="dd" * 32,
            composite=0.80,
            size_bytes=500000,
            eligible=[entry],
        )
        assert d1.held is False
        assert d1 == d2
