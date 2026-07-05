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
    content_fingerprint: list[str] | None = None,
) -> LedgerRow:
    return LedgerRow(
        miner_hotkey=miner,
        agent_id=uuid4(),
        composite=composite,
        tool_mean=composite,
        memory_mean=composite,
        first_seen=_FIRST_SEEN,
        sha256=sha256,
        size_bytes=size_bytes,
        run_id="run_1",
        seed=42,
        validator_hotkey="5Validator",
        signature="ab" * 64,
        status=AgentStatus.SCORED,
        content_fingerprint=content_fingerprint,
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

    def test_content_dup_held_when_size_drifts(self) -> None:
        # The A1 gap: a re-indented/renamed copy whose byte size moved past the
        # size tolerance, but whose normalized content is all but identical.
        shared = [f"h{i}" for i in range(20)]
        incumbent = _entry(
            composite=0.80,
            sha256="aa" * 32,
            size_bytes=500000,
            content_fingerprint=shared,
        )
        # One file edited (intersection 19, union 21 => 0.905 >= 0.90) and the
        # tarball size drifted 100 KiB past the size rule.
        copy_fp = [f"h{i}" for i in range(19)] + ["tweaked"]
        decision = evaluate_antidup(
            agent_id=uuid4(),
            miner_hotkey="5Copier",
            sha256="bb" * 32,
            composite=0.805,
            size_bytes=600000,  # well past the 8 KiB size tolerance
            content_fingerprint=copy_fp,
            eligible=[incumbent],
        )
        assert decision.held is True
        assert decision.duplicate_of == incumbent.agent_id
        assert "content near-duplicate" in (decision.reason or "")

    def test_distinct_content_not_held_despite_close_score(self) -> None:
        # Two independent harnesses that only share reference scaffolding: close
        # score but low fingerprint overlap => a genuine competitor, not a copy.
        incumbent = _entry(
            composite=0.80,
            sha256="aa" * 32,
            size_bytes=500000,
            content_fingerprint=[f"h{i}" for i in range(20)],
        )
        decision = evaluate_antidup(
            agent_id=uuid4(),
            miner_hotkey="5Challenger",
            sha256="bb" * 32,
            composite=0.805,
            size_bytes=900000,  # different size, so the size rule can't fire either
            content_fingerprint=[f"h{i}" for i in range(5)]
            + [f"x{i}" for i in range(15)],
            eligible=[incumbent],
        )
        assert decision.held is False

    def test_content_dup_ignored_when_score_far(self) -> None:
        # Near-identical content but a large score gap: outside score_tol, so the
        # content rule does not fire (a real improvement, not a copy).
        shared = [f"h{i}" for i in range(20)]
        incumbent = _entry(
            composite=0.80,
            sha256="aa" * 32,
            size_bytes=500000,
            content_fingerprint=shared,
        )
        decision = evaluate_antidup(
            agent_id=uuid4(),
            miner_hotkey="5Challenger",
            sha256="bb" * 32,
            composite=0.90,
            size_bytes=505000,
            content_fingerprint=shared,
            eligible=[incumbent],
        )
        assert decision.held is False

    def test_same_miner_content_dup_not_held(self) -> None:
        # A miner iterating on their own harness shares content with themselves —
        # never a copier, so the content rule must skip same-miner entries.
        shared = [f"h{i}" for i in range(20)]
        incumbent = _entry(
            composite=0.80, miner="5Mine", sha256="aa" * 32, content_fingerprint=shared
        )
        decision = evaluate_antidup(
            agent_id=uuid4(),
            miner_hotkey="5Mine",
            sha256="bb" * 32,
            composite=0.805,
            size_bytes=600000,
            content_fingerprint=shared,
            eligible=[incumbent],
        )
        assert decision.held is False

    def test_missing_fingerprints_fall_back_to_size_rule(self) -> None:
        # No fingerprints anywhere (legacy rows): the content rule is inert
        # (jaccard 0) and the size rule still catches a same-size near-dup.
        incumbent = _entry(composite=0.80, sha256="aa" * 32, size_bytes=500000)
        decision = evaluate_antidup(
            agent_id=uuid4(),
            miner_hotkey="5Copier",
            sha256="bb" * 32,
            composite=0.805,
            size_bytes=500100,
            content_fingerprint=None,
            eligible=[incumbent],
        )
        assert decision.held is True
        assert "near-duplicate" in (decision.reason or "")

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
            tool_mean=0.80,
            memory_mean=0.80,
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
