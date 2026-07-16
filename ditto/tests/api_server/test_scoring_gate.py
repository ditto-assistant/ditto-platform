"""Unit tests for the pure anti-copy gate :mod:`ditto.api_server.scoring_gate`."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from ditto.api_models.agent_status import AgentStatus
from ditto.api_server.fingerprint import (
    _FP_NOVELTY_VERSION,
    _FP_VERSION,
    _MINHASH_K,
    _PROMPT_VERSION,
)
from ditto.api_server.scoring_gate import evaluate_duplicate_signals
from ditto.db.queries.scores import LedgerRow

_FIRST_SEEN = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)


def _sk(shingles: set[str]) -> dict:
    """Build a fingerprint sketch from a set of shingle hashes.

    The sets here are far smaller than the bottom-k budget, so the sketch is the
    whole set and :func:`content_similarity` computes Jaccard/containment exactly —
    letting these gate tests assert on precise thresholds.
    """
    return {
        "v": _FP_VERSION,
        "k": _MINHASH_K,
        "card": len(shingles),
        "m": sorted(shingles)[:_MINHASH_K],
    }


def _nsk(shingles: set[str], *, baseline_id: str = "c8d712f1addf") -> dict:
    """Build a NOVELTY sketch (baseline-subtracted, fingerprint v2)."""
    return {
        "v": _FP_NOVELTY_VERSION,
        "k": _MINHASH_K,
        "card": len(shingles),
        "m": sorted(shingles)[:_MINHASH_K],
        "bl": baseline_id,
    }


def _psk(shingles: set[str]) -> dict:
    """Build an prompt sketch (version ``"p1"``) from a set of shingle hashes."""
    return {
        "v": _PROMPT_VERSION,
        "k": _MINHASH_K,
        "card": len(shingles),
        "m": sorted(shingles)[:_MINHASH_K],
    }


def _entry(
    *,
    composite: float,
    miner: str = "5Incumbent",
    sha256: str = "aa" * 32,
    size_bytes: int | None = 524288,
    content_fingerprint: dict | None = None,
    structural_fingerprint: dict | None = None,
    normalized_source_hash: str | None = None,
    prompt_fingerprint: dict | None = None,
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
        structural_fingerprint=structural_fingerprint,
        normalized_source_hash=normalized_source_hash,
        prompt_fingerprint=prompt_fingerprint,
    )


class TestEvaluateAntidup:
    def test_clean_submission_not_held(self) -> None:
        incumbent = _entry(composite=0.70, sha256="aa" * 32, size_bytes=500000)
        # A genuine improvement from another miner: far higher, different size+hash.
        decision = evaluate_duplicate_signals(
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
        decision = evaluate_duplicate_signals(
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

    def test_exact_repack_normalized_hash_is_held(self) -> None:
        # Different bytes (sha256) but the same canonicalized source: a reformat /
        # re-comment / file-reorder repack. Held on the hash equality alone, with
        # no score proximity (a distant score must not save it).
        incumbent = _entry(
            composite=0.60,
            sha256="cc" * 32,
            normalized_source_hash="ns" * 32,
        )
        decision = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Copier",
            sha256="dd" * 32,  # repackaged bytes differ
            composite=0.95,  # far from incumbent — proximity is irrelevant here
            size_bytes=999999,
            normalized_source_hash="ns" * 32,
            eligible=[incumbent],
        )
        assert decision.held is True
        assert decision.duplicate_of == incumbent.agent_id
        assert "repack" in (decision.reason or "")

    def test_same_miner_repack_not_held(self) -> None:
        # A miner re-uploading their OWN agent (same normalized hash) is iterating,
        # not copying — the other-miner filter must exempt it.
        own = _entry(
            composite=0.60,
            miner="5Self",
            sha256="cc" * 32,
            normalized_source_hash="ns" * 32,
        )
        decision = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Self",
            sha256="dd" * 32,
            composite=0.61,
            size_bytes=500000,
            normalized_source_hash="ns" * 32,
            eligible=[own],
        )
        assert decision.held is False

    def test_null_normalized_hash_never_matches(self) -> None:
        # An incumbent with no stored hash (uploaded before the normalized-source hash /
        # unreadable
        # tarball) must not match a challenger that also lacks one — null is
        # "no repack match", never a hit against null.
        incumbent = _entry(
            composite=0.60, sha256="cc" * 32, normalized_source_hash=None
        )
        decision = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Challenger",
            sha256="dd" * 32,
            composite=0.85,
            size_bytes=700000,
            normalized_source_hash=None,
            eligible=[incumbent],
        )
        assert decision.held is False

    def test_near_dup_from_other_miner_is_held(self) -> None:
        incumbent = _entry(composite=0.80, sha256="aa" * 32, size_bytes=500000)
        # Another miner, different bytes, near-identical size + score.
        decision = evaluate_duplicate_signals(
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
        decision = evaluate_duplicate_signals(
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
        decision = evaluate_duplicate_signals(
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
        decision = evaluate_duplicate_signals(
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
        decision = evaluate_duplicate_signals(
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
        # A reformatted/locally-edited copy whose byte size moved past the size
        # tolerance, but whose shingle sketch is all but identical.
        shared = {f"{i:016x}" for i in range(20)}
        incumbent = _entry(
            composite=0.80,
            sha256="aa" * 32,
            size_bytes=500000,
            content_fingerprint=_sk(shared),
        )
        # 19 of 20 shingles shared (Jaccard 19/21 = 0.905 >= 0.75) and the tarball
        # size drifted 100 KiB past the size rule.
        copy_fp = _sk({f"{i:016x}" for i in range(19)} | {"ff" * 8})
        decision = evaluate_duplicate_signals(
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

    def test_padding_held_by_containment(self) -> None:
        # A verbatim copy padded with junk files dilutes Jaccard below the tol but
        # stays fully contained => the containment arm of rule 2 holds it.
        shared = {f"{i:016x}" for i in range(20)}
        incumbent = _entry(
            composite=0.80,
            sha256="aa" * 32,
            size_bytes=500000,
            content_fingerprint=_sk(shared),
        )
        padded = _sk(shared | {f"pad{i:013x}" for i in range(40)})  # jaccard 20/60
        decision = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Copier",
            sha256="bb" * 32,
            composite=0.805,
            size_bytes=900000,
            content_fingerprint=padded,
            eligible=[incumbent],
        )
        assert decision.held is True
        assert "containment" in (decision.reason or "")

    def test_structural_match_alone_never_holds(self) -> None:
        # The structural (AST) sketch is whole-crate, so on starter-kit-derived
        # submissions it saturates between independent miners exactly like the
        # pre-novelty lexical channel did. Until dittobench ships
        # baseline-subtracted structural sketches it corroborates, never
        # triggers — an identical AST shape with distinct lexical novelty stays
        # unheld.
        struct = {f"{i:016x}" for i in range(30)}
        incumbent = _entry(
            composite=0.80,
            sha256="aa" * 32,
            size_bytes=500000,
            content_fingerprint=_sk({f"lex{i:013x}" for i in range(30)}),
            structural_fingerprint=_sk(struct),
        )
        decision = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Copier",
            sha256="bb" * 32,
            composite=0.805,
            size_bytes=700000,
            content_fingerprint=_sk({f"other{i:011x}" for i in range(30)}),
            structural_fingerprint=_sk(struct),
            eligible=[incumbent],
        )
        assert decision.held is False

    def test_structural_overlap_annotates_lexical_hold(self) -> None:
        # When the lexical channel fires, a high structural overlap with the
        # matched agent is appended to the audit reason as corroboration.
        lex = {f"lex{i:013x}" for i in range(30)}
        struct = {f"{i:016x}" for i in range(30)}
        incumbent = _entry(
            composite=0.80,
            sha256="aa" * 32,
            size_bytes=500000,
            content_fingerprint=_sk(lex),
            structural_fingerprint=_sk(struct),
        )
        decision = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Copier",
            sha256="bb" * 32,
            composite=0.805,
            size_bytes=700000,
            content_fingerprint=_sk(lex),
            structural_fingerprint=_sk(struct),
            eligible=[incumbent],
        )
        assert decision.held is True
        assert "content near-duplicate" in (decision.reason or "")
        assert "structural jaccard" in (decision.reason or "")

    def test_structural_below_tol_not_held(self) -> None:
        # Two crates sharing reference-harness AST scaffolding but well under the
        # (high) structural threshold, and lexically distinct => not held.
        incumbent = _entry(
            composite=0.80,
            sha256="aa" * 32,
            size_bytes=500000,
            content_fingerprint=_sk({f"lex{i:013x}" for i in range(30)}),
            structural_fingerprint=_sk({f"{i:016x}" for i in range(30)}),
        )
        decision = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Challenger",
            sha256="bb" * 32,
            composite=0.805,
            size_bytes=900000,
            content_fingerprint=_sk({f"z{i:015x}" for i in range(30)}),
            # 15 of 45 shingles shared => Jaccard 0.33, containment 0.5: below the
            # 0.85 / 0.98 structural tolerances.
            structural_fingerprint=_sk(
                {f"{i:016x}" for i in range(15)} | {f"s{i:015x}" for i in range(15)}
            ),
            eligible=[incumbent],
        )
        assert decision.held is False

    def test_distinct_content_not_held_despite_close_score(self) -> None:
        # Two independent harnesses that only share reference scaffolding: close
        # score but low fingerprint overlap (5 of 30) => a genuine competitor.
        incumbent = _entry(
            composite=0.80,
            sha256="aa" * 32,
            size_bytes=500000,
            content_fingerprint=_sk({f"{i:016x}" for i in range(20)}),
        )
        decision = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Challenger",
            sha256="bb" * 32,
            composite=0.805,
            size_bytes=900000,  # different size, so the size rule can't fire either
            content_fingerprint=_sk(
                {f"{i:016x}" for i in range(5)} | {f"x{i:015x}" for i in range(15)}
            ),
            eligible=[incumbent],
        )
        assert decision.held is False

    def test_content_dup_ignored_when_score_far(self) -> None:
        # Near-identical content but a large score gap: outside score_tol, so the
        # content rule does not fire (a real improvement, not a copy).
        shared = _sk({f"{i:016x}" for i in range(20)})
        incumbent = _entry(
            composite=0.80,
            sha256="aa" * 32,
            size_bytes=500000,
            content_fingerprint=shared,
        )
        decision = evaluate_duplicate_signals(
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
        shared = _sk({f"{i:016x}" for i in range(20)})
        incumbent = _entry(
            composite=0.80, miner="5Mine", sha256="aa" * 32, content_fingerprint=shared
        )
        decision = evaluate_duplicate_signals(
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
        # (similarity 0) and the size rule still catches a same-size near-dup.
        incumbent = _entry(composite=0.80, sha256="aa" * 32, size_bytes=500000)
        decision = evaluate_duplicate_signals(
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
        decision = evaluate_duplicate_signals(
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
        d1 = evaluate_duplicate_signals(
            agent_id=me,
            miner_hotkey="5Me",
            sha256="dd" * 32,
            composite=0.80,
            size_bytes=500000,
            eligible=[entry],
        )
        d2 = evaluate_duplicate_signals(
            agent_id=me,
            miner_hotkey="5Me",
            sha256="dd" * 32,
            composite=0.80,
            size_bytes=500000,
            eligible=[entry],
        )
        assert d1.held is False
        assert d1 == d2


class TestPromptShadowSignal:
    """Prompt fingerprint in the gate: shadow mode. It never creates a hold on
    its own; it only annotates the audit reason of a hold another rule fired."""

    def test_prompt_match_alone_never_holds(self) -> None:
        # Identical prompt sketch to the incumbent, but distant score and distinct
        # sha / size / lexical: the prompt signal must NOT hold on its own.
        shared = {"pp" + f"{i:02d}" for i in range(20)}
        incumbent = _entry(
            composite=0.60,
            sha256="cc" * 32,
            size_bytes=400000,
            prompt_fingerprint=_psk(shared),
        )
        decision = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Challenger",
            sha256="dd" * 32,
            composite=0.90,  # far from incumbent
            size_bytes=700000,  # far in size
            prompt_fingerprint=_psk(shared),
            eligible=[incumbent],
        )
        assert decision.held is False

    def test_prompt_corroboration_annotates_lexical_hold(self) -> None:
        # A lexical near-dup fires (rule 2); a shared prompt sketch adds a note.
        shingles = {f"s{i:03d}" for i in range(30)}
        shared_prompt = {"pp" + f"{i:02d}" for i in range(20)}
        incumbent = _entry(
            composite=0.80,
            sha256="aa" * 32,
            size_bytes=500000,
            content_fingerprint=_sk(shingles),
            prompt_fingerprint=_psk(shared_prompt),
        )
        decision = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Copier",
            sha256="bb" * 32,
            composite=0.805,
            size_bytes=500100,
            content_fingerprint=_sk(shingles),  # jaccard 1.0 -> rule 2 holds
            prompt_fingerprint=_psk(shared_prompt),
            eligible=[incumbent],
        )
        assert decision.held is True
        assert "content near-duplicate" in (decision.reason or "")
        assert "prompt jaccard" in (decision.reason or "")

    def test_hold_without_prompt_sketch_has_no_note(self) -> None:
        # Same lexical hold, but no prompt sketches: reason must not mention prompt.
        shingles = {f"s{i:03d}" for i in range(30)}
        incumbent = _entry(
            composite=0.80, content_fingerprint=_sk(shingles), size_bytes=500000
        )
        decision = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Copier",
            sha256="bb" * 32,
            composite=0.805,
            size_bytes=500100,
            content_fingerprint=_sk(shingles),
            eligible=[incumbent],
        )
        assert decision.held is True
        assert "prompt" not in (decision.reason or "")

    def test_low_prompt_overlap_not_noted(self) -> None:
        # Lexical hold fires, prompt sketches present but nearly disjoint (below the
        # advisory tolerance): no prompt note is added.
        shingles = {f"s{i:03d}" for i in range(30)}
        incumbent = _entry(
            composite=0.80,
            size_bytes=500000,
            content_fingerprint=_sk(shingles),
            prompt_fingerprint=_psk({f"a{i:02d}" for i in range(20)}),
        )
        decision = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Copier",
            sha256="bb" * 32,
            composite=0.805,
            size_bytes=500100,
            content_fingerprint=_sk(shingles),
            prompt_fingerprint=_psk({f"b{i:02d}" for i in range(20)}),  # disjoint
            eligible=[incumbent],
        )
        assert decision.held is True
        assert "prompt" not in (decision.reason or "")


class TestNoveltyAwareGate:
    """The starter-kit false-positive regression suite.

    Every SN118 submission derives from the same public starter kit, so before
    baseline subtraction two INDEPENDENT miners each editing a couple of lines
    of ``baseline.rs`` scored jaccard/containment ~1.0 (and were kit-sized and
    score-adjacent), holding honest work on every channel at once. These tests
    pin the redesigned behavior end to end at the gate level.
    """

    def test_independent_kit_edits_not_held(self) -> None:
        # Two honest miners, each with a small DISJOINT novel contribution,
        # kit-sized tarballs (size delta 1B) and near-identical scores: the
        # pre-novelty gate held this pair via rule 2 AND rule 3.
        incumbent = _entry(
            composite=0.80,
            sha256="aa" * 32,
            size_bytes=524288,
            content_fingerprint=_nsk({f"a{i:015x}" for i in range(8)}),
        )
        decision = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Challenger",
            sha256="bb" * 32,
            composite=0.801,
            size_bytes=524289,
            content_fingerprint=_nsk({f"b{i:015x}" for i in range(8)}),
            eligible=[incumbent],
        )
        assert decision.held is False

    def test_tweaked_copy_held_by_novel_containment(self) -> None:
        # A copier lifts the incumbent's novel work and pads a cosmetic tweak
        # of their own: the incumbent's novel set is contained in the copy's.
        stolen = {f"a{i:015x}" for i in range(8)}
        incumbent = _entry(
            composite=0.80,
            sha256="aa" * 32,
            size_bytes=524288,
            content_fingerprint=_nsk(stolen),
        )
        decision = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Copier",
            sha256="bb" * 32,
            composite=0.805,
            size_bytes=524900,
            content_fingerprint=_nsk(stolen | {f"pad{i:013x}" for i in range(3)}),
            eligible=[incumbent],
        )
        assert decision.held is True
        assert decision.duplicate_of == incumbent.agent_id
        assert "content near-duplicate" in (decision.reason or "")

    def test_below_novelty_floor_abstains(self) -> None:
        # Near-pristine kits (fewer novel shingles than one edited region) have
        # nothing meaningful to compare — identical tiny novel sets stay unheld
        # (a literal resubmission is still caught by sha256 / repack equality).
        tiny = {f"a{i:015x}" for i in range(3)}
        incumbent = _entry(
            composite=0.80,
            sha256="aa" * 32,
            size_bytes=524288,
            content_fingerprint=_nsk(tiny),
        )
        decision = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Challenger",
            sha256="bb" * 32,
            composite=0.801,
            size_bytes=524289,
            content_fingerprint=_nsk(tiny),
            eligible=[incumbent],
        )
        assert decision.held is False

    def test_size_rule_never_fires_when_both_fingerprinted(self) -> None:
        # Identical size and score but distinct novel fingerprints: the size
        # fallback must stay silent when the content channel can see the pair.
        incumbent = _entry(
            composite=0.80,
            sha256="aa" * 32,
            size_bytes=524288,
            content_fingerprint=_nsk({f"a{i:015x}" for i in range(30)}),
        )
        decision = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Challenger",
            sha256="bb" * 32,
            composite=0.80,
            size_bytes=524288,
            content_fingerprint=_nsk({f"b{i:015x}" for i in range(30)}),
            eligible=[incumbent],
        )
        assert decision.held is False

    def test_size_fallback_still_covers_unfingerprinted_rows(self) -> None:
        # A row with no usable sketch keeps the old cheap size+score catch.
        incumbent = _entry(
            composite=0.80,
            sha256="aa" * 32,
            size_bytes=524288,
            content_fingerprint=None,
        )
        decision = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Copier",
            sha256="bb" * 32,
            composite=0.801,
            size_bytes=524300,
            content_fingerprint=_nsk({f"b{i:015x}" for i in range(30)}),
            eligible=[incumbent],
        )
        assert decision.held is True
        assert "size delta" in (decision.reason or "")

    def test_legacy_and_novelty_sketches_never_match(self) -> None:
        # A v1 whole-tarball sketch measures a different quantity than a v2
        # novelty sketch; identical members must not cross-fire.
        shared = {f"a{i:015x}" for i in range(30)}
        incumbent = _entry(
            composite=0.80,
            sha256="aa" * 32,
            size_bytes=None,
            content_fingerprint=_sk(shared),
        )
        decision = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Challenger",
            sha256="bb" * 32,
            composite=0.801,
            size_bytes=None,
            content_fingerprint=_nsk(shared),
            eligible=[incumbent],
        )
        assert decision.held is False

    def test_mismatched_baseline_corpora_never_match(self) -> None:
        # Sketches subtracted against different corpus versions leave different
        # residues; their overlap is meaningless and must not hold.
        shared = {f"a{i:015x}" for i in range(30)}
        incumbent = _entry(
            composite=0.80,
            sha256="aa" * 32,
            size_bytes=None,
            content_fingerprint=_nsk(shared, baseline_id="000000000000"),
        )
        decision = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Challenger",
            sha256="bb" * 32,
            composite=0.801,
            size_bytes=None,
            content_fingerprint=_nsk(shared, baseline_id="c8d712f1addf"),
            eligible=[incumbent],
        )
        assert decision.held is False
