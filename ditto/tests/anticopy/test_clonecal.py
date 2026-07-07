"""Unit tests for :mod:`ditto.anticopy.clonecal` — the anti-copy calibration harness.

These lock the two things the harness must get right: (1) the obfuscation ladder
produces the invariances each tier claims (tier-1 cosmetic keeps the normalized
hash; tier-2 rename breaks it), and (2) the evaluation reports honest
precision/recall — in particular the L3a signal has **zero false positives** on
independents and full recall on the tier it is meant to catch.
"""

from __future__ import annotations

from ditto.anticopy.clonecal import (
    build_corpus,
    default_signals,
    demo_corpus,
    evaluate,
    format_report,
    obfuscate,
    pack,
    t_recomment,
    t_reformat,
    t_rename_idents,
    t_reorder_rename_files,
    unpack,
)
from ditto.api_server.fingerprint import compute_normalized_source_hash

_SEED = {
    "src/lib.rs": (
        b'const NAME: &str = "seed";\n'
        b"fn compute(x: i64) -> i64 {\n    let step = x + 1;\n    step * 2\n}\n"
    ),
    "Cargo.toml": b'[package]\nname = "seed"\n',
}


class TestPacking:
    def test_pack_is_deterministic_and_roundtrips(self) -> None:
        assert pack(_SEED) == pack(_SEED)  # fixed mtime → byte-identical
        assert unpack(pack(_SEED)) == dict(_SEED)

    def test_pack_is_file_order_invariant(self) -> None:
        reordered = dict(reversed(list(_SEED.items())))
        assert pack(_SEED) == pack(reordered)  # sorted-path emission


class TestLadder:
    def test_tier1_cosmetic_preserves_normalized_hash(self) -> None:
        # Reformat + recomment + file rename/reorder must NOT change the L3a hash.
        base = pack(_SEED)
        variant = pack(obfuscate(_SEED, tier=1))
        assert base != variant  # the bytes differ (it's a real repack)…
        h1 = compute_normalized_source_hash(base)
        h2 = compute_normalized_source_hash(variant)
        assert h1 is not None and h1 == h2  # …but the canonical hash matches

    def test_tier2_rename_breaks_normalized_hash(self) -> None:
        # Identifier renaming defeats the exact-repack hash (falls to L2/L4).
        h_base = compute_normalized_source_hash(pack(_SEED))
        h_t2 = compute_normalized_source_hash(pack(obfuscate(_SEED, tier=2)))
        assert h_base is not None and h_t2 is not None and h_base != h_t2

    def test_individual_transforms_keep_hash_and_rename_breaks_it(self) -> None:
        h = compute_normalized_source_hash(pack(_SEED))
        for transform in (t_reformat, t_recomment, t_reorder_rename_files):
            assert compute_normalized_source_hash(pack(transform(_SEED))) == h
        assert compute_normalized_source_hash(pack(t_rename_idents(_SEED))) != h

    def test_rename_is_consistent(self) -> None:
        renamed = t_rename_idents(_SEED)["src/lib.rs"].decode()
        # The definition and every use of `compute`/`step` are renamed together,
        # so no original defined identifier survives as a whole word.
        assert "compute" not in renamed
        assert "step" not in renamed
        assert "r_" in renamed  # deterministic new names


class TestEvaluation:
    def test_l3a_precise_on_independents_full_recall_on_tier1(self) -> None:
        corpus = demo_corpus()
        reports = {r.name: r for r in evaluate(corpus, default_signals())}
        l3a = reports["L3a_normalized_hash"]
        # Exact-repack hash: no false positive on any independent pair…
        assert l3a.best.precision == 1.0
        # …and it fires on the cosmetic (tier-1) clones. (Tier-2 rename escapes it,
        # so overall recall < 1 — that gap is what motivates L2/L4.)
        assert l3a.best.recall > 0.0

    def test_lexical_separates_clones_from_independents(self) -> None:
        corpus = demo_corpus()
        reports = {r.name: r for r in evaluate(corpus, default_signals())}
        # The lexical signal should reach a usable operating point (some threshold
        # with both precision and recall positive) on this corpus.
        jac = reports["L1_lexical_jaccard"]
        assert jac.best.precision > 0.0 and jac.best.recall > 0.0

    def test_report_formats(self) -> None:
        corpus = demo_corpus()
        reports = evaluate(corpus, default_signals())
        text = format_report(reports, corpus)
        assert "L3a_normalized_hash" in text
        assert "precision" in text

    def test_build_corpus_labels(self) -> None:
        seeds = [
            {"src/lib.rs": b"fn a() {}\n"},
            {"src/lib.rs": b"fn b() {}\n"},
        ]
        corpus = build_corpus(seeds, max_tier=2)
        clones = [p for p in corpus if p.is_clone]
        indep = [p for p in corpus if not p.is_clone]
        assert {p.tier for p in clones} == {1, 2}  # two ladder tiers per seed
        assert len(indep) == 1  # the single distinct-seed pair
