"""Tests for the shared-scaffolding baseline corpus and novelty fingerprints."""

from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path
from uuid import uuid4

import pytest

from ditto.anticopy import baseline as baseline_mod
from ditto.anticopy.baseline import load_baseline, main
from ditto.api_server.fingerprint import (
    _FP_NOVELTY_VERSION,
    compute_content_fingerprint,
    content_similarity,
)
from ditto.api_server.scoring_gate import evaluate_duplicate_signals
from ditto.tests.api_server.test_scoring_gate import _entry

# A synthetic "public starter kit": enough distinct lines that whole-tarball
# similarity between derivatives saturates, mirroring the real kit (~107k
# shingles vs a handful per honest edit).
_KIT_FILES = {
    f"src/mod_{i}.rs": "\n".join(
        f"pub fn kit_{i}_{j}(x: u64) -> u64 {{ x.wrapping_mul({i * 100 + j}) }}"
        for j in range(60)
    ).encode()
    for i in range(6)
}


def _tar(files: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as tar:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    return out.getvalue()


def _derivative(edit: str, *, second_file_edit: str | None = None) -> bytes:
    files = dict(_KIT_FILES)
    files["src/mod_0.rs"] = files["src/mod_0.rs"] + f"\n{edit}\n".encode()
    if second_file_edit is not None:
        files["src/mod_1.rs"] = (
            files["src/mod_1.rs"] + f"\n{second_file_edit}\n".encode()
        )
    return _tar(files)


# A miner's competitive contribution: a BLOCK of code, not a single line.
# Shingles are 4-line windows, so a stolen block's interior windows survive in
# the copy even when the copier edits elsewhere; a single stolen line would not
# (any adjacent edit shifts every window that contains it), which is fine — a
# one-line delta is inside benchmark noise and protected by the KOTH margin,
# not by moderation.
_INNOVATION_A = "\n".join(
    f"pub fn tuned_a_{i}(x: u64) -> u64 {{ x.rotate_left({i + 1}) ^ 0xA5 }}"
    for i in range(12)
)
_INNOVATION_B = "\n".join(
    f"pub fn tuned_b_{i}(x: u64) -> u64 {{ x.rotate_right({i + 1}) | 0x5A }}"
    for i in range(12)
)


@pytest.fixture()
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "baseline.txt.gz"
    kit_tar = tmp_path / "kit.tar.gz"
    kit_tar.write_bytes(_tar(_KIT_FILES))
    assert main(["generate", str(kit_tar), "--out", str(out)]) == 0
    monkeypatch.setenv("ANTICOPY_BASELINE_FILE", str(out))
    loaded = load_baseline()
    assert loaded is not None
    return loaded


class TestBaselineCorpus:
    def test_generate_and_load_roundtrip(self, corpus) -> None:
        assert len(corpus.shingles) > 100
        assert len(corpus.baseline_id) == 12

    def test_generation_is_deterministic(self, tmp_path: Path) -> None:
        kit_tar = tmp_path / "kit.tar.gz"
        kit_tar.write_bytes(_tar(_KIT_FILES))
        out_a, out_b = tmp_path / "a.txt.gz", tmp_path / "b.txt.gz"
        main(["generate", str(kit_tar), "--out", str(out_a)])
        main(["generate", str(kit_tar), "--out", str(out_b)])
        assert out_a.read_bytes() == out_b.read_bytes()

    def test_merge_unions_prior_corpus(self, tmp_path: Path) -> None:
        kit_tar = tmp_path / "kit.tar.gz"
        kit_tar.write_bytes(_tar(_KIT_FILES))
        prior = tmp_path / "prior.txt.gz"
        prior.write_bytes(gzip.compress(b"deadbeefdeadbeef"))
        out = tmp_path / "merged.txt.gz"
        main(["generate", str(kit_tar), "--merge", str(prior), "--out", str(out)])
        merged = set(gzip.decompress(out.read_bytes()).decode().split("\n"))
        assert "deadbeefdeadbeef" in merged

    def test_missing_corpus_disables_subtraction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTICOPY_BASELINE_FILE", str(tmp_path / "absent.gz"))
        assert load_baseline() is None

    def test_packaged_default_corpus_loads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTICOPY_BASELINE_FILE", raising=False)
        loaded = load_baseline()
        assert loaded is not None and len(loaded.shingles) > 10000


class TestNoveltyFingerprint:
    def test_novelty_sketch_shape(self, corpus) -> None:
        sketch = compute_content_fingerprint(
            _derivative("pub fn my_edit(x: u64) -> u64 { x + 1 }"),
            exclude=corpus.shingles,
            baseline_id=corpus.baseline_id,
        )
        assert sketch is not None
        assert sketch["v"] == _FP_NOVELTY_VERSION
        assert sketch["bl"] == corpus.baseline_id
        # Only the edited region's shingles remain — a handful, not hundreds.
        assert 0 < sketch["card"] < 20

    def test_pristine_kit_has_empty_novelty(self, corpus) -> None:
        sketch = compute_content_fingerprint(
            _tar(_KIT_FILES), exclude=corpus.shingles, baseline_id=corpus.baseline_id
        )
        assert sketch is not None and sketch["card"] == 0
        assert content_similarity(sketch, sketch) == (0.0, 0.0)

    def test_independent_edits_separate_copies_match(self, corpus) -> None:
        """The end-to-end property the redesign exists for.

        Whole-tarball sketches of two independent one-line kit edits are
        near-identical (the old false positive); novelty sketches of the same
        pair are disjoint, while a copy that lifts one miner's edit still
        matches it.
        """
        tar_a = _derivative(_INNOVATION_A)
        tar_b = _derivative(_INNOVATION_B)
        # The copier lifts A's block verbatim and adds their own tweak in a
        # different file to dodge exact-equality rules.
        tar_copy_of_a = _derivative(
            _INNOVATION_A, second_file_edit="// dodge\npub fn pad(x: u64) -> u64 { x }"
        )

        legacy_a = compute_content_fingerprint(tar_a)
        legacy_b = compute_content_fingerprint(tar_b)
        legacy_j, _ = content_similarity(legacy_a, legacy_b)
        assert legacy_j > 0.9  # the pre-novelty false positive, pinned

        kwargs = {"exclude": corpus.shingles, "baseline_id": corpus.baseline_id}
        novel_a = compute_content_fingerprint(tar_a, **kwargs)
        novel_b = compute_content_fingerprint(tar_b, **kwargs)
        novel_copy = compute_content_fingerprint(tar_copy_of_a, **kwargs)

        honest_j, honest_c = content_similarity(novel_a, novel_b)
        assert honest_j < 0.25 and honest_c < 0.5
        copy_j, copy_c = content_similarity(novel_a, novel_copy)
        assert copy_c >= 0.95

        # And through the gate: honest pair clears, the copy is held.
        incumbent = _entry(
            composite=0.80,
            sha256="aa" * 32,
            size_bytes=len(tar_a),
            content_fingerprint=novel_a,
        )
        honest = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Challenger",
            sha256="bb" * 32,
            composite=0.801,
            size_bytes=len(tar_b),
            content_fingerprint=novel_b,
            eligible=[incumbent],
        )
        assert honest.held is False
        copy = evaluate_duplicate_signals(
            agent_id=uuid4(),
            miner_hotkey="5Copier",
            sha256="cc" * 32,
            composite=0.802,
            size_bytes=len(tar_copy_of_a),
            content_fingerprint=novel_copy,
            eligible=[incumbent],
        )
        assert copy.held is True
        assert copy.duplicate_of == incumbent.agent_id


def test_module_baseline_cache_tracks_mtime(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "baseline.txt.gz"
    kit_tar = tmp_path / "kit.tar.gz"
    kit_tar.write_bytes(_tar(_KIT_FILES))
    main(["generate", str(kit_tar), "--out", str(out)])
    monkeypatch.setenv("ANTICOPY_BASELINE_FILE", str(out))
    first = baseline_mod.load_baseline()
    assert first is not None
    second = baseline_mod.load_baseline()
    assert second is first  # cached: same (path, mtime)
