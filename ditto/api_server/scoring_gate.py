"""Anti-copy moderation gate for the score-write path.

SN118 artifacts are downloadable, so the central threat is copying: download
the current best harness and resubmit it (verbatim or lightly tweaked) to
dethrone the original. The KOTH+ATH fold already defeats a *verbatim* copy — it
ties the incumbent, never clears the 1% margin, and first-seen protects the
original. This gate adds cheap signals against a *lightly-tweaked* copy that
nudges its score just past the incumbent: such a submission scores within a hair
of the agent it surpasses and matches on the *lexical* fingerprint channel — a
sketch of the tarball text (:mod:`ditto.api_server.fingerprint`) computed over the
submission's NOVEL content (the public starter-kit scaffolding is subtracted at
fingerprint time; see :mod:`ditto.anticopy.baseline`), which survives
reindent/reformat/localized-edit and junk-file padding. Comparison is by Jaccard
(edit-in-place) or containment (padded copy). The *structural* AST-shape sketch
(computed by dittobench, arriving on the score report) and the prompt sketch are
corroborating annotations on a hold, not triggers: both are computed over the
whole crate, so on starter-kit-derived submissions they saturate between
independent miners. Tarball-size proximity is a fallback for rows with no usable
fingerprint only.

This is **moderation, not weight logic** — it decides only whether a suspicious
high-scorer is held in ``ath_pending_review`` for human review (see
:func:`ditto.db.queries.agents.resolve_review`), never who the champion is. The
KOTH fold itself lives in the validator. The function is pure and deterministic
so re-scoring the same agent yields the same verdict.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ditto.api_server.fingerprint import content_similarity

if TYPE_CHECKING:
    from uuid import UUID

    from ditto.db.queries.scores import LedgerRow

# A challenger scoring within this of the incumbent it surpasses is "a hair past"
# — the anti-copy tolerance must exceed the benchmark's between-seed composite
# noise so a re-rolled verbatim copy cannot clear it on a lucky seed.
# DittoBench v2 / bench_version 2 (BENCHMARK-V2 §6.2, B8) targets between-seed
# σ ≤ 0.01 composite and matches the validator's 5% KOTH margin: 0.03 ≈ 3σ. Bump
# alongside the subnet VALIDATOR_KOTH_MARGIN if the hosted 30-seed σ comes in
# higher (was 0.02 for v1).
_DEFAULT_SCORE_TOL = 0.03
# A tweaked copy differs from the original by at most a few edited lines, so its
# gzipped tarball size barely moves. 8 KiB comfortably covers small edits.
_DEFAULT_SIZE_TOL = 8192
# Content-fingerprint (shingle-sketch) thresholds. Jaccard catches a copy edited
# in place; containment (overlap coefficient) catches one padded with junk files
# to dilute Jaccard. Containment's bar is higher because it is the more
# false-positive-prone of the two on shared reference-harness scaffolding — only a
# near-total subset should trip it. Both are paired with score proximity below, and
# a hold only routes to *human* review, so these are deliberately conservative
# signals, not an autoban. They want tuning against a real score/similarity corpus
# (see the subnet's KOTH-parameter validation task).
_DEFAULT_JACCARD_TOL = 0.75
_DEFAULT_CONTAINMENT_TOL = 0.95
# Structural (AST) thresholds are higher than the lexical ones: the structural
# sketch discards identifiers + formatting, so two independent crates built on the
# same reference harness share far more of their parse-tree shape than their text.
# Only a near-total structural match should trip this rename-resistant channel.
_DEFAULT_STRUCTURAL_JACCARD_TOL = 0.85
_DEFAULT_STRUCTURAL_CONTAINMENT_TOL = 0.98
# prompt overlap above which a hold's audit reason notes the corroboration.
# Advisory only: the prompt sketch (``compute_prompt_fingerprint``) does *not* hold
# an agent on its own — honest agents on the same reference harness share
# scaffolding prompts (the convergent case in ``ditto.anticopy.calibration`` scores
# ~0.8 there), and the signals orthogonal to that convergence (behavioral /
# code-embedding) are not built yet. So the prompt fingerprint runs in shadow mode:
# it enriches the moderator's audit trail on a hold another rule already fired, and
# its per-agent sketch is stored for calibration, but the active fusion hold waits
# on an orthogonal signal (the code-embedding or behavioral channel).
_PROMPT_ADVISORY_TOL = 0.5
# Minimum observed novel-shingle evidence (novelty sketches, fingerprint v2)
# before the lexical channel may hold. One edited line disturbs up to
# _SHINGLE_LINES(=4) shingles, so below this floor the "novel content" is less
# than a single edited region — statistically nothing to compare, and nothing a
# copier would need to steal (the exact-equality rules still catch literal
# resubmissions of near-pristine kits).
_MIN_NOVEL_SHINGLES = 4


@dataclass(frozen=True)
class ReviewDecision:
    """Outcome of the anti-copy gate for one just-scored agent.

    ``held`` routes the agent to ``ath_pending_review`` with ``duplicate_of``
    (the earlier submission it appears to copy) and ``reason`` recorded as the
    moderation audit trail. ``held=False`` lets the normal ``evaluating ->
    scored`` transition proceed.
    """

    held: bool
    duplicate_of: UUID | None = None
    reason: str | None = None


_NOT_HELD = ReviewDecision(held=False)


def _prompt_note(prompt_fingerprint: dict | None, e: LedgerRow) -> str:
    """Shadow-mode the prompt fingerprint suffix for a hold's audit reason.

    Returns ``"; prompt jaccard X.XXX"`` when the just-scored agent and the matched
    agent ``e`` both carry a prompt sketch overlapping at or above
    ``_PROMPT_ADVISORY_TOL``, else ``""``. Observability only — this never affects
    whether the agent is held.
    """
    j, c = content_similarity(prompt_fingerprint, e.prompt_fingerprint)
    if max(j, c) >= _PROMPT_ADVISORY_TOL:
        return f"; prompt jaccard {j:.3f}, containment {c:.3f}"
    return ""


def _structural_note(
    structural_fingerprint: dict | None,
    e: LedgerRow,
    *,
    jaccard_tol: float,
    containment_tol: float,
) -> str:
    """Advisory structural-overlap note appended to a hold's audit reason.

    The structural (AST-shape) sketch arrives from dittobench computed over the
    WHOLE crate, so on starter-kit-derived submissions it saturates near 1.0
    between independent miners exactly like the pre-novelty lexical channel did.
    Until dittobench ships baseline-subtracted structural sketches it therefore
    corroborates instead of triggering: high overlap is recorded on a hold that
    already fired so the moderator sees the rename-resistant channel's opinion.
    """
    j, c = content_similarity(structural_fingerprint, e.structural_fingerprint)
    if j >= jaccard_tol or c >= containment_tol:
        return f"; structural jaccard {j:.3f}, containment {c:.3f}"
    return ""


def _lexical_abstains(fingerprint: dict | None) -> bool:
    """Whether a novelty sketch carries too little evidence to hold on.

    Applies only to novelty sketches (``"bl"`` present): a whole-tarball legacy
    sketch keeps its original semantics until the re-fingerprint backfill
    replaces it.
    """
    if not fingerprint or "bl" not in fingerprint:
        return False
    return int(fingerprint.get("card", 0)) < _MIN_NOVEL_SHINGLES


def evaluate_duplicate_signals(
    *,
    agent_id: UUID,
    miner_hotkey: str,
    sha256: str,
    composite: float,
    size_bytes: int | None,
    eligible: Sequence[LedgerRow],
    normalized_source_hash: str | None = None,
    content_fingerprint: dict | None = None,
    structural_fingerprint: dict | None = None,
    prompt_fingerprint: dict | None = None,
    score_tol: float = _DEFAULT_SCORE_TOL,
    size_tol: int = _DEFAULT_SIZE_TOL,
    jaccard_tol: float = _DEFAULT_JACCARD_TOL,
    containment_tol: float = _DEFAULT_CONTAINMENT_TOL,
    structural_jaccard_tol: float = _DEFAULT_STRUCTURAL_JACCARD_TOL,
    structural_containment_tol: float = _DEFAULT_STRUCTURAL_CONTAINMENT_TOL,
) -> ReviewDecision:
    """Decide whether a just-scored agent should be held for copy review.

    Copying is only a threat *across* miners, so every rule ignores the agent's
    own submissions and this miner's other agents (a miner iterating on their own
    harness is not a copier). Held iff, against **another miner's** eligible agent:

    1. **Exact copy** — same ``sha256``. Byte-identical resubmission.
    1b. **Exact repack** — same ``normalized_source_hash``: the same source
       canonicalized (comments/whitespace stripped, files sorted), so a reformat /
       re-comment / file rename+reorder repack that changes ``sha256`` still
       matches. Like rule 1, held on the hash equality alone — no score proximity,
       because an exact-source match is copy evidence regardless of the score it
       happened to land.
    2. **Near-duplicate lexical fingerprint** — composites within ``score_tol``
       *and* the lexical channel (``content_fingerprint``) at least
       ``jaccard_tol`` Jaccard or ``containment_tol`` contained. On novelty
       sketches (the shared starter-kit scaffolding subtracted at fingerprint
       time) this compares only what each miner actually wrote, so independent
       few-line edits of the reference harness score ~0 against each other while
       a tweaked copy still carries the incumbent's novel shingles. The channel
       abstains when either side's observed novelty is below
       ``_MIN_NOVEL_SHINGLES`` (nothing meaningful to compare). Structural
       (``structural_fingerprint``, whole-crate AST shape from dittobench) and
       prompt overlap annotate the hold's audit reason as corroboration; neither
       triggers on its own until they are baseline-aware.

    3. **Size near-duplicate** — composites within ``score_tol`` *and* tarball
       sizes within ``size_tol``, checked ONLY when either side has no usable
       content fingerprint (uploaded before fingerprinting, or an unreadable
       tarball). Starter-kit-derived tarballs are all kit-sized, so size+score
       alone must never hold a pair the content channel can actually see.

    Rules 2 and 3 check *every* other-miner eligible agent, in either score
    direction, so a genuine unrelated agent scoring in between cannot mask the
    copy. A genuine improvement (composite more than ``score_tol`` from any other
    miner's score, with a different size and both fingerprints distinct) is never
    held. Pure + deterministic: ``eligible`` arrives in a fixed order, so the
    reported ``duplicate_of`` (the first match) is stable.

    ``prompt_fingerprint`` participates in **shadow mode only**: when a hold
    fires for another reason, a high prompt overlap with the matched agent is
    appended to the audit reason (``_PROMPT_ADVISORY_TOL``). It never creates a hold
    on its own — a prompt match alone is not copy evidence, because honest agents on
    the same reference harness share scaffolding prompts. The active prompt-fusion
    hold is deferred until an orthogonal-to-convergence signal (behavioral /
    code-embedding) exists to corroborate it.
    """
    others = [
        e for e in eligible if e.agent_id != agent_id and e.miner_hotkey != miner_hotkey
    ]

    # 1. Exact byte-identical copy of another miner's eligible artifact.
    for e in others:
        if e.sha256 == sha256:
            return ReviewDecision(
                held=True,
                duplicate_of=e.agent_id,
                reason=f"exact sha256 match of agent {e.agent_id}",
            )

    # 1b. Exact-repack copy: same canonicalized source (comments/whitespace
    #     stripped, files sorted) even when the tarball bytes differ. An equality
    #     match, held unconditionally like sha256 — no score-proximity requirement.
    #     Both hashes must be present (null = "no repack match", never a hit).
    if normalized_source_hash is not None:
        for e in others:
            if e.normalized_source_hash == normalized_source_hash:
                return ReviewDecision(
                    held=True,
                    duplicate_of=e.agent_id,
                    reason=f"normalized-source (repack) match of agent {e.agent_id}",
                )

    # 2. Near-dup fingerprint: close in score AND the lexical sketch matches.
    #    On novelty sketches (fingerprint v2) the comparison covers only the
    #    submission's OWN contribution — the shared public starter kit is
    #    subtracted at fingerprint time — so two independent miners each editing
    #    a few lines of the reference harness no longer look near-identical.
    #    The channel abstains below _MIN_NOVEL_SHINGLES of observed novelty
    #    (nothing meaningful to compare); structural and prompt overlap are
    #    appended to the audit reason as corroboration, never triggers.
    if not _lexical_abstains(content_fingerprint):
        for e in others:
            if abs(composite - e.composite) > score_tol:
                continue
            if _lexical_abstains(e.content_fingerprint):
                continue
            lex_j, lex_c = content_similarity(
                content_fingerprint, e.content_fingerprint
            )
            if lex_j >= jaccard_tol or lex_c >= containment_tol:
                return ReviewDecision(
                    held=True,
                    duplicate_of=e.agent_id,
                    reason=(
                        f"content near-duplicate of agent {e.agent_id}: "
                        f"composite delta {abs(composite - e.composite):.4f}, "
                        f"jaccard {lex_j:.3f}, containment {lex_c:.3f}"
                        + _structural_note(
                            structural_fingerprint,
                            e,
                            jaccard_tol=structural_jaccard_tol,
                            containment_tol=structural_containment_tol,
                        )
                        + _prompt_note(prompt_fingerprint, e)
                    ),
                )

    # 3. Size near-dup of another miner: close in both score and tarball size.
    #    A FALLBACK for pairs the content channel cannot see (a row uploaded
    #    before fingerprinting or an unreadable tarball) — never fired when both
    #    sides carry a usable sketch, because on starter-kit-derived submissions
    #    every honest tarball is kit-sized and score-adjacent, which made
    #    size+score the gate's dominant false-positive source.
    if size_bytes is not None:
        for e in others:
            if content_fingerprint and e.content_fingerprint:
                continue
            if (
                e.size_bytes is not None
                and abs(composite - e.composite) <= score_tol
                and abs(size_bytes - e.size_bytes) <= size_tol
            ):
                return ReviewDecision(
                    held=True,
                    duplicate_of=e.agent_id,
                    reason=(
                        f"near-duplicate of agent {e.agent_id}: "
                        f"composite delta {abs(composite - e.composite):.4f}, "
                        f"size delta {abs(size_bytes - e.size_bytes)}B"
                        + _prompt_note(prompt_fingerprint, e)
                    ),
                )

    return _NOT_HELD
