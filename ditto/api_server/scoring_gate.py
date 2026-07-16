"""Anti-copy moderation gate for the score-write path.

SN118 artifacts are downloadable, so the central threat is copying: download
the current best harness and resubmit it (verbatim or lightly tweaked) to
dethrone the original. The KOTH+ATH fold already defeats a *verbatim* copy — it
ties the incumbent, never clears the 2% margin, and first-seen protects the
original. This gate adds cheap signals against a *lightly-tweaked* copy that
nudges its score just past the incumbent: such a submission scores within a hair
of the agent it surpasses and matches on the *lexical* fingerprint channel — a
reference-aware sketch of the tarball text (:mod:`ditto.api_server.fingerprint`,
official starter-kit scaffolding subtracted before sketching), which survives
reindent/reformat/localized-edit and junk-file padding, compared by Jaccard
(edit-in-place) or containment (padded copy). The *structural* sketch of the
crate's AST shape (computed by dittobench, arriving on the score report) and the
prompt sketch corroborate a hold's audit reason but never trigger one: both are
whole-crate, so they saturate between independent starter-kit derivatives until
they are reference-aware too. Tarball-size proximity is a fallback for rows with
no comparable fingerprints only.

This is **moderation, not weight logic** — it decides only whether a suspicious
high-scorer is held in ``ath_pending_review`` for human review (see
:func:`ditto.db.queries.agents.resolve_review`), never who the champion is. The
KOTH fold itself lives in the validator. The function is pure and deterministic
so re-scoring the same agent yields the same verdict.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ditto.api_server.fingerprint import content_similarity

if TYPE_CHECKING:
    from uuid import UUID

    from ditto.db.queries.scores import LedgerRow

# A challenger scoring within this of the incumbent it surpasses is "a hair past"
# — the anti-copy tolerance must exceed the benchmark's between-seed composite
# noise so a re-rolled verbatim copy cannot clear it on a lucky seed.
# DittoBench v2 / bench_version 2 (BENCHMARK-V2 §6.2, B8) targets between-seed
# σ ≤ 0.01 composite. The 0.03 tolerance remains intentionally broader than
# the validator's 2% KOTH margin so a near-copy that barely clears the crown gate
# is still held for review. Bump this tolerance if the hosted 30-seed σ comes in
# higher (it was 0.02 for v1).
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
# Structural (AST) thresholds gate the ADVISORY structural annotation on a hold
# (see _structural_note): higher than the lexical ones because the structural
# sketch discards identifiers + formatting, so two independent crates built on the
# same reference harness share far more of their parse-tree shape than their text.
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


def _utc(dt: datetime) -> datetime:
    """Normalize database timestamps for deterministic chronology comparisons."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _fingerprint_versions_incompatible(a: dict | None, b: dict | None) -> bool:
    """Whether two present sketches use incomparable algorithms or corpora."""
    return bool(
        a
        and b
        and a.get("v") is not None
        and b.get("v") is not None
        and (a.get("v") != b.get("v") or a.get("corpus") != b.get("corpus"))
    )


def _structural_note(
    structural_fingerprint: dict | None,
    e: LedgerRow,
    *,
    jaccard_tol: float,
    containment_tol: float,
) -> str:
    """Advisory structural-overlap suffix for a hold's audit reason.

    The structural (AST-shape) sketch arrives from dittobench computed over the
    WHOLE crate — ``astfp`` performs no reference subtraction — so on
    starter-kit-derived submissions it saturates between independent miners
    exactly like the pre-reference lexical channel did (measured on the audited
    corpus: 12 of the 66 current holds sit at/above the structural thresholds,
    concentrated in the smallest-edit submissions, which the lexical
    residual-cardinality floor deliberately routes past the lexical check).
    Until dittobench ships reference-subtracted structural sketches this
    channel corroborates a hold that already fired; it never triggers one.
    """
    j, c = content_similarity(structural_fingerprint, e.structural_fingerprint)
    if j >= jaccard_tol or c >= containment_tol:
        return f"; structural jaccard {j:.3f}, containment {c:.3f}"
    return ""


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


def evaluate_duplicate_signals(
    *,
    agent_id: UUID,
    miner_hotkey: str,
    submitted_at: datetime,
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
    harness is not a copier). Originality attribution also follows upload chronology,
    not score-finalization order: normalized-source and near-duplicate rules compare
    only with another miner's strictly earlier submission. Equal timestamps use the
    UUID as a deterministic tie-break. Held iff:

    1. **Exact copy** — same ``sha256``. Byte-identical resubmission.
    1b. **Exact repack** — same ``normalized_source_hash``: the same source
       canonicalized (comments/whitespace stripped, files sorted), so a reformat /
       re-comment / file rename+reorder repack that changes ``sha256`` still
       matches. Like rule 1, held on the hash equality alone — no score proximity,
       because an exact-source match is copy evidence regardless of the score it
       happened to land.
    2. **Near-duplicate lexical fingerprint** — composites within ``score_tol``
       *and* the lexical channel (``content_fingerprint``, reference-aware) at
       least ``jaccard_tol`` Jaccard or ``containment_tol`` contained — survives
       re-indent / reformat / localized edits / junk-file padding. Structural
       (``structural_fingerprint``, whole-crate AST from dittobench — measured
       at/above its thresholds for 12 of the 66 audited holds, concentrated in
       the smallest-edit submissions) and prompt overlap annotate the hold's
       audit reason as corroboration; neither triggers until reference-aware.

    3. **Size near-duplicate fallback** — composites within ``score_tol`` and
       tarball sizes within ``size_tol``, but only when neither lexical nor
       structural fingerprints are comparable. A valid negative fingerprint is
       evidence of distinct content and must not be overridden by similar archive
       size. The fallback remains for legacy or unreadable artifacts.

    Rules 2 and 3 check *every earlier* other-miner eligible agent, in either score
    direction, so a genuine unrelated agent scoring in between cannot mask the
    copy. A genuine improvement (composite more than ``score_tol`` from any other
    miner's score, with a different size and both fingerprints distinct) is never
    held. Pure + deterministic: candidates are ordered by upload chronology, so the
    reported ``duplicate_of`` is the oldest matching submission.

    ``prompt_fingerprint`` participates in **shadow mode only**: when a hold
    fires for another reason, a high prompt overlap with the matched agent is
    appended to the audit reason (``_PROMPT_ADVISORY_TOL``). It never creates a hold
    on its own — a prompt match alone is not copy evidence, because honest agents on
    the same reference harness share scaffolding prompts. The active prompt-fusion
    hold is deferred until an orthogonal-to-convergence signal (behavioral /
    code-embedding) exists to corroborate it.
    """
    other_miners = [
        e for e in eligible if e.agent_id != agent_id and e.miner_hotkey != miner_hotkey
    ]
    submitted_key = (_utc(submitted_at), agent_id.int)
    earlier_others = sorted(
        (
            e
            for e in other_miners
            if (_utc(e.first_seen), e.agent_id.int) < submitted_key
        ),
        key=lambda e: (_utc(e.first_seen), e.agent_id.int),
    )

    # 1. Exact byte-identical copy of another miner's eligible artifact.
    # This is a defense-in-depth mirror of the separate admission-time exact-byte
    # guard. It uses the same upload chronology so a later-finalized row can never
    # become the retroactive original of an earlier submission.
    for e in earlier_others:
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
        for e in earlier_others:
            if e.normalized_source_hash == normalized_source_hash:
                return ReviewDecision(
                    held=True,
                    duplicate_of=e.agent_id,
                    reason=f"normalized-source (repack) match of agent {e.agent_id}",
                )

    # 2. Near-dup fingerprint: close in score AND a matching lexical sketch.
    #    Checked before the size rule because a fingerprint is the stronger,
    #    size-independent signal. The structural (whole-crate AST) channel
    #    saturates between independent starter-kit derivatives — astfp performs
    #    no reference subtraction — so it corroborates the audit reason instead
    #    of triggering until its sketches are reference-aware.
    for e in earlier_others:
        if abs(composite - e.composite) > score_tol:
            continue
        if _fingerprint_versions_incompatible(
            content_fingerprint, e.content_fingerprint
        ):
            return ReviewDecision(
                held=True,
                duplicate_of=e.agent_id,
                reason=(
                    f"anti-copy comparison inconclusive with agent {e.agent_id}: "
                    "incompatible lexical fingerprint version or corpus; "
                    "individual operator review required"
                ),
            )
        lex_j, lex_c = content_similarity(content_fingerprint, e.content_fingerprint)
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

    # 3. Size near-dup of another miner: a legacy fallback only when both rows
    #    predate every lexical/structural fingerprint. A versioned empty sketch is
    #    affirmative evidence that reference subtraction found too little custom
    #    surface; cross-version sketches likewise must fail open during backfill.
    if size_bytes is not None:
        for e in earlier_others:
            if (
                e.size_bytes is not None
                and abs(composite - e.composite) <= score_tol
                and abs(size_bytes - e.size_bytes) <= size_tol
                and content_fingerprint is None
                and e.content_fingerprint is None
                and structural_fingerprint is None
                and e.structural_fingerprint is None
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
