"""Anti-copy moderation gate for the score-write path.

SN118 artifacts are downloadable, so the central threat is copying: download
the current best harness and resubmit it (verbatim or lightly tweaked) to
dethrone the original. The KOTH+ATH fold already defeats a *verbatim* copy — it
ties the incumbent, never clears the 1% margin, and first-seen protects the
original. This gate adds cheap signals against a *lightly-tweaked* copy that
nudges its score just past the incumbent: such a submission scores within a hair
of the agent it surpasses and either has a near-identical tarball size or — after
a re-indent/rename/reformat/edit that moves the size — a near-identical *content*
fingerprint (the shingle MinHash sketch from :mod:`ditto.api_server.fingerprint`),
compared by Jaccard (edit-in-place copy) or containment (padded copy).

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
# — around the 1% dethrone margin a real copy needs to clear.
_DEFAULT_SCORE_TOL = 0.02
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


def evaluate_antidup(
    *,
    agent_id: UUID,
    miner_hotkey: str,
    sha256: str,
    composite: float,
    size_bytes: int | None,
    eligible: Sequence[LedgerRow],
    content_fingerprint: dict | None = None,
    score_tol: float = _DEFAULT_SCORE_TOL,
    size_tol: int = _DEFAULT_SIZE_TOL,
    jaccard_tol: float = _DEFAULT_JACCARD_TOL,
    containment_tol: float = _DEFAULT_CONTAINMENT_TOL,
) -> ReviewDecision:
    """Decide whether a just-scored agent should be held for copy review.

    Copying is only a threat *across* miners, so every rule ignores the agent's
    own submissions and this miner's other agents (a miner iterating on their own
    harness is not a copier). Held iff, against **another miner's** eligible agent:

    1. **Exact copy** — same ``sha256``. Byte-identical resubmission.
    2. **Content near-duplicate** — composites within ``score_tol`` *and* the
       shingle-sketch fingerprints either at least ``jaccard_tol`` Jaccard-similar
       (a copy edited in place) *or* at least ``containment_tol`` contained (a copy
       padded with junk files to dilute Jaccard). Survives re-indent/rename/reformat
       and localized edits that the byte-level size rule (rule 3) misses.
    3. **Size near-duplicate** — composites within ``score_tol`` *and* tarball
       sizes within ``size_tol`` (a lightly-tweaked copy barely moves either).
       Retained as a cheap catch for rows with no fingerprint (uploaded before
       fingerprinting, or an unreadable tarball).

    Rules 2 and 3 check *every* other-miner eligible agent, in either score
    direction, so a genuine unrelated agent scoring in between cannot mask the
    copy. A genuine improvement (composite more than ``score_tol`` from any other
    miner's score, with both a different size and a distinct fingerprint) is never
    held. Pure + deterministic: ``eligible`` arrives in a fixed order, so the
    reported ``duplicate_of`` (the first match) is stable.
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

    # 2. Content near-dup: close in score AND near-identical normalized source,
    #    even if the tarball size drifted (re-indent / rename / reformat / edit).
    #    Checked before the size rule because the fingerprint is the stronger,
    #    size-independent signal.
    for e in others:
        if abs(composite - e.composite) <= score_tol:
            jaccard, containment = content_similarity(
                content_fingerprint, e.content_fingerprint
            )
            if jaccard >= jaccard_tol or containment >= containment_tol:
                return ReviewDecision(
                    held=True,
                    duplicate_of=e.agent_id,
                    reason=(
                        f"content near-duplicate of agent {e.agent_id}: "
                        f"composite delta {abs(composite - e.composite):.4f}, "
                        f"jaccard {jaccard:.3f}, containment {containment:.3f}"
                    ),
                )

    # 3. Size near-dup of another miner: close in both score and tarball size.
    if size_bytes is not None:
        for e in others:
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
                    ),
                )

    return _NOT_HELD
