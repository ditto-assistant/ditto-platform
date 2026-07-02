"""Anti-copy moderation gate for the score-write path.

SN118 artifacts are downloadable, so the central threat is copying: download
the current best harness and resubmit it (verbatim or lightly tweaked) to
dethrone the original. The KOTH+ATH fold already defeats a *verbatim* copy — it
ties the incumbent, never clears the 1% margin, and first-seen protects the
original. This gate adds a cheap signal against a *lightly-tweaked* copy that
nudges its score just past the incumbent: such a submission scores within a hair
of the agent it surpasses and has a near-identical tarball size.

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

if TYPE_CHECKING:
    from uuid import UUID

    from ditto.db.queries.scores import LedgerRow

# A challenger scoring within this of the incumbent it surpasses is "a hair past"
# — around the 1% dethrone margin a real copy needs to clear.
_DEFAULT_SCORE_TOL = 0.02
# A tweaked copy differs from the original by at most a few edited lines, so its
# gzipped tarball size barely moves. 8 KiB comfortably covers small edits.
_DEFAULT_SIZE_TOL = 8192


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
    score_tol: float = _DEFAULT_SCORE_TOL,
    size_tol: int = _DEFAULT_SIZE_TOL,
) -> ReviewDecision:
    """Decide whether a just-scored agent should be held for copy review.

    Copying is only a threat *across* miners, so both rules ignore the agent's
    own submissions and this miner's other agents (a miner iterating on their own
    harness is not a copier). Held iff, against **another miner's** eligible agent:

    1. **Exact copy** — same ``sha256``. Byte-identical resubmission.
    2. **Near-duplicate** — composites within ``score_tol`` *and* tarball sizes
       within ``size_tol`` (a lightly-tweaked copy barely moves either). Checked
       against *every* other-miner eligible agent, in either score direction, so
       a genuine unrelated agent scoring in between cannot mask the copy.

    A genuine improvement (composite more than ``score_tol`` from any other
    miner's score, or a clearly different size) is never held. Pure +
    deterministic: ``eligible`` arrives in a fixed order, so the reported
    ``duplicate_of`` (the first match) is stable.
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

    # 2. Near-dup of another miner: close in both score and tarball size.
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
