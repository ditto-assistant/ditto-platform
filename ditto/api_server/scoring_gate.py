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
    sha256: str,
    composite: float,
    size_bytes: int | None,
    eligible: Sequence[LedgerRow],
    score_tol: float = _DEFAULT_SCORE_TOL,
    size_tol: int = _DEFAULT_SIZE_TOL,
) -> ReviewDecision:
    """Decide whether a just-scored agent should be held for copy review.

    Held iff either:

    1. **Exact copy** — an eligible agent (a different ``agent_id``) has the same
       ``sha256``. Byte-identical resubmission.
    2. **Near-dup dethroner** — the agent surpasses an existing eligible agent by
       only a hair (composite within ``score_tol``) *and* their tarball sizes are
       within ``size_tol``. Compared against the *incumbent* it surpasses (the
       highest eligible composite strictly below this one), because that is the
       submission a tweaked copy would be cloning to dethrone.

    A genuine improvement (composite more than ``score_tol`` above the incumbent)
    is never held. Pure + deterministic: the verdict depends only on the inputs.
    """
    # 1. Exact byte-identical copy of an already-eligible artifact.
    for e in eligible:
        if e.agent_id != agent_id and e.sha256 == sha256:
            return ReviewDecision(
                held=True,
                duplicate_of=e.agent_id,
                reason=f"exact sha256 match of agent {e.agent_id}",
            )

    # 2. Near-dup that barely surpasses the incumbent it would dethrone.
    incumbent: LedgerRow | None = None
    for e in eligible:
        if e.agent_id == agent_id or e.composite >= composite:
            continue
        if incumbent is None or e.composite > incumbent.composite:
            incumbent = e
    if (
        incumbent is not None
        and composite - incumbent.composite <= score_tol
        and size_bytes is not None
        and incumbent.size_bytes is not None
        and abs(size_bytes - incumbent.size_bytes) <= size_tol
    ):
        return ReviewDecision(
            held=True,
            duplicate_of=incumbent.agent_id,
            reason=(
                f"near-duplicate of agent {incumbent.agent_id}: "
                f"composite +{composite - incumbent.composite:.4f}, "
                f"size delta {abs(size_bytes - incumbent.size_bytes)}B"
            ),
        )

    return _NOT_HELD
