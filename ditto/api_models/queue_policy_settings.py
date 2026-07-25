"""Versioned, hot-swappable operator policy for the validator queue.

Every constant in here used to be a source-level literal, and retuning one was
a code change plus a review plus a deploy. The queue is retuned *often* --
whenever miner behaviour shifts -- so the tuning itself belongs in the operator
settings system, not in the release process.

These wire models back an append-only revision table, exactly like
``ditto.api_models.continual_retest_settings`` and
``ditto.api_models.screener_review_settings``. A board with no revision behaves
byte-identically to the previously hard-coded values, so shipping this changes
nothing until an operator writes a revision.

Three lifecycles live on one board
==================================

The board is one revision payload (whole policy, never a diff) because that is
what makes a historical revision reconstructable. But the *fields* are consumed
at three different moments, and confusing them is how a queue change becomes an
incident:

``NEXT ROLLOUT`` -- :attr:`~QueuePolicySettings.rescore_cohort_size` and
    :attr:`~QueuePolicySettings.priority_cohort_size`. Read exactly once, when
    an operator starts a rollout, then frozen onto the rollout row
    (``BenchmarkRollout.rescore_cohort_target`` /
    ``.priority_cohort_target``). Editing them never touches an in-flight
    rollout. Safe to write at any time.

``LIVE, ROLLOUT-LOCKED`` -- :attr:`~QueuePolicySettings.lane_cycle_size` and
    :attr:`~QueuePolicySettings.fresh_submission_slots`, listed in
    :data:`ROLLOUT_LOCKED_FIELDS`. Read on every validator job poll, but
    **refused while a rollout is open**. The lane counter is "jobs this
    validator completed since rollout start, mod ``lane_cycle_size``", so
    changing the modulus mid-rollout discontinuously reassigns every validator's
    position in the cycle: a validator that was about to take cohort work
    silently starts taking fresh work instead, and the reverse. The refusal
    costs nothing, because the lane split only *does* anything while a rollout
    is open -- see :func:`rollout_locked_change`.

``LIVE`` -- everything else. Read on the hot path and safe to change at any
    time, because each one only resizes or re-scopes a set rather than moving a
    counter's modulus.

Bounds are spelled out here rather than imported from
``ditto.db.queries.benchmark_rollout``: that module imports this package, so
importing back would be a cycle.
``test_queue_policy_bounds_match_queue_constants`` asserts the two agree, so
drifting them apart fails CI.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Cohort sizing bounds
# ---------------------------------------------------------------------------
# Mirrors ``PRIORITY_COHORT_SIZE``: no cohort may be smaller than the priority
# five, which is also the size of the KOTH emission set
# (``MIN_DESIRED_AUTHORITY_AGENTS``). Below five, flipping the ledger to the
# desired version would have fewer recipients than the emission split expects.
MIN_COHORT_SIZE = 5
# Mirrors ``MAX_PERSISTED_RESCORE_COHORT_SIZE`` and the
# ``benchmark_rollout_bounded_members`` CHECK constraint on
# ``benchmark_rollouts.cohort_size``. A larger value could not be persisted.
MAX_COHORT_SIZE = 25
# Mirrors ``DEFAULT_RESCORE_COHORT_SIZE``: the inherited top ten, which is what
# every rollout used before this setting existed.
DEFAULT_RESCORE_COHORT_SIZE = 10
# Mirrors ``PRIORITY_COHORT_SIZE`` as a *default*. The constant survives as the
# floor and as the KOTH top-five; this setting is only the rollout gate.
DEFAULT_PRIORITY_COHORT_SIZE = 5

# ---------------------------------------------------------------------------
# Lane bounds
# ---------------------------------------------------------------------------
# The lane cycle must have at least two slots or there is no split to make.
MIN_LANE_CYCLE_SIZE = 2
# An arbitrary but deliberate ceiling. A modulus above a dozen makes the lane a
# validator visits rarely enough that a small fleet can go a long time without
# ever taking cohort work, which is the starvation this whole mechanism exists
# to prevent -- in the other direction.
MAX_LANE_CYCLE_SIZE = 12
# Mirrors ``_LANE_CYCLE_SIZE`` / ``_FRESH_SUBMISSION_SLOTS`` in
# ``ditto.api_server.endpoints.validator``: three fresh-submission jobs for
# every one rollout-cohort job, per validator.
#
# The default slot set is NOT the contiguous prefix (0, 1, 2). It is (0, 1, 3),
# so the cohort slot sits in the middle of the cycle rather than at its end.
# That is the shipped interleave and it is preserved exactly; deriving the set
# from a count would silently change which poll takes cohort work.
DEFAULT_LANE_CYCLE_SIZE = 4
DEFAULT_FRESH_SUBMISSION_SLOTS = (0, 1, 3)

# ---------------------------------------------------------------------------
# Previous-generation carryover bounds
# ---------------------------------------------------------------------------
MAX_PREV_GEN_CARRYOVER_AGENTS = 50
DEFAULT_PREV_GEN_CARRYOVER_AGENTS = 10

# Fields the queue refuses to change while a benchmark rollout is open, because
# they move the modulus of a counter that is measured from rollout start.
ROLLOUT_LOCKED_FIELDS = ("lane_cycle_size", "fresh_submission_slots")


def _as_slot_tuple(value: object) -> object:
    """Accept a JSON array for a field that is stored as an immutable tuple.

    JSON has no tuple type, so a strict ``tuple[int, ...]`` would reject every
    real request body. Coercing here keeps the wire shape an array while the
    in-memory value stays immutable, so a resolved policy cannot be mutated by a
    caller that happens to hold a reference to it.
    """
    if isinstance(value, list):
        return tuple(value)
    return value


FreshSubmissionSlots = Annotated[tuple[int, ...], BeforeValidator(_as_slot_tuple)]


class PrevGenCarryoverSettings(BaseModel):
    """When to admit previous-generation submissions that can never finalize.

    Once a new benchmark version activates, nobody will ever issue the third
    prior-version score that a partially-scored prior-generation submission is
    waiting on. Those submissions are not *delayed*, they are permanently
    stranded: the quorum they need cannot be restored by any future event.

    This policy decides which of them the new era adopts. It ships DISABLED, so
    admitting them is always an explicit operator decision with an audit trail.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    enabled: bool = False
    """Whether stranded previous-generation submissions are admitted at all.

    Ships ``False``: the entire carryover path is inert until an operator turns
    it on, so this board's arrival is a no-op.
    """

    max_agents: Annotated[int, Field(ge=1, le=MAX_PREV_GEN_CARRYOVER_AGENTS)] = (
        DEFAULT_PREV_GEN_CARRYOVER_AGENTS
    )
    """Hard cap on how many stranded submissions are adopted, ever.

    A cap rather than a rate: carryover is a finite backlog, not a stream, so
    bounding the total is the honest control. Ordering is by demonstrated
    progress first (see :attr:`min_score_count`) and then FIFO, so a cap keeps
    the submissions that are closest to finalizing.
    """

    min_score_count: Annotated[int, Field(ge=0, le=2)] = 2
    """Fewest prior-version scores a submission must already have.

    ``2`` (the default) adopts only submissions one score short of quorum. Those
    have *demonstrated they can run* -- two validators already produced a score
    for them -- so adopting them is cheap and very likely to finalize.

    ``0`` also adopts submissions that were never ticketed at all. Those have
    never been executed by any validator, so each one is an open-ended cost with
    no evidence it will ever produce a score. That is a deliberate operator
    choice, not a default.

    The ceiling is 2 because a submission with three prior-version scores is
    finalized and by definition not stranded.
    """

    include_exhausted: bool = False
    """Whether to adopt submissions that burned all their retry attempts.

    Exhausted submissions already consumed their allowance under the prior
    generation. Adopting them hands back capacity that the retry policy
    deliberately took away, so it is off by default.
    """

    dedupe_scope: Literal["coldkey", "hotkey", "none"] = "coldkey"
    """Whose newer submissions suppress their own older stranded ones.

    ``coldkey`` (the default) is the real owner boundary: one miner routinely
    runs several hotkeys, and if they have already submitted something newer
    then their older stranded work is superseded by their own choice and should
    not consume fleet capacity. This matches the owner key the ticket allocator
    itself uses (payment-time coldkey with a hotkey fallback), so carryover and
    allocation agree on who a miner is.

    ``hotkey`` dedupes per registered key instead, which lets one owner carry
    several stranded submissions. ``none`` disables suppression entirely. Both
    are strictly wider than the default.
    """

    require_cohort_complete: bool = True
    """Whether carryover waits for the inherited rescore cohort to settle.

    ``True`` (the default) confines carryover to genuinely spare cohort-lane
    capacity, exactly like the existing source-backfill path: nothing is adopted
    until the rollout's inherited cohort has finished, so carryover can never
    delay the transition it is riding on. ``False`` interleaves carryover with
    cohort work, which finishes the backlog sooner at the cost of slowing
    activation.
    """


class QueuePolicySettings(BaseModel):
    """The complete, hot-swappable validator-queue policy.

    Each revision stores the whole policy (not a diff), so a frozen snapshot is
    always reconstructable and a read never merges partial revisions.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    # -- NEXT ROLLOUT (frozen onto the rollout row at start) ----------------

    rescore_cohort_size: Annotated[
        int, Field(ge=MIN_COHORT_SIZE, le=MAX_COHORT_SIZE)
    ] = DEFAULT_RESCORE_COHORT_SIZE
    """How many inherited agents the *next* rollout re-scores on its target
    version (``5 <= n <= 25``). Frozen onto the rollout row at start."""

    priority_cohort_size: Annotated[
        int, Field(ge=MIN_COHORT_SIZE, le=MAX_COHORT_SIZE)
    ] = DEFAULT_PRIORITY_COHORT_SIZE
    """How many of the inherited cohort's top positions gate activation.

    The rollout may not take ledger authority until every one of the top
    ``priority_cohort_size`` inherited agents holds a complete target-version
    quorum. Raising it makes activation strictly more conservative -- and
    strictly more stallable, since one unqualifiable agent inside the widened
    band now blocks the transition. It can never go below
    :data:`MIN_COHORT_SIZE`, which is the KOTH emission-set size.

    Frozen onto the rollout row at start, so raising it never re-gates a
    transition already in flight.
    """

    # -- LIVE, ROLLOUT-LOCKED ---------------------------------------------

    lane_cycle_size: Annotated[
        int, Field(ge=MIN_LANE_CYCLE_SIZE, le=MAX_LANE_CYCLE_SIZE)
    ] = DEFAULT_LANE_CYCLE_SIZE
    """Length of the per-validator lane rotation.

    Each validator counts the jobs it has completed since rollout start and
    takes its lane from ``count % lane_cycle_size``. Changing this while a
    rollout is open is refused; see :data:`ROLLOUT_LOCKED_FIELDS`.
    """

    fresh_submission_slots: FreshSubmissionSlots = DEFAULT_FRESH_SUBMISSION_SLOTS
    """Which slots of the cycle serve new submissions rather than the cohort.

    Defaults to ``(0, 1, 3)`` of a 4-slot cycle: three fresh-submission jobs per
    one rollout-cohort job, with the cohort slot in the middle. Both lanes have
    a floor of one slot -- an empty fresh lane starves new miners (the failure
    this split was written to prevent), and an empty cohort lane means the
    rollout never reaches quorum and can never activate.
    """

    # -- LIVE -------------------------------------------------------------
    #
    # ``PROVISIONAL_CONTENDER_LANE_SIZE`` (the 10-agent fast lane in
    # ``ditto.db.queries.tickets``) is a deliberate omission, not an oversight.
    # It is genuinely queue policy and it has been retuned before, but its value
    # feeds three consumers -- the ``.limit()`` on the contender subquery, the
    # tenth-place floor in ``get_score_priority_floors``, and the public
    # ``validator_queue_rank`` preview -- and that preview is already known to
    # disagree with the allocator about who a miner is (it groups by hotkey
    # where the allocator groups by payment-time coldkey). Making the size
    # operator-tunable before that divergence is fixed would let one setting
    # produce two different lanes. It is tracked separately.

    prev_gen_carryover: PrevGenCarryoverSettings = PrevGenCarryoverSettings()
    """Adoption policy for stranded previous-generation submissions."""

    @model_validator(mode="after")
    def _check_coherent(self) -> QueuePolicySettings:
        """Reject combinations that individually validate but jointly deadlock.

        Field-level ranges cannot express these: each one is a relation between
        two settings, and getting any of them wrong wedges the queue rather than
        merely mistuning it.
        """
        if self.priority_cohort_size > self.rescore_cohort_size:
            raise ValueError(
                "priority_cohort_size "
                f"({self.priority_cohort_size}) cannot exceed rescore_cohort_size "
                f"({self.rescore_cohort_size}): activation would wait on inherited "
                "positions the cohort never fills, so the rollout could never "
                "activate"
            )
        slots = self.fresh_submission_slots
        if len(set(slots)) != len(slots):
            raise ValueError(
                f"fresh_submission_slots must be unique, got {list(slots)}"
            )
        if not slots:
            raise ValueError(
                "fresh_submission_slots cannot be empty: every validator poll "
                "would serve the rollout cohort and new submissions would never "
                "be scored, which is exactly the miner starvation the lane split "
                "exists to prevent"
            )
        out_of_range = [s for s in slots if not 0 <= s < self.lane_cycle_size]
        if out_of_range:
            raise ValueError(
                f"fresh_submission_slots {out_of_range} fall outside the cycle "
                f"[0, {self.lane_cycle_size}): those slots could never come up, so "
                "the fresh-submission lane would be smaller than it looks"
            )
        if len(slots) >= self.lane_cycle_size:
            raise ValueError(
                f"fresh_submission_slots {list(slots)} fills the whole "
                f"{self.lane_cycle_size}-slot cycle: no slot would ever serve the "
                "rollout cohort, so an open rollout could never reach quorum or "
                "activate"
            )
        return self

    @property
    def sorted_fresh_submission_slots(self) -> tuple[int, ...]:
        """Slots in ascending order, for stable display and checksums."""
        return tuple(sorted(self.fresh_submission_slots))

    def fresh_submission_lane_due(self, completed_jobs: int) -> bool:
        """Whether a validator's next job comes from the fresh-submission lane.

        The single authority for the lane decision, so the endpoint and any
        projection cannot drift apart the way the queue-rank preview did.
        """
        return completed_jobs % self.lane_cycle_size in set(self.fresh_submission_slots)


def rollout_locked_change(
    current: QueuePolicySettings, proposed: QueuePolicySettings
) -> tuple[str, ...]:
    """Which rollout-locked fields differ between two policies.

    Returns the offending field names, or an empty tuple when the proposal
    leaves every locked field alone. Callers refuse a non-empty result while a
    rollout is open, which lets an operator still retune everything else --
    cohort sizes, the contender lane, carryover -- mid-rollout.
    """
    changed: list[str] = []
    for field in ROLLOUT_LOCKED_FIELDS:
        left = getattr(current, field)
        right = getattr(proposed, field)
        if field == "fresh_submission_slots":
            left, right = tuple(sorted(left)), tuple(sorted(right))
        if left != right:
            changed.append(field)
    return tuple(changed)


class QueuePolicySettingsRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int
    parent_revision: int
    scope: str
    settings: QueuePolicySettings
    reason: str
    actor: str
    created_at: datetime
    checksum: str


class EffectiveQueuePolicySettings(BaseModel):
    """What the queue is using now, plus what an open rollout already froze."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int
    scope: str
    settings: QueuePolicySettings
    checksum: str
    source: Literal["revision", "default"]
    min_cohort_size: int = MIN_COHORT_SIZE
    max_cohort_size: int = MAX_COHORT_SIZE
    rollout_locked_fields: tuple[str, ...] = ROLLOUT_LOCKED_FIELDS
    """Fields that cannot be changed while a rollout is open."""
    rollout_is_open: bool = False
    """Whether the locked fields are currently frozen."""
    open_rollout_desired_version: int | None = None
    """Target version of the open rollout, if any."""
    open_rollout_rescore_cohort_target: int | None = None
    """The rescore size that rollout froze at start. Immune to later revisions."""
    open_rollout_priority_cohort_target: int | None = None
    """The priority size that rollout froze at start. Immune to later revisions."""
    open_rollout_overrides_setting: bool = False
    """True when an open rollout froze a cohort size other than the configured
    one, so the operator can see the new value only applies to the next
    rollout."""


class AdminQueuePolicySettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scope: str = "*"
    expected_revision: Annotated[int, Field(ge=0)]
    settings: QueuePolicySettings
    reason: Annotated[str, Field(min_length=8, max_length=500)]
    actor: Annotated[str, Field(min_length=1, max_length=120)] = "admin_api"
    confirmation: str

    @model_validator(mode="after")
    def _require_complete_policy(self) -> AdminQueuePolicySettingsRequest:
        """Reject a partial policy on a whole-object store.

        :class:`QueuePolicySettings` has a default for every field, which is what
        makes a board with no revision behave like the old hard-coded queue. On a
        *write* that same convenience is a footgun: a revision stores the whole
        policy, so an operator who sends only ``{"rescore_cohort_size": 25}``
        would silently reset every other knob to its shipped default -- turning
        off a carryover gate they had enabled, or reverting a lane split -- while
        believing they changed one number.

        ``expected_revision`` cannot catch this: the operator holds the current
        revision, they just under-specified the body. So require every field to
        be present explicitly and name the missing ones.
        """
        missing = sorted(
            set(QueuePolicySettings.model_fields) - self.settings.model_fields_set
        )
        if missing:
            raise ValueError(
                "a queue policy revision stores the WHOLE policy, so every field "
                "must be sent explicitly; missing "
                f"{missing}. Read GET /admin/queue-policy-settings, change the "
                "fields you want, and send back the complete object."
            )
        carryover_missing = sorted(
            set(PrevGenCarryoverSettings.model_fields)
            - self.settings.prev_gen_carryover.model_fields_set
        )
        if carryover_missing:
            raise ValueError(
                f"prev_gen_carryover is stored whole too; missing {carryover_missing}"
            )
        return self


class AdminQueuePolicySettingsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    current: list[QueuePolicySettingsRevision]
    history: list[QueuePolicySettingsRevision]
    default: QueuePolicySettings
    effective: EffectiveQueuePolicySettings
