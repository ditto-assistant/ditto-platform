"""Versioned, hot-swappable operator policy for benchmark-rollout sizing.

The *rescore cohort* is the set of inherited, already-scored agents that a new
benchmark transition re-scores on the target version before that version may
take ledger authority. Its size has always been the hard-coded ten of
``ditto.db.queries.benchmark_rollout.DEFAULT_RESCORE_COHORT_SIZE``. As the
subnet scales an operator wants to widen it toward the persisted storage
ceiling of twenty-five so more agents carry a target-version score into the new
era, without a redeploy.

These wire models back an append-only revision table, exactly like
``ditto.api_models.continual_retest_settings``. A board with no revision
behaves byte-identically to the pre-change hard-coded ten.

**The configured value is a policy default, not live state.** It is read once,
when an operator starts a rollout, and immediately frozen onto the rollout row
(``BenchmarkRollout.rescore_cohort_target``). Changing it never resizes an
in-flight rollout; see ``docs/benchmark-v3-rollout.md``.

The bounds below are spelled out rather than imported from
``ditto.db.queries.benchmark_rollout``: that module imports this package, so
importing back would be a cycle.
``test_rescore_cohort_setting_bounds_match_query_constants`` asserts the two
agree, so drifting them apart fails CI.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# Mirrors ``PRIORITY_COHORT_SIZE``: the rescore cohort can never be smaller
# than the priority top five, which is also the KOTH emission set.
MIN_RESCORE_COHORT_SIZE = 5
# Mirrors ``MAX_PERSISTED_RESCORE_COHORT_SIZE`` and the
# ``benchmark_rollout_bounded_members`` CHECK constraint on
# ``benchmark_rollouts.cohort_size``. A larger value could not be persisted.
MAX_RESCORE_COHORT_SIZE = 25
# Mirrors ``DEFAULT_RESCORE_COHORT_SIZE``: the inherited top ten, which is what
# every rollout used before this setting existed.
DEFAULT_RESCORE_COHORT_SIZE = 10


class BenchmarkRolloutSettings(BaseModel):
    """The complete, hot-swappable benchmark-rollout policy.

    Each revision stores the whole policy (not a diff), so a frozen snapshot is
    always reconstructable and a read never merges partial revisions.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    rescore_cohort_size: Annotated[
        int, Field(ge=MIN_RESCORE_COHORT_SIZE, le=MAX_RESCORE_COHORT_SIZE)
    ] = DEFAULT_RESCORE_COHORT_SIZE
    """How many inherited agents the *next* rollout re-scores on its target
    version (``5 <= n <= 25``). Frozen onto the rollout row at start."""


class BenchmarkRolloutSettingsRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int
    parent_revision: int
    scope: str
    settings: BenchmarkRolloutSettings
    reason: str
    actor: str
    created_at: datetime
    checksum: str


class EffectiveBenchmarkRolloutSettings(BaseModel):
    """What the next rollout start will use, plus what the open one froze."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int
    scope: str
    settings: BenchmarkRolloutSettings
    checksum: str
    source: Literal["revision", "default"]
    min_rescore_cohort_size: int = MIN_RESCORE_COHORT_SIZE
    max_rescore_cohort_size: int = MAX_RESCORE_COHORT_SIZE
    open_rollout_desired_version: int | None = None
    """Target version of the open rollout, if any."""
    open_rollout_rescore_cohort_target: int | None = None
    """The size that rollout froze at start. Immune to later revisions."""
    open_rollout_overrides_setting: bool = False
    """True when an open rollout froze a size other than the configured one, so
    the operator can see the new value only applies to the next rollout."""


class AdminBenchmarkRolloutSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scope: str = "*"
    expected_revision: Annotated[int, Field(ge=0)]
    settings: BenchmarkRolloutSettings
    reason: Annotated[str, Field(min_length=8, max_length=500)]
    actor: Annotated[str, Field(min_length=1, max_length=120)] = "admin_api"
    confirmation: str


class AdminBenchmarkRolloutSettingsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    current: list[BenchmarkRolloutSettingsRevision]
    history: list[BenchmarkRolloutSettingsRevision]
    default: BenchmarkRolloutSettings
    effective: EffectiveBenchmarkRolloutSettings
