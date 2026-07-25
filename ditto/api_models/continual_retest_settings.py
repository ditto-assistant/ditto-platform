"""Hot-swappable operator policy for continual top-five retests."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# The KOTH emission set is five (champion + four-entry participation tail) and is
# frozen consensus shared with ``ditto-subnet``'s weight fold. It is the floor of
# the retest cohort, never a value the operator can lower: the shared-seed wave
# exists to keep exactly those five comparable, so a cohort that excluded one of
# them would stop the lane doing its only consensus-relevant job.
EMISSION_SET_SIZE = 5
# Ceiling on how far past the emission set an operator may extend retesting.
# Every extra member is real validator work on every wave seed, so the cap keeps
# an accidental "retest everyone" from consuming the fleet.
MAX_RETEST_COHORT_SIZE = 25


class ContinualRetestSettings(BaseModel):
    """Complete subnet-global continual-retest policy stored per revision."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    aggregate_mode: Literal["disabled", "fleet_ready", "enabled"] = "fleet_ready"
    idle_retests_enabled: bool = False
    rollout_standdown: Literal["off", "capable_validators", "all"] = (
        "capable_validators"
    )
    retest_cohort_size: Annotated[
        int, Field(ge=EMISSION_SET_SIZE, le=MAX_RETEST_COHORT_SIZE)
    ] = EMISSION_SET_SIZE
    """How many ranked agents the continual lane may rescore.

    ``5`` (the default) is the historical behaviour: the lane rescores exactly
    the emission set. Above that, the next ranked distinct-miner entrants join
    the same champion-anchored wave, so a challenger arrives at the top five
    with confirmation depth already banked instead of starting from zero.

    Raising this never changes who earns emissions, and never changes when a
    wave completes -- completion stays keyed to the emission set, so the extended
    members cannot stall the crown. They ride each wave seed with whatever
    capacity is left once every emission-set member is claimed.
    """


class ContinualRetestSettingsRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int
    parent_revision: int
    scope: str
    settings: ContinualRetestSettings
    reason: str
    actor: str
    created_at: datetime
    checksum: str


class EffectiveContinualRetestSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int
    scope: str
    settings: ContinualRetestSettings
    checksum: str
    source: Literal["revision", "default"]
    fleet_protocol_ready: bool
    aggregate_active: bool
    max_age_seconds: float
    open_rollout_desired_version: int | None = None
    rollout_standdown_active: bool = False
    emission_set_size: int = EMISSION_SET_SIZE
    max_retest_cohort_size: int = MAX_RETEST_COHORT_SIZE
    eligible_agent_count: int | None = None
    """Ranked agents the active generation could supply to the cohort right now.

    The cohort is the smaller of ``retest_cohort_size`` and this, so an operator
    who asks for 25 in a field of nine can see why only nine are being retested.
    ``None`` when the count could not be read.
    """


class AdminContinualRetestSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scope: str = "*"
    expected_revision: Annotated[int, Field(ge=0)]
    settings: ContinualRetestSettings
    reason: Annotated[str, Field(min_length=8, max_length=500)]
    actor: Annotated[str, Field(min_length=1, max_length=120)] = "admin_api"
    confirmation: str


class AdminContinualRetestSettingsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    current: list[ContinualRetestSettingsRevision]
    history: list[ContinualRetestSettingsRevision]
    default: ContinualRetestSettings
    effective: EffectiveContinualRetestSettings
