"""Versioned, hot-swappable operator policy for how many concurrent benchmark
SLOTS a single validator may hold live tickets for.

A validator advertises its own capacity in the heartbeat
(``BenchmarkCapacity.configured_slots``, bounded to eight by the protocol), and
until now the platform simply honored whatever was advertised: there was no
lever at all. These models back an append-only revision table so an operator can
cap the fleet — instantly, from backroom, with no redeploy. It is both the kill
switch (drop to 1 and multi-slot dispatch stops on the next ticket issue) and
the gradual-ramp control (2 -> 3 -> 4 as confidence grows).

Unlike the efficiency-bonus settings, this policy has **no pre-existing env
var**, so there is no seed to overlay: the module-level default in
``ditto.api_server.validator_slot_settings`` governs when no revision exists,
and every failure path falls back to it rather than to "uncapped".

Each revision carries the COMPLETE policy (never a diff), so a read never has to
merge partial revisions and a historical row always reproduces exactly what was
in force.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

HARD_SLOT_CEILING = 8
"""The protocol's own maximum advertised slots (``^slot-[0-7]$``, see
``ditto.api_models.benchmark_capacity``). The operator cap can narrow the fleet
but can never widen it past what a validator is able to advertise."""

DISK_PERCENT_QUANTUM = 5
"""``SystemMetrics.disk_percent`` is reported on a 5% grid (``multiple_of=5``).
A ceiling off that grid would fire at the next grid point up and so silently
misdescribe itself (a ceiling of 87 behaves exactly like 90), which is why the
envelope check below rejects it."""


class ValidatorSlotSettings(BaseModel):
    """The full, hot-swappable validator-slot policy.

    Both knobs are independently settable in a single revision, and a revision
    always stores both, so the row is a complete description of the policy that
    was in force -- there is no partial-revision merge on the read path.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    max_concurrent_slots: Annotated[int, Field(ge=1, le=HARD_SLOT_CEILING)] = 2
    """Maximum benchmark slots the platform will issue live tickets for on any
    ONE validator, regardless of how many it advertises (``1 <= n <= 8``).

    Deliberately defaults to 2, not 8: a fleet that has never been tuned runs
    conservatively, and an operator ramps up explicitly. ``1`` is the kill
    switch -- it restores strictly serial, one-ticket-at-a-time dispatch."""

    disk_percent_ceiling: Annotated[int, Field(ge=50, le=100)] = 90
    """Disk-utilization circuit breaker, as a percentage on the heartbeat's 5%
    grid (``50 <= n <= 100``, multiple of 5).

    When a validator's MOST RECENT heartbeat reports
    ``system_metrics.disk_percent`` at or above this, that validator is held to
    a single slot: parallel benchmark slots multiply image pulls and container
    layers, which is exactly what a nearly-full host cannot absorb.

    Evaluated only at ticket ISSUE time. It never revokes a live lease, so a
    validator that crosses the ceiling mid-benchmark still runs and reports the
    work it already holds to completion, and the restriction lifts on its own as
    soon as a fresh heartbeat reports headroom again."""

    @model_validator(mode="after")
    def validate_envelope(self) -> ValidatorSlotSettings:
        """Invariants the per-field bounds cannot express on their own."""
        if self.max_concurrent_slots > HARD_SLOT_CEILING:
            raise ValueError(
                "max_concurrent_slots cannot exceed the protocol slot ceiling "
                f"of {HARD_SLOT_CEILING}"
            )
        if self.disk_percent_ceiling % DISK_PERCENT_QUANTUM:
            raise ValueError(
                "disk_percent_ceiling must be a multiple of "
                f"{DISK_PERCENT_QUANTUM} because heartbeat disk_percent is "
                "reported on that grid"
            )
        return self


class ValidatorSlotSettingsRevision(BaseModel):
    """One append-only, operator-audited revision of the slot policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int
    parent_revision: int
    scope: str
    settings: ValidatorSlotSettings
    reason: str
    actor: str
    created_at: datetime
    checksum: str


class EffectiveValidatorSlotSettings(BaseModel):
    """What the dispatch path actually reads: the latest revision (or the
    module default when none exists), plus provenance for the operator
    console."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int
    """The governing revision number, or 0 when no revision exists."""

    scope: str
    settings: ValidatorSlotSettings
    checksum: str
    source: Literal["revision", "default"]
    """``revision`` when a stored revision governs, ``default`` for the
    module-level default (which is also every failure path's fallback)."""

    hard_slot_ceiling: int = HARD_SLOT_CEILING
    """The protocol maximum a validator can advertise -- the top of the ramp the
    operator console renders ``max_concurrent_slots`` against."""

    disk_restricted_slots: int = 1
    """How many slots a validator is held to once ``disk_percent_ceiling`` is
    tripped."""

    max_age_seconds: float
    """The resolver TTL: an upper bound on how long a backroom change can take
    to land on the dispatch path (0 means every read re-reads)."""


class AdminValidatorSlotSettingsRequest(BaseModel):
    """Append one optimistic, confirmation-gated revision."""

    model_config = ConfigDict(extra="forbid", strict=True)

    scope: str = "*"
    """Slot policy is subnet-global; only ``*`` is accepted."""

    expected_revision: Annotated[int, Field(ge=0)]
    """The revision the operator believes is current (0 = none yet). A mismatch
    is a 409 so a concurrent change is never silently clobbered."""

    settings: ValidatorSlotSettings
    reason: Annotated[str, Field(min_length=8, max_length=500)]
    actor: Annotated[str, Field(min_length=1, max_length=120)] = "admin_api"
    confirmation: str
    """Must equal ``APPLY VALIDATOR SLOT CAP <max_concurrent_slots>`` (the
    resulting cap), typed exactly -- so the number being applied is stated twice
    and a fat-fingered ramp cannot land silently."""


class AdminValidatorSlotSettingsResponse(BaseModel):
    """Current policy per scope, append-only history, the module default, and
    the settings actually in force right now."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    current: list[ValidatorSlotSettingsRevision]
    history: list[ValidatorSlotSettingsRevision]
    default: ValidatorSlotSettings
    effective: EffectiveValidatorSlotSettings
