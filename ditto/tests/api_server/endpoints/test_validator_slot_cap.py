"""The operator slot cap that bounds concurrent benchmark leases per validator.

Validators advertise the slot capacity their host can offer; how much of that
the fleet actually uses is an operator decision that must be changeable from
backroom without a release. These tests pin the decision boundary itself --
which slot ordinals are served, and what happens when the inputs are missing or
malformed -- because the failure mode of getting it wrong is fleet-wide.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ditto.api_models.validator_slot_settings import ValidatorSlotSettings
from ditto.api_server.endpoints.validator import (
    _heartbeat_disk_percent,
    _slot_ordinal,
    _validator_slot_settings,
)
from ditto.api_server.validator_slot_settings import (
    DEFAULT_SETTINGS,
    DISK_RESTRICTED_SLOTS,
    allowed_slot_count,
    disk_ceiling_tripped,
)


def _serves(slot_id: str, *, allowed: int) -> bool:
    """Mirror the endpoint's decision: a slot is served below the cap."""
    return _slot_ordinal(slot_id) < allowed


class TestSlotOrdinal:
    @pytest.mark.parametrize(
        ("slot_id", "expected"),
        [("slot-0", 0), ("slot-1", 1), ("slot-7", 7)],
    )
    def test_reads_the_ordinal(self, slot_id: str, expected: int) -> None:
        assert _slot_ordinal(slot_id) == expected

    @pytest.mark.parametrize(
        "slot_id",
        ["", "slot", "slot-", "slot-x", "SLOT-1", "slot-01x", "slot--1", "0", "-1"],
    )
    def test_unparseable_ids_sort_above_every_cap(self, slot_id: str) -> None:
        """An unrecognised id must be declined, never read as slot zero."""
        assert not _serves(slot_id, allowed=8)


class TestAllowedSlotCount:
    def test_default_policy_permits_two_slots(self) -> None:
        assert DEFAULT_SETTINGS.max_concurrent_slots == 2

    def test_cap_narrows_what_the_validator_advertises(self) -> None:
        settings = ValidatorSlotSettings(max_concurrent_slots=2)

        assert allowed_slot_count(settings, advertised_slots=4, disk_percent=35) == 2

    def test_cap_never_grants_more_than_advertised(self) -> None:
        """The platform may only ever narrow the host's own offer."""
        settings = ValidatorSlotSettings(max_concurrent_slots=8)

        assert allowed_slot_count(settings, advertised_slots=1, disk_percent=35) == 1

    def test_cap_of_one_restores_todays_behaviour(self) -> None:
        settings = ValidatorSlotSettings(max_concurrent_slots=1)

        assert _serves(
            "slot-0",
            allowed=allowed_slot_count(settings, advertised_slots=4, disk_percent=35),
        )
        assert not _serves(
            "slot-1",
            allowed=allowed_slot_count(settings, advertised_slots=4, disk_percent=35),
        )

    def test_slots_at_or_above_the_cap_are_declined(self) -> None:
        allowed = allowed_slot_count(
            ValidatorSlotSettings(max_concurrent_slots=2),
            advertised_slots=4,
            disk_percent=35,
        )

        assert [_serves(f"slot-{index}", allowed=allowed) for index in range(4)] == [
            True,
            True,
            False,
            False,
        ]


class TestDiskCeiling:
    def test_a_nearly_full_host_is_held_to_one_slot(self) -> None:
        settings = ValidatorSlotSettings(
            max_concurrent_slots=4, disk_percent_ceiling=90
        )

        assert (
            allowed_slot_count(settings, advertised_slots=4, disk_percent=95)
            == DISK_RESTRICTED_SLOTS
        )

    def test_a_host_below_the_ceiling_keeps_its_slots(self) -> None:
        settings = ValidatorSlotSettings(
            max_concurrent_slots=2, disk_percent_ceiling=90
        )

        assert allowed_slot_count(settings, advertised_slots=4, disk_percent=80) == 2

    def test_unknown_disk_does_not_trip_the_breaker(self) -> None:
        """A validator reporting no metrics must not silently lose capacity."""
        settings = ValidatorSlotSettings(
            max_concurrent_slots=2, disk_percent_ceiling=90
        )

        assert not disk_ceiling_tripped(settings, disk_percent=None)
        assert allowed_slot_count(settings, advertised_slots=4, disk_percent=None) == 2

    def test_the_breaker_can_only_reduce_never_raise(self) -> None:
        settings = ValidatorSlotSettings(
            max_concurrent_slots=1, disk_percent_ceiling=90
        )

        assert allowed_slot_count(settings, advertised_slots=1, disk_percent=99) == 1


class TestLiveLeaseExemption:
    """Lowering the cap must cost the fleet new work only, never live work.

    Every path that resumes an in-flight lease sits downstream of the cap gate,
    so a slot already running a benchmark has to be let through. Without this,
    the instant-revert lever would strand a lease on each removed ordinal until
    it expired 90 minutes later, burning a retry attempt each time.
    """

    def _gate_declines(self, *, slot_id: str, running: bool, allowed: int) -> bool:
        return not running and _slot_ordinal(slot_id) >= allowed

    def test_a_running_slot_above_the_cap_is_still_served(self) -> None:
        assert not self._gate_declines(slot_id="slot-3", running=True, allowed=2)

    def test_an_idle_slot_above_the_cap_is_declined(self) -> None:
        assert self._gate_declines(slot_id="slot-3", running=False, allowed=2)

    def test_dropping_the_cap_to_one_still_serves_every_running_slot(self) -> None:
        running = {"slot-0", "slot-1", "slot-2", "slot-3"}

        declined = [
            slot_id
            for slot_id in sorted(running)
            if self._gate_declines(slot_id=slot_id, running=True, allowed=1)
        ]

        assert declined == []

    def test_new_work_stops_immediately_when_the_cap_drops(self) -> None:
        idle = ["slot-0", "slot-1", "slot-2", "slot-3"]

        served = [
            slot_id
            for slot_id in idle
            if not self._gate_declines(slot_id=slot_id, running=False, allowed=1)
        ]

        assert served == ["slot-0"]


class TestHeartbeatDiskPercent:
    def test_reads_the_reported_percentage(self) -> None:
        heartbeat = MagicMock(system_metrics={"disk_percent": 80})

        assert _heartbeat_disk_percent(heartbeat) == 80

    @pytest.mark.parametrize(
        "metrics",
        [None, {}, {"disk_percent": None}, {"disk_percent": "80"}, "not-a-dict"],
    )
    def test_missing_or_malformed_metrics_read_as_unknown(
        self, metrics: object
    ) -> None:
        heartbeat = MagicMock(system_metrics=metrics)

        assert _heartbeat_disk_percent(heartbeat) is None

    def test_absent_heartbeat_reads_as_unknown(self) -> None:
        assert _heartbeat_disk_percent(None) is None

    def test_booleans_are_not_percentages(self) -> None:
        """``True`` is an int in Python; it must not be read as 1% disk use."""
        heartbeat = MagicMock(system_metrics={"disk_percent": True})

        assert _heartbeat_disk_percent(heartbeat) is None


class TestResolverFailsClosed:
    async def test_a_missing_resolver_uses_the_conservative_default(self) -> None:
        """A DB or wiring failure must never uncap the fleet."""
        request = MagicMock()
        request.app.state = MagicMock(spec=[])

        assert await _validator_slot_settings(request) == DEFAULT_SETTINGS

    async def test_resolver_without_a_session_maker_uses_the_default(self) -> None:
        from ditto.api_server.validator_slot_settings import (
            ValidatorSlotSettingsResolver,
        )

        request = MagicMock()
        request.app.state.validator_slot_settings = ValidatorSlotSettingsResolver()
        request.app.state.session_maker = None

        assert await _validator_slot_settings(request) == DEFAULT_SETTINGS
