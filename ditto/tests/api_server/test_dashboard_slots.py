"""Behavioural tests for the dashboard's multi-slot fleet rendering.

The dashboard is a single vanilla-JS file with no bundler or JS test runner, so
the rest of its suite asserts on served substrings. Slot fan-out is logic rather
than markup, and a substring cannot tell "renders two jobs" from "renders one
job twice", so these tests lift the real functions out of ``index.html`` and
execute them under node against fixture heartbeats.

Skipped when node is unavailable so the suite still runs on a bare checkout.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

_DASHBOARD = Path(__file__).parents[3] / "dashboard" / "index.html"
_NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(_NODE is None, reason="node is required")


def _extract(name: str) -> str:
    """Return the source of one top-level ``function <name>(...)`` block."""
    body = _DASHBOARD.read_text()
    start = body.index(f"function {name}(")
    depth = 0
    for index in range(start, len(body)):
        if body[index] == "{":
            depth += 1
        elif body[index] == "}":
            depth -= 1
            if depth == 0:
                return body[start : index + 1]
    raise AssertionError(f"unbalanced braces extracting {name}")


def _run(entry: Mapping[str, object], call: str) -> object:
    script = "\n".join(
        [
            _extract("validatorSlotIds"),
            _extract("slotOrdinal"),
            _extract("anyBenchmarkStage"),
            _extract("cappedSlotIds"),
            _extract("fundedSlotCount"),
            f"const entry = {json.dumps(dict(entry))};",
            f"process.stdout.write(JSON.stringify({call}));",
        ]
    )
    assert _NODE is not None
    result = subprocess.run(
        [_NODE, "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _benchmark(slot_id: str, *, stage: str = "running_benchmark") -> dict[str, object]:
    return {
        "slot_id": slot_id,
        "agent_id": f"agent-{slot_id}",
        "agent_name": f"Agent {slot_id}",
        "bench_version": 7,
        "stage": stage,
        "completed_checks": 10,
        "total_checks": 282,
        "percent": 5,
        "stalled": False,
    }


class TestValidatorSlotIds:
    def test_renders_one_row_per_configured_slot(self) -> None:
        entry = {
            "configured_slots": 4,
            "healthy_slots": ["slot-0", "slot-1", "slot-2", "slot-3"],
            "admission": "accepting",
            "active_benchmarks": [_benchmark("slot-0"), _benchmark("slot-2")],
            "assigned_benchmarks": [],
        }

        assert _run(entry, "validatorSlotIds(entry)") == [
            "slot-0",
            "slot-1",
            "slot-2",
            "slot-3",
        ]

    def test_two_concurrent_jobs_are_both_addressable(self) -> None:
        """The regression this change exists for: two jobs, two distinct rows."""
        entry = {
            "configured_slots": 2,
            "healthy_slots": ["slot-0", "slot-1"],
            "admission": "accepting",
            "active_benchmarks": [_benchmark("slot-0"), _benchmark("slot-1")],
            "assigned_benchmarks": [],
        }
        slots = _run(entry, "validatorSlotIds(entry)")

        assert slots == ["slot-0", "slot-1"]
        assert len(set(slots)) == 2

    def test_active_slot_outside_configured_range_is_still_shown(self) -> None:
        """A job must never be dropped just because the count looks smaller.

        Synthesising ids from ``configured_slots`` alone hid exactly this case,
        which is when an operator most needs to see the work.
        """
        entry = {
            "configured_slots": 1,
            "healthy_slots": ["slot-0"],
            "admission": "accepting",
            "active_benchmarks": [_benchmark("slot-3")],
            "assigned_benchmarks": [],
        }

        assert _run(entry, "validatorSlotIds(entry)") == ["slot-0", "slot-3"]

    def test_assigned_but_not_yet_active_slots_appear(self) -> None:
        entry = {
            "configured_slots": 1,
            "healthy_slots": ["slot-0"],
            "admission": "accepting",
            "active_benchmarks": [],
            "assigned_benchmarks": [_benchmark("slot-1")],
        }

        assert _run(entry, "validatorSlotIds(entry)") == ["slot-0", "slot-1"]

    def test_slots_order_numerically_not_lexically(self) -> None:
        entry = {
            "configured_slots": 1,
            "healthy_slots": [],
            "admission": "accepting",
            "active_benchmarks": [_benchmark("slot-2"), _benchmark("slot-1")],
            "assigned_benchmarks": [],
        }

        assert _run(entry, "validatorSlotIds(entry)") == [
            "slot-0",
            "slot-1",
            "slot-2",
        ]

    def test_single_slot_validator_is_unchanged(self) -> None:
        entry = {
            "configured_slots": 1,
            "healthy_slots": ["slot-0"],
            "admission": "accepting",
            "active_benchmarks": [_benchmark("slot-0")],
            "assigned_benchmarks": [],
        }

        assert _run(entry, "validatorSlotIds(entry)") == ["slot-0"]

    def test_missing_capacity_fields_fall_back_to_one_slot(self) -> None:
        empty: dict[str, object] = {}

        assert _run(empty, "validatorSlotIds(entry)") == ["slot-0"]


class TestAnyBenchmarkStage:
    def test_stage_on_a_higher_slot_counts_as_progress(self) -> None:
        """Keying off slot zero alone suppressed live progress for slot one."""
        entry = {
            "active_benchmarks": [_benchmark("slot-1")],
            "active_benchmark": None,
        }

        assert _run(entry, "anyBenchmarkStage(entry)") is True

    def test_idle_validator_reports_no_granular_progress(self) -> None:
        entry: dict[str, object] = {
            "active_benchmarks": [],
            "active_benchmark": None,
        }

        assert _run(entry, "anyBenchmarkStage(entry)") is False

    def test_legacy_single_benchmark_still_counts(self) -> None:
        entry = {"active_benchmarks": [], "active_benchmark": _benchmark("slot-0")}

        assert _run(entry, "anyBenchmarkStage(entry)") is True


class TestCappedSlots:
    """Advertised capacity above the operator cap must not read as idle.

    The fleet ran eight advertised slots under a cap of six, and the table drew
    eight rows the operator could reasonably read as eight usable slots. Two of
    them were never going to receive a ticket.
    """

    def _capped(self, entry: Mapping[str, object]) -> list[str]:
        result = _run(entry, "cappedSlotIds(entry)")
        assert isinstance(result, dict)
        return sorted(result, key=lambda slot: int(slot.rsplit("-", 1)[1]))

    def test_slots_above_the_cap_are_marked(self) -> None:
        """Eight advertised, cap of six: the top two are capped, not idle."""
        entry = {
            "configured_slots": 8,
            "allowed_slots": 6,
            "healthy_slots": [f"slot-{index}" for index in range(8)],
            "admission": "accepting",
            "active_benchmarks": [],
            "assigned_benchmarks": [],
        }

        assert self._capped(entry) == ["slot-6", "slot-7"]

    def test_a_validator_inside_the_cap_has_nothing_capped(self) -> None:
        entry = {
            "configured_slots": 4,
            "allowed_slots": 4,
            "healthy_slots": [f"slot-{index}" for index in range(4)],
            "admission": "accepting",
            "active_benchmarks": [],
            "assigned_benchmarks": [],
        }

        assert self._capped(entry) == []

    def test_running_work_is_charged_first_and_never_marked(self) -> None:
        """Lowering the cap costs new leases only, never one in flight."""
        entry = {
            "configured_slots": 4,
            "allowed_slots": 2,
            "healthy_slots": [f"slot-{index}" for index in range(4)],
            "admission": "accepting",
            "active_benchmarks": [_benchmark("slot-1")],
            "assigned_benchmarks": [],
        }

        # slot-1 runs and spends one of the two, slot-0 takes the other.
        assert self._capped(entry) == ["slot-2", "slot-3"]

    def test_leases_beyond_the_cap_do_not_borrow_from_idle_slots(self) -> None:
        """A cap dropped under live work caps every idle slot, not a negative."""
        entry = {
            "configured_slots": 4,
            "allowed_slots": 1,
            "healthy_slots": [f"slot-{index}" for index in range(4)],
            "admission": "accepting",
            "active_benchmarks": [_benchmark("slot-0"), _benchmark("slot-1")],
            "assigned_benchmarks": [],
        }

        assert self._capped(entry) == ["slot-2", "slot-3"]

    def test_an_unhealthy_slot_keeps_its_own_state(self) -> None:
        """Unavailable is about the validator; capped is about the operator."""
        entry = {
            "configured_slots": 4,
            "allowed_slots": 2,
            "healthy_slots": ["slot-0", "slot-2", "slot-3"],
            "admission": "accepting",
            "active_benchmarks": [],
            "assigned_benchmarks": [],
        }

        # slot-1 is unhealthy, so it is not the cap keeping work off it.
        assert self._capped(entry) == ["slot-3"]

    def test_a_payload_without_the_field_marks_nothing(self) -> None:
        """A dashboard served against an older API must not invent a cap."""
        entry = {
            "configured_slots": 8,
            "healthy_slots": [f"slot-{index}" for index in range(8)],
            "admission": "accepting",
            "active_benchmarks": [],
            "assigned_benchmarks": [],
        }

        assert self._capped(entry) == []


class TestFundedSlotCount:
    """The fleet table's numerator: what can take work, not what is advertised."""

    def test_the_cap_binds_below_healthy_capacity(self) -> None:
        entry = {
            "configured_slots": 8,
            "allowed_slots": 6,
            "healthy_slots": [f"slot-{index}" for index in range(8)],
        }

        assert _run(entry, "fundedSlotCount(entry)") == 6

    def test_health_binds_below_the_cap(self) -> None:
        entry = {
            "configured_slots": 8,
            "allowed_slots": 6,
            "healthy_slots": ["slot-0", "slot-1"],
        }

        assert _run(entry, "fundedSlotCount(entry)") == 2

    def test_without_the_field_it_falls_back_to_healthy_slots(self) -> None:
        entry = {"configured_slots": 4, "healthy_slots": ["slot-0", "slot-1"]}

        assert _run(entry, "fundedSlotCount(entry)") == 2


class TestDashboardSource:
    def test_slot_ids_are_not_synthesised_from_a_count(self) -> None:
        """The old loop built ids as ``"slot-" + index`` and dropped the rest."""
        body = _DASHBOARD.read_text()

        assert 'var slotId = "slot-" + slotIndex;' not in body
        assert "validatorSlotIds(entry)" in body

    def test_per_slot_rows_have_their_own_style_hook(self) -> None:
        body = _DASHBOARD.read_text()

        assert 'class="fleet-slot"' in body
        assert ".fleet-slot + .fleet-slot" in body

    def test_a_capped_slot_is_rendered_as_its_own_state(self) -> None:
        """Not "Idle" (claims usable capacity) and not "Unavailable" (a fault)."""
        body = _DASHBOARD.read_text()

        assert 'class="stage capped"' in body
        assert ".stage.capped {" in body

    def test_the_fleet_row_reports_funded_capacity(self) -> None:
        """The numerator is what dispatch funds, not what is advertised."""
        body = _DASHBOARD.read_text()

        assert 'String(fundedSlotCount(entry)) + " of "' in body
        assert 'String((entry.healthy_slots || []).length) + "/"' not in body

    def test_detail_modal_renders_every_active_slot(self) -> None:
        body = _DASHBOARD.read_text()
        modal_start = body.index("function renderValidatorDetail(entry)")
        modal_end = body.index("function openValidatorEntry(", modal_start)
        modal = body[modal_start:modal_end]

        assert "activeSlots.slice().sort(" in modal
        assert re.search(r'vstat\("Slots"', modal) is not None
