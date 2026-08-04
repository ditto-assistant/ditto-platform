from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[3]
RELAY_PATH_FILTER = ROOT / "scripts" / "relay-runtime-changed.sh"


def _relay_changed(*paths: str) -> bool:
    result = subprocess.run(
        [str(RELAY_PATH_FILTER)],
        input="\n".join(paths),
        text=True,
        check=False,
    )
    return result.returncode == 0


def test_relay_change_filter_ignores_tests_and_dashboard_assets() -> None:
    assert not _relay_changed(
        "dashboard/index.html",
        "ditto/tests/api_server/test_dashboard.py",
    )


def test_relay_change_filter_detects_runtime_and_release_changes() -> None:
    assert _relay_changed("ditto/api_server/inference.py")
    assert _relay_changed("alembic/versions/123_add_column.py")
    assert _relay_changed("uv.lock")
    assert _relay_changed("scripts/deploy-relay-release.sh")
