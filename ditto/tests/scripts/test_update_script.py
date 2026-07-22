from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[3]


def _write_executable(path: Path, source: str) -> None:
    path.write_text(f"#!/usr/bin/env bash\nset -eu\n{source}")
    path.chmod(0o755)


def _run_update(
    tmp_path: Path,
    *,
    gcloud_source: str,
    initial_env: str = "BASE_SETTING=kept\n",
) -> tuple[subprocess.CompletedProcess[str], str]:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    fake_bin.mkdir()
    shutil.copy2(ROOT / "scripts" / "update.sh", scripts / "update.sh")
    (repo / ".env").write_text(initial_env)

    _write_executable(
        fake_bin / "git",
        'if [ "${1:-}" = "rev-parse" ]; then printf "main\\n"; fi\n',
    )
    _write_executable(fake_bin / "uv", ":\n")
    _write_executable(fake_bin / "docker", ":\n")
    _write_executable(
        fake_bin / "pm2",
        'if [ "${1:-}" = "describe" ]; then exit 1; fi\n',
    )
    _write_executable(fake_bin / "gcloud", gcloud_source)
    _write_executable(fake_bin / "timeout", 'shift\nexec "$@"\n')

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run(
        [str(scripts / "update.sh")],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    return result, (repo / ".env").read_text()


def test_update_loads_taostats_key_without_logging_value(tmp_path: Path) -> None:
    api_key = "tao-test:example"
    result, deployed_env = _run_update(
        tmp_path,
        gcloud_source=f'printf "%s\\n" "{api_key}"\n',
    )

    assert result.returncode == 0, result.stderr
    assert f"DITTO_TAOSTATS_API_KEY={api_key}" in deployed_env
    assert (
        "DITTO_TAOSTATS_VALIDATOR_NAMES_URL="
        "https://api.taostats.io/api/dtao/validator/available/v1?netuid=118"
    ) in deployed_env
    assert api_key not in result.stdout
    assert api_key not in result.stderr


def test_update_keeps_existing_enrichment_when_secret_is_unavailable(
    tmp_path: Path,
) -> None:
    initial_env = (
        "BASE_SETTING=kept\n"
        "DITTO_TAOSTATS_API_KEY=existing-key\n"
        "DITTO_TAOSTATS_VALIDATOR_NAMES_URL=https://example.invalid/names\n"
    )
    result, deployed_env = _run_update(
        tmp_path,
        gcloud_source="exit 1\n",
        initial_env=initial_env,
    )

    assert result.returncode == 0, result.stderr
    assert deployed_env == initial_env
    assert "Taostats key unavailable" in result.stderr
