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
    initial_deploy_env: str | None = None,
    deploy_env_vars: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], str, str, int]:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    fake_bin.mkdir()
    shutil.copy2(ROOT / "scripts" / "update.sh", scripts / "update.sh")
    (repo / ".env").write_text(initial_env)
    if initial_deploy_env is not None:
        (repo / ".env.deploy").write_text(initial_deploy_env)

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
    env.update(deploy_env_vars or {})
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run(
        [str(scripts / "update.sh")],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    deploy_env = repo / ".env.deploy"
    return (
        result,
        (repo / ".env").read_text(),
        deploy_env.read_text(),
        deploy_env.stat().st_mode & 0o777,
    )


def test_update_loads_taostats_key_without_logging_value(tmp_path: Path) -> None:
    api_key = "tao-test:example"
    result, base_env, deploy_env, deploy_mode = _run_update(
        tmp_path,
        gcloud_source=f'printf "%s\\n" "{api_key}"\n',
    )

    assert result.returncode == 0, result.stderr
    assert base_env == "BASE_SETTING=kept\n"
    assert deploy_mode == 0o600
    assert f"DITTO_TAOSTATS_API_KEY={api_key}" in deploy_env
    assert (
        "DITTO_TAOSTATS_VALIDATOR_NAMES_URL="
        "https://api.taostats.io/api/dtao/validator/available/v1?netuid=118"
    ) in deploy_env
    assert api_key not in result.stdout
    assert api_key not in result.stderr


def test_update_keeps_existing_enrichment_when_secret_is_unavailable(
    tmp_path: Path,
) -> None:
    initial_deploy_env = (
        "DITTO_TAOSTATS_API_KEY=existing-key\n"
        "DITTO_TAOSTATS_VALIDATOR_NAMES_URL=https://example.invalid/names\n"
    )
    result, base_env, deploy_env, deploy_mode = _run_update(
        tmp_path,
        gcloud_source="exit 1\n",
        initial_deploy_env=initial_deploy_env,
    )

    assert result.returncode == 0, result.stderr
    assert base_env == "BASE_SETTING=kept\n"
    assert deploy_env == initial_deploy_env
    assert deploy_mode == 0o600
    assert "Taostats key unavailable" in result.stderr


def test_update_migrates_legacy_deploy_values_before_ansible_rewrites_base(
    tmp_path: Path,
) -> None:
    legacy_key = "legacy-key-must-not-be-logged"
    initial_env = (
        "BASE_SETTING=kept\n"
        f"DITTO_TAOSTATS_API_KEY={legacy_key}\n"
        "DITTO_TAOSTATS_VALIDATOR_NAMES_URL=https://example.invalid/names\n"
    )
    result, base_env, deploy_env, deploy_mode = _run_update(
        tmp_path,
        gcloud_source="exit 1\n",
        initial_env=initial_env,
    )

    assert result.returncode == 0, result.stderr
    assert base_env == initial_env
    assert f"DITTO_TAOSTATS_API_KEY={legacy_key}" in deploy_env
    assert (
        "DITTO_TAOSTATS_VALIDATOR_NAMES_URL=https://example.invalid/names" in deploy_env
    )
    assert deploy_mode == 0o600
    assert legacy_key not in result.stdout
    assert legacy_key not in result.stderr


def test_update_keeps_ansible_env_immutable_and_deploy_values_override(
    tmp_path: Path,
) -> None:
    payment = "5G6fGXnXFYdLM3ZyAm9whUbCY4ziQzcbMiTEqZB5c9KekTtR"
    base_env = (
        "BASE_SETTING=kept\nDITTO_UPLOAD_PAYMENT_ADDRESS=base-must-not-be-edited\n"
    )
    result, observed_base, deploy_env, deploy_mode = _run_update(
        tmp_path,
        gcloud_source="exit 1\n",
        initial_env=base_env,
        deploy_env_vars={"DITTO_UPLOAD_PAYMENT_ADDRESS": payment},
    )

    assert result.returncode == 0, result.stderr
    assert observed_base == base_env
    assert f"DITTO_UPLOAD_PAYMENT_ADDRESS={payment}" in deploy_env
    assert deploy_mode == 0o600


def test_update_repairs_no_final_newline_before_adding_another_key(
    tmp_path: Path,
) -> None:
    payment = "5G6fGXnXFYdLM3ZyAm9whUbCY4ziQzcbMiTEqZB5c9KekTtR"
    wandb_url = "https://wandb.ai/ditto/dev"
    result, _, deploy_env, deploy_mode = _run_update(
        tmp_path,
        gcloud_source="exit 1\n",
        initial_deploy_env=f"DITTO_UPLOAD_PAYMENT_ADDRESS={payment}",
        deploy_env_vars={"DITTO_DASHBOARD_WANDB_URL": wandb_url},
    )

    assert result.returncode == 0, result.stderr
    assert deploy_env.splitlines() == [
        f"DITTO_UPLOAD_PAYMENT_ADDRESS={payment}",
        f"DITTO_DASHBOARD_WANDB_URL={wandb_url}",
    ]
    assert deploy_mode == 0o600


def test_update_discards_truncated_fragment_and_retries_canonically(
    tmp_path: Path,
) -> None:
    payment = "5G6fGXnXFYdLM3ZyAm9whUbCY4ziQzcbMiTEqZB5c9KekTtR"
    result, _, deploy_env, deploy_mode = _run_update(
        tmp_path,
        gcloud_source="exit 1\n",
        initial_deploy_env="DITTO_UPLOAD_PAYMENT_ADD",
        deploy_env_vars={"DITTO_UPLOAD_PAYMENT_ADDRESS": payment},
    )

    assert result.returncode == 0, result.stderr
    assert deploy_env == f"DITTO_UPLOAD_PAYMENT_ADDRESS={payment}\n"
    assert deploy_mode == 0o600
