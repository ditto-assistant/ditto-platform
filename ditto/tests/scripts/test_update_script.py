from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).parents[3]


def _write_executable(path: Path, source: str) -> None:
    path.write_text(f"#!/usr/bin/env bash\nset -eu\n{source}")
    path.chmod(0o755)


@contextmanager
def _health_server(status: int = 200) -> Iterator[int]:
    """Serve ``status`` on any path so update.sh's post-deploy probe can pass.

    The probe is deliberately the one thing update.sh will not fake: it requires
    a real HTTP answer on the API port. Tests therefore need a real listener.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def log_message(self, *_args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()


def _jlist(repo: Path, *, api_status: str = "online", script: str | None = None) -> str:
    """A ``pm2 jlist`` payload whose launch identity matches ecosystem.config.js."""
    exec_path = script or str(repo / ".venv" / "bin" / "python")
    common = {
        "pm_exec_path": exec_path,
        "exec_interpreter": "none",
        "exec_mode": "fork_mode",
        "pm_cwd": str(repo),
        "restart_time": 0,
    }
    return json.dumps(
        [
            {
                "name": "ditto-api",
                "pid": 4242,
                "pm2_env": {**common, "status": api_status},
            },
            {
                "name": "ditto-screened-image-cleanup",
                "pid": 0,
                "pm2_env": {**common, "status": "stopped"},
            },
        ]
    )


def _run_update(
    tmp_path: Path,
    *,
    gcloud_source: str,
    initial_env: str = "BASE_SETTING=kept\n",
    initial_deploy_env: str | None = None,
    deploy_env_vars: dict[str, str] | None = None,
    jlist: str | None = None,
    health_status: int = 200,
    health_timeout: str = "15",
) -> tuple[subprocess.CompletedProcess[str], str, str, int]:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    fake_bin.mkdir()
    shutil.copy2(ROOT / "scripts" / "update.sh", scripts / "update.sh")
    # The deploy plan and the app definition it diffs against are part of the
    # start/reload path, so the fake repo needs both.
    shutil.copy2(
        ROOT / "scripts" / "pm2_deploy_plan.js", scripts / "pm2_deploy_plan.js"
    )
    shutil.copy2(
        ROOT / "scripts" / "ecosystem.config.js", scripts / "ecosystem.config.js"
    )
    (repo / "logs").mkdir()
    (repo / ".env").write_text(initial_env)
    if initial_deploy_env is not None:
        (repo / ".env.deploy").write_text(initial_deploy_env)

    (repo / "jlist.json").write_text(jlist if jlist is not None else _jlist(repo))

    _write_executable(
        fake_bin / "git",
        'if [ "${1:-}" = "rev-parse" ]; then printf "main\\n"; fi\n',
    )
    _write_executable(fake_bin / "uv", ":\n")
    _write_executable(fake_bin / "docker", ":\n")
    _write_executable(
        fake_bin / "pm2",
        f'if [ "${{1:-}}" = "jlist" ]; then cat "{repo}/jlist.json"; fi\n'
        f'printf "%s\\n" "pm2 $*" >> "{repo}/pm2-actions.log"\n',
    )
    _write_executable(fake_bin / "gcloud", gcloud_source)
    _write_executable(fake_bin / "timeout", 'shift\nexec "$@"\n')

    env = os.environ.copy()
    env.update(deploy_env_vars or {})
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["DITTO_HEALTH_TIMEOUT"] = health_timeout

    with _health_server(health_status) as port:
        env["API_PORT"] = str(port)
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


def _actions(tmp_path: Path) -> str:
    return (tmp_path / "repo" / "pm2-actions.log").read_text()


def test_update_reloads_in_place_when_launch_identity_matches(tmp_path: Path) -> None:
    """The ordinary code-only deploy keeps using graceful reload."""
    result, _, _, _ = _run_update(tmp_path, gcloud_source="exit 1\n")

    assert result.returncode == 0, result.stderr
    actions = _actions(tmp_path)
    assert "pm2 reload scripts/ecosystem.config.js" in actions
    assert "pm2 delete" not in actions
    assert "ditto-api: reload" in result.stdout


def test_update_recreates_the_app_when_the_script_path_drifted(tmp_path: Path) -> None:
    """The outage case: pm2 reload silently keeps the old `script`.

    pm2 is running `uv` while ecosystem.config.js now resolves to the venv
    interpreter, so the deploy must delete and start rather than reload.
    """
    repo = tmp_path / "repo"
    result, _, _, _ = _run_update(
        tmp_path,
        gcloud_source="exit 1\n",
        jlist=_jlist(repo, script="/usr/local/bin/uv"),
    )

    assert result.returncode == 0, result.stderr
    actions = _actions(tmp_path)
    assert "pm2 delete ditto-api" in actions
    assert "pm2 start scripts/ecosystem.config.js" in actions
    assert "pm2 reload" not in actions
    assert "recreate (script:" in result.stdout


def test_update_fails_when_the_api_never_comes_up(tmp_path: Path) -> None:
    """A deploy that leaves the API dead must exit non-zero, not report success."""
    result, _, _, _ = _run_update(
        tmp_path,
        gcloud_source="exit 1\n",
        jlist=_jlist(tmp_path / "repo", api_status="waiting restart"),
        health_timeout="4",
    )

    assert result.returncode != 0
    assert "deploy failed" in result.stderr
    assert "ditto-api" in result.stderr


def test_update_fails_when_the_api_serves_a_degraded_health_response(
    tmp_path: Path,
) -> None:
    """Online but /health non-200 is still a failed deploy, reported distinctly."""
    result, _, _, _ = _run_update(
        tmp_path,
        gcloud_source="exit 1\n",
        health_status=503,
        health_timeout="4",
    )

    assert result.returncode != 0
    assert "returned HTTP 503" in result.stderr


def test_update_accepts_the_stopped_one_shot_cleanup_job(tmp_path: Path) -> None:
    """`stopped` is the cron-driven cleanup job's correct terminal state."""
    result, _, _, _ = _run_update(tmp_path, gcloud_source="exit 1\n")

    assert result.returncode == 0, result.stderr
    assert "ditto-screened-image-cleanup: stopped (one-shot" in result.stdout
