from pathlib import Path

ROOT = Path(__file__).parents[3]


def test_iap_jobs_enable_numpy_before_opening_a_tunnel() -> None:
    workflow = (ROOT / ".github" / "workflows" / "deploy.yml").read_text()
    deploy_job, relay_job = workflow.split("\n  relay-release:\n", 1)
    deploy_job = deploy_job.split("\n  deploy:\n", 1)[1]

    for job in (deploy_job, relay_job):
        setup = job.index("google-github-actions/setup-gcloud@")
        python = job.index("gcloud info --format='value(basic.python_location)'")
        install = job.index('"$gcloud_python" -m pip install')
        verify = job.index('import numpy; print(f"NumPy {numpy.__version__}')
        transport = job.index("--tunnel-through-iap")
        assert setup < python < install < verify < transport
