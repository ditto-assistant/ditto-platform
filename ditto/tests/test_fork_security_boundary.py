"""One-shot public-fork security-boundary verification.

This file belongs only to the disposable fork-test pull request. It is not
intended to merge into the upstream repository.
"""

import os


def test_fork_pr_has_no_secrets_oidc_or_deploy_credentials() -> None:
    forbidden = {
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
        "GCP_WIF_PROVIDER",
        "GCP_PLATFORM_DEPLOY_SA",
        "DITTO_UPLOAD_PAYMENT_ADDRESS",
    }

    exposed = sorted(forbidden.intersection(os.environ))
    assert exposed == [], f"privileged fork context exposed: {exposed}"
