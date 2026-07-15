"""The platform must consume the standalone screening protocol package."""

from uuid import UUID

from ditto.api_models.agent_status import AgentStatus as CompatibilityAgentStatus
from ditto.api_models.screener import ScreenResultRequest as CompatibilityRequest
from ditto_screening_protocol import (
    SCREENING_POLICY_VERSION,
    AgentStatus,
    ScreenResultRequest,
    verdict_signing_message,
)


def test_compatibility_imports_are_shared_package_types() -> None:
    assert CompatibilityAgentStatus is AgentStatus
    # The request is temporarily a platform-local extension of the shared type
    # (quarantine review payloads shipped in protocol 0.9.0, while the pin is
    # still 0.8.0). It must remain a subtype so the signing and validation
    # semantics stay those of the shared package; restore the identity
    # assertion when the pin reaches >= 0.9.0.
    assert issubclass(CompatibilityRequest, ScreenResultRequest)


def test_canonical_versioned_verdict_message_is_wire_compatible() -> None:
    hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    agent_id = UUID("550e8400-e29b-41d4-a716-446655440000")
    assert (
        verdict_signing_message(
            screener_hotkey=hotkey,
            agent_id=agent_id,
            passed=True,
            policy_version=SCREENING_POLICY_VERSION,
        )
        == f"{hotkey}:{agent_id}:True:{SCREENING_POLICY_VERSION}".encode()
    )


def test_canonical_lease_bound_verdict_message_is_exact() -> None:
    hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    agent_id = UUID("550e8400-e29b-41d4-a716-446655440000")
    attempt_id = UUID("550e8400-e29b-41d4-a716-446655440001")
    assert (
        verdict_signing_message(
            screener_hotkey=hotkey,
            agent_id=agent_id,
            attempt_id=attempt_id,
            passed=False,
            policy_version=SCREENING_POLICY_VERSION,
        )
        == (
            "ditto-screen-verdict:v2:"
            f"{hotkey}:{agent_id}:{attempt_id}:False:{SCREENING_POLICY_VERSION}"
        ).encode()
    )
