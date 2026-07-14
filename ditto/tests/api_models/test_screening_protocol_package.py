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
    assert CompatibilityRequest is ScreenResultRequest


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
