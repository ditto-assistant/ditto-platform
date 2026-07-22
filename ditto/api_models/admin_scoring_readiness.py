"""Operator observability: why an agent can or cannot be leased for scoring.

The DB lives on a GCE VM operators cannot easily reach, so this exposes the
exact ticket-issuance prerequisites (`issue_ticket`) as a read model — dataset,
screened image, screening policy — to explain a submission stuck below quorum
without a live validator ever picking it up.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel


class ScreenedImageReadiness(BaseModel):
    complete: bool
    """All identity fields plus the verification timestamp are present."""
    verified: bool
    policy_ok: bool
    """Built under a screening policy at or above the active contract's minimum."""
    missing_fields: list[str]


class AgentScoringReadiness(BaseModel):
    agent_id: UUID
    agent_name: str
    miner_hotkey: str
    status: str
    active_bench_version: int
    screening_policy_version: int
    required_screening_policy_version: int
    requires_screened_image: bool
    has_versioned_dataset: bool
    screened_image: ScreenedImageReadiness
    leaseable: bool
    """True iff a validator's next sweep could lease this agent for the active
    version — i.e. ``blocking_reasons`` is empty."""
    blocking_reasons: list[str]
