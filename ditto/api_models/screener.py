"""Wire shapes for the ``/screener/*`` endpoints.

The screener is the cheap pre-evaluation gate: it pulls freshly ``uploaded``
agents, does a lint + compile + build check on the tarball, and reports a
pass/fail. A pass promotes the agent into ``evaluating`` (where the validator
queue picks it up); a fail moves it to ``screening_failed`` and it never costs a
full DittoBench run.

Mirrors the ``/validator/*`` contract on purpose so the two workers look the
same to an operator:

1. ``GET  /screener/queue`` — list agents awaiting screening (``uploaded``).
2. ``GET  /screener/agent/{id}/artifact`` — fetch a download URL for the tarball
   so the screener can build it.
3. ``POST /screener/agent/{id}/result`` — report the pass/fail verdict.

The platform stays thin: it owns only the state machine + the queue. The build
check itself lives in the screener worker (``ditto-subnet``). The screener
authenticates exactly like the validator (a permitted hotkey; the result POST is
signed) — a distinct ``screener_permit`` is a future refinement.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.upload import (
    _SIGNATURE_HEX_PATTERN,
    _SS58_PATTERN,
)


class ScreenerQueueItem(BaseModel):
    """One agent awaiting screening, returned by ``GET /screener/queue``.

    Carries what the screener needs to fetch + identify the submission; the
    tarball itself comes from the ``/artifact`` endpoint.
    """

    agent_id: Annotated[UUID, Field(description="Server-generated agent identifier.")]
    miner_hotkey: Annotated[str, Field(description="Submitting miner's SS58 hotkey.")]
    name: Annotated[str, Field(description="Miner-chosen agent name.")]
    sha256: Annotated[
        str, Field(description="SHA-256 of the uploaded tarball, lowercase hex.")
    ]
    status: Annotated[
        AgentStatus, Field(description="Lifecycle state at queue read time.")
    ]
    created_at: Annotated[
        datetime, Field(description="When the upload row was inserted (UTC).")
    ]


class ScreenerQueueResponse(BaseModel):
    """Returned by ``GET /screener/queue``.

    ``items`` is ordered oldest-first so the screener drains submissions in
    arrival order. ``count`` echoes ``len(items)``.
    """

    items: Annotated[
        list[ScreenerQueueItem],
        Field(description="Agents awaiting screening, oldest first."),
    ]
    count: Annotated[int, Field(ge=0, description="Number of items returned.")]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {
                        "agent_id": "550e8400-e29b-41d4-a716-446655440000",
                        "miner_hotkey": (
                            "5DhaT8U7LVwnnJNUU8VL1XEipicatoaDVVq7cHo227gogVZm"
                        ),
                        "name": "alpha-agent",
                        "sha256": "deadbeef" * 8,
                        "status": "uploaded",
                        "created_at": "2026-06-08T12:00:00Z",
                    }
                ],
                "count": 1,
            }
        }
    )


class ScreenResultRequest(BaseModel):
    """Body of ``POST /screener/agent/{agent_id}/result``.

    The screener authenticates by signing the verdict: the signature is over
    the UTF-8 bytes of ``f"{screener_hotkey}:{agent_id}"`` with the screener's
    hotkey keypair. ``passed`` is the gate: ``True`` promotes the agent to
    ``evaluating``, ``False`` moves it to ``screening_failed``. ``detail`` is an
    optional human-readable reason (e.g. a build-log tail) for logs/audit.
    """

    screener_hotkey: Annotated[
        str,
        Field(pattern=_SS58_PATTERN, description="Reporting screener's SS58 hotkey."),
    ]
    signature: Annotated[
        str,
        Field(
            pattern=_SIGNATURE_HEX_PATTERN,
            description="Hex sr25519 signature over ``{screener_hotkey}:{agent_id}``.",
        ),
    ]
    passed: Annotated[
        bool,
        Field(description="True promotes to evaluating; False -> screening_failed."),
    ]
    detail: Annotated[
        str,
        Field(
            default="",
            max_length=4000,
            description="Optional reason / build-log tail (logged, not persisted).",
        ),
    ]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "screener_hotkey": ("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"),
                "signature": "ab" * 64,
                "passed": True,
                "detail": "",
            }
        }
    )


class ScreenResultResponse(BaseModel):
    """Returned by ``POST /screener/agent/{agent_id}/result``.

    ``status`` is the agent's lifecycle state *after* the verdict:
    ``evaluating`` on a pass, ``screening_failed`` on a fail. Re-reporting the
    same verdict is idempotent (the status is already the target). ``accepted``
    is ``True`` when the verdict was applied or was already in effect.
    """

    agent_id: Annotated[UUID, Field(description="Echoes the path-param id.")]
    status: Annotated[
        AgentStatus, Field(description="Lifecycle state after the verdict.")
    ]
    accepted: Annotated[
        bool, Field(description="``True`` when the verdict was applied.")
    ]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "agent_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "evaluating",
                "accepted": True,
            }
        }
    )
