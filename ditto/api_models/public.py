"""Public, unauthenticated read models for the subnet dashboard.

These expose the **aggregate** shape only — composite plus tool/memory means and
rank — and deliberately omit the fields on :class:`LedgerEntry` that are either
integrity-internal (``sha256``, ``signature``, ``validator_hotkey``) or would
hand a miner the benchmark's answer key (per-case ``expected``/``called``). See
``docs/public-telemetry.md`` for the transparency policy this encodes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

_SS58_PATTERN = r"^[1-9A-HJ-NP-Za-km-z]{47,48}$"


class PublicLeaderboardEntry(BaseModel):
    """One miner's best score, aggregate-only, for public display."""

    rank: Annotated[int, Field(ge=1, description="1-based rank by composite.")]
    miner_hotkey: Annotated[
        str, Field(pattern=_SS58_PATTERN, description="Miner's SS58 hotkey.")
    ]
    composite: Annotated[
        float, Field(ge=0.0, le=1.0, description="Best composite in [0,1].")
    ]
    tool_mean: Annotated[
        float, Field(ge=0.0, le=1.0, description="Mean tool accuracy in [0,1].")
    ]
    memory_mean: Annotated[
        float, Field(ge=0.0, le=1.0, description="Mean memory recall in [0,1].")
    ]
    first_seen: Annotated[
        datetime, Field(description="When the winning agent was first uploaded (UTC).")
    ]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rank": 1,
                "miner_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
                "composite": 0.587,
                "tool_mean": 0.867,
                "memory_mean": 0.167,
                "first_seen": "2026-07-03T20:00:00Z",
            }
        }
    )


class PublicLeaderboardResponse(BaseModel):
    """The public best-score-per-miner leaderboard, highest composite first."""

    generated_at: Annotated[
        datetime, Field(description="When this snapshot was read (UTC).")
    ]
    count: Annotated[int, Field(ge=0, description="Number of entries.")]
    entries: Annotated[
        list[PublicLeaderboardEntry],
        Field(default_factory=list, description="Ranked miners, best composite first."),
    ]


class PublicHealthResponse(BaseModel):
    """Aggregate subnet-health rollup for the public dashboard.

    Derived only from what the platform records (submissions + reported scores).
    Run started/failed counts, set-weights latency and per-stage timings are
    validator-side telemetry (wandb), not served here — the platform only ever
    sees a *successful* score, so it deliberately reports no "success rate".
    """

    generated_at: Annotated[
        datetime, Field(description="When this snapshot was read (UTC).")
    ]
    miners: Annotated[
        int, Field(ge=0, description="Distinct miners who have ever submitted.")
    ]
    scored_miners: Annotated[
        int, Field(ge=0, description="Distinct miners on the leaderboard (scored).")
    ]
    scored_agents: Annotated[
        int, Field(ge=0, description="Agents currently eligible (scored).")
    ]
    last_scored_at: Annotated[
        datetime | None,
        Field(default=None, description="When a validator last scored anything (UTC)."),
    ]
    scores_24h: Annotated[
        int, Field(ge=0, description="Scores generated in the last 24h.")
    ]
    avg_latency_ms: Annotated[
        int | None,
        Field(
            default=None, ge=0, description="Mean per-score median case latency (ms)."
        ),
    ]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "generated_at": "2026-07-04T12:00:00Z",
                "miners": 12,
                "scored_miners": 5,
                "scored_agents": 7,
                "last_scored_at": "2026-07-04T11:52:00Z",
                "scores_24h": 9,
                "avg_latency_ms": 812,
            }
        }
    )
