"""Operator-tunable concurrency for the hosted v7 embedding lane.

Scope is deliberately narrow: this board governs the **hosted** embedding route
(``perplexity/pplx-embed-v1-0.6b`` through the platform proxy) and nothing else.

* The **chat** lane keeps its boot-time config. It was never sized against a
  local resource, its limits are already 8/24/72, and it is not what is
  throttling a v7 run.
* The **local Ollama** lane (bench_version 2-6, one container per validator
  host) is not reachable from here at all. dittobench-api #93 made v7 bypass it
  (``inference_broker.go``: ``if benchVersion < 7 { acquire b.embeddingSlots }``),
  and ``DITTOBENCH_MAX_CONCURRENT_MEMORY_PHASES`` still pins it at one. Nothing
  in this module can widen it.

Why the shipped defaults are much larger than the values they replace:

The old numbers (1 / 8 / 32) were sized when v7 embeddings still ran against
that local Ollama container -- a scarce, host-local, single-tenant resource. The
embedding route is now a hosted, network-bound provider call, so per-ticket
serialisation is protecting nothing. Embeddings are roughly 63% of a v7 run's
~1,067 inference requests (671 of them), and at ``per_ticket = 1`` every one of
them is strictly serial.

The defaults below are chosen so the **validator** is the binding limit, not the
platform: dittobench-api admits 8 concurrent embeddings per run, so a per-ticket
ceiling of 12 is pure headroom that is never reached in normal operation. That
ordering is intentional. A limit that binds at the platform costs a network
round trip to discover, while the same limit enforced in the broker is a local
semaphore -- and, until the fleet carries a capacity-aware build, a platform
decline is the more expensive way to find out.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The shipped values. Every one of these is a raise -- there is no configuration
# of this board that reproduces the old serialised behaviour by default, because
# a knob whose default is the old value is a knob nobody turns.
DEFAULT_EMBEDDING_PER_TICKET_CONCURRENCY = 12
DEFAULT_EMBEDDING_PER_VALIDATOR_CONCURRENCY = 48
DEFAULT_EMBEDDING_GLOBAL_CONCURRENCY = 96

# Ceilings, not recommendations. 128 is the same bound `check_config` already
# enforces on the boot-time embedding limits, kept deliberately identical so the
# board can never accept a value the boot check would have rejected. It exists
# because a fat-fingered revision must not point the whole fleet at a number
# that saturates the single platform process's admission path (see
# `ditto/db/queries/inference.py`: every reservation, chat and embedding alike,
# serialises on ONE `pg_advisory_xact_lock("inference")`).
MAX_EMBEDDING_PER_TICKET_CONCURRENCY = 128
MAX_EMBEDDING_PER_VALIDATOR_CONCURRENCY = 128
MAX_EMBEDDING_GLOBAL_CONCURRENCY = 128


class InferenceConcurrencySettings(BaseModel):
    """The whole hosted-embedding concurrency policy, stored as one object.

    The three limits are a strict hierarchy: one ticket may not exceed its
    validator's allowance, and no validator may exceed the fleet's. The
    validator enforces the same hierarchy from below, so under normal operation
    the platform numbers are headroom rather than the operative valve.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    embedding_per_ticket_concurrency: Annotated[
        int, Field(ge=1, le=MAX_EMBEDDING_PER_TICKET_CONCURRENCY)
    ] = DEFAULT_EMBEDDING_PER_TICKET_CONCURRENCY
    """Concurrent hosted embedding requests one scoring ticket's grant may hold.

    This is the number the old ``1`` lived at, and the one an operator will
    actually turn. Lowering it is the emergency brake: it takes effect on the
    next admission fleet-wide, with no release and no restart.
    """

    embedding_per_validator_concurrency: Annotated[
        int, Field(ge=1, le=MAX_EMBEDDING_PER_VALIDATOR_CONCURRENCY)
    ] = DEFAULT_EMBEDDING_PER_VALIDATOR_CONCURRENCY
    """Concurrent hosted embedding requests summed over one validator's grants.

    Sized as four slots at the per-ticket allowance, which is the wire contract's
    practical ceiling on concurrent benchmark slots per host.
    """

    embedding_global_concurrency: Annotated[
        int, Field(ge=1, le=MAX_EMBEDDING_GLOBAL_CONCURRENCY)
    ] = DEFAULT_EMBEDDING_GLOBAL_CONCURRENCY
    """Concurrent hosted embedding requests across the whole fleet.

    The one number to move cautiously. Every admission takes a **global**
    Postgres advisory lock, so this bounds sustained pressure on a serialised
    path that also admits chat requests.
    """

    @model_validator(mode="after")
    def _hierarchy_holds(self) -> InferenceConcurrencySettings:
        if (
            self.embedding_per_ticket_concurrency
            > self.embedding_per_validator_concurrency
        ):
            raise ValueError(
                "embedding_per_ticket_concurrency "
                f"({self.embedding_per_ticket_concurrency}) may not exceed "
                "embedding_per_validator_concurrency "
                f"({self.embedding_per_validator_concurrency}): a ticket cannot "
                "be allowed more concurrency than the validator hosting it"
            )
        if self.embedding_per_validator_concurrency > self.embedding_global_concurrency:
            raise ValueError(
                "embedding_per_validator_concurrency "
                f"({self.embedding_per_validator_concurrency}) may not exceed "
                f"embedding_global_concurrency ({self.embedding_global_concurrency}): "
                "a single validator cannot be allowed more concurrency than the fleet"
            )
        return self


class InferenceConcurrencySettingsRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int
    parent_revision: int
    scope: str
    settings: InferenceConcurrencySettings
    reason: str
    actor: str
    created_at: datetime
    checksum: str


class EffectiveInferenceConcurrencySettings(BaseModel):
    """What the admission path is enforcing right now, and where it came from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int
    scope: str
    settings: InferenceConcurrencySettings
    checksum: str
    source: str
    """``"revision"`` when an operator revision governs, ``"default"`` otherwise."""


class AdminInferenceConcurrencySettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scope: str = "*"
    expected_revision: Annotated[int, Field(ge=0)]
    settings: InferenceConcurrencySettings
    reason: Annotated[str, Field(min_length=8, max_length=500)]
    actor: Annotated[str, Field(min_length=1, max_length=120)] = "admin_api"
    confirmation: str

    @model_validator(mode="after")
    def _require_complete_policy(self) -> AdminInferenceConcurrencySettingsRequest:
        """Reject a partial policy on a whole-object store.

        Every field has a default, which is what makes an empty board behave
        like the shipped configuration. On a *write* that is a footgun: a
        revision stores the whole policy, so sending only
        ``{"embedding_global_concurrency": 512}`` would silently reset the two
        limits below it to their defaults while the operator believed they
        changed one number. ``expected_revision`` cannot catch that -- they hold
        the current revision, they just under-specified the body.
        """
        missing = sorted(
            set(InferenceConcurrencySettings.model_fields)
            - self.settings.model_fields_set
        )
        if missing:
            raise ValueError(
                "an inference concurrency revision stores the WHOLE policy, so "
                f"every field must be sent explicitly; missing {missing}. Read "
                "GET /admin/inference-concurrency-settings, change the fields "
                "you want, and send back the complete object."
            )
        return self


class AdminInferenceConcurrencySettingsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    current: list[InferenceConcurrencySettingsRevision]
    history: list[InferenceConcurrencySettingsRevision]
    default: InferenceConcurrencySettings
    effective: EffectiveInferenceConcurrencySettings
