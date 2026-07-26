"""Operator-tunable admission policy for the hosted v7 inference lanes.

This board governs the **hosted** embedding route
(``perplexity/pplx-embed-v1-0.6b`` through the platform proxy) plus one chat
number: the per-grant chat request budget.

* The chat lane's **concurrency and rate** limits keep their boot-time config.
  They were never sized against a local resource, they are already 8/24/72, and
  they are not what is throttling a v7 run.
* The chat lane's **request budget** is here because it is a per-lease resource
  allowance, not a rate. See ``chat_request_budget`` for why it moved.
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
# that saturates the admission path (see `ditto/db/queries/inference.py`: a
# reservation no longer serialises fleet-wide, but it still takes a grant-row
# `FOR UPDATE` plus the cross-grant admission aggregates on every call).
MAX_EMBEDDING_PER_TICKET_CONCURRENCY = 128
MAX_EMBEDDING_PER_VALIDATOR_CONCURRENCY = 128
MAX_EMBEDDING_GLOBAL_CONCURRENCY = 128

# The chat request budget, sized against the observed distribution rather than
# against the round number it replaces (1024, which was never justified against
# a real run).
#
#   * a typical v7 agent spends ~1.25 chat requests per check, ~355 per run
#   * Jupiter and KOTH_v7_1 spend ~3.85 per check, ~1090 per run
#   * a run is 279-283 checks depending on the dataset
#
# 1024 sat *below* the heaviest observed strategy, so those agents exhausted
# around check 266 and had every remaining call refused -- a resource limit
# behaving as a run failure. 8192 is ~23x the median run, ~7.5x the heaviest
# run observed, and ~29 requests per check at 283 checks. It is a real raise
# (8x) rather than a nudge, which is the point: a ceiling that a legitimate
# strategy can reach by being thorough is a ceiling in the wrong place.
DEFAULT_CHAT_REQUEST_BUDGET = 8192

# The ceiling is 2x the default, not unbounded. The request budget is not the
# only thing bounding a grant's spend -- ``token_budget`` (4,000,000 per grant,
# boot-time and unchanged) is the other, and at 8192 requests it is the one that
# binds first for any strategy averaging over ~488 tokens a call. Keeping a
# finite request ceiling matters anyway: it is the bound that survives a
# pathological loop of tiny requests, which the token budget would absorb
# slowly and the concurrency board would not catch at all.
MAX_CHAT_REQUEST_BUDGET = 16384


class InferenceConcurrencySettings(BaseModel):
    """The whole hosted-inference admission policy, stored as one object.

    The three embedding limits are a strict hierarchy: one ticket may not exceed
    its validator's allowance, and no validator may exceed the fleet's. The
    validator enforces the same hierarchy from below, so under normal operation
    the platform numbers are headroom rather than the operative valve.

    ``chat_request_budget`` stands apart from the three. It is not a rate and it
    is not enforced fleet-wide at admission -- it is *stamped onto each grant when
    the grant is minted* and thereafter read from the grant's own row. That is
    deliberate, and it is what makes this field safe to sit on a live board: a
    revision changes what the **next** lease is issued, never what a running
    lease is already spending against. An operator cannot exhaust a run in
    flight by lowering this number, which is exactly the hazard that forced the
    embedding limits to grow a capacity-decline path.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    chat_request_budget: Annotated[int, Field(ge=1, le=MAX_CHAT_REQUEST_BUDGET)] = (
        DEFAULT_CHAT_REQUEST_BUDGET
    )
    """Chat completions one scoring ticket's grant may spend, in total.

    Raising this is the lever for "let a heavier strategy finish"; lowering it is
    the lever for "stop paying for a runaway". Neither takes effect on a lease
    that has already been minted -- see the class docstring.
    """

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

    The one number to move cautiously. It is enforced by a **cross-grant**
    aggregate over every in-flight request, so unlike the per-ticket limit it
    is best-effort under a simultaneous burst: concurrent admissions can
    overshoot it by at most the number of racers. Size it as a load-shedding
    backstop with headroom, not as an exact valve.
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
