"""Prometheus counters shared by the API server and its query layer.

``GET /metrics`` (:mod:`ditto.api_server.endpoints.metrics`) exposes the default
registry, so a call site only has to import a counter and increment it.

Counters live here, above both packages, when the same event can be detected in
either an endpoint or in :mod:`ditto.db.queries` and must land on one series —
``ditto.db`` must not import ``ditto.api_server``.
"""

from __future__ import annotations

from prometheus_client import Counter

# Fires whenever a *signed, authenticated* heartbeat kept its liveness columns
# (``seen_at`` / ``reported_at``) but had its work payload dropped because the
# payload could not be validated. A non-zero rate means the fleet is live but
# some validator's reported capacity/progress is not being believed — the
# opposite of a stale heartbeat, and it must be alerted on separately. ``stage``
# names the guard that degraded (see the call sites); ``reason`` is the
# exception class name, deliberately low-cardinality.
VALIDATOR_HEARTBEAT_PAYLOAD_DEGRADED = Counter(
    "ditto_validator_heartbeat_payload_degraded_total",
    "Signed heartbeats stored liveness-only after the work payload failed validation.",
    ("stage", "reason"),
)

# Fires whenever hosted-inference admission answers ``AT_CAPACITY`` — the
# retryable decline the endpoint turns into ``503`` + ``Retry-After``.
#
# This counter exists because the platform had no way to answer the one question
# an operator actually asks before turning a concurrency knob: *is this limit
# ever the binding constraint?* Every admission limit shares a single anonymous
# exit, so the only way to find out was to reconstruct in-flight intervals from
# ``inference_requests`` after the fact. That reconstruction is what showed the
# embedding ceiling had never bound at all — peak fleet-wide in-flight of 14
# against a ceiling of 128 — which is a fact that should have been a scrape away.
#
# ``lane`` is ``chat`` or ``embedding``. ``scope`` names which of the five
# admission gates tripped, all of which are otherwise indistinguishable to the
# caller: ``token_reservation`` (in-flight reservations, not spend, overflow the
# allowance), ``per_ticket`` / ``per_validator`` / ``global`` (concurrency), and
# ``per_ticket_rpm`` / ``per_validator_rpm`` / ``global_rpm`` (rate). Both labels
# are closed sets, so cardinality is bounded at 2 x 7.
#
# A zero rate on ``embedding``/``per_ticket`` is the positive signal that the
# operator-tunable emergency brake is open. A non-zero rate on it during a
# deliberate brake application is the confirmation that the brake engaged.
INFERENCE_ADMISSION_AT_CAPACITY = Counter(
    "ditto_inference_admission_at_capacity_total",
    "Hosted-inference admissions declined as retryable backpressure, by lane and gate.",
    ("lane", "scope"),
)
