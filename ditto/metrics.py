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
