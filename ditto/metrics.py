"""Prometheus counters shared by the API server and its query layer.

``GET /metrics`` (:mod:`ditto.api_server.endpoints.metrics`) exposes the default
registry, so a call site only has to import a counter and increment it.

Counters live here, above both packages, when the same event can be detected in
either an endpoint or in :mod:`ditto.db.queries` and must land on one series —
``ditto.db`` must not import ``ditto.api_server``.
"""

from __future__ import annotations

from typing import Literal

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

DispatchDeclineReason = Literal[
    "not_accepting",
    "slot_not_healthy",
    "slot_ceiling",
    "disk_breaker",
    "slot_cap",
    "slot_occupied",
    "no_candidate",
]
"""Why a fully authenticated ``POST /validator/job`` poll left with no ticket.

Deliberately closed and low-cardinality -- one value per *gate*, never per
validator or slot (those go to the log line instead). The split that matters
operationally is the first six (dispatch refused to issue: an admission,
capacity, or policy decision the operator controls) against ``no_candidate``
(dispatch was willing but the candidate walk found no eligible row). An idle
fleet is one or the other, and telling them apart used to require
reconstructing the queue predicates as raw SQL against production.
"""

# Fires on every 204 from the validator job-dispatch path, labelled with the
# gate that turned the poll away. Observability only: nothing here participates
# in the dispatch decision, and a decline is normal traffic (k=3 means most
# polls get nothing), so alert on the *mix* shifting, never on the raw rate.
VALIDATOR_DISPATCH_DECLINED = Counter(
    "ditto_validator_dispatch_declined_total",
    "Validator job polls that were answered 204, by the gate that declined them.",
    ("reason",),
)
