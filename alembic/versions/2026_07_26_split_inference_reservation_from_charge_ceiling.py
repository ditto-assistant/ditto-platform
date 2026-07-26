"""split the inference reservation from its charge ceiling, and version the meter

``inference_requests.reserved_tokens`` was doing two incompatible jobs. It was
the amount held against a grant's token allowance while a request was in
flight, AND the ceiling that untrusted provider accounting was clamped to. Both
worked only because the value was ``max_tokens + len(body)`` -- byte length used
directly as a token count, a genuine upper bound but roughly 4x the truth.

Making the reservation an honest estimate breaks the second job: a legitimate
token-dense prompt lands a little above the estimate, and
``finish_inference_request`` marks anything over its bound non-deliverable,
which would 409 ordinary successful calls back to the harness. So the ceiling
becomes its own column and keeps the byte-derived definition.

``max_chargeable_tokens`` backfills from ``reserved_tokens``, which is exactly
right for historical rows: under the old contract that value *was* this bound.

``inference_grants.usage_accounting_version`` records which meter booked a
grant's counters, on the same principle as ``bench_version`` for scores -- a
token total is only comparable within the contract that produced it. Existing
rows are version 1 (server default); new grants are stamped 2 by the
application. There is deliberately no backfill of the counters themselves: the
over-charge happened at reservation time, so what those calls actually consumed
was never recorded and cannot be recovered.

Additive and non-destructive. No existing counter is rewritten.

Revision ID: b7e41c93a05d
Revises: f3b8c2d17a49
Create Date: 2026-07-26
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b7e41c93a05d"
down_revision: str | Sequence[str] | None = "f3b8c2d17a49"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Added nullable, backfilled, then pinned NOT NULL so the deploy does not
    # need a table rewrite with a non-trivial default under load.
    op.add_column(
        "inference_requests",
        sa.Column("max_chargeable_tokens", sa.BigInteger(), nullable=True),
    )
    op.execute(
        "UPDATE inference_requests "
        "SET max_chargeable_tokens = reserved_tokens "
        "WHERE max_chargeable_tokens IS NULL"
    )
    op.alter_column(
        "inference_requests",
        "max_chargeable_tokens",
        existing_type=sa.BigInteger(),
        nullable=False,
        server_default=sa.text("0"),
    )
    op.add_column(
        "inference_grants",
        sa.Column(
            "usage_accounting_version",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
    )


def downgrade() -> None:
    op.drop_column("inference_grants", "usage_accounting_version")
    op.drop_column("inference_requests", "max_chargeable_tokens")
