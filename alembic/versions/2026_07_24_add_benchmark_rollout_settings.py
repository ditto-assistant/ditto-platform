"""add operator-configurable benchmark rollout rescore cohort size

Two changes, one head:

* ``benchmark_rollout_settings_revisions`` -- the append-only operator policy
  (currently just ``rescore_cohort_size``), mirroring the other hot-swappable
  settings tables.
* ``benchmark_rollouts.rescore_cohort_target`` -- the effective size frozen at
  rollout START, so a later policy revision can never resize an in-flight
  rollout and a historical rollout stays explainable.

The backfill keeps behavior byte-identical. Rollouts created before this change
targeted the hard-coded ten, so they backfill to ten; the handful of legacy
snapshots that were frozen at eleven-to-twenty-five members record their own
size instead, which is the target they were actually built to. Either way
``_rollout_rescore_cohort`` short-circuits on ``len(existing) >= target``
exactly as it did before.

Revision ID: c3f1a7d92e58
Revises: b7e6d5c4a3f2
Create Date: 2026-07-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "c3f1a7d92e58"
down_revision: str | Sequence[str] | None = "b7e6d5c4a3f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.create_table(
        "benchmark_rollout_settings_revisions",
        sa.Column("revision", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("parent_revision", sa.Integer(), nullable=False),
        sa.Column("scope", sa.Text(), nullable=False),
        sa.Column("settings", json_type, nullable=False),
        sa.Column("checksum", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "scope = '*'", name="benchmark_rollout_settings_scope_check"
        ),
        sa.CheckConstraint(
            "length(checksum) = 64", name="benchmark_rollout_settings_checksum_check"
        ),
        sa.CheckConstraint(
            "parent_revision >= 0",
            name="benchmark_rollout_settings_parent_revision_check",
        ),
        sa.CheckConstraint(
            "length(trim(reason)) BETWEEN 8 AND 500",
            name="benchmark_rollout_settings_reason_check",
        ),
        sa.CheckConstraint(
            "length(trim(actor)) BETWEEN 1 AND 120",
            name="benchmark_rollout_settings_actor_check",
        ),
        sa.PrimaryKeyConstraint("revision"),
        sa.UniqueConstraint(
            "scope",
            "parent_revision",
            name="benchmark_rollout_settings_scope_parent_key",
        ),
    )
    op.create_index(
        "benchmark_rollout_settings_scope_revision_idx",
        "benchmark_rollout_settings_revisions",
        ["scope", "revision"],
        unique=True,
    )
    op.add_column(
        "benchmark_rollouts",
        sa.Column(
            "rescore_cohort_target",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("10"),
        ),
    )
    # Legacy snapshots frozen above the default record their own size; plain
    # SQL so the statement is identical on postgres and the sqlite test engine.
    op.execute(
        "UPDATE benchmark_rollouts SET rescore_cohort_target = cohort_size "
        "WHERE cohort_size > 10"
    )
    op.create_check_constraint(
        "benchmark_rollout_bounded_rescore_target",
        "benchmark_rollouts",
        "rescore_cohort_target BETWEEN 5 AND 25",
    )


def downgrade() -> None:
    op.drop_constraint(
        "benchmark_rollout_bounded_rescore_target",
        "benchmark_rollouts",
        type_="check",
    )
    op.drop_column("benchmark_rollouts", "rescore_cohort_target")
    op.drop_index(
        "benchmark_rollout_settings_scope_revision_idx",
        table_name="benchmark_rollout_settings_revisions",
    )
    op.drop_table("benchmark_rollout_settings_revisions")
