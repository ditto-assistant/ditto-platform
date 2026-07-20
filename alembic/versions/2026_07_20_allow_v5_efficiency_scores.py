"""allow bounded benchmark-v5 efficiency-adjusted composites

Revision ID: d9a4e7c21b60
Revises: b7f2c8d41a95
Create Date: 2026-07-20 16:45:00.000000

Raw per-case and suite scores remain in [0, 1]. Only the aggregate composite
may exceed 1, and only for bench_version >= 5, with the v5 formula's hard 1.25
ceiling. Historical v1-v4 rows retain their original database constraint.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "d9a4e7c21b60"
down_revision: str | Sequence[str] | None = "b7f2c8d41a95"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_V5_RANGE = (
    "composite >= 0 AND ((bench_version < 5 AND composite <= 1) OR "
    "(bench_version >= 5 AND composite <= 1.25))"
)
_LEGACY_RANGE = "composite >= 0 AND composite <= 1"


def upgrade() -> None:
    with op.batch_alter_table("scores") as batch:
        batch.drop_constraint("scores_composite_range_check", type_="check")
        batch.create_check_constraint("scores_composite_range_check", _V5_RANGE)


def downgrade() -> None:
    # Downgrade intentionally fails if v5 rows above 1 exist rather than
    # silently clipping or deleting score history.
    with op.batch_alter_table("scores") as batch:
        batch.drop_constraint("scores_composite_range_check", type_="check")
        batch.create_check_constraint("scores_composite_range_check", _LEGACY_RANGE)
