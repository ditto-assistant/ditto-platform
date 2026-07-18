"""merge screened images and validator retry recoveries

Revision ID: a6f3c8e91d42
Revises: e4a2b9c71d60, d2e5f7a9c1b3
Create Date: 2026-07-18 15:45:00.000000
"""

from collections.abc import Sequence

revision: str = "a6f3c8e91d42"
down_revision: str | Sequence[str] | None = (
    "e4a2b9c71d60",
    "d2e5f7a9c1b3",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
