"""add agents.content_fingerprint for content-level anti-copy detection

Revision ID: c4e8b1a06d72
Revises: a3f1c9d27b40
Create Date: 2026-07-05 12:00:00.000000

Strengthens the anti-copy moderation gate (:mod:`ditto.api_server.scoring_gate`).
The size+score-proximity signal is byte-level: a copier who re-indents or renames
files moves the tarball size past its tolerance and dodges it. This adds a
*content*-level fingerprint — the set of normalized per-file content hashes
computed from the tarball at upload (:mod:`ditto.api_server.fingerprint`) — so a
reindented/renamed copy of another miner's agent still scores a high Jaccard
overlap and is held for review.

- ``agents.content_fingerprint`` — JSONB array of hex hashes, one per regular
  file (normalized to ignore indentation/whitespace + filename). Nullable:
  backfill-null for rows written before this migration and for tarballs that are
  unreadable/empty (the gate treats a null fingerprint as "no content match", so
  such rows fall back to the sha256 + size signals).

No enum change and no index: the fingerprint is only read for the small
best-eligible-per-miner ledger the gate scans, never filtered on in SQL.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4e8b1a06d72"
down_revision: str | Sequence[str] | None = "a3f1c9d27b40"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable ``agents.content_fingerprint`` JSONB column."""
    op.execute("ALTER TABLE agents ADD COLUMN content_fingerprint JSONB")


def downgrade() -> None:
    """Drop ``agents.content_fingerprint``."""
    op.execute("ALTER TABLE agents DROP COLUMN IF EXISTS content_fingerprint")
