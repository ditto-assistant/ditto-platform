#!/usr/bin/env bash
#
# Start the Ditto Platform API on a host:
#   1. load .env
#   2. sync dependencies
#   3. bring up Docker infra (postgres + minio + pylon) and wait until healthy
#   4. apply database migrations
#   5. start the API under pm2 (logs -> ./logs, autorestart on)
#
# Idempotent: safe to re-run. Use scripts/update.sh for zero-downtime deploys.

set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "ERROR: .env not found. Run: cp .env.example .env  (then fill it in)" >&2
  exit 1
fi

command -v uv  >/dev/null 2>&1 || { echo "ERROR: uv not installed"  >&2; exit 1; }
command -v pm2 >/dev/null 2>&1 || { echo "ERROR: pm2 not installed (npm i -g pm2)" >&2; exit 1; }

# Export .env into the environment pm2 will inherit.
set -a; . ./.env; set +a

echo "==> syncing dependencies"
uv sync

echo "==> bringing up infra (postgres + minio + pylon)"
docker compose up -d --wait postgres minio pylon
docker compose up -d minio-create-bucket

echo "==> applying migrations"
uv run alembic upgrade head

mkdir -p logs

echo "==> starting API under pm2"
pm2 start scripts/ecosystem.config.js --update-env
pm2 save

echo ""
echo "API up on http://localhost:${API_PORT:-8000}  (docs: /docs)"
echo "  pm2 logs ditto-api   # tail logs"
echo "  pm2 status           # process state"
