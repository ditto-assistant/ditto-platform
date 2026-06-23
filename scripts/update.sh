#!/usr/bin/env bash
#
# Zero-downtime update for the Ditto Platform API:
#   git pull -> uv sync -> migrate -> pm2 reload.
# Run on the host after changes land on the deployed branch.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> pulling latest"
git pull --ff-only

echo "==> syncing dependencies"
uv sync

set -a; . ./.env; set +a

echo "==> applying migrations"
uv run alembic upgrade head

echo "==> reloading API (zero-downtime)"
pm2 reload scripts/ecosystem.config.js --update-env

echo "done. pm2 logs ditto-api"
