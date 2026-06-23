#!/usr/bin/env bash
#
# Zero-downtime update for the Ditto Platform API:
#   fetch -> reset -> uv sync -> set payment addr -> ensure Pylon -> migrate -> pm2 start/reload.
# Invoked on the host by the ditto-platform deploy workflow (push dev|main ->
# IAP SSH). DITTO_DEPLOY_BRANCH defaults to the current branch; CI passes the
# branch that was pushed so the checkout is deterministic.

set -euo pipefail
cd "$(dirname "$0")/.."

branch="${DITTO_DEPLOY_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
echo "==> fetching + resetting to origin/$branch"
git fetch --prune origin
git checkout "$branch"
git reset --hard "origin/$branch"

echo "==> syncing dependencies"
uv sync

# Deploy-supplied upload payment address (the platform repo's GitHub environment
# secret, passed in by deploy.yml). Upsert it into .env BEFORE sourcing, so the
# app — which requires it at boot — comes up with the right value. The Ansible
# role deliberately leaves this key out of the rendered .env.
if [ -n "${DITTO_UPLOAD_PAYMENT_ADDRESS:-}" ]; then
  echo "==> setting DITTO_UPLOAD_PAYMENT_ADDRESS from deploy env"
  if grep -q '^DITTO_UPLOAD_PAYMENT_ADDRESS=' .env 2>/dev/null; then
    sed -i "s|^DITTO_UPLOAD_PAYMENT_ADDRESS=.*|DITTO_UPLOAD_PAYMENT_ADDRESS=${DITTO_UPLOAD_PAYMENT_ADDRESS}|" .env
  else
    printf 'DITTO_UPLOAD_PAYMENT_ADDRESS=%s\n' "$DITTO_UPLOAD_PAYMENT_ADDRESS" >> .env
  fi
fi

set -a; . ./.env; set +a

# Ensure the Docker infra this host needs is up (Pylon on a deployed host; the
# full local stack in dev). See DITTO_COMPOSE_SERVICES in scripts/start.sh.
compose_services="${DITTO_COMPOSE_SERVICES:-postgres minio pylon}"
echo "==> ensuring infra ($compose_services)"
# shellcheck disable=SC2086
docker compose up -d --wait $compose_services

echo "==> applying migrations"
uv run alembic upgrade head

# Start-or-reload: the first deploy is the app's first start (the Ansible role
# provisions the host but never starts the app), so `pm2 reload` would fail.
echo "==> (re)starting API"
if pm2 describe ditto-api >/dev/null 2>&1; then
  pm2 reload scripts/ecosystem.config.js --update-env
else
  pm2 start scripts/ecosystem.config.js --update-env
fi
pm2 save

echo "done. pm2 logs ditto-api"
