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
# -fB force-(re)points the local branch at origin and checks it out, discarding
# any host-side tracked-file drift so the deploy can't wedge. .env/.venv/logs are
# gitignored, so they survive (NEVER `git clean -x` here).
git checkout -fB "$branch" "origin/$branch"
git reset --hard "origin/$branch"

echo "==> syncing dependencies"
uv sync

# Upsert a KEY=VALUE into .env, replacing an existing line or appending. Used for
# deploy-supplied env the Ansible role deliberately leaves out of the rendered
# .env; applied BEFORE sourcing so the app boots with the right values. Skips
# when the value is empty so an unset deploy var never blanks a key.
upsert_env() {
  local key="$1" value="$2"
  [ -n "$value" ] || return 0
  echo "==> setting $key from deploy env"
  if grep -q "^${key}=" .env 2>/dev/null; then
    # `|` delimiter + escape any `|`/`&`/`\` in the value so URLs/addresses are safe.
    local esc=${value//\\/\\\\}; esc=${esc//|/\\|}; esc=${esc//&/\\&}
    sed -i "s|^${key}=.*|${key}=${esc}|" .env
  else
    printf '%s=%s\n' "$key" "$value" >> .env
  fi
}

# Deploy-supplied values (GitHub Environment secret / variable, passed by
# deploy.yml): the upload payment address (required at boot) and the public
# wandb project URL injected into the served dashboard's telemetry link.
upsert_env DITTO_UPLOAD_PAYMENT_ADDRESS "${DITTO_UPLOAD_PAYMENT_ADDRESS:-}"
upsert_env DITTO_DASHBOARD_WANDB_URL "${DITTO_DASHBOARD_WANDB_URL:-}"

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
