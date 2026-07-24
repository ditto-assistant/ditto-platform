#!/usr/bin/env bash
#
# Zero-downtime update for the Ditto Platform API:
#   fetch -> reset -> uv sync -> set deploy config -> ensure Pylon -> migrate -> pm2 start/reload.
# Invoked on the host by the ditto-platform deploy workflow (push dev|main ->
# IAP SSH). DITTO_DEPLOY_BRANCH defaults to the current branch; CI passes the
# branch that was pushed so the checkout is deterministic.

set -euo pipefail
cd "$(dirname "$0")/.."

branch="${DITTO_DEPLOY_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
echo "==> fetching + resetting to origin/$branch"
git fetch --prune origin
# -fB force-(re)points the local branch at origin and checks it out, discarding
# any host-side tracked-file drift so the deploy can't wedge. .env,
# .env.deploy, .venv, and logs are gitignored, so they survive (NEVER
# `git clean -x` here).
git checkout -fB "$branch" "origin/$branch"
git reset --hard "origin/$branch"

echo "==> syncing dependencies"
uv sync

# Ansible is the only writer of .env. Deploy-owned values live in a separate
# mode-0600 file so a converge cannot erase them and concurrent deploy/converge
# writes cannot race on one file.
deploy_env_file=.env.deploy
touch "$deploy_env_file"
chmod 0600 "$deploy_env_file"
deploy_owned_keys=(
  DITTO_UPLOAD_PAYMENT_ADDRESS
  DITTO_DASHBOARD_WANDB_URL
  DITTO_TAOSTATS_API_KEY
  DITTO_TAOSTATS_VALIDATOR_NAMES_URL
)

# Recover deterministically from duplicate, truncated, or no-final-newline
# state. This file owns only the keys above; retain the shell-effective last
# complete assignment for each and atomically discard incomplete fragments.
normalize_deploy_env() {
  local next_env key value
  next_env="$(mktemp "${deploy_env_file}.XXXXXX")"
  for key in "${deploy_owned_keys[@]}"; do
    value="$(sed -n "s|^${key}=||p" "$deploy_env_file" 2>/dev/null | tail -n 1)"
    [ -n "$value" ] && printf '%s=%s\n' "$key" "$value" >> "$next_env"
  done
  chmod 0600 "$next_env"
  mv "$next_env" "$deploy_env_file"
}
normalize_deploy_env

# Upsert a deploy-owned KEY=VALUE, replacing an existing line or appending.
# Skips empty values so a missing deploy variable never blanks a working key.
upsert_env() {
  local key="$1" value="$2" next_env
  [ -n "$value" ] || return 0
  echo "==> setting $key from deploy env"
  next_env="$(mktemp "${deploy_env_file}.XXXXXX")"
  if grep -q "^${key}=" "$deploy_env_file" 2>/dev/null; then
    # `|` delimiter + escape any `|`/`&`/`\` in the value so URLs/addresses are safe.
    local esc=${value//\\/\\\\}; esc=${esc//|/\\|}; esc=${esc//&/\\&}
    if ! sed "s|^${key}=.*|${key}=${esc}|" "$deploy_env_file" > "$next_env"; then
      rm -f "$next_env"
      return 1
    fi
  else
    cp "$deploy_env_file" "$next_env"
    printf '%s=%s\n' "$key" "$value" >> "$next_env"
  fi
  chmod 0600 "$next_env"
  mv "$next_env" "$deploy_env_file"
}

# One-way transition for hosts that predate .env.deploy. Copy only a missing
# runtime key and preserve the shell-effective last assignment. Deploy inputs
# and fresh Secret Manager reads below remain authoritative and overwrite it.
copy_base_env_if_missing() {
  local key="$1" value
  grep -q "^${key}=" "$deploy_env_file" 2>/dev/null && return 0
  value="$(sed -n "s|^${key}=||p" .env 2>/dev/null | tail -n 1)"
  upsert_env "$key" "$value"
}

for deploy_owned_key in "${deploy_owned_keys[@]}"; do
  copy_base_env_if_missing "$deploy_owned_key"
done
unset deploy_owned_key

# Deploy-supplied values (GitHub Environment secret / variable, passed by
# deploy.yml): the upload payment address (required at boot) and the public
# wandb project URL injected into the served dashboard's telemetry link.
upsert_env DITTO_UPLOAD_PAYMENT_ADDRESS "${DITTO_UPLOAD_PAYMENT_ADDRESS:-}"
upsert_env DITTO_DASHBOARD_WANDB_URL "${DITTO_DASHBOARD_WANDB_URL:-}"

# Validator-name enrichment is optional decoration. Read its API key directly
# on the VM via the attached runtime service account so the value never crosses
# GitHub Actions or SSH. A failed/slow Secret Manager lookup keeps any existing
# .env.deploy value and must not block a platform deploy.
taostats_secret_project="${DITTO_TAOSTATS_SECRET_PROJECT:-ditto-app-dev}"
taostats_secret_id="${DITTO_TAOSTATS_SECRET_ID:-platform-taostats-api-key}"
taostats_api_key=""
if command -v gcloud >/dev/null 2>&1 && \
  taostats_api_key="$(timeout 15s gcloud secrets versions access latest \
    --project="$taostats_secret_project" \
    --secret="$taostats_secret_id" 2>/dev/null)"; then
  upsert_env DITTO_TAOSTATS_API_KEY "$taostats_api_key"
  upsert_env DITTO_TAOSTATS_VALIDATOR_NAMES_URL \
    "https://api.taostats.io/api/dtao/validator/available/v1?netuid=118"
else
  echo "==> Taostats key unavailable; keeping validator-name enrichment unchanged" >&2
fi
unset taostats_api_key taostats_secret_id taostats_secret_project

set -a
. ./.env
. ./.env.deploy
set +a

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
