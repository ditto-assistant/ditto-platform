#!/usr/bin/env bash
#
# Scripted update for the Ditto Platform API:
#   fetch -> reset -> uv sync -> set deploy config -> ensure Pylon -> migrate ->
#   pm2 start/reload/recreate -> verify the app is serving.
# NOT zero-downtime: ditto-api is a single fork-mode pm2 process, so the reload
# below is a stop/start with ~6s of refused connections (measured), not a
# rolling handover. See scripts/ecosystem.config.js.
# This script exits non-zero if the API does not come back up; see the
# verification block at the bottom. It must never report success on a dead app.
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

# --------------------------------------------------------------------------
# Start / reload / recreate.
#
# `pm2 reload <ecosystem.config.js>` does NOT reconcile the fields that decide
# how a process is launched: `script`, `interpreter`, `interpreter_args`,
# `exec_mode`, and `cwd` are kept from pm2's saved dump even when the ecosystem
# file changes them. `args` and env ARE reconciled. Changing `script` and
# reloading therefore relaunches the OLD program with the NEW args.
#
# That is exactly how the API went down in prod: the app moved from
# `script: "uv"` to `script: ".venv/bin/python"` with `args: "-m
# ditto.api_server"`, pm2 reloaded into `/usr/local/bin/uv -m ditto.api_server`,
# uv exited on `unexpected argument '-m' found`, pm2 parked the app in `waiting
# restart` with pid 0, and the site served 502 -- while this script exited 0.
#
# scripts/pm2_deploy_plan.js diffs each app's running launch identity against
# what the ecosystem file resolves to and picks per app:
#   start    -- pm2 does not know it yet (first deploy on a fresh host)
#   recreate -- launch identity drifted; `pm2 delete` + `pm2 start`
#   reload   -- identity matches; in-place `pm2 reload` (the ordinary path)
#
# Reload stays the default for ordinary code-only deploys. Today that is a
# stop/start anyway (single fork-mode process), but keeping the distinction
# means a future move to `exec_mode: "cluster"` gets real zero-downtime reloads
# without reintroducing this hazard.
echo "==> planning pm2 actions"
command -v node >/dev/null 2>&1 || { echo "ERROR: node not found (pm2 requires it)" >&2; exit 1; }

pm2_plan="$(pm2 jlist 2>/dev/null | node scripts/pm2_deploy_plan.js scripts/ecosystem.config.js)"
[ -n "$pm2_plan" ] || { echo "ERROR: empty pm2 deploy plan; refusing to touch pm2" >&2; exit 1; }

# Space-separated name lists rather than arrays: pm2 app names never contain
# whitespace, and this keeps the script working on bash 3.2 as well as the
# host's bash 5.
fresh_apps=""
reload_apps=""
service_apps=""
oneshot_apps=""
# Column 5 (the configured script path) is unused here; fail_deploy reads it
# back out of "$pm2_plan" only when it has to explain a failure.
while IFS=$'\t' read -r action name role err_log _ reason; do
  [ -n "$name" ] || continue
  case "$role" in
    oneshot) oneshot_apps="$oneshot_apps $name" ;;
    *) service_apps="$service_apps $name" ;;
  esac
  case "$action" in
    recreate)
      echo "    $name: recreate ($reason)"
      # Drift is only fixable by dropping pm2's saved definition. `|| true`:
      # a delete race must not abort a deploy that is about to re-start it.
      pm2 delete "$name" >/dev/null 2>&1 || true
      fresh_apps="$fresh_apps $name"
      ;;
    start)
      echo "    $name: start ($reason)"
      fresh_apps="$fresh_apps $name"
      ;;
    *)
      echo "    $name: reload ($reason)"
      reload_apps="$reload_apps $name"
      ;;
  esac
done <<<"$pm2_plan"

join_csv() { echo "$*" | tr -s ' ' | sed -e 's/^ //' -e 's/ /,/g'; }

if [ -n "${fresh_apps// /}" ]; then
  echo "==> starting:$fresh_apps"
  pm2 start scripts/ecosystem.config.js --only "$(join_csv "$fresh_apps")" --update-env
fi
if [ -n "${reload_apps// /}" ]; then
  echo "==> reloading:$reload_apps"
  pm2 reload scripts/ecosystem.config.js --only "$(join_csv "$reload_apps")" --update-env
fi
pm2 save

# --------------------------------------------------------------------------
# Verify the deploy actually produced a live app.
#
# The defect this closes: the script above can succeed while the app is dead.
# pm2 reporting `online` is NOT proof of life (it reports online for a process
# that never bound its port), so the gate below requires the API to answer HTTP.
DITTO_HEALTH_TIMEOUT="${DITTO_HEALTH_TIMEOUT:-120}"
# Root `/health` is the purpose-built liveness probe (cheap DB + chain reachability
# check); it is also what deploy.yml polls through Caddy. `/api/v1/public/health`
# is a different thing -- an aggregate subnet rollup -- and is not a liveness probe.
health_url="http://127.0.0.1:${API_PORT:-8000}/health"

# Print one app's live state as "status<TAB>pid<TAB>restarts<TAB>exec_path".
pm2_app_state() {
  pm2 jlist 2>/dev/null | node -e '
    const name = process.argv[1];
    let raw = "";
    process.stdin.on("data", (c) => (raw += c)).on("end", () => {
      const at = raw.indexOf("[");
      let list = [];
      try { list = at === -1 ? [] : JSON.parse(raw.slice(at)); } catch { list = []; }
      const app = (Array.isArray(list) ? list : []).find((a) => a && a.name === name);
      if (!app) { console.log("missing\t0\t0\t"); return; }
      const env = app.pm2_env || {};
      console.log([env.status || "unknown", app.pid || env.pm_pid || 0,
                   env.restart_time || 0, env.pm_exec_path || ""].join("\t"));
    });
  ' "$1"
}

# Dump everything an operator needs to diagnose a failed deploy, then exit 1.
fail_deploy() {
  local app="$1" why="$2" state err_log want running_script
  state="$(pm2_app_state "$app")"
  running_script="$(printf '%s' "$state" | cut -f4)"
  echo "" >&2
  echo "ERROR: deploy failed -- $app $why" >&2
  echo "  pm2 status/pid/restarts/script: $state" >&2
  err_log="$(printf '%s\n' "$pm2_plan" | awk -F'\t' -v n="$app" '$2 == n { print $4; exit }')"
  want="$(printf '%s\n' "$pm2_plan" | awk -F'\t' -v n="$app" '$2 == n { print $5; exit }')"
  if [ -n "$err_log" ] && [ -s "$err_log" ]; then
    echo "  --- tail -n 50 $err_log ---" >&2
    tail -n 50 "$err_log" >&2
  else
    echo "  (no error log at ${err_log:-<unset>}; try: pm2 logs $app --lines 50)" >&2
  fi
  # Only raise stale-definition suspicion when the paths actually disagree.
  # A hint that points at the wrong cause is worse than no hint.
  if [ -n "$running_script" ] && [ -n "$want" ] && [ "$running_script" != "$want" ]; then
    echo "" >&2
    echo "  pm2 is running a STALE script path (expected $want)." >&2
    echo "  Recover with: pm2 delete $app && ./scripts/update.sh" >&2
  fi
  exit 1
}

echo "==> verifying apps came up (timeout ${DITTO_HEALTH_TIMEOUT}s)"
deadline=$((SECONDS + DITTO_HEALTH_TIMEOUT))

# Unquoted on purpose: word-splitting the space-separated name list.
# shellcheck disable=SC2086
for app in $service_apps; do
  http_code=""
  while :; do
    IFS=$'\t' read -r status pid restarts _exec_path <<<"$(pm2_app_state "$app")"
    # `errored` is terminal for a service: pm2 exhausted max_restarts.
    [ "$status" = "errored" ] && fail_deploy "$app" "is in pm2 state 'errored'"

    if [ "$status" = "online" ]; then
      if [ "$app" != "ditto-api" ]; then
        echo "    $app: online (pid $pid, $restarts restarts)"
        break
      fi
      # Ground truth for the API: does the port actually answer?
      http_code="$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$health_url" 2>/dev/null || true)"
      if [ "$http_code" = "200" ]; then
        echo "    $app: online and serving 200 at $health_url (pid $pid, $restarts restarts)"
        break
      fi
    fi

    if [ "$SECONDS" -ge "$deadline" ]; then
      # Separate the two failure shapes: never came up, vs up but unhealthy.
      if [ "$status" = "online" ] && [ -n "$http_code" ] && [ "$http_code" != "000" ]; then
        fail_deploy "$app" "is serving but $health_url returned HTTP $http_code (dependency down?)"
      fi
      fail_deploy "$app" "did not come up within ${DITTO_HEALTH_TIMEOUT}s (pm2 status '$status')"
    fi
    sleep 3
  done
done

# One-shots (ditto-screened-image-cleanup) are cron-triggered with
# autorestart:false. `stopped` is their CORRECT terminal state between runs, so
# only an explicit pm2 `errored` counts as a failure here.
# shellcheck disable=SC2086
for app in $oneshot_apps; do
  IFS=$'\t' read -r status _pid _restarts _exec_path <<<"$(pm2_app_state "$app")"
  case "$status" in
    errored) fail_deploy "$app" "is in pm2 state 'errored'" ;;
    missing) fail_deploy "$app" "is not registered with pm2 after start" ;;
    *) echo "    $app: $status (one-shot; not required to be online)" ;;
  esac
done

echo "done. pm2 logs ditto-api"
