#!/usr/bin/env bash
set -euo pipefail

# Update the isolated production screener to an exact ditto-subnet commit and
# prove both systemd health and authenticated platform queue access. This script
# is copied to the VM by the private deploy workflow and run as root.

SCREENER_ROOT="${SCREENER_ROOT:-/opt/ditto/screener}"
SCREENER_USER="${SCREENER_USER:-deploy}"
SCREENER_UNIT="${SCREENER_UNIT:-ditto-screener}"
SCREENER_EXPECTED_SHA="${SCREENER_EXPECTED_SHA:?missing SCREENER_EXPECTED_SHA}"
SCREENER_UV_BIN="${SCREENER_UV_BIN:-/usr/local/bin/uv}"

checkout="$SCREENER_ROOT/src"
venv="$checkout/.venv"
env_file="$SCREENER_ROOT/screener.env"

if [[ "${EUID}" -ne 0 ]]; then
  echo "update-screener.sh must run as root" >&2
  exit 1
fi

for path in "$checkout/.git" "$env_file" "$SCREENER_UV_BIN"; do
  if [[ ! -e "$path" ]]; then
    echo "required screener deployment path is missing: $path" >&2
    exit 1
  fi
done

env_value() {
  local key="$1"
  sed -n "s/^${key}=//p" "$env_file" | tail -n 1
}

probe_platform() {
  local platform_url api_token hotkey
  platform_url="$(env_value SCREENER_PLATFORM_API_URL)"
  api_token="$(env_value SCREENER_API_TOKEN)"
  hotkey="$(env_value SCREENER_HOTKEY)"
  : "${platform_url:?missing SCREENER_PLATFORM_API_URL}"
  : "${api_token:?missing SCREENER_API_TOKEN}"
  : "${hotkey:?missing SCREENER_HOTKEY}"

  # Feed the bearer header over stdin so the token never appears in `ps`.
  curl --fail --silent --show-error --config - \
    "$platform_url/api/v1/screener/queue?limit=1" >/dev/null <<CURL_CONFIG
header = "Authorization: Bearer $api_token"
header = "X-Screener-Hotkey: $hotkey"
CURL_CONFIG
}

current_sha="$(runuser -u "$SCREENER_USER" -- git -C "$checkout" rev-parse HEAD)"
if [[ "$current_sha" == "$SCREENER_EXPECTED_SHA" ]] && \
  systemctl is-active --quiet "$SCREENER_UNIT"; then
  probe_platform
  echo "healthy: $SCREENER_UNIT already at $current_sha; platform queue auth accepted"
  exit 0
fi

echo "==> fetching $SCREENER_EXPECTED_SHA"
runuser -u "$SCREENER_USER" -- git -C "$checkout" fetch --prune origin "$SCREENER_EXPECTED_SHA"
resolved_sha="$(runuser -u "$SCREENER_USER" -- git -C "$checkout" rev-parse FETCH_HEAD)"
if [[ "$resolved_sha" != "$SCREENER_EXPECTED_SHA" ]]; then
  echo "$SCREENER_EXPECTED_SHA resolved to unexpected commit $resolved_sha" >&2
  exit 1
fi

echo "==> checking out $resolved_sha"
runuser -u "$SCREENER_USER" -- git -C "$checkout" reset --hard "$resolved_sha"

echo "==> syncing the frozen environment"
runuser -u "$SCREENER_USER" -- env UV_PROJECT_ENVIRONMENT="$venv" \
  "$SCREENER_UV_BIN" sync --frozen --project "$checkout"

echo "==> restarting $SCREENER_UNIT"
systemctl restart "$SCREENER_UNIT"
for attempt in $(seq 1 30); do
  if systemctl is-active --quiet "$SCREENER_UNIT"; then
    break
  fi
  if [[ "$attempt" -eq 30 ]]; then
    systemctl status "$SCREENER_UNIT" --no-pager >&2 || true
    exit 1
  fi
  sleep 2
done

probe_platform
actual_sha="$(runuser -u "$SCREENER_USER" -- git -C "$checkout" rev-parse HEAD)"
echo "healthy: $SCREENER_UNIT active at $actual_sha; platform queue auth accepted"
