// pm2 process definition for the Ditto Platform API.
//
// The API is a long-lived host process; Postgres/MinIO/Pylon stay in Docker.
// Env is loaded from .env by scripts/start.sh before pm2 starts (pm2 inherits
// the parent environment), so this file does not re-parse .env.
//
//   pm2 start scripts/ecosystem.config.js --update-env
//   pm2 logs ditto-api
//   pm2 reload scripts/ecosystem.config.js --update-env   # zero-downtime
//
// `uv run` is used as the launcher so the process always picks up the synced
// .venv without hardcoding an interpreter path.

const path = require("path");
const root = path.resolve(__dirname, "..");

// The screener (the pre-benchmark build + health gate) is a platform-operated
// role, not something validators run. We host it next to the API: it is
// HTTP-decoupled, so pointed at the local API it screens exactly as any external
// screener would, and it drives the host Docker daemon for the build gate.
//
// The screener code lives in the sibling ditto-subnet checkout (override with
// SCREENER_APP_DIR). It is co-hosted only when a screener hotkey is configured;
// otherwise pm2 starts the API alone. The signing wallet/hotkey and
// SCREENER_PLATFORM_API_URL (default http://localhost:8000) come from the host
// env that start.sh loads before pm2.
const subnetDir =
  process.env.SCREENER_APP_DIR || path.resolve(root, "..", "ditto-subnet");

const apps = [
  {
    name: "ditto-api",
    cwd: root,
    script: "uv",
    args: "run python -m ditto.api_server",
    interpreter: "none", // `uv` is a binary, not a Node script

    // Single instance: uvicorn manages its own worker; we run one pm2 fork.
    instances: 1,
    exec_mode: "fork",

    // Resilience.
    autorestart: true,
    max_restarts: 10,
    min_uptime: "10s",
    restart_delay: 2000,
    // Allow uvicorn's 30s graceful shutdown to complete before SIGKILL.
    kill_timeout: 35000,
    max_memory_restart: "750M",

    // Logs.
    out_file: path.join(root, "logs", "ditto-api.out.log"),
    error_file: path.join(root, "logs", "ditto-api.err.log"),
    merge_logs: true,
    time: true, // prefix every log line with a timestamp
  },
];

if (process.env.SCREENER_HOTKEY) {
  apps.push({
    name: "ditto-screener",
    cwd: subnetDir,
    script: "uv",
    args: "run python -m ditto.screener",
    interpreter: "none",

    // Singleton per screener hotkey; the worker orchestrates Docker builds in
    // child processes, so the process itself stays light.
    instances: 1,
    exec_mode: "fork",

    autorestart: true,
    max_restarts: 10,
    min_uptime: "10s",
    restart_delay: 2000,
    // The worker drains on SIGTERM; give an in-flight sweep time to unwind.
    kill_timeout: 30000,
    max_memory_restart: "500M",

    out_file: path.join(root, "logs", "ditto-screener.out.log"),
    error_file: path.join(root, "logs", "ditto-screener.err.log"),
    merge_logs: true,
    time: true,
  });
}

module.exports = { apps };
