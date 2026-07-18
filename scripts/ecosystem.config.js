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

module.exports = {
  apps: [
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
    {
      // DB-aware retention: keeps evaluating/current-best images, clears old
      // non-champions back to source-build fallback, then deletes their objects.
      // Bucket lifecycle separately aborts abandoned multipart uploads.
      name: "ditto-screened-image-cleanup",
      cwd: root,
      script: "uv",
      args: "run python scripts/cleanup_screened_images.py",
      interpreter: "none",
      autorestart: false,
      cron_restart: "17 3 * * *",
      out_file: path.join(root, "logs", "ditto-image-cleanup.out.log"),
      error_file: path.join(root, "logs", "ditto-image-cleanup.err.log"),
      merge_logs: true,
      time: true,
    },
  ],
};
