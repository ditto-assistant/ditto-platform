// pm2 process definition for the Ditto Platform API.
//
// The API is a long-lived host process; Postgres/MinIO/Pylon stay in Docker.
// Env is loaded from .env plus optional .env.deploy by scripts/start.sh or
// scripts/update.sh before pm2 starts (pm2 inherits the parent environment), so
// this file does not parse environment files itself.
//
//   pm2 start scripts/ecosystem.config.js --update-env
//   pm2 logs ditto-api
//   pm2 reload scripts/ecosystem.config.js --update-env   # see "Restarts" below
//
// !! CHANGING `script`, `interpreter`, `interpreter_args`, `exec_mode`, OR `cwd`
// !! REQUIRES RECREATING THE APP, NOT RELOADING IT.
// `pm2 reload` reconciles `args` and env but keeps those five fields from pm2's
// saved dump, so a reload after editing them relaunches the OLD program with the
// NEW args. That is exactly how moving `script` from `uv` to `.venv/bin/python`
// took prod down: pm2 ran `/usr/local/bin/uv -m ditto.api_server`, uv exited on
// `unexpected argument '-m' found`, and the API sat in `waiting restart` with
// pid 0 behind a 502.
// scripts/update.sh detects this automatically (scripts/pm2_deploy_plan.js diffs
// the running launch identity and recreates the app when it drifted), so a normal
// deploy is safe. Doing it by hand: `pm2 delete <app>` then `pm2 start`.
//
// Launcher: the venv interpreter is invoked DIRECTLY, not via `uv run`.
// `uv run` does not exec into the interpreter -- it forks it as a child process
// and proxies signals -- so pm2 ends up owning a ~59 MB launcher shim while the
// real uvicorn server (measured ~950 MB RSS in prod) lives in a grandchild pm2
// cannot see. Every per-process pm2 control measures the shim under that
// layout, which silently neutered `max_memory_restart`: the guard read ~59 MB
// forever and could never fire no matter how large the server grew.
//
// Both scripts/start.sh and scripts/update.sh run `uv sync` before they touch
// pm2, so .venv is always current by the time pm2 reads this file. Resolving the
// interpreter here costs that one implicit re-sync per launch and buys pm2
// ownership of the process that actually holds the memory.

const path = require("path");
const root = path.resolve(__dirname, "..");
// `uv sync` always materializes this; it is the same interpreter `uv run` would
// have selected, minus the intervening shim process.
const venvPython = path.join(root, ".venv", "bin", "python");

module.exports = {
  apps: [
    {
      name: "ditto-api",
      cwd: root,
      script: venvPython,
      args: "-m ditto.api_server",
      interpreter: "none", // the venv python is the interpreter, not a Node script

      // Single instance: uvicorn manages its own worker; we run one pm2 fork.
      //
      // Restarts: because this is `exec_mode: "fork"` with `instances: 1`, pm2
      // has no second instance to shift traffic onto, so `pm2 reload` degrades
      // to a hard stop/start -- roughly 6s of refused connections, measured.
      // It is NOT zero-downtime here despite what pm2's docs say about reload
      // in general. Real zero-downtime would require `exec_mode: "cluster"`,
      // which changes production restart behavior (and how uvicorn binds the
      // port) enough that it is a separate operator decision, deliberately not
      // made in the same change as the memory-guard fix.
      instances: 1,
      exec_mode: "fork",

      // Resilience.
      autorestart: true,
      max_restarts: 10,
      min_uptime: "10s",
      restart_delay: 2000,
      // Allow uvicorn's 30s graceful shutdown to complete before SIGKILL.
      kill_timeout: 35000,
      // Runaway-memory backstop, operator-approved band 2-4 GB. Steady state in
      // prod is ~950 MB RSS, so 3 GB is ~3.2x headroom: comfortably clear of the
      // normal working set and of the transient spikes from fully-exhausted
      // substrate storage reads, while still catching a genuine leak long before
      // the 16 GB host starts swapping or the kernel OOM-killer picks a victim.
      // Now that pm2 owns the server process directly, this threshold is live
      // rather than decorative -- which is also why it could not stay at 750 MB:
      // the real process already sits above that and would restart-loop.
      max_memory_restart: "3072M",

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
      script: venvPython,
      args: "scripts/cleanup_screened_images.py",
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
