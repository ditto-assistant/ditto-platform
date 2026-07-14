# Ditto SN118 public dashboard

A single self-contained `index.html` — the public "front door" for Subnet 118.
No build step, no framework, no external requests, **no secrets**. It reads the
platform's public, aggregate-only API (`GET /api/v1/public/leaderboard` and
`GET /api/v1/public/health`) and links out to wandb for the per-epoch deep dive.
This is Surface 3 in [`docs/public-telemetry.md`](../docs/public-telemetry.md).

## Layout

Desktop/widescreen chrome: a sticky left sidebar lists every section of the
site; each section is a deep-linkable, hash-routed page (`#/overview`,
`#/leaderboard`, `#/pipeline`, `#/health`, `#/benchmark`). On narrow viewports
the sidebar collapses into a horizontal tab strip. The Overview page shows the
summary cards plus condensed previews of the leaderboard and pipeline with
"view all" links into the full pages; the Leaderboard page adds a hotkey
filter and a ranked/provisional toggle; the Submission pipeline page adds
stage chips (with counts) and free-text filtering over up to 200 recent
uploads.

## What it shows

- **Summary cards** — scored miners, top composite, median composite, freshness.
- **Leaderboard** — best eligible score per miner, ranked by composite, with
  composite / tool / memory bars; the leader is highlighted. Click a row for a
  drill-down (tool-vs-memory split, first-seen, rank).
- **Subnet health** — miners, scored miners, scores in the last 24h, average
  latency, and when a validator last scored anything (from `/public/health`).
  Run failures and weight-setting telemetry live in wandb, not here — the
  platform only ever sees a successful score.
- **Anti-overfit assurance** — states plainly that only aggregates are published
  and that dataset seeds rotate every submission.

It intentionally shows **only** what the public API exposes (aggregates). Weights
and full per-epoch telemetry live in wandb (linked), matching the endpoint
boundary in `docs/public-telemetry.md` — the platform does not serve the KOTH
weight vector.

## Configure

Resolved in priority order:

| What | Query string | Meta tag (bake in) | Default |
| --- | --- | --- | --- |
| API base | `?api=https://api.host/api/v1` | `<meta name="ditto:api-base">` | same-origin `/api/v1` |
| wandb link | `?wandb=https://wandb.ai/org/ditto-sn118` | `<meta name="ditto:wandb-url">` | `https://wandb.ai/` |

For a deployed dashboard, edit the two `<meta>` tags in `index.html` so the
defaults are correct and the query string is only needed for testing.

## Run / preview

```sh
# Preview the layout (renders SAMPLE data since no API is reachable):
open dashboard/index.html            # or drag it into a browser

# Against a locally-running API (make api-up):
python -m http.server -d dashboard 8080
# then visit http://localhost:8080/?api=http://localhost:8000/api/v1
```

If the API can't be reached the page renders **sample data** behind a clearly
marked amber banner, so the layout is always previewable before deploy.

## Deploy

**Default (this repo): served by the platform, same-origin.** The API serves
this file at `/` (see `factory.py`), so on the deployed hosts it's already live:

- dev  → `https://platform-api-dev.heyditto.ai/`
- prod → `https://platform-api.heyditto.ai/`

Same-origin means the SPA's `/api/v1/public/*` calls need no CORS and the wandb
link is injected from `DITTO_DASHBOARD_WANDB_URL` at serve time — no need to edit
this file per environment. `DITTO_DASHBOARD_ENABLED=false` runs the API headless.

**Alternative: host it yourself.** It's a plain static file — upload to object
storage (S3/MinIO/GCS) behind a CDN, or any static host. A *cross-origin* host
would additionally require CORS on the API's `/public/*` routes (not currently
enabled, since the default is same-origin). The API sets
`Cache-Control: public, max-age=30` on the data; the SPA auto-refreshes on the
same 30s cadence.
