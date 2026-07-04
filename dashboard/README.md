# Ditto SN118 public dashboard

A single self-contained `index.html` — the public "front door" for Subnet 118.
No build step, no framework, no external requests, **no secrets**. It reads the
platform's public, aggregate-only API (`GET /api/v1/public/leaderboard`) and
links out to wandb for the per-epoch deep dive. This is Surface 3 in
[`docs/public-telemetry.md`](../docs/public-telemetry.md).

## What it shows

- **Summary cards** — scored miners, top composite, median composite, freshness.
- **Leaderboard** — best eligible score per miner, ranked by composite, with
  composite / tool / memory bars; the leader is highlighted. Click a row for a
  drill-down (tool-vs-memory split, first-seen, rank).
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

It's a static file: upload `index.html` to object storage (S3/MinIO/GCS) and
front it with a CDN, or serve it next to the API (same origin → the default
`/api/v1` base just works). The API sets `Cache-Control: public, max-age=30`;
the dashboard auto-refreshes on the same 30s cadence.
