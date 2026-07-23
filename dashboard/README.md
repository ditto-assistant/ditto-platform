# Ditto SN118 public dashboard

A single self-contained `index.html` — the public "front door" for Subnet 118.
No build step, no framework, no external requests, **no secrets**. It reads the
platform's public API and links out to wandb for the per-epoch deep dive.
This is Surface 3 in [`docs/public-telemetry.md`](../docs/public-telemetry.md).

## What it shows

- **Subnet snapshot** — total miners are the primary signal, with scored-miner,
  leaderboard, throughput, and latency metrics in one top-level panel.
- **Leaderboard** — best eligible score per miner, ranked by raw finalized
  composite, with a separate KOTH emissions projection that identifies the
  first-seen incumbent champion and participation-tail recipients. The projection
  applies the validator's frozen 0.007 composite-point hysteresis, statistical
  dethrone band, and v6+ high-score decay, so raw rank #1 is never mislabeled as
  champion. A native Subtensor read
  overlays the last publicly revealed validator vectors at one block, while
  explicitly separating those lagging commit-reveal inputs from stake-weighted
  Yuma emissions. Click a row for a drill-down (tool-vs-memory split, first-seen,
  raw rank, projected emissions role, and revealed validator top-choice/support
  counts). Current SN118 registration
  is reported separately: a deregistered hotkey's immutable score stays visible
  but is marked inactive and excluded from weights and emissions until that same
  hotkey registers again.
- **Submission pipeline** — screening and validator-ticket history, including a
  compact accessible benchmark progress bar for each validator currently
  evaluating the submission. Active benchmark work takes precedence over a
  submission's previously completed stage, so version-rollout rescoring stays
  aligned with the validator fleet. Running work carries its ticket-bound bench
  version, and top-five qualification rows state that the prior score remains
  authoritative while the next-version quorum is collected. Accepted numeric scores appear immediately
  in the current-version summary as provisional feedback; the prior final
  median remains authoritative until the new three-validator quorum. Each score
  includes its post-commit seed and a
  version-pinned `dittobench-datagen` reproduction command, without exposing
  ticket signatures or associating the number with a validator identity.
- **Validator fleet** — signed worker availability, coarse system health, and
  the public active agent with the same stage/progress shown in the pipeline.
  Old clients render as progress not reported; expired or stale work disappears.
- **Stable object links** — all SPA state (popup/selected-row params, the
  submissions filters, and both pagers) lives in a query string inside the hash,
  on whatever page it was opened from (`#/submissions?agent={id}`,
  `#/overview?miner={hotkey}`, `#/operations?validator={hotkey}`,
  `#/submissions?status=rejected&page=2`, `#/overview?page=2` for the
  leaderboard page). Page-scoped view state (the filters and either pager's
  `page`) is cleared when you navigate to a different page, so it never trails
  along as stale state — which is also why both pagers can share the `page` key
  without colliding.
  The real query string carries only deploy/config knobs (`?api=`, `?wandb=`),
  so the document URL — and its HTTP cache entry — stays stable while the SPA
  navigates. Agent and miner popovers link to dedicated `/agent/{id}` and
  `/miner/{hotkey}` pages. Direct visits and browser back/forward navigation
  restore the same state; older link forms (`?agent={id}#/submissions`
  real-query state, plural pathname and hash routes) are recognized and
  normalized to the current form.
- **Anti-overfit assurance** — explains that seeds are fixed only after the
  submission is committed, rotate per submission, and can reproduce a completed
  evaluation without changing the already-submitted artifact.

It intentionally shows **only** what the public API exposes. In-progress score
rows are a narrow safe projection (composite, deterministic dataset inputs, and
acceptance time); identities, signatures, ticket leases, and scorer internals
stay private. The leaderboard serves a read-only KOTH projection for explanation;
validators still compute and submit the authoritative weight vector independently,
and Yuma combines their revealed inputs stake-weightedly. Full per-epoch
telemetry remains in wandb (linked).

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
# Preview the API-unavailable state:
open dashboard/index.html            # or drag it into a browser

# Against a locally-running API (make api-up):
python -m http.server -d dashboard 8080
# then visit http://localhost:8080/?api=http://localhost:8000/api/v1
```

If the API can't be reached the page renders an explicit unavailable state. It
never substitutes sample values for live subnet data.

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
