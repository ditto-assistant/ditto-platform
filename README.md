# Ditto Platform

**The API server for Ditto, Bittensor Subnet 118 (SN118).**

Ditto Platform is the central, team-operated service that sits between **miners**
(who submit agent-memory harnesses) and **validators** (who evaluate them and set
weights on chain). It owns miner intake, on-chain payment verification, object
storage for submissions, the evaluation job queue, the score ledger, and the
operational state machine that moves a submission from `uploaded` to `live`.

> The chain is the settlement layer (weights, stake, payments). This platform is
> the **workflow** layer the chain can't hold — queues, leases, payment
> replay-protection, submission status, and the public score ledger.

---

## Where this fits

```
┌─────────────┐   upload (HTTP)    ┌──────────────────┐   poll / score (HTTP)   ┌─────────────┐
│  miner CLI  │ ─────────────────▶ │  Ditto Platform  │ ◀────────────────────── │ validators  │
│ ditto-subnet│                    │   (this repo)    │                         │ ditto-subnet│
└─────────────┘                    └──────────────────┘                         └──────┬──────┘
                                     │   │   │   │                                      │ put_weights
                                  Postgres │ MinIO/S3                                   ▼
                                       Pylon (subtensor)                          Bittensor chain
```

- **Miner side & validator daemon:** [`ditto-subnet`](https://github.com/ditto-assistant/ditto-subnet)
- **Reference memory harness (what miners fork):** [`ditto-harness`](https://github.com/ditto-assistant/ditto-harness)
- **This repo** is the platform/API only. It is intentionally split out so it can
  be deployed and scaled independently of the miner/validator code.

The contract between this service and the miner/validator is the **OpenAPI schema**
served at `/docs` (Swagger) and `/openapi.json`. There is no shared Python package;
clients are validated against that schema.

---

## API surface

### Built (miner intake + status)

| Method & path | Purpose |
| --- | --- |
| `GET /health` | Liveness + DB/chain readiness + build commit |
| `GET /metrics` | Prometheus metrics |
| `GET /api/v1/upload/eval-pricing` | Quote the upload fee in rao (CoinGecko TAO/USD oracle) |
| `POST /api/v1/upload/check` | Pre-payment dry-run validation (signature, registration, size) |
| `POST /api/v1/upload/agent` | Verified submission: re-check payment on chain → store tarball → write `agents` + `evaluation_payments` atomically |
| `GET /api/v1/retrieval/agent-by-hotkey` | Look up a miner's latest agent |
| `GET /api/v1/retrieval/agent/{id}/status` | Poll a submission's lifecycle status |

> `/health` and `/metrics` are unprefixed; all other routes are versioned under `/api/v1`.

### Built (validator-facing)

| Method & path | Purpose |
| --- | --- |
| `POST /api/v1/validator/job` | Lease a scoring ticket (seed, dataset_sha256, run_size, deadline) |
| `POST /api/v1/validator/heartbeat` | Submit a signed runtime heartbeat with optional coarse system health |
| `GET /api/v1/public/validators` | Read the public-safe validator fleet view |
| `GET /api/v1/public/screeners` | Read the public-safe platform screener fleet view |
| `GET /api/v1/public/bench/config` | The frozen benchmark setup: locked model, judge-free grading, seed derivation, mirror |
| `GET /api/v1/validator/agent/{id}/artifact` | Presigned download URL for an agent tarball |
| `POST /api/v1/validator/agent/{id}/score` | Submit a signed DittoBench score (→ `scores` table) |

### Built (screener-facing)

Screening is a **platform-operated** pre-evaluation gate: a dedicated host the team
runs (not the validators). It drains `uploaded` agents, `docker build`s and
health-checks each crate in isolation, and promotes pass → `evaluating` /
fail → `screening_failed`, so a crate that does not compile never costs a full
benchmark. It authenticates with a dedicated screener credential (an allowlisted
hotkey plus a bearer token), not a validator permit, so the screener key holds no
stake. A validator may optionally run its own screener locally, but the authoritative
gate is the one the platform hosts.

| Method & path | Purpose |
| --- | --- |
| `GET /api/v1/screener/queue` | List agents awaiting screening (status `uploaded`), oldest first |
| `POST /api/v1/screener/heartbeat` | Submit a dedicated-auth signed screener health report |
| `GET /api/v1/screener/agent/{id}/artifact` | Presigned download URL for the crate tarball |
| `POST /api/v1/screener/agent/{id}/result` | Signed pass/fail verdict that promotes the agent |

### Planned (scoring + ops)

Weight/score aggregation (`/scoring/*`) and `/admin/*`. See
[`PROJECT.md`](https://github.com/ditto-assistant/ditto-subnet/blob/main/PROJECT.md) in `ditto-subnet`
for the evaluation/scoring design.

---

## Tech stack

- **API:** FastAPI + Uvicorn (Python 3.11+)
- **Wire models:** Pydantic (`ditto/api_models`) — the only place Pydantic is used
- **Database:** PostgreSQL via SQLAlchemy 2.0 async + asyncpg, migrations with Alembic
- **Object storage:** S3-compatible via aioboto3 (MinIO locally)
- **Chain reads:** [Pylon](https://github.com/bittensor-church/bittensor-pylon) +
  `async-substrate-interface`
- **Pricing:** CoinGecko oracle with in-process cache + stale-guard
- **Observability:** Prometheus metrics, structured request-id logging
- **Tooling:** `uv` (deps/venv), `ruff` (lint/format), `mypy`, `pytest`

---

## Quickstart (local development)

### Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (Python toolchain + venv)
- Node.js 22 and npm (copy lint only)
- Docker + Docker Compose (Postgres, MinIO, Pylon)
- Python 3.11 or 3.12

### 1. Configure

```sh
cp .env.example .env
```

Then edit `.env` and set **`DITTO_UPLOAD_PAYMENT_ADDRESS`** to a real SS58 address
— the server validates it at boot and refuses to start with the placeholder. All
other defaults match the local Docker stack.

### 2. Bring up infra + the API

```sh
uv sync                # install dependencies into .venv
make stack-up          # postgres + pylon + minio (waits until healthy)
make migrate           # apply alembic migrations
make api-up            # run the FastAPI app on :8000 (foreground)
```

In another terminal:

```sh
make smoke-api         # curl /health
open http://localhost:8000/docs   # interactive API docs
```

`make stack-down` stops the Docker services. Postgres state persists in a named
volume across restarts; `docker compose down -v` for a hard reset.

> Pylon runs on host port **8001** so the API can own **8000**.

---

## Running on a host with pm2 (staging / production)

The API is a long-lived process; we run it under [pm2](https://pm2.keymetrics.io/)
on the host (the database and object store stay in Docker). Logs and zero-downtime
updates are first-class.

```sh
npm install -g pm2           # one-time, if not present
./scripts/start.sh           # infra up + migrate + start API under pm2
pm2 logs ditto-api           # tail logs
pm2 status                   # process state
./scripts/update.sh          # git pull + uv sync + migrate + zero-downtime reload
./scripts/stop.sh            # stop the API process
```

- pm2 config: [`scripts/ecosystem.config.js`](scripts/ecosystem.config.js)
- Logs are written to `./logs/ditto-api.{out,err}.log` and via `pm2 logs`.
- `pm2 startup` + `pm2 save` will resurrect the process across host reboots.

---

## Make targets

| Target | Description |
| --- | --- |
| `make lint` | `ruff format --check` + `ruff check` |
| `make lint-copy` | lint public dashboard copy with Faircopy |
| `make format` | `ruff format` + `ruff check --fix` |
| `make typecheck` | `mypy ditto/` |
| `make test` | unit test suite (`pytest`) |
| `make test-integration` | integration tests against the live stack |
| `make api-up` | run the API in the foreground against the local stack |
| `make smoke-api` | curl `/health` to confirm reachability |
| `make smoke-pylon` | exercise the chain client against live Pylon |
| `make stack-up` / `make stack-down` | bring Docker services up / down |
| `make migrate` / `make migrate-down` | apply / roll back one migration |
| `make migrate-history` / `make migrate-current` | alembic history / current head |

---

## Project layout

```
ditto/
  api_server/          FastAPI app
    endpoints/         health · metrics · upload · retrieval · validator
    middleware/        request-id · auth pass-through · error envelope
    payment_verifier/  on-chain payment proof verification
    pricing/           CoinGecko oracle + upload-fee config
    storage/           S3/MinIO client
    config.py          env-driven ApiServerConfig
    factory.py         create_api_server() + lifespan
    __main__.py        process entry point (argparse + uvicorn)
  api_models/          Pydantic wire shapes (the client contract)
    agent_status.py    canonical AgentStatus lifecycle enum
  chain/               Pylon-backed ChainClient
  db/                  SQLAlchemy models + queries + engine/session factory
  tests/               unit + integration tests
alembic/               database migrations
scripts/               pm2 ecosystem + start/stop/update + smoke_pylon
```

---

## Configuration

All configuration is environment-driven; see [`.env.example`](.env.example) for the
full annotated list. Key groups: **API** (`API_HOST/PORT/LOG_LEVEL`), **Pylon/chain**
(`PYLON_URL`, `PYLON_OPEN_ACCESS_TOKEN`, `NETUID`, `SUBTENSOR_NETWORK`), **Postgres**
(`POSTGRES_*`), **upload/pricing** (`DITTO_UPLOAD_PAYMENT_ADDRESS`,
`DITTO_UPLOAD_FEE_USD`, `DITTO_UPLOAD_FEE_BUFFER`), and **object storage**
(`STORAGE_*`). The server validates config at boot and exits non-zero on a bad value
so a supervisor restarts cleanly.

---

## Testing

```sh
make test                 # fast unit suite (default markers excluded)
make test-integration     # requires the live Docker stack (make stack-up)
```

Test markers (`slow`, `integration`, `localnet`, `e2e`) are excluded by default;
CI runs `ruff`, `mypy`, and `pytest` on every PR and on `main`.

---

## Branching

`main` (protected, release) ← `dev` (integration) ← feature branches
(`name/topic`, e.g. `dan/api_init`). Open PRs into `dev`; `dev` merges to `main`
via PR.
