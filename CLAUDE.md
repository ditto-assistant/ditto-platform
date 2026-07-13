# CLAUDE.md

Guidance for Claude Code (and humans) working in **ditto-platform** — the API
server for Bittensor Subnet 118. Read this before making changes.

## What this repo is

The platform/API service only. Miner CLI and the validator daemon live in
[`ditto-subnet`](https://github.com/ditto-assistant/ditto-subnet); the reference
memory harness lives in [`ditto-harness`](https://github.com/ditto-assistant/ditto-harness).
This service talks to clients over HTTP; the **OpenAPI schema is the contract**
(there is no shared package between repos).

**Validator boundary (server side lives here):** this repo owns the
validator-*facing* API — the `/validator/*` endpoints (`endpoints/validator.py`),
their wire models (`api_models/validator.py`), the score ledger in `ditto/db`,
and the `4xxx` validator error codes. The validator **worker/process itself does
not live here** — it runs in `ditto-subnet` (`ditto/validator/`), is stateless
(no DB), and reaches this service only over HTTP. Do not add a `ditto/validator/`
package or any weight-setting / dittobench-scoring code to this repo; that is the
subnet's job. The two `api_models/validator.py` copies are kept in sync via the
OpenAPI contract, not a shared import.

## Architecture in one paragraph

`ditto/api_server` is a FastAPI app assembled in `factory.py:create_api_server()`
from env-driven config (`config.py`). Endpoints live under `endpoints/`; shared
concerns (request-id, auth pass-through, error envelope) under `middleware/`.
Three service modules back the upload flow: `payment_verifier/` (verifies the
on-chain payment proof), `pricing/` (CoinGecko TAO/USD oracle + fee math), and
`storage/` (S3/MinIO). Persistence is `ditto/db` (SQLAlchemy 2.0 async + asyncpg,
Alembic migrations). Chain reads go through `ditto/chain` (Pylon +
async-substrate-interface). Wire shapes are `ditto/api_models` (Pydantic).

## Conventions (match the existing code)

- **Pydantic only in `ditto/api_models`.** Everything internal — configs, value
  objects, results — uses `@dataclass(frozen=True)`.
- **`AgentStatus` lives in `ditto/api_models/agent_status.py`** (it's a wire +
  DB value). `ditto/db/models.py` re-imports it, so `from ditto.db.models import
  AgentStatus` still works. Do not redefine the enum in `db`.
- **Config is env-driven dataclasses** with `parse_*_from_env()` builders and a
  `check_config()` validator that runs at boot. Fail fast with a typed
  `*ConfigError`; never boot with a placeholder.
- **Errors map to numeric codes** via `middleware/error_envelope.py`. Domain
  errors are typed exception subclasses; the envelope handler maps them to HTTP
  status + a stable error code. Add new codes in the documented ranges.
- **Async everywhere** — SQLAlchemy `AsyncSession`, aioboto3, httpx. DB mutations
  happen inside one `async with session.begin()` transaction.
- **Migrations own the schema.** `ditto/db/models.py` describes it in Python but
  Alembic under `alembic/versions/` is the source of truth — keep them in sync and
  add a migration for any schema change.

## Commands

```sh
uv sync                      # install deps
make stack-up                # postgres + pylon + minio (docker)
make migrate                 # alembic upgrade head
make api-up                  # run the API on :8000 (foreground)
make smoke-api               # curl /health
make lint typecheck test     # ruff + mypy + pytest (run before every PR)
```

Run on a host under pm2 with `./scripts/start.sh` (see README). Always run
`make lint`, `make typecheck`, and `make test` before opening a PR — CI enforces
all three on Python 3.11 and 3.12.

## Testing

- `pytest` markers `slow`, `integration`, `localnet`, `e2e` are excluded by
  default. Unit tests use a SQLite fallback (`aiosqlite`); integration tests need
  the live Docker stack.
- Put unit tests next to the package they cover under `ditto/tests/<package>`.

## Gotchas

- Pylon is on host port **8001**; the API owns **8000**.
- `/upload/agent` enforces the tarball size cap from the *actual streamed bytes*
  and re-verifies the SHA-256; `/upload/check` trusts the miner-reported size.
- The upload flow is ordered cheap-before-expensive and stores to S3 *before* the
  DB transaction (orphan blobs are cheap; orphan rows break the state machine).
- Some `/upload/*` validations are intentionally deferred (tar manifest, import
  allowlist, schema diff, banned-hotkey) pending the harness interface + the
  `banned_hotkeys` table. What miners submit and what is / isn't enforced today
  is written up in `docs/submission-contract.md`.

## Branching

`main` (release) ← `dev` (integration) ← `name/topic` feature branches. PRs into
`dev`. Do not commit directly to `main`.
