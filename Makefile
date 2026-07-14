.PHONY: lint lint-copy format typecheck test test-integration smoke-pylon smoke-api stack-up stack-down migrate migrate-down migrate-history migrate-current api-up embedder-up embedder-down smoke-embedder

lint:
	uv run ruff format --check .
	uv run ruff check .

lint-copy:
	npm run lint:copy

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy ditto/

test:
	uv run pytest

test-integration:
	set -a && . ./.env && set +a && uv run pytest -m integration

smoke-pylon:
	set -a && . ./.env && set +a && uv run python scripts/smoke_pylon.py

api-up:
	set -a && . ./.env && set +a && uv run python -m ditto.api_server

smoke-api:
	set -a && . ./.env && set +a && \
	curl -sf "http://localhost:$${API_PORT:-8000}/health" > /dev/null && echo "api ok"

stack-up:
	# Wait on the long-lived services to report healthy; bring the
	# one-shot bucket-init sidecar up separately because `--wait`
	# treats its (correct) `exited 0` terminal state as not-healthy
	# and fails the whole target.
	docker compose up -d --wait postgres pylon minio
	docker compose up -d minio-create-bucket

stack-down:
	docker compose down

embedder-up:
	# code-embedding service (opt-in `embedder` profile). First boot downloads
	# the model weights (cached in the embedder_hf_cache volume), so it may take a
	# minute to report ready.
	docker compose --profile embedder up -d embedder

embedder-down:
	docker compose --profile embedder down embedder

smoke-embedder:
	# Verify the embedder answers and returns a vector for a snippet of code.
	set -a && . ./.env && set +a && \
	curl -sf "http://localhost:$${CODE_EMBEDDER_HOST_PORT:-8080}/embed" \
		-H 'Content-Type: application/json' \
		-d '{"inputs": "fn main() { println!(\"hi\"); }", "normalize": true}' \
		| head -c 80 && echo " ... embedder ok"

migrate:
	set -a && . ./.env && set +a && uv run alembic upgrade head

migrate-down:
	set -a && . ./.env && set +a && uv run alembic downgrade -1

migrate-history:
	set -a && . ./.env && set +a && uv run alembic history --verbose

migrate-current:
	set -a && . ./.env && set +a && uv run alembic current
