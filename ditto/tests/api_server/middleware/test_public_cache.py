"""PublicCacheMiddleware TTL, single-flight, and scope coverage."""

import asyncio

import httpx
import pytest
from fastapi import FastAPI, Response

from ditto.api_server.middleware.public_cache import PublicCacheMiddleware


class _Clock:
    def __init__(self) -> None:
        self.value = 1000.0

    def __call__(self) -> float:
        return self.value


def _build(clock: _Clock, *, disabled: bool = False) -> tuple[FastAPI, dict[str, int]]:
    app = FastAPI()
    calls = {"cached": 0, "nostore": 0, "post": 0, "slow": 0, "outside": 0}

    @app.get("/api/v1/public/cached")
    async def cached(response: Response) -> dict:
        calls["cached"] += 1
        response.headers["Cache-Control"] = "public, max-age=10"
        return {"n": calls["cached"]}

    @app.get("/api/v1/public/nostore")
    async def nostore(response: Response) -> dict:
        calls["nostore"] += 1
        response.headers["Cache-Control"] = "no-store"
        return {"n": calls["nostore"]}

    @app.post("/api/v1/public/cached")
    async def posted() -> dict:
        calls["post"] += 1
        return {"n": calls["post"]}

    @app.get("/api/v1/public/slow")
    async def slow(response: Response) -> dict:
        calls["slow"] += 1
        await asyncio.sleep(0.05)
        response.headers["Cache-Control"] = "public, max-age=10"
        return {"n": calls["slow"]}

    @app.get("/api/v1/other")
    async def outside(response: Response) -> dict:
        calls["outside"] += 1
        response.headers["Cache-Control"] = "public, max-age=10"
        return {"n": calls["outside"]}

    # The endpoint-test conftest sets PUBLIC_CACHE_DISABLED for the whole
    # process, so these tests pin the flag explicitly instead of reading env.
    app.add_middleware(PublicCacheMiddleware, now=clock, disabled=disabled)
    return app, calls


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


async def test_fresh_hits_share_one_downstream_call(clock: _Clock) -> None:
    app, calls = _build(clock)
    async with _client(app) as client:
        first = await client.get("/api/v1/public/cached")
        second = await client.get("/api/v1/public/cached")
    assert first.json() == second.json() == {"n": 1}
    assert first.headers["X-Public-Cache"] == "MISS"
    assert second.headers["X-Public-Cache"] == "HIT"
    assert calls["cached"] == 1


async def test_entry_expires_after_declared_max_age(clock: _Clock) -> None:
    app, calls = _build(clock)
    async with _client(app) as client:
        await client.get("/api/v1/public/cached")
        clock.value += 11
        refreshed = await client.get("/api/v1/public/cached")
    assert refreshed.json() == {"n": 2}
    assert calls["cached"] == 2


async def test_no_store_post_and_non_public_paths_bypass(clock: _Clock) -> None:
    app, calls = _build(clock)
    async with _client(app) as client:
        for _ in range(2):
            await client.get("/api/v1/public/nostore")
            await client.post("/api/v1/public/cached")
            await client.get("/api/v1/other")
    assert calls["nostore"] == 2
    assert calls["post"] == 2
    assert calls["outside"] == 2


async def test_query_string_is_part_of_the_key(clock: _Clock) -> None:
    app, calls = _build(clock)
    async with _client(app) as client:
        await client.get("/api/v1/public/cached?page=1")
        await client.get("/api/v1/public/cached?page=2")
        await client.get("/api/v1/public/cached?page=1")
    assert calls["cached"] == 2


async def test_concurrent_misses_are_single_flighted(clock: _Clock) -> None:
    app, calls = _build(clock)
    async with _client(app) as client:
        responses = await asyncio.gather(
            *(client.get("/api/v1/public/slow") for _ in range(8))
        )
    assert calls["slow"] == 1
    assert {r.json()["n"] for r in responses} == {1}


async def test_kill_switch_disables_caching(clock: _Clock) -> None:
    app, calls = _build(clock, disabled=True)
    async with _client(app) as client:
        await client.get("/api/v1/public/cached")
        await client.get("/api/v1/public/cached")
    assert calls["cached"] == 2
