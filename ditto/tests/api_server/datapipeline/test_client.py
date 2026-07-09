"""Unit tests for the data-pipeline generate client."""

from __future__ import annotations

import httpx
import pytest

from ditto.api_server.datapipeline import (
    DataPipelineError,
    HttpDatasetGenerator,
    NullGenerator,
    create_generator,
)
from ditto.api_server.datapipeline.config import DataPipelineConfig


def _config(**overrides: object) -> DataPipelineConfig:
    base: dict[str, object] = {
        "url": "https://gen.example",
        "run_size": "full",
        "timeout_seconds": 30.0,
        "auth": "none",
    }
    base.update(overrides)
    return DataPipelineConfig(**base)  # type: ignore[arg-type]


def _generator(handler: object) -> HttpDatasetGenerator:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    return HttpDatasetGenerator(_config(), client)


async def test_create_generator_disabled_returns_null() -> None:
    gen = create_generator(_config(url=None))
    assert isinstance(gen, NullGenerator)
    assert gen.run_size is None
    with pytest.raises(DataPipelineError):
        await gen.generate(42)


async def test_generate_returns_hash_from_header() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["seed"] = request.url.params["seed"]
        seen["run_size"] = request.url.params["run_size"]
        return httpx.Response(200, headers={"X-Dataset-SHA256": "AB" * 32})

    gen = _generator(handler)
    sha = await gen.generate(8675309)
    assert sha == "ab" * 32  # normalized to lowercase
    assert seen == {"seed": "8675309", "run_size": "full"}
    await gen.aclose()


async def test_generate_raises_on_missing_header() -> None:
    gen = _generator(lambda _request: httpx.Response(200))
    with pytest.raises(DataPipelineError):
        await gen.generate(1)
    await gen.aclose()


async def test_generate_raises_on_bad_status() -> None:
    gen = _generator(lambda _request: httpx.Response(503))
    with pytest.raises(DataPipelineError):
        await gen.generate(1)
    await gen.aclose()


async def test_fetch_dataset_returns_body_and_hash() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["run_size"] == "medium"
        return httpx.Response(
            200,
            headers={"X-Dataset-SHA256": "CD" * 32},
            json={"seed": 7, "bench_version": 2, "tool_cases": []},
        )

    gen = _generator(handler)
    artifact, sha = await gen.fetch_dataset(7, "medium")
    assert sha == "cd" * 32
    assert artifact["bench_version"] == 2
    await gen.aclose()


async def test_fetch_dataset_raises_on_non_object_body() -> None:
    gen = _generator(
        lambda _r: httpx.Response(
            200, headers={"X-Dataset-SHA256": "cd" * 32}, json=[1, 2, 3]
        )
    )
    with pytest.raises(DataPipelineError):
        await gen.fetch_dataset(1, "full")
    await gen.aclose()


async def test_null_generator_fetch_dataset_raises() -> None:
    gen = create_generator(_config(url=None))
    with pytest.raises(DataPipelineError):
        await gen.fetch_dataset(1, "full")
