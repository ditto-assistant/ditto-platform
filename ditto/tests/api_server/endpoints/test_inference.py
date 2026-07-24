import base64
from datetime import UTC, datetime
from uuid import UUID, uuid4

import bittensor
import httpx
import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import HTTPException

from ditto.api_models.inference import InferenceExchangeRequest, InferenceGrantOffer
from ditto.api_server.endpoints.inference import (
    _bounded_provider_cost,
    _exchange_message,
    _locked_upstream_payload,
    _output_token_limit,
    _post_provider_with_retry,
    _provider_preferences,
    _provider_rejection_is_route_observable,
    _ProviderCallError,
    _proxy_message,
    _public_embedding_response,
    _public_provider_response,
    _upstream_provider,
    _validate_request_schema,
    _validated_embedding_payload,
)
from ditto.api_server.endpoints.validator import _verify_signature


@pytest.mark.asyncio
async def test_provider_retry_policy_retries_explicit_transient_statuses() -> None:
    statuses = iter((503, 429, 200))
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(next(statuses), request=request)

    async def no_sleep(_: float) -> None:
        return None

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _post_provider_with_retry(
            client,
            "https://provider.example/v1/request",
            payload={"model": "test"},
            headers={},
            sleep=no_sleep,
        )

    assert result.response.status_code == 200
    assert result.attempts == 3
    assert calls == 3


@pytest.mark.asyncio
async def test_provider_retry_policy_does_not_repeat_ambiguous_read_timeout() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("provider response timed out", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(_ProviderCallError) as raised:
            await _post_provider_with_retry(
                client,
                "https://provider.example/v1/request",
                payload={"model": "test"},
                headers={},
            )

    assert raised.value.attempts == 1
    assert raised.value.timed_out is True
    assert calls == 1


def _exchange(keypair: bittensor.Keypair) -> InferenceExchangeRequest:
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes_raw()
    unsigned = InferenceExchangeRequest(
        validator_hotkey=keypair.ss58_address,
        grant_id=uuid4(),
        broker_public_key=base64.urlsafe_b64encode(public).decode().rstrip("="),
        nonce=uuid4(),
        requested_at=datetime.now(UTC),
        signature="00" * 64,
    )
    return unsigned.model_copy(
        update={"signature": keypair.sign(_exchange_message(unsigned)).hex()}
    )


def test_forged_validator_and_valid_validator_wrong_ticket_fail() -> None:
    validator = bittensor.Keypair.create_from_uri("//Alice")
    forger = bittensor.Keypair.create_from_uri("//Bob")
    request = _exchange(validator)
    assert _verify_signature(
        validator.ss58_address, _exchange_message(request), request.signature
    )
    assert not _verify_signature(
        validator.ss58_address,
        _exchange_message(request),
        forger.sign(_exchange_message(request)).hex(),
    )
    wrong_ticket = request.model_copy(update={"grant_id": uuid4()})
    assert not _verify_signature(
        validator.ss58_address,
        _exchange_message(wrong_ticket),
        request.signature,
    )


def test_broker_proof_binds_generation_nonce_time_and_exact_body() -> None:
    private = Ed25519PrivateKey.generate()
    public = private.public_key()
    grant_id = uuid4()
    nonce = uuid4()
    requested_at = datetime.now(UTC)
    body = b'{"model":"qwen/qwen3-32b","messages":[]}'

    def message(generation: int, request_nonce: UUID, request_body: bytes) -> bytes:
        return _proxy_message(
            grant_id=grant_id,
            generation=generation,
            nonce=request_nonce,
            requested_at=requested_at,
            body=request_body,
        )

    signature = private.sign(message(2, nonce, body))
    public.verify(signature, message(2, nonce, body))
    for changed in (
        message(3, nonce, body),
        message(2, uuid4(), body),
        message(2, nonce, body + b" "),
    ):
        with pytest.raises(InvalidSignature):
            public.verify(signature, changed)


def test_embedding_contract_is_exact_and_response_is_sanitized() -> None:
    model = "perplexity/pplx-embed-v1-0.6b"
    payload = {
        "model": model,
        "input": ["one", "two"],
        "dimensions": 768,
        "encoding_format": "float",
    }
    assert _validated_embedding_payload(payload, model=model, dimensions=768) == [
        "one",
        "two",
    ]
    for changed in (
        {**payload, "model": "attacker/model"},
        {**payload, "dimensions": 1536},
        {**payload, "provider": {"allow_fallbacks": True}},
        {**payload, "input": []},
    ):
        with pytest.raises(HTTPException):
            _validated_embedding_payload(changed, model=model, dimensions=768)

    vector = [0.0] * 768
    public, prompt_tokens = _public_embedding_response(
        {
            "object": "list",
            "model": "pplx-embed-v1-0.6b",
            "provider": "must-not-leak",
            "data": [
                {"object": "embedding", "index": 0, "embedding": vector},
                {"object": "embedding", "index": 1, "embedding": vector},
            ],
            "usage": {"prompt_tokens": 7, "total_tokens": 7, "cost": 1},
        },
        model=model,
        dimensions=768,
        input_count=2,
    )
    assert prompt_tokens == 7
    assert public["model"] == model
    assert public["usage"] == {"prompt_tokens": 7, "total_tokens": 7}
    assert "provider" not in public
    assert "cost" not in str(public)

    for response_model in (
        "Perplexity/pplx-embed-v1-0.6b",
        "pplx-embed-v1-0.6b:latest",
        "attacker/model",
    ):
        with pytest.raises(HTTPException, match="provider identity mismatch"):
            _public_embedding_response(
                {
                    "object": "list",
                    "model": response_model,
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": vector},
                        {"object": "embedding", "index": 1, "embedding": vector},
                    ],
                    "usage": {"prompt_tokens": 7, "total_tokens": 7},
                },
                model=model,
                dimensions=768,
                input_count=2,
            )


def test_output_token_alias_cannot_bypass_ticket_limit() -> None:
    with pytest.raises(HTTPException):
        _output_token_limit({"max_tokens": 1, "max_completion_tokens": 999_999}, 8192)
    with pytest.raises(HTTPException):
        _output_token_limit({"max_completion_tokens": 8193}, 8192)
    assert _output_token_limit({"max_completion_tokens": 32}, 8192) == 32


@pytest.mark.parametrize(
    "escape",
    [
        {"models": ["attacker/model"]},
        {"plugins": [{"id": "web"}]},
        {"provider": {"allow_fallbacks": True}},
        {"reasoning": {"effort": "high"}},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": "http://local"}],
                }
            ]
        },
        {"tools": [{"type": "web_search", "web_search": {}}]},
    ],
)
def test_proxy_schema_rejects_model_provider_and_network_escapes(
    escape: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(escape)
    with pytest.raises(HTTPException):
        _validate_request_schema(payload)


def test_proxy_schema_allows_only_local_function_tools() -> None:
    _validate_request_schema(
        {
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "local harness tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    },
                }
            ],
        }
    )


def test_proxy_schema_accepts_exact_openai_text_content_parts() -> None:
    _validate_request_schema(
        {
            "model": "openai/gpt-oss-20b",
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "follow the tools"}],
                },
                {"role": "user", "content": "hello"},
            ],
        }
    )


@pytest.mark.parametrize(
    "content",
    [
        [],
        [{"type": "image_url", "image_url": {"url": "https://example.test"}}],
        [{"type": "text", "text": "hello", "url": "https://example.test"}],
        [{"type": "text", "text": 1}],
    ],
)
def test_proxy_schema_rejects_non_text_content_parts(content: object) -> None:
    with pytest.raises(HTTPException, match="text content only"):
        _validate_request_schema(
            {
                "model": "openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": content}],
            }
        )


@pytest.mark.parametrize(
    "parameter",
    [
        {"temperature": float("nan")},
        {"temperature": 2.01},
        {"top_p": -0.01},
        {"top_p": "fast"},
        {"seed": True},
        {"seed": 2**63},
        {"stop": ["a", "b", "c", "d", "e"]},
        {"stop": ["ok", 1]},
        {"parallel_tool_calls": 1},
        {"stream": "false"},
        {"n": True},
        {"n": 2},
        {"best_of": 1},
        {"tool_choice": {"type": "function", "function": {"name": ""}}},
        {"tool_choice": {"type": "function", "function": {"name": "x", "x": 1}}},
    ],
)
def test_proxy_schema_rejects_invalid_or_amplifying_scalar_controls(
    parameter: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(parameter)
    with pytest.raises(HTTPException):
        _validate_request_schema(payload)


def test_proxy_schema_accepts_bounded_scalar_controls() -> None:
    _validate_request_schema(
        {
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": -(2**63),
            "stop": ["done"],
            "parallel_tool_calls": False,
            "stream": False,
            "n": 1,
            "tool_choice": {
                "type": "function",
                "function": {"name": "lookup"},
            },
        }
    )


def test_aggregate_route_is_speed_sorted_private_and_fallback_enabled() -> None:
    assert _provider_preferences(
        routing_mode="aggregate_throughput",
        provider="openrouter",
        quantization=None,
    ) == {
        "sort": "throughput",
        "allow_fallbacks": True,
        "data_collection": "deny",
        "zdr": True,
    }
    assert _bounded_provider_cost({"usage": {"cost": 0.012345}}) == 12_345
    assert _bounded_provider_cost({"usage": {"cost": float("nan")}}) is None


def test_v7_upstream_profile_pins_medium_reasoning_without_changing_v6() -> None:
    payload = {
        "model": "attacker/model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": 999,
        "stream": False,
    }
    v7 = _locked_upstream_payload(payload, model="openai/gpt-oss-20b", max_tokens=256)
    assert v7["model"] == "openai/gpt-oss-20b"
    assert v7["max_tokens"] == 256
    assert v7["n"] == 1
    assert v7["stream"] is False
    assert v7["reasoning"] == {"effort": "medium", "exclude": True}
    assert "max_completion_tokens" not in v7

    v6 = _locked_upstream_payload(payload, model="qwen/qwen3-32b", max_tokens=256)
    assert "reasoning" not in v6


def test_caller_shape_rejections_do_not_cool_shared_provider_route() -> None:
    assert not _provider_rejection_is_route_observable(400)
    assert not _provider_rejection_is_route_observable(422)
    for status_code in (401, 402, 403, 404, 408, 409, 429, 500, 503):
        assert _provider_rejection_is_route_observable(status_code)


def test_router_metadata_provider_is_trusted_but_never_returned_to_harness() -> None:
    upstream = {
        "id": "gen-test",
        "object": "chat.completion",
        "created": 1,
        "model": "openai/gpt-oss-20b",
        "provider": "legacy-provider-must-not-leak",
        "system_fingerprint": "provider-specific-fingerprint",
        "service_tier": "provider-specific-tier",
        "openrouter_metadata": {
            "summary": "selected Groq",
            "endpoints": {
                "available": [
                    {
                        "provider": "Groq",
                        "model": "openai/gpt-oss-20b",
                        "selected": True,
                    }
                ]
            },
        },
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "native_finish_reason": "provider-native-finish",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                            "provider_extension": "must-not-leak",
                        }
                    ],
                    "provider_extension": "must-not-leak",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
            "cost": 0.01,
            "cost_details": {"upstream_inference_cost": 0.009},
            "is_byok": False,
        },
    }

    assert _upstream_provider(upstream) == "Groq"
    public = _public_provider_response(upstream)
    encoded = str(public)
    for secret in (
        "Groq",
        "legacy-provider",
        "provider-specific",
        "must-not-leak",
        "openrouter_metadata",
        "system_fingerprint",
        "service_tier",
        "native_finish_reason",
        "cost",
    ):
        assert secret not in encoded
    assert public["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "total_tokens": 7,
    }
    assert public["choices"][0]["message"]["tool_calls"][0] == {
        "id": "call-1",
        "type": "function",
        "function": {"name": "lookup", "arguments": "{}"},
    }


def test_router_metadata_rejects_ambiguous_selected_provider() -> None:
    with pytest.raises(HTTPException):
        _upstream_provider(
            {
                "openrouter_metadata": {
                    "endpoints": {
                        "available": [
                            {"provider": "Groq", "selected": True},
                            {"provider": "Together", "selected": True},
                        ]
                    }
                }
            }
        )


def test_provider_choice_error_metadata_is_not_deliverable() -> None:
    with pytest.raises(HTTPException):
        _public_provider_response(
            {
                "id": "gen-test",
                "object": "chat.completion",
                "created": 1,
                "model": "openai/gpt-oss-20b",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": None},
                        "error": {
                            "message": "Groq raw error",
                            "metadata": {"provider": "Groq"},
                        },
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 0},
            }
        )


def test_adaptive_route_remains_exact_and_disables_fallback() -> None:
    assert _provider_preferences(
        routing_mode="adaptive",
        provider="Groq",
        quantization="fp8",
    ) == {
        "only": ["Groq"],
        "quantizations": ["fp8"],
        "allow_fallbacks": False,
        "data_collection": "deny",
        "zdr": True,
    }


def test_legacy_offer_omits_additive_v7_route_identity() -> None:
    offer = InferenceGrantOffer(
        grant_id=uuid4(),
        exchange_url="https://platform.test/api/v1/inference/exchange",
        proxy_url="https://platform.test/api/v1/inference/chat/completions",
        allowed_models=["qwen/qwen3-32b"],
        request_budget=10,
        token_budget=100,
        expires_at=datetime.now(UTC),
    )
    encoded = offer.model_dump(mode="json")
    assert "provider" not in encoded
    assert "profile_revision" not in encoded
