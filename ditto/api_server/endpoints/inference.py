"""Dark-launchable ticket-scoped OpenRouter inference plane.

Only a validator-authenticated exchange can bind a grant to a trusted local
broker key. Every inference call then requires the opaque grant secret plus an
Ed25519 proof over the exact request body. Request bodies and provider payloads
are intentionally never logged.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import time
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any
from uuid import UUID

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import APIRouter, Header, HTTPException, Request, Response

from ditto.api_models.inference import (
    InferenceExchangeRequest,
    InferenceExchangeResponse,
)
from ditto.api_server.endpoints.validator import (
    ChainDep,
    SessionDep,
    ValidatorAuthError,
    _assert_validator_permitted,
    _verify_signature,
)
from ditto.api_server.inference_routing import (
    benchmark_reasoning,
    record_route_observation,
)
from ditto.db.queries.inference import (
    activate_inference_grant,
    begin_inference_request,
    finish_inference_request,
)
from ditto.db.queries.validator_auth import (
    ValidatorRequestReplayError,
    consume_validator_nonce,
)

router = APIRouter(prefix="/inference", tags=["inference"])
_EXCHANGE_MAX_AGE = timedelta(minutes=2)
_PROXY_MAX_AGE = timedelta(seconds=30)
_EMBEDDING_MAX_INPUTS = 256
_PPLX_EMBED_CONTRACT_MODEL = "perplexity/pplx-embed-v1-0.6b"
_PPLX_EMBED_RESPONSE_MODEL = "pplx-embed-v1-0.6b"


def _exchange_message(payload: InferenceExchangeRequest) -> bytes:
    requested = payload.requested_at.astimezone(UTC).isoformat(timespec="microseconds")
    return (
        f"validator-inference:v1:{payload.validator_hotkey}:{payload.grant_id}:"
        f"{payload.broker_public_key.rstrip('=')}:{payload.nonce}:{requested}"
    ).encode()


def _proxy_message(
    *, grant_id: UUID, generation: int, nonce: UUID, requested_at: datetime, body: bytes
) -> bytes:
    requested = requested_at.astimezone(UTC).isoformat(timespec="microseconds")
    digest = hashlib.sha256(body).hexdigest()
    return (
        f"ditto-inference:v1:{grant_id}:{generation}:{nonce}:{requested}:{digest}"
    ).encode()


def _decode_public_key(value: str) -> Ed25519PublicKey:
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        return Ed25519PublicKey.from_public_bytes(raw)
    except (ValueError, TypeError) as error:
        raise HTTPException(
            status_code=401, detail="invalid inference proof"
        ) from error


def _bounded_usage(payload: dict[str, Any]) -> tuple[int, int, int] | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if (
        not isinstance(prompt, int)
        or isinstance(prompt, bool)
        or prompt < 0
        or not isinstance(completion, int)
        or isinstance(completion, bool)
        or completion < 0
    ):
        return None
    if prompt + completion > (1 << 62):
        return None
    # Provider-supplied cost is intentionally ignored. The ticket pins catalog
    # prices and the trusted plane derives cost from validated token counts.
    return prompt, completion, 0


def _bounded_provider_cost(payload: dict[str, Any]) -> int | None:
    """Convert OpenRouter's direct response cost to bounded integer micro-USD."""
    usage = payload.get("usage")
    value = usage.get("cost") if isinstance(usage, dict) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    cost = float(value)
    if not math.isfinite(cost) or cost < 0 or cost > 100:
        return None
    return round(cost * 1_000_000)


def _upstream_provider(payload: dict[str, Any]) -> str | None:
    """Extract one actual provider from private OpenRouter routing metadata."""
    metadata = payload.get("openrouter_metadata")
    if metadata is None:
        # Bounded compatibility for older OpenRouter responses. Current
        # responses expose this only through opt-in router metadata, and cache
        # hits can intentionally omit that metadata.
        provider = payload.get("provider")
        return (
            provider
            if isinstance(provider, str) and 1 <= len(provider) <= 120
            else None
        )
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=502, detail="provider identity mismatch")
    endpoints = metadata.get("endpoints")
    available = endpoints.get("available") if isinstance(endpoints, dict) else None
    if not isinstance(available, list):
        raise HTTPException(status_code=502, detail="provider identity mismatch")
    selected = [
        endpoint.get("provider")
        for endpoint in available
        if isinstance(endpoint, dict) and endpoint.get("selected") is True
    ]
    if (
        len(selected) != 1
        or not isinstance(selected[0], str)
        or not 1 <= len(selected[0]) <= 120
    ):
        raise HTTPException(status_code=502, detail="provider identity mismatch")
    return selected[0]


def _public_provider_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Return only the normalized Chat Completions contract used by harnesses.

    OpenRouter's additive response surface includes provider-specific system
    fingerprints, service tiers, native finish reasons, error metadata, and
    opt-in routing details. None belongs across the untrusted harness boundary.
    """
    response_id = payload.get("id")
    created = payload.get("created")
    model = payload.get("model")
    choices = payload.get("choices")
    if (
        not isinstance(response_id, str)
        or not 1 <= len(response_id) <= 256
        or not isinstance(created, int)
        or isinstance(created, bool)
        or payload.get("object") != "chat.completion"
        or not isinstance(model, str)
        or not isinstance(choices, list)
        or len(choices) != 1
    ):
        raise HTTPException(status_code=502, detail="invalid provider response")

    choice = choices[0]
    if not isinstance(choice, dict) or "error" in choice:
        raise HTTPException(status_code=502, detail="inference provider unavailable")
    index = choice.get("index")
    finish_reason = choice.get("finish_reason")
    message = choice.get("message")
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or finish_reason not in {"stop", "length", "tool_calls", "content_filter"}
        or not isinstance(message, dict)
        or message.get("role") != "assistant"
        or message.get("content") is not None
        and not isinstance(message.get("content"), str)
    ):
        raise HTTPException(status_code=502, detail="invalid provider response")

    public_message: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content"),
    }
    tool_calls = message.get("tool_calls")
    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            raise HTTPException(status_code=502, detail="invalid provider response")
        public_calls: list[dict[str, Any]] = []
        for call in tool_calls:
            function = call.get("function") if isinstance(call, dict) else None
            if (
                not isinstance(call, dict)
                or call.get("type") != "function"
                or not isinstance(call.get("id"), str)
                or not isinstance(function, dict)
                or not isinstance(function.get("name"), str)
                or not isinstance(function.get("arguments"), str)
            ):
                raise HTTPException(status_code=502, detail="invalid provider response")
            public_calls.append(
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "arguments": function["arguments"],
                    },
                }
            )
        public_message["tool_calls"] = public_calls

    usage = _bounded_usage(payload)
    if usage is None:
        raise HTTPException(status_code=502, detail="invalid provider response")
    prompt_tokens, completion_tokens, _ = usage
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": index,
                "finish_reason": finish_reason,
                "message": public_message,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _provider_preferences(
    *,
    routing_mode: str,
    provider: str,
    quantization: str | None,
) -> dict[str, Any]:
    if routing_mode == "aggregate_throughput":
        return {
            "sort": "throughput",
            "allow_fallbacks": True,
            "data_collection": "deny",
            "zdr": True,
        }
    preferences: dict[str, Any] = {
        "only": [provider],
        "allow_fallbacks": False,
        "data_collection": "deny",
        "zdr": True,
    }
    if quantization:
        preferences["quantizations"] = [quantization]
    return preferences


_ALLOWED_REQUEST_FIELDS = {
    "model",
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "seed",
    "stop",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "n",
    "best_of",
    "stream",
}


def _validate_request_schema(payload: dict[str, Any]) -> None:
    """Accept only the text/tool subset used by the benchmark harness."""
    unknown = set(payload) - _ALLOWED_REQUEST_FIELDS
    if unknown:
        raise HTTPException(status_code=400, detail="unsupported inference parameter")
    for name in ("temperature", "top_p"):
        value = payload.get(name)
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise HTTPException(status_code=400, detail=f"invalid {name}")
    temperature = payload.get("temperature")
    if temperature is not None and not 0 <= temperature <= 2:
        raise HTTPException(status_code=400, detail="invalid temperature")
    top_p = payload.get("top_p")
    if top_p is not None and not 0 <= top_p <= 1:
        raise HTTPException(status_code=400, detail="invalid top_p")
    seed = payload.get("seed")
    if seed is not None and (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not -(2**63) <= seed < 2**63
    ):
        raise HTTPException(status_code=400, detail="invalid seed")
    stop = payload.get("stop")
    if stop is not None and not (
        isinstance(stop, str)
        or (
            isinstance(stop, list)
            and 1 <= len(stop) <= 4
            and all(isinstance(item, str) for item in stop)
        )
    ):
        raise HTTPException(status_code=400, detail="invalid stop")
    for name in ("parallel_tool_calls", "stream"):
        if name in payload and not isinstance(payload[name], bool):
            raise HTTPException(status_code=400, detail=f"invalid {name}")
    n = payload.get("n", 1)
    if not isinstance(n, int) or isinstance(n, bool) or n != 1:
        raise HTTPException(
            status_code=400, detail="multiple completions are not supported"
        )
    if "best_of" in payload:
        raise HTTPException(status_code=400, detail="best_of is not supported")
    tool_choice = payload.get("tool_choice")
    if tool_choice is not None:
        valid_named_choice = (
            isinstance(tool_choice, dict)
            and set(tool_choice) == {"type", "function"}
            and tool_choice.get("type") == "function"
            and isinstance(tool_choice.get("function"), dict)
            and set(tool_choice["function"]) == {"name"}
            and isinstance(tool_choice["function"].get("name"), str)
            and bool(tool_choice["function"]["name"])
        )
        valid_string_choice = isinstance(tool_choice, str) and tool_choice in {
            "none",
            "auto",
            "required",
        }
        if not valid_string_choice and not valid_named_choice:
            raise HTTPException(status_code=400, detail="invalid tool_choice")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages must be non-empty")
    for message in messages:
        if not isinstance(message, dict):
            raise HTTPException(status_code=400, detail="invalid message")
        role = message.get("role")
        if not isinstance(role, str):
            raise HTTPException(status_code=400, detail="invalid message")
        allowed = {
            "system": {"role", "content"},
            "user": {"role", "content"},
            "assistant": {"role", "content", "tool_calls"},
            "tool": {"role", "content", "tool_call_id"},
        }.get(role)
        if allowed is None or set(message) - allowed:
            raise HTTPException(status_code=400, detail="invalid message")
        if message.get("content") is not None and not isinstance(
            message.get("content"), str
        ):
            raise HTTPException(status_code=400, detail="text content only")
        tool_calls = message.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            raise HTTPException(status_code=400, detail="invalid tool calls")
        for call in tool_calls:
            if not isinstance(call, dict) or set(call) - {"id", "type", "function"}:
                raise HTTPException(status_code=400, detail="invalid tool call")
            function = call.get("function")
            if (
                call.get("type") != "function"
                or not isinstance(call.get("id"), str)
                or not isinstance(function, dict)
                or set(function) - {"name", "arguments"}
                or not isinstance(function.get("name"), str)
                or not isinstance(function.get("arguments"), str)
            ):
                raise HTTPException(status_code=400, detail="invalid tool call")
    tools = payload.get("tools", [])
    if not isinstance(tools, list):
        raise HTTPException(status_code=400, detail="invalid tools")
    for tool in tools:
        if not isinstance(tool, dict) or set(tool) - {"type", "function"}:
            raise HTTPException(status_code=400, detail="function tools only")
        function = tool.get("function")
        if (
            tool.get("type") != "function"
            or not isinstance(function, dict)
            or set(function) - {"name", "description", "parameters", "strict"}
            or not isinstance(function.get("name"), str)
            or not isinstance(function.get("parameters", {}), dict)
        ):
            raise HTTPException(status_code=400, detail="invalid function tool")


def _output_token_limit(payload: dict[str, Any], maximum: int) -> int:
    """Normalize OpenAI's aliases without allowing one to bypass the other."""
    max_tokens_value = payload.get("max_tokens")
    max_completion_tokens = payload.get("max_completion_tokens")
    if (
        max_tokens_value is not None
        and max_completion_tokens is not None
        and max_tokens_value != max_completion_tokens
    ):
        raise HTTPException(status_code=400, detail="conflicting output token limits")
    value = (
        max_tokens_value
        if max_tokens_value is not None
        else max_completion_tokens
        if max_completion_tokens is not None
        else maximum
    )
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= maximum
    ):
        raise HTTPException(
            status_code=400, detail="max_tokens exceeds the ticket limit"
        )
    return value


def _locked_upstream_payload(
    payload: dict[str, Any], *, model: str, max_tokens: int
) -> dict[str, Any]:
    """Force consensus model/reasoning fields before provider routing."""
    upstream = dict(payload)
    upstream.pop("max_completion_tokens", None)
    upstream.pop("best_of", None)
    upstream["model"] = model
    upstream["max_tokens"] = max_tokens
    upstream["n"] = 1
    upstream["stream"] = False
    reasoning = benchmark_reasoning(model)
    if reasoning is None:
        upstream.pop("reasoning", None)
    else:
        upstream["reasoning"] = reasoning
    return upstream


def _provider_rejection_is_route_observable(status_code: int) -> bool:
    """Exclude caller-shape failures from shared provider-route health."""
    return status_code >= 400 and status_code not in {400, 422}


def _validated_embedding_payload(
    payload: Any, *, model: str, dimensions: int
) -> list[str]:
    if not isinstance(payload, dict) or set(payload) != {
        "model",
        "input",
        "dimensions",
        "encoding_format",
    }:
        raise HTTPException(status_code=400, detail="invalid embedding request")
    inputs = payload.get("input")
    if (
        payload.get("model") != model
        or payload.get("dimensions") != dimensions
        or payload.get("encoding_format") != "float"
        or not isinstance(inputs, list)
        or not 1 <= len(inputs) <= _EMBEDDING_MAX_INPUTS
        or any(not isinstance(value, str) or not value for value in inputs)
    ):
        raise HTTPException(status_code=400, detail="invalid embedding request")
    return inputs


def _public_embedding_response(
    payload: Any, *, model: str, dimensions: int, input_count: int
) -> tuple[dict[str, Any], int]:
    response_models = {model}
    if model == _PPLX_EMBED_CONTRACT_MODEL:
        # OpenRouter accepts the catalog-qualified model ID but Perplexity's
        # response canonicalizes that exact reviewed model to its unqualified
        # name. Keep the outbound contract frozen while accepting only this
        # observed response alias.
        response_models.add(_PPLX_EMBED_RESPONSE_MODEL)
    if not isinstance(payload, dict) or payload.get("model") not in response_models:
        raise HTTPException(status_code=502, detail="provider identity mismatch")
    data = payload.get("data")
    usage = payload.get("usage")
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    if (
        not isinstance(data, list)
        or len(data) != input_count
        or not isinstance(prompt_tokens, int)
        or isinstance(prompt_tokens, bool)
        or prompt_tokens < 0
    ):
        raise HTTPException(status_code=502, detail="invalid provider response")
    public_data: list[dict[str, Any]] = []
    for expected_index, item in enumerate(data):
        vector = item.get("embedding") if isinstance(item, dict) else None
        if (
            not isinstance(item, dict)
            or item.get("index") != expected_index
            or not isinstance(vector, list)
            or len(vector) != dimensions
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in vector
            )
        ):
            raise HTTPException(status_code=502, detail="invalid provider response")
        public_data.append(
            {
                "object": "embedding",
                "index": expected_index,
                "embedding": vector,
            }
        )
    return (
        {
            "object": "list",
            "model": model,
            "data": public_data,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        },
        prompt_tokens,
    )


@router.post("/exchange", response_model=InferenceExchangeResponse)
async def exchange_inference_grant(
    payload: InferenceExchangeRequest,
    request: Request,
    response: Response,
    chain: ChainDep,
    session: SessionDep,
    x_validator_hotkey: Annotated[str | None, Header()] = None,
) -> InferenceExchangeResponse:
    """Rotate a ticket grant onto one validator-authorized trusted broker key."""
    config = request.app.state.config.inference_proxy
    if not config.enabled:
        raise HTTPException(status_code=404, detail="inference proxy is disabled")
    if x_validator_hotkey != payload.validator_hotkey:
        raise ValidatorAuthError("inference exchange hotkey mismatch")
    if (
        abs(datetime.now(UTC) - payload.requested_at.astimezone(UTC))
        > _EXCHANGE_MAX_AGE
    ):
        raise HTTPException(status_code=409, detail="inference exchange is stale")
    if not _verify_signature(
        payload.validator_hotkey, _exchange_message(payload), payload.signature
    ):
        raise ValidatorAuthError("inference exchange signature did not verify")
    await _assert_validator_permitted(
        chain,
        request.app.state.config.chain.netuid,
        payload.validator_hotkey,
        network=request.app.state.config.chain.subtensor_network,
    )
    now = datetime.now(UTC)
    async with session.begin():
        try:
            await consume_validator_nonce(
                session,
                nonce=payload.nonce,
                validator_hotkey=payload.validator_hotkey,
                now=now,
                expires_at=now + _EXCHANGE_MAX_AGE,
            )
        except ValidatorRequestReplayError as error:
            raise HTTPException(
                status_code=409, detail="inference exchange nonce was already used"
            ) from error
        activated = await activate_inference_grant(
            session,
            grant_id=payload.grant_id,
            validator_hotkey=payload.validator_hotkey,
            broker_public_key=payload.broker_public_key,
            now=now,
            config=config,
        )
    if activated is None:
        raise HTTPException(status_code=409, detail="inference grant is not live")
    grant, bearer = activated
    response.headers["Cache-Control"] = "no-store"
    return InferenceExchangeResponse(
        grant_id=grant.grant_id,
        bearer=bearer,
        proxy_url=f"{config.public_base_url}/api/v1/inference/chat/completions",
        expires_at=grant.expires_at,
        generation=grant.generation,
        provider=grant.route_provider if grant.bench_version >= 7 else None,
        profile_revision=grant.route_profile if grant.bench_version >= 7 else None,
        model=grant.allowed_models[0] if grant.bench_version >= 7 else None,
    )


@router.post("/chat/completions")
async def proxy_chat_completions(
    request: Request,
    x_ditto_grant: Annotated[UUID | None, Header()] = None,
    x_ditto_generation: Annotated[int | None, Header()] = None,
    x_ditto_nonce: Annotated[UUID | None, Header()] = None,
    x_ditto_requested_at: Annotated[datetime | None, Header()] = None,
    x_ditto_proof: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> Response:
    """Proxy one non-streaming chat request under atomic ticket budgets."""
    config = request.app.state.config.inference_proxy
    if not config.enabled or config.openrouter_api_key is None:
        raise HTTPException(status_code=404, detail="inference proxy is disabled")
    if None in {
        x_ditto_grant,
        x_ditto_generation,
        x_ditto_nonce,
        x_ditto_requested_at,
        x_ditto_proof,
        authorization,
    }:
        raise HTTPException(status_code=401, detail="missing inference proof")
    assert x_ditto_grant is not None
    assert x_ditto_generation is not None
    assert x_ditto_nonce is not None
    assert x_ditto_requested_at is not None
    assert x_ditto_proof is not None
    assert authorization is not None
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="invalid inference proof")
    body = await request.body()
    if len(body) > config.request_body_bytes:
        raise HTTPException(status_code=413, detail="inference request is too large")
    if abs(datetime.now(UTC) - x_ditto_requested_at.astimezone(UTC)) > _PROXY_MAX_AGE:
        raise HTTPException(status_code=409, detail="inference request is stale")
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise HTTPException(status_code=400, detail="invalid JSON request") from error
    if not isinstance(payload, dict) or payload.get("stream") not in {None, False}:
        raise HTTPException(status_code=400, detail="streaming is not supported")
    _validate_request_schema(payload)
    model = payload.get("model")
    if not isinstance(model, str) or model not in config.allowed_models:
        raise HTTPException(status_code=403, detail="model is not permitted")
    max_tokens = _output_token_limit(payload, config.max_output_tokens)

    session_maker = request.app.state.session_maker
    now = datetime.now(UTC)
    async with session_maker() as session, session.begin():
        from ditto.db.models import InferenceGrant

        grant = await session.get(InferenceGrant, x_ditto_grant)
        if (
            grant is None
            or grant.broker_public_key is None
            or grant.generation != x_ditto_generation
        ):
            raise HTTPException(status_code=401, detail="invalid inference proof")
        try:
            proof = base64.urlsafe_b64decode(
                x_ditto_proof + "=" * (-len(x_ditto_proof) % 4)
            )
            _decode_public_key(grant.broker_public_key).verify(
                proof,
                _proxy_message(
                    grant_id=x_ditto_grant,
                    generation=x_ditto_generation,
                    nonce=x_ditto_nonce,
                    requested_at=x_ditto_requested_at,
                    body=body,
                ),
            )
        except (ValueError, InvalidSignature) as error:
            raise HTTPException(
                status_code=401, detail="invalid inference proof"
            ) from error
        reserved = await begin_inference_request(
            session,
            grant_id=x_ditto_grant,
            nonce=x_ditto_nonce,
            bearer=authorization.removeprefix("Bearer "),
            model=model,
            # A tokenizer-independent upper bound: a token cannot consume less
            # than one byte of the UTF-8 request body. Reserve the full body
            # plus the permitted output before the provider call so concurrent
            # requests cannot collectively cross the ticket budget.
            token_reservation=max_tokens + max(1, len(body)),
            now=now,
            config=config,
        )
        if reserved is None:
            raise HTTPException(status_code=429, detail="inference grant unavailable")

    upstream_payload = _locked_upstream_payload(
        payload, model=model, max_tokens=max_tokens
    )
    reserved_grant = reserved[0]
    if not reserved_grant.route_provider:
        raise HTTPException(status_code=409, detail="inference route unavailable")
    aggregate_routing = config.routing_mode == "aggregate_throughput"
    upstream_payload["provider"] = _provider_preferences(
        routing_mode=config.routing_mode,
        provider=reserved_grant.route_provider,
        quantization=reserved_grant.route_quantization,
    )
    status = "failed"
    usage: tuple[int, int, int] | None = None
    raw: bytes | None = None
    started = time.monotonic()
    timed_out = False
    route_observable = False
    upstream_provider: str | None = None
    try:
        upstream = await request.app.state.inference_client.post(
            config.upstream_url,
            json=upstream_payload,
            headers={
                "Authorization": f"Bearer {config.openrouter_api_key}",
                "Content-Type": "application/json",
                # OpenRouter keeps route identity private unless explicitly
                # requested. It is consumed below for trusted telemetry and
                # removed by the public response allowlist.
                "X-OpenRouter-Metadata": "enabled",
            },
        )
        raw = upstream.content
        # Authentication, balance, throttling, and availability failures are
        # route-health evidence. A 400/422 can still be an unrecognized caller
        # request-shape error and must not let one ticket cool the shared route.
        route_observable = _provider_rejection_is_route_observable(upstream.status_code)
        if len(raw) > config.response_body_bytes:
            raise HTTPException(
                status_code=502, detail="provider response is too large"
            )
        if upstream.status_code >= 400:
            raise HTTPException(
                status_code=502, detail="inference provider unavailable"
            )
        try:
            decoded = upstream.json()
        except ValueError as error:
            raise HTTPException(
                status_code=502, detail="invalid provider response"
            ) from error
        route_observable = True
        usage = _bounded_usage(decoded if isinstance(decoded, dict) else {})
        if not isinstance(decoded, dict) or decoded.get("model") != model:
            raise HTTPException(status_code=502, detail="provider identity mismatch")
        provider_value = _upstream_provider(decoded)
        if provider_value is None:
            raise HTTPException(status_code=502, detail="provider identity mismatch")
        upstream_provider = provider_value
        if not aggregate_routing and provider_value != reserved_grant.route_provider:
            raise HTTPException(status_code=502, detail="provider identity mismatch")
        if usage is not None:
            prompt, completion, _ = usage
            if aggregate_routing:
                trusted_cost = _bounded_provider_cost(decoded)
                if trusted_cost is None:
                    usage = None
                else:
                    usage = (prompt, completion, trusted_cost)
            elif (
                reserved_grant.route_prompt_price_per_token is None
                or reserved_grant.route_completion_price_per_token is None
            ):
                usage = None
            else:
                trusted_cost = int(
                    (
                        prompt * reserved_grant.route_prompt_price_per_token
                        + completion * reserved_grant.route_completion_price_per_token
                    )
                    * 1_000_000
                )
                usage = (prompt, completion, trusted_cost)
        # The harness needs the normalized completion and token counts, not
        # provider, payer, router, system-fingerprint, or native-error metadata.
        public_response = _public_provider_response(decoded)
        raw = json.dumps(public_response, separators=(",", ":")).encode()
        status = "completed"
    except httpx.TimeoutException as error:
        timed_out = True
        route_observable = True
        raise HTTPException(
            status_code=504, detail="inference provider timed out"
        ) from error
    except httpx.HTTPError as error:
        route_observable = True
        raise HTTPException(
            status_code=502, detail="inference provider unavailable"
        ) from error
    except HTTPException:
        raise
    finally:
        finished_at = datetime.now(UTC)
        async with session_maker() as session, session.begin():
            deliverable = await finish_inference_request(
                session,
                grant_id=x_ditto_grant,
                nonce=x_ditto_nonce,
                generation=x_ditto_generation,
                status=status,
                prompt_tokens=usage[0] if usage is not None else 0,
                completion_tokens=usage[1] if usage is not None else 0,
                cost_microusd=usage[2] if usage is not None else 0,
                usage_available=usage is not None,
                now=finished_at,
                upstream_provider=upstream_provider,
                timed_out=timed_out,
                latency_ms=max(0, round((time.monotonic() - started) * 1000)),
            )
            from ditto.db.models import InferenceGrant

            observed_grant = await session.get(InferenceGrant, x_ditto_grant)
            if observed_grant is not None and route_observable:
                await record_route_observation(
                    session,
                    grant=observed_grant,
                    success=status == "completed" and usage is not None,
                    latency_ms=(time.monotonic() - started) * 1000,
                    completion_tokens=usage[1] if usage is not None else 0,
                    cost_microusd=usage[2] if usage is not None else 0,
                    timed_out=timed_out,
                    now=finished_at,
                )
    if not deliverable or raw is None:
        raise HTTPException(status_code=409, detail="inference grant is no longer live")
    return Response(
        content=raw,
        status_code=200,
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


@router.post("/embeddings")
async def proxy_embeddings(
    request: Request,
    x_ditto_grant: Annotated[UUID | None, Header()] = None,
    x_ditto_generation: Annotated[int | None, Header()] = None,
    x_ditto_nonce: Annotated[UUID | None, Header()] = None,
    x_ditto_requested_at: Annotated[datetime | None, Header()] = None,
    x_ditto_proof: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> Response:
    """Proxy the one reviewed v7 embedding contract under separate budgets."""
    config = request.app.state.config.inference_proxy
    if not config.enabled or config.openrouter_api_key is None:
        raise HTTPException(status_code=404, detail="inference proxy is disabled")
    if None in {
        x_ditto_grant,
        x_ditto_generation,
        x_ditto_nonce,
        x_ditto_requested_at,
        x_ditto_proof,
        authorization,
    }:
        raise HTTPException(status_code=401, detail="missing inference proof")
    assert x_ditto_grant is not None
    assert x_ditto_generation is not None
    assert x_ditto_nonce is not None
    assert x_ditto_requested_at is not None
    assert x_ditto_proof is not None
    assert authorization is not None
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="invalid inference proof")
    body = await request.body()
    if len(body) > config.embedding_request_body_bytes:
        raise HTTPException(status_code=413, detail="embedding request is too large")
    if abs(datetime.now(UTC) - x_ditto_requested_at.astimezone(UTC)) > _PROXY_MAX_AGE:
        raise HTTPException(status_code=409, detail="embedding request is stale")
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise HTTPException(status_code=400, detail="invalid JSON request") from error
    inputs = _validated_embedding_payload(
        payload, model=config.embedding_model, dimensions=config.embedding_dimensions
    )

    session_maker = request.app.state.session_maker
    now = datetime.now(UTC)
    async with session_maker() as session, session.begin():
        from ditto.db.models import InferenceGrant

        grant = await session.get(InferenceGrant, x_ditto_grant)
        if (
            grant is None
            or grant.bench_version != 7
            or grant.broker_public_key is None
            or grant.generation != x_ditto_generation
            or grant.embedding_model != config.embedding_model
            or grant.embedding_profile != config.embedding_profile
            or grant.embedding_provider != config.embedding_provider
            or grant.embedding_dimensions != config.embedding_dimensions
        ):
            raise HTTPException(status_code=401, detail="invalid inference proof")
        try:
            proof = base64.urlsafe_b64decode(
                x_ditto_proof + "=" * (-len(x_ditto_proof) % 4)
            )
            _decode_public_key(grant.broker_public_key).verify(
                proof,
                _proxy_message(
                    grant_id=x_ditto_grant,
                    generation=x_ditto_generation,
                    nonce=x_ditto_nonce,
                    requested_at=x_ditto_requested_at,
                    body=body,
                ),
            )
        except (ValueError, InvalidSignature) as error:
            raise HTTPException(
                status_code=401, detail="invalid inference proof"
            ) from error
        reserved = await begin_inference_request(
            session,
            grant_id=x_ditto_grant,
            nonce=x_ditto_nonce,
            bearer=authorization.removeprefix("Bearer "),
            model=config.embedding_model,
            token_reservation=max(1, len(body)),
            now=now,
            config=config,
            request_kind="embedding",
        )
        if reserved is None:
            raise HTTPException(status_code=429, detail="embedding grant unavailable")

    upstream_payload = {
        "model": config.embedding_model,
        "input": inputs,
        "dimensions": config.embedding_dimensions,
        "encoding_format": "float",
        "provider": {
            "order": [config.embedding_provider],
            "allow_fallbacks": False,
            "data_collection": "deny",
        },
    }
    status = "failed"
    prompt_tokens = 0
    raw: bytes | None = None
    timed_out = False
    started = time.monotonic()
    try:
        upstream: httpx.Response | None = None
        for attempt in range(3):
            try:
                candidate = await request.app.state.inference_client.post(
                    config.embedding_upstream_url,
                    json=upstream_payload,
                    headers={
                        "Authorization": f"Bearer {config.openrouter_api_key}",
                        "Content-Type": "application/json",
                    },
                )
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == 2:
                    raise
                await asyncio.sleep(0.25 * (2**attempt))
                continue
            if candidate.status_code in {408, 429, 500, 502, 503, 504} and attempt < 2:
                await asyncio.sleep(0.25 * (2**attempt))
                continue
            upstream = candidate
            break
        if upstream is None:
            raise HTTPException(
                status_code=502, detail="embedding provider unavailable"
            )
        if len(upstream.content) > config.embedding_response_body_bytes:
            raise HTTPException(
                status_code=502, detail="embedding response is too large"
            )
        if upstream.status_code >= 400:
            raise HTTPException(
                status_code=502, detail="embedding provider unavailable"
            )
        try:
            decoded = upstream.json()
        except ValueError as error:
            raise HTTPException(
                status_code=502, detail="invalid provider response"
            ) from error
        public_response, prompt_tokens = _public_embedding_response(
            decoded,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
            input_count=len(inputs),
        )
        raw = json.dumps(public_response, separators=(",", ":")).encode()
        status = "completed"
    except httpx.TimeoutException as error:
        timed_out = True
        raise HTTPException(
            status_code=504, detail="embedding provider timed out"
        ) from error
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502, detail="embedding provider unavailable"
        ) from error
    finally:
        finished_at = datetime.now(UTC)
        async with session_maker() as session, session.begin():
            deliverable = await finish_inference_request(
                session,
                grant_id=x_ditto_grant,
                nonce=x_ditto_nonce,
                generation=x_ditto_generation,
                status=status,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                # Catalog price is $0.004 / 1M input tokens. Provider-reported
                # direct cost is intentionally not trusted.
                cost_microusd=round(prompt_tokens * 0.004),
                usage_available=status == "completed",
                now=finished_at,
                upstream_provider=config.embedding_provider,
                timed_out=timed_out,
                latency_ms=max(0, round((time.monotonic() - started) * 1000)),
            )
    if not deliverable or raw is None:
        raise HTTPException(status_code=409, detail="embedding grant is no longer live")
    return Response(
        content=raw,
        status_code=200,
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


__all__ = ["router"]
