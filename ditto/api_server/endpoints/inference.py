"""Dark-launchable ticket-scoped OpenRouter inference plane.

Only a validator-authenticated exchange can bind a grant to a trusted local
broker key. Every inference call then requires the opaque grant secret plus an
Ed25519 proof over the exact request body. Request bodies and provider payloads
are intentionally never logged.
"""

from __future__ import annotations

import base64
import hashlib
import json
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
    cost = usage.get("cost", 0)
    if (
        not isinstance(prompt, int)
        or isinstance(prompt, bool)
        or prompt < 0
        or not isinstance(completion, int)
        or isinstance(completion, bool)
        or completion < 0
    ):
        return None
    try:
        microusd = max(0, int(float(cost) * 1_000_000))
    except (TypeError, ValueError, OverflowError):
        microusd = 0
    return prompt, completion, microusd


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
    model = payload.get("model")
    if not isinstance(model, str) or model not in config.allowed_models:
        raise HTTPException(status_code=403, detail="model is not permitted")
    max_tokens = _output_token_limit(payload, config.max_output_tokens)
    if payload.get("n", 1) != 1 or payload.get("best_of", 1) != 1:
        raise HTTPException(
            status_code=400, detail="multiple completions are not supported"
        )

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

    upstream_payload = dict(payload)
    upstream_payload.pop("max_completion_tokens", None)
    upstream_payload.pop("best_of", None)
    upstream_payload["model"] = model
    upstream_payload["max_tokens"] = max_tokens
    upstream_payload["stream"] = False
    upstream_payload["provider"] = {
        "only": [config.provider],
        "allow_fallbacks": False,
        "data_collection": "deny",
        "zdr": True,
    }
    status = "failed"
    usage: tuple[int, int, int] | None = None
    raw: bytes | None = None
    try:
        upstream = await request.app.state.inference_client.post(
            config.upstream_url,
            json=upstream_payload,
            headers={
                "Authorization": f"Bearer {config.openrouter_api_key}",
                "Content-Type": "application/json",
            },
        )
        raw = upstream.content
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
        usage = _bounded_usage(decoded if isinstance(decoded, dict) else {})
        status = "completed"
    except httpx.TimeoutException as error:
        raise HTTPException(
            status_code=504, detail="inference provider timed out"
        ) from error
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502, detail="inference provider unavailable"
        ) from error
    except HTTPException:
        raise
    finally:
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
                now=datetime.now(UTC),
            )
    if not deliverable or raw is None:
        raise HTTPException(status_code=409, detail="inference grant is no longer live")
    return Response(
        content=raw,
        status_code=200,
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


__all__ = ["router"]
