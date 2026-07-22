import base64
from datetime import UTC, datetime
from uuid import UUID, uuid4

import bittensor
import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import HTTPException

from ditto.api_models.inference import InferenceExchangeRequest
from ditto.api_server.endpoints.inference import (
    _exchange_message,
    _output_token_limit,
    _proxy_message,
)
from ditto.api_server.endpoints.validator import _verify_signature


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


def test_output_token_alias_cannot_bypass_ticket_limit() -> None:
    with pytest.raises(HTTPException):
        _output_token_limit({"max_tokens": 1, "max_completion_tokens": 999_999}, 8192)
    with pytest.raises(HTTPException):
        _output_token_limit({"max_completion_tokens": 8193}, 8192)
    assert _output_token_limit({"max_completion_tokens": 32}, 8192) == 32
