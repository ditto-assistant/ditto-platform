"""Upload-flow endpoints.

This module ships the pre-payment surface a miner CLI hits before
spending TAO (``/upload/eval-pricing`` + ``/upload/check``) and the
post-payment orchestrator (``/upload/agent``) that re-verifies the
proof on chain, stores the tarball in S3, and writes the matching
``agents`` + ``evaluation_payments`` rows in a single transaction.

Deferred validations (added when their dependencies land):
- tar manifest structure (needs Go-harness interface signatures)
- Go-import allowlist scan (needs the allowlist file)
- schema diff against ``schema/initial_harness.sql`` (needs the file)
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import os
import uuid
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated

import bittensor
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from ditto.api_models import (
    EvalPricingResponse,
    UploadAgentResponse,
    UploadCheckRequest,
    UploadCheckResponse,
)
from ditto.api_models.upload import (
    _BLOCK_HASH_PATTERN,
    _SHA256_PATTERN,
    _SIGNATURE_HEX_PATTERN,
    _SS58_PATTERN,
)
from ditto.api_server.dependencies import (
    get_chain_client,
    get_embedder,
    get_payment_verifier,
    get_price_oracle,
    get_session,
    get_storage_client,
)
from ditto.api_server.embedding import Embedder
from ditto.api_server.fingerprint import (
    compute_content_fingerprint,
    compute_embedding_input,
    compute_normalized_source_hash,
    compute_prompt_fingerprint,
)
from ditto.api_server.payment_verifier import (
    PaymentProof,
    PaymentReplayedError,
    PaymentVerifier,
)
from ditto.api_server.pricing import (
    MalformedPriceError,
    PriceOracle,
)
from ditto.api_server.storage import S3StorageClient
from ditto.chain import ChainError
from ditto.db.models import AgentStatus
from ditto.db.queries.agents import insert_agent
from ditto.db.queries.bans import is_hotkey_banned
from ditto.db.queries.payments import (
    consume_evaluation_credit,
    get_agent_for_payment_proof,
    get_evaluation_payment_for_proof,
    get_same_hotkey_agent_by_sha,
    get_same_owner_agent_by_sha,
    insert_evaluation_payment,
)

if TYPE_CHECKING:
    from ditto.chain import ChainClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])

# `/upload/check` + `/upload/agent` failure codes live in the 1xxx
# agent-side range per CODE-REVIEW-CHECKLIST.md. New codes added here
# go in 110x.
ERROR_CODE_BAD_SIGNATURE = 1100
ERROR_CODE_HOTKEY_NOT_REGISTERED = 1101
ERROR_CODE_TARBALL_TOO_LARGE = 1102
ERROR_CODE_HOTKEY_BANNED = 1103
ERROR_CODE_IDENTICAL_SUBMISSION = 1104

DEFAULT_MAX_TARBALL_SIZE_BYTES = 20 * 1024 * 1024


def _tarball_size_cap_from_env() -> int:
    """Return upload cap, keeping the competition default explicit.

    Rust starter-kit submissions with bundled model fixtures are expected to
    stay below the launch cap. Operators may still override it explicitly for
    local/dev runs or emergency changes.
    """
    raw = os.environ.get("DITTO_MAX_TARBALL_SIZE_BYTES", "").strip()
    if not raw:
        return DEFAULT_MAX_TARBALL_SIZE_BYTES
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "invalid DITTO_MAX_TARBALL_SIZE_BYTES=%r; falling back to 20 MiB",
            raw,
        )
        return DEFAULT_MAX_TARBALL_SIZE_BYTES
    if value <= 0:
        logger.warning(
            "non-positive DITTO_MAX_TARBALL_SIZE_BYTES=%r; falling back to 20 MiB",
            raw,
        )
        return DEFAULT_MAX_TARBALL_SIZE_BYTES
    return value


# Hard cap shared with /upload/check. Tarballs above this size are
# rejected; /upload/check enforces it from the miner-reported header,
# /upload/agent enforces it from the actual streamed bytes.
MAX_TARBALL_SIZE_BYTES = _tarball_size_cap_from_env()

# Streaming read chunk size. 256 KiB keeps memory bounded while letting
# size + sha256 update incrementally without re-reading the body.
_CHUNK_SIZE_BYTES = 256 * 1024

ChainDep = Annotated["ChainClient", Depends(get_chain_client)]
OracleDep = Annotated[PriceOracle, Depends(get_price_oracle)]
PaymentVerifierDep = Annotated[PaymentVerifier, Depends(get_payment_verifier)]
StorageDep = Annotated[S3StorageClient, Depends(get_storage_client)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]


@router.get("/eval-pricing", response_model=EvalPricingResponse)
async def eval_pricing(request: Request, oracle: OracleDep) -> EvalPricingResponse:
    """Quote the current upload fee in rao.

    ``PricingError`` subclasses propagate so the envelope handler in
    :mod:`ditto.api_server.middleware.error_envelope` can attach the
    specific 31xx error code instead of the generic 3002 catch-all.
    """
    config = request.app.state.config
    price_usd = await oracle.get_tao_usd()

    fee_tao = (config.pricing.fee_usd * config.pricing.fee_buffer) / price_usd
    amount_rao = int(fee_tao * Decimal("1e9"))
    if amount_rao <= 0:
        raise MalformedPriceError(f"computed amount_rao is non-positive: {amount_rao}")

    return EvalPricingResponse(
        amount_rao=amount_rao,
        send_address=config.upload_payment_address,
    )


@router.post("/check", response_model=UploadCheckResponse)
async def check(
    request: Request, body: UploadCheckRequest, chain: ChainDep, session: SessionDep
) -> UploadCheckResponse:
    """Pre-payment dry-run validation.

    Aggregates every failed check into ``error_codes`` + ``messages`` so
    the miner CLI sees every reason in one round trip. ``file_size_bytes``
    is miner-reported and unverified at this endpoint; the next-PR
    ``/upload/agent`` re-derives it from the actual tarball bytes.
    """
    netuid = request.app.state.config.chain.netuid
    codes: list[int] = []
    messages: list[str] = []

    # 1. Signature over UTF-8 bytes of "{hotkey}:{sha256}".
    payload = f"{body.hotkey}:{body.sha256}".encode()
    signature_valid = _verify_signature(body.hotkey, payload, body.signature)
    if not signature_valid:
        codes.append(ERROR_CODE_BAD_SIGNATURE)
        messages.append("signature did not verify against the hotkey")

    # 2. Hotkey registered. On a chain outage we return 503 instead of
    #    a silent false-pass that would lie to miners.
    try:
        registered = await chain.is_registered(body.hotkey, netuid=netuid)
    except ChainError as e:
        logger.warning(f"chain unreachable during /upload/check: {e}")
        raise HTTPException(
            status_code=503, detail="chain unavailable; retry shortly"
        ) from e
    if not registered:
        codes.append(ERROR_CODE_HOTKEY_NOT_REGISTERED)
        messages.append(f"hotkey is not registered on netuid {netuid}")

    # 3. Tarball size cap.
    if body.file_size_bytes > MAX_TARBALL_SIZE_BYTES:
        codes.append(ERROR_CODE_TARBALL_TOO_LARGE)
        messages.append(f"tarball exceeds {MAX_TARBALL_SIZE_BYTES} bytes")

    # 4. Hotkey-level ban. Reported here (dry run) so a banned miner learns it
    #    before spending TAO; /upload/agent enforces it as a hard 403.
    banned = await is_hotkey_banned(session, hotkey=body.hotkey)
    if banned:
        codes.append(ERROR_CODE_HOTKEY_BANNED)
        messages.append("hotkey is banned from submitting")

    # 5. Stop the common accidental duplicate before the miner pays. A second
    # independently seeded run remains available, but it must be explicit.
    duplicate = None
    if (
        signature_valid
        and registered
        and not banned
        and not body.allow_identical_rescore
    ):
        duplicate = await get_same_hotkey_agent_by_sha(
            session, miner_hotkey=body.hotkey, sha256=body.sha256
        )
        if duplicate:
            codes.append(ERROR_CODE_IDENTICAL_SUBMISSION)
            messages.append(
                "identical artifact already submitted; no payment is required. "
                "Set allow_identical_rescore=true only to purchase another seed."
            )

    return UploadCheckResponse(
        ok=not codes,
        error_codes=codes,
        messages=messages,
        payment_required=not codes,
        identical_agent_id=duplicate.agent_id if duplicate else None,
        identical_agent_status=duplicate.status if duplicate else None,
    )


@router.post(
    "/agent",
    response_model=UploadAgentResponse,
    response_model_exclude_defaults=True,
    status_code=200,
)
async def upload_agent(
    request: Request,
    agent_tar: Annotated[UploadFile, File(description="gzipped tarball, <=20 MB")],
    hotkey: Annotated[str, Form(pattern=_SS58_PATTERN)],
    sha256: Annotated[str, Form(pattern=_SHA256_PATTERN)],
    # The 64-character cap is a chosen value rather than a spec mandate;
    # ``agents.name`` is TEXT in the schema and the cap is the only
    # defense against pathological values polluting logs / dashboards.
    name: Annotated[str, Form(min_length=1, max_length=64)],
    signature: Annotated[str, Form(pattern=_SIGNATURE_HEX_PATTERN)],
    payment_block_hash: Annotated[str, Form(pattern=_BLOCK_HASH_PATTERN)],
    payment_block_number: Annotated[int, Form(ge=1)],
    payment_extrinsic_index: Annotated[int, Form(ge=0)],
    chain: ChainDep,
    verifier: PaymentVerifierDep,
    storage: StorageDep,
    embedder: EmbedderDep,
    session: SessionDep,
    allow_identical_rescore: Annotated[bool, Form()] = False,
) -> UploadAgentResponse:
    """Full upload submission with proof of payment.

    Ordering is cheap-before-expensive so a rejection costs the API the
    minimum work, and every mutation happens after every validation has
    passed:

    1. Form fields auto-validated by FastAPI regex (already done by
       the time this body runs; malformed input returns 422).
    2. Signature over ``f"{hotkey}:{sha256}"`` (CPU only, no I/O; 400).
    3. Hotkey registered on the configured netuid (1 Pylon call;
       400 if absent, 503 if chain unreachable).
    4. Stream tar bytes: size cap (413) + sha256 re-verify (400).
    5. Resolve the proof as an idempotent retry or an available duplicate-upload
       credit. Reusing an assigned proof for different upload data remains a
       3207 replay rejection.
    6. For a fresh proof, run ``PaymentVerifier.verify_payment`` (4 chain calls;
       3201-3206 on payment rejection, 503 if chain unreachable).
    7. Detect byte-identical source under the immutable payment-time coldkey.
       Unless explicitly opted into another seed, preserve the fresh proof as a
       reusable credit and return the original agent without storing a new one.
    8. ``agent_id = uuid4()`` and ``storage.put_object`` (an orphan blob is cheap
       on DB failure;
       orphan agent rows would break the state machine), then compute the
       content fingerprint (best-effort; ``None`` on an unreadable tarball).
    9. Atomic DB tx: insert the agent and either assign a fresh proof or consume
       the locked credit (3207 surfaces if another request won the proof race).
    10. Return ``UploadAgentResponse``.
    """
    netuid = request.app.state.config.chain.netuid

    # 2. Signature verify against the claimed hotkey + sha.
    payload = f"{hotkey}:{sha256}".encode()
    if not _verify_signature(hotkey, payload, signature):
        raise HTTPException(
            status_code=400, detail="signature did not verify against the hotkey"
        )

    # 2b. Hotkey-level ban. Checked right after the (CPU-only) signature proves
    #     the caller owns the hotkey and before any chain/payment/storage work,
    #     so a banned miner is rejected as cheaply as possible.
    if await is_hotkey_banned(session, hotkey=hotkey):
        raise HTTPException(status_code=403, detail="hotkey is banned from submitting")

    # The ban check autobegan a transaction on the pooled session. End it NOW:
    # nothing until the atomic insert (step 8) touches the database, and holding
    # a checked-out connection across the slow middle — streaming the tarball
    # from a possibly-slow miner, chain payment verification, the storage write,
    # and the CPU-bound fingerprint computes — starves the pool under concurrent
    # uploads (the 2026-07-16 outage: idle-in-transaction sessions pinned every
    # slot while requests queued 30s for a connection).
    if session.in_transaction():
        rollback_result = session.rollback()
        if inspect.isawaitable(rollback_result):
            await rollback_result

    # 3. Hotkey must be registered on this subnet. Chain outage surfaces
    # as 503; falling through would silently accept off-subnet hotkeys.
    try:
        registered = await chain.is_registered(hotkey, netuid=netuid)
    except ChainError as e:
        logger.warning(f"chain unreachable during /upload/agent: {e}")
        raise HTTPException(
            status_code=503, detail="chain unavailable; retry shortly"
        ) from e
    if not registered:
        raise HTTPException(
            status_code=400, detail=f"hotkey not registered on netuid {netuid}"
        )

    # 4. Stream the tar; enforce size cap + recompute sha256 on bytes.
    tar_bytes, actual_sha = await _read_tar_capped_with_sha(
        agent_tar, MAX_TARBALL_SIZE_BYTES
    )
    if actual_sha != sha256:
        raise HTTPException(
            status_code=400, detail="sha256 of received tarball does not match claim"
        )

    # 5. An upstream/proxy failure can hide the 200 after the original atomic
    # commit. Authenticate and re-hash first, then recover only an *exact* retry.
    # The payment proof remains non-transferable: changing hotkey, name, or bytes
    # keeps the existing 3207 replay rejection.
    existing = await get_agent_for_payment_proof(
        session,
        block_hash=payment_block_hash,
        extrinsic_index=payment_extrinsic_index,
    )
    if existing:
        if (
            existing.miner_hotkey == hotkey
            and existing.name == name
            and existing.sha256 == sha256
        ):
            assert existing.version is not None
            logger.info(
                "upload retry recovered hotkey=%s agent_id=%s version=%s block_hash=%s",
                hotkey,
                existing.agent_id,
                existing.version,
                payment_block_hash,
            )
            return UploadAgentResponse(
                agent_id=existing.agent_id,
                version=existing.version,
                status=existing.status,
            )
        raise PaymentReplayedError("payment proof already used by a different upload")

    payment_record = await get_evaluation_payment_for_proof(
        session,
        block_hash=payment_block_hash,
        extrinsic_index=payment_extrinsic_index,
    )
    if payment_record and payment_record.agent_id is not None:
        raise PaymentReplayedError("payment proof already used by a different upload")
    if payment_record and payment_record.miner_hotkey != hotkey:
        raise PaymentReplayedError("payment credit belongs to a different hotkey")
    using_credit = bool(payment_record)
    credit_owner_coldkey = payment_record.miner_coldkey if payment_record else None

    # The replay lookup autobegan a read transaction. Release that pooled
    # connection before the slow chain/storage/fingerprint work below.
    if session.in_transaction():
        rollback_result = session.rollback()
        if inspect.isawaitable(rollback_result):
            await rollback_result

    # 6. Chain-side verification. Typed PaymentVerifierError subclasses
    # are mapped to 3201-3206 by the envelope handler; we re-raise them
    # unchanged. A bare ChainError surfaces when one of the verifier's
    # four chain reads cannot reach Pylon, which we treat as a 503 to
    # match the shipped /upload/check pattern around chain.is_registered.
    verified = None
    if using_credit:
        assert credit_owner_coldkey is not None
        owner_coldkey = credit_owner_coldkey
    else:
        try:
            verified = await verifier.verify_payment(
                PaymentProof(
                    block_hash=payment_block_hash,
                    block_number=payment_block_number,
                    extrinsic_index=payment_extrinsic_index,
                ),
                expected_hotkey=hotkey,
            )
        except ChainError as e:
            logger.warning(f"chain unreachable during /upload/agent verify: {e}")
            raise HTTPException(
                status_code=503, detail="chain unavailable; retry shortly"
            ) from e
        owner_coldkey = verified.miner_coldkey

    duplicate = await get_same_owner_agent_by_sha(
        session, miner_coldkey=owner_coldkey, sha256=sha256
    )
    if duplicate and not allow_identical_rescore:
        if not using_credit:
            assert verified is not None
            if session.in_transaction():
                rollback_result = session.rollback()
                if inspect.isawaitable(rollback_result):
                    await rollback_result
            try:
                async with session.begin():
                    await insert_evaluation_payment(
                        session,
                        verified=verified,
                        credit_for_agent_id=duplicate.agent_id,
                    )
            except PaymentReplayedError:
                raced = await get_evaluation_payment_for_proof(
                    session,
                    block_hash=payment_block_hash,
                    extrinsic_index=payment_extrinsic_index,
                )
                if not (
                    raced and raced.agent_id is None and raced.miner_hotkey == hotkey
                ):
                    raise
        assert duplicate.version is not None
        return UploadAgentResponse(
            agent_id=duplicate.agent_id,
            version=duplicate.version,
            status=duplicate.status,
            payment_disposition="reusable_credit",
            credit_for_agent_id=duplicate.agent_id,
        )

    # The duplicate lookup autobegan a read transaction. Do not pin a pooled
    # connection while uploading to storage and computing fingerprints.
    if session.in_transaction():
        rollback_result = session.rollback()
        if inspect.isawaitable(rollback_result):
            await rollback_result

    # 7. Server-generated identity. The CLI cannot pre-supply it.
    agent_id = uuid.uuid4()

    # 8. S3 first: orphan blobs are cheap + invisible to the state
    # machine. Orphan agent rows would surface as undownloadable agents
    # in the validator polling flow.
    await storage.put_object(
        key=f"{agent_id}/agent.tar.gz",
        body=tar_bytes,
        content_type="application/gzip",
    )

    # 7b. Content fingerprint for the anti-copy gate's content-level signal.
    # Computed only now, on an upload that has passed every check, so a rejected
    # submission never pays the unpack cost. Offloaded to a worker thread because
    # it is CPU-bound (gunzip + shingle-hash the whole tree) and would otherwise
    # block the event loop for every concurrent request. Best-effort: an
    # unreadable/empty tarball yields None (the gate then relies on sha256 + size),
    # never a 500.
    content_fingerprint = await asyncio.to_thread(
        compute_content_fingerprint, tar_bytes
    )
    # 7c. exact-repack hash: the canonicalized-source equality signal for the
    # gate (comments/whitespace stripped, files sorted). Same CPU-bound offload +
    # best-effort None contract as the lexical fingerprint above.
    normalized_source_hash = await asyncio.to_thread(
        compute_normalized_source_hash, tar_bytes
    )
    # 7d. prompt-surface sketch (shadow mode): stored for every agent for
    # calibration/retroactive analysis; not yet a hold trigger. Same offload +
    # best-effort None contract.
    prompt_fingerprint = await asyncio.to_thread(compute_prompt_fingerprint, tar_bytes)
    # 7e. code embedding (shadow mode): build the canonical input (CPU-bound,
    # offloaded) then embed via the self-hosted service. Disabled by default
    # (null embedder -> None) and best-effort: a slow/unreachable embedder yields a
    # null vector rather than failing the upload. The provenance tag is stored so a
    # model change can drive a re-embed sweep and the gate compares only same-model
    # vectors.
    embed_input = await asyncio.to_thread(compute_embedding_input, tar_bytes)
    code_embedding = await embedder.embed(embed_input) if embed_input else None
    code_embed_model = embedder.model_tag if code_embedding is not None else None

    if session.in_transaction():
        rollback_result = session.rollback()
        if inspect.isawaitable(rollback_result):
            await rollback_result

    # 9. Atomic DB tx: agent + payment commit together or roll back
    # together. A replayed payment proof surfaces as PaymentReplayedError
    # (3207) and the envelope handler maps it to HTTP 402.
    try:
        async with session.begin():
            version = await insert_agent(
                session,
                agent_id=agent_id,
                miner_hotkey=hotkey,
                name=name,
                sha256=sha256,
                size_bytes=len(tar_bytes),
                content_fingerprint=content_fingerprint,
                normalized_source_hash=normalized_source_hash,
                prompt_fingerprint=prompt_fingerprint,
                code_embedding=code_embedding,
                code_embed_model=code_embed_model,
            )
            if using_credit:
                locked_credit = await get_evaluation_payment_for_proof(
                    session,
                    block_hash=payment_block_hash,
                    extrinsic_index=payment_extrinsic_index,
                    for_update=True,
                )
                if locked_credit is None:
                    raise PaymentReplayedError("payment credit disappeared")
                await consume_evaluation_credit(
                    session,
                    payment=locked_credit,
                    agent_id=agent_id,
                    miner_hotkey=hotkey,
                )
            else:
                assert verified is not None
                await insert_evaluation_payment(
                    session, verified=verified, agent_id=agent_id
                )
    except PaymentReplayedError:
        # A concurrent identical retry may have passed the first lookup before
        # the winning request committed. The transaction context has rolled this
        # request back, so perform one final exact-identity lookup.
        existing = await get_agent_for_payment_proof(
            session,
            block_hash=payment_block_hash,
            extrinsic_index=payment_extrinsic_index,
        )
        if existing and (
            existing.miner_hotkey == hotkey
            and existing.name == name
            and existing.sha256 == sha256
        ):
            assert existing.version is not None
            return UploadAgentResponse(
                agent_id=existing.agent_id,
                version=existing.version,
                status=existing.status,
            )
        raise

    logger.info(
        f"upload accepted hotkey={hotkey} agent_id={agent_id} version={version} "
        f"payment={'credit' if using_credit else 'fresh'} "
        f"block_hash={payment_block_hash}"
    )
    return UploadAgentResponse(
        agent_id=agent_id,
        version=version,
        status=AgentStatus.UPLOADED,
        payment_disposition="credit_consumed" if using_credit else "consumed",
    )


def _verify_signature(hotkey: str, payload: bytes, signature_hex: str) -> bool:
    """Return True iff the signature is a valid sr25519 sig over ``payload``.

    Narrow exception catch on purpose: ``ValueError`` covers malformed
    hex + malformed SS58, ``TypeError`` covers wrong-shape inputs from
    the wallet library. Other exception types are programming bugs that
    should crash the handler so the envelope catch-all returns a 500
    instead of silently reporting "signature did not verify".
    """
    try:
        keypair = bittensor.Keypair(ss58_address=hotkey)
        return bool(keypair.verify(payload, bytes.fromhex(signature_hex)))
    except (ValueError, TypeError):
        return False


async def _read_tar_capped_with_sha(
    upload: UploadFile, max_bytes: int
) -> tuple[bytes, str]:
    """Stream the upload chunk-by-chunk, enforcing size cap + computing sha256.

    Returns the bytes plus the lowercase-hex sha256 of those bytes. The
    accumulating buffer is bounded at ``max_bytes`` so an attacker
    cannot exhaust memory by streaming forever; the cap also keeps the
    happy-path footprint at the documented 2 MB ceiling.

    Raises:
        HTTPException: ``413`` when the streamed body exceeds the cap
            (mapped to ERROR_CODE_TARBALL_TOO_LARGE upstream of this
            function in the route).
    """
    sha = hashlib.sha256()
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await upload.read(_CHUNK_SIZE_BYTES)
        if not chunk:
            break
        size += len(chunk)
        if size > max_bytes:
            raise HTTPException(
                status_code=413, detail=f"tarball exceeds {max_bytes} bytes"
            )
        sha.update(chunk)
        chunks.append(chunk)
    return b"".join(chunks), sha.hexdigest()
