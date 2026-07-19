"""Unit tests for :mod:`ditto.api_server.endpoints.validator`.

These exercise the real endpoints end to end against an in-memory SQLite
database (real queries, real status transitions) with the chain + storage
dependencies mocked. Signatures are produced with a real sr25519 dev
keypair so the signature-verification path runs for real.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import bittensor
import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ditto.api_models.agent_status import AgentStatus
from ditto.api_models.benchmark_progress import (
    BenchmarkProgress,
    benchmark_progress_signing_token,
)
from ditto.api_models.screener import SCREENING_POLICY_VERSION
from ditto.api_models.stack_health import (
    ValidatorStackHealth,
    validator_stack_health_signing_token,
)
from ditto.api_models.system_health import (
    SystemMetrics,
    system_metrics_signing_token,
)
from ditto.api_models.ticket_status import TicketStatus
from ditto.api_models.validator_capabilities import (
    ScorerBenchmarkCapability,
    ValidatorCapabilities,
    ValidatorStackIdentity,
    validator_identity_signing_token,
)
from ditto.api_server.config import ValidatorCompatibilityConfig
from ditto.api_server.dependencies import (
    get_chain_client,
    get_session,
    get_storage_client,
)
from ditto.api_server.endpoints.validator import _heartbeat_signing_message
from ditto.api_server.middleware.error_envelope import (
    ERROR_CODE_AGENT_NOT_EVALUATABLE,
    ERROR_CODE_AGENT_NOT_FOUND,
    ERROR_CODE_VALIDATION,
    ERROR_CODE_VALIDATOR_AUTH,
)
from ditto.chain.models import NeuronInfo
from ditto.db.models import (
    Agent,
    AthReview,
    Base,
    BenchmarkDataset,
    BenchmarkRollout,
    Score,
    ScreenerHeartbeat,
    ValidatorHeartbeat,
    ValidatorTicket,
)

# Real dev keypairs: sign for real so _verify_signature runs end to end. The k=3
# quorum needs three distinct permitted validators before an agent finalizes.
_KEYPAIRS = [
    bittensor.Keypair.create_from_uri(uri) for uri in ("//Alice", "//Bob", "//Charlie")
]
_KEYPAIR = _KEYPAIRS[0]
_VALIDATOR_HOTKEY = _KEYPAIR.ss58_address
# A fourth validator, used only to prove an expired ticket re-opens a slot for a
# validator that was shut out when the k=3 pool was full.
_DAVE = bittensor.Keypair.create_from_uri("//Dave")
_MINER_HOTKEY = "5DhaT8U7LVwnnJNUU8VL1XEipicatoaDVVq7cHo227gogVZm"
_SHA256 = "ab" * 32
_TICKET_DEADLINE = datetime(2030, 1, 1, tzinfo=UTC)


def test_v4_heartbeat_canonical_vector() -> None:
    """Freeze the cross-repository v4 bytes independently of test helpers."""
    agent_id = UUID("11111111-2222-4333-8444-555555555555")
    progress = BenchmarkProgress(
        stage="running_benchmark",
        completed=51,
        total=114,
        ticket_deadline=_TICKET_DEADLINE,
    )
    actual = _heartbeat_signing_message(
        validator_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        software_version="1.2.3",
        protocol_version=4,
        code_digest="ab" * 32,
        state="running_benchmark",
        active_agent_id=agent_id,
        system_metrics=None,
        benchmark_progress=progress,
        timestamp=1784020800,
    )
    assert actual == (
        b"ditto-validator-heartbeat:v4:"
        b"5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY:"
        b"1.2.3:4:"
        b"abababababababababababababababababababababababababababababababab:"
        b"running_benchmark:11111111-2222-4333-8444-555555555555:-:"
        b"running_benchmark,51,114,2030-01-01T00:00:00.000000+00:00:"
        b"1784020800"
    )


@pytest.mark.parametrize(
    ("protocol_version", "domain", "suffix"),
    [
        (1, "v1", "idle:1784020800"),
        (2, "v2", "idle::1784020800"),
        (3, "v3", "idle::-:1784020800"),
        (5, "v4", "idle::-:-:1784020800"),
        (6, "v4", "idle::-:-:1784020800"),
    ],
)
def test_v1_v2_v3_v5_v6_heartbeat_domains_are_frozen(
    protocol_version: int, domain: str, suffix: str
) -> None:
    hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    digest = "ab" * 32
    actual = _heartbeat_signing_message(
        validator_hotkey=hotkey,
        software_version="1.2.3",
        protocol_version=protocol_version,
        code_digest=digest,
        state="idle",
        timestamp=1784020800,
    )
    assert (
        actual
        == (
            f"ditto-validator-heartbeat:{domain}:{hotkey}:1.2.3:"
            f"{protocol_version}:{digest}:{suffix}"
        ).encode()
    )


def test_v7_heartbeat_matches_shared_cross_repo_vector() -> None:
    fixture = json.loads(
        (
            Path(__file__).parents[2] / "contract" / "validator_heartbeat_v7.json"
        ).read_text()
    )
    request = fixture["request"]
    capabilities = ValidatorCapabilities.model_validate(request["capabilities"])
    stack = ValidatorStackIdentity.model_validate(request["stack"])
    actual = _heartbeat_signing_message(
        validator_hotkey=request["validator_hotkey"],
        software_version=request["software_version"],
        protocol_version=request["protocol_version"],
        code_digest=request["code_digest"],
        state=request["state"],
        active_agent_id=request["active_agent_id"],
        system_metrics=request["system_metrics"],
        benchmark_progress=request["benchmark_progress"],
        capabilities=capabilities,
        stack=stack,
        timestamp=request["timestamp"],
    )
    assert actual == fixture["expected_message_utf8"].encode()
    assert actual.hex() == fixture["expected_message_hex"]


def test_v9_heartbeat_matches_shared_cross_repo_vectors() -> None:
    """Both the managed-GHCR and source-Compose v9 vectors verify byte-for-byte."""
    fixtures = json.loads(
        (
            Path(__file__).parents[2] / "contract" / "validator_heartbeat_v9.json"
        ).read_text()
    )
    for name in ("managed", "source"):
        request = fixtures[name]["request"]
        capabilities = ValidatorCapabilities.model_validate_json(
            json.dumps(request["capabilities"])
        )
        stack = ValidatorStackIdentity.model_validate_json(json.dumps(request["stack"]))
        stack_health = ValidatorStackHealth.model_validate_json(
            json.dumps(request["stack_health"])
        )
        actual = _heartbeat_signing_message(
            validator_hotkey=request["validator_hotkey"],
            software_version=request["software_version"],
            protocol_version=request["protocol_version"],
            code_digest=request["code_digest"],
            state=request["state"],
            active_agent_id=request["active_agent_id"],
            system_metrics=request["system_metrics"],
            benchmark_progress=request["benchmark_progress"],
            capabilities=capabilities,
            stack=stack,
            stack_health=stack_health,
            timestamp=request["timestamp"],
        )
        assert actual == fixtures[name]["expected_message_utf8"].encode(), name
        assert (
            hashlib.sha256(actual).hexdigest()
            == fixtures[name]["expected_message_sha256"]
        ), name


def test_optional_scorer_capability_preserves_legacy_v7_token() -> None:
    fixture = json.loads(
        (
            Path(__file__).parents[2] / "contract" / "validator_heartbeat_v7.json"
        ).read_text()
    )
    request = fixture["request"]
    capabilities = ValidatorCapabilities.model_validate(request["capabilities"])
    stack = ValidatorStackIdentity.model_validate(request["stack"])

    assert capabilities.scorer_benchmarks is None
    assert (
        validator_identity_signing_token(capabilities, stack)
        in (fixture["expected_message_utf8"])
    )


def test_scorer_benchmark_capability_is_conservative_unless_fresh_verified() -> None:
    legacy = ScorerBenchmarkCapability(
        status="legacy_v2", supported_bench_versions=(2,)
    )
    assert legacy.supported_bench_versions == (2,)

    with pytest.raises(ValueError, match="may advertise only benchmark v2"):
        ScorerBenchmarkCapability(
            status="identity_mismatch", supported_bench_versions=(2, 3)
        )
    with pytest.raises(ValueError, match="requires observation and identity"):
        ScorerBenchmarkCapability(
            status="fresh_verified", supported_bench_versions=(2, 3)
        )

    verified = ScorerBenchmarkCapability(
        status="fresh_verified",
        supported_bench_versions=(2, 3),
        observed_at=1784020800,
        software_version="1.3.0",
        source_revision="a" * 40,
    )
    assert verified.supported_bench_versions == (2, 3)


def _sign(message: str) -> str:
    return _KEYPAIR.sign(message.encode()).hex()


def _score_payload(
    agent_id: UUID,
    run_id: str = "run_test_1",
    *,
    keypair: bittensor.Keypair = _KEYPAIR,
    **overrides: object,
) -> dict:
    ticket_deadline = overrides.pop("ticket_deadline", _TICKET_DEADLINE)
    assert isinstance(ticket_deadline, datetime)
    report = {
        "run_id": run_id,
        "seed": 8675309,
        "composite": 0.82,
        "tool_mean": 0.88,
        "memory_mean": 0.73,
        "median_ms": 812,
        "n": 30,
        "generated_at": "2026-06-08T12:04:30Z",
        "per_case": [],
    }
    report.update(overrides)
    hotkey = keypair.ss58_address
    lease = ticket_deadline.astimezone(UTC).isoformat(timespec="microseconds")
    signed = (
        f"{hotkey}:{agent_id}:{lease}:{run_id}:{report['composite']!r}:{report['seed']}"
    )
    # CANONICAL ORDER, mirroring _score_signing_message and ditto-subnet:
    #   base : bench_version? : transcript_sha256?
    if report.get("bench_version") is not None:
        signed += f":{report['bench_version']}"
    details = report.get("details")
    transcript = details.get("transcript_sha256") if isinstance(details, dict) else None
    if isinstance(transcript, str) and transcript:
        signed += f":{transcript}"
    return {
        "validator_hotkey": hotkey,
        "ticket_deadline": ticket_deadline.isoformat(),
        "signature": keypair.sign(signed.encode()).hex(),
        "report": report,
    }


def _job_payload(
    keypair: bittensor.Keypair = _KEYPAIR,
    *,
    nonce: UUID | None = None,
    requested_at: datetime | None = None,
) -> dict:
    nonce = nonce or uuid4()
    requested_at = requested_at or datetime.now(UTC)
    requested = requested_at.astimezone(UTC).isoformat(timespec="microseconds")
    signed = f"validator-job:{keypair.ss58_address}:{nonce}:{requested}".encode()
    return {
        "validator_hotkey": keypair.ss58_address,
        "nonce": str(nonce),
        "requested_at": requested_at.isoformat(),
        "signature": keypair.sign(signed).hex(),
    }


def _artifact_headers(
    agent_id: UUID,
    keypair: bittensor.Keypair = _KEYPAIR,
    *,
    nonce: UUID | None = None,
    requested_at: datetime | None = None,
) -> dict[str, str]:
    nonce = nonce or uuid4()
    requested_at = requested_at or datetime.now(UTC)
    requested = requested_at.astimezone(UTC).isoformat(timespec="microseconds")
    signed = (
        f"validator-artifact:v1:{keypair.ss58_address}:{agent_id}:{nonce}:{requested}"
    ).encode()
    return {
        "X-Validator-Hotkey": keypair.ss58_address,
        "X-Validator-Artifact-Nonce": str(nonce),
        "X-Validator-Artifact-Requested-At": requested_at.isoformat(),
        "X-Validator-Artifact-Signature": keypair.sign(signed).hex(),
    }


def _heartbeat_payload(
    *,
    keypair: bittensor.Keypair = _KEYPAIR,
    timestamp: int | None = None,
    code_digest: str = "ab" * 32,
    state: str = "idle",
    protocol_version: int = 1,
    active_agent_id: UUID | None = None,
    system_metrics: dict[str, object] | None = None,
    benchmark_progress: dict[str, object] | None = None,
    capabilities: dict[str, object] | None = None,
    stack: dict[str, object] | None = None,
    stack_health: dict[str, object] | None = None,
) -> dict[str, object]:
    ts = timestamp if timestamp is not None else int(datetime.now(UTC).timestamp())
    hotkey = keypair.ss58_address
    if protocol_version >= 7:
        metrics = (
            SystemMetrics.model_validate(system_metrics)
            if system_metrics is not None
            else None
        )
        progress = (
            BenchmarkProgress.model_validate_json(json.dumps(benchmark_progress))
            if benchmark_progress is not None
            else None
        )
        typed_capabilities = ValidatorCapabilities.model_validate_json(
            json.dumps(capabilities)
        )
        typed_stack = ValidatorStackIdentity.model_validate_json(json.dumps(stack))
        identity_token = validator_identity_signing_token(
            typed_capabilities, typed_stack
        )
        if protocol_version >= 9:
            typed_health = ValidatorStackHealth.model_validate_json(
                json.dumps(stack_health)
            )
            message = (
                f"ditto-validator-heartbeat:v9:{hotkey}:0.1.0:{protocol_version}:"
                f"{code_digest}:{state}:{active_agent_id or ''}:"
                f"{system_metrics_signing_token(metrics)}:"
                f"{benchmark_progress_signing_token(progress)}:"
                f"{identity_token}:"
                f"{validator_stack_health_signing_token(typed_health)}:{ts}"
            )
        else:
            domain = "v8" if protocol_version >= 8 else "v7"
            message = (
                f"ditto-validator-heartbeat:{domain}:{hotkey}:0.1.0:"
                f"{protocol_version}:"
                f"{code_digest}:{state}:{active_agent_id or ''}:"
                f"{system_metrics_signing_token(metrics)}:"
                f"{benchmark_progress_signing_token(progress)}:"
                f"{identity_token}:{ts}"
            )
    elif protocol_version >= 4:
        metrics = (
            SystemMetrics.model_validate(system_metrics)
            if system_metrics is not None
            else None
        )
        progress = (
            BenchmarkProgress.model_validate_json(json.dumps(benchmark_progress))
            if benchmark_progress is not None
            else None
        )
        message = (
            f"ditto-validator-heartbeat:v4:{hotkey}:0.1.0:{protocol_version}:"
            f"{code_digest}:{state}:{active_agent_id or ''}:"
            f"{system_metrics_signing_token(metrics)}:"
            f"{benchmark_progress_signing_token(progress)}:{ts}"
        )
    elif protocol_version >= 3:
        metrics = (
            SystemMetrics.model_validate(system_metrics)
            if system_metrics is not None
            else None
        )
        message = (
            f"ditto-validator-heartbeat:v3:{hotkey}:0.1.0:{protocol_version}:"
            f"{code_digest}:{state}:{active_agent_id or ''}:"
            f"{system_metrics_signing_token(metrics)}:{ts}"
        )
    elif protocol_version >= 2:
        message = (
            f"ditto-validator-heartbeat:v2:{hotkey}:0.1.0:{protocol_version}:"
            f"{code_digest}:{state}:{active_agent_id or ''}:{ts}"
        )
    else:
        message = (
            f"ditto-validator-heartbeat:v1:{hotkey}:0.1.0:1:{code_digest}:{state}:{ts}"
        )
    payload: dict[str, object] = {
        "validator_hotkey": hotkey,
        "software_version": "0.1.0",
        "protocol_version": protocol_version,
        "code_digest": code_digest,
        "state": state,
        "timestamp": ts,
        "signature": keypair.sign(message.encode()).hex(),
    }
    if active_agent_id is not None:
        payload["active_agent_id"] = str(active_agent_id)
    if system_metrics is not None:
        payload["system_metrics"] = system_metrics
    if benchmark_progress is not None:
        payload["benchmark_progress"] = benchmark_progress
    if capabilities is not None:
        payload["capabilities"] = capabilities
    if stack is not None:
        payload["stack"] = stack
    if stack_health is not None:
        payload["stack_health"] = stack_health
    return payload


def _progress(
    stage: str,
    *,
    completed: int | None = None,
    total: int | None = None,
    ticket_deadline: datetime = _TICKET_DEADLINE,
) -> dict[str, object]:
    """Build the exact privacy-safe progress shape accepted by protocol v4."""
    return {
        "stage": stage,
        "completed": completed,
        "total": total,
        "ticket_deadline": ticket_deadline.isoformat(),
    }


async def _score_to_quorum(
    client: httpx.AsyncClient,
    agent_id: UUID,
    *,
    maker: async_sessionmaker[AsyncSession],
    run_id: str = "run_q",
    composite: float = 0.82,
    **overrides: object,
) -> httpx.Response:
    """Seed a ticket for each quorum validator and post one score each (all at
    ``composite``, so the median is ``composite``); return the final response,
    finalized on the last."""
    resp: httpx.Response | None = None
    for i, kp in enumerate(_KEYPAIRS):
        await _seed_ticket(maker, agent_id, keypair=kp)
        resp = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(
                agent_id,
                run_id=f"{run_id}_{i}",
                keypair=kp,
                composite=composite,
                **overrides,
            ),
        )
        assert resp.status_code == 200, resp.text
    assert resp is not None
    return resp


# --- DB + dependency wiring ------------------------------------------------


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(eng.sync_engine, "connect")
    def _enable_fk(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest.fixture
def session_maker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


def _install_db(app: FastAPI, maker: async_sessionmaker[AsyncSession]) -> None:
    async def _session() -> AsyncIterator[AsyncSession]:
        async with maker() as s:
            yield s

    app.dependency_overrides[get_session] = _session


def _install_chain(
    app: FastAPI,
    *,
    permitted: bool = True,
    registered: bool = True,
    extra_keypairs: tuple[bittensor.Keypair, ...] = (),
) -> None:
    neurons = []
    if registered:
        for uid, kp in enumerate((*_KEYPAIRS, *extra_keypairs), start=1):
            neurons.append(
                NeuronInfo(
                    hotkey=kp.ss58_address,
                    coldkey="5GReceiverColdkeyPlaceholderXXXXXXXXXXXXXXXXXXX",
                    uid=uid,
                    stake=1000.0,
                    validator_permit=permitted,
                )
            )

    async def _chain() -> MagicMock:
        c = MagicMock()
        c.get_recent_neurons = AsyncMock(return_value=neurons)
        return c

    app.dependency_overrides[get_chain_client] = _chain


def _install_storage(app: FastAPI) -> MagicMock:
    storage = MagicMock()
    storage.presigned_get_url = AsyncMock(
        return_value="https://signed.example/ditto-agents/x.tar.gz?sig=1"
    )

    async def _storage() -> MagicMock:
        return storage

    app.dependency_overrides[get_storage_client] = _storage
    return storage


async def _seed_agent(
    maker: async_sessionmaker[AsyncSession],
    *,
    status: AgentStatus,
    name: str = "alpha-agent",
    created_at: datetime | None = None,
    agent_id: UUID | None = None,
    miner_hotkey: str = _MINER_HOTKEY,
    sha256: str = _SHA256,
    size_bytes: int = 524288,
    screening_policy_version: int = SCREENING_POLICY_VERSION,
) -> UUID:
    aid = agent_id or uuid4()
    async with maker() as s, s.begin():
        s.add(
            Agent(
                agent_id=aid,
                miner_hotkey=miner_hotkey,
                name=name,
                sha256=sha256,
                size_bytes=size_bytes,
                status=status,
                screening_policy_version=screening_policy_version,
                created_at=created_at or datetime.now(UTC),
            )
        )
    return aid


async def _seed_ticket(
    maker: async_sessionmaker[AsyncSession],
    agent_id: UUID,
    *,
    keypair: bittensor.Keypair = _KEYPAIR,
    deadline: datetime = _TICKET_DEADLINE,
) -> None:
    """Seat (or re-open) an issued ticket for a specific (agent, validator) so a
    score against that agent is accepted by the k=3 gate. Upserts so a test can
    simulate the platform re-issuing a ticket to the same validator."""
    async with maker() as s, s.begin():
        existing = await s.get(ValidatorTicket, (agent_id, 2, keypair.ss58_address))
        if existing is None:
            s.add(
                ValidatorTicket(
                    agent_id=agent_id,
                    validator_hotkey=keypair.ss58_address,
                    status=TicketStatus.ISSUED,
                    issued_at=datetime.now(UTC) - timedelta(seconds=1),
                    deadline=deadline,
                )
            )
        else:
            existing.status = TicketStatus.ISSUED
            existing.deadline = deadline


async def _seed_validator_heartbeat(
    maker: async_sessionmaker[AsyncSession],
    *,
    keypair: bittensor.Keypair = _KEYPAIR,
    software_version: str = "0.7.0",
    protocol_version: int = 4,
    seen_at: datetime | None = None,
    capabilities: dict[str, object] | None = None,
    stack: dict[str, object] | None = None,
    state: str = "polling",
) -> None:
    now = seen_at or datetime.now(UTC)
    async with maker() as s, s.begin():
        s.add(
            ValidatorHeartbeat(
                validator_hotkey=keypair.ss58_address,
                software_version=software_version,
                protocol_version=protocol_version,
                code_digest="ab" * 32,
                state=state,
                active_agent_id=None,
                first_seen_at=now,
                system_metrics=None,
                benchmark_progress=None,
                benchmark_progress_reported=False,
                benchmark_progress_agent_id=None,
                capabilities=capabilities,
                stack=stack,
                reported_at=now,
                seen_at=now,
                signature="ab" * 64,
            )
        )


_AUTH_HEADER = {"X-Validator-Hotkey": _VALIDATOR_HOTKEY}
_SYSTEM_METRICS = {
    "collected_at": 0,
    "cpu_percent": 15,
    "memory_percent": 40,
    "disk_percent": 55,
    "docker": {
        "status": "healthy",
        "running_containers": 4,
        "unhealthy_containers": 0,
    },
}

_V7_CAPABILITIES: dict[str, object] = {
    "screened_images": True,
    "require_screened_image": False,
    "source_build_fallback": True,
    "full_stack_managed": False,
    "stack_updater": False,
    "sandbox_egress_restricted": True,
    "executor_isolation": "privileged_dind",
}
_V7_COMPONENTS: dict[str, object] = {
    name: {
        "source_revision": f"{index:x}" * 40,
        "version": f"1.2.{index}",
        "provenance": "committed_pin",
    }
    for index, name in enumerate(
        (
            "ditto_subnet",
            "dittobench_api",
            "sandbox_docker",
            "model_relay",
            "pylon",
            "ollama",
        ),
        start=1,
    )
}
_V7_STACK: dict[str, object] = {
    "mode": "source",
    "compose_schema": 1,
    "release_descriptor_digest": None,
    "components": _V7_COMPONENTS,
}


_V9_SCORER: dict[str, object] = {
    "status": "fresh_verified",
    "supported_bench_versions": [2, 3],
    "observed_at": 1_784_020_800,
    "software_version": "1.2.2",
    "source_revision": "2" * 40,
}
_V9_CAPABILITIES: dict[str, object] = {
    **_V7_CAPABILITIES,
    "scorer_benchmarks": _V9_SCORER,
}
_V9_STACK_HEALTH: dict[str, object] = {
    "ditto_subnet": {
        "health": "healthy",
        "required": True,
        "observed_at": 1_784_020_800,
        "ready": True,
        "observed_identity": {"version": "1.2.3"},
    },
    "dittobench_api": {
        "health": "healthy",
        "required": True,
        "observed_at": 1_784_020_800,
        "ready": True,
        "observed_identity": {"source_revision": "2" * 40, "version": "1.2.2"},
    },
    "sandbox_docker": {
        "health": "unknown",
        "required": True,
    },
    "model_relay": {
        "health": "identity_mismatch",
        "required": True,
        "observed_at": 1_784_020_700,
        "ready": True,
        "model_ready": True,
        "observed_identity": {"source_revision": "c" * 40},
    },
    "pylon": {
        "health": "degraded",
        "required": True,
        "observed_at": 1_784_020_700,
        "ready": False,
    },
    "ollama": {
        "health": "unreachable",
        "required": True,
        "observed_at": 1_784_017_200,
    },
}


def _screener_heartbeat_payload(
    *, timestamp: int, system_metrics: dict[str, object]
) -> dict[str, object]:
    metrics = SystemMetrics.model_validate(system_metrics)
    message = (
        "ditto-screener-heartbeat:v1:"
        f"{_KEYPAIR.ss58_address}:0.4.2:1:{SCREENING_POLICY_VERSION}:polling::"
        f"{system_metrics_signing_token(metrics)}:{timestamp}"
    )
    return {
        "screener_hotkey": _KEYPAIR.ss58_address,
        "software_version": "0.4.2",
        "protocol_version": 1,
        "policy_version": SCREENING_POLICY_VERSION,
        "state": "polling",
        "timestamp": timestamp,
        "signature": _KEYPAIR.sign(message.encode()).hex(),
        "system_metrics": system_metrics,
    }


# --- Queue -----------------------------------------------------------------


class TestHeartbeat:
    async def test_v8_requires_signed_scorer_capability_and_v7_rejects_it(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        scorer = {
            "status": "fresh_verified",
            "supported_bench_versions": [2, 3],
            "observed_at": int(datetime.now(UTC).timestamp()),
            "software_version": "1.2.2",
            "source_revision": "2" * 40,
        }
        capabilities = {**_V7_CAPABILITIES, "scorer_benchmarks": scorer}

        rejected_v7 = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=_heartbeat_payload(
                protocol_version=7,
                capabilities=capabilities,
                stack=_V7_STACK,
            ),
        )
        assert rejected_v7.status_code == 422

        accepted_v8 = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=_heartbeat_payload(
                protocol_version=8,
                capabilities=capabilities,
                stack=_V7_STACK,
            ),
        )
        assert accepted_v8.status_code == 200, accepted_v8.text
        async with session_maker() as session:
            row = await session.get(ValidatorHeartbeat, _VALIDATOR_HOTKEY)
            assert row is not None
            assert row.protocol_version == 8
            assert row.capabilities is not None
            assert row.capabilities["scorer_benchmarks"][
                "supported_bench_versions"
            ] == [2, 3]

    async def test_v9_persists_and_publishes_component_stack_health(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)

        accepted = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=_heartbeat_payload(
                protocol_version=9,
                capabilities=_V9_CAPABILITIES,
                stack=_V7_STACK,
                stack_health=_V9_STACK_HEALTH,
            ),
        )
        assert accepted.status_code == 200, accepted.text

        expected_health = ValidatorStackHealth.model_validate_json(
            json.dumps(_V9_STACK_HEALTH)
        ).model_dump(mode="json", exclude_none=True)
        async with session_maker() as session:
            row = await session.get(ValidatorHeartbeat, _VALIDATOR_HOTKEY)
            assert row is not None
            assert row.protocol_version == 9
            assert row.stack_health == expected_health

        public = (await client.get("/api/v1/public/validators")).json()["validators"][0]
        health = public["stack_health"]
        assert health is not None
        assert health["ditto_subnet"]["health"] == "healthy"
        assert health["sandbox_docker"]["health"] == "unknown"
        assert health["model_relay"]["health"] == "identity_mismatch"
        assert health["model_relay"]["observed_identity"]["source_revision"] == "c" * 40
        assert health["pylon"]["health"] == "degraded"
        assert health["ollama"]["health"] == "unreachable"
        # Probe URLs and host identity have no schema slot; belt-and-braces
        # regression that nothing network-shaped leaked into the public view.
        assert "://" not in json.dumps(health)

    async def test_v9_requires_stack_health_and_v8_rejects_it(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)

        missing = _heartbeat_payload(
            protocol_version=9,
            capabilities=_V9_CAPABILITIES,
            stack=_V7_STACK,
            stack_health=_V9_STACK_HEALTH,
        )
        missing.pop("stack_health")
        assert (
            await client.post(
                "/api/v1/validator/heartbeat", headers=_AUTH_HEADER, json=missing
            )
        ).status_code == 422

        downgraded = _heartbeat_payload(
            protocol_version=8,
            capabilities=_V9_CAPABILITIES,
            stack=_V7_STACK,
        )
        downgraded["stack_health"] = _V9_STACK_HEALTH
        assert (
            await client.post(
                "/api/v1/validator/heartbeat", headers=_AUTH_HEADER, json=downgraded
            )
        ).status_code == 422

        tampered = _heartbeat_payload(
            protocol_version=9,
            capabilities=_V9_CAPABILITIES,
            stack=_V7_STACK,
            stack_health=_V9_STACK_HEALTH,
        )
        upgraded_health = json.loads(json.dumps(_V9_STACK_HEALTH))
        upgraded_health["ollama"] = {
            "health": "healthy",
            "required": True,
            "observed_at": 1_784_017_200,
            "ready": True,
            "model_ready": True,
        }
        tampered["stack_health"] = upgraded_health
        assert (
            await client.post(
                "/api/v1/validator/heartbeat", headers=_AUTH_HEADER, json=tampered
            )
        ).status_code == 401

    async def test_malformed_stored_stack_health_is_omitted_publicly(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        accepted = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=_heartbeat_payload(
                protocol_version=9,
                capabilities=_V9_CAPABILITIES,
                stack=_V7_STACK,
                stack_health=_V9_STACK_HEALTH,
            ),
        )
        assert accepted.status_code == 200, accepted.text
        async with session_maker() as session, session.begin():
            row = await session.get(ValidatorHeartbeat, _VALIDATOR_HOTKEY)
            assert row is not None
            row.stack_health = {"hostname": "validator-vm", "logs": ["leak"]}

        public = (await client.get("/api/v1/public/validators")).json()["validators"][0]
        assert public["stack_health"] is None

    async def test_v7_persists_and_publishes_typed_capabilities(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        payload = _heartbeat_payload(
            protocol_version=7,
            capabilities=_V7_CAPABILITIES,
            stack=_V7_STACK,
        )

        response = await client.post(
            "/api/v1/validator/heartbeat", headers=_AUTH_HEADER, json=payload
        )

        assert response.status_code == 200, response.text
        async with session_maker() as session:
            row = await session.get(ValidatorHeartbeat, _VALIDATOR_HOTKEY)
            assert row is not None
            assert row.capabilities == _V7_CAPABILITIES
            expected_stack = ValidatorStackIdentity.model_validate(
                _V7_STACK
            ).model_dump(mode="json")
            assert row.stack == expected_stack
        public = (await client.get("/api/v1/public/validators")).json()["validators"][0]
        assert public["capabilities"] == _V7_CAPABILITIES
        assert public["stack"] == expected_stack
        assert "signature" not in public

    async def test_v7_rejects_missing_contradictory_and_tampered_identity(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)

        missing = _heartbeat_payload()
        missing["protocol_version"] = 7
        assert (
            await client.post(
                "/api/v1/validator/heartbeat", headers=_AUTH_HEADER, json=missing
            )
        ).status_code == 422

        contradictory = {**_V7_CAPABILITIES, "source_build_fallback": False}
        payload = _heartbeat_payload(
            protocol_version=7,
            capabilities=_V7_CAPABILITIES,
            stack=_V7_STACK,
        )
        payload["capabilities"] = contradictory
        rejected = await client.post(
            "/api/v1/validator/heartbeat", headers=_AUTH_HEADER, json=payload
        )
        assert rejected.status_code == 422

        tampered = _heartbeat_payload(
            protocol_version=7,
            capabilities=_V7_CAPABILITIES,
            stack=_V7_STACK,
        )
        tampered_stack = dict(_V7_STACK)
        tampered_stack["compose_schema"] = 2
        tampered["stack"] = tampered_stack
        rejected = await client.post(
            "/api/v1/validator/heartbeat", headers=_AUTH_HEADER, json=tampered
        )
        assert rejected.status_code == 401

    async def test_records_signed_build_and_publishes_status(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        payload = _heartbeat_payload()
        response = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=payload,
        )
        assert response.status_code == 200, response.text
        assert response.json()["accepted"] is True

        async with session_maker() as session:
            row = await session.get(ValidatorHeartbeat, _VALIDATOR_HOTKEY)
            assert row is not None
            assert row.software_version == "0.1.0"
            assert row.code_digest == "ab" * 32
            assert row.state == "idle"

        public = await client.get("/api/v1/public/validators")
        assert public.status_code == 200
        body = public.json()
        assert body["reported_count"] == 1
        assert body["online_count"] == 1
        assert body["validators"][0]["validator_hotkey"] == _VALIDATOR_HOTKEY
        assert body["validators"][0]["state"] == "idle"
        assert body["validators"][0]["online"] is True

        replay = await client.post(
            "/api/v1/validator/heartbeat", headers=_AUTH_HEADER, json=payload
        )
        assert replay.status_code == 200
        assert replay.json()["accepted"] is False
        assert replay.json()["seen_at"] == response.json()["seen_at"]

    async def test_v2_reports_current_agent_publicly(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await _seed_ticket(session_maker, agent_id)
        _install_db(app, session_maker)
        _install_chain(app)
        payload = _heartbeat_payload(
            protocol_version=2,
            state="running_benchmark",
            active_agent_id=agent_id,
        )

        response = await client.post(
            "/api/v1/validator/heartbeat", headers=_AUTH_HEADER, json=payload
        )
        assert response.status_code == 200, response.text
        public = (await client.get("/api/v1/public/validators")).json()
        assert public["validators"][0]["active_agent_id"] == str(agent_id)
        active_benchmark = public["validators"][0]["active_benchmark"]
        started_at = datetime.fromisoformat(
            active_benchmark.pop("started_at").replace("Z", "+00:00")
        )
        assert started_at.tzinfo == UTC
        assert active_benchmark == {
            "agent_id": str(agent_id),
            "agent_name": "alpha-agent",
            "stage": None,
            "completed_checks": None,
            "total_checks": None,
            "percent": None,
        }

    async def test_operations_snapshot_is_atomic_and_synchronized(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await _seed_ticket(session_maker, agent_id)
        _install_db(app, session_maker)
        _install_chain(app)
        heartbeat = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=_heartbeat_payload(
                protocol_version=2,
                state="running_benchmark",
                active_agent_id=agent_id,
            ),
        )
        assert heartbeat.status_code == 200, heartbeat.text

        response = await client.get("/api/v1/public/operations")
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "public, max-age=10"
        snapshot = response.json()
        assert snapshot["generated_at"] == snapshot["activity"]["generated_at"]
        assert snapshot["generated_at"] == snapshot["validators"]["generated_at"]
        validator = snapshot["validators"]["validators"][0]
        assert validator["assignment_state"] == "synchronized"
        assert validator["assigned_agent_id"] == str(agent_id)
        assert validator["reported_agent_id"] == str(agent_id)
        assert validator["active_agent_id"] == str(agent_id)
        activity = snapshot["activity"]["entries"][0]
        assert activity["status"] == "evaluating"
        assert activity["active_benchmarks"][0]["agent_id"] == str(agent_id)

    async def test_operations_snapshot_surfaces_different_reported_agent(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        assigned_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        reported_id = await _seed_agent(
            session_maker, status=AgentStatus.EVALUATING, name="reported-agent"
        )
        await _seed_ticket(session_maker, assigned_id)
        await _seed_validator_heartbeat(session_maker, protocol_version=2)
        now = datetime.now(UTC)
        async with session_maker() as session, session.begin():
            heartbeat = await session.get(ValidatorHeartbeat, _VALIDATOR_HOTKEY)
            assert heartbeat is not None
            heartbeat.state = "running_benchmark"
            heartbeat.active_agent_id = reported_id
            heartbeat.reported_at = now
            heartbeat.seen_at = now
        _install_db(app, session_maker)

        snapshot = (await client.get("/api/v1/public/operations")).json()
        validator = snapshot["validators"]["validators"][0]
        assert validator["assignment_state"] == "heartbeat_mismatch"
        assert validator["assigned_agent_id"] == str(assigned_id)
        assert validator["assigned_agent_name"] == "alpha-agent"
        assert validator["reported_agent_id"] == str(reported_id)
        assert validator["active_agent_id"] is None
        assigned = next(
            entry
            for entry in snapshot["activity"]["entries"]
            if entry["agent_id"] == str(assigned_id)
        )
        assert assigned["status"] == "waiting_validator"
        assert assigned["active_benchmarks"] == []

    async def test_operations_snapshot_surfaces_stale_heartbeat_assignment(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await _seed_ticket(session_maker, agent_id)
        stale = datetime.now(UTC) - timedelta(minutes=10)
        await _seed_validator_heartbeat(
            session_maker, protocol_version=2, seen_at=stale
        )
        async with session_maker() as session, session.begin():
            heartbeat = await session.get(ValidatorHeartbeat, _VALIDATOR_HOTKEY)
            assert heartbeat is not None
            heartbeat.state = "running_benchmark"
            heartbeat.active_agent_id = agent_id
        _install_db(app, session_maker)

        snapshot = (await client.get("/api/v1/public/operations")).json()
        validator = snapshot["validators"]["validators"][0]
        assert validator["assignment_state"] == "heartbeat_stale"
        assert validator["availability"] == "stale"
        assert validator["assigned_agent_id"] == str(agent_id)
        assert validator["reported_agent_id"] == str(agent_id)
        assert validator["active_agent_id"] is None
        assert snapshot["activity"]["entries"][0]["status"] == "waiting_validator"

    @pytest.mark.e2e
    async def test_v4_progresses_public_lifecycle_and_terminal_score_clears_it(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        """Fake build -> run -> finalize -> submit against one real ticket."""
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await _seed_ticket(session_maker, agent_id)
        _install_db(app, session_maker)
        _install_chain(app)
        timestamp = int(datetime.now(UTC).timestamp())
        stages = [
            _progress("preparing"),
            _progress("building_harness"),
            _progress("starting_harness"),
            _progress("running_benchmark", completed=0, total=114),
            _progress("running_benchmark", completed=51, total=114),
            _progress("finalizing", completed=114, total=114),
            _progress("submitting_result", completed=114, total=114),
        ]

        for offset, progress in enumerate(stages):
            response = await client.post(
                "/api/v1/validator/heartbeat",
                headers=_AUTH_HEADER,
                json=_heartbeat_payload(
                    protocol_version=4,
                    timestamp=timestamp + offset,
                    state="running_benchmark",
                    active_agent_id=agent_id,
                    benchmark_progress=progress,
                ),
            )
            assert response.status_code == 200, response.text
            public = (await client.get("/api/v1/public/validators")).json()
            shown = public["validators"][0]["active_benchmark"]
            assert shown["stage"] == progress["stage"]
            assert shown["agent_id"] == str(agent_id)

            if progress["stage"] == "running_benchmark" and progress["completed"] == 51:
                started_at = datetime.fromisoformat(
                    shown.pop("started_at").replace("Z", "+00:00")
                )
                assert started_at.tzinfo == UTC
                assert shown == {
                    "agent_id": str(agent_id),
                    "agent_name": "alpha-agent",
                    "stage": "running_benchmark",
                    "completed_checks": 51,
                    "total_checks": 114,
                    "percent": 45,
                }
            if progress["stage"] in {"finalizing", "submitting_result"}:
                assert shown["percent"] == 95
                assert shown["completed_checks"] == shown["total_checks"] == 114

        pipeline = (
            await client.get(f"/api/v1/public/agent/{agent_id}/pipeline")
        ).json()
        attempt = pipeline["validation_attempts"][0]
        assert attempt["deadline"] is not None
        assert attempt["actively_running"] is True
        assert attempt["benchmark_progress"]["stage"] == "submitting_result"

        scored = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id),
        )
        assert scored.status_code == 200, scored.text
        fleet = (await client.get("/api/v1/public/validators")).json()
        assert fleet["validators"][0]["active_agent_id"] is None
        assert fleet["validators"][0]["active_benchmark"] is None
        activity = (await client.get("/api/v1/public/activity")).json()
        assert activity["entries"][0]["active_benchmarks"] == []

    async def test_v4_rejects_regression_omission_and_signature_tampering(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await _seed_ticket(session_maker, agent_id)
        _install_db(app, session_maker)
        _install_chain(app)
        timestamp = int(datetime.now(UTC).timestamp())

        initial = _heartbeat_payload(
            protocol_version=4,
            timestamp=timestamp,
            state="running_benchmark",
            active_agent_id=agent_id,
            benchmark_progress=_progress("running_benchmark", completed=51, total=114),
        )
        assert (
            await client.post(
                "/api/v1/validator/heartbeat", headers=_AUTH_HEADER, json=initial
            )
        ).status_code == 200

        regressions = [
            _progress("starting_harness"),
            _progress("running_benchmark", completed=40, total=114),
            _progress("running_benchmark", completed=52, total=120),
        ]
        for offset, progress in enumerate(regressions, start=1):
            rejected = await client.post(
                "/api/v1/validator/heartbeat",
                headers=_AUTH_HEADER,
                json=_heartbeat_payload(
                    protocol_version=4,
                    timestamp=timestamp + offset,
                    state="running_benchmark",
                    active_agent_id=agent_id,
                    benchmark_progress=progress,
                ),
            )
            assert rejected.status_code == 409, rejected.text

        omitted = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=_heartbeat_payload(
                protocol_version=4,
                timestamp=timestamp + 4,
                state="running_benchmark",
                active_agent_id=agent_id,
            ),
        )
        assert omitted.status_code == 200, omitted.text
        public_unknown = (await client.get("/api/v1/public/validators")).json()
        active_benchmark = public_unknown["validators"][0]["active_benchmark"]
        started_at = datetime.fromisoformat(
            active_benchmark.pop("started_at").replace("Z", "+00:00")
        )
        assert started_at.tzinfo == UTC
        assert active_benchmark == {
            "agent_id": str(agent_id),
            "agent_name": "alpha-agent",
            "stage": None,
            "completed_checks": None,
            "total_checks": None,
            "percent": None,
        }

        downgraded = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=_heartbeat_payload(
                protocol_version=3,
                timestamp=timestamp + 5,
                state="running_benchmark",
                active_agent_id=agent_id,
            ),
        )
        assert downgraded.status_code == 200, downgraded.text
        public_unknown = (await client.get("/api/v1/public/validators")).json()
        assert public_unknown["validators"][0]["active_benchmark"]["stage"] is None

        lower_after_downgrade = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=_heartbeat_payload(
                protocol_version=4,
                timestamp=timestamp + 6,
                state="running_benchmark",
                active_agent_id=agent_id,
                benchmark_progress=_progress(
                    "running_benchmark", completed=50, total=114
                ),
            ),
        )
        assert lower_after_downgrade.status_code == 409

        tampered = _heartbeat_payload(
            protocol_version=4,
            timestamp=timestamp + 7,
            state="running_benchmark",
            active_agent_id=agent_id,
            benchmark_progress=_progress("running_benchmark", completed=52, total=114),
        )
        assert isinstance(tampered["benchmark_progress"], dict)
        tampered["benchmark_progress"]["completed"] = 53
        rejected = await client.post(
            "/api/v1/validator/heartbeat", headers=_AUTH_HEADER, json=tampered
        )
        assert rejected.status_code == 401

        cleared = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=_heartbeat_payload(protocol_version=4, timestamp=timestamp + 8),
        )
        assert cleared.status_code == 200, cleared.text
        fleet = (await client.get("/api/v1/public/validators")).json()
        assert fleet["validators"][0]["active_benchmark"] is None

        lower_after_idle = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=_heartbeat_payload(
                protocol_version=4,
                timestamp=timestamp + 9,
                state="running_benchmark",
                active_agent_id=agent_id,
                benchmark_progress=_progress(
                    "running_benchmark", completed=1, total=114
                ),
            ),
        )
        assert lower_after_idle.status_code == 409

        other_agent_id = await _seed_agent(
            session_maker, status=AgentStatus.EVALUATING, name="new-agent"
        )
        async with session_maker() as session, session.begin():
            previous = await session.get(
                ValidatorTicket, (agent_id, 2, _VALIDATOR_HOTKEY)
            )
            assert previous is not None
            previous.status = TicketStatus.SCORED
        await _seed_ticket(session_maker, other_agent_id)
        different_agent = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=_heartbeat_payload(
                protocol_version=4,
                timestamp=timestamp + 10,
                state="running_benchmark",
                active_agent_id=other_agent_id,
                benchmark_progress=_progress(
                    "running_benchmark", completed=1, total=114
                ),
            ),
        )
        assert different_agent.status_code == 200, different_agent.text

    async def test_v4_failed_retrying_explicitly_restarts_at_preparing(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await _seed_ticket(session_maker, agent_id)
        _install_db(app, session_maker)
        _install_chain(app)
        timestamp = int(datetime.now(UTC).timestamp())

        sequence = [
            _progress("running_benchmark", completed=51, total=114),
            _progress("failed_retrying", completed=51, total=114),
        ]
        for offset, progress in enumerate(sequence):
            response = await client.post(
                "/api/v1/validator/heartbeat",
                headers=_AUTH_HEADER,
                json=_heartbeat_payload(
                    protocol_version=4,
                    timestamp=timestamp + offset,
                    state="running_benchmark",
                    active_agent_id=agent_id,
                    benchmark_progress=progress,
                ),
            )
            assert response.status_code == 200, response.text

        same_lease_restart = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=_heartbeat_payload(
                protocol_version=4,
                timestamp=timestamp + 2,
                state="running_benchmark",
                active_agent_id=agent_id,
                benchmark_progress=_progress("preparing"),
            ),
        )
        assert same_lease_restart.status_code == 200, same_lease_restart.text

        resumed = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=_heartbeat_payload(
                protocol_version=4,
                timestamp=timestamp + 3,
                state="running_benchmark",
                active_agent_id=agent_id,
                benchmark_progress=_progress(
                    "running_benchmark", completed=1, total=114
                ),
            ),
        )
        assert resumed.status_code == 200, resumed.text

        new_deadline = _TICKET_DEADLINE + timedelta(hours=1)
        await _seed_ticket(session_maker, agent_id, deadline=new_deadline)
        restarted = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=_heartbeat_payload(
                protocol_version=4,
                timestamp=timestamp + 4,
                state="running_benchmark",
                active_agent_id=agent_id,
                benchmark_progress=_progress("preparing", ticket_deadline=new_deadline),
            ),
        )
        assert restarted.status_code == 200, restarted.text

    @pytest.mark.parametrize("status", [AgentStatus.SCORED, AgentStatus.LIVE])
    async def test_v4_preserves_rollout_progress_for_scored_and_live_agents(
        self,
        status: AgentStatus,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=status)
        await _seed_ticket(session_maker, agent_id)
        _install_db(app, session_maker)
        _install_chain(app)

        response = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=_heartbeat_payload(
                protocol_version=4,
                state="running_benchmark",
                active_agent_id=agent_id,
                benchmark_progress=_progress(
                    "running_benchmark", completed=51, total=114
                ),
            ),
        )

        assert response.status_code == 200, response.text
        fleet = (await client.get("/api/v1/public/validators")).json()
        validator = fleet["validators"][0]
        assert validator["active_agent_id"] == str(agent_id)
        assert validator["active_benchmark"]["stage"] == "running_benchmark"

        pipeline = (
            await client.get(f"/api/v1/public/agent/{agent_id}/pipeline")
        ).json()
        attempt = pipeline["validation_attempts"][0]
        assert attempt["actively_running"] is True
        assert attempt["benchmark_progress"]["completed_checks"] == 51

    async def test_v4_drops_progress_for_non_scoreable_agent_with_live_ticket(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        await _seed_ticket(session_maker, agent_id)
        _install_db(app, session_maker)
        _install_chain(app)

        response = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=_heartbeat_payload(
                protocol_version=4,
                state="running_benchmark",
                active_agent_id=agent_id,
                benchmark_progress=_progress("preparing"),
            ),
        )

        assert response.status_code == 200, response.text
        fleet = (await client.get("/api/v1/public/validators")).json()
        validator = fleet["validators"][0]
        assert validator["active_agent_id"] is None
        assert validator["active_benchmark"] is None

    async def test_v4_drops_progress_without_matching_live_ticket_but_stays_live(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        timestamp = int(datetime.now(UTC).timestamp())

        missing = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=_heartbeat_payload(
                protocol_version=4,
                timestamp=timestamp,
                state="running_benchmark",
                active_agent_id=agent_id,
                benchmark_progress=_progress("preparing"),
            ),
        )
        assert missing.status_code == 200, missing.text
        assert missing.json()["accepted"] is True

        async with session_maker() as session:
            heartbeat = await session.get(ValidatorHeartbeat, _VALIDATOR_HOTKEY)
            assert heartbeat is not None
            assert heartbeat.state == "running_benchmark"
            assert heartbeat.active_agent_id is None
            assert heartbeat.benchmark_progress_reported is False

        await _seed_ticket(session_maker, agent_id)
        wrong_deadline = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=_heartbeat_payload(
                protocol_version=4,
                timestamp=timestamp + 1,
                state="running_benchmark",
                active_agent_id=agent_id,
                benchmark_progress=_progress(
                    "preparing", ticket_deadline=_TICKET_DEADLINE + timedelta(days=1)
                ),
            ),
        )
        assert wrong_deadline.status_code == 200, wrong_deadline.text
        assert wrong_deadline.json()["accepted"] is True

        async with session_maker() as session:
            heartbeat = await session.get(ValidatorHeartbeat, _VALIDATOR_HOTKEY)
            assert heartbeat is not None
            assert heartbeat.active_agent_id is None
            assert heartbeat.benchmark_progress_reported is False
            ticket = await session.get(
                ValidatorTicket, (agent_id, 2, _VALIDATOR_HOTKEY)
            )
            assert ticket is not None
            assert ticket.status == TicketStatus.ISSUED
            assert ticket.deadline.replace(tzinfo=UTC) == _TICKET_DEADLINE

    async def test_v4_expired_ticket_progress_cannot_block_heartbeat_recovery(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        deadline = datetime.now(UTC) - timedelta(minutes=1)
        await _seed_ticket(session_maker, agent_id, deadline=deadline)
        _install_db(app, session_maker)
        _install_chain(app)
        timestamp = int(datetime.now(UTC).timestamp())

        recovered = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=_heartbeat_payload(
                protocol_version=4,
                timestamp=timestamp,
                state="running_benchmark",
                active_agent_id=agent_id,
                benchmark_progress=_progress(
                    "running_benchmark",
                    completed=51,
                    total=114,
                    ticket_deadline=deadline,
                ),
            ),
        )

        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["accepted"] is True
        fleet = (await client.get("/api/v1/public/validators")).json()
        validator = fleet["validators"][0]
        assert validator["availability"] == "available"
        assert validator["online"] is True
        assert validator["active_agent_id"] is None
        assert validator["active_benchmark"] is None

        async with session_maker() as session:
            ticket = await session.get(
                ValidatorTicket, (agent_id, 2, _VALIDATOR_HOTKEY)
            )
            heartbeat = await session.get(ValidatorHeartbeat, _VALIDATOR_HOTKEY)
            assert ticket is not None
            assert ticket.status == TicketStatus.ISSUED
            assert ticket.deadline.replace(tzinfo=UTC) == deadline
            assert heartbeat is not None
            assert heartbeat.seen_at is not None
            assert heartbeat.active_agent_id is None
            assert heartbeat.benchmark_progress is None
            assert heartbeat.benchmark_progress_reported is False

    async def test_v3_binds_coarse_metrics_and_public_response_is_redacted(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        timestamp = int(datetime.now(UTC).timestamp())
        metrics = {**_SYSTEM_METRICS, "collected_at": timestamp}
        payload = _heartbeat_payload(
            protocol_version=3, timestamp=timestamp, system_metrics=metrics
        )
        response = await client.post(
            "/api/v1/validator/heartbeat", headers=_AUTH_HEADER, json=payload
        )
        assert response.status_code == 200, response.text

        public = (await client.get("/api/v1/public/validators")).json()
        entry = public["validators"][0]
        assert entry["availability"] == "available"
        assert entry["health"] == "healthy"
        assert entry["first_seen_at"] is not None
        assert entry["system_metrics"] == {
            "cpu_percent": 15,
            "memory_percent": 40,
            "disk_percent": 55,
            "docker_status": "healthy",
            "running_containers": 4,
            "unhealthy_containers": 0,
        }
        assert "signature" not in entry
        assert "code_digest" not in entry
        for forbidden in (
            "hostname",
            "ip",
            "instance_id",
            "path",
            "container_name",
            "image_digest",
        ):
            assert forbidden not in str(entry).lower()

    async def test_v3_rejects_tampered_or_malformed_metrics(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        timestamp = int(datetime.now(UTC).timestamp())
        metrics = {**_SYSTEM_METRICS, "collected_at": timestamp}
        payload = _heartbeat_payload(
            protocol_version=3, timestamp=timestamp, system_metrics=metrics
        )
        payload["system_metrics"]["memory_percent"] = 90  # type: ignore[index]
        tampered = await client.post(
            "/api/v1/validator/heartbeat", headers=_AUTH_HEADER, json=payload
        )
        assert tampered.status_code == 401

        malformed = _heartbeat_payload(
            protocol_version=3, timestamp=timestamp, system_metrics=metrics
        )
        malformed["system_metrics"]["hostname"] = "private"  # type: ignore[index]
        rejected = await client.post(
            "/api/v1/validator/heartbeat", headers=_AUTH_HEADER, json=malformed
        )
        assert rejected.status_code == 422

    async def test_heartbeat_payload_size_is_bounded(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        response = await client.post(
            "/api/v1/validator/heartbeat",
            headers={**_AUTH_HEADER, "Content-Length": "4097"},
            json=_heartbeat_payload(),
        )
        assert response.status_code == 413

        payload = json.dumps(_heartbeat_payload())
        response = await client.post(
            "/api/v1/validator/heartbeat",
            headers={**_AUTH_HEADER, "Content-Type": "application/json"},
            content=(" " * 4097) + payload,
        )
        assert response.status_code == 413

    async def test_rejects_stale_heartbeat(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        stale = int(datetime.now(UTC).timestamp()) - 301
        response = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=_heartbeat_payload(timestamp=stale),
        )
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_VALIDATOR_AUTH

    @pytest.mark.e2e
    async def test_mixed_fleet_and_malformed_telemetry(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        """Exercise reporter ingestion through both public fleet views."""
        _install_db(app, session_maker)
        _install_chain(app)
        now = datetime.now(UTC)
        timestamp = int(now.timestamp())
        metrics = {**_SYSTEM_METRICS, "collected_at": timestamp}

        old_validator = await client.post(
            "/api/v1/validator/heartbeat",
            headers={"X-Validator-Hotkey": _KEYPAIRS[1].ss58_address},
            json=_heartbeat_payload(keypair=_KEYPAIRS[1], protocol_version=2),
        )
        assert old_validator.status_code == 200, old_validator.text

        metric_validator = await client.post(
            "/api/v1/validator/heartbeat",
            headers={"X-Validator-Hotkey": _KEYPAIRS[2].ss58_address},
            json=_heartbeat_payload(
                keypair=_KEYPAIRS[2],
                protocol_version=3,
                timestamp=timestamp,
                system_metrics=metrics,
            ),
        )
        assert metric_validator.status_code == 200, metric_validator.text

        screener_headers = {
            "Authorization": "Bearer test-screener-token-at-least-32-characters",
            "X-Screener-Hotkey": _KEYPAIR.ss58_address,
        }
        healthy_screener = await client.post(
            "/api/v1/screener/heartbeat",
            headers=screener_headers,
            json=_screener_heartbeat_payload(
                timestamp=timestamp, system_metrics=metrics
            ),
        )
        assert healthy_screener.status_code == 200, healthy_screener.text

        stale_at = now - timedelta(minutes=10)
        async with session_maker() as session, session.begin():
            session.add(
                ScreenerHeartbeat(
                    screener_hotkey=_DAVE.ss58_address,
                    software_version="0.4.1",
                    protocol_version=1,
                    policy_version=SCREENING_POLICY_VERSION,
                    state="polling",
                    active_agent_id=None,
                    first_seen_at=stale_at - timedelta(hours=2),
                    system_metrics=metrics,
                    reported_at=stale_at,
                    seen_at=stale_at,
                    signature="ab" * 64,
                )
            )

        malformed = _heartbeat_payload(
            keypair=_KEYPAIRS[2],
            protocol_version=3,
            timestamp=timestamp,
            system_metrics=metrics,
        )
        malformed_metrics = malformed["system_metrics"]
        assert isinstance(malformed_metrics, dict)
        malformed_metrics["hostname"] = "must-never-be-accepted"
        rejected = await client.post(
            "/api/v1/validator/heartbeat",
            headers={"X-Validator-Hotkey": _KEYPAIRS[2].ss58_address},
            json=malformed,
        )
        assert rejected.status_code == 422

        validators = (await client.get("/api/v1/public/validators")).json()
        assert validators["reported_count"] == 2
        old = next(v for v in validators["validators"] if v["protocol_version"] == 2)
        current = next(
            v for v in validators["validators"] if v["protocol_version"] == 3
        )
        assert old["availability"] == "available"
        assert old["health"] == "unknown"
        assert old["system_metrics"] is None
        assert current["availability"] == "available"
        assert current["health"] == "healthy"

        screeners = (await client.get("/api/v1/public/screeners")).json()
        assert screeners["reported_count"] == 2
        available = next(s for s in screeners["screeners"] if s["online"])
        stale = next(s for s in screeners["screeners"] if not s["online"])
        assert available["availability"] == "available"
        assert available["health"] == "healthy"
        assert stale["availability"] == "stale"

    async def test_rejects_tampered_digest(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        payload = _heartbeat_payload()
        payload["code_digest"] = "cd" * 32
        response = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=payload,
        )
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_VALIDATOR_AUTH

    async def test_rejects_tampered_runtime_state(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        payload = _heartbeat_payload(state="idle")
        payload["state"] = "running_benchmark"
        response = await client.post(
            "/api/v1/validator/heartbeat",
            headers=_AUTH_HEADER,
            json=payload,
        )
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_VALIDATOR_AUTH


class TestArtifact:
    async def test_returns_presigned_url_and_sha(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        storage = _install_storage(app)

        response = await client.get(
            f"/api/v1/validator/agent/{agent_id}/artifact",
            headers=_artifact_headers(agent_id),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["agent_id"] == str(agent_id)
        assert body["sha256"] == _SHA256
        assert body["download_url"].startswith("https://")
        assert body["screened_image_url"] is None
        assert body["screened_image_sha256"] is None
        storage.presigned_get_url.assert_awaited_once()
        assert (
            storage.presigned_get_url.await_args.kwargs["key"]
            == f"{agent_id}/agent.tar.gz"
        )

    async def test_returns_verified_screened_image_fields(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        upload_id = uuid4()
        async with session_maker() as session, session.begin():
            agent = await session.get(Agent, agent_id)
            assert agent is not None
            agent.screened_image_sha256 = "12" * 32
            agent.screened_image_size_bytes = 123
            agent.screened_image_id = "sha256:" + "34" * 32
            agent.screened_image_ref = f"ditto-screen/{agent_id}:latest"
            agent.screened_image_upload_id = upload_id
            agent.screened_image_verified_at = datetime.now(UTC)
        _install_db(app, session_maker)
        _install_chain(app)
        storage = _install_storage(app)

        response = await client.get(
            f"/api/v1/validator/agent/{agent_id}/artifact",
            headers=_artifact_headers(agent_id),
        )

        assert response.status_code == 200
        body = response.json()
        assert body["screened_image_url"].startswith("https://")
        assert body["screened_image_sha256"] == "12" * 32
        assert body["screened_image_size_bytes"] == 123
        assert body["screened_image_id"] == "sha256:" + "34" * 32
        assert body["screened_image_ref"] == f"ditto-screen/{agent_id}:latest"
        assert storage.presigned_get_url.await_args_list[1].kwargs["key"] == (
            f"{agent_id}/screened-images/{upload_id}.tar"
        )

    async def test_unknown_agent_returns_404(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        _install_storage(app)
        agent_id = uuid4()
        response = await client.get(
            f"/api/v1/validator/agent/{agent_id}/artifact",
            headers=_artifact_headers(agent_id),
        )
        assert response.status_code == 404
        assert response.json()["error_code"] == ERROR_CODE_AGENT_NOT_FOUND

    async def test_public_validator_identity_without_signature_is_rejected(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        _install_storage(app)

        response = await client.get(
            f"/api/v1/validator/agent/{agent_id}/artifact", headers=_AUTH_HEADER
        )

        assert response.status_code == 401

    async def test_replayed_artifact_proof_is_rejected(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        _install_storage(app)
        headers = _artifact_headers(agent_id)

        first = await client.get(
            f"/api/v1/validator/agent/{agent_id}/artifact", headers=headers
        )
        replay = await client.get(
            f"/api/v1/validator/agent/{agent_id}/artifact", headers=headers
        )

        assert first.status_code == 200
        assert replay.status_code == 409


# --- Submit score ----------------------------------------------------------


class TestRequestJob:
    @staticmethod
    def _v8_capabilities() -> dict[str, object]:
        return {
            **_V7_CAPABILITIES,
            "scorer_benchmarks": {
                "status": "fresh_verified",
                "supported_bench_versions": [2, 3],
                "observed_at": int(datetime.now(UTC).timestamp()),
                "software_version": "1.2.2",
                "source_revision": "2" * 40,
            },
        }

    @staticmethod
    async def _activate_v3(
        session_maker: async_sessionmaker[AsyncSession], agent_id: UUID
    ) -> None:
        now = datetime.now(UTC)
        async with session_maker() as session, session.begin():
            agent = await session.get(Agent, agent_id)
            assert agent is not None
            agent.screening_policy_version = 9
            agent.screened_image_sha256 = "12" * 32
            agent.screened_image_size_bytes = 123
            agent.screened_image_id = "sha256:" + "34" * 32
            agent.screened_image_ref = f"ditto-screen/{agent_id}:latest"
            agent.screened_image_upload_id = uuid4()
            agent.screened_image_verified_at = now
            session.add(
                BenchmarkRollout(
                    rollout_id=uuid4(),
                    from_version=2,
                    desired_version=3,
                    status="activated",
                    cohort_size=5,
                    created_at=now,
                    activated_at=now,
                )
            )
            session.add(
                BenchmarkDataset(
                    agent_id=agent_id,
                    bench_version=3,
                    seed=8675309,
                    sha256="cd" * 32,
                    run_size="full",
                )
            )

    @staticmethod
    def _enable_compatibility_gate(app: FastAPI) -> None:
        app.state.config = replace(
            app.state.config,
            validator_compatibility=ValidatorCompatibilityConfig(
                minimum_software_version="0.7.0",
                minimum_protocol_version=4,
                heartbeat_max_age_seconds=300,
            ),
        )

    async def test_requires_heartbeat_before_issuing_work(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        self._enable_compatibility_gate(app)

        response = await client.post(
            "/api/v1/validator/job", headers=_AUTH_HEADER, json=_job_payload()
        )

        assert response.status_code == 428
        assert "heartbeat required" in response.json()["message"]

    @pytest.mark.parametrize(
        ("software_version", "protocol_version", "expected_detail"),
        [
            ("0.6.9", 4, "software '0.6.9' is below required 0.7.0"),
            ("0.7.0", 3, "protocol 3 is below required 4"),
        ],
    )
    async def test_requires_supported_validator_release(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
        software_version: str,
        protocol_version: int,
        expected_detail: str,
    ) -> None:
        await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await _seed_validator_heartbeat(
            session_maker,
            software_version=software_version,
            protocol_version=protocol_version,
        )
        _install_db(app, session_maker)
        _install_chain(app)
        self._enable_compatibility_gate(app)

        response = await client.post(
            "/api/v1/validator/job", headers=_AUTH_HEADER, json=_job_payload()
        )

        assert response.status_code == 426
        assert expected_detail in response.json()["message"]
        assert "update ditto-subnet" in response.json()["message"]

    async def test_supported_validator_receives_work(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await _seed_validator_heartbeat(session_maker)
        _install_db(app, session_maker)
        _install_chain(app)
        self._enable_compatibility_gate(app)

        response = await client.post(
            "/api/v1/validator/job", headers=_AUTH_HEADER, json=_job_payload()
        )

        assert response.status_code == 200
        assert response.json()["agent_id"] == str(agent_id)

    async def test_after_activation_v2_only_gets_no_new_v3_but_resumes_v2(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        v3_agent = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await self._activate_v3(session_maker, v3_agent)
        await _seed_validator_heartbeat(
            session_maker,
            protocol_version=7,
            capabilities=_V7_CAPABILITIES,
            stack=_V7_STACK,
        )
        _install_db(app, session_maker)
        _install_chain(app)

        no_new_work = await client.post(
            "/api/v1/validator/job", headers=_AUTH_HEADER, json=_job_payload()
        )
        assert no_new_work.status_code == 204

        legacy_agent = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await _seed_ticket(session_maker, legacy_agent)
        resumed = await client.post(
            "/api/v1/validator/job", headers=_AUTH_HEADER, json=_job_payload()
        )
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()["agent_id"] == str(legacy_agent)
        assert resumed.json()["bench_version"] == 2

    async def test_after_activation_new_submission_finalizes_on_three_v3_scores(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await self._activate_v3(session_maker, agent_id)
        capabilities = self._v8_capabilities()
        for keypair in _KEYPAIRS:
            await _seed_validator_heartbeat(
                session_maker,
                keypair=keypair,
                protocol_version=8,
                capabilities=capabilities,
                stack=_V7_STACK,
            )
        _install_db(app, session_maker)
        _install_chain(app, extra_keypairs=tuple(_KEYPAIRS[1:]))

        for index, keypair in enumerate(_KEYPAIRS, start=1):
            job = await client.post(
                "/api/v1/validator/job",
                headers={"X-Validator-Hotkey": keypair.ss58_address},
                json=_job_payload(keypair),
            )
            assert job.status_code == 200, job.text
            assert job.json()["bench_version"] == 3
            assert job.json()["minimum_screening_policy_version"] == 9
            assert job.json()["requires_screened_image"] is True
            deadline = datetime.fromisoformat(job.json()["deadline"])
            score = await client.post(
                f"/api/v1/validator/agent/{agent_id}/score",
                json=_score_payload(
                    agent_id,
                    run_id=f"v3-{index}",
                    keypair=keypair,
                    ticket_deadline=deadline,
                    bench_version=3,
                    n=114,
                    details={"bench_version": 3},
                ),
            )
            assert score.status_code == 200, score.text
            expected = AgentStatus.SCORED if index == 3 else AgentStatus.EVALUATING
            assert score.json()["status"] == expected

        async with session_maker() as session:
            scores = (
                (
                    await session.execute(
                        select(Score).where(
                            Score.agent_id == agent_id, Score.bench_version == 3
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert len(scores) == 3

    async def test_v7_screened_only_does_not_claim_source_only_work(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        capabilities = {
            **_V7_CAPABILITIES,
            "require_screened_image": True,
            "source_build_fallback": False,
        }
        await _seed_validator_heartbeat(
            session_maker,
            protocol_version=7,
            capabilities=capabilities,
            stack=_V7_STACK,
        )
        _install_db(app, session_maker)
        _install_chain(app)

        response = await client.post(
            "/api/v1/validator/job", headers=_AUTH_HEADER, json=_job_payload()
        )

        assert response.status_code == 204

    @pytest.mark.parametrize(
        ("state", "expected_status"),
        [
            ("polling", TicketStatus.EXPIRED),
            ("running_benchmark", TicketStatus.ISSUED),
        ],
    )
    async def test_v7_screened_only_does_not_resume_source_only_live_ticket(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
        state: str,
        expected_status: TicketStatus,
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        capabilities = {
            **_V7_CAPABILITIES,
            "require_screened_image": True,
            "source_build_fallback": False,
        }
        await _seed_validator_heartbeat(
            session_maker,
            protocol_version=7,
            capabilities=capabilities,
            stack=_V7_STACK,
            state=state,
        )
        now = datetime.now(UTC)
        async with session_maker() as session, session.begin():
            session.add(
                ValidatorTicket(
                    agent_id=agent_id,
                    bench_version=2,
                    validator_hotkey=_VALIDATOR_HOTKEY,
                    status=TicketStatus.ISSUED,
                    issued_at=now,
                    deadline=now + timedelta(minutes=90),
                    attempt_count=1,
                    manual_retry_grants=0,
                )
            )
        _install_db(app, session_maker)
        _install_chain(app)

        response = await client.post(
            "/api/v1/validator/job", headers=_AUTH_HEADER, json=_job_payload()
        )

        assert response.status_code == 204
        async with session_maker() as session:
            ticket = await session.get(
                ValidatorTicket, (agent_id, 2, _VALIDATOR_HOTKEY)
            )
            assert ticket is not None
            assert ticket.status == expected_status

    async def test_stale_heartbeat_cannot_claim_work(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await _seed_validator_heartbeat(
            session_maker, seen_at=datetime.now(UTC) - timedelta(minutes=6)
        )
        _install_db(app, session_maker)
        _install_chain(app)
        self._enable_compatibility_gate(app)

        response = await client.post(
            "/api/v1/validator/job", headers=_AUTH_HEADER, json=_job_payload()
        )

        assert response.status_code == 428
        assert "heartbeat is stale" in response.json()["message"]

    async def test_issues_ticket_for_evaluating_agent(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        before = datetime.now(UTC)
        resp = await client.post(
            "/api/v1/validator/job", headers=_AUTH_HEADER, json=_job_payload()
        )
        after = datetime.now(UTC)
        assert resp.status_code == 200
        body = resp.json()
        assert body["agent_id"] == str(agent_id)
        deadline = datetime.fromisoformat(body["deadline"].replace("Z", "+00:00"))
        assert before + timedelta(minutes=90) <= deadline
        assert deadline <= after + timedelta(minutes=90)

    async def test_no_work_returns_204(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        resp = await client.post(
            "/api/v1/validator/job", headers=_AUTH_HEADER, json=_job_payload()
        )
        assert resp.status_code == 204

    async def test_caps_at_quorum_across_validators(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app, extra_keypairs=(_DAVE,))
        # Three distinct validators each get the single agent (fills the pool).
        for kp in _KEYPAIRS:
            r = await client.post(
                "/api/v1/validator/job",
                headers={"X-Validator-Hotkey": kp.ss58_address},
                json=_job_payload(kp),
            )
            assert r.status_code == 200
        # A further request finds no open slot -> no job.
        r = await client.post(
            "/api/v1/validator/job",
            headers={"X-Validator-Hotkey": _DAVE.ss58_address},
            json=_job_payload(_DAVE),
        )
        assert r.status_code == 204

    async def test_unpermitted_validator_returns_401(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app, permitted=False)
        resp = await client.post(
            "/api/v1/validator/job", headers=_AUTH_HEADER, json=_job_payload()
        )
        assert resp.status_code == 401

    async def test_cannot_claim_by_naming_another_permitted_hotkey(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        forged = _job_payload(_KEYPAIRS[1])
        forged["validator_hotkey"] = _VALIDATOR_HOTKEY
        resp = await client.post(
            "/api/v1/validator/job", headers=_AUTH_HEADER, json=forged
        )
        assert resp.status_code == 401

    async def test_replayed_job_claim_is_rejected(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        claim = _job_payload()
        first = await client.post(
            "/api/v1/validator/job", headers=_AUTH_HEADER, json=claim
        )
        replay = await client.post(
            "/api/v1/validator/job", headers=_AUTH_HEADER, json=claim
        )
        assert first.status_code == 200
        assert replay.status_code == 409

    async def test_stale_job_claim_is_rejected(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        stale = _job_payload(requested_at=datetime.now(UTC) - timedelta(minutes=3))
        resp = await client.post(
            "/api/v1/validator/job", headers=_AUTH_HEADER, json=stale
        )
        assert resp.status_code == 409


class TestSubmitScore:
    async def test_rejects_score_until_current_screening_policy_passes(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            screening_policy_version=0,
        )
        _install_db(app, session_maker)
        _install_chain(app)
        await _seed_ticket(session_maker, agent_id)
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id),
        )
        assert response.status_code == 409
        async with session_maker() as session:
            assert await session.get(Score, (agent_id, 2, _VALIDATOR_HOTKEY)) is None

    async def test_records_score_and_finalizes(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)

        # A single below-quorum score records the row but keeps the agent
        # provisional (evaluating) — no finalization until the k=3 quorum.
        await _seed_ticket(session_maker, agent_id, keypair=_KEYPAIRS[0])
        first = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id, keypair=_KEYPAIRS[0]),
        )
        assert first.status_code == 200
        assert first.json()["status"] == AgentStatus.EVALUATING

        # The quorum-th score finalizes it on the median composite.
        response = await _score_to_quorum(
            client, agent_id, maker=session_maker, composite=0.82
        )
        assert response.status_code == 200
        body = response.json()
        assert body["agent_id"] == str(agent_id)
        assert body["status"] == AgentStatus.SCORED
        assert body["accepted"] is True

        # A scores row landed and the agent transitioned.
        async with session_maker() as s:
            score = await s.get(Score, (agent_id, 2, _VALIDATOR_HOTKEY))
            assert score is not None
            assert score.composite == pytest.approx(0.82)
            agent = await s.get(Agent, agent_id)
            assert agent is not None
            assert agent.status == AgentStatus.SCORED

    async def test_finalize_writes_verifiable_audit_chain(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        from ditto.db.queries.audit import (
            EVENT_FINALIZED,
            EVENT_SCORE,
            list_audit_entries,
            verify_audit_chain,
        )

        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)

        # One below-quorum score, then the k=3 quorum (which re-scores validator 0
        # with a fresh ticket): 4 append-only score events + 1 finalize.
        await _seed_ticket(session_maker, agent_id, keypair=_KEYPAIRS[0])
        await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id, keypair=_KEYPAIRS[0]),
        )
        await _score_to_quorum(client, agent_id, maker=session_maker, composite=0.82)

        async with session_maker() as s:
            entries = await list_audit_entries(s, limit=1000)
        # Append-only: the re-score is its own entry even though the table upserts.
        score_entries = [e for e in entries if e.event == EVENT_SCORE]
        finalized = [e for e in entries if e.event == EVENT_FINALIZED]
        assert len(score_entries) == 4
        assert len(finalized) == 1
        assert entries[-1].event == EVENT_FINALIZED
        # The finalize entry carries the median + quorum + scoring validators.
        fin = finalized[0].payload
        assert fin["median_composite"] == pytest.approx(0.82)
        assert fin["quorum"] == 3
        assert fin["score_count"] == 3
        assert len(fin["validator_hotkeys"]) == 3
        assert fin["status"] == AgentStatus.SCORED.value
        # The whole chain replays and verifies (tamper-evident end to end).
        assert verify_audit_chain(entries) is True

    async def test_stamps_current_bench_version_when_omitted(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # A report that omits bench_version (as the default payload does) must be
        # stamped with the current version so it is never recorded as legacy.

        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)

        await _seed_ticket(session_maker, agent_id)
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id),
        )
        assert response.status_code == 200

        async with session_maker() as s:
            score = await s.get(Score, (agent_id, 2, _VALIDATOR_HOTKEY))
            assert score is not None
            assert score.details is not None
            # The payload now declares bench_version explicitly, so stamping
            # preserves it rather than overwriting with CURRENT: a report that
            # genuinely ran an older contract stays honestly labelled, and the
            # label matches the row's key.
            assert score.details["bench_version"] == 2
            assert score.details["ticket_deadline"] == (
                _TICKET_DEADLINE.isoformat(timespec="microseconds")
            )

    async def test_overwrites_advisory_detail_with_ticket_bench_version(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # The locked ticket, not unsigned scorer details, owns benchmark identity.
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)

        await _seed_ticket(session_maker, agent_id)
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id, details={"bench_version": 1}),
        )
        assert response.status_code == 200

        async with session_maker() as s:
            score = await s.get(Score, (agent_id, 2, _VALIDATOR_HOTKEY))
            assert score is not None
            assert score.details is not None
            assert score.details["bench_version"] == 2

    async def test_one_ticket_one_score_no_rescore(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # One ticket, one score: a validator's first score is accepted and
        # consumes its ticket; a second score without a fresh ticket is rejected
        # (409), so a validator cannot re-roll for a better number.
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)

        await _seed_ticket(session_maker, agent_id)
        r1 = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id, run_id="run_a", composite=0.5),
        )
        assert r1.status_code == 200
        # Ticket spent: the re-score has no open ticket.
        r2 = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id, run_id="run_b", composite=0.9),
        )
        assert r2.status_code == 409

        async with session_maker() as s:
            from ditto.db.queries.scores import list_scores_for_agent

            scores = await list_scores_for_agent(s, agent_id=agent_id)
            assert len(scores) == 1
            assert scores[0].run_id == "run_a"  # the first (only) score stands
            assert scores[0].composite == pytest.approx(0.5)

    async def test_superseded_ticket_lease_rejects_late_score(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        old_deadline = _TICKET_DEADLINE
        new_deadline = old_deadline + timedelta(hours=1)
        await _seed_ticket(session_maker, agent_id, deadline=new_deadline)

        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id, ticket_deadline=old_deadline),
        )

        assert response.status_code == 409
        assert "no open scoring ticket" in response.json()["message"]

    async def test_bad_signature_returns_401(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)

        payload = _score_payload(agent_id)
        payload["signature"] = "ab" * 64  # well-formed but wrong
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score", json=payload
        )
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_VALIDATOR_AUTH

    async def test_unpermitted_validator_returns_401(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app, permitted=False)
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id),
        )
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_VALIDATOR_AUTH

    async def test_unknown_agent_returns_404(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        aid = uuid4()
        response = await client.post(
            f"/api/v1/validator/agent/{aid}/score", json=_score_payload(aid)
        )
        assert response.status_code == 404
        assert response.json()["error_code"] == ERROR_CODE_AGENT_NOT_FOUND

    async def test_non_scoreable_status_returns_409(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.UPLOADED)
        _install_db(app, session_maker)
        _install_chain(app)
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id),
        )
        assert response.status_code == 409
        assert response.json()["error_code"] == ERROR_CODE_AGENT_NOT_EVALUATABLE

    async def test_re_score_live_agent_keeps_live(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.LIVE)
        _install_db(app, session_maker)
        _install_chain(app)
        await _seed_ticket(session_maker, agent_id)
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id),
        )
        assert response.status_code == 200
        assert response.json()["status"] == AgentStatus.LIVE

    async def test_out_of_range_composite_returns_422(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id, composite=1.5),
        )
        assert response.status_code == 422
        assert response.json()["error_code"] == ERROR_CODE_VALIDATION

    async def test_out_of_range_seed_returns_422(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # A seed outside signed int64 would 500 at the BigInteger insert; it must
        # be a clean 422 before signing/DB work.
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(agent_id, seed=2**63),
        )
        assert response.status_code == 422
        assert response.json()["error_code"] == ERROR_CODE_VALIDATION

    async def test_cross_agent_replay_rejected(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # A signature valid for agent A must not be accepted for agent B: the
        # signed payload binds the agent id.
        agent_a = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        agent_b = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        payload = _score_payload(agent_a)  # signed for A
        response = await client.post(
            f"/api/v1/validator/agent/{agent_b}/score", json=payload
        )
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_VALIDATOR_AUTH

    async def test_tampered_composite_rejected(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # The composite is signed: altering it after signing invalidates the sig.
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        payload = _score_payload(agent_id, composite=0.50)
        payload["report"]["composite"] = 0.99  # tamper post-signing
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score", json=payload
        )
        assert response.status_code == 401
        assert response.json()["error_code"] == ERROR_CODE_VALIDATOR_AUTH


_MINER_B = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


class TestAntiCopyGate:
    """The score-write path holds a suspected copy in ath_pending_review."""

    async def _score(
        self,
        client: httpx.AsyncClient,
        agent_id: UUID,
        *,
        maker: async_sessionmaker[AsyncSession],
        run_id: str,
        composite: float,
    ) -> httpx.Response:
        # Score to the k=3 quorum so the agent finalizes and the gate runs on
        # the median (= composite, since all three validators post the same).
        return await _score_to_quorum(
            client, agent_id, maker=maker, run_id=run_id, composite=composite
        )

    async def test_exact_copy_is_held(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        # Incumbent scores + becomes eligible.
        incumbent = await _seed_agent(
            session_maker, status=AgentStatus.EVALUATING, sha256="cc" * 32
        )
        await self._score(
            client, incumbent, maker=session_maker, run_id="run_inc", composite=0.80
        )
        # A byte-identical resubmission from another miner.
        copy = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            miner_hotkey=_MINER_B,
            sha256="cc" * 32,
        )
        resp = await self._score(
            client, copy, maker=session_maker, run_id="run_copy", composite=0.80
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == AgentStatus.ATH_PENDING_REVIEW

        async with session_maker() as s:
            held = await s.get(Agent, copy)
            review = await s.scalar(select(AthReview).where(AthReview.agent_id == copy))
            assert held is not None
            assert review is not None
            assert held.status == AgentStatus.ATH_PENDING_REVIEW
            assert held.duplicate_of == incumbent
            assert "sha256" in (held.review_reason or "")
            assert review.algorithm_provenance == {
                "snapshot": "score-finalization",
                "algorithm_version": "reference-aware-v2",
                "canonical_reference_revision": (
                    "959cd69a1a8d3b0defbfb8296518adb7d4f17c14"
                ),
                "reference_corpus_id": (
                    "21dc06cd72aafefb56d0e89e8b3127280dda249ae26cb649ee855185121e9ce6"
                ),
                "reference_exclusion_mode": "starter-kit-mainline-history",
                "backfilled": False,
                "opened_at_source": "agent_finalized_audit",
            }

    async def test_near_dup_dethroner_is_held(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        incumbent = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            sha256="aa" * 32,
            size_bytes=500000,
        )
        await self._score(
            client, incumbent, maker=session_maker, run_id="run_inc", composite=0.80
        )
        # Different bytes, near-identical size, beats incumbent by a hair.
        tweaked = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            miner_hotkey=_MINER_B,
            sha256="bb" * 32,
            size_bytes=500100,
        )
        resp = await self._score(
            client, tweaked, maker=session_maker, run_id="run_tweak", composite=0.805
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == AgentStatus.ATH_PENDING_REVIEW

    async def test_later_scored_upload_is_not_original_for_earlier_submission(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        earlier_time = datetime(2026, 6, 8, 12, 0, tzinfo=UTC)
        later = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            miner_hotkey=_MINER_B,
            sha256="aa" * 32,
            size_bytes=500000,
            created_at=earlier_time + timedelta(hours=1),
        )
        await self._score(
            client, later, maker=session_maker, run_id="run_later", composite=0.80
        )
        earlier = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            sha256="aa" * 32,
            size_bytes=500100,
            created_at=earlier_time,
        )
        resp = await self._score(
            client, earlier, maker=session_maker, run_id="run_earlier", composite=0.805
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == AgentStatus.SCORED

    async def test_genuine_improvement_not_held(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        incumbent = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            sha256="aa" * 32,
            size_bytes=500000,
        )
        await self._score(
            client, incumbent, maker=session_maker, run_id="run_inc", composite=0.80
        )
        better = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            miner_hotkey=_MINER_B,
            sha256="bb" * 32,
            size_bytes=700000,
        )
        resp = await self._score(
            client, better, maker=session_maker, run_id="run_better", composite=0.92
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == AgentStatus.SCORED

    async def test_rescore_of_held_agent_stays_held_no_409(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        incumbent = await _seed_agent(
            session_maker, status=AgentStatus.EVALUATING, sha256="cc" * 32
        )
        await self._score(
            client, incumbent, maker=session_maker, run_id="run_inc", composite=0.80
        )
        copy = await _seed_agent(
            session_maker,
            status=AgentStatus.EVALUATING,
            miner_hotkey=_MINER_B,
            sha256="cc" * 32,
        )
        await self._score(
            client, copy, maker=session_maker, run_id="run_copy", composite=0.80
        )
        # Re-scoring a held agent must not 409 and must not un-hold it.
        resp = await self._score(
            client, copy, maker=session_maker, run_id="run_copy2", composite=0.81
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == AgentStatus.ATH_PENDING_REVIEW


class TestPublicMirror:
    """The finalize hook mirrors the run record to the public bucket."""

    async def test_finalize_publishes_when_public_bucket_configured(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        storage = _install_storage(app)
        storage.public_bucket = "ditto-public"
        storage.put_object = AsyncMock()
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await _score_to_quorum(
            client, agent_id, maker=session_maker, run_id="run_pub", composite=0.5
        )
        storage.put_object.assert_awaited_once()
        kwargs = storage.put_object.await_args.kwargs
        assert kwargs["bucket"] == "ditto-public"
        assert kwargs["key"] == f"scored/{agent_id}.json"
        record = json.loads(kwargs["body"])
        assert record["median_composite"] == 0.5
        assert len(record["scores"]) == 3
        assert all(sc["signature"] for sc in record["scores"])
        assert record["status"] == AgentStatus.SCORED.value

    async def test_finalize_skips_publish_when_unconfigured(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        storage = _install_storage(app)
        storage.public_bucket = None
        storage.put_object = AsyncMock()
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await _score_to_quorum(
            client, agent_id, maker=session_maker, run_id="run_nopub", composite=0.5
        )
        storage.put_object.assert_not_awaited()


class TestTranscriptPublication:
    """Offline-reproducibility hardening (v3 review finding 3): the transcript
    digest is bound into the score signature, and the transcript upload path
    only ever stores bytes that hash to a digest a signed score declared."""

    _TRANSCRIPT = b'{"run_id":"run_t_0","cases":[{"case_id":"a","response":{}}]}'
    _digest = hashlib.sha256(_TRANSCRIPT).hexdigest()

    async def test_score_signature_binds_transcript_digest(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        _install_storage(app)
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await _seed_ticket(session_maker, agent_id)
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(
                agent_id,
                run_id="run_t_0",
                details={"transcript_sha256": self._digest},
            ),
        )
        assert response.status_code == 200, response.text

    async def test_transcript_digest_outside_signature_rejected(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # A report that declares a digest the signature does not cover must be
        # rejected: otherwise the artifact binding would be spoofable.
        _install_db(app, session_maker)
        _install_chain(app)
        _install_storage(app)
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await _seed_ticket(session_maker, agent_id)
        payload = _score_payload(agent_id, run_id="run_t_0")
        payload["report"]["details"] = {"transcript_sha256": self._digest}
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score", json=payload
        )
        assert response.status_code == 401

    async def _record_score_with_transcript(
        self,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
        agent_id: UUID,
    ) -> None:
        await _seed_ticket(session_maker, agent_id)
        response = await client.post(
            f"/api/v1/validator/agent/{agent_id}/score",
            json=_score_payload(
                agent_id,
                run_id="run_t_0",
                details={"transcript_sha256": self._digest},
            ),
        )
        assert response.status_code == 200, response.text

    async def test_submit_transcript_stores_content_addressed(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        storage = _install_storage(app)
        storage.public_bucket = "ditto-public"
        storage.put_object = AsyncMock()
        storage.object_exists = AsyncMock(return_value=False)
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await self._record_score_with_transcript(client, session_maker, agent_id)

        response = await client.put(
            f"/api/v1/validator/agent/{agent_id}/transcript/run_t_0",
            content=self._TRANSCRIPT,
            headers={"X-Validator-Hotkey": _VALIDATOR_HOTKEY},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["stored"] is True
        assert body["transcript_sha256"] == self._digest
        storage.put_object.assert_awaited_once()
        kwargs = storage.put_object.await_args.kwargs
        assert kwargs["bucket"] == "ditto-public"
        assert kwargs["key"] == f"transcripts/{self._digest}.json"
        assert kwargs["body"] == self._TRANSCRIPT

        # Idempotent: a re-upload of an existing object writes nothing new.
        storage.object_exists = AsyncMock(return_value=True)
        response = await client.put(
            f"/api/v1/validator/agent/{agent_id}/transcript/run_t_0",
            content=self._TRANSCRIPT,
            headers={"X-Validator-Hotkey": _VALIDATOR_HOTKEY},
        )
        assert response.status_code == 200
        storage.put_object.assert_awaited_once()  # still exactly one write

    async def test_submit_transcript_digest_mismatch_rejected(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        storage = _install_storage(app)
        storage.public_bucket = "ditto-public"
        storage.put_object = AsyncMock()
        storage.object_exists = AsyncMock(return_value=False)
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await self._record_score_with_transcript(client, session_maker, agent_id)

        response = await client.put(
            f"/api/v1/validator/agent/{agent_id}/transcript/run_t_0",
            content=b'{"tampered": true}',
            headers={"X-Validator-Hotkey": _VALIDATOR_HOTKEY},
        )
        assert response.status_code == 409
        storage.put_object.assert_not_awaited()

    async def test_submit_transcript_without_score_rejected(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        storage = _install_storage(app)
        storage.public_bucket = "ditto-public"
        storage.put_object = AsyncMock()
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)

        response = await client.put(
            f"/api/v1/validator/agent/{agent_id}/transcript/run_t_0",
            content=self._TRANSCRIPT,
            headers={"X-Validator-Hotkey": _VALIDATOR_HOTKEY},
        )
        assert response.status_code == 409
        storage.put_object.assert_not_awaited()

    async def test_publish_record_carries_transcript_refs(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        _install_db(app, session_maker)
        _install_chain(app)
        storage = _install_storage(app)
        storage.public_bucket = "ditto-public"
        storage.put_object = AsyncMock()
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        await _score_to_quorum(
            client,
            agent_id,
            maker=session_maker,
            run_id="run_t",
            composite=0.5,
            details={"transcript_sha256": self._digest},
        )
        kwargs = storage.put_object.await_args.kwargs
        record = json.loads(kwargs["body"])
        for sc in record["scores"]:
            assert sc["transcript_sha256"] == self._digest
            assert sc["transcript_key"] == f"transcripts/{self._digest}.json"


class TestMultiValidatorConsensus:
    """The k=3 consensus semantics the decentralized design promises: the
    canonical score is the MEDIAN of the (differing) independent validator
    composites, the full per-validator record is exposed publicly, and an
    expired ticket re-opens the slot so a shut-out validator can pick the agent
    up. These exercise consensus correctness end to end, complementing the
    all-equal-composite quorum tests above."""

    async def test_finalizes_on_median_of_differing_scores(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        # Three independent validators disagree. The platform must finalize on the
        # MEDIAN (0.82), never the mean (0.7067) or any single validator's number.
        composites = {_KEYPAIRS[0]: 0.40, _KEYPAIRS[1]: 0.82, _KEYPAIRS[2]: 0.90}
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)

        last: httpx.Response | None = None
        for i, (kp, comp) in enumerate(composites.items()):
            await _seed_ticket(session_maker, agent_id, keypair=kp)
            last = await client.post(
                f"/api/v1/validator/agent/{agent_id}/score",
                json=_score_payload(
                    agent_id, run_id=f"run_med_{i}", keypair=kp, composite=comp
                ),
            )
            assert last.status_code == 200, last.text
        assert last is not None
        assert last.json()["status"] == AgentStatus.SCORED

        # Public transparency record (the diagram's "which validators / all 3
        # scores + median"): all three validators, their exact composites +
        # signatures, and the median the platform finalized on.
        record = await client.get(f"/api/v1/public/agent/{agent_id}/scores")
        assert record.status_code == 200, record.text
        body = record.json()
        assert body["score_count"] == 3
        assert body["quorum"] == 3
        assert body["median_composite"] == pytest.approx(0.82)
        by_hotkey = {s["validator_hotkey"]: s["composite"] for s in body["scores"]}
        assert by_hotkey == {
            kp.ss58_address: pytest.approx(comp) for kp, comp in composites.items()
        }
        assert all(s["signature"] for s in body["scores"])
        assert all(
            datetime.fromisoformat(s["ticket_deadline"].replace("Z", "+00:00"))
            == _TICKET_DEADLINE
            for s in body["scores"]
        )

    async def test_expired_ticket_reopens_slot_for_a_new_validator(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app, extra_keypairs=(_DAVE,))

        # Three distinct validators claim the k=3 slots via the job endpoint.
        for kp in _KEYPAIRS:
            r = await client.post(
                "/api/v1/validator/job",
                headers={"X-Validator-Hotkey": kp.ss58_address},
                json=_job_payload(kp),
            )
            assert r.status_code == 200, r.text
        # A fourth, never-assigned validator is shut out (pool full, not
        # already-mine): "no job for you".
        dave_hdr = {"X-Validator-Hotkey": _DAVE.ss58_address}
        assert (
            await client.post(
                "/api/v1/validator/job", headers=dave_hdr, json=_job_payload(_DAVE)
            )
        ).status_code == 204

        # One validator's ticket lapses past its deadline, re-opening its slot.
        async with session_maker() as s, s.begin():
            lapsed = await s.get(
                ValidatorTicket, (agent_id, 2, _KEYPAIRS[0].ss58_address)
            )
            assert lapsed is not None
            lapsed.deadline = datetime.now(UTC) - timedelta(minutes=1)

        # The fourth validator now picks up the re-opened slot.
        reopened = await client.post(
            "/api/v1/validator/job", headers=dave_hdr, json=_job_payload(_DAVE)
        )
        assert reopened.status_code == 200, reopened.text
        assert reopened.json()["agent_id"] == str(agent_id)

    async def test_expired_ticket_cools_down_before_same_validator_retry(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        agent_id = await _seed_agent(session_maker, status=AgentStatus.EVALUATING)
        _install_db(app, session_maker)
        _install_chain(app)
        keypair = _KEYPAIRS[0]
        headers = {"X-Validator-Hotkey": keypair.ss58_address}

        claimed = await client.post(
            "/api/v1/validator/job", headers=headers, json=_job_payload(keypair)
        )
        assert claimed.status_code == 200, claimed.text

        async with session_maker() as s, s.begin():
            lapsed = await s.get(ValidatorTicket, (agent_id, 2, keypair.ss58_address))
            assert lapsed is not None
            lapsed.deadline = datetime.now(UTC) - timedelta(minutes=1)

        cooling_down = await client.post(
            "/api/v1/validator/job", headers=headers, json=_job_payload(keypair)
        )
        assert cooling_down.status_code == 204

        async with session_maker() as s, s.begin():
            lapsed = await s.get(ValidatorTicket, (agent_id, 2, keypair.ss58_address))
            assert lapsed is not None
            lapsed.retry_after = datetime.now(UTC) - timedelta(seconds=1)

        retried = await client.post(
            "/api/v1/validator/job", headers=headers, json=_job_payload(keypair)
        )
        assert retried.status_code == 200, retried.text
        assert retried.json()["agent_id"] == str(agent_id)


def test_dev_bypass_permit_refused_on_mainnet(monkeypatch) -> None:
    """The dev permit-bypass flag is honored off mainnet but refused on finney,
    so a stray env var can never open the validator surface on production."""
    from ditto.api_server.endpoints.validator import _dev_bypass_permit

    # Unset: never bypass, regardless of network.
    monkeypatch.delenv("DITTO_DEV_ALLOW_UNPERMITTED_VALIDATOR", raising=False)
    assert _dev_bypass_permit("finney") is False
    assert _dev_bypass_permit("ws://localhost:9944") is False

    # Set: honored on a dev/local network...
    monkeypatch.setenv("DITTO_DEV_ALLOW_UNPERMITTED_VALIDATOR", "true")
    assert _dev_bypass_permit("ws://localhost:9944") is True
    assert _dev_bypass_permit("test") is True
    # ...but refused on mainnet even when explicitly set.
    assert _dev_bypass_permit("finney") is False
    assert _dev_bypass_permit("Finney") is False
    assert _dev_bypass_permit("mainnet") is False
