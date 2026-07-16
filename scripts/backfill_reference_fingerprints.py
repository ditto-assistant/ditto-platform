"""Recompute stored anti-copy fingerprints with the reference-aware algorithm.

The command is read-only unless ``--apply`` is passed. It changes fingerprint
metadata only; agent status, scores, duplicate attribution, and artifacts are
never mutated.

Rollout contract: deploy the matching platform version first, then run a dry pass,
apply the metadata-only backfill, and run a catch-up pass. During the bounded
transition, score-close cross-version comparisons enter individual operator review
as explicitly inconclusive; they never fall through to structural or size evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from sqlalchemy import select

from ditto.api_server.endpoints.upload import DEFAULT_MAX_TARBALL_SIZE_BYTES
from ditto.api_server.fingerprint import (
    _FP_VERSION,
    compute_content_fingerprint,
    compute_normalized_source_hash,
    compute_prompt_fingerprint,
)
from ditto.api_server.storage import (
    ObjectDownloadFailedError,
    create_storage_client,
    parse_storage_config_from_env,
)
from ditto.db import create_db_engine, create_session_maker
from ditto.db.models import Agent

logger = logging.getLogger(__name__)


def _is_current(agent: Agent) -> bool:
    fingerprint = agent.content_fingerprint
    return bool(fingerprint and fingerprint.get("v") == _FP_VERSION)


async def _run(*, apply: bool, limit: int | None) -> int:
    engine = create_db_engine()
    session_maker = create_session_maker(engine)
    storage_config = parse_storage_config_from_env()
    inspected = stale = updated = failed = 0
    try:
        async with (
            create_storage_client(storage_config) as storage,
            session_maker() as session,
        ):
            agents = (
                await session.scalars(select(Agent).order_by(Agent.agent_id))
            ).all()
            for agent in agents:
                inspected += 1
                if _is_current(agent):
                    continue
                stale += 1
                if limit is not None and updated >= limit:
                    break
                try:
                    tar_bytes = await storage.get_object(
                        key=f"{agent.agent_id}/agent.tar.gz",
                        max_bytes=DEFAULT_MAX_TARBALL_SIZE_BYTES,
                    )
                except ObjectDownloadFailedError:
                    failed += 1
                    continue
                content, normalized, prompt = await asyncio.gather(
                    asyncio.to_thread(compute_content_fingerprint, tar_bytes),
                    asyncio.to_thread(compute_normalized_source_hash, tar_bytes),
                    asyncio.to_thread(compute_prompt_fingerprint, tar_bytes),
                )
                if content is None:
                    failed += 1
                    continue
                if apply:
                    agent.content_fingerprint = content
                    agent.normalized_source_hash = normalized
                    agent.prompt_fingerprint = prompt
                    await session.commit()
                updated += 1
    finally:
        await engine.dispose()
    logger.info(
        "fingerprint backfill complete apply=%s inspected=%d stale=%d "
        "processed=%d failed=%d",
        apply,
        inspected,
        stale,
        updated,
        failed,
    )
    return 1 if failed else 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="persist recomputed fingerprint metadata"
    )
    parser.add_argument("--limit", type=int, help="maximum stale artifacts to process")
    args = parser.parse_args()
    return asyncio.run(_run(apply=args.apply, limit=args.limit))


if __name__ == "__main__":
    sys.exit(main())
