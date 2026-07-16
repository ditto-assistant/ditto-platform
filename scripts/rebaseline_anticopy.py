"""Migrate stored anti-copy fingerprints to novelty sketches and drain holds.

Rollout companion for baseline-aware anti-copy (``ditto.anticopy.baseline``):

1. ``refingerprint`` — recompute ``agents.content_fingerprint`` as a NOVELTY
   sketch (shared starter-kit scaffolding subtracted) from each agent's stored
   tarball. Until a row is re-fingerprinted its legacy whole-tarball sketch is
   version-isolated (never compared against novelty sketches), which silently
   disables the lexical channel for that pair — so run this once over the
   ledger right after deploying the corpus.

2. ``rereview`` — re-run the (now baseline-aware) gate for every
   ``ath_pending_review`` agent against the current eligible ledger and report
   which holds would no longer fire; with ``--apply`` those agents are released
   to ``scored`` via the same manual-resolution path as
   ``scripts/resolve_review.py``. Holds that STILL fire are always left for
   human review.

Reads Postgres from ``POSTGRES_*`` and object storage from ``STORAGE_*`` env
vars (see ``.env.example``). Owner-only operation; no HTTP surface.

Usage::

    uv run python scripts/rebaseline_anticopy.py refingerprint [--dry-run]
    uv run python scripts/rebaseline_anticopy.py rereview [--apply]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import sys

from sqlalchemy import select

from ditto.anticopy.baseline import load_baseline
from ditto.api_models.agent_status import AgentStatus
from ditto.api_server.fingerprint import compute_content_fingerprint
from ditto.api_server.scoring_gate import evaluate_duplicate_signals
from ditto.api_server.storage import create_storage_client
from ditto.db import create_db_engine, create_session_maker
from ditto.db.models import Agent
from ditto.db.queries.agents import resolve_review
from ditto.db.queries.scores import list_eligible_ledger, list_scores_for_agent

logger = logging.getLogger(__name__)

# Generous fetch bound: uploads are capped far below this at intake.
_MAX_TARBALL_BYTES = 64 * 1024 * 1024
# Every status whose fingerprint can still matter to the gate: the eligible
# ledger (scored/live), the held queue itself, and in-flight evaluations that
# will hit the gate at quorum.
_REFINGERPRINT_STATUSES = (
    AgentStatus.SCORED,
    AgentStatus.LIVE,
    AgentStatus.ATH_PENDING_REVIEW,
    AgentStatus.EVALUATING,
)


async def _refingerprint(*, dry_run: bool) -> int:
    baseline = load_baseline()
    if baseline is None:
        logger.error("no baseline corpus deployed; generate one first")
        return 1
    engine = create_db_engine()
    session_maker = create_session_maker(engine)
    updated = skipped = failed = 0
    try:
        async with (
            create_storage_client() as storage,
            session_maker() as session,
            session.begin(),
        ):
            rows = (
                (
                    await session.execute(
                        select(Agent).where(Agent.status.in_(_REFINGERPRINT_STATUSES))
                    )
                )
                .scalars()
                .all()
            )
            logger.info("re-fingerprinting %d agents", len(rows))
            for agent in rows:
                fp = agent.content_fingerprint
                if fp is not None and fp.get("bl") == baseline.baseline_id:
                    skipped += 1
                    continue
                try:
                    tar_bytes = await storage.get_object(
                        key=f"{agent.agent_id}/agent.tar.gz",
                        max_bytes=_MAX_TARBALL_BYTES,
                    )
                except Exception as e:  # noqa: BLE001 - per-agent best effort
                    logger.warning(
                        "agent %s: tarball fetch failed: %s", agent.agent_id, e
                    )
                    failed += 1
                    continue
                sketch = await asyncio.to_thread(
                    compute_content_fingerprint,
                    tar_bytes,
                    exclude=baseline.shingles,
                    baseline_id=baseline.baseline_id,
                )
                if sketch is None:
                    logger.warning(
                        "agent %s: unfingerprintable tarball", agent.agent_id
                    )
                    failed += 1
                    continue
                logger.info(
                    "agent %s: novelty card=%d (was v%s)",
                    agent.agent_id,
                    sketch["card"],
                    (fp or {}).get("v"),
                )
                if not dry_run:
                    agent.content_fingerprint = sketch
                updated += 1
            if dry_run:
                await session.rollback()
    finally:
        await engine.dispose()
    logger.info(
        "done: %d updated%s, %d already current, %d failed",
        updated,
        " (dry run, not written)" if dry_run else "",
        skipped,
        failed,
    )
    return 0 if failed == 0 else 1


async def _rereview(*, apply: bool) -> int:
    engine = create_db_engine()
    session_maker = create_session_maker(engine)
    releasable: list[Agent] = []
    try:
        async with session_maker() as session, session.begin():
            held = (
                (
                    await session.execute(
                        select(Agent).where(
                            Agent.status == AgentStatus.ATH_PENDING_REVIEW
                        )
                    )
                )
                .scalars()
                .all()
            )
            eligible = await list_eligible_ledger(session)
            logger.info(
                "re-reviewing %d held agents against %d eligible rows",
                len(held),
                len(eligible),
            )
            for agent in held:
                scores = await list_scores_for_agent(session, agent_id=agent.agent_id)
                if not scores:
                    logger.warning(
                        "agent %s: held with no scores; leaving", agent.agent_id
                    )
                    continue
                median = statistics.median(float(s.composite) for s in scores)
                decision = evaluate_duplicate_signals(
                    agent_id=agent.agent_id,
                    miner_hotkey=agent.miner_hotkey,
                    sha256=agent.sha256,
                    composite=median,
                    size_bytes=agent.size_bytes,
                    eligible=eligible,
                    normalized_source_hash=agent.normalized_source_hash,
                    content_fingerprint=agent.content_fingerprint,
                    structural_fingerprint=agent.structural_fingerprint,
                    prompt_fingerprint=agent.prompt_fingerprint,
                )
                if decision.held:
                    logger.info(
                        "agent %s: STILL HELD (%s)", agent.agent_id, decision.reason
                    )
                    continue
                logger.info(
                    "agent %s: no longer held (was: %s)",
                    agent.agent_id,
                    agent.review_reason,
                )
                releasable.append(agent)
            if apply:
                for agent in releasable:
                    await resolve_review(
                        session,
                        agent_id=agent.agent_id,
                        decision=AgentStatus.SCORED,
                    )
                    logger.info("agent %s: released to scored", agent.agent_id)
    finally:
        await engine.dispose()
    logger.info(
        "%d of the held agents no longer fire the gate%s",
        len(releasable),
        "" if apply else " (report only; pass --apply to release them)",
    )
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    refp = sub.add_parser("refingerprint", help="recompute stored sketches as novelty")
    refp.add_argument("--dry-run", action="store_true", help="report, write nothing")
    rerev = sub.add_parser("rereview", help="re-run the gate over held agents")
    rerev.add_argument(
        "--apply", action="store_true", help="release no-longer-held agents to scored"
    )
    args = parser.parse_args()
    if args.command == "refingerprint":
        return asyncio.run(_refingerprint(dry_run=args.dry_run))
    return asyncio.run(_rereview(apply=args.apply))


if __name__ == "__main__":
    sys.exit(main())
