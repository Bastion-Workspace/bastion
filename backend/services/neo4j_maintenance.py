"""
Background Neo4j reconnect and backlog drain for optional knowledge graph mode.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional, Tuple

from config import settings

if TYPE_CHECKING:
    from services.knowledge_graph_service import KnowledgeGraphService

logger = logging.getLogger(__name__)


async def run_neo4j_reconnect_loop(
    kg_service: "KnowledgeGraphService",
    stop_event: asyncio.Event,
) -> None:
    """Periodically purge old backlog rows, reconnect Neo4j if needed, and drain backlog."""
    interval = max(5, int(getattr(settings, "NEO4J_RECONNECT_INTERVAL_SECONDS", 60)))
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass
        try:
            from services import kg_write_backlog as kb

            await kb.purge_expired_rows()
            if not getattr(settings, "NEO4J_ENABLED", True):
                continue
            if not kg_service.is_connected():
                await kg_service.try_reconnect()
            if kg_service.is_connected():
                drained = await kb.drain_backlog_batch(kg_service, batch_size=100)
                if drained:
                    logger.info("Drained %s Neo4j backlog row(s)", drained)
        except Exception as e:
            logger.warning("Neo4j maintenance loop iteration failed: %s", e)


def spawn_neo4j_maintenance_task(
    kg_service: Optional["KnowledgeGraphService"],
) -> Tuple[Optional[asyncio.Task], Optional[asyncio.Event]]:
    """Start background maintenance; returns (task, stop_event) or (None, None)."""
    if kg_service is None:
        return None, None
    stop = asyncio.Event()
    task = asyncio.create_task(
        run_neo4j_reconnect_loop(kg_service, stop),
        name="neo4j_reconnect_loop",
    )
    return task, stop


async def cancel_neo4j_maintenance_task(
    task: Optional[asyncio.Task],
    stop: Optional[asyncio.Event],
) -> None:
    if stop is not None:
        stop.set()
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
