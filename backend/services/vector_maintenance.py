"""
Background vector-service reconnect, backlog purge, and vector_embed_backlog drain.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional, Tuple

from config import settings

if TYPE_CHECKING:
    from services.embedding_service_wrapper import EmbeddingServiceWrapper

logger = logging.getLogger(__name__)


async def run_vector_reconnect_loop(
    embedding_manager: "EmbeddingServiceWrapper",
    stop_event: asyncio.Event,
) -> None:
    interval = max(5, int(getattr(settings, "VECTOR_RECONNECT_INTERVAL_SECONDS", 60)))
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass
        try:
            from services import vector_embed_backlog as vb
            from services.vector_store_service import get_vector_store

            await vb.purge_expired_rows()
            if not getattr(settings, "VECTOR_EMBEDDING_ENABLED", True):
                continue
            vs = await get_vector_store()
            await vs.refresh_availability()
            if vs.is_vector_available():
                drained = await vb.drain_backlog_batch(embedding_manager, batch_size=100)
                if drained:
                    logger.info("Drained %s vector embed backlog row(s)", drained)
        except Exception as e:
            logger.warning("Vector maintenance loop iteration failed: %s", e)


def spawn_vector_maintenance_task(
    embedding_manager: Optional["EmbeddingServiceWrapper"],
) -> Tuple[Optional[asyncio.Task], Optional[asyncio.Event]]:
    if embedding_manager is None:
        return None, None
    stop = asyncio.Event()
    task = asyncio.create_task(
        run_vector_reconnect_loop(embedding_manager, stop),
        name="vector_reconnect_loop",
    )
    return task, stop


async def cancel_vector_maintenance_task(
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
