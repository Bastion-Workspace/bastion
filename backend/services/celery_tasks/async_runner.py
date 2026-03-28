"""
Persistent event loop for Celery workers.

Single event loop per worker process so module-level async singletons
(embedding service, gRPC clients, DB pool) stay bound to a live loop
and do not raise RuntimeError("Event loop is closed") on subsequent tasks.
"""

import asyncio
import logging
from typing import Any, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_worker_loop: asyncio.AbstractEventLoop | None = None


async def _cleanup_singletons() -> None:
    """Close shared async resources so the loop can shut down cleanly."""
    try:
        from utils.shared_db_pool import close_shared_db_pool
        await close_shared_db_pool()
    except Exception as e:
        logger.warning("Error closing shared DB pool during worker shutdown: %s", e)

    try:
        from services.embedding_service_wrapper import get_embedding_service
        svc = await get_embedding_service()
        if hasattr(svc, "close"):
            await svc.close()
    except Exception as e:
        logger.warning("Error closing embedding service during worker shutdown: %s", e)

    try:
        from clients.vector_service_client import get_vector_service_client
        client = await get_vector_service_client(required=False)
        if client and hasattr(client, "close"):
            await client.close()
    except Exception as e:
        logger.warning("Error closing vector service client during worker shutdown: %s", e)

    try:
        from clients.connections_service_client import get_connections_service_client
        client = await get_connections_service_client()
        if client and hasattr(client, "close"):
            await client.close()
    except Exception as e:
        logger.warning("Error closing connections service client during worker shutdown: %s", e)

    try:
        from clients.crawl_service_client import get_crawl_service_client
        client = await get_crawl_service_client()
        if client and hasattr(client, "close"):
            await client.close()
    except Exception as e:
        logger.warning("Error closing crawl service client during worker shutdown: %s", e)

    try:
        from clients.tool_service_client import close_tool_service_client
        await close_tool_service_client()
    except Exception as e:
        logger.warning("Error closing tool service client during worker shutdown: %s", e)


def init_worker_loop() -> None:
    """Create the worker event loop (e.g. from worker_process_init). Optional; loop is created lazily on first run_async if not set."""
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
        logger.debug("Worker event loop initialized")


def close_worker_loop() -> None:
    """Close singletons and the worker event loop (call from worker_shutdown)."""
    global _worker_loop
    if _worker_loop is None:
        return
    if _worker_loop.is_closed():
        _worker_loop = None
        return
    try:
        _worker_loop.run_until_complete(_cleanup_singletons())
    except Exception as e:
        logger.warning("Error during worker loop cleanup: %s", e)
    try:
        _worker_loop.close()
    except Exception as e:
        logger.warning("Error closing worker event loop: %s", e)
    _worker_loop = None


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine on the worker's persistent event loop. Creates the loop on first use."""
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
    return _worker_loop.run_until_complete(coro)
