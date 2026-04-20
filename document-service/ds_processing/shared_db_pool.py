"""
Shared Database Pool Manager
Provides a per-process connection pool for Celery workers and other consumers.

Note: asyncio.Lock() is NOT used here. A module-level lock created before
os.fork() becomes stale in child processes because the child inherits the lock
object's event loop reference, which points to the parent's (now dead) loop.
In a single-threaded asyncio process (each ForkPoolWorker) the `is None` check
is race-condition-free without a lock.
"""

import logging
import os
from typing import Optional
import asyncpg

logger = logging.getLogger(__name__)

# Global shared pool instance (per-process — each forked Celery worker gets its own)
_shared_db_pool: Optional[asyncpg.Pool] = None


async def get_shared_db_pool() -> asyncpg.Pool:
    """Get or create the per-process shared database connection pool."""
    global _shared_db_pool

    if _shared_db_pool is None or _shared_db_pool.is_closing():
        logger.info("Creating per-process shared database pool...")

        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise RuntimeError("DATABASE_URL environment variable is not set")

        try:
            _shared_db_pool = await asyncpg.create_pool(
                database_url,
                min_size=2,
                max_size=10,
            )
            logger.info("Per-process shared database pool created")
        except Exception as e:
            logger.error(f"Failed to create per-process shared database pool: {e}")
            raise

    return _shared_db_pool


async def close_shared_db_pool():
    """Close the per-process shared database connection pool."""
    global _shared_db_pool

    if _shared_db_pool and not _shared_db_pool.is_closing():
        await _shared_db_pool.close()
        _shared_db_pool = None
        logger.info("Per-process shared database pool closed")
