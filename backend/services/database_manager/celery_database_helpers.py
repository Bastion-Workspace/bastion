"""
Celery Database Helpers
Simple database operations for Celery worker environments using a shared pool.
"""

import logging
import os
from typing import List, Dict, Any, Optional
import asyncpg

logger = logging.getLogger(__name__)


async def _clear_rls_session_vars(conn) -> None:
    """Reset RLS GUCs before returning a connection to the pool (session-scoped set_config)."""
    try:
        await conn.execute("SELECT set_config('app.current_user_id', '', false)")
        await conn.execute("SELECT set_config('app.current_user_role', '', false)")
    except Exception as e:
        logger.debug("RLS GUC clear on release (non-fatal): %s", e)


async def _get_pool() -> asyncpg.Pool:
    """Return the shared per-process connection pool."""
    from utils.shared_db_pool import get_shared_db_pool
    return await get_shared_db_pool()


async def celery_fetch_one(query: str, *args, rls_context: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
    """Fetch one record using the shared pool (Celery-safe)."""
    try:
        pool = await _get_pool()
        logger.info(f"🔍 Celery fetch_one received rls_context: {rls_context}")

        async with pool.acquire() as conn:
            try:
                if rls_context:
                    user_id = rls_context.get('user_id', '')
                    user_role = rls_context.get('user_role', 'admin')
                    user_id_str = '' if user_id is None else str(user_id)

                    async with conn.transaction():
                        await conn.execute("SELECT set_config('app.current_user_id', $1, true)", user_id_str)
                        await conn.execute("SELECT set_config('app.current_user_role', $1, true)", user_role)

                        verify_user_id = await conn.fetchval("SELECT current_setting('app.current_user_id', true)")
                        verify_role = await conn.fetchval("SELECT current_setting('app.current_user_role', true)")
                        logger.info(f"🔐 Celery: Verified RLS context - user_id='{verify_user_id}', role='{verify_role}'")

                        result = await conn.fetchrow(query, *args)
                        return dict(result) if result else None
                else:
                    result = await conn.fetchrow(query, *args)
                    return dict(result) if result else None
            finally:
                await _clear_rls_session_vars(conn)

    except Exception as e:
        logger.error(f"❌ Celery database query failed: {e}")
        raise


async def celery_fetch_all(query: str, *args, rls_context: Dict[str, str] = None) -> List[Dict[str, Any]]:
    """Fetch all records using the shared pool (Celery-safe)."""
    try:
        pool = await _get_pool()

        async with pool.acquire() as conn:
            try:
                if rls_context:
                    user_id = rls_context.get('user_id', '')
                    user_role = rls_context.get('user_role', 'admin')
                    user_id_str = '' if user_id is None else str(user_id)

                    await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id_str)
                    await conn.execute("SELECT set_config('app.current_user_role', $1, false)", user_role)
                    logger.debug(f"🔍 Set RLS context on Celery connection: user_id={user_id_str}, role={user_role}")

                results = await conn.fetch(query, *args)
                return [dict(record) for record in results]
            finally:
                await _clear_rls_session_vars(conn)

    except Exception as e:
        logger.error(f"❌ Celery database query failed: {e}")
        raise


async def celery_fetch_value(query: str, *args, rls_context: Dict[str, str] = None) -> Any:
    """Fetch a single value using the shared pool (Celery-safe)."""
    try:
        pool = await _get_pool()

        async with pool.acquire() as conn:
            try:
                if rls_context:
                    user_id = rls_context.get('user_id', '')
                    user_role = rls_context.get('user_role', 'user')
                    user_id_str = '' if user_id is None else str(user_id)
                    await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id_str)
                    await conn.execute("SELECT set_config('app.current_user_role', $1, false)", user_role)

                return await conn.fetchval(query, *args)
            finally:
                await _clear_rls_session_vars(conn)

    except Exception as e:
        logger.error(f"❌ Celery database fetchval failed: {e}")
        raise


async def celery_execute(query: str, *args, rls_context: Dict[str, str] = None) -> str:
    """Execute a query using the shared pool (Celery-safe)."""
    try:
        pool = await _get_pool()

        async with pool.acquire() as conn:
            try:
                if rls_context:
                    user_id = rls_context.get('user_id', '')
                    user_role = rls_context.get('user_role', 'admin')
                    user_id_str = '' if user_id is None else str(user_id)

                    await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id_str)
                    await conn.execute("SELECT set_config('app.current_user_role', $1, false)", user_role)
                    logger.debug(f"🔍 Set RLS context on Celery connection: user_id={user_id_str}, role={user_role}")

                return await conn.execute(query, *args)
            finally:
                await _clear_rls_session_vars(conn)

    except Exception as e:
        logger.error(f"❌ Celery database execute failed: {e}")
        raise


def run_async_db_task(async_fn):
    """
    Run an async coroutine function from a synchronous Celery task body.
    Expects a zero-arg async callable, e.g. ``async def _run(): ...`` then ``run_async_db_task(_run)``.
    """
    from services.celery_tasks.async_runner import run_async

    return run_async(async_fn())


def is_celery_worker() -> bool:
    """Check if we're running in a Celery worker environment."""
    import sys
    import threading

    celery_indicators = [
        os.getenv('CELERY_WORKER_RUNNING', 'false').lower() == 'true',
        'celery' in os.getenv('WORKER_TYPE', '').lower(),
        'celery' in str(os.getenv('_', '')).lower(),
        os.getenv('CELERY_LOADER') is not None,
        'celery' in ' '.join(sys.argv).lower(),
        'ForkPoolWorker' in threading.current_thread().name,
        ('worker' in threading.current_thread().name.lower() and 'ForkPoolWorker' in threading.current_thread().name)
    ]

    is_worker = any(celery_indicators)

    logger.debug(f"Environment check - CELERY_WORKER_RUNNING: {os.getenv('CELERY_WORKER_RUNNING')}")
    logger.debug(f"Environment check - Thread name: {threading.current_thread().name}")
    logger.debug(f"Environment check - Command line: {' '.join(sys.argv)}")
    logger.debug(f"Environment check - Is Celery worker: {is_worker}")

    return is_worker
