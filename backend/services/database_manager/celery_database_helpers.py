"""
Celery Database Helpers
Simple database operations for Celery worker environments

**BULLY!** Simple, reliable database operations for Celery workers!
No complex connection pools - just direct connections that work!
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
import asyncpg

logger = logging.getLogger(__name__)


async def celery_fetch_one(query: str, *args, rls_context: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
    """
    Fetch one record using a simple connection (Celery-safe)
    
    **By George!** Simple database query for Celery workers!
    """
    database_url = os.getenv("DATABASE_URL", "postgresql://plato_user:plato_secure_password@localhost:5432/plato_knowledge_base")
    
    try:
        # Use simple connection for Celery workers
        conn = await asyncpg.connect(database_url)
        try:
            # Log what RLS context we received
            logger.info(f"🔍 Celery fetch_one received rls_context: {rls_context}")
            
            # Set RLS context if provided - CRITICAL: Execute as batch with BEGIN
            if rls_context:
                user_id = rls_context.get('user_id', '')
                user_role = rls_context.get('user_role', 'admin')
                
                # set_config() requires a string value, not NULL
                # For global/admin operations, use empty string
                user_id_str = '' if user_id is None else str(user_id)
                
                # Start transaction explicitly
                await conn.execute("BEGIN")
                
                # Use set_config with is_local=true for transaction-local settings
                await conn.execute("SELECT set_config('app.current_user_id', $1, true)", user_id_str)
                await conn.execute("SELECT set_config('app.current_user_role', $1, true)", user_role)
                logger.info(f"🔐 Celery: Set RLS context - user_id='{user_id_str}', role='{user_role}'")
                
                # Verify the settings were applied
                verify_user_id = await conn.fetchval("SELECT current_setting('app.current_user_id', true)")
                verify_role = await conn.fetchval("SELECT current_setting('app.current_user_role', true)")
                logger.info(f"🔐 Celery: Verified RLS context - user_id='{verify_user_id}', role='{verify_role}'")
                
                # Execute query within same transaction
                result = await conn.fetchrow(query, *args)
                
                # Commit transaction
                await conn.execute("COMMIT")
                
                return dict(result) if result else None
            else:
                # No RLS context, execute directly
                result = await conn.fetchrow(query, *args)
                return dict(result) if result else None
        except Exception as e:
            # Rollback on error if we're in a transaction
            if rls_context:
                try:
                    await conn.execute("ROLLBACK")
                except:
                    pass
            raise
        finally:
            await conn.close()
    except Exception as e:
        logger.error(f"❌ Celery database query failed: {e}")
        raise


async def celery_fetch_all(query: str, *args, rls_context: Dict[str, str] = None) -> List[Dict[str, Any]]:
    """
    Fetch all records using a simple connection (Celery-safe)
    
    **BULLY!** Get all rows for Celery workers!
    """
    database_url = os.getenv("DATABASE_URL", "postgresql://plato_user:plato_secure_password@localhost:5432/plato_knowledge_base")
    
    try:
        conn = await asyncpg.connect(database_url)
        try:
            # Set RLS context if provided
            if rls_context:
                user_id = rls_context.get('user_id', '')
                user_role = rls_context.get('user_role', 'admin')
                
                # set_config() requires a string value, not NULL
                # For global/admin operations, use empty string
                user_id_str = '' if user_id is None else str(user_id)
                
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id_str)
                await conn.execute("SELECT set_config('app.current_user_role', $1, false)", user_role)
                logger.debug(f"🔍 Set RLS context on Celery connection: user_id={user_id_str}, role={user_role}")
            
            results = await conn.fetch(query, *args)
            return [dict(record) for record in results]
        finally:
            await conn.close()
    except Exception as e:
        logger.error(f"❌ Celery database query failed: {e}")
        raise


async def celery_fetch_value(query: str, *args, rls_context: Dict[str, str] = None) -> Any:
    """
    Fetch a single value using a simple connection (Celery-safe)
    
    **BULLY!** Get a single value for Celery workers!
    """
    database_url = os.getenv("DATABASE_URL", "postgresql://plato_user:plato_secure_password@localhost:5432/plato_knowledge_base")
    
    try:
        conn = await asyncpg.connect(database_url)
        try:
            # Set RLS context if provided
            if rls_context:
                user_id = rls_context.get('user_id', '')
                user_role = rls_context.get('user_role', 'user')
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                await conn.execute("SELECT set_config('app.current_user_role', $1, false)", user_role)
            
            result = await conn.fetchval(query, *args)
            return result
        finally:
            await conn.close()
    except Exception as e:
        logger.error(f"❌ Celery database fetchval failed: {e}")
        raise


async def celery_execute(query: str, *args, rls_context: Dict[str, str] = None) -> str:
    """
    Execute a query using a simple connection (Celery-safe)
    
    **By George!** Execute commands for Celery workers!
    """
    database_url = os.getenv("DATABASE_URL", "postgresql://plato_user:plato_secure_password@localhost:5432/plato_knowledge_base")
    
    try:
        conn = await asyncpg.connect(database_url)
        try:
            # Set RLS context if provided
            if rls_context:
                user_id = rls_context.get('user_id', '')
                user_role = rls_context.get('user_role', 'admin')
                
                # set_config() requires a string value, not NULL
                # For global/admin operations, use empty string
                user_id_str = '' if user_id is None else str(user_id)
                
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id_str)
                await conn.execute("SELECT set_config('app.current_user_role', $1, false)", user_role)
                logger.debug(f"🔍 Set RLS context on Celery connection: user_id={user_id_str}, role={user_role}")
            
            result = await conn.execute(query, *args)
            return result
        finally:
            await conn.close()
    except Exception as e:
        logger.error(f"❌ Celery database execute failed: {e}")
        raise


def is_celery_worker() -> bool:
    """Check if we're running in a Celery worker environment"""
    import sys
    import threading
    
    # Multiple ways to detect Celery worker environment
    celery_indicators = [
        os.getenv('CELERY_WORKER_RUNNING', 'false').lower() == 'true',
        'celery' in os.getenv('WORKER_TYPE', '').lower(),
        'celery' in str(os.getenv('_', '')).lower(),
        os.getenv('CELERY_LOADER') is not None,
        'celery' in ' '.join(sys.argv).lower(),
        'ForkPoolWorker' in threading.current_thread().name,
        # More specific worker detection - avoid false positives from uvicorn --workers
        ('worker' in threading.current_thread().name.lower() and 'ForkPoolWorker' in threading.current_thread().name)
    ]
    
    is_worker = any(celery_indicators)
    
    # Debug logging
    logger.debug(f"Environment check - CELERY_WORKER_RUNNING: {os.getenv('CELERY_WORKER_RUNNING')}")
    logger.debug(f"Environment check - Thread name: {threading.current_thread().name}")
    logger.debug(f"Environment check - Command line: {' '.join(sys.argv)}")
    logger.debug(f"Environment check - Is Celery worker: {is_worker}")
    
    if is_worker:
        logger.debug("Detected Celery worker environment - using simple database connections")
    
    return is_worker
