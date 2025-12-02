"""
Shared Database Pool Manager
Provides a global database connection pool for use across Celery workers
"""

import logging
import asyncio
from typing import Optional
import asyncpg
from config import settings

logger = logging.getLogger(__name__)

# Global shared pool instance
_shared_db_pool: Optional[asyncpg.Pool] = None
_pool_lock = asyncio.Lock()

async def get_shared_db_pool() -> asyncpg.Pool:
    """Get or create the global shared database connection pool"""
    global _shared_db_pool
    
    async with _pool_lock:
        if _shared_db_pool is None or _shared_db_pool.is_closing():
            logger.info("🔧 Creating global shared database pool...")
            
            try:
                _shared_db_pool = await asyncpg.create_pool(
                    host=settings.POSTGRES_HOST,
                    port=settings.POSTGRES_PORT,
                    user=settings.POSTGRES_USER,
                    password=settings.POSTGRES_PASSWORD,
                    database=settings.POSTGRES_DB,
                    min_size=5,  # Reduced for Celery workers
                    max_size=20,  # Reduced for Celery workers
                    command_timeout=settings.DB_POOL_COMMAND_TIMEOUT,
                    max_queries=settings.DB_POOL_MAX_QUERIES,
                    max_inactive_connection_lifetime=settings.DB_POOL_MAX_INACTIVE_TIME
                )
                logger.info("✅ Global shared database pool created")
            except Exception as e:
                logger.error(f"❌ Failed to create global shared database pool: {e}")
                raise
        
        return _shared_db_pool

async def close_shared_db_pool():
    """Close the global shared database connection pool"""
    global _shared_db_pool
    
    async with _pool_lock:
        if _shared_db_pool and not _shared_db_pool.is_closing():
            await _shared_db_pool.close()
            _shared_db_pool = None
            logger.info("✅ Global shared database pool closed")
