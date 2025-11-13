import asyncpg
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from config.settings import settings

logger = logging.getLogger(__name__)


class DatabaseConnectionManager:
    """
    Manages PostgreSQL connection pool for data workspace database
    """
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the connection pool"""
        if self._initialized:
            logger.warning("Connection pool already initialized")
            return
        
        try:
            logger.info(f"Initializing database connection pool: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}")
            
            self.pool = await asyncpg.create_pool(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                database=settings.POSTGRES_DB,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                min_size=settings.DB_POOL_MIN_SIZE,
                max_size=settings.DB_POOL_MAX_SIZE,
                command_timeout=settings.QUERY_TIMEOUT_SECONDS,
            )
            
            self._initialized = True
            logger.info(f"Database connection pool initialized successfully (size: {settings.DB_POOL_MIN_SIZE}-{settings.DB_POOL_MAX_SIZE})")
            
            # Test connection
            async with self.pool.acquire() as conn:
                version = await conn.fetchval("SELECT version()")
                logger.info(f"PostgreSQL version: {version}")
                
        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            raise
    
    async def close(self):
        """Close the connection pool"""
        if self.pool:
            await self.pool.close()
            self._initialized = False
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool"""
        if not self._initialized or not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        async with self.pool.acquire() as conn:
            yield conn
    
    async def execute(self, query: str, *args) -> str:
        """Execute a query that doesn't return results"""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """Fetch multiple rows"""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Fetch a single row"""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args) -> Any:
        """Fetch a single value"""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)
    
    async def executemany(self, query: str, args_list: List[tuple]) -> None:
        """Execute a query multiple times with different arguments"""
        async with self.acquire() as conn:
            await conn.executemany(query, args_list)
    
    async def transaction(self):
        """Begin a transaction context"""
        async with self.acquire() as conn:
            async with conn.transaction():
                yield conn
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database connection health"""
        try:
            if not self._initialized or not self.pool:
                return {
                    "healthy": False,
                    "error": "Connection pool not initialized"
                }
            
            async with self.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                
                return {
                    "healthy": result == 1,
                    "pool_size": self.pool.get_size(),
                    "pool_free": self.pool.get_idle_size(),
                    "database": settings.POSTGRES_DB,
                    "host": settings.POSTGRES_HOST
                }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }


# Global connection manager instance
_db_manager: Optional[DatabaseConnectionManager] = None


async def get_db_manager() -> DatabaseConnectionManager:
    """Get the global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseConnectionManager()
        await _db_manager.initialize()
    return _db_manager


async def close_db_manager():
    """Close the global database manager"""
    global _db_manager
    if _db_manager:
        await _db_manager.close()
        _db_manager = None





