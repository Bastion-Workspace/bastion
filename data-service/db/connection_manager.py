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
    async def acquire(self, user_id: Optional[str] = None, user_team_ids: Optional[List[str]] = None):
        """Acquire a connection from the pool with optional RLS context"""
        if not self._initialized or not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        conn = await self.pool.acquire()
        try:
            # Set RLS context if user information provided
            if user_id:
                await self._set_rls_context(conn, user_id, user_team_ids or [])
            
            yield conn
        finally:
            # Clear RLS context before releasing
            if user_id:
                await self._clear_rls_context(conn)
            await self.pool.release(conn)
    
    async def _set_rls_context(self, conn: asyncpg.Connection, user_id: str, user_team_ids: List[str]):
        """Set PostgreSQL session variables for RLS"""
        try:
            # Set user ID for RLS policies
            await conn.execute(
                "SELECT set_config('app.current_user_id', $1, false)",
                user_id
            )
            
            # Set team IDs as comma-separated string
            team_ids_str = ','.join(user_team_ids) if user_team_ids else ''
            await conn.execute(
                "SELECT set_config('app.current_user_team_ids', $1, false)",
                team_ids_str
            )
            
            logger.debug(f"RLS context set for user: {user_id} (teams: {len(user_team_ids)})")
        except Exception as e:
            logger.error(f"Failed to set RLS context: {e}")
            raise
    
    async def _clear_rls_context(self, conn: asyncpg.Connection):
        """Clear PostgreSQL session variables"""
        try:
            await conn.execute("SELECT set_config('app.current_user_id', '', false)")
            await conn.execute("SELECT set_config('app.current_user_team_ids', '', false)")
        except Exception as e:
            logger.debug(f"Failed to clear RLS context: {e}")
    
    async def execute(self, query: str, *args, user_id: Optional[str] = None, user_team_ids: Optional[List[str]] = None) -> str:
        """Execute a query that doesn't return results"""
        async with self.acquire(user_id=user_id, user_team_ids=user_team_ids) as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args, user_id: Optional[str] = None, user_team_ids: Optional[List[str]] = None) -> List[asyncpg.Record]:
        """Fetch multiple rows"""
        async with self.acquire(user_id=user_id, user_team_ids=user_team_ids) as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args, user_id: Optional[str] = None, user_team_ids: Optional[List[str]] = None) -> Optional[asyncpg.Record]:
        """Fetch a single row"""
        async with self.acquire(user_id=user_id, user_team_ids=user_team_ids) as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args, user_id: Optional[str] = None, user_team_ids: Optional[List[str]] = None) -> Any:
        """Fetch a single value"""
        async with self.acquire(user_id=user_id, user_team_ids=user_team_ids) as conn:
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









