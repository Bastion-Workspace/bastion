"""
Database Context Manager - Handles RLS context and secure database operations
"""

import logging
import asyncio
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

import asyncpg

from models.api_models import AuthenticatedUserResponse

logger = logging.getLogger(__name__)


class DatabaseContextManager:
    """Manages database connections with RLS context"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
    
    @asynccontextmanager
    async def get_connection(self, current_user: Optional[AuthenticatedUserResponse] = None):
        """Get database connection with RLS context set"""
        conn = None
        try:
            conn = await self.db_pool.acquire()
            
            # Set RLS context if user is provided
            if current_user:
                await self._set_rls_context(conn, current_user)
                logger.debug(f"🔐 RLS context set for user: {current_user.user_id} ({current_user.role})")
            
            yield conn
            
        except Exception as e:
            logger.error(f"❌ Database context error: {e}")
            raise
        finally:
            if conn:
                # Clear RLS context before releasing connection
                if current_user:
                    await self._clear_rls_context(conn)
                await self.db_pool.release(conn)
    
    async def _set_rls_context(self, conn: asyncpg.Connection, user: AuthenticatedUserResponse):
        """Set PostgreSQL session variables for RLS"""
        try:
            # Set user context for RLS policies
            await conn.execute(
                "SELECT set_config('app.current_user_id', $1, false)",
                user.user_id
            )
            await conn.execute(
                "SELECT set_config('app.current_user_role', $1, false)",
                user.role
            )
            
            # Set application name for audit logging
            await conn.execute(
                "SELECT set_config('application_name', $1, true)",
                f"plato-app-{user.user_id}"
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to set RLS context: {e}")
            raise
    
    async def _clear_rls_context(self, conn: asyncpg.Connection):
        """Clear PostgreSQL session variables"""
        try:
            await conn.execute("SELECT set_config('app.current_user_id', '', false)")
            await conn.execute("SELECT set_config('app.current_user_role', '', false)")
            await conn.execute("SELECT set_config('application_name', '', true)")
        except Exception as e:
            logger.debug(f"Failed to clear RLS context: {e}")
    
    async def execute_with_context(self, query: str, *args, current_user: Optional[AuthenticatedUserResponse] = None, **kwargs):
        """Execute query with RLS context"""
        async with self.get_connection(current_user) as conn:
            return await conn.execute(query, *args, **kwargs)
    
    async def fetch_with_context(self, query: str, *args, current_user: Optional[AuthenticatedUserResponse] = None, **kwargs):
        """Fetch rows with RLS context"""
        async with self.get_connection(current_user) as conn:
            return await conn.fetch(query, *args, **kwargs)
    
    async def fetchrow_with_context(self, query: str, *args, current_user: Optional[AuthenticatedUserResponse] = None, **kwargs):
        """Fetch single row with RLS context"""
        async with self.get_connection(current_user) as conn:
            return await conn.fetchrow(query, *args, **kwargs)
    
    async def executemany_with_context(self, query: str, args, current_user: Optional[AuthenticatedUserResponse] = None):
        """Execute many queries with RLS context"""
        async with self.get_connection(current_user) as conn:
            return await conn.executemany(query, args)


# Global database context manager instance
db_context = None


async def initialize_db_context(db_pool: asyncpg.Pool):
    """Initialize the global database context manager"""
    global db_context
    db_context = DatabaseContextManager(db_pool)
    logger.info("✅ Database context manager initialized")


def get_db_context() -> DatabaseContextManager:
    """Get the global database context manager"""
    if db_context is None:
        raise RuntimeError("Database context manager not initialized")
    return db_context 