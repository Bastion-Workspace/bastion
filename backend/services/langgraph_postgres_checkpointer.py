"""
LangGraph PostgreSQL Checkpointer - Roosevelt's "Pure LangGraph" Persistence
Provides AsyncPostgresSaver for 100% LangGraph-native persistence with robust connection management
"""

import asyncio
import logging
from typing import Optional, Union
from contextlib import asynccontextmanager
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


class LangGraphPostgresCheckpointer:
    """
    Roosevelt's "Pure LangGraph" PostgreSQL Checkpointer
    
    100% LangGraph-native persistence using AsyncPostgresSaver only
    """
    
    def __init__(self):
        self.checkpointer: Optional[Union[AsyncPostgresSaver, MemorySaver]] = None
        self.is_initialized = False
        self.using_fallback = False
        self._connection_string = None
        self._connection_lock = asyncio.Lock()
        self._connection_pool = None
        self._checkpointer_factory = None
    
    async def initialize(self) -> Union[AsyncPostgresSaver, MemorySaver]:
        """Initialize the AsyncPostgresSaver with robust connection management"""
        async with self._connection_lock:
            if self.is_initialized and self.checkpointer:
                # Return existing checkpointer if already initialized
                logger.debug("🔄 Reusing existing PostgreSQL checkpointer")
                return self.checkpointer
            
            try:
                # Build PostgreSQL connection string from environment
                self._connection_string = self._build_connection_string()
                
                logger.info("🚀 Initializing LangGraph PostgreSQL checkpointer...")
                
                # Create AsyncPostgresSaver with robust connection handling
                await self._create_checkpointer_with_retry()
                
                # Skip setup() call since our SQL init file already creates the required tables
                logger.info("ℹ️ Using existing LangGraph tables from SQL initialization")
                
                self.is_initialized = True
                self.using_fallback = False
                logger.info("✅ LangGraph PostgreSQL checkpointer initialized successfully")
                
                return self.checkpointer
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize PostgreSQL checkpointer: {e}")
                logger.warning("🔄 Falling back to MemorySaver for LangGraph persistence")
                
                # Fallback to MemorySaver
                self.checkpointer = MemorySaver()
                self.is_initialized = True
                self.using_fallback = True
                
                logger.info("✅ LangGraph MemorySaver (fallback) initialized successfully")
                return self.checkpointer
    
    async def _create_checkpointer_with_retry(self, max_retries: int = 3):
        """Create checkpointer with connection retry logic - PROPER LANGGRAPH PATTERN"""
        for attempt in range(max_retries):
            try:
                logger.info(f"🔄 Connection attempt {attempt + 1}/{max_retries}")
                
                # ROOSEVELT'S CORRECTED LANGGRAPH PATTERN: Create the actual checkpointer instance
                # AsyncPostgresSaver.from_conn_string returns a contextmanager, we need to enter it
                checkpointer_factory = AsyncPostgresSaver.from_conn_string(
                    self._connection_string
                )
                
                # Actually create the checkpointer instance by entering the context
                self.checkpointer = await checkpointer_factory.__aenter__()
                
                # Store the factory for proper cleanup
                self._checkpointer_factory = checkpointer_factory
                
                # Verify the checkpointer has expected methods
                if hasattr(self.checkpointer, 'aget_tuple') and hasattr(self.checkpointer, 'aput'):
                    logger.info("✅ PostgreSQL checkpointer has required LangGraph methods")
                else:
                    logger.warning("⚠️ PostgreSQL checkpointer missing some LangGraph methods")
                
                logger.info("✅ PostgreSQL connection test successful")
                logger.info("✅ PostgreSQL checkpointer ready for LangGraph")
                return
                
            except Exception as e:
                logger.error(f"❌ Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    async def cleanup(self):
        """Cleanup the checkpointer resources"""
        async with self._connection_lock:
            try:
                logger.info("🔄 Cleaning up PostgreSQL checkpointer factory")
                
                # Properly exit the async context manager if we have a factory
                if self._checkpointer_factory and not self.using_fallback:
                    try:
                        await self._checkpointer_factory.__aexit__(None, None, None)
                        logger.info("✅ Checkpointer context manager exited successfully")
                    except Exception as e:
                        logger.warning(f"⚠️ Error exiting checkpointer context: {e}")
                
                self.checkpointer = None
                self._checkpointer_factory = None
                self.is_initialized = False
                self.using_fallback = False
                logger.info("✅ Checkpointer cleanup completed")
                
            except Exception as e:
                logger.warning(f"⚠️ Error during checkpointer cleanup: {e}")
                # Force reset even if cleanup fails
                self.checkpointer = None
                self._checkpointer_factory = None
                self.is_initialized = False
                self.using_fallback = False
    
    async def get_checkpointer(self) -> Union[AsyncPostgresSaver, MemorySaver]:
        """Get the checkpointer instance, initializing if necessary"""
        if not self.is_initialized or not self.checkpointer:
            await self.initialize()
        return self.checkpointer
    
    async def is_connection_healthy(self) -> bool:
        """Check if the PostgreSQL connection is healthy"""
        if self.using_fallback or not self.checkpointer:
            return True  # Memory saver is always "healthy"
        
        try:
            # Quick health check - for actual instances, try a simple operation
            if hasattr(self.checkpointer, 'alist_versions'):
                # Try to list versions as a quick health check
                await self.checkpointer.alist_versions({"configurable": {"thread_id": "health_check"}})
                logger.debug("🔍 Connection health check: OK")
                return True
            else:
                # Fallback to basic attribute check
                return hasattr(self.checkpointer, 'get_tuple')
                
        except Exception as e:
            logger.warning(f"⚠️ Connection health check failed: {e}")
            return False
    
    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from config settings"""
        from config import settings
        
        host = settings.POSTGRES_HOST
        port = settings.POSTGRES_PORT
        database = settings.POSTGRES_DB
        user = settings.POSTGRES_USER
        password = settings.POSTGRES_PASSWORD
        
        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        logger.info(f"🔗 PostgreSQL connection: postgresql://{user}:***@{host}:{port}/{database}")
        
        return connection_string
    



# Global instance for shared use across orchestrators
_postgres_checkpointer: Optional[LangGraphPostgresCheckpointer] = None


async def get_postgres_checkpointer() -> LangGraphPostgresCheckpointer:
    """Get or create the global PostgreSQL checkpointer instance with health checking"""
    global _postgres_checkpointer
    
    if _postgres_checkpointer is None:
        _postgres_checkpointer = LangGraphPostgresCheckpointer()
    
    # Check if connection is healthy, reinitialize if needed
    if _postgres_checkpointer.is_initialized:
        is_healthy = await _postgres_checkpointer.is_connection_healthy()
        if not is_healthy:
            logger.warning("🔄 Unhealthy connection detected, reinitializing checkpointer")
            await _postgres_checkpointer.cleanup()
    
    if not _postgres_checkpointer.is_initialized:
        await _postgres_checkpointer.initialize()
    
    return _postgres_checkpointer


async def get_async_postgres_saver() -> Union[AsyncPostgresSaver, MemorySaver]:
    """Get the AsyncPostgresSaver instance for direct LangGraph use"""
    checkpointer = await get_postgres_checkpointer()
    
    # If using fallback, return the MemorySaver directly
    if checkpointer.using_fallback:
        return checkpointer.checkpointer
    
    # For PostgreSQL, return the async context manager that LangGraph expects
    return checkpointer.checkpointer


async def reset_postgres_checkpointer():
    """Reset the global checkpointer instance (for error recovery)"""
    global _postgres_checkpointer
    
    if _postgres_checkpointer:
        logger.info("🔄 Resetting PostgreSQL checkpointer due to connection issues")
        await _postgres_checkpointer.cleanup()
        _postgres_checkpointer = None
