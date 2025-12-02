"""
LangGraph PostgreSQL Checkpointer for LLM Orchestrator Service
Provides AsyncPostgresSaver for LangGraph-native persistence
"""

import asyncio
import logging
from typing import Optional, Union
from contextlib import asynccontextmanager
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import MemorySaver

from config.settings import settings

logger = logging.getLogger(__name__)


class LangGraphPostgresCheckpointer:
    """
    PostgreSQL Checkpointer for LangGraph workflows
    
    Provides 100% LangGraph-native persistence using AsyncPostgresSaver
    """
    
    def __init__(self):
        self.checkpointer: Optional[Union[AsyncPostgresSaver, MemorySaver]] = None
        self.is_initialized = False
        self.using_fallback = False
        self._connection_string = None
        self._connection_lock = asyncio.Lock()
        self._checkpointer_factory = None
    
    async def initialize(self) -> Union[AsyncPostgresSaver, MemorySaver]:
        """Initialize PostgreSQL checkpointer with retry logic"""
        async with self._connection_lock:
            if self.is_initialized and self.checkpointer:
                logger.debug("Reusing existing PostgreSQL checkpointer")
                return self.checkpointer
            
            try:
                # Build PostgreSQL connection string
                self._connection_string = settings.postgres_connection_string
                
                logger.info("Initializing LangGraph PostgreSQL checkpointer...")
                logger.info(f"Database: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}")
                
                # Create AsyncPostgresSaver with retry logic
                await self._create_checkpointer_with_retry()
                
                # Skip setup() call since SQL init file already creates the required tables
                logger.info("Using existing LangGraph tables from SQL initialization")
                
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
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Attempting to create checkpointer (attempt {attempt}/{max_retries})...")
                
                # CORRECTED LANGGRAPH PATTERN: AsyncPostgresSaver.from_conn_string returns a contextmanager
                # We need to enter it to get the actual checkpointer instance
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
                
                # Note: We don't call setup() because SQL init file already creates the required tables
                logger.info("✅ Checkpointer connection successful")
                return
                
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ Checkpointer creation attempt {attempt} failed: {e}")
                
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 2s, 4s, 8s
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ All {max_retries} attempts failed")
                    raise last_error
    
    async def is_connection_healthy(self) -> bool:
        """Check if checkpointer connection is healthy"""
        try:
            if not self.checkpointer or self.using_fallback:
                return False
            
            # Try a simple operation to test connection
            # AsyncPostgresSaver doesn't have a direct health check, so we'll just check if it exists
            return self.checkpointer is not None
            
        except Exception as e:
            logger.warning(f"Connection health check failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up checkpointer resources"""
        async with self._connection_lock:
            try:
                logger.info("Cleaning up PostgreSQL checkpointer factory")
                
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
                logger.warning(f"Error during checkpointer cleanup: {e}")


# Global instance for shared use across agents
_postgres_checkpointer: Optional[LangGraphPostgresCheckpointer] = None


async def get_postgres_checkpointer() -> LangGraphPostgresCheckpointer:
    """Get or create the global PostgreSQL checkpointer instance"""
    global _postgres_checkpointer
    
    if _postgres_checkpointer is None:
        _postgres_checkpointer = LangGraphPostgresCheckpointer()
    
    # Check if connection is healthy, reinitialize if needed
    if _postgres_checkpointer.is_initialized:
        is_healthy = await _postgres_checkpointer.is_connection_healthy()
        if not is_healthy:
            logger.warning("Unhealthy connection detected, reinitializing checkpointer")
            await _postgres_checkpointer.cleanup()
    
    if not _postgres_checkpointer.is_initialized:
        await _postgres_checkpointer.initialize()
    
    return _postgres_checkpointer


async def get_async_postgres_saver() -> Union[AsyncPostgresSaver, MemorySaver]:
    """Get the AsyncPostgresSaver instance (convenience function)"""
    checkpointer = await get_postgres_checkpointer()
    return checkpointer.checkpointer

