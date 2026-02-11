"""
Database Manager Service
Centralized database connection and transaction management

**BULLY!** This service provides a single, robust connection pool for all database operations!
Following Roosevelt's "Trust-Busting" principles - one manager to rule them all!
"""

import asyncio
import logging
import time
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, AsyncContextManager, Callable, Union
from contextlib import asynccontextmanager
import asyncpg
from asyncpg import Pool, Connection, Record

from .models.database_models import (
    DatabaseConfig, ConnectionStats, QueryResult, TransactionContext,
    DatabaseHealth, ConnectionPoolStatus
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Centralized Database Manager Service
    
    **BULLY!** The single source of truth for all database operations!
    No more connection pool chaos - everything goes through here!
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or self._default_config()
        self._pool: Optional[Pool] = None
        self._initialized = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._stats = self._init_stats()
        self._active_transactions: Dict[str, TransactionContext] = {}
        self._query_history: List[Dict[str, Any]] = []
        self._error_history: List[str] = []
        self._start_time = datetime.now()
        
        logger.info(f"🏗️ DatabaseManager initialized with config: {self.config}")
    
    def _default_config(self) -> DatabaseConfig:
        """Create default database configuration"""
        return DatabaseConfig(
            database_url=os.getenv("DATABASE_URL", "postgresql://plato_user:plato_secure_password@localhost:5432/plato_knowledge_base"),
            min_pool_size=5,
            max_pool_size=20,
            command_timeout=60,
            max_queries=50000,
            max_inactive_connection_lifetime=300,
            connection_max_age=3600,
            retry_attempts=3,
            retry_delay=1.0,
            health_check_interval=30,
            enable_query_logging=False,
            enable_performance_monitoring=True
        )
    
    def _init_stats(self) -> ConnectionStats:
        """Initialize connection statistics"""
        return ConnectionStats(
            pool_status=ConnectionPoolStatus.INITIALIZING,
            total_connections=0,
            active_connections=0,
            idle_connections=0,
            pending_operations=0,
            total_queries_executed=0,
            failed_queries=0,
            average_query_time=0.0,
            last_health_check=datetime.now(),
            uptime_seconds=0,
            error_rate=0.0
        )
    
    async def initialize(self) -> None:
        """
        Initialize the database connection pool
        
        **BULLY!** Start up the cavalry charge with a robust connection pool!
        """
        if self._initialized:
            logger.warning("⚠️ DatabaseManager already initialized")
            return
        
        logger.info("🚀 Initializing DatabaseManager connection pool...")
        
        try:
            # Create the connection pool with robust configuration
            self._pool = await asyncpg.create_pool(
                self.config.database_url,
                min_size=self.config.min_pool_size,
                max_size=self.config.max_pool_size,
                command_timeout=self.config.command_timeout,
                max_queries=self.config.max_queries,
                max_inactive_connection_lifetime=self.config.max_inactive_connection_lifetime,
                server_settings={
                    'application_name': 'plato_database_manager',
                    'search_path': 'public'
                }
            )
            
            # Test the connection
            async with self._pool.acquire() as conn:
                version = await conn.fetchval("SELECT version()")
                logger.info(f"📊 Database connection successful: {version}")
            
            # Update status
            self._stats.pool_status = ConnectionPoolStatus.HEALTHY
            self._initialized = True
            
            # Start health monitoring
            if self.config.enable_performance_monitoring:
                self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            logger.info("✅ DatabaseManager initialized successfully")
            
        except Exception as e:
            self._stats.pool_status = ConnectionPoolStatus.FAILED
            logger.error(f"❌ Failed to initialize DatabaseManager: {e}")
            raise
    
    async def close(self) -> None:
        """
        Close the database connection pool and cleanup resources
        
        **By George!** Proper cleanup when the cavalry retreats!
        """
        logger.info("🔄 Shutting down DatabaseManager...")
        
        # Cancel health check task
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close active transactions
        for tx_id in list(self._active_transactions.keys()):
            logger.warning(f"⚠️ Closing active transaction: {tx_id}")
            # Transactions will be automatically rolled back when connections close
        
        # Close the connection pool
        if self._pool and not self._pool.is_closing():
            await self._pool.close()
            logger.info("🔄 Connection pool closed")
        
        self._stats.pool_status = ConnectionPoolStatus.CLOSED
        self._initialized = False
        
        logger.info("✅ DatabaseManager shutdown complete")
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncContextManager[Connection]:
        """
        Get a database connection from the pool
        
        **BULLY!** The single, reliable way to get a database connection!
        """
        if not self._initialized or not self._pool:
            raise RuntimeError("DatabaseManager not initialized")
        
        start_time = time.time()
        connection = None
        
        try:
            # Acquire connection from pool
            connection = await self._pool.acquire()
            self._stats.active_connections += 1
            
            logger.debug(f"🔗 Acquired database connection: {id(connection)}")
            yield connection
            
        except Exception as e:
            self._stats.failed_queries += 1
            self._add_error(f"Connection error: {str(e)}")
            logger.error(f"❌ Database connection error: {e}")
            raise
        
        finally:
            # Always return connection to pool
            if connection:
                try:
                    await self._pool.release(connection)
                    self._stats.active_connections -= 1
                    execution_time = time.time() - start_time
                    logger.debug(f"🔄 Released database connection: {id(connection)} ({execution_time:.3f}s)")
                except Exception as e:
                    logger.error(f"❌ Error releasing connection: {e}")
    
    async def execute_query(self, query: str, *args, **kwargs) -> QueryResult:
        """
        Execute a database query with automatic retry and monitoring
        
        **By George!** The robust way to execute any database query!
        """
        start_time = time.time()
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        
        for attempt in range(self.config.retry_attempts):
            try:
                async with self.get_connection() as conn:
                    if self.config.enable_query_logging:
                        logger.debug(f"🔍 Executing query [{query_hash}]: {query[:100]}...")
                    
                    # ROOSEVELT FIX: Set RLS context on this connection before executing query
                    # This ensures RLS context persists for the duration of this query
                    # CRITICAL: Use false (session-level) not true (transaction-local)!
                    rls_context = kwargs.get('rls_context', {})
                    if rls_context:
                        user_id = rls_context.get('user_id', '')
                        user_role = rls_context.get('user_role', 'admin')
                        
                        # set_config() requires a string value, not NULL
                        # For global/admin operations, use empty string
                        user_id_str = '' if user_id is None else str(user_id)
                        
                        await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id_str)
                        await conn.execute("SELECT set_config('app.current_user_role', $1, false)", user_role)
                        logger.debug(f"🔍 Set RLS context on connection {id(conn)}: user_id={user_id_str}, role={user_role}")
                    else:
                        logger.debug(f"🔍 No RLS context provided for query on connection {id(conn)}")
                    
                    # Execute the query
                    if 'fetch' in kwargs and kwargs['fetch'] == 'all':
                        result = await conn.fetch(query, *args)
                    elif 'fetch' in kwargs and kwargs['fetch'] == 'one':
                        result = await conn.fetchrow(query, *args)
                    elif 'fetch' in kwargs and kwargs['fetch'] == 'val':
                        result = await conn.fetchval(query, *args)
                    else:
                        # Default to execute for INSERT/UPDATE/DELETE
                        result = await conn.execute(query, *args)
                    
                    execution_time = time.time() - start_time
                    self._update_query_stats(execution_time, True)
                    
                    if self.config.enable_query_logging:
                        logger.debug(f"✅ Query [{query_hash}] completed in {execution_time:.3f}s")
                    
                    return QueryResult(
                        success=True,
                        data=result,
                        execution_time=execution_time,
                        query_hash=query_hash
                    )
                    
            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = str(e)
                
                # Check if this is a retryable error
                if attempt < self.config.retry_attempts - 1 and self._is_retryable_error(error_msg):
                    wait_time = self.config.retry_delay * (attempt + 1)
                    logger.warning(f"⚠️ Query [{query_hash}] failed (attempt {attempt + 1}/{self.config.retry_attempts}), retrying in {wait_time}s: {error_msg}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    self._update_query_stats(execution_time, False)
                    self._add_error(f"Query [{query_hash}] failed: {error_msg}")
                    logger.error(f"❌ Query [{query_hash}] failed after {self.config.retry_attempts} attempts: {error_msg}")
                    
                    return QueryResult(
                        success=False,
                        error=error_msg,
                        execution_time=execution_time,
                        query_hash=query_hash
                    )
        
        # This should never be reached, but just in case
        return QueryResult(
            success=False,
            error="Maximum retry attempts exceeded",
            query_hash=query_hash
        )
    
    async def execute_transaction(self, operations: List[Callable]) -> QueryResult:
        """
        Execute multiple operations in a database transaction
        
        **BULLY!** Atomic operations for data integrity!
        """
        transaction_id = f"tx_{int(time.time())}_{len(self._active_transactions)}"
        start_time = time.time()
        
        try:
            async with self.get_connection() as conn:
                async with conn.transaction():
                    # Track the transaction
                    tx_context = TransactionContext(
                        transaction_id=transaction_id,
                        connection_id=str(id(conn)),
                        started_at=datetime.now(),
                        operations=[],
                        is_active=True
                    )
                    self._active_transactions[transaction_id] = tx_context
                    
                    logger.debug(f"🔄 Started transaction: {transaction_id}")
                    
                    results = []
                    for i, operation in enumerate(operations):
                        try:
                            result = await operation(conn)
                            results.append(result)
                            tx_context.operations.append(f"operation_{i}")
                        except Exception as e:
                            logger.error(f"❌ Transaction {transaction_id} operation {i} failed: {e}")
                            raise
                    
                    # Transaction completed successfully
                    execution_time = time.time() - start_time
                    tx_context.is_active = False
                    
                    logger.info(f"✅ Transaction {transaction_id} completed successfully in {execution_time:.3f}s")
                    
                    return QueryResult(
                        success=True,
                        data=results,
                        execution_time=execution_time
                    )
                    
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            self._add_error(f"Transaction {transaction_id} failed: {error_msg}")
            logger.error(f"❌ Transaction {transaction_id} failed: {error_msg}")
            
            return QueryResult(
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
        
        finally:
            # Clean up transaction tracking
            if transaction_id in self._active_transactions:
                del self._active_transactions[transaction_id]
    
    def _is_retryable_error(self, error_msg: str) -> bool:
        """Check if an error is retryable"""
        retryable_errors = [
            "connection was closed",
            "connection does not exist",
            "another operation is in progress",
            "server closed the connection unexpectedly",
            "timeout",
            "connection refused"
        ]
        
        error_lower = error_msg.lower()
        return any(retryable in error_lower for retryable in retryable_errors)
    
    def _update_query_stats(self, execution_time: float, success: bool) -> None:
        """Update query execution statistics"""
        self._stats.total_queries_executed += 1
        
        if success:
            # Update average query time
            current_avg = self._stats.average_query_time
            total_queries = self._stats.total_queries_executed
            self._stats.average_query_time = ((current_avg * (total_queries - 1)) + execution_time) / total_queries
        else:
            self._stats.failed_queries += 1
        
        # Update error rate
        if self._stats.total_queries_executed > 0:
            self._stats.error_rate = (self._stats.failed_queries / self._stats.total_queries_executed) * 100
    
    def _add_error(self, error_msg: str) -> None:
        """Add an error to the history (keep last 100)"""
        timestamp = datetime.now().isoformat()
        self._error_history.append(f"[{timestamp}] {error_msg}")
        
        # Keep only last 100 errors
        if len(self._error_history) > 100:
            self._error_history = self._error_history[-100:]
    
    async def _health_check_loop(self) -> None:
        """Background task for monitoring database health"""
        logger.info("🏥 Starting database health monitoring")
        
        while self._initialized:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                logger.info("🔄 Health check loop cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Health check error: {e}")
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _perform_health_check(self) -> None:
        """Perform a database health check"""
        try:
            start_time = time.time()
            
            async with self.get_connection() as conn:
                # Simple health check query
                await conn.fetchval("SELECT 1")
            
            # Update health statistics
            self._stats.last_health_check = datetime.now()
            self._stats.uptime_seconds = int((datetime.now() - self._start_time).total_seconds())
            
            # Update pool statistics if available
            if self._pool:
                self._stats.total_connections = self._pool.get_size()
                self._stats.idle_connections = self._pool.get_idle_size()
            
            # Determine overall health status
            if self._stats.error_rate < 5.0:  # Less than 5% error rate
                self._stats.pool_status = ConnectionPoolStatus.HEALTHY
            elif self._stats.error_rate < 15.0:  # Less than 15% error rate
                self._stats.pool_status = ConnectionPoolStatus.DEGRADED
            else:
                self._stats.pool_status = ConnectionPoolStatus.FAILED
                
            logger.debug(f"🏥 Health check completed - Status: {self._stats.pool_status.value}")
            
        except Exception as e:
            self._stats.pool_status = ConnectionPoolStatus.FAILED
            self._add_error(f"Health check failed: {str(e)}")
            logger.error(f"❌ Database health check failed: {e}")
    
    def get_health_status(self) -> DatabaseHealth:
        """Get current database health status"""
        return DatabaseHealth(
            is_healthy=self._stats.pool_status == ConnectionPoolStatus.HEALTHY,
            connection_stats=self._stats,
            recent_errors=self._error_history[-10:],  # Last 10 errors
            performance_metrics={
                'average_query_time': self._stats.average_query_time,
                'error_rate': self._stats.error_rate,
                'uptime_seconds': self._stats.uptime_seconds,
                'total_queries': self._stats.total_queries_executed
            },
            last_successful_query=self._stats.last_health_check
        )
    
    def get_connection_stats(self) -> ConnectionStats:
        """Get current connection statistics"""
        return self._stats
    
    @property
    def is_initialized(self) -> bool:
        """Check if the database manager is initialized"""
        return self._initialized
    
    @property
    def is_healthy(self) -> bool:
        """Check if the database is healthy"""
        return self._stats.pool_status == ConnectionPoolStatus.HEALTHY


# Global database manager instance
_database_manager_instance: Optional[DatabaseManager] = None


async def get_database_manager(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """
    Get the global database manager instance
    
    **BULLY!** The single point of access to the database manager!
    Handles both main app and Celery worker environments!
    """
    global _database_manager_instance
    
    # Check if we're in a Celery worker environment
    import os
    is_celery_worker = os.getenv('CELERY_WORKER_RUNNING', 'false').lower() == 'true'
    
    if _database_manager_instance is None or (is_celery_worker and not _database_manager_instance.is_initialized):
        # Create fresh instance for Celery workers or if not initialized
        _database_manager_instance = DatabaseManager(config)
        try:
            await _database_manager_instance.initialize()
        except Exception as e:
            logger.error(f"❌ Failed to initialize DatabaseManager: {e}")
            # For Celery workers, try a simpler approach
            if is_celery_worker:
                logger.warning("⚠️ Celery worker detected - using simplified database manager")
                # Create a basic instance without full initialization
                _database_manager_instance = DatabaseManager(config)
    
    return _database_manager_instance


async def close_database_manager() -> None:
    """Close the global database manager instance"""
    global _database_manager_instance
    
    if _database_manager_instance:
        await _database_manager_instance.close()
        _database_manager_instance = None
