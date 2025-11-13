"""
Database Manager Models
Data structures and configuration for centralized database management
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class ConnectionPoolStatus(Enum):
    """Connection pool status enumeration"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass
class DatabaseConfig:
    """Database configuration for the centralized manager"""
    database_url: str
    min_pool_size: int = 5
    max_pool_size: int = 20
    command_timeout: int = 60
    max_queries: int = 50000
    max_inactive_connection_lifetime: int = 300
    connection_max_age: int = 3600
    retry_attempts: int = 3
    retry_delay: float = 1.0
    health_check_interval: int = 30
    enable_query_logging: bool = False
    enable_performance_monitoring: bool = True


@dataclass
class ConnectionStats:
    """Connection pool statistics and health metrics"""
    pool_status: ConnectionPoolStatus
    total_connections: int
    active_connections: int
    idle_connections: int
    pending_operations: int
    total_queries_executed: int
    failed_queries: int
    average_query_time: float
    last_health_check: datetime
    uptime_seconds: int
    error_rate: float


@dataclass
class QueryResult:
    """Standardized query result wrapper"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    affected_rows: Optional[int] = None
    query_hash: Optional[str] = None


@dataclass
class TransactionContext:
    """Context for database transactions"""
    transaction_id: str
    connection_id: str
    started_at: datetime
    operations: List[str]
    is_active: bool = True
    isolation_level: str = "read_committed"


@dataclass
class DatabaseHealth:
    """Overall database health status"""
    is_healthy: bool
    connection_stats: ConnectionStats
    recent_errors: List[str]
    performance_metrics: Dict[str, float]
    last_successful_query: datetime
    database_version: Optional[str] = None
