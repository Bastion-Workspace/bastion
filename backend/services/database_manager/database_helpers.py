"""
Database Helper Functions
Convenience functions for common database operations using the DatabaseManager

**BULLY!** Simple, robust database operations for all services!
Automatically handles both main app and Celery worker environments!
"""

import logging
import os
from typing import List, Dict, Any, Optional, Union, Set
from .database_manager_service import get_database_manager
from .models.database_models import QueryResult
from .celery_database_helpers import (
    celery_fetch_one, celery_fetch_all, celery_fetch_value, celery_execute, is_celery_worker
)

logger = logging.getLogger(__name__)


async def fetch_all(query: str, *args, rls_context: Dict[str, str] = None) -> List[Dict[str, Any]]:
    """
    Execute a query and fetch all results
    
    **By George!** Simple way to get all rows from a query!
    Automatically handles both main app and Celery worker environments!
    """
    # **BULLY!** Force simple connection approach for now to fix event loop issues!
    try:
        import threading
        if 'ForkPoolWorker' in threading.current_thread().name:
            logger.debug("Detected ForkPoolWorker - using simple connection")
            return await celery_fetch_all(query, *args, rls_context=rls_context)
    except:
        pass
    
    if is_celery_worker():
        # **BULLY!** Use simple connection for Celery workers!
        logger.debug("Using Celery helpers (is_celery_worker=True)")
        return await celery_fetch_all(query, *args, rls_context=rls_context)
    else:
        # Use DatabaseManager for main app
        logger.debug("Using DatabaseManager (is_celery_worker=False)")
        try:
            db_manager = await get_database_manager()
            result = await db_manager.execute_query(query, *args, fetch='all', rls_context=rls_context)
            
            if result.success:
                # Convert asyncpg Records to dictionaries
                return [dict(record) for record in result.data] if result.data else []
            else:
                logger.error(f"❌ fetch_all failed: {result.error}")
                raise Exception(f"Database query failed: {result.error}")
        except Exception as e:
            logger.error(f"❌ DatabaseManager failed, falling back to Celery helpers: {e}")
            return await celery_fetch_all(query, *args, rls_context=rls_context)


async def fetch_one(query: str, *args, rls_context: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
    """
    Execute a query and fetch one result
    
    **BULLY!** Get a single row from the database!
    Automatically handles both main app and Celery worker environments!
    """
    # **BULLY!** Force simple connection approach for now to fix event loop issues!
    # TODO: Improve Celery detection once working
    try:
        import threading
        if 'ForkPoolWorker' in threading.current_thread().name:
            logger.debug("Detected ForkPoolWorker - using simple connection")
            return await celery_fetch_one(query, *args, rls_context=rls_context)
    except:
        pass
    
    if is_celery_worker():
        # **BULLY!** Use simple connection for Celery workers!
        logger.debug("Using Celery helpers (is_celery_worker=True)")
        return await celery_fetch_one(query, *args, rls_context=rls_context)
    else:
        # Use DatabaseManager for main app
        logger.debug("Using DatabaseManager (is_celery_worker=False)")
        try:
            db_manager = await get_database_manager()
            result = await db_manager.execute_query(query, *args, fetch='one', rls_context=rls_context)
            
            if result.success:
                return dict(result.data) if result.data else None
            else:
                logger.error(f"❌ fetch_one failed: {result.error}")
                raise Exception(f"Database query failed: {result.error}")
        except Exception as e:
            logger.error(f"❌ DatabaseManager failed, falling back to Celery helpers: {e}")
            return await celery_fetch_one(query, *args, rls_context=rls_context)


async def fetch_value(query: str, *args, rls_context: Dict[str, str] = None) -> Any:
    """
    Execute a query and fetch a single value
    
    **By George!** Get just one value from the database!
    Automatically handles both main app and Celery worker environments!
    """
    # **BULLY!** Force simple connection approach for Celery workers to fix event loop issues!
    try:
        import threading
        if 'ForkPoolWorker' in threading.current_thread().name:
            logger.debug("Detected ForkPoolWorker - using simple connection")
            return await celery_fetch_value(query, *args, rls_context=rls_context)
    except:
        pass
    
    if is_celery_worker():
        # **BULLY!** Use simple connection for Celery workers!
        logger.debug("Using Celery helpers (is_celery_worker=True)")
        return await celery_fetch_value(query, *args, rls_context=rls_context)
    else:
        # Use DatabaseManager for main app
        logger.debug("Using DatabaseManager (is_celery_worker=False)")
        try:
            db_manager = await get_database_manager()
            result = await db_manager.execute_query(query, *args, fetch='val', rls_context=rls_context)
            
            if result.success:
                return result.data
            else:
                logger.error(f"❌ fetch_value failed: {result.error}")
                raise Exception(f"Database query failed: {result.error}")
        except Exception as e:
            logger.error(f"❌ DatabaseManager failed, falling back to Celery helpers: {e}")
            return await celery_fetch_value(query, *args, rls_context=rls_context)


async def execute(query: str, *args, rls_context: Dict[str, str] = None) -> str:
    """
    Execute a query (INSERT/UPDATE/DELETE) and return the result status
    
    **BULLY!** Execute any modification query!
    Automatically handles both main app and Celery worker environments!
    """
    # **BULLY!** Force simple connection approach for now to fix event loop issues!
    try:
        import threading
        if 'ForkPoolWorker' in threading.current_thread().name:
            logger.debug("Detected ForkPoolWorker - using simple connection")
            return await celery_execute(query, *args, rls_context=rls_context)
    except:
        pass
    
    if is_celery_worker():
        # **BULLY!** Use simple connection for Celery workers!
        logger.debug("Using Celery helpers (is_celery_worker=True)")
        return await celery_execute(query, *args, rls_context=rls_context)
    else:
        # Use DatabaseManager for main app
        logger.debug("Using DatabaseManager (is_celery_worker=False)")
        try:
            db_manager = await get_database_manager()
            result = await db_manager.execute_query(query, *args, rls_context=rls_context)
            
            if result.success:
                return result.data  # Returns something like "INSERT 0 1" or "UPDATE 3"
            else:
                logger.error(f"❌ execute failed: {result.error}")
                raise Exception(f"Database query failed: {result.error}")
        except Exception as e:
            logger.error(f"❌ DatabaseManager failed, falling back to Celery helpers: {e}")
            return await celery_execute(query, *args, rls_context=rls_context)


async def execute_transaction(operations: List) -> Any:
    """
    Execute multiple operations in a transaction
    
    **By George!** Atomic operations for data consistency!
    """
    db_manager = await get_database_manager()
    result = await db_manager.execute_transaction(operations)
    
    if result.success:
        return result.data
    else:
        logger.error(f"❌ execute_transaction failed: {result.error}")
        raise Exception(f"Database transaction failed: {result.error}")


async def check_database_health() -> Dict[str, Any]:
    """
    Check the health of the database connection
    
    **BULLY!** Monitor the cavalry's health status!
    """
    try:
        db_manager = await get_database_manager()
        health = db_manager.get_health_status()
        
        return {
            "is_healthy": health.is_healthy,
            "status": health.connection_stats.pool_status.value,
            "total_connections": health.connection_stats.total_connections,
            "active_connections": health.connection_stats.active_connections,
            "total_queries": health.connection_stats.total_queries_executed,
            "error_rate": health.connection_stats.error_rate,
            "average_query_time": health.connection_stats.average_query_time,
            "uptime_seconds": health.connection_stats.uptime_seconds,
            "recent_errors": health.recent_errors[-3:] if health.recent_errors else []
        }
    except Exception as e:
        logger.error(f"❌ Database health check failed: {e}")
        return {
            "is_healthy": False,
            "status": "failed",
            "error": str(e)
        }


# Convenience functions for common patterns
async def insert_and_return_id(table: str, data: Dict[str, Any], id_column: str = "id") -> Any:
    """
    Insert a record and return the generated ID
    
    **BULLY!** Insert and get the ID in one go!
    """
    columns = list(data.keys())
    placeholders = [f"${i+1}" for i in range(len(columns))]
    values = list(data.values())
    
    query = f"""
        INSERT INTO {table} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        RETURNING {id_column}
    """
    
    return await fetch_value(query, *values)


async def update_by_id(table: str, record_id: Any, data: Dict[str, Any], id_column: str = "id") -> bool:
    """
    Update a record by ID
    
    **By George!** Simple update by ID!
    """
    if not data:
        return False
    
    set_clauses = [f"{col} = ${i+2}" for i, col in enumerate(data.keys())]
    values = list(data.values())
    
    query = f"""
        UPDATE {table}
        SET {', '.join(set_clauses)}
        WHERE {id_column} = $1
    """
    
    result = await execute(query, record_id, *values)
    return "UPDATE 1" in result


async def delete_by_id(table: str, record_id: Any, id_column: str = "id") -> bool:
    """
    Delete a record by ID
    
    **BULLY!** Remove a record by ID!
    """
    query = f"DELETE FROM {table} WHERE {id_column} = $1"
    result = await execute(query, record_id)
    return "DELETE 1" in result


async def count_records(table: str, where_clause: str = "", *args) -> int:
    """
    Count records in a table
    
    **By George!** Count the cavalry!
    """
    query = f"SELECT COUNT(*) FROM {table}"
    if where_clause:
        query += f" WHERE {where_clause}"
    
    return await fetch_value(query, *args)


async def record_exists(table: str, where_clause: str, *args) -> bool:
    """
    Check if a record exists

    **BULLY!** Check if something exists!
    """
    query = f"SELECT EXISTS(SELECT 1 FROM {table} WHERE {where_clause})"
    return await fetch_value(query, *args)


async def get_document_ids_by_metadata(
    series: Optional[str] = None,
    image_type: Optional[str] = None,
    author: Optional[str] = None,
    date: Optional[str] = None,
    rls_context: Optional[Dict[str, str]] = None,
) -> Set[str]:
    """
    Fast SQL query for document_ids matching metadata filters (document_metadata.metadata_json).
    Used for hybrid collection search: intersect with vector results.
    """
    conditions = []
    params = []
    idx = 1
    if series:
        conditions.append(f"(metadata_json->>'series' ILIKE ${idx})")
        params.append(f"%{series}%")
        idx += 1
    if image_type:
        conditions.append(f"(metadata_json->>'image_type' = ${idx} OR metadata_json->>'type' = ${idx})")
        params.append(image_type.lower())
        idx += 1
    if author:
        conditions.append(f"(metadata_json->>'author' ILIKE ${idx})")
        params.append(f"%{author}%")
        idx += 1
    if date:
        conditions.append(f"(metadata_json->>'date' LIKE ${idx})")
        params.append(f"{date}%")
        idx += 1
    if not conditions:
        return set()
    where = " AND ".join(conditions)
    query = f"SELECT document_id FROM document_metadata WHERE {where}"
    rows = await fetch_all(query, *params, rls_context=rls_context)
    return {row["document_id"] for row in rows if row.get("document_id")}
