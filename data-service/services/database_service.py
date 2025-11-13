import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
import json

from db.connection_manager import DatabaseConnectionManager

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for managing custom databases within workspaces"""
    
    def __init__(self, db_manager: DatabaseConnectionManager):
        self.db = db_manager
    
    async def create_database(
        self,
        workspace_id: str,
        name: str,
        description: Optional[str] = None,
        source_type: str = "imported"
    ) -> Dict[str, Any]:
        """Create a new database in a workspace"""
        try:
            database_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            query = """
                INSERT INTO custom_databases 
                (database_id, workspace_id, name, description, source_type, 
                 table_count, total_rows, metadata_json, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING database_id, workspace_id, name, description, source_type,
                          table_count, total_rows, metadata_json, created_at, updated_at
            """
            
            row = await self.db.fetchrow(
                query,
                database_id, workspace_id, name, description, source_type,
                0, 0, json.dumps({}), now, now
            )
            
            logger.info(f"Created database: {database_id} in workspace: {workspace_id}")
            return self._row_to_dict(row)
            
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            raise
    
    async def list_databases(self, workspace_id: str) -> List[Dict[str, Any]]:
        """List all databases in a workspace"""
        try:
            query = """
                SELECT database_id, workspace_id, name, description, source_type,
                       table_count, total_rows, metadata_json, created_at, updated_at
                FROM custom_databases
                WHERE workspace_id = $1
                ORDER BY created_at DESC
            """
            
            rows = await self.db.fetch(query, workspace_id)
            return [self._row_to_dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to list databases: {e}")
            raise
    
    async def get_database(self, database_id: str) -> Optional[Dict[str, Any]]:
        """Get a single database by ID"""
        try:
            query = """
                SELECT database_id, workspace_id, name, description, source_type,
                       table_count, total_rows, metadata_json, created_at, updated_at
                FROM custom_databases
                WHERE database_id = $1
            """
            
            row = await self.db.fetchrow(query, database_id)
            return self._row_to_dict(row) if row else None
            
        except Exception as e:
            logger.error(f"Failed to get database {database_id}: {e}")
            raise
    
    async def update_database_stats(self, database_id: str) -> bool:
        """Update table count and total rows for a database"""
        try:
            # Count tables
            table_count_query = """
                SELECT COUNT(*) FROM custom_tables WHERE database_id = $1
            """
            table_count = await self.db.fetchval(table_count_query, database_id)
            
            # Sum total rows
            total_rows_query = """
                SELECT COALESCE(SUM(row_count), 0) 
                FROM custom_tables 
                WHERE database_id = $1
            """
            total_rows = await self.db.fetchval(total_rows_query, database_id)
            
            # Update database
            update_query = """
                UPDATE custom_databases
                SET table_count = $2, total_rows = $3, updated_at = $4
                WHERE database_id = $1
            """
            await self.db.execute(
                update_query,
                database_id,
                table_count,
                total_rows,
                datetime.utcnow()
            )
            
            logger.info(f"Updated database stats: {database_id} - {table_count} tables, {total_rows} rows")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update database stats {database_id}: {e}")
            raise
    
    async def delete_database(self, database_id: str) -> bool:
        """Delete a database and all associated tables/data"""
        try:
            query = "DELETE FROM custom_databases WHERE database_id = $1"
            result = await self.db.execute(query, database_id)
            
            deleted = result.split()[-1] != '0'
            
            if deleted:
                logger.info(f"Deleted database: {database_id}")
            else:
                logger.warning(f"Database not found for deletion: {database_id}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete database {database_id}: {e}")
            raise
    
    async def get_database_schema(self, database_id: str) -> Dict[str, Any]:
        """Get schema information for all tables in a database"""
        try:
            query = """
                SELECT table_id, name, description, row_count, schema_json
                FROM custom_tables
                WHERE database_id = $1
                ORDER BY name
            """
            
            rows = await self.db.fetch(query, database_id)
            
            tables = []
            for row in rows:
                tables.append({
                    'table_id': row['table_id'],
                    'name': row['name'],
                    'description': row['description'],
                    'row_count': row['row_count'],
                    'schema': json.loads(row['schema_json']) if isinstance(row['schema_json'], str) else row['schema_json']
                })
            
            return {
                'database_id': database_id,
                'tables': tables,
                'table_count': len(tables)
            }
            
        except Exception as e:
            logger.error(f"Failed to get database schema {database_id}: {e}")
            raise
    
    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert database row to dictionary"""
        if not row:
            return {}
        
        return {
            'database_id': row['database_id'],
            'workspace_id': row['workspace_id'],
            'name': row['name'],
            'description': row['description'],
            'source_type': row['source_type'],
            'table_count': row['table_count'],
            'total_rows': row['total_rows'],
            'metadata_json': row['metadata_json'] if isinstance(row['metadata_json'], str) else json.dumps(row['metadata_json']),
            'created_at': row['created_at'].isoformat() if row['created_at'] else None,
            'updated_at': row['updated_at'].isoformat() if row['updated_at'] else None
        }





