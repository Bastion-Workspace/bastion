"""
Data Workspace Service
Service for querying Data Workspaces via gRPC data-service
"""

import logging
import json
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Global service instance
_data_workspace_service_instance = None


class DataWorkspaceService:
    """
    Data Workspace Service for querying workspaces, schemas, and executing queries
    
    This service handles all Data Workspace operations via gRPC data-service.
    """
    
    def __init__(self):
        self._data_client = None
        logger.info("Data Workspace Service initialized")
    
    async def initialize(self):
        """Initialize the Data Workspace service with gRPC client"""
        try:
            # Import DataWorkspaceGRPCClient from backend (tools-service has access via Dockerfile)
            from backend.services.data_workspace_grpc_client import DataWorkspaceGRPCClient
            
            self._data_client = DataWorkspaceGRPCClient()
            logger.info("Data Workspace Service connected to data-service")
            
        except Exception as e:
            logger.error(f"Failed to initialize Data Workspace Service: {e}")
            raise
    
    async def _ensure_initialized(self):
        """Ensure service is initialized"""
        if self._data_client is None:
            await self.initialize()
    
    async def list_workspaces(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all workspaces for a user
        
        Args:
            user_id: User ID to list workspaces for
            
        Returns:
            List of workspace dictionaries with workspace_id, name, description, etc.
        """
        try:
            await self._ensure_initialized()
            
            workspaces = await self._data_client.list_workspaces(user_id)
            
            # Format for tool service response
            formatted_workspaces = []
            for ws in workspaces:
                formatted_workspaces.append({
                    'workspace_id': ws.get('workspace_id'),
                    'name': ws.get('name', ''),
                    'description': ws.get('description', ''),
                    'icon': ws.get('icon', ''),
                    'color': ws.get('color', ''),
                    'is_pinned': ws.get('is_pinned', False)
                })
            
            logger.info(f"Listed {len(formatted_workspaces)} workspaces for user {user_id}")
            return formatted_workspaces
            
        except Exception as e:
            logger.error(f"Failed to list workspaces: {e}")
            raise
    
    async def get_workspace_schema(
        self, 
        workspace_id: str, 
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get complete schema for a workspace (all databases, tables, and columns)
        
        Args:
            workspace_id: Workspace ID to get schema for
            user_id: User ID for access control
            
        Returns:
            Dictionary with workspace_id and list of tables with their schemas
        """
        try:
            await self._ensure_initialized()
            
            # Get all databases in workspace
            databases = await self._data_client.list_databases(
                workspace_id=workspace_id,
                user_id=user_id
            )
            
            # Collect all tables across all databases
            all_tables = []
            for database in databases:
                database_id = database.get('database_id')
                database_name = database.get('name', '')
                
                # Get tables in this database
                tables = await self._data_client.list_tables(
                    database_id=database_id,
                    user_id=user_id
                )
                
                # Get schema for each table
                for table in tables:
                    table_id = table.get('table_id')
                    
                    # Get full table details including schema
                    table_details = await self._data_client.get_table(
                        table_id=table_id,
                        user_id=user_id
                    )
                    
                    if table_details:
                        # Parse schema JSON (format: {"columns": [{"name", "type", "nullable", "description"?}, ...]})
                        schema_json = table_details.get('schema_json', '{}')
                        if isinstance(schema_json, str):
                            try:
                                schema = json.loads(schema_json)
                            except json.JSONDecodeError:
                                schema = {}
                        else:
                            schema = schema_json
                        schema_cols = schema.get('columns', []) if isinstance(schema, dict) else []
                        columns = []
                        for col in schema_cols:
                            if not isinstance(col, dict):
                                continue
                            columns.append({
                                'name': col.get('name', ''),
                                'type': col.get('type', 'text'),
                                'is_nullable': col.get('nullable', True),
                                'description': col.get('description', '') or ''
                            })
                        # Parse metadata_json for business context (agents)
                        metadata_json = table_details.get('metadata_json', '') or ''
                        if isinstance(metadata_json, str) and metadata_json.strip():
                            try:
                                metadata_json = json.loads(metadata_json) if metadata_json.strip() else {}
                            except json.JSONDecodeError:
                                metadata_json = {}
                        else:
                            metadata_json = {}
                        if not isinstance(metadata_json, dict):
                            metadata_json = {}
                        all_tables.append({
                            'table_id': table_id,
                            'name': table.get('name', ''),
                            'description': table.get('description', ''),
                            'database_id': database_id,
                            'database_name': database_name,
                            'columns': columns,
                            'row_count': table_details.get('row_count', 0),
                            'metadata_json': metadata_json
                        })
            
            logger.info(f"Retrieved schema for workspace {workspace_id}: {len(all_tables)} tables")
            
            return {
                'workspace_id': workspace_id,
                'tables': all_tables,
                'total_tables': len(all_tables)
            }
            
        except Exception as e:
            logger.error(f"Failed to get workspace schema: {e}")
            raise
    
    async def query_workspace(
        self,
        workspace_id: str,
        query: str,
        query_type: str,
        user_id: str,
        limit: int = 100,
        params: Optional[List[Any]] = None,
        read_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a query against a workspace (SQL or natural language)
        
        Args:
            workspace_id: Workspace ID to query
            query: SQL query or natural language query
            query_type: "sql" or "natural_language"
            user_id: User ID for access control
            limit: Maximum rows to return (default: 100)
            params: Optional list of values for $1, $2, ... (SQL only)
            
        Returns:
            Dictionary with query results, column names, execution time, rows_affected, returning_rows, etc.
        """
        try:
            await self._ensure_initialized()
            
            if query_type == "sql":
                # Execute SQL query directly
                result = await self._data_client.execute_sql_query(
                    workspace_id=workspace_id,
                    query=query,
                    user_id=user_id,
                    limit=limit,
                    params=params,
                    read_only=read_only,
                )
            elif query_type == "natural_language":
                # Execute natural language query (converts to SQL)
                result = await self._data_client.execute_nl_query(
                    workspace_id=workspace_id,
                    natural_query=query,
                    user_id=user_id,
                    read_only=read_only,
                )
            else:
                raise ValueError(f"Invalid query_type: {query_type}. Must be 'sql' or 'natural_language'")
            
            has_arrow = bool(result.get('has_arrow_data')) and result.get('arrow_results')
            if has_arrow:
                results = []
            else:
                results_json = result.get('results_json', '[]')
                if isinstance(results_json, str):
                    try:
                        results = json.loads(results_json)
                    except json.JSONDecodeError:
                        results = []
                else:
                    results = results_json

            returning_rows = None
            if result.get('returning_rows_json'):
                try:
                    returning_rows = json.loads(result['returning_rows_json'])
                except (json.JSONDecodeError, TypeError):
                    pass
            
            return {
                'success': result.get('error_message') is None,
                'column_names': result.get('column_names', []),
                'results': results,
                'result_count': result.get('result_count', len(results)),
                'execution_time_ms': result.get('execution_time_ms', 0),
                'generated_sql': result.get('generated_sql', ''),
                'error_message': result.get('error_message'),
                'rows_affected': result.get('rows_affected', 0),
                'returning_rows': returning_rows,
                'has_arrow_data': bool(result.get('has_arrow_data')),
                'arrow_results': result.get('arrow_results') or b'',
            }
            
        except Exception as e:
            logger.error(f"Failed to query workspace: {e}")
            return {
                'success': False,
                'column_names': [],
                'results': [],
                'result_count': 0,
                'execution_time_ms': 0,
                'generated_sql': '',
                'error_message': str(e),
                'rows_affected': 0,
                'returning_rows': None,
                'has_arrow_data': False,
                'arrow_results': b'',
            }


async def get_data_workspace_service() -> DataWorkspaceService:
    """Get global Data Workspace service instance"""
    global _data_workspace_service_instance
    if _data_workspace_service_instance is None:
        _data_workspace_service_instance = DataWorkspaceService()
        await _data_workspace_service_instance.initialize()
    return _data_workspace_service_instance
