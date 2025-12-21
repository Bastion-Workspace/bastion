import grpc
import logging
import json
import os
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Import generated protobuf code
try:
    import sys
    sys.path.append('/app')
    from protos import data_service_pb2, data_service_pb2_grpc
    GRPC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Data service gRPC protos not available: {e}")
    GRPC_AVAILABLE = False
    data_service_pb2 = None
    data_service_pb2_grpc = None


class DataWorkspaceGRPCClient:
    """
    gRPC client for communicating with data-service microservice
    """
    
    def __init__(self):
        if not GRPC_AVAILABLE:
            raise RuntimeError("Data service gRPC protos not available")
        
        self.host = os.getenv("DATA_SERVICE_HOST", "data-service")
        self.port = os.getenv("DATA_SERVICE_PORT", "50054")
        self.channel = None
        self.stub = None
        self._connect()
    
    def _connect(self):
        """Establish gRPC connection"""
        try:
            address = f"{self.host}:{self.port}"
            self.channel = grpc.aio.insecure_channel(address)
            self.stub = data_service_pb2_grpc.DataServiceStub(self.channel)
            logger.info(f"Connected to data-service at {address}")
        except Exception as e:
            logger.error(f"Failed to connect to data-service: {e}")
            raise
    
    async def close(self):
        """Close gRPC connection"""
        if self.channel:
            await self.channel.close()
    
    # Workspace methods
    async def create_workspace(
        self,
        user_id: str,
        name: str,
        description: Optional[str] = None,
        icon: Optional[str] = None,
        color: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new workspace"""
        try:
            request = data_service_pb2.CreateWorkspaceRequest(
                user_id=user_id,
                name=name,
                description=description or "",
                icon=icon or "",
                color=color or ""
            )
            
            response = await self.stub.CreateWorkspace(request)
            return self._workspace_response_to_dict(response)
            
        except grpc.RpcError as e:
            logger.error(f"gRPC error creating workspace: {e}")
            raise
    
    async def list_workspaces(self, user_id: str) -> List[Dict[str, Any]]:
        """List workspaces for a user"""
        try:
            request = data_service_pb2.ListWorkspacesRequest(user_id=user_id)
            response = await self.stub.ListWorkspaces(request)
            return [self._workspace_response_to_dict(ws) for ws in response.workspaces]
        except grpc.RpcError as e:
            logger.error(f"gRPC error listing workspaces: {e}")
            raise
    
    async def get_workspace(
        self, 
        workspace_id: str,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get a single workspace"""
        try:
            request = data_service_pb2.GetWorkspaceRequest(
                workspace_id=workspace_id,
                user_id=user_id or "",
                user_team_ids=user_team_ids or []
            )
            response = await self.stub.GetWorkspace(request)
            return self._workspace_response_to_dict(response)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return None
            logger.error(f"gRPC error getting workspace: {e}")
            raise
    
    async def update_workspace(
        self,
        workspace_id: str,
        user_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        icon: Optional[str] = None,
        color: Optional[str] = None,
        is_pinned: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        """Update a workspace"""
        try:
            request = data_service_pb2.UpdateWorkspaceRequest(
                workspace_id=workspace_id,
                user_id=user_id,
                name=name or "",
                description=description or "",
                icon=icon or "",
                color=color or "",
                is_pinned=is_pinned if is_pinned is not None else False
            )
            
            response = await self.stub.UpdateWorkspace(request)
            return self._workspace_response_to_dict(response)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return None
            logger.error(f"gRPC error updating workspace: {e}")
            raise
    
    async def delete_workspace(
        self, 
        workspace_id: str,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> bool:
        """Delete a workspace"""
        try:
            request = data_service_pb2.DeleteWorkspaceRequest(
                workspace_id=workspace_id,
                user_id=user_id or "",
                user_team_ids=user_team_ids or []
            )
            response = await self.stub.DeleteWorkspace(request)
            return response.success
        except grpc.RpcError as e:
            logger.error(f"gRPC error deleting workspace: {e}")
            raise
    
    # Database methods
    async def create_database(
        self,
        workspace_id: str,
        name: str,
        user_id: str,
        description: Optional[str] = None,
        source_type: str = "imported"
    ) -> Dict[str, Any]:
        """Create a new database"""
        try:
            request = data_service_pb2.CreateDatabaseRequest(
                workspace_id=workspace_id,
                name=name,
                user_id=user_id,
                description=description or "",
                source_type=source_type
            )
            
            response = await self.stub.CreateDatabase(request)
            return self._database_response_to_dict(response)
        except grpc.RpcError as e:
            logger.error(f"gRPC error creating database: {e}")
            raise
    
    async def list_databases(
        self, 
        workspace_id: str,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List databases in a workspace"""
        try:
            request = data_service_pb2.ListDatabasesRequest(
                workspace_id=workspace_id,
                user_id=user_id or "",
                user_team_ids=user_team_ids or []
            )
            response = await self.stub.ListDatabases(request)
            return [self._database_response_to_dict(db) for db in response.databases]
        except grpc.RpcError as e:
            logger.error(f"gRPC error listing databases: {e}")
            raise
    
    async def get_database(
        self, 
        database_id: str,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get a single database"""
        try:
            request = data_service_pb2.GetDatabaseRequest(
                database_id=database_id,
                user_id=user_id or "",
                user_team_ids=user_team_ids or []
            )
            response = await self.stub.GetDatabase(request)
            return self._database_response_to_dict(response)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return None
            logger.error(f"gRPC error getting database: {e}")
            raise
    
    async def delete_database(
        self, 
        database_id: str,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> bool:
        """Delete a database"""
        try:
            request = data_service_pb2.DeleteDatabaseRequest(
                database_id=database_id,
                user_id=user_id or "",
                user_team_ids=user_team_ids or []
            )
            response = await self.stub.DeleteDatabase(request)
            return response.success
        except grpc.RpcError as e:
            logger.error(f"gRPC error deleting database: {e}")
            raise
    
    # Import methods
    async def preview_import(
        self,
        file_path: str,
        file_type: str,
        preview_rows: int = 10
    ) -> Dict[str, Any]:
        """Preview import file"""
        try:
            request = data_service_pb2.PreviewImportRequest(
                workspace_id="",  # Not needed for preview
                file_path=file_path,
                file_type=file_type,
                preview_rows=preview_rows
            )
            
            response = await self.stub.PreviewImport(request)
            
            return {
                'column_names': list(response.column_names),
                'inferred_types': [json.loads(t) for t in response.inferred_types],
                'preview_data': json.loads(response.preview_data_json),
                'estimated_rows': response.estimated_rows,
                'total_columns': len(response.column_names)
            }
        except grpc.RpcError as e:
            logger.error(f"gRPC error previewing import: {e}")
            raise
    
    async def execute_import(
        self,
        workspace_id: str,
        database_id: str,
        table_name: str,
        file_path: str,
        user_id: str,
        field_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Execute import job"""
        try:
            request = data_service_pb2.ExecuteImportRequest(
                workspace_id=workspace_id,
                database_id=database_id,
                table_name=table_name,
                file_path=file_path,
                user_id=user_id,
                field_mapping_json=json.dumps(field_mapping) if field_mapping else ""
            )
            
            response = await self.stub.ExecuteImport(request)
            return self._import_job_response_to_dict(response)
        except grpc.RpcError as e:
            logger.error(f"gRPC error executing import: {e}")
            raise
    
    async def get_import_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get import job status"""
        try:
            request = data_service_pb2.GetImportStatusRequest(job_id=job_id)
            response = await self.stub.GetImportStatus(request)
            return self._import_job_response_to_dict(response)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return None
            logger.error(f"gRPC error getting import status: {e}")
            raise
    
    # Table data methods
    async def get_table_data(
        self,
        table_id: str,
        offset: int = 0,
        limit: int = 100,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get table data"""
        try:
            request = data_service_pb2.GetTableDataRequest(
                table_id=table_id,
                offset=offset,
                limit=limit,
                user_id=user_id or "",
                user_team_ids=user_team_ids or []
            )
            
            response = await self.stub.GetTableData(request)
            
            return {
                'table_id': response.table_id,
                'rows': [
                    {
                        'row_id': row.row_id,
                        'row_data': json.loads(row.row_data_json),
                        'row_index': row.row_index,
                        'row_color': row.row_color if row.row_color else None
                    }
                    for row in response.rows
                ],
                'total_rows': response.total_rows,
                'offset': offset,
                'limit': limit,
                'table_schema': json.loads(response.schema_json)
            }
        except grpc.RpcError as e:
            logger.error(f"gRPC error getting table data: {e}")
            raise
    
    # Table management methods
    async def create_table(
        self,
        database_id: str,
        name: str,
        user_id: str,
        description: Optional[str] = None,
        table_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new table"""
        try:
            request = data_service_pb2.CreateTableRequest(
                database_id=database_id,
                name=name,
                user_id=user_id,
                description=description or "",
                schema_json=json.dumps(table_schema) if table_schema else "{}"
            )
            
            response = await self.stub.CreateTable(request)
            return self._table_response_to_dict(response)
            
        except grpc.RpcError as e:
            logger.error(f"gRPC error creating table: {e}")
            raise
    
    async def list_tables(
        self, 
        database_id: str,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List tables in a database"""
        try:
            request = data_service_pb2.ListTablesRequest(
                database_id=database_id,
                user_id=user_id or "",
                user_team_ids=user_team_ids or []
            )
            response = await self.stub.ListTables(request)
            return [self._table_response_to_dict(table) for table in response.tables]
        except grpc.RpcError as e:
            logger.error(f"gRPC error listing tables: {e}")
            raise
    
    async def get_table(
        self, 
        table_id: str,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get a single table"""
        try:
            request = data_service_pb2.GetTableRequest(
                table_id=table_id,
                user_id=user_id or "",
                user_team_ids=user_team_ids or []
            )
            response = await self.stub.GetTable(request)
            return self._table_response_to_dict(response)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return None
            logger.error(f"gRPC error getting table: {e}")
            raise
    
    async def delete_table(
        self, 
        table_id: str,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> bool:
        """Delete a table"""
        try:
            request = data_service_pb2.DeleteTableRequest(
                table_id=table_id,
                user_id=user_id or "",
                user_team_ids=user_team_ids or []
            )
            response = await self.stub.DeleteTable(request)
            return response.success
        except grpc.RpcError as e:
            logger.error(f"gRPC error deleting table: {e}")
            raise
    
    # Table row methods
    async def insert_table_row(
        self,
        table_id: str,
        row_data: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Insert a new row"""
        try:
            request = data_service_pb2.InsertRowRequest(
                table_id=table_id,
                row_data_json=json.dumps(row_data),
                user_id=user_id
            )
            
            response = await self.stub.InsertRow(request)
            return {
                'row_id': response.row_id,
                'row_data': json.loads(response.row_data_json) if response.row_data_json else row_data
            }
            
        except grpc.RpcError as e:
            logger.error(f"gRPC error inserting row: {e}")
            raise
    
    async def update_table_row(
        self,
        table_id: str,
        row_id: str,
        row_data: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Update an existing row"""
        try:
            request = data_service_pb2.UpdateRowRequest(
                table_id=table_id,
                row_id=row_id,
                row_data_json=json.dumps(row_data),
                user_id=user_id
            )
            
            response = await self.stub.UpdateRow(request)
            return {
                'row_id': response.row_id,
                'row_data': json.loads(response.row_data_json) if response.row_data_json else row_data
            }
            
        except grpc.RpcError as e:
            logger.error(f"gRPC error updating row: {e}")
            raise
    
    async def update_table_cell(
        self,
        table_id: str,
        row_id: str,
        column_name: str,
        value: Any,
        user_id: str
    ) -> Dict[str, Any]:
        """Update a single cell"""
        try:
            request = data_service_pb2.UpdateCellRequest(
                table_id=table_id,
                row_id=row_id,
                column_name=column_name,
                value_json=json.dumps(value),
                user_id=user_id
            )
            
            response = await self.stub.UpdateCell(request)
            return {
                'row_id': response.row_id,
                'column_name': column_name,
                'value': value
            }
            
        except grpc.RpcError as e:
            logger.error(f"gRPC error updating cell: {e}")
            raise
    
    async def delete_table_row(self, table_id: str, row_id: str) -> bool:
        """Delete a row"""
        try:
            request = data_service_pb2.DeleteRowRequest(
                table_id=table_id,
                row_id=row_id
            )
            
            response = await self.stub.DeleteRow(request)
            return response.success
            
        except grpc.RpcError as e:
            logger.error(f"gRPC error deleting row: {e}")
            raise
    
    # Helper methods to convert protobuf to dict
    def _workspace_response_to_dict(self, response) -> Dict[str, Any]:
        """Convert workspace protobuf response to dict"""
        return {
            'workspace_id': response.workspace_id,
            'user_id': response.user_id,
            'name': response.name,
            'description': response.description,
            'icon': response.icon,
            'color': response.color,
            'is_pinned': response.is_pinned,
            'metadata_json': response.metadata_json,
            'created_at': response.created_at,
            'updated_at': response.updated_at,
            'updated_by': response.updated_by if hasattr(response, 'updated_by') and response.updated_by else None
        }
    
    def _database_response_to_dict(self, response) -> Dict[str, Any]:
        """Convert database protobuf response to dict"""
        return {
            'database_id': response.database_id,
            'workspace_id': response.workspace_id,
            'name': response.name,
            'description': response.description,
            'source_type': response.source_type,
            'table_count': response.table_count,
            'total_rows': response.total_rows,
            'created_at': response.created_at,
            'metadata_json': response.metadata_json if hasattr(response, 'metadata_json') else '',
            'updated_at': response.updated_at if hasattr(response, 'updated_at') else '',
            'created_by': response.created_by if hasattr(response, 'created_by') and response.created_by else None,
            'updated_by': response.updated_by if hasattr(response, 'updated_by') and response.updated_by else None
        }
    
    def _table_response_to_dict(self, response) -> Dict[str, Any]:
        """Convert table protobuf response to dict"""
        return {
            'table_id': response.table_id,
            'database_id': response.database_id,
            'name': response.name,
            'description': response.description,
            'row_count': response.row_count,
            'table_schema_json': response.schema_json,
            'styling_rules_json': response.styling_rules_json if hasattr(response, 'styling_rules_json') else '{}',
            'metadata_json': response.metadata_json if hasattr(response, 'metadata_json') else '{}',
            'created_at': response.created_at if hasattr(response, 'created_at') else '',
            'updated_at': response.updated_at if hasattr(response, 'updated_at') else '',
            'created_by': response.created_by if hasattr(response, 'created_by') and response.created_by else None,
            'updated_by': response.updated_by if hasattr(response, 'updated_by') and response.updated_by else None
        }
    
    def _import_job_response_to_dict(self, response) -> Dict[str, Any]:
        """Convert import job protobuf response to dict"""
        return {
            'job_id': response.job_id,
            'workspace_id': response.workspace_id,
            'database_id': response.database_id if response.database_id else None,
            'table_id': response.table_id if response.table_id else None,
            'status': response.status,
            'source_file': response.source_file,
            'rows_processed': response.rows_processed,
            'rows_total': response.rows_total,
            'error_log': response.error_log if response.error_log else None,
            'started_at': response.started_at if response.started_at else None,
            'completed_at': response.completed_at if response.completed_at else None,
            'progress_percent': int((response.rows_processed / response.rows_total * 100)) if response.rows_total > 0 else 0
        }
    
    # Query methods
    async def execute_sql_query(
        self,
        workspace_id: str,
        query: str,
        user_id: str,
        limit: int = 1000,
        user_team_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute SQL query against workspace databases"""
        try:
            request = data_service_pb2.SQLQueryRequest(
                workspace_id=workspace_id,
                sql_query=query,
                limit=limit,
                user_id=user_id,
                user_team_ids=user_team_ids or []
            )
            
            response = await self.stub.ExecuteSQLQuery(request)
            
            return {
                'query_id': response.query_id,
                'column_names': list(response.column_names),
                'results_json': response.results_json,
                'result_count': response.result_count,
                'execution_time_ms': response.execution_time_ms,
                'generated_sql': response.generated_sql,
                'error_message': response.error_message if response.error_message else None
            }
        except grpc.RpcError as e:
            logger.error(f"gRPC error executing SQL query: {e}")
            raise
    
    async def execute_nl_query(
        self,
        workspace_id: str,
        natural_query: str,
        user_id: str,
        include_documents: bool = False,
        user_team_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute natural language query"""
        try:
            request = data_service_pb2.NLQueryRequest(
                workspace_id=workspace_id,
                natural_language_query=natural_query,
                include_documents=include_documents,
                user_id=user_id,
                user_team_ids=user_team_ids or []
            )
            
            response = await self.stub.ExecuteNaturalLanguageQuery(request)
            
            return {
                'query_id': response.query_id,
                'column_names': list(response.column_names),
                'results_json': response.results_json,
                'result_count': response.result_count,
                'execution_time_ms': response.execution_time_ms,
                'generated_sql': response.generated_sql,
                'error_message': response.error_message if response.error_message else None
            }
        except grpc.RpcError as e:
            logger.error(f"gRPC error executing NL query: {e}")
            raise


