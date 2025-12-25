import logging
import json
import uuid
from typing import Dict, Any

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from db.connection_manager import get_db_manager
from services.workspace_service import WorkspaceService
from services.database_service import DatabaseService
from services.table_service import TableService
from services.data_import_service import DataImportService
from services.query_service import QueryService
from config.settings import settings

# Import generated protobuf code
try:
    import data_service_pb2
    import data_service_pb2_grpc
except ImportError:
    logging.error("Protobuf code not generated. Run: python -m grpc_tools.protoc...")
    raise

logger = logging.getLogger(__name__)


class DataServiceImplementation(data_service_pb2_grpc.DataServiceServicer):
    """gRPC service implementation for data workspace operations"""
    
    def __init__(self):
        self.workspace_service = None
        self.database_service = None
        self.table_service = None
        self.import_service = None
        self.query_service = None
    
    async def initialize(self):
        """Initialize services with database connection"""
        db_manager = await get_db_manager()
        self.workspace_service = WorkspaceService(db_manager)
        self.database_service = DatabaseService(db_manager)
        self.table_service = TableService(db_manager)
        self.import_service = DataImportService(db_manager)
        self.query_service = QueryService(db_manager)
        logger.info("Data service initialized successfully")
    
    async def HealthCheck(self, request, context):
        """Health check endpoint"""
        try:
            db_manager = await get_db_manager()
            health = await db_manager.health_check()
            
            return data_service_pb2.HealthCheckResponse(
                status="healthy" if health['healthy'] else "unhealthy",
                service_name=settings.SERVICE_NAME,
                version="1.0.0"
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return data_service_pb2.HealthCheckResponse(
                status="unhealthy",
                service_name=settings.SERVICE_NAME,
                version="1.0.0"
            )
    
    async def CreateWorkspace(self, request, context):
        """Create a new workspace"""
        try:
            workspace = await self.workspace_service.create_workspace(
                user_id=request.user_id,
                name=request.name,
                description=request.description if request.description else None,
                icon=request.icon if request.icon else None,
                color=request.color if request.color else None
            )
            
            return data_service_pb2.WorkspaceResponse(**workspace)
            
        except Exception as e:
            logger.error(f"Failed to create workspace: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.WorkspaceResponse()
    
    async def ListWorkspaces(self, request, context):
        """List workspaces for a user"""
        try:
            workspaces = await self.workspace_service.list_workspaces(request.user_id)
            
            workspace_responses = [
                data_service_pb2.WorkspaceResponse(**ws) for ws in workspaces
            ]
            
            return data_service_pb2.WorkspaceListResponse(
                workspaces=workspace_responses,
                total_count=len(workspaces)
            )
            
        except Exception as e:
            logger.error(f"Failed to list workspaces: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.WorkspaceListResponse()
    
    async def GetWorkspace(self, request, context):
        """Get a single workspace"""
        try:
            user_id = request.user_id if request.user_id else None
            user_team_ids = list(request.user_team_ids) if request.user_team_ids else None
            
            workspace = await self.workspace_service.get_workspace(
                request.workspace_id,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            
            if not workspace:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Workspace not found")
                return data_service_pb2.WorkspaceResponse()
            
            return data_service_pb2.WorkspaceResponse(**workspace)
            
        except Exception as e:
            logger.error(f"Failed to get workspace: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.WorkspaceResponse()
    
    async def UpdateWorkspace(self, request, context):
        """Update a workspace"""
        try:
            workspace = await self.workspace_service.update_workspace(
                workspace_id=request.workspace_id,
                user_id=request.user_id if request.user_id else None,
                name=request.name if request.name else None,
                description=request.description if request.description else None,
                icon=request.icon if request.icon else None,
                color=request.color if request.color else None,
                is_pinned=request.is_pinned if request.HasField('is_pinned') else None
            )
            
            if not workspace:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Workspace not found")
                return data_service_pb2.WorkspaceResponse()
            
            return data_service_pb2.WorkspaceResponse(**workspace)
            
        except Exception as e:
            logger.error(f"Failed to update workspace: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.WorkspaceResponse()
    
    async def DeleteWorkspace(self, request, context):
        """Delete a workspace"""
        try:
            user_id = request.user_id if request.user_id else None
            user_team_ids = list(request.user_team_ids) if request.user_team_ids else None
            
            success = await self.workspace_service.delete_workspace(
                request.workspace_id,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            
            return data_service_pb2.DeleteResponse(
                success=success,
                message="Workspace deleted successfully" if success else "Workspace not found"
            )
            
        except Exception as e:
            logger.error(f"Failed to delete workspace: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.DeleteResponse(success=False, message=str(e))
    
    async def CreateDatabase(self, request, context):
        """Create a new database"""
        try:
            database = await self.database_service.create_database(
                workspace_id=request.workspace_id,
                name=request.name,
                user_id=request.user_id if request.user_id else None,
                description=request.description if request.description else None,
                source_type=request.source_type if request.source_type else "imported"
            )
            
            return data_service_pb2.DatabaseResponse(**database)
            
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.DatabaseResponse()
    
    async def ListDatabases(self, request, context):
        """List databases in a workspace"""
        try:
            user_id = request.user_id if request.user_id else None
            user_team_ids = list(request.user_team_ids) if request.user_team_ids else None
            
            databases = await self.database_service.list_databases(
                request.workspace_id,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            
            database_responses = [
                data_service_pb2.DatabaseResponse(**db) for db in databases
            ]
            
            return data_service_pb2.DatabaseListResponse(
                databases=database_responses,
                total_count=len(databases)
            )
            
        except Exception as e:
            logger.error(f"Failed to list databases: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.DatabaseListResponse()
    
    async def GetDatabase(self, request, context):
        """Get a single database"""
        try:
            user_id = request.user_id if request.user_id else None
            user_team_ids = list(request.user_team_ids) if request.user_team_ids else None
            
            database = await self.database_service.get_database(
                request.database_id,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            
            if not database:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Database not found")
                return data_service_pb2.DatabaseResponse()
            
            return data_service_pb2.DatabaseResponse(**database)
            
        except Exception as e:
            logger.error(f"Failed to get database: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.DatabaseResponse()
    
    async def DeleteDatabase(self, request, context):
        """Delete a database"""
        try:
            user_id = request.user_id if request.user_id else None
            user_team_ids = list(request.user_team_ids) if request.user_team_ids else None
            
            success = await self.database_service.delete_database(
                request.database_id,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            
            return data_service_pb2.DeleteResponse(
                success=success,
                message="Database deleted successfully" if success else "Database not found"
            )
            
        except Exception as e:
            logger.error(f"Failed to delete database: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.DeleteResponse(success=False, message=str(e))
    
    async def PreviewImport(self, request, context):
        """Preview file import"""
        try:
            preview = await self.import_service.preview_import(
                file_path=request.file_path,
                file_type=request.file_type,
                preview_rows=request.preview_rows if request.preview_rows > 0 else 10
            )
            
            return data_service_pb2.PreviewImportResponse(
                column_names=preview['column_names'],
                inferred_types=[json.dumps(t) for t in preview['inferred_types']],
                preview_data_json=json.dumps(preview['preview_data']),
                estimated_rows=preview['estimated_rows']
            )
            
        except Exception as e:
            logger.error(f"Failed to preview import: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.PreviewImportResponse()
    
    async def ExecuteImport(self, request, context):
        """Execute import job"""
        try:
            field_mapping = json.loads(request.field_mapping_json) if request.field_mapping_json else None
            
            # Infer file type from file extension
            from pathlib import Path
            file_extension = Path(request.file_path).suffix.lower().lstrip('.')
            if file_extension == 'jsonl':
                file_type = 'jsonl'
            elif file_extension == 'json':
                file_type = 'json'
            elif file_extension in ['xlsx', 'xls']:
                file_type = 'excel'
            else:
                file_type = 'csv'
            
            job_id = await self.import_service.execute_import(
                workspace_id=request.workspace_id,
                database_id=request.database_id,
                table_name=request.table_name,
                file_path=request.file_path,
                file_type=file_type,
                user_id=request.user_id if request.user_id else None,
                field_mapping=field_mapping
            )
            
            # Get job status
            job = await self.import_service.get_import_status(job_id)
            
            if job:
                # Filter to only fields that exist in ImportJobResponse protobuf
                filtered_job = {
                    'job_id': job.get('job_id', ''),
                    'status': job.get('status', ''),
                    'rows_processed': job.get('rows_processed', 0),
                    'rows_total': job.get('rows_total', 0),
                    'error_log': job.get('error_log', ''),
                    'started_at': job.get('started_at', ''),
                    'completed_at': job.get('completed_at', '')
                }
                return data_service_pb2.ImportJobResponse(**filtered_job)
            else:
                return data_service_pb2.ImportJobResponse()
            
        except Exception as e:
            logger.error(f"Failed to execute import: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.ImportJobResponse()
    
    async def GetImportStatus(self, request, context):
        """Get import job status"""
        try:
            job = await self.import_service.get_import_status(request.job_id)
            
            if not job:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Import job not found")
                return data_service_pb2.ImportJobResponse()
            
            # Filter to only fields that exist in ImportJobResponse protobuf
            filtered_job = {
                'job_id': job.get('job_id', ''),
                'status': job.get('status', ''),
                'rows_processed': job.get('rows_processed', 0),
                'rows_total': job.get('rows_total', 0),
                'error_log': job.get('error_log', ''),
                'started_at': job.get('started_at', ''),
                'completed_at': job.get('completed_at', '')
            }
            return data_service_pb2.ImportJobResponse(**filtered_job)
            
        except Exception as e:
            logger.error(f"Failed to get import status: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.ImportJobResponse()
    
    # Table operations
    async def CreateTable(self, request, context):
        """Create a new table"""
        try:
            schema = json.loads(request.schema_json) if request.schema_json else {}
            
            table = await self.table_service.create_table(
                database_id=request.database_id,
                name=request.name,
                schema=schema,
                user_id=request.user_id if request.user_id else None,
                description=request.description
            )
            
            # Update database stats (table count)
            await self.database_service.update_database_stats(request.database_id, request.user_id if request.user_id else None)
            
            return data_service_pb2.TableResponse(**table)
            
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.TableResponse()
    
    async def ListTables(self, request, context):
        """List all tables in a database"""
        try:
            user_id = request.user_id if request.user_id else None
            user_team_ids = list(request.user_team_ids) if request.user_team_ids else None
            
            tables = await self.table_service.list_tables(
                request.database_id,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            
            table_responses = [
                data_service_pb2.TableResponse(**table)
                for table in tables
            ]
            
            return data_service_pb2.TableListResponse(
                tables=table_responses,
                total_count=len(tables)
            )
            
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.TableListResponse()
    
    async def GetTable(self, request, context):
        """Get a single table"""
        try:
            user_id = request.user_id if request.user_id else None
            user_team_ids = list(request.user_team_ids) if request.user_team_ids else None
            
            table = await self.table_service.get_table(
                request.table_id,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            
            if not table:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Table not found")
                return data_service_pb2.TableResponse()
            
            return data_service_pb2.TableResponse(**table)
            
        except Exception as e:
            logger.error(f"Failed to get table: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.TableResponse()
    
    async def DeleteTable(self, request, context):
        """Delete a table"""
        try:
            user_id = request.user_id if request.user_id else None
            user_team_ids = list(request.user_team_ids) if request.user_team_ids else None
            
            # Get table info before deleting (to get database_id)
            table = await self.table_service.get_table(
                request.table_id,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            database_id = table.get('database_id') if table else None
            
            success = await self.table_service.delete_table(
                request.table_id,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            
            # Update database stats (table count and total rows)
            if success and database_id:
                await self.database_service.update_database_stats(database_id, None)
            
            return data_service_pb2.DeleteResponse(
                success=success,
                message="Table deleted successfully" if success else "Failed to delete table"
            )
            
        except Exception as e:
            logger.error(f"Failed to delete table: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.DeleteResponse(success=False, message=str(e))
    
    async def GetTableData(self, request, context):
        """Get table data with pagination"""
        try:
            user_id = request.user_id if request.user_id else None
            user_team_ids = list(request.user_team_ids) if request.user_team_ids else None
            
            data = await self.table_service.get_table_data(
                table_id=request.table_id,
                offset=request.offset,
                limit=request.limit if request.limit > 0 else 100,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            
            row_responses = [
                data_service_pb2.RowResponse(
                    row_id=row['row_id'],
                    row_data_json=json.dumps(row['row_data']),
                    row_index=row['row_index'],
                    row_color=row.get('row_color', ''),
                    formula_data_json=json.dumps(row.get('formula_data', {}))
                )
                for row in data['rows']
            ]
            
            return data_service_pb2.TableDataResponse(
                table_id=data['table_id'],
                rows=row_responses,
                total_rows=data['total_rows'],
                schema_json=json.dumps(data['schema'])
            )
            
        except Exception as e:
            logger.error(f"Failed to get table data: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.TableDataResponse()
    
    async def InsertRow(self, request, context):
        """Insert a new row"""
        try:
            row_data = json.loads(request.row_data_json)
            
            # Get table to retrieve database_id
            table = await self.table_service.get_table(request.table_id)
            database_id = table.get('database_id') if table else None
            
            row = await self.table_service.insert_row(
                table_id=request.table_id,
                row_data=row_data,
                user_id=request.user_id if request.user_id else None
            )
            
            # Update database stats (total rows)
            if database_id:
                await self.database_service.update_database_stats(database_id, request.user_id if request.user_id else None)
            
            return data_service_pb2.RowResponse(
                row_id=row['row_id'],
                row_data_json=json.dumps(row['row_data']),
                row_index=row.get('row_index', 0),
                row_color=row.get('row_color', ''),
                formula_data_json=json.dumps(row.get('formula_data', {}))
            )
            
        except Exception as e:
            logger.error(f"Failed to insert row: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.RowResponse()
    
    async def UpdateRow(self, request, context):
        """Update an existing row"""
        try:
            row_data = json.loads(request.row_data_json)
            
            row = await self.table_service.update_row(
                table_id=request.table_id,
                row_id=request.row_id,
                row_data=row_data,
                user_id=request.user_id if request.user_id else None
            )
            
            return data_service_pb2.RowResponse(
                row_id=row['row_id'],
                row_data_json=json.dumps(row['row_data']),
                row_index=row.get('row_index', 0),
                row_color=row.get('row_color', ''),
                formula_data_json=json.dumps(row.get('formula_data', {}))
            )
            
        except Exception as e:
            logger.error(f"Failed to update row: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.RowResponse()
    
    async def UpdateCell(self, request, context):
        """Update a single cell with optional formula"""
        try:
            value = json.loads(request.value_json)
            formula = request.formula if request.formula else None
            
            row = await self.table_service.update_cell(
                table_id=request.table_id,
                row_id=request.row_id,
                column_name=request.column_name,
                value=value,
                user_id=request.user_id if request.user_id else None,
                formula=formula
            )
            
            if not row:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Row not found")
                return data_service_pb2.RowResponse()
            
            # Include formula_data in response if available
            response = data_service_pb2.RowResponse(
                row_id=row['row_id'],
                row_data_json=json.dumps(row['row_data']),
                row_index=row.get('row_index', 0),
                row_color=row.get('row_color', ''),
                formula_data_json=json.dumps(row.get('formula_data', {}))
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to update cell: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.RowResponse()
    
    async def DeleteRow(self, request, context):
        """Delete a row"""
        try:
            # Get table to retrieve database_id
            table = await self.table_service.get_table(request.table_id)
            database_id = table.get('database_id') if table else None
            
            success = await self.table_service.delete_row(
                table_id=request.table_id,
                row_id=request.row_id
            )
            
            # Update database stats (total rows)
            if success and database_id:
                await self.database_service.update_database_stats(database_id, None)
            
            return data_service_pb2.DeleteResponse(
                success=success,
                message="Row deleted successfully" if success else "Failed to delete row"
            )
            
        except Exception as e:
            logger.error(f"Failed to delete row: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.DeleteResponse(success=False, message=str(e))
    
    async def RecalculateTable(self, request, context):
        """Recalculate all formulas in a table"""
        try:
            result = await self.table_service.recalculate_table(
                table_id=request.table_id,
                user_id=request.user_id if request.user_id else None
            )
            
            return data_service_pb2.RecalculateTableResponse(
                success=result.get('success', False),
                cells_recalculated=result.get('cells_recalculated', 0),
                error_message=result.get('error_message', '') or ''
            )
            
        except Exception as e:
            logger.error(f"Failed to recalculate table: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.RecalculateTableResponse(
                success=False,
                cells_recalculated=0,
                error_message=str(e)
            )
    
    # Query operations
    async def ExecuteSQLQuery(self, request, context):
        """Execute SQL query against workspace databases"""
        try:
            user_id = request.user_id if request.user_id else None
            user_team_ids = list(request.user_team_ids) if request.user_team_ids else None
            
            result = await self.query_service.execute_sql_query(
                workspace_id=request.workspace_id,
                sql_query=request.sql_query,
                user_id=user_id,
                limit=request.limit if request.limit > 0 else 1000,
                user_team_ids=user_team_ids
            )
            
            return data_service_pb2.QueryResultResponse(**result)
            
        except Exception as e:
            logger.error(f"Failed to execute SQL query: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.QueryResultResponse(
                query_id="",
                column_names=[],
                results_json=json.dumps([]),
                result_count=0,
                execution_time_ms=0,
                generated_sql=request.sql_query if request else "",
                error_message=str(e)
            )
    
    async def ExecuteNaturalLanguageQuery(self, request, context):
        """Execute natural language query (converts to SQL and executes)"""
        try:
            user_id = request.user_id if request.user_id else None
            user_team_ids = list(request.user_team_ids) if request.user_team_ids else None
            
            # For now, basic implementation - just return error
            # TODO: Implement NL to SQL conversion using LLM
            error_message = "Natural language query conversion not yet implemented"
            
            return data_service_pb2.QueryResultResponse(
                query_id=str(uuid.uuid4()),
                column_names=[],
                results_json=json.dumps([]),
                result_count=0,
                execution_time_ms=0,
                generated_sql="",
                error_message=error_message
            )
            
        except Exception as e:
            logger.error(f"Failed to execute NL query: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return data_service_pb2.QueryResultResponse(
                query_id="",
                column_names=[],
                results_json=json.dumps([]),
                result_count=0,
                execution_time_ms=0,
                generated_sql="",
                error_message=str(e)
            )


