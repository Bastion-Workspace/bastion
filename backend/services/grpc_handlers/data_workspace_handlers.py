"""gRPC handlers for Data Workspace operations."""

import logging

import grpc
from protos import tool_service_pb2

logger = logging.getLogger(__name__)


class DataWorkspaceHandlersMixin:
    """Mixin providing Data Workspace gRPC handlers.

    Mixed into ToolServiceImplementation; accesses self._get_search_service(),
    self._get_document_repo(), etc. via standard Python MRO.
    """

    # ===== Data Workspace Operations =====

    async def ListDataWorkspaces(
        self,
        request: tool_service_pb2.ListDataWorkspacesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListDataWorkspacesResponse:
        """List all data workspaces for a user"""
        try:
            logger.info("ListDataWorkspaces: user=%s", request.user_id)

            from tools_service.services.data_workspace_service import (
                get_data_workspace_service,
                list_workspaces_grpc_payload,
            )

            service = await get_data_workspace_service()
            workspaces = await service.list_workspaces(request.user_id)
            payload = list_workspaces_grpc_payload(workspaces)

            return tool_service_pb2.ListDataWorkspacesResponse(
                workspaces=[
                    tool_service_pb2.DataWorkspaceInfo(**w) for w in payload["workspace_infos"]
                ],
                total_count=payload["total_count"],
            )

        except Exception as e:
            logger.error("ListDataWorkspaces failed: %s", e)
            return tool_service_pb2.ListDataWorkspacesResponse(workspaces=[], total_count=0)

    async def GetWorkspaceSchema(
        self,
        request: tool_service_pb2.GetWorkspaceSchemaRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetWorkspaceSchemaResponse:
        """Get complete schema for a workspace (all tables and columns)"""
        try:
            logger.info(
                "GetWorkspaceSchema: workspace=%s, user=%s",
                request.workspace_id,
                request.user_id,
            )

            from tools_service.services.data_workspace_service import (
                get_data_workspace_service,
                workspace_schema_grpc_payload,
            )

            service = await get_data_workspace_service()
            schema_result = await service.get_workspace_schema(
                workspace_id=request.workspace_id,
                user_id=request.user_id,
            )
            sp = workspace_schema_grpc_payload(schema_result)

            table_schemas = []
            for t in sp["tables"]:
                columns = [
                    tool_service_pb2.ColumnInfo(**c) for c in t.get("columns", []) or []
                ]
                table_schemas.append(
                    tool_service_pb2.TableSchema(
                        table_id=t["table_id"],
                        name=t["name"],
                        description=t["description"],
                        database_id=t["database_id"],
                        database_name=t["database_name"],
                        columns=columns,
                        row_count=t["row_count"],
                        metadata_json=t["metadata_json"],
                    )
                )

            return tool_service_pb2.GetWorkspaceSchemaResponse(
                workspace_id=sp["workspace_id"],
                tables=table_schemas,
                total_tables=sp["total_tables"],
            )

        except Exception as e:
            logger.error("GetWorkspaceSchema failed: %s", e)
            return tool_service_pb2.GetWorkspaceSchemaResponse(
                workspace_id=request.workspace_id,
                tables=[],
                total_tables=0,
                error=str(e),
            )

    async def ResolveWorkspaceLink(
        self,
        request: tool_service_pb2.ResolveWorkspaceLinkRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ResolveWorkspaceLinkResponse:
        try:
            from tools_service.services.data_workspace_service import (
                get_data_workspace_service,
                resolve_workspace_link_grpc_payload,
            )

            service = await get_data_workspace_service()
            result = await service.resolve_workspace_link(
                user_id=request.user_id,
                ref_json=request.ref_json or "{}",
            )
            p = resolve_workspace_link_grpc_payload(result)
            return tool_service_pb2.ResolveWorkspaceLinkResponse(**p)
        except Exception as e:
            logger.error("ResolveWorkspaceLink failed: %s", e)
            return tool_service_pb2.ResolveWorkspaceLinkResponse(
                success=False,
                error=str(e),
                row_found=False,
            )

    async def QueryDataWorkspace(
        self,
        request: tool_service_pb2.QueryDataWorkspaceRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.QueryDataWorkspaceResponse:
        """Execute a query against a data workspace (SQL or natural language)"""
        try:
            logger.info(
                "QueryDataWorkspace: workspace=%s, type=%s, user=%s",
                request.workspace_id,
                request.query_type,
                request.user_id,
            )

            from tools_service.services.data_workspace_service import (
                get_data_workspace_service,
                parse_optional_json_list,
                query_workspace_grpc_payload,
            )

            service = await get_data_workspace_service()
            params = parse_optional_json_list(getattr(request, "params_json", None) or "")
            read_only = bool(getattr(request, "read_only", False))
            result = await service.query_workspace(
                workspace_id=request.workspace_id,
                query=request.query,
                query_type=request.query_type,
                user_id=request.user_id,
                limit=request.limit if request.limit > 0 else 100,
                params=params,
                read_only=read_only,
            )
            qp = query_workspace_grpc_payload(result)
            response = tool_service_pb2.QueryDataWorkspaceResponse(
                success=qp["success"],
                column_names=qp["column_names"],
                results_json=qp["results_json"],
                result_count=qp["result_count"],
                execution_time_ms=qp["execution_time_ms"],
                generated_sql=qp["generated_sql"],
                rows_affected=qp["rows_affected"],
                returning_rows_json=qp["returning_rows_json"],
                arrow_results=qp["arrow_results"],
                has_arrow_data=qp["has_arrow_data"],
            )
            if qp.get("error_message"):
                response.error_message = qp["error_message"]
            return response

        except Exception as e:
            logger.error("QueryDataWorkspace failed: %s", e)
            return tool_service_pb2.QueryDataWorkspaceResponse(
                success=False,
                column_names=[],
                results_json="[]",
                result_count=0,
                execution_time_ms=0,
                generated_sql="",
                error_message=str(e),
                rows_affected=0,
                returning_rows_json="[]",
                arrow_results=b"",
                has_arrow_data=False,
            )

    async def CreateDataWorkspaceTable(
        self,
        request: tool_service_pb2.CreateDataWorkspaceTableRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateDataWorkspaceTableResponse:
        """Create a table in a workspace database (structured, no raw SQL required)."""
        try:
            from tools_service.services.data_workspace_service import (
                create_table_grpc_payload,
                get_data_workspace_service,
                parse_columns_json,
                parse_optional_metadata_json,
            )

            service = await get_data_workspace_service()
            columns = parse_columns_json(getattr(request, "columns_json", None))
            metadata = parse_optional_metadata_json(getattr(request, "metadata_json", None))
            result = await service.create_table(
                workspace_id=request.workspace_id,
                database_id=request.database_id,
                table_name=request.table_name,
                user_id=request.user_id,
                description=request.description if request.description else None,
                columns=columns,
                metadata=metadata,
            )
            p = create_table_grpc_payload(result)
            return tool_service_pb2.CreateDataWorkspaceTableResponse(**p)
        except Exception as e:
            logger.error("CreateDataWorkspaceTable failed: %s", e)
            return tool_service_pb2.CreateDataWorkspaceTableResponse(
                success=False,
                table_id="",
                table_json="{}",
                error_message=str(e),
            )

    async def InsertDataWorkspaceRows(
        self,
        request: tool_service_pb2.InsertDataWorkspaceRowsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.InsertDataWorkspaceRowsResponse:
        """Insert rows into a table (structured)."""
        try:
            from tools_service.services.data_workspace_service import (
                get_data_workspace_service,
                insert_rows_grpc_payload,
                parse_rows_json_for_insert,
            )

            service = await get_data_workspace_service()
            rows = parse_rows_json_for_insert(getattr(request, "rows_json", None))
            result = await service.insert_rows(
                workspace_id=request.workspace_id,
                table_id=request.table_id,
                user_id=request.user_id,
                rows=rows,
            )
            p = insert_rows_grpc_payload(result)
            return tool_service_pb2.InsertDataWorkspaceRowsResponse(**p)
        except Exception as e:
            logger.error("InsertDataWorkspaceRows failed: %s", e)
            return tool_service_pb2.InsertDataWorkspaceRowsResponse(
                success=False,
                rows_inserted=0,
                inserted_row_ids_json="[]",
                error_message=str(e),
            )

    async def UpdateDataWorkspaceRows(
        self,
        request: tool_service_pb2.UpdateDataWorkspaceRowsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateDataWorkspaceRowsResponse:
        """Update rows in a table (structured)."""
        try:
            from tools_service.services.data_workspace_service import (
                get_data_workspace_service,
                parse_updates_json,
                update_rows_grpc_payload,
            )

            service = await get_data_workspace_service()
            updates = parse_updates_json(getattr(request, "updates_json", None))
            result = await service.update_rows(
                workspace_id=request.workspace_id,
                table_id=request.table_id,
                user_id=request.user_id,
                updates=updates,
            )
            p = update_rows_grpc_payload(result)
            return tool_service_pb2.UpdateDataWorkspaceRowsResponse(**p)
        except Exception as e:
            logger.error("UpdateDataWorkspaceRows failed: %s", e)
            return tool_service_pb2.UpdateDataWorkspaceRowsResponse(
                success=False,
                rows_updated=0,
                updated_row_ids_json="[]",
                error_message=str(e),
            )

    async def DeleteDataWorkspaceRows(
        self,
        request: tool_service_pb2.DeleteDataWorkspaceRowsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.DeleteDataWorkspaceRowsResponse:
        """Delete rows from a table (structured)."""
        try:
            from tools_service.services.data_workspace_service import (
                delete_rows_grpc_payload,
                get_data_workspace_service,
                parse_row_ids_json,
            )

            service = await get_data_workspace_service()
            row_ids = parse_row_ids_json(getattr(request, "row_ids_json", None))
            result = await service.delete_rows(
                workspace_id=request.workspace_id,
                table_id=request.table_id,
                user_id=request.user_id,
                row_ids=row_ids,
            )
            p = delete_rows_grpc_payload(result)
            return tool_service_pb2.DeleteDataWorkspaceRowsResponse(**p)
        except Exception as e:
            logger.error("DeleteDataWorkspaceRows failed: %s", e)
            return tool_service_pb2.DeleteDataWorkspaceRowsResponse(
                success=False,
                rows_deleted=0,
                deleted_row_ids_json="[]",
                error_message=str(e),
            )
