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
                            ctype = col.get('type', 'text')
                            ref = col.get('ref') if isinstance(col.get('ref'), dict) else {}
                            cref = ''
                            if str(ctype).upper() == 'REFERENCE' and ref.get('target_table_id'):
                                cref = json.dumps({
                                    'target_table_id': ref.get('target_table_id'),
                                    'target_key': ref.get('target_key') or 'row_id',
                                    'label_field': ref.get('label_field') or 'name',
                                })
                            columns.append({
                                'name': col.get('name', ''),
                                'type': ctype,
                                'is_nullable': col.get('nullable', True),
                                'description': col.get('description', '') or '',
                                'column_ref_json': cref,
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

    async def resolve_workspace_link(self, user_id: str, ref_json: str) -> Dict[str, Any]:
        try:
            await self._ensure_initialized()
            payload = json.loads(ref_json) if ref_json and ref_json.strip() else {}
            result = await self._data_client.resolve_workspace_link(
                user_id=user_id,
                ref_payload=payload,
            )
            return result
        except Exception as e:
            logger.error(f"Failed to resolve workspace link: {e}")
            return {
                'success': False,
                'error': str(e),
                'label': '',
                'preview': {},
                'row_found': False,
                'table_id': '',
                'row_id': '',
            }
    
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

    async def create_table(
        self,
        *,
        workspace_id: str,
        database_id: str,
        table_name: str,
        user_id: str,
        description: Optional[str] = None,
        columns: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a native table in a workspace database."""
        await self._ensure_initialized()
        schema = {"columns": columns or []}
        table = await self._data_client.create_table(
            database_id=database_id,
            name=table_name,
            user_id=user_id,
            description=description,
            table_schema=schema,
            metadata=metadata,
        )
        return {"success": True, "table": table}

    async def insert_rows(
        self,
        *,
        workspace_id: str,
        table_id: str,
        user_id: str,
        rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Insert one or more rows into a table."""
        await self._ensure_initialized()
        inserted_ids: List[str] = []
        for row in rows or []:
            res = await self._data_client.insert_table_row(
                table_id=table_id, row_data=row or {}, user_id=user_id
            )
            if res and res.get("row_id"):
                inserted_ids.append(res["row_id"])
        return {"success": True, "inserted_row_ids": inserted_ids, "rows_inserted": len(inserted_ids)}

    async def update_rows(
        self,
        *,
        workspace_id: str,
        table_id: str,
        user_id: str,
        updates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Update one or more rows by row_id."""
        await self._ensure_initialized()
        updated_ids: List[str] = []
        for upd in updates or []:
            if not isinstance(upd, dict):
                continue
            row_id = upd.get("row_id")
            row_data = upd.get("row_data") or {}
            if not row_id:
                continue
            res = await self._data_client.update_table_row(
                table_id=table_id, row_id=row_id, row_data=row_data, user_id=user_id
            )
            if res and res.get("row_id"):
                updated_ids.append(res["row_id"])
        return {"success": True, "updated_row_ids": updated_ids, "rows_updated": len(updated_ids)}

    async def delete_rows(
        self,
        *,
        workspace_id: str,
        table_id: str,
        user_id: str,
        row_ids: List[str],
    ) -> Dict[str, Any]:
        """Delete one or more rows by row_id."""
        await self._ensure_initialized()
        deleted_ids: List[str] = []
        for rid in row_ids or []:
            if not rid:
                continue
            ok = await self._data_client.delete_table_row(table_id=table_id, row_id=rid)
            if ok:
                deleted_ids.append(rid)
        return {"success": True, "deleted_row_ids": deleted_ids, "rows_deleted": len(deleted_ids)}


# --- gRPC mixin helpers (dict in/out; no protobuf) ---


def parse_optional_json_list(raw: Optional[str]) -> Optional[List[Any]]:
    """Parse request JSON string into a list for SQL params; None if empty/invalid."""
    if not raw or not str(raw).strip():
        return None
    try:
        params = json.loads(raw)
        if not isinstance(params, list):
            return [params]
        return params
    except json.JSONDecodeError:
        return None


def parse_columns_json(raw: Optional[str]) -> List[Dict[str, Any]]:
    if not raw or not str(raw).strip():
        return []
    try:
        columns = json.loads(raw)
        return columns if isinstance(columns, list) else []
    except json.JSONDecodeError:
        return []


def parse_optional_metadata_json(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw or not str(raw).strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def parse_rows_json_for_insert(raw: Optional[str]) -> List[Dict[str, Any]]:
    if not raw or not str(raw).strip():
        return []
    try:
        rows = json.loads(raw)
        if isinstance(rows, dict):
            return [rows]
        return rows if isinstance(rows, list) else []
    except json.JSONDecodeError:
        return []


def parse_updates_json(raw: Optional[str]) -> List[Dict[str, Any]]:
    if not raw or not str(raw).strip():
        return []
    try:
        updates = json.loads(raw)
        return updates if isinstance(updates, list) else []
    except json.JSONDecodeError:
        return []


def parse_row_ids_json(raw: Optional[str]) -> List[str]:
    if not raw or not str(raw).strip():
        return []
    try:
        row_ids = json.loads(raw)
        return row_ids if isinstance(row_ids, list) else []
    except json.JSONDecodeError:
        return []


def list_workspaces_grpc_payload(workspaces: List[Dict[str, Any]]) -> Dict[str, Any]:
    infos: List[Dict[str, Any]] = []
    for ws in workspaces or []:
        infos.append(
            {
                "workspace_id": ws.get("workspace_id", ""),
                "name": ws.get("name", ""),
                "description": ws.get("description", ""),
                "icon": ws.get("icon", ""),
                "color": ws.get("color", ""),
                "is_pinned": ws.get("is_pinned", False),
            }
        )
    return {"workspace_infos": infos, "total_count": len(infos)}


def workspace_schema_grpc_payload(schema_result: Dict[str, Any]) -> Dict[str, Any]:
    """Build table/column dicts for GetWorkspaceSchemaResponse mapping."""
    tables_out: List[Dict[str, Any]] = []
    for table in schema_result.get("tables", []) or []:
        columns: List[Dict[str, Any]] = []
        for col in table.get("columns", []) or []:
            columns.append(
                {
                    "name": col.get("name", ""),
                    "type": col.get("type", "text"),
                    "is_nullable": col.get("is_nullable", True),
                    "description": col.get("description", "") or "",
                    "column_ref_json": col.get("column_ref_json", "") or "",
                }
            )
        meta = table.get("metadata_json")
        metadata_json_str = json.dumps(meta) if isinstance(meta, dict) and meta else ""
        tables_out.append(
            {
                "table_id": table.get("table_id", ""),
                "name": table.get("name", ""),
                "description": table.get("description", ""),
                "database_id": table.get("database_id", ""),
                "database_name": table.get("database_name", ""),
                "columns": columns,
                "row_count": table.get("row_count", 0),
                "metadata_json": metadata_json_str or "",
            }
        )
    return {
        "workspace_id": schema_result.get("workspace_id", ""),
        "tables": tables_out,
        "total_tables": len(tables_out),
    }


def resolve_workspace_link_grpc_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    preview = result.get("preview") or {}
    return {
        "success": bool(result.get("success")),
        "error": result.get("error") or "",
        "label": result.get("label") or "",
        "preview_json": json.dumps(preview),
        "row_found": bool(result.get("row_found")),
        "table_id": result.get("table_id") or "",
        "row_id": result.get("row_id") or "",
    }


def query_workspace_grpc_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize query_workspace result for gRPC response fields."""
    arrow_bytes = result.get("arrow_results") or b""
    has_arrow = bool(result.get("has_arrow_data")) and bool(arrow_bytes)

    if has_arrow:
        results_json_str = "[]"
    else:
        results_json = result.get("results", [])
        if isinstance(results_json, str):
            results_json_str = results_json
        else:
            results_json_str = json.dumps(results_json)

    returning = result.get("returning_rows") or result.get("returning_rows_json")
    if isinstance(returning, str) and returning:
        try:
            returning = json.loads(returning)
        except json.JSONDecodeError:
            returning = []
    elif not isinstance(returning, list):
        returning = []

    out: Dict[str, Any] = {
        "success": result.get("success", False),
        "column_names": result.get("column_names", []),
        "results_json": results_json_str,
        "result_count": result.get("result_count", 0),
        "execution_time_ms": result.get("execution_time_ms", 0),
        "generated_sql": result.get("generated_sql", ""),
        "rows_affected": result.get("rows_affected", 0),
        "returning_rows_json": json.dumps(returning),
        "arrow_results": arrow_bytes,
        "has_arrow_data": has_arrow,
        "error_message": result.get("error_message"),
    }
    return out


def create_table_grpc_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    table = result.get("table") or {}
    return {
        "success": True,
        "table_id": table.get("table_id", ""),
        "table_json": json.dumps(table) if table else "{}",
        "error_message": "",
    }


def insert_rows_grpc_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    ids = result.get("inserted_row_ids") or []
    return {
        "success": True,
        "rows_inserted": int(result.get("rows_inserted", len(ids)) or 0),
        "inserted_row_ids_json": json.dumps(ids),
        "error_message": "",
    }


def update_rows_grpc_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    ids = result.get("updated_row_ids") or []
    return {
        "success": True,
        "rows_updated": int(result.get("rows_updated", len(ids)) or 0),
        "updated_row_ids_json": json.dumps(ids),
        "error_message": "",
    }


def delete_rows_grpc_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    ids = result.get("deleted_row_ids") or []
    return {
        "success": True,
        "rows_deleted": int(result.get("rows_deleted", len(ids)) or 0),
        "deleted_row_ids_json": json.dumps(ids),
        "error_message": "",
    }


async def get_data_workspace_service() -> DataWorkspaceService:
    """Get global Data Workspace service instance"""
    global _data_workspace_service_instance
    if _data_workspace_service_instance is None:
        _data_workspace_service_instance = DataWorkspaceService()
        await _data_workspace_service_instance.initialize()
    return _data_workspace_service_instance
