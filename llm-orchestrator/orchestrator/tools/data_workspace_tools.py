"""
Data Workspace Tools - LangGraph tools for querying Data Workspaces
"""

import logging
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


class DataWorkspaceOutputs(BaseModel):
    """Legacy minimal outputs; prefer specific output models below."""
    formatted: str = Field(description="Human-readable result")
    success: bool = True
    error: Optional[str] = None


class ListDataWorkspacesOutputs(BaseModel):
    """Outputs for list_data_workspaces_tool."""
    workspaces: List[Dict[str, Any]] = Field(default_factory=list, description="List of workspace dicts")
    total_count: int = Field(description="Number of workspaces")
    count: int = Field(description="Number of workspaces (alias for total_count for wiring)")
    success: bool = Field(description="Whether the call succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable result")


class GetWorkspaceSchemaOutputs(BaseModel):
    """Outputs for get_workspace_schema_tool."""
    tables: List[Dict[str, Any]] = Field(default_factory=list, description="List of table dicts with columns")
    total_tables: int = Field(description="Number of tables")
    workspace_id: Optional[str] = Field(default=None, description="Workspace ID that was queried")
    workspace_name: Optional[str] = Field(default=None, description="Workspace name if available")
    success: bool = Field(description="Whether the call succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable result")


class QueryDataWorkspaceOutputs(BaseModel):
    """Outputs for query_data_workspace_tool."""
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Query result rows")
    rows: List[Dict[str, Any]] = Field(default_factory=list, description="Query result rows (alias for results for wiring)")
    column_names: List[str] = Field(default_factory=list, description="Column names")
    columns: List[str] = Field(default_factory=list, description="Column names (alias for column_names for wiring)")
    result_count: int = Field(description="Number of rows returned")
    row_count: int = Field(description="Number of rows returned (alias for result_count for wiring)")
    generated_sql: Optional[str] = Field(default=None, description="Generated SQL if natural language")
    success: bool = Field(description="Whether the query succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable result")
    rows_affected: int = Field(default=0, description="For INSERT/UPDATE/DELETE, number of rows affected")
    returning_rows: Optional[List[Dict[str, Any]]] = Field(default=None, description="For write with RETURNING clause")


async def list_data_workspaces_tool(
    user_id: str = "system"
) -> Dict[str, Any]:
    """List all data workspaces. Returns dict with formatted and workspaces."""
    try:
        logger.info(f"Listing data workspaces for user: {user_id}")
        client = await get_backend_tool_client()
        result = await client.list_data_workspaces(user_id=user_id)
        if "error" in result:
            workspaces = result.get("workspaces", [])
            total_count = result.get("total_count", 0)
            return {
                "workspaces": workspaces,
                "total_count": total_count,
                "count": total_count,
                "success": False,
                "error": result["error"],
                "formatted": f"Error listing workspaces: {result['error']}",
            }
        workspaces = result.get("workspaces", [])
        total_count = result.get("total_count", 0)
        if total_count == 0:
            msg = "No data workspaces found. Create a workspace to start storing data."
            return {"workspaces": [], "total_count": 0, "count": 0, "success": True, "error": None, "formatted": msg}
        response_parts = [f"Available data workspaces ({total_count}):\n"]
        for i, ws in enumerate(workspaces, 1):
            response_parts.append(f"\n{i}. **{ws['name']}** (ID: {ws['workspace_id']})")
            if ws.get("description"):
                response_parts.append(f"   Description: {ws['description']}")
            if ws.get("icon"):
                response_parts.append(f"   Icon: {ws['icon']}")
            if ws.get("is_pinned"):
                response_parts.append("   Pinned: Yes")
        formatted = "\n".join(response_parts)
        return {"workspaces": workspaces, "total_count": total_count, "count": total_count, "success": True, "error": None, "formatted": formatted}
    except Exception as e:
        logger.error("Data workspace list tool error: %s", e)
        err = str(e)
        return {"workspaces": [], "total_count": 0, "count": 0, "success": False, "error": err, "formatted": f"Error listing data workspaces: {err}"}


async def get_workspace_schema_tool(
    workspace_id: str,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Get complete schema for a data workspace (all tables and columns)
    
    Args:
        workspace_id: Workspace ID to get schema for
        user_id: User ID for access control
        
    Returns:
        Formatted string with workspace schema information
    """
    try:
        logger.info(f"Getting workspace schema: workspace={workspace_id}, user={user_id}")
        
        # Get backend client
        client = await get_backend_tool_client()
        
        # Perform schema retrieval via gRPC
        result = await client.get_workspace_schema(
            workspace_id=workspace_id,
            user_id=user_id
        )
        
        if "error" in result:
            return {
                "tables": result.get("tables", []),
                "total_tables": result.get("total_tables", 0),
                "workspace_id": workspace_id,
                "workspace_name": result.get("workspace_name"),
                "success": False,
                "error": result["error"],
                "formatted": f"Error getting workspace schema: {result['error']}",
            }
        tables = result.get("tables", [])
        total_tables = result.get("total_tables", 0)
        if total_tables == 0:
            msg = f"Workspace '{workspace_id}' has no tables. Create tables to start storing data."
            return {"tables": [], "total_tables": 0, "workspace_id": workspace_id, "workspace_name": result.get("workspace_name"), "success": True, "error": None, "formatted": msg}
        response_parts = [f"Workspace Schema (Workspace ID: {workspace_id}):\n"]
        response_parts.append(f"Total tables: {total_tables}\n")
        for table in tables:
            response_parts.append(f"\n**Table: {table['name']}** (ID: {table['table_id']})")
            if table.get("description"):
                response_parts.append(f"  Description: {table['description']}")
            meta = table.get("metadata_json") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta) if meta else {}
                except json.JSONDecodeError:
                    meta = {}
            if meta.get("business_context"):
                response_parts.append(f"  Business context: {meta['business_context']}")
            if meta.get("source"):
                response_parts.append(f"  Source: {meta['source']}")
            if meta.get("update_frequency"):
                response_parts.append(f"  Update frequency: {meta['update_frequency']}")
            response_parts.append(f"  Database: {table.get('database_name', 'Unknown')} (ID: {table.get('database_id', '')})")
            response_parts.append(f"  Row count: {table.get('row_count', 0)}")
            response_parts.append(f"  Columns ({len(table.get('columns', []))}):")
            for col in table.get("columns", []):
                nullable = "nullable" if col.get("is_nullable", True) else "not null"
                desc = (col.get("description") or "").strip()
                rel_hint = ""
                for rel in (meta.get("relationships") or []):
                    if isinstance(rel, dict) and rel.get("column") == col.get("name"):
                        ref_t = rel.get("references_table", "")
                        ref_c = rel.get("references_column", "id")
                        rel_hint = f" -> references {ref_t}.{ref_c}"
                        break
                if desc or rel_hint:
                    response_parts.append(f"    - {col['name']} ({col.get('type', 'text')}, {nullable}){rel_hint} — {desc}" if desc else f"    - {col['name']} ({col.get('type', 'text')}, {nullable}){rel_hint}")
                else:
                    response_parts.append(f"    - {col['name']} ({col.get('type', 'text')}, {nullable})")
            glossary = meta.get("glossary") or {}
            if isinstance(glossary, dict) and glossary:
                response_parts.append("  Glossary (column/term definitions):")
                for term, defn in glossary.items():
                    response_parts.append(f"    - {term}: {defn}")
        logger.info("Retrieved schema for %s tables", total_tables)
        formatted = "\n".join(response_parts)
        return {"tables": tables, "total_tables": total_tables, "workspace_id": workspace_id, "workspace_name": result.get("workspace_name"), "success": True, "error": None, "formatted": formatted}
    except Exception as e:
        logger.error("Get workspace schema tool error: %s", e)
        err = str(e)
        return {"tables": [], "total_tables": 0, "workspace_id": workspace_id, "workspace_name": None, "success": False, "error": err, "formatted": f"Error getting workspace schema: {err}"}


async def query_data_workspace_tool(
    workspace_id: str,
    query: str,
    query_type: str = "natural_language",
    user_id: str = "system",
    limit: int = 100,
    params: Optional[List[Any]] = None,
    read_only: bool = False,
) -> Dict[str, Any]:
    """
    Execute a query against a data workspace (SQL or natural language)
    
    Args:
        workspace_id: Workspace ID to query
        query: SQL query or natural language query
        query_type: "sql" or "natural_language" (default: "natural_language")
        user_id: User ID for access control
        limit: Maximum rows to return (default: 100)
        params: For SQL: optional list of values for $1, $2, ... (parameterized queries)
        
    Returns:
        Formatted string with query results; for writes, rows_affected and returning_rows.
    """
    try:
        logger.info(f"Querying data workspace: workspace={workspace_id}, type={query_type}, query='{query[:100]}'")
        
        # Get backend client
        client = await get_backend_tool_client()
        
        # Perform query via gRPC
        result = await client.query_data_workspace(
            workspace_id=workspace_id,
            query=query,
            query_type=query_type,
            user_id=user_id,
            limit=limit,
            params=params,
            read_only=read_only,
        )
        
        if not result.get("success", False):
            error_msg = result.get("error_message", "Unknown error")
            res_list = result.get("results", [])
            col_names = result.get("column_names", [])
            return {
                "results": res_list,
                "rows": res_list,
                "column_names": col_names,
                "columns": col_names,
                "result_count": 0,
                "row_count": 0,
                "generated_sql": result.get("generated_sql"),
                "success": False,
                "error": error_msg,
                "formatted": f"Query failed: {error_msg}",
                "rows_affected": 0,
                "returning_rows": None,
            }
        result_count = result.get("result_count", 0)
        column_names = result.get("column_names", [])
        results_list = result.get("results", [])
        generated_sql = result.get("generated_sql")
        rows_affected = result.get("rows_affected", 0)
        returning_rows = result.get("returning_rows")
        
        if result_count == 0 and rows_affected == 0 and not returning_rows:
            formatted = f"Query executed successfully but returned no results.\n\nGenerated SQL: {generated_sql or 'N/A'}"
            return {
                "results": [],
                "rows": [],
                "column_names": column_names,
                "columns": column_names,
                "result_count": 0,
                "row_count": 0,
                "generated_sql": generated_sql,
                "success": True,
                "error": None,
                "formatted": formatted,
                "rows_affected": 0,
                "returning_rows": None,
            }
        if rows_affected > 0 or returning_rows:
            response_parts = [f"Write completed. Rows affected: {rows_affected}"]
            if returning_rows:
                response_parts.append(f"Returning rows ({len(returning_rows)}):")
                if column_names and returning_rows:
                    response_parts.append("| " + " | ".join(column_names) + " |")
                    response_parts.append("| " + " | ".join(["---"] * len(column_names)) + " |")
                    for row in returning_rows[:50]:
                        cell_values = [str(row.get(c, "")).replace("|", "\\|")[:50] for c in column_names]
                        response_parts.append("| " + " | ".join(cell_values) + " |")
                    if len(returning_rows) > 50:
                        response_parts.append(f"\n*Showing first 50 of {len(returning_rows)} returned rows*")
                else:
                    response_parts.append(json.dumps(returning_rows[:20], indent=2))
            if generated_sql:
                response_parts.append(f"\nSQL: `{generated_sql}`")
            formatted = "\n".join(response_parts)
            return {
                "results": results_list or returning_rows or [],
                "rows": results_list or returning_rows or [],
                "column_names": column_names,
                "columns": column_names,
                "result_count": result_count or len(returning_rows or []),
                "row_count": result_count or len(returning_rows or []),
                "generated_sql": generated_sql,
                "success": True,
                "error": None,
                "formatted": formatted,
                "rows_affected": rows_affected,
                "returning_rows": returning_rows,
            }
        response_parts = [f"Query Results ({result_count} rows):\n"]
        if generated_sql:
            response_parts.append(f"Generated SQL: `{generated_sql}`\n")
        if column_names and results_list:
            response_parts.append("| " + " | ".join(column_names) + " |")
            response_parts.append("| " + " | ".join(["---"] * len(column_names)) + " |")
            display_limit = min(100, len(results_list))
            for row in results_list[:display_limit]:
                cell_values = []
                for col in column_names:
                    value = row.get(col, "")
                    if value is None:
                        cell_values.append("")
                    else:
                        cell_str = str(value).replace("|", "\\|")
                        if len(cell_str) > 50:
                            cell_str = cell_str[:47] + "..."
                        cell_values.append(cell_str)
                response_parts.append("| " + " | ".join(cell_values) + " |")
            if len(results_list) > display_limit:
                response_parts.append(f"\n*Showing first {display_limit} of {len(results_list)} results*")
        exec_time = result.get("execution_time_ms", 0)
        response_parts.append(f"\nExecution time: {exec_time} ms")
        logger.info("Query completed: %s rows in %sms", result_count, exec_time)
        formatted = "\n".join(response_parts)
        return {
            "results": results_list,
            "rows": results_list,
            "column_names": column_names,
            "columns": column_names,
            "result_count": result_count,
            "row_count": result_count,
            "generated_sql": generated_sql,
            "success": True,
            "error": None,
            "formatted": formatted,
            "rows_affected": 0,
            "returning_rows": None,
        }
    except Exception as e:
        logger.error("Query data workspace tool error: %s", e)
        err = str(e)
        return {
            "results": [],
            "rows": [],
            "column_names": [],
            "columns": [],
            "result_count": 0,
            "row_count": 0,
            "generated_sql": None,
            "success": False,
            "error": err,
            "formatted": f"Error querying data workspace: {err}",
            "rows_affected": 0,
            "returning_rows": None,
        }


class ListDataWorkspacesInputs(BaseModel):
    pass


class GetWorkspaceSchemaInputs(BaseModel):
    workspace_id: str = Field(description="Workspace ID")


class QueryDataWorkspaceInputs(BaseModel):
    workspace_id: str = Field(description="Workspace ID")
    query: str = Field(description="SQL or natural language query")
    query_type: str = Field(default="natural_language", description="sql or natural_language")
    limit: int = Field(default=100, description="Max rows")
    params: Optional[List[Any]] = Field(default=None, description="For SQL: list of values for $1, $2, ...")
    read_only: bool = Field(default=False, description="When true, only SELECT is allowed")


register_action(name="list_data_workspaces", category="data_workspace", description="List data workspaces", inputs_model=ListDataWorkspacesInputs, outputs_model=ListDataWorkspacesOutputs, tool_function=list_data_workspaces_tool)
register_action(name="get_workspace_schema", category="data_workspace", description="Get workspace schema", inputs_model=GetWorkspaceSchemaInputs, outputs_model=GetWorkspaceSchemaOutputs, tool_function=get_workspace_schema_tool)
register_action(name="query_data_workspace", category="data_workspace", description="Query data workspace", inputs_model=QueryDataWorkspaceInputs, outputs_model=QueryDataWorkspaceOutputs, tool_function=query_data_workspace_tool)


# Tool registry for LangGraph
DATA_WORKSPACE_TOOLS = {
    'list_data_workspaces': list_data_workspaces_tool,
    'get_workspace_schema': get_workspace_schema_tool,
    'query_data_workspace': query_data_workspace_tool
}
