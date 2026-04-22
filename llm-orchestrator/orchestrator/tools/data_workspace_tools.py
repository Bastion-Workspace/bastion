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

def _looks_like_sql(text: str) -> bool:
    if not text or not text.strip():
        return False
    u = text.strip().upper()
    return any(
        u.startswith(p)
        for p in (
            "SELECT",
            "WITH",
            "INSERT",
            "UPDATE",
            "DELETE",
            "EXPLAIN",
            "SHOW",
            "CREATE",
            "ALTER",
            "DROP",
            "TRUNCATE",
            "MERGE",
            "COPY",
        )
    )


def _strip_markdown_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        # Remove leading ```lang and trailing ```
        parts = s.split("```")
        if len(parts) >= 3:
            return parts[1].split("\n", 1)[1].strip() if "\n" in parts[1] else parts[1].strip()
    return s


def _format_schema_for_nl_to_sql(workspace_id: str, schema_result: Dict[str, Any]) -> str:
    tables = schema_result.get("tables") or []
    lines = [f"Workspace {workspace_id} tables:"]
    for t in tables:
        tname = t.get("name", "")
        lines.append(f"- {tname}")
        for c in t.get("columns", []) or []:
            cname = c.get("name", "")
            ctype = c.get("type", "text")
            desc = (c.get("description") or "").strip()
            ref_extra = ""
            if str(ctype).upper() == "REFERENCE":
                cref = c.get("column_ref_json") or ""
                if cref:
                    try:
                        rj = json.loads(cref)
                        tid = rj.get("target_table_id", "")
                        ref_extra = (
                            f"; REFERENCE -> target_table_id={tid}; "
                            f"cell stores JSON with _bastion_ref: "
                            f"v, table_id, row_id, label, optional preview"
                        )
                    except (json.JSONDecodeError, TypeError):
                        ref_extra = "; REFERENCE column (_bastion_ref JSON)"
                else:
                    ref_extra = "; REFERENCE column (_bastion_ref JSON)"
            if desc:
                lines.append(f"  - {cname} ({ctype}){ref_extra}: {desc}")
            else:
                lines.append(f"  - {cname} ({ctype}){ref_extra}")
    return "\n".join(lines)


def _get_llm_for_nl_to_sql(pipeline_metadata: Optional[Dict[str, Any]] = None):
    try:
        from langchain_openai import ChatOpenAI
        from config.settings import settings
    except Exception:
        return None
    from orchestrator.utils.llm_credentials_from_metadata import get_openrouter_credentials

    meta = pipeline_metadata or {}
    model = meta.get("user_chat_model") or settings.DEFAULT_MODEL
    api_key, base_url = get_openrouter_credentials(meta)
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0.0,
    )


async def _nl_to_sql(
    *,
    workspace_id: str,
    natural_language: str,
    schema_result: Dict[str, Any],
    read_only: bool,
    pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    llm = _get_llm_for_nl_to_sql(pipeline_metadata)
    if not llm:
        raise RuntimeError("LLM not configured for NL-to-SQL")
    from langchain_core.messages import HumanMessage, SystemMessage

    schema_txt = _format_schema_for_nl_to_sql(workspace_id, schema_result)
    allowed = "SELECT/WITH (read-only)" if read_only else "SELECT/WITH or INSERT/UPDATE/DELETE/DDL (read-write)"
    system = (
        "You translate user questions into a single PostgreSQL SQL statement for a specific workspace schema.\n"
        f"Rules:\n- Output SQL only (no prose, no markdown).\n- Use only the tables/columns provided.\n"
        f"- Allowed: {allowed}.\n"
        + ("- MUST be a SELECT or WITH query (no writes).\n" if read_only else "")
        + "- Add a LIMIT when returning rows unless the query is an aggregate.\n"
        "- Prefer ordering by a date/timestamp column when asking for 'last' or 'most recent'.\n\n"
        + schema_txt
    )
    resp = await llm.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=natural_language)]
    )
    sql = _strip_markdown_fences(getattr(resp, "content", "") or "")
    sql = sql.strip().rstrip(";").strip()
    return sql


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


class CreateWorkspaceTableOutputs(BaseModel):
    table_id: str = Field(description="Created table ID")
    table: Dict[str, Any] = Field(default_factory=dict, description="Created table dict")
    success: bool = Field(description="Whether the call succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable result")


class InsertWorkspaceRowsOutputs(BaseModel):
    rows_inserted: int = Field(description="Number of rows inserted")
    inserted_row_ids: List[str] = Field(default_factory=list, description="Inserted row IDs")
    success: bool = Field(description="Whether the call succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable result")


class UpdateWorkspaceRowsOutputs(BaseModel):
    rows_updated: int = Field(description="Number of rows updated")
    updated_row_ids: List[str] = Field(default_factory=list, description="Updated row IDs")
    success: bool = Field(description="Whether the call succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable result")


class DeleteWorkspaceRowsOutputs(BaseModel):
    rows_deleted: int = Field(description="Number of rows deleted")
    deleted_row_ids: List[str] = Field(default_factory=list, description="Deleted row IDs")
    success: bool = Field(description="Whether the call succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable result")


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
                ref_cell = ""
                if str(col.get("type", "")).upper() == "REFERENCE":
                    cref = col.get("column_ref_json") or ""
                    if cref:
                        try:
                            rj = json.loads(cref)
                            ref_cell = (
                                f"; stores _bastion_ref to table_id={rj.get('target_table_id', '')} "
                                f"(join on row_id; label_field={rj.get('label_field', 'name')})"
                            )
                        except (json.JSONDecodeError, TypeError):
                            ref_cell = "; stores _bastion_ref JSON"
                    else:
                        ref_cell = "; stores _bastion_ref JSON"
                line_base = f"    - {col['name']} ({col.get('type', 'text')}, {nullable}){rel_hint}{ref_cell}"
                if desc:
                    response_parts.append(f"{line_base} — {desc}")
                else:
                    response_parts.append(line_base)
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


class ResolveWorkspaceLinkOutputs(BaseModel):
    success: bool = Field(description="Whether resolution succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    label: str = Field(default="", description="Current display label for the linked row")
    preview: Dict[str, Any] = Field(default_factory=dict, description="Small key/value preview from target row")
    row_found: bool = Field(default=False, description="Whether the target row still exists")
    table_id: str = Field(default="", description="Target table id")
    row_id: str = Field(default="", description="Target row id")
    formatted: str = Field(description="Human-readable summary")


async def resolve_workspace_link_tool(
    ref_json: str,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Resolve a cross-table reference (_bastion_ref JSON) to the current label and preview.
    ref_json: JSON string of the cell value { \"_bastion_ref\": { \"v\", \"table_id\", \"row_id\", ... } } or the inner object.
    """
    try:
        client = await get_backend_tool_client()
        payload = json.loads(ref_json) if ref_json and ref_json.strip() else {}
        result = await client.resolve_workspace_link(ref_payload=payload, user_id=user_id)
        label = result.get("label") or ""
        found = bool(result.get("row_found"))
        formatted = (
            f"Resolved link: label={label!r}, row_found={found}, "
            f"table_id={result.get('table_id', '')}, row_id={result.get('row_id', '')}"
        )
        if result.get("preview"):
            formatted += f", preview={json.dumps(result.get('preview'))}"
        if not result.get("success") and result.get("error"):
            formatted = f"Resolve failed: {result.get('error')}"
        return {
            "success": bool(result.get("success")),
            "error": result.get("error"),
            "label": label,
            "preview": result.get("preview") or {},
            "row_found": found,
            "table_id": result.get("table_id") or "",
            "row_id": result.get("row_id") or "",
            "formatted": formatted,
        }
    except Exception as e:
        err = str(e)
        return {
            "success": False,
            "error": err,
            "label": "",
            "preview": {},
            "row_found": False,
            "table_id": "",
            "row_id": "",
            "formatted": f"Resolve workspace link failed: {err}",
        }


async def query_data_workspace_tool(
    workspace_id: str,
    query: str,
    query_type: str = "natural_language",
    user_id: str = "system",
    limit: int = 100,
    params: Optional[List[Any]] = None,
    read_only: bool = False,
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
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

        effective_query_type = (query_type or "natural_language").lower().strip()
        effective_query = query

        # Implement real NL-to-SQL here so agents can ask questions like:
        # \"When did I last change the oil on the Honda?\"
        generated_by_llm = None
        if effective_query_type == "natural_language" and not _looks_like_sql(effective_query):
            schema_list = (_pipeline_metadata or {}).get("workspace_schemas")
            schema_result = None
            if isinstance(schema_list, list):
                for item in schema_list:
                    if isinstance(item, dict) and item.get("workspace_id") == workspace_id:
                        schema_result = item
                        break
            if not schema_result:
                schema_result = await client.get_workspace_schema(
                    workspace_id=workspace_id, user_id=user_id
                )
            generated_by_llm = await _nl_to_sql(
                workspace_id=workspace_id,
                natural_language=effective_query,
                schema_result=schema_result or {"tables": []},
                read_only=read_only,
                pipeline_metadata=_pipeline_metadata,
            )

            u = (generated_by_llm or "").lstrip().upper()
            if read_only and not (u.startswith("SELECT") or u.startswith("WITH")):
                raise ValueError("NL-to-SQL produced a non-SELECT statement for a read-only workspace")
            if not read_only and not _looks_like_sql(generated_by_llm):
                raise ValueError("NL-to-SQL did not produce valid SQL")

            effective_query_type = "sql"
            effective_query = generated_by_llm
        
        # Perform query via gRPC
        result = await client.query_data_workspace(
            workspace_id=workspace_id,
            query=effective_query,
            query_type=effective_query_type,
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
        if generated_by_llm:
            generated_sql = generated_by_llm
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


async def create_workspace_table_tool(
    workspace_id: str,
    database_id: str,
    table_name: str,
    columns: List[Dict[str, Any]],
    description: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Create a native table (structured, no raw SQL required)."""
    try:
        client = await get_backend_tool_client()
        res = await client.create_data_workspace_table(
            workspace_id=workspace_id,
            database_id=database_id,
            table_name=table_name,
            user_id=user_id,
            description=description or "",
            columns=columns or [],
            metadata=metadata or {},
        )
        if not res.get("success"):
            err = res.get("error_message") or "Create table failed"
            return {"table_id": "", "table": {}, "success": False, "error": err, "formatted": f"Create table failed: {err}"}
        table_id = res.get("table_id") or (res.get("table") or {}).get("table_id", "")
        formatted = f"Created table **{table_name}** (ID: {table_id})."
        return {"table_id": table_id, "table": res.get("table") or {}, "success": True, "error": None, "formatted": formatted}
    except Exception as e:
        err = str(e)
        return {"table_id": "", "table": {}, "success": False, "error": err, "formatted": f"Create table failed: {err}"}


async def insert_workspace_rows_tool(
    workspace_id: str,
    table_id: str,
    rows: List[Dict[str, Any]],
    user_id: str = "system",
) -> Dict[str, Any]:
    """Insert one or more rows (structured)."""
    try:
        client = await get_backend_tool_client()
        res = await client.insert_data_workspace_rows(
            workspace_id=workspace_id, table_id=table_id, rows=rows or [], user_id=user_id
        )
        if not res.get("success"):
            err = res.get("error_message") or "Insert rows failed"
            return {"rows_inserted": 0, "inserted_row_ids": [], "success": False, "error": err, "formatted": f"Insert rows failed: {err}"}
        n = int(res.get("rows_inserted") or 0)
        ids = res.get("inserted_row_ids") or []
        return {"rows_inserted": n, "inserted_row_ids": ids, "success": True, "error": None, "formatted": f"Inserted {n} row(s)."}
    except Exception as e:
        err = str(e)
        return {"rows_inserted": 0, "inserted_row_ids": [], "success": False, "error": err, "formatted": f"Insert rows failed: {err}"}


async def update_workspace_rows_tool(
    workspace_id: str,
    table_id: str,
    updates: List[Dict[str, Any]],
    user_id: str = "system",
) -> Dict[str, Any]:
    """Update one or more rows by row_id (structured)."""
    try:
        client = await get_backend_tool_client()
        res = await client.update_data_workspace_rows(
            workspace_id=workspace_id, table_id=table_id, updates=updates or [], user_id=user_id
        )
        if not res.get("success"):
            err = res.get("error_message") or "Update rows failed"
            return {"rows_updated": 0, "updated_row_ids": [], "success": False, "error": err, "formatted": f"Update rows failed: {err}"}
        n = int(res.get("rows_updated") or 0)
        ids = res.get("updated_row_ids") or []
        return {"rows_updated": n, "updated_row_ids": ids, "success": True, "error": None, "formatted": f"Updated {n} row(s)."}
    except Exception as e:
        err = str(e)
        return {"rows_updated": 0, "updated_row_ids": [], "success": False, "error": err, "formatted": f"Update rows failed: {err}"}


async def delete_workspace_rows_tool(
    workspace_id: str,
    table_id: str,
    row_ids: List[str],
    user_id: str = "system",
) -> Dict[str, Any]:
    """Delete one or more rows by row_id (structured)."""
    try:
        client = await get_backend_tool_client()
        res = await client.delete_data_workspace_rows(
            workspace_id=workspace_id, table_id=table_id, row_ids=row_ids or [], user_id=user_id
        )
        if not res.get("success"):
            err = res.get("error_message") or "Delete rows failed"
            return {"rows_deleted": 0, "deleted_row_ids": [], "success": False, "error": err, "formatted": f"Delete rows failed: {err}"}
        n = int(res.get("rows_deleted") or 0)
        ids = res.get("deleted_row_ids") or []
        return {"rows_deleted": n, "deleted_row_ids": ids, "success": True, "error": None, "formatted": f"Deleted {n} row(s)."}
    except Exception as e:
        err = str(e)
        return {"rows_deleted": 0, "deleted_row_ids": [], "success": False, "error": err, "formatted": f"Delete rows failed: {err}"}


class ListDataWorkspacesInputs(BaseModel):
    pass


class GetWorkspaceSchemaInputs(BaseModel):
    workspace_id: str = Field(description="Workspace ID")


class ResolveWorkspaceLinkInputs(BaseModel):
    ref_json: str = Field(
        description='JSON for _bastion_ref cell or inner object, e.g. {"_bastion_ref":{"v":1,"table_id":"...","row_id":"..."}}'
    )


class QueryDataWorkspaceInputs(BaseModel):
    workspace_id: str = Field(description="Workspace ID")
    query: str = Field(description="SQL or natural language query")
    query_type: str = Field(default="natural_language", description="sql or natural_language")
    limit: int = Field(default=100, description="Max rows")
    params: Optional[List[Any]] = Field(default=None, description="For SQL: list of values for $1, $2, ...")
    read_only: bool = Field(default=False, description="When true, only SELECT is allowed")

class CreateWorkspaceTableInputs(BaseModel):
    workspace_id: str = Field(description="Workspace ID")
    database_id: str = Field(description="Database ID")
    table_name: str = Field(description="Table name")
    columns: List[Dict[str, Any]] = Field(description="List of column dicts (name/type/nullable/description)")
    description: str = Field(default="", description="Optional table description")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional table metadata")


class InsertWorkspaceRowsInputs(BaseModel):
    workspace_id: str = Field(description="Workspace ID")
    table_id: str = Field(description="Table ID")
    rows: List[Dict[str, Any]] = Field(description="Rows to insert")


class UpdateWorkspaceRowsInputs(BaseModel):
    workspace_id: str = Field(description="Workspace ID")
    table_id: str = Field(description="Table ID")
    updates: List[Dict[str, Any]] = Field(description="List of {row_id, row_data} updates")


class DeleteWorkspaceRowsInputs(BaseModel):
    workspace_id: str = Field(description="Workspace ID")
    table_id: str = Field(description="Table ID")
    row_ids: List[str] = Field(description="Row IDs to delete")


register_action(name="list_data_workspaces", category="data_workspace", description="List data workspaces", inputs_model=ListDataWorkspacesInputs, outputs_model=ListDataWorkspacesOutputs, tool_function=list_data_workspaces_tool)
register_action(name="get_workspace_schema", category="data_workspace", description="Get workspace schema", inputs_model=GetWorkspaceSchemaInputs, outputs_model=GetWorkspaceSchemaOutputs, tool_function=get_workspace_schema_tool)
register_action(
    name="resolve_workspace_link",
    category="data_workspace",
    description="Resolve a _bastion_ref cell to current label and preview from the target row",
    inputs_model=ResolveWorkspaceLinkInputs,
    outputs_model=ResolveWorkspaceLinkOutputs,
    tool_function=resolve_workspace_link_tool,
)
register_action(name="query_data_workspace", category="data_workspace", description="Query data workspace", inputs_model=QueryDataWorkspaceInputs, outputs_model=QueryDataWorkspaceOutputs, tool_function=query_data_workspace_tool)
register_action(name="create_workspace_table", category="data_workspace", description="Create a data workspace table", inputs_model=CreateWorkspaceTableInputs, outputs_model=CreateWorkspaceTableOutputs, tool_function=create_workspace_table_tool)
register_action(name="insert_workspace_rows", category="data_workspace", description="Insert rows into a table", inputs_model=InsertWorkspaceRowsInputs, outputs_model=InsertWorkspaceRowsOutputs, tool_function=insert_workspace_rows_tool)
register_action(name="update_workspace_rows", category="data_workspace", description="Update rows in a table", inputs_model=UpdateWorkspaceRowsInputs, outputs_model=UpdateWorkspaceRowsOutputs, tool_function=update_workspace_rows_tool)
register_action(name="delete_workspace_rows", category="data_workspace", description="Delete rows from a table", inputs_model=DeleteWorkspaceRowsInputs, outputs_model=DeleteWorkspaceRowsOutputs, tool_function=delete_workspace_rows_tool)


# Tool registry for LangGraph
DATA_WORKSPACE_TOOLS = {
    'list_data_workspaces': list_data_workspaces_tool,
    'get_workspace_schema': get_workspace_schema_tool,
    'resolve_workspace_link': resolve_workspace_link_tool,
    'query_data_workspace': query_data_workspace_tool,
    'create_workspace_table': create_workspace_table_tool,
    'insert_workspace_rows': insert_workspace_rows_tool,
    'update_workspace_rows': update_workspace_rows_tool,
    'delete_workspace_rows': delete_workspace_rows_tool,
}
