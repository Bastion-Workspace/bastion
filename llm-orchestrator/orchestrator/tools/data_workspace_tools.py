"""
Data Workspace Tools - LangGraph tools for querying Data Workspaces
"""

import logging
import json
from typing import Optional, Dict, Any

from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)


async def list_data_workspaces_tool(
    user_id: str = "system"
) -> str:
    """
    List all data workspaces available to the user
    
    Args:
        user_id: User ID for access control
        
    Returns:
        Formatted string listing available workspaces
    """
    try:
        logger.info(f"Listing data workspaces for user: {user_id}")
        
        # Get backend client
        client = await get_backend_tool_client()
        
        # Perform list via gRPC
        result = await client.list_data_workspaces(user_id=user_id)
        
        # Format results
        if 'error' in result:
            return f"Error listing workspaces: {result['error']}"
        
        if result['total_count'] == 0:
            return "No data workspaces found. Create a workspace to start storing data."
        
        # Build formatted response
        response_parts = [f"Available data workspaces ({result['total_count']}):\n"]
        
        for i, ws in enumerate(result['workspaces'], 1):
            response_parts.append(f"\n{i}. **{ws['name']}** (ID: {ws['workspace_id']})")
            if ws.get('description'):
                response_parts.append(f"   Description: {ws['description']}")
            if ws.get('icon'):
                response_parts.append(f"   Icon: {ws['icon']}")
            if ws.get('is_pinned'):
                response_parts.append(f"   Pinned: Yes")
        
        logger.info(f"Listed {result['total_count']} workspaces")
        return '\n'.join(response_parts)
        
    except Exception as e:
        logger.error(f"Data workspace list tool error: {e}")
        return f"Error listing data workspaces: {str(e)}"


async def get_workspace_schema_tool(
    workspace_id: str,
    user_id: str = "system"
) -> str:
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
        
        # Format results
        if 'error' in result:
            return f"Error getting workspace schema: {result['error']}"
        
        if result['total_tables'] == 0:
            return f"Workspace '{workspace_id}' has no tables. Create tables to start storing data."
        
        # Build formatted response
        response_parts = [f"Workspace Schema (Workspace ID: {workspace_id}):\n"]
        response_parts.append(f"Total tables: {result['total_tables']}\n")
        
        for table in result['tables']:
            response_parts.append(f"\n**Table: {table['name']}** (ID: {table['table_id']})")
            if table.get('description'):
                response_parts.append(f"  Description: {table['description']}")
            response_parts.append(f"  Database: {table.get('database_name', 'Unknown')} (ID: {table.get('database_id', '')})")
            response_parts.append(f"  Row count: {table.get('row_count', 0)}")
            response_parts.append(f"  Columns ({len(table.get('columns', []))}):")
            
            for col in table.get('columns', []):
                nullable = "nullable" if col.get('is_nullable', True) else "not null"
                response_parts.append(f"    - {col['name']} ({col.get('type', 'text')}, {nullable})")
        
        logger.info(f"Retrieved schema for {result['total_tables']} tables")
        return '\n'.join(response_parts)
        
    except Exception as e:
        logger.error(f"Get workspace schema tool error: {e}")
        return f"Error getting workspace schema: {str(e)}"


async def query_data_workspace_tool(
    workspace_id: str,
    query: str,
    query_type: str = "natural_language",
    user_id: str = "system",
    limit: int = 100
) -> str:
    """
    Execute a query against a data workspace (SQL or natural language)
    
    Args:
        workspace_id: Workspace ID to query
        query: SQL query or natural language query
        query_type: "sql" or "natural_language" (default: "natural_language")
        user_id: User ID for access control
        limit: Maximum rows to return (default: 100)
        
    Returns:
        Formatted string with query results as markdown table
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
            limit=limit
        )
        
        # Format results
        if not result.get('success', False):
            error_msg = result.get('error_message', 'Unknown error')
            return f"Query failed: {error_msg}"
        
        if result['result_count'] == 0:
            return f"Query executed successfully but returned no results.\n\nGenerated SQL: {result.get('generated_sql', 'N/A')}"
        
        # Build formatted response with markdown table
        response_parts = [f"Query Results ({result['result_count']} rows):\n"]
        
        # Add generated SQL for transparency
        if result.get('generated_sql'):
            response_parts.append(f"Generated SQL: `{result['generated_sql']}`\n")
        
        # Build markdown table
        column_names = result.get('column_names', [])
        results = result.get('results', [])
        
        if column_names and results:
            # Header row
            response_parts.append("| " + " | ".join(column_names) + " |")
            response_parts.append("| " + " | ".join(["---"] * len(column_names)) + " |")
            
            # Data rows (limit to first 100 for display)
            display_limit = min(100, len(results))
            for row in results[:display_limit]:
                # Format each cell value
                cell_values = []
                for col in column_names:
                    value = row.get(col, '')
                    # Convert to string and escape pipe characters
                    if value is None:
                        cell_values.append('')
                    else:
                        cell_str = str(value).replace('|', '\\|')
                        # Truncate long values
                        if len(cell_str) > 50:
                            cell_str = cell_str[:47] + '...'
                        cell_values.append(cell_str)
                response_parts.append("| " + " | ".join(cell_values) + " |")
            
            if len(results) > display_limit:
                response_parts.append(f"\n*Showing first {display_limit} of {len(results)} results*")
        
        # Add execution time
        exec_time = result.get('execution_time_ms', 0)
        response_parts.append(f"\nExecution time: {exec_time} ms")
        
        logger.info(f"Query completed: {result['result_count']} rows in {exec_time}ms")
        return '\n'.join(response_parts)
        
    except Exception as e:
        logger.error(f"Query data workspace tool error: {e}")
        return f"Error querying data workspace: {str(e)}"


# Tool registry for LangGraph
DATA_WORKSPACE_TOOLS = {
    'list_data_workspaces': list_data_workspaces_tool,
    'get_workspace_schema': get_workspace_schema_tool,
    'query_data_workspace': query_data_workspace_tool
}
