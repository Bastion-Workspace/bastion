"""
MCP tool invocation helpers: execution goes through Tools Service (ExecuteMcpTool).
"""

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


async def run_mcp_tool_invocation(
    user_id: str,
    server_id: int,
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call the backend MCP client for one tool."""
    from orchestrator.backend_tool_client import get_backend_tool_client

    client = await get_backend_tool_client()
    return await client.execute_mcp_tool(
        user_id=user_id,
        server_id=int(server_id),
        tool_name=tool_name or "",
        arguments_json=json.dumps(arguments or {}),
    )
