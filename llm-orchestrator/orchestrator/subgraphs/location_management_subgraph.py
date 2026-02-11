"""
Location Management Subgraph

Reusable subgraph for CRUD operations on saved locations.
Used by Navigation Agent for create, list, delete location flows.

State: operation_type (create|list|delete), location_name, address, location_id, etc.
Outputs: operation_result (dict with success, location(s), or error).
"""

import logging
from typing import Dict, Any

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

LocationManagementState = Dict[str, Any]


async def execute_location_operation_node(state: LocationManagementState) -> Dict[str, Any]:
    """Call gRPC client for location CRUD based on operation_type."""
    try:
        from orchestrator.backend_tool_client import get_backend_tool_client

        user_id = state.get("user_id", "system")
        user_role = state.get("user_role", "user")
        operation_type = state.get("operation_type", "").strip().lower()

        client = await get_backend_tool_client()

        if operation_type == "create":
            name = (state.get("location_name") or "").strip()
            address = (state.get("address") or "").strip()
            if not name or not address:
                return {
                    "operation_result": {"success": False, "error": "Location name and address are required"},
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                }
            result = await client.create_location(
                user_id=user_id,
                name=name,
                address=address,
                latitude=state.get("latitude"),
                longitude=state.get("longitude"),
                notes=state.get("notes"),
                is_global=state.get("is_global", False),
                metadata=state.get("location_metadata"),
                user_role=user_role,
            )
        elif operation_type == "list":
            result = await client.list_locations(user_id=user_id, user_role=user_role)
        elif operation_type == "delete":
            location_id = (state.get("location_id") or "").strip()
            if not location_id:
                result = {"success": False, "error": "Location ID is required for delete"}
            else:
                result = await client.delete_location(
                    user_id=user_id,
                    location_id=location_id,
                    user_role=user_role,
                )
        else:
            result = {"success": False, "error": f"Unknown operation_type: {operation_type}"}

        return {
            "operation_result": result,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
    except Exception as e:
        logger.error(f"execute_location_operation_node failed: {e}")
        return {
            "operation_result": {"success": False, "error": str(e)},
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


async def format_location_result_node(state: LocationManagementState) -> Dict[str, Any]:
    """Format operation result for the agent (summary text + structured data)."""
    result = state.get("operation_result", {})
    success = result.get("success", False)
    operation_type = (state.get("operation_type") or "").strip().lower()

    if success:
        if operation_type == "create":
            loc = result
            summary = f"Created location '{loc.get('name', '')}' at {loc.get('address', '')} (ID: {loc.get('location_id', '')})."
        elif operation_type == "list":
            locations = result.get("locations", [])
            total = result.get("total", 0)
            if not locations:
                summary = "You have no saved locations."
            else:
                lines = [f"- **{loc.get('name', '')}**: {loc.get('address', '')} (ID: {loc.get('location_id', '')})" for loc in locations]
                summary = f"You have {total} saved location(s):\n" + "\n".join(lines)
        elif operation_type == "delete":
            summary = result.get("message", "Location deleted successfully.")
        else:
            summary = str(result)
    else:
        summary = result.get("error", "Operation failed.")

    return {
        "location_management_result": {"success": success, "summary": summary, "raw": result},
        "metadata": state.get("metadata", {}),
        "user_id": state.get("user_id", "system"),
        "shared_memory": state.get("shared_memory", {}),
        "messages": state.get("messages", []),
        "query": state.get("query", ""),
    }


def build_location_management_subgraph(checkpointer=None):
    """Build the location management subgraph (create, list, delete)."""
    workflow = StateGraph(LocationManagementState)

    workflow.add_node("execute_location_operation", execute_location_operation_node)
    workflow.add_node("format_location_result", format_location_result_node)

    workflow.set_entry_point("execute_location_operation")
    workflow.add_edge("execute_location_operation", "format_location_result")
    workflow.add_edge("format_location_result", END)

    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()
