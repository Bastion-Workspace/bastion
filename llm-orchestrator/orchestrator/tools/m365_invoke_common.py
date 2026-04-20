"""
Shared helper for Microsoft 365 Graph workload tools (To Do, OneDrive, OneNote, Planner).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)


def _format_m365_payload(operation: str, payload: Dict[str, Any]) -> str:
    if payload.get("error"):
        return f"M365 {operation} error: {payload['error']}"
    if not payload:
        return f"M365 {operation}: (empty result)"
    try:
        text = json.dumps(payload, indent=0, default=str)
    except TypeError:
        text = str(payload)
    if len(text) > 12000:
        return text[:12000] + "\n… (truncated)"
    return text


async def invoke_m365_graph(
    operation: str,
    user_id: str,
    connection_id: Optional[int],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call tools-service M365GraphInvoke and return a unified dict including formatted."""
    try:
        client = await get_backend_tool_client()
        raw = await client.m365_graph_invoke(
            user_id=user_id,
            operation=operation,
            connection_id=connection_id,
            params=params or {},
        )
        if isinstance(raw, dict):
            if "formatted" not in raw or not raw.get("formatted"):
                raw["formatted"] = _format_m365_payload(operation, raw)
            return raw
        return {
            "success": False,
            "formatted": str(raw),
        }
    except Exception as e:
        logger.exception("invoke_m365_graph failed: %s", operation)
        return {"success": False, "formatted": f"Error: {e}"}
