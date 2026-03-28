"""
Agent Workspace service - shared key-value scratchpad for agent teams (Blackboard pattern).

Agents in a team can write_to_workspace(key, value) and read_workspace(key) to share
artifacts (e.g. campaign_brief, competitor_analysis) without going through the CEO.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def _check_line_access(line_id: str, user_id: str) -> bool:
    """Return True if the line exists and belongs to the user."""
    from services.database_manager.database_helpers import fetch_one
    row = await fetch_one(
        "SELECT id FROM agent_lines WHERE id = $1 AND user_id = $2",
        line_id,
        user_id,
    )
    return row is not None


async def set_workspace_entry(
    line_id: str,
    key: str,
    value: str,
    user_id: str,
    updated_by_agent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Upsert a workspace entry for the team. Key must be non-empty."""
    from services.database_manager.database_helpers import fetch_one, execute

    if not await _check_line_access(line_id, user_id):
        return {"success": False, "error": "Team not found"}
    key = (key or "").strip()
    if not key:
        return {"success": False, "error": "key is required"}
    value = value or ""
    try:
        await execute(
            """
            INSERT INTO agent_line_workspace (line_id, key, value, updated_by_agent_id, updated_at)
            VALUES ($1, $2, $3, $4::uuid, NOW())
            ON CONFLICT (line_id, key)
            DO UPDATE SET value = EXCLUDED.value, updated_by_agent_id = EXCLUDED.updated_by_agent_id, updated_at = NOW()
            """,
            line_id,
            key,
            value,
            updated_by_agent_id,
        )
        row = await fetch_one(
            "SELECT id, line_id, key, value, updated_by_agent_id, updated_at FROM agent_line_workspace WHERE line_id = $1 AND key = $2",
            line_id,
            key,
        )
        if not row:
            return {"success": True, "key": key}
        return {
            "success": True,
            "key": key,
            "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
        }
    except Exception as e:
        logger.warning("set_workspace_entry failed: %s", e)
        return {"success": False, "error": str(e)}


async def get_workspace_entry(line_id: str, key: str, user_id: str) -> Dict[str, Any]:
    """Get a single workspace entry by key. Returns {} if not found or no access."""
    from services.database_manager.database_helpers import fetch_one

    if not await _check_line_access(line_id, user_id):
        return {"success": False, "value": None, "error": "Team not found"}
    key = (key or "").strip()
    if not key:
        return {"success": False, "value": None, "error": "key is required"}
    row = await fetch_one(
        """
        SELECT w.key, w.value, w.updated_at, w.updated_by_agent_id, p.name AS updated_by_agent_name
        FROM agent_line_workspace w
        LEFT JOIN agent_profiles p ON p.id = w.updated_by_agent_id
        JOIN agent_lines t ON t.id = w.line_id AND t.user_id = $3
        WHERE w.line_id = $1 AND w.key = $2
        """,
        line_id,
        key,
        user_id,
    )
    if not row:
        return {"success": True, "value": None, "key": key}
    return {
        "success": True,
        "key": row["key"],
        "value": row["value"],
        "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
        "updated_by_agent_id": str(row["updated_by_agent_id"]) if row.get("updated_by_agent_id") else None,
        "updated_by_agent_name": row.get("updated_by_agent_name"),
    }


async def list_workspace(line_id: str, user_id: str) -> Dict[str, Any]:
    """List all workspace keys for the team (with updated_at and updated_by_agent_name)."""
    from services.database_manager.database_helpers import fetch_all

    if not await _check_line_access(line_id, user_id):
        return {"success": False, "entries": [], "error": "Team not found"}
    rows = await fetch_all(
        """
        SELECT w.key, w.updated_at, w.updated_by_agent_id, p.name AS updated_by_agent_name
        FROM agent_line_workspace w
        LEFT JOIN agent_profiles p ON p.id = w.updated_by_agent_id
        JOIN agent_lines t ON t.id = w.line_id AND t.user_id = $2
        WHERE w.line_id = $1
        ORDER BY w.updated_at DESC
        """,
        line_id,
        user_id,
    )
    entries = [
        {
            "key": r["key"],
            "updated_at": r["updated_at"].isoformat() if r.get("updated_at") else None,
            "updated_by_agent_id": str(r["updated_by_agent_id"]) if r.get("updated_by_agent_id") else None,
            "updated_by_agent_name": r.get("updated_by_agent_name"),
        }
        for r in rows
    ]
    return {"success": True, "entries": entries}


async def delete_workspace_entry(line_id: str, key: str, user_id: str) -> Dict[str, Any]:
    """Delete a workspace entry by key."""
    from services.database_manager.database_helpers import execute

    if not await _check_line_access(line_id, user_id):
        return {"success": False, "error": "Team not found"}
    key = (key or "").strip()
    if not key:
        return {"success": False, "error": "key is required"}
    try:
        await execute(
            "DELETE FROM agent_line_workspace WHERE line_id = $1 AND key = $2",
            line_id,
            key,
        )
        return {"success": True, "key": key}
    except Exception as e:
        logger.warning("delete_workspace_entry failed: %s", e)
        return {"success": False, "error": str(e)}


async def clear_all_workspace(line_id: str, user_id: str) -> int:
    """Delete all workspace entries for the team. Returns count deleted. Verifies team ownership."""
    from services.database_manager.database_helpers import fetch_one

    if not await _check_line_access(line_id, user_id):
        return 0
    row = await fetch_one(
        "WITH d AS (DELETE FROM agent_line_workspace WHERE line_id = $1 RETURNING id) SELECT COUNT(*)::int AS n FROM d",
        line_id,
    )
    return (row.get("n") or 0) if row else 0
