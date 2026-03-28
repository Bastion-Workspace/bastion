"""Agent line UUID resolution for pipeline metadata (canonical key: line_id)."""

from typing import Any, Dict, Optional


def line_id_from_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Return the agent line UUID from orchestrator pipeline metadata.
    Prefer line_id; fall back to team_id for older clients and briefings.
    """
    if not metadata:
        return None
    raw = metadata.get("line_id") or metadata.get("team_id")
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None
