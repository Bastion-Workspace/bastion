"""
Metadata allowlist for conversation history persistence.

Only these keys from orchestrator chunk metadata are stored on assistant messages.
Excluded: images, editor_operations, manuscript_edit, approval_queue_id,
static_visualization_data, chart_result, citations, sources (display-layer, not history).
"""

from typing import Any, Dict

_HISTORY_SAFE_METADATA_KEYS = frozenset({
    "tool_call_summary",
    "agent_profile_id",
    "agent_display_name",
    "task_status",
    "tools_used_categories",
    "delegated_agent",
    "duration_ms",
    "skills_used",
    "line_id",
    "line_role",
    "line_agent_handle",
    "delegated_by",
    "line_dispatch_sub_agent",
})


def filter_history_safe_metadata(metadata_received: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of metadata_received containing only keys safe to persist
    in conversation message metadata (no editor blobs, images, charts, etc.).
    """
    if not metadata_received or not isinstance(metadata_received, dict):
        return {}
    return {
        k: v for k, v in metadata_received.items()
        if k in _HISTORY_SAFE_METADATA_KEYS and v is not None
    }
