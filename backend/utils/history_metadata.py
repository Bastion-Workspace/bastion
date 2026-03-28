"""
Metadata allowlist for conversation history persistence.

Only these keys from orchestrator chunk metadata are stored on assistant messages.
Excluded: editor_operations, manuscript_edit, approval_queue_id,
static_visualization_data, chart_result, citations, sources (display-layer, not history).

images: JSON string of structured image refs (url + small metadata). Data-URI entries
are stripped via sanitize_images_for_persistence before filter_history_safe_metadata.
"""

import json
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
    "images",
})


def sanitize_images_for_persistence(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove data-URI image entries from metadata['images'] before DB save.
    Keeps URL-based refs (document file URLs, /api/images/...) for chat gallery reload.
    Mutates metadata in place.
    """
    images_raw = metadata.get("images")
    if not images_raw:
        return metadata
    try:
        images = json.loads(images_raw) if isinstance(images_raw, str) else images_raw
        if not isinstance(images, list):
            metadata.pop("images", None)
            return metadata
        url_images = [
            img
            for img in images
            if isinstance(img, dict) and not str(img.get("url", "")).startswith("data:")
        ]
        if url_images:
            metadata["images"] = json.dumps(url_images)
        else:
            metadata.pop("images", None)
    except (json.JSONDecodeError, TypeError):
        metadata.pop("images", None)
    return metadata


def filter_history_safe_metadata(metadata_received: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of metadata_received containing only keys safe to persist
    in conversation message metadata (no editor blobs, large charts, etc.).
    """
    if not metadata_received or not isinstance(metadata_received, dict):
        return {}
    return {
        k: v for k, v in metadata_received.items()
        if k in _HISTORY_SAFE_METADATA_KEYS and v is not None
    }
