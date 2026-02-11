"""
Org content tools - Read-only parsing and querying of org-mode file content.

Local implementations (no gRPC), like math_tools.py. Operate on editor content
injected at runtime via _editor_content. Used by the org_content automation skill.
"""

import json
import logging
from typing import Any, Dict, List

from orchestrator.utils.org_utils import parse_org_structure

logger = logging.getLogger(__name__)


def parse_org_structure_tool(_editor_content: str = "") -> str:
    """
    Parse the org-mode file structure: all headings with level, TODO state, tags,
    parent path, and subtree content preview. Returns JSON array of heading objects.
    Use this to understand the document outline before querying specific items.
    """
    if not (_editor_content or "").strip():
        return json.dumps({"error": "No org file content provided. Open an org file and try again."})
    try:
        structure = parse_org_structure(_editor_content)
        return json.dumps(structure, default=str)
    except Exception as e:
        logger.exception("parse_org_structure_tool failed")
        return json.dumps({"error": str(e)})


def list_org_todos_tool(
    state_filter: str = "",
    tag_filter: str = "",
    _editor_content: str = "",
) -> str:
    """
    List TODO items from the org file. Optional filters:
    - state_filter: TODO, DONE, NEXT, WAITING, HOLD, etc. (empty = all)
    - tag_filter: comma-separated tags; items must have at least one (empty = all)
    Returns JSON array of headings that have a TODO state.
    """
    if not (_editor_content or "").strip():
        return json.dumps({"error": "No org file content provided. Open an org file and try again."})
    try:
        structure = parse_org_structure(_editor_content)
        todos = [h for h in structure if h.get("todo_state")]
        if state_filter and state_filter.strip():
            want = state_filter.strip().upper()
            todos = [h for h in todos if (h.get("todo_state") or "").upper() == want]
        if tag_filter and tag_filter.strip():
            want_tags = {t.strip().lower() for t in tag_filter.split(",") if t.strip()}
            todos = [h for h in todos if want_tags & {t.lower() for t in (h.get("tags") or [])}]
        return json.dumps(todos, default=str)
    except Exception as e:
        logger.exception("list_org_todos_tool failed")
        return json.dumps({"error": str(e)})


def search_org_headings_tool(search_term: str, _editor_content: str = "") -> str:
    """
    Search headings and their subtree content for a keyword/phrase. search_term
    is case-insensitive. Returns JSON array of matching headings with their
    subtree_content preview.
    """
    if not (_editor_content or "").strip():
        return json.dumps({"error": "No org file content provided. Open an org file and try again."})
    if not (search_term or "").strip():
        return json.dumps({"error": "search_term is required."})
    try:
        structure = parse_org_structure(_editor_content)
        term = search_term.strip().lower()
        matches = [
            h
            for h in structure
            if term in (h.get("heading") or "").lower()
            or term in (h.get("subtree_content") or "").lower()
        ]
        return json.dumps(matches, default=str)
    except Exception as e:
        logger.exception("search_org_headings_tool failed")
        return json.dumps({"error": str(e)})


def get_org_statistics_tool(_editor_content: str = "") -> str:
    """
    Compute statistics for the org file: total headings, TODO counts by state,
    tag distribution, and completion rate (DONE / total with TODO state).
    Returns a JSON object with counts and rates.
    """
    if not (_editor_content or "").strip():
        return json.dumps({"error": "No org file content provided. Open an org file and try again."})
    try:
        structure = parse_org_structure(_editor_content)
        total_headings = len(structure)
        with_todo = [h for h in structure if h.get("todo_state")]
        by_state: Dict[str, int] = {}
        for h in with_todo:
            s = (h.get("todo_state") or "NONE").upper()
            by_state[s] = by_state.get(s, 0) + 1
        tag_counts: Dict[str, int] = {}
        for h in structure:
            for t in h.get("tags") or []:
                tag_counts[t] = tag_counts.get(t, 0) + 1
        total_todos = len(with_todo)
        done_count = by_state.get("DONE", 0)
        completion_rate = (done_count / total_todos) if total_todos else 0.0
        stats: Dict[str, Any] = {
            "total_headings": total_headings,
            "todo_count": total_todos,
            "todo_by_state": by_state,
            "tag_counts": tag_counts,
            "completion_rate": round(completion_rate, 2),
        }
        return json.dumps(stats, default=str)
    except Exception as e:
        logger.exception("get_org_statistics_tool failed")
        return json.dumps({"error": str(e)})
