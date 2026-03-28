"""
Org content tools - Read-only parsing and querying of org-mode file content.

Local implementations (no gRPC), like math_tools.py. Operate on editor content
injected at runtime via _editor_content. Used by the org_content automation skill.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.utils.org_utils import parse_org_structure
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


class ParseOrgStructureInputs(BaseModel):
    """Optional: editor content is usually injected at runtime."""
    pass


class ParseOrgStructureOutputs(BaseModel):
    """Output of parse_org_structure."""
    structure: List[Dict[str, Any]] = Field(description="List of heading objects")
    heading_count: int = Field(description="Number of headings")
    formatted: str = Field(description="Human-readable summary")


class SearchOrgHeadingsInputs(BaseModel):
    """Inputs for searching org headings."""
    search_term: str = Field(description="Keyword or phrase to search for")


class SearchOrgHeadingsOutputs(BaseModel):
    """Output of search_org_headings."""
    headings: List[Dict[str, Any]] = Field(description="Matching headings")
    count: int = Field(description="Number of matches")
    formatted: str = Field(description="Human-readable summary")


class GetOrgStatisticsOutputs(BaseModel):
    """Output of get_org_statistics."""
    statistics: Dict[str, Any] = Field(description="Counts and rates")
    total_headings: int = Field(description="Total number of headings")
    todo_count: int = Field(description="Total number of TODOs")
    completion_rate: float = Field(description="Completion rate (0.0-1.0)")
    formatted: str = Field(description="Human-readable summary")


def parse_org_structure_tool(_editor_content: str = "") -> Dict[str, Any]:
    """
    Parse the org-mode file structure: all headings with level, TODO state, tags,
    parent path, and subtree content preview. Returns structured dict with structure and formatted.
    """
    err_msg = "No org file content provided. Open an org file and try again."
    if not (_editor_content or "").strip():
        return {"structure": [], "heading_count": 0, "formatted": err_msg}
    try:
        structure = parse_org_structure(_editor_content)
        count = len(structure) if isinstance(structure, list) else 0
        formatted = f"Parsed {count} heading(s)." if count else "No headings found."
        return {"structure": structure, "heading_count": count, "formatted": formatted}
    except Exception as e:
        logger.exception("parse_org_structure_tool failed")
        return {"structure": [], "heading_count": 0, "formatted": f"Error: {e}"}


def search_org_headings_tool(search_term: str = "", _editor_content: str = "") -> Dict[str, Any]:
    """
    Search headings and their subtree content for a keyword/phrase. Returns structured dict with headings, count, formatted.
    """
    if not (_editor_content or "").strip():
        return {"headings": [], "count": 0, "formatted": "No org file content provided. Open an org file and try again."}
    if not (search_term or "").strip():
        return {"headings": [], "count": 0, "formatted": "search_term is required."}
    try:
        structure = parse_org_structure(_editor_content)
        term = search_term.strip().lower()
        matches = [
            h
            for h in structure
            if term in (h.get("heading") or "").lower()
            or term in (h.get("subtree_content") or "").lower()
        ]
        formatted_parts = [f"Found {len(matches)} matching heading(s)."]
        for i, h in enumerate(matches[:10], 1):
            level = h.get("level", 0)
            title = h.get("heading") or ""
            todo = h.get("todo_state") or ""
            tags_list = h.get("tags") or []
            tags_str = " ".join(f":{t}:" for t in tags_list) if tags_list else ""
            parent_path = h.get("parent_path") or []
            path_str = " > ".join(parent_path) if parent_path else ""
            subtree = (h.get("subtree_content") or "")[:200].strip()
            if len((h.get("subtree_content") or "")) > 200:
                subtree += "..."
            line = f"{i}. [Level {level}] {title}"
            if todo:
                line += f" [{todo}]"
            if tags_str:
                line += f" {tags_str}"
            if path_str:
                line += f" (under: {path_str})"
            formatted_parts.append(line)
            if subtree:
                formatted_parts.append("   " + subtree.replace("\n", "\n   "))
        if len(matches) > 10:
            formatted_parts.append(f"... and {len(matches) - 10} more.")
        formatted = "\n".join(formatted_parts)
        return {"headings": matches, "count": len(matches), "formatted": formatted}
    except Exception as e:
        logger.exception("search_org_headings_tool failed")
        return {"headings": [], "count": 0, "formatted": f"Error: {e}"}


def get_org_statistics_tool(_editor_content: str = "") -> Dict[str, Any]:
    """
    Compute statistics for the org file: total headings, TODO counts by state,
    tag distribution, and completion rate. Returns structured dict with statistics and formatted.
    """
    if not (_editor_content or "").strip():
        return {"statistics": {}, "total_headings": 0, "todo_count": 0, "completion_rate": 0.0, "formatted": "No org file content provided. Open an org file and try again."}
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
        formatted = f"Headings: {total_headings}, TODOs: {total_todos}, completion: {stats['completion_rate']}."
        return {
            "statistics": stats,
            "total_headings": total_headings,
            "todo_count": total_todos,
            "completion_rate": round(completion_rate, 2),
            "formatted": formatted,
        }
    except Exception as e:
        logger.exception("get_org_statistics_tool failed")
        return {"statistics": {}, "total_headings": 0, "todo_count": 0, "completion_rate": 0.0, "formatted": f"Error: {e}"}


register_action(
    name="parse_org_structure",
    category="org",
    description="Parse org-mode file structure (headings, TODO state, tags)",
    inputs_model=ParseOrgStructureInputs,
    outputs_model=ParseOrgStructureOutputs,
    tool_function=parse_org_structure_tool,
)
register_action(
    name="search_org_headings",
    category="org",
    description="Search org headings and subtree content for a keyword or phrase",
    inputs_model=SearchOrgHeadingsInputs,
    outputs_model=SearchOrgHeadingsOutputs,
    tool_function=search_org_headings_tool,
)
register_action(
    name="get_org_statistics",
    category="org",
    description="Compute org file statistics (headings, TODO counts, completion rate)",
    inputs_model=ParseOrgStructureInputs,
    outputs_model=GetOrgStatisticsOutputs,
    tool_function=get_org_statistics_tool,
)
