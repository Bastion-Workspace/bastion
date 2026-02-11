"""
Org Parsing Subgraph

Parses org file structure, extracts TODOs, projects, tags, and computes statistics.
"""

import logging
from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END

from orchestrator.utils.org_utils import parse_org_structure, extract_tags_from_content

logger = logging.getLogger(__name__)


class OrgParsingState(TypedDict):
    """State for org parsing subgraph"""
    # Inputs
    content: str
    filename: str
    # Outputs
    org_structure: List[Dict[str, Any]]  # All headings with hierarchy
    todos: List[Dict[str, Any]]          # All TODO items
    projects: List[Dict[str, Any]]       # Entries tagged :project:
    all_tags: Dict[str, List[str]]       # Tag → headings mapping
    statistics: Dict[str, Any]           # Counts, completion rates
    # Critical 5 (ALWAYS preserved)
    metadata: Dict[str, Any]
    user_id: str
    shared_memory: Dict[str, Any]
    messages: List[Any]
    query: str


async def extract_headings_node(state: OrgParsingState) -> Dict[str, Any]:
    """Extract all headings from org file content"""
    try:
        content = state.get("content", "")
        filename = state.get("filename", "")
        
        if not content:
            logger.warning(f"No content provided for parsing in {filename}")
            return {
                "org_structure": [],
                "todos": [],
                "projects": [],
                "all_tags": {},
                "statistics": {
                    "total_headings": 0,
                    "total_todos": 0,
                    "total_projects": 0,
                    "total_tags": 0
                },
                # ✅ CRITICAL 5 preserved
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "content": content,
                "filename": filename
            }
        
        # Parse org structure
        org_structure = parse_org_structure(content)
        logger.info(f"Parsed {len(org_structure)} headings from {filename}")
        
        return {
            "org_structure": org_structure,
            # ✅ CRITICAL 5 preserved
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "content": content,
            "filename": filename
        }
        
    except Exception as e:
        logger.error(f"Failed to extract headings: {e}")
        return {
            "org_structure": [],
            "todos": [],
            "projects": [],
            "all_tags": {},
            "statistics": {},
            "error": str(e),
            # ✅ CRITICAL 5 preserved even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "content": state.get("content", ""),
            "filename": state.get("filename", "")
        }


async def identify_todos_node(state: OrgParsingState) -> Dict[str, Any]:
    """Identify all TODO items from org structure"""
    try:
        org_structure = state.get("org_structure", [])
        
        todos = []
        for heading_info in org_structure:
            todo_state = heading_info.get("todo_state")
            if todo_state:
                todos.append({
                    "heading": heading_info.get("heading", ""),
                    "level": heading_info.get("level", 1),
                    "todo_state": todo_state,
                    "tags": heading_info.get("tags", []),
                    "parent_path": heading_info.get("parent_path", []),
                    "line_number": heading_info.get("line_number", 0),
                    "subtree_content": heading_info.get("subtree_content", "")
                })
        
        logger.info(f"Identified {len(todos)} TODO items")
        
        return {
            "todos": todos,
            # ✅ CRITICAL 5 preserved
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # Domain-specific preserved
            "org_structure": org_structure,
            "content": state.get("content", ""),
            "filename": state.get("filename", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to identify TODOs: {e}")
        return {
            "todos": [],
            "error": str(e),
            # ✅ CRITICAL 5 preserved even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "org_structure": state.get("org_structure", []),
            "content": state.get("content", ""),
            "filename": state.get("filename", "")
        }


async def extract_tags_node(state: OrgParsingState) -> Dict[str, Any]:
    """Extract tags and identify projects"""
    try:
        org_structure = state.get("org_structure", [])
        content = state.get("content", "")
        cursor_offset = state.get("shared_memory", {}).get("active_editor", {}).get("cursor_offset", -1)
        
        # Extract tags from content
        tags_info = extract_tags_from_content(content, cursor_offset)
        
        # Build tag → headings mapping
        all_tags = {}
        for tag_info in tags_info:
            for tag in tag_info.get("tags", []):
                if tag not in all_tags:
                    all_tags[tag] = []
                all_tags[tag].append(tag_info.get("heading", ""))
        
        # Identify projects (entries tagged :project:)
        projects = []
        for heading_info in org_structure:
            tags = heading_info.get("tags", [])
            if "project" in tags:
                projects.append({
                    "heading": heading_info.get("heading", ""),
                    "level": heading_info.get("level", 1),
                    "tags": tags,
                    "parent_path": heading_info.get("parent_path", []),
                    "line_number": heading_info.get("line_number", 0),
                    "subtree_content": heading_info.get("subtree_content", ""),
                    "todo_state": heading_info.get("todo_state")
                })
        
        logger.info(f"Extracted {len(all_tags)} unique tags, identified {len(projects)} projects")
        
        return {
            "all_tags": all_tags,
            "projects": projects,
            # ✅ CRITICAL 5 preserved
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # Domain-specific preserved
            "org_structure": org_structure,
            "todos": state.get("todos", []),
            "content": content,
            "filename": state.get("filename", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to extract tags: {e}")
        return {
            "all_tags": {},
            "projects": [],
            "error": str(e),
            # ✅ CRITICAL 5 preserved even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "org_structure": state.get("org_structure", []),
            "todos": state.get("todos", []),
            "content": state.get("content", ""),
            "filename": state.get("filename", "")
        }


async def compute_statistics_node(state: OrgParsingState) -> Dict[str, Any]:
    """Compute statistics about the org file"""
    try:
        org_structure = state.get("org_structure", [])
        todos = state.get("todos", [])
        projects = state.get("projects", [])
        all_tags = state.get("all_tags", {})
        
        # Count TODO states
        todo_counts = {}
        for todo in todos:
            state_name = todo.get("todo_state", "UNKNOWN")
            todo_counts[state_name] = todo_counts.get(state_name, 0) + 1
        
        # Count completion
        completed = todo_counts.get("DONE", 0) + todo_counts.get("CANCELED", 0) + todo_counts.get("CANCELLED", 0)
        total_todos = len(todos)
        completion_rate = (completed / total_todos * 100) if total_todos > 0 else 0
        
        statistics = {
            "total_headings": len(org_structure),
            "total_todos": total_todos,
            "total_projects": len(projects),
            "total_tags": len(all_tags),
            "todo_counts": todo_counts,
            "completion_rate": round(completion_rate, 1),
            "completed_todos": completed,
            "active_todos": total_todos - completed
        }
        
        logger.info(f"Computed statistics: {statistics}")
        
        return {
            "statistics": statistics,
            # ✅ CRITICAL 5 preserved
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # Domain-specific preserved
            "org_structure": org_structure,
            "todos": todos,
            "projects": projects,
            "all_tags": all_tags,
            "content": state.get("content", ""),
            "filename": state.get("filename", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to compute statistics: {e}")
        return {
            "statistics": {},
            "error": str(e),
            # ✅ CRITICAL 5 preserved even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "org_structure": state.get("org_structure", []),
            "todos": state.get("todos", []),
            "projects": state.get("projects", []),
            "all_tags": state.get("all_tags", {}),
            "content": state.get("content", ""),
            "filename": state.get("filename", "")
        }


def build_org_parsing_subgraph(checkpointer) -> StateGraph:
    """Build subgraph for parsing org file structure"""
    subgraph = StateGraph(OrgParsingState)
    
    subgraph.add_node("extract_headings", extract_headings_node)
    subgraph.add_node("identify_todos", identify_todos_node)
    subgraph.add_node("extract_tags", extract_tags_node)
    subgraph.add_node("compute_statistics", compute_statistics_node)
    
    subgraph.set_entry_point("extract_headings")
    subgraph.add_edge("extract_headings", "identify_todos")
    subgraph.add_edge("identify_todos", "extract_tags")
    subgraph.add_edge("extract_tags", "compute_statistics")
    subgraph.add_edge("compute_statistics", END)
    
    return subgraph.compile(checkpointer=checkpointer)
