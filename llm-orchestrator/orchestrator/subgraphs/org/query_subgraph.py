"""
Org Query Subgraph

Classifies query type and semantically matches user query to relevant headings, TODOs, and projects.
"""

import json
import logging
import re
from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


class OrgQueryState(TypedDict):
    """State for org query subgraph"""
    # Inputs
    query: str
    org_structure: List[Dict[str, Any]]
    todos: List[Dict[str, Any]]
    projects: List[Dict[str, Any]]
    all_tags: Dict[str, List[str]]
    content: str  # Full org file content (for preservation)
    filename: str
    # Outputs
    matched_headings: List[Dict[str, Any]]    # Semantically relevant
    matched_todos: List[Dict[str, Any]]       # Relevant TODOs
    matched_projects: List[Dict[str, Any]]    # Relevant projects
    query_type: str                           # "todo" | "project" | "content" | "tag"
    # Critical 5
    metadata: Dict[str, Any]
    user_id: str
    shared_memory: Dict[str, Any]
    messages: List[Any]


async def classify_query_type_node(state: OrgQueryState) -> Dict[str, Any]:
    """Use LLM to classify query type: todo/project/content/tag"""
    try:
        query = state.get("query", "")
        
        if not query:
            return {
                "query_type": "content",
                # ✅ CRITICAL 5 preserved
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": query,
                # Preserve parsed data
                "org_structure": state.get("org_structure", []),
                "todos": state.get("todos", []),
                "projects": state.get("projects", []),
                "all_tags": state.get("all_tags", {}),
                "content": state.get("content", ""),
                "filename": state.get("filename", "")
            }
        
        # Get LLM for classification
        from orchestrator.agents.base_agent import BaseAgent
        # We need to get LLM - create a temporary base agent instance for _get_llm
        # Actually, we'll need to pass LLM access through state or use a helper
        # For now, use simple keyword-based classification with LLM fallback
        
        query_lower = query.lower()
        
        # Simple keyword-based classification
        if any(kw in query_lower for kw in ["todo", "task", "done", "complete", "pending", "waiting", "started"]):
            query_type = "todo"
        elif any(kw in query_lower for kw in ["project", "projects"]):
            query_type = "project"
        elif any(kw in query_lower for kw in ["tag", "tagged", ":"]):
            query_type = "tag"
        else:
            query_type = "content"
        
        logger.info(f"Classified query type: {query_type}")
        
        return {
            "query_type": query_type,
            # ✅ CRITICAL 5 preserved
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": query,
            # Preserve parsed data
            "org_structure": state.get("org_structure", []),
            "todos": state.get("todos", []),
            "projects": state.get("projects", []),
            "all_tags": state.get("all_tags", {}),
            "content": state.get("content", ""),
            "filename": state.get("filename", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to classify query type: {e}")
        return {
            "query_type": "content",
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
            "all_tags": state.get("all_tags", {})
        }


async def semantic_matching_node(state: OrgQueryState) -> Dict[str, Any]:
    """Use LLM to semantically match query to headings"""
    try:
        query = state.get("query", "")
        org_structure = state.get("org_structure", [])
        
        if not org_structure or not query:
            return {
                "matched_headings": [],
                # ✅ CRITICAL 5 preserved
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": query,
                # Preserve all state
                "org_structure": org_structure,
                "todos": state.get("todos", []),
                "projects": state.get("projects", []),
                "all_tags": state.get("all_tags", {}),
                "query_type": state.get("query_type", "content")
            }
        
        # Build heading summary for LLM
        heading_summary = []
        for heading_info in org_structure[:50]:  # Limit to first 50 headings
            level = heading_info.get("level", 1)
            heading = heading_info.get("heading", "")
            tags = heading_info.get("tags", [])
            todo_state = heading_info.get("todo_state")
            parent_path = heading_info.get("parent_path", [])
            
            indent = "  " * (level - 1)
            todo_str = f"{todo_state} " if todo_state else ""
            tags_str = f" :{':'.join(tags)}:" if tags else ""
            path_str = f" ({' > '.join(parent_path)})" if parent_path else ""
            
            heading_summary.append(f"{indent}* {todo_str}{heading}{tags_str}{path_str}")
        
        # Get LLM - we need access to BaseAgent's _get_llm method
        # For now, we'll use a helper function that gets LLM from metadata
        from orchestrator.agents.base_agent import BaseAgent
        
        # Create temporary agent instance to access _get_llm
        # Actually, we should pass LLM through state or use a different approach
        # Let's use a helper that creates a minimal agent for LLM access
        temp_agent = BaseAgent("temp_org_query")
        llm = temp_agent._get_llm(temperature=0.3, state=state)
        
        system_prompt = """You are an Org Mode query analyzer. Your task is to identify which headings in an Org file are relevant to the user's query.

Analyze the user's query and the org file structure. Return a JSON array of heading texts that are relevant to the query, ordered by relevance (most relevant first).

Consider:
- Direct matches (heading text contains query keywords)
- Semantic matches (heading describes similar concepts)
- Context matches (parent headings provide relevant context)
- Tag matches (tags indicate relevance: :project:, :active:, etc.)

Return ONLY a JSON array of heading texts, nothing else. Example: ["Replace Garage Doors", "Home Improvement Projects"]"""

        user_prompt = f"""USER QUERY: {query}

ORG FILE STRUCTURE:
{chr(10).join(heading_summary)}

Which headings are relevant to this query? Return a JSON array of heading texts, ordered by relevance."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = await llm.ainvoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON response
        json_text = response_text.strip()
        if '```json' in json_text:
            match = re.search(r'```json\s*\n(.*?)\n```', json_text, re.DOTALL)
            if match:
                json_text = match.group(1).strip()
        elif '```' in json_text:
            match = re.search(r'```\s*\n(.*?)\n```', json_text, re.DOTALL)
            if match:
                json_text = match.group(1).strip()
        
        matched_heading_texts = json.loads(json_text)
        if not isinstance(matched_heading_texts, list):
            matched_heading_texts = []
        
        # Find matching headings in structure
        matched_headings = []
        for heading_text in matched_heading_texts:
            for heading_info in org_structure:
                if heading_info.get("heading", "").strip() == heading_text.strip():
                    matched_headings.append(heading_info)
                    break
        
        logger.info(f"Semantically matched {len(matched_headings)} headings to query")
        
        return {
            "matched_headings": matched_headings,
            # ✅ CRITICAL 5 preserved
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": query,
            # Preserve all state
            "org_structure": org_structure,
            "todos": state.get("todos", []),
            "projects": state.get("projects", []),
            "all_tags": state.get("all_tags", {}),
            "query_type": state.get("query_type", "content"),
            "content": state.get("content", ""),
            "filename": state.get("filename", "")
        }
        
    except Exception as e:
        logger.warning(f"Failed to semantically match query to headings: {e}")
        # Fallback to simple text matching
        query_lower = query.lower()
        matched_headings = []
        org_structure = state.get("org_structure", [])
        for heading_info in org_structure:
            heading = heading_info.get("heading", "").lower()
            if any(word in heading for word in query_lower.split() if len(word) > 3):
                matched_headings.append(heading_info)
        
        return {
            "matched_headings": matched_headings[:5],  # Limit fallback results
            # ✅ CRITICAL 5 preserved even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "org_structure": org_structure,
            "todos": state.get("todos", []),
            "projects": state.get("projects", []),
            "all_tags": state.get("all_tags", {}),
            "query_type": state.get("query_type", "content")
        }


async def filter_relevant_content_node(state: OrgQueryState) -> Dict[str, Any]:
    """Filter TODOs and projects based on query type and matched headings"""
    try:
        query_type = state.get("query_type", "content")
        matched_headings = state.get("matched_headings", [])
        todos = state.get("todos", [])
        projects = state.get("projects", [])
        all_tags = state.get("all_tags", {})
        query = state.get("query", "").lower()
        
        matched_todos = []
        matched_projects = []
        
        # Build set of matched heading texts for filtering
        matched_heading_texts = {h.get("heading", "") for h in matched_headings}
        
        if query_type == "todo":
            # Filter TODOs based on matched headings or query keywords
            for todo in todos:
                heading = todo.get("heading", "")
                if heading in matched_heading_texts:
                    matched_todos.append(todo)
                elif any(kw in query for kw in [todo.get("todo_state", "").lower(), "all", "every"]):
                    matched_todos.append(todo)
        elif query_type == "project":
            # Filter projects based on matched headings
            for project in projects:
                heading = project.get("heading", "")
                if heading in matched_heading_texts or any(kw in heading.lower() for kw in query.split() if len(kw) > 3):
                    matched_projects.append(project)
        elif query_type == "tag":
            # Filter by tag mentions in query
            for tag in all_tags.keys():
                if tag.lower() in query:
                    # Find headings with this tag
                    for heading_text in all_tags[tag]:
                        for heading_info in matched_headings:
                            if heading_info.get("heading", "") == heading_text:
                                if heading_info.get("todo_state"):
                                    # Add as TODO if it has a TODO state
                                    for todo in todos:
                                        if todo.get("heading", "") == heading_text:
                                            matched_todos.append(todo)
                                elif "project" in heading_info.get("tags", []):
                                    # Add as project if tagged
                                    for project in projects:
                                        if project.get("heading", "") == heading_text:
                                            matched_projects.append(project)
        else:
            # Content query - use matched headings to find related TODOs/projects
            for todo in todos:
                if todo.get("heading", "") in matched_heading_texts:
                    matched_todos.append(todo)
            for project in projects:
                if project.get("heading", "") in matched_heading_texts:
                    matched_projects.append(project)
        
        logger.info(f"Filtered to {len(matched_todos)} TODOs and {len(matched_projects)} projects")
        
        return {
            "matched_todos": matched_todos,
            "matched_projects": matched_projects,
            # ✅ CRITICAL 5 preserved
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # Preserve all state
            "org_structure": state.get("org_structure", []),
            "todos": todos,
            "projects": projects,
            "all_tags": all_tags,
            "matched_headings": matched_headings,
            "query_type": query_type,
            "content": state.get("content", ""),
            "filename": state.get("filename", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to filter relevant content: {e}")
        return {
            "matched_todos": [],
            "matched_projects": [],
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
            "matched_headings": state.get("matched_headings", []),
            "query_type": state.get("query_type", "content")
        }


def build_org_query_subgraph(checkpointer) -> StateGraph:
    """Build subgraph for querying org content"""
    subgraph = StateGraph(OrgQueryState)
    
    subgraph.add_node("classify_query_type", classify_query_type_node)
    subgraph.add_node("semantic_matching", semantic_matching_node)
    subgraph.add_node("filter_relevant_content", filter_relevant_content_node)
    
    subgraph.set_entry_point("classify_query_type")
    subgraph.add_edge("classify_query_type", "semantic_matching")
    subgraph.add_edge("semantic_matching", "filter_relevant_content")
    subgraph.add_edge("filter_relevant_content", END)
    
    return subgraph.compile(checkpointer=checkpointer)
