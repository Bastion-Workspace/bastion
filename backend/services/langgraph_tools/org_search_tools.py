"""
Org Search Tools - Roosevelt's "Org-Mode Research Cavalry"
Tools for searching across all org-mode files with structured parsing
"""

import logging
from typing import Dict, Any, Optional, List

from services.org_search_service import OrgSearchService

logger = logging.getLogger(__name__)


# Singleton instance
_org_search_service: Optional[OrgSearchService] = None


def _get_org_search_service() -> OrgSearchService:
    """Get or create the org search service instance"""
    global _org_search_service
    
    if _org_search_service is None:
        _org_search_service = OrgSearchService()
        logger.info("🗂️ BULLY! Org Search Service initialized for agents!")
    
    return _org_search_service


async def search_org_files(
    query: str,
    tags: Optional[List[str]] = None,
    todo_states: Optional[List[str]] = None,
    include_content: bool = True,
    limit: int = 50,
    user_id: str = None
) -> str:
    """
    Search across all user's org-mode files with full-text and metadata support
    
    **BULLY!** Search TODOs, projects, notes, and journal entries in org files!
    
    Args:
        query: Search query string (searches headings and content)
        tags: Filter by org tags (e.g., ["@work", "@home", "urgent"])
        todo_states: Filter by TODO states (e.g., ["TODO", "NEXT", "WAITING"])
        include_content: Include content in results or just headings
        limit: Maximum number of results (default: 50)
        user_id: User ID to search files for
    
    Returns:
        Formatted string with search results including file paths, headings, tags, and content
    
    Examples:
        - search_org_files("quarterly report", tags=["@work"], todo_states=["TODO"])
        - search_org_files("meeting notes", include_content=True)
        - search_org_files("", tags=["urgent"], todo_states=["TODO", "NEXT"])  # All urgent TODOs
    
    Use Cases:
        - Find specific TODO items: "What tasks do I have related to X?"
        - Search project notes: "What notes do I have about the Y project?"
        - Filter by status: "Show me all WAITING tasks"
        - Find by tags: "What's tagged with @home?"
        - Discover journal entries: "What did I write about Z?"
    """
    try:
        if not user_id:
            return "❌ Error: user_id required for org file search"
        
        logger.info(f"🔍 ORG SEARCH: query='{query}', tags={tags}, todo_states={todo_states}, user={user_id}")
        
        service = _get_org_search_service()
        result = await service.search_org_files(
            user_id=user_id,
            query=query,
            tags=tags,
            todo_states=todo_states,
            include_content=include_content,
            limit=limit
        )
        
        if not result.get("success"):
            return f"❌ Org search failed: {result.get('error', 'Unknown error')}"
        
        results = result.get("results", [])
        count = result.get("count", 0)
        
        if count == 0:
            return f"No org-mode entries found for query: '{query}' (tags: {tags}, states: {todo_states})"
        
        # Format results for agent consumption
        formatted = f"Found {count} org-mode entries:\n\n"
        
        for i, item in enumerate(results[:limit], 1):
            formatted += f"**Result {i}:**\n"
            formatted += f"- File: {item['filename']}\n"
            formatted += f"- Heading: {item['heading']}\n"
            
            if item.get('todo_state'):
                formatted += f"- Status: {item['todo_state']}\n"
            
            if item.get('tags'):
                formatted += f"- Tags: {', '.join(item['tags'])}\n"
            
            if item.get('scheduled'):
                formatted += f"- Scheduled: {item['scheduled']}\n"
            
            if item.get('deadline'):
                formatted += f"- Deadline: {item['deadline']}\n"
            
            if item.get('priority'):
                formatted += f"- Priority: {item['priority']}\n"
            
            if include_content and item.get('content'):
                # Truncate long content
                content = item['content'][:300]
                if len(item['content']) > 300:
                    content += "..."
                formatted += f"- Content: {content}\n"
            
            formatted += "\n"
        
        if count > limit:
            formatted += f"\n(Showing {limit} of {count} results. Refine query for more specific results.)\n"
        
        logger.info(f"✅ ORG SEARCH: Returned {min(count, limit)} formatted results")
        return formatted
        
    except Exception as e:
        logger.error(f"❌ Org search tool failed: {e}")
        return f"❌ Error searching org files: {str(e)}"


async def list_org_todos(
    user_id: str,
    todo_states: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    limit: int = 50
) -> str:
    """
    List all TODO items from user's org files
    
    **By George!** Quick way to see all tasks across org files!
    
    Args:
        user_id: User ID to search files for
        todo_states: Filter by states (default: ["TODO", "NEXT", "WAITING"])
        tags: Filter by tags (e.g., ["@work", "urgent"])
        limit: Maximum number of results
    
    Returns:
        Formatted list of TODO items with status, tags, and scheduling
    """
    if not todo_states:
        todo_states = ["TODO", "NEXT", "WAITING"]
    
    return await search_org_files(
        query="",  # Empty query = match all
        tags=tags,
        todo_states=todo_states,
        include_content=False,  # Just headings for TODO lists
        limit=limit,
        user_id=user_id
    )


async def search_org_by_tag(
    tag: str,
    user_id: str,
    limit: int = 30
) -> str:
    """
    Search org files by a specific tag
    
    Args:
        tag: Tag to search for (e.g., "@work", "project", "urgent")
        user_id: User ID to search files for
        limit: Maximum number of results
    
    Returns:
        Formatted results for all entries with the specified tag
    """
    return await search_org_files(
        query="",
        tags=[tag],
        include_content=True,
        limit=limit,
        user_id=user_id
    )


# Tool registration info
ORG_SEARCH_TOOLS = {
    "search_org_files": search_org_files,
    "list_org_todos": list_org_todos,
    "search_org_by_tag": search_org_by_tag,
}


ORG_SEARCH_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_org_files",
            "description": "Search across all user's org-mode files (.org) with full-text and metadata support. Use for finding TODOs, project notes, journal entries, or any org content. Can filter by tags and TODO states.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string (searches headings and content). Use empty string '' to match all entries with filters."
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: Filter by org tags (e.g., ['@work', '@home', 'urgent'])"
                    },
                    "todo_states": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: Filter by TODO states (e.g., ['TODO', 'NEXT', 'WAITING', 'DONE'])"
                    },
                    "include_content": {
                        "type": "boolean",
                        "description": "Include entry content in results (default: true)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 50)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_org_todos",
            "description": "List all TODO items from user's org files. Quick way to see tasks across all org files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todo_states": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by states (default: ['TODO', 'NEXT', 'WAITING'])"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: Filter by tags"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 50)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_org_by_tag",
            "description": "Search org files by a specific tag. Useful for finding all entries tagged with a particular category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "Tag to search for (e.g., '@work', 'project', 'urgent')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 30)"
                    }
                },
                "required": ["tag"]
            }
        }
    }
]



