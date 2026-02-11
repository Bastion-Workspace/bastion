"""
Dynamic Tool Analyzer for LLM Orchestrator Agents
Analyzes queries to determine which tools are needed for dynamic tool usage tracking

Now includes hybrid semantic + keyword-based discovery for next-generation tool selection.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class OrchestratorToolCategory:
    """Tool categories for llm-orchestrator agents"""
    SEARCH_LOCAL = "search_local"
    SEARCH_WEB = "search_web"
    DOCUMENT_OPS = "document_ops"
    EXPANSION = "expansion"
    CACHE = "cache"
    SEGMENT_SEARCH = "segment_search"
    INFORMATION_ANALYSIS = "information_analysis"
    VISUALIZATION = "visualization"
    IMAGE_SEARCH = "image_search"  # Image/comic/visual content search
    DATA_WORKSPACE = "data_workspace"  # Data workspace queries


# Tool name to category mapping
TOOL_CATEGORY_MAP = {
    "search_documents_tool": OrchestratorToolCategory.SEARCH_LOCAL,
    "search_documents_structured": OrchestratorToolCategory.SEARCH_LOCAL,
    "get_document_content_tool": OrchestratorToolCategory.DOCUMENT_OPS,
    "search_within_document_tool": OrchestratorToolCategory.DOCUMENT_OPS,
    "search_web_tool": OrchestratorToolCategory.SEARCH_WEB,
    "search_web_structured": OrchestratorToolCategory.SEARCH_WEB,
    "crawl_web_content_tool": OrchestratorToolCategory.SEARCH_WEB,
    "search_and_crawl_tool": OrchestratorToolCategory.SEARCH_WEB,
    "expand_query_tool": OrchestratorToolCategory.EXPANSION,
    "search_conversation_cache_tool": OrchestratorToolCategory.CACHE,
    "search_segments_across_documents_tool": OrchestratorToolCategory.SEGMENT_SEARCH,
    "extract_relevant_content_section": OrchestratorToolCategory.SEGMENT_SEARCH,
    "analyze_information_needs_tool": OrchestratorToolCategory.INFORMATION_ANALYSIS,
    "generate_project_aware_queries_tool": OrchestratorToolCategory.INFORMATION_ANALYSIS,
    "create_chart_tool": OrchestratorToolCategory.VISUALIZATION,
    "search_images_tool": OrchestratorToolCategory.IMAGE_SEARCH,  # Image/comic search tool
    "list_data_workspaces_tool": OrchestratorToolCategory.DATA_WORKSPACE,
    "get_workspace_schema_tool": OrchestratorToolCategory.DATA_WORKSPACE,
    "query_data_workspace_tool": OrchestratorToolCategory.DATA_WORKSPACE,
}


# Keyword mappings for category detection
CATEGORY_KEYWORDS = {
    OrchestratorToolCategory.SEARCH_WEB: [
        "search web", "look up", "find online", "web search", "internet",
        "online", "browse", "crawl", "current", "latest", "recent", "now"
    ],
    OrchestratorToolCategory.SEGMENT_SEARCH: [
        "section", "segment", "part of", "within document", "specific part"
    ],
    OrchestratorToolCategory.INFORMATION_ANALYSIS: [
        "analyze", "what information", "what do I need", "gap analysis"
    ],
    OrchestratorToolCategory.VISUALIZATION: [
        "chart", "graph", "plot", "visualize", "show me", "compare", "trend",
        "make a chart", "graph this", "plot the data", "visualization", "diagram"
    ],
    OrchestratorToolCategory.IMAGE_SEARCH: [
        "show me", "display", "find image", "show image", "comic", "comics",
        "picture", "pictures", "photo", "photos", "artwork", "meme", "memes",
        "screenshot", "diagram", "visual", "image", "images", "show", "see"
    ],
    OrchestratorToolCategory.DATA_WORKSPACE: [
        "workspace", "data workspace", "vendor", "vendors", "transaction", "transactions",
        "table", "tables", "query data", "database query", "sql", "query workspace",
        "spending", "purchase", "purchases", "cost", "costs", "price", "prices",
        "data", "workspace data", "my data", "my vendors", "my transactions"
    ],
}


def detect_tool_categories_from_query(query: str) -> List[str]:
    """
    Detect tool categories needed based on query keywords
    
    Args:
        query: User query string
        
    Returns:
        List of detected category names
    """
    query_lower = query.lower()
    detected = []
    
    # Always include core categories for research agent
    detected.append(OrchestratorToolCategory.SEARCH_LOCAL)
    detected.append(OrchestratorToolCategory.DOCUMENT_OPS)
    detected.append(OrchestratorToolCategory.EXPANSION)
    detected.append(OrchestratorToolCategory.CACHE)
    
    # Detect conditional categories
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            if category not in detected:
                detected.append(category)
    
    return detected


def get_tools_for_categories(categories: List[str]) -> List[str]:
    """
    Get tool names that match the given categories
    
    Args:
        categories: List of category names
        
    Returns:
        List of tool names in those categories
    """
    tools = []
    for tool_name, tool_category in TOOL_CATEGORY_MAP.items():
        if tool_category in categories:
            tools.append(tool_name)
    
    return tools


async def analyze_tool_needs_for_research(query: str, conversation_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Analyze query to determine which tools are needed for research agent
    
    Hybrid approach: Combines keyword-based detection with semantic discovery
    for next-generation tool selection.
    
    Args:
        query: User query string
        conversation_context: Optional conversation context
        
    Returns:
        {
            "core_tools": [...],  # Always needed
            "conditional_tools": [...],  # Triggered by query (keyword + semantic)
            "all_tools": [...],  # Combined list
            "categories": [...],  # Detected categories
            "semantic_discoveries": [...],  # Tools found via semantic search
            "keyword_matches": [...],  # Tools found via keyword matching
            "reasoning": "..."
        }
    """
    # Core tools always needed for research
    core_tools = [
        "search_documents_tool",
        "get_document_content_tool",
        "expand_query_tool",
        "search_conversation_cache_tool"
    ]
    
    # Keyword-based detection (existing logic)
    categories = detect_tool_categories_from_query(query)
    keyword_tools = []
    for category in categories:
        if category not in [OrchestratorToolCategory.SEARCH_LOCAL, 
                           OrchestratorToolCategory.DOCUMENT_OPS,
                           OrchestratorToolCategory.EXPANSION,
                           OrchestratorToolCategory.CACHE]:
            category_tools = get_tools_for_categories([category])
            keyword_tools.extend(category_tools)
    keyword_tools = list(set(keyword_tools))
    
    # Semantic discovery (new hybrid approach)
    semantic_tools = []
    semantic_discoveries = []
    try:
        from orchestrator.utils.tool_discovery import get_tool_discovery_service
        discovery_service = await get_tool_discovery_service()
        semantic_results = await discovery_service.discover_tools(
            query=query,
            top_k=5,
            min_confidence=0.6
        )
        
        # Extract tool names from semantic results
        for result in semantic_results:
            tool_name = result.get("tool_name")
            if tool_name and tool_name not in core_tools:
                semantic_tools.append(tool_name)
                semantic_discoveries.append({
                    "tool": tool_name,
                    "score": result.get("score", 0.0),
                    "pack": result.get("pack", "unknown")
                })
        
        semantic_tools = list(set(semantic_tools))
        logger.info(
            f"Semantic discovery: {len(semantic_tools)} tools found "
            f"(keyword: {len(keyword_tools)})"
        )
        
    except Exception as e:
        logger.warning(f"Semantic discovery failed, using keyword-only: {e}")
        # Fallback to keyword-only on failure
    
    # Merge keyword and semantic results (union, no duplicates)
    conditional_tools = list(set(keyword_tools + semantic_tools))
    
    # Build reasoning
    reasoning_parts = []
    if conditional_tools:
        keyword_only = [t for t in conditional_tools if t in keyword_tools and t not in semantic_tools]
        semantic_only = [t for t in conditional_tools if t in semantic_tools and t not in keyword_tools]
        both = [t for t in conditional_tools if t in keyword_tools and t in semantic_tools]
        
        if both:
            reasoning_parts.append(f"Both methods agree on: {', '.join(both)}")
        if keyword_only:
            reasoning_parts.append(f"Keyword detection found: {', '.join(keyword_only)}")
        if semantic_only:
            reasoning_parts.append(f"Semantic discovery found: {', '.join(semantic_only)}")
    else:
        reasoning_parts.append("Query requires only core research tools")
    
    # Check conversation context for hints
    if conversation_context:
        previous_tools = conversation_context.get("previous_tools_used", [])
        if previous_tools:
            if any("web" in tool.lower() for tool in previous_tools):
                if "search_and_crawl_tool" not in conditional_tools:
                    conditional_tools.append("search_and_crawl_tool")
                    reasoning_parts.append("Web tools used previously, likely needed again")
    
    # Combine all tools
    all_tools = list(set(core_tools + conditional_tools))
    
    return {
        "core_tools": core_tools,
        "conditional_tools": conditional_tools,
        "all_tools": all_tools,
        "categories": categories,
        "semantic_discoveries": semantic_discoveries,
        "keyword_matches": keyword_tools,
        "reasoning": "; ".join(reasoning_parts),
        "tool_count": len(all_tools),
        "core_count": len(core_tools),
        "conditional_count": len(conditional_tools)
    }


