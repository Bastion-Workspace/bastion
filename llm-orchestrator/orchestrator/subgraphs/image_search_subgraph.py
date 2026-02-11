"""
Image Search Subgraph

Reusable subgraph for intelligent image search with metadata sidecars.
Handles query analysis, parameter extraction, and image search execution.

Can be used by:
- Chat Agent (via intelligent_document_retrieval_subgraph)
- Research Agent (via research_workflow_subgraph)
- Any agent that needs to search for images

Inputs:
- query: Natural language query (e.g., "Can you show me Dilbert from April 16th, 1989?")
- user_id: User ID for access control
- metadata: Optional metadata for model selection
- limit: Maximum number of results (default: 10)

Outputs:
- image_search_results: Markdown-formatted results with image URLs (or None if no results)
- search_performed: Whether image search was attempted
- analysis: LLM analysis of the query (series, author, date, etc.)
"""

import logging
from typing import Any, Dict, Optional, TypedDict

from langgraph.graph import StateGraph, END

from orchestrator.tools.image_query_analyzer import analyze_image_query
from orchestrator.tools.image_search_tools import search_images_tool

logger = logging.getLogger(__name__)


# Use dict for compatibility with LangGraph StateGraph
# LangGraph cannot instantiate Dict[str, Any], must use lowercase dict
ImageSearchSubgraphState = dict


async def detect_image_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Detect if query is image-related and if it's a random request"""
    try:
        query = state.get("query", "")
        query_lower = query.lower()
        
        # Image-related keywords
        image_keywords = [
            "show me", "comic", "comics", "image", "images", "picture", "pictures",
            "photo", "photos", "artwork", "meme", "screenshot", "diagram", "visual", "display",
            "map", "maps", "chart", "charts", "graph", "graphs"
        ]
        
        is_image_query = any(keyword in query_lower for keyword in image_keywords)
        
        # Check conversation history for image-related context
        if not is_image_query:
            messages = state.get("messages", [])
            # Look at recent conversation history (last 5 messages)
            recent_messages = messages[-5:] if len(messages) > 5 else messages
            
            for msg in recent_messages:
                # Check if message content contains image-related keywords
                msg_content = ""
                if hasattr(msg, "content"):
                    msg_content = str(msg.content).lower()
                elif isinstance(msg, dict):
                    msg_content = str(msg.get("content", "")).lower()
                
                # If recent conversation mentioned images/comics, this might be a follow-up
                if any(keyword in msg_content for keyword in image_keywords):
                    is_image_query = True
                    logger.info(f"💬 Follow-up image query detected from conversation context: {query[:100]}")
                    break
        
        # Detect random requests
        is_random = "random" in query_lower
        
        if is_random:
            logger.info(f"🎲 Random image query detected: {query[:100]}")
        else:
            logger.info(f"Image query detection: {is_image_query} for query: {query[:100]}")
        
        return {
            "is_image_query": is_image_query,
            "is_random": is_random,
            # Preserve critical state
            "query": state.get("query", ""),
            "user_id": state.get("user_id", "system"),
            "metadata": state.get("metadata", {}),
            "limit": state.get("limit", 10),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", [])
        }
        
    except Exception as e:
        logger.error(f"Image query detection failed: {e}")
        return {
            "is_image_query": False,
            "is_random": False,
            "error": str(e),
            # Preserve critical state
            "query": state.get("query", ""),
            "user_id": state.get("user_id", "system"),
            "metadata": state.get("metadata", {}),
            "limit": state.get("limit", 10),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", [])
        }


async def analyze_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Use LLM to intelligently parse the image query"""
    try:
        query = state.get("query", "")
        user_id = state.get("user_id", "system")
        metadata = state.get("metadata", {})
        shared_memory = state.get("shared_memory", {})

        # Use cached analysis if available (from round1 parallel search)
        cached_analysis = shared_memory.get("image_analysis_cache")
        if cached_analysis:
            logger.info("Using cached image analysis")
            return {
                "analysis": cached_analysis,
                "query": state.get("query", ""),
                "user_id": user_id,
                "metadata": metadata,
                "limit": state.get("limit", 10),
                "is_image_query": state.get("is_image_query", False),
                "is_random": state.get("is_random", False),
                "shared_memory": shared_memory,
                "messages": state.get("messages", []),
            }

        # Skip LLM analysis for simple existence queries
        query_lower = query.lower()
        simple_indicators = ["do we have", "show me", "display", "any photos"]
        is_simple = any(ind in query_lower for ind in simple_indicators)

        if is_simple:
            logger.info(f"Skipping LLM analysis for simple image query: {query}")
            return {
                "analysis": {
                    "query": query,
                    "series": None,
                    "author": None,
                    "date": None,
                    "image_type": None,
                },
                "query": query,
                "metadata": metadata,
                "user_id": user_id,
                "limit": state.get("limit", 10),
                "shared_memory": shared_memory,
                "messages": state.get("messages", []),
                "is_image_query": state.get("is_image_query", False),
                "is_random": state.get("is_random", False),
            }

        logger.info(f"Analyzing image query with LLM: {query[:100]}")

        # Use LLM to parse query
        analysis = await analyze_image_query(
            query=query,
            user_id=user_id,
            metadata=metadata,
            shared_memory=shared_memory  # Pass shared_memory for user_chat_model lookup
        )
        
        logger.info(f"LLM analysis: series={analysis.get('series')}, author={analysis.get('author')}, date={analysis.get('date')}, type={analysis.get('image_type')}")
        
        return {
            "analysis": analysis,
            # Preserve critical state
            "query": state.get("query", ""),
            "user_id": state.get("user_id", "system"),
            "metadata": state.get("metadata", {}),
            "limit": state.get("limit", 10),
            "is_image_query": state.get("is_image_query", False),
            "is_random": state.get("is_random", False),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", [])
        }
        
    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "analysis": {
                "query": query,
                "series": None,
                "author": None,
                "date": None,
                "image_type": None
            },
            "error": str(e),
            # Preserve critical state
            "query": state.get("query", ""),
            "user_id": state.get("user_id", "system"),
            "metadata": state.get("metadata", {}),
            "limit": state.get("limit", 10),
            "is_image_query": state.get("is_image_query", False),
            "is_random": state.get("is_random", False),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", [])
        }


async def search_images_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute image search with extracted parameters"""
    try:
        query = state.get("query", "")
        user_id = state.get("user_id", "system")
        limit = state.get("limit", 10)
        analysis = state.get("analysis", {})
        is_random = state.get("is_random", False)
        
        logger.info(f"Executing image search with parameters: series={analysis.get('series')}, author={analysis.get('author')}, date={analysis.get('date')}, is_random={is_random}")
        
        # Call image search tool with LLM-extracted parameters
        image_search_results = await search_images_tool(
            query=analysis.get("query", query),
            image_type=analysis.get("image_type"),
            date=analysis.get("date"),
            author=analysis.get("author"),
            series=analysis.get("series"),
            limit=limit,
            user_id=user_id,
            is_random=is_random  # Pass random flag to search tool
        )

        # If no results and we filtered by type, retry without type filter so we don't overlook
        # images that weren't tagged (e.g. user forgot to set type: photo).
        image_type_used = analysis.get("image_type")
        if image_type_used and isinstance(image_search_results, dict):
            images_markdown = image_search_results.get("images_markdown", "")
            metadata = image_search_results.get("metadata", [])
            no_results = (
                not images_markdown or "No images found" in (images_markdown or "") or len(metadata or []) == 0
            )
            if no_results:
                try:
                    logger.info(f"Image search: no results for type '{image_type_used}', retrying without type filter")
                    fallback = await search_images_tool(
                        query=analysis.get("query", query),
                        image_type=None,
                        date=analysis.get("date"),
                        author=analysis.get("author"),
                        series=analysis.get("series"),
                        limit=limit,
                        user_id=user_id,
                        is_random=is_random,
                    )
                    fm = fallback.get("images_markdown", "") if isinstance(fallback, dict) else ""
                    flist = fallback.get("metadata", []) if isinstance(fallback, dict) else []
                    fstruct = fallback.get("images", []) if isinstance(fallback, dict) else []
                    if (fm and "No images found" not in fm) or fstruct:
                        image_search_results = fallback
                except Exception as e:
                    logger.warning(f"Fallback image search without type filter failed: {e}")
        
        # Check if results were found (handle both dict and legacy string formats)
        has_results = False
        if isinstance(image_search_results, dict):
            images_markdown = image_search_results.get("images_markdown", "")
            metadata = image_search_results.get("metadata", [])
            has_results = (
                images_markdown and
                "No images found" not in images_markdown and
                "Error" not in images_markdown and
                len(metadata) > 0
            )
            if has_results:
                logger.info(f"Image search found results: {len(images_markdown)} characters, {len(metadata)} metadata entries")
        else:
            # Legacy string format
            has_results = (
                image_search_results and
                "No images found" not in image_search_results and
                "Error" not in image_search_results
            )
            if has_results:
                logger.info(f"Image search found results: {len(image_search_results)} characters")
        
        if not has_results:
            logger.info(f"Image search returned no results")
        
        return {
            "image_search_results": image_search_results if has_results else None,
            "search_performed": True,
            # Preserve critical state
            "query": state.get("query", ""),
            "user_id": state.get("user_id", "system"),
            "metadata": state.get("metadata", {}),
            "limit": state.get("limit", 10),
            "is_image_query": state.get("is_image_query", False),
            "is_random": state.get("is_random", False),
            "analysis": state.get("analysis", {}),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", [])
        }
        
    except Exception as e:
        logger.error(f"Image search execution failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "image_search_results": None,
            "search_performed": True,
            "error": str(e),
            # Preserve critical state
            "query": state.get("query", ""),
            "user_id": state.get("user_id", "system"),
            "metadata": state.get("metadata", {}),
            "limit": state.get("limit", 10),
            "is_image_query": state.get("is_image_query", False),
            "is_random": state.get("is_random", False),
            "analysis": state.get("analysis", {}),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", [])
        }


async def skip_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Skip image search for non-image queries"""
    logger.info("Query is not image-related, skipping image search")
    return {
        "image_search_results": None,
        "search_performed": False,
        # Preserve critical state
        "query": state.get("query", ""),
        "user_id": state.get("user_id", "system"),
        "metadata": state.get("metadata", {}),
        "limit": state.get("limit", 10),
        "is_image_query": state.get("is_image_query", False),
        "is_random": state.get("is_random", False),
        "shared_memory": state.get("shared_memory", {}),
        "messages": state.get("messages", [])
    }


def build_image_search_subgraph(checkpointer=None) -> StateGraph:
    """
    Build image search subgraph
    
    Flow:
    1. Detect if query is image-related
    2. If yes: Analyze query with LLM → Execute search
    3. If no: Skip search
    
    Args:
        checkpointer: Optional LangGraph checkpointer for state persistence
    
    Returns:
        Compiled StateGraph with checkpointer (if provided)
    """
    subgraph = StateGraph(ImageSearchSubgraphState)
    
    # Add nodes
    subgraph.add_node("detect_image_query", detect_image_query_node)
    subgraph.add_node("analyze_query", analyze_query_node)
    subgraph.add_node("search_images", search_images_node)
    subgraph.add_node("skip_search", skip_search_node)
    
    # Entry point
    subgraph.set_entry_point("detect_image_query")
    
    # Conditional routing: image query or not?
    subgraph.add_conditional_edges(
        "detect_image_query",
        lambda state: "analyze_query" if state.get("is_image_query", False) else "skip_search",
        {
            "analyze_query": "analyze_query",
            "skip_search": "skip_search"
        }
    )
    
    # After analysis, execute search
    subgraph.add_edge("analyze_query", "search_images")
    
    # Both paths end
    subgraph.add_edge("search_images", END)
    subgraph.add_edge("skip_search", END)
    
    if checkpointer:
        return subgraph.compile(checkpointer=checkpointer)
    else:
        return subgraph.compile()


async def search_images_intelligently(
    query: str,
    user_id: str = "system",
    metadata: Optional[Dict[str, Any]] = None,
    limit: int = 10,
    messages: Optional[list] = None,
    shared_memory: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run image search subgraph
    
    Args:
        query: Natural language query
        user_id: User ID for access control
        metadata: Optional metadata for model selection
        limit: Maximum number of results
        messages: Conversation messages for context-aware detection
        shared_memory: Shared memory for conversation context
        
    Returns:
        Dict with:
        - image_search_results: Markdown-formatted results or None
        - search_performed: Whether search was attempted
        - analysis: LLM analysis of query
    """
    subgraph = build_image_search_subgraph()
    
    initial_state = {
        "query": query,
        "user_id": user_id,
        "metadata": metadata or {},
        "limit": limit,
        "messages": messages or [],
        "shared_memory": shared_memory or {}
    }
    
    result = await subgraph.ainvoke(initial_state)
    
    return {
        "image_search_results": result.get("image_search_results"),
        "search_performed": result.get("search_performed", False),
        "analysis": result.get("analysis", {})
    }
