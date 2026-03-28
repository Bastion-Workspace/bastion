"""
Image Search Tools - LangGraph tools for searching images with metadata sidecars
"""

import logging
import sys
import os
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field

from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


# ── I/O models for search_images_tool ───────────────────────────────────────

class SearchImagesInputs(BaseModel):
    """Required inputs for search_images_tool."""
    query: str = Field(description="Search query describing the image content")


class SearchImagesParams(BaseModel):
    """Optional parameters."""
    image_type: Optional[str] = Field(default=None, description="Filter by type: comic, artwork, meme, screenshot, etc.")
    date: Optional[str] = Field(default=None, description="Date filter YYYY-MM-DD")
    author: Optional[str] = Field(default=None, description="Author/creator filter")
    series: Optional[str] = Field(default=None, description="Series name filter")
    limit: int = Field(default=10, description="Max results to return")
    is_random: bool = Field(default=False, description="Return random images instead of semantic search")


class SearchImagesOutputs(BaseModel):
    """Typed outputs for search_images_tool."""
    images_markdown: str = Field(description="Markdown with image tags for display")
    metadata: List[Dict[str, Any]] = Field(default_factory=list, description="Metadata for each image")
    images: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Structured list for AgentResponse: url, alt_text, type, metadata")
    count: int = Field(description="Number of images returned")
    query_used: str = Field(description="Query that was executed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")

# Add backend to Python path for direct imports
# This allows us to call backend tools directly without gRPC overhead
_backend_path_added = False
def _ensure_backend_path():
    global _backend_path_added
    if not _backend_path_added:
        # Try multiple possible paths
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..', 'backend'),
            '/app/backend',
            os.path.join(os.getcwd(), 'backend'),
        ]
        for backend_path in possible_paths:
            if os.path.exists(backend_path) and backend_path not in sys.path:
                sys.path.insert(0, backend_path)
                logger.info(f"Added backend path to sys.path: {backend_path}")
                _backend_path_added = True
                break


async def search_images_tool(
    query: str,
    image_type: Optional[str] = None,
    date: Optional[str] = None,
    author: Optional[str] = None,
    series: Optional[str] = None,
    limit: int = 10,
    user_id: str = "system",
    is_random: bool = False,
    exclude_document_ids: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Search for images, comics, artwork, photos, faces, portraits, memes, screenshots, and other visual content.
    Use this — NOT search_documents — when users ask to 'show', 'display', 'find', or 'see' any visual content
    including comics, pictures, photos, artwork, faces, portraits, selfies, diagrams, or maps.
    Returns actual images that can be displayed in chat.
    
    Args:
        query: Search query describing the image content (e.g., 'Dilbert from 1989', 'office politics comic', 'family photos')
        image_type: Optional filter by image type (comic, artwork, meme, screenshot, medical, documentation, maps, photo, other)
        date: Optional date filter (YYYY-MM-DD format, e.g., '1989-04-16')
        author: Optional author/creator name filter (e.g., 'Scott Adams' for Dilbert)
        series: Optional series name filter (e.g., 'Dilbert')
        limit: Maximum number of results to return (default: 10)
        user_id: User ID for access control
        is_random: If True, return random images instead of semantic search (default: False)
        
    Returns:
        Dict with 'images_markdown' (base64 embedded images) and 'metadata' (list of metadata dicts)
    """
    try:
        if is_random:
            logger.info(f"🎲 Random image request: type={image_type}, author={author}, series={series}, limit={limit}")
        else:
            logger.info(f"Searching images: query='{query[:100]}', type={image_type}, date={date}, author={author}, series={series}")
        
        # Use backend tool client via gRPC
        from orchestrator.backend_tool_client import get_backend_tool_client
        
        client = await get_backend_tool_client()
        result = await client.search_images(
            query=query,
            image_type=image_type,
            date=date,
            author=author,
            series=series,
            limit=limit,
            user_id=user_id,
            is_random=is_random,
            exclude_document_ids=exclude_document_ids,
        )
        
        if isinstance(result, dict):
            images_markdown = result.get("images_markdown", "")
            metadata_count = len(result.get("metadata", []))
            logger.info(f"Image search completed: {len(images_markdown)} characters, {metadata_count} metadata entries")
        else:
            logger.info(f"Image search completed: {len(result)} characters returned")
            result = {
                "images_markdown": result if isinstance(result, str) else str(result),
                "metadata": []
            }
        
        meta = result.get("metadata", [])
        count = len(meta)
        formatted_parts = [f"Found {count} image(s) for query."]
        for i, m in enumerate(meta[:10], 1):
            title = m.get("title") or m.get("filename") or "Untitled"
            date_str = m.get("date") or m.get("publication_date") or ""
            series_str = m.get("series") or ""
            author_str = m.get("author") or m.get("creator") or ""
            line = f"{i}. **{title}**"
            if date_str:
                line += f" (date: {date_str})"
            if series_str:
                line += f" — series: {series_str}"
            if author_str:
                line += f" — author: {author_str}"
            formatted_parts.append(line)
        if count > 10:
            formatted_parts.append(f"... and {count - 10} more.")
        markdown_preview = result.get("images_markdown", "")
        if markdown_preview:
            if len(markdown_preview) > 500:
                formatted_parts.append("\nPreview (first 500 chars):\n" + markdown_preview[:500] + "...")
            else:
                formatted_parts.append("\n" + markdown_preview)
        formatted = "\n".join(formatted_parts)
        return {**result, "count": count, "query_used": query, "formatted": formatted}

    except Exception as e:
        logger.error("Image search tool error: %s", e)
        import traceback
        logger.error("Traceback: %s", traceback.format_exc())
        return {
            "images_markdown": f"Error searching images: {str(e)}",
            "metadata": [],
            "count": 0,
            "query_used": query,
            "formatted": f"Error searching images: {str(e)}",
        }


register_action(
    name="search_images",
    category="search",
    description="Search for images, comics, photos, artwork, faces, portraits, and all visual content with metadata. Use instead of search_documents for any visual/image query.",
    inputs_model=SearchImagesInputs,
    params_model=SearchImagesParams,
    outputs_model=SearchImagesOutputs,
    tool_function=search_images_tool,
)


# Export tools list for dynamic tool loading
IMAGE_SEARCH_TOOLS = [
    search_images_tool
]
