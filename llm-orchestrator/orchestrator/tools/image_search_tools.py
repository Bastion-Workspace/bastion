"""
Image Search Tools - LangGraph tools for searching images with metadata sidecars
"""

import logging
import sys
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

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
    Search for images, comics, artwork, memes, screenshots, and other visual content with metadata.
    Use this when users ask to 'show', 'display', 'find', or 'see' images, comics, pictures, or visual content.
    
    Args:
        query: Search query describing the image content (e.g., 'Dilbert from 1989', 'office politics comic')
        image_type: Optional filter by image type (comic, artwork, meme, screenshot, medical, documentation, maps, other)
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
            # Legacy format (backward compatibility)
            logger.info(f"Image search completed: {len(result)} characters returned")
            result = {
                "images_markdown": result if isinstance(result, str) else str(result),
                "metadata": []
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Image search tool error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "images_markdown": f"Error searching images: {str(e)}",
            "metadata": []
        }


# Export tools list for dynamic tool loading
IMAGE_SEARCH_TOOLS = [
    search_images_tool
]
