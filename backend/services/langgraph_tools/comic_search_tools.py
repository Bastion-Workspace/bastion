"""
Comic Search Tools Module
Search functionality for comic strips with date and character filtering
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from services.embedding_service_wrapper import get_embedding_service

logger = logging.getLogger(__name__)


class ComicSearchTools:
    """Search tools for comic strips"""
    
    def __init__(self):
        self._embedding_manager = None
    
    async def _get_embedding_manager(self):
        """Get embedding service wrapper with lazy initialization"""
        if self._embedding_manager is None:
            try:
                self._embedding_manager = await get_embedding_service()
                logger.info("Embedding service wrapper initialized for comic search")
            except Exception as e:
                logger.error(f"Failed to initialize embedding service wrapper: {e}")
        return self._embedding_manager
    
    def get_tools(self) -> Dict[str, Any]:
        """Get all comic search tools"""
        return {
            "search_comics": self.search_comics,
        }
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all comic search tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_comics",
                    "description": "Search for comic strips by content, date, or character. Returns markdown-formatted results with image URLs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query (e.g., 'fungus', 'office politics', 'Dilbert')"},
                            "date": {"type": "string", "description": "OPTIONAL: Filter by specific date (YYYY-MM-DD format, e.g., '2001-09-11')"},
                            "character": {"type": "string", "description": "OPTIONAL: Filter by character name (e.g., 'Dilbert', 'Dogbert')"},
                            "limit": {"type": "integer", "description": "Maximum number of results to return", "default": 10}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    async def search_comics(
        self,
        query: str,
        date: Optional[str] = None,
        character: Optional[str] = None,
        limit: int = 10
    ) -> str:
        """
        Search for comic strips by content, date, or character
        
        Args:
            query: Search query
            date: Optional date filter (YYYY-MM-DD)
            character: Optional character filter
            limit: Maximum results
            
        Returns:
            Markdown-formatted string with comic results
        """
        try:
            logger.info(f"Searching comics: query='{query}', date={date}, character={character}")
            
            embedding_manager = await self._get_embedding_manager()
            if not embedding_manager:
                return "❌ Comic search unavailable - embedding service not initialized"
            
            # Generate query embedding
            query_embeddings = await embedding_manager.generate_embeddings([query])
            if not query_embeddings or len(query_embeddings) == 0:
                return "❌ Failed to generate query embedding"
            
            # Build filters
            filter_category = "comic"
            filter_tags = []
            if character:
                filter_tags.append(character.lower())
            
            # Search with category filter
            results = await embedding_manager.search_similar(
                query_embedding=query_embeddings[0],
                limit=limit * 2,  # Get more results for date filtering
                score_threshold=0.3,
                user_id=None,  # Global comics accessible to all
                filter_category=filter_category,
                filter_tags=filter_tags if filter_tags else None
            )
            
            if not results:
                return f"🔍 No comics found matching '{query}'"
            
            # Filter by date if provided
            if date:
                try:
                    target_date = datetime.strptime(date, "%Y-%m-%d").date()
                    filtered_results = []
                    for result in results:
                        metadata = result.get("metadata", {})
                        result_date_str = metadata.get("date")
                        if result_date_str:
                            try:
                                result_date = datetime.strptime(result_date_str, "%Y-%m-%d").date()
                                if result_date == target_date:
                                    filtered_results.append(result)
                            except ValueError:
                                continue
                    results = filtered_results[:limit]
                except ValueError:
                    logger.warning(f"Invalid date format: {date}")
            
            # Limit results
            results = results[:limit]
            
            if not results:
                return f"🔍 No comics found matching '{query}'" + (f" on {date}" if date else "")
            
            # Format results as markdown
            formatted_results = [f"🔍 **Found {len(results)} comic(s) matching '{query}':**\n"]
            
            for i, result in enumerate(results, 1):
                metadata = result.get("metadata", {})
                title = metadata.get("title", "Untitled Comic")
                date_str = metadata.get("date", "")
                image_url = metadata.get("image_url", "")
                series = metadata.get("series", "")
                transcript = metadata.get("transcript", "")
                
                # Build comic header
                comic_title = f"{series} - {title}" if series else title
                if date_str:
                    comic_title += f" ({date_str})"
                
                formatted_results.append(f"### {i}. {comic_title}\n")
                
                # Add image
                if image_url:
                    formatted_results.append(f"![Comic]({image_url})\n")
                
                # Add transcript snippet
                if transcript:
                    transcript_snippet = transcript[:200] + "..." if len(transcript) > 200 else transcript
                    formatted_results.append(f"*{transcript_snippet}*\n")
                
                formatted_results.append("\n")
            
            return "".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Comic search failed: {e}")
            return f"❌ Comic search failed: {str(e)}"


# Global instance for use by tool registry
_comic_search_instance = None


async def _get_comic_search():
    """Get global comic search instance"""
    global _comic_search_instance
    if _comic_search_instance is None:
        _comic_search_instance = ComicSearchTools()
    return _comic_search_instance


async def search_comics(
    query: str,
    date: Optional[str] = None,
    character: Optional[str] = None,
    limit: int = 10
) -> str:
    """
    LangGraph tool function: Search for comic strips
    
    Returns markdown-formatted results with image URLs.
    """
    search_instance = await _get_comic_search()
    return await search_instance.search_comics(query, date, character, limit)
