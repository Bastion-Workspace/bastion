"""
Calibre Search Service - Business logic for Calibre library integration
Provides unified search across documents and Calibre library
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from repositories.calibre_repository import CalibreRepository, CalibreBook
from config import settings

logger = logging.getLogger(__name__)


class CalibreSearchService:
    """Service for Calibre library search and integration"""
    
    def __init__(self):
        self.calibre_repo = CalibreRepository()
        self._initialized = False
        self.settings_service = None
    
    async def initialize(self):
        """Initialize the Calibre search service"""
        try:
            logger.info("📚 Initializing Calibre Search Service...")
            
            # Load settings from database and sync with config
            await self._load_settings_from_database()
            
            # Only initialize repository if Calibre is enabled
            if settings.CALIBRE_ENABLED:
                await self.calibre_repo.initialize()
                self._initialized = True
                logger.info("✅ Calibre Search Service initialized and enabled")
            else:
                self._initialized = False
                logger.info("ℹ️ Calibre Search Service initialized but disabled")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Calibre Search Service: {e}")
            self._initialized = False
    
    async def _load_settings_from_database(self):
        """Load Calibre settings from database and sync with config"""
        try:
            # Import settings service here to avoid circular imports
            from services.settings_service import settings_service
            
            # Load enabled status
            calibre_enabled = await settings_service.get_setting("calibre_enabled", settings.CALIBRE_ENABLED)
            settings.CALIBRE_ENABLED = calibre_enabled
            
            # Load library path
            library_path = await settings_service.get_setting("calibre_library_path", settings.CALIBRE_LIBRARY_PATH)
            if library_path:
                settings.CALIBRE_LIBRARY_PATH = library_path
            
            # Load search weight
            search_weight = await settings_service.get_setting("calibre_search_weight", settings.CALIBRE_SEARCH_WEIGHT)
            settings.CALIBRE_SEARCH_WEIGHT = float(search_weight)
            
            # Load max results
            max_results = await settings_service.get_setting("calibre_max_results", settings.CALIBRE_MAX_RESULTS)
            settings.CALIBRE_MAX_RESULTS = int(max_results)
            
            logger.info(f"📚 Loaded Calibre settings: enabled={calibre_enabled}, path={library_path}, weight={search_weight}, max_results={max_results}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Calibre settings from database: {e}")
            # Continue with default settings from config
    
    def is_available(self) -> bool:
        """Check if Calibre search is available"""
        return self._initialized and self.calibre_repo.is_available()
    
    async def search_calibre_only(
        self, 
        query: str, 
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """Search only Calibre library and return formatted results"""
        if not self.is_available():
            logger.warning("📚 Calibre search not available")
            return []
        
        try:
            # Use configured limit if not specified
            search_limit = limit or settings.CALIBRE_MAX_RESULTS
            
            # Search Calibre library
            books = await self.calibre_repo.search_books(query, search_limit)
            
            # Convert to standard search result format
            results = []
            for book in books:
                citation = book.to_citation_dict()
                
                # Enhance with search-specific formatting
                result = {
                    "document_id": citation["id"],
                    "title": citation["title"],
                    "authors": citation["authors"],
                    "content": citation["snippet"] or f"Book by {citation['authors']}",
                    "metadata": citation["metadata"],
                    "source": "calibre",
                    "type": "book",
                    "score": self._calculate_relevance_score(book, query),
                    "timestamp": citation["metadata"]["timestamp"],
                    "formats": citation["metadata"]["formats"],
                    "series_info": self._format_series_info(book)
                }
                results.append(result)
            
            logger.info(f"📚 Calibre search returned {len(results)} results for: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"❌ Calibre search failed: {e}")
            return []
    
    async def hybrid_search(
        self, 
        query: str, 
        document_results: List[Dict[str, Any]], 
        include_calibre: bool = True,
        calibre_limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Combine document search results with Calibre results
        Returns unified, weighted, and sorted results
        """
        if not include_calibre or not self.is_available():
            return document_results
        
        try:
            # Get Calibre results
            calibre_results = await self.search_calibre_only(query, calibre_limit)
            
            if not calibre_results:
                return document_results
            
            # Apply Calibre weight to scores
            calibre_weight = settings.CALIBRE_SEARCH_WEIGHT
            for result in calibre_results:
                result["score"] *= calibre_weight
                result["weighted"] = True
            
            # Combine and sort by score
            combined_results = document_results + calibre_results
            combined_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            logger.info(f"📚 Hybrid search: {len(document_results)} docs + {len(calibre_results)} books = {len(combined_results)} total")
            
            return combined_results
            
        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            return document_results
    
    def _calculate_relevance_score(self, book: CalibreBook, query: str) -> float:
        """Calculate relevance score for a book based on query"""
        score = 0.0
        query_lower = query.lower()
        
        # Title match (highest weight)
        if query_lower in book.title.lower():
            score += 0.8
            if book.title.lower().startswith(query_lower):
                score += 0.2
        
        # Author match
        if query_lower in book.authors.lower():
            score += 0.6
        
        # Series match
        if book.series and query_lower in book.series.lower():
            score += 0.5
        
        # Tags match
        if book.tags:
            for tag in book.tags:
                if query_lower in tag.lower():
                    score += 0.3
                    break
        
        # Comments/description match
        if book.comments and query_lower in book.comments.lower():
            score += 0.4
        
        # Publisher match
        if book.publisher and query_lower in book.publisher.lower():
            score += 0.2
        
        # Boost for recently added books
        if book.timestamp:
            days_old = (datetime.now() - book.timestamp).days
            if days_old < 30:
                score += 0.1
            elif days_old < 90:
                score += 0.05
        
        # Boost for rated books
        if book.rating and book.rating >= 4:
            score += 0.1
        
        # Normalize score to 0-1 range
        return min(score, 1.0)
    
    def _format_series_info(self, book: CalibreBook) -> Optional[str]:
        """Format series information for display"""
        if not book.series:
            return None
        
        if book.series_index:
            return f"{book.series} #{book.series_index}"
        else:
            return book.series
    
    async def get_book_filters(self) -> Dict[str, List[str]]:
        """Get available filter options from Calibre library"""
        if not self.is_available():
            return {}
        
        try:
            # This would require additional repository methods
            # For now, return basic structure
            return {
                "authors": [],
                "series": [],
                "tags": [],
                "publishers": [],
                "formats": []
            }
        except Exception as e:
            logger.error(f"❌ Failed to get book filters: {e}")
            return {}
    
    async def get_library_info(self) -> Dict[str, Any]:
        """Get Calibre library information and statistics"""
        base_info = {
            "enabled": settings.CALIBRE_ENABLED,
            "configured_path": settings.CALIBRE_LIBRARY_PATH,
            "search_weight": settings.CALIBRE_SEARCH_WEIGHT,
            "max_results": settings.CALIBRE_MAX_RESULTS
        }
        
        if not self.is_available():
            base_info.update({
                "available": False,
                "total_books": 0,
                "total_authors": 0,
                "total_series": 0
            })
            
            # Add specific error message if path doesn't exist
            if not os.path.exists(settings.CALIBRE_LIBRARY_PATH):
                base_info["error"] = f"Calibre library path not found: {settings.CALIBRE_LIBRARY_PATH}"
            elif not settings.CALIBRE_ENABLED:
                base_info["error"] = "Calibre integration is disabled"
            
            return base_info
        
        try:
            stats = await self.calibre_repo.get_library_stats()
            base_info.update(stats)
            return base_info
            
        except Exception as e:
            logger.error(f"❌ Failed to get library info: {e}")
            base_info.update({
                "available": False,
                "error": str(e),
                "total_books": 0,
                "total_authors": 0,
                "total_series": 0
            })
            return base_info
    
    async def toggle_calibre_integration(self, enabled: bool) -> bool:
        """Toggle Calibre integration on/off via settings"""
        try:
            # Import settings service here to avoid circular imports
            from services.settings_service import settings_service
            
            # Update database setting
            success = await settings_service.set_setting("calibre_enabled", enabled, "boolean")
            if not success:
                logger.error("❌ Failed to update calibre_enabled setting in database")
                return False
            
            # Update config setting for immediate effect
            settings.CALIBRE_ENABLED = enabled
            
            # Reinitialize if enabled, otherwise mark as unavailable
            if enabled:
                await self.initialize()
            else:
                self._initialized = False
            
            logger.info(f"📚 Calibre integration {'enabled' if enabled else 'disabled'}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to toggle Calibre integration: {e}")
            return False
    
    async def update_calibre_settings(self, settings_dict: Dict[str, Any]) -> bool:
        """Update Calibre-related settings"""
        try:
            # Import settings service here to avoid circular imports
            from services.settings_service import settings_service
            
            success_count = 0
            total_settings = len(settings_dict)
            
            for key, value in settings_dict.items():
                if key == "library_path":
                    success = await settings_service.set_setting("calibre_library_path", value, "string")
                    if success:
                        settings.CALIBRE_LIBRARY_PATH = value
                        success_count += 1
                elif key == "search_weight":
                    success = await settings_service.set_setting("calibre_search_weight", float(value), "float")
                    if success:
                        settings.CALIBRE_SEARCH_WEIGHT = float(value)
                        success_count += 1
                elif key == "max_results":
                    success = await settings_service.set_setting("calibre_max_results", int(value), "integer")
                    if success:
                        settings.CALIBRE_MAX_RESULTS = int(value)
                        success_count += 1
                else:
                    logger.warning(f"⚠️ Unknown Calibre setting: {key}")
            
            if success_count == total_settings:
                # Only reinitialize if all settings were updated successfully
                await self.initialize()
                logger.info("📚 Calibre settings updated successfully")
                return True
            else:
                logger.error(f"❌ Only {success_count}/{total_settings} Calibre settings updated successfully")
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to update Calibre settings: {e}")
            return False
