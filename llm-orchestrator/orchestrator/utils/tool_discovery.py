"""
Tool Discovery Service - Semantic tool discovery using vector search

This module provides semantic discovery of tools based on query intent,
complementing the keyword-based detection in dynamic_tool_analyzer.
"""

import logging
from typing import List, Dict, Any, Optional

from orchestrator.utils.tool_vector_store import get_tool_vector_store
from orchestrator.tools.tool_pack_registry import ToolPack, get_tool_by_name

logger = logging.getLogger(__name__)


class ToolDiscoveryService:
    """Service for discovering tools via semantic search"""
    
    def __init__(self):
        self._vector_store = None
    
    async def _get_vector_store(self):
        """Lazy initialization of vector store"""
        if self._vector_store is None:
            self._vector_store = await get_tool_vector_store()
        return self._vector_store
    
    async def discover_tools(
        self,
        query: str,
        top_k: int = 5,
        min_confidence: float = 0.6,
        pack_filter: Optional[ToolPack] = None
    ) -> List[Dict[str, Any]]:
        """
        Discover tools semantically based on query intent
        
        Args:
            query: User query string
            top_k: Maximum number of tools to return
            min_confidence: Minimum similarity score (0.0-1.0)
            pack_filter: Optional pack to filter by
            
        Returns:
            List of discovered tools with metadata and confidence scores
        """
        try:
            vector_store = await self._get_vector_store()
            
            pack_filter_str = pack_filter.value if pack_filter else None
            results = await vector_store.search_tools(
                query=query,
                top_k=top_k,
                pack_filter=pack_filter_str
            )
            
            # Filter by minimum confidence
            filtered_results = [
                r for r in results
                if r.get("score", 0.0) >= min_confidence
            ]
            
            logger.info(
                f"Semantic discovery: {len(filtered_results)} tools found for query "
                f"(min_confidence={min_confidence})"
            )
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Semantic tool discovery failed: {e}")
            # Return empty list on failure (fallback to keyword-based)
            return []
    
    async def discover_tools_by_pack(
        self,
        query: str,
        pack: ToolPack,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Discover tools within a specific pack
        
        Args:
            query: User query string
            pack: Tool pack to search within
            top_k: Maximum number of tools to return
            
        Returns:
            List of discovered tools from the specified pack
        """
        return await self.discover_tools(
            query=query,
            top_k=top_k,
            pack_filter=pack
        )
    
    async def get_tool_names(
        self,
        query: str,
        top_k: int = 5,
        min_confidence: float = 0.6
    ) -> List[str]:
        """
        Get just the tool names (simplified interface)
        
        Args:
            query: User query string
            top_k: Maximum number of tools to return
            min_confidence: Minimum similarity score
            
        Returns:
            List of tool names
        """
        results = await self.discover_tools(
            query=query,
            top_k=top_k,
            min_confidence=min_confidence
        )
        return [r["tool_name"] for r in results if r.get("tool_name")]


# Global instance
_tool_discovery_service: Optional[ToolDiscoveryService] = None


async def get_tool_discovery_service() -> ToolDiscoveryService:
    """Get or create global tool discovery service instance"""
    global _tool_discovery_service
    if _tool_discovery_service is None:
        _tool_discovery_service = ToolDiscoveryService()
    return _tool_discovery_service
