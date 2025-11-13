"""
Direct Search Service - Provides semantic search without LLM processing
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from services.embedding_service_wrapper import get_embedding_service
from repositories.document_repository import DocumentRepository
from config import settings

logger = logging.getLogger(__name__)


class DirectSearchService:
    """Service for direct semantic search without LLM processing"""
    
    def __init__(self):
        self.embedding_manager = None
        self.document_repository = DocumentRepository()
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Lazy initialization of embedding service wrapper"""
        if not self._initialized:
            self.embedding_manager = await get_embedding_service()
            self._initialized = True
    
    async def search_documents(
        self,
        query: str,
        limit: int = 20,
        similarity_threshold: float = 0.3,  # Lowered from 0.7 to 0.3 for better recall
        user_id: Optional[str] = None,
        document_types: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Perform direct semantic search on documents
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            user_id: Optional user ID for filtering results
            document_types: Filter by document types (pdf, txt, etc.)
            categories: Filter by document categories
            tags: Filter by document tags
            date_from: Filter documents from this date
            date_to: Filter documents to this date
            include_metadata: Include document metadata in results
            
        Returns:
            Dict containing search results and metadata
        """
        try:
            logger.info(f"🔍 Direct search query: '{query}' with {limit} results")
            
            # Ensure embedding manager is initialized
            await self._ensure_initialized()
            
            # Perform vector search using EmbeddingManager's search_similar method
            # EmbeddingManager.search_similar expects query_text and handles embedding generation internally
            search_results = await self.embedding_manager.search_similar(
                query_text=query,
                limit=limit,
                score_threshold=similarity_threshold,
                user_id=user_id if user_id and user_id != "system" else None
            )
            
            # Format results (already filtered by threshold in search_similar)
            filtered_results = []
            for result in search_results[:limit]:
                formatted_result = await self._format_search_result(
                    result, 
                    query, 
                    include_metadata
                )
                if formatted_result:
                    filtered_results.append(formatted_result)
            
            logger.info(f"✅ Direct search completed: {len(filtered_results)} results")
            
            return {
                "success": True,
                "query": query,
                "results": filtered_results,
                "total_results": len(filtered_results),
                "similarity_threshold": similarity_threshold,
                "search_metadata": {
                    "query_length": len(query),
                    "embedding_dimensions": len(query_embeddings[0]) if query_embeddings else 0,
                    "search_timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Direct search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "total_results": 0
            }
    
    async def _format_search_result(
        self, 
        result: Dict, 
        query: str, 
        include_metadata: bool
    ) -> Optional[Dict]:
        """Format a single search result for display"""
        try:
            chunk_id = result.get("chunk_id")
            document_id = result.get("document_id")
            # search_similar returns 'score', not 'similarity_score'
            similarity_score = result.get("score", result.get("similarity_score", 0.0))
            # search_similar returns 'content', not 'chunk_text'
            chunk_text = result.get("content", result.get("chunk_text", ""))
            
            if not chunk_id or not document_id:
                return None
            
            # Get document metadata if requested
            document_metadata = {}
            if include_metadata:
                doc_info = await self.document_repository.get_document_by_id(document_id)
                if doc_info:
                    document_metadata = {
                        "document_id": document_id,
                        "filename": doc_info.get("filename", ""),
                        "title": doc_info.get("title", ""),
                        "doc_type": doc_info.get("doc_type", ""),
                        "category": doc_info.get("category", ""),
                        "tags": doc_info.get("tags", []),
                        "upload_date": doc_info.get("upload_date", ""),
                        "file_size": doc_info.get("file_size", 0),
                        "page_count": doc_info.get("page_count", 0),
                        "author": doc_info.get("author", ""),
                        "description": doc_info.get("description", "")
                    }
            
            # Highlight query terms in the text
            highlighted_text = self._highlight_query_terms(chunk_text, query)
            
            # Extract context around the match
            context = self._extract_context(chunk_text, query)
            
            formatted_result = {
                "chunk_id": chunk_id,
                "similarity_score": round(similarity_score, 4),
                "text": chunk_text,
                "highlighted_text": highlighted_text,
                "context": context,
                "chunk_metadata": {
                    "chunk_index": result.get("chunk_index", 0),
                    "page_number": result.get("page_number"),
                    "section_title": result.get("section_title", ""),
                    "text_length": len(chunk_text),
                    "word_count": len(chunk_text.split())
                }
            }
            
            if include_metadata:
                formatted_result["document"] = document_metadata
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"❌ Failed to format search result: {e}")
            return None
    
    def _highlight_query_terms(self, text: str, query: str) -> str:
        """Highlight query terms in the text"""
        try:
            import re
            
            # Split query into individual terms
            query_terms = [term.strip().lower() for term in query.split() if len(term.strip()) > 2]
            
            highlighted_text = text
            for term in query_terms:
                # Use word boundaries to match whole words
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                highlighted_text = pattern.sub(f"**{term}**", highlighted_text)
            
            return highlighted_text
            
        except Exception as e:
            logger.error(f"❌ Failed to highlight query terms: {e}")
            return text
    
    def _extract_context(self, text: str, query: str, context_length: int = 200) -> Dict:
        """Extract context around query matches"""
        try:
            import re
            
            query_terms = [term.strip().lower() for term in query.split() if len(term.strip()) > 2]
            
            # Find the best match position
            best_match_pos = 0
            best_match_score = 0
            
            for term in query_terms:
                match = re.search(re.escape(term), text, re.IGNORECASE)
                if match:
                    # Score based on term length and position
                    score = len(term) * (1.0 - match.start() / len(text))
                    if score > best_match_score:
                        best_match_score = score
                        best_match_pos = match.start()
            
            # Extract context around the best match
            start_pos = max(0, best_match_pos - context_length // 2)
            end_pos = min(len(text), best_match_pos + context_length // 2)
            
            context_text = text[start_pos:end_pos]
            
            # Add ellipsis if truncated
            if start_pos > 0:
                context_text = "..." + context_text
            if end_pos < len(text):
                context_text = context_text + "..."
            
            return {
                "text": context_text,
                "start_position": start_pos,
                "end_position": end_pos,
                "match_position": best_match_pos - start_pos if best_match_pos >= start_pos else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to extract context: {e}")
            return {
                "text": text[:400] + "..." if len(text) > 400 else text,
                "start_position": 0,
                "end_position": min(400, len(text)),
                "match_position": 0
            }
    
    async def get_search_suggestions(self, partial_query: str, limit: int = 10) -> List[str]:
        """Get search suggestions based on partial query"""
        try:
            # This could be enhanced with a proper suggestion system
            # For now, return some basic suggestions based on document content
            
            if len(partial_query) < 2:
                return []
            
            # Get common terms from document chunks
            suggestions = await self.embedding_manager.get_common_terms(
                prefix=partial_query,
                limit=limit
            )
            
            return suggestions
            
        except Exception as e:
            logger.error(f"❌ Failed to get search suggestions: {e}")
            return []
    
    async def get_search_filters(self) -> Dict[str, List[str]]:
        """Get available filter options for search"""
        try:
            # Get available document types, categories, and tags
            filter_options = await self.document_repository.get_filter_options()
            
            return {
                "document_types": filter_options.get("doc_types", []),
                "categories": filter_options.get("categories", []),
                "tags": filter_options.get("tags", []),
                "date_range": {
                    "earliest": filter_options.get("earliest_date"),
                    "latest": filter_options.get("latest_date")
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get search filters: {e}")
            return {
                "document_types": [],
                "categories": [],
                "tags": [],
                "date_range": {"earliest": None, "latest": None}
            }
    
    async def export_search_results(
        self, 
        results: List[Dict], 
        format_type: str = "json"
    ) -> Dict[str, Any]:
        """Export search results in various formats"""
        try:
            if format_type.lower() == "csv":
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow([
                    "Similarity Score", "Document Title", "Document Type", 
                    "Category", "Text Preview", "Page Number"
                ])
                
                # Write data
                for result in results:
                    doc = result.get("document", {})
                    chunk = result.get("chunk_metadata", {})
                    
                    writer.writerow([
                        result.get("similarity_score", 0),
                        doc.get("title", doc.get("filename", "")),
                        doc.get("doc_type", ""),
                        doc.get("category", ""),
                        result.get("text", "")[:200] + "...",
                        chunk.get("page_number", "")
                    ])
                
                return {
                    "success": True,
                    "format": "csv",
                    "data": output.getvalue(),
                    "filename": f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                }
            
            else:  # JSON format
                return {
                    "success": True,
                    "format": "json",
                    "data": results,
                    "filename": f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to export search results: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def search_web(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search the web for information using SearXNG"""
        try:
            logger.info(f"🌐 Web search query: '{query}' with {limit} results")
            
            # Use the same SearXNG implementation as the LangGraph tools
            from services.langgraph_tools.web_content_tools import WebContentTools
            web_tools = WebContentTools()
            results = await web_tools._search_searxng(query, limit)
            
            return {
                "success": True,
                "results": results,
                "count": len(results),
                "query": query,
                "message": f"Found {len(results)} results via SearXNG"
            }
            
        except Exception as e:
            logger.error(f"❌ Web search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "count": 0
            }
