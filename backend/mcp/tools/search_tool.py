"""
Search Tool - MCP Tool for Vector Search
Allows LLM to perform semantic searches of the knowledge base
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

from mcp.schemas.tool_schemas import (
    SearchDocumentsInput,
    SearchDocumentsOutput, 
    SearchResult,
    SearchFilters,
    ToolResponse,
    ToolError
)
from services.settings_service import settings_service

logger = logging.getLogger(__name__)


class SearchTool:
    """MCP tool for vector-based document search"""
    
    def __init__(self, embedding_manager=None, document_repository=None, chat_service=None):
        """Initialize with required services"""
        self.embedding_manager = embedding_manager
        self.document_repository = document_repository
        self.chat_service = chat_service  # Reference to main chat service for model selection
        self.name = "search_documents"
        self.description = "Search documents using semantic similarity (supports up to 300 results for comprehensive coverage)"
        
    async def initialize(self):
        """Initialize the search tool"""
        if not self.embedding_manager:
            raise ValueError("EmbeddingManager is required")
        if not self.document_repository:
            raise ValueError("DocumentRepository is required")
        
        # Verify embedding manager has required methods
        if not hasattr(self.embedding_manager, 'search_similar'):
            raise ValueError("EmbeddingManager missing search_similar method")
        
        logger.info("🔍 SearchTool initialized")
    
    async def execute(self, input_data: SearchDocumentsInput) -> ToolResponse:
        """Execute vector search with the given parameters"""
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Executing search: '{input_data.query}' (limit: {input_data.limit})")
            
            # Perform vector search
            search_results = await self._perform_vector_search(input_data)
            
            # Convert to search results format
            formatted_results = await self._format_search_results(search_results)
            
            # Create output
            output = SearchDocumentsOutput(
                results=formatted_results,
                total_found=len(formatted_results),
                query_used=input_data.query,
                search_time=time.time() - start_time
            )
            
            logger.info(f"✅ Search completed: {len(formatted_results)} results in {output.search_time:.2f}s")
            
            return ToolResponse(
                success=True,
                data=output,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Search tool execution failed: {e}")
            return ToolResponse(
                success=False,
                error=ToolError(
                    error_code="SEARCH_FAILED",
                    error_message=str(e),
                    details={"query": input_data.query, "limit": input_data.limit}
                ),
                execution_time=time.time() - start_time
            )
    
    async def _perform_vector_search(self, input_data: SearchDocumentsInput) -> List[Dict[str, Any]]:
        """Perform the actual vector search"""
        try:
            # Get the currently selected model from the chat service dropdown
            if self.chat_service and hasattr(self.chat_service, 'current_model') and self.chat_service.current_model:
                selected_model = self.chat_service.current_model
                logger.info(f"🔍 Search using selected model: {selected_model}")
            else:
                # Fallback to settings if chat service not available
                selected_model = await settings_service.get_setting("llm_model")
                logger.warning(f"⚠️ Search fallback to model: {selected_model}")
            
            # First, try exact filename match if the query looks like a filename
            filename_results = await self._search_by_filename(input_data.query)
            if filename_results:
                logger.info(f"✅ Found {len(filename_results)} results by filename match")
                return filename_results
            
            # Prepare search parameters
            search_params = self._prepare_search_params(input_data)
            
            # Perform filtered search if filters are provided
            if input_data.filters:
                return await self._filtered_search(input_data, search_params)
            
            # Perform standard vector search with adjacent chunks for comprehensive coverage
            return await self.embedding_manager.search_similar(
                query_text=input_data.query,
                limit=input_data.limit,
                score_threshold=search_params.get('score_threshold', 0.7),
                use_query_expansion=search_params.get('use_query_expansion', True),
                expansion_model=selected_model,
                include_adjacent_chunks=True  # Include adjacent chunks for better coverage
            )
            
        except Exception as e:
            logger.error(f"❌ Vector search failed: {e}")
            return []
    
    async def _search_by_filename(self, query: str) -> List[Dict[str, Any]]:
        """Search for documents by exact filename match"""
        try:
            # Clean the query to look for filename patterns
            clean_query = query.strip().lower()
            
            # Remove common file extensions if present
            if clean_query.endswith('.pdf'):
                clean_query = clean_query[:-4]
            elif clean_query.endswith('.txt'):
                clean_query = clean_query[:-4]
            elif clean_query.endswith('.doc'):
                clean_query = clean_query[:-4]
            elif clean_query.endswith('.docx'):
                clean_query = clean_query[:-5]
            
            # Search for documents with matching filename
            query = """
                SELECT document_id, filename, title, doc_type, category, tags, author,
                       upload_date, file_size, metadata_json
                FROM document_metadata 
                WHERE LOWER(filename) LIKE $1 OR LOWER(filename) LIKE $2
                ORDER BY upload_date DESC
                LIMIT 10
            """
            
            # Search with and without extension
            results = await self.document_repository.execute_query(
                query, 
                f"%{clean_query}%", 
                f"%{clean_query}.%"
            )
            
            if not results:
                return []
            
            # Convert to search result format
            search_results = []
            for row in results:
                # Get chunks for this document
                chunks = await self.embedding_manager.get_all_document_chunks(row['document_id'])
                
                if chunks:
                    # Use the first chunk as representative content
                    first_chunk = chunks[0]
                    search_results.append({
                        "chunk_id": first_chunk['chunk_id'],
                        "document_id": row['document_id'],
                        "content": first_chunk['content'][:500] + "..." if len(first_chunk['content']) > 500 else first_chunk['content'],
                        "score": 1.0,  # Perfect match for filename
                        "metadata": {
                            "filename": row['filename'],
                            "title": row['title'],
                            "doc_type": row['doc_type'],
                            "category": row['category'],
                            "tags": row['tags'] or [],
                            "author": row['author'],
                            "upload_date": row['upload_date'].isoformat() if row['upload_date'] else None,
                            "file_size": row['file_size'],
                            "match_type": "filename_exact"
                        },
                        "query_source": "filename_search",
                        "source_query": query
                    })
            
            logger.info(f"🔍 Filename search found {len(search_results)} documents")
            return search_results
            
        except Exception as e:
            logger.error(f"❌ Filename search failed: {e}")
            return []
    
    def _prepare_search_params(self, input_data: SearchDocumentsInput) -> Dict[str, Any]:
        """Prepare search parameters based on input"""
        # Determine if query expansion should be used
        # ROOSEVELT'S CHANGE: Default to False - agents handle expansion  
        use_query_expansion = False
        
        # Legacy logic: Previously skipped expansion for specific queries  
        # Now expansion is handled at agent level, so this is informational only
        if len(input_data.query.split()) <= 2 and any(char in input_data.query for char in ['.', '_', '-']):
            logger.info(f"🔍 Specific query detected (expansion handled by agent): {input_data.query}")
        
        # Legacy logic: Previously skipped expansion for long queries
        # Now expansion is handled at agent level, so this is informational only
        if len(input_data.query.split()) > 8:
            logger.info(f"🔍 Long query detected (expansion handled by agent): {input_data.query}")
        
        # Legacy logic: Previously detected pre-expanded queries
        # Now expansion is handled at agent level, so this is informational only  
        query_lower = input_data.query.lower()
        if any(term in query_lower for term in ['and', 'or', 'with', 'including', 'related to']):
            logger.info(f"🔍 Pre-expanded query detected (expansion handled by agent): {input_data.query}")
        
        # Legacy logic: Previously skipped expansion for very short queries
        # Now expansion is handled at agent level, so this is informational only
        if len(input_data.query.strip()) < 5:
            logger.info(f"🔍 Short query detected (expansion handled by agent): {input_data.query}")
        
        return {
            'score_threshold': 0.4,  # Lowered from 0.7 for better discovery of relevant documents
            'use_query_expansion': use_query_expansion
        }
    
    async def _filtered_search(self, input_data: SearchDocumentsInput, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform search with metadata filters"""
        try:
            # Get the currently selected model from the chat service dropdown
            if self.chat_service and hasattr(self.chat_service, 'current_model') and self.chat_service.current_model:
                selected_model = self.chat_service.current_model
            else:
                # Fallback to settings if chat service not available
                selected_model = await settings_service.get_setting("llm_model")
            
            # First get candidate documents based on filters
            candidate_doc_ids = await self._get_filtered_documents(input_data.filters)
            
            if not candidate_doc_ids:
                logger.warning(f"No documents match the specified filters")
                return []
            
            # Perform vector search within filtered documents
            if hasattr(self.embedding_manager, 'search_similar_in_documents'):
                results = await self.embedding_manager.search_similar_in_documents(
                    query_text=input_data.query,
                    document_ids=candidate_doc_ids,
                    limit=input_data.limit,
                    score_threshold=input_data.similarity_threshold
                )
            else:
                # Fallback: search all then filter results with selected model
                all_results = await self.embedding_manager.search_similar(
                    query_text=input_data.query,
                    limit=input_data.limit * 2,  # Get more to account for filtering
                    score_threshold=input_data.similarity_threshold,
                    use_query_expansion=input_data.use_expansion,
                    expansion_model=selected_model
                )
                
                # Filter results to only include documents in candidate set
                candidate_set = set(candidate_doc_ids)
                results = [r for r in all_results if r['document_id'] in candidate_set][:input_data.limit]
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Filtered search failed: {e}")
            raise
    
    async def _get_filtered_documents(self, filters: SearchFilters) -> List[str]:
        """Get document IDs matching metadata filters"""
        try:
            # Build SQL query based on filters
            query_parts = []
            params = []
            
            if filters.categories:
                placeholders = ','.join(['$' + str(len(params) + i + 1) for i in range(len(filters.categories))])
                query_parts.append(f"category = ANY(ARRAY[{placeholders}])")
                params.extend(filters.categories)
            
            if filters.tags:
                # Assuming tags are stored as JSON array
                for tag in filters.tags:
                    params.append(tag)
                    query_parts.append(f"tags ? ${len(params)}")
            
            if filters.document_types:
                placeholders = ','.join(['$' + str(len(params) + i + 1) for i in range(len(filters.document_types))])
                query_parts.append(f"doc_type = ANY(ARRAY[{placeholders}])")
                params.extend([dt.value for dt in filters.document_types])
            
            if filters.date_from:
                params.append(filters.date_from)
                query_parts.append(f"upload_date >= ${len(params)}")
            
            if filters.date_to:
                params.append(filters.date_to)
                query_parts.append(f"upload_date <= ${len(params)}")
            
            if filters.author:
                params.append(filters.author)
                query_parts.append(f"author = ${len(params)}")
            
            if filters.file_size_min:
                params.append(filters.file_size_min)
                query_parts.append(f"file_size >= ${len(params)}")
            
            if filters.file_size_max:
                params.append(filters.file_size_max)
                query_parts.append(f"file_size <= ${len(params)}")
            
            # Build final query
            base_query = "SELECT document_id FROM document_metadata"
            if query_parts:
                where_clause = " AND ".join(query_parts)
                full_query = f"{base_query} WHERE {where_clause}"
            else:
                full_query = base_query
            
            # Execute query
            results = await self.document_repository.execute_query(full_query, *params)
            document_ids = [row['document_id'] for row in results]
            
            logger.info(f"🔍 Filter query found {len(document_ids)} matching documents")
            return document_ids
            
        except Exception as e:
            logger.error(f"❌ Failed to get filtered documents: {e}")
            raise
    
    async def _format_search_results(self, search_results: List[Dict[str, Any]]) -> List[SearchResult]:
        """Format search results into standardized format"""
        formatted_results = []
        
        for result in search_results:
            try:
                # Get document metadata for title
                doc_title = await self._get_document_title(result['document_id'])
                
                formatted_result = SearchResult(
                    chunk_id=result['chunk_id'],
                    document_id=result['document_id'],
                    document_title=doc_title,
                    content=result['content'],
                    similarity_score=result['score'],
                    metadata=result.get('metadata', {})
                )
                formatted_results.append(formatted_result)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to format search result: {e}")
                continue
        
        return formatted_results
    
    async def _get_document_title(self, document_id: str) -> str:
        """Get document title for display"""
        try:
            query = "SELECT title, filename FROM document_metadata WHERE document_id = $1"
            results = await self.document_repository.execute_query(query, document_id)
            
            if results:
                doc = results[0]
                return doc['title'] or doc['filename'] or f"Document {document_id}"
            else:
                return f"Document {document_id}"
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to get document title for {document_id}: {e}")
            return f"Document {document_id}"
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition for registration"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": SearchDocumentsInput.schema(),
            "outputSchema": SearchDocumentsOutput.schema()
        }
