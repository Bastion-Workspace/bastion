"""
Unified Search Tools Module
Consolidated local search functionality for LangGraph agents
"""

import logging
import asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta
import json

from services.embedding_service_wrapper import get_embedding_service
from services.knowledge_graph_service import KnowledgeGraphService

logger = logging.getLogger(__name__)


class UnifiedSearchTools:
    """Unified search tools for LangGraph agents"""
    
    def __init__(self):
        # Use lazy initialization to avoid import-time service creation
        self._embedding_manager = None
        self._knowledge_graph_service = None
    
    async def _get_embedding_manager(self):
        """Get embedding service wrapper with lazy initialization"""
        if self._embedding_manager is None:
            try:
                self._embedding_manager = await get_embedding_service()
                logger.info("✅ Embedding service wrapper initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize embedding service wrapper: {e}")
                # Return None on failure
        return self._embedding_manager
    
    async def _get_knowledge_graph_service(self):
        """Get knowledge graph service with lazy initialization"""
        if self._knowledge_graph_service is None:
            self._knowledge_graph_service = KnowledgeGraphService()
            await self._knowledge_graph_service.initialize()
        return self._knowledge_graph_service
    
    def get_tools(self) -> Dict[str, Any]:
        """Get all unified search tools"""
        return {
            "search_local": self.search_local,
            "get_document": self.get_document,
        }
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all unified search tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_local",
                    "description": "Unified search across all local resources (documents, entity knowledge graph) - NO PERMISSION REQUIRED. Supports filtering by document tags and categories.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "search_types": {"type": "array", "items": {"type": "string"}, "description": "Types of search to perform: 'vector' (documents), 'entities' (knowledge graph), 'filename'", "default": ["vector", "entities"]},
                            "limit": {"type": "integer", "description": "Maximum number of results per search type", "default": 200},
                            "filter_tags": {"type": "array", "items": {"type": "string"}, "description": "OPTIONAL: Filter documents by tags (e.g. ['founding', 'historical']). Use when user mentions specific document collections or categories in their query."},
                            "filter_category": {"type": "string", "description": "OPTIONAL: Filter documents by category (e.g. 'technical', 'academic'). Use when user specifies a document type."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_document",
                    "description": "Get document content and metadata by document ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_id": {"type": "string", "description": "Document ID to retrieve"},
                        },
                        "required": ["document_id"]
                    }
                }
            }
        ]
    
    async def search_local(self, query: str, search_types: List[str] = None, limit: int = 200, user_id: str = None, filter_category: str = None, filter_tags: List[str] = None, team_ids: List[str] = None) -> Dict[str, Any]:
        """
        Unified search across all local resources with smart tag filtering
        
        **ROOSEVELT SMART FILTERING**: Supports category and tag-based filtering!
        """
        try:
            logger.info(f"🔍 LangGraph unified local search: {query[:50]}...")
            
            if search_types is None:
                search_types = ["vector", "entities"]
            
            # **ROOSEVELT TAG DETECTION**: Auto-detect tags from query if not explicitly provided
            if not filter_category and not filter_tags:
                try:
                    from services.langgraph_tools.tag_detection_service import get_tag_detection_service
                    from repositories.document_repository import DocumentRepository
                    
                    # Get available tags and categories from database
                    doc_repo = DocumentRepository()
                    await doc_repo.initialize()
                    
                    # Fetch available tags and categories
                    available_tags_result = await doc_repo.get_all_tags()
                    available_categories_result = await doc_repo.get_all_categories()
                    
                    available_tags = [tag for tag in available_tags_result if tag]
                    available_categories = [cat for cat in available_categories_result if cat]
                    
                    logger.info(f"📋 Available for matching: {len(available_tags)} tags, {len(available_categories)} categories")
                    
                    # Detect and match tags
                    tag_service = get_tag_detection_service()
                    detection_result = await tag_service.detect_and_match_filters(
                        query, available_tags, available_categories
                    )
                    
                    if detection_result["should_filter"]:
                        filter_category = detection_result["filter_category"]
                        filter_tags = detection_result["filter_tags"]
                        filter_msg = tag_service.format_filter_message(detection_result)
                        logger.info(f"✅ {filter_msg}")
                    else:
                        logger.info("🔍 No tag filters detected, searching all documents")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Tag detection failed, continuing without filters: {e}")
            
            # Perform searches in parallel based on requested types
            search_tasks = []
            
            if "vector" in search_types:
                search_tasks.append(self._search_vector(query, limit, user_id, filter_category, filter_tags, team_ids))
            
            if "entities" in search_types:
                search_tasks.append(self._search_entities(query, limit))
            
            if "filename" in search_types:
                search_tasks.append(self._search_filename(query, limit))
            
            # Execute searches in parallel
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine and deduplicate results
            combined_results = []
            seen_chunks = set()
            search_summary = []
            
            for result in search_results:
                if isinstance(result, dict) and result.get("success"):
                    search_type = result.get("search_type", "unknown")
                    result_count = len(result.get("results", []))
                    search_summary.append(f"{search_type}: {result_count} results")
                    
                    for item in result.get("results", []):
                        chunk_id = item.get("chunk_id")
                        if chunk_id and chunk_id not in seen_chunks:
                            combined_results.append(item)
                            seen_chunks.add(chunk_id)
                elif isinstance(result, Exception):
                    search_summary.append(f"error: {str(result)}")
                else:
                    search_summary.append("failed")
            
            # If no results found, provide detailed feedback
            if len(combined_results) == 0:
                return {
                    "success": True,
                    "results": [],
                    "count": 0,
                    "query": query,
                    "search_types_used": search_types,
                    "total_searches": len(search_tasks),
                    "search_summary": search_summary,
                    "message": f"No results found. Search summary: {', '.join(search_summary)}"
                }
            
            return {
                "success": True,
                "results": combined_results[:limit * len(search_types)],
                "count": len(combined_results),
                "query": query,
                "search_types_used": search_types,
                "total_searches": len(search_tasks),
                "search_summary": search_summary
            }
            
        except Exception as e:
            logger.error(f"❌ Unified local search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "count": 0
            }
    
    async def _search_vector(self, query: str, limit: int, user_id: str = None, filter_category: str = None, filter_tags: List[str] = None, team_ids: List[str] = None) -> Dict[str, Any]:
        """Search documents using semantic similarity with enhanced citations and optional tag filtering"""
        try:
            logger.info(f"Vector search called with user_id: {user_id}, teams: {team_ids}, category: {filter_category}, tags: {filter_tags}")
            embedding_service = await self._get_embedding_manager()
            
            if not embedding_service:
                logger.warning("Embedding service not initialized")
                return {
                    "success": True,
                    "results": [],
                    "search_type": "vector",
                    "message": "Vector search not available - embedding service not initialized"
                }
            
            # Generate query embedding
            query_embeddings = await embedding_service.generate_embeddings([query])
            if not query_embeddings or len(query_embeddings) == 0:
                logger.error("Failed to generate query embedding")
                return {
                    "success": False,
                    "results": [],
                    "search_type": "vector",
                    "message": "Failed to generate query embedding"
                }
            
            # Search via embedding service (uses VectorStoreService internally)
            logger.info(f"Searching vector store with filters - category: {filter_category}, tags: {filter_tags}, teams: {team_ids}")
            results = await embedding_service.search_similar(
                query_embedding=query_embeddings[0],
                limit=limit,
                score_threshold=0.3,
                user_id=user_id,
                team_ids=team_ids,
                filter_category=filter_category,
                filter_tags=filter_tags
            )
            
            # Enhance results with citation information
            enhanced_results = []
            for result in results:
                enhanced_result = dict(result)  # Copy the original result
                
                # Extract citation information from metadata
                metadata = result.get("metadata", {})
                document_id = result.get("document_id", "")
                
                # Create citation fields for Research Agent
                enhanced_result["citation_url"] = self._generate_document_url(document_id, metadata)
                enhanced_result["citation_title"] = self._extract_citation_title(metadata)
                enhanced_result["citation_reference"] = self._generate_document_reference(document_id, metadata)
                enhanced_result["source_type"] = "document"
                
                enhanced_results.append(enhanced_result)
            
            return {
                "success": True,
                "results": enhanced_results,
                "search_type": "vector"
            }
            
        except Exception as e:
            logger.error(f"❌ Vector search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "search_type": "vector"
            }
    
    async def _search_entities(self, query: str, limit: int) -> Dict[str, Any]:
        """Search using entity relationships"""
        try:
            knowledge_graph_service = await self._get_knowledge_graph_service()
            entities = await knowledge_graph_service.extract_entities_from_text(query)
            entity_names = [e["name"] for e in entities]
            
            if not entity_names:
                return {
                    "success": True,
                    "results": [],
                    "search_type": "entities",
                    "message": "No entities found in query"
                }
            
            # Find documents by entities
            document_ids = await knowledge_graph_service.find_documents_by_entities(entity_names)
            
            if not document_ids:
                return {
                    "success": True,
                    "results": [],
                    "search_type": "entities",
                    "entities_found": entity_names,
                    "message": "No documents found for entities"
                }
            
            # Get document content for found IDs
            embedding_manager = await self._get_embedding_manager()
            results = []
            
            # Check if embedding manager is properly initialized
            if not hasattr(embedding_manager, 'qdrant_client') or embedding_manager.qdrant_client is None:
                logger.warning("⚠️ Embedding manager not properly initialized for entity search")
                return {
                    "success": True,
                    "results": [],
                    "search_type": "entities",
                    "entities_found": entity_names,
                    "document_ids_found": len(document_ids),
                    "message": "Entity search found documents but embedding manager not initialized"
                }
            
            for doc_id in document_ids[:limit]:
                try:
                    doc_chunks = await embedding_manager.search_similar_in_documents(
                        query_text=query,
                        document_ids=[doc_id],
                        limit=5
                    )
                    results.extend(doc_chunks)
                except Exception as chunk_error:
                    logger.warning(f"⚠️ Failed to get chunks for document {doc_id}: {chunk_error}")
                    continue
            
            return {
                "success": True,
                "results": results,
                "search_type": "entities",
                "entities_found": entity_names,
                "document_ids_found": len(document_ids)
            }
            
        except Exception as e:
            logger.error(f"❌ Entity search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "search_type": "entities"
            }
    
    async def _search_filename(self, query: str, limit: int) -> Dict[str, Any]:
        """Search by filename (placeholder for now)"""
        try:
            # For now, return a placeholder since this functionality may need implementation
            return {
                "success": True,
                "results": [],
                "search_type": "filename",
                "message": "Filename search not yet implemented"
            }
            
        except Exception as e:
            logger.error(f"❌ Filename search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "search_type": "filename"
            }
    
    async def get_document(self, document_id: str, user_id: str = None) -> Dict[str, Any]:
        """Get document content and metadata by document ID"""
        try:
            logger.info(f"📄 LangGraph getting document: {document_id[:20]}...")
            
            # Import required services
            from repositories.document_repository import DocumentRepository
            from config import settings
            from pathlib import Path
            import glob
            
            # Get document metadata first
            doc_repo = DocumentRepository()
            document = await doc_repo.get_by_id(document_id)
            
            if not document:
                return {
                    "success": False,
                    "error": "Document not found in database",
                    "document_id": document_id
                }
            
            # Try to load the actual file content
            full_content = None
            content_source = "file"
            
            try:
                # **ROOSEVELT FIX**: Use folder_service to get correct file path
                from services.service_container import service_container
                folder_service = service_container.folder_service
                
                filename = getattr(document, 'filename', None)
                if not filename:
                    return {
                        "success": False,
                        "error": "Document has no filename",
                        "document_id": document_id
                    }
                
                user_doc_id = getattr(document, 'user_id', None)
                folder_id = getattr(document, 'folder_id', None)
                collection_type = getattr(document, 'collection_type', 'user')
                
                # Skip reading binary files (PDFs, images) - they don't have text content
                image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg']
                is_binary_file = filename.lower().endswith('.pdf') or any(filename.lower().endswith(ext) for ext in image_extensions)
                
                if is_binary_file:
                    logger.info(f"⏭️ Skipping content load for binary file: {filename}")
                    full_content = ""
                else:
                    # Try to get file path via folder service (handles new folder structure)
                    try:
                        file_path_str = await folder_service.get_document_file_path(
                            filename=filename,
                            folder_id=folder_id,
                            user_id=user_doc_id,
                            collection_type=collection_type
                        )
                        file_path = Path(file_path_str)
                        
                        if file_path.exists():
                            logger.info(f"✅ Found file via folder service: {file_path}")
                            with open(file_path, 'r', encoding='utf-8') as f:
                                full_content = f.read()
                    except Exception as folder_error:
                        logger.warning(f"⚠️ Folder service lookup failed: {folder_error}")
                        # Fall back to legacy search below
                
                # Fall back to legacy path search if folder service didn't work
                upload_dir = Path(settings.UPLOAD_DIR)
                
                if not full_content and hasattr(document, 'filename') and document.filename:
                    filename = document.filename
                    
                    # For web sources (RSS/scraped), look in web_sources directory
                    if filename.endswith('.md'):
                        potential_paths = [
                            upload_dir / "web_sources" / "rss_articles" / "*" / filename,
                            upload_dir / "web_sources" / "scraped_content" / "*" / filename,
                            upload_dir / filename
                        ]
                        
                        for path_pattern in potential_paths:
                            matches = glob.glob(str(path_pattern))
                            if matches:
                                file_path = Path(matches[0])
                                if file_path.exists():
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        full_content = f.read()
                                    logger.info(f"✅ Loaded content from markdown file: {file_path}")
                                    break
                    else:
                        # **ROOSEVELT FIX**: Try folder-based path first (new), then UUID-prefixed (legacy)
                        # New path: /app/uploads/Global/Reference/filename.pdf or /app/uploads/folder_path/filename.pdf
                        # Legacy path: /app/uploads/{uuid}_{filename}
                        
                        potential_file_paths = []
                        
                        # Try to construct folder-based path from document metadata
                        if hasattr(document, 'file_path') and document.file_path:
                            # Use the exact file_path if available
                            folder_path = Path(document.file_path)
                            if folder_path.exists():
                                potential_file_paths.append(folder_path)
                        
                        # Try legacy UUID-prefixed path
                        potential_file_paths.append(upload_dir / f"{document_id}_{filename}")
                        
                        # Try searching in Global and user subfolders
                        for subfolder in ["Global", "User"]:
                            subfolder_path = upload_dir / subfolder
                            if subfolder_path.exists():
                                # Search recursively for the file
                                matches = list(subfolder_path.rglob(filename))
                                potential_file_paths.extend(matches)
                        
                        # Try the first path that exists
                        file_path = None
                        for path in potential_file_paths:
                            if path.exists():
                                file_path = path
                                break
                        
                        if file_path and file_path.exists():
                            # Try different encodings for EPUB and other binary-derived files
                            encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
                            
                            for encoding in encodings_to_try:
                                try:
                                    with open(file_path, 'r', encoding=encoding) as f:
                                        full_content = f.read()
                                    logger.info(f"✅ Loaded content from file: {file_path} (encoding: {encoding})")
                                    break
                                except UnicodeDecodeError:
                                    continue
                            
                            if not full_content:
                                # If text reading fails, this might be a binary file
                                logger.warning(f"⚠️ Could not read file as text with any encoding: {file_path}")
                                return {
                                    "success": False,
                                    "error": f"File appears to be binary or corrupted: {filename}",
                                    "document_id": document_id,
                                    "metadata": {
                                        "filename": filename,
                                        "document_id": document_id,
                                        "file_path": str(file_path)
                                    }
                                }
                
                if not full_content:
                    return {
                        "success": False, 
                        "error": f"Document file not found on disk: {getattr(document, 'filename', 'unknown')}",
                        "document_id": document_id,
                        "metadata": {
                            "document_id": document_id,
                            "filename": getattr(document, 'filename', 'unknown')
                        }
                    }
                
                # Strip YAML frontmatter from markdown content  
                display_content = full_content
                if hasattr(document, 'filename') and document.filename.endswith('.md'):
                    # Simple YAML frontmatter removal
                    if display_content.startswith('---\n'):
                        try:
                            end_index = display_content.find('\n---\n', 4)
                            if end_index != -1:
                                display_content = display_content[end_index + 5:]
                        except:
                            pass  # Keep original content if parsing fails
                
                # Build metadata response
                metadata = {
                    "document_id": document.document_id,
                    "filename": getattr(document, 'filename', ''),
                    "title": getattr(document, 'title', ''),
                    "author": getattr(document, 'author', ''),
                    "category": getattr(document, 'category', ''),
                    "language": getattr(document, 'language', ''),
                    "content_length": len(display_content),
                    "content_source": content_source
                }
                
                return {
                    "success": True,
                    "document_id": document_id,
                    "content": display_content,
                    "metadata": metadata,
                    "message": f"Successfully retrieved document content ({len(display_content)} characters)"
                }
                
            except Exception as file_error:
                logger.error(f"❌ Failed to load file for document {document_id}: {file_error}")
                return {
                    "success": False,
                    "error": f"Failed to load document file: {str(file_error)}",
                    "document_id": document_id,
                    "metadata": {
                        "document_id": document_id,
                        "filename": getattr(document, 'filename', 'unknown') if document else 'unknown'
                    }
                }
            
        except Exception as e:
            logger.error(f"❌ Document retrieval failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_id": document_id
            }
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS or MM:SS timestamp"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def _generate_document_url(self, document_id: str, metadata: Dict[str, Any]) -> str:
        """Generate a URL reference for a local document"""
        try:
            # Check if the document has an original URL (web-ingested documents)
            original_url = metadata.get("url") or metadata.get("source_url")
            if original_url:
                return original_url
            
            # For local documents, create an internal reference URL
            # This could point to your document viewer or file system
            filename = metadata.get("filename", "")
            if filename:
                # Create a relative path reference
                return f"/documents/{document_id}/{filename}"
            
            # Fallback to document ID reference
            return f"/documents/{document_id}"
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate document URL: {e}")
            return f"/documents/{document_id}"
    
    def _extract_citation_title(self, metadata: Dict[str, Any]) -> str:
        """Extract a proper title for citation purposes"""
        try:
            # Try various title fields in order of preference
            title = (
                metadata.get("title") or
                metadata.get("name") or
                metadata.get("filename") or
                metadata.get("subject") or
                "Untitled Document"
            )
            
            # Clean up the title
            if isinstance(title, str):
                # Remove file extensions for cleaner citations
                if title.endswith(('.pdf', '.txt', '.doc', '.docx', '.html')):
                    title = title.rsplit('.', 1)[0]
                return title.strip()
            
            return "Untitled Document"
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract citation title: {e}")
            return "Untitled Document"
    
    def _generate_document_reference(self, document_id: str, metadata: Dict[str, Any]) -> str:
        """Generate a properly formatted document reference"""
        try:
            title = self._extract_citation_title(metadata)
            filename = metadata.get("filename", "")
            created_date = metadata.get("created_date") or metadata.get("timestamp")
            
            # Build reference string
            reference_parts = [f'"{title}"']
            
            if filename and filename not in title:
                reference_parts.append(f"[File: {filename}]")
            
            reference_parts.append(f"Doc ID: {document_id[:8]}...")
            
            if created_date:
                try:
                    # Try to format the date nicely
                    from datetime import datetime
                    if isinstance(created_date, str):
                        # Parse ISO date format
                        dt = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
                        reference_parts.append(f"({dt.strftime('%Y-%m-%d')})")
                except:
                    # Fallback to raw date string
                    reference_parts.append(f"({str(created_date)[:10]})")
            
            return " ".join(reference_parts)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate document reference: {e}")
            return f'Document {document_id[:8]}...'


# Global instance for use by tool registry
_unified_search_instance = None


async def _get_unified_search():
    """Get global unified search instance"""
    global _unified_search_instance
    if _unified_search_instance is None:
        _unified_search_instance = UnifiedSearchTools()
    return _unified_search_instance


async def unified_local_search(query: str, limit: int = 200, search_types: List[str] = None, user_id: str = None, filter_tags: List[str] = None, filter_category: str = None) -> str:
    """
    LangGraph tool function: Unified local search across all resources
    
    **ROOSEVELT SMART FILTERING**: Supports tag and category filtering!
    """
    search_instance = await _get_unified_search()
    result = await search_instance.search_local(query, search_types, limit, user_id, filter_category, filter_tags)
    
    # Format results as readable text for the LLM
    if not result.get("success"):
        return f"❌ Search failed: {result.get('error', 'Unknown error')}"
    
    results = result.get("results", [])
    count = result.get("count", 0)
    search_summary = result.get("search_summary", [])
    
    if count == 0:
        return f"🔍 No results found for '{query}'. Search summary: {', '.join(search_summary)}"
    
    # Format results with document content and metadata
    formatted_results = [f"🔍 **Found {count} relevant results for '{query}':**\n"]
    
    for i, item in enumerate(results[:20], 1):  # Limit to top 20 for readability
        doc_id = item.get("document_id", "unknown")
        score = item.get("score", 0.0)
        content = item.get("content", "")
        source_collection = item.get("source_collection", "unknown")
        metadata = item.get("metadata", {})
        
        # Get document title/filename from metadata
        title = metadata.get("title") or metadata.get("filename") or f"Document {doc_id[:8]}"
        
        # Truncate content for readability
        content_preview = content[:300] + "..." if len(content) > 300 else content
        
        formatted_results.append(
            f"\n**{i}. {title}** (Score: {score:.3f}, Collection: {source_collection})\n"
            f"Content: {content_preview}\n"
        )
    
    if count > 20:
        formatted_results.append(f"\n... and {count - 20} more results")
    
    # Add search summary
    formatted_results.append(f"\n📊 **Search Summary:** {', '.join(search_summary)}")
    
    return "".join(formatted_results)


async def search_conversation_cache(query: str, conversation_id: str = None, freshness_hours: int = 24, user_id: str = None) -> str:
    """
    ROOSEVELT'S UNIVERSAL CONVERSATION CACHE: Access previous research and chat work from this conversation
    
    This tool allows ANY agent to check if previous work in the conversation already covers the query,
    avoiding redundant searches and providing instant access to cached intelligence.
    
    Available to: ALL agents (research, chat, data_formatting, weather, etc.)
    Use Cases: Follow-up questions, timeline requests, detail expansion, model answer generation
    """
    try:
        logger.info(f"🏆 UNIVERSAL CACHE SEARCH: Checking conversation intelligence for '{query[:50]}...'")
        
        # DEPRECATED: Backend orchestrator removed - gRPC orchestrator handles its own state
        # Conversation state is managed by llm-orchestrator service
        return "🔍 Conversation cache unavailable - backend orchestrator removed. State is managed by gRPC orchestrator."
        
        # Extract cached intelligence from conversation state
        cache_analysis = await _analyze_conversation_cache(query, conversation_state)
        
        if cache_analysis["coverage_score"] > 0.7:
            logger.info(f"🏆 CACHE HIT: High coverage ({cache_analysis['coverage_score']:.2f}) found in conversation cache")
            return _format_cache_results(cache_analysis)
        elif cache_analysis["coverage_score"] > 0.3:
            logger.info(f"🔄 PARTIAL CACHE: Moderate coverage ({cache_analysis['coverage_score']:.2f}) - supplemental search recommended")
            return _format_partial_cache_results(cache_analysis)
        else:
            logger.info(f"❌ CACHE MISS: Low coverage ({cache_analysis['coverage_score']:.2f}) - new search needed")
            return "🔍 No relevant information found in conversation cache. Recommend proceeding with fresh local/web search."

    except Exception as e:
        logger.error(f"❌ Cache search failed: {e}")
        return f"❌ Cache search failed: {str(e)}"


async def _analyze_conversation_cache(query: str, conversation_state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze conversation cache to determine coverage of new query"""
    try:
        # Extract all available intelligence from conversation
        cache_sources = {
            "research_findings": conversation_state.get("shared_memory", {}).get("research_findings", {}),
            "chat_outputs": _extract_chat_outputs(conversation_state.get("messages", [])),
            "agent_results": _extract_agent_results(conversation_state),
            "citations": _extract_cached_citations(conversation_state)
        }
        
        # Use simple keyword-based coverage analysis for now
        # TODO: Enhance with semantic similarity using embeddings
        query_words = set(query.lower().split())
        
        coverage_score = 0.0
        relevant_sources = []
        
        # Check research findings
        for topic, findings in cache_sources["research_findings"].items():
            if isinstance(findings, dict):
                findings_text = findings.get("findings", "").lower()
                word_overlap = len(query_words.intersection(set(findings_text.split()))) / len(query_words)
                if word_overlap > 0.2:
                    coverage_score = max(coverage_score, word_overlap)
                    relevant_sources.append({"type": "research", "topic": topic, "content": findings})
        
        # Check chat outputs  
        for output in cache_sources["chat_outputs"]:
            content_words = set(output["content"].lower().split())
            word_overlap = len(query_words.intersection(content_words)) / len(query_words)
            if word_overlap > 0.2:
                coverage_score = max(coverage_score, word_overlap)
                relevant_sources.append({"type": "chat", "content": output["content"][:500]})
        
        return {
            "coverage_score": min(coverage_score, 1.0),
            "relevant_sources": relevant_sources,
            "cache_sources_found": len(relevant_sources),
            "query_analysis": {"words": list(query_words), "length": len(query)}
        }
        
    except Exception as e:
        logger.error(f"❌ Cache analysis failed: {e}")
        return {"coverage_score": 0.0, "relevant_sources": [], "error": str(e)}


def _extract_chat_outputs(messages: List) -> List[Dict[str, Any]]:
    """Extract previous chat agent outputs from conversation messages"""
    chat_outputs = []
    try:
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == "ai" and hasattr(msg, 'content'):
                content = msg.content
                # Look for chat agent outputs (questions, brainstorming, etc.)
                if any(indicator in content.lower() for indicator in [
                    "here are some", "questions:", "examples:", "ideas:", "brainstorm",
                    "interview questions", "you could ask", "consider these", "suggestions:"
                ]):
                    chat_outputs.append({
                        "content": content,
                        "timestamp": "recent",  # TODO: Extract actual timestamp
                        "type": "chat_generation"
                    })
    except Exception as e:
        logger.error(f"❌ Failed to extract chat outputs: {e}")
    
    return chat_outputs[-5:]  # Last 5 chat outputs


def _extract_agent_results(conversation_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract previous agent results from conversation state"""
    try:
        # Look for agent results in shared memory or state
        agent_results = []
        
        # Check agent_results in state
        if "agent_results" in conversation_state:
            agent_results.append(conversation_state["agent_results"])
        
        # TODO: Extract from message history or shared memory
        return agent_results
        
    except Exception as e:
        logger.error(f"❌ Failed to extract agent results: {e}")
        return []


def _extract_cached_citations(conversation_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract cached citations and sources from conversation"""
    try:
        citations = []
        shared_memory = conversation_state.get("shared_memory", {})
        
        # Extract from search results
        search_results = shared_memory.get("search_results", {})
        for search_type, results in search_results.items():
            if isinstance(results, dict) and "search_results" in results:
                # Web search results format
                for source in results["search_results"]:
                    citations.append({
                        "title": source.get("title", ""),
                        "url": source.get("url", ""),
                        "type": "web_source"
                    })
        
        return citations
        
    except Exception as e:
        logger.error(f"❌ Failed to extract cached citations: {e}")
        return []


def _format_cache_results(cache_analysis: Dict[str, Any]) -> str:
    """Format high-coverage cache results for agent use"""
    try:
        sources = cache_analysis["relevant_sources"]
        result_parts = [
            f"🏆 **CONVERSATION CACHE HIT** (Coverage: {cache_analysis['coverage_score']:.1%})",
            f"",
            f"**Found {len(sources)} relevant sources from this conversation:**",
            f""
        ]
        
        for i, source in enumerate(sources[:3], 1):  # Top 3 sources
            if source["type"] == "research":
                result_parts.append(f"**{i}. Research: {source['topic']}**")
                result_parts.append(f"Content: {source['content'].get('findings', '')[:300]}...")
            elif source["type"] == "chat":
                result_parts.append(f"**{i}. Previous Chat Work:**")
                result_parts.append(f"Content: {source['content'][:300]}...")
            result_parts.append("")
        
        result_parts.append("✅ **RECOMMENDATION**: Use this cached information instead of new searches.")
        
        return "\n".join(result_parts)
        
    except Exception as e:
        logger.error(f"❌ Failed to format cache results: {e}")
        return f"Cache formatting error: {str(e)}"


def _format_partial_cache_results(cache_analysis: Dict[str, Any]) -> str:
    """Format partial cache results with supplemental search recommendation"""
    try:
        sources = cache_analysis["relevant_sources"]
        result_parts = [
            f"🔄 **PARTIAL CACHE HIT** (Coverage: {cache_analysis['coverage_score']:.1%})",
            f"",
            f"**Found {len(sources)} partially relevant sources:**",
            f""
        ]
        
        for source in sources[:2]:  # Top 2 sources
            if source["type"] == "research":
                result_parts.append(f"- Research: {source['topic'][:100]}...")
            elif source["type"] == "chat":
                result_parts.append(f"- Chat Work: {source['content'][:100]}...")
        
        result_parts.extend([
            "",
            "📋 **RECOMMENDATION**: Use cached context + targeted supplemental search for missing details."
        ])
        
        return "\n".join(result_parts)
        
    except Exception as e:
        logger.error(f"❌ Failed to format partial cache results: {e}")
        return f"Partial cache formatting error: {str(e)}"


async def get_document_content(document_id: str, user_id: str = None) -> str:
    """LangGraph tool function: Get document content and metadata by ID"""
    search_instance = await _get_unified_search()
    result = await search_instance.get_document(document_id, user_id)
    
    # Format result as readable text for the LLM
    if not result.get("success"):
        error_msg = result.get("error", "Unknown error")
        return f"❌ Document retrieval failed: {error_msg}"
    
    content = result.get("content", "")
    metadata = result.get("metadata", {})
    
    if not content:
        return f"❌ No content found for document {document_id}"
    
    # Format as readable output for agents
    formatted_output = [
        f"📄 **Document Retrieved: {metadata.get('title', metadata.get('filename', document_id))}**",
        f"",
        f"**Metadata:**",
        f"- Document ID: {document_id}",
        f"- Filename: {metadata.get('filename', 'N/A')}",
        f"- Title: {metadata.get('title', 'N/A')}",
        f"- Author: {metadata.get('author', 'N/A')}",
        f"- Content Length: {metadata.get('content_length', 0)} characters",
        f"",
        f"**Content:**",
        f"{content}",
        f"",
        f"---",
        f"✅ Document content successfully retrieved and ready for analysis."
    ]
    
    return "\n".join(formatted_output)