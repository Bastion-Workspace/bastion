"""
gRPC Tool Service - Backend Data Access for LLM Orchestrator
Provides document, RSS, entity, weather, and org-mode data via gRPC
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import asyncio
import json
import uuid

import grpc
from protos import tool_service_pb2, tool_service_pb2_grpc

# Import repositories and services directly (safe - no circular dependencies)
from repositories.document_repository import DocumentRepository
from services.direct_search_service import DirectSearchService
from services.embedding_service_wrapper import get_embedding_service
from services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)


def _rss_article_to_pb(art: Any, feed_name_fallback: str = "") -> tool_service_pb2.RSSArticle:
    """Map tools_service RSSArticle (or compatible) to gRPC RSSArticle."""
    content = (getattr(art, "description", None) or "")[:5000]
    full = getattr(art, "full_content", None) or ""
    if full and len(full) > len(content):
        content = full[:5000]
    fn = feed_name_fallback or (getattr(art, "feed_name", None) or "")
    pub = getattr(art, "published_date", None)
    pd = pub.isoformat() if pub else ""
    cr = getattr(art, "created_at", None)
    ca = cr.isoformat() if cr else ""
    return tool_service_pb2.RSSArticle(
        article_id=getattr(art, "article_id", "") or "",
        title=getattr(art, "title", "") or "",
        content=content,
        url=getattr(art, "link", "") or "",
        published_at=pd,
        feed_id=getattr(art, "feed_id", "") or "",
        feed_name=fn,
        is_read=bool(getattr(art, "is_read", False)),
        is_starred=bool(getattr(art, "is_starred", False)),
        is_imported=bool(getattr(art, "is_processed", False)),
        created_at=ca,
    )


def _json_default(value: Any) -> str:
    """JSON serializer for non-primitive values returned from DB/service layers."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    return str(value)


class ToolServiceImplementation(tool_service_pb2_grpc.ToolServiceServicer):
    """
    gRPC Tool Service Implementation
    
    Provides data access methods for the LLM Orchestrator service.
    Uses repositories directly for Phase 2 (services via container in Phase 3).
    """
    
    def __init__(self):
        logger.info("Initializing gRPC Tool Service...")
        # Use direct search service for document operations
        self._search_service: Optional[DirectSearchService] = None
        self._document_repo: Optional[DocumentRepository] = None
        self._embedding_manager = None  # EmbeddingServiceWrapper
    
    async def _get_search_service(self) -> DirectSearchService:
        """Lazy initialization of search service"""
        if not self._search_service:
            self._search_service = DirectSearchService()
        return self._search_service
    
    async def _get_embedding_manager(self):
        """Lazy initialization of embedding service wrapper"""
        if not self._embedding_manager:
            self._embedding_manager = await get_embedding_service()
        return self._embedding_manager
    
    def _get_document_repo(self) -> DocumentRepository:
        """Lazy initialization of document repository"""
        if not self._document_repo:
            self._document_repo = DocumentRepository()
        return self._document_repo
    
    # ===== Document Operations =====
    
    async def SearchDocuments(
        self,
        request: tool_service_pb2.SearchRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.SearchResponse:
        """Search documents by query using direct search with optional tag/category filtering"""
        try:
            logger.info(f"SearchDocuments: user={request.user_id}, query={request.query[:100]}")
            
            # Parse filters for tags, categories, scope, folder_id, file_types, min_score, mode
            tags = []
            categories = []
            collection_scope = ""
            folder_id_filter = None
            file_types_filter = []
            min_score = 0.3
            search_mode = "hybrid"
            for filter_str in request.filters:
                if filter_str.startswith("tag:"):
                    tags.append(filter_str[4:])
                elif filter_str.startswith("category:"):
                    categories.append(filter_str[9:])
                elif filter_str.startswith("scope:"):
                    collection_scope = (filter_str[6:] or "").strip()
                elif filter_str.startswith("folder_id:"):
                    folder_id_filter = (filter_str[10:] or "").strip() or None
                elif filter_str.startswith("file_type:"):
                    ft = (filter_str[10:] or "").strip()
                    if ft and ft not in file_types_filter:
                        file_types_filter.append(ft)
                elif filter_str.startswith("min_score:"):
                    try:
                        min_score = float(filter_str[10:].strip())
                        min_score = max(0.0, min(1.0, min_score))
                    except (ValueError, TypeError):
                        pass
                elif filter_str.startswith("mode:"):
                    mode_val = (filter_str[5:] or "").strip().lower()
                    if mode_val in ("hybrid", "semantic", "fulltext"):
                        search_mode = mode_val

            if tags or categories:
                logger.info(f"SearchDocuments: Filtering by tags={tags}, categories={categories}")
            if collection_scope or folder_id_filter or file_types_filter:
                logger.info(f"SearchDocuments: scope={collection_scope}, folder_id={folder_id_filter}, file_types={file_types_filter}")

            # Get user's team IDs for hybrid search (user + team + global collections)
            team_ids = None
            user_id = request.user_id if request.user_id and request.user_id != "system" else None
            if user_id and collection_scope != "global_docs":
                try:
                    from services.team_service import TeamService
                    team_service = TeamService()
                    await team_service.initialize()
                    user_teams = await team_service.list_user_teams(user_id)
                    team_ids = [team['team_id'] for team in user_teams] if user_teams else None
                    if team_ids:
                        logger.info(f"SearchDocuments: User {user_id} is member of {len(team_ids)} teams - including team collections in search")
                except Exception as e:
                    logger.warning(f"SearchDocuments: Failed to get user teams for {user_id}: {e} - continuing without team collections")
                    team_ids = None

            if collection_scope == "my_docs":
                team_ids = []
            elif collection_scope == "global_docs":
                user_id = None
                team_ids = None

            # Get search service
            search_service = await self._get_search_service()

            # Perform direct search with optional tag/category/scope/folder/file_type filtering
            exclude_ids = list(request.exclude_document_ids) if request.exclude_document_ids else None
            search_result = await search_service.search_documents(
                query=request.query,
                limit=request.limit or 10,
                similarity_threshold=min_score,
                search_mode=search_mode,
                user_id=user_id,
                team_ids=team_ids,
                tags=tags if tags else None,
                categories=categories if categories else None,
                exclude_document_ids=exclude_ids,
                folder_id=folder_id_filter,
                file_types=file_types_filter if file_types_filter else None,
            )
            
            if not search_result.get("success"):
                logger.warning(f"SearchDocuments: Search failed - {search_result.get('error')}")
                return tool_service_pb2.SearchResponse(total_count=0)
            
            results = search_result.get("results", [])
            
            # Convert to proto response
            response = tool_service_pb2.SearchResponse(
                total_count=len(results)
            )
            
            for result in results:
                # DirectSearchService returns nested structure with document metadata
                document_metadata = result.get('document', {})

                # Get document_id from result directly (from vector search), fallback to metadata
                document_id = result.get('document_id') or document_metadata.get('document_id', '')

                doc_result = tool_service_pb2.DocumentResult(
                    document_id=str(document_id),
                    title=document_metadata.get('title', document_metadata.get('filename', '')),
                    filename=document_metadata.get('filename', ''),
                    content_preview=result.get('text', '')[:1500],  # Increased for better context
                    relevance_score=float(result.get('similarity_score', 0.0))
                )
                response.results.append(doc_result)
            
            logger.info(f"SearchDocuments: Found {len(results)} results")
            return response

        except Exception as e:
            logger.error(f"SearchDocuments error: {e}")
            import traceback
            traceback.print_exc()
            await context.abort(grpc.StatusCode.INTERNAL, f"Search failed: {str(e)}")

    async def FindDocumentsByTags(
        self,
        request: tool_service_pb2.FindDocumentsByTagsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.FindDocumentsByTagsResponse:
        """Find documents that contain ALL of the specified tags using database query"""
        try:
            logger.info(f"FindDocumentsByTags: user={request.user_id}, tags={list(request.required_tags)}, collection={request.collection_type}")

            # Debug the request
            logger.info(f"Request details: user_id={request.user_id}, required_tags={request.required_tags}, collection_type={request.collection_type}, limit={request.limit}")

            # Query database directly using the same approach that works in manual testing
            from services.database_manager.database_helpers import fetch_all

            query = """
                SELECT
                    document_id, filename, title, category, tags, description,
                    author, language, publication_date, doc_type, file_size,
                    file_hash, processing_status, upload_date, quality_score,
                    page_count, chunk_count, entity_count, user_id, collection_type,
                    folder_id
                FROM document_metadata
                WHERE tags @> $1
                ORDER BY upload_date DESC
                LIMIT $2
            """

            documents = await fetch_all(query, request.required_tags, request.limit or 20)

            logger.info(f"Found {len(documents)} documents matching tags")

            # Convert to proto response
            response = tool_service_pb2.FindDocumentsByTagsResponse(
                total_count=len(documents)
            )

            preview_extensions = ('.txt', '.md', '.markdown', '.org', '.csv', '.json', '.yaml', '.yml', '.log', '.rst')
            for doc in documents:
                content_preview = ""
                filename = doc.get('filename') or ''
                if filename and any(filename.lower().endswith(ext) for ext in preview_extensions):
                    try:
                        from pathlib import Path
                        from services.service_container import get_service_container
                        container = await get_service_container()
                        folder_service = container.folder_service
                        file_path_str = await folder_service.get_document_file_path(
                            filename=filename,
                            folder_id=doc.get('folder_id'),
                            user_id=doc.get('user_id'),
                            collection_type=doc.get('collection_type', 'user')
                        )
                        file_path = Path(file_path_str)
                        if file_path.exists():
                            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                content_preview = f.read(800)
                    except Exception as e:
                        logger.debug(f"FindDocumentsByTags: Could not load preview for {doc.get('document_id')}: {e}")

                doc_result = tool_service_pb2.DocumentResult(
                    document_id=str(doc.get('document_id', '')),
                    title=doc.get('title', doc.get('filename', '')),
                    filename=doc.get('filename', ''),
                    content_preview=content_preview,
                    relevance_score=1.0  # All matches are equally relevant
                )
                # Add metadata
                doc_result.metadata.update({
                    'tags': str(doc.get('tags', [])),
                    'category': doc.get('category', ''),
                    'user_id': doc.get('user_id', ''),
                    'collection_type': doc.get('collection_type', ''),
                    'doc_type': doc.get('doc_type', ''),
                })
                response.results.append(doc_result)

            logger.info(f"FindDocumentsByTags: Found {len(documents)} documents")
            return response

        except Exception as e:
            logger.error(f"FindDocumentsByTags error: {e}")
            import traceback
            traceback.print_exc()
            await context.abort(grpc.StatusCode.INTERNAL, f"Find by tags failed: {str(e)}")
    
    async def GetDocument(
        self,
        request: tool_service_pb2.DocumentRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DocumentResponse:
        """Get document metadata"""
        try:
            logger.info(f"GetDocument: doc_id={request.document_id}, user={request.user_id}")
            
            doc_repo = self._get_document_repo()
            # Pass user_id for RLS context
            doc = await doc_repo.get_document_by_id(document_id=request.document_id, user_id=request.user_id)
            
            if not doc:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Document not found")
            
            response = tool_service_pb2.DocumentResponse(
                document_id=str(doc.get('document_id', '')),
                title=doc.get('title', ''),
                filename=doc.get('filename', ''),
                content_type=doc.get('content_type', 'text/plain')
            )
            
            return response
            
        except (grpc.RpcError, grpc._cython.cygrpc.AbortError):
            # Re-raise gRPC errors (including AbortError from context.abort calls)
            raise
        except Exception as e:
            logger.error(f"GetDocument error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Get document failed: {str(e)}")
    
    async def GetDocumentContent(
        self,
        request: tool_service_pb2.DocumentRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DocumentContentResponse:
        """Get document full content from disk"""
        try:
            logger.info(f"GetDocumentContent: doc_id={request.document_id}, user_id={request.user_id}")
            
            doc_repo = self._get_document_repo()
            # Pass user_id for RLS context
            doc = await doc_repo.get_document_by_id(document_id=request.document_id, user_id=request.user_id)
            
            if not doc:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Document not found")
            
            # Get content from disk (same logic as REST API)
            filename = doc.get('filename')
            user_id = doc.get('user_id')
            folder_id = doc.get('folder_id')
            collection_type = doc.get('collection_type', 'user')
            
            logger.info(f"GetDocumentContent: filename={filename}, user_id={user_id}, folder_id={folder_id}, collection_type={collection_type}")
            
            full_content = None
            
            if filename:
                from pathlib import Path
                from services.service_container import get_service_container
                from utils.document_processor import DocumentProcessor
                
                container = await get_service_container()
                folder_service = container.folder_service
                
                # Skip pure binary files (images) - they don't have text content
                image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg']
                is_image_file = any(filename.lower().endswith(ext) for ext in image_extensions)
                
                if is_image_file:
                    logger.info(f"GetDocumentContent: Skipping image file content for {request.document_id} ({filename})")
                    full_content = ""
                else:
                    try:
                        logger.info(f"GetDocumentContent: Calling folder_service.get_document_file_path...")
                        file_path_str = await folder_service.get_document_file_path(
                            filename=filename,
                            folder_id=folder_id,
                            user_id=user_id,
                            collection_type=collection_type
                        )
                        logger.info(f"GetDocumentContent: Got file path: {file_path_str}")
                        file_path = Path(file_path_str)
                        
                        if file_path.exists():
                            # Detect file type and use appropriate processor
                            file_ext = file_path.suffix.lower()
                            
                            # Plain text files can be read directly
                            if file_ext in ['.txt', '.md', '.csv', '.json', '.yaml', '.yml', '.log']:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    full_content = f.read()
                                logger.info(f"GetDocumentContent: Loaded {len(full_content)} chars from plain text file {file_path}")
                            
                            # Binary document formats need special processing
                            elif file_ext in ['.docx', '.pptx', '.pdf', '.epub', '.html', '.htm', '.eml']:
                                logger.info(f"GetDocumentContent: Using DocumentProcessor for {file_ext} file")
                                doc_processor = DocumentProcessor()
                                
                                # Map extension to doc_type
                                doc_type_map = {
                                    '.docx': 'docx',
                                    '.pptx': 'pptx',
                                    '.pdf': 'pdf',
                                    '.epub': 'epub',
                                    '.html': 'html',
                                    '.htm': 'html',
                                    '.eml': 'eml'
                                }
                                doc_type = doc_type_map.get(file_ext, 'txt')
                                
                                # Extract text using appropriate processor method
                                if doc_type == 'docx':
                                    full_content = await doc_processor._process_docx(str(file_path))
                                elif doc_type == 'pdf':
                                    full_content, _, _, _ = await doc_processor._process_pdf(
                                        str(file_path), request.document_id
                                    )
                                elif doc_type == 'epub':
                                    full_content = await doc_processor._process_epub(str(file_path))
                                elif doc_type == 'html':
                                    full_content = await doc_processor._process_html(str(file_path))
                                elif doc_type == 'eml':
                                    full_content = await doc_processor._process_eml(str(file_path))
                                elif doc_type == 'pptx':
                                    full_content = await doc_processor._process_pptx(str(file_path))
                                
                                logger.info(f"GetDocumentContent: Extracted {len(full_content)} chars from {doc_type} file")
                            
                            else:
                                # Unknown format - try as plain text
                                logger.warning(f"GetDocumentContent: Unknown file type {file_ext}, attempting plain text read")
                                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                    full_content = f.read()
                        else:
                            logger.warning(f"GetDocumentContent: File not found at {file_path}")
                    except Exception as e:
                        logger.error(f"GetDocumentContent: Failed to load from folder service: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
            
            # If content is None, file wasn't found
            if full_content is None:
                logger.error(f"GetDocumentContent: File not found for document {request.document_id} (filename={filename}, folder_id={folder_id})")
                await context.abort(grpc.StatusCode.NOT_FOUND, f"Document file not found on disk")
            
            logger.info(f"GetDocumentContent: Returning content with {len(full_content)} characters")
            
            response = tool_service_pb2.DocumentContentResponse(
                document_id=str(doc.get('document_id', '')),
                content=full_content or '',
                format='text'
            )
            
            return response
            
        except (grpc.RpcError, grpc._cython.cygrpc.AbortError):
            # Re-raise gRPC errors (including AbortError from context.abort calls)
            # This prevents trying to abort twice
            raise
        except Exception as e:
            logger.error(f"GetDocumentContent error: {e}")
            import traceback
            traceback.print_exc()
            await context.abort(grpc.StatusCode.INTERNAL, f"Get content failed: {str(e)}")
    
    async def GetDocumentChunks(
        self,
        request: tool_service_pb2.DocumentRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DocumentChunksResponse:
        """Get chunks from a document using vector store"""
        try:
            logger.info(f"GetDocumentChunks: doc_id={request.document_id}")
            
            # Get vector store service
            from services.vector_store_service import get_vector_store
            vector_store = await get_vector_store()
            
            # Use Qdrant scroll to get all chunks for this document
            from qdrant_client.models import Filter, FieldCondition, MatchValue, ScrollRequest
            
            # Create filter for document_id
            document_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=request.document_id)
                    )
                ]
            )
            
            # Scroll through all collections to find chunks
            # Try global collection first
            from config import settings
            collections_to_search = [settings.VECTOR_COLLECTION_NAME]
            
            # Also try user collection if user_id is provided
            if request.user_id and request.user_id != "system":
                user_collection = f"user_{request.user_id}_documents"
                collections_to_search.append(user_collection)
            
            all_chunks = []
            
            # Use Vector Service search with filter to get chunks for this document
            # We'll search with a dummy query vector (all zeros) and rely on the filter
            # to match document_id, which is more efficient than scrolling
            dummy_vector = [0.0] * settings.EMBEDDING_DIMENSIONS
            
            for collection_name in collections_to_search:
                try:
                    # Search with document_id filter via Vector Service
                    filters = [
                        {
                            "field": "document_id",
                            "value": request.document_id,
                            "operator": "equals",
                        }
                    ]
                    
                    search_result = await vector_store.vector_service_client.search_vectors(
                        collection_name=collection_name,
                        query_vector=dummy_vector,
                        limit=1000,  # Get up to 1000 chunks per document
                        score_threshold=0.0,  # No threshold, we want all matches
                        filters=filters,
                    )
                    
                    # search_vectors returns a list of dicts with id, score, payload
                    points = search_result if isinstance(search_result, list) else []
                    for point in points:
                        payload = point.get("payload", {})
                        chunk_data = {
                            'chunk_id': payload.get('chunk_id', point.get('id', '')),
                            'document_id': payload.get('document_id', request.document_id),
                            'content': payload.get('content', ''),
                            'chunk_index': payload.get('chunk_index', 0),
                            'metadata': payload.get('metadata', {})
                        }
                        all_chunks.append(chunk_data)
                    
                    logger.debug(f"Found {len(points)} chunks in {collection_name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to search collection {collection_name}: {e}")
                    continue
            
            if not all_chunks:
                logger.warning(f"No chunks found for document {request.document_id}")
                return tool_service_pb2.DocumentChunksResponse(
                    document_id=request.document_id,
                    chunks=[]
                )
            
            # Sort by chunk_index
            all_chunks.sort(key=lambda x: x.get('chunk_index', 0))
            
            # Convert to proto response
            chunks_proto = []
            for chunk_data in all_chunks:
                chunk_proto = tool_service_pb2.DocumentChunk(
                    chunk_id=chunk_data.get('chunk_id', ''),
                    document_id=chunk_data.get('document_id', request.document_id),
                    content=chunk_data.get('content', ''),
                    chunk_index=chunk_data.get('chunk_index', 0),
                    metadata=json.dumps(chunk_data.get('metadata', {}))
                )
                chunks_proto.append(chunk_proto)
            
            logger.info(f"GetDocumentChunks: Found {len(chunks_proto)} chunks")
            return tool_service_pb2.DocumentChunksResponse(
                document_id=request.document_id,
                chunks=chunks_proto
            )
            
        except Exception as e:
            logger.error(f"GetDocumentChunks error: {e}")
            import traceback
            traceback.print_exc()
            await context.abort(grpc.StatusCode.INTERNAL, f"Get chunks failed: {str(e)}")
    
    async def FindDocumentByPath(
        self,
        request: tool_service_pb2.FindDocumentByPathRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.FindDocumentByPathResponse:
        """
        Find a document by filesystem path (true path resolution).
        
        Resolves relative paths from base_path, then finds the document record
        by matching the actual filesystem path.
        """
        try:
            from pathlib import Path
            from config import settings
            
            logger.info(f"FindDocumentByPath: user={request.user_id}, path={request.file_path}, base={request.base_path}")
            
            # Resolve the path
            file_path_str = request.file_path.strip()
            base_path_str = request.base_path.strip() if request.base_path else None
            
            # If relative path, resolve from base
            if base_path_str and not Path(file_path_str).is_absolute():
                base_path = Path(base_path_str)
                resolved_path = (base_path / file_path_str).resolve()
            else:
                resolved_path = Path(file_path_str).resolve()
            
            # Ensure .md extension if no extension
            if not resolved_path.suffix:
                resolved_path = resolved_path.with_suffix('.md')
            
            logger.info(f"FindDocumentByPath: Resolved to {resolved_path}")
            
            # Check if file exists
            if not resolved_path.exists() or not resolved_path.is_file():
                logger.warning(f"FindDocumentByPath: File not found at {resolved_path}")
                return tool_service_pb2.FindDocumentByPathResponse(
                    success=False,
                    error=f"File not found at {resolved_path}"
                )
            
            # Find document record by path using repository
            # Replicate logic from DocumentFileHandler._get_document_by_path
            from pathlib import Path as PathLib
            
            path = PathLib(resolved_path)
            filename = path.name
            
            # Parse the path to extract user context
            parts = path.parts
            uploads_idx = -1
            for i, part in enumerate(parts):
                if part == 'uploads':
                    uploads_idx = i
                    break
            
            if uploads_idx == -1:
                logger.warning(f"FindDocumentByPath: File path doesn't contain 'uploads': {resolved_path}")
                return tool_service_pb2.FindDocumentByPathResponse(
                    success=False,
                    error=f"Invalid path structure: {resolved_path}"
                )
            
            # Determine collection type and context
            doc_repo = self._get_document_repo()
            user_id = request.user_id
            collection_type = 'user'
            folder_id = None
            
            if uploads_idx + 1 < len(parts):
                collection_dir = parts[uploads_idx + 1]
                
                if collection_dir == 'Users' and uploads_idx + 2 < len(parts):
                    # User file: uploads/Users/{username}/{folders...}/{filename}
                    username = parts[uploads_idx + 2]
                    collection_type = 'user'
                    
                    # Get user_id from username if not provided
                    if not user_id:
                        from repositories.document_repository import DocumentRepository
                        temp_repo = DocumentRepository()
                        import asyncpg
                        from config import settings
                        conn = await asyncpg.connect(settings.DATABASE_URL)
                        try:
                            row = await conn.fetchrow("SELECT user_id FROM users WHERE username = $1", username)
                            if row:
                                user_id = row['user_id']
                        finally:
                            await conn.close()
                    
                    # Resolve folder hierarchy if folders exist
                    folder_start_idx = uploads_idx + 3
                    folder_end_idx = len(parts) - 1  # Exclude filename
                    
                    if folder_start_idx < folder_end_idx:
                        folder_parts = parts[folder_start_idx:folder_end_idx]
                        logger.info(f"📁 DB QUERY: Resolving folder hierarchy: {folder_parts}")
                        # Get folders and resolve hierarchy
                        folders_data = await doc_repo.get_folders_by_user(user_id, collection_type)
                        logger.info(f"📁 DB QUERY: Found {len(folders_data)} total folders for user")
                        folder_map = {(f.get('name'), f.get('parent_folder_id')): f.get('folder_id') for f in folders_data}
                        logger.info(f"📁 DB QUERY: Folder map keys: {list(folder_map.keys())[:10]}...")  # Show first 10
                        
                        parent_folder_id = None
                        for i, folder_name in enumerate(folder_parts):
                            key = (folder_name, parent_folder_id)
                            logger.info(f"📁 DB QUERY: Step {i+1}: Looking for folder '{folder_name}' with parent={parent_folder_id}")
                            if key in folder_map:
                                folder_id = folder_map[key]
                                parent_folder_id = folder_id
                                logger.info(f"✅ DB QUERY: Found folder_id={folder_id} for '{folder_name}'")
                            else:
                                logger.warning(f"❌ DB QUERY: Folder '{folder_name}' with parent={parent_folder_id} NOT FOUND in folder_map!")
                                logger.warning(f"   Available folders with parent={parent_folder_id}: {[k[0] for k in folder_map.keys() if k[1] == parent_folder_id]}")
                                folder_id = None
                                break
                
                elif collection_dir == 'Global':
                    # Global file: uploads/Global/{folders...}/{filename}
                    collection_type = 'global'
                    user_id = None
                    
                    # Resolve folder hierarchy if folders exist
                    folder_start_idx = uploads_idx + 2
                    folder_end_idx = len(parts) - 1  # Exclude filename
                    
                    if folder_start_idx < folder_end_idx:
                        folder_parts = parts[folder_start_idx:folder_end_idx]
                        # Get folders and resolve hierarchy
                        folders_data = await doc_repo.get_folders_by_user(None, collection_type)
                        folder_map = {(f.get('name'), f.get('parent_folder_id')): f.get('folder_id') for f in folders_data}
                        
                        parent_folder_id = None
                        for folder_name in folder_parts:
                            key = (folder_name, parent_folder_id)
                            if key in folder_map:
                                folder_id = folder_map[key]
                                parent_folder_id = folder_id
                            else:
                                folder_id = None
                                break
            
            # Find document by filename, user_id, and folder_id
            logger.info(f"📄 DB QUERY: Searching for filename='{filename}', user_id={user_id}, collection_type={collection_type}, folder_id={folder_id}")
            document = await doc_repo.find_by_filename_and_context(
                filename=filename,
                user_id=user_id,
                collection_type=collection_type,
                folder_id=folder_id
            )
            
            if not document:
                logger.warning(f"❌ DB QUERY: NO MATCH FOUND in database")
                logger.warning(f"   Searched for: filename='{filename}', folder_id={folder_id}, user_id={user_id}")
                logger.warning(f"   Resolved path: {resolved_path}")
                logger.warning(f"FindDocumentByPath: No document record found for {resolved_path}")
                return tool_service_pb2.FindDocumentByPathResponse(
                    success=False,
                    error=f"No document record found for {resolved_path}"
                )
            
            document_id = document.document_id
            filename = document.filename
            
            logger.info(f"FindDocumentByPath: Found document {document_id} at {resolved_path}")
            
            return tool_service_pb2.FindDocumentByPathResponse(
                success=True,
                document_id=document_id,
                filename=filename,
                resolved_path=str(resolved_path)
            )
            
        except Exception as e:
            logger.error(f"FindDocumentByPath error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return tool_service_pb2.FindDocumentByPathResponse(
                success=False,
                error=str(e)
            )
    
    # ===== RSS Operations =====
    
    async def SearchRSSFeeds(
        self,
        request: tool_service_pb2.RSSSearchRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.RSSSearchResponse:
        """Search RSS feeds and articles by query (title, description, content)."""
        try:
            logger.info(f"SearchRSSFeeds: user={request.user_id}, query={request.query[:80]}")
            from tools_service.services.rss_service import get_rss_service

            rss_service = await get_rss_service()
            limit = request.limit or 20
            articles = await rss_service.search_articles(
                user_id=request.user_id or "system",
                query=request.query or "",
                limit=limit,
                unread_only=bool(request.unread_only),
                starred_only=bool(request.starred_only),
            )
            response = tool_service_pb2.RSSSearchResponse()
            for art in articles:
                response.articles.append(_rss_article_to_pb(art))
            logger.info(f"SearchRSSFeeds: Found {len(response.articles)} articles")
            return response
        except Exception as e:
            logger.error(f"SearchRSSFeeds error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"RSS search failed: {str(e)}")

    async def GetRSSArticles(
        self,
        request: tool_service_pb2.RSSArticlesRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.RSSArticlesResponse:
        """Get articles from a specific RSS feed."""
        try:
            logger.info(f"GetRSSArticles: feed_id={request.feed_id}")
            from tools_service.services.rss_service import get_rss_service

            rss_service = await get_rss_service()
            feed = await rss_service.get_feed(request.feed_id)
            if not feed:
                return tool_service_pb2.RSSArticlesResponse()
            if feed.user_id is not None and feed.user_id != request.user_id:
                return tool_service_pb2.RSSArticlesResponse()
            limit = request.limit or 20
            uid = request.user_id or "system"
            unread_only = bool(request.unread_only)
            starred_only = bool(request.starred_only)
            if unread_only or starred_only:
                articles = await rss_service.get_feed_articles_filtered(
                    feed_id=request.feed_id,
                    user_id=uid,
                    limit=limit,
                    unread_only=unread_only,
                    starred_only=starred_only,
                )
            else:
                articles = await rss_service.get_feed_articles(
                    feed_id=request.feed_id,
                    user_id=uid,
                    limit=limit,
                )
            feed_name = feed.feed_name or ""
            response = tool_service_pb2.RSSArticlesResponse()
            for art in articles:
                response.articles.append(_rss_article_to_pb(art, feed_name_fallback=feed_name))
            logger.info(f"GetRSSArticles: Returned {len(response.articles)} articles")
            return response
        except Exception as e:
            logger.error(f"GetRSSArticles error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Get articles failed: {str(e)}")

    async def ListStarredRSSArticles(
        self,
        request: tool_service_pb2.ListStarredRSSArticlesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListStarredRSSArticlesResponse:
        """List starred RSS articles for the user across all feeds."""
        try:
            uid = (request.user_id or "").strip()
            if not uid:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "user_id is required"
                )
            from tools_service.services.rss_service import get_rss_service

            rss_service = await get_rss_service()
            limit = int(request.limit) if request.limit else 50
            limit = max(1, min(limit, 500))
            offset = max(0, int(request.offset))
            articles = await rss_service.get_starred_articles(
                user_id=uid, limit=limit, offset=offset
            )
            response = tool_service_pb2.ListStarredRSSArticlesResponse()
            for art in articles:
                response.articles.append(_rss_article_to_pb(art))
            logger.info(
                "ListStarredRSSArticles: user=%s limit=%s offset=%s count=%s",
                uid,
                limit,
                offset,
                len(response.articles),
            )
            return response
        except (grpc.RpcError, grpc._cython.cygrpc.AbortError):
            raise
        except Exception as e:
            logger.error("ListStarredRSSArticles error: %s", e)
            await context.abort(
                grpc.StatusCode.INTERNAL, f"List starred articles failed: {str(e)}"
            )
    
    # ===== RSS Management Operations =====
    
    async def AddRSSFeed(
        self,
        request: tool_service_pb2.AddRSSFeedRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.AddRSSFeedResponse:
        """Add a new RSS feed"""
        try:
            logger.info(f"AddRSSFeed: user={request.user_id}, url={request.feed_url}, is_global={request.is_global}")
            
            from services.auth_service import auth_service
            from tools_service.models.rss_models import RSSFeedCreate
            from tools_service.services.rss_service import get_rss_service
            
            rss_service = await get_rss_service()
            
            # Check permissions for global feeds
            if request.is_global:
                user_info = await auth_service.get_user_by_id(request.user_id)
                if not user_info or user_info.role != "admin":
                    return tool_service_pb2.AddRSSFeedResponse(
                        success=False,
                        error="Only admin users can add global RSS feeds"
                    )
            
            # Create RSS feed data
            feed_data = RSSFeedCreate(
                feed_url=request.feed_url,
                feed_name=request.feed_name,
                user_id=request.user_id if not request.is_global else None,  # None for global
                category=request.category or "general",
                tags=["rss", "imported"],
                check_interval=3600  # Default 1 hour
            )
            
            # Add the feed
            new_feed = await rss_service.create_feed(feed_data)
            
            logger.info(f"AddRSSFeed: Successfully added feed {new_feed.feed_id}")
            
            return tool_service_pb2.AddRSSFeedResponse(
                success=True,
                feed_id=new_feed.feed_id,
                feed_name=new_feed.feed_name,
                message=f"Successfully added {'global' if request.is_global else 'user'} RSS feed: {new_feed.feed_name}"
            )
            
        except Exception as e:
            logger.error(f"AddRSSFeed error: {e}")
            return tool_service_pb2.AddRSSFeedResponse(
                success=False,
                error=f"Failed to add RSS feed: {str(e)}"
            )
    
    async def ListRSSFeeds(
        self,
        request: tool_service_pb2.ListRSSFeedsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ListRSSFeedsResponse:
        """List RSS feeds"""
        try:
            logger.info(f"ListRSSFeeds: user={request.user_id}, scope={request.scope}")
            
            from services.auth_service import auth_service
            from tools_service.services.rss_service import get_rss_service
            
            rss_service = await get_rss_service()
            
            # Determine if user is admin for global feed access
            is_admin = False
            if request.scope == "global":
                user_info = await auth_service.get_user_by_id(request.user_id)
                is_admin = user_info and user_info.role == "admin"
            
            # Get feeds based on scope
            feeds = await rss_service.get_user_feeds(request.user_id, is_admin=is_admin)
            
            # Convert to proto response
            response = tool_service_pb2.ListRSSFeedsResponse(
                success=True,
                count=len(feeds)
            )
            counts_map = await rss_service.get_unread_count(request.user_id or "system")

            for feed in feeds:
                # Get article count for this feed
                from services.database_manager.database_helpers import fetch_value
                try:
                    article_count = await fetch_value(
                        "SELECT COUNT(*) FROM rss_articles WHERE feed_id = $1",
                        feed.feed_id
                    ) or 0
                except Exception:
                    article_count = 0
                last_chk = getattr(feed, "last_check", None) or getattr(
                    feed, "last_poll_date", None
                )
                last_polled_s = last_chk.isoformat() if last_chk else ""

                feed_details = tool_service_pb2.RSSFeedDetails(
                    feed_id=feed.feed_id,
                    feed_name=feed.feed_name,
                    feed_url=feed.feed_url,
                    category=feed.category or "general",
                    is_global=(feed.user_id is None),
                    last_polled=last_polled_s,
                    article_count=int(article_count),
                    unread_count=int(counts_map.get(feed.feed_id, 0)),
                )
                response.feeds.append(feed_details)
            
            logger.info(f"ListRSSFeeds: Found {len(feeds)} feeds")
            return response
            
        except Exception as e:
            logger.error(f"ListRSSFeeds error: {e}")
            return tool_service_pb2.ListRSSFeedsResponse(
                success=False,
                error=f"Failed to list RSS feeds: {str(e)}"
            )
    
    async def RefreshRSSFeed(
        self,
        request: tool_service_pb2.RefreshRSSFeedRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.RefreshRSSFeedResponse:
        """Refresh a specific RSS feed"""
        try:
            logger.info(f"RefreshRSSFeed: user={request.user_id}, feed_name={request.feed_name}, feed_id={request.feed_id}")
            
            from services.celery_tasks.rss_tasks import poll_rss_feeds_task
            from tools_service.services.rss_service import get_rss_service
            
            rss_service = await get_rss_service()
            
            # Find the feed by ID or name
            target_feed = None
            if request.feed_id:
                target_feed = await rss_service.get_feed(request.feed_id)
            else:
                # Find by name
                feeds = await rss_service.get_user_feeds(request.user_id, is_admin=True)
                for feed in feeds:
                    if feed.feed_name.lower() == request.feed_name.lower():
                        target_feed = feed
                        break
            
            if not target_feed:
                return tool_service_pb2.RefreshRSSFeedResponse(
                    success=False,
                    error=f"RSS feed '{request.feed_name or request.feed_id}' not found"
                )
            
            # Trigger refresh via Celery
            task = poll_rss_feeds_task.delay(
                user_id=request.user_id,
                feed_ids=[target_feed.feed_id],
                force_poll=True
            )
            
            logger.info(f"RefreshRSSFeed: Triggered refresh task {task.id} for feed {target_feed.feed_id}")
            
            return tool_service_pb2.RefreshRSSFeedResponse(
                success=True,
                feed_id=target_feed.feed_id,
                feed_name=target_feed.feed_name,
                task_id=task.id,
                message=f"Refresh initiated for RSS feed: {target_feed.feed_name}"
            )
            
        except Exception as e:
            logger.error(f"RefreshRSSFeed error: {e}")
            return tool_service_pb2.RefreshRSSFeedResponse(
                success=False,
                error=f"Failed to refresh RSS feed: {str(e)}"
            )
    
    async def DeleteRSSFeed(
        self,
        request: tool_service_pb2.DeleteRSSFeedRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DeleteRSSFeedResponse:
        """Delete an RSS feed"""
        try:
            logger.info(f"DeleteRSSFeed: user={request.user_id}, feed_name={request.feed_name}, feed_id={request.feed_id}")
            
            from tools_service.services.rss_service import get_rss_service
            
            rss_service = await get_rss_service()
            
            # Find the feed by ID or name
            target_feed = None
            if request.feed_id:
                target_feed = await rss_service.get_feed(request.feed_id)
            else:
                # Find by name
                feeds = await rss_service.get_user_feeds(request.user_id, is_admin=True)
                for feed in feeds:
                    if feed.feed_name.lower() == request.feed_name.lower():
                        target_feed = feed
                        break
            
            if not target_feed:
                return tool_service_pb2.DeleteRSSFeedResponse(
                    success=False,
                    error=f"RSS feed '{request.feed_name or request.feed_id}' not found"
                )
            
            # Check permission - only feed owner or admin can delete
            # For now, we trust the user_id passed from orchestrator
            
            # Delete the feed
            await rss_service.delete_feed(target_feed.feed_id, request.user_id, is_admin=False)
            
            logger.info(f"DeleteRSSFeed: Successfully deleted feed {target_feed.feed_id}")
            
            return tool_service_pb2.DeleteRSSFeedResponse(
                success=True,
                feed_id=target_feed.feed_id,
                message=f"Successfully deleted RSS feed: {target_feed.feed_name}"
            )
            
        except Exception as e:
            logger.error(f"DeleteRSSFeed error: {e}")
            return tool_service_pb2.DeleteRSSFeedResponse(
                success=False,
                error=f"Failed to delete RSS feed: {str(e)}"
            )

    async def MarkArticleRead(
        self,
        request: tool_service_pb2.MarkArticleReadRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.MarkArticleReadResponse:
        """Mark an RSS article as read for the requesting user."""
        try:
            uid = request.user_id or "system"
            aid = (request.article_id or "").strip()
            if not aid:
                return tool_service_pb2.MarkArticleReadResponse(
                    success=False,
                    message="",
                    error="article_id is required",
                )
            from tools_service.services.rss_service import get_rss_service

            rss_service = await get_rss_service()
            ok = await rss_service.mark_article_read(aid, uid)
            if not ok:
                return tool_service_pb2.MarkArticleReadResponse(
                    success=False,
                    message="",
                    error="Failed to mark article read",
                )
            return tool_service_pb2.MarkArticleReadResponse(
                success=True,
                message="Article marked as read",
            )
        except Exception as e:
            logger.error("MarkArticleRead error: %s", e)
            return tool_service_pb2.MarkArticleReadResponse(
                success=False,
                message="",
                error=str(e),
            )

    async def MarkArticleUnread(
        self,
        request: tool_service_pb2.MarkArticleUnreadRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.MarkArticleUnreadResponse:
        """Mark an RSS article as unread for the requesting user."""
        try:
            uid = request.user_id or "system"
            aid = (request.article_id or "").strip()
            if not aid:
                return tool_service_pb2.MarkArticleUnreadResponse(
                    success=False,
                    message="",
                    error="article_id is required",
                )
            from tools_service.services.rss_service import get_rss_service

            rss_service = await get_rss_service()
            ok = await rss_service.mark_article_unread(aid, uid)
            if not ok:
                return tool_service_pb2.MarkArticleUnreadResponse(
                    success=False,
                    message="",
                    error="Failed to mark article unread",
                )
            return tool_service_pb2.MarkArticleUnreadResponse(
                success=True,
                message="Article marked as unread",
            )
        except Exception as e:
            logger.error("MarkArticleUnread error: %s", e)
            return tool_service_pb2.MarkArticleUnreadResponse(
                success=False,
                message="",
                error=str(e),
            )

    async def SetArticleStarred(
        self,
        request: tool_service_pb2.SetArticleStarredRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SetArticleStarredResponse:
        """Set RSS article starred flag for the requesting user."""
        try:
            uid = request.user_id or "system"
            aid = (request.article_id or "").strip()
            if not aid:
                return tool_service_pb2.SetArticleStarredResponse(
                    success=False,
                    message="",
                    error="article_id is required",
                )
            from tools_service.services.rss_service import get_rss_service

            rss_service = await get_rss_service()
            ok = await rss_service.set_article_starred(aid, uid, request.starred)
            if not ok:
                return tool_service_pb2.SetArticleStarredResponse(
                    success=False,
                    message="",
                    error="Failed to update starred state",
                )
            state = "starred" if request.starred else "unstarred"
            return tool_service_pb2.SetArticleStarredResponse(
                success=True,
                message=f"Article {state}",
            )
        except Exception as e:
            logger.error("SetArticleStarred error: %s", e)
            return tool_service_pb2.SetArticleStarredResponse(
                success=False,
                message="",
                error=str(e),
            )

    async def GetUnreadCounts(
        self,
        request: tool_service_pb2.GetUnreadCountsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.GetUnreadCountsResponse:
        """Per-feed unread article counts for the user."""
        try:
            uid = request.user_id or "system"
            from tools_service.services.rss_service import get_rss_service

            rss_service = await get_rss_service()
            counts_map = await rss_service.get_unread_count(uid)
            response = tool_service_pb2.GetUnreadCountsResponse(success=True)
            for feed_id, cnt in (counts_map or {}).items():
                response.counts.append(
                    tool_service_pb2.UnreadCountEntry(
                        feed_id=feed_id,
                        count=int(cnt),
                    )
                )
            return response
        except Exception as e:
            logger.error("GetUnreadCounts error: %s", e)
            return tool_service_pb2.GetUnreadCountsResponse(
                success=False,
                error=str(e),
            )

    async def ToggleFeedActive(
        self,
        request: tool_service_pb2.ToggleFeedActiveRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ToggleFeedActiveResponse:
        """Enable or disable polling for an RSS feed."""
        try:
            uid = request.user_id or "system"
            fid = (request.feed_id or "").strip()
            if not fid:
                return tool_service_pb2.ToggleFeedActiveResponse(
                    success=False,
                    feed_id="",
                    is_active=request.is_active,
                    message="",
                    error="feed_id is required",
                )
            from services.auth_service import auth_service
            from tools_service.services.rss_service import get_rss_service

            rss_service = await get_rss_service()
            user_info = await auth_service.get_user_by_id(uid)
            is_admin = bool(user_info and user_info.role == "admin")
            ok = await rss_service.toggle_feed_active(
                fid, uid, request.is_active, is_admin=is_admin
            )
            if not ok:
                return tool_service_pb2.ToggleFeedActiveResponse(
                    success=False,
                    feed_id=fid,
                    is_active=request.is_active,
                    message="",
                    error="Not allowed or feed not found",
                )
            return tool_service_pb2.ToggleFeedActiveResponse(
                success=True,
                feed_id=fid,
                is_active=request.is_active,
                message="Feed active state updated",
            )
        except Exception as e:
            logger.error("ToggleFeedActive error: %s", e)
            return tool_service_pb2.ToggleFeedActiveResponse(
                success=False,
                feed_id=request.feed_id or "",
                is_active=request.is_active,
                message="",
                error=str(e),
            )

    # ===== Entity Operations =====
    
    async def SearchEntities(
        self,
        request: tool_service_pb2.EntitySearchRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.EntitySearchResponse:
        """Search entities"""
        try:
            logger.info(f"SearchEntities: query={request.query}")
            
            # Placeholder implementation
            response = tool_service_pb2.EntitySearchResponse()
            return response
            
        except Exception as e:
            logger.error(f"SearchEntities error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Entity search failed: {str(e)}")
    
    async def GetEntity(
        self,
        request: tool_service_pb2.EntityRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.EntityResponse:
        """Get entity details"""
        try:
            logger.info(f"GetEntity: entity_id={request.entity_id}")
            
            # Placeholder implementation
            entity = tool_service_pb2.Entity(
                entity_id=request.entity_id,
                entity_type="unknown",
                name="Placeholder"
            )
            response = tool_service_pb2.EntityResponse(entity=entity)
            return response
            
        except Exception as e:
            logger.error(f"GetEntity error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Get entity failed: {str(e)}")
    
    async def FindDocumentsByEntities(
        self,
        request: tool_service_pb2.FindDocumentsByEntitiesRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.FindDocumentsByEntitiesResponse:
        """Find documents mentioning specific entities"""
        try:
            logger.info(f"FindDocumentsByEntities: user={request.user_id}, entities={list(request.entity_names)}")
            
            # Get knowledge graph service
            from services.knowledge_graph_service import KnowledgeGraphService
            kg_service = KnowledgeGraphService()
            await kg_service.initialize()
            
            # Query knowledge graph (RLS enforced at document retrieval)
            document_ids = await kg_service.find_documents_by_entities(
                list(request.entity_names)
            )
            
            # Filter by user permissions (RLS check)
            doc_repo = self._get_document_repo()
            accessible_doc_ids = []
            for doc_id in document_ids:
                doc = await doc_repo.get_document_by_id(document_id=doc_id, user_id=request.user_id)
                if doc:  # User has access
                    accessible_doc_ids.append(doc_id)
            
            logger.info(f"Found {len(accessible_doc_ids)} accessible documents (filtered from {len(document_ids)} total)")
            
            return tool_service_pb2.FindDocumentsByEntitiesResponse(
                document_ids=accessible_doc_ids,
                total_count=len(accessible_doc_ids)
            )
            
        except Exception as e:
            logger.error(f"FindDocumentsByEntities failed: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Entity search failed: {str(e)}")

    async def FindRelatedDocumentsByEntities(
        self,
        request: tool_service_pb2.FindRelatedDocumentsByEntitiesRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.FindRelatedDocumentsByEntitiesResponse:
        """Find documents via related entities (1-2 hop traversal)"""
        try:
            logger.info(f"FindRelatedDocumentsByEntities: user={request.user_id}, entities={list(request.entity_names)}, hops={request.max_hops}")
            
            from services.knowledge_graph_service import KnowledgeGraphService
            kg_service = KnowledgeGraphService()
            await kg_service.initialize()
            
            # Query with relationship traversal
            document_ids = await kg_service.find_related_documents_by_entities(
                list(request.entity_names),
                max_hops=request.max_hops or 2
            )
            
            # RLS filtering
            doc_repo = self._get_document_repo()
            accessible_doc_ids = []
            for doc_id in document_ids:
                doc = await doc_repo.get_document_by_id(document_id=doc_id, user_id=request.user_id)
                if doc:
                    accessible_doc_ids.append(doc_id)
            
            logger.info(f"Found {len(accessible_doc_ids)} related documents accessible to user")
            
            return tool_service_pb2.FindRelatedDocumentsByEntitiesResponse(
                document_ids=accessible_doc_ids,
                total_count=len(accessible_doc_ids)
            )
            
        except Exception as e:
            logger.error(f"FindRelatedDocumentsByEntities failed: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Related entity search failed: {str(e)}")

    async def FindCoOccurringEntities(
        self,
        request: tool_service_pb2.FindCoOccurringEntitiesRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.FindCoOccurringEntitiesResponse:
        """Find entities that co-occur with given entities"""
        try:
            logger.info(f"FindCoOccurringEntities: entities={list(request.entity_names)}")
            
            from services.knowledge_graph_service import KnowledgeGraphService
            kg_service = KnowledgeGraphService()
            await kg_service.initialize()
            
            co_occurring = await kg_service.find_co_occurring_entities(
                list(request.entity_names),
                min_co_occurrences=request.min_co_occurrences or 2
            )
            
            # Convert to proto
            entities = []
            for entity in co_occurring:
                entities.append(tool_service_pb2.EntityInfo(
                    name=entity["name"],
                    type=entity["type"],
                    co_occurrence_count=entity["co_occurrence_count"]
                ))
            
            return tool_service_pb2.FindCoOccurringEntitiesResponse(entities=entities)
            
        except Exception as e:
            logger.error(f"FindCoOccurringEntities failed: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Co-occurrence search failed: {str(e)}")
    
    # ===== Weather Operations =====
    
    async def GetWeatherData(
        self,
        request: tool_service_pb2.WeatherRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.WeatherResponse:
        """Get weather data (current, forecast, or historical)"""
        aborted = False  # Track if we've already aborted to avoid double-abort
        try:
            logger.info(f"GetWeatherData: location={request.location}, user_id={request.user_id}, date_str={request.date_str if request.HasField('date_str') else None}")
            
            # Normalize location: empty string or whitespace-only → None
            location = request.location.strip() if request.location and request.location.strip() else None
            
            # Determine units from request (default to imperial)
            units = "imperial"  # Default for status bar compatibility
            
            # Check if this is a historical request
            if request.HasField("date_str") and request.date_str:
                # Historical weather request
                # Import from tools-service (same pattern as RSS service)
                from tools_service.services.weather_tools import weather_history
                
                weather_result = await weather_history(
                    location=location,
                    date_str=request.date_str,
                    units=units,
                    user_id=request.user_id if request.user_id else None
                )
                
                if not weather_result.get("success"):
                    error_msg = weather_result.get("error", "Unknown error")
                    logger.warning(f"Historical weather fetch failed: {error_msg}")
                    aborted = True
                    await context.abort(grpc.StatusCode.INTERNAL, f"Historical weather data failed: {error_msg}")
                    return  # This line won't be reached, but included for clarity
                
                # Extract historical weather data
                location_name = weather_result.get("location", {}).get("name", location or "Unknown location")
                historical = weather_result.get("historical", {})
                period = weather_result.get("period", {})
                
                # Format historical data for response
                period_type = period.get("type", "")
                if period_type == "date_range":
                    avg_temp = historical.get("average_temperature", 0)
                    temp_unit = weather_result.get("units", {}).get("temperature", "°F")
                    months_retrieved = period.get("months_retrieved", 0)
                    months_in_range = period.get("months_in_range", 0)
                    current_conditions = f"Range average ({months_retrieved}/{months_in_range} months): {avg_temp:.1f}{temp_unit}"
                elif period_type == "monthly_average":
                    avg_temp = historical.get("average_temperature", 0)
                    temp_unit = weather_result.get("units", {}).get("temperature", "°F")
                    current_conditions = f"Monthly average: {avg_temp:.1f}{temp_unit}"
                else:
                    temp = historical.get("temperature", 0)
                    temp_unit = weather_result.get("units", {}).get("temperature", "°F")
                    conditions = historical.get("conditions", "")
                    current_conditions = f"{temp:.1f}{temp_unit}, {conditions}"
                
                # Build metadata with historical information
                metadata = {
                    "location_name": location_name,
                    "date_str": request.date_str,
                    "period_type": period.get("type", ""),
                    "temperature": str(historical.get("temperature", historical.get("average_temperature", 0))),
                    "conditions": historical.get("conditions", historical.get("most_common_conditions", "")),
                    "humidity": str(historical.get("humidity", historical.get("average_humidity", 0))),
                    "wind_speed": str(historical.get("wind_speed", historical.get("average_wind_speed", 0)))
                }
                
                # Add period-specific fields
                if period_type == "date_range":
                    metadata["average_temperature"] = str(historical.get("average_temperature", 0))
                    metadata["min_temperature"] = str(historical.get("min_temperature", 0))
                    metadata["max_temperature"] = str(historical.get("max_temperature", 0))
                    metadata["months_retrieved"] = str(period.get("months_retrieved", 0))
                    metadata["months_in_range"] = str(period.get("months_in_range", 0))
                    metadata["start_date"] = period.get("start_date", "")
                    metadata["end_date"] = period.get("end_date", "")
                elif period_type == "monthly_average":
                    metadata["average_temperature"] = str(historical.get("average_temperature", 0))
                    metadata["min_temperature"] = str(historical.get("min_temperature", 0))
                    metadata["max_temperature"] = str(historical.get("max_temperature", 0))
                    metadata["sample_days"] = str(historical.get("sample_days", 0))
                
                response = tool_service_pb2.WeatherResponse(
                    location=location_name,
                    current_conditions=current_conditions,
                    metadata=metadata
                )
                
                logger.info(f"✅ Historical weather data retrieved for {location_name} on {request.date_str}")
                return response
            
            # Check if forecast is requested
            data_types = list(request.data_types) if request.data_types else ["current"]
            is_forecast_request = "forecast" in data_types
            
            # Import from tools-service (same pattern as RSS service)
            from tools_service.services.weather_tools import weather_forecast, weather_conditions
            
            if is_forecast_request:
                # Forecast request
                
                # Default to 3 days if not specified
                days = 3
                weather_result = await weather_forecast(
                    location=location,
                    days=days,
                    units=units,
                    user_id=request.user_id if request.user_id else None
                )
                
                if not weather_result.get("success"):
                    error_msg = weather_result.get("error", "Unknown error")
                    logger.warning(f"Forecast fetch failed: {error_msg}")
                    aborted = True
                    await context.abort(grpc.StatusCode.INTERNAL, f"Weather forecast failed: {error_msg}")
                    return
                
                # Extract forecast data
                location_name = weather_result.get("location", {}).get("name", location or "Unknown location")
                forecast = weather_result.get("forecast", [])
                
                # Format forecast days for response
                forecast_strings = []
                for day in forecast[:days]:
                    high = day.get("temperature", {}).get("high", 0)
                    low = day.get("temperature", {}).get("low", 0)
                    conditions = day.get("conditions", "")
                    forecast_strings.append(f"{day.get('day_name', 'Day')}: {high}°F/{low}°F, {conditions}")
                
                # Build metadata with forecast information
                metadata = {
                    "location_name": location_name,
                    "forecast_days": str(days),
                    "forecast_data": json.dumps(forecast[:days]) if forecast else "[]"
                }
                
                # Format current conditions string (use first day of forecast)
                if forecast:
                    first_day = forecast[0]
                    high = first_day.get("temperature", {}).get("high", 0)
                    low = first_day.get("temperature", {}).get("low", 0)
                    conditions = first_day.get("conditions", "")
                    current_conditions = f"Forecast: {high}°F/{low}°F, {conditions}"
                else:
                    current_conditions = "Forecast unavailable"
                
                response = tool_service_pb2.WeatherResponse(
                    location=location_name,
                    current_conditions=current_conditions,
                    forecast=forecast_strings,
                    metadata=metadata
                )
                
                logger.info(f"✅ Weather forecast retrieved for {location_name}: {days} days")
                return response
            else:
                # Default to current conditions
                weather_result = await weather_conditions(
                    location=location,
                    units=units,
                    user_id=request.user_id if request.user_id else None
                )
                
                if not weather_result.get("success"):
                    error_msg = weather_result.get("error", "Unknown error")
                    logger.warning(f"Weather fetch failed: {error_msg}")
                    aborted = True
                    await context.abort(grpc.StatusCode.INTERNAL, f"Weather data failed: {error_msg}")
                    return
                
                # Extract weather data
                location_name = weather_result.get("location", {}).get("name", location or "Unknown location")
                current = weather_result.get("current", {})
                temperature = int(current.get("temperature", 0))
                conditions = current.get("conditions", "")
                moon_phase = weather_result.get("moon_phase", {})
                
                # Build metadata dict with all weather information
                metadata = {
                    "location_name": location_name,
                    "temperature": str(temperature),
                    "conditions": conditions,
                    "moon_phase_name": moon_phase.get("phase_name", ""),
                    "moon_phase_icon": moon_phase.get("phase_icon", ""),
                    "moon_phase_value": str(moon_phase.get("phase_value", 0)),
                    "humidity": str(current.get("humidity", 0)),
                    "wind_speed": str(current.get("wind_speed", 0)),
                    "feels_like": str(current.get("feels_like", 0))
                }
                
                # Format current conditions string
                current_conditions = f"{temperature}°F, {conditions}"
                
                # Build response
                response = tool_service_pb2.WeatherResponse(
                    location=location_name,
                    current_conditions=current_conditions,
                    metadata=metadata
                )
                
                logger.info(f"✅ Weather data retrieved for {location_name}: {temperature}°F, {conditions}")
                return response
            
        except grpc.RpcError:
            # Already aborted - don't abort again
            raise
        except Exception as e:
            logger.error(f"GetWeatherData error: {e}")
            # Only abort if we haven't already aborted
            if not aborted:
                await context.abort(grpc.StatusCode.INTERNAL, f"Weather data failed: {str(e)}")
    
    # ===== Image Search Operations =====
    
    async def SearchImages(
        self,
        request: tool_service_pb2.ImageSearchRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ImageSearchResponse:
        """Search for images with metadata sidecars"""
        try:
            if request.is_random:
                logger.info(f"🎲 SearchImages (RANDOM): type={request.image_type}, author={request.author}, series={request.series}, limit={request.limit}")
            else:
                logger.info(f"SearchImages: query={request.query[:100]}, type={request.image_type}, date={request.date}, author={request.author}, series={request.series}")
            
            # Import image search tool
            from services.langgraph_tools.image_search_tools import ImageSearchTools
            
            image_search = ImageSearchTools()
            exclude_ids = list(request.exclude_document_ids) if request.exclude_document_ids else None
            result = await image_search.search_images(
                query=request.query,
                image_type=request.image_type if request.image_type else None,
                date=request.date if request.date else None,
                author=request.author if request.author else None,
                series=request.series if request.series else None,
                limit=request.limit or 10,
                user_id=request.user_id if request.user_id else None,
                is_random=request.is_random,
                exclude_document_ids=exclude_ids,
            )
            
            # Handle structured response (dict) or legacy string response
            if isinstance(result, dict):
                images_markdown = result.get("images_markdown", "")
                metadata_list = result.get("metadata", [])
                structured_images = result.get("images", [])
                
                # Convert metadata list to protobuf format
                pb_metadata = []
                for meta in metadata_list:
                    pb_meta = tool_service_pb2.ImageMetadata(
                        title=meta.get("title", ""),
                        date=meta.get("date", ""),
                        series=meta.get("series", ""),
                        author=meta.get("author", ""),
                        content=meta.get("content", ""),
                        tags=meta.get("tags", []),
                        image_type=meta.get("image_type", "")
                    )
                    pb_metadata.append(pb_meta)
                
                response = tool_service_pb2.ImageSearchResponse(
                    results=images_markdown,
                    success=True,
                    metadata=pb_metadata
                )
                if structured_images and hasattr(response, "structured_images_json"):
                    import json
                    response.structured_images_json = json.dumps(structured_images)
                return response
            else:
                # Legacy string format (backward compatibility)
                return tool_service_pb2.ImageSearchResponse(
                    results=result if isinstance(result, str) else str(result),
                    success=True
                )
            
        except Exception as e:
            logger.error(f"SearchImages error: {e}")
            import traceback
            traceback.print_exc()
            return tool_service_pb2.ImageSearchResponse(
                results="",
                success=False,
                error=str(e)
            )
    
    # ===== Face Analysis Operations =====
    
    async def DetectFaces(
        self,
        request: tool_service_pb2.DetectFacesRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DetectFacesResponse:
        """Detect faces in an attached image"""
        try:
            logger.info(f"🔍 DetectFaces: path={request.attachment_path}, user={request.user_id}")
            
            # Lazy import to avoid import errors if proto not generated
            try:
                from clients.image_vision_client import get_image_vision_client
            except ImportError as e:
                logger.error(f"❌ Failed to import Image Vision client: {e}")
                return tool_service_pb2.DetectFacesResponse(
                    success=False,
                    face_count=0,
                    error=f"Image Vision client not available: {str(e)}"
                )
            
            from pathlib import Path
            
            vision_client = await get_image_vision_client()
            await vision_client.initialize(required=False)
            
            if not vision_client._initialized:
                logger.warning("⚠️ Image Vision Service unavailable")
                return tool_service_pb2.DetectFacesResponse(
                    success=False,
                    face_count=0,
                    error="Image Vision Service unavailable"
                )
            
            # Use temporary document_id for attachment processing
            temp_document_id = f"attachment_{request.user_id}_{Path(request.attachment_path).stem}"
            detection_result = await vision_client.detect_faces(
                image_path=request.attachment_path,
                document_id=temp_document_id
            )
            
            if not detection_result or not detection_result.get("faces"):
                return tool_service_pb2.DetectFacesResponse(
                    success=True,
                    face_count=0,
                    image_width=detection_result.get("image_width") if detection_result else None,
                    image_height=detection_result.get("image_height") if detection_result else None
                )
            
            faces = detection_result["faces"]
            pb_faces = []
            
            for face in faces:
                pb_face = tool_service_pb2.FaceDetection(
                    face_encoding=face.get("face_encoding", []),
                    bbox_x=face.get("bbox_x", 0),
                    bbox_y=face.get("bbox_y", 0),
                    bbox_width=face.get("bbox_width", 0),
                    bbox_height=face.get("bbox_height", 0)
                )
                pb_faces.append(pb_face)
            
            return tool_service_pb2.DetectFacesResponse(
                success=True,
                faces=pb_faces,
                face_count=len(faces),
                image_width=detection_result.get("image_width"),
                image_height=detection_result.get("image_height")
            )
            
        except Exception as e:
            logger.error(f"DetectFaces error: {e}")
            import traceback
            traceback.print_exc()
            return tool_service_pb2.DetectFacesResponse(
                success=False,
                face_count=0,
                error=str(e)
            )
    
    async def IdentifyFaces(
        self,
        request: tool_service_pb2.IdentifyFacesRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.IdentifyFacesResponse:
        """Identify people in an attached image by matching against known identities"""
        try:
            confidence_threshold = request.confidence_threshold if request.HasField("confidence_threshold") else 0.82
            logger.info(f"👤 IdentifyFaces: path={request.attachment_path}, user={request.user_id}, threshold={confidence_threshold}")
            
            from services.attachment_processor_service import attachment_processor_service
            
            # Process image for face detection and identification
            result = await attachment_processor_service.process_image_for_search(
                attachment_path=request.attachment_path,
                user_id=request.user_id
            )
            
            if result.get("error"):
                return tool_service_pb2.IdentifyFacesResponse(
                    success=False,
                    face_count=0,
                    identified_count=0,
                    error=result.get("error")
                )
            
            face_count = result.get("face_count", 0)
            detected_identities = result.get("detected_identities", [])
            bounding_boxes = result.get("bounding_boxes", [])
            
            # Build identified faces list
            pb_identified_faces = []
            
            # Match bounding boxes with identities
            # Note: The current implementation returns a list of identity names
            # We'll associate them with the first N faces found
            for i, identity_name in enumerate(detected_identities):
                if i < len(bounding_boxes):
                    bbox = bounding_boxes[i]
                    pb_face = tool_service_pb2.IdentifiedFace(
                        identity_name=identity_name,
                        confidence=0.85,  # Default confidence from threshold
                        bbox_x=bbox.get("x", 0),
                        bbox_y=bbox.get("y", 0),
                        bbox_width=bbox.get("width", 0),
                        bbox_height=bbox.get("height", 0)
                    )
                    pb_identified_faces.append(pb_face)
            
            return tool_service_pb2.IdentifyFacesResponse(
                success=True,
                identified_faces=pb_identified_faces,
                face_count=face_count,
                identified_count=len(detected_identities)
            )
            
        except Exception as e:
            logger.error(f"IdentifyFaces error: {e}")
            import traceback
            traceback.print_exc()
            return tool_service_pb2.IdentifyFacesResponse(
                success=False,
                face_count=0,
                identified_count=0,
                error=str(e)
            )

    async def DetectObjects(
        self,
        request: tool_service_pb2.DetectObjectsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DetectObjectsResponse:
        """Detect objects in an image (YOLO + optional CLIP semantic matching)."""
        try:
            try:
                from clients.image_vision_client import get_image_vision_client
            except ImportError as e:
                logger.error(f"Failed to import Image Vision client: {e}")
                return tool_service_pb2.DetectObjectsResponse(
                    success=False,
                    object_count=0,
                    error=f"Image Vision client not available: {str(e)}"
                )

            vision_client = await get_image_vision_client()
            await vision_client.initialize(required=False)

            if not vision_client._initialized:
                return tool_service_pb2.DetectObjectsResponse(
                    success=False,
                    object_count=0,
                    error="Image Vision Service unavailable"
                )

            class_filter = list(request.class_filter) if request.class_filter else None
            semantic_descriptions = list(request.semantic_descriptions) if request.semantic_descriptions else None
            confidence_threshold = request.confidence_threshold if request.confidence_threshold > 0 else 0.5

            detection_result = await vision_client.detect_objects(
                image_path=request.attachment_path,
                document_id=request.document_id or "",
                class_filter=class_filter,
                confidence_threshold=confidence_threshold,
                semantic_descriptions=semantic_descriptions,
            )

            if not detection_result or not detection_result.get("objects"):
                return tool_service_pb2.DetectObjectsResponse(
                    success=True,
                    object_count=0,
                    image_width=detection_result.get("image_width") if detection_result else None,
                    image_height=detection_result.get("image_height") if detection_result else None,
                    processing_time_seconds=detection_result.get("processing_time_seconds") if detection_result else None,
                )

            objects = detection_result["objects"]
            pb_objects = []
            for obj in objects:
                pb_objects.append(tool_service_pb2.DetectedObjectProto(
                    class_name=obj.get("class_name", ""),
                    class_id=obj.get("class_id", 0),
                    confidence=obj.get("confidence", 0.0),
                    bbox_x=obj.get("bbox_x", 0),
                    bbox_y=obj.get("bbox_y", 0),
                    bbox_width=obj.get("bbox_width", 0),
                    bbox_height=obj.get("bbox_height", 0),
                    detection_method=obj.get("detection_method", "yolo"),
                    matched_description=obj.get("matched_description", ""),
                ))

            return tool_service_pb2.DetectObjectsResponse(
                success=True,
                objects=pb_objects,
                object_count=len(objects),
                image_width=detection_result.get("image_width"),
                image_height=detection_result.get("image_height"),
                processing_time_seconds=detection_result.get("processing_time_seconds"),
            )

        except Exception as e:
            logger.error(f"DetectObjects error: {e}")
            import traceback
            traceback.print_exc()
            return tool_service_pb2.DetectObjectsResponse(
                success=False,
                object_count=0,
                error=str(e)
            )

    async def IdentifyObjects(
        self,
        request: tool_service_pb2.IdentifyObjectsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.IdentifyObjectsResponse:
        """Identify objects in an image (YOLO + user-defined annotation matching)."""
        try:
            try:
                from services.object_detection_service import get_object_detection_service
            except ImportError as e:
                logger.error(f"Failed to import Object Detection service: {e}")
                return tool_service_pb2.IdentifyObjectsResponse(
                    success=False,
                    object_count=0,
                    identified_count=0,
                    error=f"Object Detection service not available: {str(e)}"
                )

            obj_service = await get_object_detection_service()
            result = await obj_service.detect_objects_in_image(
                document_id=request.document_id or "",
                image_path=request.attachment_path,
                user_id=request.user_id,
                class_filter=list(request.class_filter) if request.class_filter else None,
                confidence_threshold=request.confidence_threshold if request.confidence_threshold > 0 else 0.5,
                semantic_descriptions=list(request.semantic_descriptions) if request.semantic_descriptions else None,
                match_user_annotations=request.match_user_annotations,
                user_annotation_threshold=request.user_annotation_threshold if request.user_annotation_threshold > 0 else 0.75,
            )

            if result.get("error"):
                return tool_service_pb2.IdentifyObjectsResponse(
                    success=False,
                    object_count=0,
                    identified_count=0,
                    error=result.get("error")
                )

            objects = result.get("objects", [])
            pb_objects = []
            for obj in objects:
                annotation_id_str = str(obj["annotation_id"]) if obj.get("annotation_id") else None
                pb_objects.append(tool_service_pb2.IdentifiedObjectProto(
                    class_name=obj.get("class_name", ""),
                    confidence=obj.get("confidence", 0.0),
                    bbox_x=obj.get("bbox_x", 0),
                    bbox_y=obj.get("bbox_y", 0),
                    bbox_width=obj.get("bbox_width", 0),
                    bbox_height=obj.get("bbox_height", 0),
                    detection_method=obj.get("detection_method", "yolo"),
                    annotation_id=annotation_id_str,
                ))

            return tool_service_pb2.IdentifyObjectsResponse(
                success=True,
                identified_objects=pb_objects,
                object_count=len(objects),
                identified_count=len([o for o in objects if o.get("detection_method") == "user_defined"]),
            )

        except Exception as e:
            logger.error(f"IdentifyObjects error: {e}")
            import traceback
            traceback.print_exc()
            return tool_service_pb2.IdentifyObjectsResponse(
                success=False,
                object_count=0,
                identified_count=0,
                error=str(e)
            )

    # ===== Image Generation Operations =====

    async def GenerateImage(
        self,
        request: tool_service_pb2.ImageGenerationRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ImageGenerationResponse:
        """Generate images using OpenRouter image models"""
        try:
            logger.info(f"🎨 GenerateImage: prompt={request.prompt[:100]}...")
            
            # Get image generation service
            from services.image_generation_service import get_image_generation_service
            image_service = await get_image_generation_service()
            
            # Extract model from request (user preference) or use None to fall back to settings
            model = request.model if request.HasField("model") and request.model else None
            
            # Extract reference image parameters
            reference_image_data = None
            reference_image_url = None
            reference_strength = 0.5
            
            if request.HasField("reference_image_data") and request.reference_image_data:
                reference_image_data = request.reference_image_data
                logger.info("📎 Using reference_image_data for image-to-image generation")
            elif request.HasField("reference_image_url") and request.reference_image_url:
                reference_image_url = request.reference_image_url
                logger.info(f"📎 Using reference_image_url for image-to-image generation: {reference_image_url[:100]}")
            
            if request.HasField("reference_strength"):
                reference_strength = request.reference_strength
            
            folder_id = None
            if request.HasField("folder_id") and request.folder_id:
                folder_id = request.folder_id

            # Call image generation service
            result = await image_service.generate_images(
                prompt=request.prompt,
                size=request.size if request.size else "1024x1024",
                fmt=request.format if request.format else "png",
                seed=request.seed if request.HasField("seed") else None,
                num_images=request.num_images if request.num_images else 1,
                negative_prompt=request.negative_prompt if request.HasField("negative_prompt") else None,
                model=model,
                reference_image_data=reference_image_data,
                reference_image_url=reference_image_url,
                reference_strength=reference_strength,
                user_id=request.user_id if request.user_id else None,
                folder_id=folder_id,
            )
            
            # Convert result to proto response
            if result.get("success"):
                images = []
                for img in result.get("images", []):
                    gi = tool_service_pb2.GeneratedImage(
                        filename=img.get("filename", ""),
                        path=img.get("path", ""),
                        url=img.get("url", ""),
                        width=img.get("width", 1024),
                        height=img.get("height", 1024),
                        format=img.get("format", "png")
                    )
                    doc_id = img.get("document_id")
                    if doc_id and hasattr(gi, "document_id"):
                        gi.document_id = doc_id
                    images.append(gi)
                
                response = tool_service_pb2.ImageGenerationResponse(
                    success=True,
                    model=result.get("model", ""),
                    prompt=result.get("prompt", request.prompt),
                    size=result.get("size", "1024x1024"),
                    format=result.get("format", "png"),
                    images=images
                )
                logger.info(f"✅ Generated {len(images)} image(s) successfully")
                return response
            else:
                # Error occurred
                error_msg = result.get("error", "Unknown error")
                logger.error(f"❌ Image generation failed: {error_msg}")
                response = tool_service_pb2.ImageGenerationResponse(
                    success=False,
                    error=error_msg
                )
                return response
            
        except Exception as e:
            logger.error(f"❌ GenerateImage error: {e}")
            response = tool_service_pb2.ImageGenerationResponse(
                success=False,
                error=str(e)
            )
            return response

    async def GetReferenceImageForObject(
        self,
        request: tool_service_pb2.GetReferenceImageForObjectRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.GetReferenceImageForObjectResponse:
        """Return image bytes for an object name (from detected/annotated images) for use as reference in image generation."""
        try:
            from services.langgraph_tools.object_detection_tools import get_reference_image_bytes_for_object
            obj_name = (request.object_name or "").strip()
            user_id = request.user_id or "system"
            if not obj_name:
                return tool_service_pb2.GetReferenceImageForObjectResponse(
                    success=False,
                    error="object_name required"
                )
            result = await get_reference_image_bytes_for_object(object_name=obj_name, user_id=user_id)
            if not result:
                return tool_service_pb2.GetReferenceImageForObjectResponse(
                    success=False,
                    error="No image found for this object"
                )
            img_bytes, document_id = result
            return tool_service_pb2.GetReferenceImageForObjectResponse(
                success=True,
                reference_image_data=img_bytes,
                document_id=document_id
            )
        except Exception as e:
            logger.error("GetReferenceImageForObject error: %s", e)
            return tool_service_pb2.GetReferenceImageForObjectResponse(
                success=False,
                error=str(e)
            )

    async def TranscribeAudio(
        self,
        request: tool_service_pb2.TranscribeAudioRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.TranscribeAudioResponse:
        """Transcribe audio file to text (stub - not yet implemented)"""
        try:
            logger.info(f"🎤 TranscribeAudio: file_path={request.audio_file_path[:100] if request.audio_file_path else 'None'}...")
            
            # Get audio transcription service
            from services.audio_transcription_service import audio_transcription_service
            await audio_transcription_service.initialize()
            
            # Call transcription service (stub)
            result = await audio_transcription_service.transcribe_audio(
                file_path=request.audio_file_path,
                language=request.language if request.HasField("language") and request.language else None,
                model=request.model if request.HasField("model") and request.model else None,
                user_id=request.user_id
            )
            
            # Convert result to proto response
            response = tool_service_pb2.TranscribeAudioResponse(
                success=result.get("success", False),
                transcript=result.get("transcript", ""),
                language_detected=result.get("language_detected") if result.get("language_detected") else None
            )
            
            # Add segments if available
            segments = result.get("segments", [])
            for seg in segments:
                response.segments.append(
                    tool_service_pb2.TranscriptSegment(
                        start_time_ms=seg.get("start_time_ms", 0),
                        end_time_ms=seg.get("end_time_ms", 0),
                        text=seg.get("text", ""),
                        confidence=seg.get("confidence", 1.0)
                    )
                )
            
            if result.get("error"):
                response.error = result["error"]
            
            if not result.get("success"):
                logger.warning(f"⚠️ Audio transcription not yet implemented: {result.get('error', 'Unknown error')}")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ TranscribeAudio error: {e}")
            response = tool_service_pb2.TranscribeAudioResponse(
                success=False,
                transcript="",
                error=str(e)
            )
            return response
    
    # ===== Org-mode Operations =====
    
    async def SearchOrgFiles(
        self,
        request: tool_service_pb2.OrgSearchRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.OrgSearchResponse:
        """Search org-mode files"""
        try:
            logger.info(f"SearchOrgFiles: query={request.query}")
            
            # Placeholder implementation
            response = tool_service_pb2.OrgSearchResponse()
            return response
            
        except Exception as e:
            logger.error(f"SearchOrgFiles error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Org search failed: {str(e)}")
    
    async def GetOrgInboxItems(
        self,
        request: tool_service_pb2.OrgInboxRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.OrgInboxResponse:
        """Get org-mode inbox items"""
        try:
            logger.info(f"GetOrgInboxItems: user={request.user_id}")
            
            # Placeholder implementation
            response = tool_service_pb2.OrgInboxResponse()
            return response
            
        except Exception as e:
            logger.error(f"GetOrgInboxItems error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Get inbox items failed: {str(e)}")
    
    # ===== Org Inbox Management Operations =====
    
    async def ListOrgInboxItems(
        self,
        request: tool_service_pb2.ListOrgInboxItemsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ListOrgInboxItemsResponse:
        """List all org inbox items for user"""
        try:
            logger.info(f"ListOrgInboxItems: user={request.user_id}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_list_items, org_inbox_path
            
            # Get inbox path
            path = await org_inbox_path(request.user_id)
            
            # List items
            listing = await org_inbox_list_items(request.user_id)
            
            # Convert to proto response
            response = tool_service_pb2.ListOrgInboxItemsResponse(
                success=True,
                path=path
            )
            
            for item in listing.get("items", []):
                item_details = tool_service_pb2.OrgInboxItemDetails(
                    line_index=item.get("line_index", 0),
                    text=item.get("text", ""),
                    item_type=item.get("item_type", "plain"),
                    todo_state=item.get("todo_state", ""),
                    tags=item.get("tags", []),
                    is_done=item.get("is_done", False)
                )
                response.items.append(item_details)
            
            logger.info(f"ListOrgInboxItems: Found {len(response.items)} items")
            return response
            
        except Exception as e:
            logger.error(f"❌ ListOrgInboxItems error: {e}")
            return tool_service_pb2.ListOrgInboxItemsResponse(
                success=False,
                error=str(e)
            )
    
    async def AddOrgInboxItem(
        self,
        request: tool_service_pb2.AddOrgInboxItemRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.AddOrgInboxItemResponse:
        """Add new item to org inbox"""
        try:
            logger.info(f"AddOrgInboxItem: user={request.user_id}, kind={request.kind}, text={request.text[:50]}...")
            
            from services.langgraph_tools.org_inbox_tools import (
                org_inbox_add_item,
                org_inbox_append_text,
                org_inbox_list_items,
                org_inbox_set_schedule_and_repeater,
                org_inbox_apply_tags
            )
            from services.org_todo_service import _strip_trailing_org_tags_from_title

            # If tags will be applied, strip any trailing org-style tags from text to avoid duplicating
            text = request.text or ""
            if request.tags:
                text = _strip_trailing_org_tags_from_title(text)
            
            # Handle different kinds of entries
            if request.kind == "contact":
                # Build contact entry with PROPERTIES drawer
                headline = f"* {text}"
                org_entry = f"{headline}\n"
                
                if request.contact_properties:
                    org_entry += ":PROPERTIES:\n"
                    for key, value in request.contact_properties.items():
                        if value:
                            org_entry += f":{key}: {value}\n"
                    org_entry += ":END:\n"
                
                result = await org_inbox_append_text(org_entry, request.user_id)
                line_index = None  # Will determine after listing
                
            elif request.kind == "note":
                # Headline without TODO (plain note)
                headline = f"* {text}"
                org_entry = f"{headline}\n"
                result = await org_inbox_append_text(org_entry, request.user_id)
                listing = await org_inbox_list_items(request.user_id)
                items = listing.get("items", [])
                line_index = items[-1].get("line_index") if items else None
                
            elif request.schedule or request.kind == "event":
                # Build a proper org-mode entry with schedule
                org_type = "TODO" if request.kind == "todo" else ""
                headline = f"* {org_type} {text}".strip()
                org_entry = f"{headline}\n"
                result = await org_inbox_append_text(org_entry, request.user_id)
                
                # Get the line index of the newly added item
                listing = await org_inbox_list_items(request.user_id)
                items = listing.get("items", [])
                line_index = items[-1].get("line_index") if items else None
                
                # Set schedule if provided
                if line_index is not None and request.schedule:
                    await org_inbox_set_schedule_and_repeater(
                        line_index=line_index,
                        scheduled=request.schedule,
                        repeater=request.repeater if request.repeater else None,
                        user_id=request.user_id
                    )
            else:
                # Regular todo or checkbox
                kind = "todo" if request.kind != "checkbox" else "checkbox"
                result = await org_inbox_add_item(text=text, kind=kind, user_id=request.user_id)
                line_index = result.get("line_index")
            
            # Apply tags if provided
            if line_index is not None and request.tags:
                await org_inbox_apply_tags(line_index=line_index, tags=list(request.tags), user_id=request.user_id)
            elif line_index is None and request.tags:
                # Best effort: get last item's index
                listing = await org_inbox_list_items(request.user_id)
                items = listing.get("items", [])
                if items:
                    line_index = items[-1].get("line_index")
                    if line_index is not None:
                        await org_inbox_apply_tags(line_index=line_index, tags=list(request.tags), user_id=request.user_id)
            
            logger.info(f"✅ AddOrgInboxItem: Added item successfully")
            return tool_service_pb2.AddOrgInboxItemResponse(
                success=True,
                line_index=line_index if line_index is not None else 0,
                message=f"Added '{text}' to inbox.org"
            )
            
        except Exception as e:
            logger.error(f"❌ AddOrgInboxItem error: {e}")
            return tool_service_pb2.AddOrgInboxItemResponse(
                success=False,
                error=str(e)
            )

    async def CaptureJournalEntry(
        self,
        request: tool_service_pb2.CaptureJournalEntryRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.CaptureJournalEntryResponse:
        """Append a journal entry; respects user journal preferences and date hierarchy."""
        try:
            from services.org_journal_service import get_org_journal_service
            from models.org_capture_models import OrgCaptureRequest

            content = request.content or ""
            if request.HasField("title") and request.title:
                content = f"{request.title}\n{content}"
            capture_req = OrgCaptureRequest(
                content=content,
                template_type="journal",
                tags=list(request.tags) if request.tags else None,
                entry_date=request.entry_date if request.HasField("entry_date") and request.entry_date else None,
            )
            svc = await get_org_journal_service()
            response = await svc.capture_journal_entry(request.user_id, capture_req)
            return tool_service_pb2.CaptureJournalEntryResponse(
                success=response.success,
                message=response.message,
                entry_preview=response.entry_preview or "",
                file_path=response.file_path or "",
                document_id="",
            )
        except Exception as e:
            logger.error("CaptureJournalEntry error: %s", e)
            return tool_service_pb2.CaptureJournalEntryResponse(
                success=False,
                message="",
                error=str(e),
            )

    async def GetJournalEntry(
        self,
        request: tool_service_pb2.GetJournalEntryRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetJournalEntryResponse:
        """Read one date's journal entry (section-aware)."""
        try:
            from services.org_journal_service import get_org_journal_service
            svc = await get_org_journal_service()
            date_str = request.date or "today"
            result = await svc.get_journal_entry(request.user_id, date_str)
            return tool_service_pb2.GetJournalEntryResponse(
                success=result.get("success", False),
                content=result.get("content", ""),
                date=result.get("date", ""),
                heading=result.get("heading", ""),
                document_id=result.get("document_id") or "",
                file_path=result.get("file_path") or "",
                has_content=result.get("has_content", False),
                error=result.get("error") or "",
            )
        except Exception as e:
            logger.error("GetJournalEntry error: %s", e)
            return tool_service_pb2.GetJournalEntryResponse(
                success=False,
                content="",
                date="",
                heading="",
                has_content=False,
                error=str(e),
            )

    async def GetJournalEntries(
        self,
        request: tool_service_pb2.GetJournalEntriesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetJournalEntriesResponse:
        """Get full content of journal entries in a date range (review/lookback)."""
        try:
            from services.org_journal_service import get_org_journal_service
            svc = await get_org_journal_service()
            max_entries = 100
            if request.HasField("max_entries") and request.max_entries > 0:
                max_entries = request.max_entries
            result = await svc.get_journal_entries(
                request.user_id,
                start_date=request.start_date if request.HasField("start_date") and request.start_date else None,
                end_date=request.end_date if request.HasField("end_date") and request.end_date else None,
                max_entries=max_entries,
            )
            if not result.get("success"):
                return tool_service_pb2.GetJournalEntriesResponse(
                    success=False,
                    total=0,
                    error=result.get("error") or "",
                )
            entries = [
                tool_service_pb2.JournalEntryWithContent(
                    date=e["date"],
                    content=e.get("content", ""),
                    heading=e.get("heading", ""),
                    has_content=e.get("has_content", False),
                )
                for e in result.get("entries", [])
            ]
            return tool_service_pb2.GetJournalEntriesResponse(
                success=True,
                entries=entries,
                total=result.get("total", 0),
            )
        except Exception as e:
            logger.error("GetJournalEntries error: %s", e)
            return tool_service_pb2.GetJournalEntriesResponse(
                success=False,
                total=0,
                error=str(e),
            )

    async def UpdateJournalEntry(
        self,
        request: tool_service_pb2.UpdateJournalEntryRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateJournalEntryResponse:
        """Replace or append to a single date's journal section only."""
        try:
            from services.org_journal_service import get_org_journal_service
            svc = await get_org_journal_service()
            result = await svc.update_journal_entry(
                request.user_id,
                request.date or "",
                request.content or "",
                request.mode or "replace",
            )
            return tool_service_pb2.UpdateJournalEntryResponse(
                success=result.get("success", False),
                date=result.get("date", ""),
                error=result.get("error") or "",
            )
        except Exception as e:
            logger.error("UpdateJournalEntry error: %s", e)
            return tool_service_pb2.UpdateJournalEntryResponse(
                success=False,
                date=request.date or "",
                error=str(e),
            )

    async def ListJournalEntries(
        self,
        request: tool_service_pb2.ListJournalEntriesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListJournalEntriesResponse:
        """List journal entries in a date range with metadata."""
        try:
            from services.org_journal_service import get_org_journal_service
            svc = await get_org_journal_service()
            result = await svc.list_journal_entries(
                request.user_id,
                start_date=request.start_date if request.HasField("start_date") and request.start_date else None,
                end_date=request.end_date if request.HasField("end_date") and request.end_date else None,
            )
            if not result.get("success"):
                return tool_service_pb2.ListJournalEntriesResponse(
                    success=False,
                    total=0,
                    error=result.get("error") or "",
                )
            entries = [
                tool_service_pb2.JournalEntryMeta(
                    date=e["date"],
                    word_count=e.get("word_count", 0),
                    has_content=e.get("has_content", False),
                )
                for e in result.get("entries", [])
            ]
            return tool_service_pb2.ListJournalEntriesResponse(
                success=True,
                entries=entries,
                total=result.get("total", 0),
            )
        except Exception as e:
            logger.error("ListJournalEntries error: %s", e)
            return tool_service_pb2.ListJournalEntriesResponse(
                success=False,
                total=0,
                error=str(e),
            )

    async def SearchJournal(
        self,
        request: tool_service_pb2.SearchJournalRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SearchJournalResponse:
        """Search within journal entry content in a date range."""
        try:
            from services.org_journal_service import get_org_journal_service
            svc = await get_org_journal_service()
            result = await svc.search_journal_entries(
                request.user_id,
                request.query or "",
                start_date=request.start_date if request.HasField("start_date") and request.start_date else None,
                end_date=request.end_date if request.HasField("end_date") and request.end_date else None,
            )
            if not result.get("success"):
                return tool_service_pb2.SearchJournalResponse(
                    success=False,
                    count=0,
                    error=result.get("error") or "",
                )
            results = [
                tool_service_pb2.JournalSearchResult(
                    date=r["date"],
                    excerpt=r.get("excerpt", ""),
                )
                for r in result.get("results", [])
            ]
            return tool_service_pb2.SearchJournalResponse(
                success=True,
                results=results,
                count=result.get("count", 0),
            )
        except Exception as e:
            logger.error("SearchJournal error: %s", e)
            return tool_service_pb2.SearchJournalResponse(
                success=False,
                count=0,
                error=str(e),
            )

    async def ToggleOrgInboxItem(
        self,
        request: tool_service_pb2.ToggleOrgInboxItemRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ToggleOrgInboxItemResponse:
        """Toggle DONE status of org inbox item"""
        try:
            logger.info(f"ToggleOrgInboxItem: user={request.user_id}, line={request.line_index}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_toggle_done
            
            result = await org_inbox_toggle_done(line_index=request.line_index, user_id=request.user_id)
            
            if result.get("error"):
                return tool_service_pb2.ToggleOrgInboxItemResponse(
                    success=False,
                    error=result.get("error")
                )
            
            return tool_service_pb2.ToggleOrgInboxItemResponse(
                success=True,
                updated_index=result.get("updated_index", request.line_index),
                new_line=result.get("new_line", "")
            )
            
        except Exception as e:
            logger.error(f"❌ ToggleOrgInboxItem error: {e}")
            return tool_service_pb2.ToggleOrgInboxItemResponse(
                success=False,
                error=str(e)
            )
    
    async def UpdateOrgInboxItem(
        self,
        request: tool_service_pb2.UpdateOrgInboxItemRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.UpdateOrgInboxItemResponse:
        """Update org inbox item text"""
        try:
            logger.info(f"UpdateOrgInboxItem: user={request.user_id}, line={request.line_index}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_update_line
            
            result = await org_inbox_update_line(
                line_index=request.line_index,
                new_text=request.new_text,
                user_id=request.user_id
            )
            
            if result.get("error"):
                return tool_service_pb2.UpdateOrgInboxItemResponse(
                    success=False,
                    error=result.get("error")
                )
            
            return tool_service_pb2.UpdateOrgInboxItemResponse(
                success=True,
                updated_index=result.get("updated_index", request.line_index),
                new_line=result.get("new_line", "")
            )
            
        except Exception as e:
            logger.error(f"❌ UpdateOrgInboxItem error: {e}")
            return tool_service_pb2.UpdateOrgInboxItemResponse(
                success=False,
                error=str(e)
            )
    
    async def SetOrgInboxSchedule(
        self,
        request: tool_service_pb2.SetOrgInboxScheduleRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.SetOrgInboxScheduleResponse:
        """Set schedule and repeater for org inbox item"""
        try:
            logger.info(f"SetOrgInboxSchedule: user={request.user_id}, line={request.line_index}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_set_schedule_and_repeater
            
            result = await org_inbox_set_schedule_and_repeater(
                line_index=request.line_index,
                scheduled=request.scheduled,
                repeater=request.repeater if request.repeater else None,
                user_id=request.user_id
            )
            
            if result.get("error"):
                return tool_service_pb2.SetOrgInboxScheduleResponse(
                    success=False,
                    error=result.get("error")
                )
            
            return tool_service_pb2.SetOrgInboxScheduleResponse(
                success=True,
                updated_index=result.get("updated_index", request.line_index),
                scheduled_line=result.get("scheduled_line", "")
            )
            
        except Exception as e:
            logger.error(f"❌ SetOrgInboxSchedule error: {e}")
            return tool_service_pb2.SetOrgInboxScheduleResponse(
                success=False,
                error=str(e)
            )
    
    async def ApplyOrgInboxTags(
        self,
        request: tool_service_pb2.ApplyOrgInboxTagsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ApplyOrgInboxTagsResponse:
        """Apply tags to org inbox item"""
        try:
            logger.info(f"ApplyOrgInboxTags: user={request.user_id}, line={request.line_index}, tags={list(request.tags)}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_apply_tags
            
            result = await org_inbox_apply_tags(
                line_index=request.line_index,
                tags=list(request.tags),
                user_id=request.user_id
            )
            
            if result.get("error"):
                return tool_service_pb2.ApplyOrgInboxTagsResponse(
                    success=False,
                    error=result.get("error")
                )
            
            return tool_service_pb2.ApplyOrgInboxTagsResponse(
                success=True,
                applied_tags=list(request.tags)
            )
            
        except Exception as e:
            logger.error(f"❌ ApplyOrgInboxTags error: {e}")
            return tool_service_pb2.ApplyOrgInboxTagsResponse(
                success=False,
                error=str(e)
            )
    
    async def ArchiveOrgInboxDone(
        self,
        request: tool_service_pb2.ArchiveOrgInboxDoneRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ArchiveOrgInboxDoneResponse:
        """Archive all DONE items from org inbox"""
        try:
            logger.info(f"ArchiveOrgInboxDone: user={request.user_id}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_archive_done
            
            result = await org_inbox_archive_done(request.user_id)
            
            if result.get("error"):
                return tool_service_pb2.ArchiveOrgInboxDoneResponse(
                    success=False,
                    error=result.get("error")
                )
            
            archived_count = result.get("archived_count", 0)
            
            return tool_service_pb2.ArchiveOrgInboxDoneResponse(
                success=True,
                archived_count=archived_count,
                message=f"Archived {archived_count} DONE items"
            )
            
        except Exception as e:
            logger.error(f"❌ ArchiveOrgInboxDone error: {e}")
            return tool_service_pb2.ArchiveOrgInboxDoneResponse(
                success=False,
                error=str(e)
            )
    
    async def AppendOrgInboxText(
        self,
        request: tool_service_pb2.AppendOrgInboxTextRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.AppendOrgInboxTextResponse:
        """Append raw org-mode text to inbox"""
        try:
            logger.info(f"AppendOrgInboxText: user={request.user_id}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_append_text
            
            result = await org_inbox_append_text(request.text, request.user_id)
            
            if result.get("error"):
                return tool_service_pb2.AppendOrgInboxTextResponse(
                    success=False,
                    error=result.get("error")
                )
            
            return tool_service_pb2.AppendOrgInboxTextResponse(
                success=True,
                message="Text appended to inbox.org"
            )
            
        except Exception as e:
            logger.error(f"❌ AppendOrgInboxText error: {e}")
            return tool_service_pb2.AppendOrgInboxTextResponse(
                success=False,
                error=str(e)
            )
    
    async def GetOrgInboxPath(
        self,
        request: tool_service_pb2.GetOrgInboxPathRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.GetOrgInboxPathResponse:
        """Get path to user's inbox.org file"""
        try:
            logger.info(f"GetOrgInboxPath: user={request.user_id}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_path
            
            path = await org_inbox_path(request.user_id)
            
            return tool_service_pb2.GetOrgInboxPathResponse(
                success=True,
                path=path
            )
            
        except Exception as e:
            logger.error(f"❌ GetOrgInboxPath error: {e}")
            return tool_service_pb2.GetOrgInboxPathResponse(
                success=False,
                error=str(e)
            )

    # ===== Universal Todo Operations =====

    async def ListTodos(
        self,
        request: tool_service_pb2.ListTodosRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ListTodosResponse:
        """List todos; scope is all, inbox, or file path."""
        logger.info("ListTodos: user=%s scope=%s query=%s", request.user_id, request.scope or "all", (request.query or "")[:80])
        try:
            from services.org_todo_service import get_org_todo_service
            service = await get_org_todo_service()
            result = await service.list_todos(
                user_id=request.user_id,
                scope=request.scope or "all",
                states=list(request.states) if request.states else None,
                tags=list(request.tags) if request.tags else None,
                query=request.query or "",
                limit=request.limit or 100,
                include_archives=request.include_archives or False,
                include_body=getattr(request, "include_body", False) or False,
                closed_since_days=request.closed_since_days if getattr(request, "closed_since_days", 0) > 0 else None,
            )
            if not result.get("success"):
                return tool_service_pb2.ListTodosResponse(success=False, error=result.get("error", ""))
            response = tool_service_pb2.ListTodosResponse(success=True, count=result.get("count", 0), files_searched=result.get("files_searched", 0))
            for r in result.get("results", []):
                response.results.append(tool_service_pb2.TodoResult(
                    filename=r.get("filename", ""),
                    file_path=r.get("file_path", ""),
                    heading=r.get("heading", ""),
                    level=r.get("level", 0),
                    line_number=r.get("line_number", 0),
                    todo_state=r.get("todo_state", ""),
                    tags=r.get("tags", []),
                    scheduled=r.get("scheduled", "") or "",
                    deadline=r.get("deadline", "") or "",
                    document_id=r.get("document_id", "") or "",
                    preview=r.get("preview", "") or "",
                    body=r.get("body", "") or "",
                    closed=r.get("closed", "") or "",
                ))
            return response
        except Exception as e:
            logger.exception("ListTodos error")
            return tool_service_pb2.ListTodosResponse(success=False, error=str(e))

    async def CreateTodo(
        self,
        request: tool_service_pb2.CreateTodoRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.CreateTodoResponse:
        logger.info("CreateTodo: user=%s text=%s", request.user_id, (request.text or "")[:50])
        try:
            from services.org_todo_service import get_org_todo_service
            service = await get_org_todo_service()
            has_hl = getattr(request, "HasField", lambda _: False)("heading_level")
            has_ins = getattr(request, "HasField", lambda _: False)("insert_after_line_number")
            heading_level = getattr(request, "heading_level", None) if has_hl else None
            insert_after = getattr(request, "insert_after_line_number", None) if has_ins else None
            result = await service.create_todo(
                user_id=request.user_id,
                text=request.text,
                file_path=request.file_path if request.file_path else None,
                state=request.state or "TODO",
                tags=list(request.tags) if request.tags else None,
                scheduled=request.scheduled if request.scheduled else None,
                deadline=request.deadline if request.deadline else None,
                priority=request.priority if request.priority else None,
                body=(getattr(request, "body", "") or "").strip() or None,
                heading_level=heading_level,
                insert_after_line_number=insert_after,
            )
            if not result.get("success"):
                return tool_service_pb2.CreateTodoResponse(success=False, error=result.get("error", ""))
            return tool_service_pb2.CreateTodoResponse(
                success=True,
                file_path=result.get("file_path", ""),
                line_number=result.get("line_number", 0),
                heading=result.get("heading", ""),
            )
        except Exception as e:
            logger.exception("CreateTodo error")
            return tool_service_pb2.CreateTodoResponse(success=False, error=str(e))

    async def UpdateTodo(
        self,
        request: tool_service_pb2.UpdateTodoRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.UpdateTodoResponse:
        logger.info("UpdateTodo: user=%s file_path=%s line_number=%s new_state=%s", request.user_id, request.file_path, request.line_number, request.new_state or "")
        try:
            from services.org_todo_service import get_org_todo_service
            service = await get_org_todo_service()
            result = await service.update_todo(
                user_id=request.user_id,
                file_path=request.file_path,
                line_number=request.line_number,
                heading_text=request.heading_text if request.heading_text else None,
                new_state=request.new_state if request.new_state else None,
                new_text=request.new_text if request.new_text else None,
                add_tags=list(request.add_tags) if request.add_tags else None,
                remove_tags=list(request.remove_tags) if request.remove_tags else None,
                scheduled=request.scheduled if request.scheduled else None,
                deadline=request.deadline if request.deadline else None,
                priority=request.priority if request.priority else None,
                new_body=(getattr(request, "new_body", "") or "").strip() or None,
            )
            if not result.get("success"):
                return tool_service_pb2.UpdateTodoResponse(success=False, error=result.get("error", ""))
            return tool_service_pb2.UpdateTodoResponse(
                success=True,
                file_path=result.get("file_path", ""),
                line_number=result.get("line_number", 0),
                new_line=result.get("new_line", ""),
            )
        except Exception as e:
            logger.exception("UpdateTodo error")
            return tool_service_pb2.UpdateTodoResponse(success=False, error=str(e))

    async def ToggleTodo(
        self,
        request: tool_service_pb2.ToggleTodoRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ToggleTodoResponse:
        logger.info("ToggleTodo: user=%s file_path=%s line_number=%s", request.user_id, request.file_path, request.line_number)
        try:
            from services.org_todo_service import get_org_todo_service
            service = await get_org_todo_service()
            result = await service.toggle_todo(
                user_id=request.user_id,
                file_path=request.file_path,
                line_number=request.line_number,
                heading_text=request.heading_text if request.heading_text else None,
            )
            if not result.get("success"):
                logger.warning("ToggleTodo failed: %s", result.get("error", ""))
                return tool_service_pb2.ToggleTodoResponse(success=False, error=result.get("error", ""))
            return tool_service_pb2.ToggleTodoResponse(
                success=True,
                file_path=result.get("file_path", ""),
                line_number=result.get("line_number", 0),
                new_line=result.get("new_line", ""),
            )
        except Exception as e:
            logger.exception("ToggleTodo error")
            return tool_service_pb2.ToggleTodoResponse(success=False, error=str(e))

    async def DeleteTodo(
        self,
        request: tool_service_pb2.DeleteTodoRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DeleteTodoResponse:
        logger.info("DeleteTodo: user=%s file_path=%s line_number=%s", request.user_id, request.file_path, request.line_number)
        try:
            from services.org_todo_service import get_org_todo_service
            service = await get_org_todo_service()
            result = await service.delete_todo(
                user_id=request.user_id,
                file_path=request.file_path,
                line_number=request.line_number,
                heading_text=request.heading_text if request.heading_text else None,
            )
            if not result.get("success"):
                return tool_service_pb2.DeleteTodoResponse(success=False, error=result.get("error", ""))
            return tool_service_pb2.DeleteTodoResponse(
                success=True,
                file_path=result.get("file_path", ""),
                deleted_line_count=result.get("deleted_line_count", 0),
            )
        except Exception as e:
            logger.exception("DeleteTodo error")
            return tool_service_pb2.DeleteTodoResponse(success=False, error=str(e))

    async def ArchiveDoneTodos(
        self,
        request: tool_service_pb2.ArchiveDoneTodosRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ArchiveDoneTodosResponse:
        logger.info(
            "ArchiveDoneTodos: user=%s file_path=%s preview_only=%s line_number=%s",
            request.user_id, request.file_path or "inbox", getattr(request, "preview_only", False),
            getattr(request, "line_number", None),
        )
        try:
            from services.org_todo_service import get_org_todo_service
            service = await get_org_todo_service()
            line_number = None
            if hasattr(request, "line_number") and request.HasField("line_number"):
                line_number = request.line_number
            result = await service.archive_done(
                user_id=request.user_id,
                file_path=request.file_path if request.file_path else None,
                preview_only=getattr(request, "preview_only", False),
                line_number=line_number,
            )
            if result.get("error"):
                return tool_service_pb2.ArchiveDoneTodosResponse(success=False, error=result.get("error", ""))
            resp = tool_service_pb2.ArchiveDoneTodosResponse(
                success=True,
                path=result.get("path", ""),
                archived_to=result.get("archived_to", ""),
                archived_count=result.get("archived_count", 0),
            )
            if hasattr(resp, "directive_found"):
                resp.directive_found = result.get("directive_found", False)
            if hasattr(resp, "directive_value"):
                resp.directive_value = result.get("directive_value", "")
            return resp
        except Exception as e:
            logger.exception("ArchiveDoneTodos error")
            return tool_service_pb2.ArchiveDoneTodosResponse(success=False, error=str(e))

    async def RefileTodo(
        self,
        request: tool_service_pb2.RefileTodoRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.RefileTodoResponse:
        """Move a todo entry (and its subtree) from one org file to another."""
        logger.info("RefileTodo: user=%s source=%s:%s target=%s", request.user_id, request.source_file, request.source_line, request.target_file)
        try:
            from services.org_refile_service import get_org_refile_service
            service = await get_org_refile_service()
            target_heading_line = None
            if request.HasField("target_heading_line"):
                target_heading_line = request.target_heading_line + 1
            result = await service.refile_entry(
                user_id=request.user_id,
                source_file=request.source_file,
                source_line=request.source_line + 1,
                target_file=request.target_file,
                target_heading_line=target_heading_line,
            )
            if not result.get("success"):
                return tool_service_pb2.RefileTodoResponse(
                    success=False,
                    source_file=result.get("source_file", request.source_file),
                    target_file=result.get("target_file", request.target_file),
                    lines_moved=0,
                    error=result.get("error", "Unknown error"),
                )
            return tool_service_pb2.RefileTodoResponse(
                success=True,
                source_file=result.get("source_file", request.source_file),
                target_file=result.get("target_file", request.target_file),
                lines_moved=result.get("lines_moved", 0),
            )
        except Exception as e:
            logger.exception("RefileTodo error")
            return tool_service_pb2.RefileTodoResponse(
                success=False,
                source_file=request.source_file,
                target_file=request.target_file,
                lines_moved=0,
                error=str(e),
            )

    async def DiscoverRefileTargets(
        self,
        request: tool_service_pb2.DiscoverRefileTargetsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DiscoverRefileTargetsResponse:
        """List all org files and headings available as refile destinations."""
        logger.info("DiscoverRefileTargets: user=%s", request.user_id)
        try:
            from services.org_refile_service import get_org_refile_service
            service = await get_org_refile_service()
            targets = await service.discover_refile_targets(request.user_id)
            response = tool_service_pb2.DiscoverRefileTargetsResponse(success=True)
            for t in targets:
                heading_line = t.get("heading_line", 0)
                if heading_line > 0:
                    heading_line -= 1
                response.targets.append(tool_service_pb2.RefileTarget(
                    file=t.get("file", ""),
                    filename=t.get("filename", ""),
                    heading_path=t.get("heading_path", []),
                    heading_line=heading_line,
                    display_name=t.get("display_name", ""),
                    level=t.get("level", 0),
                ))
            return response
        except Exception as e:
            logger.exception("DiscoverRefileTargets error")
            return tool_service_pb2.DiscoverRefileTargetsResponse(success=False, error=str(e))

    # ===== Web Operations =====

    async def SearchWeb(
        self,
        request: tool_service_pb2.WebSearchRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.WebSearchResponse:
        """Search the web"""
        try:
            logger.info(f"SearchWeb: query={request.query}")
            
            # Import web search tool
            from services.langgraph_tools.web_content_tools import search_web
            
            # Execute search
            search_response = await search_web(query=request.query, limit=request.max_results or 15)
            
            # Parse results - search_web returns a dict with "results" key containing list
            response = tool_service_pb2.WebSearchResponse()
            
            # Extract results list from response dict
            if isinstance(search_response, dict) and search_response.get("success"):
                results_list = search_response.get("results", [])
                if isinstance(results_list, list):
                    for result in results_list[:request.max_results or 15]:
                        web_result = tool_service_pb2.WebSearchResult(
                            title=result.get('title', ''),
                            url=result.get('url', ''),
                            snippet=result.get('snippet', ''),
                            relevance_score=float(result.get('relevance_score', 0.0))
                        )
                        response.results.append(web_result)
            elif isinstance(search_response, list):
                # Fallback: if it's already a list (legacy format)
                for result in search_response[:request.max_results or 15]:
                    web_result = tool_service_pb2.WebSearchResult(
                        title=result.get('title', ''),
                        url=result.get('url', ''),
                        snippet=result.get('snippet', ''),
                        relevance_score=float(result.get('relevance_score', 0.0))
                    )
                    response.results.append(web_result)
            
            logger.info(f"SearchWeb: Found {len(response.results)} results")
            return response
            
        except Exception as e:
            logger.error(f"SearchWeb error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Web search failed: {str(e)}")
    
    async def CrawlWebContent(
        self,
        request: tool_service_pb2.WebCrawlRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.WebCrawlResponse:
        """Crawl web content from URLs, with optional pagination."""
        try:
            # Import crawl tool
            from services.langgraph_tools.crawl4ai_web_tools import crawl_web_content

            paginate = request.paginate if request.HasField("paginate") else False
            max_pages = request.max_pages if request.HasField("max_pages") else 10
            pagination_param = request.pagination_param if request.HasField("pagination_param") else None
            start_page = request.start_page if request.HasField("start_page") else 0
            next_page_css_selector = request.next_page_css_selector if request.HasField("next_page_css_selector") else None
            css_selector = request.css_selector if request.HasField("css_selector") else None
            max_urls = request.max_urls if request.HasField("max_urls") and request.max_urls > 0 else 5

            kwargs = {
                "url": request.url if request.url else None,
                "urls": list(request.urls) if request.urls else None,
                "user_id": request.user_id or "system",
                "css_selector": css_selector,
                "paginate": paginate,
                "max_pages": max_pages,
                "pagination_param": pagination_param,
                "start_page": start_page,
                "next_page_css_selector": next_page_css_selector,
                "max_urls": max_urls,
            }
            result = await crawl_web_content(**kwargs)

            response = tool_service_pb2.WebCrawlResponse()

            if isinstance(result, dict) and "results" in result:
                for item in result["results"]:
                    if not item.get("success"):
                        continue

                    metadata = item.get("metadata", {})
                    title = metadata.get("title", "") if isinstance(metadata, dict) else ""
                    content = item.get("full_content", "") or item.get("content", "")
                    html = item.get("html", "")

                    crawl_result = tool_service_pb2.WebCrawlResult(
                        url=item.get("url", ""),
                        title=title,
                        content=content,
                        html=html
                    )
                    if isinstance(metadata, dict):
                        for key, value in metadata.items():
                            crawl_result.metadata[str(key)] = str(value)
                    for img in item.get("images", [])[:20]:
                        crawl_result.images.append(img)
                    for link in item.get("links", [])[:50]:
                        crawl_result.links.append(link)

                    response.results.append(crawl_result)

            logger.info(f"CrawlWebContent: Crawled {len(response.results)} URLs")
            return response

        except Exception as e:
            logger.error(f"CrawlWebContent error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Web crawl failed: {str(e)}")
    
    async def CrawlWebsiteRecursive(
        self,
        request: tool_service_pb2.RecursiveWebsiteCrawlRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.RecursiveWebsiteCrawlResponse:
        """Recursively crawl entire website"""
        try:
            logger.info(f"CrawlWebsiteRecursive: {request.start_url}, max_pages={request.max_pages}, max_depth={request.max_depth}")
            
            # Import recursive crawler tool
            from services.langgraph_tools.website_crawler_tools import WebsiteCrawlerTools
            
            crawler = WebsiteCrawlerTools()
            
            # Execute recursive crawl
            crawl_result = await crawler.crawl_website_recursive(
                start_url=request.start_url,
                max_pages=request.max_pages if request.max_pages > 0 else 500,
                max_depth=request.max_depth if request.max_depth > 0 else 10,
                user_id=request.user_id if request.user_id else None
            )
            
            # Store crawled content (same as backend agent does)
            if crawl_result.get("success"):
                try:
                    storage_result = await self._store_crawled_website(crawl_result, request.user_id if request.user_id else None)
                    logger.info(f"CrawlWebsiteRecursive: Stored {storage_result.get('stored_count', 0)} items")
                except Exception as e:
                    logger.warning(f"CrawlWebsiteRecursive: Storage failed: {e}, but crawl succeeded")
            
            # Build response
            response = tool_service_pb2.RecursiveWebsiteCrawlResponse()
            
            if crawl_result.get("success"):
                response.success = True
                response.start_url = crawl_result.get("start_url", "")
                response.base_domain = crawl_result.get("base_domain", "")
                response.crawl_session_id = crawl_result.get("crawl_session_id", "")
                response.total_items_crawled = crawl_result.get("total_items_crawled", 0)
                response.html_pages_crawled = crawl_result.get("html_pages_crawled", 0)
                response.images_downloaded = crawl_result.get("images_downloaded", 0)
                response.documents_downloaded = crawl_result.get("documents_downloaded", 0)
                response.total_items_failed = crawl_result.get("total_items_failed", 0)
                response.max_depth_reached = crawl_result.get("max_depth_reached", 0)
                response.elapsed_time_seconds = crawl_result.get("elapsed_time_seconds", 0.0)
                
                # Add crawled pages
                crawled_pages = crawl_result.get("crawled_pages", [])
                for page in crawled_pages:
                    crawled_page = tool_service_pb2.CrawledPage()
                    crawled_page.url = page.get("url", "")
                    crawled_page.content_type = page.get("content_type", "html")
                    crawled_page.markdown_content = page.get("markdown_content", "")
                    crawled_page.html_content = page.get("html_content", "")
                    
                    # Add metadata
                    if page.get("metadata"):
                        for key, value in page["metadata"].items():
                            crawled_page.metadata[str(key)] = str(value)
                    
                    # Add links
                    crawled_page.internal_links.extend(page.get("internal_links", []))
                    crawled_page.image_links.extend(page.get("image_links", []))
                    crawled_page.document_links.extend(page.get("document_links", []))
                    
                    crawled_page.depth = page.get("depth", 0)
                    if page.get("parent_url"):
                        crawled_page.parent_url = page["parent_url"]
                    crawled_page.crawl_time = page.get("crawl_time", "")
                    
                    # Add binary content for images/documents
                    if page.get("binary_content"):
                        crawled_page.binary_content = page["binary_content"]
                    if page.get("filename"):
                        crawled_page.filename = page["filename"]
                    if page.get("mime_type"):
                        crawled_page.mime_type = page["mime_type"]
                    if page.get("size_bytes"):
                        crawled_page.size_bytes = page["size_bytes"]
                    
                    response.crawled_pages.append(crawled_page)
            else:
                response.success = False
                error_msg = crawl_result.get("error", "Unknown error")
                response.error = error_msg
            
            logger.info(f"CrawlWebsiteRecursive: Success={response.success}, Pages={response.total_items_crawled}")
            return response
            
        except Exception as e:
            logger.error(f"CrawlWebsiteRecursive error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Recursive website crawl failed: {str(e)}")
    
    async def CrawlSite(
        self,
        request: tool_service_pb2.DomainCrawlRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DomainCrawlResponse:
        """Domain-scoped crawl starting from seed URL, filtering by query criteria"""
        try:
            logger.info(f"CrawlSite: {request.seed_url}, query={request.query_criteria}, max_pages={request.max_pages}, max_depth={request.max_depth}")
            
            # Import domain-scoped crawler tool
            from services.langgraph_tools.crawl4ai_web_tools import Crawl4AIWebTools
            
            crawler = Crawl4AIWebTools()
            
            # Execute domain-scoped crawl
            crawl_result = await crawler.crawl_site(
                seed_url=request.seed_url,
                query_criteria=request.query_criteria,
                max_pages=request.max_pages if request.max_pages > 0 else 50,
                max_depth=request.max_depth if request.max_depth > 0 else 2,
                allowed_path_prefix=request.allowed_path_prefix if request.allowed_path_prefix else None,
                include_pdfs=request.include_pdfs,
                user_id=request.user_id if request.user_id else None
            )
            
            # Build response
            response = tool_service_pb2.DomainCrawlResponse()
            
            if crawl_result.get("success"):
                response.success = True
                response.domain = crawl_result.get("domain", "")
                response.successful_crawls = crawl_result.get("successful_crawls", 0)
                response.urls_considered = crawl_result.get("urls_considered", 0)
                
                # Add crawl results
                results = crawl_result.get("results", [])
                for item in results:
                    result = tool_service_pb2.DomainCrawlResult()
                    result.url = item.get("url", "")
                    result.title = ((item.get("metadata") or {}).get("title") or "No title").strip()
                    result.full_content = item.get("full_content", "")
                    result.relevance_score = item.get("relevance_score", 0.0)
                    result.success = item.get("success", False)
                    
                    # Add metadata
                    if item.get("metadata"):
                        for key, value in item["metadata"].items():
                            result.metadata[str(key)] = str(value)
                    
                    response.results.append(result)
            else:
                response.success = False
                response.error = crawl_result.get("error", "Unknown error")
            
            return response
            
        except Exception as e:
            logger.error(f"CrawlSite error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Domain crawl failed: {str(e)}")

    async def BrowserRun(
        self,
        request: tool_service_pb2.BrowserRunToolRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserRunToolResponse:
        """Run Playwright browser session: steps then final action (download, click, extract, screenshot)."""
        try:
            from services.langgraph_tools.playwright_browser_tools import browser_run
            steps = []
            for s in request.steps:
                steps.append({
                    "action": s.action or "",
                    "selector": s.selector if s.HasField("selector") else None,
                    "value": s.value if s.HasField("value") else None,
                    "wait_for": s.wait_for if s.HasField("wait_for") else None,
                    "timeout_seconds": s.timeout_seconds if s.HasField("timeout_seconds") else None,
                    "url": s.url if s.HasField("url") else None,
                })
            result = await browser_run(
                user_id=request.user_id or "system",
                url=request.url or "",
                final_action_type=request.final_action_type or "download",
                final_selector=request.final_selector or "",
                folder_path=request.folder_path or "",
                steps=steps if steps else None,
                connection_id=request.connection_id if request.HasField("connection_id") and request.connection_id else None,
                tags=list(request.tags) if request.tags else None,
                title=request.title if request.HasField("title") and request.title else None,
                goal=request.goal if request.HasField("goal") and request.goal else None,
            )
            response = tool_service_pb2.BrowserRunToolResponse()
            response.success = result.get("success", False)
            if result.get("error"):
                response.error = result["error"]
            if result.get("document_id") is not None:
                response.document_id = result["document_id"]
            if result.get("filename") is not None:
                response.filename = result["filename"]
            if result.get("file_size_bytes") is not None:
                response.file_size_bytes = result["file_size_bytes"]
            if result.get("extracted_text") is not None:
                response.extracted_text = result["extracted_text"]
            if result.get("message") is not None:
                response.message = result["message"]
            if result.get("images_markdown") is not None:
                response.images_markdown = result["images_markdown"]
            return response
        except Exception as e:
            logger.error(f"BrowserRun error: {e}")
            response = tool_service_pb2.BrowserRunToolResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserDownload(
        self,
        request: tool_service_pb2.BrowserDownloadToolRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserDownloadToolResponse:
        """Run Playwright browser session: optional steps, trigger download, save file to user folder. Delegates to BrowserRun with final_action_type=download."""
        try:
            from services.langgraph_tools.playwright_browser_tools import browser_run
            steps = []
            for s in request.steps:
                steps.append({
                    "action": s.action or "",
                    "selector": s.selector if s.HasField("selector") else None,
                    "value": s.value if s.HasField("value") else None,
                    "wait_for": s.wait_for if s.HasField("wait_for") else None,
                    "timeout_seconds": s.timeout_seconds if s.HasField("timeout_seconds") else None,
                    "url": s.url if s.HasField("url") else None,
                })
            result = await browser_run(
                user_id=request.user_id or "system",
                url=request.url or "",
                final_action_type="download",
                final_selector=request.download_selector or "",
                folder_path=request.folder_path or "Downloads",
                steps=steps if steps else None,
                connection_id=request.connection_id if request.HasField("connection_id") and request.connection_id else None,
                tags=list(request.tags) if request.tags else None,
                title=request.title if request.HasField("title") and request.title else None,
                goal=request.goal if request.HasField("goal") and request.goal else None,
            )
            response = tool_service_pb2.BrowserDownloadToolResponse()
            response.success = result.get("success", False)
            response.document_id = result.get("document_id", "")
            response.filename = result.get("filename", "")
            response.file_size_bytes = result.get("file_size_bytes", 0)
            if result.get("error"):
                response.error = result["error"]
            return response
        except Exception as e:
            logger.error(f"BrowserDownload error: {e}")
            response = tool_service_pb2.BrowserDownloadToolResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserOpenSession(
        self,
        request: tool_service_pb2.BrowserOpenSessionRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserOpenSessionResponse:
        """Create browser session; restore saved state for user/site if available."""
        try:
            from clients.crawl_service_client import get_crawl_service_client
            from services.browser_session_state_service import get_browser_session_state_service
            user_id = request.user_id or "system"
            site_domain = request.site_domain or ""
            state_svc = get_browser_session_state_service()
            state_json = await state_svc.load_session_state(user_id, site_domain) if site_domain else None
            client = await get_crawl_service_client()
            session_id = await client.browser_create_session(
                timeout_seconds=request.timeout_seconds or 30,
                storage_state_json=state_json,
            )
            response = tool_service_pb2.BrowserOpenSessionResponse()
            if session_id:
                response.success = True
                response.session_id = session_id
            else:
                response.success = False
                response.error = "Failed to create browser session"
            return response
        except Exception as e:
            logger.error(f"BrowserOpenSession error: {e}")
            response = tool_service_pb2.BrowserOpenSessionResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserNavigate(
        self,
        request: tool_service_pb2.BrowserNavigateRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserNavigateResponse:
        try:
            from clients.crawl_service_client import get_crawl_service_client
            client = await get_crawl_service_client()
            result = await client.browser_execute_action(
                request.session_id, "navigate", url=request.url or ""
            )
            response = tool_service_pb2.BrowserNavigateResponse()
            response.success = result.get("success", False)
            response.current_url = request.url or ""
            if result.get("error"):
                response.error = result["error"]
            return response
        except Exception as e:
            logger.error(f"BrowserNavigate error: {e}")
            response = tool_service_pb2.BrowserNavigateResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserClick(
        self,
        request: tool_service_pb2.BrowserClickRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserClickResponse:
        try:
            from clients.crawl_service_client import get_crawl_service_client
            client = await get_crawl_service_client()
            result = await client.browser_execute_action(
                request.session_id, "click", selector=request.selector or ""
            )
            response = tool_service_pb2.BrowserClickResponse()
            response.success = result.get("success", False)
            if result.get("error"):
                response.error = result["error"]
            return response
        except Exception as e:
            logger.error(f"BrowserClick error: {e}")
            response = tool_service_pb2.BrowserClickResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserFill(
        self,
        request: tool_service_pb2.BrowserFillRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserFillResponse:
        try:
            from clients.crawl_service_client import get_crawl_service_client
            client = await get_crawl_service_client()
            result = await client.browser_execute_action(
                request.session_id, "fill",
                selector=request.selector or "",
                value=request.value or "",
            )
            response = tool_service_pb2.BrowserFillResponse()
            response.success = result.get("success", False)
            if result.get("error"):
                response.error = result["error"]
            return response
        except Exception as e:
            logger.error(f"BrowserFill error: {e}")
            response = tool_service_pb2.BrowserFillResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserWait(
        self,
        request: tool_service_pb2.BrowserWaitRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserWaitResponse:
        try:
            from clients.crawl_service_client import get_crawl_service_client
            client = await get_crawl_service_client()
            result = await client.browser_execute_action(
                request.session_id, "wait",
                wait_selector=request.selector if request.HasField("selector") and request.selector else None,
                wait_timeout_seconds=request.timeout_seconds if request.HasField("timeout_seconds") else None,
            )
            response = tool_service_pb2.BrowserWaitResponse()
            response.success = result.get("success", False)
            response.found = result.get("success", False)
            if result.get("error"):
                response.error = result["error"]
            return response
        except Exception as e:
            logger.error(f"BrowserWait error: {e}")
            response = tool_service_pb2.BrowserWaitResponse()
            response.success = False
            response.found = False
            response.error = str(e)
            return response

    async def BrowserScroll(
        self,
        request: tool_service_pb2.BrowserScrollRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserScrollResponse:
        try:
            from clients.crawl_service_client import get_crawl_service_client
            client = await get_crawl_service_client()
            result = await client.browser_execute_action(
                request.session_id,
                "scroll",
                direction=request.direction if request.direction else "down",
                amount_pixels=request.amount_pixels if request.amount_pixels > 0 else 800,
            )
            response = tool_service_pb2.BrowserScrollResponse()
            response.success = result.get("success", False)
            if result.get("error"):
                response.error = result["error"]
            return response
        except Exception as e:
            logger.error(f"BrowserScroll error: {e}")
            response = tool_service_pb2.BrowserScrollResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserExtract(
        self,
        request: tool_service_pb2.BrowserExtractRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserExtractResponse:
        try:
            from clients.crawl_service_client import get_crawl_service_client
            client = await get_crawl_service_client()
            result = await client.browser_execute_action(
                request.session_id, "extract", selector=request.selector or ""
            )
            response = tool_service_pb2.BrowserExtractResponse()
            response.success = result.get("success", False)
            if result.get("extracted_content") is not None:
                response.extracted_text = result["extracted_content"]
            if result.get("error"):
                response.error = result["error"]
            return response
        except Exception as e:
            logger.error(f"BrowserExtract error: {e}")
            response = tool_service_pb2.BrowserExtractResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserInspect(
        self,
        request: tool_service_pb2.BrowserInspectRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserInspectResponse:
        try:
            from clients.crawl_service_client import get_crawl_service_client
            client = await get_crawl_service_client()
            result = await client.browser_inspect_page(request.session_id)
            response = tool_service_pb2.BrowserInspectResponse()
            response.success = result.get("success", False)
            if result.get("page_structure"):
                response.page_structure = result["page_structure"]
            if result.get("error"):
                response.error = result["error"]
            return response
        except Exception as e:
            logger.error(f"BrowserInspect error: {e}")
            response = tool_service_pb2.BrowserInspectResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserScreenshot(
        self,
        request: tool_service_pb2.BrowserScreenshotRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserScreenshotResponse:
        try:
            import base64
            from clients.crawl_service_client import get_crawl_service_client
            from services.langgraph_tools.file_creation_tools import create_user_file
            client = await get_crawl_service_client()
            result = await client.browser_execute_action(
                request.session_id, "screenshot"
            )
            response = tool_service_pb2.BrowserScreenshotResponse()
            if not result.get("success"):
                response.success = False
                response.error = result.get("error", "Screenshot failed")
                return response
            png_bytes = result.get("screenshot_png") or b""
            if not png_bytes:
                response.success = False
                response.error = "Screenshot produced no image"
                return response
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            response.images_markdown = f"![Screenshot](data:image/png;base64,{b64})"
            response.success = True
            if request.folder_path:
                filename = request.title if request.HasField("title") and request.title else f"screenshot_{int(__import__('time').time())}.png"
                create_result = await create_user_file(
                    filename=filename,
                    content="",
                    folder_path=request.folder_path,
                    title=request.title if request.HasField("title") and request.title else filename,
                    tags=list(request.tags) if request.tags else [],
                    user_id=request.user_id or "system",
                    content_bytes=png_bytes,
                )
                if create_result.get("success"):
                    response.document_id = create_result.get("document_id", "")
                    response.filename = create_result.get("filename", filename)
                    response.file_size_bytes = len(png_bytes)
            return response
        except Exception as e:
            logger.error(f"BrowserScreenshot error: {e}")
            response = tool_service_pb2.BrowserScreenshotResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserDownloadFile(
        self,
        request: tool_service_pb2.BrowserDownloadFileRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserDownloadFileResponse:
        try:
            from clients.crawl_service_client import get_crawl_service_client
            from services.langgraph_tools.file_creation_tools import create_user_file
            client = await get_crawl_service_client()
            download_result = await client.browser_download_file(
                session_id=request.session_id,
                trigger_selector=request.selector or "",
                timeout_seconds=60,
            )
            response = tool_service_pb2.BrowserDownloadFileResponse()
            if not download_result.get("success"):
                response.success = False
                response.error = download_result.get("error", "Download failed")
                return response
            file_content = download_result.get("file_content") or b""
            filename = download_result.get("filename") or "download"
            if not file_content:
                response.success = False
                response.error = "Download produced no content"
                return response
            create_result = await create_user_file(
                filename=filename,
                content="",
                folder_path=request.folder_path or "Downloads",
                title=request.title if request.HasField("title") and request.title else filename,
                tags=list(request.tags) if request.tags else [],
                user_id=request.user_id or "system",
                content_bytes=file_content,
            )
            response.success = create_result.get("success", False)
            if create_result.get("document_id"):
                response.document_id = create_result["document_id"]
            if create_result.get("filename"):
                response.filename = create_result["filename"]
            response.file_size_bytes = len(file_content)
            if create_result.get("error"):
                response.error = create_result["error"]
            return response
        except Exception as e:
            logger.error(f"BrowserDownloadFile error: {e}")
            response = tool_service_pb2.BrowserDownloadFileResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserCloseSession(
        self,
        request: tool_service_pb2.BrowserCloseSessionRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserCloseSessionResponse:
        try:
            from clients.crawl_service_client import get_crawl_service_client
            from services.browser_session_state_service import get_browser_session_state_service
            client = await get_crawl_service_client()
            session_saved = False
            if request.save_state and request.site_domain:
                state_json = await client.browser_save_session_state(request.session_id)
                if state_json:
                    state_svc = get_browser_session_state_service()
                    session_saved = await state_svc.save_session_state(
                        request.user_id or "system",
                        request.site_domain,
                        state_json,
                    )
            await client.browser_destroy_session(request.session_id)
            response = tool_service_pb2.BrowserCloseSessionResponse()
            response.success = True
            response.session_saved = session_saved
            return response
        except Exception as e:
            logger.error(f"BrowserCloseSession error: {e}")
            response = tool_service_pb2.BrowserCloseSessionResponse()
            response.success = False
            response.session_saved = False
            response.error = str(e)
            return response

    async def _store_crawled_website(
        self,
        crawl_result: Dict[str, Any],
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Store crawled website content as documents (same logic as backend agent)"""
        try:
            logger.info("Storing crawled website content")
            
            from services.document_service_v2 import DocumentService
            from urllib.parse import urlparse
            import hashlib
            
            # Initialize document service
            doc_service = DocumentService()
            await doc_service.initialize()
            
            # Extract website name from URL
            parsed_url = urlparse(crawl_result["start_url"])
            website_name = parsed_url.netloc.replace("www.", "")
            
            crawled_pages = crawl_result.get("crawled_pages", [])
            stored_count = 0
            failed_count = 0
            images_stored = 0
            documents_stored = 0
            
            from pathlib import Path
            from config import settings
            
            for page in crawled_pages:
                try:
                    # Generate document ID
                    doc_id = hashlib.md5(page["url"].encode()).hexdigest()[:16]
                    content_type = page.get("content_type", "html")
                    
                    # Prepare common metadata
                    base_metadata = {
                        "category": "web_crawl",
                        "source_url": page["url"],
                        "site_root": crawl_result["base_domain"],
                        "crawl_session_id": crawl_result["crawl_session_id"],
                        "depth": page["depth"],
                        "parent_url": page.get("parent_url"),
                        "crawl_date": page["crawl_time"],
                        "website_name": website_name,
                        "content_type": content_type
                    }
                    
                    success = False
                    
                    if content_type == "html":
                        # Store HTML page as markdown text document
                        metadata = {
                            **base_metadata,
                            "title": page.get("metadata", {}).get("title", page["url"]),
                            "internal_links": page.get("internal_links", []),
                            "image_links": page.get("image_links", []),
                            "document_links": page.get("document_links", []),
                            **page.get("metadata", {})
                        }
                        
                        path_part = urlparse(page["url"]).path.strip("/") or "index"
                        filename = f"{website_name}_{path_part.replace('/', '_')}.md"
                        content = page["markdown_content"]
                        page_title = page.get("metadata", {}).get("title", page["url"])
                        
                        # Store in vector database for search
                        success = await doc_service.store_text_document(
                            doc_id=doc_id,
                            content=content,
                            metadata=metadata,
                            filename=filename,
                            user_id=user_id,
                            collection_type="user" if user_id else "global"
                        )
                        
                        # ALSO create browseable markdown file using FileManager
                        if success:
                            try:
                                from services.file_manager.agent_helpers import place_web_content
                                await place_web_content(
                                    content=content,
                                    title=page_title,
                                    url=page["url"],
                                    domain=website_name,
                                    user_id=user_id,
                                    tags=["web-crawl", website_name],
                                    description=f"Crawled from {page['url']}"
                                )
                                logger.info(f"Created browseable file for: {page_title}")
                            except Exception as e:
                                logger.warning(f"Failed to create browseable file for {page['url']}: {e}")
                        
                    elif content_type == "image":
                        # Store image binary file
                        binary_content = page.get("binary_content")
                        filename = page.get("filename", "image")
                        
                        if binary_content:
                            # Save image to uploads directory
                            upload_dir = Path(settings.UPLOAD_DIR) / "web_sources" / "images" / website_name
                            upload_dir.mkdir(parents=True, exist_ok=True)
                            
                            safe_filename = filename.replace("/", "_").replace("\\", "_")
                            file_path = upload_dir / f"{doc_id}_{safe_filename}"
                            
                            with open(file_path, 'wb') as f:
                                f.write(binary_content)
                            
                            logger.info(f"Saved image: {file_path}")
                            
                            # Create metadata entry
                            metadata = {
                                **base_metadata,
                                "title": filename,
                                "file_path": str(file_path),
                                "mime_type": page.get("mime_type"),
                                "size_bytes": page.get("size_bytes", 0)
                            }
                            
                            # Store as text document with reference to image
                            content = f"Image from {page['url']}\n\nLocal path: {file_path}\n\nSource: {website_name}"
                            
                            success = await doc_service.store_text_document(
                                doc_id=doc_id,
                                content=content,
                                metadata=metadata,
                                filename=safe_filename,
                                user_id=user_id,
                                collection_type="user" if user_id else "global"
                            )
                            
                            if success:
                                images_stored += 1
                        
                    elif content_type == "document":
                        # Store document binary file (PDF, DOC, etc.)
                        binary_content = page.get("binary_content")
                        filename = page.get("filename", "document")
                        
                        if binary_content:
                            # Save document to uploads directory
                            upload_dir = Path(settings.UPLOAD_DIR) / "web_sources" / "documents" / website_name
                            upload_dir.mkdir(parents=True, exist_ok=True)
                            
                            safe_filename = filename.replace("/", "_").replace("\\", "_")
                            file_path = upload_dir / f"{doc_id}_{safe_filename}"
                            
                            with open(file_path, 'wb') as f:
                                f.write(binary_content)
                            
                            logger.info(f"Saved document: {file_path}")
                            
                            # Create metadata entry
                            metadata = {
                                **base_metadata,
                                "title": filename,
                                "file_path": str(file_path),
                                "mime_type": page.get("mime_type"),
                                "size_bytes": page.get("size_bytes", 0)
                            }
                            
                            # Store as text document with reference to file
                            content = f"Document from {page['url']}\n\nLocal path: {file_path}\n\nFilename: {filename}\n\nSource: {website_name}"
                            
                            success = await doc_service.store_text_document(
                                doc_id=doc_id,
                                content=content,
                                metadata=metadata,
                                filename=safe_filename,
                                user_id=user_id,
                                collection_type="user" if user_id else "global"
                            )
                            
                            if success:
                                documents_stored += 1
                    
                    if success:
                        stored_count += 1
                    else:
                        failed_count += 1
                        logger.warning(f"Failed to store item: {page['url']}")
                    
                except Exception as e:
                    logger.error(f"Error storing item {page.get('url', 'unknown')}: {e}")
                    failed_count += 1
            
            logger.info(f"Stored {stored_count}/{len(crawled_pages)} items ({images_stored} images, {documents_stored} documents)")
            
            return {
                "success": True,
                "stored_count": stored_count,
                "failed_count": failed_count,
                "total_items": len(crawled_pages),
                "images_stored": images_stored,
                "documents_stored": documents_stored
            }
            
        except Exception as e:
            logger.error(f"Failed to store crawled website: {e}")
            return {
                "success": False,
                "error": str(e),
                "stored_count": 0
            }
    
    # ===== Query Enhancement =====
    
    async def ExpandQuery(
        self,
        request: tool_service_pb2.QueryExpansionRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.QueryExpansionResponse:
        """Expand query with variations"""
        try:
            logger.info(f"ExpandQuery: query={request.query}")
            
            # Import expansion tool
            from services.langgraph_tools.query_expansion_tool import expand_query
            
            # Extract conversation context if provided
            conversation_context = None
            # Check if conversation_context field is set and not empty
            if hasattr(request, 'conversation_context') and request.conversation_context:
                conversation_context = request.conversation_context
                logger.info(f"ExpandQuery: Using conversation context ({len(conversation_context)} chars)")
            
            # Execute expansion - returns JSON string
            result_json = await expand_query(
                original_query=request.query, 
                num_expansions=request.num_variations or 3,
                conversation_context=conversation_context
            )
            result = json.loads(result_json)
            
            # Parse result
            response = tool_service_pb2.QueryExpansionResponse(
                original_query=request.query,
                expansion_count=0
            )
            
            if isinstance(result, dict):
                response.original_query = result.get('original_query', request.query)
                response.expanded_queries.extend(result.get('expanded_queries', []))
                response.key_entities.extend(result.get('key_entities', []))
                response.expansion_count = len(response.expanded_queries)
            
            logger.info(f"ExpandQuery: Generated {response.expansion_count} variations")
            return response
            
        except Exception as e:
            logger.error(f"ExpandQuery error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Query expansion failed: {str(e)}")
    
    # ===== Conversation Cache =====
    
    async def SearchConversationCache(
        self,
        request: tool_service_pb2.CacheSearchRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.CacheSearchResponse:
        """Search conversation cache for previous research"""
        try:
            logger.info(f"SearchConversationCache: query={request.query}")
            
            # Import cache tool
            from services.langgraph_tools.unified_search_tools import search_conversation_cache
            
            # Execute cache search
            result = await search_conversation_cache(
                query=request.query,
                conversation_id=request.conversation_id if request.conversation_id else None,
                freshness_hours=request.freshness_hours or 24
            )
            
            response = tool_service_pb2.CacheSearchResponse(cache_hit=False)
            
            # Parse result
            if isinstance(result, dict) and result.get('cache_hit'):
                response.cache_hit = True
                entries = result.get('entries', [])
                for entry in entries:
                    cache_entry = tool_service_pb2.CacheEntry(
                        content=entry.get('content', ''),
                        timestamp=entry.get('timestamp', ''),
                        agent_name=entry.get('agent_name', ''),
                        relevance_score=float(entry.get('relevance_score', 0.0))
                    )
                    response.entries.append(cache_entry)
            
            logger.info(f"SearchConversationCache: Cache hit={response.cache_hit}, {len(response.entries)} entries")
            return response
            
        except Exception as e:
            logger.error(f"SearchConversationCache error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Cache search failed: {str(e)}")
    
    async def SearchHelpDocs(
        self,
        request: tool_service_pb2.SearchHelpDocsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.SearchHelpDocsResponse:
        """Search app help documentation in the help_docs vector collection."""
        try:
            query = (request.query or "").strip()
            limit = request.limit if request.limit > 0 else 5
            logger.info("SearchHelpDocs: query=%s, limit=%d", query[:80] if query else "", limit)
            if not query:
                return tool_service_pb2.SearchHelpDocsResponse(results=[], total_count=0)
            embedding_manager = await self._get_embedding_manager()
            if not embedding_manager:
                return tool_service_pb2.SearchHelpDocsResponse(results=[], total_count=0)
            embeddings = await embedding_manager.generate_embeddings([query])
            if not embeddings or len(embeddings) == 0:
                return tool_service_pb2.SearchHelpDocsResponse(results=[], total_count=0)
            vector_store = VectorStoreService()
            await vector_store.initialize()
            results = await vector_store.search_similar(
                query_embedding=embeddings[0],
                collection_name="help_docs",
                limit=limit,
                score_threshold=0.6,
            )
            out = []
            for r in results:
                content = (r.get("content") or "").strip()
                if not content:
                    continue
                topic_id = r.get("document_id") or ""
                title = (r.get("metadata") or {}).get("title") or topic_id.replace("-", " ").title()
                score = float(r.get("score") or 0.0)
                out.append(
                    tool_service_pb2.HelpSearchResult(
                        topic_id=topic_id,
                        title=title,
                        content=content[:8000],
                        score=score,
                    )
                )
            return tool_service_pb2.SearchHelpDocsResponse(results=out, total_count=len(out))
        except Exception as e:
            logger.error("SearchHelpDocs error: %s", e)
            await context.abort(grpc.StatusCode.INTERNAL, "Help docs search failed: %s" % str(e))
    
    # ===== File Creation Operations =====
    
    async def CreateUserFile(
        self,
        request: tool_service_pb2.CreateUserFileRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.CreateUserFileResponse:
        """Create a file in the user's My Documents section"""
        try:
            logger.info(f"CreateUserFile: user={request.user_id}, filename={request.filename}")
            
            # Import file creation tool
            from services.langgraph_tools.file_creation_tools import create_user_file
            
            # Execute file creation
            result = await create_user_file(
                filename=request.filename,
                content=request.content,
                folder_id=request.folder_id if request.folder_id else None,
                folder_path=request.folder_path if request.folder_path else None,
                title=request.title if request.title else None,
                tags=list(request.tags) if request.tags else None,
                category=request.category if request.category else None,
                user_id=request.user_id,
                content_bytes=bytes(request.binary_content) if request.binary_content else None,
            )
            
            # Build response
            if result.get("success"):
                response = tool_service_pb2.CreateUserFileResponse(
                    success=True,
                    document_id=result.get("document_id", ""),
                    filename=result.get("filename", request.filename),
                    folder_id=result.get("folder_id", ""),
                    message=result.get("message", "File created successfully")
                )
                logger.info(f"CreateUserFile: Success - {response.document_id}")
            else:
                response = tool_service_pb2.CreateUserFileResponse(
                    success=False,
                    message=result.get("message", "File creation failed"),
                    error=result.get("error", "Unknown error")
                )
                logger.warning(f"CreateUserFile: Failed - {response.error}")
            
            return response
            
        except Exception as e:
            logger.error(f"CreateUserFile error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"File creation failed: {str(e)}")
    
    async def CreateUserFolder(
        self,
        request: tool_service_pb2.CreateUserFolderRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.CreateUserFolderResponse:
        """Create a folder in the user's My Documents section"""
        try:
            logger.info(f"CreateUserFolder: user={request.user_id}, folder_name={request.folder_name}")
            
            # Import folder creation tool
            from services.langgraph_tools.file_creation_tools import create_user_folder
            
            # Execute folder creation
            result = await create_user_folder(
                folder_name=request.folder_name,
                parent_folder_id=request.parent_folder_id if request.parent_folder_id else None,
                parent_folder_path=request.parent_folder_path if request.parent_folder_path else None,
                user_id=request.user_id
            )
            
            # Build response
            if result.get("success"):
                response = tool_service_pb2.CreateUserFolderResponse(
                    success=True,
                    folder_id=result.get("folder_id", ""),
                    folder_name=result.get("folder_name", request.folder_name),
                    parent_folder_id=result.get("parent_folder_id", ""),
                    message=result.get("message", "Folder created successfully")
                )
                logger.info(f"CreateUserFolder: Success - {response.folder_id}")
            else:
                response = tool_service_pb2.CreateUserFolderResponse(
                    success=False,
                    message=result.get("message", "Folder creation failed"),
                    error=result.get("error", "Unknown error")
                )
                logger.warning(f"CreateUserFolder: Failed - {response.error}")
            
            return response
            
        except Exception as e:
            logger.error(f"CreateUserFolder error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Folder creation failed: {str(e)}")

    def _flatten_folder_tree(self, roots) -> list:
        """Flatten hierarchical folder tree into list of FolderInfo for proto."""
        flat = []
        for f in roots:
            flat.append(tool_service_pb2.FolderInfo(
                folder_id=f.folder_id,
                name=f.name,
                parent_folder_id=f.parent_folder_id or "",
                collection_type=getattr(f, "collection_type", "user") or "user",
                document_count=getattr(f, "document_count", 0) or 0,
            ))
            for child in getattr(f, "children", []) or []:
                flat.extend(self._flatten_folder_tree([child]))
        return flat

    async def GetFolderTree(
        self,
        request: tool_service_pb2.GetFolderTreeRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetFolderTreeResponse:
        """Return flat list of folders in the user's document tree."""
        try:
            from services.service_container import get_service_container

            container = await get_service_container()
            folder_service = container.folder_service
            roots = await folder_service.get_folder_tree(user_id=request.user_id)
            flat = self._flatten_folder_tree(roots)
            return tool_service_pb2.GetFolderTreeResponse(
                folders=flat,
                total_folders=len(flat),
            )
        except Exception as e:
            logger.error(f"GetFolderTree error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Folder tree failed: {str(e)}")

    async def ListFolderDocuments(
        self,
        request: tool_service_pb2.ListFolderDocumentsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListFolderDocumentsResponse:
        """List documents directly in a folder (same access rules as folder contents API)."""
        try:
            from services.service_container import get_service_container

            limit = int(request.limit) if request.limit and request.limit > 0 else 500
            offset = int(request.offset) if request.offset and request.offset >= 0 else 0
            container = await get_service_container()
            folder_service = container.folder_service
            contents = await folder_service.get_folder_contents(
                request.folder_id, request.user_id, limit=limit, offset=offset
            )
            if not contents:
                return tool_service_pb2.ListFolderDocumentsResponse(
                    documents=[],
                    total_count=0,
                    error="Folder not found or access denied",
                )
            entries = []
            for d in contents.documents:
                raw_ct = getattr(d, "collection_type", "") or ""
                ct_str = str(raw_ct.value) if hasattr(raw_ct, "value") else str(raw_ct)
                entries.append(
                    tool_service_pb2.FolderDocumentEntry(
                        document_id=str(getattr(d, "document_id", "") or ""),
                        filename=str(getattr(d, "filename", "") or ""),
                        title=str(getattr(d, "title", "") or ""),
                        collection_type=ct_str,
                    )
                )
            return tool_service_pb2.ListFolderDocumentsResponse(
                documents=entries,
                total_count=int(contents.total_documents or len(entries)),
                error="",
            )
        except Exception as e:
            logger.error("ListFolderDocuments error: %s", e)
            return tool_service_pb2.ListFolderDocumentsResponse(
                documents=[],
                total_count=0,
                error=str(e)[:500],
            )

    async def PickRandomDocumentFromFolder(
        self,
        request: tool_service_pb2.PickRandomDocumentFromFolderRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.PickRandomDocumentFromFolderResponse:
        """Return a randomly chosen document from the given folder. Optional file_extension filter (e.g. png, jpg)."""
        import random
        try:
            from services.service_container import get_service_container

            container = await get_service_container()
            folder_service = container.folder_service
            contents = await folder_service.get_folder_contents(
                request.folder_id, request.user_id, limit=500, offset=0
            )
            if not contents or not contents.documents:
                return tool_service_pb2.PickRandomDocumentFromFolderResponse(
                    found=False,
                    document_id="",
                    filename="",
                    message="No documents in folder or folder not found.",
                )
            docs = list(contents.documents)
            ext = (request.file_extension or "").strip().lower()
            if ext and not ext.startswith("."):
                ext = f".{ext}"
            if ext:
                docs = [d for d in docs if (getattr(d, "filename", "") or "").lower().endswith(ext)]
            if not docs:
                return tool_service_pb2.PickRandomDocumentFromFolderResponse(
                    found=False,
                    document_id="",
                    filename="",
                    message=f"No documents in folder with extension {request.file_extension or ext}.",
                )
            doc = random.choice(docs)
            doc_type = getattr(doc, "doc_type", None)
            doc_type_str = str(doc_type.value) if hasattr(doc_type, "value") else str(doc_type or "")
            return tool_service_pb2.PickRandomDocumentFromFolderResponse(
                found=True,
                document_id=getattr(doc, "document_id", "") or "",
                filename=getattr(doc, "filename", "") or "",
                title=getattr(doc, "title", "") or "",
                folder_id=getattr(doc, "folder_id", "") or "",
                doc_type=doc_type_str or "",
                message=f"Random document: {getattr(doc, 'filename', '')}",
            )
        except Exception as e:
            logger.error(f"PickRandomDocumentFromFolder error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Pick random document failed: {str(e)}")

    async def UpdateDocumentMetadata(
        self,
        request: tool_service_pb2.UpdateDocumentMetadataRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.UpdateDocumentMetadataResponse:
        """Update document title and/or frontmatter type"""
        try:
            logger.info(f"UpdateDocumentMetadata: user={request.user_id}, doc={request.document_id}, title={request.title}, type={request.frontmatter_type}")
            
            # Import document editing tool
            from services.langgraph_tools.document_editing_tools import update_document_metadata_tool
            
            # Execute metadata update
            result = await update_document_metadata_tool(
                document_id=request.document_id,
                title=request.title if request.title else None,
                frontmatter_type=request.frontmatter_type if request.frontmatter_type else None,
                user_id=request.user_id
            )
            
            # Build response
            if result.get("success"):
                response = tool_service_pb2.UpdateDocumentMetadataResponse(
                    success=True,
                    document_id=result.get("document_id", request.document_id),
                    updated_fields=result.get("updated_fields", []),
                    message=result.get("message", "Document metadata updated successfully")
                )
                logger.info(f"UpdateDocumentMetadata: Success - updated {len(response.updated_fields)} field(s)")
            else:
                response = tool_service_pb2.UpdateDocumentMetadataResponse(
                    success=False,
                    document_id=request.document_id,
                    message=result.get("message", "Document metadata update failed"),
                    error=result.get("error", "Unknown error")
                )
                logger.warning(f"UpdateDocumentMetadata: Failed - {response.error}")
            
            return response
            
        except Exception as e:
            logger.error(f"UpdateDocumentMetadata error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Document metadata update failed: {str(e)}")
    
    async def UpdateDocumentContent(
        self,
        request: tool_service_pb2.UpdateDocumentContentRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.UpdateDocumentContentResponse:
        """Update document content (append or replace)"""
        try:
            logger.info(f"UpdateDocumentContent: user={request.user_id}, doc={request.document_id}, append={request.append}, content_length={len(request.content)}")
            
            # Import document editing tool
            from services.langgraph_tools.document_editing_tools import update_document_content_tool
            
            # Execute content update
            result = await update_document_content_tool(
                document_id=request.document_id,
                content=request.content,
                user_id=request.user_id,
                append=request.append
            )
            
            # Build response
            if result.get("success"):
                response = tool_service_pb2.UpdateDocumentContentResponse(
                    success=True,
                    document_id=result.get("document_id", request.document_id),
                    content_length=result.get("content_length", len(request.content)),
                    message=result.get("message", "Document content updated successfully")
                )
                logger.info(f"UpdateDocumentContent: Success - updated content ({response.content_length} chars)")
            else:
                response = tool_service_pb2.UpdateDocumentContentResponse(
                    success=False,
                    document_id=request.document_id,
                    message=result.get("message", "Document content update failed"),
                    error=result.get("error", "Unknown error")
                )
                logger.warning(f"UpdateDocumentContent: Failed - {response.error}")
            
            return response
            
        except Exception as e:
            logger.error(f"UpdateDocumentContent error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Document content update failed: {str(e)}")
    
    async def ProposeDocumentEdit(
        self,
        request: tool_service_pb2.ProposeDocumentEditRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ProposeDocumentEditResponse:
        """Propose a document edit for user review"""
        try:
            logger.info(f"ProposeDocumentEdit: user={request.user_id}, doc={request.document_id}, type={request.edit_type}, agent={request.agent_name}")
            
            # Import document editing tool
            from services.langgraph_tools.document_editing_tools import propose_document_edit_tool
            
            # Convert proto operations to dicts
            operations = None
            if request.edit_type == "operations" and request.operations:
                operations = []
                for op_proto in request.operations:
                    op_dict = {
                        "op_type": op_proto.op_type,
                        "start": op_proto.start,
                        "end": op_proto.end,
                        "text": op_proto.text,
                        "pre_hash": op_proto.pre_hash,
                        "original_text": op_proto.original_text if op_proto.HasField("original_text") else None,
                        "anchor_text": op_proto.anchor_text if op_proto.HasField("anchor_text") else None,
                        "left_context": op_proto.left_context if op_proto.HasField("left_context") else None,
                        "right_context": op_proto.right_context if op_proto.HasField("right_context") else None,
                        "occurrence_index": op_proto.occurrence_index if op_proto.HasField("occurrence_index") else None,
                        "note": op_proto.note if op_proto.HasField("note") else None,
                        "confidence": op_proto.confidence if op_proto.HasField("confidence") else None,
                        "search_text": op_proto.search_text if op_proto.HasField("search_text") else None,
                    }
                    operations.append(op_dict)
            
            # Convert proto content_edit to dict
            content_edit = None
            if request.edit_type == "content" and request.HasField("content_edit"):
                ce_proto = request.content_edit
                content_edit = {
                    "edit_mode": ce_proto.edit_mode,
                    "content": ce_proto.content,
                    "insert_position": ce_proto.insert_position if ce_proto.HasField("insert_position") else None,
                    "note": ce_proto.note if ce_proto.HasField("note") else None
                }
            
            # Execute proposal
            result = await propose_document_edit_tool(
                document_id=request.document_id,
                edit_type=request.edit_type,
                operations=operations,
                content_edit=content_edit,
                agent_name=request.agent_name,
                summary=request.summary,
                requires_preview=request.requires_preview,
                user_id=request.user_id
            )
            
            # Build response
            if result.get("success"):
                response = tool_service_pb2.ProposeDocumentEditResponse(
                    success=True,
                    proposal_id=result.get("proposal_id", ""),
                    document_id=result.get("document_id", request.document_id),
                    message=result.get("message", "Document edit proposal created successfully")
                )
                logger.info(f"ProposeDocumentEdit: Success - proposal_id={response.proposal_id}")
            else:
                response = tool_service_pb2.ProposeDocumentEditResponse(
                    success=False,
                    proposal_id="",
                    document_id=request.document_id,
                    message=result.get("message", "Document edit proposal failed"),
                    error=result.get("error", "Unknown error")
                )
                logger.warning(f"ProposeDocumentEdit: Failed - {response.error}")
            
            return response
            
        except Exception as e:
            logger.error(f"ProposeDocumentEdit error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Document edit proposal failed: {str(e)}")
    
    async def ApplyOperationsDirectly(
        self,
        request: tool_service_pb2.ApplyOperationsDirectlyRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ApplyOperationsDirectlyResponse:
        """Apply operations directly to a document (for authorized agents only)"""
        try:
            logger.info(f"ApplyOperationsDirectly: user={request.user_id}, doc={request.document_id}, agent={request.agent_name}, ops={len(request.operations)}")
            
            # Import document editing tool
            from services.langgraph_tools.document_editing_tools import apply_operations_directly
            
            # Convert proto operations to dicts
            operations = []
            for op_proto in request.operations:
                op_dict = {
                    "op_type": op_proto.op_type,
                    "start": op_proto.start,
                    "end": op_proto.end,
                    "text": op_proto.text,
                    "pre_hash": op_proto.pre_hash,
                    "original_text": op_proto.original_text if op_proto.HasField("original_text") else None,
                    "anchor_text": op_proto.anchor_text if op_proto.HasField("anchor_text") else None,
                    "left_context": op_proto.left_context if op_proto.HasField("left_context") else None,
                    "right_context": op_proto.right_context if op_proto.HasField("right_context") else None,
                    "occurrence_index": op_proto.occurrence_index if op_proto.HasField("occurrence_index") else None,
                    "note": op_proto.note if op_proto.HasField("note") else None,
                    "confidence": op_proto.confidence if op_proto.HasField("confidence") else None
                }
                operations.append(op_dict)
            
            # Execute direct operation application
            result = await apply_operations_directly(
                document_id=request.document_id,
                operations=operations,
                user_id=request.user_id,
                agent_name=request.agent_name
            )
            
            # Build response
            if result.get("success"):
                response = tool_service_pb2.ApplyOperationsDirectlyResponse(
                    success=True,
                    document_id=result.get("document_id", request.document_id),
                    applied_count=result.get("applied_count", len(operations)),
                    message=result.get("message", "Operations applied successfully")
                )
                logger.info(f"ApplyOperationsDirectly: Success - {result.get('applied_count')} operations applied")
                return response
            else:
                response = tool_service_pb2.ApplyOperationsDirectlyResponse(
                    success=False,
                    document_id=request.document_id,
                    applied_count=0,
                    message=result.get("message", "Failed to apply operations"),
                    error=result.get("error", "Unknown error")
                )
                logger.warning(f"ApplyOperationsDirectly: Failed - {result.get('error')}")
                return response
                
        except Exception as e:
            logger.error(f"ApplyOperationsDirectly error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Direct operation application failed: {str(e)}")
    
    async def ApplyDocumentEditProposal(
        self,
        request: tool_service_pb2.ApplyDocumentEditProposalRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ApplyDocumentEditProposalResponse:
        """Apply an approved document edit proposal"""
        try:
            logger.info(f"ApplyDocumentEditProposal: user={request.user_id}, proposal={request.proposal_id}, selected_ops={len(request.selected_operation_indices)}")
            
            # Import document editing tool
            from services.langgraph_tools.document_editing_tools import apply_document_edit_proposal
            
            # Convert repeated int32 to list
            selected_indices = list(request.selected_operation_indices) if request.selected_operation_indices else None
            
            # Execute proposal application
            result = await apply_document_edit_proposal(
                proposal_id=request.proposal_id,
                selected_operation_indices=selected_indices,
                user_id=request.user_id
            )
            
            # Build response
            if result.get("success"):
                response = tool_service_pb2.ApplyDocumentEditProposalResponse(
                    success=True,
                    document_id=result.get("document_id", ""),
                    applied_count=result.get("applied_count", 0),
                    message=result.get("message", "Document edit proposal applied successfully")
                )
                logger.info(f"ApplyDocumentEditProposal: Success - applied {response.applied_count} edit(s)")
            else:
                response = tool_service_pb2.ApplyDocumentEditProposalResponse(
                    success=False,
                    document_id="",
                    applied_count=0,
                    message=result.get("message", "Document edit proposal application failed"),
                    error=result.get("error", "Unknown error")
                )
                logger.warning(f"ApplyDocumentEditProposal: Failed - {response.error}")
            
            return response
            
        except Exception as e:
            logger.error(f"ApplyDocumentEditProposal error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Document edit proposal application failed: {str(e)}")

    async def ListDocumentProposals(
        self,
        request: tool_service_pb2.ListDocumentProposalsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ListDocumentProposalsResponse:
        """List pending document edit proposals for a document."""
        try:
            from services.langgraph_tools.document_editing_tools import list_pending_proposals_for_document
            proposals_raw = await list_pending_proposals_for_document(
                document_id=request.document_id,
                user_id=request.user_id
            )
            summaries = []
            for p in proposals_raw:
                ops = p.get("operations") or []
                count = len(ops) if isinstance(ops, list) else 0
                if p.get("edit_type") == "content" and p.get("content_edit"):
                    count = 1
                summaries.append(tool_service_pb2.ProposalSummary(
                    proposal_id=p.get("proposal_id", ""),
                    document_id=p.get("document_id", ""),
                    edit_type=p.get("edit_type", ""),
                    agent_name=p.get("agent_name", ""),
                    summary=p.get("summary", ""),
                    operations_count=count,
                    created_at=p.get("created_at") or "",
                    expires_at=p.get("expires_at") or ""
                ))
            return tool_service_pb2.ListDocumentProposalsResponse(
                success=True,
                proposals=summaries
            )
        except Exception as e:
            logger.error(f"ListDocumentProposals error: {e}")
            return tool_service_pb2.ListDocumentProposalsResponse(
                success=False,
                error=str(e)
            )

    async def GetDocumentEditProposal(
        self,
        request: tool_service_pb2.GetDocumentEditProposalRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.GetDocumentEditProposalResponse:
        """Get full details of a document edit proposal."""
        try:
            import json
            from services.langgraph_tools.document_editing_tools import get_document_edit_proposal
            proposal = await get_document_edit_proposal(
                proposal_id=request.proposal_id,
                user_id=request.user_id
            )
            if not proposal:
                return tool_service_pb2.GetDocumentEditProposalResponse(
                    success=False,
                    error="Proposal not found"
                )
            operations_json = json.dumps(proposal.get("operations") or [])
            content_edit = proposal.get("content_edit")
            content_edit_json = json.dumps(content_edit) if content_edit is not None else ""
            return tool_service_pb2.GetDocumentEditProposalResponse(
                success=True,
                proposal_id=proposal.get("proposal_id", ""),
                document_id=proposal.get("document_id", ""),
                edit_type=proposal.get("edit_type", ""),
                operations_json=operations_json,
                content_edit_json=content_edit_json,
                agent_name=proposal.get("agent_name", ""),
                summary=proposal.get("summary", ""),
                created_at=proposal.get("created_at") or ""
            )
        except Exception as e:
            logger.error(f"GetDocumentEditProposal error: {e}")
            return tool_service_pb2.GetDocumentEditProposalResponse(
                success=False,
                error=str(e)
            )

    async def RejectDocumentEditProposal(
        self,
        request: tool_service_pb2.RejectDocumentEditProposalRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.RejectDocumentEditProposalResponse:
        """Reject (delete) a document edit proposal."""
        try:
            from services.database_manager.database_helpers import execute
            from services.langgraph_tools.document_editing_tools import get_document_edit_proposal
            proposal = await get_document_edit_proposal(
                proposal_id=request.proposal_id,
                user_id=request.user_id
            )
            if not proposal:
                return tool_service_pb2.RejectDocumentEditProposalResponse(
                    success=False,
                    error="Proposal not found"
                )
            if proposal["user_id"] != request.user_id:
                return tool_service_pb2.RejectDocumentEditProposalResponse(
                    success=False,
                    error="Access denied"
                )
            await execute(
                "DELETE FROM document_edit_proposals WHERE proposal_id = $1::uuid",
                request.proposal_id,
                rls_context={"user_id": request.user_id, "user_role": "user"}
            )
            return tool_service_pb2.RejectDocumentEditProposalResponse(success=True)
        except Exception as e:
            logger.error(f"RejectDocumentEditProposal error: {e}")
            return tool_service_pb2.RejectDocumentEditProposalResponse(
                success=False,
                error=str(e)
            )

    # ===== Conversation Operations =====

    async def UpdateConversationTitle(
        self,
        request: tool_service_pb2.UpdateConversationTitleRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.UpdateConversationTitleResponse:
        """Update conversation title"""
        try:
            logger.info(f"UpdateConversationTitle: user={request.user_id}, conversation={request.conversation_id}, title={request.title[:50] if len(request.title) > 50 else request.title}")
            
            # Use shared database pool
            from utils.shared_db_pool import get_shared_db_pool
            
            # Get shared database pool
            pool = await get_shared_db_pool()
            async with pool.acquire() as conn:
                # Set user context for RLS policies
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", request.user_id)
                
                # Verify conversation exists and belongs to user
                conversation = await conn.fetchrow(
                    "SELECT conversation_id, user_id FROM conversations WHERE conversation_id = $1",
                    request.conversation_id
                )
                
                if not conversation:
                    response = tool_service_pb2.UpdateConversationTitleResponse(
                        success=False,
                        conversation_id=request.conversation_id,
                        message="Conversation not found",
                        error="Conversation not found"
                    )
                    logger.warning(f"UpdateConversationTitle: Conversation {request.conversation_id} not found")
                    return response
                
                if conversation['user_id'] != request.user_id:
                    response = tool_service_pb2.UpdateConversationTitleResponse(
                        success=False,
                        conversation_id=request.conversation_id,
                        message="Unauthorized",
                        error="User does not own this conversation"
                    )
                    logger.warning(f"UpdateConversationTitle: User {request.user_id} does not own conversation {request.conversation_id}")
                    return response
                
                # Update title in conversations table
                await conn.execute(
                    "UPDATE conversations SET title = $1, updated_at = NOW() WHERE conversation_id = $2",
                    request.title, request.conversation_id
                )
                
                # CRITICAL FIX: Also update checkpoint's channel_values.conversation_title
                # This ensures the title is available in both the database table and the checkpoint
                try:
                    from services.orchestrator_utils import normalize_thread_id
                    from datetime import datetime
                    normalized_thread_id = normalize_thread_id(request.user_id, request.conversation_id)
                    
                    # Try to find checkpoint with normalized thread_id first
                    row = await conn.fetchrow(
                        """
                        SELECT DISTINCT ON (c.thread_id) 
                            c.thread_id,
                            c.checkpoint,
                            c.checkpoint_id
                        FROM checkpoints c
                        WHERE c.thread_id = $1 
                        AND c.checkpoint -> 'channel_values' ->> 'user_id' = $2
                        ORDER BY c.thread_id, c.checkpoint_id DESC
                        LIMIT 1
                        """,
                        normalized_thread_id,
                        request.user_id
                    )
                    thread_id_used = normalized_thread_id
                    
                    # If not found, try with conversation_id directly
                    if not row:
                        row = await conn.fetchrow(
                            """
                            SELECT DISTINCT ON (c.thread_id)
                                c.thread_id,
                                c.checkpoint,
                                c.checkpoint_id
                            FROM checkpoints c
                            WHERE c.thread_id = $1 
                              AND c.checkpoint -> 'channel_values' ->> 'user_id' = $2
                            ORDER BY c.thread_id, c.checkpoint_id DESC
                            LIMIT 1
                            """,
                            request.conversation_id,
                            request.user_id,
                        )
                        if row:
                            thread_id_used = request.conversation_id
                    
                    if row:
                        checkpoint_data = row["checkpoint"]
                        if isinstance(checkpoint_data, str):
                            import json
                            try:
                                checkpoint_data = json.loads(checkpoint_data)
                            except Exception:
                                checkpoint_data = {}
                        elif checkpoint_data is None:
                            checkpoint_data = {}
                        
                        channel_values = checkpoint_data.get("channel_values", {})
                        channel_values["conversation_title"] = request.title
                        channel_values["conversation_updated_at"] = datetime.now().isoformat()
                        checkpoint_data["channel_values"] = channel_values
                        
                        checkpoint_json = json.dumps(checkpoint_data)
                        await conn.execute(
                            """
                            UPDATE checkpoints
                            SET checkpoint = $1::jsonb
                            WHERE thread_id = $2
                              AND checkpoint -> 'channel_values' ->> 'user_id' = $3
                            """,
                            checkpoint_json,
                            thread_id_used,
                            request.user_id,
                        )
                        logger.info(f"UpdateConversationTitle: Updated checkpoint title for conversation {request.conversation_id}")
                    else:
                        logger.debug(f"UpdateConversationTitle: No checkpoint found for conversation {request.conversation_id} (this is normal for new conversations)")
                except Exception as checkpoint_error:
                    logger.warning(f"UpdateConversationTitle: Failed to update checkpoint title (non-fatal): {checkpoint_error}")
                
                logger.info(f"UpdateConversationTitle: Successfully updated title for conversation {request.conversation_id}")
                
                # Send WebSocket notification for frontend refresh
                try:
                    from utils.websocket_manager import get_websocket_manager
                    websocket_manager = get_websocket_manager()
                    if websocket_manager:
                        await websocket_manager.send_to_session(
                            session_id=request.user_id,
                            message={
                                "type": "conversation_updated",
                                "data": {"conversation_id": request.conversation_id},
                            },
                        )
                        logger.debug(f"📡 Sent WebSocket notification for title update: {request.conversation_id}")
                except Exception as ws_error:
                    logger.debug(f"WebSocket notification failed (non-fatal): {ws_error}")
                
                response = tool_service_pb2.UpdateConversationTitleResponse(
                    success=True,
                    conversation_id=request.conversation_id,
                    title=request.title,
                    message="Conversation title updated successfully"
                )
                return response
                
        except Exception as e:
            logger.error(f"UpdateConversationTitle error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Conversation title update failed: {str(e)}")
    
    # ===== Visualization Operations =====
    
    async def CreateChart(
        self,
        request: tool_service_pb2.CreateChartRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.CreateChartResponse:
        """Create a chart or graph from structured data"""
        try:
            logger.info(f"CreateChart: chart_type={request.chart_type}, title={request.title}")
            
            # Import visualization service
            from services.visualization_service import create_chart
            
            # Parse data JSON
            try:
                data = json.loads(request.data_json)
            except json.JSONDecodeError as e:
                logger.error(f"CreateChart: Invalid JSON data: {e}")
                return tool_service_pb2.CreateChartResponse(
                    success=False,
                    error=f"Invalid JSON data: {str(e)}"
                )
            
            # Call visualization service
            result = await create_chart(
                chart_type=request.chart_type,
                data=data,
                title=request.title,
                x_label=request.x_label,
                y_label=request.y_label,
                interactive=request.interactive,
                color_scheme=request.color_scheme if request.color_scheme else "plotly",
                width=request.width if request.width > 0 else 800,
                height=request.height if request.height > 0 else 600,
                include_static=request.include_static
            )
            
            # Convert result to proto response
            if result.get("success"):
                metadata_json = json.dumps(result.get("metadata", {}))
                response = tool_service_pb2.CreateChartResponse(
                    success=True,
                    chart_type=result.get("chart_type", request.chart_type),
                    output_format=result.get("output_format", "html"),
                    chart_data=result.get("chart_data", ""),
                    metadata_json=metadata_json
                )

                if result.get("static_svg"):
                    response.static_svg = result["static_svg"]
                if result.get("static_format"):
                    response.static_format = result["static_format"]
                
                return response
            else:
                return tool_service_pb2.CreateChartResponse(
                    success=False,
                    error=result.get("error", "Unknown error creating chart")
                )
                
        except Exception as e:
            logger.error(f"CreateChart error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return tool_service_pb2.CreateChartResponse(
                success=False,
                error=f"Chart creation failed: {str(e)}"
            )
    
    # ===== File Analysis Operations =====
    
    async def AnalyzeTextContent(
        self,
        request: tool_service_pb2.TextAnalysisRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.TextAnalysisResponse:
        """Analyze text content and return metrics"""
        try:
            logger.info(f"AnalyzeTextContent: user={request.user_id}, include_advanced={request.include_advanced}")
            
            # Import file analysis service from tools-service
            from tools_service.services.file_analysis_service import FileAnalysisService
            
            # Initialize service
            analysis_service = FileAnalysisService()
            
            # Analyze text
            metrics = analysis_service.analyze_text(
                content=request.content,
                include_advanced=request.include_advanced
            )
            
            # Build response
            response = tool_service_pb2.TextAnalysisResponse(
                word_count=metrics.get("word_count", 0),
                line_count=metrics.get("line_count", 0),
                non_empty_line_count=metrics.get("non_empty_line_count", 0),
                character_count=metrics.get("character_count", 0),
                character_count_no_spaces=metrics.get("character_count_no_spaces", 0),
                paragraph_count=metrics.get("paragraph_count", 0),
                sentence_count=metrics.get("sentence_count", 0),
            )
            
            # Add advanced metrics if requested
            if request.include_advanced:
                response.avg_words_per_sentence = metrics.get("avg_words_per_sentence", 0.0)
                response.avg_words_per_paragraph = metrics.get("avg_words_per_paragraph", 0.0)
            
            # Add metadata JSON for extensibility
            metadata = {
                "analysis_timestamp": None,  # Could add timestamp if needed
            }
            response.metadata_json = json.dumps(metadata)
            
            logger.debug(f"AnalyzeTextContent: Analyzed {metrics.get('word_count', 0)} words")
            return response
            
        except Exception as e:
            logger.error(f"AnalyzeTextContent error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return response with zero values on error
            return tool_service_pb2.TextAnalysisResponse(
                word_count=0,
                line_count=0,
                non_empty_line_count=0,
                character_count=0,
                character_count_no_spaces=0,
                paragraph_count=0,
                sentence_count=0,
                avg_words_per_sentence=0.0,
                avg_words_per_paragraph=0.0,
                metadata_json=json.dumps({"error": str(e)})
            )
    
    # ===== System Modeling Operations =====
    
    async def DesignSystemComponent(
        self,
        request: tool_service_pb2.DesignSystemComponentRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DesignSystemComponentResponse:
        """Design/add a system component to the topology"""
        try:
            logger.info(f"DesignSystemComponent: user={request.user_id}, component={request.component_id}")
            
            from services.system_modeling_service import SystemModelingService
            
            service = SystemModelingService()
            
            result = service.design_component(
                user_id=request.user_id,
                component_id=request.component_id,
                component_type=request.component_type,
                requires=list(request.requires),
                provides=list(request.provides),
                redundancy_group=request.redundancy_group if request.HasField("redundancy_group") else None,
                criticality=request.criticality,
                metadata=dict(request.metadata),
                dependency_logic=request.dependency_logic if request.dependency_logic else "AND",
                m_of_n_threshold=request.m_of_n_threshold,
                dependency_weights=dict(request.dependency_weights),
                integrity_threshold=request.integrity_threshold if request.integrity_threshold > 0 else 0.5
            )
            
            response = tool_service_pb2.DesignSystemComponentResponse(
                success=result["success"],
                component_id=result["component_id"],
                message=result["message"],
                topology_json=result["topology_json"]
            )
            
            if not result["success"] and "error" in result:
                response.error = result["error"]
            
            return response
            
        except Exception as e:
            logger.error(f"DesignSystemComponent failed: {e}")
            return tool_service_pb2.DesignSystemComponentResponse(
                success=False,
                component_id=request.component_id,
                message=f"Failed to design component: {str(e)}",
                error=str(e),
                topology_json="{}"
            )
    
    async def SimulateSystemFailure(
        self,
        request: tool_service_pb2.SimulateSystemFailureRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.SimulateSystemFailureResponse:
        """Simulate system failure with cascade propagation"""
        try:
            logger.info(f"SimulateSystemFailure: user={request.user_id}, components={request.failed_component_ids}")
            
            from services.system_modeling_service import SystemModelingService
            
            service = SystemModelingService()
            
            result = service.simulate_failure(
                user_id=request.user_id,
                failed_component_ids=list(request.failed_component_ids),
                failure_modes=list(request.failure_modes),
                simulation_type=request.simulation_type if request.HasField("simulation_type") else "cascade",
                monte_carlo_iterations=request.monte_carlo_iterations if request.HasField("monte_carlo_iterations") else None,
                failure_parameters=dict(request.failure_parameters)
            )
            
            if not result["success"]:
                return tool_service_pb2.SimulateSystemFailureResponse(
                    success=False,
                    simulation_id=result.get("simulation_id", ""),
                    error=result.get("error", "Unknown error"),
                    topology_json=result.get("topology_json", "{}")
                )
            
            # Build component states
            component_states = []
            for state in result["component_states"]:
                comp_state = tool_service_pb2.ComponentState(
                    component_id=state["component_id"],
                    state=state["state"],
                    failed_dependencies=state.get("failed_dependencies", []),
                    failure_probability=state.get("failure_probability", 0.0),
                    metadata=state.get("metadata", {})
                )
                component_states.append(comp_state)
            
            # Build failure paths
            failure_paths = []
            for path in result["failure_paths"]:
                failure_path = tool_service_pb2.FailurePath(
                    source_component_id=path["source_component_id"],
                    affected_component_ids=path["affected_component_ids"],
                    failure_type=path["failure_type"],
                    path_length=path["path_length"]
                )
                failure_paths.append(failure_path)
            
            # Build health metrics
            health = result["health_metrics"]
            health_metrics = tool_service_pb2.SystemHealthMetrics(
                total_components=health["total_components"],
                operational_components=health["operational_components"],
                degraded_components=health["degraded_components"],
                failed_components=health["failed_components"],
                system_health_score=health["system_health_score"],
                critical_vulnerabilities=health["critical_vulnerabilities"],
                redundancy_groups_at_risk=health["redundancy_groups_at_risk"]
            )
            
            return tool_service_pb2.SimulateSystemFailureResponse(
                success=True,
                simulation_id=result["simulation_id"],
                component_states=component_states,
                failure_paths=failure_paths,
                health_metrics=health_metrics,
                topology_json=result["topology_json"]
            )
            
        except Exception as e:
            logger.error(f"SimulateSystemFailure failed: {e}")
            return tool_service_pb2.SimulateSystemFailureResponse(
                success=False,
                simulation_id=str(uuid.uuid4()),
                error=str(e),
                topology_json="{}"
            )
    
    async def GetSystemTopology(
        self,
        request: tool_service_pb2.GetSystemTopologyRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.GetSystemTopologyResponse:
        """Get system topology as JSON"""
        try:
            logger.info(f"GetSystemTopology: user={request.user_id}")
            
            from services.system_modeling_service import SystemModelingService
            
            service = SystemModelingService()
            
            result = service.get_topology(
                user_id=request.user_id,
                system_name=request.system_name if request.HasField("system_name") else None
            )
            
            response = tool_service_pb2.GetSystemTopologyResponse(
                success=result["success"],
                topology_json=result["topology_json"],
                component_count=result["component_count"],
                edge_count=result["edge_count"],
                redundancy_groups=result["redundancy_groups"]
            )
            
            if not result["success"] and "error" in result:
                response.error = result["error"]
            
            return response
            
        except Exception as e:
            logger.error(f"GetSystemTopology failed: {e}")
            return tool_service_pb2.GetSystemTopologyResponse(
                success=False,
                error=str(e),
                topology_json="{}",
                component_count=0,
                edge_count=0
            )
    
    # ===== Data Workspace Operations =====
    
    async def ListDataWorkspaces(
        self,
        request: tool_service_pb2.ListDataWorkspacesRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ListDataWorkspacesResponse:
        """List all data workspaces for a user"""
        try:
            logger.info(f"ListDataWorkspaces: user={request.user_id}")
            
            from tools_service.services.data_workspace_service import get_data_workspace_service
            
            service = await get_data_workspace_service()
            workspaces = await service.list_workspaces(request.user_id)
            
            # Convert to proto response
            workspace_infos = []
            for ws in workspaces:
                workspace_infos.append(tool_service_pb2.DataWorkspaceInfo(
                    workspace_id=ws.get('workspace_id', ''),
                    name=ws.get('name', ''),
                    description=ws.get('description', ''),
                    icon=ws.get('icon', ''),
                    color=ws.get('color', ''),
                    is_pinned=ws.get('is_pinned', False)
                ))
            
            return tool_service_pb2.ListDataWorkspacesResponse(
                workspaces=workspace_infos,
                total_count=len(workspace_infos)
            )
            
        except Exception as e:
            logger.error(f"ListDataWorkspaces failed: {e}")
            return tool_service_pb2.ListDataWorkspacesResponse(
                workspaces=[],
                total_count=0
            )
    
    async def GetWorkspaceSchema(
        self,
        request: tool_service_pb2.GetWorkspaceSchemaRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.GetWorkspaceSchemaResponse:
        """Get complete schema for a workspace (all tables and columns)"""
        try:
            logger.info(f"GetWorkspaceSchema: workspace={request.workspace_id}, user={request.user_id}")
            
            from tools_service.services.data_workspace_service import get_data_workspace_service
            
            service = await get_data_workspace_service()
            schema_result = await service.get_workspace_schema(
                workspace_id=request.workspace_id,
                user_id=request.user_id
            )
            
            # Convert to proto response (include column descriptions and table metadata for agents)
            table_schemas = []
            for table in schema_result.get('tables', []):
                columns = []
                for col in table.get('columns', []):
                    columns.append(tool_service_pb2.ColumnInfo(
                        name=col.get('name', ''),
                        type=col.get('type', 'text'),
                        is_nullable=col.get('is_nullable', True),
                        description=col.get('description', '') or ''
                    ))
                meta = table.get('metadata_json')
                metadata_json_str = json.dumps(meta) if isinstance(meta, dict) and meta else ''
                table_schemas.append(tool_service_pb2.TableSchema(
                    table_id=table.get('table_id', ''),
                    name=table.get('name', ''),
                    description=table.get('description', ''),
                    database_id=table.get('database_id', ''),
                    database_name=table.get('database_name', ''),
                    columns=columns,
                    row_count=table.get('row_count', 0),
                    metadata_json=metadata_json_str or ''
                ))
            
            return tool_service_pb2.GetWorkspaceSchemaResponse(
                workspace_id=request.workspace_id,
                tables=table_schemas,
                total_tables=len(table_schemas)
            )
            
        except Exception as e:
            logger.error(f"GetWorkspaceSchema failed: {e}")
            return tool_service_pb2.GetWorkspaceSchemaResponse(
                workspace_id=request.workspace_id,
                tables=[],
                total_tables=0,
                error=str(e)
            )
    
    async def QueryDataWorkspace(
        self,
        request: tool_service_pb2.QueryDataWorkspaceRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.QueryDataWorkspaceResponse:
        """Execute a query against a data workspace (SQL or natural language)"""
        try:
            logger.info(f"QueryDataWorkspace: workspace={request.workspace_id}, type={request.query_type}, user={request.user_id}")
            
            from tools_service.services.data_workspace_service import get_data_workspace_service
            
            service = await get_data_workspace_service()
            params = None
            if getattr(request, 'params_json', None) and request.params_json.strip():
                try:
                    params = json.loads(request.params_json)
                    if not isinstance(params, list):
                        params = [params]
                except json.JSONDecodeError:
                    params = None
            read_only = bool(getattr(request, "read_only", False))
            result = await service.query_workspace(
                workspace_id=request.workspace_id,
                query=request.query,
                query_type=request.query_type,
                user_id=request.user_id,
                limit=request.limit if request.limit > 0 else 100,
                params=params,
                read_only=read_only,
            )
            
            arrow_bytes = result.get('arrow_results') or b''
            has_arrow = bool(result.get('has_arrow_data')) and bool(arrow_bytes)

            if has_arrow:
                results_json_str = "[]"
            else:
                results_json = result.get('results', [])
                if isinstance(results_json, str):
                    results_json_str = results_json
                else:
                    results_json_str = json.dumps(results_json)

            returning = result.get('returning_rows') or result.get('returning_rows_json')
            if isinstance(returning, str) and returning:
                try:
                    returning = json.loads(returning)
                except json.JSONDecodeError:
                    returning = []
            elif not isinstance(returning, list):
                returning = []

            response = tool_service_pb2.QueryDataWorkspaceResponse(
                success=result.get('success', False),
                column_names=result.get('column_names', []),
                results_json=results_json_str,
                result_count=result.get('result_count', 0),
                execution_time_ms=result.get('execution_time_ms', 0),
                generated_sql=result.get('generated_sql', ''),
                rows_affected=result.get('rows_affected', 0),
                returning_rows_json=json.dumps(returning),
                arrow_results=arrow_bytes,
                has_arrow_data=has_arrow,
            )

            if result.get('error_message'):
                response.error_message = result['error_message']

            return response
            
        except Exception as e:
            logger.error(f"QueryDataWorkspace failed: {e}")
            return tool_service_pb2.QueryDataWorkspaceResponse(
                success=False,
                column_names=[],
                results_json="[]",
                result_count=0,
                execution_time_ms=0,
                generated_sql="",
                error_message=str(e),
                rows_affected=0,
                returning_rows_json="[]",
                arrow_results=b"",
                has_arrow_data=False,
            )

    # ===== Navigation Operations (locations and routes) =====

    async def CreateLocation(
        self,
        request: tool_service_pb2.CreateLocationRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.CreateLocationResponse:
        """Create a saved location (geocodes address if needed)."""
        try:
            from tools_service.services.navigation_tools import create_location as nav_create_location

            result = await nav_create_location(
                user_id=request.user_id,
                name=request.name,
                address=request.address,
                latitude=request.latitude if request.HasField("latitude") else None,
                longitude=request.longitude if request.HasField("longitude") else None,
                notes=request.notes if request.notes else None,
                is_global=request.is_global,
                metadata=json.loads(request.metadata_json) if request.HasField("metadata_json") and request.metadata_json else None,
                user_role=request.user_role or "user",
            )
            if not result.get("success"):
                return tool_service_pb2.CreateLocationResponse(success=False, error=result.get("error", "Unknown error"))
            return tool_service_pb2.CreateLocationResponse(
                success=True,
                location_id=result.get("location_id", ""),
                user_id=result.get("user_id", ""),
                name=result.get("name", ""),
                address=result.get("address") or "",
                latitude=result.get("latitude", 0),
                longitude=result.get("longitude", 0),
                notes=result.get("notes") or "",
                is_global=result.get("is_global", False),
                created_at=result.get("created_at") or "",
                updated_at=result.get("updated_at") or "",
            )
        except Exception as e:
            logger.error(f"CreateLocation failed: {e}")
            return tool_service_pb2.CreateLocationResponse(success=False, error=str(e))

    async def ListLocations(
        self,
        request: tool_service_pb2.ListLocationsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ListLocationsResponse:
        """List all locations accessible to the user."""
        try:
            from tools_service.services.navigation_tools import list_locations

            result = await list_locations(
                user_id=request.user_id,
                user_role=request.user_role or "user",
            )
            if not result.get("success"):
                return tool_service_pb2.ListLocationsResponse(success=False, error=result.get("error", "Unknown error"))
            locations = []
            for loc in result.get("locations", []):
                locations.append(tool_service_pb2.LocationProto(
                    location_id=loc.get("location_id", ""),
                    user_id=loc.get("user_id", ""),
                    name=loc.get("name", ""),
                    address=loc.get("address") or "",
                    latitude=loc.get("latitude", 0),
                    longitude=loc.get("longitude", 0),
                    notes=loc.get("notes") or "",
                    is_global=loc.get("is_global", False),
                    created_at=loc.get("created_at") or "",
                    updated_at=loc.get("updated_at") or "",
                ))
            return tool_service_pb2.ListLocationsResponse(success=True, locations=locations, total=result.get("total", 0))
        except Exception as e:
            logger.error(f"ListLocations failed: {e}")
            return tool_service_pb2.ListLocationsResponse(success=False, error=str(e))

    async def DeleteLocation(
        self,
        request: tool_service_pb2.DeleteLocationRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DeleteLocationResponse:
        """Delete a location by ID."""
        try:
            from tools_service.services.navigation_tools import delete_location

            result = await delete_location(
                user_id=request.user_id,
                location_id=request.location_id,
                user_role=request.user_role or "user",
            )
            if not result.get("success"):
                return tool_service_pb2.DeleteLocationResponse(success=False, error=result.get("error", "Unknown error"))
            return tool_service_pb2.DeleteLocationResponse(success=True, message=result.get("message", "Location deleted"))
        except Exception as e:
            logger.error(f"DeleteLocation failed: {e}")
            return tool_service_pb2.DeleteLocationResponse(success=False, error=str(e))

    async def ComputeRoute(
        self,
        request: tool_service_pb2.ComputeRouteRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ComputeRouteResponse:
        """Compute route between two points (location IDs or coordinates)."""
        try:
            from tools_service.services.navigation_tools import compute_route

            result = await compute_route(
                user_id=request.user_id,
                from_location_id=request.from_location_id if request.from_location_id else None,
                to_location_id=request.to_location_id if request.to_location_id else None,
                coordinates=request.coordinates if request.coordinates else None,
                profile=request.profile or "driving",
                user_role=request.user_role or "user",
            )
            if not result.get("success"):
                return tool_service_pb2.ComputeRouteResponse(success=False, error=result.get("error", "Unknown error"))
            return tool_service_pb2.ComputeRouteResponse(
                success=True,
                geometry_json=json.dumps(result.get("geometry", {})),
                legs_json=json.dumps(result.get("legs", [])),
                distance=result.get("distance", 0),
                duration=result.get("duration", 0),
                waypoints_json=json.dumps(result.get("waypoints", [])),
            )
        except Exception as e:
            logger.error(f"ComputeRoute failed: {e}")
            return tool_service_pb2.ComputeRouteResponse(success=False, error=str(e))

    async def SaveRoute(
        self,
        request: tool_service_pb2.NavSaveRouteRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.NavSaveRouteResponse:
        """Save a computed route."""
        try:
            from tools_service.services.navigation_tools import save_route

            waypoints = json.loads(request.waypoints_json) if request.waypoints_json else []
            geometry = json.loads(request.geometry_json) if request.geometry_json else {}
            steps = json.loads(request.steps_json) if request.steps_json else []

            result = await save_route(
                user_id=request.user_id,
                name=request.name,
                waypoints=waypoints,
                geometry=geometry,
                steps=steps,
                distance_meters=request.distance_meters,
                duration_seconds=request.duration_seconds,
                profile=request.profile or "driving",
                user_role=request.user_role or "user",
            )
            if not result.get("success"):
                return tool_service_pb2.NavSaveRouteResponse(success=False, error=result.get("error", "Unknown error"))
            return tool_service_pb2.NavSaveRouteResponse(
                success=True,
                route_id=result.get("route_id", ""),
                user_id=result.get("user_id", ""),
                name=result.get("name", ""),
                waypoints_json=json.dumps(result.get("waypoints", [])),
                geometry_json=json.dumps(result.get("geometry", {})),
                steps_json=json.dumps(result.get("steps", [])),
                distance_meters=result.get("distance_meters", 0),
                duration_seconds=result.get("duration_seconds", 0),
                profile=result.get("profile", "driving"),
                created_at=result.get("created_at") or "",
                updated_at=result.get("updated_at") or "",
            )
        except Exception as e:
            logger.error(f"SaveRoute failed: {e}")
            return tool_service_pb2.NavSaveRouteResponse(success=False, error=str(e))

    async def ListSavedRoutes(
        self,
        request: tool_service_pb2.ListSavedRoutesRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ListSavedRoutesResponse:
        """List saved routes for the user."""
        try:
            from tools_service.services.navigation_tools import list_saved_routes

            result = await list_saved_routes(
                user_id=request.user_id,
                user_role=request.user_role or "user",
            )
            if not result.get("success"):
                return tool_service_pb2.ListSavedRoutesResponse(success=False, error=result.get("error", "Unknown error"))
            routes = []
            for r in result.get("routes", []):
                routes.append(tool_service_pb2.SavedRouteProto(
                    route_id=r.get("route_id", ""),
                    user_id=r.get("user_id", ""),
                    name=r.get("name", ""),
                    waypoints_json=json.dumps(r.get("waypoints", [])),
                    geometry_json=json.dumps(r.get("geometry", {})),
                    steps_json=json.dumps(r.get("steps", [])),
                    distance_meters=r.get("distance_meters", 0),
                    duration_seconds=r.get("duration_seconds", 0),
                    profile=r.get("profile", "driving"),
                    created_at=r.get("created_at") or "",
                    updated_at=r.get("updated_at") or "",
                ))
            return tool_service_pb2.ListSavedRoutesResponse(success=True, routes=routes, total=result.get("total", 0))
        except Exception as e:
            logger.error(f"ListSavedRoutes failed: {e}")
            return tool_service_pb2.ListSavedRoutesResponse(success=False, error=str(e))

    async def DeleteSavedRoute(
        self,
        request: tool_service_pb2.DeleteSavedRouteRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DeleteSavedRouteResponse:
        """Delete a saved route."""
        try:
            from tools_service.services.navigation_tools import delete_saved_route

            result = await delete_saved_route(
                user_id=request.user_id,
                route_id=request.route_id,
                user_role=request.user_role or "user",
            )
            if not result.get("success"):
                return tool_service_pb2.DeleteSavedRouteResponse(success=False, error=result.get("error", "Unknown error"))
            return tool_service_pb2.DeleteSavedRouteResponse(success=True, message=result.get("message", "Route deleted"))
        except Exception as e:
            logger.error(f"DeleteSavedRoute failed: {e}")
            return tool_service_pb2.DeleteSavedRouteResponse(success=False, error=str(e))

    async def AnalyzeWebsiteSecurity(
        self,
        request: tool_service_pb2.SecurityAnalysisRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.SecurityAnalysisResponse:
        """Perform passive security analysis of a website."""
        try:
            from tools_service.services.security_analysis_service import analyze_website

            result = await analyze_website(
                url=request.target_url,
                user_id=request.user_id or "system",
                scan_depth=request.scan_depth or "comprehensive",
            )
            if not result.get("success"):
                return tool_service_pb2.SecurityAnalysisResponse(
                    success=False,
                    target_url=result.get("target_url", request.target_url),
                    scan_timestamp=result.get("scan_timestamp", ""),
                    disclaimer=result.get("disclaimer", ""),
                    error=result.get("error", "Unknown error"),
                )
            findings_proto = []
            for f in result.get("findings", []):
                findings_proto.append(tool_service_pb2.SecurityFinding(
                    category=f.get("category", ""),
                    severity=f.get("severity", ""),
                    title=f.get("title", ""),
                    description=f.get("description", ""),
                    url=f.get("url") or None,
                    evidence=f.get("evidence") or None,
                    remediation=f.get("remediation", ""),
                ))
            tech_stack = result.get("technology_stack") or {}
            sec_headers = result.get("security_headers") or {}
            headers_map = {}
            if sec_headers.get("present"):
                headers_map["present"] = ",".join(sec_headers["present"]) if isinstance(sec_headers["present"], list) else str(sec_headers["present"])
            if sec_headers.get("missing"):
                headers_map["missing"] = ",".join(sec_headers["missing"]) if isinstance(sec_headers["missing"], list) else str(sec_headers["missing"])
            return tool_service_pb2.SecurityAnalysisResponse(
                success=True,
                target_url=result.get("target_url", ""),
                scan_timestamp=result.get("scan_timestamp", ""),
                findings=findings_proto,
                technology_stack=tech_stack,
                security_headers=headers_map,
                risk_score=float(result.get("risk_score", 0.0)),
                summary=result.get("summary", ""),
                disclaimer=result.get("disclaimer", ""),
            )
        except Exception as e:
            logger.error(f"AnalyzeWebsiteSecurity failed: {e}")
            return tool_service_pb2.SecurityAnalysisResponse(
                success=False,
                target_url=request.target_url,
                scan_timestamp=datetime.now(timezone.utc).isoformat(),
                disclaimer="This security scan performs passive reconnaissance only. Use only on sites you own or have permission to test.",
                error=str(e),
            )

    # ===== Email operations (via connections-service / email_tools) =====

    async def GetEmails(
        self,
        request: tool_service_pb2.GetEmailsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetEmailsResponse:
        """Get emails for user (inbox or folder)."""
        try:
            from services.langgraph_tools.email_tools import read_recent_emails
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await read_recent_emails(
                user_id=user_id,
                folder=request.folder or "inbox",
                count=request.top or 10,
                unread_only=request.unread_only,
                connection_id=connection_id if connection_id else None,
            )
            return tool_service_pb2.GetEmailsResponse(success=True, result=result)
        except Exception as e:
            logger.error("GetEmails failed: %s", e)
            return tool_service_pb2.GetEmailsResponse(
                success=False, result="", error=str(e)
            )

    async def SearchEmails(
        self,
        request: tool_service_pb2.SearchEmailsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SearchEmailsResponse:
        """Search emails for user."""
        try:
            from services.langgraph_tools.email_tools import search_emails
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await search_emails(
                user_id=user_id,
                query=request.query,
                top=request.top or 20,
                from_address=request.from_address or None,
                connection_id=connection_id if connection_id else None,
            )
            return tool_service_pb2.SearchEmailsResponse(success=True, result=result)
        except Exception as e:
            logger.error("SearchEmails failed: %s", e)
            return tool_service_pb2.SearchEmailsResponse(
                success=False, result="", error=str(e)
            )

    async def GetEmailThread(
        self,
        request: tool_service_pb2.GetEmailThreadRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetEmailThreadResponse:
        """Get full email thread by conversation_id."""
        try:
            from services.langgraph_tools.email_tools import get_email_thread
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await get_email_thread(
                user_id=user_id,
                conversation_id=request.conversation_id,
                connection_id=connection_id if connection_id else None,
            )
            return tool_service_pb2.GetEmailThreadResponse(success=True, result=result)
        except Exception as e:
            logger.error("GetEmailThread failed: %s", e)
            return tool_service_pb2.GetEmailThreadResponse(
                success=False, result="", error=str(e)
            )

    async def SendEmail(
        self,
        request: tool_service_pb2.SendEmailRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SendEmailResponse:
        """Send email. from_source: system = Bastion SMTP, user = user's email connection (default)."""
        try:
            from services.langgraph_tools.email_tools import send_email
            user_id = request.user_id or "system"
            from_source = (request.from_source or "user").strip().lower() or "user"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await send_email(
                user_id=user_id,
                to=list(request.to),
                subject=request.subject,
                body=request.body,
                cc=list(request.cc) if request.cc else None,
                from_source=from_source,
                connection_id=connection_id if connection_id else None,
                body_is_html=getattr(request, "body_is_html", False) or False,
            )
            if result.startswith("Email sent successfully"):
                return tool_service_pb2.SendEmailResponse(success=True, result=result)
            return tool_service_pb2.SendEmailResponse(
                success=False, result=result, error=result
            )
        except Exception as e:
            logger.error("SendEmail failed: %s", e)
            return tool_service_pb2.SendEmailResponse(
                success=False, result="", error=str(e)
            )

    async def ReplyToEmail(
        self,
        request: tool_service_pb2.ReplyToEmailRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ReplyToEmailResponse:
        """Reply to an email."""
        try:
            from services.langgraph_tools.email_tools import reply_to_email
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await reply_to_email(
                user_id=user_id,
                message_id=request.message_id,
                body=request.body,
                reply_all=request.reply_all,
                connection_id=connection_id if connection_id else None,
            )
            return tool_service_pb2.ReplyToEmailResponse(success=True, result=result)
        except Exception as e:
            logger.error("ReplyToEmail failed: %s", e)
            return tool_service_pb2.ReplyToEmailResponse(
                success=False, result="", error=str(e)
            )

    async def GetEmailFolders(
        self,
        request: tool_service_pb2.GetEmailFoldersRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetEmailFoldersResponse:
        """List email folders for user."""
        try:
            from clients.connections_service_client import get_connections_service_client
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            client = await get_connections_service_client()
            data = await client.get_folders(
                user_id=user_id,
                connection_id=connection_id if connection_id else None,
            )
            if data.get("error") and not data.get("folders"):
                return tool_service_pb2.GetEmailFoldersResponse(
                    success=False, result="", error=data.get("error", "No connection")
                )
            folders = data.get("folders", [])
            lines = [
                f"- {f.get('name')} (id={f.get('id')}, unread={f.get('unread_count', 0)})"
                for f in folders
            ]
            result = "\n".join(lines) if lines else "No folders."
            return tool_service_pb2.GetEmailFoldersResponse(success=True, result=result)
        except Exception as e:
            logger.error("GetEmailFolders failed: %s", e)
            return tool_service_pb2.GetEmailFoldersResponse(
                success=False, result="", error=str(e)
            )

    async def GetEmailStatistics(
        self,
        request: tool_service_pb2.GetEmailStatisticsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetEmailStatisticsResponse:
        """Get email statistics (inbox total/unread)."""
        try:
            from services.langgraph_tools.email_tools import get_email_statistics
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await get_email_statistics(
                user_id=user_id,
                connection_id=connection_id if connection_id else None,
            )
            return tool_service_pb2.GetEmailStatisticsResponse(success=True, result=result)
        except Exception as e:
            logger.error("GetEmailStatistics failed: %s", e)
            return tool_service_pb2.GetEmailStatisticsResponse(
                success=False, result="", error=str(e)
            )

    async def MarkEmailRead(
        self,
        request: tool_service_pb2.MarkEmailReadRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.MarkEmailReadResponse:
        """Mark an email as read."""
        try:
            from services.langgraph_tools.email_tools import mark_email_as_read
            user_id = request.user_id or "system"
            result = await mark_email_as_read(
                user_id=user_id,
                message_id=request.message_id,
            )
            return tool_service_pb2.MarkEmailReadResponse(success=True, result=result)
        except Exception as e:
            logger.error("MarkEmailRead failed: %s", e)
            return tool_service_pb2.MarkEmailReadResponse(
                success=False, result="", error=str(e)
            )

    async def GetEmailById(
        self,
        request: tool_service_pb2.GetEmailByIdRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetEmailByIdResponse:
        """Get a single email by message ID (full content)."""
        try:
            from clients.connections_service_client import get_connections_service_client
            user_id = request.user_id or "system"
            message_id = request.message_id or ""
            connection_id = getattr(request, "connection_id", 0) or 0
            if not message_id:
                return tool_service_pb2.GetEmailByIdResponse(
                    success=False, result="", error="message_id is required"
                )
            client = await get_connections_service_client()
            data = await client.get_email_by_id(
                user_id=user_id,
                message_id=message_id,
                connection_id=connection_id if connection_id else None,
            )
            if data.get("error") and not data.get("message"):
                return tool_service_pb2.GetEmailByIdResponse(
                    success=False, result="", error=data.get("error", "Not found")
                )
            msg = data.get("message", {})
            parts = [
                f"Subject: {msg.get('subject', '')}",
                f"From: {msg.get('from_name', '')} <{msg.get('from_address', '')}>",
                f"To: {', '.join(msg.get('to_addresses') or [])}",
                f"Date: {msg.get('received_datetime', '')}",
                f"Read: {msg.get('is_read', False)}",
                f"Has attachments: {msg.get('has_attachments', False)}",
            ]
            body = msg.get("body_content") or msg.get("body_preview") or ""
            if body:
                parts.append(f"\nBody:\n{body}")
            result = "\n".join(parts)
            return tool_service_pb2.GetEmailByIdResponse(success=True, result=result)
        except Exception as e:
            logger.error("GetEmailById failed: %s", e)
            return tool_service_pb2.GetEmailByIdResponse(
                success=False, result="", error=str(e)
            )

    async def MoveEmail(
        self,
        request: tool_service_pb2.MoveEmailRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.MoveEmailResponse:
        """Move an email to a different folder."""
        try:
            from clients.connections_service_client import get_connections_service_client
            user_id = request.user_id or "system"
            message_id = request.message_id or ""
            destination_folder_id = request.destination_folder_id or ""
            connection_id = getattr(request, "connection_id", 0) or 0
            if not message_id or not destination_folder_id:
                return tool_service_pb2.MoveEmailResponse(
                    success=False, result="", error="message_id and destination_folder_id are required"
                )
            client = await get_connections_service_client()
            data = await client.move_email(
                user_id=user_id,
                message_id=message_id,
                destination_folder_id=destination_folder_id,
                connection_id=connection_id if connection_id else None,
            )
            if data.get("success"):
                result = f"Moved email to folder {destination_folder_id}"
                return tool_service_pb2.MoveEmailResponse(success=True, result=result)
            return tool_service_pb2.MoveEmailResponse(
                success=False, result="", error=data.get("error", "Move failed")
            )
        except Exception as e:
            logger.error("MoveEmail failed: %s", e)
            return tool_service_pb2.MoveEmailResponse(
                success=False, result="", error=str(e)
            )

    async def UpdateEmail(
        self,
        request: tool_service_pb2.UpdateEmailRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateEmailResponse:
        """Update an email (mark read/unread, set importance)."""
        try:
            from clients.connections_service_client import get_connections_service_client
            user_id = request.user_id or "system"
            message_id = request.message_id or ""
            connection_id = getattr(request, "connection_id", 0) or 0
            if not message_id:
                return tool_service_pb2.UpdateEmailResponse(
                    success=False, result="", error="message_id is required"
                )
            is_read = request.is_read if request.HasField("is_read") else None
            importance = request.importance if request.importance else None
            if is_read is None and not importance:
                return tool_service_pb2.UpdateEmailResponse(
                    success=False, result="", error="At least one of is_read or importance is required"
                )
            client = await get_connections_service_client()
            data = await client.update_email(
                user_id=user_id,
                message_id=message_id,
                is_read=is_read,
                importance=importance or None,
                connection_id=connection_id if connection_id else None,
            )
            if data.get("success"):
                result = "Email updated."
                return tool_service_pb2.UpdateEmailResponse(success=True, result=result)
            return tool_service_pb2.UpdateEmailResponse(
                success=False, result="", error=data.get("error", "Update failed")
            )
        except Exception as e:
            logger.error("UpdateEmail failed: %s", e)
            return tool_service_pb2.UpdateEmailResponse(
                success=False, result="", error=str(e)
            )

    async def CreateDraft(
        self,
        request: tool_service_pb2.CreateDraftRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateDraftResponse:
        """Create a draft email (do not send)."""
        try:
            from clients.connections_service_client import get_connections_service_client
            user_id = request.user_id or "system"
            to_list = list(request.to) if request.to else []
            subject = request.subject or ""
            body = request.body or ""
            cc_list = list(request.cc) if request.cc else []
            connection_id = getattr(request, "connection_id", 0) or 0
            if not to_list:
                return tool_service_pb2.CreateDraftResponse(
                    success=False, result="", error="At least one recipient (to) is required"
                )
            client = await get_connections_service_client()
            data = await client.create_draft(
                user_id=user_id,
                to_recipients=to_list,
                subject=subject,
                body=body,
                cc_recipients=cc_list if cc_list else None,
                connection_id=connection_id if connection_id else None,
            )
            if data.get("success"):
                msg_id = data.get("message_id", "")
                result = f"Draft created (ID: {msg_id})" if msg_id else "Draft created."
                return tool_service_pb2.CreateDraftResponse(success=True, result=result)
            return tool_service_pb2.CreateDraftResponse(
                success=False, result="", error=data.get("error", "Create draft failed")
            )
        except Exception as e:
            logger.error("CreateDraft failed: %s", e)
            return tool_service_pb2.CreateDraftResponse(
                success=False, result="", error=str(e)
            )

    async def ListCalendars(
        self,
        request: tool_service_pb2.ListCalendarsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListCalendarsResponse:
        """List user's calendars."""
        try:
            from services.langgraph_tools.calendar_tools import list_calendars
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await list_calendars(
                user_id=user_id,
                connection_id=connection_id if connection_id else None,
            )
            return tool_service_pb2.ListCalendarsResponse(success=True, result=result)
        except Exception as e:
            logger.error("ListCalendars failed: %s", e)
            return tool_service_pb2.ListCalendarsResponse(
                success=False, result="", error=str(e)
            )

    async def GetCalendarEvents(
        self,
        request: tool_service_pb2.GetCalendarEventsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetCalendarEventsResponse:
        """Get calendar events in date range."""
        try:
            from services.langgraph_tools.calendar_tools import get_calendar_events
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await get_calendar_events(
                user_id=user_id,
                start_datetime=request.start_datetime or "",
                end_datetime=request.end_datetime or "",
                calendar_id=request.calendar_id or "",
                top=request.top or 50,
                connection_id=connection_id if connection_id else None,
            )
            return tool_service_pb2.GetCalendarEventsResponse(success=True, result=result)
        except Exception as e:
            logger.error("GetCalendarEvents failed: %s", e)
            return tool_service_pb2.GetCalendarEventsResponse(
                success=False, result="", error=str(e)
            )

    async def GetCalendarEventById(
        self,
        request: tool_service_pb2.GetCalendarEventByIdRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetCalendarEventByIdResponse:
        """Get single calendar event by ID."""
        try:
            from services.langgraph_tools.calendar_tools import get_event_by_id
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await get_event_by_id(
                user_id=user_id,
                event_id=request.event_id or "",
                connection_id=connection_id if connection_id else None,
            )
            return tool_service_pb2.GetCalendarEventByIdResponse(success=True, result=result)
        except Exception as e:
            logger.error("GetCalendarEventById failed: %s", e)
            return tool_service_pb2.GetCalendarEventByIdResponse(
                success=False, result="", error=str(e)
            )

    async def CreateCalendarEvent(
        self,
        request: tool_service_pb2.CreateCalendarEventRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateCalendarEventResponse:
        """Create a calendar event."""
        try:
            from services.langgraph_tools.calendar_tools import create_event
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await create_event(
                user_id=user_id,
                subject=request.subject or "",
                start_datetime=request.start_datetime or "",
                end_datetime=request.end_datetime or "",
                connection_id=connection_id if connection_id else None,
                calendar_id=request.calendar_id or "",
                location=request.location or "",
                body=request.body or "",
                body_is_html=request.body_is_html,
                attendee_emails=list(request.attendee_emails) if request.attendee_emails else None,
                is_all_day=request.is_all_day,
            )
            if "successfully" in result and "Error" not in result:
                return tool_service_pb2.CreateCalendarEventResponse(success=True, result=result)
            return tool_service_pb2.CreateCalendarEventResponse(
                success=False, result=result, error=result
            )
        except Exception as e:
            logger.error("CreateCalendarEvent failed: %s", e)
            return tool_service_pb2.CreateCalendarEventResponse(
                success=False, result="", error=str(e)
            )

    async def UpdateCalendarEvent(
        self,
        request: tool_service_pb2.UpdateCalendarEventRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateCalendarEventResponse:
        """Update a calendar event."""
        try:
            from services.langgraph_tools.calendar_tools import update_event
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await update_event(
                user_id=user_id,
                event_id=request.event_id or "",
                connection_id=connection_id if connection_id else None,
                subject=request.subject if request.subject else None,
                start_datetime=request.start_datetime if request.start_datetime else None,
                end_datetime=request.end_datetime if request.end_datetime else None,
                location=request.location if request.location else None,
                body=request.body if request.body else None,
                body_is_html=request.body_is_html,
                attendee_emails=list(request.attendee_emails) if request.attendee_emails else None,
                is_all_day=request.is_all_day,
            )
            if result == "Event updated successfully.":
                return tool_service_pb2.UpdateCalendarEventResponse(success=True, result=result)
            return tool_service_pb2.UpdateCalendarEventResponse(
                success=False, result=result, error=result
            )
        except Exception as e:
            logger.error("UpdateCalendarEvent failed: %s", e)
            return tool_service_pb2.UpdateCalendarEventResponse(
                success=False, result="", error=str(e)
            )

    async def DeleteCalendarEvent(
        self,
        request: tool_service_pb2.DeleteCalendarEventRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.DeleteCalendarEventResponse:
        """Delete a calendar event."""
        try:
            from services.langgraph_tools.calendar_tools import delete_event
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await delete_event(
                user_id=user_id,
                event_id=request.event_id or "",
                connection_id=connection_id if connection_id else None,
            )
            if result == "Event deleted successfully.":
                return tool_service_pb2.DeleteCalendarEventResponse(success=True, result=result)
            return tool_service_pb2.DeleteCalendarEventResponse(
                success=False, result=result, error=result
            )
        except Exception as e:
            logger.error("DeleteCalendarEvent failed: %s", e)
            return tool_service_pb2.DeleteCalendarEventResponse(
                success=False, result="", error=str(e)
            )

    async def GetContacts(
        self,
        request: tool_service_pb2.GetContactsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetContactsResponse:
        """Get contacts. sources: all (O365+org), microsoft, org, caldav."""
        try:
            from services.langgraph_tools.contact_tools import (
                get_contacts,
                get_contacts_unified,
                _get_org_contacts_for_tool,
                _format_contacts,
            )
            import json

            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            sources = (getattr(request, "sources", None) or "all").strip().lower() or "all"
            folder_id = request.folder_id or ""
            top = request.top or 100

            if sources == "microsoft":
                result = await get_contacts(
                    user_id=user_id,
                    connection_id=connection_id if connection_id else None,
                    folder_id=folder_id,
                    top=top,
                )
            elif sources == "org":
                org_list = await _get_org_contacts_for_tool(user_id, limit=top)
                formatted = _format_contacts(org_list, max_items=top, include_source=True)
                result = json.dumps({"contacts": org_list, "formatted": formatted})
            else:
                result = await get_contacts_unified(
                    user_id=user_id,
                    connection_id=connection_id if connection_id else None,
                    folder_id=folder_id,
                    top=top,
                )
            return tool_service_pb2.GetContactsResponse(success=True, result=result)
        except Exception as e:
            logger.error("GetContacts failed: %s", e)
            return tool_service_pb2.GetContactsResponse(
                success=False, result="", error=str(e)
            )

    async def GetContactById(
        self,
        request: tool_service_pb2.GetContactByIdRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetContactByIdResponse:
        """Get single O365 contact by ID."""
        try:
            from services.langgraph_tools.contact_tools import get_contact_by_id
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await get_contact_by_id(
                user_id=user_id,
                contact_id=request.contact_id or "",
                connection_id=connection_id if connection_id else None,
            )
            return tool_service_pb2.GetContactByIdResponse(success=True, result=result)
        except Exception as e:
            logger.error("GetContactById failed: %s", e)
            return tool_service_pb2.GetContactByIdResponse(
                success=False, result="", error=str(e)
            )

    async def CreateContact(
        self,
        request: tool_service_pb2.CreateContactRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateContactResponse:
        """Create an O365 contact."""
        try:
            import json
            from services.langgraph_tools.contact_tools import create_contact
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            email_addresses = None
            if request.email_addresses_json:
                try:
                    email_addresses = json.loads(request.email_addresses_json)
                except json.JSONDecodeError:
                    pass
            phone_numbers = None
            if request.phone_numbers_json:
                try:
                    phone_numbers = json.loads(request.phone_numbers_json)
                except json.JSONDecodeError:
                    pass
            result = await create_contact(
                user_id=user_id,
                display_name=request.display_name or "",
                given_name=request.given_name or "",
                surname=request.surname or "",
                connection_id=connection_id if connection_id else None,
                folder_id=request.folder_id or "",
                email_addresses=email_addresses,
                phone_numbers=phone_numbers,
                company_name=request.company_name or "",
                job_title=request.job_title or "",
                birthday=request.birthday or "",
                notes=request.notes or "",
            )
            if "successfully" in result and "Error" not in result:
                return tool_service_pb2.CreateContactResponse(success=True, result=result)
            return tool_service_pb2.CreateContactResponse(
                success=False, result=result, error=result
            )
        except Exception as e:
            logger.error("CreateContact failed: %s", e)
            return tool_service_pb2.CreateContactResponse(
                success=False, result="", error=str(e)
            )

    async def UpdateContact(
        self,
        request: tool_service_pb2.UpdateContactRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateContactResponse:
        """Update an O365 contact."""
        try:
            import json
            from services.langgraph_tools.contact_tools import update_contact
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            email_addresses = None
            if request.email_addresses_json:
                try:
                    email_addresses = json.loads(request.email_addresses_json)
                except json.JSONDecodeError:
                    pass
            phone_numbers = None
            if request.phone_numbers_json:
                try:
                    phone_numbers = json.loads(request.phone_numbers_json)
                except json.JSONDecodeError:
                    pass
            result = await update_contact(
                user_id=user_id,
                contact_id=request.contact_id or "",
                connection_id=connection_id if connection_id else None,
                display_name=request.display_name if request.display_name else None,
                given_name=request.given_name if request.given_name else None,
                surname=request.surname if request.surname else None,
                email_addresses=email_addresses,
                phone_numbers=phone_numbers,
                company_name=request.company_name if request.company_name else None,
                job_title=request.job_title if request.job_title else None,
                birthday=request.birthday if request.birthday else None,
                notes=request.notes if request.notes else None,
            )
            if result == "Contact updated successfully.":
                return tool_service_pb2.UpdateContactResponse(success=True, result=result)
            return tool_service_pb2.UpdateContactResponse(
                success=False, result=result, error=result
            )
        except Exception as e:
            logger.error("UpdateContact failed: %s", e)
            return tool_service_pb2.UpdateContactResponse(
                success=False, result="", error=str(e)
            )

    async def DeleteContact(
        self,
        request: tool_service_pb2.DeleteContactRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.DeleteContactResponse:
        """Delete an O365 contact."""
        try:
            from services.langgraph_tools.contact_tools import delete_contact
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await delete_contact(
                user_id=user_id,
                contact_id=request.contact_id or "",
                connection_id=connection_id if connection_id else None,
            )
            if result == "Contact deleted successfully.":
                return tool_service_pb2.DeleteContactResponse(success=True, result=result)
            return tool_service_pb2.DeleteContactResponse(
                success=False, result=result, error=result
            )
        except Exception as e:
            logger.error("DeleteContact failed: %s", e)
            return tool_service_pb2.DeleteContactResponse(
                success=False, result="", error=str(e)
            )

    async def SearchContacts(
        self,
        request: tool_service_pb2.SearchContactsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SearchContactsResponse:
        """Search contacts by query (substring match on name, email, company)."""
        try:
            from services.langgraph_tools.contact_tools import search_contacts as search_contacts_impl

            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            query = (request.query or "").strip()
            sources = (request.sources or "all").strip().lower() or "all"
            top = request.top or 20
            result = await search_contacts_impl(
                user_id=user_id,
                query=query,
                sources=sources,
                top=top,
                connection_id=connection_id if connection_id else None,
            )
            return tool_service_pb2.SearchContactsResponse(success=True, result=result)
        except Exception as e:
            logger.error("SearchContacts failed: %s", e)
            return tool_service_pb2.SearchContactsResponse(
                success=False, result="", error=str(e)
            )

    async def ListUserAccounts(
        self,
        request: tool_service_pb2.ListUserAccountsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListUserAccountsResponse:
        """List email/calendar/contacts accounts for the user, optionally scoped to agent profile bindings."""
        try:
            from services.database_manager.database_helpers import fetch_all

            user_id = request.user_id or "system"
            profile_id = (request.agent_profile_id or "").strip()
            service_type = (request.service_type or "all").strip().lower() or "all"

            if profile_id:
                try:
                    profile_uuid = uuid.UUID(profile_id)
                except ValueError:
                    return tool_service_pb2.ListUserAccountsResponse(
                        success=False, result="[]", error="Invalid agent_profile_id"
                    )
                if service_type == "all":
                    rows = await fetch_all(
                        """
                        SELECT ec.id, ec.provider, ec.connection_type, ec.account_identifier,
                               ec.display_name
                        FROM agent_service_bindings asb
                        JOIN external_connections ec ON ec.id = asb.connection_id AND ec.user_id = $2
                        WHERE asb.agent_profile_id = $1 AND asb.is_enabled = true
                          AND ec.is_active = true
                        ORDER BY ec.connection_type, ec.provider
                        """,
                        profile_uuid,
                        user_id,
                    )
                else:
                    rows = await fetch_all(
                        """
                        SELECT ec.id, ec.provider, ec.connection_type, ec.account_identifier,
                               ec.display_name
                        FROM agent_service_bindings asb
                        JOIN external_connections ec ON ec.id = asb.connection_id AND ec.user_id = $2
                        WHERE asb.agent_profile_id = $1 AND asb.is_enabled = true
                          AND ec.is_active = true AND ec.connection_type = $3
                        ORDER BY ec.connection_type, ec.provider
                        """,
                        profile_uuid,
                        user_id,
                        service_type,
                    )
            else:
                if service_type == "all":
                    rows = await fetch_all(
                        """
                        SELECT id, provider, connection_type, account_identifier, display_name
                        FROM external_connections
                        WHERE user_id = $1 AND is_active = true
                        ORDER BY connection_type, provider
                        """,
                        user_id,
                    )
                else:
                    rows = await fetch_all(
                        """
                        SELECT id, provider, connection_type, account_identifier, display_name
                        FROM external_connections
                        WHERE user_id = $1 AND is_active = true AND connection_type = $2
                        ORDER BY connection_type, provider
                        """,
                        user_id,
                        service_type,
                    )

            accounts = [
                {
                    "connection_id": int(r["id"]),
                    "provider": r["provider"] or "",
                    "type": r["connection_type"] or "",
                    "label": (r.get("display_name") or r.get("account_identifier") or "").strip()
                    or (r.get("account_identifier") or ""),
                    "address": r.get("account_identifier") or "",
                }
                for r in rows
            ]
            return tool_service_pb2.ListUserAccountsResponse(
                success=True, result=json.dumps(accounts)
            )
        except Exception as e:
            logger.error("ListUserAccounts failed: %s", e)
            return tool_service_pb2.ListUserAccountsResponse(
                success=False, result="[]", error=str(e)
            )

    async def GetAgentProfile(
        self,
        request: tool_service_pb2.GetAgentProfileRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetAgentProfileResponse:
        """Return agent profile by ID for custom agent execution."""
        try:
            from services.database_manager.database_helpers import fetch_one
            user_id = request.user_id or "system"
            profile_id = request.profile_id
            if not profile_id:
                return tool_service_pb2.GetAgentProfileResponse(
                    success=False, profile_json="", error="profile_id required"
                )
            row = await fetch_one(
                "SELECT * FROM agent_profiles WHERE id = $1 AND user_id = $2",
                uuid.UUID(profile_id),
                user_id,
            )
            if not row:
                return tool_service_pb2.GetAgentProfileResponse(
                    success=False, profile_json="", error="Profile not found"
                )
            persona_mode = row.get("persona_mode") or "none"
            persona_id = str(row["persona_id"]) if row.get("persona_id") else None
            profile = {
                "id": str(row["id"]),
                "user_id": row["user_id"],
                "name": row["name"],
                "handle": row["handle"],
                "description": row.get("description"),
                "is_active": row.get("is_active", True),
                "model_preference": row.get("model_preference"),
                "max_research_rounds": row.get("max_research_rounds", 3),
                "system_prompt_additions": row.get("system_prompt_additions"),
                "knowledge_config": row.get("knowledge_config") or {},
                "default_playbook_id": str(row["default_playbook_id"]) if row.get("default_playbook_id") else None,
                "default_run_context": row.get("default_run_context") or "interactive",
                "default_approval_policy": row.get("default_approval_policy") or "require",
                "journal_config": row.get("journal_config") or {},
                "team_config": row.get("team_config") or {},
                "prompt_history_enabled": row.get("chat_history_enabled", False),
                "chat_history_lookback": row.get("chat_history_lookback", 10),
                "summary_threshold_tokens": row.get("summary_threshold_tokens", 5000),
                "summary_keep_messages": row.get("summary_keep_messages", 10),
                "persona_mode": persona_mode,
                "persona_id": persona_id,
                "include_user_context": row.get("include_user_context", False),
                "include_datetime_context": row.get("include_datetime_context", True),
                "include_user_facts": row.get("include_user_facts", False),
                "include_facts_categories": list(row.get("include_facts_categories") or []),
                "include_agent_memory": row.get("include_agent_memory", False),
                "auto_routable": row.get("auto_routable", False),
                "data_workspace_config": row.get("data_workspace_config") or {},
                "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
                "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
            }
            if persona_mode == "specific" and persona_id:
                from services.settings_service import settings_service
                persona = await settings_service.get_persona_by_id(persona_id, user_id)
                if persona:
                    profile["persona"] = persona
            return tool_service_pb2.GetAgentProfileResponse(
                success=True,
                profile_json=json.dumps(profile, default=_json_default),
            )
        except Exception as e:
            logger.exception("GetAgentProfile failed")
            return tool_service_pb2.GetAgentProfileResponse(
                success=False, profile_json="", error=str(e)
            )

    async def ListAutoRoutableProfiles(
        self,
        request: tool_service_pb2.ListAutoRoutableProfilesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListAutoRoutableProfilesResponse:
        """Return agent profiles for the user where auto_routable=true and is_active=true."""
        try:
            from services.database_manager.database_helpers import fetch_all
            user_id = request.user_id or "system"
            rows = await fetch_all(
                "SELECT id, handle, name, description FROM agent_profiles "
                "WHERE user_id = $1 AND is_active = true AND auto_routable = true ORDER BY name",
                user_id,
            )
            profiles = [
                {
                    "id": str(r["id"]),
                    "handle": r["handle"],
                    "name": r["name"],
                    "description": r.get("description") or "",
                }
                for r in rows
            ]
            return tool_service_pb2.ListAutoRoutableProfilesResponse(
                success=True,
                profiles_json=json.dumps(profiles),
            )
        except Exception as e:
            logger.exception("ListAutoRoutableProfiles failed")
            return tool_service_pb2.ListAutoRoutableProfilesResponse(
                success=False, profiles_json="[]", error=str(e)
            )

    async def ResolveAgentHandle(
        self,
        request: tool_service_pb2.ResolveAgentHandleRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ResolveAgentHandleResponse:
        """Resolve agent target string to agent_profile_id: handle, @handle, UUID, or display name (unique)."""
        try:
            from services.database_manager.database_helpers import fetch_all, fetch_one
            user_id = request.user_id or "system"
            raw = (request.handle or "").strip()
            if raw.startswith("@"):
                raw = raw[1:].strip()
            if not raw:
                return tool_service_pb2.ResolveAgentHandleResponse(found=False)

            # 1) Exact handle match
            row = await fetch_one(
                "SELECT id, name FROM agent_profiles WHERE user_id = $1 AND handle = $2 AND is_active = true",
                user_id,
                raw,
            )
            if row:
                return tool_service_pb2.ResolveAgentHandleResponse(
                    agent_profile_id=str(row["id"]),
                    agent_name=row.get("name") or raw,
                    found=True,
                )

            # 2) Agent profile UUID (workers sometimes paste ids from briefings)
            try:
                uid = uuid.UUID(raw)
                row = await fetch_one(
                    "SELECT id, name FROM agent_profiles WHERE user_id = $1 AND id = $2 AND is_active = true",
                    user_id,
                    uid,
                )
                if row:
                    return tool_service_pb2.ResolveAgentHandleResponse(
                        agent_profile_id=str(row["id"]),
                        agent_name=row.get("name") or raw,
                        found=True,
                    )
            except (ValueError, TypeError):
                pass

            # 3) Unique display name (case-insensitive) — matches "send to your manager (Name)" style prompts
            rows = await fetch_all(
                """
                SELECT id, name FROM agent_profiles
                WHERE user_id = $1 AND is_active = true
                  AND LOWER(TRIM(COALESCE(name, ''))) = LOWER(TRIM($2))
                """,
                user_id,
                raw,
            )
            if len(rows) == 1:
                row = rows[0]
                return tool_service_pb2.ResolveAgentHandleResponse(
                    agent_profile_id=str(row["id"]),
                    agent_name=row.get("name") or raw,
                    found=True,
                )
            if len(rows) > 1:
                logger.warning(
                    "ResolveAgentHandle: ambiguous display name %r matches %s profiles for user_id=%s",
                    raw[:80],
                    len(rows),
                    user_id,
                )
            return tool_service_pb2.ResolveAgentHandleResponse(found=False)
        except Exception as e:
            logger.exception("ResolveAgentHandle failed")
            return tool_service_pb2.ResolveAgentHandleResponse(found=False)

    async def EnqueueAgentInvocation(
        self,
        request: tool_service_pb2.EnqueueAgentInvocationRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.EnqueueAgentInvocationResponse:
        """Enqueue async agent-to-agent invocation via Celery. Used by output_router."""
        try:
            from services.celery_tasks.agent_tasks import dispatch_agent_invocation
            agent_profile_id = request.agent_profile_id or ""
            input_content = request.input_content or ""
            user_id = request.user_id or "system"
            source_agent_name = request.source_agent_name or ""
            chain_depth = request.chain_depth or 0
            chain_path_json = request.chain_path_json or "[]"
            if not agent_profile_id or not input_content:
                return tool_service_pb2.EnqueueAgentInvocationResponse(
                    success=False,
                    error="agent_profile_id and input_content required",
                )
            task = dispatch_agent_invocation.apply_async(
                args=[
                    agent_profile_id,
                    input_content,
                    user_id,
                    source_agent_name,
                    chain_depth,
                    chain_path_json,
                ],
            )
            return tool_service_pb2.EnqueueAgentInvocationResponse(
                success=True,
                task_id=task.id or "",
            )
        except Exception as e:
            logger.exception("EnqueueAgentInvocation failed")
            return tool_service_pb2.EnqueueAgentInvocationResponse(
                success=False,
                error=str(e),
            )

    async def ReadTeamPosts(
        self,
        request: tool_service_pb2.ReadTeamPostsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ReadTeamPostsResponse:
        """Return team posts (optionally since last_read_at for the user). Used by read_team_posts tool."""
        try:
            from services.database_manager.database_helpers import fetch_one, fetch_all
            from services.team_post_service import TeamPostService
            from services.team_service import TeamService
            team_id = (request.team_id or "").strip()
            user_id = request.user_id or "system"
            since_last_read = request.since_last_read
            limit = max(1, min(100, request.limit or 20))
            mark_as_read = request.mark_as_read
            if not team_id:
                return tool_service_pb2.ReadTeamPostsResponse(
                    success=False,
                    error="team_id required",
                )
            team_service = TeamService()
            await team_service.initialize()
            team_post_service = TeamPostService()
            await team_post_service.initialize(team_service=team_service)
            since_ts = None
            if since_last_read:
                row = await fetch_one(
                    "SELECT last_read_at FROM team_members WHERE team_id = $1 AND user_id = $2",
                    uuid.UUID(team_id),
                    user_id,
                )
                if row and row.get("last_read_at"):
                    since_ts = row["last_read_at"]
            posts_result = await team_post_service.get_team_posts_since(
                team_id=team_id,
                user_id=user_id,
                since_ts=since_ts,
                limit=limit,
            )
            team_row = await fetch_one(
                "SELECT team_name FROM teams WHERE team_id = $1",
                uuid.UUID(team_id),
            )
            team_name = (team_row.get("team_name") or "") if team_row else ""
            if mark_as_read and posts_result:
                await team_service.mark_team_posts_as_read(team_id, user_id)
            out_posts = []
            for p in posts_result:
                out_posts.append(
                    tool_service_pb2.TeamPost(
                        post_id=p.get("post_id", ""),
                        author_id=p.get("author_id", ""),
                        author_name=p.get("author_name") or p.get("author_display_name") or p.get("author_username") or "",
                        content=p.get("content", ""),
                        post_type=p.get("post_type", "text"),
                        created_at=p.get("created_at").isoformat() if p.get("created_at") else "",
                    )
                )
            return tool_service_pb2.ReadTeamPostsResponse(
                posts=out_posts,
                count=len(out_posts),
                team_name=team_name,
                success=True,
            )
        except PermissionError as e:
            return tool_service_pb2.ReadTeamPostsResponse(
                success=False,
                error=str(e),
            )
        except Exception as e:
            logger.exception("ReadTeamPosts failed")
            return tool_service_pb2.ReadTeamPostsResponse(
                success=False,
                error=str(e),
            )

    async def CreateTeamPost(
        self,
        request: tool_service_pb2.CreateTeamPostRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateTeamPostResponse:
        """Create a team post or comment. Used by post_to_team tool and output_router team_post destination."""
        try:
            from services.team_post_service import TeamPostService
            from services.team_service import TeamService
            team_id = (request.team_id or "").strip()
            user_id = request.user_id or "system"
            content = (request.content or "").strip()
            post_type = (request.post_type or "text").strip() or "text"
            reply_to_post_id = (request.reply_to_post_id or "").strip()
            if not team_id or not content:
                return tool_service_pb2.CreateTeamPostResponse(
                    success=False,
                    error="team_id and content required",
                )
            team_service = TeamService()
            await team_service.initialize()
            team_post_service = TeamPostService()
            await team_post_service.initialize(team_service=team_service)
            if reply_to_post_id:
                comment = await team_post_service.create_comment(
                    post_id=reply_to_post_id,
                    author_id=user_id,
                    content=content,
                )
                post_id = comment.get("comment_id", "")
            else:
                post = await team_post_service.create_post(
                    team_id=team_id,
                    author_id=user_id,
                    content=content,
                    post_type=post_type,
                    attachments=None,
                )
                post_id = post.get("post_id", "")
            return tool_service_pb2.CreateTeamPostResponse(
                post_id=post_id,
                success=True,
            )
        except PermissionError as e:
            return tool_service_pb2.CreateTeamPostResponse(
                success=False,
                error=str(e),
            )
        except Exception as e:
            logger.exception("CreateTeamPost failed")
            return tool_service_pb2.CreateTeamPostResponse(
                success=False,
                error=str(e),
            )

    async def GetPlaybook(
        self,
        request: tool_service_pb2.GetPlaybookRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetPlaybookResponse:
        """Return playbook by ID for custom agent execution."""
        try:
            from services.database_manager.database_helpers import fetch_one
            user_id = request.user_id or "system"
            playbook_id = (request.playbook_id or "").strip()
            if not playbook_id:
                return tool_service_pb2.GetPlaybookResponse(
                    success=False, playbook_json="", error="playbook_id required"
                )
            try:
                uuid.UUID(playbook_id)
            except (ValueError, TypeError):
                slug_normalized = playbook_id.replace("_", "-").lower()
                resolved = await fetch_one(
                    """SELECT id FROM custom_playbooks
                       WHERE user_id = $1 AND (
                         name = $2
                         OR LOWER(REGEXP_REPLACE(TRIM(name), '\\s+', '-')) = $3
                       )
                       LIMIT 1""",
                    user_id,
                    playbook_id,
                    slug_normalized,
                )
                if not resolved:
                    return tool_service_pb2.GetPlaybookResponse(
                        success=False, playbook_json="",
                        error="Playbook not found (use id from list_playbooks, or exact name)",
                    )
                playbook_id = str(resolved["id"])
            row = await fetch_one(
                "SELECT * FROM custom_playbooks WHERE id = $1 AND (user_id = $2 OR (user_id IS NULL AND is_builtin = true))",
                uuid.UUID(playbook_id),
                user_id,
            )
            if not row:
                return tool_service_pb2.GetPlaybookResponse(
                    success=False, playbook_json="", error="Playbook not found"
                )
            raw_def = row.get("definition") or {}
            if isinstance(raw_def, str):
                try:
                    raw_def = json.loads(raw_def) if raw_def else {}
                except (json.JSONDecodeError, TypeError):
                    raw_def = {}
            if not isinstance(raw_def, dict):
                raw_def = {}
            raw_triggers = row.get("triggers") or []
            if isinstance(raw_triggers, str):
                try:
                    raw_triggers = json.loads(raw_triggers) if raw_triggers else []
                except (json.JSONDecodeError, TypeError):
                    raw_triggers = []
            if not isinstance(raw_triggers, list):
                raw_triggers = []
            playbook = {
                "id": str(row["id"]),
                "user_id": row["user_id"],
                "name": row["name"],
                "description": row.get("description"),
                "version": row.get("version", "1.0"),
                "definition": raw_def,
                "triggers": raw_triggers,
                "is_template": row.get("is_template", False),
                "is_locked": row.get("is_locked", False),
                "category": row.get("category"),
                "tags": list(row.get("tags") or []),
                "required_connectors": list(row.get("required_connectors") or []),
                "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
                "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
            }
            return tool_service_pb2.GetPlaybookResponse(
                success=True,
                playbook_json=json.dumps(playbook),
            )
        except Exception as e:
            logger.exception("GetPlaybook failed")
            return tool_service_pb2.GetPlaybookResponse(
                success=False, playbook_json="", error=str(e)
            )

    async def GetSkillsByIds(
        self,
        request: tool_service_pb2.GetSkillsByIdsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetSkillsByIdsResponse:
        """Batch fetch skills by IDs for pipeline skill injection."""
        try:
            from services.agent_skills_service import get_skills_by_ids
            user_id = request.user_id or "system"
            skill_ids = list(request.skill_ids) if request.skill_ids else []
            skills = await get_skills_by_ids(skill_ids)
            return tool_service_pb2.GetSkillsByIdsResponse(
                success=True,
                skills_json=json.dumps(skills),
            )
        except Exception as e:
            logger.exception("GetSkillsByIds failed")
            return tool_service_pb2.GetSkillsByIdsResponse(
                success=False, skills_json="[]", error=str(e)
            )

    async def SearchSkills(
        self,
        request: tool_service_pb2.SearchSkillsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SearchSkillsResponse:
        """Semantic search over skills for auto-discovery at step invocation."""
        try:
            from services.skill_vector_service import search_skills
            user_id = request.user_id or "system"
            query = (request.query or "").strip()
            limit = request.limit or 3
            score_threshold = request.score_threshold if request.score_threshold > 0 else 0.5
            results = await search_skills(
                user_id=user_id,
                query=query,
                limit=limit,
                score_threshold=score_threshold,
            )
            return tool_service_pb2.SearchSkillsResponse(
                success=True,
                skills_json=json.dumps(results),
            )
        except Exception as e:
            logger.exception("SearchSkills failed")
            return tool_service_pb2.SearchSkillsResponse(
                success=False, skills_json="[]", error=str(e)
            )

    async def ListSkills(
        self,
        request: tool_service_pb2.ListSkillsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListSkillsResponse:
        """List user and optionally built-in skills for agent self-awareness."""
        try:
            from services.agent_skills_service import list_skills
            user_id = request.user_id or "system"
            category = request.category or None
            include_builtin = getattr(request, "include_builtin", True)
            skills = await list_skills(user_id, category=category, include_builtin=include_builtin)
            return tool_service_pb2.ListSkillsResponse(
                success=True,
                skills_json=json.dumps(skills),
            )
        except Exception as e:
            logger.exception("ListSkills failed")
            return tool_service_pb2.ListSkillsResponse(
                success=False, skills_json="[]", error=str(e)
            )

    async def CreateSkill(
        self,
        request: tool_service_pb2.CreateSkillRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateSkillResponse:
        """Create a user-authored skill."""
        try:
            from services.agent_skills_service import create_skill
            user_id = request.user_id or "system"
            name = (request.name or "").strip() or "Unnamed skill"
            slug = (request.slug or "").strip().lower().replace(" ", "-")[:100] or "unnamed-skill"
            procedure = request.procedure or ""
            required_tools = list(request.required_tools) if request.required_tools else []
            optional_tools = list(request.optional_tools) if request.optional_tools else []
            description = (request.description or "").strip() or None
            category = (request.category or "").strip() or None
            tags = list(request.tags) if request.tags else []
            skill = await create_skill(
                user_id=user_id,
                name=name,
                slug=slug,
                procedure=procedure,
                required_tools=required_tools,
                optional_tools=optional_tools,
                description=description,
                category=category,
                tags=tags,
            )
            return tool_service_pb2.CreateSkillResponse(
                success=True,
                skill_id=str(skill.get("id", "")),
                skill_json=json.dumps(skill),
            )
        except ValueError as e:
            return tool_service_pb2.CreateSkillResponse(
                success=False, skill_id="", skill_json="", error=str(e)
            )
        except Exception as e:
            logger.exception("CreateSkill failed")
            return tool_service_pb2.CreateSkillResponse(
                success=False, skill_id="", skill_json="", error=str(e)
            )

    async def UpdateSkill(
        self,
        request: tool_service_pb2.UpdateSkillRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateSkillResponse:
        """Update a user skill (creates new version). Used for propose_skill_update flow."""
        try:
            from services.agent_skills_service import update_skill
            user_id = request.user_id or "system"
            skill_id = request.skill_id or ""
            procedure = (request.procedure or "").strip() or None
            improvement_rationale = (request.improvement_rationale or "").strip() or None
            evidence_metadata = None
            if request.evidence_metadata_json:
                try:
                    evidence_metadata = json.loads(request.evidence_metadata_json)
                except json.JSONDecodeError:
                    pass
            name = (request.name or "").strip() or None
            description = (request.description or "").strip() or None
            category = (request.category or "").strip() or None
            required_tools = list(request.required_tools) if request.required_tools else None
            optional_tools = list(request.optional_tools) if request.optional_tools else None
            skill = await update_skill(
                skill_id=skill_id,
                user_id=user_id,
                procedure=procedure,
                improvement_rationale=improvement_rationale,
                evidence_metadata=evidence_metadata,
                name=name,
                description=description,
                category=category,
                required_tools=required_tools,
                optional_tools=optional_tools,
            )
            return tool_service_pb2.UpdateSkillResponse(
                success=True,
                skill_id=str(skill.get("id", "")),
                version=skill.get("version", 1),
                skill_json=json.dumps(skill),
            )
        except ValueError as e:
            return tool_service_pb2.UpdateSkillResponse(
                success=False, skill_id="", version=0, skill_json="", error=str(e)
            )
        except Exception as e:
            logger.exception("UpdateSkill failed")
            return tool_service_pb2.UpdateSkillResponse(
                success=False, skill_id="", version=0, skill_json="", error=str(e)
            )

    async def LogAgentExecution(
        self,
        request: tool_service_pb2.LogAgentExecutionRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.LogAgentExecutionResponse:
        """Insert a row into agent_execution_log for custom agent run; update agent_budgets spend."""
        try:
            from decimal import Decimal
            from services.database_manager.database_helpers import fetch_value, execute, fetch_one
            user_id = request.user_id or "system"
            profile_id = request.profile_id or None
            playbook_id = request.playbook_id or None
            if not profile_id:
                return tool_service_pb2.LogAgentExecutionResponse(
                    success=False, execution_id="", error="profile_id required"
                )
            _now = datetime.now(timezone.utc)
            started_at = request.started_at or _now.isoformat()
            completed_at = request.completed_at or _now.isoformat()
            if isinstance(started_at, str):
                started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            if isinstance(completed_at, str):
                completed_at = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            metadata = {}
            if request.metadata_json:
                try:
                    metadata = json.loads(request.metadata_json)
                except json.JSONDecodeError:
                    pass
            metadata["steps_completed"] = request.steps_completed or 0
            metadata["steps_total"] = request.steps_total or 0

            steps_data = []
            if getattr(request, "steps_json", None):
                try:
                    steps_data = json.loads(request.steps_json)
                except (json.JSONDecodeError, TypeError):
                    pass
            tokens_input = 0
            tokens_output = 0
            for s in (steps_data if isinstance(steps_data, list) else []):
                if isinstance(s, dict):
                    tokens_input += int(s.get("input_tokens") or 0)
                    tokens_output += int(s.get("output_tokens") or 0)

            model_used = (metadata.get("model_used") or "")[:255] if metadata.get("model_used") else None
            cost_usd = Decimal("0")
            if model_used and (tokens_input or tokens_output):
                try:
                    from services.service_container import get_service_container
                    container = await get_service_container()
                    if container.chat_service and hasattr(container.chat_service, "get_available_models"):
                        models = await container.chat_service.get_available_models()
                        for m in (models or []):
                            mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
                            if mid == model_used:
                                inc = (getattr(m, "input_cost", None) or (m.get("input_cost") if isinstance(m, dict) else None)) or 0
                                outc = (getattr(m, "output_cost", None) or (m.get("output_cost") if isinstance(m, dict) else None)) or 0
                                cost_usd = Decimal(str(inc)) * tokens_input + Decimal(str(outc)) * tokens_output
                                break
                        if cost_usd == Decimal("0"):
                            logger.warning(
                                "Cost lookup failed: model_used='%s' not found in %d available models",
                                model_used, len(models or []),
                            )
                except Exception as cost_err:
                    logger.debug("Resolve execution cost failed: %s", cost_err)

            execution_id = await fetch_value(
                """
                INSERT INTO agent_execution_log (
                    agent_profile_id, user_id, query, playbook_id,
                    started_at, completed_at, duration_ms, status,
                    error_details, metadata, tokens_input, tokens_output, cost_usd, model_used
                ) VALUES ($1, $2, $3, $4, $5::timestamptz, $6::timestamptz, $7, $8, $9, $10, $11, $12, $13::numeric, $14)
                RETURNING id
                """,
                uuid.UUID(profile_id) if profile_id else None,
                user_id,
                request.query or "",
                uuid.UUID(playbook_id) if playbook_id else None,
                started_at,
                completed_at,
                request.duration_ms or 0,
                request.status or "completed",
                request.error_details or None,
                json.dumps(metadata),
                tokens_input,
                tokens_output,
                float(cost_usd),
                model_used,
            )
            exec_uuid = uuid.UUID(str(execution_id)) if execution_id else None
            if execution_id and isinstance(steps_data, list) and steps_data:
                from services.database_manager.database_helpers import execute
                for s in steps_data:
                    if not isinstance(s, dict):
                        continue
                    try:
                        started_ts = s.get("started_at")
                        completed_ts = s.get("completed_at")
                        if isinstance(started_ts, str):
                            started_ts = datetime.fromisoformat(started_ts.replace("Z", "+00:00"))
                        if isinstance(completed_ts, str):
                            completed_ts = datetime.fromisoformat(completed_ts.replace("Z", "+00:00"))
                        await execute(
                            """
                            INSERT INTO agent_execution_steps (
                                execution_id, step_index, step_name, step_type, action_name,
                                status, started_at, completed_at, duration_ms,
                                inputs_json, outputs_json, error_details, tool_call_trace
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7::timestamptz, $8::timestamptz, $9, $10::jsonb, $11::jsonb, $12, $13::jsonb)
                            """,
                            exec_uuid,
                            int(s.get("step_index", 0)),
                            (s.get("step_name") or "")[:255],
                            (s.get("step_type") or "tool")[:50],
                            (s.get("action_name") or "")[:255] if s.get("action_name") else None,
                            (s.get("status") or "completed")[:50],
                            started_ts if started_ts else None,
                            completed_ts if completed_ts else None,
                            s.get("duration_ms"),
                            json.dumps(s.get("inputs_snapshot") or {}),
                            json.dumps(s.get("outputs_snapshot") or {}),
                            (s.get("error_details") or "")[:65535] if s.get("error_details") else None,
                            json.dumps(s.get("tool_call_trace") if s.get("tool_call_trace") is not None else []),
                        )
                    except Exception as step_err:
                        logger.warning("Insert agent_execution_step failed: %s", step_err)

            discoveries = metadata.get("discoveries")
            if execution_id and isinstance(discoveries, list) and discoveries:
                from services.database_manager.database_helpers import execute
                for d in discoveries:
                    if not isinstance(d, dict):
                        continue
                    try:
                        await execute(
                            """
                            INSERT INTO agent_discoveries (
                                execution_id, user_id, discovery_type, entity_name, entity_type,
                                entity_neo4j_id, relationship_type, related_entity_name,
                                source_connector, source_endpoint, confidence, details
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb)
                            """,
                            exec_uuid,
                            user_id,
                            (d.get("discovery_type") or "entity")[:50],
                            (d.get("entity_name") or "")[:500] if d.get("entity_name") else None,
                            (d.get("entity_type") or "")[:50] if d.get("entity_type") else None,
                            (d.get("entity_neo4j_id") or "")[:255] if d.get("entity_neo4j_id") else None,
                            (d.get("relationship_type") or "")[:100] if d.get("relationship_type") else None,
                            (d.get("related_entity_name") or "")[:500] if d.get("related_entity_name") else None,
                            (d.get("source_connector") or "")[:255] if d.get("source_connector") else None,
                            (d.get("source_endpoint") or "")[:255] if d.get("source_endpoint") else None,
                            float(d["confidence"]) if d.get("confidence") is not None else None,
                            json.dumps(d.get("details") or {}),
                        )
                    except Exception as ins_err:
                        logger.warning("Insert agent_discovery failed: %s", ins_err)

            if execution_id and profile_id and float(cost_usd) > 0:
                try:
                    from datetime import date
                    today = date.today()
                    period_start = today.replace(day=1)
                    row = await fetch_one(
                        "SELECT id, current_period_start, current_period_spend_usd FROM agent_budgets WHERE agent_profile_id = $1",
                        uuid.UUID(profile_id),
                    )
                    if row:
                        existing_start = row.get("current_period_start")
                        if existing_start and (getattr(existing_start, "year", None) != today.year or getattr(existing_start, "month", None) != today.month):
                            await execute(
                                "UPDATE agent_budgets SET current_period_start = $1, current_period_spend_usd = 0, updated_at = NOW() WHERE agent_profile_id = $2",
                                period_start,
                                uuid.UUID(profile_id),
                            )
                        await execute(
                            "UPDATE agent_budgets SET current_period_spend_usd = current_period_spend_usd + $1::numeric, updated_at = NOW() WHERE agent_profile_id = $2",
                            float(cost_usd),
                            uuid.UUID(profile_id),
                        )
                        budget_after = await fetch_one(
                            "SELECT monthly_limit_usd, current_period_spend_usd, warning_threshold_pct, enforce_hard_limit FROM agent_budgets WHERE agent_profile_id = $1",
                            uuid.UUID(profile_id),
                        )
                        if budget_after and budget_after.get("monthly_limit_usd") is not None:
                            limit_usd = float(budget_after["monthly_limit_usd"])
                            spend_usd = float(budget_after.get("current_period_spend_usd") or 0)
                            pct = int(budget_after.get("warning_threshold_pct") or 80)
                            enforce = bool(budget_after.get("enforce_hard_limit") is not False)
                            try:
                                from utils.websocket_manager import get_websocket_manager
                                ws = get_websocket_manager()
                                if ws and user_id:
                                    if enforce and spend_usd >= limit_usd:
                                        await ws.send_to_session(
                                            {
                                                "type": "agent_notification",
                                                "subtype": "budget_exceeded",
                                                "agent_profile_id": profile_id,
                                                "agent_name": None,
                                                "spend_usd": spend_usd,
                                                "limit_usd": limit_usd,
                                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                            },
                                            user_id,
                                        )
                                    elif spend_usd >= limit_usd * (pct / 100.0):
                                        await ws.send_to_session(
                                            {
                                                "type": "agent_notification",
                                                "subtype": "budget_warning",
                                                "agent_profile_id": profile_id,
                                                "agent_name": None,
                                                "spend_usd": spend_usd,
                                                "limit_usd": limit_usd,
                                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                            },
                                            user_id,
                                        )
                            except Exception as ws_err:
                                logger.debug("Budget WebSocket notify failed: %s", ws_err)
                except Exception as budget_err:
                    logger.warning("Update agent_budgets spend failed: %s", budget_err)

            if execution_id and user_id:
                try:
                    from utils.websocket_manager import get_websocket_manager
                    profile_row = await fetch_one(
                        "SELECT name, handle FROM agent_profiles WHERE id = $1",
                        uuid.UUID(profile_id),
                    )
                    agent_name = (profile_row.get("name") or profile_row.get("handle") or "Agent") if profile_row else "Agent"
                    ws_manager = get_websocket_manager()
                    if ws_manager:
                        subtype = "execution_completed" if request.status == "completed" else "execution_failed"
                        await ws_manager.send_to_session(
                            {
                                "type": "agent_notification",
                                "subtype": subtype,
                                "execution_id": str(execution_id),
                                "agent_profile_id": profile_id,
                                "agent_name": agent_name,
                                "status": request.status or "completed",
                                "duration_ms": request.duration_ms,
                                "cost_usd": float(cost_usd) if cost_usd is not None else None,
                                "error_details": (request.error_details or "")[:500] if request.error_details else None,
                                "trigger_type": metadata.get("trigger_type", "manual"),
                                "query": (request.query or "")[:200] if request.query else None,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                            user_id,
                        )
                except Exception as ws_err:
                    logger.debug("LogAgentExecution WebSocket notify failed: %s", ws_err)

            return tool_service_pb2.LogAgentExecutionResponse(
                success=True,
                execution_id=str(execution_id) if execution_id else "",
            )
        except Exception as e:
            logger.exception("LogAgentExecution failed")
            return tool_service_pb2.LogAgentExecutionResponse(
                success=False, execution_id="", error=str(e)
            )

    async def ParkApproval(
        self,
        request: tool_service_pb2.ParkApprovalRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ParkApprovalResponse:
        """Insert a row into agent_approval_queue for background/scheduled approval. Called from orchestrator."""
        try:
            from services.database_manager.database_helpers import execute, fetch_value
            user_id = request.user_id or "system"
            agent_profile_id = request.agent_profile_id or None
            execution_id = request.execution_id or None
            step_name = (request.step_name or "approval")[:255]
            prompt = request.prompt or "Approve to continue?"
            preview_data_json = request.preview_data_json or "{}"
            thread_id = (request.thread_id or "")[:500]
            checkpoint_ns = (request.checkpoint_ns or "")[:255]
            playbook_config_json = request.playbook_config_json or "{}"
            governance_type = (request.governance_type or "playbook_step")[:50]
            if not user_id or not step_name:
                return tool_service_pb2.ParkApprovalResponse(
                    success=False, approval_id="", error="user_id and step_name required"
                )
            profile_uuid = uuid.UUID(agent_profile_id) if agent_profile_id else None
            exec_uuid = uuid.UUID(execution_id) if execution_id else None
            approval_id = await fetch_value(
                """INSERT INTO agent_approval_queue
                   (user_id, agent_profile_id, execution_id, step_name, prompt, preview_data, governance_type, thread_id, checkpoint_ns, playbook_config, status)
                   VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10::jsonb, 'pending')
                   RETURNING id""",
                user_id,
                profile_uuid,
                exec_uuid,
                step_name,
                prompt[:10000],
                preview_data_json,
                governance_type,
                thread_id,
                checkpoint_ns,
                playbook_config_json,
            )
            if not approval_id:
                return tool_service_pb2.ParkApprovalResponse(
                    success=False, approval_id="", error="insert failed"
                )
            try:
                from utils.websocket_manager import get_websocket_manager
                ws = get_websocket_manager()
                if ws and user_id:
                    await ws.send_to_session(
                        {
                            "type": "agent_notification",
                            "subtype": "approval_required",
                            "approval_id": str(approval_id),
                            "agent_profile_id": agent_profile_id,
                            "execution_id": execution_id,
                            "step_name": step_name,
                            "prompt": (prompt or "")[:500],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                        user_id,
                    )
            except Exception as ws_err:
                logger.debug("ParkApproval WebSocket notify failed: %s", ws_err)
            return tool_service_pb2.ParkApprovalResponse(
                success=True,
                approval_id=str(approval_id),
            )
        except Exception as e:
            logger.exception("ParkApproval failed")
            return tool_service_pb2.ParkApprovalResponse(
                success=False, approval_id="", error=str(e)
            )

    async def GetAgentMemory(
        self,
        request: tool_service_pb2.GetAgentMemoryRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetAgentMemoryResponse:
        """Read a single agent memory key. Returns value as JSON string."""
        try:
            from services.database_manager.database_helpers import fetch_one
            user_id = request.user_id or "system"
            profile_id = request.agent_profile_id or None
            memory_key = (request.memory_key or "")[:500]
            if not profile_id or not memory_key:
                return tool_service_pb2.GetAgentMemoryResponse(
                    success=False, memory_value_json="", error="agent_profile_id and memory_key required"
                )
            row = await fetch_one(
                """SELECT memory_value FROM agent_memory
                   WHERE agent_profile_id = $1 AND user_id = $2 AND memory_key = $3
                   AND (expires_at IS NULL OR expires_at > NOW())""",
                uuid.UUID(profile_id),
                user_id,
                memory_key,
            )
            if not row:
                return tool_service_pb2.GetAgentMemoryResponse(
                    success=True, memory_value_json=""
                )
            val = row.get("memory_value")
            return tool_service_pb2.GetAgentMemoryResponse(
                success=True,
                memory_value_json=json.dumps(val, default=_json_default) if val is not None else "",
            )
        except Exception as e:
            logger.exception("GetAgentMemory failed")
            return tool_service_pb2.GetAgentMemoryResponse(
                success=False, memory_value_json="", error=str(e)
            )

    async def SetAgentMemory(
        self,
        request: tool_service_pb2.SetAgentMemoryRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SetAgentMemoryResponse:
        """Write or overwrite an agent memory key."""
        try:
            from services.database_manager.database_helpers import execute, fetch_one
            user_id = request.user_id or "system"
            profile_id = request.agent_profile_id or None
            memory_key = (request.memory_key or "")[:500]
            memory_value_json = request.memory_value_json or "{}"
            memory_type = (request.memory_type or "kv")[:50]
            expires_at = request.expires_at if request.expires_at else None
            if not profile_id or not memory_key:
                return tool_service_pb2.SetAgentMemoryResponse(
                    success=False, error="agent_profile_id and memory_key required"
                )
            try:
                json.loads(memory_value_json)
            except (json.JSONDecodeError, TypeError):
                return tool_service_pb2.SetAgentMemoryResponse(
                    success=False, error="Invalid memory_value_json"
                )
            if expires_at:
                await execute(
                    """INSERT INTO agent_memory (agent_profile_id, user_id, memory_key, memory_value, memory_type, updated_at, expires_at)
                       VALUES ($1, $2, $3, $4::jsonb, $5, NOW(), $6::timestamptz)
                       ON CONFLICT (agent_profile_id, memory_key)
                       DO UPDATE SET memory_value = EXCLUDED.memory_value, memory_type = EXCLUDED.memory_type,
                                     updated_at = NOW(), expires_at = EXCLUDED.expires_at""",
                    uuid.UUID(profile_id),
                    user_id,
                    memory_key,
                    memory_value_json,
                    memory_type,
                    expires_at,
                )
            else:
                await execute(
                    """INSERT INTO agent_memory (agent_profile_id, user_id, memory_key, memory_value, memory_type, updated_at)
                       VALUES ($1, $2, $3, $4::jsonb, $5, NOW())
                       ON CONFLICT (agent_profile_id, memory_key)
                       DO UPDATE SET memory_value = EXCLUDED.memory_value, memory_type = EXCLUDED.memory_type, updated_at = NOW()""",
                    uuid.UUID(profile_id),
                    user_id,
                    memory_key,
                    memory_value_json,
                    memory_type,
                )
            return tool_service_pb2.SetAgentMemoryResponse(success=True)
        except Exception as e:
            logger.exception("SetAgentMemory failed")
            return tool_service_pb2.SetAgentMemoryResponse(success=False, error=str(e))

    async def ListAgentMemories(
        self,
        request: tool_service_pb2.ListAgentMemoriesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListAgentMemoriesResponse:
        """List memory keys for an agent, optionally filtered by prefix."""
        try:
            from services.database_manager.database_helpers import fetch_all
            user_id = request.user_id or "system"
            profile_id = request.agent_profile_id or None
            key_prefix = (request.key_prefix or "").strip() or None
            if not profile_id:
                return tool_service_pb2.ListAgentMemoriesResponse(
                    success=False, memory_keys=[], error="agent_profile_id required"
                )
            if key_prefix:
                rows = await fetch_all(
                    """SELECT memory_key FROM agent_memory
                       WHERE agent_profile_id = $1 AND user_id = $2 AND memory_key LIKE $3
                       AND (expires_at IS NULL OR expires_at > NOW())
                       ORDER BY memory_key""",
                    uuid.UUID(profile_id),
                    user_id,
                    key_prefix + "%",
                )
            else:
                rows = await fetch_all(
                    """SELECT memory_key FROM agent_memory
                       WHERE agent_profile_id = $1 AND user_id = $2
                       AND (expires_at IS NULL OR expires_at > NOW())
                       ORDER BY memory_key""",
                    uuid.UUID(profile_id),
                    user_id,
                )
            keys = [r["memory_key"] for r in rows]
            return tool_service_pb2.ListAgentMemoriesResponse(
                success=True,
                memory_keys=keys,
            )
        except Exception as e:
            logger.exception("ListAgentMemories failed")
            return tool_service_pb2.ListAgentMemoriesResponse(
                success=False, memory_keys=[], error=str(e)
            )

    async def DeleteAgentMemory(
        self,
        request: tool_service_pb2.DeleteAgentMemoryRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.DeleteAgentMemoryResponse:
        """Delete an agent memory key."""
        try:
            from services.database_manager.database_helpers import execute
            user_id = request.user_id or "system"
            profile_id = request.agent_profile_id or None
            memory_key = (request.memory_key or "")[:500]
            if not profile_id or not memory_key:
                return tool_service_pb2.DeleteAgentMemoryResponse(
                    success=False, error="agent_profile_id and memory_key required"
                )
            await execute(
                "DELETE FROM agent_memory WHERE agent_profile_id = $1 AND user_id = $2 AND memory_key = $3",
                uuid.UUID(profile_id),
                user_id,
                memory_key,
            )
            return tool_service_pb2.DeleteAgentMemoryResponse(success=True)
        except Exception as e:
            logger.exception("DeleteAgentMemory failed")
            return tool_service_pb2.DeleteAgentMemoryResponse(success=False, error=str(e))

    async def AppendAgentMemory(
        self,
        request: tool_service_pb2.AppendAgentMemoryRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.AppendAgentMemoryResponse:
        """Append an entry to a log-type memory (JSON array)."""
        try:
            from services.database_manager.database_helpers import fetch_one, execute
            user_id = request.user_id or "system"
            profile_id = request.agent_profile_id or None
            memory_key = (request.memory_key or "")[:500]
            entry_json = request.entry_json or "{}"
            if not profile_id or not memory_key:
                return tool_service_pb2.AppendAgentMemoryResponse(
                    success=False, error="agent_profile_id and memory_key required"
                )
            try:
                entry = json.loads(entry_json)
            except (json.JSONDecodeError, TypeError):
                return tool_service_pb2.AppendAgentMemoryResponse(
                    success=False, error="Invalid entry_json"
                )
            row = await fetch_one(
                "SELECT memory_value, memory_type FROM agent_memory WHERE agent_profile_id = $1 AND user_id = $2 AND memory_key = $3",
                uuid.UUID(profile_id),
                user_id,
                memory_key,
            )
            if row:
                current = row.get("memory_value")
                if not isinstance(current, list):
                    current = [current] if current is not None else []
                current.append(entry)
                await execute(
                    "UPDATE agent_memory SET memory_value = $1::jsonb, updated_at = NOW() WHERE agent_profile_id = $2 AND user_id = $3 AND memory_key = $4",
                    json.dumps(current),
                    uuid.UUID(profile_id),
                    user_id,
                    memory_key,
                )
            else:
                await execute(
                    """INSERT INTO agent_memory (agent_profile_id, user_id, memory_key, memory_value, memory_type, updated_at)
                       VALUES ($1, $2, $3, $4::jsonb, 'log', NOW())""",
                    uuid.UUID(profile_id),
                    user_id,
                    memory_key,
                    json.dumps([entry]),
                )
            return tool_service_pb2.AppendAgentMemoryResponse(success=True)
        except Exception as e:
            logger.exception("AppendAgentMemory failed")
            return tool_service_pb2.AppendAgentMemoryResponse(success=False, error=str(e))

    async def GetAgentRunHistory(
        self,
        request: tool_service_pb2.GetAgentRunHistoryRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetAgentRunHistoryResponse:
        """Query agent_execution_log for a user's agent run history (access-controlled by user_id)."""
        try:
            from services.database_manager.database_helpers import fetch_all

            user_id = request.user_id or "system"
            profile_id = request.agent_profile_id if request.HasField("agent_profile_id") and request.agent_profile_id else None
            limit = 10
            if request.HasField("limit") and request.limit > 0:
                limit = min(int(request.limit), 50)
            status_filter = request.status if request.HasField("status") and request.status else None
            start_date = request.start_date if request.HasField("start_date") and request.start_date else None
            end_date = request.end_date if request.HasField("end_date") and request.end_date else None

            conditions = ["ael.user_id = $1"]
            params = [user_id]
            n = 2
            if profile_id:
                conditions.append(f"ael.agent_profile_id = ${n}")
                params.append(uuid.UUID(profile_id))
                n += 1
            if status_filter:
                conditions.append(f"ael.status = ${n}")
                params.append(status_filter)
                n += 1
            if start_date:
                conditions.append(f"ael.started_at >= ${n}::date")
                params.append(start_date)
                n += 1
            if end_date:
                conditions.append(f"ael.started_at < (${n}::date + interval '1 day')")
                params.append(end_date)
                n += 1
            params.append(limit)
            where_clause = " AND ".join(conditions)
            q = f"""
                SELECT ael.id, ael.query, ael.status, ael.started_at, ael.duration_ms,
                       ael.connectors_called, ael.entities_discovered, ael.error_details, ael.metadata,
                       ap.name AS agent_name
                FROM agent_execution_log ael
                LEFT JOIN agent_profiles ap ON ap.id = ael.agent_profile_id AND ap.user_id = ael.user_id
                WHERE {where_clause}
                ORDER BY ael.started_at DESC
                LIMIT ${n}
            """
            rows = await fetch_all(q, *params)
            runs = []
            agent_name_out = ""
            if profile_id and rows:
                agent_name_out = (rows[0].get("agent_name") or "").strip()
            for r in rows:
                meta = r.get("metadata") or {}
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except json.JSONDecodeError:
                        meta = {}
                steps_completed = meta.get("steps_completed", 0) or 0
                steps_total = meta.get("steps_total", 0) or 0
                conn_called = r.get("connectors_called")
                if isinstance(conn_called, list):
                    connectors_list = [str(x) for x in conn_called]
                elif isinstance(conn_called, str):
                    try:
                        connectors_list = list(json.loads(conn_called)) if conn_called else []
                    except json.JSONDecodeError:
                        connectors_list = []
                else:
                    connectors_list = []
                started = r.get("started_at")
                started_at_str = started.isoformat() if hasattr(started, "isoformat") else str(started or "")
                runs.append(
                    tool_service_pb2.AgentRunRecord(
                        execution_id=str(r["id"]),
                        agent_name=(r.get("agent_name") or "").strip(),
                        query=(r.get("query") or "")[:500],
                        status=(r.get("status") or "").strip(),
                        started_at=started_at_str,
                        duration_ms=int(r["duration_ms"]) if r.get("duration_ms") is not None else None,
                        connectors_called=connectors_list,
                        entities_discovered=int(r.get("entities_discovered") or 0),
                        error_details=(r.get("error_details") or "").strip() or None,
                        steps_completed=int(steps_completed),
                        steps_total=int(steps_total),
                    )
                )
            return tool_service_pb2.GetAgentRunHistoryResponse(
                success=True,
                runs=runs,
                total=len(runs),
                agent_name=agent_name_out,
            )
        except Exception as e:
            logger.exception("GetAgentRunHistory failed")
            return tool_service_pb2.GetAgentRunHistoryResponse(
                success=False,
                runs=[],
                total=0,
                agent_name="",
                error=str(e),
            )

    async def ExecuteConnector(
        self,
        request: tool_service_pb2.ExecuteConnectorRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ExecuteConnectorResponse:
        """Execute a connector endpoint; load definition and credentials from DB."""
        try:
            from services.database_manager.database_helpers import fetch_one
            from clients.connections_service_client import get_connections_service_client

            user_id = request.user_id or "system"
            profile_id = request.profile_id or None
            connector_id = request.connector_id or None
            endpoint_id = request.endpoint_id or None
            if not profile_id or not connector_id or not endpoint_id:
                return tool_service_pb2.ExecuteConnectorResponse(
                    success=False, result_json="", error="profile_id, connector_id, endpoint_id required"
                )
            params = {}
            if request.params_json:
                try:
                    params = json.loads(request.params_json)
                except json.JSONDecodeError:
                    return tool_service_pb2.ExecuteConnectorResponse(
                        success=False, result_json="", error="Invalid params_json"
                    )
            connector = await fetch_one(
                "SELECT id, definition, connector_type FROM data_source_connectors WHERE id = $1",
                connector_id,
            )
            if not connector:
                return tool_service_pb2.ExecuteConnectorResponse(
                    success=False, result_json="", error="Connector not found"
                )
            source = await fetch_one(
                "SELECT credentials_encrypted, config_overrides FROM agent_data_sources "
                "WHERE agent_profile_id = $1 AND connector_id = $2 AND is_enabled = true",
                profile_id,
                connector_id,
            )
            credentials = {}
            if source:
                creds = source.get("credentials_encrypted")
                if isinstance(creds, dict):
                    credentials = creds
                overrides = source.get("config_overrides") or {}
                if isinstance(overrides, dict) and overrides.get("api_key"):
                    credentials.setdefault("api_key", overrides["api_key"])
            definition = connector.get("definition") or {}
            if isinstance(definition, str):
                definition = json.loads(definition) if definition else {}
            client = await get_connections_service_client()
            result = await client.execute_connector_endpoint(
                definition=definition,
                credentials=credentials,
                endpoint_id=endpoint_id,
                params=params,
                connector_type=connector.get("connector_type"),
            )
            return tool_service_pb2.ExecuteConnectorResponse(
                success=True,
                result_json=json.dumps(result),
            )
        except Exception as e:
            logger.exception("ExecuteConnector failed")
            return tool_service_pb2.ExecuteConnectorResponse(
                success=False, result_json="", error=str(e)
            )

    async def ExecuteMcpTool(
        self,
        request: tool_service_pb2.ExecuteMcpToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ExecuteMcpToolResponse:
        """Call tools/call on a user-configured MCP server."""
        try:
            from services.database_manager.database_helpers import fetch_one
            from services.mcp_client_service import call_tool

            user_id = request.user_id or "system"
            server_id = int(request.server_id) if request.server_id else 0
            tool_name = (request.tool_name or "").strip()
            if not server_id or not tool_name:
                return tool_service_pb2.ExecuteMcpToolResponse(
                    success=False, result_json="", formatted="", error="server_id and tool_name required"
                )
            raw_args = request.arguments_json or "{}"
            try:
                args = json.loads(raw_args) if raw_args else {}
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                return tool_service_pb2.ExecuteMcpToolResponse(
                    success=False, result_json="", formatted="", error="Invalid arguments_json"
                )

            row = await fetch_one(
                """
                SELECT id, user_id, name, description, transport, url, command, args, env, headers, is_active
                FROM mcp_servers
                WHERE id = $1 AND user_id = $2 AND is_active = true
                """,
                server_id,
                user_id,
            )
            if not row:
                return tool_service_pb2.ExecuteMcpToolResponse(
                    success=False, result_json="", formatted="", error="MCP server not found"
                )
            cfg = dict(row)
            for key in ("args", "env", "headers"):
                v = cfg.get(key)
                if isinstance(v, str):
                    try:
                        cfg[key] = json.loads(v) if v else ([] if key == "args" else {})
                    except json.JSONDecodeError:
                        cfg[key] = [] if key == "args" else {}

            ok, result_json, formatted = await call_tool(cfg, tool_name, args)
            return tool_service_pb2.ExecuteMcpToolResponse(
                success=bool(ok),
                result_json=result_json or "",
                formatted=formatted or "",
                error="" if ok else (formatted or result_json or "MCP tool error"),
            )
        except Exception as e:
            logger.exception("ExecuteMcpTool failed")
            return tool_service_pb2.ExecuteMcpToolResponse(
                success=False, result_json="", formatted="", error=str(e)
            )

    async def GetMcpServerTools(
        self,
        request: tool_service_pb2.GetMcpServerToolsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetMcpServerToolsResponse:
        """Return tool names from cached discovered_tools for an MCP server."""
        try:
            from services.database_manager.database_helpers import fetch_one

            user_id = request.user_id or "system"
            server_id = int(request.server_id) if request.server_id else 0
            if not server_id:
                return tool_service_pb2.GetMcpServerToolsResponse(
                    success=False, tool_names=[], error="server_id required"
                )
            row = await fetch_one(
                """
                SELECT discovered_tools FROM mcp_servers
                WHERE id = $1 AND user_id = $2 AND is_active = true
                """,
                server_id,
                user_id,
            )
            if not row:
                return tool_service_pb2.GetMcpServerToolsResponse(
                    success=False, tool_names=[], error="MCP server not found"
                )
            raw = row.get("discovered_tools")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw) if raw else []
                except json.JSONDecodeError:
                    raw = []
            names: List[str] = []
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict) and item.get("name"):
                        names.append(str(item["name"]))
                    elif isinstance(item, str):
                        names.append(item)
            return tool_service_pb2.GetMcpServerToolsResponse(
                success=True,
                tool_names=names,
                error="",
            )
        except Exception as e:
            logger.exception("GetMcpServerTools failed")
            return tool_service_pb2.GetMcpServerToolsResponse(
                success=False, tool_names=[], error=str(e)
            )

    async def DiscoverMcpServer(
        self,
        request: tool_service_pb2.DiscoverMcpServerRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.DiscoverMcpServerResponse:
        """Run tools/list for a user MCP server (same runtime as ExecuteMcpTool: stdio uses uvx/npx here)."""
        try:
            from services.database_manager.database_helpers import fetch_one
            from services.mcp_client_service import discover_tools

            user_id = request.user_id or "system"
            server_id = int(request.server_id) if request.server_id else 0
            if not server_id:
                return tool_service_pb2.DiscoverMcpServerResponse(
                    success=False, tools_json="[]", error="server_id required"
                )
            row = await fetch_one(
                """
                SELECT id, user_id, name, description, transport, url, command, args, env, headers, is_active
                FROM mcp_servers
                WHERE id = $1 AND user_id = $2
                """,
                server_id,
                user_id,
            )
            if not row:
                return tool_service_pb2.DiscoverMcpServerResponse(
                    success=False, tools_json="[]", error="MCP server not found"
                )
            cfg = dict(row)
            for key in ("args", "env", "headers"):
                v = cfg.get(key)
                if isinstance(v, str):
                    try:
                        cfg[key] = json.loads(v) if v else ([] if key == "args" else {})
                    except json.JSONDecodeError:
                        cfg[key] = [] if key == "args" else {}

            tools = await discover_tools(cfg)
            return tool_service_pb2.DiscoverMcpServerResponse(
                success=True,
                tools_json=json.dumps(tools, default=str),
                error="",
            )
        except Exception as e:
            logger.exception("DiscoverMcpServer failed")
            return tool_service_pb2.DiscoverMcpServerResponse(
                success=False, tools_json="[]", error=str(e)
            )

    # ===== Data Connection Builder =====

    async def ProbeApiEndpoint(
        self,
        request: tool_service_pb2.ProbeApiEndpointRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ProbeApiEndpointResponse:
        """Raw HTTP request for API discovery; delegates to connections-service."""
        try:
            from clients.connections_service_client import get_connections_service_client
            user_id = request.user_id or "system"
            headers = {}
            if request.headers_json:
                try:
                    headers = json.loads(request.headers_json)
                except json.JSONDecodeError:
                    return tool_service_pb2.ProbeApiEndpointResponse(
                        success=False, error="Invalid headers_json"
                    )
            body = None
            if request.body_json:
                try:
                    body = json.loads(request.body_json)
                except json.JSONDecodeError:
                    return tool_service_pb2.ProbeApiEndpointResponse(
                        success=False, error="Invalid body_json"
                    )
            params = None
            if request.params_json:
                try:
                    params = json.loads(request.params_json)
                except json.JSONDecodeError:
                    return tool_service_pb2.ProbeApiEndpointResponse(
                        success=False, error="Invalid params_json"
                    )
            client = await get_connections_service_client()
            result = await client.probe_api_endpoint(
                url=request.url or "",
                method=request.method or "GET",
                headers=headers,
                body=body,
                params=params,
            )
            if not result.get("success"):
                return tool_service_pb2.ProbeApiEndpointResponse(
                    success=False,
                    error=result.get("error", "Probe failed"),
                )
            return tool_service_pb2.ProbeApiEndpointResponse(
                success=True,
                status_code=result.get("status_code", 0),
                response_headers_json=json.dumps(result.get("response_headers", {})),
                response_body=result.get("response_body", ""),
                content_type=result.get("content_type", ""),
            )
        except Exception as e:
            logger.exception("ProbeApiEndpoint failed")
            return tool_service_pb2.ProbeApiEndpointResponse(success=False, error=str(e))

    async def TestConnectorEndpoint(
        self,
        request: tool_service_pb2.TestConnectorEndpointRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.TestConnectorEndpointResponse:
        """Test a connector definition against the live API (no save required)."""
        try:
            from clients.connections_service_client import get_connections_service_client
            definition = {}
            if request.definition_json:
                definition = json.loads(request.definition_json)
            params = {}
            if request.params_json:
                params = json.loads(request.params_json)
            credentials = {}
            if request.credentials_json:
                credentials = json.loads(request.credentials_json)
            client = await get_connections_service_client()
            result = await client.execute_connector_endpoint(
                definition=definition,
                credentials=credentials,
                endpoint_id=request.endpoint_id or "",
                params=params,
                raw_response=True,
            )
            records = result.get("records", [])
            raw_response = result.get("raw_response")
            formatted = result.get("formatted", "")
            if result.get("error"):
                return tool_service_pb2.TestConnectorEndpointResponse(
                    success=False,
                    records_json="[]",
                    count=0,
                    raw_response_json="",
                    formatted=result.get("error", ""),
                    error=result.get("error"),
                )
            return tool_service_pb2.TestConnectorEndpointResponse(
                success=True,
                records_json=json.dumps(records),
                count=len(records),
                raw_response_json=json.dumps(raw_response) if raw_response is not None else "{}",
                formatted=formatted,
            )
        except json.JSONDecodeError as e:
            return tool_service_pb2.TestConnectorEndpointResponse(
                success=False, error=f"Invalid JSON: {e}"
            )
        except Exception as e:
            logger.exception("TestConnectorEndpoint failed")
            return tool_service_pb2.TestConnectorEndpointResponse(success=False, error=str(e))

    async def CreateDataConnector(
        self,
        request: tool_service_pb2.CreateDataConnectorRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateDataConnectorResponse:
        """Save a connector definition to the database."""
        try:
            from services.database_manager.database_helpers import execute, fetch_one
            user_id = request.user_id or "system"
            name = request.name or "Unnamed Connector"
            definition = {}
            if request.definition_json:
                definition = json.loads(request.definition_json)
            auth_fields = []
            if request.auth_fields_json:
                try:
                    auth_fields = json.loads(request.auth_fields_json)
                except json.JSONDecodeError:
                    pass
            await execute(
                """
                INSERT INTO data_source_connectors (
                    user_id, name, description, connector_type, version, definition,
                    is_template, requires_auth, auth_fields, icon, category, tags
                ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, false, $7, $8::jsonb, $9, $10, $11)
                """,
                user_id,
                name,
                request.description or "",
                "rest",
                "1.0",
                json.dumps(definition),
                request.requires_auth,
                json.dumps(auth_fields),
                None,
                request.category or None,
                [],
            )
            row = await fetch_one(
                "SELECT id, name FROM data_source_connectors WHERE user_id = $1 AND name = $2 ORDER BY created_at DESC LIMIT 1",
                user_id,
                name,
            )
            if not row:
                return tool_service_pb2.CreateDataConnectorResponse(
                    success=False, error="Failed to create connector"
                )
            connector_id = str(row["id"])
            formatted = f"Created data connector: {row.get('name', name)} (ID: {connector_id})"
            return tool_service_pb2.CreateDataConnectorResponse(
                success=True,
                connector_id=connector_id,
                name=row.get("name", name),
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("CreateDataConnector failed")
            return tool_service_pb2.CreateDataConnectorResponse(success=False, error=str(e))

    async def BulkScrapeUrls(
        self,
        request: tool_service_pb2.BulkScrapeUrlsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.BulkScrapeUrlsResponse:
        """Scrape URLs for content and optionally images. Inline for <20 URLs, Celery for 20+."""
        try:
            urls = []
            if request.urls_json:
                urls = json.loads(request.urls_json)
            if not isinstance(urls, list):
                urls = []
            urls = [u for u in urls if isinstance(u, str) and u.strip()]
            user_id = request.user_id or "system"
            extract_images = request.extract_images
            download_images = request.download_images
            max_concurrent = request.max_concurrent if request.max_concurrent > 0 else 10
            rate_limit_seconds = request.rate_limit_seconds if request.rate_limit_seconds > 0 else 1.0
            folder_id = request.folder_id or ""

            if len(urls) >= 20:
                from services.celery_tasks.scraper_tasks import batch_url_scrape_task
                task = batch_url_scrape_task.delay(
                    urls=urls,
                    user_id=user_id,
                    config={
                        "extract_images": extract_images,
                        "download_images": download_images,
                        "image_output_folder": request.image_output_folder or "",
                        "metadata_fields_json": request.metadata_fields_json or "[]",
                        "max_concurrent": max_concurrent,
                        "rate_limit_seconds": rate_limit_seconds,
                        "folder_id": folder_id,
                    },
                )
                return tool_service_pb2.BulkScrapeUrlsResponse(
                    success=True,
                    task_id=task.id,
                    results_json="[]",
                    count=0,
                    images_found=0,
                    images_downloaded=0,
                    formatted=f"Bulk scrape started for {len(urls)} URLs. Task ID: {task.id}. Use get_bulk_scrape_status to check progress.",
                )
            else:
                from clients.crawl_service_client import get_crawl_service_client
                client = await get_crawl_service_client()
                response = await client.crawl_many(
                    urls=urls[:20],
                    max_concurrent=max_concurrent,
                    rate_limit_seconds=rate_limit_seconds,
                    include_metadata=True,
                )
                results = response.get("results", [])
                images_found = 0
                images_downloaded = 0
                for r in results:
                    images_found += len(r.get("images", []))
                formatted = f"Crawled {len(results)} URL(s). Images found: {images_found}."
                return tool_service_pb2.BulkScrapeUrlsResponse(
                    success=True,
                    task_id="",
                    results_json=json.dumps(results),
                    count=len(results),
                    images_found=images_found,
                    images_downloaded=images_downloaded,
                    formatted=formatted,
                )
        except Exception as e:
            logger.exception("BulkScrapeUrls failed")
            return tool_service_pb2.BulkScrapeUrlsResponse(
                success=False, error=str(e)
            )

    async def GetBulkScrapeStatus(
        self,
        request: tool_service_pb2.GetBulkScrapeStatusRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetBulkScrapeStatusResponse:
        """Get status and optional results of a bulk scrape Celery task."""
        try:
            from celery.result import AsyncResult
            from services.celery_app import celery_app
            task_id = request.task_id or ""
            if not task_id:
                return tool_service_pb2.GetBulkScrapeStatusResponse(
                    success=False, error="task_id required"
                )
            result = AsyncResult(task_id, app=celery_app)
            state = result.state or "PENDING"
            progress_current = 0
            progress_total = 0
            progress_message = ""
            results_json = "[]"
            if state == "SUCCESS" and result.result:
                res = result.result
                if isinstance(res, dict):
                    progress_current = res.get("progress_current", 0)
                    progress_total = res.get("progress_total", 0)
                    progress_message = res.get("progress_message", "")
                    results_json = json.dumps(res.get("results", []))
            elif state == "PROGRESS" and result.info:
                info = result.info if isinstance(result.info, dict) else {}
                progress_current = info.get("current", 0)
                progress_total = info.get("total", 0)
                progress_message = info.get("message", "")
                results_json = json.dumps(info.get("results", []))
            formatted = f"Task {task_id}: {state}. {progress_message or state}"
            return tool_service_pb2.GetBulkScrapeStatusResponse(
                success=True,
                status=state,
                progress_current=progress_current,
                progress_total=progress_total,
                progress_message=progress_message,
                results_json=results_json,
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("GetBulkScrapeStatus failed")
            return tool_service_pb2.GetBulkScrapeStatusResponse(
                success=False, error=str(e)
            )

    async def ListControlPanes(
        self,
        request: tool_service_pb2.ListControlPanesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListControlPanesResponse:
        """List all control panes for the user."""
        try:
            from services.database_manager.database_helpers import fetch_all
            user_id = request.user_id or "system"
            rows = await fetch_all(
                """
                SELECT p.id, p.user_id, p.name, p.icon, p.connector_id, p.credentials_encrypted,
                       p.connection_id, p.controls, p.is_visible, p.sort_order, p.refresh_interval, p.created_at, p.updated_at,
                       c.name AS connector_name
                FROM user_control_panes p
                LEFT JOIN data_source_connectors c ON c.id = p.connector_id
                WHERE p.user_id = $1
                ORDER BY p.sort_order ASC, p.name ASC
                """,
                user_id,
            )
            result = []
            for r in rows:
                row = dict(r)
                controls = row.get("controls")
                if controls is not None and isinstance(controls, str):
                    try:
                        controls = json.loads(controls)
                    except json.JSONDecodeError:
                        controls = []
                result.append({
                    "id": str(row["id"]),
                    "user_id": row.get("user_id"),
                    "name": row.get("name", ""),
                    "icon": row.get("icon", "Tune"),
                    "connector_id": str(row["connector_id"]),
                    "connector_name": row.get("connector_name"),
                    "controls": controls or [],
                    "is_visible": row.get("is_visible", True),
                    "sort_order": row.get("sort_order", 0),
                    "refresh_interval": row.get("refresh_interval", 0),
                })
            parts = [f"Found {len(result)} control pane(s):"]
            for p in result:
                name = p.get("name", "(unnamed)")
                pid = p.get("id", "")
                conn = p.get("connector_name") or p.get("connector_id", "")
                parts.append(f"  - {name} (id: {pid}, connector: {conn})")
            formatted = "\n".join(parts) if result else parts[0]
            return tool_service_pb2.ListControlPanesResponse(
                success=True,
                panes_json=json.dumps(result),
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("ListControlPanes failed")
            return tool_service_pb2.ListControlPanesResponse(success=False, error=str(e))

    async def GetConnectorEndpoints(
        self,
        request: tool_service_pb2.GetConnectorEndpointsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetConnectorEndpointsResponse:
        """Return endpoint ids and metadata from a connector definition for control pane mapping."""
        try:
            from services.database_manager.database_helpers import fetch_one
            user_id = request.user_id or "system"
            connector_id = request.connector_id or ""
            if not connector_id:
                return tool_service_pb2.GetConnectorEndpointsResponse(
                    success=False, error="connector_id required"
                )
            row = await fetch_one(
                "SELECT id, definition FROM data_source_connectors WHERE id = $1 AND (user_id = $2 OR is_template = true)",
                connector_id,
                user_id,
            )
            if not row:
                return tool_service_pb2.GetConnectorEndpointsResponse(
                    success=False, error="Connector not found"
                )
            definition = row.get("definition") or {}
            if isinstance(definition, str):
                try:
                    definition = json.loads(definition)
                except json.JSONDecodeError:
                    definition = {}
            endpoints_def = definition.get("endpoints") or {}
            if isinstance(endpoints_def, list):
                endpoints_def = {ep.get("id") or ep.get("name"): ep for ep in endpoints_def if ep.get("id") or ep.get("name")}
            endpoints_list = []
            for eid, ep in (endpoints_def.items() if isinstance(endpoints_def, dict) else []):
                raw_params = ep.get("params") or []
                if isinstance(raw_params, dict):
                    raw_params = [{"name": k, "in": "query", "default": v} for k, v in raw_params.items()]
                param_list = []
                for p in raw_params:
                    name = p.get("name") or p.get("id")
                    if name:
                        param_list.append({
                            "name": name,
                            "in": p.get("in", "query"),
                            "description": p.get("description") or "",
                            "required": p.get("required", False),
                            "default": p.get("default"),
                        })
                endpoints_list.append({
                    "id": eid,
                    "path": ep.get("path", "/"),
                    "method": (ep.get("method") or "GET").upper(),
                    "description": ep.get("description") or "",
                    "params": param_list,
                })
            parts = []
            for e in endpoints_list:
                param_names = [p["name"] for p in e.get("params", [])]
                parts.append(f"  {e['id']} ({e['method']} {e['path']}) params: {param_names or 'none'}")
            formatted = f"Connector has {len(endpoints_list)} endpoint(s):\n" + "\n".join(parts)
            return tool_service_pb2.GetConnectorEndpointsResponse(
                success=True,
                endpoints_json=json.dumps(endpoints_list),
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("GetConnectorEndpoints failed")
            return tool_service_pb2.GetConnectorEndpointsResponse(success=False, error=str(e))

    async def ListDataConnectors(
        self,
        request: tool_service_pb2.ListDataConnectorsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListDataConnectorsResponse:
        """List user-owned data source connectors (non-templates)."""
        try:
            from services.database_manager.database_helpers import fetch_all
            user_id = request.user_id or "system"
            rows = await fetch_all(
                """
                SELECT id, name, description, connector_type, definition, is_locked, category, tags, created_at, updated_at
                FROM data_source_connectors
                WHERE user_id = $1 AND (is_template = false OR is_template IS NULL)
                ORDER BY updated_at DESC NULLS LAST, created_at DESC
                """,
                user_id,
            )
            result = []
            for r in rows:
                definition = r.get("definition") or {}
                if isinstance(definition, str):
                    try:
                        definition = json.loads(definition)
                    except json.JSONDecodeError:
                        definition = {}
                endpoints = definition.get("endpoints") or {}
                endpoint_count = len(endpoints) if isinstance(endpoints, dict) else 0
                result.append({
                    "id": str(r["id"]),
                    "name": r.get("name", ""),
                    "description": r.get("description"),
                    "connector_type": r.get("connector_type", "rest"),
                    "endpoint_count": endpoint_count,
                    "is_locked": r.get("is_locked", False),
                    "category": r.get("category"),
                    "tags": list(r.get("tags") or []),
                    "created_at": r.get("created_at").isoformat() if r.get("created_at") else None,
                    "updated_at": r.get("updated_at").isoformat() if r.get("updated_at") else None,
                })
            parts = [f"Found {len(result)} connector(s):"]
            for c in result:
                name = c.get("name", "(unnamed)")
                cid = c.get("id", "")
                ctype = c.get("connector_type", "rest")
                n_ep = c.get("endpoint_count", 0)
                parts.append(f"  - {name} (id: {cid}, type: {ctype}, {n_ep} endpoint(s))")
            formatted = "\n".join(parts) if result else parts[0]
            return tool_service_pb2.ListDataConnectorsResponse(
                success=True,
                connectors_json=json.dumps(result),
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("ListDataConnectors failed")
            return tool_service_pb2.ListDataConnectorsResponse(success=False, error=str(e))

    async def GetDataConnector(
        self,
        request: tool_service_pb2.GetDataConnectorRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetDataConnectorResponse:
        """Return full connector by ID (definition, endpoints; auth values redacted)."""
        try:
            from services.database_manager.database_helpers import fetch_one
            user_id = request.user_id or "system"
            connector_id = request.connector_id or ""
            if not connector_id:
                return tool_service_pb2.GetDataConnectorResponse(
                    success=False, error="connector_id required"
                )
            row = await fetch_one(
                "SELECT id, name, description, connector_type, definition, requires_auth, auth_fields, "
                "is_locked, category, tags, created_at, updated_at FROM data_source_connectors "
                "WHERE id = $1 AND user_id = $2",
                connector_id,
                user_id,
            )
            if not row:
                return tool_service_pb2.GetDataConnectorResponse(
                    success=False, error="Connector not found"
                )
            definition = row.get("definition") or {}
            if isinstance(definition, str):
                try:
                    definition = json.loads(definition) if definition else {}
                except json.JSONDecodeError:
                    definition = {}
            if not isinstance(definition, dict):
                definition = {}
            auth_fields_raw = row.get("auth_fields") or []
            if isinstance(auth_fields_raw, str):
                try:
                    auth_fields_raw = json.loads(auth_fields_raw) if auth_fields_raw else []
                except json.JSONDecodeError:
                    auth_fields_raw = []
            auth_field_names = []
            if isinstance(auth_fields_raw, list):
                for f in auth_fields_raw:
                    if isinstance(f, dict) and f.get("name"):
                        auth_field_names.append(f["name"])
                    elif isinstance(f, str):
                        auth_field_names.append(f)
            connector = {
                "id": str(row["id"]),
                "name": row.get("name", ""),
                "description": row.get("description"),
                "connector_type": row.get("connector_type", "rest"),
                "definition": definition,
                "requires_auth": row.get("requires_auth", False),
                "auth_field_names": auth_field_names,
                "is_locked": row.get("is_locked", False),
                "category": row.get("category"),
                "tags": list(row.get("tags") or []),
                "endpoint_count": len(definition.get("endpoints") or {}) if isinstance(definition.get("endpoints"), dict) else 0,
                "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
                "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
            }
            parts = [
                f"**{connector['name']}** (ID: {connector['id']})",
                f"Type: {connector['connector_type']}, Endpoints: {connector['endpoint_count']}",
            ]
            if connector.get("requires_auth"):
                parts.append(f"Auth: required (fields: {', '.join(connector.get('auth_field_names', []))})")
            return tool_service_pb2.GetDataConnectorResponse(
                success=True,
                connector_json=json.dumps(connector),
                formatted="\n".join(parts),
            )
        except Exception as e:
            logger.exception("GetDataConnector failed")
            return tool_service_pb2.GetDataConnectorResponse(
                success=False, error=str(e)
            )

    async def UpdateDataConnector(
        self,
        request: tool_service_pb2.UpdateDataConnectorRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateDataConnectorResponse:
        """Update a data connector (partial update)."""
        try:
            from services.database_manager.database_helpers import fetch_one, execute
            user_id = request.user_id or "system"
            connector_id = request.connector_id or ""
            if not connector_id:
                return tool_service_pb2.UpdateDataConnectorResponse(
                    success=False, error="connector_id required"
                )
            row = await fetch_one(
                "SELECT id, is_locked FROM data_source_connectors WHERE id = $1 AND user_id = $2",
                connector_id,
                user_id,
            )
            if not row:
                return tool_service_pb2.UpdateDataConnectorResponse(
                    success=False, error="Connector not found"
                )
            updates = {}
            if request.HasField("name"):
                updates["name"] = request.name
            if request.HasField("description"):
                updates["description"] = request.description
            if request.HasField("connector_type"):
                updates["connector_type"] = request.connector_type
            if request.HasField("definition_json"):
                updates["definition"] = request.definition_json
            if request.HasField("requires_auth"):
                updates["requires_auth"] = request.requires_auth
            if request.HasField("auth_fields_json"):
                updates["auth_fields"] = request.auth_fields_json
            if request.HasField("is_locked"):
                updates["is_locked"] = request.is_locked
            if not updates:
                formatted = "No updates provided."
                return tool_service_pb2.UpdateDataConnectorResponse(
                    success=True,
                    connector_id=connector_id,
                    formatted=formatted,
                )
            if row.get("is_locked") and set(updates.keys()) != {"is_locked"}:
                return tool_service_pb2.UpdateDataConnectorResponse(
                    success=False, error="Connector is locked; only lock toggle is allowed"
                )
            set_clauses = []
            args = []
            idx = 1
            jsonb_fields = ("definition", "auth_fields")
            for k, v in updates.items():
                if k in jsonb_fields:
                    set_clauses.append(f"{k} = ${idx}::jsonb")
                    args.append(v if isinstance(v, str) else json.dumps(v) if v is not None else "{}")
                else:
                    set_clauses.append(f"{k} = ${idx}")
                    args.append(v)
                idx += 1
            set_clauses.append("updated_at = NOW()")
            args.extend([connector_id, user_id])
            await execute(
                f"UPDATE data_source_connectors SET {', '.join(set_clauses)} WHERE id = ${idx} AND user_id = ${idx + 1}",
                *args,
            )
            formatted = f"Updated connector {connector_id}."
            return tool_service_pb2.UpdateDataConnectorResponse(
                success=True,
                connector_id=connector_id,
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("UpdateDataConnector failed")
            return tool_service_pb2.UpdateDataConnectorResponse(success=False, error=str(e))

    async def ListPlaybooks(
        self,
        request: tool_service_pb2.ListPlaybooksRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListPlaybooksResponse:
        """List playbooks owned by the user or templates."""
        try:
            from services import agent_factory_service
            user_id = request.user_id or "system"
            playbooks = await agent_factory_service.list_playbooks(user_id)
            result = []
            for p in playbooks:
                definition = p.get("definition") or {}
                if not isinstance(definition, dict):
                    definition = {}
                steps = definition.get("steps") or []
                triggers = p.get("triggers") or []
                result.append({
                    "id": p.get("id"),
                    "name": p.get("name", ""),
                    "description": p.get("description"),
                    "step_count": len(steps) if isinstance(steps, list) else 0,
                    "is_template": p.get("is_template", False),
                    "category": p.get("category"),
                    "tags": list(p.get("tags") or []),
                    "is_locked": p.get("is_locked", False),
                    "run_context": definition.get("run_context") or "background",
                    "has_triggers": len(triggers) > 0,
                })
            formatted = f"Found {len(result)} playbook(s)."
            return tool_service_pb2.ListPlaybooksResponse(
                success=True,
                playbooks_json=json.dumps(result),
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("ListPlaybooks failed")
            return tool_service_pb2.ListPlaybooksResponse(success=False, error=str(e))

    async def ListAgentProfiles(
        self,
        request: tool_service_pb2.ListAgentProfilesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListAgentProfilesResponse:
        """List agent profiles for the user with derived status."""
        try:
            from services import agent_factory_service
            user_id = request.user_id or "system"
            profiles = await agent_factory_service.list_profiles(user_id)
            for p in profiles:
                is_active = p.get("is_active", True)
                last_status = p.get("last_execution_status")
                p["status"] = "draft" if (not is_active and not last_status) else "paused" if not is_active else ("error" if last_status == "failed" else "active")
            formatted = f"Found {len(profiles)} profile(s)."
            return tool_service_pb2.ListAgentProfilesResponse(
                success=True,
                profiles_json=json.dumps(profiles),
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("ListAgentProfiles failed")
            return tool_service_pb2.ListAgentProfilesResponse(success=False, error=str(e))

    async def ListAgentSchedules(
        self,
        request: tool_service_pb2.ListAgentSchedulesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListAgentSchedulesResponse:
        """List schedules for an agent profile (user must own the profile)."""
        try:
            from services.database_manager.database_helpers import fetch_all, fetch_one
            user_id = request.user_id or "system"
            agent_id = request.agent_id or ""
            if not agent_id:
                return tool_service_pb2.ListAgentSchedulesResponse(
                    success=False, error="agent_id required"
                )
            profile = await fetch_one(
                "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
                agent_id,
                user_id,
            )
            if not profile:
                return tool_service_pb2.ListAgentSchedulesResponse(
                    success=False, error="Profile not found"
                )
            rows = await fetch_all(
                "SELECT * FROM agent_schedules WHERE agent_profile_id = $1 ORDER BY created_at",
                agent_id,
            )
            result = []
            for r in rows:
                result.append({
                    "id": str(r["id"]),
                    "agent_profile_id": str(r["agent_profile_id"]),
                    "schedule_type": r.get("schedule_type"),
                    "cron_expression": r.get("cron_expression"),
                    "interval_seconds": r.get("interval_seconds"),
                    "timezone": r.get("timezone") or "UTC",
                    "is_active": r.get("is_active", True),
                    "next_run_at": r["next_run_at"].isoformat() if r.get("next_run_at") else None,
                    "last_run_at": r["last_run_at"].isoformat() if r.get("last_run_at") else None,
                    "last_status": r.get("last_status"),
                    "run_count": r.get("run_count", 0),
                })
            parts = [f"Found {len(result)} schedule(s) for agent {agent_id}:"]
            for s in result:
                stype = s.get("schedule_type", "?")
                active = "active" if s.get("is_active") else "paused"
                parts.append(f"  - {s['id']}: {stype} ({active})")
            return tool_service_pb2.ListAgentSchedulesResponse(
                success=True,
                schedules_json=json.dumps(result),
                formatted="\n".join(parts),
            )
        except Exception as e:
            logger.exception("ListAgentSchedules failed")
            return tool_service_pb2.ListAgentSchedulesResponse(success=False, error=str(e))

    async def ListAgentDataSources(
        self,
        request: tool_service_pb2.ListAgentDataSourcesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListAgentDataSourcesResponse:
        """List data source bindings for an agent profile (user must own the profile)."""
        try:
            from services.database_manager.database_helpers import fetch_all, fetch_one
            user_id = request.user_id or "system"
            agent_id = request.agent_id or ""
            if not agent_id:
                return tool_service_pb2.ListAgentDataSourcesResponse(
                    success=False, error="agent_id required"
                )
            profile = await fetch_one(
                "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
                agent_id,
                user_id,
            )
            if not profile:
                return tool_service_pb2.ListAgentDataSourcesResponse(
                    success=False, error="Profile not found"
                )
            rows = await fetch_all(
                """
                SELECT ads.id AS binding_id, ads.connector_id, ads.config_overrides, ads.is_enabled,
                       dsc.name AS connector_name, dsc.connector_type, dsc.definition
                FROM agent_data_sources ads
                LEFT JOIN data_source_connectors dsc ON dsc.id = ads.connector_id
                WHERE ads.agent_profile_id = $1
                ORDER BY ads.created_at
                """,
                agent_id,
            )
            result = []
            for r in rows:
                definition = r.get("definition") or {}
                if isinstance(definition, str):
                    try:
                        definition = json.loads(definition)
                    except json.JSONDecodeError:
                        definition = {}
                endpoints = definition.get("endpoints") or {}
                endpoint_count = len(endpoints) if isinstance(endpoints, dict) else 0
                result.append({
                    "binding_id": str(r["binding_id"]),
                    "connector_id": str(r["connector_id"]),
                    "connector_name": r.get("connector_name", ""),
                    "connector_type": r.get("connector_type", "rest"),
                    "endpoint_count": endpoint_count,
                    "is_enabled": r.get("is_enabled", True),
                    "config_overrides": r.get("config_overrides") or {},
                })
            parts = [f"Found {len(result)} data source binding(s) for agent {agent_id}:"]
            for b in result:
                parts.append(f"  - {b['connector_name']} (connector: {b['connector_id']}, enabled: {b['is_enabled']})")
            return tool_service_pb2.ListAgentDataSourcesResponse(
                success=True,
                bindings_json=json.dumps(result),
                formatted="\n".join(parts),
            )
        except Exception as e:
            logger.exception("ListAgentDataSources failed")
            return tool_service_pb2.ListAgentDataSourcesResponse(success=False, error=str(e))

    async def CreateControlPane(
        self,
        request: tool_service_pb2.CreateControlPaneRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateControlPaneResponse:
        """Create a control pane wired to a data connector."""
        try:
            from services.database_manager.database_helpers import fetch_one, fetch_value
            user_id = request.user_id or "system"
            name = request.name or "Control Pane"
            icon = request.icon or "Tune"
            connector_id = request.connector_id or ""
            if not connector_id:
                return tool_service_pb2.CreateControlPaneResponse(
                    success=False, error="connector_id required"
                )
            conn = await fetch_one(
                "SELECT id FROM data_source_connectors WHERE id = $1 AND (user_id = $2 OR is_template = true)",
                connector_id,
                user_id,
            )
            if not conn:
                return tool_service_pb2.CreateControlPaneResponse(
                    success=False, error="Connector not found"
                )
            credentials = {}
            if request.credentials_encrypted_json:
                try:
                    credentials = json.loads(request.credentials_encrypted_json)
                except json.JSONDecodeError:
                    pass
            controls = []
            if request.controls_json:
                try:
                    controls = json.loads(request.controls_json)
                except json.JSONDecodeError:
                    pass
            if not isinstance(controls, list):
                controls = []
            refresh_interval = getattr(request, "refresh_interval", 0) or 0
            new_id = await fetch_value(
                """
                INSERT INTO user_control_panes
                (user_id, name, icon, connector_id, credentials_encrypted, connection_id, controls, is_visible, sort_order, refresh_interval)
                VALUES ($1, $2, $3, $4::uuid, $5::jsonb, $6, $7::jsonb, $8, $9, $10)
                RETURNING id
                """,
                user_id,
                name,
                icon,
                connector_id,
                json.dumps(credentials),
                request.connection_id if request.connection_id else None,
                json.dumps(controls),
                request.is_visible,
                request.sort_order,
                refresh_interval,
            )
            pane_id = str(new_id)
            formatted = f"Created control pane: {name} (ID: {pane_id})"
            try:
                from utils.websocket_manager import get_websocket_manager
                ws_manager = get_websocket_manager()
                if ws_manager:
                    await ws_manager.send_to_session(
                        {"type": "control_pane_updated", "subtype": "pane_created", "pane_id": pane_id},
                        user_id,
                    )
            except Exception as ws_err:
                logger.warning("CreateControlPane WebSocket send failed: %s", ws_err)
            return tool_service_pb2.CreateControlPaneResponse(
                success=True,
                pane_id=pane_id,
                name=name,
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("CreateControlPane failed")
            return tool_service_pb2.CreateControlPaneResponse(success=False, error=str(e))

    async def UpdateControlPane(
        self,
        request: tool_service_pb2.UpdateControlPaneRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateControlPaneResponse:
        """Update a control pane (partial update)."""
        try:
            from services.database_manager.database_helpers import fetch_one, execute
            user_id = request.user_id or "system"
            pane_id = request.pane_id or ""
            if not pane_id:
                return tool_service_pb2.UpdateControlPaneResponse(
                    success=False, error="pane_id required"
                )
            existing = await fetch_one(
                "SELECT id FROM user_control_panes WHERE id = $1 AND user_id = $2",
                pane_id,
                user_id,
            )
            if not existing:
                return tool_service_pb2.UpdateControlPaneResponse(
                    success=False, error="Control pane not found"
                )
            if request.HasField("connector_id") and request.connector_id:
                conn = await fetch_one(
                    "SELECT id FROM data_source_connectors WHERE id = $1 AND (user_id = $2 OR is_template = true)",
                    request.connector_id,
                    user_id,
                )
                if not conn:
                    return tool_service_pb2.UpdateControlPaneResponse(
                        success=False, error="Connector not found"
                    )
            updates = []
            args = []
            idx = 1
            if request.HasField("name"):
                updates.append(f"name = ${idx}")
                args.append(request.name)
                idx += 1
            if request.HasField("icon"):
                updates.append(f"icon = ${idx}")
                args.append(request.icon)
                idx += 1
            if request.HasField("connector_id"):
                updates.append(f"connector_id = ${idx}::uuid")
                args.append(request.connector_id)
                idx += 1
            if request.HasField("credentials_encrypted_json"):
                updates.append(f"credentials_encrypted = ${idx}::jsonb")
                args.append(request.credentials_encrypted_json)
                idx += 1
            if request.HasField("connection_id"):
                updates.append(f"connection_id = ${idx}")
                args.append(request.connection_id)
                idx += 1
            if request.HasField("controls_json"):
                updates.append(f"controls = ${idx}::jsonb")
                args.append(request.controls_json)
                idx += 1
            if request.HasField("is_visible"):
                updates.append(f"is_visible = ${idx}")
                args.append(request.is_visible)
                idx += 1
            if request.HasField("sort_order"):
                updates.append(f"sort_order = ${idx}")
                args.append(request.sort_order)
                idx += 1
            if request.HasField("refresh_interval"):
                updates.append(f"refresh_interval = ${idx}")
                args.append(request.refresh_interval)
                idx += 1
            if not updates:
                try:
                    from utils.websocket_manager import get_websocket_manager
                    ws_manager = get_websocket_manager()
                    if ws_manager:
                        await ws_manager.send_to_session(
                            {"type": "control_pane_updated", "subtype": "pane_updated", "pane_id": pane_id},
                            user_id,
                        )
                except Exception as ws_err:
                    logger.warning("UpdateControlPane WebSocket send failed: %s", ws_err)
                return tool_service_pb2.UpdateControlPaneResponse(
                    success=True, formatted="No updates applied"
                )
            updates.append("updated_at = NOW()")
            args.extend([pane_id, user_id])
            await execute(
                f"UPDATE user_control_panes SET {', '.join(updates)} WHERE id = ${idx}::uuid AND user_id = ${idx + 1}",
                *args,
            )
            try:
                from utils.websocket_manager import get_websocket_manager
                ws_manager = get_websocket_manager()
                if ws_manager:
                    await ws_manager.send_to_session(
                        {"type": "control_pane_updated", "subtype": "pane_updated", "pane_id": pane_id},
                        user_id,
                    )
            except Exception as ws_err:
                logger.warning("UpdateControlPane WebSocket send failed: %s", ws_err)
            return tool_service_pb2.UpdateControlPaneResponse(
                success=True,
                formatted=f"Updated control pane {pane_id}",
            )
        except Exception as e:
            logger.exception("UpdateControlPane failed")
            return tool_service_pb2.UpdateControlPaneResponse(success=False, error=str(e))

    async def DeleteControlPane(
        self,
        request: tool_service_pb2.DeleteControlPaneRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.DeleteControlPaneResponse:
        """Delete a control pane."""
        try:
            from services.database_manager.database_helpers import execute
            user_id = request.user_id or "system"
            pane_id = request.pane_id or ""
            if not pane_id:
                return tool_service_pb2.DeleteControlPaneResponse(
                    success=False, error="pane_id required"
                )
            await execute(
                "DELETE FROM user_control_panes WHERE id = $1::uuid AND user_id = $2",
                pane_id,
                user_id,
            )
            try:
                from utils.websocket_manager import get_websocket_manager
                ws_manager = get_websocket_manager()
                if ws_manager:
                    await ws_manager.send_to_session(
                        {"type": "control_pane_updated", "subtype": "pane_deleted", "pane_id": pane_id},
                        user_id,
                    )
            except Exception as ws_err:
                logger.warning("DeleteControlPane WebSocket send failed: %s", ws_err)
            return tool_service_pb2.DeleteControlPaneResponse(
                success=True,
                formatted=f"Deleted control pane {pane_id}",
            )
        except Exception as e:
            logger.exception("DeleteControlPane failed")
            return tool_service_pb2.DeleteControlPaneResponse(success=False, error=str(e))

    async def ExecuteControlPaneAction(
        self,
        request: tool_service_pb2.ExecuteControlPaneActionRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ExecuteControlPaneActionResponse:
        """Execute a connector endpoint through a saved control pane (same as REST execute)."""
        try:
            from services.database_manager.database_helpers import fetch_one
            from clients.connections_service_client import get_connections_service_client

            user_id = request.user_id or "system"
            pane_id = request.pane_id or ""
            endpoint_id = request.endpoint_id or ""
            if not pane_id or not endpoint_id:
                return tool_service_pb2.ExecuteControlPaneActionResponse(
                    success=False,
                    error="pane_id and endpoint_id required",
                )

            pane = await fetch_one(
                "SELECT id, connector_id, credentials_encrypted, connection_id FROM user_control_panes WHERE id = $1 AND user_id = $2",
                pane_id,
                user_id,
            )
            if not pane:
                return tool_service_pb2.ExecuteControlPaneActionResponse(
                    success=False,
                    error="Control pane not found",
                )

            connector = await fetch_one(
                "SELECT id, definition, connector_type FROM data_source_connectors WHERE id = $1",
                pane["connector_id"],
            )
            if not connector:
                return tool_service_pb2.ExecuteControlPaneActionResponse(
                    success=False,
                    error="Connector not found",
                )

            definition = connector.get("definition") or {}
            if isinstance(definition, str):
                try:
                    definition = json.loads(definition)
                except json.JSONDecodeError:
                    return tool_service_pb2.ExecuteControlPaneActionResponse(
                        success=False,
                        error="Invalid connector definition",
                    )

            credentials = pane.get("credentials_encrypted") or {}
            if isinstance(credentials, str):
                try:
                    credentials = json.loads(credentials)
                except json.JSONDecodeError:
                    credentials = {}
            if not isinstance(credentials, dict):
                credentials = {}

            oauth_token = None
            connection_id = pane.get("connection_id")
            if connection_id is not None:
                from services.external_connections_service import external_connections_service
                oauth_token = await external_connections_service.get_valid_access_token(
                    connection_id,
                    rls_context={"user_id": user_id},
                )
                if not oauth_token:
                    return tool_service_pb2.ExecuteControlPaneActionResponse(
                        success=False,
                        error="Could not obtain token for the selected connection",
                    )

            params = {}
            if request.params_json:
                try:
                    params = json.loads(request.params_json)
                except json.JSONDecodeError:
                    return tool_service_pb2.ExecuteControlPaneActionResponse(
                        success=False,
                        error="Invalid params_json",
                    )

            client = await get_connections_service_client()
            result = await client.execute_connector_endpoint(
                definition=definition,
                credentials=credentials,
                endpoint_id=endpoint_id,
                params=params,
                max_pages=1,
                oauth_token=oauth_token,
                raw_response=True,
                connector_type=connector.get("connector_type"),
            )

            if result.get("error"):
                return tool_service_pb2.ExecuteControlPaneActionResponse(
                    success=False,
                    raw_response_json="{}",
                    records_json="[]",
                    count=0,
                    formatted=result.get("error", ""),
                    error=result.get("error"),
                )

            records = result.get("records", [])
            raw_response = result.get("raw_response")
            formatted = result.get("formatted", "")

            return tool_service_pb2.ExecuteControlPaneActionResponse(
                success=True,
                raw_response_json=json.dumps(raw_response) if raw_response is not None else "{}",
                records_json=json.dumps(records),
                count=len(records),
                formatted=formatted,
            )
        except json.JSONDecodeError as e:
            return tool_service_pb2.ExecuteControlPaneActionResponse(
                success=False,
                error=f"Invalid JSON: {e}",
            )
        except Exception as e:
            logger.exception("ExecuteControlPaneAction failed")
            return tool_service_pb2.ExecuteControlPaneActionResponse(success=False, error=str(e))

    # ===== Agent Factory meta-tools =====

    async def CreateAgentProfile(
        self,
        request: tool_service_pb2.CreateAgentProfileRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateAgentProfileResponse:
        """Create an agent profile via Agent Factory service."""
        try:
            from services import agent_factory_service
            user_id = request.user_id or "system"
            data = {
                "name": request.name or "",
                "handle": request.handle or "",
                "description": request.description if request.description else None,
                "model_preference": request.model_preference if request.model_preference else None,
                "system_prompt_additions": request.system_prompt_additions if request.system_prompt_additions else None,
                "persona_mode": "default" if (request.persona_enabled if request.HasField("persona_enabled") else False) else "none",
                "persona_id": None,
                "include_user_context": False,
                "include_user_facts": False,
                "include_facts_categories": [],
                "auto_routable": request.auto_routable if request.HasField("auto_routable") else False,
                "prompt_history_enabled": request.chat_history_enabled if request.HasField("chat_history_enabled") else False,
                "chat_visible": request.chat_visible if request.HasField("chat_visible") else True,
                "is_active": request.is_active if request.HasField("is_active") else False,
            }
            profile = await agent_factory_service.create_profile(user_id, data)
            h = profile.get("handle") or ""
            formatted = f"Created agent profile: {profile.get('name', '')} ({'@' + h if h else 'schedule/Run-only'}) — ID: {profile.get('id', '')}"
            try:
                from utils.websocket_manager import get_websocket_manager
                ws_manager = get_websocket_manager()
                if ws_manager:
                    await ws_manager.send_to_session(
                        {
                            "type": "agent_factory_updated",
                            "subtype": "profile_created",
                            "agent_id": profile.get("id", ""),
                        },
                        user_id,
                    )
            except Exception as ws_err:
                logger.warning("CreateAgentProfile WebSocket send failed: %s", ws_err)
            return tool_service_pb2.CreateAgentProfileResponse(
                success=True,
                agent_id=profile.get("id", ""),
                name=profile.get("name", ""),
                handle=profile.get("handle", ""),
                formatted=formatted,
            )
        except ValueError as e:
            return tool_service_pb2.CreateAgentProfileResponse(
                success=False, agent_id="", name="", handle="", formatted="", error=str(e)
            )
        except Exception as e:
            logger.exception("CreateAgentProfile failed")
            return tool_service_pb2.CreateAgentProfileResponse(
                success=False, agent_id="", name="", handle="", formatted="", error=str(e)
            )

    async def SetAgentProfileStatus(
        self,
        request: tool_service_pb2.SetAgentProfileStatusRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SetAgentProfileStatusResponse:
        """Update an agent profile's is_active (pause or activate). Separate capability from creating agents."""
        try:
            from services import agent_factory_service
            user_id = request.user_id or "system"
            await agent_factory_service.update_profile(
                user_id,
                request.agent_id,
                {"is_active": request.is_active},
            )
            status = "active" if request.is_active else "paused"
            formatted = f"Agent profile {request.agent_id} set to {status}."
            return tool_service_pb2.SetAgentProfileStatusResponse(
                success=True,
                is_active=request.is_active,
                formatted=formatted,
            )
        except ValueError as e:
            return tool_service_pb2.SetAgentProfileStatusResponse(
                success=False, is_active=False, formatted="", error=str(e)
            )
        except Exception as e:
            logger.exception("SetAgentProfileStatus failed")
            return tool_service_pb2.SetAgentProfileStatusResponse(
                success=False, is_active=False, formatted="", error=str(e)
            )

    async def CreatePlaybook(
        self,
        request: tool_service_pb2.CreatePlaybookRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreatePlaybookResponse:
        """Create a custom playbook via Agent Factory service."""
        try:
            from services import agent_factory_service
            user_id = request.user_id or "system"
            definition = {}
            if request.definition_json:
                try:
                    definition = json.loads(request.definition_json)
                except json.JSONDecodeError:
                    return tool_service_pb2.CreatePlaybookResponse(
                        success=False, playbook_id="", name="", step_count=0,
                        formatted="", error="Invalid definition_json"
                    )
            if request.run_context:
                definition["run_context"] = request.run_context
            data = {
                "name": request.name or "Unnamed",
                "description": request.description if request.description else None,
                "definition": definition,
                "category": request.category if request.category else None,
                "tags": list(request.tags) if request.tags else [],
            }
            warnings = agent_factory_service.validate_playbook_definition(definition)
            playbook = await agent_factory_service.create_playbook(user_id, data)
            steps = (playbook.get("definition") or {}).get("steps") or []
            step_count = len(steps) if isinstance(steps, list) else 0
            formatted = f"Created playbook: {playbook.get('name', '')} ({step_count} steps) — ID: {playbook.get('id', '')}"
            if warnings:
                formatted += "\nValidation warnings: " + "; ".join(warnings[:5])
            try:
                from utils.websocket_manager import get_websocket_manager
                ws_manager = get_websocket_manager()
                if ws_manager:
                    await ws_manager.send_to_session(
                        {"type": "agent_factory_updated", "subtype": "playbook_created", "playbook_id": playbook.get("id", "")},
                        user_id,
                    )
            except Exception as ws_err:
                logger.warning("CreatePlaybook WebSocket send failed: %s", ws_err)
            try:
                from utils.websocket_manager import get_websocket_manager
                ws = get_websocket_manager()
                if ws:
                    await ws.send_to_session(
                        {"type": "agent_factory_updated", "subtype": "playbook_created", "playbook_id": playbook.get("id", "")},
                        user_id,
                    )
            except Exception as ws_err:
                logger.warning("CreatePlaybook WebSocket send failed: %s", ws_err)
            return tool_service_pb2.CreatePlaybookResponse(
                success=True,
                playbook_id=playbook.get("id", ""),
                name=playbook.get("name", ""),
                step_count=step_count,
                validation_warnings=warnings,
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("CreatePlaybook failed")
            return tool_service_pb2.CreatePlaybookResponse(
                success=False, playbook_id="", name="", step_count=0,
                formatted="", error=str(e)
            )

    async def AssignPlaybookToAgent(
        self,
        request: tool_service_pb2.AssignPlaybookToAgentRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.AssignPlaybookToAgentResponse:
        """Assign a playbook to an agent profile (set default_playbook_id)."""
        try:
            from services import agent_factory_service
            user_id = request.user_id or "system"
            await agent_factory_service.update_profile(
                user_id,
                request.agent_id,
                {"default_playbook_id": request.playbook_id},
            )
            formatted = f"Assigned playbook {request.playbook_id} to agent {request.agent_id}."
            return tool_service_pb2.AssignPlaybookToAgentResponse(
                success=True,
                formatted=formatted,
            )
        except ValueError as e:
            return tool_service_pb2.AssignPlaybookToAgentResponse(
                success=False, formatted="", error=str(e)
            )
        except Exception as e:
            logger.exception("AssignPlaybookToAgent failed")
            return tool_service_pb2.AssignPlaybookToAgentResponse(
                success=False, formatted="", error=str(e)
            )

    async def CreateAgentSchedule(
        self,
        request: tool_service_pb2.CreateAgentScheduleRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateAgentScheduleResponse:
        """Create a schedule for an agent profile."""
        try:
            from services import agent_factory_service
            user_id = request.user_id or "system"
            data = {
                "schedule_type": request.schedule_type or "cron",
                "cron_expression": request.cron_expression if request.cron_expression else None,
                "interval_seconds": request.interval_seconds if request.interval_seconds else None,
                "timezone": request.timezone if request.timezone else "UTC",
                "is_active": request.is_active if request.HasField("is_active") else False,
                "input_context": {},
            }
            if request.input_context_json:
                try:
                    data["input_context"] = json.loads(request.input_context_json)
                except json.JSONDecodeError:
                    pass
            schedule = await agent_factory_service.create_schedule(
                user_id,
                request.agent_id,
                data,
            )
            next_run = schedule.get("next_run_at") or ""
            is_active = schedule.get("is_active", False)
            formatted = f"Created schedule for agent {request.agent_id} — next run: {next_run}, active: {is_active}"
            return tool_service_pb2.CreateAgentScheduleResponse(
                success=True,
                schedule_id=schedule.get("id", ""),
                next_run_at=next_run,
                is_active=is_active,
                formatted=formatted,
            )
        except ValueError as e:
            return tool_service_pb2.CreateAgentScheduleResponse(
                success=False, schedule_id="", next_run_at="", is_active=False,
                formatted="", error=str(e)
            )
        except Exception as e:
            logger.exception("CreateAgentSchedule failed")
            return tool_service_pb2.CreateAgentScheduleResponse(
                success=False, schedule_id="", next_run_at="", is_active=False,
                formatted="", error=str(e)
            )

    async def BindDataSourceToAgent(
        self,
        request: tool_service_pb2.BindDataSourceToAgentRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.BindDataSourceToAgentResponse:
        """Bind a data source connector to an agent profile."""
        try:
            from services import agent_factory_service
            user_id = request.user_id or "system"
            data = {
                "connector_id": request.connector_id,
                "config_overrides": {},
                "permissions": {},
                "is_enabled": True,
            }
            if request.config_overrides_json:
                try:
                    data["config_overrides"] = json.loads(request.config_overrides_json)
                except json.JSONDecodeError:
                    pass
            if request.permissions_json:
                try:
                    data["permissions"] = json.loads(request.permissions_json)
                except json.JSONDecodeError:
                    pass
            binding = await agent_factory_service.create_data_source_binding(
                user_id,
                request.agent_id,
                data,
            )
            formatted = f"Bound connector {request.connector_id} to agent {request.agent_id} — binding ID: {binding.get('id', '')}"
            return tool_service_pb2.BindDataSourceToAgentResponse(
                success=True,
                binding_id=binding.get("id", ""),
                formatted=formatted,
            )
        except ValueError as e:
            return tool_service_pb2.BindDataSourceToAgentResponse(
                success=False, binding_id="", formatted="", error=str(e)
            )
        except Exception as e:
            logger.exception("BindDataSourceToAgent failed")
            return tool_service_pb2.BindDataSourceToAgentResponse(
                success=False, binding_id="", formatted="", error=str(e)
            )

    async def ListAvailableLlmModels(
        self,
        request: tool_service_pb2.ListAvailableLlmModelsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListAvailableLlmModelsResponse:
        """Return the list of LLM models available to the user (for Agent Factory model_preference)."""
        try:
            from services.model_source_resolver import get_available_models as resolver_get_available_models
            user_id = request.user_id or "system"
            models = await resolver_get_available_models(user_id)
            proto_models = []
            for m in models:
                mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None) or ""
                name = getattr(m, "name", None) or (m.get("name") if isinstance(m, dict) else None) or mid
                prov = getattr(m, "provider", None) or (m.get("provider") if isinstance(m, dict) else "") or ""
                proto_models.append(
                    tool_service_pb2.LlmModelInfo(
                        model_id=str(mid),
                        display_name=str(name),
                        provider=str(prov),
                    )
                )
            lines = [f"Available models ({len(models)}):"] if models else ["No models configured for this user."]
            for m in models:
                mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
                name = getattr(m, "name", None) or (m.get("name") if isinstance(m, dict) else None)
                prov = getattr(m, "provider", None) or (m.get("provider") if isinstance(m, dict) else "")
                lines.append(f"- {mid} ({name or mid}) [{prov}]")
            formatted = "\n".join(lines)
            return tool_service_pb2.ListAvailableLlmModelsResponse(
                success=True,
                models=proto_models,
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("ListAvailableLlmModels failed")
            return tool_service_pb2.ListAvailableLlmModelsResponse(
                success=False,
                models=[],
                formatted="",
                error=str(e),
            )

    async def UpdateAgentProfile(
        self,
        request: tool_service_pb2.UpdateAgentProfileRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateAgentProfileResponse:
        """Update an agent profile. Lock enforced in service (only is_active/is_locked when locked)."""
        try:
            from services import agent_factory_service
            from utils.websocket_manager import get_websocket_manager

            user_id = request.user_id or "system"
            agent_id = (request.agent_id or "").strip()
            if not agent_id:
                return tool_service_pb2.UpdateAgentProfileResponse(
                    success=False, agent_id="", name="", formatted="", error="agent_id required"
                )
            updates = {}
            if request.updates_json:
                try:
                    updates = json.loads(request.updates_json)
                except json.JSONDecodeError:
                    return tool_service_pb2.UpdateAgentProfileResponse(
                        success=False, agent_id=agent_id, name="", formatted="", error="Invalid updates_json"
                    )
            if not isinstance(updates, dict):
                return tool_service_pb2.UpdateAgentProfileResponse(
                    success=False, agent_id=agent_id, name="", formatted="", error="updates_json must be a JSON object"
                )
            profile = await agent_factory_service.update_profile(user_id, agent_id, updates)
            formatted = f"Updated agent profile: {profile.get('name', '')} (ID: {agent_id})"
            try:
                ws_manager = get_websocket_manager()
                if ws_manager:
                    await ws_manager.send_to_session(
                        {"type": "agent_factory_updated", "subtype": "profile_updated", "agent_id": agent_id},
                        user_id,
                    )
            except Exception as ws_err:
                logger.warning("UpdateAgentProfile WebSocket send failed: %s", ws_err)
            return tool_service_pb2.UpdateAgentProfileResponse(
                success=True,
                agent_id=agent_id,
                name=profile.get("name", ""),
                formatted=formatted,
            )
        except ValueError as e:
            return tool_service_pb2.UpdateAgentProfileResponse(
                success=False, agent_id=request.agent_id or "", name="", formatted="", error=str(e)
            )
        except Exception as e:
            logger.exception("UpdateAgentProfile failed")
            return tool_service_pb2.UpdateAgentProfileResponse(
                success=False, agent_id=request.agent_id or "", name="", formatted="", error=str(e)
            )

    async def DeleteAgentProfile(
        self,
        request: tool_service_pb2.DeleteAgentProfileRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.DeleteAgentProfileResponse:
        """Delete an agent profile. Blocked when profile is locked."""
        try:
            from services.database_manager.database_helpers import fetch_one, execute
            from utils.websocket_manager import get_websocket_manager

            user_id = request.user_id or "system"
            agent_id = (request.agent_id or "").strip()
            if not agent_id:
                return tool_service_pb2.DeleteAgentProfileResponse(
                    success=False, formatted="", error="agent_id required"
                )
            row = await fetch_one(
                "SELECT id, is_locked FROM agent_profiles WHERE id = $1 AND user_id = $2",
                agent_id,
                user_id,
            )
            if not row:
                return tool_service_pb2.DeleteAgentProfileResponse(
                    success=False, formatted="", error="Profile not found"
                )
            if row.get("is_locked"):
                return tool_service_pb2.DeleteAgentProfileResponse(
                    success=False, formatted="", error="Profile is locked; unlock to delete"
                )
            await execute("DELETE FROM agent_profiles WHERE id = $1 AND user_id = $2", agent_id, user_id)
            formatted = f"Deleted agent profile {agent_id}."
            try:
                ws_manager = get_websocket_manager()
                if ws_manager:
                    await ws_manager.send_to_session(
                        {"type": "agent_factory_updated", "subtype": "profile_deleted", "agent_id": agent_id},
                        user_id,
                    )
            except Exception as ws_err:
                logger.warning("DeleteAgentProfile WebSocket send failed: %s", ws_err)
            return tool_service_pb2.DeleteAgentProfileResponse(success=True, formatted=formatted)
        except Exception as e:
            logger.exception("DeleteAgentProfile failed")
            return tool_service_pb2.DeleteAgentProfileResponse(success=False, formatted="", error=str(e))

    async def UpdatePlaybook(
        self,
        request: tool_service_pb2.UpdatePlaybookRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdatePlaybookResponse:
        """Update a playbook. Lock: only is_locked toggle allowed when locked. Templates are read-only."""
        try:
            import uuid
            from services import agent_factory_service
            from services.database_manager.database_helpers import fetch_one, execute
            from utils.websocket_manager import get_websocket_manager

            user_id = request.user_id or "system"
            playbook_id = (request.playbook_id or "").strip()
            if not playbook_id:
                return tool_service_pb2.UpdatePlaybookResponse(
                    success=False, playbook_id="", name="", step_count=0,
                    validation_warnings=[], formatted="", error="playbook_id required"
                )
            # If identifier is not a UUID, resolve by playbook name or slug (e.g. "morning-intelligence-briefing")
            try:
                uuid.UUID(playbook_id)
            except (ValueError, TypeError):
                slug_normalized = playbook_id.replace("_", "-").lower()
                resolved = await fetch_one(
                    """SELECT id FROM custom_playbooks
                       WHERE user_id = $1 AND (
                         name = $2
                         OR LOWER(REGEXP_REPLACE(TRIM(name), '\\s+', '-')) = $3
                       )
                       LIMIT 1""",
                    user_id,
                    playbook_id,
                    slug_normalized,
                )
                if not resolved:
                    return tool_service_pb2.UpdatePlaybookResponse(
                        success=False, playbook_id=playbook_id, name="", step_count=0,
                        validation_warnings=[], formatted="", error="Playbook not found"
                    )
                playbook_id = str(resolved["id"])
            row = await fetch_one(
                "SELECT * FROM custom_playbooks WHERE id = $1 AND user_id = $2",
                playbook_id,
                user_id,
            )
            if not row:
                return tool_service_pb2.UpdatePlaybookResponse(
                    success=False, playbook_id=playbook_id, name="", step_count=0,
                    validation_warnings=[], formatted="", error="Playbook not found"
                )
            if row.get("is_template"):
                return tool_service_pb2.UpdatePlaybookResponse(
                    success=False, playbook_id=playbook_id, name="", step_count=0,
                    validation_warnings=[], formatted="", error="Cannot update template playbook"
                )
            updates = {}
            if request.updates_json:
                try:
                    updates = json.loads(request.updates_json)
                except json.JSONDecodeError:
                    return tool_service_pb2.UpdatePlaybookResponse(
                        success=False, playbook_id=playbook_id, name="", step_count=0,
                        validation_warnings=[], formatted="", error="Invalid updates_json"
                    )
            if not isinstance(updates, dict):
                return tool_service_pb2.UpdatePlaybookResponse(
                    success=False, playbook_id=playbook_id, name="", step_count=0,
                    validation_warnings=[], formatted="", error="updates_json must be a JSON object"
                )
            allowed_keys = {"name", "description", "version", "definition", "triggers", "is_template", "category", "tags", "required_connectors", "is_locked"}
            updates = {k: v for k, v in updates.items() if k in allowed_keys}
            if not updates:
                pb = agent_factory_service._row_to_playbook(row)
                step_count = len((pb.get("definition") or {}).get("steps") or [])
                return tool_service_pb2.UpdatePlaybookResponse(
                    success=True,
                    playbook_id=playbook_id,
                    name=pb.get("name", ""),
                    step_count=step_count,
                    formatted=f"Playbook unchanged: {pb.get('name', '')} (ID: {playbook_id})",
                )
            if row.get("is_locked") and set(updates.keys()) != {"is_locked"}:
                return tool_service_pb2.UpdatePlaybookResponse(
                    success=False, playbook_id=playbook_id, name="", step_count=0,
                    validation_warnings=[], formatted="", error="Playbook is locked; only lock toggle is allowed"
                )
            playbook_remediation_msgs: list = []
            playbook_remediation_steps: list = []
            if "definition" in updates:
                defn = updates["definition"]
                old_def = row.get("definition")
                if isinstance(old_def, str):
                    try:
                        old_def = json.loads(old_def) if old_def else {}
                    except (json.JSONDecodeError, TypeError):
                        old_def = {}
                if not isinstance(old_def, dict):
                    old_def = {}
                if isinstance(defn, dict) and defn.get("steps") and old_def:
                    agent_factory_service.merge_playbook_definition_steps(old_def, defn)
                if isinstance(defn, dict):
                    defn, playbook_remediation_steps, playbook_remediation_msgs = (
                        await agent_factory_service.validate_and_remediate_playbook_models_for_user(
                            user_id, defn
                        )
                    )
                    updates["definition"] = defn
            warnings = []
            if "definition" in updates:
                warnings = agent_factory_service.validate_playbook_definition(updates.get("definition") or {})
            set_clauses = []
            args = []
            idx = 1
            jsonb_fields = ("definition", "triggers")
            array_fields = ("tags", "required_connectors")
            for k, v in updates.items():
                if k in jsonb_fields:
                    set_clauses.append(f"{k} = ${idx}::jsonb")
                    args.append(json.dumps(v) if isinstance(v, (dict, list)) else v)
                elif k in array_fields:
                    set_clauses.append(f"{k} = ${idx}")
                    args.append(v)
                else:
                    set_clauses.append(f"{k} = ${idx}")
                    args.append(v)
                idx += 1
            set_clauses.append("updated_at = NOW()")
            args.extend([playbook_id, user_id])
            await execute(
                f"UPDATE custom_playbooks SET {', '.join(set_clauses)} WHERE id = ${idx}::uuid AND user_id = ${idx + 1}",
                *args,
            )
            row = await fetch_one("SELECT * FROM custom_playbooks WHERE id = $1", playbook_id)
            pb = agent_factory_service._row_to_playbook(row)
            step_count = len((pb.get("definition") or {}).get("steps") or [])
            formatted = f"Updated playbook: {pb.get('name', '')} ({step_count} steps) — ID: {playbook_id}"
            if warnings:
                formatted += "\nValidation warnings: " + "; ".join(warnings[:5])
            if playbook_remediation_msgs:
                await agent_factory_service.notify_playbook_model_remediation(
                    user_id, playbook_id, playbook_remediation_steps, playbook_remediation_msgs
                )
            try:
                ws_manager = get_websocket_manager()
                if ws_manager:
                    await ws_manager.send_to_session(
                        {"type": "agent_factory_updated", "subtype": "playbook_updated", "playbook_id": playbook_id},
                        user_id,
                    )
            except Exception as ws_err:
                logger.warning("UpdatePlaybook WebSocket send failed: %s", ws_err)
            return tool_service_pb2.UpdatePlaybookResponse(
                success=True,
                playbook_id=playbook_id,
                name=pb.get("name", ""),
                step_count=step_count,
                validation_warnings=warnings[:10],
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("UpdatePlaybook failed")
            return tool_service_pb2.UpdatePlaybookResponse(
                success=False, playbook_id=request.playbook_id or "", name="", step_count=0,
                validation_warnings=[], formatted="", error=str(e)
            )

    async def DeletePlaybook(
        self,
        request: tool_service_pb2.DeletePlaybookRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.DeletePlaybookResponse:
        """Delete a playbook. Blocked when locked or when playbook is a template."""
        try:
            import uuid
            from services.database_manager.database_helpers import fetch_one, execute
            from utils.websocket_manager import get_websocket_manager

            user_id = request.user_id or "system"
            playbook_id = (request.playbook_id or "").strip()
            if not playbook_id:
                return tool_service_pb2.DeletePlaybookResponse(success=False, formatted="", error="playbook_id required")
            # If identifier is not a UUID, resolve by playbook name or slug
            try:
                uuid.UUID(playbook_id)
            except (ValueError, TypeError):
                slug_normalized = playbook_id.replace("_", "-").lower()
                resolved = await fetch_one(
                    """SELECT id FROM custom_playbooks
                       WHERE user_id = $1 AND (
                         name = $2
                         OR LOWER(REGEXP_REPLACE(TRIM(name), '\\s+', '-')) = $3
                       )
                       LIMIT 1""",
                    user_id,
                    playbook_id,
                    slug_normalized,
                )
                if not resolved:
                    return tool_service_pb2.DeletePlaybookResponse(
                        success=False, formatted="", error="Playbook not found"
                    )
                playbook_id = str(resolved["id"])
            row = await fetch_one(
                "SELECT id, is_template, is_locked FROM custom_playbooks WHERE id = $1 AND user_id = $2",
                playbook_id,
                user_id,
            )
            if not row:
                return tool_service_pb2.DeletePlaybookResponse(
                    success=False, formatted="", error="Playbook not found"
                )
            if row.get("is_template"):
                return tool_service_pb2.DeletePlaybookResponse(
                    success=False, formatted="", error="Cannot delete template playbook"
                )
            if row.get("is_locked"):
                return tool_service_pb2.DeletePlaybookResponse(
                    success=False, formatted="", error="Playbook is locked; unlock to delete"
                )
            await execute("DELETE FROM custom_playbooks WHERE id = $1 AND user_id = $2", playbook_id, user_id)
            formatted = f"Deleted playbook {playbook_id}."
            try:
                ws_manager = get_websocket_manager()
                if ws_manager:
                    await ws_manager.send_to_session(
                        {"type": "agent_factory_updated", "subtype": "playbook_deleted", "playbook_id": playbook_id},
                        user_id,
                    )
            except Exception as ws_err:
                logger.warning("DeletePlaybook WebSocket send failed: %s", ws_err)
            return tool_service_pb2.DeletePlaybookResponse(success=True, formatted=formatted)
        except Exception as e:
            logger.exception("DeletePlaybook failed")
            return tool_service_pb2.DeletePlaybookResponse(success=False, formatted="", error=str(e))

    # ===== Agent-Initiated Notifications =====

    async def SendOutboundMessage(
        self,
        request: tool_service_pb2.SendOutboundMessageRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SendOutboundMessageResponse:
        """Send a proactive outbound message via a messaging bot (Telegram, Discord)."""
        try:
            from clients.connections_service_client import get_connections_service_client
            client = await get_connections_service_client()
            result = await client.send_outbound_message(
                user_id=request.user_id or "system",
                provider=request.provider or "",
                connection_id=request.connection_id or "",
                message=request.message or "",
                format=request.format or "markdown",
                recipient_chat_id=getattr(request, "recipient_chat_id", None) or "",
            )
            return tool_service_pb2.SendOutboundMessageResponse(
                success=result.get("success", False),
                message_id=result.get("message_id", ""),
                channel=result.get("channel", ""),
                error=result.get("error", ""),
            )
        except Exception as e:
            logger.error("SendOutboundMessage failed: %s", e)
            return tool_service_pb2.SendOutboundMessageResponse(
                success=False, message_id="", channel="", error=str(e)
            )

    async def CreateAgentConversation(
        self,
        request: tool_service_pb2.CreateAgentConversationRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateAgentConversationResponse:
        """Create or append to an agent-initiated conversation via backend API (ensures WebSocket events fire)."""
        import os

        user_id = request.user_id or "system"
        msg = (request.message or "").strip()
        if not msg:
            return tool_service_pb2.CreateAgentConversationResponse(
                success=False, conversation_id="", message_id="", error="Message is required"
            )

        backend_url = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
        internal_key = os.getenv("INTERNAL_SERVICE_KEY", "")
        if not internal_key:
            logger.warning("CreateAgentConversation: INTERNAL_SERVICE_KEY not set; backend may reject request")

        payload = {
            "user_id": user_id,
            "message": msg,
            "agent_name": request.agent_name or "",
            "agent_profile_id": request.agent_profile_id or "",
            "title": request.title or "",
            "conversation_id": request.conversation_id or "",
        }
        url = f"{backend_url}/api/internal/agent-conversation"
        headers = {"Content-Type": "application/json"}
        if internal_key:
            headers["X-Internal-Service-Key"] = internal_key

        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                err = resp.text or f"HTTP {resp.status_code}"
                logger.error("CreateAgentConversation backend call failed: %s", err)
                return tool_service_pb2.CreateAgentConversationResponse(
                    success=False, conversation_id="", message_id="", error=err[:500]
                )
            data = resp.json()
            return tool_service_pb2.CreateAgentConversationResponse(
                success=True,
                conversation_id=data.get("conversation_id", ""),
                message_id=data.get("message_id", ""),
                error="",
            )
        except Exception as e:
            logger.error("CreateAgentConversation failed: %s", e)
            return tool_service_pb2.CreateAgentConversationResponse(
                success=False, conversation_id="", message_id="", error=str(e)[:500]
            )

    async def CreateAgentMessage(
        self,
        request: tool_service_pb2.CreateAgentMessageRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateAgentMessageResponse:
        """Create an inter-agent message (team timeline)."""
        import json

        try:
            from services import agent_message_service

            user_id = request.user_id or "system"
            team_id = (request.team_id or "").strip()
            if not team_id:
                return tool_service_pb2.CreateAgentMessageResponse(
                    success=False, message_id="", message_json="", error="team_id is required"
                )
            from_agent_id = (request.from_agent_id or "").strip() or None
            to_agent_id = (request.to_agent_id or "").strip() or None
            message_type = (request.message_type or "report").strip()
            content = request.content or ""
            metadata = {}
            if request.metadata_json:
                try:
                    metadata = json.loads(request.metadata_json)
                except (json.JSONDecodeError, TypeError):
                    pass
            parent_message_id = (request.parent_message_id or "").strip() or None

            msg = await agent_message_service.create_message(
                line_id=team_id,
                from_agent_id=from_agent_id,
                to_agent_id=to_agent_id,
                message_type=message_type,
                content=content,
                metadata=metadata,
                parent_message_id=parent_message_id,
                user_id=user_id,
            )
            if metadata.get("trigger_dispatch") and to_agent_id and team_id:
                try:
                    from services import agent_line_service as _als
                    from services.celery_tasks.team_heartbeat_tasks import dispatch_worker_for_message

                    _team = await _als.get_line(team_id, user_id)
                    if not _team or str(_team.get("status") or "").lower() != "active":
                        logger.info("Skipping dispatch_worker_for_message: line not active")
                    else:
                        dispatch_worker_for_message.apply_async(
                            args=[team_id, user_id, to_agent_id, msg.get("id", ""), from_agent_id or ""],
                            countdown=2,
                        )
                except Exception as _dispatch_err:
                    logger.warning("Failed to enqueue dispatch_worker_for_message: %s", _dispatch_err)
            return tool_service_pb2.CreateAgentMessageResponse(
                success=True,
                message_id=msg.get("id", ""),
                message_json=json.dumps(msg),
                error="",
            )
        except Exception as e:
            logger.exception("CreateAgentMessage failed")
            return tool_service_pb2.CreateAgentMessageResponse(
                success=False, message_id="", message_json="", error=str(e)[:500]
            )

    async def AppendLineAgentChatMessage(
        self,
        request: tool_service_pb2.AppendLineAgentChatMessageRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.AppendLineAgentChatMessageResponse:
        """Persist a line sub-agent assistant turn into the user chat conversation."""
        import json

        try:
            from services.conversation_service import ConversationService
            from services.database_manager.database_helpers import fetch_one
            from utils.websocket_manager import get_websocket_manager

            user_id = (request.user_id or "system").strip()
            conversation_id = (request.conversation_id or "").strip()
            content = (request.content or "").strip()
            if not conversation_id or not content:
                return tool_service_pb2.AppendLineAgentChatMessageResponse(
                    success=False, message_id="", error="conversation_id and content are required"
                )

            agent_profile_id = (request.agent_profile_id or "").strip()
            line_id = (request.line_id or "").strip()
            line_role = (request.line_role or "").strip()
            if line_id and agent_profile_id and not line_role:
                try:
                    row = await fetch_one(
                        """
                        SELECT role FROM agent_line_memberships
                        WHERE line_id = $1::uuid AND agent_profile_id = $2::uuid
                        LIMIT 1
                        """,
                        line_id,
                        agent_profile_id,
                    )
                    if row and row.get("role"):
                        line_role = str(row["role"])
                except Exception as role_err:
                    logger.debug("Line role lookup skipped: %s", role_err)

            extra: Dict[str, Any] = {}
            if request.metadata_json:
                try:
                    extra = json.loads(request.metadata_json)
                    if not isinstance(extra, dict):
                        extra = {}
                except (json.JSONDecodeError, TypeError):
                    extra = {}

            msg_meta: Dict[str, Any] = {
                "orchestrator_system": True,
                "line_dispatch_sub_agent": True,
                "delegated_agent": (request.agent_display_name or "line_agent").strip() or "line_agent",
                "agent_profile_id": agent_profile_id or None,
                "agent_display_name": (request.agent_display_name or "").strip() or None,
                "line_id": line_id or None,
                "line_role": line_role or None,
                "line_agent_handle": (request.line_agent_handle or "").strip() or None,
                "delegated_by": (request.delegated_by_agent_id or "").strip() or None,
            }
            msg_meta.update(extra)
            msg_meta = {k: v for k, v in msg_meta.items() if v is not None}

            conversation_service = ConversationService()
            conversation_service.set_current_user(user_id)
            saved = await conversation_service.add_message(
                conversation_id=conversation_id,
                user_id=user_id,
                role="assistant",
                content=content,
                metadata=msg_meta,
            )
            mid = ""
            if saved and isinstance(saved, dict):
                mid = str(saved.get("message_id") or saved.get("id") or "")
            try:
                ws = get_websocket_manager()
                await ws.send_line_agent_chat_update(
                    conversation_id,
                    user_id,
                    {
                        "message_id": mid,
                        "content": content,
                        "role": "assistant",
                        "metadata": msg_meta,
                    },
                )
            except Exception as ws_err:
                logger.warning("AppendLineAgentChatMessage WebSocket push failed: %s", ws_err)

            return tool_service_pb2.AppendLineAgentChatMessageResponse(
                success=True, message_id=mid, error=""
            )
        except Exception as e:
            logger.exception("AppendLineAgentChatMessage failed")
            return tool_service_pb2.AppendLineAgentChatMessageResponse(
                success=False, message_id="", error=str(e)[:500]
            )

    async def ReadTeamTimeline(
        self,
        request: tool_service_pb2.ReadTeamTimelineRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ReadTeamTimelineResponse:
        """Return recent team timeline messages for agent context."""
        import json
        try:
            from services import agent_message_service
            user_id = request.user_id or "system"
            team_id = (request.team_id or "").strip()
            if not team_id:
                return tool_service_pb2.ReadTeamTimelineResponse(
                    success=False, items_json="[]", total=0, error="team_id is required"
                )
            limit = max(1, min(100, request.limit or 20))
            since_hours = request.since_hours or 0
            since = None
            if since_hours > 0:
                from datetime import datetime, timezone, timedelta
                since = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()
            result = await agent_message_service.get_line_timeline(
                line_id=team_id,
                user_id=user_id,
                limit=limit,
                offset=0,
                since=since,
            )
            items = result.get("items") or []
            total = result.get("total") or 0
            return tool_service_pb2.ReadTeamTimelineResponse(
                success=True,
                items_json=json.dumps(items),
                total=total,
                error="",
            )
        except Exception as e:
            logger.exception("ReadTeamTimeline failed")
            return tool_service_pb2.ReadTeamTimelineResponse(
                success=False, items_json="[]", total=0, error=str(e)[:500]
            )

    async def ReadAgentMessages(
        self,
        request: tool_service_pb2.ReadAgentMessagesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ReadAgentMessagesResponse:
        """Return messages to/from a specific agent in a team."""
        import json
        try:
            from services import agent_message_service
            user_id = request.user_id or "system"
            team_id = (request.team_id or "").strip()
            agent_profile_id = (request.agent_profile_id or "").strip()
            if not team_id or not agent_profile_id:
                return tool_service_pb2.ReadAgentMessagesResponse(
                    success=False, items_json="[]", total=0, error="team_id and agent_profile_id are required"
                )
            limit = max(1, min(100, request.limit or 50))
            result = await agent_message_service.get_agent_messages(
                agent_profile_id=agent_profile_id,
                line_id=team_id,
                user_id=user_id,
                limit=limit,
                offset=0,
            )
            items = result.get("items") or []
            total = result.get("total") or 0
            return tool_service_pb2.ReadAgentMessagesResponse(
                success=True,
                items_json=json.dumps(items),
                total=total,
                error="",
            )
        except Exception as e:
            logger.exception("ReadAgentMessages failed")
            return tool_service_pb2.ReadAgentMessagesResponse(
                success=False, items_json="[]", total=0, error=str(e)[:500]
            )

    async def GetTeamStatusBoard(
        self,
        request: tool_service_pb2.GetTeamStatusBoardRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetTeamStatusBoardResponse:
        """Return composed team overview: members with tasks, goals, last activity."""
        import json
        try:
            from datetime import datetime
            from services import agent_line_service, agent_task_service, agent_goal_service
            from services.database_manager.database_helpers import fetch_all

            user_id = request.user_id or "system"
            team_id = (request.team_id or "").strip()
            if not team_id:
                return tool_service_pb2.GetTeamStatusBoardResponse(
                    success=False, board_json="{}", error="team_id is required"
                )
            team = await agent_line_service.get_line(team_id, user_id)
            if not team:
                return tool_service_pb2.GetTeamStatusBoardResponse(
                    success=False, board_json="{}", error="Team not found"
                )
            members = team.get("members") or []
            tasks = await agent_task_service.list_line_tasks(team_id, user_id)
            pending_tasks = [t for t in tasks if t.get("status") not in ("done", "cancelled")]
            goals_tree = await agent_goal_service.get_goal_tree(team_id, user_id)

            def flatten_goals(nodes, out):
                for n in nodes or []:
                    if n.get("status") in ("done", "cancelled"):
                        continue
                    out.append(n)
                    flatten_goals(n.get("children") or [], out)

            goals_flat = []
            flatten_goals(goals_tree, goals_flat)
            tasks_by_agent = {}
            for t in pending_tasks:
                aid = t.get("assigned_agent_id")
                if aid:
                    tasks_by_agent.setdefault(str(aid), []).append(t)
            goals_by_agent = {}
            for g in goals_flat:
                aid = g.get("assigned_agent_id")
                if aid:
                    goals_by_agent.setdefault(str(aid), []).append(g)

            member_ids = [m["agent_profile_id"] for m in members if m.get("agent_profile_id")]
            last_msg_per_agent = {}
            last_exec_per_agent = {}
            if member_ids:
                # agent_messages: line_id = $1, member ids = $2, $3, ...
                msg_placeholders = ",".join([f"${i+2}" for i in range(len(member_ids))])
                rows = await fetch_all(
                    f"""
                    SELECT DISTINCT ON (from_agent_id) from_agent_id, created_at FROM agent_messages
                    WHERE line_id = $1 AND from_agent_id IN ({msg_placeholders})
                    ORDER BY from_agent_id, created_at DESC
                    """,
                    team_id,
                    *member_ids,
                )
                for r in rows:
                    last_msg_per_agent[str(r["from_agent_id"])] = r.get("created_at")
                # agent_execution_log: only member ids, so use $1, $2, ...
                exec_placeholders = ",".join([f"${i+1}" for i in range(len(member_ids))])
                exec_rows = await fetch_all(
                    f"""
                    SELECT DISTINCT ON (agent_profile_id) agent_profile_id, started_at FROM agent_execution_log
                    WHERE agent_profile_id IN ({exec_placeholders})
                    ORDER BY agent_profile_id, started_at DESC
                    """,
                    *member_ids,
                )
                for r in exec_rows:
                    last_exec_per_agent[str(r["agent_profile_id"])] = r.get("started_at")

            membership_id_to_agent = {
                str(m["id"]): {
                    "agent_profile_id": str(m["agent_profile_id"]),
                    "agent_name": m.get("agent_name") or m.get("agent_handle") or "Unknown",
                    "agent_handle": m.get("agent_handle") or "",
                }
                for m in members
                if m.get("id") and m.get("agent_profile_id")
            }
            board_members = []
            for m in members:
                aid = m.get("agent_profile_id")
                if not aid:
                    continue
                sid = str(aid)
                mid = str(m.get("id", ""))
                direct_reports = [
                    membership_id_to_agent[str(m2["id"])]
                    for m2 in members
                    if str(m2.get("reports_to") or "") == mid
                ]
                manager = membership_id_to_agent.get(str(m.get("reports_to") or "")) if m.get("reports_to") else None
                peers = [
                    membership_id_to_agent[str(m2["id"])]
                    for m2 in members
                    if str(m2.get("reports_to") or "") == str(m.get("reports_to") or "")
                    and str(m2.get("agent_profile_id")) != sid
                ]
                last_task_at = None
                for t in tasks_by_agent.get(sid, []):
                    u = t.get("updated_at")
                    if u:
                        try:
                            ut = datetime.fromisoformat(str(u).replace("Z", "+00:00"))
                            if last_task_at is None or ut > last_task_at:
                                last_task_at = ut
                        except (ValueError, TypeError):
                            pass
                last_msg_at = last_msg_per_agent.get(sid)
                last_exec_at = last_exec_per_agent.get(sid)
                last_activity = None
                for ts in (last_msg_at, last_task_at, last_exec_at):
                    if ts:
                        try:
                            t = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                            if last_activity is None or t > last_activity:
                                last_activity = t
                        except (ValueError, TypeError):
                            pass
                last_activity_at = None
                if last_activity:
                    last_activity_at = last_activity.isoformat() if hasattr(last_activity, "isoformat") else str(last_activity)
                board_members.append({
                    "agent_profile_id": sid,
                    "agent_name": m.get("agent_name") or m.get("agent_handle") or "Unknown",
                    "agent_handle": m.get("agent_handle") or "",
                    "role": m.get("role") or "worker",
                    "tasks": tasks_by_agent.get(sid, []),
                    "goals": goals_by_agent.get(sid, []),
                    "last_activity_at": last_activity_at,
                    "task_count": len(tasks_by_agent.get(sid, [])),
                    "goal_count": len(goals_by_agent.get(sid, [])),
                    "direct_reports": direct_reports,
                    "reports_to_agent_id": manager["agent_profile_id"] if manager else None,
                    "reports_to_agent_name": manager.get("agent_name") if manager else None,
                    "peers": peers,
                })
            board = {
                "team_name": team.get("name", "Team"),
                "line_id": team_id,
                "members": board_members,
            }
            return tool_service_pb2.GetTeamStatusBoardResponse(
                success=True,
                board_json=json.dumps(board),
                error="",
            )
        except Exception as e:
            logger.exception("GetTeamStatusBoard failed")
            return tool_service_pb2.GetTeamStatusBoardResponse(
                success=False, board_json="{}", error=str(e)[:500]
            )

    async def SetWorkspaceEntry(
        self,
        request: tool_service_pb2.SetWorkspaceEntryRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SetWorkspaceEntryResponse:
        """Upsert a team workspace entry (Blackboard pattern)."""
        try:
            from services import agent_workspace_service
            user_id = request.user_id or "system"
            team_id = (request.team_id or "").strip()
            key = (request.key or "").strip()
            value = request.value or ""
            updated_by_agent_id = (request.updated_by_agent_id or "").strip() or None
            if not team_id or not key:
                return tool_service_pb2.SetWorkspaceEntryResponse(
                    success=False, key=key, error="team_id and key are required"
                )
            result = await agent_workspace_service.set_workspace_entry(
                line_id=team_id,
                key=key,
                value=value,
                user_id=user_id,
                updated_by_agent_id=updated_by_agent_id,
            )
            if not result.get("success"):
                return tool_service_pb2.SetWorkspaceEntryResponse(
                    success=False, key=key, error=(result.get("error") or "Failed")[:500]
                )
            updated_at = result.get("updated_at") or ""
            return tool_service_pb2.SetWorkspaceEntryResponse(
                success=True, key=key, updated_at=updated_at, error=""
            )
        except Exception as e:
            logger.exception("SetWorkspaceEntry failed")
            return tool_service_pb2.SetWorkspaceEntryResponse(
                success=False, key=request.key or "", error=str(e)[:500]
            )

    async def ReadWorkspace(
        self,
        request: tool_service_pb2.ReadWorkspaceRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ReadWorkspaceResponse:
        """Read one workspace entry by key, or list all keys if key is empty."""
        import json
        try:
            from services import agent_workspace_service
            user_id = request.user_id or "system"
            team_id = (request.team_id or "").strip()
            key = (request.key or "").strip()
            if not team_id:
                return tool_service_pb2.ReadWorkspaceResponse(
                    success=False, entries_json="[]", single=False, error="team_id is required"
                )
            if key:
                result = await agent_workspace_service.get_workspace_entry(
                    line_id=team_id, key=key, user_id=user_id
                )
                if not result.get("success"):
                    return tool_service_pb2.ReadWorkspaceResponse(
                        success=False, entries_json="{}", single=True, error=(result.get("error") or "Failed")[:500]
                    )
                return tool_service_pb2.ReadWorkspaceResponse(
                    success=True,
                    entries_json=json.dumps(result),
                    single=True,
                    error="",
                )
            result = await agent_workspace_service.list_workspace(line_id=team_id, user_id=user_id)
            if not result.get("success"):
                return tool_service_pb2.ReadWorkspaceResponse(
                    success=False, entries_json="[]", single=False, error=(result.get("error") or "Failed")[:500]
                )
            return tool_service_pb2.ReadWorkspaceResponse(
                success=True,
                entries_json=json.dumps(result.get("entries") or []),
                single=False,
                error="",
            )
        except Exception as e:
            logger.exception("ReadWorkspace failed")
            return tool_service_pb2.ReadWorkspaceResponse(
                success=False, entries_json="[]", single=False, error=str(e)[:500]
            )

    async def GetGoalAncestry(
        self,
        request: tool_service_pb2.GetGoalAncestryRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetGoalAncestryResponse:
        """Get goal ancestry from leaf to root for context injection."""
        try:
            from services import agent_goal_service
            user_id = request.user_id or "system"
            goal_id = (request.goal_id or "").strip()
            if not goal_id:
                return tool_service_pb2.GetGoalAncestryResponse(success=False, goals_json="[]", error="goal_id required")
            ancestry = await agent_goal_service.get_goal_ancestry(goal_id, user_id)
            return tool_service_pb2.GetGoalAncestryResponse(
                success=True,
                goals_json=json.dumps(ancestry),
                error="",
            )
        except Exception as e:
            logger.exception("GetGoalAncestry failed")
            return tool_service_pb2.GetGoalAncestryResponse(success=False, goals_json="[]", error=str(e)[:500])

    async def GetTeamGoalsTree(
        self,
        request: tool_service_pb2.GetTeamGoalsTreeRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetTeamGoalsTreeResponse:
        """Get full goal tree for a team."""
        try:
            from services import agent_goal_service
            user_id = request.user_id or "system"
            team_id = (request.team_id or "").strip()
            if not team_id:
                return tool_service_pb2.GetTeamGoalsTreeResponse(success=False, tree_json="[]", error="team_id required")
            tree = await agent_goal_service.get_goal_tree(team_id, user_id)
            return tool_service_pb2.GetTeamGoalsTreeResponse(
                success=True,
                tree_json=json.dumps(tree),
                error="",
            )
        except Exception as e:
            logger.exception("GetTeamGoalsTree failed")
            return tool_service_pb2.GetTeamGoalsTreeResponse(success=False, tree_json="[]", error=str(e)[:500])

    async def GetGoalsForAgent(
        self,
        request: tool_service_pb2.GetGoalsForAgentRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetGoalsForAgentResponse:
        """Return goals assigned to an agent in a team."""
        try:
            from services import agent_goal_service
            import json
            user_id = request.user_id or "system"
            team_id = (request.team_id or "").strip()
            agent_profile_id = (request.agent_profile_id or "").strip()
            if not team_id or not agent_profile_id:
                return tool_service_pb2.GetGoalsForAgentResponse(success=False, goals_json="[]", error="team_id and agent_profile_id required")
            goals = await agent_goal_service.get_goals_for_agent(agent_profile_id, team_id, user_id)
            return tool_service_pb2.GetGoalsForAgentResponse(success=True, goals_json=json.dumps(goals), error="")
        except Exception as e:
            logger.exception("GetGoalsForAgent failed")
            return tool_service_pb2.GetGoalsForAgentResponse(success=False, goals_json="[]", error=str(e)[:500])

    async def UpdateGoalProgress(
        self,
        request: tool_service_pb2.UpdateGoalProgressRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateGoalProgressResponse:
        """Update goal progress percentage."""
        try:
            from services import agent_goal_service
            user_id = request.user_id or "system"
            goal_id = (request.goal_id or "").strip()
            progress_pct = max(0, min(100, request.progress_pct))
            await agent_goal_service.update_progress(goal_id, user_id, progress_pct)
            return tool_service_pb2.UpdateGoalProgressResponse(success=True, error="")
        except Exception as e:
            logger.exception("UpdateGoalProgress failed")
            return tool_service_pb2.UpdateGoalProgressResponse(success=False, error=str(e)[:500])

    async def CreateAgentTask(
        self,
        request: tool_service_pb2.CreateAgentTaskRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateAgentTaskResponse:
        """Create a team task."""
        try:
            from services import agent_task_service
            import json
            user_id = request.user_id or "system"
            line_id = (request.team_id or "").strip()
            task = await agent_task_service.create_task(
                line_id=line_id,
                user_id=user_id,
                title=(request.title or "").strip() or "Untitled",
                description=(request.description or "").strip() or None,
                assigned_agent_id=(request.assigned_agent_id or "").strip() or None,
                goal_id=(request.goal_id or "").strip() or None,
                priority=request.priority or 0,
                created_by_agent_id=(request.created_by_agent_id or "").strip() or None,
                due_date=(request.due_date or "").strip() or None,
            )
            tid = task.get("id", "")
            logger.info(
                "CreateAgentTask: user=%s line=%s task=%s assigned=%s goal=%s title=%s",
                user_id,
                line_id,
                tid,
                (request.assigned_agent_id or "").strip() or None,
                (request.goal_id or "").strip() or None,
                ((request.title or "").strip() or "Untitled")[:120],
            )
            return tool_service_pb2.CreateAgentTaskResponse(
                success=True, task_id=tid, task_json=json.dumps(task), error=""
            )
        except Exception as e:
            logger.exception("CreateAgentTask failed")
            return tool_service_pb2.CreateAgentTaskResponse(success=False, task_id="", task_json="", error=str(e)[:500])

    async def GetAgentWorkQueue(
        self,
        request: tool_service_pb2.GetAgentWorkQueueRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetAgentWorkQueueResponse:
        """Get tasks assigned to an agent in a team."""
        try:
            from services import agent_task_service
            import json
            user_id = request.user_id or "system"
            team_id = (request.team_id or "").strip()
            agent_profile_id = (request.agent_profile_id or "").strip()
            if not team_id or not agent_profile_id:
                return tool_service_pb2.GetAgentWorkQueueResponse(success=False, tasks_json="[]", error="team_id and agent_profile_id required")
            tasks = await agent_task_service.get_agent_work_queue(agent_profile_id, team_id, user_id)
            return tool_service_pb2.GetAgentWorkQueueResponse(success=True, tasks_json=json.dumps(tasks), error="")
        except Exception as e:
            logger.exception("GetAgentWorkQueue failed")
            return tool_service_pb2.GetAgentWorkQueueResponse(success=False, tasks_json="[]", error=str(e)[:500])

    async def UpdateTaskStatus(
        self,
        request: tool_service_pb2.UpdateTaskStatusRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateTaskStatusResponse:
        """Transition task to a new status."""
        try:
            from services import agent_task_service
            import json
            user_id = request.user_id or "system"
            task_id = (request.task_id or "").strip()
            new_status = (request.new_status or "").strip()
            if not task_id or not new_status:
                return tool_service_pb2.UpdateTaskStatusResponse(success=False, task_json="", error="task_id and new_status required")
            task = await agent_task_service.transition_task(task_id, user_id, new_status)
            return tool_service_pb2.UpdateTaskStatusResponse(success=True, task_json=json.dumps(task), error="")
        except ValueError as e:
            err = str(e)[:500]
            logger.warning("UpdateTaskStatus rejected: %s", err)
            return tool_service_pb2.UpdateTaskStatusResponse(success=False, task_json="", error=err)
        except Exception as e:
            logger.exception("UpdateTaskStatus failed")
            return tool_service_pb2.UpdateTaskStatusResponse(success=False, task_json="", error=str(e)[:500])

    async def AssignTaskToAgent(
        self,
        request: tool_service_pb2.AssignTaskToAgentRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.AssignTaskToAgentResponse:
        """Assign a task to an agent."""
        try:
            from services import agent_task_service
            import json
            user_id = request.user_id or "system"
            task_id = (request.task_id or "").strip()
            agent_profile_id = (request.agent_profile_id or "").strip()
            if not task_id or not agent_profile_id:
                return tool_service_pb2.AssignTaskToAgentResponse(success=False, task_json="", error="task_id and agent_profile_id required")
            task = await agent_task_service.assign_task(task_id, agent_profile_id, user_id)
            return tool_service_pb2.AssignTaskToAgentResponse(success=True, task_json=json.dumps(task), error="")
        except Exception as e:
            logger.exception("AssignTaskToAgent failed")
            return tool_service_pb2.AssignTaskToAgentResponse(success=False, task_json="", error=str(e)[:500])

    async def GetUserNotificationPreferences(
        self,
        request: tool_service_pb2.GetUserNotificationPreferencesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetUserNotificationPreferencesResponse:
        """Get user notification preferences from users.preferences JSONB."""
        try:
            from services.database_manager.database_helpers import fetch_one

            user_id = request.user_id or "system"
            row = await fetch_one(
                "SELECT preferences FROM users WHERE user_id = $1",
                user_id,
            )
            prefs = {}
            if row:
                all_prefs = row.get("preferences") or {}
                if isinstance(all_prefs, str):
                    all_prefs = json.loads(all_prefs)
                prefs = all_prefs.get("notification_preferences", {})

            return tool_service_pb2.GetUserNotificationPreferencesResponse(
                success=True,
                preferences_json=json.dumps(prefs),
                error="",
            )
        except Exception as e:
            logger.error("GetUserNotificationPreferences failed: %s", e)
            return tool_service_pb2.GetUserNotificationPreferencesResponse(
                success=False, preferences_json="{}", error=str(e)
            )

    async def GetMyProfile(
        self,
        request: tool_service_pb2.GetMyProfileRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetMyProfileResponse:
        """Get the current user's profile (email, display_name, username) and key settings from users + user_settings."""
        try:
            from services.database_manager.database_helpers import fetch_one, fetch_all

            user_id = request.user_id or "system"
            email = ""
            display_name = ""
            username = ""
            preferred_name = ""
            timezone_val = ""
            zip_code = ""
            ai_context = ""

            row = await fetch_one(
                "SELECT email, display_name, username FROM users WHERE user_id = $1",
                user_id,
            )
            if row:
                email = row.get("email") or ""
                display_name = row.get("display_name") or ""
                username = row.get("username") or ""

            settings_rows = await fetch_all(
                "SELECT key, value FROM user_settings WHERE user_id = $1 AND key IN ('preferred_name', 'timezone', 'zip_code', 'ai_context')",
                user_id,
            )
            for s in settings_rows or []:
                k, v = s.get("key"), (s.get("value") or "")
                if k == "preferred_name":
                    preferred_name = v
                elif k == "timezone":
                    timezone_val = v
                elif k == "zip_code":
                    zip_code = v
                elif k == "ai_context":
                    ai_context = v

            return tool_service_pb2.GetMyProfileResponse(
                email=email,
                display_name=display_name,
                username=username,
                preferred_name=preferred_name,
                timezone=timezone_val,
                zip_code=zip_code,
                ai_context=ai_context,
                success=True,
                error="",
            )
        except Exception as e:
            logger.error("GetMyProfile failed: %s", e)
            return tool_service_pb2.GetMyProfileResponse(
                email="",
                display_name="",
                username="",
                preferred_name="",
                timezone="",
                zip_code="",
                ai_context="",
                success=False,
                error=str(e),
            )

    async def UpsertUserFact(
        self,
        request: tool_service_pb2.UpsertUserFactRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpsertUserFactResponse:
        """Insert or update a single user fact."""
        try:
            from services.settings_service import settings_service

            user_id = request.user_id or "system"
            fact_key = request.fact_key or ""
            value = request.value or ""
            category = request.category or "general"
            if not fact_key.strip():
                return tool_service_pb2.UpsertUserFactResponse(success=False, error="fact_key is required")
            write_enabled = await settings_service.get_facts_write_enabled(user_id)
            if not write_enabled:
                return tool_service_pb2.UpsertUserFactResponse(success=False, error="Fact writing disabled by user")
            source = getattr(request, "source", None) or "user_manual"
            result = await settings_service.upsert_user_fact(
                user_id, fact_key.strip(), value, category, source=source
            )
            if result.get("success"):
                return tool_service_pb2.UpsertUserFactResponse(success=True, error="")
            if result.get("status") == "pending_review":
                msg = (
                    "Fact '%s' is currently set to '%s' by the user. "
                    "Your proposed update has been queued for user review."
                ) % (result.get("fact_key", fact_key), result.get("current_value", ""))
                return tool_service_pb2.UpsertUserFactResponse(success=False, error=msg)
            return tool_service_pb2.UpsertUserFactResponse(
                success=False, error=result.get("error", "Upsert failed")
            )
        except Exception as e:
            logger.error("UpsertUserFact failed: %s", e)
            return tool_service_pb2.UpsertUserFactResponse(success=False, error=str(e))

    async def GetUserFacts(
        self,
        request: tool_service_pb2.GetUserFactsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetUserFactsResponse:
        """Return all facts for a user (for Agent Factory include_user_facts context)."""
        try:
            from services.settings_service import settings_service

            user_id = request.user_id or "system"
            facts = await settings_service.get_user_facts(user_id)
            facts_json = json.dumps(facts, default=_json_default)
            return tool_service_pb2.GetUserFactsResponse(success=True, facts_json=facts_json, error="")
        except Exception as e:
            logger.error("GetUserFacts failed: %s", e)
            return tool_service_pb2.GetUserFactsResponse(
                success=False, facts_json="[]", error=str(e)
            )

    async def InvokeDeviceTool(
        self,
        request: tool_service_pb2.InvokeDeviceToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.InvokeDeviceToolResponse:
        """Invoke a tool on a connected local proxy device via backend (device WebSockets live there)."""
        try:
            import os

            user_id = request.user_id or "system"
            device_id = request.device_id or ""
            tool = request.tool or ""
            args_json = request.args_json or "{}"
            timeout_seconds = request.timeout_seconds or 30
            logger.info(
                "InvokeDeviceTool: user_id=%s, device_id=%s, tool=%s",
                user_id,
                device_id or "(any)",
                tool,
            )
            try:
                args = json.loads(args_json)
            except json.JSONDecodeError:
                args = {}

            backend_url = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
            internal_key = os.getenv("INTERNAL_SERVICE_KEY", "")
            if not internal_key:
                logger.warning("InvokeDeviceTool: INTERNAL_SERVICE_KEY not set; backend may reject request")

            payload = {
                "user_id": user_id,
                "tool": tool,
                "args": args,
                "timeout_seconds": timeout_seconds,
            }
            if device_id:
                payload["device_id"] = device_id

            url = f"{backend_url}/api/internal/invoke-device-tool"
            headers = {"Content-Type": "application/json"}
            if internal_key:
                headers["X-Internal-Service-Key"] = internal_key

            http_timeout = max(35.0, float(timeout_seconds) + 5.0)

            import httpx
            async with httpx.AsyncClient(timeout=http_timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                err = resp.text or f"HTTP {resp.status_code}"
                logger.error("InvokeDeviceTool backend call failed: %s", err)
                return tool_service_pb2.InvokeDeviceToolResponse(
                    success=False,
                    result_json="{}",
                    error=err[:500],
                    formatted=err[:500],
                )
            data = resp.json()
            return tool_service_pb2.InvokeDeviceToolResponse(
                success=data.get("success", False),
                result_json=data.get("result_json", "{}"),
                error=data.get("error", ""),
                formatted=data.get("formatted", ""),
            )
        except Exception as e:
            logger.error("InvokeDeviceTool failed: %s", e)
            return tool_service_pb2.InvokeDeviceToolResponse(
                success=False,
                result_json="{}",
                error=str(e),
                formatted=str(e),
            )

    async def GetDeviceCapabilities(
        self,
        request: tool_service_pb2.GetDeviceCapabilitiesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetDeviceCapabilitiesResponse:
        """Return union of capabilities from all connected devices for the user."""
        try:
            import os

            user_id = request.user_id or "system"
            backend_url = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
            internal_key = os.getenv("INTERNAL_SERVICE_KEY", "")
            headers = {}
            if internal_key:
                headers["X-Internal-Service-Key"] = internal_key

            url = f"{backend_url}/api/internal/device-list"
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params={"user_id": user_id}, headers=headers)
            if resp.status_code != 200:
                logger.warning("GetDeviceCapabilities backend call failed: %s", resp.text)
                return tool_service_pb2.GetDeviceCapabilitiesResponse(
                    capabilities=[],
                    has_device=False,
                )
            data = resp.json()
            devices = data.get("devices") or []
            all_caps = set()
            for dev in devices:
                for cap in dev.get("capabilities") or []:
                    all_caps.add(cap)
            return tool_service_pb2.GetDeviceCapabilitiesResponse(
                capabilities=sorted(all_caps),
                has_device=len(devices) > 0,
            )
        except Exception as e:
            logger.error("GetDeviceCapabilities failed: %s", e)
            return tool_service_pb2.GetDeviceCapabilitiesResponse(
                capabilities=[],
                has_device=False,
            )


async def serve_tool_service(port: int = 50052):
    """
    Start the gRPC tool service server
    
    Runs alongside the main FastAPI server to provide data access
    for the LLM orchestrator service.
    """
    try:
        # Import health checking inside function (lesson learned!)
        from grpc_health.v1 import health, health_pb2, health_pb2_grpc
        
        logger.info(f"Starting gRPC Tool Service on port {port}...")
        
        # Create gRPC server with increased message size limits
        # Default is 4MB, increase to 100MB for large document search responses
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100 MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100 MB
        ]
        server = grpc.aio.server(options=options)
        
        # Register tool service
        tool_service = ToolServiceImplementation()
        tool_service_pb2_grpc.add_ToolServiceServicer_to_server(tool_service, server)
        
        # Register health checking
        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
        health_servicer.set(
            "tool_service.ToolService",
            health_pb2.HealthCheckResponse.SERVING
        )
        
        # Bind to port (use 0.0.0.0 for IPv4 compatibility)
        server.add_insecure_port(f'0.0.0.0:{port}')
        
        # Start server
        await server.start()
        logger.info(f"✅ gRPC Tool Service listening on port {port}")

        # Single sync authority: populate skills vector collection with retry until vector-service is ready.
        async def _sync_skills_with_retry():
            max_attempts = 5
            for attempt in range(max_attempts):
                delay = 2 ** (attempt + 1)  # 2, 4, 8, 16, 32 seconds
                if attempt > 0:
                    logger.info("Skills sync attempt %d/%d in %ds...", attempt + 1, max_attempts, delay)
                    await asyncio.sleep(delay)
                try:
                    from clients.vector_service_client import get_vector_service_client
                    client = await get_vector_service_client(required=False)
                    if not getattr(client, "_initialized", False):
                        await client.initialize(required=False)
                    if not getattr(client, "_initialized", False):
                        continue
                    await client.health_check()
                except Exception as e:
                    logger.debug("Vector service not ready: %s", e)
                    continue
                try:
                    from services.skill_vector_service import sync_all_skills
                    count = await sync_all_skills(user_id=None)
                    if count > 0:
                        logger.info("Skills vector collection populated with %d built-in skills", count)
                        return
                    else:
                        logger.info("Skills sync returned 0 skills (DB may not be seeded yet), retrying...")
                        continue
                except Exception as e:
                    logger.warning("Startup skills sync failed (attempt %d/%d): %s", attempt + 1, max_attempts, e)
            logger.error("Skills sync failed after %d attempts; vector-service may be unreachable", max_attempts)

        asyncio.create_task(_sync_skills_with_retry())

        # Wait for termination
        await server.wait_for_termination()
        
    except Exception as e:
        logger.error(f"❌ gRPC Tool Service failed to start: {e}")
        raise

