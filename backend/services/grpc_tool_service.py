"""
gRPC Tool Service - Backend Data Access for LLM Orchestrator
Provides document, RSS, entity, weather, and org-mode data via gRPC
"""

import logging
from typing import Optional, Dict, Any
import asyncio
import json

import grpc
from protos import tool_service_pb2, tool_service_pb2_grpc

# Import repositories and services directly (safe - no circular dependencies)
from repositories.document_repository import DocumentRepository
from services.direct_search_service import DirectSearchService
from services.embedding_service_wrapper import get_embedding_service

logger = logging.getLogger(__name__)


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
            
            # Parse filters for tags and categories
            tags = []
            categories = []
            for filter_str in request.filters:
                if filter_str.startswith("tag:"):
                    tags.append(filter_str[4:])
                elif filter_str.startswith("category:"):
                    categories.append(filter_str[9:])
            
            if tags or categories:
                logger.info(f"SearchDocuments: Filtering by tags={tags}, categories={categories}")
            
            # Get user's team IDs for hybrid search (user + team + global collections)
            team_ids = None
            user_id = request.user_id if request.user_id and request.user_id != "system" else None
            if user_id:
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
            
            # Get search service
            search_service = await self._get_search_service()
            
            # Perform direct search with optional tag/category filtering
            # Includes hybrid search across user, team, and global collections
            search_result = await search_service.search_documents(
                query=request.query,
                limit=request.limit or 10,
                similarity_threshold=0.3,  # Lowered from 0.7 for better recall
                user_id=user_id,
                team_ids=team_ids,
                tags=tags if tags else None,
                categories=categories if categories else None
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
                    page_count, chunk_count, entity_count, user_id, collection_type
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

            for doc in documents:
                doc_result = tool_service_pb2.DocumentResult(
                    document_id=str(doc.get('document_id', '')),
                    title=doc.get('title', doc.get('filename', '')),
                    filename=doc.get('filename', ''),
                    content_preview="",  # No preview for metadata-only search
                    relevance_score=1.0  # All matches are equally relevant
                )
                # Add metadata
                doc_result.metadata.update({
                    'tags': str(doc.get('tags', [])),
                    'category': doc.get('category', ''),
                    'user_id': doc.get('user_id', ''),
                    'collection_type': doc.get('collection_type', '')
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
                
                container = await get_service_container()
                folder_service = container.folder_service
                
                # Skip PDFs - they don't have text content
                if filename.lower().endswith('.pdf'):
                    logger.info(f"GetDocumentContent: Skipping PDF content for {request.document_id}")
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
                            with open(file_path, 'r', encoding='utf-8') as f:
                                full_content = f.read()
                            logger.info(f"GetDocumentContent: Loaded {len(full_content)} chars from {file_path}")
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
            for collection_name in collections_to_search:
                try:
                    # Check if collection exists
                    collections = vector_store.client.get_collections()
                    collection_names = [col.name for col in collections.collections]
                    
                    if collection_name not in collection_names:
                        continue
                    
                    # Scroll to get all points matching the filter
                    loop = asyncio.get_event_loop()
                    scroll_result = await loop.run_in_executor(
                        None,
                        lambda: vector_store.client.scroll(
                            collection_name=collection_name,
                            scroll_filter=document_filter,
                            limit=1000,  # Get up to 1000 chunks per document
                            with_payload=True,
                            with_vectors=False
                        )
                    )
                    
                    # Extract chunks from scroll result
                    points, next_page_offset = scroll_result
                    for point in points:
                        payload = point.payload or {}
                        chunk_data = {
                            'chunk_id': payload.get('chunk_id', str(point.id)),
                            'document_id': payload.get('document_id', request.document_id),
                            'content': payload.get('content', ''),
                            'chunk_index': payload.get('chunk_index', 0),
                            'metadata': payload.get('metadata', {})
                        }
                        all_chunks.append(chunk_data)
                    
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
        """Search RSS feeds and articles"""
        try:
            logger.info(f"SearchRSSFeeds: user={request.user_id}, query={request.query}")
            
            # Placeholder implementation - Phase 2 will wire up real RSS service
            response = tool_service_pb2.RSSSearchResponse()
            logger.info(f"SearchRSSFeeds: Returning placeholder response")
            return response
            
        except Exception as e:
            logger.error(f"SearchRSSFeeds error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"RSS search failed: {str(e)}")
    
    async def GetRSSArticles(
        self,
        request: tool_service_pb2.RSSArticlesRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.RSSArticlesResponse:
        """Get articles from RSS feed"""
        try:
            logger.info(f"GetRSSArticles: feed_id={request.feed_id}")
            
            # Placeholder implementation
            response = tool_service_pb2.RSSArticlesResponse()
            return response
            
        except Exception as e:
            logger.error(f"GetRSSArticles error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Get articles failed: {str(e)}")
    
    # ===== RSS Management Operations =====
    
    async def AddRSSFeed(
        self,
        request: tool_service_pb2.AddRSSFeedRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.AddRSSFeedResponse:
        """Add a new RSS feed"""
        try:
            logger.info(f"AddRSSFeed: user={request.user_id}, url={request.feed_url}, is_global={request.is_global}")
            
            from services.service_container import get_service_container
            from services.auth_service import get_auth_service
            from models.rss_models import RSSFeedCreate
            
            service_container = await get_service_container()
            rss_service = service_container.rss_service
            auth_service = await get_auth_service()
            
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
            
            from services.service_container import get_service_container
            from services.auth_service import get_auth_service
            
            service_container = await get_service_container()
            rss_service = service_container.rss_service
            auth_service = await get_auth_service()
            
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
            
            for feed in feeds:
                # Get article count for this feed
                from services.database_manager.database_helpers import fetch_value
                try:
                    article_count = await fetch_value(
                        "SELECT COUNT(*) FROM rss_articles WHERE feed_id = $1",
                        feed.feed_id
                    ) or 0
                except:
                    article_count = 0
                
                feed_details = tool_service_pb2.RSSFeedDetails(
                    feed_id=feed.feed_id,
                    feed_name=feed.feed_name,
                    feed_url=feed.feed_url,
                    category=feed.category or "general",
                    is_global=(feed.user_id is None),
                    last_polled=feed.last_poll_date.isoformat() if feed.last_poll_date else "",
                    article_count=int(article_count)
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
            
            from services.service_container import get_service_container
            from services.celery_tasks.rss_tasks import poll_rss_feeds_task
            
            service_container = await get_service_container()
            rss_service = service_container.rss_service
            
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
            
            from services.service_container import get_service_container
            
            service_container = await get_service_container()
            rss_service = service_container.rss_service
            
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
            await rss_service.delete_feed(target_feed.feed_id)
            
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
        """Get weather data"""
        try:
            logger.info(f"GetWeatherData: location={request.location}")
            
            # Placeholder implementation
            response = tool_service_pb2.WeatherResponse(
                location=request.location,
                current_conditions="Placeholder weather data"
            )
            return response
            
        except Exception as e:
            logger.error(f"GetWeatherData error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Weather data failed: {str(e)}")
    
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
            
            # Call image generation service
            result = await image_service.generate_images(
                prompt=request.prompt,
                size=request.size if request.size else "1024x1024",
                fmt=request.format if request.format else "png",
                seed=request.seed if request.HasField("seed") else None,
                num_images=request.num_images if request.num_images else 1,
                negative_prompt=request.negative_prompt if request.HasField("negative_prompt") else None
            )
            
            # Convert result to proto response
            if result.get("success"):
                images = []
                for img in result.get("images", []):
                    images.append(tool_service_pb2.GeneratedImage(
                        filename=img.get("filename", ""),
                        path=img.get("path", ""),
                        url=img.get("url", ""),
                        width=img.get("width", 1024),
                        height=img.get("height", 1024),
                        format=img.get("format", "png")
                    ))
                
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
            
            # Handle different kinds of entries
            if request.kind == "contact":
                # Build contact entry with PROPERTIES drawer
                headline = f"* {request.text}"
                org_entry = f"{headline}\n"
                
                if request.contact_properties:
                    org_entry += ":PROPERTIES:\n"
                    for key, value in request.contact_properties.items():
                        if value:
                            org_entry += f":{key}: {value}\n"
                    org_entry += ":END:\n"
                
                result = await org_inbox_append_text(org_entry, request.user_id)
                line_index = None  # Will determine after listing
                
            elif request.schedule or request.kind == "event":
                # Build a proper org-mode entry with schedule
                org_type = "TODO" if request.kind == "todo" else ""
                headline = f"* {org_type} {request.text}".strip()
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
                result = await org_inbox_add_item(text=request.text, kind=kind, user_id=request.user_id)
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
                message=f"Added '{request.text}' to inbox.org"
            )
            
        except Exception as e:
            logger.error(f"❌ AddOrgInboxItem error: {e}")
            return tool_service_pb2.AddOrgInboxItemResponse(
                success=False,
                error=str(e)
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
        """Crawl web content from URLs"""
        try:
            urls = list(request.urls) if request.urls else ([request.url] if request.url else [])
            logger.info(f"CrawlWebContent: {len(urls)} URLs")
            
            # Import crawl tool
            from services.langgraph_tools.crawl4ai_web_tools import crawl_web_content
            
            # Execute crawl
            result = await crawl_web_content(url=request.url if request.url else None, urls=list(request.urls) if request.urls else None)
            
            response = tool_service_pb2.WebCrawlResponse()
            
            # Parse result
            if isinstance(result, dict) and 'results' in result:
                for item in result['results']:
                    if not item.get('success'):
                        continue
                    
                    # Extract title from metadata if available
                    metadata = item.get('metadata', {})
                    title = metadata.get('title', '') if isinstance(metadata, dict) else ''
                    
                    # Extract content (full_content is the main content field)
                    content = item.get('full_content', '') or item.get('content', '')
                    
                    # Extract HTML from result
                    html = item.get('html', '')
                    
                    # Create crawl result
                    crawl_result = tool_service_pb2.WebCrawlResult(
                        url=item.get('url', ''),
                        title=title,
                        content=content,
                        html=html
                    )
                    
                    # Properly assign metadata map field using update()
                    if isinstance(metadata, dict):
                        for key, value in metadata.items():
                            crawl_result.metadata[str(key)] = str(value)
                    
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
                user_id=request.user_id
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
                        "confidence": op_proto.confidence if op_proto.HasField("confidence") else None
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
                        
                        await conn.execute(
                            """
                            UPDATE checkpoints
                            SET checkpoint = $1
                            WHERE thread_id = $2
                              AND checkpoint -> 'channel_values' ->> 'user_id' = $3
                            """,
                            checkpoint_data,
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
        
        # Wait for termination
        await server.wait_for_termination()
        
    except Exception as e:
        logger.error(f"❌ gRPC Tool Service failed to start: {e}")
        raise

