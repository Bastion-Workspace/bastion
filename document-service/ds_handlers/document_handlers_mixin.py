"""gRPC handlers for Document operations."""

import json
import logging

import grpc
from protos import tool_service_pb2

logger = logging.getLogger(__name__)


class DocumentHandlersMixin:
    """Mixin providing Document gRPC handlers.

    Mixed into ToolServiceImplementation; accesses self._get_search_service(),
    self._get_document_repo(), etc. via standard Python MRO.
    """

    # ===== Document Operations =====

    async def SearchDocuments(
        self,
        request: tool_service_pb2.SearchRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.SearchResponse:
        """Search documents by query using direct search with optional tag/category filtering"""
        try:
            logger.info(f"SearchDocuments: user={request.user_id}, query={request.query[:100]}")
            
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

            team_ids = None
            user_id = request.user_id if request.user_id and request.user_id != "system" else None
            if user_id and collection_scope != "global_docs":
                try:
                    from ds_handlers.team_lookup import list_user_teams as _list_teams

                    user_teams = await _list_teams(user_id)
                    team_ids = [team["team_id"] for team in user_teams] if user_teams else None
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

            search_service = await self._get_search_service()

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
            
            response = tool_service_pb2.SearchResponse(
                total_count=len(results)
            )
            
            for result in results:
                document_metadata = result.get('document', {})
                document_id = result.get('document_id') or document_metadata.get('document_id', '')

                doc_result = tool_service_pb2.DocumentResult(
                    document_id=str(document_id),
                    title=document_metadata.get('title', document_metadata.get('filename', '')),
                    filename=document_metadata.get('filename', ''),
                    content_preview=result.get('text', '')[:1500],
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

    async def RerankDocuments(
        self,
        request: tool_service_pb2.RerankRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.RerankResponse:
        """Rerank a list of text chunks by cross-encoder relevance to a query."""
        try:
            from ds_utils.rerank_client import call_rerank_api
            from ds_config import settings

            model = request.model or settings.RERANK_MODEL
            documents = list(request.documents)
            top_n = request.top_n or 10

            logger.info(
                "RerankDocuments: user=%s, query=%s, docs=%d, top_n=%d, model=%s",
                request.user_id,
                request.query[:80],
                len(documents),
                top_n,
                model,
            )

            results = await call_rerank_api(
                query=request.query,
                documents=documents,
                top_n=top_n,
                model=model,
            )

            response = tool_service_pb2.RerankResponse(success=True, model=model)
            for r in results:
                response.results.append(
                    tool_service_pb2.RerankResultItem(
                        index=r["index"],
                        relevance_score=r["relevance_score"],
                        document=r["document"],
                    )
                )
            logger.info("RerankDocuments: returned %d results", len(results))
            return response

        except Exception as e:
            logger.error("RerankDocuments error: %s", e)
            return tool_service_pb2.RerankResponse(success=False, error=str(e))

    async def FindDocumentsByTags(
        self,
        request: tool_service_pb2.FindDocumentsByTagsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.FindDocumentsByTagsResponse:
        """Find documents that contain ALL of the specified tags using database query"""
        try:
            logger.info(f"FindDocumentsByTags: user={request.user_id}, tags={list(request.required_tags)}, collection={request.collection_type}")
            logger.info(f"Request details: user_id={request.user_id}, required_tags={request.required_tags}, collection_type={request.collection_type}, limit={request.limit}")

            from ds_db.database_manager.database_helpers import fetch_all

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
                        from shims.services.service_container import get_service_container
                        container = await get_service_container()
                        folder_service = container.folder_service
                        _tid = doc.get("team_id")
                        if _tid and not isinstance(_tid, str):
                            _tid = str(_tid)
                        file_path_str = await folder_service.get_document_file_path(
                            filename=filename,
                            folder_id=doc.get('folder_id'),
                            user_id=doc.get('user_id'),
                            collection_type=doc.get('collection_type', 'user'),
                            team_id=_tid,
                            ensure_directory=False,
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
                    relevance_score=1.0
                )
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
            doc = await doc_repo.get_document_by_id(document_id=request.document_id, user_id=request.user_id)
            
            if not doc:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Document not found")
            
            filename = doc.get('filename')
            user_id = doc.get('user_id')
            folder_id = doc.get('folder_id')
            collection_type = doc.get('collection_type', 'user')
            team_id = doc.get('team_id')
            if team_id and not isinstance(team_id, str):
                team_id = str(team_id)
            
            logger.info(f"GetDocumentContent: filename={filename}, user_id={user_id}, folder_id={folder_id}, collection_type={collection_type}")
            
            full_content = None
            canonical_disk_path = ""
            
            if filename:
                from pathlib import Path
                from shims.services.service_container import get_service_container
                from ds_processing.document_processor import DocumentProcessor
                
                container = await get_service_container()
                folder_service = container.folder_service
                
                try:
                    logger.info(f"GetDocumentContent: Calling folder_service.get_document_file_path...")
                    file_path_str = await folder_service.get_document_file_path(
                        filename=filename,
                        folder_id=folder_id,
                        user_id=user_id,
                        collection_type=collection_type,
                        team_id=team_id,
                        ensure_directory=False,
                    )
                    logger.info(f"GetDocumentContent: Got file path: {file_path_str}")
                    file_path = Path(file_path_str)
                    try:
                        canonical_disk_path = str(file_path.resolve())
                    except Exception:
                        canonical_disk_path = str(file_path)
                except Exception as path_err:
                    logger.warning(f"GetDocumentContent: Could not resolve file path: {path_err}")
                    file_path = None
                
                image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg']
                is_image_file = any(filename.lower().endswith(ext) for ext in image_extensions)
                
                if is_image_file:
                    logger.info(f"GetDocumentContent: Skipping image file content for {request.document_id} ({filename})")
                    full_content = ""
                else:
                    try:
                        if not file_path:
                            raise FileNotFoundError("No file path resolved")
                        
                        if file_path.exists():
                            file_ext = file_path.suffix.lower()
                            
                            if file_ext in ['.txt', '.md', '.csv', '.json', '.yaml', '.yml', '.log']:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    full_content = f.read()
                                logger.info(f"GetDocumentContent: Loaded {len(full_content)} chars from plain text file {file_path}")
                            
                            elif file_ext in ['.docx', '.pptx', '.pdf', '.epub', '.html', '.htm', '.eml']:
                                logger.info(f"GetDocumentContent: Using DocumentProcessor for {file_ext} file")
                                doc_processor = DocumentProcessor()
                                
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
                                logger.warning(f"GetDocumentContent: Unknown file type {file_ext}, attempting plain text read")
                                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                    full_content = f.read()
                        else:
                            logger.warning(f"GetDocumentContent: File not found at {file_path}")
                    except Exception as e:
                        logger.error(f"GetDocumentContent: Failed to load from folder service: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
            
            if full_content is None:
                logger.error(f"GetDocumentContent: File not found for document {request.document_id} (filename={filename}, folder_id={folder_id})")
                await context.abort(grpc.StatusCode.NOT_FOUND, f"Document file not found on disk")
            
            logger.info(f"GetDocumentContent: Returning content with {len(full_content)} characters")
            
            response = tool_service_pb2.DocumentContentResponse(
                document_id=str(doc.get('document_id', '')),
                content=full_content or '',
                format='text',
            )
            if canonical_disk_path:
                response.canonical_path = canonical_disk_path
            
            return response
            
        except (grpc.RpcError, grpc._cython.cygrpc.AbortError):
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
            
            from ds_services.vector_store_service import get_vector_store
            vector_store = await get_vector_store()

            from ds_config import settings
            collections_to_search = [settings.VECTOR_COLLECTION_NAME]
            
            if request.user_id and request.user_id != "system":
                user_collection = f"user_{request.user_id}_documents"
                collections_to_search.append(user_collection)
            
            all_chunks = []
            
            dummy_vector = [0.0] * settings.EMBEDDING_DIMENSIONS
            
            for collection_name in collections_to_search:
                try:
                    await vector_store.ensure_collection_ready_for_search(collection_name)
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
                        limit=1000,
                        score_threshold=0.0,
                        filters=filters,
                    )
                    
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
            
            all_chunks.sort(key=lambda x: x.get('chunk_index', 0))
            
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
        """Find a document by filesystem path (true path resolution)."""
        try:
            from pathlib import Path
            from ds_config import settings
            
            logger.info(f"FindDocumentByPath: user={request.user_id}, path={request.file_path}, base={request.base_path}")
            
            file_path_str = request.file_path.strip()
            base_path_str = request.base_path.strip() if request.base_path else None
            
            if base_path_str and not Path(file_path_str).is_absolute():
                base_path = Path(base_path_str)
                resolved_path = (base_path / file_path_str).resolve()
            else:
                resolved_path = Path(file_path_str).resolve()
            
            if not resolved_path.suffix:
                resolved_path = resolved_path.with_suffix('.md')
            
            logger.info(f"FindDocumentByPath: Resolved to {resolved_path}")
            
            if not resolved_path.exists() or not resolved_path.is_file():
                logger.warning(f"FindDocumentByPath: File not found at {resolved_path}")
                return tool_service_pb2.FindDocumentByPathResponse(
                    success=False,
                    error=f"File not found at {resolved_path}"
                )
            
            from pathlib import Path as PathLib
            
            path = PathLib(resolved_path)
            filename = path.name
            
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
            
            doc_repo = self._get_document_repo()
            user_id = request.user_id
            collection_type = 'user'
            folder_id = None
            
            if uploads_idx + 1 < len(parts):
                collection_dir = parts[uploads_idx + 1]
                
                if collection_dir == 'Users' and uploads_idx + 2 < len(parts):
                    username = parts[uploads_idx + 2]
                    collection_type = 'user'
                    
                    if not user_id:
                        from ds_db.document_repository import DocumentRepository
                        temp_repo = DocumentRepository()
                        import asyncpg
                        from ds_config import settings
                        conn = await asyncpg.connect(settings.DATABASE_URL)
                        try:
                            row = await conn.fetchrow("SELECT user_id FROM users WHERE username = $1", username)
                            if row:
                                user_id = row['user_id']
                        finally:
                            await conn.close()
                    
                    folder_start_idx = uploads_idx + 3
                    folder_end_idx = len(parts) - 1
                    
                    if folder_start_idx < folder_end_idx:
                        folder_parts = parts[folder_start_idx:folder_end_idx]
                        logger.info(f"Resolving folder hierarchy: {folder_parts}")
                        folders_data = await doc_repo.get_folders_by_user(user_id, collection_type)
                        logger.info(f"Found {len(folders_data)} total folders for user")
                        folder_map = {(f.get('name'), f.get('parent_folder_id')): f.get('folder_id') for f in folders_data}
                        
                        parent_folder_id = None
                        for i, folder_name in enumerate(folder_parts):
                            key = (folder_name, parent_folder_id)
                            logger.info(f"Step {i+1}: Looking for folder '{folder_name}' with parent={parent_folder_id}")
                            if key in folder_map:
                                folder_id = folder_map[key]
                                parent_folder_id = folder_id
                                logger.info(f"Found folder_id={folder_id} for '{folder_name}'")
                            else:
                                logger.warning(f"Folder '{folder_name}' with parent={parent_folder_id} NOT FOUND in folder_map!")
                                logger.warning(f"   Available folders with parent={parent_folder_id}: {[k[0] for k in folder_map.keys() if k[1] == parent_folder_id]}")
                                folder_id = None
                                break
                
                elif collection_dir == 'Global':
                    collection_type = 'global'
                    user_id = None
                    
                    folder_start_idx = uploads_idx + 2
                    folder_end_idx = len(parts) - 1
                    
                    if folder_start_idx < folder_end_idx:
                        folder_parts = parts[folder_start_idx:folder_end_idx]
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
            
            logger.info(f"Searching for filename='{filename}', user_id={user_id}, collection_type={collection_type}, folder_id={folder_id}")
            document = await doc_repo.find_by_filename_and_context(
                filename=filename,
                user_id=user_id,
                collection_type=collection_type,
                folder_id=folder_id
            )
            
            if not document:
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
