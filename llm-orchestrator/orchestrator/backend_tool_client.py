"""
Backend Tool Client - gRPC client for accessing backend data services
"""

import logging
import os
import json
from typing import List, Dict, Any, Optional
import asyncio

import grpc
from protos import (
    document_service_pb2_grpc,
    tool_service_pb2,
    tool_service_pb2_grpc,
)

# For type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List as TypingList

logger = logging.getLogger(__name__)


class BackendToolClient:
    """
    gRPC client for backend tool service
    
    Provides async methods to access backend data services:
    - Document search and retrieval
    - RSS feed operations
    - Entity operations
    - Weather data
    - Org-mode operations
    """
    
    def __init__(self, host: str = None, port: int = None):
        """
        Initialize backend tool client
        
        Args:
            host: Backend service host (defaults to env BACKEND_TOOL_SERVICE_HOST)
            port: Backend service port (defaults to env BACKEND_TOOL_SERVICE_PORT)
        """
        self.host = host or os.getenv('BACKEND_TOOL_SERVICE_HOST', 'backend')
        self.port = port or int(os.getenv('BACKEND_TOOL_SERVICE_PORT', '50052'))
        self.address = f'{self.host}:{self.port}'
        self._document_service_host = os.getenv(
            'DOCUMENT_SERVICE_GRPC_HOST', 'document-service'
        )
        self._document_service_port = int(
            os.getenv('DOCUMENT_SERVICE_GRPC_PORT', '50058')
        )
        self._document_service_address = (
            f'{self._document_service_host}:{self._document_service_port}'
        )
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[tool_service_pb2_grpc.ToolServiceStub] = None
        self._doc_channel: Optional[grpc.aio.Channel] = None
        self._doc_stub: Optional[document_service_pb2_grpc.DocumentServiceStub] = None

        logger.info(
            "Backend Tool Client: tools=%s document-service=%s",
            self.address,
            self._document_service_address,
        )
    
    async def connect(self):
        """Establish gRPC channels to tools-service and document-service."""
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100 MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100 MB
        ]
        if self._channel is None:
            logger.debug(f"Connecting to backend tool service at {self.address}...")
            self._channel = grpc.aio.insecure_channel(self.address, options=options)
            self._stub = tool_service_pb2_grpc.ToolServiceStub(self._channel)
            logger.debug("Connected to backend tool service")
        if self._doc_channel is None:
            logger.debug(
                "Connecting to document-service at %s...", self._document_service_address
            )
            self._doc_channel = grpc.aio.insecure_channel(
                self._document_service_address, options=options
            )
            self._doc_stub = document_service_pb2_grpc.DocumentServiceStub(
                self._doc_channel
            )
            logger.debug("Connected to document-service")
    
    async def close(self):
        """Close gRPC channels."""
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None
            logger.debug("Disconnected from backend tool service")
        if self._doc_channel:
            await self._doc_channel.close()
            self._doc_channel = None
            self._doc_stub = None
            logger.debug("Disconnected from document-service")
    
    async def _ensure_connected(self):
        """Ensure tools-service and document-service channels exist."""
        if self._stub is None or self._doc_stub is None:
            await self.connect()
    
    # ===== Document Operations =====
    
    async def search_documents(
        self,
        query: str,
        user_id: str = "system",
        limit: int = 10,
        filters: List[str] = None,
        exclude_document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Search documents by query
        
        Args:
            query: Search query
            user_id: User ID for access control
            limit: Maximum number of results
            filters: Optional filters
            
        Returns:
            Dict with 'results' (list of documents) and 'total_count'
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.SearchRequest(
                user_id=user_id,
                query=query,
                limit=limit,
                filters=filters or [],
                exclude_document_ids=exclude_document_ids or [],
            )
            
            response = await self._doc_stub.SearchDocuments(request)
            
            # Convert proto response to dict
            results = []
            for doc in response.results:
                results.append({
                    'document_id': doc.document_id,
                    'title': doc.title,
                    'filename': doc.filename,
                    'content_preview': doc.content_preview,
                    'relevance_score': doc.relevance_score,
                    'metadata': dict(doc.metadata)
                })
            
            return {
                'results': results,
                'total_count': response.total_count
            }
            
        except grpc.RpcError as e:
            logger.error(f"Document search failed: {e.code()} - {e.details()}")
            return {'results': [], 'total_count': 0, 'error': str(e)}
        except Exception as e:
            logger.error(f"Unexpected error in document search: {e}")
            return {'results': [], 'total_count': 0, 'error': str(e)}

    async def rerank_documents(
        self,
        query: str,
        documents: List[str],
        top_n: int = 10,
        model: str = "cohere/rerank-4-pro",
        user_id: str = "system",
    ) -> Dict[str, Any]:
        """
        Rerank a list of text chunks by cross-encoder relevance to a query.

        Args:
            query: The search query or question.
            documents: List of text chunks to rerank.
            top_n: Maximum number of results to return.
            model: OpenRouter rerank model identifier.
            user_id: User ID for auth context.

        Returns:
            Dict with 'results' (list of {index, relevance_score, document}) and 'model'.
        """
        try:
            await self._ensure_connected()

            request = tool_service_pb2.RerankRequest(
                query=query,
                documents=documents,
                top_n=top_n,
                model=model,
                user_id=user_id,
            )

            response = await self._doc_stub.RerankDocuments(request)

            if not response.success:
                logger.warning(f"RerankDocuments returned error: {response.error}")
                return {'results': [], 'model': model, 'error': response.error}

            results = [
                {
                    'index': r.index,
                    'relevance_score': r.relevance_score,
                    'document': r.document,
                }
                for r in response.results
            ]

            return {'results': results, 'model': response.model}

        except grpc.RpcError as e:
            logger.error(f"RerankDocuments gRPC failed: {e.code()} - {e.details()}")
            return {'results': [], 'model': model, 'error': str(e)}
        except Exception as e:
            logger.error(f"Unexpected error in rerank_documents: {e}")
            return {'results': [], 'model': model, 'error': str(e)}

    async def search_help_docs(
        self,
        query: str,
        user_id: str = "system",
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Search app help documentation.

        Args:
            query: Natural language question about the app
            user_id: User ID (for auth; help docs are global)
            limit: Max number of results

        Returns:
            Dict with 'results' (list of {topic_id, title, content, score}) and 'total_count'
        """
        try:
            await self._ensure_connected()
            request = tool_service_pb2.SearchHelpDocsRequest(
                query=query,
                user_id=user_id,
                limit=limit,
            )
            response = await self._stub.SearchHelpDocs(request)
            results = []
            for r in response.results:
                results.append({
                    'topic_id': r.topic_id,
                    'title': r.title,
                    'content': r.content,
                    'score': r.score,
                })
            return {
                'results': results,
                'total_count': response.total_count,
            }
        except grpc.RpcError as e:
            logger.error("Help docs search failed: %s - %s", e.code(), e.details())
            return {'results': [], 'total_count': 0, 'error': str(e)}
        except Exception as e:
            logger.error("Unexpected error in help docs search: %s", e)
            return {'results': [], 'total_count': 0, 'error': str(e)}
    
    async def get_document(
        self,
        document_id: str,
        user_id: str = "system"
    ) -> Optional[Dict[str, Any]]:
        """
        Get document metadata
        
        Args:
            document_id: Document ID
            user_id: User ID for access control
            
        Returns:
            Document metadata dict or None
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.DocumentRequest(
                document_id=document_id,
                user_id=user_id
            )
            
            response = await self._doc_stub.GetDocument(request)
            
            return {
                'document_id': response.document_id,
                'title': response.title,
                'filename': response.filename,
                'content_type': response.content_type,
                'metadata': dict(response.metadata)
            }
            
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                logger.warning(f"Document not found: {document_id}")
                return None
            logger.error(f"Get document failed: {e.code()} - {e.details()}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting document: {e}")
            return None
    
    async def get_document_content(
        self,
        document_id: str,
        user_id: str = "system"
    ) -> Optional[str]:
        """
        Get full document content
        
        Args:
            document_id: Document ID
            user_id: User ID for access control
            
        Returns:
            Document content string or None
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.DocumentRequest(
                document_id=document_id,
                user_id=user_id
            )
            
            response = await self._doc_stub.GetDocumentContent(request)
            
            return response.content
            
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                logger.warning(f"Document content not found: {document_id}")
                return None
            logger.error(f"Get document content failed: {e.code()} - {e.details()}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting document content: {e}")
            return None

    async def get_document_links(
        self,
        document_id: str,
        user_id: str = "system",
        direction: str = "both",
        link_types: Optional[List[str]] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Resolved document link graph from document_links (outgoing / incoming / both).
        """
        try:
            await self._ensure_connected()
            req = tool_service_pb2.GetDocumentLinksRequest(
                document_id=document_id,
                user_id=user_id,
                direction=(direction or "both").strip().lower() or "both",
                limit=int(limit) if limit else 50,
            )
            lt = link_types or []
            for t in lt:
                if t and str(t).strip():
                    req.link_types.append(str(t).strip())

            response = await self._doc_stub.GetDocumentLinks(req)
            links = []
            for l in response.links:
                links.append(
                    {
                        "document_id": l.document_id,
                        "filename": l.filename,
                        "title": l.title,
                        "canonical_path": l.canonical_path,
                        "link_type": l.link_type,
                        "line_number": l.line_number,
                        "direction": l.direction,
                    }
                )
            return {"links": links, "total": int(response.total)}
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                logger.warning("GetDocumentLinks: document not found: %s", document_id)
                return {"links": [], "total": 0, "error": "not_found"}
            logger.error(
                "GetDocumentLinks failed: %s - %s", e.code(), e.details()
            )
            return {"links": [], "total": 0, "error": str(e)}
        except Exception as e:
            logger.error("Unexpected error in get_document_links: %s", e)
            return {"links": [], "total": 0, "error": str(e)}
    
    async def get_document_size(self, document_id: str, user_id: str = "system") -> int:
        """
        Get total character count of a document
        
        Args:
            document_id: Document ID
            user_id: User ID for access control
            
        Returns:
            Total character count, or 0 if not found or error
        """
        try:
            # Get full document content and calculate size
            content = await self.get_document_content(document_id, user_id)
            if content:
                return len(content)
            return 0
            
        except Exception as e:
            logger.error(f"Failed to get document size: {e}")
            return 0
    
    async def get_document_chunks(
        self,
        document_id: str,
        user_id: str = "system",
        limit: Optional[int] = 5
    ) -> List[Dict[str, Any]]:
        """
        Get multiple chunks from a specific document, sorted by chunk_index
        
        Args:
            document_id: Document ID to retrieve chunks from
            user_id: User ID for access control
            limit: Maximum number of chunks to return (None = all chunks)
            
        Returns:
            List of chunk dictionaries with content, chunk_index, etc.
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.DocumentRequest(
                document_id=document_id,
                user_id=user_id
            )
            
            response = await self._doc_stub.GetDocumentChunks(request)
            
            # Convert proto response to list of dicts
            chunks = []
            for chunk_proto in response.chunks:
                # Parse metadata JSON string
                metadata = {}
                if chunk_proto.metadata:
                    try:
                        metadata = json.loads(chunk_proto.metadata)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        metadata = {}
                
                chunks.append({
                    'chunk_id': chunk_proto.chunk_id,
                    'document_id': chunk_proto.document_id,
                    'content': chunk_proto.content,
                    'chunk_index': chunk_proto.chunk_index,
                    'metadata': metadata
                })
            
            # Apply limit if specified
            if limit is not None:
                chunks = chunks[:limit]
            
            logger.info(f"Retrieved {len(chunks)} chunks for document {document_id}")
            return chunks
            
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                logger.warning(f"Document chunks not found: {document_id}")
                return []
            logger.error(f"Get document chunks failed: {e.code()} - {e.details()}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting document chunks: {e}")
            return []
    
    async def find_document_by_path(
        self,
        file_path: str,
        user_id: str = "system",
        base_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find a document by filesystem path (true path resolution).
        
        Args:
            file_path: Relative or absolute filesystem path (e.g., "./component_list.md", "../file.md")
            user_id: User ID for access control
            base_path: Base directory for resolving relative paths (optional)
            
        Returns:
            Dict with document_id, filename, resolved_path, or None if not found
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.FindDocumentByPathRequest(
                user_id=user_id,
                file_path=file_path,
                base_path=base_path or ""
            )
            
            response = await self._doc_stub.FindDocumentByPath(request)
            
            if not response.success:
                logger.warning(f"Document not found by path: {file_path} - {response.error}")
                return None
            
            return {
                'document_id': response.document_id,
                'filename': response.filename,
                'resolved_path': response.resolved_path
            }
            
        except grpc.RpcError as e:
            logger.error(f"Find document by path failed: {e.code()} - {e.details()}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error finding document by path: {e}")
            return None

    async def find_documents_by_tags(
        self,
        required_tags: List[str],
        user_id: str = "system",
        collection_type: str = "",
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Find documents that contain ALL of the specified tags

        Args:
            required_tags: List of tags that ALL must be present
            user_id: User ID for access control
            collection_type: Collection type filter ("user", "global", or empty)
            limit: Maximum number of results

        Returns:
            List of document dictionaries with metadata
        """
        try:
            await self._ensure_connected()

            request = tool_service_pb2.FindDocumentsByTagsRequest(
                user_id=user_id,
                required_tags=required_tags,
                collection_type=collection_type,
                limit=limit
            )

            response = await self._doc_stub.FindDocumentsByTags(request)

            # Convert response to list of dicts
            documents = []
            for result in response.results:
                doc = {
                    'document_id': result.document_id,
                    'title': result.title,
                    'filename': result.filename,
                    'content_preview': result.content_preview,
                    'relevance_score': result.relevance_score,
                    'metadata': dict(result.metadata)
                }
                documents.append(doc)

            logger.info(f"Found {len(documents)} documents with tags {required_tags}")
            return documents

        except grpc.RpcError as e:
            logger.error(f"Find documents by tags failed: {e.code()} - {e.details()}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error finding documents by tags: {e}")
            return []
    
    # ===== File Creation Operations =====
    
    async def create_user_file(
        self,
        filename: str,
        content: str,
        user_id: str = "system",
        folder_id: Optional[str] = None,
        folder_path: Optional[str] = None,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        content_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """
        Create a file in the user's My Documents section

        Args:
            filename: Name of the file to create
            content: File content as string (ignored when content_bytes is set)
            user_id: User ID (required)
            folder_id: Optional folder ID to place file in
            folder_path: Optional folder path (e.g., "Projects/Electronics") - will create if needed
            title: Optional document title (defaults to filename)
            tags: Optional list of tags for the document
            category: Optional category for the document
            content_bytes: Optional binary content (PDF, images, etc.); when set, content is ignored
        """
        try:
            await self._ensure_connected()

            request = tool_service_pb2.CreateUserFileRequest(
                user_id=user_id,
                filename=filename,
                content=content,
                folder_id=folder_id if folder_id else "",
                folder_path=folder_path if folder_path else "",
                title=title if title else "",
                tags=tags if tags else [],
                category=category if category else ""
            )
            if content_bytes is not None:
                request.binary_content = content_bytes
            
            response = await self._doc_stub.CreateUserFile(request)
            
            return {
                "success": response.success,
                "document_id": response.document_id,
                "filename": response.filename,
                "folder_id": response.folder_id,
                "message": response.message,
                "error": response.error if hasattr(response, 'error') and response.error else None
            }
            
        except grpc.RpcError as e:
            logger.error(f"Create user file failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "error": f"{e.code()}: {e.details()}",
                "message": "Failed to create file"
            }
        except Exception as e:
            logger.error(f"Unexpected error creating user file: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to create file"
            }
    
    async def create_user_folder(
        self,
        folder_name: str,
        user_id: str = "system",
        parent_folder_id: Optional[str] = None,
        parent_folder_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a folder in the user's My Documents section
        
        Args:
            folder_name: Name of the folder to create
            user_id: User ID (required)
            parent_folder_id: Optional parent folder ID
            parent_folder_path: Optional parent folder path (e.g., "Projects") - will resolve to folder_id
        
        Returns:
            Dict with success, folder_id, folder_name, parent_folder_id, and message
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.CreateUserFolderRequest(
                user_id=user_id,
                folder_name=folder_name,
                parent_folder_id=parent_folder_id if parent_folder_id else "",
                parent_folder_path=parent_folder_path if parent_folder_path else ""
            )
            
            response = await self._doc_stub.CreateUserFolder(request)
            
            return {
                "success": response.success,
                "folder_id": response.folder_id,
                "folder_name": response.folder_name,
                "parent_folder_id": response.parent_folder_id if response.parent_folder_id else None,
                "message": response.message,
                "error": response.error if hasattr(response, 'error') and response.error else None
            }
            
        except grpc.RpcError as e:
            logger.error(f"Create user folder failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "error": f"{e.code()}: {e.details()}",
                "message": "Failed to create folder"
            }
        except Exception as e:
            logger.error(f"Unexpected error creating user folder: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to create folder"
            }

    async def get_folder_tree(self, user_id: str = "system") -> List[Dict[str, Any]]:
        """
        Get flat list of folders in the user's document tree.

        Args:
            user_id: User ID (required).

        Returns:
            List of dicts with folder_id, name, parent_folder_id, collection_type, document_count.
        """
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetFolderTreeRequest(user_id=user_id)
            response = await self._doc_stub.GetFolderTree(request)
            return [
                {
                    "folder_id": f.folder_id,
                    "name": f.name,
                    "parent_folder_id": f.parent_folder_id or None,
                    "collection_type": f.collection_type,
                    "document_count": f.document_count,
                }
                for f in response.folders
            ]
        except grpc.RpcError as e:
            logger.error(f"Get folder tree failed: {e.code()} - {e.details()}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting folder tree: {e}")
            return []

    async def list_folder_documents(
        self,
        folder_id: str,
        user_id: str = "system",
        limit: int = 500,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        List documents directly in a folder (same access as folder contents API).

        Returns:
            Dict with success, documents (list of dicts), total_count, error
        """
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListFolderDocumentsRequest(
                folder_id=folder_id or "",
                user_id=user_id or "",
                limit=int(limit) if limit else 500,
                offset=int(offset) if offset else 0,
            )
            response = await self._doc_stub.ListFolderDocuments(request)
            if response.error:
                return {
                    "success": False,
                    "documents": [],
                    "total_count": 0,
                    "error": response.error,
                }
            docs = []
            for d in response.documents:
                docs.append(
                    {
                        "document_id": d.document_id,
                        "filename": d.filename,
                        "title": d.title,
                        "collection_type": d.collection_type,
                    }
                )
            return {
                "success": True,
                "documents": docs,
                "total_count": int(response.total_count or len(docs)),
                "error": None,
            }
        except grpc.RpcError as e:
            logger.error("ListFolderDocuments failed: %s - %s", e.code(), e.details())
            return {
                "success": False,
                "documents": [],
                "total_count": 0,
                "error": str(e),
            }
        except Exception as e:
            logger.error("Unexpected error in list_folder_documents: %s", e)
            return {
                "success": False,
                "documents": [],
                "total_count": 0,
                "error": str(e),
            }

    async def pick_random_document_from_folder(
        self,
        folder_id: str,
        user_id: str = "system",
        file_extension: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Pick a random document from a folder. Optional file_extension filter (e.g. png, jpg).

        Returns:
            Dict with found, document_id, filename, title, folder_id, doc_type, message.
        """
        try:
            await self._ensure_connected()
            request = tool_service_pb2.PickRandomDocumentFromFolderRequest(
                folder_id=folder_id,
                user_id=user_id,
                file_extension=file_extension or "",
            )
            response = await self._doc_stub.PickRandomDocumentFromFolder(request)
            return {
                "found": response.found,
                "document_id": response.document_id or "",
                "filename": response.filename or "",
                "title": response.title or None,
                "folder_id": response.folder_id or None,
                "doc_type": response.doc_type or None,
                "message": response.message or "",
            }
        except grpc.RpcError as e:
            logger.error(f"Pick random document failed: {e.code()} - {e.details()}")
            return {
                "found": False,
                "document_id": "",
                "filename": "",
                "title": None,
                "folder_id": None,
                "doc_type": None,
                "message": f"Pick random document failed: {e.details()}",
            }
        except Exception as e:
            logger.error(f"Unexpected error picking random document: {e}")
            return {
                "found": False,
                "document_id": "",
                "filename": "",
                "title": None,
                "folder_id": None,
                "doc_type": None,
                "message": f"Unexpected error: {e}",
            }

    # ===== Document Editing Operations =====
    
    async def update_document_metadata(
        self,
        document_id: str,
        user_id: str = "system",
        title: Optional[str] = None,
        frontmatter_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update document title and/or frontmatter type
        
        Args:
            document_id: Document ID to update
            user_id: User ID (required - must match document owner)
            title: Optional new title
            frontmatter_type: Optional frontmatter type (e.g., "electronics", "fiction")
        
        Returns:
            Dict with success, document_id, updated_fields, and message
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.UpdateDocumentMetadataRequest(
                user_id=user_id,
                document_id=document_id,
                title=title if title else "",
                frontmatter_type=frontmatter_type if frontmatter_type else ""
            )
            
            response = await self._doc_stub.UpdateDocumentMetadata(request)
            
            return {
                "success": response.success,
                "document_id": response.document_id,
                "updated_fields": list(response.updated_fields),
                "message": response.message,
                "error": response.error if hasattr(response, 'error') and response.error else None
            }
            
        except grpc.RpcError as e:
            logger.error(f"Update document metadata failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "error": f"{e.code()}: {e.details()}",
                "message": "Failed to update document metadata"
            }
        except Exception as e:
            logger.error(f"Unexpected error updating document metadata: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to update document metadata"
            }
    
    async def update_document_content(
        self,
        document_id: str,
        content: str,
        user_id: str = "system",
        append: bool = False
    ) -> Dict[str, Any]:
        """
        Update document content (append or replace)
        
        Args:
            document_id: Document ID to update
            content: New content to add (if append=True) or replace entire content (if append=False)
            user_id: User ID (required - must match document owner)
            append: If True, append content to existing; if False, replace entire content
        
        Returns:
            Dict with success, document_id, content_length, and message
        """
        try:
            await self._ensure_connected()
            
            # Omit write_initiator: document-service defaults to agent_tool (WS content_source=agent).
            request = tool_service_pb2.UpdateDocumentContentRequest(
                user_id=user_id,
                document_id=document_id,
                content=content,
                append=append,
            )
            
            response = await self._doc_stub.UpdateDocumentContent(request)
            
            return {
                "success": response.success,
                "document_id": response.document_id,
                "content_length": response.content_length,
                "message": response.message,
                "error": response.error if hasattr(response, 'error') and response.error else None
            }
            
        except grpc.RpcError as e:
            logger.error(f"Update document content failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "error": f"{e.code()}: {e.details()}",
                "message": "Failed to update document content"
            }
        except Exception as e:
            logger.error(f"Unexpected error updating document content: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to update document content"
            }
    
    async def propose_document_edit(
        self,
        document_id: str,
        edit_type: str,
        operations: Optional[List[Dict[str, Any]]] = None,
        content_edit: Optional[Dict[str, Any]] = None,
        agent_name: str = "unknown",
        summary: str = "",
        requires_preview: bool = True,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Propose a document edit for user review
        
        Args:
            document_id: Document ID to edit
            edit_type: "operations" or "content"
            operations: List of EditorOperation dicts (for operation-based edits)
            content_edit: ContentEdit dict (for content-based edits)
            agent_name: Name of proposing agent
            summary: Human-readable summary of proposed changes
            requires_preview: If False and edit is small, frontend may auto-apply
            user_id: User ID (required - must match document owner)
        
        Returns:
            Dict with success, proposal_id, document_id, and message
        """
        try:
            await self._ensure_connected()
            
            # Convert operations to proto (semantic ops: no start/end; backend resolves JIT)
            operations_proto = []
            if operations:
                for op in operations:
                    op_proto = tool_service_pb2.EditorOperationProto(
                        op_type=op.get("op_type", ""),
                        start=op.get("start", 0),
                        end=op.get("end", 0),
                        text=op.get("text", ""),
                        pre_hash=op.get("pre_hash", ""),
                        original_text=op.get("original_text") or "",
                        anchor_text=op.get("anchor_text") or "",
                        left_context=op.get("left_context") or "",
                        right_context=op.get("right_context") or "",
                        occurrence_index=op.get("occurrence_index", 0),
                        note=op.get("note") or "",
                        confidence=op.get("confidence") or 0.0,
                        search_text=op.get("search_text") or "",
                    )
                    operations_proto.append(op_proto)
            
            # Convert content_edit to proto
            content_edit_proto = None
            if content_edit:
                content_edit_proto = tool_service_pb2.ContentEditProto(
                    edit_mode=content_edit.get("edit_mode", "append"),
                    content=content_edit.get("content", ""),
                    insert_position=content_edit.get("insert_position") or 0,
                    note=content_edit.get("note") or ""
                )
            
            request = tool_service_pb2.ProposeDocumentEditRequest(
                user_id=user_id,
                document_id=document_id,
                edit_type=edit_type,
                operations=operations_proto,
                content_edit=content_edit_proto,
                agent_name=agent_name,
                summary=summary,
                requires_preview=requires_preview
            )
            
            response = await self._doc_stub.ProposeDocumentEdit(request)
            
            return {
                "success": response.success,
                "proposal_id": response.proposal_id,
                "document_id": response.document_id,
                "message": response.message,
                "error": response.error if hasattr(response, 'error') and response.error else None
            }
            
        except grpc.RpcError as e:
            logger.error(f"Propose document edit failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "error": f"{e.code()}: {e.details()}",
                "message": "Failed to propose document edit"
            }
        except Exception as e:
            logger.error(f"Unexpected error proposing document edit: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to propose document edit"
            }
    
    async def apply_operations_directly(
        self,
        document_id: str,
        operations: List[Dict[str, Any]],
        user_id: str = "system",
        agent_name: str = "unknown",
        playbook_auto_apply: bool = False,
    ) -> Dict[str, Any]:
        """
        Apply operations directly to a document (for authorized agents only)
        
        Args:
            document_id: Document ID to edit
            operations: List of EditorOperation dicts to apply
            user_id: User ID (required - must match document owner)
            agent_name: Name of agent requesting this operation (for security check)
            playbook_auto_apply: When True, document-service resolves semantic ops and skips allowlist.
        
        Returns:
            Dict with success, document_id, applied_count, and message
        """
        try:
            await self._ensure_connected()
            
            # Convert operations to proto
            operations_proto = []
            for op in operations:
                op_proto = tool_service_pb2.EditorOperationProto(
                    op_type=op.get("op_type", "replace_range"),
                    start=op.get("start", 0),
                    end=op.get("end", op.get("start", 0)),
                    text=op.get("text", ""),
                    pre_hash=op.get("pre_hash", "")
                )
                if op.get("original_text"):
                    op_proto.original_text = op["original_text"]
                if op.get("anchor_text"):
                    op_proto.anchor_text = op["anchor_text"]
                if op.get("left_context"):
                    op_proto.left_context = op["left_context"]
                if op.get("right_context"):
                    op_proto.right_context = op["right_context"]
                if op.get("occurrence_index") is not None:
                    op_proto.occurrence_index = op["occurrence_index"]
                if op.get("note"):
                    op_proto.note = op["note"]
                if op.get("confidence") is not None:
                    op_proto.confidence = op["confidence"]
                operations_proto.append(op_proto)
            
            request = tool_service_pb2.ApplyOperationsDirectlyRequest(
                user_id=user_id,
                document_id=document_id,
                operations=operations_proto,
                agent_name=agent_name,
                playbook_auto_apply=playbook_auto_apply,
            )
            
            response = await self._doc_stub.ApplyOperationsDirectly(request)
            
            return {
                "success": response.success,
                "document_id": response.document_id,
                "applied_count": response.applied_count,
                "message": response.message,
                "error": response.error if hasattr(response, 'error') and response.error else None
            }
            
        except grpc.RpcError as e:
            logger.error(f"Apply operations directly failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "error": f"{e.code()}: {e.details()}",
                "message": "Failed to apply operations directly"
            }
        except Exception as e:
            logger.error(f"Unexpected error applying operations directly: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to apply operations directly"
            }
    
    async def apply_document_edit_proposal(
        self,
        proposal_id: str,
        selected_operation_indices: Optional[List[int]] = None,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Apply an approved document edit proposal
        
        Args:
            proposal_id: ID of proposal to apply
            selected_operation_indices: Which operations to apply (None = all, only for operation-based edits)
            user_id: User ID (required - must match proposal owner)
        
        Returns:
            Dict with success, document_id, applied_count, and message
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.ApplyDocumentEditProposalRequest(
                user_id=user_id,
                proposal_id=proposal_id,
                selected_operation_indices=selected_operation_indices or []
            )
            
            response = await self._doc_stub.ApplyDocumentEditProposal(request)
            
            return {
                "success": response.success,
                "document_id": response.document_id,
                "applied_count": response.applied_count,
                "message": response.message,
                "error": response.error if hasattr(response, 'error') and response.error else None
            }
            
        except grpc.RpcError as e:
            logger.error(f"Apply document edit proposal failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "error": f"{e.code()}: {e.details()}",
                "message": "Failed to apply document edit proposal"
            }
        except Exception as e:
            logger.error(f"Unexpected error applying document edit proposal: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to apply document edit proposal"
            }

    async def list_document_proposals(
        self,
        document_id: str,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """List pending document edit proposals for a document."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListDocumentProposalsRequest(
                user_id=user_id,
                document_id=document_id
            )
            response = await self._doc_stub.ListDocumentProposals(request)
            if not response.success:
                return {
                    "success": False,
                    "proposals": [],
                    "error": response.error or "Unknown error"
                }
            proposals = [
                {
                    "proposal_id": p.proposal_id,
                    "document_id": p.document_id,
                    "edit_type": p.edit_type,
                    "agent_name": p.agent_name,
                    "summary": p.summary,
                    "operations_count": p.operations_count,
                    "created_at": p.created_at,
                    "expires_at": p.expires_at or None
                }
                for p in response.proposals
            ]
            return {"success": True, "proposals": proposals}
        except grpc.RpcError as e:
            logger.error(f"List document proposals failed: {e.code()} - {e.details()}")
            return {"success": False, "proposals": [], "error": f"{e.code()}: {e.details()}"}
        except Exception as e:
            logger.error(f"Unexpected error listing document proposals: {e}")
            return {"success": False, "proposals": [], "error": str(e)}

    async def get_document_edit_proposal(
        self,
        proposal_id: str,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """Get full details of a document edit proposal."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetDocumentEditProposalRequest(
                user_id=user_id,
                proposal_id=proposal_id
            )
            response = await self._doc_stub.GetDocumentEditProposal(request)
            if not response.success:
                return {
                    "success": False,
                    "error": response.error or "Proposal not found"
                }
            import json
            operations = json.loads(response.operations_json) if response.operations_json else []
            content_edit = json.loads(response.content_edit_json) if response.content_edit_json else None
            return {
                "success": True,
                "proposal_id": response.proposal_id,
                "document_id": response.document_id,
                "edit_type": response.edit_type,
                "operations": operations,
                "content_edit": content_edit,
                "agent_name": response.agent_name,
                "summary": response.summary,
                "created_at": response.created_at
            }
        except grpc.RpcError as e:
            logger.error(f"Get document edit proposal failed: {e.code()} - {e.details()}")
            return {"success": False, "error": f"{e.code()}: {e.details()}"}
        except Exception as e:
            logger.error(f"Unexpected error getting document edit proposal: {e}")
            return {"success": False, "error": str(e)}

    async def reject_document_edit_proposal(
        self,
        proposal_id: str,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """Reject (delete) a document edit proposal."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.RejectDocumentEditProposalRequest(
                user_id=user_id,
                proposal_id=proposal_id
            )
            response = await self._doc_stub.RejectDocumentEditProposal(request)
            return {
                "success": response.success,
                "error": response.error if response.error else None
            }
        except grpc.RpcError as e:
            logger.error(f"Reject document edit proposal failed: {e.code()} - {e.details()}")
            return {"success": False, "error": f"{e.code()}: {e.details()}"}
        except Exception as e:
            logger.error(f"Unexpected error rejecting document edit proposal: {e}")
            return {"success": False, "error": str(e)}

    # ===== Weather Operations =====
    
    async def get_weather(
        self,
        location: str,
        user_id: str = "system",
        data_types: List[str] = None,
        date_str: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get weather data for location
        
        Args:
            location: Location name
            user_id: User ID
            data_types: Types of data to retrieve (e.g., ["current", "forecast", "history"])
            date_str: Optional date string for historical data (YYYY-MM-DD or YYYY-MM)
            
        Returns:
            Weather data dict or None
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.WeatherRequest(
                location=location,
                user_id=user_id,
                data_types=data_types or ["current"]
            )
            
            # Add date_str if provided (for historical requests)
            if date_str:
                request.date_str = date_str
            
            response = await self._stub.GetWeatherData(request)
            
            return {
                'location': response.location,
                'current_conditions': response.current_conditions,
                'forecast': list(response.forecast),
                'alerts': list(response.alerts),
                'metadata': dict(response.metadata),
                'success': True
            }
            
        except grpc.RpcError as e:
            logger.error(f"Get weather failed: {e.code()} - {e.details()}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting weather: {e}")
            return None
    
    # ===== Image Generation Operations =====
    
    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        format: str = "png",
        seed: Optional[int] = None,
        num_images: int = 1,
        negative_prompt: Optional[str] = None,
        user_id: str = "system",
        model: Optional[str] = None,
        reference_image_data: Optional[bytes] = None,
        reference_image_url: Optional[str] = None,
        reference_strength: float = 0.5,
        folder_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate images using backend image generation service
        
        Args:
            prompt: Image generation prompt
            size: Image size (e.g., "1024x1024")
            format: Image format (png, jpg, etc.)
            seed: Optional random seed for reproducibility
            num_images: Number of images to generate (1-4)
            negative_prompt: Optional negative prompt
            user_id: User ID
            model: Optional model override (user's preferred image generation model)
            
        Returns:
            Dict with success, images, and metadata
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.ImageGenerationRequest(
                prompt=prompt,
                size=size,
                format=format,
                num_images=num_images,
                user_id=user_id
            )
            
            # Add optional fields
            if seed is not None:
                request.seed = seed
            if negative_prompt is not None:
                request.negative_prompt = negative_prompt
            if model is not None:
                request.model = model
            if folder_id:
                request.folder_id = folder_id
            
            # Add reference image fields
            if reference_image_data:
                request.reference_image_data = reference_image_data
                logger.info("📎 Added reference_image_data to request")
            elif reference_image_url:
                request.reference_image_url = reference_image_url
                logger.info(f"📎 Added reference_image_url to request: {reference_image_url[:100]}")
            
            if reference_strength != 0.5:
                request.reference_strength = reference_strength
            
            logger.info(f"Calling backend GenerateImage: prompt={prompt[:100]}...")
            
            response = await self._stub.GenerateImage(request)
            
            # Convert proto response to dict
            result = {
                "success": response.success,
                "model": response.model,
                "prompt": response.prompt,
                "size": response.size,
                "format": response.format,
                "images": []
            }
            
            if response.success:
                for img in response.images:
                    item = {
                        "filename": img.filename,
                        "path": img.path,
                        "url": img.url,
                        "width": img.width,
                        "height": img.height,
                        "format": img.format
                    }
                    if getattr(img, "document_id", None):
                        item["document_id"] = img.document_id
                    result["images"].append(item)
                logger.info(f"Generated {len(result['images'])} image(s)")
            else:
                result["error"] = response.error
                logger.error(f"Image generation failed: {response.error}")
            
            return result
            
        except grpc.RpcError as e:
            logger.error(f"GenerateImage failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "error": f"gRPC error: {e.details()}",
                "images": []
            }
        except Exception as e:
            logger.error(f"Unexpected error in image generation: {e}")
            return {
                "success": False,
                "error": f"gRPC call failed: {str(e)}",
                "images": []
            }

    async def get_reference_image_for_object(
        self,
        object_name: str,
        user_id: str = "system"
    ) -> Optional[bytes]:
        """
        Get reference image bytes for a named object from existing images
        (detected/annotated). Used by image generation to supply a reference
        (e.g. Farmall Tractor) for accuracy.
        """
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetReferenceImageForObjectRequest(
                object_name=object_name.strip(),
                user_id=user_id
            )
            response = await self._stub.GetReferenceImageForObject(request)
            if response.success and response.HasField("reference_image_data") and response.reference_image_data:
                return response.reference_image_data
            return None
        except Exception as e:
            logger.debug("get_reference_image_for_object failed: %s", e)
            return None

    # ===== Entity Operations =====
    
    async def search_entities(
        self,
        query: str,
        user_id: str = "system",
        entity_types: List[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search entities
        
        Args:
            query: Search query
            user_id: User ID
            entity_types: Types of entities to search
            limit: Maximum results
            
        Returns:
            List of entity dicts
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.EntitySearchRequest(
                user_id=user_id,
                query=query,
                entity_types=entity_types or [],
                limit=limit
            )
            
            response = await self._stub.SearchEntities(request)
            
            entities = []
            for entity in response.entities:
                entities.append({
                    'entity_id': entity.entity_id,
                    'entity_type': entity.entity_type,
                    'name': entity.name,
                    'properties': dict(entity.properties)
                })
            
            return entities
            
        except grpc.RpcError as e:
            logger.error(f"Entity search failed: {e.code()} - {e.details()}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error searching entities: {e}")
            return []
    
    async def get_entity(
        self,
        entity_id: str,
        user_id: str = "system"
    ) -> Optional[Dict[str, Any]]:
        """
        Get entity details
        
        Args:
            entity_id: Entity ID
            user_id: User ID
            
        Returns:
            Entity details dict or None
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.EntityRequest(
                entity_id=entity_id,
                user_id=user_id
            )
            
            response = await self._stub.GetEntity(request)
            
            return {
                'entity': {
                    'entity_id': response.entity.entity_id,
                    'entity_type': response.entity.entity_type,
                    'name': response.entity.name,
                    'properties': dict(response.entity.properties)
                },
                'related_documents': list(response.related_documents)
            }
            
        except grpc.RpcError as e:
            logger.error(f"Get entity failed: {e.code()} - {e.details()}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting entity: {e}")
            return None
    
    async def find_documents_by_entities(
        self,
        entity_names: List[str],
        user_id: str = "system"
    ) -> List[str]:
        """
        Find documents mentioning specific entities
        
        Args:
            entity_names: List of entity names to search for
            user_id: User ID for RLS filtering
            
        Returns:
            List of document IDs (RLS filtered)
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.FindDocumentsByEntitiesRequest(
                user_id=user_id,
                entity_names=entity_names
            )
            
            response = await self._doc_stub.FindDocumentsByEntities(request)
            
            logger.info(f"Found {len(response.document_ids)} documents for entities (RLS filtered)")
            return list(response.document_ids)
            
        except grpc.RpcError as e:
            logger.error(f"FindDocumentsByEntities failed: {e.code()} - {e.details()}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error finding documents by entities: {e}")
            return []

    async def find_related_documents_by_entities(
        self,
        entity_names: List[str],
        max_hops: int = 2,
        user_id: str = "system"
    ) -> List[str]:
        """
        Find documents via entity relationship traversal
        
        Args:
            entity_names: Starting entity names
            max_hops: Maximum graph traversal depth (1-2)
            user_id: User ID for RLS filtering
            
        Returns:
            List of document IDs (RLS filtered)
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.FindRelatedDocumentsByEntitiesRequest(
                user_id=user_id,
                entity_names=entity_names,
                max_hops=max_hops
            )
            
            response = await self._doc_stub.FindRelatedDocumentsByEntities(request)
            
            logger.info(f"Found {len(response.document_ids)} related documents (RLS filtered)")
            return list(response.document_ids)
            
        except grpc.RpcError as e:
            logger.error(f"FindRelatedDocumentsByEntities failed: {e.code()} - {e.details()}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error finding related documents: {e}")
            return []

    async def find_co_occurring_entities(
        self,
        entity_names: List[str],
        min_co_occurrences: int = 2,
        user_id: str = "system"
    ) -> List[Dict[str, Any]]:
        """
        Find entities that co-occur with given entities
        
        Args:
            entity_names: Target entity names
            min_co_occurrences: Minimum co-occurrence threshold
            user_id: User ID (for logging/future use)
            
        Returns:
            List of entity dicts with name, type, co_occurrence_count
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.FindCoOccurringEntitiesRequest(
                user_id=user_id,
                entity_names=entity_names,
                min_co_occurrences=min_co_occurrences
            )
            
            response = await self._stub.FindCoOccurringEntities(request)
            
            entities = []
            for entity in response.entities:
                entities.append({
                    "name": entity.name,
                    "type": entity.type,
                    "co_occurrence_count": entity.co_occurrence_count
                })
            
            logger.info(f"Found {len(entities)} co-occurring entities")
            return entities
            
        except grpc.RpcError as e:
            logger.error(f"FindCoOccurringEntities failed: {e.code()} - {e.details()}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error finding co-occurring entities: {e}")
            return []
    
    # ===== Web Operations =====
    
    async def search_web(
        self,
        query: str,
        max_results: int = 15,
        user_id: str = "system"
    ) -> List[Dict[str, Any]]:
        """
        Search the web
        
        Args:
            query: Search query
            max_results: Maximum number of results
            user_id: User ID
            
        Returns:
            List of web search results
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.WebSearchRequest(
                query=query,
                max_results=max_results,
                user_id=user_id
            )
            
            response = await self._stub.SearchWeb(request)
            
            results = []
            for result in response.results:
                results.append({
                    'title': result.title,
                    'url': result.url,
                    'snippet': result.snippet,
                    'content': result.snippet  # WebSearchResult doesn't have content field, use snippet
                })
            
            return results
            
        except grpc.RpcError as e:
            logger.error(f"Web search failed: {e.code()} - {e.details()}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in web search: {e}")
            return []
    
    async def crawl_web_content(
        self,
        url: str = None,
        urls: List[str] = None,
        user_id: str = "system",
        paginate: bool = False,
        max_pages: int = 10,
        pagination_param: str = None,
        start_page: int = 0,
        next_page_css_selector: str = None,
        css_selector: str = None
    ) -> List[Dict[str, Any]]:
        """
        Crawl web content from URLs, with optional pagination.

        Args:
            url: Single URL to crawl
            urls: Multiple URLs to crawl
            user_id: User ID
            paginate: If True, follow pagination across multiple pages
            max_pages: Max pages when paginating (default 10)
            pagination_param: URL query param for page number (e.g. "page")
            start_page: Starting page number (default 0)
            next_page_css_selector: CSS selector for next-page link
            css_selector: CSS selector for content extraction

        Returns:
            List of crawled content (each with url, title, content, html, metadata, images, links)
        """
        try:
            await self._ensure_connected()

            request_kwargs = {
                "url": url or "",
                "urls": urls or [],
                "user_id": user_id,
            }
            if paginate:
                request_kwargs["paginate"] = True
                request_kwargs["max_pages"] = max_pages
                if pagination_param is not None:
                    request_kwargs["pagination_param"] = pagination_param
                request_kwargs["start_page"] = start_page
                if next_page_css_selector is not None:
                    request_kwargs["next_page_css_selector"] = next_page_css_selector
            if css_selector is not None:
                request_kwargs["css_selector"] = css_selector

            request = tool_service_pb2.WebCrawlRequest(**request_kwargs)
            response = await self._stub.CrawlWebContent(request)

            results = []
            for result in response.results:
                results.append({
                    "url": result.url,
                    "title": result.title,
                    "content": result.content,
                    "html": result.html,
                    "metadata": dict(result.metadata),
                    "images": list(result.images),
                    "links": list(result.links),
                })
            return results

        except grpc.RpcError as e:
            logger.error(f"Web crawl failed: {e.code()} - {e.details()}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in web crawl: {e}")
            return []

    def _steps_to_protos(self, steps: Optional[List[Dict[str, Any]]]) -> list:
        """Build BrowserStepProto list from steps dicts."""
        step_protos = []
        if steps:
            for s in steps:
                step = tool_service_pb2.BrowserStepProto(action=s.get("action", ""))
                if s.get("selector") is not None:
                    step.selector = s["selector"]
                if s.get("value") is not None:
                    step.value = s["value"]
                if s.get("wait_for") is not None:
                    step.wait_for = s["wait_for"]
                if s.get("timeout_seconds") is not None:
                    step.timeout_seconds = s["timeout_seconds"]
                if s.get("url") is not None:
                    step.url = s["url"]
                step_protos.append(step)
        return step_protos

    async def browser_run(
        self,
        url: str,
        final_action_type: str,
        final_selector: str,
        folder_path: str = "",
        user_id: str = "system",
        steps: Optional[List[Dict[str, Any]]] = None,
        connection_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        title: Optional[str] = None,
        goal: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run browser automation: steps then final action (download, click, extract, screenshot).
        Returns unified dict: success, error, document_id, filename, file_size_bytes, extracted_text, message.
        """
        try:
            await self._ensure_connected()
            step_protos = self._steps_to_protos(steps)
            request = tool_service_pb2.BrowserRunToolRequest(
                user_id=user_id,
                url=url,
                steps=step_protos,
                final_action_type=final_action_type or "download",
                final_selector=final_selector or "",
                folder_path=folder_path or "",
                tags=tags or [],
            )
            if connection_id is not None:
                request.connection_id = connection_id
            if title is not None:
                request.title = title
            if goal is not None:
                request.goal = goal
            response = await self._stub.BrowserRun(request)
            out = {
                "success": response.success,
                "error": response.error if response.HasField("error") and response.error else None,
                "document_id": response.document_id if response.HasField("document_id") else None,
                "filename": response.filename if response.HasField("filename") else None,
                "file_size_bytes": response.file_size_bytes if response.HasField("file_size_bytes") else None,
                "extracted_text": response.extracted_text if response.HasField("extracted_text") else None,
                "message": response.message if response.HasField("message") else None,
                "images_markdown": response.images_markdown if response.HasField("images_markdown") and response.images_markdown else None,
            }
            return out
        except grpc.RpcError as e:
            logger.error(f"Browser run failed: {e.code()} - {e.details()}")
            return {"success": False, "error": str(e.details())}
        except Exception as e:
            logger.error(f"Unexpected error in browser run: {e}")
            return {"success": False, "error": str(e)}

    async def browser_download(
        self,
        url: str,
        download_selector: str,
        folder_path: str,
        user_id: str = "system",
        steps: Optional[List[Dict[str, Any]]] = None,
        connection_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        title: Optional[str] = None,
        goal: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run browser automation: optional steps, trigger download by selector, save file to user folder.
        Wrapper around browser_run with final_action_type=download.
        Returns: success, document_id, filename, file_size_bytes, error.
        """
        result = await self.browser_run(
            url=url,
            final_action_type="download",
            final_selector=download_selector,
            folder_path=folder_path,
            user_id=user_id,
            steps=steps,
            connection_id=connection_id,
            tags=tags,
            title=title,
            goal=goal,
        )
        if not result.get("success"):
            return {
                "success": False,
                "document_id": "",
                "filename": "",
                "file_size_bytes": 0,
                "error": result.get("error"),
            }
        return {
            "success": True,
            "document_id": result.get("document_id") or "",
            "filename": result.get("filename") or "",
            "file_size_bytes": result.get("file_size_bytes") or 0,
            "error": result.get("error"),
        }

    async def browser_open_session(
        self,
        site_domain: str,
        user_id: str = "system",
        timeout_seconds: int = 30,
    ) -> Dict[str, Any]:
        """Create browser session; restore saved state for user/site if available. Returns session_id."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.BrowserOpenSessionRequest(
                user_id=user_id,
                site_domain=site_domain or "",
                timeout_seconds=timeout_seconds,
            )
            response = await self._stub.BrowserOpenSession(request)
            return {
                "success": response.success,
                "session_id": response.session_id if response.session_id else None,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error(f"BrowserOpenSession failed: {e.code()} - {e.details()}")
            return {"success": False, "session_id": None, "error": str(e.details())}
        except Exception as e:
            logger.error(f"Unexpected error in browser_open_session: {e}")
            return {"success": False, "session_id": None, "error": str(e)}

    async def browser_navigate(
        self,
        session_id: str,
        url: str,
        user_id: str = "system",
    ) -> Dict[str, Any]:
        try:
            await self._ensure_connected()
            request = tool_service_pb2.BrowserNavigateRequest(
                user_id=user_id,
                session_id=session_id,
                url=url or "",
            )
            response = await self._stub.BrowserNavigate(request)
            return {
                "success": response.success,
                "page_title": response.page_title if response.page_title else None,
                "current_url": response.current_url if response.current_url else None,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error(f"BrowserNavigate failed: {e.code()} - {e.details()}")
            return {"success": False, "error": str(e.details())}

    async def browser_click(
        self,
        session_id: str,
        selector: str,
        user_id: str = "system",
    ) -> Dict[str, Any]:
        try:
            await self._ensure_connected()
            request = tool_service_pb2.BrowserClickRequest(
                user_id=user_id,
                session_id=session_id,
                selector=selector or "",
            )
            response = await self._stub.BrowserClick(request)
            return {"success": response.success, "error": response.error if response.error else None}
        except grpc.RpcError as e:
            logger.error(f"BrowserClick failed: {e.code()} - {e.details()}")
            return {"success": False, "error": str(e.details())}

    async def browser_fill(
        self,
        session_id: str,
        selector: str,
        value: str,
        user_id: str = "system",
    ) -> Dict[str, Any]:
        try:
            await self._ensure_connected()
            request = tool_service_pb2.BrowserFillRequest(
                user_id=user_id,
                session_id=session_id,
                selector=selector or "",
                value=value or "",
            )
            response = await self._stub.BrowserFill(request)
            return {"success": response.success, "error": response.error if response.error else None}
        except grpc.RpcError as e:
            logger.error(f"BrowserFill failed: {e.code()} - {e.details()}")
            return {"success": False, "error": str(e.details())}

    async def browser_wait(
        self,
        session_id: str,
        user_id: str = "system",
        selector: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            await self._ensure_connected()
            request = tool_service_pb2.BrowserWaitRequest(user_id=user_id, session_id=session_id)
            if selector is not None:
                request.selector = selector
            if timeout_seconds is not None:
                request.timeout_seconds = timeout_seconds
            response = await self._stub.BrowserWait(request)
            return {
                "success": response.success,
                "found": response.found,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error(f"BrowserWait failed: {e.code()} - {e.details()}")
            return {"success": False, "found": False, "error": str(e.details())}

    async def browser_scroll(
        self,
        session_id: str,
        user_id: str = "system",
        direction: str = "down",
        amount_pixels: int = 800,
    ) -> Dict[str, Any]:
        try:
            await self._ensure_connected()
            request = tool_service_pb2.BrowserScrollRequest(
                user_id=user_id,
                session_id=session_id,
                direction=direction or "down",
                amount_pixels=amount_pixels if amount_pixels > 0 else 800,
            )
            response = await self._stub.BrowserScroll(request)
            return {
                "success": response.success,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error(f"BrowserScroll failed: {e.code()} - {e.details()}")
            return {"success": False, "error": str(e.details())}

    async def browser_extract(
        self,
        session_id: str,
        selector: str = "",
        user_id: str = "system",
    ) -> Dict[str, Any]:
        try:
            await self._ensure_connected()
            request = tool_service_pb2.BrowserExtractRequest(
                user_id=user_id,
                session_id=session_id,
                selector=selector or "",
            )
            response = await self._stub.BrowserExtract(request)
            return {
                "success": response.success,
                "extracted_text": response.extracted_text if response.extracted_text else None,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error(f"BrowserExtract failed: {e.code()} - {e.details()}")
            return {"success": False, "extracted_text": None, "error": str(e.details())}

    async def browser_inspect(
        self,
        session_id: str,
        user_id: str = "system",
    ) -> Dict[str, Any]:
        try:
            await self._ensure_connected()
            request = tool_service_pb2.BrowserInspectRequest(
                user_id=user_id,
                session_id=session_id,
            )
            response = await self._stub.BrowserInspect(request)
            return {
                "success": response.success,
                "page_structure": response.page_structure if response.page_structure else None,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error(f"BrowserInspect failed: {e.code()} - {e.details()}")
            return {"success": False, "page_structure": None, "error": str(e.details())}

    async def browser_screenshot(
        self,
        session_id: str,
        user_id: str = "system",
        folder_path: str = "",
        tags: Optional[List[str]] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            await self._ensure_connected()
            request = tool_service_pb2.BrowserScreenshotRequest(
                user_id=user_id,
                session_id=session_id,
                folder_path=folder_path or "",
                tags=tags or [],
            )
            if title is not None:
                request.title = title
            response = await self._stub.BrowserScreenshot(request)
            return {
                "success": response.success,
                "images_markdown": response.images_markdown if response.images_markdown else None,
                "document_id": response.document_id if response.document_id else None,
                "filename": response.filename if response.filename else None,
                "file_size_bytes": response.file_size_bytes if response.file_size_bytes else None,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error(f"BrowserScreenshot failed: {e.code()} - {e.details()}")
            return {"success": False, "error": str(e.details())}

    async def browser_download_file(
        self,
        session_id: str,
        selector: str,
        folder_path: str,
        user_id: str = "system",
        tags: Optional[List[str]] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            await self._ensure_connected()
            request = tool_service_pb2.BrowserDownloadFileRequest(
                user_id=user_id,
                session_id=session_id,
                selector=selector or "",
                folder_path=folder_path or "",
                tags=tags or [],
            )
            if title is not None:
                request.title = title
            response = await self._stub.BrowserDownloadFile(request)
            return {
                "success": response.success,
                "document_id": response.document_id if response.document_id else None,
                "filename": response.filename if response.filename else None,
                "file_size_bytes": response.file_size_bytes if response.file_size_bytes else None,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error(f"BrowserDownloadFile failed: {e.code()} - {e.details()}")
            return {"success": False, "error": str(e.details())}

    async def browser_close_session(
        self,
        session_id: str,
        site_domain: str,
        save_state: bool = False,
        user_id: str = "system",
    ) -> Dict[str, Any]:
        try:
            await self._ensure_connected()
            request = tool_service_pb2.BrowserCloseSessionRequest(
                user_id=user_id,
                session_id=session_id,
                site_domain=site_domain or "",
                save_state=save_state,
            )
            response = await self._stub.BrowserCloseSession(request)
            return {
                "success": response.success,
                "session_saved": response.session_saved,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error(f"BrowserCloseSession failed: {e.code()} - {e.details()}")
            return {"success": False, "session_saved": False, "error": str(e.details())}

    async def crawl_website_recursive(
        self,
        start_url: str,
        max_pages: int = 500,
        max_depth: int = 10,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Recursively crawl entire website
        
        Args:
            start_url: Starting URL for the crawl
            max_pages: Maximum number of pages to crawl
            max_depth: Maximum depth to traverse
            user_id: User ID
            
        Returns:
            Dictionary with crawl results and statistics
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.RecursiveWebsiteCrawlRequest(
                start_url=start_url,
                max_pages=max_pages,
                max_depth=max_depth,
                user_id=user_id
            )
            
            response = await self._stub.CrawlWebsiteRecursive(request)
            
            if not response.success:
                return {
                    "success": False,
                    "error": response.error if response.error else "Unknown error"
                }
            
            # Convert response to dict format
            crawled_pages = []
            for page in response.crawled_pages:
                page_dict = {
                    "url": page.url,
                    "content_type": page.content_type,
                    "markdown_content": page.markdown_content,
                    "html_content": page.html_content,
                    "metadata": dict(page.metadata),
                    "internal_links": list(page.internal_links),
                    "image_links": list(page.image_links),
                    "document_links": list(page.document_links),
                    "depth": page.depth,
                    "crawl_time": page.crawl_time
                }
                if page.parent_url:
                    page_dict["parent_url"] = page.parent_url
                if page.binary_content:
                    page_dict["binary_content"] = bytes(page.binary_content)
                if page.filename:
                    page_dict["filename"] = page.filename
                if page.mime_type:
                    page_dict["mime_type"] = page.mime_type
                if page.size_bytes:
                    page_dict["size_bytes"] = page.size_bytes
                
                crawled_pages.append(page_dict)
            
            return {
                "success": True,
                "start_url": response.start_url,
                "base_domain": response.base_domain,
                "crawl_session_id": response.crawl_session_id,
                "total_items_crawled": response.total_items_crawled,
                "html_pages_crawled": response.html_pages_crawled,
                "images_downloaded": response.images_downloaded,
                "documents_downloaded": response.documents_downloaded,
                "total_items_failed": response.total_items_failed,
                "max_depth_reached": response.max_depth_reached,
                "elapsed_time_seconds": response.elapsed_time_seconds,
                "crawled_pages": crawled_pages
            }
            
        except grpc.RpcError as e:
            logger.error(f"Recursive website crawl failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "error": f"gRPC error: {e.details()}"
            }
        except Exception as e:
            logger.error(f"Unexpected error in recursive website crawl: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def crawl_site(
        self,
        seed_url: str,
        query_criteria: str,
        max_pages: int = 50,
        max_depth: int = 2,
        allowed_path_prefix: Optional[str] = None,
        include_pdfs: bool = False,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Domain-scoped crawl starting from seed URL, filtering by query criteria
        
        Args:
            seed_url: Starting URL for the crawl
            query_criteria: Criteria to identify relevant pages
            max_pages: Maximum number of pages to crawl
            max_depth: Maximum depth to traverse
            allowed_path_prefix: Optional path prefix to restrict crawl
            include_pdfs: Whether to include PDFs
            user_id: User ID
            
        Returns:
            Dictionary with crawl results
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.DomainCrawlRequest(
                seed_url=seed_url,
                query_criteria=query_criteria,
                max_pages=max_pages,
                max_depth=max_depth,
                allowed_path_prefix=allowed_path_prefix if allowed_path_prefix else "",
                include_pdfs=include_pdfs,
                user_id=user_id
            )
            
            response = await self._stub.CrawlSite(request)
            
            if not response.success:
                return {
                    "success": False,
                    "error": response.error if response.error else "Unknown error"
                }
            
            # Convert response to dict format
            results = []
            for result in response.results:
                results.append({
                    "url": result.url,
                    "title": result.title,
                    "full_content": result.full_content,
                    "metadata": dict(result.metadata),
                    "relevance_score": result.relevance_score,
                    "success": result.success
                })
            
            return {
                "success": True,
                "domain": response.domain,
                "successful_crawls": response.successful_crawls,
                "urls_considered": response.urls_considered,
                "results": results
            }
            
        except grpc.RpcError as e:
            logger.error(f"Domain crawl failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "error": f"gRPC error: {e.details()}"
            }
        except Exception as e:
            logger.error(f"Unexpected error in domain crawl: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def analyze_website_security(
        self,
        target_url: str,
        user_id: str = "system",
        scan_depth: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Analyze website for security vulnerabilities (passive reconnaissance).

        Args:
            target_url: Target website URL
            user_id: User ID for access control
            scan_depth: "basic", "intermediate", or "comprehensive"

        Returns:
            Dict with findings, technology_stack, risk_score, summary, disclaimer
        """
        try:
            await self._ensure_connected()

            request = tool_service_pb2.SecurityAnalysisRequest(
                target_url=target_url,
                user_id=user_id,
                scan_depth=scan_depth
            )

            response = await self._stub.AnalyzeWebsiteSecurity(request)

            findings = []
            for finding in response.findings:
                findings.append({
                    "category": finding.category,
                    "severity": finding.severity,
                    "title": finding.title,
                    "description": finding.description,
                    "url": finding.url if finding.url else None,
                    "evidence": finding.evidence if finding.evidence else None,
                    "remediation": finding.remediation
                })

            security_headers = dict(response.security_headers)
            if "present" in security_headers and isinstance(security_headers["present"], str):
                security_headers["present"] = [s.strip() for s in security_headers["present"].split(",") if s.strip()]
            if "missing" in security_headers and isinstance(security_headers["missing"], str):
                security_headers["missing"] = [s.strip() for s in security_headers["missing"].split(",") if s.strip()]

            return {
                "success": response.success,
                "target_url": response.target_url,
                "scan_timestamp": response.scan_timestamp,
                "findings": findings,
                "technology_stack": dict(response.technology_stack),
                "security_headers": security_headers,
                "risk_score": response.risk_score,
                "summary": response.summary,
                "disclaimer": response.disclaimer,
                "error": response.error if response.error else None
            }

        except grpc.RpcError as e:
            logger.error(f"Security analysis failed: {e.code()} - {e.details()}")
            return {"success": False, "error": str(e), "findings": []}
        except Exception as e:
            logger.error(f"Unexpected error in security analysis: {e}")
            return {"success": False, "error": str(e), "findings": []}

    # ===== Query Enhancement =====

    async def expand_query(
        self,
        query: str,
        num_variations: int = 3,
        user_id: str = "system",
        conversation_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Expand query with variations
        
        Args:
            query: Original query
            num_variations: Number of variations to generate
            user_id: User ID
            conversation_context: Optional conversation context (last 2 messages) to help resolve vague references
            
        Returns:
            Dict with 'original_query', 'expanded_queries', 'key_entities', 'expansion_count'
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.QueryExpansionRequest(
                query=query,
                num_variations=num_variations,
                user_id=user_id
            )
            
            # Add conversation context if provided
            if conversation_context:
                request.conversation_context = conversation_context
            
            response = await self._stub.ExpandQuery(request)
            
            return {
                'original_query': response.original_query,
                'expanded_queries': list(response.expanded_queries),
                'key_entities': list(response.key_entities),
                'expansion_count': response.expansion_count
            }
            
        except grpc.RpcError as e:
            logger.error(f"Query expansion failed: {e.code()} - {e.details()}")
            return {
                'original_query': query,
                'expanded_queries': [query],
                'key_entities': [],
                'expansion_count': 1
            }
        except Exception as e:
            logger.error(f"Unexpected error in query expansion: {e}")
            return {
                'original_query': query,
                'expanded_queries': [query],
                'key_entities': [],
                'expansion_count': 1
            }
    
    # ===== Conversation Cache =====
    
    async def search_images(
        self,
        query: str,
        image_type: Optional[str] = None,
        date: Optional[str] = None,
        author: Optional[str] = None,
        series: Optional[str] = None,
        limit: int = 10,
        user_id: str = "system",
        is_random: bool = False,
        exclude_document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Search for images with metadata sidecars
        
        Args:
            query: Search query
            image_type: Optional filter by image type
            date: Optional date filter (YYYY-MM-DD)
            author: Optional author filter
            series: Optional series filter
            limit: Maximum number of results
            user_id: User ID for access control
            is_random: If True, return random images instead of semantic search
            
        Returns:
            Dict with 'images_markdown' (base64 embedded images) and 'metadata' (list of metadata dicts)
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.ImageSearchRequest(
                query=query,
                image_type=image_type or "",
                date=date or "",
                author=author or "",
                series=series or "",
                limit=limit,
                user_id=user_id,
                is_random=is_random,
                exclude_document_ids=exclude_document_ids or [],
            )
            
            response = await self._stub.SearchImages(request)
            
            if response.success:
                # Convert protobuf metadata to Python dicts
                metadata_list = []
                for pb_meta in response.metadata:
                    metadata_list.append({
                        "title": pb_meta.title,
                        "date": pb_meta.date,
                        "series": pb_meta.series,
                        "author": pb_meta.author,
                        "content": pb_meta.content,
                        "tags": list(pb_meta.tags),
                        "image_type": pb_meta.image_type
                    })
                
                out = {
                    "images_markdown": response.results,
                    "metadata": metadata_list
                }
                # Pass through structured images for UI/Telegram (no markdown in response text)
                structured_json = getattr(response, "structured_images_json", None)
                if structured_json:
                    import json
                    try:
                        out["images"] = json.loads(structured_json)
                    except (TypeError, json.JSONDecodeError):
                        pass
                return out
            else:
                logger.error(f"Image search failed: {response.error}")
                return {
                    "images_markdown": f"Error searching images: {response.error}",
                    "metadata": []
                }
            
        except grpc.RpcError as e:
            logger.error(f"Image search failed: {e.code()} - {e.details()}")
            return {
                "images_markdown": f"Error searching images: {str(e)}",
                "metadata": [],
            }
        except Exception as e:
            logger.error(f"Unexpected error in image search: {e}")
            return {
                "images_markdown": f"Error searching images: {str(e)}",
                "metadata": [],
            }
    
    # ===== Face Analysis Operations =====
    
    async def detect_faces(
        self,
        attachment_path: str,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Detect faces in an attached image
        
        Args:
            attachment_path: Full path to image file
            user_id: User ID for access control
            
        Returns:
            Dict with success, faces (list of face detections), face_count, image_width, image_height
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.DetectFacesRequest(
                attachment_path=attachment_path,
                user_id=user_id
            )
            
            response = await self._stub.DetectFaces(request)
            
            if response.success:
                faces_list = []
                for pb_face in response.faces:
                    faces_list.append({
                        "face_encoding": list(pb_face.face_encoding),
                        "bbox_x": pb_face.bbox_x,
                        "bbox_y": pb_face.bbox_y,
                        "bbox_width": pb_face.bbox_width,
                        "bbox_height": pb_face.bbox_height
                    })
                
                return {
                    "success": True,
                    "faces": faces_list,
                    "face_count": response.face_count,
                    "image_width": response.image_width if response.HasField("image_width") else None,
                    "image_height": response.image_height if response.HasField("image_height") else None
                }
            else:
                logger.error(f"Face detection failed: {response.error}")
                return {
                    "success": False,
                    "faces": [],
                    "face_count": 0,
                    "error": response.error if response.HasField("error") else "Unknown error"
                }
            
        except grpc.RpcError as e:
            logger.error(f"Face detection failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "faces": [],
                "face_count": 0,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error in face detection: {e}")
            return {
                "success": False,
                "faces": [],
                "face_count": 0,
                "error": str(e)
            }
    
    async def identify_faces(
        self,
        attachment_path: str,
        user_id: str = "system",
        confidence_threshold: float = 0.85
    ) -> Dict[str, Any]:
        """
        Identify people in an attached image by matching against known identities
        
        Args:
            attachment_path: Full path to image file
            user_id: User ID for access control
            confidence_threshold: Minimum confidence for identity matches (default: 0.85)
            
        Returns:
            Dict with success, identified_faces (list of identified faces), face_count, identified_count
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.IdentifyFacesRequest(
                attachment_path=attachment_path,
                user_id=user_id,
                confidence_threshold=confidence_threshold
            )
            
            response = await self._stub.IdentifyFaces(request)
            
            if response.success:
                identified_faces_list = []
                for pb_face in response.identified_faces:
                    identified_faces_list.append({
                        "identity_name": pb_face.identity_name,
                        "confidence": pb_face.confidence,
                        "bbox_x": pb_face.bbox_x,
                        "bbox_y": pb_face.bbox_y,
                        "bbox_width": pb_face.bbox_width,
                        "bbox_height": pb_face.bbox_height
                    })
                
                return {
                    "success": True,
                    "identified_faces": identified_faces_list,
                    "face_count": response.face_count,
                    "identified_count": response.identified_count
                }
            else:
                logger.error(f"Face identification failed: {response.error}")
                return {
                    "success": False,
                    "identified_faces": [],
                    "face_count": 0,
                    "identified_count": 0,
                    "error": response.error if response.HasField("error") else "Unknown error"
                }
            
        except grpc.RpcError as e:
            logger.error(f"Face identification failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "identified_faces": [],
                "face_count": 0,
                "identified_count": 0,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error in face identification: {e}")
            return {
                "success": False,
                "identified_faces": [],
                "face_count": 0,
                "identified_count": 0,
                "error": str(e)
            }
    
    async def search_conversation_cache(
        self,
        query: str,
        conversation_id: str = None,
        freshness_hours: int = 24,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Search conversation cache for previous research
        
        Args:
            query: Search query
            conversation_id: Conversation ID (optional)
            freshness_hours: How recent to search (hours)
            user_id: User ID
            
        Returns:
            Dict with 'cache_hit' and 'entries'
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.CacheSearchRequest(
                query=query,
                conversation_id=conversation_id if conversation_id else "",
                freshness_hours=freshness_hours,
                user_id=user_id
            )
            
            response = await self._stub.SearchConversationCache(request)
            
            entries = []
            for entry in response.entries:
                entries.append({
                    'content': entry.content,
                    'timestamp': entry.timestamp,
                    'agent_name': entry.agent_name,
                    'relevance_score': entry.relevance_score
                })
            
            return {
                'cache_hit': response.cache_hit,
                'entries': entries
            }
            
        except grpc.RpcError as e:
            logger.error(f"Cache search failed: {e.code()} - {e.details()}")
            return {'cache_hit': False, 'entries': []}
        except Exception as e:
            logger.error(f"Unexpected error in cache search: {e}")
            return {'cache_hit': False, 'entries': []}
    
    # ===== Conversation Operations =====
    
    async def update_conversation_title(
        self,
        conversation_id: str,
        title: str,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Update conversation title
        
        Args:
            conversation_id: Conversation ID to update
            title: New title
            user_id: User ID (required - must match conversation owner)
        
        Returns:
            Dict with success, conversation_id, title, and message
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.UpdateConversationTitleRequest(
                user_id=user_id,
                conversation_id=conversation_id,
                title=title
            )
            
            response = await self._stub.UpdateConversationTitle(request)
            
            return {
                "success": response.success,
                "conversation_id": response.conversation_id,
                "title": response.title,
                "message": response.message,
                "error": response.error if hasattr(response, 'error') and response.error else None
            }
            
        except grpc.RpcError as e:
            logger.error(f"Update conversation title failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "error": f"{e.code()}: {e.details()}",
                "message": "Failed to update conversation title"
            }
        except Exception as e:
            logger.error(f"Unexpected error updating conversation title: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to update conversation title"
            }
    
    # ===== Visualization Operations =====
    
    async def create_chart(
        self,
        chart_type: str,
        data: Dict[str, Any],
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        interactive: bool = True,
        color_scheme: str = "plotly",
        width: int = 800,
        height: int = 600,
    ) -> Dict[str, Any]:
        """
        Create a chart or graph from structured data
        
        Args:
            chart_type: Type of chart (bar, line, pie, scatter, area, heatmap, box_plot, histogram)
            data: Chart data (format depends on chart type)
            title: Chart title (optional)
            x_label: X-axis label (optional)
            y_label: Y-axis label (optional)
            interactive: Generate interactive chart (default: True)
            color_scheme: Color scheme to use (default: "plotly")
            width: Chart width in pixels (default: 800)
            height: Chart height in pixels (default: 600)

        Returns:
            Dict with success status, output format, and chart_data
        """
        try:
            await self._ensure_connected()
            
            # Serialize data to JSON
            data_json = json.dumps(data)
            
            request = tool_service_pb2.CreateChartRequest(
                chart_type=chart_type,
                data_json=data_json,
                title=title,
                x_label=x_label,
                y_label=y_label,
                interactive=interactive,
                color_scheme=color_scheme,
                width=width,
                height=height,
            )
            
            response = await self._stub.CreateChart(request)
            
            if response.success:
                # Parse metadata JSON
                metadata = {}
                if response.metadata_json:
                    try:
                        metadata = json.loads(response.metadata_json)
                    except json.JSONDecodeError:
                        pass
                
                result = {
                    "success": True,
                    "chart_type": response.chart_type,
                    "output_format": response.output_format,
                    "chart_data": response.chart_data,
                    "metadata": metadata
                }

                return result
            else:
                return {
                    "success": False,
                    "error": response.error if response.error else "Unknown error"
                }
            
        except grpc.RpcError as e:
            logger.error(f"Create chart failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "error": f"{e.code()}: {e.details()}"
            }
        except Exception as e:
            logger.error(f"Unexpected error creating chart: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # ===== File Analysis Operations =====
    
    async def analyze_text_content(
        self,
        content: str,
        include_advanced: bool = False,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Analyze text content and return metrics
        
        Args:
            content: Text content to analyze
            include_advanced: If True, include advanced metrics (averages, etc.)
            user_id: User ID for logging
            
        Returns:
            Dict with text analysis metrics:
            - word_count: int
            - line_count: int
            - non_empty_line_count: int
            - character_count: int
            - character_count_no_spaces: int
            - paragraph_count: int
            - sentence_count: int
            - avg_words_per_sentence: float (if include_advanced)
            - avg_words_per_paragraph: float (if include_advanced)
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.TextAnalysisRequest(
                content=content,
                include_advanced=include_advanced,
                user_id=user_id
            )
            
            response = await self._stub.AnalyzeTextContent(request)
            
            # Convert proto response to dict
            metrics = {
                "word_count": response.word_count,
                "line_count": response.line_count,
                "non_empty_line_count": response.non_empty_line_count,
                "character_count": response.character_count,
                "character_count_no_spaces": response.character_count_no_spaces,
                "paragraph_count": response.paragraph_count,
                "sentence_count": response.sentence_count,
            }
            
            # Add advanced metrics if requested
            if include_advanced:
                metrics["avg_words_per_sentence"] = response.avg_words_per_sentence
                metrics["avg_words_per_paragraph"] = response.avg_words_per_paragraph
            
            # Parse metadata JSON if present
            if response.metadata_json:
                try:
                    metadata = json.loads(response.metadata_json)
                    metrics["metadata"] = metadata
                except json.JSONDecodeError:
                    pass
            
            return metrics
            
        except grpc.RpcError as e:
            logger.error(f"Analyze text content failed: {e.code()} - {e.details()}")
            # Return empty metrics on error
            return {
                "word_count": 0,
                "line_count": 0,
                "non_empty_line_count": 0,
                "character_count": 0,
                "character_count_no_spaces": 0,
                "paragraph_count": 0,
                "sentence_count": 0,
                "error": f"{e.code()}: {e.details()}"
            }
        except Exception as e:
            logger.error(f"Unexpected error analyzing text content: {e}")
            return {
                "word_count": 0,
                "line_count": 0,
                "non_empty_line_count": 0,
                "character_count": 0,
                "character_count_no_spaces": 0,
                "paragraph_count": 0,
                "sentence_count": 0,
                "error": str(e)
            }

    # ===== Org Inbox Operations =====
    
    async def list_org_inbox_items(
        self,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        List all org inbox items for user
        
        Args:
            user_id: User ID for access control
            
        Returns:
            Dict with 'success', 'items' (list), 'path', and optional 'error'
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.ListOrgInboxItemsRequest(
                user_id=user_id
            )
            
            response = await self._stub.ListOrgInboxItems(request)
            
            if not response.success:
                return {
                    "success": False,
                    "error": response.error if response.error else "Unknown error",
                    "items": [],
                    "path": ""
                }
            
            # Convert proto items to dict format
            items = []
            for item in response.items:
                items.append({
                    "line_index": item.line_index,
                    "text": item.text,
                    "item_type": item.item_type,
                    "todo_state": item.todo_state,
                    "tags": list(item.tags),
                    "is_done": item.is_done
                })
            
            return {
                "success": True,
                "items": items,
                "path": response.path
            }
            
        except grpc.RpcError as e:
            logger.error(f"List org inbox items failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "error": f"gRPC error: {e.details()}",
                "items": [],
                "path": ""
            }
        except Exception as e:
            logger.error(f"Unexpected error listing org inbox items: {e}")
            return {
                "success": False,
                "error": str(e),
                "items": [],
                "path": ""
            }

    async def add_org_inbox_item(
        self,
        user_id: str,
        text: str,
        kind: str = "todo",
        schedule: Optional[str] = None,
        repeater: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Add a new item to the user's org inbox.

        Args:
            user_id: User ID for access control
            text: Item text (e.g. "Get groceries")
            kind: "todo", "checkbox", "event", or "contact"
            schedule: Optional org timestamp (e.g. "<2026-02-05 Thu>")
            repeater: Optional repeater (e.g. "+1w")
            tags: Optional list of tags

        Returns:
            Dict with 'success', 'line_index', 'message', and optional 'error'
        """
        try:
            await self._ensure_connected()
            request = tool_service_pb2.AddOrgInboxItemRequest(
                user_id=user_id,
                text=text,
                kind=kind,
                tags=tags or [],
            )
            if schedule is not None:
                request.schedule = schedule
            if repeater is not None:
                request.repeater = repeater
            response = await self._stub.AddOrgInboxItem(request)
            if not response.success:
                return {
                    "success": False,
                    "line_index": -1,
                    "message": "",
                    "error": response.error if response.error else "Unknown error",
                }
            return {
                "success": True,
                "line_index": response.line_index,
                "message": response.message or "",
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("Add org inbox item failed: %s - %s", e.code(), e.details())
            return {
                "success": False,
                "line_index": -1,
                "message": "",
                "error": str(e.details()),
            }
        except Exception as e:
            logger.error("Unexpected error adding org inbox item: %s", e)
            return {
                "success": False,
                "line_index": -1,
                "message": "",
                "error": str(e),
            }

    async def capture_journal_entry(
        self,
        user_id: str,
        content: str,
        entry_date: Optional[str] = None,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Append an entry to the user's journal. Respects journal organization
        preferences and creates date headings if missing.

        Args:
            user_id: User ID for access control
            content: Entry body text
            entry_date: Optional date (YYYY-MM-DD); omit for today
            title: Optional heading title
            tags: Optional list of tags

        Returns:
            Dict with success, message, entry_preview, file_path, document_id, error
        """
        try:
            await self._ensure_connected()
            request = tool_service_pb2.CaptureJournalEntryRequest(
                user_id=user_id,
                content=content or "",
                tags=tags or [],
            )
            if entry_date is not None:
                request.entry_date = entry_date
            if title is not None:
                request.title = title
            response = await self._stub.CaptureJournalEntry(request)
            if not response.success:
                return {
                    "success": False,
                    "message": response.message or "",
                    "entry_preview": response.entry_preview if response.HasField("entry_preview") else None,
                    "file_path": response.file_path if response.HasField("file_path") else None,
                    "document_id": response.document_id if response.HasField("document_id") else None,
                    "error": response.error if response.HasField("error") else "Unknown error",
                }
            return {
                "success": True,
                "message": response.message or "",
                "entry_preview": response.entry_preview if response.HasField("entry_preview") else None,
                "file_path": response.file_path if response.HasField("file_path") else None,
                "document_id": response.document_id if response.HasField("document_id") else None,
                "error": None,
            }
        except grpc.RpcError as e:
            logger.error("CaptureJournalEntry failed: %s - %s", e.code(), e.details())
            return {
                "success": False,
                "message": "",
                "entry_preview": None,
                "file_path": None,
                "document_id": None,
                "error": str(e.details()),
            }
        except Exception as e:
            logger.error("Unexpected error in capture_journal_entry: %s", e)
            return {
                "success": False,
                "message": "",
                "entry_preview": None,
                "file_path": None,
                "document_id": None,
                "error": str(e),
            }

    async def get_journal_entry(
        self,
        user_id: str,
        date: str = "today",
    ) -> Dict[str, Any]:
        """Read one date's journal entry (section-aware). date: YYYY-MM-DD or 'today'."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetJournalEntryRequest(user_id=user_id, date=date or "today")
            response = await self._stub.GetJournalEntry(request)
            return {
                "success": response.success,
                "content": response.content or "",
                "date": response.date or "",
                "heading": response.heading or "",
                "document_id": response.document_id if response.document_id else None,
                "file_path": response.file_path if response.file_path else None,
                "has_content": response.has_content,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("GetJournalEntry failed: %s - %s", e.code(), e.details())
            return {
                "success": False,
                "content": "",
                "date": "",
                "heading": "",
                "document_id": None,
                "file_path": None,
                "has_content": False,
                "error": str(e.details()),
            }
        except Exception as e:
            logger.error("Unexpected error in get_journal_entry: %s", e)
            return {
                "success": False,
                "content": "",
                "date": "",
                "heading": "",
                "document_id": None,
                "file_path": None,
                "has_content": False,
                "error": str(e),
            }

    async def get_journal_entries(
        self,
        user_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_entries: int = 100,
    ) -> Dict[str, Any]:
        """Get full content of journal entries in a date range (review/lookback)."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetJournalEntriesRequest(user_id=user_id)
            if start_date is not None:
                request.start_date = start_date
            if end_date is not None:
                request.end_date = end_date
            if max_entries is not None and max_entries > 0:
                request.max_entries = max_entries
            response = await self._stub.GetJournalEntries(request)
            if not response.success:
                return {
                    "success": False,
                    "entries": [],
                    "total": 0,
                    "error": response.error or "",
                }
            entries = [
                {
                    "date": e.date,
                    "content": e.content or "",
                    "heading": e.heading or "",
                    "has_content": e.has_content,
                }
                for e in response.entries
            ]
            return {"success": True, "entries": entries, "total": response.total, "error": None}
        except grpc.RpcError as e:
            logger.error("GetJournalEntries failed: %s - %s", e.code(), e.details())
            return {"success": False, "entries": [], "total": 0, "error": str(e.details())}
        except Exception as e:
            logger.error("Unexpected error in get_journal_entries: %s", e)
            return {"success": False, "entries": [], "total": 0, "error": str(e)}

    async def update_journal_entry(
        self,
        user_id: str,
        date: str,
        content: str,
        mode: str = "replace",
    ) -> Dict[str, Any]:
        """Replace or append to a single date's journal section only. mode: 'replace' or 'append'."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.UpdateJournalEntryRequest(
                user_id=user_id,
                date=date,
                content=content or "",
                mode=mode or "replace",
            )
            response = await self._stub.UpdateJournalEntry(request)
            return {
                "success": response.success,
                "date": response.date or "",
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("UpdateJournalEntry failed: %s - %s", e.code(), e.details())
            return {"success": False, "date": date, "error": str(e.details())}
        except Exception as e:
            logger.error("Unexpected error in update_journal_entry: %s", e)
            return {"success": False, "date": date, "error": str(e)}

    async def list_journal_entries(
        self,
        user_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List journal entries in a date range with metadata (date, word_count, has_content)."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListJournalEntriesRequest(user_id=user_id)
            if start_date is not None:
                request.start_date = start_date
            if end_date is not None:
                request.end_date = end_date
            response = await self._stub.ListJournalEntries(request)
            if not response.success:
                return {
                    "success": False,
                    "entries": [],
                    "total": 0,
                    "error": response.error or "",
                }
            entries = [
                {"date": e.date, "word_count": e.word_count, "has_content": e.has_content}
                for e in response.entries
            ]
            return {"success": True, "entries": entries, "total": response.total, "error": None}
        except grpc.RpcError as e:
            logger.error("ListJournalEntries failed: %s - %s", e.code(), e.details())
            return {"success": False, "entries": [], "total": 0, "error": str(e.details())}
        except Exception as e:
            logger.error("Unexpected error in list_journal_entries: %s", e)
            return {"success": False, "entries": [], "total": 0, "error": str(e)}

    async def search_journal(
        self,
        user_id: str,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search within journal entry content in a date range. Returns list of {date, excerpt}."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.SearchJournalRequest(user_id=user_id, query=query or "")
            if start_date is not None:
                request.start_date = start_date
            if end_date is not None:
                request.end_date = end_date
            response = await self._stub.SearchJournal(request)
            if not response.success:
                return {
                    "success": False,
                    "results": [],
                    "count": 0,
                    "error": response.error or "",
                }
            results = [{"date": r.date, "excerpt": r.excerpt or ""} for r in response.results]
            return {"success": True, "results": results, "count": response.count, "error": None}
        except grpc.RpcError as e:
            logger.error("SearchJournal failed: %s - %s", e.code(), e.details())
            return {"success": False, "results": [], "count": 0, "error": str(e.details())}
        except Exception as e:
            logger.error("Unexpected error in search_journal: %s", e)
            return {"success": False, "results": [], "count": 0, "error": str(e)}

    async def get_agent_run_history(
        self,
        user_id: str,
        agent_profile_id: Optional[str] = None,
        limit: int = 10,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query agent run history for the user. Optional filter by agent_profile_id, status, date range."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetAgentRunHistoryRequest(user_id=user_id)
            if agent_profile_id is not None:
                request.agent_profile_id = agent_profile_id
            if limit > 0:
                request.limit = min(limit, 50)
            if status is not None:
                request.status = status
            if start_date is not None:
                request.start_date = start_date
            if end_date is not None:
                request.end_date = end_date
            response = await self._stub.GetAgentRunHistory(request)
            if not response.success:
                return {
                    "success": False,
                    "runs": [],
                    "total": 0,
                    "agent_name": "",
                    "error": response.error or "",
                }
            runs = []
            for r in response.runs:
                runs.append({
                    "execution_id": r.execution_id or "",
                    "agent_name": r.agent_name or "",
                    "query": r.query or "",
                    "status": r.status or "",
                    "started_at": r.started_at or "",
                    "duration_ms": r.duration_ms if r.HasField("duration_ms") else None,
                    "connectors_called": list(r.connectors_called) if r.connectors_called else [],
                    "entities_discovered": r.entities_discovered or 0,
                    "error_details": r.error_details if r.HasField("error_details") and r.error_details else None,
                    "steps_completed": r.steps_completed or 0,
                    "steps_total": r.steps_total or 0,
                })
            return {
                "success": True,
                "runs": runs,
                "total": response.total,
                "agent_name": response.agent_name or "",
                "error": None,
            }
        except grpc.RpcError as e:
            logger.error("GetAgentRunHistory failed: %s - %s", e.code(), e.details())
            return {
                "success": False,
                "runs": [],
                "total": 0,
                "agent_name": "",
                "error": str(e.details()),
            }
        except Exception as e:
            logger.error("Unexpected error in get_agent_run_history: %s", e)
            return {
                "success": False,
                "runs": [],
                "total": 0,
                "agent_name": "",
                "error": str(e),
            }

    async def get_execution_trace(
        self,
        user_id: str,
        execution_id: str,
        include_io: bool = True,
        include_tool_calls: bool = False,
    ) -> Dict[str, Any]:
        """Fetch one agent execution with per-step trace (same data as REST execution detail)."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetExecutionTraceRequest(
                user_id=user_id,
                execution_id=execution_id or "",
            )
            request.include_io = include_io
            request.include_tool_calls = include_tool_calls
            response = await self._stub.GetExecutionTrace(request)
            if not response.success:
                return {
                    "success": False,
                    "error": response.error or "Failed to get execution trace.",
                    "execution_id": response.execution_id or execution_id or "",
                    "agent_name": "",
                    "query": "",
                    "status": "",
                    "started_at": "",
                    "completed_at": "",
                    "duration_ms": None,
                    "tokens_input": None,
                    "tokens_output": None,
                    "cost_usd": None,
                    "model_used": "",
                    "error_details": "",
                    "steps": [],
                }
            steps = []
            for st in response.steps:
                step_dict: Dict[str, Any] = {
                    "step_index": st.step_index,
                    "step_name": st.step_name or "",
                    "step_type": st.step_type or "",
                    "action_name": st.action_name or "",
                    "status": st.status or "",
                    "started_at": st.started_at or "",
                    "completed_at": st.completed_at or "",
                    "duration_ms": st.duration_ms if st.HasField("duration_ms") else None,
                    "inputs_json": st.inputs_json or "",
                    "outputs_json": st.outputs_json or "",
                    "error_details": st.error_details or "",
                    "tool_call_trace_json": st.tool_call_trace_json or "",
                    "input_tokens": st.input_tokens or 0,
                    "output_tokens": st.output_tokens or 0,
                }
                steps.append(step_dict)
            return {
                "success": True,
                "error": None,
                "execution_id": response.execution_id or "",
                "agent_name": response.agent_name or "",
                "query": response.query or "",
                "status": response.status or "",
                "started_at": response.started_at or "",
                "completed_at": response.completed_at or "",
                "duration_ms": response.duration_ms if response.HasField("duration_ms") else None,
                "tokens_input": response.tokens_input if response.HasField("tokens_input") else None,
                "tokens_output": response.tokens_output if response.HasField("tokens_output") else None,
                "cost_usd": response.cost_usd if response.HasField("cost_usd") else None,
                "model_used": response.model_used or "",
                "error_details": response.error_details or "",
                "steps": steps,
            }
        except grpc.RpcError as e:
            logger.error("GetExecutionTrace failed: %s - %s", e.code(), e.details())
            return {
                "success": False,
                "error": str(e.details()),
                "execution_id": execution_id or "",
                "agent_name": "",
                "query": "",
                "status": "",
                "started_at": "",
                "completed_at": "",
                "duration_ms": None,
                "tokens_input": None,
                "tokens_output": None,
                "cost_usd": None,
                "model_used": "",
                "error_details": "",
                "steps": [],
            }
        except Exception as e:
            logger.error("Unexpected error in get_execution_trace: %s", e)
            return {
                "success": False,
                "error": str(e),
                "execution_id": execution_id or "",
                "agent_name": "",
                "query": "",
                "status": "",
                "started_at": "",
                "completed_at": "",
                "duration_ms": None,
                "tokens_input": None,
                "tokens_output": None,
                "cost_usd": None,
                "model_used": "",
                "error_details": "",
                "steps": [],
            }

    # ===== Universal Todo Operations =====

    async def list_todos(
        self,
        user_id: str,
        scope: str = "all",
        states: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        query: str = "",
        limit: int = 0,
        include_archives: bool = False,
        include_body: bool = False,
        closed_since_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """List todos. scope: all, inbox, or file path. limit 0 = no cap. closed_since_days: only DONE items closed in last N days (e.g. 7 for last week). Returns success, results, count, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListTodosRequest(
                user_id=user_id,
                scope=scope,
                query=query,
                limit=limit,
                include_archives=include_archives,
            )
            if hasattr(request, "include_body"):
                request.include_body = include_body
            if hasattr(request, "closed_since_days") and closed_since_days is not None and closed_since_days > 0:
                request.closed_since_days = closed_since_days
            if states:
                request.states.extend(states)
            if tags:
                request.tags.extend(tags)
            response = await self._stub.ListTodos(request)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error", "results": [], "count": 0}
            results = []
            for r in response.results:
                item = {
                    "filename": r.filename,
                    "file_path": r.file_path,
                    "heading": r.heading,
                    "level": r.level,
                    "line_number": r.line_number,
                    "todo_state": r.todo_state,
                    "tags": list(r.tags),
                    "scheduled": r.scheduled or None,
                    "deadline": r.deadline or None,
                    "document_id": r.document_id or None,
                    "preview": r.preview or "",
                    "closed": getattr(r, "closed", None) or None,
                }
                if include_body:
                    item["body"] = getattr(r, "body", "") or ""
                results.append(item)
            return {"success": True, "results": results, "count": response.count, "files_searched": response.files_searched}
        except grpc.RpcError as e:
            logger.error("list_todos failed: %s - %s", e.code(), e.details())
            return {"success": False, "error": e.details() or str(e), "results": [], "count": 0}
        except Exception as e:
            logger.error("list_todos error: %s", e)
            return {"success": False, "error": str(e), "results": [], "count": 0}

    async def create_todo(
        self,
        user_id: str,
        text: str,
        file_path: Optional[str] = None,
        state: str = "TODO",
        tags: Optional[List[str]] = None,
        scheduled: Optional[str] = None,
        deadline: Optional[str] = None,
        priority: Optional[str] = None,
        body: Optional[str] = None,
        heading_level: Optional[int] = None,
        insert_after_line_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a todo. file_path None = inbox. heading_level 1-6 = org stars. insert_after_line_number = insert after this 0-based line (else append)."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.CreateTodoRequest(user_id=user_id, text=text, state=state)
            if file_path:
                request.file_path = file_path
            if tags:
                request.tags.extend(tags)
            if scheduled:
                request.scheduled = scheduled
            if deadline:
                request.deadline = deadline
            if priority:
                request.priority = priority
            if body and body.strip():
                if hasattr(request, "body"):
                    request.body = body.strip()
            if heading_level is not None and hasattr(request, "heading_level"):
                request.heading_level = max(1, min(6, heading_level))
            if insert_after_line_number is not None and hasattr(request, "insert_after_line_number"):
                request.insert_after_line_number = insert_after_line_number
            response = await self._stub.CreateTodo(request)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error"}
            return {"success": True, "file_path": response.file_path, "line_number": response.line_number, "heading": response.heading}
        except grpc.RpcError as e:
            logger.error("create_todo failed: %s - %s", e.code(), e.details())
            return {"success": False, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("create_todo error: %s", e)
            return {"success": False, "error": str(e)}

    async def update_todo(
        self,
        user_id: str,
        file_path: str,
        line_number: int,
        heading_text: Optional[str] = None,
        new_state: Optional[str] = None,
        new_text: Optional[str] = None,
        add_tags: Optional[List[str]] = None,
        remove_tags: Optional[List[str]] = None,
        scheduled: Optional[str] = None,
        deadline: Optional[str] = None,
        priority: Optional[str] = None,
        new_body: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update a todo. Returns success, file_path, line_number, new_line, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.UpdateTodoRequest(user_id=user_id, file_path=file_path, line_number=line_number)
            if heading_text:
                request.heading_text = heading_text
            if new_state:
                request.new_state = new_state
            if new_text:
                request.new_text = new_text
            if add_tags:
                request.add_tags.extend(add_tags)
            if remove_tags:
                request.remove_tags.extend(remove_tags)
            if scheduled:
                request.scheduled = scheduled
            if deadline:
                request.deadline = deadline
            if priority:
                request.priority = priority
            if new_body is not None and (isinstance(new_body, str) and new_body.strip()) and hasattr(request, "new_body"):
                request.new_body = new_body.strip()
            response = await self._stub.UpdateTodo(request)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error"}
            return {"success": True, "file_path": response.file_path, "line_number": response.line_number, "new_line": response.new_line}
        except grpc.RpcError as e:
            logger.error("update_todo failed: %s - %s", e.code(), e.details())
            return {"success": False, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("update_todo error: %s", e)
            return {"success": False, "error": str(e)}

    async def toggle_todo(
        self,
        user_id: str,
        file_path: str,
        line_number: int,
        heading_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Toggle TODO <-> DONE. Returns success, file_path, line_number, new_line, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ToggleTodoRequest(user_id=user_id, file_path=file_path, line_number=line_number)
            if heading_text:
                request.heading_text = heading_text
            response = await self._stub.ToggleTodo(request)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error"}
            return {"success": True, "file_path": response.file_path, "line_number": response.line_number, "new_line": response.new_line}
        except grpc.RpcError as e:
            logger.error("toggle_todo failed: %s - %s", e.code(), e.details())
            return {"success": False, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("toggle_todo error: %s", e)
            return {"success": False, "error": str(e)}

    async def delete_todo(
        self,
        user_id: str,
        file_path: str,
        line_number: int,
        heading_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete a todo line. Returns success, file_path, deleted_line_count, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.DeleteTodoRequest(user_id=user_id, file_path=file_path, line_number=line_number)
            if heading_text:
                request.heading_text = heading_text
            response = await self._stub.DeleteTodo(request)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error"}
            return {"success": True, "file_path": response.file_path, "deleted_line_count": response.deleted_line_count}
        except grpc.RpcError as e:
            logger.error("delete_todo failed: %s - %s", e.code(), e.details())
            return {"success": False, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("delete_todo error: %s", e)
            return {"success": False, "error": str(e)}

    async def archive_done_todos(
        self,
        user_id: str,
        file_path: Optional[str] = None,
        preview_only: bool = False,
        line_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Archive one entry (line_number set) or bulk closed items. file_path None = inbox. preview_only=True returns path/count without writing. Returns success, path, archived_to, archived_count, directive_found, directive_value, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ArchiveDoneTodosRequest(user_id=user_id)
            if file_path:
                request.file_path = file_path
            if hasattr(request, "preview_only"):
                request.preview_only = preview_only
            if line_number is not None and hasattr(request, "line_number"):
                request.line_number = line_number
            response = await self._stub.ArchiveDoneTodos(request)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error"}
            out = {
                "success": True,
                "path": response.path,
                "archived_to": response.archived_to,
                "archived_count": response.archived_count,
            }
            if hasattr(response, "directive_found"):
                out["directive_found"] = response.directive_found
            if hasattr(response, "directive_value"):
                out["directive_value"] = response.directive_value
            return out
        except grpc.RpcError as e:
            logger.error("archive_done_todos failed: %s - %s", e.code(), e.details())
            return {"success": False, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("archive_done_todos error: %s", e)
            return {"success": False, "error": str(e)}

    async def refile_todo(
        self,
        user_id: str,
        source_file: str,
        source_line: int,
        target_file: str,
        target_heading_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Move a todo entry (and its subtree) from one org file to another. Returns success, source_file, target_file, lines_moved, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.RefileTodoRequest(
                user_id=user_id,
                source_file=source_file,
                source_line=source_line,
                target_file=target_file,
            )
            if target_heading_line is not None:
                request.target_heading_line = target_heading_line
            response = await self._stub.RefileTodo(request)
            if not response.success:
                return {
                    "success": False,
                    "source_file": response.source_file,
                    "target_file": response.target_file,
                    "lines_moved": response.lines_moved,
                    "error": response.error or "Unknown error",
                }
            return {
                "success": True,
                "source_file": response.source_file,
                "target_file": response.target_file,
                "lines_moved": response.lines_moved,
            }
        except grpc.RpcError as e:
            logger.error("refile_todo failed: %s - %s", e.code(), e.details())
            return {"success": False, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("refile_todo error: %s", e)
            return {"success": False, "error": str(e)}

    async def discover_refile_targets(self, user_id: str) -> Dict[str, Any]:
        """List all org files and headings available as refile destinations. Returns success, targets, count, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.DiscoverRefileTargetsRequest(user_id=user_id)
            response = await self._stub.DiscoverRefileTargets(request)
            if not response.success:
                return {"success": False, "targets": [], "count": 0, "error": response.error or "Unknown error"}
            targets = []
            for t in response.targets:
                targets.append({
                    "file": t.file,
                    "filename": t.filename,
                    "heading_path": list(t.heading_path),
                    "heading_line": t.heading_line,
                    "display_name": t.display_name,
                    "level": t.level,
                })
            return {"success": True, "targets": targets, "count": len(targets)}
        except grpc.RpcError as e:
            logger.error("discover_refile_targets failed: %s - %s", e.code(), e.details())
            return {"success": False, "targets": [], "count": 0, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("discover_refile_targets error: %s", e)
            return {"success": False, "targets": [], "count": 0, "error": str(e)}

    # ===== RSS Feed Operations =====

    async def add_rss_feed(
        self,
        user_id: str,
        feed_url: str,
        feed_name: str = "",
        category: str = "",
        is_global: bool = False,
    ) -> Dict[str, Any]:
        """Add an RSS feed. Returns dict with success, feed_id, message, or error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.AddRSSFeedRequest(
                user_id=user_id,
                feed_url=feed_url,
                feed_name=feed_name or feed_url,
                category=category,
                is_global=is_global,
            )
            response = await self._stub.AddRSSFeed(request)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error"}
            return {
                "success": True,
                "feed_id": response.feed_id,
                "feed_name": response.feed_name or "",
                "message": response.message or "Feed added",
            }
        except grpc.RpcError as e:
            logger.error("add_rss_feed failed: %s - %s", e.code(), e.details())
            return {"success": False, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("add_rss_feed error: %s", e)
            return {"success": False, "error": str(e)}

    async def list_rss_feeds(
        self,
        user_id: str,
        scope: str = "user",
    ) -> Dict[str, Any]:
        """List RSS feeds. scope: 'user' or 'global'. Returns dict with success, feeds, count, or error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListRSSFeedsRequest(user_id=user_id, scope=scope)
            response = await self._stub.ListRSSFeeds(request)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error", "feeds": [], "count": 0}
            feeds = []
            for f in response.feeds:
                feeds.append({
                    "feed_id": f.feed_id,
                    "feed_name": f.feed_name,
                    "feed_url": f.feed_url,
                    "category": f.category,
                    "is_global": f.is_global,
                    "last_polled": f.last_polled or None,
                    "article_count": f.article_count,
                    "unread_count": int(f.unread_count),
                })
            return {"success": True, "feeds": feeds, "count": response.count}
        except grpc.RpcError as e:
            logger.error("list_rss_feeds failed: %s - %s", e.code(), e.details())
            return {"success": False, "error": e.details() or str(e), "feeds": [], "count": 0}
        except Exception as e:
            logger.error("list_rss_feeds error: %s", e)
            return {"success": False, "error": str(e), "feeds": [], "count": 0}

    async def refresh_rss_feed(
        self,
        user_id: str,
        feed_name: str = "",
        feed_id: str = "",
    ) -> Dict[str, Any]:
        """Refresh an RSS feed by name or ID. Returns dict with success, feed_id, task_id, message, or error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.RefreshRSSFeedRequest(
                user_id=user_id,
                feed_name=feed_name,
                feed_id=feed_id,
            )
            response = await self._stub.RefreshRSSFeed(request)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error"}
            return {
                "success": True,
                "feed_id": response.feed_id,
                "feed_name": response.feed_name or "",
                "task_id": response.task_id or "",
                "message": response.message or "Refresh triggered",
            }
        except grpc.RpcError as e:
            logger.error("refresh_rss_feed failed: %s - %s", e.code(), e.details())
            return {"success": False, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("refresh_rss_feed error: %s", e)
            return {"success": False, "error": str(e)}

    @staticmethod
    def _rss_article_pb_to_dict(a: tool_service_pb2.RSSArticle) -> Dict[str, Any]:
        return {
            "article_id": a.article_id,
            "title": a.title,
            "content": a.content,
            "url": a.url,
            "published_at": a.published_at,
            "feed_id": a.feed_id,
            "feed_name": a.feed_name,
            "is_read": bool(a.is_read),
            "is_starred": bool(a.is_starred),
            "is_imported": bool(a.is_imported),
            "created_at": a.created_at,
        }

    async def get_rss_articles(
        self,
        feed_id: str,
        user_id: str = "system",
        limit: int = 20,
        unread_only: bool = False,
        starred_only: bool = False,
    ) -> Dict[str, Any]:
        """Retrieve articles from a specific RSS feed. Returns dict with articles list."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.RSSArticlesRequest(
                feed_id=feed_id,
                user_id=user_id,
                limit=limit,
                unread_only=unread_only,
                starred_only=starred_only,
            )
            response = await self._stub.GetRSSArticles(request)
            articles = [
                self._rss_article_pb_to_dict(a) for a in response.articles
            ]
            return {"articles": articles, "count": len(articles)}
        except grpc.RpcError as e:
            logger.error("get_rss_articles failed: %s - %s", e.code(), e.details())
            return {"articles": [], "count": 0, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("get_rss_articles error: %s", e)
            return {"articles": [], "count": 0, "error": str(e)}

    async def list_starred_rss_articles(
        self,
        user_id: str = "system",
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List starred RSS articles across all feeds for the user. Returns dict with articles list."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListStarredRSSArticlesRequest(
                user_id=user_id,
                limit=int(limit),
                offset=int(offset),
            )
            response = await self._stub.ListStarredRSSArticles(request)
            articles = [
                self._rss_article_pb_to_dict(a) for a in response.articles
            ]
            return {
                "articles": articles,
                "count": len(articles),
                "limit": limit,
                "offset": offset,
            }
        except grpc.RpcError as e:
            logger.error(
                "list_starred_rss_articles failed: %s - %s", e.code(), e.details()
            )
            return {
                "articles": [],
                "count": 0,
                "limit": limit,
                "offset": offset,
                "error": e.details() or str(e),
            }
        except Exception as e:
            logger.error("list_starred_rss_articles error: %s", e)
            return {
                "articles": [],
                "count": 0,
                "limit": limit,
                "offset": offset,
                "error": str(e),
            }

    async def search_rss(
        self,
        query: str,
        user_id: str = "system",
        limit: int = 20,
        unread_only: bool = False,
        starred_only: bool = False,
    ) -> Dict[str, Any]:
        """Full-text search across RSS article content and titles. Returns dict with articles list."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.RSSSearchRequest(
                user_id=user_id,
                query=query,
                limit=limit,
                unread_only=unread_only,
                starred_only=starred_only,
            )
            response = await self._stub.SearchRSSFeeds(request)
            articles = [
                self._rss_article_pb_to_dict(a) for a in response.articles
            ]
            return {"articles": articles, "count": len(articles), "query_used": query}
        except grpc.RpcError as e:
            logger.error("search_rss failed: %s - %s", e.code(), e.details())
            return {"articles": [], "count": 0, "query_used": query, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("search_rss error: %s", e)
            return {"articles": [], "count": 0, "query_used": query, "error": str(e)}

    async def delete_rss_feed(
        self,
        user_id: str,
        feed_name: str = "",
        feed_id: str = "",
    ) -> Dict[str, Any]:
        """Delete an RSS feed by name or ID."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.DeleteRSSFeedRequest(
                user_id=user_id,
                feed_name=feed_name,
                feed_id=feed_id,
            )
            response = await self._stub.DeleteRSSFeed(request)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error"}
            return {
                "success": True,
                "feed_id": response.feed_id,
                "message": response.message or "Feed deleted",
            }
        except grpc.RpcError as e:
            logger.error("delete_rss_feed failed: %s - %s", e.code(), e.details())
            return {"success": False, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("delete_rss_feed error: %s", e)
            return {"success": False, "error": str(e)}

    async def mark_article_read(
        self,
        user_id: str,
        article_id: str,
    ) -> Dict[str, Any]:
        """Mark an RSS article as read."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.MarkArticleReadRequest(
                user_id=user_id,
                article_id=article_id,
            )
            response = await self._stub.MarkArticleRead(request)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error"}
            return {"success": True, "message": response.message or "Marked read"}
        except grpc.RpcError as e:
            logger.error("mark_article_read failed: %s - %s", e.code(), e.details())
            return {"success": False, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("mark_article_read error: %s", e)
            return {"success": False, "error": str(e)}

    async def mark_article_unread(
        self,
        user_id: str,
        article_id: str,
    ) -> Dict[str, Any]:
        """Mark an RSS article as unread."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.MarkArticleUnreadRequest(
                user_id=user_id,
                article_id=article_id,
            )
            response = await self._stub.MarkArticleUnread(request)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error"}
            return {"success": True, "message": response.message or "Marked unread"}
        except grpc.RpcError as e:
            logger.error("mark_article_unread failed: %s - %s", e.code(), e.details())
            return {"success": False, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("mark_article_unread error: %s", e)
            return {"success": False, "error": str(e)}

    async def set_article_starred(
        self,
        user_id: str,
        article_id: str,
        starred: bool,
    ) -> Dict[str, Any]:
        """Set RSS article starred flag."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.SetArticleStarredRequest(
                user_id=user_id,
                article_id=article_id,
                starred=starred,
            )
            response = await self._stub.SetArticleStarred(request)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error"}
            return {"success": True, "message": response.message or "Updated"}
        except grpc.RpcError as e:
            logger.error("set_article_starred failed: %s - %s", e.code(), e.details())
            return {"success": False, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("set_article_starred error: %s", e)
            return {"success": False, "error": str(e)}

    async def get_unread_counts(
        self,
        user_id: str,
    ) -> Dict[str, Any]:
        """Per-feed unread counts for the user."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetUnreadCountsRequest(user_id=user_id)
            response = await self._stub.GetUnreadCounts(request)
            if not response.success:
                return {"success": False, "counts": {}, "error": response.error or "Unknown error"}
            counts = {c.feed_id: int(c.count) for c in response.counts}
            return {"success": True, "counts": counts}
        except grpc.RpcError as e:
            logger.error("get_unread_counts failed: %s - %s", e.code(), e.details())
            return {"success": False, "counts": {}, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("get_unread_counts error: %s", e)
            return {"success": False, "counts": {}, "error": str(e)}

    async def toggle_feed_active(
        self,
        user_id: str,
        feed_id: str,
        is_active: bool,
    ) -> Dict[str, Any]:
        """Enable or disable RSS feed polling."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ToggleFeedActiveRequest(
                user_id=user_id,
                feed_id=feed_id,
                is_active=is_active,
            )
            response = await self._stub.ToggleFeedActive(request)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error"}
            return {
                "success": True,
                "feed_id": response.feed_id,
                "is_active": response.is_active,
                "message": response.message or "Updated",
            }
        except grpc.RpcError as e:
            logger.error("toggle_feed_active failed: %s - %s", e.code(), e.details())
            return {"success": False, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("toggle_feed_active error: %s", e)
            return {"success": False, "error": str(e)}

    # ===== Data Workspace Operations =====
    
    async def list_data_workspaces(
        self,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        List all data workspaces for a user
        
        Args:
            user_id: User ID for access control
            
        Returns:
            Dict with 'workspaces' (list) and 'total_count'
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.ListDataWorkspacesRequest(
                user_id=user_id
            )
            
            response = await self._stub.ListDataWorkspaces(request)
            
            # Convert proto response to dict
            workspaces = []
            for ws in response.workspaces:
                workspaces.append({
                    'workspace_id': ws.workspace_id,
                    'name': ws.name,
                    'description': ws.description,
                    'icon': ws.icon,
                    'color': ws.color,
                    'is_pinned': ws.is_pinned
                })
            
            return {
                'workspaces': workspaces,
                'total_count': response.total_count
            }
            
        except grpc.RpcError as e:
            logger.error(f"List data workspaces failed: {e.code()} - {e.details()}")
            return {
                'workspaces': [],
                'total_count': 0,
                'error': str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error listing data workspaces: {e}")
            return {
                'workspaces': [],
                'total_count': 0,
                'error': str(e)
            }

    async def list_code_workspaces(self, user_id: str = "system") -> Dict[str, Any]:
        """List saved local code workspaces for the user."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListCodeWorkspacesRequest(user_id=user_id)
            response = await self._stub.ListCodeWorkspaces(request)
            workspaces = []
            for ws in response.workspaces:
                workspaces.append(
                    {
                        "workspace_id": ws.workspace_id,
                        "name": ws.name,
                        "device_id": ws.device_id,
                        "workspace_path": ws.workspace_path,
                        "last_git_branch": ws.last_git_branch,
                        "updated_at": ws.updated_at,
                    }
                )
            return {"workspaces": workspaces, "total": response.total}
        except grpc.RpcError as e:
            logger.error("ListCodeWorkspaces failed: %s - %s", e.code(), e.details())
            return {"workspaces": [], "total": 0, "error": str(e)}
        except Exception as e:
            logger.error("list_code_workspaces error: %s", e)
            return {"workspaces": [], "total": 0, "error": str(e)}

    async def get_code_workspace(self, workspace_id: str, user_id: str = "system") -> Dict[str, Any]:
        """Fetch one code workspace row by id."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetCodeWorkspaceRequest(
                user_id=user_id, workspace_id=workspace_id
            )
            response = await self._stub.GetCodeWorkspace(request)
            if not response.success:
                return {
                    "success": False,
                    "error": response.error if response.HasField("error") else "unknown",
                }
            w = response.workspace
            return {
                "success": True,
                "workspace": {
                    "workspace_id": w.workspace_id,
                    "user_id": w.user_id,
                    "name": w.name,
                    "device_id": w.device_id,
                    "device_name": w.device_name,
                    "workspace_path": w.workspace_path,
                    "last_git_branch": w.last_git_branch,
                    "last_file_tree_json": w.last_file_tree_json,
                    "settings_json": w.settings_json,
                    "conversation_id": w.conversation_id,
                    "created_at": w.created_at,
                    "updated_at": w.updated_at,
                },
            }
        except grpc.RpcError as e:
            logger.error("GetCodeWorkspace failed: %s - %s", e.code(), e.details())
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error("get_code_workspace error: %s", e)
            return {"success": False, "error": str(e)}

    async def upsert_code_workspace_chunks(
        self,
        user_id: str,
        workspace_id: str,
        replace_workspace: bool,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Persist indexed code chunks and embed into vector store."""
        try:
            await self._ensure_connected()
            pb_chunks = []
            for c in chunks or []:
                pb_chunks.append(
                    tool_service_pb2.CodeWorkspaceChunkProto(
                        file_path=str(c.get("file_path", "")),
                        chunk_index=int(c.get("chunk_index", 0)),
                        start_line=int(c.get("start_line", 1)),
                        end_line=int(c.get("end_line", 1)),
                        content=str(c.get("content", "")),
                        language=str(c.get("language", "")),
                        git_sha=str(c.get("git_sha", "")),
                    )
                )
            request = tool_service_pb2.UpsertCodeWorkspaceChunksRequest(
                user_id=user_id,
                workspace_id=workspace_id,
                replace_workspace=replace_workspace,
                chunks=pb_chunks,
            )
            response = await self._stub.UpsertCodeWorkspaceChunks(request)
            return {
                "success": response.success,
                "inserted": response.inserted,
                "embedded": response.embedded,
                "error": response.error if response.HasField("error") else None,
            }
        except grpc.RpcError as e:
            logger.error("UpsertCodeWorkspaceChunks failed: %s - %s", e.code(), e.details())
            return {"success": False, "inserted": 0, "embedded": 0, "error": str(e)}
        except Exception as e:
            logger.error("upsert_code_workspace_chunks error: %s", e)
            return {"success": False, "inserted": 0, "embedded": 0, "error": str(e)}

    async def code_semantic_search(
        self,
        user_id: str,
        workspace_id: str,
        query: str,
        limit: int = 20,
        file_glob: str = "",
    ) -> Dict[str, Any]:
        """Hybrid semantic + FTS search over indexed code chunks."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.CodeSemanticSearchRequest(
                user_id=user_id,
                workspace_id=workspace_id,
                query=query,
                limit=int(limit),
                file_glob=file_glob or "",
            )
            response = await self._stub.CodeSemanticSearch(request)
            if not response.success:
                return {
                    "success": False,
                    "hits": [],
                    "error": response.error if response.HasField("error") else "search_failed",
                }
            hits = []
            for h in response.hits:
                hits.append(
                    {
                        "chunk_id": h.chunk_id,
                        "file_path": h.file_path,
                        "start_line": h.start_line,
                        "end_line": h.end_line,
                        "snippet": h.snippet,
                        "score": h.score,
                    }
                )
            return {"success": True, "hits": hits}
        except grpc.RpcError as e:
            logger.error("CodeSemanticSearch failed: %s - %s", e.code(), e.details())
            return {"success": False, "hits": [], "error": str(e)}
        except Exception as e:
            logger.error("code_semantic_search error: %s", e)
            return {"success": False, "hits": [], "error": str(e)}
    
    async def get_workspace_schema(
        self,
        workspace_id: str,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Get complete schema for a workspace (all tables and columns)
        
        Args:
            workspace_id: Workspace ID to get schema for
            user_id: User ID for access control
            
        Returns:
            Dict with 'workspace_id', 'tables' (list), and 'total_tables'
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.GetWorkspaceSchemaRequest(
                workspace_id=workspace_id,
                user_id=user_id
            )
            
            response = await self._stub.GetWorkspaceSchema(request)
            
            # Convert proto response to dict (include description and metadata for agents)
            tables = []
            for table in response.tables:
                columns = []
                for col in table.columns:
                    cref = getattr(col, 'column_ref_json', None) or ''
                    columns.append({
                        'name': col.name,
                        'type': col.type,
                        'is_nullable': col.is_nullable,
                        'description': getattr(col, 'description', None) or '',
                        'column_ref_json': cref,
                    })
                meta = getattr(table, 'metadata_json', None) or ''
                try:
                    metadata_json = json.loads(meta) if meta else {}
                except (json.JSONDecodeError, TypeError):
                    metadata_json = {}
                tables.append({
                    'table_id': table.table_id,
                    'name': table.name,
                    'description': table.description,
                    'database_id': table.database_id,
                    'database_name': table.database_name,
                    'columns': columns,
                    'row_count': table.row_count,
                    'metadata_json': metadata_json
                })
            
            result = {
                'workspace_id': response.workspace_id,
                'tables': tables,
                'total_tables': response.total_tables
            }
            
            if response.HasField("error") and response.error:
                result['error'] = response.error
            
            return result
            
        except grpc.RpcError as e:
            logger.error(f"Get workspace schema failed: {e.code()} - {e.details()}")
            return {
                'workspace_id': workspace_id,
                'tables': [],
                'total_tables': 0,
                'error': str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error getting workspace schema: {e}")
            return {
                'workspace_id': workspace_id,
                'tables': [],
                'total_tables': 0,
                'error': str(e)
            }

    async def resolve_workspace_link(
        self,
        ref_payload: Any,
        user_id: str = "system",
    ) -> Dict[str, Any]:
        """Resolve a _bastion_ref cell value to current label and preview."""
        try:
            await self._ensure_connected()
            ref_json = ref_payload if isinstance(ref_payload, str) else json.dumps(ref_payload)
            request = tool_service_pb2.ResolveWorkspaceLinkRequest(
                user_id=user_id,
                ref_json=ref_json or "{}",
            )
            response = await self._stub.ResolveWorkspaceLink(request)
            preview = {}
            if response.preview_json:
                try:
                    preview = json.loads(response.preview_json)
                except (json.JSONDecodeError, TypeError):
                    preview = {}
            return {
                "success": response.success,
                "error": getattr(response, "error", None) or "",
                "label": response.label or "",
                "preview": preview,
                "row_found": response.row_found,
                "table_id": response.table_id or "",
                "row_id": response.row_id or "",
            }
        except grpc.RpcError as e:
            logger.error(f"Resolve workspace link failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "error": str(e),
                "label": "",
                "preview": {},
                "row_found": False,
                "table_id": "",
                "row_id": "",
            }
    
    async def query_data_workspace(
        self,
        workspace_id: str,
        query: str,
        query_type: str = "natural_language",
        user_id: str = "system",
        limit: int = 100,
        params: Optional[List[Any]] = None,
        read_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a query against a data workspace (SQL or natural language)
        
        Args:
            workspace_id: Workspace ID to query
            query: SQL query or natural language query
            query_type: "sql" or "natural_language"
            user_id: User ID for access control
            limit: Maximum rows to return (default: 100)
            params: Optional list of values for $1, $2, ... (SQL only)
            
        Returns:
            Dict with 'success', 'column_names', 'results', 'result_count', 
            'execution_time_ms', 'generated_sql', 'error_message',
            'rows_affected', 'returning_rows'
        """
        try:
            await self._ensure_connected()
            
            import json
            params_json = json.dumps(params) if params else ""
            request = tool_service_pb2.QueryDataWorkspaceRequest(
                workspace_id=workspace_id,
                query=query,
                query_type=query_type,
                user_id=user_id,
                limit=limit,
                params_json=params_json,
                read_only=read_only,
            )
            
            response = await self._stub.QueryDataWorkspace(request)

            results = []
            if getattr(response, "has_arrow_data", False) and response.arrow_results:
                from orchestrator.utils.arrow_decode import decode_query_arrow

                try:
                    results = decode_query_arrow(bytes(response.arrow_results))
                except Exception as ex:
                    logger.warning("Failed to decode Arrow query results: %s", ex)
                    results = []
            elif response.results_json:
                try:
                    results = json.loads(response.results_json)
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse results JSON: %s",
                        response.results_json[:100],
                    )
                    results = []
            
            returning_rows = []
            if getattr(response, 'returning_rows_json', None) and response.returning_rows_json:
                try:
                    returning_rows = json.loads(response.returning_rows_json)
                except json.JSONDecodeError:
                    pass
            
            result = {
                'success': response.success,
                'column_names': list(response.column_names),
                'results': results,
                'result_count': response.result_count,
                'execution_time_ms': response.execution_time_ms,
                'generated_sql': response.generated_sql,
                'rows_affected': getattr(response, 'rows_affected', 0) or 0,
                'returning_rows': returning_rows
            }
            
            if response.HasField("error_message") and response.error_message:
                result['error_message'] = response.error_message
            
            return result
            
        except grpc.RpcError as e:
            logger.error(f"Query data workspace failed: {e.code()} - {e.details()}")
            return {
                'success': False,
                'column_names': [],
                'results': [],
                'result_count': 0,
                'execution_time_ms': 0,
                'generated_sql': '',
                'error_message': str(e),
                'rows_affected': 0,
                'returning_rows': []
            }
        except Exception as e:
            logger.error(f"Unexpected error querying data workspace: {e}")
            return {
                'success': False,
                'column_names': [],
                'results': [],
                'result_count': 0,
                'execution_time_ms': 0,
                'generated_sql': '',
                'error_message': str(e),
                'rows_affected': 0,
                'returning_rows': []
            }

    async def create_data_workspace_table(
        self,
        workspace_id: str,
        database_id: str,
        table_name: str,
        user_id: str = "system",
        description: str = "",
        columns: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a table in a data workspace (structured)."""
        try:
            await self._ensure_connected()
            req = tool_service_pb2.CreateDataWorkspaceTableRequest(
                workspace_id=workspace_id,
                database_id=database_id,
                table_name=table_name,
                description=description or "",
                user_id=user_id,
                columns_json=json.dumps(columns or []),
                metadata_json=json.dumps(metadata or {}),
            )
            resp = await self._stub.CreateDataWorkspaceTable(req)
            table = {}
            if resp.table_json:
                try:
                    table = json.loads(resp.table_json)
                except json.JSONDecodeError:
                    table = {}
            return {
                "success": bool(resp.success),
                "table_id": resp.table_id,
                "table": table,
                "error_message": resp.error_message or "",
            }
        except grpc.RpcError as e:
            return {"success": False, "table_id": "", "table": {}, "error_message": e.details() or str(e)}
        except Exception as e:
            return {"success": False, "table_id": "", "table": {}, "error_message": str(e)}

    async def insert_data_workspace_rows(
        self,
        workspace_id: str,
        table_id: str,
        rows: List[Dict[str, Any]],
        user_id: str = "system",
    ) -> Dict[str, Any]:
        """Insert rows into a table (structured)."""
        try:
            await self._ensure_connected()
            req = tool_service_pb2.InsertDataWorkspaceRowsRequest(
                workspace_id=workspace_id,
                table_id=table_id,
                user_id=user_id,
                rows_json=json.dumps(rows or []),
            )
            resp = await self._stub.InsertDataWorkspaceRows(req)
            ids = []
            if resp.inserted_row_ids_json:
                try:
                    ids = json.loads(resp.inserted_row_ids_json)
                except json.JSONDecodeError:
                    ids = []
            return {
                "success": bool(resp.success),
                "rows_inserted": int(resp.rows_inserted),
                "inserted_row_ids": ids,
                "error_message": resp.error_message or "",
            }
        except grpc.RpcError as e:
            return {"success": False, "rows_inserted": 0, "inserted_row_ids": [], "error_message": e.details() or str(e)}
        except Exception as e:
            return {"success": False, "rows_inserted": 0, "inserted_row_ids": [], "error_message": str(e)}

    async def update_data_workspace_rows(
        self,
        workspace_id: str,
        table_id: str,
        updates: List[Dict[str, Any]],
        user_id: str = "system",
    ) -> Dict[str, Any]:
        """Update rows in a table (structured)."""
        try:
            await self._ensure_connected()
            req = tool_service_pb2.UpdateDataWorkspaceRowsRequest(
                workspace_id=workspace_id,
                table_id=table_id,
                user_id=user_id,
                updates_json=json.dumps(updates or []),
            )
            resp = await self._stub.UpdateDataWorkspaceRows(req)
            ids = []
            if resp.updated_row_ids_json:
                try:
                    ids = json.loads(resp.updated_row_ids_json)
                except json.JSONDecodeError:
                    ids = []
            return {
                "success": bool(resp.success),
                "rows_updated": int(resp.rows_updated),
                "updated_row_ids": ids,
                "error_message": resp.error_message or "",
            }
        except grpc.RpcError as e:
            return {"success": False, "rows_updated": 0, "updated_row_ids": [], "error_message": e.details() or str(e)}
        except Exception as e:
            return {"success": False, "rows_updated": 0, "updated_row_ids": [], "error_message": str(e)}

    async def delete_data_workspace_rows(
        self,
        workspace_id: str,
        table_id: str,
        row_ids: List[str],
        user_id: str = "system",
    ) -> Dict[str, Any]:
        """Delete rows from a table (structured)."""
        try:
            await self._ensure_connected()
            req = tool_service_pb2.DeleteDataWorkspaceRowsRequest(
                workspace_id=workspace_id,
                table_id=table_id,
                user_id=user_id,
                row_ids_json=json.dumps(row_ids or []),
            )
            resp = await self._stub.DeleteDataWorkspaceRows(req)
            ids = []
            if resp.deleted_row_ids_json:
                try:
                    ids = json.loads(resp.deleted_row_ids_json)
                except json.JSONDecodeError:
                    ids = []
            return {
                "success": bool(resp.success),
                "rows_deleted": int(resp.rows_deleted),
                "deleted_row_ids": ids,
                "error_message": resp.error_message or "",
            }
        except grpc.RpcError as e:
            return {"success": False, "rows_deleted": 0, "deleted_row_ids": [], "error_message": e.details() or str(e)}
        except Exception as e:
            return {"success": False, "rows_deleted": 0, "deleted_row_ids": [], "error_message": str(e)}

    # ===== Navigation Operations (locations and routes) =====

    async def create_location(
        self,
        user_id: str,
        name: str,
        address: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        notes: Optional[str] = None,
        is_global: bool = False,
        metadata: Optional[dict] = None,
        user_role: str = "user",
    ) -> Dict[str, Any]:
        """Create a saved location (geocodes address if needed)."""
        try:
            await self._ensure_connected()
            req = tool_service_pb2.CreateLocationRequest(
                user_id=user_id,
                name=name,
                address=address,
                notes=notes or "",
                is_global=is_global,
                user_role=user_role,
            )
            if latitude is not None:
                req.latitude = latitude
            if longitude is not None:
                req.longitude = longitude
            if metadata is not None:
                req.metadata_json = json.dumps(metadata)
            response = await self._stub.CreateLocation(req)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error"}
            return {
                "success": True,
                "location_id": response.location_id,
                "user_id": response.user_id,
                "name": response.name,
                "address": response.address or None,
                "latitude": response.latitude,
                "longitude": response.longitude,
                "notes": response.notes or None,
                "is_global": response.is_global,
                "created_at": response.created_at or None,
                "updated_at": response.updated_at or None,
            }
        except grpc.RpcError as e:
            logger.error(f"create_location failed: {e.code()} - {e.details()}")
            return {"success": False, "error": e.details() or str(e)}
        except Exception as e:
            logger.error(f"create_location error: {e}")
            return {"success": False, "error": str(e)}

    async def list_locations(
        self,
        user_id: str,
        user_role: str = "user",
    ) -> Dict[str, Any]:
        """List all locations accessible to the user."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListLocationsRequest(user_id=user_id, user_role=user_role)
            response = await self._stub.ListLocations(request)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error", "locations": [], "total": 0}
            locations = []
            for loc in response.locations:
                locations.append({
                    "location_id": loc.location_id,
                    "user_id": loc.user_id,
                    "name": loc.name,
                    "address": loc.address or None,
                    "latitude": loc.latitude,
                    "longitude": loc.longitude,
                    "notes": loc.notes or None,
                    "is_global": loc.is_global,
                    "created_at": loc.created_at or None,
                    "updated_at": loc.updated_at or None,
                })
            return {"success": True, "locations": locations, "total": response.total}
        except grpc.RpcError as e:
            logger.error(f"list_locations failed: {e.code()} - {e.details()}")
            return {"success": False, "error": e.details() or str(e), "locations": [], "total": 0}
        except Exception as e:
            logger.error(f"list_locations error: {e}")
            return {"success": False, "error": str(e), "locations": [], "total": 0}

    async def delete_location(
        self,
        user_id: str,
        location_id: str,
        user_role: str = "user",
    ) -> Dict[str, Any]:
        """Delete a location by ID."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.DeleteLocationRequest(
                user_id=user_id,
                location_id=location_id,
                user_role=user_role,
            )
            response = await self._stub.DeleteLocation(request)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error"}
            return {"success": True, "message": response.message or "Location deleted"}
        except grpc.RpcError as e:
            logger.error(f"delete_location failed: {e.code()} - {e.details()}")
            return {"success": False, "error": e.details() or str(e)}
        except Exception as e:
            logger.error(f"delete_location error: {e}")
            return {"success": False, "error": str(e)}

    async def compute_route(
        self,
        user_id: str,
        from_location_id: Optional[str] = None,
        to_location_id: Optional[str] = None,
        coordinates: Optional[str] = None,
        profile: str = "driving",
        user_role: str = "user",
    ) -> Dict[str, Any]:
        """Compute route between two points (location IDs or coordinate string)."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ComputeRouteRequest(
                user_id=user_id,
                profile=profile,
                user_role=user_role,
            )
            if from_location_id:
                request.from_location_id = from_location_id
            if to_location_id:
                request.to_location_id = to_location_id
            if coordinates:
                request.coordinates = coordinates
            response = await self._stub.ComputeRoute(request)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error"}
            geometry = json.loads(response.geometry_json) if response.geometry_json else {}
            legs = json.loads(response.legs_json) if response.legs_json else []
            waypoints = json.loads(response.waypoints_json) if response.waypoints_json else []
            return {
                "success": True,
                "geometry": geometry,
                "legs": legs,
                "distance": response.distance,
                "duration": response.duration,
                "waypoints": waypoints,
            }
        except grpc.RpcError as e:
            logger.error(f"compute_route failed: {e.code()} - {e.details()}")
            return {"success": False, "error": e.details() or str(e)}
        except Exception as e:
            logger.error(f"compute_route error: {e}")
            return {"success": False, "error": str(e)}

    async def save_route(
        self,
        user_id: str,
        name: str,
        waypoints: List[dict],
        geometry: dict,
        steps: List[dict],
        distance_meters: float,
        duration_seconds: float,
        profile: str = "driving",
        user_role: str = "user",
    ) -> Dict[str, Any]:
        """Save a computed route."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.NavSaveRouteRequest(
                user_id=user_id,
                name=name,
                waypoints_json=json.dumps(waypoints),
                geometry_json=json.dumps(geometry),
                steps_json=json.dumps(steps),
                distance_meters=distance_meters,
                duration_seconds=duration_seconds,
                profile=profile,
                user_role=user_role,
            )
            response = await self._stub.SaveRoute(request)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error"}
            return {
                "success": True,
                "route_id": response.route_id,
                "user_id": response.user_id,
                "name": response.name,
                "waypoints": json.loads(response.waypoints_json) if response.waypoints_json else [],
                "geometry": json.loads(response.geometry_json) if response.geometry_json else {},
                "steps": json.loads(response.steps_json) if response.steps_json else [],
                "distance_meters": response.distance_meters,
                "duration_seconds": response.duration_seconds,
                "profile": response.profile,
                "created_at": response.created_at or None,
                "updated_at": response.updated_at or None,
            }
        except grpc.RpcError as e:
            logger.error(f"save_route failed: {e.code()} - {e.details()}")
            return {"success": False, "error": e.details() or str(e)}
        except Exception as e:
            logger.error(f"save_route error: {e}")
            return {"success": False, "error": str(e)}

    async def list_saved_routes(
        self,
        user_id: str,
        user_role: str = "user",
    ) -> Dict[str, Any]:
        """List saved routes for the user."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListSavedRoutesRequest(user_id=user_id, user_role=user_role)
            response = await self._stub.ListSavedRoutes(request)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error", "routes": [], "total": 0}
            routes = []
            for r in response.routes:
                routes.append({
                    "route_id": r.route_id,
                    "user_id": r.user_id,
                    "name": r.name,
                    "waypoints": json.loads(r.waypoints_json) if r.waypoints_json else [],
                    "geometry": json.loads(r.geometry_json) if r.geometry_json else {},
                    "steps": json.loads(r.steps_json) if r.steps_json else [],
                    "distance_meters": r.distance_meters,
                    "duration_seconds": r.duration_seconds,
                    "profile": r.profile,
                    "created_at": r.created_at or None,
                    "updated_at": r.updated_at or None,
                })
            return {"success": True, "routes": routes, "total": response.total}
        except grpc.RpcError as e:
            logger.error(f"list_saved_routes failed: {e.code()} - {e.details()}")
            return {"success": False, "error": e.details() or str(e), "routes": [], "total": 0}
        except Exception as e:
            logger.error(f"list_saved_routes error: {e}")
            return {"success": False, "error": str(e), "routes": [], "total": 0}

    async def delete_saved_route(
        self,
        user_id: str,
        route_id: str,
        user_role: str = "user",
    ) -> Dict[str, Any]:
        """Delete a saved route."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.DeleteSavedRouteRequest(
                user_id=user_id,
                route_id=route_id,
                user_role=user_role,
            )
            response = await self._stub.DeleteSavedRoute(request)
            if not response.success:
                return {"success": False, "error": response.error or "Unknown error"}
            return {"success": True, "message": response.message or "Route deleted"}
        except grpc.RpcError as e:
            logger.error(f"delete_saved_route failed: {e.code()} - {e.details()}")
            return {"success": False, "error": e.details() or str(e)}
        except Exception as e:
            logger.error(f"delete_saved_route error: {e}")
            return {"success": False, "error": str(e)}

    # ===== Email operations (via backend tool service -> connections-service) =====

    async def get_emails(
        self,
        user_id: str,
        folder: str = "inbox",
        top: int = 10,
        skip: int = 0,
        unread_only: bool = False,
        connection_id: Optional[int] = None,
    ) -> str:
        """Get emails for user. Returns formatted string for LLM."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetEmailsRequest(
                user_id=user_id,
                folder=folder,
                top=top,
                skip=skip,
                unread_only=unread_only,
                connection_id=connection_id or 0,
            )
            response = await self._stub.GetEmails(request)
            if not response.success:
                return response.error or "Failed to get emails"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"get_emails failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"get_emails error: {e}")
            return f"Error: {e}"

    async def search_emails(
        self,
        user_id: str,
        query: str,
        top: int = 20,
        from_address: str = "",
        connection_id: Optional[int] = None,
    ) -> str:
        """Search emails. Returns formatted string for LLM."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.SearchEmailsRequest(
                user_id=user_id,
                query=query,
                top=top,
                from_address=from_address or "",
                connection_id=connection_id or 0,
            )
            response = await self._stub.SearchEmails(request)
            if not response.success:
                return response.error or "Search failed"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"search_emails failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"search_emails error: {e}")
            return f"Error: {e}"

    async def get_email_thread(
        self,
        user_id: str,
        conversation_id: str,
        connection_id: Optional[int] = None,
    ) -> str:
        """Get full email thread. Returns formatted string for LLM."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetEmailThreadRequest(
                user_id=user_id,
                conversation_id=conversation_id,
                connection_id=connection_id or 0,
            )
            response = await self._stub.GetEmailThread(request)
            if not response.success:
                return response.error or "Failed to get thread"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"get_email_thread failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"get_email_thread error: {e}")
            return f"Error: {e}"

    async def send_email(
        self,
        user_id: str,
        to: List[str],
        subject: str,
        body: str,
        cc: List[str] = None,
        from_source: str = "user",
        connection_id: Optional[int] = None,
        body_is_html: bool = False,
    ) -> str:
        """Send email. from_source: 'system' = Bastion SMTP, 'user' = user's email connection. Returns success or error message."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.SendEmailRequest(
                user_id=user_id,
                to=to,
                subject=subject,
                body=body,
                cc=cc or [],
                from_source=(from_source or "user").strip().lower() or "user",
                connection_id=connection_id or 0,
                body_is_html=body_is_html,
            )
            response = await self._stub.SendEmail(request)
            if not response.success:
                return response.error or "Send failed"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"send_email failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"send_email error: {e}")
            return f"Error: {e}"

    async def reply_to_email(
        self,
        user_id: str,
        message_id: str,
        body: str,
        reply_all: bool = False,
        connection_id: Optional[int] = None,
    ) -> str:
        """Reply to an email. Returns success or error message."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ReplyToEmailRequest(
                user_id=user_id,
                message_id=message_id,
                body=body,
                reply_all=reply_all,
                connection_id=connection_id or 0,
            )
            response = await self._stub.ReplyToEmail(request)
            if not response.success:
                return response.error or "Reply failed"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"reply_to_email failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"reply_to_email error: {e}")
            return f"Error: {e}"

    async def get_email_folders(
        self,
        user_id: str,
        connection_id: Optional[int] = None,
    ) -> str:
        """List email folders. Returns formatted string for LLM."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetEmailFoldersRequest(
                user_id=user_id,
                connection_id=connection_id or 0,
            )
            response = await self._stub.GetEmailFolders(request)
            if not response.success:
                return response.error or "Failed to get folders"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"get_email_folders failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"get_email_folders error: {e}")
            return f"Error: {e}"

    async def get_email_statistics(
        self,
        user_id: str,
        connection_id: Optional[int] = None,
    ) -> str:
        """Get email statistics (inbox total/unread). Returns formatted string."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetEmailStatisticsRequest(
                user_id=user_id,
                connection_id=connection_id or 0,
            )
            response = await self._stub.GetEmailStatistics(request)
            if not response.success:
                return response.error or "Failed to get statistics"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"get_email_statistics failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"get_email_statistics error: {e}")
            return f"Error: {e}"

    async def mark_email_read(
        self,
        user_id: str,
        message_id: str,
    ) -> str:
        """Mark an email as read. Returns success or error message."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.MarkEmailReadRequest(
                user_id=user_id,
                message_id=message_id,
            )
            response = await self._stub.MarkEmailRead(request)
            if not response.success:
                return response.error or "Failed to mark as read"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"mark_email_read failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"mark_email_read error: {e}")
            return f"Error: {e}"

    async def get_email_by_id(
        self,
        user_id: str,
        message_id: str,
        connection_id: Optional[int] = None,
    ) -> str:
        """Get a single email by message ID (full content). Returns formatted string for LLM."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetEmailByIdRequest(
                user_id=user_id,
                message_id=message_id,
                connection_id=connection_id or 0,
            )
            response = await self._stub.GetEmailById(request)
            if not response.success:
                return response.error or "Failed to get email"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"get_email_by_id failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"get_email_by_id error: {e}")
            return f"Error: {e}"

    async def move_email(
        self,
        user_id: str,
        message_id: str,
        destination_folder_id: str,
        connection_id: Optional[int] = None,
    ) -> str:
        """Move an email to a different folder. Returns success or error message."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.MoveEmailRequest(
                user_id=user_id,
                message_id=message_id,
                destination_folder_id=destination_folder_id,
                connection_id=connection_id or 0,
            )
            response = await self._stub.MoveEmail(request)
            if not response.success:
                return response.error or "Failed to move email"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"move_email failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"move_email error: {e}")
            return f"Error: {e}"

    async def update_email(
        self,
        user_id: str,
        message_id: str,
        is_read: Optional[bool] = None,
        importance: Optional[str] = None,
        connection_id: Optional[int] = None,
    ) -> str:
        """Update an email (mark read/unread, set importance). Returns success or error message."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.UpdateEmailRequest(
                user_id=user_id,
                message_id=message_id,
                connection_id=connection_id or 0,
            )
            if is_read is not None:
                request.is_read = is_read
            if importance is not None:
                request.importance = importance
            response = await self._stub.UpdateEmail(request)
            if not response.success:
                return response.error or "Failed to update email"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"update_email failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"update_email error: {e}")
            return f"Error: {e}"

    async def create_draft(
        self,
        user_id: str,
        to: List[str],
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        connection_id: Optional[int] = None,
    ) -> str:
        """Create a draft email (do not send). Returns success or error message."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.CreateDraftRequest(
                user_id=user_id,
                to=to or [],
                subject=subject,
                body=body,
                cc=cc or [],
                connection_id=connection_id or 0,
            )
            response = await self._stub.CreateDraft(request)
            if not response.success:
                return response.error or "Failed to create draft"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"create_draft failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"create_draft error: {e}")
            return f"Error: {e}"

    # ===== Calendar operations (via backend tool service -> connections-service) =====

    async def list_calendars(
        self,
        user_id: str,
        connection_id: Optional[int] = None,
    ) -> str:
        """List user's calendars. Returns formatted string for LLM."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListCalendarsRequest(
                user_id=user_id,
                connection_id=connection_id or 0,
            )
            response = await self._stub.ListCalendars(request)
            if not response.success:
                return response.error or "Failed to list calendars"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"list_calendars failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"list_calendars error: {e}")
            return f"Error: {e}"

    async def get_calendar_events(
        self,
        user_id: str,
        start_datetime: str,
        end_datetime: str,
        calendar_id: str = "",
        top: int = 50,
        connection_id: Optional[int] = None,
    ) -> str:
        """Get calendar events in date range. Returns formatted string for LLM."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetCalendarEventsRequest(
                user_id=user_id,
                calendar_id=calendar_id,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                top=top,
                connection_id=connection_id or 0,
            )
            response = await self._stub.GetCalendarEvents(request)
            if not response.success:
                return response.error or "Failed to get calendar events"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"get_calendar_events failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"get_calendar_events error: {e}")
            return f"Error: {e}"

    async def get_calendar_event_by_id(
        self,
        user_id: str,
        event_id: str,
        connection_id: Optional[int] = None,
    ) -> str:
        """Get a single calendar event by ID. Returns formatted string for LLM."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetCalendarEventByIdRequest(
                user_id=user_id,
                event_id=event_id,
                connection_id=connection_id or 0,
            )
            response = await self._stub.GetCalendarEventById(request)
            if not response.success:
                return response.error or "Failed to get event"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"get_calendar_event_by_id failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"get_calendar_event_by_id error: {e}")
            return f"Error: {e}"

    async def create_calendar_event(
        self,
        user_id: str,
        subject: str,
        start_datetime: str,
        end_datetime: str,
        connection_id: Optional[int] = None,
        calendar_id: str = "",
        location: str = "",
        body: str = "",
        body_is_html: bool = False,
        attendee_emails: Optional[List[str]] = None,
        is_all_day: bool = False,
    ) -> str:
        """Create a calendar event. Returns success or error message."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.CreateCalendarEventRequest(
                user_id=user_id,
                calendar_id=calendar_id,
                subject=subject,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                location=location,
                body=body,
                body_is_html=body_is_html,
                attendee_emails=attendee_emails or [],
                is_all_day=is_all_day,
                connection_id=connection_id or 0,
            )
            response = await self._stub.CreateCalendarEvent(request)
            if not response.success:
                return response.error or "Failed to create event"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"create_calendar_event failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"create_calendar_event error: {e}")
            return f"Error: {e}"

    async def update_calendar_event(
        self,
        user_id: str,
        event_id: str,
        connection_id: Optional[int] = None,
        subject: str = "",
        start_datetime: str = "",
        end_datetime: str = "",
        location: str = "",
        body: str = "",
        body_is_html: bool = False,
        attendee_emails: Optional[List[str]] = None,
        is_all_day: bool = False,
    ) -> str:
        """Update a calendar event. Returns success or error message."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.UpdateCalendarEventRequest(
                user_id=user_id,
                event_id=event_id,
                subject=subject,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                location=location,
                body=body,
                body_is_html=body_is_html,
                attendee_emails=attendee_emails or [],
                is_all_day=is_all_day,
                connection_id=connection_id or 0,
            )
            response = await self._stub.UpdateCalendarEvent(request)
            if not response.success:
                return response.error or "Failed to update event"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"update_calendar_event failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"update_calendar_event error: {e}")
            return f"Error: {e}"

    async def delete_calendar_event(
        self,
        user_id: str,
        event_id: str,
        connection_id: Optional[int] = None,
    ) -> str:
        """Delete a calendar event. Returns success or error message."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.DeleteCalendarEventRequest(
                user_id=user_id,
                event_id=event_id,
                connection_id=connection_id or 0,
            )
            response = await self._stub.DeleteCalendarEvent(request)
            if not response.success:
                return response.error or "Failed to delete event"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"delete_calendar_event failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"delete_calendar_event error: {e}")
            return f"Error: {e}"

    async def get_contacts(
        self,
        user_id: str,
        connection_id: Optional[int] = None,
        folder_id: str = "",
        top: int = 100,
        sources: str = "all",
    ) -> str:
        """Get contacts. sources: all (O365+org), microsoft, org, caldav. Returns formatted string or JSON string."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetContactsRequest(
                user_id=user_id,
                folder_id=folder_id,
                top=top,
                connection_id=connection_id or 0,
                sources=(sources or "all").strip().lower() or "all",
            )
            response = await self._stub.GetContacts(request)
            if not response.success:
                return response.error or "Failed to get contacts"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"get_contacts failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"get_contacts error: {e}")
            return f"Error: {e}"

    async def get_contact_by_id(
        self,
        user_id: str,
        contact_id: str,
        connection_id: Optional[int] = None,
    ) -> str:
        """Get a single O365 contact by ID. Returns formatted string for LLM."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetContactByIdRequest(
                user_id=user_id,
                contact_id=contact_id,
                connection_id=connection_id or 0,
            )
            response = await self._stub.GetContactById(request)
            if not response.success:
                return response.error or "Failed to get contact"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"get_contact_by_id failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"get_contact_by_id error: {e}")
            return f"Error: {e}"

    async def create_contact(
        self,
        user_id: str,
        display_name: str = "",
        given_name: str = "",
        surname: str = "",
        connection_id: Optional[int] = None,
        folder_id: str = "",
        email_addresses: Optional[List[Dict[str, str]]] = None,
        phone_numbers: Optional[List[Dict[str, str]]] = None,
        company_name: str = "",
        job_title: str = "",
        birthday: str = "",
        notes: str = "",
    ) -> str:
        """Create an O365 contact. Returns success or error message."""
        try:
            await self._ensure_connected()
            email_json = json.dumps(email_addresses or [])
            phone_json = json.dumps(phone_numbers or [])
            request = tool_service_pb2.CreateContactRequest(
                user_id=user_id,
                folder_id=folder_id,
                display_name=display_name,
                given_name=given_name,
                surname=surname,
                company_name=company_name,
                job_title=job_title,
                birthday=birthday,
                notes=notes,
                email_addresses_json=email_json,
                phone_numbers_json=phone_json,
                connection_id=connection_id or 0,
            )
            response = await self._stub.CreateContact(request)
            if not response.success:
                return response.error or "Failed to create contact"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"create_contact failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"create_contact error: {e}")
            return f"Error: {e}"

    async def update_contact(
        self,
        user_id: str,
        contact_id: str,
        connection_id: Optional[int] = None,
        display_name: Optional[str] = None,
        given_name: Optional[str] = None,
        surname: Optional[str] = None,
        email_addresses: Optional[List[Dict[str, str]]] = None,
        phone_numbers: Optional[List[Dict[str, str]]] = None,
        company_name: Optional[str] = None,
        job_title: Optional[str] = None,
        birthday: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> str:
        """Update an O365 contact. Returns success or error message."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.UpdateContactRequest(
                user_id=user_id,
                contact_id=contact_id,
                connection_id=connection_id or 0,
            )
            if display_name is not None:
                request.display_name = display_name
            if given_name is not None:
                request.given_name = given_name
            if surname is not None:
                request.surname = surname
            if company_name is not None:
                request.company_name = company_name
            if job_title is not None:
                request.job_title = job_title
            if birthday is not None:
                request.birthday = birthday
            if notes is not None:
                request.notes = notes
            if email_addresses is not None:
                request.email_addresses_json = json.dumps(email_addresses)
            if phone_numbers is not None:
                request.phone_numbers_json = json.dumps(phone_numbers)
            response = await self._stub.UpdateContact(request)
            if not response.success:
                return response.error or "Failed to update contact"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"update_contact failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"update_contact error: {e}")
            return f"Error: {e}"

    async def delete_contact(
        self,
        user_id: str,
        contact_id: str,
        connection_id: Optional[int] = None,
    ) -> str:
        """Delete an O365 contact. Returns success or error message."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.DeleteContactRequest(
                user_id=user_id,
                contact_id=contact_id,
                connection_id=connection_id or 0,
            )
            response = await self._stub.DeleteContact(request)
            if not response.success:
                return response.error or "Failed to delete contact"
            return response.result
        except grpc.RpcError as e:
            logger.error(f"delete_contact failed: {e.code()} - {e.details()}")
            return f"Error: {e.details() or str(e)}"
        except Exception as e:
            logger.error(f"delete_contact error: {e}")
            return f"Error: {e}"

    async def m365_graph_invoke(
        self,
        user_id: str,
        operation: str,
        connection_id: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Microsoft 365 Graph workloads (To Do, Drive, OneNote, Planner) via tools-service."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.M365GraphInvokeRequest(
                user_id=user_id,
                connection_id=int(connection_id or 0),
                operation=(operation or "").strip(),
                params_json=json.dumps(params or {}),
            )
            response = await self._stub.M365GraphInvoke(request)
            if not response.success:
                return {
                    "success": False,
                    "formatted": response.error or "M365 operation failed",
                }
            try:
                data = json.loads(response.result_json or "{}")
            except json.JSONDecodeError:
                data = {}
            if not isinstance(data, dict):
                data = {"result": data}
            parts = [f"M365 {operation}"]
            if data.get("error"):
                parts.append(f"error: {data['error']}")
            else:
                for key in ("lists", "tasks", "items", "notebooks", "sections", "pages", "plans"):
                    if key in data and isinstance(data[key], list):
                        parts.append(f"{key}: {len(data[key])}")
                if "task_id" in data:
                    parts.append(f"task_id: {data.get('task_id')}")
                if "item_id" in data:
                    parts.append(f"item_id: {data.get('item_id')}")
                if "page_id" in data:
                    parts.append(f"page_id: {data.get('page_id')}")
                if data.get("success") is False:
                    parts.append("success: false")
            formatted = "\n".join(parts)
            out: Dict[str, Any] = {"success": True, "formatted": formatted}
            out.update(data)
            return out
        except grpc.RpcError as e:
            logger.error("m365_graph_invoke failed: %s - %s", e.code(), e.details())
            return {"success": False, "formatted": str(e.details() or e)}
        except Exception as e:
            logger.error("m365_graph_invoke error: %s", e)
            return {"success": False, "formatted": f"Error: {e}"}

    async def list_user_accounts(
        self,
        user_id: str,
        agent_profile_id: str = "",
        service_type: str = "all",
    ) -> List[Dict[str, Any]]:
        """List email/calendar/contacts accounts. If agent_profile_id is set, return only accounts bound to that profile."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListUserAccountsRequest(
                user_id=user_id,
                agent_profile_id=agent_profile_id,
                service_type=service_type,
            )
            response = await self._stub.ListUserAccounts(request)
            if response.success and response.result:
                try:
                    return json.loads(response.result)
                except json.JSONDecodeError:
                    return []
            return []
        except grpc.RpcError as e:
            logger.error("list_user_accounts failed: %s - %s", e.code(), e.details())
            return []
        except Exception as e:
            logger.error("list_user_accounts error: %s", e)
            return []

    async def search_contacts(
        self,
        user_id: str,
        query: str,
        sources: str = "all",
        top: int = 20,
        connection_id: Optional[int] = None,
    ) -> str:
        """Search contacts by query. Returns JSON string with contacts and formatted."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.SearchContactsRequest(
                user_id=user_id,
                query=query,
                sources=sources,
                top=top,
                connection_id=connection_id or 0,
            )
            response = await self._stub.SearchContacts(request)
            if not response.success:
                return json.dumps({"contacts": [], "formatted": response.error or "Search failed"})
            return response.result or json.dumps({"contacts": [], "formatted": "No results."})
        except grpc.RpcError as e:
            logger.error("search_contacts failed: %s - %s", e.code(), e.details())
            return json.dumps({"contacts": [], "formatted": str(e.details() or e)})
        except Exception as e:
            logger.error("search_contacts error: %s", e)
            return json.dumps({"contacts": [], "formatted": str(e)})

    async def get_agent_profile(
        self,
        user_id: str,
        profile_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get agent profile by ID for custom agent execution. Returns profile dict or None."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetAgentProfileRequest(
                user_id=user_id,
                profile_id=profile_id,
            )
            response = await self._stub.GetAgentProfile(request)
            if not response.success:
                logger.warning("get_agent_profile: %s", response.error)
                return None
            return json.loads(response.profile_json)
        except grpc.RpcError as e:
            logger.error("get_agent_profile failed: %s - %s", e.code(), e.details())
            return None
        except Exception as e:
            logger.error("get_agent_profile error: %s", e)
            return None

    async def ensure_default_profile(self, user_id: str) -> Optional[str]:
        """Return the user's active builtin profile id, creating one if needed. Returns profile_id or None."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.EnsureDefaultProfileRequest(user_id=user_id)
            response = await self._stub.EnsureDefaultProfile(request)
            pid = response.profile_id or ""
            if pid:
                if response.was_created:
                    logger.info("ensure_default_profile: created builtin profile %s for user %s", pid, user_id)
                return pid
            return None
        except grpc.RpcError as e:
            logger.error("ensure_default_profile gRPC failed: %s - %s", e.code(), e.details())
            return None
        except Exception as e:
            logger.error("ensure_default_profile error: %s", e)
            return None

    async def list_auto_routable_profiles(self, user_id: str) -> List[Dict[str, Any]]:
        """List agent profiles for the user where auto_routable=true and is_active=true."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListAutoRoutableProfilesRequest(user_id=user_id)
            response = await self._stub.ListAutoRoutableProfiles(request)
            if not response.success:
                logger.warning("list_auto_routable_profiles: %s", response.error)
                return []
            return json.loads(response.profiles_json or "[]")
        except grpc.RpcError as e:
            logger.warning("list_auto_routable_profiles failed: %s - %s", e.code(), e.details())
            return []
        except Exception as e:
            logger.warning("list_auto_routable_profiles error: %s", e)
            return []

    async def resolve_agent_handle(
        self,
        handle: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Resolve @handle to agent_profile_id and agent_name. Returns dict with agent_profile_id, agent_name, found or None."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ResolveAgentHandleRequest(
                handle=handle.strip(),
                user_id=user_id,
            )
            response = await self._stub.ResolveAgentHandle(request)
            if not response.found:
                return None
            return {
                "agent_profile_id": response.agent_profile_id,
                "agent_name": response.agent_name,
                "found": True,
            }
        except grpc.RpcError as e:
            logger.error("resolve_agent_handle failed: %s - %s", e.code(), e.details())
            return None
        except Exception as e:
            logger.error("resolve_agent_handle error: %s", e)
            return None

    async def enqueue_agent_invocation(
        self,
        agent_profile_id: str,
        input_content: str,
        user_id: str,
        source_agent_name: str = "",
        chain_depth: int = 0,
        chain_path: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Enqueue async agent-to-agent invocation. Returns dict with success, task_id, error."""
        try:
            await self._ensure_connected()
            import json as _json
            chain_path_json = _json.dumps(chain_path or [])
            request = tool_service_pb2.EnqueueAgentInvocationRequest(
                agent_profile_id=agent_profile_id,
                input_content=input_content,
                user_id=user_id,
                source_agent_name=source_agent_name,
                chain_depth=chain_depth,
                chain_path_json=chain_path_json,
            )
            response = await self._stub.EnqueueAgentInvocation(request)
            return {
                "success": response.success,
                "task_id": response.task_id or "",
                "error": response.error or "",
            }
        except grpc.RpcError as e:
            logger.error("enqueue_agent_invocation failed: %s - %s", e.code(), e.details())
            return {"success": False, "task_id": "", "error": str(e.details() or e.code())}
        except Exception as e:
            logger.error("enqueue_agent_invocation error: %s", e)
            return {"success": False, "task_id": "", "error": str(e)}

    async def read_team_posts(
        self,
        team_id: str,
        user_id: str,
        since_last_read: bool = True,
        limit: int = 20,
        mark_as_read: bool = True,
    ) -> Dict[str, Any]:
        """Read team posts (optionally since last_read). Returns dict with posts, count, team_name, success, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ReadTeamPostsRequest(
                team_id=team_id,
                user_id=user_id,
                since_last_read=since_last_read,
                limit=limit,
                mark_as_read=mark_as_read,
            )
            response = await self._stub.ReadTeamPosts(request)
            posts = [
                {
                    "post_id": p.post_id,
                    "author_id": p.author_id,
                    "author_name": p.author_name,
                    "content": p.content,
                    "post_type": p.post_type,
                    "created_at": p.created_at,
                }
                for p in response.posts
            ]
            return {
                "posts": posts,
                "count": response.count,
                "team_name": response.team_name or "",
                "success": response.success,
                "error": response.error or "",
            }
        except grpc.RpcError as e:
            logger.error("read_team_posts failed: %s - %s", e.code(), e.details())
            return {"posts": [], "count": 0, "team_name": "", "success": False, "error": str(e.details() or e.code())}
        except Exception as e:
            logger.error("read_team_posts error: %s", e)
            return {"posts": [], "count": 0, "team_name": "", "success": False, "error": str(e)}

    async def create_team_post(
        self,
        team_id: str,
        user_id: str,
        content: str,
        post_type: str = "text",
        reply_to_post_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a team post or comment. Returns dict with post_id, success, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.CreateTeamPostRequest(
                team_id=team_id,
                user_id=user_id,
                content=content.strip(),
                post_type=post_type or "text",
                reply_to_post_id=reply_to_post_id or "",
            )
            response = await self._stub.CreateTeamPost(request)
            return {
                "post_id": response.post_id or "",
                "success": response.success,
                "error": response.error or "",
            }
        except grpc.RpcError as e:
            logger.error("create_team_post failed: %s - %s", e.code(), e.details())
            return {"post_id": "", "success": False, "error": str(e.details() or e.code())}
        except Exception as e:
            logger.error("create_team_post error: %s", e)
            return {"post_id": "", "success": False, "error": str(e)}

    async def get_playbook(
        self,
        user_id: str,
        playbook_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get playbook by ID for custom agent execution. Returns playbook dict or None."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetPlaybookRequest(
                user_id=user_id,
                playbook_id=playbook_id,
            )
            response = await self._stub.GetPlaybook(request)
            if not response.success:
                logger.warning("get_playbook: %s", response.error)
                return None
            return json.loads(response.playbook_json)
        except grpc.RpcError as e:
            logger.error("get_playbook failed: %s - %s", e.code(), e.details())
            return None
        except Exception as e:
            logger.error("get_playbook error: %s", e)
            return None

    async def get_skills_by_ids(
        self,
        user_id: str,
        skill_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Batch fetch skills by IDs for pipeline injection. Returns list of skill dicts."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetSkillsByIdsRequest(
                user_id=user_id,
                skill_ids=skill_ids,
            )
            response = await self._stub.GetSkillsByIds(request)
            if not response.success:
                logger.warning("get_skills_by_ids: %s", response.error)
                return []
            return json.loads(response.skills_json or "[]")
        except grpc.RpcError as e:
            logger.error("get_skills_by_ids failed: %s - %s", e.code(), e.details())
            return []
        except Exception as e:
            logger.error("get_skills_by_ids error: %s", e)
            return []

    async def search_skills(
        self,
        user_id: str,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.5,
        active_connection_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over skills for auto-discovery. Returns list of dicts with id, similarity_score, etc."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.SearchSkillsRequest(
                user_id=user_id,
                query=query or "",
                limit=limit,
                score_threshold=score_threshold,
            )
            call_kwargs: Dict[str, Any] = {}
            if active_connection_types is not None:
                call_kwargs["metadata"] = (
                    ("active-connection-types", json.dumps(list(active_connection_types))),
                )
            response = await self._stub.SearchSkills(request, **call_kwargs)
            if not response.success:
                logger.warning("search_skills: %s", response.error)
                return []
            return json.loads(response.skills_json or "[]")
        except grpc.RpcError as e:
            logger.error("search_skills failed: %s - %s", e.code(), e.details())
            return []
        except Exception as e:
            logger.error("search_skills error: %s", e)
            return []

    async def list_skills(
        self,
        user_id: str,
        category: Optional[str] = None,
        include_builtin: bool = True,
    ) -> List[Dict[str, Any]]:
        """List user and optionally built-in skills. Returns list of skill dicts."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListSkillsRequest(
                user_id=user_id,
                category=category or "",
                include_builtin=include_builtin,
            )
            response = await self._stub.ListSkills(request)
            if not response.success:
                logger.warning("list_skills: %s", response.error)
                return []
            return json.loads(response.skills_json or "[]")
        except grpc.RpcError as e:
            logger.error("list_skills failed: %s - %s", e.code(), e.details())
            return []
        except Exception as e:
            logger.error("list_skills error: %s", e)
            return []

    async def list_skill_summaries(
        self,
        user_id: str,
        include_builtin: bool = True,
    ) -> List[Dict[str, Any]]:
        """Lightweight skill summaries for manifest/catalog injection (no procedure/schemas)."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListSkillSummariesRequest(
                user_id=user_id,
                include_builtin=include_builtin,
            )
            response = await self._stub.ListSkillSummaries(request)
            if not response.success:
                logger.warning("list_skill_summaries: %s", response.error)
                return []
            return json.loads(response.summaries_json or "[]")
        except grpc.RpcError as e:
            logger.error("list_skill_summaries failed: %s - %s", e.code(), e.details())
            return []
        except Exception as e:
            logger.error("list_skill_summaries error: %s", e)
            return []

    async def get_skill_by_slug(
        self,
        user_id: str,
        slug: str,
    ) -> Optional[Dict[str, Any]]:
        """Fetch a single skill by slug for direct acquisition. Returns full skill dict or None."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetSkillBySlugRequest(
                user_id=user_id,
                slug=slug,
            )
            response = await self._stub.GetSkillBySlug(request)
            if not response.success:
                logger.warning("get_skill_by_slug(%s): %s", slug, response.error)
                return None
            return json.loads(response.skill_json or "null")
        except grpc.RpcError as e:
            logger.error("get_skill_by_slug failed: %s - %s", e.code(), e.details())
            return None
        except Exception as e:
            logger.error("get_skill_by_slug error: %s", e)
            return None

    async def get_candidate_for_slug(
        self,
        user_id: str,
        slug: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the candidate version of a skill by slug, or None if no candidate."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetCandidateForSlugRequest(
                user_id=user_id,
                slug=slug,
            )
            response = await self._stub.GetCandidateForSlug(request)
            if not response.success or not response.has_candidate:
                return None
            return json.loads(response.skill_json or "null")
        except grpc.RpcError as e:
            logger.error("get_candidate_for_slug failed: %s - %s", e.code(), e.details())
            return None
        except Exception as e:
            logger.error("get_candidate_for_slug error: %s", e)
            return None

    async def get_skills_by_slugs(
        self,
        user_id: str,
        slugs: List[str],
    ) -> List[Dict[str, Any]]:
        """Batch fetch skills by slugs for dependency resolution."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetSkillsBySlugsRequest(
                user_id=user_id,
                slugs=slugs,
            )
            response = await self._stub.GetSkillsBySlugs(request)
            if not response.success:
                logger.warning("get_skills_by_slugs: %s", response.error)
                return []
            return json.loads(response.skills_json or "[]")
        except grpc.RpcError as e:
            logger.error("get_skills_by_slugs failed: %s - %s", e.code(), e.details())
            return []
        except Exception as e:
            logger.error("get_skills_by_slugs error: %s", e)
            return []

    async def create_skill(
        self,
        user_id: str,
        name: str,
        slug: str,
        procedure: str,
        required_tools: Optional[List[str]] = None,
        optional_tools: Optional[List[str]] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a user skill. Returns {success, skill_id, skill, error}."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.CreateSkillRequest(
                user_id=user_id,
                name=name or "Unnamed skill",
                slug=(slug or "").strip().lower().replace(" ", "-")[:100] or "unnamed-skill",
                procedure=procedure or "",
                required_tools=required_tools or [],
                optional_tools=optional_tools or [],
                description=description or "",
                category=category or "",
                tags=tags or [],
            )
            response = await self._stub.CreateSkill(request)
            if not response.success:
                return {"success": False, "skill_id": "", "skill": None, "error": response.error}
            skill = json.loads(response.skill_json) if response.skill_json else None
            return {"success": True, "skill_id": response.skill_id, "skill": skill, "error": None}
        except grpc.RpcError as e:
            logger.error("create_skill failed: %s - %s", e.code(), e.details())
            return {"success": False, "skill_id": "", "skill": None, "error": str(e.details() or e)}
        except Exception as e:
            logger.error("create_skill error: %s", e)
            return {"success": False, "skill_id": "", "skill": None, "error": str(e)}

    async def update_skill(
        self,
        user_id: str,
        skill_id: str,
        procedure: Optional[str] = None,
        improvement_rationale: Optional[str] = None,
        evidence_metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        required_tools: Optional[List[str]] = None,
        optional_tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Update a user skill (new version). Returns {success, skill_id, version, skill, error}."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.UpdateSkillRequest(
                user_id=user_id,
                skill_id=skill_id,
                procedure=procedure or "",
                improvement_rationale=improvement_rationale or "",
                evidence_metadata_json=json.dumps(evidence_metadata or {}),
                name=name or "",
                description=description or "",
                category=category or "",
                required_tools=required_tools or [],
                optional_tools=optional_tools or [],
            )
            response = await self._stub.UpdateSkill(request)
            if not response.success:
                return {"success": False, "skill_id": "", "version": 0, "skill": None, "error": response.error}
            skill = json.loads(response.skill_json) if response.skill_json else None
            return {
                "success": True,
                "skill_id": response.skill_id,
                "version": response.version,
                "skill": skill,
                "error": None,
            }
        except grpc.RpcError as e:
            logger.error("update_skill failed: %s - %s", e.code(), e.details())
            return {"success": False, "skill_id": "", "version": 0, "skill": None, "error": str(e.details() or e)}
        except Exception as e:
            logger.error("update_skill error: %s", e)
            return {"success": False, "skill_id": "", "version": 0, "skill": None, "error": str(e)}

    async def log_agent_execution(
        self,
        user_id: str,
        profile_id: str,
        playbook_id: str,
        query: str,
        status: str,
        duration_ms: int,
        steps_completed: int = 0,
        steps_total: int = 0,
        error_details: Optional[str] = None,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        steps_json: Optional[str] = None,
    ) -> Optional[str]:
        """Log a custom agent execution to agent_execution_log. Returns execution_id or None."""
        try:
            from datetime import datetime, timezone
            await self._ensure_connected()
            now = datetime.now(timezone.utc).isoformat()
            request = tool_service_pb2.LogAgentExecutionRequest(
                user_id=user_id,
                profile_id=profile_id,
                playbook_id=playbook_id or "",
                query=query or "",
                status=status or "completed",
                duration_ms=duration_ms,
                steps_completed=steps_completed,
                steps_total=steps_total,
                error_details=error_details or "",
                started_at=started_at or now,
                completed_at=completed_at or now,
                metadata_json=json.dumps(metadata or {}),
                steps_json=steps_json or "",
            )
            response = await self._stub.LogAgentExecution(request)
            if not response.success:
                logger.warning("log_agent_execution: %s", response.error)
                return None
            return response.execution_id or None
        except grpc.RpcError as e:
            logger.error("log_agent_execution failed: %s - %s", e.code(), e.details())
            return None
        except Exception as e:
            logger.error("log_agent_execution error: %s", e)
            return None

    async def park_approval(
        self,
        user_id: str,
        agent_profile_id: str,
        execution_id: Optional[str],
        step_name: str,
        prompt: str,
        preview_data: Optional[Dict[str, Any]] = None,
        thread_id: str = "",
        checkpoint_ns: str = "",
        playbook_config: Optional[Dict[str, Any]] = None,
        governance_type: str = "playbook_step",
    ) -> Optional[str]:
        """Park an approval request for background/scheduled run. Returns approval_id or None."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ParkApprovalRequest(
                user_id=user_id,
                agent_profile_id=agent_profile_id or "",
                execution_id=execution_id or "",
                step_name=step_name[:255],
                prompt=prompt[:10000],
                preview_data_json=json.dumps(preview_data or {}),
                thread_id=thread_id[:500],
                checkpoint_ns=checkpoint_ns[:255],
                playbook_config_json=json.dumps(playbook_config or {}),
                governance_type=governance_type[:50],
            )
            response = await self._stub.ParkApproval(request)
            if not response.success:
                logger.warning("park_approval: %s", response.error)
                return None
            return response.approval_id or None
        except grpc.RpcError as e:
            logger.error("park_approval failed: %s - %s", e.code(), e.details())
            return None
        except Exception as e:
            logger.error("park_approval error: %s", e)
            return None

    async def get_user_shell_policy(self, user_id: str) -> Dict[str, Any]:
        """Fetch user shell policy rules as JSON list. Returns success, rules (list), rules_json, formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetUserShellPolicyRequest(user_id=user_id or "")
            response = await self._stub.GetUserShellPolicy(request)
            rules: List[Any] = []
            if response.success and (response.rules_json or "").strip():
                try:
                    rules = json.loads(response.rules_json)
                except (json.JSONDecodeError, TypeError):
                    rules = []
            return {
                "success": response.success,
                "rules": rules if isinstance(rules, list) else [],
                "rules_json": response.rules_json or "[]",
                "formatted": response.rules_json or "[]",
                "error": response.error or "",
            }
        except grpc.RpcError as e:
            logger.error("get_user_shell_policy failed: %s - %s", e.code(), e.details())
            return {"success": False, "rules": [], "rules_json": "[]", "formatted": "", "error": e.details() or ""}
        except Exception as e:
            logger.error("get_user_shell_policy error: %s", e)
            return {"success": False, "rules": [], "rules_json": "[]", "formatted": "", "error": str(e)}

    async def upsert_shell_policy_rule(
        self,
        *,
        user_id: str,
        rule_id: str = "",
        pattern: str,
        match_mode: str = "prefix",
        action: str = "allow",
        scope_workspace_id: str = "",
        label: str = "",
        priority: int = 50,
    ) -> Dict[str, Any]:
        """Insert or update a shell policy rule. Returns success, rule_id, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.UpsertShellPolicyRuleRequest(
                user_id=user_id or "",
                rule_id=rule_id or "",
                pattern=pattern or "",
                match_mode=match_mode or "prefix",
                action=action or "allow",
                scope_workspace_id=scope_workspace_id or "",
                label=label or "",
                priority=int(priority),
            )
            response = await self._stub.UpsertShellPolicyRule(request)
            return {
                "success": response.success,
                "rule_id": response.rule_id or "",
                "error": response.error or "",
            }
        except grpc.RpcError as e:
            logger.error("upsert_shell_policy_rule failed: %s - %s", e.code(), e.details())
            return {"success": False, "rule_id": "", "error": e.details() or ""}
        except Exception as e:
            logger.error("upsert_shell_policy_rule error: %s", e)
            return {"success": False, "rule_id": "", "error": str(e)}

    async def delete_shell_policy_rule(self, user_id: str, rule_id: str) -> Dict[str, Any]:
        """Delete a shell policy rule by id."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.DeleteShellPolicyRuleRequest(
                user_id=user_id or "",
                rule_id=rule_id or "",
            )
            response = await self._stub.DeleteShellPolicyRule(request)
            return {"success": response.success, "error": response.error or ""}
        except grpc.RpcError as e:
            logger.error("delete_shell_policy_rule failed: %s - %s", e.code(), e.details())
            return {"success": False, "error": e.details() or ""}
        except Exception as e:
            logger.error("delete_shell_policy_rule error: %s", e)
            return {"success": False, "error": str(e)}

    async def grant_and_consume_shell_approval(
        self,
        user_id: str,
        *,
        approval_id: str = "",
        command: str = "",
        consume: bool = True,
    ) -> Dict[str, Any]:
        """Grant pending shell approval (consume=false) or consume an approved row (consume=true)."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GrantAndConsumeShellApprovalRequest(
                user_id=user_id or "",
                approval_id=approval_id or "",
                command=command or "",
                consume=bool(consume),
            )
            response = await self._stub.GrantAndConsumeShellApproval(request)
            return {
                "success": response.success,
                "granted_or_consumed": response.granted_or_consumed,
                "error": response.error or "",
            }
        except grpc.RpcError as e:
            logger.error("grant_and_consume_shell_approval failed: %s - %s", e.code(), e.details())
            return {"success": False, "granted_or_consumed": False, "error": e.details() or ""}
        except Exception as e:
            logger.error("grant_and_consume_shell_approval error: %s", e)
            return {"success": False, "granted_or_consumed": False, "error": str(e)}

    async def get_agent_memory(
        self,
        user_id: str,
        agent_profile_id: str,
        memory_key: str,
    ) -> Optional[Dict[str, Any]]:
        """Get agent memory value. Returns parsed JSON or None."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetAgentMemoryRequest(
                user_id=user_id,
                agent_profile_id=agent_profile_id,
                memory_key=memory_key[:500],
            )
            response = await self._stub.GetAgentMemory(request)
            if not response.success:
                return None
            if not response.memory_value_json:
                return None
            return json.loads(response.memory_value_json)
        except Exception as e:
            logger.debug("get_agent_memory error: %s", e)
            return None

    async def set_agent_memory(
        self,
        user_id: str,
        agent_profile_id: str,
        memory_key: str,
        memory_value: Dict[str, Any],
        memory_type: str = "kv",
        expires_at: Optional[str] = None,
    ) -> bool:
        """Set agent memory key. Returns True on success."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.SetAgentMemoryRequest(
                user_id=user_id,
                agent_profile_id=agent_profile_id,
                memory_key=memory_key[:500],
                memory_value_json=json.dumps(memory_value),
                memory_type=memory_type[:50],
                expires_at=expires_at or "",
            )
            response = await self._stub.SetAgentMemory(request)
            return bool(response.success)
        except Exception as e:
            logger.debug("set_agent_memory error: %s", e)
            return False

    async def list_agent_memories(
        self,
        user_id: str,
        agent_profile_id: str,
        key_prefix: Optional[str] = None,
    ) -> Optional[List[str]]:
        """List agent memory keys, optionally with prefix filter."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListAgentMemoriesRequest(
                user_id=user_id,
                agent_profile_id=agent_profile_id,
                key_prefix=key_prefix or "",
            )
            response = await self._stub.ListAgentMemories(request)
            if not response.success:
                return None
            return list(response.memory_keys)
        except Exception as e:
            logger.debug("list_agent_memories error: %s", e)
            return None

    async def delete_agent_memory(
        self,
        user_id: str,
        agent_profile_id: str,
        memory_key: str,
    ) -> bool:
        """Delete agent memory key. Returns True on success."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.DeleteAgentMemoryRequest(
                user_id=user_id,
                agent_profile_id=agent_profile_id,
                memory_key=memory_key[:500],
            )
            response = await self._stub.DeleteAgentMemory(request)
            return bool(response.success)
        except Exception as e:
            logger.debug("delete_agent_memory error: %s", e)
            return False

    async def append_agent_memory(
        self,
        user_id: str,
        agent_profile_id: str,
        memory_key: str,
        entry: Dict[str, Any],
    ) -> bool:
        """Append entry to a log-type memory. Returns True on success."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.AppendAgentMemoryRequest(
                user_id=user_id,
                agent_profile_id=agent_profile_id,
                memory_key=memory_key[:500],
                entry_json=json.dumps(entry),
            )
            response = await self._stub.AppendAgentMemory(request)
            return bool(response.success)
        except Exception as e:
            logger.debug("append_agent_memory error: %s", e)
            return False

    async def execute_connector(
        self,
        user_id: str,
        profile_id: str,
        connector_id: str,
        endpoint_id: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute a connector endpoint via Tools Service. Returns { records, count, formatted, error? } or None."""
        import json
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ExecuteConnectorRequest(
                user_id=user_id,
                profile_id=profile_id,
                connector_id=connector_id,
                endpoint_id=endpoint_id,
                params_json=json.dumps(params or {}),
            )
            response = await self._stub.ExecuteConnector(request)
            if not response.success:
                logger.warning("execute_connector: %s", response.error)
                return {"records": [], "count": 0, "formatted": response.error or "Failed", "error": response.error}
            if response.result_json:
                return json.loads(response.result_json)
            return {"records": [], "count": 0, "formatted": ""}
        except grpc.RpcError as e:
            logger.error("execute_connector failed: %s - %s", e.code(), e.details())
            return None
        except Exception as e:
            logger.error("execute_connector error: %s", e)
            return None

    async def execute_github_endpoint(
        self,
        user_id: str,
        connection_id: int,
        endpoint_id: str,
        params: Optional[Dict[str, Any]] = None,
        max_pages: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """Execute a GitHub REST endpoint via Tools Service (OAuth from external_connections)."""
        import json

        try:
            await self._ensure_connected()
            request = tool_service_pb2.ExecuteGitHubEndpointRequest(
                user_id=user_id,
                connection_id=int(connection_id),
                endpoint_id=endpoint_id or "",
                params_json=json.dumps(params or {}),
                max_pages=int(max_pages),
            )
            response = await self._stub.ExecuteGitHubEndpoint(request)
            if not response.success:
                logger.warning("execute_github_endpoint: %s", response.error)
                return {
                    "records": [],
                    "count": 0,
                    "formatted": response.error or "Failed",
                    "error": response.error,
                }
            if response.result_json:
                return json.loads(response.result_json)
            return {"records": [], "count": 0, "formatted": ""}
        except grpc.RpcError as e:
            logger.error("execute_github_endpoint failed: %s - %s", e.code(), e.details())
            return None
        except Exception as e:
            logger.error("execute_github_endpoint error: %s", e)
            return None

    async def execute_mcp_tool(
        self,
        user_id: str,
        server_id: int,
        tool_name: str,
        arguments_json: str = "{}",
    ) -> Dict[str, Any]:
        """Invoke tools/call on a user MCP server via Tools Service."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ExecuteMcpToolRequest(
                server_id=int(server_id),
                tool_name=tool_name or "",
                arguments_json=arguments_json or "{}",
                user_id=user_id or "system",
            )
            response = await self._stub.ExecuteMcpTool(request)
            err = getattr(response, "error", "") or ""
            formatted = (response.formatted or "").strip() or (response.result_json or "")
            base: Dict[str, Any] = {
                "success": bool(response.success),
                "formatted": formatted,
                "result_json": response.result_json or "",
                "error": err,
            }
            if response.success and response.result_json:
                try:
                    parsed = json.loads(response.result_json)
                    if isinstance(parsed, dict):
                        for k, v in parsed.items():
                            if k not in base:
                                base[k] = v
                except json.JSONDecodeError:
                    pass
            if not response.success:
                base["formatted"] = err or formatted or "MCP tool failed"
            return base
        except grpc.RpcError as e:
            logger.error("execute_mcp_tool failed: %s - %s", e.code(), e.details())
            return {
                "success": False,
                "formatted": e.details() or str(e.code()),
                "error": e.details() or str(e.code()),
            }
        except Exception as e:
            logger.error("execute_mcp_tool error: %s", e)
            return {"success": False, "formatted": str(e), "error": str(e)}

    async def get_mcp_server_tool_names(self, user_id: str, server_id: int) -> List[str]:
        """Tool names from cached MCP discovery for this server."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetMcpServerToolsRequest(
                server_id=int(server_id),
                user_id=user_id or "system",
            )
            response = await self._stub.GetMcpServerTools(request)
            if not response.success:
                return []
            return list(response.tool_names or [])
        except grpc.RpcError as e:
            logger.error("get_mcp_server_tool_names failed: %s - %s", e.code(), e.details())
            return []
        except Exception as e:
            logger.error("get_mcp_server_tool_names error: %s", e)
            return []

    async def probe_api_endpoint(
        self,
        user_id: str,
        url: str,
        method: str = "GET",
        headers_json: str = "",
        body_json: str = "",
        params_json: str = "",
    ) -> Dict[str, Any]:
        """Raw HTTP probe for API discovery. Returns status_code, response_headers, response_body, content_type, error."""
        import json
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ProbeApiEndpointRequest(
                user_id=user_id,
                url=url,
                method=method,
                headers_json=headers_json or "{}",
                body_json=body_json or "",
                params_json=params_json or "{}",
            )
            response = await self._stub.ProbeApiEndpoint(request)
            if not response.success:
                return {"success": False, "error": response.error or "Probe failed"}
            return {
                "success": True,
                "status_code": response.status_code,
                "response_headers_json": response.response_headers_json,
                "response_body": response.response_body,
                "content_type": response.content_type,
            }
        except grpc.RpcError as e:
            logger.error("probe_api_endpoint failed: %s - %s", e.code(), e.details())
            return {"success": False, "error": e.details() or str(e.code())}
        except Exception as e:
            logger.error("probe_api_endpoint error: %s", e)
            return {"success": False, "error": str(e)}

    async def test_connector_endpoint(
        self,
        user_id: str,
        definition_json: str,
        endpoint_id: str,
        params_json: str = "{}",
        credentials_json: str = "{}",
    ) -> Dict[str, Any]:
        """Test a connector definition against the live API. Returns records, count, raw_response, formatted, error."""
        import json
        try:
            await self._ensure_connected()
            request = tool_service_pb2.TestConnectorEndpointRequest(
                user_id=user_id,
                definition_json=definition_json,
                endpoint_id=endpoint_id,
                params_json=params_json,
                credentials_json=credentials_json,
            )
            response = await self._stub.TestConnectorEndpoint(request)
            if not response.success:
                return {
                    "success": False,
                    "records": [],
                    "count": 0,
                    "raw_response": None,
                    "formatted": response.formatted or response.error,
                    "error": response.error,
                }
            records = []
            if response.records_json:
                try:
                    records = json.loads(response.records_json)
                except json.JSONDecodeError:
                    pass
            raw_response = None
            if response.raw_response_json:
                try:
                    raw_response = json.loads(response.raw_response_json)
                except json.JSONDecodeError:
                    pass
            return {
                "success": True,
                "records": records,
                "count": response.count,
                "raw_response": raw_response,
                "formatted": response.formatted or "",
            }
        except grpc.RpcError as e:
            logger.error("test_connector_endpoint failed: %s - %s", e.code(), e.details())
            return {"success": False, "records": [], "count": 0, "formatted": e.details(), "error": e.details()}
        except Exception as e:
            logger.error("test_connector_endpoint error: %s", e)
            return {"success": False, "records": [], "count": 0, "formatted": str(e), "error": str(e)}

    async def create_data_connector(
        self,
        user_id: str,
        name: str,
        description: str,
        definition_json: str,
        requires_auth: bool = False,
        auth_fields_json: str = "[]",
        category: str = "",
    ) -> Dict[str, Any]:
        """Save a connector definition to the database. Returns connector_id, name, formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.CreateDataConnectorRequest(
                user_id=user_id,
                name=name,
                description=description,
                definition_json=definition_json,
                requires_auth=requires_auth,
                auth_fields_json=auth_fields_json,
                category=category or "",
            )
            response = await self._stub.CreateDataConnector(request)
            return {
                "success": response.success,
                "connector_id": response.connector_id,
                "name": response.name,
                "formatted": response.formatted or response.error,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("create_data_connector failed: %s - %s", e.code(), e.details())
            return {"success": False, "connector_id": "", "name": "", "formatted": e.details(), "error": e.details()}
        except Exception as e:
            logger.error("create_data_connector error: %s", e)
            return {"success": False, "connector_id": "", "name": "", "formatted": str(e), "error": str(e)}

    async def bulk_scrape_urls(
        self,
        user_id: str,
        urls_json: str,
        extract_images: bool = True,
        download_images: bool = True,
        image_output_folder: str = "",
        metadata_fields_json: str = "[]",
        max_concurrent: int = 10,
        rate_limit_seconds: float = 1.0,
        folder_id: str = "",
    ) -> Dict[str, Any]:
        """Bulk scrape URLs. Returns task_id (if backgrounded), results, count, images_found, images_downloaded, formatted."""
        import json
        try:
            await self._ensure_connected()
            request = tool_service_pb2.BulkScrapeUrlsRequest(
                user_id=user_id,
                urls_json=urls_json,
                extract_images=extract_images,
                download_images=download_images,
                image_output_folder=image_output_folder,
                metadata_fields_json=metadata_fields_json,
                max_concurrent=max_concurrent,
                rate_limit_seconds=rate_limit_seconds,
                folder_id=folder_id,
            )
            response = await self._stub.BulkScrapeUrls(request)
            results = []
            if response.results_json:
                try:
                    results = json.loads(response.results_json)
                except json.JSONDecodeError:
                    pass
            return {
                "success": response.success,
                "task_id": response.task_id or "",
                "results": results,
                "count": response.count,
                "images_found": response.images_found,
                "images_downloaded": response.images_downloaded,
                "formatted": response.formatted or response.error,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("bulk_scrape_urls failed: %s - %s", e.code(), e.details())
            return {"success": False, "task_id": "", "results": [], "formatted": e.details(), "error": e.details()}
        except Exception as e:
            logger.error("bulk_scrape_urls error: %s", e)
            return {"success": False, "task_id": "", "results": [], "formatted": str(e), "error": str(e)}

    async def get_bulk_scrape_status(
        self,
        user_id: str,
        task_id: str,
    ) -> Dict[str, Any]:
        """Get status of a bulk scrape Celery task. Returns status, progress_*, results, formatted."""
        import json
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetBulkScrapeStatusRequest(
                user_id=user_id,
                task_id=task_id,
            )
            response = await self._stub.GetBulkScrapeStatus(request)
            results = []
            if response.results_json:
                try:
                    results = json.loads(response.results_json)
                except json.JSONDecodeError:
                    pass
            return {
                "success": response.success,
                "status": response.status,
                "progress_current": response.progress_current,
                "progress_total": response.progress_total,
                "progress_message": response.progress_message,
                "results": results,
                "formatted": response.formatted or response.error,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("get_bulk_scrape_status failed: %s - %s", e.code(), e.details())
            return {"success": False, "status": "UNKNOWN", "formatted": e.details(), "error": e.details()}
        except Exception as e:
            logger.error("get_bulk_scrape_status error: %s", e)
            return {"success": False, "status": "UNKNOWN", "formatted": str(e), "error": str(e)}

    async def list_control_panes(self, user_id: str) -> Dict[str, Any]:
        """List all control panes for the user. Returns panes (list), formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListControlPanesRequest(user_id=user_id)
            response = await self._stub.ListControlPanes(request)
            panes = []
            if response.panes_json:
                try:
                    panes = json.loads(response.panes_json)
                except json.JSONDecodeError:
                    pass
            return {
                "success": response.success,
                "panes": panes,
                "formatted": response.formatted or response.error,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("list_control_panes failed: %s - %s", e.code(), e.details())
            return {"success": False, "panes": [], "formatted": e.details(), "error": e.details()}
        except Exception as e:
            logger.error("list_control_panes error: %s", e)
            return {"success": False, "panes": [], "formatted": str(e), "error": str(e)}

    async def get_connector_endpoints(
        self, user_id: str, connector_id: str
    ) -> Dict[str, Any]:
        """Get endpoint ids and metadata from a connector definition. Returns endpoints (list), formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetConnectorEndpointsRequest(
                user_id=user_id,
                connector_id=connector_id,
            )
            response = await self._stub.GetConnectorEndpoints(request)
            endpoints = []
            if response.endpoints_json:
                try:
                    endpoints = json.loads(response.endpoints_json)
                except json.JSONDecodeError:
                    pass
            return {
                "success": response.success,
                "endpoints": endpoints,
                "formatted": response.formatted or response.error,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("get_connector_endpoints failed: %s - %s", e.code(), e.details())
            return {"success": False, "endpoints": [], "formatted": e.details(), "error": e.details()}
        except Exception as e:
            logger.error("get_connector_endpoints error: %s", e)
            return {"success": False, "endpoints": [], "formatted": str(e), "error": str(e)}

    async def list_data_connectors(self, user_id: str) -> Dict[str, Any]:
        """List user-owned data source connectors. Returns connectors (list), formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListDataConnectorsRequest(user_id=user_id)
            response = await self._stub.ListDataConnectors(request)
            connectors = []
            if response.connectors_json:
                try:
                    connectors = json.loads(response.connectors_json)
                except json.JSONDecodeError:
                    pass
            return {
                "success": response.success,
                "connectors": connectors,
                "formatted": response.formatted or response.error,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("list_data_connectors failed: %s - %s", e.code(), e.details())
            return {"success": False, "connectors": [], "formatted": e.details(), "error": e.details()}
        except Exception as e:
            logger.error("list_data_connectors error: %s", e)
            return {"success": False, "connectors": [], "formatted": str(e), "error": str(e)}

    async def get_data_connector(
        self,
        user_id: str,
        connector_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get full data connector by ID (definition, endpoints; auth values redacted). Returns connector dict or None."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetDataConnectorRequest(
                user_id=user_id,
                connector_id=connector_id,
            )
            response = await self._stub.GetDataConnector(request)
            if not response.success:
                logger.warning("get_data_connector: %s", response.error)
                return None
            if not response.connector_json:
                return None
            return json.loads(response.connector_json)
        except grpc.RpcError as e:
            logger.warning("get_data_connector failed: %s - %s", e.code(), e.details())
            return None
        except Exception as e:
            logger.error("get_data_connector error: %s", e)
            return None

    async def update_data_connector(
        self,
        user_id: str,
        connector_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        connector_type: Optional[str] = None,
        definition_json: Optional[str] = None,
        requires_auth: Optional[bool] = None,
        auth_fields_json: Optional[str] = None,
        is_locked: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Update a data connector (partial update). Returns success, connector_id, formatted, error."""
        try:
            await self._ensure_connected()
            req = tool_service_pb2.UpdateDataConnectorRequest(
                user_id=user_id,
                connector_id=connector_id,
            )
            if name is not None:
                req.name = name
            if description is not None:
                req.description = description
            if connector_type is not None:
                req.connector_type = connector_type
            if definition_json is not None:
                req.definition_json = definition_json
            if requires_auth is not None:
                req.requires_auth = requires_auth
            if auth_fields_json is not None:
                req.auth_fields_json = auth_fields_json
            if is_locked is not None:
                req.is_locked = is_locked
            response = await self._stub.UpdateDataConnector(req)
            return {
                "success": response.success,
                "connector_id": response.connector_id,
                "formatted": response.formatted or response.error,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("update_data_connector failed: %s - %s", e.code(), e.details())
            return {"success": False, "connector_id": connector_id, "formatted": e.details(), "error": e.details()}
        except Exception as e:
            logger.error("update_data_connector error: %s", e)
            return {"success": False, "connector_id": connector_id, "formatted": str(e), "error": str(e)}

    async def list_playbooks(self, user_id: str) -> Dict[str, Any]:
        """List playbooks owned by the user or templates. Returns playbooks (list), formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListPlaybooksRequest(user_id=user_id)
            response = await self._stub.ListPlaybooks(request)
            playbooks = []
            if response.playbooks_json:
                try:
                    playbooks = json.loads(response.playbooks_json)
                except json.JSONDecodeError:
                    pass
            return {
                "success": response.success,
                "playbooks": playbooks,
                "formatted": response.formatted or response.error,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("list_playbooks failed: %s - %s", e.code(), e.details())
            return {"success": False, "playbooks": [], "formatted": e.details(), "error": e.details()}
        except Exception as e:
            logger.error("list_playbooks error: %s", e)
            return {"success": False, "playbooks": [], "formatted": str(e), "error": str(e)}

    async def list_agent_profiles(self, user_id: str) -> Dict[str, Any]:
        """List agent profiles for the user with derived status. Returns profiles (list), formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListAgentProfilesRequest(user_id=user_id)
            response = await self._stub.ListAgentProfiles(request)
            profiles = []
            if response.profiles_json:
                try:
                    profiles = json.loads(response.profiles_json)
                except json.JSONDecodeError:
                    pass
            return {
                "success": response.success,
                "profiles": profiles,
                "formatted": response.formatted or response.error,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("list_agent_profiles failed: %s - %s", e.code(), e.details())
            return {"success": False, "profiles": [], "formatted": e.details(), "error": e.details()}
        except Exception as e:
            logger.error("list_agent_profiles error: %s", e)
            return {"success": False, "profiles": [], "formatted": str(e), "error": str(e)}

    async def create_control_pane(
        self,
        user_id: str,
        name: str,
        connector_id: str,
        icon: str = "Tune",
        credentials_encrypted_json: str = "{}",
        connection_id: Optional[int] = None,
        controls_json: str = "[]",
        is_visible: bool = True,
        sort_order: int = 0,
        refresh_interval: int = 0,
    ) -> Dict[str, Any]:
        """Create a control pane. Returns pane_id, name, formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.CreateControlPaneRequest(
                user_id=user_id,
                name=name,
                icon=icon,
                connector_id=connector_id,
                credentials_encrypted_json=credentials_encrypted_json,
                connection_id=connection_id or 0,
                controls_json=controls_json,
                is_visible=is_visible,
                sort_order=sort_order,
                refresh_interval=refresh_interval,
            )
            response = await self._stub.CreateControlPane(request)
            return {
                "success": response.success,
                "pane_id": response.pane_id,
                "name": response.name,
                "formatted": response.formatted or response.error,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("create_control_pane failed: %s - %s", e.code(), e.details())
            return {"success": False, "pane_id": "", "name": "", "formatted": e.details(), "error": e.details()}
        except Exception as e:
            logger.error("create_control_pane error: %s", e)
            return {"success": False, "pane_id": "", "name": "", "formatted": str(e), "error": str(e)}

    async def update_control_pane(
        self,
        user_id: str,
        pane_id: str,
        name: Optional[str] = None,
        icon: Optional[str] = None,
        connector_id: Optional[str] = None,
        credentials_encrypted_json: Optional[str] = None,
        connection_id: Optional[int] = None,
        controls_json: Optional[str] = None,
        is_visible: Optional[bool] = None,
        sort_order: Optional[int] = None,
        refresh_interval: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update a control pane (partial). Returns formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.UpdateControlPaneRequest(
                user_id=user_id,
                pane_id=pane_id,
            )
            if name is not None:
                request.name = name
            if icon is not None:
                request.icon = icon
            if connector_id is not None:
                request.connector_id = connector_id
            if credentials_encrypted_json is not None:
                request.credentials_encrypted_json = credentials_encrypted_json
            if connection_id is not None:
                request.connection_id = connection_id
            if controls_json is not None:
                request.controls_json = controls_json
            if is_visible is not None:
                request.is_visible = is_visible
            if sort_order is not None:
                request.sort_order = sort_order
            if refresh_interval is not None:
                request.refresh_interval = refresh_interval
            response = await self._stub.UpdateControlPane(request)
            return {
                "success": response.success,
                "formatted": response.formatted or response.error,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("update_control_pane failed: %s - %s", e.code(), e.details())
            return {"success": False, "formatted": e.details(), "error": e.details()}
        except Exception as e:
            logger.error("update_control_pane error: %s", e)
            return {"success": False, "formatted": str(e), "error": str(e)}

    async def delete_control_pane(self, user_id: str, pane_id: str) -> Dict[str, Any]:
        """Delete a control pane. Returns formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.DeleteControlPaneRequest(
                user_id=user_id,
                pane_id=pane_id,
            )
            response = await self._stub.DeleteControlPane(request)
            return {
                "success": response.success,
                "formatted": response.formatted or response.error,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("delete_control_pane failed: %s - %s", e.code(), e.details())
            return {"success": False, "formatted": e.details(), "error": e.details()}
        except Exception as e:
            logger.error("delete_control_pane error: %s", e)
            return {"success": False, "formatted": str(e), "error": str(e)}

    async def execute_control_pane_action(
        self,
        user_id: str,
        pane_id: str,
        endpoint_id: str,
        params_json: str = "{}",
    ) -> Dict[str, Any]:
        """Execute a connector endpoint through a saved control pane. Returns raw_response, records, count, formatted, error."""
        import json
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ExecuteControlPaneActionRequest(
                user_id=user_id,
                pane_id=pane_id,
                endpoint_id=endpoint_id,
                params_json=params_json,
            )
            response = await self._stub.ExecuteControlPaneAction(request)
            if not response.success:
                return {
                    "success": False,
                    "raw_response": None,
                    "records": [],
                    "count": 0,
                    "formatted": response.formatted or response.error,
                    "error": response.error,
                }
            records = []
            if response.records_json:
                try:
                    records = json.loads(response.records_json)
                except json.JSONDecodeError:
                    pass
            raw_response = None
            if response.raw_response_json:
                try:
                    raw_response = json.loads(response.raw_response_json)
                except json.JSONDecodeError:
                    pass
            return {
                "success": True,
                "raw_response": raw_response,
                "records": records,
                "count": response.count,
                "formatted": response.formatted or "",
            }
        except grpc.RpcError as e:
            logger.error("execute_control_pane_action failed: %s - %s", e.code(), e.details())
            return {"success": False, "raw_response": None, "records": [], "count": 0, "formatted": e.details(), "error": e.details()}
        except Exception as e:
            logger.error("execute_control_pane_action error: %s", e)
            return {"success": False, "raw_response": None, "records": [], "count": 0, "formatted": str(e), "error": str(e)}

    # ===== Agent Factory meta-tools =====

    async def create_agent_profile(
        self,
        user_id: str,
        name: str,
        handle: Optional[str] = None,
        description: Optional[str] = None,
        model_preference: Optional[str] = None,
        system_prompt_additions: Optional[str] = None,
        persona_enabled: bool = False,
        auto_routable: bool = False,
        chat_history_enabled: bool = False,
        chat_visible: bool = True,
        is_active: bool = False,
    ) -> Dict[str, Any]:
        """Create an agent profile via Tools Service. Default is_active=False (paused). Handle optional (empty = schedule/Run-only). Returns agent_id, name, handle, formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.CreateAgentProfileRequest(
                user_id=user_id,
                name=name,
                handle=handle or "",
                description=description or "",
                model_preference=model_preference or "",
                system_prompt_additions=system_prompt_additions or "",
                persona_enabled=persona_enabled,
                auto_routable=auto_routable,
                chat_history_enabled=chat_history_enabled,
                chat_visible=chat_visible,
                is_active=is_active,
            )
            response = await self._stub.CreateAgentProfile(request)
            return {
                "success": response.success,
                "agent_id": response.agent_id,
                "name": response.name,
                "handle": response.handle,
                "formatted": response.formatted,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("create_agent_profile failed: %s - %s", e.code(), e.details())
            return {"success": False, "agent_id": "", "name": "", "handle": "", "formatted": "", "error": e.details() or str(e)}
        except Exception as e:
            logger.error("create_agent_profile error: %s", e)
            return {"success": False, "agent_id": "", "name": "", "handle": "", "formatted": "", "error": str(e)}

    async def set_agent_profile_status(
        self,
        user_id: str,
        agent_id: str,
        is_active: bool,
    ) -> Dict[str, Any]:
        """Set an agent profile's active status (pause or activate). Returns success, is_active, formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.SetAgentProfileStatusRequest(
                user_id=user_id,
                agent_id=agent_id,
                is_active=is_active,
            )
            response = await self._stub.SetAgentProfileStatus(request)
            return {
                "success": response.success,
                "is_active": response.is_active,
                "formatted": response.formatted,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("set_agent_profile_status failed: %s - %s", e.code(), e.details())
            return {"success": False, "is_active": False, "formatted": "", "error": e.details() or str(e)}
        except Exception as e:
            logger.error("set_agent_profile_status error: %s", e)
            return {"success": False, "is_active": False, "formatted": "", "error": str(e)}

    async def create_playbook(
        self,
        user_id: str,
        name: str,
        definition: Dict[str, Any],
        description: Optional[str] = None,
        run_context: str = "background",
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a custom playbook via Tools Service. Returns playbook_id, name, step_count, formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.CreatePlaybookRequest(
                user_id=user_id,
                name=name,
                description=description or "",
                definition_json=json.dumps(definition),
                run_context=run_context or "background",
                category=category or "",
                tags=tags or [],
            )
            response = await self._stub.CreatePlaybook(request)
            return {
                "success": response.success,
                "playbook_id": response.playbook_id,
                "name": response.name,
                "step_count": response.step_count,
                "validation_warnings": list(response.validation_warnings) if response.validation_warnings else [],
                "formatted": response.formatted,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("create_playbook failed: %s - %s", e.code(), e.details())
            return {"success": False, "playbook_id": "", "name": "", "step_count": 0, "formatted": "", "error": e.details() or str(e)}
        except Exception as e:
            logger.error("create_playbook error: %s", e)
            return {"success": False, "playbook_id": "", "name": "", "step_count": 0, "formatted": "", "error": str(e)}

    async def assign_playbook_to_agent(
        self,
        user_id: str,
        agent_id: str,
        playbook_id: str,
    ) -> Dict[str, Any]:
        """Assign a playbook to an agent profile (set default_playbook_id). Returns success, formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.AssignPlaybookToAgentRequest(
                user_id=user_id,
                agent_id=agent_id,
                playbook_id=playbook_id,
            )
            response = await self._stub.AssignPlaybookToAgent(request)
            return {
                "success": response.success,
                "formatted": response.formatted,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("assign_playbook_to_agent failed: %s - %s", e.code(), e.details())
            return {"success": False, "formatted": "", "error": e.details() or str(e)}
        except Exception as e:
            logger.error("assign_playbook_to_agent error: %s", e)
            return {"success": False, "formatted": "", "error": str(e)}

    async def update_agent_profile(
        self,
        user_id: str,
        agent_id: str,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update an agent profile. Lock enforced on backend (only is_active/is_locked when locked). Returns success, agent_id, name, formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.UpdateAgentProfileRequest(
                user_id=user_id,
                agent_id=agent_id,
                updates_json=json.dumps(updates) if updates else "{}",
            )
            response = await self._stub.UpdateAgentProfile(request)
            return {
                "success": response.success,
                "agent_id": response.agent_id,
                "name": response.name,
                "formatted": response.formatted,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("update_agent_profile failed: %s - %s", e.code(), e.details())
            return {"success": False, "agent_id": agent_id, "name": "", "formatted": "", "error": e.details() or str(e)}
        except Exception as e:
            logger.error("update_agent_profile error: %s", e)
            return {"success": False, "agent_id": agent_id, "name": "", "formatted": "", "error": str(e)}

    async def delete_agent_profile(
        self,
        user_id: str,
        agent_id: str,
    ) -> Dict[str, Any]:
        """Delete an agent profile. Blocked when profile is locked. Returns success, formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.DeleteAgentProfileRequest(
                user_id=user_id,
                agent_id=agent_id,
            )
            response = await self._stub.DeleteAgentProfile(request)
            return {
                "success": response.success,
                "formatted": response.formatted,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("delete_agent_profile failed: %s - %s", e.code(), e.details())
            return {"success": False, "formatted": "", "error": e.details() or str(e)}
        except Exception as e:
            logger.error("delete_agent_profile error: %s", e)
            return {"success": False, "formatted": "", "error": str(e)}

    async def update_playbook(
        self,
        user_id: str,
        playbook_id: str,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update a playbook. Lock enforced on backend (only is_locked toggle when locked). Returns success, playbook_id, name, step_count, validation_warnings, formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.UpdatePlaybookRequest(
                user_id=user_id,
                playbook_id=playbook_id,
                updates_json=json.dumps(updates) if updates else "{}",
            )
            response = await self._stub.UpdatePlaybook(request)
            return {
                "success": response.success,
                "playbook_id": response.playbook_id,
                "name": response.name,
                "step_count": response.step_count,
                "validation_warnings": list(response.validation_warnings) if response.validation_warnings else [],
                "formatted": response.formatted,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("update_playbook failed: %s - %s", e.code(), e.details())
            return {"success": False, "playbook_id": playbook_id, "name": "", "step_count": 0, "validation_warnings": [], "formatted": "", "error": e.details() or str(e)}
        except Exception as e:
            logger.error("update_playbook error: %s", e)
            return {"success": False, "playbook_id": playbook_id, "name": "", "step_count": 0, "validation_warnings": [], "formatted": "", "error": str(e)}

    async def delete_playbook(
        self,
        user_id: str,
        playbook_id: str,
    ) -> Dict[str, Any]:
        """Delete a playbook. Blocked when playbook is locked or is a template. Returns success, formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.DeletePlaybookRequest(
                user_id=user_id,
                playbook_id=playbook_id,
            )
            response = await self._stub.DeletePlaybook(request)
            return {
                "success": response.success,
                "formatted": response.formatted,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("delete_playbook failed: %s - %s", e.code(), e.details())
            return {"success": False, "formatted": "", "error": e.details() or str(e)}
        except Exception as e:
            logger.error("delete_playbook error: %s", e)
            return {"success": False, "formatted": "", "error": str(e)}

    async def list_available_llm_models(self, user_id: str = "system") -> Dict[str, Any]:
        """List LLM models available to the user (for Agent Factory model_preference). Returns success, models, formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListAvailableLlmModelsRequest(user_id=user_id)
            response = await self._stub.ListAvailableLlmModels(request)
            models = [
                {"model_id": m.model_id, "display_name": m.display_name, "provider": m.provider}
                for m in response.models
            ]
            return {
                "success": response.success,
                "models": models,
                "formatted": response.formatted,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("list_available_llm_models failed: %s - %s", e.code(), e.details())
            return {"success": False, "models": [], "formatted": "", "error": e.details() or str(e)}
        except Exception as e:
            logger.error("list_available_llm_models error: %s", e)
            return {"success": False, "models": [], "formatted": "", "error": str(e)}

    async def list_agent_schedules(
        self,
        user_id: str,
        agent_id: str,
    ) -> Dict[str, Any]:
        """List schedules for an agent profile. Returns success, schedules (list), formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListAgentSchedulesRequest(
                user_id=user_id,
                agent_id=agent_id,
            )
            response = await self._stub.ListAgentSchedules(request)
            schedules = []
            if response.schedules_json:
                try:
                    schedules = json.loads(response.schedules_json)
                except json.JSONDecodeError:
                    pass
            return {
                "success": response.success,
                "schedules": schedules,
                "formatted": response.formatted or response.error,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.warning("list_agent_schedules failed: %s - %s", e.code(), e.details())
            return {"success": False, "schedules": [], "formatted": e.details() or "", "error": e.details()}
        except Exception as e:
            logger.error("list_agent_schedules error: %s", e)
            return {"success": False, "schedules": [], "formatted": str(e), "error": str(e)}

    async def list_agent_data_sources(
        self,
        user_id: str,
        agent_id: str,
    ) -> Dict[str, Any]:
        """List data source bindings for an agent profile. Returns success, bindings (list), formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ListAgentDataSourcesRequest(
                user_id=user_id,
                agent_id=agent_id,
            )
            response = await self._stub.ListAgentDataSources(request)
            bindings = []
            if response.bindings_json:
                try:
                    bindings = json.loads(response.bindings_json)
                except json.JSONDecodeError:
                    pass
            return {
                "success": response.success,
                "bindings": bindings,
                "formatted": response.formatted or response.error,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.warning("list_agent_data_sources failed: %s - %s", e.code(), e.details())
            return {"success": False, "bindings": [], "formatted": e.details() or "", "error": e.details()}
        except Exception as e:
            logger.error("list_agent_data_sources error: %s", e)
            return {"success": False, "bindings": [], "formatted": str(e), "error": str(e)}

    async def create_agent_schedule(
        self,
        user_id: str,
        agent_id: str,
        schedule_type: str,
        cron_expression: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        timezone: str = "UTC",
        is_active: bool = False,
        input_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a schedule for an agent profile. Returns schedule_id, next_run_at, is_active, formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.CreateAgentScheduleRequest(
                user_id=user_id,
                agent_id=agent_id,
                schedule_type=schedule_type,
                cron_expression=cron_expression or "",
                interval_seconds=interval_seconds or 0,
                timezone=timezone or "UTC",
                is_active=is_active,
                input_context_json=json.dumps(input_context or {}),
            )
            response = await self._stub.CreateAgentSchedule(request)
            return {
                "success": response.success,
                "schedule_id": response.schedule_id,
                "next_run_at": response.next_run_at,
                "is_active": response.is_active,
                "formatted": response.formatted,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("create_agent_schedule failed: %s - %s", e.code(), e.details())
            return {"success": False, "schedule_id": "", "next_run_at": "", "is_active": False, "formatted": "", "error": e.details() or str(e)}
        except Exception as e:
            logger.error("create_agent_schedule error: %s", e)
            return {"success": False, "schedule_id": "", "next_run_at": "", "is_active": False, "formatted": "", "error": str(e)}

    async def bind_data_source_to_agent(
        self,
        user_id: str,
        agent_id: str,
        connector_id: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        permissions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Bind a data source connector to an agent profile. Returns binding_id, formatted, error."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.BindDataSourceToAgentRequest(
                user_id=user_id,
                agent_id=agent_id,
                connector_id=connector_id,
                config_overrides_json=json.dumps(config_overrides or {}),
                permissions_json=json.dumps(permissions or {}),
            )
            response = await self._stub.BindDataSourceToAgent(request)
            return {
                "success": response.success,
                "binding_id": response.binding_id,
                "formatted": response.formatted,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("bind_data_source_to_agent failed: %s - %s", e.code(), e.details())
            return {"success": False, "binding_id": "", "formatted": "", "error": e.details() or str(e)}
        except Exception as e:
            logger.error("bind_data_source_to_agent error: %s", e)
            return {"success": False, "binding_id": "", "formatted": "", "error": str(e)}

    # ===== Agent-Initiated Notifications =====

    async def send_outbound_message(
        self,
        user_id: str,
        provider: str,
        connection_id: str = "",
        message: str = "",
        format: str = "markdown",
        recipient_chat_id: str = "",
    ) -> Dict[str, Any]:
        """Send a proactive outbound message via a messaging bot (Telegram, Discord)."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.SendOutboundMessageRequest(
                user_id=user_id,
                provider=provider,
                connection_id=connection_id,
                message=message,
                format=format,
                recipient_chat_id=recipient_chat_id or "",
            )
            response = await self._stub.SendOutboundMessage(request)
            return {
                "success": response.success,
                "message_id": response.message_id,
                "channel": response.channel,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("send_outbound_message failed: %s - %s", e.code(), e.details())
            return {"success": False, "message_id": "", "channel": provider, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("send_outbound_message error: %s", e)
            return {"success": False, "message_id": "", "channel": provider, "error": str(e)}

    async def create_agent_conversation(
        self,
        user_id: str,
        message: str,
        agent_name: str = "",
        agent_profile_id: str = "",
        title: str = "",
        conversation_id: str = "",
    ) -> Dict[str, Any]:
        """Create or append to an agent-initiated conversation."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.CreateAgentConversationRequest(
                user_id=user_id,
                message=message,
                agent_name=agent_name,
                agent_profile_id=agent_profile_id,
                title=title,
                conversation_id=conversation_id,
            )
            response = await self._stub.CreateAgentConversation(request)
            return {
                "success": response.success,
                "conversation_id": response.conversation_id,
                "message_id": response.message_id,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("create_agent_conversation failed: %s - %s", e.code(), e.details())
            return {"success": False, "conversation_id": "", "message_id": "", "error": e.details() or str(e)}
        except Exception as e:
            logger.error("create_agent_conversation error: %s", e)
            return {"success": False, "conversation_id": "", "message_id": "", "error": str(e)}

    async def create_agent_message(
        self,
        user_id: str,
        team_id: str,
        from_agent_id: Optional[str] = None,
        to_agent_id: Optional[str] = None,
        message_type: str = "report",
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        parent_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an inter-agent message (team timeline)."""
        try:
            await self._ensure_connected()
            metadata_json = json.dumps(metadata or {})
            request = tool_service_pb2.CreateAgentMessageRequest(
                user_id=user_id,
                team_id=team_id,
                from_agent_id=from_agent_id or "",
                to_agent_id=to_agent_id or "",
                message_type=message_type,
                content=content,
                metadata_json=metadata_json,
                parent_message_id=parent_message_id or "",
            )
            response = await self._stub.CreateAgentMessage(request)
            out = {
                "success": response.success,
                "message_id": response.message_id,
                "message": None,
                "error": response.error if response.error else None,
            }
            if response.success and response.message_json:
                try:
                    out["message"] = json.loads(response.message_json)
                except (json.JSONDecodeError, TypeError):
                    pass
            return out
        except grpc.RpcError as e:
            logger.error("create_agent_message failed: %s - %s", e.code(), e.details())
            return {"success": False, "message_id": "", "message": None, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("create_agent_message error: %s", e)
            return {"success": False, "message_id": "", "message": None, "error": str(e)}

    async def append_line_agent_chat_message(
        self,
        user_id: str,
        conversation_id: str,
        content: str,
        agent_profile_id: str,
        agent_display_name: str = "",
        line_id: str = "",
        line_role: str = "",
        line_agent_handle: str = "",
        delegated_by_agent_id: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist a line sub-agent assistant message into the user's chat conversation."""
        try:
            await self._ensure_connected()
            meta_json = json.dumps(extra_metadata or {})
            request = tool_service_pb2.AppendLineAgentChatMessageRequest(
                user_id=user_id,
                conversation_id=conversation_id or "",
                content=content or "",
                agent_profile_id=agent_profile_id or "",
                agent_display_name=agent_display_name or "",
                line_id=line_id or "",
                line_role=line_role or "",
                line_agent_handle=line_agent_handle or "",
                delegated_by_agent_id=delegated_by_agent_id or "",
                metadata_json=meta_json,
            )
            response = await self._stub.AppendLineAgentChatMessage(request)
            return {
                "success": response.success,
                "message_id": response.message_id,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("append_line_agent_chat_message failed: %s - %s", e.code(), e.details())
            return {"success": False, "message_id": "", "error": e.details() or str(e)}
        except Exception as e:
            logger.error("append_line_agent_chat_message error: %s", e)
            return {"success": False, "message_id": "", "error": str(e)}

    async def read_team_timeline(
        self,
        team_id: str,
        user_id: str = "system",
        limit: int = 20,
        since_hours: int = 0,
    ) -> Dict[str, Any]:
        """Read recent team timeline messages (for agent context)."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ReadTeamTimelineRequest(
                user_id=user_id,
                team_id=team_id,
                limit=limit,
                since_hours=since_hours,
            )
            response = await self._stub.ReadTeamTimeline(request)
            if not response.success:
                return {"success": False, "items": [], "total": 0, "error": response.error}
            items = json.loads(response.items_json) if response.items_json else []
            return {"success": True, "items": items, "total": response.total}
        except grpc.RpcError as e:
            logger.error("read_team_timeline failed: %s - %s", e.code(), e.details())
            return {"success": False, "items": [], "total": 0, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("read_team_timeline error: %s", e)
            return {"success": False, "items": [], "total": 0, "error": str(e)}

    async def read_agent_messages(
        self,
        team_id: str,
        agent_profile_id: str,
        user_id: str = "system",
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Read messages to/from a specific agent in a team."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ReadAgentMessagesRequest(
                user_id=user_id,
                team_id=team_id,
                agent_profile_id=agent_profile_id,
                limit=limit,
            )
            response = await self._stub.ReadAgentMessages(request)
            if not response.success:
                return {"success": False, "items": [], "total": 0, "error": response.error}
            items = json.loads(response.items_json) if response.items_json else []
            return {"success": True, "items": items, "total": response.total}
        except grpc.RpcError as e:
            logger.error("read_agent_messages failed: %s - %s", e.code(), e.details())
            return {"success": False, "items": [], "total": 0, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("read_agent_messages error: %s", e)
            return {"success": False, "items": [], "total": 0, "error": str(e)}

    async def get_team_status_board(
        self,
        team_id: str,
        user_id: str = "system",
    ) -> Dict[str, Any]:
        """Get composed team overview: members with tasks, goals, last activity."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetTeamStatusBoardRequest(
                user_id=user_id,
                team_id=team_id,
            )
            response = await self._stub.GetTeamStatusBoard(request)
            if not response.success:
                return {"success": False, "board": {}, "error": response.error}
            board = json.loads(response.board_json) if response.board_json else {}
            return {"success": True, "board": board}
        except grpc.RpcError as e:
            logger.error("get_team_status_board failed: %s - %s", e.code(), e.details())
            return {"success": False, "board": {}, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("get_team_status_board error: %s", e)
            return {"success": False, "board": {}, "error": str(e)}

    async def set_workspace_entry(
        self,
        team_id: str,
        key: str,
        value: str,
        user_id: str = "system",
        updated_by_agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write a key-value entry to the team workspace (Blackboard pattern)."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.SetWorkspaceEntryRequest(
                user_id=user_id,
                team_id=team_id,
                key=key,
                value=value,
                updated_by_agent_id=updated_by_agent_id or "",
            )
            response = await self._stub.SetWorkspaceEntry(request)
            return {
                "success": response.success,
                "key": response.key,
                "updated_at": response.updated_at or None,
                "error": response.error if response.error else None,
            }
        except grpc.RpcError as e:
            logger.error("set_workspace_entry failed: %s - %s", e.code(), e.details())
            return {"success": False, "key": key, "updated_at": None, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("set_workspace_entry error: %s", e)
            return {"success": False, "key": key, "updated_at": None, "error": str(e)}

    async def read_workspace(
        self,
        team_id: str,
        user_id: str = "system",
        key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Read one workspace entry by key, or list all keys if key is None/empty."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ReadWorkspaceRequest(
                user_id=user_id,
                team_id=team_id,
                key=key or "",
            )
            response = await self._stub.ReadWorkspace(request)
            if not response.success:
                return {"success": False, "entry": None, "entries": [], "single": False, "error": response.error}
            data = json.loads(response.entries_json) if response.entries_json else ({} if response.single else [])
            if response.single:
                return {"success": True, "entry": data, "entries": [], "single": True}
            return {"success": True, "entry": None, "entries": data if isinstance(data, list) else [], "single": False}
        except grpc.RpcError as e:
            logger.error("read_workspace failed: %s - %s", e.code(), e.details())
            return {"success": False, "entry": None, "entries": [], "single": False, "error": e.details() or str(e)}
        except Exception as e:
            logger.error("read_workspace error: %s", e)
            return {"success": False, "entry": None, "entries": [], "single": False, "error": str(e)}

    async def get_goal_ancestry(self, goal_id: str, user_id: str = "system") -> Dict[str, Any]:
        """Get goal ancestry (root to leaf) for context injection."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetGoalAncestryRequest(goal_id=goal_id, user_id=user_id)
            response = await self._stub.GetGoalAncestry(request)
            if not response.success:
                return {"success": False, "goals": [], "error": response.error}
            goals = json.loads(response.goals_json) if response.goals_json else []
            return {"success": True, "goals": goals}
        except Exception as e:
            logger.error("get_goal_ancestry error: %s", e)
            return {"success": False, "goals": [], "error": str(e)}

    async def get_team_goals_tree(self, team_id: str, user_id: str = "system") -> Dict[str, Any]:
        """Get full goal tree for a team."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetTeamGoalsTreeRequest(team_id=team_id, user_id=user_id)
            response = await self._stub.GetTeamGoalsTree(request)
            if not response.success:
                return {"success": False, "tree": [], "error": response.error}
            tree = json.loads(response.tree_json) if response.tree_json else []
            return {"success": True, "tree": tree}
        except Exception as e:
            logger.error("get_team_goals_tree error: %s", e)
            return {"success": False, "tree": [], "error": str(e)}

    async def get_goals_for_agent(
        self,
        team_id: str,
        agent_profile_id: str,
        user_id: str = "system",
    ) -> Dict[str, Any]:
        """Get goals assigned to an agent in a team."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetGoalsForAgentRequest(
                team_id=team_id, agent_profile_id=agent_profile_id, user_id=user_id
            )
            response = await self._stub.GetGoalsForAgent(request)
            goals = []
            if response.success and response.goals_json:
                goals = json.loads(response.goals_json)
            return {"success": response.success, "goals": goals, "error": response.error or None}
        except Exception as e:
            logger.error("get_goals_for_agent error: %s", e)
            return {"success": False, "goals": [], "error": str(e)}

    async def update_goal_progress(
        self, goal_id: str, user_id: str = "system", progress_pct: int = 0
    ) -> Dict[str, Any]:
        """Update goal progress percentage."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.UpdateGoalProgressRequest(
                goal_id=goal_id, user_id=user_id, progress_pct=progress_pct
            )
            response = await self._stub.UpdateGoalProgress(request)
            return {"success": response.success, "error": response.error or None}
        except Exception as e:
            logger.error("update_goal_progress error: %s", e)
            return {"success": False, "error": str(e)}

    async def create_agent_task(
        self,
        team_id: str,
        user_id: str = "system",
        title: str = "Untitled",
        description: Optional[str] = None,
        assigned_agent_id: Optional[str] = None,
        goal_id: Optional[str] = None,
        priority: int = 0,
        created_by_agent_id: Optional[str] = None,
        due_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a team task."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.CreateAgentTaskRequest(
                team_id=team_id,
                user_id=user_id,
                title=title,
                description=description or "",
                assigned_agent_id=assigned_agent_id or "",
                goal_id=goal_id or "",
                priority=priority,
                created_by_agent_id=created_by_agent_id or "",
                due_date=due_date or "",
            )
            response = await self._stub.CreateAgentTask(request)
            task = {}
            if response.task_json:
                task = json.loads(response.task_json)
            return {"success": response.success, "task_id": response.task_id, "task": task, "error": response.error or None}
        except Exception as e:
            logger.error("create_agent_task error: %s", e)
            return {"success": False, "task_id": "", "task": {}, "error": str(e)}

    async def get_agent_work_queue(
        self, team_id: str, agent_profile_id: str, user_id: str = "system"
    ) -> Dict[str, Any]:
        """Get tasks assigned to an agent in a team."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetAgentWorkQueueRequest(
                team_id=team_id, agent_profile_id=agent_profile_id, user_id=user_id
            )
            response = await self._stub.GetAgentWorkQueue(request)
            tasks = []
            if response.success and response.tasks_json:
                tasks = json.loads(response.tasks_json)
            return {"success": response.success, "tasks": tasks, "error": response.error or None}
        except Exception as e:
            logger.error("get_agent_work_queue error: %s", e)
            return {"success": False, "tasks": [], "error": str(e)}

    async def update_task_status(
        self, task_id: str, user_id: str, new_status: str
    ) -> Dict[str, Any]:
        """Transition task to a new status."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.UpdateTaskStatusRequest(
                task_id=task_id, user_id=user_id, new_status=new_status
            )
            response = await self._stub.UpdateTaskStatus(request)
            task = {}
            if response.task_json:
                task = json.loads(response.task_json)
            return {"success": response.success, "task": task, "error": response.error or None}
        except Exception as e:
            logger.error("update_task_status error: %s", e)
            return {"success": False, "task": {}, "error": str(e)}

    async def assign_task_to_agent(
        self, task_id: str, agent_profile_id: str, user_id: str = "system"
    ) -> Dict[str, Any]:
        """Assign a task to an agent."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.AssignTaskToAgentRequest(
                task_id=task_id, agent_profile_id=agent_profile_id, user_id=user_id
            )
            response = await self._stub.AssignTaskToAgent(request)
            task = {}
            if response.task_json:
                task = json.loads(response.task_json)
            return {"success": response.success, "task": task, "error": response.error or None}
        except Exception as e:
            logger.error("assign_task_to_agent error: %s", e)
            return {"success": False, "task": {}, "error": str(e)}

    async def get_user_notification_preferences(
        self,
        user_id: str,
    ) -> Dict[str, Any]:
        """Get user notification preferences for channel routing."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetUserNotificationPreferencesRequest(
                user_id=user_id,
            )
            response = await self._stub.GetUserNotificationPreferences(request)
            if response.success and response.preferences_json:
                return json.loads(response.preferences_json)
            return {}
        except grpc.RpcError as e:
            logger.error("get_user_notification_preferences failed: %s - %s", e.code(), e.details())
            return {}
        except Exception as e:
            logger.error("get_user_notification_preferences error: %s", e)
            return {}

    async def get_my_profile(self, user_id: str = "system") -> Dict[str, Any]:
        """Get the current user's profile (email, display_name, username) and key settings."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetMyProfileRequest(user_id=user_id)
            response = await self._stub.GetMyProfile(request)
            if not response.success:
                return {
                    "email": "",
                    "display_name": "",
                    "username": "",
                    "preferred_name": "",
                    "timezone": "",
                    "zip_code": "",
                    "ai_context": "",
                    "success": False,
                    "error": response.error or "Unknown error",
                }
            return {
                "email": response.email or "",
                "display_name": response.display_name or "",
                "username": response.username or "",
                "preferred_name": response.preferred_name or "",
                "timezone": response.timezone or "",
                "zip_code": response.zip_code or "",
                "ai_context": response.ai_context or "",
                "success": True,
                "error": "",
            }
        except grpc.RpcError as e:
            logger.error("get_my_profile failed: %s - %s", e.code(), e.details())
            return {
                "email": "",
                "display_name": "",
                "username": "",
                "preferred_name": "",
                "timezone": "",
                "zip_code": "",
                "ai_context": "",
                "success": False,
                "error": str(e.details()),
            }
        except Exception as e:
            logger.error("get_my_profile error: %s", e)
            return {
                "email": "",
                "display_name": "",
                "username": "",
                "preferred_name": "",
                "timezone": "",
                "zip_code": "",
                "ai_context": "",
                "success": False,
                "error": str(e),
            }

    async def upsert_user_fact(
        self,
        user_id: str = "system",
        fact_key: str = "",
        value: str = "",
        category: str = "general",
        source: str = "agent",
    ) -> Dict[str, Any]:
        """Insert or update a single user fact."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.UpsertUserFactRequest(
                user_id=user_id,
                fact_key=fact_key,
                value=value,
                category=category,
                source=source,
            )
            response = await self._stub.UpsertUserFact(request)
            return {
                "success": response.success,
                "error": response.error or "",
            }
        except grpc.RpcError as e:
            logger.error("upsert_user_fact failed: %s - %s", e.code(), e.details())
            return {"success": False, "error": str(e.details())}
        except Exception as e:
            logger.error("upsert_user_fact error: %s", e)
            return {"success": False, "error": str(e)}

    async def get_user_facts(
        self,
        user_id: str = "system",
        *,
        query: str = "",
        use_themed_memory: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Get facts for a user. Optional query + use_themed_memory for theme-first retrieval."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetUserFactsRequest(user_id=user_id)
            if query:
                request.query = query
            if use_themed_memory is False:
                request.use_full_fact_list = True
            response = await self._stub.GetUserFacts(request)
            if not response.success:
                return {"success": False, "facts": [], "error": response.error or "Unknown error"}
            facts = json.loads(response.facts_json) if response.facts_json else []
            return {"success": True, "facts": facts, "error": ""}
        except grpc.RpcError as e:
            logger.error("get_user_facts failed: %s - %s", e.code(), e.details())
            return {"success": False, "facts": [], "error": str(e.details())}
        except Exception as e:
            logger.error("get_user_facts error: %s", e)
            return {"success": False, "facts": [], "error": str(e)}

    async def read_scratchpad(
        self,
        user_id: str = "system",
        pad_index: int = -1,
    ) -> Dict[str, Any]:
        """Read the user's scratch pad pads. pad_index=-1 returns all four pads; 0-3 returns a single pad."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ReadScratchpadRequest(
                user_id=user_id,
                pad_index=pad_index,
            )
            response = await self._stub.ReadScratchpad(request)
            if not response.success:
                return {"success": False, "pads": [], "active_index": 0, "error": response.error or "Unknown error"}
            pads = [
                {"index": p.index, "label": p.label, "body": p.body}
                for p in response.pads
            ]
            return {
                "success": True,
                "pads": pads,
                "active_index": response.active_index,
                "error": "",
            }
        except grpc.RpcError as e:
            logger.error("read_scratchpad failed: %s - %s", e.code(), e.details())
            return {"success": False, "pads": [], "active_index": 0, "error": str(e.details())}
        except Exception as e:
            logger.error("read_scratchpad error: %s", e)
            return {"success": False, "pads": [], "active_index": 0, "error": str(e)}

    async def write_scratchpad_pad(
        self,
        pad_index: int,
        body: str,
        label: str = "",
        user_id: str = "system",
    ) -> Dict[str, Any]:
        """Overwrite the body (and optionally label) of a single scratch pad for the user."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.WriteScratchpadPadRequest(
                user_id=user_id,
                pad_index=pad_index,
                body=body,
                label=label,
            )
            response = await self._stub.WriteScratchpadPad(request)
            return {"success": response.success, "error": response.error or ""}
        except grpc.RpcError as e:
            logger.error("write_scratchpad_pad failed: %s - %s", e.code(), e.details())
            return {"success": False, "error": str(e.details())}
        except Exception as e:
            logger.error("write_scratchpad_pad error: %s", e)
            return {"success": False, "error": str(e)}

    async def invoke_device_tool(
        self,
        user_id: str = "system",
        tool: str = "",
        args: Optional[Dict[str, Any]] = None,
        device_id: str = "",
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Invoke a tool on a connected local proxy device. Returns dict with success, result_json, error, formatted."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.InvokeDeviceToolRequest(
                user_id=user_id,
                device_id=device_id,
                tool=tool,
                args_json=json.dumps(args or {}),
                timeout_seconds=timeout,
            )
            response = await self._stub.InvokeDeviceTool(request)
            result = {
                "success": response.success,
                "result_json": response.result_json or "{}",
                "error": response.error or "",
                "formatted": response.formatted or "",
            }
            if response.success and response.result_json:
                try:
                    result["result"] = json.loads(response.result_json)
                except json.JSONDecodeError:
                    result["result"] = {}
            else:
                result["result"] = {}
            return result
        except grpc.RpcError as e:
            logger.error("invoke_device_tool failed: %s - %s", e.code(), e.details())
            return {"success": False, "result": {}, "result_json": "{}", "error": str(e.details()), "formatted": str(e.details())}
        except Exception as e:
            logger.error("invoke_device_tool error: %s", e)
            return {"success": False, "result": {}, "result_json": "{}", "error": str(e), "formatted": str(e)}

    async def set_device_workspace(
        self,
        user_id: str = "system",
        workspace_root: str = "",
        device_id: str = "",
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Set the device-side active workspace root. Returns dict with success, result_json, error, formatted."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.SetDeviceWorkspaceRequest(
                user_id=user_id,
                device_id=device_id,
                workspace_root=workspace_root,
                timeout_seconds=timeout,
            )
            response = await self._stub.SetDeviceWorkspace(request)
            result = {
                "success": response.success,
                "result_json": response.result_json or "{}",
                "error": response.error or "",
                "formatted": response.formatted or "",
            }
            if response.success and response.result_json:
                try:
                    result["result"] = json.loads(response.result_json)
                except json.JSONDecodeError:
                    result["result"] = {}
            else:
                result["result"] = {}
            return result
        except grpc.RpcError as e:
            logger.error("set_device_workspace failed: %s - %s", e.code(), e.details())
            return {"success": False, "result": {}, "result_json": "{}", "error": str(e.details()), "formatted": str(e.details())}
        except Exception as e:
            logger.error("set_device_workspace error: %s", e)
            return {"success": False, "result": {}, "result_json": "{}", "error": str(e), "formatted": str(e)}

    async def get_device_capabilities(self, user_id: str = "system") -> List[str]:
        """Return union of capabilities from all connected devices for this user."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetDeviceCapabilitiesRequest(user_id=user_id)
            response = await self._stub.GetDeviceCapabilities(request)
            return list(response.capabilities) if response.capabilities else []
        except grpc.RpcError as e:
            logger.warning("get_device_capabilities failed: %s - %s", e.code(), e.details())
            return []
        except Exception as e:
            logger.warning("get_device_capabilities error: %s", e)
            return []


# Global client instance
_backend_tool_client: Optional[BackendToolClient] = None


async def get_backend_tool_client() -> BackendToolClient:
    """Get or create the global backend tool client"""
    global _backend_tool_client

    # Reuse existing connection - gRPC channels are designed to be long-lived
    if _backend_tool_client is None:
        _backend_tool_client = BackendToolClient()
        await _backend_tool_client.connect()
        logger.info(f"✅ Backend tool client initialized and connected")

    return _backend_tool_client


async def close_backend_tool_client():
    """Close the global backend tool client"""
    global _backend_tool_client
    
    if _backend_tool_client:
        await _backend_tool_client.close()
        _backend_tool_client = None

