"""
Backend Tool Client - gRPC client for accessing backend data services
"""

import logging
import os
import json
from typing import List, Dict, Any, Optional
import asyncio

import grpc
from protos import tool_service_pb2, tool_service_pb2_grpc

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
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[tool_service_pb2_grpc.ToolServiceStub] = None
        
        logger.info(f"Backend Tool Client configured for {self.address}")
    
    async def connect(self):
        """Establish connection to backend tool service"""
        if self._channel is None:
            logger.debug(f"Connecting to backend tool service at {self.address}...")
            # Increase message size limits for large responses (default is 4MB)
            options = [
                ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100 MB
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100 MB
            ]
            self._channel = grpc.aio.insecure_channel(self.address, options=options)
            self._stub = tool_service_pb2_grpc.ToolServiceStub(self._channel)
            logger.debug(f"✅ Connected to backend tool service")
    
    async def close(self):
        """Close connection to backend tool service"""
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None
            logger.debug("Disconnected from backend tool service")
    
    async def _ensure_connected(self):
        """Ensure connection is established"""
        if self._stub is None:
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
            
            response = await self._stub.SearchDocuments(request)
            
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
            
            response = await self._stub.GetDocument(request)
            
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
            
            response = await self._stub.GetDocumentContent(request)
            
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
            
            response = await self._stub.GetDocumentChunks(request)
            
            # Convert proto response to list of dicts
            chunks = []
            for chunk_proto in response.chunks:
                # Parse metadata JSON string
                metadata = {}
                if chunk_proto.metadata:
                    try:
                        import json
                        metadata = json.loads(chunk_proto.metadata)
                    except:
                        pass
                
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
            
            response = await self._stub.FindDocumentByPath(request)
            
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

            response = await self._stub.FindDocumentsByTags(request)

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
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a file in the user's My Documents section
        
        Args:
            filename: Name of the file to create
            content: File content as string
            user_id: User ID (required)
            folder_id: Optional folder ID to place file in
            folder_path: Optional folder path (e.g., "Projects/Electronics") - will create if needed
            title: Optional document title (defaults to filename)
            tags: Optional list of tags for the document
            category: Optional category for the document
        
        Returns:
            Dict with success, document_id, filename, folder_id, and message
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
            
            response = await self._stub.CreateUserFile(request)
            
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
            
            response = await self._stub.CreateUserFolder(request)
            
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
            response = await self._stub.GetFolderTree(request)
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
            
            response = await self._stub.UpdateDocumentMetadata(request)
            
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
            
            request = tool_service_pb2.UpdateDocumentContentRequest(
                user_id=user_id,
                document_id=document_id,
                content=content,
                append=append
            )
            
            response = await self._stub.UpdateDocumentContent(request)
            
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
            
            # Convert operations to proto
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
                        confidence=op.get("confidence") or 0.0
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
            
            response = await self._stub.ProposeDocumentEdit(request)
            
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
        agent_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Apply operations directly to a document (for authorized agents only)
        
        Args:
            document_id: Document ID to edit
            operations: List of EditorOperation dicts to apply
            user_id: User ID (required - must match document owner)
            agent_name: Name of agent requesting this operation (for security check)
        
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
                agent_name=agent_name
            )
            
            response = await self._stub.ApplyOperationsDirectly(request)
            
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
            
            response = await self._stub.ApplyDocumentEditProposal(request)
            
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
        reference_strength: float = 0.5
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
                    result["images"].append({
                        "filename": img.filename,
                        "path": img.path,
                        "url": img.url,
                        "width": img.width,
                        "height": img.height,
                        "format": img.format
                    })
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
            
            response = await self._stub.FindDocumentsByEntities(request)
            
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
            
            response = await self._stub.FindRelatedDocumentsByEntities(request)
            
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
        user_id: str = "system"
    ) -> List[Dict[str, Any]]:
        """
        Crawl web content from URLs
        
        Args:
            url: Single URL to crawl
            urls: Multiple URLs to crawl
            user_id: User ID
            
        Returns:
            List of crawled content
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.WebCrawlRequest(
                url=url if url else "",
                urls=urls if urls else [],
                user_id=user_id
            )
            
            response = await self._stub.CrawlWebContent(request)
            
            results = []
            for result in response.results:
                results.append({
                    'url': result.url,
                    'title': result.title,
                    'content': result.content,
                    'html': result.html,  # WebCrawlResponse (singular) has html field
                    'metadata': dict(result.metadata)
                })
            
            return results
            
        except grpc.RpcError as e:
            logger.error(f"Web crawl failed: {e.code()} - {e.details()}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in web crawl: {e}")
            return []
    
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
            return f"Error searching images: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in image search: {e}")
            return f"Error searching images: {str(e)}"
    
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
        include_static: bool = False
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
            include_static: Also generate a static SVG version (default: False)
            
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
                include_static=include_static
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

                if response.HasField("static_svg"):
                    result["static_svg"] = response.static_svg
                if response.HasField("static_format"):
                    result["static_format"] = response.static_format
                
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
            
            # Convert proto response to dict
            tables = []
            for table in response.tables:
                columns = []
                for col in table.columns:
                    columns.append({
                        'name': col.name,
                        'type': col.type,
                        'is_nullable': col.is_nullable
                    })
                
                tables.append({
                    'table_id': table.table_id,
                    'name': table.name,
                    'description': table.description,
                    'database_id': table.database_id,
                    'database_name': table.database_name,
                    'columns': columns,
                    'row_count': table.row_count
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
    
    async def query_data_workspace(
        self,
        workspace_id: str,
        query: str,
        query_type: str = "natural_language",
        user_id: str = "system",
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Execute a query against a data workspace (SQL or natural language)
        
        Args:
            workspace_id: Workspace ID to query
            query: SQL query or natural language query
            query_type: "sql" or "natural_language"
            user_id: User ID for access control
            limit: Maximum rows to return (default: 100)
            
        Returns:
            Dict with 'success', 'column_names', 'results', 'result_count', 
            'execution_time_ms', 'generated_sql', 'error_message'
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.QueryDataWorkspaceRequest(
                workspace_id=workspace_id,
                query=query,
                query_type=query_type,
                user_id=user_id,
                limit=limit
            )
            
            response = await self._stub.QueryDataWorkspace(request)
            
            # Parse results JSON
            import json
            results = []
            if response.results_json:
                try:
                    results = json.loads(response.results_json)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse results JSON: {response.results_json[:100]}")
                    results = []
            
            result = {
                'success': response.success,
                'column_names': list(response.column_names),
                'results': results,
                'result_count': response.result_count,
                'execution_time_ms': response.execution_time_ms,
                'generated_sql': response.generated_sql
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
                'error_message': str(e)
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
                'error_message': str(e)
            }

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
    ) -> str:
        """Search emails. Returns formatted string for LLM."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.SearchEmailsRequest(
                user_id=user_id,
                query=query,
                top=top,
                from_address=from_address or "",
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
    ) -> str:
        """Get full email thread. Returns formatted string for LLM."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetEmailThreadRequest(
                user_id=user_id,
                conversation_id=conversation_id,
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
    ) -> str:
        """Send email. Returns success or error message. HITL should be applied by agent."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.SendEmailRequest(
                user_id=user_id,
                to=to,
                subject=subject,
                body=body,
                cc=cc or [],
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
    ) -> str:
        """Reply to an email. Returns success or error message."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.ReplyToEmailRequest(
                user_id=user_id,
                message_id=message_id,
                body=body,
                reply_all=reply_all,
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

    async def get_email_folders(self, user_id: str) -> str:
        """List email folders. Returns formatted string for LLM."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetEmailFoldersRequest(user_id=user_id)
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

    async def get_email_statistics(self, user_id: str) -> str:
        """Get email statistics (inbox total/unread). Returns formatted string."""
        try:
            await self._ensure_connected()
            request = tool_service_pb2.GetEmailStatisticsRequest(user_id=user_id)
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

