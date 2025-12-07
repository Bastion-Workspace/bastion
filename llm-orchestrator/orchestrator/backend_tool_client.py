"""
Backend Tool Client - gRPC client for accessing backend data services
"""

import logging
import os
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
        filters: List[str] = None
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
                filters=filters or []
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
        data_types: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get weather data for location
        
        Args:
            location: Location name
            user_id: User ID
            data_types: Types of data to retrieve (e.g., ["current", "forecast"])
            
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
            
            response = await self._stub.GetWeatherData(request)
            
            return {
                'location': response.location,
                'current_conditions': response.current_conditions,
                'forecast': list(response.forecast),
                'alerts': list(response.alerts),
                'metadata': dict(response.metadata)
            }
            
        except grpc.RpcError as e:
            logger.error(f"Get weather failed: {e.code()} - {e.details()}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting weather: {e}")
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
    
    async def search_and_crawl(
        self,
        query: str,
        max_results: int = 10,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Combined search and crawl operation
        
        Args:
            query: Search query
            max_results: Maximum number of results to crawl
            user_id: User ID
            
        Returns:
            Dict with 'search_results' and 'crawled_content'
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.SearchAndCrawlRequest(
                query=query,
                num_results=max_results,
                user_id=user_id
            )
            
            response = await self._stub.SearchAndCrawl(request)
            
            search_results = []
            for result in response.search_results:
                search_results.append({
                    'title': result.title,
                    'url': result.url,
                    'snippet': result.snippet,
                    'content': result.snippet  # WebSearchResult doesn't have content field, use snippet
                })
            
            crawled_content = []
            for result in response.crawl_results:
                crawled_content.append({
                    'url': result.url,
                    'title': result.title,
                    'content': result.content,
                    'html': result.content,  # WebCrawlResult doesn't have html field, use content
                    'metadata': dict(result.metadata)
                })
            
            return {
                'search_results': search_results,
                'crawled_content': crawled_content
            }
            
        except grpc.RpcError as e:
            logger.error(f"Search and crawl failed: {e.code()} - {e.details()}")
            return {'search_results': [], 'crawled_content': []}
        except Exception as e:
            logger.error(f"Unexpected error in search and crawl: {e}")
            return {'search_results': [], 'crawled_content': []}
    
    # ===== Query Enhancement =====
    
    async def expand_query(
        self,
        query: str,
        num_variations: int = 3,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Expand query with variations
        
        Args:
            query: Original query
            num_variations: Number of variations to generate
            user_id: User ID
            
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

