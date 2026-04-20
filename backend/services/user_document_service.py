"""
User Document Service - User-isolated document management
Handles document operations with complete user isolation using separate vector collections
"""

import logging
from typing import List, Optional, Dict, Any

from fastapi import UploadFile, HTTPException

from config import settings
from models.api_models import (
    DocumentInfo,
    ProcessingStatus,
    DocumentUploadResponse,
)
from clients.document_service_client import get_document_service_client
from services.embedding_service_wrapper import get_embedding_service
logger = logging.getLogger(__name__)


class UserDocumentService:
    """Service for user-isolated document management"""
    
    def __init__(self):
        self.document_service = None
        self.embedding_manager = None
    
    async def initialize(self):
        """Bind to shared container document facade and embedding service."""
        logger.info("Initializing User Document Service")
        from services.service_container import get_service_container

        container = await get_service_container()
        self.document_service = container.document_service
        self.embedding_manager = await get_embedding_service()
        logger.info("User Document Service initialized")
    
    async def upload_user_document(self, file: UploadFile, user_id: str, doc_type: str = None) -> DocumentUploadResponse:
        """Upload a document to user's private collection via document-service."""
        try:
            logger.info("Uploading document for user %s: %s", user_id, file.filename)
            dsc = get_document_service_client()
            await dsc.initialize(required=True)
            result = await dsc.upload_via_document_service(
                file,
                doc_type=doc_type,
                user_id=user_id,
                folder_id=None,
                team_id=None,
                collection_type="user",
            )
            if result.document_id:
                await self._associate_document_with_user(result.document_id, user_id)
            return result
        except Exception as e:
            logger.error("Failed to upload document for user %s: %s", user_id, e)
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}") from e
    
    async def search_user_documents(
        self, 
        query: str, 
        user_id: str, 
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search within user's private document collection"""
        try:
            logger.info(f"🔍 Searching user {user_id} documents for: {query[:50]}...")
            
            # Search only in user's collection
            results = await self.embedding_manager.search_similar(
                query_text=query,
                limit=limit,
                score_threshold=score_threshold,
                user_id=user_id
            )
            
            logger.info(f"✅ Found {len(results)} results for user {user_id}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    async def get_user_documents(self, user_id: str, skip: int = 0, limit: int = 100) -> List[DocumentInfo]:
        """Get user's documents (filtered by user_id)"""
        try:
            # Get all documents and filter by user
            # In a production system, you'd add user_id to the document_metadata table
            # and filter at the database level
            all_documents = await self.document_service.document_repository.list_documents(skip=0, limit=1000)
            
            # For now, filter based on document metadata or a separate user_documents table
            # This is a simplified example - in production, add proper user_id foreign key
            user_documents = []
            for doc in all_documents:
                # Check if document belongs to user (you'd implement proper user association)
                if await self._document_belongs_to_user(doc.document_id, user_id):
                    user_documents.append(doc)
            
            # Apply pagination
            start = skip
            end = skip + limit
            return user_documents[start:end]
            
        except Exception as e:
            logger.error(f"❌ Failed to get documents for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")
    
    async def delete_user_document(self, document_id: str, user_id: str) -> bool:
        """Delete a document from user's collection"""
        try:
            logger.info(f"🗑️  Deleting document {document_id} for user {user_id}")
            
            # Verify document belongs to user
            if not await self._document_belongs_to_user(document_id, user_id):
                raise HTTPException(status_code=403, detail="Document not found or access denied")
            
            # Delete from vector database (user-specific collection)
            await self.embedding_manager.delete_document_chunks(document_id, user_id)
            
            success = await self.document_service.delete_document(
                document_id, user_id=user_id
            )
            
            if success:
                # Remove user association
                await self._remove_user_document_association(document_id, user_id)
                logger.info(f"✅ Deleted document {document_id} for user {user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to delete document for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
    
    async def get_user_collection_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about user's document collection"""
        try:
            # Get vector database stats
            vector_stats = await self.embedding_manager.get_user_collection_stats(user_id)
            
            # Get document count from metadata
            user_documents = await self.get_user_documents(user_id)
            
            return {
                "total_documents": len(user_documents),
                "total_embeddings": vector_stats["total_points"],
                "vector_dimensions": vector_stats["vector_size"],
                "collection_exists": vector_stats["collection_exists"]
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get stats for user {user_id}: {e}")
            return {
                "total_documents": 0,
                "total_embeddings": 0,
                "vector_dimensions": settings.EMBEDDING_DIMENSIONS,
                "collection_exists": False
            }
    
    async def delete_user_collection(self, user_id: str) -> bool:
        """Delete user's entire collection (for account deletion)"""
        try:
            logger.info(f"🗑️  Deleting entire collection for user {user_id}")
            
            # Delete vector collection
            vector_success = await self.embedding_manager.delete_user_collection(user_id)
            
            # Delete document metadata associations
            user_docs = await self.get_user_documents(user_id)
            for doc in user_docs:
                await self.document_service.delete_document(doc.document_id)
                await self._remove_user_document_association(doc.document_id, user_id)
            
            logger.info(f"✅ Deleted collection for user {user_id}")
            return vector_success
            
        except Exception as e:
            logger.error(f"❌ Failed to delete collection for user {user_id}: {e}")
            return False
    
    # Helper methods for user-document association
    # In production, implement these with a proper user_documents table
    
    async def _associate_document_with_user(self, document_id: str, user_id: str):
        """Associate a document with a user"""
        try:
            # In production: INSERT INTO user_documents (user_id, document_id, created_at) VALUES (...)
            # For now, you could store this in document metadata or a separate table
            logger.debug(f"Associated document {document_id} with user {user_id}")
        except Exception as e:
            logger.error(f"Failed to associate document with user: {e}")
    
    async def _document_belongs_to_user(self, document_id: str, user_id: str) -> bool:
        """Check if a document belongs to a specific user or is accessible to them"""
        try:
            doc_info = await self.document_service.get_document(document_id)
            if not doc_info:
                return False
            
            # User owns the document
            if getattr(doc_info, 'user_id', None) == user_id:
                return True
            
            # Document is global (everyone can read)
            if getattr(doc_info, 'collection_type', 'user') == 'global':
                return True
            
            # Document is in a team the user belongs to
            doc_team_id = getattr(doc_info, 'team_id', None)
            if doc_team_id:
                from api.teams_api import team_service
                role = await team_service.check_team_access(doc_team_id, user_id)
                return role is not None
            
            return False
        except Exception as e:
            logger.error(f"Failed to check document ownership: {e}")
            return False
    
    async def _remove_user_document_association(self, document_id: str, user_id: str):
        """Remove user-document association"""
        try:
            # In production: DELETE FROM user_documents WHERE user_id = ? AND document_id = ?
            logger.debug(f"Removed association between document {document_id} and user {user_id}")
        except Exception as e:
            logger.error(f"Failed to remove document association: {e}")
    
    async def store_text_document_for_user(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any],
        user_id: str,
        filename: str = None,
    ) -> bool:
        """Persist text via document-service StoreTextDocument JSON."""
        try:
            fn = filename or f"{doc_id}.txt"
            dsc = get_document_service_client()
            await dsc.initialize(required=True)
            ok, data, err = await dsc.store_text_document_json(
                user_id,
                {
                    "doc_id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "filename": fn,
                    "user_id": user_id,
                    "collection_type": "user",
                },
                timeout=600.0,
            )
            if ok and data and data.get("success"):
                await self._associate_document_with_user(doc_id, user_id)
                return True
            logger.error("store_text_document_json failed: %s", err or data)
            return False
        except Exception as e:
            logger.error("Failed to store text document %s for user %s: %s", doc_id, user_id, e)
            return False

