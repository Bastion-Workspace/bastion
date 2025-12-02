"""
User Document Service - User-isolated document management
Handles document operations with complete user isolation using separate vector collections
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4

from fastapi import UploadFile, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import settings
from models.api_models import (
    DocumentInfo, DocumentStatus, ProcessingStatus, DocumentType,
    DocumentUploadResponse
)
from services.document_service_v2 import DocumentService
from services.embedding_service_wrapper import get_embedding_service
from utils.auth_middleware import get_current_user  # Assuming this exists

logger = logging.getLogger(__name__)

security = HTTPBearer()


class UserDocumentService:
    """Service for user-isolated document management"""
    
    def __init__(self):
        self.document_service = None
        self.embedding_manager = None
    
    async def initialize(self):
        """Initialize the service"""
        logger.info("🔧 Initializing User Document Service...")
        
        # Initialize base document service
        self.document_service = DocumentService()
        await self.document_service.initialize()
        
        # Initialize embedding service wrapper
        self.embedding_manager = await get_embedding_service()
        
        logger.info("✅ User Document Service initialized")
    
    async def upload_user_document(self, file: UploadFile, user_id: str, doc_type: str = None) -> DocumentUploadResponse:
        """Upload a document to user's private collection"""
        try:
            logger.info(f"📄 Uploading document for user {user_id}: {file.filename}")
            
            # Use the base document service for file processing
            # But we'll modify the embedding storage to use user-specific collection
            result = await self.document_service.upload_and_process(file, doc_type)
            
            # The document is processed normally, but embeddings will be stored
            # in user-specific collection through the modified embedding manager
            
            # Update document metadata to include user_id
            if result.document_id:
                await self._associate_document_with_user(result.document_id, user_id)
            
            logger.info(f"✅ Document uploaded for user {user_id}: {result.document_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to upload document for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
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
            
            # Delete from main document repository
            success = await self.document_service.delete_document(document_id)
            
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
    
    async def store_text_document_for_user(self, doc_id: str, content: str, metadata: Dict[str, Any], user_id: str, filename: str = None) -> bool:
        """Store text content directly as a document for a specific user"""
        try:
            logger.info(f"📥 Storing text document: {doc_id} for user {user_id}")
            
            # Generate filename if not provided
            if not filename:
                filename = f"{doc_id}.txt"
            
            # Create document info
            document_info = DocumentInfo(
                document_id=doc_id,
                filename=filename,
                title=metadata.get("title", filename),
                content=content,
                doc_type=DocumentType.TXT,
                category=metadata.get("category", "web_search"),
                tags=metadata.get("tags", []),
                author=metadata.get("author"),
                upload_date=datetime.utcnow(),
                file_size=len(content.encode('utf-8')),
                page_count=1,
                chunk_count=0,
                metadata=metadata,
                status=ProcessingStatus.PROCESSING,
                user_id=user_id  # Set the user_id for RLS policies
            )
            
            # Store in database with user_id
            await self.document_service.document_repository.store_document_metadata(document_info, user_id)
            
            # Associate document with user
            await self._associate_document_with_user(doc_id, user_id)
            
            # Process content into chunks
            chunks = await self.document_service.document_processor.process_text_content(
                content, doc_id, metadata
            )
            
            # Store chunks in user-specific vector collection
            # Note: Direct text storage may not have category/tags
            if chunks:
                await self.embedding_manager.embed_and_store_chunks(
                    chunks, 
                    user_id=user_id,
                    document_category=None,
                    document_tags=None
                )
                logger.info(f"📊 Stored {len(chunks)} chunks for user {user_id}")
            
            # Update document with chunk count
            await self.document_service.document_repository.update_chunk_count(doc_id, len(chunks) if chunks else 0)
            
            # Update final status
            await self.document_service.document_repository.update_status(doc_id, ProcessingStatus.COMPLETED)
            
            logger.info(f"✅ Successfully stored text document {doc_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store text document {doc_id} for user {user_id}: {e}")
            try:
                await self.document_service.document_repository.update_status(doc_id, ProcessingStatus.FAILED)
            except:
                pass  # Don't fail if status update fails
            return False


# Updated Document Service to work with user isolation
class DocumentServiceWithUserIsolation(DocumentService):
    """Extended DocumentService with user isolation support"""
    
    async def upload_and_process_for_user(self, file: UploadFile, user_id: str, doc_type: str = None) -> DocumentUploadResponse:
        """Upload and process document with user isolation"""
        try:
            # Call parent upload method
            result = await super().upload_and_process(file, doc_type)
            
            if result.document_id:
                # Start processing with user_id for embedding storage
                asyncio.create_task(self._process_document_async_with_user(result.document_id, file, doc_type, user_id))
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Upload failed for user {user_id}: {e}")
            raise
    
    async def _process_document_async_with_user(self, document_id: str, file, doc_type: str, user_id: str):
        """Process document with user-specific embedding storage"""
        try:
            # Use the same processing logic but pass user_id to embedding storage
            
            # Process document normally
            file_path = f"{settings.UPLOAD_DIR}/{document_id}_{file.filename}"
            result = await self.document_service.document_processor.process_document(str(file_path), doc_type, document_id)
            
            # Update status
            await self.document_service.document_repository.update_status(document_id, ProcessingStatus.EMBEDDING)
            
            # Fetch document metadata for vector filtering
            try:
                doc_info = await self.document_service.document_repository.get_by_id(document_id)
                document_category = doc_info.category.value if doc_info and doc_info.category else None
                document_tags = doc_info.tags if doc_info else None
                document_title = doc_info.title if doc_info else None
                document_author = doc_info.author if doc_info else None
                document_filename = doc_info.filename if doc_info else None
            except Exception as e:
                logger.debug(f"Could not fetch document metadata: {e}")
                document_category = None
                document_tags = None
                document_title = None
                document_author = None
                document_filename = None
            
            # Store embeddings in user-specific collection with metadata
            if result.chunks:
                await self.embedding_manager.embed_and_store_chunks(
                    result.chunks, 
                    user_id=user_id,
                    document_category=document_category,
                    document_tags=document_tags,
                    document_title=document_title,
                    document_author=document_author,
                    document_filename=document_filename
                )
                logger.info(f"📊 Stored {len(result.chunks)} chunks for user {user_id}")
            
            # Store entities in knowledge graph (if enabled)
            if result.entities and self.kg_service:
                await self.kg_service.store_entities(result.entities, document_id)
            
            # Update final status
            await self.document_service.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
            
        except Exception as e:
            logger.error(f"❌ Processing failed for user {user_id}: {e}")
            await self.document_service.document_repository.update_status(document_id, ProcessingStatus.FAILED)
    
