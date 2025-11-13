"""
Embedding Service Wrapper

Provides unified interface for embedding generation that can use either:
1. Legacy EmbeddingManager (direct OpenAI calls)
2. New Vector Service (gRPC microservice)

This wrapper enables gradual migration via feature flag.
"""

import logging
from typing import List, Optional, Dict, Any

from config import get_settings
from utils.embedding_manager import EmbeddingManager
from clients.vector_service_client import get_vector_service_client
from models.api_models import Chunk

logger = logging.getLogger(__name__)


class EmbeddingServiceWrapper:
    """
    Unified embedding service wrapper
    
    Provides the same interface as EmbeddingManager, but routes to either:
    - Legacy EmbeddingManager (USE_VECTOR_SERVICE=False)
    - New Vector Service (USE_VECTOR_SERVICE=True)
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_manager: Optional[EmbeddingManager] = None
        self.vector_service_client = None
        self._initialized = False
        self._use_vector_service = self.settings.USE_VECTOR_SERVICE
    
    async def initialize(self):
        """Initialize the appropriate embedding backend"""
        if self._initialized:
            return
        
        if self._use_vector_service:
            logger.info("Initializing Vector Service client for embeddings")
            self.vector_service_client = await get_vector_service_client()
            logger.info("✅ Vector Service client initialized")
        else:
            logger.info("Initializing legacy EmbeddingManager")
            self.embedding_manager = EmbeddingManager()
            await self.embedding_manager.initialize()
            logger.info("✅ Legacy EmbeddingManager initialized")
        
        self._initialized = True
    
    async def generate_embeddings(
        self, 
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of texts to embed
            model: Optional model name
            
        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            await self.initialize()
        
        if self._use_vector_service:
            # Use new Vector Service
            return await self.vector_service_client.generate_embeddings(
                texts=texts,
                model=model
            )
        else:
            # Use legacy EmbeddingManager
            return await self.embedding_manager.generate_embeddings(texts)
    
    async def embed_and_store_chunks(
        self,
        chunks: List[Any],
        user_id: Optional[str] = None,
        document_category: Optional[str] = None,
        document_tags: Optional[List[str]] = None,
        document_title: Optional[str] = None,
        document_author: Optional[str] = None,
        document_filename: Optional[str] = None
    ):
        """
        Generate embeddings for chunks and store in Qdrant
        
        Note: This always uses EmbeddingManager for Qdrant storage logic.
        In Vector Service mode, we generate embeddings via gRPC, then store via EmbeddingManager.
        
        Args:
            chunks: Document chunks to embed
            user_id: User ID for user-specific collection
            document_category: Document category for filtering
            document_tags: Document tags for filtering
            document_title: Document title
            document_author: Document author
            document_filename: Document filename
        """
        if not self._initialized:
            await self.initialize()
        
        if self._use_vector_service:
            # Generate embeddings via Vector Service
            texts = [chunk.content for chunk in chunks]
            embeddings = await self.vector_service_client.generate_embeddings(texts)
            
            # Store in Qdrant using EmbeddingManager's storage logic
            # We need to initialize EmbeddingManager just for storage if not already done
            if not self.embedding_manager:
                self.embedding_manager = EmbeddingManager()
                await self.embedding_manager.initialize()
            
            # Use EmbeddingManager's storage method with pre-generated embeddings
            await self._store_embeddings_with_metadata(
                chunks=chunks,
                embeddings=embeddings,
                user_id=user_id,
                document_category=document_category,
                document_tags=document_tags,
                document_title=document_title,
                document_author=document_author,
                document_filename=document_filename
            )
        else:
            # Use legacy path - generate and store in one call
            await self.embedding_manager.embed_and_store_chunks(
                chunks=chunks,
                user_id=user_id,
                document_category=document_category,
                document_tags=document_tags,
                document_title=document_title,
                document_author=document_author,
                document_filename=document_filename
            )
    
    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in Qdrant
        
        Note: This always uses EmbeddingManager for Qdrant search logic.
        In Vector Service mode, query embeddings are generated via gRPC.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum results to return
            score_threshold: Minimum similarity score
            user_id: User ID for user-specific search
            filters: Additional Qdrant filters
            
        Returns:
            List of search results
        """
        if not self._initialized:
            await self.initialize()
        
        # Search always uses EmbeddingManager (Qdrant logic)
        if not self.embedding_manager:
            self.embedding_manager = EmbeddingManager()
            await self.embedding_manager.initialize()
        
        return await self.embedding_manager.search_similar(
            query_embedding=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            user_id=user_id,
            filters=filters
        )
    
    async def clear_cache(self):
        """Clear embedding cache (if using Vector Service)"""
        if not self._initialized:
            await self.initialize()
        
        if self._use_vector_service and self.vector_service_client:
            await self.vector_service_client.clear_cache(clear_all=True)
            logger.info("✅ Vector Service cache cleared")
    
    async def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics (if using Vector Service)"""
        if not self._initialized:
            await self.initialize()
        
        if self._use_vector_service and self.vector_service_client:
            return await self.vector_service_client.get_cache_stats()
        return None
    
    async def _store_embeddings_with_metadata(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        user_id: Optional[str] = None,
        document_category: Optional[str] = None,
        document_tags: Optional[List[str]] = None,
        document_title: Optional[str] = None,
        document_author: Optional[str] = None,
        document_filename: Optional[str] = None
    ):
        """
        Store pre-generated embeddings in Qdrant with metadata
        
        This method replicates the storage logic from EmbeddingManager.embed_and_store_chunks
        but accepts pre-generated embeddings instead of generating them.
        """
        from qdrant_client.models import PointStruct
        from config import settings
        
        try:
            if not chunks or not embeddings:
                return
            
            if len(chunks) != len(embeddings):
                raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch")
            
            # Determine collection to use
            if user_id:
                await self.embedding_manager._ensure_user_collection_exists(user_id)
                collection_name = self.embedding_manager._get_user_collection_name(user_id)
            else:
                collection_name = settings.VECTOR_COLLECTION_NAME
            
            # Deduplicate chunks by content hash
            unique_chunks = self.embedding_manager._deduplicate_chunks(chunks)
            logger.info(f"📊 Deduplicated {len(chunks)} chunks to {len(unique_chunks)} unique chunks for user {user_id or 'system'}")
            
            # Build mapping of unique chunks to embeddings
            chunk_to_embedding = {}
            for chunk, embedding in zip(chunks, embeddings):
                content_hash = abs(hash(chunk.content))
                chunk_to_embedding[content_hash] = embedding
            
            # Prepare metadata info for logging
            metadata_info = ""
            if document_category:
                metadata_info += f" category={document_category}"
            if document_tags:
                metadata_info += f" tags={document_tags}"
            if document_title:
                metadata_info += f" title='{document_title}'"
            if document_author:
                metadata_info += f" author='{document_author}'"
            if document_filename:
                metadata_info += f" filename='{document_filename}'"
            
            if metadata_info:
                logger.info(f"📋 Including document metadata in vector payloads:{metadata_info}")
            
            # Prepare points for Qdrant
            points = []
            for chunk in unique_chunks:
                content_hash = abs(hash(chunk.content))
                embedding = chunk_to_embedding.get(content_hash)
                
                if not embedding:
                    logger.warning(f"⚠️ No embedding found for chunk {chunk.chunk_id}, skipping")
                    continue
                
                # Build payload with metadata
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "quality_score": chunk.quality_score,
                    "method": chunk.method,
                    "metadata": chunk.metadata,
                    "content_hash": content_hash,
                    "user_id": user_id
                }
                
                # Add optional metadata
                if document_category:
                    payload["document_category"] = document_category
                if document_tags:
                    payload["document_tags"] = document_tags
                if document_title:
                    payload["document_title"] = document_title
                if document_author:
                    payload["document_author"] = document_author
                if document_filename:
                    payload["document_filename"] = document_filename
                
                point = PointStruct(
                    id=content_hash,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            # Store in collection
            await self.embedding_manager._store_points_with_retry(points, collection_name)
            
            collection_type = "user" if user_id else "global"
            logger.info(f"✅ Stored {len(points)} unique embeddings in {collection_type} collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to store embeddings with metadata for user {user_id}: {e}")
            raise


# Singleton instance
_embedding_service_wrapper: Optional[EmbeddingServiceWrapper] = None


async def get_embedding_service() -> EmbeddingServiceWrapper:
    """Get or create singleton embedding service wrapper"""
    global _embedding_service_wrapper
    
    if _embedding_service_wrapper is None:
        _embedding_service_wrapper = EmbeddingServiceWrapper()
        await _embedding_service_wrapper.initialize()
    
    return _embedding_service_wrapper

