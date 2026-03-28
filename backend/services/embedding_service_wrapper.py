"""
Embedding Service Wrapper

Unified interface for embedding generation and storage using Vector Service.
All embedding generation routed through dedicated Vector Service microservice.
All storage operations handled by VectorStoreService.
Full-text search: chunks are also stored in PostgreSQL document_chunks table.
"""

import logging
from typing import List, Optional, Dict, Any

from config import get_settings, settings
from clients.vector_service_client import get_vector_service_client
from services.vector_store_service import get_vector_store
from models.api_models import Chunk
from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)

# Batch size for PostgreSQL chunk inserts
DOCUMENT_CHUNKS_INSERT_BATCH = 50


class EmbeddingServiceWrapper:
    """
    Unified embedding service wrapper - Vector Service only
    
    Architecture:
    - Embedding generation: Vector Service (gRPC microservice)
    - Vector storage: VectorStoreService (Qdrant)
    - No fallback: Always uses Vector Service
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.vector_service_client = None
        self.vector_store = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize Vector Service client and vector store"""
        if self._initialized:
            return
        
        # Initialize vector store for storage operations
        self.vector_store = await get_vector_store()
        logger.debug("Vector Store Service initialized")
        
        # Initialize Vector Service for embedding generation (non-blocking)
        logger.debug("Initializing Vector Service client for embeddings")
        self.vector_service_client = await get_vector_service_client(required=False)
        logger.info("Vector Service client initialized (may retry connection later)")
        
        self._initialized = True
    
    async def generate_embeddings(
        self, 
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for texts via Vector Service
        
        Args:
            texts: List of texts to embed
            model: Optional model name
            
        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            await self.initialize()
        
        return await self.vector_service_client.generate_embeddings(
            texts=texts,
            model=model
        )
    
    async def embed_and_store_chunks(
        self,
        chunks: List[Any],
        user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        document_category: Optional[str] = None,
        document_tags: Optional[List[str]] = None,
        document_title: Optional[str] = None,
        document_author: Optional[str] = None,
        document_filename: Optional[str] = None,
        is_image_sidecar: bool = False,
    ):
        """
        Generate embeddings for chunks and store in vector database
        
        Uses Vector Service for generation, VectorStoreService for storage.
        
        Args:
            chunks: Document chunks to embed
            user_id: User ID for user-specific collection
            document_category: Document category for filtering
            document_tags: Document tags for filtering
            document_title: Document title
            document_author: Document author
            document_filename: Document filename
            is_image_sidecar: If True, payload is tagged so document search can exclude it
        """
        if not self._initialized:
            await self.initialize()
        
        # Generate embeddings via Vector Service
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.vector_service_client.generate_embeddings(texts)
        
        # Store via VectorStoreService
        await self._store_embeddings_with_metadata(
            chunks=chunks,
            embeddings=embeddings,
            user_id=user_id,
            team_id=team_id,
            document_category=document_category,
            document_tags=document_tags,
            document_title=document_title,
            document_author=document_author,
            document_filename=document_filename,
            is_image_sidecar=is_image_sidecar,
        )
    
    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        user_id: Optional[str] = None,
        team_ids: Optional[List[str]] = None,
        filter_category: Optional[str] = None,
        filter_tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in vector database
        
        Uses VectorStoreService for search operations.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum results to return
            score_threshold: Minimum similarity score
            user_id: User ID for user-specific search
            filter_category: Filter by document category
            filter_tags: Filter by document tags
            
        Returns:
            List of search results
        """
        if not self._initialized:
            await self.initialize()
        
        return await self.vector_store.search_similar(
            query_embedding=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            user_id=user_id,
            team_ids=team_ids,
            filter_category=filter_category,
            filter_tags=filter_tags
        )
    
    async def clear_cache(self):
        """Clear Vector Service embedding cache"""
        if not self._initialized:
            await self.initialize()
        
        if self.vector_service_client:
            await self.vector_service_client.clear_cache(clear_all=True)
            logger.info("Vector Service cache cleared")
    
    async def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get Vector Service cache statistics"""
        if not self._initialized:
            await self.initialize()
        
        if self.vector_service_client:
            return await self.vector_service_client.get_cache_stats()
        return None
    
    async def delete_document_chunks(
        self,
        document_id: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Remove document from all search indexes: Qdrant vectors and PostgreSQL
        document_chunks (full-text search). Optionally zeros chunk_count on
        document_metadata.
        """
        if not self._initialized:
            await self.initialize()

        result = await self.vector_store.delete_points_by_filter(
            document_id=document_id,
            user_id=user_id
        )

        # Remove from PostgreSQL full-text search index (admin context for writes)
        try:
            from services.database_manager.database_helpers import execute
            rls_context = {"user_id": "", "user_role": "admin"}
            await execute(
                "DELETE FROM document_chunks WHERE document_id = $1",
                document_id,
                rls_context=rls_context,
            )
            await execute(
                "UPDATE document_metadata SET chunk_count = 0 WHERE document_id = $1",
                document_id,
                rls_context=rls_context,
            )
        except Exception as e:
            logger.warning("Failed to delete document_chunks or update chunk_count for %s: %s", document_id, e)

        return result
    
    async def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about vector collections
        
        Args:
            collection_name: Optional specific collection to get stats for
            
        Returns:
            Dictionary with collection statistics
        """
        if not self._initialized:
            await self.initialize()
        
        return await self.vector_store.get_collection_stats(collection_name)
    
    async def _store_embeddings_with_metadata(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        document_category: Optional[str] = None,
        document_tags: Optional[List[str]] = None,
        document_title: Optional[str] = None,
        document_author: Optional[str] = None,
        document_filename: Optional[str] = None,
        is_image_sidecar: bool = False,
    ):
        """
        Store pre-generated embeddings in vector database with metadata
        
        Uses VectorStoreService for storage operations.
        """
        try:
            if not chunks or not embeddings:
                return
            
            if len(chunks) != len(embeddings):
                raise ValueError(
                    f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
                )
            
            # Ensure collection exists if needed
            if team_id:
                await self.vector_store.ensure_team_collection_exists(team_id)
            elif user_id:
                await self.vector_store.ensure_user_collection_exists(user_id)
            
            # Deduplicate chunks by content hash
            unique_chunks = []
            seen_hashes = set()
            for chunk in chunks:
                normalized_content = ' '.join(chunk.content.split()).lower()
                content_hash = hash(normalized_content)
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    unique_chunks.append(chunk)
            
            logger.info(
                f"Deduplicated {len(chunks)} chunks to {len(unique_chunks)} "
                f"unique chunks for {'team ' + team_id if team_id else ('user ' + user_id if user_id else 'system')}"
            )
            
            # Build mapping of unique chunks to embeddings
            chunk_to_embedding = {}
            for chunk, embedding in zip(chunks, embeddings):
                content_hash = abs(hash(chunk.content))
                chunk_to_embedding[content_hash] = embedding
            
            # Log metadata info
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
                logger.info(f"Including document metadata in vector payloads:{metadata_info}")
            
            # Prepare points for storage
            points = []
            for chunk in unique_chunks:
                content_hash = abs(hash(chunk.content))
                embedding = chunk_to_embedding.get(content_hash)
                
                if not embedding:
                    logger.warning(f"No embedding found for chunk {chunk.chunk_id}, skipping")
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
                    "user_id": user_id,
                    "team_id": team_id
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
                if is_image_sidecar:
                    payload["is_image_sidecar"] = "true"
                
                point = PointStruct(
                    id=content_hash,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            # Store via VectorStoreService
            collection_name = None
            if team_id:
                collection_name = self.vector_store._get_team_collection_name(team_id)
            elif user_id:
                collection_name = self.vector_store._get_user_collection_name(user_id)
            
            success = await self.vector_store.insert_points(
                points=points,
                collection_name=collection_name
            )
            
            if success:
                if team_id:
                    collection_type = "team"
                elif user_id:
                    collection_type = "user"
                else:
                    collection_type = "global"
                logger.info(
                    f"Stored {len(points)} unique embeddings in "
                    f"{collection_type} collection"
                )
                # Store chunks in PostgreSQL for full-text search (admin context for writes)
                pg_collection_type = "global" if not user_id and not team_id else "user"
                await self._store_chunks_postgres(
                    chunks=unique_chunks,
                    user_id=user_id,
                    team_id=team_id,
                    collection_type=pg_collection_type,
                    is_image_sidecar=is_image_sidecar,
                )
            else:
                logger.error("Failed to store embeddings")
            
        except Exception as e:
            logger.error(f"Failed to store embeddings with metadata for {'team ' + team_id if team_id else ('user ' + user_id if user_id else 'system')}: {e}")
            raise
    
    async def _store_chunks_postgres(
        self,
        chunks: List[Chunk],
        user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        collection_type: str = "user",
        is_image_sidecar: bool = False,
    ) -> None:
        """
        Persist chunks to document_chunks table for full-text search.
        Uses admin RLS context so backend can write regardless of current user.
        """
        if not chunks:
            return
        try:
            from services.database_manager.database_helpers import execute
            rls_context = {"user_id": "", "user_role": "admin"}
            # Remove existing chunks for this document (reprocess may have different count)
            document_id = chunks[0].document_id
            await execute(
                "DELETE FROM document_chunks WHERE document_id = $1",
                document_id,
                rls_context=rls_context,
            )
            # Batch insert with ON CONFLICT DO UPDATE for idempotent reprocessing
            for i in range(0, len(chunks), DOCUMENT_CHUNKS_INSERT_BATCH):
                batch = chunks[i : i + DOCUMENT_CHUNKS_INSERT_BATCH]
                placeholders = []
                args = []
                for idx, ch in enumerate(batch):
                    base = idx * 10
                    placeholders.append(
                        f"(${base+1}, ${base+2}, ${base+3}, ${base+4}, ${base+5}, ${base+6}, ${base+7}, ${base+8}, ${base+9}, ${base+10})"
                    )
                    args.extend([
                        ch.chunk_id,
                        ch.document_id,
                        ch.content,
                        ch.chunk_index,
                        user_id,
                        collection_type,
                        team_id,
                        is_image_sidecar,
                        ch.metadata.get("page_start"),
                        ch.metadata.get("page_end"),
                    ])
                values_sql = ", ".join(placeholders)
                sql = f"""
                    INSERT INTO document_chunks (chunk_id, document_id, content, chunk_index, user_id, collection_type, team_id, is_image_sidecar, page_start, page_end)
                    VALUES {values_sql}
                """
                await execute(sql, *args, rls_context=rls_context)
            logger.debug(f"Stored {len(chunks)} chunks in document_chunks for full-text search")
        except Exception as e:
            logger.error(f"Failed to store chunks in document_chunks: {e}")
            # Do not raise: vector store is source of truth; full-text is additive
    
    async def close(self):
        """Clean up resources"""
        try:
            if self.vector_service_client and hasattr(self.vector_service_client, 'close'):
                await self.vector_service_client.close()
                logger.info("Vector Service client closed")
            
            if self.vector_store and hasattr(self.vector_store, 'close'):
                await self.vector_store.close()
                logger.info("Vector Store Service closed")
            
            self._initialized = False
            logger.info("Embedding Service Wrapper closed")
        except Exception as e:
            logger.warning(f"Error closing Embedding Service Wrapper: {e}")


# Singleton instance
_embedding_service_wrapper: Optional[EmbeddingServiceWrapper] = None


async def get_embedding_service() -> EmbeddingServiceWrapper:
    """Get or create singleton embedding service wrapper"""
    global _embedding_service_wrapper
    
    if _embedding_service_wrapper is None:
        _embedding_service_wrapper = EmbeddingServiceWrapper()
        await _embedding_service_wrapper.initialize()
    
    return _embedding_service_wrapper
