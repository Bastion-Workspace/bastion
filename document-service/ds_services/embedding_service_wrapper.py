"""
Embedding Service Wrapper

Unified interface for embedding generation and storage using Vector Service.
All embedding generation routed through dedicated Vector Service microservice.
All storage operations handled by VectorStoreService.
Full-text search: chunks are also stored in PostgreSQL document_chunks table.
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Callable, Awaitable

from ds_config import get_settings, settings
from ds_clients.vector_service_client import (
    get_vector_service_client,
    VectorUnavailableError,
)
from ds_services.vector_store_service import get_vector_store
from ds_models.api_models import Chunk
from ds_models.vector_point import VectorPoint

logger = logging.getLogger(__name__)

# Batch size for PostgreSQL chunk inserts
DOCUMENT_CHUNKS_INSERT_BATCH = 50

# Stable namespace for UUID5-based Qdrant point IDs.
# Using document_id + chunk_index ensures IDs are deterministic across
# processes and reprocessing runs, which prevents duplicate points in Qdrant.
_CHUNK_ID_NAMESPACE = uuid.UUID("b3d4f2a1-1234-5678-abcd-9f0e1a2b3c4d")


def _stable_content_hash(content: str) -> str:
    """Return a stable SHA-256 hex digest for chunk content deduplication.

    Unlike Python's built-in hash(), this is deterministic across processes
    and interpreter restarts (no PYTHONHASHSEED randomisation).
    """
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


@dataclass
class SidecarEmbedItem:
    """One sidecar row for batched embedding (startup scan)."""

    chunk: Chunk
    user_id: Optional[str]
    team_id: Optional[str] = None
    document_category: Optional[str] = None
    document_tags: Optional[List[str]] = None
    document_title: Optional[str] = None
    document_author: Optional[str] = None
    document_filename: Optional[str] = None
    is_image_sidecar: bool = False
    pending_image_completion: bool = False
    document_id: str = ""


def _deduplicate_chunks(chunks: List[Any]) -> List[Any]:
    unique_chunks: List[Any] = []
    seen_hashes: set = set()
    for chunk in chunks:
        normalized_content = " ".join(chunk.content.split()).lower()
        content_hash = _stable_content_hash(normalized_content)
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_chunks.append(chunk)
    return unique_chunks


def _stable_point_id(document_id: str, chunk_index: int) -> str:
    """Return a stable UUID string suitable as a Qdrant point ID.

    Uses UUID5 over the combination of document_id and chunk_index so the ID
    is fully deterministic: re-embedding the same document at the same position
    produces the same ID, allowing Qdrant upsert to overwrite rather than
    create duplicate points.
    """
    return str(uuid.uuid5(_CHUNK_ID_NAMESPACE, f"{document_id}:{chunk_index}"))


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

    def is_vector_stack_available(self) -> bool:
        if not getattr(settings, "VECTOR_EMBEDDING_ENABLED", True):
            return False
        if not self.vector_store:
            return False
        return bool(self.vector_store.is_vector_available())

    async def initialize(self):
        """Initialize Vector Service client and vector store"""
        if self._initialized:
            return

        self.vector_store = await get_vector_store()
        logger.debug("Vector Store Service initialized")

        logger.debug("Initializing Vector Service client for embeddings")
        self.vector_service_client = await get_vector_service_client(required=False)
        logger.info("Vector Service client initialized (may retry connection later)")

        self._initialized = True

        if getattr(settings, "VECTOR_EMBEDDING_REQUIRED", False) and not self.is_vector_stack_available():
            raise RuntimeError(
                "VECTOR_EMBEDDING_REQUIRED is set but vector stack is not available"
            )

    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        if not self._initialized:
            await self.initialize()

        if not self.vector_service_client:
            return []
        try:
            return await self.vector_service_client.generate_embeddings(
                texts=texts,
                model=model,
            )
        except VectorUnavailableError:
            return []

    async def replay_sync_document_vectors(
        self,
        document_id: str,
        user_id: Optional[str],
        payload: Dict[str, Any],
    ) -> bool:
        from ds_db.database_manager.database_helpers import fetch_all
        from ds_models.api_models import Chunk

        team_id = payload.get("team_id")
        rows = await fetch_all(
            """
            SELECT chunk_id, document_id, content, chunk_index, user_id, team_id,
                   is_image_sidecar, page_start, page_end
            FROM document_chunks
            WHERE document_id = $1
            ORDER BY chunk_index ASC
            """,
            document_id,
            rls_context=None,
        )
        if not rows:
            logger.warning("replay_sync_document_vectors: no document_chunks for %s", document_id)
            return False
        chunks: List[Chunk] = []
        for r in rows:
            meta: Dict[str, Any] = {}
            if r.get("page_start") is not None:
                meta["page_start"] = r["page_start"]
            if r.get("page_end") is not None:
                meta["page_end"] = r["page_end"]
            chunks.append(
                Chunk(
                    chunk_id=r["chunk_id"],
                    document_id=r["document_id"],
                    content=r["content"] or "",
                    chunk_index=int(r["chunk_index"] or 0),
                    quality_score=1.0,
                    method="backlog_replay",
                    metadata=meta,
                )
            )
        uid = user_id or (rows[0].get("user_id") if rows else None)
        tid = team_id if team_id is not None else rows[0].get("team_id")
        is_image_sidecar = bool(rows[0].get("is_image_sidecar"))
        unique_chunks = _deduplicate_chunks(chunks)
        texts = [c.content for c in unique_chunks]
        try:
            embeddings = await self.vector_service_client.generate_embeddings(texts)
        except VectorUnavailableError:
            return False
        except Exception as e:
            logger.warning("replay_sync_document_vectors embed failed: %s", e)
            return False
        chunk_to_embedding = {
            _stable_content_hash(ch.content): e for ch, e in zip(unique_chunks, embeddings)
        }
        return await self._upsert_vectors_from_embeddings(
            unique_chunks=unique_chunks,
            chunk_to_embedding=chunk_to_embedding,
            user_id=uid,
            team_id=tid,
            document_category=payload.get("document_category"),
            document_tags=payload.get("document_tags"),
            document_title=payload.get("document_title"),
            document_author=payload.get("document_author"),
            document_filename=payload.get("document_filename"),
            is_image_sidecar=is_image_sidecar,
        )

    async def replay_delete_document_vectors(
        self,
        document_id: str,
        user_id: Optional[str],
        payload: Dict[str, Any],
    ) -> bool:
        team_id = (payload or {}).get("team_id")
        collection_name = None
        if team_id:
            collection_name = self.vector_store._get_team_collection_name(str(team_id))
        return bool(
            await self.vector_store.delete_points_by_filter(
                document_id,
                user_id,
                collection_name=collection_name,
            )
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
        embed_progress: Optional[Callable[[int, int], Awaitable[None]]] = None,
    ):
        if not self._initialized:
            await self.initialize()
        if not chunks:
            return

        from ds_services import vector_embed_backlog as vb

        unique_chunks = _deduplicate_chunks(chunks)
        if team_id:
            pg_collection_type = "team"
        elif user_id:
            pg_collection_type = "user"
        else:
            pg_collection_type = "global"

        if embed_progress and chunks:
            await embed_progress(0, len(chunks))

        await self._store_chunks_postgres(
            unique_chunks,
            user_id=user_id,
            team_id=team_id,
            collection_type=pg_collection_type,
            is_image_sidecar=is_image_sidecar,
            embed_progress=embed_progress,
        )

        if not getattr(settings, "VECTOR_EMBEDDING_ENABLED", True):
            return

        doc_id = unique_chunks[0].document_id
        extra = {
            "document_category": document_category,
            "document_tags": document_tags,
            "document_title": document_title,
            "document_author": document_author,
            "document_filename": document_filename,
        }

        texts = [c.content for c in unique_chunks]
        try:
            embeddings = await self.vector_service_client.generate_embeddings(texts)
        except (VectorUnavailableError, Exception) as e:
            logger.warning("Embedding generation deferred to backlog: %s", e)
            await vb.enqueue_sync_document_vectors(doc_id, user_id, team_id, extra=extra)
            return

        if len(embeddings) != len(unique_chunks):
            logger.error(
                "embed_and_store_chunks: embedding count mismatch (%s vs %s)",
                len(embeddings),
                len(unique_chunks),
            )
            await vb.enqueue_sync_document_vectors(doc_id, user_id, team_id, extra=extra)
            return

        chunk_to_embedding = {
            _stable_content_hash(c.content): e for c, e in zip(unique_chunks, embeddings)
        }
        ok = await self._upsert_vectors_from_embeddings(
            unique_chunks=unique_chunks,
            chunk_to_embedding=chunk_to_embedding,
            user_id=user_id,
            team_id=team_id,
            document_category=document_category,
            document_tags=document_tags,
            document_title=document_title,
            document_author=document_author,
            document_filename=document_filename,
            is_image_sidecar=is_image_sidecar,
        )
        if not ok:
            await vb.enqueue_sync_document_vectors(doc_id, user_id, team_id, extra=extra)

    async def embed_and_store_sidecar_batch(self, items: List[SidecarEmbedItem]) -> int:
        """One embedding RPC for many sidecar texts; one store call per item (metadata differs)."""
        if not self._initialized:
            await self.initialize()
        if not items:
            return 0
        if not getattr(settings, "VECTOR_EMBEDDING_ENABLED", True):
            return 0
        texts = [it.chunk.content for it in items]
        try:
            embeddings = await self.vector_service_client.generate_embeddings(texts)
        except (VectorUnavailableError, Exception) as e:
            logger.warning("embed_and_store_sidecar_batch: embeddings unavailable: %s", e)
            return 0
        if len(embeddings) != len(items):
            logger.error(
                "embed_and_store_sidecar_batch: embedding count mismatch (%s vs %s)",
                len(embeddings),
                len(items),
            )
            return 0
        stored = 0
        for it, emb in zip(items, embeddings):
            await self._store_embeddings_with_metadata(
                chunks=[it.chunk],
                embeddings=[emb],
                user_id=it.user_id,
                team_id=it.team_id,
                document_category=it.document_category,
                document_tags=it.document_tags,
                document_title=it.document_title,
                document_author=it.document_author,
                document_filename=it.document_filename,
                is_image_sidecar=it.is_image_sidecar,
            )
            stored += 1
        return stored
    
    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        user_id: Optional[str] = None,
        team_ids: Optional[List[str]] = None,
        filter_category: Optional[str] = None,
        filter_tags: Optional[List[str]] = None,
        shared_collection_scopes: Optional[List[Tuple[str, List[str]]]] = None,
        query_text: Optional[str] = None,
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
            query_text: Raw query text for BM25 sparse search (hybrid mode)
            
        Returns:
            List of search results
        """
        if not self._initialized:
            await self.initialize()

        if not self.is_vector_stack_available():
            return []

        return await self.vector_store.search_similar(
            query_embedding=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            user_id=user_id,
            team_ids=team_ids,
            filter_category=filter_category,
            filter_tags=filter_tags,
            shared_collection_scopes=shared_collection_scopes,
            query_text=query_text,
        )
    
    async def clear_cache(self):
        """Clear Vector Service embedding cache"""
        if not self._initialized:
            await self.initialize()
        
        if self.vector_service_client and self.vector_service_client.is_ready():
            await self.vector_service_client.clear_cache(clear_all=True)
            logger.info("Vector Service cache cleared")
    
    async def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get Vector Service cache statistics"""
        if not self._initialized:
            await self.initialize()
        
        if self.vector_service_client and self.vector_service_client.is_ready():
            return await self.vector_service_client.get_cache_stats()
        return None
    
    async def delete_document_chunks(
        self,
        document_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Remove document from all search indexes: Qdrant vectors and PostgreSQL
        document_chunks (full-text search). Optionally zeros chunk_count on
        document_metadata.
        """
        if not self._initialized:
            await self.initialize()

        team_id = None
        try:
            from ds_db.database_manager.database_helpers import fetch_one

            row = await fetch_one(
                "SELECT team_id FROM document_metadata WHERE document_id = $1",
                document_id,
                rls_context={"user_id": "", "user_role": "admin"},
            )
            if row and row.get("team_id"):
                team_id = row["team_id"]
        except Exception:
            pass

        collection_name = None
        if team_id:
            collection_name = self.vector_store._get_team_collection_name(str(team_id))

        result = await self.vector_store.delete_points_by_filter(
            document_id=document_id,
            user_id=user_id,
            collection_name=collection_name,
        )
        if not result and getattr(settings, "VECTOR_EMBEDDING_ENABLED", True):
            from ds_services import vector_embed_backlog as vb

            await vb.enqueue_delete_document_vectors(
                document_id, user_id or "", team_id=str(team_id) if team_id else None
            )

        # Remove from PostgreSQL full-text search index (admin context for writes)
        try:
            from ds_db.database_manager.database_helpers import execute
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
            await execute(
                """
                UPDATE document_metadata
                SET chunk_indexed_at = NULL,
                    chunk_indexed_file_hash = NULL,
                    chunk_index_schema_version = 0,
                    updated_at = CURRENT_TIMESTAMP
                WHERE document_id = $1
                """,
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
    
    async def _upsert_vectors_from_embeddings(
        self,
        unique_chunks: List[Chunk],
        chunk_to_embedding: Dict[str, List[float]],
        user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        document_category: Optional[str] = None,
        document_tags: Optional[List[str]] = None,
        document_title: Optional[str] = None,
        document_author: Optional[str] = None,
        document_filename: Optional[str] = None,
        is_image_sidecar: bool = False,
    ) -> bool:
        try:
            if not unique_chunks:
                return True
            if team_id:
                await self.vector_store.ensure_team_collection_exists(team_id)
            elif user_id:
                await self.vector_store.ensure_user_collection_exists(user_id)

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
                logger.info("Including document metadata in vector payloads:%s", metadata_info)

            points: List[VectorPoint] = []
            for chunk in unique_chunks:
                embedding = chunk_to_embedding.get(_stable_content_hash(chunk.content))
                if not embedding:
                    logger.warning("No embedding found for chunk %s, skipping", chunk.chunk_id)
                    continue
                point_id = _stable_point_id(chunk.document_id, chunk.chunk_index)
                payload: Dict[str, Any] = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "quality_score": chunk.quality_score,
                    "method": chunk.method,
                    "metadata": chunk.metadata,
                    "user_id": user_id,
                    "team_id": team_id,
                }
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
                points.append(
                    VectorPoint(id=point_id, vector=embedding, payload=payload)
                )

            sparse_vectors_list = None
            if getattr(settings, "HYBRID_SEARCH_ENABLED", False):
                try:
                    from ds_services.bm25_encoder import get_default_bm25_encoder

                    encoder = get_default_bm25_encoder()
                    sparse_vectors_list = []
                    for chunk in unique_chunks:
                        sv = encoder.encode(chunk.content)
                        sparse_vectors_list.append(
                            sv if sv and sv.get("indices") else None
                        )
                except Exception as e:
                    logger.warning(
                        "BM25 encoding failed for chunks, continuing without sparse vectors: %s",
                        e,
                    )
                    sparse_vectors_list = None

            collection_name = None
            if team_id:
                collection_name = self.vector_store._get_team_collection_name(team_id)
            elif user_id:
                collection_name = self.vector_store._get_user_collection_name(user_id)

            success = await self.vector_store.insert_points(
                points=points,
                collection_name=collection_name,
                sparse_vectors=sparse_vectors_list,
            )
            if success:
                scope = (
                    "team " + str(team_id)
                    if team_id
                    else ("user " + str(user_id) if user_id else "system")
                )
                logger.info(
                    "Stored %s unique embeddings in %s collection",
                    len(points),
                    scope,
                )
            else:
                logger.error("Failed to store embeddings in vector database")
            return bool(success)
        except Exception as e:
            logger.error("_upsert_vectors_from_embeddings failed: %s", e)
            return False

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
        embed_progress: Optional[Callable[[int, int], Awaitable[None]]] = None,
    ):
        from ds_services import vector_embed_backlog as vb

        if not chunks or not embeddings:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
            )

        unique_chunks = _deduplicate_chunks(chunks)
        logger.info(
            "Deduplicated %s chunks to %s unique chunks for %s",
            len(chunks),
            len(unique_chunks),
            "team " + str(team_id)
            if team_id
            else ("user " + str(user_id) if user_id else "system"),
        )

        chunk_to_embedding: Dict[str, List[float]] = {}
        for chunk, embedding in zip(chunks, embeddings):
            chunk_to_embedding[_stable_content_hash(chunk.content)] = embedding

        if team_id:
            pg_collection_type = "team"
        elif user_id:
            pg_collection_type = "user"
        else:
            pg_collection_type = "global"

        await self._store_chunks_postgres(
            unique_chunks,
            user_id=user_id,
            team_id=team_id,
            collection_type=pg_collection_type,
            is_image_sidecar=is_image_sidecar,
            embed_progress=embed_progress,
        )

        ok = await self._upsert_vectors_from_embeddings(
            unique_chunks=unique_chunks,
            chunk_to_embedding=chunk_to_embedding,
            user_id=user_id,
            team_id=team_id,
            document_category=document_category,
            document_tags=document_tags,
            document_title=document_title,
            document_author=document_author,
            document_filename=document_filename,
            is_image_sidecar=is_image_sidecar,
        )
        if not ok and getattr(settings, "VECTOR_EMBEDDING_ENABLED", True):
            doc_id = unique_chunks[0].document_id
            extra = {
                "document_category": document_category,
                "document_tags": document_tags,
                "document_title": document_title,
                "document_author": document_author,
                "document_filename": document_filename,
            }
            await vb.enqueue_sync_document_vectors(doc_id, user_id, team_id, extra=extra)

    async def _store_chunks_postgres(
        self,
        chunks: List[Chunk],
        user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        collection_type: str = "user",
        is_image_sidecar: bool = False,
        embed_progress: Optional[Callable[[int, int], Awaitable[None]]] = None,
    ) -> None:
        """
        Persist chunks to document_chunks table for full-text search.
        Uses admin RLS context so backend can write regardless of current user.
        """
        if not chunks:
            return
        try:
            from ds_db.database_manager.database_helpers import execute
            rls_context = {"user_id": "", "user_role": "admin"}
            # Remove existing chunks for this document (reprocess may have different count)
            document_id = chunks[0].document_id
            await execute(
                "DELETE FROM document_chunks WHERE document_id = $1",
                document_id,
                rls_context=rls_context,
            )
            total = len(chunks)
            # Batch insert with ON CONFLICT DO UPDATE for idempotent reprocessing
            for i in range(0, total, DOCUMENT_CHUNKS_INSERT_BATCH):
                batch = chunks[i : i + DOCUMENT_CHUNKS_INSERT_BATCH]
                placeholders = []
                args = []
                for idx, ch in enumerate(batch):
                    base = idx * 11
                    placeholders.append(
                        f"(${base+1}, ${base+2}, ${base+3}, ${base+4}, ${base+5}, ${base+6}, ${base+7}, ${base+8}, ${base+9}, ${base+10}, ${base+11})"
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
                        _stable_point_id(ch.document_id, ch.chunk_index),
                    ])
                values_sql = ", ".join(placeholders)
                sql = f"""
                    INSERT INTO document_chunks (chunk_id, document_id, content, chunk_index, user_id, collection_type, team_id, is_image_sidecar, page_start, page_end, qdrant_point_id)
                    VALUES {values_sql}
                """
                await execute(sql, *args, rls_context=rls_context)
                if embed_progress and total:
                    done = min(i + len(batch), total)
                    await embed_progress(done, total)
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
