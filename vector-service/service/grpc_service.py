"""
gRPC Service Implementation - Vector Service (Embedding Generation + vector store).
"""

import json
import logging
import sys
from typing import Any, Dict, List

import grpc

sys.path.insert(0, "/app")

from protos import vector_service_pb2, vector_service_pb2_grpc

from config.settings import settings
from service.backends.base import (
    CreateCollectionInput,
    SparseVectorData,
    VectorFilterInput,
    VectorPointInput,
)
from service.backends.factory import get_vector_backend
from service.embedding_cache import EmbeddingCache
from service.embedding_engine import EmbeddingEngine

logger = logging.getLogger(__name__)


def _vector_filters_from_proto(repeated_filters) -> List[VectorFilterInput]:
    out: List[VectorFilterInput] = []
    for vf in repeated_filters:
        out.append(
            VectorFilterInput(
                field=vf.field,
                value=vf.value,
                operator=vf.operator,
                values=list(vf.values) if vf.values else [],
            )
        )
    return out


def _payload_from_proto_map(payload_map) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in payload_map.items():
        try:
            payload[key] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            payload[key] = value
    return payload


def _payload_to_proto_map(payload: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, (list, dict)):
            out[key] = json.dumps(value)
        else:
            out[key] = str(value)
    return out


class VectorServiceImplementation(vector_service_pb2_grpc.VectorServiceServicer):
    """
    Vector Service gRPC Implementation: embeddings plus pluggable vector store (Qdrant, Milvus, or Elasticsearch/OpenSearch).
    """

    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
        db_path = None
        if settings.EMBEDDING_CACHE_ENABLED:
            p = (getattr(settings, "EMBEDDING_CACHE_DB_PATH", None) or "").strip()
            db_path = p if p else None
        self.embedding_cache = EmbeddingCache(
            ttl_seconds=settings.EMBEDDING_CACHE_TTL,
            db_path=db_path,
        )
        self._vector_backend = get_vector_backend(settings)
        self._initialized = False

    async def initialize(self):
        """Initialize embedding stack and vector store backend."""
        try:
            await self.embedding_engine.initialize()
            await self.embedding_cache.initialize()
            self._vector_backend.initialize()
            if not self._vector_backend.is_configured():
                logger.warning(
                    "Vector store is not configured for the selected backend; "
                    "vector store features may be unavailable"
                )

            self._initialized = True
            logger.info("Vector Service initialized successfully")
            logger.info("Service mode: Knowledge Hub (Embeddings + vector store)")
        except Exception as e:
            logger.error("Failed to initialize Vector Service: %s", e)
            raise

    def _vector_store_ready(self) -> bool:
        return bool(
            self._initialized
            and self._vector_backend is not None
            and self._vector_backend.is_available()
        )

    async def UpsertVectors(self, request, context):
        """Store arbitrary vectors (documents, faces, objects, etc.)"""
        try:
            if not self._vector_store_ready():
                return vector_service_pb2.UpsertVectorsResponse(
                    success=False,
                    points_stored=0,
                    error="Service or Qdrant not initialized",
                )

            if not request.points:
                return vector_service_pb2.UpsertVectorsResponse(
                    success=True,
                    points_stored=0,
                )

            points_in: List[VectorPointInput] = []
            for pb_point in request.points:
                payload = _payload_from_proto_map(pb_point.payload)
                sparse = None
                if pb_point.HasField("sparse_vector"):
                    sparse = SparseVectorData(
                        indices=list(pb_point.sparse_vector.indices),
                        values=list(pb_point.sparse_vector.values),
                    )
                points_in.append(
                    VectorPointInput(
                        id=pb_point.id,
                        vector=list(pb_point.vector),
                        payload=payload,
                        sparse=sparse,
                    )
                )

            total_stored = self._vector_backend.upsert_vectors(
                request.collection_name,
                points_in,
            )
            return vector_service_pb2.UpsertVectorsResponse(
                success=True,
                points_stored=total_stored,
            )

        except ValueError as e:
            logger.error("UpsertVectors schema mismatch: %s", e)
            return vector_service_pb2.UpsertVectorsResponse(
                success=False,
                points_stored=0,
                error=str(e),
            )
        except Exception as e:
            logger.error("UpsertVectors failed: %s", e)
            import traceback

            traceback.print_exc()
            return vector_service_pb2.UpsertVectorsResponse(
                success=False,
                points_stored=0,
                error=str(e),
            )

    async def SearchVectors(self, request, context):
        """Search for similar vectors across any collection"""
        try:
            if not self._vector_store_ready():
                return vector_service_pb2.SearchVectorsResponse(
                    success=False,
                    error="Service or Qdrant not initialized",
                )

            sparse_q = None
            if request.HasField("sparse_query_vector"):
                sparse_q = SparseVectorData(
                    indices=list(request.sparse_query_vector.indices),
                    values=list(request.sparse_query_vector.values),
                )

            hits = self._vector_backend.search_vectors(
                collection_name=request.collection_name,
                query_vector=list(request.query_vector),
                limit=request.limit or 50,
                score_threshold=request.score_threshold,
                filters=_vector_filters_from_proto(request.filters),
                sparse_query=sparse_q,
                fusion_mode=request.fusion_mode or "",
            )

            results = []
            for hit in hits:
                results.append(
                    vector_service_pb2.VectorSearchResult(
                        id=hit.id,
                        score=hit.score,
                        payload=_payload_to_proto_map(hit.payload),
                    )
                )

            return vector_service_pb2.SearchVectorsResponse(
                success=True,
                results=results,
            )

        except Exception as e:
            logger.error("SearchVectors failed: %s", e)
            import traceback

            traceback.print_exc()
            if (
                "404" in str(e)
                or "doesn't exist" in str(e).lower()
                or "not found" in str(e).lower()
            ):
                logger.info("SearchVectors: Collection not found, returning empty results")
                return vector_service_pb2.SearchVectorsResponse(
                    success=True,
                    results=[],
                )
            return vector_service_pb2.SearchVectorsResponse(
                success=False,
                error=str(e),
            )

    async def ScrollPoints(self, request, context):
        """Paginated payload scroll for audits (no embedding API calls)."""
        try:
            if not self._vector_store_ready():
                return vector_service_pb2.ScrollPointsResponse(
                    success=False,
                    error="Service or Qdrant not initialized",
                )

            page = self._vector_backend.scroll_points(
                collection_name=request.collection_name,
                filters=_vector_filters_from_proto(request.filters),
                limit=request.limit or 256,
                offset=(request.offset or "").strip(),
                with_vectors=bool(request.with_vectors),
            )
            out_points = [
                vector_service_pb2.ScrolledPoint(
                    id=p.id,
                    payload=_payload_to_proto_map(p.payload),
                )
                for p in page.points
            ]
            return vector_service_pb2.ScrollPointsResponse(
                success=True,
                points=out_points,
                next_offset=page.next_offset,
            )
        except Exception as e:
            logger.error("ScrollPoints failed: %s", e)
            import traceback

            traceback.print_exc()
            return vector_service_pb2.ScrollPointsResponse(success=False, error=str(e))

    async def CountVectors(self, request, context):
        """Count with optional filter (metadata-only)."""
        try:
            if not self._vector_store_ready():
                return vector_service_pb2.CountVectorsResponse(
                    success=False,
                    count=0,
                    error="Service or Qdrant not initialized",
                )

            n = self._vector_backend.count_vectors(
                request.collection_name,
                _vector_filters_from_proto(request.filters),
            )
            return vector_service_pb2.CountVectorsResponse(success=True, count=n)
        except Exception as e:
            logger.error("CountVectors failed: %s", e)
            return vector_service_pb2.CountVectorsResponse(
                success=False,
                count=0,
                error=str(e),
            )

    async def DeleteVectors(self, request, context):
        """Delete vectors by filter (e.g., delete all for document_id)"""
        try:
            if not self._vector_store_ready():
                return vector_service_pb2.DeleteVectorsResponse(
                    success=False,
                    points_deleted=0,
                    error="Service or Qdrant not initialized",
                )

            deleted, err = self._vector_backend.delete_vectors_equality(
                request.collection_name,
                _vector_filters_from_proto(request.filters),
            )
            if err:
                return vector_service_pb2.DeleteVectorsResponse(
                    success=False,
                    points_deleted=0,
                    error=err,
                )
            return vector_service_pb2.DeleteVectorsResponse(
                success=True,
                points_deleted=deleted,
            )

        except Exception as e:
            logger.error("DeleteVectors failed: %s", e)
            return vector_service_pb2.DeleteVectorsResponse(
                success=False,
                points_deleted=0,
                error=str(e),
            )

    async def UpdateVectorMetadata(self, request, context):
        """Update metadata for vectors matching filters"""
        try:
            if not self._vector_store_ready():
                return vector_service_pb2.UpdateVectorMetadataResponse(
                    success=False,
                    points_updated=0,
                    error="Service or Qdrant not initialized",
                )

            payload_updates: Dict[str, Any] = {}
            for key, value in request.metadata_updates.items():
                try:
                    payload_updates[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    payload_updates[key] = value

            updated, err = self._vector_backend.update_metadata_equality(
                request.collection_name,
                _vector_filters_from_proto(request.filters),
                payload_updates,
            )
            if err:
                return vector_service_pb2.UpdateVectorMetadataResponse(
                    success=False,
                    points_updated=0,
                    error=err,
                )
            return vector_service_pb2.UpdateVectorMetadataResponse(
                success=True,
                points_updated=updated,
            )

        except Exception as e:
            logger.error("UpdateVectorMetadata failed: %s", e)
            return vector_service_pb2.UpdateVectorMetadataResponse(
                success=False,
                points_updated=0,
                error=str(e),
            )

    async def CreateCollection(self, request, context):
        """Create a new collection with specified dimensions"""
        try:
            if not self._vector_store_ready():
                return vector_service_pb2.CreateCollectionResponse(
                    success=False,
                    error="Service or Qdrant not initialized",
                )

            err = self._vector_backend.create_collection(
                CreateCollectionInput(
                    collection_name=request.collection_name,
                    vector_size=request.vector_size,
                    distance=request.distance or "",
                    enable_sparse=bool(request.enable_sparse),
                )
            )
            if err:
                return vector_service_pb2.CreateCollectionResponse(
                    success=False,
                    error=err,
                )
            return vector_service_pb2.CreateCollectionResponse(success=True)

        except Exception as e:
            logger.error("CreateCollection failed: %s", e)
            return vector_service_pb2.CreateCollectionResponse(
                success=False,
                error=str(e),
            )

    async def DeleteCollection(self, request, context):
        """Delete a collection"""
        try:
            if not self._vector_store_ready():
                return vector_service_pb2.DeleteCollectionResponse(
                    success=False,
                    error="Service or Qdrant not initialized",
                )

            err = self._vector_backend.delete_collection(request.collection_name)
            if err:
                return vector_service_pb2.DeleteCollectionResponse(
                    success=False,
                    error=err,
                )
            return vector_service_pb2.DeleteCollectionResponse(success=True)

        except Exception as e:
            logger.error("DeleteCollection failed: %s", e)
            return vector_service_pb2.DeleteCollectionResponse(
                success=False,
                error=str(e),
            )

    async def ListCollections(self, request, context):
        """List all collections"""
        try:
            if not self._vector_store_ready():
                return vector_service_pb2.ListCollectionsResponse(
                    success=False,
                    error="Service or Qdrant not initialized",
                )

            infos, err = self._vector_backend.list_collections()
            if err:
                return vector_service_pb2.ListCollectionsResponse(
                    success=False,
                    error=err,
                )
            collection_infos = [
                vector_service_pb2.CollectionInfo(
                    name=c.name,
                    vector_size=c.vector_size,
                    distance=c.distance,
                    points_count=c.points_count,
                    status=c.status,
                    schema_type=c.schema_type,
                )
                for c in infos
            ]
            return vector_service_pb2.ListCollectionsResponse(
                success=True,
                collections=collection_infos,
            )

        except Exception as e:
            logger.error("ListCollections failed: %s", e)
            return vector_service_pb2.ListCollectionsResponse(
                success=False,
                error=str(e),
            )

    async def GetCollectionInfo(self, request, context):
        """Get information about a specific collection"""
        try:
            if not self._vector_store_ready():
                return vector_service_pb2.GetCollectionInfoResponse(
                    success=False,
                    error="Service or Qdrant not initialized",
                )

            res = self._vector_backend.get_collection_info(request.collection_name)
            if not res.success:
                return vector_service_pb2.GetCollectionInfoResponse(
                    success=False,
                    error=res.error or "Unknown error",
                )
            c = res.collection
            if c is None:
                return vector_service_pb2.GetCollectionInfoResponse(
                    success=False,
                    error="Unknown error",
                )
            collection_info = vector_service_pb2.CollectionInfo(
                name=c.name,
                vector_size=c.vector_size,
                distance=c.distance,
                points_count=c.points_count,
                status=c.status,
                schema_type=c.schema_type,
            )
            return vector_service_pb2.GetCollectionInfoResponse(
                success=True,
                collection=collection_info,
            )

        except Exception as e:
            logger.error("GetCollectionInfo failed: %s", e)
            err = str(e)
            if (
                "404" in err
                or "doesn't exist" in err.lower()
                or "not found" in err.lower()
            ):
                return vector_service_pb2.GetCollectionInfoResponse(
                    success=False,
                    error=f"Collection '{request.collection_name}' doesn't exist",
                )
            return vector_service_pb2.GetCollectionInfoResponse(
                success=False,
                error=err,
            )

    async def GenerateEmbedding(self, request, context):
        """Generate single embedding with cache lookup"""
        try:
            if not self._initialized:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Service not initialized")
                return vector_service_pb2.EmbeddingResponse()

            from_cache = False
            active_model = request.model or (self.embedding_engine.model or "")

            if settings.EMBEDDING_CACHE_ENABLED:
                content_hash = self.embedding_cache.hash_text(request.text, active_model)
                cached_embedding = await self.embedding_cache.get(content_hash)

                if cached_embedding:
                    logger.debug("Cache hit for embedding")
                    from_cache = True
                    return vector_service_pb2.EmbeddingResponse(
                        embedding=cached_embedding,
                        token_count=len(request.text.split()),
                        model=active_model,
                        from_cache=True,
                    )

            embedding = await self.embedding_engine.generate_embedding(request.text)

            if settings.EMBEDDING_CACHE_ENABLED:
                content_hash = self.embedding_cache.hash_text(request.text, active_model)
                await self.embedding_cache.set(content_hash, embedding)

            return vector_service_pb2.EmbeddingResponse(
                embedding=embedding,
                token_count=len(request.text.split()),
                model=active_model,
                from_cache=False,
            )

        except Exception as e:
            logger.error("GenerateEmbedding failed: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vector_service_pb2.EmbeddingResponse()

    async def GenerateBatchEmbeddings(self, request, context):
        """Generate batch embeddings with parallel processing and cache lookup"""
        try:
            if not self._initialized:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Service not initialized")
                return vector_service_pb2.BatchEmbeddingResponse()

            texts = list(request.texts)
            embeddings = []
            texts_to_generate = []
            text_indices = []
            cache_hits = 0
            cache_misses = 0
            active_model = request.model or (self.embedding_engine.model or "")

            if settings.EMBEDDING_CACHE_ENABLED:
                for idx, text in enumerate(texts):
                    content_hash = self.embedding_cache.hash_text(text, active_model)
                    cached_embedding = await self.embedding_cache.get(content_hash)

                    if cached_embedding:
                        embeddings.append((idx, cached_embedding, True))
                        cache_hits += 1
                    else:
                        texts_to_generate.append(text)
                        text_indices.append(idx)
                        cache_misses += 1
            else:
                texts_to_generate = texts
                text_indices = list(range(len(texts)))
                cache_misses = len(texts)

            if texts_to_generate:
                new_embeddings = await self.embedding_engine.generate_batch_embeddings(
                    texts=texts_to_generate,
                    batch_size=request.batch_size or settings.BATCH_SIZE,
                )

                if settings.EMBEDDING_CACHE_ENABLED:
                    for text, embedding, idx in zip(
                        texts_to_generate, new_embeddings, text_indices
                    ):
                        content_hash = self.embedding_cache.hash_text(text, active_model)
                        await self.embedding_cache.set(content_hash, embedding)
                        embeddings.append((idx, embedding, False))
                else:
                    for embedding, idx in zip(new_embeddings, text_indices):
                        embeddings.append((idx, embedding, False))

            embeddings.sort(key=lambda x: x[0])

            embedding_vectors = []
            for idx, (original_idx, embedding, from_cache) in enumerate(embeddings):
                embedding_vectors.append(
                    vector_service_pb2.EmbeddingVector(
                        vector=embedding,
                        index=idx,
                        token_count=len(texts[idx].split()),
                        from_cache=from_cache,
                    )
                )

            logger.info(
                "Batch embeddings: %s hits, %s misses",
                cache_hits,
                cache_misses,
            )

            return vector_service_pb2.BatchEmbeddingResponse(
                embeddings=embedding_vectors,
                total_tokens=sum(len(t.split()) for t in texts),
                model=active_model,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
            )

        except Exception as e:
            logger.error("GenerateBatchEmbeddings failed: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vector_service_pb2.BatchEmbeddingResponse()

    async def ClearEmbeddingCache(self, request, context):
        """Clear embedding cache"""
        try:
            if request.content_hash:
                cleared = await self.embedding_cache.clear(request.content_hash)
            else:
                cleared = await self.embedding_cache.clear()

            return vector_service_pb2.ClearCacheResponse(
                success=True,
                entries_cleared=cleared,
            )

        except Exception as e:
            logger.error("ClearEmbeddingCache failed: %s", e)
            return vector_service_pb2.ClearCacheResponse(
                success=False,
                error=str(e),
            )

    async def GetCacheStats(self, request, context):
        """Get cache statistics"""
        try:
            stats = self.embedding_cache.get_stats()

            return vector_service_pb2.CacheStatsResponse(
                embedding_cache_size=stats["size"],
                embedding_cache_hits=stats["hits"],
                embedding_cache_misses=stats["misses"],
                cache_hit_rate=stats["hit_rate"],
                ttl_seconds=stats["ttl_seconds"],
            )

        except Exception as e:
            logger.error("GetCacheStats failed: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vector_service_pb2.CacheStatsResponse()

    async def HealthCheck(self, request, context):
        """Health check endpoint"""
        try:
            openai_ok = (
                await self.embedding_engine.health_check() if self._initialized else False
            )

            status = "healthy" if (openai_ok and self._initialized) else "degraded"
            if not self._initialized:
                status = "unhealthy"

            cache_stats = self.embedding_cache.get_stats()
            details = {
                "cache_size": str(cache_stats["size"]),
                "cache_hit_rate": f"{cache_stats['hit_rate']:.2%}",
                "mode": "embedding_generation_only",
                "vector_db_backend": (settings.VECTOR_DB_BACKEND or "qdrant").lower(),
                "vector_store_configured": str(
                    bool(self._vector_backend and self._vector_backend.is_available())
                ).lower(),
            }
            if self._initialized and self.embedding_engine.provider:
                details["embedding_provider"] = self.embedding_engine.provider.provider_name
                details["model"] = self.embedding_engine.model or ""

            return vector_service_pb2.HealthCheckResponse(
                status=status,
                openai_available=openai_ok,
                service_version="1.0.0",
                details=details,
            )

        except Exception as e:
            logger.error("HealthCheck failed: %s", e)
            return vector_service_pb2.HealthCheckResponse(
                status="unhealthy",
                openai_available=False,
                service_version="1.0.0",
            )
