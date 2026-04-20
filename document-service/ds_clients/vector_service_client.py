"""
Vector Service gRPC Client

Provides client interface to the Vector Service for embedding generation.
"""

import grpc
import asyncio
import logging
from typing import List, Dict, Any, Optional
import hashlib

from ds_config import get_settings
from protos import vector_service_pb2, vector_service_pb2_grpc

logger = logging.getLogger(__name__)


class VectorUnavailableError(RuntimeError):
    """Raised when embedding RPC cannot run because the vector service is not connected."""


class VectorServiceClient:
    """Client for interacting with the Vector Service via gRPC"""
    
    def __init__(self, service_url: Optional[str] = None):
        """
        Initialize Vector Service client
        
        Args:
            service_url: gRPC service URL (default: from config)
        """
        self.settings = get_settings()
        self.service_url = service_url or self.settings.VECTOR_SERVICE_URL
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[vector_service_pb2_grpc.VectorServiceStub] = None
        self._initialized = False
        self._embedding_disabled_logged = False

        # Semaphore to limit concurrent delete operations (prevent overwhelming vector service)
        self._delete_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent deletions

    def _embedding_enabled(self) -> bool:
        return bool(getattr(self.settings, "VECTOR_EMBEDDING_ENABLED", True))

    def is_ready(self) -> bool:
        return bool(
            self._embedding_enabled() and self._initialized and self.stub is not None
        )

    async def _reset_connection(self) -> None:
        if self.channel:
            try:
                await self.channel.close()
            except Exception:
                pass
        self.channel = None
        self.stub = None
        self._initialized = False

    async def try_reconnect(self) -> bool:
        if not self._embedding_enabled():
            return False
        if self.is_ready():
            return True
        await self._reset_connection()
        await self.initialize(required=False)
        return self.is_ready()

    async def initialize(self, required: bool = False):
        """Initialize the gRPC channel and stub

        Args:
            required: If True, raise exception on failure. If False, log warning and continue.
        """
        if not self._embedding_enabled():
            if required:
                raise RuntimeError(
                    "Vector embedding is disabled (VECTOR_EMBEDDING_ENABLED=false) but a required "
                    "connection was requested"
                )
            if not self._embedding_disabled_logged:
                logger.info(
                    "Vector gRPC client disabled by configuration (VECTOR_EMBEDDING_ENABLED=false)"
                )
                self._embedding_disabled_logged = True
            await self._reset_connection()
            return

        if self._initialized:
            return

        await self._reset_connection()

        try:
            logger.debug("Connecting to Vector Service at %s", self.service_url)

            options = [
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 300000),
                ("grpc.keepalive_timeout_ms", 20000),
                ("grpc.keepalive_permit_without_calls", 1),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.http2.min_time_between_pings_ms", 60000),
                ("grpc.http2.min_ping_interval_without_data_ms", 300000),
            ]
            self.channel = grpc.aio.insecure_channel(self.service_url, options=options)
            self.stub = vector_service_pb2_grpc.VectorServiceStub(self.channel)

            health_request = vector_service_pb2.HealthCheckRequest()
            response = await self.stub.HealthCheck(health_request, timeout=5.0)

            if response.status == "healthy":
                logger.info(
                    "Connected to Vector Service v%s (openai_available=%s)",
                    response.service_version,
                    response.openai_available,
                )
                self._initialized = True
            else:
                msg = f"Vector Service health check returned: {response.status}"
                logger.warning(msg)
                await self._reset_connection()
                if required:
                    raise RuntimeError(msg)

        except Exception as e:
            logger.error("Failed to connect to Vector Service: %s", e)
            await self._reset_connection()
            if required:
                raise
            logger.warning(
                "Vector Service unavailable; embedding calls will fail until reconnect succeeds"
            )

    async def close(self):
        """Close the gRPC channel"""
        if self.channel:
            await self.channel.close()
        self.channel = None
        self.stub = None
        self._initialized = False
        logger.info("Vector Service client closed")
    
    async def generate_embedding(
        self, 
        text: str, 
        model: Optional[str] = None
    ) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            model: Model name (default: from service)
            
        Returns:
            List of floats representing the embedding vector
        """
        if not self.is_ready():
            await self.initialize(required=False)
        if not self.is_ready():
            req = getattr(get_settings(), "VECTOR_EMBEDDING_REQUIRED", False)
            if req:
                raise RuntimeError("Vector Service is not available")
            raise VectorUnavailableError("Vector Service is not available")

        try:
            request = vector_service_pb2.EmbeddingRequest(
                text=text,
                model=model or ""
            )
            
            response = await self.stub.GenerateEmbedding(request, timeout=30.0)
            return list(response.embedding)  # Single embedding uses 'embedding' field
            
        except grpc.RpcError as e:
            logger.error(f"❌ gRPC error generating embedding: {e.code()}: {e.details()}")
            # Mark as uninitialized so next call will retry connection
            self._initialized = False
            raise
        except Exception as e:
            logger.error(f"❌ Error generating embedding: {e}")
            raise
    
    async def generate_embeddings(
        self, 
        texts: List[str], 
        model: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            model: Model name (default: from service)
            batch_size: Batch size for processing (default: from service)
            
        Returns:
            List of embedding vectors
        """
        if not self.is_ready():
            await self.initialize(required=False)
        if not self.is_ready():
            req = getattr(get_settings(), "VECTOR_EMBEDDING_REQUIRED", False)
            if req:
                raise RuntimeError("Vector Service is not available")
            raise VectorUnavailableError("Vector Service is not available")

        try:
            request = vector_service_pb2.BatchEmbeddingRequest(
                texts=texts,
                model=model or "",
                batch_size=batch_size or 0
            )
            
            response = await self.stub.GenerateBatchEmbeddings(request, timeout=60.0)
            return [list(emb.vector) for emb in response.embeddings]
            
        except grpc.RpcError as e:
            logger.error(f"❌ gRPC error generating batch embeddings: {e.code()}: {e.details()}")
            # Mark as uninitialized so next call will retry connection
            self._initialized = False
            raise
        except Exception as e:
            logger.error(f"❌ Error generating batch embeddings: {e}")
            raise
    
    async def clear_cache(
        self, 
        clear_all: bool = False, 
        content_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Clear embedding cache
        
        Args:
            clear_all: Clear all cache entries
            content_hash: Clear specific hash entry
            
        Returns:
            Dictionary with success status and entries cleared
        """
        if not self.is_ready():
            await self.initialize(required=False)
        if not self.is_ready():
            return {"success": False, "entries_cleared": 0, "error": "vector service unavailable"}

        try:
            request = vector_service_pb2.ClearCacheRequest(
                clear_all=clear_all,
                content_hash=content_hash or ""
            )

            response = await self.stub.ClearEmbeddingCache(request, timeout=10.0)
            return {
                "success": response.success,
                "entries_cleared": response.entries_cleared,
                "error": response.error if response.error else None
            }
            
        except grpc.RpcError as e:
            logger.error(f"❌ gRPC error clearing cache: {e.code()}: {e.details()}")
            raise
        except Exception as e:
            logger.error(f"❌ Error clearing cache: {e}")
            raise
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            request = vector_service_pb2.CacheStatsRequest()
            response = await self.stub.GetCacheStats(request, timeout=5.0)
            
            return {
                "cache_size": response.embedding_cache_size,
                "cache_hits": response.embedding_cache_hits,
                "cache_misses": response.embedding_cache_misses,
                "hit_rate": response.cache_hit_rate,
                "ttl_seconds": response.ttl_seconds
            }
            
        except grpc.RpcError as e:
            logger.error(f"❌ gRPC error getting cache stats: {e.code()}: {e.details()}")
            raise
        except Exception as e:
            logger.error(f"❌ Error getting cache stats: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check service health
        
        Returns:
            Dictionary with health status
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            request = vector_service_pb2.HealthCheckRequest()
            response = await self.stub.HealthCheck(request, timeout=5.0)
            
            return {
                "status": response.status,
                "openai_available": response.openai_available,
                "version": response.service_version,
                "details": dict(response.details) if response.details else {}
            }
            
        except grpc.RpcError as e:
            logger.error(f"❌ gRPC error checking health: {e.code()}: {e.details()}")
            raise
        except Exception as e:
            logger.error(f"❌ Error checking health: {e}")
            raise
    
    async def upsert_vectors(
        self,
        collection_name: str,
        points: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Store vectors in Qdrant via Vector Service
        
        Args:
            collection_name: Target collection name
            points: List of point dicts with 'id', 'vector', 'payload' keys
            
        Returns:
            Dict with success, points_stored, error
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            import json
            
            # Convert points to VectorPoint messages
            vector_points = []
            for point in points:
                # Convert payload to map<string, string> (JSON encode complex types)
                payload_map = {}
                for key, value in point.get("payload", {}).items():
                    if isinstance(value, (list, dict)):
                        payload_map[key] = json.dumps(value)
                    else:
                        payload_map[key] = str(value)
                
                vp = vector_service_pb2.VectorPoint(
                    id=str(point.get("id", "")),
                    vector=point.get("vector", []),
                    payload=payload_map,
                )

                sparse = point.get("sparse_vector")
                if sparse and sparse.get("indices"):
                    vp.sparse_vector.CopyFrom(
                        vector_service_pb2.SparseVector(
                            indices=sparse["indices"],
                            values=sparse["values"],
                        )
                    )

                vector_points.append(vp)
            
            request = vector_service_pb2.UpsertVectorsRequest(
                collection_name=collection_name,
                points=vector_points
            )
            
            response = await self.stub.UpsertVectors(request, timeout=60.0)
            
            return {
                "success": response.success,
                "points_stored": response.points_stored,
                "error": response.error if response.HasField("error") else None
            }
            
        except grpc.RpcError as e:
            logger.error(f"UpsertVectors failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "points_stored": 0,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error in upsert_vectors: {e}")
            return {
                "success": False,
                "points_stored": 0,
                "error": str(e)
            }
    
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 50,
        score_threshold: float = 0.7,
        filters: List[Dict[str, str]] = None,
        sparse_query_vector: Optional[Dict[str, Any]] = None,
        fusion_mode: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Search vectors via Vector Service
        
        Args:
            collection_name: Collection to search
            query_vector: Query embedding vector
            limit: Maximum results
            score_threshold: Minimum similarity score
            filters: List of filter dicts with 'field', 'value', 'operator' keys
            sparse_query_vector: Optional BM25 sparse vector {"indices": [...], "values": [...]}
            fusion_mode: "rrf" to enable Reciprocal Rank Fusion with sparse+dense
            
        Returns:
            List of search result dicts with 'id', 'score', 'payload'
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            import json
            
            # Convert filters to VectorFilter messages
            vector_filters = []
            if filters:
                for f in filters:
                    vf = vector_service_pb2.VectorFilter(
                        field=f.get("field", ""),
                        value=f.get("value", ""),
                        operator=f.get("operator", "equals")
                    )
                    if f.get("values") is not None:
                        vf.values.extend(f.get("values"))
                    vector_filters.append(vf)
            
            request = vector_service_pb2.SearchVectorsRequest(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                filters=vector_filters,
            )

            if sparse_query_vector and sparse_query_vector.get("indices"):
                request.sparse_query_vector.CopyFrom(
                    vector_service_pb2.SparseVector(
                        indices=sparse_query_vector["indices"],
                        values=sparse_query_vector["values"],
                    )
                )
            if fusion_mode:
                request.fusion_mode = fusion_mode
            
            response = await self.stub.SearchVectors(request, timeout=30.0)
            
            if not response.success:
                error_msg = response.error if response.HasField("error") else "Search failed"
                logger.error(f"SearchVectors failed: {error_msg}")
                return []
            
            # Convert results to dicts
            results = []
            for result in response.results:
                # Parse payload - handle JSON-encoded complex types
                payload = {}
                for key, value in result.payload.items():
                    try:
                        parsed = json.loads(value)
                        payload[key] = parsed
                    except (json.JSONDecodeError, TypeError):
                        payload[key] = value
                
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": payload
                })
            
            return results
            
        except grpc.RpcError as e:
            logger.error(f"SearchVectors failed: {e.code()} - {e.details()}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in search_vectors: {e}")
            return []

    async def scroll_points(
        self,
        collection_name: str,
        filters: Optional[List[Dict[str, Any]]] = None,
        limit: int = 256,
        offset: Optional[str] = None,
        with_vectors: bool = False,
    ) -> Dict[str, Any]:
        """Paginated scroll with payload (metadata-only by default)."""
        if not self._initialized:
            await self.initialize()

        try:
            import json

            vector_filters = []
            if filters:
                for f in filters:
                    vf = vector_service_pb2.VectorFilter(
                        field=f.get("field", ""),
                        value=f.get("value", ""),
                        operator=f.get("operator", "equals"),
                    )
                    if f.get("values") is not None:
                        vf.values.extend(f.get("values"))
                    vector_filters.append(vf)

            request = vector_service_pb2.ScrollPointsRequest(
                collection_name=collection_name,
                limit=limit,
                with_vectors=with_vectors,
                filters=vector_filters,
            )
            if offset:
                request.offset = offset

            response = await self.stub.ScrollPoints(request, timeout=300.0)
            points_out = []
            for sp in response.points:
                payload = {}
                for key, value in sp.payload.items():
                    try:
                        parsed = json.loads(value)
                        payload[key] = parsed
                    except (json.JSONDecodeError, TypeError):
                        payload[key] = value
                points_out.append({"id": sp.id, "payload": payload})

            return {
                "success": response.success,
                "points": points_out,
                "next_offset": response.next_offset or "",
                "error": response.error if response.HasField("error") else None,
            }
        except grpc.RpcError as e:
            logger.error("ScrollPoints failed: %s - %s", e.code(), e.details())
            return {
                "success": False,
                "points": [],
                "next_offset": "",
                "error": str(e),
            }
        except Exception as e:
            logger.error("Unexpected error in scroll_points: %s", e)
            return {
                "success": False,
                "points": [],
                "next_offset": "",
                "error": str(e),
            }

    async def count_vectors(
        self,
        collection_name: str,
        filters: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Qdrant count with optional filter."""
        if not self._initialized:
            await self.initialize()

        try:
            vector_filters = []
            if filters:
                for f in filters:
                    vf = vector_service_pb2.VectorFilter(
                        field=f.get("field", ""),
                        value=f.get("value", ""),
                        operator=f.get("operator", "equals"),
                    )
                    if f.get("values") is not None:
                        vf.values.extend(f.get("values"))
                    vector_filters.append(vf)

            request = vector_service_pb2.CountVectorsRequest(
                collection_name=collection_name,
                filters=vector_filters,
            )
            response = await self.stub.CountVectors(request, timeout=120.0)
            return {
                "success": response.success,
                "count": int(response.count),
                "error": response.error if response.HasField("error") else None,
            }
        except grpc.RpcError as e:
            logger.error("CountVectors failed: %s - %s", e.code(), e.details())
            return {"success": False, "count": 0, "error": str(e)}
        except Exception as e:
            logger.error("Unexpected error in count_vectors: %s", e)
            return {"success": False, "count": 0, "error": str(e)}
    
    async def delete_vectors(
        self,
        collection_name: str,
        filters: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Delete vectors by filter via Vector Service with concurrency control
        
        Args:
            collection_name: Collection to delete from
            filters: List of filter dicts with 'field', 'value', 'operator' keys
            
        Returns:
            Dict with success, points_deleted, error
        """
        if not self._initialized:
            await self.initialize()
        
        # Use semaphore to limit concurrent deletions and prevent overwhelming vector service
        async with self._delete_semaphore:
            try:
                # Convert filters to VectorFilter messages
                vector_filters = []
                for f in filters:
                    vector_filters.append(
                        vector_service_pb2.VectorFilter(
                            field=f.get("field", ""),
                            value=f.get("value", ""),
                            operator=f.get("operator", "equals")
                        )
                    )
                
                request = vector_service_pb2.DeleteVectorsRequest(
                    collection_name=collection_name,
                    filters=vector_filters
                )
                
                # Increase timeout to 120s for bulk operations
                # Add retry logic for CANCELLED errors
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = await self.stub.DeleteVectors(request, timeout=120.0)
                        
                        return {
                            "success": response.success,
                            "points_deleted": response.points_deleted,
                            "error": response.error if response.HasField("error") else None
                        }
                    
                    except grpc.RpcError as e:
                        # Retry on CANCELLED errors (typically due to service being overwhelmed)
                        if e.code() == grpc.StatusCode.CANCELLED and attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                            logger.warning(f"DeleteVectors cancelled, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        # Final failure or non-retryable error
                        logger.error(f"DeleteVectors failed: {e.code()} - {e.details()}")
                        return {
                            "success": False,
                            "points_deleted": 0,
                            "error": f"{e.code()}: {e.details()}"
                        }
                
                # Shouldn't reach here, but just in case
                return {
                    "success": False,
                    "points_deleted": 0,
                    "error": "Max retries exceeded"
                }
                
            except Exception as e:
                logger.error(f"Unexpected error in delete_vectors: {e}")
                return {
                    "success": False,
                    "points_deleted": 0,
                    "error": str(e)
                }
    
    async def update_vector_metadata(
        self,
        collection_name: str,
        filters: List[Dict[str, str]],
        metadata_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update metadata for vectors matching filters via Vector Service
        
        Args:
            collection_name: Collection to update
            filters: List of filter dicts to identify vectors
            metadata_updates: Dict of metadata key-value pairs to update
            
        Returns:
            Dict with success, points_updated, error
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            import json
            
            # Convert filters
            vector_filters = []
            for f in filters:
                vector_filters.append(
                    vector_service_pb2.VectorFilter(
                        field=f.get("field", ""),
                        value=f.get("value", ""),
                        operator=f.get("operator", "equals")
                    )
                )
            
            # Convert metadata updates to map<string, string> (JSON encode complex types)
            metadata_map = {}
            for key, value in metadata_updates.items():
                if isinstance(value, (list, dict)):
                    metadata_map[key] = json.dumps(value)
                else:
                    metadata_map[key] = str(value)
            
            request = vector_service_pb2.UpdateVectorMetadataRequest(
                collection_name=collection_name,
                filters=vector_filters,
                metadata_updates=metadata_map
            )
            
            response = await self.stub.UpdateVectorMetadata(request, timeout=60.0)
            
            return {
                "success": response.success,
                "points_updated": response.points_updated,
                "error": response.error if response.HasField("error") else None
            }
            
        except grpc.RpcError as e:
            logger.error(f"UpdateVectorMetadata failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "points_updated": 0,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error in update_vector_metadata: {e}")
            return {
                "success": False,
                "points_updated": 0,
                "error": str(e)
            }
    
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "COSINE",
        enable_sparse: bool = False,
    ) -> Dict[str, Any]:
        """
        Create collection via Vector Service
        
        Args:
            collection_name: Name of collection
            vector_size: Vector dimensions
            distance: Distance metric ("COSINE", "EUCLIDEAN", "DOT")
            enable_sparse: If True, create with named dense + sparse vectors for hybrid search
            
        Returns:
            Dict with success, error
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            request = vector_service_pb2.CreateCollectionRequest(
                collection_name=collection_name,
                vector_size=vector_size,
                distance=distance,
                enable_sparse=enable_sparse,
            )
            
            response = await self.stub.CreateCollection(request, timeout=60.0)
            
            return {
                "success": response.success,
                "error": response.error if response.HasField("error") else None
            }
            
        except grpc.RpcError as e:
            logger.error(f"CreateCollection failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error in create_collection: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def delete_collection(
        self,
        collection_name: str
    ) -> Dict[str, Any]:
        """
        Delete collection via Vector Service
        
        Args:
            collection_name: Name of collection to delete
            
        Returns:
            Dict with success, error
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            request = vector_service_pb2.DeleteCollectionRequest(
                collection_name=collection_name
            )
            
            response = await self.stub.DeleteCollection(request, timeout=60.0)
            
            return {
                "success": response.success,
                "error": response.error if response.HasField("error") else None
            }
            
        except grpc.RpcError as e:
            logger.error(f"DeleteCollection failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error in delete_collection: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_collections(self) -> Dict[str, Any]:
        """
        List all collections via Vector Service
        
        Returns:
            Dict with success, collections (list of CollectionInfo dicts), error
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            request = vector_service_pb2.ListCollectionsRequest()
            response = await self.stub.ListCollections(request, timeout=10.0)
            
            if not response.success:
                error_msg = response.error if response.HasField("error") else "List failed"
                return {
                    "success": False,
                    "collections": [],
                    "error": error_msg
                }
            
            collections = []
            for col_info in response.collections:
                collections.append({
                    "name": col_info.name,
                    "vector_size": col_info.vector_size,
                    "distance": col_info.distance,
                    "points_count": col_info.points_count,
                    "status": col_info.status,
                    "schema_type": getattr(col_info, "schema_type", "") or "",
                })
            
            return {
                "success": True,
                "collections": collections,
                "error": None
            }
            
        except grpc.RpcError as e:
            logger.error(f"ListCollections failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "collections": [],
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error in list_collections: {e}")
            return {
                "success": False,
                "collections": [],
                "error": str(e)
            }
    
    async def get_collection_info(
        self,
        collection_name: str
    ) -> Dict[str, Any]:
        """
        Get collection info via Vector Service
        
        Args:
            collection_name: Name of collection
            
        Returns:
            Dict with success, collection (CollectionInfo dict), error
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            request = vector_service_pb2.GetCollectionInfoRequest(
                collection_name=collection_name
            )
            
            response = await self.stub.GetCollectionInfo(request, timeout=10.0)
            
            if not response.success:
                error_msg = response.error if response.HasField("error") else "Get info failed"
                return {
                    "success": False,
                    "collection": None,
                    "error": error_msg
                }
            
            col_info = response.collection
            return {
                "success": True,
                "collection": {
                    "name": col_info.name,
                    "vector_size": col_info.vector_size,
                    "distance": col_info.distance,
                    "points_count": col_info.points_count,
                    "status": col_info.status,
                    "schema_type": getattr(col_info, "schema_type", "") or "",
                },
                "error": None
            }
            
        except grpc.RpcError as e:
            logger.error(f"GetCollectionInfo failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "collection": None,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error in get_collection_info: {e}")
            return {
                "success": False,
                "collection": None,
                "error": str(e)
            }


# Singleton instance
_vector_service_client: Optional[VectorServiceClient] = None

async def get_vector_service_client(required: bool = False) -> VectorServiceClient:
    """Get or create singleton Vector Service client
    
    Args:
        required: If True, raise exception if service unavailable. If False, return client anyway.
    """
    global _vector_service_client
    
    if _vector_service_client is None:
        _vector_service_client = VectorServiceClient()
        await _vector_service_client.initialize(required=required)
    
    return _vector_service_client

