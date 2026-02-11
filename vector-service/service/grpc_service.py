"""
gRPC Service Implementation - Vector Service (Embedding Generation Only)
"""

import grpc
import logging
import time
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from concurrent import futures

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse

# Import generated proto files (will be generated during Docker build)
import sys
sys.path.insert(0, '/app')

from protos import vector_service_pb2, vector_service_pb2_grpc

from service.embedding_engine import EmbeddingEngine
from service.embedding_cache import EmbeddingCache
from config.settings import settings

logger = logging.getLogger(__name__)


class VectorServiceImplementation(vector_service_pb2_grpc.VectorServiceServicer):
    """
    Vector Service gRPC Implementation - Knowledge Hub Edition!
    
    Now owning both embedding generation AND Qdrant vector operations
    to ensure a centralized, elegant architecture for our agents!
    """
    
    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
        self.embedding_cache = EmbeddingCache(ttl_seconds=settings.EMBEDDING_CACHE_TTL)
        self.qdrant_client: Optional[QdrantClient] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize all components including Qdrant connection"""
        try:
            await self.embedding_engine.initialize()
            await self.embedding_cache.initialize()
            
            # Initialize Qdrant client (timeout avoids default 5s read timeout on slow upserts)
            if settings.QDRANT_URL:
                self.qdrant_client = QdrantClient(
                    url=settings.QDRANT_URL,
                    timeout=settings.QDRANT_TIMEOUT
                )
                logger.info(f"Connected to Qdrant at {settings.QDRANT_URL} (timeout={settings.QDRANT_TIMEOUT}s)")
                # Ensure tools collection exists
                self._ensure_collection_exists(settings.TOOL_COLLECTION_NAME)
            else:
                logger.warning("QDRANT_URL not set, vector store features will be unavailable")
            
            self._initialized = True
            logger.info("Vector Service initialized successfully")
            logger.info("Service mode: Knowledge Hub (Embeddings + Qdrant Ops)")
        except Exception as e:
            logger.error(f"Failed to initialize Vector Service: {e}")
            raise

    def _ensure_collection_exists(self, collection_name: str):
        """Ensure a Qdrant collection exists with proper dimensions"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if collection_name not in collection_names:
                logger.info(f"Creating collection '{collection_name}' in Qdrant")
                # text-embedding-3-large is 3072 dimensions
                # text-embedding-3-small is 1536 dimensions
                # We default to large in settings
                dimensions = 3072 if "large" in settings.OPENAI_EMBEDDING_MODEL else 1536
                
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=dimensions,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection '{collection_name}' created with {dimensions} dimensions")
            else:
                logger.debug(f"Collection '{collection_name}' already exists")
        except Exception as e:
            logger.error(f"Failed to ensure collection '{collection_name}' exists: {e}")
            # Don't raise here, allow other features to work if possible

    async def UpsertTools(self, request, context):
        """Vectorize and store tools in Qdrant (Knowledge Hub Maneuver!)"""
        try:
            if not self._initialized or not self.qdrant_client:
                return vector_service_pb2.UpsertToolsResponse(
                    success=False, 
                    error="Service or Qdrant not initialized"
                )

            success_count = 0
            for tool in request.tools:
                # Create semantic text for embedding (name + description + keywords)
                semantic_text = f"{tool.name} {tool.description} {' '.join(tool.keywords)}"
                
                # Generate embedding
                embedding = await self.embedding_engine.generate_embedding(semantic_text)
                
                # Create point with stable ID from name
                tool_id = int(hashlib.md5(tool.name.encode()).hexdigest(), 16) % (2**63)
                
                point = PointStruct(
                    id=tool_id,
                    vector=embedding,
                    payload={
                        "name": tool.name,
                        "description": tool.description,
                        "pack": tool.pack,
                        "keywords": list(tool.keywords)
                    }
                )
                
                # Upsert to Qdrant
                self.qdrant_client.upsert(
                    collection_name=settings.TOOL_COLLECTION_NAME,
                    points=[point]
                )
                success_count += 1
                
            logger.info(f"Successfully vectorized and stored {success_count} tools")
            return vector_service_pb2.UpsertToolsResponse(success=True, count=success_count)
            
        except Exception as e:
            logger.error(f"UpsertTools failed: {e}")
            return vector_service_pb2.UpsertToolsResponse(success=False, error=str(e))

    async def SearchTools(self, request, context):
        """Search for tools by semantic similarity (Librarian nodes at work!)"""
        try:
            if not self._initialized or not self.qdrant_client:
                return vector_service_pb2.SearchToolsResponse(
                    error="Service or Qdrant not initialized"
                )

            # Generate query embedding
            query_embedding = await self.embedding_engine.generate_embedding(request.query)
            
            # Build filter if pack specified
            query_filter = None
            if request.pack_filter:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="pack",
                            match=MatchValue(value=request.pack_filter)
                        )
                    ]
                )
            
            # Search Qdrant
            search_results = self.qdrant_client.search(
                collection_name=settings.TOOL_COLLECTION_NAME,
                query_vector=query_embedding,
                limit=request.limit or 5,
                query_filter=query_filter,
                score_threshold=request.min_score or 0.5
            )
            
            # Format results
            matches = []
            for result in search_results:
                matches.append(vector_service_pb2.ToolMatch(
                    name=result.payload.get("name"),
                    description=result.payload.get("description"),
                    pack=result.payload.get("pack"),
                    score=result.score
                ))
                
            logger.info(f"Found {len(matches)} tool matches for query: {request.query[:50]}...")
            return vector_service_pb2.SearchToolsResponse(matches=matches)
            
        except Exception as e:
            logger.error(f"SearchTools failed: {e}")
            return vector_service_pb2.SearchToolsResponse(error=str(e))

    # ===== Generic Vector Operations =====
    
    async def UpsertVectors(self, request, context):
        """Store arbitrary vectors in Qdrant (documents, faces, objects, etc.)"""
        try:
            if not self._initialized or not self.qdrant_client:
                return vector_service_pb2.UpsertVectorsResponse(
                    success=False,
                    points_stored=0,
                    error="Service or Qdrant not initialized"
                )
            
            if not request.points:
                return vector_service_pb2.UpsertVectorsResponse(
                    success=True,
                    points_stored=0
                )
            
            # Ensure collection exists
            self._ensure_collection_for_vectors(
                collection_name=request.collection_name,
                vector_size=len(request.points[0].vector) if request.points else 128
            )
            
            # Convert VectorPoint messages to PointStruct
            qdrant_points = []
            for pb_point in request.points:
                # Parse payload - handle JSON-encoded complex types
                payload = {}
                for key, value in pb_point.payload.items():
                    # Try to parse as JSON for complex types (lists, dicts)
                    try:
                        parsed = json.loads(value)
                        payload[key] = parsed
                    except (json.JSONDecodeError, TypeError):
                        # Keep as string if not valid JSON
                        payload[key] = value
                
                # Convert point ID - handle both UUID strings and numeric IDs
                point_id = pb_point.id
                try:
                    # Try to parse as integer (for document chunks using content_hash)
                    point_id = int(point_id)
                except (ValueError, TypeError):
                    # Keep as string (for UUIDs)
                    pass
                
                qdrant_points.append(
                    PointStruct(
                        id=point_id,
                        vector=list(pb_point.vector),
                        payload=payload
                    )
                )
            
            # Batch upsert (100 points per batch) with retry on timeout/connection errors
            batch_size = 100
            total_stored = 0
            max_retries = getattr(settings, "QDRANT_UPSERT_MAX_RETRIES", 3)

            for i in range(0, len(qdrant_points), batch_size):
                batch = qdrant_points[i:i + batch_size]
                last_error = None
                for attempt in range(max_retries):
                    try:
                        self.qdrant_client.upsert(
                            collection_name=request.collection_name,
                            points=batch
                        )
                        total_stored += len(batch)
                        break
                    except Exception as batch_err:
                        last_error = batch_err
                        err_str = str(batch_err).lower()
                        is_retryable = (
                            "timeout" in err_str or "timed out" in err_str or
                            "connection" in err_str or "read" in err_str
                        )
                        if is_retryable and attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            logger.warning(
                                f"Qdrant upsert batch failed (attempt {attempt + 1}/{max_retries}): {batch_err}; "
                                f"will retry in {wait_time}s"
                            )
                            time.sleep(wait_time)
                        else:
                            raise

            logger.info(f"Upserted {total_stored} vectors to collection '{request.collection_name}'")
            return vector_service_pb2.UpsertVectorsResponse(
                success=True,
                points_stored=total_stored
            )
            
        except Exception as e:
            logger.error(f"UpsertVectors failed: {e}")
            import traceback
            traceback.print_exc()
            return vector_service_pb2.UpsertVectorsResponse(
                success=False,
                points_stored=0,
                error=str(e)
            )
    
    async def SearchVectors(self, request, context):
        """Search for similar vectors across any collection"""
        try:
            if not self._initialized or not self.qdrant_client:
                return vector_service_pb2.SearchVectorsResponse(
                    success=False,
                    error="Service or Qdrant not initialized"
                )
            
            # Build Qdrant filter from VectorFilter messages
            query_filter = None
            if request.filters:
                filter_conditions = []
                for vf in request.filters:
                    if vf.operator == "equals":
                        filter_conditions.append(
                            FieldCondition(
                                key=vf.field,
                                match=MatchValue(value=vf.value)
                            )
                        )
                    elif vf.operator == "in":
                        # For array fields - check if value is in array
                        # Qdrant doesn't have native "in" for arrays, use "contains" for now
                        filter_conditions.append(
                            FieldCondition(
                                key=vf.field,
                                match=MatchValue(value=vf.value)
                            )
                        )
                    # Add more operators as needed
                
                if filter_conditions:
                    query_filter = Filter(must=filter_conditions)
            
            # Check if collection exists first
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if request.collection_name not in collection_names:
                # Collection doesn't exist - return empty results (not an error)
                logger.info(f"SearchVectors: Collection '{request.collection_name}' doesn't exist, returning empty results")
                return vector_service_pb2.SearchVectorsResponse(
                    success=True,
                    results=[]
                )
            
            # Execute search (may throw if collection was deleted between check and search)
            # Use 0.0 when score_threshold is 0 or unset so "no threshold" is honored (0.0 is falsy in Python)
            effective_threshold = request.score_threshold if request.score_threshold > 0 else 0.0
            try:
                search_results = self.qdrant_client.search(
                    collection_name=request.collection_name,
                    query_vector=list(request.query_vector),
                    limit=request.limit or 50,
                    query_filter=query_filter,
                    score_threshold=effective_threshold
                )
            except UnexpectedResponse as e:
                # Handle 404 from Qdrant (collection doesn't exist)
                if e.status_code == 404 or "doesn't exist" in str(e).lower() or "not found" in str(e).lower():
                    logger.info(f"SearchVectors: Collection '{request.collection_name}' not found during search, returning empty results")
                    return vector_service_pb2.SearchVectorsResponse(
                        success=True,
                        results=[]
                    )
                # Re-raise other UnexpectedResponse errors
                raise
            
            # Convert results to proto format
            results = []
            for hit in search_results:
                # Convert payload dict to map<string, string> (JSON encode complex types)
                payload_map = {}
                for key, value in hit.payload.items():
                    if isinstance(value, (list, dict)):
                        payload_map[key] = json.dumps(value)
                    else:
                        payload_map[key] = str(value)
                
                results.append(
                    vector_service_pb2.VectorSearchResult(
                        id=str(hit.id),
                        score=hit.score,
                        payload=payload_map
                    )
                )
            
            logger.info(f"SearchVectors: Found {len(results)} results in '{request.collection_name}'")
            return vector_service_pb2.SearchVectorsResponse(
                success=True,
                results=results
            )
            
        except Exception as e:
            logger.error(f"SearchVectors failed: {e}")
            import traceback
            traceback.print_exc()
            # For 404 errors (collection not found), return empty results instead of error
            if "404" in str(e) or "doesn't exist" in str(e).lower() or "not found" in str(e).lower():
                logger.info(f"SearchVectors: Collection not found, returning empty results")
                return vector_service_pb2.SearchVectorsResponse(
                    success=True,
                    results=[]
                )
            return vector_service_pb2.SearchVectorsResponse(
                success=False,
                error=str(e)
            )
    
    async def DeleteVectors(self, request, context):
        """Delete vectors by filter (e.g., delete all for document_id)"""
        try:
            if not self._initialized or not self.qdrant_client:
                return vector_service_pb2.DeleteVectorsResponse(
                    success=False,
                    points_deleted=0,
                    error="Service or Qdrant not initialized"
                )
            
            # Build Qdrant filter from VectorFilter messages
            if not request.filters:
                return vector_service_pb2.DeleteVectorsResponse(
                    success=False,
                    points_deleted=0,
                    error="No filters provided - use DeleteCollection for full deletion"
                )
            
            filter_conditions = []
            for vf in request.filters:
                filter_conditions.append(
                    FieldCondition(
                        key=vf.field,
                        match=MatchValue(value=vf.value)
                    )
                )
            
            query_filter = Filter(must=filter_conditions)
            
            # Execute delete
            delete_result = self.qdrant_client.delete(
                collection_name=request.collection_name,
                points_selector=query_filter
            )
            
            # Qdrant delete returns operation result, extract deleted count if available
            deleted_count = 0
            if hasattr(delete_result, 'operation_id'):
                # For async operations, we may need to check status
                deleted_count = 1  # Indicate success, exact count may not be available
            
            logger.info(f"DeleteVectors: Deleted vectors from '{request.collection_name}' with filters")
            return vector_service_pb2.DeleteVectorsResponse(
                success=True,
                points_deleted=deleted_count
            )
            
        except Exception as e:
            logger.error(f"DeleteVectors failed: {e}")
            return vector_service_pb2.DeleteVectorsResponse(
                success=False,
                points_deleted=0,
                error=str(e)
            )
    
    async def UpdateVectorMetadata(self, request, context):
        """Update metadata for vectors matching filters"""
        try:
            if not self._initialized or not self.qdrant_client:
                return vector_service_pb2.UpdateVectorMetadataResponse(
                    success=False,
                    points_updated=0,
                    error="Service or Qdrant not initialized"
                )
            
            # Build filter
            filter_conditions = []
            for vf in request.filters:
                filter_conditions.append(
                    FieldCondition(
                        key=vf.field,
                        match=MatchValue(value=vf.value)
                    )
                )
            
            query_filter = Filter(must=filter_conditions)
            
            # Parse metadata updates - handle JSON-encoded values
            payload_updates = {}
            for key, value in request.metadata_updates.items():
                try:
                    parsed = json.loads(value)
                    payload_updates[key] = parsed
                except (json.JSONDecodeError, TypeError):
                    payload_updates[key] = value
            
            # Update payload using set_payload
            # Note: Qdrant's set_payload requires scroll + upsert pattern for filtered updates
            # Scroll with with_vectors=True so PointStruct(id, vector, payload) has valid vector for upsert
            scroll_result = self.qdrant_client.scroll(
                collection_name=request.collection_name,
                scroll_filter=query_filter,
                limit=10000,
                with_vectors=True
            )
            
            updated_count = 0
            if scroll_result and scroll_result[0]:
                points_to_update = []
                for point in scroll_result[0]:
                    # Update payload
                    new_payload = point.payload.copy()
                    new_payload.update(payload_updates)
                    
                    points_to_update.append(
                        PointStruct(
                            id=point.id,
                            vector=point.vector,
                            payload=new_payload
                        )
                    )
                
                # Upsert updated points
                if points_to_update:
                    self.qdrant_client.upsert(
                        collection_name=request.collection_name,
                        points=points_to_update
                    )
                    updated_count = len(points_to_update)
            
            logger.info(f"UpdateVectorMetadata: Updated {updated_count} vectors in '{request.collection_name}'")
            return vector_service_pb2.UpdateVectorMetadataResponse(
                success=True,
                points_updated=updated_count
            )
            
        except Exception as e:
            logger.error(f"UpdateVectorMetadata failed: {e}")
            return vector_service_pb2.UpdateVectorMetadataResponse(
                success=False,
                points_updated=0,
                error=str(e)
            )
    
    async def CreateCollection(self, request, context):
        """Create a new Qdrant collection with specified dimensions"""
        try:
            if not self._initialized or not self.qdrant_client:
                return vector_service_pb2.CreateCollectionResponse(
                    success=False,
                    error="Service or Qdrant not initialized"
                )
            
            # Check if collection already exists
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if request.collection_name in collection_names:
                logger.info(f"Collection '{request.collection_name}' already exists (idempotent success)")
                return vector_service_pb2.CreateCollectionResponse(success=True)

            # Qdrant client uses Distance.EUCLID (not EUCLIDEAN)
            distance_map = {
                "COSINE": Distance.COSINE,
                "DOT": Distance.DOT,
                "EUCLIDEAN": Distance.EUCLID,
                "EUCLID": Distance.EUCLID,
            }
            distance_str = (request.distance or "").strip().upper() or "COSINE"
            distance = distance_map.get(distance_str, Distance.COSINE)

            self.qdrant_client.create_collection(
                collection_name=request.collection_name,
                vectors_config=VectorParams(
                    size=request.vector_size,
                    distance=distance,
                ),
            )

            logger.info(
                "Created collection '%s' with %s dimensions, distance=%s",
                request.collection_name,
                request.vector_size,
                distance_str,
            )
            return vector_service_pb2.CreateCollectionResponse(success=True)

        except Exception as e:
            logger.error("CreateCollection failed: %s: %s", type(e).__name__, e)
            return vector_service_pb2.CreateCollectionResponse(
                success=False,
                error=str(e),
            )
    
    async def DeleteCollection(self, request, context):
        """Delete a Qdrant collection"""
        try:
            if not self._initialized or not self.qdrant_client:
                return vector_service_pb2.DeleteCollectionResponse(
                    success=False,
                    error="Service or Qdrant not initialized"
                )
            
            self.qdrant_client.delete_collection(collection_name=request.collection_name)
            
            logger.info(f"Deleted collection '{request.collection_name}'")
            return vector_service_pb2.DeleteCollectionResponse(success=True)
            
        except Exception as e:
            logger.error(f"DeleteCollection failed: {e}")
            return vector_service_pb2.DeleteCollectionResponse(
                success=False,
                error=str(e)
            )
    
    async def ListCollections(self, request, context):
        """List all Qdrant collections"""
        try:
            if not self._initialized or not self.qdrant_client:
                return vector_service_pb2.ListCollectionsResponse(
                    success=False,
                    error="Service or Qdrant not initialized"
                )
            
            collections = self.qdrant_client.get_collections()
            
            collection_infos = []
            for col in collections.collections:
                # Get collection info for points count
                try:
                    col_info = self.qdrant_client.get_collection(col.name)
                    points_count = col_info.points_count if hasattr(col_info, 'points_count') else 0
                    vector_size = col_info.config.params.vectors.size if hasattr(col_info.config.params, 'vectors') else 0
                    distance = str(col_info.config.params.vectors.distance) if hasattr(col_info.config.params, 'vectors') else "COSINE"
                except Exception:
                    points_count = 0
                    vector_size = 0
                    distance = "COSINE"
                
                collection_infos.append(
                    vector_service_pb2.CollectionInfo(
                        name=col.name,
                        vector_size=vector_size,
                        distance=distance,
                        points_count=points_count,
                        status="green"
                    )
                )
            
            return vector_service_pb2.ListCollectionsResponse(
                success=True,
                collections=collection_infos
            )
            
        except Exception as e:
            logger.error(f"ListCollections failed: {e}")
            return vector_service_pb2.ListCollectionsResponse(
                success=False,
                error=str(e)
            )
    
    async def GetCollectionInfo(self, request, context):
        """Get information about a specific collection"""
        try:
            if not self._initialized or not self.qdrant_client:
                return vector_service_pb2.GetCollectionInfoResponse(
                    success=False,
                    error="Service or Qdrant not initialized"
                )
            
            # Check if collection exists first
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if request.collection_name not in collection_names:
                # Collection doesn't exist - return success=False with clear error
                logger.info(f"GetCollectionInfo: Collection '{request.collection_name}' doesn't exist")
                return vector_service_pb2.GetCollectionInfoResponse(
                    success=False,
                    error=f"Collection '{request.collection_name}' doesn't exist"
                )
            
            col_info = self.qdrant_client.get_collection(request.collection_name)
            
            vector_size = col_info.config.params.vectors.size if hasattr(col_info.config.params, 'vectors') else 0
            distance = str(col_info.config.params.vectors.distance) if hasattr(col_info.config.params, 'vectors') else "COSINE"
            points_count = col_info.points_count if hasattr(col_info, 'points_count') else 0
            
            collection_info = vector_service_pb2.CollectionInfo(
                name=request.collection_name,
                vector_size=vector_size,
                distance=distance,
                points_count=points_count,
                status="green"
            )
            
            return vector_service_pb2.GetCollectionInfoResponse(
                success=True,
                collection=collection_info
            )
            
        except Exception as e:
            logger.error(f"GetCollectionInfo failed: {e}")
            # For 404 errors (collection not found), return clear error message
            if "404" in str(e) or "doesn't exist" in str(e).lower() or "not found" in str(e).lower():
                return vector_service_pb2.GetCollectionInfoResponse(
                    success=False,
                    error=f"Collection '{request.collection_name}' doesn't exist"
                )
            return vector_service_pb2.GetCollectionInfoResponse(
                success=False,
                error=str(e)
            )
    
    def _ensure_collection_for_vectors(self, collection_name: str, vector_size: int):
        """Ensure collection exists with correct dimensions"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if collection_name not in collection_names:
                logger.info(f"Creating collection '{collection_name}' with {vector_size} dimensions")
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            logger.error(f"Failed to ensure collection '{collection_name}' exists: {e}")
            raise

    async def GenerateEmbedding(self, request, context):
        """Generate single embedding with cache lookup"""
        try:
            if not self._initialized:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Service not initialized")
                return vector_service_pb2.EmbeddingResponse()
            
            from_cache = False
            
            # Check cache first
            if settings.EMBEDDING_CACHE_ENABLED:
                content_hash = self.embedding_cache.hash_text(request.text)
                cached_embedding = await self.embedding_cache.get(content_hash)
                
                if cached_embedding:
                    logger.debug(f"Cache hit for embedding")
                    from_cache = True
                    return vector_service_pb2.EmbeddingResponse(
                        embedding=cached_embedding,
                        token_count=len(request.text.split()),
                        model=request.model or settings.OPENAI_EMBEDDING_MODEL,
                        from_cache=True
                    )
            
            # Cache miss - generate embedding
            embedding = await self.embedding_engine.generate_embedding(request.text)
            
            # Store in cache
            if settings.EMBEDDING_CACHE_ENABLED:
                content_hash = self.embedding_cache.hash_text(request.text)
                await self.embedding_cache.set(content_hash, embedding)
            
            return vector_service_pb2.EmbeddingResponse(
                embedding=embedding,
                token_count=len(request.text.split()),
                model=request.model or settings.OPENAI_EMBEDDING_MODEL,
                from_cache=False
            )
            
        except Exception as e:
            logger.error(f"GenerateEmbedding failed: {e}")
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
            
            # Check cache for each text
            texts = list(request.texts)
            embeddings = []
            texts_to_generate = []
            text_indices = []
            cache_hits = 0
            cache_misses = 0
            
            if settings.EMBEDDING_CACHE_ENABLED:
                for idx, text in enumerate(texts):
                    content_hash = self.embedding_cache.hash_text(text)
                    cached_embedding = await self.embedding_cache.get(content_hash)
                    
                    if cached_embedding:
                        embeddings.append((idx, cached_embedding, True))  # from_cache=True
                        cache_hits += 1
                    else:
                        texts_to_generate.append(text)
                        text_indices.append(idx)
                        cache_misses += 1
            else:
                texts_to_generate = texts
                text_indices = list(range(len(texts)))
                cache_misses = len(texts)
            
            # Generate embeddings for cache misses
            if texts_to_generate:
                new_embeddings = await self.embedding_engine.generate_batch_embeddings(
                    texts=texts_to_generate,
                    batch_size=request.batch_size or settings.BATCH_SIZE
                )
                
                # Cache new embeddings and add to results
                if settings.EMBEDDING_CACHE_ENABLED:
                    for text, embedding, idx in zip(texts_to_generate, new_embeddings, text_indices):
                        content_hash = self.embedding_cache.hash_text(text)
                        await self.embedding_cache.set(content_hash, embedding)
                        embeddings.append((idx, embedding, False))  # from_cache=False
                else:
                    for embedding, idx in zip(new_embeddings, text_indices):
                        embeddings.append((idx, embedding, False))
            
            # Sort by original index
            embeddings.sort(key=lambda x: x[0])
            
            # Convert to proto format
            embedding_vectors = []
            for idx, (original_idx, embedding, from_cache) in enumerate(embeddings):
                embedding_vectors.append(
                    vector_service_pb2.EmbeddingVector(
                        vector=embedding,
                        index=idx,
                        token_count=len(texts[idx].split()),
                        from_cache=from_cache
                    )
                )
            
            logger.info(f"Batch embeddings: {cache_hits} hits, {cache_misses} misses")
            
            return vector_service_pb2.BatchEmbeddingResponse(
                embeddings=embedding_vectors,
                total_tokens=sum(len(t.split()) for t in texts),
                model=request.model or settings.OPENAI_EMBEDDING_MODEL,
                cache_hits=cache_hits,
                cache_misses=cache_misses
            )
            
        except Exception as e:
            logger.error(f"GenerateBatchEmbeddings failed: {e}")
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
                entries_cleared=cleared
            )
            
        except Exception as e:
            logger.error(f"ClearEmbeddingCache failed: {e}")
            return vector_service_pb2.ClearCacheResponse(
                success=False,
                error=str(e)
            )
    
    async def GetCacheStats(self, request, context):
        """Get cache statistics"""
        try:
            stats = self.embedding_cache.get_stats()
            
            return vector_service_pb2.CacheStatsResponse(
                embedding_cache_size=stats['size'],
                embedding_cache_hits=stats['hits'],
                embedding_cache_misses=stats['misses'],
                cache_hit_rate=stats['hit_rate'],
                ttl_seconds=stats['ttl_seconds']
            )
            
        except Exception as e:
            logger.error(f"GetCacheStats failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vector_service_pb2.CacheStatsResponse()
    
    async def HealthCheck(self, request, context):
        """Health check endpoint"""
        try:
            openai_ok = await self.embedding_engine.health_check() if self._initialized else False
            
            status = "healthy" if (openai_ok and self._initialized) else "degraded"
            if not self._initialized:
                status = "unhealthy"
            
            cache_stats = self.embedding_cache.get_stats()
            details = {
                'cache_size': str(cache_stats['size']),
                'cache_hit_rate': f"{cache_stats['hit_rate']:.2%}",
                'mode': 'embedding_generation_only'
            }
            
            return vector_service_pb2.HealthCheckResponse(
                status=status,
                openai_available=openai_ok,
                service_version="1.0.0",
                details=details
            )
            
        except Exception as e:
            logger.error(f"HealthCheck failed: {e}")
            return vector_service_pb2.HealthCheckResponse(
                status="unhealthy",
                openai_available=False,
                service_version="1.0.0"
            )
