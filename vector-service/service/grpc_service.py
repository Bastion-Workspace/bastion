"""
gRPC Service Implementation - Vector Service (Embedding Generation Only)
"""

import grpc
import logging
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from concurrent import futures

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

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
            
            # Initialize Qdrant client
            if settings.QDRANT_URL:
                self.qdrant_client = QdrantClient(url=settings.QDRANT_URL)
                logger.info(f"Connected to Qdrant at {settings.QDRANT_URL}")
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
