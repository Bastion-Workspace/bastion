"""
gRPC Service Implementation - Vector Service (Embedding Generation Only)
"""

import grpc
import logging
import time
from datetime import datetime
from typing import Dict, Any
from concurrent import futures

# Import generated proto files (will be generated during Docker build)
import sys
sys.path.insert(0, '/app')

from protos import vector_service_pb2, vector_service_pb2_grpc

from service.embedding_engine import EmbeddingEngine
from service.embedding_cache import EmbeddingCache
from config.settings import settings

logger = logging.getLogger(__name__)


class VectorServiceImplementation(vector_service_pb2_grpc.VectorServiceServicer):
    """Vector service gRPC implementation - Embedding generation and caching only"""
    
    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
        self.embedding_cache = EmbeddingCache(ttl_seconds=settings.EMBEDDING_CACHE_TTL)
        self._initialized = False
    
    async def initialize(self):
        """Initialize all components"""
        try:
            await self.embedding_engine.initialize()
            await self.embedding_cache.initialize()
            self._initialized = True
            logger.info("Vector Service initialized successfully")
            logger.info("Service mode: Embedding generation + caching (caller handles Qdrant)")
        except Exception as e:
            logger.error(f"Failed to initialize Vector Service: {e}")
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
