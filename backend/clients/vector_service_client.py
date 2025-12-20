"""
Vector Service gRPC Client

Provides client interface to the Vector Service for embedding generation.
"""

import grpc
import logging
from typing import List, Dict, Any, Optional
import hashlib

from config import get_settings
from protos import vector_service_pb2, vector_service_pb2_grpc

logger = logging.getLogger(__name__)

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
    
    async def initialize(self, required: bool = False):
        """Initialize the gRPC channel and stub
        
        Args:
            required: If True, raise exception on failure. If False, log warning and continue.
        """
        if self._initialized:
            return
        
        try:
            logger.info(f"Connecting to Vector Service at {self.service_url}")
            
            # Create insecure channel with increased message size limits
            # Default is 4MB, increase to 100MB for large batch embedding responses
            options = [
                ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100 MB
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100 MB
            ]
            self.channel = grpc.aio.insecure_channel(self.service_url, options=options)
            self.stub = vector_service_pb2_grpc.VectorServiceStub(self.channel)
            
            # Test connection
            health_request = vector_service_pb2.HealthCheckRequest()
            response = await self.stub.HealthCheck(health_request, timeout=5.0)
            
            if response.status == "healthy":
                logger.info(f"✅ Connected to Vector Service v{response.service_version}")
                logger.info(f"   OpenAI Available: {response.openai_available}")
                self._initialized = True
            else:
                logger.warning(f"⚠️ Vector Service health check returned: {response.status}")
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to Vector Service: {e}")
            if required:
                raise
            else:
                logger.warning("⚠️ Vector Service unavailable - backend will start without embedding support")
                logger.warning("⚠️ Embeddings will be retried when needed")
    
    async def close(self):
        """Close the gRPC channel"""
        if self.channel:
            await self.channel.close()
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
        if not self._initialized:
            logger.info("Vector Service not initialized, attempting to connect...")
            try:
                await self.initialize(required=True)
            except Exception as e:
                logger.error(f"❌ Cannot generate embedding: Vector Service unavailable: {e}")
                raise RuntimeError("Vector Service is not available") from e
        
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
        if not self._initialized:
            logger.info("Vector Service not initialized, attempting to connect...")
            try:
                await self.initialize(required=True)
            except Exception as e:
                logger.error(f"❌ Cannot generate embeddings: Vector Service unavailable: {e}")
                raise RuntimeError("Vector Service is not available") from e
        
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
        if not self._initialized:
            await self.initialize()
        
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
                "service_name": response.service_name,
                "version": response.version,
                "details": dict(response.details) if response.details else {}
            }
            
        except grpc.RpcError as e:
            logger.error(f"❌ gRPC error checking health: {e.code()}: {e.details()}")
            raise
        except Exception as e:
            logger.error(f"❌ Error checking health: {e}")
            raise


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

