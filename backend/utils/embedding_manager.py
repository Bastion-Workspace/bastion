"""
Embedding Manager - DEPRECATED

This module is deprecated. All embedding operations now route through:
- Vector Service (embedding generation via gRPC microservice)
- VectorStoreService (vector storage and search)

Use get_embedding_service() from services.embedding_service_wrapper instead.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    DEPRECATED: Use EmbeddingServiceWrapper with Vector Service instead
    
    This stub exists only to catch accidental direct usage and provide
    helpful error messages directing developers to the correct service.
    
    Architecture changed to:
    - Vector Service: Dedicated gRPC microservice for embedding generation
    - VectorStoreService: Dedicated service for vector storage/search
    - EmbeddingServiceWrapper: Unified interface combining both
    """
    
    def __init__(self):
        logger.warning(
            "EmbeddingManager is deprecated. "
            "Use get_embedding_service() from services.embedding_service_wrapper"
        )
        self._initialized = False
    
    async def initialize(self):
        """Deprecated - raises NotImplementedError"""
        raise NotImplementedError(
            "EmbeddingManager is deprecated. "
            "Use: from services.embedding_service_wrapper import get_embedding_service\n"
            "Then: embedding_service = await get_embedding_service()"
        )
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Deprecated - raises NotImplementedError"""
        raise NotImplementedError(
            "Direct EmbeddingManager usage is deprecated.\n"
            "Use: embedding_service = await get_embedding_service()\n"
            "Then: embeddings = await embedding_service.generate_embeddings(texts)"
        )
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Deprecated - raises NotImplementedError"""
        raise NotImplementedError(
            "Direct EmbeddingManager usage is deprecated.\n"
            "Use: embedding_service = await get_embedding_service()\n"
            "Then: embeddings = await embedding_service.generate_embeddings([text])"
        )


# Legacy singleton for backward compatibility (will raise errors if used)
_embedding_manager_instance: Optional[EmbeddingManager] = None


async def get_embedding_manager() -> EmbeddingManager:
    """
    DEPRECATED: Get embedding manager instance
    
    Use get_embedding_service() from services.embedding_service_wrapper instead.
    """
    logger.warning(
        "get_embedding_manager() is deprecated. "
        "Use get_embedding_service() from services.embedding_service_wrapper"
    )
    
    global _embedding_manager_instance
    if _embedding_manager_instance is None:
        _embedding_manager_instance = EmbeddingManager()
    return _embedding_manager_instance
