"""
Embedding Engine - Provider-based embedding generation with batching
"""

import logging
from typing import List, Optional

from service.embedding_provider import EmbeddingProvider
from service.embedding_provider_factory import EmbeddingProviderFactory

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Handles embedding generation via the configured provider."""

    def __init__(self):
        self.provider: Optional[EmbeddingProvider] = None
        self.model: Optional[str] = None

    async def initialize(self) -> None:
        """Initialize the active embedding provider."""
        self.provider = EmbeddingProviderFactory.get_provider()
        await self.provider.initialize()
        self.model = self.provider.model_name
        logger.info(
            f"Embedding engine initialized with provider: {self.provider.provider_name} model={self.model}"
        )

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding
        """
        if not self.provider:
            raise RuntimeError("Embedding engine not initialized")
        return await self.provider.generate_embedding(text)

    async def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch

        Returns:
            List of embeddings in same order as input texts
        """
        if not self.provider:
            raise RuntimeError("Embedding engine not initialized")
        return await self.provider.generate_batch_embeddings(
            texts, batch_size=batch_size
        )

    async def health_check(self) -> bool:
        """
        Check if the embedding provider is accessible.

        Returns:
            True if healthy, False otherwise
        """
        if not self.provider:
            return False
        return await self.provider.health_check()
