"""
Embedding Engine - OpenAI embedding generation with batching
"""

import asyncio
import logging
from typing import List, Optional
from openai import AsyncOpenAI
from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Handles embedding generation via OpenAI API"""
    
    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None
        self.model = settings.OPENAI_EMBEDDING_MODEL
        self.max_retries = settings.OPENAI_MAX_RETRIES
        self.timeout = settings.OPENAI_TIMEOUT
    
    async def initialize(self):
        """Initialize OpenAI client"""
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=self.timeout,
            max_retries=self.max_retries
        )
        
        logger.info(f"Embedding engine initialized with model: {self.model}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        if not self.client:
            raise RuntimeError("Embedding engine not initialized")
        
        try:
            # Truncate if needed
            text = self._truncate_text(text)
            
            response = await self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding (dim: {len(embedding)})")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def generate_batch_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch
            
        Returns:
            List of embeddings in same order as input texts
        """
        if not self.client:
            raise RuntimeError("Embedding engine not initialized")
        
        if not texts:
            return []
        
        # Truncate all texts
        truncated_texts = [self._truncate_text(text) for text in texts]
        
        # Process in batches
        all_embeddings = []
        total_batches = (len(truncated_texts) + batch_size - 1) // batch_size
        
        logger.info(f"Generating embeddings for {len(texts)} texts in {total_batches} batches")
        
        for i in range(0, len(truncated_texts), batch_size):
            batch = truncated_texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
                
                response = await self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                
                # Extract embeddings in order
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                logger.debug(f"Batch {batch_num}/{total_batches} complete")
                
            except Exception as e:
                logger.error(f"Failed to generate batch {batch_num}: {e}")
                raise
        
        logger.info(f"Generated {len(all_embeddings)} embeddings successfully")
        return all_embeddings
    
    def _truncate_text(self, text: str) -> str:
        """
        Truncate text to fit within token limits
        
        Args:
            text: Text to truncate
            
        Returns:
            Truncated text
        """
        max_length = settings.MAX_TEXT_LENGTH
        
        if len(text) <= max_length:
            return text
        
        # Simple character-based truncation
        # For production, use tiktoken for proper token counting
        truncated = text[:max_length]
        
        # Try to break at a word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:
            truncated = truncated[:last_space]
        
        logger.debug(f"Truncated text from {len(text)} to {len(truncated)} chars")
        return truncated.strip()
    
    async def health_check(self) -> bool:
        """
        Check if OpenAI API is accessible
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self.client:
                return False
            
            # Try to generate a simple embedding
            await self.generate_embedding("health check")
            return True
            
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False

