"""
Embedding Manager - Handles text embedding generation ONLY

Vector storage operations moved to VectorStoreService for clean separation of concerns.
This service focuses exclusively on generating embeddings via OpenAI API.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages text embedding generation using OpenAI API
    
    Responsibilities:
    - Generate embeddings for text content
    - Batch embedding generation
    - Retry logic and error handling
    - Text preprocessing and validation
    
    NOT responsible for:
    - Vector storage (see VectorStoreService)
    - Vector search (see VectorStoreService)
    - Collection management (see VectorStoreService)
    """
    
    def __init__(self):
        self.openai_client: Optional[AsyncOpenAI] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize OpenAI client for embedding generation"""
        if self._initialized:
            return
            
        logger.info("Initializing Embedding Manager (generation only)...")
        
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")
        
        if settings.OPENAI_API_KEY.startswith("sk-"):
            logger.info("OpenAI API key found")
        else:
            logger.warning("OpenAI API key format may be incorrect")
        
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        self._initialized = True
        logger.info("Embedding Manager initialized (generation only)")
    
    async def generate_embeddings(
        self,
        texts: List[str],
        max_retries: int = 5
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts with retry logic
        
        Args:
            texts: List of text strings to embed
            max_retries: Maximum retry attempts for API calls
            
        Returns:
            List of embedding vectors (one per input text)
        """
        try:
            # Validate and clean input texts
            if not texts:
                logger.warning("Empty text list provided for embedding generation")
                return []
            
            # Filter out empty or None texts
            valid_texts = []
            for i, text in enumerate(texts):
                if text is None:
                    logger.warning(f"Text at index {i} is None, skipping")
                    continue
                if not isinstance(text, str):
                    logger.warning(f"Text at index {i} is not a string ({type(text)}), converting")
                    text = str(text)
                if not text.strip():
                    logger.warning(f"Text at index {i} is empty or whitespace, skipping")
                    continue
                valid_texts.append(text.strip())
            
            if not valid_texts:
                logger.error("No valid texts to embed after filtering")
                return []
            
            logger.info(f"Generating embeddings for {len(valid_texts)} texts")
            
            # Check if OpenAI client is initialized
            if not self.openai_client:
                logger.error("OpenAI client not initialized")
                return []
            
            # Generate embeddings with retry logic
            for attempt in range(max_retries):
                try:
                    logger.info(
                        f"Attempt {attempt + 1}: Calling OpenAI embeddings API "
                        f"(model: {settings.EMBEDDING_MODEL})"
                    )
                    
                    response = await asyncio.wait_for(
                        self.openai_client.embeddings.create(
                            model=settings.EMBEDDING_MODEL,
                            input=valid_texts,
                            encoding_format="float"
                        ),
                        timeout=30.0
                    )
                    
                    # Extract embeddings from response
                    embeddings = []
                    for data in response.data:
                        if hasattr(data, 'embedding') and data.embedding:
                            embeddings.append(data.embedding)
                        else:
                            logger.warning("Empty embedding received from OpenAI")
                            embeddings.append([0.0] * settings.EMBEDDING_DIMENSIONS)
                    
                    logger.info(f"Generated {len(embeddings)} embeddings successfully")
                    return embeddings
                    
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Embedding generation timeout "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(1 * (attempt + 1))
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(
                        f"Embedding generation failed "
                        f"(attempt {attempt + 1}/{max_retries}): {error_msg}"
                    )
                    
                    # Check if it's a non-retryable error
                    if "invalid_request_error" in error_msg.lower() or "400" in error_msg:
                        logger.error(f"Non-retryable error: {error_msg}")
                        raise
                    
                    if attempt == max_retries - 1:
                        raise
                    
                    # Wait before retry with exponential backoff
                    wait_time = min(2 ** attempt, 10)
                    await asyncio.sleep(wait_time)
            
            return []
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
    
    async def generate_embedding(self, text: str, max_retries: int = 5) -> Optional[List[float]]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string to embed
            max_retries: Maximum retry attempts
            
        Returns:
            Embedding vector or None if failed
        """
        embeddings = await self.generate_embeddings([text], max_retries)
        return embeddings[0] if embeddings else None
    
    async def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100,
        max_retries: int = 5
    ) -> List[List[float]]:
        """
        Generate embeddings in batches for large text lists
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts per batch
            max_retries: Maximum retry attempts per batch
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        logger.info(f"Generating embeddings in {total_batches} batches of {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            batch_embeddings = await self.generate_embeddings(batch, max_retries)
            
            if not batch_embeddings:
                logger.error(f"Failed to generate embeddings for batch {batch_num}")
                # Append zero vectors for failed batch
                batch_embeddings = [[0.0] * settings.EMBEDDING_DIMENSIONS] * len(batch)
            
            all_embeddings.extend(batch_embeddings)
        
        logger.info(f"Generated {len(all_embeddings)} total embeddings")
        return all_embeddings
    
    def truncate_text(self, text: str, max_tokens: int = 8000) -> str:
        """
        Truncate text to fit within token limits
        
        Uses approximate character-based truncation (1 token ≈ 4 characters)
        
        Args:
            text: Text to truncate
            max_tokens: Maximum token count
            
        Returns:
            Truncated text
        """
        if not text:
            return ""
        
        # Approximate: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return text
        
        # Truncate and try to end at a sentence boundary
        truncated = text[:max_chars]
        
        # Try to find last sentence boundary
        for delimiter in ['. ', '.\n', '! ', '?\n', '? ']:
            last_delimiter = truncated.rfind(delimiter)
            if last_delimiter > len(truncated) * 0.8:
                return truncated[:last_delimiter + 1].strip()
        
        # Try to find last word boundary
        last_space = truncated.rfind(' ')
        if last_space > len(truncated) * 0.8:
            return truncated[:last_space].strip()
        
        return truncated.strip()
    
    def validate_text(self, text: str) -> bool:
        """
        Validate text is suitable for embedding generation
        
        Args:
            text: Text to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not text or not isinstance(text, str):
            return False
        
        if not text.strip():
            return False
        
        # Check minimum length (at least 10 characters)
        if len(text.strip()) < 10:
            logger.warning(f"Text too short for embedding: {len(text.strip())} chars")
            return False
        
        return True


# Singleton instance
_embedding_manager_instance: Optional[EmbeddingManager] = None


async def get_embedding_manager() -> EmbeddingManager:
    """Get singleton embedding manager instance"""
    global _embedding_manager_instance
    if _embedding_manager_instance is None:
        _embedding_manager_instance = EmbeddingManager()
        await _embedding_manager_instance.initialize()
    return _embedding_manager_instance





