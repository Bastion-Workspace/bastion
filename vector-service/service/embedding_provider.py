"""
Embedding Provider Abstraction

Supports multiple embedding providers (OpenAI, OpenRouter, VLLM, Ollama)
with a unified interface.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Base class for embedding providers"""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Active model name for this provider"""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider (clients, connections)."""
        pass

    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    async def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100,
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is accessible."""
        pass

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to fit within character limits."""
        if len(text) <= max_length:
            return text
        truncated = text[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > max_length * 0.8:
            truncated = truncated[:last_space]
        logger.debug(f"Truncated text from {len(text)} to {len(truncated)} chars")
        return truncated.strip()


class OpenAICompatibleProvider(EmbeddingProvider):
    """
    Handles OpenAI, OpenRouter, and VLLM via the OpenAI-compatible
    /v1/embeddings API (same request/response shape).
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str],
        model: str,
        timeout: int,
        max_retries: int,
        provider_label: str,
        max_text_length: int = 8000,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._provider_label = provider_label
        self._max_text_length = max_text_length
        self._client = None

    @property
    def provider_name(self) -> str:
        return self._provider_label

    @property
    def model_name(self) -> str:
        return self._model

    async def initialize(self) -> None:
        from openai import AsyncOpenAI

        kwargs = {
            "api_key": self._api_key,
            "timeout": self._timeout,
            "max_retries": self._max_retries,
        }
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = AsyncOpenAI(**kwargs)
        logger.info(
            f"Embedding provider initialized: {self._provider_label} model={self._model}"
        )

    async def generate_embedding(self, text: str) -> List[float]:
        if not self._client:
            raise RuntimeError("Provider not initialized")
        text = self._truncate_text(text, self._max_text_length)
        response = await self._client.embeddings.create(
            input=[text],
            model=self._model,
        )
        embedding = response.data[0].embedding
        logger.debug(f"Generated embedding (dim: {len(embedding)})")
        return embedding

    async def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100,
    ) -> List[List[float]]:
        if not self._client:
            raise RuntimeError("Provider not initialized")
        if not texts:
            return []
        truncated_texts = [
            self._truncate_text(t, self._max_text_length) for t in texts
        ]
        all_embeddings: List[List[float]] = []
        total_batches = (len(truncated_texts) + batch_size - 1) // batch_size
        logger.info(
            f"Generating embeddings for {len(texts)} texts in {total_batches} batches"
        )
        for i in range(0, len(truncated_texts), batch_size):
            batch = truncated_texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            logger.debug(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)"
            )
            response = await self._client.embeddings.create(
                input=batch,
                model=self._model,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            logger.debug(f"Batch {batch_num}/{total_batches} complete")
        logger.info(f"Generated {len(all_embeddings)} embeddings successfully")
        return all_embeddings

    async def health_check(self) -> bool:
        try:
            if not self._client:
                return False
            await self.generate_embedding("health check")
            return True
        except Exception as e:
            logger.error(f"{self._provider_label} health check failed: {e}")
            return False


class OllamaProvider(EmbeddingProvider):
    """
    Ollama embedding provider using the native /api/embed endpoint.
    Supports single and batch input; response shape differs from OpenAI.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 30,
        max_text_length: int = 8000,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._max_text_length = max_text_length
        self._client: Optional[object] = None  # httpx.AsyncClient

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def initialize(self) -> None:
        import httpx

        self._client = httpx.AsyncClient(timeout=self._timeout)
        logger.info(
            f"Embedding provider initialized: ollama model={self._model} base_url={self._base_url}"
        )

    async def generate_embedding(self, text: str) -> List[float]:
        if not self._client:
            raise RuntimeError("Provider not initialized")
        text = self._truncate_text(text, self._max_text_length)
        resp = await self._client.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise ValueError("Ollama returned no embeddings")
        return embeddings[0]

    async def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100,
    ) -> List[List[float]]:
        if not self._client:
            raise RuntimeError("Provider not initialized")
        if not texts:
            return []
        truncated_texts = [
            self._truncate_text(t, self._max_text_length) for t in texts
        ]
        all_embeddings: List[List[float]] = []
        total_batches = (len(truncated_texts) + batch_size - 1) // batch_size
        logger.info(
            f"Generating embeddings for {len(texts)} texts in {total_batches} batches (ollama)"
        )
        for i in range(0, len(truncated_texts), batch_size):
            batch = truncated_texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            resp = await self._client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": batch},
            )
            resp.raise_for_status()
            data = resp.json()
            batch_embeddings = data.get("embeddings", [])
            all_embeddings.extend(batch_embeddings)
            logger.debug(f"Batch {batch_num}/{total_batches} complete")
        logger.info(f"Generated {len(all_embeddings)} embeddings successfully")
        return all_embeddings

    async def health_check(self) -> bool:
        try:
            if not self._client:
                return False
            await self.generate_embedding("health check")
            return True
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
