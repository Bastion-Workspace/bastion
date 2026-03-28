"""
Embedding Provider Factory

Creates the configured embedding provider instance based on
EMBEDDING_PROVIDER and related environment variables.
"""

import logging
from typing import Optional

from config.settings import settings
from service.embedding_provider import (
    EmbeddingProvider,
    OpenAICompatibleProvider,
    OllamaProvider,
)

logger = logging.getLogger(__name__)


class EmbeddingProviderFactory:
    """Factory for creating the active embedding provider."""

    _provider: Optional[EmbeddingProvider] = None

    @classmethod
    def create_provider(cls) -> EmbeddingProvider:
        """Create and return the provider for the configured EMBEDDING_PROVIDER."""
        provider_type = (settings.EMBEDDING_PROVIDER or "openai").strip().lower()

        if provider_type == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
            logger.info(
                f"Initializing OpenAI embedding provider with model: {settings.OPENAI_EMBEDDING_MODEL}"
            )
            return OpenAICompatibleProvider(
                api_key=settings.OPENAI_API_KEY,
                base_url=None,
                model=settings.OPENAI_EMBEDDING_MODEL,
                timeout=settings.OPENAI_TIMEOUT,
                max_retries=settings.OPENAI_MAX_RETRIES,
                provider_label="openai",
                max_text_length=settings.MAX_TEXT_LENGTH,
            )

        if provider_type == "openrouter":
            if not settings.OPENROUTER_API_KEY:
                raise ValueError(
                    "OPENROUTER_API_KEY is required for OpenRouter provider"
                )
            logger.info(
                f"Initializing OpenRouter embedding provider with model: {settings.OPENROUTER_EMBEDDING_MODEL}"
            )
            return OpenAICompatibleProvider(
                api_key=settings.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                model=settings.OPENROUTER_EMBEDDING_MODEL,
                timeout=settings.OPENAI_TIMEOUT,
                max_retries=settings.OPENAI_MAX_RETRIES,
                provider_label="openrouter",
                max_text_length=settings.MAX_TEXT_LENGTH,
            )

        if provider_type == "vllm":
            if not settings.VLLM_BASE_URL:
                raise ValueError("VLLM_BASE_URL is required for VLLM provider")
            if not settings.VLLM_EMBEDDING_MODEL:
                raise ValueError(
                    "VLLM_EMBEDDING_MODEL is required for VLLM provider"
                )
            logger.info(
                f"Initializing VLLM embedding provider with model: {settings.VLLM_EMBEDDING_MODEL}"
            )
            return OpenAICompatibleProvider(
                api_key=settings.VLLM_API_KEY or "dummy",
                base_url=settings.VLLM_BASE_URL.rstrip("/"),
                model=settings.VLLM_EMBEDDING_MODEL,
                timeout=settings.OPENAI_TIMEOUT,
                max_retries=settings.OPENAI_MAX_RETRIES,
                provider_label="vllm",
                max_text_length=settings.MAX_TEXT_LENGTH,
            )

        if provider_type == "ollama":
            logger.info(
                f"Initializing Ollama embedding provider with model: {settings.OLLAMA_EMBEDDING_MODEL}"
            )
            return OllamaProvider(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_EMBEDDING_MODEL,
                timeout=settings.OPENAI_TIMEOUT,
                max_text_length=settings.MAX_TEXT_LENGTH,
            )

        raise ValueError(
            f"Unknown embedding provider: {provider_type!r}. "
            "Must be one of: openai, openrouter, vllm, ollama"
        )

    @classmethod
    def get_provider(cls) -> EmbeddingProvider:
        """Get or create the configured embedding provider (singleton)."""
        if cls._provider is None:
            cls._provider = cls.create_provider()
        return cls._provider

    @classmethod
    def reset_provider(cls) -> None:
        """Clear the cached provider instance (for testing or config changes)."""
        cls._provider = None
