"""
Vector Service Configuration
"""

import os
from typing import Optional


class Settings:
    """Vector service settings from environment variables"""

    # Service Configuration
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "vector-service")
    GRPC_PORT: int = int(os.getenv("GRPC_PORT", "50053"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Embedding Provider Selection
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))

    # OpenAI Configuration (for embeddings)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"
    )
    OPENAI_MAX_RETRIES: int = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
    OPENAI_TIMEOUT: int = int(os.getenv("OPENAI_TIMEOUT", "30"))

    # OpenRouter Configuration
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_EMBEDDING_MODEL: str = os.getenv(
        "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-large"
    )

    # Ollama Configuration (self-hosted)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    OLLAMA_EMBEDDING_MODEL: str = os.getenv(
        "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
    )

    # VLLM Configuration (self-hosted)
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1")
    VLLM_EMBEDDING_MODEL: str = os.getenv("VLLM_EMBEDDING_MODEL", "")
    VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", "")

    # Qdrant Configuration (Knowledge Hub Extension!)
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    QDRANT_TIMEOUT: int = int(os.getenv("QDRANT_TIMEOUT", "30"))
    QDRANT_UPSERT_MAX_RETRIES: int = int(os.getenv("QDRANT_UPSERT_MAX_RETRIES", "3"))
    TOOL_COLLECTION_NAME: str = os.getenv("TOOL_COLLECTION_NAME", "tools")

    # Performance Tuning
    PARALLEL_WORKERS: int = int(os.getenv("PARALLEL_WORKERS", "4"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))
    MAX_TEXT_LENGTH: int = int(os.getenv("MAX_TEXT_LENGTH", "8000"))

    # Cache Configuration
    EMBEDDING_CACHE_ENABLED: bool = (
        os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
    )
    EMBEDDING_CACHE_TTL: int = int(os.getenv("EMBEDDING_CACHE_TTL", "10800"))  # 3 hours
    CACHE_CLEANUP_INTERVAL: int = int(
        os.getenv("CACHE_CLEANUP_INTERVAL", "3600")
    )  # 1 hour

    @classmethod
    def validate(cls) -> None:
        """Validate required settings for the configured embedding provider."""
        provider = (cls.EMBEDDING_PROVIDER or "openai").strip().lower()

        if provider == "openai":
            if not cls.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY must be set for OpenAI provider")
            if cls.OPENAI_API_KEY and not cls.OPENAI_API_KEY.startswith("sk-"):
                raise ValueError("OPENAI_API_KEY format appears incorrect")
            return

        if provider == "openrouter":
            if not cls.OPENROUTER_API_KEY:
                raise ValueError(
                    "OPENROUTER_API_KEY must be set for OpenRouter provider"
                )
            return

        if provider == "vllm":
            if not cls.VLLM_BASE_URL:
                raise ValueError("VLLM_BASE_URL must be set for VLLM provider")
            if not cls.VLLM_EMBEDDING_MODEL:
                raise ValueError(
                    "VLLM_EMBEDDING_MODEL must be set for VLLM provider"
                )
            return

        if provider == "ollama":
            return

        raise ValueError(
            f"Unknown embedding provider: {provider!r}. "
            "Must be one of: openai, openrouter, vllm, ollama"
        )


# Global settings instance
settings = Settings()
