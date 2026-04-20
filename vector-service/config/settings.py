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

    # Vector store backend: qdrant | milvus | elasticsearch
    VECTOR_DB_BACKEND: str = os.getenv("VECTOR_DB_BACKEND", "qdrant").lower()

    # Elasticsearch / OpenSearch (when VECTOR_DB_BACKEND=elasticsearch)
    ES_URL: str = os.getenv("ES_URL", "").strip()
    ES_API_KEY: str = os.getenv("ES_API_KEY", "").strip()
    ES_USERNAME: str = os.getenv("ES_USERNAME", "").strip()
    ES_PASSWORD: str = os.getenv("ES_PASSWORD", "").strip()
    ES_INDEX_PREFIX: str = os.getenv("ES_INDEX_PREFIX", "bastion_").strip() or "bastion_"
    ES_VERIFY_CERTS: bool = os.getenv("ES_VERIFY_CERTS", "true").lower() in (
        "true",
        "1",
        "yes",
    )
    ES_CA_CERTS: str = os.getenv("ES_CA_CERTS", "").strip()
    ES_UPSERT_MAX_RETRIES: int = int(
        os.getenv(
            "ES_UPSERT_MAX_RETRIES",
            os.getenv("QDRANT_UPSERT_MAX_RETRIES", "3"),
        )
    )

    # Milvus (when VECTOR_DB_BACKEND=milvus)
    MILVUS_URI: str = os.getenv("MILVUS_URI", "").strip()
    MILVUS_TOKEN: Optional[str] = os.getenv("MILVUS_TOKEN")
    MILVUS_DB_NAME: str = os.getenv("MILVUS_DB_NAME", "default").strip() or "default"
    MILVUS_CONSISTENCY_LEVEL: str = os.getenv(
        "MILVUS_CONSISTENCY_LEVEL", "Bounded"
    ).strip()
    MILVUS_UPSERT_MAX_RETRIES: int = int(
        os.getenv("MILVUS_UPSERT_MAX_RETRIES", os.getenv("QDRANT_UPSERT_MAX_RETRIES", "3"))
    )

    # Qdrant Configuration (Knowledge Hub Extension!)
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    QDRANT_TIMEOUT: int = int(os.getenv("QDRANT_TIMEOUT", "30"))
    QDRANT_UPSERT_MAX_RETRIES: int = int(os.getenv("QDRANT_UPSERT_MAX_RETRIES", "3"))

    # BM25 Sparse Encoding
    BM25_K1: float = float(os.getenv("BM25_K1", "1.5"))
    BM25_B: float = float(os.getenv("BM25_B", "0.75"))
    BM25_DEFAULT_IDF_PATH: str = os.getenv("BM25_DEFAULT_IDF_PATH", "/app/data/bm25_default_idf.json")

    HYBRID_SEARCH_ENABLED: bool = os.getenv("HYBRID_SEARCH_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    # Performance Tuning
    PARALLEL_WORKERS: int = int(os.getenv("PARALLEL_WORKERS", "4"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))
    MAX_TEXT_LENGTH: int = int(os.getenv("MAX_TEXT_LENGTH", "8000"))

    # Cache Configuration
    EMBEDDING_CACHE_ENABLED: bool = (
        os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
    )
    EMBEDDING_CACHE_TTL: int = int(os.getenv("EMBEDDING_CACHE_TTL", "10800"))  # 3 hours
    # Default under /app/.cache when no volume: writable as non-root. Compose overrides to /data/embedding_cache.db.
    EMBEDDING_CACHE_DB_PATH: str = os.getenv(
        "EMBEDDING_CACHE_DB_PATH", "/app/.cache/embedding_cache.db"
    )
    CACHE_CLEANUP_INTERVAL: int = int(
        os.getenv("CACHE_CLEANUP_INTERVAL", "3600")
    )  # 1 hour

    @classmethod
    def validate(cls) -> None:
        """Validate required settings for the configured embedding provider."""
        vb = (cls.VECTOR_DB_BACKEND or "qdrant").strip().lower()
        if vb not in ("qdrant", "milvus", "elasticsearch"):
            raise ValueError(
                f"Unsupported VECTOR_DB_BACKEND={vb!r}; "
                "supported values: 'qdrant', 'milvus', 'elasticsearch'."
            )
        if vb == "milvus":
            if not cls.MILVUS_URI:
                raise ValueError(
                    "MILVUS_URI must be set when VECTOR_DB_BACKEND=milvus "
                    "(e.g. http://milvus:19530)"
                )
        if vb == "elasticsearch":
            if not cls.ES_URL:
                raise ValueError(
                    "ES_URL must be set when VECTOR_DB_BACKEND=elasticsearch "
                    "(e.g. http://elasticsearch:9200)"
                )

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
