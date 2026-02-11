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
    
    # OpenAI Configuration (for embeddings)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    OPENAI_MAX_RETRIES: int = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
    OPENAI_TIMEOUT: int = int(os.getenv("OPENAI_TIMEOUT", "30"))
    
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
    EMBEDDING_CACHE_ENABLED: bool = os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
    EMBEDDING_CACHE_TTL: int = int(os.getenv("EMBEDDING_CACHE_TTL", "10800"))  # 3 hours
    CACHE_CLEANUP_INTERVAL: int = int(os.getenv("CACHE_CLEANUP_INTERVAL", "3600"))  # 1 hour
    
    @classmethod
    def validate(cls) -> None:
        """Validate required settings"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set")
        
        if not cls.OPENAI_API_KEY.startswith("sk-"):
            raise ValueError("OPENAI_API_KEY format appears incorrect")

# Global settings instance
settings = Settings()
