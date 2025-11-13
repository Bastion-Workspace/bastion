"""
Configuration settings for Bastion AI Workspace
"""

import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    SECRET_KEY: str = "bastion-secret-key-change-in-production"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Authentication Configuration
    JWT_SECRET_KEY: str = "bastion-jwt-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 1440  # 24 hours
    
    # Default Admin User (created at startup)
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "admin123"
    ADMIN_EMAIL: str = "admin@localhost"
    
    # Security Settings
    PASSWORD_MIN_LENGTH: int = 8
    MAX_FAILED_LOGINS: int = 5
    ACCOUNT_LOCKOUT_MINUTES: int = 30
    
    # CORS - Allow all origins for development and reverse proxies
    CORS_ORIGINS: List[str] = ["*"]
    
    # Database URLs
    DATABASE_URL: str = "postgresql://bastion_user:bastion_secure_password@localhost:5432/bastion_knowledge_base"
    QDRANT_URL: str = "http://localhost:6333"
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "bastion_password"
    REDIS_URL: str = "redis://localhost:6379"
    SEARXNG_URL: str = "http://localhost:8888"  # SearXNG search engine
    
    # Microservices
    VECTOR_SERVICE_URL: str = "vector-service:50053"
    DATA_SERVICE_HOST: str = "data-service"
    DATA_SERVICE_PORT: int = 50054
    
    # WebDAV Configuration for OrgMode mobile sync
    WEBDAV_HOST: str = "0.0.0.0"
    WEBDAV_PORT: int = 8001
    WEBDAV_ENABLED: bool = True
    
    # Calibre Integration
    CALIBRE_LIBRARY_PATH: str = "/app/calibre"  # Docker mount path
    CALIBRE_ENABLED: bool = False  # Disabled by default, admin can enable
    CALIBRE_SEARCH_WEIGHT: float = 0.3  # Weight in combined search results
    CALIBRE_MAX_RESULTS: int = 50  # Maximum results from Calibre per search
    
    # PostgreSQL Configuration (parsed from DATABASE_URL)
    @property
    def POSTGRES_HOST(self) -> str:
        """Extract PostgreSQL host from DATABASE_URL"""
        import urllib.parse
        parsed = urllib.parse.urlparse(self.DATABASE_URL)
        return parsed.hostname or "localhost"
    
    @property
    def POSTGRES_PORT(self) -> int:
        """Extract PostgreSQL port from DATABASE_URL"""
        import urllib.parse
        parsed = urllib.parse.urlparse(self.DATABASE_URL)
        return parsed.port or 5432
    
    @property
    def POSTGRES_USER(self) -> str:
        """Extract PostgreSQL user from DATABASE_URL"""
        import urllib.parse
        parsed = urllib.parse.urlparse(self.DATABASE_URL)
        return parsed.username or "bastion_user"
    
    @property
    def POSTGRES_PASSWORD(self) -> str:
        """Extract PostgreSQL password from DATABASE_URL"""
        import urllib.parse
        parsed = urllib.parse.urlparse(self.DATABASE_URL)
        return parsed.password or "bastion_secure_password"
    
    @property
    def POSTGRES_DB(self) -> str:
        """Extract PostgreSQL database from DATABASE_URL"""
        import urllib.parse
        parsed = urllib.parse.urlparse(self.DATABASE_URL)
        return parsed.path.lstrip('/') or "bastion_knowledge_base"
    
    # API Keys
    OPENAI_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    OPENWEATHERMAP_API_KEY: str = ""
    
    # AWS Configuration for pricing calculator
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_DEFAULT_REGION: str = "us-east-1"  # Default region for pricing queries
    
    # LLM Configuration
    DEFAULT_MODEL: str = "anthropic/claude-3.5-haiku"  # Default model for general tasks
    FAST_MODEL: str = "anthropic/claude-3.5-haiku"  # Fast model for lightweight ops (query expansion, title generation, intent classification)
    INTENT_CLASSIFICATION_MODEL: str = ""  # Optional override for capability-based intent classifier
    OPENROUTER_MODEL: str = ""  # Optional default chat model used when selecting primary model
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMENSIONS: int = 3072
    
    # LLM Output Configuration
    DEFAULT_MAX_TOKENS: int = 80000  # Default max_tokens for comprehensive LLM outputs (summaries, analysis, etc.)
    
    # Document Processing  
    UPLOAD_MAX_SIZE: str = "1500MB"  # Support very large files and zip archives
    PROCESSING_TIMEOUT: int = 3600   # 60 minutes for large ZIP file processing
    QUALITY_THRESHOLD: float = 0.7
    EMBEDDING_BATCH_SIZE: int = 100
    
    # Feature Flags
    USE_VECTOR_SERVICE: bool = False  # Use new Vector Service for embeddings (gradual rollout)
    
    # Embedding Storage Configuration
    STORAGE_BATCH_SIZE: int = 50     # Smaller batches for more reliable storage
    STORAGE_TIMEOUT_SECONDS: int = 30 # Timeout per batch storage operation
    STORAGE_MAX_RETRIES: int = 3      # Maximum retry attempts per batch
    STORAGE_RETRY_DELAY_BASE: int = 2 # Base delay for exponential backoff (seconds)
    STORAGE_BATCH_DELAY: float = 0.5  # Delay between batches to avoid overwhelming DB
    
    # File Storage
    UPLOAD_DIR: str = "/app/uploads"
    PROCESSED_DIR: str = "/app/processed"
    LOGS_DIR: str = "/app/logs"
    
    # OCR Configuration
    OCR_LANGUAGES: List[str] = ["eng", "fra", "deu"]
    OCR_CONFIDENCE_THRESHOLD: float = 0.6
    
    # Vector Database
    VECTOR_COLLECTION_NAME: str = "documents"
    VECTOR_DISTANCE_METRIC: str = "cosine"
    
    # Knowledge Graph
    KG_BATCH_SIZE: int = 1000
    ENTITY_CONFIDENCE_THRESHOLD: float = 0.8
    
    # Database Pool Configuration
    DB_POOL_MIN_SIZE: int = 5   # Reduced from 10
    DB_POOL_MAX_SIZE: int = 30  # Reduced from 75 to prevent connection exhaustion
    DB_POOL_COMMAND_TIMEOUT: int = 120  # Increased timeout for RSS operations
    DB_POOL_MAX_QUERIES: int = 50000
    DB_POOL_MAX_INACTIVE_TIME: float = 300.0  # 5 minutes
    
    # Chat Configuration
    CONVERSATION_MEMORY_SIZE: int = 10
    MAX_RETRIEVAL_RESULTS: int = 500  # Increased query results limit
    MAX_ENTITY_RESULTS: int = 200  # Increased entity results limit
    
    # Weather Configuration
    WEATHER_UNITS: str = "imperial"  # imperial, metric, or kelvin
    WEATHER_CACHE_MINUTES: int = 10  # Cache weather data for 10 minutes
    WEATHER_DEFAULT_LOCATION: str = ""  # Default ZIP code if none provided
    
    # Deduplication Configuration
    DEDUPLICATION_ENABLED: bool = True
    CONTENT_SIMILARITY_THRESHOLD: float = 0.85  # Higher threshold - only remove very similar content
    EMAIL_THREAD_DEDUP_ENABLED: bool = True
    TEMPORAL_DEDUP_WINDOW_HOURS: int = 24
    FINAL_RESULT_LIMIT: int = 50
    DEDUPLICATION_TIMEOUT_SECONDS: int = 30  # Timeout for deduplication process
    FAST_DEDUP_CHUNK_THRESHOLD: int = 100  # Use fast algorithm above this many chunks
    # No per-document or per-source limits - rely purely on content similarity

    # Org-Mode Settings
    ORG_TODO_SEQUENCE: str = "TODO|NEXT|WAITING|HOLD|DONE|CANCELLED"
    ORG_DEFAULT_TAGS: str = "home,work,personal,errand,finance,health,admin,learning,writing,reading,project,meeting"
    ORG_SUGGEST_TAGS: bool = True
    ORG_TAG_SUGGESTION_MODE: str = "local"  # local|llm|hybrid
    ORG_TAG_AUTOCOMMIT_CONFIDENCE: float = 0.8
    
    # Messaging Configuration
    MESSAGING_ENABLED: bool = True
    MESSAGE_ENCRYPTION_AT_REST: bool = False  # Environment toggle for at-rest encryption
    MESSAGE_ENCRYPTION_MASTER_KEY: str = ""  # Fernet key from environment (optional)
    MESSAGE_MAX_LENGTH: int = 10000  # Maximum message content length
    MESSAGE_RETENTION_DAYS: int = 0  # 0 = indefinite retention
    PRESENCE_HEARTBEAT_SECONDS: int = 30  # How often clients should ping presence
    PRESENCE_OFFLINE_THRESHOLD_SECONDS: int = 90  # When to mark user offline
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get global settings instance"""
    return settings
