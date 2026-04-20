"""
Configuration settings for Bastion AI Workspace
"""

import os
from typing import List
from pydantic import Field
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
    # 24 hours default. Set to 0 or negative for no expiration (long-lived tokens).
    JWT_EXPIRATION_MINUTES: int = 1440  # 24 hours

    # Google Reader compatible RSS API (/api/greader/...)
    GREADER_API_ENABLED: bool = True

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
    
    # Site URL Configuration
    SITE_URL: str = "http://localhost:3051"  # Base URL for HTTP-Referer headers and external links

    # Bastion-to-Bastion federation (Phase 1+)
    FEDERATION_ENABLED: bool = False
    FEDERATION_DISPLAY_NAME: str = ""
    FEDERATION_POLL_INTERVAL_SECONDS: int = 5
    FEDERATION_CONNECTIVITY_PROBE_TIMEOUT: float = 5.0
    FEDERATION_HTTP_TIMEOUT: float = 15.0
    FEDERATION_OUTBOX_MAX_AGE_HOURS: int = 72
    FEDERATION_OUTBOX_MAX_PER_PEER: int = 5000
    FEDERATION_RATE_LIMIT_PER_PEER_PER_MINUTE: int = 120
    FEDERATION_PRESENCE_SYNC_INTERVAL_SECONDS: int = 30
    FEDERATION_ATTACHMENT_TOKEN_TTL_SECONDS: int = 900

    # Database URLs
    DATABASE_URL: str = "postgresql://bastion_user:bastion_secure_password@localhost:5432/bastion_knowledge_base"
    # Optional dedicated role for LangGraph AsyncPostgresSaver (empty = use DATABASE_URL user)
    LANGGRAPH_POSTGRES_USER: str = ""
    LANGGRAPH_POSTGRES_PASSWORD: str = ""
    # When DATABASE_URL points at PgBouncer, set these to the real Postgres host/port so
    # LangGraph AsyncPostgresSaver connects directly (session-safe, avoids pooler quirks).
    LANGGRAPH_POSTGRES_HOST: str = Field(
        default="",
        description="Override Postgres host for LangGraph checkpointer; empty = use DATABASE_URL host.",
    )
    LANGGRAPH_POSTGRES_PORT: int = Field(
        default=5432,
        description="Port for LANGGRAPH_POSTGRES_HOST when host override is set.",
    )
    QDRANT_URL: str = "http://localhost:6333"
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "bastion_password"
    # When False, skip Neo4j entirely (no driver, no backlog drain).
    NEO4J_ENABLED: bool = True
    # When True, startup fails if Neo4j is unreachable (legacy strict behavior).
    NEO4J_REQUIRED: bool = False
    NEO4J_RECONNECT_INTERVAL_SECONDS: int = 60
    KG_BACKLOG_MAX_ATTEMPTS: int = 10
    KG_BACKLOG_PURGE_DAYS: int = 14

    # Vector / embedding stack (optional at startup; backlog when unavailable)
    VECTOR_EMBEDDING_ENABLED: bool = True
    VECTOR_EMBEDDING_REQUIRED: bool = False
    VECTOR_RECONNECT_INTERVAL_SECONDS: int = 60
    VECTOR_EMBED_BACKLOG_MAX_ATTEMPTS: int = 10
    VECTOR_EMBED_BACKLOG_PURGE_DAYS: int = 14
    REDIS_URL: str = "redis://localhost:6379"
    # Encrypted documents: Redis-backed unlock sessions and rate limits
    FILE_ENCRYPTION_SESSION_TTL_SECONDS: int = 900
    FILE_ENCRYPTION_MAX_ATTEMPTS: int = 5
    FILE_ENCRYPTION_LOCKOUT_SECONDS: int = 900
    SEARXNG_URL: str = "http://localhost:8888"  # SearXNG search engine
    
    # Microservices
    VECTOR_SERVICE_URL: str = "vector-service:50053"
    DATA_SERVICE_HOST: str = "data-service"
    DATA_SERVICE_PORT: int = 50054
    CRAWL4AI_SERVICE_HOST: str = os.getenv("CRAWL4AI_SERVICE_HOST", "crawl4ai-service")
    CRAWL4AI_SERVICE_PORT: int = int(os.getenv("CRAWL4AI_SERVICE_PORT", "50055"))
    IMAGE_VISION_SERVICE_HOST: str = os.getenv("IMAGE_VISION_SERVICE_HOST", "image-vision-service")
    IMAGE_VISION_SERVICE_PORT: int = int(os.getenv("IMAGE_VISION_SERVICE_PORT", "50056"))
    # When False, backend never opens a gRPC channel to image-vision (no DNS/connect attempts).
    IMAGE_VISION_ENABLED: bool = True
    # When True, API startup fails if image-vision is not healthy (see main.py lifespan).
    IMAGE_VISION_REQUIRED: bool = False
    CONNECTIONS_SERVICE_HOST: str = os.getenv("CONNECTIONS_SERVICE_HOST", "connections-service")
    CONNECTIONS_SERVICE_PORT: int = int(os.getenv("CONNECTIONS_SERVICE_PORT", "50057"))
    DOCUMENT_SERVICE_HOST: str = os.getenv("DOCUMENT_SERVICE_HOST", "document-service")
    DOCUMENT_SERVICE_PORT: int = int(os.getenv("DOCUMENT_SERVICE_PORT", "50058"))
    VOICE_SERVICE_HOST: str = os.getenv("VOICE_SERVICE_HOST", "voice-service")
    VOICE_SERVICE_PORT: int = int(os.getenv("VOICE_SERVICE_PORT", "50059"))
    # Same key as voice-service when using built-in Hedra TTS (optional; lists TTS engines in settings UI)
    HEDRA_API_KEY: str = os.getenv("HEDRA_API_KEY", "")

    # Microsoft OAuth (external connections)
    MICROSOFT_CLIENT_ID: str = os.getenv("MICROSOFT_CLIENT_ID", "")
    MICROSOFT_CLIENT_SECRET: str = os.getenv("MICROSOFT_CLIENT_SECRET", "")
    MICROSOFT_TENANT_ID: str = os.getenv("MICROSOFT_TENANT_ID", "common")
    MICROSOFT_REDIRECT_URI: str = os.getenv("MICROSOFT_REDIRECT_URI", "")

    @property
    def effective_microsoft_redirect_uri(self) -> str:
        """Redirect URI for Microsoft OAuth; derived from SITE_URL when not set."""
        if self.MICROSOFT_REDIRECT_URI:
            return self.MICROSOFT_REDIRECT_URI
        return self.SITE_URL.rstrip("/") + "/api/oauth/microsoft/callback"

    # GitHub OAuth (external connections, code_platform); set via environment / docker-compose
    GITHUB_CLIENT_ID: str = ""
    GITHUB_CLIENT_SECRET: str = ""
    GITHUB_REDIRECT_URI: str = ""
    GITHUB_SCOPES: str = "repo read:org read:user read:discussion"

    @property
    def effective_github_redirect_uri(self) -> str:
        """Redirect URI for GitHub OAuth; derived from SITE_URL when not set."""
        if self.GITHUB_REDIRECT_URI:
            return self.GITHUB_REDIRECT_URI
        return self.SITE_URL.rstrip("/") + "/api/oauth/github/callback"

    # Internal service-to-service auth (connections-service -> backend external-chat)
    INTERNAL_SERVICE_KEY: str = os.getenv("INTERNAL_SERVICE_KEY", "")

    @property
    def IMAGE_VISION_SERVICE_URL(self) -> str:
        """Get Image Vision Service gRPC URL"""
        return f"{self.IMAGE_VISION_SERVICE_HOST}:{self.IMAGE_VISION_SERVICE_PORT}"

    @property
    def CONNECTIONS_SERVICE_URL(self) -> str:
        """Get Connections Service gRPC URL"""
        return f"{self.CONNECTIONS_SERVICE_HOST}:{self.CONNECTIONS_SERVICE_PORT}"

    @property
    def DOCUMENT_SERVICE_URL(self) -> str:
        """Get Document Service gRPC URL (entity extraction)."""
        return f"{self.DOCUMENT_SERVICE_HOST}:{self.DOCUMENT_SERVICE_PORT}"

    @property
    def VOICE_SERVICE_URL(self) -> str:
        """Get Voice Service gRPC URL (STT/TTS)."""
        return f"{self.VOICE_SERVICE_HOST}:{self.VOICE_SERVICE_PORT}"

    # WebDAV Configuration for OrgMode mobile sync
    WEBDAV_HOST: str = "0.0.0.0"
    WEBDAV_PORT: int = 8001
    WEBDAV_ENABLED: bool = True
    
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

    @property
    def langgraph_postgres_user(self) -> str:
        """Effective DB user for LangGraph checkpointer (LANGGRAPH_POSTGRES_USER or DATABASE_URL user)."""
        u = (self.LANGGRAPH_POSTGRES_USER or "").strip()
        return u if u else self.POSTGRES_USER

    @property
    def langgraph_postgres_password(self) -> str:
        """Password for LangGraph checkpointer user."""
        u = (self.LANGGRAPH_POSTGRES_USER or "").strip()
        if u:
            return self.LANGGRAPH_POSTGRES_PASSWORD or self.POSTGRES_PASSWORD
        return self.POSTGRES_PASSWORD
    
    # API Keys
    OPENAI_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    OPENWEATHERMAP_API_KEY: str = ""
    OSRM_BASE_URL: str = "http://osrm:5000"
    VALHALLA_BASE_URL: str = "http://valhalla:8002"
    ROUTING_PROVIDER: str = "valhalla"  # "valhalla" or "osrm"

    # LLM Configuration
    DEFAULT_MODEL: str = "anthropic/claude-3.5-haiku"  # Default model for general tasks
    FAST_MODEL: str = "anthropic/claude-3.5-haiku"  # Fast model for lightweight ops (query expansion, title generation, intent classification)
    EMBEDDING_MODEL: str = "text-embedding-3-large"

    # Dense vector size for Qdrant collections and placeholders. Must match vector-service
    # EMBEDDING_DIMENSIONS (set the same value in Docker env / .env for backend, celery_worker,
    # and vector-service). Overridden by env var EMBEDDING_DIMENSIONS.
    EMBEDDING_DIMENSIONS: int = Field(
        default=3072,
        description="Dense embedding dimensions; must match vector-service and provider output.",
    )

    # Admin LLM provider base URL overrides (optional; defaults from provider_models.CLOUD_BASE_URLS)
    OPENAI_BASE_URL: str = ""  # e.g. https://api.openai.com/v1
    OPENROUTER_BASE_URL: str = ""  # e.g. https://openrouter.ai/api/v1
    GROQ_BASE_URL: str = ""  # e.g. https://api.groq.com/openai/v1

    # Reasoning Configuration
    REASONING_ENABLED: bool = True  # Enable reasoning tokens support
    REASONING_EFFORT: str = "medium"  # "high", "medium", "low", "minimal", "none"

    # LLM Output Configuration
    DEFAULT_MAX_TOKENS: int = 80000  # Default max_tokens for comprehensive LLM outputs (summaries, analysis, etc.)
    
    # Document Processing  
    UPLOAD_MAX_SIZE: str = "1500MB"  # Support very large files and zip archives
    PROCESSING_TIMEOUT: int = 3600   # 60 minutes for large ZIP file processing
    EMBEDDING_BATCH_SIZE: int = 100

    # Celery gRPC: wall-clock cap for consuming a full orchestrator StreamChat (seconds).
    # Keep strictly below celery_app task_time_limit (1200) so the worker returns a structured failure.
    CELERY_ORCHESTRATOR_STREAM_TIMEOUT_SEC: float = Field(
        default=1080.0,
        description="Max seconds for Celery to read an orchestrator StreamChat RPC end-to-end.",
    )
    
    # Feature Flags
    USE_VECTOR_SERVICE: bool = False  # Use new Vector Service for embeddings (gradual rollout)
    HYBRID_SEARCH_ENABLED: bool = False  # Enable BM25 + dense vector hybrid search with RRF

    # ReRank Configuration
    RERANK_ENABLED: bool = False  # Enable cross-encoder reranking via OpenRouter after RRF merge
    RERANK_MODEL: str = "cohere/rerank-4-pro"  # OpenRouter rerank model
    RERANK_CANDIDATE_MULTIPLIER: int = 3  # Fetch this many × limit candidates before reranking
    
    # Embedding Storage Configuration
    STORAGE_BATCH_SIZE: int = 50     # Smaller batches for more reliable storage
    STORAGE_TIMEOUT_SECONDS: int = 30 # Timeout per batch storage operation
    STORAGE_MAX_RETRIES: int = 3      # Maximum retry attempts per batch
    STORAGE_RETRY_DELAY_BASE: int = 2 # Base delay for exponential backoff (seconds)
    STORAGE_BATCH_DELAY: float = 0.5  # Delay between batches to avoid overwhelming DB
    
    # Redis channel for document-service -> backend WebSocket status relay
    DOCUMENT_STATUS_REDIS_CHANNEL: str = "bastion:document_status"
    # Redis channel for folder/file UI events from document-service
    FOLDER_EVENTS_REDIS_CHANNEL: str = os.getenv(
        "FOLDER_EVENTS_REDIS_CHANNEL", "bastion:folder_events"
    )

    # File Storage
    # Logical document library root (matches document-service UPLOAD_DIR for path strings in APIs/tools).
    UPLOAD_DIR: str = "/app/uploads"
    # Scraped/generated images and sidecars (backend volume; not the user document library on DS).
    WEB_SOURCES_ROOT: str = "/app/web_sources"
    # Ephemeral exports (audio/epub jobs) on backend — not document-service library files.
    EXPORTS_DIR: str = "/app/exports"
    PROCESSED_DIR: str = "/app/processed"
    LOGS_DIR: str = "/app/logs"
    # Ephemeral staging for data workspace CSV/Excel/JSON imports (not the document library)
    DATA_IMPORT_DIR: str = "/app/data_imports"

    # Messaging Attachment Configuration
    MESSAGING_ATTACHMENT_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB
    MESSAGING_ATTACHMENT_ALLOWED_TYPES: List[str] = [
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
        "audio/mpeg",  # MP3
        "audio/wav",
        "audio/wave",
        "audio/x-wav",
        "audio/aac",
        "audio/aacp",
        "audio/flac",
        "audio/ogg",
        "audio/opus",
        "audio/mp4",  # M4A
        "audio/x-m4a",
        "audio/x-ms-wma"
    ]
    MESSAGING_ATTACHMENT_STORAGE_PATH: str = "/app/data_imports/messaging_attachments"
    AUDIO_ATTACHMENT_MAX_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # OCR Configuration
    OCR_LANGUAGES: List[str] = ["eng", "fra", "deu"]
    OCR_CONFIDENCE_THRESHOLD: float = 0.6
    
    # Vector Database
    VECTOR_COLLECTION_NAME: str = "documents"
    VECTOR_DISTANCE_METRIC: str = "cosine"
    
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
    
    # Deduplication Configuration
    DEDUPLICATION_ENABLED: bool = True
    CONTENT_SIMILARITY_THRESHOLD: float = 0.85  # Higher threshold - only remove very similar content
    EMAIL_THREAD_DEDUP_ENABLED: bool = True
    FINAL_RESULT_LIMIT: int = 50
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
    
    # Email Configuration (SMTP)
    # Configure for your SMTP service (SendGrid, Mailgun, Postfix, etc.)
    SMTP_ENABLED: bool = False  # Set to True when SMTP is configured
    SMTP_HOST: str = ""  # SMTP server hostname (e.g., smtp.sendgrid.net, smtp.mailgun.org)
    SMTP_PORT: int = 587  # SMTP port (587 for TLS, 465 for SSL, 25 for unencrypted)
    SMTP_USER: str = ""  # SMTP username (e.g., "apikey" for SendGrid)
    SMTP_PASSWORD: str = ""  # SMTP password or API key
    SMTP_USE_TLS: bool = True  # Use TLS encryption (recommended)
    SMTP_FROM_EMAIL: str = "noreply@bastion.local"  # From email address
    SMTP_FROM_NAME: str = "Bastion AI Workspace"  # From name
    
    # Email Agent Configuration
    EMAIL_AGENT_ENABLED: bool = True
    EMAIL_VERIFICATION_REQUIRED: bool = True
    EMAIL_VERIFICATION_EXPIRY_HOURS: int = 24
    
    # Rate Limiting (Conservative: 5/hour, 20/day)
    EMAIL_HOURLY_LIMIT: int = 5
    EMAIL_DAILY_LIMIT: int = 20
    EMAIL_RATE_LIMITING_ENABLED: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get global settings instance"""
    return settings
