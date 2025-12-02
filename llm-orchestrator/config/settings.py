"""
LLM Orchestrator Service Configuration
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Configuration settings for LLM Orchestrator Service"""
    
    # Service Configuration
    SERVICE_NAME: str = "llm-orchestrator"
    GRPC_PORT: int = 50051
    LOG_LEVEL: str = "INFO"
    
    # Database Configuration (for LangGraph checkpointer)
    # Note: These defaults are overridden by docker-compose.yml environment variables
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "bastion_knowledge_base"
    POSTGRES_USER: str = "bastion_user"
    POSTGRES_PASSWORD: str = "bastion_secure_password"
    
    # Backend Tool Service
    BACKEND_TOOL_SERVICE_HOST: str = "backend"
    BACKEND_TOOL_SERVICE_PORT: int = 50052
    
    # LLM Configuration
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL: str = "anthropic/claude-3.5-sonnet"
    FAST_MODEL: str = "anthropic/claude-3-haiku"
    
    # OpenAI Configuration (for embeddings)
    OPENAI_API_KEY: Optional[str] = None
    
    # Feature Flags
    ENABLE_STREAMING: bool = True
    ENABLE_TOOL_CALLBACKS: bool = True
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT_SECONDS: int = 300
    
    # Checkpointer Configuration
    CHECKPOINT_SCHEMA: str = "public"
    CHECKPOINT_TABLE_PREFIX: str = "langgraph"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @property
    def postgres_connection_string(self) -> str:
        """Build PostgreSQL connection string for LangGraph checkpointer"""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


# Global settings instance
settings = Settings()

