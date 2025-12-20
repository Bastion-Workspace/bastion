"""
Crawl4AI Service Configuration
"""

import os
from typing import Optional

class Settings:
    """Crawl service settings from environment variables"""
    
    # Service Configuration
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "crawl4ai-service")
    GRPC_PORT: int = int(os.getenv("GRPC_PORT", "50055"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Crawl4AI Configuration
    MAX_CONCURRENT_CRAWLS: int = int(os.getenv("MAX_CONCURRENT_CRAWLS", "10"))
    DEFAULT_TIMEOUT_SECONDS: int = int(os.getenv("DEFAULT_TIMEOUT_SECONDS", "60"))
    DEFAULT_MAX_CONTENT_LENGTH: int = int(os.getenv("DEFAULT_MAX_CONTENT_LENGTH", "1000000"))
    
    # Browser Configuration
    BROWSER_POOL_SIZE: int = int(os.getenv("BROWSER_POOL_SIZE", "5"))
    HEADLESS: bool = os.getenv("HEADLESS", "true").lower() == "true"
    
    # Performance Tuning
    RATE_LIMIT_SECONDS: float = float(os.getenv("RATE_LIMIT_SECONDS", "1.0"))
    
    # LLM Configuration (for LLM extraction strategies)
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "anthropic/claude-3-haiku")
    
    @classmethod
    def validate(cls) -> None:
        """Validate required settings"""
        # No strict requirements - service can work without LLM for basic crawling
        pass

# Global settings instance
settings = Settings()







