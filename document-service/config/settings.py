"""
Document Service Configuration
"""

import os


class Settings:
    """Document service settings from environment variables"""

    # Service Configuration
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "document-service")
    GRPC_PORT: int = int(os.getenv("GRPC_PORT", "50058"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # spaCy model for entity extraction
    SPACY_MODEL: str = os.getenv("SPACY_MODEL", "en_core_web_lg")

    # Extraction limits (spaCy is lightweight; cap for very large docs)
    MAX_TEXT_LENGTH: int = int(os.getenv("MAX_TEXT_LENGTH", "1000000"))
    MAX_ENTITY_RESULTS: int = int(os.getenv("MAX_ENTITY_RESULTS", "200"))

    # Concurrency
    PARALLEL_WORKERS: int = int(os.getenv("PARALLEL_WORKERS", "4"))

    @classmethod
    def validate(cls) -> None:
        """Validate required settings."""
        if not cls.SPACY_MODEL or not cls.SPACY_MODEL.strip():
            raise ValueError("SPACY_MODEL must be set")


# Global settings instance
settings = Settings()
