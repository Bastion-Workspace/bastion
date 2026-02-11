"""
Connections Service Configuration
"""

import os
from typing import Optional


class Settings:
    """Connections service settings from environment variables."""

    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "connections-service")
    GRPC_PORT: int = int(os.getenv("GRPC_PORT", "50057"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    PARALLEL_WORKERS: int = int(os.getenv("PARALLEL_WORKERS", "4"))

    # Microsoft Graph (used by Microsoft provider)
    MICROSOFT_GRAPH_BASE: str = os.getenv("MICROSOFT_GRAPH_BASE", "https://graph.microsoft.com/v1.0")
    GRAPH_REQUEST_TIMEOUT: int = int(os.getenv("GRAPH_REQUEST_TIMEOUT", "30"))

    # Backend internal API (for external-chat)
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://backend:8000")
    INTERNAL_SERVICE_KEY: str = os.getenv("INTERNAL_SERVICE_KEY", "")
    EXTERNAL_CHAT_TIMEOUT: float = float(os.getenv("EXTERNAL_CHAT_TIMEOUT", "300"))

    @classmethod
    def validate(cls) -> None:
        """Validate required settings."""
        pass


settings = Settings()
