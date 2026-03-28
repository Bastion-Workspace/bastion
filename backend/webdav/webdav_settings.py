"""
Minimal settings for the WebDAV container. Replaces backend/config dependency.
"""

import urllib.parse
from pydantic_settings import BaseSettings


class WebDAVSettings(BaseSettings):
    """Settings for WebDAV server - only what this container needs."""

    DATABASE_URL: str = "postgresql://bastion_user:changeme@localhost:5432/bastion_knowledge_base"
    WEBDAV_HOST: str = "0.0.0.0"
    WEBDAV_PORT: int = 8001
    JWT_SECRET_KEY: str = "changeme"
    LOG_LEVEL: str = "INFO"

    @property
    def POSTGRES_HOST(self) -> str:
        parsed = urllib.parse.urlparse(self.DATABASE_URL)
        return parsed.hostname or "localhost"

    @property
    def POSTGRES_PORT(self) -> int:
        parsed = urllib.parse.urlparse(self.DATABASE_URL)
        return parsed.port or 5432

    @property
    def POSTGRES_USER(self) -> str:
        parsed = urllib.parse.urlparse(self.DATABASE_URL)
        return parsed.username or "bastion_user"

    @property
    def POSTGRES_PASSWORD(self) -> str:
        parsed = urllib.parse.urlparse(self.DATABASE_URL)
        return parsed.password or "bastion_secure_password"

    @property
    def POSTGRES_DB(self) -> str:
        parsed = urllib.parse.urlparse(self.DATABASE_URL)
        return parsed.path.lstrip("/") or "bastion_knowledge_base"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = WebDAVSettings()
