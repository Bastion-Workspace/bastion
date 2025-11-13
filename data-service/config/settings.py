import os
from typing import Optional

class Settings:
    """Data service configuration"""
    
    # Service Configuration
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "data-service")
    GRPC_PORT: int = int(os.getenv("GRPC_PORT", "50054"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Database Configuration
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "postgres-data")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "data_workspace")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "data_user")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "data_workspace_secure_password")
    
    # Connection Pool Settings
    DB_POOL_MIN_SIZE: int = int(os.getenv("DB_POOL_MIN_SIZE", "2"))
    DB_POOL_MAX_SIZE: int = int(os.getenv("DB_POOL_MAX_SIZE", "10"))
    
    # Import Settings
    MAX_IMPORT_FILE_SIZE: int = int(os.getenv("MAX_IMPORT_FILE_SIZE", "524288000"))  # 500MB
    IMPORT_BATCH_SIZE: int = int(os.getenv("IMPORT_BATCH_SIZE", "1000"))
    
    # Query Settings
    MAX_QUERY_RESULTS: int = int(os.getenv("MAX_QUERY_RESULTS", "10000"))
    QUERY_TIMEOUT_SECONDS: int = int(os.getenv("QUERY_TIMEOUT_SECONDS", "60"))
    
    @property
    def database_url(self) -> str:
        """Get PostgreSQL connection URL"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def async_database_url(self) -> str:
        """Get async PostgreSQL connection URL (for asyncpg)"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"


settings = Settings()





