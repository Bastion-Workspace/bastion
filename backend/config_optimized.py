"""
Optimized Configuration for Plato Knowledge Base
Reduces resource usage while maintaining performance
"""

import os
from typing import Dict, Any

# Import base configuration
from config import *

# Optimized worker pool configurations based on environment
WORKER_SCALING = {
    "development": {
        "embedding_workers": 2,
        "storage_workers": 1,
        "document_workers": 3,
        "process_workers": 1,
        "background_chat_workers": 1
    },
    "production": {
        "embedding_workers": 4,
        "storage_workers": 2,
        "document_workers": 6,
        "process_workers": 2,
        "background_chat_workers": 2
    },
    "testing": {
        "embedding_workers": 1,
        "storage_workers": 1,
        "document_workers": 2,
        "process_workers": 1,
        "background_chat_workers": 1
    }
}

# Get current environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
CURRENT_SCALING = WORKER_SCALING.get(ENVIRONMENT, WORKER_SCALING["development"])

# Optimized embedding configuration
EMBEDDING_CONFIG = {
    "max_concurrent_requests": CURRENT_SCALING["embedding_workers"],
    "max_concurrent_storage": CURRENT_SCALING["storage_workers"],
    "embedding_batch_size": 20,
    "storage_batch_size": 100,
    "max_retries": 3,
    "retry_delay": 1.0,
    "rate_limit_delay": 0.1,
    "enable_batch_optimization": True,
    "enable_concurrent_storage": True
}

# Optimized document processing configuration
PROCESSING_CONFIG = {
    "max_concurrent_documents": CURRENT_SCALING["document_workers"],
    "max_concurrent_chunks": CURRENT_SCALING["document_workers"] * 2,
    "max_concurrent_embeddings": CURRENT_SCALING["embedding_workers"] * 2,
    "strategy": "hybrid",
    "chunk_batch_size": 50,
    "embedding_batch_size": 20,
    "enable_document_level_parallelism": True,
    "enable_chunk_level_parallelism": True,
    "enable_io_parallelism": True,
    "thread_pool_size": CURRENT_SCALING["document_workers"],
    "process_pool_size": CURRENT_SCALING["process_workers"]
}

# Background chat configuration
BACKGROUND_CHAT_CONFIG = {
    "max_workers": CURRENT_SCALING["background_chat_workers"],
    "queue_size": 100,
    "job_timeout": 300,  # 5 minutes
    "cleanup_interval": 60  # 1 minute
}

# Connection pool optimization
DATABASE_POOL_CONFIG = {
    "min_size": 5,
    "max_size": 20,  # Reduced from potential higher values
    "command_timeout": 60,
    "server_settings": {
        "application_name": "plato_knowledge_base",
        "jit": "off"
    }
}

# Redis connection optimization
REDIS_POOL_CONFIG = {
    "max_connections": 10,  # Reduced from potential higher values
    "retry_on_timeout": True,
    "health_check_interval": 30
}

# Qdrant client optimization
QDRANT_CONFIG = {
    "timeout": 30,
    "prefer_grpc": True,
    "grpc_port": 6334,
    "api_key": None,
    "prefix": None
}

# Service initialization timeouts
SERVICE_TIMEOUTS = {
    "database": 30,
    "redis": 15,
    "qdrant": 20,
    "neo4j": 25,
    "embedding_manager": 45,
    "document_service": 60,
    "chat_service": 30
}

# Memory optimization settings
MEMORY_CONFIG = {
    "enable_gc_optimization": True,
    "gc_threshold": (700, 10, 10),  # More aggressive garbage collection
    "max_chunk_size": 1000,  # Smaller chunks to reduce memory usage
    "batch_processing_size": 50,  # Smaller batches
    "enable_memory_profiling": ENVIRONMENT == "development"
}

# Logging optimization
LOGGING_CONFIG = {
    "level": "INFO" if ENVIRONMENT == "production" else "DEBUG",
    "reduce_verbose_logs": ENVIRONMENT == "production",
    "log_worker_details": ENVIRONMENT == "development",
    "log_service_stats": True,
    "performance_logging": True
}

def get_optimized_config() -> Dict[str, Any]:
    """Get the complete optimized configuration"""
    return {
        "environment": ENVIRONMENT,
        "worker_scaling": CURRENT_SCALING,
        "embedding_config": EMBEDDING_CONFIG,
        "processing_config": PROCESSING_CONFIG,
        "background_chat_config": BACKGROUND_CHAT_CONFIG,
        "database_pool_config": DATABASE_POOL_CONFIG,
        "redis_pool_config": REDIS_POOL_CONFIG,
        "qdrant_config": QDRANT_CONFIG,
        "service_timeouts": SERVICE_TIMEOUTS,
        "memory_config": MEMORY_CONFIG,
        "logging_config": LOGGING_CONFIG
    }

def log_optimization_settings():
    """Log the current optimization settings"""
    import logging
    logger = logging.getLogger(__name__)
    
    config = get_optimized_config()
    logger.info(f"🔧 Plato Knowledge Base - Optimized Configuration")
    logger.info(f"   Environment: {config['environment']}")
    logger.info(f"   Embedding Workers: {config['worker_scaling']['embedding_workers']}")
    logger.info(f"   Storage Workers: {config['worker_scaling']['storage_workers']}")
    logger.info(f"   Document Workers: {config['worker_scaling']['document_workers']}")
    logger.info(f"   Process Workers: {config['worker_scaling']['process_workers']}")
    logger.info(f"   Background Chat Workers: {config['worker_scaling']['background_chat_workers']}")
    logger.info(f"   Total Workers: {sum(config['worker_scaling'].values())}")
    logger.info(f"   Memory Optimization: {'Enabled' if config['memory_config']['enable_gc_optimization'] else 'Disabled'}")

# Export optimized settings for easy import
__all__ = [
    'WORKER_SCALING',
    'CURRENT_SCALING', 
    'EMBEDDING_CONFIG',
    'PROCESSING_CONFIG',
    'BACKGROUND_CHAT_CONFIG',
    'DATABASE_POOL_CONFIG',
    'REDIS_POOL_CONFIG',
    'QDRANT_CONFIG',
    'SERVICE_TIMEOUTS',
    'MEMORY_CONFIG',
    'LOGGING_CONFIG',
    'get_optimized_config',
    'log_optimization_settings'
]
