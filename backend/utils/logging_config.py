"""
Logging configuration for Bastion Workspace
Roosevelt's "Structured Logging" System - Dual format for AI analysis and human debugging

**BULLY!** Efficient logging that serves both AI intelligence and human understanding!
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import structlog
from config import settings


# === ROOSEVELT'S STRUCTURED LOGGING SYSTEM ===

def log_structured(component: str, action: str, status: str, **kwargs) -> str:
    """
    Create structured log message with dual format
    
    **BULLY!** AI-optimized structure + human-readable context!
    
    Format: COMPONENT:ACTION:STATUS:key=value:key=value | Human description
    """
    # Build structured part
    structured_parts = [component.upper(), action.upper(), status.upper()]
    
    # Add key-value data
    if kwargs:
        data_parts = [f"{k}={v}" for k, v in kwargs.items() if v is not None]
        structured_parts.extend(data_parts)
    
    structured = ":".join(structured_parts)
    
    # Build human-readable part
    status_emoji = {
        "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️", 
        "START": "🔄", "COMPLETE": "✅", "FAIL": "❌",
        "INIT": "🔧", "PROCESSING": "⚡", "FOUND": "🔍"
    }.get(status.upper(), "ℹ️")
    
    human_parts = [f"{status_emoji} {component}"]
    
    # Add action description
    action_desc = {
        "INIT": "initialized", "START": "starting", "COMPLETE": "completed",
        "ERROR": "failed", "SUCCESS": "succeeded", "PROCESSING": "processing",
        "QUERY": "querying", "STORE": "storing", "GET": "retrieving",
        "CREATE": "creating", "UPDATE": "updating", "DELETE": "deleting"
    }.get(action.upper(), action.lower())
    
    human_parts.append(action_desc)
    
    # Add context
    if kwargs:
        context_parts = []
        for k, v in kwargs.items():
            if k == "doc" and v:
                context_parts.append(f"document {str(v)[:12]}")
            elif k == "user" and v:
                context_parts.append(f"user {v}")
            elif k == "count" and v:
                context_parts.append(f"{v} items")
            elif k == "error" and v:
                context_parts.append(f"error: {v}")
            elif k == "time" and v:
                context_parts.append(f"in {v}s")
            else:
                context_parts.append(f"{k}: {v}")
        
        if context_parts:
            human_parts.append(f"({', '.join(context_parts)})")
    
    human = " ".join(human_parts)
    
    return f"{structured} | {human}"


class ComponentLogger:
    """Base class for component-specific structured loggers"""
    
    def __init__(self, component_name: str):
        self.component = component_name
        self.logger = logging.getLogger(f"structured.{component_name}")
    
    def _log(self, level: str, action: str, status: str, **kwargs):
        """Internal logging with structured format"""
        # Remove component from kwargs since it's already passed as first positional argument
        kwargs.pop('component', None)
        message = log_structured(self.component, action, status, **kwargs)
        getattr(self.logger, level.lower())(message)
    
    def info(self, action: str, status: str, **kwargs):
        self._log("INFO", action, status, **kwargs)
    
    def error(self, action: str, status: str, **kwargs):
        self._log("ERROR", action, status, **kwargs)
    
    def warning(self, action: str, status: str, **kwargs):
        self._log("WARNING", action, status, **kwargs)


class DatabaseLogger(ComponentLogger):
    """Structured logger for database operations"""
    
    def __init__(self):
        super().__init__("DB")
    
    def init_success(self, manager_type: str, config: str = None):
        self.info("INIT", "SUCCESS", manager=manager_type, config=config)
    
    def init_error(self, error: str):
        self.error("INIT", "ERROR", error=error)
    
    def query_start(self, query_type: str, doc_id: str = None, user_id: str = None):
        self.info("QUERY", "START", type=query_type, doc=doc_id, user=user_id)
    
    def query_success(self, query_type: str, count: int = None, time: float = None):
        self.info("QUERY", "SUCCESS", type=query_type, count=count, time=time)
    
    def query_error(self, query_type: str, error: str):
        self.error("QUERY", "ERROR", type=query_type, error=error)
    
    def connection_acquired(self, connection_id: str):
        self.info("CONNECTION", "ACQUIRED", id=connection_id)
    
    def connection_released(self, connection_id: str, duration: float = None):
        self.info("CONNECTION", "RELEASED", id=connection_id, time=duration)


class APILogger(ComponentLogger):
    """Structured logger for API operations"""
    
    def __init__(self):
        super().__init__("API")
    
    def request_start(self, endpoint: str, user: str, method: str = "GET"):
        self.info("REQUEST", "START", endpoint=endpoint, user=user, method=method)
    
    def request_success(self, endpoint: str, status_code: int, time: float = None):
        self.info("REQUEST", "SUCCESS", endpoint=endpoint, status=status_code, time=time)
    
    def request_error(self, endpoint: str, error: str, status_code: int = None):
        self.error("REQUEST", "ERROR", endpoint=endpoint, error=error, status=status_code)
    
    def auth_success(self, user: str, role: str):
        self.info("AUTH", "SUCCESS", user=user, role=role)
    
    def auth_failure(self, reason: str):
        self.error("AUTH", "FAIL", reason=reason)


class VectorLogger(ComponentLogger):
    """Structured logger for vector database operations"""
    
    def __init__(self):
        super().__init__("VECTOR")
    
    def search_start(self, query: str, user_id: str = None, collection: str = None):
        self.info("SEARCH", "START", query=query[:30], user=user_id, collection=collection)
    
    def search_success(self, results: int, time: float = None, query: str = None):
        self.info("SEARCH", "SUCCESS", results=results, time=time, query=query[:20] if query else None)
    
    def store_start(self, doc_id: str, chunks: int):
        self.info("STORE", "START", doc=doc_id, chunks=chunks)
    
    def store_success(self, doc_id: str, chunks: int, time: float = None):
        self.info("STORE", "SUCCESS", doc=doc_id, chunks=chunks, time=time)
    
    def store_error(self, doc_id: str, error: str):
        self.error("STORE", "ERROR", doc=doc_id, error=error)


class RSSLogger(ComponentLogger):
    """Structured logger for RSS operations"""
    
    def __init__(self):
        super().__init__("RSS")
    
    def feed_create(self, feed_id: str, feed_name: str):
        self.info("FEED_CREATE", "SUCCESS", feed=feed_id, name=feed_name)
    
    def article_process_start(self, article_id: str, feed: str):
        self.info("ARTICLE_PROCESS", "START", article=article_id, feed=feed)
    
    def article_process_success(self, article_id: str, doc_id: str, chunks: int):
        self.info("ARTICLE_PROCESS", "SUCCESS", article=article_id, doc=doc_id, chunks=chunks)
    
    def polling_start(self, feed_id: str):
        self.info("POLLING", "START", feed=feed_id)
    
    def polling_complete(self, feed_id: str, articles: int):
        self.info("POLLING", "COMPLETE", feed=feed_id, articles=articles)


class FileManagerLogger(ComponentLogger):
    """Structured logger for file management operations"""
    
    def __init__(self):
        super().__init__("FILEMANAGER")
    
    def placement_start(self, source_type: str, title: str):
        self.info("PLACE", "START", source=source_type, title=title[:30])
    
    def placement_success(self, doc_id: str, folder_id: str, filename: str):
        self.info("PLACE", "SUCCESS", doc=doc_id, folder=folder_id, file=filename)
    
    def markdown_save(self, file_path: str, size: int):
        self.info("MARKDOWN", "SAVED", path=file_path, size=size)


# Global structured logger instances
db_log = DatabaseLogger()
api_log = APILogger()
vector_log = VectorLogger()
rss_log = RSSLogger()
filemanager_log = FileManagerLogger()


class CleanFormatter(logging.Formatter):
    """Formatter that strips trailing newlines from log messages"""
    
    def format(self, record):
        # Get the formatted message
        message = super().format(record)
        # Strip trailing whitespace and newlines, then ensure single newline
        message = message.rstrip()
        return message


def setup_logging():
    """Configure structured logging for the application"""
    
    # Create logs directory if it doesn't exist
    logs_dir = Path(settings.LOGS_DIR)
    logs_dir.mkdir(exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Create formatter that strips trailing whitespace
    formatter = CleanFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Create handlers with clean formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(logs_dir / "codex.log")
    file_handler.setFormatter(formatter)
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[console_handler, file_handler],
        force=True  # Override any existing configuration
    )
    
    # Set specific log levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("🔧 Logging configured successfully")
    
    # Test structured logging
    db_log.info("SYSTEM", "INIT", component="structured_logging")
    logger.info("✅ Roosevelt's Structured Logging System initialized - dual format ready!")
