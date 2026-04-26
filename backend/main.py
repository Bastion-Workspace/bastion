"""
Bastion Workspace - Main FastAPI Application V2 (Optimized)
A sophisticated RAG system with PostgreSQL-backed document storage
"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
import uvicorn

import asyncpg

from config import settings
from services.service_container import service_container
from services.schema_guards import ensure_user_memory_schema_columns
from version import __version__
from utils.string_utils import strip_yaml_frontmatter

# Initialize Celery app
from services.celery_app import celery_app

# Import Celery tasks to ensure they are registered
import services.celery_tasks.agent_tasks
import services.celery_tasks.rss_tasks
import services.celery_tasks.audio_export_tasks


from services.settings_service import settings_service
from services.auth_service import auth_service

from services.user_document_service import UserDocumentService
from models.api_models import (
    URLImportRequest, ImportImageRequest, QueryRequest, DocumentListResponse, 
    DocumentUploadResponse, DocumentStatus, QueryResponse, 
    QueryHistoryResponse, AvailableModelsResponse,
    ModelConfigRequest, DocumentFilterRequest, DocumentUpdateRequest,
    BulkCategorizeRequest, DocumentCategoriesResponse, BulkOperationResponse,
    SettingsResponse, SettingUpdateRequest, BulkSettingsUpdateRequest, SettingUpdateResponse,
    ProcessingStatus, DocumentType, DocumentInfo,
    # Authentication models
    LoginRequest, LoginResponse, UserCreateRequest, UserUpdateRequest,
    PasswordChangeRequest, UserResponse, UsersListResponse, AuthenticatedUserResponse,
    # Folder models
    DocumentFolder, FolderCreateRequest, FolderUpdateRequest, FolderMetadataUpdateRequest, 
    FolderTreeResponse, FolderContentsResponse
)
from models.conversation_models import (
    CreateConversationRequest, CreateMessageRequest, ConversationResponse,
    MessageResponse, ConversationListResponse, MessageListResponse,
    ReorderConversationsRequest, UpdateConversationRequest
)
from utils.websocket_manager import WebSocketManager
from utils.auth_middleware import get_current_user, get_current_user_optional, require_admin
from services.user_settings_kv_service import get_user_setting
from services.user_llm_provider_service import user_llm_provider_service
from services.model_source_resolver import (
    get_available_models as resolver_get_available_models,
    get_chat_selectable_model_ids_for_user,
    get_enabled_models as resolver_get_enabled_models,
    get_enabled_models_catalog_slice,
)
from services.admin_provider_registry import admin_provider_registry

from utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Global service references (managed by service container)
document_service = None
migration_service = None
chat_service = None

knowledge_graph_service = None
_neo4j_maint_task = None
_neo4j_maint_stop = None
_vector_maint_task = None
_vector_maint_stop = None
_presence_cleanup_task = None
collection_analysis_service = None
enhanced_pdf_segmentation_service = None
category_service = None

embedding_manager = None
websocket_manager = None

conversation_service = None
folder_service = None

# Import API routers
from api.settings_api import router as settings_router
from api.learning_api import router as learning_router

# Import FileManager service
from services.file_manager import get_file_manager

async def _ensure_agent_profiles_context_columns() -> None:
    """Ensure agent_profiles has context columns and model metadata (migrations 060/061/065/066, 081)."""
    logger.info("Ensuring agent_profiles context and model columns")
    try:
        conn = await asyncpg.connect(settings.DATABASE_URL)
        try:
            for column_name, default_sql in (
                ("include_user_context", "BOOLEAN NOT NULL DEFAULT false"),
                ("include_datetime_context", "BOOLEAN NOT NULL DEFAULT true"),
                ("include_user_facts", "BOOLEAN NOT NULL DEFAULT false"),
                ("include_facts_categories", "JSONB DEFAULT '[]'::jsonb"),
                ("model_source", "VARCHAR(50)"),
                ("model_provider_type", "VARCHAR(50)"),
            ):
                exists = await conn.fetchval(
                    """
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = 'agent_profiles' AND column_name = $1
                    """,
                    column_name,
                )
                if not exists:
                    await conn.execute(
                        f"ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS {column_name} {default_sql}"
                    )
                    logger.info("Added missing column agent_profiles.%s", column_name)
        finally:
            await conn.close()
    except Exception as e:
        logger.error("Could not ensure agent_profiles context columns: %s", e)


async def _refresh_oauth_tokens_on_startup() -> None:
    """Background task: refresh expired OAuth (email) tokens shortly after backend is ready."""
    await asyncio.sleep(3)
    max_attempts = 10
    delay = 5.0
    for attempt in range(max_attempts):
        try:
            from services.external_connections_service import external_connections_service
            result = await external_connections_service.refresh_all_expired_oauth_tokens()
            if result.get("refreshed") or result.get("failed"):
                logger.info(
                    "OAuth startup refresh: %s refreshed, %s failed",
                    result.get("refreshed", 0),
                    result.get("failed", 0),
                )
            return
        except Exception as e:
            logger.warning(
                "OAuth token refresh attempt %s/%s failed: %s",
                attempt + 1,
                max_attempts,
                e,
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay)


async def _get_user_integer_id(user_uuid: str) -> int:
    """Convert user UUID to integer primary key from users table"""
    try:
        # Use the settings service's database connection
        async with settings_service.async_session_factory() as session:
            from sqlalchemy import text
            result = await session.execute(
                text("SELECT id FROM users WHERE user_id = :user_uuid"),
                {"user_uuid": user_uuid}
            )
            row = result.fetchone()
            
            if not row:
                logger.error(f"❌ User not found for UUID: {user_uuid}")
                raise HTTPException(status_code=404, detail="User not found")
            
            user_integer_id = row[0]
            logger.debug(f"🔄 Converted user UUID {user_uuid} to integer ID {user_integer_id}")
            return user_integer_id
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to convert user UUID to integer ID: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve user ID")


async def retry_with_backoff(func, max_retries=5, base_delay=2, max_delay=30, service_name="service"):
    """Retry function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            delay = min(base_delay * (2 ** attempt), max_delay)
            if attempt < max_retries - 1:
                logger.warning(f"🔄 {service_name} connection attempt {attempt + 1} failed: {e}")
                logger.info(f"⏱️  Retrying {service_name} in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"❌ {service_name} failed after {max_retries} attempts: {e}")
                raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optimized application lifespan manager using service container"""
    # Startup
    logger.info("🚀 Starting Plato Knowledge Base Backend with Optimized Configuration...")
    
    global document_service, migration_service, chat_service
    global knowledge_graph_service, collection_analysis_service, enhanced_pdf_segmentation_service
    global _neo4j_maint_task, _neo4j_maint_stop, _vector_maint_task, _vector_maint_stop
    global _presence_cleanup_task
    global category_service, conversation_service, embedding_manager
    global websocket_manager
    global folder_service
    
    try:
        # Initialize the service container with optimized configuration
        optimization_config = {
            'worker_scaling': {
                'embedding_workers': 4,  # Reduced from 8
                'storage_workers': 2,    # Reduced from 4
                'document_workers': 6,   # Reduced from 12
                'process_workers': 2     # Reduced from 4
            }
        }
        
        service_container.config = optimization_config
        await service_container.initialize()

        if settings.IMAGE_VISION_REQUIRED and not settings.IMAGE_VISION_ENABLED:
            raise RuntimeError(
                "IMAGE_VISION_REQUIRED=true cannot be used with IMAGE_VISION_ENABLED=false"
            )
        if settings.IMAGE_VISION_REQUIRED:
            from clients.image_vision_client import get_image_vision_client

            _vision_client = await get_image_vision_client()
            await _vision_client.initialize(required=True)

        if settings.VECTOR_EMBEDDING_REQUIRED and not settings.VECTOR_EMBEDDING_ENABLED:
            raise RuntimeError(
                "VECTOR_EMBEDDING_REQUIRED=true cannot be used with VECTOR_EMBEDDING_ENABLED=false"
            )
        if settings.VECTOR_EMBEDDING_REQUIRED:
            from clients.vector_service_client import get_vector_service_client

            _vector_client = await get_vector_service_client(required=False)
            await _vector_client.initialize(required=True)

        # Get service references from container
        document_service = service_container.document_service
        chat_service = service_container.chat_service

        knowledge_graph_service = service_container.knowledge_graph_service
        from services.neo4j_maintenance import spawn_neo4j_maintenance_task

        _neo4j_maint_task, _neo4j_maint_stop = spawn_neo4j_maintenance_task(
            knowledge_graph_service
        )
        from services.vector_maintenance import spawn_vector_maintenance_task

        _vector_maint_task, _vector_maint_stop = spawn_vector_maintenance_task(
            service_container.embedding_manager
        )
        collection_analysis_service = service_container.collection_analysis_service
        enhanced_pdf_segmentation_service = service_container.enhanced_pdf_service
        category_service = service_container.category_service

        conversation_service = service_container.conversation_service
        embedding_manager = service_container.embedding_manager
        websocket_manager = service_container.websocket_manager
        folder_service = service_container.folder_service
        
        # Initialize migration service separately (not in container)
        from services.migration_service import MigrationService
        migration_service = MigrationService(service_container.document_repository)
        
        # Run migration from JSON to PostgreSQL
        logger.info("🔄 Running document migration from JSON to PostgreSQL...")
        migration_result = await migration_service.migrate_json_to_postgres()
        
        if migration_result["migrated_documents"] > 0:
            logger.info(f"✅ Migration completed: {migration_result['migrated_documents']} documents migrated")
        if migration_result["skipped_documents"] > 0:
            logger.info(f"ℹ️ {migration_result['skipped_documents']} documents already existed in database")
        if migration_result["failed_migrations"] > 0:
            logger.error(f"❌ Migration had {migration_result['failed_migrations']} failures")
        if migration_result["errors"]:
            for error in migration_result["errors"]:
                logger.error(f"❌ Migration error: {error}")

        await _ensure_agent_profiles_context_columns()
        await ensure_user_memory_schema_columns()

        # Seed Agent Factory built-in skills (idempotent; no-op if table not yet migrated)
        try:
            from services.agent_skills_service import seed_builtin_skills
            await seed_builtin_skills()
            logger.info("Agent Factory built-in skills seeded")
        except Exception as e:
            logger.debug("Agent Factory skills seed skipped or failed (table may not exist yet): %s", e)

        async def _reembed_user_custom_skill_vectors():
            try:
                from services.database_manager.database_helpers import fetch_all
                from services.skill_vector_service import sync_all_skills

                rows = await fetch_all(
                    """
                    SELECT DISTINCT user_id FROM agent_skills
                    WHERE COALESCE(is_builtin, false) = false
                      AND user_id IS NOT NULL
                      AND LENGTH(TRIM(user_id::text)) > 0
                    """
                )
                total = 0
                for row in rows:
                    uid = row.get("user_id")
                    if not uid:
                        continue
                    try:
                        n = await sync_all_skills(user_id=str(uid), upsert_only=True)
                        total += n
                    except Exception as inner:
                        logger.warning(
                            "User skill vector sync failed for user %s (non-fatal): %s",
                            uid,
                            inner,
                        )
                if total > 0:
                    logger.info(
                        "User-defined skills vector sync after collection migration: %d embeddings",
                        total,
                    )
            except Exception as e:
                logger.warning("User-defined skills vector re-sync failed (non-fatal): %s", e)

        async def _sync_builtin_skill_vectors():
            await asyncio.sleep(3)
            try:
                from services.skill_vector_service import ensure_skills_collection, sync_all_skills

                ok, migrated = await ensure_skills_collection()
                if not ok:
                    logger.warning("Built-in skills vector sync skipped (collection unavailable)")
                    return
                count = await sync_all_skills(user_id=None, upsert_only=True)
                logger.info("Built-in skills vector sync: embedded %d skills", count)
                if migrated:
                    await _reembed_user_custom_skill_vectors()
            except Exception as e:
                logger.warning("Built-in skills vector sync failed (non-fatal): %s", e)

        asyncio.create_task(_sync_builtin_skill_vectors())

        async def _sync_help_docs():
            await asyncio.sleep(5)
            try:
                from services.help_docs_sync_service import HelpDocsSyncService
                await HelpDocsSyncService().sync()
                logger.info("Help docs vector sync complete")
            except Exception as e:
                logger.warning("Help docs vector sync failed (non-fatal): %s", e)

        asyncio.create_task(_sync_help_docs())

        async def _check_dimension_migration():
            """Auto-recreate document collections and queue re-embed when EMBEDDING_DIMENSIONS changes."""
            await asyncio.sleep(8)
            try:
                import json as _json
                from pathlib import Path as _Path
                from services.document_vector_audit_service import recreate_document_collections_and_queue

                marker_path = _Path(settings.UPLOAD_DIR) / ".embedding_dimensions_marker"
                current_dims = settings.EMBEDDING_DIMENSIONS

                previous_dims = None
                if marker_path.exists():
                    try:
                        data = _json.loads(marker_path.read_text(encoding="utf-8"))
                        previous_dims = data.get("dimensions")
                    except Exception:
                        pass

                if previous_dims == current_dims:
                    return

                if previous_dims is None:
                    marker_path.parent.mkdir(parents=True, exist_ok=True)
                    marker_path.write_text(
                        _json.dumps({"dimensions": current_dims}), encoding="utf-8"
                    )
                    logger.info("Embedding dimensions marker initialized: %d", current_dims)
                    return

                logger.warning(
                    "EMBEDDING_DIMENSIONS changed: %d -> %d. Queuing document re-embed.",
                    previous_dims,
                    current_dims,
                )
                result = await recreate_document_collections_and_queue(
                    scope="all",
                    user_id=None,
                    team_id=None,
                    dry_run=False,
                    queue_reembed=True,
                    throttle_seconds=0.2,
                    max_concurrent=5,
                    include_all_qdrant_embedding_collections=True,
                )
                if result.get("success"):
                    marker_path.write_text(
                        _json.dumps({"dimensions": current_dims}), encoding="utf-8"
                    )
                    logger.info(
                        "Dimension migration: recreated %d collections, queued %d documents",
                        len(result.get("collections_recreated", [])),
                        result.get("queued_for_reembed", 0),
                    )
                    q_seen = result.get("qdrant_embedding_collections_discovered") or []
                    if q_seen:
                        logger.info(
                            "Dimension migration: Qdrant listed %d text-embedding collection(s)",
                            len(q_seen),
                        )
                else:
                    logger.error(
                        "Dimension migration failed (will retry on next restart): %s",
                        result.get("errors"),
                    )
            except Exception as e:
                logger.warning("Dimension migration check failed (non-fatal): %s", e)

        asyncio.create_task(_check_dimension_migration())

        # Seed default built-in agent profile for all users (idempotent; Bastion Assistant only)
        try:
            from services.agent_factory_service import seed_default_agent_profiles

            await seed_default_agent_profiles()
            logger.info("Default Bastion Assistant agent profile seeded (where missing)")
        except Exception as e:
            logger.debug("Default agent profile seed skipped or failed: %s", e)

        # Initialize settings service (if not already initialized by service container)
        try:
            if not hasattr(settings_service, '_initialized') or not settings_service._initialized:
                await settings_service.initialize()
                logger.info("⚙️ Settings Service initialized")
            else:
                logger.info("⚙️ Settings Service already initialized by service container")
        except Exception as e:
            logger.error(f"❌ Settings Service initialization failed: {e}")
            # Don't raise here as the app can still function with limited settings

        # Initialize folder service
        try:
            from services.folder_service import FolderService
            folder_service = FolderService()
            await folder_service.initialize()
            logger.info("📁 Folder Service initialized")
        except Exception as e:
            logger.error(f"❌ Folder Service initialization failed: {e}")
            folder_service = None
        
        # Messaging WebSocket - Initialize messaging service
        try:
            from services.messaging.messaging_service import messaging_service
            await messaging_service.initialize(service_container.db_pool)
            logger.info("Messaging service initialized")
        except Exception as e:
            logger.error(f"❌ Messaging Service initialization failed: {e}")
            # Don't raise here as the app can still function without messaging
        
        # Initialize Teams Services
        try:
            from services.team_service import TeamService
            from services.team_invitation_service import TeamInvitationService
            from services.team_post_service import TeamPostService
            from api.teams_api import team_service, invitation_service, post_service
            
            # Initialize team service with messaging service
            await team_service.initialize(
                shared_db_pool=service_container.db_pool,
                messaging_service=messaging_service
            )
            
            # Initialize invitation service
            await invitation_service.initialize(
                shared_db_pool=service_container.db_pool,
                messaging_service=messaging_service,
                team_service=team_service
            )
            
            # Initialize post service
            await post_service.initialize(
                shared_db_pool=service_container.db_pool,
                team_service=team_service
            )
            
            logger.info("✅ Teams Services initialized")
        except Exception as e:
            logger.error(f"❌ Teams Services initialization failed: {e}")
            # Don't raise here as the app can still function without teams

        
        # Initialize available models from admin providers on startup
        logger.info("Fetching available models from admin providers on startup...")
        try:
            available_models = await admin_provider_registry.get_all_admin_models()
            logger.info("Fetched %s models from admin registry", len(available_models))
            enabled_models = await settings_service.get_enabled_models()
            if len(enabled_models) == 0:
                logger.warning("No models are currently enabled. Admin must configure models in Settings before chat functionality will work.")
        except Exception as e:
            logger.warning("Failed to fetch models on startup: %s", e)
        
        # Get system status after migration
        try:
            stats = await document_service.get_documents_stats()
            logger.info(f"📊 System Status:")
            logger.info(f"   Total documents: {stats.get('total_documents', 0)}")
            logger.info(f"   Completed documents: {stats.get('completed_documents', 0)}")
            logger.info(f"   Processing documents: {stats.get('processing_documents', 0)}")
            logger.info(f"   Failed documents: {stats.get('failed_documents', 0)}")
            logger.info(f"   Total embeddings: {stats.get('total_embeddings', 0)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to get system status: {e}")
        
        logger.info("✅ All services initialized successfully with optimized configuration")
        
        # Start File System Watcher
        try:
            from services.file_watcher_service import get_file_watcher
            file_watcher = await get_file_watcher()
            await file_watcher.start()
            logger.info("👀 File System Watcher started - monitoring uploads directory")
        except Exception as e:
            logger.error(f"❌ Failed to start File System Watcher: {e}")
            file_watcher = None

        # Chat bot connections are self-restored by connections-service on startup
        # and periodically re-synced by Celery Beat (sync_chat_bot_connections).

        # Proactively refresh expired OAuth (email) tokens on startup
        asyncio.create_task(_refresh_oauth_tokens_on_startup())

        async def _presence_stale_cleanup_loop():
            from services.messaging.messaging_service import messaging_service
            from utils.websocket_manager import get_websocket_manager

            while True:
                try:
                    await asyncio.sleep(35)
                    user_ids = await messaging_service.cleanup_stale_presence()
                    if not user_ids:
                        continue
                    ws_mgr = get_websocket_manager()
                    for uid in user_ids:
                        try:
                            await ws_mgr.broadcast_presence_update(uid, "offline")
                        except Exception as broadcast_err:
                            logger.warning(
                                "Presence offline broadcast failed for %s: %s",
                                uid,
                                broadcast_err,
                            )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning("Presence stale cleanup loop error (non-fatal): %s", e)

        _presence_cleanup_task = asyncio.create_task(_presence_stale_cleanup_loop())
        logger.info("Presence stale cleanup background task started")

        # gRPC Tool Service moved to dedicated tools-service container

        from services.document_status_redis_subscriber import (
            document_status_redis_subscriber,
        )

        document_status_redis_subscriber.start()
        logger.info(
            "Document status Redis subscriber started (channel=%s)",
            settings.DOCUMENT_STATUS_REDIS_CHANNEL,
        )

    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    
    # Service is ready to accept requests
    logger.info("✅ Tools-Service Up - FastAPI application ready to serve requests")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Plato Knowledge Base...")

    if _presence_cleanup_task:
        _presence_cleanup_task.cancel()
        try:
            await _presence_cleanup_task
        except asyncio.CancelledError:
            pass
        _presence_cleanup_task = None

    from services.neo4j_maintenance import cancel_neo4j_maintenance_task

    await cancel_neo4j_maintenance_task(_neo4j_maint_task, _neo4j_maint_stop)
    _neo4j_maint_task = None
    _neo4j_maint_stop = None

    from services.vector_maintenance import cancel_vector_maintenance_task

    await cancel_vector_maintenance_task(_vector_maint_task, _vector_maint_stop)
    _vector_maint_task = None
    _vector_maint_stop = None

    from services.document_status_redis_subscriber import (
        document_status_redis_subscriber,
    )

    await document_status_redis_subscriber.stop()
    
    # Stop File System Watcher
    try:
        from services.file_watcher_service import get_file_watcher
        file_watcher = await get_file_watcher()
        await file_watcher.stop()
    except Exception as e:
        logger.error(f"❌ Error stopping File System Watcher: {e}")
    
    # Close migration service
    if migration_service:
        await migration_service.close()
    
    # Close service container (handles all other services)
    await service_container.close()
    
    # Close singleton services
    from services.auth_service import auth_service
    await auth_service.close()
    await settings_service.close()
    
    logger.info("👋 Plato Knowledge Base shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Bastion Workspace V2",
    description="A sophisticated RAG system with PostgreSQL-backed document storage and knowledge graph integration",
    version=__version__,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add authentication middleware
from utils.auth_middleware import AuthenticationMiddleware
app.add_middleware(AuthenticationMiddleware)

# Global exception handler to ensure JSON responses
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Specific handler for request validation errors"""
    logger.error(f"❌ RequestValidationError on {request.url}: {exc.errors()}")
    logger.error(f"❌ Request method: {request.method}")
    logger.error(f"❌ Request body: {await request.body()}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "errors": exc.errors()}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler with detailed logging"""
    logger.error(f"❌ Global exception handler caught: {type(exc).__name__}: {str(exc)}")
    
    # Special handling for Pydantic validation errors
    if hasattr(exc, 'errors') and hasattr(exc, 'model'):
        logger.error(f"❌ Pydantic validation error for model: {exc.model}")
        logger.error(f"❌ Validation errors: {exc.errors}")
        for error in exc.errors():
            logger.error(f"❌ Field: {error.get('loc', 'unknown')}, Error: {error.get('msg', 'unknown')}, Type: {error.get('type', 'unknown')}")
        return JSONResponse(
            status_code=422,
            content={"detail": "Validation error", "errors": exc.errors()}
        )
    
    # Special handling for RequestValidationError
    if "RequestValidationError" in str(type(exc)):
        logger.error(f"❌ Request validation error: {exc}")
        logger.error(f"❌ Request body: {getattr(exc, 'body', 'No body')}")
        logger.error(f"❌ Request URL: {request.url}")
        logger.error(f"❌ Request method: {request.method}")
        return JSONResponse(
            status_code=422,
            content={"detail": "Request validation error", "error": str(exc)}
        )
    
    import traceback
    logger.error(f"❌ Full traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Mount static files for serving PDF images
import os
from pathlib import Path
pdf_images_path = Path("processed/pdf_images")
pdf_images_path.mkdir(parents=True, exist_ok=True)
app.mount("/api/files", StaticFiles(directory=str(pdf_images_path)), name="pdf_images")

# Web sources: single root static mount.
# All runtime content (generated images, crawled images/docs, bulk scrape) is under
# WEB_SOURCES_ROOT on the backend volume.  Comics are regular library images in
# UPLOAD_DIR/Global/Comics/ served by /api/images/; web_sources/comics/ is not used.
ws_root = Path(settings.WEB_SOURCES_ROOT)
ws_root.mkdir(parents=True, exist_ok=True)
images_path = ws_root / "images"
images_path.mkdir(parents=True, exist_ok=True)
app.mount("/api/web-sources", StaticFiles(directory=str(ws_root)), name="web_sources")


@app.get("/static/images/{path:path}")
async def _redirect_legacy_static_images(path: str):
    """301 to unified web_sources static mount (preserves old /static/images/... URLs)."""
    suffix = path.strip("/")
    target = f"/api/web-sources/images/{suffix}" if suffix else "/api/web-sources/images/"
    return RedirectResponse(target, status_code=301)


@app.get("/api/images/{filename:path}")
async def serve_image(
    filename: str,
    request: Request,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Serve images with proper content-type headers for browser display.
    Supports JPG, PNG, GIF, WEBP, and other image formats.
    
    SECURITY: Requires authentication and verifies user has access to the image
    by checking if it's associated with a document the user owns or has access to.
    """
    import mimetypes
    from urllib.parse import unquote
    from fastapi import HTTPException
    
    try:
        # URL decode the filename in case it's encoded
        decoded_filename = unquote(filename)
        
        # Security: Prevent path traversal - get just the basename
        safe_filename = os.path.basename(decoded_filename)
        if not safe_filename or safe_filename in ('.', '..'):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        logger.info(f"🖼️ Serving image: {safe_filename} for user: {current_user.username}")
        
        # SECURITY: Check if this image is associated with a document
        # If so, verify the user has access to that document and resolve file path
        from services.database_manager.database_helpers import fetch_all
        from services.service_container import get_service_container
        from api.document_api import check_document_access
        
        has_access = False
        image_file_path = None
        doc_info = None
        library_stream_doc_id: Optional[str] = None

        # Search for documents with this filename using RLS context
        # Use user context so RLS allows access to global documents
        rls_context = {'user_id': current_user.user_id, 'user_role': 'user'}
        doc_query = """
            SELECT document_id, user_id, collection_type, folder_id
            FROM document_metadata
            WHERE filename = $1
            LIMIT 10
        """
        doc_rows = await fetch_all(doc_query, safe_filename, rls_context=rls_context)
        
        if doc_rows:
            # Image is associated with one or more documents - check access
            for doc_row in doc_rows:
                doc_id = doc_row.get('document_id')
                if doc_id:
                    try:
                        # Check if user has access to this document
                        doc_info = await check_document_access(doc_id, current_user, "read")
                        if doc_info:
                            has_access = True
                            logger.info(f"✅ User {current_user.username} has access via document {doc_id}")
                            
                            # Resolve file path using folder service
                            container = await get_service_container()
                            folder_service = container.folder_service
                            
                            folder_id = getattr(doc_info, 'folder_id', None)
                            user_id = getattr(doc_info, 'user_id', None)
                            collection_type = getattr(doc_info, 'collection_type', 'user')
                            
                            file_path = await folder_service.get_document_file_path(
                                filename=safe_filename,
                                folder_id=folder_id,
                                user_id=user_id,
                                collection_type=collection_type
                            )

                            uploads_base = Path(settings.UPLOAD_DIR).resolve()
                            if file_path:
                                try:
                                    fp_resolved = file_path.resolve()
                                except Exception:
                                    fp_resolved = file_path
                                if str(fp_resolved).startswith(str(uploads_base)):
                                    library_stream_doc_id = doc_id
                                    image_file_path = file_path
                                    logger.info(
                                        "✅ Library image for doc %s — streaming from document-service",
                                        doc_id,
                                    )
                                    break
                                if file_path.exists():
                                    image_file_path = file_path
                                    logger.info(f"✅ Resolved file path: {image_file_path}")
                                    break
                                logger.warning(
                                    "⚠️ File path resolved but not on backend disk: %s", file_path
                                )
                            else:
                                logger.warning("⚠️ Could not resolve file path for document %s", doc_id)
                                # Continue checking other documents
                    except HTTPException:
                        # User doesn't have access to this document, continue checking others
                        continue
                    except Exception as e:
                        logger.error(f"❌ Error resolving file path for document {doc_id}: {e}")
                        continue
        
        # If not found in documents, check if image is in a document_id subdirectory
        # and verify access to that document (fallback for old-style images)
        if not has_access or not image_file_path:
            # Construct file path
            fallback_path = images_path / safe_filename
            
            # Check subdirectories (some images may be in document_id subdirectories)
            if not fallback_path.exists():
                found_path = False
                for subdir in images_path.iterdir():
                    if subdir.is_dir():
                        potential_path = subdir / safe_filename
                        if potential_path.exists():
                            fallback_path = potential_path
                            found_path = True
                            # Check if subdirectory name is a document_id
                            potential_doc_id = subdir.name
                            try:
                                doc_info = await check_document_access(potential_doc_id, current_user, "read")
                                if doc_info:
                                    has_access = True
                                    image_file_path = fallback_path
                                    logger.info(f"✅ User {current_user.username} has access via document_id subdirectory {potential_doc_id}")
                                    break
                            except HTTPException:
                                # Not a valid document_id or no access, continue
                                continue
                
                if not found_path and not image_file_path:
                    logger.warning(f"❌ Image not found: {safe_filename}")
                    raise HTTPException(status_code=404, detail="Image not found")
            else:
                # Image is in root directory - for standalone generated images,
                # we allow access if user is authenticated (generated images are
                # typically shown in user's own conversations)
                if not has_access:
                    has_access = True
                    image_file_path = fallback_path
                    logger.info(f"✅ Allowing access to standalone generated image: {safe_filename}")
        
        if not has_access:
            logger.warning(f"❌ Access denied: User {current_user.username} does not have access to image {safe_filename}")
            raise HTTPException(status_code=403, detail="Access denied")

        if library_stream_doc_id:
            from clients.document_service_client import get_document_service_client

            dsc = get_document_service_client()
            await dsc.initialize(required=True)

            media_type, _ = mimetypes.guess_type(safe_filename)
            if not media_type:
                ext = Path(safe_filename).suffix.lower()
                media_type_map = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                    ".bmp": "image/bmp",
                    ".svg": "image/svg+xml",
                }
                media_type = media_type_map.get(ext, "application/octet-stream")

            async def _library_image_bytes():
                async for ch in dsc.download_document_stream(
                    library_stream_doc_id,
                    current_user.user_id,
                    role=getattr(current_user, "role", "") or "",
                ):
                    if ch.data:
                        yield ch.data

            logger.info(
                "✅ Streaming library image %s (doc %s) content-type: %s",
                safe_filename,
                library_stream_doc_id,
                media_type,
            )
            return StreamingResponse(
                _library_image_bytes(),
                media_type=media_type,
                headers={"Content-Disposition": f'inline; filename="{safe_filename}"'},
            )

        if not image_file_path:
            logger.error(f"❌ File path not resolved for image: {safe_filename}")
            raise HTTPException(status_code=404, detail="Image not found")

        # SECURITY: Verify resolved path is under document library or web_sources images
        try:
            uploads_base = Path(settings.UPLOAD_DIR).resolve()
            web_sources_images = (Path(settings.WEB_SOURCES_ROOT) / "images").resolve()
            image_file_path_resolved = image_file_path.resolve()
            resolved_str = str(image_file_path_resolved)
            if not (
                resolved_str.startswith(str(uploads_base))
                or resolved_str.startswith(str(web_sources_images))
            ):
                logger.error(f"Path traversal attempt detected: {image_file_path_resolved}")
                raise HTTPException(status_code=403, detail="Access denied")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Path validation error: {e}")
            raise HTTPException(status_code=403, detail="Access denied")

        # Determine media type from file extension
        media_type, _ = mimetypes.guess_type(str(image_file_path))
        if not media_type:
            # Fallback: check common image extensions
            ext = image_file_path.suffix.lower()
            media_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.bmp': 'image/bmp',
                '.svg': 'image/svg+xml',
            }
            media_type = media_type_map.get(ext, 'application/octet-stream')

        logger.info(f"✅ Serving image {safe_filename} with content-type: {media_type}")

        # Serve file with proper content-type (local web_sources volume only; library uses DS stream above)
        return FileResponse(
            path=str(image_file_path),
            filename=safe_filename,
            media_type=media_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to serve image: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# Include unified chat API routes
from api.unified_chat_api import router as unified_chat_router
app.include_router(unified_chat_router)
logger.debug("✅ Unified chat API routes registered")



# Include classification model API routes
from api.classification_model_api import router as classification_model_router
app.include_router(classification_model_router)
logger.debug("✅ Classification model API routes registered")

# Include text completion model API routes
from api.text_completion_model_api import router as text_completion_model_router
app.include_router(text_completion_model_router)
logger.debug("✅ Text completion model API routes registered")

# Learning API routes
app.include_router(learning_router)
logger.debug("✅ Learning API routes registered")

# Include admin API routes
from api.admin_api import router as admin_router
from api.admin_vector_audit_api import router as admin_vector_audit_router
app.include_router(admin_router)
app.include_router(admin_vector_audit_router)
logger.debug("✅ Admin API routes registered")

from api.federation_api import router as federation_router
app.include_router(federation_router)
logger.debug("✅ Federation API routes registered")

# Include resilient embedding API routes
from api.resilient_embedding_api import router as resilient_embedding_router
app.include_router(resilient_embedding_router)
logger.debug("✅ Resilient embedding API routes registered")

# Include settings API routes
app.include_router(settings_router)

from api.dashboard_api import router as dashboard_router
app.include_router(dashboard_router)
from api.scratchpad_api import router as scratchpad_router
app.include_router(scratchpad_router)
from api.saved_artifact_api import router as saved_artifact_router, public_router as saved_artifact_public_router
app.include_router(saved_artifact_router)
app.include_router(saved_artifact_public_router)
from api.document_pins_api import router as document_pins_router
app.include_router(document_pins_router)
logger.debug("Home dashboard API routes registered")
from api.device_proxy_api import router as device_proxy_router
app.include_router(device_proxy_router)

# Code workspaces (local proxy-backed)
from api.code_workspace_api import router as code_workspace_router
app.include_router(code_workspace_router)

# User LLM providers (per-user API keys and models)
from api.user_llm_provider_api import router as user_llm_provider_router
app.include_router(user_llm_provider_router)
logger.debug("User LLM provider API routes registered")

from api.user_voice_provider_api import router as user_voice_provider_router
app.include_router(user_voice_provider_router)
logger.debug("User voice provider API routes registered")

# Template management removed - functionality not in use
logger.debug("✅ Settings API routes registered")

# Services API removed - Twitter integration removed
logger.debug("✅ Services API routes registered")
logger.debug("✅ Template management API routes registered")
logger.debug("✅ Template execution API routes registered")

# Include Export API routes
from api.export_api import router as export_router
app.include_router(export_router)
logger.debug("✅ Export API routes registered")

# Org search API router
from api.org_search_api import router as org_search_router
app.include_router(org_search_router)
logger.debug("✅ Org Search API routes registered")

# Org Quick Capture API
from api.org_capture_api import router as org_capture_router
app.include_router(org_capture_router)
logger.debug("✅ Org Capture API routes registered")

# Org Journal API (journal-for-the-day get/update)
from api.org_journal_api import router as org_journal_router
app.include_router(org_journal_router)
logger.debug("✅ Org Journal API routes registered")

# Universal Org Todo API
from api.org_todo_api import router as org_todo_router
app.include_router(org_todo_router)
logger.debug("✅ Org Todo API routes registered")

# Include org settings API
from api.org_settings_api import router as org_settings_router
app.include_router(org_settings_router)
from api.zettelkasten_api import router as zettelkasten_router
app.include_router(zettelkasten_router)
logger.debug("✅ Org Settings API routes registered")

# Include org tag API
from api.org_tag_api import router as org_tag_router
app.include_router(org_tag_router)
logger.debug("✅ Org Tag API routes registered")

# Calendar API (Agenda view: O365, future CalDAV)
from api.calendar_api import router as calendar_router
app.include_router(calendar_router)
logger.debug("Calendar API routes registered")

# Contacts API (Contacts view: O365 + org-mode)
from api.contacts_api import router as contacts_router
app.include_router(contacts_router)
logger.debug("Contacts API routes registered")

# Include editor API
from api.editor_api import router as editor_router
app.include_router(editor_router)
logger.debug("✅ Editor API routes registered")

# Research plan API routes removed - migrated to LangGraph subgraph workflows

# Include agent API routes

logger.debug("✅ Agent API routes registered")

# Context-aware research API routes removed - migrated to LangGraph subgraph workflows

# Deprecated LangGraph APIs removed - using async_orchestrator_api for all LangGraph functionality
# Orchestrator chat API removed - deprecated endpoint that returned 410 errors

# Include Async Orchestrator API routes
from api.async_orchestrator_api import router as async_orchestrator_router
app.include_router(async_orchestrator_router)
logger.debug("✅ Async Orchestrator API routes registered")

# gRPC Orchestrator Proxy (Phase 5 - Microservices)
from api.grpc_orchestrator_proxy import router as grpc_orchestrator_proxy_router
app.include_router(grpc_orchestrator_proxy_router)
logger.debug("✅ gRPC Orchestrator Proxy routes registered (Phase 5)")

# Agent Factory (Workflow Composer action registry)
from api.agent_factory_api import router as agent_factory_router
from api.agent_line_api import router as agent_line_router
from api.agent_line_analytics_api import router as agent_line_analytics_router
from api.mcp_servers_api import router as mcp_servers_router
app.include_router(agent_factory_router)
app.include_router(mcp_servers_router)
app.include_router(agent_line_router, prefix="/api/agent-factory")
app.include_router(agent_line_analytics_router, prefix="/api/agent-factory")
logger.debug("✅ Agent Factory API routes registered")

# Control Panes (status bar custom controls)
from api.control_panes_api import router as control_panes_router
app.include_router(control_panes_router)
logger.debug("✅ Control Panes API routes registered")

# Browser Auth (interactive login capture for playbooks)
from api.browser_auth_api import router as browser_auth_router
app.include_router(browser_auth_router)
logger.debug("✅ Browser Auth API routes registered")

# Include Conversation API routes (moved from main)
from api.conversation_api import router as conversation_router
app.include_router(conversation_router)
logger.debug("✅ Conversation API routes registered")

# Include Conversation Sharing API routes
from api.conversation_sharing_api import router as conversation_sharing_router
app.include_router(conversation_sharing_router)

# Include Document API routes
from api.document_api import router as document_router, check_document_access
app.include_router(document_router)
# Document version history (list, content, diff, rollback)
from api.document_version_api import router as document_version_router
app.include_router(document_version_router)
from api.document_sharing_api import router as document_sharing_router
app.include_router(document_sharing_router)
from api.collab_api import router as collab_router
app.include_router(collab_router)
from api.document_encryption_api import router as document_encryption_router
app.include_router(document_encryption_router)
logger.debug("✅ Document API routes registered")

# Include Graph API routes (link-graph for file relation cloud)
from api.graph_api import router as graph_router
app.include_router(graph_router)
logger.debug("✅ Graph API routes registered")

# Include Image Metadata API routes
try:
    from api.image_metadata_api import router as image_metadata_router
    app.include_router(image_metadata_router)
    logger.info("✅ Image Metadata API routes registered")
    # Log registered routes for debugging
    for route in image_metadata_router.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            logger.info(f"  📍 Route: {list(route.methods)} {route.path}")
except Exception as e:
    logger.error(f"❌ Failed to register Image Metadata API routes: {e}")
    import traceback
    traceback.print_exc()

# Include Document Metadata API routes (doc-metadata sidecars, generate-summary)
try:
    from api.document_metadata_api import router as document_metadata_router
    app.include_router(document_metadata_router)
    logger.info("✅ Document Metadata API routes registered")
except Exception as e:
    logger.error(f"❌ Failed to register Document Metadata API routes: {e}")
    import traceback
    traceback.print_exc()

# Include Folder API routes
from api.folder_api import router as folder_router
app.include_router(folder_router)
logger.debug("✅ Folder API routes registered")

# Include Location API routes
from api.location_api import router as location_router
app.include_router(location_router)
logger.debug("✅ Location API routes registered")

# Include GeoJSON API routes (map layers: locations, future articles/events)
from api.geojson_api import router as geojson_router
app.include_router(geojson_router)
logger.debug("✅ GeoJSON API routes registered")

# Include Routes API (OSRM routing and saved routes)
from api.routes_api import router as routes_router
app.include_router(routes_router)
logger.debug("✅ Routes API routes registered")

# Include OCR API routes

# Include Search API routes
from api.search_api import router as search_router
app.include_router(search_router)
logger.debug("✅ Search API routes registered")

# Include Segmentation API routes

# Include PDF Text API routes

# Include Category API routes
from api.category_api import router as category_router
app.include_router(category_router)
logger.debug("✅ Category API routes registered")


# Conversation create endpoint moved to api/conversation_api.py
# Agent Chaining API removed - deprecated functionality, all agents migrated to llm-orchestrator

# Include RSS API routes
from api.rss_api import router as rss_router
app.include_router(rss_router)

from api.ebooks_api import router as ebooks_router
app.include_router(ebooks_router)

# Include Oregon Trail game API
try:
    from api.oregon_trail_api import router as oregon_trail_router
    app.include_router(oregon_trail_router)
    logger.debug("Oregon Trail API routes registered")
except Exception as e:
    logger.warning(f"Oregon Trail API routes skipped: {e}")

logger.debug("✅ RSS API routes registered")

if settings.GREADER_API_ENABLED:
    from api.greader_api import router as greader_router

    app.include_router(greader_router, prefix="/api/greader")
    logger.debug("✅ GReader API routes registered")

# Include FileManager API routes
from api.file_manager_api import router as file_manager_router
app.include_router(file_manager_router)
logger.debug("✅ FileManager API routes registered")

# Include Authentication API routes
from api.auth_api import router as auth_router
app.include_router(auth_router)
logger.debug("✅ Authentication API routes registered")

# Messaging WebSocket API
from api.messaging_api import router as messaging_router
app.include_router(messaging_router)
logger.debug("Messaging API routes registered")

# Teams API
from api.teams_api import router as teams_router
app.include_router(teams_router)
logger.debug("✅ Teams API routes registered")

# Include Data Workspace API routes
from api.data_workspace_api import router as data_workspace_router
app.include_router(data_workspace_router)
logger.debug("✅ Data Workspace API routes registered")

# Include Audio Transcription API routes
try:
    from api.audio_api import router as audio_router
    app.include_router(audio_router)
    logger.debug("✅ Audio API routes registered")
except Exception as e:
    logger.warning(f"⚠️ Failed to register Audio API routes: {e}")

# Include Voice API routes (TTS)
try:
    from api.voice_api import router as voice_router
    app.include_router(voice_router)
    logger.debug("✅ Voice API routes registered")
except Exception as e:
    logger.warning("⚠️ Failed to register Voice API routes: %s", e)

try:
    from api.audio_export_api import router as audio_export_router
    app.include_router(audio_export_router)
    logger.debug("✅ Audio export API routes registered")
except Exception as e:
    logger.warning("⚠️ Failed to register Audio export API routes: %s", e)

# Include Projects API routes
from api.projects_api import router as projects_router
app.include_router(projects_router)
logger.debug("✅ Projects API routes registered")

# Include Status Bar API routes
from api.status_bar_api import router as status_bar_router
app.include_router(status_bar_router)
logger.debug("✅ Status Bar API routes registered")

# Include Help API routes
from api.help_api import router as help_router
app.include_router(help_router)
logger.debug("✅ Help API routes registered")

# Include Music API routes
from api.music_api import router as music_router
app.include_router(music_router)
logger.debug("✅ Music API routes registered")

from api.emby_api import router as emby_router
app.include_router(emby_router)
logger.debug("Emby API routes registered")

# External connections (OAuth / email)
from api.external_connections_api import router as external_connections_router
from api.internal_chat_api import router as internal_chat_router
app.include_router(external_connections_router, prefix="/api")
app.include_router(internal_chat_router)
logger.debug("✅ External connections API routes registered")

try:
    from api.email_api import router as email_router
    app.include_router(email_router)
    logger.debug("✅ Email API routes registered")
except Exception as e:
    logger.warning("⚠️ Failed to register Email API routes: %s", e)

# Include HITL Orchestrator API routes
# HITL orchestrator API removed - using official orchestrator
logger.debug("✅ Legacy HITL Orchestrator API removed")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Bastion Workspace V2",
        "version": __version__,
        "storage": "PostgreSQL"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "message": "Welcome to Bastion Workspace V2",
        "description": "A sophisticated RAG system with PostgreSQL-backed document storage",
        "docs": "/docs",
        "health": "/health",
        "version": __version__
    }





@app.get("/api/health/websockets")
async def websocket_health():
    """Health check for WebSocket connections"""
    return {
        "status": "healthy",
        "total_connections": websocket_manager.get_connection_count(),
        "session_connections": websocket_manager.get_session_count(),
        "job_connections": len(websocket_manager.job_connections),
        "job_connection_details": {
            job_id: len(connections) for job_id, connections in websocket_manager.job_connections.items()
        }
    }


@app.get("/api/health/websocket-test/{job_id}")
async def websocket_test_endpoint(job_id: str):
    """Test endpoint to verify WebSocket routing is working"""
    return {
        "message": "WebSocket endpoint should be reachable",
        "job_id": job_id,
        "websocket_path": f"/api/ws/job-progress/{job_id}",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/health/jwt-test")
async def jwt_test_endpoint(token: str):
    """Test endpoint to verify JWT token decoding is working"""
    try:
        from utils.auth_middleware import decode_jwt_token
        payload = decode_jwt_token(token)
        return {
            "message": "JWT token decoded successfully",
            "user_id": payload.get("user_id"),
            "username": payload.get("username"),
            "role": payload.get("role"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "message": "JWT token decode failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.websocket("/api/ws/test")
async def websocket_test(websocket: WebSocket):
    """Simple test WebSocket endpoint without authentication"""
    logger.info("🔌 Test WebSocket connection attempt")
    try:
        await websocket.accept()
        logger.info("✅ Test WebSocket accepted")
        
        await websocket.send_json({
            "type": "test",
            "message": "WebSocket connection successful",
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            while True:
                data = await websocket.receive_text()
                await websocket.send_json({
                    "type": "echo",
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                })
        except WebSocketDisconnect:
            logger.info("📡 Test WebSocket disconnected")
            
    except Exception as e:
        logger.error(f"❌ Test WebSocket error: {e}")
        try:
            await websocket.close(code=4000, reason="Test failed")
        except Exception:
            pass





@app.get("/api/health/document-processor")
async def document_processor_health():
    """Health check for DocumentProcessor singleton"""
    try:
        from utils.document_processor import DocumentProcessor
        processor = DocumentProcessor.get_instance()
        status = processor.get_status()
        
        return {
            "status": "healthy" if status["initialized"] else "not_initialized",
            "document_processor": status,
            "singleton_info": {
                "instance_id": status["instance_id"],
                "spacy_model": status["spacy_model"],
                "ocr_service_available": status["ocr_service_loaded"]
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# ===== MODEL CONFIGURATION ENDPOINTS =====

@app.get("/api/models/available", response_model=AvailableModelsResponse)
async def get_available_models(
    current_user: Optional[AuthenticatedUserResponse] = Depends(get_current_user_optional),
):
    """Get list of available models. When user has use_admin_models=false, returns models from their
    own providers; otherwise returns admin provider registry (OpenRouter, OpenAI, Groq from env)."""
    try:
        user_id = current_user.user_id if current_user else None
        try:
            models = await resolver_get_available_models(user_id)
            return AvailableModelsResponse(models=models)
        except Exception as e:
            logger.warning("Resolver get_available_models failed, returning empty: %s", e)
            return AvailableModelsResponse(models=[])
    except Exception as e:
        logger.error("Failed to get available models: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/enabled")
async def get_enabled_models(
    current_user: Optional[AuthenticatedUserResponse] = Depends(get_current_user_optional),
):
    """Get full list of enabled model IDs. When user has use_admin_models=false, returns their
    own enabled models; otherwise returns admin-enabled models. Also returns image_generation_model."""
    try:
        image_generation_model = await settings_service.get_image_generation_model()
        user_id = current_user.user_id if current_user else None
        enabled_models = await resolver_get_enabled_models(user_id)
        catalog_slice = await get_enabled_models_catalog_slice(
            user_id, image_generation_model or ""
        )
        return {
            "enabled_models": enabled_models,
            "image_generation_model": image_generation_model or "",
            "orphaned_enabled_models": catalog_slice["orphaned_enabled_models"],
            "catalog_verified": catalog_slice["catalog_verified"],
            "effective_enabled_models": catalog_slice["effective_enabled_models"],
            "selectable_chat_models": catalog_slice["selectable_chat_models"],
            "orphaned_role_models": catalog_slice.get("orphaned_role_models") or {},
        }
    except Exception as e:
        logger.error("Failed to get enabled models: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/refresh")
async def refresh_available_models():
    """Force refresh the cached available models (admin provider registry)."""
    try:
        admin_provider_registry.refresh()
        if chat_service:
            await chat_service.refresh_available_models()
        models = await admin_provider_registry.get_all_admin_models()
        return {"message": f"Successfully refreshed {len(models)} models", "models": len(models)}

    except Exception as e:
        logger.error(f"❌ Failed to refresh models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/catalog-slice/invalidate")
async def invalidate_catalog_slice(
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
):
    """Recompute org catalog slice on next read (after role model settings change, etc.)."""
    try:
        admin_provider_registry.invalidate_slice()
        return {"status": "success"}
    except Exception as e:
        logger.error("Failed to invalidate catalog slice: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/current")
async def get_current_model():
    """Get currently selected model"""
    try:
        if chat_service:
            # Ensure model is selected before returning
            await chat_service.ensure_model_selected()
            current_model = chat_service.current_model
            logger.info(f"📊 Current model status: {current_model}")
            return {"current_model": current_model}
        else:
            return {"current_model": None}
        
    except Exception as e:
        logger.error(f"❌ Failed to get current model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/enabled")
async def set_enabled_models(
    request: Dict[str, List[str]], 
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Set list of enabled model IDs (admin only)"""
    try:
        logger.info(f"🔧 Admin {current_user.username} updating enabled models")
        model_ids = request.get("model_ids", [])
        success = await settings_service.set_enabled_models(model_ids)
        
        if success:
            admin_provider_registry.invalidate_slice()
            logger.info(f"✅ Admin {current_user.username} successfully updated enabled models: {model_ids}")
            return {"status": "success", "enabled_models": model_ids}
        else:
            raise HTTPException(status_code=500, detail="Failed to update enabled models")
        
    except Exception as e:
        logger.error(f"❌ Failed to set enabled models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/select")
async def select_model(
    request: Dict[str, str], 
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Select a model for the current user"""
    try:
        model_name = request.get("model_name")
        if not model_name:
            raise HTTPException(status_code=400, detail="model_name is required")

        selectable = await get_chat_selectable_model_ids_for_user(current_user.user_id)
        if model_name not in selectable:
            raise HTTPException(
                status_code=400,
                detail="That model is not available from your AI provider or is not enabled for chat.",
            )

        # Update the chat service's current model
        if chat_service:
            chat_service.current_model = model_name
            logger.info(f"✅ User {current_user.username} selected model: {model_name}")
            
            # Also save to settings as the user's preference
            from services.settings_service import settings_service
            try:
                await settings_service.set_llm_model(model_name)
                logger.info(f"💾 Saved model selection to settings: {model_name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to save model selection to settings: {e}")
            
            return {"status": "success", "current_model": model_name}
        else:
            raise HTTPException(status_code=503, detail="Chat service not available")
    except Exception as e:
        logger.error(f"❌ Failed to select model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== WEBSOCKET ENDPOINTS =====

@app.websocket("/api/ws/conversations")
async def websocket_conversations(websocket: WebSocket):
    """WebSocket endpoint for conversation updates"""
    logger.info("🔌 Conversation WebSocket connection attempt")
    try:
        # Get token from query parameters
        token = websocket.query_params.get("token")
        if not token:
            logger.error("❌ Conversation WebSocket missing token")
            await websocket.close(code=4001, reason="Missing token")
            return
        
        logger.info("🔐 Conversation WebSocket token received")
        
        # Validate token and get user
        try:
            from utils.auth_middleware import decode_jwt_token
            payload = decode_jwt_token(token)
            user_id = payload.get("user_id")
            if not user_id:
                logger.error("❌ Conversation WebSocket invalid token")
                await websocket.close(code=4003, reason="Invalid token")
                return
        except Exception as e:
            logger.error(f"❌ Conversation WebSocket token validation failed: {e}")
            await websocket.close(code=4003, reason="Invalid token")
            return
        
        logger.info(f"✅ Conversation WebSocket token validated for user: {user_id}")
        
        # Connect to WebSocket manager (use singleton to ensure same instance as team broadcasts)
        from utils.websocket_manager import get_websocket_manager
        ws_manager = get_websocket_manager()
        
        await ws_manager.connect(websocket, session_id=user_id)
        logger.info(f"✅ Conversation WebSocket connected to manager for user: {user_id}")
        logger.info(f"📊 Active sessions after connect: {list(ws_manager.session_connections.keys())}")
        logger.info(f"📊 Total connections: {len(ws_manager.active_connections)}")
        
        try:
            # Keep connection alive and handle messages
            while True:
                # Wait for any message (ping/pong, heartbeat, or data)
                try:
                    data = await websocket.receive_json()
                    
                    # Handle heartbeat to keep connection alive
                    if isinstance(data, dict) and data.get("type") == "heartbeat":
                        await websocket.send_json({
                            "type": "heartbeat_ack",
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        # Echo back for other message types
                        await websocket.send_json({
                            "type": "echo",
                            "data": data,
                            "timestamp": datetime.now().isoformat()
                        })
                except ValueError:
                    # Handle non-JSON messages (like plain text pings)
                    try:
                        data = await websocket.receive_text()
                        if data == "ping":
                            await websocket.send_text("pong")
                        else:
                            await websocket.send_json({
                                "type": "echo",
                                "data": data,
                                "timestamp": datetime.now().isoformat()
                            })
                    except Exception as text_error:
                        logger.warning(f"⚠️ Error handling WebSocket text message: {text_error}")
                        break
                
        except WebSocketDisconnect:
            logger.info(f"📡 Conversation WebSocket disconnected for user: {user_id}")
        except Exception as e:
            logger.error(f"❌ Conversation WebSocket error for user {user_id}: {e}", exc_info=True)
        finally:
            ws_manager.disconnect(websocket, session_id=user_id)
            logger.info(f"🧹 Conversation WebSocket cleaned up for user: {user_id}")
            
    except Exception as e:
        logger.error(f"❌ Conversation WebSocket connection failed: {e}")
        try:
            await websocket.close(code=4000, reason="Connection failed")
        except Exception:
            pass


@app.websocket("/api/ws/folders")
async def websocket_folders(websocket: WebSocket):
    """WebSocket endpoint for folder and document updates"""
    logger.info("🔌 Folder updates WebSocket connection attempt")
    try:
        # Get token from query parameters
        token = websocket.query_params.get("token")
        if not token:
            logger.error("❌ Folder WebSocket missing token")
            await websocket.close(code=4001, reason="Missing token")
            return
        
        logger.info("🔐 Folder WebSocket token received")
        
        # Validate token and get user
        try:
            from utils.auth_middleware import decode_jwt_token
            payload = decode_jwt_token(token)
            user_id = payload.get("user_id")
            if not user_id:
                logger.error("❌ Folder WebSocket invalid token")
                await websocket.close(code=4003, reason="Invalid token")
                return
        except Exception as e:
            logger.error(f"❌ Folder WebSocket token validation failed: {e}")
            await websocket.close(code=4003, reason="Invalid token")
            return
        
        logger.info(f"✅ Folder WebSocket token validated for user: {user_id}")
        
        # Connect to WebSocket manager (use singleton to ensure same instance)
        from utils.websocket_manager import get_websocket_manager
        ws_manager = get_websocket_manager()
        
        await ws_manager.connect(websocket, session_id=user_id)
        logger.info(f"✅ Folder WebSocket connected to manager for user: {user_id}")
        
        try:
            # Keep connection alive and handle messages
            while True:
                # Wait for any message (ping/pong or data)
                data = await websocket.receive_text()
                
                # Echo back for now (can be extended for real-time features)
                await websocket.send_json({
                    "type": "folder_echo",
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                })
                
        except WebSocketDisconnect:
            logger.info(f"📡 Folder WebSocket disconnected for user: {user_id}")
        except Exception as e:
            logger.error(f"❌ Folder WebSocket error for user {user_id}: {e}")
        finally:
            ws_manager.disconnect(websocket, session_id=user_id)
            logger.info(f"🧹 Folder WebSocket cleaned up for user: {user_id}")
            
    except Exception as e:
        logger.error(f"❌ Folder WebSocket connection failed: {e}")
        try:
            await websocket.close(code=4000, reason="Connection failed")
        except Exception:
            pass


@app.websocket("/api/ws/job-progress/{job_id}")
async def websocket_job_progress(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for job progress tracking"""
    logger.info(f"🔌 Job progress WebSocket connection attempt for job: {job_id}")
    try:
        # Get token from query parameters
        token = websocket.query_params.get("token")
        if not token:
            logger.error(f"❌ Job progress WebSocket missing token for job: {job_id}")
            await websocket.close(code=4001, reason="Missing token")
            return
        
        logger.info(f"🔐 Job progress WebSocket token received for job: {job_id}")
        
        # Validate token and get user
        try:
            from utils.auth_middleware import decode_jwt_token
            payload = decode_jwt_token(token)
            user_id = payload.get("user_id")
            if not user_id:
                logger.error(f"❌ Job progress WebSocket invalid token for job: {job_id}")
                await websocket.close(code=4003, reason="Invalid token")
                return
        except Exception as e:
            logger.error(f"❌ Job progress WebSocket token validation failed for job {job_id}: {e}")
            await websocket.close(code=4003, reason="Invalid token")
            return
        
        logger.info(f"✅ Job progress WebSocket token validated for job: {job_id}, user: {user_id}")
        
        # Connect to WebSocket manager for job tracking
        await websocket_manager.connect_to_job(websocket, job_id)
        logger.info(f"✅ Job progress WebSocket connected to manager for job: {job_id}")
        
        try:
            # Keep connection alive and handle messages
            while True:
                # Wait for any message (ping/pong or data)
                data = await websocket.receive_text()
                
                # Echo back for now (can be extended for real-time features)
                await websocket.send_json({
                    "type": "job_progress_echo",
                    "job_id": job_id,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                })
                
        except WebSocketDisconnect:
            logger.info(f"📡 Job progress WebSocket disconnected for job: {job_id}")
        except Exception as e:
            logger.error(f"❌ Job progress WebSocket error for job {job_id}: {e}")
        finally:
            websocket_manager.disconnect(websocket, session_id=None)
            logger.info(f"🧹 Job progress WebSocket cleaned up for job: {job_id}")
            
    except Exception as e:
        logger.error(f"❌ Job progress WebSocket connection failed for job {job_id}: {e}")
        try:
            await websocket.close(code=4000, reason="Connection failed")
        except Exception:
            pass


@app.websocket("/api/ws/agent-status/{conversation_id}")
async def websocket_agent_status(websocket: WebSocket, conversation_id: str):
    """
    WebSocket endpoint for real-time agent tool execution status
    
    This is the OUT-OF-BAND channel for LLM status updates that appear/disappear as the agent works.
    """
    logger.info(f"🤖 Agent Status WebSocket connection attempt for conversation: {conversation_id}")
    try:
        # Get token from query parameters
        token = websocket.query_params.get("token")
        if not token:
            logger.error(f"❌ Agent Status WebSocket missing token for conversation: {conversation_id}")
            await websocket.close(code=4001, reason="Missing token")
            return
        
        logger.info(f"🔐 Agent Status WebSocket token received for conversation: {conversation_id}")
        
        # Validate token and get user
        try:
            from utils.auth_middleware import decode_jwt_token
            payload = decode_jwt_token(token)
            user_id = payload.get("user_id")
            if not user_id:
                logger.error(f"❌ Agent Status WebSocket invalid token for conversation: {conversation_id}")
                await websocket.close(code=4003, reason="Invalid token")
                return
        except Exception as e:
            logger.error(f"❌ Agent Status WebSocket token validation failed for conversation {conversation_id}: {e}")
            await websocket.close(code=4003, reason="Invalid token")
            return
        
        logger.info(f"✅ Agent Status WebSocket token validated for conversation: {conversation_id}, user: {user_id}")
        
        # Connect to WebSocket manager for conversation-level agent status tracking
        await websocket_manager.connect_to_conversation(websocket, conversation_id, user_id)
        logger.info(f"✅ Agent Status WebSocket connected to manager for conversation: {conversation_id}")
        
        # Send confirmation message
        await websocket.send_json({
            "type": "agent_status_connected",
            "conversation_id": conversation_id,
            "message": "🤖 Connected to agent status channel - you'll see real-time updates as agents work!",
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            # Keep connection alive and handle messages
            while True:
                # Wait for any message (ping/pong or data)
                data = await websocket.receive_text()
                
                # Echo back for keepalive
                await websocket.send_json({
                    "type": "agent_status_echo",
                    "conversation_id": conversation_id,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                })
                
        except WebSocketDisconnect:
            logger.info(f"📡 Agent Status WebSocket disconnected for conversation: {conversation_id}")
        except Exception as e:
            logger.error(f"❌ Agent Status WebSocket error for conversation {conversation_id}: {e}")
        finally:
            websocket_manager.disconnect(websocket, session_id=None)
            logger.info(f"🧹 Agent Status WebSocket cleaned up for conversation: {conversation_id}")
            
    except Exception as e:
        logger.error(f"❌ Agent Status WebSocket connection failed for conversation {conversation_id}: {e}")
        try:
            await websocket.close(code=4000, reason="Connection failed")
        except Exception:
            pass


@app.websocket("/api/ws/line-timeline/{line_id}")
async def websocket_line_timeline(websocket: WebSocket, line_id: str):
    """WebSocket endpoint for line timeline live updates (inter-agent messages)."""
    logger.info(f"Line timeline WebSocket connection attempt for line: {line_id}")
    try:
        token = websocket.query_params.get("token")
        if not token:
            await websocket.close(code=4001, reason="Missing token")
            return
        try:
            from utils.auth_middleware import decode_jwt_token
            payload = decode_jwt_token(token)
            user_id = payload.get("user_id")
            if not user_id:
                await websocket.close(code=4003, reason="Invalid token")
                return
        except Exception as e:
            logger.error("Team timeline WebSocket token validation failed: %s", e)
            await websocket.close(code=4003, reason="Invalid token")
            return
        from services import agent_line_service
        team = await agent_line_service.get_line(line_id, user_id)
        if not team:
            await websocket.close(code=4043, reason="Line not found")
            return
        await websocket_manager.connect_to_line_timeline(websocket, line_id)
        await websocket.send_json({
            "type": "line_timeline_connected",
            "line_id": line_id,
            "timestamp": datetime.now().isoformat(),
        })
        try:
            while True:
                data = await websocket.receive_text()
                await websocket.send_json({
                    "type": "line_timeline_echo",
                    "line_id": line_id,
                    "data": data,
                    "timestamp": datetime.now().isoformat(),
                })
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error("Team timeline WebSocket error: %s", e)
        finally:
            websocket_manager.disconnect(websocket, session_id=None)
    except Exception as e:
        logger.error("Team timeline WebSocket connection failed: %s", e)
        try:
            await websocket.close(code=4000, reason="Connection failed")
        except Exception:
            pass


