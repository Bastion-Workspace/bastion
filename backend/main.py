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

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Depends, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
import uvicorn
import fitz  # PyMuPDF for PDF text layer extraction
import PyPDF2
import pdfplumber
from pydantic import BaseModel

from config import settings
from services.service_container import service_container
from version import __version__


def strip_yaml_frontmatter(content: str) -> str:
    """
    Strip YAML frontmatter from markdown content
    
    **BULLY!** Clean display without duplicate metadata!
    """
    import re
    
    # Pattern to match YAML frontmatter between --- markers
    frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
    
    # Remove the frontmatter and return the rest
    cleaned_content = re.sub(frontmatter_pattern, '', content, flags=re.DOTALL)
    
    # Clean up any leading/trailing whitespace
    return cleaned_content.strip()

# Initialize Celery app
from services.celery_app import celery_app

# Import Celery tasks to ensure they are registered
# Note: orchestrator_tasks.process_orchestrator_query removed - deprecated task
import services.celery_tasks.orchestrator_tasks
import services.celery_tasks.agent_tasks
import services.celery_tasks.rss_tasks


from services.settings_service import settings_service
from services.auth_service import auth_service

from services.user_document_service import UserDocumentService
from models.api_models import (
    URLImportRequest, QueryRequest, DocumentListResponse, 
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
from utils.logging_config import setup_logging
from utils.websocket_manager import WebSocketManager
from utils.auth_middleware import get_current_user, get_current_user_optional, require_admin

# Setup logging
from utils.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# Global service references (managed by service container)
document_service = None
migration_service = None
chat_service = None

knowledge_graph_service = None
collection_analysis_service = None
enhanced_pdf_segmentation_service = None
category_service = None

embedding_manager = None
websocket_manager = None

conversation_service = None
folder_service = None

# Import API routers
from api.settings_api import router as settings_router

# Import FileManager service
from services.file_manager import get_file_manager
from services.file_manager.models.file_placement_models import FolderStructureRequest

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
        
        # Get service references from container
        document_service = service_container.document_service
        chat_service = service_container.chat_service

        knowledge_graph_service = service_container.knowledge_graph_service
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

        # Initialize template service
        try:
            from services.template_service import template_service
            await template_service.initialize()
            logger.info("📋 Template Service initialized")
        except Exception as e:
            logger.error(f"❌ Template Service initialization failed: {e}")
            # Don't raise here as the app can still function without templates
        

        

        

        
        # Initialize folder service
        try:
            from services.folder_service import FolderService
            folder_service = FolderService()
            await folder_service.initialize()
            logger.info("📁 Folder Service initialized")
        except Exception as e:
            logger.error(f"❌ Folder Service initialization failed: {e}")
            folder_service = None
        
        # ROOSEVELT'S MESSAGING CAVALRY - Initialize messaging service
        try:
            from services.messaging.messaging_service import messaging_service
            await messaging_service.initialize(service_container.db_pool)
            logger.info("💬 BULLY! Messaging Service initialized")
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

        
        # Initialize available models from OpenRouter on startup
        logger.info("🔍 Fetching available models from OpenRouter on startup...")
        try:
            from services.settings_service import settings_service
            available_models = await chat_service.get_available_models()
            logger.info(f"✅ Fetched {len(available_models)} models from OpenRouter")
            
            # Check if any models are enabled
            enabled_models = await settings_service.get_enabled_models()
            if len(enabled_models) == 0:
                logger.warning("⚠️ No models are currently enabled. Admin must configure models in Settings before chat functionality will work.")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch models on startup: {e}")
        
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
        
        # Start gRPC Tool Service (Phase 2 - for LLM Orchestrator)
        grpc_task = None
        try:
            from services.grpc_tool_service import serve_tool_service
            grpc_port = int(os.getenv('GRPC_TOOL_SERVICE_PORT', '50052'))
            grpc_task = asyncio.create_task(serve_tool_service(grpc_port))
            logger.info(f"🔧 gRPC Tool Service started on port {grpc_port}")
        except Exception as e:
            logger.error(f"❌ Failed to start gRPC Tool Service: {e}")
            grpc_task = None
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Plato Knowledge Base...")
    
    # Stop gRPC Tool Service
    if grpc_task and not grpc_task.done():
        logger.info("🔧 Stopping gRPC Tool Service...")
        grpc_task.cancel()
        try:
            await grpc_task
        except asyncio.CancelledError:
            logger.info("✅ gRPC Tool Service stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping gRPC Tool Service: {e}")
    
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

# Mount static files for serving RSS article images
images_path = Path(f"{settings.UPLOAD_DIR}/web_sources/images")
images_path.mkdir(parents=True, exist_ok=True)
app.mount("/static/images", StaticFiles(directory=str(images_path)), name="images")



# Include unified chat API routes
from api.unified_chat_api import router as unified_chat_router
app.include_router(unified_chat_router)
logger.info("✅ Unified chat API routes registered")



# Include classification model API routes
from api.classification_model_api import router as classification_model_router
app.include_router(classification_model_router)
logger.info("✅ Classification model API routes registered")

# Include text completion model API routes
from api.text_completion_model_api import router as text_completion_model_router
app.include_router(text_completion_model_router)
logger.info("✅ Text completion model API routes registered")

# Include admin API routes
from api.admin_api import router as admin_router
app.include_router(admin_router)
logger.info("✅ Admin API routes registered")

# Include resilient embedding API routes
from api.resilient_embedding_api import router as resilient_embedding_router
app.include_router(resilient_embedding_router)
logger.info("✅ Resilient embedding API routes registered")

# Include settings API routes
app.include_router(settings_router)

# Template management removed - functionality not in use
logger.info("✅ Settings API routes registered")

# Services API removed - Twitter integration removed
logger.info("✅ Services API routes registered")
logger.info("✅ Template management API routes registered")
logger.info("✅ Template execution API routes registered")

# Include Export API routes
from api.export_api import router as export_router
app.include_router(export_router)
logger.info("✅ Export API routes registered")

# ROOSEVELT'S ORG SEARCH API
from api.org_search_api import router as org_search_router
app.include_router(org_search_router)
logger.info("✅ Org Search API routes registered")

# Org Quick Capture API
from api.org_capture_api import router as org_capture_router
app.include_router(org_capture_router)
logger.info("✅ Org Capture API routes registered")

# Include org settings API
from api.org_settings_api import router as org_settings_router
app.include_router(org_settings_router)
logger.info("✅ Org Settings API routes registered")

# Include org tag API
from api.org_tag_api import router as org_tag_router
app.include_router(org_tag_router)
logger.info("✅ Org Tag API routes registered")

# Include editor API
from api.editor_api import router as editor_router
app.include_router(editor_router)
logger.info("✅ Editor API routes registered")

# Research plan API routes removed - migrated to LangGraph subgraph workflows

# Include agent API routes

logger.info("✅ Agent API routes registered")

# Context-aware research API routes removed - migrated to LangGraph subgraph workflows

# Deprecated LangGraph APIs removed - using async_orchestrator_api for all LangGraph functionality
# Orchestrator chat API removed - deprecated endpoint that returned 410 errors

# Include Async Orchestrator API routes
from api.async_orchestrator_api import router as async_orchestrator_router
app.include_router(async_orchestrator_router)
logger.info("✅ Async Orchestrator API routes registered")

# gRPC Orchestrator Proxy (Phase 5 - Microservices)
from api.grpc_orchestrator_proxy import router as grpc_orchestrator_proxy_router
app.include_router(grpc_orchestrator_proxy_router)
logger.info("✅ gRPC Orchestrator Proxy routes registered (Phase 5)")

# Include Conversation API routes (moved from main)
from api.conversation_api import router as conversation_router
app.include_router(conversation_router)
logger.info("✅ Conversation API routes registered")

# Include Conversation Sharing API routes
from api.conversation_sharing_api import router as conversation_sharing_router
app.include_router(conversation_sharing_router)
logger.info("✅ Conversation Sharing API routes registered")

@app.post("/api/conversations", response_model=ConversationResponse)
async def create_conversation(request: CreateConversationRequest, current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Create a new conversation"""
    try:
        logger.info(f"💬 Creating new conversation: {request.title}")
        
        # Set the current user for this operation
        conversation_service.set_current_user(current_user.user_id)
        
        # Pass the user_id and initial message for title generation
        conversation_summary = await conversation_service.create_conversation(
            user_id=current_user.user_id,
            initial_message=request.initial_message,
            initial_mode="chat",
            metadata={"title": request.title} if request.title else None
        )
        
        # Convert dictionary response to ConversationDetail for the response
        from models.conversation_models import ConversationDetail
        conversation_detail = ConversationDetail(
            conversation_id=conversation_summary["conversation_id"],
            user_id=conversation_summary["user_id"],
            title=conversation_summary.get("title"),
            description=conversation_summary.get("description"),
            is_pinned=conversation_summary.get("is_pinned", False),
            is_archived=conversation_summary.get("is_archived", False),
            tags=conversation_summary.get("tags", []),
            metadata_json=conversation_summary.get("metadata_json", {}),
            message_count=conversation_summary.get("message_count", 0),
            last_message_at=conversation_summary.get("last_message_at"),
            manual_order=conversation_summary.get("manual_order"),
            order_locked=conversation_summary.get("order_locked", False),
            created_at=conversation_summary["created_at"],
            updated_at=conversation_summary["updated_at"],
            messages=[]  # New conversation has no messages yet
        )
        
        logger.info(f"✅ Conversation created: {conversation_summary['conversation_id']}")
        return ConversationResponse(conversation=conversation_detail)
        
    except Exception as e:
        logger.error(f"❌ Failed to create conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Include Agent Chaining API routes
from api.agent_chaining_api import router as agent_chaining_router
app.include_router(agent_chaining_router, prefix="/api/v1/agents")
logger.info("✅ Agent chaining API routes registered")

# Include RSS API routes
from api.rss_api import router as rss_router
app.include_router(rss_router)

from api.entertainment_sync_api import router as entertainment_sync_router
app.include_router(entertainment_sync_router)
logger.info("✅ RSS API routes registered")


# Include News API routes
try:
    from api.news_api import router as news_router
    app.include_router(news_router)
    logger.info("✅ News API routes registered")
except Exception as e:
    logger.warning(f"⚠️ Failed to register News API routes: {e}")

# Include FileManager API routes
from api.file_manager_api import router as file_manager_router
app.include_router(file_manager_router)
logger.info("✅ FileManager API routes registered")

# Include Authentication API routes
from api.auth_api import router as auth_router
app.include_router(auth_router)
logger.info("✅ Authentication API routes registered")

# ROOSEVELT'S MESSAGING CAVALRY API
from api.messaging_api import router as messaging_router
app.include_router(messaging_router)
logger.info("✅ BULLY! Messaging API routes registered")

# Teams API
from api.teams_api import router as teams_router
app.include_router(teams_router)
logger.info("✅ Teams API routes registered")

# Include Data Workspace API routes
from api.data_workspace_api import router as data_workspace_router
app.include_router(data_workspace_router)
logger.info("✅ Data Workspace API routes registered")

# Include Audio Transcription API routes
try:
    from api.audio_api import router as audio_router
    app.include_router(audio_router)
    logger.info("✅ Audio API routes registered")
except Exception as e:
    logger.warning(f"⚠️ Failed to register Audio API routes: {e}")

# Include Projects API routes
from api.projects_api import router as projects_router
app.include_router(projects_router)
logger.info("✅ Projects API routes registered")

# Include Status Bar API routes
from api.status_bar_api import router as status_bar_router
app.include_router(status_bar_router)
logger.info("✅ Status Bar API routes registered")

# Include Music API routes
from api.music_api import router as music_router
app.include_router(music_router)
logger.info("✅ Music API routes registered")

# Include HITL Orchestrator API routes
# HITL orchestrator API removed - using official orchestrator
logger.info("✅ Legacy HITL Orchestrator API removed")

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
        except:
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
async def get_available_models():
    """Get list of available OpenRouter models"""
    try:
        # Get available models from chat service
        models = await chat_service.get_available_models()
        return AvailableModelsResponse(models=models)
        
    except Exception as e:
        logger.error(f"❌ Failed to get available models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/enabled")
async def get_enabled_models():
    """Get list of enabled model IDs (excluding image generation model)"""
    try:
        enabled_models = await settings_service.get_enabled_models()
        
        # Filter out the image generation model if it's set
        # This prevents it from appearing in the chat model dropdown
        image_generation_model = await settings_service.get_image_generation_model()
        if image_generation_model and image_generation_model in enabled_models:
            enabled_models = [m for m in enabled_models if m != image_generation_model]
            logger.debug(f"🔍 Filtered out image generation model '{image_generation_model}' from enabled models list")
        
        return {"enabled_models": enabled_models}

    except Exception as e:
        logger.error(f"❌ Failed to get enabled models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/refresh")
async def refresh_available_models():
    """Force refresh the cached available models from OpenRouter"""
    try:
        models = await chat_service.refresh_available_models()
        return {"message": f"Successfully refreshed {len(models)} models", "models": len(models)}

    except Exception as e:
        logger.error(f"❌ Failed to refresh models: {str(e)}")
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
            
            # Refresh MCP service model reference
            
                logger.info(f"🔄 MCP service model reference refreshed")
            
            return {"status": "success", "current_model": model_name}
        else:
            raise HTTPException(status_code=503, detail="Chat service not available")
        
    except Exception as e:
        logger.error(f"❌ Failed to select model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


"""
Conversation endpoints moved to api/conversation_api.py
"""


"""
Conversation list endpoint moved to api/conversation_api.py
"""


"""
Conversation get endpoint moved to api/conversation_api.py
"""


"""
Conversation update endpoint moved to api/conversation_api.py
"""


@app.get("/api/conversations/{conversation_id}/messages", response_model=MessageListResponse)
async def get_conversation_messages(conversation_id: str, skip: int = 0, limit: int = 100, current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """
    Get messages for a conversation - MIGRATION-COMPATIBLE APPROACH
    
    Priority order (consistent with orchestrator migration):
    1. Conversation database (primary source - populated by backend proxy)
    2. LangGraph checkpoints (fallback for legacy conversations)
    
    This ensures new orchestrator conversations work correctly while maintaining
    backward compatibility with old conversations that only have checkpoints.
    """
    try:
        # Check read permission
        from utils.auth_middleware import validate_conversation_access
        has_access = await validate_conversation_access(
            user_id=current_user.user_id,
            conversation_id=conversation_id,
            required_permission="read"
        )
        if not has_access:
            raise HTTPException(status_code=403, detail="You do not have access to this conversation")
        
        logger.info(f"💬 Getting messages for conversation: {conversation_id}")
        
        messages = []
        messages_from_checkpoint = False  # Track if we got messages from checkpoint
        messages_from_database = False  # Track if we got messages from database
        
        # PRIORITY 1: Read from conversation database (primary source for orchestrator conversations)
        try:
            from services.conversation_service import ConversationService
            conversation_service = ConversationService()
            conversation_service.set_current_user(current_user.user_id)
            
            db_messages_result = await conversation_service.get_conversation_messages(
                conversation_id=conversation_id,
                user_id=current_user.user_id,
                skip=skip,
                limit=limit
            )
            
            # ConversationService returns dict with "messages" key
            logger.info(f"🔍 get_conversation_messages result keys: {list(db_messages_result.keys()) if db_messages_result else 'None'}")
            if db_messages_result and "messages" in db_messages_result:
                db_messages = db_messages_result.get("messages", [])
                logger.info(f"🔍 get_conversation_messages returned {len(db_messages)} messages")
                if db_messages:
                    logger.info(f"✅ Retrieved {len(db_messages)} messages from conversation database (primary source)")
                    
                    # Convert database messages to API format
                    for msg in db_messages:
                        metadata_json = msg.get("metadata_json", {})
                        
                        # Extract editor_operations and manuscript_edit from metadata for top-level access
                        editor_operations = metadata_json.get("editor_operations", []) if isinstance(metadata_json, dict) else []
                        manuscript_edit = metadata_json.get("manuscript_edit") if isinstance(metadata_json, dict) else None
                        
                        message_obj = {
                            "message_id": msg.get("message_id"),
                            "conversation_id": conversation_id,
                            "message_type": msg.get("message_type", "user"),
                            "role": msg.get("message_type", "user"),
                            "content": msg.get("content", ""),
                            "sequence_number": msg.get("sequence_number", 0),
                            "created_at": msg.get("created_at").isoformat() if hasattr(msg.get("created_at"), "isoformat") else str(msg.get("created_at")),
                            "updated_at": msg.get("updated_at").isoformat() if hasattr(msg.get("updated_at"), "isoformat") else str(msg.get("updated_at")),
                            "metadata_json": metadata_json,
                            "citations": metadata_json.get("citations", []) if isinstance(metadata_json, dict) else [],
                            "edit_history": []
                        }
                        
                        # Add editor_operations and manuscript_edit as top-level fields if present
                        if editor_operations:
                            message_obj["editor_operations"] = editor_operations
                        if manuscript_edit:
                            message_obj["manuscript_edit"] = manuscript_edit
                        
                        messages.append(message_obj)
                    messages_from_database = True
                else:
                    logger.info(f"📚 No messages in conversation database, will try checkpoints as fallback")
            else:
                logger.debug(f"📚 Conversation database query returned no messages")
        except Exception as db_error:
            logger.warning(f"⚠️ Failed to load messages from conversation database: {db_error}")
        
        # PRIORITY 2: Fallback to LangGraph checkpoints (for legacy conversations)
        if not messages_from_database:
            logger.info(f"📚 Falling back to LangGraph checkpoints for conversation {conversation_id}")
            try:
                from services.langgraph_postgres_checkpointer import get_postgres_checkpointer
                checkpointer = await get_postgres_checkpointer()
                
                if not checkpointer.is_initialized:
                    logger.warning("⚠️ LangGraph checkpointer not initialized, skipping checkpoint fallback")
                else:
                    # ROOSEVELT'S THREAD_ID FIX: Use normalized thread_id for proper conversation lookup
                    from services.orchestrator_utils import normalize_thread_id
                    normalized_thread_id = normalize_thread_id(current_user.user_id, conversation_id)
                    
                    # Get conversation state from LangGraph checkpoint
                    config = {"configurable": {"thread_id": normalized_thread_id}}
                    
                    # Try to get current state with all messages using the actual checkpointer instance
                    if checkpointer.checkpointer and not checkpointer.using_fallback:
                        actual_checkpointer = checkpointer.checkpointer
                        if hasattr(actual_checkpointer, 'aget_tuple'):
                            checkpoint_tuple = await actual_checkpointer.aget_tuple(config)
                        else:
                            logger.warning("⚠️ aget_tuple method not available on checkpointer")
                            checkpoint_tuple = None
                            
                        if checkpoint_tuple and checkpoint_tuple.checkpoint:
                            # Extract messages from checkpoint state
                            checkpoint_state = checkpoint_tuple.checkpoint
                            # ROOSEVELT'S POSTGRESQL JSON FIX: Handle both dict and JSON string formats
                            if isinstance(checkpoint_state, str):
                                import json
                                try:
                                    checkpoint_state = json.loads(checkpoint_state)
                                except json.JSONDecodeError as e:
                                    logger.error(f"❌ Failed to parse checkpoint state JSON: {e}")
                                    checkpoint_state = {}
                            elif checkpoint_state is None:
                                checkpoint_state = {}
                            
                            state_data = checkpoint_state.get("channel_values", {})
                            logger.info(f"🔍 Checkpoint state keys: {list(state_data.keys())}")
                            logger.info(f"🔍 Checkpoint state data: {state_data}")
                            
                            if "messages" in state_data:
                                langgraph_messages = state_data["messages"]
                                logger.info(f"✅ Found {len(langgraph_messages)} messages in LangGraph checkpoint")
                                
                                # ROOSEVELT'S DEBUG LOGGING: Log message types for debugging
                                for i, msg in enumerate(langgraph_messages):
                                    msg_type = "unknown"
                                    if hasattr(msg, '__class__'):
                                        msg_type = str(msg.__class__)
                                    elif hasattr(msg, 'type'):
                                        msg_type = msg.type
                                    logger.debug(f"🔍 Message {i}: type={msg_type}, content_length={len(msg.content) if hasattr(msg, 'content') else 0}")
                                
                                # Convert LangGraph messages to API format
                                for i, msg in enumerate(langgraph_messages):
                                    if hasattr(msg, 'content'):
                                        # ROOSEVELT'S MESSAGE TYPE FIX: Proper LangGraph message type detection
                                        message_type = "user"
                                        role = "user"
                                        
                                        # Check for HumanMessage (user messages)
                                        if hasattr(msg, '__class__') and 'HumanMessage' in str(msg.__class__):
                                            message_type = "user"
                                            role = "user"
                                        # Check for AIMessage (assistant messages)  
                                        elif hasattr(msg, '__class__') and 'AIMessage' in str(msg.__class__):
                                            message_type = "assistant"
                                            role = "assistant"
                                        # Fallback to type attribute if available
                                        elif hasattr(msg, 'type'):
                                            if msg.type == "human":
                                                message_type = "user"
                                                role = "user"
                                            elif msg.type == "ai":
                                                message_type = "assistant"
                                                role = "assistant"
                                        
                                        # ROOSEVELT'S CITATION FIX: Extract citations from THIS MESSAGE's additional_kwargs
                                        citations = []
                                        if hasattr(msg, 'additional_kwargs') and isinstance(msg.additional_kwargs, dict):
                                            citations = msg.additional_kwargs.get("citations", [])
                                            if citations:
                                                logger.info(f"🔗 EXTRACTED {len(citations)} CITATIONS from message {i} additional_kwargs")
                                        
                                        # Build metadata from additional_kwargs
                                        metadata_json = {}
                                        if hasattr(msg, 'additional_kwargs') and isinstance(msg.additional_kwargs, dict):
                                            metadata_json = {
                                                "citations": citations,
                                                "research_mode": msg.additional_kwargs.get("research_mode"),
                                                "timestamp": msg.additional_kwargs.get("timestamp")
                                            }
                                        
                                        messages.append({
                                            "message_id": f"lg_{conversation_id}_{i}",
                                            "conversation_id": conversation_id,
                                            "message_type": message_type,
                                            "role": role,
                                            "content": msg.content,
                                            "sequence_number": i,
                                            "created_at": datetime.now().isoformat(),
                                            "updated_at": datetime.now().isoformat(),
                                            "metadata_json": metadata_json if metadata_json.get("citations") else {},
                                            "citations": citations,
                                            "edit_history": []
                                        })
                                messages_from_checkpoint = True  # Mark that we got messages from checkpoint
                            else:
                                logger.info(f"⚠️ No messages found in checkpoint state for {conversation_id}")
                        else:
                            logger.info(f"⚠️ No checkpoint found for conversation {conversation_id}")
                    else:
                        logger.warning("⚠️ aget_tuple method not available on checkpointer")
            except Exception as checkpoint_error:
                # This is normal for new conversations that don't have checkpoints yet
                if "aget_tuple" in str(checkpoint_error) or "get_next_version" in str(checkpoint_error):
                    logger.debug(f"🔧 LangGraph checkpointer API issue (expected for new conversations): {checkpoint_error}")
                else:
                    logger.warning(f"⚠️ Failed to read from LangGraph checkpoint: {checkpoint_error}")
        
        total_count = len(messages)
        has_more = False  # Since we're getting all messages from checkpoint or database
        
        # Determine source for logging
        if messages_from_checkpoint and total_count > 0:
            source = "checkpoint"
        elif total_count > 0:
            source = "database"
        else:
            source = "none"
        
        logger.info(f"✅ Retrieved {total_count} messages for conversation {conversation_id} (source: {source})")
        return MessageListResponse(
            messages=messages, 
            total_count=total_count,
            has_more=has_more
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get conversation messages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/conversations/{conversation_id}/messages", response_model=MessageResponse)
async def add_message_to_conversation(conversation_id: str, request: CreateMessageRequest, current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Add a message to a conversation"""
    try:
        # Check comment permission
        from utils.auth_middleware import validate_conversation_access
        has_access = await validate_conversation_access(
            user_id=current_user.user_id,
            conversation_id=conversation_id,
            required_permission="comment"
        )
        if not has_access:
            raise HTTPException(status_code=403, detail="You do not have permission to add messages to this conversation")
        
        logger.info(f"💬 Adding message to conversation: {conversation_id}")
        
        # Set the current user for this operation
        conversation_service.set_current_user(current_user.user_id)
        
        message = await conversation_service.add_message(
            conversation_id=conversation_id,
            user_id=current_user.user_id,
            role=request.message_type.value,  # Convert enum to string
            content=request.content,
            metadata=request.metadata
        )
        
        logger.info(f"✅ Message added to conversation {conversation_id}")
        
        # Broadcast message to all conversation participants
        try:
            from services.conversation_sharing_service import get_conversation_sharing_service
            from utils.websocket_manager import get_websocket_manager
            
            sharing_service = await get_conversation_sharing_service()
            participants = await sharing_service.get_conversation_participants(
                conversation_id=conversation_id,
                user_id=current_user.user_id
            )
            
            websocket_manager = get_websocket_manager()
            if websocket_manager and participants:
                for participant in participants:
                    if participant["user_id"] != current_user.user_id:  # Don't notify sender
                        try:
                            await websocket_manager.send_to_session(
                                message={
                                    "type": "participant_message",
                                    "data": {
                                        "conversation_id": conversation_id,
                                        "sender_id": current_user.user_id,
                                        "message_id": message.get("message_id"),
                                        "content": message.get("content", "")[:100]  # Preview
                                    }
                                },
                                session_id=participant["user_id"]
                            )
                        except Exception as ws_error:
                            logger.debug(f"Failed to notify participant {participant['user_id']}: {ws_error}")
        except Exception as collab_error:
            logger.debug(f"Collaboration notification failed (non-fatal): {collab_error}")
        
        return MessageResponse(message=message)
        
    except Exception as e:
        logger.error(f"❌ Failed to add message to conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Delete a conversation and all its messages (LangGraph + legacy)"""
    try:
        logger.info(f"💬 Deleting conversation: {conversation_id} for user: {current_user.user_id}")
        
        # ROOSEVELT'S DUAL DELETION: Delete from both LangGraph checkpoints AND legacy tables
        
        # Step 1: Delete from LangGraph checkpoints
        import asyncpg
        from config import settings
        
        connection_string = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        conn = await asyncpg.connect(connection_string)
        
        try:
            # Delete all checkpoints for this conversation/thread
            deleted_checkpoints = await conn.execute("""
                DELETE FROM checkpoints 
                WHERE thread_id = $1 
                AND checkpoint -> 'channel_values' ->> 'user_id' = $2
            """, conversation_id, current_user.user_id)
            
            # Also delete from checkpoint_blobs and checkpoint_writes
            await conn.execute("""
                DELETE FROM checkpoint_blobs 
                WHERE thread_id = $1
            """, conversation_id)
            
            await conn.execute("""
                DELETE FROM checkpoint_writes 
                WHERE thread_id = $1
            """, conversation_id)
            
            logger.info(f"🗑️ Deleted LangGraph checkpoints: {deleted_checkpoints}")
            
        finally:
            await conn.close()
        
        # Step 2: Delete from legacy conversation tables (for completeness)
        conversation_service.set_current_user(current_user.user_id)
        legacy_success = await conversation_service.delete_conversation(conversation_id)
        
        logger.info(f"🔍 Legacy delete result: {legacy_success}")
        logger.info(f"✅ Conversation deleted from both LangGraph and legacy systems: {conversation_id}")
        return {"status": "success", "message": f"Conversation {conversation_id} deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/conversations")
async def delete_all_conversations(current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Delete ALL conversations for the current user (LangGraph + legacy)"""
    try:
        logger.info(f"💬 Deleting ALL conversations for user: {current_user.user_id}")
        
        # ROOSEVELT'S MASS DELETION: Delete from both LangGraph checkpoints AND legacy tables
        
        # Step 1: Delete from LangGraph checkpoints
        import asyncpg
        from config import settings
        
        connection_string = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        conn = await asyncpg.connect(connection_string)
        
        try:
            # Delete all checkpoints for this user
            deleted_checkpoints = await conn.execute("""
                DELETE FROM checkpoints 
                WHERE checkpoint -> 'channel_values' ->> 'user_id' = $1
            """, current_user.user_id)
            
            # Also delete from checkpoint_blobs and checkpoint_writes for this user
            # First get all thread_ids for this user
            user_threads = await conn.fetch("""
                SELECT DISTINCT thread_id FROM checkpoints 
                WHERE checkpoint -> 'channel_values' ->> 'user_id' = $1
            """, current_user.user_id)
            
            thread_ids = [row['thread_id'] for row in user_threads]
            
            if thread_ids:
                # Delete from checkpoint_blobs and checkpoint_writes for user's threads
                await conn.execute("""
                    DELETE FROM checkpoint_blobs 
                    WHERE thread_id = ANY($1)
                """, thread_ids)
                
                await conn.execute("""
                    DELETE FROM checkpoint_writes 
                    WHERE thread_id = ANY($1)
                """, thread_ids)
            
            logger.info(f"🗑️ Deleted LangGraph checkpoints for user: {deleted_checkpoints}")
            
        finally:
            await conn.close()
        
        # Step 2: Delete from legacy conversation tables
        conversation_service.set_current_user(current_user.user_id)
        
        # Get all conversations for the user first
        conversations = await conversation_service.list_conversations(0, 10000)  # Large limit to get all
        conversation_ids = [conv['conversation_id'] for conv in conversations.get('conversations', [])]
        
        # Delete each conversation
        deleted_count = 0
        for conv_id in conversation_ids:
            try:
                await conversation_service.delete_conversation(conv_id)
                deleted_count += 1
            except Exception as e:
                logger.error(f"❌ Failed to delete conversation {conv_id}: {e}")
        
        logger.info(f"✅ Deleted {deleted_count} conversations from legacy system")
        logger.info(f"✅ ALL conversations deleted for user: {current_user.user_id}")
        
        return {
            "status": "success", 
            "message": f"Deleted {deleted_count} conversations",
            "deleted_count": deleted_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete all conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/debug/conversations/{conversation_id}")
async def debug_conversation(conversation_id: str, current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Debug endpoint to check conversation ownership"""
    try:
        logger.info(f"🔍 Debugging conversation: {conversation_id} for user: {current_user.user_id}")
        
        pool = await conversation_service.lifecycle_manager._get_db_pool()
        async with pool.acquire() as conn:
            # Check if conversation exists
            conversation = await conn.fetchrow("""
                SELECT conversation_id, user_id, title FROM conversations 
                WHERE conversation_id = $1
            """, conversation_id)
            
            if not conversation:
                return {
                    "conversation_exists": False,
                    "current_user_id": current_user.user_id,
                    "message": "Conversation not found"
                }
            
            return {
                "conversation_exists": True,
                "conversation_id": conversation["conversation_id"],
                "conversation_user_id": conversation["user_id"],
                "current_user_id": current_user.user_id,
                "conversation_title": conversation["title"],
                "user_matches": conversation["user_id"] == current_user.user_id,
                "message": "Conversation found"
            }
            
    except Exception as e:
        logger.error(f"❌ Debug conversation failed: {str(e)}")
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
        except:
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
        except:
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
        except:
            pass


@app.websocket("/api/ws/agent-status/{conversation_id}")
async def websocket_agent_status(websocket: WebSocket, conversation_id: str):
    """
    ROOSEVELT'S AGENT STATUS CHANNEL: WebSocket endpoint for real-time agent tool execution status
    
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
        except:
            pass


# ===== MIGRATION ENDPOINTS =====

@app.get("/api/migration/status")
async def get_migration_status():
    """Get migration status and statistics"""
    try:
        status = await migration_service.get_migration_status()
        return status
        
    except Exception as e:
        logger.error(f"❌ Failed to get migration status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/migration/run")
async def run_migration():
    """Manually run migration from JSON to PostgreSQL"""
    try:
        logger.info("🔄 Starting manual migration...")
        
        result = await migration_service.migrate_documents()
        
        logger.info(f"✅ Manual migration completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Manual migration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== DOCUMENT MANAGEMENT ENDPOINTS =====

@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form(None),
    folder_id: str = Form(None),
    category: str = Form(None),
    tags: str = Form(None),  # Comma-separated tags
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Upload and process a document to global collection (admin only) - **BULLY!** Now with category and tags!"""
    try:
        logger.info(f"📄 Admin {current_user.username} uploading document to global collection: {file.filename} to folder: {folder_id}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Validate folder access if folder_id is provided
        if folder_id:
            # Check if folder exists and is a global folder
            folder = await folder_service.get_folder(folder_id, None)  # Global folders have no user_id
            if not folder or folder.collection_type != "global":
                raise HTTPException(status_code=404, detail="Global folder not found or access denied")
        
        # Process document (no user_id = global collection)
        # **ROOSEVELT FIX**: Pass folder_id to ensure files land in correct subfolder on disk
        result = await document_service.upload_and_process(file, doc_type, folder_id=folder_id)
        
        # **ROOSEVELT METADATA UPDATE**: Update category and tags if provided
        if result.document_id and (category or tags):
            from models.api_models import DocumentUpdateRequest, DocumentCategory
            
            # Parse tags from comma-separated string
            tags_list = [tag.strip() for tag in tags.split(',')] if tags else []
            
            # Parse category enum
            doc_category = None
            if category:
                try:
                    doc_category = DocumentCategory(category)
                except ValueError:
                    logger.warning(f"⚠️ Invalid category '{category}', ignoring")
            
            # Update document metadata
            update_request = DocumentUpdateRequest(
                category=doc_category,
                tags=tags_list if tags_list else None
            )
            await document_service.update_document_metadata(result.document_id, update_request)
            logger.info(f"📋 Updated document metadata: category={category}, tags={tags_list}")
        
        # Assign document to folder if specified (immediate assignment with retry)
        if folder_id and result.document_id:
            # Try immediate folder assignment first
            try:
                success = await document_service.document_repository.update_document_folder(result.document_id, folder_id, None)  # None for admin/global
                if success:
                    logger.info(f"✅ Global document {result.document_id} assigned to folder {folder_id} immediately")
                    
                    # Small delay to ensure transaction is committed before frontend queries
                    await asyncio.sleep(0.1)
                    
                    # Send WebSocket notification for optimistic UI update
                    try:
                        from utils.websocket_manager import get_websocket_manager
                        websocket_manager = get_websocket_manager()
                        if websocket_manager:
                            await websocket_manager.send_to_session({
                                "type": "folder_event",
                                "action": "file_created",
                                "folder_id": folder_id,
                                "document_id": result.document_id,
                                "filename": file.filename,
                                "user_id": current_user.user_id,
                                "timestamp": datetime.now().isoformat()
                            }, current_user.user_id)
                            logger.info(f"📡 Sent file creation notification for user {current_user.user_id}")
                        else:
                            logger.warning("⚠️ WebSocket manager not available for file creation notification")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to send WebSocket notification: {e}")
                else:
                    # If immediate assignment fails, start background retry task
                    logger.warning(f"⚠️ Immediate folder assignment failed, starting background retry for document {result.document_id}")
                    
                    async def assign_document_to_folder():
                        try:
                            # Wait a bit longer for the document to be fully committed
                            await asyncio.sleep(2.0)
                            
                            # Try folder assignment with retry logic
                            max_retries = 10
                            for attempt in range(max_retries):
                                try:
                                    success = await document_service.document_repository.update_document_folder(result.document_id, folder_id, None)  # None for admin/global
                                    if success:
                                        logger.info(f"✅ Global document {result.document_id} assigned to folder {folder_id} (background attempt {attempt + 1})")
                                        
                                        # Send WebSocket notification for optimistic UI update
                                        try:
                                            from utils.websocket_manager import get_websocket_manager
                                            websocket_manager = get_websocket_manager()
                                            if websocket_manager:
                                                await websocket_manager.send_to_session({
                                                    "type": "folder_event",
                                                    "action": "file_created",
                                                    "folder_id": folder_id,
                                                    "document_id": result.document_id,
                                                    "filename": file.filename,
                                                    "user_id": current_user.user_id,
                                                    "timestamp": datetime.now().isoformat()
                                                }, current_user.user_id)
                                                logger.info(f"📡 Sent file creation notification for user {current_user.user_id}")
                                            else:
                                                logger.warning("⚠️ WebSocket manager not available for file creation notification")
                                        except Exception as e:
                                            logger.warning(f"⚠️ Failed to send WebSocket notification: {e}")
                                        return
                                    else:
                                        logger.warning(f"⚠️ Failed to assign global document {result.document_id} to folder {folder_id} (background attempt {attempt + 1})")
                                        if attempt < max_retries - 1:
                                            await asyncio.sleep(2.0 * (attempt + 1))  # Longer exponential backoff
                                except Exception as e:
                                    logger.error(f"❌ Error assigning global document to folder (background attempt {attempt + 1}): {e}")
                                    if attempt < max_retries - 1:
                                        await asyncio.sleep(2.0 * (attempt + 1))
                            else:
                                logger.error(f"❌ Failed to assign global document {result.document_id} to folder {folder_id} after {max_retries} background attempts")
                        except Exception as e:
                            logger.error(f"❌ Background folder assignment failed: {e}")
                    
                    # Start the background task without waiting for it
                    asyncio.create_task(assign_document_to_folder())
                    
            except Exception as e:
                logger.error(f"❌ Immediate folder assignment failed: {e}")
                # Start background retry task
                asyncio.create_task(assign_document_to_folder())
        
        logger.info(f"✅ Global document uploaded successfully: {result.document_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Global upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/upload-multiple")
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Upload and process multiple documents to global collection (admin only)"""
    try:
        logger.info(f"📄 Uploading {len(files)} documents")
        
        # Validate files
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        for file in files:
            if not file.filename:
                raise HTTPException(status_code=400, detail=f"No filename provided for one of the files")
        
        # Use parallel document service for bulk upload
        if hasattr(document_service, 'upload_multiple_documents'):
            result = await document_service.upload_multiple_documents(files, enable_parallel=True)
        else:
            # Fallback to sequential processing
            logger.warning("⚠️ Parallel upload not available, falling back to sequential processing")
            upload_results = []
            successful_uploads = 0
            failed_uploads = 0
            
            for file in files:
                try:
                    single_result = await document_service.upload_and_process(file)
                    upload_results.append(single_result)
                    if single_result.status != ProcessingStatus.FAILED:
                        successful_uploads += 1
                    else:
                        failed_uploads += 1
                except Exception as e:
                    failed_uploads += 1
                    upload_results.append(DocumentUploadResponse(
                        document_id="",
                        filename=file.filename,
                        status=ProcessingStatus.FAILED,
                        message=f"Upload failed: {str(e)}"
                    ))
            
            from models.api_models import BulkUploadResponse
            result = BulkUploadResponse(
                total_files=len(files),
                successful_uploads=successful_uploads,
                failed_uploads=failed_uploads,
                upload_results=upload_results,
                processing_time=0,
                message=f"Sequential upload completed: {successful_uploads}/{len(files)} files successful"
            )
        
        logger.info(f"✅ Bulk upload completed: {result.successful_uploads}/{result.total_files} successful")
        return result
        
    except Exception as e:
        logger.error(f"❌ Bulk upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/import-url", response_model=DocumentUploadResponse)
async def import_from_url(
    request: URLImportRequest,
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Import content from URL to global collection (admin only)"""
    try:
        logger.info(f"🔗 Admin {current_user.username} importing from URL: {request.url}")
        
        result = await document_service.import_from_url(request.url, request.content_type)
        
        logger.info(f"✅ URL imported successfully: {result.document_id}")
        return result
        
    except Exception as e:
        logger.error(f"❌ URL import failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== USER DOCUMENT MANAGEMENT ENDPOINTS =====

@app.post("/api/user/documents/upload", response_model=DocumentUploadResponse)
async def upload_user_document(
    file: UploadFile = File(...),
    doc_type: str = Form(None),
    folder_id: str = Form(None),
    category: str = Form(None),
    tags: str = Form(None),  # Comma-separated tags
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Upload a document to user's private collection - Roosevelt Architecture - **BULLY!** Now with category and tags!"""
    try:
        logger.info(f"📄 User {current_user.username} uploading document: {file.filename} to folder: {folder_id}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # **BULLY!** INBOX.ORG SAFEGUARD - Prevent duplicate inbox files
        if file.filename.lower() == "inbox.org":
            from pathlib import Path
            from services.database_manager.database_helpers import fetch_one, fetch_all
            
            # First check database for existing inbox.org (most reliable)
            existing_docs = await fetch_all(
                """
                SELECT document_id, filename, folder_id
                FROM document_metadata
                WHERE user_id = $1 AND LOWER(filename) = 'inbox.org'
                """,
                current_user.user_id
            )
            
            if existing_docs:
                # Found existing inbox in database - reject upload
                error_msg = (
                    f"⚠️ INBOX.ORG ALREADY EXISTS in your documents. "
                    f"Only one inbox.org per user is allowed. "
                    f"Use quick capture (Ctrl+Shift+C) to add content, or delete the existing inbox before uploading a new one."
                )
                if len(existing_docs) > 1:
                    error_msg += f" (Note: {len(existing_docs)} inbox files found in database - consider cleaning up duplicates)"
                
                logger.warning(f"🚫 BLOCKED: User {current_user.username} tried to upload inbox.org when one already exists (document_id: {existing_docs[0]['document_id']})")
                raise HTTPException(status_code=409, detail=error_msg)
            
            # Also check filesystem as backup (in case DB is out of sync)
            row = await fetch_one("SELECT username FROM users WHERE user_id = $1", current_user.user_id)
            username = row['username'] if row else current_user.user_id
            
            upload_dir = Path(settings.UPLOAD_DIR)
            user_base_dir = upload_dir / "Users" / username
            
            if user_base_dir.exists():
                existing_inboxes = list(user_base_dir.rglob("inbox.org"))
                if existing_inboxes:
                    # Found existing inbox on disk but not in DB - warn and allow upload (will recover file)
                    existing_paths = [str(f.relative_to(user_base_dir)) for f in existing_inboxes]
                    logger.warning(f"⚠️  Found orphaned inbox.org on disk at {existing_paths[0]} - allowing upload to recover file")
                    # Continue with upload - this will overwrite the orphaned file
        
        # Validate folder access if folder_id is provided
        if folder_id:
            # Check if folder exists and user has access
            folder = await folder_service.get_folder(folder_id, current_user.user_id)
            if not folder:
                raise HTTPException(status_code=404, detail="Folder not found or access denied")
            
            # For admin users, allow uploading to global folders
            if folder.collection_type == "global" and current_user.role != "admin":
                raise HTTPException(status_code=403, detail="Only admins can upload to global folders")
        
        # ROOSEVELT FIX: Pass folder_id to upload_and_process for transaction-level folder assignment
        # This ensures folder assignment happens within the same transaction as document creation
        
        # Process document with user_id and folder_id for immediate folder assignment
        result = await document_service.upload_and_process(file, doc_type, user_id=current_user.user_id, folder_id=folder_id)
        
        # **ROOSEVELT METADATA UPDATE**: Update category and tags if provided
        if result.document_id and (category or tags):
            from models.api_models import DocumentUpdateRequest, DocumentCategory
            
            # Parse tags from comma-separated string
            tags_list = [tag.strip() for tag in tags.split(',')] if tags else []
            
            # Parse category enum
            doc_category = None
            if category:
                try:
                    doc_category = DocumentCategory(category)
                except ValueError:
                    logger.warning(f"⚠️ Invalid category '{category}', ignoring")
            
            # Update document metadata
            update_request = DocumentUpdateRequest(
                category=doc_category,
                tags=tags_list if tags_list else None
            )
            await document_service.update_document_metadata(result.document_id, update_request)
            logger.info(f"📋 Updated user document metadata: category={category}, tags={tags_list}")
        
        # ROOSEVELT FIX: Folder assignment is now handled within the upload_and_process transaction
        # Send WebSocket notification for optimistic UI update if folder assignment was successful
        if folder_id and result.document_id:
            try:
                from utils.websocket_manager import get_websocket_manager
                websocket_manager = get_websocket_manager()
                if websocket_manager:
                    await websocket_manager.send_to_session({
                        "type": "folder_event",
                        "action": "file_created",
                        "folder_id": folder_id,
                        "document_id": result.document_id,
                        "filename": file.filename,
                        "user_id": current_user.user_id,
                        "timestamp": datetime.now().isoformat()
                    }, current_user.user_id)
                    logger.info(f"📡 Sent file creation notification for user {current_user.user_id}")
                else:
                    logger.warning("⚠️ WebSocket manager not available for file creation notification")
            except Exception as e:
                logger.warning(f"⚠️ Failed to send WebSocket notification: {e}")
        
        logger.info(f"✅ User document uploaded successfully: {result.document_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ User upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/user/documents/search")
async def search_user_and_global_documents(
    request: QueryRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Search both user's private documents and global shared documents"""
    try:
        logger.info(f"🔍 User {current_user.username} searching: {request.query[:50]}...")
        
        # Use embedding manager's hybrid search (searches both collections)
        results = await document_service.embedding_manager.search_similar(
            query_text=request.query,
            limit=getattr(request, 'limit', 10),
            score_threshold=getattr(request, 'similarity_threshold', 0.7),
            user_id=current_user.user_id  # This triggers hybrid search
        )
        
        logger.info(f"✅ Found {len(results)} results for user {current_user.username}")
        return {"results": results, "total_results": len(results)}
        
    except Exception as e:
        logger.error(f"❌ User search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user/documents", response_model=DocumentListResponse)
async def list_user_documents(
    skip: int = 0, 
    limit: int = 100,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """List user's private documents"""
    try:
        logger.info(f"📄 Listing documents for user {current_user.username}")
        
        # Get user-specific documents
        documents = await document_service.document_repository.list_user_documents(
            current_user.user_id, skip, limit
        )
        
        logger.info(f"📄 Found {len(documents)} documents for user {current_user.username}")
        logger.debug(f"📄 User ID: {current_user.user_id}")
        logger.debug(f"📄 Documents: {[doc.document_id for doc in documents]}")
        return DocumentListResponse(documents=documents, total=len(documents))
        
    except Exception as e:
        logger.error(f"❌ Failed to list user documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user/collection/stats")
async def get_user_collection_stats(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get user's collection statistics"""
    try:
        logger.info(f"📊 Getting collection stats for user {current_user.username}")
        
        # Get stats from embedding manager
        stats = await document_service.embedding_manager.get_user_collection_stats(current_user.user_id)
        
        return {
            "user_id": current_user.user_id,
            "username": current_user.username,
            "collection_stats": stats
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get user collection stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user/documents/debug")
async def debug_user_documents(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Debug endpoint to check user documents"""
    try:
        logger.info(f"🔍 Debug: Checking documents for user {current_user.username}")
        
        # Get user documents directly from repository
        documents = await document_service.document_repository.list_user_documents(
            current_user.user_id, 0, 100
        )
        
        # Get all documents to compare
        all_documents = await document_service.document_repository.list_documents(0, 100)
        
        return {
            "user_id": current_user.user_id,
            "username": current_user.username,
            "user_documents_count": len(documents),
            "user_documents": [
                {
                    "document_id": doc.document_id,
                    "filename": doc.filename,
                    "user_id": getattr(doc, 'user_id', None),
                    "upload_date": doc.upload_date.isoformat() if doc.upload_date else None,
                    "status": doc.status.value if doc.status else None
                }
                for doc in documents
            ],
            "all_documents_count": len(all_documents),
            "all_documents_sample": [
                {
                    "document_id": doc.document_id,
                    "filename": doc.filename,
                    "user_id": getattr(doc, 'user_id', None),
                    "upload_date": doc.upload_date.isoformat() if doc.upload_date else None
                }
                for doc in all_documents[:5]  # Show first 5
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Debug failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/folders/debug")
async def debug_folders(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Debug endpoint to check folders"""
    try:
        logger.info(f"🔍 Debug: Checking folders for user {current_user.username}")
        
        # Get user folders
        user_folders = await folder_service.document_repository.get_folders_by_user(current_user.user_id, "user")
        
        # Get global folders
        global_folders = await folder_service.document_repository.get_folders_by_user(None, "global")
        
        # Get folder tree
        folder_tree = await folder_service.get_folder_tree(current_user.user_id, "user")
        
        return {
            "user_id": current_user.user_id,
            "username": current_user.username,
            "role": current_user.role,
            "user_folders_count": len(user_folders),
            "global_folders_count": len(global_folders),
            "folder_tree_count": len(folder_tree),
            "user_folders": [
                {
                    "folder_id": folder["folder_id"],
                    "name": folder["name"],
                    "collection_type": folder["collection_type"],
                    "user_id": folder["user_id"],
                    "parent_folder_id": folder["parent_folder_id"]
                }
                for folder in user_folders
            ],
            "global_folders": [
                {
                    "folder_id": folder["folder_id"],
                    "name": folder["name"],
                    "collection_type": folder["collection_type"],
                    "user_id": folder["user_id"],
                    "parent_folder_id": folder["parent_folder_id"]
                }
                for folder in global_folders
            ],
            "folder_tree": [
                {
                    "folder_id": folder.folder_id,
                    "name": folder.name,
                    "collection_type": folder.collection_type,
                    "user_id": folder.user_id,
                    "parent_folder_id": folder.parent_folder_id
                }
                for folder in folder_tree
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Folder debug failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/api/admin/clear-documents")
async def clear_all_documents(
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Clear all documents from all user folders, vector DB collections, and knowledge graph (admin only)"""
    try:
        logger.info(f"🗑️ Admin {current_user.username} starting complete document clearance")
        
        # Initialize counters
        deleted_documents = 0
        deleted_collections = 0
        errors = []
        
        # Step 1: Get all documents across the system
        logger.info("📋 Getting all documents from database...")
        all_documents = await document_service.list_documents(skip=0, limit=10000)
        logger.info(f"📋 Found {len(all_documents)} documents to delete")
        
        # Step 2: Delete all documents (this will also clean up vector embeddings and knowledge graph entities)
        logger.info("🗑️ Deleting all documents...")
        for doc in all_documents:
            try:
                success = await document_service.delete_document(doc.document_id)
                if success:
                    deleted_documents += 1
                    logger.info(f"🗑️ Deleted document: {doc.filename} ({doc.document_id})")
                else:
                    errors.append(f"Failed to delete document {doc.filename}")
            except Exception as e:
                error_msg = f"Error deleting document {doc.filename}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"❌ {error_msg}")
        
        # Step 3: Get all users and clear their vector collections
        logger.info("👥 Getting all users to clear their collections...")
        users_response = await auth_service.get_users(skip=0, limit=1000)
        users = users_response.users
        
        logger.info(f"👥 Found {len(users)} users, clearing their vector collections...")
        user_doc_service = UserDocumentService()
        await user_doc_service.initialize()
        
        for user in users:
            try:
                success = await user_doc_service.delete_user_collection(user.user_id)
                if success:
                    deleted_collections += 1
                    logger.info(f"🗑️ Cleared collection for user: {user.username}")
                else:
                    errors.append(f"Failed to clear collection for user {user.username}")
            except Exception as e:
                error_msg = f"Error clearing collection for user {user.username}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"❌ {error_msg}")
        
        # Step 4: Clear the global vector collection completely
        logger.info("🗑️ Clearing global vector collection...")
        try:
            from qdrant_client import QdrantClient
            from config import settings
            
            qdrant_client = QdrantClient(url=settings.QDRANT_URL)
            
            # Check if global collection exists
            collections = qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if settings.VECTOR_COLLECTION_NAME in collection_names:
                # Delete and recreate the global collection
                qdrant_client.delete_collection(settings.VECTOR_COLLECTION_NAME)
                
                # Recreate empty collection
                from qdrant_client.models import VectorParams, Distance
                qdrant_client.create_collection(
                    collection_name=settings.VECTOR_COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSIONS,
                        distance=Distance.COSINE
                    )
                )
                logger.info("🗑️ Cleared and recreated global vector collection")
            else:
                logger.info("ℹ️ Global vector collection didn't exist")
        except Exception as e:
            error_msg = f"Error clearing global vector collection: {str(e)}"
            errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
        
        # Step 5: Clear knowledge graph completely
        logger.info("🗑️ Clearing knowledge graph...")
        try:
            if knowledge_graph_service:
                await knowledge_graph_service.clear_all_data()
                logger.info("🗑️ Cleared all knowledge graph data")
            else:
                logger.warning("⚠️ Knowledge graph service not available")
        except Exception as e:
            error_msg = f"Error clearing knowledge graph: {str(e)}"
            errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
        
        # Step 6: Clean up orphaned files
        logger.info("🧹 Cleaning up orphaned files...")
        try:
            from pathlib import Path
            upload_dir = Path(settings.UPLOAD_DIR)
            processed_dir = Path(settings.PROCESSED_DIR) if hasattr(settings, 'PROCESSED_DIR') else None
            
            deleted_files = 0
            
            # Clean upload directory
            if upload_dir.exists():
                for file_path in upload_dir.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                        deleted_files += 1
                logger.info(f"🧹 Cleaned {deleted_files} files from upload directory")
            
            # Clean processed directory if it exists
            if processed_dir and processed_dir.exists():
                processed_files = 0
                for file_path in processed_dir.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                        processed_files += 1
                logger.info(f"🧹 Cleaned {processed_files} files from processed directory")
        except Exception as e:
            error_msg = f"Error cleaning up files: {str(e)}"
            errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
        
        # Prepare response
        success_message = f"✅ Clearance completed: {deleted_documents} documents deleted, {deleted_collections} user collections cleared"
        
        if errors:
            success_message += f". {len(errors)} errors encountered."
            logger.warning(f"⚠️ Clearance completed with {len(errors)} errors")
        else:
            logger.info("✅ Complete document clearance successful")
        
        return {
            "success": True,
            "message": success_message,
            "deleted_documents": deleted_documents,
            "deleted_collections": deleted_collections,
            "errors": errors[:10] if errors else [],  # Limit errors shown
            "total_errors": len(errors)
        }
        
    except Exception as e:
        logger.error(f"❌ Admin clearance failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")


@app.post("/api/admin/clear-neo4j")
async def clear_neo4j(
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Clear all data from Neo4j knowledge graph (admin only)"""
    try:
        logger.info(f"🗑️ Admin {current_user.username} clearing Neo4j knowledge graph")
        
        if not knowledge_graph_service:
            raise HTTPException(status_code=503, detail="Knowledge graph service not available")
        
        # Clear all knowledge graph data
        await knowledge_graph_service.clear_all_data()
        
        # Get stats to confirm clearing
        try:
            stats = await knowledge_graph_service.get_graph_stats()
            logger.info(f"📊 Neo4j stats after clear: {stats}")
        except Exception as e:
            logger.warning(f"⚠️ Could not get stats after clearing: {e}")
            stats = {"total_entities": 0, "total_documents": 0, "total_relationships": 0}
        
        logger.info("✅ Neo4j knowledge graph cleared successfully")
        
        return {
            "success": True,
            "message": "✅ Neo4j knowledge graph cleared successfully",
            "stats_after_clear": stats
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to clear Neo4j: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear Neo4j: {str(e)}")


@app.post("/api/admin/clear-qdrant")
async def clear_qdrant(
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Clear all data from Qdrant vector database (admin only)"""
    try:
        logger.info(f"🗑️ Admin {current_user.username} clearing Qdrant vector database")
        
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance
        from config import settings
        
        # Initialize counters
        cleared_collections = 0
        cleared_global = False
        errors = []
        
        # Initialize Qdrant client
        qdrant_client = QdrantClient(url=settings.QDRANT_URL)
        
        # Get all existing collections
        try:
            collections = qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            logger.info(f"📋 Found {len(collection_names)} collections in Qdrant")
        except Exception as e:
            logger.error(f"❌ Failed to list Qdrant collections: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to connect to Qdrant: {str(e)}")
        
        # Clear global collection
        if settings.VECTOR_COLLECTION_NAME in collection_names:
            try:
                # Delete and recreate global collection
                qdrant_client.delete_collection(settings.VECTOR_COLLECTION_NAME)
                
                # Recreate empty global collection
                qdrant_client.create_collection(
                    collection_name=settings.VECTOR_COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSIONS,
                        distance=Distance.COSINE
                    )
                )
                cleared_global = True
                cleared_collections += 1
                logger.info(f"🗑️ Cleared and recreated global collection: {settings.VECTOR_COLLECTION_NAME}")
            except Exception as e:
                error_msg = f"Failed to clear global collection {settings.VECTOR_COLLECTION_NAME}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"❌ {error_msg}")
        
        # Clear user collections (collections that start with 'user_')
        user_collections = [name for name in collection_names if name.startswith('user_')]
        logger.info(f"👥 Found {len(user_collections)} user collections to clear")
        
        for collection_name in user_collections:
            try:
                qdrant_client.delete_collection(collection_name)
                cleared_collections += 1
                logger.info(f"🗑️ Deleted user collection: {collection_name}")
            except Exception as e:
                error_msg = f"Failed to delete collection {collection_name}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"❌ {error_msg}")
        
        # Get final collection count
        try:
            final_collections = qdrant_client.get_collections()
            remaining_count = len(final_collections.collections)
            logger.info(f"📊 Collections remaining after clear: {remaining_count}")
        except Exception as e:
            logger.warning(f"⚠️ Could not get final collection count: {e}")
            remaining_count = "unknown"
        
        # Prepare response
        success_message = f"✅ Qdrant cleared: {cleared_collections} collections processed"
        if cleared_global:
            success_message += " (including global collection)"
        
        if errors:
            success_message += f", {len(errors)} errors encountered"
            logger.warning(f"⚠️ Qdrant clearing completed with {len(errors)} errors")
        else:
            logger.info("✅ Qdrant vector database cleared successfully")
        
        return {
            "success": True,
            "message": success_message,
            "cleared_collections": cleared_collections,
            "cleared_global": cleared_global,
            "remaining_collections": remaining_count,
            "errors": errors[:5] if errors else [],  # Limit errors shown
            "total_errors": len(errors)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to clear Qdrant: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear Qdrant: {str(e)}")


@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents(skip: int = 0, limit: int = 100):
    """List global/admin documents only"""
    try:
        # Get only global documents (admin uploads or approved submissions)
        documents = await document_service.document_repository.list_global_documents(skip, limit)
        return DocumentListResponse(documents=documents, total=len(documents))
        
    except Exception as e:
        logger.error(f"❌ Failed to list documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{doc_id}/status", response_model=DocumentStatus)
async def get_processing_status(doc_id: str):
    """Get document processing status and quality metrics"""
    try:
        status = await document_service.get_document_status(doc_id)
        if not status:
            raise HTTPException(status_code=404, detail="Document not found")
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get document status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/{doc_id}/reprocess")
async def reprocess_document(doc_id: str, current_user: AuthenticatedUserResponse = Depends(require_admin())):
    """Re-process a failed or completed document"""
    try:
        logger.info(f"🔄 Re-processing document: {doc_id}")
        
        # Get document info
        doc_info = await document_service.get_document(doc_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Reset status to processing
        await document_service.document_repository.update_status(doc_id, ProcessingStatus.PROCESSING)
        
        # Clear existing embeddings for this document
        try:
            await document_service.embedding_manager.delete_document_chunks(doc_id)
            logger.info(f"🗑️  Cleared existing embeddings for document {doc_id}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to clear embeddings for {doc_id}: {e}")
        
        # Determine file path using new folder structure
        file_path = None
        
        # Try new folder structure first
        try:
            from services.service_container import get_service_container
            container = await get_service_container()
            folder_service = container.folder_service
            
            logger.info(f"🔍 Looking for file: {doc_info.filename}")
            logger.info(f"🔍 Document folder_id: {doc_info.folder_id if hasattr(doc_info, 'folder_id') else 'None'}")
            logger.info(f"🔍 Document user_id: {doc_info.user_id if hasattr(doc_info, 'user_id') else 'None'}")
            
            # Try without doc_id prefix first (new style)
            folder_path = await folder_service.get_document_file_path(
                filename=doc_info.filename,
                folder_id=doc_info.folder_id if hasattr(doc_info, 'folder_id') else None,
                user_id=doc_info.user_id if hasattr(doc_info, 'user_id') else None,
                collection_type="global" if not doc_info.user_id else "user"
            )
            
            logger.info(f"🔍 Checking path (no prefix): {folder_path}")
            
            if folder_path and Path(folder_path).exists():
                file_path = Path(folder_path)
                logger.info(f"📂 Found file in new structure (no prefix): {file_path}")
            else:
                # Try with doc_id prefix (legacy style)
                filename_with_id = f"{doc_id}_{doc_info.filename}"
                folder_path = await folder_service.get_document_file_path(
                    filename=filename_with_id,
                    folder_id=doc_info.folder_id if hasattr(doc_info, 'folder_id') else None,
                    user_id=doc_info.user_id if hasattr(doc_info, 'user_id') else None,
                    collection_type="global" if not doc_info.user_id else "user"
                )
                
                logger.info(f"🔍 Checking path (with prefix): {folder_path}")
                
                if folder_path and Path(folder_path).exists():
                    file_path = Path(folder_path)
                    logger.info(f"📂 Found file in new structure (with prefix): {file_path}")
        except Exception as e:
            logger.warning(f"⚠️ Error checking new folder structure: {e}")
        
        # Fallback to legacy structure
        if not file_path:
            upload_dir = Path(settings.UPLOAD_DIR)
            for potential_file in upload_dir.glob(f"{doc_id}_*"):
                file_path = potential_file
                logger.info(f"📂 Found file in legacy structure: {file_path}")
                break
        
        if not file_path or not file_path.exists():
            # If original file not found, check if it's a URL import
            if doc_info.doc_type == DocumentType.URL:
                # Re-import from URL (global documents don't have user_id)
                asyncio.create_task(document_service._process_url_async(doc_id, doc_info.filename, "html"))
                logger.info(f"🔗 Re-importing URL document: {doc_info.filename}")
            else:
                raise HTTPException(status_code=404, detail="Original file not found - cannot reprocess")
        else:
            # Re-process the existing file, preserving original user_id/collection type
            doc_type = document_service._detect_document_type(doc_info.filename)
            asyncio.create_task(document_service._process_document_async(
                doc_id, 
                file_path, 
                doc_type, 
                user_id=doc_info.user_id  # Preserve original collection type
            ))
            logger.info(f"📄 Re-processing file: {file_path} (user_id={doc_info.user_id}, collection={'global' if not doc_info.user_id else 'user'})")
        
        logger.info(f"✅ Document {doc_id} queued for re-processing")
        return {
            "status": "success", 
            "message": f"Document {doc_id} queued for re-processing",
            "document_id": doc_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to re-process document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/user/documents/rescan")
async def rescan_user_files(
    dry_run: bool = False,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Scan user's filesystem and recover orphaned files
    
    **BULLY!** Bring those lost files back into the fold!
    
    Args:
        dry_run: If true, only report what would be recovered without making changes
    
    Returns:
        Recovery results with statistics
    """
    try:
        from services.file_recovery_service import get_file_recovery_service
        
        logger.info(f"🔍 ROOSEVELT: Rescanning files for user {current_user.user_id}")
        
        recovery_service = await get_file_recovery_service()
        result = await recovery_service.scan_and_recover_user_files(
            user_id=current_user.user_id,
            dry_run=dry_run
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ File rescan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/user/documents/{doc_id}/reprocess")
async def reprocess_user_document(doc_id: str, current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Re-process a user's failed or completed document"""
    try:
        logger.info(f"🔄 Re-processing user document: {doc_id} for user: {current_user.user_id}")
        
        # Get document info and verify ownership
        doc_info = await document_service.get_document(doc_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if user owns this document (this would need to be implemented in the document service)
        # For now, we'll assume user documents are stored with user_id in the database
        # This is a simplified check - in a real implementation, you'd verify ownership
        
        # Reset status to processing
        await document_service.document_repository.update_status(doc_id, ProcessingStatus.PROCESSING)
        
        # Clear existing embeddings for this document
        try:
            await document_service.embedding_manager.delete_document_chunks(doc_id)
            logger.info(f"🗑️  Cleared existing embeddings for document {doc_id}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to clear embeddings for {doc_id}: {e}")
        
        # Determine file path using new folder structure
        file_path = None
        
        # Try new folder structure first
        try:
            from services.service_container import get_service_container
            container = await get_service_container()
            folder_service = container.folder_service
            
            logger.info(f"🔍 Looking for file: {doc_info.filename}")
            logger.info(f"🔍 Document folder_id: {doc_info.folder_id if hasattr(doc_info, 'folder_id') else 'None'}")
            logger.info(f"🔍 Document user_id: {doc_info.user_id if hasattr(doc_info, 'user_id') else 'None'}")
            
            # Try without doc_id prefix first (new style)
            folder_path = await folder_service.get_document_file_path(
                filename=doc_info.filename,
                folder_id=doc_info.folder_id if hasattr(doc_info, 'folder_id') else None,
                user_id=current_user.user_id,
                collection_type="user"
            )
            
            logger.info(f"🔍 Checking path (no prefix): {folder_path}")
            
            if folder_path and Path(folder_path).exists():
                file_path = Path(folder_path)
                logger.info(f"📂 Found file in new structure (no prefix): {file_path}")
            else:
                # Try with doc_id prefix (legacy style)
                filename_with_id = f"{doc_id}_{doc_info.filename}"
                folder_path = await folder_service.get_document_file_path(
                    filename=filename_with_id,
                    folder_id=doc_info.folder_id if hasattr(doc_info, 'folder_id') else None,
                    user_id=current_user.user_id,
                    collection_type="user"
                )
                
                logger.info(f"🔍 Checking path (with prefix): {folder_path}")
                
                if folder_path and Path(folder_path).exists():
                    file_path = Path(folder_path)
                    logger.info(f"📂 Found file in new structure (with prefix): {file_path}")
        except Exception as e:
            logger.warning(f"⚠️ Error checking new folder structure: {e}")
        
        # Fallback to legacy structure
        if not file_path:
            upload_dir = Path(settings.UPLOAD_DIR)
            for potential_file in upload_dir.glob(f"{doc_id}_*"):
                file_path = potential_file
                logger.info(f"📂 Found file in legacy structure: {file_path}")
                break
        
        if not file_path or not file_path.exists():
            # If original file not found, check if it's a URL import
            if doc_info.doc_type == DocumentType.URL:
                # Re-import from URL with user_id
                asyncio.create_task(document_service._process_url_async(doc_id, doc_info.filename, "html", current_user.user_id))
                logger.info(f"🔗 Re-importing URL document: {doc_info.filename}")
            else:
                raise HTTPException(status_code=404, detail="Original file not found - cannot reprocess")
        else:
            # Re-process the existing file with user_id
            doc_type = document_service._detect_document_type(doc_info.filename)
            asyncio.create_task(document_service._process_document_async(doc_id, file_path, doc_type, current_user.user_id))
            logger.info(f"📄 Re-processing file: {file_path}")
        
        logger.info(f"✅ User document {doc_id} queued for re-processing")
        return {
            "status": "success", 
            "message": f"Document {doc_id} queued for re-processing",
            "document_id": doc_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to re-process user document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def check_document_access(
    doc_id: str,
    current_user: AuthenticatedUserResponse,
    required_permission: str = "read"
) -> DocumentInfo:
    """
    Check if user has access to a document
    
    Args:
        doc_id: Document ID
        current_user: Current authenticated user
        required_permission: "read", "write", or "delete"
        
    Returns:
        DocumentInfo if access granted
        
    Raises:
        HTTPException: 403 if access denied, 404 if not found
    """
    doc_info = await document_service.get_document(doc_id)
    if not doc_info:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Admin has full access
    if current_user.role == "admin":
        return doc_info
    
    collection_type = getattr(doc_info, 'collection_type', 'user')
    doc_user_id = getattr(doc_info, 'user_id', None)
    doc_team_id = getattr(doc_info, 'team_id', None)
    
    # Global collection: read-only for users, write for admins
    if collection_type == 'global':
        if required_permission in ['write', 'delete']:
            raise HTTPException(status_code=403, detail="Only admins can modify global documents")
        return doc_info  # Anyone can read global docs
    
    # Team collection: check team membership
    if doc_team_id:
        from api.teams_api import team_service
        role = await team_service.check_team_access(doc_team_id, current_user.user_id)
        if not role:
            raise HTTPException(status_code=403, detail="Not a team member")
        
        # Check permission based on team role
        if required_permission == 'delete':
            if role != 'admin':
                raise HTTPException(status_code=403, detail="Only team admins can delete team documents")
        
        return doc_info
    
    # User collection: only owner has access
    if doc_user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return doc_info


@app.get("/api/documents/{doc_id}/pdf")
async def get_document_pdf(
    doc_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Serve the original PDF file for a document"""
    try:
        logger.info(f"📄 Serving PDF file for document: {doc_id}")
        
        # SECURITY: Check access authorization
        doc_info = await check_document_access(doc_id, current_user, "read")
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if it's a PDF document
        if not doc_info.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Document is not a PDF")
        
        # **ROOSEVELT FIX**: Use folder service to get correct file path
        from pathlib import Path
        import os
        
        filename = getattr(doc_info, 'filename', None)
        user_id = getattr(doc_info, 'user_id', None)
        folder_id = getattr(doc_info, 'folder_id', None)
        collection_type = getattr(doc_info, 'collection_type', 'user')
        
        # SECURITY: Sanitize filename to prevent path traversal
        if filename:
            safe_filename = os.path.basename(filename)
            if not safe_filename or safe_filename in ('.', '..'):
                logger.error(f"Invalid filename in document metadata: {filename}")
                raise HTTPException(status_code=500, detail="Invalid file metadata")
            filename = safe_filename
        
        file_path = None
        
        if filename:
            # Try new folder structure first
            try:
                file_path_str = await folder_service.get_document_file_path(
                    filename=filename,
                    folder_id=folder_id,
                    user_id=user_id,
                    collection_type=collection_type
                )
                file_path = Path(file_path_str)
                
                # SECURITY: Verify resolved path is within uploads directory
                try:
                    uploads_base = Path(settings.UPLOAD_DIR).resolve()
                    file_path_resolved = file_path.resolve()
                    
                    if not str(file_path_resolved).startswith(str(uploads_base)):
                        logger.error(f"Path traversal attempt detected in document: {doc_id} -> {file_path_resolved}")
                        raise HTTPException(status_code=403, detail="Access denied")
                except Exception as e:
                    logger.error(f"Path validation error for document {doc_id}: {e}")
                    raise HTTPException(status_code=403, detail="Access denied")
                
                if not file_path.exists():
                    logger.warning(f"⚠️ PDF not found at computed path: {file_path}")
                    file_path = None
            except Exception as e:
                logger.warning(f"⚠️ Failed to compute file path with folder service: {e}")
        
        # Fall back to legacy flat structure if not found in new structure
        if file_path is None or not file_path.exists():
            upload_dir = Path(settings.UPLOAD_DIR)
            legacy_paths = [
                upload_dir / f"{doc_id}_{doc_info.filename}",
                upload_dir / doc_info.filename
            ]
            
            for legacy_path in legacy_paths:
                if legacy_path.exists():
                    file_path = legacy_path
                    logger.info(f"📄 Found PDF in legacy location: {file_path}")
                    break
        
        if file_path is None or not file_path.exists():
            raise HTTPException(status_code=404, detail="PDF file not found on disk")
        
        logger.info(f"✅ Serving PDF file: {file_path}")
        return FileResponse(
            path=str(file_path),
            filename=doc_info.filename,
            media_type="application/pdf"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to serve PDF file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{doc_id}/file")
async def get_document_file(
    doc_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Serve the original file for a document (audio, images, etc.)"""
    try:
        logger.info(f"📄 Serving file for document: {doc_id}")
        
        # SECURITY: Check access authorization
        doc_info = await check_document_access(doc_id, current_user, "read")
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get file path using folder service
        from pathlib import Path
        from fastapi.responses import FileResponse
        import os
        import mimetypes
        
        filename = getattr(doc_info, 'filename', None)
        user_id = getattr(doc_info, 'user_id', None)
        folder_id = getattr(doc_info, 'folder_id', None)
        collection_type = getattr(doc_info, 'collection_type', 'user')
        
        # SECURITY: Sanitize filename to prevent path traversal
        if filename:
            safe_filename = os.path.basename(filename)
            if not safe_filename or safe_filename in ('.', '..'):
                logger.error(f"Invalid filename in document metadata: {filename}")
                raise HTTPException(status_code=500, detail="Invalid file metadata")
            filename = safe_filename
        
        file_path = None
        
        if filename:
            # Try new folder structure first
            try:
                file_path_str = await folder_service.get_document_file_path(
                    filename=filename,
                    folder_id=folder_id,
                    user_id=user_id,
                    collection_type=collection_type
                )
                file_path = Path(file_path_str)
                
                # SECURITY: Verify resolved path is within uploads directory
                try:
                    uploads_base = Path(settings.UPLOAD_DIR).resolve()
                    file_path_resolved = file_path.resolve()
                    
                    if not str(file_path_resolved).startswith(str(uploads_base)):
                        logger.error(f"Path traversal attempt detected in document: {doc_id} -> {file_path_resolved}")
                        raise HTTPException(status_code=403, detail="Access denied")
                except Exception as e:
                    logger.error(f"Path validation error for document {doc_id}: {e}")
                    raise HTTPException(status_code=403, detail="Access denied")
                
                if not file_path.exists():
                    logger.warning(f"⚠️ File not found at computed path: {file_path}")
                    file_path = None
            except Exception as e:
                logger.warning(f"⚠️ Failed to compute file path with folder service: {e}")
        
        # Fall back to legacy flat structure if not found in new structure
        if file_path is None or not file_path.exists():
            upload_dir = Path(settings.UPLOAD_DIR)
            legacy_paths = [
                upload_dir / f"{doc_id}_{doc_info.filename}",
                upload_dir / doc_info.filename
            ]
            
            for legacy_path in legacy_paths:
                if legacy_path.exists():
                    file_path = legacy_path
                    logger.info(f"📄 Found file in legacy location: {file_path}")
                    break
        
        if file_path is None or not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")
        
        # Determine media type from file extension
        media_type, _ = mimetypes.guess_type(str(file_path))
        if not media_type:
            media_type = "application/octet-stream"
        
        logger.info(f"✅ Serving file: {file_path} (type: {media_type})")
        return FileResponse(
            path=str(file_path),
            filename=doc_info.filename,
            media_type=media_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to serve file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete a document and all its embeddings"""
    try:
        logger.info(f"🗑️  Deleting document: {doc_id}")
        
        # SECURITY: Check delete authorization
        await check_document_access(doc_id, current_user, "delete")
        
        success = await document_service.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"✅ Document {doc_id} deleted successfully by user {current_user.user_id}")
        return {"status": "success", "message": f"Document {doc_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/stats")
async def get_documents_stats():
    """Get statistics about stored documents and embeddings"""
    try:
        stats = await document_service.get_documents_stats()
        return stats
        
    except Exception as e:
        logger.error(f"❌ Failed to get documents stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/cleanup")
async def cleanup_orphaned_embeddings():
    """Clean up embeddings for documents that no longer exist"""
    try:
        logger.info("🧹 Starting cleanup of orphaned embeddings...")
        
        cleaned_count = await document_service.cleanup_orphaned_embeddings()
        
        return {
            "status": "success", 
            "message": f"Cleaned up {cleaned_count} orphaned document embeddings"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to cleanup orphaned embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/duplicates")
async def get_duplicate_documents():
    """Get all duplicate documents grouped by file hash"""
    try:
        logger.info("🔍 Getting duplicate documents")
        
        duplicates = await document_service.get_duplicate_documents()
        
        # Convert to a more API-friendly format
        duplicate_groups = []
        for file_hash, docs in duplicates.items():
            duplicate_groups.append({
                "file_hash": file_hash,
                "document_count": len(docs),
                "total_size": sum(doc.file_size for doc in docs),
                "documents": [
                    {
                        "document_id": doc.document_id,
                        "filename": doc.filename,
                        "upload_date": doc.upload_date.isoformat(),
                        "file_size": doc.file_size,
                        "status": doc.status.value
                    }
                    for doc in docs
                ]
            })
        
        logger.info(f"✅ Found {len(duplicate_groups)} groups of duplicate documents")
        return {
            "duplicate_groups": duplicate_groups,
            "total_groups": len(duplicate_groups),
            "total_duplicates": sum(group["document_count"] for group in duplicate_groups),
            "wasted_storage": sum(group["total_size"] - (group["total_size"] // group["document_count"]) for group in duplicate_groups)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get duplicate documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== DOCUMENT CATEGORIZATION ENDPOINTS =====

@app.post("/api/documents/filter", response_model=DocumentListResponse)
async def filter_documents(filter_request: DocumentFilterRequest):
    """Filter and search documents with advanced criteria"""
    try:
        logger.info(f"🔍 Filtering documents with criteria")
        
        result = await document_service.filter_documents(filter_request)
        
        logger.info(f"✅ Found {result.total} documents matching criteria")
        return result
        
    except Exception as e:
        logger.error(f"❌ Document filtering failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/documents/{doc_id}/metadata")
async def update_document_metadata(doc_id: str, update_request: DocumentUpdateRequest):
    """Update document metadata (title, category, tags, etc.)"""
    try:
        logger.info(f"📝 Updating metadata for document: {doc_id}")
        
        success = await document_service.update_document_metadata(doc_id, update_request)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"✅ Document metadata updated: {doc_id}")
        return {"status": "success", "message": f"Document {doc_id} metadata updated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update document metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/bulk-categorize", response_model=BulkOperationResponse)
async def bulk_categorize_documents(bulk_request: BulkCategorizeRequest):
    """Bulk categorize multiple documents"""
    try:
        logger.info(f"📋 Bulk categorizing {len(bulk_request.document_ids)} documents")
        
        result = await document_service.bulk_categorize_documents(bulk_request)
        
        logger.info(f"✅ Bulk categorization completed: {result.success_count} successful")
        return result
        
    except Exception as e:
        logger.error(f"❌ Bulk categorization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/categories", response_model=DocumentCategoriesResponse)
async def get_document_categories_overview():
    """Get overview of document categories and tags"""
    try:
        logger.info("📊 Getting document categories overview")
        
        overview = await document_service.get_document_categories_overview()
        
        logger.info(f"✅ Categories overview retrieved: {len(overview.categories)} categories, {len(overview.tags)} tags")
        return overview
        
    except Exception as e:
        logger.error(f"❌ Failed to get categories overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== FOLDER MANAGEMENT ENDPOINTS =====

@app.get("/api/folders/tree", response_model=FolderTreeResponse)
async def get_folder_tree(
    collection_type: str = "user",
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get the complete folder tree for the current user"""
    try:
        logger.debug(f"📁 Getting folder tree for user: {current_user.user_id}, collection_type: {collection_type}")
        folders = await folder_service.get_folder_tree(
            user_id=current_user.user_id, 
            collection_type=collection_type
        )
        logger.debug(f"📁 Found {len(folders)} folders")
        return FolderTreeResponse(
            folders=folders,
            total_folders=len(folders)
        )
    except Exception as e:
        logger.error(f"❌ Failed to get folder tree: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/folders/{folder_id}/contents", response_model=FolderContentsResponse)
async def get_folder_contents(
    folder_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get contents of a specific folder"""
    try:
        logger.info(f"🔍 API: Getting folder contents for {folder_id} (user: {current_user.user_id})")
        contents = await folder_service.get_folder_contents(folder_id, current_user.user_id)
        if not contents:
            logger.warning(f"⚠️ API: Folder {folder_id} not found or access denied for user {current_user.user_id}")
            raise HTTPException(status_code=404, detail="Folder not found or access denied")
        
        logger.info(f"✅ API: Returning folder contents for {folder_id}: {contents.total_documents} docs, {contents.total_subfolders} subfolders")
        return contents
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get folder contents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/folders", response_model=DocumentFolder)
async def create_folder(
    request: FolderCreateRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Create a new folder using FileManager for consistent WebSocket notifications"""
    try:
        logger.info(f"🔍 API: Create folder request from user {current_user.username} (role: {current_user.role})")
        logger.info(f"🔍 API: Request data: {request.dict()}")
        
        # Get FileManager service
        file_manager = await get_file_manager()
        
        # Determine collection type based on user role and request
        collection_type = "user"
        user_id = current_user.user_id
        team_id = None
        admin_user_id = None
        
        # Check if parent folder is provided - inherit collection type from parent
        if request.parent_folder_id:
            # Get parent folder to determine collection type and team_id
            parent_folder = await folder_service.get_folder(request.parent_folder_id, current_user.user_id, current_user.role)
            if parent_folder:
                parent_collection_type = parent_folder.collection_type or 'user'
                parent_team_id = parent_folder.team_id
                
                # Inherit collection type from parent
                if parent_collection_type == "team" and parent_team_id:
                    collection_type = "team"
                    team_id = str(parent_team_id) if parent_team_id else None
                    # For team folders, user_id must be NULL (per schema constraint)
                    user_id = None
                    # For team folders, use the creator's user_id as admin_user_id (for RLS and created_by)
                    admin_user_id = current_user.user_id
                    # Verify user is a team member
                    from api.teams_api import team_service
                    member_role = await team_service.check_team_access(team_id, current_user.user_id)
                    if not member_role:
                        raise HTTPException(status_code=403, detail="You must be a team member to create folders in team folders")
                    logger.info(f"🔍 API: Creating team folder - team_id: {team_id}, admin_user_id: {admin_user_id}")
                elif parent_collection_type == "global":
                    # Only admins can create folders in global folders
                    if current_user.role != "admin":
                        raise HTTPException(status_code=403, detail="Only admins can create folders in global folders")
                    collection_type = "global"
                    user_id = None
                    logger.info(f"🔍 API: Creating global folder")
                else:
                    # User folder - inherit user_id from parent or use current user
                    user_id = parent_folder.user_id or current_user.user_id
                    logger.info(f"🔍 API: Creating user folder - user_id: {user_id}")
                
                logger.info(f"🔍 API: Creating folder '{request.name}' in parent folder '{parent_folder.name}' (collection_type: {collection_type})")
            else:
                logger.warning(f"⚠️ Parent folder {request.parent_folder_id} not found, creating at root level")
        else:
            # No parent - check explicit collection_type in request
            if current_user.role == "admin" and getattr(request, 'collection_type', None) == "global":
                collection_type = "global"
                user_id = None  # Global folders have no user_id
                logger.info(f"🔍 API: Admin creating global folder - setting user_id to None")
            else:
                # Default to user folder
                logger.info(f"🔍 API: Creating folder '{request.name}' at root level (user folder)")
        
        logger.info(f"🔍 API: Final parameters - collection_type: {collection_type}, user_id: {user_id}, team_id: {team_id}")
        
        # Additional security validation
        if collection_type == "team":
            # Team folder creation - already validated team membership above
            pass
        elif current_user.role != "admin":
            # Regular users can only create folders for themselves
            if collection_type == "user" and user_id != current_user.user_id:
                raise HTTPException(status_code=403, detail="Regular users can only create folders for themselves")
            if collection_type == "global":
                raise HTTPException(status_code=403, detail="Regular users cannot create global folders")
        else:
            # Admins can only create folders for themselves or global folders
            if collection_type == "user" and user_id != current_user.user_id:
                raise HTTPException(status_code=403, detail="Admins can only create folders for themselves or global folders")
        
        # Build folder path
        folder_path = [request.name]
        
        # Create folder using FileManager for consistent WebSocket notifications
        # For team folders, admin_user_id should be the creator (for RLS context)
        # For user folders, admin_user_id should be the current user's ID (for validation)
        # For global folders, admin_user_id is only set if user is admin
        final_admin_user_id = None
        if collection_type == "team":
            # Team folders: admin_user_id is the creator (for RLS context)
            final_admin_user_id = admin_user_id or current_user.user_id
        elif collection_type == "user":
            # User folders: admin_user_id is the current user's ID (for validation in create_folder)
            final_admin_user_id = current_user.user_id
        elif current_user.role == "admin" and collection_type == "global":
            # Admin creating global folder
            final_admin_user_id = current_user.user_id
        
        folder_request = FolderStructureRequest(
            folder_path=folder_path,
            parent_folder_id=request.parent_folder_id,
            user_id=user_id,
            collection_type=collection_type,
            description=f"Folder created by {current_user.username}",
            current_user_role=current_user.role,
            admin_user_id=final_admin_user_id
        )
        
        logger.info(f"🔍 API: Creating folder via FileManager: {folder_request.dict()}")
        response = await file_manager.create_folder_structure(folder_request)
        
        # Get the created folder info to return
        # For team folders, use admin_user_id (creator) for RLS context
        # For user/global folders, use user_id (may be None for global)
        query_user_id = final_admin_user_id if collection_type == "team" else user_id
        folder = await folder_service.get_folder(response.folder_id, query_user_id, current_user.role)
        if not folder:
            raise HTTPException(status_code=500, detail="Folder created but could not retrieve folder info")
        
        # Single event system handles all notifications via FileManager
        logger.info(f"📡 Folder event notification handled by FileManager")
        
        logger.info(f"✅ API: Folder created successfully via FileManager: {folder.folder_id}")
        return folder
        
    except Exception as e:
        import traceback
        logger.error(f"❌ API: Failed to create folder via FileManager: {e}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/folders/{folder_id}/metadata")
async def update_folder_metadata(
    folder_id: str,
    request: FolderMetadataUpdateRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Update folder metadata (category, tags, inherit_tags)
    
    **ROOSEVELT FOLDER TAGGING PHASE 1**: Documents uploaded to this folder will inherit these tags!
    """
    try:
        logger.info(f"📋 Updating folder metadata: {folder_id} by user {current_user.username}")
        
        # Verify folder access
        folder = await folder_service.get_folder(folder_id, current_user.user_id, current_user.role)
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found or access denied")
        
        # Parse category value if provided
        category_value = request.category.value if request.category else None
        
        # Update metadata
        success = await folder_service.update_folder_metadata(
            folder_id,
            category=category_value,
            tags=request.tags,
            inherit_tags=request.inherit_tags
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update folder metadata")
        
        logger.info(f"✅ Folder metadata updated: {folder_id}")
        return {"success": True, "message": "Folder metadata updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update folder metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/folders/{folder_id}", response_model=DocumentFolder)
async def update_folder(
    folder_id: str,
    request: FolderUpdateRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Update folder information"""
    try:
        updated_folder = await folder_service.update_folder(folder_id, request, current_user.user_id, current_user.role)
        if not updated_folder:
            raise HTTPException(status_code=404, detail="Folder not found or access denied")
        return updated_folder
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/folders/{folder_id}")
async def delete_folder(
    folder_id: str,
    recursive: bool = False,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete a folder"""
    try:
        # Get folder info before deletion for WebSocket notification
        folder = await folder_service.get_folder(folder_id, current_user.user_id, current_user.role)
        
        success = await folder_service.delete_folder(folder_id, current_user.user_id, recursive, current_user.role)
        if not success:
            raise HTTPException(status_code=404, detail="Folder not found or access denied")
        
        # Send WebSocket notification for folder deletion
        if folder:
            try:
                from utils.websocket_manager import get_websocket_manager
                websocket_manager = get_websocket_manager()
                if websocket_manager:
                    folder_data = {
                        "folder_id": folder_id,
                        "name": folder.name,
                        "parent_folder_id": folder.parent_folder_id,
                        "user_id": folder.user_id,
                        "collection_type": folder.collection_type,
                        "team_id": folder.team_id if hasattr(folder, 'team_id') else None,
                        "deleted_at": datetime.now().isoformat()
                    }
                    notification = {
                        "type": "folder_event",
                        "action": "deleted",
                        "folder": folder_data,
                        "user_id": current_user.user_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # For team folders, send to all team members
                    if folder.collection_type == "team" and folder.team_id:
                        from api.teams_api import team_service
                        members = await team_service.get_team_members(str(folder.team_id), current_user.user_id)
                        for member in members:
                            await websocket_manager.send_to_session(notification, member.get('user_id'))
                        logger.info(f"📡 Sent folder deletion notification to {len(members)} team members")
                    else:
                        # For user/global folders, send only to the owner
                        await websocket_manager.send_to_session(notification, current_user.user_id)
                        logger.info(f"📡 Sent folder deletion notification for user {current_user.user_id}")
                else:
                    logger.warning("⚠️ WebSocket manager not available for folder deletion notification")
            except Exception as e:
                logger.warning(f"⚠️ Failed to send WebSocket notification: {e}")
        
        return {"message": "Folder deleted successfully"}
    except PermissionError as e:
        logger.warning(f"⛔ Permission denied for folder deletion: {e}")
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/folders/{folder_id}/move")
async def move_folder(
    folder_id: str,
    new_parent_id: str = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Move a folder to a new parent"""
    try:
        # Get current folder to capture old parent for notification
        old_folder = await folder_service.get_folder(folder_id, current_user.user_id, current_user.role)
        old_parent_id = getattr(old_folder, 'parent_folder_id', None) if old_folder else None

        success = await folder_service.move_folder(folder_id, new_parent_id, current_user.user_id, current_user.role)
        if not success:
            raise HTTPException(status_code=404, detail="Folder not found or access denied")

        # Get updated folder for notification payload
        updated_folder = await folder_service.get_folder(folder_id, current_user.user_id, current_user.role)

        # Notify via WebSocket so UI updates immediately
        try:
            from utils.websocket_manager import get_websocket_manager
            from datetime import datetime
            websocket_manager = get_websocket_manager()
            if websocket_manager and updated_folder:
                folder_data = {
                    "folder_id": updated_folder.folder_id,
                    "name": updated_folder.name,
                    "parent_folder_id": getattr(updated_folder, 'parent_folder_id', None),
                    "user_id": getattr(updated_folder, 'user_id', None),
                    "collection_type": getattr(updated_folder, 'collection_type', None),
                    "updated_at": datetime.now().isoformat()
                }
                await websocket_manager.send_to_session({
                    "type": "folder_event",
                    "action": "moved",
                    "folder": folder_data,
                    "old_parent_id": old_parent_id,
                    "new_parent_id": getattr(updated_folder, 'parent_folder_id', None),
                    "user_id": current_user.user_id,
                }, current_user.user_id)
        except Exception as ne:
            logger.warning(f"⚠️ Failed to send folder moved notification: {ne}")

        return {"message": "Folder moved successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to move folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/folders/default")
async def create_default_folders(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Create default folder structure for the current user"""
    try:
        folders = await folder_service.create_default_folders(current_user.user_id)
        return {
            "message": f"Default folders created successfully",
            "folders": [folder.dict() for folder in folders]
        }
    except Exception as e:
        logger.error(f"❌ Failed to create default folders: {e}")
        raise HTTPException(status_code=500, detail=str(e))










# ===== UNIVERSAL CATEGORY ENDPOINTS =====

@app.get("/api/categories")
async def get_all_categories():
    """Get all categories used across documents and notes"""
    try:
        logger.info("🏷️ Getting all universal categories")
        
        result = await category_service.get_all_categories()
        
        logger.info(f"✅ Retrieved {len(result['categories'])} categories")
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to get categories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/categories/suggestions")
async def get_category_suggestions(content: str = "", content_type: str = "note"):
    """Get category suggestions for content"""
    try:
        logger.info(f"🏷️ Getting category suggestions for {content_type}")
        
        suggestions = await category_service.suggest_categories(content, content_type)
        
        logger.info(f"✅ Generated {len(suggestions)} category suggestions")
        return {"suggestions": suggestions}
        
    except Exception as e:
        logger.error(f"❌ Failed to get category suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/categories")
async def create_category(request: dict):
    """Create a new category"""
    try:
        category_name = request.get("name", "").strip()
        description = request.get("description", "")
        category_type = request.get("type", "custom")
        
        if not category_name:
            raise HTTPException(status_code=400, detail="Category name is required")
        
        logger.info(f"🏷️ Creating category: {category_name}")
        
        category = await category_service.create_category(category_name, description, category_type)
        
        logger.info(f"✅ Category created: {category_name}")
        return category.dict()
        
    except Exception as e:
        logger.error(f"❌ Failed to create category: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/categories/{category_name}")
async def update_category(category_name: str, request: dict):
    """Update a category"""
    try:
        description = request.get("description", "")
        
        logger.info(f"🏷️ Updating category: {category_name}")
        
        success = await category_service.update_category(category_name, description)
        if not success:
            raise HTTPException(status_code=404, detail="Category not found")
        
        logger.info(f"✅ Category updated: {category_name}")
        return {"status": "success", "message": f"Category '{category_name}' updated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update category: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/categories/{category_name}")
async def delete_category(category_name: str):
    """Delete a category (only custom categories can be deleted)"""
    try:
        logger.info(f"🏷️ Deleting category: {category_name}")
        
        success = await category_service.delete_category(category_name)
        if not success:
            raise HTTPException(status_code=404, detail="Category not found or cannot be deleted")
        
        logger.info(f"✅ Category deleted: {category_name}")
        return {"status": "success", "message": f"Category '{category_name}' deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete category: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/categories/statistics")
async def get_category_statistics():
    """Get statistics about category usage"""
    try:
        logger.info("🏷️ Getting category statistics")
        
        stats = await category_service.get_category_statistics()
        
        logger.info("✅ Category statistics retrieved")
        return stats
        
    except Exception as e:
        logger.error(f"❌ Failed to get category statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))






# ===== OCR ENDPOINTS =====

@app.post("/api/ocr/process")
async def process_document_with_ocr(request: dict):
    """Process a document with OCR and optionally preserve hOCR data"""
    try:
        logger.info("🔄 Processing document with OCR")
        
        from models.api_models import OCRProcessingRequest
        from services.ocr_service import OCRService
        from services.direct_search_service import DirectSearchService
        
        ocr_request = OCRProcessingRequest(**request)
        
        # Initialize OCR service
        ocr_service = OCRService()
        await ocr_service.initialize()
        
        # Get document info to find the file path
        doc_info = await document_service.get_document(ocr_request.document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Construct file path
        from pathlib import Path
        upload_dir = Path(settings.UPLOAD_DIR)
        file_path = upload_dir / f"{ocr_request.document_id}_{doc_info.filename}"
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Document file not found")
        
        # Process with OCR
        result = await ocr_service.process_pdf_with_ocr(
            str(file_path),
            ocr_request.document_id,
            ocr_request.force_ocr,
            ocr_request.preserve_hocr
        )
        
        from models.api_models import OCRProcessingResponse
        return OCRProcessingResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ OCR processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ocr/{document_id}/hocr")
async def get_hocr_data(document_id: str):
    """Get hOCR data for a document"""
    try:
        logger.info(f"📄 Getting hOCR data for document: {document_id}")
        
        from services.ocr_service import OCRService
        
        ocr_service = OCRService()
        await ocr_service.initialize()
        
        hocr_data = await ocr_service.get_hocr_data(document_id)
        if not hocr_data:
            raise HTTPException(status_code=404, detail="hOCR data not found for this document")
        
        from models.api_models import HOCRData
        return HOCRData(**hocr_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get hOCR data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/ocr/{document_id}/update-word")
async def update_hocr_word(document_id: str, request: dict):
    """Update a word in the hOCR file"""
    try:
        logger.info(f"✏️ Updating hOCR word for document: {document_id}")
        
        from models.api_models import HOCRUpdateRequest
        from services.ocr_service import OCRService
        
        update_request = HOCRUpdateRequest(document_id=document_id, **request)
        
        ocr_service = OCRService()
        await ocr_service.initialize()
        
        success = await ocr_service.update_hocr_text(
            update_request.document_id,
            update_request.page_number,
            update_request.word_id,
            update_request.new_text
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Word not found or update failed")
        
        return {"status": "success", "message": "Word updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update hOCR word: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ocr/{document_id}/batch-update")
async def batch_update_hocr_words(document_id: str, request: dict):
    """Batch update multiple words in the hOCR file"""
    try:
        logger.info(f"✏️ Batch updating hOCR words for document: {document_id}")
        
        from models.api_models import HOCRBatchUpdateRequest
        from services.ocr_service import OCRService
        
        batch_request = HOCRBatchUpdateRequest(document_id=document_id, **request)
        
        ocr_service = OCRService()
        await ocr_service.initialize()
        
        result = await ocr_service.batch_update_words(
            batch_request.document_id,
            batch_request.updates
        )
        
        from models.api_models import HOCRBatchUpdateResponse
        return HOCRBatchUpdateResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Failed to batch update hOCR words: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ocr/{document_id}/confidence-stats")
async def get_ocr_confidence_stats(document_id: str):
    """Get OCR confidence statistics for a document"""
    try:
        logger.info(f"📊 Getting OCR confidence stats for document: {document_id}")
        
        from services.ocr_service import OCRService
        
        ocr_service = OCRService()
        await ocr_service.initialize()
        
        stats = await ocr_service.get_ocr_confidence_stats(document_id)
        if not stats:
            raise HTTPException(status_code=404, detail="OCR confidence stats not available for this document")
        
        from models.api_models import OCRConfidenceStats
        return OCRConfidenceStats(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get OCR confidence stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ocr/{document_id}/low-confidence-words")
async def get_low_confidence_words(document_id: str, threshold: int = 60):
    """Get words with low confidence scores for review"""
    try:
        logger.info(f"🔍 Getting low confidence words for document: {document_id}")
        
        from services.ocr_service import OCRService
        
        ocr_service = OCRService()
        await ocr_service.initialize()
        
        words = await ocr_service.get_low_confidence_words(document_id, threshold)
        
        from models.api_models import LowConfidenceWordsResponse, LowConfidenceWord
        return LowConfidenceWordsResponse(
            words=[LowConfidenceWord(**word) for word in words],
            total_count=len(words),
            threshold=threshold
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get low confidence words: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ocr/{document_id}/corrected-text")
async def get_corrected_text(document_id: str):
    """Export corrected text from hOCR file"""
    try:
        logger.info(f"📄 Exporting corrected text for document: {document_id}")
        
        from services.ocr_service import OCRService
        
        ocr_service = OCRService()
        await ocr_service.initialize()
        
        corrected_text = await ocr_service.export_corrected_text(document_id)
        if not corrected_text:
            raise HTTPException(status_code=404, detail="Corrected text not available for this document")
        
        # Get hOCR data for additional info
        hocr_data = await ocr_service.get_hocr_data(document_id)
        
        from models.api_models import CorrectedTextResponse
        return CorrectedTextResponse(
            document_id=document_id,
            corrected_text=corrected_text,
            page_count=hocr_data.get("page_count", 0) if hocr_data else 0,
            word_count=hocr_data.get("total_words", 0) if hocr_data else 0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to export corrected text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== DIRECT SEARCH ENDPOINTS =====

@app.post("/api/search/direct")
async def direct_search_documents(request: dict):
    """Perform direct semantic search without LLM processing"""
    try:
        logger.info(f"🔍 Direct search query: {request.get('query', '')[:100]}...")
        
        from models.api_models import DirectSearchRequest
        from services.direct_search_service import DirectSearchService
        
        search_request = DirectSearchRequest(**request)
        
        # Initialize direct search service
        direct_search_service = DirectSearchService()
        
        result = await direct_search_service.search_documents(
            query=search_request.query,
            limit=search_request.limit,
            similarity_threshold=search_request.similarity_threshold,
            document_types=search_request.document_types,
            categories=search_request.categories,
            tags=search_request.tags,
            date_from=search_request.date_from,
            date_to=search_request.date_to,
            include_metadata=search_request.include_metadata
        )
        
        from models.api_models import DirectSearchResponse
        return DirectSearchResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Direct search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search/filters")
async def get_search_filters():
    """Get available filter options for direct search"""
    try:
        logger.info("🔍 Getting search filter options")
        
        from services.direct_search_service import DirectSearchService
        
        direct_search_service = DirectSearchService()
        filters = await direct_search_service.get_search_filters()
        
        return filters
        
    except Exception as e:
        logger.error(f"❌ Failed to get search filters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search/suggestions")
async def get_search_suggestions(query: str, limit: int = 10):
    """Get search suggestions based on partial query"""
    try:
        logger.info(f"🔍 Getting search suggestions for: {query}")
        
        from services.direct_search_service import DirectSearchService
        
        direct_search_service = DirectSearchService()
        suggestions = await direct_search_service.get_search_suggestions(query, limit)
        
        return {"suggestions": suggestions}
        
    except Exception as e:
        logger.error(f"❌ Failed to get search suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/export")
async def export_search_results(request: dict):
    """Export search results in various formats"""
    try:
        logger.info("📤 Exporting search results")
        
        from services.direct_search_service import DirectSearchService
        
        results = request.get("results", [])
        format_type = request.get("format", "json")
        
        direct_search_service = DirectSearchService()
        export_result = await direct_search_service.export_search_results(results, format_type)
        
        if format_type.lower() == "csv":
            from fastapi.responses import Response
            return Response(
                content=export_result["data"],
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={export_result['filename']}"
                }
            )
        else:
            return export_result
        
    except Exception as e:
        logger.error(f"❌ Failed to export search results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== ENHANCED PDF SEGMENTATION ENDPOINTS =====

@app.post("/api/segmentation/enhanced/extract")
async def enhanced_extract_pdf_info(request: dict):
    """Extract PDF information without creating images (enhanced workflow)"""
    try:
        if not enhanced_pdf_segmentation_service:
            raise HTTPException(status_code=500, detail="Enhanced PDF segmentation service not initialized")
        
        from models.segmentation_models import PDFExtractionRequest
        extraction_request = PDFExtractionRequest(**request)
        
        result = await enhanced_pdf_segmentation_service.extract_pdf_info(extraction_request)
        return result.dict()
        
    except Exception as e:
        logger.error(f"❌ Enhanced PDF extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/segmentation/enhanced/extract-text")
async def extract_text_from_pdf_region(request: dict):
    """Extract text from a specific PDF region"""
    try:
        if not enhanced_pdf_segmentation_service:
            raise HTTPException(status_code=500, detail="Enhanced PDF segmentation service not initialized")
        
        from models.segmentation_models import PDFTextExtractionRequest
        text_request = PDFTextExtractionRequest(**request)
        
        result = await enhanced_pdf_segmentation_service.extract_text_from_region(text_request)
        return result.dict()
        
    except Exception as e:
        logger.error(f"❌ Text extraction from PDF region failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/segmentation/enhanced/segments")
async def create_enhanced_segment(request: dict):
    """Create a segment with automatic text extraction"""
    try:
        if not enhanced_pdf_segmentation_service:
            raise HTTPException(status_code=500, detail="Enhanced PDF segmentation service not initialized")
        
        from models.segmentation_models import CreateSegmentRequest
        segment_request = CreateSegmentRequest(**request)
        
        result = await enhanced_pdf_segmentation_service.create_segment_with_text_extraction(segment_request)
        return result.dict()
        
    except Exception as e:
        logger.error(f"❌ Enhanced segment creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/segmentation/enhanced/multiple-selections")
async def create_multiple_selections(request: dict):
    """Create multiple segments from selections"""
    try:
        if not enhanced_pdf_segmentation_service:
            raise HTTPException(status_code=500, detail="Enhanced PDF segmentation service not initialized")
        
        from models.segmentation_models import MultipleSelectionRequest
        selection_request = MultipleSelectionRequest(**request)
        
        result = await enhanced_pdf_segmentation_service.create_multiple_selections(selection_request)
        return result.dict()
        
    except Exception as e:
        logger.error(f"❌ Multiple selections creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/segmentation/enhanced/crop/{segment_id}")
async def crop_segment_to_pdf(segment_id: str, request: dict):
    """Crop a segment to a new PDF document"""
    try:
        if not enhanced_pdf_segmentation_service:
            raise HTTPException(status_code=500, detail="Enhanced PDF segmentation service not initialized")
        
        from models.segmentation_models import PDFSegmentCropRequest
        crop_request = PDFSegmentCropRequest(segment_id=segment_id, **request)
        
        result = await enhanced_pdf_segmentation_service.crop_segment_to_pdf(crop_request)
        return result.dict()
        
    except Exception as e:
        logger.error(f"❌ Segment cropping failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/segmentation/enhanced/segments/{segment_id}/text")
async def edit_segment_text(segment_id: str, request: dict):
    """Edit the text content of a segment"""
    try:
        if not enhanced_pdf_segmentation_service:
            raise HTTPException(status_code=500, detail="Enhanced PDF segmentation service not initialized")
        
        from models.segmentation_models import TextEditRequest
        edit_request = TextEditRequest(segment_id=segment_id, **request)
        
        result = await enhanced_pdf_segmentation_service.edit_segment_text(edit_request)
        return result.dict()
        
    except Exception as e:
        logger.error(f"❌ Segment text editing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/segmentation/enhanced/region-info/{document_id}/{page_number}")
async def get_pdf_region_info(document_id: str, page_number: int, x: float, y: float, width: float, height: float):
    """Get detailed information about a PDF region"""
    try:
        if not enhanced_pdf_segmentation_service:
            raise HTTPException(status_code=500, detail="Enhanced PDF segmentation service not initialized")
        
        from models.segmentation_models import SegmentBounds
        bounds = SegmentBounds(x=x, y=y, width=width, height=height)
        
        result = await enhanced_pdf_segmentation_service.get_pdf_region_info(document_id, page_number, bounds)
        return result.dict()
        
    except Exception as e:
        logger.error(f"❌ Failed to get PDF region info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/segmentation/enhanced/cropped-pdfs/{filename}")
async def download_cropped_pdf(filename: str):
    """Download a cropped PDF file"""
    try:
        from pathlib import Path
        
        cropped_pdfs_path = Path("processed/cropped_pdfs")
        file_path = cropped_pdfs_path / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Cropped PDF not found")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/pdf"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to download cropped PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== PDF TEXT LAYER ENDPOINTS =====

@app.post("/api/pdf-text/extract-layer")
async def extract_pdf_text_layer(request: dict):
    """Extract existing PDF text layer with word coordinates for editing"""
    try:
        logger.info("🔄 Extracting PDF text layer with coordinates")
        
        document_id = request.get("document_id")
        page_number = request.get("page_number", 1)
        
        if not document_id:
            raise HTTPException(status_code=400, detail="document_id is required")
        
        # Get document info
        doc_info = await document_service.get_document(document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Construct file path
        from pathlib import Path
        upload_dir = Path(settings.UPLOAD_DIR)
        file_path = upload_dir / f"{document_id}_{doc_info.filename}"
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Document file not found")
        
        # Debug: Check file access and try exact document processor approach
        logger.info(f"📝 File path: {file_path}")
        logger.info(f"📝 File exists: {os.path.exists(file_path)}")
        logger.info(f"📝 File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'}")
        
        text_blocks = []
        
        # For text layer editing, we need real coordinates, so go directly to PyMuPDF
        logger.info(f"📝 Using PyMuPDF for coordinate-based text extraction...")
        
        # Method: Use PyMuPDF to get words with precise coordinates
        try:
            with fitz.open(str(file_path)) as pdf_doc:
                if page_number > len(pdf_doc):
                    raise HTTPException(status_code=400, detail="Page number out of range")
                
                page = pdf_doc[page_number - 1]
                
                # Get words with coordinates - format: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
                words = page.get_text("words")
                
                logger.info(f"📝 Found {len(words)} words on page {page_number}")
                
                if len(words) > 0:
                    logger.info(f"📝 First few words: {words[:5]}")
                    
                    # Group PyMuPDF words into text blocks for easier editing
                    current_block = {"words": [], "bbox": [float('inf'), float('inf'), 0, 0]}
                    block_threshold = 10  # pixels - words closer than this are grouped

                    for word_data in words:
                        x0, y0, x1, y1, text, block_no, line_no, word_no = word_data
                        
                        word_info = {
                            "id": f"word_{page_number}_{block_no}_{line_no}_{word_no}",
                            "text": text,
                            "bbox": [float(x0), float(y0), float(x1), float(y1)],
                            "block_no": block_no,
                            "line_no": line_no,
                            "word_no": word_no
                        }
                        
                        # Check if this word should be in a new block
                        if (current_block["words"] and 
                            (abs(y0 - current_block["words"][-1]["bbox"][1]) > block_threshold or
                             block_no != current_block["words"][-1]["block_no"])):
                            
                            # Finalize current block
                            if current_block["words"]:
                                text_blocks.append({
                                    "id": f"block_{page_number}_{len(text_blocks)}",
                                    "text": " ".join(w["text"] for w in current_block["words"]),
                                    "bbox": current_block["bbox"],
                                    "words": current_block["words"],
                                    "confidence": 1.0  # High confidence for existing PDF text
                                })
                            
                            # Start new block
                            current_block = {"words": [word_info], "bbox": [x0, y0, x1, y1]}
                        else:
                            # Add to current block
                            current_block["words"].append(word_info)
                            # Update block bbox
                            current_block["bbox"][0] = min(current_block["bbox"][0], x0)
                            current_block["bbox"][1] = min(current_block["bbox"][1], y0) 
                            current_block["bbox"][2] = max(current_block["bbox"][2], x1)
                            current_block["bbox"][3] = max(current_block["bbox"][3], y1)
                    
                    # Add final block
                    if current_block["words"]:
                        text_blocks.append({
                            "id": f"block_{page_number}_{len(text_blocks)}",
                            "text": " ".join(w["text"] for w in current_block["words"]),
                            "bbox": current_block["bbox"],
                            "words": current_block["words"],
                            "confidence": 1.0
                        })
                        
                    logger.info(f"📝 Created {len(text_blocks)} text blocks from PyMuPDF")
                else:
                    logger.warning(f"⚠️ No words found with PyMuPDF")
                    
        except Exception as pymupdf_error:
            logger.error(f"📝 PyMuPDF extraction failed: {pymupdf_error}")
            
        # Fallback: Check if this document has hOCR data from previous OCR processing
        if not text_blocks:
            from services.ocr_service import OCRService
            try:
                # First try to get hOCR data (most likely for newspaper scans)
                logger.info(f"📝 Checking for existing hOCR data...")
                ocr_service = OCRService()
                await ocr_service.initialize()
                
                hocr_data = await ocr_service.get_hocr_data(document_id)
                if hocr_data and hocr_data.get("pages"):
                    logger.info(f"📝 Found hOCR data with {len(hocr_data['pages'])} pages")
                    
                    # Get the specific page from hOCR data
                    if page_number <= len(hocr_data["pages"]):
                        page_data = hocr_data["pages"][page_number - 1]
                        logger.info(f"📝 hOCR page has {len(page_data.get('lines', []))} lines")
                        
                        # Convert hOCR data to text blocks
                        for line_idx, line in enumerate(page_data.get("lines", [])):
                            if line.get("words"):
                                # Create a text block from each line
                                line_text = " ".join(word.get("text", "") for word in line["words"])
                                if line_text.strip():
                                    # Calculate bounding box for the line
                                    line_bbox = line.get("bbox", [0, 0, 100, 20])
                                    
                                    text_blocks.append({
                                        "id": f"hocr_block_{page_number}_{line_idx}",
                                        "text": line_text.strip(),
                                        "bbox": line_bbox,
                                        "words": line["words"],
                                        "confidence": 0.8  # hOCR confidence
                                    })
                        
                        logger.info(f"📝 Created {len(text_blocks)} text blocks from hOCR data")
                    else:
                        logger.warning(f"📝 Page {page_number} not found in hOCR data")
                else:
                    logger.info(f"📝 No hOCR data found for document {document_id}")
                    
            except Exception as hocr_error:
                logger.warning(f"📝 hOCR extraction failed: {hocr_error}")
        
        logger.info(f"📝 Created {len(text_blocks)} text blocks total")
        
        return {
            "document_id": document_id,
            "page_number": page_number,
            "text_blocks": text_blocks,
            "total_blocks": len(text_blocks),
            "extraction_method": "pdf_text_layer"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ PDF text layer extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pdf-text/update-block")
async def update_pdf_text_block(request: dict):
    """Update a text block in the PDF (for now, just store changes)"""
    try:
        logger.info("🔄 Updating PDF text block")
        
        document_id = request.get("document_id")
        page_number = request.get("page_number")
        block_id = request.get("block_id")
        new_text = request.get("new_text")
        
        if not all([document_id, page_number, block_id, new_text is not None]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        # For now, we'll store the edits in a simple format
        # In a production system, you'd want to store these in your database
        # and potentially recreate the PDF with updated text
        
        logger.info(f"✅ Text block update recorded: {block_id} -> '{new_text[:50]}...'")
        
        return {
            "success": True,
            "document_id": document_id,
            "page_number": page_number,
            "block_id": block_id,
            "updated_text": new_text,
            "message": "Text block updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ PDF text block update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== AUTHENTICATION ENDPOINTS =====

# ===== AUTHENTICATION ENDPOINTS MOVED TO api/auth_api.py =====






# ... existing code ...

@app.get("/api/documents/{doc_id}/content")
async def get_document_content(
    doc_id: str,
    request: Request,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get document content by ID"""
    try:
        logger.info(f"📄 API: Getting content for document {doc_id}")
        
        # SECURITY: Check read authorization
        document = await check_document_access(doc_id, current_user, "read")
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # ALWAYS get content from the actual file on disk
        full_content = None  # None = file not found, "" = empty file, "content" = file with content
        content_source = "file"
        
        try:
            from pathlib import Path
            
            # Use folder service to find the file in the new structure
            filename = getattr(document, 'filename', None)
            user_id = getattr(document, 'user_id', None)
            folder_id = getattr(document, 'folder_id', None)
            collection_type = getattr(document, 'collection_type', 'user')
            
            # Skip content loading for PDFs - they're served via the /pdf endpoint
            # But still compute the file path for canonical_path
            if filename and filename.lower().endswith('.pdf'):
                logger.info(f"📄 API: Skipping content load for PDF: {filename} (use /pdf endpoint instead)")
                full_content = ""  # Empty content for PDFs
                content_source = "pdf_binary"
                # Still compute file path for metadata
                try:
                    file_path_str = await folder_service.get_document_file_path(
                        filename=filename,
                        folder_id=folder_id,
                        user_id=user_id,
                        collection_type=collection_type
                    )
                    file_path = Path(file_path_str)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to compute file path for PDF: {e}")
            elif filename:
                # Try new folder structure first
                try:
                    file_path_str = await folder_service.get_document_file_path(
                        filename=filename,
                        folder_id=folder_id,
                        user_id=user_id,
                        collection_type=collection_type
                    )
                    file_path = Path(file_path_str)
                    
                    if file_path.exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            full_content = f.read()
                        logger.info(f"✅ API: Loaded content from file: {file_path}")
                    else:
                        logger.warning(f"⚠️ File not found at computed path: {file_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to compute file path with folder service: {e}")
                
                # Fall back to searching user directory for org files
                if full_content is None and filename.endswith('.org'):
                    # For org files, search user directory tree
                    username = await folder_service._get_username(user_id) if user_id else None
                    if username:
                        user_dir = Path(settings.UPLOAD_DIR) / "Users" / username
                        if user_dir.exists():
                            # Search recursively for the org file
                            matching_files = list(user_dir.rglob(filename))
                            if matching_files:
                                file_path = matching_files[0]  # Use first match
                                if file_path.exists():
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        full_content = f.read()
                                    logger.info(f"✅ API: Found org file in subdirectory: {file_path}")
                
                # Fall back to old paths if still not found
                if full_content is None:
                    upload_dir = Path(settings.UPLOAD_DIR)
                    legacy_paths = [
                        upload_dir / f"{doc_id}_{filename}",
                        upload_dir / filename
                    ]
                    
                    # For markdown, also check web_sources
                    if filename.endswith('.md'):
                        import glob
                        legacy_paths.extend([
                            upload_dir / "web_sources" / "rss_articles" / "*" / filename,
                            upload_dir / "web_sources" / "scraped_content" / "*" / filename
                        ])
                    
                    import glob
                    for path_pattern in legacy_paths:
                        matches = glob.glob(str(path_pattern)) if '*' in str(path_pattern) else [str(path_pattern)]
                        if matches:
                            file_path = Path(matches[0])
                            if file_path.exists():
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    full_content = f.read()
                                logger.info(f"✅ API: Loaded content from legacy path: {file_path}")
                                break
            
            # If file not found, this is an error - vectors are NOT for viewing
            # **BULLY!** Allow empty files - distinguish between "file not found" vs "empty file"
            if full_content is None:
                logger.error(f"❌ API: File not found for document {doc_id} (filename: {getattr(document, 'filename', 'unknown')})")
                raise HTTPException(status_code=404, detail=f"Document file not found on disk for {doc_id}")
            elif full_content == "":
                logger.info(f"📝 API: Document {doc_id} has empty content (file exists but is empty)")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ API: Failed to load file for document {doc_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load document file: {str(e)}")
        
        # **BULLY!** For editing, we need the full content WITH frontmatter
        # Only strip frontmatter for read-only display, not for editing
        display_content = full_content
        
        # **ROOSEVELT'S CANONICAL PATH FIX**: Ensure we always have canonical_path and folder info
        # Get folder information for frontend use
        folder_name = None
        try:
            if folder_id:
                folder = await folder_service.get_folder(folder_id, user_id)
                if folder:
                    # folder is a DocumentFolder Pydantic model, access attribute directly
                    folder_name = getattr(folder, 'name', None)
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch folder name: {e}")
        
        # Ensure canonical_path is set - this is critical for relative reference resolution!
        canonical_path_str = None
        if 'file_path' in locals() and file_path:
            canonical_path_str = str(file_path)
        else:
            # If we got here, we found the content but file_path wasn't preserved
            # Reconstruct it from the folder service
            try:
                if filename:
                    file_path_str = await folder_service.get_document_file_path(
                        filename=filename,
                        folder_id=folder_id,
                        user_id=user_id,
                        collection_type=collection_type
                    )
                    canonical_path_str = file_path_str
            except Exception as e:
                logger.warning(f"⚠️ Could not construct canonical_path: {e}")
        
        # Get metadata from document
        metadata = {
            "document_id": document.document_id,
            "title": document.title,
            "filename": document.filename,
            "author": document.author,
            "description": document.description,
            "category": document.category.value if document.category else None,
            "tags": document.tags,
            "created_at": document.upload_date.isoformat() if document.upload_date else None,
            "status": document.status.value if document.status else None,
            "file_size": document.file_size,
            "language": document.language,
            "user_id": getattr(document, 'user_id', None),
            "collection_type": getattr(document, 'collection_type', None),
            "folder_id": folder_id,  # **ROOSEVELT: Add folder_id**
            "folder_name": folder_name,  # **ROOSEVELT: Add folder_name for display**
            "canonical_path": canonical_path_str  # **ROOSEVELT: Reliable canonical path!**
        }
        
        response_data = {
            "content": display_content,
            "metadata": metadata,
            "total_length": len(display_content),
            "content_source": content_source,
            "chunk_count": 0  # For PDFs and other files, chunk count is not relevant for viewing
        }
        
        logger.info(f"✅ API: Returning content for {doc_id} from {content_source}: {len(full_content)} characters")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ API: Failed to get document content for {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document content: {str(e)}")


@app.post("/api/documents/{doc_id}/exempt")
async def exempt_document_from_vectorization(
    doc_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Exempt a document from vectorization and knowledge graph processing"""
    try:
        success = await document_service.exempt_document_from_vectorization(
            doc_id, 
            current_user.user_id
        )
        if success:
            return {"status": "success", "message": "Document exempted from vectorization"}
        else:
            raise HTTPException(status_code=500, detail="Failed to exempt document")
    except Exception as e:
        logger.error(f"❌ Failed to exempt document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{doc_id}/exempt")
async def remove_document_exemption(
    doc_id: str,
    inherit: bool = False,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Remove exemption from document.
    
    Args:
        doc_id: Document ID
        inherit: If True, set to inherit from folder. If False, set to explicit vectorize.
    """
    try:
        success = await document_service.remove_document_exemption(
            doc_id,
            current_user.user_id,
            inherit=inherit
        )
        if success:
            if inherit:
                return {"status": "success", "message": "Document now inherits from folder"}
            else:
                return {"status": "success", "message": "Document exemption removed and re-processed"}
        else:
            raise HTTPException(status_code=500, detail="Failed to remove exemption")
    except Exception as e:
        logger.error(f"❌ Failed to remove exemption for document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/folders/{folder_id}/exempt")
async def exempt_folder_from_vectorization(
    folder_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Exempt a folder and all descendants from vectorization"""
    logger.info(f"🚫 API: Exempting folder {folder_id} for user {current_user.user_id}")
    try:
        from services.service_container import get_service_container
        container = await get_service_container()
        folder_service = container.folder_service

        success = await folder_service.exempt_folder_from_vectorization(
            folder_id,
            current_user.user_id
        )
        if success:
            logger.info(f"✅ API: Folder {folder_id} exempted successfully")
            return {"status": "success", "message": "Folder and descendants exempted from vectorization"}
        else:
            logger.error(f"❌ API: Failed to exempt folder {folder_id} - method returned false")
            raise HTTPException(status_code=500, detail="Failed to exempt folder")
    except Exception as e:
        logger.error(f"❌ Failed to exempt folder {folder_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/folders/{folder_id}/exempt")
async def remove_folder_exemption(
    folder_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Remove exemption from a folder (set to inherit from parent), re-process all documents"""
    try:
        from services.service_container import get_service_container
        container = await get_service_container()
        folder_service = container.folder_service
        
        success = await folder_service.remove_folder_exemption(
            folder_id,
            current_user.user_id
        )
        if success:
            return {"status": "success", "message": "Folder exemption removed - now inherits from parent"}
        else:
            raise HTTPException(status_code=500, detail="Failed to remove exemption")
    except Exception as e:
        logger.error(f"❌ Failed to remove exemption for folder {folder_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/folders/{folder_id}/exempt/override")
async def override_folder_exemption(
    folder_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Set folder to explicitly NOT exempt (override parent exemption)"""
    try:
        from services.service_container import get_service_container
        container = await get_service_container()
        folder_service = container.folder_service
        
        success = await folder_service.override_folder_exemption(
            folder_id,
            current_user.user_id
        )
        if success:
            return {"status": "success", "message": "Folder set to override parent exemption - not exempt"}
        else:
            raise HTTPException(status_code=500, detail="Failed to set override")
    except Exception as e:
        logger.error(f"❌ Failed to set override for folder {folder_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/documents/{doc_id}/content")
async def update_document_content(
    doc_id: str,
    request: Request,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Update a text-based document's content on disk and re-embed chunks.
    Supports .txt, .md, .org. Non-text or binary docs are rejected.
    """
    try:
        # SECURITY: Check write authorization
        await check_document_access(doc_id, current_user, "write")
        
        body = await request.json()
        new_content = body.get("content")
        if new_content is None:
            raise HTTPException(status_code=400, detail="Missing 'content' in request body")

        # Fetch document metadata
        doc_info = await document_service.get_document(doc_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")

        filename = getattr(doc_info, 'filename', None) or ""
        if not filename:
            raise HTTPException(status_code=400, detail="Document filename missing")

        # Only allow text-editable types
        editable_exts = (".txt", ".md", ".org")
        if not str(filename).lower().endswith(editable_exts):
            raise HTTPException(status_code=400, detail="Only .txt, .md, and .org documents can be edited")

        # Locate file path on disk using folder service
        from pathlib import Path
        user_id = getattr(doc_info, 'user_id', None)
        folder_id = getattr(doc_info, 'folder_id', None)
        collection_type = getattr(doc_info, 'collection_type', 'user')
        
        file_path = None
        try:
            file_path_str = await folder_service.get_document_file_path(
                filename=filename,
                folder_id=folder_id,
                user_id=user_id,
                collection_type=collection_type
            )
            file_path = Path(file_path_str)
            
            if not file_path.exists():
                logger.warning(f"⚠️ File not found at computed path: {file_path}")
                file_path = None
        except Exception as e:
            logger.warning(f"⚠️ Failed to compute file path with folder service: {e}")
        
        # Fall back to legacy paths if not found
        if file_path is None or not file_path.exists():
            upload_dir = Path(settings.UPLOAD_DIR)
            legacy_paths = [
                upload_dir / f"{doc_id}_{filename}",
                upload_dir / filename
            ]
            
            # For markdown, also check web_sources
            if filename.lower().endswith('.md'):
                import glob
                legacy_paths.extend([
                    upload_dir / "web_sources" / "rss_articles" / "*" / filename,
                    upload_dir / "web_sources" / "scraped_content" / "*" / filename
                ])
            
            import glob
            for path_pattern in legacy_paths:
                matches = glob.glob(str(path_pattern)) if '*' in str(path_pattern) else [str(path_pattern)]
                if matches:
                    candidate = Path(matches[0])
                    if candidate.exists():
                        file_path = candidate
                        break
            
            if file_path is None or not file_path.exists():
                raise HTTPException(status_code=404, detail="Original file not found on disk")

        # Write content to disk
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            logger.info(f"📝 Updated file on disk: {file_path}")
        except Exception as e:
            logger.error(f"❌ Failed to write updated content: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save content: {str(e)}")

        # Update file size
        try:
            await document_service.document_repository.update_file_size(doc_id, len(new_content.encode('utf-8')))
        except Exception as e:
            logger.warning(f"⚠️ Failed to update file size metadata: {e}")

        # Check if document is exempt from vectorization BEFORE processing
        is_exempt = await document_service.document_repository.is_document_exempt(doc_id, current_user.user_id)
        if is_exempt:
            logger.info(f"🚫 Document {doc_id} is exempt from vectorization - skipping embedding and entity extraction")
            await document_service.document_repository.update_status(doc_id, ProcessingStatus.COMPLETED)
            return {"status": "success", "message": "Content updated (exempt from vectorization)", "document_id": doc_id}
        
        # Re-embed the updated content
        try:
            await document_service.document_repository.update_status(doc_id, ProcessingStatus.EMBEDDING)
            
            # **ROOSEVELT'S COMPLETE CLEANUP!** Delete old vectors AND knowledge graph entities
            await document_service.embedding_manager.delete_document_chunks(doc_id)
            
            # Delete old knowledge graph entities
            if document_service.kg_service:
                try:
                    await document_service.kg_service.delete_document_entities(doc_id)
                    logger.info(f"🗑️  Deleted old knowledge graph entities for {doc_id}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to delete old KG entities for {doc_id}: {e}")

            # **ROOSEVELT FOLDER INHERITANCE**: Apply folder metadata if document is in a folder
            document_category = None
            document_tags = None
            
            if folder_id:
                try:
                    from services.service_container import get_service_container
                    container = await get_service_container()
                    folder_service_inst = container.folder_service
                    folder_metadata = await folder_service_inst.get_folder_metadata(folder_id)
                    
                    if folder_metadata.get('inherit_tags', True):
                        folder_category = folder_metadata.get('category')
                        folder_tags = folder_metadata.get('tags', [])
                        
                        if folder_category or folder_tags:
                            # Get current document metadata
                            doc_category_val = getattr(doc_info, 'category', None)
                            doc_tags_val = getattr(doc_info, 'tags', []) or []
                            
                            # If document doesn't have category/tags, use folder's
                            if not doc_category_val and folder_category:
                                document_category = folder_category
                            else:
                                document_category = doc_category_val.value if doc_category_val else None
                            
                            # Merge document tags with folder tags
                            if folder_tags:
                                merged_tags = list(set(doc_tags_val + folder_tags))
                                document_tags = merged_tags
                            else:
                                document_tags = doc_tags_val
                            
                            logger.info(f"📋 FOLDER INHERITANCE (editor): Applied folder metadata to {doc_id} - category={document_category}, tags={document_tags}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to apply folder metadata inheritance in editor: {e}")
                    # Continue with whatever document already has
                    document_category = getattr(doc_info, 'category', None)
                    document_tags = getattr(doc_info, 'tags', None)
                    if document_category:
                        document_category = document_category.value if hasattr(document_category, 'value') else document_category
            else:
                # No folder - use document's existing metadata
                document_category = getattr(doc_info, 'category', None)
                document_tags = getattr(doc_info, 'tags', None)
                if document_category:
                    document_category = document_category.value if hasattr(document_category, 'value') else document_category
            
            # Process content into chunks directly to avoid re-reading file
            chunks = await document_service.document_processor.process_text_content(new_content, doc_id, {
                "filename": filename,
                "doc_type": str(getattr(doc_info, 'doc_type', 'txt')),
            })

            if chunks:
                # Embed with folder-inherited metadata
                await document_service.embedding_manager.embed_and_store_chunks(
                    chunks,
                    user_id=user_id,
                    document_category=document_category,
                    document_tags=document_tags
                )
                await document_service.document_repository.update_chunk_count(doc_id, len(chunks))
            
            # Extract entities (document is not exempt, so proceed)
            # **BULLY!** Extract and store NEW entities using PROPER spaCy NER
            elif document_service.kg_service:
                try:
                    # Use DocumentProcessor's sophisticated entity extraction (spaCy + patterns)
                    entities = await document_service.document_processor._extract_entities(new_content, chunks or [])
                    if entities:
                        await document_service.kg_service.store_entities(entities, doc_id)
                        logger.info(f"🔗 Extracted and stored {len(entities)} NEW entities using spaCy NER for {doc_id}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to extract/store new entities for {doc_id}: {e}")
                
                # **BULLY! ENTERTAINMENT KG EXTRACTION!** 🎬
                # If this is an entertainment document, extract domain-specific relationships
                try:
                    from services.entertainment_kg_extractor import get_entertainment_kg_extractor
                    ent_extractor = get_entertainment_kg_extractor()
                    
                    if doc_info and ent_extractor.should_extract_from_document(doc_info):
                        logger.info(f"🎬 Extracting entertainment entities from {doc_id}")
                        ent_entities, ent_relationships = ent_extractor.extract_entities_and_relationships(
                            new_content, doc_info
                        )
                        
                        if ent_entities or ent_relationships:
                            await document_service.kg_service.store_entertainment_entities_and_relationships(
                                ent_entities, ent_relationships, doc_id
                            )
                            logger.info(f"🎬 Stored entertainment graph: {len(ent_entities)} entities, {len(ent_relationships)} relationships")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to extract/store entertainment entities for {doc_id}: {e}")
            
            await document_service.document_repository.update_status(doc_id, ProcessingStatus.COMPLETED)
            logger.info(f"✅ Re-embedded document {doc_id} with {len(chunks) if chunks else 0} chunks")
        except Exception as e:
            logger.error(f"❌ Re-embedding failed: {e}")
            await document_service.document_repository.update_status(doc_id, ProcessingStatus.FAILED)
            raise HTTPException(status_code=500, detail=f"Re-embedding failed: {str(e)}")

        return {"status": "success", "message": "Content updated and re-embedded", "document_id": doc_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update document content for {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update document content: {str(e)}")

@app.get("/api/documents/{doc_id}/status", response_model=DocumentStatus)
async def get_processing_status(doc_id: str):
    """Get document processing status and quality metrics"""
    try:
        status = await document_service.get_document_status(doc_id)
        if not status:
            raise HTTPException(status_code=404, detail="Document not found")
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get document status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
