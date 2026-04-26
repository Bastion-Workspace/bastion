"""
Enhanced Async Orchestrator API
Background processing endpoints for the Enhanced Orchestrator
Now with multi-operation state management and context-aware routing!
"""

import logging
import json
import asyncio
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.celery_utils import safe_serialize_error
from services.prompt_service import prompt_service
from utils.auth_middleware import get_current_user, AuthenticatedUserResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Async Orchestrator"])


class AsyncOrchestratorRequest(BaseModel):
    """Request to start async orchestrator processing"""
    query: str
    conversation_id: str
    session_id: str = "default"
    persist_conversation: bool = True
    priority: str = "normal"  # normal, high, low
    active_editor: Optional[dict] = None  # {is_editable, filename, language, content, content_length, frontmatter}
    editor_preference: Optional[str] = None  # 'prefer' | 'ignore'
    active_data_workspace: Optional[dict] = None  # {workspace_id, table_id, schema, visible_rows, ...}
    data_workspace_preference: Optional[str] = None  # 'auto' | 'ignore'
    active_artifact: Optional[dict] = None  # {artifact_type, title, code, language} when chat artifact drawer is open
    base_checkpoint_id: Optional[str] = None  # Optional: start from this checkpoint to branch
    locked_agent: Optional[str] = None  # Optional: lock conversation routing to a specific agent
    agent_profile_id: Optional[str] = None  # Agent Factory: route to custom agent when set
    user_chat_model: Optional[str] = None  # Chat sidebar selected model
    is_branch_resend: bool = False
    branch_message_id: Optional[str] = None
    code_workspace_id: Optional[str] = None  # Bastion Code Space: inject settings.rules_text for custom agents


# Deprecated response models removed - no longer needed


# Deprecated endpoints removed:
# - POST /start - called broken Celery task
# - GET /status/{task_id} - only used for fallback path
# - POST /cancel/{task_id} - only used for fallback path
# - GET /queue/status - unused
# - GET /tasks/active - unused


@router.post("/api/async/orchestrator/stream")
async def stream_orchestrator_response(
    request: AsyncOrchestratorRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> StreamingResponse:
    """Stream ENHANCED orchestrator response with multi-operation state management"""
    try:
        logger.info(f"🌊 STREAMING ORCHESTRATOR: Starting for user {current_user.user_id}: {request.query[:100]}...")
        logger.info(f"🔍 REQUEST DEBUG: active_editor={request.active_editor is not None}, editor_preference={request.editor_preference}, locked_agent={request.locked_agent}")
        if request.active_editor:
            logger.info(f"🔍 ACTIVE EDITOR DETAILS: filename={request.active_editor.get('filename')}, is_editable={request.active_editor.get('is_editable')}, has_content={bool(request.active_editor.get('content'))}")
        
        # Always use gRPC orchestrator - backend orchestrator removed
        logger.info(f"🎯 ROUTING TO gRPC ORCHESTRATOR")
        # Forward to gRPC orchestrator microservice
        from api.grpc_orchestrator_proxy import stream_from_grpc_orchestrator
        
        # Build request context from frontend fields (active_editor, etc.)
        request_context = {
            "active_editor": request.active_editor,
            "editor_preference": request.editor_preference,
            "active_data_workspace": request.active_data_workspace,
            "data_workspace_preference": request.data_workspace_preference,
            "active_artifact": request.active_artifact,
            "pipeline_preference": None,  # Not in AsyncOrchestratorRequest
            "active_pipeline_id": None,  # Not in AsyncOrchestratorRequest
            "locked_agent": request.locked_agent,
            "base_checkpoint_id": request.base_checkpoint_id,
            "agent_profile_id": request.agent_profile_id,
            "user_chat_model": request.user_chat_model,
            "code_workspace_id": request.code_workspace_id,
            "persist_conversation": request.persist_conversation,
        }
        
        # Remove None values
        request_context = {k: v for k, v in request_context.items() if v is not None}

        stream_run_id = str(uuid.uuid4())
        request_context["client_run_id"] = stream_run_id

        # Log active_editor for debugging
        if request.active_editor:
            logger.info(f"📝 ACTIVE EDITOR: Passing to gRPC orchestrator (file={request.active_editor.get('filename', 'unknown')}, type={request.active_editor.get('frontmatter', {}).get('type', 'unknown')})")
        else:
            logger.info(f"📝 ACTIVE EDITOR: No active editor in request")
        
        # Conversation state is loaded by llm-orchestrator from LangGraph checkpoint
        return StreamingResponse(
            stream_from_grpc_orchestrator(
                query=request.query,
                conversation_id=request.conversation_id,
                user_id=current_user.user_id,
                session_id=request.session_id,
                request_context=request_context if request_context else None,
                state=None,
                is_branch_resend=bool(request.is_branch_resend),
                branch_message_id=request.branch_message_id,
                client_run_id=stream_run_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Run-Id": stream_run_id,
            }
        )
        
    except Exception as e:
        logger.error(f"❌ STREAM SETUP ERROR: {e}")
        error_data = safe_serialize_error(e, "Stream setup")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to setup streaming: {error_data['error_message']}"
        )


# POST /test endpoint removed - test endpoint for deprecated Celery tasks
