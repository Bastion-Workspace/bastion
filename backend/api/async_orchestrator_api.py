"""
Enhanced Async Orchestrator API
Background processing endpoints for Roosevelt's "Big Stick" Enhanced Orchestrator
Now with multi-operation state management and context-aware routing!
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.celery_app import celery_app, TaskStatus
from services.celery_utils import safe_serialize_error, make_json_safe
from services.prompt_service import prompt_service
from utils.auth_middleware import get_current_user, AuthenticatedUserResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/async/orchestrator", tags=["Async Orchestrator"])


class AsyncOrchestratorRequest(BaseModel):
    """Request to start async orchestrator processing"""
    query: str
    conversation_id: str
    session_id: str = "default"
    priority: str = "normal"  # normal, high, low
    active_editor: Optional[dict] = None  # {is_editable, filename, language, content, content_length, frontmatter}
    editor_preference: Optional[str] = None  # 'prefer' | 'ignore'
    base_checkpoint_id: Optional[str] = None  # Optional: start from this checkpoint to branch
    locked_agent: Optional[str] = None  # Optional: lock conversation routing to a specific agent


class AsyncTaskResponse(BaseModel):
    """Response when starting an async task"""
    success: bool
    task_id: str
    status: str
    message: str
    estimated_completion: Optional[str] = None
    conversation_id: str


class TaskStatusResponse(BaseModel):
    """Response for task status queries"""
    task_id: str
    status: str
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@router.post("/start", response_model=AsyncTaskResponse)
async def start_async_orchestrator_task(
    request: AsyncOrchestratorRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> AsyncTaskResponse:
    """Start an async orchestrator task in the background"""
    try:
        logger.info(f"🚀 ASYNC START: Orchestrator task for user {current_user.user_id}: {request.query[:100]}...")
        
        # Get user settings for persona
        user_settings = await prompt_service.get_user_settings_for_service(current_user.user_id)
        persona = {
            "ai_name": user_settings.ai_name if user_settings else "Kodex",
            "persona_style": user_settings.persona_style.value if user_settings else "professional",
            "political_bias": user_settings.political_bias.value if user_settings else "neutral"
        } if user_settings else None
        
        # Start the background task
        task = celery_app.send_task(
            "orchestrator.process_query",
            kwargs={
                "user_id": current_user.user_id,
                "conversation_id": request.conversation_id,
                "query": request.query,
                "persona": persona,
                "base_checkpoint_id": request.base_checkpoint_id
            }
        )
        
        logger.info(f"✅ ASYNC TASK STARTED: {task.id}")
        
        return AsyncTaskResponse(
            success=True,
            task_id=task.id,
            status=TaskStatus.PENDING,
            message="Orchestrator task started successfully",
            estimated_completion="2-5 minutes",
            conversation_id=request.conversation_id
        )
        
    except Exception as e:
        logger.error(f"❌ ASYNC START ERROR: {e}")
        error_data = safe_serialize_error(e, "Async start")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to start async orchestrator task: {error_data['error_message']}"
        )


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_async_task_status(
    task_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> TaskStatusResponse:
    """Get the status of an async orchestrator task"""
    try:
        logger.info(f"📊 STATUS CHECK: Task {task_id} for user {current_user.user_id}")
        
        # Get task result from Celery with safe handling
        result = celery_app.AsyncResult(task_id)
        
        # Try to get result from our custom Redis storage first
        safe_result_data = None
        safe_error_message = None
        
        try:
            # Check our custom Redis storage for the actual result
            import redis.asyncio as redis
            import os
            import json
            
            redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
            redis_client = redis.from_url(redis_url)
            
            stored_result = await redis_client.get(f"orchestrator_result:{task_id}")
            await redis_client.close()
            
            if stored_result:
                safe_result_data = json.loads(stored_result.decode())
                logger.info(f"✅ Retrieved result from Redis for task {task_id}")
            elif result.result:
                # Fallback to Celery result if Redis doesn't have it
                safe_result_data = make_json_safe(result.result)
                logger.info(f"🔄 Using Celery result for task {task_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to get task result: {e}")
            safe_result_data = {"error": "Result data could not be retrieved"}
        
        # Build response based on task state
        response_data = {
            "task_id": task_id,
            "status": result.status,
        }
        
        if result.status == TaskStatus.PENDING:
            response_data.update({
                "progress": {"message": "Task is queued and waiting to start", "percentage": 0},
                "result": None,
                "error": None
            })
        elif result.status == TaskStatus.PROGRESS:
            # Safely handle progress data
            progress_data = safe_result_data if safe_result_data else {"message": "Processing...", "percentage": 50}
            response_data.update({
                "progress": progress_data,
                "result": None,
                "error": None
            })
        elif result.status == TaskStatus.SUCCESS:
            response_data.update({
                "progress": {"message": "Task completed successfully", "percentage": 100},
                "result": safe_result_data,
                "error": None,
                "completed_at": datetime.now().isoformat()
            })
        elif result.status == TaskStatus.FAILURE:
            # Safely extract error information
            if safe_result_data:
                safe_error_message = safe_result_data.get("error", "Unknown error") if isinstance(safe_result_data, dict) else str(safe_result_data)
            else:
                safe_error_message = "Task failed with unknown error"
            
            response_data.update({
                "progress": {"message": "Task failed", "percentage": 0},
                "result": None,
                "error": safe_error_message[:1000]  # Limit error message length
            })
        else:
            response_data.update({
                "progress": {"message": f"Task status: {result.status}", "percentage": 0},
                "result": safe_result_data,
                "error": None
            })
        
        return TaskStatusResponse(**response_data)
        
    except Exception as e:
        logger.error(f"❌ STATUS CHECK ERROR: {e}")
        
        # Create safe error response
        error_data = safe_serialize_error(e, "Status check")
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task status: {error_data['error_message']}"
        )


@router.post("/cancel/{task_id}")
async def cancel_async_task(
    task_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Cancel a running async orchestrator task"""
    try:
        logger.info(f"🛑 CANCEL TASK: {task_id} for user {current_user.user_id}")
        
        # Cancel the task
        celery_app.control.revoke(task_id, terminate=True)
        
        return {
            "success": True,
            "task_id": task_id,
            "status": "CANCELLED",
            "message": "Task cancelled successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ CANCEL ERROR: {e}")
        error_data = safe_serialize_error(e, "Cancel task")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel task: {error_data['error_message']}"
        )


@router.get("/queue/status")
async def get_queue_status(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get Celery queue status and worker information"""
    try:
        # Get active tasks
        active_tasks = celery_app.control.inspect().active()
        
        # Get queue lengths (simplified)
        stats = celery_app.control.inspect().stats()
        
        # Get worker status
        workers = celery_app.control.inspect().ping()
        
        return {
            "success": True,
            "workers_online": len(workers) if workers else 0,
            "active_tasks": active_tasks,
            "queue_stats": stats,
            "worker_list": list(workers.keys()) if workers else [],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ QUEUE STATUS ERROR: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Could not retrieve queue status"
        }


@router.get("/tasks/active")
async def get_active_tasks(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get list of currently active tasks"""
    try:
        # Get active tasks from Celery
        inspect = celery_app.control.inspect()
        active = inspect.active()
        scheduled = inspect.scheduled()
        reserved = inspect.reserved()
        
        return {
            "success": True,
            "active_tasks": active or {},
            "scheduled_tasks": scheduled or {},
            "reserved_tasks": reserved or {},
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ ACTIVE TASKS ERROR: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Could not retrieve active tasks"
        }


@router.post("/stream")
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
            "pipeline_preference": None,  # Not in AsyncOrchestratorRequest
            "active_pipeline_id": None,  # Not in AsyncOrchestratorRequest
            "locked_agent": request.locked_agent,
            "base_checkpoint_id": request.base_checkpoint_id
        }
        
        # Remove None values
        request_context = {k: v for k, v in request_context.items() if v is not None}
        
        # Log active_editor for debugging
        if request.active_editor:
            logger.info(f"📝 ACTIVE EDITOR: Passing to gRPC orchestrator (file={request.active_editor.get('filename', 'unknown')}, type={request.active_editor.get('frontmatter', {}).get('type', 'unknown')})")
        else:
            logger.info(f"📝 ACTIVE EDITOR: No active editor in request")
        
        # Load conversation state from database for continuity
        from api.grpc_orchestrator_proxy import _load_conversation_state
        conversation_state = await _load_conversation_state(current_user.user_id, request.conversation_id)
        
        return StreamingResponse(
            stream_from_grpc_orchestrator(
                query=request.query,
                conversation_id=request.conversation_id,
                user_id=current_user.user_id,
                session_id=request.session_id,
                request_context=request_context if request_context else None,
                state=conversation_state
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ STREAM SETUP ERROR: {e}")
        error_data = safe_serialize_error(e, "Stream setup")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to setup streaming: {error_data['error_message']}"
        )


@router.post("/test")
async def test_celery_connection(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Test Celery connection and Redis backend"""
    try:
        # Test simple task
        from services.celery_tasks.orchestrator_tasks import get_task_status
        
        # Create a test task
        test_task = celery_app.send_task(
            "orchestrator.get_task_status",
            args=["test_task_id"]
        )
        
        return {
            "success": True,
            "test_task_id": test_task.id,
            "message": "Celery connection test successful",
            "broker_url": celery_app.conf.broker_url,
            "result_backend": celery_app.conf.result_backend
        }
        
    except Exception as e:
        logger.error(f"❌ CELERY TEST ERROR: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Celery connection test failed"
        }
