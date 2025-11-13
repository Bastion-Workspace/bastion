"""
Orchestrator Celery Tasks
Background processing for the "Big Stick" Orchestrator
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
import json

from celery import current_task
from services.celery_app import celery_app, update_task_progress, TaskStatus
from services.celery_utils import (
    safe_serialize_error, 
    safe_update_task_state, 
    clean_result_for_storage,
    create_progress_meta,
    safe_task_wrapper
)
from services.langgraph_official_orchestrator import get_official_orchestrator
from services.conversation_service import ConversationService
from services.prompt_service import PromptService

logger = logging.getLogger(__name__)


async def _store_task_result_in_redis(task_id: str, result: Dict[str, Any]):
    """Store task result manually in Redis to avoid Celery serialization issues"""
    try:
        import redis.asyncio as redis
        import os
        
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        redis_client = redis.from_url(redis_url)
        
        # Store with expiration (1 hour)
        await redis_client.setex(
            f"orchestrator_result:{task_id}",
            3600,  # 1 hour
            json.dumps(result)
        )
        
        await redis_client.close()
        logger.info(f"✅ Stored task result in Redis: {task_id}")
        
    except Exception as e:
        logger.error(f"❌ Failed to store task result in Redis: {e}")


@celery_app.task(bind=True, name="orchestrator.process_query")
def process_orchestrator_query(
    self,
    user_id: str,
    conversation_id: str,
    query: str,
    persona: Optional[Dict[str, Any]] = None,
    base_checkpoint_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Background task for processing orchestrator queries
    This is the main async entry point for the "Big Stick" system
    """
    try:
        logger.info(f"🎯 ASYNC ORCHESTRATOR: Starting background processing for user {user_id}")
        
        # Update progress - Initialization
        update_task_progress(self, 1, 5, "Initializing orchestrator system...")
        
        # Run the async orchestrator processing
        result = asyncio.run(_async_process_orchestrator_query(
            self, user_id, conversation_id, query, persona, base_checkpoint_id
        ))
        
        logger.info(f"✅ ASYNC ORCHESTRATOR: Completed successfully")
        
        # Store result manually in Redis to avoid Celery serialization issues
        cleaned_result = clean_result_for_storage(result)
        asyncio.run(_store_task_result_in_redis(self.request.id, cleaned_result))
        
        # Return minimal result to avoid Celery backend issues
        return {
            "success": True,
            "task_id": self.request.id,
            "timestamp": datetime.now().isoformat(),
            "stored_in_redis": True
        }
        
    except Exception as e:
        logger.error(f"❌ ASYNC ORCHESTRATOR ERROR: {e}")
        
        # Safe error serialization
        error_data = safe_serialize_error(e, "Orchestrator processing")
        
        # Update task state safely
        safe_update_task_state(
            self,
            TaskStatus.FAILURE,
            {
                "error": error_data["error_message"],
                "error_type": error_data["error_type"],
                "message": "Orchestrator processing failed",
                "timestamp": error_data["timestamp"]
            }
        )
        
        return clean_result_for_storage({
            "success": False,
            "error": error_data["error_message"],
            "error_type": error_data["error_type"],
            "message": "Background orchestrator processing failed",
            "timestamp": error_data["timestamp"]
        })


async def _async_process_orchestrator_query(
    task,
    user_id: str,
    conversation_id: str, 
    query: str,
    persona: Optional[Dict[str, Any]] = None,
    base_checkpoint_id: Optional[str] = None
) -> Dict[str, Any]:
    """Internal async function for orchestrator processing"""
    # Update progress - Loading context
    update_task_progress(task, 2, 5, "Loading conversation context and user settings...")
    
    # Get user settings for persona if not provided
    if not persona:
        try:
            user_settings = await PromptService.get_user_settings_for_service(user_id)
            persona = {
                "ai_name": user_settings.ai_name if user_settings else "Kodex",
                "persona_style": user_settings.persona_style.value if user_settings else "professional",
                "political_bias": user_settings.political_bias.value if user_settings else "neutral"
            } if user_settings else None
        except Exception as e:
            logger.warning(f"⚠️ Could not load user settings: {e}")
            persona = None
    
    # Store user message in conversation BEFORE processing
    conversation_service = ConversationService()
    conversation_service.set_current_user(user_id)
    
    user_message_result = await conversation_service.add_message(
        conversation_id=conversation_id,
        user_id=user_id,
        role="user",
        content=query,
        metadata={
            "async_orchestrator": True,
            "task_id": task.request.id,
            "background_processing": True
        }
    )
    
    # Update progress - Processing with orchestrator
    update_task_progress(task, 3, 5, "Orchestrator analyzing request and delegating to agents...")
    
    # Process through official orchestrator
    orchestrator = await get_official_orchestrator()
    # Process through official LangGraph orchestrator
    result = await orchestrator.process_user_query(
        user_message=query,
        user_id=user_id,
        conversation_id=conversation_id,
        persona=persona,
        base_checkpoint_id=base_checkpoint_id
    )
    final_result = result.get("final_state", {})
    
    # Update progress - Saving results
    update_task_progress(task, 4, 5, "Saving orchestrator results...")
    
    if final_result and final_result.get("messages"):
        # Extract final assistant message
        messages = final_result.get("messages", [])
        last_message = messages[-1] if messages else None
        response_content = last_message.content if last_message and hasattr(last_message, 'content') else "No response generated"
        
        # Save assistant response to conversation
        message_result = await conversation_service.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role="assistant",
            content=response_content,
            metadata={
                "async_orchestrator": True,
                "task_id": task.request.id,
                "delegated_agent": final_result.get("active_agent", "unknown"),
                "background_processing": True
            }
        )
        
        # Update progress - Complete
        update_task_progress(task, 5, 5, "Processing completed successfully!")
        
        # Final success state with safe serialization
        success_meta = clean_result_for_storage({
            "result": {
                "response": response_content,
                "delegated_agent": final_result.get("active_agent", "unknown"),
                "success": True
            },
            "message": "Orchestrator processing completed successfully",
            "delegated_agent": final_result.get("active_agent", "unknown"),
            "message_id": message_result.get("message_id") if message_result else None,
            "timestamp": datetime.now().isoformat()
        })
        
        safe_update_task_state(task, TaskStatus.SUCCESS, success_meta)
        
        return clean_result_for_storage({
            "success": True,
            "response": response_content,
            "delegated_agent": final_result.get("active_agent", "unknown"),
            "message_id": message_result.get("message_id") if message_result else None,
            "task_id": task.request.id
        })
    else:
        # Handle orchestrator failure
        error_msg = "Orchestrator failed to process query or returned no results"
        
        safe_update_task_state(
            task,
            TaskStatus.FAILURE,
            {
                "error": str(error_msg)[:1000],  # Limit error message length
                "message": "Orchestrator processing failed",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return clean_result_for_storage({
            "success": False,
            "error": str(error_msg),
            "message": "Orchestrator processing failed",
            "timestamp": datetime.now().isoformat()
        })


@celery_app.task(bind=True, name="orchestrator.get_task_status")
def get_task_status(self, task_id: str) -> Dict[str, Any]:
    """Get the status of an orchestrator task"""
    try:
        # Get task result
        result = celery_app.AsyncResult(task_id)
        
        return clean_result_for_storage({
            "task_id": task_id,
            "status": result.status,
            "result": result.result,
            "ready": result.ready(),
            "successful": result.successful() if result.ready() else None,
            "failed": result.failed() if result.ready() else None,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting task status: {e}")
        error_data = safe_serialize_error(e, "Get task status")
        return clean_result_for_storage({
            "task_id": task_id,
            "status": "ERROR", 
            "error": error_data["error_message"],
            "error_type": error_data["error_type"],
            "timestamp": error_data["timestamp"]
        })


@celery_app.task(bind=True, name="orchestrator.cancel_task")
def cancel_orchestrator_task(self, task_id: str) -> Dict[str, Any]:
    """Cancel a running orchestrator task"""
    try:
        # Revoke the task
        celery_app.control.revoke(task_id, terminate=True)
        
        logger.info(f"🛑 TASK CANCELLED: {task_id}")
        
        return clean_result_for_storage({
            "task_id": task_id,
            "status": "CANCELLED",
            "message": "Task successfully cancelled",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error cancelling task: {e}")
        error_data = safe_serialize_error(e, "Cancel task")
        return clean_result_for_storage({
            "task_id": task_id,
            "status": "ERROR",
            "error": error_data["error_message"],
            "error_type": error_data["error_type"],
            "message": "Failed to cancel task",
            "timestamp": error_data["timestamp"]
        })
