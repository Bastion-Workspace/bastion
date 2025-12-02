"""
LangGraph Native Streaming API - Roosevelt's "Best Practices" Implementation
Following official LangGraph streaming and HITL patterns
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from utils.auth_middleware import get_current_user, AuthenticatedUserResponse
from services.prompt_service import prompt_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/langgraph/stream", tags=["LangGraph Native Streaming"])


class LangGraphStreamRequest(BaseModel):
    """Request for LangGraph native streaming"""
    query: str
    conversation_id: str
    session_id: str = "default"


@router.post("/native")
async def stream_langgraph_native(
    request: LangGraphStreamRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> StreamingResponse:
    """
    ROOSEVELT'S LANGGRAPH NATIVE STREAMING
    Following official LangGraph best practices for streaming and HITL
    """
    try:
        logger.info(f"🌊 LANGGRAPH NATIVE: Starting stream for user {current_user.user_id}")
        
        async def generate_native_stream() -> AsyncGenerator[str, None]:
            try:
                # Get user settings for persona
                user_settings = await prompt_service.get_user_settings_for_service(current_user.user_id)
                persona = {
                    "ai_name": user_settings.ai_name if user_settings else "Kodex",
                    "persona_style": user_settings.persona_style.value if user_settings else "professional",
                    "political_bias": user_settings.political_bias.value if user_settings else "neutral"
                } if user_settings else None
                
                # DEPRECATED: Backend orchestrator removed
                raise HTTPException(
                    status_code=410,
                    detail="This endpoint is deprecated. Backend orchestrator has been removed. Use /api/async/orchestrator/stream instead."
                )
                
                # Get the compiled graph with checkpointer
                graph = orchestrator.graph
                
                # Prepare input for LangGraph
                initial_input = {
                    "messages": [{
                        "role": "user", 
                        "content": request.query,
                        "timestamp": datetime.now().isoformat()
                    }],
                    "user_id": current_user.user_id,
                    "conversation_id": request.conversation_id,
                    "persona": persona,
                    "shared_memory": {}
                }
                
                # Use namespaced thread_id for strict per-user isolation
                from services.orchestrator_utils import normalize_thread_id, validate_thread_id
                thread_id = normalize_thread_id(current_user.user_id, request.conversation_id)
                validate_thread_id(current_user.user_id, thread_id)
                config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "user_id": current_user.user_id
                    }
                }
                
                # LANGGRAPH NATIVE STREAMING: Use astream with multiple modes
                logger.info("🌊 LANGGRAPH NATIVE: Starting astream with values mode")
                
                try:
                    # Stream with 'values' mode for state updates
                    async for chunk in graph.astream(initial_input, config=config, stream_mode="values"):
                        # Send state updates
                        yield f"data: {json.dumps({
                            'type': 'state_update',
                            'chunk': str(chunk),
                            'timestamp': datetime.now().isoformat()
                        })}\n\n"
                        await asyncio.sleep(0.05)  # Small delay for UX
                        
                except Exception as interrupt_error:
                    # Check if this is a LangGraph interrupt (HITL)
                    if "interrupt" in str(interrupt_error).lower():
                        logger.info("🛑 LANGGRAPH INTERRUPT: HITL checkpoint reached")
                        
                        # Get the current state to extract permission message
                        try:
                            # Use get_state to retrieve current conversation state
                            state = await graph.aget_state(config)
                            
                            if state and state.values:
                                messages = state.values.get("messages", [])
                                
                                # Find the permission message
                                permission_message = None
                                for msg in reversed(messages):
                                    if (hasattr(msg, 'content') and 
                                        ('🔍 Web Search Permission Request' in str(msg.content) or 
                                         'web search' in str(msg.content).lower())):
                                        permission_message = str(msg.content)
                                        break
                                
                                if permission_message:
                                    # Stream the permission message
                                    yield f"data: {json.dumps({
                                        'type': 'permission_request',
                                        'content': permission_message,
                                        'requires_approval': True,
                                        'timestamp': datetime.now().isoformat()
                                    })}\n\n"
                                    
                                    # Stream completion with HITL status
                                    yield f"data: {json.dumps({
                                        'type': 'complete_hitl',
                                        'status': 'awaiting_permission',
                                        'conversation_id': request.conversation_id,
                                        'thread_id': request.conversation_id,
                                        'timestamp': datetime.now().isoformat()
                                    })}\n\n"
                                    
                                    logger.info("✅ LANGGRAPH HITL: Permission request streamed successfully")
                                    return
                        
                        except Exception as state_error:
                            logger.error(f"❌ LANGGRAPH STATE ERROR: {state_error}")
                    
                    # Re-raise if not a handled interrupt
                    raise interrupt_error
                
                # If we reach here, the graph completed without interruption
                # Stream normal completion
                yield f"data: {json.dumps({
                    'type': 'complete',
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                })}\n\n"
                
            except Exception as e:
                logger.error(f"❌ LANGGRAPH NATIVE STREAMING ERROR: {e}")
                yield f"data: {json.dumps({
                    'type': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })}\n\n"
        
        return StreamingResponse(
            generate_native_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
        
    except Exception as e:
        logger.error(f"❌ LANGGRAPH NATIVE SETUP ERROR: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to setup LangGraph native streaming: {str(e)}"
        )


@router.post("/resume")
async def resume_langgraph_hitl(
    request: LangGraphStreamRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> StreamingResponse:
    """
    ROOSEVELT'S LANGGRAPH NATIVE RESUME
    Resume interrupted conversation using LangGraph Command primitive
    """
    try:
        logger.info(f"🔄 LANGGRAPH RESUME: Continuing conversation {request.conversation_id}")
        
        async def generate_resume_stream() -> AsyncGenerator[str, None]:
            try:
                # DEPRECATED: Backend orchestrator removed
                raise HTTPException(
                    status_code=410,
                    detail="This endpoint is deprecated. Backend orchestrator has been removed. Use /api/async/orchestrator/stream instead."
                )
                graph = orchestrator.graph
                
                config = {
                    "configurable": {
                        "thread_id": request.conversation_id,
                        "user_id": current_user.user_id
                    }
                }
                
                # LANGGRAPH NATIVE RESUME: Use Command primitive (if available) or direct input
                resume_input = {
                    "messages": [{
                        "role": "user",
                        "content": request.query,  # User's permission response (yes/no)
                        "timestamp": datetime.now().isoformat()
                    }]
                }
                
                # Stream the resumed execution
                async for chunk in graph.astream(resume_input, config=config, stream_mode="values"):
                    yield f"data: {json.dumps({
                        'type': 'resume_update',
                        'chunk': str(chunk),
                        'timestamp': datetime.now().isoformat()
                    })}\n\n"
                    await asyncio.sleep(0.05)
                
                # Stream completion
                yield f"data: {json.dumps({
                    'type': 'complete',
                    'status': 'resumed_success',
                    'timestamp': datetime.now().isoformat()
                })}\n\n"
                
            except Exception as e:
                logger.error(f"❌ LANGGRAPH RESUME ERROR: {e}")
                yield f"data: {json.dumps({
                    'type': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })}\n\n"
        
        return StreamingResponse(
            generate_resume_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
        
    except Exception as e:
        logger.error(f"❌ LANGGRAPH RESUME SETUP ERROR: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to setup LangGraph resume: {str(e)}"
        )
