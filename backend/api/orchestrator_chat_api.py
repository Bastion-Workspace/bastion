"""
Orchestrator Chat API
Advanced agentic chat using the "Big Stick" Orchestrator pattern
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from services.langgraph_official_orchestrator import get_official_orchestrator
from services.prompt_service import prompt_service
from services.conversation_service import ConversationService
from utils.auth_middleware import get_current_user, AuthenticatedUserResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/orchestrator", tags=["Orchestrator Chat"])


class OrchestratorChatRequest(BaseModel):
    """Chat request for Orchestrator system"""
    query: str
    conversation_id: str
    session_id: str = "default"
    active_editor: Optional[dict] = None
    editor_preference: Optional[str] = None  # 'prefer' | 'ignore'


class OrchestratorChatResponse(BaseModel):
    """Chat response from Orchestrator system"""
    success: bool
    answer: str
    conversation_id: str
    message_id: Optional[str] = None
    execution_mode: str
    delegated_agent: str
    orchestrator_decision: dict
    processing_time: float
    error: Optional[str] = None


@router.post("/chat", response_model=OrchestratorChatResponse)
async def orchestrator_chat(
    request: OrchestratorChatRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> OrchestratorChatResponse:
    """Process a chat message through the Orchestrator system"""
    start_time = datetime.now()
    
    try:
        logger.info(f"🎯 ORCHESTRATOR CHAT: Request from user {current_user.user_id}: {request.query[:100]}...")
        
        # Get user settings to fetch persona
        user_settings = await prompt_service.get_user_settings_for_service(current_user.user_id)
        persona = {
            "ai_name": user_settings.ai_name if user_settings else "Kodex",
            "persona_style": user_settings.persona_style.value if user_settings else "professional",
            "political_bias": user_settings.political_bias.value if user_settings else "neutral"
        } if user_settings else None
        
        # Store user message BEFORE processing
        conversation_service = ConversationService()
        conversation_service.set_current_user(current_user.user_id)
        
        # Add user message to conversation first
        user_message_result = await conversation_service.add_message(
            conversation_id=request.conversation_id,
            user_id=current_user.user_id,
            role="user",
            content=request.query,
            metadata={"orchestrator_system": True}
        )
        
        logger.info(f"✅ Stored user message before Orchestrator processing")
        
        # Process through Orchestrator LangGraph
        orchestrator = await get_official_orchestrator()

        # Respect editor preference: if ignore, drop active_editor
        if (request.editor_preference or '').lower() == 'ignore':
            request.active_editor = None

        # Validate active editor (hard gate: .md + frontmatter.type=fiction)
        validated_active_editor = None
        try:
            ae = request.active_editor or None
            if isinstance(ae, dict) and ae.get('is_editable') and isinstance(ae.get('filename'), str):
                fname = ae['filename'].lower()
                fm_type = ((ae.get('frontmatter') or {}).get('type') or '').lower()
                if fname.endswith('.md') and fm_type == 'fiction':
                    validated_active_editor = {
                        'is_editable': True,
                        'filename': ae.get('filename'),
                        'language': ae.get('language') or 'markdown',
                        'content': ae.get('content') or '',
                        'content_length': ae.get('content_length') or 0,
                        'frontmatter': ae.get('frontmatter') or {}
                    }
        except Exception:
            validated_active_editor = None

        # Process through orchestrator
        result = await orchestrator.process_user_query(
            user_message=request.query,
            user_id=current_user.user_id,
            conversation_id=request.conversation_id,
            persona=persona,
            extra_shared_memory={"active_editor": validated_active_editor} if validated_active_editor else None
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if result.get("success"):
            # Save the assistant response to conversation
            message_result = await conversation_service.add_message(
                conversation_id=request.conversation_id,
                user_id=current_user.user_id,
                role="assistant",
                content=result.get("response", "No response generated"),
                metadata={
                    "orchestrator_system": True,
                    "delegated_agent": result.get("delegated_agent", "unknown"),
                    "orchestrator_decision": result.get("orchestrator_decision", {}),
                    "processing_time": processing_time
                }
            )
            
            logger.info(f"✅ ORCHESTRATOR CHAT: Completed successfully for conversation {request.conversation_id}")
            
            return OrchestratorChatResponse(
                success=True,
                answer=result.get("response", "No response generated"),
                conversation_id=request.conversation_id,
                message_id=message_result.get("message_id") if message_result else None,
                execution_mode="orchestrator",
                delegated_agent=result.get("delegated_agent", "unknown"),
                orchestrator_decision=result.get("orchestrator_decision", {}),
                processing_time=processing_time,
                error=None
            )
        else:
            error_msg = result.get("error", "Orchestrator processing failed")
            logger.error(f"❌ ORCHESTRATOR CHAT failed: {error_msg}")
            
            return OrchestratorChatResponse(
                success=False,
                answer=f"❌ Orchestrator Error: {error_msg}",
                conversation_id=request.conversation_id,
                message_id=None,
                execution_mode="orchestrator",
                delegated_agent="error",
                orchestrator_decision={"error": error_msg},
                processing_time=processing_time,
                error=error_msg
            )
            
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ ORCHESTRATOR CHAT error: {e}")
        raise HTTPException(status_code=500, detail=f"Orchestrator chat failed: {str(e)}")


@router.get("/status")
async def orchestrator_status():
    """Get orchestrator system status"""
    try:
        orchestrator = await get_official_orchestrator()
        
        return {
            "status": "active",
            "orchestrator_initialized": orchestrator.is_initialized,
            "available_agents": [
                "chat", "research"
            ],
            "architecture": "big_stick_orchestrator",
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"❌ Orchestrator status error: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")
