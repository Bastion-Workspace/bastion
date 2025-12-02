"""
Template Execution API - Roosevelt's "Execute Template Research Plans"
Handles template-based research plan execution with LangGraph integration
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

# DEPRECATED: Backend orchestrator removed
# from services.langgraph_official_orchestrator import LangGraphOfficialOrchestrator
from utils.auth_middleware import get_current_user
from models.api_models import AuthenticatedUserResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/template-execution", tags=["Template Execution"])


# ===== REQUEST/RESPONSE MODELS =====

class TemplateExecutionRequest(BaseModel):
    """Request model for executing a template-based research plan"""
    conversation_id: str = Field(description="Conversation ID for context")
    template_id: str = Field(description="Template to use for research")
    query: str = Field(description="Original research query")
    custom_instructions: Optional[str] = Field(None, description="Additional user instructions")
    priority_sections: Optional[list] = Field(None, description="Sections to prioritize")


class TemplateConfirmationRequest(BaseModel):
    """Request model for confirming template usage"""
    conversation_id: str = Field(description="Conversation ID")
    template_id: str = Field(description="Template to confirm")
    action: str = Field(description="User action: accept, decline, modify")
    modifications: Optional[str] = Field(None, description="User modifications if action is modify")


class ExecutionResponse(BaseModel):
    """Response model for template execution"""
    success: bool
    message: str
    execution_id: Optional[str] = None
    status: str
    result: Optional[Dict[str, Any]] = None


# ===== TEMPLATE EXECUTION ENDPOINTS =====

@router.post("/execute", response_model=ExecutionResponse)
async def execute_template_research(
    request: TemplateExecutionRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Execute a research plan using a specific template"""
    try:
        logger.info(f"🎯 Executing template research: {request.template_id} for user {current_user.username}")
        
        # Initialize orchestrator
        # DEPRECATED: Backend orchestrator removed
        raise HTTPException(
            status_code=410,
            detail="This endpoint is deprecated. Backend orchestrator has been removed."
        )
        
        # Create execution query with template specification
        execution_query = f"EXECUTE_TEMPLATE:{request.template_id}|{request.query}"
        if request.custom_instructions:
            execution_query += f"|INSTRUCTIONS:{request.custom_instructions}"
        
        # Process through LangGraph with template context
        result = await orchestrator.process_user_query(
            query=execution_query,
            conversation_id=request.conversation_id,
            user_id=current_user.user_id
        )
        
        return ExecutionResponse(
            success=True,
            message="Template research execution started",
            execution_id=request.conversation_id,
            status="processing",
            result=result
        )
        
    except Exception as e:
        logger.error(f"❌ Template execution failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Template execution failed: {str(e)}"
        )


@router.post("/confirm", response_model=ExecutionResponse)
async def confirm_template_usage(
    request: TemplateConfirmationRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Handle user confirmation of template usage"""
    try:
        logger.info(f"✅ Template confirmation: {request.action} for template {request.template_id}")
        
        # Initialize orchestrator
        # DEPRECATED: Backend orchestrator removed
        raise HTTPException(
            status_code=410,
            detail="This endpoint is deprecated. Backend orchestrator has been removed."
        )
        
        # Create confirmation message for LangGraph
        if request.action == "accept":
            confirmation_query = f"TEMPLATE_CONFIRMED:accept|{request.template_id}"
        elif request.action == "decline":
            confirmation_query = "TEMPLATE_CONFIRMED:decline|general_research"
        elif request.action == "modify":
            confirmation_query = f"TEMPLATE_CONFIRMED:modify|{request.template_id}|{request.modifications or ''}"
        else:
            raise ValueError(f"Invalid action: {request.action}")
        
        # Send confirmation to LangGraph
        result = await orchestrator.continue_conversation(
            query=confirmation_query,
            conversation_id=request.conversation_id,
            user_id=current_user.user_id
        )
        
        return ExecutionResponse(
            success=True,
            message=f"Template {request.action}ed successfully",
            execution_id=request.conversation_id,
            status="confirmed",
            result=result
        )
        
    except Exception as e:
        logger.error(f"❌ Template confirmation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Template confirmation failed: {str(e)}"
        )


@router.get("/status/{conversation_id}")
async def get_execution_status(
    conversation_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get the status of a template execution"""
    try:
        logger.info(f"📊 Getting execution status for conversation: {conversation_id}")
        
        # Initialize orchestrator to check status
        # DEPRECATED: Backend orchestrator removed
        raise HTTPException(
            status_code=410,
            detail="This endpoint is deprecated. Backend orchestrator has been removed."
        )
        
        # Get conversation state from LangGraph checkpoint
        state = await orchestrator.get_conversation_state(conversation_id)
        
        if not state:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Extract execution status from state
        agent_results = state.get("agent_results", {})
        template_info = agent_results.get("template_info", {})
        
        return ExecutionResponse(
            success=True,
            message="Status retrieved successfully",
            execution_id=conversation_id,
            status=agent_results.get("status", "unknown"),
            result={
                "template_used": template_info.get("template_id"),
                "sections_completed": template_info.get("sections_completed", []),
                "progress": agent_results.get("progress", 0),
                "current_step": agent_results.get("current_step", "unknown")
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get execution status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get execution status: {str(e)}"
        )


# ===== HELPER ENDPOINTS =====

@router.get("/plan/{template_id}")
async def generate_template_plan(
    template_id: str,
    query: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Generate a research plan preview for a template"""
    try:
        logger.info(f"📋 Generating plan preview for template: {template_id}")
        
        from services.template_service import template_service
        
        # Get template
        template = await template_service.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        # Generate plan preview
        plan_steps = []
        for section in template.sections:
            step = {
                "section_name": section.section_name,
                "description": section.description,
                "fields": [field.field_name for field in section.fields],
                "required": section.required
            }
            plan_steps.append(step)
        
        return {
            "success": True,
            "template_name": template.template_name,
            "query": query,
            "plan_steps": plan_steps,
            "estimated_time": f"{len(plan_steps) * 2-3} minutes",
            "sections_count": len(plan_steps)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to generate template plan: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate template plan: {str(e)}"
        )

