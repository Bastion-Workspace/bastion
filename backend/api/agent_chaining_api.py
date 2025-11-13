"""
Agent Chaining API
API endpoints for testing and using agent chaining workflows
Theodore Roosevelt's "Square Deal" for agent coordination!
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from services.langgraph_agents.agent_workflow_engine import get_workflow_engine
from services.langgraph_agents.agent_chain_memory import get_agent_chain_memory
from utils.auth_middleware import get_current_user
from models.api_models import AuthenticatedUserResponse

logger = logging.getLogger(__name__)

router = APIRouter()


class ChainRequest(BaseModel):
    agents: list[str]
    query: str
    user_id: str
    conversation_id: Optional[str] = None
    persona: Optional[Dict[str, Any]] = None


class WorkflowRequest(BaseModel):
    workflow_template: str
    query: str
    user_id: str
    conversation_id: Optional[str] = None
    persona: Optional[Dict[str, Any]] = None
    customizations: Optional[Dict[str, Any]] = None


@router.post("/chain/execute")
async def execute_agent_chain(request: ChainRequest, current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Execute a simple agent chain"""
    try:
        logger.info(f"🔗 Executing agent chain: {' → '.join(request.agents)}")
        
        workflow_engine = get_workflow_engine()
        
        user_context = {
            "user_id": request.user_id,
            "conversation_id": request.conversation_id,
            "query": request.query,
            "messages": [],
            "persona": request.persona
        }
        
        # Execute chain and collect all results
        chain_updates = []
        final_result = None
        
        async for update in workflow_engine.execute_simple_chain(
            agents=request.agents,
            user_context=user_context
        ):
            chain_updates.append(update)
            if update.get("type") == "chain_completed":
                final_result = update.get("final_response")
        
        return {
            "success": True,
            "chain_id": chain_updates[0].get("chain_id") if chain_updates else None,
            "agents": request.agents,
            "final_response": final_result,
            "chain_updates": chain_updates[-5:] if len(chain_updates) > 5 else chain_updates,  # Last 5 updates
            "total_updates": len(chain_updates),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Agent chain execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chain execution failed: {str(e)}")


@router.get("/chain/{chain_id}/stream")
async def stream_agent_chain(
    agents: str,  # Comma-separated agent names
    query: str,
    user_id: str,
    chain_id: str,
    conversation_id: Optional[str] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Stream agent chain execution in real-time"""
    try:
        logger.info(f"🌊 Streaming agent chain: {chain_id}")
        
        agent_list = [agent.strip() for agent in agents.split(",")]
        workflow_engine = get_workflow_engine()
        
        user_context = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query": query,
            "messages": [],
            "persona": None
        }
        
        async def stream_generator():
            try:
                async for update in workflow_engine.execute_simple_chain(
                    agents=agent_list,
                    user_context=user_context,
                    chain_id=chain_id
                ):
                    # Stream each update as JSON
                    yield f"data: {json.dumps(update)}\n\n"
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'stream_complete', 'timestamp': datetime.now().isoformat()})}\n\n"
                
            except Exception as e:
                error_update = {
                    "type": "stream_error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(error_update)}\n\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Chain streaming failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chain streaming failed: {str(e)}")


@router.post("/workflow/execute")
async def execute_workflow(request: WorkflowRequest, current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Execute a complex workflow using templates"""
    try:
        logger.info(f"🏭 Executing workflow: {request.workflow_template}")
        
        workflow_engine = get_workflow_engine()
        
        user_context = {
            "user_id": request.user_id,
            "conversation_id": request.conversation_id,
            "original_query": request.query,
            "messages": [],
            "persona": request.persona
        }
        
        # Execute workflow and collect all results
        workflow_updates = []
        final_result = None
        workflow_id = None
        
        async for update in workflow_engine.execute_workflow(
            workflow_template=request.workflow_template,
            user_context=user_context
        ):
            workflow_updates.append(update)
            
            if update.get("type") == "workflow_started":
                workflow_id = update.get("workflow_id")
            elif update.get("type") == "workflow_completed":
                final_status = update.get("final_status")
                if final_status:
                    # Try to get final response from the workflow
                    chain_memory = get_agent_chain_memory()
                    if workflow_id and workflow_id in chain_memory.active_workflows:
                        workflow_data = chain_memory.active_workflows[workflow_id]
                        completed_steps = [s for s in workflow_data["steps"] if s.get("status") == "completed"]
                        if completed_steps:
                            last_step = completed_steps[-1]
                            final_result = last_step.get("result", {}).get("response", "Workflow completed.")
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "template": request.workflow_template,
            "final_response": final_result,
            "workflow_updates": workflow_updates[-5:] if len(workflow_updates) > 5 else workflow_updates,
            "total_updates": len(workflow_updates),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")


@router.get("/workflow/{workflow_id}/stream")
async def stream_workflow(
    workflow_template: str,
    query: str,
    user_id: str,
    workflow_id: str,
    conversation_id: Optional[str] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Stream workflow execution in real-time"""
    try:
        logger.info(f"🌊 Streaming workflow: {workflow_id}")
        
        workflow_engine = get_workflow_engine()
        
        user_context = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "original_query": query,
            "messages": [],
            "persona": None
        }
        
        async def stream_generator():
            try:
                async for update in workflow_engine.execute_workflow(
                    workflow_template=workflow_template,
                    user_context=user_context,
                    workflow_id=workflow_id
                ):
                    # Stream each update as JSON
                    yield f"data: {json.dumps(update)}\n\n"
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'stream_complete', 'timestamp': datetime.now().isoformat()})}\n\n"
                
            except Exception as e:
                error_update = {
                    "type": "stream_error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(error_update)}\n\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Workflow streaming failed: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow streaming failed: {str(e)}")


@router.get("/templates")
async def get_workflow_templates(current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Get available workflow templates"""
    try:
        chain_memory = get_agent_chain_memory()
        
        return {
            "success": True,
            "templates": {
                name: {
                    "name": template["name"],
                    "description": template["description"],
                    "steps": len(template["steps"]),
                    "step_types": [step["agent_type"] for step in template["steps"]]
                }
                for name, template in chain_memory.workflow_templates.items()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get workflow templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")


@router.get("/status")
async def get_chaining_status(current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Get current status of agent chaining system"""
    try:
        workflow_engine = get_workflow_engine()
        chain_memory = get_agent_chain_memory()
        
        # Get performance metrics
        performance_metrics = chain_memory.get_performance_metrics()
        
        # Get active executions
        active_executions = workflow_engine.get_active_executions()
        
        return {
            "success": True,
            "status": "operational",
            "performance_metrics": performance_metrics,
            "active_executions": active_executions,
            "available_agents": list(workflow_engine.agent_registry.keys()),
            "available_templates": list(chain_memory.workflow_templates.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get chaining status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/memory/cleanup")
async def cleanup_memory(
    max_age_hours: int = 24,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Clean up old workflow memory"""
    try:
        chain_memory = get_agent_chain_memory()
        
        before_count = len(chain_memory.active_workflows)
        chain_memory.cleanup_completed_workflows(max_age_hours)
        after_count = len(chain_memory.active_workflows)
        
        cleaned_count = before_count - after_count
        
        return {
            "success": True,
            "cleaned_workflows": cleaned_count,
            "remaining_workflows": after_count,
            "max_age_hours": max_age_hours,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Memory cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Memory cleanup failed: {str(e)}")


@router.get("/workflow/{workflow_id}/details")
async def get_workflow_details(workflow_id: str, current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Get detailed information about a specific workflow"""
    try:
        chain_memory = get_agent_chain_memory()
        
        # Check active workflows first
        if workflow_id in chain_memory.active_workflows:
            workflow = chain_memory.active_workflows[workflow_id]
            
            return {
                "success": True,
                "workflow": workflow,
                "status": chain_memory.get_workflow_status(workflow_id),
                "source": "active",
                "timestamp": datetime.now().isoformat()
            }
        
        # Check workflow history
        for historical_workflow in chain_memory.chain_history:
            if historical_workflow.get("workflow_id") == workflow_id:
                return {
                    "success": True,
                    "workflow": historical_workflow,
                    "status": {"status": historical_workflow.get("status", "unknown")},
                    "source": "history",
                    "timestamp": datetime.now().isoformat()
                }
        
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get workflow details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow details: {str(e)}")


# Export the router
__all__ = ["router"]
