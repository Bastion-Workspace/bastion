"""
gRPC Orchestrator Proxy - Forwards requests to LLM Orchestrator microservice
Phase 5: Integration endpoint for new microservices architecture
"""

import logging
import asyncio
import json
from typing import AsyncIterator, Dict, Any
from datetime import datetime

import grpc
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from utils.auth_middleware import get_current_user, AuthenticatedUserResponse

logger = logging.getLogger(__name__)

# Import proto files (will be generated)
try:
    from protos import orchestrator_pb2, orchestrator_pb2_grpc
    GRPC_AVAILABLE = True
except ImportError:
    logger.warning("gRPC orchestrator protos not available - proxy disabled")
    GRPC_AVAILABLE = False

router = APIRouter()


def format_sse_message(data: Dict[str, Any]) -> str:
    """
    Centralized SSE message formatter - converts Python dict to proper JSON format
    
    This ensures all streaming responses use valid JSON with double quotes,
    not Python dict repr with single quotes.
    
    Args:
        data: Dictionary to serialize
        
    Returns:
        SSE-formatted message with proper JSON
    """
    json_str = json.dumps(data)
    return f"data: {json_str}\n\n"


class OrchesterRequest(BaseModel):
    """Request model for orchestrator proxy"""
    query: str
    conversation_id: str = None
    user_id: str = None
    agent_type: str = None  # Optional: "research", "chat", "data_formatting", "auto"
    routing_reason: str = None  # Optional: why this agent was selected
    session_id: str = "default"  # Session identifier
    
    # Frontend context fields
    active_editor: Dict[str, Any] = None
    editor_preference: str = None  # "prefer", "ignore", "require"
    pipeline_preference: str = None  # "prefer", "ignore", "require"
    active_pipeline_id: str = None
    locked_agent: str = None  # Agent routing lock
    base_checkpoint_id: str = None  # For conversation branching


async def stream_from_grpc_orchestrator(
    query: str,
    conversation_id: str,
    user_id: str,
    session_id: str = "default",
    agent_type: str = None,
    routing_reason: str = None,
    request_context: Dict[str, Any] = None,
    state: Dict[str, Any] = None
) -> AsyncIterator[str]:
    """
    Stream responses from gRPC orchestrator microservice
    
    Args:
        query: User query
        conversation_id: Conversation ID
        user_id: User ID
        session_id: Session identifier
        agent_type: Optional agent type ("research", "chat", "data_formatting", "auto")
        routing_reason: Optional reason for agent selection
        request_context: Frontend request context (active_editor, pipeline, etc.)
        state: LangGraph conversation state (if available)
        
    Yields:
        SSE-formatted events
    """
    try:
        # Connect to gRPC orchestrator service
        orchestrator_host = 'llm-orchestrator'
        orchestrator_port = 50051
        
        logger.info(f"Connecting to gRPC orchestrator at {orchestrator_host}:{orchestrator_port}")
        if agent_type:
            logger.info(f"Requesting agent type: {agent_type}")
        
        async with grpc.aio.insecure_channel(f'{orchestrator_host}:{orchestrator_port}') as channel:
            stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)
            
            # Use context gatherer to build comprehensive request
            from services.grpc_context_gatherer import get_context_gatherer
            context_gatherer = get_context_gatherer()
            
            grpc_request = await context_gatherer.build_chat_request(
                query=query,
                user_id=user_id,
                conversation_id=conversation_id,
                session_id=session_id,
                request_context=request_context,
                state=state,
                agent_type=agent_type,
                routing_reason=routing_reason
            )
            
            logger.info(f"Forwarding to gRPC orchestrator: {query[:100]}")
            
            # Stream chunks from gRPC service
            chunk_count = 0
            async for chunk in stub.StreamChat(grpc_request):
                chunk_count += 1
                
                # Convert gRPC chunk to SSE format using centralized JSON formatter
                if chunk.type == "status":
                    yield format_sse_message({
                        'type': 'status',
                        'content': chunk.message,
                        'agent': chunk.agent_name,
                        'timestamp': chunk.timestamp
                    })
                
                elif chunk.type == "content":
                    yield format_sse_message({
                        'type': 'content',
                        'content': chunk.message,
                        'agent': chunk.agent_name
                    })
                
                elif chunk.type == "complete":
                    yield format_sse_message({
                        'type': 'complete',
                        'content': chunk.message,
                        'agent': chunk.agent_name
                    })
                
                elif chunk.type == "error":
                    yield format_sse_message({
                        'type': 'error',
                        'content': chunk.message,
                        'agent': chunk.agent_name
                    })
                
                # Flush immediately
                await asyncio.sleep(0)
            
            logger.info(f"Received {chunk_count} chunks from gRPC orchestrator")
            
            # Send final complete event
            yield format_sse_message({
                'type': 'done',
                'conversation_id': conversation_id
            })
            
    except grpc.RpcError as e:
        logger.error(f"gRPC error: {e.code()} - {e.details()}")
        yield format_sse_message({
            'type': 'error',
            'content': f"gRPC Orchestrator Error: {e.details()}"
        })
    
    except Exception as e:
        logger.error(f"Error streaming from gRPC orchestrator: {e}")
        import traceback
        traceback.print_exc()
        yield format_sse_message({
            'type': 'error',
            'content': f"Orchestrator error: {str(e)}"
        })


@router.post("/api/async/orchestrator/grpc/stream")
async def stream_orchestrator_grpc(
    request: OrchesterRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Stream orchestrator responses via gRPC microservice
    
    This endpoint forwards requests to the LLM Orchestrator microservice
    running on port 50051.
    """
    if not GRPC_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="gRPC orchestrator not available - protos not generated"
        )
    
    try:
        logger.info(f"gRPC Orchestrator Proxy: User {current_user.user_id} query: {request.query[:100]}")
        
        # Use provided user_id or get from auth
        target_user_id = request.user_id or current_user.user_id
        
        # Build request context from frontend fields
        request_context = {
            "active_editor": request.active_editor,
            "editor_preference": request.editor_preference,
            "pipeline_preference": request.pipeline_preference,
            "active_pipeline_id": request.active_pipeline_id,
            "locked_agent": request.locked_agent,
            "base_checkpoint_id": request.base_checkpoint_id
        }
        
        # Remove None values
        request_context = {k: v for k, v in request_context.items() if v is not None}
        
        return StreamingResponse(
            stream_from_grpc_orchestrator(
                query=request.query,
                conversation_id=request.conversation_id,
                user_id=target_user_id,
                session_id=request.session_id,
                agent_type=request.agent_type,
                routing_reason=request.routing_reason,
                request_context=request_context if request_context else None,
                state=None  # TODO: Fetch from conversation DB if needed
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    except Exception as e:
        logger.error(f"Error in gRPC orchestrator proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/async/orchestrator/grpc/health")
async def grpc_orchestrator_health():
    """Check health of gRPC orchestrator service"""
    if not GRPC_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "gRPC protos not generated"
        }
    
    try:
        async with grpc.aio.insecure_channel('llm-orchestrator:50051') as channel:
            stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)
            
            health_req = orchestrator_pb2.HealthCheckRequest()
            health_resp = await stub.HealthCheck(health_req)
            
            return {
                "status": health_resp.status,
                "details": dict(health_resp.details)
            }
    
    except Exception as e:
        logger.error(f"gRPC health check failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

