"""
gRPC Orchestrator Proxy - Forwards requests to LLM Orchestrator microservice
Phase 5: Integration endpoint for new microservices architecture
"""

import logging
import asyncio
import json
from typing import AsyncIterator, Dict, Any, Optional
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
    json_str = json.dumps(data, ensure_ascii=False)
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
    # Track if title was updated (for sending conversation_updated event even on errors)
    title_updated = False
    
    try:
        # Connect to gRPC orchestrator service
        orchestrator_host = 'llm-orchestrator'
        orchestrator_port = 50051
        
        logger.info(f"Connecting to gRPC orchestrator at {orchestrator_host}:{orchestrator_port}")
        if agent_type:
            logger.info(f"Requesting agent type: {agent_type}")
        
        # Increase message size limits for large responses (default is 4MB)
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100 MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100 MB
        ]
        async with grpc.aio.insecure_channel(f'{orchestrator_host}:{orchestrator_port}', options=options) as channel:
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
            
            # Initialize title_updated flag before try block to avoid UnboundLocalError
            title_updated = False
            
            # Save user message to conversation BEFORE processing
            # This will also trigger title generation if it's the first message
            try:
                from services.conversation_service import ConversationService
                conversation_service = ConversationService()
                conversation_service.set_current_user(user_id)
                
                user_message_result = await conversation_service.add_message(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    role="user",
                    content=query,
                    metadata={"orchestrator_system": True, "streaming": True}
                )
                logger.info(f"✅ Saved user message to conversation {conversation_id}")
                
                # Check if title was updated (first message triggers title generation)
                # The add_message method updates the title if it was "New Conversation"
                # We'll emit a conversation_updated event after streaming completes
                title_updated = True
            except Exception as save_error:
                logger.warning(f"⚠️ Failed to save user message: {save_error}")
                # Continue even if message save fails
            
            # Stream chunks from gRPC service
            chunk_count = 0
            agent_name_used = None  # Track which agent was used
            accumulated_response = ""  # Accumulate full response content
            editor_operations_received = None  # Track editor operations for metadata
            manuscript_edit_received = None  # Track manuscript edit for metadata
            
            async for chunk in stub.StreamChat(grpc_request):
                chunk_count += 1
                
                # Track agent name from chunks (use the most specific one, not "orchestrator" or "system")
                if chunk.agent_name and chunk.agent_name not in ["orchestrator", "system"]:
                    agent_name_used = chunk.agent_name
                
                # Accumulate content chunks for saving as last_response
                if chunk.type == "content" and chunk.message:
                    accumulated_response += chunk.message
                
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
                
                elif chunk.type == "editor_operations":
                    # Parse JSON from message field and forward as editor_operations type
                    try:
                        logger.info(f"📥 PROXY: Received editor_operations chunk from gRPC orchestrator (message length: {len(chunk.message)})")
                        editor_ops_data = json.loads(chunk.message)
                        operations = editor_ops_data.get('operations', [])
                        manuscript_edit = editor_ops_data.get('manuscript_edit')
                        
                        # Store for saving to message metadata
                        editor_operations_received = operations
                        manuscript_edit_received = manuscript_edit
                        
                        logger.info(f"📥 PROXY: Parsed {len(operations)} operation(s) from editor_operations chunk")
                        
                        # **CRITICAL**: SSE messages have size limits (~16-64KB depending on proxy/browser)
                        # If operations are large (e.g., full chapter generation), send them one at a time
                        # to avoid message truncation
                        
                        # Calculate total size
                        total_size = len(json.dumps(editor_ops_data))
                        MAX_SSE_SIZE = 50000  # 50KB safety limit
                        
                        if total_size > MAX_SSE_SIZE and len(operations) > 0:
                            # Send operations individually to avoid SSE size limits
                            logger.info(f"📦 Editor operations are large ({total_size} bytes) - chunking into {len(operations)} separate messages")
                            for idx, op in enumerate(operations):
                                yield format_sse_message({
                                    'type': 'editor_operations_chunk',
                                    'chunk_index': idx,
                                    'total_chunks': len(operations),
                                    'operation': op,
                                    'manuscript_edit': manuscript_edit if idx == len(operations) - 1 else None  # Include metadata on last chunk only
                                })
                                logger.info(f"📦 Sent chunk {idx + 1}/{len(operations)} (op_type={op.get('op_type')}, text_length={len(op.get('text', ''))})")
                        else:
                            # Small enough to send as single message
                            logger.info(f"✅ PROXY: Sending {len(operations)} editor operation(s) as single SSE message (size: {total_size} bytes)")
                            yield format_sse_message({
                                'type': 'editor_operations',
                                'operations': operations,
                                'manuscript_edit': manuscript_edit
                            })
                            logger.info(f"✅ PROXY: Sent editor_operations SSE message to frontend")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse editor_operations JSON: {e}")
                    except Exception as e:
                        logger.error(f"❌ PROXY: Error processing editor_operations chunk: {e}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                
                elif chunk.type == "title":
                    # Forward title chunks immediately so frontend can update UI right away
                    title_updated = True
                    yield format_sse_message({
                        'type': 'title',
                        'message': chunk.message,
                        'timestamp': chunk.timestamp,
                        'agent': chunk.agent_name
                    })
                    logger.info(f"🔤 Forwarded title chunk to frontend: {chunk.message}")
                
                # Flush immediately
                await asyncio.sleep(0)
            
            logger.info(f"Received {chunk_count} chunks from gRPC orchestrator")
            
            # Save assistant response to conversation AFTER streaming completes
            if accumulated_response:
                try:
                    from services.conversation_service import ConversationService
                    conversation_service = ConversationService()
                    conversation_service.set_current_user(user_id)
                    
                    # Build metadata with editor operations if they were received
                    metadata = {
                        "orchestrator_system": True,
                        "streaming": True,
                        "delegated_agent": agent_name_used or "unknown",
                        "chunk_count": chunk_count
                    }
                    
                    # Add editor_operations and manuscript_edit to metadata for persistence
                    if editor_operations_received:
                        metadata["editor_operations"] = editor_operations_received
                        logger.info(f"💾 PROXY: Including {len(editor_operations_received)} editor_operations in message metadata")
                    if manuscript_edit_received:
                        metadata["manuscript_edit"] = manuscript_edit_received
                    
                    assistant_message_result = await conversation_service.add_message(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        role="assistant",
                        content=accumulated_response,
                        metadata=metadata
                    )
                    logger.info(f"✅ Saved assistant response to conversation {conversation_id} (agent: {agent_name_used or 'unknown'})")
                    
                    # Save agent routing metadata to backend DB (source of truth)
                    if agent_name_used:
                        try:
                            await conversation_service.update_agent_metadata(
                                conversation_id=conversation_id,
                                user_id=user_id,
                                primary_agent_selected=agent_name_used,
                                last_agent=agent_name_used
                            )
                            logger.info(f"✅ Saved agent metadata to backend DB: {agent_name_used}")
                        except Exception as agent_save_error:
                            logger.warning(f"⚠️ Failed to save agent metadata: {agent_save_error}")
                    
                except Exception as save_error:
                    logger.warning(f"⚠️ Failed to save assistant response: {save_error}")
                    # Continue even if message save fails
            
            # NOTE: gRPC orchestrator handles its own state management via LangGraph checkpointing
            # No need to manually update backend orchestrator state
            
            # Send final complete event with conversation update flag
            # This signals the frontend to refresh the conversation list (title may have been updated)
            yield format_sse_message({
                'type': 'done',
                'conversation_id': conversation_id,
                'conversation_updated': True  # Signal that conversation metadata may have changed
            })
            
    except grpc.RpcError as e:
        logger.error(f"gRPC error: {e.code()} - {e.details()}")
        yield format_sse_message({
            'type': 'error',
            'content': f"gRPC Orchestrator Error: {e.details()}"
        })
        # Send done event if title was updated (title generation happens before streaming)
        if title_updated:
            yield format_sse_message({
                'type': 'done',
                'conversation_id': conversation_id,
                'conversation_updated': True
            })
    
    except Exception as e:
        logger.error(f"Error streaming from gRPC orchestrator: {e}")
        import traceback
        traceback.print_exc()
        yield format_sse_message({
            'type': 'error',
            'content': f"Orchestrator error: {str(e)}"
        })
        # Send done event if title was updated (title generation happens before streaming)
        if title_updated:
            yield format_sse_message({
                'type': 'done',
                'conversation_id': conversation_id,
                'conversation_updated': True
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
        
        # NOTE: gRPC orchestrator handles its own state management via LangGraph checkpointing
        # State is automatically retrieved by the gRPC service
        conversation_state = None
        
        return StreamingResponse(
            stream_from_grpc_orchestrator(
                query=request.query,
                conversation_id=request.conversation_id,
                user_id=target_user_id,
                session_id=request.session_id,
                agent_type=request.agent_type,
                routing_reason=request.routing_reason,
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
        # Increase message size limits (default is 4MB)
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100 MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100 MB
        ]
        async with grpc.aio.insecure_channel('llm-orchestrator:50051', options=options) as channel:
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


async def _load_conversation_state(user_id: str, conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    DEPRECATED: This function is no longer needed.
    
    The llm-orchestrator loads shared_memory (including primary_agent_selected, 
    last_agent, etc.) directly from the LangGraph checkpoint in its StreamChat handler.
    
    Returning None here - orchestrator handles all checkpoint loading.
    
    Args:
        user_id: User UUID
        conversation_id: Conversation UUID

    Returns:
        None - orchestrator loads checkpoint directly
    """
    # The llm-orchestrator microservice loads checkpoint shared_memory directly
    # in grpc_service.py::StreamChat() at line ~580 via _load_checkpoint_shared_memory()
    # This includes primary_agent_selected, last_agent, and all continuity data.
    # No need for backend to duplicate this loading.
    logger.debug(f"Conversation state loading delegated to llm-orchestrator for {conversation_id}")
    return None

