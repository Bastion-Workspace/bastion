"""
Agent-Specific Celery Tasks
Background processing for individual agents
Uses gRPC orchestrator for all agent processing
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
import grpc

from services.celery_app import celery_app, update_task_progress, TaskStatus

logger = logging.getLogger(__name__)

# Import proto files for gRPC orchestrator
try:
    from protos import orchestrator_pb2, orchestrator_pb2_grpc
    GRPC_AVAILABLE = True
except ImportError:
    logger.warning("gRPC orchestrator protos not available - background tasks disabled")
    GRPC_AVAILABLE = False


async def _call_grpc_orchestrator(
    query: str,
    user_id: str,
    conversation_id: str,
    agent_type: Optional[str] = None,
    persona: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Call gRPC orchestrator and collect full response (non-streaming)
    
    Args:
        query: User query
        user_id: User ID
        conversation_id: Conversation ID
        agent_type: Optional agent type ("research", "chat", "coding", etc.)
        persona: Optional persona settings
        
    Returns:
        Dict with 'response', 'success', 'agent_type', etc.
    """
    if not GRPC_AVAILABLE:
        return {
            "success": False,
            "error": "gRPC orchestrator not available",
            "message": "Background task processing unavailable"
        }
    
    try:
        # Connect to gRPC orchestrator service
        orchestrator_host = 'llm-orchestrator'
        orchestrator_port = 50051
        
        logger.info(f"🔗 Background task connecting to gRPC orchestrator at {orchestrator_host}:{orchestrator_port}")
        
        # Increase message size limits for large responses
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100 MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100 MB
        ]
        
        async with grpc.aio.insecure_channel(f'{orchestrator_host}:{orchestrator_port}', options=options) as channel:
            stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)
            
            # Use context gatherer to build comprehensive request
            from services.grpc_context_gatherer import get_context_gatherer
            context_gatherer = get_context_gatherer()
            
            # Build request context
            request_context = {}
            if persona:
                request_context["persona"] = persona
            
            grpc_request = await context_gatherer.build_chat_request(
                query=query,
                user_id=user_id,
                conversation_id=conversation_id,
                session_id="background_task",
                request_context=request_context if request_context else None,
                state=None,
                agent_type=agent_type,
                routing_reason=f"Background task: {agent_type or 'auto'}"
            )
            
            logger.info(f"📤 Background task forwarding to gRPC orchestrator: {query[:100]}")
            
            # Collect all chunks from streaming response
            full_response = ""
            agent_name = None
            status_messages = []
            
            async for chunk in stub.StreamChat(grpc_request):
                if chunk.type == "status":
                    status_messages.append(chunk.message)
                    if chunk.agent_name:
                        agent_name = chunk.agent_name
                    logger.debug(f"📊 Status: {chunk.message}")
                
                elif chunk.type == "content":
                    full_response += chunk.message
                    if chunk.agent_name:
                        agent_name = chunk.agent_name
                
                elif chunk.type == "error":
                    logger.error(f"❌ gRPC orchestrator error: {chunk.message}")
                    return {
                        "success": False,
                        "error": chunk.message,
                        "message": "Background task processing failed"
                    }
            
            logger.info(f"✅ Background task received response from {agent_name or 'orchestrator'}: {len(full_response)} chars")
            
            return {
                "success": True,
                "response": full_response,
                "agent_type": agent_name or agent_type or "unknown",
                "status_messages": status_messages
            }
            
    except Exception as e:
        logger.error(f"❌ Background task gRPC orchestrator error: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Background task processing failed"
        }


@celery_app.task(bind=True, name="agents.research_task")
def research_background_task(
    self,
    user_id: str,
    conversation_id: str,
    query: str,
    persona: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Background task specifically for research operations"""
    try:
        logger.info(f"🔬 ASYNC RESEARCH: Starting background research for user {user_id}")
        
        update_task_progress(self, 1, 4, "Initializing research agent...")
        
        # Run async research processing
        result = asyncio.run(_async_research_processing(
            self, user_id, conversation_id, query, persona
        ))
        
        logger.info(f"✅ ASYNC RESEARCH: Completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"❌ ASYNC RESEARCH ERROR: {e}")
        
        self.update_state(
            state=TaskStatus.FAILURE,
            meta={
                "error": str(e),
                "message": "Research processing failed",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": False,
            "error": str(e),
            "message": "Background research processing failed"
        }


async def _async_research_processing(
    task,
    user_id: str,
    conversation_id: str,
    query: str,
    persona: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Internal async function for research processing - uses gRPC orchestrator"""
    try:
        update_task_progress(task, 2, 4, "Loading conversation context...")
        
        # Load conversation history for context (gRPC context gatherer will use this)
        from services.conversation_service import get_conversation_service
        conv_service = await get_conversation_service()
        conversation_messages = await conv_service.get_messages(conversation_id, user_id)
        
        update_task_progress(task, 3, 4, "Processing with gRPC orchestrator (research agent)...")
        
        # Use gRPC orchestrator with research agent
        result = await _call_grpc_orchestrator(
            query=query,
            user_id=user_id,
            conversation_id=conversation_id,
            agent_type="research",
            persona=persona
        )
        
        if not result.get("success"):
            raise Exception(result.get("error", "Unknown error"))
        
        update_task_progress(task, 4, 4, "Research completed!")
        
        task.update_state(
            state=TaskStatus.SUCCESS,
            meta={
                "result": result,
                "message": "Research completed successfully",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": True,
            "response": result.get("response", "Research completed"),
            "agent_type": "research",
            "task_id": task.request.id
        }
        
    except Exception as e:
        logger.error(f"❌ Async research processing error: {e}")
        
        task.update_state(
            state=TaskStatus.FAILURE,
            meta={
                "error": str(e),
                "message": "Research processing error",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": False,
            "error": str(e),
            "message": "Research processing failed"
        }


@celery_app.task(bind=True, name="agents.coding_task")
def coding_background_task(
    self,
    user_id: str,
    conversation_id: str,
    query: str,
    persona: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Background task specifically for coding operations"""
    try:
        logger.info(f"💻 ASYNC CODING: Starting background coding for user {user_id}")
        
        update_task_progress(self, 1, 3, "Initializing coding agent...")
        
        result = asyncio.run(_async_coding_processing(
            self, user_id, conversation_id, query, persona
        ))
        
        logger.info(f"✅ ASYNC CODING: Completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"❌ ASYNC CODING ERROR: {e}")
        
        self.update_state(
            state=TaskStatus.FAILURE,
            meta={
                "error": str(e),
                "message": "Coding processing failed",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": False,
            "error": str(e),
            "message": "Background coding processing failed"
        }


async def _async_coding_processing(
    task,
    user_id: str,
    conversation_id: str,
    query: str,
    persona: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Internal async function for coding processing - uses gRPC orchestrator"""
    try:
        update_task_progress(task, 2, 3, "Processing with gRPC orchestrator (chat agent)...")
        
        # Use gRPC orchestrator with chat agent (coding routes to chat)
        result = await _call_grpc_orchestrator(
            query=query,
            user_id=user_id,
            conversation_id=conversation_id,
            agent_type="chat",  # Coding requests route to chat agent
            persona=persona
        )
        
        if not result.get("success"):
            raise Exception(result.get("error", "Unknown error"))
        
        update_task_progress(task, 3, 3, "Coding completed!")
        
        task.update_state(
            state=TaskStatus.SUCCESS,
            meta={
                "result": result,
                "message": "Coding completed successfully",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": True,
            "response": result.get("response", "Coding completed"),
            "agent_type": "coding",
            "task_id": task.request.id
        }
        
    except Exception as e:
        logger.error(f"❌ Async coding processing error: {e}")
        
        task.update_state(
            state=TaskStatus.FAILURE,
            meta={
                "error": str(e),
                "message": "Coding processing error",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": False,
            "error": str(e),
            "message": "Coding processing failed"
        }


@celery_app.task(bind=True, name="agents.batch_processing")
def batch_agent_processing(
    self,
    tasks: list,
    user_id: str,
    conversation_id: str
) -> Dict[str, Any]:
    """Process multiple agent tasks in sequence"""
    try:
        logger.info(f"📦 BATCH PROCESSING: {len(tasks)} tasks for user {user_id}")
        
        results = []
        total_tasks = len(tasks)
        
        for i, task_config in enumerate(tasks):
            update_task_progress(
                self, i + 1, total_tasks, 
                f"Processing task {i+1}/{total_tasks}: {task_config.get('agent_type', 'unknown')}"
            )
            
            # Process individual task based on type
            agent_type = task_config.get("agent_type")
            query = task_config.get("query")
            persona = task_config.get("persona")
            
            if agent_type == "research":
                result = asyncio.run(_async_research_processing(
                    self, user_id, conversation_id, query, persona
                ))
            elif agent_type == "coding":
                result = asyncio.run(_async_coding_processing(
                    self, user_id, conversation_id, query, persona
                ))
            else:
                result = {"success": False, "error": f"Unknown agent type: {agent_type}"}
            
            results.append({
                "task_index": i,
                "agent_type": agent_type,
                "result": result
            })
        
        self.update_state(
            state=TaskStatus.SUCCESS,
            meta={
                "results": results,
                "message": f"Batch processing completed: {len(tasks)} tasks",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": True,
            "results": results,
            "total_tasks": total_tasks,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(f"❌ BATCH PROCESSING ERROR: {e}")
        
        self.update_state(
            state=TaskStatus.FAILURE,
            meta={
                "error": str(e),
                "message": "Batch processing failed",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": False,
            "error": str(e),
            "message": "Batch processing failed"
        }
