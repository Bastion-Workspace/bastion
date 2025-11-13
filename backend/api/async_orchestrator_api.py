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
        
        # Check if we should use gRPC orchestrator
        from utils.feature_flags import use_grpc_orchestrator
        if use_grpc_orchestrator(current_user.user_id):
            logger.info(f"🎯 ROUTING TO gRPC ORCHESTRATOR (Phase 5)")
            # Forward to gRPC orchestrator microservice
            from api.grpc_orchestrator_proxy import stream_from_grpc_orchestrator
            return StreamingResponse(
                stream_from_grpc_orchestrator(
                    query=request.query,
                    conversation_id=request.conversation_id,
                    user_id=current_user.user_id
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        logger.info(f"🏛️ USING LOCAL ORCHESTRATOR (gRPC disabled)")
        
        async def generate_stream():
            try:
                # Get user settings for persona
                user_settings = await prompt_service.get_user_settings_for_service(current_user.user_id)
                persona = {
                    "ai_name": user_settings.ai_name if user_settings else "Kodex",
                    "persona_style": user_settings.persona_style.value if user_settings else "professional",
                    "political_bias": user_settings.political_bias.value if user_settings else "neutral"
                } if user_settings else None
                
                # ROOSEVELT'S PURE LANGGRAPH: Let LangGraph handle ALL message persistence
                # Pure LangGraph AsyncPostgresSaver handles all persistence automatically
                # No dual persistence needed - 100% LangGraph native
                
                logger.info(f"🎯 ROOSEVELT'S PURE LANGGRAPH: User message will be handled by LangGraph orchestrator")
                
                # Initialize OFFICIAL orchestrator foundation
                from services.langgraph_official_orchestrator import get_official_orchestrator
                orchestrator = await get_official_orchestrator()
                
                # Process with official orchestrator (with simulated streaming for UX)
                yield f"data: {json.dumps({'type': 'status', 'message': '🎯 Official LangGraph Orchestrator analyzing request...', 'timestamp': datetime.now().isoformat()})}\n\n"
                
                # Process with official orchestrator
                # Respect editor preference: if ignore, drop active_editor
                if (request.editor_preference or '').lower() == 'ignore':
                  request.active_editor = None

                # Validate and attach active editor (gate: editable .md). Type is advisory, not required.
                shared_active_editor = None
                try:
                  ae = request.active_editor or None
                  if isinstance(ae, dict) and ae.get('is_editable') and isinstance(ae.get('filename'), str):
                    fname = ae['filename'].lower()
                    if fname.endswith('.md'):
                      # Normalize frontmatter to a dict from multiple possible fields
                      def _lower_keys(d: dict) -> dict:
                        try:
                          return { (str(k).lower() if isinstance(k, str) else k): v for k, v in d.items() }
                        except Exception:
                          return d
                      import yaml, re
                      content_text = ae.get('content') or ''
                      # Candidates for frontmatter sources
                      fm_candidates = [
                        ae.get('frontmatter'),
                        ae.get('frontMatter'),
                        ae.get('front_matter'),
                        ae.get('metadata'),
                        ae.get('frontmatter_text'),
                        ae.get('frontmatterYaml'),
                        ae.get('frontmatter_yaml'),
                      ]
                      fm: dict = {}
                      parsed_any = False
                      for cand in fm_candidates:
                        if isinstance(cand, dict):
                          fm = _lower_keys(cand)
                          parsed_any = True
                          break
                        if isinstance(cand, str) and cand.strip():
                          try:
                            parsed = yaml.safe_load(cand)
                            if isinstance(parsed, dict):
                              fm = _lower_keys(parsed)
                              parsed_any = True
                              break
                          except Exception:
                            pass
                      # Fallback: try to extract YAML frontmatter from content
                      if not parsed_any and isinstance(content_text, str):
                        try:
                          m = re.search(r'^\ufeff?\s*---\s*\r?\n([\s\S]*?)\r?\n---\s*(?:\r?\n|$)', content_text, re.MULTILINE)
                          if m:
                            fm_block = m.group(1)
                            parsed = yaml.safe_load(fm_block)
                            if isinstance(parsed, dict):
                              fm = _lower_keys(parsed)
                        except Exception:
                          fm = {}
                      shared_active_editor = {
                        'is_editable': True,
                        'filename': ae.get('filename'),
                        'language': ae.get('language') or 'markdown',
                        'content': content_text,
                        'content_length': ae.get('content_length') or 0,
                        'frontmatter': fm,
                        # optional cursor/selection metadata for editing agent
                        'cursor_offset': ae.get('cursor_offset', -1),
                        'selection_start': ae.get('selection_start', -1),
                        'selection_end': ae.get('selection_end', -1),
                        'canonical_path': ae.get('canonical_path') or None,
                      }
                      try:
                        fm_type_dbg = str((fm.get('type') if isinstance(fm, dict) else None) or '').strip().lower()
                        logger.info(f"🧭 EDITOR FRONTMATTER: filename='{shared_active_editor['filename']}', type='{fm_type_dbg}'")
                      except Exception:
                        pass
                except Exception:
                  shared_active_editor = None

                # Orchestrator call
                result = await orchestrator.process_user_query(
                    user_message=request.query,
                    user_id=current_user.user_id,
                    conversation_id=request.conversation_id,
                    persona=persona,
                    extra_shared_memory={
                        **({"active_editor": shared_active_editor} if shared_active_editor else {}),
                        **({"locked_agent": request.locked_agent} if request.locked_agent else {})
                    } if (shared_active_editor or request.locked_agent) else None,
                    base_checkpoint_id=request.base_checkpoint_id
                )
                
                # Extract final response - ROOSEVELT'S HITL PERMISSION HANDLING
                final_response = result.get("response", "No response generated")
                status = result.get("status", "complete")
                is_interrupted = result.get("interrupted", False)
                permission_request = result.get("permission_request", False)

                # ROOSEVELT'S BULLETPROOF HITL: Prefer structured permission data, then fallback to text scanning
                permission_message_found = False

                if status == "interrupted" or is_interrupted or permission_request:
                    logger.info("🛡️ HITL DETECTED: Attempting permission message extraction (structured-first)...")

                    final_state = result.get("final_state", {})
                    agent_results = final_state.get("agent_results", {}) if isinstance(final_state, dict) else {}

                    # 1) Prefer top-level structured permission object with message
                    if isinstance(permission_request, dict) and permission_request.get("message"):
                        final_response = permission_request.get("message")
                        permission_message_found = True
                        logger.info("✅ HITL PERMISSION (structured, top-level): Using permission_request.message")

                    # 2) Then prefer agent_results.permission_message if provided by agent
                    if not permission_message_found and isinstance(agent_results, dict):
                        perm_msg = agent_results.get("permission_message")
                        if isinstance(perm_msg, str) and perm_msg.strip():
                            final_response = perm_msg
                            permission_message_found = True
                            logger.info("✅ HITL PERMISSION (structured, agent_results.permission_message)")

                    # 3) Next, check structured_response.permission_request (string justification)
                    if not permission_message_found and isinstance(agent_results, dict):
                        structured_response = agent_results.get("structured_response", {})
                        if isinstance(structured_response, dict):
                            perm_justification = structured_response.get("permission_request")
                            if isinstance(perm_justification, str) and perm_justification.strip():
                                final_response = (
                                    "🔍 Local research complete but insufficient.\n\n"
                                    f"🌐 Web Search Needed: {perm_justification}\n\n"
                                    "Would you like me to search the web for additional information? (yes/no)"
                                )
                                permission_message_found = True
                                logger.info("✅ HITL PERMISSION (structured, justification-based message)")

                    # 4) Fallback to message scanning only if structured extraction failed
                    if not permission_message_found:
                        messages = final_state.get("messages", [])
                        for msg in reversed(messages):
                            if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == 'ai':
                                content = str(msg.content)
                                if any(indicator in content.lower() for indicator in [
                                    "permission request", "web search permission",
                                    "would you like me to proceed", "reply with \"yes\" or \"no\"",
                                    "estimated cost", "safety level"
                                ]):
                                    final_response = content
                                    permission_message_found = True
                                    logger.info(f"✅ HITL PERMISSION (fallback text scan): {content[:100]}...")
                                    break

                        # Final fallback: use latest AI message
                        if not permission_message_found and messages:
                            for msg in reversed(messages):
                                if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == 'ai':
                                    final_response = str(msg.content)
                                    logger.info(f"🛡️ HITL FALLBACK: Using latest AI message: {final_response[:100]}...")
                                    break
                
                logger.info(f"🛡️ HITL DEBUG: final_response length: {len(final_response)}, interrupted: {is_interrupted}, permission_request: {permission_request}, status: {status}")
                
                # ROOSEVELT'S TOOL STATUS STREAMING: Extract and stream tool status updates
                final_state = result.get("final_state", {})
                # If we validated active editor, ensure it is reflected for routing in state
                if shared_active_editor and isinstance(final_state, dict):
                    final_state.setdefault('shared_memory', {})
                    try:
                        final_state['shared_memory']['active_editor'] = shared_active_editor
                    except Exception:
                        pass
                tool_status_updates = final_state.get("tool_status_updates", [])
                
                for status_update in tool_status_updates:
                    yield f"data: {json.dumps(status_update)}\n\n"
                    await asyncio.sleep(0.1)  # Small delay for UX
                
                # ROOSEVELT'S EDITOR OPS STREAM: Emit editor operations for HITL preview/apply
                try:
                    # Check both locations where editor operations might be stored
                    editor_ops = None
                    manuscript_edit = None
                    
                    # DEBUG: Log what's in final_state
                    logger.info(f"🔍 EDITOR OPS DEBUG: final_state keys: {list(final_state.keys()) if isinstance(final_state, dict) else 'NOT A DICT'}")
                    
                    # First check direct state storage (used by proofreading_agent and fiction_editing_agent)
                    if isinstance(final_state, dict):
                        editor_ops = final_state.get("editor_operations")
                        manuscript_edit = final_state.get("manuscript_edit")
                        logger.info(f"🔍 EDITOR OPS DEBUG: Direct state check - editor_ops={'list with ' + str(len(editor_ops)) + ' items' if isinstance(editor_ops, list) else str(type(editor_ops))}")
                    
                    # Fallback to agent_results if not found in direct state
                    if not editor_ops:
                        agent_results = final_state.get("agent_results", {}) if isinstance(final_state, dict) else {}
                        logger.info(f"🔍 EDITOR OPS DEBUG: agent_results keys: {list(agent_results.keys()) if isinstance(agent_results, dict) else 'NOT A DICT'}")
                        if isinstance(agent_results, dict):
                            editor_ops = agent_results.get("editor_operations")
                            logger.info(f"🔍 EDITOR OPS DEBUG: agent_results check - editor_ops={'list with ' + str(len(editor_ops)) + ' items' if isinstance(editor_ops, list) else str(type(editor_ops))}")
                            if not manuscript_edit:
                                manuscript_edit = agent_results.get("manuscript_edit")
                    
                    # Stream editor operations if found
                    if isinstance(editor_ops, list) and editor_ops:
                        yield f"data: {json.dumps({'type': 'editor_operations', 'operations': editor_ops, 'manuscript_edit': manuscript_edit, 'timestamp': datetime.now().isoformat()})}\n\n"
                        logger.info(f"🎯 EDITOR OPS: Streamed {len(editor_ops)} operations from {'direct state' if final_state.get('editor_operations') else 'agent_results'}")
                    else:
                        logger.warning(f"⚠️ EDITOR OPS NOT STREAMED: editor_ops={'list with ' + str(len(editor_ops)) + ' items' if isinstance(editor_ops, list) else str(type(editor_ops))}")
                except Exception as e:
                    logger.error(f"❌ EDITOR OPS STREAM ERROR: {e}")
                    import traceback
                    logger.error(f"❌ TRACEBACK: {traceback.format_exc()}")
                    pass
                
                # ROOSEVELT'S PURE LANGGRAPH OPTIMIZATION: 
                # Pure LangGraph AsyncPostgresSaver handles all message persistence automatically
                # No need for redundant API-level message saving
                logger.info(f"✅ ROOSEVELT'S PURE LANGGRAPH: Messages persisted automatically by AsyncPostgresSaver")
                
                # Simulate progress updates for UX
                yield f"data: {json.dumps({'type': 'progress', 'node': 'intent_classifier', 'message': 'Classifying intent...', 'timestamp': datetime.now().isoformat()})}\n\n"
                await asyncio.sleep(0.2)
                yield f"data: {json.dumps({'type': 'progress', 'node': 'research_agent', 'message': 'Processing with research agent...', 'timestamp': datetime.now().isoformat()})}\n\n"
                await asyncio.sleep(0.3)
                
                # ROOSEVELT'S HITL STREAMING FIX: Use permission message instead of empty final_response
                response_to_stream = final_response

                logger.info(f"🛡️ HITL DEBUG: final_response length: {len(final_response)}, interrupted: {is_interrupted}, permission_request: {permission_request}")
                
                # No additional logic needed - final_response should already contain the permission message from above
                
                # ROOSEVELT'S MARKDOWN-PRESERVING STREAMING FIX
                # Split response by lines to preserve markdown structure, then by words within lines
                lines = response_to_stream.split('\n')
                
                for line_idx, line in enumerate(lines):
                    if line.strip():  # Only process non-empty lines
                        # For long lines, split by words but preserve line structure
                        if len(line) > 100:
                            words = line.split()
                            word_chunks = [' '.join(words[i:i+15]) for i in range(0, len(words), 15)]
                            for word_chunk_idx, word_chunk in enumerate(word_chunks):
                                yield f"data: {json.dumps({'type': 'content', 'content': word_chunk, 'timestamp': datetime.now().isoformat()})}\n\n"
                                await asyncio.sleep(0.05)
                        else:
                            # Short lines: send as-is
                            yield f"data: {json.dumps({'type': 'content', 'content': line, 'timestamp': datetime.now().isoformat()})}\n\n"
                            await asyncio.sleep(0.05)
                    
                    # Add newline after each line (except the last one)
                    if line_idx < len(lines) - 1:
                        newline_char = '\n'  # F-strings cannot contain backslashes in expressions
                        yield f"data: {json.dumps({'type': 'content', 'content': newline_char, 'timestamp': datetime.now().isoformat()})}\n\n"
                        await asyncio.sleep(0.02)
                
                # **ROOSEVELT'S CITATION CAVALRY STREAM**: Emit citations AFTER content but BEFORE complete!
                # This ensures the streaming message exists when citations arrive
                try:
                    agent_results = final_state.get("agent_results", {}) if isinstance(final_state, dict) else {}
                    citations = []
                    
                    if isinstance(agent_results, dict):
                        # Check agent_results for citations first
                        citations = agent_results.get("citations", [])
                        
                        # Also check structured_response for citations (some agents put them here)
                        if not citations:
                            structured_response = agent_results.get("structured_response", {})
                            if isinstance(structured_response, dict):
                                citations = structured_response.get("citations", [])
                    
                    # Stream citations if found
                    if isinstance(citations, list) and citations:
                        yield f"data: {json.dumps({'type': 'citations', 'citations': citations, 'timestamp': datetime.now().isoformat()})}\n\n"
                        logger.info(f"🔗 CITATIONS: Streamed {len(citations)} citations to frontend AFTER content!")
                    else:
                        logger.info(f"🔍 CITATIONS: No citations found in agent_results")
                except Exception as e:
                    logger.error(f"❌ CITATION STREAM ERROR: {e}")
                    pass
                
                # ROOSEVELT'S ENHANCED HITL: Send permission request or normal completion
                if status == "interrupted" or permission_request or permission_message_found:
                    # Send permission request message with special tagging
                    yield f"data: {json.dumps({'type': 'permission_request', 'content': final_response, 'requires_approval': True, 'conversation_id': request.conversation_id, 'timestamp': datetime.now().isoformat()})}\n\n"
                    
                    # Send HITL completion signal
                    yield f"data: {json.dumps({'type': 'complete_hitl', 'status': 'awaiting_permission', 'conversation_id': request.conversation_id, 'timestamp': datetime.now().isoformat()})}\n\n"
                    
                    logger.info("✅ PERMISSION REQUEST: Streamed to frontend with special tagging")
                else:
                    # Send clean orchestrator metadata for normal responses
                    yield f"data: {json.dumps({'type': 'metadata', 'orchestrator': 'clean_langgraph', 'agents': ['chat', 'research'], 'hitl_enabled': True, 'timestamp': datetime.now().isoformat()})}\n\n"
                    
                    # Send normal completion signal
                    yield f"data: {json.dumps({'type': 'complete', 'final_content': final_response, 'timestamp': datetime.now().isoformat()})}\n\n"
                
            except Exception as e:
                logger.error(f"❌ STREAMING ERROR: {e}")
                error_data = safe_serialize_error(e, "Streaming orchestrator")
                yield f"data: {json.dumps({'type': 'error', 'error': error_data['error_message']})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
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
