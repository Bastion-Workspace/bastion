"""
gRPC Orchestrator Proxy - Forwards requests to LLM Orchestrator microservice
Phase 5: Integration endpoint for new microservices architecture
"""

import logging
import asyncio
import json
import uuid
from typing import AsyncIterator, Dict, Any, Optional
from datetime import datetime

import grpc
from fastapi import APIRouter, HTTPException, Depends, Request
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
    agent_profile_id: str = None  # Agent Factory: route to custom agent when set
    is_branch_resend: bool = False  # User row already created by branch API
    branch_message_id: str = None  # That user message_id
    code_workspace_id: str = None  # Code Space UUID for agent rules / device context


async def stream_from_grpc_orchestrator(
    query: str,
    conversation_id: str,
    user_id: str,
    session_id: str = "default",
    agent_type: str = None,
    routing_reason: str = None,
    request_context: Dict[str, Any] = None,
    state: Dict[str, Any] = None,
    is_branch_resend: bool = False,
    branch_message_id: str = None,
    client_run_id: Optional[str] = None,
    originating_surface_id: Optional[str] = None,
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
            
            rc = dict(request_context or {})
            if (
                is_branch_resend
                and branch_message_id
                and conversation_id
                and str(conversation_id).strip()
            ):
                from services.conversation_service import ConversationService as _CS

                _cs = _CS()
                _cs.set_current_user(user_id)
                path = await _cs.get_active_path_messages(conversation_id, user_id)
                history = path[:-1] if len(path) > 1 else []
                tid = await _cs.get_branch_thread_id(conversation_id, user_id)
                if tid.get("thread_id_suffix"):
                    rc["branch_thread_suffix"] = tid["thread_id_suffix"]
                rc["active_path_messages"] = history
                request_context = rc

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
                routing_reason=routing_reason,
            )

            request_persona_ai_name = (grpc_request.persona.ai_name or "").strip()

            run_id = (client_run_id or grpc_request.metadata.get("client_run_id") or "").strip() or str(
                uuid.uuid4()
            )
            if not grpc_request.metadata.get("client_run_id"):
                grpc_request.metadata["client_run_id"] = run_id
            yield format_sse_message({"type": "run_started", "run_id": run_id})

            logger.info("Forwarding to gRPC orchestrator run_id=%s query=%s...", run_id, query[:80])

            # Agent Factory custom agent runs: skip persistence only when there is no conversation
            # (e.g. scheduled/headless run). When the user is chatting with a custom agent in the UI,
            # we have a conversation_id and must persist so conversation history (and tool_call_summary
            # for "pending operations") is available on the next turn.
            is_custom_agent_request = bool(request_context and request_context.get("agent_profile_id"))
            persist_conversation = True
            try:
                if isinstance(rc, dict) and rc.get("persist_conversation") is False:
                    persist_conversation = False
            except Exception:
                persist_conversation = True
            skip_persistence = (not persist_conversation) or (
                is_custom_agent_request and not (conversation_id and str(conversation_id).strip())
            )
            if skip_persistence:
                logger.info("Skipping conversation persistence (persist_conversation=%s)", persist_conversation)
            
            # Initialize title_updated flag before try block to avoid UnboundLocalError
            title_updated = False
            
            # Save user message to conversation BEFORE processing (skip only when no conversation_id)
            # Branch edit-and-resend persists the user message via POST .../messages/{id}/branch first.
            if not skip_persistence and not (
                is_branch_resend and branch_message_id and str(branch_message_id).strip()
            ):
                try:
                    from services.conversation_service import ConversationService
                    conversation_service = ConversationService()
                    conversation_service.set_current_user(user_id)

                    saved_user = await conversation_service.add_message(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        role="user",
                        content=query,
                        metadata={"orchestrator_system": True, "streaming": True},
                    )
                    persisted_user_message_id = (saved_user or {}).get("message_id")
                    if persisted_user_message_id:
                        logger.info(
                            "Saved user message to conversation %s message_id=%s",
                            conversation_id,
                            persisted_user_message_id,
                        )
                        yield format_sse_message(
                            {
                                "type": "user_message_persisted",
                                "message_id": persisted_user_message_id,
                                "conversation_id": conversation_id,
                            }
                        )
                    else:
                        logger.info("Saved user message to conversation %s", conversation_id)

                    # Check if title was updated (first message triggers title generation)
                    # The add_message method updates the title if it was "New Conversation"
                    # We'll emit a conversation_updated event after streaming completes
                    title_updated = True
                except Exception as save_error:
                    logger.warning("Failed to save user message: %s", save_error)
                    # Continue even if message save fails
            
            # Stream chunks from gRPC service (poll Redis so Stop cancels without relying on fetch alone)
            from services.stream_run_cancel import is_stream_cancel_requested

            chunk_count = 0
            agent_name_used = None
            accumulated_response = ""
            metadata_received = {}
            user_cancelled = False
            persisted_user_message_id: Optional[str] = None
            queue: asyncio.Queue = asyncio.Queue(maxsize=128)

            async def _grpc_pump():
                try:
                    async for chunk in stub.StreamChat(grpc_request):
                        await queue.put(("c", chunk))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await queue.put(("e", exc))
                finally:
                    await queue.put(("x", None))

            pump_task = asyncio.create_task(_grpc_pump())
            pump_exc = None
            try:
                while True:
                    try:
                        kind, payload = await asyncio.wait_for(queue.get(), timeout=0.25)
                    except asyncio.TimeoutError:
                        if await is_stream_cancel_requested(run_id):
                            user_cancelled = True
                            pump_task.cancel()
                            try:
                                await pump_task
                            except asyncio.CancelledError:
                                pass
                            yield format_sse_message(
                                {"type": "cancelled", "message": "Stopped by user", "run_id": run_id}
                            )
                            break
                        continue
                    if kind == "x":
                        break
                    if kind == "e":
                        pump_exc = payload
                        break
                    chunk = payload
                    chunk_count += 1

                    if chunk.agent_name and chunk.agent_name not in ["orchestrator", "system"]:
                        agent_name_used = chunk.agent_name

                    if chunk.metadata:
                        metadata_received.update(dict(chunk.metadata))

                    if chunk.type == "content" and chunk.message:
                        accumulated_response += chunk.message

                    if chunk.type == "status":
                        status_sse = {
                            "type": "status",
                            "message": chunk.message,
                            "content": chunk.message,
                            "agent": chunk.agent_name,
                            "timestamp": chunk.timestamp,
                        }
                        if chunk.metadata:
                            for mk, mv in chunk.metadata.items():
                                if mv is None:
                                    continue
                                status_sse[str(mk)] = str(mv)
                        pan = (chunk.metadata or {}).get("persona_ai_name") if chunk.metadata else None
                        pan = (pan or "").strip() or None
                        if not pan and request_persona_ai_name and not is_custom_agent_request:
                            pan = request_persona_ai_name
                        if pan:
                            status_sse["persona_ai_name"] = pan
                        yield format_sse_message(status_sse)

                    elif chunk.type == "content":
                        content_sse = {
                            "type": "content",
                            "content": chunk.message,
                            "agent": chunk.agent_name,
                        }
                        if chunk.metadata and chunk.metadata.get("agent_display_name"):
                            content_sse["agent_display_name"] = chunk.metadata["agent_display_name"]
                        pan_c = (chunk.metadata or {}).get("persona_ai_name") if chunk.metadata else None
                        pan_c = (pan_c or "").strip() or None
                        if not pan_c and request_persona_ai_name and not is_custom_agent_request:
                            pan_c = request_persona_ai_name
                        if pan_c:
                            content_sse["persona_ai_name"] = pan_c
                        yield format_sse_message(content_sse)

                    elif chunk.type == "permission_request":
                        try:
                            payload = json.loads(chunk.message) if chunk.message else {}
                        except (json.JSONDecodeError, TypeError):
                            payload = {"prompt": chunk.message or "Approve to continue?"}
                        sse_msg = {
                            "type": "permission_request",
                            "requires_approval": True,
                            "step_name": payload.get("step_name", ""),
                            "prompt": payload.get("prompt", "Approve to continue?"),
                            "content": chunk.message,
                        }
                        if payload.get("pending_auth"):
                            sse_msg["pending_auth"] = payload["pending_auth"]
                        if payload.get("interaction_type"):
                            sse_msg["interaction_type"] = payload["interaction_type"]
                        if payload.get("approval_id"):
                            sse_msg["approval_id"] = payload["approval_id"]
                        if payload.get("command") is not None:
                            sse_msg["command"] = payload["command"]
                        yield format_sse_message(sse_msg)

                    elif chunk.type == "complete":
                        yield format_sse_message(
                            {
                                "type": "complete",
                                "content": chunk.message,
                                "agent": chunk.agent_name,
                                "metadata": dict(chunk.metadata) if chunk.metadata else {},
                            }
                        )

                    elif chunk.type == "error":
                        yield format_sse_message(
                            {
                                "type": "error",
                                "content": chunk.message,
                                "message": chunk.message,
                                "agent": chunk.agent_name,
                            }
                        )

                    elif chunk.type == "title":
                        title_updated = True
                        yield format_sse_message(
                            {
                                "type": "title",
                                "message": chunk.message,
                                "timestamp": chunk.timestamp,
                                "agent": chunk.agent_name,
                            }
                        )
                        logger.info("Forwarded title chunk to frontend: %s", chunk.message)

                    elif chunk.type == "notification":
                        notification_metadata = dict(chunk.metadata) if chunk.metadata else {}
                        browser_notify = notification_metadata.get("browser_notify", "false").lower() == "true"
                        yield format_sse_message(
                            {
                                "type": "notification",
                                "message": chunk.message,
                                "severity": notification_metadata.get("severity", "info"),
                                "temporary": notification_metadata.get("temporary", "false").lower() == "true",
                                "timestamp": chunk.timestamp,
                                "agent": chunk.agent_name,
                                "browser_notify": browser_notify,
                                "metadata": notification_metadata,
                            }
                        )
                        logger.info(
                            "Forwarded notification chunk: %s",
                            chunk.message,
                        )

                    await asyncio.sleep(0)
            finally:
                if not pump_task.done():
                    pump_task.cancel()
                    try:
                        await pump_task
                    except (asyncio.CancelledError, Exception):
                        pass

            if pump_exc is not None:
                raise pump_exc

            logger.info(
                "gRPC stream finished run_id=%s chunks=%s user_cancelled=%s",
                run_id,
                chunk_count,
                user_cancelled,
            )

            if user_cancelled and persisted_user_message_id and conversation_id and user_id:
                try:
                    from services.conversation_service import ConversationService

                    _del_cs = ConversationService()
                    _del_cs.set_current_user(user_id)
                    deleted = await _del_cs.delete_conversation_message(
                        conversation_id, user_id, persisted_user_message_id
                    )
                    if deleted:
                        logger.info(
                            "Removed cancelled stream user message %s from conversation %s",
                            persisted_user_message_id,
                            conversation_id,
                        )
                    else:
                        logger.warning(
                            "Could not delete cancelled user message %s (conversation %s)",
                            persisted_user_message_id,
                            conversation_id,
                        )
                except Exception as del_err:
                    logger.warning("Failed to delete cancelled user message: %s", del_err)
            
            # Save assistant response to conversation AFTER streaming completes
            if accumulated_response and not skip_persistence and not user_cancelled:
                try:
                    from utils.message_sanitizer import strip_tool_actions_prefix
                    accumulated_response = strip_tool_actions_prefix(accumulated_response)
                    from services.conversation_service import ConversationService
                    conversation_service = ConversationService()
                    conversation_service.set_current_user(user_id)
                    
                    # Build metadata: base fields + allowlisted keys (strip data-URI images)
                    from utils.history_metadata import (
                        filter_history_safe_metadata,
                        sanitize_images_for_persistence,
                    )
                    persist_metadata = dict(metadata_received)
                    sanitize_images_for_persistence(persist_metadata)
                    safe_meta = filter_history_safe_metadata(persist_metadata)
                    metadata = {
                        "orchestrator_system": True,
                        "streaming": True,
                        "delegated_agent": agent_name_used or "unknown",
                        "chunk_count": chunk_count,
                        **safe_meta,
                    }
                    message_branch_id = None
                    if is_branch_resend and branch_message_id and str(branch_message_id).strip():
                        ref_row = await conversation_service.get_message_by_id(
                            conversation_id, user_id, branch_message_id
                        )
                        if ref_row:
                            message_branch_id = ref_row.get("branch_id")

                    await conversation_service.add_message(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        role="assistant",
                        content=accumulated_response,
                        metadata=metadata,
                        message_branch_id=message_branch_id,
                    )
                    logger.info(f"✅ Saved assistant response to conversation {conversation_id} (agent: {agent_name_used or 'unknown'})")

                    title_for_notify = None
                    try:
                        conv_row = await conversation_service.get_conversation(conversation_id, user_id)
                        if conv_row:
                            title_for_notify = conv_row.get("title")
                    except Exception:
                        pass
                    from services.chat_completion_notifier import (
                        notify_chat_reply_ready,
                        resolve_display_agent_name,
                    )

                    display_agent = await resolve_display_agent_name(
                        user_id, metadata_received, agent_name_used
                    )
                    await notify_chat_reply_ready(
                        user_id,
                        conversation_id,
                        response_text=accumulated_response,
                        agent_name=display_agent,
                        conversation_title=title_for_notify,
                        originating_surface_id=originating_surface_id,
                    )

                    # Record skill execution metrics (fire-and-forget)
                    _raw_skill_events = metadata_received.get("skill_execution_events")
                    if _raw_skill_events:
                        try:
                            import json as _json
                            _events = _json.loads(_raw_skill_events) if isinstance(_raw_skill_events, str) else _raw_skill_events
                            if isinstance(_events, list) and _events:
                                from services.agent_skills_service import record_skill_execution_events
                                _profile_id = metadata_received.get("agent_profile_id")
                                _inserted = await record_skill_execution_events(
                                    events=_events,
                                    user_id=user_id,
                                    agent_profile_id=_profile_id,
                                )
                                if _inserted:
                                    logger.info("Recorded %d skill execution event(s)", _inserted)
                        except Exception as _skill_evt_err:
                            logger.debug("Skill execution event recording skipped: %s", _skill_evt_err)

                    # Save agent routing metadata to backend DB (source of truth)
                    if agent_name_used:
                        try:
                            if metadata_received.get("line_dispatch_session") == "true":
                                await conversation_service.update_agent_metadata(
                                    conversation_id=conversation_id,
                                    user_id=user_id,
                                    primary_agent_selected="line_dispatch",
                                    last_agent=agent_name_used or "line_dispatch",
                                    clear_agent_profile_id=True,
                                )
                                logger.info("Saved agent metadata for line dispatch session (no sticky CEO profile)")

                                # Persist @team chat as a durable directive in the team timeline and
                                # optionally wake an idle team by enqueuing an ad-hoc heartbeat.
                                try:
                                    _line_id = metadata_received.get("line_id") or metadata_received.get("team_id")
                                    _ceo_id = metadata_received.get("agent_profile_id") or metadata_received.get(
                                        "ceo_profile_id"
                                    )
                                    if _line_id and (query or "").strip():
                                        from services.celery_tasks.chat_directive_task import (
                                            post_chat_directive_to_timeline,
                                        )

                                        post_chat_directive_to_timeline.apply_async(
                                            kwargs={
                                                "line_id": str(_line_id),
                                                "user_id": user_id,
                                                "user_message": query,
                                                "leader_response": (accumulated_response or "")[:2000],
                                                "conversation_id": conversation_id or "",
                                                "ceo_profile_id": str(_ceo_id or ""),
                                            },
                                            countdown=2,
                                        )
                                except Exception as directive_err:
                                    logger.debug("Line dispatch directive injection skipped: %s", directive_err)
                            else:
                                agent_profile_id_saved = metadata_received.get("agent_profile_id")
                                await conversation_service.update_agent_metadata(
                                    conversation_id=conversation_id,
                                    user_id=user_id,
                                    primary_agent_selected=agent_name_used,
                                    last_agent=agent_name_used,
                                    agent_profile_id=agent_profile_id_saved,
                                )
                                logger.info(f"✅ Saved agent metadata to backend DB: {agent_name_used}")
                        except Exception as agent_save_error:
                            logger.warning(f"⚠️ Failed to save agent metadata: {agent_save_error}")
                    
                    # Session-level memory: needs_session_summary is set in add_message; idle beat runs post_session_analysis
                    
                except Exception as save_error:
                    logger.warning(f"⚠️ Failed to save assistant response: {save_error}")
                    # Continue even if message save fails
            
            # NOTE: gRPC orchestrator handles its own state management via LangGraph checkpointing
            # No need to manually update backend orchestrator state
            
            # Send final complete event with conversation update flag (skip refresh for Agent Factory runs)
            done_payload: Dict[str, Any] = {
                'type': 'done',
                'conversation_id': conversation_id,
                'conversation_updated': not skip_persistence,
            }
            if conversation_id and user_id:
                try:
                    from services.conversation_service import ConversationService

                    _cs = ConversationService()
                    _am = await _cs.get_agent_metadata(conversation_id, user_id)
                    if _am.get("active_line_id"):
                        done_payload["active_line_id"] = _am["active_line_id"]
                    if _am.get("active_line_name"):
                        done_payload["active_line_name"] = _am["active_line_name"]
                except Exception as done_meta_err:
                    logger.debug("done event line metadata skipped: %s", done_meta_err)
            yield format_sse_message(done_payload)
            
    except grpc.RpcError as e:
        logger.error(f"gRPC error: {e.code()} - {e.details()}")
        yield format_sse_message({
            'type': 'error',
            'content': f"gRPC Orchestrator Error: {e.details()}",
            'message': f"gRPC Orchestrator Error: {e.details()}"
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
            'content': f"Orchestrator error: {str(e)}",
            'message': f"Orchestrator error: {str(e)}"
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
    http_request: Request,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
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
        
        # Always use authenticated user for security (ignore client-supplied user_id)
        target_user_id = current_user.user_id
        
        # Build request context from frontend fields
        request_context = {
            "active_editor": request.active_editor,
            "editor_preference": request.editor_preference,
            "pipeline_preference": request.pipeline_preference,
            "active_pipeline_id": request.active_pipeline_id,
            "locked_agent": request.locked_agent,
            "base_checkpoint_id": request.base_checkpoint_id,
            "agent_profile_id": request.agent_profile_id,
            "code_workspace_id": request.code_workspace_id,
        }

        # Remove None values
        request_context = {k: v for k, v in request_context.items() if v is not None}

        stream_run_id = str(uuid.uuid4())
        request_context["client_run_id"] = stream_run_id

        surface_hdr = (http_request.headers.get("x-surface-id") or "").strip() or None

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
                state=conversation_state,
                is_branch_resend=bool(getattr(request, "is_branch_resend", False)),
                branch_message_id=getattr(request, "branch_message_id", None),
                client_run_id=stream_run_id,
                originating_surface_id=surface_hdr,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Run-Id": stream_run_id,
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

