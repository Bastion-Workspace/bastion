"""
Agent-Specific Celery Tasks
Background processing for individual agents
Uses gRPC orchestrator for all agent processing
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import grpc

from config import settings
from services.celery_app import celery_app, update_task_progress, TaskStatus
from services.celery_tasks.async_runner import run_async

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

            acc: Dict[str, Any] = {
                "full_response": "",
                "agent_name": None,
                "status_messages": [],
            }

            async def _consume_stream() -> Dict[str, Any]:
                async for chunk in stub.StreamChat(grpc_request):
                    if chunk.type == "status":
                        acc["status_messages"].append(chunk.message)
                        if chunk.agent_name:
                            acc["agent_name"] = chunk.agent_name
                        logger.debug(f"📊 Status: {chunk.message}")

                    elif chunk.type == "content":
                        acc["full_response"] += chunk.message
                        if chunk.agent_name:
                            acc["agent_name"] = chunk.agent_name

                    elif chunk.type == "error":
                        logger.error(f"❌ gRPC orchestrator error: {chunk.message}")
                        return {
                            "success": False,
                            "error": chunk.message,
                            "message": "Background task processing failed",
                        }

                logger.info(
                    "✅ Background task received response from %s: %s chars",
                    acc["agent_name"] or "orchestrator",
                    len(acc["full_response"]),
                )
                return {
                    "success": True,
                    "response": acc["full_response"],
                    "agent_type": acc["agent_name"] or agent_type or "unknown",
                    "status_messages": acc["status_messages"],
                }

            try:
                return await asyncio.wait_for(
                    _consume_stream(),
                    timeout=settings.CELERY_ORCHESTRATOR_STREAM_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                cap = settings.CELERY_ORCHESTRATOR_STREAM_TIMEOUT_SEC
                plen = len(acc["full_response"])
                logger.error(
                    "Background task StreamChat exceeded %.0fs (partial response %s chars)",
                    cap,
                    plen,
                )
                return {
                    "success": False,
                    "error": f"orchestrator_stream_timeout_{int(cap)}s_partial_{plen}_chars",
                    "message": (
                        f"Orchestrator stream exceeded {cap}s; partial response was {plen} characters."
                    ),
                }
            
    except Exception as e:
        logger.error(f"❌ Background task gRPC orchestrator error: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Background task processing failed"
        }


async def _call_grpc_orchestrator_custom_agent(
    agent_profile_id: str,
    query: str,
    user_id: str,
    conversation_id: str = "",
    trigger_input: Optional[str] = None,
    extra_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Call gRPC orchestrator for a custom (Agent Factory) agent by profile ID.
    Used by scheduled agent execution, async agent-to-agent dispatch, and team post reactions.
    Routes to CustomAgentRunner via metadata. Optional trigger_input and extra_context (e.g. team_id, post_id).
    """
    if not GRPC_AVAILABLE:
        return {
            "success": False,
            "error": "gRPC orchestrator not available",
            "message": "Scheduled agent execution unavailable",
        }
    try:
        orchestrator_host = "llm-orchestrator"
        orchestrator_port = 50051
        options = [
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ]
        async with grpc.aio.insecure_channel(
            f"{orchestrator_host}:{orchestrator_port}", options=options
        ) as channel:
            stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)
            from services.grpc_context_gatherer import get_context_gatherer
            context_gatherer = get_context_gatherer()
            request_context = {"agent_profile_id": str(agent_profile_id)}
            if trigger_input is not None:
                request_context["trigger_input"] = trigger_input
            if extra_context:
                request_context.update(extra_context)
            grpc_request = await context_gatherer.build_chat_request(
                query=query,
                user_id=user_id,
                conversation_id=conversation_id,
                session_id="scheduled_agent",
                request_context=request_context,
                state=None,
                agent_type=None,
                routing_reason="Scheduled agent execution",
            )
            acc: Dict[str, Any] = {
                "full_response": "",
                "agent_name": None,
                "task_status": None,
                "approval_queue_id": None,
            }

            async def _consume_custom_stream() -> Dict[str, Any]:
                async for chunk in stub.StreamChat(grpc_request):
                    if chunk.type == "status" and chunk.agent_name:
                        acc["agent_name"] = chunk.agent_name
                    elif chunk.type == "content":
                        acc["full_response"] += chunk.message
                        if chunk.agent_name:
                            acc["agent_name"] = chunk.agent_name
                    elif chunk.type == "complete" and chunk.metadata:
                        acc["task_status"] = chunk.metadata.get("task_status") or acc["task_status"]
                        acc["approval_queue_id"] = (
                            chunk.metadata.get("approval_queue_id") or acc["approval_queue_id"]
                        )
                    elif chunk.type == "error":
                        return {
                            "success": False,
                            "error": chunk.message,
                            "message": "Scheduled agent processing failed",
                        }
                out: Dict[str, Any] = {
                    "success": True,
                    "response": acc["full_response"],
                    "agent_type": acc["agent_name"] or "custom_agent",
                }
                if acc["task_status"]:
                    out["task_status"] = acc["task_status"]
                if acc["approval_queue_id"]:
                    out["approval_queue_id"] = acc["approval_queue_id"]
                return out

            try:
                return await asyncio.wait_for(
                    _consume_custom_stream(),
                    timeout=settings.CELERY_ORCHESTRATOR_STREAM_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                cap = settings.CELERY_ORCHESTRATOR_STREAM_TIMEOUT_SEC
                plen = len(acc["full_response"])
                logger.error(
                    "Scheduled agent StreamChat exceeded %.0fs (partial response %s chars)",
                    cap,
                    plen,
                )
                return {
                    "success": False,
                    "error": f"orchestrator_stream_timeout_{int(cap)}s_partial_{plen}_chars",
                    "message": (
                        f"Orchestrator stream exceeded {cap}s; partial response was {plen} characters."
                    ),
                }
    except Exception as e:
        logger.error("Scheduled agent gRPC error: %s", e)
        return {
            "success": False,
            "error": str(e),
            "message": "Scheduled agent processing failed",
        }


@celery_app.task(bind=True, name="agents.dispatch_agent_invocation")
def dispatch_agent_invocation(
    self,
    agent_profile_id: str,
    input_content: str,
    user_id: str,
    source_agent_name: str = "",
    chain_depth: int = 0,
    chain_path_json: str = "[]",
    team_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Enqueue async agent-to-agent invocation. Runs the target agent with input_content
    as trigger_input. When team_id is provided, enforces team budget before running.
    """
    try:
        if team_id:
            from services import agent_line_service
            allowed, over_limit = run_async(agent_line_service.check_line_budget(team_id, user_id))
            if not allowed and over_limit:
                logger.warning("Agent invocation skipped: team %s budget limit reached", team_id)
                return {"success": False, "error": "Team monthly budget limit reached", "message": "Budget limit reached"}

        logger.info(
            "Dispatch agent invocation: profile_id=%s user_id=%s source=%s",
            agent_profile_id, user_id, source_agent_name or "unknown",
        )
        extra = {"team_id": team_id} if team_id else None
        team_name = None
        if team_id:
            try:
                from services import agent_line_service
                team = run_async(agent_line_service.get_line(team_id, user_id))
                team_name = (team.get("name", "Team") or "Team") if isinstance(team, dict) else "Team"
            except Exception:
                team_name = "Team"
        if team_id:
            try:
                from utils.websocket_manager import get_websocket_manager
                from datetime import datetime, timezone
                ws = get_websocket_manager()
                run_async(ws.send_line_timeline_update(
                    team_id,
                    {"type": "execution_status", "status": "running", "agent_id": agent_profile_id},
                ))
                run_async(ws.send_to_session({
                    "type": "team_execution_status",
                    "team_id": team_id,
                    "team_name": team_name or "Team",
                    "status": "running",
                    "agent_id": agent_profile_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, user_id))
            except Exception as ws_err:
                logger.debug("Send execution_status running failed: %s", ws_err)
        try:
            result = run_async(
                _call_grpc_orchestrator_custom_agent(
                    agent_profile_id=agent_profile_id,
                    query=input_content,
                    user_id=user_id,
                    conversation_id="",
                    trigger_input=input_content,
                    extra_context=extra,
                )
            )
            if team_id and result.get("success"):
                resp_text = (result.get("response") or "").strip()
                if resp_text:
                    try:
                        from services import agent_message_service
                        run_async(agent_message_service.create_message(
                            team_id=team_id,
                            from_agent_id=agent_profile_id,
                            to_agent_id=None,
                            message_type="report",
                            content=resp_text,
                            metadata={"trigger_type": "agent_invocation", "source": source_agent_name},
                            user_id=user_id,
                        ))
                    except Exception as post_err:
                        logger.warning("dispatch_agent_invocation: post response to timeline failed: %s", post_err)
            return result
        finally:
            if team_id:
                try:
                    from utils.websocket_manager import get_websocket_manager
                    from datetime import datetime, timezone
                    ws = get_websocket_manager()
                    run_async(ws.send_line_timeline_update(
                        team_id,
                        {"type": "execution_status", "status": "idle", "agent_id": agent_profile_id},
                    ))
                    run_async(ws.send_to_session({
                        "type": "team_execution_status",
                        "team_id": team_id,
                        "team_name": team_name or "Team",
                        "status": "idle",
                        "agent_id": agent_profile_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }, user_id))
                except Exception:
                    pass
    except Exception as e:
        logger.exception("dispatch_agent_invocation failed: %s", e)
        return {"success": False, "error": str(e), "message": "Agent invocation failed"}


@celery_app.task(bind=True, name="agents.dispatch_start_discussion")
def dispatch_start_discussion(
    self,
    team_id: str,
    user_id: str,
    initiator_profile_id: str,
    participant_handles: list,
    seed_message: str,
    moderator_handle: Optional[str] = None,
    max_turns: int = 10,
) -> Dict[str, Any]:
    """
    Start a multi-agent discussion by invoking the initiator agent with a prompt
    that instructs it to call start_agent_conversation. team_id in extra_context
    ensures team_tools (including start_agent_conversation) are available.
    """
    try:
        if team_id:
            from services import agent_line_service
            allowed, over_limit = run_async(agent_line_service.check_line_budget(team_id, user_id))
            if not allowed and over_limit:
                logger.warning("Start discussion skipped: team %s budget limit reached", team_id)
                return {"success": False, "error": "Team monthly budget limit reached", "message": "Budget limit reached"}

        participant_list = ", ".join(f"@{h}" for h in (participant_handles or []))
        moderator_clause = f'- moderator: @{moderator_handle}' if moderator_handle else "- moderator: none (omit or leave empty)"
        prompt = (
            "You are starting a group discussion. Call the start_agent_conversation tool with these parameters:\n"
            f"- participants: [{participant_list}]\n"
            f'- seed_message: "{seed_message}"\n'
            f"- max_turns: {max_turns}\n"
            f"{moderator_clause}\n\n"
            "Do not answer the topic yourself; only invoke the tool to start the conversation."
        )

        team_name = None
        if team_id:
            try:
                from services import agent_line_service
                team = run_async(agent_line_service.get_line(team_id, user_id))
                team_name = (team.get("name", "Team") or "Team") if isinstance(team, dict) else "Team"
            except Exception:
                team_name = "Team"
        try:
            from utils.websocket_manager import get_websocket_manager
            from datetime import datetime, timezone
            ws = get_websocket_manager()
            run_async(ws.send_line_timeline_update(
                team_id,
                {"type": "execution_status", "status": "running", "agent_id": initiator_profile_id},
            ))
            run_async(ws.send_to_session({
                "type": "team_execution_status",
                "team_id": team_id,
                "team_name": team_name or "Team",
                "status": "running",
                "agent_id": initiator_profile_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, user_id))
        except Exception as ws_err:
            logger.debug("Send execution_status running failed: %s", ws_err)

        try:
            result = run_async(
                _call_grpc_orchestrator_custom_agent(
                    agent_profile_id=initiator_profile_id,
                    query=prompt,
                    user_id=user_id,
                    conversation_id="",
                    trigger_input=prompt,
                    extra_context={"team_id": team_id},
                )
            )
            return result
        finally:
            if team_id:
                try:
                    from utils.websocket_manager import get_websocket_manager
                    from datetime import datetime, timezone
                    ws = get_websocket_manager()
                    run_async(ws.send_line_timeline_update(
                        team_id,
                        {"type": "execution_status", "status": "idle", "agent_id": initiator_profile_id},
                    ))
                    run_async(ws.send_to_session({
                        "type": "team_execution_status",
                        "team_id": team_id,
                        "team_name": team_name or "Team",
                        "status": "idle",
                        "agent_id": initiator_profile_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }, user_id))
                except Exception:
                    pass
    except Exception as e:
        logger.exception("dispatch_start_discussion failed: %s", e)
        return {"success": False, "error": str(e), "message": "Discussion start failed"}


def _get_redis():
    import os
    import redis
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    return redis.Redis.from_url(redis_url, decode_responses=True)


LOCK_TTL_SECONDS = 330
CONVERSATION_COOLDOWN_SECONDS = 60


@celery_app.task(bind=True, name="agents.dispatch_team_post_reaction")
def dispatch_team_post_reaction(
    self,
    agent_profile_id: str,
    team_id: str,
    post_id: str,
    post_content: str,
    post_author_id: str,
    user_id: str,
    respond_as: str = "comment",
) -> Dict[str, Any]:
    """
    Trigger a custom agent when a new team post is created (event-driven).
    Runs the agent with post_content as trigger_input and injects team_id, post_id
    so the output_router can post a comment via team_post destination.
    Uses Redis lock per (agent_profile_id, team_id) to prevent duplicate concurrent reactions.
    """
    try:
        logger.info(
            "Dispatch team post reaction: profile_id=%s team_id=%s post_id=%s",
            agent_profile_id, team_id, post_id,
        )
        redis_client = _get_redis()
        lock_key = f"team_reaction:{agent_profile_id}:{team_id}:lock"
        if not redis_client.set(lock_key, "1", nx=True, ex=LOCK_TTL_SECONDS):
            logger.warning("Team post reaction skipped: overlap for profile=%s team=%s", agent_profile_id, team_id)
            return {"success": False, "error": "Overlap: agent already reacting to this team"}
        try:
            extra_context = {
                "team_id": team_id,
                "post_id": post_id,
                "post_author_id": post_author_id,
                "respond_as": respond_as or "comment",
            }
            result = run_async(
                _call_grpc_orchestrator_custom_agent(
                    agent_profile_id=agent_profile_id,
                    query=post_content,
                    user_id=user_id,
                    conversation_id="",
                    trigger_input=post_content,
                    extra_context=extra_context,
                )
            )
            return result
        finally:
            try:
                redis_client.delete(lock_key)
            except Exception:
                pass
    except Exception as e:
        logger.exception("dispatch_team_post_reaction failed: %s", e)
        return {"success": False, "error": str(e), "message": "Team post reaction failed"}


def _handle_approved_hire(user_id: str, preview_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create agent profile and add to team for approved hire_agent governance."""
    from services import agent_line_service
    from services import agent_factory_service
    team_id = preview_data.get("line_id") or preview_data.get("team_id")
    proposed_name = preview_data.get("proposed_name") or "New agent"
    proposed_role = preview_data.get("proposed_role") or "worker"
    proposed_handle = preview_data.get("proposed_handle")
    if not team_id:
        return {"success": False, "error": "preview_data missing line_id"}
    team = run_async(agent_line_service.get_line(team_id, user_id))
    if not team:
        return {"success": False, "error": "Team not found"}
    profile_data = {
        "name": proposed_name,
        "handle": proposed_handle.strip() if proposed_handle and isinstance(proposed_handle, str) else None,
        "is_active": True,
    }
    try:
        profile = run_async(agent_factory_service.create_profile(user_id, profile_data))
    except Exception as e:
        logger.warning("handle_approved_hire: create_profile failed: %s", e)
        return {"success": False, "error": str(e)}
    new_profile_id = profile.get("id")
    if not new_profile_id:
        return {"success": False, "error": "Profile created but no id returned"}
    try:
        run_async(agent_line_service.add_member(team_id, user_id, new_profile_id, role=proposed_role))
    except Exception as e:
        logger.warning("handle_approved_hire: add_member failed: %s", e)
        return {"success": False, "error": str(e)}
    return {"success": True, "agent_profile_id": new_profile_id}


def _handle_approved_strategy_change(user_id: str, preview_data: Dict[str, Any]) -> Dict[str, Any]:
    """Log approved strategy change; optional future: apply playbook/goal updates."""
    logger.info(
        "Governance: strategy_change approved for user_id=%s preview_data=%s",
        user_id,
        preview_data,
    )
    return {"success": True}


@celery_app.task(bind=True, name="agents.resume_approved_agent")
def resume_approved_agent(self, approval_id: str, user_id: str) -> Dict[str, Any]:
    """
    Resume a parked approval: branch on governance_type.
    - playbook_step: call orchestrator to resume playbook from checkpoint.
    - hire_agent: create profile and add to team.
    - strategy_change: log and acknowledge.
    """
    try:
        from services.database_manager.celery_database_helpers import celery_fetch_one
        import uuid as uuid_mod
        import json as json_mod
        row = run_async(celery_fetch_one(
            "SELECT agent_profile_id, execution_id, playbook_config, governance_type, preview_data, status FROM agent_approval_queue WHERE id = $1 AND user_id = $2",
            uuid_mod.UUID(approval_id),
            user_id,
            rls_context={"user_id": user_id, "user_role": "user"},
        ))
        if not row:
            logger.warning("resume_approved_agent: approval not found %s", approval_id)
            return {"success": False, "error": "Approval not found"}
        if row.get("status") != "approved":
            logger.warning("resume_approved_agent: approval %s not approved (status=%s)", approval_id, row.get("status"))
            return {"success": False, "error": f"Approval not approved (status={row.get('status')})"}
        governance_type = (row.get("governance_type") or "playbook_step").strip()
        preview_data = row.get("preview_data") or {}
        if isinstance(preview_data, str):
            try:
                preview_data = json_mod.loads(preview_data)
            except Exception:
                preview_data = {}

        if governance_type == "hire_agent":
            return _handle_approved_hire(user_id, preview_data)
        if governance_type == "strategy_change":
            return _handle_approved_strategy_change(user_id, preview_data)

        agent_profile_id = str(row["agent_profile_id"]) if row.get("agent_profile_id") else None
        execution_id = str(row["execution_id"]) if row.get("execution_id") else None
        playbook_config = row.get("playbook_config")
        if not agent_profile_id or not playbook_config:
            return {"success": False, "error": "Missing agent_profile_id or playbook_config for playbook resume"}
        playbook_config_json = playbook_config if isinstance(playbook_config, str) else json_mod.dumps(playbook_config)
        extra_context = {
            "resume_approval_id": approval_id,
            "resume_playbook_config_json": playbook_config_json,
        }
        if execution_id:
            extra_context["execution_id"] = execution_id
        result = run_async(
            _call_grpc_orchestrator_custom_agent(
                agent_profile_id=agent_profile_id,
                query="yes",
                user_id=user_id,
                conversation_id="",
                extra_context=extra_context,
            )
        )
        if result.get("success") and execution_id:
            try:
                from services.celery_tasks.scheduled_agent_tasks import _update_execution_log_complete
                run_async(_update_execution_log_complete(execution_id, "completed", duration_ms=0))
            except Exception as upd_err:
                logger.warning("resume_approved_agent: update execution log failed: %s", upd_err)
        return result
    except Exception as e:
        logger.exception("resume_approved_agent failed: %s", e)
        return {"success": False, "error": str(e), "message": "Resume failed"}


@celery_app.task(bind=True, name="agents.dispatch_email_reaction")
def dispatch_email_reaction(
    self,
    agent_profile_id: str,
    user_id: str,
    email_subject: str,
    email_sender: str,
    email_body_preview: str,
    email_id: str,
    connection_id: int,
) -> Dict[str, Any]:
    """
    Trigger a custom agent when a new email matches a watch (poll-based).
    Uses Redis lock per (agent_profile_id, email_id) to prevent duplicate processing.
    """
    try:
        logger.info(
            "Dispatch email reaction: profile_id=%s email_id=%s",
            agent_profile_id, email_id,
        )
        redis_client = _get_redis()
        lock_key = f"email_reaction:{agent_profile_id}:{email_id}:lock"
        if not redis_client.set(lock_key, "1", nx=True, ex=LOCK_TTL_SECONDS):
            logger.warning(
                "Email reaction skipped: already processing profile=%s email=%s",
                agent_profile_id, email_id,
            )
            return {"success": False, "error": "Already processing this email"}
        try:
            trigger_text = (
                f"Subject: {email_subject or '(none)'}\n"
                f"From: {email_sender or '(unknown)'}\n\n"
                f"{email_body_preview or '(no preview)'}"
            )
            extra_context = {
                "email_id": email_id,
                "email_subject": email_subject or "",
                "email_sender": email_sender or "",
                "connection_id": connection_id,
            }
            result = run_async(
                _call_grpc_orchestrator_custom_agent(
                    agent_profile_id=agent_profile_id,
                    query=trigger_text,
                    user_id=user_id,
                    conversation_id="",
                    trigger_input=trigger_text,
                    extra_context=extra_context,
                )
            )
            return result
        finally:
            try:
                redis_client.delete(lock_key)
            except Exception:
                pass
    except Exception as e:
        logger.exception("dispatch_email_reaction failed: %s", e)
        return {"success": False, "error": str(e), "message": "Email reaction failed"}


@celery_app.task(bind=True, name="agents.dispatch_folder_file_reaction")
def dispatch_folder_file_reaction(
    self,
    agent_profile_id: str,
    user_id: str,
    document_id: str,
    filename: str,
    folder_id: str,
    folder_path: str,
    file_type: str,
) -> Dict[str, Any]:
    """
    Trigger a custom agent when a new file appears in a watched folder.
    Uses Redis lock per (agent_profile_id, document_id) to prevent duplicate processing.
    """
    try:
        logger.info(
            "Dispatch folder file reaction: profile_id=%s document_id=%s",
            agent_profile_id, document_id,
        )
        redis_client = _get_redis()
        lock_key = f"folder_reaction:{agent_profile_id}:{document_id}:lock"
        if not redis_client.set(lock_key, "1", nx=True, ex=LOCK_TTL_SECONDS):
            logger.warning(
                "Folder file reaction skipped: already processing profile=%s document=%s",
                agent_profile_id, document_id,
            )
            return {"success": False, "error": "Already processing this document"}
        try:
            trigger_text = f"New file: {filename} in {folder_path or folder_id or 'folder'}"
            extra_context = {
                "document_id": document_id,
                "filename": filename,
                "folder_id": folder_id,
                "folder_path": folder_path or "",
                "file_type": file_type or "",
            }
            result = run_async(
                _call_grpc_orchestrator_custom_agent(
                    agent_profile_id=agent_profile_id,
                    query=trigger_text,
                    user_id=user_id,
                    conversation_id="",
                    trigger_input=trigger_text,
                    extra_context=extra_context,
                )
            )
            return result
        finally:
            try:
                redis_client.delete(lock_key)
            except Exception:
                pass
    except Exception as e:
        logger.exception("dispatch_folder_file_reaction failed: %s", e)
        return {"success": False, "error": str(e), "message": "Folder file reaction failed"}


async def _get_agent_bot_user_id(agent_profile_id: str) -> Optional[str]:
    """Look up the bot_user_id for an agent profile."""
    from services.database_manager.database_helpers import fetch_one
    from utils.grpc_rls import grpc_admin_rls as _bot_lookup_rls
    row = await fetch_one(
        "SELECT bot_user_id FROM agent_profiles WHERE id = $1::uuid",
        agent_profile_id,
        rls_context=_bot_lookup_rls(),
    )
    return row["bot_user_id"] if row and row.get("bot_user_id") else None


async def _load_room_history_context(room_id: str, limit: int = 20) -> str:
    """Load recent room messages for agent context."""
    from services.messaging.messaging_service import MessagingService
    svc = MessagingService()
    try:
        msgs = await svc.get_room_messages(room_id=room_id, user_id="system", limit=limit)
        if not msgs:
            return ""
        lines = []
        for m in msgs:
            sender = m.get("display_name") or m.get("username") or m.get("sender_id", "Unknown")
            lines.append(f"[{sender}]: {m.get('content', '')}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning("Failed to load room history for context: %s", e)
        return ""


async def _post_agent_response_to_room(
    agent_profile_id: str,
    room_id: str,
    response_text: str,
) -> None:
    """Write the agent's response back to the chat room and broadcast via WebSocket."""
    bot_user_id = await _get_agent_bot_user_id(agent_profile_id)
    if not bot_user_id:
        logger.warning("No bot_user_id for agent %s, cannot post to room", agent_profile_id)
        return
    from services.messaging.messaging_service import MessagingService
    svc = MessagingService()
    msg = await svc.send_message(
        room_id=room_id,
        sender_id=bot_user_id,
        content=response_text,
        message_type="text",
        metadata={"from_agent_profile_id": str(agent_profile_id)},
    )
    if msg:
        from utils.websocket_manager import get_websocket_manager
        ws_mgr = get_websocket_manager()
        await ws_mgr.broadcast_to_room(
            room_id=room_id,
            message={"type": "new_message", "message": msg},
            exclude_user_id=None,
        )
        logger.info("Agent %s posted response to room %s", agent_profile_id, room_id)


@celery_app.task(bind=True, name="agents.dispatch_conversation_reaction")
def dispatch_conversation_reaction(
    self,
    agent_profile_id: str,
    user_id: str,
    message_content: str,
    message_sender: str,
    watch_type: str,
    conversation_id: Optional[str] = None,
    room_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Trigger a custom agent when a new message appears in a watched AI conversation or chat room.
    Uses Redis cooldown lock per (agent_profile_id, target_id) to avoid rapid re-triggering.
    For chat_room watches, loads room history as context and posts the response back to the room.
    """
    target_id = conversation_id or room_id or ""
    try:
        logger.info(
            "Dispatch conversation reaction: profile_id=%s watch_type=%s target=%s",
            agent_profile_id, watch_type, target_id,
        )
        redis_client = _get_redis()
        lock_key = f"conversation_reaction:{agent_profile_id}:{target_id}:lock"
        if not redis_client.set(lock_key, "1", nx=True, ex=CONVERSATION_COOLDOWN_SECONDS):
            logger.warning(
                "Conversation reaction skipped: cooldown profile=%s target=%s",
                agent_profile_id, target_id,
            )
            return {"success": False, "error": "Cooldown: wait before re-triggering"}
        try:
            extra_context = {
                "watch_type": watch_type,
                "message_sender": message_sender or "",
            }
            if conversation_id:
                extra_context["conversation_id"] = conversation_id
            if room_id:
                extra_context["room_id"] = room_id
                room_history = run_async(_load_room_history_context(room_id, limit=20))
                if room_history:
                    extra_context["room_history"] = room_history

            result = run_async(
                _call_grpc_orchestrator_custom_agent(
                    agent_profile_id=agent_profile_id,
                    query=message_content,
                    user_id=user_id,
                    conversation_id=conversation_id or "",
                    trigger_input=message_content,
                    extra_context=extra_context,
                )
            )

            if room_id and result.get("success") and result.get("response"):
                run_async(_post_agent_response_to_room(
                    agent_profile_id=agent_profile_id,
                    room_id=room_id,
                    response_text=result["response"],
                ))

            return result
        finally:
            try:
                redis_client.delete(lock_key)
            except Exception:
                pass
    except Exception as e:
        logger.exception("dispatch_conversation_reaction failed: %s", e)
        return {"success": False, "error": str(e), "message": "Conversation reaction failed"}


def _parse_email_datetime(received_str: Optional[str]):
    """Parse ISO-ish received_datetime string to timezone-aware datetime."""
    if not received_str:
        return None
    from datetime import timezone
    try:
        s = received_str.replace("Z", "+00:00").strip()
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


async def _async_poll_watched_emails() -> Dict[str, Any]:
    """Fetch new emails for all active email watches, apply filters, dispatch reactions."""
    from datetime import timezone
    from services.database_manager.celery_database_helpers import celery_fetch_all, celery_execute
    from utils.grpc_rls import grpc_admin_rls as _email_poll_admin_rls

    try:
        watches = await celery_fetch_all(
            "SELECT agent_profile_id, connection_id, user_id, is_active, subject_pattern, "
            "sender_pattern, folder, last_checked_at FROM agent_email_watches WHERE is_active = true",
            rls_context=_email_poll_admin_rls(),
        )
        if not watches:
            return {"processed": 0, "dispatched": 0}
        key = lambda w: (w["user_id"], w["connection_id"], (w.get("folder") or "Inbox").strip().lower())
        groups = {}
        for w in watches:
            k = key(w)
            if k not in groups:
                groups[k] = []
            groups[k].append(w)
        from clients.connections_service_client import get_connections_service_client
        client = await get_connections_service_client()
        dispatched = 0
        for (user_id, connection_id, folder), group_watches in groups.items():
            rls = {"user_id": str(user_id)}
            try:
                result = await client.get_emails(
                    user_id=str(user_id),
                    connection_id=int(connection_id),
                    folder_id=folder or "inbox",
                    top=100,
                    unread_only=False,
                    rls_context=rls,
                )
            except Exception as e:
                logger.warning("Poll emails failed for user=%s connection=%s: %s", user_id, connection_id, e)
                continue
            messages = result.get("messages") or []
            for msg in messages:
                received_str = msg.get("received_datetime")
                received_dt = _parse_email_datetime(received_str)
                for w in group_watches:
                    last_checked = w.get("last_checked_at")
                    if last_checked and received_dt and received_dt.tzinfo is None:
                        received_dt = received_dt.replace(tzinfo=timezone.utc)
                    if last_checked and received_dt and received_dt < last_checked:
                        continue
                    subj = (msg.get("subject") or "").lower()
                    sender = (msg.get("from_address") or msg.get("from_name") or "").lower()
                    sp = (w.get("subject_pattern") or "").strip().lower()
                    if sp and sp not in subj:
                        continue
                    sr = (w.get("sender_pattern") or "").strip().lower()
                    if sr and sr not in sender:
                        continue
                    dispatch_email_reaction.delay(
                        agent_profile_id=str(w["agent_profile_id"]),
                        user_id=str(w["user_id"]),
                        email_subject=msg.get("subject") or "",
                        email_sender=msg.get("from_address") or msg.get("from_name") or "",
                        email_body_preview=(msg.get("body_preview") or "")[:500],
                        email_id=str(msg.get("id") or ""),
                        connection_id=int(w["connection_id"]),
                    )
                    dispatched += 1
            for w in group_watches:
                await celery_execute(
                    "UPDATE agent_email_watches SET last_checked_at = NOW() "
                    "WHERE agent_profile_id = $1::uuid AND connection_id = $2",
                    w["agent_profile_id"],
                    w["connection_id"],
                    rls_context={"user_id": str(w["user_id"]), "user_role": "user"},
                )
        return {"processed": len(watches), "dispatched": dispatched}
    except Exception as e:
        logger.exception("poll_watched_emails failed: %s", e)
        return {"processed": 0, "dispatched": 0, "error": str(e)}


@celery_app.task(bind=True)
def poll_watched_emails(self) -> Dict[str, Any]:
    """Celery Beat: poll email watches every 5 min, fetch new emails, dispatch reactions."""
    try:
        return run_async(_async_poll_watched_emails())
    except Exception as e:
        logger.exception("poll_watched_emails failed: %s", e)
        return {"processed": 0, "dispatched": 0, "error": str(e)}


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
        result = run_async(_async_research_processing(
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
        
        result = run_async(_async_coding_processing(
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
                result = run_async(_async_research_processing(
                    self, user_id, conversation_id, query, persona
                ))
            elif agent_type == "coding":
                result = run_async(_async_coding_processing(
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
