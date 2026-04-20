"""gRPC handlers for Agent messaging, notifications, tasks, workspace, and device operations."""

import json
import logging
from typing import Any, Dict

import grpc
from protos import tool_service_pb2

from services.grpc_handlers._utils import json_default

logger = logging.getLogger(__name__)


class AgentMessagingHandlersMixin:
    """Mixin providing Agent messaging, notification, task, workspace, and device gRPC handlers.

    Mixed into ToolServiceImplementation; accesses self.* helpers
    via standard Python MRO.
    """

    # ===== Agent-Initiated Notifications =====

    async def SendOutboundMessage(
        self,
        request: tool_service_pb2.SendOutboundMessageRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SendOutboundMessageResponse:
        """Send a proactive outbound message via a messaging bot (Telegram, Discord)."""
        try:
            from clients.connections_service_client import get_connections_service_client
            client = await get_connections_service_client()
            result = await client.send_outbound_message(
                user_id=request.user_id or "system",
                provider=request.provider or "",
                connection_id=request.connection_id or "",
                message=request.message or "",
                format=request.format or "markdown",
                recipient_chat_id=getattr(request, "recipient_chat_id", None) or "",
            )
            return tool_service_pb2.SendOutboundMessageResponse(
                success=result.get("success", False),
                message_id=result.get("message_id", ""),
                channel=result.get("channel", ""),
                error=result.get("error", ""),
            )
        except Exception as e:
            logger.error("SendOutboundMessage failed: %s", e)
            return tool_service_pb2.SendOutboundMessageResponse(
                success=False, message_id="", channel="", error=str(e)
            )

    async def CreateAgentConversation(
        self,
        request: tool_service_pb2.CreateAgentConversationRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateAgentConversationResponse:
        """Create or append to an agent-initiated conversation via backend API (ensures WebSocket events fire)."""
        import os

        user_id = request.user_id or "system"
        msg = (request.message or "").strip()
        if not msg:
            return tool_service_pb2.CreateAgentConversationResponse(
                success=False, conversation_id="", message_id="", error="Message is required"
            )

        backend_url = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
        internal_key = os.getenv("INTERNAL_SERVICE_KEY", "")
        if not internal_key:
            logger.warning("CreateAgentConversation: INTERNAL_SERVICE_KEY not set; backend may reject request")

        payload = {
            "user_id": user_id,
            "message": msg,
            "agent_name": request.agent_name or "",
            "agent_profile_id": request.agent_profile_id or "",
            "title": request.title or "",
            "conversation_id": request.conversation_id or "",
        }
        url = f"{backend_url}/api/internal/agent-conversation"
        headers = {"Content-Type": "application/json"}
        if internal_key:
            headers["X-Internal-Service-Key"] = internal_key

        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                err = resp.text or f"HTTP {resp.status_code}"
                logger.error("CreateAgentConversation backend call failed: %s", err)
                return tool_service_pb2.CreateAgentConversationResponse(
                    success=False, conversation_id="", message_id="", error=err[:500]
                )
            data = resp.json()
            return tool_service_pb2.CreateAgentConversationResponse(
                success=True,
                conversation_id=data.get("conversation_id", ""),
                message_id=data.get("message_id", ""),
                error="",
            )
        except Exception as e:
            logger.error("CreateAgentConversation failed: %s", e)
            return tool_service_pb2.CreateAgentConversationResponse(
                success=False, conversation_id="", message_id="", error=str(e)[:500]
            )

    async def UpdateConversationTitle(
        self,
        request: tool_service_pb2.UpdateConversationTitleRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateConversationTitleResponse:
        """Persist orchestrator-generated or placeholder conversation title (owner-scoped)."""
        from services.database_manager.database_helpers import fetch_one, execute
        from utils.grpc_rls import grpc_user_rls as _grpc_rls

        user_id = (request.user_id or "").strip()
        conversation_id = (request.conversation_id or "").strip()
        title = (request.title or "").strip()
        if not user_id or not conversation_id:
            return tool_service_pb2.UpdateConversationTitleResponse(
                success=False,
                conversation_id=conversation_id,
                title="",
                message="Missing user_id or conversation_id",
                error="user_id and conversation_id are required",
            )
        if not title:
            return tool_service_pb2.UpdateConversationTitleResponse(
                success=False,
                conversation_id=conversation_id,
                title="",
                message="Empty title",
                error="title is required",
            )
        title_stored = title[:500]
        ctx = _grpc_rls(user_id)
        try:
            row = await fetch_one(
                "SELECT conversation_id FROM conversations WHERE conversation_id = $1 AND user_id = $2",
                conversation_id,
                user_id,
                rls_context=ctx,
            )
            if not row:
                return tool_service_pb2.UpdateConversationTitleResponse(
                    success=False,
                    conversation_id=conversation_id,
                    title="",
                    message="Conversation not found",
                    error="Conversation not found or access denied",
                )
            await execute(
                "UPDATE conversations SET title = $1, updated_at = NOW() "
                "WHERE conversation_id = $2 AND user_id = $3",
                title_stored,
                conversation_id,
                user_id,
                rls_context=ctx,
            )
            return tool_service_pb2.UpdateConversationTitleResponse(
                success=True,
                conversation_id=conversation_id,
                title=title_stored,
                message="Title updated",
            )
        except Exception as e:
            logger.exception("UpdateConversationTitle failed")
            return tool_service_pb2.UpdateConversationTitleResponse(
                success=False,
                conversation_id=conversation_id,
                title="",
                message="Update failed",
                error=str(e)[:500],
            )

    async def CreateAgentMessage(
        self,
        request: tool_service_pb2.CreateAgentMessageRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateAgentMessageResponse:
        """Create an inter-agent message (team timeline)."""
        import json

        try:
            from services import agent_message_service

            user_id = request.user_id or "system"
            team_id = (request.team_id or "").strip()
            if not team_id:
                return tool_service_pb2.CreateAgentMessageResponse(
                    success=False, message_id="", message_json="", error="team_id is required"
                )
            from_agent_id = (request.from_agent_id or "").strip() or None
            to_agent_id = (request.to_agent_id or "").strip() or None
            message_type = (request.message_type or "report").strip()
            content = request.content or ""
            metadata = {}
            if request.metadata_json:
                try:
                    metadata = json.loads(request.metadata_json)
                except (json.JSONDecodeError, TypeError):
                    pass
            parent_message_id = (request.parent_message_id or "").strip() or None

            msg = await agent_message_service.create_message(
                line_id=team_id,
                from_agent_id=from_agent_id,
                to_agent_id=to_agent_id,
                message_type=message_type,
                content=content,
                metadata=metadata,
                parent_message_id=parent_message_id,
                user_id=user_id,
            )
            if metadata.get("trigger_dispatch") and to_agent_id and team_id:
                try:
                    from services import agent_line_service as _als
                    from services.celery_tasks.team_heartbeat_tasks import dispatch_worker_for_message

                    _team = await _als.get_line(team_id, user_id)
                    if not _team or str(_team.get("status") or "").lower() != "active":
                        logger.info("Skipping dispatch_worker_for_message: line not active")
                    else:
                        dispatch_worker_for_message.apply_async(
                            args=[team_id, user_id, to_agent_id, msg.get("id", ""), from_agent_id or ""],
                            countdown=2,
                        )
                except Exception as _dispatch_err:
                    logger.warning("Failed to enqueue dispatch_worker_for_message: %s", _dispatch_err)
            return tool_service_pb2.CreateAgentMessageResponse(
                success=True,
                message_id=msg.get("id", ""),
                message_json=json.dumps(msg),
                error="",
            )
        except Exception as e:
            logger.exception("CreateAgentMessage failed")
            return tool_service_pb2.CreateAgentMessageResponse(
                success=False, message_id="", message_json="", error=str(e)[:500]
            )

    async def AppendLineAgentChatMessage(
        self,
        request: tool_service_pb2.AppendLineAgentChatMessageRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.AppendLineAgentChatMessageResponse:
        """Persist a line sub-agent assistant turn into the user chat conversation."""
        import json

        try:
            from services.conversation_service import ConversationService
            from services.database_manager.database_helpers import fetch_one
            from utils.websocket_manager import get_websocket_manager

            user_id = (request.user_id or "system").strip()
            conversation_id = (request.conversation_id or "").strip()
            content = (request.content or "").strip()
            if not conversation_id or not content:
                return tool_service_pb2.AppendLineAgentChatMessageResponse(
                    success=False, message_id="", error="conversation_id and content are required"
                )

            agent_profile_id = (request.agent_profile_id or "").strip()
            line_id = (request.line_id or "").strip()
            line_role = (request.line_role or "").strip()
            if line_id and agent_profile_id and not line_role:
                try:
                    row = await fetch_one(
                        """
                        SELECT role FROM agent_line_memberships
                        WHERE line_id = $1::uuid AND agent_profile_id = $2::uuid
                        LIMIT 1
                        """,
                        line_id,
                        agent_profile_id,
                    )
                    if row and row.get("role"):
                        line_role = str(row["role"])
                except Exception as role_err:
                    logger.debug("Line role lookup skipped: %s", role_err)

            extra: Dict[str, Any] = {}
            if request.metadata_json:
                try:
                    extra = json.loads(request.metadata_json)
                    if not isinstance(extra, dict):
                        extra = {}
                except (json.JSONDecodeError, TypeError):
                    extra = {}

            msg_meta: Dict[str, Any] = {
                "orchestrator_system": True,
                "line_dispatch_sub_agent": True,
                "delegated_agent": (request.agent_display_name or "line_agent").strip() or "line_agent",
                "agent_profile_id": agent_profile_id or None,
                "agent_display_name": (request.agent_display_name or "").strip() or None,
                "line_id": line_id or None,
                "line_role": line_role or None,
                "line_agent_handle": (request.line_agent_handle or "").strip() or None,
                "delegated_by": (request.delegated_by_agent_id or "").strip() or None,
            }
            msg_meta.update(extra)
            msg_meta = {k: v for k, v in msg_meta.items() if v is not None}

            conversation_service = ConversationService()
            conversation_service.set_current_user(user_id)
            saved = await conversation_service.add_message(
                conversation_id=conversation_id,
                user_id=user_id,
                role="assistant",
                content=content,
                metadata=msg_meta,
            )
            mid = ""
            if saved and isinstance(saved, dict):
                mid = str(saved.get("message_id") or saved.get("id") or "")
            try:
                ws = get_websocket_manager()
                await ws.send_line_agent_chat_update(
                    conversation_id,
                    user_id,
                    {
                        "message_id": mid,
                        "content": content,
                        "role": "assistant",
                        "metadata": msg_meta,
                    },
                )
            except Exception as ws_err:
                logger.warning("AppendLineAgentChatMessage WebSocket push failed: %s", ws_err)

            return tool_service_pb2.AppendLineAgentChatMessageResponse(
                success=True, message_id=mid, error=""
            )
        except Exception as e:
            logger.exception("AppendLineAgentChatMessage failed")
            return tool_service_pb2.AppendLineAgentChatMessageResponse(
                success=False, message_id="", error=str(e)[:500]
            )

    async def ReadTeamTimeline(
        self,
        request: tool_service_pb2.ReadTeamTimelineRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ReadTeamTimelineResponse:
        """Return recent team timeline messages for agent context."""
        import json
        try:
            from services import agent_message_service
            user_id = request.user_id or "system"
            team_id = (request.team_id or "").strip()
            if not team_id:
                return tool_service_pb2.ReadTeamTimelineResponse(
                    success=False, items_json="[]", total=0, error="team_id is required"
                )
            limit = max(1, min(100, request.limit or 20))
            since_hours = request.since_hours or 0
            since = None
            if since_hours > 0:
                from datetime import datetime, timezone, timedelta
                since = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()
            result = await agent_message_service.get_line_timeline(
                line_id=team_id,
                user_id=user_id,
                limit=limit,
                offset=0,
                since=since,
            )
            items = result.get("items") or []
            total = result.get("total") or 0
            return tool_service_pb2.ReadTeamTimelineResponse(
                success=True,
                items_json=json.dumps(items),
                total=total,
                error="",
            )
        except Exception as e:
            logger.exception("ReadTeamTimeline failed")
            return tool_service_pb2.ReadTeamTimelineResponse(
                success=False, items_json="[]", total=0, error=str(e)[:500]
            )

    async def ReadAgentMessages(
        self,
        request: tool_service_pb2.ReadAgentMessagesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ReadAgentMessagesResponse:
        """Return messages to/from a specific agent in a team."""
        import json
        try:
            from services import agent_message_service
            user_id = request.user_id or "system"
            team_id = (request.team_id or "").strip()
            agent_profile_id = (request.agent_profile_id or "").strip()
            if not team_id or not agent_profile_id:
                return tool_service_pb2.ReadAgentMessagesResponse(
                    success=False, items_json="[]", total=0, error="team_id and agent_profile_id are required"
                )
            limit = max(1, min(100, request.limit or 50))
            result = await agent_message_service.get_agent_messages(
                agent_profile_id=agent_profile_id,
                line_id=team_id,
                user_id=user_id,
                limit=limit,
                offset=0,
            )
            items = result.get("items") or []
            total = result.get("total") or 0
            return tool_service_pb2.ReadAgentMessagesResponse(
                success=True,
                items_json=json.dumps(items),
                total=total,
                error="",
            )
        except Exception as e:
            logger.exception("ReadAgentMessages failed")
            return tool_service_pb2.ReadAgentMessagesResponse(
                success=False, items_json="[]", total=0, error=str(e)[:500]
            )

    async def GetTeamStatusBoard(
        self,
        request: tool_service_pb2.GetTeamStatusBoardRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetTeamStatusBoardResponse:
        """Return composed team overview: members with tasks, goals, last activity."""
        import json
        try:
            from datetime import datetime
            from services import agent_line_service, agent_task_service, agent_goal_service
            from services.database_manager.database_helpers import fetch_all
            from utils.grpc_rls import grpc_admin_rls as _grpc_admin_rls

            user_id = request.user_id or "system"
            team_id = (request.team_id or "").strip()
            if not team_id:
                return tool_service_pb2.GetTeamStatusBoardResponse(
                    success=False, board_json="{}", error="team_id is required"
                )
            team = await agent_line_service.get_line(team_id, user_id)
            if not team:
                return tool_service_pb2.GetTeamStatusBoardResponse(
                    success=False, board_json="{}", error="Team not found"
                )
            members = team.get("members") or []
            tasks = await agent_task_service.list_line_tasks(team_id, user_id)
            pending_tasks = [t for t in tasks if t.get("status") not in ("done", "cancelled")]
            goals_tree = await agent_goal_service.get_goal_tree(team_id, user_id)

            def flatten_goals(nodes, out):
                for n in nodes or []:
                    if n.get("status") in ("done", "cancelled"):
                        continue
                    out.append(n)
                    flatten_goals(n.get("children") or [], out)

            goals_flat = []
            flatten_goals(goals_tree, goals_flat)
            tasks_by_agent = {}
            for t in pending_tasks:
                aid = t.get("assigned_agent_id")
                if aid:
                    tasks_by_agent.setdefault(str(aid), []).append(t)
            goals_by_agent = {}
            for g in goals_flat:
                aid = g.get("assigned_agent_id")
                if aid:
                    goals_by_agent.setdefault(str(aid), []).append(g)

            member_ids = [m["agent_profile_id"] for m in members if m.get("agent_profile_id")]
            last_msg_per_agent = {}
            last_exec_per_agent = {}
            if member_ids:
                # agent_messages: line_id = $1, member ids = $2, $3, ...
                msg_placeholders = ",".join([f"${i+2}" for i in range(len(member_ids))])
                rows = await fetch_all(
                    f"""
                    SELECT DISTINCT ON (from_agent_id) from_agent_id, created_at FROM agent_messages
                    WHERE line_id = $1 AND from_agent_id IN ({msg_placeholders})
                    ORDER BY from_agent_id, created_at DESC
                    """,
                    team_id,
                    *member_ids,
                )
                for r in rows:
                    last_msg_per_agent[str(r["from_agent_id"])] = r.get("created_at")
                # agent_execution_log: only member ids, so use $1, $2, ...
                exec_placeholders = ",".join([f"${i+1}" for i in range(len(member_ids))])
                exec_rows = await fetch_all(
                    f"""
                    SELECT DISTINCT ON (agent_profile_id) agent_profile_id, started_at FROM agent_execution_log
                    WHERE agent_profile_id IN ({exec_placeholders})
                    ORDER BY agent_profile_id, started_at DESC
                    """,
                    *member_ids,
                    rls_context=_grpc_admin_rls(),
                )
                for r in exec_rows:
                    last_exec_per_agent[str(r["agent_profile_id"])] = r.get("started_at")

            membership_id_to_agent = {
                str(m["id"]): {
                    "agent_profile_id": str(m["agent_profile_id"]),
                    "agent_name": m.get("agent_name") or m.get("agent_handle") or "Unknown",
                    "agent_handle": m.get("agent_handle") or "",
                }
                for m in members
                if m.get("id") and m.get("agent_profile_id")
            }
            board_members = []
            for m in members:
                aid = m.get("agent_profile_id")
                if not aid:
                    continue
                sid = str(aid)
                mid = str(m.get("id", ""))
                direct_reports = [
                    membership_id_to_agent[str(m2["id"])]
                    for m2 in members
                    if str(m2.get("reports_to") or "") == mid
                ]
                manager = membership_id_to_agent.get(str(m.get("reports_to") or "")) if m.get("reports_to") else None
                peers = [
                    membership_id_to_agent[str(m2["id"])]
                    for m2 in members
                    if str(m2.get("reports_to") or "") == str(m.get("reports_to") or "")
                    and str(m2.get("agent_profile_id")) != sid
                ]
                last_task_at = None
                for t in tasks_by_agent.get(sid, []):
                    u = t.get("updated_at")
                    if u:
                        try:
                            ut = datetime.fromisoformat(str(u).replace("Z", "+00:00"))
                            if last_task_at is None or ut > last_task_at:
                                last_task_at = ut
                        except (ValueError, TypeError):
                            pass
                last_msg_at = last_msg_per_agent.get(sid)
                last_exec_at = last_exec_per_agent.get(sid)
                last_activity = None
                for ts in (last_msg_at, last_task_at, last_exec_at):
                    if ts:
                        try:
                            t = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                            if last_activity is None or t > last_activity:
                                last_activity = t
                        except (ValueError, TypeError):
                            pass
                last_activity_at = None
                if last_activity:
                    last_activity_at = last_activity.isoformat() if hasattr(last_activity, "isoformat") else str(last_activity)
                board_members.append({
                    "agent_profile_id": sid,
                    "agent_name": m.get("agent_name") or m.get("agent_handle") or "Unknown",
                    "agent_handle": m.get("agent_handle") or "",
                    "role": m.get("role") or "worker",
                    "tasks": tasks_by_agent.get(sid, []),
                    "goals": goals_by_agent.get(sid, []),
                    "last_activity_at": last_activity_at,
                    "task_count": len(tasks_by_agent.get(sid, [])),
                    "goal_count": len(goals_by_agent.get(sid, [])),
                    "direct_reports": direct_reports,
                    "reports_to_agent_id": manager["agent_profile_id"] if manager else None,
                    "reports_to_agent_name": manager.get("agent_name") if manager else None,
                    "peers": peers,
                })
            board = {
                "team_name": team.get("name", "Team"),
                "line_id": team_id,
                "members": board_members,
            }
            return tool_service_pb2.GetTeamStatusBoardResponse(
                success=True,
                board_json=json.dumps(board),
                error="",
            )
        except Exception as e:
            logger.exception("GetTeamStatusBoard failed")
            return tool_service_pb2.GetTeamStatusBoardResponse(
                success=False, board_json="{}", error=str(e)[:500]
            )

    async def SetWorkspaceEntry(
        self,
        request: tool_service_pb2.SetWorkspaceEntryRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SetWorkspaceEntryResponse:
        """Upsert a team workspace entry (Blackboard pattern)."""
        try:
            from services import agent_workspace_service
            user_id = request.user_id or "system"
            team_id = (request.team_id or "").strip()
            key = (request.key or "").strip()
            value = request.value or ""
            updated_by_agent_id = (request.updated_by_agent_id or "").strip() or None
            if not team_id or not key:
                return tool_service_pb2.SetWorkspaceEntryResponse(
                    success=False, key=key, error="team_id and key are required"
                )
            result = await agent_workspace_service.set_workspace_entry(
                line_id=team_id,
                key=key,
                value=value,
                user_id=user_id,
                updated_by_agent_id=updated_by_agent_id,
            )
            if not result.get("success"):
                return tool_service_pb2.SetWorkspaceEntryResponse(
                    success=False, key=key, error=(result.get("error") or "Failed")[:500]
                )
            updated_at = result.get("updated_at") or ""
            return tool_service_pb2.SetWorkspaceEntryResponse(
                success=True, key=key, updated_at=updated_at, error=""
            )
        except Exception as e:
            logger.exception("SetWorkspaceEntry failed")
            return tool_service_pb2.SetWorkspaceEntryResponse(
                success=False, key=request.key or "", error=str(e)[:500]
            )

    async def ReadWorkspace(
        self,
        request: tool_service_pb2.ReadWorkspaceRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ReadWorkspaceResponse:
        """Read one workspace entry by key, or list all keys if key is empty."""
        import json
        try:
            from services import agent_workspace_service
            user_id = request.user_id or "system"
            team_id = (request.team_id or "").strip()
            key = (request.key or "").strip()
            if not team_id:
                return tool_service_pb2.ReadWorkspaceResponse(
                    success=False, entries_json="[]", single=False, error="team_id is required"
                )
            if key:
                result = await agent_workspace_service.get_workspace_entry(
                    line_id=team_id, key=key, user_id=user_id
                )
                if not result.get("success"):
                    return tool_service_pb2.ReadWorkspaceResponse(
                        success=False, entries_json="{}", single=True, error=(result.get("error") or "Failed")[:500]
                    )
                return tool_service_pb2.ReadWorkspaceResponse(
                    success=True,
                    entries_json=json.dumps(result),
                    single=True,
                    error="",
                )
            result = await agent_workspace_service.list_workspace(line_id=team_id, user_id=user_id)
            if not result.get("success"):
                return tool_service_pb2.ReadWorkspaceResponse(
                    success=False, entries_json="[]", single=False, error=(result.get("error") or "Failed")[:500]
                )
            return tool_service_pb2.ReadWorkspaceResponse(
                success=True,
                entries_json=json.dumps(result.get("entries") or []),
                single=False,
                error="",
            )
        except Exception as e:
            logger.exception("ReadWorkspace failed")
            return tool_service_pb2.ReadWorkspaceResponse(
                success=False, entries_json="[]", single=False, error=str(e)[:500]
            )

    async def GetGoalAncestry(
        self,
        request: tool_service_pb2.GetGoalAncestryRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetGoalAncestryResponse:
        """Get goal ancestry from leaf to root for context injection."""
        try:
            from services import agent_goal_service
            user_id = request.user_id or "system"
            goal_id = (request.goal_id or "").strip()
            if not goal_id:
                return tool_service_pb2.GetGoalAncestryResponse(success=False, goals_json="[]", error="goal_id required")
            ancestry = await agent_goal_service.get_goal_ancestry(goal_id, user_id)
            return tool_service_pb2.GetGoalAncestryResponse(
                success=True,
                goals_json=json.dumps(ancestry),
                error="",
            )
        except Exception as e:
            logger.exception("GetGoalAncestry failed")
            return tool_service_pb2.GetGoalAncestryResponse(success=False, goals_json="[]", error=str(e)[:500])

    async def GetTeamGoalsTree(
        self,
        request: tool_service_pb2.GetTeamGoalsTreeRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetTeamGoalsTreeResponse:
        """Get full goal tree for a team."""
        try:
            from services import agent_goal_service
            user_id = request.user_id or "system"
            team_id = (request.team_id or "").strip()
            if not team_id:
                return tool_service_pb2.GetTeamGoalsTreeResponse(success=False, tree_json="[]", error="team_id required")
            tree = await agent_goal_service.get_goal_tree(team_id, user_id)
            return tool_service_pb2.GetTeamGoalsTreeResponse(
                success=True,
                tree_json=json.dumps(tree),
                error="",
            )
        except Exception as e:
            logger.exception("GetTeamGoalsTree failed")
            return tool_service_pb2.GetTeamGoalsTreeResponse(success=False, tree_json="[]", error=str(e)[:500])

    async def GetGoalsForAgent(
        self,
        request: tool_service_pb2.GetGoalsForAgentRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetGoalsForAgentResponse:
        """Return goals assigned to an agent in a team."""
        try:
            from services import agent_goal_service
            import json
            user_id = request.user_id or "system"
            team_id = (request.team_id or "").strip()
            agent_profile_id = (request.agent_profile_id or "").strip()
            if not team_id or not agent_profile_id:
                return tool_service_pb2.GetGoalsForAgentResponse(success=False, goals_json="[]", error="team_id and agent_profile_id required")
            goals = await agent_goal_service.get_goals_for_agent(agent_profile_id, team_id, user_id)
            return tool_service_pb2.GetGoalsForAgentResponse(success=True, goals_json=json.dumps(goals), error="")
        except Exception as e:
            logger.exception("GetGoalsForAgent failed")
            return tool_service_pb2.GetGoalsForAgentResponse(success=False, goals_json="[]", error=str(e)[:500])

    async def UpdateGoalProgress(
        self,
        request: tool_service_pb2.UpdateGoalProgressRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateGoalProgressResponse:
        """Update goal progress percentage."""
        try:
            from services import agent_goal_service
            user_id = request.user_id or "system"
            goal_id = (request.goal_id or "").strip()
            progress_pct = max(0, min(100, request.progress_pct))
            await agent_goal_service.update_progress(goal_id, user_id, progress_pct)
            return tool_service_pb2.UpdateGoalProgressResponse(success=True, error="")
        except Exception as e:
            logger.exception("UpdateGoalProgress failed")
            return tool_service_pb2.UpdateGoalProgressResponse(success=False, error=str(e)[:500])

    async def CreateAgentTask(
        self,
        request: tool_service_pb2.CreateAgentTaskRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateAgentTaskResponse:
        """Create a team task."""
        try:
            from services import agent_task_service
            import json
            user_id = request.user_id or "system"
            line_id = (request.team_id or "").strip()
            task = await agent_task_service.create_task(
                line_id=line_id,
                user_id=user_id,
                title=(request.title or "").strip() or "Untitled",
                description=(request.description or "").strip() or None,
                assigned_agent_id=(request.assigned_agent_id or "").strip() or None,
                goal_id=(request.goal_id or "").strip() or None,
                priority=request.priority or 0,
                created_by_agent_id=(request.created_by_agent_id or "").strip() or None,
                due_date=(request.due_date or "").strip() or None,
            )
            tid = task.get("id", "")
            logger.info(
                "CreateAgentTask: user=%s line=%s task=%s assigned=%s goal=%s title=%s",
                user_id,
                line_id,
                tid,
                (request.assigned_agent_id or "").strip() or None,
                (request.goal_id or "").strip() or None,
                ((request.title or "").strip() or "Untitled")[:120],
            )
            return tool_service_pb2.CreateAgentTaskResponse(
                success=True, task_id=tid, task_json=json.dumps(task), error=""
            )
        except Exception as e:
            logger.exception("CreateAgentTask failed")
            return tool_service_pb2.CreateAgentTaskResponse(success=False, task_id="", task_json="", error=str(e)[:500])

    async def GetAgentWorkQueue(
        self,
        request: tool_service_pb2.GetAgentWorkQueueRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetAgentWorkQueueResponse:
        """Get tasks assigned to an agent in a team."""
        try:
            from services import agent_task_service
            import json
            user_id = request.user_id or "system"
            team_id = (request.team_id or "").strip()
            agent_profile_id = (request.agent_profile_id or "").strip()
            if not team_id or not agent_profile_id:
                return tool_service_pb2.GetAgentWorkQueueResponse(success=False, tasks_json="[]", error="team_id and agent_profile_id required")
            tasks = await agent_task_service.get_agent_work_queue(agent_profile_id, team_id, user_id)
            return tool_service_pb2.GetAgentWorkQueueResponse(success=True, tasks_json=json.dumps(tasks), error="")
        except Exception as e:
            logger.exception("GetAgentWorkQueue failed")
            return tool_service_pb2.GetAgentWorkQueueResponse(success=False, tasks_json="[]", error=str(e)[:500])

    async def UpdateTaskStatus(
        self,
        request: tool_service_pb2.UpdateTaskStatusRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateTaskStatusResponse:
        """Transition task to a new status."""
        try:
            from services import agent_task_service
            import json
            user_id = request.user_id or "system"
            task_id = (request.task_id or "").strip()
            new_status = (request.new_status or "").strip()
            if not task_id or not new_status:
                return tool_service_pb2.UpdateTaskStatusResponse(success=False, task_json="", error="task_id and new_status required")
            task = await agent_task_service.transition_task(task_id, user_id, new_status)
            return tool_service_pb2.UpdateTaskStatusResponse(success=True, task_json=json.dumps(task), error="")
        except ValueError as e:
            err = str(e)[:500]
            logger.warning("UpdateTaskStatus rejected: %s", err)
            return tool_service_pb2.UpdateTaskStatusResponse(success=False, task_json="", error=err)
        except Exception as e:
            logger.exception("UpdateTaskStatus failed")
            return tool_service_pb2.UpdateTaskStatusResponse(success=False, task_json="", error=str(e)[:500])

    async def AssignTaskToAgent(
        self,
        request: tool_service_pb2.AssignTaskToAgentRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.AssignTaskToAgentResponse:
        """Assign a task to an agent."""
        try:
            from services import agent_task_service
            import json
            user_id = request.user_id or "system"
            task_id = (request.task_id or "").strip()
            agent_profile_id = (request.agent_profile_id or "").strip()
            if not task_id or not agent_profile_id:
                return tool_service_pb2.AssignTaskToAgentResponse(success=False, task_json="", error="task_id and agent_profile_id required")
            task = await agent_task_service.assign_task(task_id, agent_profile_id, user_id)
            return tool_service_pb2.AssignTaskToAgentResponse(success=True, task_json=json.dumps(task), error="")
        except Exception as e:
            logger.exception("AssignTaskToAgent failed")
            return tool_service_pb2.AssignTaskToAgentResponse(success=False, task_json="", error=str(e)[:500])

    async def GetUserNotificationPreferences(
        self,
        request: tool_service_pb2.GetUserNotificationPreferencesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetUserNotificationPreferencesResponse:
        """Get user notification preferences from users.preferences JSONB."""
        try:
            from services.database_manager.database_helpers import fetch_one

            user_id = request.user_id or "system"
            row = await fetch_one(
                "SELECT preferences FROM users WHERE user_id = $1",
                user_id,
            )
            prefs = {}
            if row:
                all_prefs = row.get("preferences") or {}
                if isinstance(all_prefs, str):
                    all_prefs = json.loads(all_prefs)
                prefs = all_prefs.get("notification_preferences", {})

            return tool_service_pb2.GetUserNotificationPreferencesResponse(
                success=True,
                preferences_json=json.dumps(prefs),
                error="",
            )
        except Exception as e:
            logger.error("GetUserNotificationPreferences failed: %s", e)
            return tool_service_pb2.GetUserNotificationPreferencesResponse(
                success=False, preferences_json="{}", error=str(e)
            )

    async def GetMyProfile(
        self,
        request: tool_service_pb2.GetMyProfileRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetMyProfileResponse:
        """Get the current user's profile (email, display_name, username) and key settings from users + user_settings."""
        try:
            import json as _json
            from services.database_manager.database_helpers import fetch_one, fetch_all

            user_id = request.user_id or "system"
            email = ""
            display_name = ""
            username = ""
            preferred_name = ""
            timezone_val = ""
            zip_code = ""
            ai_context = ""

            row = await fetch_one(
                "SELECT email, display_name, username, preferences FROM users WHERE user_id = $1",
                user_id,
            )
            if row:
                email = row.get("email") or ""
                display_name = row.get("display_name") or ""
                username = row.get("username") or ""
                prefs = row.get("preferences") or {}
                if isinstance(prefs, str):
                    try:
                        prefs = _json.loads(prefs)
                    except Exception:
                        prefs = {}
                if isinstance(prefs, dict):
                    timezone_val = prefs.get("timezone") or ""
                    zip_code = prefs.get("zip_code") or ""

            settings_rows = await fetch_all(
                "SELECT key, value FROM user_settings WHERE user_id = $1 AND key IN ('user_preferred_name', 'user_ai_context')",
                user_id,
            )
            for s in settings_rows or []:
                k, v = s.get("key"), (s.get("value") or "")
                if k == "user_preferred_name":
                    preferred_name = v
                elif k == "user_ai_context":
                    ai_context = v

            return tool_service_pb2.GetMyProfileResponse(
                email=email,
                display_name=display_name,
                username=username,
                preferred_name=preferred_name,
                timezone=timezone_val,
                zip_code=zip_code,
                ai_context=ai_context,
                success=True,
                error="",
            )
        except Exception as e:
            logger.error("GetMyProfile failed: %s", e)
            return tool_service_pb2.GetMyProfileResponse(
                email="",
                display_name="",
                username="",
                preferred_name="",
                timezone="",
                zip_code="",
                ai_context="",
                success=False,
                error=str(e),
            )

    async def UpsertUserFact(
        self,
        request: tool_service_pb2.UpsertUserFactRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpsertUserFactResponse:
        """Insert or update a single user fact."""
        try:
            from services.settings_service import settings_service

            user_id = request.user_id or "system"
            fact_key = request.fact_key or ""
            value = request.value or ""
            category = request.category or "general"
            if not fact_key.strip():
                return tool_service_pb2.UpsertUserFactResponse(success=False, error="fact_key is required")
            write_enabled = await settings_service.get_facts_write_enabled(user_id)
            if not write_enabled:
                return tool_service_pb2.UpsertUserFactResponse(success=False, error="Fact writing disabled by user")
            source = getattr(request, "source", None) or "user_manual"
            result = await settings_service.upsert_user_fact(
                user_id, fact_key.strip(), value, category, source=source
            )
            if result.get("success"):
                return tool_service_pb2.UpsertUserFactResponse(success=True, error="")
            if result.get("status") == "pending_review":
                msg = (
                    "Fact '%s' is currently set to '%s' by the user. "
                    "Your proposed update has been queued for user review."
                ) % (result.get("fact_key", fact_key), result.get("current_value", ""))
                return tool_service_pb2.UpsertUserFactResponse(success=False, error=msg)
            return tool_service_pb2.UpsertUserFactResponse(
                success=False, error=result.get("error", "Upsert failed")
            )
        except Exception as e:
            logger.error("UpsertUserFact failed: %s", e)
            return tool_service_pb2.UpsertUserFactResponse(success=False, error=str(e))

    async def GetUserFacts(
        self,
        request: tool_service_pb2.GetUserFactsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetUserFactsResponse:
        """Return facts for a user (optionally query-filtered with theme-first adaptive retrieval)."""
        try:
            from services.settings_service import settings_service

            user_id = request.user_id or "system"
            facts = await settings_service.get_user_facts(user_id)
            query = (request.query or "").strip()
            use_full_fact_list = bool(request.use_full_fact_list)
            if query and not use_full_fact_list:
                try:
                    from services.embedding_service_wrapper import get_embedding_service
                    from services.fact_theme_service import (
                        load_themes_for_user,
                        select_facts_for_query,
                    )

                    emb_svc = await get_embedding_service()
                    vecs = await emb_svc.generate_embeddings([query])
                    qvec = vecs[0] if vecs else None
                    if qvec:
                        themes = await load_themes_for_user(user_id)
                        facts = select_facts_for_query(
                            facts, themes, qvec, use_themed_memory=True
                        )
                except Exception as filt_e:
                    logger.warning("GetUserFacts themed filter failed, using full list: %s", filt_e)
            slim = []
            for f in facts:
                d = {k: v for k, v in dict(f).items() if k != "embedding"}
                slim.append(d)
            facts_json = json.dumps(slim, default=json_default)
            return tool_service_pb2.GetUserFactsResponse(success=True, facts_json=facts_json, error="")
        except Exception as e:
            logger.error("GetUserFacts failed: %s", e)
            return tool_service_pb2.GetUserFactsResponse(
                success=False, facts_json="[]", error=str(e)
            )

    async def ReadScratchpad(
        self,
        request: tool_service_pb2.ReadScratchpadRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ReadScratchpadResponse:
        """Return the user's scratch pad pads from user_settings. pad_index=-1 returns all four pads."""
        try:
            import json as _json
            from models.home_dashboard_models import (
                SCRATCH_PAD_COUNT,
                SCRATCH_PAD_SETTING_KEY,
                default_scratchpad_data,
            )
            from services.user_settings_kv_service import get_user_setting

            user_id = request.user_id or "system"
            # proto3 default for int32 is 0; we use -1 as sentinel for "all pads"
            pad_index = request.pad_index  # caller passes -1 for all, 0-3 for specific

            raw = await get_user_setting(user_id, SCRATCH_PAD_SETTING_KEY)
            if raw and str(raw).strip():
                try:
                    data = _json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    data = default_scratchpad_data().model_dump(mode="json")
            else:
                data = default_scratchpad_data().model_dump(mode="json")

            pads_data = data.get("pads", [])
            active_index = int(data.get("active_index", 0))

            def _make_pad_msg(i, p):
                return tool_service_pb2.ScratchpadPad(
                    index=i,
                    label=str(p.get("label", f"Pad {i + 1}")),
                    body=str(p.get("body", "")),
                )

            if pad_index == -1:
                pads = [_make_pad_msg(i, p) for i, p in enumerate(pads_data)]
            elif 0 <= pad_index < SCRATCH_PAD_COUNT:
                p = pads_data[pad_index] if pad_index < len(pads_data) else {}
                pads = [_make_pad_msg(pad_index, p)]
            else:
                return tool_service_pb2.ReadScratchpadResponse(
                    success=False,
                    error=f"pad_index must be -1 (all) or 0-{SCRATCH_PAD_COUNT - 1}",
                )

            return tool_service_pb2.ReadScratchpadResponse(
                success=True,
                error="",
                pads=pads,
                active_index=active_index,
            )
        except Exception as e:
            logger.error("ReadScratchpad failed: %s", e)
            return tool_service_pb2.ReadScratchpadResponse(success=False, error=str(e))

    async def WriteScratchpadPad(
        self,
        request: tool_service_pb2.WriteScratchpadPadRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.WriteScratchpadPadResponse:
        """Overwrite the body (and optionally label) of a single scratch pad for the user."""
        try:
            import json as _json
            from models.home_dashboard_models import (
                SCRATCH_PAD_COUNT,
                MAX_SCRATCH_PAD_BODY_CHARS,
                MAX_SCRATCH_PAD_LABEL_LEN,
                SCRATCH_PAD_SETTING_KEY,
                default_scratchpad_data,
            )
            from services.user_settings_kv_service import get_user_setting, set_user_setting

            user_id = request.user_id or "system"
            pad_index = request.pad_index
            body = request.body or ""
            label = request.label or ""

            if not (0 <= pad_index < SCRATCH_PAD_COUNT):
                return tool_service_pb2.WriteScratchpadPadResponse(
                    success=False,
                    error=f"pad_index must be 0-{SCRATCH_PAD_COUNT - 1}",
                )
            if len(body) > MAX_SCRATCH_PAD_BODY_CHARS:
                return tool_service_pb2.WriteScratchpadPadResponse(
                    success=False,
                    error=f"body exceeds maximum length of {MAX_SCRATCH_PAD_BODY_CHARS} characters",
                )
            if label and len(label) > MAX_SCRATCH_PAD_LABEL_LEN:
                return tool_service_pb2.WriteScratchpadPadResponse(
                    success=False,
                    error=f"label exceeds maximum length of {MAX_SCRATCH_PAD_LABEL_LEN} characters",
                )

            raw = await get_user_setting(user_id, SCRATCH_PAD_SETTING_KEY)
            if raw and str(raw).strip():
                try:
                    data = _json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    data = default_scratchpad_data().model_dump(mode="json")
            else:
                data = default_scratchpad_data().model_dump(mode="json")

            pads = data.get("pads", [])
            while len(pads) < SCRATCH_PAD_COUNT:
                i = len(pads)
                pads.append({"label": f"Pad {i + 1}", "body": ""})

            pads[pad_index]["body"] = body
            if label:
                pads[pad_index]["label"] = label

            data["pads"] = pads
            ok = await set_user_setting(
                user_id,
                SCRATCH_PAD_SETTING_KEY,
                _json.dumps(data, ensure_ascii=False),
                "json",
            )
            if not ok:
                return tool_service_pb2.WriteScratchpadPadResponse(
                    success=False, error="Failed to persist scratch pad"
                )
            return tool_service_pb2.WriteScratchpadPadResponse(success=True, error="")
        except Exception as e:
            logger.error("WriteScratchpadPad failed: %s", e)
            return tool_service_pb2.WriteScratchpadPadResponse(success=False, error=str(e))

    async def InvokeDeviceTool(
        self,
        request: tool_service_pb2.InvokeDeviceToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.InvokeDeviceToolResponse:
        """Invoke a tool on a connected local proxy device via backend (device WebSockets live there)."""
        try:
            import os

            user_id = request.user_id or "system"
            device_id = request.device_id or ""
            tool = request.tool or ""
            args_json = request.args_json or "{}"
            timeout_seconds = request.timeout_seconds or 30
            logger.info(
                "InvokeDeviceTool: user_id=%s, device_id=%s, tool=%s",
                user_id,
                device_id or "(any)",
                tool,
            )
            try:
                args = json.loads(args_json)
            except json.JSONDecodeError:
                args = {}

            backend_url = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
            internal_key = os.getenv("INTERNAL_SERVICE_KEY", "")
            if not internal_key:
                logger.warning("InvokeDeviceTool: INTERNAL_SERVICE_KEY not set; backend may reject request")

            payload = {
                "user_id": user_id,
                "tool": tool,
                "args": args,
                "timeout_seconds": timeout_seconds,
            }
            if device_id:
                payload["device_id"] = device_id

            url = f"{backend_url}/api/internal/invoke-device-tool"
            headers = {"Content-Type": "application/json"}
            if internal_key:
                headers["X-Internal-Service-Key"] = internal_key

            http_timeout = max(35.0, float(timeout_seconds) + 5.0)

            import httpx
            async with httpx.AsyncClient(timeout=http_timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                err = resp.text or f"HTTP {resp.status_code}"
                logger.error("InvokeDeviceTool backend call failed: %s", err)
                return tool_service_pb2.InvokeDeviceToolResponse(
                    success=False,
                    result_json="{}",
                    error=err[:500],
                    formatted=err[:500],
                )
            data = resp.json()
            return tool_service_pb2.InvokeDeviceToolResponse(
                success=data.get("success", False),
                result_json=data.get("result_json", "{}"),
                error=data.get("error", ""),
                formatted=data.get("formatted", ""),
            )
        except Exception as e:
            logger.error("InvokeDeviceTool failed: %s", e)
            return tool_service_pb2.InvokeDeviceToolResponse(
                success=False,
                result_json="{}",
                error=str(e),
                formatted=str(e),
            )

    async def SetDeviceWorkspace(
        self,
        request: tool_service_pb2.SetDeviceWorkspaceRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SetDeviceWorkspaceResponse:
        """Set the active workspace root on a connected local proxy device."""
        try:
            import os

            user_id = request.user_id or "system"
            device_id = request.device_id or ""
            workspace_root = request.workspace_root or ""
            timeout_seconds = request.timeout_seconds or 30

            if not workspace_root:
                return tool_service_pb2.SetDeviceWorkspaceResponse(
                    success=False,
                    result_json="{}",
                    error="workspace_root is required",
                    formatted="workspace_root is required",
                )

            backend_url = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
            internal_key = os.getenv("INTERNAL_SERVICE_KEY", "")
            if not internal_key:
                logger.warning("SetDeviceWorkspace: INTERNAL_SERVICE_KEY not set; backend may reject request")

            payload = {
                "user_id": user_id,
                "workspace_root": workspace_root,
                "timeout_seconds": timeout_seconds,
            }
            if device_id:
                payload["device_id"] = device_id

            url = f"{backend_url}/api/internal/set-device-workspace"
            headers = {"Content-Type": "application/json"}
            if internal_key:
                headers["X-Internal-Service-Key"] = internal_key

            http_timeout = max(35.0, float(timeout_seconds) + 5.0)
            import httpx
            async with httpx.AsyncClient(timeout=http_timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                err = resp.text or f"HTTP {resp.status_code}"
                return tool_service_pb2.SetDeviceWorkspaceResponse(
                    success=False,
                    result_json="{}",
                    error=err[:500],
                    formatted=err[:500],
                )
            data = resp.json()
            return tool_service_pb2.SetDeviceWorkspaceResponse(
                success=data.get("success", False),
                result_json=data.get("result_json", "{}"),
                error=data.get("error", ""),
                formatted=data.get("formatted", ""),
            )
        except Exception as e:
            logger.error("SetDeviceWorkspace failed: %s", e)
            return tool_service_pb2.SetDeviceWorkspaceResponse(
                success=False,
                result_json="{}",
                error=str(e),
                formatted=str(e),
            )

    async def GetDeviceCapabilities(
        self,
        request: tool_service_pb2.GetDeviceCapabilitiesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetDeviceCapabilitiesResponse:
        """Return union of capabilities from all connected devices for the user."""
        try:
            import os

            user_id = request.user_id or "system"
            backend_url = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
            internal_key = os.getenv("INTERNAL_SERVICE_KEY", "")
            headers = {}
            if internal_key:
                headers["X-Internal-Service-Key"] = internal_key

            url = f"{backend_url}/api/internal/device-list"
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params={"user_id": user_id}, headers=headers)
            if resp.status_code != 200:
                logger.warning("GetDeviceCapabilities backend call failed: %s", resp.text)
                return tool_service_pb2.GetDeviceCapabilitiesResponse(
                    capabilities=[],
                    has_device=False,
                )
            data = resp.json()
            devices = data.get("devices") or []
            all_caps = set()
            for dev in devices:
                for cap in dev.get("capabilities") or []:
                    all_caps.add(cap)
            return tool_service_pb2.GetDeviceCapabilitiesResponse(
                capabilities=sorted(all_caps),
                has_device=len(devices) > 0,
            )
        except Exception as e:
            logger.error("GetDeviceCapabilities failed: %s", e)
            return tool_service_pb2.GetDeviceCapabilitiesResponse(
                capabilities=[],
                has_device=False,
            )

    # ===== Profile Guarantee =====

    async def EnsureDefaultProfile(
        self,
        request: tool_service_pb2.EnsureDefaultProfileRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.EnsureDefaultProfileResponse:
        """Return the user's active builtin profile id, creating one if none exists."""
        try:
            from services.database_manager.database_helpers import fetch_one
            from services.agent_factory_service import seed_default_agent_profiles
            from utils.grpc_rls import grpc_user_rls as _grpc_rls

            user_id = request.user_id or ""
            if not user_id:
                return tool_service_pb2.EnsureDefaultProfileResponse(
                    profile_id="", was_created=False
                )
            ctx = _grpc_rls(user_id)

            row = await fetch_one(
                "SELECT id FROM agent_profiles WHERE user_id = $1 AND is_builtin = true AND is_active = true LIMIT 1",
                user_id,
                rls_context=ctx,
            )
            if row and row.get("id"):
                return tool_service_pb2.EnsureDefaultProfileResponse(
                    profile_id=str(row["id"]), was_created=False
                )

            await seed_default_agent_profiles(user_id=user_id)

            row = await fetch_one(
                "SELECT id FROM agent_profiles WHERE user_id = $1 AND is_builtin = true AND is_active = true LIMIT 1",
                user_id,
                rls_context=ctx,
            )
            if row and row.get("id"):
                logger.info("EnsureDefaultProfile created builtin profile for user %s", user_id)
                return tool_service_pb2.EnsureDefaultProfileResponse(
                    profile_id=str(row["id"]), was_created=True
                )

            return tool_service_pb2.EnsureDefaultProfileResponse(
                profile_id="", was_created=False
            )
        except Exception as e:
            logger.error("EnsureDefaultProfile failed: %s", e)
            return tool_service_pb2.EnsureDefaultProfileResponse(
                profile_id="", was_created=False
            )
