"""
Unified Dispatcher - Single entry point for route-based dispatch.

All requests flow through this: discover route -> CustomAgentRunner -> yield ChatChunk stream.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

from protos import orchestrator_pb2

from orchestrator.routes import get_route_registry, load_all_routes

logger = logging.getLogger(__name__)

_routes_loaded = False


def _persona_ai_name_metadata(result: Dict[str, Any], request_metadata: Dict[str, Any]) -> Dict[str, str]:
    """Chat UI label: prefer resolved profile persona, else request-level persona."""
    name = (result.get("persona_ai_name") or "").strip()
    if not name:
        p = request_metadata.get("persona") or {}
        if isinstance(p, dict):
            name = (p.get("ai_name") or "").strip()
    if name:
        return {"persona_ai_name": name}
    return {}


def _ensure_routes_loaded() -> None:
    global _routes_loaded
    if not _routes_loaded:
        load_all_routes()
        _routes_loaded = True


class UnifiedDispatcher:
    """
    Dispatches requests to CustomAgentRunner for the resolved route (default or custom profile).
    Yields ChatChunk stream (status, content, complete, diagram, chart, editor_operation as needed).
    """

    async def dispatch_custom_agent(
        self,
        agent_profile_id: str,
        query: str,
        metadata: Dict[str, Any],
        messages: List[Any],
        cancellation_token: Optional[Any] = None,
    ) -> AsyncIterator[orchestrator_pb2.ChatChunk]:
        """
        Run a custom Agent Factory profile via CustomAgentRunner: load profile and playbook,
        execute pipeline, route outputs, stream results.
        """
        user_id = metadata.get("user_id", "system")
        agent_name = "custom_agent"

        agent_display_name = None
        try:
            from orchestrator.backend_tool_client import get_backend_tool_client
            client = await get_backend_tool_client()
            profile = await client.get_agent_profile(user_id, agent_profile_id)
            if profile and profile.get("name"):
                agent_display_name = profile["name"]
        except Exception as e:
            logger.warning("Failed to fetch profile name for display: %s", e)

        status_metadata = {}
        if agent_display_name:
            status_metadata["agent_display_name"] = agent_display_name
        yield orchestrator_pb2.ChatChunk(
            type="status",
            message="Loading custom agent profile...",
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            metadata=status_metadata if status_metadata else None,
        )

        runner_metadata = {**metadata, "agent_profile_id": agent_profile_id}
        from orchestrator.agents.custom_agent_runner import CustomAgentRunner
        runner = CustomAgentRunner()

        try:
            result = await runner.process(
                query=query,
                metadata=runner_metadata,
                messages=messages,
                cancellation_token=cancellation_token,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Custom agent failed: %s", e)
            yield orchestrator_pb2.ChatChunk(
                type="content",
                message=f"Custom agent error: {str(e)}",
                timestamp=datetime.now().isoformat(),
                agent_name=agent_name,
            )
            yield orchestrator_pb2.ChatChunk(
                type="complete",
                message="Complete",
                timestamp=datetime.now().isoformat(),
                agent_name="system",
            )
            return

        profile_name_from_result = result.get("agent_profile_name")
        if profile_name_from_result:
            agent_display_name = profile_name_from_result

        content = result.get("formatted") or result.get("response", "")
        pending = result.get("pending_approval")
        pending_auth = result.get("pending_auth")
        if pending or pending_auth:
            payload = {
                "step_name": (pending or pending_auth or {}).get("step_name", ""),
                "prompt": (pending or pending_auth or {}).get("prompt", "Approve to continue?"),
                "on_reject": (pending or {}).get("on_reject", "stop"),
            }
            if pending_auth:
                payload["pending_auth"] = pending_auth
            content = content or f"Approval required at step '{payload['step_name']}'."
            yield orchestrator_pb2.ChatChunk(
                type="permission_request",
                message=json.dumps(payload),
                timestamp=datetime.now().isoformat(),
                agent_name=agent_name,
            )

        content_metadata = {**_persona_ai_name_metadata(result, metadata)}
        if agent_display_name:
            content_metadata["agent_display_name"] = agent_display_name
        yield orchestrator_pb2.ChatChunk(
            type="content",
            message=content,
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            metadata=content_metadata if content_metadata else None,
        )

        complete_metadata = {
            "agent_profile_id": agent_profile_id,
            **_persona_ai_name_metadata(result, metadata),
        }
        if agent_display_name:
            complete_metadata["agent_display_name"] = agent_display_name
        task_status = result.get("task_status")
        if task_status:
            complete_metadata["task_status"] = task_status
        if result.get("approval_queue_id"):
            complete_metadata["approval_queue_id"] = result["approval_queue_id"]
        images = result.get("images") or result.get("structured_images")
        if images:
            complete_metadata["images"] = json.dumps(images)
            logger.info("Custom agent dispatch: including %d image(s) in complete metadata", len(images))
        artifact = result.get("artifact")
        if artifact and isinstance(artifact, dict):
            complete_metadata["artifact"] = json.dumps(artifact)
        artifacts = result.get("artifacts")
        if artifacts and isinstance(artifacts, list):
            complete_metadata["artifacts"] = json.dumps(artifacts)
        tools_cats = result.get("tools_used_categories") or []
        if tools_cats:
            complete_metadata["tools_used_categories"] = json.dumps(tools_cats)
        tool_call_summary = result.get("tool_call_summary", "")
        if tool_call_summary:
            complete_metadata["tool_call_summary"] = tool_call_summary
        acquired_log = result.get("acquired_tool_log")
        if acquired_log:
            complete_metadata["acquired_tool_log"] = json.dumps(acquired_log, default=str)
        skill_exec_events = result.get("skill_execution_events")
        if skill_exec_events:
            complete_metadata["skill_execution_events"] = json.dumps(skill_exec_events, default=str)
        yield orchestrator_pb2.ChatChunk(
            type="complete",
            message="Complete",
            timestamp=datetime.now().isoformat(),
            agent_name="system",
            metadata=complete_metadata,
        )

    async def dispatch_line(
        self,
        query: str,
        metadata: Dict[str, Any],
        messages: List[Any],
        cancellation_token: Optional[Any] = None,
    ) -> AsyncIterator[orchestrator_pb2.ChatChunk]:
        """Run line CEO with chat briefing (metadata must include ceo_profile_id, line_id)."""
        ceo_id = str(metadata.get("ceo_profile_id") or "").strip()
        user_id = metadata.get("user_id", "system")
        agent_name = "line_dispatch"

        agent_display_name = None
        try:
            from orchestrator.backend_tool_client import get_backend_tool_client

            client = await get_backend_tool_client()
            profile = await client.get_agent_profile(user_id, ceo_id)
            if profile and profile.get("name"):
                agent_display_name = profile["name"]
        except Exception as e:
            logger.warning("Line dispatch: failed to fetch CEO profile name: %s", e)

        status_metadata = {}
        if agent_display_name:
            status_metadata["agent_display_name"] = agent_display_name
        yield orchestrator_pb2.ChatChunk(
            type="status",
            message="Running agent line (CEO)...",
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            metadata=status_metadata if status_metadata else None,
        )

        from orchestrator.engines.line_dispatch_engine import LineDispatchEngine

        engine = LineDispatchEngine()
        try:
            result = await engine.process(
                query=query,
                metadata=metadata,
                messages=messages,
                cancellation_token=cancellation_token,
            )
        except Exception as e:
            logger.exception("Line dispatch engine failed: %s", e)
            yield orchestrator_pb2.ChatChunk(
                type="content",
                message=f"Line dispatch error: {str(e)}",
                timestamp=datetime.now().isoformat(),
                agent_name=agent_name,
            )
            yield orchestrator_pb2.ChatChunk(
                type="complete",
                message="Complete",
                timestamp=datetime.now().isoformat(),
                agent_name="system",
            )
            return

        profile_name_from_result = result.get("agent_profile_name")
        if profile_name_from_result:
            agent_display_name = profile_name_from_result

        content = result.get("formatted") or result.get("response", "")
        pending = result.get("pending_approval")
        pending_auth = result.get("pending_auth")
        if pending or pending_auth:
            payload = {
                "step_name": (pending or pending_auth or {}).get("step_name", ""),
                "prompt": (pending or pending_auth or {}).get("prompt", "Approve to continue?"),
                "on_reject": (pending or {}).get("on_reject", "stop"),
            }
            if pending_auth:
                payload["pending_auth"] = pending_auth
            content = content or f"Approval required at step '{payload['step_name']}'."
            yield orchestrator_pb2.ChatChunk(
                type="permission_request",
                message=json.dumps(payload),
                timestamp=datetime.now().isoformat(),
                agent_name=agent_name,
            )

        content_metadata = {**_persona_ai_name_metadata(result, metadata)}
        if agent_display_name:
            content_metadata["agent_display_name"] = agent_display_name
        yield orchestrator_pb2.ChatChunk(
            type="content",
            message=content,
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            metadata=content_metadata if content_metadata else None,
        )

        complete_metadata = {"agent_profile_id": ceo_id, **_persona_ai_name_metadata(result, metadata)}
        line_id = metadata.get("line_id") or metadata.get("team_id")
        if line_id:
            complete_metadata["line_id"] = str(line_id)
        if agent_display_name:
            complete_metadata["agent_display_name"] = agent_display_name
        task_status = result.get("task_status")
        if task_status:
            complete_metadata["task_status"] = task_status
        if result.get("approval_queue_id"):
            complete_metadata["approval_queue_id"] = result["approval_queue_id"]
        images = result.get("images") or result.get("structured_images")
        if images:
            complete_metadata["images"] = json.dumps(images)
        artifact = result.get("artifact")
        if artifact and isinstance(artifact, dict):
            complete_metadata["artifact"] = json.dumps(artifact)
        artifacts = result.get("artifacts")
        if artifacts and isinstance(artifacts, list):
            complete_metadata["artifacts"] = json.dumps(artifacts)
        tools_cats = result.get("tools_used_categories") or []
        if tools_cats:
            complete_metadata["tools_used_categories"] = json.dumps(tools_cats)
        tool_call_summary = result.get("tool_call_summary", "")
        if tool_call_summary:
            complete_metadata["tool_call_summary"] = tool_call_summary
        complete_metadata["line_dispatch_session"] = "true"
        yield orchestrator_pb2.ChatChunk(
            type="complete",
            message="Complete",
            timestamp=datetime.now().isoformat(),
            agent_name="system",
            metadata=complete_metadata,
        )

    async def dispatch(
        self,
        skill_name: str,
        query: str,
        metadata: Dict[str, Any],
        messages: List[Any],
        cancellation_token: Optional[Any] = None,
    ) -> AsyncIterator[orchestrator_pb2.ChatChunk]:
        """
        Run the route via its engine and yield ChatChunk stream.
        """
        _ensure_routes_loaded()
        registry = get_route_registry()
        route = registry.get(skill_name)
        if not route:
            logger.warning("Route not found: %s, falling back to content chunk", skill_name)
            yield orchestrator_pb2.ChatChunk(
                type="content",
                message="Route not found. Please try again.",
                timestamp=datetime.now().isoformat(),
                agent_name=skill_name or "system",
            )
            yield orchestrator_pb2.ChatChunk(
                type="complete",
                message="Complete",
                timestamp=datetime.now().isoformat(),
                agent_name="system",
            )
            return

        agent_label = f"{skill_name}_agent" if not skill_name.endswith("_agent") else skill_name

        from orchestrator.routes.definitions import resolve_custom_skill_profile_id

        uid = metadata.get("user_id") or "system"
        pid = resolve_custom_skill_profile_id(uid, skill_name)
        if not pid:
            pid = metadata.get("default_agent_profile_id")

        if not pid and uid != "system":
            try:
                from orchestrator.backend_tool_client import get_backend_tool_client
                client = await get_backend_tool_client()
                pid = await client.ensure_default_profile(uid)
                if pid:
                    logger.info(
                        "UnifiedDispatcher: ensured default profile %s for user %s on route %s",
                        pid, uid, skill_name,
                    )
            except Exception as e:
                logger.warning("UnifiedDispatcher: ensure_default_profile failed for user %s: %s", uid, e)

        if pid:
            async for c in self.dispatch_custom_agent(pid, query, metadata, messages, cancellation_token):
                yield c
            return
        logger.warning(
            "UnifiedDispatcher: route %s has no agent_profile_id for user %s (engine=%s)",
            skill_name,
            uid,
            route.engine,
        )
        yield orchestrator_pb2.ChatChunk(
            type="content",
            message="No agent profile is available for this route. Set a default chat agent in settings or select an agent.",
            timestamp=datetime.now().isoformat(),
            agent_name=agent_label,
        )
        yield orchestrator_pb2.ChatChunk(
            type="complete",
            message="Complete",
            timestamp=datetime.now().isoformat(),
            agent_name="system",
        )


_dispatcher_instance: Optional[UnifiedDispatcher] = None


def get_unified_dispatcher() -> UnifiedDispatcher:
    """Return singleton dispatcher."""
    global _dispatcher_instance
    if _dispatcher_instance is None:
        _dispatcher_instance = UnifiedDispatcher()
    return _dispatcher_instance
