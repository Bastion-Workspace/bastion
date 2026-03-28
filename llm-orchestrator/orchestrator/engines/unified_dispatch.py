"""
Unified Dispatcher - Single entry point for route-based dispatch.

All requests flow through this: discover route -> pick engine -> run -> yield ChatChunk stream.
"""

import json
import logging
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

from protos import orchestrator_pb2

from orchestrator.routes import get_route_registry, load_all_routes
from orchestrator.routes.route_schema import EngineType

logger = logging.getLogger(__name__)

_routes_loaded = False


def _ensure_routes_loaded() -> None:
    global _routes_loaded
    if not _routes_loaded:
        load_all_routes()
        _routes_loaded = True


class UnifiedDispatcher:
    """
    Dispatches requests to the appropriate engine based on discovered skill.
    Yields ChatChunk stream (status, content, complete, diagram, chart, editor_operation as needed).
    """

    def __init__(self) -> None:
        self._research_engine = None
        self._conversational_engine = None

    def _get_conversational_engine(self):
        if self._conversational_engine is None:
            from orchestrator.engines.conversational_engine import ConversationalEngine
            self._conversational_engine = ConversationalEngine()
        return self._conversational_engine

    def _get_research_engine(self):
        if self._research_engine is None:
            from orchestrator.engines.research_engine import ResearchEngine
            self._research_engine = ResearchEngine()
        return self._research_engine

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

        content_metadata = {}
        if agent_display_name:
            content_metadata["agent_display_name"] = agent_display_name
        yield orchestrator_pb2.ChatChunk(
            type="content",
            message=content,
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            metadata=content_metadata if content_metadata else None,
        )

        complete_metadata = {"agent_profile_id": agent_profile_id}
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
        tools_cats = result.get("tools_used_categories") or []
        if tools_cats:
            complete_metadata["tools_used_categories"] = json.dumps(tools_cats)
        tool_call_summary = result.get("tool_call_summary", "")
        if tool_call_summary:
            complete_metadata["tool_call_summary"] = tool_call_summary
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

        content_metadata = {}
        if agent_display_name:
            content_metadata["agent_display_name"] = agent_display_name
        yield orchestrator_pb2.ChatChunk(
            type="content",
            message=content,
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            metadata=content_metadata if content_metadata else None,
        )

        complete_metadata = {"agent_profile_id": ceo_id}
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

        if route.engine == EngineType.RESEARCH:
            yield orchestrator_pb2.ChatChunk(
                type="status",
                message=f"Research: {route.description[:50]}...",
                timestamp=datetime.now().isoformat(),
                agent_name=agent_label,
            )
            engine = self._get_research_engine()
            try:
                result = await engine.process(
                    query=query,
                    metadata=metadata,
                    messages=messages,
                    skill_name=skill_name,
                    cancellation_token=cancellation_token,
                )
            except Exception as e:
                logger.exception("Research engine failed: %s", e)
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=f"Error: {e}",
                    timestamp=datetime.now().isoformat(),
                    agent_name=agent_label,
                )
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message="Complete",
                    timestamp=datetime.now().isoformat(),
                    agent_name="system",
                )
                return
            # FullResearchAgent returns dict with "response", "images", citations, etc. (AgentResponse contract)
            response_text = result.get("response", "")
            if isinstance(response_text, dict):
                response_text = response_text.get("response", response_text.get("message", "")) or ""
            else:
                response_text = str(response_text) if response_text else ""
            yield orchestrator_pb2.ChatChunk(
                type="content",
                message=response_text or "Done.",
                timestamp=datetime.now().isoformat(),
                agent_name=agent_label,
            )
            # Include structured images and other metadata in complete chunk (frontend expects metadata.images)
            images = result.get("images") or result.get("structured_images")
            chunk_metadata = {}
            if images:
                chunk_metadata["images"] = json.dumps(images)
                logger.info("Research dispatch: including %d image(s) in complete metadata", len(images))
            for key in ("citations", "sources", "static_visualization_data", "static_format", "chart_result"):
                val = result.get(key)
                if val is not None:
                    chunk_metadata[key] = json.dumps(val) if not isinstance(val, str) else val
            prev_tools = (result.get("shared_memory") or {}).get("previous_tools_used") or []
            if prev_tools:
                from orchestrator.utils.action_io_registry import get_categories_for_tools
                chunk_metadata["tools_used_categories"] = json.dumps(get_categories_for_tools(prev_tools))
            yield orchestrator_pb2.ChatChunk(
                type="complete",
                message="Complete",
                timestamp=datetime.now().isoformat(),
                agent_name="system",
                metadata=chunk_metadata if chunk_metadata else None,
            )
            return

        if route.engine == EngineType.CONVERSATIONAL:
            yield orchestrator_pb2.ChatChunk(
                type="status",
                message=f"Chat: {route.description[:50]}...",
                timestamp=datetime.now().isoformat(),
                agent_name=agent_label,
            )
            engine = self._get_conversational_engine()
            try:
                result = await engine.process(
                    query=query,
                    metadata=metadata,
                    messages=messages,
                    skill_name=skill_name,
                    cancellation_token=cancellation_token,
                )
            except Exception as e:
                logger.exception("Conversational engine failed: %s", e)
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=f"Error: {e}",
                    timestamp=datetime.now().isoformat(),
                    agent_name=agent_label,
                )
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message="Complete",
                    timestamp=datetime.now().isoformat(),
                    agent_name="system",
                )
                return
            result_status = result.get("task_status") or (result.get("response", {}) or {}).get("task_status", "")
            if result_status == "rejected":
                yield orchestrator_pb2.ChatChunk(
                    type="rejected",
                    message="Route rejected this query",
                    timestamp=datetime.now().isoformat(),
                    agent_name=agent_label,
                )
                return
            # ChatAgent returns result with "response" key (AgentResponse dict or handoff data)
            response = result.get("response", "")
            if isinstance(response, dict):
                response_text = response.get("response", response.get("message", "")) or ""
            else:
                response_text = str(response) if response else ""
            yield orchestrator_pb2.ChatChunk(
                type="content",
                message=response_text or "Done.",
                timestamp=datetime.now().isoformat(),
                agent_name=agent_label,
            )
            # Include images and other metadata when chat handed off to research (same as RESEARCH path)
            images = result.get("images") or result.get("structured_images")
            chunk_metadata = {}
            if images:
                chunk_metadata["images"] = json.dumps(images)
                logger.info("Chat dispatch: including %d image(s) in complete metadata (handoff)", len(images))
            for key in ("citations", "sources", "static_visualization_data", "static_format", "chart_result"):
                val = result.get(key)
                if val is not None:
                    chunk_metadata[key] = json.dumps(val) if not isinstance(val, str) else val
            yield orchestrator_pb2.ChatChunk(
                type="complete",
                message="Complete",
                timestamp=datetime.now().isoformat(),
                agent_name="system",
                metadata=chunk_metadata if chunk_metadata else None,
            )
            return

        if route.engine == EngineType.CUSTOM_AGENT:
            from orchestrator.routes.definitions import resolve_custom_skill_profile_id
            uid = metadata.get("user_id") or "system"
            pid = resolve_custom_skill_profile_id(uid, skill_name)
            if not pid:
                pid = metadata.get("default_agent_profile_id")
            if pid:
                async for c in self.dispatch_custom_agent(pid, query, metadata, messages, cancellation_token):
                    yield c
                return
            logger.warning("UnifiedDispatcher: CUSTOM_AGENT route %s has no profile_id for user %s", skill_name, uid)

        logger.warning("UnifiedDispatcher: engine %s not yet implemented, route=%s", route.engine, skill_name)
        yield orchestrator_pb2.ChatChunk(
            type="content",
            message="This capability is not yet available via route dispatch.",
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
