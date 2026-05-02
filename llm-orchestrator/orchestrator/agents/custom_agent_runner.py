"""
Custom Agent Runner - LangGraph workflow for Agent Factory custom agents.

Loads profile and playbook from the backend, executes the pipeline via a dynamic
LangGraph (playbook_graph_builder), and formats the final response.
Output is handled by tool steps within the playbook (e.g. send_channel_message).
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import AIMessage, HumanMessage

from config.settings import settings
from orchestrator.agents.base_agent import BaseAgent
from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.engines.playbook_graph_builder import build_playbook_graph
from orchestrator.utils.line_context import line_id_from_metadata
from orchestrator.middleware.message_preprocessor import MessagePreprocessor
from orchestrator.middleware.summarization_node import SummarizationNode
from orchestrator.utils.message_sanitizer import strip_tool_actions_prefix
from orchestrator.utils.async_invoke_timeout import invoke_with_optional_timeout
from orchestrator.checkpointer import clear_checkpoint_thread

logger = logging.getLogger(__name__)

_WIKILINK_PARSE_RE = re.compile(r"\[\[([^\]|]{1,200})(?:\|[^\]]*)?\]\]")
_WIKILINK_SKIP_PREFIXES = ("file:", "id:", "http://", "https://", "#")


async def _await_cancelable_ainvoke(
    coro,
    cancellation_token: Optional[asyncio.Event],
    timeout_sec: Optional[float] = None,
):
    """Wait for a coroutine unless cancellation_token is set (gRPC Stop / client disconnect)."""

    async def _work():
        return await invoke_with_optional_timeout(coro, timeout_sec)

    if cancellation_token is None:
        return await _work()
    main_task = asyncio.create_task(_work())
    wait_task = asyncio.create_task(cancellation_token.wait())
    done, _ = await asyncio.wait(
        {main_task, wait_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    if wait_task in done and cancellation_token.is_set():
        if not main_task.done():
            main_task.cancel()
        try:
            await main_task
        except asyncio.CancelledError:
            pass
        raise asyncio.CancelledError()
    wait_task.cancel()
    try:
        await wait_task
    except asyncio.CancelledError:
        pass
    return main_task.result()


def _format_user_context(profile: Dict[str, Any]) -> str:
    """Format user profile for system prompt (name, email, timezone, zip, ai_context)."""
    parts = []
    name = profile.get("preferred_name") or profile.get("display_name") or profile.get("username")
    if name:
        parts.append(f"User name: {name}")
    if profile.get("email"):
        parts.append(f"User email: {profile['email']}")
    if profile.get("timezone"):
        parts.append(f"User timezone: {profile['timezone']}")
    if profile.get("zip_code"):
        parts.append(f"User ZIP: {profile['zip_code']}")
    if profile.get("ai_context"):
        parts.append(f"User context: {profile['ai_context']}")
    return "\n".join(parts)


def _definition_steps(playbook: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return playbook definition steps, tolerating definition as dict or JSON string."""
    definition = playbook.get("definition") or {}
    if isinstance(definition, str):
        try:
            definition = json.loads(definition) if definition else {}
        except (json.JSONDecodeError, TypeError):
            definition = {}
    if not isinstance(definition, dict):
        return []
    return definition.get("steps", [])


def _resolve_heading_level(steps: List[Dict[str, Any]]) -> int:
    """
    Resolve heading level for editor/ref location tracking (sections, current/previous/next).
    First top-level step with heading_level set (1-6) wins; else default 2.
    """
    for step in (steps or []):
        if not isinstance(step, dict):
            continue
        val = step.get("heading_level")
        if val is None:
            continue
        try:
            level = int(val)
            if 1 <= level <= 6:
                return level
        except (TypeError, ValueError):
            continue
    return 2


def _aggregate_acquired_tool_log(execution_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten mid-loop skill acquisition entries from all llm_agent steps in the trace."""
    out: List[Dict[str, Any]] = []
    for entry in execution_trace or []:
        chunk = entry.get("acquired_tool_log")
        if isinstance(chunk, list):
            out.extend(c for c in chunk if isinstance(c, dict))
    return out


def _aggregate_skill_execution_events(execution_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten _skill_execution_events from all steps in the trace."""
    out: List[Dict[str, Any]] = []
    for entry in execution_trace or []:
        chunk = entry.get("skill_execution_events")
        if isinstance(chunk, list):
            out.extend(c for c in chunk if isinstance(c, dict))
    return out


def _build_tool_call_summary(execution_trace: List[Dict[str, Any]]) -> str:
    """
    Build a concise summary of tool calls from execution_trace for conversation history.
    Format: "tool_name(key_args) -> brief_result; ..." truncated to MAX_SUMMARY_CHARS.
    Pending operations (confirmed=False) get a higher result budget so the next turn can replay them.
    """
    if not execution_trace:
        return ""
    MAX_SUMMARY_CHARS = 1500
    MAX_RESULT_CHARS = 150
    MAX_PENDING_RESULT_CHARS = 800
    parts: List[str] = []
    total_len = 0
    for entry in execution_trace:
        tool_trace = entry.get("tool_call_trace") or []
        for tc in tool_trace:
            name = tc.get("tool_name", "?")
            args = tc.get("args") or {}
            is_pending = args.get("confirmed") is False
            max_result = MAX_PENDING_RESULT_CHARS if is_pending else MAX_RESULT_CHARS
            args_str = ", ".join(f"{k}={repr(v)}" for k, v in sorted(args.items()) if k not in ("user_id", "_pipeline_metadata"))
            if args_str:
                call_str = f"{name}({args_str})"
            else:
                call_str = f"{name}()"
            result_raw = tc.get("result") or ""
            result_brief = (result_raw[:max_result] + "...") if len(result_raw) > max_result else result_raw
            if result_brief:
                part = f"{call_str} -> {result_brief}"
            else:
                part = call_str
            if total_len + len(part) + 2 > MAX_SUMMARY_CHARS:
                if parts:
                    return "; ".join(parts) + "; ..."
                return part[:MAX_SUMMARY_CHARS]
            parts.append(part)
            total_len += len(part) + 2
    return "; ".join(parts) if parts else ""


class CustomAgentState(TypedDict, total=False):
    """State for the custom agent LangGraph workflow."""

    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    agent_profile: Dict[str, Any]
    playbook: Dict[str, Any]
    playbook_id: str
    pipeline_results: Dict[str, Any]
    pending_approval: Optional[Dict[str, Any]]
    pending_auth: Optional[Dict[str, Any]]
    playbook_config: Optional[Dict[str, Any]]
    typed_outputs: Dict[str, Any]
    response: Dict[str, Any]
    task_status: str
    error: str
    execution_trace: List[Dict[str, Any]]


class CustomAgentRunner(BaseAgent):
    """
    Executes Agent Factory custom agents: load profile/playbook, run pipeline, format response.
    """

    def __init__(self) -> None:
        super().__init__("custom_agent")
        self._stream_cancellation_token: Optional[asyncio.Event] = None

    def _build_workflow(self, checkpointer: AsyncPostgresSaver) -> StateGraph:
        workflow = StateGraph(CustomAgentState)
        workflow.add_node("load_profile", self._load_profile_node)
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("summarize_history", self._summarize_history_node)
        workflow.add_node("execute_pipeline", self._execute_pipeline_node)
        workflow.add_node("format_response", self._format_response_node)
        workflow.add_node("approval_gate", self._approval_gate_node)

        workflow.set_entry_point("load_profile")
        workflow.add_edge("load_profile", "prepare_context")
        workflow.add_edge("prepare_context", "summarize_history")
        workflow.add_edge("summarize_history", "execute_pipeline")
        workflow.add_conditional_edges(
            "execute_pipeline",
            self._route_after_pipeline,
            {
                "approval": "approval_gate",
                "continue": "format_response",
            },
        )
        workflow.add_edge("format_response", END)
        workflow.add_conditional_edges(
            "approval_gate",
            self._route_after_approval,
            {"format_response": "format_response", "end": END},
        )

        return workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=["approval_gate"],
        )

    def _route_after_pipeline(self, state: CustomAgentState) -> str:
        if state.get("pending_approval") or state.get("pending_auth"):
            return "approval"
        return "continue"

    def _route_after_approval(self, state: CustomAgentState) -> str:
        """After approval_gate: continue to format_response if no pending approval/auth, else END (stay interrupted)."""
        if state.get("pending_approval") is None and state.get("pending_auth") is None:
            return "format_response"
        return "end"

    async def _load_profile_node(self, state: CustomAgentState) -> Dict[str, Any]:
        """Fetch agent profile and playbook from backend."""
        metadata = state.get("metadata", {})
        user_id = state.get("user_id", "system")
        query = state.get("query", "")

        profile_id = metadata.get("agent_profile_id")
        if not profile_id:
            return {
                "response": {"formatted": "No agent_profile_id in metadata."},
                "task_status": "error",
                "error": "Missing agent_profile_id",
                "metadata": metadata,
                "user_id": user_id,
                "query": query,
                "messages": state.get("messages", []),
                "shared_memory": state.get("shared_memory", {}),
            }

        try:
            client = await get_backend_tool_client()
            profile = await client.get_agent_profile(user_id, profile_id)
            if not profile:
                return {
                    "response": {"formatted": "Profile not found or access denied."},
                    "task_status": "error",
                    "error": "Profile not found",
                    "metadata": metadata,
                    "user_id": user_id,
                    "query": query,
                    "messages": state.get("messages", []),
                    "shared_memory": state.get("shared_memory", {}),
                }

            playbook_id = metadata.get("playbook_id") or profile.get("default_playbook_id")
            if not playbook_id:
                return {
                    "response": {
                        "formatted": f"Profile '{profile.get('name', '')}' has no playbook linked. Set playbook_id or default_playbook_id."
                    },
                    "task_status": "error",
                    "error": "No playbook",
                    "metadata": metadata,
                    "user_id": user_id,
                    "query": query,
                    "messages": state.get("messages", []),
                    "shared_memory": state.get("shared_memory", {}),
                }

            playbook = await client.get_playbook(user_id, playbook_id)
            if not playbook:
                return {
                    "response": {"formatted": "Playbook not found or access denied."},
                    "task_status": "error",
                    "error": "Playbook not found",
                    "metadata": metadata,
                    "user_id": user_id,
                    "query": query,
                    "messages": state.get("messages", []),
                    "shared_memory": state.get("shared_memory", {}),
                }

            steps = _definition_steps(playbook)
            if not steps:
                return {
                    "response": {"formatted": f"Playbook '{playbook.get('name', '')}' has no steps defined."},
                    "task_status": "error",
                    "error": "No steps",
                    "metadata": metadata,
                    "user_id": user_id,
                    "query": query,
                    "messages": state.get("messages", []),
                    "shared_memory": state.get("shared_memory", {}),
                }

            prompt_history_enabled = profile.get("prompt_history_enabled", profile.get("chat_history_enabled", False))
            chat_history_lookback = profile.get("chat_history_lookback", 10)
            persona_mode = profile.get("persona_mode") or "none"
            apply_persona = persona_mode in ("default", "specific")
            profile_persona_id = profile.get("persona_id")
            logger.info(
                "Custom agent profile %s: persona_mode=%s persona_id=%s embedded_persona=%s",
                profile_id,
                persona_mode,
                profile_persona_id,
                bool(profile.get("persona")),
            )

            def _user_default_persona() -> Optional[Dict[str, Any]]:
                return (
                    metadata.get("persona")
                    or (state.get("shared_memory", {}) or {}).get("persona")
                )

            def _alex_fallback_persona() -> Dict[str, Any]:
                fallback_timezone = (
                    metadata.get("user_timezone")
                    or (state.get("shared_memory", {}) or {}).get("user_timezone")
                    or "UTC"
                )
                return {
                    "ai_name": "Alex",
                    "persona_style": "professional",
                    "political_bias": "neutral",
                    "timezone": fallback_timezone,
                }

            resolved_persona: Optional[Dict[str, Any]] = None
            if persona_mode == "specific":
                if profile.get("persona"):
                    resolved_persona = profile["persona"]
                    logger.info(
                        "Custom agent profile %s: using specific persona name=%s ai_name=%s",
                        profile_id,
                        resolved_persona.get("name"),
                        resolved_persona.get("ai_name"),
                    )
                else:
                    logger.warning(
                        "Custom agent profile %s: persona_mode=specific but persona not embedded "
                        "(persona_id=%s); falling back to user default persona from request metadata",
                        profile_id,
                        profile_persona_id,
                    )
                    resolved_persona = _user_default_persona()
                    if not resolved_persona:
                        resolved_persona = _alex_fallback_persona()
                        logger.warning(
                            "Custom agent profile %s: specific persona fallback had no user default; "
                            "using professional fallback persona",
                            profile_id,
                        )
            elif persona_mode == "default":
                resolved_persona = _user_default_persona()
                if not resolved_persona:
                    resolved_persona = _alex_fallback_persona()
                    logger.warning(
                        "Default persona metadata missing for profile %s; using professional fallback persona",
                        profile_id,
                    )
            else:
                resolved_persona = None
            allowed_conns = profile.get("allowed_connections") or []
            filtered_connections_map = metadata.get("active_connections_map", "")
            if isinstance(allowed_conns, list) and len(allowed_conns) > 0:
                try:
                    from orchestrator.engines.tool_resolution import connection_allow_ids_from_entries

                    allowed_ids = connection_allow_ids_from_entries(
                        [e for e in allowed_conns if isinstance(e, dict)]
                    )
                    if not allowed_ids:
                        logger.warning(
                            "allowed_connections has %d entries but none parsed to valid "
                            "connection IDs (profile_id=%s); treating as unrestricted (all accounts)",
                            len(allowed_conns),
                            profile_id,
                        )
                    else:
                        raw_cmap_str = metadata.get("active_connections_map", "")
                        full_cmap: Dict[str, Any] = {}
                        if isinstance(raw_cmap_str, str) and raw_cmap_str.strip():
                            parsed = json.loads(raw_cmap_str)
                            if isinstance(parsed, dict):
                                full_cmap = parsed
                        filtered: Dict[str, Any] = {}
                        for ctype, entries in full_cmap.items():
                            if not isinstance(entries, list):
                                continue
                            kept = [
                                e
                                for e in entries
                                if isinstance(e, dict)
                                and e.get("id") is not None
                                and int(e["id"]) in allowed_ids
                            ]
                            if kept:
                                filtered[str(ctype)] = kept
                        filtered_connections_map = json.dumps(filtered)
                        if full_cmap and not filtered:
                            logger.warning(
                                "allowed_connections filter removed all connections "
                                "(profile_id=%s incoming_types=%s raw_allow_entries=%s "
                                "parsed_allow_ids=%s)",
                                profile_id,
                                sorted(full_cmap.keys()),
                                len(allowed_conns),
                                sorted(allowed_ids),
                            )
                        elif filtered:
                            logger.info(
                                "Custom agent profile %s: active_connections_map after "
                                "allowed_connections filter types=%s",
                                profile_id,
                                sorted(filtered.keys()),
                            )
                        elif not full_cmap and allowed_conns:
                            logger.info(
                                "Custom agent profile %s: allowed_connections set but "
                                "incoming active_connections_map was empty",
                                profile_id,
                            )
                except (TypeError, ValueError, json.JSONDecodeError) as _e:
                    logger.warning(
                        "allowed_connections filter skipped (invalid data): profile_id=%s error=%s",
                        profile_id,
                        _e,
                    )

            metadata_with_name = {
                **metadata,
                "active_connections_map": filtered_connections_map,
                "agent_profile_name": profile.get("name", ""),
                "prompt_history_enabled": prompt_history_enabled,
                "chat_history_lookback": chat_history_lookback,
                "persona_enabled": apply_persona,
                "persona_mode": persona_mode,
                "persona": resolved_persona if apply_persona else None,
                "system_prompt_additions": profile.get("system_prompt_additions") or "",
                "include_user_context": profile.get("include_user_context", False),
                "include_datetime_context": profile.get("include_datetime_context", True),
                "include_user_facts": profile.get("include_user_facts", False),
                "include_facts_categories": profile.get("include_facts_categories") or [],
                "use_themed_memory": profile.get("use_themed_memory", True),
                "include_agent_memory": profile.get("include_agent_memory", False),
            }

            return {
                "agent_profile": profile,
                "playbook": playbook,
                "playbook_id": playbook_id,
                "metadata": metadata_with_name,
                "user_id": user_id,
                "query": query,
                "messages": state.get("messages", []),
                "shared_memory": state.get("shared_memory", {}),
            }
        except Exception as e:
            logger.exception("Load profile failed: %s", e)
            return {
                "response": {"formatted": f"Failed to load profile: {str(e)}"},
                "task_status": "error",
                "error": str(e),
                "metadata": metadata,
                "user_id": user_id,
                "query": query,
                "messages": state.get("messages", []),
                "shared_memory": state.get("shared_memory", {}),
            }

    async def _prepare_context_node(self, state: CustomAgentState) -> Dict[str, Any]:
        """Build initial context from shared_memory and active_editor. Auto-inject workspace schema if profile or agent line has data_workspace_config."""
        shared_memory = dict(state.get("shared_memory", {}))
        profile = state.get("agent_profile", {})
        metadata = state.get("metadata", {}) or {}

        def _parse_profile_dw(raw: Any) -> Dict[str, Any]:
            if isinstance(raw, str):
                try:
                    return json.loads(raw) if raw.strip() else {}
                except (json.JSONDecodeError, TypeError):
                    return {}
            if isinstance(raw, dict):
                return raw
            return {}

        profile_dw = _parse_profile_dw(profile.get("data_workspace_config"))
        line_raw = metadata.get("line_data_workspace_config")
        if isinstance(line_raw, str):
            try:
                line_dw = json.loads(line_raw) if line_raw.strip() else {}
            except (json.JSONDecodeError, TypeError):
                line_dw = {}
        elif isinstance(line_raw, dict):
            line_dw = line_raw
        else:
            line_dw = {}

        workspace_access_modes: Dict[str, str] = {}
        profile_ids = list(profile_dw.get("workspace_ids") or [])
        for wid in profile_ids:
            if wid:
                workspace_access_modes[str(wid)] = "read_write"

        for entry in (line_dw.get("workspaces") or []):
            if not isinstance(entry, dict) or not entry.get("workspace_id"):
                continue
            wid = str(entry["workspace_id"]).strip()
            if not wid:
                continue
            acc = (entry.get("access") or "read").strip().lower()
            workspace_access_modes[wid] = "read_write" if acc == "read_write" else "read"

        merged_ids: list = []
        seen_m: set = set()
        for wid in profile_ids:
            w = str(wid).strip() if wid else ""
            if w and w not in seen_m:
                seen_m.add(w)
                merged_ids.append(w)
        for entry in (line_dw.get("workspaces") or []):
            if not isinstance(entry, dict) or not entry.get("workspace_id"):
                continue
            w = str(entry["workspace_id"]).strip()
            if w and w not in seen_m:
                seen_m.add(w)
                merged_ids.append(w)

        auto_inject = bool(profile_dw.get("auto_inject_schema")) or bool(line_dw.get("auto_inject_schema"))
        instr_parts = []
        pi = (profile_dw.get("context_instructions") or "").strip()
        li = (line_dw.get("context_instructions") or "").strip()
        if pi:
            instr_parts.append(pi)
        if li:
            instr_parts.append(li)
        merged_instructions = "\n\n".join(instr_parts)

        if auto_inject and merged_ids:
            try:
                client = await get_backend_tool_client()
                user_id = state.get("user_id", "system")
                workspace_schemas = []
                for ws_id in merged_ids:
                    schema_result = await client.get_workspace_schema(
                        workspace_id=ws_id,
                        user_id=user_id
                    )
                    if schema_result.get("tables") is not None:
                        workspace_schemas.append(schema_result)
                if workspace_schemas:
                    shared_memory["workspace_schemas"] = workspace_schemas
                    shared_memory["workspace_ids"] = list(merged_ids)
                    shared_memory["workspace_context_instructions"] = merged_instructions
                    shared_memory["workspace_access_modes"] = workspace_access_modes
            except Exception as e:
                logger.warning("Failed to auto-inject workspace schema: %s", e)
        elif merged_ids:
            shared_memory["workspace_ids"] = list(merged_ids)
            shared_memory["workspace_access_modes"] = workspace_access_modes
            if merged_instructions:
                shared_memory["workspace_context_instructions"] = merged_instructions
        return {
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "query": state.get("query", ""),
            "messages": state.get("messages", []),
            "shared_memory": shared_memory,
            "agent_profile": profile,
            "playbook": state.get("playbook", {}),
            "playbook_id": state.get("playbook_id", ""),
        }

    async def _summarize_history_node(self, state: CustomAgentState) -> Dict[str, Any]:
        """
        Compress long conversation history before playbook runs when prompt history is enabled.
        Uses SummarizationNode thresholds from agent_profile (optional).
        """
        metadata = state.get("metadata", {}) or {}
        if not metadata.get("prompt_history_enabled"):
            return {}
        agent_profile = state.get("agent_profile", {}) or {}
        threshold = int(agent_profile.get("summary_threshold_tokens", 5000))
        keep = int(agent_profile.get("summary_keep_messages", 10))
        summarizer = SummarizationNode(trigger_tokens=threshold, keep_messages=keep)
        summarized = await summarizer({"messages": state.get("messages", [])})
        new_messages = summarized.get("messages", state.get("messages", []))
        return {
            "messages": new_messages,
            "metadata": metadata,
            "user_id": state.get("user_id", "system"),
            "query": state.get("query", ""),
            "shared_memory": state.get("shared_memory", {}),
            "agent_profile": agent_profile,
            "playbook": state.get("playbook", {}),
            "playbook_id": state.get("playbook_id", ""),
        }

    async def _execute_pipeline_node(self, state: CustomAgentState) -> Dict[str, Any]:
        """Run playbook steps via dynamic LangGraph (each step is a node)."""
        if state.get("task_status") == "error":
            return {
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "query": state.get("query", ""),
                "messages": state.get("messages", []),
                "shared_memory": state.get("shared_memory", {}),
                "response": state.get("response", {}),
                "task_status": state.get("task_status", ""),
                "error": state.get("error", ""),
            }

        playbook = state.get("playbook", {})
        steps = _definition_steps(playbook)
        heading_level = _resolve_heading_level(steps)
        user_id = state.get("user_id", "system")
        metadata = state.get("metadata", {})
        query = state.get("query", "")

        playbook_id = state.get("playbook_id", "")
        shared_memory = dict(state.get("shared_memory") or {})
        state["shared_memory"] = shared_memory
        inputs: Dict[str, Any] = {"query": query}
        inputs["query_length"] = str(len(query))
        user_weather_location = (metadata or {}).get("user_weather_location") or ""
        if user_weather_location:
            inputs["user_weather_location"] = user_weather_location
        trigger_input = (metadata or {}).get("trigger_input")
        if trigger_input is not None:
            inputs["trigger_input"] = trigger_input

        from orchestrator.checkpointer import get_async_postgres_saver
        checkpointer = await get_async_postgres_saver()
        playbook_thread_id = f"playbook_{user_id}_{playbook_id}"
        playbook_config = {"configurable": {"thread_id": playbook_thread_id}}

        pipeline_metadata = dict(metadata or {})
        pipeline_metadata["playbook_config"] = playbook_config
        pipeline_metadata["thread_id"] = playbook_thread_id
        pipeline_metadata["langgraph_thread_id"] = playbook_thread_id
        if shared_memory.get("workspace_ids") is not None:
            pipeline_metadata["workspace_ids"] = shared_memory.get("workspace_ids")
        if shared_memory.get("workspace_access_modes") is not None:
            pipeline_metadata["workspace_access_modes"] = shared_memory.get("workspace_access_modes")
        if pipeline_metadata.get("prompt_history_enabled"):
            lookback = max(1, min(50, pipeline_metadata.get("chat_history_lookback", 10)))
            messages = state.get("messages", [])
            pipeline_metadata["conversation_messages"] = messages[-lookback:] if messages else []

        if pipeline_metadata.get("prompt_history_enabled"):
            conv_msgs = pipeline_metadata.get("conversation_messages") or []
            to_process = [m for m in conv_msgs if isinstance(m, (HumanMessage, AIMessage))]
            cleaned = MessagePreprocessor.preprocess_history(
                to_process,
                limit=len(to_process) if to_process else 0,
                sanitize_ai_responses=False,
            )
            parts = [
                f"{'USER' if row['role'] == 'user' else 'ASSISTANT'}: {row['content']}"
                for row in cleaned
            ]
            if parts:
                inputs["history"] = "=== CONVERSATION HISTORY ===\n" + "\n".join(parts) + "\n"
            else:
                inputs["history"] = ""
        else:
            inputs["history"] = ""

        if pipeline_metadata.get("include_user_context"):
            try:
                client = await get_backend_tool_client()
                profile_data = await client.get_my_profile(user_id=user_id)
                pipeline_metadata["user_context_str"] = _format_user_context(profile_data)
            except Exception as e:
                logger.warning("Failed to load user context for system prompt: %s", e)
                pipeline_metadata["user_context_str"] = ""
        else:
            pipeline_metadata["user_context_str"] = ""

        if pipeline_metadata.get("include_user_facts"):
            try:
                client = await get_backend_tool_client()
                use_themed = pipeline_metadata.get("use_themed_memory", True)
                q = (state.get("query") or "").strip()
                result = await client.get_user_facts(
                    user_id=user_id,
                    query=q if use_themed else "",
                    use_themed_memory=use_themed,
                )
                if result.get("success") and result.get("facts"):
                    facts = result["facts"]
                    categories = pipeline_metadata.get("include_facts_categories") or []
                    if isinstance(categories, list) and len(categories) > 0:
                        categories_set = set(categories)
                        facts = [f for f in facts if (f.get("category") or "general") in categories_set]
                    from orchestrator.utils.fact_utils import format_user_facts_for_prompt
                    pipeline_metadata["user_facts_str"] = format_user_facts_for_prompt(facts)
                else:
                    pipeline_metadata["user_facts_str"] = ""
            except Exception as e:
                logger.warning("Failed to load user facts for system prompt: %s", e)
                pipeline_metadata["user_facts_str"] = ""
        else:
            pipeline_metadata["user_facts_str"] = ""

        profile_id = (metadata or {}).get("agent_profile_id")
        if pipeline_metadata.get("include_agent_memory") and profile_id:
            try:
                client = await get_backend_tool_client()
                pid = str(profile_id)
                keys = await client.list_agent_memories(
                    user_id=user_id,
                    agent_profile_id=pid,
                )
                if keys:
                    parts = ["=== AGENT MEMORY ===\n"]
                    for k in keys[:20]:
                        val = await client.get_agent_memory(
                            user_id=user_id,
                            agent_profile_id=pid,
                            memory_key=k,
                        )
                        if val is not None:
                            parts.append(f"{k}: {json.dumps(val)[:300]}{'...' if len(json.dumps(val)) > 300 else ''}\n")
                    pipeline_metadata["agent_memory_str"] = "\n".join(parts)
                else:
                    pipeline_metadata["agent_memory_str"] = ""
            except Exception as e:
                logger.warning("Failed to load agent memory for system prompt: %s", e)
                pipeline_metadata["agent_memory_str"] = ""
        else:
            pipeline_metadata["agent_memory_str"] = ""

        line_id = line_id_from_metadata(metadata or {})
        agent_profile_id = (metadata or {}).get("agent_profile_id")
        if line_id and agent_profile_id:
            try:
                client = await get_backend_tool_client()
                goals_result = await client.get_goals_for_agent(
                    team_id=str(line_id),
                    agent_profile_id=str(agent_profile_id),
                    user_id=user_id,
                )
                goal_context_parts = []
                if goals_result.get("success") and goals_result.get("goals"):
                    goals = goals_result["goals"]
                    first_goal_id = (goals[0].get("id") if goals else None)
                    if first_goal_id:
                        ancestry_result = await client.get_goal_ancestry(goal_id=first_goal_id, user_id=user_id)
                        if ancestry_result.get("success") and ancestry_result.get("goals"):
                            ancestry = ancestry_result["goals"]
                            for i, g in enumerate(reversed(ancestry)):
                                prefix = "  " * i + ("-> " if i else "Mission: ")
                                goal_context_parts.append(f"{prefix}\"{g.get('title', '')}\" [{g.get('status', '')}] {g.get('progress_pct', 0)}%")
                                desc = (g.get("description") or "").strip()
                                if desc:
                                    goal_context_parts.append("  " * (i + 1) + "Brief: " + desc)
                            if goal_context_parts:
                                pipeline_metadata["goal_context_str"] = (
                                    "GOAL CONTEXT:\n"
                                    + "\n".join(goal_context_parts)
                                    + "\n\nProceed with the goal using the information above. Make reasonable assumptions where details are missing; only ask the user for clarification when you are truly blocked."
                                )
                            else:
                                pipeline_metadata["goal_context_str"] = ""
                        else:
                            pipeline_metadata["goal_context_str"] = ""
                    else:
                        pipeline_metadata["goal_context_str"] = ""
                else:
                    pipeline_metadata["goal_context_str"] = ""
            except Exception as e:
                logger.warning("Failed to load goal context for system prompt: %s", e)
                pipeline_metadata["goal_context_str"] = ""
        else:
            pipeline_metadata["goal_context_str"] = ""

        inputs["line_refs"] = ""
        inputs["line_ref_count"] = "0"
        inputs["line_ref_ids"] = "[]"
        inputs["line_ref_skipped_count"] = "0"
        line_ref_config_raw = (metadata or {}).get("line_reference_config")
        if line_ref_config_raw:
            try:
                import json as _json

                ref_config = (
                    _json.loads(line_ref_config_raw)
                    if isinstance(line_ref_config_raw, str)
                    else line_ref_config_raw
                )
                if isinstance(ref_config, dict) and ref_config:
                    shared_memory["line_reference_config"] = ref_config
                    from orchestrator.utils.line_reference_loader import (
                        line_ref_safe_key,
                        load_line_references,
                    )

                    ref_result = await load_line_references(ref_config, user_id)
                    inputs["line_refs"] = ref_result.get("combined", "")
                    inputs["line_ref_count"] = str(ref_result.get("file_count", 0))
                    inputs["line_ref_skipped_count"] = str(
                        ref_result.get("skipped_count", 0)
                    )
                    try:
                        inputs["line_ref_ids"] = _json.dumps(
                            ref_result.get("ref_ids") or [],
                            ensure_ascii=False,
                        )
                    except Exception:
                        inputs["line_ref_ids"] = "[]"
                    for entry_name, content in (ref_result.get("by_entry") or {}).items():
                        sk = line_ref_safe_key(entry_name)
                        inputs[f"line_ref_{sk}"] = content
            except Exception as e:
                logger.warning("Failed to load line reference files: %s", e)
                inputs["line_refs"] = ""
                inputs["line_ref_count"] = "0"
                inputs["line_ref_ids"] = "[]"
                inputs["line_ref_skipped_count"] = "0"

        include_datetime = pipeline_metadata.get("include_datetime_context", True)
        if include_datetime:
            user_tz = (
                state.get("shared_memory", {}).get("user_timezone")
                or state.get("metadata", {}).get("user_timezone")
                or "UTC"
            )
            try:
                from datetime import timezone as dt_timezone
                import pytz
                if user_tz.upper() == "UTC":
                    now = datetime.now(dt_timezone.utc)
                else:
                    tz = pytz.timezone(user_tz)
                    utc_naive = datetime.utcnow()
                    utc_aware = pytz.utc.localize(utc_naive)
                    now = utc_aware.astimezone(tz)
            except Exception:
                from datetime import timezone as dt_timezone
                now = datetime.now(dt_timezone.utc)
            inputs["today"] = now.strftime("%Y-%m-%dT00:00:00")
            inputs["tomorrow"] = (now + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")
            inputs["today_end"] = now.strftime("%Y-%m-%dT23:59:59")
            inputs["today_day_of_week"] = now.strftime("%A")
            pipeline_metadata["datetime_context_str"] = self._get_datetime_context(state)
        else:
            inputs["today"] = ""
            inputs["tomorrow"] = ""
            inputs["today_end"] = ""
            inputs["today_day_of_week"] = ""
            pipeline_metadata["datetime_context_str"] = ""

        editor_pref = (
            shared_memory.get("editor_preference")
            or metadata.get("editor_preference")
        )
        if editor_pref == "ignore":
            active_editor = {}
        else:
            active_editor = state.get("shared_memory", {}).get("active_editor", {})
        if active_editor and active_editor.get("content"):
            filename = active_editor.get("filename", "unknown")
            content = active_editor.get("content", "")
            inputs["editor"] = f"=== FILE: {filename} ===\n{content}\n=== END FILE ==="
            inputs["editor_document_id"] = active_editor.get("document_id") or ""
            inputs["editor_filename"] = filename
            inputs["editor_length"] = str(len(content))
        else:
            inputs["editor"] = ""
            inputs["editor_document_id"] = ""
            inputs["editor_filename"] = ""
            inputs["editor_length"] = "0"

        if active_editor:
            frontmatter = active_editor.get("frontmatter", {})
            inputs["editor_document_type"] = (frontmatter.get("type") or "").strip().lower()
            inputs["editor_cursor_offset"] = str(active_editor.get("cursor_offset", -1))
            sel_start = active_editor.get("selection_start", -1)
            sel_end = active_editor.get("selection_end", -1)
            content = active_editor.get("content", "")
            if sel_start >= 0 and sel_end > sel_start and content:
                inputs["editor_selection"] = content[sel_start:sel_end]
            else:
                inputs["editor_selection"] = ""
            cursor_offset = active_editor.get("cursor_offset", -1)
            if cursor_offset >= 0 and content:
                from orchestrator.utils.section_scoping import extract_scoped_context, find_heading_sections
                scoped = extract_scoped_context(content, cursor_offset, max_level=heading_level, adjacent=1)
                inputs["editor_current_section"] = scoped["current_section"]
                inputs["editor_current_heading"] = scoped["current_heading"]
                inputs["editor_section_index"] = str(scoped["current_section_index"])
                inputs["editor_previous_section"] = scoped["previous_section"]
                inputs["editor_next_section"] = scoped["next_section"]
                inputs["editor_adjacent_sections"] = scoped["adjacent_sections"]
                inputs["editor_total_sections"] = str(scoped["total_sections"])
                total_sections = scoped["total_sections"]
                idx = scoped["current_section_index"]
                inputs["editor_is_first_section"] = "true" if idx == 0 else ""
                inputs["editor_is_last_section"] = "true" if total_sections <= 1 or idx >= total_sections - 1 else ""
                toc_sections = find_heading_sections(content, max_level=heading_level)
                inputs["editor_toc"] = "\n".join(s.heading_text for s in toc_sections) if toc_sections else ""
            else:
                inputs["editor_current_section"] = ""
                inputs["editor_current_heading"] = ""
                inputs["editor_section_index"] = ""
                inputs["editor_previous_section"] = ""
                inputs["editor_next_section"] = ""
                inputs["editor_adjacent_sections"] = ""
                inputs["editor_total_sections"] = ""
                inputs["editor_is_first_section"] = ""
                inputs["editor_is_last_section"] = ""
                if content:
                    from orchestrator.utils.section_scoping import find_heading_sections
                    toc_sections = find_heading_sections(content, max_level=heading_level)
                    inputs["editor_toc"] = "\n".join(s.heading_text for s in toc_sections) if toc_sections else ""
                else:
                    inputs["editor_toc"] = ""
        else:
            inputs["editor_document_type"] = ""
            inputs["editor_cursor_offset"] = ""
            inputs["editor_selection"] = ""
            inputs["editor_current_section"] = ""
            inputs["editor_current_heading"] = ""
            inputs["editor_section_index"] = ""
            inputs["editor_previous_section"] = ""
            inputs["editor_next_section"] = ""
            inputs["editor_adjacent_sections"] = ""
            inputs["editor_total_sections"] = ""
            inputs["editor_is_first_section"] = ""
            inputs["editor_is_last_section"] = ""
            inputs["editor_toc"] = ""

        # Plain [[wikilinks]] from open file — ambient context for agents (no DB call)
        if active_editor and active_editor.get("content"):
            content_wl = active_editor["content"]
            raw_titles = _WIKILINK_PARSE_RE.findall(content_wl)
            plain: List[str] = []
            for t in dict.fromkeys(s.strip() for s in raw_titles):
                tl = t.lower()
                if (
                    t
                    and not any(tl.startswith(p) for p in _WIKILINK_SKIP_PREFIXES)
                    and "/" not in t
                    and "\\" not in t
                ):
                    plain.append(t)
                if len(plain) >= 15:
                    break
            inputs["editor_linked_notes"] = (
                ", ".join(f"[[{t}]]" for t in plain) if plain else ""
            )
        else:
            inputs["editor_linked_notes"] = ""

        active_artifact = shared_memory.get("active_artifact", {})
        if active_artifact and active_artifact.get("code"):
            inputs["previous_artifact"] = active_artifact["code"]
            inputs["previous_artifact_type"] = active_artifact.get("artifact_type", "")
            inputs["previous_artifact_title"] = active_artifact.get("title", "")
            inputs["previous_artifact_language"] = active_artifact.get("language", "")
        else:
            inputs["previous_artifact"] = ""
            inputs["previous_artifact_type"] = ""
            inputs["previous_artifact_title"] = ""
            inputs["previous_artifact_language"] = ""

        pin = self._get_valid_pinned_document(shared_memory)
        if pin:
            from orchestrator.tools.document_tools import get_document_content_tool
            content_result = await get_document_content_tool(
                pin["document_id"], user_id=user_id
            )
            doc_text = content_result.get("content", "")
            inputs["document_context"] = (
                f"=== DOCUMENT: {pin.get('title', '')} (ID: {pin['document_id']}) ===\n"
                f"{doc_text}\n=== END DOCUMENT ==="
            )
            inputs["pinned_document_id"] = pin["document_id"]
            self._pin_document(shared_memory, pin["document_id"])
        else:
            inputs["document_context"] = ""
            inputs["pinned_document_id"] = ""

        ltr = state.get("shared_memory", {}).get("last_tool_results")
        inputs["last_tool_results"] = json.dumps(ltr, indent=2, default=str) if ltr else ""

        all_prompts = " ".join(
            (s.get("prompt") or s.get("prompt_template") or "")
            for s in steps
            if isinstance(s, dict)
        )
        if active_editor:
            try:
                from orchestrator.tools.reference_file_loader import load_file_by_path, extract_ref_prefix_paths
                frontmatter = active_editor.get("frontmatter", {})
                ref_items = extract_ref_prefix_paths(frontmatter)
                if ref_items:
                    load_tasks = [
                        load_file_by_path(path, user_id, active_editor=active_editor)
                        for path, _ in ref_items
                    ]
                    docs = await asyncio.gather(*load_tasks, return_exceptions=True)

                    loaded_by_cat: Dict[str, List[Any]] = {}
                    for (_, category), doc in zip(ref_items, docs):
                        if not isinstance(doc, Exception) and doc and doc.get("found") and doc.get("content"):
                            loaded_by_cat.setdefault(category, []).append(doc)

                    blob_parts = ["=== REFERENCED FILES ==="]
                    for category, files in loaded_by_cat.items():
                        cat_parts = []
                        for f in files:
                            blob_parts.append(f"--- {f['filename']} ({category}) ---")
                            blob_parts.append(f["content"])
                            cat_parts.append(f["content"])
                        inputs[f"editor_refs_{category}"] = "\n\n".join(cat_parts)
                        cat_content = inputs[f"editor_refs_{category}"]
                        manuscript_heading = inputs.get("editor_current_heading", "")
                        if cat_content:
                            from orchestrator.utils.section_scoping import scope_reference_by_heading

                            scoped = scope_reference_by_heading(
                                cat_content, manuscript_heading, max_level=heading_level
                            )
                            inputs[f"editor_refs_{category}_toc"] = scoped["toc"]
                            inputs[f"editor_refs_{category}_current"] = scoped["current"]
                            inputs[f"editor_refs_{category}_adjacent"] = scoped["adjacent"]
                            inputs[f"editor_refs_{category}_previous"] = scoped["previous"]
                            inputs[f"editor_refs_{category}_next"] = scoped["next"]
                        else:
                            inputs[f"editor_refs_{category}_toc"] = ""
                            inputs[f"editor_refs_{category}_current"] = ""
                            inputs[f"editor_refs_{category}_adjacent"] = ""
                            inputs[f"editor_refs_{category}_previous"] = ""
                            inputs[f"editor_refs_{category}_next"] = ""
                    blob_parts.append("=== END REFERENCED FILES ===")
                    inputs["editor_refs"] = "\n".join(blob_parts) if len(blob_parts) > 2 else ""
                    inputs["editor_ref_count"] = str(len(loaded_by_cat))
                else:
                    inputs["editor_refs"] = ""
                    inputs["editor_ref_count"] = "0"
            except Exception as e:
                logger.warning("Failed to load editor_refs: %s", e)
                inputs["editor_refs"] = ""
                inputs["editor_ref_count"] = "0"
        else:
            inputs["editor_refs"] = ""
            inputs["editor_ref_count"] = "0"

        active_data_workspace = state.get("shared_memory", {}).get("active_data_workspace", {})
        if active_data_workspace and active_data_workspace.get("table_id"):
            schema = active_data_workspace.get("schema", [])
            visible_rows = active_data_workspace.get("visible_rows", [])
            row_count = active_data_workspace.get("row_count", 0)
            visible_row_count = active_data_workspace.get("visible_row_count", 0)
            col_names = [c.get("name", "") for c in schema if c.get("name")]
            schema_header = "| Column | Type | Description |"
            schema_sep = "|--------|------|-------------|"
            schema_lines = [schema_header, schema_sep]
            for c in schema:
                name = (c.get("name") or "").replace("|", "\\|")
                typ = (c.get("type") or "TEXT").replace("|", "\\|")
                desc = (c.get("description") or "").replace("|", "\\|")
                schema_lines.append(f"| {name} | {typ} | {desc} |")
            inputs["data_workspace_schema"] = "\n".join(schema_lines)
            rows_md = []
            if col_names and visible_rows:
                header = "| " + " | ".join(col_names) + " |"
                sep = "|" + "|".join(["---" for _ in col_names]) + "|"
                rows_md.append(header)
                rows_md.append(sep)
                for r in visible_rows:
                    row_data = r.get("row_data", r) if isinstance(r, dict) else {}
                    if not isinstance(row_data, dict):
                        row_data = {}
                    cells = []
                    for cn in col_names:
                        val = row_data.get(cn)
                        if val is None:
                            cells.append("")
                        else:
                            cells.append(str(val).replace("|", "\\|").replace("\n", " "))
                    rows_md.append("| " + " | ".join(cells) + " |")
            inputs["data_workspace_visible_rows"] = "\n".join(rows_md) if rows_md else ""
            full_parts = [
                f"Table: {active_data_workspace.get('table_name', '')} ({row_count} rows)",
                f"Database: {active_data_workspace.get('database_name', '')} | Workspace: {active_data_workspace.get('workspace_name', '')}",
                "",
                "Schema:",
                inputs["data_workspace_schema"],
                ""
            ]
            if inputs["data_workspace_visible_rows"]:
                full_parts.append(f"Sample data ({visible_row_count} visible rows):")
                full_parts.append("")
                full_parts.append(inputs["data_workspace_visible_rows"])
            inputs["data_workspace"] = "\n".join(full_parts)
            inputs["data_workspace_name"] = active_data_workspace.get("workspace_name", "")
            inputs["data_workspace_database"] = active_data_workspace.get("database_name", "")
            inputs["data_workspace_table"] = active_data_workspace.get("table_name", "")
            inputs["data_workspace_table_id"] = active_data_workspace.get("table_id", "")
            inputs["data_workspace_workspace_id"] = active_data_workspace.get("workspace_id", "")
            inputs["data_workspace_row_count"] = str(row_count)
            inputs["data_workspace_visible_row_count"] = str(visible_row_count)
        else:
            inputs["data_workspace"] = ""
            inputs["data_workspace_name"] = ""
            inputs["data_workspace_database"] = ""
            inputs["data_workspace_table"] = ""
            inputs["data_workspace_table_id"] = ""
            inputs["data_workspace_workspace_id"] = ""
            inputs["data_workspace_schema"] = ""
            inputs["data_workspace_row_count"] = ""
            inputs["data_workspace_visible_rows"] = ""
            inputs["data_workspace_visible_row_count"] = ""

        if "{profile}" in all_prompts:
            try:
                client = await get_backend_tool_client()
                profile_data = await client.get_my_profile(user_id=user_id)
                inputs["profile"] = _format_user_context(profile_data)
            except Exception as e:
                logger.warning("Failed to load profile for {profile}: %s", e)
                inputs["profile"] = ""
        else:
            inputs["profile"] = ""

        try:
            graph = build_playbook_graph(steps, checkpointer=checkpointer)
            profile = state.get("agent_profile") or {}
            defn = playbook.get("definition")
            if isinstance(defn, str):
                try:
                    defn = json.loads(defn) if defn else {}
                except Exception:
                    defn = {}
            if not isinstance(defn, dict):
                defn = {}
            initial_playbook_state = {
                "playbook_state": {},
                "inputs": inputs,
                "user_id": user_id,
                "metadata": pipeline_metadata,
                "execution_trace": [],
            }
            ws_state: Dict[str, Any] = {}
            if shared_memory.get("workspace_schemas") is not None:
                ws_state["workspace_schemas"] = shared_memory.get("workspace_schemas")
            if shared_memory.get("workspace_ids") is not None:
                ws_state["workspace_ids"] = shared_memory.get("workspace_ids")
            if shared_memory.get("workspace_context_instructions"):
                ws_state["workspace_context_instructions"] = shared_memory.get("workspace_context_instructions", "")
            if shared_memory.get("workspace_access_modes") is not None:
                ws_state["workspace_access_modes"] = shared_memory.get("workspace_access_modes")
            if ws_state:
                initial_playbook_state["playbook_state"] = ws_state
            result = await _await_cancelable_ainvoke(
                graph.ainvoke(initial_playbook_state, config=playbook_config),
                self._stream_cancellation_token,
                settings.PLAYBOOK_GRAPH_INVOKE_TIMEOUT_SEC,
            )
            playbook_state = result.get("playbook_state") or {}
            pending_approval = result.get("pending_approval")
            pending_auth = result.get("pending_auth")
            execution_trace = result.get("execution_trace") or []
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            cap = settings.PLAYBOOK_GRAPH_INVOKE_TIMEOUT_SEC
            logger.warning("Pipeline execution timed out after %s s", cap)
            return {
                "pipeline_results": {},
                "typed_outputs": {},
                "pending_approval": None,
                "pending_auth": None,
                "playbook_config": None,
                "response": {"formatted": f"Pipeline timed out after {cap}s."},
                "task_status": "error",
                "error": f"Playbook graph timed out after {cap}s",
                "metadata": metadata,
                "user_id": user_id,
                "query": query,
                "messages": state.get("messages", []),
                "shared_memory": state.get("shared_memory", {}),
            }
        except Exception as e:
            logger.exception("Pipeline execution failed: %s", e)
            return {
                "pipeline_results": {},
                "typed_outputs": {},
                "pending_approval": None,
                "pending_auth": None,
                "playbook_config": None,
                "response": {"formatted": f"Pipeline error: {str(e)}"},
                "task_status": "error",
                "error": str(e),
                "metadata": metadata,
                "user_id": user_id,
                "query": query,
                "messages": state.get("messages", []),
                "shared_memory": state.get("shared_memory", {}),
            }

        typed_outputs: Dict[str, Any] = {}
        for key, value in playbook_state.items():
            if key.startswith("_"):
                continue
            if isinstance(value, dict) and "formatted" in value:
                typed_outputs[key] = {k: v for k, v in value.items() if k != "formatted"}

        for key, typed in typed_outputs.items():
            doc_id = typed.get("document_id")
            if doc_id:
                sm = state.get("shared_memory") or {}
                self._pin_document(
                    sm,
                    doc_id,
                    title=typed.get("title", ""),
                    filename=typed.get("filename", ""),
                )
                state["shared_memory"] = sm
                break

        sm = state.get("shared_memory") or {}
        self._persist_tool_results_to_shared_memory(sm, typed_outputs)
        state["shared_memory"] = sm

        return {
            "pipeline_results": playbook_state,
            "typed_outputs": typed_outputs,
            "pending_approval": pending_approval,
            "pending_auth": pending_auth,
            "playbook_config": playbook_config if (pending_approval or pending_auth) else None,
            "execution_trace": execution_trace,
            "metadata": metadata,
            "user_id": user_id,
            "query": query,
            "messages": state.get("messages", []),
            "shared_memory": state.get("shared_memory", {}),
            "agent_profile": state.get("agent_profile", {}),
            "playbook": playbook,
            "playbook_id": state.get("playbook_id", ""),
        }

    def _collect_images_from_pipeline(self, pipeline_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect structured images from any step that returned images or structured_images."""
        collected: List[Dict[str, Any]] = []
        for key, value in pipeline_results.items():
            if key.startswith("_"):
                continue
            if not isinstance(value, dict):
                continue
            step_images = value.get("images") or value.get("structured_images")
            if isinstance(step_images, list):
                for item in step_images:
                    if isinstance(item, dict):
                        collected.append(item)
        return collected

    def _collect_editor_proposals_from_pipeline(self, pipeline_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect editor_proposals from all steps (e.g. llm_agent steps that called patch_file)."""
        collected: List[Dict[str, Any]] = []
        for key, value in pipeline_results.items():
            if key.startswith("_"):
                continue
            if not isinstance(value, dict):
                continue
            proposals = value.get("editor_proposals")
            if isinstance(proposals, list):
                for p in proposals:
                    if isinstance(p, dict) and p.get("proposal_id"):
                        collected.append(p)
        return collected

    def _collect_artifacts_from_pipeline(self, pipeline_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect validated artifact payloads from steps (e.g. create_artifact, create_chart)."""
        collected: List[Dict[str, Any]] = []
        for key, value in pipeline_results.items():
            if key.startswith("_"):
                continue
            if not isinstance(value, dict):
                continue
            arts_list = value.get("artifacts")
            if isinstance(arts_list, list) and arts_list:
                for a in arts_list:
                    if (
                        isinstance(a, dict)
                        and a.get("artifact_type")
                        and isinstance(a.get("code"), str)
                    ):
                        collected.append(a)
                continue
            art = value.get("artifact")
            if not isinstance(art, dict):
                continue
            atype = art.get("artifact_type")
            code = art.get("code")
            if not atype or not isinstance(code, str):
                continue
            collected.append(art)
        return collected

    async def _format_response_node(self, state: CustomAgentState) -> Dict[str, Any]:
        """Build final response payload (formatted text, typed outputs, structured images)."""
        parts: List[str] = []
        pipeline_results = state.get("pipeline_results", {})
        for key, value in pipeline_results.items():
            if key.startswith("_"):
                continue
            if isinstance(value, dict) and value.get("formatted"):
                parts.append(value["formatted"])
            elif isinstance(value, str):
                parts.append(value)
        formatted = "\n\n".join(parts) if parts else "Pipeline completed (no formatted output)."
        formatted = strip_tool_actions_prefix(formatted)

        images = self._collect_images_from_pipeline(pipeline_results)
        editor_proposals = self._collect_editor_proposals_from_pipeline(pipeline_results)
        artifacts = self._collect_artifacts_from_pipeline(pipeline_results)
        categories: set = set()
        for key, value in pipeline_results.items():
            if key.startswith("_") or not isinstance(value, dict):
                continue
            if value.get("_action_category"):
                categories.add(value["_action_category"])
            for cat in value.get("_tools_used_categories") or []:
                categories.add(cat)

        metadata = state.get("metadata", {})
        response_payload: Dict[str, Any] = {
            "formatted": formatted,
            "typed_outputs": state.get("typed_outputs", {}),
            "agent_profile_name": metadata.get("agent_profile_name", ""),
        }
        if images:
            response_payload["images"] = images
        if editor_proposals:
            response_payload["editor_proposals"] = editor_proposals
        if artifacts:
            response_payload["artifact"] = artifacts[0]
            if len(artifacts) > 1:
                response_payload["artifacts"] = artifacts
        if categories:
            response_payload["tools_used_categories"] = sorted(categories)

        state_copy = dict(state)
        state_copy["shared_memory"] = dict(state.get("shared_memory", {}))
        self._clear_request_scoped_data(state_copy)
        clean_shared_memory = state_copy.get("shared_memory", {})

        return {
            "response": response_payload,
            "task_status": "complete",
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "query": state.get("query", ""),
            "messages": state.get("messages", []),
            "shared_memory": clean_shared_memory,
            "execution_trace": state.get("execution_trace", []),
        }

    def _is_approval_message(self, content: str) -> bool:
        """True if message content indicates approval (yes, ok, approve, etc.)."""
        if not content or not isinstance(content, str):
            return False
        normalized = content.strip().lower()
        return normalized in ("yes", "y", "ok", "okay", "approve", "approved", "proceed", "continue", "send")

    def _is_rejection_message(self, content: str) -> bool:
        """True if message content indicates rejection."""
        if not content or not isinstance(content, str):
            return False
        normalized = content.strip().lower()
        return normalized in ("no", "n", "reject", "rejected", "cancel", "stop", "don't", "dont")

    async def _approval_gate_node(self, state: CustomAgentState) -> Dict[str, Any]:
        """Handle approval-required or auth-required state: detect approve/reject from last message, optionally resume pipeline."""
        pending = state.get("pending_approval") or {}
        pending_auth = state.get("pending_auth") or {}
        messages = state.get("messages", [])
        last_content = ""
        if messages:
            last_msg = messages[-1]
            last_content = getattr(last_msg, "content", "") if hasattr(last_msg, "content") else str(last_msg)

        if self._is_rejection_message(last_content):
            on_reject = pending.get("on_reject", "stop")
            step_name = pending.get("step_name") or pending_auth.get("step_name", "")
            formatted = f"Approval rejected at step '{step_name}'. {on_reject}"
            return {
                "response": {"formatted": formatted, "typed_outputs": state.get("typed_outputs", {})},
                "task_status": "rejected",
                "pending_approval": None,
                "pending_auth": None,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "query": state.get("query", ""),
                "messages": messages,
                "shared_memory": state.get("shared_memory", {}),
                "pipeline_results": state.get("pipeline_results", {}),
                "typed_outputs": state.get("typed_outputs", {}),
                "output_destinations": state.get("output_destinations", []),
                "execution_trace": state.get("execution_trace", []),
            }

        if self._is_approval_message(last_content) and (pending.get("step_name") or pending_auth.get("step_name")):
            playbook_config = state.get("playbook_config")
            if not playbook_config:
                return {
                    "response": {"formatted": "No playbook config to resume.", "typed_outputs": state.get("typed_outputs", {})},
                    "task_status": "error",
                    "pending_approval": None,
                    "pending_auth": None,
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "query": state.get("query", ""),
                    "messages": messages,
                    "shared_memory": state.get("shared_memory", {}),
                    "pipeline_results": state.get("pipeline_results", {}),
                    "typed_outputs": state.get("typed_outputs", {}),
                    "output_destinations": state.get("output_destinations", []),
                    "execution_trace": state.get("execution_trace", []),
                }
            playbook = state.get("playbook", {})
            steps = _definition_steps(playbook)
            user_id = state.get("user_id", "system")
            metadata = state.get("metadata", {})
            if pending_auth.get("interaction_type") == "shell_command_approval":
                try:
                    from orchestrator.backend_tool_client import get_backend_tool_client

                    _idata = pending_auth.get("interaction_data") or {}
                    _aid = (_idata.get("approval_id") or pending_auth.get("approval_id") or "").strip()
                    if _aid:
                        _cli = await get_backend_tool_client()
                        await _cli.grant_and_consume_shell_approval(
                            user_id,
                            approval_id=_aid,
                            command="",
                            consume=False,
                        )
                except Exception as _shell_grant_err:
                    logger.warning("shell_command_approval grant failed: %s", _shell_grant_err)
            try:
                from orchestrator.checkpointer import get_async_postgres_saver
                checkpointer = await get_async_postgres_saver()
                graph = build_playbook_graph(steps, checkpointer=checkpointer)
                result = await _await_cancelable_ainvoke(
                    graph.ainvoke({}, config=playbook_config),
                    self._stream_cancellation_token,
                    settings.PLAYBOOK_GRAPH_INVOKE_TIMEOUT_SEC,
                )
                playbook_state = result.get("playbook_state") or {}
                new_pending = result.get("pending_approval")
                new_pending_auth = result.get("pending_auth")
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                cap = settings.PLAYBOOK_GRAPH_INVOKE_TIMEOUT_SEC
                logger.warning("Pipeline resume after approval timed out after %s s", cap)
                return {
                    "response": {"formatted": f"Resume timed out after {cap}s.", "typed_outputs": state.get("typed_outputs", {})},
                    "task_status": "error",
                    "pending_approval": None,
                    "pending_auth": None,
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "query": state.get("query", ""),
                    "messages": messages,
                    "shared_memory": state.get("shared_memory", {}),
                    "pipeline_results": state.get("pipeline_results", {}),
                    "typed_outputs": state.get("typed_outputs", {}),
                    "output_destinations": state.get("output_destinations", []),
                    "execution_trace": state.get("execution_trace", []),
                }
            except Exception as e:
                logger.exception("Pipeline resume after approval failed: %s", e)
                return {
                    "response": {"formatted": f"Resume failed: {e}", "typed_outputs": state.get("typed_outputs", {})},
                    "task_status": "error",
                    "pending_approval": None,
                    "pending_auth": None,
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "query": state.get("query", ""),
                    "messages": messages,
                    "shared_memory": state.get("shared_memory", {}),
                    "pipeline_results": state.get("pipeline_results", {}),
                    "typed_outputs": state.get("typed_outputs", {}),
                    "output_destinations": state.get("output_destinations", []),
                    "execution_trace": state.get("execution_trace", []),
                }
            typed_outputs: Dict[str, Any] = {}
            for key, value in playbook_state.items():
                if key.startswith("_"):
                    continue
                if isinstance(value, dict) and "formatted" in value:
                    typed_outputs[key] = {k: v for k, v in value.items() if k != "formatted"}
            return {
                "pipeline_results": playbook_state,
                "typed_outputs": typed_outputs,
                "pending_approval": new_pending,
                "pending_auth": new_pending_auth,
                "playbook_config": playbook_config if (new_pending or new_pending_auth) else None,
                "metadata": metadata,
                "user_id": user_id,
                "query": state.get("query", ""),
                "messages": messages,
                "shared_memory": state.get("shared_memory", {}),
                "agent_profile": state.get("agent_profile", {}),
                "playbook": playbook,
                "playbook_id": state.get("playbook_id", ""),
                "output_destinations": state.get("output_destinations", []),
                "execution_trace": result.get("execution_trace", state.get("execution_trace", [])),
            }

        step_name = pending.get("step_name") or pending_auth.get("step_name", "")
        prompt = pending.get("prompt") or pending_auth.get("prompt", "Approve to continue?")
        metadata = state.get("metadata", {})
        playbook_config = state.get("playbook_config")

        if (
            metadata.get("trigger_type") == "scheduled"
            and metadata.get("execution_id")
            and playbook_config
        ):
            try:
                client = await get_backend_tool_client()
                configurable = playbook_config.get("configurable") or {}
                thread_id = str(configurable.get("thread_id", ""))
                approval_id = await client.park_approval(
                    user_id=state.get("user_id", "system"),
                    agent_profile_id=metadata.get("agent_profile_id", ""),
                    execution_id=metadata.get("execution_id"),
                    step_name=step_name,
                    prompt=prompt,
                    preview_data=pending.get("preview_data") or pending_auth.get("preview_data"),
                    thread_id=thread_id,
                    checkpoint_ns="",
                    playbook_config=playbook_config,
                )
                if approval_id:
                    formatted = (
                        f"Approval required at step '{step_name}'. "
                        f"{prompt} (Parked for background run; approve in dashboard.)"
                    )
                    return {
                        "response": {
                            "formatted": formatted,
                            "pending_approval": pending,
                            "pending_auth": pending_auth,
                            "typed_outputs": state.get("typed_outputs", {}),
                            "approval_queue_id": approval_id,
                        },
                        "task_status": "approval_parked",
                        "approval_queue_id": approval_id,
                        "pending_approval": pending,
                        "pending_auth": pending_auth,
                        "playbook_config": playbook_config,
                        "metadata": metadata,
                        "user_id": state.get("user_id", "system"),
                        "query": state.get("query", ""),
                        "messages": state.get("messages", []),
                        "shared_memory": state.get("shared_memory", {}),
                        "pipeline_results": state.get("pipeline_results", {}),
                        "typed_outputs": state.get("typed_outputs", {}),
                        "output_destinations": state.get("output_destinations", []),
                        "execution_trace": state.get("execution_trace", []),
                    }
            except Exception as e:
                logger.warning("ParkApproval failed, falling back to approval_required: %s", e)

        formatted = (
            f"Approval required at step '{step_name}'. "
            f"{prompt} Reply with yes/no to continue."
        )
        return {
            "response": {
                "formatted": formatted,
                "pending_approval": pending,
                "pending_auth": pending_auth,
                "typed_outputs": state.get("typed_outputs", {}),
            },
            "task_status": "approval_required",
            "pending_approval": pending,
            "pending_auth": pending_auth,
            "playbook_config": state.get("playbook_config"),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "query": state.get("query", ""),
            "messages": state.get("messages", []),
            "shared_memory": state.get("shared_memory", {}),
            "pipeline_results": state.get("pipeline_results", {}),
            "typed_outputs": state.get("typed_outputs", {}),
            "output_destinations": state.get("output_destinations", []),
            "execution_trace": state.get("execution_trace", []),
        }

    async def process(
        self,
        query: str,
        metadata: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Any]] = None,
        cancellation_token: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run the custom agent workflow and return the final response dict.
        Caller (unified_dispatch) uses response.formatted and response.typed_outputs for ChatChunks.
        """
        import time
        from datetime import datetime, timezone

        self._stream_cancellation_token = cancellation_token
        metadata = metadata or {}
        user_id = metadata.get("user_id", "system")
        shared_memory = metadata.get("shared_memory", {})
        started_at = time.perf_counter()
        started_at_iso = datetime.now(timezone.utc).isoformat()

        resume_approval_id = metadata.get("resume_approval_id")
        resume_playbook_config_json = metadata.get("resume_playbook_config_json")
        if resume_approval_id and resume_playbook_config_json:
            try:
                playbook_config = json.loads(resume_playbook_config_json) if isinstance(resume_playbook_config_json, str) else resume_playbook_config_json
                agent_profile_id = metadata.get("agent_profile_id")
                if not agent_profile_id:
                    return self._create_error_response("Resume missing agent_profile_id")
                client = await get_backend_tool_client()
                profile = await client.get_agent_profile(agent_profile_id=agent_profile_id, user_id=user_id)
                if not profile:
                    return self._create_error_response("Profile not found for resume")
                playbook_id = profile.get("default_playbook_id") or profile.get("playbook_id")
                if not playbook_id:
                    return self._create_error_response("Profile has no playbook for resume")
                playbook = await client.get_playbook(playbook_id=playbook_id, user_id=user_id)
                if not playbook:
                    return self._create_error_response("Playbook not found for resume")
                steps = _definition_steps(playbook)
                if not steps:
                    return self._create_error_response("Playbook has no steps for resume")
                from orchestrator.checkpointer import get_async_postgres_saver
                checkpointer = await get_async_postgres_saver()
                graph = build_playbook_graph(steps, checkpointer=checkpointer)
                result = await _await_cancelable_ainvoke(
                    graph.ainvoke({}, config=playbook_config),
                    self._stream_cancellation_token,
                    settings.PLAYBOOK_GRAPH_INVOKE_TIMEOUT_SEC,
                )
                playbook_state = result.get("playbook_state") or {}
                pending_approval = result.get("pending_approval")
                pending_auth = result.get("pending_auth")
                if pending_approval or pending_auth:
                    step_name = (pending_approval or pending_auth or {}).get("step_name", "")
                    formatted = (
                        f"Approval still required at step '{step_name}' after resume. "
                        "Approve again in the dashboard."
                    )
                    return {
                        "response": formatted,
                        "formatted": formatted,
                        "typed_outputs": {},
                        "pending_approval": pending_approval,
                        "pending_auth": pending_auth,
                        "task_status": "approval_required",
                        "route_results": {},
                    }
                typed_outputs = {}
                for key, value in playbook_state.items():
                    if key.startswith("_"):
                        continue
                    if isinstance(value, dict) and "formatted" in value:
                        typed_outputs[key] = {k: v for k, v in value.items() if k != "formatted"}
                response_formatted = ""
                for key, value in playbook_state.items():
                    if key.startswith("_"):
                        continue
                    if isinstance(value, dict) and value.get("formatted"):
                        response_formatted = value.get("formatted", "")
                        break
                return {
                    "response": response_formatted or "Resume completed.",
                    "formatted": response_formatted or "Resume completed.",
                    "typed_outputs": typed_outputs,
                    "pending_approval": None,
                    "task_status": "complete",
                    "route_results": {},
                }
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                cap = settings.PLAYBOOK_GRAPH_INVOKE_TIMEOUT_SEC
                logger.warning("Resume after approval timed out after %s s", cap)
                return self._create_error_response(f"Resume timed out after {cap}s")
            except Exception as e:
                logger.exception("Resume after approval failed: %s", e)
                return self._create_error_response(f"Resume failed: {e}")

        checkpoint_state = None
        config: Optional[Dict[str, Any]] = None
        try:
            workflow = await self._get_workflow()
            config = self._get_checkpoint_config(metadata)
            base_messages = list(messages) if messages else []
            new_messages = self._prepare_messages_with_query(base_messages, query)
            context_window = int(metadata.get("context_window_size", 20))
            conversation_messages = await self._load_and_merge_checkpoint_messages(
                workflow, config, new_messages, look_back_limit=context_window
            )
            checkpoint_state = await workflow.aget_state(config)
            existing_shared_memory = {}
            if checkpoint_state and checkpoint_state.values:
                existing_shared_memory = checkpoint_state.values.get("shared_memory", {})
            shared_memory_merged = dict(existing_shared_memory)
            shared_memory_merged.update(shared_memory)

            editor_pref = (
                shared_memory.get("editor_preference")
                or metadata.get("editor_preference")
            )
            if editor_pref == "ignore" and "active_editor" in shared_memory_merged:
                del shared_memory_merged["active_editor"]

            initial_state: CustomAgentState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory_merged,
            }

            result_state = await _await_cancelable_ainvoke(
                workflow.ainvoke(initial_state, config=config),
                self._stream_cancellation_token,
                settings.PLAYBOOK_GRAPH_INVOKE_TIMEOUT_SEC,
            )
        except asyncio.CancelledError:
            if config:
                try:
                    outer_thread = config.get("configurable", {}).get("thread_id")
                    if outer_thread:
                        await clear_checkpoint_thread(outer_thread)
                    playbook_id = metadata.get("playbook_id")
                    if playbook_id:
                        await clear_checkpoint_thread(
                            f"playbook_{user_id}_{playbook_id}"
                        )
                except Exception as cleanup_err:
                    logger.warning(
                        "Checkpoint cleanup after cancel failed: %s", cleanup_err
                    )
            raise
        except asyncio.TimeoutError:
            cap = settings.PLAYBOOK_GRAPH_INVOKE_TIMEOUT_SEC
            logger.warning("Custom agent workflow timed out after %s s", cap)
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            profile_id = metadata.get("agent_profile_id")
            playbook_id = metadata.get("playbook_id", "")
            if profile_id:
                try:
                    client = await get_backend_tool_client()
                    profile = metadata.get("profile") or {}
                    log_meta = {"model_used": profile.get("model_preference") or metadata.get("model_preference")}
                    await client.log_agent_execution(
                        user_id=user_id,
                        profile_id=profile_id,
                        playbook_id=playbook_id or "",
                        query=query,
                        status="failed",
                        duration_ms=duration_ms,
                        steps_completed=0,
                        steps_total=0,
                        error_details=f"Playbook graph timed out after {cap}s",
                        started_at=started_at_iso,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        metadata=log_meta,
                    )
                except Exception as log_err:
                    logger.warning("Failed to log agent execution (timeout path): %s", log_err)
            return self._create_error_response(f"Playbook timed out after {cap}s")
        except Exception as e:
            logger.exception("Custom agent workflow failed: %s", e)
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            profile_id = metadata.get("agent_profile_id")
            playbook_id = metadata.get("playbook_id", "")
            if profile_id:
                try:
                    client = await get_backend_tool_client()
                    profile = metadata.get("profile") or {}
                    log_meta = {"model_used": profile.get("model_preference") or metadata.get("model_preference")}
                    await client.log_agent_execution(
                        user_id=user_id,
                        profile_id=profile_id,
                        playbook_id=playbook_id or "",
                        query=query,
                        status="failed",
                        duration_ms=duration_ms,
                        steps_completed=0,
                        steps_total=0,
                        error_details=str(e)[:2000],
                        started_at=started_at_iso,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        metadata=log_meta,
                    )
                except Exception as log_err:
                    logger.warning("Failed to log agent execution (error path): %s", log_err)
            return self._create_error_response(str(e))

        pending_approval = result_state.get("pending_approval")
        pending_auth = result_state.get("pending_auth")
        if pending_approval or pending_auth:
            if result_state.get("task_status") == "approval_parked":
                approval_queue_id = result_state.get("approval_queue_id", "")
                formatted = (
                    f"Approval parked at step '{(pending_approval or pending_auth or {}).get('step_name', '')}'. "
                    "Approve or reject in the Operations dashboard to continue."
                )
                return {
                    "response": formatted,
                    "formatted": formatted,
                    "typed_outputs": result_state.get("typed_outputs", {}),
                    "pending_approval": pending_approval,
                    "pending_auth": pending_auth,
                    "task_status": "approval_parked",
                    "approval_queue_id": approval_queue_id,
                    "route_results": {},
                }
            step_name = (pending_approval or pending_auth or {}).get("step_name", "")
            formatted = (
                f"Approval required at step '{step_name}'. "
                "Human-in-the-loop: approve or reject to continue."
            )
            return {
                "response": formatted,
                "formatted": formatted,
                "typed_outputs": result_state.get("typed_outputs", {}),
                "pending_approval": pending_approval,
                "pending_auth": pending_auth,
                "task_status": "approval_required",
                "route_results": {},
            }

        task_status = result_state.get("task_status", "complete")
        response = result_state.get("response", {})

        duration_ms = int((time.perf_counter() - started_at) * 1000)
        profile_id = metadata.get("agent_profile_id")
        playbook_id = result_state.get("playbook_id")
        playbook = result_state.get("playbook") or {}
        steps_total = len(_definition_steps(playbook))
        execution_trace = result_state.get("execution_trace") or []
        all_silent = task_status == "complete" and not execution_trace
        if profile_id and playbook_id is not None and not all_silent:
            try:
                client = await get_backend_tool_client()
                steps_completed = steps_total if task_status == "complete" and not result_state.get("pending_approval") and not result_state.get("pending_auth") else 0
                status = "completed" if task_status == "complete" else ("running" if (result_state.get("pending_approval") or result_state.get("pending_auth")) else "failed")
                steps_json = json.dumps(execution_trace) if execution_trace else ""
                profile = result_state.get("profile") or metadata.get("profile") or {}
                log_meta = {"model_used": profile.get("model_preference") or metadata.get("model_preference")}
                await client.log_agent_execution(
                    user_id=user_id,
                    profile_id=profile_id,
                    playbook_id=playbook_id or "",
                    query=query,
                    status=status,
                    duration_ms=duration_ms,
                    steps_completed=steps_completed,
                    steps_total=steps_total or 0,
                    error_details=result_state.get("error", "")[:2000] if task_status == "error" else None,
                    started_at=started_at_iso,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    steps_json=steps_json,
                    metadata=log_meta,
                )
            except Exception as log_err:
                logger.warning("Failed to log agent execution: %s", log_err)

        if task_status == "error":
            error_msg = result_state.get("error", "Unknown error")
            return self._create_error_response(error_msg)

        if task_status == "rejected":
            response = result_state.get("response", {})
            formatted = response.get("formatted", "Approval rejected.") if isinstance(response, dict) else str(response)
            return {
                "response": formatted,
                "formatted": formatted,
                "typed_outputs": result_state.get("typed_outputs", {}),
                "pending_approval": None,
                "task_status": "rejected",
                "route_results": {},
            }

        resp_images = response.get("images") if isinstance(response, dict) else None
        result_metadata = result_state.get("metadata", {})
        execution_trace = result_state.get("execution_trace") or []
        tool_call_summary = _build_tool_call_summary(execution_trace)
        acquired_tool_log = _aggregate_acquired_tool_log(execution_trace)
        skill_execution_events = _aggregate_skill_execution_events(execution_trace)
        out: Dict[str, Any] = {
            "response": response.get("formatted", ""),
            "formatted": response.get("formatted", ""),
            "typed_outputs": response.get("typed_outputs", {}),
            "pending_approval": response.get("pending_approval"),
            "task_status": task_status,
            "route_results": response.get("route_results", {}),
            "images": resp_images if isinstance(resp_images, list) else None,
            "agent_profile_name": result_metadata.get("agent_profile_name", ""),
            "tool_call_summary": tool_call_summary,
            "acquired_tool_log": acquired_tool_log,
            "skill_execution_events": skill_execution_events,
        }
        if result_metadata.get("persona_enabled"):
            _p = result_metadata.get("persona")
            if isinstance(_p, dict):
                _pan = (_p.get("ai_name") or "").strip()
                if _pan:
                    out["persona_ai_name"] = _pan
        if isinstance(response, dict):
            if response.get("editor_proposals"):
                out["editor_proposals"] = response["editor_proposals"]
            art = response.get("artifact")
            if isinstance(art, dict) and art.get("artifact_type"):
                out["artifact"] = art
            arts = response.get("artifacts")
            if isinstance(arts, list) and arts:
                out["artifacts"] = arts
        return out

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        return {
            "response": error_message,
            "formatted": error_message,
            "typed_outputs": {},
            "pending_approval": None,
            "task_status": "error",
            "route_results": {},
        }
