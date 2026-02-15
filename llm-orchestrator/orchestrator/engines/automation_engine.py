"""
Automation Engine - Generic tool-calling engine for simple skills.

Runs skills that are defined by: system prompt + optional tools + LLM response.
Replaces dedicated agent classes for weather, dictionary, help, email, navigation,
rss, entertainment, org_capture, image_generation, image_description, reference.
"""

import asyncio
import inspect
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TypedDict, get_type_hints

from pydantic import create_model

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END

from orchestrator.agents.base_agent import BaseAgent, TaskStatus
from orchestrator.models.agent_response_contract import AgentResponse, TaskStatus as ResponseTaskStatus
from orchestrator.skills import get_skill_registry

logger = logging.getLogger(__name__)


OUT_OF_SCOPE_SIGNAL = "[OUT_OF_SCOPE]"
SKILLS_CAN_REJECT = frozenset({"help"})


class AutomationEngineState(TypedDict, total=False):
    """State for Automation Engine LangGraph workflow.
    Store only tool names (strings) in state so the checkpointer can serialize it.
    """

    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    skill_name: str
    skill_config: Dict[str, Any]
    resolved_tool_names: List[str]
    llm_messages: List[Any]
    tool_results: Dict[str, Any]
    response: Dict[str, Any]
    task_status: str
    error: str


class AutomationEngine(BaseAgent):
    """
    Generic automation engine: load skill -> prepare context -> execute (tools or LLM) -> format response.
    """

    def __init__(self) -> None:
        super().__init__("automation_engine")
        logger.info("Automation engine initialized")

    def _build_workflow(self, checkpointer) -> StateGraph:
        workflow = StateGraph(AutomationEngineState)
        workflow.add_node("load_skill", self._load_skill_node)
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("execute_with_tools", self._execute_with_tools_node)
        workflow.add_node("check_rejection", self._check_rejection_node)
        workflow.add_node("format_response", self._format_response_node)
        workflow.set_entry_point("load_skill")
        workflow.add_edge("load_skill", "prepare_context")
        workflow.add_edge("prepare_context", "execute_with_tools")
        workflow.add_edge("execute_with_tools", "check_rejection")
        workflow.add_edge("check_rejection", "format_response")
        workflow.add_edge("format_response", END)
        return workflow.compile(checkpointer=checkpointer)

    def _preserve_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "query": state.get("query", ""),
            "user_id": state.get("user_id", "system"),
            "metadata": state.get("metadata", {}),
            "messages": state.get("messages", []),
            "shared_memory": state.get("shared_memory", {}),
            "skill_name": state.get("skill_name", ""),
            "skill_config": state.get("skill_config", {}),
            "resolved_tool_names": state.get("resolved_tool_names", []),
        }

    async def _load_skill_node(self, state: AutomationEngineState) -> Dict[str, Any]:
        """Load skill from registry and resolve tools."""
        try:
            skill_name = state.get("skill_name", "")
            if not skill_name:
                return {
                    **self._preserve_state(state),
                    "error": "skill_name required",
                    "task_status": "error",
                }
            registry = get_skill_registry()
            skill = registry.get(skill_name)
            if not skill:
                return {
                    **self._preserve_state(state),
                    "error": f"Skill not found: {skill_name}",
                    "task_status": "error",
                }
            skill_config = skill.model_dump()
            resolved_tool_names: List[str] = []
            if skill.tools:
                try:
                    import orchestrator.tools as tools_module
                    for tool_name in skill.tools:
                        fn = getattr(tools_module, tool_name, None)
                        if callable(fn):
                            resolved_tool_names.append(tool_name)
                        else:
                            logger.debug("Tool not found in orchestrator.tools: %s", tool_name)
                except Exception as e:
                    logger.warning("Resolving tools failed: %s", e)
            pack_names = state.get("metadata", {}).get("tool_packs") or []
            if pack_names:
                try:
                    import orchestrator.tools as tools_module
                    from orchestrator.tools.tool_pack_registry import resolve_pack_tools
                    pack_tool_names = resolve_pack_tools(pack_names)
                    for tool_name in pack_tool_names:
                        if tool_name not in resolved_tool_names:
                            fn = getattr(tools_module, tool_name, None)
                            if callable(fn):
                                resolved_tool_names.append(tool_name)
                except Exception as e:
                    logger.warning("Resolving tool pack tools failed: %s", e)
            return {
                **self._preserve_state(state),
                "skill_config": skill_config,
                "resolved_tool_names": resolved_tool_names,
                "task_status": "processing",
            }
        except Exception as e:
            logger.error("load_skill node failed: %s", e)
            return {
                **self._preserve_state(state),
                "error": str(e),
                "task_status": "error",
            }

    def _extract_image_base64(self, state: AutomationEngineState) -> Optional[str]:
        """Extract image base64 from shared_memory or metadata for vision skills."""
        metadata = state.get("metadata", {})
        shared_memory = state.get("shared_memory", {})
        image_base64 = metadata.get("image_base64") or metadata.get("image_data_base64")
        if not image_base64 and shared_memory.get("attached_images"):
            first_img = shared_memory["attached_images"][0]
            if isinstance(first_img, dict):
                image_base64 = first_img.get("data") or first_img.get("base64")
            elif isinstance(first_img, str):
                if first_img.startswith("data:"):
                    match = re.search(r"base64,(.+)", first_img)
                    if match:
                        image_base64 = match.group(1).strip()
                else:
                    image_base64 = first_img
        if not image_base64:
            return None
        if isinstance(image_base64, str) and image_base64.startswith("data:"):
            match = re.search(r"base64,(.+)", image_base64)
            if match:
                image_base64 = match.group(1).strip()
        return image_base64

    def _gather_prior_step_content(self, shared_memory: Dict[str, Any]) -> str:
        """Collect content from prior_step_*_response keys (compound plan context bridge)."""
        if not shared_memory:
            return ""
        parts = []
        for key in sorted(shared_memory.keys()):
            if key.startswith("prior_step_") and key.endswith("_response"):
                val = shared_memory.get(key)
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
        return "\n\n".join(parts) if parts else ""

    def _build_scoped_capture_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        messages_list: List[Any],
        state: Optional[Dict[str, Any]] = None,
        look_back_limit: int = 6,
    ) -> List[Any]:
        """Build messages for org_capture with history consolidated as context-only."""
        messages: List[Any] = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=self._get_datetime_context(state)),
        ]

        if messages_list and look_back_limit > 0:
            history = self._extract_conversation_history(messages_list, limit=look_back_limit)
            if history:
                history_lines: List[str] = []
                for msg in history:
                    role_label = "User" if msg.get("role") == "user" else "Assistant"
                    history_lines.append(f"{role_label}: {msg.get('content', '')}")
                messages.append(
                    SystemMessage(
                        content=(
                            "CONVERSATION HISTORY (for context only — do NOT capture items from this history unless the "
                            "current request explicitly asks you to):\n\n"
                            + "\n\n".join(history_lines)
                            + "\n\n---\nEND OF HISTORY. Only act on the current user message below."
                        )
                    )
                )

        messages.append(HumanMessage(content=user_prompt))
        return messages

    async def _prepare_context_node(self, state: AutomationEngineState) -> Dict[str, Any]:
        """Build LLM messages with skill system prompt and optional conversation history."""
        try:
            skill_config = state.get("skill_config", {})
            system_prompt = skill_config.get("system_prompt") or "You are a helpful assistant."
            query = state.get("query", "")
            messages_list = state.get("messages", [])
            skill_name = state.get("skill_name", "")
            shared_memory = state.get("shared_memory") or {}
            user_prompt = query
            if skill_name == "org_capture":
                prior_content = self._gather_prior_step_content(shared_memory)
                if prior_content:
                    user_prompt = (
                        "Content to capture (from a previous step):\n\n"
                        f"{prior_content}\n\n"
                        f"User request: {query}"
                    )
            if skill_name == "org_capture":
                llm_messages = self._build_scoped_capture_messages(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    messages_list=messages_list,
                    state=state,
                    look_back_limit=6,
                )
            else:
                look_back_limit = 0 if skill_config.get("stateless", False) else 6
                llm_messages = self._build_conversational_agent_messages(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    messages_list=messages_list,
                    look_back_limit=look_back_limit,
                    state=state,
                )
            if skill_config.get("requires_image_context"):
                image_base64 = self._extract_image_base64(state)
                if image_base64:
                    data_uri = f"data:image/jpeg;base64,{image_base64}" if not image_base64.startswith("data:") else image_base64
                    if not data_uri.startswith("data:"):
                        data_uri = f"data:image/jpeg;base64,{image_base64}"
                    content = [
                        {"type": "text", "text": query or "Describe this image in detail."},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ]
                    llm_messages[-1] = HumanMessage(content=content)
            return {
                **self._preserve_state(state),
                "llm_messages": llm_messages,
                "task_status": "processing",
            }
        except Exception as e:
            logger.error("prepare_context node failed: %s", e)
            return {
                **self._preserve_state(state),
                "error": str(e),
                "task_status": "error",
            }

    def _resolve_tool_names_to_functions(self, tool_names: List[str]) -> List[tuple]:
        """Resolve tool names to (name, func) tuples. Used only inside execute node (not stored in state)."""
        import orchestrator.tools as tools_module
        resolved: List[tuple] = []
        for name in tool_names:
            fn = getattr(tools_module, name, None)
            if callable(fn):
                resolved.append((name, fn))
        return resolved

    @staticmethod
    def _normalize_tool_name(name: str) -> str:
        """Normalize tool name for lookup (models may strip underscores)."""
        return name.replace("_", "").replace("-", "").lower()

    async def _execute_with_tools_node(self, state: AutomationEngineState) -> Dict[str, Any]:
        """Run LLM with optional tool binding; one round of tool calls max."""
        try:
            llm_messages = state.get("llm_messages", [])
            resolved_tool_names = state.get("resolved_tool_names", [])
            resolved_tools = self._resolve_tool_names_to_functions(resolved_tool_names) if resolved_tool_names else []
            if not llm_messages:
                return {
                    **self._preserve_state(state),
                    "error": "No LLM messages",
                    "task_status": "error",
                }
            llm = self._get_llm(temperature=0.7, state=state)
            response_text = ""
            if resolved_tools:
                try:
                    from langchain_core.tools import StructuredTool

                    def _wrap_async_tool(name: str, func: Any) -> StructuredTool:
                        sig = inspect.signature(func)
                        try:
                            hints = get_type_hints(func)
                        except Exception:
                            hints = {}
                        fields = {}
                        for pname, param in sig.parameters.items():
                            if pname in ("user_id", "_editor_content"):
                                continue
                            ann = hints.get(pname, str)
                            default = param.default if param.default != inspect.Parameter.empty else ...
                            fields[pname] = (ann, default)
                        schema_name = (name.replace("-", "_") + "_schema").replace(".", "_")
                        ArgsModel = create_model(schema_name, **fields)

                        async def _run(**kwargs: Any) -> str:
                            try:
                                if asyncio.iscoroutinefunction(func):
                                    out = await func(**kwargs)
                                else:
                                    out = func(**kwargs)
                                return str(out) if out is not None else ""
                            except Exception as e:
                                return f"Error: {e}"

                        return StructuredTool(
                            name=name,
                            description=(getattr(func, "__doc__") or name).strip(),
                            coroutine=_run,
                            args_schema=ArgsModel,
                        )

                    tools = [_wrap_async_tool(name, func) for name, func in resolved_tools]
                    if tools:
                        bound_llm = llm.bind_tools(tools)
                        response = await self._safe_llm_invoke(bound_llm, llm_messages, "automation_engine tools")
                        if getattr(response, "tool_calls", None):
                            tool_map = {
                                self._normalize_tool_name(n): (n, f) for n, f in resolved_tools
                            }
                            tool_messages = []
                            executed: Dict[Tuple[str, str], str] = {}
                            for tc in response.tool_calls:
                                tool_name = tc.get("name", "")
                                tool_args = tc.get("args", {}) or {}
                                tool_id = tc.get("id", "")
                                args = dict(tool_args)
                                if "kwargs" in args and isinstance(args["kwargs"], dict) and len(args) <= 2:
                                    args = dict(args["kwargs"])
                                dedupe_key = (
                                    self._normalize_tool_name(tool_name),
                                    json.dumps({k: v for k, v in sorted(args.items()) if k not in ("user_id", "_editor_content")}, sort_keys=True),
                                )
                                if dedupe_key in executed:
                                    result_str = executed[dedupe_key]
                                else:
                                    match = tool_map.get(self._normalize_tool_name(tool_name))
                                    original_name, tool_func = match if match else (None, None)
                                    if tool_func and callable(tool_func):
                                        try:
                                            sig = inspect.signature(tool_func)
                                            if "user_id" not in args and "user_id" in sig.parameters:
                                                args["user_id"] = state.get("user_id", "system")
                                            if "_editor_content" in sig.parameters:
                                                active_editor = state.get("shared_memory", {}).get("active_editor", {})
                                                args["_editor_content"] = active_editor.get("content", "")
                                            if asyncio.iscoroutinefunction(tool_func):
                                                result = await tool_func(**args)
                                            else:
                                                result = tool_func(**args)
                                            result_str = str(result) if result is not None else ""
                                        except Exception as e:
                                            result_str = f"Error: {e}"
                                    else:
                                        result_str = "Tool not available"
                                    executed[dedupe_key] = result_str
                                tool_messages.append(ToolMessage(content=result_str, tool_call_id=tool_id))
                            new_messages = llm_messages + [response] + tool_messages
                            final_response = await self._safe_llm_invoke(bound_llm, new_messages, "automation_engine tools round 2")
                            response_text = getattr(final_response, "content", "") or str(final_response)
                            # If model returned tool_calls but no text, use tool results so user sees something
                            if not (response_text or "").strip() and tool_messages:
                                first_result = (tool_messages[0].content or "").strip()
                                if first_result and "error" not in first_result.lower()[:200]:
                                    response_text = "Here's what I found in your org file:\n\n" + (first_result[:2000] + "..." if len(first_result) > 2000 else first_result)
                                else:
                                    response_text = "I couldn't generate a summary. Please try rephrasing or check that an org file is open."
                            elif not (response_text or "").strip():
                                response_text = "I couldn't complete that request. Please try again or rephrase."
                        else:
                            response_text = getattr(response, "content", "") or str(response)
                            if not (response_text or "").strip() and getattr(response, "tool_calls", None):
                                response_text = "I couldn't complete that request. Please try again or rephrase."
                    else:
                        response = await self._safe_llm_invoke(llm, llm_messages, "automation_engine")
                        response_text = getattr(response, "content", "") or str(response)
                except Exception as e:
                    logger.warning("Tool execution failed, falling back to LLM only: %s", e)
                    response = await self._safe_llm_invoke(llm, llm_messages, "automation_engine")
                    response_text = getattr(response, "content", "") or str(response)
            else:
                response = await self._safe_llm_invoke(llm, llm_messages, "automation_engine")
                response_text = getattr(response, "content", "") or str(response)
            if not (response_text or "").strip():
                response_text = "I couldn't generate a response. Please try again."
            return {
                **self._preserve_state(state),
                "response": {
                    "response": response_text,
                    "task_status": "complete",
                    "timestamp": datetime.now().isoformat(),
                },
                "task_status": "complete",
            }
        except Exception as e:
            logger.error("execute_with_tools node failed: %s", e)
            return self._handle_node_error(e, state, "execute_with_tools")

    async def _check_rejection_node(self, state: AutomationEngineState) -> Dict[str, Any]:
        """Check if response contains out-of-scope signal; set task_status=rejected for orchestration retry."""
        try:
            skill_name = state.get("skill_name", "")
            if skill_name not in SKILLS_CAN_REJECT:
                return {
                    **self._preserve_state(state),
                    "response": state.get("response", {}),
                    "task_status": state.get("task_status", "complete"),
                }
            resp = state.get("response", {})
            if not isinstance(resp, dict):
                return {
                    **self._preserve_state(state),
                    "response": state.get("response", {}),
                    "task_status": state.get("task_status", "complete"),
                }
            response_text = (resp.get("response") or "") if isinstance(resp.get("response"), str) else ""
            if isinstance(response_text, str) and response_text.strip().startswith(OUT_OF_SCOPE_SIGNAL):
                logger.info(
                    "Automation engine: skill %s signaled out-of-scope, task_status=rejected for: %s...",
                    skill_name,
                    (state.get("query") or "")[:60],
                )
                stripped = response_text.strip()
                if stripped.startswith(OUT_OF_SCOPE_SIGNAL):
                    stripped = stripped[len(OUT_OF_SCOPE_SIGNAL):].strip()
                updated_resp = dict(resp)
                updated_resp["response"] = stripped
                updated_resp["task_status"] = "rejected"
                return {
                    **self._preserve_state(state),
                    "response": updated_resp,
                    "task_status": "rejected",
                }
            return {
                **self._preserve_state(state),
                "response": state.get("response", {}),
                "task_status": state.get("task_status", "complete"),
            }
        except Exception as e:
            logger.error("check_rejection node failed: %s", e)
            return {
                **self._preserve_state(state),
                "response": state.get("response", {}),
                "task_status": state.get("task_status", "complete"),
            }

    async def _format_response_node(self, state: AutomationEngineState) -> Dict[str, Any]:
        """Build AgentResponse from state."""
        try:
            resp = state.get("response", {})
            response_text = resp.get("response", "") if isinstance(resp, dict) else ""
            task_status_str = state.get("task_status", "complete")
            try:
                task_status_enum = ResponseTaskStatus(task_status_str)
            except ValueError:
                task_status_enum = ResponseTaskStatus.COMPLETE
            skill_name = state.get("skill_name", "automation_engine")
            if isinstance(resp, dict) and resp.get("agent_type"):
                agent_type = resp["agent_type"]
            else:
                agent_type = f"{skill_name}_agent" if skill_name and not skill_name.endswith("_agent") else skill_name or "automation_engine"
            agent_response = AgentResponse(
                response=response_text,
                task_status=task_status_enum,
                agent_type=agent_type,
                timestamp=datetime.now().isoformat(),
            )
            return {
                **self._preserve_state(state),
                "response": agent_response.model_dump(),
                "task_status": task_status_str,
            }
        except Exception as e:
            logger.error("format_response node failed: %s", e)
            return {
                **self._preserve_state(state),
                "error": str(e),
                "task_status": "error",
            }

    async def process(
        self,
        query: str,
        metadata: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Any]] = None,
        skill_name: Optional[str] = None,
        cancellation_token: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Process automation request for the given skill."""
        metadata = metadata or {}
        messages = messages or []
        skill_name = skill_name or metadata.get("skill_name", "")
        if not skill_name:
            return {
                "task_status": "error",
                "response": {"response": "skill_name required", "task_status": "error"},
                "error": "skill_name required",
            }
        user_id = metadata.get("user_id", "system")
        workflow = await self._get_workflow()
        config = self._get_checkpoint_config(metadata)
        new_messages = self._prepare_messages_with_query(messages, query)
        conversation_messages = await self._load_and_merge_checkpoint_messages(workflow, config, new_messages)
        checkpoint_state = await workflow.aget_state(config)
        existing_shared_memory = {}
        if checkpoint_state and checkpoint_state.values:
            existing_shared_memory = checkpoint_state.values.get("shared_memory", {})
        shared_memory_merged = existing_shared_memory.copy()
        shared_memory_merged.update(metadata.get("shared_memory") or {})
        initial_state: AutomationEngineState = {
            "query": query,
            "user_id": user_id,
            "metadata": metadata,
            "messages": conversation_messages,
            "shared_memory": shared_memory_merged,
            "skill_name": skill_name,
            "skill_config": {},
            "resolved_tool_names": [],
            "llm_messages": [],
            "tool_results": {},
            "response": {},
            "task_status": "",
            "error": "",
        }
        result_state = await workflow.ainvoke(initial_state, config=config)
        response = result_state.get("response", {})
        task_status = result_state.get("task_status", "complete")
        if task_status == "error":
            error_msg = result_state.get("error", "Unknown error")
            return self._create_error_response(error_msg)
        return response
