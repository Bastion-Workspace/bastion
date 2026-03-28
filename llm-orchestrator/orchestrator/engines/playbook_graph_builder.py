"""
Playbook Graph Builder - Builds dynamic LangGraph workflows from Agent Factory playbook steps.

Each playbook step becomes a real LangGraph node. Supports tool, llm_task, llm_agent,
approval, and loop step types. Approval steps use interrupt_after so the runner
can show HITL UI; loop steps run a child graph in a loop.
Output is handled via tool steps (e.g. send_channel_message, save_to_document).
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from orchestrator.engines.pipeline_executor import (
    execute_step,
    _execute_llm_step,
    _execute_llm_agent_step,
    _execute_deep_agent_step,
    _evaluate_condition,
    _resolve_inputs,
)

logger = logging.getLogger(__name__)

# Normalize step type: playbook definitions may use "type" or "step_type"
def _step_type(step: Dict[str, Any]) -> str:
    return (step.get("step_type") or step.get("type") or "tool") or "tool"

# Max size for stored output snapshot (chars)
_TRACE_OUTPUT_MAX = 2048


def _truncate_for_trace(obj: Any) -> Any:
    """Recursively truncate strings in a dict/list for execution trace storage."""
    if isinstance(obj, str):
        return obj[:_TRACE_OUTPUT_MAX] + ("..." if len(obj) > _TRACE_OUTPUT_MAX else "")
    if isinstance(obj, dict):
        return {k: _truncate_for_trace(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_for_trace(v) for v in obj]
    return obj


class PlaybookGraphState(TypedDict, total=False):
    """State for the dynamic playbook LangGraph."""

    playbook_state: Dict[str, Any]
    inputs: Dict[str, Any]
    user_id: str
    metadata: Dict[str, Any]
    pending_approval: Optional[Dict[str, Any]]
    pending_auth: Optional[Dict[str, Any]]
    execution_trace: List[Dict[str, Any]]


def _wrap_node_with_condition(step: Dict[str, Any], node_impl):
    """Wrap a node so it is skipped when step.condition evaluates to false."""
    async def _node(state: PlaybookGraphState) -> Dict[str, Any]:
        cond = step.get("condition")
        if cond:
            ps = state.get("playbook_state") or {}
            inputs = state.get("inputs") or {}
            if not _evaluate_condition(cond, ps, inputs):
                key = step.get("output_key") or step.get("name")
                new_ps = {**(state.get("playbook_state") or {})}
                if key:
                    new_ps[key] = {"_skipped": True}
                return {**state, "playbook_state": new_ps}
        return await node_impl(state)
    return _node


def _wrap_with_tracing(
    step: Dict[str, Any],
    node_impl,
    step_index: int,
) -> Any:
    """Wrap a node to record execution_trace entry (started_at, completed_at, inputs, outputs, status)."""
    step_name = step.get("name") or step.get("output_key") or f"step_{step_index}"
    step_type = _step_type(step)
    action_name = step.get("action") if step_type == "tool" else None

    async def _node(state: PlaybookGraphState) -> Dict[str, Any]:
        ps = state.get("playbook_state") or {}
        inputs = state.get("inputs") or {}
        inputs_snapshot = _resolve_inputs(step.get("inputs") or {}, ps, inputs)
        started_at = datetime.now(timezone.utc).isoformat()
        trace = list(state.get("execution_trace") or [])

        try:
            result = await node_impl(state)
        except Exception as e:
            logger.exception("Playbook step failed: %s", step_name)
            duration_ms = None
            completed_at = datetime.now(timezone.utc).isoformat()
            trace.append({
                "step_index": step_index,
                "step_name": step_name,
                "step_type": step_type,
                "action_name": action_name,
                "status": "failed",
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_ms": duration_ms,
                "inputs_snapshot": _truncate_for_trace(inputs_snapshot),
                "outputs_snapshot": {},
                "error_details": str(e),
                "tool_call_trace": None,
            })
            raise

        completed_at = datetime.now(timezone.utc).isoformat()
        try:
            start_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            duration_ms = int((end_dt - start_dt).total_seconds() * 1000)
        except Exception:
            duration_ms = None

        new_ps = result.get("playbook_state") or ps
        key = step.get("output_key") or step_name
        step_result = new_ps.get(key) if key else None
        tool_call_trace = None
        if step_type == "llm_agent" and isinstance(step_result, dict):
            tool_call_trace = step_result.pop("_tool_call_trace", None)
            trace_inputs = step_result.pop("_trace_inputs", None)
            if isinstance(trace_inputs, dict) and trace_inputs:
                inputs_snapshot = {**inputs_snapshot, **trace_inputs}
        token_usage = None
        if isinstance(step_result, dict):
            token_usage = step_result.get("_token_usage")
            if step_result.get("_error"):
                status = "failed"
                error_details = step_result.get("_error", "")
            elif step_result.get("_skipped"):
                status = "skipped"
                error_details = None
            else:
                status = "completed"
                error_details = None
            outputs_snapshot = _truncate_for_trace({k: v for k, v in step_result.items() if k != "_token_usage"})
        else:
            status = "completed"
            error_details = None
            outputs_snapshot = _truncate_for_trace(step_result) if step_result is not None else {}

        trace_entry = {
            "step_index": step_index,
            "step_name": step_name,
            "step_type": step_type,
            "action_name": action_name,
            "status": status,
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_ms": duration_ms,
            "inputs_snapshot": _truncate_for_trace(inputs_snapshot),
            "outputs_snapshot": outputs_snapshot,
            "error_details": error_details,
            "tool_call_trace": tool_call_trace,
        }
        if token_usage and isinstance(token_usage, dict):
            trace_entry["input_tokens"] = token_usage.get("input_tokens", 0)
            trace_entry["output_tokens"] = token_usage.get("output_tokens", 0)
        trace.append(trace_entry)
        return {**result, "execution_trace": trace}
    return _node


def _make_tool_node(step: Dict[str, Any]):
    """Return a node that runs a single tool step."""
    async def _node(state: PlaybookGraphState) -> Dict[str, Any]:
        ps = state.get("playbook_state") or {}
        result = await execute_step(
            step, ps, state.get("inputs") or {},
            user_id=state.get("user_id", "system"),
            metadata=state.get("metadata"),
        )
        new_ps = {**ps}
        key = step.get("output_key") or step.get("name")
        if key:
            new_ps[key] = result
        return {
            **state,
            "playbook_state": new_ps,
        }
    return _node


def _make_deep_agent_node(step: Dict[str, Any]):
    """Return a node that runs a deep_agent step (multi-phase graph)."""
    async def _node(state: PlaybookGraphState) -> Dict[str, Any]:
        ps = state.get("playbook_state") or {}
        result = await _execute_deep_agent_step(
            step, ps, state.get("inputs") or {},
            user_id=state.get("user_id", "system"),
            metadata=state.get("metadata"),
        )
        new_ps = {**ps}
        key = step.get("output_key") or step.get("name")
        if key:
            new_ps[key] = result
        return {
            **state,
            "playbook_state": new_ps,
        }
    return _node


def _make_llm_task_node(step: Dict[str, Any]):
    """Return a node that runs an llm_task step."""
    async def _node(state: PlaybookGraphState) -> Dict[str, Any]:
        ps = state.get("playbook_state") or {}
        meta = dict(state.get("metadata") or {})
        result = await _execute_llm_step(
            step, ps, state.get("inputs") or {},
            user_id=state.get("user_id", "system"),
            metadata=meta,
        )
        new_ps = {**ps}
        key = step.get("output_key") or step.get("name")
        if key:
            new_ps[key] = result
        return {
            **state,
            "playbook_state": new_ps,
        }
    return _node


def _make_llm_agent_node(step: Dict[str, Any]):
    """Return a node that runs an llm_agent step (ReAct with bound tools). May set pending_auth on _interaction_required."""
    async def _node(state: PlaybookGraphState) -> Dict[str, Any]:
        ps = state.get("playbook_state") or {}
        meta = dict(state.get("metadata") or {})
        result = await _execute_llm_agent_step(
            step, ps, state.get("inputs") or {},
            user_id=state.get("user_id", "system"),
            metadata=meta,
        )
        new_ps = {**ps}
        key = step.get("output_key") or step.get("name")
        if key:
            new_ps[key] = result
        if isinstance(result, dict) and result.get("_interaction_required"):
            interaction_data = result.get("interaction_data") or {}
            pending_auth = {
                "step_name": step.get("name") or step.get("output_key") or "llm_agent",
                "interaction_type": result.get("interaction_type", "browser_login"),
                "interaction_data": interaction_data,
                "session_id": result.get("session_id"),
                "site_domain": result.get("site_domain"),
                "screenshot": interaction_data.get("screenshot"),
                "login_url": interaction_data.get("login_url"),
                "prompt": result.get("formatted", "Authentication required."),
            }
            return {
                **state,
                "playbook_state": new_ps,
                "pending_auth": pending_auth,
            }
        return {
            **state,
            "playbook_state": new_ps,
        }
    return _node


def _make_browser_authenticate_node(step: Dict[str, Any]):
    """Open session, verify auth (optional selector), or set pending_auth for interactive login. Uses interrupt_after."""
    async def _node(state: PlaybookGraphState) -> Dict[str, Any]:
        ps = state.get("playbook_state") or {}
        inputs = state.get("inputs") or {}
        user_id = state.get("user_id", "system")
        metadata = state.get("metadata")
        site_domain = (step.get("site_domain") or "").strip()
        login_url = (step.get("login_url") or "").strip()
        verify_url = (step.get("verify_url") or login_url or "").strip()
        verify_selector = (step.get("verify_selector") or "").strip()

        if not site_domain:
            return {
                **state,
                "playbook_state": {**ps, (step.get("output_key") or step.get("name") or "auth"): {
                    "_error": "browser_authenticate requires site_domain",
                    "formatted": "Error: site_domain is required",
                }},
            }

        open_step = {
            "step_type": "tool",
            "action": "browser_open_session",
            "name": "_auth_open",
            "output_key": "_auth_open",
            "inputs": {"site_domain": site_domain},
            "params": {"timeout_seconds": 60},
        }
        open_result = await execute_step(
            open_step, ps, inputs, user_id=user_id, metadata=metadata
        )
        if open_result.get("_error") or not open_result.get("session_id"):
            return {
                **state,
                "playbook_state": {**ps, (step.get("output_key") or step.get("name") or "auth"): open_result},
            }
        session_id = open_result.get("session_id") or ""
        ps = {**ps, "_auth_open": open_result}

        nav_verify_step = {
            "step_type": "tool",
            "action": "browser_navigate",
            "name": "_auth_nav",
            "output_key": "_auth_nav",
            "inputs": {"session_id": session_id, "url": verify_url or login_url},
        }
        nav_result = await execute_step(nav_verify_step, ps, inputs, user_id=user_id, metadata=metadata)
        ps = {**ps, "_auth_nav": nav_result}

        authenticated = False
        if verify_selector:
            extract_step = {
                "step_type": "tool",
                "action": "browser_extract",
                "name": "_auth_check",
                "output_key": "_auth_check",
                "inputs": {"session_id": session_id, "selector": verify_selector},
            }
            check_result = await execute_step(
                extract_step, ps, inputs, user_id=user_id, metadata=metadata
            )
            ps = {**ps, "_auth_check": check_result}
            if not check_result.get("_error"):
                content = (check_result.get("extracted_content") or check_result.get("formatted") or "").strip()
                if content:
                    authenticated = True
        else:
            authenticated = False

        if authenticated:
            key = step.get("output_key") or step.get("name") or "auth_session"
            return {
                **state,
                "playbook_state": {
                    **ps,
                    key: {"session_id": session_id, "authenticated": True, "formatted": f"Authenticated for {site_domain}"},
                },
            }

        if login_url:
            nav_login_step = {
                "step_type": "tool",
                "action": "browser_navigate",
                "name": "_auth_login_nav",
                "output_key": "_auth_login_nav",
                "inputs": {"session_id": session_id, "url": login_url},
            }
            await execute_step(nav_login_step, ps, inputs, user_id=user_id, metadata=metadata)

        screenshot_step = {
            "step_type": "tool",
            "action": "browser_screenshot",
            "name": "_auth_screenshot",
            "output_key": "_auth_screenshot",
            "inputs": {"session_id": session_id},
        }
        screenshot_result = await execute_step(
            screenshot_step, ps, inputs, user_id=user_id, metadata=metadata
        )
        screenshot_data_uri = None
        if isinstance(screenshot_result, dict):
            if screenshot_result.get("images_markdown"):
                markdown = screenshot_result.get("images_markdown", "")
                m = re.search(r"(data:image/[^;]+;base64,[A-Za-z0-9+/=]+)", markdown)
                if m:
                    screenshot_data_uri = m.group(1)
            elif screenshot_result.get("screenshot_b64"):
                screenshot_data_uri = "data:image/png;base64," + screenshot_result.get("screenshot_b64")

        pending_auth = {
            "step_name": step.get("name") or step.get("output_key") or "browser_authenticate",
            "site_domain": site_domain,
            "session_id": session_id,
            "login_url": login_url or verify_url,
            "screenshot": screenshot_data_uri,
            "prompt": f"Log in to {site_domain} in the browser, then confirm when done.",
        }
        key = step.get("output_key") or step.get("name") or "auth_session"
        return {
            **state,
            "playbook_state": {**ps, key: {"session_id": session_id, "authenticated": False, "pending_auth": True}},
            "pending_auth": pending_auth,
        }
    return _node


def _make_approval_node(step: Dict[str, Any]):
    """Return a node that sets pending_approval for HITL. Used with interrupt_after."""
    async def _node(state: PlaybookGraphState) -> Dict[str, Any]:
        name = step.get("name") or step.get("output_key") or ""
        output_key = step.get("output_key")
        preview = (state.get("playbook_state") or {}).get(output_key or name) if (output_key or name) else None
        pending = {
            "step_name": name,
            "preview_data": preview,
            "prompt": step.get("prompt", "Approve to continue?"),
            "timeout_minutes": step.get("timeout_minutes"),
            "on_reject": step.get("on_reject", "stop"),
        }
        return {
            **state,
            "pending_approval": pending,
        }
    return _node


def _make_parallel_node(step: Dict[str, Any]):
    """Return a node that runs parallel_steps concurrently via asyncio.gather."""
    children = step.get("parallel_steps", [])

    async def _node(state: PlaybookGraphState) -> Dict[str, Any]:
        ps = state.get("playbook_state") or {}
        inputs = state.get("inputs") or {}
        user_id = state.get("user_id", "system")
        metadata = state.get("metadata")

        async def _run_child(child: Dict[str, Any]) -> tuple:
            st = _step_type(child)
            child_ps = dict(ps)
            if st == "llm_task":
                result = await _execute_llm_step(child, child_ps, inputs, user_id, metadata=metadata)
            elif st == "llm_agent":
                result = await _execute_llm_agent_step(child, child_ps, inputs, user_id, metadata=metadata)
            else:
                result = await execute_step(child, child_ps, inputs, user_id=user_id, metadata=metadata)
            return child, result

        results = await asyncio.gather(*[_run_child(c) for c in children])
        new_ps = {**ps}
        for child, result in results:
            key = child.get("output_key") or child.get("name")
            if key:
                new_ps[key] = result
        return {
            **state,
            "playbook_state": new_ps,
        }

    return _node


def _build_loop_subgraph(step: Dict[str, Any], checkpointer: Optional[AsyncPostgresSaver] = None):
    """Build a compiled subgraph for loop steps; the node invokes it max_iterations times."""
    child_steps = step.get("steps", [])
    if not child_steps:
        async def _empty_loop_node(state: PlaybookGraphState) -> Dict[str, Any]:
            return state
        return _empty_loop_node

    child_graph = build_playbook_graph(child_steps, checkpointer=checkpointer)

    async def _loop_node(state: PlaybookGraphState) -> Dict[str, Any]:
        max_iter = max(1, int(step.get("max_iterations", 3)))
        ps = state.get("playbook_state") or {}
        inputs = state.get("inputs") or {}
        user_id = state.get("user_id", "system")
        metadata = state.get("metadata")

        for iteration in range(max_iter):
            ps["_iteration"] = iteration + 1
            initial: PlaybookGraphState = {
                "playbook_state": dict(ps),
                "inputs": inputs,
                "user_id": user_id,
                "metadata": metadata or {},
            }
            result = await child_graph.ainvoke(initial)
            ps = result.get("playbook_state") or ps
            if result.get("pending_approval"):
                return {
                    **state,
                    "playbook_state": ps,
                    "pending_approval": result["pending_approval"],
                }
            if result.get("pending_auth"):
                return {
                    **state,
                    "playbook_state": ps,
                    "pending_auth": result["pending_auth"],
                }
        return {
            **state,
            "playbook_state": ps,
        }
    return _loop_node


def _make_branch_node(step: Dict[str, Any], checkpointer: Optional[AsyncPostgresSaver] = None):
    """Return a node that evaluates branch_condition and runs then_steps or else_steps."""
    then_steps = step.get("then_steps", [])
    else_steps = step.get("else_steps", [])
    condition_expr = step.get("branch_condition", "")

    then_graph = build_playbook_graph(then_steps, checkpointer=checkpointer) if then_steps else None
    else_graph = build_playbook_graph(else_steps, checkpointer=checkpointer) if else_steps else None

    async def _node(state: PlaybookGraphState) -> Dict[str, Any]:
        ps = state.get("playbook_state") or {}
        inputs = state.get("inputs") or {}

        condition_met = _evaluate_condition(condition_expr, ps, inputs)
        target_graph = then_graph if condition_met else else_graph

        if target_graph is None:
            return state

        result = await target_graph.ainvoke(
            {
                "playbook_state": dict(ps),
                "inputs": inputs,
                "user_id": state.get("user_id", "system"),
                "metadata": state.get("metadata") or {},
            }
        )
        return {
            **state,
            "playbook_state": result.get("playbook_state") or ps,
        }

    return _node


def build_playbook_graph(
    steps: List[Dict[str, Any]],
    checkpointer: Optional[AsyncPostgresSaver] = None,
):
    """
    Build a compiled LangGraph from playbook steps. Each step becomes a node.
    Approval nodes are listed in interrupt_after so the runner can show HITL UI.
    """
    if not steps:
        workflow = StateGraph(PlaybookGraphState)
        workflow.add_node("_noop", lambda s: s)
        workflow.set_entry_point("_noop")
        workflow.add_edge("_noop", END)
        return workflow.compile(checkpointer=checkpointer)

    workflow = StateGraph(PlaybookGraphState)
    interrupt_after_nodes: List[str] = []
    prev_node_name: Optional[str] = None
    prev_step_type: Optional[str] = None

    for i, step in enumerate(steps):
        step_type = _step_type(step)
        name = step.get("name") or step.get("output_key") or f"step_{i}"

        def _wrap_cond_and_trace(raw_node):
            return _wrap_with_tracing(step, _wrap_node_with_condition(step, raw_node), i)

        if step_type == "approval":
            workflow.add_node(name, _wrap_cond_and_trace(_make_approval_node(step)))
            interrupt_after_nodes.append(name)
        elif step_type == "browser_authenticate":
            workflow.add_node(name, _wrap_cond_and_trace(_make_browser_authenticate_node(step)))
            interrupt_after_nodes.append(name)
        elif step_type == "loop":
            workflow.add_node(name, _wrap_cond_and_trace(_build_loop_subgraph(step, checkpointer=checkpointer)))
            interrupt_after_nodes.append(name)
        elif step_type == "parallel":
            workflow.add_node(name, _wrap_cond_and_trace(_make_parallel_node(step)))
        elif step_type == "branch":
            workflow.add_node(name, _wrap_cond_and_trace(_make_branch_node(step, checkpointer=checkpointer)))
        elif step_type == "llm_agent":
            workflow.add_node(name, _wrap_cond_and_trace(_make_llm_agent_node(step)))
        elif step_type == "deep_agent":
            workflow.add_node(name, _wrap_cond_and_trace(_make_deep_agent_node(step)))
        elif step_type == "llm_task":
            workflow.add_node(name, _wrap_cond_and_trace(_make_llm_task_node(step)))
        else:
            workflow.add_node(name, _wrap_cond_and_trace(_make_tool_node(step)))

        if i == 0:
            workflow.set_entry_point(name)
        if prev_node_name is not None:
            if prev_step_type == "llm_agent":
                next_name = name

                def _route_after_llm_agent(s: PlaybookGraphState, next_node: str = next_name) -> str:
                    return "end" if s.get("pending_auth") else next_node

                workflow.add_conditional_edges(
                    prev_node_name,
                    _route_after_llm_agent,
                    {"end": END, next_name: name},
                )
            else:
                workflow.add_edge(prev_node_name, name)
        prev_node_name = name
        prev_step_type = step_type

    if prev_node_name is not None:
        workflow.add_edge(prev_node_name, END)

    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after_nodes if interrupt_after_nodes else None,
    )
