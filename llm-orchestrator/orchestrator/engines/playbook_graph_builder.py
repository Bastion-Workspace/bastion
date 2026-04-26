"""
Playbook Graph Builder - Builds dynamic LangGraph workflows from Agent Factory playbook steps.

Each playbook step becomes a real LangGraph node. Supports tool, llm_task, llm_agent,
approval, and loop step types. Approval steps use interrupt_after so the runner
can show HITL UI; loop steps run a child graph in a loop.
Output is handled via tool steps (e.g. send_channel_message, save_to_document).
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from config.settings import settings
from orchestrator.engines.playbook_limits import MAX_BEST_OF_N_SAMPLES, MAX_PARALLEL_SUBSTEPS
from orchestrator.engines.pipeline_executor import (
    execute_step,
    _execute_llm_step,
    _execute_llm_agent_step,
    _execute_deep_agent_step,
    _evaluate_condition,
    _get_llm_for_pipeline,
    _resolve_inputs,
)
from orchestrator.utils.async_invoke_timeout import invoke_with_optional_timeout
from orchestrator.engines.playbook_ui_progress import emit_playbook_ui_progress

logger = logging.getLogger(__name__)

# Normalize step type: playbook definitions may use "type" or "step_type"
def _step_type(step: Dict[str, Any]) -> str:
    return (step.get("step_type") or step.get("type") or "tool") or "tool"


def _deep_phases_plan_for_ui(step: Dict[str, Any]) -> str:
    """Compact phase flow for chat status (deep_agent steps)."""
    phases = step.get("phases") or []
    if not isinstance(phases, list) or not phases:
        return ""
    bits: List[str] = []
    for p in phases:
        if not isinstance(p, dict):
            continue
        n = (p.get("name") or "").strip() or "?"
        t = (p.get("type") or "").strip() or "?"
        bits.append(f"{n}:{t}")
    return " → ".join(bits) if bits else ""


_ITEM_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _resolve_dot_path(root: Any, path: str) -> Any:
    """Walk dict keys by dot-separated path (e.g. plan.items). Returns None if missing."""
    if not path or not str(path).strip():
        return None
    cur: Any = root
    for part in str(path).strip().split("."):
        if not part:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _effective_samples(step: Dict[str, Any]) -> int:
    raw = step.get("samples", 1)
    try:
        n = int(raw) if raw is not None else 1
    except (TypeError, ValueError):
        return 1
    return max(1, min(MAX_BEST_OF_N_SAMPLES, n))


def _state_for_fan_item(state: "PlaybookGraphState", item: Any, item_var: str) -> "PlaybookGraphState":
    ps = {**(state.get("playbook_state") or {}), item_var: item}
    inputs = {**(state.get("inputs") or {}), item_var: item}
    return {**state, "playbook_state": ps, "inputs": inputs}


def _compose_fan_out_merged_value(results: List[Dict[str, Any]], merge: str) -> Dict[str, Any]:
    merge_l = (merge or "list").strip().lower()
    count = len(results)
    if merge_l == "concat":
        parts: List[str] = []
        for i, r in enumerate(results):
            if isinstance(r, dict):
                parts.append(f"## Item {i + 1}\n\n" + (r.get("formatted") or ""))
            else:
                parts.append(f"## Item {i + 1}\n\n{str(r)}")
        formatted = "\n\n".join(parts) if parts else ""
        return {"items": results, "formatted": formatted, "_fan_out": True, "_fan_out_count": count}
    formatted = "\n\n---\n\n".join(
        (r.get("formatted") if isinstance(r, dict) else str(r)) or "" for r in results
    )
    return {"items": results, "formatted": formatted, "_fan_out": True, "_fan_out_count": count}


def _last_evaluate_score_from_phase_trace(phase_trace: Any) -> Optional[float]:
    if not isinstance(phase_trace, list):
        return None
    last: Optional[float] = None
    for entry in phase_trace:
        if isinstance(entry, dict) and entry.get("type") == "evaluate" and "score" in entry:
            try:
                last = float(entry["score"])
            except (TypeError, ValueError):
                continue
    return last


def _pick_best_index_by_highest_score(results: List[Dict[str, Any]]) -> Optional[int]:
    scores: List[Optional[float]] = []
    for r in results:
        if not isinstance(r, dict):
            scores.append(None)
        else:
            scores.append(_last_evaluate_score_from_phase_trace(r.get("phase_trace")))
    if all(s is None for s in scores):
        return None
    best_i = 0
    best_s = float("-inf")
    for i, s in enumerate(scores):
        if s is None:
            continue
        if s > best_s:
            best_s = s
            best_i = i
    return best_i


async def _llm_judge_pick_best_index(
    results: List[Dict[str, Any]],
    step: Dict[str, Any],
    state: "PlaybookGraphState",
    criteria: str,
) -> int:
    """0-based index into results; defaults to 0."""
    if not results:
        return 0
    llm = _get_llm_for_pipeline({**(state.get("metadata") or {}), "pipeline_llm_temperature": 0.2})
    if not llm:
        return 0
    chunks: List[str] = []
    for i, r in enumerate(results):
        if isinstance(r, dict):
            body = (r.get("formatted") or r.get("raw") or "")[:8000]
        else:
            body = str(r)[:8000]
        chunks.append(f"### Candidate {i}\n{body}")
    crit = (criteria or "").strip() or "Select the highest quality, most complete response."
    hi = max(0, len(results) - 1)
    prompt = (
        f"You are judging which of {len(results)} model outputs best meets the criteria.\n\n"
        f"Criteria:\n{crit}\n\n"
        + "\n\n".join(chunks)
        + f'\n\nRespond with a single JSON object only: {{"best_index": <integer 0..{hi}>}}.\n'
    )
    try:
        resp = await invoke_with_optional_timeout(
            llm.ainvoke([HumanMessage(content=prompt)]),
            settings.PIPELINE_LLM_INVOKE_TIMEOUT_SEC,
        )
        content = (getattr(resp, "content", None) or "").strip()
        raw = content
        if "```json" in raw:
            start = raw.find("```json") + 7
            end = raw.find("```", start)
            raw = raw[start:end].strip() if end != -1 else raw
        elif "```" in raw:
            start = raw.find("```") + 3
            end = raw.find("```", start)
            raw = raw[start:end].strip() if end != -1 else raw
        parsed = json.loads(raw)
        idx = int(parsed.get("best_index", 0))
        return max(0, min(len(results) - 1, idx))
    except Exception as e:
        logger.warning("Best-of-N LLM judge failed: %s", e)
        return 0


async def _select_best_of_n_result(
    results: List[Dict[str, Any]],
    step: Dict[str, Any],
    state: "PlaybookGraphState",
    *,
    is_deep_agent: bool,
) -> Dict[str, Any]:
    if not results:
        return {"_error": "no_samples", "formatted": "No best-of-N results."}
    strategy = (step.get("selection_strategy") or "llm_judge").strip().lower()
    n = len(results)
    chosen = 0
    if strategy == "highest_score" and is_deep_agent:
        hi = _pick_best_index_by_highest_score(results)
        if hi is not None:
            chosen = hi
        else:
            chosen = await _llm_judge_pick_best_index(
                results, step, state, str(step.get("selection_criteria") or "")
            )
    else:
        chosen = await _llm_judge_pick_best_index(
            results, step, state, str(step.get("selection_criteria") or "")
        )
    out = dict(results[chosen])
    out["_best_of_n"] = {
        "samples": n,
        "chosen_index": chosen,
        "strategy": strategy,
    }
    return out


async def _gather_llm_agent_samples(
    state: "PlaybookGraphState",
    step: Dict[str, Any],
    n: int,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Run n llm_agent executions; return (results, pending_auth if any sample needs interaction)."""
    meta_base = dict(state.get("metadata") or {})

    async def _one_sample() -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        st = {**state, "metadata": {**meta_base, "pipeline_llm_temperature": 0.65}}
        ps = st.get("playbook_state") or {}
        res = await _execute_llm_agent_step(
            step,
            ps,
            st.get("inputs") or {},
            user_id=st.get("user_id", "system"),
            metadata=st.get("metadata"),
        )
        if isinstance(res, dict) and res.get("_interaction_required"):
            interaction_data = res.get("interaction_data") or {}
            pending_auth = {
                "step_name": step.get("name") or step.get("output_key") or "llm_agent",
                "interaction_type": res.get("interaction_type", "browser_login"),
                "interaction_data": interaction_data,
                "session_id": res.get("session_id"),
                "site_domain": res.get("site_domain"),
                "screenshot": interaction_data.get("screenshot"),
                "login_url": interaction_data.get("login_url"),
                "prompt": res.get("formatted", "Authentication required."),
            }
            return res, pending_auth
        return res, None

    if n <= 1:
        ps = state.get("playbook_state") or {}
        meta = dict(meta_base)
        meta.pop("pipeline_llm_temperature", None)
        res = await _execute_llm_agent_step(
            step,
            ps,
            state.get("inputs") or {},
            user_id=state.get("user_id", "system"),
            metadata=meta,
        )
        if isinstance(res, dict) and res.get("_interaction_required"):
            interaction_data = res.get("interaction_data") or {}
            pending_auth = {
                "step_name": step.get("name") or step.get("output_key") or "llm_agent",
                "interaction_type": res.get("interaction_type", "browser_login"),
                "interaction_data": interaction_data,
                "session_id": res.get("session_id"),
                "site_domain": res.get("site_domain"),
                "screenshot": interaction_data.get("screenshot"),
                "login_url": interaction_data.get("login_url"),
                "prompt": res.get("formatted", "Authentication required."),
            }
            return [res], pending_auth
        return [res], None

    pairs = await asyncio.gather(*[_one_sample() for _ in range(n)])
    for res, pend in pairs:
        if pend is not None:
            return [res], pend
    return [p[0] for p in pairs], None


async def _gather_deep_agent_samples(
    state: "PlaybookGraphState",
    step: Dict[str, Any],
    n: int,
) -> List[Dict[str, Any]]:
    meta_base = dict(state.get("metadata") or {})

    async def _one_sample() -> Dict[str, Any]:
        st = {**state, "metadata": {**meta_base, "pipeline_llm_temperature": 0.65}}
        ps = st.get("playbook_state") or {}
        return await _execute_deep_agent_step(
            step,
            ps,
            st.get("inputs") or {},
            user_id=st.get("user_id", "system"),
            metadata=st.get("metadata"),
        )

    if n <= 1:
        meta = dict(meta_base)
        meta.pop("pipeline_llm_temperature", None)
        ps = state.get("playbook_state") or {}
        return [
            await _execute_deep_agent_step(
                step,
                ps,
                state.get("inputs") or {},
                user_id=state.get("user_id", "system"),
                metadata=meta,
            )
        ]
    return list(await asyncio.gather(*[_one_sample() for _ in range(n)]))

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


def _step_history_silent(step: Dict[str, Any]) -> bool:
    """True when step should not append to execution_trace (silent history policy)."""
    raw = (step.get("history_policy") or "").strip().lower()
    return raw in ("silent", "off")


def _is_exclusive(step: Dict[str, Any]) -> bool:
    """When true and the step runs (not condition-skipped), graph routes to END instead of the next node."""
    return bool(step.get("exclusive"))


def _wrap_with_tracing(
    step: Dict[str, Any],
    node_impl,
    step_index: int,
) -> Any:
    """Wrap a node to record execution_trace entry (started_at, completed_at, inputs, outputs, status)."""
    step_name = step.get("name") or step.get("output_key") or f"step_{step_index}"
    step_type = _step_type(step)
    action_name = step.get("action") if step_type == "tool" else None
    silent = _step_history_silent(step)

    async def _node(state: PlaybookGraphState) -> Dict[str, Any]:
        ps = state.get("playbook_state") or {}
        inputs = state.get("inputs") or {}
        inputs_snapshot = _resolve_inputs(step.get("inputs") or {}, ps, inputs)
        started_at = datetime.now(timezone.utc).isoformat()
        trace = list(state.get("execution_trace") or [])

        meta = state.get("metadata")
        if isinstance(meta, dict):
            prog_payload: Dict[str, Any] = {
                "playbook_activity_step": step_name,
                "playbook_activity_type": step_type,
            }
            if step_type == "deep_agent":
                plan = _deep_phases_plan_for_ui(step)
                if plan:
                    prog_payload["deep_phases_plan"] = plan
                prog_payload["deep_phase_name"] = ""
                prog_payload["deep_phase_type"] = ""
            await emit_playbook_ui_progress(meta, prog_payload)

        try:
            result = await node_impl(state)
        except Exception as e:
            logger.exception("Playbook step failed: %s", step_name)
            duration_ms = None
            completed_at = datetime.now(timezone.utc).isoformat()
            if not silent:
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
                    "acquired_tool_log": None,
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
        acquired_tool_log = None
        skill_execution_events = None
        if step_type == "llm_agent" and isinstance(step_result, dict):
            tool_call_trace = step_result.pop("_tool_call_trace", None)
            acquired_tool_log = step_result.pop("_acquired_tool_log", None)
            skill_execution_events = step_result.pop("_skill_execution_events", None)
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
            "acquired_tool_log": acquired_tool_log,
            "skill_execution_events": skill_execution_events,
        }
        if token_usage and isinstance(token_usage, dict):
            trace_entry["input_tokens"] = token_usage.get("input_tokens", 0)
            trace_entry["output_tokens"] = token_usage.get("output_tokens", 0)
        # Do not journal condition-skipped steps: they never ran; a lone inherit step skipped
        # would otherwise leave a trace entry and force log_agent_execution for an otherwise all-silent run.
        if not silent and status != "skipped":
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
        if isinstance(result, dict) and result.get("_needs_human_interaction"):
            idata = result.get("interaction_data") or {}
            pending_auth = {
                "step_name": step.get("name") or step.get("output_key") or "tool",
                "interaction_type": result.get("interaction_type", "shell_command_approval"),
                "interaction_data": idata,
                "approval_id": idata.get("approval_id"),
                "prompt": result.get("formatted", "Approval required."),
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


def _make_deep_agent_node(step: Dict[str, Any]):
    """Return a node that runs a deep_agent step (multi-phase graph), with optional fan-out and best-of-N."""

    async def _node(state: PlaybookGraphState) -> Dict[str, Any]:
        fan = step.get("fan_out")
        n = _effective_samples(step)
        key = step.get("output_key") or step.get("name")

        async def _run_single(sub_state: PlaybookGraphState) -> Dict[str, Any]:
            results = await _gather_deep_agent_samples(sub_state, step, n)
            if n <= 1:
                return results[0]
            return await _select_best_of_n_result(results, step, sub_state, is_deep_agent=True)

        if isinstance(fan, dict) and str(fan.get("source") or "").strip():
            source = str(fan.get("source")).strip()
            items = _resolve_dot_path(state.get("playbook_state") or {}, source)
            if not isinstance(items, list):
                items = [items] if items is not None and items != [] else []
            item_var = (fan.get("item_variable") or "current_item").strip() or "current_item"
            if not _ITEM_VAR_RE.match(item_var):
                item_var = "current_item"
            try:
                mi = int(fan.get("max_items", 10))
            except (TypeError, ValueError):
                mi = 10
            mi = max(1, min(MAX_PARALLEL_SUBSTEPS, mi))
            items = items[:mi]
            if not items:
                new_ps = {**(state.get("playbook_state") or {})}
                empty = {
                    "_fan_out": True,
                    "_fan_out_count": 0,
                    "items": [],
                    "formatted": "(No items to process.)",
                }
                if key:
                    new_ps[key] = empty
                return {**state, "playbook_state": new_ps}
            sub_results = await asyncio.gather(
                *[_run_single(_state_for_fan_item(state, it, item_var)) for it in items]
            )
            merged = _compose_fan_out_merged_value(sub_results, str(fan.get("merge") or "list"))
            new_ps = {**(state.get("playbook_state") or {})}
            if key:
                new_ps[key] = merged
            return {**state, "playbook_state": new_ps}

        result = await _run_single(state)
        new_ps = {**(state.get("playbook_state") or {})}
        if key:
            new_ps[key] = result
        return {**state, "playbook_state": new_ps}

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
    """Return a node that runs an llm_agent step (ReAct), with optional fan-out and best-of-N."""

    async def _node(state: PlaybookGraphState) -> Dict[str, Any]:
        fan = step.get("fan_out")
        n = _effective_samples(step)
        key = step.get("output_key") or step.get("name")

        async def _run_single(sub_state: PlaybookGraphState) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
            results, pend = await _gather_llm_agent_samples(sub_state, step, n)
            if pend is not None:
                return results[0], pend
            if n <= 1:
                return results[0], None
            best = await _select_best_of_n_result(results, step, sub_state, is_deep_agent=False)
            return best, None

        if isinstance(fan, dict) and str(fan.get("source") or "").strip():
            source = str(fan.get("source")).strip()
            items = _resolve_dot_path(state.get("playbook_state") or {}, source)
            if not isinstance(items, list):
                items = [items] if items is not None and items != [] else []
            item_var = (fan.get("item_variable") or "current_item").strip() or "current_item"
            if not _ITEM_VAR_RE.match(item_var):
                item_var = "current_item"
            try:
                mi = int(fan.get("max_items", 10))
            except (TypeError, ValueError):
                mi = 10
            mi = max(1, min(MAX_PARALLEL_SUBSTEPS, mi))
            items = items[:mi]
            if not items:
                new_ps = {**(state.get("playbook_state") or {})}
                empty = {
                    "_fan_out": True,
                    "_fan_out_count": 0,
                    "items": [],
                    "formatted": "(No items to process.)",
                }
                if key:
                    new_ps[key] = empty
                return {**state, "playbook_state": new_ps}

            async def _per_item(item: Any) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
                sub = _state_for_fan_item(state, item, item_var)
                return await _run_single(sub)

            pairs = await asyncio.gather(*[_per_item(it) for it in items])
            for res, pend in pairs:
                if pend is not None:
                    new_ps = {**(state.get("playbook_state") or {})}
                    if key:
                        new_ps[key] = res
                    return {**state, "playbook_state": new_ps, "pending_auth": pend}
            merged = _compose_fan_out_merged_value([p[0] for p in pairs], str(fan.get("merge") or "list"))
            new_ps = {**(state.get("playbook_state") or {})}
            if key:
                new_ps[key] = merged
            return {**state, "playbook_state": new_ps}

        result, pend = await _run_single(state)
        new_ps = {**(state.get("playbook_state") or {})}
        if key:
            new_ps[key] = result
        if pend is not None:
            return {**state, "playbook_state": new_ps, "pending_auth": pend}
        return {**state, "playbook_state": new_ps}

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
    raw = step.get("parallel_steps") or []
    if len(raw) > MAX_PARALLEL_SUBSTEPS:
        logger.warning(
            "Parallel step %s: %d sub-steps exceeds max %d; running first %d only",
            step.get("name") or step.get("output_key") or "parallel",
            len(raw),
            MAX_PARALLEL_SUBSTEPS,
            MAX_PARALLEL_SUBSTEPS,
        )
    children = raw[:MAX_PARALLEL_SUBSTEPS]

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
            result = await invoke_with_optional_timeout(
                child_graph.ainvoke(initial),
                settings.PLAYBOOK_GRAPH_INVOKE_TIMEOUT_SEC,
            )
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
        branch_selected = "then" if condition_met else "else"
        key = step.get("output_key") or step.get("name")
        selected_steps = then_steps if condition_met else else_steps
        selected_step_names = [
            (s.get("name") or s.get("output_key") or "").strip()
            for s in (selected_steps or [])
            if isinstance(s, dict)
        ]
        selected_step_names = [n for n in selected_step_names if n]
        # Small, high-signal snapshot for common conditions like "{bastion_emails.count} > 0"
        count_snapshot = None
        try:
            if isinstance(ps.get("bastion_emails"), dict):
                count_snapshot = ps.get("bastion_emails", {}).get("count")
        except Exception:
            count_snapshot = None

        if target_graph is None:
            new_ps = {**ps}
            if key:
                new_ps[key] = {
                    "branch_condition": condition_expr,
                    "branch_selected": branch_selected,
                    "selected_steps": selected_step_names,
                    "snapshot": {"bastion_emails.count": count_snapshot} if count_snapshot is not None else {},
                    "formatted": f"Branch selected: {branch_selected} (no steps)",
                }
            return {**state, "playbook_state": new_ps}

        result = await invoke_with_optional_timeout(
            target_graph.ainvoke(
                {
                    "playbook_state": dict(ps),
                    "inputs": inputs,
                    "user_id": state.get("user_id", "system"),
                    "metadata": state.get("metadata") or {},
                }
            ),
            settings.PLAYBOOK_GRAPH_INVOKE_TIMEOUT_SEC,
        )
        new_ps = result.get("playbook_state") or ps
        new_ps = {**ps, **new_ps}
        if key:
            new_ps[key] = {
                "branch_condition": condition_expr,
                "branch_selected": branch_selected,
                "selected_steps": selected_step_names,
                "snapshot": {"bastion_emails.count": count_snapshot} if count_snapshot is not None else {},
                "formatted": f"Branch selected: {branch_selected}",
            }
        return {
            **state,
            "playbook_state": new_ps,
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

        def _wrap_cond_and_trace(raw_node, _step=step, _i=i):
            # Default-arg bind: loop `step`/`i` must not close over the final iteration (wrong condition/trace).
            return _wrap_with_tracing(_step, _wrap_node_with_condition(_step, raw_node), _i)

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
            prev_step = steps[i - 1]
            prev_output_key = prev_step.get("output_key") or prev_step.get("name") or f"step_{i - 1}"
            prev_exclusive = _is_exclusive(prev_step)
            next_name = name

            if prev_step_type == "llm_agent" and prev_exclusive:

                def _route_after_exclusive_llm_agent(
                    s: PlaybookGraphState,
                    _pk: str = prev_output_key,
                    _next: str = next_name,
                ) -> str:
                    if s.get("pending_auth"):
                        return "end"
                    ps = s.get("playbook_state") or {}
                    result = ps.get(_pk)
                    if isinstance(result, dict) and result.get("_skipped"):
                        return _next
                    return "end"

                workflow.add_conditional_edges(
                    prev_node_name,
                    _route_after_exclusive_llm_agent,
                    {"end": END, next_name: name},
                )
            elif prev_exclusive:

                def _route_after_exclusive(
                    s: PlaybookGraphState,
                    _pk: str = prev_output_key,
                    _next: str = next_name,
                ) -> str:
                    if s.get("pending_auth"):
                        return "end"
                    ps = s.get("playbook_state") or {}
                    result = ps.get(_pk)
                    if isinstance(result, dict) and result.get("_skipped"):
                        return _next
                    return "end"

                workflow.add_conditional_edges(
                    prev_node_name,
                    _route_after_exclusive,
                    {"end": END, next_name: name},
                )
            elif prev_step_type == "llm_agent":

                def _route_after_llm_agent(s: PlaybookGraphState, next_node: str = next_name) -> str:
                    return "end" if s.get("pending_auth") else next_node

                workflow.add_conditional_edges(
                    prev_node_name,
                    _route_after_llm_agent,
                    {"end": END, next_name: name},
                )
            else:

                def _route_after_step_pending_auth(
                    s: PlaybookGraphState,
                    next_node: str = next_name,
                ) -> str:
                    return "end" if s.get("pending_auth") else next_node

                workflow.add_conditional_edges(
                    prev_node_name,
                    _route_after_step_pending_auth,
                    {"end": END, next_name: name},
                )
        prev_node_name = name
        prev_step_type = step_type

    if prev_node_name is not None:
        workflow.add_edge(prev_node_name, END)

    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after_nodes if interrupt_after_nodes else None,
    )
