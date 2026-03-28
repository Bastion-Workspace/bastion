"""
Deep Agent Executor - Compiles and runs multi-phase reasoning workflows for the deep_agent playbook step.

Phases: reason, act, search, evaluate, synthesize, refine. Builds a LangGraph StateGraph at runtime,
resolves variables from playbook_state + inputs + phase_results, and returns formatted output plus phase_trace.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.engines.pipeline_executor import _extract_usage_metadata

logger = logging.getLogger(__name__)

_DEFAULT_TOKEN_USAGE = {"input_tokens": 0, "output_tokens": 0}


def _merge_token_usage(state: Dict[str, Any], response: Any) -> Dict[str, int]:
    """Accumulate usage from response into state's _token_usage."""
    acc = state.get("_token_usage") or _DEFAULT_TOKEN_USAGE.copy()
    usage = _extract_usage_metadata(response)
    return {
        "input_tokens": acc.get("input_tokens", 0) + usage.get("input_tokens", 0),
        "output_tokens": acc.get("output_tokens", 0) + usage.get("output_tokens", 0),
    }

_REF_PATTERN = re.compile(r"\{([^}]+)\}")


def _ensure_list(value: Any) -> List[str]:
    """Ensure value is a list of strings. Handles JSON strings like '[]' to prevent list('[]') -> ['[', ']']."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(t).strip() for t in value if t and str(t).strip()]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(t).strip() for t in parsed if t and str(t).strip()]
        except (json.JSONDecodeError, TypeError):
            pass
        return []
    return []


def _build_phase_results_for_namespace(phase_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten phase_results so that phase_name.output, phase_name.feedback, phase_name.score are in namespace."""
    out: Dict[str, Any] = {}
    for name, data in (phase_results or {}).items():
        if isinstance(data, dict):
            out[name] = data
        else:
            out[name] = {"output": data}
    return out


async def _run_reason_node(
    phase: Dict[str, Any],
    state: Dict[str, Any],
    llm: Any,
    resolve_fn: Callable[[str, Dict[str, Any]], str],
    system_msg: Optional[SystemMessage],
) -> Dict[str, Any]:
    """Single LLM call, no tools. Store result in phase_results[phase.name]."""
    from datetime import datetime, timezone
    started = datetime.now(timezone.utc)
    phase_name = (phase.get("name") or "").strip()
    template = (phase.get("prompt") or "").strip() or "Analyze the context and respond."
    phase_results = state.get("phase_results") or {}
    namespace = _build_phase_results_for_namespace(phase_results)
    prompt = resolve_fn(template, namespace)
    messages: List[Any] = []
    if system_msg:
        messages.append(system_msg)
    messages.append(HumanMessage(content=prompt))
    response = None
    try:
        if hasattr(llm, "ainvoke"):
            response = await llm.ainvoke(messages)
        else:
            response = llm.invoke(messages)
        content = (getattr(response, "content", "") or "").strip()
    except Exception as e:
        logger.exception("Deep agent reason phase failed: %s", e)
        content = f"Error: {e}"
    new_results = dict(phase_results)
    new_results[phase_name] = {"output": content}
    trace = list(state.get("phase_trace") or [])
    duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
    trace.append({"phase": phase_name, "type": "reason", "status": "completed", "duration_ms": duration_ms})
    token_usage = _merge_token_usage(state, response) if response else (state.get("_token_usage") or _DEFAULT_TOKEN_USAGE).copy()
    return {"phase_results": new_results, "phase_trace": trace, "_token_usage": token_usage}


async def _run_search_node(
    phase: Dict[str, Any],
    state: Dict[str, Any],
    tools_map: Dict[str, Tuple[Any, Any]],
    resolve_fn: Callable[[str, Dict[str, Any]], str],
    user_id: str,
    metadata: Optional[Dict[str, Any]],
    step_palette_tools: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run search_tools in parallel, merge results, store in phase_results[phase.name].
    If phase has no search_tools/available_tools and step_palette_tools is provided, use step_palette_tools."""
    from datetime import datetime, timezone
    started = datetime.now(timezone.utc)
    phase_name = (phase.get("name") or "").strip()
    tool_names = _ensure_list(phase.get("search_tools")) or _ensure_list(phase.get("available_tools"))
    if not tool_names and step_palette_tools:
        tool_names = [t for t in step_palette_tools if t in tools_map]
    strategy = (phase.get("strategy") or "parallel").lower()
    phase_results = state.get("phase_results") or {}
    namespace = _build_phase_results_for_namespace(phase_results)
    prompt = (phase.get("prompt") or "").strip()
    if prompt:
        prompt = resolve_fn(prompt, namespace)
    parts: List[str] = [prompt] if prompt else []
    if strategy == "sequential":
        for tname in tool_names:
            t = tools_map.get(tname)
            if not t:
                continue
            func = t[0] if isinstance(t, tuple) else t
            try:
                import inspect
                sig = getattr(func, "__signature__", None) or inspect.signature(func)
                kwargs: Dict[str, Any] = {}
                if "user_id" in sig.parameters:
                    kwargs["user_id"] = user_id
                if asyncio.iscoroutinefunction(func):
                    out = await func(**kwargs)
                else:
                    out = func(**kwargs)
                parts.append(f"--- {tname} ---\n{(out.get('formatted', str(out)) if isinstance(out, dict) else out)}")
            except Exception as e:
                logger.warning("Search tool %s failed: %s", tname, e)
                parts.append(f"--- {tname} --- Error: {e}")
    else:
        async def _run_one(name: str) -> Tuple[str, str]:
            t = tools_map.get(name)
            if not t:
                return name, ""
            func = t[0] if isinstance(t, tuple) else t
            try:
                import inspect
                sig = getattr(func, "__signature__", None) or inspect.signature(func)
                kwargs = {}
                if "user_id" in sig.parameters:
                    kwargs["user_id"] = user_id
                if asyncio.iscoroutinefunction(func):
                    out = await func(**kwargs)
                else:
                    out = func(**kwargs)
                return name, (out.get("formatted", str(out)) if isinstance(out, dict) else str(out))
            except Exception as e:
                logger.warning("Search tool %s failed: %s", name, e)
                return name, f"Error: {e}"

        results = await asyncio.gather(*[_run_one(n) for n in tool_names])
        for name, text in results:
            if text:
                parts.append(f"--- {name} ---\n{text}")
    content = "\n\n".join(parts) if parts else "No results."
    new_results = dict(phase_results)
    new_results[phase_name] = {"output": content}
    trace = list(state.get("phase_trace") or [])
    duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
    trace.append({"phase": phase_name, "type": "search", "status": "completed", "duration_ms": duration_ms})
    token_usage = state.get("_token_usage") or _DEFAULT_TOKEN_USAGE.copy()
    return {"phase_results": new_results, "phase_trace": trace, "_token_usage": token_usage}


async def _run_evaluate_node(
    phase: Dict[str, Any],
    state: Dict[str, Any],
    llm: Any,
    resolve_fn: Callable[[str, Dict[str, Any]], str],
    system_msg: Optional[SystemMessage],
) -> Dict[str, Any]:
    """LLM evaluates; return score, pass, feedback. Store in phase_results and set routing."""
    phase_name = (phase.get("name") or "").strip()
    criteria = (phase.get("criteria") or "").strip() or "Evaluate quality and completeness."
    threshold = float(phase.get("pass_threshold", 0.7))
    max_retries = max(0, int(phase.get("max_retries", 2)))
    phase_results = state.get("phase_results") or {}
    iteration_counts = dict(state.get("iteration_counts") or {})
    namespace = _build_phase_results_for_namespace(phase_results)
    resolved_criteria = resolve_fn(criteria, namespace)
    prompt = (
        f"Criteria:\n{resolved_criteria}\n\n"
        'Respond with a single JSON object: {"score": <0-1 number>, "pass": <boolean>, "feedback": "<string>"}.'
    )
    messages: List[Any] = []
    if system_msg:
        messages.append(system_msg)
    messages.append(HumanMessage(content=prompt))
    response = None
    try:
        if hasattr(llm, "ainvoke"):
            response = await llm.ainvoke(messages)
        else:
            response = llm.invoke(messages)
        content = (getattr(response, "content", "") or "").strip()
    except Exception as e:
        logger.exception("Deep agent evaluate phase failed: %s", e)
        content = '{"score": 0, "pass": false, "feedback": "Evaluation failed."}'
    score = 0.0
    passed = False
    feedback = ""
    if content:
        raw = content
        if "```json" in raw:
            start = raw.find("```json") + 7
            end = raw.find("```", start)
            raw = raw[start:end].strip() if end != -1 else raw
        elif "```" in raw:
            start = raw.find("```") + 3
            end = raw.find("```", start)
            raw = raw[start:end].strip() if end != -1 else raw
        try:
            parsed = json.loads(raw)
            score = float(parsed.get("score", 0))
            passed = bool(parsed.get("pass", score >= threshold))
            feedback = str(parsed.get("feedback", ""))
        except json.JSONDecodeError:
            pass
    current_iter = iteration_counts.get(phase_name, 0)
    if not passed and current_iter >= max_retries:
        passed = True
    new_results = dict(phase_results)
    new_results[phase_name] = {"output": content, "score": score, "pass": passed, "feedback": feedback}
    new_counts = dict(iteration_counts)
    if not passed:
        new_counts[phase_name] = current_iter + 1
    trace = list(state.get("phase_trace") or [])
    trace.append({
        "phase": phase_name,
        "type": "evaluate",
        "status": "completed",
        "score": score,
        "pass": passed,
        "iteration": new_counts.get(phase_name, 0),
    })
    token_usage = _merge_token_usage(state, response) if response else (state.get("_token_usage") or _DEFAULT_TOKEN_USAGE).copy()
    return {
        "phase_results": new_results,
        "iteration_counts": new_counts,
        "phase_trace": trace,
        "_evaluate_pass": passed,
        "_evaluate_on_pass": phase.get("on_pass") or "end",
        "_evaluate_on_fail": phase.get("on_fail"),
        "_token_usage": token_usage,
    }


async def _run_synthesize_node(
    phase: Dict[str, Any],
    state: Dict[str, Any],
    llm: Any,
    resolve_fn: Callable[[str, Dict[str, Any]], str],
    system_msg: Optional[SystemMessage],
) -> Dict[str, Any]:
    """Single LLM call to combine context; store in phase_results[phase.name]."""
    from datetime import datetime, timezone
    started = datetime.now(timezone.utc)
    phase_name = (phase.get("name") or "").strip()
    template = (phase.get("prompt") or "").strip() or "Synthesize the information above."
    phase_results = state.get("phase_results") or {}
    namespace = _build_phase_results_for_namespace(phase_results)
    prompt = resolve_fn(template, namespace)
    messages: List[Any] = []
    if system_msg:
        messages.append(system_msg)
    messages.append(HumanMessage(content=prompt))
    response = None
    try:
        if hasattr(llm, "ainvoke"):
            response = await llm.ainvoke(messages)
        else:
            response = llm.invoke(messages)
        content = (getattr(response, "content", "") or "").strip()
    except Exception as e:
        logger.exception("Deep agent synthesize phase failed: %s", e)
        content = f"Error: {e}"
    new_results = dict(phase_results)
    new_results[phase_name] = {"output": content}
    token_usage = _merge_token_usage(state, response) if response else (state.get("_token_usage") or _DEFAULT_TOKEN_USAGE).copy()
    return {"phase_results": new_results, "phase_trace": state.get("phase_trace", []), "_token_usage": token_usage}


async def _run_refine_node(
    phase: Dict[str, Any],
    state: Dict[str, Any],
    llm: Any,
    resolve_fn: Callable[[str, Dict[str, Any]], str],
    system_msg: Optional[SystemMessage],
) -> Dict[str, Any]:
    """LLM revises target phase output using feedback; store in phase_results[phase.name]."""
    from datetime import datetime, timezone
    started = datetime.now(timezone.utc)
    phase_name = (phase.get("name") or "").strip()
    template = (phase.get("prompt") or "").strip() or "Revise the content based on feedback."
    phase_results = state.get("phase_results") or {}
    namespace = _build_phase_results_for_namespace(phase_results)
    prompt = resolve_fn(template, namespace)
    messages: List[Any] = []
    if system_msg:
        messages.append(system_msg)
    messages.append(HumanMessage(content=prompt))
    response = None
    try:
        if hasattr(llm, "ainvoke"):
            response = await llm.ainvoke(messages)
        else:
            response = llm.invoke(messages)
        content = (getattr(response, "content", "") or "").strip()
    except Exception as e:
        logger.exception("Deep agent refine phase failed: %s", e)
        content = f"Error: {e}"
    new_results = dict(phase_results)
    new_results[phase_name] = {"output": content}
    trace = list(state.get("phase_trace") or [])
    duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
    trace.append({"phase": phase_name, "type": "refine", "status": "completed", "duration_ms": duration_ms})
    token_usage = _merge_token_usage(state, response) if response else (state.get("_token_usage") or _DEFAULT_TOKEN_USAGE).copy()
    return {"phase_results": new_results, "phase_trace": trace, "_token_usage": token_usage}


async def _run_act_node(
    phase: Dict[str, Any],
    state: Dict[str, Any],
    llm: Any,
    tools_map: Dict[str, Tuple[Any, Any]],
    resolve_fn: Callable[[str, Dict[str, Any]], str],
    system_msg: Optional[SystemMessage],
    user_id: str,
    metadata: Optional[Dict[str, Any]],
    execute_llm_agent_step_fn: Any,
    step_palette_tools: Optional[List[str]] = None,
    parent_step_for_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a mini ReACT loop using available_tools via passed execute_llm_agent_step_fn."""
    phase_name = (phase.get("name") or "").strip()
    template = (phase.get("prompt") or "").strip() or "Use the available tools to complete the task."
    phase_results = state.get("phase_results") or {}
    playbook_state = dict(state.get("playbook_state") or {})
    inputs = state.get("inputs") or {}
    namespace = _build_phase_results_for_namespace(phase_results)
    for k, v in namespace.items():
        playbook_state[k] = v
    prompt = resolve_fn(template, namespace)
    phase_tools = _ensure_list(phase.get("available_tools")) or _ensure_list(phase.get("search_tools"))
    if not phase_tools and step_palette_tools:
        phase_tools = [t for t in step_palette_tools if t in tools_map]
    fake_step = {
        "name": phase_name,
        "output_key": phase_name,
        "prompt": prompt,
        "available_tools": phase_tools,
        "max_iterations": phase.get("max_iterations", 5),
    }
    if parent_step_for_policy is not None and "user_facts_policy" in parent_step_for_policy:
        fake_step["user_facts_policy"] = parent_step_for_policy["user_facts_policy"]
    result = await execute_llm_agent_step_fn(
        fake_step,
        playbook_state,
        inputs,
        user_id=user_id,
        metadata=metadata,
    )
    formatted = result.get("formatted", str(result))
    new_results = dict(phase_results)
    new_results[phase_name] = {"output": formatted}
    trace = list(state.get("phase_trace") or [])
    trace.append({"phase": phase_name, "type": "act", "status": "completed"})
    token_usage = dict(state.get("_token_usage") or _DEFAULT_TOKEN_USAGE)
    step_usage = result.get("_token_usage")
    if step_usage:
        token_usage = {
            "input_tokens": token_usage.get("input_tokens", 0) + step_usage.get("input_tokens", 0),
            "output_tokens": token_usage.get("output_tokens", 0) + step_usage.get("output_tokens", 0),
        }
    return {"phase_results": new_results, "phase_trace": trace, "_token_usage": token_usage}


def build_deep_agent_graph(
    phases: List[Dict[str, Any]],
    resolve_fn: Callable[[str, Dict[str, Any]], str],
    llm: Any,
    tools_map: Dict[str, Tuple[Any, Any]],
    playbook_state: Dict[str, Any],
    inputs: Dict[str, Any],
    user_id: str,
    metadata: Optional[Dict[str, Any]],
    execute_llm_agent_step_fn: Any,
    system_msg: Optional[SystemMessage] = None,
    step_palette_tools: Optional[List[str]] = None,
    parent_step_for_policy: Optional[Dict[str, Any]] = None,
):
    """
    Build a LangGraph StateGraph from phase definitions. Each phase is a node.
    Returns (compiled_graph, initial_state_dict).
    """
    from langgraph.graph import END, StateGraph

    phase_names = []
    for p in phases:
        name = (p.get("name") or "").strip()
        if name:
            phase_names.append(name)

    if not phase_names:
        workflow = StateGraph(dict)
        workflow.add_node("_noop", lambda s: s)
        workflow.set_entry_point("_noop")
        workflow.add_edge("_noop", END)
        return workflow.compile(), {
            "phase_results": {},
            "phase_trace": [],
            "playbook_state": playbook_state,
            "inputs": inputs,
            "_token_usage": _DEFAULT_TOKEN_USAGE.copy(),
        }

    workflow = StateGraph(dict)
    node_handlers: Dict[str, Any] = {}

    for phase in phases:
        pname = (phase.get("name") or "").strip()
        if not pname:
            continue
        ptype = (phase.get("type") or "").strip().lower()

        if ptype == "reason":
            async def _reason(s, _p=phase, _llm=llm, _res=resolve_fn, _sys=system_msg):
                return await _run_reason_node(_p, s, _llm, _res, _sys)
            node_handlers[pname] = _reason
        elif ptype == "search":
            async def _search(s, _p=phase, _t=tools_map, _res=resolve_fn, _uid=user_id, _meta=metadata, _palette=step_palette_tools):
                return await _run_search_node(_p, s, _t, _res, _uid, _meta, _palette)
            node_handlers[pname] = _search
        elif ptype == "evaluate":
            async def _evaluate(s, _p=phase, _llm=llm, _res=resolve_fn, _sys=system_msg):
                return await _run_evaluate_node(_p, s, _llm, _res, _sys)
            node_handlers[pname] = _evaluate
        elif ptype == "synthesize":
            async def _synthesize(s, _p=phase, _llm=llm, _res=resolve_fn, _sys=system_msg):
                return await _run_synthesize_node(_p, s, _llm, _res, _sys)
            node_handlers[pname] = _synthesize
        elif ptype == "refine":
            async def _refine(s, _p=phase, _llm=llm, _res=resolve_fn, _sys=system_msg):
                return await _run_refine_node(_p, s, _llm, _res, _sys)
            node_handlers[pname] = _refine
        elif ptype == "act":
            async def _act(
                s,
                _p=phase,
                _llm=llm,
                _t=tools_map,
                _res=resolve_fn,
                _sys=system_msg,
                _uid=user_id,
                _meta=metadata,
                _exec=execute_llm_agent_step_fn,
                _palette=step_palette_tools,
                _parent=parent_step_for_policy,
            ):
                return await _run_act_node(
                    _p, s, _llm, _t, _res, _sys, _uid, _meta, _exec, _palette, _parent
                )
            node_handlers[pname] = _act
        else:
            async def _fallback(s, _p=phase, _llm=llm, _res=resolve_fn, _sys=system_msg):
                return await _run_reason_node(_p, s, _llm, _res, _sys)
            node_handlers[pname] = _fallback

    for name, handler in node_handlers.items():
        workflow.add_node(name, handler)

    first = phase_names[0]
    workflow.set_entry_point(first)

    for i, phase in enumerate(phases):
        pname = (phase.get("name") or "").strip()
        if not pname or pname not in node_handlers:
            continue
        ptype = (phase.get("type") or "").strip().lower()

        if ptype == "evaluate":
            on_pass = (phase.get("on_pass") or "end").strip().lower()
            on_fail = (phase.get("on_fail") or "").strip()
            pass_target = on_pass if on_pass != "end" and on_pass in node_handlers else None
            fail_target = on_fail if on_fail in node_handlers else None

            def _route_evaluate(state, _pt=pass_target, _ft=fail_target):
                if state.get("_evaluate_pass"):
                    return _pt if _pt else "__end__"
                return _ft if _ft else "__end__"

            edges_map: Dict[str, Any] = {"__end__": END}
            if pass_target:
                edges_map[pass_target] = pass_target
            if fail_target:
                edges_map[fail_target] = fail_target
            workflow.add_conditional_edges(pname, _route_evaluate, edges_map)
        elif ptype == "refine":
            next_name = (phase.get("next") or "").strip()
            next_target = next_name if next_name in node_handlers else (phase_names[i + 1] if i + 1 < len(phase_names) else None)
            if next_target:
                workflow.add_edge(pname, next_target)
            else:
                workflow.add_edge(pname, END)
        else:
            if i + 1 >= len(phase_names):
                workflow.add_edge(pname, END)
            else:
                next_name = phase_names[i + 1]
                workflow.add_edge(pname, next_name)

    initial_state = {
        "phase_results": {},
        "iteration_counts": {},
        "phase_trace": [],
        "playbook_state": playbook_state,
        "inputs": inputs,
        "user_id": user_id,
        "metadata": metadata,
        "_token_usage": _DEFAULT_TOKEN_USAGE.copy(),
    }
    return workflow.compile(), initial_state


async def run_deep_agent(
    phases: List[Dict[str, Any]],
    resolve_fn: Callable[[str, Dict[str, Any]], str],
    llm: Any,
    tools_map: Dict[str, Tuple[Any, Any]],
    playbook_state: Dict[str, Any],
    inputs: Dict[str, Any],
    user_id: str,
    metadata: Optional[Dict[str, Any]],
    execute_llm_agent_step_fn: Any,
    system_msg: Optional[SystemMessage] = None,
    step_palette_tools: Optional[List[str]] = None,
    parent_step_for_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the deep agent graph, run it, and return {formatted, phase_trace, raw, ...}.
    The last phase that produces output (synthesize or refine) provides 'formatted'.
    """
    graph, initial_state = build_deep_agent_graph(
        phases=phases,
        resolve_fn=resolve_fn,
        llm=llm,
        tools_map=tools_map,
        playbook_state=playbook_state,
        inputs=inputs,
        user_id=user_id,
        metadata=metadata,
        execute_llm_agent_step_fn=execute_llm_agent_step_fn,
        system_msg=system_msg,
        step_palette_tools=step_palette_tools,
        parent_step_for_policy=parent_step_for_policy,
    )
    final = await graph.ainvoke(initial_state)
    phase_results = final.get("phase_results") or {}
    phase_trace = final.get("phase_trace") or []
    phase_names_list = [(p.get("name") or "").strip() for p in phases if (p.get("name") or "").strip()]
    formatted = ""
    raw = ""
    for name in reversed(phase_names_list):
        pr = phase_results.get(name) or {}
        if isinstance(pr, dict) and pr.get("output"):
            formatted = pr["output"]
            raw = pr["output"]
            break
    return {
        "formatted": formatted,
        "raw": raw,
        "phase_trace": phase_trace,
        "phase_results": phase_results,
        "_token_usage": final.get("_token_usage") or _DEFAULT_TOKEN_USAGE.copy(),
    }
