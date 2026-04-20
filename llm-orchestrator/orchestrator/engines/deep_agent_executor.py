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

from config.settings import settings
from orchestrator.engines.pipeline_executor import _extract_usage_metadata
from orchestrator.utils.async_invoke_timeout import invoke_with_optional_timeout
from orchestrator.engines.tool_resolution import (
    inject_skill_manifest_effective,
    max_runtime_skill_acquisitions_from_step,
)

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
            response = await invoke_with_optional_timeout(
                llm.ainvoke(messages),
                settings.PIPELINE_LLM_INVOKE_TIMEOUT_SEC,
            )
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
    If phase has no search_tools/available_tools and step_palette_tools is provided, use step_palette_tools.
    Also collects raw_results (list of structured doc dicts) from tools that return them,
    so a downstream rerank phase can reorder by cross-encoder relevance."""
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
    all_raw_results: List[Dict[str, Any]] = []

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
                if isinstance(out, dict):
                    raw = out.get("documents") or out.get("results") or []
                    if isinstance(raw, list):
                        all_raw_results.extend(raw)
            except Exception as e:
                logger.warning("Search tool %s failed: %s", tname, e)
                parts.append(f"--- {tname} --- Error: {e}")
    else:
        async def _run_one(name: str) -> Tuple[str, str, List[Dict[str, Any]]]:
            t = tools_map.get(name)
            if not t:
                return name, "", []
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
                text = out.get("formatted", str(out)) if isinstance(out, dict) else str(out)
                raw: List[Dict[str, Any]] = []
                if isinstance(out, dict):
                    r = out.get("documents") or out.get("results") or []
                    if isinstance(r, list):
                        raw = r
                return name, text, raw
            except Exception as e:
                logger.warning("Search tool %s failed: %s", name, e)
                return name, f"Error: {e}", []

        gathered = await asyncio.gather(*[_run_one(n) for n in tool_names])
        for name, text, raw in gathered:
            if text:
                parts.append(f"--- {name} ---\n{text}")
            all_raw_results.extend(raw)

    content = "\n\n".join(parts) if parts else "No results."
    new_results = dict(phase_results)
    new_results[phase_name] = {"output": content, "raw_results": all_raw_results}
    trace = list(state.get("phase_trace") or [])
    duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
    trace.append({"phase": phase_name, "type": "search", "status": "completed", "duration_ms": duration_ms})
    token_usage = state.get("_token_usage") or _DEFAULT_TOKEN_USAGE.copy()
    return {"phase_results": new_results, "phase_trace": trace, "_token_usage": token_usage}


async def _run_rerank_node(
    phase: Dict[str, Any],
    state: Dict[str, Any],
    tools_map: Dict[str, Tuple[Any, Any]],
    resolve_fn: Callable[[str, Dict[str, Any]], str],
    user_id: str,
    metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Rerank raw_results from a preceding search phase using the rerank_documents tool.

    Phase config:
      source_phase: name of the search phase whose raw_results to rerank
                    (defaults to the most recent phase that has raw_results)
      top_n:        number of results to keep after reranking (default 10)

    Degrades gracefully when:
      - rerank_documents tool is not in the tool palette
      - source phase has no raw_results (e.g. older search tools)
      - rerank call fails

    In degraded mode the source phase's output is passed through unchanged.
    """
    from datetime import datetime, timezone
    started = datetime.now(timezone.utc)
    phase_name = (phase.get("name") or "").strip()
    source_phase_name = (phase.get("source_phase") or "").strip()
    top_n = max(1, int(phase.get("top_n") or 10))

    phase_results = state.get("phase_results") or {}
    namespace = _build_phase_results_for_namespace(phase_results)
    trace = list(state.get("phase_trace") or [])
    token_usage = state.get("_token_usage") or _DEFAULT_TOKEN_USAGE.copy()

    # Locate source data: explicit source_phase or last phase with raw_results
    source_data: Optional[Dict[str, Any]] = None
    if source_phase_name:
        source_data = phase_results.get(source_phase_name)
    if not source_data:
        for v in reversed(list(phase_results.values())):
            if isinstance(v, dict) and v.get("raw_results"):
                source_data = v
                break

    raw_results: List[Dict[str, Any]] = (source_data or {}).get("raw_results") or []
    passthrough_output: str = (source_data or {}).get("output", "No results.")

    def _degrade(reason: str) -> Dict[str, Any]:
        logger.debug("rerank phase '%s' degraded: %s", phase_name, reason)
        new_results = dict(phase_results)
        new_results[phase_name] = {"output": passthrough_output, "raw_results": raw_results}
        duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
        trace.append({"phase": phase_name, "type": "rerank", "status": "degraded", "reason": reason, "duration_ms": duration_ms})
        return {"phase_results": new_results, "phase_trace": trace, "_token_usage": token_usage}

    if not raw_results:
        return _degrade("no raw_results in source phase")

    # Resolve the query from inputs namespace
    inputs = state.get("inputs") or {}
    query = inputs.get("query") or resolve_fn("{query}", namespace) or ""
    if not query:
        return _degrade("could not resolve query")

    # Find the rerank tool in the palette
    rerank_entry = tools_map.get("rerank_documents") or tools_map.get("rerank_documents_tool")
    if not rerank_entry:
        return _degrade("rerank_documents not in tool palette")

    rerank_fn = rerank_entry[0] if isinstance(rerank_entry, tuple) else rerank_entry

    # Extract text from raw results — prefer content_preview, then text
    documents = [
        (r.get("content_preview") or r.get("text") or "").strip()
        for r in raw_results
    ]
    # Filter empty strings but track original indices
    indexed = [(i, d) for i, d in enumerate(documents) if d]
    if not indexed:
        return _degrade("raw_results have no text content")

    original_indices, doc_texts = zip(*indexed)

    try:
        import inspect
        kwargs: Dict[str, Any] = {"query": query, "documents": list(doc_texts), "top_n": top_n}
        sig = getattr(rerank_fn, "__signature__", None) or inspect.signature(rerank_fn)
        if "user_id" in sig.parameters:
            kwargs["user_id"] = user_id
        if asyncio.iscoroutinefunction(rerank_fn):
            result = await rerank_fn(**kwargs)
        else:
            result = rerank_fn(**kwargs)
    except Exception as e:
        logger.warning("rerank_documents call failed in phase '%s': %s", phase_name, e)
        return _degrade(f"tool call failed: {e}")

    reranked_items = result.get("results", []) if isinstance(result, dict) else []
    formatted_output = result.get("formatted", passthrough_output) if isinstance(result, dict) else passthrough_output

    # Rebuild raw_results in reranked order so further phases can use them
    reranked_raw: List[Dict[str, Any]] = []
    for item in reranked_items:
        relative_idx = item.get("index", 0)
        if 0 <= relative_idx < len(original_indices):
            original_idx = original_indices[relative_idx]
            if 0 <= original_idx < len(raw_results):
                entry = dict(raw_results[original_idx])
                entry["rerank_score"] = item.get("relevance_score", 0.0)
                reranked_raw.append(entry)

    new_results = dict(phase_results)
    new_results[phase_name] = {"output": formatted_output, "raw_results": reranked_raw}
    duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
    trace.append({
        "phase": phase_name,
        "type": "rerank",
        "status": "completed",
        "source_phase": source_phase_name or "(auto)",
        "input_count": len(raw_results),
        "output_count": len(reranked_raw),
        "duration_ms": duration_ms,
    })
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
            response = await invoke_with_optional_timeout(
                llm.ainvoke(messages),
                settings.PIPELINE_LLM_INVOKE_TIMEOUT_SEC,
            )
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
            response = await invoke_with_optional_timeout(
                llm.ainvoke(messages),
                settings.PIPELINE_LLM_INVOKE_TIMEOUT_SEC,
            )
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
            response = await invoke_with_optional_timeout(
                llm.ainvoke(messages),
                settings.PIPELINE_LLM_INVOKE_TIMEOUT_SEC,
            )
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
    # Phases inherit the full step palette — no per-phase tool narrowing.
    # Legacy phase-level available_tools are ignored in favour of the resolved palette.
    phase_tools = list(step_palette_tools or [])
    if not phase_tools:
        phase_tools = list(tools_map.keys())
    if parent_step_for_policy and inject_skill_manifest_effective(parent_step_for_policy):
        if "search_and_acquire_skills" in tools_map and "search_and_acquire_skills" not in phase_tools:
            phase_tools = list(phase_tools) + ["search_and_acquire_skills"]
    fake_step = {
        "name": phase_name,
        "output_key": phase_name,
        "prompt": prompt,
        "available_tools": phase_tools,
        "max_iterations": phase.get("max_iterations", 5),
    }
    if parent_step_for_policy is not None and "user_facts_policy" in parent_step_for_policy:
        fake_step["user_facts_policy"] = parent_step_for_policy["user_facts_policy"]
    if parent_step_for_policy is not None and "agent_memory_policy" in parent_step_for_policy:
        fake_step["agent_memory_policy"] = parent_step_for_policy["agent_memory_policy"]
    if parent_step_for_policy is not None and "persona_policy" in parent_step_for_policy:
        fake_step["persona_policy"] = parent_step_for_policy["persona_policy"]
    if parent_step_for_policy is not None and "history_policy" in parent_step_for_policy:
        fake_step["history_policy"] = parent_step_for_policy["history_policy"]
    if parent_step_for_policy:
        for _dk in (
            "skill_discovery_mode",
            "max_discovered_skills",
            "max_skill_acquisitions",
            "max_auto_skills",
            "auto_discover_skills",
            "dynamic_tool_discovery",
        ):
            if _dk in parent_step_for_policy:
                fake_step[_dk] = parent_step_for_policy[_dk]
        if inject_skill_manifest_effective(parent_step_for_policy):
            fake_step["max_skill_acquisitions"] = max_runtime_skill_acquisitions_from_step(
                parent_step_for_policy
            )
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
        elif ptype == "rerank":
            async def _rerank(s, _p=phase, _t=tools_map, _res=resolve_fn, _uid=user_id, _meta=metadata):
                return await _run_rerank_node(_p, s, _t, _res, _uid, _meta)
            node_handlers[pname] = _rerank
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
    try:
        final = await invoke_with_optional_timeout(
            graph.ainvoke(initial_state),
            settings.PLAYBOOK_GRAPH_INVOKE_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        cap = settings.PLAYBOOK_GRAPH_INVOKE_TIMEOUT_SEC
        logger.warning("Deep agent graph ainvoke timed out after %s s", cap)
        return {
            "formatted": f"Deep agent graph timed out after {cap}s.",
            "raw": "",
            "phase_trace": [],
            "phase_results": {},
            "_token_usage": _DEFAULT_TOKEN_USAGE.copy(),
            "_error": "graph_invoke_timeout",
        }
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
