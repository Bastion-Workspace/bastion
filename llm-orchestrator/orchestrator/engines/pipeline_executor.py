"""
Pipeline Executor - Executes Agent Factory playbook steps with typed data flow.

Handles tool steps (deterministic), llm_task steps (LLM call), and approval steps (HITL).
Resolves step inputs from {step_name.field} and playbook input variables,
applies type coercion via the Action I/O Registry, and stores typed results in playbook_state.
"""

import asyncio
import inspect
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Type, get_type_hints

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field, create_model

from orchestrator.utils.line_context import line_id_from_metadata
from orchestrator.middleware.structured_output_parser import StructuredOutputParser

try:
    from openai import BadRequestError as OpenAIBadRequestError
except ImportError:
    OpenAIBadRequestError = None

logger = logging.getLogger(__name__)

# Normalize step type: playbook definitions may use "type" or "step_type"
def _step_type(step: Dict[str, Any]) -> str:
    return (step.get("step_type") or step.get("type") or "tool") or "tool"

# Pattern for variable references: {step_name.field} or {var_name}
_REF_PATTERN = re.compile(r"\{([^}]+)\}")

# Max length for string values in tool call trace args (avoid bloat in execution log)
_TRACE_ARGS_MAX = 512

# Max characters for a single tool result sent to the LLM (~20K tokens); backstop to prevent context overflow
MAX_TOOL_RESULT_CHARS = 80000

# Tool names that read/write user facts. When profile has include_user_facts=False, these are excluded so the agent runs "in a vacuum."
USER_FACT_TOOL_NAMES = frozenset({"get_user_facts", "save_user_fact"})

_VALID_USER_FACTS_POLICIES = frozenset({"inherit", "no_write", "isolated"})


def _effective_user_facts_policy(
    step: Optional[Dict[str, Any]], metadata: Optional[Dict[str, Any]]
) -> str:
    """
    Effective policy for this step: vacuum | inherit | no_write | isolated.
    vacuum matches legacy profile-wide disable (include_user_facts false); step cannot expand beyond that.
    """
    if metadata is not None and not metadata.get("include_user_facts", True):
        return "vacuum"
    raw = (step or {}).get("user_facts_policy")
    if raw is None or raw == "":
        return "inherit"
    policy = str(raw).strip().lower()
    if policy not in _VALID_USER_FACTS_POLICIES:
        logger.warning("Invalid user_facts_policy %r on step, using inherit", raw)
        return "inherit"
    return policy


def _filter_user_fact_tools_by_policy(tool_names: List[str], policy: str) -> List[str]:
    """Remove user-fact tools according to effective policy (vacuum, isolated, no_write, inherit)."""
    if policy in ("vacuum", "isolated"):
        return [n for n in tool_names if n not in USER_FACT_TOOL_NAMES]
    if policy == "no_write":
        return [n for n in tool_names if n != "save_user_fact"]
    return tool_names


def _core_tool_name_from_pipeline_action(action_name: str) -> str:
    """Resolve registry tool name from a pipeline action (handles email:/calendar: prefixes)."""
    if action_name.startswith("email:") and action_name.count(":") >= 2:
        return action_name.split(":", 2)[2]
    if action_name.startswith("calendar:") and action_name.count(":") >= 2:
        return action_name.split(":", 2)[2]
    return action_name


def _user_facts_policy_blocks_tool_step(action_name: str, step: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    If a deterministic tool step must not run due to user_facts_policy, return user-facing error text; else None.
    """
    policy = _effective_user_facts_policy(step, metadata)
    core = _core_tool_name_from_pipeline_action(action_name)
    if core == "save_user_fact" and policy in ("vacuum", "no_write", "isolated"):
        return (
            "Saving user facts is blocked for this step "
            "(profile has include_user_facts off, or user_facts_policy is no_write/isolated)."
        )
    if core == "get_user_facts" and policy in ("vacuum", "isolated"):
        return (
            "Reading user facts is blocked for this step "
            "(profile has include_user_facts off, or user_facts_policy is isolated)."
        )
    return None


def _ensure_list(value: Any) -> List[str]:
    """Ensure value is a list of strings (e.g. tool names). Handles JSON strings like '[]' or '[\"tool\"]'."""
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


def _ensure_list_like(value: Any) -> List[Any]:
    """Ensure value is a list (elements unchanged). If it's a JSON string, parse it. Prevents list('[]') -> ['[', ']']."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return []
    return []


async def _build_augmented_tool_names(
    base_tool_names: List[str],
    step: Dict[str, Any],
    metadata: Optional[Dict[str, Any]],
    user_id: str,
    step_prompt: str = "",
) -> Tuple[List[str], str]:
    """
    Shared tool + skill resolution for llm_agent and deep_agent steps.
    Merges team skills, resolves step/team tool packs with mode, injects team_tools and member_additional_tools,
    expands email base and local_device_tools, filters local_proxy and user-fact tools.
    Returns (tool_names, skill_guidance).
    """
    step_skill_ids = list(step.get("skill_ids") or step.get("skills") or [])
    if metadata and metadata.get("team_skill_ids"):
        try:
            parsed = json.loads(metadata["team_skill_ids"])
            if isinstance(parsed, list):
                for s in parsed:
                    if s and s not in step_skill_ids:
                        step_skill_ids.append(s)
        except (TypeError, json.JSONDecodeError):
            pass
    skill_guidance, skill_required_tools = await _resolve_and_inject_skills(
        step_skill_ids=step_skill_ids,
        user_id=user_id,
        auto_discover_skills=bool(step.get("auto_discover_skills", True)),
        max_auto_skills=max(1, min(10, int(step.get("max_auto_skills") or 3))),
        step_prompt=step_prompt,
    )
    tool_names: List[str] = []
    seen: set = set()
    for t in base_tool_names:
        if t and t not in seen:
            seen.add(t)
            tool_names.append(t)
    for t in skill_required_tools:
        if t and t not in seen:
            seen.add(t)
            tool_names.append(t)
    pack_entries = _ensure_list_like(step.get("tool_packs"))
    if metadata and line_id_from_metadata(metadata):
        if not any(
            (e.get("pack") if isinstance(e, dict) else e) == "team_tools"
            for e in pack_entries
        ):
            pack_entries.append("team_tools")
    if metadata and metadata.get("team_tool_packs"):
        try:
            team_packs = json.loads(metadata["team_tool_packs"])
            if isinstance(team_packs, list):
                pack_entries.extend(p for p in team_packs if p)
        except (TypeError, json.JSONDecodeError):
            pass
    if pack_entries:
        try:
            from orchestrator.tools.tool_pack_registry import resolve_pack_tools_with_mode
            from orchestrator.tools.external_pack_registry import split_pack_entries, resolve_external_pack_tools

            builtin_entries, external_entries = split_pack_entries(pack_entries)
            for name in resolve_pack_tools_with_mode(builtin_entries):
                if name not in seen:
                    seen.add(name)
                    tool_names.append(name)
            for name in await resolve_external_pack_tools(external_entries, user_id):
                if name not in seen:
                    seen.add(name)
                    tool_names.append(name)
        except Exception as e:
            logger.warning("Resolving tool_packs in _build_augmented_tool_names failed: %s", e)
    if metadata and metadata.get("member_additional_tools"):
        try:
            parsed = json.loads(metadata["member_additional_tools"])
            if isinstance(parsed, list):
                for t in parsed:
                    if t and isinstance(t, str) and t not in seen:
                        seen.add(t)
                        tool_names.append(t)
        except (TypeError, json.JSONDecodeError):
            pass
    if any((n or "").startswith("email:") for n in tool_names):
        for base_name in ("list_emails", "search_emails", "list_email_folders", "read_email", "search_contacts"):
            if base_name not in seen:
                seen.add(base_name)
                tool_names.append(base_name)
    if any((n or "").startswith("calendar:") for n in tool_names):
        for base_name in (
            "list_calendars",
            "get_calendar_events",
            "get_event_by_id",
            "create_event",
            "update_event",
            "delete_event",
        ):
            if base_name not in seen:
                seen.add(base_name)
                tool_names.append(base_name)
    if "local_device_tools" in tool_names:
        tool_names = [n for n in tool_names if n != "local_device_tools"]
        try:
            from orchestrator.tools.local_proxy_tools import get_available_local_proxy_tools
            available_local = await get_available_local_proxy_tools(user_id)
            for t in available_local:
                if t not in seen:
                    seen.add(t)
                    tool_names.append(t)
        except Exception as e:
            logger.warning("Expanding local_device_tools in _build_augmented_tool_names failed: %s", e)
    try:
        from orchestrator.utils.action_io_registry import get_action
        from orchestrator.tools.local_proxy_tools import get_available_local_proxy_tools
        available_local = await get_available_local_proxy_tools(user_id)
        filtered = []
        for name in tool_names:
            contract = get_action(name)
            if contract is None and (name or "").endswith("_tool"):
                contract = get_action((name or "")[:-5])
            if contract and getattr(contract, "category", None) == "local_proxy":
                name_for_check = name if (name or "").endswith("_tool") else f"{name}_tool"
                if name_for_check in available_local:
                    filtered.append(name)
            else:
                filtered.append(name)
        tool_names = filtered
    except Exception as e:
        logger.warning("Filtering local_proxy in _build_augmented_tool_names failed: %s", e)
    policy = _effective_user_facts_policy(step, metadata)
    tool_names = _filter_user_fact_tools_by_policy(tool_names, policy)
    return (tool_names, skill_guidance or "")


async def _resolve_and_inject_skills(
    step_skill_ids: Optional[List[str]] = None,
    user_id: str = "system",
    auto_discover_skills: bool = False,
    max_auto_skills: int = 3,
    step_prompt: str = "",
) -> Tuple[str, List[str]]:
    """
    Resolve skill IDs for this step; optionally run semantic search (auto-discovery);
    fetch skill records via gRPC; return (guidance_text for system message, list of required_tools to merge into step).
    """
    from orchestrator.backend_tool_client import get_backend_tool_client
    merged = []
    seen = set()
    for sid in (step_skill_ids or []):
        s = (sid or "").strip()
        if s and s not in seen:
            seen.add(s)
            merged.append(s)
    if auto_discover_skills and (step_prompt or "").strip():
        try:
            client = await get_backend_tool_client()
            discovered = await client.search_skills(
                user_id=user_id,
                query=(step_prompt or "").strip(),
                limit=max(1, min(10, max_auto_skills)),
                score_threshold=0.5,
            )
            for hit in discovered:
                sid = (hit.get("id") or hit.get("skill_id") or "").strip()
                if sid and sid not in seen:
                    seen.add(sid)
                    merged.append(sid)
        except Exception as e:
            logger.warning("search_skills (auto-discovery) failed: %s", e)
    if not merged:
        return "", []
    try:
        client = await get_backend_tool_client()
        skills = await client.get_skills_by_ids(user_id, merged)
    except Exception as e:
        logger.warning("get_skills_by_ids failed: %s", e)
        return "", []
    parts = []
    required_tools = []
    for s in skills:
        name = s.get("name") or s.get("slug") or "Skill"
        procedure = (s.get("procedure") or "").strip()
        if procedure:
            parts.append(f"## Skill: {name}\n{procedure}")
        required_tools.extend(s.get("required_tools") or [])
    guidance = "\n\n".join(parts) if parts else ""
    required_deduped = list(dict.fromkeys(required_tools))
    return guidance, required_deduped


def _truncate_for_tool_trace(obj: Any, max_len: int = _TRACE_ARGS_MAX) -> Any:
    """Recursively truncate strings in a dict/list for tool call trace storage."""
    if isinstance(obj, str):
        return obj[:max_len] + ("..." if len(obj) > max_len else "")
    if isinstance(obj, dict):
        return {k: _truncate_for_tool_trace(v, max_len) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_for_tool_trace(v, max_len) for v in obj]
    return obj


# INFO logs include truncated args for these tools (agent line / delegation visibility).
_AGENT_LINE_TOOL_LOG_NAMES = frozenset(
    {
        "create_task_for_agent",
        "update_task_status",
        "check_my_tasks",
        "send_to_agent",
        "get_team_status_board",
        "read_team_timeline",
        "read_my_messages",
        "delegate_goal_to_tasks",
        "report_goal_progress",
        "escalate_task",
        "list_team_goals",
        "propose_hire",
        "propose_strategy_change",
    }
)


def _resolve_value(value: Any, playbook_state: Dict[str, Any], inputs: Dict[str, Any]) -> Any:
    """
    Resolve a single value. If it is a string containing {ref}, resolve ref from
    playbook_state (step_name.field) or inputs (var_name). Otherwise return as-is.
    """
    if not isinstance(value, str):
        return value
    match = _REF_PATTERN.fullmatch(value.strip())
    if not match:
        return value
    ref = match.group(1).strip()
    # Nested: step_name.field or step_name.nested.key
    if "." in ref:
        parts = ref.split(".")
        key = parts[0]
        if key in inputs:
            obj = inputs[key]
        else:
            obj = playbook_state.get(key)
        for part in parts[1:]:
            if obj is None:
                return None
            obj = obj.get(part) if isinstance(obj, dict) else getattr(obj, part, None)
        return obj
    # Wildcard ref: editor_refs_<prefix>* aggregates all keys with that prefix
    if ref.endswith("*"):
        return _resolve_wildcard_ref(ref, inputs, playbook_state)
    return playbook_state.get(ref, inputs.get(ref))


def _resolve_wildcard_ref(ref: str, inputs: Dict[str, Any], playbook_state: Dict[str, Any]) -> Optional[str]:
    """
    If ref ends with *, treat it as a prefix and return the concatenation of all
    input/playbook_state values whose keys start with that prefix. Otherwise return None.
    """
    if not ref.endswith("*"):
        return None
    prefix = ref[:-1]
    keys = sorted(
        set(k for k in inputs if k.startswith(prefix))
        | set(k for k in playbook_state if k.startswith(prefix))
    )
    parts = []
    for k in keys:
        v = inputs.get(k) if k in inputs else playbook_state.get(k)
        if v is not None and str(v).strip():
            parts.append(str(v))
    return "\n\n".join(parts)


# Dynamic ref: editor_refs_CATEGORY_section:Heading Name (resolved from full reference content)
_SECTION_REF_RE = re.compile(r"^editor_refs_(\w+)_section:(.+)$")


def _resolve_dynamic_section_ref(ref: str, inputs: Dict[str, Any]) -> Optional[str]:
    """Resolve {editor_refs_CATEGORY_section:Heading Name} from editor_refs_CATEGORY content."""
    m = _SECTION_REF_RE.match(ref.strip())
    if not m:
        return None
    category = m.group(1)
    heading_query = m.group(2).strip()
    full_content = inputs.get(f"editor_refs_{category}", "")
    if not full_content:
        return None
    from orchestrator.utils.section_scoping import extract_named_section

    result = extract_named_section(full_content, heading_query)
    return result if result else None


def _resolve_inputs(
    inputs_spec: Dict[str, Any],
    playbook_state: Dict[str, Any],
    inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve all input spec values against playbook_state and inputs."""
    resolved: Dict[str, Any] = {}
    for name, value in (inputs_spec or {}).items():
        if isinstance(value, str) and _REF_PATTERN.search(value):
            resolved[name] = _resolve_value(value, playbook_state, inputs)
        elif isinstance(value, dict):
            resolved[name] = _resolve_inputs(value, playbook_state, inputs)
        elif isinstance(value, list):
            resolved[name] = [
                _resolve_value(v, playbook_state, inputs) if isinstance(v, str) and _REF_PATTERN.search(str(v)) else v
                for v in value
            ]
        else:
            resolved[name] = value
    return resolved


def _coerce_for_input(value: Any, target_type: str) -> Any:
    """Coerce value to target_type (text, number, boolean, etc.) when possible."""
    if value is None:
        return None
    if target_type == "any" or target_type == "text":
        return value if target_type == "any" else str(value)
    if target_type == "number":
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                return int(value) if "." not in value else float(value)
            except ValueError:
                pass
    if target_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1")
        if isinstance(value, (int, float)):
            return value != 0
    return value


def _evaluate_condition(
    expr: str,
    playbook_state: Dict[str, Any],
    inputs: Dict[str, Any],
) -> bool:
    """
    Evaluate a condition expression against playbook_state and inputs.
    Resolves {ref} placeholders, then evaluates comparisons and compound AND/OR.
    No eval() - safe parsing only.
    Supported: >, <, ==, !=, >=, <=; "is defined", "is not defined"; "matches" (regex); AND, OR.
    """
    if not expr or not isinstance(expr, str):
        return True
    expr = expr.strip()
    if not expr:
        return True

    # Replace each {ref} with placeholder __Vn__ and store resolved value
    values: List[Any] = []
    def repl(match) -> str:
        ref = match.group(1).strip()
        val = _resolve_value("{" + ref + "}", playbook_state, inputs)
        idx = len(values)
        values.append(val)
        return f"__V{idx}__"

    normalized = _REF_PATTERN.sub(repl, expr)

    def get_val(s: str) -> Any:
        s = s.strip()
        m = re.match(r"__V(\d+)__", s)
        if m:
            idx = int(m.group(1))
            return values[idx] if idx < len(values) else None
        if s.startswith('"') and s.endswith('"') and len(s) >= 2:
            return s[1:-1].replace('\\"', '"')
        if s.startswith("'") and s.endswith("'") and len(s) >= 2:
            return s[1:-1].replace("\\'", "'")
        # Resolve bare variable names (e.g. editor_length) from inputs/playbook_state
        if s in inputs:
            return inputs[s]
        if s in playbook_state:
            return playbook_state[s]
        try:
            if "." in s:
                return float(s)
            return int(s)
        except ValueError:
            return s

    def eval_atom(atom: str) -> bool:
        atom = atom.strip()
        if not atom:
            return True
        # "X is defined" / "X is not defined"
        if atom.endswith(" is defined"):
            left = atom[:-len(" is defined")].strip()
            v = get_val(left)
            return v is not None
        if atom.endswith(" is not defined"):
            left = atom[:-len(" is not defined")].strip()
            v = get_val(left)
            return v is None
        # "X matches PATTERN" — case-insensitive regex
        if " matches " in atom:
            parts = atom.split(" matches ", 1)
            if len(parts) == 2:
                left_v = str(get_val(parts[0].strip()) or "")
                pattern = get_val(parts[1].strip())
                if isinstance(pattern, str):
                    pattern = pattern.strip('"').strip("'")
                else:
                    pattern = str(pattern or "")
                try:
                    return bool(re.search(pattern, left_v, re.IGNORECASE))
                except re.error:
                    return False
        # Comparison: L op R (op in ==, !=, >, <, >=, <=)
        for op in ("==", "!=", ">=", "<=", ">", "<"):
            if op in atom:
                parts = atom.split(op, 1)
                if len(parts) != 2:
                    continue
                left_s, right_s = parts[0].strip(), parts[1].strip()
                left_v = get_val(left_s)
                right_v = get_val(right_s)
                if op == "==":
                    return left_v == right_v
                if op == "!=":
                    return left_v != right_v
                try:
                    l, r = float(left_v) if left_v is not None else 0, float(right_v) if right_v is not None else 0
                except (TypeError, ValueError):
                    l, r = str(left_v), str(right_v)
                if op == ">":
                    return l > r
                if op == "<":
                    return l < r
                if op == ">=":
                    return l >= r
                if op == "<=":
                    return l <= r
        return False

    # Split by OR (lowest precedence), then AND
    or_parts = [p.strip() for p in re.split(r"\s+OR\s+", normalized, flags=re.IGNORECASE)]
    for or_part in or_parts:
        and_parts = [p.strip() for p in re.split(r"\s+AND\s+", or_part, flags=re.IGNORECASE)]
        and_ok = True
        for and_part in and_parts:
            if not eval_atom(and_part):
                and_ok = False
                break
        if and_ok:
            return True
    return False


def _get_input_types_for_action(action_name: str) -> Dict[str, str]:
    """Return input field name -> type from Action I/O Registry, or empty dict."""
    from orchestrator.utils.action_io_registry import get_action

    contract = get_action(action_name)
    if not contract:
        return {}
    types: Dict[str, str] = {}
    for f in contract.get_input_fields():
        name = f.get("name")
        if name:
            types[name] = f.get("type", "text")
    return types


def _get_llm_for_pipeline(metadata: Optional[Dict[str, Any]] = None):
    """Return ChatOpenAI for pipeline LLM steps, using user model and credentials if in metadata."""
    try:
        from langchain_openai import ChatOpenAI
        from config.settings import settings
    except ImportError:
        return None
    meta = metadata or {}
    model = meta.get("user_chat_model") or settings.DEFAULT_MODEL
    api_key = meta.get("user_llm_api_key") or settings.OPENROUTER_API_KEY or settings.OPENAI_API_KEY
    base_url = meta.get("user_llm_base_url") or settings.OPENROUTER_BASE_URL
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0.3,
    )


# Pattern for conditional blocks: {{#var}}...{{/var}} (var may include * for wildcard refs)
_COND_BLOCK_RE = re.compile(r"\{\{#([^}]+)\}\}(.*?)\{\{/\1\}\}", re.DOTALL)
# Pattern for raw blocks: content is passed through without variable resolution
_RAW_BLOCK_RE = re.compile(r"\{raw_block\}(.*?)\{/raw_block\}", re.DOTALL)


# Expression conditionals: {{#editor_length > 5000}}...{{/editor_length > 5000}}
_COND_EXPR_RE = re.compile(r"[<>]=?|!=|==")


def _apply_conditional_blocks(
    template: str, inputs: Dict[str, Any], playbook_state: Optional[Dict[str, Any]] = None
) -> str:
    """Strip {{#var}}...{{/var}} blocks when var is empty/falsy, or when expression is false.
    Supports: plain var (include if non-empty), wildcard refs (var ending with *), and
    expression conditionals (e.g. editor_length > 5000, editor_length < 5000)."""
    ps = playbook_state or {}

    def _replace(m):
        name = m.group(1).strip()
        # Expression conditional: e.g. editor_length > 5000
        if _COND_EXPR_RE.search(name):
            include = _evaluate_condition(name, ps, inputs)
            return m.group(2) if include else ""
        if name.endswith("*"):
            val = _resolve_wildcard_ref(name, inputs, ps)
        else:
            val = inputs.get(name) or ps.get(name)
        if not val and _SECTION_REF_RE.match(name):
            val = _resolve_dynamic_section_ref(name, inputs)
        return m.group(2) if val else ""

    return _COND_BLOCK_RE.sub(_replace, template)


def _resolve_prompt_template(
    template: str,
    playbook_state: Dict[str, Any],
    inputs: Dict[str, Any],
    step_inputs: Optional[Dict[str, Any]] = None,
) -> str:
    """Replace {ref} placeholders in template with values from playbook_state, inputs, and step_inputs (LLM task wiring)."""
    step_inputs = step_inputs or {}
    raw_blocks: Dict[str, str] = {}

    def _stash_raw(m):
        key = f"__RAW_{len(raw_blocks)}__"
        raw_blocks[key] = m.group(1)
        return key

    template = _RAW_BLOCK_RE.sub(_stash_raw, template)
    template = _apply_conditional_blocks(template, inputs, playbook_state)
    result = []
    i = 0
    while i < len(template):
        start = template.find("{", i)
        if start == -1:
            result.append(template[i:])
            break
        result.append(template[i:start])
        end = template.find("}", start + 1)
        if end == -1:
            result.append(template[start:])
            break
        ref = template[start + 1 : end].strip()
        if ref.startswith("literal:"):
            result.append("{" + ref[len("literal:") :].strip() + "}")
            i = end + 1
            continue
        if ref in step_inputs:
            raw = step_inputs[ref]
            if isinstance(raw, str) and _REF_PATTERN.fullmatch(raw.strip()):
                val = _resolve_value(raw, playbook_state, inputs)
            else:
                val = raw
        else:
            val = _resolve_value("{" + ref + "}", playbook_state, inputs)
        if val is None or str(val) == "{" + ref + "}":
            dynamic_val = _resolve_dynamic_section_ref(ref, inputs)
            if dynamic_val is not None:
                val = dynamic_val
        if val is not None and str(val) != "{" + ref + "}":
            result.append(str(val))
        else:
            result.append(template[start : end + 1])
        i = end + 1
    resolved = "".join(result)
    for key, content in raw_blocks.items():
        resolved = resolved.replace(key, content)
    return resolved


def _extract_usage_metadata(response: Any) -> Dict[str, int]:
    """Extract input/output token counts from LLM response (LangChain AIMessage.usage_metadata)."""
    meta = getattr(response, "usage_metadata", None)
    if not meta or not isinstance(meta, dict):
        return {"input_tokens": 0, "output_tokens": 0}
    result = {
        "input_tokens": int(meta.get("input_tokens") or meta.get("input_tokens_used") or meta.get("prompt_tokens") or 0),
        "output_tokens": int(meta.get("output_tokens") or meta.get("output_tokens_used") or meta.get("completion_tokens") or 0),
    }
    if meta and not result["input_tokens"] and not result["output_tokens"]:
        logger.debug("usage_metadata present but no tokens found. Keys: %s", list(meta.keys()))
    return result


def _log_prompt_variable_sizes(
    step_name: str,
    inputs: Dict[str, Any],
    playbook_state: Dict[str, Any],
    top_n: int = 10,
) -> None:
    """Log character counts per variable (inputs + playbook_state) used in prompt assembly. One compact INFO line."""
    sizes: List[tuple[str, int]] = []
    for d in (inputs or {}, playbook_state or {}):
        for k, v in d.items():
            if isinstance(v, str):
                sizes.append((k, len(v)))
    if not sizes:
        logger.info("Prompt variables (step=%s): 0 string vars", step_name)
        return
    sizes.sort(key=lambda x: -x[1])
    total = sum(n for _, n in sizes)
    top = sizes[:top_n]
    parts = [f"{k}={n:,}" for k, n in top]
    rest = len(sizes) - len(top)
    if rest > 0:
        parts.append(f"+{rest} more")
    logger.info(
        "Prompt variables (step=%s): %d vars, %s total chars | %s",
        step_name,
        len(sizes),
        f"{total:,}",
        " ".join(parts),
    )


def _deep_agent_step_prompt_for_skills(phases: List[Dict[str, Any]]) -> str:
    """Build a concatenated prompt from phase prompts/criteria for skill auto-discovery."""
    parts = []
    for p in (phases or [])[:5]:
        prompt = (p.get("prompt") or "").strip()
        if prompt:
            parts.append(prompt)
        criteria = (p.get("criteria") or "").strip()
        if criteria:
            parts.append(criteria)
    return "\n\n".join(parts) if parts else ""


def _format_workspace_schemas_for_prompt(
    workspace_schemas: Optional[List[Dict[str, Any]]],
    context_instructions: Optional[str],
) -> str:
    """Format workspace schema list and context instructions for injection into system prompt."""
    if not workspace_schemas and not (context_instructions and context_instructions.strip()):
        return ""
    parts = []
    if context_instructions and context_instructions.strip():
        parts.append("Data workspace context (follow these rules when querying):")
        parts.append(context_instructions.strip())
    if workspace_schemas:
        parts.append("\nData workspace schema(s) (use when generating SQL or answering data questions):")
        for i, schema in enumerate(workspace_schemas):
            ws_id = schema.get("workspace_id", "")
            tables = schema.get("tables", [])
            parts.append(f"\n--- Workspace: {ws_id} ({len(tables)} tables) ---")
            for table in tables:
                parts.append(f"Table: {table.get('name', '')} (ID: {table.get('table_id', '')})")
                if table.get("description"):
                    parts.append(f"  Description: {table['description']}")
                for col in table.get("columns", []):
                    desc = (col.get("description") or "").strip()
                    if desc:
                        parts.append(f"  - {col.get('name', '')} ({col.get('type', 'text')}): {desc}")
                    else:
                        parts.append(f"  - {col.get('name', '')} ({col.get('type', 'text')})")
    return "\n".join(parts) if parts else ""


def _build_system_message(
    metadata: Optional[Dict[str, Any]], step: Optional[Dict[str, Any]] = None
) -> Optional[SystemMessage]:
    """Build a SystemMessage from persona, system_prompt_additions, and user context."""
    parts = []
    facts_policy = _effective_user_facts_policy(step, metadata)

    if metadata and metadata.get("persona_enabled"):
        persona = metadata.get("persona") or {}
        ai_name = persona.get("ai_name") or "Alex"
        political_bias = persona.get("political_bias") or "neutral"
        timezone = persona.get("timezone") or "UTC"
        custom_prefs = persona.get("custom_preferences") or {}
        style_instruction = custom_prefs.get("style_instruction") or persona.get("style_instruction")
        if not style_instruction:
            style = persona.get("persona_style") or "professional"
            style_instruction = f"Respond in a {style} style."
        persona_parts = [f"Your name is {ai_name}.", style_instruction]
        if political_bias and political_bias != "neutral":
            persona_parts.append(f"Political leaning: {political_bias}.")
        if timezone and timezone != "UTC":
            persona_parts.append(f"The user's timezone is {timezone}.")
        parts.append(" ".join(persona_parts))

    additions = (metadata or {}).get("system_prompt_additions") or ""
    if additions.strip():
        parts.append(additions.strip())

    user_context = (metadata or {}).get("user_context_str") or ""
    if user_context.strip():
        parts.append(user_context.strip())

    user_facts = (metadata or {}).get("user_facts_str") or ""
    if user_facts.strip() and facts_policy not in ("vacuum", "isolated"):
        parts.append(user_facts.strip())

    agent_memory = (metadata or {}).get("agent_memory_str") or ""
    if agent_memory.strip():
        parts.append(agent_memory.strip())

    goal_context = (metadata or {}).get("goal_context_str") or ""
    if goal_context.strip():
        parts.append(goal_context.strip())

    if not parts:
        return None
    return SystemMessage(content="\n\n".join(parts))


async def _execute_llm_step(
    step: Dict[str, Any],
    playbook_state: Dict[str, Any],
    inputs: Dict[str, Any],
    user_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute an llm_task step: build prompt from template + resolved refs, call LLM, return dict with formatted + parsed.
    """
    llm = _get_llm_for_pipeline(metadata)
    if not llm:
        logger.warning("LLM not available for llm_task step")
        return {"_error": "LLM not available", "formatted": "LLM not configured for pipeline."}
    template = step.get("prompt_template") or step.get("prompt") or "Please analyze: {query}"
    step_inputs = step.get("inputs") or {}
    prompt = _resolve_prompt_template(template, playbook_state, inputs, step_inputs=step_inputs)
    llm_messages: List[Any] = []
    system_msg = _build_system_message(metadata, step)
    if system_msg:
        llm_messages.append(system_msg)
    meta = metadata or {}
    skill_guidance, _ = await _resolve_and_inject_skills(
        step_skill_ids=step.get("skill_ids") or step.get("skills"),
        user_id=user_id,
        auto_discover_skills=bool(step.get("auto_discover_skills", True)),
        max_auto_skills=max(1, min(10, int(step.get("max_auto_skills") or 3))),
        step_prompt=prompt,
    )
    if skill_guidance:
        skill_msg = SystemMessage(content=skill_guidance)
        llm_messages.append(skill_msg)
    if meta.get("include_datetime_context", True) and meta.get("datetime_context_str"):
        llm_messages.append(SystemMessage(content=meta["datetime_context_str"]))
    llm_messages.append(HumanMessage(content=prompt))
    try:
        if hasattr(llm, "ainvoke"):
            response = await llm.ainvoke(llm_messages)
        else:
            response = llm.invoke(llm_messages)
        content = (getattr(response, "content", "") or "").strip()
        token_usage = _extract_usage_metadata(response)
    except Exception as e:
        logger.exception("LLM step failed: %s", e)
        return {"_error": str(e), "formatted": f"LLM error: {str(e)}"}
    output_json = step.get("output_schema") or step.get("json_output", False)
    if output_json and content.strip():
        schema = output_json if isinstance(output_json, dict) else None
        parsed = StructuredOutputParser.parse(content, schema=schema)
        if parsed is not None:
            parsed.setdefault("formatted", content)
            parsed["_token_usage"] = token_usage
            return parsed
    return {"formatted": content, "raw": content, "_token_usage": token_usage}


async def _execute_deep_agent_step(
    step: Dict[str, Any],
    playbook_state: Dict[str, Any],
    inputs: Dict[str, Any],
    user_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute a deep_agent step: build LangGraph from phases, run it, return formatted + phase_trace.
    Step-level available_tools and skill_ids are merged with phase-level tools; skill procedure is injected into system message.
    """
    from orchestrator.engines.deep_agent_executor import run_deep_agent

    phases = step.get("phases") or []
    if not isinstance(phases, list) or not phases:
        return {"_error": "No phases", "formatted": "Deep agent step has no phases.", "phase_trace": []}

    step_tools = _ensure_list(step.get("available_tools"))
    phase_tools: List[str] = []
    for p in phases:
        phase_tools.extend(_ensure_list(p.get("available_tools")))
        phase_tools.extend(_ensure_list(p.get("search_tools")))
    base_tool_names = step_tools + phase_tools
    tool_names, skill_guidance = await _build_augmented_tool_names(
        base_tool_names,
        step,
        metadata,
        user_id,
        step_prompt=_deep_agent_step_prompt_for_skills(phases),
    )
    resolved_tools = await _resolve_llm_agent_tools(tool_names, user_id=user_id)
    tools_map: Dict[str, Tuple[Any, Any]] = {}
    for name, func, contract, _ in resolved_tools:
        tools_map[name] = (func, contract)

    def _resolve_fn(template: str, phase_results: Dict[str, Any]) -> str:
        merged_ps = dict(playbook_state)
        for k, v in (phase_results or {}).items():
            merged_ps[k] = v
        return _resolve_prompt_template(template, merged_ps, inputs)

    llm = _get_llm_for_pipeline(metadata)
    if not llm:
        logger.warning("LLM not available for deep_agent step")
        return {"_error": "LLM not available", "formatted": "LLM not configured.", "phase_trace": []}
    model_override = step.get("model_override")
    if model_override:
        try:
            from langchain_openai import ChatOpenAI
            from orchestrator.utils.llm_credentials_from_metadata import get_openrouter_credentials
            api_key, base_url = get_openrouter_credentials(metadata)
            llm = ChatOpenAI(
                model=model_override,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=0.3,
            )
        except Exception as e:
            logger.warning("Model override failed for deep_agent step: %s", e)
    system_msg = _build_system_message(metadata, step)
    if skill_guidance:
        if system_msg and hasattr(system_msg, "content"):
            system_msg = SystemMessage(content=(system_msg.content or "").strip() + "\n\n" + skill_guidance)
        else:
            system_msg = SystemMessage(content=skill_guidance)

    result = await run_deep_agent(
        phases=phases,
        resolve_fn=_resolve_fn,
        llm=llm,
        tools_map=tools_map,
        playbook_state=playbook_state,
        inputs=inputs,
        user_id=user_id,
        metadata=metadata,
        execute_llm_agent_step_fn=_execute_llm_agent_step,
        system_msg=system_msg,
        step_palette_tools=tool_names,
        parent_step_for_policy=step,
    )
    return result


async def _resolve_llm_agent_tools(
    tool_names: List[str],
    user_id: str = "system",
) -> List[Tuple[str, Any, Optional[Any], Optional[str]]]:
    """Resolve tool names to (name, func, contract, description_override) via Action I/O Registry or orchestrator.tools.
    Handles email:<connection_id>:<tool_name>, agent:<profile_id>[:playbook_id], and playbook:<playbook_id> by wrapping with fixed params.
    When description_override is set, the LLM sees that instead of the contract's generic description."""
    from orchestrator.utils.action_io_registry import get_action

    try:
        import orchestrator.tools as tools_module
    except ImportError:
        tools_module = None
    resolved: List[Tuple[str, Any, Optional[Any], Optional[str]]] = []
    for name in (tool_names or []):
        if name.startswith("email:") and name.count(":") >= 2:
            parts = name.split(":", 2)
            try:
                connection_id = int(parts[1])
                tool_name = parts[2]
            except (ValueError, IndexError):
                continue
            contract = get_action(tool_name)
            if contract and callable(getattr(contract, "tool_function", None)):
                base_fn = contract.tool_function
                base_sig = inspect.signature(base_fn)

                def _make_email_wrapper(base_f: Any, cid: int) -> Any:
                    async def _async_wrapper(**kwargs: Any) -> Any:
                        kwargs["connection_id"] = cid
                        return await base_f(**kwargs)

                    def _sync_wrapper(**kwargs: Any) -> Any:
                        kwargs["connection_id"] = cid
                        return base_f(**kwargs)

                    w = _async_wrapper if asyncio.iscoroutinefunction(base_f) else _sync_wrapper
                    w.__signature__ = base_sig
                    return w

                resolved.append((name, _make_email_wrapper(base_fn, connection_id), contract, None))
            continue
        if name.startswith("calendar:") and name.count(":") >= 2:
            parts = name.split(":", 2)
            try:
                connection_id = int(parts[1])
                tool_name = parts[2]
            except (ValueError, IndexError):
                continue
            contract = get_action(tool_name)
            if contract and callable(getattr(contract, "tool_function", None)):
                base_fn = contract.tool_function
                base_sig = inspect.signature(base_fn)

                def _make_calendar_wrapper(base_f: Any, cid: int) -> Any:
                    async def _async_wrapper(**kwargs: Any) -> Any:
                        kwargs["connection_id"] = cid
                        return await base_f(**kwargs)

                    def _sync_wrapper(**kwargs: Any) -> Any:
                        kwargs["connection_id"] = cid
                        return base_f(**kwargs)

                    w = _async_wrapper if asyncio.iscoroutinefunction(base_f) else _sync_wrapper
                    w.__signature__ = base_sig
                    return w

                resolved.append((name, _make_calendar_wrapper(base_fn, connection_id), contract, None))
            continue
        if name.startswith("mcp:") and name.count(":") >= 2:
            parts = name.split(":", 2)
            try:
                server_id = int(parts[1])
                mcp_tool = parts[2]
            except (ValueError, IndexError):
                continue
            if not mcp_tool:
                continue

            def _make_mcp_llm_wrapper(sid: int, tname: str) -> Any:
                from orchestrator.tools.mcp_tools import run_mcp_tool_invocation

                async def _mcp_bound_wrapper(
                    arguments_json: str = "{}",
                    user_id: str = "system",
                ) -> Any:
                    try:
                        args = json.loads(arguments_json or "{}")
                        if not isinstance(args, dict):
                            args = {}
                    except json.JSONDecodeError:
                        args = {}
                    return await run_mcp_tool_invocation(user_id, sid, tname, args)

                _mcp_bound_wrapper.__doc__ = (
                    f"MCP tool `{tname}` on server id {sid}. "
                    "Pass tool parameters as a JSON object in `arguments_json`."
                )
                return _mcp_bound_wrapper

            wrapper_fn = _make_mcp_llm_wrapper(server_id, mcp_tool)
            resolved.append(
                (
                    name,
                    wrapper_fn,
                    None,
                    f"MCP server tool `{mcp_tool}` (server id {server_id}). "
                    "Provide parameters as JSON in arguments_json.",
                )
            )
            continue
        if name.startswith("playbook:") and name.count(":") >= 1:
            playbook_id = name.split(":", 1)[1].strip()
            if not playbook_id:
                continue
            contract = get_action("invoke_playbook")
            if contract and callable(getattr(contract, "tool_function", None)):
                base_fn = contract.tool_function
                try:
                    from orchestrator.backend_tool_client import get_backend_tool_client
                    client = await get_backend_tool_client()
                    playbook = await client.get_playbook(user_id, playbook_id)
                    pb_name = (playbook.get("name") or playbook_id) if playbook else playbook_id
                    pb_desc = (playbook.get("description") or "").strip() if playbook else ""
                except Exception:
                    pb_name = playbook_id
                    pb_desc = ""

                def _make_playbook_scoped_wrapper(base_f: Any, pbid: str) -> Any:
                    async def _async_wrapper(**kwargs: Any) -> Any:
                        kwargs["playbook_id"] = pbid
                        return await base_f(**kwargs)
                    return _async_wrapper

                wrapper = _make_playbook_scoped_wrapper(base_fn, playbook_id)
                display_name = f"run_playbook_{re.sub(r'[^a-zA-Z0-9_-]', '_', pb_name)[:48]}"
                description = f"[{pb_name}] {pb_desc}" if pb_desc else f"Run playbook: {pb_name}"
                output_fields = contract.get_output_fields() if hasattr(contract, "get_output_fields") else []
                if output_fields:
                    returns = ", ".join(f"{f['name']} ({f['type']})" for f in output_fields[:5])
                    description = f"{description}\nReturns: {returns}".strip()
                resolved.append((display_name, wrapper, contract, description))
            continue
        if name.startswith("agent:") and name.count(":") >= 1:
            parts = name.split(":", 2)
            profile_id = (parts[1] or "").strip()
            playbook_id = (parts[2] or "").strip() if len(parts) >= 3 else None
            if not profile_id:
                continue
            contract = get_action("invoke_agent")
            if contract and callable(getattr(contract, "tool_function", None)):
                base_fn = contract.tool_function
                try:
                    from orchestrator.backend_tool_client import get_backend_tool_client
                    client = await get_backend_tool_client()
                    profile = await client.get_agent_profile(user_id, profile_id)
                    agent_name = (profile.get("name") or profile_id) if profile else profile_id
                    agent_desc = (profile.get("description") or "").strip() if profile else ""
                except Exception:
                    agent_name = profile_id
                    agent_desc = ""

                def _make_agent_scoped_wrapper(
                    base_f: Any, pid: str, pbid: Optional[str]
                ) -> Any:
                    async def _async_wrapper(**kwargs: Any) -> Any:
                        kwargs["agent_profile_id"] = pid
                        kwargs["playbook_id"] = pbid
                        kwargs["agent_handle"] = kwargs.get("agent_handle") or ""
                        return await base_f(**kwargs)
                    return _async_wrapper

                wrapper = _make_agent_scoped_wrapper(base_fn, profile_id, playbook_id)
                display_name = f"invoke_agent_{re.sub(r'[^a-zA-Z0-9_-]', '_', agent_name)[:48]}" + (
                    f"_{re.sub(r'[^a-zA-Z0-9_-]', '_', playbook_id)[:16]}" if playbook_id else ""
                )
                description = f"[{agent_name}] {agent_desc}" if agent_desc else f"Invoke agent: {agent_name}"
                output_fields = contract.get_output_fields() if hasattr(contract, "get_output_fields") else []
                if output_fields:
                    returns = ", ".join(f"{f['name']} ({f['type']})" for f in output_fields[:5])
                    description = f"{description}\nReturns: {returns}".strip()
                resolved.append((display_name, wrapper, contract, description))
            continue
        contract = get_action(name)
        if not contract and name.endswith("_tool"):
            registry_name = name[:-5]
            contract = get_action(registry_name)
            if contract and callable(getattr(contract, "tool_function", None)):
                resolved.append((registry_name, contract.tool_function, contract, None))
                continue
        if contract and callable(getattr(contract, "tool_function", None)):
            resolved.append((name, contract.tool_function, contract, None))
            continue
        if tools_module:
            fn = getattr(tools_module, name, None)
            if callable(fn):
                resolved.append((name, fn, None, None))
    return resolved


def _normalize_tool_name(name: str) -> str:
    """Normalize tool name for lookup (models may strip underscores)."""
    return name.replace("_", "").replace("-", "").lower()


def _sanitize_tool_name_for_llm(name: str) -> str:
    """Sanitize tool name for LLM API (e.g. Anthropic: ^[a-zA-Z0-9_-]{1,128}$).
    For email:N:tool_name patterns, present as tool_name_N so the LLM sees a clean name
    (connection_id is pre-wired by _resolve_llm_agent_tools)."""
    if name.startswith("email:") and name.count(":") >= 2:
        parts = name.split(":", 2)
        connection_id = parts[1]
        tool_name = parts[2]
        base = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_name)[:100]
        return f"{base}_{connection_id}"[:128]
    if name.startswith("calendar:") and name.count(":") >= 2:
        parts = name.split(":", 2)
        connection_id = parts[1]
        tool_name = parts[2]
        base = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_name)[:100]
        return f"{base}_{connection_id}"[:128]
    if name.startswith("mcp:") and name.count(":") >= 2:
        parts = name.split(":", 2)
        server_id = parts[1]
        tool_name = parts[2]
        base = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_name)[:80]
        return f"mcp_{base}_{server_id}"[:128]
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return sanitized[:128] if len(sanitized) > 128 else sanitized


def _build_dynamic_args_model(name: str, func: Any) -> Type[BaseModel]:
    """Build args schema from function signature (no field descriptions). Fallback when no registry contract."""
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}
    fields = {}
    for pname, param in inspect.signature(func).parameters.items():
        if pname in ("user_id", "_editor_content", "_pipeline_metadata"):
            continue
        ann = hints.get(pname, str)
        default = param.default if param.default != inspect.Parameter.empty else ...
        fields[pname] = (ann, default)
    schema_name = (name.replace("-", "_") + "_schema").replace(".", "_")
    return create_model(schema_name, **fields)


def _merge_input_params_models(
    inputs_model: Type[BaseModel],
    params_model: Type[BaseModel],
    name: str,
) -> Type[BaseModel]:
    """Merge inputs_model and params_model into one Pydantic model, preserving field descriptions."""
    fields = {}
    for model in (inputs_model, params_model):
        for fname, finfo in model.model_fields.items():
            if fname in fields:
                continue
            default = finfo.default if not finfo.is_required() else ...
            desc = getattr(finfo, "description", None) or ""
            fields[fname] = (finfo.annotation, Field(default=default, description=desc))
    schema_name = (name.replace("-", "_") + "_merged_schema").replace(".", "_")
    return create_model(schema_name, **fields)


def _build_rich_tool_description(
    name: str, func: Any, contract: Optional[Any]
) -> Tuple[str, Type[BaseModel]]:
    """Build description and args_schema from registry contract or function signature."""
    from orchestrator.utils.action_io_registry import ActionContract

    if contract and isinstance(contract, ActionContract):
        desc = contract.description or ""
        output_fields = contract.get_output_fields()
        if output_fields:
            returns = ", ".join(f"{f['name']} ({f['type']})" for f in output_fields[:5])
            desc = f"{desc}\nReturns: {returns}".strip() if desc else f"Returns: {returns}"
        if contract.inputs_model:
            if contract.params_model:
                args_model = _merge_input_params_models(
                    contract.inputs_model, contract.params_model, name
                )
            else:
                args_model = contract.inputs_model
        else:
            args_model = _build_dynamic_args_model(name, func)
    else:
        desc = (getattr(func, "__doc__") or name).strip()
        args_model = _build_dynamic_args_model(name, func)
    return desc, args_model


def _wrap_tool_for_llm_agent(
    name: str,
    func: Any,
    user_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    contract: Optional[Any] = None,
    description_override: Optional[str] = None,
) -> Any:
    """Wrap a tool function as StructuredTool for bind_tools; inject user_id and return formatted."""
    from langchain_core.tools import StructuredTool

    sig = inspect.signature(func)
    if description_override is not None:
        from orchestrator.utils.action_io_registry import ActionContract
        desc = description_override
        if contract and isinstance(contract, ActionContract):
            output_fields = contract.get_output_fields()
            if output_fields:
                returns = ", ".join(f"{f['name']} ({f['type']})" for f in output_fields[:5])
                if "Returns:" not in desc:
                    desc = f"{desc}\nReturns: {returns}".strip()
        description = desc
        _, ArgsModel = _build_rich_tool_description(name, func, contract)
    else:
        description, ArgsModel = _build_rich_tool_description(name, func, contract)

    async def _run(**kwargs: Any) -> str:
        if "user_id" in sig.parameters:
            kwargs["user_id"] = user_id
        if "_pipeline_metadata" in sig.parameters:
            kwargs["_pipeline_metadata"] = metadata or {}
        # Auto-fill workspace_id for data workspace tools when profile has a single bound workspace
        norm_name = (name or "").replace("-", "_").replace(" ", "_").lower()
        if not kwargs.get("workspace_id"):
            if "query_data_workspace" in norm_name or "get_workspace_schema" in norm_name:
                ws_ids = (metadata or {}).get("workspace_ids")
                if isinstance(ws_ids, list) and len(ws_ids) == 1:
                    kwargs["workspace_id"] = ws_ids[0]
        ws_id_kw = kwargs.get("workspace_id")
        if ws_id_kw and "query_data_workspace" in norm_name:
            modes = (metadata or {}).get("workspace_access_modes") or {}
            if isinstance(modes, dict) and modes.get(str(ws_id_kw)) == "read":
                kwargs["read_only"] = True
        if "_editor_content" in sig.parameters:
            active_editor = (metadata or {}).get("shared_memory", {}).get("active_editor", {})
            kwargs["_editor_content"] = (active_editor or {}).get("content", "")
        from orchestrator.utils.action_io_registry import ActionContract
        from orchestrator.middleware.tool_retry_node import ToolRetryNode, TRANSIENT_EXCEPTIONS

        async def _invoke_wrapped() -> Any:
            if asyncio.iscoroutinefunction(func):
                return await func(**kwargs)
            return func(**kwargs)

        try:
            if contract is None:
                use_retry = True
            elif isinstance(contract, ActionContract):
                use_retry = contract.retriable
            else:
                use_retry = True
            if use_retry:
                retry_er = ToolRetryNode(
                    max_retries=2,
                    initial_delay=0.5,
                    retryable_exceptions=TRANSIENT_EXCEPTIONS,
                )
                out = await retry_er.execute_with_retry(_invoke_wrapped)
            else:
                out = await _invoke_wrapped()
            if isinstance(out, dict) and "formatted" in out:
                return out["formatted"]
            return str(out) if out is not None else ""
        except Exception as e:
            return f"Error: {e}"

    return StructuredTool(
        name=name,
        description=description,
        coroutine=_run,
        args_schema=ArgsModel,
    )


async def _execute_llm_agent_step(
    step: Dict[str, Any],
    playbook_state: Dict[str, Any],
    inputs: Dict[str, Any],
    user_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute an llm_agent step: bind available_tools to LLM, run ReAct loop, return formatted + typed.
    """
    # Ensure we use real user_id from metadata when playbook state has "system" (e.g. device tools like local_screenshot)
    if not user_id or user_id == "system":
        user_id = (metadata or {}).get("user_id", "system")

    llm = _get_llm_for_pipeline(metadata)
    if not llm:
        logger.warning("LLM not available for llm_agent step")
        return {"_error": "LLM not available", "formatted": "LLM not configured for pipeline."}
    model_override = step.get("model_override")
    if model_override:
        try:
            from langchain_openai import ChatOpenAI
            from orchestrator.utils.llm_credentials_from_metadata import get_openrouter_credentials
            api_key, base_url = get_openrouter_credentials(metadata)
            llm = ChatOpenAI(
                model=model_override,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=0.3,
            )
        except Exception as e:
            logger.warning("Model override failed for llm_agent step: %s", e)

    step_name = step.get("name") or step.get("output_key") or "llm_agent"
    template = step.get("prompt_template") or step.get("prompt") or "Use the available tools to: {query}"
    step_inputs = step.get("inputs") or {}
    _log_prompt_variable_sizes(step_name, inputs, playbook_state)
    prompt = _resolve_prompt_template(template, playbook_state, inputs, step_inputs=step_inputs)
    logger.info("Resolved prompt (step=%s): %s chars", step_name, f"{len(prompt):,}")
    tool_names, skill_guidance = await _build_augmented_tool_names(
        _ensure_list(step.get("available_tools")),
        step,
        metadata,
        user_id,
        step_prompt=prompt,
    )
    resolved_tools = await _resolve_llm_agent_tools(tool_names, user_id=user_id)
    logger.info("LLM agent step: step=%s tools=%s", step_name, tool_names)
    raw_max = step.get("max_iterations", 3)
    try:
        max_iterations = max(1, min(25, int(raw_max) if raw_max is not None else 3))
    except (TypeError, ValueError):
        max_iterations = 3
        logger.warning(
            "LLM agent step %s: max_iterations=%r invalid, using 3",
            step_name,
            raw_max,
        )
    else:
        logger.info("LLM agent step %s: max_iterations=%s (raw=%r)", step_name, max_iterations, raw_max)

    # Use sanitized names for bind_tools (provider requirement: ^[a-zA-Z0-9_-]{1,128}$)
    wrapped_tools = [
        _wrap_tool_for_llm_agent(_sanitize_tool_name_for_llm(name), func, user_id, metadata, contract, description_override)
        for name, func, contract, description_override in resolved_tools
    ]
    # Map normalized(sanitized_name) -> (original_name, func) for tool_calls lookup and executed_tool_names
    tool_map = {
        _normalize_tool_name(_sanitize_tool_name_for_llm(name)): (name, f, c)
        for name, f, c, _ in resolved_tools
    }

    messages: List[Any] = []
    system_msg = _build_system_message(metadata, step)
    if skill_guidance:
        if system_msg and hasattr(system_msg, "content"):
            system_msg = SystemMessage(content=(system_msg.content or "").strip() + "\n\n" + skill_guidance)
        else:
            system_msg = SystemMessage(content=skill_guidance)
    workspace_schema_text = _format_workspace_schemas_for_prompt(
        playbook_state.get("workspace_schemas"),
        playbook_state.get("workspace_context_instructions"),
    )
    if workspace_schema_text:
        if system_msg and hasattr(system_msg, "content"):
            system_msg = SystemMessage(content=(system_msg.content or "").strip() + "\n\n" + workspace_schema_text)
        else:
            system_msg = SystemMessage(content=workspace_schema_text)
    if system_msg:
        messages.append(system_msg)
    messages.append(HumanMessage(content=prompt))
    _msg_content = lambda m: (getattr(m, "content", None) or "") if hasattr(m, "content") else str(m)
    _total_msg_chars = sum(len(_msg_content(m)) for m in messages)
    logger.info(
        "Assembled messages (step=%s): %d parts, %s total chars (~%s tokens)",
        step_name,
        len(messages),
        f"{_total_msg_chars:,}",
        f"{_total_msg_chars // 4:,}",
    )
    content = ""
    output_json = step.get("output_schema") or step.get("json_output", False)
    executed_tool_names: List[str] = []
    tool_call_trace: List[Dict[str, Any]] = []
    extracted_images_from_tools: List[Dict[str, Any]] = []
    extracted_proposals: List[Dict[str, Any]] = []
    skill_acquisitions = 0
    max_skill_acquisitions = int(step.get("max_skill_acquisitions", 2))
    token_acc: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    for iteration in range(max_iterations):
        if wrapped_tools:
            bound_llm = llm.bind_tools(wrapped_tools)
            try:
                response = await bound_llm.ainvoke(messages)
            except Exception as e:
                if OpenAIBadRequestError and isinstance(e, OpenAIBadRequestError):
                    err_str = str(e)
                    if "prompt is too long" in err_str or "maximum context length" in err_str:
                        return {
                            "formatted": "The retrieved documents were too large for the model's context window. "
                                          "Try asking a more specific question so I can retrieve only the relevant sections.",
                            "_error": "context_length_exceeded",
                            "_tool_call_trace": tool_call_trace,
                            "_token_usage": token_acc,
                        }
                raise
        else:
            try:
                response = await llm.ainvoke(messages)
            except Exception as e:
                if OpenAIBadRequestError and isinstance(e, OpenAIBadRequestError):
                    err_str = str(e)
                    if "prompt is too long" in err_str or "maximum context length" in err_str:
                        return {
                            "formatted": "The retrieved documents were too large for the model's context window. "
                                          "Try asking a more specific question so I can retrieve only the relevant sections.",
                            "_error": "context_length_exceeded",
                            "_tool_call_trace": tool_call_trace,
                            "_token_usage": token_acc,
                        }
                raise
        u = _extract_usage_metadata(response)
        token_acc["input_tokens"] = token_acc.get("input_tokens", 0) + u.get("input_tokens", 0)
        token_acc["output_tokens"] = token_acc.get("output_tokens", 0) + u.get("output_tokens", 0)
        content = (getattr(response, "content", "") or "").strip()
        tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls:
            break

        tool_messages = []
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {}) or {}
            tool_id = tc.get("id", "")
            args = dict(tool_args)
            if "kwargs" in args and isinstance(args["kwargs"], dict) and len(args) <= 2:
                args = dict(args["kwargs"])
            match = tool_map.get(_normalize_tool_name(tool_name))
            original_name, tool_func, tool_contract = match if match else (None, None, None)
            display_name = original_name or tool_name

            tc_started = datetime.now(timezone.utc).isoformat()
            result_str = ""
            tc_status = "completed"
            tc_error: Optional[str] = None

            if tool_func and callable(tool_func):
                try:
                    sig = inspect.signature(tool_func)
                    if "user_id" not in args and "user_id" in sig.parameters:
                        args["user_id"] = user_id
                    if "_pipeline_metadata" in sig.parameters:
                        args["_pipeline_metadata"] = metadata or {}
                    safe_args = {k: v for k, v in args.items() if k not in ("user_id", "_pipeline_metadata")}
                    if display_name in _AGENT_LINE_TOOL_LOG_NAMES:
                        logger.info(
                            "Tool call: name=%s args=%s",
                            display_name,
                            _truncate_for_tool_trace(safe_args, max_len=400),
                        )
                    else:
                        logger.info("Tool call: name=%s", display_name)
                    from orchestrator.utils.action_io_registry import ActionContract
                    from orchestrator.middleware.tool_retry_node import ToolRetryNode, TRANSIENT_EXCEPTIONS

                    if tool_contract is None:
                        use_retry = True
                    elif isinstance(tool_contract, ActionContract):
                        use_retry = tool_contract.retriable
                    else:
                        use_retry = True

                    async def _invoke_raw_tool() -> Any:
                        if asyncio.iscoroutinefunction(tool_func):
                            return await tool_func(**args)
                        return tool_func(**args)

                    if use_retry:
                        retry_er = ToolRetryNode(
                            max_retries=2,
                            initial_delay=0.5,
                            retryable_exceptions=TRANSIENT_EXCEPTIONS,
                        )
                        result = await retry_er.execute_with_retry(_invoke_raw_tool)
                    else:
                        result = await _invoke_raw_tool()
                    if isinstance(result, dict) and result.get("_needs_human_interaction"):
                        return {
                            "_interaction_required": True,
                            "interaction_type": result.get("interaction_type", "browser_login"),
                            "interaction_data": result.get("interaction_data", {}),
                            "session_id": result.get("session_id"),
                            "site_domain": result.get("site_domain"),
                            "formatted": result.get("formatted", "Authentication required."),
                            "_token_usage": token_acc,
                        }
                    if original_name:
                        executed_tool_names.append(original_name)
                    result_str = result.get("formatted", str(result)) if isinstance(result, dict) else str(result or "")
                    # If tool returned an image (e.g. local_screenshot), don't send huge base64 to LLM;
                    # use a short placeholder and collect the image for the final response.
                    if isinstance(result, dict) and result.get("image_data_uri"):
                        width = result.get("width") or 0
                        height = result.get("height") or 0
                        extracted_images_from_tools.append({
                            "url": result["image_data_uri"],
                            "alt_text": "Screenshot",
                            "type": "screenshot",
                            "metadata": {"width": width, "height": height} if (width and height) else {},
                        })
                        result_str = f"Screenshot captured ({width}x{height})." if (width and height) else "Screenshot captured."
                    # Collect structured images from image search (and similar tools) for final response
                    if isinstance(result, dict) and result.get("images") and isinstance(result["images"], list):
                        for img in result["images"]:
                            if isinstance(img, dict) and img.get("url"):
                                extracted_images_from_tools.append(img)
                    # Collect editor proposals from patch_file (and similar) for approval UI
                    if isinstance(result, dict) and result.get("proposal_id"):
                        extracted_proposals.append({
                            "proposal_id": result["proposal_id"],
                            "document_id": result.get("document_id", ""),
                            "operations_applied": result.get("operations_applied", 0),
                            "summary": result.get("formatted", ""),
                        })
                    # Mid-loop skill acquisition: merge new tools and guidance for next iteration
                    if (
                        isinstance(result, dict)
                        and result.get("_acquire_skills")
                        and skill_acquisitions < max_skill_acquisitions
                    ):
                        new_tool_names = result.get("_acquired_tools") or []
                        new_guidance = result.get("_skill_guidance") or ""
                        actually_added = []
                        for tn in new_tool_names:
                            if tn not in tool_names:
                                tool_names.append(tn)
                                actually_added.append(tn)
                        if actually_added:
                            new_resolved = await _resolve_llm_agent_tools(actually_added, user_id=user_id)
                            for rname, rfunc, rcontract, rdesc in new_resolved:
                                san = _sanitize_tool_name_for_llm(rname)
                                wrapped_tools.append(
                                    _wrap_tool_for_llm_agent(san, rfunc, user_id, metadata, rcontract, rdesc)
                                )
                                tool_map[_normalize_tool_name(san)] = (rname, rfunc, rcontract)
                        if new_guidance:
                            messages.append(SystemMessage(content=new_guidance))
                        skill_acquisitions += 1
                        logger.info(
                            "Skill acquisition %d/%d: added %d tools",
                            skill_acquisitions,
                            max_skill_acquisitions,
                            len(actually_added),
                        )
                except Exception as e:
                    logger.error("Tool call failed: name=%s error=%s", display_name, e)
                    result_str = f"Error: {e}"
                    tc_status = "failed"
                    tc_error = str(e)
            else:
                logger.warning("Tool not available: name=%s", tool_name)
                result_str = "Tool not available"
                tc_status = "failed"
                tc_error = "Tool not available"

            if len(result_str) > MAX_TOOL_RESULT_CHARS:
                result_str = result_str[:MAX_TOOL_RESULT_CHARS] + (
                    f"\n\n[Tool output truncated from {len(result_str)} to "
                    f"{MAX_TOOL_RESULT_CHARS} characters]"
                )

            tc_completed = datetime.now(timezone.utc).isoformat()
            try:
                start_dt = datetime.fromisoformat(tc_started.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(tc_completed.replace("Z", "+00:00"))
                duration_ms = int((end_dt - start_dt).total_seconds() * 1000)
            except Exception:
                duration_ms = None

            args_for_trace = {k: v for k, v in args.items() if k not in ("user_id", "_pipeline_metadata")}
            tool_call_trace.append({
                "iteration": iteration,
                "tool_name": display_name,
                "args": _truncate_for_tool_trace(args_for_trace),
                "result": (result_str or "")[:1024] + ("..." if len(result_str or "") > 1024 else ""),
                "started_at": tc_started,
                "completed_at": tc_completed,
                "duration_ms": duration_ms,
                "status": tc_status,
                "error": tc_error,
            })
            tool_messages.append(ToolMessage(content=result_str, tool_call_id=tool_id))
        messages = messages + [response] + tool_messages

    if not (content or "").strip() and messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "content") and last_msg.content:
            content = last_msg.content

    from orchestrator.utils.action_io_registry import get_categories_for_tools
    categories = get_categories_for_tools(executed_tool_names) if executed_tool_names else []

    if output_json and (content or "").strip():
        schema = output_json if isinstance(output_json, dict) else None
        parsed = StructuredOutputParser.parse(content, schema=schema)
        if parsed is not None:
            parsed.setdefault("formatted", content)
            if categories:
                parsed["_tools_used_categories"] = categories
            parsed["_tool_call_trace"] = tool_call_trace
            parsed["_trace_inputs"] = {
                "prompt": (prompt or "")[:2048] + ("..." if len(prompt or "") > 2048 else ""),
                "tools_available": len(tool_names),
                "tool_calls_count": len(tool_call_trace),
            }
            if extracted_images_from_tools:
                parsed["images"] = extracted_images_from_tools
            if extracted_proposals:
                parsed["editor_proposals"] = extracted_proposals
            parsed["_token_usage"] = token_acc
            return parsed
    result = {
        "formatted": content or "(no output)",
        "raw": content,
        "_tool_call_trace": tool_call_trace,
        "_trace_inputs": {
            "prompt": (prompt or "")[:2048] + ("..." if len(prompt or "") > 2048 else ""),
            "tools_available": len(tool_names),
            "tool_calls_count": len(tool_call_trace),
        },
        "_token_usage": token_acc,
    }
    if categories:
        result["_tools_used_categories"] = categories
    if extracted_images_from_tools:
        result["images"] = extracted_images_from_tools
    if extracted_proposals:
        result["editor_proposals"] = extracted_proposals
    return result


async def execute_step(
    step: Dict[str, Any],
    playbook_state: Dict[str, Any],
    inputs: Dict[str, Any],
    user_id: str = "system",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute a single tool step: resolve inputs, look up action, coerce types, call tool, return result.

    step must have: step_type="tool", action=<registry name>, and optionally inputs, params, output_key, name.
    Result is the tool's return dict (with formatted and typed fields); store under playbook_state[output_key].
    """
    from orchestrator.utils.action_io_registry import get_action

    step_type = _step_type(step)
    if step_type != "tool":
        return {"_skipped": True, "_reason": f"Unsupported step_type: {step_type}"}

    action_name = step.get("action")
    if not action_name:
        logger.warning("Pipeline step missing action: %s", step.get("name"))
        return {"_error": "Missing action", "formatted": "Step has no action."}

    inputs_spec = step.get("inputs", {})
    params_spec = step.get("params", {})
    resolved = _resolve_inputs(inputs_spec, playbook_state, inputs)

    # Auto-resolve workspace_id for data workspace tools when profile has a single bound workspace
    if action_name in ("query_data_workspace", "get_workspace_schema") and not resolved.get("workspace_id"):
        workspace_ids = playbook_state.get("workspace_ids")
        if isinstance(workspace_ids, list) and len(workspace_ids) == 1:
            resolved["workspace_id"] = workspace_ids[0]

    if action_name == "query_data_workspace" and resolved.get("workspace_id"):
        modes = (playbook_state or {}).get("workspace_access_modes") or (metadata or {}).get("workspace_access_modes") or {}
        if isinstance(modes, dict) and modes.get(str(resolved["workspace_id"])) == "read":
            resolved["read_only"] = True

    blocked = _user_facts_policy_blocks_tool_step(action_name, step, metadata)
    if blocked:
        return {"_error": "user_facts_policy", "formatted": blocked}

    # Connector steps: action name is "connector:<connector_id>:<endpoint_id>"
    if action_name.startswith("connector:"):
        parts = action_name.split(":", 2)
        if len(parts) >= 3:
            _, connector_id, endpoint_id = parts[0], parts[1], parts[2]
            profile_id = (metadata or {}).get("agent_profile_id") if metadata else None
            if not profile_id:
                return {"_error": "No profile", "formatted": "Connector steps require agent_profile_id in metadata."}
            from orchestrator.backend_tool_client import get_backend_tool_client
            client = await get_backend_tool_client()
            result = await client.execute_connector(
                user_id=user_id,
                profile_id=profile_id,
                connector_id=connector_id,
                endpoint_id=endpoint_id,
                params={**resolved, **params_spec},
            )
            if result is None:
                return {"_error": "Connector failed", "formatted": "Connector execution failed."}
            if isinstance(result, dict) and "formatted" not in result:
                result.setdefault("formatted", str(result.get("records", [])))
            return result
        return {"_error": "Invalid connector action", "formatted": "Use connector:connector_id:endpoint_id"}

    # Email-scoped steps: action name is "email:<connection_id>:<tool_name>"
    if action_name.startswith("email:"):
        parts = action_name.split(":", 2)
        if len(parts) >= 3:
            try:
                connection_id = int(parts[1])
                tool_name = parts[2]
            except (ValueError, IndexError):
                return {"_error": "Invalid email action", "formatted": "Use email:connection_id:tool_name"}
            contract = get_action(tool_name)
            if not contract or not callable(getattr(contract, "tool_function", None)):
                return {"_error": f"Unknown email tool: {tool_name}", "formatted": f"Unknown email tool: {tool_name}"}
            tool_fn = contract.tool_function
            kwargs = {**resolved, **params_spec, "user_id": user_id, "connection_id": connection_id}
            sig = inspect.signature(tool_fn)
            final_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            if "user_id" not in sig.parameters and "user_id" in kwargs:
                final_kwargs.pop("user_id", None)
            if "_pipeline_metadata" in sig.parameters:
                final_kwargs["_pipeline_metadata"] = metadata or {}
            try:
                if inspect.iscoroutinefunction(tool_fn):
                    result = await tool_fn(**final_kwargs)
                else:
                    result = tool_fn(**final_kwargs)
            except Exception as e:
                logger.exception("Pipeline email step failed: action=%s", action_name)
                return {"_error": str(e), "formatted": f"Error executing {action_name}: {e}"}
            if isinstance(result, dict):
                return result
            return {"formatted": str(result), "result": result}
        return {"_error": "Invalid email action", "formatted": "Use email:connection_id:tool_name"}

    # Calendar-scoped steps: action name is "calendar:<connection_id>:<tool_name>"
    if action_name.startswith("calendar:"):
        parts = action_name.split(":", 2)
        if len(parts) >= 3:
            try:
                connection_id = int(parts[1])
                tool_name = parts[2]
            except (ValueError, IndexError):
                return {"_error": "Invalid calendar action", "formatted": "Use calendar:connection_id:tool_name"}
            contract = get_action(tool_name)
            if not contract or not callable(getattr(contract, "tool_function", None)):
                return {
                    "_error": f"Unknown calendar tool: {tool_name}",
                    "formatted": f"Unknown calendar tool: {tool_name}",
                }
            tool_fn = contract.tool_function
            kwargs = {**resolved, **params_spec, "user_id": user_id, "connection_id": connection_id}
            sig = inspect.signature(tool_fn)
            final_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            if "user_id" not in sig.parameters and "user_id" in kwargs:
                final_kwargs.pop("user_id", None)
            if "_pipeline_metadata" in sig.parameters:
                final_kwargs["_pipeline_metadata"] = metadata or {}
            try:
                if inspect.iscoroutinefunction(tool_fn):
                    result = await tool_fn(**final_kwargs)
                else:
                    result = tool_fn(**final_kwargs)
            except Exception as e:
                logger.exception("Pipeline calendar step failed: action=%s", action_name)
                return {"_error": str(e), "formatted": f"Error executing {action_name}: {e}"}
            if isinstance(result, dict):
                return result
            return {"formatted": str(result), "result": result}
        return {"_error": "Invalid calendar action", "formatted": "Use calendar:connection_id:tool_name"}

    # MCP steps: action name is "mcp:<server_id>:<tool_name>"
    if action_name.startswith("mcp:"):
        parts = action_name.split(":", 2)
        if len(parts) >= 3:
            try:
                server_id = int(parts[1])
                mcp_tool = parts[2]
            except (ValueError, IndexError):
                return {"_error": "Invalid MCP action", "formatted": "Use mcp:server_id:tool_name"}
            from orchestrator.tools.mcp_tools import run_mcp_tool_invocation

            args_payload = {**resolved, **params_spec}
            if "arguments" in args_payload and isinstance(args_payload["arguments"], dict):
                args = args_payload["arguments"]
            elif "arguments_json" in args_payload and isinstance(args_payload["arguments_json"], str):
                try:
                    args = json.loads(args_payload["arguments_json"] or "{}")
                except json.JSONDecodeError:
                    args = {}
            else:
                args = {k: v for k, v in args_payload.items() if k not in ("user_id",)}
            try:
                result = await run_mcp_tool_invocation(user_id, server_id, mcp_tool, args)
            except Exception as e:
                logger.exception("Pipeline MCP step failed: action=%s", action_name)
                return {"_error": str(e), "formatted": f"Error executing {action_name}: {e}"}
            if isinstance(result, dict):
                out = dict(result)
                if not (out.get("formatted") or "").strip():
                    out["formatted"] = (out.get("error") or out.get("result_json") or str(out)).strip()
                return out
            return {"formatted": str(result), "result": result}
        return {"_error": "Invalid MCP action", "formatted": "Use mcp:server_id:tool_name"}

    contract = get_action(action_name)
    if not contract:
        logger.warning("Pipeline action not in registry: %s", action_name)
        return {"_error": f"Unknown action: {action_name}", "formatted": f"Unknown action: {action_name}"}

    if contract.category.startswith("plugin:"):
        plugin_name = contract.category.split(":", 1)[1]
        plugin_creds_raw = (metadata or {}).get("plugin_credentials")
        if plugin_creds_raw and isinstance(plugin_creds_raw, str):
            try:
                plugin_creds_all = json.loads(plugin_creds_raw)
                creds = plugin_creds_all.get(plugin_name) or {}
            except (json.JSONDecodeError, TypeError):
                creds = {}
        else:
            creds = {}
        try:
            from orchestrator.plugins.plugin_loader import get_plugin_loader
            loader = get_plugin_loader()
            plugin = loader.get_plugin(plugin_name)
            if plugin:
                plugin.configure(creds)
        except Exception as e:
            logger.warning("Failed to configure plugin %s: %s", plugin_name, e)

    input_types = _get_input_types_for_action(action_name)
    for name, target_type in input_types.items():
        if name in resolved and resolved[name] is not None and target_type != "any":
            resolved[name] = _coerce_for_input(resolved[name], target_type)

    # Merge params (static) over resolved inputs; add user_id for tools that need it
    kwargs: Dict[str, Any] = {**resolved, **params_spec, "user_id": user_id}
    # Tool functions may not accept user_id; try without if they don't
    tool_fn = contract.tool_function
    import inspect
    sig = inspect.signature(tool_fn)
    final_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    if "user_id" not in sig.parameters and "user_id" in kwargs:
        final_kwargs.pop("user_id", None)
    if "agent_name" in sig.parameters and "agent_name" not in final_kwargs:
        profile_name = (metadata or {}).get("agent_profile_name", "")
        if profile_name:
            final_kwargs["agent_name"] = profile_name
    if "_pipeline_metadata" in sig.parameters:
        final_kwargs["_pipeline_metadata"] = metadata or {}

    step_name = step.get("name") or step.get("output_key") or "?"
    logger.info("Tool step: action=%s step=%s", action_name, step_name)
    try:
        if inspect.iscoroutinefunction(tool_fn):
            result = await tool_fn(**final_kwargs)
        else:
            result = tool_fn(**final_kwargs)
    except Exception as e:
        logger.exception("Pipeline step failed: action=%s", action_name)
        return {
            "_error": str(e),
            "formatted": f"Error executing {action_name}: {e}",
        }

    if isinstance(result, dict):
        result["_action_category"] = contract.category
        return result
    return {"formatted": str(result), "result": result, "_action_category": contract.category}


async def execute_pipeline(
    steps: List[Dict[str, Any]],
    initial_state: Dict[str, Any],
    inputs: Dict[str, Any],
    user_id: str = "system",
    metadata: Optional[Dict[str, Any]] = None,
    resume_after_step_name: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Execute a sequence of tool and llm_task steps. Resolves inputs from state and inputs, runs each step,
    stores results in playbook_state by step name and output_key. Returns (playbook_state, pending_approval).

    If a step has step_type="approval", execution stops and returns (state, approval_dict).
    Tool steps and llm_task steps are executed in order; metadata is used for user model preference on llm_task.

    When resume_after_step_name is set (e.g. after HITL approval), skips steps until the step after that name,
    then continues. Used to resume after an approval gate.
    """
    playbook_state = dict(initial_state)
    skip_until_resumed = bool(resume_after_step_name)
    for step in steps:
        step_type = _step_type(step)
        name = step.get("name", "")
        output_key = step.get("output_key")

        if skip_until_resumed:
            if name == resume_after_step_name:
                skip_until_resumed = False
            continue

        if step_type == "approval":
            preview = playbook_state.get(output_key or name) if (output_key or name) else None
            return playbook_state, {
                "step_name": name,
                "preview_data": preview,
                "prompt": step.get("prompt", "Approve to continue?"),
                "timeout_minutes": step.get("timeout_minutes"),
                "on_reject": step.get("on_reject", "stop"),
            }

        if step_type == "loop":
            max_iter = max(1, int(step.get("max_iterations", 3)))
            child_steps = step.get("steps", [])
            for iteration in range(max_iter):
                playbook_state["_iteration"] = iteration + 1
                child_state, child_pending = await execute_pipeline(
                    child_steps,
                    playbook_state,
                    inputs,
                    user_id=user_id,
                    metadata=metadata,
                )
                playbook_state.update(child_state)
                if child_pending:
                    return playbook_state, child_pending
            continue

        if step_type == "parallel":
            children = step.get("parallel_steps", [])
            inputs_ref = inputs
            user_id_ref = user_id
            metadata_ref = metadata

            async def _run_one(child: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
                st = _step_type(child)
                child_ps = dict(playbook_state)
                if st == "llm_task":
                    result = await _execute_llm_step(child, child_ps, inputs_ref, user_id_ref, metadata=metadata_ref)
                elif st == "llm_agent":
                    result = await _execute_llm_agent_step(child, child_ps, inputs_ref, user_id_ref, metadata=metadata_ref)
                else:
                    result = await execute_step(child, child_ps, inputs_ref, user_id=user_id_ref, metadata=metadata_ref)
                return child, result

            results = await asyncio.gather(*[_run_one(c) for c in children])
            for child, result in results:
                key = child.get("output_key") or child.get("name")
                if key:
                    playbook_state[key] = result
                if child.get("name"):
                    playbook_state[child["name"]] = result
            continue

        if step_type == "branch":
            condition_expr = step.get("branch_condition", "")
            condition_met = _evaluate_condition(condition_expr, playbook_state, inputs)
            branch_steps = step.get("then_steps", []) if condition_met else step.get("else_steps", [])
            if branch_steps:
                child_state, child_pending = await execute_pipeline(
                    branch_steps,
                    playbook_state,
                    inputs,
                    user_id=user_id,
                    metadata=metadata,
                )
                playbook_state.update(child_state)
                if child_pending:
                    return playbook_state, child_pending
            continue

        condition = step.get("condition")
        if condition and not _evaluate_condition(condition, playbook_state, inputs):
            key = step.get("output_key") or name
            if key:
                playbook_state[key] = {"_skipped": True}
            if name:
                playbook_state[name] = {"_skipped": True}
            continue

        if step_type == "llm_task":
            result = await _execute_llm_step(step, playbook_state, inputs, user_id, metadata=metadata)
        elif step_type == "deep_agent":
            result = await _execute_deep_agent_step(step, playbook_state, inputs, user_id, metadata=metadata)
        elif step_type == "llm_agent":
            result = await _execute_llm_agent_step(step, playbook_state, inputs, user_id, metadata=metadata)
        else:
            result = await execute_step(step, playbook_state, inputs, user_id=user_id, metadata=metadata)

        if output_key:
            playbook_state[output_key] = result
        if name:
            playbook_state[name] = result

    return playbook_state, None
