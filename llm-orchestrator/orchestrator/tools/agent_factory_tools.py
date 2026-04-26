"""
Agent Factory meta-tools - create agents, playbooks, schedules, and data source bindings from natural language.

Zone 1: list_available_actions_tool (reads Action I/O Registry in-process).
Zone 2: create_agent_profile, create_playbook, assign_playbook_to_agent, create_agent_schedule, bind_data_source_to_agent (via gRPC).
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.playbook_contracts import (
    DEEP_AGENT_SISTER_PHASE_OUTPUT_FIELDS,
    VALID_DEEP_AGENT_PHASE_TYPES,
    VALID_PLAYBOOK_STEP_TYPES,
)
from orchestrator.utils.action_io_registry import (
    get_all_actions,
    get_actions_by_category,
    is_type_compatible,
    register_action,
)
import re

logger = logging.getLogger(__name__)

# Max chars for including full playbook definition in get_playbook_detail formatted output (under pipeline MAX_TOOL_RESULT_CHARS)
GET_PLAYBOOK_FULL_DEFINITION_MAX_CHARS = 45000

# ── I/O models ─────────────────────────────────────────────────────────────

class ListAvailableActionsInputs(BaseModel):
    """Optional filters for listing actions."""
    category: Optional[str] = Field(default=None, description="Filter by category (e.g. search, email)")
    query: Optional[str] = Field(default=None, description="Filter by name/description (substring)")


class ListAvailableActionsOutputs(BaseModel):
    """Outputs for list_available_actions_tool."""
    actions: List[Dict[str, Any]] = Field(description="List of action records with name, category, description, input_fields, output_fields")
    count: int = Field(description="Number of actions returned")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class CreateAgentProfileInputs(BaseModel):
    """Required inputs for create_agent_profile_tool."""
    name: str = Field(description="Agent display name")
    handle: Optional[str] = Field(default=None, description="Unique handle for @mention (e.g. invoice-monitor). Omit or leave empty for schedule/Run-only agents (not @mentionable).")


class CreateAgentProfileParams(BaseModel):
    """Optional configuration for create_agent_profile_tool."""
    description: Optional[str] = Field(default=None, description="Agent description")
    model_preference: Optional[str] = Field(default=None, description="LLM model ID")
    system_prompt_additions: Optional[str] = Field(default=None, description="Additional system prompt text")
    persona_enabled: bool = Field(default=False, description="Enable persona")
    auto_routable: bool = Field(default=False, description="Allow routing to this agent")
    chat_visible: bool = Field(default=True, description="Show in chat @ menu; when false, agent is still addressable by other agents in teams")
    confirmed: bool = Field(default=False, description="If false, return preview only; if true, create in DB")


class CreateAgentProfileOutputs(BaseModel):
    """Outputs for create_agent_profile_tool."""
    agent_id: str = Field(description="Created profile UUID")
    name: str = Field(description="Agent name")
    handle: Optional[str] = Field(default=None, description="Agent handle, or null for schedule/Run-only")
    is_draft: bool = Field(description="True if preview (confirmed=False), false if created")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class CreatePlaybookInputs(BaseModel):
    """Required inputs for create_playbook_tool."""
    name: str = Field(description="Playbook name")
    steps: List[Dict[str, Any]] = Field(description="List of step definitions (name, step_type, output_key, action for tool steps, inputs, etc.)")


class CreatePlaybookParams(BaseModel):
    """Optional configuration for create_playbook_tool."""
    description: Optional[str] = Field(default=None, description="Playbook description")
    run_context: str = Field(default="background", description="interactive or background")
    category: Optional[str] = Field(default=None, description="Category")
    tags: Optional[List[str]] = Field(default=None, description="Tags")
    confirmed: bool = Field(default=False, description="If false, validate and return preview; if true, create in DB")


class CreatePlaybookOutputs(BaseModel):
    """Outputs for create_playbook_tool."""
    playbook_id: str = Field(description="Created playbook UUID")
    name: str = Field(description="Playbook name")
    step_count: int = Field(description="Number of steps")
    validation_warnings: List[str] = Field(description="Validation warnings if any")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class AssignPlaybookToAgentInputs(BaseModel):
    """Inputs for assign_playbook_to_agent_tool."""
    agent_id: str = Field(description="Agent profile UUID")
    playbook_id: str = Field(description="Playbook UUID")


class AssignPlaybookToAgentParams(BaseModel):
    """Optional for assign_playbook_to_agent_tool."""
    confirmed: bool = Field(default=False, description="If false, preview; if true, apply")


class AssignPlaybookToAgentOutputs(BaseModel):
    """Outputs for assign_playbook_to_agent_tool."""
    success: bool = Field(description="Whether assignment succeeded")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class SetAgentProfileStatusInputs(BaseModel):
    """Inputs for set_agent_profile_status_tool."""
    agent_id: str = Field(description="Agent profile UUID")


class SetAgentProfileStatusParams(BaseModel):
    """Optional for set_agent_profile_status_tool."""
    is_active: bool = Field(description="True to activate, False to pause")
    confirmed: bool = Field(default=False, description="If false, preview; if true, apply")


class SetAgentProfileStatusOutputs(BaseModel):
    """Outputs for set_agent_profile_status_tool."""
    success: bool = Field(description="Whether the status update succeeded")
    is_active: bool = Field(description="New active status")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class CreateAgentScheduleInputs(BaseModel):
    """Required inputs for create_agent_schedule_tool."""
    agent_id: str = Field(description="Agent profile UUID")
    schedule_type: str = Field(description="cron or interval")


class CreateAgentScheduleParams(BaseModel):
    """Optional configuration for create_agent_schedule_tool."""
    cron_expression: Optional[str] = Field(default=None, description="Cron expression (e.g. 0 8 * * * for 8 AM daily)")
    interval_seconds: Optional[int] = Field(default=None, description="Seconds between runs for interval type")
    timezone: str = Field(default="UTC", description="Timezone for cron")
    is_active: bool = Field(default=False, description="Start active (default False for LLM-created schedules)")
    confirmed: bool = Field(default=False, description="If false, preview; if true, create")


class CreateAgentScheduleOutputs(BaseModel):
    """Outputs for create_agent_schedule_tool."""
    schedule_id: str = Field(description="Created schedule UUID")
    next_run_at: str = Field(description="Next run time (ISO or empty)")
    is_active: bool = Field(description="Whether schedule is active")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class ListAgentSchedulesInputs(BaseModel):
    """Inputs for list_agent_schedules_tool."""
    agent_id: str = Field(description="Agent profile UUID to list schedules for")


class ListAgentSchedulesOutputs(BaseModel):
    """Outputs for list_agent_schedules_tool."""
    schedules: List[Dict[str, Any]] = Field(description="List of schedule objects (id, schedule_type, cron_expression, is_active, next_run_at, etc.)")
    count: int = Field(description="Number of schedules")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class ListAgentDataSourcesInputs(BaseModel):
    """Inputs for list_agent_data_sources_tool."""
    agent_id: str = Field(description="Agent profile UUID to list data source bindings for")


class ListAgentDataSourcesOutputs(BaseModel):
    """Outputs for list_agent_data_sources_tool."""
    bindings: List[Dict[str, Any]] = Field(description="List of binding objects (binding_id, connector_id, connector_name, is_enabled, etc.)")
    count: int = Field(description="Number of bindings")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class BindDataSourceToAgentInputs(BaseModel):
    """Inputs for bind_data_source_to_agent_tool."""
    agent_id: str = Field(description="Agent profile UUID")
    connector_id: str = Field(description="Data source connector UUID")


class BindDataSourceToAgentParams(BaseModel):
    """Optional for bind_data_source_to_agent_tool."""
    config_overrides: Optional[Dict[str, Any]] = Field(default=None, description="Connector config overrides")
    confirmed: bool = Field(default=False, description="If false, preview; if true, create binding")


class BindDataSourceToAgentOutputs(BaseModel):
    """Outputs for bind_data_source_to_agent_tool."""
    binding_id: str = Field(description="Created binding UUID")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class UpdateAgentProfileInputs(BaseModel):
    """Inputs for update_agent_profile_tool."""
    agent_id: str = Field(description="Agent profile UUID to update")
    updates: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Fields to update (e.g. name, description, model_preference, default_playbook_id, chat_visible, is_locked, prompt_history_enabled, chat_history_lookback, summary_threshold_tokens 500–100000, summary_keep_messages 1–50). When profile is locked only is_active and is_locked are allowed.",
    )


class UpdateAgentProfileParams(BaseModel):
    """Optional for update_agent_profile_tool."""
    confirmed: bool = Field(default=False, description="If false, preview; if true, apply update")


class UpdateAgentProfileOutputs(BaseModel):
    """Outputs for update_agent_profile_tool."""
    success: bool = Field(description="Whether the update succeeded")
    agent_id: str = Field(description="Agent profile UUID")
    name: str = Field(description="Updated agent name")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class DeleteAgentProfileInputs(BaseModel):
    """Inputs for delete_agent_profile_tool."""
    agent_id: str = Field(description="Agent profile UUID to delete")


class DeleteAgentProfileParams(BaseModel):
    """Optional for delete_agent_profile_tool."""
    confirmed: bool = Field(default=False, description="If false, preview; if true, delete (irreversible). Blocked when profile is locked.")


class DeleteAgentProfileOutputs(BaseModel):
    """Outputs for delete_agent_profile_tool."""
    success: bool = Field(description="Whether the delete succeeded")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class UpdatePlaybookInputs(BaseModel):
    """Inputs for update_playbook_tool."""
    playbook_id: str = Field(description="Playbook identifier: UUID (e.g. b6fa2492-30ec-45c4-8459-a26e0f1812cb) or name/slug (e.g. morning-intelligence-briefing)")
    updates: Optional[Dict[str, Any]] = Field(default=None, description="Fields to update (e.g. name, description, definition with steps, category, tags, is_locked). When playbook is locked only is_locked is allowed.")


class UpdatePlaybookParams(BaseModel):
    """Optional for update_playbook_tool."""
    confirmed: bool = Field(default=False, description="If false, preview; if true, apply update")


class UpdatePlaybookOutputs(BaseModel):
    """Outputs for update_playbook_tool."""
    success: bool = Field(description="Whether the update succeeded")
    playbook_id: str = Field(description="Playbook UUID")
    name: str = Field(description="Updated playbook name")
    step_count: int = Field(description="Number of steps")
    validation_warnings: List[str] = Field(description="Validation warnings if any")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class DeletePlaybookInputs(BaseModel):
    """Inputs for delete_playbook_tool."""
    playbook_id: str = Field(description="Playbook UUID to delete")


class DeletePlaybookParams(BaseModel):
    """Optional for delete_playbook_tool."""
    confirmed: bool = Field(default=False, description="If false, preview; if true, delete (irreversible). Blocked when playbook is locked or is a template.")


class DeletePlaybookOutputs(BaseModel):
    """Outputs for delete_playbook_tool."""
    success: bool = Field(description="Whether the delete succeeded")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class ListPlaybooksInputs(BaseModel):
    pass


class ListPlaybooksOutputs(BaseModel):
    playbooks: List[Dict[str, Any]] = Field(description="List of playbooks (id, name, description, step_count, is_template)")
    count: int = Field(description="Number of playbooks")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class GetPlaybookDetailInputs(BaseModel):
    """Inputs for get_playbook_detail_tool."""
    playbook_id: str = Field(description="Playbook UUID to fetch (full definition and steps)")


class GetPlaybookDetailOutputs(BaseModel):
    """Outputs for get_playbook_detail_tool."""
    playbook_id: str = Field(description="Playbook UUID")
    name: str = Field(description="Playbook name")
    description: Optional[str] = Field(default=None, description="Playbook description")
    version: Optional[str] = Field(default=None, description="Version string")
    definition: Dict[str, Any] = Field(description="Full definition including steps")
    triggers: List[Any] = Field(description="Trigger patterns (keyword/regex)")
    category: Optional[str] = Field(default=None, description="Category")
    tags: List[str] = Field(description="Tags")
    step_count: int = Field(description="Number of steps")
    is_template: bool = Field(description="Whether playbook is a template")
    is_locked: bool = Field(description="Whether playbook is locked")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class ListAgentProfilesInputs(BaseModel):
    pass


class ListAgentProfilesOutputs(BaseModel):
    profiles: List[Dict[str, Any]] = Field(description="List of agent profiles (id, name, handle, description, status, ...)")
    count: int = Field(description="Number of profiles")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class GetAgentProfileDetailInputs(BaseModel):
    """Inputs for get_agent_profile_detail_tool."""
    agent_id: str = Field(description="Agent profile UUID to fetch (full details)")


class GetAgentProfileDetailOutputs(BaseModel):
    """Outputs for get_agent_profile_detail_tool."""
    agent_id: str = Field(description="Agent profile UUID")
    name: str = Field(description="Agent display name")
    handle: Optional[str] = Field(default=None, description="Handle for @mention")
    description: Optional[str] = Field(default=None, description="Agent description")
    model_preference: Optional[str] = Field(default=None, description="LLM model ID")
    system_prompt_additions: Optional[str] = Field(default=None, description="Additional system prompt text")
    default_playbook_id: Optional[str] = Field(default=None, description="Default playbook UUID")
    is_active: bool = Field(description="Whether agent is active")
    is_locked: bool = Field(description="Whether profile is locked")
    prompt_history_enabled: bool = Field(description="Whether history is injected into the LLM prompt")
    summary_threshold_tokens: Optional[int] = Field(default=None, description="Token estimate threshold before compressing older history (500–100000)")
    summary_keep_messages: Optional[int] = Field(default=None, description="Recent messages kept verbatim when summarizing (1–50)")
    persona_mode: Optional[str] = Field(default=None, description="Persona mode")
    auto_routable: bool = Field(description="Whether agent is auto-routable")
    knowledge_config: Dict[str, Any] = Field(description="Knowledge config")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class ListSkillsInputs(BaseModel):
    """Optional inputs for list_skills_tool."""
    category: Optional[str] = Field(default=None, description="Filter by category")


class ListSkillsOutputs(BaseModel):
    """Outputs for list_skills_tool."""
    skills: List[Dict[str, Any]] = Field(description="List of skill records (id, name, slug, category, description, required_tools)")
    count: int = Field(description="Number of skills")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class CreateSkillInputs(BaseModel):
    """Required inputs for create_skill_tool."""
    name: str = Field(description="Display name for the skill")
    slug: str = Field(description="Stable identifier (e.g. pfas-research-guide)")
    procedure: str = Field(description="Markdown procedure text: how to use tools and perform the task")


class CreateSkillParams(BaseModel):
    """Optional for create_skill_tool."""
    description: Optional[str] = Field(default=None, description="Short description")
    category: Optional[str] = Field(default=None, description="Category (e.g. research, email)")
    required_tools: Optional[List[str]] = Field(default=None, description="Tool names to auto-bind")
    optional_tools: Optional[List[str]] = Field(default=None, description="Optional tool names")
    tags: Optional[List[str]] = Field(default=None, description="Tags")
    confirmed: bool = Field(default=False, description="If false, preview; if true, create")


class CreateSkillOutputs(BaseModel):
    """Outputs for create_skill_tool."""
    skill_id: str = Field(description="Created skill UUID")
    name: str = Field(description="Skill name")
    slug: str = Field(description="Skill slug")
    is_draft: bool = Field(description="True if preview (confirmed=False)")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class ProposeSkillUpdateInputs(BaseModel):
    """Required inputs for propose_skill_update_tool."""
    skill_id: str = Field(description="User skill UUID to update")
    proposed_procedure: str = Field(description="New procedure text (replaces or extends current)")
    rationale: str = Field(description="Why this change improves the skill")


class ProposeSkillUpdateParams(BaseModel):
    """Optional for propose_skill_update_tool."""
    evidence_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional evidence (e.g. execution counts, failure cases)")
    confirmed: bool = Field(default=False, description="If false, preview; if true, create new version")


class ProposeSkillUpdateOutputs(BaseModel):
    """Outputs for propose_skill_update_tool."""
    success: bool = Field(description="Whether the update succeeded")
    skill_id: str = Field(description="Skill UUID")
    version: int = Field(description="New version number")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class GetSkillDetailInputs(BaseModel):
    """Required inputs for get_skill_detail_tool."""
    skill_id: str = Field(description="Skill UUID to fetch")


class GetSkillDetailOutputs(BaseModel):
    """Outputs for get_skill_detail_tool."""
    skill_id: str = Field(description="Skill UUID")
    name: str = Field(description="Skill display name")
    slug: str = Field(description="Skill slug identifier")
    procedure: str = Field(description="Full procedure text (markdown)")
    required_tools: List[str] = Field(description="Tools required by this skill")
    optional_tools: List[str] = Field(description="Optional tools")
    version: int = Field(description="Current version number")
    category: str = Field(description="Skill category")
    tags: List[str] = Field(description="Tags")
    improvement_rationale: str = Field(description="Rationale for last update (if any)")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class ValidatePlaybookWiringInputs(BaseModel):
    """Inputs for validate_playbook_wiring_tool."""
    definition: Any = Field(default=None, description="Playbook definition with 'steps' key, or a list of step objects.")


class ValidatePlaybookWiringOutputs(BaseModel):
    """Outputs for validate_playbook_wiring_tool."""
    issues: List[Dict[str, Any]] = Field(description="List of {step_name, input_key, message}")
    count: int = Field(description="Number of wiring issues found")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


# ── Helpers ────────────────────────────────────────────────────────────────

VALID_STEP_TYPES = VALID_PLAYBOOK_STEP_TYPES

# Runtime variables that can be referenced without a dot (e.g. {today})
RUNTIME_VARS = frozenset({
    "today", "today_end", "tomorrow", "today_day_of_week", "query", "query_length", "history",
    "user_weather_location", "trigger_input",
    "editor", "editor_refs", "editor_document_id", "editor_filename", "editor_length",
    "editor_document_type", "editor_cursor_offset", "editor_selection",
    "editor_current_section", "editor_current_heading",
    "editor_previous_section", "editor_next_section",
    "editor_section_index", "editor_adjacent_sections", "editor_total_sections",
    "editor_toc", "editor_linked_notes",
    "editor_is_first_section", "editor_is_last_section", "editor_ref_count",
    "document_context", "pinned_document_id", "last_tool_results",
    "profile",
    "current_item",  # fan_out default item_variable injected into playbook_state
})

_REF_PATTERN = re.compile(r"\{([^}]+)\}")


def _is_runtime_var(ref: str) -> bool:
    """True if ref is a known runtime var or an editor_refs_CATEGORY variable."""
    return ref in RUNTIME_VARS or ref.startswith("editor_refs_")


def _get_output_fields_for_step(step: Dict[str, Any], actions_by_name: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of {name, type} for step outputs. Mirrors frontend getOutputFieldsForStep."""
    action = actions_by_name.get(step.get("action") or "")
    if action and action.get("output_fields"):
        return action["output_fields"]
    step_type = step.get("step_type") or "tool"
    if step_type == "deep_agent":
        return [
            {"name": "formatted", "type": "text"},
            {"name": "phase_trace", "type": "record"},
            {"name": "raw", "type": "text"},
        ]
    if step_type == "llm_task":
        out = [{"name": "formatted", "type": "text"}, {"name": "raw", "type": "text"}]
        schema_props = (step.get("output_schema") or {}).get("properties") or {}
        for fname, fdef in schema_props.items():
            if fname not in ("formatted", "raw") and isinstance(fdef, dict):
                ftype = (fdef.get("type") or "string").replace("string", "text").replace("integer", "number").replace("number", "number").replace("boolean", "boolean").replace("array", "list[any]").replace("object", "record")
                if ftype not in ("text", "number", "boolean", "date", "record", "any") and not ftype.startswith("list["):
                    ftype = "text"
                out.append({"name": fname, "type": ftype})
        return out
    if step_type == "approval":
        return [{"name": "approved", "type": "boolean"}]
    return [{"name": "formatted", "type": "text"}]


def _expand_upstream_steps(steps_before: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand parallel/branch into flat list of (key, step). Uses output_key || name || action as key."""
    expanded: List[Dict[str, Any]] = []
    for s in steps_before or []:
        if s.get("step_type") == "parallel":
            for child in (s.get("parallel_steps") or []):
                k = child.get("output_key") or child.get("name") or child.get("action")
                if k:
                    expanded.append({"key": k, "step": child})
        elif s.get("step_type") == "branch":
            for child in (s.get("then_steps") or []) + (s.get("else_steps") or []):
                k = child.get("output_key") or child.get("name") or child.get("action")
                if k:
                    expanded.append({"key": k, "step": child})
        else:
            k = s.get("output_key") or s.get("name") or s.get("action")
            if k:
                expanded.append({"key": k, "step": s})
    return expanded


def _scan_template_for_ref_issues(
    template: str,
    field_label: str,
    upstream_by_key: Dict[str, Dict[str, Any]],
    actions_by_name: Dict[str, Any],
    current_step: Dict[str, Any],
    *,
    sister_phase_names: Optional[Set[str]] = None,
    check_tool_input_types: bool = False,
    tool_input_key: Optional[str] = None,
) -> List[tuple]:
    """Return list of (field_label, message) for each invalid {ref} in template."""
    issues: List[tuple] = []
    if not isinstance(template, str):
        return issues
    for match in _REF_PATTERN.finditer(template):
        ref = match.group(1).strip()
        dot = ref.find(".")
        if ref.startswith("{"):
            inner = ref.lstrip("{").strip()
            if inner.startswith("#") or inner.startswith("/"):
                continue
            if inner in RUNTIME_VARS:
                hint = f"replace {{{{{inner}}}}} with {{{inner}}}"
            else:
                hint = (
                    f"replace {{{{{inner}}}}} with {{upstream_output_key.formatted}} using the actual output_key of an upstream step"
                )
            issues.append((field_label, f"Double braces detected — {hint}. Never use {{{{double}}}} braces."))
            continue
        if dot == -1:
            if _is_runtime_var(ref):
                continue
            issues.append(
                (
                    field_label,
                    f'Invalid reference "{{{ref}}}": use {{output_key.field_name}} or runtime var (today, today_end, tomorrow, query, history).',
                )
            )
            continue
        ref_step_key = ref[:dot].strip()
        ref_field = ref[dot + 1:].strip()
        upstream = upstream_by_key.get(ref_step_key)
        if upstream:
            output_fields = _get_output_fields_for_step(upstream, actions_by_name)
            out_field = next((f for f in output_fields if f.get("name") == ref_field), None)
            if not out_field:
                issues.append(
                    (
                        field_label,
                        f'Upstream step "{ref_step_key}" has no output "{ref_field}". '
                        f'Fix: use .formatted or .raw (e.g. {{{ref_step_key}.formatted}}), or declare "{ref_field}" in that step\'s output_schema.properties.',
                    )
                )
                continue
            if check_tool_input_types and tool_input_key:
                current_action = actions_by_name.get(current_step.get("action") or "")
                input_fields = (current_action or {}).get("input_fields") or []
                in_field = next((f for f in input_fields if f.get("name") == tool_input_key), None)
                target_type = (in_field or {}).get("type") or "text"
                if not is_type_compatible((out_field or {}).get("type") or "text", target_type):
                    issues.append(
                        (
                            field_label,
                            f"Type mismatch: {ref_step_key}.{ref_field} is not compatible with {tool_input_key} ({target_type}). "
                            f"Fix: wire a different output field (e.g. .formatted for text) or use an upstream step that returns {target_type}.",
                        )
                    )
            continue
        if sister_phase_names is not None and ref_step_key in sister_phase_names:
            if ref_field not in DEEP_AGENT_SISTER_PHASE_OUTPUT_FIELDS:
                issues.append(
                    (
                        field_label,
                        f'Unknown field "{ref_field}" for phase "{ref_step_key}" in the same deep_agent step. '
                        f'Use one of: {", ".join(sorted(DEEP_AGENT_SISTER_PHASE_OUTPUT_FIELDS))}.',
                    )
                )
            continue
        if sister_phase_names is not None:
            issues.append(
                (
                    field_label,
                    f'Unknown upstream step or phase "{ref_step_key}" in {field_label}. '
                    "Use the output_key of a prior step or the name of another phase in this deep_agent.",
                )
            )
        else:
            issues.append(
                (field_label, f'Unknown upstream step "{ref_step_key}" in {field_label}. Use the output_key of a prior step.')
            )
    return issues


def _validate_playbook_wiring(steps: List[Dict[str, Any]], actions_by_name: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Validate playbook step wirings. Returns list of {step_index, step_name, input_key, message}.
    Mirrors frontend validatePlaybookWiring; uses output_key for upstream step identity.
    """
    errors: List[Dict[str, Any]] = []
    if not isinstance(steps, list) or not actions_by_name:
        return errors
    step_types_human = ", ".join(sorted(VALID_STEP_TYPES))
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        step_name = step.get("name") or step.get("output_key") or step.get("action") or f"Step {i + 1}"
        inputs = step.get("inputs") or {}
        step_type = step.get("step_type") or "tool"

        if step_type and step_type not in VALID_STEP_TYPES:
            errors.append(
                {
                    "step_index": i,
                    "step_name": step_name,
                    "input_key": "step_type",
                    "message": (
                        f'Invalid step_type "{step_type}". Must be one of: {step_types_human}. '
                        'Fix: use "tool" for action calls, "llm_task" for LLM prompts, "deep_agent" for multi-phase workflows, '
                        '"browser_authenticate" for browser login flows.'
                    ),
                }
            )

        if step_type == "tool" and step.get("action"):
            action = actions_by_name.get(step["action"])
            input_fields = (action or {}).get("input_fields") or []
            for field in input_fields:
                if not field.get("required"):
                    continue
                val = inputs.get(field["name"])
                if val is None or (isinstance(val, str) and not val.strip()):
                    errors.append(
                        {
                            "step_index": i,
                            "step_name": step_name,
                            "input_key": field["name"],
                            "message": (
                                f'Required input "{field["name"]}" has no value. Wire it to an upstream step '
                                f"(e.g. {{step_1.formatted}}) or enter a literal (e.g. {{today}})."
                            ),
                        }
                    )

        expanded_upstream = _expand_upstream_steps(steps[:i])
        upstream_by_key = {e["key"]: e["step"] for e in expanded_upstream}

        sister_names: Optional[Set[str]] = None
        if step_type == "deep_agent":
            phases_list = step.get("phases") if isinstance(step.get("phases"), list) else []
            sister_names = {
                (p.get("name") or "").strip()
                for p in phases_list
                if isinstance(p, dict) and (p.get("name") or "").strip()
            }

        def _emit_template_issues(
            template: str,
            fld: str,
            *,
            sisters: Optional[Set[str]] = None,
            check_types: bool = False,
            tool_ik: Optional[str] = None,
        ) -> None:
            for fld_out, msg in _scan_template_for_ref_issues(
                template,
                fld,
                upstream_by_key,
                actions_by_name,
                step,
                sister_phase_names=sisters,
                check_tool_input_types=check_types,
                tool_input_key=tool_ik,
            ):
                errors.append({"step_index": i, "step_name": step_name, "input_key": fld_out, "message": msg})

        for input_key, value in (inputs or {}).items():
            if isinstance(value, str):
                _emit_template_issues(value, input_key, sisters=None, check_types=True, tool_ik=input_key)

        if step_type in ("llm_task", "llm_agent"):
            prompt = step.get("prompt") or step.get("prompt_template") or ""
            _emit_template_issues(prompt, "prompt", sisters=None)

        if step_type == "deep_agent":
            phases_list = step.get("phases") if isinstance(step.get("phases"), list) else []
            for pj, phase in enumerate(phases_list):
                if not isinstance(phase, dict):
                    continue
                pname = (phase.get("name") or "").strip() or f"phase_{pj}"
                for key in ("prompt", "criteria"):
                    val = phase.get(key)
                    if isinstance(val, str):
                        _emit_template_issues(val, f"phases[{pj}].{key} ({pname})", sisters=sister_names)
            for tmpl_key in ("output_template", "prompt", "prompt_template"):
                tv = step.get(tmpl_key)
                if isinstance(tv, str) and tv.strip():
                    _emit_template_issues(tv, tmpl_key, sisters=sister_names)

    return errors


def _step_has_nonempty_condition(step: Dict[str, Any]) -> bool:
    c = step.get("condition")
    if c is None:
        return False
    return bool(str(c).strip())


def _step_exclusive_set(step: Dict[str, Any]) -> bool:
    return bool(step.get("exclusive"))


def _step_type_for_exclusive_warn(step: Dict[str, Any]) -> str:
    return str(step.get("step_type") or step.get("type") or "").strip().lower()


def _warn_missing_exclusive(steps: List[Any], warnings: List[str]) -> None:
    """
    Warn when 2+ consecutive steps have a condition but not `exclusive`, followed by a step
    with no condition (catch-all). Without exclusive, the catch-all still runs after a match.
    Skips runs where every step is type branch (different routing semantics).
    """
    if not isinstance(steps, list) or len(steps) < 3:
        return
    n = len(steps)
    i = 0
    while i < n:
        step = steps[i]
        if not isinstance(step, dict) or not _step_has_nonempty_condition(step) or _step_exclusive_set(step):
            i += 1
            continue
        j = i
        run_indices: List[int] = []
        while j < n:
            s = steps[j]
            if not isinstance(s, dict) or not _step_has_nonempty_condition(s) or _step_exclusive_set(s):
                break
            run_indices.append(j)
            j += 1
        if len(run_indices) >= 2 and j < n:
            nxt = steps[j]
            if isinstance(nxt, dict) and not _step_has_nonempty_condition(nxt):
                if not all(_step_type_for_exclusive_warn(steps[k]) == "branch" for k in run_indices):
                    names: List[str] = []
                    for k in run_indices:
                        nm = str(steps[k].get("name") or steps[k].get("output_key") or f"step_{k}").strip()
                        quoted = f'"{nm}"' if nm else f"step_{k}"
                        names.append(quoted)
                    qnames = ", ".join(names)
                    lo, hi = run_indices[0], run_indices[-1]
                    catch = nxt.get("name") or nxt.get("output_key") or f"step_{j}"
                    catch_s = str(catch).strip() or f"step_{j}"
                    warnings.append(
                        f"Steps {qnames} (steps {lo}-{hi}) have conditions but are not marked exclusive; "
                        f'step {j} ("{catch_s}") has no condition and will always run — even after a conditional '
                        f'step matches. Enable "Exclusive (stop after match)" on the conditional steps, or add a '
                        f'condition to "{catch_s}".'
                    )
        i = run_indices[0] + 1


def _validate_playbook_steps(steps: List[Dict[str, Any]]) -> List[str]:
    """Minimal structure validation for playbook steps. Returns list of warning strings."""
    warnings: List[str] = []
    if not isinstance(steps, list):
        return ["steps must be a list"]
    seen_names = set()
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            warnings.append(f"step {i}: must be an object")
            continue
        name = (step.get("name") or "").strip()
        if not name:
            warnings.append(f"step {i}: missing or empty 'name'")
        step_type = step.get("step_type")
        if not step_type:
            warnings.append(f"step {i} ({name or '?'}): missing 'step_type'")
        elif step_type not in VALID_STEP_TYPES:
            warnings.append(f"step {i} ({name or '?'}): invalid step_type '{step_type}'")
        if not (step.get("output_key") or "").strip():
            warnings.append(f"step {i} ({name or '?'}): missing or empty 'output_key'")
        if step_type == "tool" and not step.get("action"):
            warnings.append(f"step {i} ({name or '?'}): tool step requires 'action'")
        if step_type == "llm_agent" and "max_iterations" not in step:
            warnings.append(
                f"step {i} ({name or '?'}): llm_agent step has no max_iterations (defaults to 3); consider setting max_iterations (1–50) for complex or multi-tool tasks."
            )
        if step_type == "llm_agent":
            available_tools = step.get("available_tools") or []
            max_iter = step.get("max_iterations")
            if len(available_tools) >= 4 and (max_iter is None or max_iter == 3):
                warnings.append(
                    f"step {i} ({name or '?'}): llm_agent has {len(available_tools)} tools but max_iterations is {max_iter or 3}; consider setting max_iterations higher (e.g. 8–15) so the agent can use multiple tools."
                )
        if step_type in ("llm_agent", "deep_agent"):
            from orchestrator.engines.tool_resolution import skill_discovery_mode_from_step
            _step_tools = list(step.get("available_tools") or [])
            _step_skills = list(step.get("skill_ids") or step.get("skills") or [])
            if not _step_tools and not _step_skills and skill_discovery_mode_from_step(step) == "off":
                warnings.append(
                    f"step {i} ({name or '?'}): {step_type} has no available_tools and no pinned skill_ids, "
                    "and skill discovery is off — the agent will have no capabilities at runtime. "
                    "Add tools to available_tools, add skill_ids, or enable skill discovery "
                    "(set discovery_mode to 'auto', 'catalog', or 'full')."
                )
        if step_type in ("llm_agent", "deep_agent"):
            dm_raw = step.get("delegation_mode")
            if dm_raw is not None and str(dm_raw).strip():
                dm = str(dm_raw).strip().lower()
                if dm not in ("supervised", "parallel", "sequential"):
                    warnings.append(
                        f"step {i} ({name or '?'}): invalid delegation_mode '{dm_raw}' (use supervised, parallel, or sequential)"
                    )
            raw_sa = step.get("subagents")
            if raw_sa is not None and not isinstance(raw_sa, list):
                warnings.append(f"step {i} ({name or '?'}): subagents must be a list")
            elif isinstance(raw_sa, list):
                for sj, sa in enumerate(raw_sa):
                    if not isinstance(sa, dict):
                        warnings.append(f"step {i} ({name or '?'}): subagents[{sj}] must be an object")
                        continue
                    if not (sa.get("agent_profile_id") or "").strip():
                        warnings.append(
                            f"step {i} ({name or '?'}): subagents[{sj}] missing agent_profile_id"
                        )
            raw_samples = step.get("samples")
            if raw_samples is not None:
                try:
                    sn = int(raw_samples)
                    if sn < 1 or sn > 5:
                        warnings.append(
                            f"step {i} ({name or '?'}): samples must be an integer 1–5 (got {raw_samples!r})"
                        )
                except (TypeError, ValueError):
                    warnings.append(
                        f"step {i} ({name or '?'}): samples must be an integer 1–5 (got {raw_samples!r})"
                    )
            ss_raw = step.get("selection_strategy")
            if ss_raw is not None and str(ss_raw).strip():
                ss = str(ss_raw).strip().lower()
                if ss not in ("llm_judge", "highest_score"):
                    warnings.append(
                        f"step {i} ({name or '?'}): invalid selection_strategy {ss_raw!r} (use llm_judge or highest_score)"
                    )
                elif ss == "highest_score" and step_type != "deep_agent":
                    warnings.append(
                        f"step {i} ({name or '?'}): selection_strategy highest_score is only for deep_agent"
                    )
            fo = step.get("fan_out")
            if fo is not None:
                if step_type not in ("llm_agent", "deep_agent"):
                    warnings.append(f"step {i} ({name or '?'}): fan_out only on llm_agent or deep_agent")
                elif not isinstance(fo, dict):
                    warnings.append(f"step {i} ({name or '?'}): fan_out must be an object")
                elif not str(fo.get("source") or "").strip():
                    warnings.append(f"step {i} ({name or '?'}): fan_out.source is required")
        if step_type == "deep_agent":
            phases = step.get("phases")
            if not isinstance(phases, list):
                warnings.append(f"step {i} ({name or '?'}): deep_agent step requires 'phases' (list)")
            else:
                valid_phase_types = VALID_DEEP_AGENT_PHASE_TYPES
                for pj, phase in enumerate(phases):
                    if not isinstance(phase, dict):
                        warnings.append(f"step {i} ({name or '?'}): phase {pj} must be an object")
                        continue
                    pname = (phase.get("name") or "").strip()
                    ptype = (phase.get("type") or "").strip().lower()
                    if not ptype:
                        warnings.append(f"step {i} ({name or '?'}): phase {pj} ({pname or '?'}) missing 'type'")
                    elif ptype not in valid_phase_types:
                        warnings.append(f"step {i} ({name or '?'}): phase {pj} ({pname or '?'}) invalid type '{ptype}'")
                    if ptype == "evaluate" and not (phase.get("criteria") or "").strip():
                        warnings.append(f"step {i} ({name or '?'}): phase {pj} (evaluate) requires 'criteria'")
                    if ptype == "refine" and not (phase.get("target") or "").strip():
                        warnings.append(f"step {i} ({name or '?'}): phase {pj} (refine) requires 'target' (phase name)")
        if name:
            seen_names.add(name)
    _warn_missing_exclusive(steps, warnings)
    return warnings


# ── Tool functions ─────────────────────────────────────────────────────────

async def list_available_actions_tool(
    category: Optional[str] = None,
    query: Optional[str] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    List available actions/tools for playbook wiring. Use this first when composing playbooks.
    Returns name, category, description, input_fields, output_fields per action.
    """
    try:
        if category:
            registry = get_actions_by_category(category)
        else:
            registry = get_all_actions()
        actions = []
        for name, contract in registry.items():
            if query and query.strip():
                q = query.strip().lower()
                if q not in (contract.description or "").lower() and q not in name.lower() and q not in (contract.category or "").lower():
                    continue
            actions.append({
                "name": name,
                "category": contract.category,
                "description": contract.description,
                "input_fields": contract.get_input_fields(),
                "output_fields": contract.get_output_fields(),
            })
        count = len(actions)
        lines = [f"Found {count} action(s):"]
        for a in actions:
            lines.append(f"- **{a['name']}** ({a['category']}): {a.get('description', '')[:80]}")
        guide = (
            "\n\nCONFIRM-FLOW RULE: Tools that accept confirmed (e.g. update_playbook, create_playbook, create_agent_profile, update_agent_profile, delete_playbook, delete_agent_profile, assign_playbook_to_agent, set_agent_profile_status, create_agent_schedule, bind_data_source_to_agent, create_skill, propose_skill_update): (1) After showing a preview (confirmed=False), when the user approves in any later message (e.g. 'yes', 'go ahead', 'apply'), you MUST call the same tool again with the same arguments and confirmed=True in your very next response—before replying in text. (2) Use the same playbook_id and updates (or other arguments) from your earlier tool call in this conversation. (3) Do not tell the user the change is done until you have called with confirmed=True and received a success response. Do not only acknowledge in text—make the call.\n"
            "\nPLAYBOOK DEFINITION SHAPE (same for create_playbook and update_playbook):\n"
            "- Playbook definition: {\"steps\": [...], \"run_context\": \"interactive\" | \"background\"}. Steps is a list of step objects.\n"
            "- Step fields (required per type: step_type, output_key; tool needs action; llm_task needs prompt/prompt_template; llm_agent has optional available_tools, may be empty for toolless steps; deep_agent needs phases): name, step_type, output_key, action (tool), inputs, params, prompt, prompt_template, condition, branch_condition, then_steps, else_steps, parallel_steps, steps (loop), available_tools, max_iterations, system_prompt_additions, phases (deep_agent), skill_ids, discovery_mode, max_discovered_skills, model_override, output_schema, timeout_minutes, on_reject, subagents, delegation_mode, samples, selection_strategy, selection_criteria, fan_out.\n"
            "- subagents (llm_agent, deep_agent): optional list of {agent_profile_id, playbook_id?, role?, accepts?, returns?}. Adds delegate_subagent_* tools and shared scratchpad. delegation_mode: supervised (default, LLM delegates via tools), parallel (pre-run all subagents then synthesize), sequential (pre-run subagents in order then synthesize).\n"
            "- samples (llm_agent, deep_agent): optional integer 1–5 (default 1). Runs the step N times independently with raised temperature and selects the best result. selection_strategy: llm_judge (default, LLM picks best) or highest_score (deep_agent only, uses last evaluate phase score from phase_trace; falls back to llm_judge if no scores). selection_criteria: optional string for the judge when using llm_judge.\n"
            "- fan_out (llm_agent, deep_agent): optional object {source, item_variable?, max_items?, merge?}. Reads a list from playbook_state (source is a dot-path like plan.items), runs the step once per item in parallel (capped by max_items, default 10), merges results. item_variable (default current_item) is injected into playbook_state and inputs for each copy — use {current_item} in prompts. merge: list (default: items array plus formatted join) or concat (formatted sections per item).\n"
            "- For get_playbook_detail: when the response includes a 'Full definition' JSON block, use that exact structure in update_playbook (with your edits) to avoid dropping fields.\n"
            "\nPLAYBOOK WIRING RULES:\n"
            "- Use exactly ONE opening brace and ONE closing brace: {output_key.field}. If you use {{double}} the ref breaks (system will see unknown step). Exception: {{#var}}...{{/var}} is valid for conditional blocks.\n"
            "- Conditional blocks: wrap a section in {{#var}}...{{/var}}; the section is included only when var is non-empty. Example: {{#editor}}Document: {editor}{{/editor}} shows the document block only when a file is open. Expression conditionals: {{#editor_length > 5000}}...{{/editor_length > 5000}} or {{#search_docs.count > 0}}...{{/search_docs.count > 0}} (step output fields resolve from playbook state). In branch_condition you can use: 'X is defined' / 'X is not defined'; 'X matches \"regex\"'; AND / OR (e.g. {editor_document_type} == \"fiction\" AND {editor_selection} is defined).\n"
            "- Tool step inputs: wire with {output_key.field} e.g. \"start_datetime\": \"{get_date.formatted}\" or \"{today}\" for literals.\n"
            "- LLM task prompt strings: embed refs with single braces e.g. \"Weather: {get_weather.formatted}, Cal: {fetch_cal.formatted}\".\n"
            "- Valid runtime variables: today, today_end, tomorrow, today_day_of_week, query, query_length, history, user_weather_location, trigger_input, editor, editor_refs, editor_document_id, editor_filename, editor_length, editor_document_type, editor_cursor_offset, editor_selection, editor_current_section, editor_current_heading, editor_previous_section, editor_next_section, editor_section_index, editor_adjacent_sections, editor_total_sections, editor_toc, editor_is_first_section, editor_is_last_section, editor_ref_count, document_context, pinned_document_id, last_tool_results, profile, and any editor_refs_CATEGORY (e.g. editor_refs_rules, editor_refs_style). Do NOT invent other names — use upstream output_key (e.g. {fetch_todos.formatted}).\n"
            "- editor: full content of the currently open file (filename + content block). Empty if no file is open.\n"
            "- editor_filename: base filename of the open file (e.g. chapter_01.md). Empty if no file open.\n"
            "- editor_length: character count of the open file body (number as string). Use in conditionals: {{#editor_length > 5000}}...{{/editor_length > 5000}} to include a block only when the document is over 5000 characters; use {{#editor_length < 5000}} for short documents.\n"
            "- query_length: character count of the user's message (number as string). Use {{#query_length > 200}} for long-query prompts.\n"
            "- today_day_of_week: day name (Monday, Tuesday, etc.) in user timezone. Empty if datetime context disabled.\n"
            "- editor_is_first_section: \"true\" when cursor is in the first section, else empty. Use {{#editor_is_first_section}} to show content only at start of doc.\n"
            "- editor_is_last_section: \"true\" when cursor is in the last section, else empty.\n"
            "- editor_ref_count: number of loaded ref_* files (number as string). Use {{#editor_ref_count > 0}} when refs exist.\n"
            "- last_tool_results: JSON of the most recent tool results from the conversation. Empty if none.\n"
            "- document_context: full content of the pinned document (when one is set). Empty if no pin.\n"
            "- pinned_document_id: document ID of the pinned document. Empty if no pin.\n"
            "- editor_refs: all referenced files from the open file's frontmatter (see ref_* below). Empty if no refs or no file open.\n"
            "- editor_refs_CATEGORY: frontmatter keys prefixed with ref_ (e.g. ref_rules: ./rules.md, ref_style: ./style.md) are loaded and exposed as {editor_refs_rules}, {editor_refs_style}, etc. Use {{#editor_refs_rules}}...{{/editor_refs_rules}} to show a section only when that ref exists.\n"
            "- editor_refs_<prefix>*: use a trailing * to include all refs whose category starts with the prefix (e.g. {editor_refs_character_*} includes editor_refs_character_adam, editor_refs_character_betty, etc.). Works in prompts and in conditional blocks: {{#editor_refs_character_*}}...{{/editor_refs_character_*}}.\n"
            "- editor_refs_CATEGORY_previous, editor_refs_CATEGORY_next: section before/after the cursor-matched section in that ref file. Same category names as editor_refs_CATEGORY.\n"
            "- editor_document_id: document ID of the open file. Use as document_id when calling propose_document_edit or update_document_content. Empty if no file open.\n"
            "- editor_document_type: frontmatter type of the open file (e.g. fiction, outline), lowercased. Use in branch_condition: '{editor_document_type} == \"fiction\"' to route by document type.\n"
            "- editor_cursor_offset: character offset of the cursor in the open file (-1 if none).\n"
            "- editor_selection: selected text in the open file (content between selection_start and selection_end). Empty if no selection.\n"
            "- editor_current_section: content of the markdown section (## heading) containing the cursor. Empty if no sections or no file.\n"
            "- editor_previous_section: content of the section immediately before the current one. Empty if cursor is in the first section.\n"
            "- editor_next_section: content of the section immediately after the current one. Empty if cursor is in the last section.\n"
            "- editor_current_heading: heading text of the section containing the cursor (e.g. \"## Chapter 3\").\n"
            "- editor_section_index: 0-based index of the section containing the cursor. Use with editor_total_sections for context.\n"
            "- editor_adjacent_sections: concatenated content of previous + current + next sections (cursor-scoped context). Use {{#editor_adjacent_sections}}...{{/editor_adjacent_sections}} in prompts to give the LLM focused context without the full file.\n"
            "- editor_total_sections: total number of ## (or configured) sections in the file.\n"
            "- profile: user's profile (name, email, timezone, ZIP, AI context). Empty if not used or load fails.\n"
            "- For dates: use {today}, {today_end}, {tomorrow} only. No {today_iso} or similar.\n"
            "- output_key is the key for wiring: reference upstream steps as {output_key.field}. Step display name is irrelevant; use the output_key from the step definition.\n"
            "- action field: use the exact name from this list (no _tool suffix). Example: \"action\": \"get_my_profile\", NOT \"action\": \"get_my_profile_tool\".\n"
            "- llm_task output fields: only .formatted and .raw exist by default. To use .summary or similar, declare it in output_schema.properties. Never reference .content, .result, .text unless declared there.\n"
            "- Runtime vars in strings: {today}, {today_end}, {tomorrow} are valid anywhere, including mixed with literals (e.g. \"Briefing for {today}\"). Do NOT use {today_iso}, {current_date}, or similar.\n"
            "- output_key must match ref prefix: the output_key you set on a step is the ONLY valid prefix in downstream refs. If output_key is \"fetch_cal\", wire as {fetch_cal.field}. The step name (display label) cannot be used in refs.\n"
            "- ALWAYS WRONG: {{today_iso}}, {{weather_report}}, {{summary_text}} — double braces (except {{#var}} and {{/var}}) and invented names are never valid. "
            "CORRECT: {today} for today's date, {my_step.formatted} for an upstream step's output (use its output_key). "
            "There is no {{today_iso}} — use {today}. There is no {{weather_report}} — use {get_weather.formatted}.\n"
            "- step_type must be EXACTLY one of: tool, llm_task, llm_agent, approval, loop, parallel, branch, deep_agent, browser_authenticate. "
            "Never use tool_call, llm_call, llm_analysis, analysis, or any other name.\n"
            "- llm_agent = ONE step where the LLM runs a ReAct loop and may call tools (which tools and how many times is decided by the LLM, up to max_iterations). Use llm_agent when the task needs dynamic tool choice or multi-step reasoning; use multiple tool steps when the sequence is fixed (e.g. get_weather -> llm_task -> send_email).\n"
            "- llm_agent available_tools: list of action names the agent can call (e.g. [\"send_email\", \"list_todos\"]). "
            "May be empty for conversational/reasoning-only steps (toolless agent: one LLM call, no tool loop).\n"
            "- deep_agent = ONE step that runs a multi-phase LangGraph workflow. Requires a \"phases\" list; each phase has type "
            "(reason, act, search, evaluate, synthesize, refine, rerank), name, and optional prompt/available_tools/search_tools/criteria/target/source_phase/top_n. "
            "rerank: optional phase after search; uses rerank_documents from the tool palette; source_phase names the search phase whose raw_results to rerank (or defaults to the latest search with raw_results). "
            "act: optional per-phase available_tools narrows the ReAct mini-loop to a subset of the step's resolved tool palette (omit or leave empty to inherit the full palette). "
            "Step-level output_phase (name of a phase) and output_template (string with {phase.field} refs) control the formatted string returned by the step; output_template wins when both are set.\n"
            "\nSKILLS & DISCOVERY (llm_agent and deep_agent steps):\n"
            "- skill_ids: list of skill UUID strings. Call list_skills first and use each skill's id field (not name or slug). Alias \"skills\" is accepted. Pinned skills are always attached regardless of discovery_mode.\n"
            "- discovery_mode (or skill_discovery_mode): \"off\" | \"auto\" | \"catalog\" | \"full\". If omitted, effective mode is typically auto (same as Workflow Composer default).\n"
            "  * off — only pinned skill_ids; no automatic skill attachment from the catalog or search.\n"
            "  * auto — before the step runs, semantic search matches skills to the step prompt and attaches up to max_discovered_skills; recommended default for most agents.\n"
            "  * catalog — injects a compressed catalog of core skills into the system prompt so the model knows what procedures exist; use when the agent should choose procedures by name.\n"
            "  * full — catalog injection plus the model may call search_and_acquire_skills during the ReAct loop to attach more skills at runtime; use for open-ended or multi-domain work.\n"
            "- max_discovered_skills: integer 1–10 (default 3); caps how many skills auto/catalog/full discovery adds in the pre-run phase.\n"
            "- Workflow: list_skills (or get_skill_detail) to pick UUIDs → set skill_ids on the step; ensure available_tools covers tools required by those skills or enable discovery.\n"
            "\nEXAMPLE PLAYBOOK (morning briefing pattern):\n"
            "Step 1: step_type: tool, action: get_my_profile, output_key: user_profile, inputs: {}\n"
            "Step 2: step_type: tool, action: get_weather, output_key: weather_data, inputs: {data_types: \"current,forecast\"}\n"
            "Step 3: step_type: tool, action: get_calendar_events, output_key: cal_events, inputs: {start_datetime: \"{today}\", end_datetime: \"{today_end}\"}\n"
            "Step 4: step_type: llm_task, output_key: briefing, prompt: \"Weather: {weather_data.formatted}\\nCalendar: {cal_events.formatted}\\nCompose a morning briefing.\"\n"
            "Step 5: step_type: tool, action: send_email, output_key: delivery, inputs: {to: \"{user_profile.email}\", subject: \"Briefing for {today}\", body: \"{briefing.formatted}\", confirmed: true}\n"
            "  (Alternative: action: send_channel_message, inputs: {channel: \"email\", to_email: \"{user_profile.email}\", subject: \"Briefing for {today}\", message: \"{briefing.formatted}\"})\n"
            "\nEXAMPLE (llm_agent with pinned skill + discovery; replace UUID via list_skills):\n"
            "Step: step_type: llm_agent, name: research, output_key: research_result, "
            "available_tools: [\"search_documents\", \"search_web\"], "
            "skill_ids: [\"<uuid-from-list_skills>\"], discovery_mode: \"auto\", max_discovered_skills: 3, max_iterations: 8, "
            "prompt: \"Research the following topic and summarize findings: {query}\"\n"
        )
        formatted = "\n".join(lines) + guide
        return {"actions": actions, "count": count, "formatted": formatted}
    except Exception as e:
        logger.error("list_available_actions_tool error: %s", e)
        return {"actions": [], "count": 0, "formatted": f"Error listing actions: {e}"}


async def validate_playbook_wiring_tool(
    definition: Any = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Validate a playbook definition (steps, refs, types). Call before create_playbook or after building definition for update_playbook."""
    try:
        if definition is None:
            definition = {}
        if isinstance(definition, str):
            try:
                definition = json.loads(definition)
            except json.JSONDecodeError:
                return {
                    "issues": [],
                    "count": 0,
                    "formatted": f"Invalid JSON for definition: pass a playbook object with 'steps' or a list of step objects.",
                }
        if isinstance(definition, list):
            steps = definition
        else:
            steps = (definition or {}).get("steps") if isinstance(definition, dict) else []
            if steps is None:
                steps = []
        if not steps:
            return {
                "issues": [],
                "count": 0,
                "formatted": "No steps to validate. Pass a playbook definition with 'steps' or a list of step objects.",
            }
        registry = get_all_actions()
        actions_by_name = {}
        for name, contract in registry.items():
            actions_by_name[name] = {
                "name": name,
                "input_fields": contract.get_input_fields(),
                "output_fields": contract.get_output_fields(),
            }
        issues = _validate_playbook_wiring(steps, actions_by_name)
        count = len(issues)
        if not issues:
            formatted = f"Wiring valid: {len(steps)} step(s), no issues."
        else:
            lines = [f"Wiring issues ({count}):"]
            for i in issues:
                step_name = i.get("step_name", "?")
                input_key = i.get("input_key", "?")
                msg = i.get("message", "")
                lines.append(f"- {step_name}.{input_key}: {msg}")
            formatted = "\n".join(lines)
        return {"issues": issues, "count": count, "formatted": formatted}
    except Exception as e:
        logger.error("validate_playbook_wiring_tool error: %s", e)
        return {"issues": [], "count": 0, "formatted": f"Validation error: {e}"}


async def create_agent_profile_tool(
    name: str,
    handle: Optional[str] = None,
    description: Optional[str] = None,
    model_preference: Optional[str] = None,
    system_prompt_additions: Optional[str] = None,
    persona_enabled: bool = False,
    auto_routable: bool = False,
    prompt_history_enabled: bool = False,
    chat_visible: bool = True,
    confirmed: bool = False,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Create ONE agent profile. Call once with confirmed=False to preview, present to the user, wait for approval, then call once with confirmed=True. Do not call multiple times for the same agent.
    If handle is omitted or empty, the agent is schedule/Run-only (not @mentionable in chat).
    """
    handle_val = (handle or "").strip() or None
    if not confirmed:
        mention_part = f" (@{handle_val})" if handle_val else " (schedule/Run-only, no @handle)"
        formatted = f"[Preview] Would create agent profile: **{name}**{mention_part} (paused by default)"
        if description:
            formatted += f"\nDescription: {description[:200]}"
        if model_preference:
            formatted += f"\nModel: {model_preference}"
        formatted += "\nWhen the user approves, you MUST call create_agent_profile again with the same arguments and confirmed=True. Do not only ask — make that call."
        return {
            "agent_id": "",
            "name": name,
            "handle": handle_val,
            "is_draft": True,
            "formatted": formatted,
        }
    try:
        client = await get_backend_tool_client()
        result = await client.create_agent_profile(
            user_id=user_id,
            name=name,
            handle=handle_val,
            description=description,
            model_preference=model_preference,
            system_prompt_additions=system_prompt_additions,
            persona_enabled=persona_enabled,
            auto_routable=auto_routable,
            chat_history_enabled=prompt_history_enabled,
            chat_visible=chat_visible,
            is_active=False,
        )
        if not result.get("success"):
            return {
                "agent_id": "",
                "name": name,
                "handle": handle_val,
                "is_draft": False,
                "formatted": result.get("formatted") or result.get("error", "Create failed"),
            }
        created_handle = result.get("handle", handle_val)
        done_msg = f"Created agent {name} (@{created_handle}) (paused)" if created_handle else f"Created agent {name} (schedule/Run-only) (paused)"
        return {
            "agent_id": result.get("agent_id", ""),
            "name": result.get("name", name),
            "handle": created_handle,
            "is_draft": False,
            "formatted": result.get("formatted", f"{done_msg} — ID: {result.get('agent_id', '')}. Use set_agent_profile_status to activate."),
        }
    except Exception as e:
        logger.error("create_agent_profile_tool error: %s", e)
        return {"agent_id": "", "name": name, "handle": handle_val, "is_draft": False, "formatted": f"Error: {e}"}


async def create_playbook_tool(
    name: str,
    steps: List[Dict[str, Any]],
    description: Optional[str] = None,
    run_context: str = "background",
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    confirmed: bool = False,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Create a new playbook. Pass name and steps (full step objects). Same step shape as get_playbook_detail / update_playbook. confirmed=False preview; confirmed=True create."""
    definition = {"steps": steps, "run_context": run_context}
    validate_warnings = _validate_playbook_steps(steps)
    registry = get_all_actions()
    actions_by_name = {
        n: {"name": n, "input_fields": c.get_input_fields(), "output_fields": c.get_output_fields()}
        for n, c in registry.items()
    }
    wiring_errors = _validate_playbook_wiring(steps if isinstance(steps, list) else [], actions_by_name)

    if not confirmed:
        step_count = len(steps) if isinstance(steps, list) else 0
        formatted = f"[Preview] Would create playbook: **{name}** ({step_count} steps)"
        from orchestrator.engines.tool_resolution import skill_discovery_mode_from_step

        agent_step_previews = []
        for s in steps if isinstance(steps, list) else []:
            if not isinstance(s, dict):
                continue
            stype = s.get("step_type")
            if stype not in ("llm_agent", "deep_agent"):
                continue
            label = s.get("name") or s.get("output_key") or "?"
            n_skills = len(list(s.get("skill_ids") or s.get("skills") or []))
            dm = skill_discovery_mode_from_step(s)
            parts = [f"skill_ids={n_skills}", f"discovery_mode={dm}"]
            if stype == "llm_agent":
                mi = s.get("max_iterations")
                parts.insert(0, f"max_iterations={mi if mi is not None else '3 (default)'}")
            elif s.get("max_iterations") is not None:
                parts.insert(0, f"max_iterations={s.get('max_iterations')}")
            agent_step_previews.append(f"{stype} {label} ({', '.join(parts)})")
        if agent_step_previews:
            formatted += "\nAgent steps (llm_agent / deep_agent): " + "; ".join(agent_step_previews)
        if validate_warnings:
            formatted += "\nValidation warnings: " + "; ".join(validate_warnings[:5])
        if wiring_errors:
            formatted += f"\nWiring errors ({len(wiring_errors)}): " + "; ".join(
                f"{e['step_name']}.{e['input_key']}: {e['message']}" for e in wiring_errors[:5]
            )
        formatted += "\nWhen the user approves, you MUST call create_playbook again with the same name and steps and confirmed=True. Do not only ask — make that call."
        return {
            "playbook_id": "",
            "name": name,
            "step_count": step_count,
            "validation_warnings": validate_warnings,
            "formatted": formatted,
        }
    if wiring_errors:
        lines = [f"Cannot create playbook — {len(wiring_errors)} wiring error(s):"]
        for e in wiring_errors[:8]:
            lines.append(f"- {e['step_name']}.{e['input_key']}: {e['message']}")
        lines.append("Fix these issues and try again.")
        return {
            "playbook_id": "",
            "name": name,
            "step_count": len(steps) if isinstance(steps, list) else 0,
            "validation_warnings": [e["message"] for e in wiring_errors],
            "formatted": "\n".join(lines),
        }
    try:
        client = await get_backend_tool_client()
        result = await client.create_playbook(
            user_id=user_id,
            name=name,
            definition=definition,
            description=description,
            run_context=run_context,
            category=category,
            tags=tags or [],
        )
        if not result.get("success"):
            return {
                "playbook_id": "",
                "name": name,
                "step_count": len(steps) if isinstance(steps, list) else 0,
                "validation_warnings": [],
                "formatted": result.get("formatted") or result.get("error", "Create failed"),
            }
        return {
            "playbook_id": result.get("playbook_id", ""),
            "name": result.get("name", name),
            "step_count": result.get("step_count", len(steps) if isinstance(steps, list) else 0),
            "validation_warnings": result.get("validation_warnings", []),
            "formatted": result.get("formatted", f"Created playbook {name} — ID: {result.get('playbook_id', '')}"),
        }
    except Exception as e:
        logger.error("create_playbook_tool error: %s", e)
        return {
            "playbook_id": "",
            "name": name,
            "step_count": len(steps) if isinstance(steps, list) else 0,
            "validation_warnings": [],
            "formatted": f"Error: {e}",
        }


async def assign_playbook_to_agent_tool(
    agent_id: str,
    playbook_id: str,
    confirmed: bool = False,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Assign a playbook to an agent as its default. confirmed=False preview; confirmed=True apply."""
    if not confirmed:
        return {
            "success": False,
            "formatted": f"[Preview] Would assign playbook {playbook_id} to agent {agent_id}. When the user approves, call assign_playbook_to_agent again with the same arguments and confirmed=True. Do not only ask — make that call.",
        }
    try:
        client = await get_backend_tool_client()
        result = await client.assign_playbook_to_agent(user_id=user_id, agent_id=agent_id, playbook_id=playbook_id)
        return {"success": result.get("success", False), "formatted": result.get("formatted", result.get("error", ""))}
    except Exception as e:
        logger.error("assign_playbook_to_agent_tool error: %s", e)
        return {"success": False, "formatted": f"Error: {e}"}


async def set_agent_profile_status_tool(
    agent_id: str,
    is_active: bool,
    confirmed: bool = False,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Pause or activate an agent profile. Separate capability from creating agents. Call with confirmed=False first, then confirmed=True to apply."""
    if not confirmed:
        status = "active" if is_active else "paused"
        return {
            "success": False,
            "is_active": is_active,
            "formatted": f"[Preview] Would set agent {agent_id} to {status}. When the user approves, call set_agent_profile_status again with the same arguments and confirmed=True. Do not only ask — make that call.",
        }
    try:
        client = await get_backend_tool_client()
        result = await client.set_agent_profile_status(user_id=user_id, agent_id=agent_id, is_active=is_active)
        return {
            "success": result.get("success", False),
            "is_active": result.get("is_active", is_active),
            "formatted": result.get("formatted", result.get("error", "")),
        }
    except Exception as e:
        logger.error("set_agent_profile_status_tool error: %s", e)
        return {"success": False, "is_active": False, "formatted": f"Error: {e}"}


async def create_agent_schedule_tool(
    agent_id: str,
    schedule_type: str,
    cron_expression: Optional[str] = None,
    interval_seconds: Optional[int] = None,
    timezone: str = "UTC",
    is_active: bool = False,
    confirmed: bool = False,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Create a schedule for an agent (cron or interval). Call with confirmed=False first, then confirmed=True. Schedules start paused (is_active=False) unless set."""
    if not confirmed:
        formatted = f"[Preview] Would create {schedule_type} schedule for agent {agent_id}"
        if schedule_type == "cron" and cron_expression:
            formatted += f" — cron: {cron_expression}, tz: {timezone}"
        elif schedule_type == "interval" and interval_seconds:
            formatted += f" — every {interval_seconds}s"
        formatted += f", active: {is_active}. When the user approves, call create_agent_schedule again with the same arguments and confirmed=True. Do not only ask — make that call."
        return {"schedule_id": "", "next_run_at": "", "is_active": is_active, "formatted": formatted}
    try:
        client = await get_backend_tool_client()
        result = await client.create_agent_schedule(
            user_id=user_id,
            agent_id=agent_id,
            schedule_type=schedule_type,
            cron_expression=cron_expression,
            interval_seconds=interval_seconds,
            timezone=timezone,
            is_active=is_active,
        )
        return {
            "schedule_id": result.get("schedule_id", ""),
            "next_run_at": result.get("next_run_at", ""),
            "is_active": result.get("is_active", False),
            "formatted": result.get("formatted", result.get("error", "")),
        }
    except Exception as e:
        logger.error("create_agent_schedule_tool error: %s", e)
        return {"schedule_id": "", "next_run_at": "", "is_active": False, "formatted": f"Error: {e}"}


async def list_agent_schedules_tool(
    agent_id: str,
    user_id: str = "system",
) -> Dict[str, Any]:
    """List schedules for an agent profile. Use before creating schedules to avoid duplicates."""
    try:
        client = await get_backend_tool_client()
        result = await client.list_agent_schedules(user_id=user_id, agent_id=agent_id)
        if not result.get("success"):
            return {"schedules": [], "count": 0, "formatted": result.get("formatted") or result.get("error", "List failed")}
        schedules = result.get("schedules", [])
        return {"schedules": schedules, "count": len(schedules), "formatted": result.get("formatted", f"Found {len(schedules)} schedule(s).")}
    except Exception as e:
        logger.error("list_agent_schedules_tool error: %s", e)
        return {"schedules": [], "count": 0, "formatted": str(e)}


async def list_agent_data_sources_tool(
    agent_id: str,
    user_id: str = "system",
) -> Dict[str, Any]:
    """List data source bindings for an agent profile. Use to see which connectors are bound to an agent."""
    try:
        client = await get_backend_tool_client()
        result = await client.list_agent_data_sources(user_id=user_id, agent_id=agent_id)
        if not result.get("success"):
            return {"bindings": [], "count": 0, "formatted": result.get("formatted") or result.get("error", "List failed")}
        bindings = result.get("bindings", [])
        return {"bindings": bindings, "count": len(bindings), "formatted": result.get("formatted", f"Found {len(bindings)} binding(s).")}
    except Exception as e:
        logger.error("list_agent_data_sources_tool error: %s", e)
        return {"bindings": [], "count": 0, "formatted": str(e)}


async def bind_data_source_to_agent_tool(
    agent_id: str,
    connector_id: str,
    config_overrides: Optional[Dict[str, Any]] = None,
    confirmed: bool = False,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Bind a data source connector to an agent. Call with confirmed=False first, then confirmed=True after approval."""
    if not confirmed:
        return {
            "binding_id": "",
            "formatted": f"[Preview] Would bind connector {connector_id} to agent {agent_id}. When the user approves, call bind_data_source_to_agent again with the same arguments and confirmed=True. Do not only ask — make that call.",
        }
    try:
        client = await get_backend_tool_client()
        result = await client.bind_data_source_to_agent(
            user_id=user_id,
            agent_id=agent_id,
            connector_id=connector_id,
            config_overrides=config_overrides,
        )
        return {"binding_id": result.get("binding_id", ""), "formatted": result.get("formatted", result.get("error", ""))}
    except Exception as e:
        logger.error("bind_data_source_to_agent_tool error: %s", e)
        return {"binding_id": "", "formatted": f"Error: {e}"}


async def update_agent_profile_tool(
    agent_id: str,
    updates: Optional[Dict[str, Any]] = None,
    confirmed: bool = False,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Update an existing agent profile. confirmed=False is a dry-run (no changes). When profile is locked only is_active and is_locked can be changed.
    """
    if not confirmed:
        return {
            "success": False,
            "agent_id": agent_id,
            "name": "",
            "formatted": f"[Preview] Would update agent profile {agent_id} with: {list((updates or {}).keys()) or 'no changes'}. When the user approves, call update_agent_profile again with the same arguments and confirmed=True. Do not only ask — make that call.",
        }
    try:
        client = await get_backend_tool_client()
        result = await client.update_agent_profile(user_id=user_id, agent_id=agent_id, updates=updates or {})
        return {
            "success": result.get("success", False),
            "agent_id": result.get("agent_id", agent_id),
            "name": result.get("name", ""),
            "formatted": result.get("formatted", result.get("error", "Update failed")),
        }
    except Exception as e:
        logger.error("update_agent_profile_tool error: %s", e)
        return {"success": False, "agent_id": agent_id, "name": "", "formatted": f"Error: {e}"}


async def delete_agent_profile_tool(
    agent_id: str,
    confirmed: bool = False,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Delete an agent profile. confirmed=False is a dry-run. Delete is irreversible. Blocked when profile is locked; unlock first.
    """
    if not confirmed:
        return {
            "success": False,
            "formatted": f"[Preview] Would delete agent profile {agent_id}. This is irreversible. When the user approves, call delete_agent_profile again with the same agent_id and confirmed=True. Do not only ask — make that call.",
        }
    try:
        client = await get_backend_tool_client()
        result = await client.delete_agent_profile(user_id=user_id, agent_id=agent_id)
        return {
            "success": result.get("success", False),
            "formatted": result.get("formatted", result.get("error", "Delete failed")),
        }
    except Exception as e:
        logger.error("delete_agent_profile_tool error: %s", e)
        return {"success": False, "formatted": f"Error: {e}"}


async def update_playbook_tool(
    playbook_id: str,
    updates: Optional[Dict[str, Any]] = None,
    confirmed: bool = False,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Update a playbook. Pass playbook_id and updates (e.g. name, description, definition with steps). Send full definition.steps when changing steps; backend merges in missing fields. confirmed=False preview; confirmed=True apply."""
    if not confirmed:
        return {
            "success": False,
            "playbook_id": playbook_id,
            "name": "",
            "step_count": 0,
            "validation_warnings": [],
            "formatted": (
                "[Preview] No changes have been saved yet. Would update playbook "
                f"{playbook_id} with: {list((updates or {}).keys()) or 'no changes'}. "
                "When the user says yes/approve (in this or a later message), you MUST call update_playbook again with the same playbook_id and updates and confirmed=True before replying that it is done. Use the same updates from this call."
            ),
        }
    try:
        client = await get_backend_tool_client()
        result = await client.update_playbook(user_id=user_id, playbook_id=playbook_id, updates=updates or {})
        formatted = result.get("formatted", result.get("error", "Update failed"))
        if result.get("success"):
            formatted = f"{formatted}\n\nChanges saved to playbook."
        return {
            "success": result.get("success", False),
            "playbook_id": result.get("playbook_id", playbook_id),
            "name": result.get("name", ""),
            "step_count": result.get("step_count", 0),
            "validation_warnings": result.get("validation_warnings", []),
            "formatted": formatted,
        }
    except Exception as e:
        logger.error("update_playbook_tool error: %s", e)
        return {
            "success": False,
            "playbook_id": playbook_id,
            "name": "",
            "step_count": 0,
            "validation_warnings": [],
            "formatted": f"Error: {e}",
        }


async def delete_playbook_tool(
    playbook_id: str,
    confirmed: bool = False,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Delete a playbook. confirmed=False preview; confirmed=True delete. Blocked if locked or template."""
    if not confirmed:
        return {
            "success": False,
            "formatted": f"[Preview] Would delete playbook {playbook_id}. This is irreversible. When the user approves, call delete_playbook again with the same playbook_id and confirmed=True. Do not only ask — make that call.",
        }
    try:
        client = await get_backend_tool_client()
        result = await client.delete_playbook(user_id=user_id, playbook_id=playbook_id)
        return {
            "success": result.get("success", False),
            "formatted": result.get("formatted", result.get("error", "Delete failed")),
        }
    except Exception as e:
        logger.error("delete_playbook_tool error: %s", e)
        return {"success": False, "formatted": f"Error: {e}"}


async def list_playbooks_tool(user_id: str = "system") -> Dict[str, Any]:
    """List playbooks and templates (id, name, description). Use to find playbook_id before get_playbook_detail or update_playbook."""
    try:
        client = await get_backend_tool_client()
        result = await client.list_playbooks(user_id=user_id)
        if not result.get("success"):
            return {"playbooks": [], "count": 0, "formatted": result.get("formatted") or result.get("error", "List failed")}
        playbooks = result.get("playbooks", [])
        lines = [f"Found {len(playbooks)} playbook(s):"]
        for p in playbooks:
            name = p.get("name") or "(unnamed)"
            pid = p.get("id") or ""
            lines.append(f"  - **{name}** (id: {pid})")
        formatted = "\n".join(lines) if playbooks else lines[0]
        return {"playbooks": playbooks, "count": len(playbooks), "formatted": formatted}
    except Exception as e:
        logger.error("list_playbooks_tool error: %s", e)
        return {"playbooks": [], "count": 0, "formatted": str(e)}


async def get_playbook_detail_tool(
    playbook_id: str,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Get one playbook by id: metadata plus full definition (steps, run_context). When small enough, response includes full JSON to copy into update_playbook. Always call before editing an existing playbook."""
    try:
        client = await get_backend_tool_client()
        playbook = await client.get_playbook(user_id=user_id, playbook_id=playbook_id)
        if not playbook:
            return {
                "playbook_id": playbook_id,
                "name": "",
                "description": None,
                "version": None,
                "definition": {},
                "triggers": [],
                "category": None,
                "tags": [],
                "step_count": 0,
                "is_template": False,
                "is_locked": False,
                "formatted": f"Playbook not found: {playbook_id}",
            }
        definition = playbook.get("definition") or {}
        steps = definition.get("steps") or []
        run_context = definition.get("run_context") or "interactive"
        triggers = playbook.get("triggers") or []
        tags = list(playbook.get("tags") or [])
        max_prompt_chars = 800
        lines = [
            f"**{playbook.get('name', '')}** (ID: {playbook.get('id', playbook_id)})",
            f"Description: {playbook.get('description') or '(none)'}",
            f"run_context: {run_context}",
            f"Steps: {len(steps)}",
            "When updating, preserve all step fields you are not editing (prompt, prompt_template, inputs, params, condition, then_steps, else_steps, parallel_steps, phases, available_tools, max_iterations, skill_ids, discovery_mode, max_discovered_skills, etc.).",
        ]
        if triggers:
            lines.append(f"Triggers: {len(triggers)} pattern(s)")
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            name = step.get("name") or step.get("output_key") or f"Step {i + 1}"
            step_type = step.get("step_type") or "tool"
            output_key = step.get("output_key") or ""
            cond = step.get("condition") or step.get("branch_condition")
            cond_str = f" [condition: {cond}]" if cond else ""
            action_str = f" action={step.get('action')}" if step_type == "tool" else ""
            extra = []
            if step.get("max_iterations") is not None:
                extra.append(f"max_iterations={step.get('max_iterations')}")
            if step.get("params"):
                extra.append("params set")
            if step.get("skill_ids") or step.get("skills"):
                extra.append("skill_ids set")
            if step.get("then_steps") or step.get("else_steps"):
                extra.append("branch")
            if step.get("parallel_steps"):
                extra.append("parallel")
            if step.get("phases"):
                extra.append(f"phases={len(step.get('phases', []))}")
            extra_str = " " + ", ".join(extra) if extra else ""
            lines.append(f"  {i + 1}. {name} ({step_type}, output_key={output_key}{action_str}){cond_str}{extra_str}")
            prompt = step.get("prompt") or step.get("prompt_template") or ""
            if isinstance(prompt, str) and prompt.strip():
                snippet = prompt.strip().replace("\n", " ")[:max_prompt_chars]
                if len(prompt.strip()) > max_prompt_chars:
                    snippet += "..."
                lines.append(f"      prompt: {snippet}")
            if step_type == "llm_agent":
                if step.get("available_tools"):
                    lines.append(f"      available_tools: {step.get('available_tools')}")
                if (step.get("system_prompt_additions") or "").strip():
                    sp = (step.get("system_prompt_additions") or "").strip().replace("\n", " ")[:max_prompt_chars]
                    lines.append(f"      system_prompt_additions: {sp}")
        formatted = "\n".join(lines)
        try:
            definition_json = json.dumps(definition, separators=(",", ":"))
        except (TypeError, ValueError):
            definition_json = "{}"
        if len(definition_json) <= GET_PLAYBOOK_FULL_DEFINITION_MAX_CHARS:
            formatted += "\n\nFull definition (copy this when calling update_playbook to preserve all fields):\n```json\n"
            formatted += definition_json
            formatted += "\n```"
        else:
            formatted += "\n\nDefinition too large to show in full; only summary above is shown. When calling update_playbook, send the exact steps you want (backend will merge missing fields from the existing playbook)."
        return {
            "playbook_id": str(playbook.get("id", playbook_id)),
            "name": playbook.get("name", ""),
            "description": playbook.get("description"),
            "version": playbook.get("version"),
            "definition": definition,
            "triggers": triggers,
            "category": playbook.get("category"),
            "tags": tags,
            "step_count": len(steps),
            "is_template": playbook.get("is_template", False),
            "is_locked": playbook.get("is_locked", False),
            "formatted": formatted,
        }
    except Exception as e:
        logger.error("get_playbook_detail_tool error: %s", e)
        return {
            "playbook_id": playbook_id,
            "name": "",
            "description": None,
            "version": None,
            "definition": {},
            "triggers": [],
            "category": None,
            "tags": [],
            "step_count": 0,
            "is_template": False,
            "is_locked": False,
            "formatted": f"Error: {e}",
        }


async def list_agent_profiles_tool(user_id: str = "system") -> Dict[str, Any]:
    """List agent profiles for the user with derived status. Call before creating to avoid duplicates."""
    try:
        client = await get_backend_tool_client()
        result = await client.list_agent_profiles(user_id=user_id)
        if not result.get("success"):
            return {"profiles": [], "count": 0, "formatted": result.get("formatted") or result.get("error", "List failed")}
        profiles = result.get("profiles", [])
        lines = [f"Found {len(profiles)} profile(s):"]
        for p in profiles:
            name = p.get("name") or "(unnamed)"
            handle = p.get("handle")
            pid = p.get("id") or ""
            handle_str = f" @{handle}" if handle else ""
            lines.append(f"  - **{name}**{handle_str} (id: {pid})")
        formatted = "\n".join(lines) if profiles else lines[0]
        return {"profiles": profiles, "count": len(profiles), "formatted": formatted}
    except Exception as e:
        logger.error("list_agent_profiles_tool error: %s", e)
        return {"profiles": [], "count": 0, "formatted": str(e)}


async def get_agent_profile_detail_tool(
    agent_id: str,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Get full agent profile details including model, system prompt, playbook, and config.
    Use before editing a profile or inspecting its configuration.
    """
    try:
        client = await get_backend_tool_client()
        profile = await client.get_agent_profile(user_id=user_id, profile_id=agent_id)
        if not profile:
            return {
                "agent_id": agent_id,
                "name": "",
                "handle": None,
                "description": None,
                "model_preference": None,
                "system_prompt_additions": None,
                "default_playbook_id": None,
                "is_active": False,
                "is_locked": False,
                "prompt_history_enabled": False,
                "persona_mode": None,
                "auto_routable": False,
                "knowledge_config": {},
                "formatted": f"Agent profile not found: {agent_id}",
            }
        name = profile.get("name", "")
        handle = profile.get("handle")
        lines = [
            f"**{name}**" + (f" (@{handle})" if handle else " (no @handle)"),
            f"ID: {profile.get('id', agent_id)}",
            f"Active: {profile.get('is_active', False)}",
            f"Model: {profile.get('model_preference') or '(default)'}",
            f"Default playbook: {profile.get('default_playbook_id') or '(none)'}",
        ]
        if profile.get("system_prompt_additions"):
            lines.append(f"System prompt additions: {profile['system_prompt_additions'][:200]}...")
        lines.append(
            f"History: {'on' if profile.get('prompt_history_enabled', profile.get('chat_history_enabled', False)) else 'off'}, "
            f"lookback {profile.get('chat_history_lookback', 10)}, "
            f"summarize over ~{profile.get('summary_threshold_tokens', 5000)} tokens, "
            f"keep {profile.get('summary_keep_messages', 10)} messages verbatim"
        )
        formatted = "\n".join(lines)
        return {
            "agent_id": str(profile.get("id", agent_id)),
            "name": name,
            "handle": handle,
            "description": profile.get("description"),
            "model_preference": profile.get("model_preference"),
            "system_prompt_additions": profile.get("system_prompt_additions"),
            "default_playbook_id": str(profile["default_playbook_id"]) if profile.get("default_playbook_id") else None,
            "is_active": profile.get("is_active", False),
            "is_locked": profile.get("is_locked", False),
            "prompt_history_enabled": profile.get("prompt_history_enabled", profile.get("chat_history_enabled", False)),
            "summary_threshold_tokens": profile.get("summary_threshold_tokens"),
            "summary_keep_messages": profile.get("summary_keep_messages"),
            "persona_mode": profile.get("persona_mode"),
            "auto_routable": profile.get("auto_routable", False),
            "knowledge_config": profile.get("knowledge_config") or {},
            "formatted": formatted,
        }
    except Exception as e:
        logger.error("get_agent_profile_detail_tool error: %s", e)
        return {
            "agent_id": agent_id,
            "name": "",
            "handle": None,
            "description": None,
            "model_preference": None,
            "system_prompt_additions": None,
            "default_playbook_id": None,
            "is_active": False,
            "is_locked": False,
            "prompt_history_enabled": False,
            "persona_mode": None,
            "auto_routable": False,
            "knowledge_config": {},
            "formatted": f"Error: {e}",
        }


async def list_skills_tool(
    category: Optional[str] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    List available skills (built-in and user-authored) for agent self-awareness.
    Use this to see which skills exist and what procedures they encode.
    """
    try:
        client = await get_backend_tool_client()
        skills = await client.list_skills(user_id=user_id, category=category, include_builtin=True)
        count = len(skills)
        lines = [f"Found {count} skill(s):"]
        for s in skills:
            name = s.get("name") or s.get("slug") or "?"
            sid = s.get("id") or ""
            cat = s.get("category") or "General"
            builtin = " (built-in)" if s.get("is_builtin") else ""
            desc_snippet = (s.get("description") or "")[:60]
            desc_str = f": {desc_snippet}" if desc_snippet else ""
            lines.append(f"- **{name}** (id: {sid}) [{cat}]{builtin}{desc_str}")
        return {"skills": skills, "count": count, "formatted": "\n".join(lines)}
    except Exception as e:
        logger.error("list_skills_tool error: %s", e)
        return {"skills": [], "count": 0, "formatted": f"Error listing skills: {e}"}


async def create_skill_tool(
    name: str,
    slug: str,
    procedure: str,
    description: Optional[str] = None,
    category: Optional[str] = None,
    required_tools: Optional[List[str]] = None,
    optional_tools: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    confirmed: bool = False,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Create a new user-authored skill from discovered patterns. Call with confirmed=False to preview, then confirmed=True after approval.
    """
    slug_clean = (slug or "").strip().lower().replace(" ", "-")[:100] or "unnamed-skill"
    if not confirmed:
        formatted = (
            f"[Preview] Would create skill: **{name}** (slug: {slug_clean})\n"
            f"Procedure length: {len(procedure or '')} chars. "
            f"Required tools: {required_tools or []}. "
            "When the user approves, you MUST call create_skill again with the same arguments and confirmed=True. Do not only ask — make that call."
        )
        return {"skill_id": "", "name": name, "slug": slug_clean, "is_draft": True, "formatted": formatted}
    try:
        client = await get_backend_tool_client()
        result = await client.create_skill(
            user_id=user_id,
            name=name or "Unnamed skill",
            slug=slug_clean,
            procedure=procedure or "",
            required_tools=required_tools or [],
            optional_tools=optional_tools or [],
            description=description,
            category=category,
            tags=tags or [],
        )
        if not result.get("success"):
            return {
                "skill_id": "",
                "name": name,
                "slug": slug_clean,
                "is_draft": False,
                "formatted": result.get("error", "Create failed"),
            }
        skill = result.get("skill") or {}
        formatted = f"Created skill **{skill.get('name', name)}** (ID: {result.get('skill_id', '')}). It can be assigned to profiles and steps."
        return {
            "skill_id": result.get("skill_id", ""),
            "name": skill.get("name", name),
            "slug": skill.get("slug", slug_clean),
            "is_draft": False,
            "formatted": formatted,
        }
    except Exception as e:
        logger.error("create_skill_tool error: %s", e)
        return {"skill_id": "", "name": name, "slug": slug_clean, "is_draft": False, "formatted": f"Error: {e}"}


async def propose_skill_update_tool(
    skill_id: str,
    proposed_procedure: str,
    rationale: str,
    evidence_metadata: Optional[Dict[str, Any]] = None,
    confirmed: bool = False,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Propose an update to a user skill (procedure, rationale, evidence). Creates a new version on approval.
    Call with confirmed=False to show preview (current vs proposed), then confirmed=True after user approval.
    """
    if not confirmed:
        try:
            client = await get_backend_tool_client()
            skills = await client.get_skills_by_ids(user_id, [skill_id])
            current = skills[0] if skills else {}
            cur_proc = (current.get("procedure") or "")[:200]
            formatted = (
                f"[Preview] Would update skill **{current.get('name', skill_id)}** (v{current.get('version', 1)})\n"
                f"Rationale: {rationale[:300]}\n"
                f"Current procedure (excerpt): {cur_proc}...\n"
                f"Proposed procedure length: {len(proposed_procedure or '')} chars. "
                "When the user approves, you MUST call propose_skill_update again with the same skill_id, proposed_procedure, rationale and confirmed=True. Do not only ask — make that call."
            )
        except Exception as e:
            formatted = f"[Preview] Would update skill {skill_id}. Rationale: {rationale[:200]}. Error fetching current: {e}. Call with confirmed=True to apply."
        return {"success": False, "skill_id": skill_id, "version": 0, "formatted": formatted}
    try:
        client = await get_backend_tool_client()
        result = await client.update_skill(
            user_id=user_id,
            skill_id=skill_id,
            procedure=proposed_procedure,
            improvement_rationale=rationale,
            evidence_metadata=evidence_metadata,
        )
        if not result.get("success"):
            return {
                "success": False,
                "skill_id": skill_id,
                "version": 0,
                "formatted": result.get("error", "Update failed"),
            }
        skill = result.get("skill") or {}
        formatted = f"Updated skill **{skill.get('name', skill_id)}** to version {result.get('version', 0)}. New procedure and rationale saved."
        return {
            "success": True,
            "skill_id": result.get("skill_id", skill_id),
            "version": result.get("version", 0),
            "formatted": formatted,
        }
    except Exception as e:
        logger.error("propose_skill_update_tool error: %s", e)
        return {"success": False, "skill_id": skill_id, "version": 0, "formatted": f"Error: {e}"}


def _is_valid_skill_id(s: str) -> bool:
    """Return True if s looks like a UUID (required for skill_id)."""
    if not s or not isinstance(s, str):
        return False
    s = s.strip()
    if len(s) != 36:
        return False
    try:
        import uuid
        uuid.UUID(s)
        return True
    except (ValueError, TypeError):
        return False


async def get_skill_detail_tool(
    skill_id: str,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Get full details of a skill including its complete procedure text,
    required tools, version, and metadata. Use this to read a skill's
    procedure before proposing updates via propose_skill_update.
    skill_id must be a UUID from list_skills (not a name or slug).
    """
    if not _is_valid_skill_id(skill_id):
        return {
            "skill_id": (skill_id or "").strip(),
            "name": "",
            "slug": "",
            "procedure": "",
            "required_tools": [],
            "optional_tools": [],
            "version": 0,
            "category": "",
            "tags": [],
            "improvement_rationale": "",
            "formatted": (
                f"Invalid skill_id: '{skill_id}'. skill_id must be a UUID from list_skills "
                "(use the id field from list_skills, not name or slug)."
            ),
        }
    try:
        client = await get_backend_tool_client()
        skills = await client.get_skills_by_ids(user_id, [skill_id.strip()])
        if not skills:
            return {
                "skill_id": skill_id,
                "name": "",
                "slug": "",
                "procedure": "",
                "required_tools": [],
                "optional_tools": [],
                "version": 0,
                "category": "",
                "tags": [],
                "improvement_rationale": "",
                "formatted": f"Skill not found: {skill_id}",
            }
        skill = skills[0]
        proc = skill.get("procedure") or ""
        proc_preview = proc[:500] + "..." if len(proc) > 500 else proc
        formatted = (
            f"**{skill.get('name', '')}** (v{skill.get('version', 1)})\n"
            f"Slug: {skill.get('slug', '')}\n"
            f"Category: {skill.get('category', 'General')}\n"
            f"Required tools: {skill.get('required_tools', [])}\n"
            f"Optional tools: {skill.get('optional_tools', [])}\n"
            f"Tags: {skill.get('tags', [])}\n"
        )
        if skill.get("improvement_rationale"):
            formatted += f"Last update rationale: {skill['improvement_rationale']}\n"
        formatted += f"\n**Procedure:**\n{proc_preview}"
        return {
            "skill_id": skill.get("id", skill_id),
            "name": skill.get("name", ""),
            "slug": skill.get("slug", ""),
            "procedure": proc,
            "required_tools": skill.get("required_tools", []),
            "optional_tools": skill.get("optional_tools", []),
            "version": skill.get("version", 1),
            "category": skill.get("category") or "",
            "tags": skill.get("tags", []),
            "improvement_rationale": skill.get("improvement_rationale") or "",
            "formatted": formatted,
        }
    except Exception as e:
        logger.error("get_skill_detail_tool error: %s", e)
        return {
            "skill_id": skill_id,
            "name": "",
            "slug": "",
            "procedure": "",
            "required_tools": [],
            "optional_tools": [],
            "version": 0,
            "category": "",
            "tags": [],
            "improvement_rationale": "",
            "formatted": f"Error: {e}",
        }


# ── Register actions ────────────────────────────────────────────────────────

register_action(
    name="list_available_actions",
    category="agent_factory",
    description="List available actions/tools for composing playbooks.",
    inputs_model=ListAvailableActionsInputs,
    params_model=None,
    outputs_model=ListAvailableActionsOutputs,
    tool_function=list_available_actions_tool,
)
register_action(
    name="validate_playbook_wiring",
    category="agent_factory",
    description="Validate a playbook definition (steps, refs, types). Call before create_playbook or after building definition for update_playbook.",
    inputs_model=ValidatePlaybookWiringInputs,
    params_model=None,
    outputs_model=ValidatePlaybookWiringOutputs,
    tool_function=validate_playbook_wiring_tool,
)
register_action(
    name="create_agent_profile",
    category="agent_factory",
    description="Create an agent profile (paused by default). confirmed=False preview, confirmed=True create.",
    short_description="Create an agent profile",
    inputs_model=CreateAgentProfileInputs,
    params_model=CreateAgentProfileParams,
    outputs_model=CreateAgentProfileOutputs,
    tool_function=create_agent_profile_tool,
)
register_action(
    name="set_agent_profile_status",
    category="agent_factory",
    description="Pause or activate an agent profile.",
    inputs_model=SetAgentProfileStatusInputs,
    params_model=SetAgentProfileStatusParams,
    outputs_model=SetAgentProfileStatusOutputs,
    tool_function=set_agent_profile_status_tool,
)
register_action(
    name="create_playbook",
    category="agent_factory",
    description="Create a new playbook. Pass name and steps (full step objects). Same step shape as get_playbook_detail / update_playbook. confirmed=False preview; confirmed=True create.",
    short_description="Create a playbook with steps",
    inputs_model=CreatePlaybookInputs,
    params_model=CreatePlaybookParams,
    outputs_model=CreatePlaybookOutputs,
    tool_function=create_playbook_tool,
)
register_action(
    name="assign_playbook_to_agent",
    category="agent_factory",
    description="Assign a playbook to an agent as its default. confirmed=False preview; confirmed=True apply.",
    inputs_model=AssignPlaybookToAgentInputs,
    params_model=AssignPlaybookToAgentParams,
    outputs_model=AssignPlaybookToAgentOutputs,
    tool_function=assign_playbook_to_agent_tool,
)
register_action(
    name="update_agent_profile",
    category="agent_factory",
    description="Update an agent profile. confirmed=False dry-run.",
    inputs_model=UpdateAgentProfileInputs,
    params_model=UpdateAgentProfileParams,
    outputs_model=UpdateAgentProfileOutputs,
    tool_function=update_agent_profile_tool,
)
register_action(
    name="delete_agent_profile",
    category="agent_factory",
    description="Delete an agent profile. confirmed=False dry-run. Irreversible when unlocked.",
    short_description="Delete an agent profile",
    inputs_model=DeleteAgentProfileInputs,
    params_model=DeleteAgentProfileParams,
    outputs_model=DeleteAgentProfileOutputs,
    tool_function=delete_agent_profile_tool,
)
register_action(
    name="update_playbook",
    category="agent_factory",
    description="Update a playbook. Pass playbook_id and updates (e.g. name, description, definition with steps). Send full definition.steps when changing steps; backend merges in missing fields from existing. confirmed=False preview; confirmed=True apply.",
    inputs_model=UpdatePlaybookInputs,
    params_model=UpdatePlaybookParams,
    outputs_model=UpdatePlaybookOutputs,
    tool_function=update_playbook_tool,
)
register_action(
    name="delete_playbook",
    category="agent_factory",
    description="Delete a playbook. confirmed=False preview; confirmed=True delete. Blocked if locked or template.",
    short_description="Delete a playbook",
    inputs_model=DeletePlaybookInputs,
    params_model=DeletePlaybookParams,
    outputs_model=DeletePlaybookOutputs,
    tool_function=delete_playbook_tool,
)
register_action(
    name="list_playbooks",
    category="agent_factory",
    description="List playbooks and templates (id, name, description). Use to find playbook_id before get_playbook_detail or update_playbook.",
    short_description="List playbooks and templates",
    inputs_model=ListPlaybooksInputs,
    params_model=None,
    outputs_model=ListPlaybooksOutputs,
    tool_function=list_playbooks_tool,
)
register_action(
    name="get_playbook_detail",
    category="agent_factory",
    description="Get one playbook by id: metadata plus full definition (steps, run_context). When definition is small enough, response includes full JSON so you can copy it into update_playbook. Always call before editing an existing playbook.",
    short_description="Get full playbook definition and steps",
    inputs_model=GetPlaybookDetailInputs,
    params_model=None,
    outputs_model=GetPlaybookDetailOutputs,
    tool_function=get_playbook_detail_tool,
)
register_action(
    name="list_agent_profiles",
    category="agent_factory",
    description="List agent profiles for the user with derived status; call before creating to avoid duplicates",
    short_description="List agent profiles",
    inputs_model=ListAgentProfilesInputs,
    params_model=None,
    outputs_model=ListAgentProfilesOutputs,
    tool_function=list_agent_profiles_tool,
)
register_action(
    name="get_agent_profile_detail",
    category="agent_factory",
    description="Get full agent profile details including model, system prompt, playbook; use before editing a profile",
    short_description="Get full agent profile details",
    inputs_model=GetAgentProfileDetailInputs,
    params_model=None,
    outputs_model=GetAgentProfileDetailOutputs,
    tool_function=get_agent_profile_detail_tool,
)
register_action(
    name="create_agent_schedule",
    category="agent_factory",
    description="Create a schedule for an agent (cron or interval).",
    inputs_model=CreateAgentScheduleInputs,
    params_model=CreateAgentScheduleParams,
    outputs_model=CreateAgentScheduleOutputs,
    tool_function=create_agent_schedule_tool,
)
register_action(
    name="list_agent_schedules",
    category="agent_factory",
    description="List schedules for an agent profile; use before creating to avoid duplicates",
    inputs_model=ListAgentSchedulesInputs,
    params_model=None,
    outputs_model=ListAgentSchedulesOutputs,
    tool_function=list_agent_schedules_tool,
)
register_action(
    name="list_agent_data_sources",
    category="agent_factory",
    description="List data source bindings for an agent profile",
    inputs_model=ListAgentDataSourcesInputs,
    params_model=None,
    outputs_model=ListAgentDataSourcesOutputs,
    tool_function=list_agent_data_sources_tool,
)
register_action(
    name="bind_data_source_to_agent",
    category="agent_factory",
    description="Bind a data source connector to an agent.",
    inputs_model=BindDataSourceToAgentInputs,
    params_model=BindDataSourceToAgentParams,
    outputs_model=BindDataSourceToAgentOutputs,
    tool_function=bind_data_source_to_agent_tool,
)
register_action(
    name="list_skills",
    category="agent_factory",
    description="List available skills (built-in and user) for self-awareness.",
    short_description="List available skills",
    inputs_model=ListSkillsInputs,
    params_model=None,
    outputs_model=ListSkillsOutputs,
    tool_function=list_skills_tool,
)
register_action(
    name="create_skill",
    category="agent_factory",
    description="Create a new user skill. confirmed=False preview, confirmed=True create.",
    short_description="Create a new user skill",
    inputs_model=CreateSkillInputs,
    params_model=CreateSkillParams,
    outputs_model=CreateSkillOutputs,
    tool_function=create_skill_tool,
)
register_action(
    name="propose_skill_update",
    category="agent_factory",
    description="Propose a skill procedure update (new version). confirmed=False preview, confirmed=True apply.",
    short_description="Propose a skill procedure update",
    inputs_model=ProposeSkillUpdateInputs,
    params_model=ProposeSkillUpdateParams,
    outputs_model=ProposeSkillUpdateOutputs,
    tool_function=propose_skill_update_tool,
)
register_action(
    name="get_skill_detail",
    category="agent_factory",
    description="Get full details of a skill including complete procedure, tools, and version.",
    short_description="Get full details of a skill",
    inputs_model=GetSkillDetailInputs,
    params_model=None,
    outputs_model=GetSkillDetailOutputs,
    tool_function=get_skill_detail_tool,
)

AGENT_FACTORY_TOOLS = [
    "list_available_actions_tool",
    "validate_playbook_wiring_tool",
    "list_available_llm_models_tool",
    "create_agent_profile_tool",
    "set_agent_profile_status_tool",
    "create_playbook_tool",
    "assign_playbook_to_agent_tool",
    "update_agent_profile_tool",
    "delete_agent_profile_tool",
    "list_playbooks_tool",
    "get_playbook_detail_tool",
    "list_agent_profiles_tool",
    "get_agent_profile_detail_tool",
    "update_playbook_tool",
    "delete_playbook_tool",
    "create_agent_schedule_tool",
    "list_agent_schedules_tool",
    "list_agent_data_sources_tool",
    "bind_data_source_to_agent_tool",
    "list_skills_tool",
    "create_skill_tool",
    "propose_skill_update_tool",
    "get_skill_detail_tool",
]
