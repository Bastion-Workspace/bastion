"""
Fiction Editing Subgraph - Flat Architecture

Uses flat node structure (no nested compiled subgraphs) to match the proven
pattern of outline/character/rules/style subgraphs. State flows through
sequential nodes without crossing subgraph boundaries.

Node logic is imported from:
- fiction_context_subgraph: context preparation nodes
- fiction_generation_subgraph: generation and LLM nodes
- fiction_validation_subgraph: outline sync and consistency validation
- fiction_resolution_subgraph: operation resolution nodes
- fiction_editing_agent: routing, mode detection, format response
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END


class FictionEditingSubgraphState(TypedDict, total=False):
    """
    State schema for the fiction editing subgraph.
    Declares all state keys so LangGraph uses merge semantics and preserves
    dynamically set keys (e.g. current_request) across nodes.
    Keys mirror preserve_fiction_state() in writing_subgraph_utilities.py.
    """
    metadata: Dict[str, Any]
    user_id: str
    shared_memory: Dict[str, Any]
    messages: List[Any]
    query: str
    active_editor: Dict[str, Any]
    manuscript: str
    manuscript_content: str
    filename: str
    frontmatter: Dict[str, Any]
    current_chapter_text: str
    current_chapter_number: Optional[int]
    prev_chapter_text: Optional[str]
    prev_chapter_number: Optional[int]
    next_chapter_text: Optional[str]
    next_chapter_number: Optional[int]
    chapter_ranges: List[Any]
    explicit_primary_chapter: Optional[int]
    requested_chapter_number: Optional[int]
    reference_chapter_numbers: List[Any]
    reference_chapter_texts: Dict[str, Any]
    outline_body: Optional[str]
    rules_body: Optional[str]
    style_body: Optional[str]
    characters_bodies: List[Any]
    series_body: Optional[str]
    outline_current_chapter_text: Optional[str]
    outline_prev_chapter_text: Optional[str]
    story_overview: Optional[str]
    book_map: Optional[Any]
    system_prompt: str
    datetime_context: str
    current_request: str
    selection_start: int
    selection_end: int
    cursor_offset: int
    mode_guidance: str
    reference_guidance: str
    reference_quality: Dict[str, Any]
    reference_warnings: List[Any]
    has_references: bool
    creative_freedom_requested: bool
    request_type: str
    outline_sync_analysis: Optional[Any]
    consistency_warnings: List[Any]
    structured_edit: Optional[Dict[str, Any]]
    editor_operations: List[Any]
    failed_operations: List[Any]
    response: Optional[Dict[str, Any]]
    task_status: str
    error: str
    mentioned_chapters: List[Any]
    generation_mode: str
    generated_chapters: Dict[str, Any]
    is_multi_chapter: bool
    target_chapter_number: Optional[int]
    prev_chapter_last_line: Optional[str]
    generation_context_parts: List[Any]
    generation_messages: List[Any]
    llm_response: str
    anchor_validation_passed: bool
    resolution_complete: bool
    mode: str
    resolution_manuscript: str
    resolution_structured_edit: Optional[Dict[str, Any]]
    resolution_operations: List[Any]
    resolution_selection: Optional[Any]
    resolution_cursor_offset: Optional[int]
    resolution_fm_end_idx: int
    resolution_is_empty_file: bool
    resolution_desired_ch_num: int
    resolution_chapter_ranges: List[Any]
    resolution_current_chapter_number: Optional[int]
    resolution_requested_chapter_number: Optional[int]
    resolved_operations: List[Any]
    validated_operations: List[Any]


# Context preparation (flat nodes - no subgraph boundary)
from orchestrator.subgraphs.fiction_context_subgraph import (
    prepare_context_node,
    detect_chapter_mentions_node,
    analyze_scope_node,
    load_references_node,
    assess_reference_quality_node,
)
# Generation (flat nodes)
from orchestrator.subgraphs.fiction_generation_subgraph import (
    build_generation_context_node,
    build_generation_prompt_node,
    call_generation_llm_node,
    validate_generated_output_node,
    validate_anchors_node,
    self_heal_anchors_node,
    merge_operations_node,
)
# Validation (flat nodes)
from orchestrator.subgraphs.fiction_validation_subgraph import (
    detect_outline_changes_node as _detect_outline_changes_node_raw,
    validate_consistency_node,
)
# Resolution (flat nodes)
from orchestrator.subgraphs.fiction_resolution_subgraph import (
    prepare_resolution_context_node,
    resolve_individual_operations_node,
    validate_resolved_operations_node,
    finalize_operations_node,
)
# Proofreading (kept as compiled subgraph - working path)
from orchestrator.subgraphs.proofreading_subgraph import build_proofreading_subgraph

logger = logging.getLogger(__name__)


async def inject_fiction_state_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Copy active_editor from shared_memory into state for downstream nodes."""
    try:
        shared_memory = state.get("shared_memory", {}) or {}
        active_editor = shared_memory.get("active_editor", {}) or {}
        return {
            "active_editor": active_editor,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": shared_memory,
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
    except Exception as e:
        logger.error("inject_fiction_state_node failed: %s", e)
        return {
            "active_editor": {},
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


def _route_after_request_type(state: Dict[str, Any]) -> str:
    """Route to answer_question or prepare_generation based on request_type."""
    request_type = state.get("request_type", "edit_request")
    if request_type == "question":
        return "answer_question"
    return "prepare_generation"


def _route_after_anchor_validation(state: Dict[str, Any]) -> str:
    """Route based on anchor validation - merge or self_heal."""
    anchor_validation_passed = state.get("anchor_validation_passed", True)
    if anchor_validation_passed:
        return "merge_operations"
    return "self_heal_anchors"


def _route_after_prepare_resolution(state: Dict[str, Any]) -> str:
    """Route after prepare_resolution_context - early exit or resolve."""
    if state.get("resolution_complete", False):
        return "finalize_operations"
    return "resolve_individual_operations"


def build_fiction_editing_subgraph(checkpointer, llm_factory, get_datetime_context):
    """
    Build flat fiction editing subgraph. All nodes are direct nodes;
    no nested compiled subgraphs except proofreading (kept as-is).
    """
    from orchestrator.agents.fiction_editing_agent import get_fiction_editing_agent

    fiction_agent = get_fiction_editing_agent()
    proofreading_compiled = build_proofreading_subgraph(
        checkpointer, llm_factory=llm_factory, get_datetime_context=get_datetime_context
    )

    # Wrappers for nodes that need llm_factory / get_datetime_context
    async def call_llm_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        return await call_generation_llm_node(state, llm_factory)

    async def self_heal_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        return await self_heal_anchors_node(state, llm_factory)

    async def detect_outline_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        return await _detect_outline_changes_node_raw(
            state, llm_factory, get_datetime_context
        )

    workflow = StateGraph(FictionEditingSubgraphState)

    # --- Context preparation (flat) ---
    workflow.add_node("inject_fiction_state", inject_fiction_state_node)
    workflow.add_node("prepare_context", prepare_context_node)
    workflow.add_node("detect_chapter_mentions", detect_chapter_mentions_node)
    workflow.add_node("analyze_scope", analyze_scope_node)
    workflow.add_node("load_references", load_references_node)
    workflow.add_node("assess_references", assess_reference_quality_node)

    # --- Agent nodes ---
    workflow.add_node("validate_fiction_type", fiction_agent._validate_fiction_type_node)
    workflow.add_node("detect_mode", fiction_agent._detect_mode_and_intent_node)
    workflow.add_node("detect_request_type", fiction_agent._detect_request_type_node)
    workflow.add_node("answer_question", fiction_agent._answer_question_node)
    workflow.add_node("prepare_generation", fiction_agent._prepare_generation_node)
    workflow.add_node("generate_simple_edit", fiction_agent._generate_simple_edit_node)
    workflow.add_node("format_response", fiction_agent._format_response_node)

    # --- Generation (flat) ---
    workflow.add_node("build_generation_context", build_generation_context_node)
    workflow.add_node("build_generation_prompt", build_generation_prompt_node)
    workflow.add_node("call_llm", call_llm_wrapper)
    workflow.add_node("validate_llm_output", validate_generated_output_node)
    workflow.add_node("validate_anchors", validate_anchors_node)
    workflow.add_node("self_heal_anchors", self_heal_wrapper)
    workflow.add_node("merge_operations", merge_operations_node)

    # --- Validation (flat) ---
    workflow.add_node("detect_outline_changes", detect_outline_wrapper)
    workflow.add_node("validate_consistency", validate_consistency_node)

    # --- Resolution (flat) ---
    workflow.add_node("prepare_resolution_context", prepare_resolution_context_node)
    workflow.add_node("resolve_individual_operations", resolve_individual_operations_node)
    workflow.add_node("validate_resolved_operations", validate_resolved_operations_node)
    workflow.add_node("finalize_operations", finalize_operations_node)

    # --- Proofreading (compiled subgraph - single node) ---
    workflow.add_node("proofreading", proofreading_compiled)

    # --- Entry and context flow ---
    workflow.set_entry_point("inject_fiction_state")
    workflow.add_edge("inject_fiction_state", "prepare_context")
    workflow.add_edge("prepare_context", "detect_chapter_mentions")
    workflow.add_edge("detect_chapter_mentions", "analyze_scope")
    workflow.add_edge("analyze_scope", "load_references")
    workflow.add_edge("load_references", "assess_references")
    workflow.add_edge("assess_references", "validate_fiction_type")

    # --- After validate_fiction_type ---
    workflow.add_conditional_edges(
        "validate_fiction_type",
        fiction_agent._route_after_validate_type,
        {"error": "format_response", "continue": "detect_mode"},
    )

    # --- After detect_mode: error / proofreading / simple / full ---
    workflow.add_conditional_edges(
        "detect_mode",
        fiction_agent._route_after_context,
        {
            "format_response": "format_response",
            "proofreading": "proofreading",
            "simple_path": "generate_simple_edit",
            "full_path": "detect_request_type",
        },
    )

    # --- Proofreading path ---
    workflow.add_edge("proofreading", "prepare_resolution_context")

    # --- Simple path ---
    workflow.add_edge("generate_simple_edit", "prepare_resolution_context")

    # --- Full path: detect_request_type -> question or prepare_generation ---
    workflow.add_conditional_edges(
        "detect_request_type",
        _route_after_request_type,
        {
            "answer_question": "answer_question",
            "prepare_generation": "prepare_generation",
        },
    )

    # --- Question path ---
    workflow.add_edge("answer_question", "prepare_resolution_context")

    # --- Full generation path ---
    workflow.add_edge("prepare_generation", "build_generation_context")
    workflow.add_edge("build_generation_context", "build_generation_prompt")
    workflow.add_edge("build_generation_prompt", "call_llm")
    workflow.add_edge("call_llm", "validate_llm_output")
    workflow.add_edge("validate_llm_output", "validate_anchors")

    workflow.add_conditional_edges(
        "validate_anchors",
        _route_after_anchor_validation,
        {
            "merge_operations": "merge_operations",
            "self_heal_anchors": "self_heal_anchors",
        },
    )
    workflow.add_edge("self_heal_anchors", "merge_operations")
    workflow.add_edge("merge_operations", "detect_outline_changes")
    workflow.add_edge("detect_outline_changes", "validate_consistency")
    workflow.add_edge("validate_consistency", "prepare_resolution_context")

    # --- Resolution flow ---
    workflow.add_conditional_edges(
        "prepare_resolution_context",
        _route_after_prepare_resolution,
        {
            "finalize_operations": "finalize_operations",
            "resolve_individual_operations": "resolve_individual_operations",
        },
    )
    workflow.add_edge("resolve_individual_operations", "validate_resolved_operations")
    workflow.add_edge("validate_resolved_operations", "finalize_operations")
    workflow.add_edge("finalize_operations", "format_response")
    workflow.add_edge("format_response", END)

    return workflow.compile(checkpointer=checkpointer)
