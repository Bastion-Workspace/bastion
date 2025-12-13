"""
Fiction Editing Agent - LangGraph Implementation
Gated to fiction manuscripts. Consumes active editor manuscript, cursor, and
referenced outline/rules/style/characters. Produces ManuscriptEdit with HITL.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Tuple, TYPE_CHECKING

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import ValidationError, BaseModel, Field

from .base_agent import BaseAgent, TaskStatus
from orchestrator.models.editor_models import EditorOperation, ManuscriptEdit
from orchestrator.models.continuity_models import ContinuityState
from orchestrator.services.fiction_continuity_tracker import FictionContinuityTracker
from orchestrator.utils.editor_operation_resolver import resolve_editor_operation
from orchestrator.subgraphs import (
    build_context_preparation_subgraph,
    build_validation_subgraph,
    build_generation_subgraph,
    build_resolution_subgraph,
)
from orchestrator.utils.fiction_utilities import (
    ChapterRange,
    CHAPTER_PATTERN,
    find_chapter_ranges,
    locate_chapter_index,
    get_adjacent_chapters,
    slice_hash as _slice_hash,
    strip_frontmatter_block as _strip_frontmatter_block,
    frontmatter_end_index as _frontmatter_end_index,
    unwrap_json_response as _unwrap_json_response,
    normalize_for_overlap_check as _normalize_for_overlap_check,
    looks_like_outline_copied as _looks_like_outline_copied,
    extract_chapter_number_from_request as _extract_chapter_number_from_request,
    extract_chapter_range_from_request as _extract_chapter_range_from_request,
    ensure_chapter_heading as _ensure_chapter_heading,
    extract_character_name as _extract_character_name,
    detect_chapter_heading_at_position as _detect_chapter_heading_at_position,
    strip_chapter_heading_from_text as _strip_chapter_heading_from_text,
)

if TYPE_CHECKING:
    # Forward reference for type hints only
    pass

logger = logging.getLogger(__name__)


# Pydantic models for structured outputs
class OutlineDiscrepancy(BaseModel):
    """A single discrepancy between outline and manuscript"""
    type: str = Field(description="Type of discrepancy: missing_beat, changed_beat, character_action_mismatch, story_progression_issue")
    outline_expectation: str = Field(description="What the outline says should happen")
    manuscript_current: str = Field(description="What the manuscript currently has (or 'missing')")
    severity: str = Field(description="Severity: critical, major, minor")
    suggestion: str = Field(description="Suggestion for how to resolve the discrepancy")


class OutlineSyncAnalysis(BaseModel):
    """Analysis of outline vs manuscript alignment"""
    needs_sync: bool = Field(description="Whether the manuscript needs updates to match the outline")
    discrepancies: List[OutlineDiscrepancy] = Field(default_factory=list, description="List of discrepancies found")
    summary: str = Field(description="Summary of the alignment analysis")


# ============================================
# Utilities imported from fiction_utilities
# ============================================
# All chapter/text processing utilities are now in orchestrator.utils.fiction_utilities


# ============================================
# LangGraph State
# ============================================

class FictionEditingState(TypedDict):
    """State for fiction editing agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    active_editor: Dict[str, Any]
    manuscript: str
    filename: str
    frontmatter: Dict[str, Any]
    cursor_offset: int
    selection_start: int
    selection_end: int
    chapter_ranges: List[ChapterRange]
    active_chapter_idx: int
    current_chapter_text: str
    current_chapter_number: Optional[int]
    prev_chapter_text: Optional[str]
    prev_chapter_number: Optional[int]
    next_chapter_text: Optional[str]
    next_chapter_number: Optional[int]
    outline_body: Optional[str]
    rules_body: Optional[str]
    style_body: Optional[str]
    characters_bodies: List[str]
    outline_current_chapter_text: Optional[str]
    current_request: str
    requested_chapter_number: Optional[int]
    system_prompt: str
    llm_response: str
    structured_edit: Optional[Dict[str, Any]]
    editor_operations: List[Dict[str, Any]]
    response: Dict[str, Any]
    task_status: str
    error: str
    # New fields for mode tracking and validation
    generation_mode: str
    creative_freedom_requested: bool
    mode_guidance: str
    reference_quality: Dict[str, Any]
    reference_warnings: List[str]
    reference_guidance: str
    consistency_warnings: List[str]
    # Multi-chapter generation fields
    is_multi_chapter: bool
    chapter_range: Optional[Tuple[int, int]]  # (start, end) inclusive
    current_generation_chapter: Optional[int]  # Current chapter being generated in multi-chapter mode
    generated_chapters: Dict[int, str]  # Map of chapter_number -> generated text for continuity
    # Outline sync detection
    outline_sync_analysis: Optional[Dict[str, Any]]      # Analysis of outline vs manuscript discrepancies
    outline_needs_sync: bool  # Whether manuscript needs updates to match outline
    # Question answering
    request_type: str  # "question" | "edit_request" | "hybrid" | "unknown"
    # Continuity tracking
    continuity_state: Optional[Dict[str, Any]]  # Serialized ContinuityState
    continuity_document_id: Optional[str]  # Document ID of .continuity.json
    continuity_violations: List[Dict[str, Any]]  # Detected violations
    # Explicit chapter detection from user query
    explicit_primary_chapter: Optional[int]  # Primary chapter to edit (from query)
    explicit_secondary_chapters: List[int]  # Secondary chapters for context (from query)


# ============================================
# Type-Safe State Access Helpers
# ============================================

def _get_structured_edit(state: "FictionEditingState") -> Optional[ManuscriptEdit]:
    """
    Safely extract and validate structured_edit from state as ManuscriptEdit model.
    
    Returns None if structured_edit is missing or invalid.
    Provides type-safe access to ManuscriptEdit fields.
    """
    edit_dict = state.get("structured_edit")
    if not edit_dict:
        return None
    
    if isinstance(edit_dict, ManuscriptEdit):
        # Already a model (shouldn't happen in state, but handle gracefully)
        return edit_dict
    
    if not isinstance(edit_dict, dict):
        logger.warning(f"structured_edit is not a dict: {type(edit_dict)}")
        return None
    
    try:
        return ManuscriptEdit(**edit_dict)
    except ValidationError as e:
        logger.error(f"Failed to validate structured_edit as ManuscriptEdit: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error converting structured_edit to ManuscriptEdit: {e}")
        return None


def _get_continuity_state(state: "FictionEditingState") -> Optional[ContinuityState]:
    """
    Safely extract and validate continuity_state from state as ContinuityState model.
    
    Returns None if continuity_state is missing or invalid.
    Provides type-safe access to ContinuityState fields.
    """
    continuity_dict = state.get("continuity_state")
    if not continuity_dict:
        return None
    
    if isinstance(continuity_dict, ContinuityState):
        # Already a model (shouldn't happen in state, but handle gracefully)
        return continuity_dict
    
    if not isinstance(continuity_dict, dict):
        logger.warning(f"continuity_state is not a dict: {type(continuity_dict)}")
        return None
    
    try:
        return ContinuityState(**continuity_dict)
    except ValidationError as e:
        logger.error(f"Failed to validate continuity_state as ContinuityState: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error converting continuity_state to ContinuityState: {e}")
        return None


# ============================================
# Fiction Editing Agent
# ============================================

class FictionEditingAgent(BaseAgent):
    """
    Fiction Editing Agent for manuscript editing and generation
    
    Gated to fiction manuscripts. Consumes active editor manuscript, cursor, and
    referenced outline/rules/style/characters. Produces ManuscriptEdit with HITL.
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("fiction_editing_agent")
        self._grpc_client = None
        self._continuity_tracker = FictionContinuityTracker(llm_factory=self._get_llm)
        logger.info("Fiction Editing Agent ready!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for fiction editing agent"""
        workflow = StateGraph(FictionEditingState)
        
        # Build subgraphs
        context_subgraph = build_context_preparation_subgraph(checkpointer)
        validation_subgraph = build_validation_subgraph(
            checkpointer,
            llm_factory=self._get_llm,
            get_datetime_context=self._get_datetime_context,
            continuity_tracker=self._continuity_tracker
        )
        generation_subgraph = build_generation_subgraph(
            checkpointer,
            llm_factory=self._get_llm,
            get_datetime_context=self._get_datetime_context
        )
        resolution_subgraph = build_resolution_subgraph(checkpointer)
        
        # Phase 1: Context preparation (now a subgraph)
        workflow.add_node("context_preparation", context_subgraph)
        workflow.add_node("validate_fiction_type", self._validate_fiction_type_node)
        
        # Phase 2: Pre-generation assessment
        workflow.add_node("detect_mode", self._detect_mode_and_intent_node)
        workflow.add_node("detect_request_type", self._detect_request_type_node)
        
        # Phase 3: Multi-chapter loop control
        workflow.add_node("check_multi_chapter", self._check_multi_chapter_node)
        workflow.add_node("prepare_chapter_context", self._prepare_chapter_context_node)
        workflow.add_node("prepare_generation", self._prepare_generation_node)
        
        # Phase 4: Generation (now a subgraph)
        workflow.add_node("generate_edit_plan", generation_subgraph)
        workflow.add_node("generate_simple_edit", self._generate_simple_edit_node)
        
        # Phase 5: Post-generation validation (now a subgraph)
        workflow.add_node("validation", validation_subgraph)
        
        # Phase 6: Resolution (now a subgraph) and response
        workflow.add_node("resolve_operations", resolution_subgraph)
        workflow.add_node("accumulate_chapter", self._accumulate_chapter_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("context_preparation")
        
        # Flow: context preparation subgraph -> validate fiction type -> detect mode
        workflow.add_edge("context_preparation", "validate_fiction_type")
        workflow.add_conditional_edges(
            "validate_fiction_type",
            self._route_after_validate_type,
            {
                "error": "format_response",  # Error - skip to format
                "continue": "detect_mode"     # Valid - continue
            }
        )
        
        # Short-circuit: if no references, use fast path directly
        workflow.add_conditional_edges(
            "detect_mode",
            self._route_after_context,
            {
                "simple_path": "generate_simple_edit",  # No references -> fast path
                "full_path": "detect_request_type"       # Has references -> full path
            }
        )
        
        # Route to check_multi_chapter (both questions and edit requests go through same path)
        workflow.add_edge("detect_request_type", "check_multi_chapter")
        
        # Single chapter: prepare generation before calling subgraph
        workflow.add_conditional_edges(
            "check_multi_chapter",
            self._route_multi_chapter,
            {
                "multi_chapter_loop": "prepare_chapter_context",
                "single_chapter": "prepare_generation"
            }
        )
        
        # After prepare_generation, go to generation subgraph
        workflow.add_edge("prepare_generation", "generate_edit_plan")
        
        # Shared flow: both single and multi-chapter go through generation pipeline
        workflow.add_edge("generate_edit_plan", "validation")
        workflow.add_edge("validation", "resolve_operations")
        
        # Simple path: skip validation, go straight to resolution
        workflow.add_edge("generate_simple_edit", "resolve_operations")
        
        # Route after resolve_operations: simple path skips, full path continues
        workflow.add_conditional_edges(
            "resolve_operations",
            self._route_after_resolve,
            {
                "format_response": "format_response",  # Simple path: skip
                "accumulate_chapter": "accumulate_chapter"  # Multi-chapter: accumulate
            }
        )
        
        # For multi-chapter: loop back to prepare_chapter_context if more chapters needed
        workflow.add_conditional_edges(
            "accumulate_chapter",
            self._route_after_accumulate,
            {
                "next_chapter": "prepare_chapter_context",  # More chapters - loop back
                "format_response": "format_response"  # All done
            }
        )
        
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for fiction editing"""
        return (
            "You are a MASTER NOVELIST editor/generator. Persona disabled.\n\n"
            "=== STYLE GUIDE FIRST PRINCIPLE ===\n\n"
            "**The Style Guide is HOW to write. The Outline is WHAT happens.**\n\n"
            "When generating narrative prose:\n"
            "- The Style Guide establishes your narrative voice, techniques, and craft (POV, tense, pacing, dialogue style, sensory detail level)\n"
            "- The Outline provides story structure and plot beats (what events occur, character arcs, story progression)\n"
            "- Your task: Write natural, compelling narrative in the Style Guide's voice that achieves the Outline's story goals\n"
            "- NEVER convert outline beats mechanically - craft scenes that flow naturally and happen to hit those beats\n"
            "- The Style Guide voice must permeate every sentence - internalize it BEFORE writing, not as an afterthought\n\n"
            "Maintain originality and do not copy from references. Adhere strictly to the project's Style Guide and Rules above all else.\n\n"
            "**REFERENCE USAGE (CRITICAL)**:\n"
            "When references are available, you MUST use them appropriately:\n\n"
            "- **STYLE GUIDE**: Use for HOW to write narrative prose\n"
            "  - Internalize narrative voice, POV, tense, pacing BEFORE writing\n"
            "  - Apply dialogue style, sensory detail level, and show-don't-tell techniques\n"
            "  - Match sentence structure patterns, rhythm, and descriptive style\n"
            "  - The Style Guide voice must permeate every sentence - not an afterthought\n\n"
            "- **OUTLINE**: Use for WHAT happens in the story (NEVER for text matching!)\n"
            "  - ⚠️ CRITICAL: Outline text does NOT exist in manuscript - NEVER use for anchors/original_text!\n"
            "  - 🚫 ABSOLUTE PROHIBITION: DO NOT copy, paraphrase, or reuse outline synopsis/beat text in your narrative prose\n"
            "  - ✅ DO creatively interpret outline beats into original narrative scenes with full prose\n"
            "  - Follow story structure and plot beats from the outline as GUIDANCE, not as SOURCE TEXT to copy\n"
            "  - Achieve outline's story goals through natural storytelling (don't convert beats mechanically)\n"
            "  - Reference character arcs and story progression from outline for INSPIRATION, not for copying\n"
            "  - Use outline context to inform description choices and scene emphasis - but write ORIGINAL prose\n"
            "  - For all text matching (anchors), use MANUSCRIPT text only, never outline text\n"
            "  - **REMEMBER**: Outline = inspiration for WHAT happens, Style Guide = HOW to write it, Your prose = ORIGINAL creative narrative\n\n"
            "- **CHARACTER PROFILES**: Use when writing character appearances, actions, dialogue, and internal thoughts\n"
            "  - **CRITICAL**: Each character profile is for a DIFFERENT character with distinct traits, dialogue patterns, and behaviors\n"
            "  - **PAY ATTENTION**: When writing dialogue or character actions, match the CORRECT character's profile - do not confuse characters\n"
            "  - Reference character traits, motivations, and backgrounds when writing character actions\n"
            "  - Ensure character dialogue patterns match established character voice from profiles\n"
            "  - Verify character appearances align with profile descriptions (vary phrasing, keep facts consistent)\n"
            "  - Check that character relationships and dynamics match profile information\n"
            "  - Ensure character actions are consistent with their personality, strengths, and flaws\n"
            "  - Draw from character profiles for authentic details, but express them differently each time\n"
            "  - **DO NOT MIX UP**: Each character has unique dialogue patterns - Jack's dialogue should match Jack's profile, not another character's\n\n"
            "- **UNIVERSE RULES**: Use to ensure world-building consistency in narrative prose\n"
            "  - Verify all world-building elements (magic systems, technology, physics) align with universe rules\n"
            "  - Ensure plot events and character actions don't violate established universe constraints\n"
            "  - Check that character abilities respect power systems and limitations from rules\n"
            "  - Ensure timeline and events are consistent with universe history\n"
            "  - Validate cultural, social, geographic, and environmental rules are followed\n\n"
            "**IMPORTANT**: References are provided in the context below. Always check them when generating prose to ensure consistency.\n\n"
            "=== CREATIVE VARIATION TO AVOID REPETITION ===\n\n"
            "**Vary descriptions and phrasing while maintaining consistency:**\n\n"
            "When writing prose, especially across multiple chapters or scenes:\n"
            "- **Vary character descriptions**: Use different details, angles, and phrasings when describing the same character\n"
            "  - Draw from character profiles for authentic details, but express them differently each time\n"
            "  - Example: Character profile says 'tall, dark hair, intense gaze' - vary descriptions:\n"
            "    * First mention: 'He towered over the desk, dark hair falling across his brow'\n"
            "    * Later: 'His height made the doorway seem small, and those dark eyes missed nothing'\n"
            "    * Another scene: 'The intensity in his gaze could cut glass, matched by the sharp line of his jaw'\n"
            "- **Vary location descriptions**: Same place, different details and perspectives based on context\n"
            "  - Consider: time of day, weather, character's emotional state, what's happening in the scene\n"
            "  - Example: A room described as 'cozy' in one scene might be 'claustrophobic' in another, depending on context\n"
            "- **Vary action descriptions**: Similar actions should feel fresh, not repetitive\n"
            "  - Example: Instead of always 'walked quickly,' vary with 'strode,' 'hurried,' 'moved with purpose,' 'covered ground in long steps'\n"
            "- **Vary dialogue tags and beats**: Mix dialogue tags, action beats, and internal thoughts\n"
            "  - Avoid repetitive patterns like always using 'he said' or always using action beats\n"
            "- **Use outline context creatively**: The outline tells you WHAT happens, but you decide HOW to describe it\n"
            "  - Same outline beat can be written with different emphasis, details, and perspective\n"
            "  - Example: Outline says 'Character enters room' - vary based on scene context:\n"
            "    * Tense scene: 'He pushed the door open, every creak a potential warning'\n"
            "    * Calm scene: 'The door swung open to reveal sunlight streaming through dusty windows'\n"
            "    * Emotional scene: 'She hesitated at the threshold, hand hovering over the knob'\n\n"
            "**Consistency Requirements (DO NOT BREAK):**\n"
            "- Character profiles: Core traits, appearance, personality must remain consistent\n"
            "- Universe rules: Physical laws, magic systems, technology must remain consistent\n"
            "- Style Guide: Narrative voice, POV, tense, pacing must remain consistent\n"
            "- Story continuity: Established facts, character knowledge, plot threads must remain consistent\n\n"
            "**Balance Variation with Consistency:**\n"
            "- Vary HOW you describe things (phrasing, details, perspective)\n"
            "- Keep WHAT you describe consistent (character traits, world rules, story facts)\n"
            "- Example: Character is always 'tall' (consistent), but describe it differently each time (varied)\n"
            "- Example: Magic system rules stay the same (consistent), but how magic looks/feels can vary by context (varied)\n\n"
            "**When in doubt:**\n"
            "- Check character profiles for authentic details to draw from\n"
            "- Check outline for story context that informs description choices\n"
            "- Check Style Guide for voice and technique requirements\n"
            "- Vary descriptions naturally based on scene context, character perspective, and emotional tone\n\n"
            "STRUCTURED OUTPUT REQUIRED: You MUST return ONLY raw JSON (no prose, no markdown, no code fences) matching this schema:\n"
            "{\n"
            '  "type": "ManuscriptEdit",\n'
            '  "target_filename": string (REQUIRED),\n'
            '  "scope": one of ["paragraph", "chapter", "multi_chapter"] (REQUIRED),\n'
            '  "summary": string (REQUIRED),\n'
            '  "chapter_index": integer|null (optional),\n'
            '  "safety": one of ["low", "medium", "high"] (REQUIRED),\n'
            '  "operations": [\n'
            "    {\n"
            '      "op_type": one of ["replace_range", "delete_range", "insert_after_heading", "insert_after"] (REQUIRED),\n'
            '      "start": integer (REQUIRED - approximate, anchors take precedence),\n'
            '      "end": integer (REQUIRED - approximate, anchors take precedence),\n'
            '      "text": string (REQUIRED - new prose for replace/insert),\n'
            '      "original_text": string (⚠️ REQUIRED for replace_range/delete_range - EXACT 20-40 words from manuscript!),\n'
            '      "anchor_text": string (⚠️ REQUIRED for insert_after_heading - EXACT complete line/paragraph to insert after!),\n'
            '      "left_context": string (optional - text before target),\n'
            '      "right_context": string (optional - text after target),\n'
            '      "occurrence_index": integer (optional - which occurrence, 0-based, default 0)\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "⚠️ ⚠️ ⚠️ CRITICAL FIELD REQUIREMENTS:\n"
            "- replace_range → MUST include 'original_text' with EXACT text from manuscript (MINIMUM size for uniqueness, typically 10-40 words)\n"
            "  * PREFER SMALLER matches (10-20 words) when possible - only use larger matches when necessary for uniqueness\n"
            "  * For word-level changes: Use minimal context (10-15 words) around the word/phrase\n"
            "  * For sentence changes: Use just the sentence(s) that need changing (15-30 words)\n"
            "  * Only use 30-40 words when smaller matches aren't unique enough\n"
            "- delete_range → MUST include 'original_text' with EXACT text to delete (minimal size for uniqueness)\n"
            "- insert_after_heading → MUST include 'anchor_text' with EXACT complete line/paragraph to insert after\n"
            "  * EXCEPTION: For EMPTY files (only frontmatter, no chapters), anchor_text is OPTIONAL\n"
            "  * For empty files: Use insert_after_heading WITHOUT anchor_text to create the first chapter\n"
            "- If you don't provide these fields, the operation will FAIL!\n\n"
            "OUTPUT RULES:\n"
            "- Output MUST be a single JSON object only.\n"
            "- Do NOT include triple backticks or language tags.\n"
            "- Do NOT include explanatory text before or after the JSON.\n\n"
            "=== THREE FUNDAMENTAL OPERATIONS ===\n\n"
            "**1. replace_range**: Replace existing text with new text\n"
            "   USE WHEN: User wants to revise, improve, change, modify, or rewrite existing prose\n"
            "   ANCHORING: Provide 'original_text' with EXACT, VERBATIM text from manuscript\n\n"
            "   **⚠️ CRITICAL: PREFER GRANULAR, WORD-LEVEL EDITS ⚠️**\n\n"
            "   **DEFAULT PREFERENCE: Make the SMALLEST possible edit that achieves the user's goal.**\n"
            "   - If changing one word → change ONLY that word (with minimal surrounding context for uniqueness)\n"
            "   - If changing a phrase → change ONLY that phrase (with minimal surrounding context)\n"
            "   - If changing a sentence → change ONLY that sentence (not the entire paragraph)\n"
            "   - If changing multiple sentences → use MULTIPLE operations (one per sentence) rather than one large operation\n"
            "   - Only use large paragraph-level edits when absolutely necessary (major rewrites, structural changes)\n\n"
            "   **GRANULAR EDIT PRINCIPLES:**\n"
            "   1. **Minimize 'original_text' size**: Use the SMALLEST unique text match possible\n"
            "      - For word changes: Include just enough context (10-20 words) to uniquely identify the location\n"
            "      - For phrase changes: Include the phrase plus minimal surrounding context (15-25 words)\n"
            "      - For sentence changes: Include just the sentence (or 2-3 sentences if needed for uniqueness)\n"
            "      - ⚠️ DO NOT default to 20-40 words - use the MINIMUM needed for reliable matching\n\n"
            "   2. **Preserve surrounding text**: Only change what needs changing\n"
            "      - Keep all text before and after the edit unchanged\n"
            "      - Match whitespace, punctuation, and formatting exactly\n"
            "      - Example: If changing 'canoe' to 'boat' in 'He paddled the canoe across the river':\n"
            "        * original_text: 'He paddled the canoe across the river' (minimal, unique match)\n"
            "        * text: 'He paddled the boat across the river' (ONLY the word changed)\n\n"
            "   3. **Break large edits into multiple operations**: If editing multiple locations, use separate operations\n"
            "      - One operation per sentence/phrase that needs changing\n"
            "      - Each operation with its own minimal 'original_text' match\n"
            "      - This ensures precise alignment and easier validation\n\n"
            "   4. **Word-level precision examples:**\n"
            "      - User: 'boat not canoe' → Find 'canoe' in context, replace ONLY 'canoe' with 'boat'\n"
            "      - User: 'should be 24, not 23' → Find '23' in context, replace ONLY '23' with '24'\n"
            "      - User: 'more descriptive' → Find the specific sentence needing description, enhance ONLY that sentence\n"
            "      - User: 'fix the dialogue' → Find the specific dialogue line, fix ONLY that line\n\n"
            "   **WHEN LARGE EDITS ARE REQUIRED (AND FULLY SUPPORTED):**\n"
            "   - **Complete block removal**: When user wants to delete entire paragraphs, scenes, or sections\n"
            "     * Use delete_range with the FULL block in 'original_text' (entire paragraph/scene)\n"
            "     * Example: User says 'remove this paragraph' → delete_range with full paragraph\n\n"
            "   - **Complete block replacement**: When user wants to rewrite entire paragraphs, scenes, or sections\n"
            "     * Use replace_range with the FULL block in 'original_text' (entire paragraph/scene)\n"
            "     * Example: User says 'rewrite this paragraph' → replace_range with full paragraph\n\n"
            "   - **Major structural rewrites**: When entire scenes need reworking or restructuring\n"
            "   - **Adding substantial new content**: When inserting large blocks of new prose within existing paragraphs\n"
            "   - **Multi-sentence rewrites**: When sentences are tightly interconnected and must change together\n"
            "   - **Fundamental paragraph issues**: When a paragraph fundamentally doesn't work and needs complete replacement\n"
            "   - **User explicitly requests large changes**: When user says 'rewrite this paragraph', 'replace this scene', 'remove this section', etc.\n\n"
            "   **KEY PRINCIPLE**: Match edit size to the user's request and the scope of change needed.\n"
            "   - **Granular edits** for PRECISION: word/phrase changes, single sentence fixes\n"
            "     * If changing a few words → granular edit (small original_text match, 10-20 words)\n"
            "     * If fixing one sentence → granular edit (just that sentence, 15-30 words)\n\n"
            "   - **Large edits** for SCOPE: entire blocks, scenes, or major rewrites\n"
            "     * If rewriting entire paragraph → large edit (full paragraph in original_text, 50-200+ words)\n"
            "     * If removing entire scene → large delete_range (full scene in original_text, 100-500+ words)\n"
            "     * If user says 'rewrite this' or 'replace this section' → large edit is appropriate\n"
            "     * If entire block fundamentally doesn't work → large replacement is appropriate\n\n"
            "   **BOTH ARE VALID**: Use granular when precision is needed, use large when scope requires it.\n\n"
            "   **WHEN TO USE MULTIPLE SMALL OPERATIONS:**\n"
            "   - Editing multiple sentences in different locations → One operation per sentence\n"
            "   - Fixing multiple word/phrase issues → One operation per issue\n"
            "   - Adding description in multiple places → One operation per location\n"
            "   - This ensures each edit aligns precisely with the document\n\n"
            "**2. insert_after_heading**: Insert new text AFTER a specific location WITHOUT replacing\n"
            "   USE WHEN: User wants to add, append, or insert NEW content (not replace existing)\n"
            "   ⚠️ THIS IS FOR NEW TEXT THAT DOESN'T EXIST YET - use this when adding content!\n"
            "   ANCHORING: Provide 'anchor_text' with EXACT, COMPLETE, VERBATIM paragraph/sentence to insert after\n"
            "   - Find the sentence/paragraph in the manuscript where the new text should go\n"
            "   - Copy that sentence/paragraph EXACTLY as 'anchor_text'\n"
            "   - The new text will be inserted immediately after that anchor\n"
            "   - Example: If inserting after 'She closed the door.', use anchor_text='She closed the door.'\n\n"
            "**3. delete_range**: Remove text\n"
            "   USE WHEN: User wants to delete, remove, or cut content\n"
            "   ANCHORING: Provide 'original_text' with EXACT text to delete\n\n"
            "=== CHAPTER BOUNDARIES ARE SACRED ===\n\n"
            "Chapters are marked by \"## Chapter N\" headings.\n"
            "⚠️ CRITICAL: NEVER include the next chapter's heading in your operation!\n\n"
            "=== CRITICAL TEXT PRECISION REQUIREMENTS ===\n\n"
            "For 'original_text' and 'anchor_text' fields:\n"
            "- Must be EXACT, COMPLETE, and VERBATIM from the current manuscript text (not from outline, not from any reference documents)\n"
            "- ⚠️⚠️⚠️ NEVER EVER use text from the outline for anchors - outline text does NOT exist in the manuscript!\n"
            "- The outline tells you WHAT happens (story beats), but those words aren't in the manuscript\n"
            "- You must copy text from the '=== MANUSCRIPT CONTEXT ===' sections (current chapter, previous chapter, next chapter)\n"
            "- Include ALL whitespace, line breaks, and formatting exactly as written\n"
            "- Include complete sentences or natural text boundaries (periods, paragraph breaks)\n"
            "- NEVER paraphrase, summarize, or reformat the text\n"
            "- Minimum 10-20 words for unique identification\n"
            "- ⚠️ NEVER include chapter headers (##) in original_text for replace_range!\n\n"
            "=== CREATIVE ADDITIONS POLICY ===\n\n"
            "**You have creative freedom to enhance the story with additions:**\n\n"
            "When the user requests additions, enhancements, or expansions, you may add story elements\n"
            "that are NOT explicitly in the outline, as long as they maintain consistency.\n\n"
            "**MANDATORY CONSISTENCY CHECKS for all additions:**\n"
            "Before adding ANY new story element, verify:\n"
            "1. ✅ Style Guide compliance - matches established voice/tone/pacing\n"
            "2. ✅ Universe Rules compliance - no violations of established physics/magic/tech\n"
            "3. ✅ Character consistency - behavior matches character profiles\n"
            "4. ✅ Manuscript continuity - no contradictions with established facts\n"
            "5. ✅ Timeline coherence - events fit logically in story sequence\n\n"
            "**ALLOWED additions (enhance without changing plot):**\n"
            "- Sensory details and atmospheric descriptions\n"
            "- Internal character thoughts and emotional reactions\n"
            "- Brief character interactions that deepen relationships\n"
            "- Worldbuilding details that enrich the setting\n"
            "- Transitional moments that improve flow\n"
            "- Foreshadowing elements for later story beats\n"
            "- Tension-building moments within existing scenes\n"
            "- Character vulnerability or growth moments\n"
            "- Dialogue that reveals character or advances relationships\n\n"
            "**FORBIDDEN additions (require user approval):**\n"
            "- Major plot events not in outline (character deaths, revelations, etc.)\n"
            "- New characters with significant roles\n"
            "- World-altering events or discoveries\n"
            "- Changes to story direction or timeline\n"
            "- Events that contradict outline's plot structure\n\n"
            "**When uncertain about an addition:**\n"
            "Use 'clarifying_questions' field to ask:\n"
            "- 'Adding [X] would enhance [Y], but it's not in the outline. Should I include it?'\n"
            "- 'This addition might affect [later plot point]. Should I proceed?'\n"
            "- 'The outline doesn't specify [detail]. Should I add [specific element]?'\n\n"
            "**Default behavior:**\n"
            "- If generating full chapter from outline → Follow outline structure, add enriching details\n"
            "- If user says 'stick to outline exactly' → Strict adherence, minimal additions\n"
            "- If user says 'add/enhance/expand/enrich' → Creative freedom with consistency checks\n"
            "- If editing existing prose → Full creative freedom with consistency checks\n\n"
            "=== CONSISTENCY VALIDATION FRAMEWORK ===\n\n"
            "For EVERY operation, especially creative additions, validate against ALL references:\n\n"
            "**Style Guide Validation:**\n"
            "- Narrative voice (POV, tense, formality level)\n"
            "- Sentence structure patterns and rhythm\n"
            "- Dialogue style and character speech patterns\n"
            "- Pacing and scene construction approach\n"
            "- Descriptive style (minimalist vs. rich, etc.)\n\n"
            "**Universe Rules Validation:**\n"
            "- Physics/magic/technology constraints\n"
            "- Cultural and social rules\n"
            "- Historical facts and timeline\n"
            "- Geographic and environmental limits\n"
            "- Power systems and their limitations\n\n"
            "**Character Profile Validation:**\n"
            "- Core personality traits and behaviors\n"
            "- Motivations and goals\n"
            "- Speech patterns and vocabulary\n"
            "- Relationships and dynamics with other characters\n"
            "- Backstory and experiences that shape decisions\n"
            "- Emotional state and character arc position\n\n"
            "**Manuscript Continuity Validation:**\n"
            "- Established facts from earlier chapters\n"
            "- Character emotional states and development\n"
            "- Ongoing plot threads and setups\n"
            "- Previously mentioned details (locations, objects, events)\n"
            "- Cause-and-effect relationships\n"
            "- Character knowledge and awareness\n\n"
            "**When adding new elements, ask yourself:**\n"
            "- Would this character realistically do/say this based on their profile?\n"
            "- Does this violate any established universe rules?\n"
            "- Does this contradict anything in previous chapters?\n"
            "- Does this match the established narrative voice?\n"
            "- Will this create problems for outlined future events?\n\n"
            "**CRITICAL: Continuity Maintenance Requirements:**\n"
            "- BEFORE making any revision, check if it would break continuity with adjacent chapters or previously established facts\n"
            "- If your revision would create a contradiction or inconsistency, you MUST provide additional operations to fix it\n"
            "- You may need to edit multiple locations in the current chapter to maintain continuity\n"
            "- Example: If you change a character's age in Chapter 3, but Chapter 2 mentioned their age, provide operations to update both\n"
            "- Example: If you change a location name, search the current chapter for other references and update them all\n"
            "- Example: If you change a character's emotional state, ensure it's consistent with how the chapter began\n"
            "- Include ALL necessary operations in your operations array to ensure complete continuity\n"
            "- If fixing continuity requires editing text outside the current chapter, note this in your summary and ask the user\n"
            "- Only use clarifying_questions if the continuity issue is ambiguous or requires user decision\n\n"
            "=== NARRATIVE CRAFT PRINCIPLES ===\n\n"
            "**Show, Don't Tell:**\n"
            "- Reveal character emotions through actions, dialogue, and physical reactions, not statements\n"
            "- Build atmosphere through sensory details (sight, sound, smell, texture, temperature)\n"
            "- Let readers infer meaning from scene details rather than explaining directly\n"
            "- Example: Instead of 'Peterson was worried,' show: 'Peterson's fingers drummed the desk, each tap louder than the last.'\n\n"
            "**Scene-Building vs Summary:**\n"
            "- Write complete scenes with setting, action, dialogue, and character internality\n"
            "- Avoid summary prose that reports events ('He went to the office and found the file')\n"
            "- Build scenes moment-by-moment with specific details and natural pacing\n"
            "- Let story events emerge organically within scenes, not as mechanical beat-conversion\n"
            "- Transitions between beats should flow naturally, not feel like checklist items\n\n"
            "**Character Voice and Dialogue:**\n"
            "- Dialogue must sound natural and character-specific, not expository or mechanical\n"
            "- Each character's speech patterns should reflect their personality, background, and emotional state\n"
            "- Avoid dialogue that merely conveys plot information - let characters speak as real people would\n"
            "- Internal thoughts should match the character's voice and perspective (POV)\n\n"
            "**Sensory Details and Atmosphere:**\n"
            "- Ground every scene in specific sensory details (what characters see, hear, feel, smell, taste)\n"
            "- Use atmospheric details to establish mood and tone, not just setting\n"
            "- Balance sensory detail level according to Style Guide requirements\n"
            "- Create immersive scenes that readers can experience, not just observe\n"
            "- **VARY descriptions to avoid repetition** - use different sensory details, perspectives, and phrasings\n"
            "- When describing similar elements (characters, locations, actions), find fresh angles and details\n"
            "- Draw from character profiles and outline context to inform varied descriptions\n"
            "- Example: Instead of repeatedly saying 'her blue eyes,' vary with 'eyes the color of storm clouds,' 'her gaze sharp as ice,' 'blue eyes that seemed to see through him'\n"
            "- **CRITICAL**: Variation must still match Style Guide voice and character profiles - don't vary so much that it breaks consistency\n\n"
            "**Organic Pacing vs Mechanical Beat-Following:**\n"
            "- Write scenes that flow naturally with appropriate pacing for the moment\n"
            "- Don't rush through beats to 'cover' all outline points - let scenes breathe\n"
            "- Build tension, emotion, and character development organically within scenes\n"
            "- Outline beats are story goals to achieve, not items to check off sequentially\n"
            "- A single scene can achieve multiple outline beats naturally if the story flows that way\n"
            "- Conversely, a single outline beat might require multiple scenes if the story demands it\n\n"
            "**Style Guide Integration:**\n"
            "- Every sentence must sound like it was written in the Style Guide's voice\n"
            "- Apply Style Guide techniques (POV, tense, pacing, dialogue style) consistently throughout\n"
            "- If Style Guide includes writing samples, emulate their technique and voice, never copy content\n"
            "- The Style Guide voice should be so internalized that it feels natural, not forced\n\n"
            "=== CONTENT GENERATION RULES ===\n\n"
            "1. operations[].text MUST contain final prose (no placeholders or notes)\n"
            "2. For chapter generation: begin with '## Chapter N'\n"
            "3. Do NOT impose a word count limit. Write as much prose as needed to produce a complete, compelling scene.\n"
            "4. If outline present: Treat outline beats as plot objectives to achieve, NOT text to expand or reuse.\n"
            "   - Create fresh scenes with original dialogue, action, and narration in the Style Guide voice.\n"
            "   - Do not copy or lightly paraphrase outline phrasing.\n"
            "   - Write complete narrative with dialogue, action, description, character voice, and emotional depth\n"
            "   - Add all enriching details needed to bring the beats to life as compelling prose\n"
            "4. NO YAML frontmatter in operations[].text\n"
            "5. Match established voice and style\n"
            "6. Complete sentences with proper grammar\n"
            "7. NEVER simply paraphrase outline beats - always craft full narrative prose\n"
        )
    
    async def _validate_fiction_type_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Parse user query for explicit chapter mentions"""
        try:
            logger.info("Detecting chapter mentions in user query...")
            
            current_request = state.get("current_request", "")
            if not current_request:
                logger.info("No current request - skipping chapter detection")
                return {
                    "explicit_primary_chapter": None,
                    "explicit_secondary_chapters": []
                }
            
            import re
            
            # Regex patterns for chapter detection
            CHAPTER_PATTERNS = [
                # Action verbs + Chapter
                r'\b(?:Look over|Review|Check|Edit|Update|Revise|Modify|Address|Fix|Change)\s+[Cc]hapter\s+(\d+)\b',  # "Look over Chapter 2", "Review Chapter 3"
                # Preposition + Chapter
                r'\b(?:in|at|for|to)\s+[Cc]hapter\s+(\d+)\b',  # "in Chapter 2", "at Chapter 3"
                # Chapter + verb
                r'\b[Cc]hapter\s+(\d+)\s+(?:needs|has|shows|contains|should|must|is|requires)',  # "Chapter 2 needs", "Chapter 8 is"
                # Verb + in/at + Chapter
                r'\b(?:address|fix|change|edit|update|revise|modify)\s+(?:in|at)\s+[Cc]hapter\s+(\d+)\b',  # "address in Chapter 8"
                # Chapter + punctuation + relative clause
                r'\b[Cc]hapter\s+(\d+)[:,]?\s+(?:where|when|that|which)',  # "Chapter 2: where", "Chapter 8, that"
            ]
            
            all_mentions = []
            for pattern in CHAPTER_PATTERNS:
                matches = re.finditer(pattern, current_request)
                for match in matches:
                    chapter_num = int(match.group(1))
                    # Store with position to determine primary vs secondary
                    all_mentions.append({
                        "chapter": chapter_num,
                        "position": match.start(),
                        "text": match.group(0)
                    })
            
            # Remove duplicates while preserving order
            seen = set()
            unique_mentions = []
            for mention in all_mentions:
                if mention["chapter"] not in seen:
                    seen.add(mention["chapter"])
                    unique_mentions.append(mention)
            
            # Sort by position in query (first mention = primary)
            unique_mentions.sort(key=lambda x: x["position"])
            
            primary_chapter = None
            secondary_chapters = []
            
            if unique_mentions:
                # First mention is primary (what to edit)
                primary_chapter = unique_mentions[0]["chapter"]
                # Additional mentions are secondary (for context)
                secondary_chapters = [m["chapter"] for m in unique_mentions[1:]]
                
                logger.info(f"📖 Detected chapters in query:")
                logger.info(f"   Primary (to edit): Chapter {primary_chapter}")
                if secondary_chapters:
                    logger.info(f"   Secondary (context): Chapters {secondary_chapters}")
            else:
                logger.info("   No explicit chapter mentions found in query")
            
            return {
                "explicit_primary_chapter": primary_chapter,
                "explicit_secondary_chapters": secondary_chapters
            }
            
        except Exception as e:
            logger.error(f"Failed to detect chapter mentions: {e}")
            return {
                "explicit_primary_chapter": None,
                "explicit_secondary_chapters": []
            }
    
    async def _validate_fiction_type_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Validate that active editor is a fiction manuscript"""
        try:
            frontmatter = state.get("frontmatter", {}) or {}
            fm_type = str(frontmatter.get("type", "")).lower()
            
            if fm_type != "fiction":
                return {
                    "error": "Active editor is not a fiction manuscript; editing agent skipping.",
                    "task_status": "error",
                    "response": {
                        "response": "Active editor is not a fiction manuscript; editing agent skipping.",
                        "task_status": "error",
                        "agent_type": "fiction_editing_agent"
                    }
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to validate fiction type: {e}")
            return {
                "error": str(e),
                "task_status": "error"
            }
    
    def _route_after_validate_type(self, state: FictionEditingState) -> str:
        """Route after fiction type validation"""
        if state.get("task_status") == "error":
            return "error"
        return "continue"
    
    def _route_after_context(self, state: FictionEditingState) -> str:
        """Route after context preparation: check if references exist"""
        # Diagnostic logging - check state after context subgraph
        logger.info("="*80)
        logger.info("🔍 STATE AFTER context_preparation subgraph:")
        logger.info(f"   state.get('current_request'): {repr(state.get('current_request', 'NOT_FOUND'))}")
        logger.info(f"   state.get('query'): {repr(state.get('query', 'NOT_FOUND'))}")
        logger.info(f"   state.get('has_references'): {state.get('has_references', 'NOT_FOUND')}")
        logger.info(f"   state keys: {list(state.keys())[:20]}...")  # First 20 keys
        logger.info("="*80)
        
        # Check for errors first
        if state.get("task_status") == "error":
            return "format_response"  # Skip to format to return error
        
        has_references = state.get("has_references", False)
        
        # If no references, use simple fast path
        if not has_references:
            logger.info("⚡ No references found - using FAST PATH (no validation, no continuity)")
            state["is_simple_request"] = True
            return "simple_path"
        
        # References exist - use full path
        logger.info("📚 References found - using full workflow path")
        return "full_path"
    
    async def _check_simple_request_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Check if request is simple enough to skip full workflow (no references needed)"""
        try:
            logger.info("🔍 Checking if request can use simple path...")
            
            current_request = state.get("current_request", "")
            current_chapter_text = state.get("current_chapter_text", "")
            cursor_offset = state.get("cursor_offset", -1)
            
            # Build prompt to ask LLM if this is a simple request
            prompt = f"""Analyze this user request and determine if it's a SIMPLE request that can be handled with just the manuscript context (no outline, rules, style, or character references needed).

**USER REQUEST**: {current_request}

**CONTEXT**:
- Current chapter text length: {len(current_chapter_text)} characters
- Cursor position: {cursor_offset}
- No references available (no outline, rules, style, or characters)

**SIMPLE REQUESTS** (can use fast path):
- Continuation requests: "continue the paragraph", "finish the paragraph", "write the next paragraph", "keep going", "continue writing"
- Simple revisions: "revise this for clarity", "make this clearer", "improve this sentence", "tighten this paragraph"
- Simple edits: "fix the grammar here", "correct the spelling", "add a comma", "remove this sentence"
- Free-writing: "write more", "expand this", "add detail here"
- These requests only need the manuscript context around the cursor/selection

**COMPLEX REQUESTS** (need full workflow):
- Requests that need story structure: "add a scene where...", "introduce a character", "follow the outline"
- Requests that need style guide: "match the style guide", "use the established voice"
- Requests that need character consistency: "make sure Tom's dialogue matches his profile"
- Requests that need plot continuity: "ensure this follows the plot", "check continuity"
- Multi-chapter requests: "generate chapters 2-5", "write the next 3 chapters"
- Questions about the story: "what happens next?", "does this follow the outline?"

**OUTPUT**: Return ONLY valid JSON:
{{
  "is_simple": true | false,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why this is simple or complex"
}}

**CRITICAL**: 
- If request is a simple continuation/revision that only needs manuscript context → is_simple: true
- If request needs references, structure, or complex analysis → is_simple: false
- When in doubt, choose false (use full workflow)"""
            
            # Call LLM with structured output
            llm = self._get_llm(temperature=0.1, state=state)  # Low temperature for consistent classification
            
            messages = [
                SystemMessage(content="You are a request classifier. Analyze user requests and determine if they can be handled simply with just manuscript context, or if they need the full workflow with references."),
                HumanMessage(content=prompt)
            ]
            
            response = await self._safe_llm_invoke(llm, messages, state)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            content = _unwrap_json_response(content)
            
            try:
                result = json.loads(content)
                is_simple = result.get("is_simple", False)
                confidence = result.get("confidence", 0.5)
                reasoning = result.get("reasoning", "")
                
                logger.info(f"Simple request check: is_simple={is_simple} (confidence: {confidence:.0%}, reasoning: {reasoning})")
                
                # Default to full path if confidence is low
                if confidence < 0.6:
                    logger.warning(f"Low confidence ({confidence:.0%}) - defaulting to full workflow")
                    is_simple = False
                
                return {
                    "is_simple_request": is_simple
                }
                
            except Exception as parse_error:
                logger.error(f"Failed to parse simple request check: {parse_error}")
                logger.warning("Defaulting to full workflow due to parse error")
                return {"is_simple_request": False}
            
        except Exception as e:
            logger.error(f"Failed to check simple request: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Default to full workflow on error
            return {"is_simple_request": False}
    
    def _route_from_simple_check(self, state: FictionEditingState) -> str:
        """Route based on simple request check result"""
        is_simple = state.get("is_simple_request", False)
        
        if is_simple:
            logger.info("⚡ Using simple fast path - skipping references and validation")
            return "simple_path"
        else:
            logger.info("📚 Using full workflow path")
            return "full_path"
    
    async def _generate_simple_edit_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Generate edit using only manuscript context (fast path for simple requests)"""
        try:
            logger.info("⚡ Generating simple edit (fast path - no references)...")
            
            # Mark state as simple request for downstream nodes
            state["is_simple_request"] = True
            
            manuscript = state.get("manuscript", "")
            filename = state.get("filename", "manuscript.md")
            
            # Diagnostic logging for current_request
            logger.info("="*80)
            logger.info("🔍 CHECKING current_request in _generate_simple_edit_node:")
            logger.info(f"   state.get('current_request'): {repr(state.get('current_request', 'NOT_FOUND'))}")
            logger.info(f"   state.get('query'): {repr(state.get('query', 'NOT_FOUND'))}")
            messages = state.get("messages", [])
            logger.info(f"   state.get('messages'): {len(messages) if messages else 'NOT_FOUND'} messages")
            if messages:
                latest_message = messages[-1] if messages else None
                logger.info(f"   Latest message type: {type(latest_message)}")
                if latest_message:
                    msg_content = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
                    logger.info(f"   Latest message content: {repr(msg_content[:100])}")
            logger.info("="*80)
            
            current_request = state.get("current_request", "")
            
            if not current_request:
                logger.error("❌ current_request is EMPTY in _generate_simple_edit_node - this should have been set by context subgraph!")
                logger.error("   This indicates a state flow issue between context subgraph and simple edit node")
                return {
                    "error": "No user request found - state flow issue",
                    "task_status": "error"
                }
            
            logger.info(f"📝 User request for simple edit: {current_request[:100]}")
            
            current_chapter_text = state.get("current_chapter_text", "")
            cursor_offset = state.get("cursor_offset", -1)
            selection_start = state.get("selection_start", -1)
            selection_end = state.get("selection_end", -1)
            
            # If cursor is -1, treat it as end of document
            if cursor_offset == -1:
                cursor_offset = len(manuscript)
                logger.info(f"🔍 Cursor at -1 (end of document), setting to manuscript length: {cursor_offset}")
            
            # Use chapter context if available, otherwise fall back to full manuscript
            prev_chapter_text = state.get("prev_chapter_text", "")
            next_chapter_text = state.get("next_chapter_text", "")
            
            if current_chapter_text:
                # Use full chapter context (with prev/next chapters if available)
                context_parts = []
                if prev_chapter_text:
                    context_parts.append(f"[PREVIOUS CHAPTER]\n{prev_chapter_text}\n")
                context_parts.append(f"[CURRENT CHAPTER]\n{current_chapter_text}")
                if next_chapter_text:
                    context_parts.append(f"\n[NEXT CHAPTER]\n{next_chapter_text}")
                context_text = "\n".join(context_parts)
                logger.info(f"🔍 Using chapter context: current={len(current_chapter_text)} chars, prev={len(prev_chapter_text)}, next={len(next_chapter_text)}")
            else:
                # No chapters found - use entire manuscript for consistency
                # Without chapter structure, LLM needs full context to maintain character names, plot points, etc.
                context_text = manuscript
                logger.info(f"🔍 No chapters found - using ENTIRE manuscript for consistency: {len(context_text)} chars")
            
            if len(context_text) < 100:
                logger.warning(f"⚠️ Very short context ({len(context_text)} chars) - may not be enough for continuation")
            
            # Build simple prompt focused on manuscript context only
            system_prompt = (
                "You are a fiction editor. Generate precise, granular edits based ONLY on the manuscript context provided.\n\n"
                "**CRITICAL EDITING RULES**:\n"
                "1. Use GRANULAR edits - prefer word-level or sentence-level changes\n"
                "2. Keep original_text small (10-20 words) for precise matching\n"
                "3. Break large edits into multiple operations\n"
                "4. For continuing/adding text: use 'insert_after' with anchor_text = last few words\n"
                "5. For replacements: use 'replace_range' with original_text\n"
                "6. For deletions: use 'delete_range' with original_text\n"
                "7. Match the established voice and style from the manuscript\n"
                "8. Complete sentences with proper grammar\n"
            )
            
            user_prompt = f"""**USER REQUEST**: {current_request}

**MANUSCRIPT CONTEXT** (around cursor position {cursor_offset}):
{context_text}

**SELECTION** (if any):
{f"Selected text: {manuscript[selection_start:selection_end]}" if selection_start >= 0 and selection_end > selection_start else "No selection"}

**INSTRUCTIONS**:
- Generate edits based ONLY on the manuscript context above
- Use the text around the cursor/selection to find anchor points
- **FOR CONTINUATION REQUESTS** (e.g., "continue the paragraph", "finish the sentence", "write more"):
  - **PREFER** to generate at least ONE operation with new text
  - Use "insert_after" operation type with anchor_text = last few words before cursor
  - Continue naturally from where the text ends, matching the style and voice
  - Generate substantial continuation (at least 1-2 sentences, not just a few words)
  - **IF YOU CANNOT CONTINUE** (e.g., unclear context, missing information, concerns about style/consistency):
    - Return empty operations array
    - **EXPLAIN CLEARLY in the summary** why you cannot proceed and what information/clarification you need
- **FOR REVISION REQUESTS**: Make precise, targeted changes
- Keep edits granular and focused
- Always include target_filename and scope in your response
- **IMPORTANT**: Empty operations are allowed if you have legitimate concerns - but you MUST explain them in the summary field

**OUTPUT FORMAT**: Return ONLY valid JSON matching this schema:
{{
  "target_filename": "{filename}",
  "scope": "paragraph" | "chapter" | "multi_chapter",
  "summary": "Brief description of what was done",
  "operations": [
    {{
      "op_type": "replace_range" | "insert_after" | "delete_range",
      "original_text": "exact text to find (10-20 words for uniqueness) - REQUIRED for replace_range/delete_range",
      "anchor_text": "last few words of text to insert after - REQUIRED for insert_after",
      "text": "new text content",
      "occurrence_index": 0
    }}
  ]
}}

**CRITICAL scope VALUES** (use EXACTLY one of these):
- "paragraph" - edits within a single paragraph (most common for simple requests)
- "chapter" - edits spanning a single chapter
- "multi_chapter" - edits spanning multiple chapters

**CRITICAL op_type VALUES** (use EXACTLY these):
- "replace_range" - replace existing text (requires original_text)
- "insert_after" - insert new text after existing text (requires anchor_text with last few words)
- "delete_range" - delete existing text (requires original_text)

**CRITICAL**: 
- Return ONLY the JSON object, no markdown, no code blocks
- **Empty operations are allowed** if you have concerns or need clarification
- **IF RETURNING EMPTY OPERATIONS**: You MUST provide a clear explanation in the summary field explaining:
  - Why you cannot proceed (e.g., "Context is unclear", "Need clarification on X", "Concerned about Y")
  - What information would help you proceed
  - Any specific questions you have
- **PREFER to generate operations** when possible, but prioritize accuracy and clarity over forcing edits
- Use original_text from the manuscript context above for precise matching"""
            
            # Call LLM
            llm = self._get_llm(temperature=0.7, state=state)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self._safe_llm_invoke(llm, messages, state)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Log raw response for debugging
            logger.info(f"🔍 LLM raw response (first 500 chars): {content[:500]}")
            
            # Parse JSON response
            content = _unwrap_json_response(content)
            
            try:
                result = json.loads(content)
                
                # Log parsed result for debugging
                logger.info(f"🔍 Parsed result: operations={len(result.get('operations', []))}, scope={result.get('scope')}, summary={result.get('summary', '')[:100]}")
                
                # Fix invalid scope values before validation
                if "scope" in result:
                    scope = result["scope"]
                    # Map common invalid values to valid ones
                    scope_mapping = {
                        "scene": "paragraph",
                        "sentence": "paragraph",
                        "word": "paragraph",
                        "section": "chapter",
                        "page": "chapter",
                        "document": "multi_chapter"
                    }
                    if scope in scope_mapping:
                        logger.warning(f"⚠️ Invalid scope '{scope}' mapped to '{scope_mapping[scope]}'")
                        result["scope"] = scope_mapping[scope]
                    elif scope not in ("paragraph", "chapter", "multi_chapter"):
                        # Default to paragraph for unknown values
                        logger.warning(f"⚠️ Unknown scope '{scope}' defaulting to 'paragraph'")
                        result["scope"] = "paragraph"
                
                # Validate with Pydantic
                structured_edit = ManuscriptEdit(**result)
                
                if len(structured_edit.operations) == 0:
                    summary = structured_edit.summary or "No summary provided"
                    logger.info(f"ℹ️ LLM returned 0 operations for request: {current_request}")
                    logger.info(f"ℹ️ LLM explanation: {summary}")
                    # This is valid - LLM may have concerns or need clarification
                
                logger.info(f"✅ Simple edit generated: {len(structured_edit.operations)} operations")
                
                return {
                    "structured_edit": structured_edit.model_dump(),
                    "llm_response": content,
                    "task_status": "in_progress",
                    "is_simple_request": True  # Mark for downstream routing
                }
                
            except ValidationError as e:
                logger.error(f"Failed to validate simple edit response: {e}")
                return {
                    "error": f"Invalid edit format: {str(e)}",
                    "task_status": "error"
                }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse simple edit JSON: {e}")
                return {
                    "error": f"Invalid JSON: {str(e)}",
                    "task_status": "error"
            }
            
        except Exception as e:
            logger.error(f"Failed to generate simple edit: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "error": str(e),
                "task_status": "error"
            }
    
    async def _detect_mode_and_intent_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Detect generation mode and creative freedom intent from user request"""
        try:
            logger.info("Detecting mode and creative intent...")
            
            current_request = state.get("current_request", "")
            current_request_lower = current_request.lower()
            current_chapter_text = state.get("current_chapter_text", "")
            outline_current_chapter_text = state.get("outline_current_chapter_text")
            
            # Check for multi-chapter request first
            chapter_range = _extract_chapter_range_from_request(current_request)
            is_multi_chapter = chapter_range is not None
            
            # Extract requested chapter number from user request (for single chapter)
            requested_chapter_number = _extract_chapter_number_from_request(current_request) if not is_multi_chapter else None
            
            # Detect creative freedom keywords
            creative_keywords = [
                "add", "enhance", "expand", "enrich", "include", 
                "give", "show", "more", "develop", "deepen"
            ]
            creative_freedom_requested = any(kw in current_request_lower for kw in creative_keywords)
            
            # Detect strict adherence keywords
            strict_keywords = [
                "stick to outline", "follow outline exactly", "only outline", 
                "strictly follow", "outline only"
            ]
            strict_mode_requested = any(kw in current_request for kw in strict_keywords)
            
            # Determine mode
            if len(current_chapter_text.strip()) < 100:
                # Empty or very short chapter - likely generation
                mode = "generation"
            elif creative_freedom_requested and not strict_mode_requested:
                mode = "enhancement"
            else:
                mode = "editing"
            
            # Build mode-specific guidance for LLM
            if mode == "generation":
                mode_guidance = (
                    "\n\n=== MODE: GENERATION ===\n"
                    "You are generating NEW narrative prose from outline beats.\n"
                    "- Transform outline beats into full, vivid narrative scenes\n"
                    "- Craft complete prose with dialogue, action, description, and character voice\n"
                    "- Follow outline structure as story blueprint (not a script to paraphrase)\n"
                    "- Add enriching details: sensory details, character thoughts, emotional depth\n"
                    "- **CRITICAL: Follow ALL Style Guide narrative instructions** (voice, tone, POV, tense, pacing, techniques)\n"
                    "- Maintain Style Guide voice precisely - it overrides default assumptions\n"
                    "- Respect Universe Rules absolutely\n"
                    "- Use Character profiles for authentic behavior and varied descriptions\n"
                    "- **Vary descriptions to avoid repetition** - use different phrasings, details, and perspectives while maintaining consistency\n"
                    "- Do not target a specific word count. Write as much as needed for a complete, compelling scene.\n"
                    "- Treat outline beats as plot objectives; do not copy or lightly paraphrase outline phrasing\n"
                )
            elif mode == "enhancement":
                mode_guidance = (
                    "\n\n=== MODE: CREATIVE ENHANCEMENT ===\n"
                    "You have creative freedom to add story elements.\n"
                    "- User has requested additions/enhancements\n"
                    "- Add elements that enrich the narrative\n"
                    "- CRITICAL: Maintain consistency with all references\n"
                    "- Validate additions against Style/Rules/Characters/Continuity\n"
                    "- Use clarifying_questions if additions might conflict with outline\n"
                )
            else:
                mode_guidance = (
                    "\n\n=== MODE: EDITING ===\n"
                    "You are revising EXISTING content.\n"
                    "- Maintain consistency with established story\n"
                    "- Respect existing character development\n"
                    "- Keep tone consistent unless explicitly asked to change\n"
                )
            
            return {
                "generation_mode": mode,
                "creative_freedom_requested": creative_freedom_requested or mode == "enhancement",
                "mode_guidance": mode_guidance,
                "requested_chapter_number": requested_chapter_number,
                "is_multi_chapter": is_multi_chapter,
                "chapter_range": chapter_range
            }
            
        except Exception as e:
            logger.error(f"Mode detection failed: {e}")
            return {
                "generation_mode": "editing",
                "creative_freedom_requested": False,
                "mode_guidance": "",
                "requested_chapter_number": None,
                "is_multi_chapter": False,
                "chapter_range": None
            }
    
    async def _detect_request_type_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Detect if user request is a question or an edit request - trust LLM to figure it out"""
        try:
            logger.info("Detecting request type (question vs edit request)...")
            
            current_request = state.get("current_request", "")
            if not current_request:
                logger.warning("No current request found - defaulting to edit_request")
                return {"request_type": "edit_request"}
            
            current_chapter_text = state.get("current_chapter_text", "")
            style_body = state.get("style_body")
            rules_body = state.get("rules_body")
            characters_bodies = state.get("characters_bodies", [])
            
            # Build simple prompt for LLM to determine intent
            prompt = f"""Analyze the user's request and determine if it's a QUESTION or an EDIT REQUEST.

**USER REQUEST**: {current_request}

**CONTEXT**:
- Current chapter: {current_chapter_text[:500] if current_chapter_text else "No chapter selected"}
- Has style reference: {bool(style_body)}
- Has rules reference: {bool(rules_body)}
- Has {len(characters_bodies)} character reference(s)

**INTENT DETECTION**:
- QUESTIONS (including pure questions and conditional edits): User is asking a question - may or may not want edits
  - Pure questions: "How old is Tom here?", "What style is this?", "Does this follow our style guide?"
  - Conditional edits: "Is Tom 23? We want him to be 24", "Are we using enough description? Revise if necessary"
  - Questions often start with: "How", "Why", "What", "Does", "Is", "Can you", "Where", "Are we"
  - **Key insight**: Questions can be answered, and IF edits are needed based on the answer, they can be made
  - Route ALL questions to edit path - LLM can decide if edits are needed
  
- EDIT REQUESTS: User wants to create, modify, or generate content - NO question asked
  - Examples: "Add a scene", "Revise Chapter 2", "Generate prose for this chapter", "Change the dialogue"
  - Edit requests are action-oriented: "add", "create", "update", "generate", "change", "replace", "revise", "write"
  - Edit requests specify what content to create or modify
  - **Key indicator**: Action verbs present, no question asked

**OUTPUT**: Return ONLY valid JSON:
{{
  "request_type": "question" | "edit_request",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why this classification"
}}

**CRITICAL**: 
- If request contains a question (even with action verbs) → "question" (will route to edit path, LLM decides if edits needed)
- If request is ONLY action verbs with NO question → "edit_request"
- Trust your semantic understanding - questions go to edit path where LLM can analyze and optionally edit"""
            
            # Call LLM with structured output
            llm = self._get_llm(temperature=0.1, state=state)  # Low temperature for consistent classification
            
            datetime_context = self._get_datetime_context()
            messages = [
                SystemMessage(content="You are an intent classifier. Analyze user requests and determine if they are questions or edit requests. Return only valid JSON."),
                SystemMessage(content=datetime_context),
                HumanMessage(content=prompt)
            ]
            
            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            content = _unwrap_json_response(content)
            
            try:
                result = json.loads(content)
                request_type = result.get("request_type", "edit_request")
                confidence = result.get("confidence", 0.5)
                reasoning = result.get("reasoning", "")
                
                # Default to edit_request if confidence is low
                if confidence < 0.6:
                    logger.warning(f"Low confidence ({confidence:.0%}) - defaulting to edit_request")
                    request_type = "edit_request"
                
                return {
                    "request_type": request_type
                }
                
            except Exception as parse_error:
                logger.error(f"Failed to parse request type detection: {parse_error}")
                logger.warning("Defaulting to edit_request due to parse error")
                return {"request_type": "edit_request"}
            
        except Exception as e:
            logger.error(f"Failed to detect request type: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Default to edit_request on error
            return {"request_type": "edit_request"}
    
    def _route_from_request_type(self, state: FictionEditingState) -> str:
        """Route based on detected request type"""
        request_type = state.get("request_type", "edit_request")
        # ALL questions route to edit path (can analyze and optionally edit)
        # Pure questions will analyze and return no edits; conditional questions will analyze and edit if needed
        # Both question and edit_request route to the same path (check_multi_chapter)
        return "edit_request"  # Both question and edit_request go to edit path
    
    async def _answer_question_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Answer user questions about the manuscript, references, or general information"""
        try:
            logger.info("Answering user question...")
            
            current_request = state.get("current_request", "")
            manuscript = state.get("manuscript", "")
            current_chapter_text = state.get("current_chapter_text", "")
            current_chapter_number = state.get("current_chapter_number")
            chapter_ranges = state.get("chapter_ranges", [])
            style_body = state.get("style_body")
            rules_body = state.get("rules_body")
            characters_bodies = state.get("characters_bodies", [])
            outline_body = state.get("outline_body")
            outline_current_chapter_text = state.get("outline_current_chapter_text")
            
            # ALWAYS use cursor-based chapter detection (from _analyze_scope_node)
            # This ensures we show the chapter where the cursor actually is
            chapter_text_to_show = current_chapter_text
            chapter_number_to_show = current_chapter_number
            prev_chapter_text = state.get("prev_chapter_text")
            next_chapter_text = state.get("next_chapter_text")
            
            # Log what we're showing
            logger.info(f"📖 Chapter context for question: Chapter {chapter_number_to_show} (cursor-based)")
            if prev_chapter_text:
                logger.info(f"📖 Previous chapter available ({len(prev_chapter_text)} chars)")
            if next_chapter_text:
                logger.info(f"📖 Next chapter available ({len(next_chapter_text)} chars)")
            logger.info(f"📖 Current chapter text length: {len(chapter_text_to_show) if chapter_text_to_show else 0} chars")
            
            # Build context for answering
            context_parts = []
            
            # Include conversation history
            messages = state.get("messages", [])
            conversation_history = self._format_conversation_history_for_prompt(messages, look_back_limit=6)
            if conversation_history:
                context_parts.append(conversation_history)
            
            context_parts.append("=== USER QUESTION ===\n")
            context_parts.append(f"{current_request}\n\n")
            
            # Chapter context - show current, previous, and next chapters (cursor-based)
            if prev_chapter_text:
                prev_chapter_num = chapter_number_to_show - 1 if chapter_number_to_show else None
                context_parts.append(f"=== PREVIOUS CHAPTER {prev_chapter_num or 'N'} (FOR CONTEXT) ===\n")
                context_parts.append(f"{prev_chapter_text}\n\n")
            
            if chapter_text_to_show:
                context_parts.append(f"=== CURRENT CHAPTER {chapter_number_to_show or 'N'} (CURSOR LOCATION) ===\n")
                # Show FULL chapter for questions (not truncated) - this is the chapter where cursor is
                context_parts.append(f"{chapter_text_to_show}\n\n")
            elif manuscript:
                context_parts.append("=== MANUSCRIPT (NO CHAPTER DETECTED) ===\n")
                context_parts.append(f"{manuscript[:2000]}\n\n")
            
            if next_chapter_text:
                next_chapter_num = chapter_number_to_show + 1 if chapter_number_to_show else None
                context_parts.append(f"=== NEXT CHAPTER {next_chapter_num or 'N'} (FOR CONTEXT) ===\n")
                context_parts.append(f"{next_chapter_text}\n\n")
            
            # Reference information
            if style_body:
                context_parts.append("=== STYLE GUIDE (SUMMARY) ===\n")
                style_summary = style_body[:1000] + "..." if len(style_body) > 1000 else style_body
                context_parts.append(f"{style_summary}\n\n")
            
            if rules_body:
                context_parts.append("=== UNIVERSE RULES (SUMMARY) ===\n")
                rules_summary = rules_body[:1000] + "..." if len(rules_body) > 1000 else rules_body
                context_parts.append(f"{rules_summary}\n\n")
            
            if characters_bodies:
                context_parts.append(f"=== CHARACTER PROFILES ({len(characters_bodies)} character(s)) ===\n")
                context_parts.append("**NOTE**: Each character profile below is for a DIFFERENT character with distinct traits and dialogue patterns.\n\n")
                for i, char_body in enumerate(characters_bodies, 1):
                    char_name = _extract_character_name(char_body)
                    char_summary = char_body[:400] + "..." if len(char_body) > 400 else char_body
                    context_parts.append(f"**Character {i}: {char_name}**\n{char_summary}\n\n")
            
            # Outline context for questions:
            # Frame as objectives, not source text, to reduce accidental copying in answers.
            if outline_current_chapter_text or (outline_body and chapter_number_to_show):
                outline_text_for_question = outline_current_chapter_text
                if not outline_text_for_question and outline_body and chapter_number_to_show:
                    import re
                    chapter_pattern = rf"(?i)(?:^|\n)##?\s*(?:Chapter\s+)?{chapter_number_to_show}[:\s]*(.*?)(?=\n##?\s*(?:Chapter\s+)?\d+|\Z)"
                    match = re.search(chapter_pattern, outline_body, re.DOTALL)
                    if match:
                        outline_text_for_question = match.group(1).strip()

                if outline_text_for_question:
                    context_parts.append("=== STORY OUTLINE: PLOT STRUCTURE & STORY BEATS ===\n")
                    context_parts.append("This section contains planned story structure including plot points and beats to achieve.\n")
                    context_parts.append("USE AS CREATIVE GOALS TO ACHIEVE, NOT TEXT TO EXPAND OR COPY.\n\n")
                    context_parts.append("DO NOT:\n")
                    context_parts.append("- Copy outline bullet points into prose or answers\n")
                    context_parts.append("- Expand outline text directly into paragraphs\n")
                    context_parts.append("- Reuse outline phrasing as the basis for sentences\n\n")
                    context_parts.append("INSTEAD DO:\n")
                    context_parts.append("- Use outline as guidance for future scenes and planned events\n")
                    context_parts.append("- Reference manuscript text for what's actually written\n")
                    context_parts.append("- Provide original analysis in your own words\n\n")
                    context_parts.append(f"{outline_text_for_question[:1500]}\n\n")
            
            # Build prompt for answering
            answer_prompt = "".join(context_parts)
            answer_prompt += """**YOUR TASK**: Answer the user's question clearly and helpfully.

- If they're asking about style, explain what style is being used and why
- If they're asking about content, describe what's in the manuscript
- If they're asking about references (characters, rules, style), confirm what's loaded and provide relevant information
- If they're asking about narrative style or writing choices, explain what you observe
- Be conversational and helpful - this is a question, not a content generation request
- If you don't have certain information, say so honestly
- Keep your answer focused and relevant to the question
- You can discuss the manuscript, style, structure, and references - this is a conversation

**OUTPUT**: Provide a natural, conversational answer to the user's question. Do NOT generate JSON or editor operations."""
            
            # Call LLM for conversational response
            llm = self._get_llm(temperature=0.7, state=state)  # Higher temperature for natural conversation
            
            datetime_context = self._get_datetime_context()
            messages = [
                SystemMessage(content="You are a helpful fiction editing assistant. Answer user questions about the manuscript, style, references, and related information. Be conversational and helpful."),
                SystemMessage(content=datetime_context),
                HumanMessage(content=answer_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            answer_text = response.content if hasattr(response, 'content') else str(response)
            
            logger.info(f"Generated answer (first 200 chars): {answer_text[:200]}...")
            
            # Store as response (no operations needed for questions)
            return {
                "response": {
                    "response": answer_text,
                    "task_status": "complete",
                    "agent_type": "fiction_editing_agent",
                    "timestamp": datetime.now().isoformat()
                },
                "task_status": "complete",
                "editor_operations": []  # No operations for questions
            }
            
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "response": {
                    "response": f"I encountered an error while trying to answer your question: {str(e)}",
                    "task_status": "error",
                    "agent_type": "fiction_editing_agent"
                },
                "task_status": "error",
                "error": str(e)
            }
    
    async def _check_multi_chapter_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Check if this is a multi-chapter generation request and initialize state"""
        try:
            is_multi_chapter = state.get("is_multi_chapter", False)
            chapter_range = state.get("chapter_range")
            
            if is_multi_chapter and chapter_range:
                start_ch, end_ch = chapter_range
                # Initialize multi-chapter state
                return {
                    "is_multi_chapter": True,
                    "chapter_range": chapter_range,
                    "current_generation_chapter": start_ch,
                    "generated_chapters": {}
                }
            else:
                return {
                    "is_multi_chapter": False,
                    "current_generation_chapter": None,
                    "generated_chapters": {}
                }
        except Exception as e:
            logger.error(f"Multi-chapter check failed: {e}")
            return {
                "is_multi_chapter": False,
                "current_generation_chapter": None,
                "generated_chapters": {}
            }
    
    def _route_multi_chapter(self, state: FictionEditingState) -> str:
        """Route to multi-chapter loop or single chapter generation"""
        if state.get("is_multi_chapter", False):
            return "multi_chapter_loop"
        return "single_chapter"
    
    async def _prepare_chapter_context_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Prepare context for current chapter in multi-chapter generation loop"""
        try:
            current_ch = state.get("current_generation_chapter")
            chapter_range = state.get("chapter_range")
            generated_chapters = state.get("generated_chapters", {})
            manuscript = state.get("manuscript", "")
            chapter_ranges = state.get("chapter_ranges", [])
            
            if not current_ch or not chapter_range:
                return {"error": "Invalid multi-chapter state", "task_status": "error"}
            
            start_ch, end_ch = chapter_range
            
            # Get previous chapter text from generated chapters or manuscript
            prev_chapter_text = None
            if current_ch > start_ch:
                # Previous chapter was just generated
                prev_chapter_text = generated_chapters.get(current_ch - 1)
            elif current_ch == start_ch and start_ch > 1:
                # First chapter in range, but previous chapters exist in manuscript
                for r in chapter_ranges:
                    if r.chapter_number == current_ch - 1:
                        prev_chapter_text = _strip_frontmatter_block(manuscript[r.start:r.end])
                        break
            
            # Get next chapter text from manuscript (if exists)
            next_chapter_text = None
            # Find current chapter range to ensure we don't accidentally use it as "next"
            current_chapter_range = None
            for r in chapter_ranges:
                if r.chapter_number == current_ch:
                    current_chapter_range = r
                    break
            
            # Only get next chapter if it's actually different from current
            for r in chapter_ranges:
                if r.chapter_number == current_ch + 1:
                    # Defensive check: ensure next chapter is actually different from current
                    if current_chapter_range and r.start == current_chapter_range.start:
                        logger.warning(f"⚠️ Next chapter {r.chapter_number} has same start as current chapter {current_ch} - likely bug. Skipping.")
                        next_chapter_text = None
                    else:
                        next_chapter_text = _strip_frontmatter_block(manuscript[r.start:r.end])
                        logger.info(f"📖 Extracted next chapter for multi-chapter: {r.chapter_number} ({len(next_chapter_text)} chars)")
                    break
            
            # Extract outline for current chapter
            outline_body = state.get("outline_body")
            outline_current_chapter_text = None
            if outline_body and current_ch:
                chapter_pattern = rf"(?i)(?:^|\n)##?\s*(?:Chapter\s+)?{current_ch}[:\s]*(.*?)(?=\n##?\s*(?:Chapter\s+)?\d+|\Z)"
                match = re.search(chapter_pattern, outline_body, re.DOTALL)
                if match:
                    outline_current_chapter_text = match.group(1).strip()
            
            # Update state for current chapter generation
            return {
                "current_chapter_number": current_ch,
                "requested_chapter_number": current_ch,
                "current_chapter_text": "",  # Empty for new generation
                "prev_chapter_text": prev_chapter_text,
                "next_chapter_text": next_chapter_text,
                "outline_current_chapter_text": outline_current_chapter_text
            }
            
        except Exception as e:
            logger.error(f"Failed to prepare chapter context: {e}")
            return {
                "error": str(e),
                "task_status": "error"
            }
    
    async def _prepare_generation_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Prepare generation state: set system_prompt and datetime_context for generation subgraph"""
        try:
            # Build system prompt
            system_prompt = self._build_system_prompt()
            
            # Get datetime context
            datetime_context = self._get_datetime_context()
            
            return {
                "system_prompt": system_prompt,
                "datetime_context": datetime_context
            }
        except Exception as e:
            logger.error(f"Failed to prepare generation: {e}")
            return {
                "error": str(e),
                "task_status": "error"
            }
    
    async def _accumulate_chapter_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Accumulate generated chapter and update state for next iteration"""
        try:
            current_ch = state.get("current_generation_chapter")
            chapter_range = state.get("chapter_range")
            generated_chapters = state.get("generated_chapters", {})
            editor_operations = state.get("editor_operations", [])
            
            if not current_ch or not chapter_range:
                return {}
            
            # Extract generated text from operations
            generated_text = "\n\n".join([
                op.get("text", "").strip()
                for op in editor_operations
                if op.get("text", "").strip()
            ]).strip()
            
            if generated_text:
                generated_chapters[current_ch] = generated_text
                logger.info(f"Accumulated Chapter {current_ch} ({len(generated_text)} chars)")
            
            # Determine next chapter number for continuation
            start_ch, end_ch = chapter_range
            next_ch = None
            if current_ch < end_ch:
                next_ch = current_ch + 1
            
            return {
                "generated_chapters": generated_chapters,
                "current_generation_chapter": next_ch  # Update to next chapter or None if done
            }
            
        except Exception as e:
            logger.error(f"Failed to accumulate chapter: {e}")
            return {}
    
    def _route_chapter_completion(self, state: FictionEditingState) -> str:
        """Check if more chapters need to be generated"""
        current_ch = state.get("current_generation_chapter")
        chapter_range = state.get("chapter_range")
        
        if not chapter_range:
            return "complete"
        
        start_ch, end_ch = chapter_range
        
        if current_ch and current_ch < end_ch:
            # More chapters to generate - continue loop
            return "next_chapter"
        
        return "complete"
    
    def _route_after_resolve(self, state: FictionEditingState) -> str:
        """Route after resolve_operations: simple path skips, full path continues"""
        # Check if we came from simple path
        is_simple = state.get("is_simple_request", False)
        
        if is_simple:
            # Simple path: skip everything, go straight to format
            logger.info("⚡ Simple path: skipping accumulation")
            return "format_response"
        
        # Full path: check if multi-chapter
        if state.get("is_multi_chapter", False):
            return "accumulate_chapter"
        
        # Single chapter: go to format (continuity already updated in validation subgraph)
        return "format_response"
    
    def _route_after_accumulate(self, state: FictionEditingState) -> str:
        """Route after accumulate_chapter: check if more chapters needed"""
        current_ch = state.get("current_generation_chapter")
        chapter_range = state.get("chapter_range")
        
        if chapter_range:
            start_ch, end_ch = chapter_range
            # If current_ch is set and <= end_ch, we have more chapters to generate
            if current_ch is not None and current_ch <= end_ch:
                logger.info(f"More chapters to generate: current={current_ch}, end={end_ch}")
                return "next_chapter"
        
        logger.info("All chapters generated - formatting response")
        return "format_response"
    
    def _route_after_continuity_update(self, state: FictionEditingState) -> str:
        """Route after update_continuity: single goes to format, multi checks if more chapters needed"""
        if state.get("is_multi_chapter", False):
            # Check if more chapters needed
            current_ch = state.get("current_generation_chapter")
            chapter_range = state.get("chapter_range")
            
            if chapter_range:
                start_ch, end_ch = chapter_range
                if current_ch and current_ch < end_ch:
                    return "next_chapter"
            
            return "format_response"  # Multi-chapter complete
        else:
            return "format_response"  # Single chapter - go to format
    
    # _generate_edit_plan_node removed - now handled by generation_subgraph
    # Functionality moved to: orchestrator/subgraphs/fiction_generation_subgraph.py
    
    # _resolve_operations_node removed - now handled by resolution_subgraph
    # Functionality moved to: orchestrator/subgraphs/fiction_resolution_subgraph.py
    
    async def _format_response_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Format final response with editor operations"""
        try:
            logger.info("Formatting response...")
            
            # Check if this is a question response (already formatted in answer_question_node)
            # BUT: If operations were generated, include them even for questions
            existing_response = state.get("response", {})
            editor_operations = state.get("editor_operations", [])
            
            if existing_response and isinstance(existing_response, dict) and existing_response.get("response"):
                # Question response already formatted - but check if we have operations to include
                if editor_operations:
                    # Question with operations: include them in the response
                    logger.info(f"Question response with {len(editor_operations)} editor operations - including them")
                    existing_response["editor_operations"] = editor_operations
                    structured_edit = _get_structured_edit(state)
                    if structured_edit:
                        existing_response["manuscript_edit"] = {
                            "target_filename": structured_edit.target_filename,
                            "scope": structured_edit.scope,
                            "summary": structured_edit.summary,
                            "chapter_index": structured_edit.chapter_index,
                            "safety": structured_edit.safety,
                            "operations": editor_operations
                        }
                else:
                    # Pure question with no operations
                    logger.info("Returning question response (no editor operations)")
                return {
                    "response": existing_response,
                    "task_status": existing_response.get("task_status", "complete")
                }
            
            structured_edit = _get_structured_edit(state)
            # editor_operations already extracted above
            task_status = state.get("task_status", "complete")
            is_multi_chapter = state.get("is_multi_chapter", False)
            generated_chapters = state.get("generated_chapters", {})
            request_type = state.get("request_type", "")
            
            if task_status == "error":
                error_msg = state.get("error", "Unknown error")
                return {
                    "response": {
                        "response": f"Fiction editing failed: {error_msg}",
                        "task_status": "error",
                        "agent_type": "fiction_editing_agent"
                    },
                    "task_status": "error"
                }
            
            # Handle questions with no operations (analysis-only response)
            if request_type == "question" and len(editor_operations) == 0 and structured_edit:
                summary = structured_edit.summary
                if summary:
                    logger.info("Question request with no operations - using summary as response")
                    return {
                        "response": {
                            "response": summary,
                            "task_status": "complete",
                            "agent_type": "fiction_editing_agent"
                        },
                        "task_status": "complete",
                        "editor_operations": []
                }
            
            # Build response text - use summary, not full text
            if is_multi_chapter and generated_chapters:
                # Multi-chapter: brief summary only
                response_text = f"Generated {len(generated_chapters)} chapters."
                if structured_edit and structured_edit.summary:
                    response_text = f"{structured_edit.summary}\n\n{response_text}"
            else:
                # Single chapter: use summary from structured_edit, not full text
                if structured_edit and structured_edit.summary:
                    response_text = structured_edit.summary
                elif editor_operations:
                    # Fallback: brief description of operations
                    op_count = len(editor_operations)
                    response_text = f"Made {op_count} edit(s) to align with style guide and improve narrative flow."
                else:
                    response_text = "Edit plan ready."
            
            # Add clarifying questions if present
            clarifying_questions = structured_edit.clarifying_questions if structured_edit else []
            if clarifying_questions:
                questions_section = "\n\n**Questions for clarification:**\n" + "\n".join([
                    f"- {q}" for q in clarifying_questions
                ])
                response_text = response_text + questions_section
            
            # Add continuity violations if present
            continuity_violations = state.get("continuity_violations", [])
            if continuity_violations:
                violations_section = "\n\n⚠️ **CONTINUITY WARNINGS:**\n"
                for violation in continuity_violations:
                    severity = violation.get("severity", "medium").upper()
                    description = violation.get("description", "")
                    suggestion = violation.get("suggestion", "")
                    violations_section += f"- [{severity}] {description}\n"
                    if suggestion:
                        violations_section += f"  Suggestion: {suggestion}\n"
                response_text = response_text + violations_section
            
            # Add outline sync analysis if detected
            outline_needs_sync = state.get("outline_needs_sync", False)
            outline_sync_analysis = state.get("outline_sync_analysis")
            if outline_needs_sync and outline_sync_analysis:
                discrepancies = outline_sync_analysis.get("discrepancies", [])
                summary = outline_sync_analysis.get("summary", "")
                if discrepancies:
                    sync_section = "\n\n**⚠️ OUTLINE SYNC ALERT**\n"
                    sync_section += f"The outline for this chapter has changed. {summary}\n\n"
                    sync_section += "**Discrepancies detected:**\n"
                    for i, disc in enumerate(discrepancies, 1):
                        disc_type = disc.get("type", "unknown")
                        outline_exp = disc.get("outline_expectation", "")
                        manuscript_curr = disc.get("manuscript_current", "")
                        severity = disc.get("severity", "medium")
                        suggestion = disc.get("suggestion", "")
                        sync_section += f"{i}. [{severity.upper()}] {disc_type}:\n"
                        sync_section += f"   Outline expects: {outline_exp}\n"
                        sync_section += f"   Manuscript has: {manuscript_curr}\n"
                        sync_section += f"   → {suggestion}\n\n"
                    sync_section += "**Note**: These are advisory - the outline is a guide, not a strict requirement. Use your judgment about which discrepancies need addressing.\n"
                    response_text = sync_section + "\n\n" + response_text
            
            # Add consistency warnings if present
            consistency_warnings = state.get("consistency_warnings", [])
            reference_warnings = state.get("reference_warnings", [])
            
            all_warnings = consistency_warnings + reference_warnings
            if all_warnings:
                warnings_section = "\n\n**⚠️ Validation Notices:**\n" + "\n".join(all_warnings)
                response_text = response_text + warnings_section
            
            # Build response with editor operations
            response = {
                "response": response_text,
                "task_status": task_status,
                "agent_type": "fiction_editing_agent",
                "timestamp": datetime.now().isoformat()
            }
            
            # For multi-chapter, combine all operations from all chapters
            if is_multi_chapter and generated_chapters:
                # Build combined operations for all chapters
                all_operations = []
                chapter_range = state.get("chapter_range")
                if chapter_range:
                    start_ch, end_ch = chapter_range
                    for ch_num in range(start_ch, end_ch + 1):
                        ch_text = generated_chapters.get(ch_num, "")
                        if ch_text:
                            # Create operation for this chapter
                            all_operations.append({
                                "op_type": "insert_after_heading",
                                "text": ch_text,
                                "chapter_index": ch_num - 1  # 0-indexed
                            })
                
                if all_operations:
                    response["editor_operations"] = all_operations
                    filename = state.get("filename", "manuscript.md")
                    if structured_edit:
                        response["manuscript_edit"] = {
                            "target_filename": structured_edit.target_filename or filename,
                            "scope": "multi_chapter",
                            "summary": f"Generated chapters {start_ch}-{end_ch}",
                            "safety": structured_edit.safety or "medium",
                            "operations": all_operations
                        }
                    else:
                        response["manuscript_edit"] = {
                            "target_filename": filename,
                            "scope": "multi_chapter",
                            "summary": f"Generated chapters {start_ch}-{end_ch}",
                            "safety": "medium",
                        "operations": all_operations
                    }
                    logger.info(f"✅ Format response: Added {len(all_operations)} multi-chapter editor_operations to response dict")
            elif editor_operations:
                # Always include editor_operations when present
                response["editor_operations"] = editor_operations
                if structured_edit:
                    response["manuscript_edit"] = {
                        "target_filename": structured_edit.target_filename,
                        "scope": structured_edit.scope,
                        "summary": structured_edit.summary,
                        "chapter_index": structured_edit.chapter_index,
                        "safety": structured_edit.safety,
                        "operations": editor_operations
                    }
                else:
                    # Fallback if structured_edit is missing
                    logger.warning("No structured_edit but operations exist - creating minimal manuscript_edit")
                    response["manuscript_edit"] = {
                        "target_filename": state.get("filename", "manuscript.md"),
                        "scope": "unknown",
                        "summary": "Edit operations generated",
                        "chapter_index": None,
                        "safety": "medium",
                    "operations": editor_operations
                }
                logger.info(f"✅ Format response: Added {len(editor_operations)} editor_operations to response dict")
                logger.info(f"✅ Format response: First operation keys: {list(editor_operations[0].keys()) if editor_operations else 'none'}")
            else:
                logger.info(f"ℹ️ Format response: No editor_operations to include (editor_operations from state: {len(state.get('editor_operations', []))})")
                # If no operations, use LLM's summary to explain why
                if structured_edit and structured_edit.summary:
                    # LLM provided explanation for empty operations - use it
                    summary = structured_edit.summary
                    logger.info(f"ℹ️ Using LLM's explanation for empty operations: {summary[:200]}")
                    # If response_text is just default, replace with LLM's explanation
                    if response.get("response") == "Fiction editing complete" or not response.get("response"):
                        response["response"] = summary
                    else:
                        # Append LLM's explanation to existing response
                        response["response"] = f"{response.get('response', '')}\n\n{summary}"
                elif not response.get("response") or response.get("response") == "Fiction editing complete":
                    # No summary provided - generic message
                    response["response"] = "No edits were generated. The agent may need more context or clarification. Please try rephrasing your request or providing more details."
            
            # Defensive check: ensure editor_operations are always in response if they exist in state
            if not response.get("editor_operations") and state.get("editor_operations"):
                logger.warning(f"⚠️ Format response: editor_operations missing from response but present in state - adding them")
                response["editor_operations"] = state.get("editor_operations")
                if structured_edit:
                    response["manuscript_edit"] = {
                        "target_filename": structured_edit.target_filename,
                        "scope": structured_edit.scope,
                        "summary": structured_edit.summary,
                        "chapter_index": structured_edit.chapter_index,
                        "safety": structured_edit.safety,
                        "operations": state.get("editor_operations")
                    }
            
            return {
                "response": response,
                "task_status": task_status
            }
            
        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return {
                "response": self._create_error_response(str(e)),
                "task_status": "error"
            }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """Process fiction editing query using LangGraph workflow"""
        try:
            logger.info(f"Fiction editing agent processing: {query[:100]}...")
            
            # Extract user_id and shared_memory from metadata
            metadata = metadata or {}
            user_id = metadata.get("user_id", "system")
            shared_memory = metadata.get("shared_memory", {}) or {}
            
            # Prepare new messages (current query)
            new_messages = self._prepare_messages_with_query(messages, query)
            
            # Get workflow to access checkpoint
            workflow = await self._get_workflow()
            config = self._get_checkpoint_config(metadata)
            
            # Load and merge checkpointed messages to preserve conversation history
            conversation_messages = await self._load_and_merge_checkpoint_messages(
                workflow, config, new_messages
            )
            
            # Load shared_memory from checkpoint if available
            checkpoint_state = await workflow.aget_state(config)
            existing_shared_memory = {}
            if checkpoint_state and checkpoint_state.values:
                existing_shared_memory = checkpoint_state.values.get("shared_memory", {})
            
            # Merge shared_memory: start with checkpoint, then update with NEW data (so new active_editor overwrites old)
            shared_memory_merged = existing_shared_memory.copy()
            shared_memory_merged.update(shared_memory)  # New data (including updated active_editor) takes precedence
            
            # Extract active_editor from shared_memory for direct state access
            active_editor = shared_memory_merged.get("active_editor", {}) or {}
            manuscript = active_editor.get("content", "") or ""
            filename = active_editor.get("filename") or "manuscript.md"
            frontmatter = active_editor.get("frontmatter", {}) or {}
            cursor_offset = int(active_editor.get("cursor_offset", -1))
            selection_start = int(active_editor.get("selection_start", -1))
            selection_end = int(active_editor.get("selection_end", -1))
            
            # Initialize state for LangGraph workflow
            initial_state: FictionEditingState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory_merged,
                "active_editor": active_editor,
                "manuscript": manuscript,
                "filename": filename,
                "frontmatter": frontmatter,
                "cursor_offset": cursor_offset,
                "selection_start": selection_start,
                "selection_end": selection_end,
                "chapter_ranges": [],
                "active_chapter_idx": -1,
                "current_chapter_text": "",
                "current_chapter_number": None,
                "prev_chapter_text": None,
                "next_chapter_text": None,
                "outline_body": None,
                "rules_body": None,
                "style_body": None,
                "characters_bodies": [],
                "outline_current_chapter_text": None,
                "current_request": "",
                "requested_chapter_number": None,
                "system_prompt": "",
                "llm_response": "",
                "structured_edit": None,
                "editor_operations": [],
                "response": {},
                "task_status": "",
                "error": "",
                # New fields for mode tracking and validation
                "generation_mode": "",
                "creative_freedom_requested": False,
                "mode_guidance": "",
                "reference_quality": {},
                "reference_warnings": [],
                "reference_guidance": "",
                "consistency_warnings": [],
                # Multi-chapter generation fields
                "is_multi_chapter": False,
                "chapter_range": None,
                "current_generation_chapter": None,
                "generated_chapters": {},
                # Outline sync detection
                "outline_sync_analysis": None,
                "outline_needs_sync": False
            }
            
            # Run LangGraph workflow with checkpointing (workflow and config already created above)
            result_state = await workflow.ainvoke(initial_state, config=config)
            
            # Extract final response
            response = result_state.get("response", {})
            task_status = result_state.get("task_status", "complete")
            
            # Debug: log response structure
            logger.info(f"🔍 Process: response type={type(response)}, keys={list(response.keys()) if isinstance(response, dict) else 'not a dict'}")
            logger.info(f"🔍 Process: editor_operations in state={bool(result_state.get('editor_operations'))}, count={len(result_state.get('editor_operations', []))}")
            
            if task_status == "error":
                error_msg = result_state.get("error", "Unknown error")
                logger.error(f"Fiction editing agent failed: {error_msg}")
                return {
                    "response": f"Fiction editing failed: {error_msg}",
                    "task_status": "error",
                    "agent_results": {}
                }
            
            # Get editor_operations from response dict OR directly from state (defensive)
            editor_ops_from_response = response.get("editor_operations", []) if isinstance(response, dict) else []
            editor_ops_from_state = result_state.get("editor_operations", [])
            editor_operations = editor_ops_from_response or editor_ops_from_state
            
            if editor_ops_from_state and not editor_ops_from_response:
                logger.warning(f"⚠️ Process: editor_operations missing from response dict but present in state - using state")
            
            # Build result dict matching character_development_agent pattern
            result = {
                "response": response.get("response", "Fiction editing complete") if isinstance(response, dict) else str(response),
                "task_status": task_status,
                "agent_results": {
                    "editor_operations": editor_operations,
                    "manuscript_edit": response.get("manuscript_edit") if isinstance(response, dict) else None
                }
            }
            
            # Add editor operations at top level for compatibility
            if editor_operations:
                result["editor_operations"] = editor_operations
                logger.info(f"✅ Fiction editing agent: Added {len(editor_operations)} editor_operations to result")
                if editor_operations:
                    logger.info(f"✅ Fiction editing agent: First operation type={editor_operations[0].get('op_type')}, has_text={bool(editor_operations[0].get('text'))}")
            else:
                logger.warning(f"⚠️ Fiction editing agent: No editor_operations found (response keys: {list(response.keys()) if isinstance(response, dict) else 'not a dict'}, state has ops: {bool(editor_ops_from_state)})")
            
            if isinstance(response, dict) and response.get("manuscript_edit"):
                result["manuscript_edit"] = response["manuscript_edit"]
            
            logger.info(f"Fiction editing agent completed: {task_status}, result has editor_operations: {bool(result.get('editor_operations'))}")
            return result
            
        except Exception as e:
            logger.error(f"Fiction editing agent failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "response": f"Fiction editing failed: {str(e)}",
                "task_status": "error",
                "agent_results": {}
            }


def get_fiction_editing_agent() -> FictionEditingAgent:
    """Get singleton fiction editing agent instance"""
    global _fiction_editing_agent
    if _fiction_editing_agent is None:
        _fiction_editing_agent = FictionEditingAgent()
    return _fiction_editing_agent


_fiction_editing_agent: Optional[FictionEditingAgent] = None

