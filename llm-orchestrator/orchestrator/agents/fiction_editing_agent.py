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
from orchestrator.utils.editor_operation_resolver import resolve_editor_operation
# build_context_preparation_subgraph and build_resolution_subgraph removed; fiction flow uses flat fiction_editing_subgraph
from orchestrator.subgraphs import (
    build_validation_subgraph,
    build_generation_subgraph,
    build_proofreading_subgraph
)
build_context_preparation_subgraph = None  # deprecated; fiction_editing_subgraph uses flat nodes
build_resolution_subgraph = None  # deprecated; fiction_editing_subgraph uses flat nodes
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
    ensure_chapter_heading as _ensure_chapter_heading,
    extract_chapter_outline,
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
    series_body: Optional[str]
    outline_current_chapter_text: Optional[str]
    outline_prev_chapter_text: Optional[str]
    has_references: bool  # CRITICAL: Flag from subgraph indicating if references were loaded
    current_request: str
    requested_chapter_number: Optional[int]
    system_prompt: str
    datetime_context: str  # ✅ PRESERVE datetime_context for generation subgraph
    llm_response: str
    structured_edit: Optional[Dict[str, Any]]
    editor_operations: List[Dict[str, Any]]
    failed_operations: List[Dict[str, Any]]
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
    # Outline sync detection
    outline_sync_analysis: Optional[Dict[str, Any]]      # Analysis of outline vs manuscript discrepancies
    outline_needs_sync: bool  # Whether manuscript needs updates to match outline
    # Question answering
    request_type: str  # "question" | "edit_request" | "hybrid" | "unknown"
    analysis_scope: str  # "chapter" | "manuscript" - scope of analysis for questions
    # Explicit chapter detection from user query
    explicit_primary_chapter: Optional[int]  # Primary chapter to edit (from query)


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
        logger.info("Fiction Editing Agent ready!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for fiction editing agent. Deprecated: fiction flow uses fiction_editing_subgraph."""
        if build_context_preparation_subgraph is None or build_resolution_subgraph is None:
            raise NotImplementedError(
                "Fiction editing agent full workflow is deprecated; use fiction_editing_subgraph via WritingAssistantAgent"
            )
        workflow = StateGraph(FictionEditingState)
        
        # Build subgraphs
        context_subgraph = build_context_preparation_subgraph(checkpointer)
        validation_subgraph = build_validation_subgraph(
            checkpointer,
            llm_factory=self._get_llm,
            get_datetime_context=self._get_datetime_context
        )
        generation_subgraph = build_generation_subgraph(
            checkpointer,
            llm_factory=self._get_llm,
            get_datetime_context=self._get_datetime_context
        )
        resolution_subgraph = build_resolution_subgraph(checkpointer)
        proofreading_subgraph = build_proofreading_subgraph(
            checkpointer,
            llm_factory=self._get_llm,
            get_datetime_context=self._get_datetime_context
        )
        
        # Phase 1: Context preparation (now a subgraph)
        workflow.add_node("context_preparation", context_subgraph)
        workflow.add_node("validate_fiction_type", self._validate_fiction_type_node)
        
        # Phase 2: Pre-generation assessment
        workflow.add_node("detect_mode", self._detect_mode_and_intent_node)
        workflow.add_node("detect_request_type", self._detect_request_type_node)
        
        # Phase 3: Generation preparation
        workflow.add_node("prepare_generation", self._prepare_generation_node)
        
        # Phase 4: Generation (now a subgraph)
        workflow.add_node("generate_edit_plan", generation_subgraph)
        workflow.add_node("generate_simple_edit", self._generate_simple_edit_node)
        
        # Phase 4.5: Proofreading (reusable subgraph)
        workflow.add_node("proofreading", proofreading_subgraph)
        
        # Phase 5: Post-generation validation (now a subgraph)
        workflow.add_node("validation", validation_subgraph)
        
        # Phase 6: Resolution (now a subgraph) and response
        workflow.add_node("resolve_operations", resolution_subgraph)
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
        
        # Route after detect_mode: check for proofreading intent or continue to generation
        workflow.add_conditional_edges(
            "detect_mode",
            self._route_after_context,
            {
                "proofreading": "proofreading",  # Proofreading intent -> proofreading subgraph
                "simple_path": "generate_simple_edit",  # No references -> fast path
                "full_path": "detect_request_type"       # Has references -> full path
            }
        )
        
        # Proofreading subgraph flows to resolution (skip generation/validation)
        workflow.add_edge("proofreading", "resolve_operations")
        
        # Route to generation preparation (always single chapter)
        workflow.add_edge("detect_request_type", "prepare_generation")
        
        # After prepare_generation, go to generation subgraph
        workflow.add_edge("prepare_generation", "generate_edit_plan")
        
        # Generation pipeline flow
        workflow.add_edge("generate_edit_plan", "validation")
        workflow.add_edge("validation", "resolve_operations")
        
        # Simple path: skip validation, go straight to resolution
        workflow.add_edge("generate_simple_edit", "resolve_operations")
        
        # Route after resolve_operations: always to format_response
        workflow.add_edge("resolve_operations", "format_response")
        
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for fiction editing"""
        prompt = (
            "CODE VERSION: 2026-02-10-00:00 UTC - IF YOU SEE THIS, CODE IS FRESH!\n\n"
            "You are a MASTER NOVELIST. Your job is to write fiction — not to expand outlines.\n\n"
            "=== YOUR CREATIVE PROCESS (READ THIS FIRST) ===\n\n"
            "**1. ABSORB THE STYLE GUIDE** — This is your primary creative lens.\n"
            "   The Style Guide defines HOW you write: voice, tone, sentence length, pacing,\n"
            "   words and phrasing to use, words and phrasing to AVOID, level of detail,\n"
            "   dialogue style, POV, tense, and narrative technique.\n"
            "   If a writing sample is provided, it is your north star — study its rhythms,\n"
            "   its choices, what it lingers on, what it skips. Your prose must read as though\n"
            "   it came from the same author as that sample.\n\n"
            "**2. UNDERSTAND THE STORY** — Read the Outline to know what happens.\n"
            "   The Outline tells you the events, plot beats, and story goals for the chapter.\n"
            "   Internalize these as background knowledge — the way an author knows their plot\n"
            "   before sitting down to write. You are NOT converting an outline into prose.\n"
            "   You are writing a chapter of a novel, and you happen to know what needs to happen.\n\n"
            "**3. WRITE IN SCENES** — Think like a novelist, not a summarizer.\n"
            "   Combine related beats into cohesive scenes with natural flow.\n"
            "   Not every beat needs its own scene or paragraph. Some beats are a single line\n"
            "   of dialogue; others deserve a full page. Let the Style Guide's pacing sensibility\n"
            "   determine how much space each moment gets.\n"
            "   Build tension, create atmosphere, develop character through action and dialogue.\n"
            "   The reader should never feel they are reading an expanded outline.\n\n"
            "=== REFERENCE HIERARCHY ===\n\n"
            "References are listed in order of creative authority:\n\n"
            "1. **STYLE GUIDE** (highest authority — HOW to write)\n"
            "   - This is your voice. Every word choice, sentence rhythm, and descriptive\n"
            "     approach must conform to the Style Guide.\n"
            "   - If the Style Guide says 'avoid adverbs,' you avoid adverbs.\n"
            "   - If it provides a writing sample, match its cadence, density, and feel.\n"
            "   - If the Style Guide conflicts with your instincts, the Style Guide wins.\n\n"
            "2. **OUTLINE** (story content — WHAT happens)\n"
            "   - Provides the events, character actions, and plot progression for the chapter.\n"
            "   - Treat beats as story ingredients, not a recipe to follow in order.\n"
            "   - You may reorder, combine, interleave, or restructure beats into scenes\n"
            "     as long as the chapter achieves the same narrative goals.\n"
            "   - NEVER copy or paraphrase outline language into your prose.\n"
            "   - CRITICAL: Outline text does NOT exist in manuscript — never use for anchors/original_text!\n\n"
            "3. **CHARACTER PROFILES** (character authenticity)\n"
            "   - Each profile belongs to a DIFFERENT character — do not confuse them.\n"
            "   - Match the correct character's dialogue patterns, traits, and behaviors.\n"
            "   - Draw from profiles for authentic details, but vary phrasing each time.\n\n"
            "4. **UNIVERSE RULES** (world-building consistency)\n"
            "   - All world-building elements must align with established constraints.\n"
            "   - Plot events and character actions must not violate universe rules.\n\n"
            "5. **SERIES TIMELINE** (if provided — cross-book continuity)\n"
            "   - Ensure timeline consistency with established series events.\n\n"
            "=== WRITING QUALITY STANDARDS ===\n\n"
            "**Scene Construction:**\n"
            "- Build scenes with narrative purpose — each scene advances plot, reveals character, or creates atmosphere\n"
            "- Open scenes in medias res or with sensory grounding, not with summary\n"
            "- Let dialogue carry exposition naturally; avoid info-dumps\n"
            "- End scenes with forward momentum — a question, a tension, a shift\n\n"
            "**Variation and Freshness:**\n"
            "- Vary character descriptions: same traits, different angles and phrasing each time\n"
            "- Vary location descriptions based on character mood, time of day, narrative context\n"
            "- Mix dialogue tags, action beats, and internal thoughts — avoid repetitive patterns\n"
            "- Same events can be written with different emphasis depending on scene context\n\n"
            "**Consistency Requirements (DO NOT BREAK):**\n"
            "- Character profiles: Core traits, appearance, personality remain consistent\n"
            "- Universe rules: Physical laws, magic systems, technology remain consistent\n"
            "- Style Guide: Voice, POV, tense, and pacing remain consistent throughout\n"
            "- Story continuity: Established facts, character knowledge, plot threads remain consistent\n\n"
            "**When in doubt:** The Style Guide is always right. Write the way it tells you to write.\n\n" +
            "="*80 + "\n" +
            "STRUCTURED OUTPUT REQUIRED\n" +
            "="*80 + "\n\n" +
            "You MUST return ONLY raw JSON (no prose, no markdown fences, no explanatory text).\n\n"
            "REQUIRED STRUCTURE (valid JSON example):\n\n"
            "{\n"
            '  "target_filename": "manuscript.md",\n'
            '  "scope": "chapter",\n'
            '  "summary": "Generated Chapter 1 following outline and style guide",\n'
            '  "safety": "medium",\n'
            '  "chapter_index": 0,\n'
            '  "operations": [\n'
            "    {\n"
            '      "op_type": "insert_after_heading",\n'
            '      "start": 62,\n'
            '      "end": 62,\n'
            '      "anchor_text": "---",\n'
            '      "text": "## Chapter 1: Title\\n\\nAmanda stepped into the office. The morning light..."\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "**CRITICAL FIELD REQUIREMENTS:**\n"
            '- "scope" MUST be EXACTLY one of: "paragraph", "chapter", or "multi_chapter" (no other values allowed!)\n'
            '- "safety" MUST be EXACTLY one of: "low", "medium", or "high" (no other values allowed!)\n'
            '- "operations" is an array of operation objects (can be empty [] for question-only requests)\n'
            '- "summary" is a string describing what was done or answering the question\n\n'
            "FIELD DEFINITIONS:\n"
            "- target_filename: Name of manuscript file (string, REQUIRED)\n"
            "- scope: Edit scope - \"paragraph\", \"chapter\", or \"multi_chapter\" (REQUIRED)\n"
            "- summary: Brief description of changes OR conversational answer to user's question (string, REQUIRED)\n"
            "  * For edit requests: Brief description of what was changed\n"
            "  * For questions: Natural, conversational answer with analysis and feedback\n"
            "- safety: Risk level - \"low\", \"medium\", or \"high\" (REQUIRED, default \"medium\")\n"
            "- chapter_index: Zero-based chapter index (integer, optional, null for multi-chapter)\n"
            "- operations: Array of edit operations (REQUIRED, can be empty for questions/analysis)\n\n"
            "=== ANSWERING QUESTIONS CONVERSATIONALLY ===\n\n"
            "**You are both an editor AND a conversational assistant.** When users ask questions, answer them naturally and helpfully.\n\n"
            "**QUESTION HANDLING:**\n"
            "- **Pure questions** (e.g., 'How does our chapter look so far?', 'What's the pacing like?', 'Does this follow the style guide?'):\n"
            "  * Answer the question conversationally in the 'summary' field\n"
            "  * Provide specific observations, analysis, and feedback based on the manuscript and references\n"
            "  * Return empty operations array: []\n"
            "  * Be helpful, specific, and reference actual content when relevant\n"
            "  * Example: 'How does our chapter look so far?' → Summary: 'The chapter has strong pacing with good tension building. The dialogue feels natural, and the character development is progressing well. The prose matches the style guide's voice. One area to consider: the transition between scenes could be smoother.' Operations: []\n\n"
            "- **Questions with conditional edits** (e.g., 'Are we using enough description? Revise if needed', 'Does this follow the outline? Fix if not'):\n"
            "  * First, answer the question in the 'summary' field\n"
            "  * Then, if your analysis indicates edits are needed, provide operations\n"
            "  * If no edits needed, return empty operations array but still answer the question\n"
            "  * Example: 'Are we using enough description?' → Summary: 'The description level is moderate. Adding more sensory details to enhance immersion.' Operations: [replace_range with enhanced description]\n\n"
            "- **Evaluation questions** (e.g., 'How's the character development?', 'What's the tone like?'):\n"
            "  * Provide thoughtful analysis based on the manuscript, style guide, and character profiles\n"
            "  * Reference specific examples from the text when helpful\n"
            "  * Return empty operations unless the user explicitly asks for revisions\n"
            "  * Be conversational and helpful - you're providing feedback, not just editing\n\n"
            "**KEY PRINCIPLE**: When answering questions, think like a helpful editor giving feedback, not just a code generator.\n"
            "The 'summary' field is your opportunity to have a conversation with the user about their work.\n\n"
            "OPERATION FIELDS (inside operations array):\n"
            "- op_type: \"replace_range\", \"delete_range\", \"insert_after_heading\", or \"insert_after\" (REQUIRED)\n"
            "- start: Approximate character offset (integer, REQUIRED, anchors take precedence)\n"
            "- end: Approximate character offset (integer, REQUIRED, anchors take precedence)\n"
            "- text: New prose for replace/insert (string, REQUIRED)\n"
            "- original_text: EXACT text from manuscript for replace/delete (string, REQUIRED for replace_range/delete_range)\n"
            "- anchor_text: EXACT text to insert after (string, REQUIRED for insert_after_heading)\n"
            "- left_context: Text before target (string, optional)\n"
            "- right_context: Text after target (string, optional)\n"
            "- occurrence_index: Which occurrence if text appears multiple times (integer, optional, default 0)\n\n"
            "OUTPUT RULES:\n"
            "- Output MUST be a single JSON object only.\n"
            "- Do NOT include triple backticks or language tags.\n"
            "- Do NOT include explanatory text before or after the JSON.\n\n"
            "=== THREE FUNDAMENTAL OPERATIONS ===\n\n"
            "**1. replace_range**: Replace existing text with new text\n"
            "   USE WHEN: User wants to revise, improve, change, modify, or rewrite existing prose\n"
            "   ANCHORING: Provide 'original_text' with EXACT, VERBATIM text from manuscript\n\n"
            "   **CRITICAL: PREFER GRANULAR, WORD-LEVEL EDITS**\n\n"
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
            "      - DO NOT default to 20-40 words - use the MINIMUM needed for reliable matching\n\n"
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
            "   THIS IS FOR NEW TEXT THAT DOESN'T EXIST YET - use this when adding content!\n"
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
            "CRITICAL: NEVER include the next chapter's heading in your operation!\n\n"
            "**CROSS-CHAPTER EDITS FORBIDDEN:**\n"
            "original_text and anchor_text must NEVER span from one chapter into another. Each operation targets ONE chapter only.\n"
            "If editing near the end of a chapter, do NOT include the next chapter's heading or any text from the next chapter.\n"
            "If editing near the start of a chapter, do NOT include the previous chapter's last line. Stay strictly within the chapter you are editing.\n\n"
            "**CRITICAL TEXT PRECISION REQUIREMENTS:**\n"
            "For 'original_text' and 'anchor_text' fields:\n"
            "- Must be EXACT, COMPLETE, and VERBATIM from the current manuscript text (not from outline, not from any reference documents)\n"
            "- NEVER EVER use text from the outline for anchors - outline text does NOT exist in the manuscript!\n"
            "- You must copy text from the '=== MANUSCRIPT CONTEXT ===' sections (current chapter, previous chapter, next chapter)\n"
            "- Include ALL whitespace, line breaks, and formatting exactly as written\n"
            "- Include complete sentences or natural text boundaries (periods, paragraph breaks)\n"
            "- NEVER paraphrase, summarize, or reformat the text\n"
            "- Minimum 10-20 words for unique identification (see granular edit principles above)\n"
            "- NEVER include chapter headers (##) in original_text for replace_range!\n\n"
            "=== CREATIVE ADDITIONS POLICY ===\n\n"
            "**You have creative freedom to enhance the story with additions:**\n\n"
            "When the user requests additions, enhancements, or expansions, you may add story elements\n"
            "that are NOT explicitly in the outline, as long as they maintain consistency.\n\n"
            "**MANDATORY CONSISTENCY CHECKS for all additions:**\n"
            "Before adding ANY new story element, verify:\n"
            "1. Style Guide compliance - matches established voice/tone/pacing\n"
            "2. Universe Rules compliance - no violations of established physics/magic/tech\n"
            "3. Character consistency - behavior matches character profiles\n"
            "4. Manuscript continuity - no contradictions with established facts\n"
            "5. Timeline coherence - events fit logically in story sequence\n\n"
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
            "=== CONSISTENCY VALIDATION ===\n\n"
            "For EVERY operation, validate against ALL references (see REFERENCE USAGE above):\n\n"
            "**Before adding or changing content, verify:**\n"
            "- Style Guide compliance: narrative voice, POV, tense, pacing, dialogue style\n"
            "- Universe Rules compliance: no violations of physics/magic/tech constraints\n"
            "- Character consistency: behavior matches character profiles\n"
            "- Manuscript continuity: no contradictions with established facts from earlier chapters\n"
            "- Timeline coherence: events fit logically in story sequence\n\n"
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
            "- Vary descriptions to avoid repetition (see CREATIVE VARIATION section above)\n\n"
            "**Organic Pacing vs Mechanical Beat-Following:**\n"
            "- Write scenes that flow naturally with appropriate pacing for the moment\n"
            "- Don't rush through beats to 'cover' all outline points - let scenes breathe\n"
            "- Build tension, emotion, and character development organically within scenes\n"
            "- Outline beats are story goals to achieve, not items to check off sequentially\n"
            "- A single scene can achieve multiple outline beats naturally if the story flows that way\n"
            "- Conversely, a single outline beat might require multiple scenes if the story demands it\n\n"
            "=== CONTENT GENERATION RULES ===\n\n"
            "1. operations[].text MUST contain final prose (no placeholders or notes)\n"
            "2. For chapter generation: begin with '## Chapter N'\n"
            "3. Do NOT impose a word count limit. Write as much prose as needed to produce a complete, compelling scene.\n"
            "4. Treat outline beats as plot objectives to achieve, NOT text to expand or reuse (see REFERENCE USAGE above)\n"
            "5. NO YAML frontmatter in operations[].text\n"
            "6. Complete sentences with proper grammar\n"
            "7. Write complete narrative with dialogue, action, description, character voice, and emotional depth\n"
        )
        return prompt
    
    async def _validate_fiction_type_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Parse user query for explicit chapter mentions"""
        try:
            logger.info("Detecting chapter mentions in user query...")
            
            current_request = state.get("current_request", "")
            if not current_request:
                logger.info("No current request - skipping chapter detection")
                return {
                    "explicit_primary_chapter": None
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
            
            if unique_mentions:
                # First mention is primary (what to edit)
                primary_chapter = unique_mentions[0]["chapter"]
            else:
                logger.info("   No explicit chapter mentions found in query")
            
            return {
                "explicit_primary_chapter": primary_chapter
            }
            
        except Exception as e:
            logger.error(f"Failed to detect chapter mentions: {e}")
            return {
                "explicit_primary_chapter": None
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
                    },
                    # ✅ CRITICAL: Preserve even on error!
                    "current_chapter_text": state.get("current_chapter_text", ""),
                    "current_chapter_number": state.get("current_chapter_number"),
                    "manuscript": state.get("manuscript", ""),
                    "filename": state.get("filename", ""),
                    "chapter_ranges": state.get("chapter_ranges", []),
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                }
            
            return {
                # ✅ CRITICAL: Preserve all state even on success!
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "prev_chapter_text": state.get("prev_chapter_text"),
                "prev_chapter_number": state.get("prev_chapter_number"),
                "next_chapter_text": state.get("next_chapter_text"),
                "next_chapter_number": state.get("next_chapter_number"),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "cursor_offset": state.get("cursor_offset", -1),
                "outline_body": state.get("outline_body"),
                "rules_body": state.get("rules_body"),
                "style_body": state.get("style_body"),
                "characters_bodies": state.get("characters_bodies", []),
                "series_body": state.get("series_body"),
                "outline_current_chapter_text": state.get("outline_current_chapter_text"),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
            
        except Exception as e:
            logger.error(f"Failed to validate fiction type: {e}")
            return {
                "error": str(e),
                "task_status": "error",
                # ✅ CRITICAL: Preserve even on error!
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "chapter_ranges": state.get("chapter_ranges", []),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
    
    def _route_after_validate_type(self, state: FictionEditingState) -> str:
        """Route after fiction type validation"""
        if state.get("task_status") == "error":
            return "error"
        return "continue"
    
    def _route_after_context(self, state: FictionEditingState) -> str:
        """Route after context preparation: check for proofreading intent or if references exist"""
        # Check for errors first
        if state.get("task_status") == "error":
            return "format_response"  # Skip to format to return error
        
        # Check for proofreading intent
        current_request = state.get("current_request", "").lower()
        proofreading_keywords = [
            "proofread", "check grammar", "fix typos", "style corrections",
            "grammar check", "spell check", "proofreading", "grammar", "typos",
            "check for filter words", "filter words"
        ]
        is_proofreading = any(kw in current_request for kw in proofreading_keywords)
        
        if is_proofreading:
            logger.info("Detected proofreading intent - routing to proofreading subgraph")
            return "proofreading"
        
        # Not proofreading - continue with normal routing
        has_references = state.get("has_references", False)
        
        # If no references, use simple fast path
        if not has_references:
            state["is_simple_request"] = True
            return "simple_path"
        
        # References exist - use full path
        return "full_path"
    
    async def _check_simple_request_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Check if request is simple enough to skip full workflow (no references needed)"""
        try:
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
            return "simple_path"
        else:
            return "full_path"
    
    async def _generate_simple_edit_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Generate edit using only manuscript context (fast path for simple requests)"""
        try:
            # Mark state as simple request for downstream nodes
            state["is_simple_request"] = True
            
            manuscript = state.get("manuscript", "")
            filename = state.get("filename", "manuscript.md")
            
            current_request = state.get("current_request", "")
            
            if not current_request:
                logger.error("current_request is EMPTY in _generate_simple_edit_node - this should have been set by context subgraph!")
                logger.error("This indicates a state flow issue between context subgraph and simple edit node")
                return {
                    "error": "No user request found - state flow issue",
                    "task_status": "error"
                }
            
            current_chapter_text = state.get("current_chapter_text", "")
            cursor_offset = state.get("cursor_offset", -1)
            selection_start = state.get("selection_start", -1)
            selection_end = state.get("selection_end", -1)
            
            # If cursor is -1, treat it as end of document
            if cursor_offset == -1:
                cursor_offset = len(manuscript)
            
            # Use chapter context if available, otherwise fall back to full manuscript
            prev_chapter_text = state.get("prev_chapter_text") or ""
            next_chapter_text = state.get("next_chapter_text") or ""
            
            if current_chapter_text:
                # Use full chapter context (with prev/next chapters if available)
                context_parts = []
                if prev_chapter_text:
                    context_parts.append(f"[PREVIOUS CHAPTER]\n{prev_chapter_text}\n")
                context_parts.append(f"[CURRENT CHAPTER]\n{current_chapter_text}")
                if next_chapter_text:
                    context_parts.append(f"\n[NEXT CHAPTER]\n{next_chapter_text}")
                context_text = "\n".join(context_parts)
            else:
                # No chapters found - use entire manuscript for consistency
                # Without chapter structure, LLM needs full context to maintain character names, plot points, etc.
                context_text = manuscript
            
            if len(context_text) < 100:
                logger.warning(f"⚠️ Very short context ({len(context_text)} chars) - may not be enough for continuation")
            
            # Build simple prompt focused on manuscript context only
            system_prompt = (
                "You are a fiction editor. Generate precise, granular edits based ONLY on the manuscript context provided.\n\n"
                "**CRITICAL EDITING RULES**:\n"
                "1. Use GRANULAR edits - prefer word-level or sentence-level changes\n"
                "2. Keep original_text small (10-20 words) for precise matching\n"
                "3. Break large edits into multiple operations if needed\n"
                "4. For continuing/adding text: use 'insert_after' with anchor_text = last few words\n"
                "5. For replacements: use 'replace_range' with original_text\n"
                "6. For deletions: use 'delete_range' with original_text\n"
                "7. Match the established voice and style from the manuscript\n"
                "8. Complete sentences with proper grammar\n"
                "9. **CROSS-CHAPTER EDITS FORBIDDEN**: original_text and anchor_text must NEVER span from one chapter into another. Stay within a single chapter.\n"
                "10. **FOR CONTINUATIONS**: Generate COMPLETE continuations that cover ALL material provided by the user\n"
                "   - If the user provides multiple points/events/material to cover, address ALL of them\n"
                "   - Use multiple operations if needed to ensure comprehensive coverage\n"
                "   - Do NOT truncate or stop early - ensure all requested material is generated\n"
            )
            
            user_prompt = f"""**USER REQUEST**: {current_request}

**MANUSCRIPT CONTEXT** (around cursor position {cursor_offset}):
{context_text}

**SELECTION** (if any):
{f"Selected text: {manuscript[selection_start:selection_end]}" if selection_start >= 0 and selection_end > selection_start else "No selection"}

**INSTRUCTIONS**:
- Generate edits based ONLY on the manuscript context above
- Use the text around the cursor/selection to find anchor points
- **FOR CONTINUATION REQUESTS** (e.g., "continue the paragraph", "finish the sentence", "write more", "continue the story"):
  - **PREFER** to generate at least ONE operation with new text
  - Use "insert_after" operation type with anchor_text = last few words before cursor
  - Continue naturally from where the text ends, matching the style and voice
  - **CRITICAL FOR COMPREHENSIVE CONTINUATIONS**:
    - If the user provides material/points/events to cover, you MUST generate text that addresses ALL of it
    - Generate substantial continuation that fully covers the requested material (multiple paragraphs if needed)
    - Do NOT stop after just a few sentences if more material needs to be covered
    - If the continuation is very long, you may break it into multiple operations, but ensure ALL material is covered
    - The continuation should be complete and comprehensive, not truncated
  - Generate substantial continuation (at least 1-2 sentences minimum, but more if material is provided)
  - **IF YOU CANNOT CONTINUE** (e.g., unclear context, missing information, concerns about style/consistency):
    - Return empty operations array
    - **EXPLAIN CLEARLY in the summary** why you cannot proceed and what information/clarification you need
- **FOR REVISION REQUESTS**: Make precise, targeted changes
- Keep edits granular and focused, but ensure completeness for continuation requests
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

**CRITICAL**: "scope" MUST be EXACTLY one of these three strings: "paragraph", "chapter", or "multi_chapter" (no other values allowed!)


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
- **FOR CONTINUATION REQUESTS**: Ensure ALL material provided by the user is covered in your generated text
  - If multiple points/events are mentioned, address ALL of them
  - Use multiple operations if needed to ensure comprehensive coverage
  - Do NOT stop after a few sentences if more material needs to be covered
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
                    # This is valid - LLM may have concerns or need clarification
                    pass
                
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
            
            # Extract requested chapter number from user request (single chapter only)
            requested_chapter_number = _extract_chapter_number_from_request(current_request)
            
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
                # ✅ CRITICAL: Preserve manuscript context!
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "prev_chapter_text": state.get("prev_chapter_text"),
                "prev_chapter_number": state.get("prev_chapter_number"),
                "next_chapter_text": state.get("next_chapter_text"),
                "next_chapter_number": state.get("next_chapter_number"),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "cursor_offset": state.get("cursor_offset", -1),
                # ✅ CRITICAL: Preserve reference context!
                "outline_body": state.get("outline_body"),
                "rules_body": state.get("rules_body"),
                "style_body": state.get("style_body"),
                "characters_bodies": state.get("characters_bodies", []),
                "series_body": state.get("series_body"),
                "outline_current_chapter_text": state.get("outline_current_chapter_text"),
                # ✅ CRITICAL: Preserve critical 5 keys!
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
            
        except Exception as e:
            logger.error(f"Mode detection failed: {e}")
            return {
                "generation_mode": "editing",
                "creative_freedom_requested": False,
                "mode_guidance": "",
                "requested_chapter_number": None,
                # ✅ CRITICAL: Preserve even on error!
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "chapter_ranges": state.get("chapter_ranges", []),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
    
    async def _detect_request_type_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Detect if user request is a question or an edit request - trust LLM to figure it out"""
        try:
            logger.info("Detecting request type (question vs edit request)...")
            
            current_request = state.get("current_request", "")
            if not current_request:
                logger.warning("No current request found - defaulting to edit_request")
                return {
                    "request_type": "edit_request",
                    "analysis_scope": "chapter",
                    # ✅ CRITICAL: Preserve metadata even in early return!
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                }
            
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
- EDIT REQUESTS (user wants changes applied to the manuscript): Classify as "edit_request" when the user wants you to *do* something to the text, not just answer. This includes:
  - Explicit action: "Revise Chapter 2", "Edit for repetitiveness", "Fix the dialogue", "Add a scene", "Tighten this section", "Remove redundancy"
  - **Question form but action intent**: If the request is phrased as a question but contains action-oriented words, treat as edit_request so the agent will produce editor operations. Examples: "Is there any revision to this chapter for repetitiveness?" (has "revision" → user wants revisions applied), "Can you fix the discrepancies?", "Should we edit out the redundancy?" (has "edit" → user wants edits)
  - Action words/phrases: revise, revision, edit, fix, improve, tighten, remove redundancy, cut, change, replace, add, rewrite, rework, trim, simplify, rephrase
  - **Result**: Agent will analyze AND apply suggested edits to the document.

- QUESTIONS (informational only, no edits applied): Classify as "question" when the user is asking for information or analysis only, with no action words implying they want changes made. Answer will be conversational; no editor operations.
  - Pure informational: "Is there any discrepancies?" (no action word — user wants to know, not necessarily to fix), "What style is this?", "How old is Tom here?", "Does this follow our style guide?"
  - No revise/edit/fix/improve etc.: If the user is only asking "is there X?" or "what/where/how" without implying "then fix it" or "revise for it", use "question"
  - **Result**: Agent will answer in natural language only; no edits to the document.

**DISTINCTION**:
- "Is there any revision to this chapter for repetitiveness?" → edit_request (contains "revision"; user wants revisions suggested/applied)
- "Is there any discrepancies?" → question (informational; no action word)
- "Revise to fix any discrepancies" → edit_request (explicit action)

**ANALYSIS SCOPE** (only when request_type is "question"):
- "manuscript": Whole-story / whole-book analysis. The user wants to analyze the entire manuscript, not just the current chapter. Indicators: "the story", "the book", "the manuscript", "overall", "across chapters", "character arcs", "narrative arc", "thematic consistency", "pacing across", "story structure", no specific chapter mentioned. Examples: "Analyze the story", "How's the pacing across the book?", "Review the manuscript"
- "chapter": Chapter-specific analysis (default). The user is asking about the current chapter or a specific chapter. Indicators: "this chapter", "chapter N", "this scene", "here", cursor-specific, or any chapter-scoped question. Examples: "How's this chapter?", "Review chapter 7", "What style is used here?"

**OUTPUT**: Return ONLY valid JSON:
{{
  "request_type": "question" | "edit_request",
  "analysis_scope": "chapter" | "manuscript",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why this classification"
}}

**CRITICAL**:
- If the request contains action words (revise, edit, fix, improve, tighten, revision, redundancy, etc.) — even in question form — → "edit_request" (user wants edits; agent will produce editor operations)
- If the request is purely informational (no action words) → "question" (agent will answer only; no editor operations)
- For "question", set analysis_scope to "manuscript" when the user asks about the whole story/book; otherwise "chapter"
"""
            
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
                analysis_scope = (result.get("analysis_scope") or "chapter").strip().lower()
                if analysis_scope not in ("chapter", "manuscript"):
                    analysis_scope = "chapter"
                if request_type != "question":
                    analysis_scope = "chapter"
                # Preserve early whole-manuscript detection from context subgraph
                if state.get("analysis_scope") == "manuscript":
                    analysis_scope = "manuscript"
                confidence = result.get("confidence", 0.5)
                reasoning = result.get("reasoning", "")
                logger.info(
                    "Request type detection: request_type=%s, analysis_scope=%s, confidence=%.2f",
                    request_type,
                    analysis_scope,
                    confidence,
                )
                # Default to edit_request if confidence is low
                if confidence < 0.6:
                    logger.warning(f"Low confidence ({confidence:.0%}) - defaulting to edit_request")
                    request_type = "edit_request"
                    analysis_scope = "chapter"
                
                return {
                    "request_type": request_type,
                    "analysis_scope": analysis_scope,
                    # ✅ CRITICAL: Preserve manuscript context!
                    "current_chapter_text": state.get("current_chapter_text", ""),
                    "current_chapter_number": state.get("current_chapter_number"),
                    "prev_chapter_text": state.get("prev_chapter_text"),
                    "prev_chapter_number": state.get("prev_chapter_number"),
                    "next_chapter_text": state.get("next_chapter_text"),
                    "next_chapter_number": state.get("next_chapter_number"),
                    "manuscript": state.get("manuscript", ""),
                    "filename": state.get("filename", ""),
                    "chapter_ranges": state.get("chapter_ranges", []),
                    "current_request": state.get("current_request", ""),
                    "selection_start": state.get("selection_start", -1),
                    "selection_end": state.get("selection_end", -1),
                    "cursor_offset": state.get("cursor_offset", -1),
                    # ✅ CRITICAL: Preserve reference context!
                    "outline_body": state.get("outline_body"),
                    "rules_body": state.get("rules_body"),
                    "style_body": state.get("style_body"),
                    "characters_bodies": state.get("characters_bodies", []),
                    "outline_current_chapter_text": state.get("outline_current_chapter_text"),
                    # ✅ CRITICAL: Preserve critical 5 keys!
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                }
                
            except Exception as parse_error:
                logger.error(f"Failed to parse request type detection: {parse_error}")
                logger.warning("Defaulting to edit_request due to parse error")
                return {
                    "request_type": "edit_request",
                    "analysis_scope": "chapter",
                    # ✅ CRITICAL: Preserve even on error!
                    "current_chapter_text": state.get("current_chapter_text", ""),
                    "current_chapter_number": state.get("current_chapter_number"),
                    "manuscript": state.get("manuscript", ""),
                    "filename": state.get("filename", ""),
                    "chapter_ranges": state.get("chapter_ranges", []),
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                }
            
        except Exception as e:
            logger.error(f"Failed to detect request type: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Default to edit_request on error
            return {
                "request_type": "edit_request",
                "analysis_scope": "chapter",
                # ✅ CRITICAL: Preserve even on error!
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "chapter_ranges": state.get("chapter_ranges", []),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
    
    def _route_from_request_type(self, state: FictionEditingState) -> str:
        """Route based on detected request type - DEPRECATED: All requests go through generation subgraph"""
        # This method is no longer used - all requests (questions and edits) go through the same path
        # The generation subgraph intelligently handles questions: can analyze and optionally edit
        request_type = state.get("request_type", "edit_request")
        return "edit_request"  # Both question and edit_request go to edit path
    
    async def _answer_question_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Answer user questions about the manuscript, references, or general information"""
        try:
            logger.info("Answering user question...")
            
            analysis_scope = state.get("analysis_scope", "chapter")
            logger.info("Answering question with analysis_scope=%s", analysis_scope)
            current_request = state.get("current_request", "")
            manuscript = state.get("manuscript", "")
            current_chapter_text = state.get("current_chapter_text", "")
            current_chapter_number = state.get("current_chapter_number")
            chapter_ranges = state.get("chapter_ranges", [])
            style_body = state.get("style_body")
            rules_body = state.get("rules_body")
            characters_bodies = state.get("characters_bodies", [])
            series_body = state.get("series_body")
            outline_body = state.get("outline_body")
            outline_current_chapter_text = state.get("outline_current_chapter_text")
            
            # ALWAYS use cursor-based chapter detection (from _analyze_scope_node) for chapter scope
            chapter_text_to_show = current_chapter_text
            chapter_number_to_show = current_chapter_number
            prev_chapter_text = state.get("prev_chapter_text")
            next_chapter_text = state.get("next_chapter_text")
            
            # Build context for answering
            context_parts = []
            
            if analysis_scope == "manuscript":
                # Full-manuscript context: full text only; no summarization
                MANUSCRIPT_CHAR_LIMIT = 750_000
                if len(manuscript) > MANUSCRIPT_CHAR_LIMIT:
                    logger.warning(
                        "Whole-story analysis skipped: manuscript %d chars exceeds limit %d",
                        len(manuscript),
                        MANUSCRIPT_CHAR_LIMIT,
                    )
                    limit_mb = MANUSCRIPT_CHAR_LIMIT / 1_000_000
                    user_msg = (
                        f"**Whole-story analysis** — I'm looking at the entire manuscript.\n\n"
                        f"Whole-story analysis isn't available for this request. The manuscript has **{len(manuscript):,} characters**, "
                        f"which exceeds the limit of **{MANUSCRIPT_CHAR_LIMIT:,}** characters (~{limit_mb:.1f}M) for full-manuscript analysis. "
                        f"I don't summarize by chapter; I either analyze the full text or explain why I can't.\n\n"
                        f"**What you can do:** Ask about specific chapters (e.g. \"How does Chapter 5 work?\") or narrow your question to a section of the book."
                    )
                    return {
                        "response": {
                            "response": user_msg,
                            "task_status": "complete",
                            "agent_type": "fiction_editing_agent",
                            "timestamp": datetime.now().isoformat(),
                        },
                        "task_status": "complete",
                        "editor_operations": [],
                        "metadata": state.get("metadata", {}),
                        "user_id": state.get("user_id", "system"),
                        "shared_memory": state.get("shared_memory", {}),
                        "messages": state.get("messages", []),
                        "query": state.get("query", ""),
                    }
                logger.info("Manuscript scope: full text (%d chars)", len(manuscript))
                context_parts.append("=== FULL MANUSCRIPT ===\n")
                context_parts.append(f"{_strip_frontmatter_block(manuscript)}\n\n")
            else:
                # Chapter context - show current, previous, and next chapters (cursor-based)
                if prev_chapter_text:
                    prev_chapter_num = chapter_number_to_show - 1 if chapter_number_to_show else None
                    context_parts.append(f"=== PREVIOUS CHAPTER {prev_chapter_num or 'N'} (FOR CONTEXT) ===\n")
                    context_parts.append(f"{prev_chapter_text}\n\n")
                
                if chapter_text_to_show:
                    context_parts.append(f"=== CURRENT CHAPTER {chapter_number_to_show or 'N'} (CURSOR LOCATION) ===\n")
                    context_parts.append(f"{chapter_text_to_show}\n\n")
                elif manuscript:
                    context_parts.append("=== MANUSCRIPT (NO CHAPTER DETECTED) ===\n")
                    context_parts.append(f"{manuscript[:2000]}\n\n")
                
                if next_chapter_text:
                    next_chapter_num = chapter_number_to_show + 1 if chapter_number_to_show else None
                    context_parts.append(f"=== NEXT CHAPTER {next_chapter_num or 'N'} (FOR CONTEXT) ===\n")
                    context_parts.append(f"{next_chapter_text}\n\n")
            
            # Reference information: omit for whole-story analysis (analyze on its own merits); include for chapter scope
            if analysis_scope != "manuscript":
                if style_body:
                    context_parts.append("=== STYLE GUIDE (SUMMARY) ===\n")
                    style_summary = style_body[:1000] + "..." if len(style_body) > 1000 else style_body
                    context_parts.append(f"{style_summary}\n\n")
                if rules_body:
                    context_parts.append("=== UNIVERSE RULES (SUMMARY) ===\n")
                    rules_summary = rules_body[:1000] + "..." if len(rules_body) > 1000 else rules_body
                    context_parts.append(f"{rules_summary}\n\n")
                if series_body:
                    context_parts.append("=== SERIES TIMELINE (SUMMARY) ===\n")
                    series_summary = series_body[:1000] + "..." if len(series_body) > 1000 else series_body
                    context_parts.append(f"{series_summary}\n\n")
                if characters_bodies:
                    context_parts.append(f"=== CHARACTER PROFILES ({len(characters_bodies)} character(s)) ===\n")
                    context_parts.append("**NOTE**: Each character profile below is for a DIFFERENT character with distinct traits and dialogue patterns.\n\n")
                    for i, char_body in enumerate(characters_bodies, 1):
                        char_name = _extract_character_name(char_body)
                        char_summary = char_body[:400] + "..." if len(char_body) > 400 else char_body
                        context_parts.append(f"**Character {i}: {char_name}**\n{char_summary}\n\n")
                if outline_current_chapter_text or (outline_body and chapter_number_to_show):
                    outline_text_for_question = outline_current_chapter_text
                    if not outline_text_for_question and outline_body and chapter_number_to_show:
                        outline_text_for_question = extract_chapter_outline(outline_body, chapter_number_to_show)
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
            
            # Build request with instructions (manuscript vs chapter framing)
            if analysis_scope == "manuscript":
                request_with_instructions = f"""=== USER QUESTION ===
{current_request}

**YOUR TASK**: Perform a whole-manuscript analysis using ONLY the manuscript text provided. Analyze the story on its own merits: structure, pacing, character arcs, thematic consistency, and narrative quality as they appear in the prose. Do not compare to outline, style guide, or other reference documents — you have only the manuscript. Reference specific chapters and passages to support your observations. Be thorough and specific.

**OUTPUT**: Provide a natural, conversational answer. Do NOT generate JSON or editor operations."""
            else:
                request_with_instructions = f"""=== USER QUESTION ===
{current_request}

**YOUR TASK**: Answer the user's question clearly and helpfully.

- If they're asking about style, explain what style is being used and why
- If they're asking about content, describe what's in the manuscript
- If they're asking about references (characters, rules, style), confirm what's loaded and provide relevant information
- If they're asking about narrative style or writing choices, explain what you observe
- Be conversational and helpful - this is a question, not a content generation request
- If you don't have certain information, say so honestly
- Keep your answer focused and relevant to the question
- You can discuss the manuscript, style, structure, and references - this is a conversation

**OUTPUT**: Provide a natural, conversational answer to the user's question. Do NOT generate JSON or editor operations."""
            
            # Use standardized helper for message construction with conversation history
            messages_list = state.get("messages", [])
            if analysis_scope == "manuscript":
                system_prompt = (
                    "You are a professional fiction editor performing a whole-manuscript analysis. "
                    "You have access ONLY to the manuscript text (no outline, style guide, or character notes). "
                    "Analyze the story on its own merits: structure, pacing, character arcs, "
                    "thematic consistency, and narrative quality as they appear in the prose. "
                    "Reference specific chapters and passages to support your observations."
                )
            else:
                system_prompt = "You are a helpful fiction editing assistant. Answer user questions about the manuscript, style, references, and related information. Be conversational and helpful."
            messages = self._build_editing_agent_messages(
                system_prompt=system_prompt,
                context_parts=context_parts,
                current_request=request_with_instructions,
                messages_list=messages_list,
                look_back_limit=6
            )
            
            # Call LLM for conversational response
            llm = self._get_llm(temperature=0.7, state=state)  # Higher temperature for natural conversation
            
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
                "editor_operations": [],  # No operations for questions
                # ✅ CRITICAL: Preserve metadata for downstream nodes!
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
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
                "error": str(e),
                # ✅ CRITICAL: Preserve metadata even on error!
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
    
    async def _prepare_generation_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Prepare generation state: set system_prompt and datetime_context for generation subgraph"""
        try:
            # ⚠️ ALWAYS build system prompt FRESH - NEVER use checkpointed version
            # The system prompt should only contain instructions, not reference content
            system_prompt = self._build_system_prompt()
            
            # Get datetime context
            datetime_context = self._get_datetime_context()
            
            return {
                "system_prompt": system_prompt,
                "datetime_context": datetime_context,
                # ✅ CRITICAL: Preserve ALL manuscript context for generation subgraph!
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "prev_chapter_text": state.get("prev_chapter_text"),
                "prev_chapter_number": state.get("prev_chapter_number"),
                "next_chapter_text": state.get("next_chapter_text"),
                "next_chapter_number": state.get("next_chapter_number"),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "cursor_offset": state.get("cursor_offset", -1),
                "requested_chapter_number": state.get("requested_chapter_number"),
                "explicit_primary_chapter": state.get("explicit_primary_chapter"),  # CRITICAL: For validating requested_chapter_number freshness
                # ✅ CRITICAL: Preserve reference context too!
                "outline_body": state.get("outline_body"),
                "rules_body": state.get("rules_body"),
                "style_body": state.get("style_body"),
                "characters_bodies": state.get("characters_bodies", []),
                "series_body": state.get("series_body"),
                "outline_current_chapter_text": state.get("outline_current_chapter_text"),
                # ✅ CRITICAL: Preserve critical 5 keys!
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
        except Exception as e:
            logger.error(f"Failed to prepare generation: {e}")
            return {
                "error": str(e),
                "task_status": "error",
                # ✅ CRITICAL: Preserve even on error!
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "chapter_ranges": state.get("chapter_ranges", []),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
    
    def _route_after_resolve(self, state: FictionEditingState) -> str:
        """Route after resolve_operations: always to format_response"""
        return "format_response"
    
    # _generate_edit_plan_node removed - now handled by generation_subgraph
    # Functionality moved to: orchestrator/subgraphs/fiction_generation_subgraph.py
    
    # _resolve_operations_node removed - now handled by resolution_subgraph
    # Functionality moved to: orchestrator/subgraphs/fiction_resolution_subgraph.py
    
    async def _format_response_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Format final response with editor operations"""
        try:
            logger.info("Formatting response...")
            
            # Extract operations from state FIRST (needed throughout function)
            editor_operations = state.get("editor_operations", [])
            failed_operations = state.get("failed_operations", [])
            
            # Check if this is a question response (already formatted in answer_question_node)
            # BUT: If operations were generated, include them even for questions
            existing_response = state.get("response", {})
            
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
                # Mention whole-story mode when applicable
                if state.get("analysis_scope") == "manuscript":
                    prefix = "**Whole-story analysis** — I'm looking at the entire manuscript.\n\n"
                    existing_response["response"] = prefix + (existing_response.get("response") or "")
                return {
                    "response": existing_response,
                    "task_status": existing_response.get("task_status", "complete")
                }
            
            structured_edit = _get_structured_edit(state)
            # editor_operations already extracted above
            task_status = state.get("task_status", "complete")
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
            
            # Handle questions (with or without operations)
            if request_type == "question":
                if structured_edit and structured_edit.summary:
                    summary = structured_edit.summary
                    if len(editor_operations) == 0:
                        # Check if we have failed_operations that should be converted for manual application
                        if failed_operations:
                            logger.info(f"Question request with no successful operations but {len(failed_operations)} failed_operations - will convert for manual application")
                            # Don't return early - fall through to handle failed_operations conversion below
                            response_text = summary
                        else:
                            # Pure question with no operations at all - return summary as response
                            logger.info("Question request with no operations - using summary as response")
                            response_content = summary
                            if state.get("analysis_scope") == "manuscript":
                                response_content = "**Whole-story analysis** — I'm looking at the entire manuscript.\n\n" + response_content
                            return {
                                "response": {
                                    "response": response_content,
                                    "task_status": "complete",
                                    "agent_type": "fiction_editing_agent"
                                },
                                "task_status": "complete",
                                "editor_operations": []
                            }
                    else:
                        # Question with operations - use summary as response text (conversational feedback)
                        logger.info(f"Question request with {len(editor_operations)} operations - using summary as conversational response")
                        response_text = summary
                else:
                    # Question but no summary - fallback
                    logger.warning("Question request but no summary in structured_edit - using fallback")
                    if editor_operations:
                        response_text = f"Analysis complete. Made {len(editor_operations)} edit(s) based on your question."
                    elif failed_operations:
                        response_text = f"Analysis complete. Generated {len(failed_operations)} edit(s) that require manual placement."
                    else:
                        response_text = "Analysis complete."
            # Build response text for edit requests - use summary, not full text
            else:
                # Check if this was a proofreading request (has mode in state)
                mode = state.get("mode", "")
                is_proofreading = mode in ["clarity", "compliance", "accuracy"]
                
                if is_proofreading:
                    # Proofreading-specific response formatting
                    if structured_edit and structured_edit.summary:
                        response_text = f"## Proofreading ({mode})\n\n{structured_edit.summary}\n\n"
                    elif editor_operations:
                        op_count = len(editor_operations)
                        response_text = f"## Proofreading ({mode})\n\n**{op_count} correction(s)** ready to apply.\n"
                    else:
                        response_text = f"## Proofreading ({mode})\n\nProofreading complete. No issues found.\n"
                else:
                    # Regular editing response
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
            
            # Add failed operations if present (already extracted at top of function)
            if failed_operations:
                failed_section = "\n\n**⚠️ UNRESOLVED EDITS (Manual Action Required)**\n"
                failed_section += "The following generated content could not be automatically placed in the manuscript. You can copy and paste these sections manually:\n\n"
                
                for i, op in enumerate(failed_operations, 1):
                    op_type = op.get("op_type", "edit")
                    error = op.get("error", "Anchor text not found")
                    text = op.get("text", "")
                    anchor = op.get("anchor_text") or op.get("original_text")
                    
                    failed_section += f"#### Unresolved Edit {i} ({op_type})\n"
                    failed_section += f"- **Reason**: {error}\n"
                    if anchor:
                        # Use a blockquote for the anchor to keep it distinct but readable
                        failed_section += f"- **Intended near**:\n> {anchor[:200]}...\n"
                    
                    failed_section += "\n**Generated Content** (Scroll-safe):\n"
                    # Use a standard markdown block for the content, which usually handles wrapping better in sidebars
                    # than nested JSON or complex pre tags.
                    failed_section += f"{text}\n\n"
                    failed_section += "---\n"
                
                response_text = response_text + failed_section
            
            # Mention whole-story mode when applicable
            if state.get("analysis_scope") == "manuscript":
                response_text = "**Whole-story analysis** — I'm looking at the entire manuscript.\n\n" + response_text
            
            # Build response with editor operations
            response = {
                "response": response_text,
                "task_status": task_status,
                "agent_type": "fiction_editing_agent",
                "timestamp": datetime.now().isoformat()
            }
            
            # Include editor operations when present
            if editor_operations:
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
            else:
                logger.info(f"ℹ️ Format response: No editor_operations to include (editor_operations from state: {len(state.get('editor_operations', []))})")
                
                # Check if we have failed_operations that should be converted to editor_operations for manual application
                if failed_operations:
                    logger.info(f"⚠️ No successful editor_operations, but {len(failed_operations)} failed_operations found - converting to editor_operations for manual application")
                    
                    # Convert failed_operations to editor_operations format
                    # These will be marked as requiring manual placement since they couldn't be auto-resolved
                    manual_editor_operations = []
                    for failed_op in failed_operations:
                        op_type = failed_op.get("op_type", "insert_after")
                        text = failed_op.get("text", "")
                        original_text = failed_op.get("original_text", "")
                        anchor_text = failed_op.get("anchor_text", "")
                        
                        # Create editor operation in standard format
                        # Use -1 for start/end to indicate manual placement needed
                        manual_op = {
                            "op_type": op_type,
                            "text": text,
                            "original_text": original_text or anchor_text,  # Use anchor/original as context
                            "anchor_text": anchor_text or original_text,
                            "start": -1,  # Indicates manual placement required
                            "end": -1,    # Indicates manual placement required
                            "requires_manual_placement": True,  # Flag for frontend
                            "error": failed_op.get("error", "Anchor or original text not found")
                        }
                        manual_editor_operations.append(manual_op)
                    
                    # Include converted operations as editor_operations
                    response["editor_operations"] = manual_editor_operations
                    logger.info(f"✅ Converted {len(manual_editor_operations)} failed_operations to editor_operations for manual application")
                    
                    # Create manuscript_edit structure for failed operations
                    if structured_edit:
                        response["manuscript_edit"] = {
                            "target_filename": structured_edit.target_filename,
                            "scope": structured_edit.scope,
                            "summary": structured_edit.summary or "Generated content requires manual placement",
                            "chapter_index": structured_edit.chapter_index,
                            "safety": structured_edit.safety,
                            "operations": manual_editor_operations
                        }
                    else:
                        response["manuscript_edit"] = {
                            "target_filename": state.get("filename", "manuscript.md"),
                            "scope": "unknown",
                            "summary": "Generated content requires manual placement - anchor text not found",
                            "chapter_index": None,
                            "safety": "medium",
                            "operations": manual_editor_operations
                        }
                else:
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
                
                # ⚠️ CRITICAL: Clear heavy reference fields from checkpointed state
                # These fields should be loaded FRESH every turn from files, not accumulated
                # This prevents sending Chapter 1 content multiple times!
                heavy_fields_to_clear = [
                    "outline_body", "rules_body", "style_body", "characters_bodies", "series_body",
                    "outline_current_chapter_text", "loaded_references",
                    # Also clear manuscript content (will be refreshed from active_editor)
                    "manuscript", "current_chapter_text", "prev_chapter_text", "next_chapter_text",
                    # ⚠️ CRITICAL: Also clear system_prompt as it can get bloated with references
                    "system_prompt"
                ]
                cleared_count = 0
                for field in heavy_fields_to_clear:
                    if existing_shared_memory.pop(field, None) is not None:
                        cleared_count += 1
                
                if cleared_count > 0:
                    logger.info(f"🧹 Cleared {cleared_count} heavy reference fields from checkpointed shared_memory (will reload fresh)")
            
            # CRITICAL: Ensure user_chat_model is in BOTH metadata and shared_memory, prioritizing NEW values
            # This ensures _get_llm() can find it in metadata first (which takes precedence)
            # Priority: new metadata > new shared_memory > checkpoint shared_memory
            user_chat_model = None
            if "user_chat_model" in metadata:
                user_chat_model = metadata["user_chat_model"]
            elif "user_chat_model" in shared_memory:
                user_chat_model = shared_memory["user_chat_model"]
            elif existing_shared_memory and "user_chat_model" in existing_shared_memory:
                user_chat_model = existing_shared_memory["user_chat_model"]
            
            # Merge shared_memory: start with checkpoint, then update with NEW data (so new active_editor overwrites old)
            shared_memory_merged = existing_shared_memory.copy()
            shared_memory_merged.update(shared_memory)  # New data (including updated active_editor) takes precedence
            
            # Ensure user_chat_model is in both places with the correct (newest) value
            if user_chat_model:
                metadata["user_chat_model"] = user_chat_model
                shared_memory_merged["user_chat_model"] = user_chat_model
                logger.debug(f"🎯 ENSURED user_chat_model in both metadata and shared_memory: {user_chat_model}")
            
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
                "series_body": None,
                "outline_current_chapter_text": None,
                "current_request": "",
                "requested_chapter_number": None,
                "system_prompt": "",
                "llm_response": "",
                "structured_edit": None,
                "editor_operations": [],
                "failed_operations": [],
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
                # Outline sync detection
                "outline_sync_analysis": None,
                "outline_needs_sync": False
            }
            
            # Run LangGraph workflow with checkpointing (workflow and config already created above)
            result_state = await workflow.ainvoke(initial_state, config=config)
            
            # Extract final response
            response = result_state.get("response", {})
            task_status = result_state.get("task_status", "complete")
            
            if task_status == "error":
                error_msg = result_state.get("error", "Unknown error")
                logger.error(f"Fiction editing agent failed: {error_msg}")
                return {
                    "response": f"Fiction editing failed: {error_msg}",
                    "task_status": "error",
                    "agent_results": {}
                }
            
            # Get editor_operations and failed_operations
            editor_ops_from_response = response.get("editor_operations", []) if isinstance(response, dict) else []
            editor_ops_from_state = result_state.get("editor_operations", [])
            editor_operations = editor_ops_from_response or editor_ops_from_state
            
            failed_ops = result_state.get("failed_operations", [])
            
            if editor_ops_from_state and not editor_ops_from_response:
                logger.warning(f"⚠️ Process: editor_operations missing from response dict but present in state - using state")
            
            # Extract response text - ensure it's never empty
            if isinstance(response, dict):
                response_text = response.get("response", "")
                # Fallback to summary from manuscript_edit if response text is empty
                if not response_text:
                    manuscript_edit = response.get("manuscript_edit")
                    if manuscript_edit and isinstance(manuscript_edit, dict):
                        response_text = manuscript_edit.get("summary", "")
                # Final fallback
                if not response_text:
                    if editor_operations:
                        response_text = f"Generated {len(editor_operations)} edit(s) to the manuscript."
                    else:
                        response_text = "Fiction editing complete"
            else:
                response_text = str(response) if response else "Fiction editing complete"
            
            # Build result dict matching character_development_agent pattern
            result = {
                "response": response_text,
                "task_status": task_status,
                "agent_results": {
                    "editor_operations": editor_operations,
                    "failed_operations": failed_ops,
                    "manuscript_edit": response.get("manuscript_edit") if isinstance(response, dict) else None
                }
            }
            
            # Add editor operations at top level for compatibility
            if editor_operations:
                result["editor_operations"] = editor_operations
            else:
                logger.warning(f"⚠️ Fiction editing agent: No editor_operations found (response keys: {list(response.keys()) if isinstance(response, dict) else 'not a dict'}, state has ops: {bool(editor_ops_from_state)})")
            
            if isinstance(response, dict) and response.get("manuscript_edit"):
                result["manuscript_edit"] = response["manuscript_edit"]
            
            logger.info(f"Fiction editing agent completed: {task_status}, result has editor_operations: {bool(result.get('editor_operations'))}, response_text length: {len(response_text)}")
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

