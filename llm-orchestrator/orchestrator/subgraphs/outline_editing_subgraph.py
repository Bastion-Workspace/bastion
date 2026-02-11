"""
Outline Editing Subgraph

Reusable subgraph for outline document editing workflows.
Used by Writing Assistant when the active editor is an outline document (frontmatter.type: outline).

Supports outline development with:
- Type gating: Strict validation for type: outline documents
- Chapter detection: Finds chapter ranges, counts beats per chapter
- Structure analysis: Assesses outline completeness, chapter count, identifies missing sections
- Reference quality assessment: Evaluates loaded references (style, rules, characters) and generates warnings
- 100 Beat Limit Enforcement: System prompt instructs LLM to prune beats when approaching 100-beat chapter limit
- Chapter Heading Format: Strict enforcement of "## Chapter N" format (no titles in headings)
- Clarification Request Handling: Supports multi-turn clarification for empty outlines
- Series Timeline Support: Loads series timeline for cross-book continuity
- Question vs Edit Detection: Distinguishes analysis questions from edit requests

Produces EditorOperations suitable for Prefer Editor HITL application.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import ValidationError

from orchestrator.models.editor_models import ManuscriptEdit
from orchestrator.models.agent_response_contract import AgentResponse, ManuscriptEditMetadata
from orchestrator.utils.editor_operation_resolver import resolve_editor_operation
from orchestrator.utils.writing_subgraph_utilities import (
    preserve_critical_state,
    create_writing_error_response,
    extract_user_request,
    paragraph_bounds,
    strip_frontmatter_block,
    slice_hash,
    build_response_text_for_question,
    build_response_text_for_edit,
    build_failed_operations_section,
    create_manuscript_edit_metadata,
    prepare_writing_context,
    load_writing_references
)

logger = logging.getLogger(__name__)


# ============================================
# Utility Functions
# ============================================

# Utility functions moved to writing_subgraph_utilities
# Imported: slice_hash, strip_frontmatter_block, paragraph_bounds


def _frontmatter_end_index(text: str) -> int:
    """Return the end index of a leading YAML frontmatter block if present, else 0."""
    try:
        m = re.match(r'^(---\s*\n[\s\S]*?\n---\s*\n)', text, flags=re.MULTILINE)
        if m:
            return m.end()
        return 0
    except Exception:
        return 0


def _unwrap_json_response(content: str) -> str:
    """Extract raw JSON from LLM output if wrapped in code fences or prose."""
    try:
        json.loads(content)
        return content
    except Exception:
        pass
    try:
        text = content.strip()
        text = re.sub(r"^```(?:json)?\s*\n([\s\S]*?)\n```\s*$", r"\1", text)
        try:
            json.loads(text)
            return text
        except Exception:
            pass
        start = text.find('{')
        if start == -1:
            return content
        brace = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == '{':
                brace += 1
            elif ch == '}':
                brace -= 1
                if brace == 0:
                    snippet = text[start:i+1]
                    try:
                        json.loads(snippet)
                        return snippet
                    except Exception:
                        break
        return content
    except Exception:
        return content


# paragraph_bounds moved to writing_subgraph_utilities


@dataclass
class ChapterRange:
    heading_text: str
    chapter_number: Optional[int]
    start: int
    end: int


CHAPTER_PATTERN = re.compile(r"^##\s+Chapter\s+(\d+)\b.*$", re.MULTILINE)


def find_chapter_ranges(text: str) -> List[ChapterRange]:
    """Find all chapter ranges in text."""
    if not text:
        return []
    matches = list(CHAPTER_PATTERN.finditer(text))
    if not matches:
        return []
    ranges: List[ChapterRange] = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chapter_num: Optional[int] = None
        try:
            chapter_num = int(m.group(1))
        except Exception:
            chapter_num = None
        ranges.append(ChapterRange(heading_text=m.group(0), chapter_number=chapter_num, start=start, end=end))
    return ranges


def count_beats_in_chapter(chapter_text: str) -> int:
    """Count the number of beats (lines starting with '- ') in a chapter."""
    if not chapter_text:
        return 0
    lines = chapter_text.split('\n')
    beat_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('- '):
            beat_count += 1
    return beat_count


def find_last_line_of_last_chapter(outline: str) -> Optional[str]:
    """Find the last non-empty line of the last chapter in the outline.
    
    Returns the actual last line of text (could be a bullet point, summary sentence, etc.)
    that can be used as anchor_text for inserting a new chapter.
    Preserves original whitespace for exact matching in the resolver.
    """
    if not outline:
        return None
    
    chapter_ranges = find_chapter_ranges(outline)
    if not chapter_ranges:
        # No chapters found - return last non-empty line of entire document
        lines = outline.rstrip().split('\n')
        for line in reversed(lines):
            if line.strip():
                # Preserve original line (with original whitespace) for exact matching
                return line.rstrip()
        return None
    
    # Get the last chapter
    last_chapter = chapter_ranges[-1]
    chapter_content = outline[last_chapter.start:last_chapter.end]
    
    # Find the last non-empty line in this chapter
    # Split preserving line endings - we'll reconstruct with original content
    lines = chapter_content.split('\n')
    for line in reversed(lines):
        stripped = line.strip()
        # Skip the chapter heading itself and empty lines
        if stripped and not stripped.startswith('## Chapter'):
            # Return the line with original whitespace preserved (just strip trailing newline)
            return line.rstrip()
    
    # If no content found (empty chapter), return the chapter heading
    return last_chapter.heading_text.rstrip()


def _assess_reference_quality(content: str, ref_type: str) -> Tuple[float, List[str]]:
    """
    Assess reference quality and return (quality_score, warnings).
    Returns quality score 0.0-1.0 and list of warnings.
    """
    if not content or len(content.strip()) < 50:
        return 0.0, ["Reference content is too short or empty"]
    
    quality_score = 0.5  # Base score for existing content
    warnings = []
    
    content_length = len(content.strip())
    
    if ref_type == "rules":
        # Good rules have structure and specificity
        if "## " in content or "- " in content:  # Has structure
            quality_score += 0.2
        else:
            warnings.append("Rules lack clear structure (no headings or bullets)")
        
        if content_length > 500:  # Substantial content
            quality_score += 0.3
        elif content_length < 200:
            warnings.append("Rules content is quite brief")
        
        # Check for key rule indicators
        rule_keywords = ["rule", "constraint", "limit", "cannot", "must", "cannot", "forbidden"]
        if any(kw in content.lower() for kw in rule_keywords):
            quality_score += 0.1
    
    elif ref_type == "style":
        # Good style guides have examples and specifics
        if "example" in content.lower() or "```" in content:
            quality_score += 0.2
        else:
            warnings.append("Style guide lacks examples")
        
        if content_length > 300:
            quality_score += 0.3
        elif content_length < 150:
            warnings.append("Style guide is quite brief")
        
        # Check for style indicators
        style_keywords = ["voice", "tone", "pacing", "dialogue", "narrative", "tense", "pov"]
        if any(kw in content.lower() for kw in style_keywords):
            quality_score += 0.1
    
    elif ref_type == "characters":
        # Good character profiles have detail
        if content_length > 400:
            quality_score += 0.3
        elif content_length < 200:
            warnings.append("Character profile is quite brief")
        
        # Check for character depth indicators
        char_keywords = ["motivation", "personality", "goal", "backstory", "trait", "relationship"]
        if any(kw in content.lower() for kw in char_keywords):
            quality_score += 0.2
        else:
            warnings.append("Character profile lacks depth indicators")
    
    return min(1.0, quality_score), warnings


def _extract_conversation_history(messages: List[Any], limit: int = 10) -> List[Dict[str, str]]:
    """Extract conversation history from LangChain messages, filtering out large data URIs"""
    try:
        history = []
        for msg in messages[-limit:]:
            if hasattr(msg, 'content'):
                role = "assistant" if hasattr(msg, 'type') and msg.type == "ai" else "user"
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                history.append({
                    "role": role,
                    "content": content
                })
        return history
    except Exception as e:
        logger.error(f"Failed to extract conversation history: {e}")
        return []


def _build_editing_agent_messages(
    system_prompt: str,
    context_parts: List[str],
    current_request: str,
    messages_list: List[Any],
    get_datetime_context: Callable,
    look_back_limit: int = 6
) -> List[Any]:
    """
    Build message list for editing agents with conversation history and separate context
    
    STANDALONE VERSION for subgraph use (doesn't require BaseAgent)
    
    Message Structure:
    1. SystemMessage: system_prompt
    2. SystemMessage: datetime_context
    3. Conversation history as alternating HumanMessage/AIMessage objects
    4. HumanMessage: file context (from context_parts - file content, references)
    5. HumanMessage: current_request (user query + mode-specific instructions)
    
    Args:
        system_prompt: System-level instructions for the agent
        context_parts: List of context strings (file content, references, etc.)
        current_request: User's request with mode-specific instructions
        messages_list: Conversation history from state.get("messages", [])
        get_datetime_context: Function that returns datetime context string
        look_back_limit: Number of previous messages to include (default: 6)
        
    Returns:
        List of LangChain message objects ready for LLM
    """
    # Start with system messages
    messages = [
        SystemMessage(content=system_prompt),
        SystemMessage(content=get_datetime_context())
    ]
    
    # Add conversation history as proper message objects
    if messages_list:
        conversation_history = _extract_conversation_history(
            messages_list, 
            limit=look_back_limit
        )
        
        # Remove last message if it duplicates current_request
        if conversation_history and len(conversation_history) > 0:
            last_msg = conversation_history[-1]
            if last_msg.get("content") == current_request:
                conversation_history = conversation_history[:-1]
        
        # Add as proper message objects
        for msg in conversation_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
    
    # Add file context as separate message
    if context_parts:
        messages.append(HumanMessage(content="".join(context_parts)))
    
    # Add current request with instructions as separate message
    if current_request:
        messages.append(HumanMessage(content=current_request))
    
    return messages


# ============================================
# System Prompt Builder
# ============================================

def _build_system_prompt() -> str:
    """Build system prompt for outline editing"""
    return (
        "You are an outline editor. Generate operations to edit story outlines.\n\n"
        "**CRITICAL: WORK WITH AVAILABLE INFORMATION FIRST**\n"
        "Always start by working with what you know from the request, existing outline content, and references:\n"
        "- Make edits based on available information - don't wait for clarification\n"
        "- Use context from rules, style guide, character profiles, and series timeline to inform your work\n"
        "- Add or revise content based on reasonable inferences from the request\n"
        "- **SERIES TIMELINE (if provided)**: Use for cross-book continuity and timeline consistency\n"
        "  - Reference major events from previous books when relevant to current outline\n"
        "  - Ensure timeline consistency (character ages, years, historical events)\n"
        "  - Maintain continuity with established series events\n"
        "  - Example: If series timeline says 'Franklin died in 1962 (Book 12)', ensure later books reflect this\n\n"
        "- **FOR EMPTY FILES**: When the outline is empty (only frontmatter), ASK QUESTIONS FIRST before creating content\n"
        "  * Don't create the entire outline structure at once\n"
        "  * Ask about story genre, main characters, key plot points, or chapter count\n"
        "  * Build incrementally based on user responses\n"
        "- Only proceed without questions when you have enough information to make meaningful edits\n"
        "\n"
        "**WHEN TO ASK QUESTIONS**:\n"
        "- **ALWAYS for empty files**: When outline is empty, ask questions about story basics before creating content\n"
        "- When the request is vague and you cannot make reasonable edits (e.g., 'improve outline' with no existing content)\n"
        "- When there's a critical plot conflict that requires user decision (e.g., existing beat directly contradicts new request)\n"
        "- When user requests a large amount of content (e.g., 'create the whole outline') - break it into steps and ask about priorities\n"
        "- When asking, you can provide operations for what you CAN do, then ask questions in the summary about what you need\n"
        "\n"
        "**HOW TO ASK QUESTIONS**: Include operations for work you CAN do, then add questions/suggestions in the summary field.\n"
        "For empty files, it's acceptable to return a clarification request with questions instead of operations.\n"
        "DO NOT return empty operations array for edit requests - always provide edits OR ask questions.\n\n"
        "**HANDLING QUESTIONS THAT DON'T REQUIRE EDITS**:\n"
        "- If the user is asking a question that can be answered WITHOUT making edits to the outline\n"
        "- Examples: \"What unresolved plot points do we have?\", \"Give me a list of...\", \"Show me...\", \"Analyze...\", \"What chapters...\"\n"
        "- OR if the user explicitly says \"don't edit\", \"no edits\", \"just answer\", \"only analyze\", or similar phrases\n"
        "- THEN return a ManuscriptEdit with:\n"
        "  * Use standard scope value (\"paragraph\" is fine for questions)\n"
        "  * EMPTY operations array ([])\n"
        "  * Your complete answer in the summary field\n"
        "- The summary should contain the full answer to the user's question (e.g., bullet points, analysis, recommendations, lists)\n"
        "- This allows you to provide information, analysis, or recommendations without making any edits to the outline\n\n"
        "OUTLINE STRUCTURE:\n"
        "# Overall Synopsis (high-level story summary - major elements only)\n"
        "# Notes (rules, themes, worldbuilding)\n"
        "# Characters (BRIEF list only: names and roles like 'Protagonist: John', 'Antagonist: Sarah')\n"
        "  **CRITICAL**: Character DETAILS belong in character profile files, NOT in the outline!\n"
        "  The outline should only have brief character references (name + role), not full profiles.\n"
        "  Do NOT copy character descriptions, backstories, or traits into the outline.\n"
        "## Chapter N\n"
        "### Status (optional, for chapters after Chapter 1)\n"
        "### Pacing (optional, recommended for Chapter 2+)\n"
        "### Summary (brief 3-5 sentence overview)\n"
        "### Beats (detailed bullet point events, max 100)\n\n"
        "**CRITICAL - CHAPTER HEADING FORMAT**:\n"
        "- Chapter headings MUST be EXACTLY \"## Chapter N\" where N is the chapter number\n"
        "- **NEVER add titles, names, or descriptions to chapter headings**\n"
        "- **CORRECT**: \"## Chapter 1\", \"## Chapter 5\", \"## Chapter 12\"\n"
        "- **WRONG**: \"## Chapter 1: The Beginning\", \"## Chapter 5 - The Confrontation\", \"## Chapter 12: Final Battle\"\n"
        "- If you want chapter titles, you can add them yourself or ask the user - the outline agent will NEVER add them automatically\n"
        "- Chapter headings are ONLY for numbering - no additional text after the number\n\n"
        "CHAPTER STATUS TRACKING (OPTIONAL):\n"
        "- If the outline frontmatter defines 'tracked_items', each chapter after Chapter 1 should include a status block\n"
        "- **CRITICAL PLACEMENT**: Status blocks appear IMMEDIATELY after the chapter heading (## Chapter N), BEFORE the summary paragraph\n"
        "- **CRITICAL MEANING**: Status represents the state of tracked items AT THE BEGINNING of the chapter (based on previous chapter's events)\n"
        "- Format:\n"
        "  ### Status\n"
        "  - [Item name]: [State at beginning of chapter]\n"
        "  - [Item name]: [State at beginning of chapter]\n"
        "- Include tracked items that are: (1) relevant to the current chapter, OR (2) have changed, OR (3) should be carried forward from previous chapter\n"
        "- **CARRY FORWARD**: If a tracked item appeared in previous chapter's status and is still active/relevant, include it even if unchanged\n"
        "- This ensures the fiction agent always has context for tracked items, especially relationships\n"
        "- Keep status descriptions brief (one line per item)\n"
        "- For characters: location at chapter start, state (alive/dead/injured), where they ended up in previous chapter\n"
        "- For locations: condition at chapter start, occupancy, what happened in previous chapter\n"
        "- For items: location at chapter start, condition, ownership, where they ended up in previous chapter\n"
        "- For relationships: current state at chapter start, how they changed in previous chapter\n"
        "- Chapter 1 MAY have a status block (optional) - useful for series or stories starting mid-action\n"
        "  - If Chapter 1 has status, it should describe the initial state at story start\n"
        "  - Can be manually filled in for series continuity or pre-existing story state\n"
        "  - Status generation will NOT automatically create Chapter 1 status (no previous chapter to track from)\n"
        "- For chapters after Chapter 1, status blocks are automatically generated/updated by the system\n"
        "- **PLACEMENT RULE**: Status must be inserted RIGHT AFTER the chapter heading, before any pacing, summary, or beats\n\n"
        "CHAPTER PACING TRANSITIONS (OPTIONAL, RECOMMENDED FOR CHAPTER 2+):\n"
        "- Pacing blocks help ensure smooth emotional and tonal transitions between chapters\n"
        "- **CRITICAL PLACEMENT**: Pacing blocks appear AFTER Status (if present), BEFORE Summary\n"
        "- **CRITICAL MEANING**: Pacing guides how to transition from the previous chapter's emotional/tonal state to the current chapter's target state\n"
        "- Format:\n"
        "  ### Pacing\n"
        "  FROM: [Emotional/tonal state from previous chapter's ending]\n"
        "  TO: [Target emotional/tonal state for current chapter]\n"
        "  TECHNIQUE: [Specific transition approach - how to achieve the shift]\n"
        "- The FROM state for Chapter N should match the TO state from Chapter N-1 (ensures continuity)\n"
        "- TECHNIQUE examples:\n"
        "  * 'Start with lingering calm, introduce first unsettling detail midway, accelerate toward revelation cliffhanger'\n"
        "  * 'Gradual escalation through scene progression, building tension incrementally'\n"
        "  * 'Sharp tonal break with cold open in new emotional register, flashback provides bridge'\n"
        "  * 'Maintain previous tone until midpoint shift, then rapid escalation'\n"
        "- Pacing is OPTIONAL for Chapter 1 (no previous chapter to transition from) but RECOMMENDED for Chapter 2+\n"
        "- Pacing guidance helps prevent jarring tonal shifts and creates publication-ready prose quality\n"
        "- Think of pacing as the emotional arc between chapters, not just plot progression\n"
        "- **PLACEMENT RULE**: Pacing must be inserted AFTER Status (if present), BEFORE Summary\n\n"
        "CHAPTER STRUCTURE (CRITICAL):\n"
        "- Chapter format MUST follow this exact structure:\n"
        "  ## Chapter N\n"
        "  ### Status (optional - can appear in any chapter, including Chapter 1)\n"
        "  ### Pacing (optional - recommended for Chapter 2+)\n"
        "  ### Summary\n"
        "  ### Beats\n\n"
        "- Example format for Chapter 2+ (with Status and Pacing):\n"
        "  ## Chapter 2\n"
        "  ### Status\n"
        "  - Clarissa: Went home in chapter 1\n"
        "  - Benedict: At the warehouse\n"
        "  ### Pacing\n"
        "  FROM: The contemplative calm of Chapter 1's resolution\n"
        "  TO: Rising tension as discoveries create new urgency\n"
        "  TECHNIQUE: Start with lingering calm, introduce first unsettling detail midway, accelerate toward revelation cliffhanger\n"
        "  ### Summary\n"
        "  Benedict uncovers evidence linking the warehouse to Clarissa's past while Clarissa finds a hidden letter that changes everything she thought she knew.\n"
        "  ### Beats\n"
        "  - Benedict interviews witness at warehouse\n"
        "  - Clarissa discovers hidden letter in library\n"
        "  - [additional beats...]\n\n"
        "- Example format for Chapter 1 (with optional status, no pacing needed):\n"
        "  ## Chapter 1\n"
        "  ### Status\n"
        "  - [Can be manually filled in for series continuity or initial state]\n"
        "  ### Summary\n"
        "  Clarissa discovers the truth about her past while Benedict investigates the warehouse explosion. Tensions rise as both characters uncover secrets that will change everything.\n"
        "  ### Beats\n"
        "  - Clarissa receives mysterious letter\n"
        "  - Benedict arrives at warehouse crime scene\n"
        "  - [additional beats...]\n\n"
        "CHAPTER SUMMARY REQUIREMENTS (CRITICAL):\n"
        "- Each chapter MUST have a '### Summary' header followed by a BRIEF, HIGH-LEVEL summary paragraph (3-5 sentences MAXIMUM)\n"
        "- The summary should be a QUICK OVERVIEW of the chapter's main events, NOT a detailed synopsis\n"
        "- Think of it as a \"back of the book\" description for this chapter - what happens in broad strokes?\n"
        "- DO NOT write lengthy, detailed chapter-by-chapter synopses - keep summaries concise and focused\n"
        "- The summary should capture the ESSENCE of the chapter, not every plot detail (details go in beats)\n"
        "- If your summary exceeds 5 sentences, it's too detailed - trim it down to the core story elements\n"
        "- Format: '### Summary' on its own line, followed by the summary paragraph\n\n"
        "BEAT FORMATTING AND LIMITS:\n"
        "- Each chapter MUST have a '### Beats' header followed by bullet point beats\n"
        "- Every beat MUST start with '- ' (dash space)\n"
        "- **NEVER number beats** - Do NOT use numbered lists (1., 2., 3., etc.) or any numbering format\n"
        "- Beats are specific plot events/actions (THIS is where details belong)\n"
        "- Format: '### Beats' on its own line, followed by bullet points (dash space only, NO numbers)\n"
        "- **CRITICAL - 100 BEAT LIMIT**: Each chapter MUST have a MAXIMUM of 100 beats\n"
        "- When adding new beats to a chapter that already has beats:\n"
        "  * Count the existing beats first\n"
        "  * If adding new beats would exceed 100, you MUST prune less important beats to make room\n"
        "  * Prioritize plot-critical beats over minor details\n"
        "  * Remove redundant or less essential beats to stay within the 100-beat limit\n"
        "  * Example: If chapter has 48 beats and user wants to add 5 new beats, remove 3 least important existing beats\n"
        "- Each chapter needs: (1) BRIEF 3-5 sentence summary paragraph AND (2) detailed beats (max 100)\n"
        "- **CRITICAL**: The summary is BRIEF and HIGH-LEVEL; the beats are DETAILED\n\n"
        "**ABSOLUTE PROHIBITION ON DIALOGUE**:\n"
        "- **NEVER include actual dialogue** (quoted speech) in outline beats\n"
        "- Dialogue belongs in the fiction manuscript, NOT in the outline\n"
        "- You CAN mention talking/conversation as an event: \"- Character discusses plan with ally\"\n"
        "- You CAN describe what is discussed: \"- Character reveals secret to ally during conversation\"\n"
        "- You CANNOT include: \"- Character says 'I have a secret'\" or any quoted dialogue\n"
        "- Think of beats as plot events, not prose - describe what happens, not how characters speak\n"
        "- Example CORRECT: \"- Character confronts antagonist about betrayal\"\n"
        "- Example WRONG: \"- Character says 'You betrayed me!' to antagonist\"\n\n"
        "OPERATIONS:\n\n"
        "**1. replace_range - REPLACING OR MODIFYING EXISTING TEXT ONLY**:\n"
        "- Use this ONLY when you need to CHANGE, MODIFY, or FIX existing content\n"
        "- You MUST provide 'original_text' with EXACT text from the current outline that you want to replace\n"
        "- Copy 20+ words of EXACT text that you want to replace\n"
        "- Use cases: Fixing errors, changing wording, updating specific beats, correcting information\n"
        "- Example: Chapter has \"- Character escapes in a canoe\", you want \"- Character escapes in a boat\"\n"
        "  then original_text=\"- Character escapes in a canoe\" and text=\"- Character escapes in a boat\"\n"
        "- ⚠️ **CRITICAL: DO NOT use replace_range for ADDING new content!**\n"
        "  * If you want to ADD beats, use insert_after_heading instead\n"
        "  * Only use replace_range when you need to CHANGE existing text\n"
        "  * replace_range is for MODIFICATIONS, not ADDITIONS\n"
        "- If you can't find exact text in the outline, DO NOT use replace_range\n\n"
        "**2. insert_after_heading - ADDING NEW TEXT (MOST COMMON FOR ADDITIONS)**:\n"
        "- Use this when ADDING new content to chapters or sections\n"
        "- ⚠️ **CRITICAL: ONLY include NEW content in the 'text' field - DO NOT include existing beats!**\n"
        "  * Your 'text' field should ONLY contain the NEW beats you're adding\n"
        "  * DO NOT copy existing beats \"for context\" - just provide the new ones\n"
        "  * The system will insert your new content after the anchor point\n"
        "- **CRITICAL FOR EMPTY FILES**: If the outline file is empty (only frontmatter), DO NOT provide 'anchor_text'\n"
        "  * Empty file = no content exists yet, so there's nothing to anchor to\n"
        "  * Omit the 'anchor_text' field entirely - the system will insert after frontmatter automatically\n"
        "  * Example for empty file: {\"op_type\": \"insert_after_heading\", \"text\": \"## Chapter 1\\n\\n[content]\"}\n"
        "- **For files with content**: You MUST provide 'anchor_text' with EXACT text from outline to insert after\n"
        "- ⚠️ **CRITICAL FOR TOP-LEVEL SECTIONS (Synopsis, Notes, Characters)**: When adding to sections that already have content:\n"
        "  * DO NOT use the section heading (\"# Overall Synopsis\", \"# Notes\", \"# Characters\") as anchor_text!\n"
        "  * Using a section heading as anchor when content exists will INSERT BETWEEN the heading and existing content!\n"
        "  * This SPLITS THE SECTION and disrupts the document structure - NEVER do this!\n"
        "  * Instead: Find the LAST LINE of text in that section and use it as anchor_text\n"
        "  * OR: Use replace_range to modify the existing section content instead of inserting\n"
        "  * Example WRONG: '# Overall Synopsis\\n[INSERT HERE splits section]\\n[Existing synopsis]' ← WRONG!\n"
        "  * Example CORRECT: anchor_text='[last sentence of existing synopsis]' → inserts after existing content\n"
        "  * Example CORRECT: Use replace_range to update the entire synopsis section\n"
        "  * Only use section headings as anchors when the section is COMPLETELY EMPTY (no text below heading)\n"
        "- **CRITICAL FOR EMPTY FILES**: If the outline file is empty (only frontmatter), DO NOT provide 'anchor_text'\n"
        "  * Empty file = no content exists yet, so there's nothing to anchor to\n"
        "  * Omit the 'anchor_text' field entirely - the system will insert after frontmatter automatically\n"
        "  * Example for empty file: {\"op_type\": \"insert_after_heading\", \"text\": \"## Chapter 1\\n\\n[content]\"}\n"
        "- **CRITICAL FOR NEW CHAPTERS**: When adding a NEW chapter (e.g., \"Create Chapter 7\"):\n"
        "  * anchor_text MUST be the LAST LINE of the PREVIOUS chapter (Chapter 6 in this example)\n"
        "  * DO NOT use the chapter heading (\"## Chapter 6\") as anchor_text - this will insert BETWEEN the heading and content!\n"
        "  * Find the actual last line of text in Chapter 6 (could be a beat, summary sentence, etc.)\n"
        "  * Example: If Chapter 6 ends with \"- Fleet coordinates the rescue operation\", use that EXACT line as anchor_text\n"
        "  * This ensures Chapter 7 is inserted AFTER all of Chapter 6's content, not in the middle of it\n"
        "  * **CRITICAL - AVOID REPETITION**: When creating a NEW chapter, DO NOT repeat beats from the end of the previous chapter!\n"
        "    - The previous chapter's final beats are COMPLETE - they represent where that chapter ENDS\n"
        "    - Your new chapter should START fresh with NEW events that logically follow from where the previous chapter ended\n"
        "    - Example: If Chapter 6 ends with \"- Character arrives at the city\", Chapter 7 should NOT start with \"- Character arrives at the city\"\n"
        "    - Instead, Chapter 7 should start with NEW events that happen AFTER arrival (e.g., \"- Character explores the city streets\")\n"
        "    - Think of chapter transitions: each chapter should pick up the narrative thread naturally, not repeat the previous chapter's conclusion\n"
        "    - The new chapter's first beats should represent NEW plot developments, not a rehash of the previous chapter's ending\n"
        "- **YOUR GENERATED CONTENT MUST BE 100% NEW** - do not copy existing beats in your 'text' field!\n"
        "- **NEVER use anchor_text that references context headers** - Text like '=== CURRENT OUTLINE ===' or 'File: filename.md' does NOT exist in the file\n\n"
        "OUTPUT FORMAT - ManuscriptEdit JSON:\n"
        "{\n"
        '  "type": "ManuscriptEdit",\n'
        '  "target_filename": "filename.md",\n'
        '  "scope": "paragraph|chapter|multi_chapter",\n'
        '  "summary": "brief description (or questions if seeking clarification)",\n'
        '  "operations": [\n'
        "    {\n"
        '      "op_type": "replace_range|delete_range|insert_after_heading|insert_after",\n'
        '      "start": 0,\n'
        '      "end": 0,\n'
        '      "text": "content to insert/replace",\n'
        '      "original_text": "exact text from file (for replace/delete)",\n'
        '      "anchor_text": "exact line to insert after (for insert_after_heading)"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "**OUTPUT RULES**:\n"
        "- Return raw JSON only (no markdown fences, no explanatory text)\n"
        "- Always provide operations based on available information - work with what you know\n"
        "- If you need clarification, include it in the summary field AFTER describing the work you've done\n"
        "- Never return empty operations array unless the request is completely impossible to fulfill\n"
    )


# ============================================
# Subgraph Nodes
# ============================================

async def prepare_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare context: extract editor context, validate type: outline, handle clarification context"""
    try:
        logger.info("Preparing context for outline editing...")
        
        # Use shared utility for base context preparation
        context = await prepare_writing_context(
            state=state,
            doc_type="outline",
            default_filename="outline.md",
            content_key="outline",
            validate_type=True,  # Hard gate on type: outline
            clarification_key="pending_outline_clarification"
        )
        
        # Check for errors from shared utility
        if context.get("error"):
            return context
        
        # Add outline-specific clarification context handling
        shared_memory = state.get("shared_memory", {}) or {}
        previous_clarification = shared_memory.get("pending_outline_clarification")
        clarification_context = ""
        if previous_clarification:
            logger.info("Detected previous clarification request, including context")
            clarification_context = (
                "\n\n=== PREVIOUS CLARIFICATION REQUEST ===\n"
                f"Context: {previous_clarification.get('context', '')}\n"
                f"Questions Asked:\n"
            )
            for i, q in enumerate(previous_clarification.get('questions', []), 1):
                clarification_context += f"{i}. {q}\n"
            clarification_context += "\nThe user's response is in their latest message. Use this context to proceed with the outline development.\n"
        
        # Extract user request using shared utility
        current_request = extract_user_request(state)
        
        # Add outline-specific fields
        context.update({
            "clarification_context": clarification_context,
            "current_request": current_request.strip()
        })
        
        return context
        
    except Exception as e:
        logger.error(f"Failed to prepare context: {e}")
        return create_writing_error_response(
            str(e),
            "outline_editing_subgraph",
            state
        )


async def load_references_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load referenced files from outline frontmatter + assess quality"""
    try:
        logger.info("Loading referenced context files from outline frontmatter...")
        
        from orchestrator.tools.reference_file_loader import load_referenced_files
        
        active_editor = state.get("active_editor", {})
        user_id = state.get("user_id", "system")
        frontmatter = state.get("frontmatter", {})
        
        # Outline reference configuration - load directly from outline's frontmatter (no cascading)
        reference_config = {
            "style": ["style"],
            "rules": ["rules"],
            "characters": ["characters", "character_*"],  # Support both list and individual keys
            "series": ["series"]
        }
        
        # Use unified loader (no cascade_config - outline loads directly)
        result = await load_referenced_files(
            active_editor=active_editor,
            user_id=user_id,
            reference_config=reference_config,
            doc_type_filter="outline",
            cascade_config=None  # No cascading for outline
        )
        
        loaded_files = result.get("loaded_files", {})
        
        # Extract content from loaded files and assess quality
        style_body = None
        style_quality = 0.0
        style_warnings = []
        
        if loaded_files.get("style") and len(loaded_files["style"]) > 0:
            style_body = loaded_files["style"][0].get("content", "")
            if style_body:
                style_body = strip_frontmatter_block(style_body)
                style_quality, style_warnings = _assess_reference_quality(style_body, "style")
        
        rules_body = None
        rules_quality = 0.0
        rules_warnings = []
        
        if loaded_files.get("rules") and len(loaded_files["rules"]) > 0:
            rules_body = loaded_files["rules"][0].get("content", "")
            if rules_body:
                rules_body = strip_frontmatter_block(rules_body)
                rules_quality, rules_warnings = _assess_reference_quality(rules_body, "rules")
        
        characters_bodies = []
        characters_qualities = []
        characters_warnings = []
        
        if loaded_files.get("characters"):
            for char_file in loaded_files["characters"]:
                char_content = char_file.get("content", "")
                if char_content:
                    char_content = strip_frontmatter_block(char_content)
                    char_quality, char_warnings = _assess_reference_quality(char_content, "characters")
                    characters_bodies.append(char_content)
                    characters_qualities.append(char_quality)
                    characters_warnings.extend(char_warnings)
        
        series_body = None
        if loaded_files.get("series") and len(loaded_files["series"]) > 0:
            series_body = loaded_files["series"][0].get("content", "")
            if series_body:
                series_body = strip_frontmatter_block(series_body)
        
        # Calculate average character quality
        avg_character_quality = sum(characters_qualities) / len(characters_qualities) if characters_qualities else 0.0
        
        # Collect all warnings
        all_warnings = []
        if style_quality < 0.4 and style_body:
            all_warnings.append(f"Style guide quality is low ({style_quality:.0%})")
        all_warnings.extend(style_warnings)
        
        if rules_quality < 0.4 and rules_body:
            all_warnings.append(f"Rules quality is low ({rules_quality:.0%})")
        all_warnings.extend(rules_warnings)
        
        if avg_character_quality < 0.4 and characters_bodies:
            all_warnings.append(f"Character profiles quality is low ({avg_character_quality:.0%})")
        all_warnings.extend(characters_warnings)
        
        # Analyze mode based on available references
        current_request = state.get("current_request", "").lower()
        reference_quality = {
            "style": style_quality,
            "rules": rules_quality,
            "characters": avg_character_quality
        }
        
        # Detect available references (consider quality - low quality < 0.4 treated as not present)
        has_style = style_body is not None and len(style_body.strip()) > 50 and style_quality >= 0.4
        has_rules = rules_body is not None and len(rules_body.strip()) > 50 and rules_quality >= 0.4
        has_characters = len(characters_bodies) > 0 and avg_character_quality >= 0.4
        
        available_references = {
            "style": has_style,
            "rules": has_rules,
            "characters": has_characters
        }
        
        # Detect creative freedom keywords
        freehand_keywords = ["freehand", "creative freedom", "ignore references", 
                              "new direction", "fresh start", "brainstorm", "from scratch"]
        creative_freedom_requested = any(kw in current_request for kw in freehand_keywords)
        
        # Determine mode
        ref_count = sum(available_references.values())
        has_any_refs = ref_count > 0
        
        if creative_freedom_requested:
            generation_mode = "freehand"
            mode_guidance = "CREATIVE FREEDOM MODE - Full creative latitude. References available for optional consistency checks."
        elif has_any_refs and ref_count == 3:
            generation_mode = "fully_referenced"
            mode_guidance = "FULLY REFERENCED - Complete universe context available. Use as guidelines, user's request is primary."
        elif has_any_refs:
            generation_mode = "partial_references"
            refs_available = [k for k, v in available_references.items() if v]
            mode_guidance = f"PARTIAL REFERENCES - Available: {', '.join(refs_available)}. Fill gaps with creativity."
        else:
            generation_mode = "freehand"
            mode_guidance = "FREEHAND MODE - No references. Full creative freedom based on user's request."
        
        # Build reference summary
        ref_parts = []
        if has_style:
            ref_parts.append(f"Style guide (quality: {style_quality:.0%})")
        if has_rules:
            ref_parts.append(f"Universe rules (quality: {rules_quality:.0%})")
        if has_characters:
            ref_parts.append(f"{len(characters_bodies)} character profile(s) (quality: {avg_character_quality:.0%})")
        
        if ref_parts:
            reference_summary = f"Available: {', '.join(ref_parts)}"
        else:
            reference_summary = "No references available - freehand mode"
        
        if all_warnings:
            reference_summary += f"\nWarnings: {'; '.join(all_warnings[:3])}"  # Limit to 3 warnings
        
        # Analyze outline structure
        body_only = state.get("body_only", "")
        
        # Detect existing sections
        has_synopsis = bool(re.search(r"^#\s+(Overall\s+)?Synopsis\s*$", body_only, re.MULTILINE | re.IGNORECASE))
        has_notes = bool(re.search(r"^#\s+Notes\s*$", body_only, re.MULTILINE | re.IGNORECASE))
        has_characters_section = bool(re.search(r"^#\s+Characters?\s*$", body_only, re.MULTILINE | re.IGNORECASE))
        has_outline = bool(re.search(r"^#\s+Outline\s*$", body_only, re.MULTILINE | re.IGNORECASE))
        
        # Count chapters
        chapter_matches = list(CHAPTER_PATTERN.finditer(body_only))
        chapter_count = len(chapter_matches)
        
        # Assess completeness
        sections_present = sum([has_synopsis, has_notes, has_characters_section, has_outline])
        completeness_score = sections_present / 4.0 if sections_present > 0 else 0.0
        
        # Detect structural issues
        structure_warnings = []
        if chapter_count == 0 and has_outline:
            structure_warnings.append("Outline section exists but no chapters defined")
        if not has_synopsis and chapter_count > 0:
            structure_warnings.append("Chapters exist without Overall Synopsis")
        if has_characters_section and not re.search(r"Protagonist|Antagonist|Supporting", body_only, re.IGNORECASE):
            structure_warnings.append("Characters section missing protagonist/antagonist designation")
        if chapter_count > 0:
            # Check chapter numbering
            chapter_nums = []
            for m in chapter_matches:
                try:
                    chapter_nums.append(int(m.group(1)))
                except Exception:
                    pass
            if chapter_nums:
                expected = list(range(1, len(chapter_nums) + 1))
                if chapter_nums != expected:
                    structure_warnings.append(f"Chapter numbering is non-sequential: {chapter_nums}")
        
        # Generate structure-specific guidance with section availability
        section_list = []
        if has_synopsis:
            section_list.append("Overall Synopsis (can edit)")
        if has_notes:
            section_list.append("Notes (can edit)")
        if has_characters_section:
            section_list.append("Characters (can edit)")
        if has_outline:
            section_list.append("Outline")
        if chapter_count > 0:
            section_list.append(f"{chapter_count} chapter(s) (can edit)")
        
        sections_available = ", ".join(section_list) if section_list else "No sections yet"
        
        if completeness_score < 0.25:
            structure_guidance = f"Outline {completeness_score:.0%} complete ({sections_present}/4 sections). Available sections: {sections_available}. Build full structure or edit existing sections."
        elif completeness_score < 0.75:
            structure_guidance = f"Outline {completeness_score:.0%} complete. Available sections: {sections_available}. Continue developing or edit existing sections."
        elif structure_warnings:
            structure_guidance = f"Available sections: {sections_available}. Issues: {'; '.join(structure_warnings[:2])}"  # Limit to 2
        else:
            structure_guidance = f"Structure complete. Available sections: {sections_available}. You can edit any existing section."
        
        logger.info(f"Outline generation mode: {generation_mode}")
        logger.info(f"Available references: {', '.join([k for k, v in available_references.items() if v]) or 'none'}")
        logger.info(f"Outline completeness: {completeness_score:.0%} ({sections_present}/4 sections)")
        logger.info(f"Chapter count: {chapter_count}")
        
        # Load tracked items from frontmatter
        from orchestrator.utils.fiction_utilities import get_tracked_items
        tracked_items = get_tracked_items(frontmatter)
        
        return {
            "rules_body": rules_body,
            "style_body": style_body,
            "characters_bodies": characters_bodies,
            "series_body": series_body,
            "reference_quality": reference_quality,
            "reference_warnings": all_warnings,
            "available_references": available_references,
            "generation_mode": generation_mode,
            "reference_summary": reference_summary,
            "mode_guidance": mode_guidance,
            "outline_completeness": completeness_score,
            "chapter_count": chapter_count,
            "structure_warnings": structure_warnings,
            "structure_guidance": structure_guidance,
            "has_synopsis": has_synopsis,
            "has_notes": has_notes,
            "has_characters": has_characters_section,
            "has_outline_section": has_outline,
            "tracked_items": tracked_items,
            # ✅ CRITICAL: Preserve state for subsequent nodes
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "frontmatter": state.get("frontmatter", {}),  # CRITICAL: Preserve frontmatter for temperature access
            # ✅ Preserve outline context
            "active_editor": state.get("active_editor", {}),
            "outline": state.get("outline", ""),
            "filename": state.get("filename", ""),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "body_only": state.get("body_only", ""),
            "clarification_context": state.get("clarification_context", ""),
            "current_request": state.get("current_request", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to load references: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Load tracked items even on error (from frontmatter)
        from orchestrator.utils.fiction_utilities import get_tracked_items
        tracked_items = get_tracked_items(state.get("frontmatter", {}))
        return {
            "rules_body": None,
            "style_body": None,
            "characters_bodies": [],
            "series_body": None,
            "reference_quality": {},
            "reference_warnings": [],
            "available_references": {},
            "generation_mode": "freehand",
            "reference_summary": "Error loading references - defaulting to freehand",
            "mode_guidance": "Freehand mode - proceed with creative freedom.",
            "outline_completeness": 0.0,
            "chapter_count": 0,
            "structure_warnings": [],
            "structure_guidance": "Unable to analyze structure - proceed with caution.",
            "has_synopsis": False,
            "has_notes": False,
            "has_characters": False,
            "has_outline_section": False,
            "tracked_items": tracked_items,
            "error": str(e),
            # ✅ CRITICAL: Preserve state even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "frontmatter": state.get("frontmatter", {}),  # CRITICAL: Preserve frontmatter for temperature access
            # ✅ Preserve outline context
            "active_editor": state.get("active_editor", {}),
            "outline": state.get("outline", ""),
            "filename": state.get("filename", ""),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "body_only": state.get("body_only", ""),
            "clarification_context": state.get("clarification_context", ""),
            "current_request": state.get("current_request", "")
        }


# ============================================
# Node: Detect Request Type
# ============================================

async def detect_request_type_node(
    state: Dict[str, Any],
    llm_factory: Callable
) -> Dict[str, Any]:
    """Detect if user request is a question or an edit request"""
    try:
        logger.info("Detecting request type (question vs edit request)...")
        
        current_request = state.get("current_request", "")
        if not current_request:
            logger.warning("No current request found - defaulting to edit_request")
            return {
                "request_type": "edit_request",
                # ✅ CRITICAL: Preserve all state
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                # ✅ Domain-specific: Preserve outline context
                "outline": state.get("outline", ""),
                "filename": state.get("filename", ""),
                "frontmatter": state.get("frontmatter", {}),
                "body_only": state.get("body_only", ""),
                "tracked_items": state.get("tracked_items", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "active_editor": state.get("active_editor", {}),
                "current_request": state.get("current_request", ""),
                "rules_body": state.get("rules_body"),
                "style_body": state.get("style_body"),
                "characters_bodies": state.get("characters_bodies", []),
                "generation_mode": state.get("generation_mode", ""),
                "available_references": state.get("available_references", {}),
                "reference_summary": state.get("reference_summary", ""),
                "mode_guidance": state.get("mode_guidance", ""),
                "outline_completeness": state.get("outline_completeness", 0.0),
                "chapter_count": state.get("chapter_count", 0),
                "structure_guidance": state.get("structure_guidance", "")
            }
        
        body_only = state.get("body_only", "")
        rules_body = state.get("rules_body")
        style_body = state.get("style_body")
        characters_bodies = state.get("characters_bodies", [])
        
        # Build simple prompt for LLM to determine intent
        prompt = f"""Analyze the user's request and determine if it's a QUESTION or an EDIT REQUEST.

**USER REQUEST**: {current_request}

**CONTEXT**:
- Current outline: {body_only[:500] if body_only else "Empty outline"}
- Has rules reference: {bool(rules_body)}
- Has style reference: {bool(style_body)}
- Has {len(characters_bodies)} character reference(s)

**INTENT DETECTION**:
- QUESTIONS (including pure questions and conditional edits): User is asking a question - may or may not want edits
  - Pure questions: "Do you see our characters?", "What rules are loaded?", "How many chapters do we have?"
  - Conditional edits: "Do we have a synopsis? Add one if not", "How many chapters? Add 3 more if less than 10", "Is Chapter 2 complete? Finish it if not"
  - Questions often start with: "Do you", "What", "Can you", "Are there", "How many", "Show me", "Is", "Does", "Are we"
  - **Key insight**: Questions can be answered, and IF edits are needed based on the answer, they can be made
  - Route ALL questions to edit path - LLM can decide if edits are needed
  
- EDIT REQUESTS: User wants to create, modify, or generate content - NO question asked
  - Examples: "Add a chapter", "Create an outline", "Update the synopsis", "Generate outline for chapter 2"
  - Edit requests are action-oriented: "add", "create", "update", "generate", "change", "replace"
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
        llm = llm_factory(temperature=0.1, state=state)  # Low temperature for consistent classification
        
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content="You are an intent classifier. Analyze user requests and determine if they are questions or edit requests. Return only valid JSON."),
            HumanMessage(content=prompt)
        ]
        
        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)
        content = _unwrap_json_response(content)
        
        # Check for empty response
        if not content or not content.strip():
            logger.error("Request type detection: LLM returned empty response, defaulting to edit_request")
            return {
                "request_type": "edit_request",
                # ✅ CRITICAL 5: Always preserve
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                # Preserve outline state
                "body_only": state.get("body_only", ""),
                "filename": state.get("filename", ""),
                "frontmatter": state.get("frontmatter", {}),
                "tracked_items": state.get("tracked_items", {}),
                "active_editor": state.get("active_editor", {})
            }
        
        try:
            result = json.loads(content)
            request_type = result.get("request_type", "edit_request")
            confidence = result.get("confidence", 0.5)
            reasoning = result.get("reasoning", "")
            
            logger.info(f"Request type detected: {request_type} (confidence: {confidence:.0%}, reasoning: {reasoning})")
            
            # Default to edit_request if confidence is low
            if confidence < 0.6:
                logger.warning(f"Low confidence ({confidence:.0%}) - defaulting to edit_request")
                request_type = "edit_request"
            
            return {
                "request_type": request_type,
                # ✅ CRITICAL 5: Always preserve
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                # ✅ Domain-specific: Preserve outline context
                "outline": state.get("outline", ""),
                "filename": state.get("filename", ""),
                "frontmatter": state.get("frontmatter", {}),
                "body_only": state.get("body_only", ""),
                "tracked_items": state.get("tracked_items", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "active_editor": state.get("active_editor", {}),
                "current_request": state.get("current_request", ""),
                # ✅ Reference context
                "rules_body": state.get("rules_body"),
                "style_body": state.get("style_body"),
                "characters_bodies": state.get("characters_bodies", []),
                # ✅ Analysis results from previous nodes
                "generation_mode": state.get("generation_mode", ""),
                "available_references": state.get("available_references", {}),
                "reference_summary": state.get("reference_summary", ""),
                "mode_guidance": state.get("mode_guidance", ""),
                "outline_completeness": state.get("outline_completeness", 0.0),
                "chapter_count": state.get("chapter_count", 0),
                "structure_guidance": state.get("structure_guidance", "")
            }
            
        except Exception as parse_error:
            logger.error(f"Failed to parse request type detection: {parse_error}")
            logger.warning("Defaulting to edit_request due to parse error")
            return {
                "request_type": "edit_request",
                # ✅ CRITICAL: Preserve even on error!
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "outline": state.get("outline", ""),
                "filename": state.get("filename", ""),
                "frontmatter": state.get("frontmatter", {}),
                "body_only": state.get("body_only", ""),
                "tracked_items": state.get("tracked_items", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "active_editor": state.get("active_editor", {}),
                "current_request": state.get("current_request", ""),
                "rules_body": state.get("rules_body"),
                "style_body": state.get("style_body"),
                "characters_bodies": state.get("characters_bodies", []),
                "generation_mode": state.get("generation_mode", ""),
                "available_references": state.get("available_references", {}),
                "reference_summary": state.get("reference_summary", ""),
                "mode_guidance": state.get("mode_guidance", ""),
                "outline_completeness": state.get("outline_completeness", 0.0),
                "chapter_count": state.get("chapter_count", 0),
                "structure_guidance": state.get("structure_guidance", "")
            }
            
    except Exception as e:
        logger.error(f"Failed to detect request type: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Default to edit_request on error
        return {
            "request_type": "edit_request",
            # ✅ CRITICAL: Preserve even on error!
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "outline": state.get("outline", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "body_only": state.get("body_only", ""),
            "tracked_items": state.get("tracked_items", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "active_editor": state.get("active_editor", {}),
            "current_request": state.get("current_request", ""),
            "rules_body": state.get("rules_body"),
            "style_body": state.get("style_body"),
            "characters_bodies": state.get("characters_bodies", []),
            "generation_mode": state.get("generation_mode", ""),
            "available_references": state.get("available_references", {}),
            "reference_summary": state.get("reference_summary", ""),
            "mode_guidance": state.get("mode_guidance", ""),
            "outline_completeness": state.get("outline_completeness", 0.0),
            "chapter_count": state.get("chapter_count", 0),
            "structure_guidance": state.get("structure_guidance", "")
        }


# ============================================
# Node: Generate Status Summary
# ============================================

async def generate_status_summary_node(
    state: Dict[str, Any],
    llm_factory: Callable,
    get_datetime_context: Callable
) -> Dict[str, Any]:
    """
    Generate or update status summaries for chapters based on tracked items.
    
    This node:
    1. Checks if tracked_items exist in frontmatter
    2. Identifies chapters needing status updates (new chapters, edits mentioning tracked items, explicit requests)
    3. For each chapter, generates status block using LLM analysis
    4. Stores status updates in chapter_status_updates for later insertion
    """
    try:
        tracked_items = state.get("tracked_items", {})
        body_only = state.get("body_only", "")
        current_request = state.get("current_request", "").lower()
        
        # Check if tracked items are configured
        has_tracked_items = bool(tracked_items and any(
            items for items in tracked_items.values() if isinstance(items, list) and items
        ))
        
        if not has_tracked_items:
            logger.debug("No tracked items configured - skipping status generation")
            return {
                "chapter_status_updates": {},
                # ✅ CRITICAL: Preserve all state
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "outline": state.get("outline", ""),
                "filename": state.get("filename", ""),
                "frontmatter": state.get("frontmatter", {}),
                "body_only": state.get("body_only", ""),
                "tracked_items": tracked_items,
                "current_request": state.get("current_request", ""),
            }
        
        # Determine which chapters need status updates
        from orchestrator.utils.fiction_utilities import (
            find_chapter_ranges,
            extract_chapter_outline,
            extract_status_block
        )
        
        chapter_ranges = find_chapter_ranges(body_only) if body_only else []
        existing_chapters = {ch.chapter_number for ch in chapter_ranges if ch.chapter_number is not None}
        
        # Check for explicit status update request
        explicit_status_request = any(phrase in current_request for phrase in [
            "update status", "regenerate status", "refresh status", "status summary",
            "update status summaries", "regenerate status summaries"
        ])
        
        # Extract chapter number from request (if creating/editing specific chapter)
        from orchestrator.utils.fiction_utilities import extract_chapter_number_from_request
        requested_chapter = extract_chapter_number_from_request(current_request)
        
        chapters_to_update = {}
        
        # Case 1: Explicit status update request - update all chapters (including Chapter 1 if it exists)
        if explicit_status_request:
            logger.info("Explicit status update requested - updating all chapters")
            for chapter_num in sorted(existing_chapters):
                if chapter_num:
                    if chapter_num == 1:
                        # Chapter 1 status only if explicitly requested (for manual filling)
                        chapters_to_update[chapter_num] = {"reason": "explicit_request_chapter_1"}
                    else:
                        chapters_to_update[chapter_num] = {"reason": "explicit_request"}
        
        # Case 2: New chapter being created
        elif requested_chapter and requested_chapter not in existing_chapters:
            logger.info(f"New chapter {requested_chapter} being created - will generate status")
            if requested_chapter > 1:  # Skip Chapter 1 (no previous chapter to track from)
                chapters_to_update[requested_chapter] = {"reason": "new_chapter"}
            elif requested_chapter == 1 and explicit_status_request:
                # Allow Chapter 1 status only if explicitly requested (for manual filling or series continuity)
                chapters_to_update[requested_chapter] = {"reason": "explicit_request_chapter_1"}
        
        # Case 3: Editing an existing chapter 2+ when tracked_items are configured - always update status
        # so relationships and characters stay in sync (not only when request mentions a tracked name)
        elif requested_chapter and requested_chapter > 1 and requested_chapter in existing_chapters:
            logger.info(f"Editing chapter {requested_chapter} with tracked_items - will update status")
            chapters_to_update[requested_chapter] = {"reason": "tracked_items_configured"}
        
        # If no chapters to update, skip
        if not chapters_to_update:
            logger.debug("No chapters need status updates")
            return {
                "chapter_status_updates": {},
                # ✅ CRITICAL: Preserve all state
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "outline": state.get("outline", ""),
                "filename": state.get("filename", ""),
                "frontmatter": state.get("frontmatter", {}),
                "body_only": state.get("body_only", ""),
                "tracked_items": tracked_items,
                "current_request": state.get("current_request", ""),
            }
        
        # Generate status for each chapter
        llm = llm_factory(temperature=0.3, state=state)  # Lower temperature for consistent status generation
        status_updates = {}
        
        for chapter_num in sorted(chapters_to_update.keys()):
            try:
                # Get current chapter content
                chapter_outline = extract_chapter_outline(body_only, chapter_num)
                if not chapter_outline:
                    logger.warning(f"Could not extract outline for Chapter {chapter_num} - skipping status")
                    continue
                
                # Get previous chapter's status and content (if exists)
                previous_status = None
                previous_chapter_content = None
                if chapter_num > 1:
                    prev_chapter_outline = extract_chapter_outline(body_only, chapter_num - 1)
                    if prev_chapter_outline:
                        previous_status = extract_status_block(prev_chapter_outline)
                        previous_chapter_content = prev_chapter_outline
                elif chapter_num == 1:
                    # For Chapter 1, note that there's no previous chapter
                    previous_status = None
                    previous_chapter_content = None
                
                # Build prompt
                prompt = f"""Analyze the chapter content and generate a status summary for tracked items.

**CHAPTER NUMBER**: {chapter_num}

**TRACKED ITEMS** (from outline frontmatter):
"""
                for category, items in tracked_items.items():
                    if items:
                        prompt += f"- {category.capitalize()}: {', '.join(items)}\n"
                
                if previous_chapter_content:
                    prompt += f"\n**PREVIOUS CHAPTER (Chapter {chapter_num - 1}) - WHAT ACTUALLY HAPPENED**:\n{previous_chapter_content}\n"
                    prompt += f"\n**NOTE**: The status for Chapter {chapter_num} should reflect the END STATE of Chapter {chapter_num - 1}.\n"
                    prompt += f"Read the beats above to understand where things stood when Chapter {chapter_num - 1} ended.\n"
                    if previous_status:
                        prompt += f"\n**PREVIOUS CHAPTER'S STATUS BLOCK** (Chapter {chapter_num - 1}):\n{previous_status}\n"
                        prompt += f"(This shows what the beginning state was for Chapter {chapter_num - 1})\n"
                elif chapter_num == 1:
                    prompt += f"\n**PREVIOUS CHAPTER**: None (this is Chapter 1 - initial state)\n"
                    prompt += f"**NOTE**: For Chapter 1, status should describe the initial state at story start.\n"
                    prompt += f"This is useful for series continuity or stories beginning mid-action.\n"
                else:
                    prompt += f"\n**PREVIOUS CHAPTER**: None (Chapter {chapter_num - 1} doesn't exist yet)\n"
                
                prompt += f"\n**CURRENT CHAPTER CONTENT** (Chapter {chapter_num}):\n{chapter_outline}\n"
                prompt += f"\n**CRITICAL: CURRENT CHAPTER CONTENT IS FOR CONTEXT ONLY**\n"
                prompt += f"- The content above shows what WILL happen in Chapter {chapter_num}\n"
                prompt += f"- These events have NOT happened yet - they will happen IN this chapter\n"
                prompt += f"- **ABSOLUTE PROHIBITION**: DO NOT include ANY events from the current chapter content in the status!\n"
                prompt += f"- Use the current chapter content ONLY to identify which tracked items are relevant\n"
                prompt += f"- The status should reflect the state BEFORE these events occur\n\n"
                
                if previous_chapter_content:
                    prompt += f"**HOW TO GENERATE CORRECT STATUS**:\n"
                    prompt += f"1. Read the PREVIOUS CHAPTER (Chapter {chapter_num - 1}) beats to see what happened\n"
                    prompt += f"2. Identify where each tracked item ended up at the END of Chapter {chapter_num - 1}\n"
                    prompt += f"3. That END state from Chapter {chapter_num - 1} becomes the BEGINNING state for Chapter {chapter_num}\n"
                    prompt += f"4. Look at CURRENT CHAPTER only to see which items will be relevant (don't use its events!)\n"
                    prompt += f"5. Write status describing where things stood when Chapter {chapter_num} STARTS\n\n"
                else:
                    prompt += f"**HOW TO GENERATE CORRECT STATUS (No Previous Chapter)**:\n"
                    prompt += f"1. Since there's no Chapter {chapter_num - 1}, infer the initial state from context\n"
                    prompt += f"2. DO NOT use events from Chapter {chapter_num}'s beats - those haven't happened yet!\n"
                    prompt += f"3. Describe the state BEFORE Chapter {chapter_num}'s events begin\n\n"
                
                prompt += """
**CRITICAL: STATUS REPRESENTS BEGINNING STATE**
- The status block describes the state of tracked items AT THE BEGINNING of this chapter
- It reflects what happened in PREVIOUS chapters, not what happens IN this chapter
- The status is a snapshot of where things stand when this chapter STARTS (before any events in this chapter)

**CONCRETE EXAMPLE**:
- Chapter 3 beats say: "Benedict is murdered by Archibald at the warehouse"
- Chapter 4 status should say: "Benedict: Murdered by Archibald at warehouse in Chapter 3"
- Chapter 4 status should NOT say: "Benedict: Alive at warehouse" (that was Chapter 3's beginning state)
- Chapter 4 status should NOT include anything from Chapter 4's beats!

**WRONG EXAMPLE** (DO NOT DO THIS):
- Chapter 2 beats say: "Jack drinks drugged brandy and falls into deep sleep"
- Chapter 2 status says: "Jack: In bedroom after drinking drugged brandy, now in deep sleep" ❌ WRONG!
- This describes the END of Chapter 2, not the BEGINNING!

**CORRECT EXAMPLE**:
- Chapter 1 beats end with: "Jack and Reya arrive at castle, are greeted by Dracula"  
- Chapter 2 status should say: "Jack: Arrived at castle in Chapter 1, currently in guest quarters" ✅ CORRECT!
- This describes where Jack is at the START of Chapter 2, based on Chapter 1's ending

**TASK**: Generate a status summary for tracked items that are:
1. Relevant to this chapter (will be mentioned or affected - use current chapter content to identify these)
2. Have changed since the previous chapter (based on PREVIOUS chapter's events ONLY)
3. Need status updates to show beginning state (state BEFORE current chapter events)
4. **CARRY FORWARD**: If a tracked item appeared in previous chapter's status and is still active/relevant, include it even if unchanged
   - This ensures the fiction agent always has context for tracked items
   - Relationships especially should be carried forward if they're ongoing
   - Example: If "Clarissa-Benedict" was "Allies investigating together" in Chapter 2, and they're still working together in Chapter 3, include it as "Allies investigating together (ongoing)"

**STATUS FORMAT GUIDELINES**:
- For characters: location at chapter start, state (alive/dead/injured), where they ended up in previous chapter
- For locations: condition at chapter start, occupancy, what happened to them in previous chapter
- For items: location at chapter start, condition, ownership, where they ended up in previous chapter
- For relationships: current state at chapter start, how they changed in previous chapter

**OUTPUT FORMAT**: Return ONLY valid JSON:
{
  "status_items": [
    {"item": "Character/Location/Item/Relationship name", "status": "Brief status description"},
    ...
  ]
}

**RULES**:
- Include tracked items that are: (1) relevant to this chapter, OR (2) have changed, OR (3) should be carried forward from previous chapter
- **CARRY FORWARD RULE**: If a tracked item was in the previous chapter's status and is still active/relevant, include it even if unchanged
  - Relationships should be carried forward if they're ongoing (e.g., "Clarissa-Benedict: Allies (ongoing)")
  - Characters should be carried forward if they're still part of the story (e.g., "Clarissa: At home (ongoing)")
  - Locations should be carried forward if they're still relevant (e.g., "Warehouse: Under investigation (ongoing)")
  - Items should be carried forward if they're still in play (e.g., "Artifact: In Clarissa's possession (ongoing)")
- Keep status descriptions brief (one line per item)
- Reference previous chapter numbers when relevant (e.g., "Went home in chapter 2")
- Use "(ongoing)" or similar notation when carrying forward unchanged status
- If no tracked items are relevant or need carrying forward, return empty array: {"status_items": []}
- Be specific about what changed (e.g., "Murdered by Archibald in Chapter 3", not just "dead")
- **ABSOLUTE PROHIBITION**: DO NOT include any events from the current chapter content in the status!
- Status describes BEGINNING state, based ONLY on what happened in PREVIOUS chapters
- Current chapter content shows what WILL happen - those events have NOT occurred yet
- Status = state BEFORE current chapter events, based on previous chapter events
"""
                
                # Call LLM
                datetime_context = get_datetime_context(state)
                messages = [
                    SystemMessage(content="You are a status tracking assistant. Analyze chapter content and generate status summaries for tracked items. Return only valid JSON."),
                    SystemMessage(content=datetime_context),
                    HumanMessage(content=prompt)
                ]
                
                response = await llm.ainvoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                content = _unwrap_json_response(content)
                
                # Check for empty response
                if not content or not content.strip():
                    logger.warning("Status detection: LLM returned empty response, using empty status")
                    status_items = []
                else:
                    # Parse response
                    try:
                        result = json.loads(content)
                        status_items = result.get("status_items", [])
                    except json.JSONDecodeError as e:
                        logger.error(f"Status detection: Failed to parse JSON: {e}")
                        logger.error(f"Raw content (first 500 chars): {content[:500]}")
                        status_items = []
                
                if status_items:
                    # Format status block
                    from orchestrator.utils.fiction_utilities import format_status_block
                    status_block = format_status_block(status_items)
                    
                    if status_block:
                        status_updates[chapter_num] = {
                            "status_block": status_block,
                            "reason": chapters_to_update[chapter_num]["reason"]
                        }
                        logger.info(f"Generated status block for Chapter {chapter_num} ({len(status_items)} items)")
                
            except Exception as e:
                logger.error(f"Failed to generate status for Chapter {chapter_num}: {e}")
                continue
        
        return {
            "chapter_status_updates": status_updates,
            # ✅ CRITICAL: Preserve all state
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "outline": state.get("outline", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "body_only": state.get("body_only", ""),
            "tracked_items": tracked_items,
            "current_request": state.get("current_request", ""),
        }
        
    except Exception as e:
        logger.error(f"Failed to generate status summaries: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "chapter_status_updates": {},
            "error": str(e),
            # ✅ CRITICAL: Preserve all state even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "outline": state.get("outline", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "body_only": state.get("body_only", ""),
            "tracked_items": state.get("tracked_items", {}),
            "current_request": state.get("current_request", ""),
        }


# ============================================
# Generate Edit Plan Node
# ============================================

async def generate_edit_plan_node(
    state: Dict[str, Any],
    llm_factory: Callable,
    get_datetime_context: Callable
) -> Dict[str, Any]:
    """Generate edit plan using LLM"""
    try:
        logger.info("Generating outline edit plan...")
        
        outline = state.get("outline", "")
        filename = state.get("filename", "outline.md")
        body_only = state.get("body_only", "")
        current_request = state.get("current_request", "")
        clarification_context = state.get("clarification_context", "")
        
        rules_body = state.get("rules_body")
        style_body = state.get("style_body")
        characters_bodies = state.get("characters_bodies", [])
        
        selection_start = state.get("selection_start", -1)
        selection_end = state.get("selection_end", -1)
        
        # Get mode and structure context
        mode_guidance = state.get("mode_guidance", "")
        structure_guidance = state.get("structure_guidance", "")
        reference_summary = state.get("reference_summary", "")
        
        # Build dynamic system prompt
        system_prompt = _build_system_prompt()
        
        # Build context message - simple and focused
        context_parts = []
        
        # User request will be separated into request_with_instructions below
        if not current_request:
            logger.error("current_request is empty - user's request will not be sent to LLM")
        
        # Current outline - detect chapter context from cursor position
        is_empty_file = not body_only.strip()
        context_parts.append("=== CURRENT OUTLINE ===\n")
        context_parts.append(f"File: {filename}\n")
        if is_empty_file:
            context_parts.append("\n⚠️ EMPTY FILE DETECTED: This file contains only frontmatter (no content yet)\n\n")
        else:
            # Detect which chapter the cursor is in (if cursor is valid)
            cursor_offset = state.get("cursor_offset", -1)
            outline = state.get("outline", "")
            current_chapter_text = None
            current_chapter_number = None
            prev_chapter_text = None
            prev_chapter_number = None
            
            # Find all chapter ranges (we'll use this for both cursor detection and display)
            chapter_ranges = find_chapter_ranges(body_only)
            
            if cursor_offset >= 0 and body_only and outline and chapter_ranges:
                # Adjust cursor offset for frontmatter (cursor is relative to full outline)
                fm_end_idx = _frontmatter_end_index(outline)
                cursor_in_body = cursor_offset - fm_end_idx
                
                # Only proceed if cursor is actually in the body (not in frontmatter)
                if cursor_in_body >= 0:
                    # Find which chapter contains the cursor
                    for i, chapter_range in enumerate(chapter_ranges):
                        if chapter_range.start <= cursor_in_body < chapter_range.end:
                            # Cursor is in this chapter
                            current_chapter_text = body_only[chapter_range.start:chapter_range.end].strip()
                            current_chapter_number = chapter_range.chapter_number
                            
                            # Get previous chapter if it exists
                            if i > 0:
                                prev_range = chapter_ranges[i - 1]
                                prev_chapter_text = body_only[prev_range.start:prev_range.end].strip()
                                prev_chapter_number = prev_range.chapter_number
                            break
            
            # Show ALL chapters with clear delineation
            if chapter_ranges:
                if current_chapter_text and current_chapter_number is not None:
                    context_parts.append(f"\n**YOU ARE WORKING IN CHAPTER {current_chapter_number}**\n")
                    context_parts.append("(Cursor position detected - showing all chapters with clear progression)\n\n")
                else:
                    context_parts.append("\n**ALL CHAPTERS IN OUTLINE** (showing complete story progression):\n\n")
                
                # Show ALL chapters with clear section markers
                for i, chapter_range in enumerate(chapter_ranges):
                    chapter_text = body_only[chapter_range.start:chapter_range.end].strip()
                    chapter_num = chapter_range.chapter_number
                    
                    # Count beats in this chapter
                    beat_count = count_beats_in_chapter(chapter_text)
                    
                    # Determine section label based on position relative to current chapter
                    if current_chapter_number is not None:
                        if chapter_num == current_chapter_number:
                            section_label = f"=== CHAPTER {chapter_num} (CURRENT - WHERE YOU ARE WORKING) ==="
                        elif chapter_num and chapter_num < current_chapter_number:
                            section_label = f"=== CHAPTER {chapter_num} (PREVIOUS - ALREADY COMPLETED) ==="
                        elif chapter_num and chapter_num > current_chapter_number:
                            section_label = f"=== CHAPTER {chapter_num} (FUTURE - NOT YET WRITTEN) ==="
                        else:
                            section_label = f"=== CHAPTER {chapter_num} ==="
                    else:
                        # No cursor detected, just show chapter number
                        section_label = f"=== CHAPTER {chapter_num} ==="
                    
                    context_parts.append(f"{section_label}\n")
                    context_parts.append(f"**Beat count: {beat_count}/100**\n")
                    if beat_count >= 100:
                        context_parts.append(f"⚠️ **WARNING**: Chapter {chapter_num} is at the 100-beat limit! Any new beats require removing existing ones.\n")
                    elif beat_count >= 95:
                        context_parts.append(f"⚠️ **NOTE**: Chapter {chapter_num} is approaching the 100-beat limit ({beat_count} beats). Consider pruning less important beats if adding more.\n")
                    context_parts.append(f"{chapter_text}\n\n")
                    
                    # Add transition guidance after each chapter (except the last)
                    if i < len(chapter_ranges) - 1:
                        next_chapter_num = chapter_ranges[i + 1].chapter_number
                        if next_chapter_num:
                            context_parts.append(f"--- END OF CHAPTER {chapter_num} → BEGINNING OF CHAPTER {next_chapter_num} ---\n\n")
                
                # Add critical transition reminder
                if prev_chapter_text and prev_chapter_number is not None:
                    context_parts.append(f"\n**CRITICAL TRANSITION REMINDER**:\n")
                    context_parts.append(f"- Chapter {prev_chapter_number} ENDED with the beats shown above.\n")
                    context_parts.append(f"- Chapter {current_chapter_number} (where you are working) should start with NEW events that happen AFTER Chapter {prev_chapter_number}'s conclusion.\n")
                    context_parts.append(f"- Do NOT repeat the ending beats from Chapter {prev_chapter_number} in Chapter {current_chapter_number}!\n")
                    context_parts.append(f"- Each chapter should advance the plot forward, not rehash previous chapter endings.\n\n")
            else:
                # No chapters found, show full outline
                context_parts.append("\n" + body_only + "\n\n")
        
        # References (if present)
        if rules_body:
            context_parts.append("=== UNIVERSE RULES ===\n")
            context_parts.append(f"{rules_body}\n\n")
        
        if style_body:
            context_parts.append("=== STYLE GUIDE ===\n")
            context_parts.append(f"{style_body}\n\n")
        
        if characters_bodies:
            context_parts.append("=== CHARACTER PROFILES ===\n")
            context_parts.append("**NOTE**: Character details (descriptions, backstories, traits) belong in character profile files.\n")
            context_parts.append("The outline should only reference characters briefly (name + role), not include full character information.\n\n")
            context_parts.append("".join([f"{b}\n---\n" for b in characters_bodies]))
            context_parts.append("\n")
        
        if clarification_context:
            context_parts.append(clarification_context)
        
        # Add "WORK FIRST" guidance and cross-reference instructions (like character agent)
        context_parts.append(
            "\n=== USER REQUEST: ANALYZE AND RESPOND APPROPRIATELY ===\n"
            "Analyze the user's request to determine if it requires edits or just an answer.\n\n"
            "**YOUR APPROACH**:\n"
            "1. **QUESTIONS THAT DON'T REQUIRE EDITS**:\n"
            "   - If the user is asking for information, analysis, lists, or recommendations\n"
            "   - AND you can answer without modifying the outline\n"
            "   - THEN return scope=\"paragraph\" with 0 operations and your complete answer in the summary field\n"
            "   - Examples: \"What unresolved plot points?\", \"Give me a list of...\", \"Analyze the structure\"\n"
            "2. **EDIT REQUESTS**:\n"
            "   - If the user wants to add, change, or modify content\n"
            "   - THEN generate operations to make those edits\n"
            "3. **FOR EMPTY FILES**: If the outline is empty, ASK QUESTIONS FIRST before creating content\n"
            "   - Don't create the entire outline structure at once\n"
            "   - Ask about story basics: genre, main characters, key plot points, chapter count\n"
            "   - Build incrementally - create one section at a time based on user responses\n"
            "4. **FOR FILES WITH CONTENT**: Make edits based on the request and available context (outline file, rules, style, characters)\n"
            "5. **USE INFERENCE**: Make reasonable inferences from the request - but ask if starting from scratch\n"
            "6. **ASK ALONG THE WAY**: If you need specific details, include questions in the summary AFTER describing the work you've done\n"
            "7. **CHARACTER INFORMATION**: Keep character details in character profiles, not in the outline!\n"
            "   - Outline should only have brief character references (name + role)\n"
            "   - Do NOT copy character descriptions, backstories, or traits into the outline\n\n"
            "CRITICAL: CHECK FOR DUPLICATES FIRST\n"
            "Before adding ANY new content:\n"
            "1. **CHECK FOR SIMILAR CONTENT** - Does similar plot/beat information already exist in related chapters?\n"
            "2. **CONSOLIDATE IF NEEDED** - If plot point appears in multiple places, ensure each adds unique perspective\n"
            "3. **AVOID REDUNDANCY** - Don't add identical information to multiple chapters\n"
            "\n"
            "CRITICAL: CROSS-REFERENCE RELATED SECTIONS\n"
            "After checking for duplicates:\n"
            "1. **SCAN THE ENTIRE OUTLINE** - Read through ALL chapters to identify related plot information\n"
            "2. **IDENTIFY ALL AFFECTED SECTIONS** - When adding/updating plot content, find ALL places it should appear\n"
            "3. **GENERATE MULTIPLE OPERATIONS** - If plot content affects multiple chapters, create operations for EACH affected section\n"
            "4. **ENSURE CONSISTENCY** - Related chapters must be updated together to maintain plot coherence\n"
            "\n"
            "Examples of when to generate multiple operations:\n"
            "- Adding character introduction → Update chapter where introduced AND character list section\n"
            "- Adding plot twist → Update chapter with twist AND any earlier chapters that need foreshadowing\n"
            "- Updating character arc → Update relevant chapters AND character section if arc affects character description\n"
            "- Adding worldbuilding detail → Update chapter where detail appears AND Notes section if it's a major world element\n\n"
        )
        
        # Build request with instructions
        request_with_instructions = ""
        if current_request:
            # Check if this is a question that doesn't require edits
            current_request_lower = current_request.lower()
            is_question_no_edit = (
                # Explicit "don't edit" phrases
                any(phrase in current_request_lower for phrase in [
                    "don't edit", "dont edit", "no edit", "no edits", "just answer", "only analyze",
                    "don't change", "dont change", "no changes", "just tell me", "just show me"
                ]) or
                # Question words/phrases that typically don't require edits
                any(phrase in current_request_lower for phrase in [
                    "what are", "what is", "what do", "what does", "what have", "what has",
                    "show me", "give me", "tell me", "list", "analyze", "assess", "evaluate",
                    "how many", "which", "where are", "when do", "who are"
                ]) and not any(phrase in current_request_lower for phrase in [
                    "add", "create", "update", "change", "modify", "revise", "edit", "generate"
                ])
            )
            
            if is_question_no_edit:
                request_with_instructions = f"""=== USER REQUEST ===
{current_request}

**IMPORTANT: This appears to be a question that can be answered WITHOUT making edits to the outline.**

Analyze the request:
- If the user is asking for information, analysis, lists, or recommendations that don't require changing the outline
- Then return a ManuscriptEdit with:
  - operations: [] (empty array - no edits needed)
  - summary: Your complete answer to the user's question (e.g., bullet points, analysis, recommendations, lists)

- If the question DOES require edits to answer properly, then generate operations as normal

Be flexible - if you can answer the question without edits, use 0 operations and put the full answer in the summary field."""
            else:
                request_with_instructions = f"""=== USER REQUEST ===
{current_request}

Generate ManuscriptEdit JSON with operations to fulfill the user's request.
Use replace_range for changing existing content, insert_after_heading for adding new content, delete_range for removing content.

**CRITICAL RULES FOR BEAT GENERATION**:
1. **CHAPTER HEADING FORMAT (ABSOLUTE REQUIREMENT)**:
   - Chapter headings MUST be EXACTLY "## Chapter N" where N is the number
   - **NEVER add titles, names, descriptions, colons, or dashes to chapter headings**
   - **CORRECT**: "## Chapter 1", "## Chapter 5", "## Chapter 12"
   - **WRONG**: "## Chapter 1: The Beginning", "## Chapter 5 - Confrontation", "## Chapter 12: Final Battle"
   - If chapter titles are desired, the user will add them - you should NEVER add them automatically

2. **100-BEAT LIMIT**: Each chapter MUST have a MAXIMUM of 100 beats
   - Before adding beats, count existing beats in the target chapter
   - If adding beats would exceed 100, you MUST remove less important beats first
   - Prioritize plot-critical beats over minor details
   - Example: If chapter has 48 beats and user wants 5 new beats, remove 3 least important existing beats

3. **NO NUMBERING IN BEATS (ABSOLUTE REQUIREMENT)**:
   - Beats MUST use bullet points starting with '- ' (dash space)
   - **NEVER use numbered lists** (1., 2., 3., etc.) or any numbering format
   - **CORRECT**: "- Character arrives at location"
   - **WRONG**: "1. Character arrives at location" or "1) Character arrives at location"

4. **NO DIALOGUE IN OUTLINES**: 
   - NEVER include quoted dialogue (e.g., "Character says 'Hello'")
   - Dialogue belongs in fiction manuscripts, NOT outlines
   - You CAN mention talking as an event: "- Character discusses plan with ally"
   - You CAN describe what is discussed: "- Character reveals secret during conversation"
   - Think of beats as plot events, not prose - describe what happens, not how characters speak

**NOTE**: If this request is actually a question that can be answered without edits, return 0 operations and put your answer in the summary field."""
                
                # Detect if creating a new chapter and add specific transition guidance
                new_chapter_patterns = [
                    r'create\s+chapter\s+(\d+)',
                    r'add\s+chapter\s+(\d+)',
                    r'new\s+chapter\s+(\d+)',
                    r'outline\s+for\s+chapter\s+(\d+)',
                    r'chapter\s+(\d+)\s+(?:outline|beats|content)',
                ]
                is_new_chapter_request = False
                new_chapter_num = None
                for pattern in new_chapter_patterns:
                    match = re.search(pattern, current_request, re.IGNORECASE)
                    if match:
                        is_new_chapter_request = True
                        new_chapter_num = int(match.group(1))
                        break
                
                # Also check if routing plan indicates new chapter creation
                routing_plan = state.get("routing_plan", {})
                if routing_plan and not is_new_chapter_request:
                    for piece in routing_plan.get("content_pieces", []):
                        target_section = piece.get("target_section", "")
                        if target_section.startswith("## Chapter"):
                            chapter_match = re.match(r'^##\s+Chapter\s+(\d+)', target_section)
                            if chapter_match:
                                # Check if this chapter exists in the outline
                                chapter_num = int(chapter_match.group(1))
                                if body_only:
                                    chapter_ranges = find_chapter_ranges(body_only)
                                    existing_chapter_numbers = {ch.chapter_number for ch in chapter_ranges if ch.chapter_number is not None}
                                    if chapter_num not in existing_chapter_numbers:
                                        is_new_chapter_request = True
                                        new_chapter_num = chapter_num
                                        break
                
                if is_new_chapter_request and new_chapter_num and not is_empty_file:
                    request_with_instructions += f"""

⚠️ **CRITICAL: CREATING NEW CHAPTER {new_chapter_num} - AVOID REPETITION**

You are creating a NEW chapter that does not yet exist in the outline. Follow these rules:

**CHAPTER TRANSITION RULES:**
1. **DO NOT REPEAT PREVIOUS CHAPTER'S ENDING BEATS**
   - The previous chapter's final beats represent where that chapter COMPLETELY ENDS
   - Your new Chapter {new_chapter_num} should START with NEW events that happen AFTER the previous chapter concluded
   - Example: If Chapter {new_chapter_num - 1} ends with "- Character arrives at the city", Chapter {new_chapter_num} should NOT start with "- Character arrives at the city"
   - Instead, Chapter {new_chapter_num} should start with NEW events like "- Character explores the city streets" or "- Character meets with contact"

2. **SMOOTH NARRATIVE TRANSITION**
   - Each chapter should pick up the narrative thread naturally from where the previous chapter ended
   - Think of it as a story continuation: the previous chapter's ending is the setup, your new chapter is the payoff
   - The new chapter should feel like a natural progression, not a repetition

3. **FRESH START WITH NEW BEATS**
   - Your new chapter's first beats should represent NEW plot developments
   - These should be events that logically follow from the previous chapter's conclusion
   - Avoid rehashing or repeating the previous chapter's ending beats

4. **QUALITY CHECK BEFORE GENERATING**
   - Review the last 2-3 beats of the previous chapter (Chapter {new_chapter_num - 1})
   - Ensure your new Chapter {new_chapter_num} beats are DISTINCT and ADVANCE the plot
   - If your first beat is too similar to the previous chapter's last beat, revise it to be more distinct

**Remember**: A good chapter transition moves the story forward, not backward or in place!"""
                
            if is_empty_file:
                request_with_instructions += """

⚠️ CRITICAL: EMPTY FILE INSTRUCTIONS
Since this file is empty (only frontmatter), follow these rules:
1. **DO NOT use anchor_text** - The file has no content to anchor to
2. **Use insert_after_heading WITHOUT anchor_text** - The system will automatically insert after frontmatter
3. **Include section headings in your text** - For example, if creating the first chapter, include '## Chapter 1' in the text
   - **CRITICAL**: Chapter headings must be EXACTLY "## Chapter N" with no titles or additional text
   - **CORRECT**: "## Chapter 1", "## Chapter 5"
   - **WRONG**: "## Chapter 1: Title", "## Chapter 5 - Name"
4. **Example operation for empty file**:
   {"op_type": "insert_after_heading", "text": "## Chapter 1\\n\\n[your content]"}
   (Note: NO anchor_text field needed - omit it entirely)
5. **DO NOT reference context headers** - Text like '=== CURRENT OUTLINE ===' or 'File: filename.md' does NOT exist in the file
6. **DO NOT use anchor_text like '# Notes' or '# Characters'** - These sections don't exist yet in an empty file"""
        
        # Use standardized helper for message construction with conversation history
        messages_list = state.get("messages", [])
        messages = _build_editing_agent_messages(
            system_prompt=system_prompt,
            context_parts=context_parts,
            current_request=request_with_instructions,
            messages_list=messages_list,
            look_back_limit=6,
            get_datetime_context=get_datetime_context
        )
        
        # Check frontmatter for temperature override (default: 0.3 for outline generation)
        frontmatter = state.get("frontmatter", {})
        temperature = frontmatter.get("temperature", 0.3)
        if temperature != 0.3:
            logger.info(f"🌡️ Using frontmatter temperature: {temperature} (default: 0.3)")
        
        # Call LLM - use llm_factory to get LLM with user's model preference
        llm = llm_factory(temperature=temperature, state=state)
        from datetime import datetime
        start_time = datetime.now()
        
        try:
            response = await llm.ainvoke(messages)
        except Exception as e:
            logger.error(f"Failed to generate edit plan: {e}")
            return {
                "structured_edit": None,
                "error": str(e),
                "task_status": "error",
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        content = response.content if hasattr(response, 'content') else str(response)
        content = _unwrap_json_response(content)
        
        # Check for empty or whitespace-only response
        if not content or not content.strip():
            logger.error("LLM returned empty response")
            return {
                "structured_edit": None,
                "error": "LLM returned empty response. This may indicate context overload or model issues.",
                "task_status": "error",
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        # Log the raw LLM response for debugging
        logger.info(f"LLM generated edit plan (first 500 chars): {content[:500]}")
        
        # Try parsing as clarification request first (RARE)
        clarification_request = None
        structured_edit = None
        
        try:
            raw = json.loads(content)
            if isinstance(raw, dict) and raw.get("clarification_needed") is True:
                confidence = raw.get("confidence_without_clarification", 0.5)
                if confidence < 0.4:
                    clarification_request = raw
                    logger.warning(f"Requesting clarification (confidence={confidence:.2f}) - RARE PATH")
                else:
                    logger.info(f"Agent wanted clarification but confidence={confidence:.2f} is too high - expecting content instead")
        except Exception:
            pass
        
        # If it's a genuine clarification request, return early
        if clarification_request:
            processing_time = (datetime.now() - start_time).total_seconds()
            questions = clarification_request.get("questions", [])
            questions_formatted = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
            response_text = (
                f"**Critical Ambiguity Detected**\n\n"
                f"{clarification_request.get('context', '')}\n\n"
                f"I cannot proceed without clarification on:\n\n"
                f"{questions_formatted}\n\n"
                f"(Confidence without clarification: {clarification_request.get('confidence_without_clarification', 0.3):.0%})"
            )
            if clarification_request.get("suggested_direction"):
                response_text += f"\n\n**Suggestion:** {clarification_request.get('suggested_direction')}\n"
            
            # Store clarification request in shared_memory for next turn
            shared_memory = state.get("shared_memory", {}) or {}
            shared_memory_out = shared_memory.copy()
            shared_memory_out["pending_outline_clarification"] = clarification_request
            
            return {
                "clarification_request": clarification_request,
                "llm_response": content,
                "response": {
                    "response": response_text,
                    "requires_user_input": True,
                    "task_status": "incomplete",
                    "agent_type": "outline_editing_agent",
                    "shared_memory": shared_memory_out
                },
                "task_status": "incomplete",
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": shared_memory_out,
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        # Otherwise, parse as ManuscriptEdit with Pydantic validation
        try:
            # Parse JSON first
            raw = json.loads(content)
            
            # Ensure required fields have defaults
            if isinstance(raw, dict):
                raw.setdefault("target_filename", filename)
                raw.setdefault("scope", "paragraph")
                raw.setdefault("summary", "Planned outline edit generated from context.")
                raw.setdefault("safety", "medium")
                raw.setdefault("operations", [])
            
            # Validate with Pydantic model
            try:
                manuscript_edit = ManuscriptEdit(**raw)
                
                # Log operation details (preserve existing logging behavior)
                for idx, op in enumerate(manuscript_edit.operations):
                    op_text = op.text
                    if op_text:
                        logger.info(f"Operation {idx} text length: {len(op_text)} chars, preview: {op_text[:100]}")
                    else:
                        logger.warning(f"Operation {idx} has EMPTY text field!")
                
                # Convert to dict for state storage (TypedDict compatibility)
                structured_edit = manuscript_edit.model_dump()
                logger.info(f"✅ Validated ManuscriptEdit with {len(manuscript_edit.operations)} operations")
            except ValidationError as ve:
                # Provide detailed validation error
                error_details = []
                for error in ve.errors():
                    field = " -> ".join(str(loc) for loc in error.get("loc", []))
                    msg = error.get("msg", "Validation error")
                    error_details.append(f"{field}: {msg}")
                
                error_msg = f"ManuscriptEdit validation failed:\n" + "\n".join(error_details)
                logger.error(f"❌ {error_msg}")
                return {
                    "llm_response": content,
                    "structured_edit": None,
                    "error": error_msg,
                    "task_status": "error",
                    # ✅ CRITICAL: Preserve state even on error
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Raw content (first 1000 chars): {content[:1000]}")
            logger.error(f"Raw content (last 200 chars): {content[-200:]}")
            logger.error(f"Total content length: {len(content)} characters")
            
            # Check if this looks like a truncated response (unterminated string/object)
            is_truncated = "Unterminated string" in str(e) or not content.rstrip().endswith('}')
            
            if is_truncated:
                error_msg = (
                    f"LLM response was truncated mid-JSON ({len(content)} chars). "
                    "This usually means the edit is too large for a single operation. "
                    "Try breaking your request into smaller chunks (e.g., edit 2-3 chapters at a time instead of 8+)."
                )
            else:
                error_msg = f"Failed to parse LLM response as JSON: {str(e)}. Response may be malformed."
            
            return {
                "llm_response": content,
                "structured_edit": None,
                "error": error_msg,
                "task_status": "error",
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        except Exception as e:
            logger.error(f"Failed to parse structured edit: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "llm_response": content,
                "structured_edit": None,
                "error": f"Failed to parse edit plan: {str(e)}",
                "task_status": "error",
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        if structured_edit is None:
            return {
                "llm_response": content,
                "structured_edit": None,
                "error": "Failed to produce a valid Outline edit plan. Ensure ONLY raw JSON ManuscriptEdit with operations is returned.",
                "task_status": "error",
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        # Check if this is a question with no operations - generate conversational answer
        operations = structured_edit.get("operations", [])
        if len(operations) == 0:
            # Check if user request looks like a question or "don't edit" request
            current_request = state.get("current_request", "").lower()
            is_question = any(keyword in current_request for keyword in [
                "?", "how", "what", "why", "when", "where", "who", "which",
                "assess", "evaluate", "review", "analyze", "check", "examine",
                "tell me", "explain", "describe", "summarize", "looks like", "looking",
                "give me", "show me", "list", "what are", "what is", "recommend"
            ]) or any(phrase in current_request for phrase in [
                "don't edit", "dont edit", "no edit", "no edits", "just answer", "only analyze",
                "don't change", "dont change", "no changes", "just tell me", "just show me"
            ])
            
            if is_question:
                logger.info("Question detected with no operations - using summary as answer")
                # For questions, the summary should already contain the answer
                structured_edit["summary"] = structured_edit.get("summary", "Analysis complete.")
                structured_edit["is_question_answer"] = True
            else:
                # Check if summary contains a substantial answer (might be a "don't edit" request that wasn't caught)
                summary = structured_edit.get("summary", "")
                if summary and len(summary) > 100:
                    # Summary is substantial - treat as answer even if keyword detection didn't match
                    logger.info("No operations but substantial summary found - treating as answer")
                    structured_edit["is_question_answer"] = True
                else:
                    # Edit request with no operations and no substantial summary - this is an error!
                    # The LLM should have generated operations for edit requests
                    logger.error("⚠️ Edit request detected but no operations generated - this should not happen!")
                    logger.error(f"   Request: {current_request[:200]}")
                    logger.error(f"   Summary: {summary[:200] if summary else 'None'}")
                    # Force an error response
                    return {
                        "llm_response": content,
                        "structured_edit": None,
                        "error": "Edit request received but no operations were generated. Please try rephrasing your request or be more specific about what you want to add or change.",
                        "task_status": "error",
                        # ✅ CRITICAL: Preserve state even on error
                        "metadata": state.get("metadata", {}),
                        "user_id": state.get("user_id", "system"),
                        "shared_memory": state.get("shared_memory", {}),
                        "messages": state.get("messages", []),
                        "query": state.get("query", "")
                    }
        
        # Incorporate status updates into operations if they exist
        chapter_status_updates = state.get("chapter_status_updates", {})
        if chapter_status_updates and structured_edit:
            from orchestrator.utils.fiction_utilities import extract_status_block, extract_chapter_outline
            
            operations = structured_edit.get("operations", [])
            body_only = state.get("body_only", "")
            
            for chapter_num, status_info in sorted(chapter_status_updates.items()):
                status_block = status_info.get("status_block", "")
                if not status_block:
                    continue
                
                # Check if status block already exists in chapter
                chapter_outline = extract_chapter_outline(body_only, chapter_num) if body_only else None
                existing_status = extract_status_block(chapter_outline) if chapter_outline else None
                
                # Create operation to insert or update status block
                chapter_heading = f"## Chapter {chapter_num}"
                
                if existing_status:
                    # Update existing status block using replace_range
                    # Find the status block in the chapter text (supports both "Status:" and "### Status")
                    status_start = -1
                    for pattern in ["### Status", "Status:"]:
                        pos = chapter_outline.find(pattern)
                        if pos >= 0:
                            status_start = pos
                            break
                    
                    if status_start >= 0:
                        # Find end of status block (next non-bullet line or end)
                        status_lines = chapter_outline[status_start:].split('\n')
                        status_end_idx = 0
                        for i, line in enumerate(status_lines[1:], 1):  # Skip header line
                            if line.strip() and not line.strip().startswith('-'):
                                status_end_idx = i
                                break
                        if status_end_idx == 0:
                            status_end_idx = len(status_lines)
                        
                        original_status = '\n'.join(status_lines[:status_end_idx])
                        # Create replace_range operation
                        status_op = {
                            "op_type": "replace_range",
                            "original_text": original_status,
                            "text": status_block.strip() + "\n"  # Ensure blank line after
                        }
                        operations.append(status_op)
                        logger.info(f"Added status update operation for Chapter {chapter_num} (replace existing)")
                else:
                    # Insert new status block IMMEDIATELY after chapter heading
                    # Use insert_after_heading with explicit instruction to place at beginning
                    status_op = {
                        "op_type": "insert_after_heading",
                        "anchor_text": chapter_heading,
                        "text": "\n" + status_block.strip() + "\n"  # Blank line before and after status block
                    }
                    operations.append(status_op)
                    logger.info(f"Added status insert operation for Chapter {chapter_num} (new status block - placed after heading)")
            
            # Update structured_edit with merged operations
            structured_edit["operations"] = operations
            if chapter_status_updates:
                logger.info(f"✅ Incorporated {len(chapter_status_updates)} status update(s) into edit plan")
        
        return {
            "llm_response": content,
            "structured_edit": structured_edit,
            "system_prompt": system_prompt,
            # ✅ CRITICAL: Preserve state for subsequent nodes
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to generate edit plan: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "llm_response": "",
            "structured_edit": None,
            "error": str(e),
            "task_status": "error",
            # ✅ CRITICAL: Preserve state even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


# ============================================
# Resolve Operations Node
# ============================================

async def resolve_operations_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve editor operations with progressive search"""
    try:
        logger.info("Resolving editor operations...")
        
        outline = state.get("outline", "")
        structured_edit = state.get("structured_edit")
        selection_start = state.get("selection_start", -1)
        selection_end = state.get("selection_end", -1)
        
        if not structured_edit or not isinstance(structured_edit.get("operations"), list):
            return {
                "editor_operations": [],
                "error": "No operations to resolve",
                "task_status": "error",
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        fm_end_idx = _frontmatter_end_index(outline)
        selection = {"start": selection_start, "end": selection_end} if selection_start >= 0 and selection_end >= 0 else None
        
        # Check if file is empty (only frontmatter)
        body_only = strip_frontmatter_block(outline)
        is_empty_file = not body_only.strip()
        
        editor_operations = []
        failed_operations = state.get("failed_operations", []) or []
        operations = structured_edit.get("operations", [])
        
        # Auto-correct anchor_text for new chapter insertions
        # If anchor_text is a chapter heading, find the last line of that chapter instead
        if body_only:
            # Find chapter ranges in the full outline (with frontmatter offset)
            chapter_ranges_full = find_chapter_ranges(outline)
            chapter_ranges_body = find_chapter_ranges(body_only)
            existing_chapter_numbers = {ch.chapter_number for ch in chapter_ranges_body if ch.chapter_number is not None}
            
            for op in operations:
                op_type = op.get("op_type", "")
                anchor_text = op.get("anchor_text", "")
                op_text = op.get("text", "")
                
                # Check if this is inserting a new chapter after an existing chapter heading
                if op_type == "insert_after_heading" and anchor_text:
                    # Check if anchor_text is a chapter heading
                    chapter_match = re.match(r'^##\s+Chapter\s+(\d+)', anchor_text.strip())
                    if chapter_match:
                        anchor_chapter_num = int(chapter_match.group(1))
                        # Check if the text being inserted is a new chapter
                        text_chapter_match = re.search(r'##\s+Chapter\s+(\d+)', op_text)
                        if text_chapter_match:
                            new_chapter_num = int(text_chapter_match.group(1))
                            # If new chapter number is higher than anchor chapter, this is a new chapter insertion
                            if new_chapter_num > anchor_chapter_num:
                                # Find the last line of the anchor chapter in the FULL outline
                                anchor_chapter = None
                                for ch in chapter_ranges_full:
                                    if ch.chapter_number == anchor_chapter_num:
                                        anchor_chapter = ch
                                        break
                                
                                if anchor_chapter:
                                    # Get the last non-empty line of this chapter from the full outline
                                    chapter_content = outline[anchor_chapter.start:anchor_chapter.end]
                                    lines = chapter_content.split('\n')
                                    last_line = None
                                    for line in reversed(lines):
                                        stripped = line.strip()
                                        if stripped and not stripped.startswith('## Chapter'):
                                            last_line = line.rstrip()
                                            break
                                    
                                    if last_line:
                                        logger.info(f"🔧 Auto-correcting anchor_text for new Chapter {new_chapter_num}")
                                        logger.info(f"   Old anchor_text (chapter heading): {anchor_text[:100]}...")
                                        logger.info(f"   New anchor_text (last line of Chapter {anchor_chapter_num}): {last_line[:100]}...")
                                        op["anchor_text"] = last_line
        
        for op in operations:
            # Sanitize op text
            op_text = op.get("text", "")
            if isinstance(op_text, str):
                op_text = strip_frontmatter_block(op_text)
                # Strip placeholder filler lines
                op_text = re.sub(r"^\s*-\s*\[To be.*?\]|^\s*\[To be.*?\]|^\s*TBD\s*$", "", op_text, flags=re.IGNORECASE | re.MULTILINE)
                op_text = re.sub(r"\n{3,}", "\n\n", op_text)
            
            # Resolve operation
            try:
                # Use centralized resolver
                cursor_pos = state.get("cursor_offset", -1)
                cursor_pos = cursor_pos if cursor_pos >= 0 else None
                resolved_start, resolved_end, resolved_text, resolved_confidence = resolve_editor_operation(
                    content=outline,
                    op_dict=op,
                    selection=selection,
                    frontmatter_end=fm_end_idx,
                    cursor_offset=cursor_pos
                )
                
                # Check if resolver returned failure signal (anchor text not found)
                if resolved_start == -1 and resolved_end == -1:
                    # Anchor text not found - check if file is empty first
                    op_type = op.get("op_type", "")
                    anchor_text = op.get("anchor_text", "")
                    
                    # For empty files, if anchor_text was provided but not found, insert after frontmatter
                    if is_empty_file and op_type == "insert_after_heading":
                        resolved_start = fm_end_idx
                        resolved_end = fm_end_idx
                        resolved_confidence = 0.7
                        logger.info(f"Empty file: anchor_text not found or invalid, inserting after frontmatter at {fm_end_idx}")
                    elif op_type == "insert_after_heading" and anchor_text:
                        # Try to find the last chapter heading as fallback
                        chapter_pattern = re.compile(r"\n##\s+Chapter\s+\d+", re.MULTILINE)
                        matches = list(chapter_pattern.finditer(outline, fm_end_idx))
                        if matches:
                            # Found chapters - insert after the last one
                            last_chapter_match = matches[-1]
                            # Find end of that chapter (next chapter or end of doc)
                            next_match = chapter_pattern.search(outline, last_chapter_match.end())
                            if next_match:
                                resolved_start = next_match.start()
                                resolved_end = next_match.start()
                            else:
                                # Last chapter - insert at end
                                resolved_start = len(outline)
                                resolved_end = len(outline)
                            resolved_confidence = 0.6
                            logger.info(f"Anchor text not found, using fallback: Inserting after last chapter at position {last_chapter_match.start()}")
                        else:
                            # No chapters found - insert after frontmatter
                            resolved_start = fm_end_idx
                            resolved_end = fm_end_idx
                            resolved_confidence = 0.5
                            logger.info(f"Anchor text not found, no chapters found, inserting after frontmatter at {fm_end_idx}")
                    else:
                        # Not a chapter insertion - use standard fallback
                        body_only = strip_frontmatter_block(outline)
                        if not body_only.strip():
                            # Empty file - insert after frontmatter
                            resolved_start = fm_end_idx
                            resolved_end = fm_end_idx
                            resolved_confidence = 0.5
                        else:
                            # Use frontmatter end
                            resolved_start = fm_end_idx
                            resolved_end = fm_end_idx
                            resolved_confidence = 0.4
                
                # Special handling for empty files: ensure operations insert after frontmatter
                if is_empty_file and resolved_start < fm_end_idx:
                    resolved_start = fm_end_idx
                    resolved_end = fm_end_idx
                    resolved_confidence = 0.7
                    logger.info(f"Empty file detected - adjusting operation to insert after frontmatter at {fm_end_idx}")
                
                logger.info(f"Resolved {op.get('op_type')} [{resolved_start}:{resolved_end}] confidence={resolved_confidence:.2f}")
                
                # Validate delete_range operations - reject if they can't find exact matches
                op_type = op.get("op_type", "")
                if op_type == "delete_range":
                    original_text = op.get("original_text", "")
                    # If original_text is large (>1000 chars) and confidence is low, this is dangerous
                    if len(original_text) > 1000 and resolved_confidence < 0.8:
                        logger.error(f"⚠️ REJECTING delete_range operation: Large deletion ({len(original_text)} chars) with low confidence ({resolved_confidence:.2f})")
                        logger.error(f"   This could delete the wrong content! Original text preview: {original_text[:200]}...")
                        # Add to failed operations
                        failed_operations = state.get("failed_operations", [])
                        failed_operations.append({
                            "op_type": op_type,
                            "original_text": original_text,
                            "text": "",
                            "error": f"Large deletion with low confidence ({resolved_confidence:.2f})"
                        })
                        # Skip this operation - don't add it to editor_operations
                        continue
                    # If confidence is very low (<0.6), reject regardless of size
                    if resolved_confidence < 0.6:
                        logger.error(f"⚠️ REJECTING delete_range operation: Very low confidence ({resolved_confidence:.2f}) - exact match not found")
                        logger.error(f"   Original text preview: {original_text[:200]}...")
                        # Add to failed operations
                        failed_operations = state.get("failed_operations", [])
                        failed_operations.append({
                            "op_type": op_type,
                            "original_text": original_text,
                            "text": "",
                            "error": f"Very low confidence ({resolved_confidence:.2f})"
                        })
                        # Skip this operation
                        continue
                
                # Calculate pre_hash
                pre_slice = outline[resolved_start:resolved_end]
                pre_hash = slice_hash(pre_slice)
                
                # Build operation dict
                resolved_op = {
                    "op_type": op.get("op_type", "replace_range"),
                    "start": resolved_start,
                    "end": resolved_end,
                    "text": resolved_text,
                    "pre_hash": pre_hash,
                    "original_text": op.get("original_text"),
                    "anchor_text": op.get("anchor_text"),
                    "left_context": op.get("left_context"),
                    "right_context": op.get("right_context"),
                    "occurrence_index": op.get("occurrence_index", 0),
                    "confidence": resolved_confidence
                }
                
                editor_operations.append(resolved_op)
                
            except Exception as e:
                logger.warning(f"Operation resolution failed: {e}, using fallback")
                # Standard fallback positioning - for empty files, insert after frontmatter
                body_only = strip_frontmatter_block(outline)
                if not body_only.strip():
                    # Empty file - insert after frontmatter
                    fallback_start = fm_end_idx
                    fallback_end = fm_end_idx
                else:
                    # Use frontmatter end
                    fallback_start = fm_end_idx
                    fallback_end = fm_end_idx
                
                pre_slice = outline[fallback_start:fallback_end]
                resolved_op = {
                    "op_type": op.get("op_type", "replace_range"),
                    "start": fallback_start,
                    "end": fallback_end,
                    "text": op_text,
                    "pre_hash": slice_hash(pre_slice),
                    "confidence": 0.3
                }
                editor_operations.append(resolved_op)
        
        return {
            "editor_operations": editor_operations,
            "failed_operations": failed_operations,
            # ✅ CRITICAL: Preserve state for subsequent nodes
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to resolve operations: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "editor_operations": [],
            "failed_operations": state.get("failed_operations", []),
            "error": str(e),
            "task_status": "error",
            # ✅ CRITICAL: Preserve state even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


# ============================================
# Format Response Node
# ============================================

async def format_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Format final response with editor operations"""
    try:
        logger.info("Formatting response...")
        
        structured_edit = state.get("structured_edit", {})
        editor_operations = state.get("editor_operations", [])
        clarification_request = state.get("clarification_request")
        task_status = state.get("task_status", "complete")
        request_type = state.get("request_type", "edit_request")
        
        # Handle questions (with or without operations)
        if request_type == "question":
            if structured_edit and structured_edit.get("summary"):
                summary = structured_edit.get("summary")
                if len(editor_operations) == 0:
                    # Pure question with no operations - return summary as response
                    logger.info("Question request with no operations - using summary as response")
                    from datetime import datetime
                    return {
                        "response": {
                            "response": summary,
                            "task_status": "complete",
                            "agent_type": "outline_editing_agent",
                            "timestamp": datetime.now().isoformat()
                        },
                        "task_status": "complete",
                        "editor_operations": [],
                        # ✅ CRITICAL: Preserve state
                        "metadata": state.get("metadata", {}),
                        "user_id": state.get("user_id", "system"),
                        "shared_memory": state.get("shared_memory", {}),
                        "messages": state.get("messages", []),
                        "query": state.get("query", "")
                    }
                else:
                    # Question with operations - use summary as conversational response
                    logger.info(f"Question request with {len(editor_operations)} operations - using summary as conversational response")
                    response_text = summary
            else:
                # Question but no summary - fallback
                logger.warning("Question request but no summary - using fallback")
                if editor_operations:
                    response_text = f"Analysis complete. Generated {len(editor_operations)} edit(s) based on your question."
                else:
                    response_text = "Analysis complete."
        # Build response text for edit requests - use summary, not full text
        else:
            # Edit request: use summary from structured_edit, not full text
            if structured_edit and structured_edit.get("summary"):
                response_text = structured_edit.get("summary")
            elif editor_operations:
                # Fallback: brief description of operations
                op_count = len(editor_operations)
                response_text = f"Made {op_count} edit(s) to the outline."
            else:
                response_text = "Edit plan ready."
        
        # If we have a clarification request, it was already formatted in generate_edit_plan
        if clarification_request:
            response = state.get("response", {})
            # Check if response is already in standard format
            if isinstance(response, dict) and all(key in response for key in ["response", "task_status", "agent_type", "timestamp"]):
                logger.info("📝 OUTLINE SUBGRAPH FORMAT: Clarification request - response already in standard format")
                return {
                    "response": response,
                    "task_status": "incomplete",
                    # ✅ CRITICAL: Preserve state (final node, but good practice)
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            else:
                # Legacy format - wrap in AgentResponse
                response_text = response.get("response", "") if isinstance(response, dict) else str(response) if response else "Clarification needed"
                standard_response = AgentResponse(
                    response=response_text,
                    task_status="incomplete",
                    agent_type="outline_editing_subgraph",
                    timestamp=datetime.now().isoformat()
                )
                return {
                    "response": standard_response.dict(exclude_none=True),
                    "task_status": "incomplete",
                    # ✅ CRITICAL: Preserve state (final node, but good practice)
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
        
        if task_status == "error":
            error_msg = state.get("error", "Unknown error")
            logger.error(f"❌ OUTLINE SUBGRAPH FORMAT: Error state detected: {error_msg}")
            error_response = AgentResponse(
                response=f"Outline editing failed: {error_msg}",
                task_status="error",
                agent_type="outline_editing_subgraph",
                timestamp=datetime.now().isoformat(),
                error=error_msg
            )
            return {
                "response": error_response.dict(exclude_none=True),
                "task_status": "error",
                # ✅ CRITICAL: Preserve state (final node, but good practice)
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        # Build response text using shared utilities
        failed_operations = state.get("failed_operations", [])
        
        if request_type == "question":
            response_text = build_response_text_for_question(
                structured_edit,
                editor_operations,
                fallback="Analysis complete."
            )
        else:
            # Edit request - use shared utility
            response_text = build_response_text_for_edit(
                structured_edit,
                editor_operations,
                fallback="Edit plan ready."
            )
            # Add failed operations section if present
            if failed_operations:
                failed_section = build_failed_operations_section(failed_operations, "outline")
                response_text = response_text + failed_section
        
        # Build manuscript_edit metadata using shared utility
        manuscript_edit_metadata = create_manuscript_edit_metadata(structured_edit, editor_operations)
        
        # Build standard response using AgentResponse contract (WITHOUT editor_operations/manuscript_edit)
        from datetime import datetime
        logger.info(f"📊 OUTLINE SUBGRAPH FORMAT: Creating AgentResponse with task_status='{task_status}', {len(editor_operations)} operation(s)")
        standard_response = AgentResponse(
            response=response_text,
            task_status=task_status,
            agent_type="outline_editing_subgraph",
            timestamp=datetime.now().isoformat()
            # NO editor_operations, NO manuscript_edit (they go at state level)
        )
        
        # Clear any pending clarification since we're completing successfully
        shared_memory = state.get("shared_memory", {}) or {}
        shared_memory_out = shared_memory.copy()
        shared_memory_out.pop("pending_outline_clarification", None)
        
        logger.info(f"📊 OUTLINE SUBGRAPH FORMAT: Response text length: {len(response_text)} chars")
        logger.info(f"📊 OUTLINE SUBGRAPH FORMAT: Editor operations: {len(editor_operations)} operation(s)")
        logger.info(f"📊 OUTLINE SUBGRAPH FORMAT: Manuscript edit: {'present' if manuscript_edit_metadata else 'missing'}")
        logger.info(f"📤 OUTLINE SUBGRAPH FORMAT: Returning standard AgentResponse with {len(editor_operations)} editor operation(s)")
        
        return {
            "response": standard_response.dict(exclude_none=True),
            "editor_operations": editor_operations,  # STATE LEVEL (primary source)
            "manuscript_edit": manuscript_edit_metadata.dict(exclude_none=True) if manuscript_edit_metadata else None,  # STATE LEVEL
            "task_status": task_status,
            "shared_memory": shared_memory_out,
            **preserve_critical_state(state)
        }
        
    except Exception as e:
        logger.error(f"❌ OUTLINE SUBGRAPH FORMAT: Failed to format response: {e}")
        import traceback
        logger.error(f"❌ OUTLINE SUBGRAPH FORMAT: Traceback: {traceback.format_exc()}")
        # Return standard error response using shared utility
        return create_writing_error_response(
            str(e),
            "outline_editing_subgraph",
            state
        )


# ============================================
# Subgraph Builder
# ============================================

def build_outline_editing_subgraph(
    checkpointer,
    llm_factory: Callable,
    get_datetime_context: Callable
):
    """
    Build outline editing subgraph
    
    Args:
        checkpointer: LangGraph checkpointer for state persistence
        llm_factory: Callable that returns LLM instance (takes temperature and state)
        get_datetime_context: Callable that returns datetime context string
    
    Returns:
        Compiled StateGraph ready for integration into parent agents
    """
    from typing import TypedDict
    
    # Define state schema (matches outline_editing_agent state)
    class OutlineEditingSubgraphState(TypedDict):
        # Critical 5 keys
        metadata: Dict[str, Any]
        user_id: str
        shared_memory: Dict[str, Any]
        messages: List[Any]
        query: str
        
        # Editor context
        active_editor: Dict[str, Any]
        outline: str
        filename: str
        frontmatter: Dict[str, Any]
        cursor_offset: int
        selection_start: int
        selection_end: int
        body_only: str
        
        # References
        rules_body: Optional[str]
        style_body: Optional[str]
        characters_bodies: List[str]
        
        # Request context
        clarification_context: str
        current_request: str
        system_prompt: str
        llm_response: str
        structured_edit: Optional[Dict[str, Any]]
        clarification_request: Optional[Dict[str, Any]]
        request_type: str
        
        # Operations
        editor_operations: List[Dict[str, Any]]
        failed_operations: List[Dict[str, Any]]
        
        # Response
        response: Dict[str, Any]
        task_status: str
        error: str
        
        # Mode tracking
        generation_mode: str
        available_references: Dict[str, Any]
        reference_summary: str
        mode_guidance: str
        reference_quality: Dict[str, Any]
        reference_warnings: List[str]
        
        # Structure analysis
        outline_completeness: float
        chapter_count: int
        structure_warnings: List[str]
        structure_guidance: str
        has_synopsis: bool
        has_notes: bool
        has_characters: bool
        has_outline_section: bool
        
        # Content routing
        routing_plan: Optional[Dict[str, Any]]
        
        # Tracked items and status tracking
        tracked_items: Dict[str, List[str]]  # Loaded from frontmatter
        chapter_status_updates: Dict[int, Dict[str, Any]]  # Pending status updates by chapter
    
    # Create workflow
    workflow = StateGraph(OutlineEditingSubgraphState)
    
    # Add nodes (wrapped to pass llm_factory and get_datetime_context)
    async def prepare_context_wrapper(state):
        return await prepare_context_node(state)
    
    async def load_references_wrapper(state):
        return await load_references_node(state)
    
    async def detect_request_type_wrapper(state):
        return await detect_request_type_node(state, llm_factory)
    
    async def generate_status_summary_wrapper(state):
        return await generate_status_summary_node(state, llm_factory, get_datetime_context)
    
    async def generate_edit_plan_wrapper(state):
        return await generate_edit_plan_node(state, llm_factory, get_datetime_context)
    
    async def resolve_operations_wrapper(state):
        return await resolve_operations_node(state)
    
    async def format_response_wrapper(state):
        return await format_response_node(state)
    
    workflow.add_node("prepare_context", prepare_context_wrapper)
    workflow.add_node("load_references", load_references_wrapper)
    workflow.add_node("detect_request_type", detect_request_type_wrapper)
    workflow.add_node("generate_status_summary", generate_status_summary_wrapper)
    workflow.add_node("generate_edit_plan", generate_edit_plan_wrapper)
    workflow.add_node("resolve_operations", resolve_operations_wrapper)
    workflow.add_node("format_response", format_response_wrapper)
    
    # Set entry point
    workflow.set_entry_point("prepare_context")
    
    # Define edges
    workflow.add_edge("prepare_context", "load_references")
    workflow.add_edge("load_references", "detect_request_type")
    
    # Conditional routing: route to status generation if tracked items exist and conditions are met
    def route_after_detect_request_type(state: Dict[str, Any]) -> str:
        tracked_items = state.get("tracked_items", {})
        has_tracked_items = bool(tracked_items and any(
            items for items in tracked_items.values() if isinstance(items, list) and items
        ))
        
        if not has_tracked_items:
            return "generate_edit_plan"
        
        # Check if we should generate status
        current_request = state.get("current_request", "").lower()
        
        # Explicit status update request
        explicit_status_request = any(phrase in current_request for phrase in [
            "update status", "regenerate status", "refresh status", "status summary",
            "update status summaries", "regenerate status summaries"
        ])
        
        # Check for new chapter creation or editing an existing chapter
        from orchestrator.utils.fiction_utilities import extract_chapter_number_from_request, find_chapter_ranges
        requested_chapter = extract_chapter_number_from_request(current_request)
        body_only = state.get("body_only", "")
        chapter_ranges = find_chapter_ranges(body_only) if body_only else []
        existing_chapters = {ch.chapter_number for ch in chapter_ranges if ch.chapter_number is not None}
        is_new_chapter = requested_chapter and requested_chapter not in existing_chapters and requested_chapter > 1
        is_editing_chapter_2_plus = (
            requested_chapter and requested_chapter > 1 and requested_chapter in existing_chapters
        )
        
        # Route to status generation if:
        # 1. Explicit status request (includes Chapter 1 if explicitly requested), OR
        # 2. New chapter being created (after Chapter 1), OR
        # 3. Editing any chapter 2+ when tracked_items are configured (so relationships/characters stay in sync)
        if explicit_status_request or is_new_chapter or (has_tracked_items and is_editing_chapter_2_plus):
            return "generate_status_summary"
        
        return "generate_edit_plan"
    
    workflow.add_conditional_edges(
        "detect_request_type",
        route_after_detect_request_type,
        {
            "generate_status_summary": "generate_status_summary",
            "generate_edit_plan": "generate_edit_plan"
        }
    )
    
    # Status generation always goes to edit plan (which will incorporate status updates)
    workflow.add_edge("generate_status_summary", "generate_edit_plan")
    workflow.add_edge("generate_edit_plan", "resolve_operations")
    workflow.add_edge("resolve_operations", "format_response")
    workflow.add_edge("format_response", END)
    
    # Compile with checkpointer
    return workflow.compile(checkpointer=checkpointer)
