"""
Style Editing Subgraph

Reusable subgraph for style guide development and editing workflows.
Used by Writing Assistant when the active editor is a style document (frontmatter.type: style).

Supports two modes:
1. Analysis Mode: Analyze narrative examples to generate style guide
2. Editing Mode: Edit existing style guides using structured operations

Produces EditorOperations suitable for Prefer Editor HITL application.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from orchestrator.utils.editor_operation_resolver import resolve_editor_operation
from orchestrator.models.agent_response_contract import AgentResponse, ManuscriptEditMetadata
from orchestrator.utils.writing_subgraph_utilities import (
    preserve_critical_state,
    create_writing_error_response,
    extract_user_request,
    paragraph_bounds,
    strip_frontmatter_block,
    slice_hash,
    build_response_text_for_question,
    build_response_text_for_edit,
    create_manuscript_edit_metadata,
    prepare_writing_context,
    load_writing_references,
    detect_writing_request_type
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


def _detect_analysis_mode(user_request: str, body_only: str) -> Tuple[bool, str]:
    """
    Detect if user wants to analyze narrative examples.
    Returns (is_analysis_mode, extracted_examples)
    """
    if not user_request:
        return False, ""
    
    request_lower = user_request.lower()
    
    # Check for explicit analysis keywords
    analysis_keywords = [
        "analyze this", "analysis of", "generate style from", "create style guide from",
        "extract style from", "from these examples", "based on these", "style from this"
    ]
    
    has_analysis_keyword = any(kw in request_lower for kw in analysis_keywords)
    
    # CRITICAL: Only trigger analysis mode if user provides SUBSTANTIAL narrative content (>200 chars)
    # Short requests like "Craft the style guide" should NOT trigger analysis mode!
    has_substantial_content = len(user_request) > 200
    
    # Check if request contains narrative prose (not just instructions)
    narrative_indicators = [
        '"' in user_request,  # Actual dialogue quotes (not apostrophes)
        user_request.count('\n') >= 3,  # Multi-line prose (3+ newlines)
        re.search(r'\b(he|she|they)\s+(walked|ran|said|thought|felt|saw|heard)', request_lower),  # Narrative patterns
    ]
    
    has_narrative_content = sum(narrative_indicators) >= 2
    
    # Analysis mode ONLY if:
    # 1. Explicit "analyze this text" + substantial content, OR
    # 2. File is empty AND user provides narrative prose (200+ chars with narrative patterns)
    is_empty_file = not body_only.strip()
    
    if has_analysis_keyword and has_substantial_content:
        # Explicit analysis request with substantial examples
        example_markers = ["analyze", "examples:", "this:", "these:", "from:"]
        extracted = user_request
        for marker in example_markers:
            if marker in request_lower:
                idx = request_lower.find(marker)
                extracted = user_request[idx + len(marker):].strip()
                break
        return True, extracted
    elif is_empty_file and has_substantial_content and has_narrative_content:
        # Empty file + user provided substantial narrative prose to analyze
        return True, user_request
    
    return False, ""


def _extract_conversation_history(messages: List[Any], limit: int = 10) -> List[Dict[str, str]]:
    """Extract conversation history from LangChain messages, filtering out large data URIs"""
    try:
        history = []
        for msg in messages[-limit:]:
            if hasattr(msg, 'content'):
                role = "assistant" if hasattr(msg, 'type') and msg.type == "ai" else "user"
                # Simple content extraction (no filtering for now - can be enhanced)
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

def _build_system_prompt(analysis_mode: bool = False, genre: str = "fiction") -> str:
    """Build system prompt for style editing (fiction or non-fiction)"""
    if genre == "nonfiction":
        base_prompt = (
            "You are a MASTER STYLE ARCHITECT for NON-FICTION STYLE GUIDE documents (voice, structure, clarity). "
            "Persona disabled. Adhere strictly to frontmatter and project guidelines.\n\n"
        )
    else:
        base_prompt = (
            "You are a MASTER STYLE ARCHITECT for FICTION STYLE GUIDE documents (narrative voice, technique, craft). "
            "Persona disabled. Adhere strictly to frontmatter, project Rules, and Style.\n\n"
        )
    
    base_prompt += (
        "**CRITICAL: ASK CLARIFYING QUESTIONS WHEN NEEDED**\n"
        "If the user's request is ambiguous, incomplete, or could benefit from clarification:\n"
        "- Ask specific, targeted questions to understand style preferences\n"
        "- Suggest style considerations the user might not have thought of\n"
        "- Point out potential conflicts with existing style guidelines\n"
        "- Propose style refinements that would make the guide more useful\n"
        "Examples of when to ask questions:\n"
        "- User: 'Add POV guidelines' → Ask: 'Which POV? (first-person, third-person limited, third-person omniscient?) Single POV or multiple? How strict are the boundaries?'\n"
        "- User: 'Add pacing guidelines' → Ask: 'What pacing style? (fast-paced thriller, slow literary, varied?) Scene-level or sentence-level pacing techniques?'\n"
        "- User adds conflicting guideline → Point out: 'This conflicts with existing guideline X. Should I update the old guideline, or are these context-dependent?'\n"
        "\n"
        "**HOW TO ASK QUESTIONS**: Return operations array EMPTY and use summary field for questions/suggestions.\n\n"
        "STRUCTURED OUTPUT REQUIRED: Return ONLY raw JSON (no prose, no markdown, no code fences) matching this schema:\n"
        "{\n"
        '  "type": "ManuscriptEdit",\n'
        '  "target_filename": string,\n'
        '  "scope": one of ["paragraph", "chapter", "multi_chapter"],\n'
        '  "summary": string,\n'
        '  "chapter_index": integer|null,\n'
        '  "safety": one of ["low", "medium", "high"],\n'
        '  "operations": [\n'
        '    { "op_type": "replace_range|delete_range|insert_after_heading|insert_after", "start": 0, "end": 0,\n'
        '      "text": "content to insert or replace with",\n'
        '      "original_text": "exact text from file (REQUIRED for replace_range/delete_range)",\n'
        '      "anchor_text": "exact line to insert after (REQUIRED for insert_after_heading/insert_after)" }\n'
        "  ]\n"
        "}\n\n"
        "OUTPUT RULES:\n"
        "- Output MUST be a single JSON object only.\n"
        "- Do NOT include triple backticks or language tags.\n"
        "- Do NOT include explanatory text before or after the JSON.\n"
        "- If asking questions/seeking clarification: Return empty operations array and put questions in summary field\n"
        "- If making edits: Return operations array with edits and brief description in summary field\n\n"
    )
    
    if analysis_mode:
        if genre == "nonfiction":
            base_prompt += (
                "=== ANALYSIS MODE (NON-FICTION) ===\n"
                "You are analyzing non-fiction examples to extract style characteristics and generate a style guide.\n\n"
                "**CRITICAL**: The Style Guide is HOW to write, not WHAT to cover. Focus on:\n"
                "- Voice and tone (formal, conversational, academic, journalistic, etc.)\n"
                "- Point of View (first-person 'I', third-person observational, etc.)\n"
                "- Tense usage (past, present, mixed)\n"
                "- Structure and organization patterns\n"
                "- Detail level (concise, moderate, comprehensive)\n"
                "- Sentence structure and rhythm\n"
                "- Technical language handling\n"
                "- Evidence and citation style\n\n"
                "**DO NOT include**:\n"
                "- Content topics (those are determined by the outline or article)\n"
                "- Specific facts or data\n"
                "- Section structure (that's in the outline)\n\n"
                "**EMPTY FILE HANDLING** (CRITICAL):\n"
                "When inserting analyzed style guide into an EMPTY file (only frontmatter, no body):\n"
                "- Create a SINGLE operation that includes ALL headers and content\n"
                "- Use op_type='insert_after' with anchor_text='---' (frontmatter closing)\n"
                "- The 'text' field must contain the COMPLETE style guide with ALL headers\n"
                "- Do NOT create separate operations for each section (headers don't exist yet to anchor to!)\n"
                "- Example text: '\\n\\n## Voice and Tone\\n[content]\\n\\n## Point of View\\n[content]...'\n\n"
            )
        else:
            base_prompt += (
                "=== ANALYSIS MODE (FICTION) ===\n"
                "You are analyzing narrative examples to extract style characteristics and generate a style guide.\n\n"
                "**CRITICAL**: The Style Guide is HOW to write, not WHAT happens. Focus on:\n"
                "- Narrative voice and tone (not plot or story content)\n"
                "- Point of View (POV) technique\n"
                "- Tense usage\n"
                "- Pacing patterns\n"
                "- Sensory detail level\n"
                "- Sentence structure and rhythm\n"
                "- Descriptive techniques\n"
                "- Show-don't-tell approach\n\n"
                "**DO NOT include**:\n"
                "- Dialogue style (that's character-specific, belongs in character profiles)\n"
                "- Plot elements or story content\n"
                "- Character-specific voice patterns\n\n"
                "**EMPTY FILE HANDLING** (CRITICAL):\n"
                "When inserting analyzed style guide into an EMPTY file (only frontmatter, no body):\n"
                "- Create a SINGLE operation that includes ALL headers and content\n"
                "- Use op_type='insert_after' with anchor_text='---' (frontmatter closing)\n"
                "- The 'text' field must contain the COMPLETE style guide with ALL headers\n"
                "- Do NOT create separate operations for each section (headers don't exist yet to anchor to!)\n"
                "- Example text: '\\n\\n## Narrative Voice\\n[content]\\n\\n## Point of View\\n[content]...'\n\n"
            )
    else:
        base_prompt += (
            "FORMATTING CONTRACT (STYLE DOCUMENTS):\n"
            "- Never emit YAML frontmatter in operations[].text. Preserve existing frontmatter as-is.\n"
            "- Use Markdown headings and lists for the body.\n"
            "- When creating or normalizing structure, prefer this scaffold (top-level headings):\n"
            "  ## Narrative Voice\n"
            "  ## Point of View\n"
            "  ## Tense\n"
            "  ## Pacing\n"
            "  ## Sensory Detail Level\n"
            "  ## Sentence Structure\n"
            "  ## Descriptive Techniques\n"
            "  ## Rhythm and Flow\n"
            "  ## Notes\n\n"
            "**IMPORTANT**: Dialogue style is character-specific and belongs in character profiles, NOT in the overall style guide.\n\n"
        )
    
    base_prompt += (
        "=== INTENT AND OPERATION SELECTION (TARGETED EDITS ONLY) ===\n"
        "**NEVER replace the entire style guide or an entire section in one operation.** Use granular operations compatible with the Outline subgraph.\n\n"
        "**DETECT USER INTENT AND CHOOSE OPERATIONS:**\n"
        "- **ADDITION INTENT** (user wants to add new guidelines without removing existing):\n"
        "  * Section is COMPLETELY EMPTY below header → insert_after_heading with anchor_text='## Section Name'\n"
        "  * Section ALREADY HAS CONTENT → insert_after with anchor_text=EXACT last line of that section (last bullet or last sentence). Do NOT replace the whole section.\n"
        "  * Your 'text' field contains ONLY the new guideline(s) to add. Do NOT copy existing content.\n"
        "- **GRANULAR CORRECTION INTENT** (user wants to change a specific word/phrase):\n"
        "  * Use replace_range with original_text=FULL bullet or sentence from file (10-20+ words), text=same content with only the word/phrase changed.\n"
        "  * Example: User says 'use present tense not past' and file has '- Prefer past tense for narrative' → original_text='- Prefer past tense for narrative', text='- Prefer present tense for narrative'.\n"
        "- **REVISION INTENT** (user wants to update/expand existing guideline):\n"
        "  * Use replace_range on the SPECIFIC paragraph or bullet being revised. original_text=exact current text, text=revised version.\n"
        "  * Do NOT replace the whole section. One operation per changed paragraph/bullet when possible.\n"
        "- **DELETION INTENT** (user wants to remove a guideline):\n"
        "  * Use delete_range with original_text=EXACT text to remove (no headers). Or replace_range with text=''.\n"
        "- **FULL SECTION REWRITE** (user explicitly asks to rewrite a whole section):\n"
        "  * Prefer one replace_range with original_text=entire section body (no header) and text=new content.\n"
        "  * If section is very large, split into multiple replace_range operations (e.g. paragraph by paragraph).\n\n"
        "**CRITICAL**: replace_range is for CHANGING existing text only. Do NOT use replace_range to 'add' content by replacing a whole section with section+new; use insert_after instead.\n\n"
        "RULES FOR EDITS:\n"
        "0) **ASK QUESTIONS WHEN NEEDED**: If request is ambiguous, lacks style detail, or has implications the user may not have considered, ASK instead of guessing. Return empty operations and use summary for questions.\n"
        "1) Make focused, surgical edits near the cursor/selection unless the user requests re-organization.\n"
        "2) Maintain the scaffold above; if missing, create only the minimal sections the user asked for.\n"
        "3) Prefer paragraph/sentence-level replacements; avoid large-span rewrites unless asked.\n"
        "4) Enforce consistency: cross-check against Rules and Characters for worldbuilding alignment.\n"
        "5) **CRITICAL: CROSS-REFERENCE RELATED SECTIONS** - When adding or updating style guidelines, you MUST:\n"
        "   - Scan the ENTIRE document for related sections that should be updated together\n"
        "   - Identify ALL sections that reference or relate to the style guideline being added/updated\n"
        "   - Generate MULTIPLE operations if a single guideline addition requires updates to multiple related sections\n"
        "   - Example: If adding POV guidelines, check Point of View section AND Narrative Voice section if they reference each other\n"
        "   - Example: If updating pacing techniques, check Pacing section AND Rhythm and Flow section if they overlap\n"
        "   - Example: If adding descriptive techniques, check Descriptive Techniques section AND Sentence Structure section if they relate\n"
        "   - NEVER update only one section when related sections exist that should be updated together\n"
        "   - The operations array can contain MULTIPLE operations - use it to update all related sections in one pass\n\n"
        "ANCHOR REQUIREMENTS (CRITICAL):\n"
        "For EVERY operation, you MUST provide precise anchors:\n\n"
        "REVISE/DELETE Operations:\n"
        "- ALWAYS include 'original_text' with EXACT, VERBATIM text from the file\n"
        "- Minimum 10-20 words, include complete sentences with natural boundaries\n"
        "- Copy and paste directly - do NOT retype or modify\n"
        "- NEVER include header lines (###, ##, #) in original_text!\n"
        "- OR provide both 'left_context' and 'right_context' (exact surrounding text)\n\n"
        "INSERT Operations (ONLY for truly empty sections!):\n"
        "- **insert_after_heading**: Use ONLY when section is completely empty below the header\n"
        "  * op_type='insert_after_heading' with anchor_text='## Section' (exact header line)\n"
        "  * Example: Adding style guidelines after '## Narrative Voice' header when section is completely empty\n"
        "  * ⚠️ CRITICAL WARNING: Before using insert_after_heading, you MUST verify the section is COMPLETELY EMPTY!\n"
        "  * ⚠️ If there is ANY text below the header (even a single line), use replace_range instead!\n"
        "  * ⚠️ Using insert_after_heading when content exists will INSERT BETWEEN the header and existing text, splitting the section!\n"
        "  * ⚠️ This creates duplicate content and breaks the section structure - NEVER do this!\n"
        "  * Example of WRONG behavior: '## Pacing\\n[INSERT HERE splits section]\\n- Existing guideline' ← WRONG! Use replace_range on existing content!\n"
        "  * Example of CORRECT usage: '## Pacing\\n[empty - no text below]' ← OK to use insert_after_heading\n"
        "  * This is the SAFEST method - it NEVER deletes headers, always inserts AFTER them - BUT ONLY FOR EMPTY SECTIONS\n\n"
        "- **insert_after**: Use when continuing text mid-paragraph, mid-sentence, or after specific text\n"
        "  * op_type='insert_after' with anchor_text='last few words before insertion point'\n"
        "  * Example: Continuing a sentence or adding to an existing paragraph\n\n"
        "- **REPLACE Operations (PREFERRED for updating existing content!):\n"
        "- **replace_range**: Use when section exists but needs improvement, completion, or revision\n"
        "  * If section has ANY content (even incomplete or placeholder), use replace_range to update it\n"
        "  * Example: Section has '- Third-person limited' but needs more detail → replace_range with original_text='- Third-person limited' and expanded text\n"
        "  * Example: Section has '[To be developed]' → replace_range with original_text='[To be developed]' and actual content\n"
        "  * This ensures existing content is replaced/updated, not duplicated\n\n"
        "Additional Options:\n"
        "- 'occurrence_index' if text appears multiple times (0-based, default 0)\n"
        "- Start/end indices are approximate; anchors take precedence\n\n"
        "=== DECISION TREE FOR OPERATION TYPE ===\n"
        "**STEP 1: Read the section content carefully!**\n"
        "- Look at what exists below the header\n"
        "- Is there ANY text at all? Even a single line?\n"
        "\n"
        "**STEP 2: Choose operation based on what exists:**\n"
        "1. Section is COMPLETELY EMPTY below header (no text at all)? → insert_after_heading with anchor_text=\"## Section\"\n"
        "2. Section has ANY content and you are UPDATING/REVISING it? → replace_range on the specific paragraph/bullet only (NO headers in original_text!)\n"
        "3. Section has content and you are ADDING new guidelines? → insert_after with anchor_text=EXACT last line of that section (last bullet or last sentence). Do NOT replace the whole section.\n"
        "4. Adding to existing list (one new bullet)? → insert_after with anchor_text=last bullet line; or replace_range on last bullet to expand it\n"
        "5. Deleting SPECIFIC content? → delete_range with original_text (NO headers!)\n"
        "6. Continuing mid-sentence? → insert_after\n\n"
        "CRITICAL: When updating existing content (even if incomplete), use 'replace_range' on the existing content!\n"
        "NEVER include headers in 'original_text' for replace_range - headers will be deleted!\n"
        "⚠️ NEVER use insert_after_heading when content exists - it will SPLIT the section and create duplicates!\n"
        "\n"
        "**CORRECT EXAMPLE**:\n"
        "- Updating existing content: {\"op_type\": \"replace_range\", \"original_text\": \"- Third-person limited\", \"text\": \"- Third-person limited\\n- Focus on single character's perspective\"}\n"
        "\n"
        "**WRONG EXAMPLE**:\n"
        "- ❌ {\"op_type\": \"insert_after_heading\"} when section has content - will split section!\n\n"
        "=== SPACING RULES (CRITICAL - READ CAREFULLY!) ===\n"
        "YOUR TEXT MUST END IMMEDIATELY AFTER THE LAST CHARACTER!\n\n"
        'CORRECT: "- Item 1\\n- Item 2\\n- Item 3"  ← Ends after "3" with NO \\n\n'
        'WRONG: "- Item 1\\n- Item 2\\n"  ← Extra \\n after last line creates blank line!\n'
        'WRONG: "- Item 1\\n- Item 2\\n\\n"  ← \\n\\n creates 2 blank lines!\n'
        'WRONG: "- Item 1\\n\\n- Item 2"  ← Double \\n\\n between items creates blank line!\n\n'
        "IRON-CLAD RULE: After last line = ZERO \\n (nothing!)\n"
        "5) Headings must be clear; do not duplicate sections. If an equivalent heading exists, update it in place.\n"
        "6) NO PLACEHOLDER FILLERS: If a requested section has no content yet, create the heading only and leave the body blank. Do NOT insert placeholders like '[To be developed]' or 'TBD'.\n"
    )
    
    return base_prompt


# ============================================
# Node Functions
# ============================================

async def prepare_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare context: extract active editor, validate style type"""
    try:
        logger.info("Preparing context for style editing...")
        
        # Use shared utility for base context preparation
        context = await prepare_writing_context(
            state=state,
            doc_type="style",
            default_filename="style.md",
            content_key="style",
            validate_type=True  # Hard gate on type: style
        )
        
        # Check for errors from shared utility
        if context.get("error"):
            return context
        
        # Normalize line endings (style-specific)
        style = context.get("style", "")
        normalized_text = style.replace("\r\n", "\n")
        context["style_content"] = normalized_text
        context["style"] = normalized_text  # Alias for compatibility
        context["body_only"] = strip_frontmatter_block(normalized_text)
        
        # Extract user request using shared utility
        current_request = extract_user_request(state)
        
        # Detect genre from frontmatter (fiction or nonfiction) - defaults to fiction for backward compatibility
        frontmatter = context.get("frontmatter", {})
        genre = str(frontmatter.get("genre", "fiction")).strip().lower()
        if genre not in ["fiction", "nonfiction"]:
            genre = "fiction"  # Default to fiction
        
        # Add style-specific fields
        context.update({
            "current_request": current_request.strip(),
            "genre": genre
        })
        
        return context
        
    except Exception as e:
        logger.error(f"Failed to prepare context: {e}")
        return create_writing_error_response(
            str(e),
            "style_editing_subgraph",
            state
        )


async def load_references_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load referenced context files (rules, characters) directly from style frontmatter"""
    try:
        logger.info("Loading referenced context files from style frontmatter...")
        
        # Use shared utility for reference loading
        result = await load_writing_references(
            state=state,
            reference_config={
                "rules": ["rules"],
                "characters": ["characters", "character_*"]
            },
            cascade_config=None,  # No cascading for style
            doc_type_filter="style"
        )
        
        # Check for errors
        if result.get("error"):
            return result
        
        # Preserve style-specific context
        result.update({
            "active_editor": state.get("active_editor", {}),
            "style_content": state.get("style_content", ""),
            "style": state.get("style", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "body_only": state.get("body_only", ""),
            "current_request": state.get("current_request", "")
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to load references: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "rules_body": None,
            "characters_bodies": [],
            "error": str(e),
            "task_status": "error",
            **preserve_critical_state(state),
            # Preserve style-specific context even on error
            "active_editor": state.get("active_editor", {}),
            "style_content": state.get("style_content", ""),
            "style": state.get("style", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "body_only": state.get("body_only", ""),
            "current_request": state.get("current_request", "")
        }


async def detect_request_type_node(
    state: Dict[str, Any],
    llm_factory: Callable
) -> Dict[str, Any]:
    """Detect if request is a question or edit request"""
    try:
        logger.info("Detecting request type...")
        
        current_request = state.get("current_request", "")
        body_only = state.get("body_only", "")
        rules_body = state.get("rules_body")
        characters_bodies = state.get("characters_bodies", [])
        
        # Build simple prompt for LLM to determine intent
        prompt = f"""Analyze the user's request and determine if it's a QUESTION or an EDIT REQUEST.

**USER REQUEST**: {current_request}

**CONTEXT**:
- Current style guide: {body_only[:500] if body_only else "Empty style guide"}
- Has rules reference: {bool(rules_body)}
- Has {len(characters_bodies)} character reference(s)

**INTENT DETECTION**:
- QUESTIONS (including pure questions and conditional edits): User is asking a question - may or may not want edits
  - Pure questions: "What style rules do we have?", "Does this follow our style guide?", "Show me the narrative voice guidelines", "What POV are we using?"
  - Conditional edits: "Do we have POV guidelines? Add them if not", "What style rules? Suggest improvements if needed"
  - Questions often start with: "Do you", "What", "Can you", "Are there", "How many", "Show me", "Is", "Does", "Are we", "Suggest"
  - **Key insight**: Questions can be answered, and IF edits are needed based on the answer, they can be made
  - Route ALL questions to edit path - LLM can decide if edits are needed
  
- EDIT REQUESTS: User wants to create, modify, or generate content - NO question asked
  - Examples: "Add POV guidelines", "Create a style guide", "Update the narrative voice section", "Revise the dialogue patterns"
  - Edit requests are action-oriented: "add", "create", "update", "generate", "change", "replace", "revise"
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
        llm = llm_factory(temperature=0.1, state=state)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        content = response.content if hasattr(response, 'content') else str(response)
        content = _unwrap_json_response(content)
        
        # Parse response
        try:
            result = json.loads(content)
            request_type = result.get("request_type", "edit_request")
            confidence = result.get("confidence", 0.5)
            reasoning = result.get("reasoning", "")
            
            logger.info(f"Request type detected: {request_type} (confidence: {confidence:.2f}) - {reasoning}")
            
            return {
                "request_type": request_type,
                # ✅ CRITICAL: Preserve all state
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                # ✅ CRITICAL: Preserve style-specific context
                "active_editor": state.get("active_editor", {}),
                "style_content": state.get("style_content", ""),
                "style": state.get("style", ""),
                "filename": state.get("filename", ""),
                "frontmatter": state.get("frontmatter", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "body_only": state.get("body_only", ""),
                "rules_body": state.get("rules_body"),
                "characters_bodies": state.get("characters_bodies", []),
                "current_request": state.get("current_request", "")
            }
        except Exception as e:
            logger.warning(f"Failed to parse request type detection: {e}, defaulting to edit_request")
            return {
                "request_type": "edit_request",
                **preserve_critical_state(state),
                # ✅ CRITICAL: Preserve style-specific context
                "active_editor": state.get("active_editor", {}),
                "style_content": state.get("style_content", ""),
                "style": state.get("style", ""),
                "filename": state.get("filename", ""),
                "frontmatter": state.get("frontmatter", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "body_only": state.get("body_only", ""),
                "rules_body": state.get("rules_body"),
                "characters_bodies": state.get("characters_bodies", []),
                "current_request": state.get("current_request", ""),
                "genre": state.get("genre", "fiction")
            }
        
    except Exception as e:
        logger.error(f"Failed to detect request type: {e}")
        return {
            "request_type": "edit_request",
            # ✅ CRITICAL: Preserve all state
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve style-specific context
            "active_editor": state.get("active_editor", {}),
            "style_content": state.get("style_content", ""),
            "style": state.get("style", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "body_only": state.get("body_only", ""),
            "rules_body": state.get("rules_body"),
            "characters_bodies": state.get("characters_bodies", []),
            "current_request": state.get("current_request", ""),
            "genre": state.get("genre", "fiction")
        }


async def detect_mode_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Detect if we're in analysis mode or editing mode"""
    try:
        logger.info("Detecting mode (analysis vs editing)...")
        
        current_request = state.get("current_request", "")
        body_only = state.get("body_only", "")
        
        analysis_mode, narrative_examples = _detect_analysis_mode(current_request, body_only)
        
        logger.info(f"Mode detected: {'ANALYSIS' if analysis_mode else 'EDITING'}")
        if analysis_mode:
            logger.info(f"Extracted narrative examples length: {len(narrative_examples)} chars")
        
        return {
            "analysis_mode": analysis_mode,
            "narrative_examples": narrative_examples,
            # ✅ CRITICAL: Preserve all state
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve style-specific context
            "active_editor": state.get("active_editor", {}),
            "style_content": state.get("style_content", ""),
            "style": state.get("style", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "body_only": state.get("body_only", ""),
            "rules_body": state.get("rules_body"),
            "characters_bodies": state.get("characters_bodies", []),
            "current_request": state.get("current_request", ""),
            "request_type": state.get("request_type", "edit_request"),
            "genre": state.get("genre", "fiction")
        }
        
    except Exception as e:
        logger.error(f"Failed to detect mode: {e}")
        return {
            "analysis_mode": False,
            "narrative_examples": "",
            # ✅ CRITICAL: Preserve all state even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve style-specific context
            "active_editor": state.get("active_editor", {}),
            "style_content": state.get("style_content", ""),
            "style": state.get("style", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "body_only": state.get("body_only", ""),
            "rules_body": state.get("rules_body"),
            "characters_bodies": state.get("characters_bodies", []),
            "current_request": state.get("current_request", ""),
            "request_type": state.get("request_type", "edit_request"),
            "genre": state.get("genre", "fiction")
        }


async def analyze_examples_node(
    state: Dict[str, Any],
    llm_factory: Callable,
    get_datetime_context: Callable
) -> Dict[str, Any]:
    """Analyze examples to extract style characteristics (fiction or non-fiction)"""
    try:
        logger.info("Analyzing examples to extract style characteristics...")
        
        narrative_examples = state.get("narrative_examples", "")
        current_request = state.get("current_request", "")
        rules_body = state.get("rules_body")
        characters_bodies = state.get("characters_bodies", [])
        genre = state.get("genre", "fiction")
        
        if not narrative_examples:
            logger.warning("No examples to analyze")
            return {
                # ✅ CRITICAL: Preserve all state even when skipping
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                # ✅ CRITICAL: Preserve style-specific context
                "active_editor": state.get("active_editor", {}),
                "style_content": state.get("style_content", ""),
                "style": state.get("style", ""),
                "filename": state.get("filename", ""),
                "frontmatter": state.get("frontmatter", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "body_only": state.get("body_only", ""),
                "rules_body": state.get("rules_body"),
                "characters_bodies": state.get("characters_bodies", []),
                "current_request": state.get("current_request", ""),
                "request_type": state.get("request_type", "edit_request"),
                "analysis_mode": state.get("analysis_mode", False),
                "narrative_examples": state.get("narrative_examples", ""),
                "genre": state.get("genre", "fiction")
            }
        
        # Build genre-appropriate analysis prompt
        if genre == "nonfiction":
            analysis_prompt = (
                "=== EXAMPLES TO ANALYZE ===\n"
                f"{narrative_examples}\n\n"
                "=== TASK ===\n"
                "Analyze these non-fiction examples and extract style characteristics. Focus on:\n\n"
                "1. **Voice and Tone**: What is the authorial voice? (formal, conversational, academic, journalistic, personal, etc.)\n"
                "2. **Point of View**: What POV is used? (first-person 'I', third-person observational, second-person 'you', etc.)\n"
                "3. **Tense**: What tense is used? (past, present, mixed)\n"
                "4. **Structure and Organization**: How is information organized? (chronological, thematic, problem-solution, etc.)\n"
                "5. **Detail Level**: How much detail is provided? (minimal/concise, moderate, comprehensive/exhaustive)\n"
                "6. **Sentence Structure**: What patterns in sentence length, rhythm, and flow?\n"
                "7. **Technical Language**: How are technical terms handled? (defined, assumed knowledge, analogies, etc.)\n"
                "8. **Evidence and Citations**: How are sources and evidence referenced?\n\n"
                "Generate a style guide following this structure:\n"
                "## Voice and Tone\n"
                "[Voice description, tone, formality level]\n\n"
                "## Point of View\n"
                "[POV type and usage]\n\n"
                "## Tense\n"
                "[Tense usage and patterns]\n\n"
                "## Structure and Organization\n"
                "[How content is organized and presented]\n\n"
                "## Detail Level\n"
                "[Level of detail and explanation]\n\n"
                "## Sentence Structure\n"
                "[Sentence length patterns, rhythm, flow]\n\n"
                "## Technical Language\n"
                "[How technical terms are handled]\n\n"
                "## Evidence and Citations\n"
                "[How sources are referenced]\n\n"
                "## Notes\n"
                "[Additional style considerations or guidelines]\n\n"
            )
        else:
            # Fiction analysis prompt
            analysis_prompt = (
                "=== NARRATIVE EXAMPLES TO ANALYZE ===\n"
                f"{narrative_examples}\n\n"
                "=== TASK ===\n"
                "Analyze these narrative examples and extract style characteristics. Focus on:\n\n"
                "1. **Narrative Voice**: What is the tone, personality, and voice? (formal, casual, poetic, gritty, etc.)\n"
                "2. **Point of View**: What POV is used? (first-person, third-person limited, third-person omniscient, etc.)\n"
                "3. **Tense**: What tense is used? (past, present, mixed)\n"
                "4. **Pacing**: How is pacing handled? (fast, slow, varied, with specific techniques)\n"
                "5. **Sensory Detail Level**: How much sensory detail? (minimal, moderate, rich)\n"
                "6. **Sentence Structure**: What patterns in sentence length, rhythm, and flow?\n"
                "7. **Descriptive Techniques**: How is description handled? (show-don't-tell, metaphor usage, imagery style)\n"
                "8. **Rhythm and Flow**: How are paragraphs structured? What pacing techniques and transitions are used?\n\n"
                "**CRITICAL**: Do NOT include dialogue style - that's character-specific and belongs in character profiles.\n\n"
                "Generate a style guide following this structure:\n"
                "## Narrative Voice\n"
                "[Voice description, tone, personality]\n\n"
                "## Point of View\n"
                "[POV type and technique]\n\n"
                "## Tense\n"
                "[Tense usage and patterns]\n\n"
                "## Pacing\n"
                "[Pacing characteristics and techniques]\n\n"
                "## Sensory Detail Level\n"
                "[Level of sensory description]\n\n"
                "## Sentence Structure\n"
                "[Sentence length patterns, rhythm, flow]\n\n"
                "## Descriptive Techniques\n"
                "[Show-don't-tell approach, metaphor usage, imagery style]\n\n"
                "## Rhythm and Flow\n"
                "[Paragraph structure, pacing techniques, transitions]\n\n"
                "## Notes\n"
                "[Additional style considerations, techniques, or guidelines]\n\n"
            )
        
        if rules_body:
            analysis_prompt += f"=== UNIVERSE RULES (for context) ===\n{rules_body[:500]}...\n\n"
        
        if characters_bodies:
            analysis_prompt += f"=== CHARACTER PROFILES (for context - dialogue style is character-specific) ===\n{characters_bodies[0][:300]}...\n\n"
        
        analysis_prompt += (
            "Return ONLY the style guide content (markdown format with headings and content). "
            "Do NOT include frontmatter. Do NOT include dialogue style sections."
        )
        
        # Call LLM for analysis
        llm = llm_factory(temperature=0.7, state=state)  # Higher temperature for creative analysis
        
        datetime_context = get_datetime_context()
        system_msg = "You are a style analysis expert. Analyze examples and extract style characteristics to create a comprehensive style guide."
        if genre == "nonfiction":
            system_msg = "You are a non-fiction style analysis expert. Analyze non-fiction examples and extract writing style characteristics to create a comprehensive style guide."
        
        messages = [
            SystemMessage(content=system_msg),
            SystemMessage(content=datetime_context),
            HumanMessage(content=analysis_prompt)
        ]
        
        response = await llm.ainvoke(messages)
        analyzed_content = response.content if hasattr(response, 'content') else str(response)
        
        logger.info(f"Analysis complete, generated {len(analyzed_content)} chars of style guide content")
        
        # Store analyzed content for use in edit plan generation
        return {
            "narrative_examples": narrative_examples,
            "analyzed_style_content": analyzed_content,
            # ✅ CRITICAL: Preserve all state
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve style-specific context
            "active_editor": state.get("active_editor", {}),
            "style_content": state.get("style_content", ""),
            "style": state.get("style", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "body_only": state.get("body_only", ""),
            "rules_body": state.get("rules_body"),
            "characters_bodies": state.get("characters_bodies", []),
            "current_request": state.get("current_request", ""),
            "request_type": state.get("request_type", "edit_request"),
            "analysis_mode": state.get("analysis_mode", False),
            "genre": state.get("genre", "fiction")
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze examples: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "analyzed_style_content": "",
            "error": str(e),
            # ✅ CRITICAL: Preserve all state even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve style-specific context
            "active_editor": state.get("active_editor", {}),
            "style_content": state.get("style_content", ""),
            "style": state.get("style", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "body_only": state.get("body_only", ""),
            "rules_body": state.get("rules_body"),
            "characters_bodies": state.get("characters_bodies", []),
            "current_request": state.get("current_request", ""),
            "request_type": state.get("request_type", "edit_request"),
            "analysis_mode": state.get("analysis_mode", False),
            "narrative_examples": state.get("narrative_examples", ""),
            "genre": state.get("genre", "fiction")
        }


async def generate_edit_plan_node(
    state: Dict[str, Any],
    llm_factory: Callable,
    get_datetime_context: Callable
) -> Dict[str, Any]:
    """Generate edit plan using LLM"""
    try:
        logger.info("Generating style edit plan...")
        
        style = state.get("style_content") or state.get("style", "")
        filename = state.get("filename", "style.md")
        body_only = state.get("body_only", "")
        current_request = state.get("current_request", "")
        analysis_mode = state.get("analysis_mode", False)
        analyzed_style_content = state.get("analyzed_style_content", "")
        request_type = state.get("request_type", "edit_request")
        is_question = request_type == "question"
        genre = state.get("genre", "fiction")
        
        rules_body = state.get("rules_body")
        characters_bodies = state.get("characters_bodies", [])
        
        selection_start = state.get("selection_start", -1)
        selection_end = state.get("selection_end", -1)
        
        # Build system prompt (analysis mode or editing mode, genre-aware)
        system_prompt = _build_system_prompt(analysis_mode=analysis_mode, genre=genre)
        
        # Build context message
        context_parts = [
            "=== STYLE GUIDE CONTEXT ===\n",
            f"File: {filename}\n\n",
        ]
        
        if analysis_mode and analyzed_style_content:
            # Analysis mode: use analyzed content
            context_parts.append("=== ANALYZED STYLE GUIDE CONTENT ===\n")
            context_parts.append(f"{analyzed_style_content}\n\n")
            
            # Check if file is empty
            is_empty = not body_only.strip()
            
            if is_empty:
                context_parts.append(
                    "**CRITICAL - FILE IS EMPTY AND ANALYZED CONTENT IS READY**\n\n"
                    "The style guide file is currently EMPTY (only frontmatter). You have already analyzed examples and generated style guide content above.\n\n"
                    "**YOUR ONLY TASK**: Create ONE operation that inserts ALL the analyzed content into the empty file.\n\n"
                    "**DO NOT**:\n"
                    "- Ask clarification questions (you already have the content!)\n"
                    "- Request more information (the analyzed content is complete!)\n"
                    "- Return 0 operations (you MUST create the insert operation!)\n\n"
                    "**MANDATORY OPERATION**:\n"
                    "{\n"
                    '  "op_type": "insert_after",\n'
                    '  "anchor_text": "---",\n'
                    '  "text": "\\n\\n[ALL the analyzed style guide content from above, starting with the first ## header]"\n'
                    "}\n\n"
                    "The 'text' field must contain the ENTIRE analyzed style guide content shown above, including ALL headers and sections.\n"
                    "Do NOT split into multiple operations - the file is empty, so headers don't exist to anchor to yet!\n\n"
                )
            else:
                context_parts.append("Provide a ManuscriptEdit JSON plan to insert this style guide content into the document.\n")
                context_parts.append("Sections may already exist - use insert_after_heading or replace_range to update them appropriately.\n")
        else:
            # Editing mode: use current content
            context_parts.append("Current Buffer (frontmatter stripped):\n" + body_only + "\n\n")
            
            # Add mode-specific instructions
            if is_question:
                context_parts.append(
                    "\n=== QUESTION REQUEST: ANALYZE AND OPTIONALLY EDIT ===\n"
                    "The user has asked a question about the style guide.\n\n"
                    "**YOUR TASK**:\n"
                    "1. **ANALYZE FIRST**: Answer the user's question by evaluating the current content\n"
                    "   - Pure questions: 'What style rules do we have?' → Report current style guidelines\n"
                    "   - Verification questions: 'Does this follow our style guide?' → Check compliance, report findings\n"
                    "   - Suggestion questions: 'Suggest improvements to the style guide' → Analyze current content, then suggest additions\n"
                    "   - Conditional questions: 'Do we have POV guidelines? Add them if not' → Check, then edit if needed\n"
                    "2. **THEN EDIT IF NEEDED**: Based on your analysis, make edits if necessary\n"
                    "   - If question implies a desired state ('Add them if not') → Provide editor operations\n"
                    "   - If question asks for suggestions ('Suggest improvements') → Provide editor operations with suggested additions\n"
                    "   - If question is pure information ('What style rules?') → No edits needed, just answer\n"
                    "   - Include your analysis in the 'summary' field of your response\n\n"
                    "**RESPONSE FORMAT**:\n"
                    "- In the 'summary' field: Answer the question clearly and explain your analysis\n"
                    "- In the 'operations' array: Provide editor operations ONLY if edits are needed\n"
                    "- If no edits needed: Return empty operations array, but answer the question in summary\n"
                    "- If edits needed: Provide operations AND explain what you found in summary\n\n"
                )
            else:
                context_parts.append("Provide a ManuscriptEdit JSON plan for the style guide document.\n")
        
        if rules_body:
            context_parts.append(f"=== UNIVERSE RULES (for consistency) ===\n{rules_body[:500]}...\n\n")
        
        if characters_bodies:
            context_parts.append("=== CHARACTER PROFILES (for context) ===\n")
            context_parts.append(f"{characters_bodies[0][:300]}...\n\n")
        
        # Build request with mode-specific instructions
        request_with_instructions = ""
        if current_request:
            if is_question:
                request_with_instructions = (
                    f"USER REQUEST: {current_request}\n\n"
                    "**QUESTION MODE**: Answer the question first, then provide edits if needed.\n\n"
                    "**TARGETED EDITS ONLY** (if edits are needed): One operation per changed paragraph/bullet. Never replace the entire style guide or an entire section unless the user explicitly asks for a full rewrite.\n\n"
                    "CRITICAL: CROSS-REFERENCE AND MULTIPLE OPERATIONS (if edits are needed)\n"
                    "Before generating operations, you MUST:\n"
                    "1. **SCAN THE ENTIRE DOCUMENT** - Read through ALL sections to identify related style guidelines\n"
                    "2. **IDENTIFY ALL AFFECTED SECTIONS** - When adding/updating a style guideline, find ALL places it should appear\n"
                    "3. **GENERATE MULTIPLE OPERATIONS** - If a guideline affects multiple sections, create operations for EACH affected section\n"
                    "4. **ENSURE CONSISTENCY** - Related sections must be updated together to maintain style guide coherence\n"
                    "\n"
                    "CRITICAL ANCHORING INSTRUCTIONS (if edits are needed):\n"
                    "- **BEFORE using insert_after_heading**: Verify the section is COMPLETELY EMPTY (no text below header)\n"
                    "- **If section has ANY content**: Use replace_range to update it, NOT insert_after_heading\n"
                    "- **insert_after_heading will SPLIT sections**: If you use it when content exists, it inserts BETWEEN header and existing text!\n"
                    "- **UPDATING EXISTING CONTENT**: If a section exists but needs improvement/completion, use 'replace_range' with 'original_text' matching the EXISTING content\n"
                    "  * Example: Section has '- Third-person limited' but needs more → replace_range with original_text='- Third-person limited' and expanded text\n"
                    "  * Example: Section has placeholder '[To be developed]' → replace_range with original_text='[To be developed]' and actual content\n"
                    "- **ADDING TO EMPTY SECTIONS**: Only use 'insert_after_heading' when section is completely empty below the header\n"
                    "- For REVISE/DELETE: Provide 'original_text' with EXACT, VERBATIM text from file (10-20+ words, complete sentences)\n"
                    "- For INSERT: Use 'insert_after_heading' with 'anchor_text' ONLY for completely empty sections, or 'insert_after' for mid-paragraph\n"
                    "- NEVER include header lines in original_text for replace_range operations\n"
                    "- Copy text directly from the file - do NOT retype or paraphrase\n"
                    "- Without precise anchors, the operation WILL FAIL\n"
                    "- **KEY RULE**: If content exists (even if incomplete), use replace_range to update it. Only use insert_after_heading for truly empty sections.\n"
                    "- You can return MULTIPLE operations in the operations array - one for each related section that needs updating"
                )
            else:
                request_with_instructions = (
                    f"USER REQUEST: {current_request}\n\n"
                    "**TARGETED EDITS ONLY**: Use one operation per changed paragraph/bullet when possible. Never replace the entire style guide or an entire section unless the user explicitly asks for a full rewrite.\n\n"
                    "CRITICAL: CROSS-REFERENCE AND MULTIPLE OPERATIONS\n"
                    "Before generating operations, you MUST:\n"
                    "1. **SCAN THE ENTIRE DOCUMENT** - Read through ALL sections to identify related style guidelines\n"
                    "2. **IDENTIFY ALL AFFECTED SECTIONS** - When adding/updating a style guideline, find ALL places it should appear\n"
                    "3. **GENERATE MULTIPLE OPERATIONS** - If a guideline affects multiple sections, create operations for EACH affected section\n"
                    "4. **ENSURE CONSISTENCY** - Related sections must be updated together to maintain style guide coherence\n"
                    "\n"
                    "Examples of when to generate multiple operations:\n"
                    "- Adding POV guidelines → Update 'Point of View' section AND 'Narrative Voice' section if they reference each other\n"
                    "- Adding pacing techniques → Update 'Pacing' section AND 'Rhythm and Flow' section if they overlap\n"
                    "- Adding descriptive techniques → Update 'Descriptive Techniques' section AND 'Sentence Structure' section if they relate\n"
                    "- Updating a style guideline → If guideline appears in multiple sections, update ALL occurrences, not just one\n"
                    "\n"
                    "CRITICAL ANCHORING INSTRUCTIONS:\n"
                    "- **BEFORE using insert_after_heading**: Verify the section is COMPLETELY EMPTY (no text below header)\n"
                    "- **If section has ANY content**: Use replace_range to update it, NOT insert_after_heading\n"
                    "- **insert_after_heading will SPLIT sections**: If you use it when content exists, it inserts BETWEEN header and existing text!\n"
                    "- For REVISE/DELETE: Provide 'original_text' with EXACT, VERBATIM text from file (10-20+ words, complete sentences)\n"
                    "- For INSERT: Use 'insert_after_heading' with 'anchor_text' ONLY for completely empty sections, or 'insert_after' for mid-paragraph\n"
                    "- NEVER include header lines in original_text for replace_range operations\n"
                    "- Copy text directly from the file - do NOT retype or paraphrase\n"
                    "- Without precise anchors, the operation WILL FAIL\n"
                    "- For granular revisions, use replace_range with exact original_text matching the text to change\n"
                    "- You can return MULTIPLE operations in the operations array - one for each related section that needs updating"
                )
        
        # Use standardized helper for message construction with conversation history
        messages_list = state.get("messages", [])
        messages = _build_editing_agent_messages(
            system_prompt=system_prompt,
            context_parts=context_parts,
            current_request=request_with_instructions,
            messages_list=messages_list,
            get_datetime_context=get_datetime_context,
            look_back_limit=6
        )
        
        # Call LLM
        llm = llm_factory(temperature=0.3, state=state)
        response = await llm.ainvoke(messages)
        
        content = response.content if hasattr(response, 'content') else str(response)
        content = _unwrap_json_response(content)
        
        # Parse structured edit
        structured_edit = None
        try:
            raw = json.loads(content)
            if isinstance(raw, dict) and isinstance(raw.get("operations"), list):
                raw.setdefault("target_filename", filename)
                raw.setdefault("scope", "paragraph")
                raw.setdefault("summary", "Planned style guide edit generated from context.")
                raw.setdefault("safety", "medium")
                
                # Process operations to preserve anchor fields
                ops = []
                for op in raw["operations"]:
                    if not isinstance(op, dict):
                        continue
                    op_type = op.get("op_type")
                    if op_type not in ("replace_range", "delete_range", "insert_after_heading", "insert_after"):
                        op_type = "replace_range"
                    
                    ops.append({
                        "op_type": op_type,
                        "start": op.get("start", 0),
                        "end": op.get("end", 0),
                        "text": op.get("text", ""),
                        "original_text": op.get("original_text"),
                        "anchor_text": op.get("anchor_text"),
                        "left_context": op.get("left_context"),
                        "right_context": op.get("right_context"),
                        "occurrence_index": op.get("occurrence_index", 0)
                    })
                raw["operations"] = ops
                structured_edit = raw
            else:
                structured_edit = json.loads(content)
        except Exception as e:
            logger.error(f"Failed to parse structured edit: {e}")
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
                "query": state.get("query", ""),
                # ✅ CRITICAL: Preserve style-specific context
                "active_editor": state.get("active_editor", {}),
                "style_content": state.get("style_content", ""),
                "style": state.get("style", ""),
                "filename": state.get("filename", ""),
                "frontmatter": state.get("frontmatter", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "body_only": state.get("body_only", ""),
                "rules_body": state.get("rules_body"),
                "characters_bodies": state.get("characters_bodies", []),
                "current_request": state.get("current_request", ""),
                "request_type": state.get("request_type", "edit_request"),
                "analysis_mode": state.get("analysis_mode", False),
                "narrative_examples": state.get("narrative_examples", ""),
                "analyzed_style_content": state.get("analyzed_style_content", ""),
                "genre": state.get("genre", "fiction")
            }
        
        if structured_edit is None:
            return {
                "llm_response": content,
                "structured_edit": None,
                "error": "Failed to produce a valid Style guide edit plan. Ensure ONLY raw JSON ManuscriptEdit with operations is returned.",
                "task_status": "error",
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                # ✅ CRITICAL: Preserve style-specific context
                "active_editor": state.get("active_editor", {}),
                "style_content": state.get("style_content", ""),
                "style": state.get("style", ""),
                "filename": state.get("filename", ""),
                "frontmatter": state.get("frontmatter", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "body_only": state.get("body_only", ""),
                "rules_body": state.get("rules_body"),
                "characters_bodies": state.get("characters_bodies", []),
                "current_request": state.get("current_request", ""),
                "request_type": state.get("request_type", "edit_request"),
                "analysis_mode": state.get("analysis_mode", False),
                "narrative_examples": state.get("narrative_examples", ""),
                "analyzed_style_content": state.get("analyzed_style_content", ""),
                "genre": state.get("genre", "fiction")
            }
        
        # Log what we got from the LLM
        ops_count = len(structured_edit.get("operations", [])) if structured_edit else 0
        logger.info(f"LLM generated {ops_count} operation(s)")
        if ops_count > 0:
            for i, op in enumerate(structured_edit.get("operations", [])):
                op_type = op.get("op_type", "unknown")
                text_preview = (op.get("text", "") or "")[:100]
                logger.info(f"  Operation {i+1}: {op_type}, text preview: {text_preview}...")
        
        # CRITICAL FIX: If analysis mode with analyzed content, empty file, but LLM returned 0 operations, auto-generate the operation!
        if analysis_mode and analyzed_style_content and not body_only.strip() and ops_count == 0:
            logger.warning("⚠️ ANALYSIS MODE FIX: LLM returned 0 operations despite having analyzed content and empty file!")
            logger.info("🔧 AUTO-GENERATING insert operation to ensure analyzed content is inserted...")
            
            # Auto-generate the operation
            auto_operation = {
                "op_type": "insert_after",
                "anchor_text": "---",
                "text": f"\n\n{analyzed_style_content}",
                "start": 0,
                "end": 0
            }
            structured_edit["operations"] = [auto_operation]
            structured_edit["summary"] = structured_edit.get("summary", "Style guide generated from analysis")
            logger.info(f"✅ AUTO-GENERATED operation: insert_after with {len(analyzed_style_content)} chars of analyzed content")
        
        return {
            "llm_response": content,
            "structured_edit": structured_edit,
            "system_prompt": system_prompt,
            # ✅ CRITICAL: Preserve all state
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve style-specific context
            "active_editor": state.get("active_editor", {}),
            "style_content": state.get("style_content", ""),
            "style": state.get("style", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "body_only": state.get("body_only", ""),
            "rules_body": state.get("rules_body"),
            "characters_bodies": state.get("characters_bodies", []),
            "current_request": state.get("current_request", ""),
            "request_type": state.get("request_type", "edit_request"),
            "analysis_mode": state.get("analysis_mode", False),
            "narrative_examples": state.get("narrative_examples", ""),
            "analyzed_style_content": state.get("analyzed_style_content", ""),
            "genre": state.get("genre", "fiction")
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
            # ✅ CRITICAL: Preserve all state even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve style-specific context
            "active_editor": state.get("active_editor", {}),
            "style_content": state.get("style_content", ""),
            "style": state.get("style", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "body_only": state.get("body_only", ""),
            "rules_body": state.get("rules_body"),
            "characters_bodies": state.get("characters_bodies", []),
            "current_request": state.get("current_request", ""),
            "request_type": state.get("request_type", "edit_request"),
            "analysis_mode": state.get("analysis_mode", False),
            "narrative_examples": state.get("narrative_examples", ""),
            "analyzed_style_content": state.get("analyzed_style_content", ""),
            "genre": state.get("genre", "fiction")
        }


async def resolve_operations_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve editor operations with progressive search"""
    try:
        logger.info("Resolving editor operations...")
        
        style = state.get("style_content") or state.get("style", "")
        structured_edit = state.get("structured_edit")
        selection_start = state.get("selection_start", -1)
        selection_end = state.get("selection_end", -1)
        current_request = state.get("current_request", "")
        
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
                "query": state.get("query", ""),
                # ✅ CRITICAL: Preserve style-specific context
                "active_editor": state.get("active_editor", {}),
                "style_content": state.get("style_content", ""),
                "style": state.get("style", ""),
                "filename": state.get("filename", ""),
                "frontmatter": state.get("frontmatter", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "body_only": state.get("body_only", ""),
                "rules_body": state.get("rules_body"),
                "characters_bodies": state.get("characters_bodies", []),
                "current_request": state.get("current_request", ""),
                "request_type": state.get("request_type", "edit_request"),
                "analysis_mode": state.get("analysis_mode", False),
                "narrative_examples": state.get("narrative_examples", ""),
                "analyzed_style_content": state.get("analyzed_style_content", ""),
                "structured_edit": state.get("structured_edit"),
                "genre": state.get("genre", "fiction")
            }
        
        fm_end_idx = _frontmatter_end_index(style)
        selection = {"start": selection_start, "end": selection_end} if selection_start >= 0 and selection_end >= 0 else None
        
        # Check if file is empty (only frontmatter)
        body_only = strip_frontmatter_block(style)
        is_empty_file = not body_only.strip()
        
        # Check revision mode
        revision_mode = current_request and any(k in current_request.lower() for k in ["revise", "revision", "tweak", "adjust", "polish", "tighten", "edit only"])
        
        editor_operations = []
        operations = structured_edit.get("operations", [])
        
        logger.info(f"Resolving {len(operations)} operation(s) from structured_edit")
        
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
                    content=style,
                    op_dict=op,
                    selection=selection,
                    frontmatter_end=fm_end_idx,
                    cursor_offset=cursor_pos
                )
                
                # Special handling for empty files: ensure operations insert after frontmatter
                if is_empty_file and resolved_start < fm_end_idx:
                    resolved_start = fm_end_idx
                    resolved_end = fm_end_idx
                    resolved_confidence = 0.7
                    logger.info(f"Empty file detected - adjusting operation to insert after frontmatter at {fm_end_idx}")
                
                logger.info(f"Resolved {op.get('op_type')} [{resolved_start}:{resolved_end}] confidence={resolved_confidence:.2f}")
                
                # Protect YAML frontmatter
                if resolved_start < fm_end_idx:
                    if op.get("op_type") == "delete_range":
                        continue
                    if resolved_end <= fm_end_idx:
                        resolved_start = fm_end_idx
                        resolved_end = fm_end_idx
                    else:
                        resolved_start = fm_end_idx
                
                # Clamp to selection in revision mode
                if revision_mode and op.get("op_type") != "delete_range":
                    if selection_start >= 0 and selection_end > selection_start:
                        resolved_start = max(selection_start, resolved_start)
                        resolved_end = min(selection_end, resolved_end)
                
                resolved_start = max(0, min(len(style), resolved_start))
                resolved_end = max(resolved_start, min(len(style), resolved_end))
                
                # Handle spacing for inserts
                if resolved_start == resolved_end:
                    left_tail = style[max(0, resolved_start-2):resolved_start]
                    if left_tail.endswith("\n\n"):
                        needed_prefix = ""
                    elif left_tail.endswith("\n"):
                        needed_prefix = "\n"
                    else:
                        needed_prefix = "\n\n"
                    try:
                        leading_stripped = re.sub(r'^\n+', '', resolved_text)
                        resolved_text = f"{needed_prefix}{leading_stripped}"
                    except Exception:
                        resolved_text = f"{needed_prefix}{resolved_text}"
                
                # Calculate pre_hash
                pre_slice = style[resolved_start:resolved_end]
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
                    "occurrence_index": op.get("occurrence_index", 0)
                }
                
                editor_operations.append(resolved_op)
                
            except Exception as e:
                logger.warning(f"Failed to resolve operation: {e}")
                continue
        
        logger.info(f"Successfully resolved {len(editor_operations)} operation(s) out of {len(operations)}")
        
        return {
            "editor_operations": editor_operations,
            # ✅ CRITICAL: Preserve all state
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve style-specific context
            "active_editor": state.get("active_editor", {}),
            "style_content": state.get("style_content", ""),
            "style": state.get("style", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "body_only": state.get("body_only", ""),
            "rules_body": state.get("rules_body"),
            "characters_bodies": state.get("characters_bodies", []),
            "current_request": state.get("current_request", ""),
            "request_type": state.get("request_type", "edit_request"),
            "analysis_mode": state.get("analysis_mode", False),
            "narrative_examples": state.get("narrative_examples", ""),
            "analyzed_style_content": state.get("analyzed_style_content", ""),
            "structured_edit": state.get("structured_edit"),
            "genre": state.get("genre", "fiction")
        }
        
    except Exception as e:
        logger.error(f"Failed to resolve operations: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "editor_operations": [],
            "error": str(e),
            "task_status": "error",
            # ✅ CRITICAL: Preserve all state even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve style-specific context
            "active_editor": state.get("active_editor", {}),
            "style_content": state.get("style_content", ""),
            "style": state.get("style", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "body_only": state.get("body_only", ""),
            "rules_body": state.get("rules_body"),
            "characters_bodies": state.get("characters_bodies", []),
            "current_request": state.get("current_request", ""),
            "request_type": state.get("request_type", "edit_request"),
            "analysis_mode": state.get("analysis_mode", False),
            "narrative_examples": state.get("narrative_examples", ""),
            "analyzed_style_content": state.get("analyzed_style_content", ""),
            "structured_edit": state.get("structured_edit"),
            "genre": state.get("genre", "fiction")
        }


async def format_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Format final response with editor operations"""
    try:
        logger.info("📝 STYLE SUBGRAPH FORMAT: Formatting style editing response...")
        
        structured_edit = state.get("structured_edit")
        editor_operations = state.get("editor_operations", [])
        current_request = state.get("current_request", "")
        request_type = state.get("request_type", "edit_request")
        task_status = state.get("task_status", "complete")
        
        # Normalize task_status to valid enum value
        if task_status not in ["complete", "incomplete", "permission_required", "error"]:
            logger.warning(f"⚠️ STYLE SUBGRAPH FORMAT: Invalid task_status '{task_status}', normalizing to 'complete'")
            task_status = "complete"
        
        if not structured_edit:
            error = state.get("error", "Unknown error")
            logger.error(f"❌ STYLE SUBGRAPH FORMAT: No structured_edit found: {error}")
            error_response = AgentResponse(
                response=f"Failed to generate style guide edit plan: {error}",
                task_status="error",
                agent_type="style_editing_subgraph",
                timestamp=datetime.now().isoformat(),
                error=error
            )
            return {
                "response": error_response.dict(exclude_none=True),
                "task_status": "error",
                # ✅ CRITICAL: Preserve all state
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        # Build response text using shared utilities
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
        
        # Build manuscript_edit metadata using shared utility
        manuscript_edit_metadata = create_manuscript_edit_metadata(structured_edit, editor_operations)
        
        # Build standard response using AgentResponse contract (WITHOUT editor_operations/manuscript_edit)
        logger.info(f"📊 STYLE SUBGRAPH FORMAT: Creating AgentResponse with task_status='complete', {len(editor_operations)} operation(s)")
        standard_response = AgentResponse(
            response=response_text,
            task_status="complete",
            agent_type="style_editing_subgraph",
            timestamp=datetime.now().isoformat()
            # NO editor_operations, NO manuscript_edit (they go at state level)
        )
        
        logger.info(f"📊 STYLE SUBGRAPH FORMAT: Response text length: {len(response_text)} chars")
        logger.info(f"📊 STYLE SUBGRAPH FORMAT: Editor operations: {len(editor_operations)} operation(s)")
        logger.info(f"📊 STYLE SUBGRAPH FORMAT: Manuscript edit: {'present' if manuscript_edit_metadata else 'missing'}")
        logger.info(f"📤 STYLE SUBGRAPH FORMAT: Returning standard AgentResponse with {len(editor_operations)} editor operation(s)")
        
        return {
            "response": standard_response.dict(exclude_none=True),
            "editor_operations": editor_operations,  # STATE LEVEL (primary source)
            "manuscript_edit": manuscript_edit_metadata.dict(exclude_none=True) if manuscript_edit_metadata else None,  # STATE LEVEL
            "task_status": "complete",
            **preserve_critical_state(state),
            # ✅ CRITICAL: Preserve style-specific context and outputs
            "active_editor": state.get("active_editor", {}),
            "style_content": state.get("style_content", ""),
            "style": state.get("style", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "body_only": state.get("body_only", ""),
            "rules_body": state.get("rules_body"),
            "characters_bodies": state.get("characters_bodies", []),
            "current_request": state.get("current_request", ""),
            "request_type": state.get("request_type", "edit_request"),
            "analysis_mode": state.get("analysis_mode", False),
            "narrative_examples": state.get("narrative_examples", ""),
            "analyzed_style_content": state.get("analyzed_style_content", ""),
            "structured_edit": structured_edit
        }
    
    except Exception as e:
        logger.error(f"❌ STYLE SUBGRAPH FORMAT: Failed to format response: {e}")
        import traceback
        logger.error(f"❌ STYLE SUBGRAPH FORMAT: Traceback: {traceback.format_exc()}")
        # Return standard error response
        error_response = AgentResponse(
            response=f"Failed to format response: {str(e)}",
            task_status="error",
            agent_type="style_editing_subgraph",
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )
        # Return standard error response using shared utility
        return create_writing_error_response(
            str(e),
            "style_editing_subgraph",
            state
        )


# ============================================
# Subgraph Builder
# ============================================

def build_style_editing_subgraph(
    checkpointer,
    llm_factory: Callable,
    get_datetime_context: Callable
) -> StateGraph:
    """
    Build style editing subgraph for integration into parent agents.
    
    This subgraph handles style guide development and editing:
    - Analysis Mode: Generate style guide from narrative examples
    - Editing Mode: Edit existing style guides with structured operations
    
    Args:
        checkpointer: LangGraph checkpointer for state persistence
        llm_factory: Function that creates LLM instances
            Signature: llm_factory(temperature: float, state: Dict[str, Any]) -> LLM
        get_datetime_context: Function that returns datetime context string
            Signature: get_datetime_context() -> str
    
    Expected state inputs:
        - query: str - User's style editing request
        - user_id: str - User identifier
        - metadata: Dict[str, Any] - Contains user_chat_model
        - messages: List[Any] - Conversation history
        - shared_memory: Dict[str, Any] - Contains active_editor with:
            - content: str - Full document (must have frontmatter type: "style")
            - filename: str - Document filename
            - frontmatter: Dict[str, Any] - Parsed frontmatter
            - cursor_offset: int - Cursor position
            - selection_start/end: int - Selection range
    
    Returns state with:
        - structured_edit: Dict[str, Any] - ManuscriptEdit with operations
        - editor_operations: List[Dict[str, Any]] - Resolved operations
        - response_text: str - Natural language summary
        - task_status: str - "complete" or "error"
        - All input state preserved
    """
    subgraph = StateGraph(Dict[str, Any])
    
    # Add nodes (some need llm_factory binding)
    subgraph.add_node("prepare_context", prepare_context_node)
    subgraph.add_node("load_references", load_references_node)
    
    # Bind llm_factory to nodes that need it
    async def detect_request_node(state):
        return await detect_request_type_node(state, llm_factory)
    
    async def generate_edit_node(state):
        return await generate_edit_plan_node(state, llm_factory, get_datetime_context)
    
    async def analyze_examples_bound(state):
        return await analyze_examples_node(state, llm_factory, get_datetime_context)
    
    subgraph.add_node("detect_request_type", detect_request_node)
    subgraph.add_node("detect_mode", detect_mode_node)
    subgraph.add_node("analyze_examples", analyze_examples_bound)
    subgraph.add_node("generate_edit_plan", generate_edit_node)
    subgraph.add_node("resolve_operations", resolve_operations_node)
    subgraph.add_node("format_response", format_response_node)
    
    # Entry point
    subgraph.set_entry_point("prepare_context")
    
    # Define edges
    subgraph.add_edge("prepare_context", "load_references")
    subgraph.add_edge("load_references", "detect_request_type")
    subgraph.add_edge("detect_request_type", "detect_mode")
    
    # Conditional: analysis mode vs editing mode
    subgraph.add_conditional_edges(
        "detect_mode",
        lambda state: "analyze_examples" if state.get("analysis_mode", False) else "generate_edit_plan",
        {
            "analyze_examples": "analyze_examples",
            "generate_edit_plan": "generate_edit_plan"
        }
    )
    
    subgraph.add_edge("analyze_examples", "generate_edit_plan")
    subgraph.add_edge("generate_edit_plan", "resolve_operations")
    subgraph.add_edge("resolve_operations", "format_response")
    subgraph.add_edge("format_response", END)
    
    return subgraph.compile(checkpointer=checkpointer)
