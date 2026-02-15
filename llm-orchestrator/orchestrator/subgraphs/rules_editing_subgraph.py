"""
Rules Editing Subgraph

Reusable subgraph for rules document editing workflows, used by Writing Assistant
when the active document has type: rules. Standalone rules_editing_agent removed;
this subgraph is the canonical implementation.

Supports rules editing with:
- Style and character reference loading (from frontmatter)
- Request type detection (question vs edit_request)
- Cross-referencing related rules (generates multiple operations)
- Bullet point preference (scannable rules)
- Universe consistency validation

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
    sanitize_ai_response_for_history,
    strip_frontmatter_block,
    slice_hash,
    build_response_text_for_question,
    build_response_text_for_edit,
    build_failed_operations_section,
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


def _find_section_bounds(content: str, section_header: str) -> Optional[Tuple[int, int, str]]:
    """
    Find the start and end bounds of a section in the rules document.
    
    Returns: (section_start, section_end, section_content) or None if not found
    
    section_start: index of the header line
    section_end: index where the next header starts or end of document
    section_content: the full section including header
    """
    try:
        # Find the header (could be ## or ###)
        header_pattern = rf'^(#{1,3})\s+{re.escape(section_header)}\s*$'
        header_match = re.search(header_pattern, content, re.MULTILINE)
        
        if not header_match:
            return None
        
        section_start = header_match.start()
        header_level = len(header_match.group(1))  # Count # marks
        
        # Find the next header of same or higher level
        next_header_pattern = rf'^#{{{1,{header_level}}}}\s+\S'
        remaining_content = content[header_match.end():]
        next_header_match = re.search(next_header_pattern, remaining_content, re.MULTILINE)
        
        if next_header_match:
            section_end = header_match.end() + next_header_match.start()
        else:
            section_end = len(content)
        
        section_content = content[section_start:section_end]
        return (section_start, section_end, section_content)
        
    except Exception as e:
        logger.error(f"Error finding section bounds for '{section_header}': {e}")
        return None


def _find_last_bullet_in_section(section_content: str) -> Optional[str]:
    """
    Find the last bullet point in a section.
    
    For multi-line bullets, returns the LAST LINE of the LAST BULLET.
    Returns None if no bullets found.
    """
    try:
        # Find all bullet points (lines starting with "- ")
        bullet_pattern = r'^-\s+.+$'
        bullets = list(re.finditer(bullet_pattern, section_content, re.MULTILINE))
        
        if not bullets:
            return None
        
        # Get the last bullet
        last_bullet_match = bullets[-1]
        last_bullet_start = last_bullet_match.start()
        
        # Find the end of this bullet (either next bullet, next header, or end of section)
        remaining = section_content[last_bullet_match.end():]
        
        # Look for next bullet or header
        next_item_pattern = r'^(?:-\s+|#{1,3}\s+)'
        next_item_match = re.search(next_item_pattern, remaining, re.MULTILINE)
        
        if next_item_match:
            # Bullet ends before the next item
            bullet_end = last_bullet_match.end() + next_item_match.start()
        else:
            # This is the last item in the section
            bullet_end = len(section_content)
        
        # Extract the full bullet text
        full_bullet = section_content[last_bullet_start:bullet_end].strip()
        
        # For multi-line bullets, get the LAST LINE
        lines = full_bullet.split('\n')
        last_line = lines[-1].strip()
        
        # Remove leading bullet marker from first line if present in single-line result
        if len(lines) == 1 and last_line.startswith('- '):
            last_line = last_line[2:].strip()
        
        logger.info(f"Found last bullet line in section: '{last_line[:80]}...'")
        return last_line
        
    except Exception as e:
        logger.error(f"Error finding last bullet in section: {e}")
        return None


def _validate_operation_for_bullet_lists(op: Dict[str, Any], content: str) -> Tuple[bool, Optional[str]]:
    """
    Validate operations for bullet lists:
    1. WARN on overly broad replace_range operations (5+ bullets, 500+ chars)
    
    This implements logging/monitoring of potentially problematic operations.
    Operations are not rejected programmatically; we rely on prompting guidance
    and only log warnings here.
    
    Args:
        op: The operation dict from the LLM
        content: The full rules document content
        
    Returns:
        Tuple of (is_valid, error_message)
        Always returns (True, None) - no operations are rejected, only warned about
    """
    try:
        op_type = op.get("op_type")
        op_text = op.get("text", "")
        original_text = op.get("original_text", "")
        
        # WARN on overly broad replace_range operations
        if op_type == "replace_range" and original_text:
            original_bullet_count = len(re.findall(r'^-\s+', original_text, re.MULTILINE))
            original_len = len(original_text)
            
            # Warn if original_text contains 5+ bullets or 500+ chars
            if original_bullet_count >= 5 or original_len >= 500:
                warning_msg = (
                    f"⚠️ OVERLY BROAD replace_range detected! "
                    f"original_text contains {original_bullet_count} bullet(s), {original_len} chars. "
                    f"This may delete more content than intended (including unrelated bullets or headings). "
                    f"BETTER APPROACH: Use replace_range with original_text matching ONLY the content being changed. "
                    f"Original agent relies on prompting guidance, not programmatic rejection."
                )
                logger.warning(warning_msg)
                # Don't reject - just warn. Original agent doesn't reject operations programmatically.
                # Rely on prompting guidance to prevent issues.
        
        return True, None
        
    except Exception as e:
        logger.error(f"Error validating operation for bullet lists: {e}")
        # On error, allow operation through (don't break functionality)
        return True, None


def _correct_anchor_for_section(op: Dict[str, Any], content: str) -> Dict[str, Any]:
    """
    Correct the anchor_text for insert_after operations to use the last bullet in the target section.
    
    This implements programmatic section boundary detection like the old rules agent had.
    
    Args:
        op: The operation dict from the LLM
        content: The full rules document content
        
    Returns:
        Corrected operation dict with updated anchor_text
    """
    try:
        op_type = op.get("op_type")
        anchor_text = op.get("anchor_text", "")
        
        # Only process insert_after operations with anchor text
        if op_type != "insert_after" or not anchor_text:
            return op
        
        # Try to detect which section the LLM is targeting based on the anchor text
        # We'll search for the section that contains the anchor text
        body_only = strip_frontmatter_block(content)
        
        # Find all section headers (## and ###)
        header_pattern = r'^(#{2,3})\s+(.+)$'
        headers = list(re.finditer(header_pattern, body_only, re.MULTILINE))
        
        # Find which section contains the anchor text
        target_section_header = None
        target_section_bounds = None
        
        for i, header_match in enumerate(headers):
            header_text = header_match.group(2).strip()
            
            # Get section bounds
            section_bounds = _find_section_bounds(body_only, header_text)
            if not section_bounds:
                continue
            
            section_start, section_end, section_content = section_bounds
            
            # Check if anchor text is in this section
            if anchor_text in section_content:
                target_section_header = header_text
                target_section_bounds = section_bounds
                break
        
        if not target_section_header or not target_section_bounds:
            # Anchor text not found in any section - let resolver handle it
            logger.warning(f"Could not find section containing anchor text: '{anchor_text[:50]}...'")
            return op
        
        # Found the target section - find the last bullet
        section_start, section_end, section_content = target_section_bounds
        last_bullet_line = _find_last_bullet_in_section(section_content)
        
        if not last_bullet_line:
            # No bullets in section - let resolver handle it
            logger.warning(f"No bullets found in section '{target_section_header}'")
            return op
        
        # Check if the current anchor is already the last bullet
        if anchor_text.strip() == last_bullet_line.strip():
            logger.info(f"✅ Anchor text is already the last bullet in section '{target_section_header}'")
            return op
        
        # Correct the anchor to be the last bullet
        logger.info(f"🔧 CORRECTING ANCHOR for section '{target_section_header}':")
        logger.info(f"   ❌ OLD anchor: '{anchor_text[:80]}...'")
        logger.info(f"   ✅ NEW anchor: '{last_bullet_line[:80]}...'")
        
        # Create corrected operation
        corrected_op = op.copy()
        corrected_op["anchor_text"] = last_bullet_line
        
        return corrected_op
        
    except Exception as e:
        logger.error(f"Error correcting anchor for section: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return original operation on error
        return op


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


def _extract_conversation_history(messages: List[Any], limit: int = 10) -> List[Dict[str, str]]:
    """Extract conversation history from LangChain messages, filtering out large data URIs"""
    try:
        history = []
        for msg in messages[-limit:]:
            if hasattr(msg, 'content'):
                role = "assistant" if hasattr(msg, 'type') and msg.type == "ai" else "user"
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if role == "assistant":
                    content = sanitize_ai_response_for_history(content)
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
    """Build system prompt for rules editing"""
    return (
        "You are a MASTER UNIVERSE ARCHITECT for RULES documents (worldbuilding, series continuity). "
        "Persona disabled. Adhere strictly to frontmatter, project Rules, and Style.\n\n"
        "**TERMINOLOGY GUIDELINES (CRITICAL)**:\n"
        "- Define CONCEPTS and CONSTRAINTS, not IN-UNIVERSE FORMAL TERMS\n"
        "- Use descriptive language that explains what happens\n"
        "  GOOD: 'Creatures can transform into their natural state when threatened'\n"
        "  BAD: 'Creatures enter their True Form when threatened'\n"
        "- Document WHAT happens and the rules governing it, not formal capitalized terminology\n"
        "  GOOD: 'Creatures enter a predatory hunting state with enhanced senses'\n"
        "  BAD: 'Creatures enter Predatory Fugue with enhanced senses'\n"
        "- Avoid creating formal terminology unless explicitly part of the user's request\n"
        "- Rules should read like a technical manual, not like in-universe lore documents\n"
        "- The fiction agent will use these rules as CONSTRAINTS, not as terms to include in prose\n\n"
        "**CRITICAL: EXPAND ON USER'S STATEMENTS, BUT DON'T INFER NEW RULES**\n"
        "You MUST work within the concepts and ideas the user provides:\n"
        "- ✅ CAN expand and elaborate on the user's statements (add detail, explain, clarify what they said)\n"
        "  * Example: User says 'Magic requires components' → You can expand to 'Magic requires physical components that are consumed during casting'\n"
        "  * Example: User says 'Time period is medieval' → You can expand to 'Time period: medieval (approximately 1000-1500 CE equivalent)'\n"
        "- ❌ CANNOT infer additional rules or concepts that the user didn't mention\n"
        "  * Example: User says 'Magic requires components' → Do NOT add 'Different spell types require different component categories' (user didn't mention spell types)\n"
        "  * Example: User says 'Magic requires components' → Do NOT add 'Components degrade over time' (user didn't mention degradation)\n"
        "  * Example: User says 'Time period is medieval' → Do NOT add 'No gunpowder exists' (user didn't mention gunpowder)\n"
        "- The key distinction: Expand/elaborate on what the user SAID, but don't add NEW rules they didn't mention\n"
        "- Use context from style guide and character profiles ONLY to ensure consistency, not to add new concepts\n"
        "\n"
        "**WHEN TO ASK QUESTIONS (Ask When Information Is Needed)**:\n"
        "- When the user's request is vague or incomplete and you need specific details to proceed\n"
        "- When multiple interpretations are possible and you need clarification on which direction to take\n"
        "- When you need to know where in the document to place content (if not clear from context)\n"
        "- When you need to know the scope or level of detail the user wants\n"
        "- When there's a conflict between existing rules and the new request that requires user decision\n"
        "\n"
        "**HOW TO ASK QUESTIONS**:\n"
        "- Provide questions in the summary field of your response\n"
        "- Use EITHER multiple choice options OR free-form questions:\n"
        "  * Multiple choice: 'Which time period should I use? A) Medieval (1000-1500 CE), B) Renaissance (1400-1600 CE), C) Industrial Revolution (1750-1850 CE), D) Other (please specify)'\n"
        "  * Free form: 'What specific details should I include about the magic system? (e.g., how magic is learned, who can use it, what are its limitations?)'\n"
        "- If you can make partial edits based on available information, do so and ask questions about the missing parts\n"
        "- If the request is completely unclear, return empty operations array and ask clarifying questions in the summary\n"
        "- Be specific about what information you need - don't ask vague questions\n\n"
        "STRUCTURED OUTPUT REQUIRED: Return ONLY raw JSON (no prose, no markdown, no code fences) matching this schema:\n"
        "{\n"
        '  "type": "ManuscriptEdit",\n'
        '  "target_filename": string,\n'
        '  "scope": one of ["paragraph", "chapter", "multi_chapter"],\n'
        '  "summary": string,\n'
        '  "chapter_index": integer|null,\n'
        '  "safety": one of ["low", "medium", "high"],\n'
        '  "operations": [ { "op_type": one of ["replace_range", "delete_range", "insert_after_heading", "insert_after"], "start": integer, "end": integer, "text": string } ]\n'
        "}\n\n"
        "OUTPUT RULES:\n"
        "- Output MUST be a single JSON object only.\n"
        "- Do NOT include triple backticks or language tags.\n"
        "- Do NOT include explanatory text before or after the JSON.\n"
        "- If asking questions/seeking clarification: Return empty operations array and put questions in summary field\n"
        "- If making edits: Return operations array with edits and brief description in summary field\n\n"
        "FORMATTING CONTRACT (RULES DOCUMENTS):\n"
        "- Never emit YAML frontmatter in operations[].text. Preserve existing frontmatter as-is.\n"
        "- Use Markdown headings and lists for the body.\n"
        "- When creating or normalizing structure, prefer this scaffold (top-level headings):\n"
        "  ## Background\n"
        "  ## Universe Constraints (physical/magical/technological laws)\n"
        "  ## Systems\n"
        "  ### Magic or Technology Systems\n"
        "  ### Resource & Economy Constraints\n"
        "  ## Social Structures & Culture\n"
        "  ### Institutions & Power Dynamics\n"
        "  ## Geography & Environment\n"
        "  ## Religion & Philosophy\n"
        "  ## Timeline & Continuity\n"
        "  ### Chronology (canonical)\n"
        "  ### Continuity Rules (no-retcon constraints)\n"
        "  ## Series Synopsis\n"
        "  ### Book 1\n"
        "  ### Book 2\n"
        "  ... (as needed)\n"
        "  ## Character References\n"
        "  ### Cast Integration & Constraints\n\n"
        "RULES FOR EDITS:\n"
        "0) **WORK FIRST, ASK LATER**: Always make edits based on available information. Use context from the request, existing rules content, style guide, and character profiles to inform your work. Only ask questions in the summary if critical information is missing that prevents meaningful progress. Never return empty operations unless the request is completely impossible.\n"
        "1) **CRITICAL: NEVER USE insert_after FOR BULLET LISTS** - This is the #1 cause of errors!\n"
        "   - **ABSOLUTE PROHIBITION**: NEVER use insert_after when adding bullets to a section with existing bullets\n"
        "   - **ALWAYS use replace_range** to expand bullet lists (replace ONLY the last bullet, not all bullets)\n"
        "   - insert_after is ONLY for continuing narrative paragraphs or prose, NEVER for bullet points\n"
        "   - WHY: insert_after will INSERT IN THE MIDDLE if you pick the wrong anchor, fragmenting the list\n"
        "   - **CRITICAL: KEEP original_text SURGICAL AND MINIMAL**:\n"
        "     * For editing ONE bullet: original_text = ONLY that bullet (COMPLETE, including all lines if multi-line)\n"
        "     * For adding bullets: original_text = ONLY THE LAST BULLET (surgical!), text = last bullet + new bullets\n"
        "     * NEVER capture 5+ bullets in original_text - this will delete adjacent content and headers!\n"
        "     * NEVER include ANY part of headers (###, ##) in original_text\n"
        "     * **COPY COMPLETE BULLET TEXT**: Multi-line bullets MUST include ALL continuation lines through final period/char\n"
        "     * **NO TRUNCATION**: original_text MUST end where the bullet actually ends, not mid-sentence!\n"
        "     * ❌ WRONG: original_text=\"- Long bullet that continues...\" (truncated mid-bullet - leaves orphan text!)\n"
        "     * ✅ CORRECT: original_text=\"- Long bullet that continues on multiple lines to completion.\" (COMPLETE)\n"
        "   - **CORRECT APPROACH**: Find ONLY THE LAST BULLET in section, copy COMPLETE TEXT to original_text, append new bullets in text field\n"
        "   - **EXAMPLE**: Section has 3 bullets → original_text=\"- Bullet 3 complete text here.\" (LAST ONLY, COMPLETE), text=\"- Bullet 3 complete text here.\\n- NEW Bullet 4\"\n"
        "   - **WRONG**: original_text=\"- Bullet 1\\n- Bullet 2\\n- Bullet 3\" (too broad - will delete following content!)\n\n"
        "2) Make focused, surgical edits near the cursor/selection unless the user requests re-organization.\n"
        "3) Maintain the scaffold above; if missing, create only the minimal sections the user asked for.\n"
        "4) Prefer paragraph/sentence-level replacements; avoid large-span rewrites unless asked.\n"
        "5) Enforce consistency: cross-check constraints against Series Synopsis and Characters.\n"
        "6) **EXPAND ON USER'S STATEMENTS, BUT DON'T INFER NEW RULES** - When the user provides concepts or ideas, you MUST:\n"
        "   - ✅ CAN expand and elaborate on what the user said (add detail, explain, clarify their statements)\n"
        "   - ❌ CANNOT infer additional rules or concepts that the user didn't mention\n"
        "   - **PREFER BULLET POINTS OVER PARAGRAPHS**: Rules should be concise and scannable. Use bullet points to break down concepts into clear, digestible points.\n"
        "   - Example: User says 'Magic requires components' → You CAN expand to:\n"
        "     * Magic requires physical components that are consumed during casting\n"
        "     * Components must be gathered or obtained before casting\n"
        "     (This expands on what they said - 'requires components' → explains what that means)\n"
        "     But do NOT add: 'Different spell types require different component categories' (user didn't mention spell types)\n"
        "   - Example: User says 'Time period is medieval' → You CAN expand to:\n"
        "     * Time period: medieval (approximately 1000-1500 CE equivalent)\n"
        "     (This expands on what they said - adds time range clarification)\n"
        "     But do NOT add: 'No gunpowder exists' or 'Feudal hierarchy' (user didn't mention these)\n"
        "   - The key: Expand/elaborate on what the user SAID, but don't add NEW rules they didn't mention\n"
        "   - If you need more information to complete the rule, ASK QUESTIONS (see question format guidelines above)\n"
        "   - Use paragraphs ONLY when the user provides narrative explanation or complex relationships\n"
        "   - The goal is to expand on the user's statements while staying within their concepts\n\n"
        "7) **CRITICAL: ORGANIZE AND CONSOLIDATE RULES** - Before adding rules, you MUST:\n"
        "   - **Check for DUPLICATES**: Scan the ENTIRE document to see if this rule already exists somewhere\n"
        "   - **Identify the BEST LOCATION**: Determine which section is most appropriate for each rule\n"
        "   - **CONSOLIDATE duplicates**: If a rule exists in multiple places, keep it in the MOST appropriate section and DELETE it from others\n"
        "   - **MOVE misplaced rules**: If a rule is in the wrong section, DELETE it from the wrong place and ADD it to the right place\n"
        "   - **Avoid redundancy**: Do NOT add the same rule to multiple sections unless there's a specific reason for cross-referencing\n"
        "   - Example: If 'Magic requires components' exists under Systems AND Universe Constraints, keep it in Systems (more specific) and delete from Universe Constraints\n"
        "   - Example: If adding a geography rule but it's already in Background section, MOVE it to Geography section (delete_range from Background, insert_after_heading in Geography)\n"
        "   - Example: If the same timeline event is in both Timeline AND Series Synopsis, consolidate - keep detailed version in Timeline, brief reference in Series Synopsis\n\n"
        "8) **CRITICAL: CROSS-REFERENCE RELATED RULES** - When adding or updating a concept, you MUST:\n"
        "   - Scan the ENTIRE document for related rules that should be updated together\n"
        "   - Identify ALL sections that reference or relate to the concept being added/updated\n"
        "   - Generate MULTIPLE operations if a single concept addition requires updates to multiple related rules\n"
        "   - Example: If adding a magic system rule, check Systems, Universe Constraints, Timeline, and Series Synopsis sections\n"
        "   - Example: If updating a character constraint, check Character References, Series Synopsis, and Continuity Rules\n"
        "   - Example: If adding a timeline event, check Chronology, Continuity Rules, and Series Synopsis for consistency\n"
        "   - NEVER update only one rule when related rules exist that should be updated together\n"
        "   - The operations array can contain MULTIPLE operations - use it to update all related sections in one pass\n\n"
        "ANCHOR REQUIREMENTS (CRITICAL):\n"
        "For EVERY operation, you MUST provide precise anchors:\n\n"
        "REVISE/DELETE Operations:\n"
        "- ALWAYS include 'original_text' with EXACT, VERBATIM text from the file\n"
        "- **CRITICAL FOR PARAGRAPH REPLACEMENT**: When replacing a paragraph with bullet points, you MUST include the COMPLETE paragraph text in 'original_text', not just a snippet!\n"
        "- Include the ENTIRE paragraph from start to finish - all sentences, all content that needs to be removed\n"
        "- Minimum 10-20 words for small edits, but for paragraph replacements: include the FULL paragraph (could be 50-200+ words)\n"
        "- Copy and paste directly - do NOT retype or modify\n"
        "- NEVER include header lines (###, ##, #) in original_text!\n"
        "- OR provide both 'left_context' and 'right_context' (exact surrounding text)\n"
        "- **Example of CORRECT paragraph replacement**: If replacing a 150-word paragraph, 'original_text' must contain all 150 words, not just the first 20 words!\n\n"
        "INSERT Operations (ONLY for truly empty sections!):\n"
        "- **insert_after_heading**: Use ONLY when section is completely empty below the header\n"
        "  * op_type='insert_after_heading' with anchor_text='## Section' (exact header line)\n"
        "  * Example: Adding rules after '## Magic System' header when section is completely empty\n"
        "  * ⚠️ CRITICAL WARNING: Before using insert_after_heading, you MUST verify the section is COMPLETELY EMPTY!\n"
        "  * ⚠️ If there is ANY text below the header (even a single line), use replace_range instead!\n"
        "  * ⚠️ Using insert_after_heading when content exists will INSERT BETWEEN the header and existing text, splitting the section!\n"
        "  * ⚠️ This creates duplicate content and breaks the section structure - NEVER do this!\n"
        "  * Example of WRONG behavior: '## Magic\\n[INSERT HERE splits section]\\n- Existing rule' ← WRONG! Use replace_range on existing content!\n"
        "  * Example of CORRECT usage: '## Magic\\n[empty - no text below]' ← OK to use insert_after_heading\n"
        "  * This is the SAFEST method - it NEVER deletes headers, always inserts AFTER them - BUT ONLY FOR EMPTY SECTIONS\n\n"
        "- **insert_after**: Use when continuing text mid-paragraph, mid-sentence, or after specific text\n"
        "  * op_type='insert_after' with anchor_text='last few words before insertion point'\n"
        "  * Example: Continuing a sentence or adding to an existing paragraph\n\n"
        "- **REPLACE Operations (PREFERRED for updating existing content!):\n"
        "- **replace_range**: Use when section exists but needs improvement, completion, or revision\n"
        "  * If section has ANY content (even incomplete or placeholder), use replace_range to update it\n"
        "  * **CRITICAL**: When converting paragraphs to bullet points, 'original_text' must include the COMPLETE paragraph (all sentences, all content to be removed)\n"
        "  * Example: Section has '- Magic requires physical components' but needs more detail → replace_range with original_text='- Magic requires physical components' and expanded text\n"
        "  * Example: Section has '[To be developed]' → replace_range with original_text='[To be developed]' and actual content\n"
        "  * Example: Converting paragraph to bullets → replace_range with original_text='[ENTIRE paragraph text from start to finish]' and text='[bullet points]'\n"
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
        "2. Section has ANY content (even incomplete/placeholder/single line)? → replace_range to update it (NO headers in original_text!)\n"
        "3. Adding to existing list/paragraph? → replace_range with original_text matching existing content\n"
        "4. Deleting SPECIFIC content? → delete_range with original_text (NO headers!)\n"
        "5. Continuing mid-sentence? → insert_after\n"
        "6. Rule exists in wrong section? → Two operations: delete_range from wrong section, then insert/replace in correct section\n"
        "7. Same rule exists in multiple sections? → Keep in most appropriate section, delete_range from others\n\n"
        "CRITICAL: When updating existing content (even if incomplete), use 'replace_range' on the existing content!\n"
        "NEVER include headers in 'original_text' for replace_range - headers will be deleted!\n"
        "⚠️ NEVER use insert_after_heading when content exists - it will SPLIT the section and create duplicates!\n"
        "\n"
        "**CORRECT EXAMPLES**:\n"
        "- Updating existing content: {\"op_type\": \"replace_range\", \"original_text\": \"- Magic requires physical components\", \"text\": \"- Magic requires physical components\\n- Components must be consumed during casting\"}\n"
        "- Moving rule to better section: [{\"op_type\": \"delete_range\", \"original_text\": \"- Magic uses verbal components\"}, {\"op_type\": \"insert_after_heading\", \"anchor_text\": \"## Magic Systems\", \"text\": \"- Magic uses verbal components\\n- Verbal components must be spoken clearly\"}]\n"
        "- Consolidating duplicates: [{\"op_type\": \"delete_range\", \"original_text\": \"- Dragons are rare creatures\" (from Background)}, {\"op_type\": \"replace_range\", \"original_text\": \"- Dragons exist\", \"text\": \"- Dragons are rare creatures found in mountain regions\" (in Geography)}]\n"
        "\n"
        "**WRONG EXAMPLES**:\n"
        "- ❌ {\"op_type\": \"insert_after_heading\"} when section has content - will split section!\n"
        "- ❌ Adding same rule to multiple sections without consolidating - creates duplicates!\n"
        "- ❌ Not checking if rule already exists elsewhere - creates redundancy!\n\n"
        "=== SPACING RULES (CRITICAL - READ CAREFULLY!) ===\n"
        "YOUR TEXT MUST END IMMEDIATELY AFTER THE LAST CHARACTER!\n\n"
        'CORRECT: "- Rule 1\\n- Rule 2\\n- Rule 3"  ← Ends after "3" with NO \\n\n'
        'WRONG: "- Rule 1\\n- Rule 2\\n"  ← Extra \\n after last line creates blank line!\n'
        'WRONG: "- Rule 1\\n- Rule 2\\n\\n"  ← \\n\\n creates 2 blank lines!\n'
        'WRONG: "- Rule 1\\n\\n- Rule 2"  ← Double \\n\\n between items creates blank line!\n\n'
        "IRON-CLAD RULE: After last line = ZERO \\n (nothing!)\n"
        "5) Headings must be clear; do not duplicate sections. If an equivalent heading exists, update it in place.\n"
        "6) When adding Timeline & Continuity entries, keep a chronological order and explicit constraints (MUST/MUST NOT).\n"
        "7) When adding Series Synopsis entries, keep book-by-book bullets with continuity notes.\n"
        "8) NO PLACEHOLDER FILLERS: If a requested section has no content yet, create the heading only and leave the body blank. Do NOT insert placeholders like '[To be developed]' or 'TBD'.\n"
    )


# ============================================
# Subgraph Nodes
# ============================================

async def prepare_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare context: extract editor info (no type gating - parent handles routing)"""
    try:
        logger.info("Preparing context for rules editing...")
        
        shared_memory = state.get("shared_memory", {}) or {}
        active_editor = shared_memory.get("active_editor", {}) or {}
        
        rules = active_editor.get("content", "") or ""
        filename = active_editor.get("filename") or "rules.md"
        frontmatter = active_editor.get("frontmatter", {}) or {}
        cursor_offset = int(active_editor.get("cursor_offset", -1))
        selection_start = int(active_editor.get("selection_start", -1))
        selection_end = int(active_editor.get("selection_end", -1))
        
        # Extract user request
        messages = state.get("messages", [])
        try:
            if messages:
                latest_message = messages[-1]
                current_request = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
            else:
                current_request = state.get("query", "")
        except Exception:
            current_request = ""
        
        # Get paragraph bounds
        normalized_text = rules.replace("\r\n", "\n")
        body_only = strip_frontmatter_block(normalized_text)
        
        return {
            "active_editor": active_editor,
            "rules": normalized_text,
            "filename": filename,
            "frontmatter": frontmatter,
            "cursor_offset": cursor_offset,
            "selection_start": selection_start,
            "selection_end": selection_end,
            "body_only": body_only,
            "current_request": current_request.strip(),
            # ✅ CRITICAL: Preserve state for subsequent nodes
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to prepare context: {e}")
        return {
            "error": str(e),
            "task_status": "error",
            # ✅ CRITICAL: Preserve state even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


async def load_references_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load referenced context files (style, characters) directly from rules frontmatter"""
    try:
        logger.info("Loading referenced context files from rules frontmatter...")
        
        # Use shared utility for reference loading
        result = await load_writing_references(
            state=state,
            reference_config={
                "style": ["style"],
                "characters": ["characters", "character_*"]
            },
            cascade_config=None,  # No cascading for rules
            doc_type_filter="rules"
        )
        
        # Check for errors
        if result.get("error"):
            return result
        
        # Content is already stripped by load_writing_references
        # characters_bodies is handled by the utility (returns list)
        
        # Preserve rules-specific context
        result.update({
            "active_editor": state.get("active_editor", {}),
            "rules": state.get("rules", ""),
            "filename": state.get("filename", "rules.md"),
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
            "style_body": None,
            "characters_bodies": [],
            "error": str(e),
            # ✅ CRITICAL: Preserve state even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve rules-specific context
            "active_editor": state.get("active_editor", {}),
            "rules": state.get("rules", ""),
            "filename": state.get("filename", "rules.md"),
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
        style_body = state.get("style_body")
        characters_bodies = state.get("characters_bodies", [])
        
        # Build simple prompt for LLM to determine intent
        prompt = f"""Analyze the user's request and determine if it's a QUESTION or an EDIT REQUEST.

**USER REQUEST**: {current_request}

**CONTEXT**:
- Current rules: {body_only[:500] if body_only else "Empty rules"}
- Has style reference: {bool(style_body)}
- Has {len(characters_bodies)} character reference(s)

**INTENT DETECTION**:
- QUESTIONS (including pure questions and conditional edits): User is asking a question - may or may not want edits
  - Pure questions: "What rules are defined?", "Do we have a rule about magic?", "Show me the worldbuilding rules", "What's our time period setting?"
  - Conditional edits: "Do we have magic rules? Add them if not", "What rules? Suggest additions if needed"
  - Questions often start with: "Do you", "What", "Can you", "Are there", "How many", "Show me", "Is", "Does", "Are we", "Suggest"
  - **Key insight**: Questions can be answered, and IF edits are needed based on the answer, they can be made
  - Route ALL questions to edit path - LLM can decide if edits are needed
  
- EDIT REQUESTS: User wants to create, modify, or generate content - NO question asked
  - Examples: "Add magic system rules", "Create worldbuilding rules", "Update the time period section", "Revise the geography rules"
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
                # ✅ CRITICAL: Preserve rules-specific context
                "active_editor": state.get("active_editor", {}),
                "rules": state.get("rules", ""),
                "filename": state.get("filename", "rules.md"),
                "frontmatter": state.get("frontmatter", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "body_only": state.get("body_only", ""),
                "current_request": state.get("current_request", ""),
                "style_body": state.get("style_body"),
                "characters_bodies": state.get("characters_bodies", [])
            }
        except Exception as e:
            logger.warning(f"Failed to parse request type detection: {e}, defaulting to edit_request")
            return {
                "request_type": "edit_request",
                **preserve_critical_state(state),
                # ✅ CRITICAL: Preserve rules-specific context
                "active_editor": state.get("active_editor", {}),
                "rules": state.get("rules", ""),
                "filename": state.get("filename", "rules.md"),
                "frontmatter": state.get("frontmatter", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "body_only": state.get("body_only", ""),
                "current_request": state.get("current_request", ""),
                "style_body": state.get("style_body"),
                "characters_bodies": state.get("characters_bodies", [])
            }
        
    except Exception as e:
        logger.error(f"Failed to detect request type: {e}")
        return {
            "request_type": "edit_request",
            # ✅ CRITICAL: Preserve all state even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve rules-specific context
            "active_editor": state.get("active_editor", {}),
            "rules": state.get("rules", ""),
            "filename": state.get("filename", "rules.md"),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "body_only": state.get("body_only", ""),
            "current_request": state.get("current_request", ""),
            "style_body": state.get("style_body"),
            "characters_bodies": state.get("characters_bodies", [])
        }


async def generate_edit_plan_node(
    state: Dict[str, Any],
    llm_factory: Callable,
    get_datetime_context: Callable
) -> Dict[str, Any]:
    """Generate edit plan using LLM"""
    try:
        logger.info("Generating rules edit plan...")
        
        rules = state.get("rules", "")
        filename = state.get("filename", "rules.md")
        body_only = state.get("body_only", "")
        current_request = state.get("current_request", "")
        request_type = state.get("request_type", "edit_request")
        is_question = request_type == "question"
        
        style_body = state.get("style_body")
        characters_bodies = state.get("characters_bodies", [])
        
        selection_start = state.get("selection_start", -1)
        selection_end = state.get("selection_end", -1)
        
        # Build system prompt
        system_prompt = _build_system_prompt()
        
        # Build context message
        context_parts = [
            "=== RULES FILE CONTENT (COPY ANCHOR TEXT FROM HERE!) ===\n\n",
            f"**CRITICAL**: All 'anchor_text' and 'original_text' MUST be copied EXACTLY from the text below.\n"
            f"Do a mental Ctrl+F search in the text below to verify your anchor exists!\n\n",
            f"File: {filename}\n\n",
            "=== START OF RULES CONTENT ===\n" + body_only + "\n=== END OF RULES CONTENT ===\n\n",
            "**REMINDER**: Copy anchor_text/original_text VERBATIM from between START/END markers above!\n\n"
        ]
        
        if style_body:
            context_parts.append(f"=== STYLE GUIDE ===\n{style_body}\n\n")
        
        if characters_bodies:
            context_parts.append("".join([f"=== CHARACTER DOC ===\n{b}\n\n" for b in characters_bodies]))
        
        # Add mode-specific instructions
        if is_question:
            context_parts.append(
                "\n=== QUESTION REQUEST: ANALYZE AND OPTIONALLY EDIT ===\n"
                "The user has asked a question about the rules document.\n\n"
                "**YOUR TASK**:\n"
                "1. **ANALYZE FIRST**: Answer the user's question by evaluating the current content\n"
                "   - Pure questions: 'What rules are defined?' → Report current rules\n"
                "   - Verification questions: 'Do we have a rule about magic?' → Check for rule, report findings\n"
                "   - Suggestion questions: 'Suggest additions to the rules' → Analyze current content, then suggest additions\n"
                "   - Conditional questions: 'Do we have magic rules? Add them if not' → Check, then edit if needed\n"
                "2. **THEN EDIT IF NEEDED**: Based on your analysis, make edits if necessary\n"
                "   - If question implies a desired state ('Add them if not') → Provide editor operations\n"
                "   - If question asks for suggestions ('Suggest additions') → Provide editor operations with suggested additions\n"
                "   - If question is pure information ('What rules?') → No edits needed, just answer\n"
                "   - Include your analysis in the 'summary' field of your response\n\n"
                "**RESPONSE FORMAT**:\n"
                "- In the 'summary' field: Answer the question clearly and explain your analysis\n"
                "- In the 'operations' array: Provide editor operations ONLY if edits are needed\n"
                "- If no edits needed: Return empty operations array, but answer the question in summary\n"
                "- If edits needed: Provide operations AND explain what you found in summary\n\n"
            )
        else:
            # Edit request mode - add "EXPAND BUT DON'T INFER" guidance
            context_parts.append(
                "\n=== EDIT REQUEST: EXPAND ON USER'S STATEMENTS, BUT DON'T INFER NEW RULES ===\n"
                "The user wants you to add or revise rules content.\n\n"
                "**YOUR APPROACH**:\n"
                "1. **EXPAND ON STATEMENTS**: You CAN expand and elaborate on what the user said (add detail, explain, clarify)\n"
                "2. **DON'T INFER NEW RULES**: You CANNOT infer additional rules or concepts that the user didn't mention\n"
                "3. **ASK WHEN NEEDED**: If you need specific details to complete the rule, ask questions in the summary (use multiple choice or free form)\n"
                "4. **PARTIAL EDITS OK**: If you can make partial edits based on available information, do so and ask questions about the missing parts\n"
                "5. **EMPTY OPERATIONS IF UNCLEAR**: If the request is completely unclear, return empty operations array and ask clarifying questions in the summary\n\n"
            )
            context_parts.append("Provide a ManuscriptEdit JSON plan for the rules document.")
        
        # Build request with mode-specific instructions
        request_with_instructions = ""
        if current_request:
            if is_question:
                request_with_instructions = (
                    f"USER REQUEST: {current_request}\n\n"
                    "**QUESTION MODE**: Answer the question first, then provide edits if needed.\n\n"
                    "CRITICAL: EXPAND ON USER'S STATEMENTS, BUT DON'T INFER NEW RULES (if edits are needed)\n"
                    "When the user provides concepts or ideas, you MUST:\n"
                    "- ✅ CAN expand and elaborate on what the user said (add detail, explain, clarify their statements)\n"
                    "- ❌ CANNOT infer additional rules or concepts that the user didn't mention\n"
                    "- **PREFER BULLET POINTS**: Use concise bullet points rather than long paragraphs. Rules should be scannable and easy to reference.\n"
                    "- Example: User says 'Magic requires components' → You CAN expand to 'Magic requires physical components that are consumed during casting' (elaborates on what they said)\n"
                    "- Example: User says 'Magic requires components' → You CANNOT add 'Different spell types require different component categories' (user didn't mention spell types)\n"
                    "- The key: Expand/elaborate on what the user SAID, but don't add NEW rules they didn't mention\n"
                    "- If you need more information to complete the rule, ASK QUESTIONS in the summary field (see question format guidelines)\n"
                    "\n"
                    "CRITICAL: ORGANIZE AND CONSOLIDATE RULES FIRST (if edits are needed)\n"
                    "Before adding ANY new content, you MUST:\n"
                    "1. **CHECK FOR DUPLICATES** - Does this rule already exist somewhere in the document?\n"
                    "2. **IDENTIFY BEST LOCATION** - Which section is most appropriate for this rule?\n"
                    "3. **CONSOLIDATE IF NEEDED** - If rule exists in multiple places, keep it in the MOST appropriate section and DELETE from others\n"
                    "4. **MOVE MISPLACED RULES** - If rule is in wrong section, DELETE from wrong place and ADD to right place\n"
                    "\n"
                    "CRITICAL: CROSS-REFERENCE RELATED RULES (if edits are needed)\n"
                    "After organizing, identify related rules:\n"
                    "1. **SCAN THE ENTIRE DOCUMENT** - Read through ALL sections to identify related rules\n"
                    "2. **IDENTIFY ALL AFFECTED SECTIONS** - When adding/updating a concept, find ALL places it should appear\n"
                    "3. **GENERATE MULTIPLE OPERATIONS** - If a concept affects multiple rules, create operations for EACH affected section\n"
                    "4. **ENSURE CONSISTENCY** - Related rules must be updated together to maintain document coherence\n"
                    "\n"
                    "See the system prompt for full editor operation instructions.\n"
                )
            else:
                request_with_instructions = (
                    f"USER REQUEST: {current_request}\n\n"
                    "**EXPAND ON USER'S STATEMENTS, BUT DON'T INFER NEW RULES**: Make edits based on what the user provides. You CAN expand/elaborate on their statements, but CANNOT infer additional rules they didn't mention. If you need more information, ask questions in the summary.\n\n"
                    "CRITICAL: EXPAND ON USER'S STATEMENTS, BUT DON'T INFER NEW RULES\n"
                    "When the user provides concepts or ideas, you MUST:\n"
                    "- ✅ CAN expand and elaborate on what the user said (add detail, explain, clarify their statements)\n"
                    "- ❌ CANNOT infer additional rules or concepts that the user didn't mention\n"
                    "- **PREFER BULLET POINTS**: Use concise bullet points rather than long paragraphs. Rules should be scannable and easy to reference.\n"
                    "- The key distinction: Expand/elaborate on what the user SAID, but don't add NEW rules they didn't mention\n"
                    "- If you need more information to complete the rule, ASK QUESTIONS in the summary field (see question format guidelines below)\n"
                    "\n"
                    "Examples of correct expansion (elaborating on what user said):\n"
                    "- User: 'Magic requires components' → You CAN expand to:\n"
                    "  * Magic requires physical components that are consumed during casting\n"
                    "  * Components must be gathered or obtained before casting\n"
                    "  (This expands on what they said - explains what 'requires components' means)\n"
                    "- User: 'Time period is medieval' → You CAN expand to:\n"
                    "  * Time period: medieval (approximately 1000-1500 CE equivalent)\n"
                    "  (This expands on what they said - adds time range clarification)\n"
                    "\n"
                    "Examples of incorrect inference (adding rules user didn't mention):\n"
                    "- User: 'Magic requires components' → You CANNOT add:\n"
                    "  * Different spell types require different component categories (user didn't mention spell types)\n"
                    "  * Components degrade over time (user didn't mention degradation)\n"
                    "- User: 'Time period is medieval' → You CANNOT add:\n"
                    "  * No gunpowder exists (user didn't mention gunpowder)\n"
                    "  * Feudal hierarchy (user didn't mention social structure)\n"
                    "\n"
                    "**QUESTION FORMAT GUIDELINES**:\n"
                    "When you need more information, ask questions in the summary field using EITHER:\n"
                    "- Multiple choice: 'Which time period should I use? A) Medieval (1000-1500 CE), B) Renaissance (1400-1600 CE), C) Industrial Revolution (1750-1850 CE), D) Other (please specify)'\n"
                    "- Free form: 'What specific details should I include about the magic system? (e.g., how magic is learned, who can use it, what are its limitations?)'\n"
                    "- Be specific about what information you need - don't ask vague questions\n"
                    "\n"
                    "CRITICAL: ORGANIZE AND CONSOLIDATE RULES FIRST\n"
                    "Before adding ANY new content, you MUST:\n"
                    "1. **CHECK FOR DUPLICATES** - Does this rule already exist somewhere in the document?\n"
                    "2. **IDENTIFY BEST LOCATION** - Which section is most appropriate for this rule?\n"
                    "3. **CONSOLIDATE IF NEEDED** - If rule exists in multiple places, keep it in the MOST appropriate section and DELETE from others\n"
                    "4. **MOVE MISPLACED RULES** - If rule is in wrong section, DELETE from wrong place and ADD to right place\n"
                    "\n"
                    "Examples of organization operations:\n"
                    "- User adds 'Magic requires components' but it already exists in Background → DELETE from Background, ADD expanded version to Magic Systems section\n"
                    "- Same timeline event in Timeline AND Series Synopsis → Keep detailed version in Timeline, brief reference in Series Synopsis\n"
                    "- Geography rule in Background section → MOVE to Geography section (delete from Background, insert in Geography)\n"
                    "\n"
                    "CRITICAL: CROSS-REFERENCE RELATED RULES\n"
                    "After organizing, identify related rules:\n"
                    "1. **SCAN THE ENTIRE DOCUMENT** - Read through ALL sections to identify related rules\n"
                    "2. **IDENTIFY ALL AFFECTED SECTIONS** - When adding/updating a concept, find ALL places it should appear\n"
                    "3. **GENERATE MULTIPLE OPERATIONS** - If a concept affects multiple rules, create operations for EACH affected section\n"
                    "4. **ENSURE CONSISTENCY** - Related rules must be updated together to maintain document coherence\n"
                    "\n"
                    "Examples of when to generate multiple operations:\n"
                    "- Adding magic system → Update 'Magic Systems' section AND 'Universe Constraints' section if they reference each other\n"
                    "- Adding character constraint → Update 'Character References' AND 'Series Synopsis' if character appears there\n"
                    "- Adding timeline event → Update 'Chronology' AND 'Continuity Rules' AND 'Series Synopsis' if event affects plot\n"
                    "- Updating a concept → If concept appears in multiple sections, update ALL occurrences, not just one\n"
                    "\n"
                    "See the system prompt for full editor operation instructions.\n"
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
        
        # Call LLM (moderate temperature for accurate documentation)
        llm = llm_factory(temperature=0.5, state=state)
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
                raw.setdefault("summary", "Planned rules edit generated from context.")
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
                # ✅ CRITICAL: Preserve all state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                # ✅ CRITICAL: Preserve rules-specific context
                "active_editor": state.get("active_editor", {}),
                "rules": state.get("rules", ""),
                "filename": state.get("filename", "rules.md"),
                "frontmatter": state.get("frontmatter", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "body_only": state.get("body_only", ""),
                "current_request": state.get("current_request", ""),
                "request_type": state.get("request_type", "edit_request"),
                "style_body": state.get("style_body"),
                "characters_bodies": state.get("characters_bodies", [])
            }
        
        if structured_edit is None:
            return {
                "llm_response": content,
                "structured_edit": None,
                "error": "Failed to produce a valid Rules edit plan. Ensure ONLY raw JSON ManuscriptEdit with operations is returned.",
                "task_status": "error",
                # ✅ CRITICAL: Preserve all state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                # ✅ CRITICAL: Preserve rules-specific context
                "active_editor": state.get("active_editor", {}),
                "rules": state.get("rules", ""),
                "filename": state.get("filename", "rules.md"),
                "frontmatter": state.get("frontmatter", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "body_only": state.get("body_only", ""),
                "current_request": state.get("current_request", ""),
                "request_type": state.get("request_type", "edit_request"),
                "style_body": state.get("style_body"),
                "characters_bodies": state.get("characters_bodies", [])
            }
        
        # Log what we got from the LLM
        ops_count = len(structured_edit.get("operations", [])) if structured_edit else 0
        logger.info(f"LLM generated {ops_count} operation(s)")
        if ops_count > 0:
            for i, op in enumerate(structured_edit.get("operations", [])):
                op_type = op.get("op_type", "unknown")
                text_preview = (op.get("text", "") or "")[:100]
                logger.info(f"  Operation {i+1}: {op_type}, text preview: {text_preview}...")
        
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
            # ✅ CRITICAL: Preserve rules-specific context
            "active_editor": state.get("active_editor", {}),
            "rules": state.get("rules", ""),
            "filename": state.get("filename", "rules.md"),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "body_only": state.get("body_only", ""),
            "current_request": state.get("current_request", ""),
            "request_type": state.get("request_type", "edit_request"),
            "style_body": state.get("style_body"),
            "characters_bodies": state.get("characters_bodies", [])
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
            # ✅ CRITICAL: Preserve rules-specific context
            "active_editor": state.get("active_editor", {}),
            "rules": state.get("rules", ""),
            "filename": state.get("filename", "rules.md"),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "body_only": state.get("body_only", ""),
            "current_request": state.get("current_request", ""),
            "request_type": state.get("request_type", "edit_request"),
            "style_body": state.get("style_body"),
            "characters_bodies": state.get("characters_bodies", [])
        }


async def resolve_operations_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve editor operations with progressive search"""
    try:
        logger.info("Resolving editor operations...")
        
        rules = state.get("rules", "")
        structured_edit = state.get("structured_edit")
        selection_start = state.get("selection_start", -1)
        selection_end = state.get("selection_end", -1)
        current_request = state.get("current_request", "")
        
        if not structured_edit or not isinstance(structured_edit.get("operations"), list):
            return {
                "editor_operations": [],
                "failed_operations": [],
                "error": "No operations to resolve",
                "task_status": "error",
                # ✅ CRITICAL: Preserve all state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                # ✅ CRITICAL: Preserve rules-specific context
                "active_editor": state.get("active_editor", {}),
                "rules": state.get("rules", ""),
                "filename": state.get("filename", "rules.md"),
                "frontmatter": state.get("frontmatter", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "body_only": state.get("body_only", ""),
                "current_request": state.get("current_request", ""),
                "request_type": state.get("request_type", "edit_request"),
                "style_body": state.get("style_body"),
                "characters_bodies": state.get("characters_bodies", []),
                "structured_edit": state.get("structured_edit")
            }
        
        fm_end_idx = _frontmatter_end_index(rules)
        selection = {"start": selection_start, "end": selection_end} if selection_start >= 0 and selection_end >= 0 else None
        
        # Check if file is empty (only frontmatter)
        body_only = strip_frontmatter_block(rules)
        is_empty_file = not body_only.strip()
        
        # Check revision mode
        revision_mode = current_request and any(k in current_request.lower() for k in ["revise", "revision", "tweak", "adjust", "polish", "tighten", "edit only"])
        
        editor_operations = []
        failed_operations = []
        operations = structured_edit.get("operations", [])
        
        logger.info(f"Resolving {len(operations)} operation(s) from structured_edit")
        
        for op in operations:
            # ✅ PROGRAMMATIC ENFORCEMENT: Validate operation doesn't violate bullet list rules
            is_valid, validation_error = _validate_operation_for_bullet_lists(op, rules)
            if not is_valid:
                # Reject this operation - it violates the bullet list prohibition
                logger.error(f"❌ REJECTING OPERATION: {validation_error}")
                failed_operations.append({
                    "op_type": op.get("op_type", "edit"),
                    "original_text": op.get("original_text"),
                    "anchor_text": op.get("anchor_text"),
                    "text": op.get("text", ""),
                    "error": validation_error
                })
                continue
            
            # ✅ PROGRAMMATIC SECTION DETECTION: Correct anchor_text for insert_after operations
            # This ensures insertions go to the END of the target section, not the middle
            op = _correct_anchor_for_section(op, rules)
            
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
                    content=rules,
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
                
                # Validate operation safety - warn about potential over-broad matches
                original_text = op.get("original_text", "")
                if original_text and op.get("op_type") in ("replace_range", "delete_range"):
                    # Check for overly large original_text (likely to include unrelated content)
                    if len(original_text) > 300:
                        logger.warning(
                            f"⚠️ LARGE original_text ({len(original_text)} chars) - may replace more than intended. "
                            f"Preview: {original_text[:100]}..."
                        )
                    
                    # Check for multiple bullet points (likely to delete adjacent points)
                    bullet_count = original_text.count("\n-") + (1 if original_text.strip().startswith("-") else 0)
                    if bullet_count > 2:
                        logger.warning(
                            f"⚠️ original_text contains {bullet_count} bullet points - may delete adjacent unrelated points. "
                            f"Consider using smaller match for single bullet."
                        )
                    
                    # Check for section headers (should never be in original_text for replace_range)
                    if re.search(r'\n#{2,3}\s+\w', original_text):
                        logger.error(
                            f"❌ CRITICAL: original_text contains section headers! This will DELETE the headers. "
                            f"Use insert_after_heading instead of replace_range for adding content below headers."
                        )
                
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
                
                resolved_start = max(0, min(len(rules), resolved_start))
                resolved_end = max(resolved_start, min(len(rules), resolved_end))
                
                # Handle spacing for inserts
                if resolved_start == resolved_end:
                    left_tail = rules[max(0, resolved_start-2):resolved_start]
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
                
                # Check if resolution failed (-1, -1) - some cases might return this
                if resolved_start == -1 and resolved_end == -1:
                    logger.error(f"Operation resolution FAILED - original_text or anchor_text not found")
                    failed_operations.append({
                        "op_type": op.get("op_type", "edit"),
                        "original_text": op.get("original_text"),
                        "anchor_text": op.get("anchor_text"),
                        "text": op.get("text", ""),
                        "error": "Anchor or original text not found"
                    })
                    continue
                
                # Calculate pre_hash
                pre_slice = rules[resolved_start:resolved_end]
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
                # Collect failed operation
                failed_operations.append({
                    "op_type": op.get("op_type", "edit"),
                    "original_text": op.get("original_text"),
                    "anchor_text": op.get("anchor_text"),
                    "text": op.get("text", ""),
                    "error": str(e)
                })
                continue
        
        logger.info(f"Successfully resolved {len(editor_operations)} operation(s) out of {len(operations)}")
        
        return {
            "editor_operations": editor_operations,
            "failed_operations": failed_operations,
            # ✅ CRITICAL: Preserve all state
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve rules-specific context
            "active_editor": state.get("active_editor", {}),
            "rules": state.get("rules", ""),
            "filename": state.get("filename", "rules.md"),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "body_only": state.get("body_only", ""),
            "current_request": state.get("current_request", ""),
            "request_type": state.get("request_type", "edit_request"),
            "style_body": state.get("style_body"),
            "characters_bodies": state.get("characters_bodies", []),
            "structured_edit": state.get("structured_edit")
        }
        
    except Exception as e:
        logger.error(f"Failed to resolve operations: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "editor_operations": [],
            "failed_operations": [],
            "error": str(e),
            "task_status": "error",
            # ✅ CRITICAL: Preserve all state even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve rules-specific context
            "active_editor": state.get("active_editor", {}),
            "rules": state.get("rules", ""),
            "filename": state.get("filename", "rules.md"),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "body_only": state.get("body_only", ""),
            "current_request": state.get("current_request", ""),
            "request_type": state.get("request_type", "edit_request"),
            "style_body": state.get("style_body"),
            "characters_bodies": state.get("characters_bodies", []),
            "structured_edit": state.get("structured_edit")
        }


async def format_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Format final response with editor operations"""
    try:
        logger.info("📝 RULES SUBGRAPH FORMAT: Formatting rules editing response...")
        
        structured_edit = state.get("structured_edit")
        editor_operations = state.get("editor_operations", [])
        current_request = state.get("current_request", "")
        request_type = state.get("request_type", "edit_request")
        task_status = state.get("task_status", "complete")
        
        # Normalize task_status to valid enum value
        if task_status not in ["complete", "incomplete", "permission_required", "error"]:
            logger.warning(f"⚠️ RULES SUBGRAPH FORMAT: Invalid task_status '{task_status}', normalizing to 'complete'")
            task_status = "complete"
        
        if not structured_edit:
            error = state.get("error", "Unknown error")
            logger.error(f"❌ RULES SUBGRAPH FORMAT: No structured_edit found: {error}")
            error_response = AgentResponse(
                response=f"Failed to generate rules edit plan: {error}",
                task_status="error",
                agent_type="rules_editing_subgraph",
                timestamp=datetime.now().isoformat(),
                error=error
            )
            return {
                "response": error_response.dict(exclude_none=True),
                "task_status": "error",
                # ✅ CRITICAL: Preserve all state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        # Build preview text from operations (for logging purposes)
        preview = "\n\n".join([op.get("text", "").strip() for op in editor_operations if op.get("text", "").strip()])
        
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
                failed_section = build_failed_operations_section(failed_operations, "rules")
                response_text = response_text + failed_section
        
        logger.info(f"📊 RULES SUBGRAPH FORMAT: Response formatting: {len(editor_operations)} operation(s), response_text: {response_text[:200]}...")
        
        # Build manuscript_edit metadata using shared utility
        manuscript_edit_metadata = create_manuscript_edit_metadata(structured_edit, editor_operations)
        
        # Build standard response using AgentResponse contract (WITHOUT editor_operations/manuscript_edit)
        logger.info(f"📊 RULES SUBGRAPH FORMAT: Creating AgentResponse with task_status='{task_status}', {len(editor_operations)} operation(s)")
        standard_response = AgentResponse(
            response=response_text,
            task_status=task_status,
            agent_type="rules_editing_subgraph",
            timestamp=datetime.now().isoformat()
            # NO editor_operations, NO manuscript_edit (they go at state level)
        )
        
        logger.info(f"📊 RULES SUBGRAPH FORMAT: Response text length: {len(response_text)} chars")
        logger.info(f"📊 RULES SUBGRAPH FORMAT: Editor operations: {len(editor_operations)} operation(s)")
        logger.info(f"📊 RULES SUBGRAPH FORMAT: Failed operations: {len(failed_operations)} operation(s)")
        logger.info(f"📊 RULES SUBGRAPH FORMAT: Manuscript edit: {'present' if manuscript_edit_metadata else 'missing'}")
        
        logger.info(f"📤 RULES SUBGRAPH FORMAT: Returning standard AgentResponse with {len(editor_operations)} editor operation(s)")
        
        # DEBUG: Log exactly what we're returning
        return_dict = {
            "response": standard_response.dict(exclude_none=True),
            "editor_operations": editor_operations,  # STATE LEVEL (primary source)
            "manuscript_edit": manuscript_edit_metadata.dict(exclude_none=True) if manuscript_edit_metadata else None,  # STATE LEVEL
            "task_status": task_status,
            **preserve_critical_state(state)
        }
        
        logger.info(f"📤 RULES SUBGRAPH FORMAT: Return dict keys: {list(return_dict.keys())}")
        logger.info(f"📤 RULES SUBGRAPH FORMAT: editor_operations in return: {'editor_operations' in return_dict}")
        logger.info(f"📤 RULES SUBGRAPH FORMAT: editor_operations value type: {type(return_dict.get('editor_operations'))}, length: {len(return_dict.get('editor_operations', []))}")
        logger.info(f"📤 RULES SUBGRAPH FORMAT: manuscript_edit in return: {'manuscript_edit' in return_dict}")
        
        return return_dict
        
    except Exception as e:
        logger.error(f"❌ RULES SUBGRAPH FORMAT: Failed to format response: {e}")
        import traceback
        logger.error(f"❌ RULES SUBGRAPH FORMAT: Traceback: {traceback.format_exc()}")
        # Return standard error response using shared utility
        return create_writing_error_response(
            str(e),
            "rules_editing_subgraph",
            state
        )


# ============================================
# Subgraph Builder
# ============================================

def build_rules_editing_subgraph(
    checkpointer,
    llm_factory: Callable,
    get_datetime_context: Callable
) -> StateGraph:
    """
    Build rules editing subgraph for integration into parent agents.
    
    This subgraph handles rules document editing:
    - Style and character reference loading (from frontmatter)
    - Request classification: Distinguishes questions from edit requests
    - Cross-referencing: Generates multiple operations for related rules
    - Bullet point preference: Rules use concise bullet lists
    
    Args:
        checkpointer: LangGraph checkpointer for state persistence
        llm_factory: Function that creates LLM instances
            Signature: llm_factory(temperature: float, state: Dict[str, Any]) -> LLM
        get_datetime_context: Function that returns datetime context string
            Signature: get_datetime_context() -> str
    
    Expected state inputs:
        - query: str - User's rules editing request
        - user_id: str - User identifier
        - metadata: Dict[str, Any] - Contains user_chat_model
        - messages: List[Any] - Conversation history
        - shared_memory: Dict[str, Any] - Contains active_editor with:
            - content: str - Full document (must have frontmatter type: "rules")
            - filename: str - Document filename
            - frontmatter: Dict[str, Any] - Parsed frontmatter
            - cursor_offset: int - Cursor position
            - selection_start/end: int - Selection range
    
    Returns state with:
        - response: Dict[str, Any] - Formatted response with messages and agent_results
        - editor_operations: List[Dict[str, Any]] - Resolved operations
        - task_status: str - "complete", "error"
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
    
    subgraph.add_node("detect_request_type", detect_request_node)
    subgraph.add_node("generate_edit_plan", generate_edit_node)
    subgraph.add_node("resolve_operations", resolve_operations_node)
    subgraph.add_node("format_response", format_response_node)
    
    # Entry point
    subgraph.set_entry_point("prepare_context")
    
    # Define edges
    subgraph.add_edge("prepare_context", "load_references")
    subgraph.add_edge("load_references", "detect_request_type")
    subgraph.add_edge("detect_request_type", "generate_edit_plan")
    subgraph.add_edge("generate_edit_plan", "resolve_operations")
    subgraph.add_edge("resolve_operations", "format_response")
    subgraph.add_edge("format_response", END)
    
    return subgraph.compile(checkpointer=checkpointer)
