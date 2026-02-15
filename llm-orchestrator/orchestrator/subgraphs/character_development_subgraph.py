"""
Character Development Subgraph

Reusable subgraph for character document editing workflows.
Encapsulates all functionality from character_development_agent for integration into multiple agents.

Supports character development with:
- Multi-character reference loading (other character files from frontmatter)
- Request type detection (question vs edit_request)
- Cross-referencing related sections (generates multiple operations)
- Bullet point enforcement (never paragraphs)
- Universe rules integration
- Relationship consistency validation

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
from orchestrator.utils.writing_subgraph_utilities import sanitize_ai_response_for_history

logger = logging.getLogger(__name__)


# ============================================
# Utility Functions
# ============================================

def _slice_hash(text: str) -> str:
    """Match frontend sliceHash: 32-bit rolling hash to hex string."""
    try:
        h = 0
        for ch in text:
            h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        return format(h, 'x')
    except Exception:
        return ""


def _strip_frontmatter_block(text: str) -> str:
    """Strip only the leading YAML frontmatter block from text. Uses \\A so scene breaks (---) in the body are never stripped."""
    try:
        return re.sub(r'\A---\s*\n[\s\S]*?\n---\s*\n', '', text)
    except Exception:
        return text


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
    Find the start and end bounds of a section in the character document.
    
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
    The original character_development_agent.py does NOT reject operations programmatically,
    it relies entirely on prompting guidance. We only log warnings here.
    
    Args:
        op: The operation dict from the LLM
        content: The full character document content
        
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
    
    This implements programmatic section boundary detection like the rules agent has.
    
    Args:
        op: The operation dict from the LLM
        content: The full character document content
        
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
        body_only = _strip_frontmatter_block(content)
        
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


def _build_failed_operations_section(failed_operations: List[Dict[str, Any]]) -> str:
    """
    Build a human-readable section describing failed operations.
    
    Args:
        failed_operations: List of failed operation dicts with error messages
        
    Returns:
        Formatted string describing the failed operations
    """
    if not failed_operations:
        return ""
    
    section = f"\n\n⚠️ **{len(failed_operations)} operation(s) were rejected:**\n\n"
    
    for i, failed_op in enumerate(failed_operations, 1):
        op_type = failed_op.get("op_type", "unknown")
        error = failed_op.get("error", "Unknown error")
        anchor_text = failed_op.get("anchor_text", "")
        original_text = failed_op.get("original_text", "")
        
        section += f"{i}. **{op_type}** operation rejected:\n"
        section += f"   - **Error**: {error}\n"
        
        if anchor_text:
            preview = anchor_text[:50] + "..." if len(anchor_text) > 50 else anchor_text
            section += f"   - **Anchor**: `{preview}`\n"
        
        if original_text:
            preview = original_text[:50] + "..." if len(original_text) > 50 else original_text
            section += f"   - **Original text**: `{preview}`\n"
        
        section += "\n"
    
    return section


def _extract_llm_content(response: Any) -> str:
    """
    Extract string content from an LLM response (AIMessage or similar).
    Handles content as str or as list of content blocks (e.g. LangChain multimodal).
    """
    if response is None:
        return ""
    raw = getattr(response, "content", response)
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = []
        for block in raw:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text is not None:
                    parts.append(str(text))
            else:
                parts.append(str(block))
        return "\n".join(parts) if parts else ""
    return str(raw) if raw is not None else ""


def _unwrap_json_response(content: str) -> str:
    """Extract raw JSON from LLM output if wrapped in code fences or prose."""
    if not isinstance(content, str):
        content = str(content) if content is not None else ""
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


def paragraph_bounds(text: str, cursor_offset: int) -> Tuple[int, int]:
    """Find paragraph boundaries around cursor."""
    if not text:
        return 0, 0
    cursor = max(0, min(len(text), cursor_offset))
    left = text.rfind("\n\n", 0, cursor)
    start = left + 2 if left != -1 else 0
    right = text.find("\n\n", cursor)
    end = right if right != -1 else len(text)
    return start, end


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
    """Build system prompt for character development"""
    return (
        "You are a Character Development Assistant for type: character files. Persona disabled."
        " Preserve frontmatter; write clean Markdown in body.\n\n"
        "**CRITICAL: WORK WITH AVAILABLE INFORMATION FIRST**\n"
        "Always start by working with what you know from the request, existing character content, and references:\n"
        "- Make edits based on available information - don't wait for clarification\n"
        "- Use context from outline, rules, style guide, and related characters to inform your work\n"
        "- Add or revise content based on reasonable inferences from the request\n"
        "- Only ask questions when critical information is missing that prevents you from making meaningful progress\n"
        "\n"
        "**WHEN TO ASK QUESTIONS (Rarely - Only When Truly Necessary)**:\n"
        "- Only when the request is so vague that you cannot make ANY reasonable edits (e.g., 'improve character' with no existing content)\n"
        "- Only when there's a critical conflict that requires user decision (e.g., existing trait directly contradicts new request)\n"
        "- When asking, provide operations for what you CAN do, then ask questions in the summary about what you need\n"
        "\n"
        "**HOW TO ASK QUESTIONS**: Include operations for work you CAN do, then add questions/suggestions in the summary field.\n"
        "DO NOT return empty operations array - always provide edits based on available information.\n\n"
        "STRUCTURED OUTPUT REQUIRED: Return ONLY raw JSON (no prose, no markdown, no code fences) matching this schema:\n"
        "{\n"
        "  \"type\": \"ManuscriptEdit\",\n"
        "  \"target_filename\": string,\n"
        "  \"scope\": one of [\"paragraph\", \"chapter\", \"multi_chapter\"],\n"
        "  \"summary\": string,\n"
        "  \"chapter_index\": integer|null,\n"
        "  \"safety\": one of [\"low\", \"medium\", \"high\"],\n"
        "  \"operations\": [\n"
        "    {\n"
        "      \"op_type\": one of [\"replace_range\", \"delete_range\", \"insert_after_heading\", \"insert_after\"],\n"
        "      \"start\": integer (approximate),\n"
        "      \"end\": integer (approximate),\n"
        "      \"text\": string,\n"
        "      \"original_text\": string (REQUIRED for replace/delete, optional for insert - EXACT verbatim text from file),\n"
        "      \"anchor_text\": string (optional - for inserts, exact line to insert after),\n"
        "      \"left_context\": string (optional - text before target),\n"
        "      \"right_context\": string (optional - text after target),\n"
        "      \"occurrence_index\": integer (optional, default 0 if text appears multiple times)\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "OUTPUT RULES:\n"
        "- Output MUST be a single JSON object only.\n"
        "- Do NOT include triple backticks or language tags.\n"
        "- Do NOT include explanatory text before or after the JSON.\n"
        "- Always provide operations based on available information - work with what you know\n"
        "- If you need clarification, include it in the summary field AFTER describing the work you've done\n"
        "- Never return empty operations array unless the request is completely impossible to fulfill\n\n"
        "FORMATTING CONTRACT (CHARACTER FILES):\n"
        "- Never emit YAML frontmatter in operations[].text; preserve existing YAML.\n"
        "- **CRITICAL: USE BULLET POINTS, NOT PARAGRAPHS**\n"
        "- Character profiles must be concise and scannable - use bullet lists for ALL content, never write full paragraphs\n"
        "- Each piece of information should be a separate bullet point\n"
        "- Use Markdown headings for section organization, then bullet lists for all content within sections\n"
        "- Example format:\n"
        "  ### Personality\n"
        "  - Trait: Analytical and methodical\n"
        "  - Strength: Excellent problem-solving under pressure\n"
        "  - Flaw: Tends to overthink simple decisions\n"
        "- NOT this format (avoid paragraphs):\n"
        "  ### Personality\n"
        "  The character is analytical and methodical, with excellent problem-solving skills that shine under pressure. However, they tend to overthink simple decisions...\n"
        "- Preferred major-character scaffold: Basic Information, Personality (traits/strengths/flaws), Dialogue Patterns, Internal Monologue, Relationships, Character Arc.\n"
        "- Supporting cast: concise entries (Role, Traits, Speech, Relation to MC, Notes).\n"
        "- Relationships doc: pairs with Relationship Type, Dynamics, Conflict Sources, Interaction Patterns, Evolution.\n\n"
        "**UNIVERSE RULES (if provided)**: Use for universe consistency and worldbuilding constraints\n"
        "- Rules define the world's constraints: magic systems, technology levels, social structures, geography, timeline, etc.\n"
        "- When developing character abilities, powers, or skills, ensure they align with the universe's magic/technology rules\n"
        "- When adding character background, verify it fits within the established timeline, geography, and social structures\n"
        "- When defining character affiliations or organizations, check rules for established groups, hierarchies, and power structures\n"
        "- When adding character traits that involve world elements (e.g., 'knows ancient magic'), verify against rules for magic systems\n"
        "- Use rules to inform character limitations, capabilities, and constraints within the universe\n"
        "- Example: If rules specify 'magic requires physical components', character abilities should reflect this constraint\n"
        "- Example: If rules define 'medieval technology level', character background shouldn't include modern technology\n"
        "- Example: If rules establish 'noble houses', character affiliations should reference these houses, not create new ones\n\n"
        "**CHARACTER REFERENCES (if provided)**: Use for relationship consistency and character harmony\n"
        "- Each referenced character is a DIFFERENT character with distinct traits, dialogue patterns, and behaviors\n"
        "- Check for relationship consistency (A's view of B should match B's view of A)\n"
        "- Verify trait conflicts/complementarity in relationships\n"
        "- Ensure power dynamics and hierarchies are consistent across character sheets\n"
        "- Use for comparison when user asks about character differences or relationships\n"
        "- When adding relationships, cross-reference the other character's sheet to ensure mutual consistency\n"
        "- When adding traits, consider how they contrast or complement referenced characters\n"
        "- When updating character dynamics, verify consistency with how the relationship is described in other character sheets\n\n"
        "EDIT RULES:\n"
        "0) **WORK FIRST, ASK LATER**: Always make edits based on available information. Use context from the request, existing character content, outline, rules, and related characters to inform your work. Only ask questions in the summary if critical information is missing that prevents meaningful progress. Never return empty operations unless the request is completely impossible.\n"
        "1) **BULLET POINTS ONLY**: Always format content as bullet points, never as paragraphs. Each fact, trait, or piece of information should be its own bullet point.\n"
        "   - When reformatting existing paragraph-based content (e.g., 'trim down', 'convert to bullets', 'make concise'), break paragraphs into individual bullet points\n"
        "   - Extract key facts, traits, and information from paragraphs and convert each into a separate bullet point\n"
        "   - Example: 'The character is analytical and methodical, with excellent problem-solving skills that shine under pressure. However, they tend to overthink simple decisions.'\n"
        "     → Convert to: '- Analytical and methodical\\n- Excellent problem-solving under pressure\\n- Tends to overthink simple decisions'\n"
        "2) Make surgical edits unless re-organization is requested.\n"
        "3) Maintain existing structure; update in place; avoid duplicate headings.\n"
        "4) Enforce universe consistency against Rules and outline-provided character network.\n"
        "5) NO PLACEHOLDER FILLERS: If a requested section has no content yet, create the heading only and leave the body blank. Do NOT insert placeholders like '[To be developed]' or 'TBD'.\n"
        "6) For GRANULAR REVISIONS: Use replace_range with exact original_text matching the specific text to change (e.g., 'blue eyes' → 'green eyes')\n"
        "7) NEVER insert content at the beginning of the file - always use proper anchors after frontmatter\n"
        "8) **CHECK FOR DUPLICATES IN RELATED SECTIONS** - Before adding character information:\n"
        "   - Check if similar information already exists in related sections\n"
        "   - If trait appears in both Personality AND Character Arc, consider consolidating or updating consistently\n"
        "   - If backstory detail is scattered across sections, consolidate to most appropriate location\n"
        "   - Example: 'Protective of family' in both Personality and Relationships → Update Personality (trait definition), reference in Relationships (how it affects dynamics)\n"
        "   - Avoid redundant identical information - each section should add unique perspective\n\n"
        "9) **CRITICAL: CROSS-REFERENCE RELATED SECTIONS** - When adding or updating character information, you MUST:\n"
        "   - Scan the ENTIRE document for related sections that should be updated together\n"
        "   - Identify ALL sections that reference or relate to the information being added/updated\n"
        "   - Generate MULTIPLE operations if a single addition requires updates to multiple related sections\n"
        "   - Example: If adding a personality trait, check Personality, Relationships, Character Arc, and Dialogue Patterns sections\n"
        "   - Example: If updating a relationship, check Relationships section AND any character arc notes that reference it\n"
        "   - Example: If adding a backstory detail, check Basic Information, Personality, and Character Arc sections\n"
        "   - NEVER update only one section when related sections exist that should be updated together\n"
        "   - The operations array can contain MULTIPLE operations - use it to update all related sections in one pass\n\n"
        "ANCHOR REQUIREMENTS (CRITICAL):\n"
        "For EVERY operation, you MUST provide precise anchors:\n\n"
        "REVISE/DELETE Operations:\n"
        "- ALWAYS include 'original_text' with EXACT, VERBATIM text from the file\n"
        "- For granular edits: Match the EXACT text to change (e.g., if changing 'blue' to 'green', original_text='blue')\n"
        "- For bullet point edits: Each bullet should be concise (5-15 words typically), but include enough context to be meaningful\n"
        "- When replacing existing bullet points: Match the exact original_text including the bullet marker (e.g., '- Original text here')\n"
        "- Copy and paste directly - do NOT retype or modify\n"
        "- NEVER include header lines (###, ##) in original_text!\n"
        "- NEVER target frontmatter - all operations must be after frontmatter end\n"
        "- OR provide both 'left_context' and 'right_context' (exact surrounding text)\n\n"
        "INSERT Operations (ONLY for truly empty sections!):\n"
        "- **insert_after_heading**: Use ONLY when section is completely empty below the header\n"
        "  * op_type='insert_after_heading' with anchor_text='### Header' (exact header line - ANY header the user has created)\n"
        "  * Example: Adding traits after '### Personality' header when section is completely empty\n"
        "  * Example: Adding powers after '### Special Abilities' header (custom section) when section is completely empty\n"
        "  * Works with ANY section header (### Basic Information, ### Combat Style, ### Magic Powers, ### Backstory, etc.)\n"
        "  * ⚠️ CRITICAL WARNING: Before using insert_after_heading, you MUST verify the section is COMPLETELY EMPTY!\n"
        "  * ⚠️ If there is ANY text below the header (even a single line), use replace_range instead!\n"
        "  * ⚠️ Using insert_after_heading when content exists will INSERT BETWEEN the header and existing text, splitting the section!\n"
        "  * ⚠️ This creates duplicate content and breaks the section structure - NEVER do this!\n"
        "  * Example of WRONG behavior: '### Personality\\n[INSERT HERE splits section]\\n- Existing trait' ← WRONG! Use replace_range on existing content!\n"
        "  * Example of CORRECT usage: '### Personality\\n[empty - no text below]' ← OK to use insert_after_heading\n"
        "  * This is the SAFEST method - it NEVER deletes headers, always inserts AFTER them - BUT ONLY FOR EMPTY SECTIONS\n\n"
        "- **insert_after**: Use when continuing text mid-bullet, mid-sentence, or after specific text\n"
        "  * op_type='insert_after' with anchor_text='last few words before insertion point'\n"
        "  * Example: Continuing a bullet point or adding to an existing bullet list\n"
        "  * anchor_text should be the exact text (last few words) where you want to insert after\n"
        "  * The resolver will find the end of the bullet point containing the anchor and insert there\n"
        "  * **PREFER**: Adding new bullet points rather than extending existing ones - keep bullets concise\n\n"
        "- **REPLACE Operations (PREFERRED for updating existing content!):\n"
        "- **replace_range**: Use when section exists but needs improvement, completion, or revision\n"
        "  * If section has ANY content (even incomplete or placeholder), use replace_range to update it\n"
        "  * Example: Section has '- Analytical thinker' but needs more traits → replace_range with original_text='- Analytical thinker' and expanded text\n"
        "  * Example: Section has '[To be developed]' → replace_range with original_text='[To be developed]' and actual content\n"
        "  * Example: Section has paragraph text that needs conversion to bullets → replace_range with original_text matching the entire paragraph, and new_text as bullet points\n"
        "  * When reformatting paragraphs to bullets: Match the exact paragraph text in original_text, then provide bullet points in new_text\n"
        "  * This ensures existing content is replaced/updated, not duplicated\n\n"
        "- **CRITICAL ANCHORING RULES**:\n"
        "  * Provide 'anchor_text' with EXACT, COMPLETE text to insert after (verbatim from file)\n"
        "  * Provide 'original_text' with EXACT, VERBATIM existing content to replace (verbatim from file)\n"
        "  * NEVER insert at position 0 or before frontmatter end - always use proper anchors\n"
        "  * ALTERNATIVE: Provide 'original_text' with text to insert after\n"
        "  * FALLBACK: Provide 'left_context' with text before insertion point (minimum 10-20 words)\n\n"
        "- **DECISION TREE**:\n"
        "  **STEP 1: Read the section content carefully!**\n"
        "  - Look at what exists below the header\n"
        "  - Is there ANY text at all? Even a single line?\n"
        "  \n"
        "  **STEP 2: Choose operation based on what exists:**\n"
        "  * Section is COMPLETELY EMPTY below header (no text at all)? → insert_after_heading\n"
        "  * Section has ANY content (even incomplete/placeholder/single line)? → replace_range to update it\n"
        "  * Adding to existing bullet list? → replace_range with original_text matching existing content, or add new bullet points\n"
        "  * Continuing mid-bullet or mid-sentence? → insert_after (but prefer adding new bullet points instead)\n"
        "  * Same info in multiple sections? → Update consistently or consolidate\n"
        "  * **CRITICAL**: When improving/completing existing sections, ALWAYS use replace_range to update, not insert_after_heading (which would duplicate content)\n\n"
        "NO PLACEHOLDER TEXT: Leave empty sections blank, do NOT insert '[To be developed]' or 'TBD'.\n"
    )


# ============================================
# Subgraph Nodes
# ============================================

async def prepare_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare context: extract editor info and check document type"""
    try:
        logger.info("Preparing context for character development...")
        
        shared_memory = state.get("shared_memory", {}) or {}
        active_editor = shared_memory.get("active_editor", {}) or {}
        
        text = active_editor.get("content", "") or ""
        filename = active_editor.get("filename") or "character.md"
        frontmatter = active_editor.get("frontmatter", {}) or {}
        cursor_offset = int(active_editor.get("cursor_offset", -1))
        selection_start = int(active_editor.get("selection_start", -1))
        selection_end = int(active_editor.get("selection_end", -1))
        
        # ✅ DEBUG: Log what we extract from active_editor
        logger.info(f"📊 PREPARE CONTEXT: filename='{filename}', text length={len(text)}, cursor={cursor_offset}, selection=[{selection_start}:{selection_end}]")
        logger.info(f"🔍 PREPARE CONTEXT: text preview (first 300 chars): {text[:300]}")
        if not text or len(text.strip()) == 0:
            logger.warning(f"⚠️ PREPARE CONTEXT: active_editor.content is EMPTY for file '{filename}'! File may not be loaded correctly.")
        
        # Gate by type: character (strict)
        doc_type = str(frontmatter.get("type", "")).strip().lower()
        if doc_type != "character":
            return {
                "response": {
                    "messages": [AIMessage(content="Active editor is not a character file; skipping.")],
                    "agent_results": {
                        "agent_type": "character_development_subgraph",
                        "is_complete": True,
                        "skipped": True
                    },
                    "is_complete": True
                },
                "task_status": "skipped",
                # ✅ CRITICAL: Preserve all state even when skipping
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
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
        
        return {
            "active_editor": active_editor,
            "text": text,
            "filename": filename,
            "frontmatter": frontmatter,
            "cursor_offset": cursor_offset,
            "selection_start": selection_start,
            "selection_end": selection_end,
            "current_request": current_request.strip(),
            # ✅ CRITICAL: Preserve all state
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to prepare context: {e}")
        return {
            "text": "",
            "filename": "character.md",
            "frontmatter": {},
            "error": str(e),
            "task_status": "error",
            # ✅ CRITICAL: Preserve all state even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


async def load_references_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load referenced context files (characters, style, rules) directly from character frontmatter"""
    try:
        logger.info("Loading referenced context files from character frontmatter...")
        
        from orchestrator.tools.reference_file_loader import load_referenced_files
        
        active_editor = state.get("active_editor", {})
        user_id = state.get("user_id", "system")
        
        # Character reference configuration - load directly from character's frontmatter (no cascading)
        reference_config = {
            "characters": ["characters", "character_*"],  # Other character sheets
            "style": ["style"],                           # Optional: style guide
            "rules": ["rules"]                             # Optional: world rules
        }
        
        # Use unified loader (no cascade_config - character loads directly)
        result = await load_referenced_files(
            active_editor=active_editor,
            user_id=user_id,
            reference_config=reference_config,
            doc_type_filter="character",
            cascade_config=None  # No cascading for character files
        )
        
        loaded_files = result.get("loaded_files", {})
        
        # Extract content from loaded files
        outline_body = None  # Characters don't typically reference outlines directly
        
        rules_body = None
        if loaded_files.get("rules") and len(loaded_files["rules"]) > 0:
            rules_body = loaded_files["rules"][0].get("content", "")
            if rules_body:
                rules_body = _strip_frontmatter_block(rules_body)
        
        style_text = None
        if loaded_files.get("style") and len(loaded_files["style"]) > 0:
            style_text = loaded_files["style"][0].get("content", "")
            if style_text:
                style_text = _strip_frontmatter_block(style_text)
        
        character_bodies = []
        if loaded_files.get("characters"):
            for char_file in loaded_files["characters"]:
                char_content = char_file.get("content", "")
                if char_content:
                    char_content = _strip_frontmatter_block(char_content)
                    character_bodies.append(char_content)
        
        logger.info(f"Loaded {len(character_bodies)} character reference(s), style: {bool(style_text)}, rules: {bool(rules_body)}")
        
        return {
            "outline_body": outline_body,
            "rules_body": rules_body,
            "style_text": style_text,
            "character_bodies": character_bodies,
            # ✅ CRITICAL: Preserve all state
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve character-specific context
            "active_editor": state.get("active_editor", {}),
            "text": state.get("text", ""),
            "filename": state.get("filename", "character.md"),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "current_request": state.get("current_request", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to load references: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "outline_body": None,
            "rules_body": None,
            "style_text": None,
            "character_bodies": [],
            "error": str(e),
            # ✅ CRITICAL: Preserve all state even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve character-specific context
            "active_editor": state.get("active_editor", {}),
            "text": state.get("text", ""),
            "filename": state.get("filename", "character.md"),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
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
        text = state.get("text", "")
        outline_body = state.get("outline_body")
        rules_body = state.get("rules_body")
        style_text = state.get("style_text")
        character_bodies = state.get("character_bodies", [])
        
        # Build simple prompt for LLM to determine intent
        body_only = _strip_frontmatter_block(text)
        prompt = f"""Analyze the user's request and determine if it's a QUESTION or an EDIT REQUEST.

**USER REQUEST**: {current_request}

**CONTEXT**:
- Current character: {body_only[:500] if body_only else "Empty character"}
- Has rules reference: {bool(rules_body)}
- Has style reference: {bool(style_text)}
- Has outline reference: {bool(outline_body)}
- Has {len(character_bodies)} character reference(s)

**INTENT DETECTION**:
- QUESTIONS (including pure questions and conditional edits): User is asking a question - may or may not want edits
  - Pure questions: "Does she have blue eyes?", "What traits does this character have?", "Show me the character profile"
  - Conditional edits: "Does she have blue eyes? Revise to ensure", "What traits? Add three more if less than five"
  - Questions often start with: "Do you", "What", "Can you", "Are there", "How many", "Show me", "Is", "Does", "Are we", "Suggest"
  - **Key insight**: Questions can be answered, and IF edits are needed based on the answer, they can be made
  - Route ALL questions to edit path - LLM can decide if edits are needed
  
- EDIT REQUESTS: User wants to create, modify, or generate content - NO question asked
  - Examples: "Add three traits", "Create a character profile", "Update the personality section", "Revise the dialogue patterns"
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
        
        content = _extract_llm_content(response)
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
                # ✅ CRITICAL: Preserve character-specific context
                "active_editor": state.get("active_editor", {}),
                "text": state.get("text", ""),
                "filename": state.get("filename", "character.md"),
                "frontmatter": state.get("frontmatter", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "current_request": state.get("current_request", ""),
                "outline_body": state.get("outline_body"),
                "rules_body": state.get("rules_body"),
                "style_text": state.get("style_text"),
                "character_bodies": state.get("character_bodies", [])
            }
        except Exception as e:
            logger.warning(f"Failed to parse request type detection: {e}, defaulting to edit_request")
            return {
                "request_type": "edit_request",
                # ✅ CRITICAL: Preserve all state
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                # ✅ CRITICAL: Preserve character-specific context
                "active_editor": state.get("active_editor", {}),
                "text": state.get("text", ""),
                "filename": state.get("filename", "character.md"),
                "frontmatter": state.get("frontmatter", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "current_request": state.get("current_request", ""),
                "outline_body": state.get("outline_body"),
                "rules_body": state.get("rules_body"),
                "style_text": state.get("style_text"),
                "character_bodies": state.get("character_bodies", [])
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
            # ✅ CRITICAL: Preserve character-specific context
            "active_editor": state.get("active_editor", {}),
            "text": state.get("text", ""),
            "filename": state.get("filename", "character.md"),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "current_request": state.get("current_request", ""),
            "outline_body": state.get("outline_body"),
            "rules_body": state.get("rules_body"),
            "style_text": state.get("style_text"),
            "character_bodies": state.get("character_bodies", [])
        }


async def generate_edit_plan_node(
    state: Dict[str, Any],
    llm_factory: Callable,
    get_datetime_context: Callable
) -> Dict[str, Any]:
    """Generate edit plan using LLM"""
    try:
        logger.info("Generating character edit plan...")
        
        text = state.get("text", "")
        filename = state.get("filename", "character.md")
        frontmatter = state.get("frontmatter", {})
        outline_body = state.get("outline_body")
        rules_body = state.get("rules_body")
        style_text = state.get("style_text")
        character_bodies = state.get("character_bodies", [])
        current_request = state.get("current_request", "")
        
        # Build system prompt
        system_prompt = _build_system_prompt()
        
        # Determine if this is a question or edit request
        request_type = state.get("request_type", "edit_request")
        is_question = request_type == "question"
        
        # Build user message with context
        # Extract body content (strip frontmatter)
        body_only = _strip_frontmatter_block(text)
        
        # ✅ DEBUG: Log what content we're working with
        logger.info(f"📊 CHAR SUBGRAPH CONTEXT: text length={len(text)}, body_only length={len(body_only)}, body_only stripped length={len(body_only.strip())}")
        logger.info(f"🔍 CHAR SUBGRAPH CONTENT PREVIEW: body_only first 300 chars: {body_only[:300]}")
        
        # ⚠️ CRITICAL FIX: Only use conversation history fallback if file is COMPLETELY empty (0 chars)
        # The old < 200 threshold was too aggressive and would activate for files with real content
        # This caused the subgraph to recreate entire profiles instead of adding to existing content
        # If the file has ANY content (even 10 chars), we should use that content, not conversation history
        effective_character_content = body_only
        if len(body_only.strip()) == 0:  # CHANGED FROM < 200 TO == 0 - ONLY completely empty files
            logger.warning(f"⚠️ CHAR SUBGRAPH: File is near-empty ({len(body_only.strip())} chars) - checking conversation history for recent edits...")
            # File is near-empty - check conversation history for recently generated content
            messages_list = state.get("messages", [])
            if messages_list and len(messages_list) >= 2:
                logger.info(f"🔍 CHAR SUBGRAPH: Checking last {min(4, len(messages_list))} messages for character profile content...")
                # Look at the last assistant message for generated character content
                for i, msg in enumerate(reversed(messages_list[-4:])):  # Check last 4 messages
                    if hasattr(msg, 'type') and msg.type == "ai":
                        msg_content = msg.content if hasattr(msg, 'content') else str(msg)
                        # Check if this message contains character profile content (has headers like ### Basic Information)
                        if "### Basic Information" in msg_content or "### Personality" in msg_content or "### Character Arc" in msg_content:
                            logger.warning(f"⚠️ CHAR SUBGRAPH FALLBACK ACTIVATED: Found character profile in message {i+1} from end - USING CONVERSATION HISTORY INSTEAD OF FILE CONTENT!")
                            logger.warning(f"⚠️ This may cause the agent to recreate the entire profile instead of adding to existing content!")
                            # Extract the character profile sections from the message
                            # Don't use the full message (includes summary), just extract the profile parts
                            # This is a simplified extraction - we just need to show the LLM what was recently created
                            effective_character_content = msg_content
                            break
                else:
                    logger.info("✅ CHAR SUBGRAPH: No recent character profile found in conversation history - using file content as-is")
            else:
                logger.info("✅ CHAR SUBGRAPH: Not enough conversation history - using file content as-is")
        
        context_parts = [
            "=== CHARACTER CONTEXT ===\n",
            f"File: {filename}\n\n"
        ]
        
        # ⚠️ CRITICAL FIX: Only show conversation history if file is COMPLETELY empty AND we found recent content
        # Changed from < 200 to == 0 to prevent using stale conversation history for files with content
        if len(body_only.strip()) == 0 and effective_character_content != body_only:
            # Show both file state and recently generated content
            logger.warning(f"⚠️ CHAR SUBGRAPH CONTEXT SWAP: Using conversation history content ({len(effective_character_content)} chars) instead of file content ({len(body_only)} chars)")
            logger.warning(f"⚠️ LLM will be told 'Build on recent content, DON'T recreate' - but may ignore this and recreate anyway!")
            context_parts.append(
                "**IMPORTANT**: File is currently empty, but you recently generated this content (not yet applied by user):\n"
                "Recent Content from Last Response:\n" + effective_character_content + "\n\n"
                "**CRITICAL**: Build on the recently generated content above. DO NOT recreate the entire profile!\n"
                "- If adding new information, use replace_range to update existing sections\n"
                "- If filling in empty sections, use insert_after_heading for those specific sections only\n"
                "- DO NOT recreate sections that already exist in the recent content\n\n"
            )
        else:
            logger.info(f"✅ CHAR SUBGRAPH CONTEXT: Using actual file content ({len(body_only)} chars stripped)")
            context_parts.append("Current Character (frontmatter stripped):\n" + body_only + "\n\n")
        
        if outline_body:
            context_parts.append("=== OUTLINE (if present) ===\n" + outline_body + "\n\n")
        if rules_body:
            context_parts.append("=== RULES (if present) ===\n" + rules_body + "\n\n")
        if style_text:
            context_parts.append("=== STYLE GUIDE (if present) ===\n" + style_text + "\n\n")
        if character_bodies:
            context_parts.append("".join(["=== RELATED CHARACTER DOC ===\n" + b + "\n\n" for b in character_bodies]))
        
        # Add mode-specific instructions
        if is_question:
            context_parts.append(
                "\n=== QUESTION REQUEST: ANSWER FROM EXISTING CONTENT ===\n"
                "The user has asked a question about the character.\n\n"
                "**CRITICAL: This is an INFORMATION REQUEST, not an edit request!**\n"
                "- User wants to KNOW what's in the document, not CHANGE it\n"
                "- Example: 'Can you describe the character's arc?' = Tell me what arc is currently written\n"
                "- Example: 'Does she have blue eyes?' = Tell me what eye color is currently written\n"
                "- Example: 'What are her personality traits?' = List what traits are currently written\n\n"
                "**YOUR TASK**:\n"
                "1. **READ the current content** - Look at what actually exists in the character file\n"
                "2. **REPORT what you find** - Answer the question based on existing content\n"
                "3. **DO NOT CREATE OR EDIT** - Unless explicitly requested\n\n"
                "**WHEN TO EDIT (RARE)**:\n"
                "- ONLY if question explicitly asks for changes: 'Can you add...', 'Revise to...', 'Change... to...'\n"
                "- ONLY if question asks for suggestions: 'Suggest additions', 'What could be improved?'\n"
                "- NEVER edit for pure information questions: 'Can you describe...', 'What is...', 'Does she have...'\n\n"
                "**RESPONSE FORMAT**:\n"
                "- In the 'summary' field: Answer the question by describing what's currently in the document\n"
                "- In the 'operations' array: **EMPTY** for information questions - NO EDITS!\n"
                "- Only provide operations if the question explicitly requests changes or suggestions\n\n"
            )
        else:
            # Edit request mode
            context_parts.append(
                "\n=== EDIT REQUEST: WORK WITH AVAILABLE INFORMATION ===\n"
                "The user wants you to add or revise character content.\n\n"
                "**YOUR APPROACH**:\n"
                "1. **WORK FIRST**: Make edits based on the request and available context (character file, outline, rules, related characters)\n"
                "2. **USE INFERENCE**: Make reasonable inferences from the request - don't wait for clarification\n"
                "3. **ASK ALONG THE WAY**: If you need specific details, include questions in the summary AFTER describing the work you've done\n"
                "4. **NEVER EMPTY OPERATIONS**: Always provide operations based on what you can determine from the request and context\n\n"
            )
            if current_request and any(k in current_request.lower() for k in ["revise", "revision", "tweak", "adjust", "polish", "tighten"]):
                context_parts.append("REVISION MODE: Apply minimal targeted edits; use paragraph-level replace_range ops.\n\n")
        
        context_parts.append("Provide a ManuscriptEdit JSON plan strictly within scope.")
        
        # Build request with mode-specific instructions
        request_with_instructions = ""
        if current_request:
            if is_question:
                request_with_instructions = (
                    f"USER REQUEST: {current_request}\n\n"
                    "**QUESTION MODE**: Answer the question first, then provide edits if needed.\n\n"
                    "CRITICAL: CHECK FOR DUPLICATES FIRST (if edits are needed)\n"
                    "Before adding ANY new content:\n"
                    "1. **CHECK FOR SIMILAR INFO** - Does similar character info already exist in related sections?\n"
                    "2. **CONSOLIDATE IF NEEDED** - If trait appears in multiple places, ensure each adds unique perspective\n"
                    "3. **AVOID REDUNDANCY** - Don't add identical information to multiple sections\n"
                    "\n"
                    "CRITICAL: CROSS-REFERENCE RELATED SECTIONS (if edits are needed)\n"
                    "After checking for duplicates:\n"
                    "1. **SCAN THE ENTIRE DOCUMENT** - Read through ALL sections to identify related character information\n"
                    "2. **IDENTIFY ALL AFFECTED SECTIONS** - When adding/updating character info, find ALL places it should appear\n"
                    "3. **GENERATE MULTIPLE OPERATIONS** - If character info affects multiple sections, create operations for EACH affected section\n"
                    "4. **ENSURE CONSISTENCY** - Related sections must be updated together to maintain character coherence\n"
                    "\n"
                    "CRITICAL ANCHORING INSTRUCTIONS (if edits are needed):\n"
                    "- ALWAYS include 'original_text' with EXACT, VERBATIM text from the file\n"
                    "- For bullet point edits: Match the exact original_text including the bullet marker (e.g., '- Original text here')\n"
                    "- NEVER include header lines (###, ##) in original_text - they will be deleted!\n"
                    "- Copy text directly from the file - do NOT retype or paraphrase\n"
                    "- **BEFORE using insert_after_heading**: Verify the section is COMPLETELY EMPTY (no text below header)\n"
                    "- **If section has ANY content**: Use replace_range to update it, NOT insert_after_heading\n"
                    "- **insert_after_heading will SPLIT sections**: If you use it when content exists, it inserts BETWEEN header and existing text!\n"
                    "- **UPDATING EXISTING CONTENT**: If a section exists but needs improvement/completion, use 'replace_range' with 'original_text' matching the EXISTING content\n"
                    "  * Example: Section has '- Analytical thinker' but needs more → replace_range with original_text='- Analytical thinker' and text='- Analytical thinker\\n- Methodical problem-solver\\n- Protective of family'\n"
                    "  * Example: Section has placeholder '[To be developed]' → replace_range with original_text='[To be developed]' and actual content\n"
                    "- You can return MULTIPLE operations in the operations array - for checking/consolidating duplicates AND updating related sections"
                )
            else:
                # Check if near-empty to emphasize scaffolding
                body_only = _strip_frontmatter_block(text)
                is_near_empty = len(body_only.strip()) < 200
                
                if is_near_empty:
                    request_with_instructions = (
                        f"USER REQUEST: {current_request}\n\n"
                        "**NEAR-EMPTY FILE - USE SCAFFOLDING APPROACH**:\n"
                        "The character file is nearly empty (< 200 chars). You MUST create the profile using MULTIPLE GRANULAR OPERATIONS:\n"
                        "1. **ONE SECTION PER OPERATION** - Each section (Basic Information, Personality, Relationships, etc.) gets its own insert_after_heading operation\n"
                        "2. **CREATE REASONABLE HEADERS** - Use the preferred scaffold (Basic Information, Personality, Relationships, Character Arc)\n"
                        "3. **SKIP SECTIONS WITH NO CONTENT** - Only create headers you can populate from the user's request\n"
                        "4. **NEVER SINGLE OPERATION** - Don't create one large insert_after with the entire profile\n"
                        "5. **EXAMPLE**: For 'Create profile for John Smith, age 30, teacher':\n"
                        "   - Op 1: insert_after_heading(anchor='### Basic Information', text='- Name: John Smith\\n- Age: 30\\n- Occupation: Teacher')\n"
                        "   - Op 2: insert_after_heading(anchor='### Personality', text='...')\n"
                        "   - Op 3: insert_after_heading(anchor='### Character Arc', text='...')\n"
                        "   - Result: 3+ separate operations, each easily reviewable\n\n"
                        "CRITICAL: CHECK FOR DUPLICATES FIRST\n"
                        "Before adding ANY new content:\n"
                        "1. **CHECK FOR SIMILAR INFO** - Does similar character info already exist in related sections?\n"
                        "2. **CONSOLIDATE IF NEEDED** - If trait appears in multiple places, ensure each adds unique perspective\n"
                        "3. **AVOID REDUNDANCY** - Don't add identical information to multiple sections\n"
                        "\n"
                        "CRITICAL: CROSS-REFERENCE AND MULTIPLE OPERATIONS\n"
                        "After checking for duplicates:\n"
                    )
                else:
                    request_with_instructions = (
                        f"USER REQUEST: {current_request}\n\n"
                        "**WORK FIRST**: Make edits based on the request and available context. Use reasonable inferences - don't wait for clarification. Only ask questions in the summary if critical information is truly missing.\n\n"
                        "CRITICAL: CHECK FOR DUPLICATES FIRST\n"
                        "Before adding ANY new content:\n"
                        "1. **CHECK FOR SIMILAR INFO** - Does similar character info already exist in related sections?\n"
                        "2. **CONSOLIDATE IF NEEDED** - If trait appears in multiple places, ensure each adds unique perspective\n"
                        "3. **AVOID REDUNDANCY** - Don't add identical information to multiple sections\n"
                        "\n"
                        "CRITICAL: CROSS-REFERENCE AND MULTIPLE OPERATIONS\n"
                        "After checking for duplicates:\n"
                        "1. **SCAN THE ENTIRE DOCUMENT** - Read through ALL sections to identify related character information\n"
                    "2. **IDENTIFY ALL AFFECTED SECTIONS** - When adding/updating character info, find ALL places it should appear\n"
                    "3. **GENERATE MULTIPLE OPERATIONS** - If character info affects multiple sections, create operations for EACH affected section\n"
                    "4. **ENSURE CONSISTENCY** - Related sections must be updated together to maintain character coherence\n"
                    "\n"
                    "Examples of when to generate multiple operations:\n"
                    "- Adding personality trait → Update 'Personality' section AND 'Dialogue Patterns' if trait affects speech\n"
                    "- Adding relationship detail → Update 'Relationships' section AND 'Character Arc' if relationship affects development\n"
                    "- Adding backstory → Update 'Basic Information' AND 'Personality' AND 'Character Arc' if backstory shapes character\n"
                    "- Updating character info → If info appears in multiple sections, update ALL occurrences, not just one\n"
                    "\n"
                    "CRITICAL ANCHORING INSTRUCTIONS:\n"
                    "- ALWAYS include 'original_text' with EXACT, VERBATIM text from the file\n"
                    "- For bullet point edits: Match the exact original_text including the bullet marker (e.g., '- Original text here')\n"
                    "- NEVER include header lines (###, ##) in original_text - they will be deleted!\n"
                    "- Copy text directly from the file - do NOT retype or paraphrase\n"
                    "- **BEFORE using insert_after_heading**: Verify the section is COMPLETELY EMPTY (no text below header)\n"
                    "- **If section has ANY content**: Use replace_range to update it, NOT insert_after_heading\n"
                    "- **insert_after_heading will SPLIT sections**: If you use it when content exists, it inserts BETWEEN header and existing text!\n"
                    "- **UPDATING EXISTING CONTENT**: If a section exists but needs improvement/completion, use 'replace_range' with 'original_text' matching the EXISTING content\n"
                    "  * Example: Section has '- Analytical thinker' but needs more → replace_range with original_text='- Analytical thinker' and text='- Analytical thinker\\n- Methodical problem-solver\\n- Protective of family'\n"
                    "  * Example: Section has placeholder '[To be developed]' → replace_range with original_text='[To be developed]' and new content\n"
                    "- You can return MULTIPLE operations in the operations array - for checking/consolidating duplicates AND updating related sections"
                )
        
        # Use standardized helper for message construction with conversation history
        messages_list = state.get("messages", [])
        langchain_messages = _build_editing_agent_messages(
            system_prompt=system_prompt,
            context_parts=context_parts,
            current_request=request_with_instructions,
            messages_list=messages_list,
            get_datetime_context=get_datetime_context,
            look_back_limit=6
        )
        
        # Call LLM using llm_factory - pass state to access user's model selection
        llm = llm_factory(temperature=0.35, state=state)
        
        response = await llm.ainvoke(langchain_messages)
        
        content = _extract_llm_content(response)
        if not content or not content.strip():
            raw_content = getattr(response, "content", response)
            logger.error(
                "LLM returned empty content for character edit plan. "
                "Raw response type=%s, content repr=%s",
                type(response).__name__,
                repr(raw_content)[:300] if raw_content is not None else "None",
            )
            return {
                "llm_response": "",
                "structured_edit": None,
                "error": "The model returned no output. Try again or use a different model.",
                "task_status": "error",
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "active_editor": state.get("active_editor", {}),
                "text": state.get("text", ""),
                "filename": state.get("filename", "character.md"),
                "frontmatter": state.get("frontmatter", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "current_request": state.get("current_request", ""),
                "request_type": state.get("request_type", "edit_request"),
                "outline_body": state.get("outline_body"),
                "rules_body": state.get("rules_body"),
                "style_text": state.get("style_text"),
                "character_bodies": state.get("character_bodies", []),
            }
        content = _unwrap_json_response(content)
        
        # Parse structured response
        structured_edit = None
        try:
            raw = json.loads(content)
            if isinstance(raw, dict) and isinstance(raw.get("operations"), list):
                raw.setdefault("target_filename", filename)
                raw.setdefault("scope", "paragraph")
                raw.setdefault("summary", "Planned character edit generated from context.")
                structured_edit = raw
            else:
                structured_edit = json.loads(content)
        except Exception as e:
            logger.error(f"Failed to parse structured edit: {e}")
            if content:
                logger.error(f"LLM returned content (first 500 chars): {content[:500]}")
                logger.error(f"LLM returned content (last 500 chars): {content[-500:]}")
            else:
                logger.error("LLM returned content (first 500 chars): (empty)")
                logger.error("LLM returned content (last 500 chars): (empty)")
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
                # ✅ CRITICAL: Preserve character-specific context
                "active_editor": state.get("active_editor", {}),
                "text": state.get("text", ""),
                "filename": state.get("filename", "character.md"),
                "frontmatter": state.get("frontmatter", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "current_request": state.get("current_request", ""),
                "request_type": state.get("request_type", "edit_request"),
                "outline_body": state.get("outline_body"),
                "rules_body": state.get("rules_body"),
                "style_text": state.get("style_text"),
                "character_bodies": state.get("character_bodies", [])
            }
        
        if structured_edit is None:
            return {
                "llm_response": content,
                "structured_edit": None,
                "error": "Failed to produce a valid Character edit plan. Ensure ONLY raw JSON ManuscriptEdit with operations is returned (no code fences or prose).",
                "task_status": "error",
                # ✅ CRITICAL: Preserve all state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                # ✅ CRITICAL: Preserve character-specific context
                "active_editor": state.get("active_editor", {}),
                "text": state.get("text", ""),
                "filename": state.get("filename", "character.md"),
                "frontmatter": state.get("frontmatter", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "current_request": state.get("current_request", ""),
                "request_type": state.get("request_type", "edit_request"),
                "outline_body": state.get("outline_body"),
                "rules_body": state.get("rules_body"),
                "style_text": state.get("style_text"),
                "character_bodies": state.get("character_bodies", [])
            }
        
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
            # ✅ CRITICAL: Preserve character-specific context
            "active_editor": state.get("active_editor", {}),
            "text": state.get("text", ""),
            "filename": state.get("filename", "character.md"),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "current_request": state.get("current_request", ""),
            "request_type": state.get("request_type", "edit_request"),
            "outline_body": state.get("outline_body"),
            "rules_body": state.get("rules_body"),
            "style_text": state.get("style_text"),
            "character_bodies": state.get("character_bodies", [])
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
            # ✅ CRITICAL: Preserve character-specific context
            "active_editor": state.get("active_editor", {}),
            "text": state.get("text", ""),
            "filename": state.get("filename", "character.md"),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "current_request": state.get("current_request", ""),
            "request_type": state.get("request_type", "edit_request"),
            "outline_body": state.get("outline_body"),
            "rules_body": state.get("rules_body"),
            "style_text": state.get("style_text"),
            "character_bodies": state.get("character_bodies", [])
        }


async def resolve_operations_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve editor operations with progressive search"""
    try:
        logger.info("Resolving editor operations...")
        
        text = state.get("text", "")
        structured_edit = state.get("structured_edit")
        selection_start = state.get("selection_start", -1)
        selection_end = state.get("selection_end", -1)
        
        # ✅ DEBUG: Log what manuscript content the resolver receives
        logger.info(f"📊 RESOLVER INPUT: text length={len(text)}, text preview (first 200 chars): {text[:200]}")
        if not text or len(text.strip()) == 0:
            logger.error("❌ RESOLVER: Received empty manuscript! Operations will fail or insert at position 0!")
        body_without_frontmatter = _strip_frontmatter_block(text)
        logger.info(f"📊 RESOLVER INPUT: body (no frontmatter) length={len(body_without_frontmatter)}, preview: {body_without_frontmatter[:200]}")
        
        if not structured_edit or not isinstance(structured_edit.get("operations"), list):
            error_msg = state.get("error") or "No operations to resolve"
            return {
                "editor_operations": [],
                "failed_operations": [],
                "error": error_msg,
                "task_status": "error",
                # ✅ CRITICAL: Preserve all state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                # ✅ CRITICAL: Preserve character-specific context
                "active_editor": state.get("active_editor", {}),
                "text": state.get("text", ""),
                "filename": state.get("filename", "character.md"),
                "frontmatter": state.get("frontmatter", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "current_request": state.get("current_request", ""),
                "request_type": state.get("request_type", "edit_request"),
                "outline_body": state.get("outline_body"),
                "rules_body": state.get("rules_body"),
                "style_text": state.get("style_text"),
                "character_bodies": state.get("character_bodies", []),
                "structured_edit": state.get("structured_edit")
            }
        
        fm_end_idx = _frontmatter_end_index(text)
        selection = {"start": selection_start, "end": selection_end} if selection_start >= 0 and selection_end >= 0 else None
        
        editor_operations = []
        failed_operations = []
        operations = structured_edit.get("operations", [])
        
        logger.info(f"Resolving {len(operations)} operation(s) from structured_edit")
        
        for op in operations:
            # ✅ PROGRAMMATIC ENFORCEMENT: Validate operation doesn't violate bullet list rules
            is_valid, validation_error = _validate_operation_for_bullet_lists(op, text)
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
            op = _correct_anchor_for_section(op, text)
            
            # Resolve operation
            try:
                # Use centralized resolver
                cursor_pos = state.get("cursor_offset", -1)
                cursor_pos = cursor_pos if cursor_pos >= 0 else None
                resolved_start, resolved_end, resolved_text, resolved_confidence = resolve_editor_operation(
                    content=text,
                    op_dict=op,
                    selection=selection,
                    frontmatter_end=fm_end_idx,
                    cursor_offset=cursor_pos
                )
                
                # CRITICAL: Ensure operations never occur before frontmatter end
                if resolved_start < fm_end_idx:
                    if op.get("op_type") == "delete_range":
                        # Skip deletions targeting frontmatter
                        logger.warning(f"Skipping delete_range operation that targets frontmatter")
                        continue
                    # For inserts/replaces, clamp to frontmatter end
                    if resolved_end <= fm_end_idx:
                        resolved_start = fm_end_idx
                        resolved_end = fm_end_idx
                    else:
                        # Overlap: clamp start to body start
                        resolved_start = fm_end_idx
                
                # Validate resolved positions
                if resolved_start < 0 or resolved_end < 0:
                    logger.warning(f"Invalid resolved positions [{resolved_start}:{resolved_end}], skipping operation")
                    continue
                
                logger.info(f"Resolved {op.get('op_type')} [{resolved_start}:{resolved_end}] confidence={resolved_confidence:.2f}")
                
                # Clean text (remove frontmatter if accidentally included)
                if isinstance(resolved_text, str):
                    resolved_text = _strip_frontmatter_block(resolved_text)
                
                # Calculate pre_hash
                pre_slice = text[resolved_start:resolved_end] if resolved_start < len(text) and resolved_end <= len(text) else ""
                pre_hash = _slice_hash(pre_slice)
                
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
                # Fallback positioning - insert after frontmatter
                fallback_start = fm_end_idx
                fallback_end = fm_end_idx
                
                pre_slice = text[fallback_start:fallback_end] if fallback_start < len(text) else ""
                resolved_op = {
                    "op_type": op.get("op_type", "replace_range"),
                    "start": fallback_start,
                    "end": fallback_end,
                    "text": _strip_frontmatter_block(op.get("text", "")),
                    "pre_hash": _slice_hash(pre_slice),
                    "confidence": 0.3
                }
                editor_operations.append(resolved_op)
        
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
            # ✅ CRITICAL: Preserve character-specific context
            "active_editor": state.get("active_editor", {}),
            "text": state.get("text", ""),
            "filename": state.get("filename", "character.md"),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "current_request": state.get("current_request", ""),
            "request_type": state.get("request_type", "edit_request"),
            "outline_body": state.get("outline_body"),
            "rules_body": state.get("rules_body"),
            "style_text": state.get("style_text"),
            "character_bodies": state.get("character_bodies", []),
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
            # ✅ CRITICAL: Preserve character-specific context
            "active_editor": state.get("active_editor", {}),
            "text": state.get("text", ""),
            "filename": state.get("filename", "character.md"),
            "frontmatter": state.get("frontmatter", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "current_request": state.get("current_request", ""),
            "request_type": state.get("request_type", "edit_request"),
            "outline_body": state.get("outline_body"),
            "rules_body": state.get("rules_body"),
            "style_text": state.get("style_text"),
            "character_bodies": state.get("character_bodies", []),
            "structured_edit": state.get("structured_edit")
        }


async def format_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Format final response with editor operations"""
    try:
        logger.info("📝 CHAR SUBGRAPH FORMAT: Formatting response...")
        
        structured_edit = state.get("structured_edit", {})
        editor_operations = state.get("editor_operations", [])
        failed_operations = state.get("failed_operations", [])
        task_status = state.get("task_status", "complete")
        request_type = state.get("request_type", "edit_request")
        
        # Normalize task_status to valid enum value
        if task_status not in ["complete", "incomplete", "permission_required", "error"]:
            logger.warning(f"⚠️ CHAR SUBGRAPH FORMAT: Invalid task_status '{task_status}', normalizing to 'complete'")
            task_status = "complete"
        
        if task_status == "error":
            error_msg = state.get("error", "Unknown error")
            logger.error(f"❌ CHAR SUBGRAPH FORMAT: Error state detected: {error_msg}")
            error_response = AgentResponse(
                response=f"Character development failed: {error_msg}",
                task_status="error",
                agent_type="character_development_subgraph",
                timestamp=datetime.now().isoformat(),
                error=error_msg
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
        
        # Handle questions - prioritize summary (answer/analysis) over operation text
        if request_type == "question":
            summary = structured_edit.get("summary", "") if structured_edit else ""
            if summary and len(summary.strip()) > 20:
                # Question with answer - use summary as response text (conversational feedback)
                logger.info(f"Question request with {len(editor_operations)} operations - using summary as conversational response")
                response_text = summary
            elif editor_operations:
                # Question with operations but no summary - create fallback
                logger.warning("Question request with operations but no summary - using fallback")
                response_text = f"Analysis complete. Made {len(editor_operations)} edit(s) based on your question."
            else:
                # Pure question with no operations - use summary or fallback
                response_text = summary if summary else "Analysis complete."
        else:
            # Edit request - use summary if available, otherwise operation preview
            summary = structured_edit.get("summary", "") if structured_edit else ""
            if summary and len(summary.strip()) > 20:
                response_text = summary
                logger.info(f"Using summary as response text ({len(summary)} chars)")
            else:
                # Build prose preview from operations
                generated_preview = "\n\n".join([
                    op.get("text", "").strip()
                    for op in editor_operations
                    if op.get("text", "").strip()
                ]).strip()
                logger.info(f"Generated preview from {len(editor_operations)} operations: {len(generated_preview)} chars")
                if not generated_preview and editor_operations:
                    logger.warning(f"Operations have no 'text' field! First op keys: {list(editor_operations[0].keys()) if editor_operations else 'N/A'}")
                response_text = generated_preview if generated_preview else "Edit plan ready."
            
            # Add failed operations section if present
            if failed_operations:
                failed_section = _build_failed_operations_section(failed_operations)
                response_text = response_text + failed_section
        
        # Build manuscript_edit metadata
        manuscript_edit_metadata = None
        if structured_edit:
            manuscript_edit_metadata = ManuscriptEditMetadata(
                target_filename=structured_edit.get("target_filename"),
                scope=structured_edit.get("scope"),
                summary=structured_edit.get("summary"),
                chapter_index=structured_edit.get("chapter_index"),
                safety=structured_edit.get("safety"),
                operations=editor_operations
            )
        
        # Build standard response using AgentResponse contract
        logger.info(f"📊 CHAR SUBGRAPH FORMAT: Creating AgentResponse with task_status='{task_status}', {len(editor_operations)} operation(s)")
        standard_response = AgentResponse(
            response=response_text,
            task_status=task_status,
            agent_type="character_development_subgraph",
            timestamp=datetime.now().isoformat(),
            editor_operations=editor_operations if editor_operations else None,
            manuscript_edit=manuscript_edit_metadata.dict(exclude_none=True) if manuscript_edit_metadata else None
        )
        
        logger.info(f"📊 CHAR SUBGRAPH FORMAT: Response text length: {len(response_text)} chars")
        logger.info(f"📊 CHAR SUBGRAPH FORMAT: Editor operations: {len(editor_operations)} operation(s)")
        logger.info(f"📊 CHAR SUBGRAPH FORMAT: Manuscript edit: {'present' if manuscript_edit_metadata else 'missing'}")
        
        # Verify operations have required fields
        if editor_operations:
            for i, op in enumerate(editor_operations):
                logger.info(f"🔍 CHAR SUBGRAPH FORMAT: Operation {i}: op_type={op.get('op_type')}, start={op.get('start')}, end={op.get('end')}, has_text={bool(op.get('text'))}, text_length={len(op.get('text', ''))}, text_preview={op.get('text', '')[:100] if op.get('text') else 'N/A'}")
        
        logger.info(f"📤 CHAR SUBGRAPH FORMAT: Returning standard AgentResponse with {len(editor_operations)} editor operation(s)")
        return {
            "response": standard_response.dict(exclude_none=True),
            "editor_operations": editor_operations,  # Keep at state level for parent agent extraction
            "task_status": task_status,
            # ✅ CRITICAL: Preserve all state
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
        
    except Exception as e:
        logger.error(f"❌ CHAR SUBGRAPH FORMAT: Failed to format response: {e}")
        import traceback
        logger.error(f"❌ CHAR SUBGRAPH FORMAT: Traceback: {traceback.format_exc()}")
        # Return standard error response
        error_response = AgentResponse(
            response=f"Character development failed: {str(e)}",
            task_status="error",
            agent_type="character_development_subgraph",
            timestamp=datetime.now().isoformat(),
            error=str(e)
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


# ============================================
# Subgraph Builder
# ============================================

def build_character_development_subgraph(
    checkpointer,
    llm_factory: Callable,
    get_datetime_context: Callable
) -> StateGraph:
    """
    Build character development subgraph for integration into parent agents.
    
    This subgraph handles character document editing:
    - Type gating: Strict validation for type: character documents
    - Multi-character loading: Loads other character files from frontmatter
    - Request classification: Distinguishes questions from edit requests
    - Cross-referencing: Generates multiple operations for related sections
    - Bullet point enforcement: Character profiles use concise bullet lists
    
    Args:
        checkpointer: LangGraph checkpointer for state persistence
        llm_factory: Function that creates LLM instances
            Signature: llm_factory(temperature: float, state: Dict[str, Any]) -> LLM
        get_datetime_context: Function that returns datetime context string
            Signature: get_datetime_context() -> str
    
    Expected state inputs:
        - query: str - User's character editing request
        - user_id: str - User identifier
        - metadata: Dict[str, Any] - Contains user_chat_model
        - messages: List[Any] - Conversation history
        - shared_memory: Dict[str, Any] - Contains active_editor with:
            - content: str - Full document (must have frontmatter type: "character")
            - filename: str - Document filename
            - frontmatter: Dict[str, Any] - Parsed frontmatter
            - cursor_offset: int - Cursor position
            - selection_start/end: int - Selection range
    
    Returns state with:
        - response: Dict[str, Any] - Formatted response with messages and agent_results
        - editor_operations: List[Dict[str, Any]] - Resolved operations
        - task_status: str - "complete", "error", or "skipped"
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
    subgraph.add_conditional_edges(
        "prepare_context",
        lambda state: "END" if state.get("task_status") == "skipped" else "load_references",
        {"END": END, "load_references": "load_references"}
    )
    subgraph.add_edge("load_references", "detect_request_type")
    subgraph.add_edge("detect_request_type", "generate_edit_plan")
    subgraph.add_edge("generate_edit_plan", "resolve_operations")
    subgraph.add_edge("resolve_operations", "format_response")
    subgraph.add_edge("format_response", END)
    
    return subgraph.compile(checkpointer=checkpointer)
