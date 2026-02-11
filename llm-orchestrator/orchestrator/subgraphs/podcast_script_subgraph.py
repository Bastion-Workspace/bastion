"""
Podcast Script Subgraph

Reusable subgraph for podcast script generation and editing workflows.
Encapsulates all functionality from podcast_script_agent for integration into multiple agents.

Supports podcast script workflows with:
- BOTH generation mode (creating new scripts) and editing mode (modifying existing scripts)
- ElevenLabs TTS cue lexicon integration
- Section extraction (persona, background, articles, tweets)
- URL content fetching
- Structured operations for editing mode

Produces EditorOperations (editing mode) or script text with metadata (generation mode).
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from orchestrator.utils.editor_operation_resolver import resolve_editor_operation
from orchestrator.models.agent_response_contract import AgentResponse, ManuscriptEditMetadata

logger = logging.getLogger(__name__)


# ============================================
# Utility Functions
# ============================================

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


def _get_frontmatter_end(content: str) -> int:
    """Find frontmatter end position"""
    match = re.match(r'^---\s*\n[\s\S]*?\n---\s*\n', content)
    return match.end() if match else 0


def _extract_sections(editor_content: str, user_message: str) -> Tuple[Optional[str], Optional[str], List[str], Optional[str]]:
    """Extract structured sections from content"""
    try:
        def extract_section(title: str, text: str) -> Optional[str]:
            """Extract ## Title or Title: sections"""
            lines = text.split('\n')
            start = None
            
            # Try markdown heading: ## Title
            pattern_md = rf"^##\s*{title}\s*$"
            for idx, line in enumerate(lines):
                if re.match(pattern_md, line.strip(), flags=re.IGNORECASE):
                    start = idx + 1
                    break
            
            # Try simple heading: Title:
            if start is None:
                pattern_simple = rf"^{title}\s*:\s*$"
                for idx, line in enumerate(lines):
                    if re.match(pattern_simple, line.strip(), flags=re.IGNORECASE):
                        start = idx + 1
                        break
            
            if start is None:
                return None
            
            # Capture until next heading
            collected = []
            for line in lines[start:]:
                stripped = line.strip()
                if stripped.startswith('## ') or re.match(r'^(background|article|persona|tweet)\s*(\d+)?:', stripped, re.IGNORECASE):
                    break
                collected.append(line)
            
            body = '\n'.join(collected).strip()
            return body or None
        
        # Extract from editor first, then user message
        persona_text = extract_section('Persona', editor_content) or extract_section('Persona', user_message)
        background_text = extract_section('Background', user_message)
        tweets_text = extract_section('Tweet', user_message) or extract_section('Tweets', user_message)
        
        # Extract articles
        articles = []
        for i in range(1, 4):
            article = extract_section(f'Article {i}', user_message)
            if article:
                articles.append(article)
        
        # Also try generic "Article"
        generic_article = extract_section('Article', user_message)
        if generic_article and generic_article not in articles:
            articles.insert(0, generic_article)
        
        return persona_text, background_text, articles, tweets_text
        
    except Exception as e:
        logger.warning(f"Section extraction failed: {e}")
        return None, None, [], None


def _build_content_block(
    persona_text: Optional[str],
    background_text: Optional[str],
    articles: List[str],
    tweets_text: Optional[str]
) -> str:
    """Build content block for LLM"""
    sections = []
    
    if persona_text:
        sections.append("=== PERSONA DEFINITIONS ===\n" + persona_text)
    if background_text:
        sections.append("=== BACKGROUND ===\n" + background_text)
    
    for i, article in enumerate(articles, 1):
        sections.append(f"=== ARTICLE {i if len(articles) > 1 else ''} ===\n" + article)
    
    if tweets_text:
        sections.append("=== TWEETS ===\n" + tweets_text)
    
    return "\n\n".join(sections)


def _build_task_block(
    user_message: str,
    target_length: int,
    tone: str,
    pacing: str,
    include_music: bool,
    include_sfx: bool,
    editing_mode: bool = False
) -> str:
    """Build task instruction block"""
    base = (
        "=== REQUEST ===\n"
        f"User instruction: {user_message.strip()}\n\n"
        f"Target length: {target_length} words\n"
        f"Tone: {tone}\n"
        f"Pacing: {pacing}\n"
        f"Include music cues: {include_music}\n"
        f"Include SFX cues: {include_sfx}\n\n"
    )
    
    if editing_mode:
        base += (
            "EDITING MODE: Generate targeted edit operations.\n"
            "Review the current editor content and create operations to modify it.\n"
            "Provide EXACT original_text from the document for replace/delete operations.\n\n"
            "RESPOND WITH JSON:\n"
            "{\n"
            '  "summary": "Brief description of changes",\n'
            '  "operations": [\n'
            "    {\n"
            '      "op_type": "replace_range",\n'
            '      "start": 0,\n'
            '      "end": 50,\n'
            '      "text": "New script content with [brackets]",\n'
            '      "original_text": "Exact text from document (20-40 words)",\n'
            '      "occurrence_index": 0\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )
    else:
        base += (
            "FORMAT GUIDELINES:\n"
            "- Read user request carefully for format (monologue vs dialogue)\n"
            "- If 'monologue' or 'commentary' → single narrator WITHOUT speaker labels\n"
            "- If 'dialogue', 'debate', 'conversation' → multi-speaker WITH labels (HOST:, CALLER:, etc.)\n"
            "- If unclear, default to MONOLOGUE\n"
            "- Ground content in provided sources - cite names, quotes, specifics\n"
            "- Use bracket cues FREQUENTLY: [excited], [mocking], [shouting], [pause], [breathes]\n"
            "- Include natural stammering: 'F...folks', 'This is—ugh—OUTRAGEOUS'\n"
            "- MAXIMUM 3,000 characters total (strict limit)\n\n"
            "RESPOND WITH JSON:\n"
            "{\n"
            '  "task_status": "complete",\n'
            '  "script_text": "Your podcast script here with [bracket cues]",\n'
            '  "metadata": {"words": 900, "estimated_duration_sec": 180, "tag_counts": {}}\n'
            "}\n"
        )
    
    return base


def _parse_response(content: str) -> Tuple[str, Dict[str, Any]]:
    """Parse LLM JSON response (generation mode)"""
    try:
        # Strip code fences
        text = content.strip()
        if '```json' in text:
            m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
            if m:
                text = m.group(1).strip()
        elif '```' in text:
            text = text.replace('```', '').strip()
        
        data = json.loads(text)
        
        script_text = data.get("script_text", "")
        metadata = data.get("metadata", {})
        
        # Strip disallowed tags
        script_text = re.sub(r"\[pause:[^\]]+\]", "", script_text, flags=re.IGNORECASE)
        script_text = re.sub(r"\[(?:beat|breath)\]", "", script_text, flags=re.IGNORECASE)
        
        # Ensure metadata has required fields
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("words", len(script_text.split()))
        metadata.setdefault("estimated_duration_sec", max(60, metadata.get("words", 0) // 3))
        metadata.setdefault("tag_counts", {})
        
        return script_text, metadata
        
    except Exception as e:
        logger.warning(f"JSON parse failed, using fallback: {e}")
        
        # Fallback: treat as raw text
        error_text = f"Failed to parse response: {e}\n\nRaw content: {content[:500]}..."
        return error_text, {
            "words": 0,
            "estimated_duration_sec": 0,
            "tag_counts": {}
        }


def _parse_editing_response(content: str) -> Dict[str, Any]:
    """Parse LLM JSON response (editing mode)"""
    try:
        # Strip code fences
        text = content.strip()
        if '```json' in text:
            m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
            if m:
                text = m.group(1).strip()
        elif '```' in text:
            text = text.replace('```', '').strip()
        
        data = json.loads(text)
        
        # Validate structure
        if not isinstance(data, dict):
            raise ValueError("Response is not a JSON object")
        
        operations = data.get("operations", [])
        if not isinstance(operations, list):
            operations = []
        
        return {
            "summary": data.get("summary", "Edit plan ready"),
            "operations": operations
        }
        
    except Exception as e:
        logger.warning(f"JSON parse failed for editing response: {e}")
        return {
            "summary": "Failed to parse edit plan",
            "operations": []
        }


# ============================================
# System Prompt Builder
# ============================================

def _build_system_prompt(editing_mode: bool = False) -> str:
    """Build podcast script system prompt"""
    base = (
        "You are a professional podcast scriptwriter. "
        "Produce a single-narrator plain-text script suitable for ElevenLabs TTS, with inline bracket cues.\n\n"
    )
    
    # Add editing mode instructions if in editing mode
    if editing_mode:
        base += (
            "=== EDITING MODE - STRUCTURED OPERATIONS ===\n"
            "The editor contains existing content. You must generate targeted edit operations as JSON:\n\n"
            "{\n"
            '  "summary": "Brief description of changes",\n'
            '  "operations": [\n'
            "    {\n"
            '      "op_type": "replace_range",\n'
            '      "start": 0,\n'
            '      "end": 50,\n'
            '      "text": "New script content with [brackets]",\n'
            '      "original_text": "Exact text from document (20-40 words)",\n'
            '      "occurrence_index": 0\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Operation types:\n"
            "- replace_range: Replace existing text with new text\n"
            "- delete_range: Remove existing text\n"
            "- insert_after_heading: Add content after a specific heading\n"
            "- insert_after: Insert text after a specific anchor (for continuing paragraphs/sentences)\n\n"
            "CRITICAL: Provide \"original_text\" with EXACT verbatim text from the document for replace/delete operations.\n"
            "CRITICAL: For insert_after_heading, provide \"anchor_text\" with the exact heading line.\n\n"
        )
    
    base += (
        "=== INLINE CUE LEXICON (ElevenLabs v3 Audio Tags) ===\n"
        "TIMING & RHYTHM CONTROL:\n"
        "- [pause] - Brief pause for effect\n"
        "- [breathes] - Natural breathing moment\n"
        "- [continues after a beat] - Thoughtful pause before continuing\n"
        "- [rushed] - Fast-paced, urgent delivery\n"
        "- [slows down] - Deliberate, measured speech\n"
        "- [deliberate] - Intentionally careful pacing\n"
        "- [rapid-fire] - Very fast, machine-gun delivery\n"
        "- [drawn out] - Extended, stretched pronunciation\n\n"
        "EMPHASIS & STRESS:\n"
        "- [stress on next word] - Emphasizes the following word\n"
        "- [emphasized] - General emphasis on phrase\n"
        "- [understated] - Downplayed, subtle delivery\n"
        "- ALL CAPS for additional emphasis and urgency\n\n"
        "HESITATION & RHYTHM:\n"
        "- [stammers] - Verbal stumbling\n"
        "- [repeats] - Repetition of words for effect\n"
        "- [timidly] - Uncertain, hesitant delivery\n"
        "- [suspicious tone] - Questioning, doubtful\n"
        "- [questioning] - Inquiry or doubt\n\n"
        "TONE & EMOTION:\n"
        "- [flatly] - Monotone, emotionless\n"
        "- [warmly] - Friendly, welcoming\n"
        "- [whisper] / [whispering] - Quiet, conspiratorial\n"
        "- [excited] - Enthusiastic energy\n"
        "- [quietly] - Subdued volume\n"
        "- [hesitant] - Uncertain, cautious\n"
        "- [nervous] - Anxious, worried\n"
        "- [angrily] - Angry tone\n"
        "- [fed up] - Exasperated, at limit\n"
        "- [mocking] - Sarcastic, derisive\n"
        "- [exasperated] / [exasperated sigh] - Frustrated, weary\n"
        "- [disgusted] - Revulsion, contempt\n"
        "- [outraged] / [indignant] - Moral objection\n"
        "- [shouting] / [frustrated shouting] / [enraged] / [furious] - Intense anger\n"
        "- [annoyed] / [building anger] - Escalating irritation\n"
        "- [incensed] / [ranting] - Passionate tirades\n"
        "- [dramatically] - Theatrical emphasis\n\n"
        "STAGE DIRECTIONS & REACTIONS:\n"
        "- [laughs] / [laughing] / [chuckles] / [giggle] / [big laugh] - Laughter variations\n"
        "- [clears throat] - Throat clearing\n"
        "- [sighs] - Audible sigh\n"
        "- [gasp] / [shudder] - Shock and revulsion\n"
        "- [gulps] - Nervous swallowing\n\n"
        "NATURAL STAMMERING & VERBAL STUMBLES (NO TAGS - use creative spelling):\n"
        "- When excited or angry, include natural verbal stumbles for authenticity.\n"
        "- Examples: 'F...folks', 'I...I just can't believe', 'This is—this is OUTRAGEOUS', 'ugh', 'gah', 'argh'\n"
        "- Use ellipses (...) and em-dashes (—) to show hesitation, interruption, or passionate stammering.\n"
        "- 'They're trying to—to TELL US that...', 'This is just—ugh—DISGUSTING'\n\n"
        "CRITICAL: Plain text ONLY. No markdown, no code fences, no SSML.\n"
        "CRITICAL: Keep short paragraphs for breath; tasteful em-dashes.\n"
        "CRITICAL: Use bracket cues FREQUENTLY and ANIMATEDLY for dynamic delivery.\n"
        "CRITICAL: USE EMOTIONAL CUES LIBERALLY.\n"
        "CRITICAL: DON'T BE AFRAID TO SHOUT for emphasis - this adds passion and engagement!\n"
        "CRITICAL: Include NATURAL STAMMERING when excited/angry.\n"
        "CRITICAL: STRICT LENGTH LIMIT - Maximum 3,000 characters total. Be punchy and impactful, not verbose!\n"
    )
    
    return base


# ============================================
# Subgraph Nodes
# ============================================

async def prepare_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare context: extract message and check editor (no type gating - parent handles routing)"""
    try:
        logger.info("Preparing context for podcast script generation...")
        
        messages = state.get("messages", [])
        shared_memory = state.get("shared_memory", {})
        active_editor = shared_memory.get("active_editor", {})
        
        # Get user message and editor content
        latest_message = messages[-1] if messages else None
        user_message = latest_message.content if hasattr(latest_message, 'content') else ""
        editor_content = active_editor.get("content", "")
        frontmatter = active_editor.get("frontmatter", {})
        
        # Detect editing mode: if editor has content (after frontmatter), use editing mode
        editing_mode = False
        if editor_content:
            # Strip frontmatter to check if there's actual content
            frontmatter_match = re.match(r'^---\s*\n[\s\S]*?\n---\s*\n', editor_content)
            if frontmatter_match:
                body_content = editor_content[frontmatter_match.end():].strip()
                editing_mode = len(body_content) > 0
            else:
                editing_mode = len(editor_content.strip()) > 0
        
        logger.info(f"Podcast agent mode: {'EDITING' if editing_mode else 'GENERATION'}")
        
        return {
            "user_message": user_message,
            "editor_content": editor_content,
            "frontmatter": frontmatter,
            "editing_mode": editing_mode,
            "editor_operations": [],
            "structured_edit": None,
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
            "user_message": "",
            "editor_content": "",
            "frontmatter": {},
            "editing_mode": False,
            "error": str(e),
            "task_status": "error",
            # ✅ CRITICAL: Preserve state even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


async def extract_content_node(
    state: Dict[str, Any],
    grpc_client_factory: Callable
) -> Dict[str, Any]:
    """Extract content sections and build content/task blocks"""
    try:
        logger.info("Extracting content sections...")
        
        user_message = state.get("user_message", "")
        editor_content = state.get("editor_content", "")
        frontmatter = state.get("frontmatter", {})
        
        # Extract request parameters
        target_length = int(frontmatter.get("target_length_words", 900))
        tone = str(frontmatter.get("tone", "warm")).lower()
        pacing = str(frontmatter.get("pacing", "moderate")).lower()
        include_music = bool(frontmatter.get("include_music_cues", False))
        include_sfx = bool(frontmatter.get("include_sfx_cues", False))
        
        # Extract structured sections
        persona_text, background_text, articles, tweets_text = _extract_sections(
            editor_content, user_message
        )
        
        # Fetch URL if provided
        fetched_content = await _fetch_url_content(user_message, grpc_client_factory)
        if fetched_content:
            articles.insert(0, fetched_content)
        
        # Build content block
        content_block = _build_content_block(
            persona_text, background_text, articles, tweets_text
        )
        
        # Build system prompt (check editing mode from state)
        editing_mode = state.get("editing_mode", False)
        system_prompt = _build_system_prompt(editing_mode=editing_mode)
        
        # Build task instructions (check editing mode from state)
        task_block = _build_task_block(
            user_message, target_length, tone, pacing, include_music, include_sfx, editing_mode=editing_mode
        )
        
        return {
            "content_block": content_block,
            "system_prompt": system_prompt,
            "task_block": task_block,
            # ✅ CRITICAL: Preserve all state
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve podcast-specific context
            "user_message": state.get("user_message", ""),
            "editor_content": state.get("editor_content", ""),
            "frontmatter": state.get("frontmatter", {}),
            "editing_mode": state.get("editing_mode", False)
        }
        
    except Exception as e:
        logger.error(f"Failed to extract content: {e}")
        return {
            "content_block": "",
            "system_prompt": "",
            "task_block": "",
            "error": str(e),
            # ✅ CRITICAL: Preserve all state even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve podcast-specific context
            "user_message": state.get("user_message", ""),
            "editor_content": state.get("editor_content", ""),
            "frontmatter": state.get("frontmatter", {}),
            "editing_mode": state.get("editing_mode", False)
        }


async def _fetch_url_content(user_message: str, grpc_client_factory: Callable) -> Optional[str]:
    """Fetch content from URL if present in message"""
    try:
        url_match = re.search(r'https?://[^\s)>\"]+', user_message)
        if not url_match:
            return None
        
        url = url_match.group(0)
        logger.info(f"Fetching URL: {url}")
        
        grpc_client = await grpc_client_factory()
        
        # Use web search to get content
        result = await grpc_client.search_web(query=url, max_results=1)
        
        if result.get("success") and result.get("results"):
            return result["results"][0].get("snippet", "")
        
        return None
        
    except Exception as e:
        logger.warning(f"URL fetch failed: {e}")
        return None


async def generate_script_node(
    state: Dict[str, Any],
    llm_factory: Callable,
    get_datetime_context: Callable
) -> Dict[str, Any]:
    """Generate podcast script using LLM"""
    try:
        editing_mode = state.get("editing_mode", False)
        editor_content = state.get("editor_content", "")
        
        if editing_mode:
            logger.info("Generating podcast script edits (editing mode)...")
        else:
            logger.info("Generating podcast script (generation mode)...")
        
        content_block = state.get("content_block", "")
        system_prompt = state.get("system_prompt", "")
        task_block = state.get("task_block", "")
        
        # Use centralized LLM access
        llm = llm_factory(temperature=0.3, state=state)
        
        # Build LangChain messages
        datetime_context = get_datetime_context()
        messages = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=datetime_context)
        ]
        
        if content_block:
            messages.append(HumanMessage(content=content_block))
        
        # In editing mode, include current editor content
        if editing_mode and editor_content:
            # Strip frontmatter for editing context
            frontmatter_match = re.match(r'^---\s*\n[\s\S]*?\n---\s*\n', editor_content)
            if frontmatter_match:
                body_content = editor_content[frontmatter_match.end():]
            else:
                body_content = editor_content
            
            messages.append(HumanMessage(
                content=f"=== CURRENT EDITOR CONTENT ===\n{body_content}\n\n=== END CURRENT CONTENT ===\n\n{task_block}"
            ))
        else:
            messages.append(HumanMessage(content=task_block))
        
        logger.info(f"Generating podcast script with {len(messages)} messages")
        
        # Use LangChain interface
        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)
        
        if editing_mode:
            # Parse structured edit operations
            structured_edit = _parse_editing_response(content)
            logger.info("Podcast Script Agent: Edit plan generation complete")
            return {
                "structured_edit": structured_edit,
                "task_status": "complete",
                # ✅ CRITICAL: Preserve all state
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                # ✅ CRITICAL: Preserve podcast-specific context
                "user_message": state.get("user_message", ""),
                "editor_content": state.get("editor_content", ""),
                "frontmatter": state.get("frontmatter", {}),
                "editing_mode": state.get("editing_mode", False),
                "content_block": state.get("content_block", ""),
                "system_prompt": state.get("system_prompt", ""),
                "task_block": state.get("task_block", "")
            }
        else:
            # Parse structured response (generation mode)
            script_text, metadata = _parse_response(content)
            logger.info("Podcast Script Agent: Script generation complete")
            
            return {
                "script_text": script_text,
                "metadata_result": metadata,
                "task_status": "complete",
                # ✅ CRITICAL: Preserve all state
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                # ✅ CRITICAL: Preserve podcast-specific context
                "user_message": state.get("user_message", ""),
                "editor_content": state.get("editor_content", ""),
                "frontmatter": state.get("frontmatter", {}),
                "editing_mode": state.get("editing_mode", False),
                "content_block": state.get("content_block", ""),
                "system_prompt": state.get("system_prompt", ""),
                "task_block": state.get("task_block", "")
            }
        
    except Exception as e:
        logger.error(f"Script generation failed: {e}")
        return {
            "task_status": "error",
            "error": str(e),
            # ✅ CRITICAL: Preserve all state even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve podcast-specific context
            "user_message": state.get("user_message", ""),
            "editor_content": state.get("editor_content", ""),
            "frontmatter": state.get("frontmatter", {}),
            "editing_mode": state.get("editing_mode", False),
            "content_block": state.get("content_block", ""),
            "system_prompt": state.get("system_prompt", ""),
            "task_block": state.get("task_block", "")
        }


async def resolve_operations_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve editor operations with progressive search"""
    try:
        logger.info("Resolving podcast script operations...")
        
        editor_content = state.get("editor_content", "")
        structured_edit = state.get("structured_edit")
        
        if not structured_edit or not isinstance(structured_edit.get("operations"), list):
            return {
                "editor_operations": [],
                "error": "No operations to resolve",
                "task_status": "error",
                # ✅ CRITICAL: Preserve all state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                # ✅ CRITICAL: Preserve podcast-specific context
                "user_message": state.get("user_message", ""),
                "editor_content": state.get("editor_content", ""),
                "frontmatter": state.get("frontmatter", {}),
                "editing_mode": state.get("editing_mode", False),
                "structured_edit": state.get("structured_edit")
            }
        
        fm_end_idx = _get_frontmatter_end(editor_content)
        
        # Check if file is empty (only frontmatter)
        body_only = editor_content[fm_end_idx:] if fm_end_idx < len(editor_content) else ""
        is_empty_file = not body_only.strip()
        
        editor_operations = []
        operations = structured_edit.get("operations", [])
        
        for op in operations:
            try:
                # Use centralized resolver
                resolved_start, resolved_end, resolved_text, resolved_confidence = resolve_editor_operation(
                    content=editor_content,
                    op_dict=op,
                    selection=None,
                    frontmatter_end=fm_end_idx,
                    cursor_offset=None
                )
                
                # Special handling for empty files: ensure operations insert after frontmatter
                if is_empty_file and resolved_start < fm_end_idx:
                    resolved_start = fm_end_idx
                    resolved_end = fm_end_idx
                    resolved_confidence = 0.7
                    logger.info(f"Empty file detected - adjusting operation to insert after frontmatter at {fm_end_idx}")
                
                logger.info(f"Resolved {op.get('op_type')} [{resolved_start}:{resolved_end}] confidence={resolved_confidence:.2f}")
                
                # Build operation dict
                resolved_op = {
                    "op_type": op.get("op_type", "replace_range"),
                    "start": resolved_start,
                    "end": resolved_end,
                    "text": resolved_text,
                    "original_text": op.get("original_text"),
                    "anchor_text": op.get("anchor_text"),
                    "occurrence_index": op.get("occurrence_index", 0),
                    "confidence": resolved_confidence
                }
                
                editor_operations.append(resolved_op)
                
            except Exception as e:
                logger.warning(f"Operation resolution failed: {e}")
                continue
        
        return {
            "editor_operations": editor_operations,
            # ✅ CRITICAL: Preserve all state
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve podcast-specific context
            "user_message": state.get("user_message", ""),
            "editor_content": state.get("editor_content", ""),
            "frontmatter": state.get("frontmatter", {}),
            "editing_mode": state.get("editing_mode", False),
            "structured_edit": state.get("structured_edit")
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
            # ✅ CRITICAL: Preserve podcast-specific context
            "user_message": state.get("user_message", ""),
            "editor_content": state.get("editor_content", ""),
            "frontmatter": state.get("frontmatter", {}),
            "editing_mode": state.get("editing_mode", False),
            "structured_edit": state.get("structured_edit")
        }


async def format_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Format final response with editor operations or script text"""
    try:
        logger.info("📝 PODCAST SUBGRAPH FORMAT: Formatting response...")
        
        editing_mode = state.get("editing_mode", False)
        editor_operations = state.get("editor_operations", [])
        structured_edit = state.get("structured_edit", {})
        script_text = state.get("script_text", "")
        metadata_result = state.get("metadata_result", {})
        task_status = state.get("task_status", "complete")
        
        # Normalize task_status to valid enum value
        if task_status not in ["complete", "incomplete", "permission_required", "error"]:
            logger.warning(f"⚠️ PODCAST SUBGRAPH FORMAT: Invalid task_status '{task_status}', normalizing to 'complete'")
            task_status = "complete"
        
        if editing_mode:
            # Editing mode: return operations
            logger.info(f"📝 PODCAST SUBGRAPH FORMAT: Editing mode - {len(editor_operations)} operation(s)")
            preview = "\n\n".join([
                op.get("text", "").strip()
                for op in editor_operations
                if op.get("text", "").strip()
            ]).strip()
            response_text = preview if preview else (structured_edit.get("summary", "Edit plan ready."))
            
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
            logger.info(f"📊 PODCAST SUBGRAPH FORMAT: Creating AgentResponse (editing mode) with task_status='{task_status}', {len(editor_operations)} operation(s)")
            standard_response = AgentResponse(
                response=response_text,
                task_status=task_status,
                agent_type="podcast_script_subgraph",
                timestamp=datetime.now().isoformat(),
                editor_operations=editor_operations if editor_operations else None,
                manuscript_edit=manuscript_edit_metadata.dict(exclude_none=True) if manuscript_edit_metadata else None
            )
            
            logger.info(f"📊 PODCAST SUBGRAPH FORMAT: Response text length: {len(response_text)} chars")
            logger.info(f"📊 PODCAST SUBGRAPH FORMAT: Editor operations: {len(editor_operations)} operation(s)")
            logger.info(f"📊 PODCAST SUBGRAPH FORMAT: Manuscript edit: {'present' if manuscript_edit_metadata else 'missing'}")
            
            logger.info(f"📤 PODCAST SUBGRAPH FORMAT: Returning standard AgentResponse (editing mode) with {len(editor_operations)} editor operation(s)")
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
        else:
            # Generation mode: return script text
            logger.info(f"📝 PODCAST SUBGRAPH FORMAT: Generation mode - script text length: {len(script_text)} chars")
            
            # Build standard response using AgentResponse contract
            logger.info(f"📊 PODCAST SUBGRAPH FORMAT: Creating AgentResponse (generation mode) with task_status='{task_status}'")
            standard_response = AgentResponse(
                response=script_text,
                task_status=task_status,
                agent_type="podcast_script_subgraph",
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"📊 PODCAST SUBGRAPH FORMAT: Script text length: {len(script_text)} chars")
            logger.info(f"📊 PODCAST SUBGRAPH FORMAT: Metadata result: {'present' if metadata_result else 'missing'}")
            
            logger.info(f"📤 PODCAST SUBGRAPH FORMAT: Returning standard AgentResponse (generation mode)")
            return {
                "response": standard_response.dict(exclude_none=True),
                "script_text": script_text,  # Keep at state level for parent agent extraction
                "metadata_result": metadata_result,
                "task_status": task_status,
                # ✅ CRITICAL: Preserve all state
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
    except Exception as e:
        logger.error(f"❌ PODCAST SUBGRAPH FORMAT: Format response failed: {e}")
        import traceback
        logger.error(f"❌ PODCAST SUBGRAPH FORMAT: Traceback: {traceback.format_exc()}")
        # Return standard error response
        error_response = AgentResponse(
            response=f"Response formatting failed: {str(e)}",
            task_status="error",
            agent_type="podcast_script_subgraph",
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

def build_podcast_script_subgraph(
    checkpointer,
    llm_factory: Callable,
    get_datetime_context: Callable,
    grpc_client_factory: Callable
) -> StateGraph:
    """
    Build podcast script subgraph for integration into parent agents.
    
    Supports BOTH generation mode (creating new scripts) and editing mode
    (modifying existing scripts with structured operations).
    
    Args:
        checkpointer: LangGraph checkpointer for state persistence
        llm_factory: Function that creates LLM instances
            Signature: llm_factory(temperature: float, state: Dict[str, Any]) -> LLM
        get_datetime_context: Function that returns datetime context string
            Signature: get_datetime_context() -> str
        grpc_client_factory: Function that returns gRPC client for URL fetching
            Signature: grpc_client_factory() -> gRPC client
    
    Expected state inputs:
        - query: str - User's podcast script request
        - user_id: str - User identifier
        - metadata: Dict[str, Any] - Contains user_chat_model
        - messages: List[Any] - Conversation history
        - shared_memory: Dict[str, Any] - Contains active_editor with:
            - content: str - Full document (must have frontmatter type: "podcast")
            - filename: str - Document filename
            - frontmatter: Dict[str, Any] - Parsed frontmatter
            - cursor_offset: int - Cursor position
            - selection_start/end: int - Selection range
    
    Returns state with:
        - response: Dict[str, Any] - Formatted response with messages and agent_results
        - editor_operations: List[Dict[str, Any]] (editing mode only)
        - script_text: str (generation mode only)
        - metadata_result: Dict[str, Any] (generation mode only)
        - task_status: str - "complete", "error"
        - All input state preserved
    """
    subgraph = StateGraph(Dict[str, Any])
    
    # Add nodes (some need factory binding)
    subgraph.add_node("prepare_context", prepare_context_node)
    
    # Bind grpc_client_factory to extract_content_node
    async def extract_content_node_bound(state):
        return await extract_content_node(state, grpc_client_factory)
    
    # Bind llm_factory and get_datetime_context to generate_script_node
    async def generate_script_node_bound(state):
        return await generate_script_node(state, llm_factory, get_datetime_context)
    
    subgraph.add_node("extract_content", extract_content_node_bound)
    subgraph.add_node("generate_script", generate_script_node_bound)
    subgraph.add_node("resolve_operations", resolve_operations_node)
    subgraph.add_node("format_response", format_response_node)
    
    # Entry point
    subgraph.set_entry_point("prepare_context")
    
    # Define edges
    subgraph.add_edge("prepare_context", "extract_content")
    subgraph.add_edge("extract_content", "generate_script")
    
    # Conditional routing based on editing mode
    def route_after_generation(state: Dict[str, Any]) -> str:
        if state.get("editing_mode"):
            return "resolve_operations"
        return "format_response"
    
    subgraph.add_conditional_edges(
        "generate_script",
        route_after_generation,
        {
            "resolve_operations": "resolve_operations",
            "format_response": "format_response"
        }
    )
    
    subgraph.add_edge("resolve_operations", "format_response")
    subgraph.add_edge("format_response", END)
    
    return subgraph.compile(checkpointer=checkpointer)
