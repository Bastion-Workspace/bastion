"""
Article Writing Subgraph

Reusable subgraph for article, substack, and blog document workflows.
Used by Writing Assistant when active editor type is article, substack, or blog.

Supports:
- Type gating: article, substack, blog
- Nfoutline + cascade (style, references) for type: article
- Generation and editing modes (unified)
- Section extraction (## Persona, ## Background, ## Article, etc.)
- URL fetching from user message
- Proofreading subgraph integration (conditional)
- Editor operations via centralized resolver

Produces EditorOperations and AgentResponse for Writing Assistant extraction.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from orchestrator.utils.editor_operation_resolver import resolve_editor_operation
from orchestrator.utils.writing_subgraph_utilities import (
    preserve_critical_state,
    create_writing_error_response,
    strip_frontmatter_block,
    build_response_text_for_edit,
    create_manuscript_edit_metadata,
    build_editor_operation_instructions,
)
from orchestrator.models.agent_response_contract import AgentResponse, TaskStatus
from orchestrator.tools.reference_file_loader import load_referenced_files
from orchestrator.subgraphs.proofreading_subgraph import build_proofreading_subgraph
from orchestrator.subgraphs.nonfiction_outline_subgraph import find_section_ranges

logger = logging.getLogger(__name__)


# ============================================
# State Definition
# ============================================

class ArticleWritingState(TypedDict, total=False):
    """State for article writing subgraph LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    persona: Optional[Dict[str, Any]]
    user_message: str
    editor_content: str
    frontmatter: Dict[str, Any]
    editing_mode: bool
    structured_edit: Optional[Dict[str, Any]]
    editor_operations: List[Dict[str, Any]]
    content_block: str
    system_prompt: str
    task_block: str
    article_text: str
    metadata_result: Dict[str, Any]
    response: Dict[str, Any]
    task_status: str
    error: str
    manuscript_edit: Dict[str, Any]
    nfoutline_body: Optional[str]
    outline_sections_list: List[str]
    style_body: Optional[str]
    reference_bodies: List[str]
    reference_titles: List[str]
    manuscript_content: str


# ============================================
# Helper Functions (article-specific)
# ============================================

def _get_frontmatter_end(content: str) -> int:
    """Return end index of leading YAML frontmatter block, or 0."""
    m = re.match(r'^---\s*\n[\s\S]*?\n---\s*\n', content)
    return m.end() if m else 0


def _extract_section(title: str, text: str) -> Optional[str]:
    """Extract ## Title or Title: section body from text."""
    if not text:
        return None
    lines = text.split('\n')
    start = None
    pattern_md = rf"^##\s*{re.escape(title)}\s*$"
    for idx, line in enumerate(lines):
        if re.match(pattern_md, line.strip(), flags=re.IGNORECASE):
            start = idx + 1
            break
    if start is None:
        pattern_simple = rf"^{re.escape(title)}\s*:\s*$"
        for idx, line in enumerate(lines):
            if re.match(pattern_simple, line.strip(), flags=re.IGNORECASE):
                start = idx + 1
                break
    if start is None:
        return None
    collected = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith('## ') or re.match(r'^(background|article|persona|tweet)\s*(\d+)?:', stripped, re.IGNORECASE):
            break
        collected.append(line)
    body = '\n'.join(collected).strip()
    return body or None


def _extract_sections(editor_content: str, user_message: str) -> Tuple[Optional[str], Optional[str], List[str], Optional[str]]:
    """Extract Persona, Background, Articles, Tweets from editor and user message."""
    try:
        persona_text = _extract_section('Persona', editor_content) or _extract_section('Persona', user_message)
        background_text = _extract_section('Background', editor_content) or _extract_section('Background', user_message)
        tweets_text = (
            _extract_section('Tweet', editor_content) or _extract_section('Tweets', editor_content)
            or _extract_section('Tweet', user_message) or _extract_section('Tweets', user_message)
        )
        articles = []
        for i in range(1, 4):
            a = _extract_section(f'Article {i}', editor_content) or _extract_section(f'Article {i}', user_message)
            if a:
                articles.append(a)
        generic = _extract_section('Article', editor_content) or _extract_section('Article', user_message)
        if generic and generic not in articles:
            articles.insert(0, generic)
        return persona_text, background_text, articles, tweets_text
    except Exception as e:
        logger.warning(f"Section extraction failed: {e}")
        return None, None, [], None


def _build_content_block(
    persona_text: Optional[str],
    background_text: Optional[str],
    articles: List[str],
    tweets_text: Optional[str],
    *,
    nfoutline_body: Optional[str] = None,
    outline_sections_list: Optional[List[str]] = None,
    style_body: Optional[str] = None,
    reference_bodies: Optional[List[str]] = None,
    reference_titles: Optional[List[str]] = None,
) -> str:
    """Build content block for LLM."""
    sections = []
    if nfoutline_body or (outline_sections_list and len(outline_sections_list) > 0):
        sections.append("=== NON-FICTION OUTLINE (follow this structure) ===")
        if outline_sections_list:
            sections.append(
                "Section order (use these ## headings in the article):\n"
                + "\n".join(f"- {name}" for name in outline_sections_list)
            )
        if nfoutline_body:
            sections.append("\nOutline content:\n" + nfoutline_body)
        sections.append("")
    if style_body:
        sections.append("=== STYLE GUIDE ===\n" + style_body + "\n")
    if reference_bodies:
        titles = reference_titles if reference_titles and len(reference_titles) >= len(reference_bodies) else [f"Reference {i+1}" for i in range(len(reference_bodies))]
        for body, title in zip(reference_bodies, titles):
            sections.append(f"=== REFERENCE: {title} ===\n" + body + "\n")
    if persona_text:
        sections.append("=== PERSONA ===\n" + persona_text)
    if background_text:
        sections.append("=== BACKGROUND ===\n" + background_text)
    for i, article in enumerate(articles, 1):
        sections.append(f"=== ARTICLE {i} ===\n" + article)
    if tweets_text:
        sections.append("=== TWEETS ===\n" + tweets_text)
    return "\n\n".join(sections)


def _build_system_prompt(persona: Optional[Dict[str, Any]] = None, editing_mode: bool = False) -> str:
    """Build article writing system prompt (no tweet mode)."""
    base = (
        "You are a professional long-form article writer specializing in blog posts and article publications. "
        "Your task is to synthesize multiple source materials into a cohesive, engaging article.\n\n"
    )
    if editing_mode:
        base += (
            "=== EDITING MODE - STRUCTURED OPERATIONS ===\n"
            "The editor contains existing content. Treat it as full context: what is already written, which sections "
            "and arguments are in place, and how to continue (e.g. when adding the next section, build on that context).\n"
            "You must generate targeted edit operations as JSON:\n\n"
            "{\n"
            '  "summary": "Brief description of changes",\n'
            '  "operations": [\n'
            "    {\n"
            '      "op_type": "replace_range",\n'
            '      "start": 0,\n'
            '      "end": 50,\n'
            '      "text": "New article content",\n'
            '      "original_text": "Exact text from document (20-40 words)",\n'
            '      "occurrence_index": 0\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
        )
        base += build_editor_operation_instructions(
            document_type="article",
            include_chapter_guidance=False,
            include_domain_guidance=False,
            custom_rules=[
                "Copy EXACT verbatim text from the article body; never paraphrase anchor_text or original_text.",
                "For insert_after: anchor_text MUST be the exact last sentence or last 10-20 words of the paragraph you are inserting after, copied character-for-character from the document.",
            ],
        )
        base += "\n\n"
    base += (
        "=== ARTICLE WRITING PRINCIPLES ===\n"
        "STRUCTURE:\n"
        "- Compelling title that captures the essence\n"
        "- Strong opening hook that draws readers in\n"
        "- Clear thesis or central argument\n"
        "- Well-organized body sections with logical flow\n"
        "- Smooth transitions between ideas\n"
        "- Satisfying conclusion that reinforces main points\n\n"
        "STYLE:\n"
        "- Active voice and strong verbs\n"
        "- Varied sentence structure for rhythm\n"
        "- Concrete examples and specific details\n"
        "- Direct quotes from source material when impactful\n"
        "- Clear, accessible language (avoid jargon unless necessary)\n"
        "- Natural conversational tone while maintaining professionalism\n\n"
        "CONTENT INTEGRATION:\n"
        "- Weave multiple sources together seamlessly\n"
        "- Compare and contrast different perspectives\n"
        "- Provide context and analysis, not just summary\n"
        "- Cite sources naturally in text (e.g., 'According to...', 'As reported in...')\n"
        "- Build original arguments on top of source material\n\n"
        "MARKDOWN FORMATTING:\n"
        "- Use ## for main section headers\n"
        "- Use ### for subsection headers\n"
        "- **Bold** for emphasis on key points\n"
        "- *Italics* for subtle emphasis or titles\n"
        "- > Blockquotes for direct quotations from sources\n"
        "- Bullet points or numbered lists for clarity when appropriate\n"
        "- Em-dashes (—) for parenthetical thoughts\n\n"
        "CRITICAL REQUIREMENTS:\n"
        "- Target length: 2000-5000 words (adjustable per request)\n"
        "- Ground content in provided sources—cite specific names, quotes, statistics\n"
        "- Maintain consistent voice and perspective throughout\n"
        "- Provide original insight and analysis, not just summarization\n"
        "- Make it publishable—polished, professional, engaging\n"
    )
    if persona:
        persona_name = persona.get("ai_name", "")
        if persona_name:
            base += f"\n\nYou are {persona_name}."
        persona_style = persona.get("persona_style", "professional")
        if persona_style and persona_style != "professional":
            base += f"\n\nAdopt a {persona_style} tone and style."
    return base


def _build_task_block(
    user_message: str,
    target_length: int,
    tone: str,
    style: str,
    frontmatter: Dict[str, Any],
    editing_mode: bool = False,
    outline_sections_list: Optional[List[str]] = None,
) -> str:
    """Build task instruction block (no tweet mode)."""
    if editing_mode:
        return (
            "=== EDITING MODE ===\n"
            "Generate targeted edit operations.\n"
            "The content under CURRENT EDITOR CONTENT above is the article as it exists now. Use it as full context: "
            "what has already been written, which sections exist, and what arguments or rhetoric are in place. "
            "For requests like 'craft the next section' or 'write the following section', add content that continues "
            "from the existing material—match tone and style, avoid repeating points, and extend the argument or narrative.\n"
            "Provide EXACT original_text from the document for replace/delete operations.\n\n"
            "RESPOND WITH JSON:\n"
            "{\n"
            '  "summary": "Brief description of changes",\n'
            '  "operations": [\n'
            "    {\n"
            '      "op_type": "replace_range",\n'
            '      "start": 0,\n'
            '      "end": 50,\n'
            '      "text": "New article content",\n'
            '      "original_text": "Exact text from document (20-40 words)",\n'
            '      "occurrence_index": 0\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"User instruction: {user_message}\n"
        )
    base = (
        "=== ARTICLE WRITING INSTRUCTIONS ===\n\n"
        "Generate a long-form article synthesizing provided sources.\n\n"
        f"Target length: {target_length} words\n"
        f"Tone: {tone}\n"
        f"Style: {style}\n\n"
    )
    if outline_sections_list and len(outline_sections_list) > 0:
        base += (
            "STRUCTURE (CRITICAL): Follow the NON-FICTION OUTLINE section order.\n"
            "Use exactly these ## section headings in this order: "
            + ", ".join(outline_sections_list)
            + "\n"
            "Do not add or remove top-level sections; write content under each heading.\n\n"
        )
    base += (
        "FORMAT: Start with # Title, use ## for sections (as above when outline provided), ### for subsections\n"
        "CITE SOURCES: Reference articles and reference materials naturally in text\n\n"
        "RESPOND WITH JSON:\n"
        "{\n"
        '  "task_status": "complete",\n'
        '  "article_text": "# Title\\n\\n## Section...\\n\\nYour article markdown here",\n'
        '  "metadata": {"word_count": 2500, "reading_time_minutes": 10, "section_count": 5}\n'
        "}\n\n"
        f"User instruction: {user_message}\n"
    )
    return base


def _parse_response(content: str) -> Tuple[str, Dict[str, Any]]:
    """Parse LLM JSON response (generation mode)."""
    try:
        text = content.strip()
        if '```json' in text:
            m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
            if m:
                text = m.group(1).strip()
        elif '```' in text:
            text = text.replace('```', '').strip()
        data = json.loads(text)
        article_text = data.get("article_text", "")
        metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
        metadata.setdefault("word_count", len(article_text.split()))
        metadata.setdefault("reading_time_minutes", max(1, metadata.get("word_count", 0) // 200))
        metadata.setdefault("section_count", article_text.count("##"))
        return article_text, metadata
    except Exception as e:
        logger.warning(f"JSON parse failed: {e}")
        return (
            f"Failed to parse response: {e}\n\nRaw content: {content[:500]}...",
            {"word_count": 0, "reading_time_minutes": 0, "section_count": 0},
        )


def _parse_editing_response(content: str) -> Dict[str, Any]:
    """Parse LLM JSON response (editing mode)."""
    try:
        text = content.strip()
        if '```json' in text:
            m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
            if m:
                text = m.group(1).strip()
        elif '```' in text:
            text = text.replace('```', '').strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Response is not a JSON object")
        operations = data.get("operations", [])
        if not isinstance(operations, list):
            operations = []
        return {"summary": data.get("summary", "Edit plan ready"), "operations": operations}
    except Exception as e:
        logger.warning(f"JSON parse failed for editing response: {e}")
        return {"summary": "Failed to parse edit plan", "operations": []}


# ============================================
# Node Implementations
# ============================================

async def _prepare_context_node(state: ArticleWritingState) -> Dict[str, Any]:
    """Prepare context: type gating (article/substack/blog), mode detection."""
    try:
        logger.info("Preparing context for article writing...")
        shared_memory = state.get("shared_memory", {}) or {}
        active_editor = shared_memory.get("active_editor", {})
        if not active_editor:
            return create_writing_error_response(
                "No active editor found. Article writing requires an active editor with type article, substack, or blog.",
                "article_writing",
                state,
            )
        frontmatter = active_editor.get("frontmatter", {}) or {}
        doc_type = str(frontmatter.get("type", "")).strip().lower()
        if doc_type not in ["substack", "blog", "article"]:
            return create_writing_error_response(
                f"Active editor is not article/substack/blog (type='{doc_type}').",
                "article_writing",
                state,
            )
        messages = state.get("messages", [])
        latest_message = messages[-1] if messages else None
        user_message = latest_message.content if hasattr(latest_message, "content") else ""
        editor_content = active_editor.get("content", "")
        editing_mode = False
        if editor_content:
            fm_match = re.match(r'^---\s*\n[\s\S]*?\n---\s*\n', editor_content)
            if fm_match:
                body_content = editor_content[fm_match.end() :].strip()
                editing_mode = len(body_content) > 0
            else:
                editing_mode = len(editor_content.strip()) > 0
        logger.info(f"Article writing mode: {'EDITING' if editing_mode else 'GENERATION'}")
        persona = (state.get("metadata") or {}).get("persona")
        return {
            "user_message": user_message,
            "editor_content": editor_content,
            "frontmatter": frontmatter,
            "editing_mode": editing_mode,
            "editor_operations": [],
            "structured_edit": None,
            "persona": persona,
            **preserve_critical_state(state),
        }
    except Exception as e:
        logger.error(f"Failed to prepare context: {e}")
        return create_writing_error_response(str(e), "article_writing", state)


async def _load_references_node(state: ArticleWritingState) -> Dict[str, Any]:
    """Load nfoutline + cascade style/references for type: article."""
    try:
        frontmatter = state.get("frontmatter", {}) or {}
        doc_type = str(frontmatter.get("type", "")).strip().lower()
        nfoutline_body = None
        outline_sections_list = []
        style_body = None
        reference_bodies = []
        reference_titles = []
        if doc_type == "article" and (frontmatter.get("outline") or frontmatter.get("nfoutline")):
            logger.info("Loading nfoutline and cascaded style/references for article...")
            active_editor = (state.get("shared_memory") or {}).get("active_editor", {})
            user_id = state.get("user_id", "system")
            result = await load_referenced_files(
                active_editor=active_editor,
                user_id=user_id,
                reference_config={"outline": ["outline", "nfoutline"]},
                doc_type_filter="article",
                cascade_config={
                    "outline": {
                        "style": ["style"],
                        "references": ["reference", "reference_*", "references"],
                    }
                },
            )
            loaded = result.get("loaded_files", {})
            if loaded.get("outline") and len(loaded["outline"]) > 0:
                primary = loaded["outline"][0]
                raw_content = primary.get("content", "")
                if raw_content:
                    nfoutline_body = strip_frontmatter_block(raw_content)
                    section_ranges = find_section_ranges(nfoutline_body)
                    outline_sections_list = [r.section_name for r in section_ranges if r.section_name]
                    logger.info(f"Nfoutline loaded: {len(outline_sections_list)} sections")
            if loaded.get("style") and len(loaded["style"]) > 0:
                style_body = strip_frontmatter_block(loaded["style"][0].get("content", ""))
            if loaded.get("references"):
                for ref_doc in loaded["references"]:
                    ref_content = ref_doc.get("content", "")
                    if ref_content:
                        reference_bodies.append(strip_frontmatter_block(ref_content))
                        reference_titles.append(ref_doc.get("filename", ref_doc.get("title", "reference")))
        return {
            "nfoutline_body": nfoutline_body,
            "outline_sections_list": outline_sections_list,
            "style_body": style_body,
            "reference_bodies": reference_bodies,
            "reference_titles": reference_titles,
            **preserve_critical_state(state),
        }
    except Exception as e:
        logger.error(f"Failed to load references: {e}")
        return create_writing_error_response(str(e), "article_writing", state)


async def _fetch_url_content(user_message: str) -> Optional[str]:
    """Fetch content from URL if present in message."""
    try:
        url_match = re.search(r'https?://[^\s)>\"]+', user_message)
        if not url_match:
            return None
        url = url_match.group(0)
        logger.info(f"Fetching URL: {url}")
        from orchestrator.backend_tool_client import get_backend_tool_client
        client = await get_backend_tool_client()
        result = await client.search_web(query=url, max_results=1)
        if result.get("success") and result.get("results"):
            return result["results"][0].get("snippet", "")
        return None
    except Exception as e:
        logger.warning(f"URL fetch failed: {e}")
        return None


async def _extract_content_node(state: ArticleWritingState) -> Dict[str, Any]:
    """Extract sections and build content/task blocks."""
    try:
        logger.info("Extracting content sections...")
        user_message = state.get("user_message", "")
        editor_content = state.get("editor_content", "")
        frontmatter = state.get("frontmatter", {}) or {}
        persona = state.get("persona")
        target_length = int(frontmatter.get("target_length_words", 2500))
        tone = str(frontmatter.get("tone", "conversational")).lower()
        style = str(frontmatter.get("style", "commentary")).lower()
        editing_mode = state.get("editing_mode", False)
        nfoutline_body = state.get("nfoutline_body")
        outline_sections_list = state.get("outline_sections_list", [])
        style_body = state.get("style_body")
        reference_bodies = state.get("reference_bodies", [])
        reference_titles = state.get("reference_titles", [])
        persona_text, background_text, articles, tweets_text = _extract_sections(editor_content, user_message)
        fetched = await _fetch_url_content(user_message)
        if fetched:
            articles.append(fetched)
        content_block = _build_content_block(
            persona_text,
            background_text,
            articles,
            tweets_text,
            nfoutline_body=nfoutline_body,
            outline_sections_list=outline_sections_list,
            style_body=style_body,
            reference_bodies=reference_bodies,
            reference_titles=reference_titles,
        )
        system_prompt = _build_system_prompt(persona, editing_mode=editing_mode)
        task_block = _build_task_block(
            user_message,
            target_length,
            tone,
            style,
            frontmatter,
            editing_mode=editing_mode,
            outline_sections_list=outline_sections_list if outline_sections_list else None,
        )
        return {
            "content_block": content_block,
            "system_prompt": system_prompt,
            "task_block": task_block,
            **preserve_critical_state(state),
        }
    except Exception as e:
        logger.error(f"Failed to extract content: {e}")
        return create_writing_error_response(str(e), "article_writing", state)


async def _generate_article_node_impl(
    state: ArticleWritingState,
    llm_factory: Callable,
    get_datetime_context: Callable,
) -> Dict[str, Any]:
    """Generate article or edit operations using LLM."""
    try:
        editing_mode = state.get("editing_mode", False)
        editor_content = state.get("editor_content", "")
        content_block = state.get("content_block", "")
        system_prompt = state.get("system_prompt", "")
        task_block = state.get("task_block", "")
        llm = llm_factory(temperature=0.4, state=state)
        datetime_context = get_datetime_context()
        messages = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=datetime_context),
        ]
        if content_block:
            messages.append(HumanMessage(content=content_block))
        if editing_mode and editor_content:
            fm_match = re.match(r'^---\s*\n[\s\S]*?\n---\s*\n', editor_content)
            body_content = editor_content[fm_match.end() :] if fm_match else editor_content
            messages.append(
                HumanMessage(
                    content=f"=== CURRENT EDITOR CONTENT ===\n{body_content}\n\n=== END CURRENT CONTENT ===\n\n{task_block}"
                )
            )
        else:
            messages.append(HumanMessage(content=task_block))
        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        if editing_mode:
            structured_edit = _parse_editing_response(content)
            return {
                "structured_edit": structured_edit,
                "task_status": "complete",
                **preserve_critical_state(state),
            }
        article_text, metadata = _parse_response(content)
        result = {
            "messages": [AIMessage(content=article_text)],
            "agent_results": {
                "agent_type": "article_writing",
                "success": True,
                "metadata": metadata,
                "is_complete": True,
            },
            "is_complete": True,
        }
        return {
            "response": result,
            "article_text": article_text,
            "metadata_result": metadata,
            "task_status": "complete",
            "manuscript_content": article_text,
            **preserve_critical_state(state),
        }
    except Exception as e:
        logger.error(f"Article generation failed: {e}")
        return {
            "response": {
                "messages": [AIMessage(content=f"Article generation failed: {str(e)}")],
                "agent_results": {"agent_type": "article_writing", "success": False, "error": str(e), "is_complete": True},
                "is_complete": True,
            },
            "task_status": "error",
            "error": str(e),
            **preserve_critical_state(state),
        }


async def _resolve_operations_node(state: ArticleWritingState) -> Dict[str, Any]:
    """Resolve editor operations with centralized resolver."""
    try:
        logger.info("Resolving article operations...")
        editor_content = state.get("editor_content", "")
        structured_edit = state.get("structured_edit")
        if not structured_edit or not isinstance(structured_edit.get("operations"), list):
            return {
                "editor_operations": [],
                "error": "No operations to resolve",
                "task_status": "error",
                **preserve_critical_state(state),
            }
        fm_end_idx = _get_frontmatter_end(editor_content)
        body_only = editor_content[fm_end_idx:] if fm_end_idx < len(editor_content) else ""
        is_empty_file = not body_only.strip()
        editor_operations = []
        for op in structured_edit.get("operations", []):
            try:
                resolved_start, resolved_end, resolved_text, resolved_confidence = resolve_editor_operation(
                    content=editor_content,
                    op_dict=op,
                    selection=None,
                    frontmatter_end=fm_end_idx,
                    cursor_offset=None,
                )
                if is_empty_file and resolved_start < fm_end_idx:
                    resolved_start = fm_end_idx
                    resolved_end = fm_end_idx
                    resolved_confidence = 0.7
                resolved_op = {
                    "op_type": op.get("op_type", "replace_range"),
                    "start": resolved_start,
                    "end": resolved_end,
                    "text": resolved_text,
                    "original_text": op.get("original_text"),
                    "anchor_text": op.get("anchor_text"),
                    "occurrence_index": op.get("occurrence_index", 0),
                    "confidence": resolved_confidence,
                }
                editor_operations.append(resolved_op)
            except Exception as e:
                logger.warning(f"Operation resolution failed: {e}")
                continue
        return {
            "editor_operations": editor_operations,
            **preserve_critical_state(state),
        }
    except Exception as e:
        logger.error(f"Failed to resolve operations: {e}")
        return {
            "editor_operations": [],
            "error": str(e),
            "task_status": "error",
            **preserve_critical_state(state),
        }


async def _format_response_node(state: ArticleWritingState) -> Dict[str, Any]:
    """Format final response with editor_operations and manuscript_edit at state level."""
    try:
        if state.get("task_status") == "error" and state.get("response"):
            return {
                "response": state.get("response"),
                "task_status": state.get("task_status"),
                **preserve_critical_state(state),
            }
        editing_mode = state.get("editing_mode", False)
        editor_operations = state.get("editor_operations", [])
        structured_edit = state.get("structured_edit", {}) or {}
        article_text = state.get("article_text", "")
        metadata_result = state.get("metadata_result", {})
        if editing_mode:
            response_text = build_response_text_for_edit(structured_edit, editor_operations, "Edit plan ready.")
            filename = (state.get("shared_memory") or {}).get("active_editor") or {}
            filename = filename.get("filename", "document.md") if isinstance(filename, dict) else "document.md"
            structured_edit_with_meta = {**structured_edit, "target_filename": filename, "scope": "document"}
            manuscript_edit_meta = create_manuscript_edit_metadata(structured_edit_with_meta, editor_operations)
            manuscript_edit_dict = None
            if manuscript_edit_meta:
                manuscript_edit_dict = {
                    **manuscript_edit_meta.model_dump(),
                    "operations": editor_operations,
                }
            elif structured_edit and editor_operations:
                manuscript_edit_dict = {**structured_edit, "operations": editor_operations}
            agent_response = AgentResponse(
                response=response_text,
                task_status=TaskStatus.COMPLETE,
                agent_type="article_writing",
                timestamp=datetime.now().isoformat(),
                editor_operations=editor_operations,
                manuscript_edit=manuscript_edit_meta,
            )
            resp_dict = agent_response.model_dump(exclude_none=True)
            if manuscript_edit_dict is not None:
                resp_dict["manuscript_edit"] = manuscript_edit_dict
            return {
                "response": resp_dict,
                "task_status": "complete",
                "editor_operations": editor_operations,
                "manuscript_edit": manuscript_edit_dict,
                **preserve_critical_state(state),
            }
        result = AgentResponse(
            response=article_text,
            task_status=TaskStatus.COMPLETE,
            agent_type="article_writing",
            timestamp=datetime.now().isoformat(),
            structured_data=metadata_result,
        )
        return {
            "response": result.model_dump(exclude_none=True),
            "task_status": "complete",
            **preserve_critical_state(state),
        }
    except Exception as e:
        logger.error(f"Format response failed: {e}")
        return create_writing_error_response(str(e), "article_writing", state)


def _route_after_generation(state: ArticleWritingState) -> str:
    """Route after generation: proofreading intent or resolve/format."""
    query = (state.get("query", "") or "").lower()
    user_message = (state.get("user_message", "") or "").lower()
    proofreading_keywords = [
        "proofread", "check grammar", "fix typos", "style corrections",
        "grammar check", "spell check", "proofreading", "grammar", "typos",
    ]
    if any(kw in query or kw in user_message for kw in proofreading_keywords):
        logger.info("Detected proofreading intent - routing to proofreading subgraph")
        return "proofreading"
    if state.get("editing_mode"):
        return "resolve_operations"
    return "format_response"


# ============================================
# Subgraph Factory
# ============================================

def build_article_writing_subgraph(
    checkpointer,
    llm_factory: Callable,
    get_datetime_context: Callable,
):
    """Build article writing subgraph with proofreading integration."""
    workflow = StateGraph(ArticleWritingState)
    proofreading_subgraph = build_proofreading_subgraph(
        checkpointer,
        llm_factory=llm_factory,
        get_datetime_context=get_datetime_context,
    )

    async def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return await _generate_article_node_impl(state, llm_factory, get_datetime_context)

    workflow.add_node("prepare_context", _prepare_context_node)
    workflow.add_node("load_references", _load_references_node)
    workflow.add_node("extract_content", _extract_content_node)
    workflow.add_node("generate_article", generate_node)
    workflow.add_node("proofreading", proofreading_subgraph)
    workflow.add_node("resolve_operations", _resolve_operations_node)
    workflow.add_node("format_response", _format_response_node)

    def _route_after_prepare(state: Dict[str, Any]) -> str:
        if state.get("task_status") == "error":
            return "format_response"
        return "load_references"

    workflow.set_entry_point("prepare_context")
    workflow.add_conditional_edges(
        "prepare_context",
        _route_after_prepare,
        {"format_response": "format_response", "load_references": "load_references"},
    )
    workflow.add_edge("load_references", "extract_content")
    workflow.add_edge("extract_content", "generate_article")
    workflow.add_conditional_edges(
        "generate_article",
        _route_after_generation,
        {
            "proofreading": "proofreading",
            "resolve_operations": "resolve_operations",
            "format_response": "format_response",
        },
    )
    workflow.add_conditional_edges(
        "proofreading",
        lambda s: "resolve_operations" if s.get("editing_mode") else "format_response",
        {"resolve_operations": "resolve_operations", "format_response": "format_response"},
    )
    workflow.add_edge("resolve_operations", "format_response")
    workflow.add_edge("format_response", END)

    return workflow.compile(checkpointer=checkpointer)
