"""
LLM-primary route selector: select best route from eligible list using descriptions.

Used as the primary routing mechanism after hard-gate filtering. Uses the fast model
and route descriptions (no keyword scoring).
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.routes.route_schema import Route
from orchestrator.utils.openrouter_client import get_openrouter_client

logger = logging.getLogger(__name__)

MIN_CONFIDENCE_FALLBACK = 0.3


@dataclass
class RoutingResult:
    """Result of ranked route selection: primary route and ordered fallbacks for rejection retry."""
    primary: str
    fallback_stack: List[str]  # ordered runner-ups, e.g. ["research", "chat"]
    confidences: Dict[str, float]  # route name -> confidence


def _build_selection_prompt(
    eligible: List[Route],
    query: str,
    editor_context: Optional[Dict[str, Any]],
    conversation_context: Optional[Dict[str, Any]],
) -> str:
    """Build prompt for LLM route selection from eligible routes (descriptions only)."""
    parts = [
        "You are a route selector. Given the user's query and context, select the best route.",
        "",
        f'USER QUERY: "{query}"',
    ]
    has_editor = False
    if editor_context and (editor_context.get("type") or editor_context.get("filename")):
        t = editor_context.get("type") or "unknown"
        fn = editor_context.get("filename") or ""
        parts.append(f"EDITOR: type={t}, filename={fn}")
        has_editor = True
    else:
        parts.append("EDITOR: No editor active")
    shared = (conversation_context or {}).get("shared_memory") or {}
    last_route = shared.get("last_agent") or shared.get("primary_agent_selected")
    if last_route:
        if last_route.endswith("_agent"):
            last_route = last_route[:-6]
        parts.append(f'CONTINUITY: Previously using "{last_route}"')
    else:
        parts.append("CONTINUITY: New conversation")
    parts.append("")
    parts.append("AVAILABLE ROUTES:")
    for i, r in enumerate(eligible, 1):
        desc = (r.description or "").strip()
        if len(desc) > 120:
            desc = desc[:117] + "..."
        parts.append(f'{i}. {r.name} - {desc}')
    parts.append("")
    parts.append(
        "Select the single best route for this query. "
        "Use \"chat\" for casual conversation or when uncertain."
    )

    # Editor-specific guidance
    if has_editor:
        parts.append(
            "IMPORTANT: When an editor is active, STRONGLY prefer the matching editor route "
            "for analyzing, reviewing, or editing the active document. "
            "Multi-faceted questions about the same document (e.g. 'How is Chapter 7? Any issues?') "
            "should use ONE editor route, not multiple routes. "
            "Editor context takes precedence over conversation continuity - if the user previously used "
            "a conversational route but now has a file open, switch to the appropriate editor route. "
            "For fiction/manuscript: 'Generate Chapter N' or 'write chapter N' = fiction_editing (content in "
            "current manuscript), NOT document_creator (create new file). "
            "CRITICAL: Distinguish analysis vs editing. Use content_analysis ONLY for read-only assessment "
            "(summarize, compare, critique, identify gaps). If the user asks to FIX, EDIT, MODIFY, REWRITE, "
            "IMPROVE, UPDATE, or CHANGE the document, choose the appropriate editor route (e.g. article_writing) "
            "instead of content_analysis. For grammar/spelling/typos, use proofreading."
        )
    else:
        parts.append(
            "When the user asks a factual or how-to question (e.g. 'how do I', 'how to', 'what is', 'how can I') "
            "and no editor is active, prefer \"research\" over editor routes (electronics, general_project, etc.). "
            "Research looks up information; editor routes need an open document."
        )

    parts.append(
        "NOTE: Queries about the user's own collection (e.g. 'do we have', 'find me', 'show me' "
        "comics/photos/images in their library) use \"research\", not \"entertainment\" and not \"rss\". "
        "Entertainment is for recommendations only; rss is for subscribing to/managing RSS feeds. "
        "Research searches local documents and images."
    )
    parts.append(
        "CAPTURE-TO-INBOX RULE: 'Capture to inbox', 'Add to inbox', 'Quick capture' + [anything] -> use \"org_capture\" ONLY. "
        "Do NOT use research. Inbox is for short idea capture; just put what the user said into the inbox with proper org formatting. "
        "Examples: 'Capture to my inbox: Article on X' -> org_capture (capture that idea/reminder as a note); "
        "'Add to inbox: buy milk' -> org_capture. "
        "Only use research when the user EXPLICITLY asks for research first (e.g. 'Research X then capture to inbox')."
    )
    parts.append(
        "org_capture is ONLY for explicit capture requests. If the user asks a hypothetical ('what if', 'what would happen if'), "
        "analytical ('how would', 'would it help if'), or conversational question - even about topics related to their journal - "
        "use \"chat\" or \"org_content\" (if an org file is open), NOT org_capture."
    )
    parts.append('JSON only: {"skill": "<name>", "confidence": 0.0-1.0, "reason": "brief"}')
    return "\n".join(parts)


def _build_ranked_selection_prompt(
    eligible: List[Route],
    query: str,
    editor_context: Optional[Dict[str, Any]],
    conversation_context: Optional[Dict[str, Any]],
) -> str:
    """Same as _build_selection_prompt but asks for ranked top-3 routes for fallback stack."""
    base = _build_selection_prompt(eligible, query, editor_context, conversation_context)
    base = base.replace(
        'JSON only: {"skill": "<name>", "confidence": 0.0-1.0, "reason": "brief"}',
        'JSON only: {"ranked": [{"skill": "<name>", "confidence": 0.0-1.0}, ...], "reason": "brief"}. '
        "Return up to 3 routes in order of best fit (best first). Use exact route names from AVAILABLE ROUTES.",
    )
    return base


def _parse_selection_response(content: str, eligible: List[Route]) -> Tuple[Optional[str], float]:
    """Parse LLM JSON response; return (route_name, confidence) or (None, 0.0)."""
    raw = (content or "").strip()
    if not raw:
        return None, 0.0
    text = raw
    if "```json" in text:
        match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    elif "```" in text:
        text = text.replace("```", "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Route selection: invalid JSON %s", e)
        return None, 0.0
    name = (data.get("skill") or "").strip().lower().replace("-", "_")
    conf = float(data.get("confidence", 0)) if data.get("confidence") is not None else 0.0
    if not name:
        return None, conf
    for r in eligible:
        if r.name.lower().replace("-", "_") == name:
            return r.name, conf
    if name.endswith("_agent"):
        name_short = name[:-6]
        for r in eligible:
            if r.name.lower().replace("-", "_") == name_short:
                return r.name, conf
    logger.warning("Route selection: LLM chose unknown route %s", data.get("skill"))
    return None, conf


def _normalize_route_name(name: str) -> str:
    """Normalize route name for matching (lowercase, underscores, no _agent suffix)."""
    n = (name or "").strip().lower().replace("-", "_")
    if n.endswith("_agent"):
        n = n[:-6]
    return n


def _resolve_route_name(raw: str, eligible: List[Route]) -> Optional[str]:
    """Resolve raw LLM route name to canonical route name from eligible list."""
    n = _normalize_route_name(raw)
    if not n:
        return None
    for r in eligible:
        if _normalize_route_name(r.name) == n:
            return r.name
    return None


def _parse_ranked_response(content: str, eligible: List[Route]) -> Optional[RoutingResult]:
    """Parse LLM ranked JSON; return RoutingResult or None on failure."""
    raw = (content or "").strip()
    if not raw:
        return None
    text = raw
    if "```json" in text:
        match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    elif "```" in text:
        text = text.replace("```", "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Ranked selection: invalid JSON %s", e)
        return None
    ranked = data.get("ranked")
    if not ranked or not isinstance(ranked, list):
        single_name = (data.get("skill") or "").strip()
        if single_name:
            resolved = _resolve_route_name(single_name, eligible)
            if resolved:
                conf = float(data.get("confidence", 0)) if data.get("confidence") is not None else 0.0
                return RoutingResult(primary=resolved, fallback_stack=[], confidences={resolved: conf})
        return None
    seen: set = set()
    primary = None
    fallback_stack: List[str] = []
    confidences: Dict[str, float] = {}
    for i, item in enumerate(ranked[:3]):
        if not isinstance(item, dict):
            continue
        name_raw = (item.get("skill") or "").strip()
        if not name_raw:
            continue
        resolved = _resolve_route_name(name_raw, eligible)
        if not resolved or resolved in seen:
            continue
        seen.add(resolved)
        conf = float(item.get("confidence", 0)) if item.get("confidence") is not None else 0.0
        confidences[resolved] = conf
        if primary is None:
            primary = resolved
        else:
            fallback_stack.append(resolved)
    if not primary:
        return None
    return RoutingResult(primary=primary, fallback_stack=fallback_stack, confidences=confidences)


async def llm_select_route(
    eligible: List[Route],
    query: str,
    editor_context: Optional[Dict[str, Any]] = None,
    conversation_context: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Primary route selector: select best route from eligible list using fast LLM.

    Returns the selected route name (e.g. "weather", "chat"). Returns "chat" on
    failure, missing API key, or confidence below MIN_CONFIDENCE_FALLBACK.
    """
    if not eligible:
        return "chat"
    try:
        from config.settings import settings
        model = getattr(settings, "FAST_MODEL", "anthropic/claude-3-haiku")
    except Exception:
        model = os.getenv("FAST_MODEL", "anthropic/claude-3-haiku")
    from orchestrator.utils.llm_credentials_from_metadata import get_openrouter_credentials
    api_key, base_url = get_openrouter_credentials(metadata)
    if not api_key:
        logger.warning("Route selection: OpenRouter API key not configured, falling back to chat")
        return "chat"
    prompt = _build_selection_prompt(eligible, query, editor_context, conversation_context)
    try:
        client = get_openrouter_client(api_key=api_key, base_url=base_url)
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a route selector. Respond with JSON only (no prose, no markdown fences)."},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.1,
            max_tokens=200,
        )
    except Exception as e:
        logger.warning("Route selection: LLM call failed %s, falling back to chat", e)
        return "chat"
    content = (response.choices[0].message.content or "").strip()
    selected_name, conf = _parse_selection_response(content, eligible)
    if not selected_name:
        return "chat"
    if conf < MIN_CONFIDENCE_FALLBACK:
        logger.info("Route selection: low confidence %.2f for %s, falling back to chat", conf, selected_name)
        return "chat"
    logger.info("Route selection: %s (confidence=%.2f)", selected_name, conf)
    return selected_name


async def llm_select_route_ranked(
    eligible: List[Route],
    query: str,
    editor_context: Optional[Dict[str, Any]] = None,
    conversation_context: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> RoutingResult:
    """
    Ranked route selector: returns primary route and ordered fallback stack for rejection retry.
    Uses same LLM call as single selection but asks for top-3 ranked list.
    """
    if not eligible:
        return RoutingResult(primary="chat", fallback_stack=[], confidences={})
    try:
        from config.settings import settings
        model = getattr(settings, "FAST_MODEL", "anthropic/claude-3-haiku")
    except Exception:
        model = os.getenv("FAST_MODEL", "anthropic/claude-3-haiku")
    from orchestrator.utils.llm_credentials_from_metadata import get_openrouter_credentials
    api_key, base_url = get_openrouter_credentials(metadata)
    if not api_key:
        logger.warning("Ranked route selection: OpenRouter API key not configured, falling back to chat")
        return RoutingResult(primary="chat", fallback_stack=[], confidences={})
    prompt = _build_ranked_selection_prompt(eligible, query, editor_context, conversation_context)
    try:
        client = get_openrouter_client(api_key=api_key, base_url=base_url)
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a route selector. Respond with JSON only (no prose, no markdown fences)."},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.1,
            max_tokens=250,
        )
    except Exception as e:
        logger.warning("Ranked route selection: LLM call failed %s, falling back to chat", e)
        return RoutingResult(primary="chat", fallback_stack=[], confidences={})
    content = (response.choices[0].message.content or "").strip()
    result = _parse_ranked_response(content, eligible)
    if not result:
        return RoutingResult(primary="chat", fallback_stack=[], confidences={})
    primary_conf = result.confidences.get(result.primary, 0.0)
    if primary_conf < MIN_CONFIDENCE_FALLBACK:
        logger.info(
            "Ranked route selection: low confidence %.2f for %s, falling back to chat",
            primary_conf,
            result.primary,
        )
        return RoutingResult(primary="chat", fallback_stack=[], confidences=result.confidences)
    logger.info(
        "Ranked route selection: %s (confidence=%.2f), fallbacks=%s",
        result.primary,
        primary_conf,
        result.fallback_stack,
    )
    return result
