"""
LLM-primary skill router: select best skill from eligible list using descriptions.

Used as the primary routing mechanism after hard-gate filtering. Uses the fast model
and skill descriptions (no keyword scoring).
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.engines.plan_models import ExecutionPlan, PlanStep
from orchestrator.skills.skill_schema import Skill
from orchestrator.utils.openrouter_client import get_openrouter_client

logger = logging.getLogger(__name__)

MIN_CONFIDENCE_FALLBACK = 0.3


def _build_selection_prompt(
    eligible: List[Skill],
    query: str,
    editor_context: Optional[Dict[str, Any]],
    conversation_context: Optional[Dict[str, Any]],
) -> str:
    """Build prompt for LLM skill selection from eligible skills (descriptions only)."""
    parts = [
        "You are a skill router. Given the user's query and context, select the best skill.",
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
    last_skill = shared.get("last_agent") or shared.get("primary_agent_selected")
    if last_skill:
        if last_skill.endswith("_agent"):
            last_skill = last_skill[:-6]
        parts.append(f'CONTINUITY: Previously using "{last_skill}"')
    else:
        parts.append("CONTINUITY: New conversation")
    parts.append("")
    parts.append("AVAILABLE SKILLS:")
    for i, sk in enumerate(eligible, 1):
        desc = (sk.description or "").strip()
        if len(desc) > 120:
            desc = desc[:117] + "..."
        parts.append(f'{i}. {sk.name} - {desc}')
    parts.append("")
    parts.append(
        "Select the single best skill for this query. "
        "Use \"chat\" for casual conversation or when uncertain."
    )
    
    # Editor-specific guidance
    if has_editor:
        parts.append(
            "IMPORTANT: When an editor is active, STRONGLY prefer the matching editor skill "
            "for analyzing, reviewing, or editing the active document. "
            "Multi-faceted questions about the same document (e.g. 'How is Chapter 7? Any issues?') "
            "should use ONE editor skill, not multiple skills. "
            "Editor context takes precedence over conversation continuity - if the user previously used "
            "a conversational skill but now has a file open, switch to the appropriate editor skill. "
            "For fiction/manuscript: 'Generate Chapter N' or 'write chapter N' = fiction_editing (content in "
            "current manuscript), NOT document_creator (create new file)."
        )
    else:
        parts.append(
            "When the user asks a factual or how-to question (e.g. 'how do I', 'how to', 'what is', 'how can I') "
            "and no editor is active, prefer \"research\" over editor skills (electronics, general_project, etc.). "
            "Research looks up information; editor skills need an open document."
        )
    
    parts.append(
        "NOTE: Queries about the user's own collection (e.g. 'do we have', 'find me', 'show me' "
        "comics/photos/images in their library) use \"research\", not \"entertainment\" and not \"rss\". "
        "Entertainment is for recommendations only; rss is for subscribing to/managing RSS feeds. "
        "Research searches local documents and images."
    )
    parts.append('JSON only: {"skill": "<name>", "confidence": 0.0-1.0, "reason": "brief"}')
    return "\n".join(parts)


def _parse_selection_response(content: str, eligible: List[Skill]) -> Tuple[Optional[str], float]:
    """Parse LLM JSON response; return (skill_name, confidence) or (None, 0.0)."""
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
        logger.warning("Skill selection: invalid JSON %s", e)
        return None, 0.0
    name = (data.get("skill") or "").strip().lower().replace("-", "_")
    conf = float(data.get("confidence", 0)) if data.get("confidence") is not None else 0.0
    if not name:
        return None, conf
    for sk in eligible:
        if sk.name.lower().replace("-", "_") == name:
            return sk.name, conf
    if name.endswith("_agent"):
        name_short = name[:-6]
        for sk in eligible:
            if sk.name.lower().replace("-", "_") == name_short:
                return sk.name, conf
    logger.warning("Skill selection: LLM chose unknown skill %s", data.get("skill"))
    return None, conf


async def llm_select_skill(
    eligible: List[Skill],
    query: str,
    editor_context: Optional[Dict[str, Any]] = None,
    conversation_context: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Primary skill router: select best skill from eligible list using fast LLM.

    Returns the selected skill name (e.g. "weather", "chat"). Returns "chat" on
    failure, missing API key, or confidence below MIN_CONFIDENCE_FALLBACK.
    """
    if not eligible:
        return "chat"
    try:
        from config.settings import settings
        model = getattr(settings, "FAST_MODEL", "anthropic/claude-3-haiku")
    except Exception:
        model = os.getenv("FAST_MODEL", "anthropic/claude-3-haiku")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("Skill selection: OPENROUTER_API_KEY not set, falling back to chat")
        return "chat"
    prompt = _build_selection_prompt(eligible, query, editor_context, conversation_context)
    try:
        client = get_openrouter_client(api_key=api_key)
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a skill router. Respond with JSON only (no prose, no markdown fences)."},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.1,
            max_tokens=200,
        )
    except Exception as e:
        logger.warning("Skill selection: LLM call failed %s, falling back to chat", e)
        return "chat"
    content = (response.choices[0].message.content or "").strip()
    selected_name, conf = _parse_selection_response(content, eligible)
    if not selected_name:
        return "chat"
    if conf < MIN_CONFIDENCE_FALLBACK:
        logger.info("Skill selection: low confidence %.2f for %s, falling back to chat", conf, selected_name)
        return "chat"
    logger.info("Skill selection: %s (confidence=%.2f)", selected_name, conf)
    return selected_name


def _build_plan_selection_prompt(
    eligible: List[Skill],
    query: str,
    editor_context: Optional[Dict[str, Any]],
    conversation_context: Optional[Dict[str, Any]],
) -> str:
    """Build prompt for compound-aware skill selection (single skill or multi-step plan)."""
    parts = [
        "You are a skill router. Given the user's query and context, either select ONE skill or decompose into a multi-step plan.",
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
    last_skill = shared.get("last_agent") or shared.get("primary_agent_selected")
    if last_skill:
        if last_skill.endswith("_agent"):
            last_skill = last_skill[:-6]
        parts.append(f'CONTINUITY: Previously using "{last_skill}"')
    else:
        parts.append("CONTINUITY: New conversation")
    parts.append("")
    parts.append("AVAILABLE SKILLS (use exact name):")
    eligible_names = []
    for sk in eligible:
        desc = (sk.description or "").strip()
        if len(desc) > 100:
            desc = desc[:97] + "..."
        parts.append(f"  - {sk.name}: {desc}")
        eligible_names.append(sk.name)
    parts.append("")
    parts.append(
        "If the query needs exactly ONE skill, respond with: "
        '{"is_compound": false, "skill": "<name>", "confidence": 0.0-1.0, "reasoning": "brief"}'
    )
    parts.append("")
    parts.append(
        "If the query clearly needs MULTIPLE skills in sequence (e.g. research then capture then edit), "
        'respond with: {"is_compound": true, "steps": [{"step_id": 1, "skill_name": "<name>", "sub_query": "...", "depends_on": [], "context_keys": []}, ...], "reasoning": "brief"}'
    )
    parts.append("")
    parts.append(
        "NOTE: Queries about the user's collection (e.g. 'do we have any X comics', 'find me images of Y') "
        "use \"research\" to search local documents and images; use \"rss\" only for subscribing to or managing RSS feeds."
    )
    parts.append("")
    parts.append(
        "CAPTURE-TO-INBOX RULE: 'Capture to inbox', 'Add to inbox', 'Quick capture' + [anything] → use \"org_capture\" ONLY. "
        "Do NOT research. Inbox is for short idea capture; just put what the user said into the inbox with proper org formatting. "
        "Examples: 'Capture to my inbox: Article on X' → org_capture (capture that idea/reminder as a note); "
        "'Add to inbox: buy milk' → org_capture. "
        "Only use compound (research + org_capture) when the user EXPLICITLY asks for research first "
        "(e.g. 'Research X then capture to inbox', 'Find an article on X and add it to my inbox')."
    )
    parts.append("")
    parts.append(
        "RULES: Max 4 steps. Each step must use a skill from AVAILABLE SKILLS. "
        "Only use is_compound true when the user explicitly asks for multiple distinct actions in sequence. "
        "When in doubt, return is_compound false with the single best skill."
    )
    
    # CRITICAL: Add editor-context-specific guidance
    if has_editor:
        parts.append("")
        parts.append(
            "CRITICAL: When an editor is active, multi-faceted questions about the SAME document "
            "(e.g. 'How is Chapter 7? Any redundancy? Style issues?') are ONE skill invocation, NOT compound. "
            "Only use compound if the user asks to perform actions on DIFFERENT documents or systems "
            "(e.g. 'Research X, then edit my document, then capture to inbox'). "
            "STRONGLY prefer the matching editor skill for the active document - editor context takes "
            "precedence over conversation continuity. "
            "For fiction/manuscript: 'Generate Chapter N' or 'write chapter N' means fiction_editing (generate "
            "content in the current manuscript), NOT document_creator (create a new file in a folder)."
        )
    
    parts.append("Respond with JSON only (no prose, no markdown fences).")
    return "\n".join(parts)


def _parse_plan_response(content: str, eligible: List[Skill]) -> Optional[ExecutionPlan]:
    """Parse LLM response into ExecutionPlan; return None on failure or invalid plan."""
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
        logger.warning("Plan selection: invalid JSON %s", e)
        return None
    eligible_names = {sk.name.lower().replace("-", "_") for sk in eligible}
    if data.get("is_compound") and data.get("steps"):
        steps_raw = data.get("steps", [])
        if not isinstance(steps_raw, list) or len(steps_raw) > 4 or len(steps_raw) < 2:
            logger.warning("Plan selection: invalid steps count (need 2-4)")
            return None
        step_id_by_skill: Dict[str, int] = {}
        for i, s in enumerate(steps_raw):
            if not isinstance(s, dict):
                continue
            norm = (s.get("skill_name") or "").strip().lower().replace("-", "_")
            if norm and norm in eligible_names:
                step_id_by_skill[norm] = int(s.get("step_id", i + 1))
        steps = []
        for i, s in enumerate(steps_raw):
            if not isinstance(s, dict):
                continue
            name = (s.get("skill_name") or "").strip().lower().replace("-", "_")
            if name not in eligible_names:
                for sk in eligible:
                    if sk.name.lower().replace("-", "_") == name:
                        name = sk.name
                        break
                else:
                    logger.warning("Plan selection: step %s references unknown skill %s", i + 1, s.get("skill_name"))
                    return None
            else:
                for sk in eligible:
                    if sk.name.lower().replace("-", "_") == name:
                        name = sk.name
                        break
            raw_deps = s.get("depends_on") or []
            depends_on: List[int] = []
            for d in raw_deps:
                if isinstance(d, int):
                    depends_on.append(d)
                elif isinstance(d, str):
                    norm_d = d.strip().lower().replace("-", "_")
                    if norm_d in step_id_by_skill:
                        depends_on.append(step_id_by_skill[norm_d])
            steps.append(
                PlanStep(
                    step_id=int(s.get("step_id", i + 1)),
                    skill_name=name,
                    sub_query=(s.get("sub_query") or "").strip() or "Proceed with the task.",
                    depends_on=depends_on,
                    context_keys=s.get("context_keys") or [],
                )
            )
        if len(steps) < 2:
            return None
        return ExecutionPlan(
            is_compound=True,
            skill=None,
            confidence=0.0,
            steps=steps,
            reasoning=(data.get("reasoning") or "").strip(),
        )
    skill_name = (data.get("skill") or "").strip().lower().replace("-", "_")
    if not skill_name:
        return None
    for sk in eligible:
        if sk.name.lower().replace("-", "_") == skill_name:
            return ExecutionPlan(
                is_compound=False,
                skill=sk.name,
                confidence=float(data.get("confidence", 0)) if data.get("confidence") is not None else 0.0,
                steps=[],
                reasoning=(data.get("reasoning") or "").strip(),
            )
    if skill_name.endswith("_agent"):
        skill_name = skill_name[:-6]
        for sk in eligible:
            if sk.name.lower().replace("-", "_") == skill_name:
                return ExecutionPlan(
                    is_compound=False,
                    skill=sk.name,
                    confidence=float(data.get("confidence", 0)) if data.get("confidence") is not None else 0.0,
                    steps=[],
                    reasoning=(data.get("reasoning") or "").strip(),
                )
    logger.warning("Plan selection: unknown skill %s", data.get("skill"))
    return None


async def llm_select_skill_or_plan(
    eligible: List[Skill],
    query: str,
    editor_context: Optional[Dict[str, Any]] = None,
    conversation_context: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[ExecutionPlan]:
    """
    Compound-aware skill router: returns ExecutionPlan (single skill or multi-step).

    On parse failure or missing API key, returns None; caller should fall back to llm_select_skill().
    """
    if not eligible:
        return None
    try:
        from config.settings import settings
        model = getattr(settings, "FAST_MODEL", "anthropic/claude-3-haiku")
    except Exception:
        model = os.getenv("FAST_MODEL", "anthropic/claude-3-haiku")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("Plan selection: OPENROUTER_API_KEY not set")
        return None
    prompt = _build_plan_selection_prompt(eligible, query, editor_context, conversation_context)
    try:
        client = get_openrouter_client(api_key=api_key)
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a skill router. Respond with JSON only (no prose, no markdown fences)."},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.1,
            max_tokens=500,
        )
    except Exception as e:
        logger.warning("Plan selection: LLM call failed %s", e)
        return None
    content = (response.choices[0].message.content or "").strip()
    plan = _parse_plan_response(content, eligible)
    if plan and not plan.is_compound and plan.confidence < MIN_CONFIDENCE_FALLBACK:
        logger.info("Plan selection: low confidence %.2f for %s", plan.confidence, plan.skill)
        return ExecutionPlan(is_compound=False, skill="chat", confidence=plan.confidence, reasoning=plan.reasoning)
    return plan
