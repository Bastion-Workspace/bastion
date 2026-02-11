"""
LLM-powered query classifier for research workflow routing.
Classifies queries into collection_search, factual_query, or exploratory_search for optimal execution.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import ValidationError

from orchestrator.agents.base_agent import BaseAgent
from orchestrator.models.query_classification_models import QueryPlan
from config.settings import settings

logger = logging.getLogger(__name__)

# Fallback when no user model is set; prefer user_chat_model (chat sidebar selection) then user_fast_model
DEFAULT_CLASSIFIER_MODEL = "x-ai/grok-4.1-fast"

# Canonical content_type values (backend expects these). Map plurals/typos -> canonical.
CONTENT_TYPE_CANONICAL = {
    "comic": "comic",
    "comics": "comic",
    "photo": "photo",
    "photos": "photo",
    "pictures": "photo",
    "artwork": "artwork",
    "artworks": "artwork",
    "meme": "meme",
    "memes": "meme",
    "screenshot": "screenshot",
    "screenshots": "screenshot",
    "medical": "medical",
    "documentation": "documentation",
    "maps": "maps",
    "other": "other",
}


def _format_messages_for_classifier(messages: List[Any], limit: int = 3) -> str:
    """Extract last N messages and format as conversation context string."""
    if not messages or limit <= 0:
        return ""
    last = messages[-limit:]
    parts = []
    for msg in last:
        content = ""
        if hasattr(msg, "content") and msg.content:
            content = str(msg.content).strip()
        elif isinstance(msg, dict):
            content = str(msg.get("content", "") or "").strip()
        if not content or len(content) > 500:
            continue
        role = "user"
        if hasattr(msg, "type") and msg.type == "ai":
            role = "assistant"
        elif isinstance(msg, dict) and msg.get("role") == "assistant":
            role = "assistant"
        parts.append(f"{role}: {content}")
    return "\n".join(parts) if parts else ""


async def classify_query_intent(
    query: str,
    user_model: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    messages: Optional[List[Any]] = None,
) -> QueryPlan:
    """
    Classify query into one of three execution paths using a fast LLM call.

    Args:
        query: User's research query.
        user_model: Optional model override (e.g. user_chat_model from metadata = chat sidebar selection).
        metadata: Optional metadata dict; may contain user_chat_model, user_fast_model.
        messages: Optional conversation history for context (last 2-3 messages used).

    Returns:
        QueryPlan with query_type, extracted filters, and execution hints.
    """
    metadata = metadata or {}
    # Use selected chat sidebar model first, then fast model, then default
    model = user_model or metadata.get("user_chat_model") or metadata.get("user_fast_model") or DEFAULT_CLASSIFIER_MODEL

    base_agent = BaseAgent("query_classifier")
    state = {"metadata": metadata, "shared_memory": metadata.get("shared_memory", {})}
    llm = base_agent._get_llm(temperature=0.1, model=model, state=state)

    system_prompt = """You are a query classifier for a document and image retrieval system.

The user has a personal collection containing:
- Comics (Dilbert, Calvin and Hobbes, Garfield, etc.)
- Photos (personal, events, locations)
- Books (PDFs, ebooks, technical docs)
- Medical images (X-rays, MRI scans)
- Screenshots and technical documentation
- Reference materials and articles

**Allowed content_type values (use EXACTLY one of these, or null):**
comic | photo | artwork | meme | screenshot | medical | documentation | maps | other

Map user wording to the correct value: plurals (comics, photos) -> singular (comic, photo); fix typos and case; synonyms (comic strips, pictures, images) -> comic or photo as appropriate. If the user says "comics" or "comic strips", use content_type="comic". If they say "photos" or "pictures", use content_type="photo". Output only the canonical value from the list above.

Classify the user's query into EXACTLY ONE of these three types:

1. **collection_search**: User wants to FIND or LIST specific items FROM their collection (browse, show me, get items)
   - They want to SEE items matching filters: series name, author, date, or content type (comic, photo, etc.)
   - AND they mention a concept or topic (e.g. "about brains", "showing dogs")
   - Use ONLY when the intent is to retrieve/list items, NOT when they ask for analysis or summary.
   - Examples: "Dilbert comics about brains", "Show me my photos from December", "Find Garfield strips with dogs"
   - NOT collection_search: "How is my weight loss looking?", "Summarize my diet for last month", "How did I do with exercise?" (these need document content analysis -> exploratory_search)
   - Extract: series, author, content_type (from allowed list), date_range, concept. If user says "give me N more" or "N more", set requested_count to N (integer).
   - Set: query_type="collection_search", has_collection_filters=true, use_hybrid_search=true, skip_permission=true, estimated_time="1-2 seconds"

2. **factual_query**: User wants information ABOUT something (facts, biography, definitions)
   - Questions like "who", "what", "when", "tell me about"
   - They are asking ABOUT an entity, not searching their collection for items
   - Examples: "Tell me about Dilbert", "Who created Calvin and Hobbes?", "What is a brain tumor?"
   - Set: query_type="factual_query", quick_local_check=true, needs_web_search=true, estimated_time="3-5 seconds"

3. **exploratory_search**: Use when user wants ANALYSIS, SUMMARY, or REVIEW of their data; or when ambiguous/broad
   - "How is my [X] looking/going for [period]?", "Summarize my [X]", "How did I do with [X]?", "Review my progress"
   - Personal data review: weight loss, diet, exercise, habits, progress over time (needs full document retrieval and synthesis)
   - No specific series/author/date for a simple list, or mixed intent
   - Examples: "How is my weight loss looking for the last month?", "Comics about brains" (no series), "Find medical information"
   - Set: query_type="exploratory_search", estimated_time="10-15 seconds"

When in doubt: if the user asks for analysis, summary, trend, or "how is X looking/going" over their data, use exploratory_search (full research pipeline with document content). Use collection_search only for "find/show/list items" with clear filters.

Respond with valid JSON only. Include: query_type, reasoning, and for collection_search also: series, author, content_type (exactly one of: comic, photo, artwork, meme, screenshot, medical, documentation, maps, other), date_range, concept, has_collection_filters, use_hybrid_search, skip_permission, requested_count (integer when user asks for N more). Include estimated_time for all."""

    conversation_context = _format_messages_for_classifier(messages or [], limit=3)
    if conversation_context:
        user_prompt = f"""CONVERSATION CONTEXT:
{conversation_context}

Classify this query: "{query}"

Extract query_type, reasoning, and any filters (series, author, content_type, date_range, concept). Return only valid JSON."""
    else:
        user_prompt = f'Classify this query: "{query}"\n\nExtract query_type, reasoning, and any filters (series, author, content_type, date_range, concept). Return only valid JSON.'

    try:
        # Prefer structured output when available
        structured_llm = llm.with_structured_output(QueryPlan)
        plan = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        if isinstance(plan, QueryPlan):
            _normalize_plan_content_type(plan)
            logger.info(f"Query classified as: {plan.query_type}, reasoning: {plan.reasoning[:80]}...")
            return plan
        # If adapter returned dict, build QueryPlan
        plan_dict = plan if isinstance(plan, dict) else getattr(plan, "model_dump", lambda: plan.dict())()
        if plan_dict.get("content_type"):
            plan_dict["content_type"] = CONTENT_TYPE_CANONICAL.get(
                str(plan_dict["content_type"]).lower().strip(),
                plan_dict["content_type"],
            )
            if plan_dict["content_type"] not in CONTENT_TYPE_CANONICAL.values():
                plan_dict["content_type"] = None
        return QueryPlan(**plan_dict)
    except (ValidationError, TypeError, AttributeError) as e:
        logger.warning(f"Structured classification failed, using JSON fallback: {e}")
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        content = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_json_classification(content)
        return QueryPlan(**parsed)


def _normalize_plan_content_type(plan: QueryPlan) -> None:
    """Normalize plan.content_type to canonical value (comic not comics, etc.)."""
    if not plan.content_type or not isinstance(plan.content_type, str):
        return
    raw = plan.content_type.lower().strip()
    canonical = CONTENT_TYPE_CANONICAL.get(raw, raw)
    if canonical in CONTENT_TYPE_CANONICAL.values():
        plan.content_type = canonical
    else:
        plan.content_type = None


def _parse_json_classification(content: str) -> Dict[str, Any]:
    """Extract JSON from LLM response and normalize to QueryPlan fields."""
    text = content.strip()
    if "```json" in text:
        m = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    elif "```" in text:
        m = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {}
    # Normalize keys and set defaults
    query_type = data.get("query_type", "exploratory_search")
    if query_type not in ("collection_search", "factual_query", "exploratory_search"):
        query_type = "exploratory_search"
    content_type_raw = data.get("content_type")
    content_type = None
    if content_type_raw and isinstance(content_type_raw, str):
        content_type = CONTENT_TYPE_CANONICAL.get(content_type_raw.lower().strip(), content_type_raw.lower().strip())
        if content_type not in CONTENT_TYPE_CANONICAL.values():
            content_type = None
    return {
        "query_type": query_type,
        "reasoning": data.get("reasoning", ""),
        "has_collection_filters": data.get("has_collection_filters", False),
        "series": data.get("series"),
        "author": data.get("author"),
        "content_type": content_type,
        "date_range": data.get("date_range"),
        "concept": data.get("concept"),
        "use_hybrid_search": data.get("use_hybrid_search", False),
        "quick_local_check": data.get("quick_local_check", False),
        "needs_web_search": data.get("needs_web_search", False),
        "skip_permission": data.get("skip_permission", False),
        "estimated_time": data.get("estimated_time", ""),
        "requested_count": data.get("requested_count"),
    }
