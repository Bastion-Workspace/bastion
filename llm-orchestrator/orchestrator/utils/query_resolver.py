"""
Follow-up query resolution for research workflow.

Detects short, context-dependent queries (e.g. "How about 5 more?") and rewrites
them into self-contained queries using conversation history.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from orchestrator.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

FOLLOW_UP_CONTINUATIONS = [
    "more", "another", "also", "again", "else", "additional", "next", "other",
]
FOLLOW_UP_PRONOUNS = ["it", "them", "those", "that", "these", "same"]
MAX_WORDS_FOR_FOLLOW_UP = 7
DEFAULT_RESOLVER_MODEL = "x-ai/grok-4.1-fast"


def _get_message_content(msg: Any) -> str:
    if hasattr(msg, "content") and msg.content:
        return str(msg.content).strip()
    if isinstance(msg, dict):
        return str(msg.get("content", "") or "").strip()
    return ""


async def resolve_follow_up_query(
    query: str,
    messages: Optional[List[Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Detect follow-up queries and rewrite them into self-contained queries using
    conversation context. Returns the original query if not a follow-up.

    Args:
        query: Current user query.
        messages: Conversation history (LangChain messages or dicts with content).
        metadata: Optional metadata (user_chat_model, user_fast_model).

    Returns:
        Resolved query string (rewritten or original).
    """
    metadata = metadata or {}
    query_stripped = (query or "").strip()
    if not query_stripped:
        return query_stripped

    words = query_stripped.lower().split()
    if len(words) > MAX_WORDS_FOR_FOLLOW_UP:
        return query_stripped

    has_continuation = any(w in FOLLOW_UP_CONTINUATIONS for w in words)
    has_pronoun = any(w in FOLLOW_UP_PRONOUNS for w in words)
    has_numeric_more = any(
        w.isdigit() and i + 1 < len(words) and words[i + 1] == "more"
        for i, w in enumerate(words)
    ) or "more" in words

    if not (has_continuation or has_pronoun or has_numeric_more):
        return query_stripped

    if not messages or len(messages) < 2:
        logger.debug("Follow-up likely but no conversation context, returning original query")
        return query_stripped

    last_messages = messages[-4:] if len(messages) > 4 else messages
    context_parts = []
    for msg in last_messages:
        content = _get_message_content(msg)
        if not content or len(content) > 800:
            continue
        if isinstance(msg, AIMessage) or (isinstance(msg, dict) and msg.get("role") == "assistant"):
            context_parts.append(f"Assistant: {content}")
        else:
            context_parts.append(f"User: {content}")

    if not context_parts:
        return query_stripped

    conversation_context = "\n".join(context_parts)
    model = (
        metadata.get("user_fast_model")
        or metadata.get("user_chat_model")
        or DEFAULT_RESOLVER_MODEL
    )
    base_agent = BaseAgent("query_resolver")
    state = {"metadata": metadata, "shared_memory": metadata.get("shared_memory", {})}
    llm = base_agent._get_llm(temperature=0.1, model=model, state=state)

    system_prompt = """You resolve follow-up queries into a single, self-contained search query.

Given the current user message and recent conversation, output ONLY the resolved query: a full question or request that could be understood without the conversation. Preserve intent and any numbers (e.g. "5 more" -> "5 more XKCD comics about gravity"). One line, no explanation."""

    user_prompt = f"""Conversation:
{conversation_context}

Current user message: "{query_stripped}"

Resolved query:"""

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        content = getattr(response, "content", None) or str(response)
        resolved = (content or "").strip().strip('"')
        if resolved and resolved != query_stripped:
            logger.info(f"Resolved follow-up query: '{query_stripped[:50]}...' -> '{resolved[:80]}...'")
            return resolved
    except Exception as e:
        logger.debug(f"Follow-up resolution failed: {e}, using original query")

    return query_stripped
