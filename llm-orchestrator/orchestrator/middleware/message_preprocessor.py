"""
Unified conversation message preprocessing for LLM context.

Centralizes strip-tool-prefix, data-URI filtering, assistant JSON sanitization,
and lookback slicing so agents and subgraphs share one pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from orchestrator.utils.message_sanitizer import strip_tool_actions_prefix
from orchestrator.utils.writing_subgraph_utilities import sanitize_ai_response_for_history

logger = logging.getLogger(__name__)


def _filter_data_uris(content: str) -> str:
    """Remove large data URIs and HTML chart blocks from message text."""
    if not content or not isinstance(content, str):
        return content

    filtered = content
    data_image_pattern = r"!\[([^\]]*)\]\(data:image/[^)]+\)"

    def replace_image(match: re.Match) -> str:
        alt_text = match.group(1) if match.groups() else "Image"
        return f"![{alt_text}](<image_removed_from_context>)"

    filtered = re.sub(data_image_pattern, replace_image, filtered)
    html_chart_pattern = r"```html:chart\n.*?\n```"
    filtered = re.sub(
        html_chart_pattern,
        "```html:chart\n<chart_removed_from_context>\n```",
        filtered,
        flags=re.DOTALL,
    )
    other_data_pattern = r"data:[^)]+"
    filtered = re.sub(other_data_pattern, "<data_uri_removed_from_context>", filtered)
    return filtered


class MessagePreprocessor:
    """Stateless helpers for building LangChain message lists from raw history."""

    @staticmethod
    def filter_large_data_from_content(content: str) -> str:
        """Remove data URIs and large chart blocks (delegates to shared filter)."""
        return _filter_data_uris(content)

    @staticmethod
    def preprocess_history(
        messages: List[Any],
        limit: int = 10,
        *,
        filter_data_uris: bool = True,
        sanitize_ai_responses: bool = False,
        strip_tool_prefixes: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Extract recent turns as role/content dicts with optional sanitization.

        If limit <= 0, returns [] (avoids messages[-0:] returning full list).
        """
        if not messages or limit <= 0:
            return []
        try:
            history: List[Dict[str, str]] = []
            for msg in messages[-limit:]:
                if not hasattr(msg, "content"):
                    continue
                raw = msg.content
                content = raw if isinstance(raw, str) else str(raw)
                role = (
                    "assistant"
                    if hasattr(msg, "type") and getattr(msg, "type", None) == "ai"
                    else "user"
                )
                if strip_tool_prefixes:
                    content = strip_tool_actions_prefix(content)
                if filter_data_uris:
                    content = _filter_data_uris(content)
                if sanitize_ai_responses and role == "assistant":
                    content = sanitize_ai_response_for_history(content)
                history.append({"role": role, "content": content})
            return history
        except Exception as e:
            logger.error("Failed to preprocess conversation history: %s", e)
            return []

    @staticmethod
    def build_conversational_messages(
        system_prompt: str,
        user_prompt: str,
        messages_list: List[Any],
        look_back_limit: int = 10,
        datetime_context: str = "",
        *,
        filter_data_uris: bool = True,
        sanitize_ai_responses: bool = False,
        strip_tool_prefixes: bool = True,
    ) -> List[Any]:
        """System + datetime + recent history + current user prompt."""
        messages: List[Any] = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=datetime_context),
        ]
        if messages_list:
            conversation_history = MessagePreprocessor.preprocess_history(
                messages_list,
                limit=look_back_limit,
                filter_data_uris=filter_data_uris,
                sanitize_ai_responses=sanitize_ai_responses,
                strip_tool_prefixes=strip_tool_prefixes,
            )
            for msg in conversation_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=user_prompt))
        return messages

    @staticmethod
    def build_editing_messages(
        system_prompt: str,
        context_parts: List[str],
        current_request: str,
        messages_list: List[Any],
        datetime_context: str = "",
        look_back_limit: int = 6,
        *,
        filter_data_uris: bool = True,
        sanitize_ai_responses: bool = True,
        strip_tool_prefixes: bool = True,
    ) -> List[Any]:
        """System + datetime + history (dedupe last if matches request) + context + request."""
        messages: List[Any] = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=datetime_context),
        ]
        if messages_list:
            conversation_history = MessagePreprocessor.preprocess_history(
                messages_list,
                limit=look_back_limit,
                filter_data_uris=filter_data_uris,
                sanitize_ai_responses=sanitize_ai_responses,
                strip_tool_prefixes=strip_tool_prefixes,
            )
            if conversation_history and conversation_history[-1].get("content") == current_request:
                conversation_history = conversation_history[:-1]
            for msg in conversation_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        if context_parts:
            messages.append(HumanMessage(content="".join(context_parts)))
        if current_request:
            messages.append(HumanMessage(content=current_request))
        return messages
