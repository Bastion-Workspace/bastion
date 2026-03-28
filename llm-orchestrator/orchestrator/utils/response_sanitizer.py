"""Sanitize assistant message content for conversation history."""

import json
import logging
import re

logger = logging.getLogger(__name__)


def sanitize_ai_response_for_history(content: str) -> str:
    """Strip ManuscriptEdit JSON from AI responses, keeping only the conversational summary.

    Prior AI turns may contain full ManuscriptEdit JSON with original_text anchors.
    Including these in conversation history causes the LLM to recycle stale anchors
    instead of extracting fresh ones from the current manuscript.
    """
    if not content or not isinstance(content, str):
        return content or ""

    stripped = content.strip()
    if not (stripped.startswith("{") or stripped.startswith("```")):
        return content  # Plain text, not JSON

    try:
        # Handle markdown-wrapped JSON
        json_text = stripped
        if "```json" in json_text:
            match = re.search(r"```json\s*\n(.*?)\n```", json_text, re.DOTALL)
            if match:
                json_text = match.group(1).strip()
        elif json_text.startswith("```"):
            match = re.search(r"```\s*\n(.*?)\n```", json_text, re.DOTALL)
            if match:
                json_text = match.group(1).strip()

        parsed = json.loads(json_text)
        if isinstance(parsed, dict) and "operations" in parsed:
            # ManuscriptEdit JSON - extract only the conversational response
            response = parsed.get("response", "")
            if response:
                logger.debug(
                    "Stripped ManuscriptEdit JSON from AI history message (%d chars -> %d chars)",
                    len(content),
                    len(response),
                )
                return response
            return "(edit operations proposed)"
        return content  # Valid JSON but not ManuscriptEdit
    except (json.JSONDecodeError, TypeError):
        return content  # Not JSON, return as-is
