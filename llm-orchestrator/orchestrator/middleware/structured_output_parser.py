"""
Structured LLM output parsing for Agent Factory pipeline steps.

Centralizes markdown fence stripping and JSON parsing for `llm_task` and `llm_agent`
steps that use `output_schema` / `json_output`.
"""

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class StructuredOutputParser:
    """Strip common LLM JSON wrappers and parse to a dict for playbook results."""

    @staticmethod
    def extract_json(content: str) -> str:
        """Strip markdown code fences and return the inner JSON string."""
        raw = (content or "").strip()
        if "```json" in raw:
            start = raw.find("```json") + 7
            end = raw.find("```", start)
            raw = raw[start:end].strip() if end != -1 else raw
        elif "```" in raw:
            start = raw.find("```") + 3
            end = raw.find("```", start)
            raw = raw[start:end].strip() if end != -1 else raw
        return raw.strip()

    @staticmethod
    def parse(
        content: str,
        schema: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from content and parse to a dict.

        Args:
            content: Raw model output (may include ```json fences).
            schema: Optional JSON Schema dict from playbook `output_schema`.
                Reserved for future validation; currently not enforced.

        Returns:
            Parsed dict, or None if content is not valid JSON object.
        """
        _ = schema  # hook for future jsonschema / Pydantic validation
        raw = StructuredOutputParser.extract_json(content)
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            logger.debug("Structured output parsed to non-dict type: %s", type(parsed).__name__)
            return None
        except json.JSONDecodeError as e:
            logger.debug("Structured output JSON parse failed: %s", e)
            return None
