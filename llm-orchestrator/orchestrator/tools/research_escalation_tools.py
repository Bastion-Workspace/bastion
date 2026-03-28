"""
Research escalation tool - Invoke the full Research Engine from an LLM agent step.

Allows the default ReAct agent to delegate deep multi-round research to the
specialized Research Engine (FullResearchAgent) via a single tool call.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


class EscalateToResearchInputs(BaseModel):
    """Required inputs for escalate_to_research."""
    query: str = Field(description="The research query to investigate thoroughly (e.g. a question or topic)")


class EscalateToResearchOutputs(BaseModel):
    """Typed outputs for escalate_to_research_tool."""
    response: str = Field(description="Research findings text")
    citations: Optional[List[str]] = Field(default=None, description="Source citations when available")
    formatted: str = Field(description="Human-readable research results for LLM/chat")


async def escalate_to_research_tool(
    query: str,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the full Research Engine on the given query. Use when the user needs
    thorough, multi-source research with document search, web search, and synthesis.
    Returns the research response and optional citations.
    """
    try:
        from orchestrator.engines.research_engine import ResearchEngine

        metadata: Dict[str, Any] = {"user_id": user_id}
        if _pipeline_metadata:
            for key in ("conversation_id", "shared_memory"):
                if key in _pipeline_metadata:
                    metadata[key] = _pipeline_metadata[key]
            # Pass datetime context so research uses user's timezone (not UTC)
            if "user_timezone" in _pipeline_metadata:
                metadata["user_timezone"] = _pipeline_metadata["user_timezone"]
            if "include_datetime_context" in _pipeline_metadata:
                metadata["include_datetime_context"] = _pipeline_metadata["include_datetime_context"]
        # Ensure shared_memory has user_timezone for _get_datetime_context(state)
        shared = metadata.get("shared_memory") or {}
        if "user_timezone" not in shared and metadata.get("user_timezone"):
            shared = {**shared, "user_timezone": metadata["user_timezone"]}
            metadata["shared_memory"] = shared
        engine = ResearchEngine()
        result = await engine.process(query=query, metadata=metadata)

        response_text = result.get("response", "")
        if isinstance(response_text, dict):
            response_text = response_text.get("response", response_text.get("message", "")) or ""
        else:
            response_text = str(response_text) if response_text else ""

        citations = result.get("citations")
        if citations is not None and not isinstance(citations, list):
            if isinstance(citations, str):
                try:
                    citations = json.loads(citations) if citations.strip().startswith("[") else [citations]
                except json.JSONDecodeError:
                    citations = [citations]
            else:
                citations = list(citations) if citations else None

        formatted_parts = [response_text or "No research results returned."]
        if citations:
            formatted_parts.append("\n\nSources / citations:")
            for i, c in enumerate(citations[:20], 1):
                formatted_parts.append(f"  {i}. {c}" if isinstance(c, str) else f"  {i}. {json.dumps(c)}")

        formatted = "\n".join(formatted_parts)

        return {
            "response": response_text or "",
            "citations": citations,
            "formatted": formatted,
        }
    except Exception as e:
        logger.exception("escalate_to_research failed: %s", e)
        return {
            "response": "",
            "citations": None,
            "formatted": f"Research escalation failed: {e}",
        }


register_action(
    name="escalate_to_research",
    category="research",
    description="Run thorough multi-source research on a query (documents, web, synthesis). Use when the user needs in-depth research with citations.",
    inputs_model=EscalateToResearchInputs,
    params_model=None,
    outputs_model=EscalateToResearchOutputs,
    tool_function=escalate_to_research_tool,
)
