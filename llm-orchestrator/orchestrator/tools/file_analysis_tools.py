"""
File Analysis Tools - Deterministic text analysis for LangGraph agents
Provides word count, line count, character count, and other text metrics
"""

import logging
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


# ── I/O models for file analysis tools ──────────────────────────────────────

class AnalyzeTextMetricsInputs(BaseModel):
    """Required inputs for analyze_text_metrics."""
    text: str = Field(description="Text content to analyze")


class AnalyzeTextMetricsParams(BaseModel):
    """Optional parameters."""
    include_advanced: bool = Field(default=False, description="Include advanced metrics (averages)")


class AnalyzeTextMetricsOutputs(BaseModel):
    """Outputs for analyze_text_metrics."""
    word_count: int = Field(description="Word count")
    line_count: int = Field(description="Line count")
    non_empty_line_count: int = Field(description="Non-empty line count")
    character_count: int = Field(description="Character count")
    character_count_no_spaces: int = Field(description="Character count excluding spaces")
    paragraph_count: int = Field(description="Paragraph count")
    sentence_count: int = Field(description="Sentence count")
    avg_words_per_sentence: Optional[float] = Field(default=None, description="Average words per sentence (if include_advanced)")
    avg_words_per_paragraph: Optional[float] = Field(default=None, description="Average words per paragraph (if include_advanced)")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def analyze_text_metrics(
    text: str,
    include_advanced: bool = False,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Analyze raw text content and return metrics
    
    Args:
        text: Text content to analyze
        include_advanced: If True, include advanced metrics (averages, etc.)
        user_id: User ID for logging
        
    Returns:
        Dictionary with text analysis metrics:
        - word_count: int
        - line_count: int
        - non_empty_line_count: int
        - character_count: int
        - character_count_no_spaces: int
        - paragraph_count: int
        - sentence_count: int
        - avg_words_per_sentence: float (if include_advanced)
        - avg_words_per_paragraph: float (if include_advanced)
    """
    try:
        logger.info(f"📊 Analyzing text metrics: {len(text):,} chars, include_advanced={include_advanced}")
        
        # Get backend client
        client = await get_backend_tool_client()
        
        # Call backend analysis service via gRPC
        metrics = await client.analyze_text_content(
            content=text,
            include_advanced=include_advanced,
            user_id=user_id
        )
        
        if "error" in metrics:
            logger.error(f"Text analysis failed: {metrics['error']}")
            return {**metrics, "formatted": f"Text analysis failed: {metrics['error']}"}
        
        logger.info(f"✅ Text analysis complete: {metrics.get('word_count', 0):,} words, {metrics.get('sentence_count', 0):,} sentences, {metrics.get('paragraph_count', 0):,} paragraphs")
        wc, sc, pc = metrics.get("word_count", 0), metrics.get("sentence_count", 0), metrics.get("paragraph_count", 0)
        formatted = f"Text metrics: {wc} words, {sc} sentences, {pc} paragraphs."
        return {**metrics, "formatted": formatted}
        
    except Exception as e:
        logger.error(f"Error analyzing text metrics: {e}")
        err = str(e)
        return {
            "word_count": 0,
            "line_count": 0,
            "non_empty_line_count": 0,
            "character_count": 0,
            "character_count_no_spaces": 0,
            "paragraph_count": 0,
            "sentence_count": 0,
            "error": err,
            "formatted": f"Text analysis failed: {err}"
        }


register_action(
    name="analyze_text_metrics",
    category="analysis",
    description="Analyze raw text and return word/line/paragraph/sentence metrics",
    inputs_model=AnalyzeTextMetricsInputs,
    params_model=AnalyzeTextMetricsParams,
    outputs_model=AnalyzeTextMetricsOutputs,
    tool_function=analyze_text_metrics,
)
