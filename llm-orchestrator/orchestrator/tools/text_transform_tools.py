"""
Text transform tools for bridging context between plan steps.

Uses fast LLM for summarization and extraction. No gRPC; local to orchestrator.
"""

import logging
import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from orchestrator.utils.openrouter_client import get_openrouter_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


class TextTransformOutputs(BaseModel):
    """Legacy minimal outputs; prefer specific output models below."""
    formatted: str = Field(description="Human-readable result")
    summary: str = Field(default="", description="Summary or extracted content")


class SummarizeTextOutputs(BaseModel):
    """Outputs for summarize_text_tool."""
    summary: str = Field(description="Summarized text")
    original_length: int = Field(description="Character count of input text")
    summary_length: int = Field(description="Character count of summary")
    formatted: str = Field(description="Human-readable result")


class ExtractStructuredDataOutputs(BaseModel):
    """Outputs for extract_structured_data_tool."""
    extracted: str = Field(description="Extracted structured content")
    source_text_length: int = Field(description="Character count of source text")
    formatted: str = Field(description="Human-readable result")


class TransformFormatOutputs(BaseModel):
    """Outputs for transform_format_tool."""
    content: str = Field(description="Transformed text content")
    source_format: str = Field(description="Source format used")
    target_format: str = Field(description="Target format used")
    formatted: str = Field(description="Human-readable result")


class MergeTextsOutputs(BaseModel):
    """Outputs for merge_texts_tool."""
    merged: str = Field(description="Combined text")
    source_count: int = Field(description="Number of source sections merged")
    formatted: str = Field(description="Human-readable result")


class CompareTextsOutputs(BaseModel):
    """Outputs for compare_texts_tool."""
    comparison: str = Field(description="Comparison result text")
    similarity_score: Optional[float] = Field(default=None, description="Similarity score if available")
    formatted: str = Field(description="Human-readable result")


async def summarize_text_tool(
    text: str,
    max_sentences: int = 3,
    style: str = "bullets",
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Distill long text to a short summary. Returns dict with summary, original_length, summary_length, formatted."""
    orig_len = len((text or "").strip())
    if not (text or "").strip():
        return {"formatted": "", "summary": "", "original_length": 0, "summary_length": 0}
    try:
        model = os.getenv("FAST_MODEL", "anthropic/claude-3-haiku")
        try:
            from config.settings import settings
            model = getattr(settings, "FAST_MODEL", model)
        except Exception:
            pass
        from orchestrator.utils.llm_credentials_from_metadata import get_openrouter_credentials
        api_key, base_url = get_openrouter_credentials(_pipeline_metadata)
        if not api_key:
            content = text[:500] + "..." if len(text) > 500 else text
            return {"formatted": content, "summary": content, "original_length": orig_len, "summary_length": len(content)}
        client = get_openrouter_client(api_key=api_key, base_url=base_url)
        style_instruction = "as bullet points" if style == "bullets" else "as a short paragraph"
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"Summarize the user's text in at most {max_sentences} sentences or points, {style_instruction}. Output only the summary, no preamble."},
                {"role": "user", "content": text[:15000]},
            ],
            model=model,
            temperature=0.2,
            max_tokens=400,
        )
        content = (response.choices[0].message.content or "").strip() or text[:500]
        return {"formatted": content, "summary": content, "original_length": orig_len, "summary_length": len(content)}
    except Exception as e:
        logger.warning("summarize_text_tool failed: %s", e)
        content = text[:500] + "..." if len(text) > 500 else text
        return {"formatted": content, "summary": content, "original_length": orig_len, "summary_length": len(content)}


async def extract_structured_data_tool(
    text: str,
    extract_types: str = "entities,dates,urls",
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract structured data from text. Returns dict with extracted, source_text_length, formatted."""
    src_len = len((text or "").strip())
    if not (text or "").strip():
        return {"formatted": "", "extracted": "", "source_text_length": 0}
    try:
        model = os.getenv("FAST_MODEL", "anthropic/claude-3-haiku")
        try:
            from config.settings import settings
            model = getattr(settings, "FAST_MODEL", model)
        except Exception:
            pass
        from orchestrator.utils.llm_credentials_from_metadata import get_openrouter_credentials
        api_key, base_url = get_openrouter_credentials(_pipeline_metadata)
        if not api_key:
            msg = "Extraction requires OpenRouter API key (configure in Settings > AI Models)."
            return {"formatted": msg, "extracted": msg, "source_text_length": src_len}
        client = get_openrouter_client(api_key=api_key, base_url=base_url)
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"Extract the following from the user's text: {extract_types}. Output a clear list or key-value format. Output only the extracted data, no preamble."},
                {"role": "user", "content": text[:12000]},
            ],
            model=model,
            temperature=0.1,
            max_tokens=500,
        )
        content = (response.choices[0].message.content or "").strip() or "No structured data extracted."
        return {"formatted": content, "extracted": content, "source_text_length": src_len}
    except Exception as e:
        logger.warning("extract_structured_data_tool failed: %s", e)
        msg = f"Extraction failed: {e}"
        return {"formatted": msg, "extracted": msg, "source_text_length": src_len}


def transform_format_tool(
    text: str,
    from_format: str = "plain",
    to_format: str = "markdown",
    user_id: str = "system",
) -> Dict[str, Any]:
    """Convert text between formats. Returns dict with content, source_format, target_format, formatted."""
    if not (text or "").strip():
        return {"formatted": "", "content": "", "source_format": from_format, "target_format": to_format}
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return {"formatted": "", "content": "", "source_format": from_format, "target_format": to_format}
    try:
        out_str = ""
        if to_format == "plain":
            out_str = "\n".join(lines)
        elif to_format == "markdown":
            out = [ln for ln in lines]
            out_str = "\n\n".join(out)
        elif to_format == "bullets":
            out_str = "\n".join("- " + ln for ln in lines)
        elif to_format == "numbered":
            out_str = "\n".join(f"{i}. {ln}" for i, ln in enumerate(lines, 1))
        elif to_format == "json":
            import json
            out_str = json.dumps({"lines": lines}, indent=2)
        else:
            out_str = "\n".join(lines)
        return {"formatted": out_str, "content": out_str, "source_format": from_format, "target_format": to_format}
    except Exception as e:
        logger.warning("transform_format_tool failed: %s", e)
        return {"formatted": text, "content": text, "source_format": from_format, "target_format": to_format}


def merge_texts_tool(
    texts: str,
    separator: str = "\n\n",
    style: str = "concat",
    user_id: str = "system",
) -> Dict[str, Any]:
    """Combine multiple text sections into one. Returns dict with merged, source_count, formatted."""
    if not (texts or "").strip():
        return {"formatted": "", "merged": "", "source_count": 0}
    raw = texts.strip().replace("---", "\n")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return {"formatted": "", "merged": "", "source_count": 0}
    if style == "dedupe":
        seen = set()
        unique = []
        for ln in lines:
            key = ln.lower()
            if key not in seen:
                seen.add(key)
                unique.append(ln)
        lines = unique
    out = separator.join(lines)
    return {"formatted": out, "merged": out, "source_count": len(lines)}


async def compare_texts_tool(
    text_a: str,
    text_b: str,
    mode: str = "structural",
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare two texts. Returns dict with comparison, similarity_score (optional), formatted."""
    if not (text_a or "").strip() and not (text_b or "").strip():
        msg = "Both texts are empty."
        return {"formatted": msg, "comparison": msg, "similarity_score": None}
    if mode == "structural":
        lines_a = [ln for ln in (text_a or "").strip().splitlines()]
        lines_b = [ln for ln in (text_b or "").strip().splitlines()]
        from difflib import unified_diff
        diff = list(unified_diff(lines_a, lines_b, lineterm="", fromfile="a", tofile="b"))[:80]
        out = "\n".join(diff) if diff else "No differences (structural)."
        return {"formatted": out, "comparison": out, "similarity_score": None}
    if mode == "semantic":
        try:
            model = os.getenv("FAST_MODEL", "anthropic/claude-3-haiku")
            try:
                from config.settings import settings
                model = getattr(settings, "FAST_MODEL", model)
            except Exception:
                pass
            from orchestrator.utils.llm_credentials_from_metadata import get_openrouter_credentials
            api_key, base_url = get_openrouter_credentials(_pipeline_metadata)
            if not api_key:
                msg = "Semantic comparison requires OpenRouter API key (configure in Settings > AI Models)."
                return {"formatted": msg, "comparison": msg, "similarity_score": None}
            client = get_openrouter_client(api_key=api_key, base_url=base_url)
            response = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Compare the two texts. List main similarities and differences in meaning, tone, or content. Be concise. Output only the comparison."},
                    {"role": "user", "content": f"Text A:\n{text_a[:6000]}\n\nText B:\n{text_b[:6000]}"},
                ],
                model=model,
                temperature=0.2,
                max_tokens=400,
            )
            content = (response.choices[0].message.content or "").strip() or "No semantic comparison produced."
            return {"formatted": content, "comparison": content, "similarity_score": None}
        except Exception as e:
            logger.warning("compare_texts_tool semantic failed: %s", e)
            msg = f"Semantic comparison failed: {e}"
            return {"formatted": msg, "comparison": msg, "similarity_score": None}
    msg = "Unknown mode. Use structural or semantic."
    return {"formatted": msg, "comparison": msg, "similarity_score": None}


class SummarizeTextInputs(BaseModel):
    text: str = Field(description="Text to summarize")


class ExtractStructuredDataInputs(BaseModel):
    text: str = Field(description="Text to extract from")
    extract_types: str = Field(default="entities,dates,urls", description="Comma-separated types")


class TransformFormatInputs(BaseModel):
    text: str = Field(description="Text to convert")
    from_format: str = Field(default="plain", description="Source format")
    to_format: str = Field(default="markdown", description="Target format")


class MergeTextsInputs(BaseModel):
    texts: str = Field(description="Text sections separated by newlines or ---")
    separator: str = Field(default="\n\n", description="Separator between sections")
    style: str = Field(default="concat", description="concat or dedupe")


class CompareTextsInputs(BaseModel):
    text_a: str = Field(description="First text")
    text_b: str = Field(description="Second text")
    mode: str = Field(default="structural", description="structural or semantic")


register_action(name="summarize_text", category="text", description="Summarize text", inputs_model=SummarizeTextInputs, outputs_model=SummarizeTextOutputs, tool_function=summarize_text_tool)
register_action(name="extract_structured_data", category="text", description="Extract structured data from text", inputs_model=ExtractStructuredDataInputs, outputs_model=ExtractStructuredDataOutputs, tool_function=extract_structured_data_tool)
register_action(name="transform_format", category="text", description="Convert text between formats", inputs_model=TransformFormatInputs, outputs_model=TransformFormatOutputs, tool_function=transform_format_tool)
register_action(name="merge_texts", category="text", description="Combine text sections", inputs_model=MergeTextsInputs, outputs_model=MergeTextsOutputs, tool_function=merge_texts_tool)
register_action(name="compare_texts", category="text", description="Compare two texts", inputs_model=CompareTextsInputs, outputs_model=CompareTextsOutputs, tool_function=compare_texts_tool)
