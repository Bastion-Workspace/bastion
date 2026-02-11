"""
Text transform tools for bridging context between plan steps.

Uses fast LLM for summarization and extraction. No gRPC; local to orchestrator.
"""

import logging
import os

from orchestrator.utils.openrouter_client import get_openrouter_client

logger = logging.getLogger(__name__)


async def summarize_text_tool(
    text: str,
    max_sentences: int = 3,
    style: str = "bullets",
    user_id: str = "system",
) -> str:
    """
    Distill long text to a short summary using the fast model.

    Args:
        text: The text to summarize.
        max_sentences: Maximum number of sentences or bullet points (default 3).
        style: "bullets" for bullet points, "prose" for a short paragraph.
        user_id: User ID (injected by engine if omitted).

    Returns:
        Summarized text.
    """
    if not (text or "").strip():
        return ""
    try:
        model = os.getenv("FAST_MODEL", "anthropic/claude-3-haiku")
        try:
            from config.settings import settings
            model = getattr(settings, "FAST_MODEL", model)
        except Exception:
            pass
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return text[:500] + "..." if len(text) > 500 else text
        client = get_openrouter_client(api_key=api_key)
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
        content = (response.choices[0].message.content or "").strip()
        return content or text[:500]
    except Exception as e:
        logger.warning("summarize_text_tool failed: %s", e)
        return text[:500] + "..." if len(text) > 500 else text


async def extract_structured_data_tool(
    text: str,
    extract_types: str = "entities,dates,urls",
    user_id: str = "system",
) -> str:
    """
    Extract structured data (entities, dates, URLs, etc.) from text using the fast model.

    Args:
        text: The text to extract from.
        extract_types: Comma-separated types: entities, dates, urls, key_values (default entities,dates,urls).
        user_id: User ID (injected by engine if omitted).

    Returns:
        Extracted data as a formatted string (e.g. key-value lines or list).
    """
    if not (text or "").strip():
        return ""
    try:
        model = os.getenv("FAST_MODEL", "anthropic/claude-3-haiku")
        try:
            from config.settings import settings
            model = getattr(settings, "FAST_MODEL", model)
        except Exception:
            pass
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return "Extraction requires OPENROUTER_API_KEY."
        client = get_openrouter_client(api_key=api_key)
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"Extract the following from the user's text: {extract_types}. Output a clear list or key-value format. Output only the extracted data, no preamble."},
                {"role": "user", "content": text[:12000]},
            ],
            model=model,
            temperature=0.1,
            max_tokens=500,
        )
        content = (response.choices[0].message.content or "").strip()
        return content or "No structured data extracted."
    except Exception as e:
        logger.warning("extract_structured_data_tool failed: %s", e)
        return f"Extraction failed: {e}"
