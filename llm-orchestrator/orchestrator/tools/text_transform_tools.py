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


def transform_format_tool(
    text: str,
    from_format: str = "plain",
    to_format: str = "markdown",
    user_id: str = "system",
) -> str:
    """
    Convert text between formats using pure string manipulation.

    Args:
        text: The text to convert.
        from_format: Source format: plain, markdown, bullets, numbered, json.
        to_format: Target format: plain, markdown, bullets, numbered, json.
        user_id: User ID (injected by engine if omitted).

    Returns:
        Text in the target format.
    """
    if not (text or "").strip():
        return ""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    try:
        if to_format == "plain":
            return "\n".join(lines)
        if to_format == "markdown":
            out = []
            for ln in lines:
                if ln.startswith(("-", "*", "•")) or (len(ln) > 1 and ln[0].isdigit() and ln[1] in ".):"):
                    out.append(ln)
                else:
                    out.append(ln)
            return "\n\n".join(out)
        if to_format == "bullets":
            return "\n".join("- " + ln for ln in lines)
        if to_format == "numbered":
            return "\n".join(f"{i}. {ln}" for i, ln in enumerate(lines, 1))
        if to_format == "json":
            import json
            return json.dumps({"lines": lines}, indent=2)
        return "\n".join(lines)
    except Exception as e:
        logger.warning("transform_format_tool failed: %s", e)
        return text


def merge_texts_tool(
    texts: str,
    separator: str = "\n\n",
    style: str = "concat",
    user_id: str = "system",
) -> str:
    """
    Combine multiple text sections into one. Sections in input are newline-separated or separated by '---'.

    Args:
        texts: Multiple text sections separated by newlines and/or '---'.
        separator: String to insert between sections in output (default double newline).
        style: "concat" to join as-is, "dedupe" to remove duplicate lines.

    Returns:
        Merged text.
    """
    if not (texts or "").strip():
        return ""
    raw = texts.strip().replace("---", "\n")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return ""
    if style == "dedupe":
        seen = set()
        unique = []
        for ln in lines:
            key = ln.lower()
            if key not in seen:
                seen.add(key)
                unique.append(ln)
        lines = unique
    return separator.join(lines)


async def compare_texts_tool(
    text_a: str,
    text_b: str,
    mode: str = "structural",
    user_id: str = "system",
) -> str:
    """
    Compare two texts. structural: line-by-line diff-style; semantic: LLM summary of differences.

    Args:
        text_a: First text.
        text_b: Second text.
        mode: "structural" for line-based diff, "semantic" for LLM summary of meaning differences.
        user_id: User ID (injected by engine if omitted).

    Returns:
        Comparison result as string.
    """
    if not (text_a or "").strip() and not (text_b or "").strip():
        return "Both texts are empty."
    if mode == "structural":
        lines_a = [ln for ln in (text_a or "").strip().splitlines()]
        lines_b = [ln for ln in (text_b or "").strip().splitlines()]
        from difflib import unified_diff
        diff = list(unified_diff(lines_a, lines_b, lineterm="", fromfile="a", tofile="b"))[:80]
        return "\n".join(diff) if diff else "No differences (structural)."
    if mode == "semantic":
        try:
            model = os.getenv("FAST_MODEL", "anthropic/claude-3-haiku")
            try:
                from config.settings import settings
                model = getattr(settings, "FAST_MODEL", model)
            except Exception:
                pass
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                return "Semantic comparison requires OPENROUTER_API_KEY."
            client = get_openrouter_client(api_key=api_key)
            response = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Compare the two texts. List main similarities and differences in meaning, tone, or content. Be concise. Output only the comparison."},
                    {"role": "user", "content": f"Text A:\n{text_a[:6000]}\n\nText B:\n{text_b[:6000]}"},
                ],
                model=model,
                temperature=0.2,
                max_tokens=400,
            )
            content = (response.choices[0].message.content or "").strip()
            return content or "No semantic comparison produced."
        except Exception as e:
            logger.warning("compare_texts_tool semantic failed: %s", e)
            return f"Semantic comparison failed: {e}"
    return "Unknown mode. Use structural or semantic."
