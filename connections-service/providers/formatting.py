"""
Format LLM Markdown for messaging platforms.

- Telegram: CommonMark-style (**, *, `, ```) is converted to Telegram HTML
  so parse_mode=HTML renders bold, italic, code, and code blocks.
- Discord: Renders **bold**, *italic*, `code`, and ```blocks``` natively;
  no conversion needed.
"""

import re

# Telegram allows: <b>, <strong>, <i>, <em>, <u>, <ins>, <s>, <strike>, <del>, <code>, <pre>.
# We escape & < > in the whole string first, then add tags, so literal characters are safe.


def markdown_to_telegram_html(text: str) -> str:
    """
    Convert CommonMark-style Markdown to Telegram-safe HTML.

    Handles: **bold**, *italic*, __bold__, _italic_, ~~strikethrough~~,
    `inline code`, and ```fenced code blocks```.
    Escapes & < > in the original text so literal characters do not break parsing.
    """
    if not text or not text.strip():
        return text
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    out = escaped
    # Process code blocks first (they may contain backticks and asterisks)
    out = re.sub(
        r"```[\w]*\n(.*?)```",
        lambda m: "<pre>" + m.group(1).strip() + "</pre>",
        out,
        flags=re.DOTALL,
    )
    # Inline code
    out = re.sub(r"`([^`]+)`", lambda m: "<code>" + m.group(1) + "</code>", out)
    # Bold: **text**
    out = re.sub(r"\*\*(.+?)\*\*", lambda m: "<b>" + m.group(1) + "</b>", out)
    # Bold: __text__
    out = re.sub(r"__(.+?)__", lambda m: "<b>" + m.group(1) + "</b>", out)
    # Italic: *text* (single asterisk)
    out = re.sub(
        r"(?<!\*)\*(?!\*)(.+?)\*(?!\*)",
        lambda m: "<i>" + m.group(1) + "</i>",
        out,
    )
    # Italic: _text_
    out = re.sub(
        r"(?<!_)_(?!_)(.+?)_(?!_)",
        lambda m: "<i>" + m.group(1) + "</i>",
        out,
    )
    # Strikethrough: ~~text~~
    out = re.sub(r"~~(.+?)~~", lambda m: "<s>" + m.group(1) + "</s>", out)
    return out


def format_text_for_platform(text: str, platform: str) -> str:
    """
    Return text formatted for the given platform.

    - telegram: Markdown converted to Telegram HTML (for parse_mode=HTML).
    - discord: Return as-is; Discord renders **, *, ` natively.
    - other: Return as-is.
    """
    if not text:
        return text
    if platform == "telegram":
        return markdown_to_telegram_html(text)
    return text

