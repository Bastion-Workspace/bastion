"""
Format LLM Markdown for messaging platforms.

- Telegram: CommonMark-style (**, *, `, ```) is converted to Telegram HTML
  so parse_mode=HTML renders bold, italic, code, and code blocks.
- Discord: Renders **bold**, *italic*, `code`, and ```blocks``` natively;
  no conversion needed.
- Slack: CommonMark converted to mrkdwn (*bold*, _italic_, ~strikethrough~,
  `code`, ```blocks```, <url|text> for links).
"""

import re
from typing import List, Optional, Tuple

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


# Tags produced by markdown_to_telegram_html (Telegram HTML parse_mode).
_TELEGRAM_BODY_TAG_RE = re.compile(r"<(/?)(b|i|code|pre|s)\b[^>]*>", re.I)


def _telegram_html_fragment_tags_balanced(fragment: str) -> bool:
    """True if fragment has no dangling <b>, <pre>, etc. (stack empty at end)."""
    if not fragment:
        return True
    if re.search(r"<[^>]*$", fragment):
        return False
    stack: List[str] = []
    for m in _TELEGRAM_BODY_TAG_RE.finditer(fragment):
        is_close = m.group(1) == "/"
        tag = m.group(2).lower()
        if is_close:
            if not stack or stack[-1] != tag:
                return False
            stack.pop()
        else:
            stack.append(tag)
    return len(stack) == 0


def _strip_telegram_html_to_plain(s: str) -> str:
    """Remove formatting tags for parse_mode=None fallback (avoids entity parse errors)."""
    t = _TELEGRAM_BODY_TAG_RE.sub("", s)
    return t.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def split_telegram_html_for_sends(html: str, max_len: int = 4096) -> List[Tuple[str, Optional[str]]]:
    """
    Split Telegram HTML into chunks <= max_len without breaking tag boundaries.

    Returns list of (text, parse_mode): parse_mode is \"HTML\" or None (plain text fallback
    when no balanced prefix fits in max_len, e.g. an oversized <pre> block).
    """
    if not html:
        return []
    if len(html) <= max_len:
        return [(html, "HTML")]
    out: List[Tuple[str, Optional[str]]] = []
    pos = 0
    n = len(html)
    while pos < n:
        if pos + max_len >= n:
            tail = html[pos:]
            if _telegram_html_fragment_tags_balanced(tail):
                out.append((tail, "HTML"))
            else:
                out.append((_strip_telegram_html_to_plain(tail), None))
            break
        end_limit = pos + max_len
        cut: Optional[int] = None
        for c in range(end_limit, pos, -1):
            if _telegram_html_fragment_tags_balanced(html[pos:c]):
                cut = c
                break
        if cut is not None:
            out.append((html[pos:cut], "HTML"))
            pos = cut
        else:
            frag = html[pos:end_limit]
            if _telegram_html_fragment_tags_balanced(frag):
                out.append((frag, "HTML"))
            else:
                out.append((_strip_telegram_html_to_plain(frag), None))
            pos = end_limit
    return out


def markdown_to_slack_mrkdwn(text: str) -> str:
    """
    Convert CommonMark-style Markdown to Slack mrkdwn.

    Slack mrkdwn: *bold*, _italic_, ~strikethrough~, `code`, ```blocks```,
    and <url|text> for links. Escapes & < > so literal characters are safe.
    """
    if not text or not text.strip():
        return text
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    out = escaped
    # Code blocks first (may contain other syntax)
    out = re.sub(
        r"```[\w]*\n(.*?)```",
        lambda m: "```" + m.group(1).strip() + "```",
        out,
        flags=re.DOTALL,
    )
    # Links: [text](url) -> <url|text>
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: "<" + m.group(2) + "|" + m.group(1) + ">", out)
    # Bold: **text** and __text__ -> *text*
    out = re.sub(r"\*\*(.+?)\*\*", lambda m: "*" + m.group(1) + "*", out)
    out = re.sub(r"__(.+?)__", lambda m: "*" + m.group(1) + "*", out)
    # Italic: *text* (single) and _text_ -> _text_
    out = re.sub(
        r"(?<!\*)\*(?!\*)(.+?)\*(?!\*)",
        lambda m: "_" + m.group(1) + "_",
        out,
    )
    out = re.sub(
        r"(?<!_)_(?!_)(.+?)_(?!_)",
        lambda m: "_" + m.group(1) + "_",
        out,
    )
    # Strikethrough: ~~text~~ -> ~text~
    out = re.sub(r"~~(.+?)~~", lambda m: "~" + m.group(1) + "~", out)
    return out


def format_text_for_platform(text: str, platform: str) -> str:
    """
    Return text formatted for the given platform.

    - telegram: Markdown converted to Telegram HTML (for parse_mode=HTML).
    - discord: Return as-is; Discord renders **, *, ` natively.
    - slack: Markdown converted to Slack mrkdwn.
    - other: Return as-is.
    """
    if not text:
        return text
    if platform == "telegram":
        return markdown_to_telegram_html(text)
    if platform == "slack":
        return markdown_to_slack_mrkdwn(text)
    return text

