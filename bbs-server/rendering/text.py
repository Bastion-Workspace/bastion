"""
Markdown-ish to ANSI, word wrap, simple box frames.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional

from rendering.ansi import Theme


def normalize_for_telnet(text: str) -> str:
    """
    Replace common Unicode punctuation with ASCII so UTF-8 bytes are not misread
    as CP437 on clients like mTCP (em dash, smart quotes, ellipsis, etc.).
    """
    if not text:
        return text
    out = text
    pairs = (
        ("\u2014", "--"),  # em dash
        ("\u2013", "-"),  # en dash
        ("\u2012", "-"),  # figure dash
        ("\u2011", "-"),  # non-breaking hyphen
        ("\u2010", "-"),  # hyphen
        ("\u2212", "-"),  # minus sign
        ("\u2018", "'"),  # left single quotation mark
        ("\u2019", "'"),  # right single quotation mark
        ("\u201a", "'"),  # single low-9 quotation mark
        ("\u201b", "'"),  # single high-reversed-9 quotation mark
        ("\u201c", '"'),  # left double quotation mark
        ("\u201d", '"'),  # right double quotation mark
        ("\u201e", '"'),  # double low-9 quotation mark
        ("\u201f", '"'),  # double high-reversed-9 quotation mark
        ("\u00ab", '"'),  # left-pointing double angle quotation
        ("\u00bb", '"'),  # right-pointing double angle quotation
        ("\u2039", "'"),  # single left-pointing angle quotation
        ("\u203a", "'"),  # single right-pointing angle quotation
        ("\u2026", "..."),  # horizontal ellipsis
        ("\u00a0", " "),  # no-break space
        ("\u2009", " "),  # thin space
        ("\u2002", " "),  # en space
        ("\u2003", " "),  # em space
        ("\u202f", " "),  # narrow no-break space
        ("\u2007", " "),  # figure space
        ("\u00ad", ""),  # soft hyphen
        ("\u200b", ""),  # zero width space
        ("\u200c", ""),  # zero width non-joiner
        ("\u200d", ""),  # zero width joiner
        ("\ufeff", ""),  # byte order mark
        ("\u2022", "*"),  # bullet
        ("\u00b7", "*"),  # middle dot
        ("\u2043", "-"),  # hyphen bullet
        ("\u2192", "->"),  # rightwards arrow
        ("\u2190", "<-"),  # leftwards arrow
    )
    for src, repl in pairs:
        out = out.replace(src, repl)
    return out


def format_header_datetime(include_time: bool = True) -> str:
    """Weekday, date, and optional clock (snapshot when the screen is drawn; no polling)."""
    n = datetime.now()
    if include_time:
        return n.strftime("%a %b %d, %Y %H:%M")
    return n.strftime("%a %b %d, %Y")


def format_header_context(display_name: str, include_time: bool = True) -> str:
    """User-facing name plus date line for section_header context (title | this)."""
    name = (display_name or "").strip() or "User"
    return f"{name} | {format_header_datetime(include_time)}"


def _build_header_label(title: str, context: Optional[str], max_label_chars: int) -> str:
    if not context:
        inner = f" {title} "
        if len(inner) <= max_label_chars:
            return inner
        return inner[: max_label_chars - 2].rstrip() + "... "
    sep = " | "
    inner = f" {title}{sep}{context} "
    if len(inner) <= max_label_chars:
        return inner
    room_ctx = max_label_chars - len(f" {title}{sep}") - 2
    if room_ctx < 10:
        t2 = title[: max_label_chars - 4].rstrip()
        return f" {t2}... "
    ctx = context[:room_ctx] + ("..." if len(context) > room_ctx else "")
    return f" {title}{sep}{ctx} "


def word_wrap(text: str, width: int) -> List[str]:
    if width < 10:
        width = 10
    text = normalize_for_telnet(text)
    lines: List[str] = []
    for para in text.split("\n"):
        if not para.strip():
            lines.append("")
            continue
        cur = ""
        for word in para.split():
            if not cur:
                cur = word
            elif len(cur) + 1 + len(word) <= width:
                cur += " " + word
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
    return lines


def draw_box(title: str, inner_width: int, theme: Theme, subtitle: str = "") -> List[str]:
    """Top box line is `title`; optional `subtitle` row (e.g. weekday + date + time).

    Horizontal borders (top/bottom) use slow blink + bright cyan when ANSI is enabled.
    Name and date rows use stable colors (no blink) so they do not flicker on vintage terminals.
    """
    rows = [title]
    sub = (subtitle or "").strip()
    if sub:
        rows.append(sub)
    inner = max(len(r) for r in rows)
    w = max(inner_width, inner + 4)
    top = "+" + "-" * w + "+"
    bot = "+" + "-" * w + "+"
    cyan = theme.fg_bright_cyan
    blink_on = theme.slow_blink or ""
    blink_off = theme.slow_blink_off or ""
    top_line = blink_on + cyan + top + blink_off + theme.reset
    bot_line = blink_on + cyan + bot + blink_off + theme.reset
    out = [top_line]
    for i, r in enumerate(rows):
        inner_cell = r.ljust(w - 2)[: w - 2]
        if i == 0:
            body = theme.title_color() + inner_cell + theme.reset
        else:
            body = theme.dim + inner_cell + theme.reset
        mid = cyan + "| " + theme.reset + body + cyan + " |" + theme.reset
        out.append(mid)
    out.append(bot_line)
    return out


def section_header(title: str, width: int, theme: Theme, *, context: Optional[str] = None) -> str:
    """Themed header bar using ASCII '=' (CP437 / vintage telnet safe; not Unicode box-drawing)."""
    bar_char = "="
    max_label = max(4, width - 4)
    label = _build_header_label(title, context, max_label_chars=max_label)
    side = max(2, (width - len(label)) // 2)
    line = bar_char * side + label + bar_char * side
    if len(line) < width:
        line += bar_char * (width - len(line))
    return theme.title_color() + line[:width] + theme.reset


def _replace_image_md(m: re.Match) -> str:
    alt = (m.group(1) or "").strip()
    return f"[Image: {alt}]" if alt else "[Image]"


def markdown_to_ansi(text: str, theme: Theme) -> str:
    if not text:
        return ""
    out = normalize_for_telnet(text)
    out = re.sub(r"!\[([^\]]*)\]\([^)]+\)", _replace_image_md, out)
    out = re.sub(
        r"\*\*([^*]+)\*\*",
        lambda m: theme.bold + m.group(1) + theme.reset,
        out,
    )
    out = re.sub(
        r"^###\s+(.+)$",
        lambda m: theme.header2_color() + m.group(1) + theme.reset,
        out,
        flags=re.MULTILINE,
    )
    out = re.sub(
        r"^##\s+(.+)$",
        lambda m: theme.header2_color() + m.group(1) + theme.reset,
        out,
        flags=re.MULTILINE,
    )
    out = re.sub(
        r"^#\s+(.+)$",
        lambda m: theme.title_color() + m.group(1) + theme.reset,
        out,
        flags=re.MULTILINE,
    )
    out = re.sub(
        r"`([^`]+)`",
        lambda m: theme.code_color() + m.group(1) + theme.reset,
        out,
    )
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", out)
    return out
