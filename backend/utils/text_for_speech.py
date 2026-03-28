"""
Normalize markdown / org-mode text for TTS (matches frontend textForSpeech.js behavior).
"""

from __future__ import annotations

import html
import re
from typing import List
from urllib.parse import urlparse

# Match frontend ttsStreamUtils.js
TTS_CHUNK_THRESHOLD_CHARS = 500
TTS_CHUNK_MAX_CHARS = 900


def _strip_html_tags(text: str) -> str:
    if not text:
        return ""
    s = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    s = re.sub(
        r"</(p|div|h[1-6]|li|tr|blockquote)>",
        ". ",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"<[^>]+>", " ", s)
    return s


def _strip_yaml_frontmatter(text: str) -> str:
    t = text.lstrip()
    if not t.startswith("---"):
        return text
    after_first = t[3:]
    if not after_first:
        return text
    nl = 0
    if after_first[0] == "\r":
        nl = 2 if len(after_first) > 1 and after_first[1] == "\n" else 1
    elif after_first[0] == "\n":
        nl = 1
    else:
        return text
    rest = t[3 + nl :]
    m = re.search(r"\n---\s*(?:\n|$)", rest)
    if not m:
        return text
    return rest[m.end() :].lstrip()


def _strip_markdown_tables(text: str) -> str:
    text = re.sub(r"^\|[\s\-:|]+\|$", " ", text, flags=re.MULTILINE)

    def pipe_row(line: str) -> str:
        line = line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            return line
        parts = [c.strip() for c in line.split("|") if c.strip()]
        return " ".join(parts)

    lines = text.split("\n")
    return "\n".join(pipe_row(L) for L in lines)


def _strip_org_tables(text: str) -> str:
    text = re.sub(r"^\|[\-+]+$", " ", text, flags=re.MULTILINE)

    def pipe_row(line: str) -> str:
        line = line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            return line
        parts = [c.strip() for c in line.split("|") if c.strip()]
        return " ".join(parts)

    lines = text.split("\n")
    return "\n".join(pipe_row(L) for L in lines)


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _bare_url_to_host(url: str) -> str:
    url = re.sub(r"[.,;:!?)]+$", "", url.strip())
    try:
        if "://" not in url:
            return ""
        u = urlparse(url)
        host = (u.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def strip_markdown_for_speech(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    s = _strip_yaml_frontmatter(text)
    s = _strip_html_tags(s)
    s = html.unescape(s)
    s = _strip_markdown_tables(s)
    s = re.sub(r"^\s*[-*_]{3,}\s*$", " ", s, flags=re.MULTILINE)
    s = re.sub(r"\[\^[^\]]*\]", " ", s)
    s = re.sub(
        r"https?://[^\s)\]>'\"<]+",
        lambda m: _bare_url_to_host(m.group(0)) or " ",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"```[\s\S]*?```", " ", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"^#{1,6}\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*\d+\.\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"^>\s?", "", s, flags=re.MULTILINE)
    s = re.sub(r"[*_~>|]", " ", s)
    return _normalize_whitespace(s)


def strip_org_for_speech(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    s = text
    s = re.sub(r":PROPERTIES:\s*[\s\S]*?:END:", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"^#\+BEGIN_[\s\S]*?#\+END_[^\n]*$", " ", s, flags=re.MULTILINE | re.IGNORECASE)
    s = re.sub(r"^#\+[A-Z_]+:\s*", "", s, flags=re.MULTILINE | re.IGNORECASE)
    s = re.sub(
        r"^\*+\s+(TODO|NEXT|STARTED|WAITING|HOLD|DONE|CANCELED|CANCELLED)?\s*",
        "",
        s,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    s = re.sub(r"\[\[([^\]]+)\]\[([^\]]+)\]\]", r"\2", s)
    s = re.sub(r"\[\[([^\]]+)\]\]", r"\1", s)
    s = re.sub(r"^\s*[-+]\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"\*(.*?)\*", r"\1", s)
    s = re.sub(r"/(.*?)/", r"\1", s)
    s = re.sub(r"=(.*?)=", r"\1", s)
    s = re.sub(r"~(.*?)~", r"\1", s)
    s = _strip_html_tags(s)
    s = html.unescape(s)
    s = _strip_org_tables(s)
    s = re.sub(r"^\s*[-*_]{3,}\s*$", " ", s, flags=re.MULTILINE)
    s = re.sub(r"\[\^[^\]]*\]", " ", s)
    s = re.sub(
        r"https?://[^\s)\]>'\"<]+",
        lambda m: _bare_url_to_host(m.group(0)) or " ",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"^\s*\d+\.\s+", "", s, flags=re.MULTILINE)
    return _normalize_whitespace(s)


def strip_text_for_speech(text: str, mode: str = "markdown") -> str:
    if mode == "org":
        return strip_org_for_speech(text)
    return strip_markdown_for_speech(text)


def split_text_for_tts(
    text: str,
    threshold: int = TTS_CHUNK_THRESHOLD_CHARS,
    max_chunk: int = TTS_CHUNK_MAX_CHARS,
) -> List[str]:
    """Sentence-ish chunking aligned with frontend splitTextForTts."""
    t = (text or "").strip()
    if not t or len(t) <= threshold:
        return [t] if t else []

    sentences: List[str] = []
    for m in re.finditer(r"[^.!?\n]+(?:[.!?]+|\n+)", t):
        seg = m.group(0).strip()
        if seg:
            sentences.append(seg)

    if not sentences:
        out: List[str] = []
        for i in range(0, len(t), max_chunk):
            chunk = t[i : i + max_chunk].strip()
            if chunk:
                out.append(chunk)
        return out

    merged: List[str] = []
    acc = ""
    for s in sentences:
        nxt = f"{acc} {s}".strip() if acc else s
        if len(nxt) <= max_chunk:
            acc = nxt
        else:
            if acc:
                merged.append(acc.strip())
            remaining = s
            while len(remaining) > max_chunk:
                merged.append(remaining[:max_chunk].strip())
                remaining = remaining[max_chunk:].strip()
            acc = remaining
    if acc:
        merged.append(acc.strip())
    return merged if merged else [t]
