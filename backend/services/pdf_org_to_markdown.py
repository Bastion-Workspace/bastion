"""
Convert Org outline headlines to Markdown ATX headings for PDF markdown processing.

Org uses * ** *** for levels; Markdown uses # ## ###. Without this step, PDF export
treats * lines as bullet lists, so no <h1>-<h6> and heading CSS never applies.

Source normalization (fullwidth number signs, BOM) runs before Python-Markdown so
ATX lines parse as real headings. Post-markdown HTML repair promotes any remaining
<p># Title</p> blocks (e.g. fullwidth # in source that was not normalized).
"""

from __future__ import annotations

import re

_FULLWIDTH_NUMBER_SIGN = "\uff03"

_ORG_TODO_PREFIX = re.compile(
    r"^(TODO|NEXT|STARTED|WAITING|HOLD|DONE|CANCELED|CANCELLED)\s+",
    re.IGNORECASE,
)


def convert_org_headlines_to_markdown_atx(text: str) -> str:
    """
    Map Org headlines at line start (* .. ******) to # .. ###### lines.
    Skips #+ blocks. ATX headings are emitted at column 0 with blank lines
    before/after so Python-Markdown parses them as h1-h6 (not paragraphs).
    """
    if not text:
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    out: list[str] = []
    in_org_block = False

    for line in lines:
        stripped_left = line.lstrip()

        if in_org_block:
            out.append(line)
            if re.match(r"^#\+END_\S+", stripped_left, re.IGNORECASE):
                in_org_block = False
            continue

        if re.match(r"^#\+BEGIN_\S+", stripped_left, re.IGNORECASE):
            in_org_block = True
            out.append(line)
            continue

        if re.match(r"^#\+", stripped_left):
            out.append(line)
            continue

        m = re.match(r"^(\*{1,6})\s+(\S.*)$", stripped_left)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            title = _ORG_TODO_PREFIX.sub("", title)
            hashes = "#" * level
            if out and out[-1].strip():
                out.append("")
            out.append(f"{hashes} {title}")
            out.append("")
        else:
            out.append(line)

    return "\n".join(out)


# ATX run at line start: ASCII # and/or fullwidth ＃ (1–6 chars), then title.
_RE_ATX_HASH_RUN = re.compile(
    r"^(\s{0,3})((?:#|" + _FULLWIDTH_NUMBER_SIGN + r"){1,6})(\s*(?:.+)?)$"
)


def normalize_pdf_heading_hash_characters(text: str) -> str:
    """
    Replace U+FF03 fullwidth number signs with ASCII # on ATX-like lines so
    Python-Markdown emits <h1>-<h6>. Skips fenced code blocks.
    """
    if not text or _FULLWIDTH_NUMBER_SIGN not in text:
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        raw = line.rstrip("\r")
        m = _RE_ATX_HASH_RUN.match(raw)
        if m:
            hashes = m.group(2).replace(_FULLWIDTH_NUMBER_SIGN, "#")
            if hashes == "#" * len(hashes) and 1 <= len(hashes) <= 6:
                line = m.group(1) + hashes + m.group(3)
        out.append(line)
    return "\n".join(out)


def prepare_body_for_pdf_markdown(body: str, source_format: str) -> str:
    """Normalize source text before markdown.markdown() for PDF."""
    body = body.lstrip("\ufeff")
    fmt = (source_format or "markdown").lower().strip()
    if fmt == "org":
        body = convert_org_headlines_to_markdown_atx(body)
    body = normalize_pdf_heading_hash_characters(body)
    return body
