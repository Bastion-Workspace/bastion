"""
Section scoping utilities for cursor-aware context extraction.

Finds markdown heading sections (e.g. ## Chapter 3, ## Design Notes) and extracts
the section containing the cursor plus adjacent sections, for use in playbook
prompts and agents that need to scope down long documents.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SectionRange:
    """Represents a section delimited by a markdown heading."""
    heading_text: str
    level: int  # number of # characters
    start: int  # char offset of heading line
    end: int    # char offset of next heading (or EOF)


def find_heading_sections(text: str, max_level: int = 2) -> List[SectionRange]:
    """
    Find all markdown heading sections up to max_level (default: ##).
    Headings match ^#{1,max_level}\\s+.+$ (e.g. # Title, ## Section).
    """
    if not text:
        return []
    pattern = re.compile(r"^(#{1," + str(max(1, min(6, max_level))) + r"})\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return []
    ranges: List[SectionRange] = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        level = len(m.group(1))
        heading_text = m.group(0).strip()
        ranges.append(SectionRange(heading_text=heading_text, level=level, start=start, end=end))
    return ranges


def locate_section_at_cursor(sections: List[SectionRange], cursor_offset: int) -> int:
    """Return index of section containing cursor, or -1."""
    if cursor_offset < 0 or not sections:
        return -1
    for i, r in enumerate(sections):
        if r.start <= cursor_offset < r.end:
            return i
    return -1


def extract_scoped_context(
    text: str,
    cursor_offset: int,
    max_level: int = 2,
    adjacent: int = 1,
) -> Dict[str, Any]:
    """
    Extract cursor-aware section context from text.

    Returns dict with:
      current_section: content of the section containing the cursor
      current_heading: heading text of that section
      current_section_index: 0-based index of the section
      previous_section: content of the section before current (empty if none)
      next_section: content of the section after current (empty if none)
      adjacent_sections: concatenated content of prev + current + next sections
      total_sections: total number of sections found
    """
    empty = {
        "current_section": "",
        "current_heading": "",
        "current_section_index": -1,
        "previous_section": "",
        "next_section": "",
        "adjacent_sections": "",
        "total_sections": 0,
    }
    if not text:
        return empty
    sections = find_heading_sections(text, max_level=max_level)
    total = len(sections)
    if total == 0:
        return {**empty, "total_sections": 0}
    idx = locate_section_at_cursor(sections, cursor_offset)
    if idx < 0:
        # Cursor is before the first heading (e.g., in YAML frontmatter).
        # Default to the first section so agents see chapter 1 context.
        idx = 0
    r = sections[idx]
    current_section = text[r.start:r.end].strip()
    current_heading = r.heading_text
    previous_section = ""
    if idx > 0:
        prev_r = sections[idx - 1]
        previous_section = text[prev_r.start:prev_r.end].strip()
    next_section = ""
    if idx + 1 < total:
        next_r = sections[idx + 1]
        next_section = text[next_r.start:next_r.end].strip()
    lo = max(0, idx - adjacent)
    hi = min(total, idx + adjacent + 1)
    parts = [text[sections[i].start:sections[i].end].strip() for i in range(lo, hi)]
    adjacent_sections = "\n\n".join(parts)
    return {
        "current_section": current_section,
        "current_heading": current_heading,
        "current_section_index": idx,
        "previous_section": previous_section,
        "next_section": next_section,
        "adjacent_sections": adjacent_sections,
        "total_sections": total,
    }


def _heading_plain_text(heading_line: str) -> str:
    """Strip markdown # prefix and whitespace from a heading line."""
    if not heading_line:
        return ""
    s = heading_line.strip()
    while s.startswith("#"):
        s = s[1:].lstrip()
    return s.strip()


def extract_named_section(text: str, heading_query: str, max_level: int = 3) -> str:
    """
    Return full content of the first section whose heading contains heading_query
    (case-insensitive substring). Empty string if not found.
    """
    if not text or not (heading_query or "").strip():
        return ""
    query = heading_query.strip().lower()
    sections = find_heading_sections(text, max_level=max_level)
    for r in sections:
        plain = _heading_plain_text(r.heading_text).lower()
        if query in plain:
            return text[r.start : r.end].strip()
    return ""


def scope_reference_by_heading(
    text: str,
    manuscript_heading: str,
    adjacent: int = 1,
    max_level: int = 3,
) -> Dict[str, str]:
    """
    Match reference headings to manuscript_heading and return toc/current/adjacent/previous/next slices.

    Returns dict with keys: toc, current, adjacent, previous, next.
    previous: section before the matching heading (empty if none).
    next: section after the matching heading (empty if none).
    When no section matches manuscript_heading, current, adjacent, previous, next are empty; toc still set.
    """
    empty = {"toc": "", "current": "", "adjacent": "", "previous": "", "next": ""}
    if not text:
        return empty
    sections = find_heading_sections(text, max_level=max_level)
    if not sections:
        return empty
    toc = "\n".join(s.heading_text for s in sections)
    if not (manuscript_heading or "").strip():
        return {"toc": toc, "current": text, "adjacent": text, "previous": "", "next": ""}
    target = _heading_plain_text(manuscript_heading).lower()
    if not target:
        return {"toc": toc, "current": "", "adjacent": "", "previous": "", "next": ""}
    match_idx = -1
    for i, r in enumerate(sections):
        if _heading_plain_text(r.heading_text).lower() == target:
            match_idx = i
            break
    if match_idx < 0:
        return {"toc": toc, "current": "", "adjacent": "", "previous": "", "next": ""}
    r = sections[match_idx]
    current = text[r.start : r.end].strip()
    total = len(sections)
    previous = ""
    if match_idx > 0:
        prev_r = sections[match_idx - 1]
        previous = text[prev_r.start : prev_r.end].strip()
    next_sect = ""
    if match_idx + 1 < total:
        next_r = sections[match_idx + 1]
        next_sect = text[next_r.start : next_r.end].strip()
    lo = max(0, match_idx - adjacent)
    hi = min(total, match_idx + adjacent + 1)
    parts = [text[sections[j].start : sections[j].end].strip() for j in range(lo, hi)]
    adjacent_text = "\n\n".join(parts)
    return {"toc": toc, "current": current, "adjacent": adjacent_text, "previous": previous, "next": next_sect}
