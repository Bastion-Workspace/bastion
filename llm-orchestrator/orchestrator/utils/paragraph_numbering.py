"""
Paragraph numbering utility for fiction editing.

Splits chapter text into paragraphs, assigns [P1], [P2], ... IDs,
and returns numbered text plus offset mapping for two-phase editing.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)

CHAPTER_HEADING_PATTERN = re.compile(r"^##\s+Chapter\s+\d+", re.MULTILINE)


@dataclass
class ParagraphInfo:
    """One paragraph with ID and exact offsets in the original (un-numbered) text."""

    id: str
    start: int
    end: int
    text: str
    is_heading: bool


def number_paragraphs(text: str) -> Tuple[str, List[ParagraphInfo]]:
    """
    Split text on blank lines (\\n\\n boundaries), assign [P1], [P2], ... IDs.

    Returns:
        numbered_text: The chapter text with [Pn] prefixes before each paragraph.
        paragraph_list: List of ParagraphInfo with id, start, end, text, is_heading.

    Headings (## Chapter N lines) get their own paragraph entry and is_heading=True.
    """
    if not text or not text.strip():
        return "", []

    paragraphs: List[ParagraphInfo] = []
    numbered_parts: List[str] = []
    paragraph_id = 0
    start = 0
    i = 0
    n = len(text)

    while i < n:
        # Skip leading blank lines
        while i < n and text[i] in "\n\r":
            i += 1
        if i >= n:
            break
        block_start = i
        # Find end of this paragraph (next \n\n or end of text)
        while i < n:
            if text[i] == "\n":
                # Peek: is next char also newline or end? Then paragraph ends here
                j = i + 1
                while j < n and text[j] in " \t":
                    j += 1
                if j >= n or text[j] == "\n":
                    break
            i += 1
        else:
            i = n
        block_end = i
        block_text = text[block_start:block_end].rstrip()
        block_end_actual = block_start + len(block_text)
        if not block_text:
            continue
        paragraph_id += 1
        pid = f"P{paragraph_id}"
        is_heading = bool(CHAPTER_HEADING_PATTERN.match(block_text.strip()))
        paragraphs.append(
            ParagraphInfo(
                id=pid,
                start=block_start,
                end=block_end_actual,
                text=block_text,
                is_heading=is_heading,
            )
        )
        numbered_parts.append(f"[{pid}] {block_text}")
        i = block_end
        if i < n and text[i] == "\n":
            i += 1

    numbered_text = "\n\n".join(numbered_parts)
    return numbered_text, paragraphs
