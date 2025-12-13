"""
Fiction Utilities - Shared utilities for fiction editing and processing

Contains chapter detection, text processing, outline detection, and other
fiction-specific utility functions used across multiple agents and subgraphs.
"""

import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================
# Chapter Scope Utilities
# ============================================

@dataclass
class ChapterRange:
    """Represents a chapter range in a manuscript"""
    heading_text: str
    chapter_number: Optional[int]
    start: int
    end: int


CHAPTER_PATTERN = re.compile(r"^##\s+Chapter\s+(\d+)\b.*$", re.MULTILINE)


def find_chapter_ranges(text: str) -> List[ChapterRange]:
    """Find all chapter ranges in text."""
    if not text:
        return []
    matches = list(CHAPTER_PATTERN.finditer(text))
    if not matches:
        return []
    ranges: List[ChapterRange] = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chapter_num: Optional[int] = None
        try:
            chapter_num = int(m.group(1))
        except Exception:
            chapter_num = None
        ranges.append(ChapterRange(heading_text=m.group(0), chapter_number=chapter_num, start=start, end=end))
    return ranges


def locate_chapter_index(ranges: List[ChapterRange], cursor_offset: int) -> int:
    """Locate which chapter contains the cursor."""
    if cursor_offset < 0:
        return -1
    for i, r in enumerate(ranges):
        if r.start <= cursor_offset < r.end:
            return i
    return -1


def get_adjacent_chapters(ranges: List[ChapterRange], idx: int) -> Tuple[Optional[ChapterRange], Optional[ChapterRange]]:
    """Get previous and next chapters."""
    prev_c = ranges[idx - 1] if 0 <= idx - 1 < len(ranges) else None
    next_c = ranges[idx + 1] if 0 <= idx + 1 < len(ranges) else None
    return prev_c, next_c


# ============================================
# Text Processing Utilities
# ============================================

def slice_hash(text: str) -> str:
    """Match frontend simple hash (31-bit rolling, hex)."""
    try:
        h = 0
        for ch in text:
            h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        return format(h, 'x')
    except Exception:
        return ""


def strip_frontmatter_block(text: str) -> str:
    """Strip YAML frontmatter from text."""
    try:
        return re.sub(r'^---\s*\n[\s\S]*?\n---\s*\n', '', text, flags=re.MULTILINE)
    except Exception:
        return text


def frontmatter_end_index(text: str) -> int:
    """Return the end index of a leading YAML frontmatter block if present, else 0."""
    try:
        m = re.match(r'^(---\s*\n[\s\S]*?\n---\s*\n)', text, flags=re.MULTILINE)
        if m:
            return m.end()
        return 0
    except Exception:
        return 0


def unwrap_json_response(content: str) -> str:
    """Extract raw JSON from LLM output if wrapped in code fences or prose."""
    try:
        json.loads(content)
        return content
    except Exception:
        pass
    try:
        text = content.strip()
        text = re.sub(r"^```(?:json)?\s*\n([\s\S]*?)\n```\s*$", r"\1", text)
        try:
            json.loads(text)
            return text
        except Exception:
            pass
        start = text.find('{')
        if start == -1:
            return content
        brace = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == '{':
                brace += 1
            elif ch == '}':
                brace -= 1
                if brace == 0:
                    snippet = text[start:i+1]
                    try:
                        json.loads(snippet)
                        return snippet
                    except Exception:
                        break
        return content
    except Exception:
        return content


# ============================================
# Outline Detection Utilities
# ============================================

def normalize_for_overlap_check(text: str) -> str:
    """Normalize text for similarity/overlap checks (whitespace + lowercase)."""
    try:
        return " ".join((text or "").split()).lower()
    except Exception:
        return (text or "").lower()


def looks_like_outline_copied(generated_prose: str, outline_text: Optional[str]) -> bool:
    """
    Heuristic to detect outline copying/paraphrase leakage into generated prose.
    We intentionally bias toward catching obvious copy/paste of beats.
    """
    if not generated_prose or not outline_text:
        return False

    g = normalize_for_overlap_check(generated_prose)
    o = normalize_for_overlap_check(outline_text)

    # Quick reject: if outline is tiny, avoid over-triggering.
    if len(o) < 80 or len(g) < 200:
        return False

    # Check direct beat-line reuse: if multiple long outline lines appear inside prose.
    outline_lines = [ln.strip() for ln in outline_text.splitlines() if ln.strip()]
    long_lines = [ln for ln in outline_lines if len(ln) >= 35]
    reused = 0
    for ln in long_lines[:40]:  # cap work
        ln_n = normalize_for_overlap_check(ln)
        if len(ln_n) >= 30 and ln_n in g:
            reused += 1
            if reused >= 2:
                return True

    # Similarity fallback: high similarity between outline and prose is suspicious.
    try:
        ratio = SequenceMatcher(None, o[:4000], g[:4000]).ratio()
        return ratio >= 0.35
    except Exception:
        return False


# ============================================
# Chapter Extraction Utilities
# ============================================

def extract_chapter_number_from_request(request: str) -> Optional[int]:
    """Extract chapter number from user request like 'Chapter 1', 'generate chapter 2', etc."""
    if not request:
        return None
    # Pattern: "Chapter N" or "chapter N" (case insensitive)
    patterns = [
        r'(?:^|\s)(?:chapter|ch\.?)\s+(\d+)(?:\s|$|[^\d])',
        r'(?:^|\s)(\d+)(?:\s+chapter|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, request, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue
    return None


def extract_chapter_range_from_request(request: str) -> Optional[Tuple[int, int]]:
    """
    Extract chapter range from user request.
    Returns (start_chapter, end_chapter) inclusive, or None if not a range request.
    
    Examples:
    - "generate the first few chapters" -> (1, 3)  # Default: first 3
    - "generate chapters 1-3" -> (1, 3)
    - "generate chapters 1 through 5" -> (1, 5)
    - "generate the first 5 chapters" -> (1, 5)
    - "generate chapter 1" -> None (single chapter, use extract_chapter_number_from_request)
    """
    if not request:
        return None
    
    request_lower = request.lower()
    
    # Pattern 1: Explicit range "chapters 1-3" or "chapters 1 through 5"
    range_patterns = [
        r'(?:chapters?|ch\.?)\s+(\d+)\s*[-–—]\s*(\d+)',  # "chapters 1-3"
        r'(?:chapters?|ch\.?)\s+(\d+)\s+through\s+(\d+)',  # "chapters 1 through 5"
        r'(?:chapters?|ch\.?)\s+(\d+)\s+to\s+(\d+)',  # "chapters 1 to 5"
    ]
    for pattern in range_patterns:
        match = re.search(pattern, request_lower)
        if match:
            try:
                start = int(match.group(1))
                end = int(match.group(2))
                if start <= end:
                    return (start, end)
            except (ValueError, IndexError):
                continue
    
    # Pattern 2: "first N chapters" or "first few chapters"
    first_patterns = [
        r'first\s+(\d+)\s+chapters?',  # "first 5 chapters"
        r'first\s+few\s+chapters?',  # "first few chapters" -> default to 3
    ]
    for pattern in first_patterns:
        match = re.search(pattern, request_lower)
        if match:
            try:
                if match.group(1):
                    count = int(match.group(1))
                    return (1, count)
                else:
                    # "first few" -> default to 3 chapters
                    return (1, 3)
            except (ValueError, IndexError):
                if 'few' in request_lower:
                    return (1, 3)
    
    # Pattern 3: "chapters N through M" (alternative wording)
    through_pattern = r'chapter\s+(\d+)\s+through\s+chapter\s+(\d+)'
    match = re.search(through_pattern, request_lower)
    if match:
        try:
            start = int(match.group(1))
            end = int(match.group(2))
            if start <= end:
                return (start, end)
        except (ValueError, IndexError):
            pass
    
    return None


# ============================================
# Heading Manipulation Utilities
# ============================================

def ensure_chapter_heading(text: str, chapter_number: int) -> str:
    """Ensure the text begins with '## Chapter N' heading."""
    try:
        if re.match(r'^\s*#{1,6}\s*Chapter\s+\d+\b', text, flags=re.IGNORECASE):
            return text
        heading = f"## Chapter {chapter_number}\n\n"
        return heading + text.lstrip('\n')
    except Exception:
        return text


def extract_character_name(profile_body: str) -> str:
    """Extract character name from profile body (typically starts with '# Name')."""
    try:
        # Look for markdown header pattern: # Name or ## Name
        match = re.match(r'^\s*#{1,6}\s+([A-Z][a-zA-Z\s]+)', profile_body.strip())
        if match:
            return match.group(1).strip()
        # Fallback: look for "Name:" pattern
        name_match = re.search(r'(?:^|\n)\s*(?:name|Name|NAME)[:：]\s*([A-Z][a-zA-Z\s]+)', profile_body, re.IGNORECASE)
        if name_match:
            return name_match.group(1).strip()
        return "Unknown Character"
    except Exception:
        return "Unknown Character"


def detect_chapter_heading_at_position(manuscript: str, position: int) -> Optional[Tuple[int, int, str]]:
    """
    Detect if there's a chapter heading at or near the given position.
    Returns (heading_start, heading_end, heading_text) if found, None otherwise.
    """
    if position < 0 or position >= len(manuscript):
        return None
    
    # Look backwards from position to find chapter heading (within 200 chars)
    search_start = max(0, position - 200)
    search_text = manuscript[search_start:position + 50]
    
    # Find all chapter headings in the search area
    matches = list(CHAPTER_PATTERN.finditer(search_text))
    if not matches:
        return None
    
    # Find the last heading before or at the position
    for match in reversed(matches):
        heading_start = search_start + match.start()
        heading_end = search_start + match.end()
        heading_text = match.group(0)
        
        # Check if this heading is at or very close to the position (within 10 chars)
        if heading_end <= position + 10:
            return (heading_start, heading_end, heading_text)
    
    return None


def strip_chapter_heading_from_text(text: str) -> Tuple[str, bool]:
    """
    Strip chapter heading from the start of text if present.
    Returns (stripped_text, had_heading).
    """
    stripped = text.lstrip()
    match = re.match(r'^#{1,6}\s+Chapter\s+\d+\b.*?\n+', stripped, re.MULTILINE | re.IGNORECASE)
    if match:
        # Remove the heading and any trailing newlines
        remaining = stripped[match.end():].lstrip('\n')
        return remaining, True
    return text, False

