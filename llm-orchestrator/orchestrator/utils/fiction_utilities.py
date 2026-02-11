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
from typing import Dict, List, Optional, Tuple, Union

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


# ============================================
# Outline Extraction Utilities
# ============================================

def extract_chapter_outline(outline_body: str, chapter_identifier: Union[int, str]) -> Optional[str]:
    """
    Extract outline section for a specific chapter number or special section name.
    
    Args:
        outline_body: Full outline text
        chapter_identifier: Chapter number (int) or special section name (str) like "Introduction", "Prologue", "Epilogue"
        
    Returns:
        Chapter outline text if found, None otherwise
    """
    if not outline_body or not chapter_identifier:
        return None
    
    # Handle special section names (Introduction, Prologue, Epilogue)
    if isinstance(chapter_identifier, str):
        # Match special section headers: ## Introduction, ## Prologue, ## Epilogue
        # Case-insensitive, with optional colon and title
        special_pattern = rf"(?i)(?:^|\n)##?\s*\b{re.escape(chapter_identifier)}\b[:\s]*(.*?)(?=\n##?\s*(?:Chapter\s+)?\b\d+\b|\n##?\s*\b(?:Introduction|Prologue|Epilogue)\b|\Z)"
        match = re.search(special_pattern, outline_body, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    # Handle numbered chapters: ## Chapter N or ## N
    # Improved regex: Match chapter header with word boundaries to avoid partial matches
    chapter_number = chapter_identifier
    chapter_pattern = rf"(?i)(?:^|\n)##?\s*(?:Chapter\s+)?\b{chapter_number}\b[:\s]*(.*?)(?=\n##?\s*(?:Chapter\s+)?\b\d+\b|\n##?\s*\b(?:Introduction|Prologue|Epilogue)\b|\Z)"
    match = re.search(chapter_pattern, outline_body, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_story_overview(outline_body: str) -> Optional[str]:
    """
    Extract story overview/synopsis from outline (content before first chapter header).
    
    Chapter headers include both numbered chapters (## Chapter 1, ## 1) and special sections
    (## Introduction, ## Prologue, ## Epilogue). The overview is everything before the first
    of any of these.
    
    Args:
        outline_body: Full outline text
        
    Returns:
        Story overview text if found, None otherwise
    """
    if not outline_body:
        return None
    
    # Find the first chapter-like header (numbered chapter OR special section)
    # Pattern matches: ## Chapter N, ## N, ## Introduction, ## Prologue, ## Epilogue
    first_chapter_pattern = r"(?i)(?:^|\n)##?\s*(?:Chapter\s+)?\b\d+\b|\n##?\s*\b(?:Introduction|Prologue|Epilogue)\b"
    match = re.search(first_chapter_pattern, outline_body)
    
    if match:
        # Extract everything before the first chapter header
        overview = outline_body[:match.start()].strip()
        return overview if overview else None
    
    # If no chapter headers found, return the entire outline as overview
    return outline_body.strip() if outline_body.strip() else None


def extract_book_map(outline_body: str) -> List[Tuple[Union[int, str], str]]:
    """
    Extract a map of all chapters and their headers from the outline.
    
    Includes both numbered chapters (## Chapter 1, ## 1) and special sections
    (## Introduction, ## Prologue, ## Epilogue).
    
    Args:
        outline_body: Full outline text
        
    Returns:
        List of (chapter_identifier, header_text) tuples where identifier is either
        an int (for numbered chapters) or str (for special sections like "Introduction").
        Special sections come first, then numbered chapters sorted by number.
    """
    if not outline_body:
        return []
    
    book_map = []
    
    # First, find all numbered chapters: ## Chapter N or ## N
    numbered_pattern = r"(?i)(?:^|\n)(##?\s*(?:Chapter\s+)?(\d+)\b[:\s]*(.*?))(?=\n|$)"
    numbered_matches = re.finditer(numbered_pattern, outline_body, re.MULTILINE)
    
    for match in numbered_matches:
        chapter_num_str = match.group(2)
        header_text = match.group(1).strip()
        try:
            chapter_num = int(chapter_num_str)
            book_map.append((chapter_num, header_text))
        except (ValueError, IndexError):
            continue
    
    # Then, find special sections: ## Introduction, ## Prologue, ## Epilogue
    special_pattern = r"(?i)(?:^|\n)(##?\s*\b(Introduction|Prologue|Epilogue)\b[:\s]*(.*?))(?=\n|$)"
    special_matches = re.finditer(special_pattern, outline_body, re.MULTILINE)
    
    for match in special_matches:
        section_name = match.group(2)  # "Introduction", "Prologue", or "Epilogue"
        header_text = match.group(1).strip()
        book_map.append((section_name, header_text))
    
    # Sort: special sections first (in order: Introduction, Prologue, Epilogue), then numbered chapters
    def sort_key(item):
        identifier, _ = item
        if isinstance(identifier, str):
            # Special sections: Introduction=0, Prologue=1, Epilogue=2, others=3
            order_map = {"Introduction": 0, "Prologue": 1, "Epilogue": 2}
            return (0, order_map.get(identifier, 3))
        else:
            # Numbered chapters come after special sections
            return (1, identifier)
    
    book_map.sort(key=sort_key)
    return book_map


# ============================================
# Tracked Items and Status Block Utilities
# ============================================

def get_tracked_items(frontmatter: dict) -> dict:
    """
    Extract tracked items configuration from outline frontmatter.
    
    Args:
        frontmatter: Outline document frontmatter dictionary
        
    Returns:
        Dictionary with keys: 'characters', 'locations', 'items', 'relationships'
        Each value is a list of item names (strings)
    """
    if not frontmatter:
        return {}
    
    tracked_items = frontmatter.get("tracked_items", {})
    if not isinstance(tracked_items, dict):
        return {}
    
    result = {
        "characters": [],
        "locations": [],
        "items": [],
        "relationships": []
    }
    
    for category in result.keys():
        items = tracked_items.get(category, [])
        if isinstance(items, list):
            result[category] = [str(item) for item in items if item]
    
    return result


def validate_tracked_item(item_name: str, category: str, tracked_items: dict) -> bool:
    """
    Validate that an item exists in the tracked items list for the given category.
    
    Args:
        item_name: Name of the item to validate
        category: Category ('characters', 'locations', 'items', 'relationships')
        tracked_items: Dictionary from get_tracked_items()
        
    Returns:
        True if item exists in category, False otherwise
    """
    if not item_name or not category or not tracked_items:
        return False
    
    category_items = tracked_items.get(category, [])
    if not isinstance(category_items, list):
        return False
    
    return item_name in category_items


def extract_status_block(chapter_text: str) -> Optional[str]:
    """
    Extract existing status block from chapter text.
    
    Status blocks appear immediately after chapter heading, before summary paragraph.
    Format:
    ### Status
    - [Item name]: [Status description]
    - [Item name]: [Status description]
    
    Args:
        chapter_text: Full chapter text including heading
        
    Returns:
        Status block text if found (including "### Status" header), None otherwise
    """
    if not chapter_text:
        return None
    
    # Pattern: Match "### Status" or "Status:" followed by bullet list items
    # Status block ends when we hit a paragraph (non-bullet line) or another heading
    # Support both old format (Status:) and new format (### Status)
    status_pattern = r'(?i)(?:^|\n)(?:###\s+Status|Status:)\s*\n((?:- .+?\n)+)(?=\n[^-]|\n##|\Z)'
    match = re.search(status_pattern, chapter_text, re.MULTILINE)
    
    if match:
        # Always return in new format (### Status)
        status_block = "### Status\n" + match.group(1)
        return status_block.strip()
    
    return None


def format_status_block(status_items: List[dict]) -> str:
    """
    Format status items as a status block (bullet list format).
    
    Args:
        status_items: List of dicts with keys 'item' and 'status'
        Example: [{"item": "Clarissa", "status": "Went home in chapter 2"}]
        
    Returns:
        Formatted status block text (with "### Status" header)
    """
    if not status_items:
        return ""
    
    lines = ["### Status"]
    for item in status_items:
        item_name = item.get("item", "").strip()
        item_status = item.get("status", "").strip()
        if item_name and item_status:
            lines.append(f"- {item_name}: {item_status}")
    
    if len(lines) == 1:  # Only "### Status" header, no items
        return ""
    
    return "\n".join(lines) + "\n"


def extract_pacing_block(chapter_text: str) -> Optional[Dict[str, str]]:
    """
    Extract pacing transition block from chapter text.
    
    Pacing blocks appear after Status (if present), before Summary.
    Format:
    ### Pacing
    FROM: [Emotional/tonal state from previous chapter's ending]
    TO: [Target emotional/tonal state for current chapter]
    TECHNIQUE: [Specific transition approach - how to achieve the shift]
    
    Also supports free-form pacing text without explicit FROM:/TO:/TECHNIQUE: labels.
    
    Args:
        chapter_text: Full chapter text including heading
        
    Returns:
        Dictionary with keys: 'from_state', 'to_state', 'technique', 'full_text'
        or None if no pacing block found.
        If structured format (FROM:/TO:/TECHNIQUE:) is found, parses into dict.
        If free-form text is found, stores in 'full_text' and sets other keys to None.
    """
    if not chapter_text:
        return None
    
    # Pattern: Match "### Pacing" header followed by content
    # Pacing block ends when we hit another heading (### or ##)
    # Support both structured (FROM:/TO:/TECHNIQUE:) and free-form text
    pacing_pattern = r'(?i)(?:^|\n)###\s+Pacing\s*\n((?:(?!\n###|\n##).)*?)(?=\n###|\n##|\Z)'
    match = re.search(pacing_pattern, chapter_text, re.MULTILINE | re.DOTALL)
    
    if not match:
        return None
    
    pacing_content = match.group(1).strip()
    if not pacing_content:
        return None
    
    # Try to parse structured format (FROM:/TO:/TECHNIQUE:)
    from_match = re.search(r'(?i)^FROM:\s*(.+?)(?=\nTO:|\nTECHNIQUE:|\Z)', pacing_content, re.MULTILINE | re.DOTALL)
    to_match = re.search(r'(?i)^TO:\s*(.+?)(?=\nFROM:|\nTECHNIQUE:|\Z)', pacing_content, re.MULTILINE | re.DOTALL)
    technique_match = re.search(r'(?i)^TECHNIQUE:\s*(.+?)(?=\nFROM:|\nTO:|\Z)', pacing_content, re.MULTILINE | re.DOTALL)
    
    if from_match or to_match or technique_match:
        # Structured format found
        return {
            "from_state": from_match.group(1).strip() if from_match else None,
            "to_state": to_match.group(1).strip() if to_match else None,
            "technique": technique_match.group(1).strip() if technique_match else None,
            "full_text": pacing_content
        }
    else:
        # Free-form text - store as full_text
        return {
            "from_state": None,
            "to_state": None,
            "technique": None,
            "full_text": pacing_content
        }

