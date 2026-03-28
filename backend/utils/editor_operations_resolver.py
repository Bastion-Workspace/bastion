"""
Unified Editor Operation Resolver (backend).

Single source of truth for resolving semantic editor operations to (start, end)
positions. Used at display time and apply time. Supports:

- replace_range / delete_range: search_text (or original_text) with progressive
  matching: exact, normalized whitespace, line-anchored fuzzy, sentence anchoring.
- insert_after_heading / insert_after: anchor_text with fuzzy heading match and
  section-boundary detection.

All insertion spacing (newlines before/after) is normalized here via
normalize_insertion_spacing(); no other layer should add newlines.
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Optional, Tuple, Any, NamedTuple

logger = logging.getLogger(__name__)


class SearchResult(NamedTuple):
    """Result of text search with confidence scoring."""
    start: int
    end: int
    confidence: float
    strategy: str


def get_frontmatter_end(content: str) -> int:
    """Return character index where frontmatter ends (0 if no frontmatter)."""
    m = re.match(r"^(---\s*\r?\n[\s\S]*?\r?\n---\s*\r?\n)", content)
    return m.end() if m else 0


def normalize_insertion_spacing(content: str, insert_pos: int, text: str) -> str:
    """
    Normalize spacing around an insertion so headings have a blank line before
    them and paragraphs are properly separated. Single place for all newline
    logic; no other layer should add newlines to insert text.

    Args:
        content: Full document content
        insert_pos: Position where text will be inserted
        text: Raw insertion text (no leading/trailing newlines added yet)

    Returns:
        Text with appropriate leading and trailing spacing.
    """
    if not text:
        return ""
    prefix = ""
    trailing = ""
    # Leading: ensure we don't double newlines and headings get blank line before
    if insert_pos > 0:
        left_tail = content[max(0, insert_pos - 2) : insert_pos]
        if left_tail.endswith("\n\n"):
            prefix = ""
        elif left_tail.endswith("\n"):
            prefix = "" if text.startswith("\n") else "\n"
        else:
            # No newline before insert pos: add one or two
            if text.strip().startswith("#"):
                prefix = "\n\n"  # Headings get blank line before
            else:
                prefix = "\n" if not text.startswith("\n") else ""
    elif text.startswith("\n"):
        prefix = ""
    else:
        prefix = "\n"
    # Trailing: ensure we don't leave trailing content without newline
    if not text.endswith("\n"):
        trailing = "\n"
    return f"{prefix}{text}{trailing}"


def _normalize_whitespace(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip()) if t else ""


def _extract_key_phrases(text: str) -> Tuple[str, str]:
    words = text.split()
    if len(words) < 3:
        return text, text
    return " ".join(words[:3]), " ".join(words[-3:])


# ---- Replace/Delete: search_text matching (4 strategies) ----

def _try_exact_match(
    hay: str, needle: str, window_start: int, occurrence_index: int = 0
) -> Optional[SearchResult]:
    if not needle:
        return None
    count = 0
    search_from = 0
    while True:
        pos = hay.find(needle, search_from)
        if pos == -1:
            break
        if count == occurrence_index:
            return SearchResult(
                window_start + pos,
                window_start + pos + len(needle),
                1.0,
                "exact_match",
            )
        count += 1
        search_from = pos + 1
    return None


def _try_normalized_whitespace(hay: str, needle: str, window_start: int) -> Optional[SearchResult]:
    if not needle:
        return None
    n_needle = _normalize_whitespace(needle)
    n_hay = _normalize_whitespace(hay)
    if not n_needle or n_needle not in n_hay:
        return None
    pos = n_hay.find(n_needle)
    words_before = len(n_hay[:pos].split())
    hay_words = hay.split()
    estimated_pos = 0
    word_count = 0
    for w in hay_words:
        if word_count >= words_before:
            break
        estimated_pos = hay.find(w, estimated_pos) + len(w)
        word_count += 1
    estimated_end = min(estimated_pos + len(needle) + 50, len(hay))
    candidate = hay[max(0, estimated_pos - 10) : estimated_end]
    if n_needle not in _normalize_whitespace(candidate):
        return None
    return SearchResult(
        window_start + max(0, estimated_pos - 10),
        window_start + min(estimated_pos + len(needle), len(hay)),
        0.95,
        "normalized_whitespace",
    )


def _try_line_anchored_fuzzy(hay: str, needle: str, window_start: int) -> Optional[SearchResult]:
    """Match by finding each line of needle with normalized matching."""
    if not needle or ("\n" not in needle and len(needle) < 20):
        return None
    lines = [ln.strip() for ln in needle.strip().split("\n") if ln.strip()]
    if not lines:
        return None
    search_from = 0
    start_pos = None
    for line in lines:
        n_line = _normalize_whitespace(line)
        if not n_line:
            continue
        found = hay.find(line, search_from)
        if found == -1:
            found = hay.find(n_line, search_from)
        if found == -1:
            segment = hay[search_from:]
            for j in range(max(0, len(segment) - len(n_line) + 1)):
                candidate = segment[j : j + len(line) + 50]
                if n_line in _normalize_whitespace(candidate):
                    found = search_from + j
                    break
        if found == -1:
            return None
        if start_pos is None:
            start_pos = found
        search_from = found + len(line)
    if start_pos is None:
        return None
    end_pos = min(search_from, len(hay))
    return SearchResult(
        window_start + start_pos,
        window_start + end_pos,
        0.9,
        "line_anchored_fuzzy",
    )


def _try_sentence_anchoring(hay: str, needle: str, window_start: int) -> Optional[SearchResult]:
    """Match first and last sentences of the search block."""
    if not needle or len(needle) < 20:
        return None
    first_end = max(needle.find(". "), needle.find("! "), needle.find("? "))
    if first_end == -1:
        return None
    first_sentence = needle[: first_end + 1].strip()
    if len(first_sentence) < 10:
        return None
    last_words = " ".join(needle.split()[-3:])
    pos = hay.find(first_sentence)
    if pos == -1:
        return None
    expected_end = pos + len(needle)
    actual_end = min(expected_end + 50, len(hay))
    candidate = hay[pos:actual_end]
    if last_words not in candidate:
        return None
    return SearchResult(
        window_start + pos,
        window_start + min(pos + len(needle), actual_end),
        0.85,
        "sentence_anchoring",
    )


def _resolve_replace_delete(
    content: str,
    op: Dict[str, Any],
    frontmatter_end: int,
) -> Tuple[int, int, str, float]:
    """Resolve replace_range or delete_range. Uses search_text or original_text."""
    op_type = op.get("op_type", "replace_range")
    search_text = op.get("search_text") or op.get("original_text") or op.get("original") or ""
    text = op.get("text", "") if op_type == "replace_range" else ""
    occurrence_index = int(op.get("occurrence_index") or 0)
    if not search_text or not search_text.strip():
        return -1, -1, text, 0.0
    window = content[frontmatter_end:]
    ws = frontmatter_end
    result = (
        _try_exact_match(window, search_text, ws, occurrence_index)
        or _try_normalized_whitespace(window, search_text, ws)
        or _try_line_anchored_fuzzy(window, search_text, ws)
        or _try_sentence_anchoring(window, search_text, ws)
    )
    if not result:
        return -1, -1, text, 0.0
    start = max(result.start, frontmatter_end)
    end = min(result.end, len(content))
    if content[start:end] != search_text and _normalize_whitespace(content[start:end]) != _normalize_whitespace(search_text):
        logger.warning("Resolved replace/delete span does not match search_text closely")
    return start, end, text, result.confidence


# ---- Insert: anchor text fuzzy match and section boundaries ----

def _find_anchor_text_fuzzy(
    content: str, anchor_text: str, search_start: int = 0
) -> Optional[Tuple[int, float]]:
    """Progressive fuzzy search for anchor (exact, case-insensitive, normalized, regex for headings)."""
    if not anchor_text:
        return None
    search_content = content[search_start:]
    pos = search_content.rfind(anchor_text)
    if pos != -1:
        return (search_start + pos, 1.0)
    anchor_lower = anchor_text.lower()
    pos_lower = search_content.lower().rfind(anchor_lower)
    if pos_lower != -1:
        actual = search_start + pos_lower
        if actual == 0 or content[actual - 1] == "\n":
            return (actual, 0.9)
    normalized_anchor = " ".join(anchor_text.split())
    if normalized_anchor != anchor_text:
        lines = search_content.split("\n")
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            if normalized_anchor in " ".join(line.split()):
                line_start = search_start + sum(len(l) + 1 for l in lines[:i])
                words = normalized_anchor.split()
                if words:
                    wp = line.lower().find(words[0].lower())
                    if wp >= 0:
                        return (line_start + wp, 0.85)
                return (line_start, 0.85)
    heading_match = re.match(r"^(#{1,6})\s*(.+)$", anchor_text.strip())
    if heading_match:
        hash_count = len(heading_match.group(1))
        heading_text = heading_match.group(2).strip()
        pattern = rf'^{"#" * hash_count}\s+{re.escape(heading_text)}'
        matches = list(re.finditer(pattern, search_content, re.MULTILINE | re.IGNORECASE))
        if matches:
            return (search_start + matches[-1].start(), 0.8)
    return None


def _insert_end_pos_for_anchor(
    content: str,
    anchor_pos: int,
    anchor_text: str,
    op_type: str,
    text: str,
) -> int:
    """Compute insertion end position (after anchor/section) for insert_after_heading or insert_after."""
    anchor_end = anchor_pos + len(anchor_text)
    if anchor_text.strip().startswith("## Chapter") or anchor_text.strip().startswith("# Chapter"):
        text_preview = (text or "").strip()[:50]
        if text_preview.startswith("###"):
            line_end = content.find("\n", anchor_end)
            return (line_end + 1) if line_end != -1 else anchor_end
        next_chapter = re.compile(r"\n##\s+Chapter\s+\d+", re.MULTILINE)
        m = next_chapter.search(content, anchor_end)
        return m.start() if m else len(content)
    if op_type == "insert_after":
        para = content.find("\n\n", anchor_end)
        head = content.find("\n#", anchor_end)
        if head != -1 and (para == -1 or head < para):
            return head
        if para != -1:
            return para
        return len(content)
    heading_match = re.match(r"^(#{1,6})\s+", anchor_text.strip())
    if heading_match:
        level = len(heading_match.group(1))
        pattern_str = r"\n(#{1," + str(level) + r"})\s+"
        m = re.search(pattern_str, content[anchor_end:])
        return anchor_end + m.start() if m else len(content)
    line_end = content.find("\n", anchor_pos)
    return (line_end + 1) if line_end != -1 else len(content)


def _resolve_insert(
    content: str,
    op: Dict[str, Any],
    frontmatter_end: int,
) -> Tuple[int, int, str, float]:
    """Resolve insert_after_heading or insert_after."""
    op_type = op.get("op_type", "insert_after_heading")
    anchor_text = (op.get("anchor_text") or "").strip()
    text = op.get("text", "")
    body = content[frontmatter_end:].strip()
    if not body:
        insert_pos = frontmatter_end
        resolved_text = normalize_insertion_spacing(content, insert_pos, text)
        return insert_pos, insert_pos, resolved_text, 0.8
    if not anchor_text:
        insert_pos = frontmatter_end
        resolved_text = normalize_insertion_spacing(content, insert_pos, text)
        return insert_pos, insert_pos, resolved_text, 0.5
    result = _find_anchor_text_fuzzy(content, anchor_text, frontmatter_end)
    if not result:
        result = _find_anchor_text_fuzzy(content, anchor_text, 0)
    if not result:
        return -1, -1, text, 0.0
    anchor_pos, confidence = result
    end_pos = _insert_end_pos_for_anchor(content, anchor_pos, anchor_text, op_type, text)
    end_pos = max(end_pos, frontmatter_end)
    resolved_text = normalize_insertion_spacing(content, end_pos, text)
    return end_pos, end_pos, resolved_text, max(confidence, 0.8)


# ---- Single-operation API (backward compatible) ----

def resolve_operation(
    full_text: str,
    op: Dict[str, Any],
    *,
    selection: Optional[Dict[str, int]] = None,
    heading_hint: Optional[Dict[str, str]] = None,
    frontmatter_end: Optional[int] = None,
    require_anchors: bool = False,
) -> Tuple[int, int, str, float]:
    """
    Resolve a single operation to (start, end, text, confidence).
    Kept for backward compatibility. Prefer resolve_operations() for batch.
    """
    if frontmatter_end is None:
        frontmatter_end = get_frontmatter_end(full_text)
    op_type = (op.get("op_type") or op.get("action") or "replace_range").strip().lower()
    if op_type in ("revise", "replace"):
        op_type = "replace_range"
    elif op_type == "delete":
        op_type = "delete_range"
    elif op_type in ("insert", "insert_after"):
        op_type = "insert_after" if op_type == "insert_after" else "insert_after_heading"
    op = dict(op)
    op["op_type"] = op_type
    if op_type in ("replace_range", "delete_range"):
        return _resolve_replace_delete(full_text, op, frontmatter_end)
    return _resolve_insert(full_text, op, frontmatter_end)


# ---- Batch API ----

def resolve_operations(
    content: str,
    operations: List[Dict[str, Any]],
    frontmatter_end: Optional[int] = None,
    cursor_offset: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Resolve a batch of semantic operations to positioned operations.
    Each op gets start, end, text (with spacing), and confidence added.
    Operations that fail resolution get confidence=0 and resolved_start=-1, resolved_end=-1.
    """
    if frontmatter_end is None:
        frontmatter_end = get_frontmatter_end(content)
    result = []
    for op in operations:
        op = dict(op)
        op_type = (op.get("op_type") or "replace_range").strip().lower()
        if op_type in ("revise", "replace"):
            op_type = "replace_range"
        elif op_type == "delete":
            op_type = "delete_range"
        elif op_type in ("insert",):
            op_type = "insert_after"
        op["op_type"] = op_type
        try:
            if op_type in ("replace_range", "delete_range"):
                start, end, text, confidence = _resolve_replace_delete(content, op, frontmatter_end)
            else:
                start, end, text, confidence = _resolve_insert(content, op, frontmatter_end)
        except Exception as e:
            logger.warning("Resolve op failed: %s", e)
            start, end, text, confidence = -1, -1, op.get("text", ""), 0.0
        op["start"] = start
        op["end"] = end
        op["text"] = text
        op["confidence"] = confidence
        op["resolved_start"] = start
        op["resolved_end"] = end
        result.append(op)
    return result
