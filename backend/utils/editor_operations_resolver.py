"""
Roosevelt's Progressive Search Resolver for Precise Editor Operations

Implements 4-strategy progressive search inspired by WPF desktop app:
1. Exact Match (fastest, confidence 1.0)
2. Normalized Whitespace Match (confidence 0.9)
3. Sentence Boundary Match (confidence 0.8)
4. Key Phrase Anchoring (confidence 0.7)

Returns absolute indices within full_text and confidence score for each operation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, NamedTuple
import re
import logging

logger = logging.getLogger(__name__)


class SearchResult(NamedTuple):
    """Result of text search with confidence scoring."""
    start: int
    end: int
    confidence: float
    strategy: str


def _heading_bounds(full_text: str, heading_hint: Optional[Dict[str, str]]) -> Tuple[int, int]:
    """Find the bounds of a section based on heading hint."""
    if not heading_hint:
        return 0, len(full_text)
    try:
        title = str(heading_hint.get("title") or "").strip()
        level = int(heading_hint.get("level") or 2)
        if not title:
            return 0, len(full_text)
        pat = re.compile(rf"^{'#'*level}\s+{re.escape(title)}\s*$", re.IGNORECASE | re.MULTILINE)
        m = pat.search(full_text)
        if not m:
            return 0, len(full_text)
        start = m.start()
        next_pat = re.compile(rf"^#{{1,{level}}}\s+.+$", re.MULTILINE)
        m2 = next_pat.search(full_text[m.end():])
        end = m.end() + (m2.start() if m2 else len(full_text) - m.end())
        return start, end
    except Exception:
        return 0, len(full_text)


def _selection_bounds(full_text: str, selection: Optional[Dict[str, int]]) -> Tuple[int, int]:
    """Extract selection bounds from selection dict."""
    if not selection:
        return -1, -1
    try:
        s = max(0, min(len(full_text), int(selection.get("start", -1))))
        e = max(0, min(len(full_text), int(selection.get("end", -1))))
        # Valid selection must have s < e (non-zero length) and both >= 0
        if 0 <= s < e <= len(full_text):
            return s, e
        return -1, -1
    except Exception:
        return -1, -1


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace for flexible matching."""
    try:
        # Collapse all whitespace to single spaces
        normalized = re.sub(r'\s+', ' ', text.strip())
        return normalized
    except Exception:
        return text


def _extract_key_phrases(text: str) -> Tuple[str, str]:
    """Extract first and last 3 words as key phrases."""
    try:
        words = text.split()
        if len(words) < 3:
            return text, text
        first_phrase = " ".join(words[:3])
        last_phrase = " ".join(words[-3:])
        return first_phrase, last_phrase
    except Exception:
        return text, text


# ============================================
# Strategy 1: Exact Match (Confidence 1.0)
# ============================================
def _try_exact_match(hay: str, needle: str, window_start: int, occurrence_index: int = 0) -> Optional[SearchResult]:
    """Try exact string match with occurrence support."""
    if not needle:
        return None
    
    try:
        count = 0
        search_from = 0
        while True:
            pos = hay.find(needle, search_from)
            if pos == -1:
                break
            if count == occurrence_index:
                logger.info(f"✅ EXACT MATCH found at occurrence {occurrence_index}")
                return SearchResult(
                    start=window_start + pos,
                    end=window_start + pos + len(needle),
                    confidence=1.0,
                    strategy="exact_match"
                )
            count += 1
            search_from = pos + 1
        return None
    except Exception as e:
        logger.warning(f"Exact match failed: {e}")
        return None


# ============================================
# Strategy 2: Normalized Whitespace (Confidence 0.9)
# ============================================
def _try_normalized_whitespace(hay: str, needle: str, window_start: int) -> Optional[SearchResult]:
    """Try matching with normalized whitespace."""
    if not needle:
        return None
    
    try:
        normalized_needle = _normalize_whitespace(needle)
        normalized_hay = _normalize_whitespace(hay)
        
        if not normalized_needle:
            return None
        
        pos = normalized_hay.find(normalized_needle)
        if pos == -1:
            return None
        
        # Map back to original text position (approximate)
        # Count words before match point to estimate original position
        words_before = len(normalized_hay[:pos].split())
        hay_words = hay.split()
        
        # Estimate original position by reconstructing up to that word count
        estimated_pos = 0
        word_count = 0
        for i, word in enumerate(hay_words):
            if word_count >= words_before:
                break
            estimated_pos = hay.find(word, estimated_pos) + len(word)
            word_count += 1
        
        # Validate by checking if normalized forms match
        estimated_end = min(estimated_pos + len(needle) + 50, len(hay))
        candidate = hay[max(0, estimated_pos - 10):estimated_end]
        
        if normalized_needle in _normalize_whitespace(candidate):
            logger.info("✅ NORMALIZED WHITESPACE match found")
            return SearchResult(
                start=window_start + max(0, estimated_pos - 10),
                end=window_start + min(estimated_pos + len(needle), len(hay)),
                confidence=0.9,
                strategy="normalized_whitespace"
            )
        
        return None
    except Exception as e:
        logger.warning(f"Normalized whitespace match failed: {e}")
        return None


# ============================================
# Strategy 3: Sentence Boundary Match (Confidence 0.8)
# ============================================
def _try_sentence_boundary(hay: str, needle: str, window_start: int) -> Optional[SearchResult]:
    """Try matching first sentence and extending to expected length."""
    if not needle or len(needle) < 20:
        return None
    
    try:
        # Split needle into sentences (rough heuristic)
        first_sentence_end = max(
            needle.find('. '),
            needle.find('! '),
            needle.find('? ')
        )
        
        if first_sentence_end == -1:
            return None
        
        first_sentence = needle[:first_sentence_end + 1].strip()
        
        if not first_sentence or len(first_sentence) < 10:
            return None
        
        # Find first sentence in hay
        pos = hay.find(first_sentence)
        if pos == -1:
            return None
        
        # Extend match to expected length
        expected_end = pos + len(needle)
        actual_end = min(expected_end + 20, len(hay))  # Allow some flex
        
        # Validate last few words are present
        last_words = " ".join(needle.split()[-3:])
        candidate = hay[pos:actual_end]
        
        if last_words in candidate:
            logger.info("✅ SENTENCE BOUNDARY match found")
            return SearchResult(
                start=window_start + pos,
                end=window_start + min(pos + len(needle), actual_end),
                confidence=0.8,
                strategy="sentence_boundary"
            )
        
        return None
    except Exception as e:
        logger.warning(f"Sentence boundary match failed: {e}")
        return None


# ============================================
# Strategy 4: Key Phrase Anchoring (Confidence 0.7)
# ============================================
def _try_key_phrase_anchoring(hay: str, needle: str, window_start: int) -> Optional[SearchResult]:
    """Try matching using first and last 3 words as anchors."""
    if not needle or len(needle.split()) < 6:
        return None
    
    try:
        first_phrase, last_phrase = _extract_key_phrases(needle)
        
        if not first_phrase or not last_phrase:
            return None
        
        # Find first phrase
        start_pos = hay.find(first_phrase)
        if start_pos == -1:
            return None
        
        # Look for last phrase within reasonable range
        expected_range = len(needle) + 100  # Allow some flex
        search_end = min(start_pos + expected_range, len(hay))
        
        last_pos = hay.find(last_phrase, start_pos, search_end)
        if last_pos == -1:
            return None
        
        # Match found - span from first phrase to end of last phrase
        end_pos = last_pos + len(last_phrase)
        
        logger.info(f"✅ KEY PHRASE ANCHORING match found (first: '{first_phrase[:20]}...', last: '...{last_phrase[-20:]}')")
        return SearchResult(
            start=window_start + start_pos,
            end=window_start + end_pos,
            confidence=0.7,
            strategy="key_phrase_anchoring"
        )
    except Exception as e:
        logger.warning(f"Key phrase anchoring failed: {e}")
        return None


# ============================================
# Validation Checks
# ============================================
def _validate_match(matched_text: str, original_text: str) -> Tuple[bool, str]:
    """Validate that matched text is reasonable."""
    try:
        # Check length sanity
        if len(matched_text) < len(original_text) * 0.5:
            return False, "Matched text too short (< 50% of original)"
        
        # Check key phrase presence
        original_words = original_text.split()
        if len(original_words) >= 6:
            first_phrase = " ".join(original_words[:3])
            last_phrase = " ".join(original_words[-3:])
            
            if first_phrase not in matched_text:
                return False, f"Matched text missing first phrase: '{first_phrase}'"
            
            if last_phrase not in matched_text:
                return False, f"Matched text missing last phrase: '{last_phrase}'"
        
        return True, "Valid"
    except Exception as e:
        return False, f"Validation error: {e}"


# ============================================
# Main Resolver with Progressive Search
# ============================================
def resolve_operation(
    full_text: str,
    op: Dict,
    *,
    selection: Optional[Dict[str, int]] = None,
    heading_hint: Optional[Dict[str, str]] = None,
    frontmatter_end: int = 0,
    require_anchors: bool = False,
) -> Tuple[int, int, str, float]:
    """Resolve absolute (start, end), sanitized text, and confidence for an op.

    Returns: (start, end, insert_text, confidence)
    
    op fields honored if present: 
    - action: "insert"|"revise"|"delete" (or legacy op_type)
    - original_text: exact text to find/replace
    - anchor_text: exact text to insert after (for inserts)
    - left_context: text before insertion point
    - right_context: text after modification point
    - occurrence_index: which occurrence to match (0-based)
    - text: replacement/insertion text
    """
    # 1) Determine search window
    sel_bounds = _selection_bounds(full_text, selection)
    if sel_bounds != (-1, -1):
        ws, we = sel_bounds
        logger.info(f"📍 Using selection window: [{ws}:{we}]")
    else:
        ws, we = _heading_bounds(full_text, heading_hint)
        logger.info(f"📍 Using heading/default window: [{ws}:{we}]")
    
    logger.info(f"📍 Frontmatter end: {frontmatter_end}, doc length: {len(full_text)}")
    ws = max(ws, frontmatter_end)
    we = max(we, ws)
    
    window = full_text[ws:we]
    logger.info(f"📍 Final search window: [{ws}:{we}] = {len(window)} chars")
    
    # 2) Normalize action
    action_raw = str(op.get("action") or "").strip().lower()
    action = action_raw or str(op.get("op_type") or "").strip().lower()
    if action in ("revise", "replace", "replace_range"):
        action = "revise"
    elif action in ("delete", "delete_range"):
        action = "delete"
    elif action in ("insert", "insert_after", "insert_after_heading"):
        action = "insert"
    else:
        action = action or "revise"
    
    # 3) Extract anchors
    original_text = str(op.get("original_text") or op.get("original") or "")
    anchor_text = str(op.get("anchor_text") or "")  # New field for explicit "insert after" anchoring
    left_ctx = str(op.get("left_context") or "")
    right_ctx = str(op.get("right_context") or "")
    occurrence_index = int(op.get("occurrence_index") or 0)
    
    # **DEBUG: Log what the LLM provided**
    logger.info(f"🔍 RESOLVER DEBUG: action={action}, require_anchors={require_anchors}")
    logger.info(f"🔍 RESOLVER DEBUG: original_text={'[PRESENT: ' + original_text[:60] + '...]' if original_text else '[NONE]'}")
    logger.info(f"🔍 RESOLVER DEBUG: anchor_text={'[PRESENT: ' + anchor_text[:60] + '...]' if anchor_text else '[NONE]'}")
    logger.info(f"🔍 RESOLVER DEBUG: left_context={'[PRESENT]' if left_ctx else '[NONE]'}")
    logger.info(f"🔍 RESOLVER DEBUG: right_context={'[PRESENT]' if right_ctx else '[NONE]'}")
    
    # 4) Enforce anchor requirements if strict mode
    if require_anchors:
        if action in ("revise", "delete") and not original_text:
            raise ValueError(
                f"Anchor required: original_text is required for {action}. "
                "Provide EXACT, VERBATIM text from the file (minimum 10-20 words, complete sentences)."
            )
        if action == "insert" and not (original_text or anchor_text or left_ctx):
            raise ValueError(
                "Anchor required: insert must include anchor_text (exact line to insert after), "
                "original_text (verbatim text to insert after), or left_context (text before insertion point)."
            )
    
    # 5) Progressive search for original_text or anchor_text
    search_result = None
    search_target = anchor_text or original_text
    
    if search_target:
        logger.info(f"🔍 PROGRESSIVE SEARCH: Searching for {len(search_target)} chars in {len(window)} char window")
        
        # Strategy 1: Exact Match
        search_result = _try_exact_match(window, search_target, ws, occurrence_index)
        
        # Strategy 2: Normalized Whitespace
        if not search_result:
            search_result = _try_normalized_whitespace(window, search_target, ws)
        
        # Strategy 3: Sentence Boundary
        if not search_result:
            search_result = _try_sentence_boundary(window, search_target, ws)
        
        # Strategy 4: Key Phrase Anchoring
        if not search_result:
            search_result = _try_key_phrase_anchoring(window, search_target, ws)
        
        # Validate match if found (skip validation for exact matches, they're inherently valid)
        if search_result and search_result.confidence >= 0.7:
            matched_text = full_text[search_result.start:search_result.end]
            
            # Only validate non-exact matches (exact matches are already perfect)
            if search_result.confidence < 1.0:
                is_valid, validation_msg = _validate_match(matched_text, search_target)
                
                if not is_valid:
                    logger.warning(f"⚠️ VALIDATION FAILED: {validation_msg}")
                    if require_anchors:
                        raise ValueError(f"Match validation failed: {validation_msg}")
                    search_result = None
                else:
                    logger.info(f"✅ VALIDATED: Match confidence {search_result.confidence} using {search_result.strategy}")
            else:
                logger.info(f"✅ EXACT MATCH: Skipping validation (confidence {search_result.confidence})")
    
    # 6) Context-based search (left/right_context) if no result yet
    start = end = -1
    confidence = 0.0
    
    if search_result:
        start = search_result.start
        end = search_result.end
        confidence = search_result.confidence
        
        # For inserts with anchor_text, insert AFTER the found text
        if action == "insert" and anchor_text:
            start = end
    else:
        # Try context-based matching
        if left_ctx and right_ctx:
            try:
                # Match with bounded gap between contexts
                pat = re.compile(re.escape(left_ctx) + r"([\s\S]{0,400}?)" + re.escape(right_ctx))
                m = pat.search(window)
                if m:
                    start = ws + m.start(1)
                    end = ws + m.end(1)
                    confidence = 0.6
                    logger.info("✅ CONTEXT MATCH (left+right)")
            except Exception as e:
                logger.warning(f"Context match failed: {e}")
        
        elif left_ctx and not right_ctx:
            # Insert after left_ctx
            try:
                pos = window.rfind(left_ctx)
                if pos != -1:
                    start = ws + pos + len(left_ctx)
                    end = start
                    confidence = 0.6
                    logger.info("✅ CONTEXT MATCH (left only)")
            except Exception:
                pass
    
    # 7) Fallback for pure inserts - find reasonable insertion point
    if start == -1:
        if action == "insert":
            # Try to find last paragraph boundary or heading in window
            lb = max(window.rfind("\n\n"), window.rfind("\n#"))
            if lb != -1:
                start = end = ws + lb + (2 if window[lb:lb+2] == "\n\n" else 1)
                confidence = 0.3
                logger.info("⚠️ FALLBACK: Inserting at last blank line in window")
            else:
                start = end = we
                confidence = 0.1
                logger.warning("⚠️ FALLBACK: Inserting at end of window (weak anchor)")
        else:
            # Revise/delete with no anchor found - fail in strict mode
            if require_anchors:
                raise ValueError(
                    f"No match found for {action} operation. Provide more specific anchors:\n"
                    f"- original_text: EXACT, VERBATIM text from file (minimum 10-20 words)\n"
                    f"- Include complete sentences with natural boundaries"
                )
            else:
                start = end = ws
                confidence = 0.0
                logger.error(f"❌ NO MATCH: {action} operation failed to find target")
    
    # 8) Prepare insertion text with spacing guards
    insert_text = str(op.get("text") or "")
    
    if start == end and insert_text:  # Pure insertion
        # Add appropriate spacing
        left_tail = full_text[max(0, start-2):start]
        if left_tail.endswith("\n\n"):
            prefix = ""
        elif left_tail.endswith("\n"):
            prefix = "\n"
        else:
            prefix = "\n\n"
        
        trailing = ""
        if not insert_text.endswith("\n"):
            trailing = "\n"
        
        insert_text = f"{prefix}{insert_text}{trailing}"
    
    # 9) Guard frontmatter
    start = max(start, frontmatter_end)
    end = max(end, start)
    
    logger.info(f"🎯 RESOLVED: [{start}:{end}] confidence={confidence:.2f}")
    
    return start, end, insert_text, confidence


