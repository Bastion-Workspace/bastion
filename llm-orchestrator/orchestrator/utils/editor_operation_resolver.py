"""
Centralized Editor Operation Resolver

Provides a unified resolver for editor operations across all agents.
Eliminates code duplication and ensures consistent behavior.

Supports all operation types:
- replace_range: Replace existing text (requires original_text)
- insert_after: Insert after text (requires anchor_text)
- insert_after_heading: Insert after heading (requires anchor_text)
- delete_range: Delete text (requires original_text)
"""

import re
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def _fuzzy_match_in_window(
    content: str,
    search_text: str,
    expected_start: int,
    window_size: int = 1000,
    min_similarity: float = 0.7
) -> Optional[Tuple[int, int, float]]:
    """
    Fuzzy search for text within a window around the expected position.
    
    Uses a simple character-based similarity metric to find the best match
    when exact matching fails due to document drift.
    
    Args:
        content: Full document content
        search_text: Text to search for
        expected_start: Expected start position (from operation)
        window_size: Search window size (default 1000 chars on each side)
        min_similarity: Minimum similarity threshold (0.0-1.0)
    
    Returns:
        Tuple of (start_pos, end_pos, similarity_score) or None if no good match found
    """
    if not search_text or len(search_text) < 10:
        return None
    
    # Define search window around expected position
    window_start = max(0, expected_start - window_size)
    window_end = min(len(content), expected_start + len(search_text) + window_size)
    search_window = content[window_start:window_end]
    
    if len(search_window) < len(search_text):
        return None
    
    # Try sliding window approach: check each possible position
    best_match = None
    best_similarity = 0.0
    
    # Search for best match within window
    for i in range(len(search_window) - len(search_text) + 1):
        candidate = search_window[i:i + len(search_text)]
        
        # Calculate similarity: count matching characters
        matches = sum(1 for a, b in zip(search_text, candidate) if a == b)
        similarity = matches / len(search_text) if search_text else 0.0
        
        # Also check if key phrases match (first and last 20 chars)
        if len(search_text) > 40:
            first_20_match = search_text[:20] == candidate[:20]
            last_20_match = search_text[-20:] == candidate[-20:]
            if first_20_match:
                similarity += 0.1
            if last_20_match:
                similarity += 0.1
        
        if similarity > best_similarity:
            best_similarity = similarity
            actual_start = window_start + i
            actual_end = actual_start + len(search_text)
            best_match = (actual_start, actual_end, similarity)
    
    # Return best match if it meets minimum similarity threshold
    if best_match and best_similarity >= min_similarity:
        logger.info(
            f"   🔍 Fuzzy match found: similarity={best_similarity:.2f}, "
            f"expected={expected_start}, actual={best_match[0]}"
        )
        return best_match
    
    return None


def _find_anchor_text_fuzzy(
    content: str,
    anchor_text: str,
    search_start: int = 0
) -> Optional[Tuple[int, float]]:
    """
    Progressive fuzzy search for anchor text in document.
    
    Tries multiple strategies in order:
    1. Exact match (case-sensitive, exact whitespace)
    2. Case-insensitive match (for headings)
    3. Normalized whitespace match (collapses whitespace)
    4. Regex pattern match (for markdown headings with flexible spacing)
    
    Args:
        content: Full document content
        anchor_text: Anchor text to search for
        search_start: Position to start searching from
    
    Returns:
        Tuple of (position, confidence) or None if no match found
        Confidence: 1.0 for exact, 0.9 for case-insensitive, 0.85 for normalized, 0.8 for regex
    """
    if not anchor_text:
        return None
    
    search_content = content[search_start:]
    
    # Strategy 1: Exact match (highest confidence)
    pos = search_content.rfind(anchor_text)
    if pos != -1:
        actual_pos = search_start + pos
        logger.debug(f"✅ Anchor text exact match at position {actual_pos}")
        return (actual_pos, 1.0)
    
    # Strategy 2: Case-insensitive match (useful for headings)
    anchor_lower = anchor_text.lower()
    pos_lower = search_content.lower().rfind(anchor_lower)
    if pos_lower != -1:
        # Verify it's actually at a reasonable position (start of line or after newline)
        actual_pos = search_start + pos_lower
        if actual_pos == 0 or content[actual_pos - 1] == '\n':
            logger.info(f"💡 Anchor text case-insensitive match at position {actual_pos}")
            return (actual_pos, 0.9)
    
    # Strategy 3: Normalized whitespace (collapse multiple spaces/newlines to single space)
    # Search line by line for better position accuracy
    normalized_anchor = ' '.join(anchor_text.split())
    if normalized_anchor != anchor_text:  # Only if normalization changed something
        lines = search_content.split('\n')
        # Search from end (last occurrence) to match rfind behavior
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            normalized_line = ' '.join(line.split())
            if normalized_anchor in normalized_line:
                # Found in this line - calculate line start position
                line_start = search_start + sum(len(l) + 1 for l in lines[:i])  # +1 for newline
                # Try to find the anchor in the original line using a simple approach
                # Look for the first word of the anchor
                anchor_words = normalized_anchor.split()
                if anchor_words:
                    first_word = anchor_words[0]
                    # Find first word in original line (case-insensitive for robustness)
                    word_pos = line.lower().find(first_word.lower())
                    if word_pos >= 0:
                        actual_pos = line_start + word_pos
                        logger.info(f"💡 Anchor text normalized whitespace match at position {actual_pos}")
                        return (actual_pos, 0.85)
                    else:
                        # Fallback: use line start
                        logger.info(f"💡 Anchor text normalized whitespace match (line start) at position {line_start}")
                        return (line_start, 0.85)
    
    # Strategy 4: Regex pattern match for markdown headings (flexible spacing)
    # This handles cases like "## Chapter 1" vs "##Chapter 1" or "##  Chapter 1"
    heading_match = re.match(r'^(#{1,6})\s*(.+)$', anchor_text.strip())
    if heading_match:
        hash_count = len(heading_match.group(1))
        heading_text = heading_match.group(2).strip()
        
        # Build flexible regex pattern: allows variable spacing after hashes
        # Match: ##\s+Chapter\s+1 (allows multiple spaces)
        pattern = rf'^{"#" * hash_count}\s+{re.escape(heading_text)}'
        regex_matches = list(re.finditer(pattern, search_content, re.MULTILINE | re.IGNORECASE))
        if regex_matches:
            # Use last match (rfind behavior)
            last_match = regex_matches[-1]
            actual_pos = search_start + last_match.start()
            logger.info(f"💡 Anchor text regex pattern match at position {actual_pos}")
            return (actual_pos, 0.8)
    
    # No match found
    return None


# Import frontmatter utilities from shared module
from orchestrator.utils.frontmatter_utils import strip_frontmatter_block, frontmatter_end_index


def resolve_editor_operation(
    content: str,
    op_dict: Dict[str, Any],
    selection: Optional[Dict[str, int]] = None,
    frontmatter_end: int = 0,
    cursor_offset: Optional[int] = None
) -> Tuple[int, int, str, float]:
    """
    Resolve editor operation to precise (start, end) positions.
    
    Uses progressive search strategies:
    1. Selection-based (when user has text selected)
    2. Exact original_text matching
    3. Anchor text matching (for insert operations) - searches ENTIRE document, uses last occurrence
    4. Normalized whitespace matching
    5. Substring fallback matching
    6. Context-based matching (left_context + right_context)
    
    Note: For anchor text searches, the entire document is searched (not just near cursor).
    This ensures maximum flexibility - edits can target any section regardless of cursor position.
    The last occurrence is used for consistency (new chapters at end, most recent section edits).
    
    Args:
        content: Full document content
        op_dict: Operation dictionary with op_type, text, original_text, anchor_text, etc.
        selection: Optional selection dict with 'start' and 'end' keys
        frontmatter_end: Character offset where frontmatter ends (0 if no frontmatter)
        cursor_offset: Optional cursor position (currently not used for anchor text searches)
    
    Returns:
        Tuple of (start_pos, end_pos, text, confidence)
        Returns (-1, -1, text, 0.0) if resolution fails
    """
    op_type = op_dict.get("op_type", "replace_range")
    original_text = op_dict.get("original_text")
    anchor_text = op_dict.get("anchor_text")
    left_context = op_dict.get("left_context")
    right_context = op_dict.get("right_context")
    occurrence_index = op_dict.get("occurrence_index", 0)
    text = op_dict.get("text", "")
    
    # Auto-detect frontmatter if not provided
    if frontmatter_end == 0:
        frontmatter_end = frontmatter_end_index(content)
    
    # Strategy 1: Selection-based resolution (highest priority)
    if selection and selection.get("start", -1) >= 0:
        sel_start = selection["start"]
        sel_end = selection["end"]
        # Only use selection if it's an actual selection (start != end), not just cursor position
        if op_type == "replace_range" and sel_start != sel_end:
            return sel_start, sel_end, text, 1.0
    
    # Strategy 2: Exact match with original_text (for replace_range/delete_range)
    if original_text and op_type in ("replace_range", "delete_range"):
        # CRITICAL: If cursor_offset is provided, prefer matches near cursor (within ±10K chars)
        # This ensures edits target the chapter where the user's cursor is located
        cursor_search_window = 10000  # ±10K characters around cursor
        cursor_matches = []
        all_matches = []
        
        # First, try searching near cursor if available
        if cursor_offset is not None and cursor_offset >= 0:
            search_start = max(frontmatter_end, cursor_offset - cursor_search_window)
            search_end = min(len(content), cursor_offset + cursor_search_window)
            cursor_window = content[search_start:search_end]
            
            # Search in cursor window
            search_from = 0
            while True:
                pos_in_window = cursor_window.find(original_text, search_from)
                if pos_in_window == -1:
                    break
                # Convert window-relative position to document-relative position
                pos = search_start + pos_in_window
                cursor_matches.append(pos)
                search_from = pos_in_window + 1
            
            # If we found matches near cursor, prefer those
            if cursor_matches:
                # Use occurrence_index to select which match (default to first)
                match_idx = min(occurrence_index, len(cursor_matches) - 1)
                pos = cursor_matches[match_idx]
                end_pos = pos + len(original_text)
                
                # DEBUG: Check what text is actually at this position
                actual_text_at_pos = content[pos:end_pos]
                logger.debug(f"🔍 RESOLVER DEBUG: original_text={repr(original_text[:50])}...")
                logger.debug(f"🔍 RESOLVER DEBUG: actual text at [{pos}:{end_pos}]={repr(actual_text_at_pos[:50])}...")
                logger.debug(f"🔍 RESOLVER DEBUG: 10 chars BEFORE pos={repr(content[max(0, pos-10):pos])}")
                
                # Guard frontmatter: ensure operations never occur before frontmatter end
                pos = max(pos, frontmatter_end)
                end_pos = max(end_pos, pos)
                logger.debug(f"✅ Found original_text near cursor at position {pos} (match {match_idx+1}/{len(cursor_matches)}, cursor_offset={cursor_offset})")
                return pos, end_pos, text, 1.0
        
        # If no match near cursor (or no cursor), search entire document
        count = 0
        search_from = 0
        while True:
            pos = content.find(original_text, search_from)
            if pos == -1:
                break
            all_matches.append(pos)
            if count == occurrence_index:
                end_pos = pos + len(original_text)
                # Guard frontmatter: ensure operations never occur before frontmatter end
                pos = max(pos, frontmatter_end)
                end_pos = max(end_pos, pos)
                logger.debug(f"✅ Found original_text at position {pos} (occurrence {count}, total_matches={len(all_matches)})")
                return pos, end_pos, text, 1.0
            count += 1
            search_from = pos + 1
        
        # If exact match failed, try with normalized whitespace
        if not all_matches:
            logger.warning(f"⚠️ Exact match failed for original_text (length: {len(original_text)})")
            logger.debug(f"   original_text preview: {repr(original_text[:100])}")
            
            # Normalize whitespace: collapse multiple spaces/newlines to single space
            normalized_original = ' '.join(original_text.split())
            normalized_content = ' '.join(content.split())
            
            # Try to find normalized text and map back
            pos_normalized = normalized_content.find(normalized_original)
            if pos_normalized >= 0:
                # Try to find a unique substring that's likely to match
                if len(original_text) > 20:
                    # Try to find a unique 20-character substring
                    for i in range(len(original_text) - 20):
                        substring = original_text[i:i+20]
                        # Remove leading/trailing whitespace but preserve internal
                        substring_clean = substring.strip()
                        if len(substring_clean) >= 15:
                            pos_clean = content.find(substring_clean)
                            if pos_clean >= 0:
                                # Found a substring match - use it as anchor
                                logger.info(f"   💡 Found substring match at position {pos_clean} (using substring: {repr(substring_clean[:30])})")
                                # Try to find the full text around this position
                                search_window_start = max(0, pos_clean - 50)
                                search_window_end = min(len(content), pos_clean + len(original_text) + 50)
                                window = content[search_window_start:search_window_end]
                                
                                # Try fuzzy match in this window
                                window_normalized = ' '.join(window.split())
                                original_normalized = ' '.join(original_text.split())
                                if original_normalized in window_normalized:
                                    # Found in window - calculate position
                                    window_pos = window_normalized.find(original_normalized)
                                    # Map back (rough approximation)
                                    actual_pos = search_window_start + window_pos
                                    end_pos = actual_pos + len(original_text)
                                    actual_pos = max(actual_pos, frontmatter_end)
                                    end_pos = max(end_pos, actual_pos)
                                    logger.info(f"   ✅ Found normalized match at position {actual_pos}")
                                    return actual_pos, end_pos, text, 0.85
            
            # Last resort: try searching for first 30 characters (more lenient)
            if len(original_text) > 30:
                first_30 = original_text[:30].strip()
                pos_30 = content.find(first_30)
                if pos_30 >= 0:
                    logger.info(f"   💡 Found first 30 chars at position {pos_30} - using as anchor")
                    # Use this as the start position
                    end_pos = pos_30 + len(original_text)
                    pos_30 = max(pos_30, frontmatter_end)
                    end_pos = max(end_pos, pos_30)
                    return pos_30, end_pos, text, 0.7
                
                # Try last 30 chars as well
                last_30 = original_text[-30:].strip()
                pos_last = content.find(last_30)
                if pos_last >= 0:
                    start_pos = max(0, pos_last - (len(original_text) - 30))
                    start_pos = max(start_pos, frontmatter_end)
                    logger.info(f"   💡 Found last 30 chars at position {pos_last}, estimated start: {start_pos}")
                    end_pos = pos_last + 30
                    return start_pos, end_pos, text, 0.6
            
            # Strategy 2.5: Fuzzy match fallback - search within window around expected position
            # This handles cases where document has shifted but text still exists nearby
            expected_start = op_dict.get("start", 0)
            if expected_start > 0:
                fuzzy_result = _fuzzy_match_in_window(
                    content, original_text, expected_start, window_size=1000, min_similarity=0.7
                )
                if fuzzy_result:
                    fuzzy_start, fuzzy_end, fuzzy_similarity = fuzzy_result
                    # Guard frontmatter
                    fuzzy_start = max(fuzzy_start, frontmatter_end)
                    fuzzy_end = max(fuzzy_end, fuzzy_start)
                    # Convert similarity to confidence (0.7-0.85 range for fuzzy matches)
                    confidence = 0.7 + (fuzzy_similarity - 0.7) * 0.5  # Scale 0.7-1.0 to 0.7-0.85
                    logger.info(
                        f"   ✅ Fuzzy match successful: position={fuzzy_start}, "
                        f"similarity={fuzzy_similarity:.2f}, confidence={confidence:.2f}"
                    )
                    return fuzzy_start, fuzzy_end, text, confidence
            
            logger.error(f"   ❌ Could not find original_text in content even with fuzzy search")
            logger.error(f"   original_text length: {len(original_text)}, preview: {repr(original_text[:200])}")
            logger.error(f"   Content length: {len(content)}, expected_start: {op_dict.get('start', 0)}")
            # Return failure signal
            return -1, -1, text, 0.0
    
    # Strategy 3: Anchor text for insert_after_heading or insert_after
    if anchor_text and op_type in ("insert_after_heading", "insert_after"):
        # Check if file is empty (only frontmatter) - if so, insert after frontmatter without requiring anchor
        body_only = content[frontmatter_end:].strip()
        if not body_only:
            # Empty file - insert after frontmatter without requiring anchor_text
            logger.info("Empty file detected - inserting after frontmatter without anchor")
            return frontmatter_end, frontmatter_end, text, 0.8
        
        # For outline editing, we need maximum flexibility - search the ENTIRE document
        # This allows edits anywhere: Chapter 4 edits can impact synopsis, cursor position doesn't matter
        # Always find the LAST occurrence to ensure consistent behavior (new chapters at end, edits to most recent section)
        search_start = frontmatter_end
        
        # Progressive fuzzy search for anchor text
        # Try exact match first, then fall back to fuzzy matching strategies
        anchor_result = _find_anchor_text_fuzzy(content, anchor_text, search_start)
        
        # If not found in body, try searching from start (in case frontmatter_end calculation was off)
        if anchor_result is None:
            anchor_result = _find_anchor_text_fuzzy(content, anchor_text, 0)
        
        if anchor_result:
            pos, confidence = anchor_result
            # For chapter headings, need to determine WHERE to insert based on what we're inserting
            if anchor_text.startswith("## Chapter") or anchor_text.startswith("# Chapter"):
                # Log content after anchor for debugging
                after_anchor_pos = pos + len(anchor_text)
                preview_end = min(len(content), after_anchor_pos + 300)
                content_after = content[after_anchor_pos:preview_end]
                logger.debug(f"Content after chapter anchor (first 300 chars): {repr(content_after[:300])}")
                
                # Check if we're inserting subsections (###) vs new chapter or beats
                # Subsections (### Status, ### Pacing) should go RIGHT AFTER the chapter heading
                # New chapters or beats should go at the END of the chapter
                text_preview = text.strip()[:50]  # Look at first 50 chars of what we're inserting
                
                if text_preview.startswith("###"):
                    # Inserting subsections within the chapter - put right after heading line
                    # Find the end of the heading line (next newline after anchor)
                    end_of_line = content.find("\n", pos + len(anchor_text))
                    if end_of_line != -1:
                        end_pos = end_of_line + 1  # After the newline
                    else:
                        # No newline found - use right after anchor text
                        end_pos = pos + len(anchor_text)
                    logger.info(f"Inserting subsection (###) right after chapter heading at position {end_pos}")
                else:
                    # Inserting new chapter or beats - find end of this chapter
                    # Find end of this chapter by looking for next chapter heading
                    # This ensures new chapters are inserted in the correct order:
                    # - If anchor is "## Chapter 3", we find where Chapter 3 ends
                    # - If Chapter 4 already exists, insert before it (shouldn't happen when adding new chapter)
                    # - If Chapter 4 doesn't exist yet, insert at end (correct - new chapter goes at end)
                    next_chapter_pattern = re.compile(r"\n##\s+Chapter\s+\d+", re.MULTILINE)
                    search_start_pos = pos + len(anchor_text)
                    match = next_chapter_pattern.search(content, search_start_pos)
                    if match:
                        # Insert before the next chapter (maintains sequential order)
                        end_pos = match.start()
                        logger.debug(f"Found next chapter at position {match.start()}, inserting before it")
                    else:
                        # This is the last chapter, insert at end of document
                        # This is correct for adding new chapters - they go after the last existing chapter
                        end_pos = len(content)
                        # Log to help diagnose why next chapter wasn't found
                        snippet_start = max(0, search_start_pos)
                        snippet_end = min(len(content), search_start_pos + 500)
                        snippet = content[snippet_start:snippet_end]
                        logger.debug(f"No next chapter found after position {search_start_pos}")
                        logger.debug(f"Content snippet after anchor (first 200 chars): {repr(snippet[:200])}")
            elif op_type == "insert_after":
                # For insert_after (continuing text), find the end of the paragraph containing the anchor
                # This ensures we continue from the END of the paragraph, not mid-paragraph
                anchor_end = pos + len(anchor_text)
                
                # Find the end of the paragraph: look for paragraph break (\n\n) or heading (\n#)
                # Start searching from the end of the anchor_text match
                para_break = content.find("\n\n", anchor_end)
                heading_break = content.find("\n#", anchor_end)
                
                # Use whichever comes first (or end of document if neither found)
                if heading_break != -1 and (para_break == -1 or heading_break < para_break):
                    # Heading comes first - insert before it
                    end_pos = heading_break
                elif para_break != -1:
                    # Paragraph break found - insert there
                    end_pos = para_break
                else:
                    # No clear paragraph boundary - use end of document
                    end_pos = len(content)
            else:
                # For insert_after_heading (non-chapter headings), find end of section
                # Similar to chapters: find where this section ends (next heading at same or higher level)
                anchor_start = pos
                anchor_end = pos + len(anchor_text)
                
                # Check if anchor_text is a markdown heading (starts with #)
                heading_match = re.match(r'^(#{1,6})\s+', anchor_text.strip())
                if heading_match:
                    # It's a markdown heading - find the end of this section
                    heading_level = len(heading_match.group(1))
                    
                    # Find next heading at same or higher level (marks end of this section)
                    # Pattern: \n followed by 1 to heading_level #, then space and text
                    # We want headings at level <= heading_level (same or higher level = section boundary)
                    # Example: For level 2 (##), match \n## or \n# (but not \n###)
                    # Build pattern string: {1,heading_level} means 1 to heading_level hashes
                    hash_range = '{1,' + str(heading_level) + '}'
                    pattern_str = r'\n(#' + hash_range + r')\s+'
                    next_heading_pattern = re.compile(pattern_str, re.MULTILINE)
                    match = next_heading_pattern.search(content, anchor_end)
                    
                    if match:
                        # Found next section - insert before it (at end of current section)
                        end_pos = match.start()
                    else:
                        # This is the last section, insert at end of document
                        end_pos = len(content)
                    
                    logger.info(f"Section heading detected (level {heading_level}) - inserting at section end: position {end_pos}")
                else:
                    # Not a markdown heading - fall back to end of line
                    end_pos = content.find("\n", pos)
                    if end_pos == -1:
                        end_pos = len(content)
                    else:
                        end_pos += 1
            # Guard frontmatter: ensure insertions never occur before frontmatter end
            end_pos = max(end_pos, frontmatter_end)
            # Use confidence from fuzzy matching (0.8-1.0), but cap minimum at 0.8 for successful matches
            final_confidence = max(confidence, 0.8)
            return end_pos, end_pos, text, final_confidence
        else:
            # Anchor text not found - return failure signal so fallback can handle it
            logger.warning(f"⚠️ Anchor text not found: {repr(anchor_text[:50])}")
            return -1, -1, text, 0.0
    
    # Strategy 4: Left + right context
    if left_context and right_context:
        pattern = re.escape(left_context) + r"([\s\S]{0,400}?)" + re.escape(right_context)
        m = re.search(pattern, content)
        if m:
            # Guard frontmatter: ensure operations never occur before frontmatter end
            start = max(m.start(1), frontmatter_end)
            end = max(m.end(1), start)
            return start, end, text, 0.8
    
    # Strategy 5: Fallback for operations without original_text (e.g., insert operations without anchor)
    # Special handling for insert_after_heading without anchor_text (empty files)
    if op_type == "insert_after_heading" and not anchor_text:
        # Empty file case - insert after frontmatter
        body_only = content[frontmatter_end:].strip()
        if not body_only:
            logger.info("insert_after_heading without anchor_text on empty file - inserting after frontmatter")
            return frontmatter_end, frontmatter_end, text, 0.8
        else:
            # File has content but no anchor - use frontmatter end as fallback
            logger.warning("insert_after_heading without anchor_text on non-empty file - using frontmatter end")
            return frontmatter_end, frontmatter_end, text, 0.5
    
    # If we have original_text and got here, it's an error - should have been caught above
    if not original_text or op_type not in ("replace_range", "delete_range"):
        start = op_dict.get("start", 0)
        end = op_dict.get("end", 0)
        # Handle None values (malformed operations)
        if start is None:
            start = 0
        if end is None:
            end = 0
        # Guard frontmatter: ensure operations never occur before frontmatter end
        start = max(start, frontmatter_end)
        end = max(end, start)
        
        # Special handling for empty files: if body is empty, insert at frontmatter end
        body_only = strip_frontmatter_block(content)
        if not body_only.strip() and op_type in ("insert_after_heading", "insert_after"):
            # Empty file - insert at frontmatter end
            return frontmatter_end, frontmatter_end, text, 0.6
        
        return start, end, text, 0.5
    else:
        # We have original_text but search failed - this should never happen if we got here
        # (should have been caught in the error handling above)
        logger.error(f"❌ INTERNAL ERROR: original_text search failed but reached fallback")
        return -1, -1, text, 0.0

