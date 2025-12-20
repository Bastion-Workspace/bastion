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
        # First try exact match
        count = 0
        search_from = 0
        found_positions = []
        while True:
            pos = content.find(original_text, search_from)
            if pos == -1:
                break
            found_positions.append(pos)
            if count == occurrence_index:
                end_pos = pos + len(original_text)
                # Guard frontmatter: ensure operations never occur before frontmatter end
                pos = max(pos, frontmatter_end)
                end_pos = max(end_pos, pos)
                logger.debug(f"✅ Found original_text at position {pos} (occurrence {count})")
                return pos, end_pos, text, 1.0
            count += 1
            search_from = pos + 1
        
        # If exact match failed, try with normalized whitespace
        if not found_positions:
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
            
            logger.error(f"   ❌ Could not find original_text in content even with normalization")
            logger.error(f"   original_text length: {len(original_text)}, preview: {repr(original_text[:200])}")
            logger.error(f"   Content length: {len(content)}")
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
        
        # Search entire document (from frontmatter_end to end) for anchor text
        # Use last occurrence for consistency - this ensures:
        # - New chapters are added at the end (last "## Chapter 2" match)
        # - Edits to sections use the most recent occurrence
        # - Cursor position doesn't constrain where we can anchor
        pos = content.rfind(anchor_text, search_start)
        
        # If not found in body, try searching from start (in case frontmatter_end calculation was off)
        if pos == -1:
            pos = content.rfind(anchor_text)
        
        if pos != -1:
            # For chapter headings, find the end of the entire chapter (not just the heading line)
            # This ensures proper sequential ordering: Chapter 4 comes after Chapter 3, etc.
            if anchor_text.startswith("## Chapter") or anchor_text.startswith("# Chapter"):
                # Find end of this chapter by looking for next chapter heading
                # This ensures new chapters are inserted in the correct order:
                # - If anchor is "## Chapter 3", we find where Chapter 3 ends
                # - If Chapter 4 already exists, insert before it (shouldn't happen when adding new chapter)
                # - If Chapter 4 doesn't exist yet, insert at end (correct - new chapter goes at end)
                next_chapter_pattern = re.compile(r"\n##\s+Chapter\s+\d+", re.MULTILINE)
                match = next_chapter_pattern.search(content, pos + len(anchor_text))
                if match:
                    # Insert before the next chapter (maintains sequential order)
                    end_pos = match.start()
                else:
                    # This is the last chapter, insert at end of document
                    # This is correct for adding new chapters - they go after the last existing chapter
                    end_pos = len(content)
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
                # For insert_after_heading (non-chapter headings), find end of line/paragraph
                end_pos = content.find("\n", pos)
                if end_pos == -1:
                    end_pos = len(content)
                else:
                    end_pos += 1
            # Guard frontmatter: ensure insertions never occur before frontmatter end
            end_pos = max(end_pos, frontmatter_end)
            return end_pos, end_pos, text, 0.9
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

