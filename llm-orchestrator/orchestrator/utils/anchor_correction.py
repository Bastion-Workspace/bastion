"""
Anchor Text Correction Utilities

Provides fuzzy matching and auto-correction for LLM-generated anchor texts
to prevent minor hallucinations (tense changes, punctuation, etc.) from 
breaking operations.
"""

import logging
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def extract_anchor_candidates(
    manuscript: str,
    num_lines: int = 10,
    from_end: bool = True
) -> List[Dict[str, any]]:
    """
    Extract potential anchor lines from manuscript for structured selection.
    
    Args:
        manuscript: The manuscript text to extract from
        num_lines: Number of lines to extract
        from_end: If True, extract from end (for new chapter insertion)
                 If False, extract from beginning
    
    Returns:
        List of dicts with 'index', 'text', and 'recommended' fields
    """
    lines = [line for line in manuscript.split('\n') if line.strip()]
    
    if not lines:
        return []
    
    if from_end:
        # Get last N lines
        candidates = lines[-num_lines:] if len(lines) > num_lines else lines
        start_idx = len(lines) - len(candidates)
    else:
        # Get first N lines
        candidates = lines[:num_lines]
        start_idx = 0
    
    result = []
    for i, line in enumerate(candidates):
        result.append({
            "index": start_idx + i,
            "text": line.strip(),
            "recommended": (i == len(candidates) - 1) if from_end else (i == 0)
        })
    
    return result


def fuzzy_match_anchor(
    attempted_anchor: str,
    manuscript: str,
    threshold: float = 0.85,
    max_candidates: int = 1000
) -> Optional[Tuple[str, float]]:
    """
    Find the closest matching text in manuscript using fuzzy string matching.
    
    Uses Python's difflib.SequenceMatcher for similarity scoring.
    
    Args:
        attempted_anchor: The anchor text attempted by the LLM
        manuscript: The manuscript to search in
        threshold: Minimum similarity ratio (0.0-1.0) to accept
        max_candidates: Maximum number of candidate chunks to check
    
    Returns:
        Tuple of (matched_text, similarity_score) or None if no match above threshold
    """
    if not attempted_anchor or not manuscript:
        return None
    
    # Normalize whitespace for comparison
    normalized_attempt = " ".join(attempted_anchor.split())
    
    # Split manuscript into potential candidates
    # Try multiple granularities: lines, sentences, sliding windows
    candidates = []
    
    # 1. Line-based candidates
    lines = manuscript.split('\n')
    candidates.extend([line.strip() for line in lines if line.strip()])
    
    # 2. Sentence-based candidates (split on period, but preserve)
    sentences = re.split(r'(?<=[.!?])\s+', manuscript)
    candidates.extend([sent.strip() for sent in sentences if sent.strip() and len(sent) > 20])
    
    # 3. Sliding window candidates (for multi-line anchors)
    attempt_length = len(normalized_attempt)
    if attempt_length > 100:  # Only for longer anchors
        window_size = attempt_length + 50  # Add buffer
        step = window_size // 2
        for i in range(0, len(manuscript) - window_size, step):
            chunk = manuscript[i:i+window_size].strip()
            if chunk:
                candidates.append(chunk)
    
    # Limit candidates to prevent performance issues
    if len(candidates) > max_candidates:
        # Sample evenly across the manuscript
        step = len(candidates) // max_candidates
        candidates = candidates[::step][:max_candidates]
    
    # Find best match
    best_match = None
    best_score = 0.0
    
    for candidate in candidates:
        normalized_candidate = " ".join(candidate.split())
        
        # Use SequenceMatcher for similarity
        matcher = SequenceMatcher(None, normalized_attempt, normalized_candidate)
        score = matcher.ratio()
        
        if score > best_score:
            best_score = score
            best_match = candidate
    
    if best_score >= threshold:
        logger.info(f"🔍 Fuzzy match found: score={best_score:.2f}")
        logger.info(f"   Attempted: {normalized_attempt[:100]}...")
        logger.info(f"   Matched:   {best_match[:100] if best_match else 'None'}...")
        return (best_match, best_score)
    
    logger.warning(f"⚠️ No fuzzy match above threshold {threshold} (best={best_score:.2f})")
    return None


def auto_correct_operation_anchor(
    operation: Dict[str, any],
    manuscript: str,
    threshold: float = 0.85
) -> Tuple[Dict[str, any], bool]:
    """
    Auto-correct an operation's anchor text using fuzzy matching.
    
    Args:
        operation: The operation dict with 'anchor_text' or 'original_text'
        manuscript: The manuscript to search for corrections
        threshold: Minimum similarity for auto-correction
    
    Returns:
        Tuple of (corrected_operation, was_corrected)
    """
    op_type = operation.get("op_type", "")
    
    # Determine which field needs correction
    if op_type in ("replace_range", "delete_range"):
        anchor_field = "original_text"
    elif op_type in ("insert_after_heading", "insert_after"):
        anchor_field = "anchor_text"
    else:
        # Unknown operation type
        return (operation, False)
    
    attempted_anchor = operation.get(anchor_field)
    if not attempted_anchor:
        # No anchor to correct
        return (operation, False)
    
    # Try fuzzy match
    match_result = fuzzy_match_anchor(attempted_anchor, manuscript, threshold)
    
    if match_result:
        corrected_text, score = match_result
        
        # Create corrected operation
        corrected_op = operation.copy()
        corrected_op[anchor_field] = corrected_text
        corrected_op["_anchor_corrected"] = True
        corrected_op["_anchor_correction_score"] = score
        corrected_op["_anchor_original_attempt"] = attempted_anchor
        
        logger.info(f"✅ Auto-corrected {anchor_field} (similarity: {score:.2f})")
        return (corrected_op, True)
    
    # No correction possible
    return (operation, False)


def batch_correct_operations(
    operations: List[Dict[str, any]],
    manuscript: str,
    threshold: float = 0.85
) -> Tuple[List[Dict[str, any]], int]:
    """
    Auto-correct multiple operations in batch.
    
    Args:
        operations: List of operation dicts
        manuscript: The manuscript to search for corrections
        threshold: Minimum similarity for auto-correction
    
    Returns:
        Tuple of (corrected_operations, num_corrected)
    """
    corrected_ops = []
    num_corrected = 0
    
    for op in operations:
        corrected_op, was_corrected = auto_correct_operation_anchor(op, manuscript, threshold)
        corrected_ops.append(corrected_op)
        if was_corrected:
            num_corrected += 1
    
    if num_corrected > 0:
        logger.info(f"📊 Auto-corrected {num_corrected}/{len(operations)} operations")
    
    return (corrected_ops, num_corrected)


def format_anchor_candidates_for_prompt(
    candidates: List[Dict[str, any]],
    context_name: str = "previous chapter"
) -> str:
    """
    Format anchor candidates as a numbered list for LLM selection.
    
    Args:
        candidates: List of candidate dicts from extract_anchor_candidates()
        context_name: Descriptive name for context (e.g., "previous chapter", "current section")
    
    Returns:
        Formatted string for inclusion in LLM prompt
    """
    if not candidates:
        return ""
    
    lines = []
    lines.append(f"\n=== ANCHOR SELECTION: Last lines of {context_name} ===\n")
    lines.append("CRITICAL: Do NOT copy these lines - reference by INDEX NUMBER!\n")
    lines.append("The system will use the EXACT text - this prevents copy errors!\n\n")
    
    for candidate in candidates:
        idx = candidate["index"]
        text = candidate["text"]
        recommended = " ← RECOMMENDED for new chapter" if candidate["recommended"] else ""
        lines.append(f'[{idx}] "{text}"{recommended}\n')
    
    lines.append("\nIn your JSON response, use: \"anchor_index\": <number>\n")
    lines.append("Example: \"anchor_index\": 4  (to insert after line [4])\n")
    lines.append("="*80 + "\n")
    
    return "".join(lines)


def resolve_anchor_index(
    anchor_index: Optional[int],
    candidates: List[Dict[str, any]]
) -> Optional[str]:
    """
    Resolve an anchor index to the actual text.
    
    Args:
        anchor_index: The index selected by the LLM
        candidates: List of candidate dicts
    
    Returns:
        The exact anchor text, or None if invalid index
    """
    if anchor_index is None:
        return None
    
    for candidate in candidates:
        if candidate["index"] == anchor_index:
            return candidate["text"]
    
    logger.warning(f"⚠️ Invalid anchor_index {anchor_index} - not in candidates")
    return None
