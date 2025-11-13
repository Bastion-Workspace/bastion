"""
Tag Detection Service - Roosevelt's Smart Tag Matching

Detects tag/category references in queries and fuzzy matches to actual tags
"""

import logging
import re
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class TagDetectionService:
    """
    Smart tag detection and fuzzy matching for research queries
    
    **ROOSEVELT'S SMART FILTERING DOCTRINE**: Help users find their organized documents!
    """
    
    # Common phrases that indicate tag/category filtering intent
    TAG_INDICATORS = [
        r"from my ([^,.!?]+)",
        r"in my ([^,.!?]+)",
        r"within my ([^,.!?]+)",
        r"tagged (?:with |as )?([^,.!?]+)",
        r"labelled (?:with |as )?([^,.!?]+)",
        r"categorized as ([^,.!?]+)",
        r"category ([^,.!?]+)",
        r"with tag ([^,.!?]+)",
    ]
    
    def __init__(self):
        self.fuzzy_threshold = 0.75  # 75% similarity required for fuzzy match
        self.high_confidence_threshold = 0.85  # 85% for high confidence
    
    def detect_tag_references(self, query: str, available_tags: List[str] = None, available_categories: List[str] = None) -> List[str]:
        """
        Extract potential tag references from query text
        
        **ROOSEVELT TAG-FIRST APPROACH**: Check if known tags appear in the query!
        Much more reliable than regex pattern matching.
        
        Returns list of phrases that might be tag names
        """
        detected_phrases = []
        query_lower = query.lower()
        
        # **NEW APPROACH**: If we have available tags, check if they appear in the query
        if available_tags:
            for tag in available_tags:
                tag_lower = tag.lower()
                # Check for the tag itself or common variations
                tag_variations = [
                    tag_lower,
                    tag_lower.replace('-', ' '),
                    tag_lower.replace('_', ' '),
                    f"{tag_lower} documents",
                    f"{tag_lower} files",
                    f"{tag_lower} docs",
                ]
                
                for variation in tag_variations:
                    if variation in query_lower:
                        detected_phrases.append(tag)  # Add the original tag, not the variation
                        logger.info(f"✅ Tag '{tag}' found in query via variation '{variation}'")
                        break  # Don't add the same tag multiple times
        
        # **FALLBACK**: Also check categories if provided
        if available_categories:
            for category in available_categories:
                category_lower = category.lower()
                if category_lower in query_lower:
                    detected_phrases.append(category)
                    logger.info(f"✅ Category '{category}' found in query")
        
        # **OLD REGEX APPROACH**: Still use as fallback for phrases we might have missed
        for pattern in self.TAG_INDICATORS:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                phrase = match.group(1).strip()
                # Remove common articles and clean up
                phrase = re.sub(r'\b(the|a|an)\b', '', phrase).strip()
                if phrase and phrase not in detected_phrases:
                    detected_phrases.append(phrase)
                    logger.info(f"📋 Phrase '{phrase}' detected via regex pattern")
        
        return detected_phrases
    
    def fuzzy_match_tag(self, query_phrase: str, available_tags: List[str]) -> Optional[Tuple[str, float]]:
        """
        Find best matching tag using fuzzy matching
        
        Returns (matched_tag, confidence_score) or None
        """
        if not available_tags:
            return None
        
        best_match = None
        best_score = 0.0
        
        query_phrase_lower = query_phrase.lower()
        query_words = set(query_phrase_lower.split())
        
        for tag in available_tags:
            tag_lower = tag.lower()
            
            # Strategy 1: Exact match
            if query_phrase_lower == tag_lower:
                return (tag, 1.0)
            
            # Strategy 2: Substring match (e.g., "founding docs" in "founding-documents")
            if query_phrase_lower in tag_lower or tag_lower in query_phrase_lower:
                score = 0.9
                if score > best_score:
                    best_match = tag
                    best_score = score
                continue
            
            # Strategy 3: Word overlap (e.g., "documents founding" matches "founding-documents")
            tag_words = set(tag_lower.replace('-', ' ').replace('_', ' ').split())
            if query_words and tag_words:
                overlap = len(query_words & tag_words)
                if overlap > 0:
                    score = 0.7 + (overlap / max(len(query_words), len(tag_words))) * 0.2
                    if score > best_score:
                        best_match = tag
                        best_score = score
            
            # Strategy 4: Levenshtein-based similarity (typo tolerance)
            similarity = SequenceMatcher(None, query_phrase_lower, tag_lower).ratio()
            if similarity > best_score:
                best_match = tag
                best_score = similarity
        
        # Only return if above threshold
        if best_score >= self.fuzzy_threshold:
            return (best_match, best_score)
        
        return None
    
    def fuzzy_match_category(self, query_phrase: str, available_categories: List[str]) -> Optional[Tuple[str, float]]:
        """
        Find best matching category using fuzzy matching
        
        Similar to tag matching but for categories
        """
        return self.fuzzy_match_tag(query_phrase, available_categories)
    
    async def detect_and_match_filters(
        self, 
        query: str, 
        available_tags: List[str], 
        available_categories: List[str]
    ) -> Dict:
        """
        Complete tag/category detection and matching pipeline
        
        Returns:
        {
            "filter_tags": ["founding-documents"],
            "filter_category": "research",
            "confidence": "high",  # high/medium/low/none
            "matched_phrases": [("founding docs", "founding-documents", 0.95)],
            "should_filter": True
        }
        """
        result = {
            "filter_tags": [],
            "filter_category": None,
            "confidence": "none",
            "matched_phrases": [],
            "should_filter": False
        }
        
        # **ROOSEVELT TAG-FIRST**: Pass available tags to detection for direct matching
        detected_phrases = self.detect_tag_references(query, available_tags, available_categories)
        
        if not detected_phrases:
            logger.info("🔍 No tag references detected in query")
            return result
        
        logger.info(f"🔍 Detected {len(detected_phrases)} potential tag references: {detected_phrases}")
        
        # Try to match each detected phrase
        matched_tags = []
        overall_confidence = 0.0
        
        for phrase in detected_phrases:
            # **ROOSEVELT OPTIMIZATION**: If phrase is already a known tag (from direct detection), use it directly
            if phrase in available_tags:
                if phrase not in matched_tags:
                    matched_tags.append(phrase)
                    result["matched_phrases"].append((phrase, phrase, 1.0))  # Perfect match!
                    overall_confidence = 1.0
                    logger.info(f"✅ Direct match: '{phrase}' is a known tag (confidence: 1.0)")
                continue
            
            # Check if it's a known category
            if phrase in available_categories:
                result["filter_category"] = phrase
                result["matched_phrases"].append((phrase, f"category:{phrase}", 1.0))
                overall_confidence = 1.0
                logger.info(f"✅ Direct match: '{phrase}' is a known category (confidence: 1.0)")
                continue
            
            # Try fuzzy category match (for misspellings/variations)
            category_match = self.fuzzy_match_category(phrase, available_categories)
            if category_match:
                matched_category, confidence = category_match
                result["filter_category"] = matched_category
                result["matched_phrases"].append((phrase, f"category:{matched_category}", confidence))
                overall_confidence = max(overall_confidence, confidence)
                logger.info(f"✅ Fuzzy matched '{phrase}' to category '{matched_category}' (confidence: {confidence:.2f})")
                continue
            
            # Try fuzzy tag match (for misspellings/variations)
            tag_match = self.fuzzy_match_tag(phrase, available_tags)
            if tag_match:
                matched_tag, confidence = tag_match
                if matched_tag not in matched_tags:
                    matched_tags.append(matched_tag)
                    result["matched_phrases"].append((phrase, matched_tag, confidence))
                    overall_confidence = max(overall_confidence, confidence)
                    logger.info(f"✅ Fuzzy matched '{phrase}' to tag '{matched_tag}' (confidence: {confidence:.2f})")
            else:
                logger.info(f"⚠️ No match found for '{phrase}' (will search all documents)")
        
        result["filter_tags"] = matched_tags
        
        # Determine overall confidence level
        if overall_confidence >= self.high_confidence_threshold:
            result["confidence"] = "high"
        elif overall_confidence >= self.fuzzy_threshold:
            result["confidence"] = "medium"
        elif matched_tags or result["filter_category"]:
            result["confidence"] = "low"
        else:
            result["confidence"] = "none"
        
        # Should we actually filter?
        result["should_filter"] = bool(matched_tags or result["filter_category"])
        
        return result
    
    def format_filter_message(self, detection_result: Dict) -> str:
        """
        Format a user-friendly message about detected filters
        
        For logging and optional user feedback
        """
        if not detection_result["should_filter"]:
            return "Searching all documents (no tag filters detected)"
        
        parts = []
        
        if detection_result["filter_category"]:
            parts.append(f"category '{detection_result['filter_category']}'")
        
        if detection_result["filter_tags"]:
            tags_str = "', '".join(detection_result["filter_tags"])
            parts.append(f"tags '{tags_str}'")
        
        confidence = detection_result["confidence"]
        filter_desc = " and ".join(parts)
        
        return f"Filtering by {filter_desc} ({confidence} confidence)"


# Singleton instance
_tag_detection_service = None

def get_tag_detection_service() -> TagDetectionService:
    """Get or create tag detection service singleton"""
    global _tag_detection_service
    if _tag_detection_service is None:
        _tag_detection_service = TagDetectionService()
    return _tag_detection_service

