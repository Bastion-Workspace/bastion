"""
Entity Extraction Utility
Uses backend's DocumentProcessor spaCy NER via gRPC
"""

import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


async def extract_entities_from_text(text: str, backend_client) -> List[Dict[str, Any]]:
    """
    Extract entities from text using backend DocumentProcessor
    
    Falls back to simple heuristics if backend unavailable
    
    Args:
        text: Text to extract entities from
        backend_client: BackendToolClient instance
        
    Returns:
        List of entity dicts with name, type, confidence
    """
    try:
        # TODO: Add gRPC method for entity extraction via DocumentProcessor
        # For now, use simple heuristic extraction
        
        entities = []
        seen = set()
        
        # Extract capitalized phrases (2-3 words max)
        words = text.split()
        i = 0
        while i < len(words):
            # Look for capitalized sequences
            if words[i] and words[i][0].isupper():
                phrase = words[i]
                
                # Check if next 1-2 words are also capitalized
                if i + 1 < len(words) and words[i+1] and words[i+1][0].isupper():
                    phrase += " " + words[i+1]
                    if i + 2 < len(words) and words[i+2] and words[i+2][0].isupper():
                        phrase += " " + words[i+2]
                
                # Clean and validate
                phrase = phrase.strip(".,!?;:")
                if len(phrase) > 2 and phrase.lower() not in ["the", "and", "or"] and phrase not in seen:
                    entities.append({
                        "name": phrase,
                        "type": "PERSON",  # Default - backend would provide proper typing
                        "confidence": 0.7
                    })
                    seen.add(phrase)
            
            i += 1
        
        logger.info(f"Extracted {len(entities)} entities from text")
        return entities[:20]  # Limit to top 20
        
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return []


def extract_entities_from_search_results(
    vector_results: List[Dict[str, Any]],
    max_results: int = 3
) -> List[str]:
    """
    Extract entity names from vector search result previews
    
    Args:
        vector_results: List of document search results
        max_results: Number of top results to analyze
        
    Returns:
        List of unique entity names
    """
    try:
        combined_text = ""
        for result in vector_results[:max_results]:
            preview = result.get("content_preview", "")
            combined_text += " " + preview
        
        # Simple extraction - look for capitalized words/phrases
        entities = set()
        words = combined_text.split()
        
        for i, word in enumerate(words):
            if word and len(word) > 2 and word[0].isupper():
                clean_word = word.strip(".,!?;:")
                if clean_word and clean_word.lower() not in ["the", "and", "or", "but"]:
                    entities.add(clean_word)
        
        return list(entities)[:15]
        
    except Exception as e:
        logger.error(f"Entity extraction from results failed: {e}")
        return []

