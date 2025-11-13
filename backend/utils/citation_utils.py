"""
Citation utilities for ensuring proper formatting and preventing corruption
"""
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def validate_and_format_citations(citations: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Validate and format citations to ensure they're always a proper list of dictionaries.
    
    Args:
        citations: Raw citations data that might be None, string, or malformed
        
    Returns:
        Properly formatted list of citation dictionaries
    """
    if citations is None:
        return []
    
    if isinstance(citations, str):
        try:
            parsed = json.loads(citations)
            if isinstance(parsed, list):
                return parsed
            else:
                logger.warning(f"Citations string parsed to non-list: {parsed}")
                return []
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse citations string: {citations} - Error: {e}")
            return []
    
    if isinstance(citations, list):
        # Validate each citation is a dictionary
        valid_citations = []
        for i, citation in enumerate(citations):
            if isinstance(citation, dict):
                valid_citations.append(citation)
            else:
                logger.warning(f"Invalid citation at index {i}: {citation} (not a dict)")
        return valid_citations
    
    logger.warning(f"Invalid citations type: {type(citations)} - {citations}")
    return []


def citations_to_json(citations: Optional[List[Dict[str, Any]]]) -> str:
    """
    Convert citations to JSON string for database storage.
    
    Args:
        citations: List of citation dictionaries
        
    Returns:
        JSON string representation
    """
    formatted_citations = validate_and_format_citations(citations)
    return json.dumps(formatted_citations)


def citations_from_json(citations_json: Optional[str]) -> List[Dict[str, Any]]:
    """
    Parse citations from JSON string from database.
    
    Args:
        citations_json: JSON string from database
        
    Returns:
        List of citation dictionaries
    """
    if not citations_json:
        return []
    
    try:
        parsed = json.loads(citations_json)
        return validate_and_format_citations(parsed)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse citations JSON: {citations_json} - Error: {e}")
        return [] 