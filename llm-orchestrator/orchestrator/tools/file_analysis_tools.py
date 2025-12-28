"""
File Analysis Tools - Deterministic text analysis for LangGraph agents
Provides word count, line count, character count, and other text metrics
"""

import logging
from typing import Dict, Any, Optional

from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)


async def analyze_text_metrics(
    text: str,
    include_advanced: bool = False,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Analyze raw text content and return metrics
    
    Args:
        text: Text content to analyze
        include_advanced: If True, include advanced metrics (averages, etc.)
        user_id: User ID for logging
        
    Returns:
        Dictionary with text analysis metrics:
        - word_count: int
        - line_count: int
        - non_empty_line_count: int
        - character_count: int
        - character_count_no_spaces: int
        - paragraph_count: int
        - sentence_count: int
        - avg_words_per_sentence: float (if include_advanced)
        - avg_words_per_paragraph: float (if include_advanced)
    """
    try:
        logger.debug(f"Analyzing text metrics: {len(text)} chars, include_advanced={include_advanced}")
        
        # Get backend client
        client = await get_backend_tool_client()
        
        # Call backend analysis service via gRPC
        metrics = await client.analyze_text_content(
            content=text,
            include_advanced=include_advanced,
            user_id=user_id
        )
        
        if "error" in metrics:
            logger.error(f"Text analysis failed: {metrics['error']}")
            return metrics
        
        logger.debug(f"Text analysis complete: {metrics.get('word_count', 0)} words")
        return metrics
        
    except Exception as e:
        logger.error(f"Error analyzing text metrics: {e}")
        return {
            "word_count": 0,
            "line_count": 0,
            "non_empty_line_count": 0,
            "character_count": 0,
            "character_count_no_spaces": 0,
            "paragraph_count": 0,
            "sentence_count": 0,
            "error": str(e)
        }


async def analyze_document_metrics(
    document_id: str,
    user_id: str = "system",
    include_advanced: bool = False
) -> Dict[str, Any]:
    """
    Analyze document content by document_id and return metrics
    
    Args:
        document_id: Document ID to analyze
        user_id: User ID for access control
        include_advanced: If True, include advanced metrics (averages, etc.)
        
    Returns:
        Dictionary with text analysis metrics (same as analyze_text_metrics)
        or error dict if document not found
    """
    try:
        logger.info(f"Analyzing document metrics: document_id={document_id}")
        
        # Get backend client
        client = await get_backend_tool_client()
        
        # Fetch document content
        content = await client.get_document_content(document_id, user_id)
        
        if content is None:
            logger.warning(f"Document not found or access denied: {document_id}")
            return {
                "word_count": 0,
                "line_count": 0,
                "non_empty_line_count": 0,
                "character_count": 0,
                "character_count_no_spaces": 0,
                "paragraph_count": 0,
                "sentence_count": 0,
                "error": "Document not found or access denied"
            }
        
        # Analyze content
        return await analyze_text_metrics(
            text=content,
            include_advanced=include_advanced,
            user_id=user_id
        )
        
    except Exception as e:
        logger.error(f"Error analyzing document metrics: {e}")
        return {
            "word_count": 0,
            "line_count": 0,
            "non_empty_line_count": 0,
            "character_count": 0,
            "character_count_no_spaces": 0,
            "paragraph_count": 0,
            "sentence_count": 0,
            "error": str(e)
        }


async def analyze_active_editor_metrics(
    active_editor: Dict[str, Any],
    include_advanced: bool = False,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Analyze active editor content and return metrics
    
    Args:
        active_editor: Active editor dict with 'content' key
        include_advanced: If True, include advanced metrics (averages, etc.)
        user_id: User ID for logging
        
    Returns:
        Dictionary with text analysis metrics (same as analyze_text_metrics)
        or error dict if active_editor is invalid
    """
    try:
        if not active_editor:
            logger.warning("Active editor is empty")
            return {
                "word_count": 0,
                "line_count": 0,
                "non_empty_line_count": 0,
                "character_count": 0,
                "character_count_no_spaces": 0,
                "paragraph_count": 0,
                "sentence_count": 0,
                "error": "Active editor is empty"
            }
        
        content = active_editor.get("content", "")
        if not content:
            logger.warning("Active editor content is empty")
            return {
                "word_count": 0,
                "line_count": 0,
                "non_empty_line_count": 0,
                "character_count": 0,
                "character_count_no_spaces": 0,
                "paragraph_count": 0,
                "sentence_count": 0,
                "error": "Active editor content is empty"
            }
        
        logger.debug(f"Analyzing active editor metrics: {len(content)} chars")
        
        # Analyze content
        return await analyze_text_metrics(
            text=content,
            include_advanced=include_advanced,
            user_id=user_id
        )
        
    except Exception as e:
        logger.error(f"Error analyzing active editor metrics: {e}")
        return {
            "word_count": 0,
            "line_count": 0,
            "non_empty_line_count": 0,
            "character_count": 0,
            "character_count_no_spaces": 0,
            "paragraph_count": 0,
            "sentence_count": 0,
            "error": str(e)
        }
