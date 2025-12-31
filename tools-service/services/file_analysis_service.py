"""
File Analysis Service - Deterministic text analysis utilities
Provides word count, line count, character count, and other text metrics
"""

import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class FileAnalysisService:
    """
    File Analysis Service for deterministic text metrics
    
    Provides pure Python text analysis functions with no external dependencies.
    Used by gRPC service and can be called directly for local analysis.
    """
    
    def __init__(self):
        """Initialize file analysis service"""
        logger.info("File Analysis Service initialized")
    
    def analyze_text(self, content: str, include_advanced: bool = False) -> Dict[str, Any]:
        """
        Analyze text content and return comprehensive metrics
        
        Args:
            content: Text content to analyze
            include_advanced: If True, include advanced metrics (averages, etc.)
            
        Returns:
            Dictionary with text analysis metrics
        """
        if not content:
            return self._empty_metrics(include_advanced)
        
        try:
            metrics = {
                "word_count": self.count_words(content),
                "line_count": self.count_lines(content),
                "non_empty_line_count": self.count_non_empty_lines(content),
                "character_count": self.count_characters(content),
                "character_count_no_spaces": self.count_characters_no_spaces(content),
                "paragraph_count": self.count_paragraphs(content),
                "sentence_count": self.count_sentences(content),
            }
            
            if include_advanced:
                metrics.update(self._calculate_advanced_metrics(content, metrics))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return self._empty_metrics(include_advanced)
    
    def count_words(self, text: str) -> int:
        """
        Count words in text using regex pattern
        
        Uses \b\w+\b pattern to match word boundaries.
        Consistent with prose_quality.py word_count() function.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Number of words
        """
        if not text:
            return 0
        return len(re.findall(r"\b\w+\b", text))
    
    def count_lines(self, text: str) -> int:
        """
        Count total lines in text (including empty lines)
        
        Args:
            text: Text content to analyze
            
        Returns:
            Number of lines
        """
        if not text:
            return 0
        # Count newlines, add 1 if text doesn't end with newline
        lines = text.count('\n')
        if text and not text.endswith('\n'):
            lines += 1
        return max(1, lines)  # At least 1 line if text exists
    
    def count_non_empty_lines(self, text: str) -> int:
        """
        Count lines with actual content (non-empty)
        
        Args:
            text: Text content to analyze
            
        Returns:
            Number of non-empty lines
        """
        if not text:
            return 0
        lines = text.splitlines()
        return len([line for line in lines if line.strip()])
    
    def count_characters(self, text: str) -> int:
        """
        Count total characters in text (including spaces)
        
        Args:
            text: Text content to analyze
            
        Returns:
            Number of characters
        """
        if not text:
            return 0
        return len(text)
    
    def count_characters_no_spaces(self, text: str) -> int:
        """
        Count characters excluding spaces
        
        Args:
            text: Text content to analyze
            
        Returns:
            Number of characters without spaces
        """
        if not text:
            return 0
        return len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
    
    def count_paragraphs(self, text: str) -> int:
        """
        Count paragraphs by detecting paragraph breaks (\n\n)
        
        Args:
            text: Text content to analyze
            
        Returns:
            Number of paragraphs
        """
        if not text:
            return 0
        
        # Count double newlines (paragraph breaks)
        # Also handle single newline at start/end
        text_stripped = text.strip()
        if not text_stripped:
            return 0
        
        # Split on double newlines
        paragraphs = re.split(r'\n\s*\n', text_stripped)
        # Filter out empty paragraphs
        paragraphs = [p for p in paragraphs if p.strip()]
        
        return max(1, len(paragraphs))  # At least 1 paragraph if text exists
    
    def count_sentences(self, text: str) -> int:
        """
        Approximate sentence count using punctuation markers
        
        Counts sentences by detecting sentence-ending punctuation (. ! ?)
        This is an approximation and may not be 100% accurate for all cases.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Approximate number of sentences
        """
        if not text:
            return 0
        
        # Remove common abbreviations that end with periods
        # This is a simple heuristic - not perfect but reasonable
        text_cleaned = text
        
        # Count sentence-ending punctuation
        # Look for . ! ? followed by space, newline, or end of string
        sentence_endings = re.findall(r'[.!?]+(?:\s+|$|\n)', text_cleaned)
        
        # If no sentence endings found but text exists, count as 1 sentence
        if not sentence_endings and text.strip():
            return 1
        
        return len(sentence_endings)
    
    def _calculate_advanced_metrics(self, content: str, basic_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate advanced metrics from basic metrics
        
        Args:
            content: Original text content
            basic_metrics: Dictionary with basic metrics already calculated
            
        Returns:
            Dictionary with advanced metrics
        """
        advanced = {}
        
        # Average words per sentence
        word_count = basic_metrics.get("word_count", 0)
        sentence_count = basic_metrics.get("sentence_count", 0)
        if sentence_count > 0:
            advanced["avg_words_per_sentence"] = round(word_count / sentence_count, 2)
        else:
            advanced["avg_words_per_sentence"] = 0.0
        
        # Average words per paragraph
        paragraph_count = basic_metrics.get("paragraph_count", 0)
        if paragraph_count > 0:
            advanced["avg_words_per_paragraph"] = round(word_count / paragraph_count, 2)
        else:
            advanced["avg_words_per_paragraph"] = 0.0
        
        return advanced
    
    def _empty_metrics(self, include_advanced: bool = False) -> Dict[str, Any]:
        """
        Return empty metrics structure
        
        Args:
            include_advanced: If True, include advanced metric fields
            
        Returns:
            Dictionary with zero/empty metrics
        """
        metrics = {
            "word_count": 0,
            "line_count": 0,
            "non_empty_line_count": 0,
            "character_count": 0,
            "character_count_no_spaces": 0,
            "paragraph_count": 0,
            "sentence_count": 0,
        }
        
        if include_advanced:
            metrics.update({
                "avg_words_per_sentence": 0.0,
                "avg_words_per_paragraph": 0.0,
            })
        
        return metrics
