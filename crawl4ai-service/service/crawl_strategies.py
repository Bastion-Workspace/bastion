"""
Crawl4AI Strategy Configuration
Handles extraction strategy setup for different crawl types
"""

import logging
from typing import Optional, Any, Dict
from crawl4ai import (
    LLMExtractionStrategy,
    CosineStrategy,
    RegexChunking,
    LLMConfig
)

logger = logging.getLogger(__name__)


def get_extraction_strategy(
    strategy_name: str,
    chunking_strategy: str = "RegexChunking",
    word_count_threshold: int = 10,
    llm_question: Optional[str] = None
) -> Optional[Any]:
    """
    Get configured extraction strategy for Crawl4AI
    
    Args:
        strategy_name: Name of extraction strategy ("LLMExtractionStrategy", "CosineStrategy", etc.)
        chunking_strategy: Chunking strategy name
        word_count_threshold: Minimum word count for chunks
        llm_question: Optional question for LLM extraction
        
    Returns:
        Configured extraction strategy or None for basic extraction
    """
    try:
        # Configure chunking strategy
        if chunking_strategy == "RegexChunking":
            chunking = RegexChunking()
        else:
            # Default to RegexChunking
            chunking = RegexChunking()
        
        # Configure extraction strategy
        if strategy_name == "LLMExtractionStrategy" or strategy_name == "llm_extraction":
            instruction = llm_question or (
                "Extract the main content, key points, and important information from this web page. "
                "Focus on factual content, quotes, data, and key insights. "
                "Format as structured JSON with content blocks."
            )
            
            return LLMExtractionStrategy(
                llm_config=LLMConfig(provider="openai"),
                instruction=instruction,
                extraction_type="block",
                apply_chunking=True,
                chunking_strategy=chunking,
                word_count_threshold=word_count_threshold
            )
        elif strategy_name == "CosineStrategy":
            return CosineStrategy(
                semantic_filter="Extract main content and key information",
                word_count_threshold=word_count_threshold,
                apply_chunking=True,
                chunking_strategy=chunking
            )
        else:
            # Basic extraction - return None to use default
            return None
            
    except Exception as e:
        logger.warning(f"Failed to configure extraction strategy {strategy_name}: {e}, using basic extraction")
        return None


def prepare_crawl_kwargs(
    url: str,
    extraction_strategy: Optional[str] = None,
    chunking_strategy: str = "RegexChunking",
    css_selector: Optional[str] = None,
    llm_question: Optional[str] = None,
    max_content_length: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
    virtual_scroll: bool = False,
    scroll_delay: float = 1.0,
    use_fit_markdown: bool = False,
    word_count_threshold: int = 10
) -> Dict[str, Any]:
    """
    Prepare kwargs for crawler.arun() call
    
    Returns:
        Dictionary of kwargs for crawler.arun()
    """
    kwargs = {
        "url": url,
        "bypass_cache": False,
    }
    
    # Add CSS selector if provided
    if css_selector:
        kwargs["css_selector"] = css_selector
    
    # Add virtual scroll if requested
    if virtual_scroll:
        kwargs["virtual_scroll"] = True
        kwargs["scroll_delay"] = scroll_delay
    
    # Add markdown generator
    if use_fit_markdown:
        kwargs["markdown_generator"] = "fit_markdown"
    
    # Add timeout if provided
    if timeout_seconds:
        kwargs["timeout"] = timeout_seconds * 1000  # Convert to milliseconds
    
    # Add extraction strategy if provided
    if extraction_strategy:
        strategy_config = get_extraction_strategy(
            extraction_strategy,
            chunking_strategy,
            word_count_threshold,
            llm_question
        )
        if strategy_config is not None:
            kwargs["extraction_strategy"] = strategy_config
    
    return kwargs







