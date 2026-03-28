"""
LLM Credentials Helper - Extract OpenRouter/OpenAI credentials from orchestrator metadata.

Centralizes the credential resolution logic used by BaseAgent._get_llm and pipeline_executor._get_llm_for_pipeline
to ensure consistent behavior across all tools and services.
"""

from typing import Dict, Optional, Tuple

try:
    from config.settings import settings
except ImportError:
    settings = None


def get_openrouter_credentials(metadata: Optional[Dict] = None) -> Tuple[Optional[str], str]:
    """
    Extract OpenRouter/OpenAI API key and base URL from metadata with fallback to settings.
    
    Precedence for api_key:
    1. metadata.get("user_llm_api_key")
    2. settings.OPENROUTER_API_KEY
    3. settings.OPENAI_API_KEY
    
    Precedence for base_url:
    1. metadata.get("user_llm_base_url")
    2. settings.OPENROUTER_BASE_URL
    
    Args:
        metadata: Optional metadata dict (from orchestrator state or gRPC request)
        
    Returns:
        Tuple of (api_key, base_url). api_key may be None if no credentials configured.
    """
    meta = metadata or {}
    
    # Resolve API key with fallback chain
    api_key = meta.get("user_llm_api_key")
    if not api_key and settings:
        api_key = getattr(settings, "OPENROUTER_API_KEY", None)
        if not api_key:
            api_key = getattr(settings, "OPENAI_API_KEY", None)
    
    # Resolve base URL with fallback
    base_url = meta.get("user_llm_base_url")
    if not base_url and settings:
        base_url = getattr(settings, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    return (api_key, base_url)
