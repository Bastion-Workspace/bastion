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


def _normalize_openai_api_base(
    base_url: str, provider_type: str = ""
) -> str:
    """
    OpenAI-compatible local servers (Ollama, vLLM) expect base URL to end in /v1
    so that POST /chat/completions maps to the server's /v1/chat/completions.
    """
    u = (base_url or "").strip().rstrip("/")
    if not u or u.endswith("/v1"):
        return u
    p = (provider_type or "").lower()
    if p in ("ollama", "vllm") or ":11434" in u:
        return f"{u}/v1"
    return u


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

    ptype = (meta.get("user_llm_provider_type") or "").strip()
    if base_url:
        base_url = _normalize_openai_api_base(base_url, ptype)

    return (api_key, base_url)
