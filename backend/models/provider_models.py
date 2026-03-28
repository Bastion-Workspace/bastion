"""
Shared provider/source contract for admin and user LLM providers.
Used by ModelSourceResolver, AdminProviderRegistry, and UserLLMProviderService.
"""

from typing import Dict, Optional, Tuple

# Source of model list and credentials
ProviderSource = str  # "admin" | "user"

# All supported provider types (user and admin)
PROVIDER_TYPES: Tuple[str, ...] = ("openai", "openrouter", "groq", "ollama", "vllm")

# Cloud providers with fixed base URLs (no user-supplied base_url)
CLOUD_PROVIDER_TYPES: Tuple[str, ...] = ("openai", "openrouter", "groq")

# Base URL per cloud provider (path may include /v1 for models endpoint)
CLOUD_BASE_URLS: Dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
}

# Providers that require OpenRouter-style headers (HTTP-Referer, X-Title)
OPENROUTER_HEADERS_PROVIDERS: Tuple[str, ...] = ("openrouter",)

# Providers that do not support reasoning-token extras (e.g. Groq chat-completions); do not send extra_body.reasoning
REASONING_UNSUPPORTED_PROVIDERS: Tuple[str, ...] = ("groq",)


def get_base_url_for_provider(
    provider_type: str, user_base_url: Optional[str] = None
) -> str:
    """Return base URL for a provider. Cloud providers use fixed URL; ollama/vllm use user_base_url."""
    if provider_type in CLOUD_BASE_URLS:
        return CLOUD_BASE_URLS[provider_type]
    return (user_base_url or "").rstrip("/")


def needs_openrouter_headers(provider_type: str) -> bool:
    """Whether to send HTTP-Referer and X-Title (OpenRouter-only)."""
    return provider_type in OPENROUTER_HEADERS_PROVIDERS
