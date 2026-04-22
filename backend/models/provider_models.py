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


def normalize_openai_compatible_base_url(user_base_url: Optional[str]) -> str:
    """
    Ollama and vLLM use the OpenAI API under a /v1 prefix. ChatOpenAI posts to
    {base_url}/chat/completions, so the base must end with /v1 (same as api.openai.com).
    A value like http://ollama:11434 makes the client call .../chat/completions instead
    of .../v1/chat/completions and Ollama returns 404.
    """
    u = (user_base_url or "").strip().rstrip("/")
    if not u:
        return u
    if u.endswith("/v1"):
        return u
    return f"{u}/v1"


def get_base_url_for_provider(
    provider_type: str, user_base_url: Optional[str] = None
) -> str:
    """Return base URL for a provider. Cloud providers use fixed URL; ollama/vllm use user_base_url."""
    if provider_type in CLOUD_BASE_URLS:
        return CLOUD_BASE_URLS[provider_type]
    if provider_type in ("ollama", "vllm"):
        return normalize_openai_compatible_base_url(user_base_url)
    return (user_base_url or "").rstrip("/")


def needs_openrouter_headers(provider_type: str) -> bool:
    """Whether to send HTTP-Referer and X-Title (OpenRouter-only)."""
    return provider_type in OPENROUTER_HEADERS_PROVIDERS
