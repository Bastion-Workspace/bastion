"""
OpenRouter Client Wrapper

Automatically adds reasoning support to all OpenRouter API calls.
This ensures reasoning is included by default without needing to add it manually to each call.

API OPTIONS:
1. Chat Completions API (current/default): /api/v1/chat/completions
   - Stable, works with OpenAI SDK
   - Reasoning via extra_body parameter
   - Standard request/response format
   
2. Responses API (beta): /api/v1/responses
   - Beta API, different format
   - Better for advanced reasoning features
   - Requires custom HTTP calls (not via OpenAI SDK)

We use Chat Completions API by default (stable, works with OpenAI SDK).
Responses API can be added later if needed for advanced reasoning features.

WHY A WRAPPER INSTEAD OF MODIFYING AsyncOpenAI?
- AsyncOpenAI is from the 'openai' package (third-party code we don't control)
- We can't modify it directly (it's in site-packages, gets overwritten on updates)
- We COULD monkey-patch it, but that's fragile and breaks on library updates
- Wrapper is clean, maintainable, and explicit about what we're doing
"""

import logging
import os
from typing import Optional, Dict, Any, List
from openai import AsyncOpenAI
from orchestrator.utils.llm_reasoning_utils import add_reasoning_to_extra_body

logger = logging.getLogger(__name__)

# Get settings from environment or use defaults
REASONING_ENABLED = os.getenv("REASONING_ENABLED", "true").lower() == "true"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


class OpenRouterClient:
    """
    Wrapper around AsyncOpenAI that automatically adds reasoning support to all chat completion calls.
    
    This ensures reasoning is included by default for all OpenRouter interactions,
    without needing to manually add extra_body to each call.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        """
        Initialize OpenRouter client wrapper.
        
        Args:
            api_key: OpenRouter API key
            base_url: OpenRouter API base URL (defaults to OpenRouter)
        """
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.api_key = api_key
        self.base_url = base_url
    
    @property
    def chat(self):
        """Return chat completions interface with automatic reasoning support"""
        return ChatCompletionsWrapper(self._client)
    
    # Delegate other attributes to the underlying client
    def __getattr__(self, name):
        """Delegate any other attributes to the underlying AsyncOpenAI client"""
        return getattr(self._client, name)


class ChatCompletionsWrapper:
    """
    Wrapper for chat.completions that automatically adds reasoning to create() calls.
    """
    
    def __init__(self, client: AsyncOpenAI):
        self._client = client
        self.completions = self
    
    async def create(self, *args, model: Optional[str] = None, extra_body: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Create chat completion with automatic reasoning support.
        
        Automatically adds reasoning configuration to extra_body if:
        - REASONING_ENABLED is True
        - Model is provided (for format detection)
        
        All other parameters are passed through to the underlying client.
        """
        # Automatically add reasoning if enabled
        if REASONING_ENABLED:
            # Get model from kwargs if not in args
            if model is None:
                model = kwargs.get('model')
            
            # Add reasoning to extra_body
            extra_body = add_reasoning_to_extra_body(extra_body=extra_body, model=model)
        
        # Call the underlying client
        return await self._client.chat.completions.create(
            *args,
            model=model,
            extra_body=extra_body,
            **kwargs
        )
    
    # Delegate streaming and other methods
    async def stream(self, *args, **kwargs):
        """Stream chat completions with automatic reasoning support"""
        model = kwargs.get('model')
        extra_body = kwargs.get('extra_body')
        
        if REASONING_ENABLED:
            extra_body = add_reasoning_to_extra_body(extra_body=extra_body, model=model)
            kwargs['extra_body'] = extra_body
        
        return await self._client.chat.completions.stream(*args, **kwargs)


def get_openrouter_client(api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1") -> OpenRouterClient:
    """
    Factory function to create an OpenRouter client with automatic reasoning support.
    
    Args:
        api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
        base_url: OpenRouter API base URL
        
    Returns:
        OpenRouterClient instance with automatic reasoning support
    """
    if api_key is None:
        api_key = OPENROUTER_API_KEY
    
    return OpenRouterClient(api_key=api_key, base_url=base_url)

