"""
LLM Reasoning Utilities

Utility functions for handling OpenRouter reasoning tokens and preserving reasoning details
across API calls, especially in iterative tool-calling workflows.
"""

import logging
import os
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Get settings from environment or use defaults
REASONING_ENABLED = os.getenv("REASONING_ENABLED", "true").lower() == "true"
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "medium")


def get_reasoning_config(effort: Optional[str] = None, 
                         model: Optional[str] = None,
                         max_tokens: Optional[int] = None,
                         exclude: bool = False,
                         use_enabled_flag: bool = False) -> Optional[Dict[str, Any]]:
    """
    Get reasoning configuration for OpenRouter API calls.
    
    Uses the unified OpenRouter format that works for all reasoning-capable models
    (OpenAI o-series, Grok, Anthropic Claude, Gemini, etc.).
    
    Reference: https://openrouter.ai/docs/use-cases/reasoning-tokens#controlling-reasoning-tokens

    Args:
        effort: Reasoning effort level. One of "high", "medium", "low", "minimal", "none".
                If None, uses REASONING_EFFORT env var.
        model: Model name (for logging/debugging, not used for format selection).
        max_tokens: Optional max tokens for reasoning (for Anthropic/Gemini models).
                   If provided, takes precedence over effort.
        exclude: If True, model uses reasoning internally but excludes it from response.
        use_enabled_flag: If True, use {"enabled": true} format (enables at "medium" effort).
                         Simpler alternative to specifying effort explicitly.

    Returns:
        Reasoning config dict for extra_body parameter, or None if disabled.
        Format: {"reasoning": {"effort": "..."}} or {"reasoning": {"enabled": true}}
    """
    if not REASONING_ENABLED:
        return None

    # Build reasoning config object
    reasoning_config: Dict[str, Any] = {}
    
    # Use max_tokens if provided (for Anthropic/Gemini models)
    if max_tokens is not None:
        reasoning_config["max_tokens"] = max_tokens
    elif use_enabled_flag:
        # Use simplified "enabled: true" format (defaults to "medium" effort)
        reasoning_config["enabled"] = True
    else:
        # Use effort level (for OpenAI o-series, Grok, etc.)
        final_effort = effort or REASONING_EFFORT
        valid_efforts = {"high", "medium", "low", "minimal", "none"}
        
        if final_effort not in valid_efforts:
            logger.warning(f"⚠️ Invalid reasoning effort '{final_effort}', using 'medium'")
            final_effort = "medium"
        
        # Use simpler "enabled: true" format when effort is "medium" (the default)
        # This matches OpenRouter's recommended approach for default configuration
        if final_effort == "medium":
            reasoning_config["enabled"] = True
        else:
            reasoning_config["effort"] = final_effort
    
    # Add exclude flag if needed
    if exclude:
        reasoning_config["exclude"] = True
    
    return {
        "reasoning": reasoning_config
    }


def extract_reasoning_details(response_message) -> Optional[List[Dict[str, Any]]]:
    """
    Extract reasoning_details from an OpenAI-style response message.

    Args:
        response_message: The message object from OpenAI API response

    Returns:
        reasoning_details if available, None otherwise
    """
    try:
        # Use getattr to safely access reasoning_details attribute
        return getattr(response_message, 'reasoning_details', None)
    except AttributeError:
        return None


def build_assistant_message_with_reasoning(response_message) -> Dict[str, Any]:
    """
    Build an assistant message dict that preserves reasoning_details for future API calls.

    Args:
        response_message: The message object from OpenAI API response

    Returns:
        Dict containing role, content, tool_calls, and reasoning_details
    """
    assistant_message = {
        "role": "assistant",
        "content": getattr(response_message, 'content', None),
        "tool_calls": getattr(response_message, 'tool_calls', None),
    }

    # Preserve reasoning_details if available
    reasoning_details = extract_reasoning_details(response_message)
    if reasoning_details:
        assistant_message["reasoning_details"] = reasoning_details
        logger.debug(f"🧠 Preserved reasoning_details with {len(reasoning_details)} items")

    return assistant_message


def add_reasoning_to_extra_body(extra_body: Optional[Dict[str, Any]] = None,
                                effort: Optional[str] = None,
                                model: Optional[str] = None,
                                max_tokens: Optional[int] = None,
                                exclude: bool = False,
                                use_enabled_flag: bool = False) -> Optional[Dict[str, Any]]:
    """
    Add reasoning configuration to extra_body parameter for OpenRouter API calls.
    
    Uses the unified OpenRouter format that works for all reasoning-capable models.
    OpenRouter will handle unsupported models gracefully (ignoring the parameter).
    
    Supports two formats:
    1. Explicit effort: {"reasoning": {"effort": "medium"}}
    2. Enabled flag: {"reasoning": {"enabled": true}} (simpler, defaults to "medium")

    Args:
        extra_body: Existing extra_body dict, or None
        effort: Reasoning effort override ("high", "medium", "low", "minimal", "none")
        model: Model name (for logging/debugging purposes)
        max_tokens: Optional max tokens for reasoning (for Anthropic/Gemini models)
        exclude: If True, exclude reasoning tokens from response
        use_enabled_flag: If True, use {"enabled": true} format instead of explicit effort

    Returns:
        Updated extra_body dict with reasoning config, or original if reasoning disabled
    """
    reasoning_config = get_reasoning_config(effort, model, max_tokens, exclude, use_enabled_flag)

    if not reasoning_config:
        return extra_body

    if extra_body is None:
        return reasoning_config

    # Merge reasoning config into existing extra_body
    # If extra_body already has a "reasoning" key, merge the configs
    if "reasoning" in extra_body and isinstance(extra_body["reasoning"], dict):
        extra_body["reasoning"].update(reasoning_config["reasoning"])
    else:
        extra_body.update(reasoning_config)
    
    return extra_body


def log_reasoning_info(response_message, context: str = "") -> None:
    """
    Log reasoning information for debugging and monitoring.

    Args:
        response_message: The message object from OpenAI API response
        context: Additional context for logging
    """
    reasoning_details = extract_reasoning_details(response_message)
    if reasoning_details:
        total_items = len(reasoning_details)
        logger.info(f"🧠 {context} Reasoning details: {total_items} items")

        # Log summary of reasoning types
        types_count = {}
        for item in reasoning_details:
            item_type = item.get("type", "unknown")
            types_count[item_type] = types_count.get(item_type, 0) + 1

        type_summary = ", ".join(f"{t}: {c}" for t, c in types_count.items())
        logger.debug(f"🧠 Reasoning types: {type_summary}")
    else:
        logger.debug(f"🧠 {context} No reasoning details available")

