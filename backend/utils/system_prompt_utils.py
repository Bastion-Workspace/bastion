"""
Utility functions for adding date/time context to system prompts
"""
import logging
from datetime import datetime
from typing import Optional


def get_current_datetime_context(timezone_str: str = "UTC") -> str:
    """
    Get current date and time context for system prompts
    
    Args:
        timezone_str: Timezone string (e.g., "UTC", "America/New_York", "Europe/London")
    
    Returns:
        Formatted string with current date/time information
    """
    from datetime import timezone as dt_timezone
    import pytz
    
    try:
        # Get timezone object and current time
        if timezone_str.upper() == "UTC":
            current_time = datetime.now(dt_timezone.utc)
            timezone_name = "UTC"
        else:
            # For pytz timezones, use the recommended pytz approach
            tz = pytz.timezone(timezone_str)
            # Get naive UTC time, then localize to target timezone
            utc_naive = datetime.utcnow()
            utc_aware = pytz.utc.localize(utc_naive)
            current_time = utc_aware.astimezone(tz)
            timezone_name = current_time.strftime('%Z')  # Use strftime to get timezone abbreviation
        
        current_date = current_time.strftime("%A, %B %d, %Y")
        current_time_str = current_time.strftime("%I:%M %p")
        current_year = current_time.year
        
        return f"""**Current Context:**
- Today's date: {current_date}
- Current time: {current_time_str} ({timezone_name})
- Current year: {current_year}
- When users refer to "today", "yesterday", "this week", "this month", or "this year", use this date context to understand what they mean."""
    
    except Exception as e:
        # Fallback to UTC if timezone is invalid
        logger = logging.getLogger(__name__)
        logger.warning(f"Invalid timezone '{timezone_str}', falling back to UTC: {e}")
        
        current_time = datetime.now(dt_timezone.utc)
        current_date = current_time.strftime("%A, %B %d, %Y")
        current_time_str = current_time.strftime("%I:%M %p")
        current_year = current_time.year
        
        return f"""**Current Context:**
- Today's date: {current_date}
- Current time: {current_time_str} (UTC)
- Current year: {current_year}
- When users refer to "today", "yesterday", "this week", "this month", or "this year", use this date context to understand what they mean."""


def add_datetime_context_to_system_prompt(system_prompt: str, include_context: bool = True, timezone_str: str = "UTC") -> str:
    """
    Add current date/time context to a system prompt
    
    Args:
        system_prompt: The original system prompt
        include_context: Whether to include the date/time context section
        timezone_str: Timezone string (e.g., "UTC", "America/New_York", "Europe/London")
        
    Returns:
        System prompt with date/time context added
    """
    if not include_context:
        return system_prompt
    
    datetime_context = get_current_datetime_context(timezone_str)
    
    # Check if the prompt already has date context
    if "Current Context:" in system_prompt or "Today's date:" in system_prompt:
        return system_prompt
    
    # Add context after the first line (usually the role description)
    lines = system_prompt.split('\n')
    if len(lines) > 1:
        # Insert after the first line
        lines.insert(1, "")
        lines.insert(2, datetime_context)
        lines.insert(3, "")
    else:
        # Single line prompt, add context after
        lines.append("")
        lines.append(datetime_context)
    
    return '\n'.join(lines)


def create_system_prompt_with_context(base_prompt: str, additional_context: Optional[str] = None, timezone_str: str = "UTC") -> str:
    """
    Create a complete system prompt with date/time context and optional additional context
    
    Args:
        base_prompt: The base system prompt
        additional_context: Optional additional context to include
        timezone_str: Timezone string (e.g., "UTC", "America/New_York", "Europe/London")
        
    Returns:
        Complete system prompt with all context
    """
    datetime_context = get_current_datetime_context(timezone_str)
    
    # Build the complete prompt
    prompt_parts = [base_prompt]
    
    if additional_context:
        prompt_parts.append("")
        prompt_parts.append(additional_context)
    
    prompt_parts.append("")
    prompt_parts.append(datetime_context)
    
    return '\n'.join(prompt_parts)


async def get_user_timezone(user_id: Optional[str] = None) -> str:
    """
    Get user's timezone preference from settings service
    
    Args:
        user_id: Optional user ID. If None, returns UTC
        
    Returns:
        Timezone string (e.g., "UTC", "America/New_York", "Europe/London")
    """
    if not user_id:
        return "UTC"
    
    try:
        from services.settings_service import settings_service
        
        # Ensure settings service is initialized
        if not hasattr(settings_service, '_initialized') or not settings_service._initialized:
            await settings_service.initialize()
        
        user_timezone = await settings_service.get_user_timezone(user_id)
        return user_timezone if user_timezone else "UTC"
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to get timezone for user {user_id}, falling back to UTC: {e}")
        return "UTC"


async def get_current_datetime_context_for_user(user_id: Optional[str] = None) -> str:
    """
    Get current date/time context for system prompts using user's configured timezone
    
    Args:
        user_id: Optional user ID. If None, uses UTC
        
    Returns:
        Formatted string with current date/time information
    """
    timezone_str = await get_user_timezone(user_id)
    return get_current_datetime_context(timezone_str)


async def add_datetime_context_to_system_prompt_for_user(
    system_prompt: str, 
    user_id: Optional[str] = None,
    include_context: bool = True
) -> str:
    """
    Add current date/time context to a system prompt using user's configured timezone
    
    Args:
        system_prompt: The original system prompt
        user_id: Optional user ID. If None, uses UTC
        include_context: Whether to include the date/time context section
        
    Returns:
        System prompt with date/time context added
    """
    if not include_context:
        return system_prompt
    
    timezone_str = await get_user_timezone(user_id)
    return add_datetime_context_to_system_prompt(system_prompt, include_context, timezone_str)


async def create_system_prompt_with_context_for_user(
    base_prompt: str, 
    user_id: Optional[str] = None,
    additional_context: Optional[str] = None
) -> str:
    """
    Create a complete system prompt with date/time context using user's configured timezone
    
    Args:
        base_prompt: The base system prompt
        user_id: Optional user ID. If None, uses UTC
        additional_context: Optional additional context to include
        
    Returns:
        Complete system prompt with all context
    """
    timezone_str = await get_user_timezone(user_id)
    return create_system_prompt_with_context(base_prompt, additional_context, timezone_str) 