"""
Intent Classifier Utility Functions

Shared utility functions for context analysis and continuity detection.
"""

from typing import Optional


def calculate_agent_switches(messages: list) -> int:
    """
    Count how many times the agent changed in conversation history
    
    Args:
        messages: List of conversation messages with role/content
    
    Returns:
        Number of agent switches detected
    """
    if not messages or len(messages) < 2:
        return 0
    
    # Extract agent mentions from assistant messages
    # This is a simple heuristic - could be enhanced with actual agent tracking
    agent_mentions = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            # Look for agent indicators in response
            # This is a placeholder - actual implementation would track agent per message
            pass
    
    # For now, return 0 as we don't have explicit agent tracking per message
    # This can be enhanced when we have better message metadata
    return 0


def infer_action_from_agent(agent_name: str) -> Optional[str]:
    """
    Map agent type to typical action intent
    
    Args:
        agent_name: Name of the agent
    
    Returns:
        Typical action intent for this agent, or None
    """
    agent_action_map = {
        "fiction_editing_agent": "generation",
        "story_analysis_agent": "analysis",
        "content_analysis_agent": "analysis",
        "research_agent": "query",
        "chat_agent": "observation",
        "electronics_agent": "generation",
        "writing_assistant_agent": "generation",
        "series_editing_agent": "generation",
        "org_content_agent": "query",
        "org_capture_agent": "management",
        "website_crawler_agent": "management",
        "podcast_script_agent": "generation",
        "email_agent": "generation",
        "navigation_agent": "query",
    }
    return agent_action_map.get(agent_name)


def should_boost_continuity(
    primary_agent: Optional[str],
    last_agent: Optional[str],
    last_response: Optional[str],
    user_message: str
) -> bool:
    """
    Determine if strong continuity signal is present
    
    Args:
        primary_agent: Primary agent from previous turn
        last_agent: Last agent used
        last_response: Last agent response content
        user_message: Current user message
    
    Returns:
        True if continuity should be strongly weighted
    """
    # Strong continuity signals
    if not primary_agent:
        return False
    
    # Check if user message is clearly a follow-up
    follow_up_indicators = [
        "yes", "no", "ok", "okay", "sure", "please", "continue", "more",
        "that", "this", "it", "what about", "how about", "tell me more",
        "show me", "save", "update", "change", "modify", "edit that",
        "search for more", "find more", "get more", "do more", "go ahead",
        "detail", "details", "elaborate", "explain further"
    ]
    
    message_lower = user_message.lower().strip()
    
    # Short responses are likely continuations
    if len(message_lower.split()) <= 8:  # Increased from 5 to catch longer follow-ups
        if any(indicator in message_lower for indicator in follow_up_indicators):
            return True
    
    # Check for explicit continuation phrases
    continuation_phrases = [
        "continue", "go on", "more details", "more detail", "expand", "elaborate",
        "what's next", "and then", "also", "additionally", "tell me more",
        "need more", "give me more", "further detail", "further details"
    ]
    
    if any(phrase in message_lower for phrase in continuation_phrases):
        return True
    
    # If same agent and short message, likely continuation
    if primary_agent == last_agent and len(message_lower.split()) <= 5:
        return True
    
    return False
