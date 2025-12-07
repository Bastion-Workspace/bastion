"""
Orchestrator Utility Helpers - Roosevelt's Field Manual
Extracted helpers for conversation context, routing hints, and titles.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


def build_conversation_context_for_intent_classifier(ctx, state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        shared_memory = state.get("shared_memory", {})
        messages = state.get("messages", [])
        recent_messages = []
        for msg in messages[-6:]:
            if hasattr(msg, 'content') and hasattr(msg, 'type'):
                recent_messages.append({
                    "role": "assistant" if msg.type == "ai" else "user",
                    "content": msg.content[:200]
                })
        active_agent_context = analyze_active_agent_context(messages, shared_memory)
        last_assistant_message = None
        for msg in reversed(messages[:-1]):
            if hasattr(msg, 'type') and msg.type == "ai":
                last_assistant_message = msg.content
                break
        
        # Store last agent response in shared_memory for intent classifier
        if last_assistant_message and isinstance(shared_memory, dict):
            shared_memory["last_response"] = last_assistant_message
            logger.debug(f"📋 Stored last agent response ({len(last_assistant_message)} chars) for intent classifier context")
        
        location_clarification_requested = False
        if last_assistant_message:
            clarification_indicators = [
                "where would you like weather for",
                "please provide a city name",
                "provide a location",
                "which location would you like",
            ]
            location_clarification_requested = any(ind in last_assistant_message.lower() for ind in clarification_indicators)
        # ROOSEVELT'S COLLABORATION TRACKING: Check for recent collaboration suggestions
        collaboration_context = ""
        if active_agent_context.get("operation") == "collaboration_suggestion":
            collaboration_context = f"RECENT COLLABORATION: {active_agent_context.get('agent', 'unknown')} agent offered collaboration"
        
        context = {
            "shared_memory": shared_memory,
            "recent_messages": recent_messages[-4:],
            "location_clarification_requested": location_clarification_requested,
            "conversation_turn_count": len(messages),
            "last_assistant_response_preview": last_assistant_message[:100] if last_assistant_message else None,
            "active_agent_context": active_agent_context,
            "collaboration_context": collaboration_context,
        }
        return context
    except Exception as e:
        logger.error(f"❌ Failed to build conversation context: {e}")
        return {"shared_memory": state.get("shared_memory", {})}


def analyze_active_agent_context(messages: List, shared_memory: Dict[str, Any]) -> Dict[str, Any]:
    try:
        pending_operations = shared_memory.get("pending_operations", [])
        if pending_operations:
            latest_operation = pending_operations[-1]
            return {
                "agent": latest_operation.get("agent_type", "unknown"),
                "operation": latest_operation.get("operation", "unknown"),
                "context": "pending_operation",
                "confidence": 0.9,
            }
        pending_collaboration = shared_memory.get("pending_collaboration")
        if pending_collaboration:
            return {
                "agent": pending_collaboration.get("suggested_agent", "unknown"),
                "operation": "collaboration_suggestion",
                "context": "pending_collaboration",
                "confidence": 0.85,
            }
        
        # ROOSEVELT'S COLLABORATION INTELLIGENCE: Check for recent collaboration suggestions
        # Look for collaboration suggestions in recent AI messages
        if len(messages) >= 2:
            for msg in reversed(messages[-3:]):  # Check last 3 messages
                if hasattr(msg, 'type') and msg.type == "ai" and hasattr(msg, 'content'):
                    content_lower = msg.content.lower()
                    # Detect collaboration suggestion patterns
                    collaboration_indicators = [
                        "i could also", "would you like me to", "i can help", 
                        "shall i", "i can create", "i can generate",
                        "would you like", "i can provide", "i can format"
                    ]
                    if any(indicator in content_lower for indicator in collaboration_indicators):
                        # Extract suggested agent/capability
                        suggested_agent = "chat_agent"  # Default to chat agent
                        if "format" in content_lower or "table" in content_lower:
                            suggested_agent = "data_formatting_agent"
                        elif "weather" in content_lower:
                            suggested_agent = "weather_agent"
                        
                        return {
                            "agent": suggested_agent,
                            "operation": "collaboration_suggestion", 
                            "context": "recent_collaboration_offer",
                            "confidence": 0.8,
                        }
        
        # ROOSEVELT'S AGENT PATTERN ANALYSIS: Check for agent conversation patterns
        if len(messages) >= 2:
            recent_msgs = messages[-4:] if len(messages) >= 4 else messages
            agent_indicators = {
                "weather_agent": [
                    "weather", "temperature", "forecast", "conditions",
                    "where would you like weather", "location.*weather",
                    "provide.*city", "zip code.*weather",
                ],
                "mapping_agent": [
                    "directions", "navigate", "map", "address",
                    "find.*location", "where is", "routing",
                    "geographical", "coordinates",
                ],
                "research_agent": [
                    "research", "search", "investigate", "find information",
                    "look into", "tell me about",
                ],
            }
            agent_scores = {}
            for agent_type, indicators in agent_indicators.items():
                score = 0
                for msg in recent_msgs:
                    if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == "ai":
                        msg_content = msg.content.lower()
                        for indicator in indicators:
                            if indicator in msg_content:
                                score += 1
                agent_scores[agent_type] = score
            if agent_scores:
                active_agent = max(agent_scores, key=agent_scores.get)
                max_score = agent_scores[active_agent]
                if max_score > 0:
                    return {
                        "agent": active_agent,
                        "operation": "conversation_context",
                        "context": "message_pattern_analysis",
                        "confidence": min(0.8, max_score * 0.2),
                    }
        return {"agent": "none", "operation": "none", "context": "no_active_context", "confidence": 0.0}
    except Exception as e:
        logger.error(f"❌ Failed to analyze active agent context: {e}")
        return {"agent": "unknown", "operation": "error", "context": "analysis_failed", "confidence": 0.0}


def get_latest_user_message(state: Dict[str, Any]) -> str:
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, 'type') and msg.type == "human":
            content = msg.content
            if content.startswith("[SYSTEM TIME CONTEXT:"):
                lines = content.split("\n")
                if len(lines) > 2:
                    return "\n".join(lines[2:]).strip()
            return content
        elif hasattr(msg, '__class__') and 'Human' in msg.__class__.__name__:
            content = msg.content
            if content.startswith("[SYSTEM TIME CONTEXT:"):
                lines = content.split("\n")
                if len(lines) > 2:
                    return "\n".join(lines[2:]).strip()
            return content
    return ""


async def update_conversation_metadata(ctx, state: Dict[str, Any], user_message: str, is_complete: bool) -> None:
    """
    Update conversation metadata in state.
    
    NOTE: Title generation moved to LLM orchestrator. This function now only sets
    a simple fallback title. The LLM orchestrator will generate the proper title
    after the first agent response.
    """
    try:
        state["conversation_updated_at"] = datetime.now().isoformat()
        if not state.get("conversation_title") and user_message and len(user_message.strip()) > 0:
            # Simple fallback title - LLM orchestrator will replace with proper title
            title = user_message[:60] + "..." if len(user_message) > 60 else user_message
            state["conversation_title"] = title
            logger.debug(f"Set fallback conversation title: '{title}'")
        if user_message and len(user_message.strip()) > 0:
            topic = user_message[:50] + "..." if len(user_message) > 50 else user_message
            state["conversation_topic"] = topic.strip()
    except Exception as e:
        logger.warning(f"⚠️ Failed to update conversation metadata: {e}")


async def generate_conversation_title(user_message: str) -> str:
    """
    DEPRECATED: Title generation moved to LLM orchestrator.
    This function now just returns a simple fallback title.
    The LLM orchestrator will generate the proper title after the first agent response.
    """
    # Simple fallback - orchestrator will replace with proper title
    return user_message[:60] + "..." if len(user_message) > 60 else user_message


def normalize_thread_id(user_id: str, conversation_id: str) -> str:
    """Produce a namespaced thread_id ensuring per-user isolation.

    Format: "{user_id}:{conversation_id}". If conversation_id already appears namespaced, return as-is.
    """
    if not user_id or not conversation_id:
        raise ValueError("normalize_thread_id requires both user_id and conversation_id")
    if ":" in conversation_id and conversation_id.startswith(f"{user_id}:"):
        return conversation_id
    return f"{user_id}:{conversation_id}"


def validate_thread_id(user_id: str, thread_id: str) -> None:
    """Assert thread_id is correctly scoped to user_id."""
    if not thread_id.startswith(f"{user_id}:"):
        raise ValueError("Thread isolation error: thread_id not scoped to user_id")


