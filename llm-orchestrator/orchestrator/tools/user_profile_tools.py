"""
User Profile Tools - Get the current user's personal data (email, name, settings) via backend gRPC.
Zone 2: implementation in backend grpc_tool_service; this module is the orchestrator wrapper.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action
from orchestrator.utils.fact_utils import format_user_facts_for_prompt

logger = logging.getLogger(__name__)


class GetMyProfileInputs(BaseModel):
    """No required inputs; user_id is injected by the engine."""


class GetMyProfileOutputs(BaseModel):
    """Outputs for get_my_profile_tool."""
    email: str = Field(description="User email address")
    display_name: str = Field(description="Display name from account")
    username: str = Field(description="Login username")
    preferred_name: str = Field(description="Preferred name from user settings")
    timezone: str = Field(description="User timezone setting")
    zip_code: str = Field(description="User ZIP code setting")
    ai_context: str = Field(description="AI context from user settings")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def get_my_profile_tool(user_id: str = "system") -> Dict[str, Any]:
    """
    Get the current user's profile: email, display_name, username, preferred_name, timezone, zip_code, ai_context.
    Use this when building agents that need the user's contact info (e.g. email agent, contact watcher).
    """
    try:
        client = await get_backend_tool_client()
        result = await client.get_my_profile(user_id=user_id)
        if not result.get("success", False):
            err = result.get("error", "Unknown error")
            formatted = f"Could not load profile: {err}"
            return {
                "email": result.get("email", ""),
                "display_name": result.get("display_name", ""),
                "username": result.get("username", ""),
                "preferred_name": result.get("preferred_name", ""),
                "timezone": result.get("timezone", ""),
                "zip_code": result.get("zip_code", ""),
                "ai_context": result.get("ai_context", ""),
                "formatted": formatted,
            }
        parts = []
        name = result.get("preferred_name") or result.get("display_name") or result.get("username") or "User"
        parts.append(f"Name: {name}")
        if result.get("email"):
            parts.append(f"Email: {result['email']}")
        if result.get("username"):
            parts.append(f"Username: {result['username']}")
        if result.get("timezone"):
            parts.append(f"Timezone: {result['timezone']}")
        if result.get("zip_code"):
            parts.append(f"ZIP: {result['zip_code']}")
        if result.get("ai_context"):
            parts.append(f"AI context: {result['ai_context'][:200]}{'...' if len(result.get('ai_context', '')) > 200 else ''}")
        formatted = "\n".join(parts) if parts else "No profile data available."
        return {
            "email": result.get("email", ""),
            "display_name": result.get("display_name", ""),
            "username": result.get("username", ""),
            "preferred_name": result.get("preferred_name", ""),
            "timezone": result.get("timezone", ""),
            "zip_code": result.get("zip_code", ""),
            "ai_context": result.get("ai_context", ""),
            "formatted": formatted,
        }
    except Exception as e:
        logger.error("get_my_profile_tool error: %s", e)
        err = str(e)
        return {
            "email": "",
            "display_name": "",
            "username": "",
            "preferred_name": "",
            "timezone": "",
            "zip_code": "",
            "ai_context": "",
            "formatted": f"Error loading profile: {err}",
        }


register_action(
    name="get_my_profile",
    category="user",
    description="Get the current user's profile (email, display name, username, preferred name, timezone, ZIP, AI context). Use when building agents that need the user's contact info.",
    short_description="Get the current user's profile",
    inputs_model=GetMyProfileInputs,
    params_model=None,
    outputs_model=GetMyProfileOutputs,
    tool_function=get_my_profile_tool,
)


class GetUserFactsInputs(BaseModel):
    """Optional inputs; user_id is injected by the engine."""

    category: Optional[str] = Field(
        default=None,
        description="Filter by category: general, work, preferences, personal. Omit for all.",
    )


class GetUserFactsOutputs(BaseModel):
    """Outputs for get_user_facts_tool."""

    facts: List[Dict[str, Any]] = Field(description="List of fact dicts (fact_key, value, category)")
    count: int = Field(description="Number of facts returned")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def get_user_facts_tool(
    category: Optional[str] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Get the current user's remembered facts. Optionally filter by category.
    Use when you need to query what the system knows about the user (e.g. before suggesting actions).
    """
    try:
        client = await get_backend_tool_client()
        result = await client.get_user_facts(user_id=user_id)
        if not result.get("success", False):
            err = result.get("error", "Unknown error")
            return {
                "facts": [],
                "count": 0,
                "formatted": f"Could not load facts: {err}",
            }
        facts = result.get("facts") or []
        if category and category.strip():
            cat = category.strip().lower()
            facts = [f for f in facts if (f.get("category") or "general").lower() == cat]
        formatted = format_user_facts_for_prompt(facts) if facts else "No facts stored for this user."
        return {
            "facts": facts,
            "count": len(facts),
            "formatted": formatted,
        }
    except Exception as e:
        logger.error("get_user_facts_tool error: %s", e)
        return {
            "facts": [],
            "count": 0,
            "formatted": f"Error loading facts: {str(e)}",
        }


register_action(
    name="get_user_facts",
    category="user",
    description="Get the user's remembered facts, optionally filtered by category (general, work, preferences, personal). Use when you need to know what the system has stored about the user.",
    short_description="Get the user's remembered facts",
    inputs_model=GetUserFactsInputs,
    params_model=None,
    outputs_model=GetUserFactsOutputs,
    tool_function=get_user_facts_tool,
)


class SaveUserFactInputs(BaseModel):
    """Required inputs for save_user_fact_tool."""

    fact_key: str = Field(description="Short snake_case key, e.g. 'job_title', 'preferred_language'")
    value: str = Field(description="The fact value to store")


class SaveUserFactParams(BaseModel):
    """Optional params for save_user_fact_tool."""

    category: str = Field(default="general", description="Category for UI grouping: general, work, preferences, etc.")


class SaveUserFactOutputs(BaseModel):
    """Outputs for save_user_fact_tool."""

    success: bool = Field(description="Whether the fact was saved")
    fact_key: str = Field(description="The key that was saved")
    value: str = Field(description="The value that was saved")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def save_user_fact_tool(
    fact_key: str,
    value: str,
    category: str = "general",
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Save or update a fact about the user in their persistent fact store.
    Use when the user shares personal information they'd like remembered across conversations.
    Examples: job title, location, hobbies, dietary restrictions, programming language preference.
    """
    try:
        client = await get_backend_tool_client()
        result = await client.upsert_user_fact(
            user_id=user_id,
            fact_key=fact_key,
            value=value,
            category=category,
            source="agent",
        )
        success = result.get("success", False)
        err = result.get("error", "")
        formatted = (
            f"Saved fact: **{fact_key}** = {value}"
            if success
            else f"Failed to save fact: {err}"
        )
        return {
            "success": success,
            "fact_key": fact_key,
            "value": value,
            "formatted": formatted,
        }
    except Exception as e:
        logger.error("save_user_fact_tool error: %s", e)
        return {
            "success": False,
            "fact_key": fact_key,
            "value": value,
            "formatted": f"Error saving fact: {str(e)}",
        }


register_action(
    name="save_user_fact",
    category="user",
    description="Save or update a fact about the user in their persistent fact store. Use when the user shares information they want remembered (e.g. job title, location, dietary preference, hobbies).",
    short_description="Save or update a fact about the user",
    inputs_model=SaveUserFactInputs,
    params_model=SaveUserFactParams,
    outputs_model=SaveUserFactOutputs,
    tool_function=save_user_fact_tool,
    retriable=False,
)
