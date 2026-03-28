"""
Settings API - Handles all settings-related endpoints
"""

import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

from services.settings_service import settings_service
from services.prompt_service import UserPromptSettings, PoliticalBias, PersonaStyle, prompt_service
from utils.auth_middleware import get_current_user
from models.api_models import (
    SettingsResponse, SettingUpdateRequest, BulkSettingsUpdateRequest, 
    SettingUpdateResponse, AuthenticatedUserResponse
)

logger = logging.getLogger(__name__)


def _get_bias_label(bias_value: str) -> str:
    """Get professional label for bias values"""
    bias_labels = {
        "neutral": "Neutral",
        "mildly_left": "Mild Left",
        "mildly_right": "Mild Right", 
        "extreme_left": "Extreme Left",
        "extreme_right": "Extreme Right"
    }
    return bias_labels.get(bias_value, bias_value.replace("_", " ").title())

router = APIRouter(tags=["Settings"])


# Pydantic models for settings validation
# IntentClassificationModelRequest removed - no longer used (deprecated endpoints removed)

class TimezoneRequest(BaseModel):
    timezone: str


class ZipCodeRequest(BaseModel):
    zip_code: str


class TimeFormatRequest(BaseModel):
    time_format: str


class PreferredNameRequest(BaseModel):
    preferred_name: str


class AiContextRequest(BaseModel):
    ai_context: str


class PhoneNumberRequest(BaseModel):
    phone_number: str


class BirthdayRequest(BaseModel):
    birthday: str


class UserFactRequest(BaseModel):
    fact_key: str
    value: str
    category: str = "general"
    expires_at: Optional[str] = None


class FactsPreferencesResponse(BaseModel):
    facts_inject_enabled: bool = True
    facts_write_enabled: bool = True


class EpisodesPreferencesResponse(BaseModel):
    episodes_inject_enabled: bool = True


class ResolvePendingFactRequest(BaseModel):
    action: str = "accept"


class VisionFeaturesRequest(BaseModel):
    enabled: bool


class PromptSettingsRequest(BaseModel):
    """Request model for updating prompt settings"""
    ai_name: str = Field("Alex", description="Name for the AI assistant")
    political_bias: PoliticalBias = PoliticalBias.NEUTRAL
    persona_style: PersonaStyle = PersonaStyle.PROFESSIONAL


class PromptSettingsResponse(BaseModel):
    """Response model for prompt settings"""
    ai_name: str
    political_bias: PoliticalBias
    persona_style: PersonaStyle
    available_biases: list[str]
    available_personas: list[str]


@router.get("/api/settings", response_model=SettingsResponse)
async def get_all_settings():
    """Get all settings grouped by category"""
    try:
        logger.info("⚙️ Getting all settings")
        settings = await settings_service.get_all_settings()
        return SettingsResponse(settings=settings)
    except Exception as e:
        logger.error(f"❌ Failed to get all settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))





# Deprecated endpoints removed:
# - PUT /intent_classification_model - Use /api/models/classification instead
# - GET /intent_classification_model - Use /api/models/classification instead
# - POST /test/intent-classification-model - Test endpoint removed


@router.put("/api/settings/{key}", response_model=SettingUpdateResponse)
async def update_setting(
    key: str, 
    request: SettingUpdateRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Update a single setting"""
    try:
        logger.info(f"⚙️ Updating setting: {key}")
        
        # Determine value type if not provided
        value_type = "string"
        if isinstance(request.value, bool):
            value_type = "boolean"
        elif isinstance(request.value, int):
            value_type = "integer"
        elif isinstance(request.value, float):
            value_type = "float"
        elif isinstance(request.value, (dict, list)):
            value_type = "json"
        
        success = await settings_service.set_setting(
            key, 
            request.value, 
            value_type, 
            request.description, 
            request.category
        )
        
        if success:
            logger.info(f"✅ Setting '{key}' updated successfully")
            return SettingUpdateResponse(
                success=True, 
                message=f"Setting '{key}' updated successfully"
            )
        else:
            logger.error(f"❌ Failed to update setting '{key}'")
            return SettingUpdateResponse(
                success=False, 
                message=f"Failed to update setting '{key}'"
            )
        
    except Exception as e:
        logger.error(f"❌ Failed to update setting '{key}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/settings/bulk", response_model=SettingUpdateResponse)
async def bulk_update_settings(
    request: BulkSettingsUpdateRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Update multiple settings at once"""
    try:
        logger.info(f"⚙️ Bulk updating {len(request.settings)} settings")
        
        results = await settings_service.bulk_update_settings(request.settings)
        
        success_count = sum(1 for success in results.values() if success)
        failed_count = len(results) - success_count
        
        logger.info(f"✅ Bulk update completed: {success_count} successful, {failed_count} failed")
        
        return SettingUpdateResponse(
            success=failed_count == 0,
            message=f"Updated {success_count} settings successfully" + 
                   (f", {failed_count} failed" if failed_count > 0 else ""),
            updated_settings=results
        )
        
    except Exception as e:
        logger.error(f"❌ Bulk settings update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/settings/{key}")
async def delete_setting(
    key: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete a setting"""
    try:
        logger.info(f"⚙️ Deleting setting: {key}")
        
        success = await settings_service.delete_setting(key)
        
        if success:
            logger.info(f"✅ Setting '{key}' deleted successfully")
            return {
                "success": True,
                "message": f"Setting '{key}' deleted successfully"
            }
        else:
            logger.error(f"❌ Failed to delete setting '{key}'")
            return {
                "success": False,
                "message": f"Failed to delete setting '{key}'"
            }
        
    except Exception as e:
        logger.error(f"❌ Failed to delete setting '{key}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/settings/user/timezone")
async def get_user_timezone(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get current user's timezone preference"""
    try:
        logger.info(f"🌍 Getting timezone for user: {current_user.username}")
        timezone = await settings_service.get_user_timezone(current_user.user_id)
        return {
            "success": True,
            "timezone": timezone,
            "user_id": current_user.user_id
        }
    except Exception as e:
        logger.error(f"❌ Failed to get timezone for user {current_user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/settings/user/timezone")
async def set_user_timezone(
    request: TimezoneRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Set current user's timezone preference"""
    try:
        logger.info(f"🌍 Setting timezone for user {current_user.username} to: {request.timezone}")
        success = await settings_service.set_user_timezone(current_user.user_id, request.timezone)
        
        if success:
            logger.info(f"✅ Timezone updated successfully for user {current_user.username}")
            return {
                "success": True,
                "message": f"Timezone updated to {request.timezone}",
                "timezone": request.timezone,
                "user_id": current_user.user_id
            }
        else:
            logger.error(f"❌ Failed to update timezone for user {current_user.username}")
            return {
                "success": False,
                "message": "Failed to update timezone"
            }
        
    except Exception as e:
        logger.error(f"❌ Failed to set timezone for user {current_user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/settings/user/zip-code")
async def get_user_zip_code(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get current user's zip code preference"""
    try:
        logger.info(f"📍 Getting zip code for user: {current_user.username}")
        zip_code = await settings_service.get_user_zip_code(current_user.user_id)
        return {
            "success": True,
            "zip_code": zip_code,
            "user_id": current_user.user_id
        }
    except Exception as e:
        logger.error(f"❌ Failed to get zip code for user {current_user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/settings/user/zip-code")
async def set_user_zip_code(
    request: ZipCodeRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Set current user's zip code preference"""
    try:
        # Validate zip code format (5 digits for US)
        if not request.zip_code or not re.match(r'^\d{5}$', request.zip_code):
            raise HTTPException(status_code=400, detail="Zip code must be 5 digits")
        
        logger.info(f"📍 Setting zip code for user {current_user.username} to: {request.zip_code}")
        success = await settings_service.set_user_zip_code(current_user.user_id, request.zip_code)
        
        if success:
            logger.info(f"✅ Zip code updated successfully for user {current_user.username}")
            return {
                "success": True,
                "message": f"Zip code updated to {request.zip_code}",
                "zip_code": request.zip_code,
                "user_id": current_user.user_id
            }
        else:
            logger.error(f"❌ Failed to update zip code for user {current_user.username}")
            return {
                "success": False,
                "message": "Failed to update zip code"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to set zip code for user {current_user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/settings/user/time-format")
async def get_user_time_format(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get current user's time format preference"""
    try:
        logger.info(f"🕐 Getting time format for user: {current_user.username}")
        time_format = await settings_service.get_user_time_format(current_user.user_id)
        return {
            "success": True,
            "time_format": time_format,
            "user_id": current_user.user_id
        }
    except Exception as e:
        logger.error(f"❌ Failed to get time format for user {current_user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/settings/user/time-format")
async def set_user_time_format(
    request: TimeFormatRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Set current user's time format preference"""
    try:
        # Validate time format
        if request.time_format not in ["12h", "24h"]:
            raise HTTPException(status_code=400, detail="Time format must be '12h' or '24h'")
        
        logger.info(f"🕐 Setting time format for user {current_user.username} to: {request.time_format}")
        success = await settings_service.set_user_time_format(current_user.user_id, request.time_format)
        
        if success:
            logger.info(f"✅ Time format updated successfully for user {current_user.username}")
            return {
                "success": True,
                "message": f"Time format updated to {request.time_format}",
                "time_format": request.time_format,
                "user_id": current_user.user_id
            }
        else:
            logger.error(f"❌ Failed to update time format for user {current_user.username}")
            return {
                "success": False,
                "message": "Failed to update time format"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to set time format for user {current_user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/settings/prompt", response_model=PromptSettingsResponse)
async def get_prompt_settings(current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Get current user's prompt settings (from default persona when available)."""
    try:
        persona = await settings_service.get_default_persona(current_user.user_id)
        if persona:
            return PromptSettingsResponse(
                ai_name=persona.get("ai_name") or "Alex",
                political_bias=PoliticalBias(persona.get("political_bias") or "neutral"),
                persona_style=PersonaStyle.PROFESSIONAL,
                available_biases=[bias.value for bias in PoliticalBias],
                available_personas=[p.value for p in PersonaStyle]
            )
        user_settings = await settings_service.get_user_prompt_settings(current_user.user_id)
        if user_settings is None:
            user_settings = UserPromptSettings()
        return PromptSettingsResponse(
            ai_name=user_settings.ai_name,
            political_bias=user_settings.political_bias,
            persona_style=user_settings.persona_style,
            available_biases=[bias.value for bias in PoliticalBias],
            available_personas=[persona.value for persona in PersonaStyle]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve prompt settings: {str(e)}"
        )


@router.post("/api/settings/prompt", response_model=PromptSettingsResponse)
async def update_prompt_settings(
    settings: PromptSettingsRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Update user's prompt settings"""
    try:
        # Create UserPromptSettings object
        user_settings = UserPromptSettings(
            ai_name=settings.ai_name,
            political_bias=settings.political_bias,
            persona_style=settings.persona_style
        )
        
        # Validate settings (this will raise ValueError if AI name requirement not met)
        try:
            # Test prompt assembly to trigger validation
            from services.prompt_service import AgentMode
            test_prompt = prompt_service.assemble_prompt(
                agent_mode=AgentMode.CHAT,
                tools_description="test",
                user_settings=user_settings
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        # Save settings to database
        await settings_service.save_user_prompt_settings(current_user.user_id, user_settings)
        
        return PromptSettingsResponse(
            ai_name=user_settings.ai_name,
            political_bias=user_settings.political_bias,
            persona_style=user_settings.persona_style,
            available_biases=[bias.value for bias in PoliticalBias],
            available_personas=[persona.value for persona in PersonaStyle]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update prompt settings: {str(e)}"
        )


@router.get("/api/settings/prompt/options")
async def get_prompt_options():
    """Get available prompt options for the frontend"""
    return {
        "political_biases": [
            {"value": bias.value, "label": _get_bias_label(bias.value)} 
            for bias in PoliticalBias
        ],
        "persona_styles": [
            {"value": persona.value, "label": persona.value.replace("_", " ").title()} 
            for persona in PersonaStyle
        ],
        "historical_figures": [
            {"value": persona.value, "label": persona.value.replace("_", " ").title()} 
            for persona in PersonaStyle if persona.value in [
                "amelia_earhart", "theodore_roosevelt", "winston_churchill", "mr_spock", 
                "abraham_lincoln", "napoleon_bonaparte", "isaac_newton", "george_washington",
                "mark_twain", "edgar_allan_poe", "jane_austen", "albert_einstein", "nikola_tesla"
            ]
        ],
        "default_settings": {
            "ai_name": "Alex",
            "political_bias": PoliticalBias.NEUTRAL.value,
            "persona_style": PersonaStyle.PROFESSIONAL.value
        }
    }


# -------------------------------------------------------------------------
# Personas API (built-in + custom; default persona in user settings)
# -------------------------------------------------------------------------

class PersonaCreateRequest(BaseModel):
    """Request model for creating a custom persona"""
    name: str = Field(..., min_length=1, max_length=255)
    ai_name: Optional[str] = Field("Alex", max_length=100)
    style_instruction: Optional[str] = Field("", description="Free-form style instructions")
    political_bias: str = Field("neutral", max_length=50)
    description: Optional[str] = Field("", max_length=2000)


class PersonaUpdateRequest(BaseModel):
    """Request model for updating a custom persona"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    ai_name: Optional[str] = Field(None, max_length=100)
    style_instruction: Optional[str] = None
    political_bias: Optional[str] = Field(None, max_length=50)
    description: Optional[str] = Field(None, max_length=2000)


class DefaultPersonaRequest(BaseModel):
    """Request model for setting default persona"""
    persona_id: Optional[str] = Field(None, description="UUID of persona, or null to clear default")


class DefaultChatAgentProfileRequest(BaseModel):
    """Set which non-built-in agent profile is used for chat when there is no @mention / sticky profile."""
    agent_profile_id: Optional[str] = Field(
        None,
        description="UUID of a non-built-in active agent profile, or null to use factory built-in",
    )


@router.get("/api/personas")
async def list_personas(
    include_builtin: bool = True,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """List all personas available to the user (built-in + custom)."""
    try:
        personas = await settings_service.get_personas(current_user.user_id, include_builtin=include_builtin)
        return {"personas": personas}
    except Exception as e:
        logger.error(f"Failed to list personas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/personas/{persona_id}")
async def get_persona(
    persona_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get a single persona by id."""
    try:
        persona = await settings_service.get_persona_by_id(persona_id, current_user.user_id)
        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found")
        return persona
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get persona {persona_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/personas")
async def create_persona(
    request: PersonaCreateRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Create a custom persona."""
    try:
        persona = await settings_service.create_persona(
            user_id=current_user.user_id,
            name=request.name,
            ai_name=request.ai_name or "Alex",
            style_instruction=request.style_instruction or "",
            political_bias=request.political_bias or "neutral",
            description=request.description or "",
        )
        return persona
    except Exception as e:
        logger.error(f"Failed to create persona: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/personas/{persona_id}")
async def update_persona(
    persona_id: str,
    request: PersonaUpdateRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Update a custom persona (built-in personas cannot be edited)."""
    try:
        persona = await settings_service.update_persona(
            persona_id=persona_id,
            user_id=current_user.user_id,
            name=request.name,
            ai_name=request.ai_name,
            style_instruction=request.style_instruction,
            political_bias=request.political_bias,
            description=request.description,
        )
        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found or cannot be edited")
        return persona
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update persona {persona_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/personas/{persona_id}")
async def delete_persona(
    persona_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete a custom persona (built-in personas cannot be deleted)."""
    try:
        success = await settings_service.delete_persona(persona_id, current_user.user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Persona not found or cannot be deleted")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete persona {persona_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/settings/default-persona")
async def get_default_persona(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get the user's default persona."""
    try:
        persona = await settings_service.get_default_persona(current_user.user_id)
        return {"persona": persona}
    except Exception as e:
        logger.error(f"Failed to get default persona: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/settings/default-persona")
async def set_default_persona(
    request: DefaultPersonaRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Set the user's default persona (pass null to clear)."""
    try:
        success = await settings_service.set_default_persona(
            current_user.user_id,
            request.persona_id,
        )
        if not success and request.persona_id is not None:
            raise HTTPException(status_code=404, detail="Persona not found")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set default persona: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/settings/default-chat-agent-profile")
async def get_default_chat_agent_profile(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Non-built-in agent profile used for chat without @mention; null means factory built-in."""
    try:
        detail = await settings_service.get_default_chat_agent_profile_detail(current_user.user_id)
        return detail
    except Exception as e:
        logger.error("Failed to get default chat agent profile: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/settings/default-chat-agent-profile")
async def set_default_chat_agent_profile(
    request: DefaultChatAgentProfileRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Set or clear default chat agent profile (must be non-built-in and active when set)."""
    try:
        success = await settings_service.set_default_chat_agent_profile_id(
            current_user.user_id,
            request.agent_profile_id,
        )
        if not success and request.agent_profile_id is not None:
            raise HTTPException(
                status_code=404,
                detail="Agent profile not found, inactive, or built-in profiles cannot be set as default",
            )
        out = await settings_service.get_default_chat_agent_profile_detail(current_user.user_id)
        return {"success": True, **out}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to set default chat agent profile: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/settings/user/preferred-name")
async def get_user_preferred_name(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get current user's preferred name"""
    try:
        logger.info(f"👤 Getting preferred name for user: {current_user.username}")
        preferred_name = await settings_service.get_user_preferred_name(current_user.user_id)
        return {
            "success": True,
            "preferred_name": preferred_name or "",
            "user_id": current_user.user_id
        }
    except Exception as e:
        logger.error(f"❌ Failed to get preferred name for user {current_user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/settings/user/preferred-name")
async def set_user_preferred_name(
    request: PreferredNameRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Set current user's preferred name"""
    try:
        logger.info(f"👤 Setting preferred name for user {current_user.username} to: {request.preferred_name}")
        success = await settings_service.set_user_preferred_name(current_user.user_id, request.preferred_name)
        
        if success:
            logger.info(f"✅ Preferred name updated successfully for user {current_user.username}")
            return {
                "success": True,
                "message": "Preferred name updated successfully",
                "preferred_name": request.preferred_name,
                "user_id": current_user.user_id
            }
        else:
            logger.error(f"❌ Failed to update preferred name for user {current_user.username}")
            return {
                "success": False,
                "message": "Failed to update preferred name"
            }
        
    except Exception as e:
        logger.error(f"❌ Failed to set preferred name for user {current_user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/settings/user/phone-number")
async def get_user_phone_number(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get current user's phone number."""
    try:
        phone_number = await settings_service.get_user_phone_number(current_user.user_id)
        return {
            "success": True,
            "phone_number": phone_number or "",
            "user_id": current_user.user_id
        }
    except Exception as e:
        logger.error(f"❌ Failed to get phone number for user {current_user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/settings/user/phone-number")
async def set_user_phone_number(
    request: PhoneNumberRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Set current user's phone number."""
    try:
        phone_number = (request.phone_number or "").strip()
        if phone_number and not re.match(r"^[\d+\-().\s]{7,25}$", phone_number):
            raise HTTPException(status_code=400, detail="Invalid phone number format")

        success = await settings_service.set_user_phone_number(current_user.user_id, phone_number)
        if success:
            return {
                "success": True,
                "message": "Phone number updated successfully",
                "phone_number": phone_number,
                "user_id": current_user.user_id
            }
        return {
            "success": False,
            "message": "Failed to update phone number"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to set phone number for user {current_user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/settings/user/birthday")
async def get_user_birthday(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get current user's birthday."""
    try:
        birthday = await settings_service.get_user_birthday(current_user.user_id)
        return {
            "success": True,
            "birthday": birthday or "",
            "user_id": current_user.user_id
        }
    except Exception as e:
        logger.error(f"❌ Failed to get birthday for user {current_user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/settings/user/birthday")
async def set_user_birthday(
    request: BirthdayRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Set current user's birthday (YYYY-MM-DD)."""
    try:
        birthday = (request.birthday or "").strip()
        if birthday:
            datetime.strptime(birthday, "%Y-%m-%d")
        success = await settings_service.set_user_birthday(current_user.user_id, birthday)
        if success:
            return {
                "success": True,
                "message": "Birthday updated successfully",
                "birthday": birthday,
                "user_id": current_user.user_id
            }
        return {
            "success": False,
            "message": "Failed to update birthday"
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Birthday must be in YYYY-MM-DD format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to set birthday for user {current_user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/settings/user/ai-context")
async def get_user_ai_context(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get current user's AI context"""
    try:
        logger.info(f"🤖 Getting AI context for user: {current_user.username}")
        ai_context = await settings_service.get_user_ai_context(current_user.user_id)
        return {
            "success": True,
            "ai_context": ai_context or "",
            "user_id": current_user.user_id
        }
    except Exception as e:
        logger.error(f"❌ Failed to get AI context for user {current_user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/settings/user/ai-context")
async def set_user_ai_context(
    request: AiContextRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Set current user's AI context (max 500 characters)"""
    try:
        # Validate length
        if len(request.ai_context) > 500:
            raise HTTPException(status_code=400, detail="AI context must be 500 characters or less")
        
        logger.info(f"🤖 Setting AI context for user {current_user.username}")
        success = await settings_service.set_user_ai_context(current_user.user_id, request.ai_context)
        
        if success:
            logger.info(f"✅ AI context updated successfully for user {current_user.username}")
            return {
                "success": True,
                "message": "AI context updated successfully",
                "ai_context": request.ai_context,
                "user_id": current_user.user_id
            }
        else:
            logger.error(f"❌ Failed to update AI context for user {current_user.username}")
            return {
                "success": False,
                "message": "Failed to update AI context"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to set AI context for user {current_user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/settings/user/facts")
async def get_user_facts(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get all facts for the current user"""
    try:
        logger.info(f"Getting user facts for user: {current_user.username}")
        facts = await settings_service.get_user_facts(current_user.user_id)
        return {
            "success": True,
            "facts": facts,
            "user_id": current_user.user_id
        }
    except Exception as e:
        logger.error(f"Failed to get user facts for user {current_user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/settings/user/facts")
async def add_user_fact(
    request: UserFactRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Create or update a single user fact"""
    try:
        if not request.fact_key or not request.fact_key.strip():
            raise HTTPException(status_code=400, detail="fact_key is required")
        logger.info(f"Upserting fact {request.fact_key} for user {current_user.username}")
        result = await settings_service.upsert_user_fact(
            current_user.user_id,
            request.fact_key.strip(),
            request.value or "",
            request.category or "general",
            source="user_manual",
            expires_at=request.expires_at,
        )
        if result.get("success"):
            return {
                "success": True,
                "message": "Fact saved",
                "fact_key": request.fact_key.strip(),
                "value": request.value,
                "category": request.category or "general",
                "user_id": current_user.user_id
            }
        if result.get("status") == "pending_review":
            return {
                "success": False,
                "message": "Fact is set by you; agent update queued for review",
                "fact_key": result.get("fact_key"),
                "current_value": result.get("current_value"),
            }
        return {
            "success": False,
            "message": result.get("error", "Failed to save fact")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add user fact for user {current_user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/settings/user/facts/{fact_key}")
async def delete_user_fact(
    fact_key: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete a single user fact by key"""
    try:
        logger.info(f"Deleting fact {fact_key} for user {current_user.username}")
        success = await settings_service.delete_user_fact(current_user.user_id, fact_key)
        if success:
            return {
                "success": True,
                "message": "Fact deleted",
                "fact_key": fact_key,
                "user_id": current_user.user_id
            }
        return {
            "success": False,
            "message": "Failed to delete fact"
        }
    except Exception as e:
        logger.error(f"Failed to delete user fact for user {current_user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/settings/user/facts-preferences")
async def get_facts_preferences(current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    try:
        inject = await settings_service.get_facts_inject_enabled(current_user.user_id)
        write = await settings_service.get_facts_write_enabled(current_user.user_id)
        return {"facts_inject_enabled": inject, "facts_write_enabled": write}
    except Exception as e:
        logger.error("Failed to get facts preferences: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/settings/user/facts-preferences")
async def set_facts_preferences(
    request: FactsPreferencesResponse,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    try:
        from services.user_settings_kv_service import set_user_setting
        await set_user_setting(
            current_user.user_id, "facts_inject_enabled",
            "true" if request.facts_inject_enabled else "false", "boolean"
        )
        await set_user_setting(
            current_user.user_id, "facts_write_enabled",
            "true" if request.facts_write_enabled else "false", "boolean"
        )
        return {"facts_inject_enabled": request.facts_inject_enabled, "facts_write_enabled": request.facts_write_enabled}
    except Exception as e:
        logger.error("Failed to set facts preferences: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/settings/user/episodes")
async def get_user_episodes(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
    limit: int = 50,
    days: Optional[int] = 30,
):
    """Get recent activity episodes for the current user"""
    try:
        from services.episode_service import episode_service
        episodes = await episode_service.get_user_episodes(
            current_user.user_id, limit=limit, days=days
        )
        return {
            "success": True,
            "episodes": episodes,
            "user_id": current_user.user_id,
        }
    except Exception as e:
        logger.error("Failed to get user episodes: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/settings/user/episodes/{episode_id}")
async def delete_user_episode(
    episode_id: int,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Delete a single episode by id"""
    try:
        from services.episode_service import episode_service
        success = await episode_service.delete_episode(current_user.user_id, episode_id)
        if success:
            return {
                "success": True,
                "message": "Episode deleted",
                "episode_id": episode_id,
                "user_id": current_user.user_id,
            }
        return {"success": False, "message": "Failed to delete episode"}
    except Exception as e:
        logger.error("Failed to delete episode: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/settings/user/episodes")
async def delete_all_user_episodes(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Delete all episodes for the current user"""
    try:
        from services.episode_service import episode_service
        deleted = await episode_service.delete_all_episodes(current_user.user_id)
        return {
            "success": True,
            "message": f"Deleted {deleted} episode(s)",
            "deleted_count": deleted,
            "user_id": current_user.user_id,
        }
    except Exception as e:
        logger.error("Failed to delete all episodes: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/settings/user/episodes-preferences")
async def get_episodes_preferences(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        inject = await settings_service.get_episodes_inject_enabled(current_user.user_id)
        return {"episodes_inject_enabled": inject}
    except Exception as e:
        logger.error("Failed to get episodes preferences: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/settings/user/episodes-preferences")
async def set_episodes_preferences(
    request: EpisodesPreferencesResponse,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        from services.user_settings_kv_service import set_user_setting
        await set_user_setting(
            current_user.user_id,
            "episodes_inject_enabled",
            "true" if request.episodes_inject_enabled else "false",
            "boolean",
        )
        return {"episodes_inject_enabled": request.episodes_inject_enabled}
    except Exception as e:
        logger.error("Failed to set episodes preferences: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/settings/user/facts/pending")
async def get_pending_facts(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Get pending fact updates (agent proposed changes awaiting user review)."""
    try:
        pending = await settings_service.get_pending_facts(current_user.user_id)
        return {
            "success": True,
            "pending": pending,
            "user_id": current_user.user_id,
        }
    except Exception as e:
        logger.error("Failed to get pending facts: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/settings/user/facts/pending/{history_id}/resolve")
async def resolve_pending_fact(
    history_id: int,
    request: ResolvePendingFactRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Accept or reject a pending fact update."""
    try:
        action = (request.action or "accept").strip().lower()
        if action not in ("accept", "reject"):
            raise HTTPException(status_code=400, detail="action must be 'accept' or 'reject'")
        result = await settings_service.resolve_pending_fact(
            current_user.user_id, history_id, action
        )
        if result.get("success"):
            return {
                "success": True,
                "message": result.get("message", ""),
                "user_id": current_user.user_id,
            }
        return {
            "success": False,
            "message": result.get("message", "Failed to resolve"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resolve pending fact: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/settings/user/facts/history")
async def get_fact_history(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
    fact_key: Optional[str] = None,
    limit: int = 50,
):
    """Get fact change history, optionally filtered by fact_key."""
    try:
        history = await settings_service.get_fact_history(
            current_user.user_id, fact_key=fact_key, limit=limit
        )
        return {
            "success": True,
            "history": history,
            "user_id": current_user.user_id,
        }
    except Exception as e:
        logger.error("Failed to get fact history: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/vision/service-status")
async def get_vision_service_status(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Check if vision service is available"""
    try:
        from services.capabilities_service import capabilities_service
        available = await capabilities_service.is_vision_service_available()
        return {
            "available": available,
            "status": "healthy" if available else "unavailable"
        }
    except Exception as e:
        logger.error(f"❌ Failed to check vision service status: {str(e)}")
        return {
            "available": False,
            "status": "unavailable"
        }


@router.get("/api/settings/user/vision-features")
async def get_vision_features_enabled(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get user's vision features opt-in status"""
    try:
        logger.info(f"👁️ Getting vision features status for user: {current_user.username}")
        enabled = await settings_service.get_vision_features_enabled(current_user.user_id)
        return {
            "success": True,
            "enabled": enabled,
            "user_id": current_user.user_id
        }
    except Exception as e:
        logger.error(f"❌ Failed to get vision features status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/settings/user/vision-features")
async def set_vision_features_enabled(
    request: VisionFeaturesRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Set user's vision features opt-in"""
    try:
        logger.info(f"👁️ Setting vision features for user {current_user.username} to: {request.enabled}")
        success = await settings_service.set_vision_features_enabled(
            current_user.user_id,
            request.enabled
        )
        
        if success:
            return {
                "success": True,
                "enabled": request.enabled,
                "message": "Vision features setting updated successfully",
                "user_id": current_user.user_id
            }
        else:
            return {
                "success": False,
                "message": "Failed to update vision features setting"
            }
    except Exception as e:
        logger.error(f"❌ Failed to set vision features: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/settings/{category}")
async def get_settings_by_category(category: str):
    """Get settings by category"""
    try:
        logger.info(f"⚙️ Getting settings for category: {category}")
        settings = await settings_service.get_settings_by_category(category)
        return {
            "category": category,
            "settings": settings,
            "count": len(settings)
        }
    except Exception as e:
        logger.error(f"❌ Failed to get settings for category {category}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Device tokens (Bastion Local Proxy) ---

class DeviceTokenCreateRequest(BaseModel):
    device_name: str = Field(..., min_length=1, max_length=255)


class DeviceTokenCreateResponse(BaseModel):
    token_id: str
    token: str
    device_name: str
    message: str = "Copy the token now; it will not be shown again."


@router.post("/api/settings/device-tokens", response_model=DeviceTokenCreateResponse)
async def create_device_token(
    request: DeviceTokenCreateRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Create a new device token for the Bastion Local Proxy daemon."""
    try:
        from services.device_token_service import create_device_token as svc_create
        token_id, raw_token = await svc_create(current_user.user_id, request.device_name)
        return DeviceTokenCreateResponse(
            token_id=token_id,
            token=raw_token,
            device_name=request.device_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create device token: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/settings/device-tokens")
async def list_device_tokens(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """List device tokens for the current user (without raw token)."""
    try:
        from services.device_token_service import list_device_tokens as svc_list
        tokens = await svc_list(current_user.user_id)
        return {"tokens": tokens}
    except Exception as e:
        logger.error(f"Failed to list device tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/settings/device-tokens/{token_id}")
async def revoke_device_token(
    token_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Revoke a device token."""
    try:
        from services.device_token_service import revoke_device_token as svc_revoke
        await svc_revoke(token_id, current_user.user_id)
        return {"success": True, "message": "Token revoked"}
    except Exception as e:
        logger.error(f"Failed to revoke device token: {e}")
        raise HTTPException(status_code=500, detail=str(e))