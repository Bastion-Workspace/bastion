"""
Settings API - Handles all settings-related endpoints
"""

import logging
import re
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

router = APIRouter(prefix="/api/settings", tags=["Settings"])


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


@router.get("", response_model=SettingsResponse)
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


@router.put("/{key}", response_model=SettingUpdateResponse)
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


@router.post("/bulk", response_model=SettingUpdateResponse)
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


@router.delete("/{key}")
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


@router.get("/user/timezone")
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


@router.put("/user/timezone")
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


@router.get("/user/zip-code")
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


@router.put("/user/zip-code")
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


@router.get("/user/time-format")
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


@router.put("/user/time-format")
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


@router.get("/prompt", response_model=PromptSettingsResponse)
async def get_prompt_settings(current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Get current user's prompt settings"""
    try:
        # Get user's current settings from database
        user_settings = await settings_service.get_user_prompt_settings(current_user.user_id)
        
        # If no settings exist, return defaults
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


@router.post("/prompt", response_model=PromptSettingsResponse)
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


@router.get("/prompt/options")
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


@router.get("/user/preferred-name")
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


@router.put("/user/preferred-name")
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


@router.get("/user/ai-context")
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


@router.put("/user/ai-context")
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


@router.get("/{category}")
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