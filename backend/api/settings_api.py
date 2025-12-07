"""
Settings API - Handles all settings-related endpoints
"""

import logging
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
class IntentClassificationModelRequest(BaseModel):
    value: str
    description: str = "Model used for fast intent classification"
    category: str = "classification"


class TimezoneRequest(BaseModel):
    timezone: str


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





# DEPRECATED: Use /api/models/classification instead
@router.put("/intent_classification_model")
async def update_intent_classification_model_deprecated(
    request: IntentClassificationModelRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """DEPRECATED: Use /api/models/classification endpoint instead"""
    logger.warning(f"⚠️ DEPRECATED: User {current_user.username} is using deprecated /api/settings/intent_classification_model endpoint")
    raise HTTPException(
        status_code=422, 
        detail="This endpoint is deprecated. Please use /api/models/classification instead. The frontend should be updated to use the new endpoint."
    )


@router.get("/intent_classification_model")
async def get_intent_classification_model_deprecated():
    """DEPRECATED: Use /api/models/classification endpoint instead"""
    logger.warning(f"⚠️ DEPRECATED: User is using deprecated GET /api/settings/intent_classification_model endpoint")
    raise HTTPException(
        status_code=422, 
        detail="This endpoint is deprecated. Please use /api/models/classification instead. The frontend should be updated to use the new endpoint."
    )


@router.post("/test/intent-classification-model")
async def test_intent_classification_model_validation(
    request: IntentClassificationModelRequest
):
    """Test endpoint to verify Pydantic model validation"""
    logger.info(f"🧪 Test endpoint called with request: {request}")
    logger.info(f"🧪 Request type: {type(request)}")
    logger.info(f"🧪 Request dict: {request.dict()}")
    return {
        "success": True,
        "message": "Pydantic model validation successful",
        "data": request.dict()
        } 



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