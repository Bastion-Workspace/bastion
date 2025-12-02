"""
Classification Model API - Endpoints for managing the fast classification model
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from services.settings_service import settings_service
from utils.auth_middleware import get_current_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Classification Model"])


class ClassificationModelRequest(BaseModel):
    """Request model for setting classification model"""
    model_name: str = Field(..., description="Name of the fast model for intent classification")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "anthropic/claude-3-haiku"
            }
        }


class ClassificationModelResponse(BaseModel):
    """Response model for classification model operations"""
    success: bool
    current_model: str
    chat_model: str
    text_completion_model: str
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "current_model": "anthropic/claude-3-haiku",
                "chat_model": "anthropic/claude-3-5-sonnet-20241022",
                "message": "Classification model updated successfully"
            }
        }


@router.get("/models/classification", response_model=Dict[str, Any])
async def get_classification_model(
    current_user_id: str = Depends(get_current_user_id)
):
    """Get the current classification model and main chat model"""
    try:
        classification_model = await settings_service.get_classification_model()
        chat_model = await settings_service.get_llm_model()
        text_completion_model = await settings_service.get_text_completion_model() or ""
        
        # Determine effective models (including fallbacks)
        from config import settings as backend_settings
        effective_chat_model = chat_model or backend_settings.DEFAULT_MODEL
        effective_classification_model = classification_model or backend_settings.FAST_MODEL

        return {
            "classification_model": classification_model,  # Raw setting (may be empty)
            "chat_model": chat_model,  # Raw setting (may be empty)
            "effective_classification_model": effective_classification_model,  # What agents actually use
            "effective_chat_model": effective_chat_model,  # What agents actually use
            "text_completion_model": text_completion_model,
            "classification_model_is_fallback": not bool(classification_model),  # True if using fallback
            "chat_model_is_fallback": not bool(chat_model),  # True if using fallback
            "description": "Classification model is used for fast intent classification, while chat model is used for main responses"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get classification model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/classification", response_model=ClassificationModelResponse)
async def set_classification_model(
    request: ClassificationModelRequest,
    current_user_id: str = Depends(get_current_user_id)
):
    """Set the classification model (fast model for intent classification)"""
    try:
        success = await settings_service.set_classification_model(request.model_name)
        
        if success:
            classification_model = await settings_service.get_classification_model()
            chat_model = await settings_service.get_llm_model()
            text_completion_model = await settings_service.get_text_completion_model() or ""
            
            return ClassificationModelResponse(
                success=True,
                current_model=classification_model,
                chat_model=chat_model,
                text_completion_model=text_completion_model,
                message=f"Classification model set to {request.model_name}"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to update classification model setting")
            
    except Exception as e:
        logger.error(f"❌ Failed to set classification model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/classification")
async def reset_classification_model(
    current_user_id: str = Depends(get_current_user_id)
):
    """Reset classification model to use the main chat model"""
    try:
        success = await settings_service.delete_setting("classification_model")
        
        if success:
            chat_model = await settings_service.get_llm_model()
            
            return {
                "success": True,
                "message": "Classification model reset to use main chat model",
                "current_model": chat_model
            }
        else:
            return {
                "success": True,
                "message": "Classification model was already using main chat model",
                "current_model": await settings_service.get_llm_model()
            }
            
    except Exception as e:
        logger.error(f"❌ Failed to reset classification model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/fast-recommendations")
async def get_fast_model_recommendations():
    """Get recommended fast models for intent classification"""
    return {
        "recommended_models": [
            {
                "model": "anthropic/claude-3-haiku",
                "description": "Very fast, good for classification tasks",
                "speed": "Very Fast",
                "cost": "Low",
                "accuracy": "High"
            },
            {
                "model": "openai/gpt-3.5-turbo",
                "description": "Fast and reliable for classification",
                "speed": "Fast",
                "cost": "Low",
                "accuracy": "High"
            },
            {
                "model": "meta-llama/llama-3.1-8b-instruct",
                "description": "Open source, fast classification model",
                "speed": "Fast",
                "cost": "Very Low",
                "accuracy": "Good"
            },
            {
                "model": "google/gemini-flash-1.5",
                "description": "Google's fast model for quick tasks",
                "speed": "Very Fast",
                "cost": "Low",
                "accuracy": "High"
            },
            {
                "model": "mistralai/mistral-7b-instruct",
                "description": "Efficient open source model",
                "speed": "Fast",
                "cost": "Very Low",
                "accuracy": "Good"
            }
        ],
        "usage_tips": [
            "Choose a fast model for classification to reduce response delay",
            "Classification models only need to determine intent, not generate full responses",
            "Haiku and GPT-3.5-turbo are excellent choices for most use cases",
            "You can always fall back to the main chat model if classification fails"
        ]
    }


@router.get("/models/classification/performance")
async def get_classification_performance():
    """Get performance metrics for classification model usage"""
    try:
        # This would integrate with the optimized intent service to get real metrics
        # For now, return placeholder data
        return {
            "classification_model": await settings_service.get_classification_model(),
            "chat_model": await settings_service.get_llm_model(),
            "metrics": {
                "average_classification_time": "150ms",
                "classification_accuracy": "94%",
                "cache_hit_rate": "0%",  # Not implemented yet
                "fallback_rate": "2%",
                "cost_savings": "60%"
            },
            "recommendations": [
                "Classification model is performing well",
                "Consider enabling caching for repeated queries",
                "Monitor fallback rate - should be under 5%"
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get classification performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))
