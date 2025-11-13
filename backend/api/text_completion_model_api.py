"""
Text Completion Model API - Endpoints for managing fast text-completion model
Used by proofreading/editor features; distinct from main chat model.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from services.settings_service import settings_service
from utils.auth_middleware import get_current_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Text Completion Model"])


class TextCompletionModelRequest(BaseModel):
    model_name: str = Field(..., description="OpenRouter model id for fast text completion")


@router.get("/models/text-completion")
async def get_text_completion_model(current_user_id: str = Depends(get_current_user_id)):
    try:
        model = await settings_service.get_text_completion_model()
        return {"text_completion_model": model or ""}
    except Exception as e:
        logger.error(f"❌ Failed to get text completion model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/text-completion")
async def set_text_completion_model(
    request: TextCompletionModelRequest,
    current_user_id: str = Depends(get_current_user_id)
):
    try:
        success = await settings_service.set_text_completion_model(request.model_name)
        if success:
            return {
                "success": True,
                "text_completion_model": request.model_name,
                "message": f"Text completion model set to {request.model_name}"
            }
        raise HTTPException(status_code=500, detail="Failed to update text completion model setting")
    except Exception as e:
        logger.error(f"❌ Failed to set text completion model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


