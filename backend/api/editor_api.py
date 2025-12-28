"""
Editor API
Provides suggestion endpoint for inline ghost-text completions in the editor.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends

from utils.auth_middleware import get_current_user, AuthenticatedUserResponse
from models.editor_models import EditorSuggestionRequest, EditorSuggestionResponse
from services.editor_suggestion_service import editor_suggestion_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Editor"])


@router.post("/api/editor/suggest", response_model=EditorSuggestionResponse)
async def suggest_inline(
    req: EditorSuggestionRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    try:
        result = await editor_suggestion_service.suggest(req)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Editor suggestion error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate suggestion")


