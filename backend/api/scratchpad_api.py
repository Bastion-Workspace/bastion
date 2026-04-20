"""
User scratch pad: four pads stored in user_settings (key scratchpad_pads), shared across home dashboards.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException

from models.api_models import AuthenticatedUserResponse
from models.home_dashboard_models import (
    SCRATCH_PAD_SETTING_KEY,
    ScratchPadData,
    default_scratchpad_data,
)
from services.user_settings_kv_service import get_user_setting, set_user_setting
from utils.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Scratch pad"])


@router.get("/api/scratchpad", response_model=ScratchPadData)
async def get_scratchpad(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> ScratchPadData:
    raw = await get_user_setting(current_user.user_id, SCRATCH_PAD_SETTING_KEY)
    if not raw or not str(raw).strip():
        return default_scratchpad_data()
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(parsed, dict):
            return default_scratchpad_data()
        return ScratchPadData.model_validate(parsed)
    except Exception as e:
        logger.warning("Invalid scratchpad JSON for user %s: %s", current_user.user_id, e)
        return default_scratchpad_data()


@router.put("/api/scratchpad", response_model=ScratchPadData)
async def put_scratchpad(
    body: ScratchPadData,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> ScratchPadData:
    try:
        payload = body.model_dump(mode="json")
        ok = await set_user_setting(
            current_user.user_id,
            SCRATCH_PAD_SETTING_KEY,
            json.dumps(payload, ensure_ascii=False),
            "json",
        )
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to save scratch pad")
        return body
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
