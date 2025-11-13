import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from api.auth_api import get_current_user, AuthenticatedUserResponse
from services.user_settings_kv_service import get_user_setting, set_user_setting

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/services", tags=["Services"])


class TwitterConfigRequest(BaseModel):
    enabled: Optional[bool] = None
    bearer_token: Optional[str] = None
    user_id: Optional[str] = None
    poll_interval_minutes: Optional[int] = None
    backfill_days: Optional[int] = None
    include_replies: Optional[bool] = None
    include_retweets: Optional[bool] = None
    include_quotes: Optional[bool] = None


@router.get("/twitter/config")
async def get_twitter_config(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    try:
        user_id = current_user.user_id
        enabled = (await get_user_setting(user_id, "twitter.enabled")) == "true"
        cfg = {
            "enabled": enabled,
            "bearer_token": (await get_user_setting(user_id, "twitter.bearer_token")) or "",
            "user_id": (await get_user_setting(user_id, "twitter.user_id")) or "",
            "poll_interval_minutes": int((await get_user_setting(user_id, "twitter.poll_interval_minutes")) or 5),
            "backfill_days": int((await get_user_setting(user_id, "twitter.backfill_days")) or 60),
            "include_replies": ((await get_user_setting(user_id, "twitter.include_replies")) or "true") == "true",
            "include_retweets": ((await get_user_setting(user_id, "twitter.include_retweets")) or "true") == "true",
            "include_quotes": ((await get_user_setting(user_id, "twitter.include_quotes")) or "true") == "true",
            "last_poll_at": (await get_user_setting(user_id, "twitter.last_poll_at")) or None,
        }
        return cfg
    except Exception as e:
        logger.error(f"❌ Failed to get twitter config: {e}")
        raise HTTPException(status_code=500, detail="Failed to load Twitter configuration")


@router.post("/twitter/config")
async def set_twitter_config(
    request: TwitterConfigRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    try:
        user_id = current_user.user_id
        if request.enabled is not None:
            await set_user_setting(user_id, "twitter.enabled", "true" if request.enabled else "false")
        if request.bearer_token is not None:
            await set_user_setting(user_id, "twitter.bearer_token", request.bearer_token)
        if request.user_id is not None:
            await set_user_setting(user_id, "twitter.user_id", request.user_id)
        if request.poll_interval_minutes is not None:
            await set_user_setting(user_id, "twitter.poll_interval_minutes", str(int(request.poll_interval_minutes)), data_type="integer")
        if request.backfill_days is not None:
            await set_user_setting(user_id, "twitter.backfill_days", str(int(request.backfill_days)), data_type="integer")
        if request.include_replies is not None:
            await set_user_setting(user_id, "twitter.include_replies", "true" if request.include_replies else "false", data_type="boolean")
        if request.include_retweets is not None:
            await set_user_setting(user_id, "twitter.include_retweets", "true" if request.include_retweets else "false", data_type="boolean")
        if request.include_quotes is not None:
            await set_user_setting(user_id, "twitter.include_quotes", "true" if request.include_quotes else "false", data_type="boolean")

        return {"success": True, "message": "Twitter configuration updated"}
    except Exception as e:
        logger.error(f"❌ Failed to set twitter config: {e}")
        raise HTTPException(status_code=500, detail="Failed to save Twitter configuration")


class ToggleRequest(BaseModel):
    enabled: bool


@router.post("/twitter/toggle")
async def toggle_twitter(
    request: ToggleRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    try:
        await set_user_setting(current_user.user_id, "twitter.enabled", "true" if request.enabled else "false", data_type="boolean")
        return {"enabled": request.enabled}
    except Exception as e:
        logger.error(f"❌ Failed to toggle twitter: {e}")
        raise HTTPException(status_code=500, detail="Failed to toggle Twitter ingestion")


@router.post("/twitter/test")
async def test_twitter_connection(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    try:
        token = await get_user_setting(current_user.user_id, "twitter.bearer_token")
        if not token:
            return {"success": False, "message": "Missing bearer token"}
        # Placeholder connectivity check; real call can be added later
        return {"success": True, "message": "Credentials present"}
    except Exception as e:
        logger.error(f"❌ Twitter test failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to test Twitter connectivity")


@router.get("/twitter/status")
async def twitter_status(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    try:
        enabled = (await get_user_setting(current_user.user_id, "twitter.enabled")) == "true"
        last_poll_at = await get_user_setting(current_user.user_id, "twitter.last_poll_at")
        return {"enabled": enabled, "last_poll_at": last_poll_at}
    except Exception as e:
        logger.error(f"❌ Failed to get twitter status: {e}")
        raise HTTPException(status_code=500, detail="Failed to load Twitter status")


