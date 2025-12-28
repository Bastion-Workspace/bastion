"""
Status Bar API - Provides status bar data (time, weather, version)
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from clients.tool_service_client import get_tool_service_client
from services.auth_service import auth_service
from utils.auth_middleware import get_current_user
from models.api_models import AuthenticatedUserResponse
from version import __version__

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Status Bar"])


class StatusBarDataResponse(BaseModel):
    """Response model for status bar data"""
    current_time: str
    date_formatted: str
    weather: Optional[Dict[str, Any]] = None
    app_version: str


@router.get("/api/status-bar/data", response_model=StatusBarDataResponse)
async def get_status_bar_data(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> StatusBarDataResponse:
    """Get status bar data including current time, weather, and app version"""
    try:
        # Get current time
        now = datetime.utcnow()
        current_time = now.strftime("%H:%M:%S")
        date_formatted = now.strftime("%m/%d/%Y")
        
        # Get zip code from user preferences
        user_preferences = current_user.preferences or {}
        zip_code = user_preferences.get("zip_code")
        
        weather_data = None
        if zip_code:
            try:
                # Fetch weather data using Tool Service gRPC client
                tool_client = await get_tool_service_client()
                weather_data = await tool_client.get_weather_data(
                    location=zip_code,
                    user_id=current_user.user_id,
                    data_types=["current"]
                )
                
                if not weather_data:
                    logger.warning(f"Weather fetch failed for zip code: {zip_code}")
            except Exception as e:
                logger.error(f"Error fetching weather data: {e}")
        
        return StatusBarDataResponse(
            current_time=current_time,
            date_formatted=date_formatted,
            weather=weather_data,
            app_version=__version__
        )
        
    except Exception as e:
        logger.error(f"Error getting status bar data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

