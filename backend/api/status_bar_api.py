"""
Status Bar API - Provides status bar data (time, weather, version)
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from services.langgraph_tools.weather_tools import weather_conditions
from services.auth_service import auth_service
from utils.auth_middleware import get_current_user
from models.api_models import AuthenticatedUserResponse
from version import __version__

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/status-bar", tags=["Status Bar"])


class StatusBarDataResponse(BaseModel):
    """Response model for status bar data"""
    current_time: str
    date_formatted: str
    weather: Optional[Dict[str, Any]] = None
    app_version: str


@router.get("/data", response_model=StatusBarDataResponse)
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
                # Fetch weather data using weather tools
                weather_result = await weather_conditions(zip_code, units="imperial", user_id=current_user.user_id)
                
                if weather_result.get("success"):
                    weather_data = {
                        "location": weather_result.get("location", {}).get("name", zip_code),
                        "temperature": int(weather_result.get("current", {}).get("temperature", 0)),
                        "conditions": weather_result.get("current", {}).get("conditions", ""),
                        "moon_phase": weather_result.get("moon_phase", {})
                    }
                else:
                    logger.warning(f"Weather fetch failed: {weather_result.get('error', 'Unknown error')}")
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

