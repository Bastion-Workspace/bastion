"""
Weather Tool Models
Pydantic models for weather tool requests with automatic location resolution
"""

from typing import Optional, Literal, Tuple
from pydantic import BaseModel, Field, field_validator
import logging

logger = logging.getLogger(__name__)


class WeatherLocationRequest(BaseModel):
    """
    Weather location request with automatic resolution to user's ZIP code
    
    If location is None, empty, or vague AND user_id is provided,
    automatically resolves to user's ZIP code from profile.
    
    Returns clear error if user_id provided but no ZIP code found.
    """
    location: Optional[str] = Field(
        default=None,
        description="Location (ZIP code, city name, or 'city,country' format). "
                   "If None/empty/vague and user_id provided, uses user's ZIP from profile."
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User ID for automatic location fallback to profile ZIP code"
    )
    
    @field_validator('location')
    @classmethod
    def validate_location(cls, v: Optional[str]) -> Optional[str]:
        """Normalize location string"""
        if v is None:
            return None
        return v.strip() if isinstance(v, str) else str(v).strip()
    
    async def resolve_location(self) -> Tuple[str, Optional[str]]:
        """
        Resolve location to a valid location string
        
        Returns:
            tuple: (resolved_location, error_message)
            - If successful: (location_string, None)
            - If error: (None, error_message)
        """
        # Vague location terms that trigger ZIP code fallback
        vague_terms = [
            "user's location", "my location", "current location",
            "none", "unknown", "infer from", "document context",
            "user location", "home", "here"
        ]
        
        location_lower = (self.location or "").lower()
        is_vague = not self.location or any(term in location_lower for term in vague_terms)
        
        # If location is provided and not vague, use it
        if self.location and not is_vague:
            return self.location, None
        
        # If location is vague/missing, try to use user's ZIP code
        if is_vague and self.user_id:
            try:
                from services.settings_service import settings_service
                user_zip = await settings_service.get_user_zip_code(self.user_id)
                
                if user_zip:
                    logger.info(f"📍 Resolved vague location '{self.location}' to user's ZIP: {user_zip}")
                    return user_zip, None
                else:
                    error_msg = (
                        "Location is required for weather queries. "
                        f"Please set your ZIP code in your user profile settings, "
                        f"or specify a location (e.g., 'What's the weather in Los Angeles?')."
                    )
                    logger.warning(f"⚠️ Vague location '{self.location}' provided but no ZIP code found in profile for user {self.user_id}")
                    return None, error_msg
            except Exception as e:
                logger.error(f"❌ Error retrieving user ZIP code: {e}")
                return None, f"Error retrieving location from profile: {str(e)}"
        
        # No location and no user_id - return error
        if not self.location:
            return None, (
                "Location is required for weather queries. "
                "Please specify a location (e.g., 'What's the weather in Los Angeles?') "
                "or set your ZIP code in your user profile settings."
            )
        
        # Location is vague but no user_id - return error
        return None, (
            "Location is required for weather queries. "
            "Please specify a location (e.g., 'What's the weather in Los Angeles?') "
            "or ensure you're authenticated so the system can use your profile ZIP code."
        )


class WeatherConditionsRequest(WeatherLocationRequest):
    """Request for current weather conditions"""
    units: Literal["imperial", "metric", "kelvin"] = Field(
        default="imperial",
        description="Temperature units"
    )


class WeatherForecastRequest(WeatherLocationRequest):
    """Request for weather forecast"""
    days: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of days to forecast (1-5)"
    )
    units: Literal["imperial", "metric", "kelvin"] = Field(
        default="imperial",
        description="Temperature units"
    )


class WeatherHistoryRequest(WeatherLocationRequest):
    """Request for historical weather data"""
    date_str: str = Field(
        description="Date string: 'YYYY-MM-DD' (specific day), 'YYYY-MM' (monthly average), "
                   "or 'YYYY-MM to YYYY-MM' (date range, max 24 months)"
    )
    units: Literal["imperial", "metric", "kelvin"] = Field(
        default="imperial",
        description="Temperature units"
    )
