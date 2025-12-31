"""
Tool Service gRPC Client

Provides client interface to the Tool Service for weather and other tool operations.
"""

import grpc
import logging
import os
from typing import Dict, Any, Optional

from config import get_settings
from protos import tool_service_pb2, tool_service_pb2_grpc

logger = logging.getLogger(__name__)


class ToolServiceClient:
    """Client for interacting with the Tool Service via gRPC"""
    
    def __init__(self, service_host: Optional[str] = None, service_port: Optional[int] = None):
        """
        Initialize Tool Service client
        
        Args:
            service_host: gRPC service host (default: from env or config)
            service_port: gRPC service port (default: from env or config)
        """
        self.settings = get_settings()
        self.service_host = service_host or os.getenv('BACKEND_TOOL_SERVICE_HOST', 'tools-service')
        self.service_port = service_port or int(os.getenv('BACKEND_TOOL_SERVICE_PORT', '50052'))
        self.service_url = f"{self.service_host}:{self.service_port}"
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[tool_service_pb2_grpc.ToolServiceStub] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the gRPC channel and stub"""
        if self._initialized:
            return
        
        try:
            logger.info(f"Connecting to Tool Service at {self.service_url}")
            
            # Create insecure channel
            self.channel = grpc.aio.insecure_channel(self.service_url)
            self.stub = tool_service_pb2_grpc.ToolServiceStub(self.channel)
            
            self._initialized = True
            logger.info(f"✅ Connected to Tool Service at {self.service_url}")
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to Tool Service: {e}")
            raise
    
    async def close(self):
        """Close the gRPC channel"""
        if self.channel:
            await self.channel.close()
            self._initialized = False
            logger.info("Tool Service client closed")
    
    async def get_weather_data(
        self,
        location: str,
        user_id: str = "system",
        data_types: Optional[list] = None,
        date_str: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get weather data for a location
        
        Args:
            location: Location (ZIP code, city name, etc.)
            user_id: User ID for access control
            data_types: Types of data to retrieve (e.g., ["current", "forecast", "history"])
            date_str: Optional date string for historical data (YYYY-MM-DD or YYYY-MM)
            
        Returns:
            Weather data dict with location, temperature, conditions, moon_phase, forecast, etc.
        """
        try:
            await self.initialize()
            
            request = tool_service_pb2.WeatherRequest(
                location=location,
                user_id=user_id,
                data_types=data_types or ["current"]
            )
            
            # Add date_str if provided (for historical requests)
            if date_str:
                request.date_str = date_str
            
            response = await self.stub.GetWeatherData(request)
            
            # Extract data from response
            metadata = dict(response.metadata)
            
            # Build weather data dict with full metadata for comprehensive access
            # Also maintain backward compatibility format for status bar API
            weather_data = {
                "location": response.location,
                "current_conditions": response.current_conditions,
                "forecast": list(response.forecast),
                "alerts": list(response.alerts),
                "metadata": metadata,
                # Backward compatibility fields for status bar API
                "temperature": int(metadata.get("temperature", 0)),
                "conditions": metadata.get("conditions", ""),
                "moon_phase": {
                    "phase_name": metadata.get("moon_phase_name", ""),
                    "phase_icon": metadata.get("moon_phase_icon", ""),
                    "phase_value": int(metadata.get("moon_phase_value", 0))
                }
            }
            
            return weather_data
            
        except grpc.RpcError as e:
            logger.error(f"Weather data request failed: {e.code()} - {e.details()}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting weather data: {e}")
            return None


# Global client instance
_tool_service_client: Optional[ToolServiceClient] = None


async def get_tool_service_client() -> ToolServiceClient:
    """Get or create the global tool service client"""
    global _tool_service_client
    
    if _tool_service_client is None:
        _tool_service_client = ToolServiceClient()
        await _tool_service_client.initialize()
    
    return _tool_service_client


async def close_tool_service_client():
    """Close the global tool service client"""
    global _tool_service_client
    
    if _tool_service_client:
        await _tool_service_client.close()
        _tool_service_client = None

