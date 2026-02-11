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
    
    async def detect_faces(
        self,
        attachment_path: str,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Detect faces in an image using Tools Service
        
        Args:
            attachment_path: Full path to image file
            user_id: User ID for access control
            
        Returns:
            Dict with success, faces (list of face detections), face_count, image_width, image_height, error
        """
        try:
            await self.initialize()
            
            request = tool_service_pb2.DetectFacesRequest(
                attachment_path=attachment_path,
                user_id=user_id
            )
            
            response = await self.stub.DetectFaces(request)
            
            if response.success:
                faces_list = []
                for pb_face in response.faces:
                    faces_list.append({
                        "face_encoding": list(pb_face.face_encoding),
                        "bbox_x": pb_face.bbox_x,
                        "bbox_y": pb_face.bbox_y,
                        "bbox_width": pb_face.bbox_width,
                        "bbox_height": pb_face.bbox_height
                    })
                
                return {
                    "success": True,
                    "faces": faces_list,
                    "face_count": response.face_count,
                    "image_width": response.image_width if response.HasField("image_width") else None,
                    "image_height": response.image_height if response.HasField("image_height") else None
                }
            else:
                error_msg = response.error if response.HasField("error") else "Unknown error"
                logger.error(f"Face detection failed: {error_msg}")
                return {
                    "success": False,
                    "faces": [],
                    "face_count": 0,
                    "error": error_msg
                }
            
        except grpc.RpcError as e:
            logger.error(f"Face detection failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "faces": [],
                "face_count": 0,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error in face detection: {e}")
            return {
                "success": False,
                "faces": [],
                "face_count": 0,
                "error": str(e)
            }
    
    async def identify_faces(
        self,
        attachment_path: str,
        user_id: str = "system",
        confidence_threshold: float = 0.82
    ) -> Dict[str, Any]:
        """
        Identify people in an image by matching against known identities using Tools Service
        
        Args:
            attachment_path: Full path to image file
            user_id: User ID for access control
            confidence_threshold: Minimum confidence for identity matches (default: 0.82, aligns with L2 < 0.6)
            
        Returns:
            Dict with success, identified_faces (list of identified faces), face_count, identified_count, error
        """
        try:
            await self.initialize()
            
            request = tool_service_pb2.IdentifyFacesRequest(
                attachment_path=attachment_path,
                user_id=user_id,
                confidence_threshold=confidence_threshold
            )
            
            response = await self.stub.IdentifyFaces(request)
            
            if response.success:
                identified_faces_list = []
                for pb_face in response.identified_faces:
                    identified_faces_list.append({
                        "identity_name": pb_face.identity_name,
                        "confidence": pb_face.confidence,
                        "bbox_x": pb_face.bbox_x,
                        "bbox_y": pb_face.bbox_y,
                        "bbox_width": pb_face.bbox_width,
                        "bbox_height": pb_face.bbox_height
                    })
                
                return {
                    "success": True,
                    "identified_faces": identified_faces_list,
                    "face_count": response.face_count,
                    "identified_count": response.identified_count
                }
            else:
                error_msg = response.error if response.HasField("error") else "Unknown error"
                logger.error(f"Face identification failed: {error_msg}")
                return {
                    "success": False,
                    "identified_faces": [],
                    "face_count": 0,
                    "identified_count": 0,
                    "error": error_msg
                }
            
        except grpc.RpcError as e:
            logger.error(f"Face identification failed: {e.code()} - {e.details()}")
            return {
                "success": False,
                "identified_faces": [],
                "face_count": 0,
                "identified_count": 0,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error in face identification: {e}")
            return {
                "success": False,
                "identified_faces": [],
                "face_count": 0,
                "identified_count": 0,
                "error": str(e)
            }


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

