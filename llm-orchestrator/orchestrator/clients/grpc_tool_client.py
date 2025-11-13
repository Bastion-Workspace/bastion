"""
gRPC Tool Client for LLM Orchestrator
Connects to backend Tool Service for data access and file operations
"""

import logging
import os
import json
from typing import Optional
import grpc

from protos import tool_service_pb2, tool_service_pb2_grpc

logger = logging.getLogger(__name__)


class GRPCToolClient:
    """Client for backend gRPC Tool Service"""
    
    def __init__(self):
        self.host = os.getenv("BACKEND_TOOL_SERVICE_HOST", "backend")
        self.port = os.getenv("BACKEND_TOOL_SERVICE_PORT", "50052")
        self.address = f"{self.host}:{self.port}"
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[tool_service_pb2_grpc.ToolServiceStub] = None
        logger.info(f"GRPCToolClient configured for {self.address}")
    
    async def _ensure_connected(self):
        """Ensure gRPC channel is connected"""
        if self._channel is None:
            self._channel = grpc.aio.insecure_channel(self.address)
            self._stub = tool_service_pb2_grpc.ToolServiceStub(self._channel)
            logger.info(f"✅ Connected to backend tool service at {self.address}")
    
    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        format: str = "png",
        seed: Optional[int] = None,
        num_images: int = 1,
        negative_prompt: Optional[str] = None,
        user_id: str = "system"
    ) -> str:
        """
        Generate images using backend image generation service
        
        Returns JSON string with image URLs and metadata
        """
        try:
            await self._ensure_connected()
            
            # Build request
            request = tool_service_pb2.ImageGenerationRequest(
                prompt=prompt,
                size=size,
                format=format,
                num_images=num_images,
                user_id=user_id
            )
            
            # Add optional fields
            if seed is not None:
                request.seed = seed
            if negative_prompt is not None:
                request.negative_prompt = negative_prompt
            
            logger.info(f"🎨 Calling backend GenerateImage: prompt={prompt[:100]}...")
            
            # Call gRPC method
            response = await self._stub.GenerateImage(request)
            
            # Convert proto response to dict
            result = {
                "success": response.success,
                "model": response.model,
                "prompt": response.prompt,
                "size": response.size,
                "format": response.format,
                "images": []
            }
            
            if response.success:
                for img in response.images:
                    result["images"].append({
                        "filename": img.filename,
                        "path": img.path,
                        "url": img.url,
                        "width": img.width,
                        "height": img.height,
                        "format": img.format
                    })
                logger.info(f"✅ Generated {len(result['images'])} image(s)")
            else:
                result["error"] = response.error
                logger.error(f"❌ Image generation failed: {response.error}")
            
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"❌ GenerateImage gRPC call failed: {e}")
            error_result = {
                "success": False,
                "error": f"gRPC call failed: {str(e)}"
            }
            return json.dumps(error_result)
    
    async def close(self):
        """Close gRPC channel"""
        if self._channel:
            await self._channel.close()
            logger.info("Closed backend tool service connection")


# Global instance
_grpc_tool_client: Optional[GRPCToolClient] = None


async def get_grpc_tool_client() -> GRPCToolClient:
    """Get global gRPC tool client instance"""
    global _grpc_tool_client
    if _grpc_tool_client is None:
        _grpc_tool_client = GRPCToolClient()
    return _grpc_tool_client

