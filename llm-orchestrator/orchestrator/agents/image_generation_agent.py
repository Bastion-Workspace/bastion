"""
Image Generation Agent
Generates images from user prompts using OpenRouter image models via backend gRPC tool
"""

import json
import logging
from typing import Dict, Any, Optional
from langchain_core.messages import AIMessage

from orchestrator.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ImageGenerationAgent(BaseAgent):
    """
    Image Generation Agent
    Generates images from user prompts using configured OpenRouter image models
    """
    
    def __init__(self):
        super().__init__("image_generation_agent")
        self._grpc_client = None
    
    async def _get_grpc_client(self):
        """Get or create gRPC client for backend tools"""
        if self._grpc_client is None:
            from orchestrator.clients.grpc_tool_client import get_grpc_tool_client
            self._grpc_client = await get_grpc_tool_client()
        return self._grpc_client
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process image generation request"""
        try:
            # Extract user prompt from messages
            messages = state.get("messages", []) or []
            user_prompt = ""
            
            for msg in reversed(messages):
                # Check LangChain message objects
                content = getattr(msg, "content", None)
                role = getattr(msg, "type", None)
                if content and (role == "human" or role is None):
                    user_prompt = str(content)
                    break
                
                # Check dict messages
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_prompt = str(msg.get("content") or "")
                    break
            
            user_prompt = user_prompt.strip()
            if not user_prompt:
                return self._create_error_result("No prompt provided for image generation")
            
            # Extract optional parameters from shared_memory
            sm = state.get("shared_memory", {}) or {}
            gen_params = (sm.get("image_generation_params") or {}) if isinstance(sm, dict) else {}
            size = str(gen_params.get("size", "1024x1024"))
            fmt = str(gen_params.get("format", "png"))
            seed = gen_params.get("seed")
            num_images = int(gen_params.get("num_images", 1))
            negative_prompt = gen_params.get("negative_prompt")
            
            # Call backend gRPC tool for image generation
            grpc_client = await self._get_grpc_client()
            
            logger.info(f"🎨 Generating {num_images} image(s) with prompt: {user_prompt[:100]}...")
            
            # Call GenerateImage gRPC method
            tool_result_json = await grpc_client.generate_image(
                prompt=user_prompt,
                size=size,
                format=fmt,
                seed=seed,
                num_images=num_images,
                negative_prompt=negative_prompt
            )
            
            tool_result = json.loads(tool_result_json) if isinstance(tool_result_json, str) else tool_result_json
            
            if not tool_result.get("success"):
                error_msg = tool_result.get("error", "Unknown error")
                logger.error(f"❌ Image generation failed: {error_msg}")
                return self._create_error_result(f"Image generation failed: {error_msg}")
            
            # Extract image URLs from result
            images = tool_result.get("images", [])
            urls = [img.get("url") for img in images if isinstance(img, dict) and img.get("url")]
            urls_text = "\n".join(urls) if urls else ""
            
            # Build persona-aware response
            response_text = self._format_completion_message(
                len(images), 
                urls_text, 
                state.get("persona")
            )
            
            logger.info(f"✅ Successfully generated {len(images)} image(s)")
            
            # Build structured response
            structured = {
                "task_status": "complete",
                "prompt": user_prompt,
                "model_used": tool_result.get("model", "unknown"),
                "images": images,
                "message": response_text,
            }
            
            return {
                "messages": [AIMessage(content=response_text)],
                "agent_results": {
                    "agent_type": "image_generation_agent",
                    "structured_response": structured,
                    "task_status": "complete",
                    "is_complete": True
                },
                "is_complete": True,
                "shared_memory": state.get("shared_memory", {})
            }
        
        except Exception as e:
            logger.error(f"❌ ImageGenerationAgent failed: {e}")
            return self._create_error_result(f"Image generation failed: {str(e)}")
    
    def _format_completion_message(
        self, 
        num_images: int, 
        urls_text: str, 
        persona: Optional[Dict[str, Any]]
    ) -> str:
        """Format completion message based on persona style"""
        style = str((persona or {}).get("persona_style") or "professional").lower()
        
        if style == "friendly":
            base = f"All set! Generated {num_images} image(s)."
        elif style == "snarky":
            base = f"Done. Generated {num_images} image(s)."
        elif style == "sycophantic":
            base = f"Delighted to deliver {num_images} image(s)."
        elif style == "rude_insulting":
            # Keep output neutral even for abrasive styles in utility responses
            base = f"Generated {num_images} image(s)."
        else:
            base = f"Your image request is complete. Generated {num_images} image(s)."
        
        return (base + ("\n" + urls_text if urls_text else "")).strip()
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        logger.error(f"❌ Image generation error: {error_message}")
        return {
            "messages": [AIMessage(content=error_message)],
            "agent_results": {
                "agent_type": "image_generation_agent",
                "task_status": "error",
                "error_message": error_message,
                "is_complete": True
            },
            "is_complete": True
        }


# Singleton instance
_image_generation_agent_instance = None


def get_image_generation_agent() -> ImageGenerationAgent:
    """Get global image generation agent instance"""
    global _image_generation_agent_instance
    if _image_generation_agent_instance is None:
        _image_generation_agent_instance = ImageGenerationAgent()
    return _image_generation_agent_instance

