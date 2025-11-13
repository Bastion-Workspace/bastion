"""
Image Generation Agent

Generates images from user prompts using the configured OpenRouter image model.
Follows LangGraph best practices and centralized tool registry access.
"""

import json
import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent
from models.agent_response_models import TaskStatus
from services.langgraph_tools.centralized_tool_registry import get_tool_function, AgentType
from services.settings_service import settings_service


logger = logging.getLogger(__name__)


class ImageGenerationAgent(BaseAgent):
    def __init__(self):
        super().__init__("image_generation_agent")

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            messages = state.get("messages", []) or []
            user_prompt = ""
            for msg in reversed(messages):
                content = getattr(msg, "content", None)
                role = getattr(msg, "type", None)
                if content and (role == "human" or role is None):
                    user_prompt = str(content)
                    break
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_prompt = str(msg.get("content") or "")
                    break

            user_prompt = user_prompt.strip()
            if not user_prompt:
                return self._create_error_result("No prompt provided for image generation")

            # Ensure model is configured
            model = await settings_service.get_image_generation_model()
            if not model:
                return self._create_error_result("Image generation model not configured. Set it in Settings → Models.")

            # Optional parameters from shared_memory
            sm = state.get("shared_memory", {}) or {}
            gen_params = (sm.get("image_generation_params") or {}) if isinstance(sm, dict) else {}
            size = str(gen_params.get("size", "1024x1024"))
            fmt = str(gen_params.get("format", "png"))
            seed = gen_params.get("seed")
            num_images = int(gen_params.get("num_images", 1))

            # Call tool via registry
            tool = await get_tool_function("generate_image", AgentType.IMAGE_GENERATION_AGENT)
            if not tool:
                return self._create_error_result("Image generation tool unavailable")

            tool_result_json = await tool(
                prompt=user_prompt,
                size=size,
                format=fmt,
                seed=seed,
                num_images=num_images,
            )
            tool_result = json.loads(tool_result_json)

            if not tool_result.get("success"):
                return self._create_error_result(f"Image generation failed: {tool_result.get('error', 'unknown error')}")

            images = tool_result.get("images", [])
            urls = [img.get("url") for img in images if isinstance(img, dict) and img.get("url")]
            urls_text = "\n".join(urls) if urls else ""

            # Build persona-aware, neutral response text
            def _format_completion_message(num_images: int, urls_text_value: str, persona: Optional[Dict[str, Any]]) -> str:
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
                return (base + ("\n" + urls_text_value if urls_text_value else "")).strip()

            response_text = _format_completion_message(len(images), urls_text, state.get("persona"))

            structured = {
                "task_status": TaskStatus.COMPLETE,
                "prompt": user_prompt,
                "model_used": model,
                "images": images,
                "message": response_text,
            }

            return {
                "agent_results": {
                    "agent_type": "image_generation_agent",
                    "structured_response": structured,
                    "task_status": "complete",
                },
                "latest_response": response_text,
                "shared_memory": state.get("shared_memory", {}),
            }

        except Exception as e:
            logger.error(f"❌ ImageGenerationAgent failed: {e}")
            # Return a standardized error without relying on BaseAgent.AgentError
            error_message = f"Image generation failed: {str(e)}"
            return {
                "agent_results": {
                    "agent_type": "image_generation_agent",
                    "task_status": "error",
                    "error_message": error_message,
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                },
                "latest_response": error_message,
                "is_complete": True,
            }


