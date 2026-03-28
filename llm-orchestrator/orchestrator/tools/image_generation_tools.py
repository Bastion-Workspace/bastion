"""
Image Generation Tools - Image generation and reference lookup via backend gRPC
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


def _format_generate_result(data: Dict[str, Any]) -> str:
    """Format generate_image result dict into a string for the LLM."""
    if not data:
        return "No result."
    if not data.get("success"):
        return data.get("error", "Image generation failed.")
    images = data.get("images") or []
    if not images:
        return "No images generated."
    parts = [f"Generated {len(images)} image(s). Model: {data.get('model', '')}"]
    for i, img in enumerate(images, 1):
        url = img.get("url", "")
        path = img.get("path", "")
        w = img.get("width")
        h = img.get("height")
        parts.append(f"  {i}. URL: {url}" if url else f"  {i}. Path: {path}")
        if w and h:
            parts.append(f"      Size: {w}x{h}")
    return "\n".join(parts)


async def generate_image_tool(
    prompt: str,
    user_id: str = "system",
    size: str = "1024x1024",
    num_images: int = 1,
    negative_prompt: Optional[str] = None,
    model: Optional[str] = None,
    check_reference_first: bool = False,
    folder_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate image(s) from a text prompt. When check_reference_first=True, returns reference image if one exists for the prompt/object name. Pass folder_id to place promoted documents in a specific folder."""
    try:
        if not prompt or not prompt.strip():
            msg = "Error: prompt is required."
            return {"success": False, "error": msg, "formatted": msg, "image_count": 0, "image_urls": [], "document_ids": [], "found": False}
        client = await get_backend_tool_client()
        if check_reference_first:
            data = await client.get_reference_image_for_object(
                object_name=prompt.strip(),
                user_id=user_id,
            )
            if data:
                msg = f"Reference image found for '{prompt.strip()[:80]}'. You can describe this object in the prompt when generating."
                return {"found": True, "formatted": msg, "success": True, "image_count": 0, "image_urls": [], "document_ids": []}
            # Fall through to generation
        logger.info("generate_image: prompt=%s size=%s", prompt[:80], size)
        result = await client.generate_image(
            prompt=prompt.strip(),
            size=size,
            format="png",
            seed=None,
            num_images=num_images,
            negative_prompt=negative_prompt,
            user_id=user_id,
            model=model,
            reference_image_data=None,
            reference_image_url=None,
            reference_strength=0.5,
            folder_id=folder_id,
        )
        out = dict(result) if isinstance(result, dict) else {}
        out["formatted"] = _format_generate_result(result)
        images = out.get("images") or []
        out["image_count"] = len(images)
        out["image_urls"] = [img.get("url") for img in images if img.get("url")]
        out["document_ids"] = [img.get("document_id") for img in images if img.get("document_id")]
        return out
    except Exception as e:
        logger.error("generate_image_tool error: %s", e)
        err = str(e)
        return {"success": False, "error": err, "formatted": f"Error: {err}", "image_count": 0, "image_urls": [], "document_ids": [], "found": False}


class GenerateImageInputs(BaseModel):
    prompt: str = Field(description="Description of the image to generate")
    folder_id: Optional[str] = Field(
        default=None,
        description="Optional folder ID where promoted image documents should be placed",
    )


class ImageGenerationOutputs(BaseModel):
    """Outputs for generate_image_tool."""
    formatted: str = Field(description="Human-readable result")
    success: bool = True
    error: Optional[str] = None
    found: Optional[bool] = None
    image_count: int = Field(default=0, description="Number of images generated")
    image_urls: List[str] = Field(default_factory=list, description="URLs of generated images")
    document_ids: List[str] = Field(default_factory=list, description="Document IDs if saved to workspace")


register_action(name="generate_image", category="image", description="Generate image(s) from a text prompt; set check_reference_first=True to return reference image if one exists", inputs_model=GenerateImageInputs, outputs_model=ImageGenerationOutputs, tool_function=generate_image_tool)
