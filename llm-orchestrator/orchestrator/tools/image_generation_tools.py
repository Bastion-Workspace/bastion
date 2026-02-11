"""
Image Generation Tools - Image generation and reference lookup via backend gRPC
"""

import logging
from typing import Any, Dict, Optional

from orchestrator.backend_tool_client import get_backend_tool_client

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
) -> str:
    """
    Generate image(s) from a text prompt.

    Args:
        prompt: Description of the image to generate (required).
        user_id: User ID (injected by engine if omitted).
        size: Image size, e.g. "1024x1024", "512x512". Default "1024x1024".
        num_images: Number of images to generate (1-4). Default 1.
        negative_prompt: Optional; what to avoid in the image.
        model: Optional model name (user preference may override).

    Returns:
        Summary with image URL(s) or error message.
    """
    try:
        if not prompt or not prompt.strip():
            return "Error: prompt is required."
        logger.info("generate_image: prompt=%s size=%s", prompt[:80], size)
        client = await get_backend_tool_client()
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
        )
        return _format_generate_result(result)
    except Exception as e:
        logger.error("generate_image_tool error: %s", e)
        return f"Error: {str(e)}"


async def get_reference_image_tool(
    object_name: str,
    user_id: str = "system",
) -> str:
    """
    Check whether a reference image exists for a named object (e.g. from the user's images).
    Used to improve accuracy when the user wants a specific object in a generated image.
    Returns a message only; the actual reference is used by the backend when generating.

    Args:
        object_name: Name of the object (e.g. "Farmall tractor", "blue sedan").
        user_id: User ID (injected by engine if omitted).

    Returns:
        Message indicating whether a reference was found (for LLM context).
    """
    try:
        if not object_name or not object_name.strip():
            return "Error: object_name is required."
        logger.info("get_reference_image: object=%s", object_name[:80])
        client = await get_backend_tool_client()
        data = await client.get_reference_image_for_object(
            object_name=object_name.strip(),
            user_id=user_id,
        )
        if data:
            return f"Reference image found for '{object_name}'. You can describe this object in the prompt when generating."
        return f"No reference image found for '{object_name}'."
    except Exception as e:
        logger.error("get_reference_image_tool error: %s", e)
        return f"Error: {str(e)}"
