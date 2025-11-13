"""
Image Generation Tools - Roosevelt's Picture Forge

Async tool wrappers around ImageGenerationService for LangGraph agents.
"""

from typing import Optional

from services.image_generation_service import get_image_generation_service


async def generate_image(
    prompt: str,
    size: str = "1024x1024",
    format: str = "png",
    seed: Optional[int] = None,
    num_images: int = 1,
    negative_prompt: Optional[str] = None,
) -> str:
    """Generate images and return JSON string with image URLs and metadata."""
    svc = await get_image_generation_service()
    result = await svc.generate_images(
        prompt=prompt,
        size=size,
        fmt=format,
        seed=seed,
        num_images=num_images,
        negative_prompt=negative_prompt,
    )
    import json
    return json.dumps(result, ensure_ascii=False)


