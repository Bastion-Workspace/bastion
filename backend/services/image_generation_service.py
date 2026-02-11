"""
Image Generation Service - Roosevelt's Artistic Artillery

Uses OpenRouter image-capable models (e.g., Gemini) to generate images from prompts
and stores them under `uploads/web_sources/images` for serving at `/static/images/*`.
"""

import base64
import os
import uuid
import logging
from typing import Dict, Any, List, Optional

import httpx

from config import settings
from services.settings_service import settings_service


logger = logging.getLogger(__name__)


class ImageGenerationService:
    """Service for generating images via OpenRouter and saving to disk."""

    def __init__(self):
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=60.0)
        return self._http_client

    async def _ensure_images_dir(self) -> str:
        images_dir = os.path.join(settings.UPLOAD_DIR, "web_sources", "images")
        os.makedirs(images_dir, exist_ok=True)
        return images_dir

    async def generate_images(
        self,
        prompt: str,
        size: str = "1024x1024",
        fmt: str = "png",
        seed: Optional[int] = None,
        num_images: int = 1,
        negative_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate images via OpenRouter image models.

        Args:
            prompt: Image generation prompt
            size: Image size (e.g., "1024x1024")
            fmt: Image format (png, jpg, etc.)
            seed: Optional random seed for reproducibility
            num_images: Number of images to generate (1-4)
            negative_prompt: Optional negative prompt
            model: Optional model override (uses user preference or falls back to settings)

        Returns a dict with metadata and list of files saved (relative and absolute URL paths).
        """
        try:
            # Use provided model, or fall back to settings
            if not model:
                model = await settings_service.get_image_generation_model()
            if not model:
                raise ValueError("Image generation model not configured. Set 'image_generation_model' in settings.")

            api_key = settings.OPENROUTER_API_KEY
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY is not set.")

            width, height = 1024, 1024
            try:
                if isinstance(size, str) and "x" in size:
                    w_str, h_str = size.lower().split("x", 1)
                    width, height = int(w_str), int(h_str)
            except Exception:
                width, height = 1024, 1024

            payload: Dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "size": f"{width}x{height}",
                "num_images": max(1, min(num_images, 4)),
                "response_format": "b64_json",
                "format": fmt,
            }
            if seed is not None:
                payload["seed"] = seed
            if negative_prompt:
                payload["negative_prompt"] = negative_prompt

            client = await self._get_http_client()

            # Attempt OpenRouter images endpoint first; fallback to responses API if needed
            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": settings.SITE_URL,
                "X-Title": "Bastion Image Generation",
                "Content-Type": "application/json",
            }

            images_dir = await self._ensure_images_dir()

            # Primary attempt per OpenRouter docs: chat/completions with modalities ["image","text"]
            url_primary = "https://openrouter.ai/api/v1/chat/completions"
            
            # Build message content - include reference image if provided
            message_content = []
            
            # Add reference image if provided
            if reference_image_data:
                # Decode base64 if needed
                if isinstance(reference_image_data, str):
                    ref_image_b64 = reference_image_data
                else:
                    ref_image_b64 = base64.b64encode(reference_image_data).decode('utf-8')
                
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{ref_image_b64}"
                    }
                })
            elif reference_image_url:
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": reference_image_url
                    }
                })
            
            # Add text prompt
            message_content.append({
                "type": "text",
                "text": prompt
            })
            
            chat_payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": message_content}
                ],
                "modalities": ["image", "text"],
                # Some providers accept additional config in vendor params; keep payload minimal per docs
            }
            response = await client.post(url_primary, json=chat_payload, headers=headers)
            if response.status_code >= 300:
                # Log snippet of text to aid debugging, then attempt generic responses fallback
                text_snippet = None
                try:
                    text_snippet = response.text[:200]
                except Exception:
                    text_snippet = None
                logger.warning(f"OpenRouter chat/completions returned {response.status_code}: {text_snippet}")
                url_fallback = "https://openrouter.ai/api/v1/responses"
                fallback_payload = {
                    "model": model,
                    "input": prompt,
                    "modalities": ["image", "text"],
                }
                response = await client.post(url_fallback, json=fallback_payload, headers=headers)
                if response.status_code >= 300:
                    snippet = None
                    try:
                        snippet = response.text[:200]
                    except Exception:
                        pass
                    raise ValueError(f"OpenRouter image generation failed: {response.status_code} {snippet}")

            # Attempt JSON parse; if it fails, log first bytes of body
            try:
                data = response.json()
            except Exception:
                body_preview = None
                try:
                    body_preview = response.text[:200]
                except Exception:
                    pass
                raise ValueError(f"Unexpected non-JSON response from OpenRouter: {body_preview}")

            # Normalize outputs to a list of base64 images
            b64_images: List[str] = []
            data_urls: List[str] = []
            # Known shapes: {data:[{b64_json:...}]} or {output:[{type:'image', b64_json:...}]}
            try:
                if isinstance(data, dict):
                    # Common shapes
                    if "data" in data and isinstance(data.get("data"), list):
                        for item in (data.get("data") or []):
                            if isinstance(item, dict):
                                b64 = item.get("b64_json") or item.get("b64")
                                if b64:
                                    b64_images.append(b64)
                    if not b64_images and "output" in data and isinstance(data.get("output"), list):
                        for item in (data.get("output") or []):
                            if isinstance(item, dict) and (item.get("type") == "image" or "b64" in item or "b64_json" in item):
                                b64 = item.get("b64_json") or item.get("b64")
                                if b64:
                                    b64_images.append(b64)
                    # Sometimes nested under choices/messages (LLM-style)
                    if not b64_images and "choices" in data:
                        for c in data.get("choices") or []:
                            msg = (c or {}).get("message") or {}
                            images = (msg or {}).get("images") or []
                            for img in images:
                                if isinstance(img, dict):
                                    # Preferred per docs: image_url.url -> data URL
                                    if "image_url" in img and isinstance(img.get("image_url"), dict):
                                        url_val = img["image_url"].get("url")
                                        if isinstance(url_val, str) and url_val.startswith("data:image"):
                                            data_urls.append(url_val)
                                    # Some providers may still return raw base64 fields
                                    b64 = img.get("b64_json") or img.get("b64")
                                    if b64:
                                        b64_images.append(b64)
            except Exception as ex:
                logger.warning(f"Image parse attempt failed: {ex}")

            # Convert any data URLs to bare base64
            for durl in data_urls:
                try:
                    prefix = durl.split(",", 1)[0]
                    base64_part = durl.split(",", 1)[1]
                    # Basic sanity check on prefix
                    if prefix.startswith("data:image"):
                        b64_images.append(base64_part)
                except Exception:
                    continue

            if not b64_images:
                raise ValueError("No images returned from OpenRouter response")

            saved: List[Dict[str, Any]] = []
            for b64 in b64_images:
                image_bytes = base64.b64decode(b64)
                file_id = uuid.uuid4().hex
                filename = f"gen_{file_id}.{fmt.lower()}"
                abs_path = os.path.join(images_dir, filename)
                with open(abs_path, "wb") as f:
                    f.write(image_bytes)
                # Use /api/images/ endpoint for proper content-type headers and authentication
                rel_path = f"/api/images/{filename}"
                saved.append({
                    "filename": filename,
                    "path": abs_path,
                    "url": rel_path,
                    "width": width,
                    "height": height,
                    "format": fmt.lower(),
                })

            return {
                "success": True,
                "model": model,
                "prompt": prompt,
                "size": f"{width}x{height}",
                "format": fmt.lower(),
                "images": saved,
            }

        except Exception as e:
            logger.error(f"❌ Image generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }


_image_generation_service: Optional[ImageGenerationService] = None


async def get_image_generation_service() -> ImageGenerationService:
    global _image_generation_service
    if _image_generation_service is None:
        _image_generation_service = ImageGenerationService()
    return _image_generation_service


