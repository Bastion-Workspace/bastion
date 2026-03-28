"""
Image generation via OpenRouter image-capable models.
Saves under uploads/web_sources/images and optionally promotes to document_metadata.
"""

import base64
import json
import os
import uuid
import logging
from io import BytesIO
from typing import Dict, Any, List, Optional

import httpx
from fastapi import UploadFile

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

    async def _promote_to_document(
        self,
        *,
        filename: str,
        image_bytes: bytes,
        user_id: str,
        folder_id: Optional[str],
        prompt: str,
        model: str,
        size_label: str,
        width: int,
        height: int,
    ) -> Optional[str]:
        """Create a document record for a generated image (user library). Returns document_id or None."""
        try:
            from services.document_service_v2 import DocumentService

            buf = BytesIO(image_bytes)
            buf.seek(0)
            upload = UploadFile(filename=filename, file=buf)
            doc_service = DocumentService()
            result = await doc_service.upload_and_process(
                upload,
                doc_type="image",
                user_id=user_id,
                folder_id=folder_id,
            )
            doc_id = result.document_id
            if not doc_id:
                logger.warning("upload_and_process returned no document_id for generated image %s", filename)
                return None
            meta = {
                "image_generation": {
                    "prompt": prompt,
                    "model": model,
                    "size": size_label,
                    "width": width,
                    "height": height,
                }
            }
            await doc_service.document_repository.update(
                doc_id,
                user_id=user_id,
                metadata_json=json.dumps(meta),
            )
            logger.info("Promoted generated image %s to document %s", filename, doc_id)
            return doc_id
        except Exception as e:
            logger.error("Failed to promote generated image to document: %s", e)
            return None

    async def generate_images(
        self,
        prompt: str,
        size: str = "1024x1024",
        fmt: str = "png",
        seed: Optional[int] = None,
        num_images: int = 1,
        negative_prompt: Optional[str] = None,
        model: Optional[str] = None,
        reference_image_data: Optional[bytes] = None,
        reference_image_url: Optional[str] = None,
        reference_strength: float = 0.5,
        user_id: Optional[str] = None,
        folder_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate images via OpenRouter image models.

        When user_id is set and not \"system\", each saved file is also registered as a user
        document (first-class workspace artifact) with generation metadata in metadata_json.
        """
        _ = reference_strength  # reserved for future vendor-specific image-to-image tuning
        try:
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

            client = await self._get_http_client()

            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": settings.SITE_URL,
                "X-Title": "Bastion Image Generation",
                "Content-Type": "application/json",
            }

            images_dir = await self._ensure_images_dir()

            url_primary = "https://openrouter.ai/api/v1/chat/completions"

            message_content: List[Dict[str, Any]] = []

            if reference_image_data:
                if isinstance(reference_image_data, str):
                    ref_image_b64 = reference_image_data
                else:
                    ref_image_b64 = base64.b64encode(reference_image_data).decode("utf-8")

                message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{ref_image_b64}"},
                    }
                )
            elif reference_image_url:
                message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": reference_image_url},
                    }
                )

            message_content.append({"type": "text", "text": prompt})

            chat_payload = {
                "model": model,
                "messages": [{"role": "user", "content": message_content}],
                "modalities": ["image", "text"],
            }
            response = await client.post(url_primary, json=chat_payload, headers=headers)
            if response.status_code >= 300:
                text_snippet = None
                try:
                    text_snippet = response.text[:200]
                except Exception:
                    text_snippet = None
                logger.warning("OpenRouter chat/completions returned %s: %s", response.status_code, text_snippet)
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

            try:
                data = response.json()
            except Exception:
                body_preview = None
                try:
                    body_preview = response.text[:200]
                except Exception:
                    pass
                raise ValueError(f"Unexpected non-JSON response from OpenRouter: {body_preview}")

            b64_images: List[str] = []
            data_urls: List[str] = []
            try:
                if isinstance(data, dict):
                    if "data" in data and isinstance(data.get("data"), list):
                        for item in (data.get("data") or []):
                            if isinstance(item, dict):
                                b64 = item.get("b64_json") or item.get("b64")
                                if b64:
                                    b64_images.append(b64)
                    if not b64_images and "output" in data and isinstance(data.get("output"), list):
                        for item in (data.get("output") or []):
                            if isinstance(item, dict) and (
                                item.get("type") == "image" or "b64" in item or "b64_json" in item
                            ):
                                b64 = item.get("b64_json") or item.get("b64")
                                if b64:
                                    b64_images.append(b64)
                    if not b64_images and "choices" in data:
                        for c in data.get("choices") or []:
                            msg = (c or {}).get("message") or {}
                            images = (msg or {}).get("images") or []
                            for img in images:
                                if isinstance(img, dict):
                                    if "image_url" in img and isinstance(img.get("image_url"), dict):
                                        url_val = img["image_url"].get("url")
                                        if isinstance(url_val, str) and url_val.startswith("data:image"):
                                            data_urls.append(url_val)
                                    b64 = img.get("b64_json") or img.get("b64")
                                    if b64:
                                        b64_images.append(b64)
            except Exception as ex:
                logger.warning("Image parse attempt failed: %s", ex)

            for durl in data_urls:
                try:
                    prefix = durl.split(",", 1)[0]
                    base64_part = durl.split(",", 1)[1]
                    if prefix.startswith("data:image"):
                        b64_images.append(base64_part)
                except Exception:
                    continue

            if not b64_images:
                raise ValueError("No images returned from OpenRouter response")

            max_n = max(1, min(num_images, 4))
            b64_images = b64_images[:max_n]

            size_label = f"{width}x{height}"
            saved: List[Dict[str, Any]] = []
            promote = bool(user_id and user_id.strip() and user_id.strip() != "system")

            for b64 in b64_images:
                image_bytes = base64.b64decode(b64)
                file_id = uuid.uuid4().hex
                filename = f"gen_{file_id}.{fmt.lower()}"
                abs_path = os.path.join(images_dir, filename)
                with open(abs_path, "wb") as f:
                    f.write(image_bytes)
                rel_path = f"/api/images/{filename}"
                entry: Dict[str, Any] = {
                    "filename": filename,
                    "path": abs_path,
                    "url": rel_path,
                    "width": width,
                    "height": height,
                    "format": fmt.lower(),
                }
                if promote:
                    doc_id = await self._promote_to_document(
                        filename=filename,
                        image_bytes=image_bytes,
                        user_id=user_id.strip(),
                        folder_id=folder_id,
                        prompt=prompt,
                        model=model,
                        size_label=size_label,
                        width=width,
                        height=height,
                    )
                    if doc_id:
                        entry["document_id"] = doc_id
                saved.append(entry)

            return {
                "success": True,
                "model": model,
                "prompt": prompt,
                "size": size_label,
                "format": fmt.lower(),
                "images": saved,
            }

        except Exception as e:
            logger.error("Image generation failed: %s", e)
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
