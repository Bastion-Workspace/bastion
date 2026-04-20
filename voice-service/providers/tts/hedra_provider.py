"""
Hedra TTS provider (cloud). Async job: POST generation, poll status, download audio.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List

import httpx

from config.settings import settings
from providers.tts.base_tts_provider import (
    BaseTTSProvider,
    SynthesisResult,
    VoiceInfo,
)

logger = logging.getLogger(__name__)

HEDRA_PUBLIC_BASE = "https://api.hedra.com/web-app/public"
HEDRA_VOICES_PATH = "/voices"
HEDRA_GENERATIONS_PATH = "/generations"


class HedraProvider(BaseTTSProvider):
    """Hedra text-to-speech via public REST API."""

    def provider_name(self) -> str:
        return "hedra"

    def _effective_key(self, api_key: str) -> str:
        return (api_key or "").strip() or (settings.HEDRA_API_KEY or "")

    def _headers(self, api_key: str) -> Dict[str, str]:
        return {
            "X-API-Key": self._effective_key(api_key),
            "Content-Type": "application/json",
        }

    async def is_available(self, api_key: str = "", base_url: str = "") -> bool:
        del base_url
        key = self._effective_key(api_key)
        if not key:
            return False
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.get(
                    f"{HEDRA_PUBLIC_BASE}{HEDRA_VOICES_PATH}",
                    headers={"X-API-Key": key},
                )
                return r.status_code == 200
        except Exception as e:
            logger.warning("Hedra not available: %s", e)
            return False

    def _voice_labels_map(self, voice: Dict[str, Any]) -> Dict[str, str]:
        asset = voice.get("asset") or {}
        labels = asset.get("labels") or []
        out: Dict[str, str] = {}
        for item in labels:
            if isinstance(item, dict):
                name = str(item.get("name", "") or "").strip()
                val = str(item.get("value", "") or "").strip()
                if name:
                    out[name] = val
        return out

    async def list_voices(
        self, api_key: str = "", base_url: str = ""
    ) -> List[VoiceInfo]:
        del base_url
        key = self._effective_key(api_key)
        if not key:
            return []
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.get(
                f"{HEDRA_PUBLIC_BASE}{HEDRA_VOICES_PATH}",
                headers={"X-API-Key": key},
            )
            r.raise_for_status()
            data = r.json()
        if not isinstance(data, list):
            logger.warning("Hedra voices: unexpected response shape")
            return []
        voices: List[VoiceInfo] = []
        for v in data:
            if not isinstance(v, dict):
                continue
            vid = str(v.get("id") or "").strip()
            name = str(v.get("name") or "Unknown").strip()
            labels = self._voice_labels_map(v)
            gender = labels.get("gender", "neutral")
            lang = labels.get("language") or labels.get("accent") or "en"
            voices.append(
                VoiceInfo(
                    voice_id=vid,
                    name=name or "Unknown",
                    provider="hedra",
                    language=lang,
                    gender=gender,
                )
            )
        return voices

    async def _poll_until_complete(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        generation_id: str,
        max_seconds: float = 120.0,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + max_seconds
        delay = 1.0
        max_delay = 10.0
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Hedra generation {generation_id} did not complete within {max_seconds}s"
                )
            r = await client.get(
                f"{HEDRA_PUBLIC_BASE}{HEDRA_GENERATIONS_PATH}/{generation_id}/status",
                headers={"X-API-Key": api_key},
            )
            r.raise_for_status()
            body = r.json()
            status = str(body.get("status") or "").lower()
            if status == "error":
                msg = body.get("error_message") or body.get("error") or "Hedra generation failed"
                raise RuntimeError(str(msg))
            if status == "complete":
                return body
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)

    def _audio_format_from_headers(
        self, content_type: str, output_format: str
    ) -> str:
        ct = (content_type or "").split(";")[0].strip().lower()
        if "mpeg" in ct or ct == "audio/mp3":
            return "mp3"
        if "wav" in ct:
            return "wav"
        if "ogg" in ct or "opus" in ct:
            return "ogg"
        fmt = (output_format or "mp3").strip().lower()
        if fmt in ("mp3", "wav", "ogg", "opus"):
            return "opus" if fmt == "opus" else fmt
        return "mp3"

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        output_format: str,
        api_key: str = "",
        base_url: str = "",
        model_id: str = "",
    ) -> SynthesisResult:
        del base_url
        key = self._effective_key(api_key)
        if not key:
            raise ValueError("Hedra API key is required")
        vid = (voice_id or "").strip()
        if not vid:
            raise ValueError("Hedra requires a voice_id")

        mid = (model_id or "").strip()
        payload: Dict[str, Any] = {
            "type": "text_to_speech",
            "voice_id": vid,
            "text": text,
        }
        if mid:
            payload["model_id"] = mid

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=30.0)
        ) as client:
            r = await client.post(
                f"{HEDRA_PUBLIC_BASE}{HEDRA_GENERATIONS_PATH}",
                headers=self._headers(key),
                json=payload,
            )
            if not r.is_success:
                detail = r.text[:500] if r.text else r.status_code
                raise RuntimeError(f"Hedra generation start failed: {detail}")
            start = r.json()
            gen_id = str(start.get("id") or "").strip()
            if not gen_id:
                raise RuntimeError("Hedra did not return a generation id")

            final = await self._poll_until_complete(client, key, gen_id)
            url = (final.get("download_url") or "").strip()
            if not url:
                raise RuntimeError("Hedra completed but returned no download_url")

            dl = await client.get(url, follow_redirects=True)
            dl.raise_for_status()
            audio_bytes = dl.content

        audio_fmt = self._audio_format_from_headers(
            dl.headers.get("content-type") or "",
            output_format,
        )

        return SynthesisResult(
            audio_data=audio_bytes,
            audio_format=audio_fmt,
            duration_seconds=0.0,
        )
