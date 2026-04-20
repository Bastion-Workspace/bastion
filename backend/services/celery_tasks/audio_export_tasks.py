"""
Background export of document text to a single MP3 via voice-service (chunked synthesis).
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from config import settings
from services.celery_app import celery_app
from services.celery_tasks.async_runner import run_async

logger = logging.getLogger(__name__)


def _safe_download_basename(filename: Optional[str], title: Optional[str]) -> str:
    base = (title or filename or "document").strip() or "document"
    base = re.sub(r"[^\w\s\-_.]+", "", base, flags=re.UNICODE)
    base = base.strip("._- ") or "document"
    if not base.lower().endswith(".mp3"):
        base = f"{base}.mp3"
    return base


async def _async_export_document_audio(
    task,
    document_id: str,
    user_id: str,
    provider_override: str = "",
    voice_id_override: str = "",
) -> Dict[str, Any]:
    from api.voice_api import SynthesizeRequest, _resolve_tts_synthesis_params
    from clients.voice_service_client import get_voice_service_client
    from services.document_text_file_reader import read_user_document_text
    from utils.text_for_speech import split_text_for_tts, strip_text_for_speech

    raw = await read_user_document_text(document_id, user_id)
    if raw is None:
        return {"success": False, "error": "Could not load document text", "document_id": document_id}

    from services.service_container import get_service_container

    container = await get_service_container()
    document_service = container.document_service
    doc_info = await document_service.get_document(document_id)
    fname = (doc_info.filename or "").lower() if doc_info else ""
    mode = "org" if fname.endswith(".org") else "markdown"
    speech_text = strip_text_for_speech(raw, mode)
    if not speech_text.strip():
        return {"success": False, "error": "No speakable text after sanitization", "document_id": document_id}

    chunks = split_text_for_tts(speech_text)
    total = len(chunks)
    if total == 0:
        return {"success": False, "error": "Empty chunks", "document_id": document_id}

    req = SynthesizeRequest(
        text=".",
        voice_id=(voice_id_override or "").strip(),
        provider=(provider_override or "").strip(),
        output_format="mp3",
    )
    voice_id, provider, api_key, base_url, model_id = (
        await _resolve_tts_synthesis_params(user_id, req)
    )

    client = await get_voice_service_client()
    combined = bytearray()

    for i, chunk in enumerate(chunks):
        task.update_state(
            state="PROGRESS",
            meta={
                "current_chunk": i + 1,
                "total_chunks": total,
                "status": "progress",
            },
        )
        seg = chunk.strip()
        if not seg:
            continue
        result = await client.synthesize(
            text=seg,
            voice_id=voice_id,
            provider=provider,
            output_format="mp3",
            api_key=api_key,
            base_url=base_url,
            model_id=model_id,
        )
        if result.get("error"):
            return {
                "success": False,
                "error": result["error"],
                "document_id": document_id,
                "current_chunk": i + 1,
                "total_chunks": total,
            }
        data = result.get("audio_data") or b""
        if not data:
            return {
                "success": False,
                "error": "Empty audio from voice service",
                "document_id": document_id,
                "current_chunk": i + 1,
                "total_chunks": total,
            }
        combined.extend(data)

    out_dir = Path(settings.EXPORTS_DIR) / "audio_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    task_id = task.request.id
    out_path = out_dir / f"{task_id}.mp3"
    out_path.write_bytes(combined)

    download_name = _safe_download_basename(
        doc_info.filename if doc_info else None,
        doc_info.title if doc_info else None,
    )

    return {
        "success": True,
        "document_id": document_id,
        "user_id": user_id,
        "file_path": str(out_path.resolve()),
        "download_filename": download_name,
        "total_chunks": total,
    }


@celery_app.task(bind=True, name="services.celery_tasks.audio_export_tasks.export_document_audio")
def export_document_audio_task(
    self,
    document_id: str,
    user_id: str,
    provider: str = "",
    voice_id: str = "",
) -> Dict[str, Any]:
    try:
        return run_async(
            _async_export_document_audio(
                self,
                document_id,
                user_id,
                provider_override=provider or "",
                voice_id_override=voice_id or "",
            )
        )
    except Exception as e:
        logger.exception("Audio export task failed: %s", e)
        return {"success": False, "error": str(e), "document_id": document_id}
