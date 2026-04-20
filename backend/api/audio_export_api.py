"""
API for background document-to-audio export (Celery + voice-service).
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.document_api import check_document_access
from config import settings
from services.celery_app import celery_app
from services.celery_tasks.audio_export_tasks import export_document_audio_task
from services.document_text_file_reader import is_text_document_filename
from utils.auth_middleware import AuthenticatedUserResponse, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Audio export"])


class AudioExportStartRequest(BaseModel):
    document_id: str = Field(..., min_length=1)
    provider: Optional[str] = ""
    voice_id: Optional[str] = ""


@router.post("/api/audio-export")
async def start_audio_export(
    body: AudioExportStartRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Dict[str, Any]:
    doc = await check_document_access(body.document_id, current_user, "read")
    if not is_text_document_filename(getattr(doc, "filename", None)):
        raise HTTPException(
            status_code=400,
            detail="Only markdown, plain text, and org documents can be exported as audio",
        )
    task = export_document_audio_task.delay(
        body.document_id,
        current_user.user_id,
        provider=(body.provider or "").strip(),
        voice_id=(body.voice_id or "").strip(),
    )
    return {"task_id": task.id, "status": "pending"}


@router.get("/api/audio-export/{task_id}/status")
async def audio_export_status(
    task_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Dict[str, Any]:
    res = AsyncResult(task_id, app=celery_app)
    if res.state == "PENDING":
        return {"status": "pending", "task_id": task_id}

    if res.state == "PROGRESS":
        info = res.info if isinstance(res.info, dict) else {}
        return {
            "status": "progress",
            "task_id": task_id,
            "current_chunk": info.get("current_chunk"),
            "total_chunks": info.get("total_chunks"),
        }

    if res.state == "FAILURE":
        err = str(res.result) if res.result else "Task failed"
        return {"status": "failure", "task_id": task_id, "error": err}

    if res.state == "SUCCESS":
        data = res.result
        if not isinstance(data, dict):
            return {
                "status": "failure",
                "task_id": task_id,
                "error": "Invalid task result",
            }
        if data.get("user_id") and data.get("user_id") != current_user.user_id:
            raise HTTPException(status_code=403, detail="Not allowed to view this export")
        if not data.get("success"):
            return {
                "status": "failure",
                "task_id": task_id,
                "error": data.get("error", "Export failed"),
                "current_chunk": data.get("current_chunk"),
                "total_chunks": data.get("total_chunks"),
            }
        return {
            "status": "complete",
            "task_id": task_id,
            "total_chunks": data.get("total_chunks"),
        }

    return {"status": res.state.lower(), "task_id": task_id}


@router.get("/api/audio-export/{task_id}/download")
async def audio_export_download(
    task_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> FileResponse:
    res = AsyncResult(task_id, app=celery_app)
    if not res.successful():
        raise HTTPException(status_code=400, detail="Export is not ready or failed")

    data = res.result
    if not isinstance(data, dict) or not data.get("success"):
        raise HTTPException(status_code=400, detail="Export failed")

    if data.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not allowed to download this export")

    file_path = data.get("file_path")
    if not file_path:
        raise HTTPException(status_code=404, detail="Export file missing")

    path = Path(file_path).resolve()
    root = (Path(settings.EXPORTS_DIR).resolve() / "audio_exports")
    try:
        path.relative_to(root)
    except ValueError:
        logger.warning("Rejected audio export download path outside export dir: %s", path)
        raise HTTPException(status_code=403, detail="Invalid export path")

    if not path.is_file():
        raise HTTPException(status_code=404, detail="Export file no longer on disk")

    filename = data.get("download_filename") or "document.mp3"
    return FileResponse(
        path,
        filename=filename,
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-store"},
    )
