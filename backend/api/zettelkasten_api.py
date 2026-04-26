"""Zettelkasten settings and daily-note / wikilink-create APIs."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from models.zettelkasten_models import (
    ZettelkastenSettingsResponse,
    ZettelkastenSettingsUpdate,
)
from services.zettelkasten_settings_service import get_zettelkasten_settings_service
from services.zettelkasten_daily_note_service import get_zettelkasten_daily_note_service
from utils.auth_middleware import get_current_user, AuthenticatedUserResponse
from config import settings
from services.database_manager.database_helpers import fetch_one

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Zettelkasten"])


class CreateWikilinkNoteBody(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)


@router.get("/api/zettelkasten/settings", response_model=ZettelkastenSettingsResponse)
async def get_zk_settings(current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    svc = await get_zettelkasten_settings_service()
    s = await svc.get_settings(current_user.user_id)
    return ZettelkastenSettingsResponse(success=True, settings=s, message="ok")


@router.put("/api/zettelkasten/settings", response_model=ZettelkastenSettingsResponse)
async def put_zk_settings(
    body: ZettelkastenSettingsUpdate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        svc = await get_zettelkasten_settings_service()
        s = await svc.create_or_update_settings(current_user.user_id, body)
        return ZettelkastenSettingsResponse(success=True, settings=s, message="saved")
    except Exception as e:
        logger.error("zk settings save failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/api/zettelkasten/settings", response_model=ZettelkastenSettingsResponse)
async def delete_zk_settings(current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    svc = await get_zettelkasten_settings_service()
    await svc.delete_settings(current_user.user_id)
    s = await svc.get_settings(current_user.user_id)
    return ZettelkastenSettingsResponse(success=True, settings=s, message="reset")


@router.get("/api/zettelkasten/settings/locations")
async def zk_journal_locations(current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Directory tree under the user's library (same shape as org journal-locations)."""
    try:
        row = await fetch_one(
            "SELECT username FROM users WHERE user_id = $1", current_user.user_id
        )
        username = row["username"] if row else current_user.user_id
        upload_dir = Path(settings.UPLOAD_DIR)
        user_dir = upload_dir / "Users" / username

        from services import ds_upload_library_fs as dsf

        async def build_directory_tree(path: Path, relative_path: str = "") -> List[Dict[str, Any]]:
            directories: List[Dict[str, Any]] = []
            if not await dsf.is_dir(current_user.user_id, path):
                return directories
            try:
                for name in sorted(await dsf.list_dir_names(current_user.user_id, path)):
                    if name.startswith("."):
                        continue
                    item = path / name
                    if not await dsf.is_dir(current_user.user_id, item):
                        continue
                    item_relative = f"{relative_path}/{name}" if relative_path else name
                    children = await build_directory_tree(item, item_relative)
                    directories.append({"name": name, "path": item_relative, "children": children})
            except PermissionError:
                logger.warning("Permission denied accessing %s", path)
            return directories

        tree = await build_directory_tree(user_dir)
        return {
            "success": True,
            "directories": tree,
            "root_path": str(user_dir.relative_to(upload_dir)),
        }
    except Exception as e:
        logger.error("zk locations failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/zettelkasten/daily-note")
async def get_daily_note(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    settings_svc = await get_zettelkasten_settings_service()
    zk = await settings_svc.get_settings(current_user.user_id)
    daily_svc = await get_zettelkasten_daily_note_service()
    result = await daily_svc.get_or_create_daily_note(current_user.user_id, zk)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "failed"))
    return result


@router.post("/api/zettelkasten/create-wikilink-note")
async def create_wikilink_note(
    body: CreateWikilinkNoteBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    settings_svc = await get_zettelkasten_settings_service()
    zk = await settings_svc.get_settings(current_user.user_id)
    daily_svc = await get_zettelkasten_daily_note_service()
    result = await daily_svc.create_wikilink_note(current_user.user_id, zk, body.title)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "failed"))
    return result
