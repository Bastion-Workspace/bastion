"""
REST API for saved chat artifacts and public share/export.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

from models.api_models import AuthenticatedUserResponse
from models.saved_artifact_models import (
    PublicArtifactResponse,
    SavedArtifactCreate,
    SavedArtifactListResponse,
    SavedArtifactResponse,
    SavedArtifactShareResponse,
    SavedArtifactUpdate,
)
from services import saved_artifact_service as svc
from utils.auth_middleware import get_current_user

router = APIRouter(tags=["Saved Artifacts"])
public_router = APIRouter(tags=["Public Artifacts"])


def _request_base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


@router.post("/api/saved-artifacts", response_model=SavedArtifactResponse)
async def create_saved_artifact(
    body: SavedArtifactCreate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> SavedArtifactResponse:
    try:
        return await svc.create_saved_artifact(current_user.user_id, body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/api/saved-artifacts", response_model=SavedArtifactListResponse)
async def list_saved_artifacts(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> SavedArtifactListResponse:
    return await svc.list_saved_artifacts(current_user.user_id)


@router.get("/api/saved-artifacts/{artifact_id}", response_model=SavedArtifactResponse)
async def get_saved_artifact(
    artifact_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> SavedArtifactResponse:
    row = await svc.get_saved_artifact(current_user.user_id, artifact_id)
    if not row:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return row


@router.patch("/api/saved-artifacts/{artifact_id}", response_model=SavedArtifactResponse)
async def patch_saved_artifact(
    artifact_id: str,
    body: SavedArtifactUpdate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> SavedArtifactResponse:
    row = await svc.update_saved_artifact(current_user.user_id, artifact_id, body)
    if not row:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return row


@router.delete("/api/saved-artifacts/{artifact_id}", status_code=204)
async def delete_saved_artifact(
    artifact_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Response:
    ok = await svc.delete_saved_artifact(current_user.user_id, artifact_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return Response(status_code=204)


@router.post("/api/saved-artifacts/{artifact_id}/share", response_model=SavedArtifactShareResponse)
async def share_saved_artifact(
    artifact_id: str,
    request: Request,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> SavedArtifactShareResponse:
    base = _request_base_url(request)
    out = await svc.generate_share_token(current_user.user_id, artifact_id, base)
    if not out:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return out


@router.delete("/api/saved-artifacts/{artifact_id}/share", status_code=204)
async def unshare_saved_artifact(
    artifact_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Response:
    ok = await svc.revoke_share_token(current_user.user_id, artifact_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return Response(status_code=204)


@router.get("/api/saved-artifacts/{artifact_id}/export")
async def export_saved_artifact(
    artifact_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Response:
    pair = await svc.get_export_html(current_user.user_id, artifact_id)
    if not pair:
        raise HTTPException(status_code=404, detail="Artifact not found")
    html, filename = pair
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": "text/html; charset=utf-8",
    }
    return Response(content=html, media_type="text/html; charset=utf-8", headers=headers)


@public_router.get("/api/public/artifacts/{share_token}")
async def get_public_artifact(
    share_token: str,
    request: Request,
    format: Optional[str] = None,
) -> Response:
    art = await svc.get_artifact_by_share_token(share_token)
    if not art:
        raise HTTPException(status_code=404, detail="Not found")
    accept = (request.headers.get("accept") or "").lower()
    want_json = format == "json" or (
        "application/json" in accept and "text/html" not in accept
    )
    if want_json:
        return JSONResponse(art.model_dump(mode="json"))
    html = svc.build_standalone_export_html(art.artifact_type, art.title, art.code)
    return HTMLResponse(content=html)
