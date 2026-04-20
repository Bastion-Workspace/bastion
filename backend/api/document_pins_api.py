"""REST API for user document pins (Home dashboard + future document UI)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from models.api_models import AuthenticatedUserResponse
from models.document_pin_models import (
    DocumentPinCreateRequest,
    DocumentPinItem,
    DocumentPinsListResponse,
    DocumentPinReorderRequest,
)
from services import document_pins_service as pins_svc
from utils.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Document Pins"])


@router.get("/api/document-pins", response_model=DocumentPinsListResponse)
async def list_document_pins(
    include_preview: bool = Query(False, description="Include short description preview per pin"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> DocumentPinsListResponse:
    return await pins_svc.list_pins(current_user.user_id, include_preview=include_preview)


@router.post("/api/document-pins", response_model=DocumentPinItem)
async def create_document_pin(
    body: DocumentPinCreateRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> DocumentPinItem:
    try:
        return await pins_svc.add_pin(current_user.user_id, body)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.delete("/api/document-pins/{pin_id}")
async def remove_document_pin(
    pin_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> dict:
    try:
        await pins_svc.delete_pin(current_user.user_id, pin_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"success": True}


@router.put("/api/document-pins/reorder", response_model=DocumentPinsListResponse)
async def reorder_document_pins(
    body: DocumentPinReorderRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> DocumentPinsListResponse:
    try:
        return await pins_svc.reorder_pins(current_user.user_id, body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
