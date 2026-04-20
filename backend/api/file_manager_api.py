"""
FileManager API - REST endpoints for centralized file management (document-service gRPC).
"""

import logging
from typing import Any, Dict, Optional, Type, TypeVar

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

TResponse = TypeVar("TResponse", bound=BaseModel)

from clients.document_service_client import get_document_service_client
from services.file_manager.models.file_placement_models import (
    FilePlacementRequest,
    FilePlacementResponse,
    FileMoveRequest,
    FileMoveResponse,
    FileDeleteRequest,
    FileDeleteResponse,
    FileRenameRequest,
    FileRenameResponse,
    FolderStructureRequest,
    FolderStructureResponse,
)
from utils.auth_middleware import get_current_user
from api.document_api import check_document_access
from models.api_models import AuthenticatedUserResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["File Manager"])


def _ds_user_id(current_user: Optional[AuthenticatedUserResponse]) -> str:
    if current_user is None:
        return ""
    return str(getattr(current_user, "user_id", "") or "")


def _unwrap_ds_response(
    ok: bool,
    data: Optional[Dict[str, Any]],
    err: Optional[str],
    response_model: Type[TResponse],
) -> TResponse:
    if not ok or data is None:
        raise HTTPException(status_code=500, detail=err or "Document service request failed")
    try:
        return response_model.model_validate(data)
    except Exception as e:
        logger.error("Invalid document-service response for %s: %s", response_model.__name__, e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/file-manager/place-file", response_model=FilePlacementResponse)
async def place_file(
    request: FilePlacementRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Place a file in the appropriate folder structure

    SECURITY: Regular users cannot create files in Global folders (read-only access).
    """
    try:
        if request.target_folder_id:
            from services.service_container import get_service_container

            container = await get_service_container()
            folder_service = container.folder_service

            folder = await folder_service.get_folder(request.target_folder_id, current_user.user_id)
            if not folder:
                raise HTTPException(status_code=404, detail="Folder not found or access denied")

            if folder.collection_type == "global" and current_user.role != "admin":
                raise HTTPException(
                    status_code=403,
                    detail="Creating files in Global folders requires Admin privileges",
                )

        if not request.user_id and current_user:
            request.user_id = current_user.user_id
        if not getattr(request, "current_user_role", None):
            request.current_user_role = current_user.role

        dsc = get_document_service_client()
        await dsc.initialize(required=True)
        payload = request.model_dump(mode="json")
        ok, data, err = await dsc.place_file_json(_ds_user_id(current_user), payload)
        resp = _unwrap_ds_response(ok, data, err, FilePlacementResponse)
        logger.info("File placed via API (document-service): %s", getattr(resp, "document_id", ""))
        return resp
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to place file via API: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/file-manager/move-file", response_model=FileMoveResponse)
async def move_file(
    request: FileMoveRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Move a file to a different folder"""
    try:
        from services.service_container import get_service_container

        container = await get_service_container()
        doc_info = await container.document_service.get_document(request.document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")

        collection_type = getattr(doc_info, "collection_type", "user")

        try:
            await check_document_access(request.document_id, current_user, "write")
        except HTTPException as e:
            if e.status_code == 403 and collection_type == "global":
                raise HTTPException(
                    status_code=403,
                    detail="Moving a Global file requires Admin privileges",
                ) from e
            raise

        if not request.user_id and current_user:
            request.user_id = current_user.user_id
            request.current_user_role = current_user.role

        dsc = get_document_service_client()
        await dsc.initialize(required=True)
        ok, data, err = await dsc.move_file_json(_ds_user_id(current_user), request.model_dump(mode="json"))
        resp = _unwrap_ds_response(ok, data, err, FileMoveResponse)
        logger.info("File moved via API (document-service): %s", getattr(resp, "document_id", ""))
        return resp
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to move file via API: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/file-manager/delete-file", response_model=FileDeleteResponse)
async def delete_file(
    request: FileDeleteRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Delete a file or folder"""
    try:
        from services.service_container import get_service_container

        container = await get_service_container()
        doc_info = await container.document_service.get_document(request.document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")

        collection_type = getattr(doc_info, "collection_type", "user")

        try:
            await check_document_access(request.document_id, current_user, "delete")
        except HTTPException as e:
            if e.status_code == 403 and collection_type == "global":
                raise HTTPException(
                    status_code=403,
                    detail="Deleting a Global file requires Admin privileges",
                ) from e
            raise

        if not request.user_id and current_user:
            request.user_id = current_user.user_id

        dsc = get_document_service_client()
        await dsc.initialize(required=True)
        ok, data, err = await dsc.delete_file_json(_ds_user_id(current_user), request.model_dump(mode="json"))
        resp = _unwrap_ds_response(ok, data, err, FileDeleteResponse)
        logger.info("File deleted via API (document-service): %s", getattr(resp, "document_id", ""))
        return resp
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete file via API: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/file-manager/rename-file", response_model=FileRenameResponse)
async def rename_file(
    request: FileRenameRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Rename a file"""
    try:
        from services.service_container import get_service_container

        container = await get_service_container()
        doc_info = await container.document_service.get_document(request.document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")

        collection_type = getattr(doc_info, "collection_type", "user")

        try:
            await check_document_access(request.document_id, current_user, "write")
        except HTTPException as e:
            if e.status_code == 403 and collection_type == "global":
                raise HTTPException(
                    status_code=403,
                    detail="Renaming a Global file requires Admin privileges",
                ) from e
            raise

        if not request.user_id and current_user:
            request.user_id = current_user.user_id if hasattr(current_user, "user_id") else str(current_user)

        dsc = get_document_service_client()
        await dsc.initialize(required=True)
        ok, data, err = await dsc.rename_file_json(_ds_user_id(current_user), request.model_dump(mode="json"))
        resp = _unwrap_ds_response(ok, data, err, FileRenameResponse)
        logger.info("File renamed via API (document-service): %s", getattr(resp, "document_id", ""))
        return resp
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to rename file via API: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/file-manager/create-folder-structure", response_model=FolderStructureResponse)
async def create_folder_structure(
    request: FolderStructureRequest,
    current_user: Optional[AuthenticatedUserResponse] = Depends(get_current_user),
):
    """Create a folder structure"""
    try:
        uid = _ds_user_id(current_user)
        if not request.user_id and uid:
            request.user_id = uid

        dsc = get_document_service_client()
        await dsc.initialize(required=True)
        ok, data, err = await dsc.create_folder_structure_json(
            uid or (request.user_id or ""),
            request.model_dump(mode="json"),
        )
        resp = _unwrap_ds_response(ok, data, err, FolderStructureResponse)
        logger.info("Folder structure created via API (document-service): %s", getattr(resp, "folder_id", ""))
        return resp
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create folder structure via API: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
