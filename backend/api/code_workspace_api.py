import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator

from models.api_models import AuthenticatedUserResponse
from services.database_manager.database_helpers import execute, fetch_all, fetch_one
from utils.auth_middleware import get_current_user
from utils.websocket_manager import get_websocket_manager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["code_workspace"])

class CodeWorkspaceCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    workspace_path: Optional[str] = Field(default=None, max_length=4096)
    parent_path: Optional[str] = Field(default=None, max_length=4096)
    folder_name: Optional[str] = Field(default=None, max_length=255)
    device_id: Optional[str] = Field(default=None, max_length=255)
    device_name: Optional[str] = Field(default=None, max_length=255)
    conversation_id: Optional[str] = Field(default=None)
    settings: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_path_spec(self):
        wp = (self.workspace_path or "").strip()
        pp = (self.parent_path or "").strip()
        fn = (self.folder_name or "").strip()
        if wp and (pp or fn):
            raise ValueError("Use either workspace_path or parent_path with folder_name, not both")
        if wp:
            self.workspace_path = wp
            return self
        if pp and fn:
            if "/" in fn or "\\" in fn or fn in (".", ".."):
                raise ValueError("folder_name must be a single path segment")
            self.parent_path = pp
            self.folder_name = fn
            self.workspace_path = None
            return self
        if pp or fn:
            raise ValueError("parent_path and folder_name are required together for new folder mode")
        raise ValueError("Provide workspace_path or parent_path and folder_name")


class CodeWorkspaceUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    workspace_path: Optional[str] = Field(default=None, min_length=1, max_length=4096)
    device_name: Optional[str] = Field(default=None, max_length=255)
    device_id: Optional[str] = Field(default=None, max_length=255)
    conversation_id: Optional[str] = Field(default=None)
    settings: Optional[Dict[str, Any]] = None


def _rls_context(user_id: str) -> Dict[str, str]:
    return {"user_id": user_id, "user_role": "user"}


def _resolve_device_id(ws_manager: Any, user_id: str, device_id: Optional[str]) -> str:
    devices: List[dict] = ws_manager.get_user_devices(user_id)
    if not devices:
        raise HTTPException(
            status_code=400,
            detail="No local proxy connected. Connect the Bastion local proxy and try again.",
        )
    if device_id:
        if not any(d.get("device_id") == device_id for d in devices):
            raise HTTPException(status_code=400, detail="device_id is not among connected devices")
        return device_id
    if len(devices) > 1:
        raise HTTPException(
            status_code=400,
            detail="Multiple devices connected; specify device_id",
        )
    return devices[0]["device_id"]


def _join_parent_folder(parent: str, folder: str) -> str:
    p = parent.rstrip("/\\")
    fn = folder.strip()
    if "\\" in p:
        return f"{p}\\{fn}"
    return f"{p}/{fn}"


def _final_path_and_mkdir(request: CodeWorkspaceCreateRequest) -> Tuple[str, bool]:
    wp = (request.workspace_path or "").strip()
    if wp:
        return wp, False
    return _join_parent_folder(request.parent_path, request.folder_name.strip()), True


@router.get("/api/code-workspaces/connected-devices")
async def list_connected_code_workspace_devices(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    ws_manager = get_websocket_manager()
    return {"devices": ws_manager.get_user_devices(current_user.user_id)}


@router.post("/api/code-workspaces")
async def create_code_workspace(
    request: CodeWorkspaceCreateRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        ws_manager = get_websocket_manager()
        target_device = _resolve_device_id(ws_manager, current_user.user_id, request.device_id)
        final_path, need_mkdir = _final_path_and_mkdir(request)

        if need_mkdir:
            mkdir_result = await ws_manager.invoke_device_tool(
                user_id=current_user.user_id,
                device_id=target_device,
                tool="create_directory",
                args={"path": final_path},
                timeout=60,
            )
            if not mkdir_result.get("success"):
                raise HTTPException(
                    status_code=400,
                    detail=mkdir_result.get("error") or mkdir_result.get("formatted") or "create_directory failed",
                )

        bind = await ws_manager.set_device_workspace(
            user_id=current_user.user_id,
            workspace_root=final_path,
            device_id=target_device,
            timeout=60,
        )
        if not bind.get("success"):
            raise HTTPException(
                status_code=502,
                detail=bind.get("error") or bind.get("formatted") or "Failed to set device workspace",
            )

        settings_payload = dict(request.settings or {})
        display_name = (request.device_name or "").strip() or None

        row = await fetch_one(
            """
            INSERT INTO code_workspaces (user_id, name, device_name, device_id, workspace_path, settings, conversation_id)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::uuid)
            RETURNING id, user_id, name, device_name, device_id, workspace_path, settings, conversation_id, created_at, updated_at
            """,
            current_user.user_id,
            request.name,
            display_name,
            target_device,
            final_path,
            json.dumps(settings_payload),
            request.conversation_id,
            rls_context=_rls_context(current_user.user_id),
        )
        if not row:
            raise HTTPException(status_code=500, detail="Failed to create code workspace")
        return row
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create code workspace: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/code-workspaces")
async def list_code_workspaces(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        rows = await fetch_all(
            """
            SELECT id, user_id, name, device_name, device_id, workspace_path, last_git_branch, conversation_id,
                   created_at, updated_at
            FROM code_workspaces
            WHERE user_id = $1
            ORDER BY updated_at DESC
            """,
            current_user.user_id,
            rls_context=_rls_context(current_user.user_id),
        )
        return {"workspaces": rows}
    except Exception as e:
        logger.error("Failed to list code workspaces: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/code-workspaces/{workspace_id}")
async def get_code_workspace(
    workspace_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        row = await fetch_one(
            """
            SELECT id, user_id, name, device_name, device_id, workspace_path, last_file_tree, last_git_branch,
                   settings, conversation_id, created_at, updated_at
            FROM code_workspaces
            WHERE id = $1::uuid AND user_id = $2
            """,
            workspace_id,
            current_user.user_id,
            rls_context=_rls_context(current_user.user_id),
        )
        if not row:
            raise HTTPException(status_code=404, detail="Code workspace not found")
        return row
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get code workspace: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/code-workspaces/{workspace_id}")
async def update_code_workspace(
    workspace_id: str,
    request: CodeWorkspaceUpdateRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        existing = await fetch_one(
            "SELECT id, settings, workspace_path, device_id, device_name FROM code_workspaces WHERE id = $1::uuid AND user_id = $2",
            workspace_id,
            current_user.user_id,
            rls_context=_rls_context(current_user.user_id),
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Code workspace not found")

        new_settings_json: Optional[str] = None
        if request.settings is not None:
            merged = dict(existing.get("settings") or {})
            merged.update(request.settings)
            new_settings_json = json.dumps(merged)

        await execute(
            """
            UPDATE code_workspaces
            SET
                name = COALESCE($3, name),
                device_name = COALESCE($4, device_name),
                device_id = COALESCE($5, device_id),
                workspace_path = COALESCE($6, workspace_path),
                conversation_id = COALESCE($7::uuid, conversation_id),
                settings = COALESCE($8::jsonb, settings),
                updated_at = NOW()
            WHERE id = $1::uuid AND user_id = $2
            """,
            workspace_id,
            current_user.user_id,
            request.name,
            request.device_name,
            request.device_id,
            request.workspace_path,
            request.conversation_id,
            new_settings_json,
            rls_context=_rls_context(current_user.user_id),
        )

        row = await get_code_workspace(workspace_id, current_user)

        rebind = request.workspace_path is not None or request.device_id is not None
        if rebind:
            ws_manager = get_websocket_manager()
            dev = row.get("device_id") or row.get("device_name")
            if not dev:
                raise HTTPException(status_code=400, detail="Workspace has no device_id; set device_id before changing path")
            bind = await ws_manager.set_device_workspace(
                user_id=current_user.user_id,
                workspace_root=row["workspace_path"],
                device_id=dev,
                timeout=60,
            )
            if not bind.get("success"):
                raise HTTPException(
                    status_code=502,
                    detail=bind.get("error") or bind.get("formatted") or "Failed to set device workspace",
                )

        return row
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update code workspace: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/code-workspaces/{workspace_id}")
async def delete_code_workspace(
    workspace_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        await execute(
            "DELETE FROM code_workspaces WHERE id = $1::uuid AND user_id = $2",
            workspace_id,
            current_user.user_id,
            rls_context=_rls_context(current_user.user_id),
        )
        return {"success": True}
    except Exception as e:
        logger.error("Failed to delete code workspace: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/code-workspaces/{workspace_id}/refresh-tree")
async def refresh_code_workspace_tree(
    workspace_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """
    Refresh the cached file tree by invoking the device's file_tree capability.
    This requires the local proxy daemon to be connected and to have file_tree enabled.
    """
    try:
        ws = await fetch_one(
            """
            SELECT id, device_name, device_id, workspace_path, settings
            FROM code_workspaces
            WHERE id = $1::uuid AND user_id = $2
            """,
            workspace_id,
            current_user.user_id,
            rls_context=_rls_context(current_user.user_id),
        )
        if not ws:
            raise HTTPException(status_code=404, detail="Code workspace not found")

        settings = ws.get("settings") or {}
        ignore_patterns = settings.get("ignore_patterns")
        max_depth = settings.get("max_depth", 10)
        include_hidden = bool(settings.get("include_hidden", False))

        device_target = ws.get("device_id") or ws.get("device_name")

        ws_manager = get_websocket_manager()
        invoke_result = await ws_manager.invoke_device_tool(
            user_id=current_user.user_id,
            device_id=device_target,
            tool="file_tree",
            args={
                "path": ws.get("workspace_path"),
                "max_depth": max_depth,
                "ignore_patterns": ignore_patterns,
                "include_hidden": include_hidden,
            },
            timeout=60,
        )

        if not invoke_result.get("success"):
            raise HTTPException(status_code=400, detail=invoke_result.get("error") or "Failed to refresh tree")

        tree_json = invoke_result.get("result_json") or "{}"
        tree_obj = json.loads(tree_json)

        await execute(
            """
            UPDATE code_workspaces
            SET last_file_tree = $3::jsonb, updated_at = NOW()
            WHERE id = $1::uuid AND user_id = $2
            """,
            workspace_id,
            current_user.user_id,
            json.dumps(tree_obj),
            rls_context=_rls_context(current_user.user_id),
        )

        return {"success": True, "file_tree": tree_obj, "formatted": invoke_result.get("formatted", "")}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to refresh code workspace tree: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
