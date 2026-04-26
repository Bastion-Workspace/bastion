import json
import logging
import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from models.api_models import AuthenticatedUserResponse
from services.database_manager.database_helpers import execute, fetch_all, fetch_one
from utils.auth_middleware import get_current_user
from utils.websocket_manager import get_websocket_manager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["code_workspace"])


def _settings_as_dict(raw: Any) -> Dict[str, Any]:
    """JSONB may come back as dict (asyncpg) or str depending on codec / Celery path."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            logger.warning("code_workspace settings: invalid JSON, ignoring")
            return {}
    return {}


def _last_file_tree_as_obj(raw: Any) -> Any:
    """JSONB last_file_tree may be dict or JSON string depending on DB codec / query path."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            logger.warning("code_workspace last_file_tree: invalid JSON, ignoring")
            return None
    return None


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


class CodeWorkspaceFileWriteRequest(BaseModel):
    """Write a file under a code workspace using a path relative to workspace root (as in file_tree)."""

    path: str = Field(..., min_length=1, max_length=4096)
    content: str = Field(default="")


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


def _resolve_device_for_workspace_record(
    ws_manager: Any,
    user_id: str,
    stored_device_id: Optional[str],
    stored_device_name: Optional[str],
) -> Optional[str]:
    """
    Map DB device_id / device_name to a currently connected proxy id.
    If the stored id no longer matches (e.g. proxy config changed) but exactly one device
    is connected, use that device and log a warning.
    """
    devices = ws_manager.get_user_devices(user_id)
    if not devices:
        return None
    conn_ids = [d.get("device_id") for d in devices if d.get("device_id")]
    conn_set = set(conn_ids)
    for cand in (stored_device_id, stored_device_name):
        if cand and str(cand).strip() and cand in conn_set:
            return cand
    if len(conn_ids) == 1:
        sole = conn_ids[0]
        logger.warning(
            "code_workspace: stored device_id=%r device_name=%r not among connected %s; "
            "using sole connected device %r (update workspace device_id in UI if needed)",
            stored_device_id,
            stored_device_name,
            conn_ids,
            sole,
        )
        return sole
    return (stored_device_id or stored_device_name) or None


def _join_parent_folder(parent: str, folder: str) -> str:
    p = parent.rstrip("/\\")
    fn = folder.strip()
    if "\\" in p:
        return f"{p}\\{fn}"
    return f"{p}/{fn}"


_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def _workspace_uses_windows_paths(workspace_root: str) -> bool:
    s = (workspace_root or "").strip()
    if not s:
        return False
    if s.startswith("\\\\"):
        return True
    if _WIN_DRIVE_RE.match(s):
        return True
    if s.count("\\") > s.count("/"):
        return True
    return False


def _normalize_relative_workspace_file_path(rel: str) -> str:
    """Return forward-slash relative path with no empty, '.', or '..' segments."""
    s = (rel or "").strip().replace("\\", "/")
    if not s:
        raise ValueError("File path is empty")
    if s.startswith("/"):
        raise ValueError("File path must be relative to the workspace root")
    parts = [p for p in s.split("/") if p != ""]
    for p in parts:
        if p == "." or p == "..":
            raise ValueError("Invalid path segment in file path")
    return "/".join(parts)


def _safe_join_workspace_path(workspace_root: str, relative_path: str) -> str:
    """
    Join DB workspace_path with a tree-relative path for the local proxy (absolute on device).
    Does not access the filesystem; only string rules to block traversal.
    """
    root = (workspace_root or "").strip()
    if not root:
        raise ValueError("Workspace has no root path")
    norm_rel = _normalize_relative_workspace_file_path(relative_path)
    win = _workspace_uses_windows_paths(root)
    root_pp: Union[PureWindowsPath, PurePosixPath]
    rel_pp: Union[PureWindowsPath, PurePosixPath]
    if win:
        root_pp = PureWindowsPath(root.rstrip("/\\"))
        rel_pp = PureWindowsPath(norm_rel)
    else:
        root_pp = PurePosixPath(root.rstrip("/\\"))
        rel_pp = PurePosixPath(norm_rel)
    joined = root_pp / rel_pp
    return str(joined)


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
        out = dict(row)
        out["settings"] = _settings_as_dict(out.get("settings"))
        out["last_file_tree"] = _last_file_tree_as_obj(out.get("last_file_tree"))
        return out
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get code workspace: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/code-workspaces/{workspace_id}/file")
async def get_code_workspace_file(
    workspace_id: str,
    path: str = Query(..., description="Path relative to workspace root (as in file_tree)"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Read a file from the device via read_file (absolute path derived server-side)."""
    try:
        ws = await fetch_one(
            """
            SELECT id, device_name, device_id, workspace_path
            FROM code_workspaces
            WHERE id = $1::uuid AND user_id = $2
            """,
            workspace_id,
            current_user.user_id,
            rls_context=_rls_context(current_user.user_id),
        )
        if not ws:
            raise HTTPException(status_code=404, detail="Code workspace not found")
        workspace_path = (ws.get("workspace_path") or "").strip()
        if not workspace_path:
            raise HTTPException(status_code=400, detail="Workspace has no workspace_path")
        try:
            absolute_path = _safe_join_workspace_path(workspace_path, path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        ws_manager = get_websocket_manager()
        device_target = _resolve_device_for_workspace_record(
            ws_manager,
            current_user.user_id,
            ws.get("device_id"),
            ws.get("device_name"),
        )
        if not device_target:
            raise HTTPException(
                status_code=400,
                detail="No matching local proxy connected for this workspace",
            )

        invoke_result = await ws_manager.invoke_device_tool(
            user_id=current_user.user_id,
            device_id=device_target,
            tool="read_file",
            args={"path": absolute_path},
            timeout=90,
        )
        if not invoke_result.get("success"):
            err = (invoke_result.get("error") or "").strip()
            fmt = (invoke_result.get("formatted") or "").strip()
            raise HTTPException(status_code=400, detail=err or fmt or "read_file failed")

        raw = invoke_result.get("result_json") or "{}"
        try:
            data = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except json.JSONDecodeError:
            raise HTTPException(status_code=502, detail="Invalid read_file response")
        return {
            "content": data.get("content", "") if isinstance(data, dict) else "",
            "size_bytes": int(data.get("size_bytes", 0)) if isinstance(data, dict) else 0,
            "path": data.get("path", absolute_path) if isinstance(data, dict) else absolute_path,
            "relative_path": _normalize_relative_workspace_file_path(path),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to read code workspace file: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/api/code-workspaces/{workspace_id}/file")
async def put_code_workspace_file(
    workspace_id: str,
    request: CodeWorkspaceFileWriteRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Write a file on the device via write_file (absolute path derived server-side)."""
    try:
        ws = await fetch_one(
            """
            SELECT id, device_name, device_id, workspace_path
            FROM code_workspaces
            WHERE id = $1::uuid AND user_id = $2
            """,
            workspace_id,
            current_user.user_id,
            rls_context=_rls_context(current_user.user_id),
        )
        if not ws:
            raise HTTPException(status_code=404, detail="Code workspace not found")
        workspace_path = (ws.get("workspace_path") or "").strip()
        if not workspace_path:
            raise HTTPException(status_code=400, detail="Workspace has no workspace_path")
        try:
            absolute_path = _safe_join_workspace_path(workspace_path, request.path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        ws_manager = get_websocket_manager()
        device_target = _resolve_device_for_workspace_record(
            ws_manager,
            current_user.user_id,
            ws.get("device_id"),
            ws.get("device_name"),
        )
        if not device_target:
            raise HTTPException(
                status_code=400,
                detail="No matching local proxy connected for this workspace",
            )

        invoke_result = await ws_manager.invoke_device_tool(
            user_id=current_user.user_id,
            device_id=device_target,
            tool="write_file",
            args={"path": absolute_path, "content": request.content, "append": False},
            timeout=90,
        )
        if not invoke_result.get("success"):
            err = (invoke_result.get("error") or "").strip()
            fmt = (invoke_result.get("formatted") or "").strip()
            raise HTTPException(status_code=400, detail=err or fmt or "write_file failed")

        return {
            "success": True,
            "path": absolute_path,
            "relative_path": _normalize_relative_workspace_file_path(request.path),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to write code workspace file: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


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
            merged = dict(_settings_as_dict(existing.get("settings")))
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

        settings = _settings_as_dict(ws.get("settings"))
        ignore_patterns = settings.get("ignore_patterns")
        max_depth = settings.get("max_depth", 10)
        include_hidden = bool(settings.get("include_hidden", False))

        workspace_path = (ws.get("workspace_path") or "").strip()
        if not workspace_path:
            raise HTTPException(status_code=400, detail="Workspace has no workspace_path; set a path before refreshing the tree")

        ws_manager = get_websocket_manager()
        device_target = _resolve_device_for_workspace_record(
            ws_manager,
            current_user.user_id,
            ws.get("device_id"),
            ws.get("device_name"),
        )

        invoke_result = await ws_manager.invoke_device_tool(
            user_id=current_user.user_id,
            device_id=device_target,
            tool="file_tree",
            args={
                "path": workspace_path,
                "max_depth": max_depth,
                "ignore_patterns": ignore_patterns,
                "include_hidden": include_hidden,
            },
            timeout=60,
        )

        if not invoke_result.get("success"):
            err = (invoke_result.get("error") or "").strip()
            fmt = (invoke_result.get("formatted") or "").strip()
            detail = err or fmt or "Failed to refresh tree"
            logger.warning(
                "refresh-tree failed workspace_id=%s user_id=%s detail=%s raw=%s",
                workspace_id,
                current_user.user_id,
                detail,
                invoke_result,
            )
            raise HTTPException(status_code=400, detail=detail)

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
