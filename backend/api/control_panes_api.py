"""
Control Panes API - CRUD and execute for user-defined status bar control panes.
Panes are wired to data source connectors; execute runs connector endpoints with pane credentials.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from models.api_models import AuthenticatedUserResponse
from utils.auth_middleware import get_current_user
from utils.websocket_manager import get_websocket_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Control Panes"])


# ---------- Pydantic models ----------


class ControlPaneCreate(BaseModel):
    """Request body for creating a control pane."""
    name: str = Field(..., min_length=1, max_length=255)
    icon: str = Field(default="Tune", max_length=100)
    pane_type: str = Field(default="connector", description="connector | artifact")
    connector_id: Optional[str] = Field(None, description="Data source connector UUID (required for connector panes)")
    artifact_id: Optional[str] = Field(None, description="Saved artifact UUID (required for artifact panes)")
    artifact_popover_width: Optional[int] = Field(None, ge=200, le=1200, description="Popover width in px for artifact panes")
    artifact_popover_height: Optional[int] = Field(None, ge=120, le=1600, description="Popover height in px for artifact panes")
    credentials_encrypted: Optional[Dict[str, Any]] = Field(default_factory=dict)
    connection_id: Optional[int] = Field(None, description="External connection ID for OAuth")
    controls: List[Dict[str, Any]] = Field(default_factory=list)
    is_visible: bool = True
    sort_order: int = 0
    refresh_interval: int = Field(default=0, ge=0, description="Auto-refresh interval in seconds; 0 = no polling")

    @field_validator("credentials_encrypted", mode="before")
    @classmethod
    def _credentials_create_to_dict(cls, v: Any) -> Any:
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v) if v.strip() else {}
            except json.JSONDecodeError:
                return {}
        return {}

    @model_validator(mode="after")
    def _validate_pane_type(self) -> "ControlPaneCreate":
        pt = (self.pane_type or "connector").strip().lower()
        if pt not in ("connector", "artifact"):
            raise ValueError("pane_type must be connector or artifact")
        object.__setattr__(self, "pane_type", pt)
        if pt == "artifact":
            if not self.artifact_id or not str(self.artifact_id).strip():
                raise ValueError("artifact_id is required for artifact panes")
        else:
            if not self.connector_id or not str(self.connector_id).strip():
                raise ValueError("connector_id is required for connector panes")
        return self


class ControlPaneUpdate(BaseModel):
    """Request body for partial update of a control pane."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    icon: Optional[str] = Field(None, max_length=100)
    pane_type: Optional[str] = None
    connector_id: Optional[str] = None
    artifact_id: Optional[str] = None
    artifact_popover_width: Optional[int] = Field(None, ge=200, le=1200)
    artifact_popover_height: Optional[int] = Field(None, ge=120, le=1600)
    credentials_encrypted: Optional[Dict[str, Any]] = None
    connection_id: Optional[int] = None
    controls: Optional[List[Dict[str, Any]]] = None
    is_visible: Optional[bool] = None
    sort_order: Optional[int] = None
    refresh_interval: Optional[int] = Field(None, ge=0, description="Auto-refresh interval in seconds; 0 = no polling")

    @field_validator("credentials_encrypted", mode="before")
    @classmethod
    def _credentials_update_to_dict(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v) if v.strip() else {}
            except json.JSONDecodeError:
                return {}
        return {}


class ControlPaneVisibilityUpdate(BaseModel):
    """Request body for toggling visibility."""
    is_visible: bool


class ControlPaneExecuteRequest(BaseModel):
    """Request body for executing a connector endpoint through a pane."""
    endpoint_id: str = Field(..., description="Endpoint id from connector definition")
    params: Dict[str, Any] = Field(default_factory=dict, description="Endpoint parameters")


class ControlPaneTestRequest(BaseModel):
    """Request body for testing a connector endpoint without a saved pane."""
    connector_id: str = Field(..., description="Data source connector UUID")
    endpoint_id: str = Field(..., description="Endpoint id from connector definition")
    params: Dict[str, Any] = Field(default_factory=dict, description="Endpoint parameters")
    credentials: Optional[Dict[str, Any]] = Field(None, description="Inline credentials for testing")
    connection_id: Optional[int] = Field(None, description="External connection ID for OAuth")


def _row_to_pane(row: Optional[Dict[str, Any]], connector_name: Optional[str] = None) -> Dict[str, Any]:
    """Convert DB row to API-friendly dict."""
    if not row:
        return {}
    definition = row.get("definition")
    if definition is not None and isinstance(definition, str):
        try:
            definition = json.loads(definition)
        except json.JSONDecodeError:
            definition = {}
    controls = row.get("controls")
    if controls is not None and isinstance(controls, str):
        try:
            controls = json.loads(controls)
        except json.JSONDecodeError:
            controls = []
    cid = row.get("connector_id")
    aid = row.get("artifact_id")
    out = {
        "id": str(row["id"]),
        "user_id": row.get("user_id"),
        "name": row.get("name", ""),
        "icon": row.get("icon", "Tune"),
        "pane_type": row.get("pane_type") or "connector",
        "connector_id": str(cid) if cid else None,
        "artifact_id": str(aid) if aid else None,
        "artifact_popover_width": row.get("artifact_popover_width"),
        "artifact_popover_height": row.get("artifact_popover_height"),
        "credentials_encrypted": row.get("credentials_encrypted") or {},
        "connection_id": row.get("connection_id"),
        "controls": controls or [],
        "is_visible": row.get("is_visible", True),
        "sort_order": row.get("sort_order", 0),
        "refresh_interval": row.get("refresh_interval", 0),
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
    }
    if connector_name is not None:
        out["connector_name"] = connector_name
    return out


# ---------- Endpoints ----------


@router.get("/control-panes")
async def list_control_panes(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """List all control panes for the current user, with connector name joined."""
    from services.database_manager.database_helpers import fetch_all

    rows = await fetch_all(
        """
        SELECT p.id, p.user_id, p.name, p.icon, p.pane_type, p.connector_id, p.artifact_id,
               p.artifact_popover_width, p.artifact_popover_height,
               p.credentials_encrypted, p.connection_id, p.controls, p.is_visible, p.sort_order, p.refresh_interval,
               p.created_at, p.updated_at, c.name AS connector_name
        FROM user_control_panes p
        LEFT JOIN data_source_connectors c ON c.id = p.connector_id
        WHERE p.user_id = $1
        ORDER BY p.sort_order ASC, p.name ASC
        """,
        current_user.user_id,
    )
    result = []
    for r in rows:
        result.append(_row_to_pane(dict(r), connector_name=r.get("connector_name")))
    return JSONResponse(content=result)


@router.post("/control-panes/test-endpoint")
async def test_control_pane_endpoint(
    body: ControlPaneTestRequest = ...,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """
    Test a connector endpoint without a saved pane.
    Uses inline credentials or connection_id for OAuth. Returns raw response for path discovery.
    """
    from services.database_manager.database_helpers import fetch_one
    from clients.connections_service_client import get_connections_service_client

    row = await fetch_one(
        "SELECT id, definition, connector_type FROM data_source_connectors WHERE id = $1 AND (user_id = $2 OR is_template = true)",
        body.connector_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Connector not found")

    definition = row.get("definition") or {}
    if isinstance(definition, str):
        try:
            definition = json.loads(definition)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid connector definition")

    credentials = body.credentials or {}
    oauth_token = None
    if body.connection_id is not None:
        from services.external_connections_service import external_connections_service
        oauth_token = await external_connections_service.get_valid_access_token(
            body.connection_id,
            rls_context={"user_id": current_user.user_id},
        )
        if not oauth_token:
            raise HTTPException(status_code=400, detail="Could not obtain token for the selected connection")
        conn = await external_connections_service.get_connection(
            body.connection_id,
            rls_context={"user_id": current_user.user_id},
        )
        if not conn or str(conn.get("user_id")) != str(current_user.user_id):
            raise HTTPException(status_code=404, detail="Connection not found")

    client = await get_connections_service_client()
    result = await client.execute_connector_endpoint(
        definition=definition,
        credentials=credentials,
        endpoint_id=body.endpoint_id,
        params=body.params,
        max_pages=1,
        oauth_token=oauth_token,
        raw_response=True,
        connector_type=row.get("connector_type"),
    )
    return JSONResponse(content=result)


@router.get("/control-panes/{pane_id}")
async def get_control_pane(
    pane_id: str = Path(..., description="Pane UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Get a single control pane by ID (user-owned only)."""
    from services.database_manager.database_helpers import fetch_one

    row = await fetch_one(
        """
        SELECT p.id, p.user_id, p.name, p.icon, p.pane_type, p.connector_id, p.artifact_id,
               p.artifact_popover_width, p.artifact_popover_height,
               p.credentials_encrypted, p.connection_id, p.controls, p.is_visible, p.sort_order, p.refresh_interval,
               p.created_at, p.updated_at, c.name AS connector_name
        FROM user_control_panes p
        LEFT JOIN data_source_connectors c ON c.id = p.connector_id
        WHERE p.id = $1 AND p.user_id = $2
        """,
        pane_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Control pane not found")
    return JSONResponse(content=_row_to_pane(dict(row), connector_name=row.get("connector_name")))


@router.post("/control-panes")
async def create_control_pane(
    body: ControlPaneCreate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Create a new control pane."""
    from services.database_manager.database_helpers import fetch_one, fetch_value

    pane_type = body.pane_type or "connector"
    credentials = body.credentials_encrypted or {}
    controls = body.controls or []
    controls_json = json.dumps(controls) if controls else "[]"
    credentials_json = json.dumps(credentials) if credentials else "{}"

    if pane_type == "artifact":
        art = await fetch_one(
            "SELECT id FROM saved_artifacts WHERE id = $1::uuid AND user_id = $2",
            body.artifact_id,
            current_user.user_id,
        )
        if not art:
            raise HTTPException(status_code=404, detail="Saved artifact not found")
        aw = body.artifact_popover_width
        ah = body.artifact_popover_height
        new_id = await fetch_value(
            """
            INSERT INTO user_control_panes
            (user_id, name, icon, pane_type, connector_id, artifact_id, artifact_popover_width, artifact_popover_height,
             credentials_encrypted, connection_id, controls, is_visible, sort_order, refresh_interval)
            VALUES ($1, $2, $3, $4, NULL, $5::uuid, $6, $7, $8::jsonb, NULL, '[]'::jsonb, $9, $10, $11)
            RETURNING id
            """,
            current_user.user_id,
            body.name,
            body.icon,
            pane_type,
            body.artifact_id,
            aw,
            ah,
            credentials_json,
            body.is_visible,
            body.sort_order,
            body.refresh_interval,
        )
    else:
        conn = await fetch_one(
            "SELECT id FROM data_source_connectors WHERE id = $1 AND (user_id = $2 OR is_template = true)",
            body.connector_id,
            current_user.user_id,
        )
        if not conn:
            raise HTTPException(status_code=404, detail="Connector not found")

        new_id = await fetch_value(
            """
            INSERT INTO user_control_panes
            (user_id, name, icon, pane_type, connector_id, artifact_id, artifact_popover_width, artifact_popover_height,
             credentials_encrypted, connection_id, controls, is_visible, sort_order, refresh_interval)
            VALUES ($1, $2, $3, $4, $5::uuid, NULL, NULL, NULL, $6::jsonb, $7, $8::jsonb, $9, $10, $11)
            RETURNING id
            """,
            current_user.user_id,
            body.name,
            body.icon,
            pane_type,
            body.connector_id,
            credentials_json,
            body.connection_id,
            controls_json,
            body.is_visible,
            body.sort_order,
            body.refresh_interval,
        )
    row = await fetch_one(
        """
        SELECT p.id, p.user_id, p.name, p.icon, p.pane_type, p.connector_id, p.artifact_id,
               p.artifact_popover_width, p.artifact_popover_height,
               p.credentials_encrypted, p.connection_id, p.controls, p.is_visible, p.sort_order, p.refresh_interval,
               p.created_at, p.updated_at, c.name AS connector_name
        FROM user_control_panes p
        LEFT JOIN data_source_connectors c ON c.id = p.connector_id
        WHERE p.id = $1
        """,
        str(new_id),
    )
    pane_id = str(new_id)
    try:
        ws_manager = get_websocket_manager()
        if ws_manager:
            await ws_manager.send_to_session(
                {"type": "control_pane_updated", "subtype": "pane_created", "pane_id": pane_id},
                current_user.user_id,
            )
    except Exception as ws_err:
        logger.warning("Control pane create WebSocket send failed: %s", ws_err)
    return JSONResponse(content=_row_to_pane(dict(row), connector_name=row.get("connector_name")))


@router.put("/control-panes/{pane_id}")
async def update_control_pane(
    pane_id: str,
    body: ControlPaneUpdate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Update a control pane (partial update)."""
    from services.database_manager.database_helpers import fetch_one, execute

    existing = await fetch_one(
        "SELECT id, connector_id FROM user_control_panes WHERE id = $1 AND user_id = $2",
        pane_id,
        current_user.user_id,
    )
    if not existing:
        raise HTTPException(status_code=404, detail="Control pane not found")

    if body.connector_id is not None:
        conn = await fetch_one(
            "SELECT id FROM data_source_connectors WHERE id = $1 AND (user_id = $2 OR is_template = true)",
            body.connector_id,
            current_user.user_id,
        )
        if not conn:
            raise HTTPException(status_code=404, detail="Connector not found")

    if body.artifact_id is not None:
        art = await fetch_one(
            "SELECT id FROM saved_artifacts WHERE id = $1::uuid AND user_id = $2",
            body.artifact_id,
            current_user.user_id,
        )
        if not art:
            raise HTTPException(status_code=404, detail="Saved artifact not found")

    updates = []
    args = []
    idx = 1
    if body.name is not None:
        updates.append(f"name = ${idx}")
        args.append(body.name)
        idx += 1
    if body.icon is not None:
        updates.append(f"icon = ${idx}")
        args.append(body.icon)
        idx += 1
    if body.pane_type is not None:
        pt = body.pane_type.strip().lower()
        if pt not in ("connector", "artifact"):
            raise HTTPException(status_code=400, detail="pane_type must be connector or artifact")
        updates.append(f"pane_type = ${idx}")
        args.append(pt)
        idx += 1
    if body.connector_id is not None:
        updates.append(f"connector_id = ${idx}::uuid")
        args.append(body.connector_id)
        idx += 1
    if body.artifact_id is not None:
        updates.append(f"artifact_id = ${idx}::uuid")
        args.append(body.artifact_id)
        idx += 1
    if body.artifact_popover_width is not None:
        updates.append(f"artifact_popover_width = ${idx}")
        args.append(body.artifact_popover_width)
        idx += 1
    if body.artifact_popover_height is not None:
        updates.append(f"artifact_popover_height = ${idx}")
        args.append(body.artifact_popover_height)
        idx += 1
    if body.credentials_encrypted is not None:
        updates.append(f"credentials_encrypted = ${idx}::jsonb")
        args.append(json.dumps(body.credentials_encrypted))
        idx += 1
    if body.connection_id is not None:
        updates.append(f"connection_id = ${idx}")
        args.append(body.connection_id)
        idx += 1
    if body.controls is not None:
        updates.append(f"controls = ${idx}::jsonb")
        args.append(json.dumps(body.controls))
        idx += 1
    if body.is_visible is not None:
        updates.append(f"is_visible = ${idx}")
        args.append(body.is_visible)
        idx += 1
    if body.sort_order is not None:
        updates.append(f"sort_order = ${idx}")
        args.append(body.sort_order)
        idx += 1
    if body.refresh_interval is not None:
        updates.append(f"refresh_interval = ${idx}")
        args.append(body.refresh_interval)
        idx += 1

    if not updates:
        row = await fetch_one(
            """
            SELECT p.id, p.user_id, p.name, p.icon, p.pane_type, p.connector_id, p.artifact_id,
                   p.artifact_popover_width, p.artifact_popover_height,
                   p.credentials_encrypted, p.connection_id, p.controls, p.is_visible, p.sort_order, p.refresh_interval,
                   p.created_at, p.updated_at, c.name AS connector_name
            FROM user_control_panes p
            LEFT JOIN data_source_connectors c ON c.id = p.connector_id
            WHERE p.id = $1 AND p.user_id = $2
            """,
            pane_id,
            current_user.user_id,
        )
        try:
            ws_manager = get_websocket_manager()
            if ws_manager:
                await ws_manager.send_to_session(
                    {"type": "control_pane_updated", "subtype": "pane_updated", "pane_id": pane_id},
                    current_user.user_id,
                )
        except Exception as ws_err:
            logger.warning("Control pane update WebSocket send failed: %s", ws_err)
        return JSONResponse(content=_row_to_pane(dict(row), connector_name=row.get("connector_name")))

    updates.append("updated_at = NOW()")
    args.extend([pane_id, current_user.user_id])
    await execute(
        f"UPDATE user_control_panes SET {', '.join(updates)} WHERE id = ${idx}::uuid AND user_id = ${idx + 1}",
        *args,
    )
    row = await fetch_one(
        """
        SELECT p.id, p.user_id, p.name, p.icon, p.pane_type, p.connector_id, p.artifact_id,
               p.artifact_popover_width, p.artifact_popover_height,
               p.credentials_encrypted, p.connection_id, p.controls, p.is_visible, p.sort_order, p.refresh_interval,
               p.created_at, p.updated_at, c.name AS connector_name
        FROM user_control_panes p
        LEFT JOIN data_source_connectors c ON c.id = p.connector_id
        WHERE p.id = $1 AND p.user_id = $2
        """,
        pane_id,
        current_user.user_id,
    )
    try:
        ws_manager = get_websocket_manager()
        if ws_manager:
            await ws_manager.send_to_session(
                {"type": "control_pane_updated", "subtype": "pane_updated", "pane_id": pane_id},
                current_user.user_id,
            )
    except Exception as ws_err:
        logger.warning("Control pane update WebSocket send failed: %s", ws_err)
    return JSONResponse(content=_row_to_pane(dict(row), connector_name=row.get("connector_name")))


@router.delete("/control-panes/{pane_id}")
async def delete_control_pane(
    pane_id: str = Path(..., description="Pane UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Delete a control pane."""
    from services.database_manager.database_helpers import execute

    await execute(
        "DELETE FROM user_control_panes WHERE id = $1 AND user_id = $2",
        pane_id,
        current_user.user_id,
    )
    try:
        ws_manager = get_websocket_manager()
        if ws_manager:
            await ws_manager.send_to_session(
                {"type": "control_pane_updated", "subtype": "pane_deleted", "pane_id": pane_id},
                current_user.user_id,
            )
    except Exception as ws_err:
        logger.warning("Control pane delete WebSocket send failed: %s", ws_err)
    return JSONResponse(content={"deleted": True, "id": pane_id})


@router.patch("/control-panes/{pane_id}/visibility")
async def update_control_pane_visibility(
    pane_id: str = Path(..., description="Pane UUID"),
    body: ControlPaneVisibilityUpdate = ...,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Toggle visibility of a control pane in the status bar."""
    from services.database_manager.database_helpers import fetch_one, execute

    await execute(
        "UPDATE user_control_panes SET is_visible = $1, updated_at = NOW() WHERE id = $2::uuid AND user_id = $3",
        body.is_visible,
        pane_id,
        current_user.user_id,
    )
    row = await fetch_one(
        "SELECT id, is_visible FROM user_control_panes WHERE id = $1 AND user_id = $2",
        pane_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Control pane not found")
    try:
        ws_manager = get_websocket_manager()
        if ws_manager:
            await ws_manager.send_to_session(
                {"type": "control_pane_updated", "subtype": "pane_visibility_changed", "pane_id": pane_id},
                current_user.user_id,
            )
    except Exception as ws_err:
        logger.warning("Control pane visibility WebSocket send failed: %s", ws_err)
    return JSONResponse(content={"id": str(row["id"]), "is_visible": row["is_visible"]})


@router.post("/control-panes/{pane_id}/execute")
async def execute_control_pane(
    pane_id: str = Path(..., description="Pane UUID"),
    body: ControlPaneExecuteRequest = ...,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """
    Execute a connector endpoint through this pane.
    Loads pane credentials (or OAuth via connection_id), runs the connector endpoint, returns result.
    """
    from services.database_manager.database_helpers import fetch_one
    from clients.connections_service_client import get_connections_service_client

    pane = await fetch_one(
        """
        SELECT id, pane_type, connector_id, credentials_encrypted, connection_id
        FROM user_control_panes WHERE id = $1 AND user_id = $2
        """,
        pane_id,
        current_user.user_id,
    )
    if not pane:
        raise HTTPException(status_code=404, detail="Control pane not found")

    if (pane.get("pane_type") or "connector") == "artifact":
        raise HTTPException(
            status_code=400,
            detail="Artifact control panes do not support connector execution",
        )

    connector = await fetch_one(
        "SELECT id, definition, connector_type FROM data_source_connectors WHERE id = $1",
        pane["connector_id"],
    )
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")

    definition = connector.get("definition") or {}
    if isinstance(definition, str):
        try:
            definition = json.loads(definition)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid connector definition")

    credentials = pane.get("credentials_encrypted") or {}
    if isinstance(credentials, str):
        try:
            credentials = json.loads(credentials)
        except json.JSONDecodeError:
            credentials = {}
    if not isinstance(credentials, dict):
        credentials = {}

    oauth_token = None
    connection_id = pane.get("connection_id")
    if connection_id is not None:
        from services.external_connections_service import external_connections_service
        oauth_token = await external_connections_service.get_valid_access_token(
            connection_id,
            rls_context={"user_id": current_user.user_id},
        )
        if not oauth_token:
            raise HTTPException(status_code=400, detail="Could not obtain token for the selected connection")

    auth_type = "oauth" if oauth_token else ("credentials" if credentials else "none")
    logger.info(
        "Control pane execute: pane_id=%s endpoint_id=%s auth=%s",
        pane_id,
        body.endpoint_id,
        auth_type,
    )

    client = await get_connections_service_client()
    result = await client.execute_connector_endpoint(
        definition=definition,
        credentials=credentials,
        endpoint_id=body.endpoint_id,
        params=body.params,
        max_pages=1,
        oauth_token=oauth_token,
        raw_response=True,
        connector_type=connector.get("connector_type"),
    )
    response_keys = list(result.keys()) if isinstance(result, dict) else []
    logger.info(
        "Control pane execute result: endpoint_id=%s response_keys=%s",
        body.endpoint_id,
        response_keys,
    )
    return JSONResponse(content=result)
