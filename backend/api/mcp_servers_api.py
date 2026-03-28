"""
REST API for user-configured MCP (Model Context Protocol) servers.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from models.api_models import AuthenticatedUserResponse
from services.database_manager.database_helpers import execute, fetch_all, fetch_one
from utils.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mcp-servers", tags=["mcp-servers"])


def _row_to_dict(row) -> Dict[str, Any]:
    if not row:
        return {}
    d = dict(row)
    for k in ("args", "env", "headers", "discovered_tools"):
        if k in d and d[k] is not None and not isinstance(d[k], (dict, list)):
            try:
                d[k] = json.loads(d[k]) if isinstance(d[k], str) else d[k]
            except (json.JSONDecodeError, TypeError):
                d[k] = [] if k in ("args", "discovered_tools") else {}
    # JSONB should be an array of tools; coerce mistaken objects so the UI can list them
    if isinstance(d.get("discovered_tools"), dict):
        d["discovered_tools"] = []
    return d


class McpServerCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    transport: str = Field(..., description="stdio | sse | streamable_http")
    url: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    headers: Dict[str, str] = Field(default_factory=dict)
    is_active: bool = True


class McpServerUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    transport: Optional[str] = None
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    headers: Optional[Dict[str, str]] = None
    is_active: Optional[bool] = None


@router.get("")
async def list_mcp_servers(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    rows = await fetch_all(
        """
        SELECT id, user_id, name, description, transport, url, command, args, env, headers,
               is_active, discovered_tools, last_discovery_at, created_at, updated_at
        FROM mcp_servers
        WHERE user_id = $1
        ORDER BY name
        """,
        current_user.user_id,
    )
    return JSONResponse(content=jsonable_encoder([_row_to_dict(r) for r in rows]))


@router.post("")
async def create_mcp_server(
    body: McpServerCreate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    t = body.transport.strip().lower()
    if t not in ("stdio", "sse", "streamable_http"):
        raise HTTPException(status_code=400, detail="transport must be stdio, sse, or streamable_http")
    if t == "stdio" and not (body.command or "").strip():
        raise HTTPException(status_code=400, detail="stdio requires command")
    if t in ("sse", "streamable_http") and not (body.url or "").strip():
        raise HTTPException(status_code=400, detail="sse/streamable_http requires url")

    try:
        await execute(
            """
            INSERT INTO mcp_servers (
                user_id, name, description, transport, url, command, args, env, headers, is_active
            ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9::jsonb, $10)
            """,
            current_user.user_id,
            body.name.strip(),
            body.description,
            t,
            body.url,
            body.command,
            json.dumps(body.args or []),
            json.dumps(body.env or {}),
            json.dumps(body.headers or {}),
            body.is_active,
        )
    except Exception as e:
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            raise HTTPException(status_code=409, detail="An MCP server with this name already exists") from e
        logger.exception("create_mcp_server failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    row = await fetch_one(
        """
        SELECT id, user_id, name, description, transport, url, command, args, env, headers,
               is_active, discovered_tools, last_discovery_at, created_at, updated_at
        FROM mcp_servers WHERE user_id = $1 AND name = $2
        """,
        current_user.user_id,
        body.name.strip(),
    )
    return JSONResponse(content=jsonable_encoder(_row_to_dict(row)))


@router.put("/{server_id}")
async def update_mcp_server(
    server_id: int,
    body: McpServerUpdate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    row = await fetch_one(
        "SELECT id FROM mcp_servers WHERE id = $1 AND user_id = $2",
        server_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="MCP server not found")

    cur = await fetch_one(
        """
        SELECT name, description, transport, url, command, args, env, headers, is_active
        FROM mcp_servers WHERE id = $1 AND user_id = $2
        """,
        server_id,
        current_user.user_id,
    )
    if not cur:
        raise HTTPException(status_code=404, detail="MCP server not found")

    name = body.name if body.name is not None else cur["name"]
    description = body.description if body.description is not None else cur.get("description")
    transport = (body.transport or cur["transport"] or "").strip().lower()
    url = body.url if body.url is not None else cur.get("url")
    command = body.command if body.command is not None else cur.get("command")
    args = body.args if body.args is not None else cur.get("args") or []
    env = body.env if body.env is not None else cur.get("env") or {}
    headers = body.headers if body.headers is not None else cur.get("headers") or {}
    is_active = body.is_active if body.is_active is not None else cur.get("is_active", True)

    if transport not in ("stdio", "sse", "streamable_http"):
        raise HTTPException(status_code=400, detail="invalid transport")
    if transport == "stdio" and not (str(command or "").strip()):
        raise HTTPException(status_code=400, detail="stdio requires command")
    if transport in ("sse", "streamable_http") and not (str(url or "").strip()):
        raise HTTPException(status_code=400, detail="sse/streamable_http requires url")

    await execute(
        """
        UPDATE mcp_servers SET
            name = $1, description = $2, transport = $3, url = $4, command = $5,
            args = $6::jsonb, env = $7::jsonb, headers = $8::jsonb, is_active = $9,
            updated_at = NOW()
        WHERE id = $10 AND user_id = $11
        """,
        name,
        description,
        transport,
        url,
        command,
        json.dumps(list(args) if isinstance(args, list) else []),
        json.dumps(dict(env) if isinstance(env, dict) else {}),
        json.dumps(dict(headers) if isinstance(headers, dict) else {}),
        is_active,
        server_id,
        current_user.user_id,
    )
    updated = await fetch_one(
        """
        SELECT id, user_id, name, description, transport, url, command, args, env, headers,
               is_active, discovered_tools, last_discovery_at, created_at, updated_at
        FROM mcp_servers WHERE id = $1
        """,
        server_id,
    )
    return JSONResponse(content=jsonable_encoder(_row_to_dict(updated)))


@router.delete("/{server_id}")
async def delete_mcp_server(
    server_id: int,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    existing = await fetch_one(
        "SELECT id FROM mcp_servers WHERE id = $1 AND user_id = $2",
        server_id,
        current_user.user_id,
    )
    if not existing:
        raise HTTPException(status_code=404, detail="MCP server not found")
    await execute(
        "DELETE FROM mcp_servers WHERE id = $1 AND user_id = $2",
        server_id,
        current_user.user_id,
    )
    return JSONResponse(content={"ok": True})


@router.post("/{server_id}/discover")
async def discover_mcp_server(
    server_id: int,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    row = await fetch_one(
        """
        SELECT id, user_id, name, description, transport, url, command, args, env, headers, is_active
        FROM mcp_servers WHERE id = $1 AND user_id = $2
        """,
        server_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="MCP server not found")

    from clients.tool_service_client import get_tool_service_client

    client = await get_tool_service_client()
    result = await client.discover_mcp_server(server_id, current_user.user_id)
    if not result.get("success"):
        detail = result.get("error") or "Discovery failed"
        logger.warning("discover_mcp_server via tools-service failed: %s", detail)
        raise HTTPException(status_code=400, detail=detail)

    tools = result.get("tools") or []
    if not isinstance(tools, list):
        tools = []

    await execute(
        """
        UPDATE mcp_servers SET discovered_tools = $1::jsonb, last_discovery_at = NOW(), updated_at = NOW()
        WHERE id = $2 AND user_id = $3
        """,
        json.dumps(tools),
        server_id,
        current_user.user_id,
    )
    return JSONResponse(content={"ok": True, "tools": tools, "count": len(tools)})


@router.post("/{server_id}/test")
async def test_mcp_server(
    server_id: int,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    row = await fetch_one(
        """
        SELECT id, user_id, name, description, transport, url, command, args, env, headers, is_active
        FROM mcp_servers WHERE id = $1 AND user_id = $2
        """,
        server_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="MCP server not found")

    from clients.tool_service_client import get_tool_service_client

    client = await get_tool_service_client()
    result = await client.discover_mcp_server(server_id, current_user.user_id)
    if not result.get("success"):
        err = result.get("error") or "Connection failed"
        logger.warning("test_mcp_server via tools-service failed: %s", err)
        return JSONResponse(
            status_code=200,
            content={"ok": False, "error": err},
        )

    tools = result.get("tools") or []
    if not isinstance(tools, list):
        tools = []

    await execute(
        """
        UPDATE mcp_servers SET discovered_tools = $1::jsonb, last_discovery_at = NOW(), updated_at = NOW()
        WHERE id = $2 AND user_id = $3
        """,
        json.dumps(tools),
        server_id,
        current_user.user_id,
    )
    return JSONResponse(
        content={
            "ok": True,
            "message": "Connected and listed tools",
            "tool_count": len(tools),
            "tool_names": [t.get("name") for t in tools if t.get("name")],
        }
    )


@router.get("/{server_id}")
async def get_mcp_server(
    server_id: int,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    row = await fetch_one(
        """
        SELECT id, user_id, name, description, transport, url, command, args, env, headers,
               is_active, discovered_tools, last_discovery_at, created_at, updated_at
        FROM mcp_servers WHERE id = $1 AND user_id = $2
        """,
        server_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return JSONResponse(content=jsonable_encoder(_row_to_dict(row)))
