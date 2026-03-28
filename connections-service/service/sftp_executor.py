"""
SFTP connector executor — list, read, write, delete, mkdir via asyncssh.

Definitions use connector_type \"sftp\", host, port, base_path, and endpoints with
operation + path template (e.g. \"{remote_path}\").
"""

import base64
import logging
import posixpath
import stat
from typing import Any, Dict, List, Optional

import asyncssh

from service.connector_executor import _substitute_params

logger = logging.getLogger(__name__)


def _full_remote_path(base_path: str, relative: str) -> str:
    base = (base_path or "").strip().rstrip("/")
    rel = (relative or "").strip().lstrip("/")
    if base and rel:
        return posixpath.normpath(f"{base}/{rel}")
    if base:
        return posixpath.normpath(base)
    if rel:
        return posixpath.normpath(f"/{rel}")
    return "/"


def _decode_write_payload(params: Dict[str, Any]) -> bytes:
    if params.get("content_base64") is not None:
        raw = params["content_base64"]
        if isinstance(raw, str):
            return base64.b64decode(raw)
        return bytes(raw)
    text = params.get("content_text")
    if text is not None:
        return str(text).encode("utf-8")
    raise ValueError("write requires content_base64 or content_text in params")


async def _ensure_parent_dirs(sftp: asyncssh.SFTPClient, remote_path: str) -> None:
    parent = posixpath.dirname(remote_path)
    if parent in ("", "/", "."):
        return
    await _ensure_parent_dirs(sftp, parent)
    try:
        await sftp.mkdir(parent)
    except (OSError, asyncssh.Error):
        pass


async def execute_sftp_operation(
    definition: Dict[str, Any],
    credentials: Dict[str, Any],
    endpoint_id: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    params = dict(params or {})
    endpoints = definition.get("endpoints") or {}
    if isinstance(endpoints, list):
        endpoints = {ep.get("id") or ep.get("name"): ep for ep in endpoints if ep.get("id") or ep.get("name")}
    endpoint_def = endpoints.get(endpoint_id) if isinstance(endpoints, dict) else None
    if not endpoint_def:
        return {
            "records": [],
            "count": 0,
            "formatted": "Endpoint not found",
            "error": f"Unknown endpoint: {endpoint_id}",
        }

    operation = (endpoint_def.get("operation") or "").lower()
    path_template = endpoint_def.get("path") or endpoint_def.get("key") or ""
    merged = {**endpoint_def.get("defaults") or {}, **params}
    rel_path = _substitute_params(path_template, merged) if path_template else merged.get("remote_path", "")
    base_path = definition.get("base_path") or ""
    remote_path = _full_remote_path(base_path, rel_path)

    host = definition.get("host") or ""
    port = int(definition.get("port") or 22)
    username = (credentials.get("username") or credentials.get("user") or "").strip()
    if not host or not username:
        return {
            "records": [],
            "count": 0,
            "formatted": "Missing host or username",
            "error": "SFTP requires host in definition and username in credentials",
        }

    password = credentials.get("password")
    private_key_pem = credentials.get("private_key") or credentials.get("private_key_pem")
    passphrase = credentials.get("passphrase") or None

    connect_kwargs: Dict[str, Any] = {
        "host": host,
        "port": port,
        "username": username,
        "known_hosts": None,
    }
    if private_key_pem:
        try:
            pem = private_key_pem.encode() if isinstance(private_key_pem, str) else private_key_pem
            key = asyncssh.import_private_key(pem, passphrase)
            connect_kwargs["client_keys"] = [key]
        except Exception as e:
            return {
                "records": [],
                "count": 0,
                "formatted": f"Invalid private key: {e}",
                "error": str(e),
            }
    elif password:
        connect_kwargs["password"] = password
    else:
        return {
            "records": [],
            "count": 0,
            "formatted": "No password or private_key in credentials",
            "error": "SFTP requires password or private_key in credentials",
        }

    try:
        async with asyncssh.connect(**connect_kwargs) as conn:
            async with conn.start_sftp_client() as sftp:
                if operation == "list":
                    records = await _sftp_list(sftp, remote_path)
                    formatted = f"Listed {len(records)} item(s) at {remote_path}"
                    return {"records": records, "count": len(records), "formatted": formatted}
                if operation == "read":
                    rec = await _sftp_read(sftp, remote_path)
                    formatted = f"Read {rec.get('size', 0)} byte(s) from {remote_path}"
                    return {"records": [rec], "count": 1, "formatted": formatted}
                if operation == "write":
                    data = _decode_write_payload(params)
                    await _ensure_parent_dirs(sftp, remote_path)
                    async with sftp.open(remote_path, "wb") as f:
                        await f.write(data)
                    rec = {"path": remote_path, "operation": "write", "success": True, "bytes_written": len(data)}
                    return {"records": [rec], "count": 1, "formatted": f"Wrote {len(data)} byte(s) to {remote_path}"}
                if operation == "delete":
                    await sftp.remove(remote_path)
                    rec = {"path": remote_path, "operation": "delete", "success": True}
                    return {"records": [rec], "count": 1, "formatted": f"Deleted {remote_path}"}
                if operation == "mkdir":
                    await _ensure_parent_dirs(sftp, remote_path)
                    try:
                        await sftp.mkdir(remote_path)
                    except (OSError, asyncssh.Error):
                        pass
                    rec = {"path": remote_path, "operation": "mkdir", "success": True}
                    return {"records": [rec], "count": 1, "formatted": f"Created directory {remote_path}"}
                return {
                    "records": [],
                    "count": 0,
                    "formatted": f"Unknown operation: {operation}",
                    "error": f"Unsupported SFTP operation: {operation}",
                }
    except Exception as e:
        logger.exception("SFTP operation failed: %s", e)
        return {
            "records": [],
            "count": 0,
            "formatted": str(e),
            "error": str(e),
        }


async def _sftp_list(sftp: asyncssh.SFTPClient, remote_path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        names = await sftp.readdir(remote_path)
    except FileNotFoundError:
        return []
    for entry in names:
        name = entry.filename
        if name in (".", ".."):
            continue
        full = posixpath.join(remote_path.rstrip("/"), name)
        perms = getattr(entry.attrs, "permissions", None) or 0
        is_dir = stat.S_ISDIR(perms)
        size = int(entry.attrs.size or 0)
        mtime = None
        if entry.attrs.mtime:
            mtime = int(entry.attrs.mtime)
        records.append(
            {
                "name": name,
                "path": full,
                "is_dir": is_dir,
                "size": size,
                "modified": mtime,
            }
        )
    return records


async def _sftp_read(sftp: asyncssh.SFTPClient, remote_path: str) -> Dict[str, Any]:
    async with sftp.open(remote_path, "rb") as f:
        data = await f.read()
    b64 = base64.b64encode(data).decode("ascii")
    return {
        "path": remote_path,
        "content": b64,
        "content_base64": b64,
        "size": len(data),
    }
