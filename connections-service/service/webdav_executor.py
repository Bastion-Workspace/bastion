"""
WebDAV connector executor — list, read, write, delete, mkdir via httpx.

Uses PROPFIND, GET, PUT, DELETE, MKCOL. HTTP Basic auth from credentials.
"""

import base64
import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from urllib.parse import quote, unquote, urljoin, urlparse

import httpx

from service.connector_executor import (
    CONNECTOR_CONNECT_TIMEOUT,
    DEFAULT_CONNECTOR_TIMEOUT,
    _substitute_params,
)

logger = logging.getLogger(__name__)


def _local_tag(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


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


def _build_resource_url(definition: Dict[str, Any], relative_path: str) -> str:
    base_url = (definition.get("base_url") or "").rstrip("/")
    base_path = (definition.get("base_path") or "").strip().strip("/")
    rel = (relative_path or "").strip().lstrip("/")
    segs: List[str] = []
    if base_path:
        segs.extend(s for s in base_path.split("/") if s)
    if rel:
        segs.extend(s for s in rel.split("/") if s)
    if not segs:
        return f"{base_url}/"
    path = "/".join(quote(s, safe="@+") for s in segs)
    return f"{base_url}/{path}"


def _propfind_parse(text: str, collection_url: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return records
    coll_parsed = urlparse(collection_url)
    coll_path = coll_parsed.path.rstrip("/") or "/"

    for el in root.iter():
        if _local_tag(el.tag) != "response":
            continue
        href_el = None
        is_collection = False
        size = 0
        modified = None
        for child in el:
            tag = _local_tag(child.tag)
            if tag == "href" and child.text:
                href_el = child.text.strip()
            elif tag == "propstat":
                for ps_ch in child:
                    if _local_tag(ps_ch.tag) != "prop":
                        continue
                    for p in ps_ch:
                        pt = _local_tag(p.tag)
                        if pt == "resourcetype":
                            for rt in p:
                                if _local_tag(rt.tag) == "collection":
                                    is_collection = True
                        elif pt == "getcontentlength" and (p.text and p.text.strip()):
                            try:
                                size = int(p.text.strip())
                            except ValueError:
                                size = 0
                        elif pt == "getlastmodified" and (p.text and p.text.strip()):
                            modified = p.text.strip()
        if not href_el:
            continue
        full = urljoin(collection_url, href_el)
        parsed = urlparse(full)
        path = unquote(parsed.path)
        if path.rstrip("/") == coll_path.rstrip("/"):
            continue
        name = path.rstrip("/").rsplit("/", 1)[-1] if path else href_el.rstrip("/").rsplit("/", 1)[-1]
        records.append(
            {
                "name": name,
                "path": path,
                "href": href_el,
                "is_dir": is_collection,
                "size": size,
                "modified": modified,
            }
        )
    return records


async def _mkcol_parents(client: httpx.AsyncClient, url: str, auth: Optional[tuple]) -> None:
    parsed = urlparse(url)
    segments = [s for s in parsed.path.split("/") if s]
    base = f"{parsed.scheme}://{parsed.netloc}"
    cur = ""
    for seg in segments[:-1]:
        cur = f"{cur}/{seg}"
        col_url = f"{base}{cur}/"
        try:
            r = await client.request("MKCOL", col_url, auth=auth)
            if r.status_code in (201, 405, 409):
                continue
        except Exception:
            pass


async def execute_webdav_operation(
    definition: Dict[str, Any],
    credentials: Dict[str, Any],
    endpoint_id: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    params = dict(params or {})
    base_url = (definition.get("base_url") or "").strip()
    if not base_url:
        return {
            "records": [],
            "count": 0,
            "formatted": "Missing base_url in definition",
            "error": "WebDAV connector requires base_url",
        }

    user = credentials.get("username") or credentials.get("user") or ""
    password = credentials.get("password") or ""
    auth = (user, password) if user else None

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
    path_template = endpoint_def.get("path") or ""
    merged = {**endpoint_def.get("defaults") or {}, **params}
    rel_path = _substitute_params(path_template, merged) if path_template else merged.get("remote_path", "")
    resource_url = _build_resource_url(definition, rel_path)

    timeout = httpx.Timeout(DEFAULT_CONNECTOR_TIMEOUT, connect=CONNECTOR_CONNECT_TIMEOUT)
    headers = {"User-Agent": "bastion-connections-webdav/1.0"}

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            if operation == "list":
                h = {**headers, "Depth": "1", "Content-Type": "application/xml"}
                body = '<?xml version="1.0"?><d:propfind xmlns:d="DAV:"><d:prop><d:resourcetype/><d:getcontentlength/><d:getlastmodified/></d:prop></d:propfind>'
                r = await client.request("PROPFIND", resource_url, content=body, headers=h, auth=auth)
                if r.status_code >= 400:
                    return {
                        "records": [],
                        "count": 0,
                        "formatted": r.text[:500],
                        "error": f"PROPFIND failed: HTTP {r.status_code}",
                    }
                records = _propfind_parse(r.text, resource_url)
                formatted = f"Listed {len(records)} item(s)"
                return {"records": records, "count": len(records), "formatted": formatted}
            if operation == "read":
                r = await client.get(resource_url, headers=headers, auth=auth)
                if r.status_code >= 400:
                    return {
                        "records": [],
                        "count": 0,
                        "formatted": r.text[:500],
                        "error": f"GET failed: HTTP {r.status_code}",
                    }
                data = r.content
                b64 = base64.b64encode(data).decode("ascii")
                rec = {
                    "path": rel_path,
                    "url": resource_url,
                    "content": b64,
                    "content_base64": b64,
                    "size": len(data),
                }
                return {"records": [rec], "count": 1, "formatted": f"Read {len(data)} byte(s)"}
            if operation == "write":
                data = _decode_write_payload(params)
                ct = params.get("content_type") or "application/octet-stream"
                await _mkcol_parents(client, resource_url, auth)
                r = await client.put(resource_url, content=data, headers={**headers, "Content-Type": ct}, auth=auth)
                if r.status_code >= 400:
                    return {
                        "records": [],
                        "count": 0,
                        "formatted": r.text[:500],
                        "error": f"PUT failed: HTTP {r.status_code}",
                    }
                rec = {"path": rel_path, "url": resource_url, "operation": "write", "success": True, "bytes_written": len(data)}
                return {"records": [rec], "count": 1, "formatted": f"Wrote {len(data)} byte(s)"}
            if operation == "delete":
                r = await client.delete(resource_url, headers=headers, auth=auth)
                if r.status_code >= 400:
                    return {
                        "records": [],
                        "count": 0,
                        "formatted": r.text[:500],
                        "error": f"DELETE failed: HTTP {r.status_code}",
                    }
                rec = {"path": rel_path, "url": resource_url, "operation": "delete", "success": True}
                return {"records": [rec], "count": 1, "formatted": f"Deleted {resource_url}"}
            if operation == "mkdir":
                await _mkcol_parents(client, resource_url, auth)
                col_url = resource_url if resource_url.endswith("/") else resource_url + "/"
                r = await client.request("MKCOL", col_url, headers=headers, auth=auth)
                if r.status_code >= 400 and r.status_code != 405:
                    return {
                        "records": [],
                        "count": 0,
                        "formatted": r.text[:500],
                        "error": f"MKCOL failed: HTTP {r.status_code}",
                    }
                rec = {"path": rel_path, "url": col_url, "operation": "mkdir", "success": True}
                return {"records": [rec], "count": 1, "formatted": f"Created collection {col_url}"}
            return {
                "records": [],
                "count": 0,
                "formatted": f"Unknown operation: {operation}",
                "error": f"Unsupported WebDAV operation: {operation}",
            }
    except Exception as e:
        logger.exception("WebDAV operation failed: %s", e)
        return {
            "records": [],
            "count": 0,
            "formatted": str(e),
            "error": str(e),
        }
