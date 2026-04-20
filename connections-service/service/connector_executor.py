"""
Connector Executor - Executes connector definitions (REST API, parameter substitution, pagination).

Runs in the connections-service as the single gateway for third-party API calls.
Definitions and credentials are passed from the backend via gRPC.
"""

import base64
import copy
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# Default timeout for connector HTTP requests (read). Slow or long-polling APIs (e.g. Snapcast) may need more.
DEFAULT_CONNECTOR_TIMEOUT = 60.0
CONNECTOR_CONNECT_TIMEOUT = 15.0

# Pattern for path/query substitution: {param_name}
_PARAM_PATTERN = re.compile(r"\{([^}]+)\}")


def _safe_url_for_log(url: str) -> str:
    """Return URL safe for logging: scheme + netloc + path only (no query or fragment)."""
    try:
        parsed = urlparse(url)
        if parsed.scheme and parsed.netloc:
            path = parsed.path or "/"
            return f"{parsed.scheme}://{parsed.netloc}{path}"
        return url.split("?")[0] if "?" in url else url
    except Exception:
        return "[invalid url]"


def _connection_error_message(url: str, exc: Exception) -> str:
    """Build a user-facing message for connection failures, with hints for localhost and DNS."""
    base = str(exc)
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").split(":")[0].lower()
        if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            return (
                f"{base}. The connector runs inside Docker; "
                "localhost/127.0.0.1 refers to the container, not your machine. "
                "Use your host's LAN IP, or on Docker Desktop add extra_hosts for host.docker.internal."
            )
        # DNS / host resolution failure (e.g. "Name or service not known", getaddrinfo failed)
        lower = base.lower()
        if "name or service not known" in lower or "nodename nor servname" in lower or "getaddrinfo" in lower:
            return (
                f"Host '{host}' could not be resolved. "
                "Ensure base_url is reachable from the server (use a hostname that resolves in your network or the server's DNS)."
            )
    except Exception:
        pass
    return base


def _substitute_params(template: str, params: Dict[str, Any]) -> str:
    """Replace {key} in template with params.get(key)."""
    if not template:
        return template
    result = template
    for key, value in params.items():
        if value is None:
            value = ""
        result = result.replace("{" + key + "}", str(value))
    return result


def _substitute_template(obj: Any, params: Dict[str, Any]) -> Any:
    """
    Recursively substitute {param_name} placeholders in a body template.
    When the entire value is a placeholder (e.g. "{percent}"), return the raw param value for correct JSON types.
    """
    if isinstance(obj, str):
        if len(obj) > 2 and obj.startswith("{") and obj.endswith("}"):
            key = obj[1:-1]
            if key in params:
                return params[key]
            return _substitute_params(obj, params)
        return _substitute_params(obj, params)
    if isinstance(obj, dict):
        return {k: _substitute_template(v, params) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_template(item, params) for item in obj]
    return obj


def _get_auth_headers(
    auth_config: Dict[str, Any],
    credentials: Dict[str, Any],
    oauth_token: Optional[str] = None,
) -> Dict[str, str]:
    """Build headers for auth from config and credentials. For oauth_connection, caller passes oauth_token."""
    if not auth_config:
        return {}
    auth_type = (auth_config.get("type") or "bearer").lower()
    if auth_type == "oauth_connection" and oauth_token:
        return {"Authorization": f"Bearer {oauth_token}"}

    if auth_type == "basic":
        if not credentials:
            return {}
        username_field = auth_config.get("username_field") or "username"
        password_field = auth_config.get("password_field") or "password"
        username = str(credentials.get(username_field) or "")
        password = str(credentials.get(password_field) or "")
        if not username:
            return {}
        basic_token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {basic_token}"}

    if auth_type == "jwt":
        if not credentials:
            return {}
        from service.jwt_auth import generate_jwt_token

        token = generate_jwt_token(auth_config, credentials)
        if not token:
            return {}
        header_prefix = auth_config.get("header_prefix") or "Bearer"
        return {"Authorization": f"{header_prefix} {token}"}

    if not credentials:
        return {}
    key_name = auth_config.get("credentials_key") or auth_config.get("header_key") or "api_key"
    value = credentials.get(key_name) or credentials.get("api_key") or ""
    if not value:
        return {}
    if auth_type == "bearer":
        return {"Authorization": f"Bearer {value}"}
    if auth_type == "api_key":
        if (auth_config.get("location") or "header").lower() == "query":
            return {}
        header = auth_config.get("header_name") or "X-API-Key"
        return {header: value}
    return {}


def _get_auth_query_params(auth_config: Dict[str, Any], credentials: Dict[str, Any]) -> Dict[str, str]:
    """Build query params for api_key auth when location is 'query' (e.g. Alpha Vantage)."""
    if not auth_config or not credentials:
        return {}
    if (auth_config.get("type") or "").lower() != "api_key":
        return {}
    if (auth_config.get("location") or "header").lower() != "query":
        return {}
    key_name = auth_config.get("credentials_key") or auth_config.get("header_key") or "api_key"
    value = credentials.get(key_name) or credentials.get("api_key") or ""
    if not value:
        return {}
    param_name = auth_config.get("param_name") or "apikey"
    return {param_name: value}


async def execute_endpoint(
    definition: Dict[str, Any],
    credentials: Dict[str, Any],
    endpoint_id: str,
    params: Optional[Dict[str, Any]] = None,
    max_pages: int = 5,
    oauth_token: Optional[str] = None,
    raw_response: bool = False,
) -> Dict[str, Any]:
    """
    Execute a single connector endpoint and return typed results.

    definition: connector definition (from data_source_connectors.definition).
        Expected shape: {
            "base_url": "https://api.example.com",
            "auth": { "type": "bearer", "credentials_key": "api_key" },
            "endpoints": {
                "search": {
                    "path": "/v1/search",
                    "method": "GET",
                    "params": [{"name": "q", "in": "query"}, {"name": "limit", "in": "query", "default": 10}],
                    "response_list_path": "results",
                    "pagination": { "type": "offset", "limit_param": "limit", "offset_param": "offset" }
                }
            }
        }
    credentials: decrypted credentials dict (e.g. {"api_key": "..."}).
    endpoint_id: key into definition["endpoints"].
    params: values for path/query params (substituted into path and sent as query/body).
    max_pages: max pagination pages to fetch (default 5).

    Returns: {"records": [...], "formatted": "human summary", "count": N, "error": optional }
    """
    params = params or {}
    base_url = (definition.get("base_url") or "").rstrip("/")
    auth_config = definition.get("auth") or {}
    endpoints = definition.get("endpoints") or {}
    if isinstance(endpoints, list):
        endpoints = {ep.get("id") or ep.get("name"): ep for ep in endpoints if ep.get("id") or ep.get("name")}

    endpoint_def = endpoints.get(endpoint_id) if isinstance(endpoints, dict) else None
    if not endpoint_def:
        return {"records": [], "count": 0, "formatted": "Endpoint not found", "error": f"Unknown endpoint: {endpoint_id}"}

    path = endpoint_def.get("path", "/")
    method = (endpoint_def.get("method") or "GET").upper()
    response_list_path = endpoint_def.get("response_list_path", "")
    pagination = endpoint_def.get("pagination") or {}
    body_template = endpoint_def.get("body_template")

    # Merge endpoint definition params (value/default) with request params so static params (e.g. function=GLOBAL_QUOTE) are sent
    def_params = endpoint_def.get("params") or []
    merged_params = {}
    for p in def_params:
        if not isinstance(p, dict):
            continue
        name = p.get("name")
        if not name:
            continue
        if "value" in p:
            merged_params[name] = p["value"]
        elif "default" in p:
            merged_params[name] = p["default"]
    merged_params.update(params or {})

    params_in_path = set(_PARAM_PATTERN.findall(path))
    next_params = {k: v for k, v in merged_params.items() if k not in params_in_path}
    auth_query = _get_auth_query_params(auth_config, credentials)
    next_params = {**next_params, **auth_query}

    timeout_seconds = endpoint_def.get("timeout_seconds") or definition.get("timeout_seconds")
    if timeout_seconds is None or timeout_seconds <= 0:
        timeout_seconds = DEFAULT_CONNECTOR_TIMEOUT
    timeout = httpx.Timeout(timeout_seconds, connect=CONNECTOR_CONNECT_TIMEOUT)

    url = base_url + _substitute_params(path, merged_params)
    headers = _get_auth_headers(auth_config, credentials, oauth_token=oauth_token)
    headers.setdefault("Accept", "application/json")
    for k, v in (definition.get("headers") or {}).items():
        if k and v is not None:
            headers[k] = str(v)
    for k, v in (endpoint_def.get("headers") or {}).items():
        if k and v is not None:
            headers[k] = str(v)

    all_records: List[Dict[str, Any]] = []
    raw_response_data: Optional[Dict[str, Any]] = None
    page = 0

    while page < max_pages:
        try:
            body_mode = (endpoint_def.get("body_mode") or "").lower()
            if body_mode == "multipart_file" and method != "GET":
                file_b64 = merged_params.get("file_base64") or merged_params.get("content_base64")
                filename = str(merged_params.get("filename") or "upload.bin")
                field_name = endpoint_def.get("multipart_field_name") or "file"
                content_type = str(merged_params.get("content_type") or "application/octet-stream")
                if not file_b64:
                    return {
                        "records": all_records,
                        "count": len(all_records),
                        "formatted": "Missing file_base64 (or content_base64) for multipart upload",
                        "error": "Missing file_base64",
                    }
                try:
                    raw_bytes = base64.b64decode(file_b64, validate=False)
                except Exception as e:
                    return {
                        "records": all_records,
                        "count": len(all_records),
                        "formatted": f"Invalid base64 file payload: {e}",
                        "error": str(e),
                    }
                multipart_headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}
                files = {field_name: (filename, raw_bytes, content_type)}
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.request(method, url, files=files, headers=multipart_headers)
            elif body_template and method != "GET":
                body = copy.deepcopy(body_template)
                body = _substitute_template(body, merged_params)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.request(method, url, json=body, headers=headers)
            else:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    if method == "GET":
                        resp = await client.get(url, params=next_params, headers=headers)
                    elif method == "POST":
                        resp = await client.post(url, json=next_params, headers=headers)
                    else:
                        resp = await client.request(method, url, params=next_params if method == "GET" else None, json=next_params if method != "GET" else None, headers=headers)
        except httpx.ConnectError as e:
            safe_url = _safe_url_for_log(url)
            logger.error(
                "Connector request failed (connection): %s -> %s",
                safe_url,
                e,
                exc_info=False,
            )
            msg = _connection_error_message(url, e)
            out = {"records": all_records, "count": len(all_records), "formatted": f"Request failed: {msg}", "error": msg}
            if raw_response and raw_response_data is not None:
                out["raw_response"] = raw_response_data
            return out
        except Exception as e:
            logger.exception("Connector request failed: %s -> %s", _safe_url_for_log(url), e)
            out = {"records": all_records, "count": len(all_records), "formatted": f"Request failed: {e}", "error": str(e)}
            if raw_response and raw_response_data is not None:
                out["raw_response"] = raw_response_data
            return out

        if resp.status_code >= 400:
            out = {"records": all_records, "count": len(all_records), "formatted": f"HTTP {resp.status_code}", "error": resp.text[:500]}
            if raw_response and raw_response_data is not None:
                out["raw_response"] = raw_response_data
            return out

        try:
            data = resp.json()
        except Exception as e:
            out = {"records": all_records, "count": len(all_records), "formatted": "Invalid JSON response", "error": str(e)}
            if raw_response and raw_response_data is not None:
                out["raw_response"] = raw_response_data
            return out

        if raw_response and raw_response_data is None:
            raw_response_data = data

        # Extract list from response (e.g. data.results or results)
        if not response_list_path or response_list_path.strip() in (".", ""):
            records = data if isinstance(data, list) else ([data] if data is not None else [])
        else:
            records = data
            for part in response_list_path.split("."):
                if part:
                    records = (records or {}).get(part) if isinstance(records, dict) else None
            if not isinstance(records, list):
                records = [data] if data is not None else []
        all_records.extend(records)

        # Pagination
        pag_type = pagination.get("type") or "none"
        if pag_type == "none" or not records:
            break
        if pag_type == "offset":
            limit_param = pagination.get("limit_param") or "limit"
            offset_param = pagination.get("offset_param") or "offset"
            limit = next_params.get(limit_param) or 10
            next_params[offset_param] = (next_params.get(offset_param) or 0) + len(records)
            if len(records) < limit:
                break
        elif pag_type == "page":
            page_param = pagination.get("page_param") or "page"
            next_params[page_param] = (next_params.get(page_param) or 1) + 1
            if len(records) == 0:
                break
        elif pag_type == "cursor":
            cursor_path = pagination.get("cursor_path") or "next_cursor"
            next_cursor = data
            for part in cursor_path.split("."):
                next_cursor = (next_cursor or {}).get(part) if isinstance(next_cursor, dict) else None
            if not next_cursor:
                break
            next_params[pagination.get("cursor_param") or "cursor"] = next_cursor
        else:
            break
        page += 1

    formatted = f"Retrieved {len(all_records)} record(s)."
    out = {"records": all_records, "count": len(all_records), "formatted": formatted}
    if raw_response and raw_response_data is not None:
        out["raw_response"] = raw_response_data
    return out


async def execute_connector(
    definition: Dict[str, Any],
    credentials: Dict[str, Any],
    endpoint_id: str,
    params: Optional[Dict[str, Any]] = None,
    max_pages: int = 5,
    oauth_token: Optional[str] = None,
    raw_response: bool = False,
) -> Dict[str, Any]:
    """
    Dispatch connector execution by definition.connector_type.
    REST and web_fetch use execute_endpoint; sftp, s3, and webdav use dedicated executors.
    """
    params = params or {}
    connector_type = (definition.get("connector_type") or "rest").lower()
    if connector_type in ("rest", "web_fetch"):
        return await execute_endpoint(
            definition,
            credentials,
            endpoint_id,
            params,
            max_pages=max_pages,
            oauth_token=oauth_token,
            raw_response=raw_response,
        )
    if connector_type == "sftp":
        from service.sftp_executor import execute_sftp_operation

        return await execute_sftp_operation(definition, credentials, endpoint_id, params)
    if connector_type == "s3":
        from service.s3_executor import execute_s3_operation

        return await execute_s3_operation(definition, credentials, endpoint_id, params)
    if connector_type == "webdav":
        from service.webdav_executor import execute_webdav_operation

        return await execute_webdav_operation(definition, credentials, endpoint_id, params)
    return {
        "records": [],
        "count": 0,
        "formatted": f"Unknown connector type: {connector_type}",
        "error": f"Unsupported connector_type: {connector_type}",
    }


# No response body size limit for API probe — return full body (non-LLM path).
# Consumers (e.g. UI or LLM callers) may cap when building prompts.
async def probe_api_endpoint(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Any] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Perform a raw HTTP request to the given URL. Used for API discovery (no connector definition).
    Returns status_code, response_headers, response_body, content_type. Full body is returned.
    """
    if not url or not url.strip():
        return {"success": False, "error": "URL is required"}
    method = (method or "GET").upper()
    headers = dict(headers or {})
    headers.setdefault("Accept", "application/json")
    try:
        timeout = httpx.Timeout(DEFAULT_CONNECTOR_TIMEOUT, connect=CONNECTOR_CONNECT_TIMEOUT)
        async with httpx.AsyncClient(timeout=timeout) as client:
            if method == "GET":
                resp = await client.get(url, params=params or None, headers=headers)
            elif method == "POST":
                resp = await client.post(url, json=body, params=params or None, headers=headers)
            elif method == "PUT":
                resp = await client.put(url, json=body, params=params or None, headers=headers)
            elif method == "PATCH":
                resp = await client.patch(url, json=body, params=params or None, headers=headers)
            elif method == "DELETE":
                resp = await client.delete(url, params=params or None, headers=headers)
            else:
                resp = await client.request(method, url, json=body, params=params or None, headers=headers)
    except httpx.ConnectError as e:
        logger.error(
            "Probe request failed (connection): %s -> %s",
            _safe_url_for_log(url),
            e,
            exc_info=False,
        )
        return {"success": False, "error": _connection_error_message(url, e)}
    except Exception as e:
        logger.exception("Probe request failed: %s", e)
        return {"success": False, "error": str(e)}

    response_headers = dict(resp.headers)
    content_type = response_headers.get("content-type", "").split(";")[0].strip()
    try:
        body_text = resp.text
    except Exception as e:
        body_text = f"[Could not decode body: {e}]"
    return {
        "success": True,
        "status_code": resp.status_code,
        "response_headers": response_headers,
        "response_body": body_text,
        "content_type": content_type,
    }
