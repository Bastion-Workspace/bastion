"""
Data Connection Builder tools - analyze APIs, build and test connector definitions, bulk scrape URLs.

Zone 1: analyze_openapi_spec_tool, draft_connector_definition_tool, validate_connector_definition_tool (in-process).
Zone 2: probe_api_endpoint_tool, test_connector_endpoint_tool, create_data_connector_tool,
        bulk_scrape_urls_tool, get_bulk_scrape_status_tool (via gRPC backend).
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)

# Example connector definition schema (for draft_connector_definition output)
# Endpoint keys MUST be descriptive snake_case (e.g. server_get_status), never numeric (0, 1, 2).
# These keys are internal labels only — they select which config (path, method, body) to use; they are NOT sent to the API.
CONNECTOR_DEFINITION_SCHEMA = """
{
  "base_url": "https://api.example.com",
  "auth": { "type": "bearer"|"api_key"|"oauth_connection", "credentials_key": "api_key", "header_name": "X-API-Key" },
  "headers": { "Content-Type": "application/json" },
  "endpoints": {
    "endpoint_id": {
      "path": "/path/or/{param}",
      "method": "GET",
      "headers": { "X-Custom": "value" },
      "params": [{"name": "param", "in": "path"|"query"|"body", "description": "...", "required": true}],
      "body_template": { "fixed_field": "value", "variable": "{param_name}" },
      "response_list_path": "data.results",
      "pagination": { "type": "offset"|"page"|"cursor"|"none", "limit_param": "limit", "offset_param": "offset" }
    }
  }
}
"""


# ── I/O models ─────────────────────────────────────────────────────────────

class ProbeApiEndpointInputs(BaseModel):
    url: str = Field(description="Full URL to probe (e.g. https://api.example.com/v1/data)")
    method: str = Field(default="GET", description="HTTP method")


class ProbeApiEndpointParams(BaseModel):
    headers_json: Optional[str] = Field(default=None, description="Optional JSON object of headers")
    body_json: Optional[str] = Field(default=None, description="Optional JSON body for POST/PUT")
    params_json: Optional[str] = Field(default=None, description="Optional query params as JSON object")


class ProbeApiEndpointOutputs(BaseModel):
    status_code: int = Field(description="HTTP status code")
    response_headers: Dict[str, str] = Field(description="Response headers")
    response_body: str = Field(description="Response body")
    content_type: str = Field(description="Content-Type header value")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class AnalyzeOpenapiSpecInputs(BaseModel):
    spec_json: Optional[str] = Field(default=None, description="OpenAPI/Swagger JSON string (inline)")
    spec_url: Optional[str] = Field(default=None, description="URL to fetch OpenAPI spec from (if not inline)")


class AnalyzeOpenapiSpecOutputs(BaseModel):
    endpoints: List[Dict[str, Any]] = Field(description="List of endpoint definitions (path, method, params, etc.)")
    auth_schemes: List[Dict[str, Any]] = Field(description="Auth schemes from the spec")
    base_url: str = Field(description="Base URL or server URL from spec")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class DraftConnectorDefinitionInputs(BaseModel):
    api_description: str = Field(description="Description of the API: endpoints, auth, pagination")
    example_response: Optional[str] = Field(default=None, description="Optional raw API response to infer response_list_path")


class DraftConnectorDefinitionOutputs(BaseModel):
    definition: Dict[str, Any] = Field(description="Connector definition JSON (may be empty; use schema to fill)")
    name: str = Field(description="Suggested connector name")
    description: str = Field(description="Suggested description")
    requires_auth: bool = Field(description="Whether auth is required")
    auth_fields: List[Dict[str, Any]] = Field(description="Suggested auth field definitions")
    formatted: str = Field(description="Human-readable summary and schema for LLM/chat")


class ValidateConnectorDefinitionInputs(BaseModel):
    definition: Dict[str, Any] = Field(description="Connector definition to validate")


class ValidateConnectorDefinitionOutputs(BaseModel):
    is_valid: bool = Field(description="True if definition is valid")
    errors: List[str] = Field(description="Validation errors")
    warnings: List[str] = Field(description="Validation warnings")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class TestConnectorEndpointInputs(BaseModel):
    definition_json: str = Field(description="Full connector definition as JSON string")
    endpoint_id: str = Field(
        description="Endpoint id from the definition. Internal key that selects which endpoint config (path, method, body) to use — not sent to the API. Must be descriptive snake_case (e.g. server_get_status). Never use numeric IDs (0, 1, 2)."
    )
    params: Optional[Dict[str, Any]] = Field(default=None, description="Parameter values for the endpoint")
    credentials: Optional[Dict[str, Any]] = Field(default=None, description="Optional credentials for auth")


class TestConnectorEndpointOutputs(BaseModel):
    records: List[Dict[str, Any]] = Field(description="Extracted records from the API")
    count: int = Field(description="Number of records")
    raw_response: Optional[Dict[str, Any]] = Field(default=None, description="Full API response")
    formatted: str = Field(description="Human-readable summary for LLM/chat")
    value_path_candidates: List[str] = Field(
        default_factory=list,
        description="Dot-notation paths to arrays in the response; use as value_path or response_list_path",
    )
    scalar_paths: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Paths to scalar values [{path, value}] for display controls",
    )


class CreateDataConnectorInputs(BaseModel):
    name: str = Field(description="Connector display name")
    description: str = Field(default="", description="Connector description")
    definition: Dict[str, Any] = Field(
        description=(
            "Connector definition (base_url, auth, endpoints). "
            "Endpoint keys are internal labels that select which config (path, method, body) to use — they are not sent to the API. "
            "Use descriptive snake_case (e.g. server_get_status). Never numeric (0, 1, 2). "
            "Call test_connector_endpoint first and use value_path_candidates to set response_list_path and control pane value_path."
        )
    )
    requires_auth: bool = Field(default=False, description="Whether the connector requires credentials")
    auth_fields: Optional[List[Dict[str, Any]]] = Field(default=None, description="Auth field definitions")
    category: Optional[str] = Field(default=None, description="Category label")
    confirmed: bool = Field(default=False, description="If false, preview only; if true, save to DB")


class CreateDataConnectorOutputs(BaseModel):
    connector_id: str = Field(description="Created connector UUID")
    name: str = Field(description="Connector name")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class ListDataConnectorsInputs(BaseModel):
    pass


class ListDataConnectorsOutputs(BaseModel):
    connectors: List[Dict[str, Any]] = Field(description="List of connectors (id, name, description, connector_type, endpoint_count, created_at, updated_at)")
    count: int = Field(description="Number of connectors")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class GetDataConnectorDetailInputs(BaseModel):
    """Inputs for get_data_connector_detail_tool."""
    connector_id: str = Field(description="Data connector UUID to fetch (full definition, endpoints; auth values redacted)")


class GetDataConnectorDetailOutputs(BaseModel):
    """Outputs for get_data_connector_detail_tool."""
    connector_id: str = Field(description="Connector UUID")
    name: str = Field(description="Connector name")
    description: Optional[str] = Field(default=None, description="Connector description")
    connector_type: str = Field(description="Connector type (e.g. rest)")
    definition: Dict[str, Any] = Field(description="Full definition including endpoints (auth values redacted)")
    endpoint_count: int = Field(description="Number of endpoints")
    is_locked: bool = Field(description="Whether connector is locked")
    category: Optional[str] = Field(default=None, description="Category")
    tags: List[str] = Field(description="Tags")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class UpdateDataConnectorInputs(BaseModel):
    connector_id: str = Field(description="Connector UUID to update")
    confirmed: bool = Field(default=False, description="If false, preview only; if true, apply update")


class UpdateDataConnectorParams(BaseModel):
    name: Optional[str] = Field(default=None, description="New connector name")
    description: Optional[str] = Field(default=None, description="New description")
    connector_type: Optional[str] = Field(default=None, description="Connector type")
    definition: Optional[Dict[str, Any]] = Field(default=None, description="Full connector definition JSON")
    requires_auth: Optional[bool] = Field(default=None, description="Whether connector requires credentials")
    auth_fields: Optional[List[Dict[str, Any]]] = Field(default=None, description="Auth field definitions")


class UpdateDataConnectorOutputs(BaseModel):
    connector_id: str = Field(description="Updated connector UUID")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class BulkScrapeUrlsInputs(BaseModel):
    urls: List[str] = Field(description="List of URLs to scrape")
    extract_images: bool = Field(default=True, description="Extract image URLs from pages")
    download_images: bool = Field(default=True, description="Download and store images")
    image_output_folder: Optional[str] = Field(default=None, description="Subfolder for downloaded images")
    metadata_fields: Optional[List[str]] = Field(default=None, description="Metadata field names to capture")


class BulkScrapeUrlsParams(BaseModel):
    max_concurrent: int = Field(default=10, description="Max concurrent requests per batch")
    rate_limit_seconds: float = Field(default=1.0, description="Delay between requests")
    folder_id: Optional[str] = Field(default=None, description="Target folder ID for stored documents")


class BulkScrapeUrlsOutputs(BaseModel):
    task_id: str = Field(description="Celery task ID when backgrounded (20+ URLs)")
    results: List[Dict[str, Any]] = Field(description="Page results when inline")
    count: int = Field(description="Number of URLs crawled")
    images_found: int = Field(description="Total image URLs found")
    images_downloaded: int = Field(description="Images downloaded and stored")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class GetBulkScrapeStatusInputs(BaseModel):
    task_id: str = Field(description="Celery task ID from bulk_scrape_urls")


class GetBulkScrapeStatusOutputs(BaseModel):
    status: str = Field(description="PENDING, RUNNING, SUCCESS, FAILURE")
    progress_current: int = Field(description="Current step")
    progress_total: int = Field(description="Total steps")
    progress_message: str = Field(description="Status message")
    results: List[Dict[str, Any]] = Field(description="Partial or final results")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


# ── Zone 1: In-process tools ───────────────────────────────────────────────

def _parse_openapi_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Extract endpoints and auth from OpenAPI 3.x or Swagger 2 spec."""
    endpoints = []
    auth_schemes = []
    base_url = ""
    servers = spec.get("servers") or []
    if servers and isinstance(servers[0], dict):
        base_url = servers[0].get("url", "")
    if not base_url and spec.get("host"):
        base_url = (spec.get("schemes", ["https"])[0] if spec.get("schemes") else "https") + "://" + spec.get("host", "")
    paths = spec.get("paths") or {}
    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        for method in ["get", "post", "put", "patch", "delete"]:
            op = path_item.get(method)
            if not op:
                continue
            params = [p for p in (op.get("parameters") or []) if isinstance(p, dict)]
            endpoints.append({
                "path": path,
                "method": method.upper(),
                "summary": op.get("summary") or op.get("description", ""),
                "parameters": [{"name": p.get("name"), "in": p.get("in"), "required": p.get("required")} for p in params],
            })
    sec = spec.get("components", {}).get("securitySchemes") or spec.get("securityDefinitions") or {}
    for name, scheme in (sec.items() if isinstance(sec, dict) else []):
        if isinstance(scheme, dict):
            auth_schemes.append({"name": name, "type": scheme.get("type"), "in": scheme.get("in")})
    return {"endpoints": endpoints, "auth_schemes": auth_schemes, "base_url": base_url}


async def analyze_openapi_spec_tool(
    spec_json: Optional[str] = None,
    spec_url: Optional[str] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Parse OpenAPI/Swagger spec and extract endpoints and auth. Provide spec_json or spec_url."""
    try:
        if spec_url and not spec_json:
            client = await get_backend_tool_client()
            results = await client.crawl_web_content(url=spec_url, urls=[spec_url])
            if results:
                import re
                raw = results[0].get("content", "") or results[0].get("html", "")
                m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
                spec_json = m.group(1).strip() if m else (raw.strip() if raw.strip().startswith("{") else "")
            if not spec_json:
                return {"endpoints": [], "auth_schemes": [], "base_url": "", "formatted": "Could not fetch or find JSON in spec URL."}
        if not spec_json:
            return {"endpoints": [], "auth_schemes": [], "base_url": "", "formatted": "Provide spec_json or spec_url."}
        spec = json.loads(spec_json)
        out = _parse_openapi_spec(spec)
        formatted = f"Base URL: {out['base_url']}\nEndpoints: {len(out['endpoints'])}\nAuth schemes: {len(out['auth_schemes'])}"
        return {"endpoints": out["endpoints"], "auth_schemes": out["auth_schemes"], "base_url": out["base_url"], "formatted": formatted}
    except json.JSONDecodeError as e:
        return {"endpoints": [], "auth_schemes": [], "base_url": "", "formatted": f"Invalid JSON: {e}"}
    except Exception as e:
        logger.exception("analyze_openapi_spec_tool: %s", e)
        return {"endpoints": [], "auth_schemes": [], "base_url": "", "formatted": str(e)}


async def draft_connector_definition_tool(
    api_description: str,
    example_response: Optional[str] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Return connector definition schema and instructions. Use the schema to produce a definition matching the API.

    Endpoint keys in definition.endpoints are internal labels: they select which
    config (path, method, body_template) to use when calling the API; they are
    not sent to the external API. Use descriptive snake_case (e.g. server_get_status,
    client_set_volume). Never use numeric IDs (0, 1, 2); validate_connector_definition
    will reject them. These ids appear in the UI and in control pane controls.
    """
    instructions = "Use the following schema to build a connector definition. Output valid JSON only.\n\nSchema:\n" + CONNECTOR_DEFINITION_SCHEMA
    instructions += "\n\nAPI description:\n" + (api_description or "")
    if example_response:
        instructions += "\n\nExample response (use to set response_list_path):\n" + (example_response[:2000] if len(example_response or "") > 2000 else (example_response or ""))
    return {
        "definition": {},
        "name": "API Connector",
        "description": api_description[:200] if api_description else "",
        "requires_auth": "auth" in (api_description or "").lower() or "key" in (api_description or "").lower(),
        "auth_fields": [],
        "formatted": instructions,
    }


async def validate_connector_definition_tool(definition: Dict[str, Any], user_id: str = "system") -> Dict[str, Any]:
    """Validate a connector definition without making HTTP calls.

    Endpoint keys in definition.endpoints are internal labels (they select which
    path/method/body to use; not sent to the API). They must be descriptive
    snake_case (e.g. server_get_status). Numeric IDs (0, 1, 2) are rejected —
    fix them before calling create_data_connector.
    """
    errors = []
    warnings = []
    base_url = (definition.get("base_url") or "").strip()
    if not base_url:
        errors.append("base_url is required")
    elif not (base_url.startswith("http://") or base_url.startswith("https://")):
        errors.append("base_url must be http or https")
    conn_headers = definition.get("headers")
    if conn_headers is not None:
        if not isinstance(conn_headers, dict):
            errors.append("headers must be an object")
        else:
            for k, v in conn_headers.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    errors.append("headers must have string keys and string values")
                    break
    endpoints = definition.get("endpoints")
    if not endpoints or not isinstance(endpoints, dict):
        errors.append("endpoints must be a non-empty object")
    else:
        for eid, ep in endpoints.items():
            if str(eid).strip().lstrip("-").isdigit():
                errors.append(
                    f"Endpoint id '{eid}' must be a descriptive snake_case name "
                    f"(e.g. server_get_status, client_set_volume). Numeric IDs are not allowed."
                )
                continue
            if not isinstance(ep, dict):
                errors.append(f"endpoint '{eid}' must be an object")
                continue
            if not ep.get("path"):
                errors.append(f"endpoint '{eid}' missing path")
            if not ep.get("method"):
                warnings.append(f"endpoint '{eid}' missing method (defaults to GET)")
            ep_headers = ep.get("headers")
            if ep_headers is not None:
                if not isinstance(ep_headers, dict):
                    errors.append(f"endpoint '{eid}' headers must be an object")
                else:
                    for k, v in ep_headers.items():
                        if not isinstance(k, str) or not isinstance(v, str):
                            errors.append(f"endpoint '{eid}' headers must have string keys and string values")
                            break
            body_template = ep.get("body_template")
            if body_template is not None:
                if not isinstance(body_template, dict):
                    errors.append(f"endpoint '{eid}' body_template must be an object")
                else:

                    def _collect_placeholders(obj: Any, out: set) -> None:
                        if isinstance(obj, str) and len(obj) > 2 and obj.startswith("{") and obj.endswith("}"):
                            out.add(obj[1:-1])
                        elif isinstance(obj, dict):
                            for v in obj.values():
                                _collect_placeholders(v, out)
                        elif isinstance(obj, list):
                            for v in obj:
                                _collect_placeholders(v, out)

                    param_names = {p.get("name") for p in (ep.get("params") or []) if isinstance(p, dict) and p.get("name")}
                    placeholders = set()
                    _collect_placeholders(body_template, placeholders)
                    for placeholder in placeholders:
                        if param_names and placeholder not in param_names:
                            warnings.append(f"endpoint '{eid}' body_template placeholder {{{placeholder}}} has no matching param")
            for p in ep.get("params") or []:
                if not isinstance(p, dict) or not p.get("name"):
                    errors.append(f"endpoint '{eid}' has invalid param entry")
                    break
    auth = definition.get("auth") or {}
    if auth and isinstance(auth, dict) and auth.get("type") and not auth.get("credentials_key") and auth.get("type") != "oauth_connection":
        warnings.append("auth has type but no credentials_key")
    is_valid = len(errors) == 0
    formatted = "Valid." if is_valid else "Errors: " + "; ".join(errors)
    if warnings:
        formatted += " Warnings: " + "; ".join(warnings)
    return {"is_valid": is_valid, "errors": errors, "warnings": warnings, "formatted": formatted}


# ── Zone 2: gRPC-backed tools ───────────────────────────────────────────────


def _analyze_response_paths(
    obj: Any,
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 4,
    results: Optional[List[tuple]] = None,
) -> List[tuple]:
    """Walk raw API response and collect dot-notation paths; tag arrays and scalars."""
    if results is None:
        results = []
    if depth > max_depth or obj is None:
        return results
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, list):
                results.append((path, "array", len(v)))
            elif isinstance(v, (str, int, float, bool)):
                results.append((path, "scalar", v))
            else:
                _analyze_response_paths(v, path, depth + 1, max_depth, results)
    elif isinstance(obj, list) and obj:
        _analyze_response_paths(obj[0], prefix, depth + 1, max_depth, results)
    return results


async def probe_api_endpoint_tool(
    url: str,
    method: str = "GET",
    headers_json: Optional[str] = None,
    body_json: Optional[str] = None,
    params_json: Optional[str] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Perform a raw HTTP request to a URL. Returns status, headers, and body (truncated)."""
    try:
        client = await get_backend_tool_client()
        result = await client.probe_api_endpoint(
            user_id=user_id,
            url=url,
            method=method,
            headers_json=headers_json or "{}",
            body_json=body_json or "",
            params_json=params_json or "{}",
        )
        if not result.get("success"):
            return {
                "status_code": 0,
                "response_headers": {},
                "response_body": "",
                "content_type": "",
                "formatted": result.get("error", "Probe failed"),
            }
        headers = {}
        if result.get("response_headers_json"):
            try:
                headers = json.loads(result["response_headers_json"])
            except json.JSONDecodeError:
                pass
        body = result.get("response_body", "")
        formatted = f"Status: {result.get('status_code')}. Content-Type: {result.get('content_type', '')}. Body length: {len(body)} chars."
        return {
            "status_code": result.get("status_code", 0),
            "response_headers": headers,
            "response_body": body,
            "content_type": result.get("content_type", ""),
            "formatted": formatted,
        }
    except Exception as e:
        logger.exception("probe_api_endpoint_tool: %s", e)
        return {"status_code": 0, "response_headers": {}, "response_body": "", "content_type": "", "formatted": str(e)}


async def test_connector_endpoint_tool(
    definition_json: str,
    endpoint_id: str,
    params: Optional[Dict[str, Any]] = None,
    credentials: Optional[Dict[str, Any]] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Test a connector endpoint against the live API. No save required.

    Call this before setting control pane refresh_endpoint_id and value_path to see
    the actual response shape. For JSON-RPC APIs the data is under a result
    envelope (e.g. result.server.groups); use that prefix in value_path.
    """
    try:
        client = await get_backend_tool_client()
        result = await client.test_connector_endpoint(
            user_id=user_id,
            definition_json=definition_json,
            endpoint_id=endpoint_id,
            params_json=json.dumps(params or {}),
            credentials_json=json.dumps(credentials or {}),
        )
        if not result.get("success"):
            return {
                "records": [],
                "count": 0,
                "raw_response": None,
                "formatted": result.get("formatted") or result.get("error", "Test failed"),
                "value_path_candidates": [],
                "scalar_paths": [],
            }
        raw_response = result.get("raw_response")
        records = result.get("records", [])
        count = result.get("count", 0)
        value_path_candidates: List[str] = []
        scalar_paths: List[Dict[str, Any]] = []
        formatted = result.get("formatted", f"Retrieved {count} record(s).")

        if isinstance(raw_response, dict):
            path_tuples = _analyze_response_paths(raw_response)
            array_paths = [p for p, t, _ in path_tuples if t == "array"]
            value_path_candidates = array_paths
            scalar_paths = [
                {"path": p, "value": v} for p, t, v in path_tuples if t == "scalar"
            ]
            top_keys = list(raw_response.keys()) if raw_response else []
            status = result.get("status_code") or 200
            formatted_parts = [
                f"Endpoint {endpoint_id} responded ({status}):",
                f"  Top-level keys: {', '.join(top_keys)}",
                "",
                "  Paths in response:",
            ]
            for path, ptype, payload in path_tuples:
                if ptype == "array":
                    formatted_parts.append(f"    {path}  [array, {payload} items]  <-- value_path candidate")
                else:
                    sample = repr(payload)[:60] + ("..." if len(repr(payload)) > 60 else "")
                    formatted_parts.append(f"    {path}  scalar: {sample}")
            if array_paths:
                formatted_parts.extend([
                    "",
                    "  Suggested value_path / response_list_path candidates (arrays):",
                ] + [f"    {p}" for p in array_paths])
            if count == 0:
                formatted_parts.append(
                    "\n  Note: records=0 because response_list_path is not set in the connector definition. "
                    "Set response_list_path on the endpoint to extract records automatically."
                )
            formatted = "\n".join(formatted_parts)

        return {
            "records": records,
            "count": count,
            "raw_response": raw_response,
            "formatted": formatted,
            "value_path_candidates": value_path_candidates,
            "scalar_paths": scalar_paths,
        }
    except Exception as e:
        logger.exception("test_connector_endpoint_tool: %s", e)
        return {
            "records": [],
            "count": 0,
            "raw_response": None,
            "formatted": str(e),
            "value_path_candidates": [],
            "scalar_paths": [],
        }


async def create_data_connector_tool(
    name: str,
    description: str,
    definition: Dict[str, Any],
    requires_auth: bool = False,
    auth_fields: Optional[List[Dict[str, Any]]] = None,
    category: Optional[str] = None,
    confirmed: bool = False,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Save a connector definition to the database. Use confirmed=False to preview, confirmed=True to create.

    Endpoint keys in definition.endpoints are internal labels (they select which
    path/method/body to use; they are not sent to the API). Use descriptive
    snake_case (e.g. server_get_status, client_set_volume). Never numeric (0, 1, 2);
    validate_connector_definition rejects them. These ids appear in the UI and
    in control pane controls.
    """
    if not confirmed:
        return {
            "connector_id": "",
            "name": name,
            "formatted": f"[Preview] Would create data connector '{name}' with {len(definition.get('endpoints') or {})} endpoint(s). Call again with confirmed=True to save.",
        }
    try:
        client = await get_backend_tool_client()
        result = await client.create_data_connector(
            user_id=user_id,
            name=name,
            description=description,
            definition_json=json.dumps(definition),
            requires_auth=requires_auth,
            auth_fields_json=json.dumps(auth_fields or []),
            category=category or "",
        )
        if not result.get("success"):
            return {"connector_id": "", "name": name, "formatted": result.get("formatted") or result.get("error", "Create failed")}
        return {
            "connector_id": result.get("connector_id", ""),
            "name": result.get("name", name),
            "formatted": result.get("formatted", f"Created connector: {result.get('name')}"),
        }
    except Exception as e:
        logger.exception("create_data_connector_tool: %s", e)
        return {"connector_id": "", "name": name, "formatted": str(e)}


async def list_data_connectors_tool(user_id: str = "system") -> Dict[str, Any]:
    """List user-owned data source connectors (non-templates). Use before creating to avoid duplicates."""
    try:
        client = await get_backend_tool_client()
        result = await client.list_data_connectors(user_id=user_id)
        if not result.get("success"):
            return {"connectors": [], "count": 0, "formatted": result.get("formatted") or result.get("error", "List failed")}
        connectors = result.get("connectors", [])
        parts = [f"Found {len(connectors)} connector(s):"]
        for c in connectors:
            name = c.get("name", "(unnamed)")
            cid = c.get("id", "")
            ctype = c.get("connector_type", "rest")
            n_ep = c.get("endpoint_count", 0)
            parts.append(f"  - {name} (id: {cid}, type: {ctype}, {n_ep} endpoint(s))")
        formatted = "\n".join(parts) if connectors else parts[0]
        return {"connectors": connectors, "count": len(connectors), "formatted": formatted}
    except Exception as e:
        logger.exception("list_data_connectors_tool: %s", e)
        return {"connectors": [], "count": 0, "formatted": str(e)}


async def get_data_connector_detail_tool(
    connector_id: str,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Get full data connector details including definition and endpoints.
    Auth values are redacted; requires_auth and auth_field_names are returned.
    Use before editing a connector or binding it to an agent.
    """
    try:
        client = await get_backend_tool_client()
        connector = await client.get_data_connector(user_id=user_id, connector_id=connector_id)
        if not connector:
            return {
                "connector_id": connector_id,
                "name": "",
                "description": None,
                "connector_type": "rest",
                "definition": {},
                "endpoint_count": 0,
                "is_locked": False,
                "category": None,
                "tags": [],
                "formatted": f"Connector not found: {connector_id}",
            }
        definition = connector.get("definition") or {}
        tags = list(connector.get("tags") or [])
        parts = [
            f"**{connector.get('name', '')}** (ID: {connector.get('id', connector_id)})",
            f"Type: {connector.get('connector_type', 'rest')}, Endpoints: {connector.get('endpoint_count', 0)}",
        ]
        if connector.get("requires_auth"):
            parts.append(f"Auth: required (fields: {', '.join(connector.get('auth_field_names', []))})")
        return {
            "connector_id": str(connector.get("id", connector_id)),
            "name": connector.get("name", ""),
            "description": connector.get("description"),
            "connector_type": connector.get("connector_type", "rest"),
            "definition": definition,
            "endpoint_count": connector.get("endpoint_count", 0),
            "is_locked": connector.get("is_locked", False),
            "category": connector.get("category"),
            "tags": tags,
            "formatted": "\n".join(parts),
        }
    except Exception as e:
        logger.exception("get_data_connector_detail_tool: %s", e)
        return {
            "connector_id": connector_id,
            "name": "",
            "description": None,
            "connector_type": "rest",
            "definition": {},
            "endpoint_count": 0,
            "is_locked": False,
            "category": None,
            "tags": [],
            "formatted": str(e),
        }


async def update_data_connector_tool(
    connector_id: str,
    confirmed: bool = False,
    name: Optional[str] = None,
    description: Optional[str] = None,
    connector_type: Optional[str] = None,
    definition: Optional[Dict[str, Any]] = None,
    requires_auth: Optional[bool] = None,
    auth_fields: Optional[List[Dict[str, Any]]] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Update an existing connector (partial update). Use confirmed=False to preview, confirmed=True to apply."""
    if not confirmed:
        return {
            "connector_id": connector_id,
            "formatted": f"[Preview] Would update connector {connector_id}. Call again with confirmed=True to apply.",
        }
    try:
        client = await get_backend_tool_client()
        definition_json = json.dumps(definition) if definition is not None else None
        auth_fields_json = json.dumps(auth_fields) if auth_fields is not None else None
        result = await client.update_data_connector(
            user_id=user_id,
            connector_id=connector_id,
            name=name,
            description=description,
            connector_type=connector_type,
            definition_json=definition_json,
            requires_auth=requires_auth,
            auth_fields_json=auth_fields_json,
        )
        if not result.get("success"):
            return {"connector_id": connector_id, "formatted": result.get("formatted") or result.get("error", "Update failed")}
        return {"connector_id": result.get("connector_id", connector_id), "formatted": result.get("formatted", f"Updated connector {connector_id}")}
    except Exception as e:
        logger.exception("update_data_connector_tool: %s", e)
        return {"connector_id": connector_id, "formatted": str(e)}


async def bulk_scrape_urls_tool(
    urls: List[str],
    extract_images: bool = True,
    download_images: bool = True,
    image_output_folder: Optional[str] = None,
    metadata_fields: Optional[List[str]] = None,
    max_concurrent: int = 10,
    rate_limit_seconds: float = 1.0,
    folder_id: Optional[str] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Scrape URLs for content and optionally images. Under 20 URLs: inline. 20+: background task."""
    try:
        client = await get_backend_tool_client()
        result = await client.bulk_scrape_urls(
            user_id=user_id,
            urls_json=json.dumps(urls),
            extract_images=extract_images,
            download_images=download_images,
            image_output_folder=image_output_folder or "",
            metadata_fields_json=json.dumps(metadata_fields or []),
            max_concurrent=max_concurrent,
            rate_limit_seconds=rate_limit_seconds,
            folder_id=folder_id or "",
        )
        if not result.get("success"):
            return {
                "task_id": "",
                "results": [],
                "count": 0,
                "images_found": 0,
                "images_downloaded": 0,
                "formatted": result.get("formatted") or result.get("error", "Bulk scrape failed"),
            }
        return {
            "task_id": result.get("task_id", ""),
            "results": result.get("results", []),
            "count": result.get("count", 0),
            "images_found": result.get("images_found", 0),
            "images_downloaded": result.get("images_downloaded", 0),
            "formatted": result.get("formatted", ""),
        }
    except Exception as e:
        logger.exception("bulk_scrape_urls_tool: %s", e)
        return {"task_id": "", "results": [], "count": 0, "images_found": 0, "images_downloaded": 0, "formatted": str(e)}


async def get_bulk_scrape_status_tool(task_id: str, user_id: str = "system") -> Dict[str, Any]:
    """Get status and optional results of a bulk scrape task (when 20+ URLs)."""
    try:
        client = await get_backend_tool_client()
        result = await client.get_bulk_scrape_status(user_id=user_id, task_id=task_id)
        if not result.get("success"):
            return {
                "status": "UNKNOWN",
                "progress_current": 0,
                "progress_total": 0,
                "progress_message": "",
                "results": [],
                "formatted": result.get("formatted") or result.get("error", "Status check failed"),
            }
        return {
            "status": result.get("status", "UNKNOWN"),
            "progress_current": result.get("progress_current", 0),
            "progress_total": result.get("progress_total", 0),
            "progress_message": result.get("progress_message", ""),
            "results": result.get("results", []),
            "formatted": result.get("formatted", ""),
        }
    except Exception as e:
        logger.exception("get_bulk_scrape_status_tool: %s", e)
        return {"status": "UNKNOWN", "progress_current": 0, "progress_total": 0, "progress_message": "", "results": [], "formatted": str(e)}


# ── Register all tools ──────────────────────────────────────────────────────

register_action(
    name="probe_api_endpoint",
    category="data_connection_builder",
    description="Perform a raw HTTP request to a URL for API discovery",
    inputs_model=ProbeApiEndpointInputs,
    params_model=ProbeApiEndpointParams,
    outputs_model=ProbeApiEndpointOutputs,
    tool_function=probe_api_endpoint_tool,
)

register_action(
    name="analyze_openapi_spec",
    category="data_connection_builder",
    description="Parse OpenAPI/Swagger spec and extract endpoints and auth",
    inputs_model=AnalyzeOpenapiSpecInputs,
    params_model=None,
    outputs_model=AnalyzeOpenapiSpecOutputs,
    tool_function=analyze_openapi_spec_tool,
)

register_action(
    name="draft_connector_definition",
    category="data_connection_builder",
    description="Return connector definition schema from API description",
    inputs_model=DraftConnectorDefinitionInputs,
    params_model=None,
    outputs_model=DraftConnectorDefinitionOutputs,
    tool_function=draft_connector_definition_tool,
)

register_action(
    name="validate_connector_definition",
    category="data_connection_builder",
    description="Validate a connector definition without making HTTP calls",
    inputs_model=ValidateConnectorDefinitionInputs,
    params_model=None,
    outputs_model=ValidateConnectorDefinitionOutputs,
    tool_function=validate_connector_definition_tool,
)

register_action(
    name="test_connector_endpoint",
    category="data_connection_builder",
    description="Test a connector endpoint against the live API",
    inputs_model=TestConnectorEndpointInputs,
    params_model=None,
    outputs_model=TestConnectorEndpointOutputs,
    tool_function=test_connector_endpoint_tool,
)

register_action(
    name="create_data_connector",
    category="data_connection_builder",
    description="Save a connector definition to the database",
    inputs_model=CreateDataConnectorInputs,
    params_model=None,
    outputs_model=CreateDataConnectorOutputs,
    tool_function=create_data_connector_tool,
)

register_action(
    name="list_data_connectors",
    category="data_connection_builder",
    description="List user-owned data source connectors; call before creating to avoid duplicates",
    inputs_model=ListDataConnectorsInputs,
    params_model=None,
    outputs_model=ListDataConnectorsOutputs,
    tool_function=list_data_connectors_tool,
)
register_action(
    name="get_data_connector_detail",
    category="data_connection_builder",
    description="Get full connector definition and endpoints; auth values redacted; use before editing or binding",
    inputs_model=GetDataConnectorDetailInputs,
    params_model=None,
    outputs_model=GetDataConnectorDetailOutputs,
    tool_function=get_data_connector_detail_tool,
)
register_action(
    name="update_data_connector",
    category="data_connection_builder",
    description="Update an existing data connector",
    inputs_model=UpdateDataConnectorInputs,
    params_model=UpdateDataConnectorParams,
    outputs_model=UpdateDataConnectorOutputs,
    tool_function=update_data_connector_tool,
)

register_action(
    name="bulk_scrape_urls",
    category="data_connection_builder",
    description="Scrape URLs for content and optionally download images",
    inputs_model=BulkScrapeUrlsInputs,
    params_model=BulkScrapeUrlsParams,
    outputs_model=BulkScrapeUrlsOutputs,
    tool_function=bulk_scrape_urls_tool,
)

register_action(
    name="get_bulk_scrape_status",
    category="data_connection_builder",
    description="Get status and results of a bulk scrape task",
    inputs_model=GetBulkScrapeStatusInputs,
    params_model=None,
    outputs_model=GetBulkScrapeStatusOutputs,
    tool_function=get_bulk_scrape_status_tool,
)
