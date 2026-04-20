"""
Control Pane Builder tools - list, create, update, delete status bar control panes; inspect connector endpoints.

Zone 2: all tools via gRPC backend (touch user_control_panes and data_source_connectors).
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)

# Control types: slider, dropdown, toggle, button, text_display
# Each control: type, label, endpoint_id, param_key; optional refresh_endpoint_id, value_path; slider: min, max, step; dropdown: options_*
CONTROL_TYPES = ("slider", "dropdown", "toggle", "button", "text_display")


# ----- I/O models -----

class ListControlPanesInputs(BaseModel):
    pass


class ListControlPanesOutputs(BaseModel):
    panes: List[Dict[str, Any]] = Field(
        description="List of control panes: id, name, icon, pane_type (connector|artifact), connector_id, artifact_id, "
        "artifact_popover_width/height, connector_name, controls, is_visible, sort_order, refresh_interval. "
        "Artifact panes embed a saved library artifact in the status bar; connector panes use data source connectors."
    )
    count: int = Field(description="Number of panes")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class GetConnectorEndpointsInputs(BaseModel):
    connector_id: str = Field(description="Data source connector UUID")


class GetConnectorEndpointsOutputs(BaseModel):
    endpoints: List[Dict[str, Any]] = Field(
        description="List of {id, path, method, description, params}. id is the internal endpoint key (snake_case); not sent to the API — it selects which config to use."
    )
    count: int = Field(description="Number of endpoints")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class CreateControlPaneInputs(BaseModel):
    name: str = Field(description="Display name for the control pane")
    connector_id: str = Field(
        description="Data source connector UUID to wire the pane to. "
        "Saved-artifact panes (status bar embeds from the artifact library) are created in the app Control Panes settings, not via this tool."
    )
    controls: List[Dict[str, Any]] = Field(
        description="Array of control objects: type (slider|dropdown|toggle|button|text_display), label, endpoint_id, param_key. "
        "endpoint_id is the connector's internal endpoint key (snake_case, e.g. server_get_status) — it selects which endpoint config to call; not sent to the API. Never use numeric IDs (0, 1, 2). "
        "Get valid endpoint_id and param_key from get_connector_endpoints. "
        "For controls that show state: set refresh_endpoint_id and value_path. Call test_connector_endpoint first to see the real response; for JSON-RPC use result.* (e.g. result.server.groups). "
        "For text_display list/summary point value_path to the array; for scalars (volume, mute) point to the leaf (e.g. result.server.groups.0.muted). Slider: min, max, step. "
        "Dropdown: options_endpoint_id for dynamic options, or options: [{label, value}] for static options (e.g. [{label: 'Play', value: 'play'}, ...]). "
        "Button: optional icon (PlayArrow, Pause, Stop, SkipNext, SkipPrevious, Refresh, PowerSettingsNew, Lightbulb, Thermostat, VolumeUp, VolumeOff). "
        "param_source: list of {param, from_control_id} to inject another controls value into this controls params when it runs. refresh_param_source: same at refresh time for state-display controls."
    )


class CreateControlPaneParams(BaseModel):
    icon: str = Field(default="Tune", description="Icon name: Tune, Settings, VolumeUp, PlayArrow, TouchApp, Dashboard, ToggleOn, SmartToy")
    credentials_encrypted: Optional[Dict[str, Any]] = Field(default=None, description="Optional credentials JSON for the connector")
    connection_id: Optional[int] = Field(default=None, description="Optional OAuth connection ID")
    is_visible: bool = Field(default=True, description="Show pane in status bar")
    sort_order: int = Field(default=0, description="Order in status bar")
    refresh_interval: int = Field(default=0, ge=0, description="Auto-refresh interval in seconds (0 = manual only; 5–30 recommended for live state)")


class CreateControlPaneOutputs(BaseModel):
    pane_id: str = Field(description="Created pane UUID")
    name: str = Field(description="Pane name")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class UpdateControlPaneInputs(BaseModel):
    pane_id: str = Field(description="Control pane UUID to update")
    name: Optional[str] = Field(default=None, description="New display name")
    controls: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="New controls array (replaces existing). endpoint_id is the connector's internal endpoint key (snake_case, e.g. server_get_status); not sent to the API. Never numeric (0, 1, 2). "
        "Use endpoint_id and param_key from get_connector_endpoints; set refresh_endpoint_id and value_path from test_connector_endpoint response (JSON-RPC: result.* paths). "
        "Dropdown: options_endpoint_id for dynamic options, or options: [{label, value}] for static options. Button: optional icon (PlayArrow, Pause, Stop, SkipNext, SkipPrevious, Refresh, PowerSettingsNew, Lightbulb, VolumeUp, VolumeOff).",
    )


class UpdateControlPaneParams(BaseModel):
    icon: Optional[str] = Field(default=None, description="Icon name")
    connector_id: Optional[str] = Field(default=None, description="Connector UUID")
    credentials_encrypted_json: Optional[str] = Field(default=None, description="Credentials JSON")
    connection_id: Optional[int] = Field(default=None, description="OAuth connection ID")
    is_visible: Optional[bool] = Field(default=None, description="Visibility")
    sort_order: Optional[int] = Field(default=None, description="Sort order")
    refresh_interval: Optional[int] = Field(default=None, ge=0, description="Auto-refresh interval in seconds (0 = off; 5–30 for live state)")


class UpdateControlPaneOutputs(BaseModel):
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class DeleteControlPaneInputs(BaseModel):
    pane_id: str = Field(description="Control pane UUID to delete")


class DeleteControlPaneOutputs(BaseModel):
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class ExecuteControlActionInputs(BaseModel):
    pane_id: str = Field(description="Control pane UUID (from list_control_panes)")
    endpoint_id: str = Field(
        description="Connector endpoint key (snake_case, e.g. server_set_volume). Must match a control's endpoint_id on the pane."
    )
    params: Optional[Dict[str, Any]] = Field(default=None, description="Parameter values to send (e.g. {\"volume\": 50}). Omit for no params.")


class ExecuteControlActionOutputs(BaseModel):
    raw_response: Optional[Dict[str, Any]] = Field(default=None, description="Full API response from the connector")
    records: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted records if the connector returns a list")
    count: int = Field(description="Number of records")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


# ----- Tool functions -----

async def list_control_panes_tool(user_id: str = "system") -> Dict[str, Any]:
    """List all control panes for the user."""
    try:
        client = await get_backend_tool_client()
        result = await client.list_control_panes(user_id=user_id)
        if not result.get("success"):
            return {"panes": [], "count": 0, "formatted": result.get("formatted") or result.get("error", "List failed")}
        panes = result.get("panes", [])
        parts = [f"Found {len(panes)} control pane(s):"]
        for p in panes:
            name = p.get("name", "(unnamed)")
            pid = p.get("id", "")
            if (p.get("pane_type") or "connector") == "artifact":
                extra = f"artifact_id={p.get('artifact_id', '')}"
            else:
                conn_name = p.get("connector_name") or p.get("connector_id", "")
                extra = f"connector: {conn_name}"
            parts.append(f"  - {name} (id: {pid}, {extra})")
        formatted = "\n".join(parts) if panes else parts[0]
        return {"panes": panes, "count": len(panes), "formatted": formatted}
    except Exception as e:
        logger.exception("list_control_panes_tool: %s", e)
        return {"panes": [], "count": 0, "formatted": str(e)}


async def get_connector_endpoints_tool(
    connector_id: str,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Get endpoint ids and metadata from a connector for mapping to control pane controls.

    Endpoint ids are internal keys: they select which endpoint config (path, method,
    body) to use; they are not sent to the external API. They are always descriptive
    snake_case (e.g. server_get_status). Use the returned ids as endpoint_id and
    refresh_endpoint_id in control definitions. Before setting value_path, call
    test_connector_endpoint with the connector definition and that endpoint_id to see
    the actual response shape.
    """
    try:
        client = await get_backend_tool_client()
        result = await client.get_connector_endpoints(user_id=user_id, connector_id=connector_id)
        if not result.get("success"):
            return {"endpoints": [], "count": 0, "formatted": result.get("formatted") or result.get("error", "Failed")}
        endpoints = result.get("endpoints", [])
        parts = []
        for ep in endpoints:
            param_names = [p["name"] for p in ep.get("params", [])]
            parts.append(f"  {ep['id']} ({ep.get('method', 'GET')} {ep.get('path', '/')}) params: {param_names or 'none'}")
        formatted = f"Connector has {len(endpoints)} endpoint(s):\n" + "\n".join(parts) if parts else result.get("formatted", f"Connector has {len(endpoints)} endpoint(s).")
        return {"endpoints": endpoints, "count": len(endpoints), "formatted": formatted}
    except Exception as e:
        logger.exception("get_connector_endpoints_tool: %s", e)
        return {"endpoints": [], "count": 0, "formatted": str(e)}


def _ensure_control_ids(controls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure each control has an id for the frontend."""
    out = []
    for i, c in enumerate(controls):
        copy = dict(c)
        if not copy.get("id"):
            copy["id"] = f"control_{i}_{hash(json.dumps(copy, sort_keys=True)) % 100000}"
        out.append(copy)
    return out


async def create_control_pane_tool(
    name: str,
    connector_id: str,
    controls: List[Dict[str, Any]],
    icon: str = "Tune",
    credentials_encrypted: Optional[Dict[str, Any]] = None,
    connection_id: Optional[int] = None,
    is_visible: bool = True,
    sort_order: int = 0,
    refresh_interval: int = 0,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Create a control pane wired to a data connector. Controls appear in the status bar.

    Call list_control_panes first — if a pane for this connector exists, use
    update_control_pane instead. endpoint_id is the connector's internal key (selects
    which endpoint config to call; not sent to the API). Use snake_case ids from
    get_connector_endpoints (e.g. server_get_status), never numbers. For controls
    with refresh_endpoint_id and value_path, call test_connector_endpoint to see the
    response shape; for JSON-RPC use result.* in value_path. Use refresh_interval for
    live state. Dropdown: options_endpoint_id or static options. Button: optional icon.
    Cross-control binding: param_source and refresh_param_source (list of {param, from_control_id}) inject another controls value into params; e.g. Snapcast group selector id into volume or mute controls.
    """
    try:
        controls = _ensure_control_ids(controls)
        client = await get_backend_tool_client()
        result = await client.create_control_pane(
            user_id=user_id,
            name=name,
            connector_id=connector_id,
            icon=icon,
            credentials_encrypted_json=json.dumps(credentials_encrypted or {}),
            connection_id=connection_id,
            controls_json=json.dumps(controls),
            is_visible=is_visible,
            sort_order=sort_order,
            refresh_interval=refresh_interval,
        )
        if not result.get("success"):
            return {"pane_id": "", "name": name, "formatted": result.get("formatted") or result.get("error", "Create failed")}
        return {"pane_id": result.get("pane_id", ""), "name": result.get("name", name), "formatted": result.get("formatted", f"Created control pane: {name}")}
    except Exception as e:
        logger.exception("create_control_pane_tool: %s", e)
        return {"pane_id": "", "name": name, "formatted": str(e)}


async def update_control_pane_tool(
    pane_id: str,
    name: Optional[str] = None,
    controls: Optional[List[Dict[str, Any]]] = None,
    icon: Optional[str] = None,
    connector_id: Optional[str] = None,
    credentials_encrypted_json: Optional[str] = None,
    connection_id: Optional[int] = None,
    is_visible: Optional[bool] = None,
    sort_order: Optional[int] = None,
    refresh_interval: Optional[int] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Update a control pane (partial update).

    When changing controls, endpoint_id is the connector's internal key (not sent to
    the API); use snake_case from get_connector_endpoints (e.g. server_get_status),
    never numbers. value_path must match the connector response shape; for JSON-RPC
    use result.*. Use test_connector_endpoint to verify paths. Can set refresh_interval,
    static dropdown options, and button icon. Use param_source and refresh_param_source for cross-control binding.
    """
    try:
        if controls is not None:
            controls = _ensure_control_ids(controls)
        client = await get_backend_tool_client()
        result = await client.update_control_pane(
            user_id=user_id,
            pane_id=pane_id,
            name=name,
            icon=icon,
            connector_id=connector_id,
            credentials_encrypted_json=credentials_encrypted_json,
            connection_id=connection_id,
            controls_json=json.dumps(controls) if controls is not None else None,
            is_visible=is_visible,
            sort_order=sort_order,
            refresh_interval=refresh_interval,
        )
        if not result.get("success"):
            return {"formatted": result.get("formatted") or result.get("error", "Update failed")}
        return {"formatted": result.get("formatted", f"Updated control pane {pane_id}")}
    except Exception as e:
        logger.exception("update_control_pane_tool: %s", e)
        return {"formatted": str(e)}


async def delete_control_pane_tool(pane_id: str, user_id: str = "system") -> Dict[str, Any]:
    """Delete a control pane."""
    try:
        client = await get_backend_tool_client()
        result = await client.delete_control_pane(user_id=user_id, pane_id=pane_id)
        if not result.get("success"):
            return {"formatted": result.get("formatted") or result.get("error", "Delete failed")}
        return {"formatted": result.get("formatted", f"Deleted control pane {pane_id}")}
    except Exception as e:
        logger.exception("delete_control_pane_tool: %s", e)
        return {"formatted": str(e)}


async def execute_control_action_tool(
    pane_id: str,
    endpoint_id: str,
    params: Optional[Dict[str, Any]] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Execute a connector endpoint through a saved control pane (simulate a control action).

    Use after creating or updating a pane to verify a control works: e.g. trigger a button,
    set volume, or fetch the response that populates a dropdown. Returns the full API response
    so you can confirm value_path or use response data to wire param_source on other controls.
    """
    try:
        client = await get_backend_tool_client()
        result = await client.execute_control_pane_action(
            user_id=user_id,
            pane_id=pane_id,
            endpoint_id=endpoint_id,
            params_json=json.dumps(params or {}),
        )
        if not result.get("success"):
            formatted = result.get("formatted") or result.get("error", "Execute failed")
            return {
                "raw_response": None,
                "records": [],
                "count": 0,
                "formatted": formatted,
            }
        raw_response = result.get("raw_response")
        records = result.get("records", [])
        count = result.get("count", 0)
        formatted = result.get("formatted", "")
        if not formatted and raw_response is not None:
            formatted = f"Success; response keys: {list(raw_response.keys()) if isinstance(raw_response, dict) else 'n/a'}"
        return {
            "raw_response": raw_response,
            "records": records,
            "count": count,
            "formatted": formatted,
        }
    except Exception as e:
        logger.exception("execute_control_action_tool: %s", e)
        return {"raw_response": None, "records": [], "count": 0, "formatted": str(e)}


# ----- Registry -----

register_action(
    name="list_control_panes",
    category="control_pane_builder",
    description="List the user's control panes (status bar controls wired to data connectors)",
    short_description="List control panes for the status bar",
    inputs_model=ListControlPanesInputs,
    params_model=None,
    outputs_model=ListControlPanesOutputs,
    tool_function=list_control_panes_tool,
)

register_action(
    name="get_connector_endpoints",
    category="control_pane_builder",
    description="Get endpoint ids and params from a data connector",
    inputs_model=GetConnectorEndpointsInputs,
    params_model=None,
    outputs_model=GetConnectorEndpointsOutputs,
    tool_function=get_connector_endpoints_tool,
)

register_action(
    name="create_control_pane",
    category="control_pane_builder",
    description="Create a control pane wired to a data connector",
    inputs_model=CreateControlPaneInputs,
    params_model=CreateControlPaneParams,
    outputs_model=CreateControlPaneOutputs,
    tool_function=create_control_pane_tool,
)

register_action(
    name="update_control_pane",
    category="control_pane_builder",
    description="Update a control pane",
    inputs_model=UpdateControlPaneInputs,
    params_model=UpdateControlPaneParams,
    outputs_model=UpdateControlPaneOutputs,
    tool_function=update_control_pane_tool,
)

register_action(
    name="delete_control_pane",
    category="control_pane_builder",
    description="Delete a control pane",
    inputs_model=DeleteControlPaneInputs,
    params_model=None,
    outputs_model=DeleteControlPaneOutputs,
    tool_function=delete_control_pane_tool,
)

register_action(
    name="execute_control_action",
    category="control_pane_builder",
    description="Execute a connector endpoint through a saved control pane (simulate a control: button, slider, etc.). Returns the API response for verification or to wire param_source.",
    short_description="Execute a control pane action (button, slider, etc.)",
    inputs_model=ExecuteControlActionInputs,
    params_model=None,
    outputs_model=ExecuteControlActionOutputs,
    tool_function=execute_control_action_tool,
)

