"""
Agent Factory API - Workflow Composer, action registry, and CRUD for profiles, data sources, playbooks.

Exposes registered action I/O contracts and CRUD for agent profiles, data source bindings, and playbooks.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional

import grpc
import yaml
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator
from pydantic import AliasChoices

from utils.auth_middleware import get_current_user, AuthenticatedUserResponse

from services import agent_factory_service
from services import agent_skills_service

logger = logging.getLogger(__name__)


async def _notify_agent_handles_for_profile(profile_id: str) -> None:
    """Push user WebSocket hint so @mention menus refresh (owner + share recipients)."""
    try:
        from services.agent_handles_notify import (
            collect_users_for_agent_profile_handle_notifications,
            notify_agent_handles_changed,
        )

        uids = await collect_users_for_agent_profile_handle_notifications(str(profile_id))
        await notify_agent_handles_changed(uids)
    except Exception as e:
        logger.warning("agent_handles_changed notify failed for profile %s: %s", profile_id, e)


def _api_user_rls(user_id: str) -> Dict[str, str]:
    """RLS session context for Agent Factory HTTP handlers (matches database_manager GUC keys)."""
    return {"user_id": user_id, "user_role": "user"}


async def _agent_factory_request_rls(
    current_user: Annotated[AuthenticatedUserResponse, Depends(get_current_user)],
):
    """Bind default DB RLS GUCs for this request (database_helpers.http_request_rls_context)."""
    from services.database_manager.database_helpers import http_request_rls_context

    token = http_request_rls_context.set(_api_user_rls(current_user.user_id))
    try:
        yield
    finally:
        http_request_rls_context.reset(token)


router = APIRouter(
    prefix="/api/agent-factory",
    tags=["Agent Factory"],
    dependencies=[Depends(_agent_factory_request_rls)],
)


try:
    from protos import orchestrator_pb2, orchestrator_pb2_grpc
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False


# ---------- Pydantic models ----------

class AgentProfileCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    handle: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None
    is_active: bool = True
    model_preference: Optional[str] = None
    model_source: Optional[str] = Field(None, max_length=50)
    model_provider_type: Optional[str] = Field(None, max_length=50)
    max_research_rounds: int = Field(default=3, ge=1, le=10)
    system_prompt_additions: Optional[str] = None
    knowledge_config: Dict[str, Any] = Field(default_factory=dict)
    default_playbook_id: Optional[str] = None
    default_run_context: str = Field(default="interactive", max_length=50)
    default_approval_policy: str = Field(default="require", max_length=50)
    journal_config: Dict[str, Any] = Field(default_factory=dict)
    team_config: Dict[str, Any] = Field(default_factory=dict)
    watch_config: Dict[str, Any] = Field(default_factory=dict)
    prompt_history_enabled: Annotated[
        bool,
        Field(
            default=False,
            validation_alias=AliasChoices("prompt_history_enabled", "chat_history_enabled"),
            serialization_alias="prompt_history_enabled",
        ),
    ]
    chat_history_lookback: int = Field(default=10, ge=1, le=50)
    summary_threshold_tokens: int = Field(default=5000, ge=500, le=100000)
    summary_keep_messages: int = Field(default=10, ge=1, le=50)
    persona_mode: str = Field(default="none", max_length=50)
    persona_id: Optional[str] = None
    include_user_context: bool = False
    include_datetime_context: bool = True
    include_user_facts: bool = False
    include_facts_categories: List[str] = Field(default_factory=list)
    use_themed_memory: bool = True
    auto_routable: bool = False
    chat_visible: bool = True
    data_workspace_config: Dict[str, Any] = Field(default_factory=dict)
    include_agent_memory: bool = False
    allowed_connections: List[Dict[str, Any]] = Field(default_factory=list)
    category: Optional[str] = Field(None, max_length=100)

    @field_validator("include_facts_categories", mode="before")
    @classmethod
    def coerce_list_fields(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            if not v.strip():
                return []
            try:
                parsed = json.loads(v)
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                return []
        return []

    @field_validator("allowed_connections", mode="before")
    @classmethod
    def coerce_allowed_connections_create(cls, v: Any) -> List[Dict[str, Any]]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str) and v.strip():
            try:
                parsed = json.loads(v)
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                return []
        return []


class AgentProfileUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    handle: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None
    is_active: Optional[bool] = None
    model_preference: Optional[str] = None
    model_source: Optional[str] = Field(None, max_length=50)
    model_provider_type: Optional[str] = Field(None, max_length=50)
    max_research_rounds: Optional[int] = Field(None, ge=1, le=10)
    system_prompt_additions: Optional[str] = None
    knowledge_config: Optional[Dict[str, Any]] = None
    default_playbook_id: Optional[str] = None
    default_run_context: Optional[str] = Field(None, max_length=50)
    default_approval_policy: Optional[str] = Field(None, max_length=50)
    journal_config: Optional[Dict[str, Any]] = None
    team_config: Optional[Dict[str, Any]] = None
    watch_config: Optional[Dict[str, Any]] = None
    prompt_history_enabled: Annotated[
        Optional[bool],
        Field(
            default=None,
            validation_alias=AliasChoices("prompt_history_enabled", "chat_history_enabled"),
            serialization_alias="prompt_history_enabled",
        ),
    ]
    chat_history_lookback: Optional[int] = Field(None, ge=1, le=50)
    summary_threshold_tokens: Optional[int] = Field(None, ge=500, le=100000)
    summary_keep_messages: Optional[int] = Field(None, ge=1, le=50)
    chat_visible: Optional[bool] = None
    persona_mode: Optional[str] = Field(None, max_length=50)
    persona_id: Optional[str] = None
    include_user_context: Optional[bool] = None
    include_datetime_context: Optional[bool] = None
    include_user_facts: Optional[bool] = None
    include_facts_categories: Optional[List[str]] = None
    use_themed_memory: Optional[bool] = None
    include_agent_memory: Optional[bool] = None
    auto_routable: Optional[bool] = None
    is_locked: Optional[bool] = None
    data_workspace_config: Optional[Dict[str, Any]] = None
    allowed_connections: Optional[List[Dict[str, Any]]] = None
    category: Optional[str] = Field(None, max_length=100)

    @field_validator("include_facts_categories", mode="before")
    @classmethod
    def coerce_json_list(cls, v: Any) -> Optional[List[str]]:
        if v is None:
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            if not v.strip():
                return []
            try:
                parsed = json.loads(v)
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                return []
        return None

    @field_validator("knowledge_config", "journal_config", "team_config", "watch_config", "data_workspace_config", mode="before")
    @classmethod
    def coerce_json_dict(cls, v: Any) -> Optional[Dict[str, Any]]:
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            if not v.strip():
                return {}
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return None


class AgentDataSourceCreate(BaseModel):
    connector_id: str = Field(..., description="UUID of data_source_connectors row")
    credentials_encrypted: Optional[Dict[str, Any]] = None
    config_overrides: Dict[str, Any] = Field(default_factory=dict)
    permissions: Dict[str, Any] = Field(default_factory=dict)
    is_enabled: bool = True


class DataSourceFromTemplateCreate(BaseModel):
    template_name: str = Field(..., description="Name of template from connector-templates list")


class ConnectorCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    connector_type: str = Field(default="rest", max_length=50)
    definition: Dict[str, Any] = Field(default_factory=dict)
    requires_auth: bool = False
    auth_fields: List[Dict[str, Any]] = Field(default_factory=list)


class ConnectorUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    connector_type: Optional[str] = Field(None, max_length=50)
    definition: Optional[Dict[str, Any]] = None
    requires_auth: Optional[bool] = None
    auth_fields: Optional[List[Dict[str, Any]]] = None
    is_locked: Optional[bool] = None
    category: Optional[str] = Field(None, max_length=100)


class ConnectorFromTemplateCreate(BaseModel):
    template_name: str = Field(..., description="Name of template from connector-templates list")


class AgentDataSourceUpdate(BaseModel):
    credentials_encrypted: Optional[Dict[str, Any]] = None
    config_overrides: Optional[Dict[str, Any]] = None
    permissions: Optional[Dict[str, Any]] = None
    is_enabled: Optional[bool] = None


class CustomPlaybookCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    version: str = Field(default="1.0", max_length=20)
    definition: Dict[str, Any] = Field(default_factory=dict)
    triggers: List[Any] = Field(default_factory=list)
    is_template: bool = False
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    required_connectors: List[str] = Field(default_factory=list)


class CustomPlaybookUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    version: Optional[str] = Field(None, max_length=20)
    definition: Optional[Dict[str, Any]] = None
    triggers: Optional[List[Any]] = None
    is_template: Optional[bool] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    required_connectors: Optional[List[str]] = None
    is_locked: Optional[bool] = None


class SkillCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=100)
    procedure: str = Field(..., min_length=1)
    required_tools: List[str] = Field(default_factory=list)
    required_connection_types: List[str] = Field(default_factory=list)
    optional_tools: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    category: Optional[str] = Field(None, max_length=100)
    inputs_schema: Dict[str, Any] = Field(default_factory=dict)
    outputs_schema: Dict[str, Any] = Field(default_factory=dict)
    examples: List[Any] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    is_core: bool = Field(default=False, description="Core skills always appear in the condensed skill catalog")
    depends_on: List[str] = Field(default_factory=list, description="Slugs of dependency skills")


class SkillUpdate(BaseModel):
    procedure: Optional[str] = None
    required_tools: Optional[List[str]] = None
    required_connection_types: Optional[List[str]] = None
    optional_tools: Optional[List[str]] = None
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    category: Optional[str] = Field(None, max_length=100)
    inputs_schema: Optional[Dict[str, Any]] = None
    outputs_schema: Optional[Dict[str, Any]] = None
    examples: Optional[List[Any]] = None
    tags: Optional[List[str]] = None
    improvement_rationale: Optional[str] = None
    evidence_metadata: Optional[Dict[str, Any]] = None
    is_core: Optional[bool] = None
    depends_on: Optional[List[str]] = None
    as_candidate: bool = False


class SidebarCategoryCreate(BaseModel):
    section: str = Field(..., description="agents | playbooks | skills | connectors")
    name: str = Field(..., min_length=1, max_length=255)


class SidebarCategoryUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    sort_order: Optional[int] = None


class AgentScheduleCreate(BaseModel):
    schedule_type: str = Field(..., description="cron or interval")
    cron_expression: Optional[str] = Field(None, description="Cron expression e.g. 0 8 * * 1-5")
    interval_seconds: Optional[int] = Field(None, description="Seconds between runs for interval type")
    timezone: Optional[str] = Field(None, description="IANA timezone e.g. America/New_York; omit to use your profile timezone")
    timeout_seconds: int = Field(default=300, ge=60, le=3600)
    max_consecutive_failures: int = Field(default=5, ge=1, le=20)
    input_context: Dict[str, Any] = Field(default_factory=dict)


class AgentScheduleUpdate(BaseModel):
    schedule_type: Optional[str] = None
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    timezone: Optional[str] = Field(None, max_length=100)
    timeout_seconds: Optional[int] = Field(None, ge=60, le=3600)
    max_consecutive_failures: Optional[int] = Field(None, ge=1, le=20)
    input_context: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


# ---------- Actions (gRPC) ----------

async def _get_actions_from_orchestrator() -> List[Any]:
    """Call orchestrator gRPC GetActions and return parsed actions list."""
    if not GRPC_AVAILABLE:
        return []
    orchestrator_host = "llm-orchestrator"
    orchestrator_port = 50051
    try:
        async with grpc.aio.insecure_channel(f"{orchestrator_host}:{orchestrator_port}") as channel:
            stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)
            response = await stub.GetActions(orchestrator_pb2.GetActionsRequest())
            data = json.loads(response.actions_json)
            if isinstance(data, dict) and "actions" in data:
                return data["actions"]
            if isinstance(data, list):
                return data
            return []
    except Exception as e:
        logger.warning("GetActions gRPC call failed: %s", e)
        return []


async def _get_plugins_from_orchestrator() -> List[Dict[str, Any]]:
    """Call orchestrator gRPC GetPlugins and return list of plugin info."""
    if not GRPC_AVAILABLE:
        return []
    orchestrator_host = "llm-orchestrator"
    orchestrator_port = 50051
    try:
        async with grpc.aio.insecure_channel(f"{orchestrator_host}:{orchestrator_port}") as channel:
            stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)
            response = await stub.GetPlugins(orchestrator_pb2.GetPluginsRequest())
            return [
                {
                    "name": p.name,
                    "version": p.version,
                    "connection_requirements": [
                        {"key": f.key, "label": f.label}
                        for f in p.connection_requirements
                    ],
                }
                for p in response.plugins
            ]
    except Exception as e:
        logger.warning("GetPlugins gRPC call failed: %s", e)
        return []


def _connector_endpoint_input_fields(endpoint_def: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build input_fields list from connector endpoint params (query/path/body)."""
    params = endpoint_def.get("params") or []
    if isinstance(params, dict):
        params = [{"name": k, "in": "query", "default": v} for k, v in params.items()]
    fields = []
    for p in params:
        name = p.get("name") or p.get("id")
        if not name:
            continue
        fields.append({
            "name": name,
            "type": "text",
            "description": p.get("description") or f"Parameter: {name}",
            "required": p.get("required", False),
            "default": p.get("default"),
        })
    return fields


async def _get_connector_actions_for_profile(profile_id: str, user_id: str) -> List[Dict[str, Any]]:
    """
    Load agent_data_sources for profile, then for each connector and endpoint
    build a synthetic action with name connector:<connector_id>:<endpoint_id>.
    """
    from services.database_manager.database_helpers import fetch_all, fetch_one

    profile = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        user_id,
    )
    if not profile:
        return []

    rows = await fetch_all(
        "SELECT ads.connector_id, dsc.name AS connector_name, dsc.definition "
        "FROM agent_data_sources ads "
        "JOIN data_source_connectors dsc ON dsc.id = ads.connector_id "
        "WHERE ads.agent_profile_id = $1 AND ads.is_enabled = true",
        profile_id,
    )
    actions = []
    for row in rows:
        connector_id = str(row["connector_id"])
        connector_name = row.get("connector_name") or "Connector"
        definition = row.get("definition") or {}
        if isinstance(definition, str):
            try:
                definition = json.loads(definition)
            except json.JSONDecodeError:
                continue
        endpoints = definition.get("endpoints") or {}
        if isinstance(endpoints, list):
            endpoints = {ep.get("id") or ep.get("name"): ep for ep in endpoints if ep.get("id") or ep.get("name")}
        for endpoint_id, endpoint_def in (endpoints or {}).items():
            if not endpoint_id:
                continue
            action_name = f"connector:{connector_id}:{endpoint_id}"
            desc = endpoint_def.get("description") or f"{connector_name} – {endpoint_id}"
            input_fields = _connector_endpoint_input_fields(endpoint_def)
            actions.append({
                "name": action_name,
                "category": "connector",
                "description": desc,
                "input_schema": {"type": "object", "properties": {f["name"]: {"type": "string"} for f in input_fields}},
                "params_schema": {},
                "output_schema": {"type": "object", "properties": {"records": {}, "count": {}, "formatted": {}}},
                "input_fields": input_fields,
                "output_fields": [
                    {"name": "records", "type": "list[record]", "description": "List of records from the endpoint"},
                    {"name": "count", "type": "number", "description": "Number of records"},
                    {"name": "formatted", "type": "text", "description": "Human-readable summary"},
                ],
            })
    return actions


# Email tools replaced by account-scoped variants from user connections (email:<connection_id>:<tool_name>).
EMAIL_CATEGORY_TOOL_NAMES = {"list_emails", "search_emails", "get_email_thread", "read_email", "move_email", "list_email_folders", "update_email", "create_draft", "get_email_statistics", "send_email", "reply_to_email"}

CALENDAR_CATEGORY_TOOL_NAMES = {
    "list_calendars",
    "get_calendar_events",
    "get_event_by_id",
    "create_event",
    "update_event",
    "delete_event",
}

EMAIL_TOOL_SPECS = [
    ("list_emails", "Get emails from {account}", [
        {"name": "folder", "type": "text", "description": "Folder (e.g. inbox)", "required": False, "default": "inbox"},
        {"name": "top", "type": "number", "description": "Max number of emails", "required": False, "default": 10},
        {"name": "unread_only", "type": "boolean", "description": "Only unread", "required": False, "default": False},
    ], [{"name": "emails", "type": "list[record]", "description": "Email refs"}, {"name": "count", "type": "number", "description": "Count"}, {"name": "formatted", "type": "text", "description": "Human-readable"}]),
    ("search_emails", "Search emails in {account}", [
        {"name": "query", "type": "text", "description": "Search query", "required": True},
        {"name": "top", "type": "number", "description": "Max results", "required": False, "default": 20},
    ], [{"name": "emails", "type": "list[record]", "description": "Matching emails"}, {"name": "count", "type": "number", "description": "Count"}, {"name": "formatted", "type": "text", "description": "Human-readable"}]),
    ("get_email_thread", "Get email thread from {account}", [
        {"name": "thread_id", "type": "text", "description": "Thread/conversation ID", "required": True},
    ], [{"name": "messages", "type": "list[record]", "description": "Messages in thread"}, {"name": "count", "type": "number", "description": "Count"}, {"name": "formatted", "type": "text", "description": "Human-readable"}]),
    ("read_email", "Read email from {account}", [
        {"name": "message_id", "type": "text", "description": "Email message ID", "required": True},
    ], [{"name": "message_id", "type": "text", "description": "Message ID"}, {"name": "formatted", "type": "text", "description": "Full email content"}]),
    ("move_email", "Move email in {account}", [
        {"name": "message_id", "type": "text", "description": "Message ID to move", "required": True},
        {"name": "destination_folder_id", "type": "text", "description": "Target folder ID", "required": True},
    ], [{"name": "success", "type": "boolean", "description": "Whether move succeeded"}, {"name": "formatted", "type": "text", "description": "Human-readable"}]),
    ("list_email_folders", "List folders in {account}", [], [{"name": "folders", "type": "list[record]", "description": "Mailbox folders"}, {"name": "formatted", "type": "text", "description": "Human-readable"}]),
    ("update_email", "Update email in {account}", [
        {"name": "message_id", "type": "text", "description": "Message ID to update", "required": True},
        {"name": "is_read", "type": "boolean", "description": "Set read state", "required": False},
        {"name": "importance", "type": "text", "description": "low, normal, high", "required": False},
    ], [{"name": "success", "type": "boolean", "description": "Whether update succeeded"}, {"name": "formatted", "type": "text", "description": "Human-readable"}]),
    ("create_draft", "Create draft in {account}", [
        {"name": "to", "type": "text", "description": "Recipients (comma-separated)", "required": True},
        {"name": "subject", "type": "text", "description": "Subject", "required": True},
        {"name": "body", "type": "text", "description": "Body", "required": True},
    ], [{"name": "success", "type": "boolean", "description": "Whether draft was created"}, {"name": "message_id", "type": "text", "description": "Draft message ID"}, {"name": "formatted", "type": "text", "description": "Human-readable"}]),
    ("get_email_statistics", "Get email statistics for {account}", [], [{"name": "total_messages", "type": "number", "description": "Total"}, {"name": "unread_count", "type": "number", "description": "Unread"}, {"name": "formatted", "type": "text", "description": "Human-readable"}]),
    ("send_email", "Send email from {account}", [
        {"name": "to", "type": "text", "description": "Recipients", "required": True},
        {"name": "subject", "type": "text", "description": "Subject", "required": True},
        {"name": "body", "type": "text", "description": "Body", "required": True},
        {"name": "confirmed", "type": "boolean", "description": "User confirmed send", "required": False, "default": False},
    ], [{"name": "success", "type": "boolean", "description": "Whether sent"}, {"name": "message_id", "type": "text", "description": "Provider message ID"}, {"name": "formatted", "type": "text", "description": "Human-readable"}]),
    ("reply_to_email", "Reply to email from {account}", [
        {"name": "message_id", "type": "text", "description": "Message ID to reply to", "required": True},
        {"name": "body", "type": "text", "description": "Reply body", "required": True},
        {"name": "reply_all", "type": "boolean", "description": "Reply all", "required": False, "default": False},
        {"name": "confirmed", "type": "boolean", "description": "User confirmed", "required": False, "default": False},
    ], [{"name": "success", "type": "boolean", "description": "Whether sent"}, {"name": "message_id", "type": "text", "description": "New message ID"}, {"name": "formatted", "type": "text", "description": "Human-readable"}]),
]


CALENDAR_TOOL_SPECS = [
    ("list_calendars", "List calendars for {account}", [], [
        {"name": "calendars", "type": "list[record]", "description": "Calendars"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("get_calendar_events", "Get calendar events for {account}", [
        {"name": "calendar_id", "type": "text", "description": "Calendar id", "required": True},
        {"name": "start_datetime", "type": "text", "description": "Range start (ISO 8601)", "required": False, "default": ""},
        {"name": "end_datetime", "type": "text", "description": "Range end (ISO 8601)", "required": False, "default": ""},
    ], [
        {"name": "events", "type": "list[record]", "description": "Events"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("get_event_by_id", "Get a single event from {account}", [
        {"name": "calendar_id", "type": "text", "description": "Calendar id", "required": True},
        {"name": "event_id", "type": "text", "description": "Event id", "required": True},
    ], [
        {"name": "event", "type": "record", "description": "Event"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("create_event", "Create event in {account}", [
        {"name": "calendar_id", "type": "text", "description": "Calendar id", "required": True},
        {"name": "subject", "type": "text", "description": "Title", "required": True},
        {"name": "start_datetime", "type": "text", "description": "Start (ISO 8601)", "required": True},
        {"name": "end_datetime", "type": "text", "description": "End (ISO 8601)", "required": True},
    ], [
        {"name": "success", "type": "boolean", "description": "Created"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("update_event", "Update event in {account}", [
        {"name": "calendar_id", "type": "text", "description": "Calendar id", "required": True},
        {"name": "event_id", "type": "text", "description": "Event id", "required": True},
        {"name": "subject", "type": "text", "description": "New subject", "required": False},
    ], [
        {"name": "success", "type": "boolean", "description": "Updated"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("delete_event", "Delete event in {account}", [
        {"name": "calendar_id", "type": "text", "description": "Calendar id", "required": True},
        {"name": "event_id", "type": "text", "description": "Event id", "required": True},
    ], [
        {"name": "success", "type": "boolean", "description": "Deleted"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
]


async def _get_email_actions_for_user(user_id: str) -> List[Dict[str, Any]]:
    """Per-connection email actions for all active email external_connections (no profile binding)."""
    from services.database_manager.database_helpers import fetch_all

    rows = await fetch_all(
        """
        SELECT id AS connection_id, account_identifier, display_name, provider
        FROM external_connections
        WHERE user_id = $1 AND connection_type = 'email' AND is_active = true
        ORDER BY id
        """,
        user_id,
    )
    actions: List[Dict[str, Any]] = []
    for row in rows:
        connection_id = int(row["connection_id"])
        account = (row.get("display_name") or row.get("account_identifier") or "").strip() or row.get("account_identifier") or f"connection {connection_id}"
        provider = (row.get("provider") or "").strip()
        if provider:
            account = f"{account} ({provider})"
        for tool_name, desc_tpl, input_fields, output_fields in EMAIL_TOOL_SPECS:
            action_name = f"email:{connection_id}:{tool_name}"
            description = desc_tpl.format(account=account)
            actions.append({
                "name": action_name,
                "category": "email",
                "description": description,
                "input_schema": {"type": "object", "properties": {f["name"]: {"type": "string"} for f in input_fields}},
                "params_schema": {},
                "output_schema": {"type": "object", "properties": {"formatted": {"type": "string"}}},
                "input_fields": input_fields,
                "output_fields": output_fields,
            })
    return actions


async def _get_calendar_actions_for_user(user_id: str) -> List[Dict[str, Any]]:
    """Per-connection calendar actions: CalDAV calendar rows plus Microsoft 365 email (Graph calendar scopes)."""
    from services.database_manager.database_helpers import fetch_all

    rows = await fetch_all(
        """
        SELECT id AS connection_id, account_identifier, display_name, provider, connection_type
        FROM external_connections
        WHERE user_id = $1 AND is_active = true
          AND (
            connection_type = 'calendar'
            OR (connection_type = 'email' AND LOWER(TRIM(COALESCE(provider, ''))) = 'microsoft')
          )
        ORDER BY id
        """,
        user_id,
    )
    actions: List[Dict[str, Any]] = []
    for row in rows:
        connection_id = int(row["connection_id"])
        account = (row.get("display_name") or row.get("account_identifier") or "").strip() or row.get("account_identifier") or f"connection {connection_id}"
        provider = (row.get("provider") or "").strip()
        ctype = (row.get("connection_type") or "").strip()
        if provider:
            account = f"{account} ({provider})"
        if ctype == "email" and (provider or "").strip().lower() == "microsoft":
            account = f"{account} — calendar (Microsoft 365)"
        for tool_name, desc_tpl, input_fields, output_fields in CALENDAR_TOOL_SPECS:
            action_name = f"calendar:{connection_id}:{tool_name}"
            description = desc_tpl.format(account=account)
            actions.append({
                "name": action_name,
                "category": "calendar",
                "description": description,
                "input_schema": {"type": "object", "properties": {f["name"]: {"type": "string"} for f in input_fields}},
                "params_schema": {},
                "output_schema": {"type": "object", "properties": {"formatted": {"type": "string"}}},
                "input_fields": input_fields,
                "output_fields": output_fields,
            })
    return actions


# (tool_name, description_template, input_fields, output_fields) for Workflow Composer.
# Tool names and order must match llm-orchestrator/orchestrator/tools/tool_pack_registry.py GITHUB_PACK_ALL_TOOLS.
_GITHUB_TOOL_ROWS = [
    ("github_list_repos", "List GitHub repositories for {account}", [], [
        {"name": "records", "type": "list[record]", "description": "Raw API records"},
        {"name": "count", "type": "number", "description": "Count"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("github_get_repo", "Get repo metadata for {account}", [
        {"name": "owner", "type": "text", "description": "Owner", "required": True},
        {"name": "repo", "type": "text", "description": "Repository name", "required": True},
    ], [
        {"name": "records", "type": "list[record]", "description": "Repo object"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("github_list_issues", "List issues in {account}", [
        {"name": "owner", "type": "text", "description": "Owner", "required": True},
        {"name": "repo", "type": "text", "description": "Repository", "required": True},
        {"name": "state", "type": "text", "description": "open, closed, all", "required": False, "default": "open"},
    ], [
        {"name": "records", "type": "list[record]", "description": "Issues"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("github_get_issue", "Get issue in {account}", [
        {"name": "owner", "type": "text", "description": "Owner", "required": True},
        {"name": "repo", "type": "text", "description": "Repository", "required": True},
        {"name": "issue_number", "type": "number", "description": "Issue number", "required": True},
    ], [
        {"name": "records", "type": "list[record]", "description": "Issue"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("github_list_issue_comments", "List issue/PR comments in {account}", [
        {"name": "owner", "type": "text", "description": "Owner", "required": True},
        {"name": "repo", "type": "text", "description": "Repository", "required": True},
        {"name": "issue_number", "type": "number", "description": "Issue or PR number", "required": True},
    ], [
        {"name": "records", "type": "list[record]", "description": "Comments"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("github_list_pulls", "List pull requests in {account}", [
        {"name": "owner", "type": "text", "description": "Owner", "required": True},
        {"name": "repo", "type": "text", "description": "Repository", "required": True},
        {"name": "state", "type": "text", "description": "open, closed, all", "required": False, "default": "open"},
    ], [
        {"name": "records", "type": "list[record]", "description": "PRs"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("github_get_pull", "Get pull request in {account}", [
        {"name": "owner", "type": "text", "description": "Owner", "required": True},
        {"name": "repo", "type": "text", "description": "Repository", "required": True},
        {"name": "pull_number", "type": "number", "description": "PR number", "required": True},
    ], [
        {"name": "records", "type": "list[record]", "description": "PR"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("github_get_pull_diff", "PR changed files and patches in {account}", [
        {"name": "owner", "type": "text", "description": "Owner", "required": True},
        {"name": "repo", "type": "text", "description": "Repository", "required": True},
        {"name": "pull_number", "type": "number", "description": "PR number", "required": True},
    ], [
        {"name": "records", "type": "list[record]", "description": "Files"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("github_list_pull_reviews", "List PR reviews in {account}", [
        {"name": "owner", "type": "text", "description": "Owner", "required": True},
        {"name": "repo", "type": "text", "description": "Repository", "required": True},
        {"name": "pull_number", "type": "number", "description": "PR number", "required": True},
    ], [
        {"name": "records", "type": "list[record]", "description": "Reviews"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("github_list_pull_comments", "List PR review line comments in {account}", [
        {"name": "owner", "type": "text", "description": "Owner", "required": True},
        {"name": "repo", "type": "text", "description": "Repository", "required": True},
        {"name": "pull_number", "type": "number", "description": "PR number", "required": True},
    ], [
        {"name": "records", "type": "list[record]", "description": "Comments"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("github_list_commits", "List commits in {account}", [
        {"name": "owner", "type": "text", "description": "Owner", "required": True},
        {"name": "repo", "type": "text", "description": "Repository", "required": True},
    ], [
        {"name": "records", "type": "list[record]", "description": "Commits"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("github_get_commit", "Get commit in {account}", [
        {"name": "owner", "type": "text", "description": "Owner", "required": True},
        {"name": "repo", "type": "text", "description": "Repository", "required": True},
        {"name": "sha", "type": "text", "description": "Commit SHA", "required": True},
    ], [
        {"name": "records", "type": "list[record]", "description": "Commit"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("github_compare_refs", "Compare two refs in {account}", [
        {"name": "owner", "type": "text", "description": "Owner", "required": True},
        {"name": "repo", "type": "text", "description": "Repository", "required": True},
        {"name": "base_ref", "type": "text", "description": "Base branch/tag/SHA", "required": True},
        {"name": "head_ref", "type": "text", "description": "Head branch/tag/SHA", "required": True},
    ], [
        {"name": "records", "type": "list[record]", "description": "Compare result"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("github_get_file_content", "Read file or directory in {account}", [
        {"name": "owner", "type": "text", "description": "Owner", "required": True},
        {"name": "repo", "type": "text", "description": "Repository", "required": True},
        {"name": "path", "type": "text", "description": "File path", "required": True},
    ], [
        {"name": "records", "type": "list[record]", "description": "Content metadata"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("github_list_branches", "List branches in {account}", [
        {"name": "owner", "type": "text", "description": "Owner", "required": True},
        {"name": "repo", "type": "text", "description": "Repository", "required": True},
    ], [
        {"name": "records", "type": "list[record]", "description": "Branches"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("github_search_code", "Search code (GitHub query syntax) in {account}", [
        {"name": "q", "type": "text", "description": "Search query", "required": True},
    ], [
        {"name": "records", "type": "list[record]", "description": "Matches"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("github_create_issue", "Create issue in {account}", [
        {"name": "owner", "type": "text", "description": "Owner", "required": True},
        {"name": "repo", "type": "text", "description": "Repository", "required": True},
        {"name": "title", "type": "text", "description": "Title", "required": True},
        {"name": "body", "type": "text", "description": "Body", "required": False},
    ], [
        {"name": "records", "type": "list[record]", "description": "Created issue"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("github_create_issue_comment", "Comment on issue/PR in {account}", [
        {"name": "owner", "type": "text", "description": "Owner", "required": True},
        {"name": "repo", "type": "text", "description": "Repository", "required": True},
        {"name": "issue_number", "type": "number", "description": "Issue or PR number", "required": True},
        {"name": "body", "type": "text", "description": "Comment body", "required": True},
    ], [
        {"name": "records", "type": "list[record]", "description": "Comment"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    ("github_create_pr_review", "Submit PR review in {account}", [
        {"name": "owner", "type": "text", "description": "Owner", "required": True},
        {"name": "repo", "type": "text", "description": "Repository", "required": True},
        {"name": "pull_number", "type": "number", "description": "PR number", "required": True},
        {"name": "event", "type": "text", "description": "COMMENT, APPROVE, REQUEST_CHANGES", "required": True},
        {"name": "body", "type": "text", "description": "Review body", "required": False},
    ], [
        {"name": "records", "type": "list[record]", "description": "Review"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
]

# Registry names for the GitHub pack; same order as _GITHUB_TOOL_ROWS. Filter duplicate base actions in get_actions.
GITHUB_CATEGORY_TOOL_NAMES = frozenset(row[0] for row in _GITHUB_TOOL_ROWS)


async def _get_github_actions_for_user(user_id: str) -> List[Dict[str, Any]]:
    """Per-connection GitHub actions for code_platform OAuth connections."""
    from services.database_manager.database_helpers import fetch_all

    rows = await fetch_all(
        """
        SELECT id AS connection_id, account_identifier, display_name, provider
        FROM external_connections
        WHERE user_id = $1 AND connection_type = 'code_platform' AND provider = 'github' AND is_active = true
        ORDER BY id
        """,
        user_id,
    )
    actions: List[Dict[str, Any]] = []
    for row in rows:
        connection_id = int(row["connection_id"])
        account = (row.get("display_name") or row.get("account_identifier") or "").strip() or row.get("account_identifier") or f"github {connection_id}"
        provider = (row.get("provider") or "").strip()
        if provider:
            account = f"{account} ({provider})"
        for tool_name, desc_tpl, input_fields, output_fields in _GITHUB_TOOL_ROWS:
            action_name = f"github:{connection_id}:{tool_name}"
            description = desc_tpl.format(account=account)
            actions.append({
                "name": action_name,
                "category": "github",
                "description": description,
                "input_schema": {"type": "object", "properties": {f["name"]: {"type": "string"} for f in input_fields}},
                "params_schema": {},
                "output_schema": {"type": "object", "properties": {"formatted": {"type": "string"}}},
                "input_fields": input_fields,
                "output_fields": output_fields,
            })
    return actions


async def _get_mcp_actions_for_user(user_id: str) -> List[Dict[str, Any]]:
    """MCP tools from discovered_tools on each active mcp_servers row."""
    from services.database_manager.database_helpers import fetch_all

    rows = await fetch_all(
        """
        SELECT id, name, discovered_tools
        FROM mcp_servers
        WHERE user_id = $1 AND is_active = true
        ORDER BY LOWER(name) ASC NULLS LAST
        """,
        user_id,
    )
    actions: List[Dict[str, Any]] = []
    for row in rows:
        sid = int(row["id"])
        sname = (row.get("name") or "").strip() or f"server_{sid}"
        raw = row.get("discovered_tools")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw) if raw else []
            except json.JSONDecodeError:
                raw = []
        if not isinstance(raw, list):
            continue
        for tool in raw:
            if isinstance(tool, dict):
                tname = tool.get("name")
                tdesc = (tool.get("description") or "").strip()
            elif isinstance(tool, str):
                tname = tool
                tdesc = ""
            else:
                continue
            if not tname:
                continue
            action_name = f"mcp:{sid}:{tname}"
            description = tdesc or f"MCP tool {tname} ({sname})"
            actions.append({
                "name": action_name,
                "category": f"MCP: {sname}",
                "description": description,
                "input_schema": {"type": "object", "properties": {"arguments_json": {"type": "string"}}},
                "params_schema": {},
                "output_schema": {"type": "object", "properties": {"formatted": {"type": "string"}, "result_json": {"type": "string"}}},
                "input_fields": [
                    {
                        "name": "arguments_json",
                        "type": "text",
                        "description": "JSON object of parameters for this MCP tool",
                        "required": False,
                        "default": "{}",
                    }
                ],
                "output_fields": [
                    {"name": "formatted", "type": "text", "description": "Human-readable result"},
                    {"name": "result_json", "type": "text", "description": "Raw JSON payload"},
                ],
            })
    return actions


# DEPRECATED: Tool packs replaced by skills in Skills-First Architecture.
# Kept for backward compat with stored playbooks that still reference tool_packs.
# NOTE: Remove after legacy playbooks no longer reference tool_packs (post UI re-save migration).
TOOL_PACKS_LIST: List[Dict[str, Any]] = [
    {"name": "text_transforms", "description": "Text manipulation: summarize, extract, format conversion, merge, compare", "has_write_tools": False, "read_tool_count": 5},
    {"name": "session_memory", "description": "Ephemeral clipboard for passing data between plan steps", "has_write_tools": True, "read_tool_count": 1},
    {"name": "planning", "description": "Self-managed task planning: create, track, and adapt a multi-step plan", "has_write_tools": True, "read_tool_count": 1},
    {"name": "discovery", "description": "Search across the knowledge base (semantic), by tags, images, and the web", "has_write_tools": False, "read_tool_count": 6},
    {"name": "knowledge", "description": "Read or search within documents when you have a document_id or path", "has_write_tools": False, "read_tool_count": 4},
    {"name": "knowledge_graph", "description": "Entity-driven: find documents by person, organisation, location; traverse relationships", "has_write_tools": False, "read_tool_count": 5},
    {"name": "rss", "description": "RSS feed management and article retrieval", "has_write_tools": True, "read_tool_count": 5},
    {"name": "document_management", "description": "Create typed documents with frontmatter templates, update content, read metadata", "has_write_tools": True, "read_tool_count": 1},
    {"name": "file_management", "description": "Create raw files and organize folders; patch_file and append_to_file", "has_write_tools": True, "read_tool_count": 1},
    {"name": "org_management", "description": "Org-mode file structure parsing and headings search (read-only)", "has_write_tools": False, "read_tool_count": 3},
    {"name": "task_management", "description": "Universal todo list: create, update, toggle, delete, archive, refile", "has_write_tools": True, "read_tool_count": 2},
    {"name": "math", "description": "Math calculations, formula evaluation, unit conversions", "has_write_tools": False, "read_tool_count": 4},
    {"name": "utility", "description": "State management: counters, dates, booleans, lists", "has_write_tools": True, "read_tool_count": 3},
    {"name": "contacts", "description": "O365 and org-mode contacts: list, get, create, update, delete", "has_write_tools": True, "read_tool_count": 2},
    {"name": "notifications", "description": "Send messages via in-app, Telegram, Discord, or email", "has_write_tools": True, "read_tool_count": 0},
    {"name": "email", "description": "Read, search, send, draft, and manage emails via O365/Microsoft Graph", "has_write_tools": True, "read_tool_count": 6},
    {"name": "calendar", "description": "Read and manage O365 calendar events", "has_write_tools": True, "read_tool_count": 3},
    {"name": "todo", "description": "Microsoft To Do lists and tasks (Microsoft 365)", "has_write_tools": True, "read_tool_count": 2},
    {"name": "files", "description": "OneDrive files and folders (Microsoft 365)", "has_write_tools": True, "read_tool_count": 3},
    {"name": "onenote", "description": "OneNote notebooks, sections, and pages (Microsoft 365)", "has_write_tools": True, "read_tool_count": 3},
    {"name": "planner", "description": "Microsoft Planner plans and tasks (Microsoft 365)", "has_write_tools": True, "read_tool_count": 2},
    {"name": "github", "description": "GitHub repos, issues, PRs, diffs, and code via OAuth", "has_write_tools": True, "read_tool_count": 15},
    {"name": "gitea", "description": "Gitea repos, issues, PRs (same tools as GitHub pack) via personal access token", "has_write_tools": True, "read_tool_count": 15},
    {"name": "navigation", "description": "Save named locations; compute and save routes", "has_write_tools": True, "read_tool_count": 3},
    {"name": "data_workspace", "description": "Query tabular data in Data Workspaces", "has_write_tools": False, "read_tool_count": 3},
    {"name": "image_generation", "description": "Generate images from text descriptions", "has_write_tools": True, "read_tool_count": 0},
    {"name": "visualization", "description": "Create charts (bar, line, pie, scatter) from data", "has_write_tools": True, "read_tool_count": 0},
    {"name": "data_connection_builder", "description": "Analyze APIs, build data connections, bulk scrape, control panes", "has_write_tools": True, "read_tool_count": 10},
    {"name": "browser", "description": "Browser automation: navigate, click, fill, extract, screenshot", "has_write_tools": True, "read_tool_count": 5},
    {"name": "local_device", "description": "Tools from Bastion Local Proxy (screenshot, clipboard, shell, etc.)", "has_write_tools": True, "read_tool_count": 6},
    {"name": "team_tools", "description": "Agent team tools: messaging, tasks, goals, workspace (auto-injected in team context)", "has_write_tools": True, "read_tool_count": 7},
]

# Tool names per pack (for UI overlap hint). Must stay in sync with llm-orchestrator tool_pack_registry.
PACK_TOOLS: Dict[str, List[str]] = {
    "text_transforms": ["summarize_text_tool", "extract_structured_data_tool", "transform_format_tool", "merge_texts_tool", "compare_texts_tool"],
    "session_memory": ["clipboard_store_tool", "clipboard_get_tool"],
    "planning": ["create_plan_tool", "get_plan_tool", "update_plan_step_tool", "add_plan_step_tool"],
    "discovery": ["search_documents_tool", "search_by_tags_tool", "search_images_tool", "search_web_tool", "enhance_query_tool"],
    "knowledge": ["get_document_content_tool", "find_document_by_path_tool", "search_within_document_tool", "search_segments_across_documents_tool"],
    "knowledge_graph": ["find_documents_by_entities_tool", "find_related_documents_by_entities_tool", "find_co_occurring_entities_tool", "search_entities_tool", "get_entity_tool"],
    "rss": [
        "list_rss_feeds_tool",
        "add_rss_feed_tool",
        "refresh_rss_feed_tool",
        "get_rss_articles_tool",
        "search_rss_tool",
        "list_starred_rss_articles_tool",
        "delete_rss_feed_tool",
        "mark_article_read_tool",
        "mark_article_unread_tool",
        "set_article_starred_tool",
        "get_unread_counts_tool",
        "toggle_feed_active_tool",
    ],
    "document_management": ["create_typed_document_tool", "update_document_content_tool", "get_document_metadata_tool"],
    "file_management": ["list_folders_tool", "create_user_file_tool", "create_user_folder_tool", "patch_file_tool", "append_to_file_tool"],
    "org_management": ["parse_org_structure_tool", "search_org_headings_tool", "get_org_statistics_tool"],
    "task_management": ["list_todos_tool", "create_todo_tool", "update_todo_tool", "toggle_todo_tool", "delete_todo_tool", "archive_done_tool", "refile_todo_tool", "discover_refile_targets_tool"],
    "math": ["calculate_expression_tool", "evaluate_formula_tool", "convert_units_tool", "list_available_formulas_tool"],
    "utility": ["adjust_number_tool", "adjust_date_tool", "parse_date_tool", "compare_dates_tool", "set_value_tool", "toggle_boolean_tool", "append_to_list_tool", "get_list_length_tool"],
    "contacts": [
        "get_contacts_tool",
        "get_contact_by_id_tool",
        "create_contact_tool",
        "update_contact_tool",
        "delete_contact_tool",
        "search_contacts_tool",
    ],
    "notifications": ["notify_user_tool", "send_channel_message_tool", "schedule_reminder_tool"],
    "email": ["get_emails_tool", "search_emails_tool", "get_email_thread_tool", "read_email_tool", "send_email_tool", "reply_to_email_tool", "create_draft_tool", "move_email_tool", "update_email_tool", "get_email_folders_tool", "get_email_statistics_tool"],
    "calendar": ["list_calendars_tool", "get_calendar_events_tool", "get_event_by_id_tool", "create_event_tool", "update_event_tool", "delete_event_tool"],
    "todo": [
        "list_todo_lists_tool",
        "get_todo_tasks_tool",
        "create_todo_task_tool",
        "update_todo_task_tool",
        "delete_todo_task_tool",
    ],
    "files": [
        "list_drive_items_tool",
        "get_drive_item_tool",
        "search_drive_tool",
        "get_onedrive_file_content_tool",
        "upload_onedrive_file_tool",
        "create_drive_folder_tool",
        "move_drive_item_tool",
        "delete_drive_item_tool",
    ],
    "onenote": [
        "list_onenote_notebooks_tool",
        "list_onenote_sections_tool",
        "list_onenote_pages_tool",
        "get_onenote_page_content_tool",
        "create_onenote_page_tool",
    ],
    "planner": [
        "list_planner_plans_tool",
        "get_planner_tasks_tool",
        "create_planner_task_tool",
        "update_planner_task_tool",
        "delete_planner_task_tool",
    ],
    "github": [row[0] for row in _GITHUB_TOOL_ROWS],
    "gitea": [row[0] for row in _GITHUB_TOOL_ROWS],
    "navigation": ["create_location_tool", "list_locations_tool", "delete_location_tool", "compute_route_tool", "save_route_tool", "list_saved_routes_tool"],
    "data_workspace": [
        "list_data_workspaces_tool",
        "get_workspace_schema_tool",
        "resolve_workspace_link_tool",
        "query_data_workspace_tool",
    ],
    "image_generation": ["generate_image_tool"],
    "visualization": ["create_chart_tool"],
    "data_connection_builder": ["probe_api_endpoint_tool", "analyze_openapi_spec_tool", "draft_connector_definition_tool", "validate_connector_definition_tool", "test_connector_endpoint_tool", "create_data_connector_tool", "list_data_connectors_tool", "update_data_connector_tool", "bulk_scrape_urls_tool", "get_bulk_scrape_status_tool", "bind_data_source_to_agent_tool", "list_control_panes_tool", "get_connector_endpoints_tool", "create_control_pane_tool", "update_control_pane_tool", "delete_control_pane_tool", "execute_control_action_tool", "crawl_web_content_tool", "search_web_tool"],
    "browser": ["browser_open_session_tool", "browser_navigate_tool", "browser_click_tool", "browser_fill_tool", "browser_wait_tool", "browser_scroll_tool", "browser_extract_tool", "browser_inspect_tool", "browser_screenshot_tool", "browser_download_file_tool", "browser_close_session_tool"],
    "local_device": ["local_screenshot_tool", "local_clipboard_read_tool", "local_clipboard_write_tool", "local_system_info_tool", "local_desktop_notify_tool", "local_shell_execute_tool", "local_read_file_tool", "local_list_directory_tool", "local_write_file_tool", "local_list_processes_tool", "local_open_url_tool"],
    "team_tools": ["send_to_agent", "start_agent_conversation", "halt_agent_conversation", "read_team_timeline", "read_my_messages", "get_team_status_board", "write_to_workspace", "read_workspace", "create_task_for_agent", "check_my_tasks", "update_task_status", "escalate_task", "list_team_goals", "report_goal_progress", "delegate_goal_to_tasks", "propose_hire", "propose_strategy_change", "get_agent_run_history"],
}


@router.get("/tool-packs", deprecated=True)
async def get_tool_packs(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """DEPRECATED: Tool packs replaced by skills in Skills-First Architecture. Kept for backward compat."""
    from services.database_manager.database_helpers import fetch_all

    out: List[Dict[str, Any]] = []
    for p in TOOL_PACKS_LIST:
        entry = dict(p)
        name = entry.get("name")
        entry["tools"] = PACK_TOOLS.get(name, [])
        if name in (
            "email",
            "calendar",
            "contacts",
            "todo",
            "files",
            "onenote",
            "planner",
            "github",
            "gitea",
        ):
            entry["type"] = "external"
            entry["available_connections"] = []
        else:
            entry["type"] = "builtin"
        out.append(entry)

    uid = current_user.user_id
    if uid:
        from services.agent_factory_m365_actions import (
            CONTACTS_CATEGORY_TOOL_NAMES as M365_CONTACTS_TOOL_NAMES,
            FILES_REGISTRY_NAMES,
            ONENOTE_REGISTRY_NAMES,
            PLANNER_REGISTRY_NAMES,
            TODO_REGISTRY_NAMES,
        )

        email_rows = await fetch_all(
            """
            SELECT id, account_identifier, display_name, provider
            FROM external_connections
            WHERE user_id = $1 AND connection_type = 'email' AND is_active = true
            ORDER BY id
            """,
            uid,
        )
        cal_rows = await fetch_all(
            """
            SELECT id, account_identifier, display_name, provider
            FROM external_connections
            WHERE user_id = $1 AND connection_type = 'calendar' AND is_active = true
            ORDER BY id
            """,
            uid,
        )
        for i, e in enumerate(out):
            if e.get("name") == "email":
                conns = []
                for r in email_rows:
                    cid = int(r["id"])
                    label = (r.get("display_name") or r.get("account_identifier") or "").strip() or str(cid)
                    prov = (r.get("provider") or "").strip()
                    if prov:
                        label = f"{label} ({prov})"
                    conns.append({"id": cid, "label": label})
                out[i] = {**e, "available_connections": conns, "tools": sorted(EMAIL_CATEGORY_TOOL_NAMES)}
                break
        for i, e in enumerate(out):
            if e.get("name") == "calendar":
                conns = []
                seen_cal = set()
                for r in cal_rows:
                    cid = int(r["id"])
                    if cid in seen_cal:
                        continue
                    seen_cal.add(cid)
                    label = (r.get("display_name") or r.get("account_identifier") or "").strip() or str(cid)
                    prov = (r.get("provider") or "").strip()
                    if prov:
                        label = f"{label} ({prov})"
                    conns.append({"id": cid, "label": label})
                for r in email_rows:
                    if (r.get("provider") or "").strip().lower() != "microsoft":
                        continue
                    cid = int(r["id"])
                    if cid in seen_cal:
                        continue
                    seen_cal.add(cid)
                    label = (r.get("display_name") or r.get("account_identifier") or "").strip() or str(cid)
                    prov = (r.get("provider") or "").strip()
                    if prov:
                        label = f"{label} ({prov})"
                    conns.append({"id": cid, "label": f"{label} — includes calendar"})
                out[i] = {**e, "available_connections": conns, "tools": sorted(CALENDAR_CATEGORY_TOOL_NAMES)}
                break

        ms_pack_conns: List[Dict[str, Any]] = []
        for r in email_rows:
            if (r.get("provider") or "").strip().lower() != "microsoft":
                continue
            cid = int(r["id"])
            label = (r.get("display_name") or r.get("account_identifier") or "").strip() or str(cid)
            prov = (r.get("provider") or "").strip()
            if prov:
                label = f"{label} ({prov})"
            ms_pack_conns.append({"id": cid, "label": f"{label} — Microsoft 365"})

        for i, e in enumerate(out):
            if e.get("name") == "contacts":
                out[i] = {
                    **e,
                    "available_connections": ms_pack_conns,
                    "tools": sorted(M365_CONTACTS_TOOL_NAMES),
                }
                break
        for pack_name, reg_names in (
            ("todo", TODO_REGISTRY_NAMES),
            ("files", FILES_REGISTRY_NAMES),
            ("onenote", ONENOTE_REGISTRY_NAMES),
            ("planner", PLANNER_REGISTRY_NAMES),
        ):
            for i, e in enumerate(out):
                if e.get("name") == pack_name:
                    out[i] = {**e, "available_connections": ms_pack_conns, "tools": sorted(reg_names)}
                    break

        gh_rows = await fetch_all(
            """
            SELECT id, account_identifier, display_name, provider
            FROM external_connections
            WHERE user_id = $1 AND connection_type = 'code_platform' AND provider = 'github' AND is_active = true
            ORDER BY id
            """,
            uid,
        )
        for i, e in enumerate(out):
            if e.get("name") == "github":
                conns = []
                for r in gh_rows:
                    cid = int(r["id"])
                    label = (r.get("display_name") or r.get("account_identifier") or "").strip() or str(cid)
                    prov = (r.get("provider") or "").strip()
                    if prov:
                        label = f"{label} ({prov})"
                    conns.append({"id": cid, "label": label})
                out[i] = {**e, "available_connections": conns, "tools": sorted(GITHUB_CATEGORY_TOOL_NAMES)}
                break

        gitea_rows = await fetch_all(
            """
            SELECT id, account_identifier, display_name, provider
            FROM external_connections
            WHERE user_id = $1 AND connection_type = 'code_platform' AND provider = 'gitea' AND is_active = true
            ORDER BY id
            """,
            uid,
        )
        for i, e in enumerate(out):
            if e.get("name") == "gitea":
                conns = []
                for r in gitea_rows:
                    cid = int(r["id"])
                    label = (r.get("display_name") or r.get("account_identifier") or "").strip() or str(cid)
                    prov = (r.get("provider") or "").strip()
                    if prov:
                        label = f"{label} ({prov})"
                    conns.append({"id": cid, "label": label})
                out[i] = {**e, "available_connections": conns, "tools": sorted(GITHUB_CATEGORY_TOOL_NAMES)}
                break

        mcp_rows = await fetch_all(
            """
            SELECT id, name, description, discovered_tools
            FROM mcp_servers
            WHERE user_id = $1 AND is_active = true
            ORDER BY LOWER(name) ASC NULLS LAST
            """,
            uid,
        )
        for r in mcp_rows:
            sid = int(r["id"])
            sname = (r.get("name") or "").strip() or f"server_{sid}"
            desc = (r.get("description") or "").strip() or f"MCP server: {sname}"
            raw = r.get("discovered_tools")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw) if raw else []
                except json.JSONDecodeError:
                    raw = []
            tool_names: List[str] = []
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict) and item.get("name"):
                        tool_names.append(str(item["name"]))
                    elif isinstance(item, str):
                        tool_names.append(item)
            out.append({
                "name": f"mcp:{sid}",
                "type": "external",
                "description": desc,
                "has_write_tools": True,
                "read_tool_count": 0,
                "available_connections": [{"id": sid, "label": sname}],
                "tools": tool_names,
            })

    return JSONResponse(content=out)


@router.get("/actions")
async def get_actions(
    profile_id: Optional[str] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """
    Return all registered action I/O contracts for the Workflow Composer.

    Each action includes name, category, description, input/output schemas,
    and input_fields / output_fields for wiring UI.

    Email, calendar, and MCP actions are merged from the user's external_connections and mcp_servers
    (no service binding required). If profile_id is provided, connector-derived actions are merged in.
    """
    actions = await _get_actions_from_orchestrator()
    # Replace individual local_proxy tools with a single meta-entry when a device is connected
    actions = [a for a in actions if a.get("category") != "local_proxy"]
    if current_user.user_id:
        from utils.websocket_manager import get_websocket_manager
        ws_manager = get_websocket_manager()
        devices = ws_manager.get_user_devices(current_user.user_id)
        if devices:
            actions.append({
                "name": "local_device_tools",
                "category": "Local Device",
                "description": "All tools from your connected local proxy (screenshot, clipboard, shell, filesystem, notifications, etc.). Available tools are determined at execution time based on your device's enabled capabilities.",
                "input_schema": {},
                "params_schema": {},
                "output_schema": {},
                "input_fields": [],
                "output_fields": [],
                "is_dynamic_pack": True,
            })

    connector_actions: List[Dict[str, Any]] = []
    if profile_id and current_user.user_id:
        connector_actions = await _get_connector_actions_for_profile(profile_id, current_user.user_id)

    email_actions: List[Dict[str, Any]] = []
    calendar_actions: List[Dict[str, Any]] = []
    contacts_actions: List[Dict[str, Any]] = []
    m365_todo_actions: List[Dict[str, Any]] = []
    m365_files_actions: List[Dict[str, Any]] = []
    m365_onenote_actions: List[Dict[str, Any]] = []
    m365_planner_actions: List[Dict[str, Any]] = []
    github_actions: List[Dict[str, Any]] = []
    mcp_actions: List[Dict[str, Any]] = []
    if current_user.user_id:
        from services.agent_factory_m365_actions import (
            CONTACTS_CATEGORY_TOOL_NAMES,
            M365_SCOPED_REGISTRY_NAMES,
            get_contacts_actions_for_user,
            get_m365_files_actions_for_user,
            get_m365_onenote_actions_for_user,
            get_m365_planner_actions_for_user,
            get_m365_todo_actions_for_user,
        )

        email_actions = await _get_email_actions_for_user(current_user.user_id)
        calendar_actions = await _get_calendar_actions_for_user(current_user.user_id)
        contacts_actions = await get_contacts_actions_for_user(current_user.user_id)
        m365_todo_actions = await get_m365_todo_actions_for_user(current_user.user_id)
        m365_files_actions = await get_m365_files_actions_for_user(current_user.user_id)
        m365_onenote_actions = await get_m365_onenote_actions_for_user(current_user.user_id)
        m365_planner_actions = await get_m365_planner_actions_for_user(current_user.user_id)
        github_actions = await _get_github_actions_for_user(current_user.user_id)
        mcp_actions = await _get_mcp_actions_for_user(current_user.user_id)

    actions = list(actions)
    if email_actions:
        actions = [a for a in actions if not (a.get("category") == "email" and a.get("name") in EMAIL_CATEGORY_TOOL_NAMES)]
    if calendar_actions:
        actions = [a for a in actions if not (a.get("category") == "calendar" and a.get("name") in CALENDAR_CATEGORY_TOOL_NAMES)]
    if contacts_actions:
        actions = [
            a
            for a in actions
            if not (a.get("category") == "contacts" and a.get("name") in CONTACTS_CATEGORY_TOOL_NAMES)
        ]
    if m365_todo_actions or m365_files_actions or m365_onenote_actions or m365_planner_actions:
        actions = [a for a in actions if a.get("name") not in M365_SCOPED_REGISTRY_NAMES]
    if github_actions:
        actions = [a for a in actions if not (a.get("category") == "github" and a.get("name") in GITHUB_CATEGORY_TOOL_NAMES)]

    list_accounts_action = next((a for a in actions if a.get("name") == "list_accounts"), None)
    if list_accounts_action:
        actions = [a for a in actions if a.get("name") != "list_accounts"]

    tail: List[Dict[str, Any]] = []
    tail.extend(connector_actions)
    if list_accounts_action:
        tail.append(list_accounts_action)
    tail.extend(email_actions)
    tail.extend(calendar_actions)
    tail.extend(contacts_actions)
    tail.extend(m365_todo_actions)
    tail.extend(m365_files_actions)
    tail.extend(m365_onenote_actions)
    tail.extend(m365_planner_actions)
    tail.extend(github_actions)
    tail.extend(mcp_actions)
    actions = actions + tail
    return JSONResponse(content=actions)


@router.get("/plugins")
async def get_plugins(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """
    Return available Agent Factory plugins and their connection requirements.
    Used by the frontend to show plugin configuration UI.
    """
    plugins = await _get_plugins_from_orchestrator()
    return JSONResponse(content=plugins)


class ExecuteConnectorRequest(BaseModel):
    """Request body for execute-connector."""
    profile_id: str = Field(..., description="Agent profile UUID (for credentials lookup)")
    connector_id: str = Field(..., description="Data source connector UUID")
    endpoint_id: str = Field(..., description="Endpoint id from connector definition")
    params: Dict[str, Any] = Field(default_factory=dict, description="Endpoint parameters")


class ConnectorTestRequest(BaseModel):
    """Request body for standalone connector test (no agent profile)."""
    endpoint_id: str = Field(..., description="Endpoint id from connector definition")
    params: Dict[str, Any] = Field(default_factory=dict, description="Endpoint parameters")
    credentials: Optional[Dict[str, Any]] = Field(None, description="Inline credentials for testing (API key, etc.)")
    connection_id: Optional[int] = Field(None, description="External Connection ID for OAuth token")


@router.get("/connector-templates")
async def get_connector_templates(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """
    Return pre-built connector templates. Use these to create new data source
    connectors (create connector from template, then attach to agent).
    """
    from services.connector_templates import CONNECTOR_TEMPLATES
    return JSONResponse(content=CONNECTOR_TEMPLATES)


@router.get("/connectors")
async def list_connectors(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """List all user-owned data source connectors (non-templates)."""
    from services.database_manager.database_helpers import fetch_all
    rows = await fetch_all(
        """
        SELECT id, name, description, connector_type, definition, created_at, updated_at
        FROM data_source_connectors
        WHERE user_id = $1 AND (is_template = false OR is_template IS NULL)
        ORDER BY LOWER(name) ASC NULLS LAST
        """,
        current_user.user_id,
    )
    result = []
    for r in rows:
        definition = r.get("definition") or {}
        if isinstance(definition, str):
            try:
                definition = json.loads(definition)
            except json.JSONDecodeError:
                definition = {}
        endpoints = definition.get("endpoints") or {}
        endpoint_count = len(endpoints) if isinstance(endpoints, dict) else 0
        result.append({
            "id": str(r["id"]),
            "name": r.get("name", ""),
            "description": r.get("description"),
            "connector_type": r.get("connector_type", "rest"),
            "endpoint_count": endpoint_count,
            "created_at": r.get("created_at").isoformat() if r.get("created_at") else None,
            "updated_at": r.get("updated_at").isoformat() if r.get("updated_at") else None,
        })
    return JSONResponse(content=result)


@router.get("/connectors/export")
async def export_connectors(
    ids: Optional[str] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Response:
    """Export data source connectors as YAML. Optional query 'ids' = comma-separated UUIDs; if omitted, export all user connectors."""
    from services.database_manager.database_helpers import fetch_all, fetch_one
    if ids:
        id_list = [s.strip() for s in ids.split(",") if s.strip()]
        if not id_list:
            rows = []
        else:
            placeholders = ", ".join(f"${i + 1}::uuid" for i in range(len(id_list)))
            rows = await fetch_all(
                f"""SELECT id, name, description, connector_type, version, definition,
                    icon, category, tags, requires_auth, auth_fields
                    FROM data_source_connectors
                    WHERE user_id = $1 AND id IN ({placeholders})""",
                current_user.user_id,
                *id_list,
            )
    else:
        rows = await fetch_all(
            """
            SELECT id, name, description, connector_type, version, definition,
                   icon, category, tags, requires_auth, auth_fields
            FROM data_source_connectors
            WHERE user_id = $1 AND (is_template = false OR is_template IS NULL)
            ORDER BY name
            """,
            current_user.user_id,
        )
    connectors_export: List[Dict[str, Any]] = []
    for r in rows:
        r = dict(r)
        definition = r.get("definition")
        if isinstance(definition, str):
            try:
                definition = json.loads(definition)
            except json.JSONDecodeError:
                definition = {}
        connectors_export.append({
            "name": r.get("name"),
            "description": r.get("description"),
            "connector_type": r.get("connector_type", "rest"),
            "version": r.get("version") or "1.0",
            "definition": definition or {},
            "icon": r.get("icon"),
            "category": r.get("category"),
            "tags": list(r["tags"]) if r.get("tags") else [],
            "requires_auth": r.get("requires_auth", False),
            "auth_fields": _ensure_json_obj(r.get("auth_fields"), []),
        })
    bundle = {
        "bastion_connectors_bundle": {
            "version": "1",
            "exported_at": datetime.now(timezone.utc).isoformat(),
        },
        "connectors": connectors_export,
    }
    yaml_str = yaml.dump(bundle, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return Response(
        content=yaml_str,
        media_type="application/x-yaml",
        headers={"Content-Disposition": 'attachment; filename="connectors.yaml"'},
    )


@router.post("/connectors/import")
async def import_connectors(
    body: Dict[str, Any],
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Import data source connectors from YAML. Body: { yaml: "..." }. Creates new connectors (name dedup with suffix)."""
    from services.database_manager.database_helpers import fetch_one, execute
    yaml_str = body.get("yaml")
    if not yaml_str or not isinstance(yaml_str, str):
        raise HTTPException(status_code=400, detail="Request body must include 'yaml' string")
    try:
        bundle = yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")
    if not isinstance(bundle, dict):
        raise HTTPException(status_code=400, detail="YAML must be a mapping")
    connectors_data = bundle.get("connectors")
    if not isinstance(connectors_data, list):
        raise HTTPException(status_code=400, detail="Bundle must include 'connectors' list")
    created: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for ds in connectors_data:
        if not isinstance(ds, dict):
            continue
        name = (ds.get("name") or "Imported connector")[:255]
        base_name = name
        suffix = 0
        while True:
            candidate = f"{base_name}_{suffix}" if suffix else base_name
            existing = await fetch_one(
                "SELECT id FROM data_source_connectors WHERE user_id = $1 AND name = $2",
                current_user.user_id,
                candidate,
            )
            if not existing:
                name = candidate
                break
            suffix += 1
        definition = ds.get("definition") or {}
        if isinstance(definition, str):
            try:
                definition = json.loads(definition)
            except json.JSONDecodeError:
                definition = {}
        tags = ds.get("tags")
        if not isinstance(tags, list):
            tags = []
        try:
            await execute(
                """
                INSERT INTO data_source_connectors (
                    user_id, name, description, connector_type, version, definition,
                    is_template, requires_auth, auth_fields, icon, category, tags
                ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, false, $7, $8::jsonb, $9, $10, $11)
                """,
                current_user.user_id,
                name,
                ds.get("description"),
                ds.get("connector_type", "rest"),
                ds.get("version", "1.0"),
                json.dumps(definition),
                ds.get("requires_auth", False),
                json.dumps(ds.get("auth_fields") or []),
                ds.get("icon"),
                ds.get("category"),
                tags,
            )
            row = await fetch_one(
                "SELECT id, name FROM data_source_connectors WHERE user_id = $1 AND name = $2 ORDER BY created_at DESC LIMIT 1",
                current_user.user_id,
                name,
            )
            if row:
                created.append({"id": str(row["id"]), "name": row["name"]})
        except Exception as e:
            errors.append({"name": name, "error": str(e)})
    return JSONResponse(content={"created": created, "errors": errors})


@router.get("/connectors/{connector_id}")
async def get_connector(
    connector_id: str = Path(..., description="Connector UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Get a single connector by ID (user-owned only)."""
    from services.database_manager.database_helpers import fetch_one
    row = await fetch_one(
        "SELECT * FROM data_source_connectors WHERE id = $1 AND user_id = $2",
        connector_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Connector not found")
    definition = row.get("definition") or {}
    if isinstance(definition, str):
        try:
            definition = json.loads(definition)
        except json.JSONDecodeError:
            definition = {}
    endpoints = definition.get("endpoints") or {}
    return JSONResponse(content={
        "id": str(row["id"]),
        "name": row.get("name", ""),
        "description": row.get("description"),
        "connector_type": row.get("connector_type", "rest"),
        "definition": definition,
        "endpoint_count": len(endpoints) if isinstance(endpoints, dict) else 0,
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
        "is_locked": row.get("is_locked", False),
    })


@router.get("/connectors/{connector_id}/usage")
async def get_connector_usage(
    connector_id: str = Path(..., description="Connector UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Return which agent profiles use this connector (for usage warning)."""
    from services.database_manager.database_helpers import fetch_all
    rows = await fetch_all(
        """
        SELECT ap.id, ap.name, ap.handle
        FROM agent_data_sources ads
        JOIN agent_profiles ap ON ap.id = ads.agent_profile_id
        WHERE ads.connector_id = $1 AND ap.user_id = $2
        ORDER BY ap.name
        """,
        connector_id,
        current_user.user_id,
    )
    result = [{"id": str(r["id"]), "name": r.get("name", ""), "handle": r.get("handle", "")} for r in rows]
    return JSONResponse(content=result)


@router.post("/connectors")
async def create_connector(
    body: ConnectorCreate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Create a connector from scratch (no template)."""
    from services.database_manager.database_helpers import fetch_one, execute
    await execute(
        """
        INSERT INTO data_source_connectors (
            user_id, name, description, connector_type, version, definition,
            is_template, requires_auth, auth_fields, icon, category, tags
        ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, false, $7, $8::jsonb, $9, $10, $11)
        """,
        current_user.user_id,
        body.name,
        body.description,
        body.connector_type,
        "1.0",
        json.dumps(body.definition),
        body.requires_auth,
        json.dumps(body.auth_fields),
        None,
        None,
        [],
    )
    row = await fetch_one(
        "SELECT * FROM data_source_connectors WHERE user_id = $1 AND name = $2 ORDER BY created_at DESC LIMIT 1",
        current_user.user_id,
        body.name,
    )
    if not row:
        raise HTTPException(status_code=500, detail="Failed to create connector")
    definition = row.get("definition") or {}
    if isinstance(definition, str):
        try:
            definition = json.loads(definition)
        except json.JSONDecodeError:
            definition = {}
    endpoints = definition.get("endpoints") or {}
    return JSONResponse(content={
        "id": str(row["id"]),
        "name": row.get("name", ""),
        "description": row.get("description"),
        "connector_type": row.get("connector_type", "rest"),
        "definition": definition,
        "endpoint_count": len(endpoints) if isinstance(endpoints, dict) else 0,
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
    }, status_code=201)


@router.post("/connectors/from-template")
async def create_connector_from_template(
    body: ConnectorFromTemplateCreate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Create a connector from a template (standalone, not attached to any agent)."""
    from services.database_manager.database_helpers import fetch_one, execute
    from services.connector_templates import CONNECTOR_TEMPLATES
    template = next((t for t in CONNECTOR_TEMPLATES if t.get("name") == body.template_name), None)
    if not template:
        raise HTTPException(status_code=400, detail=f"Unknown template: {body.template_name}")
    definition = template.get("definition") or {}
    await execute(
        """
        INSERT INTO data_source_connectors (
            user_id, name, description, connector_type, version, definition,
            is_template, requires_auth, auth_fields, icon, category, tags
        ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, false, $7, $8::jsonb, $9, $10, $11)
        """,
        current_user.user_id,
        template.get("name"),
        template.get("description"),
        template.get("connector_type", "rest"),
        "1.0",
        json.dumps(definition),
        template.get("requires_auth", False),
        json.dumps(template.get("auth_fields") or []),
        template.get("icon"),
        template.get("category"),
        [],
    )
    row = await fetch_one(
        "SELECT * FROM data_source_connectors WHERE user_id = $1 AND name = $2 ORDER BY created_at DESC LIMIT 1",
        current_user.user_id,
        template.get("name"),
    )
    if not row:
        raise HTTPException(status_code=500, detail="Failed to create connector")
    definition = row.get("definition") or {}
    if isinstance(definition, str):
        try:
            definition = json.loads(definition)
        except json.JSONDecodeError:
            definition = {}
    endpoints = definition.get("endpoints") or {}
    return JSONResponse(content={
        "id": str(row["id"]),
        "name": row.get("name", ""),
        "description": row.get("description"),
        "connector_type": row.get("connector_type", "rest"),
        "definition": definition,
        "endpoint_count": len(endpoints) if isinstance(endpoints, dict) else 0,
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
    }, status_code=201)


@router.put("/connectors/{connector_id}")
async def update_connector(
    connector_id: str = Path(..., description="Connector UUID"),
    body: ConnectorUpdate = ...,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Update a connector (owner only). Blocked when connector is locked except for lock toggle."""
    from services.database_manager.database_helpers import fetch_one, execute
    row = await fetch_one(
        "SELECT id, is_locked FROM data_source_connectors WHERE id = $1 AND user_id = $2",
        connector_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Connector not found")
    updates = body.model_dump(exclude_unset=True)
    if not updates:
        row = await fetch_one("SELECT * FROM data_source_connectors WHERE id = $1", connector_id)
        definition = row.get("definition") or {}
        if isinstance(definition, str):
            try:
                definition = json.loads(definition)
            except json.JSONDecodeError:
                definition = {}
        endpoints = definition.get("endpoints") or {}
        return JSONResponse(content={
            "id": str(row["id"]),
            "name": row.get("name", ""),
            "description": row.get("description"),
            "connector_type": row.get("connector_type", "rest"),
            "definition": definition,
            "endpoint_count": len(endpoints) if isinstance(endpoints, dict) else 0,
            "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
            "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
            "is_locked": row.get("is_locked", False),
        })
    if row.get("is_locked") and set(updates.keys()) != {"is_locked"}:
        raise HTTPException(status_code=403, detail="Connector is locked; only lock toggle is allowed")
    set_clauses = []
    args = []
    idx = 1
    jsonb_fields = ("definition", "auth_fields")
    for k, v in updates.items():
        if k in jsonb_fields:
            set_clauses.append(f"{k} = ${idx}::jsonb")
            args.append(json.dumps(v) if v is not None else "{}")
        else:
            set_clauses.append(f"{k} = ${idx}")
            args.append(v)
        idx += 1
    set_clauses.append("updated_at = NOW()")
    args.extend([connector_id, current_user.user_id])
    await execute(
        f"UPDATE data_source_connectors SET {', '.join(set_clauses)} WHERE id = ${idx} AND user_id = ${idx + 1}",
        *args,
    )
    row = await fetch_one("SELECT * FROM data_source_connectors WHERE id = $1", connector_id)
    definition = row.get("definition") or {}
    if isinstance(definition, str):
        try:
            definition = json.loads(definition)
        except json.JSONDecodeError:
            definition = {}
    endpoints = definition.get("endpoints") or {}
    return JSONResponse(content={
        "id": str(row["id"]),
        "name": row.get("name", ""),
        "description": row.get("description"),
        "connector_type": row.get("connector_type", "rest"),
        "definition": definition,
        "endpoint_count": len(endpoints) if isinstance(endpoints, dict) else 0,
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
        "is_locked": row.get("is_locked", False),
    })


@router.delete("/connectors/{connector_id}")
async def delete_connector(
    connector_id: str = Path(..., description="Connector UUID"),
    force: bool = False,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Delete a connector. Returns 409 if in use unless force=true. Blocked when connector is locked."""
    from services.database_manager.database_helpers import fetch_one, fetch_all, execute
    row = await fetch_one(
        "SELECT id, is_locked FROM data_source_connectors WHERE id = $1 AND user_id = $2",
        connector_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Connector not found")
    if row.get("is_locked"):
        raise HTTPException(status_code=403, detail="Connector is locked; unlock to delete")
    usage = await fetch_all(
        """
        SELECT ap.id, ap.name, ap.handle
        FROM agent_data_sources ads
        JOIN agent_profiles ap ON ap.id = ads.agent_profile_id
        WHERE ads.connector_id = $1 AND ap.user_id = $2
        """,
        connector_id,
        current_user.user_id,
    )
    if usage and not force:
        result = [{"id": str(r["id"]), "name": r.get("name", ""), "handle": r.get("handle", "")} for r in usage]
        return JSONResponse(
            status_code=409,
            content={"message": "Connector is in use", "agents": result},
        )
    await execute("DELETE FROM agent_data_sources WHERE connector_id = $1", connector_id)
    await execute("DELETE FROM data_source_connectors WHERE id = $1 AND user_id = $2", connector_id, current_user.user_id)
    return JSONResponse(content={"deleted": True, "id": connector_id})


@router.post("/connectors/{connector_id}/test")
async def test_connector(
    connector_id: str = Path(..., description="Connector UUID"),
    body: ConnectorTestRequest = ...,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Execute an endpoint without an agent profile. For building/testing connectors. Returns raw_response when available."""
    from services.database_manager.database_helpers import fetch_one
    from clients.connections_service_client import get_connections_service_client

    row = await fetch_one(
        "SELECT id, definition, connector_type FROM data_source_connectors WHERE id = $1 AND user_id = $2",
        connector_id,
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


@router.post("/execute-connector")
async def execute_connector(
    body: ExecuteConnectorRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """
    Execute a connector endpoint. Loads definition and credentials from DB,
    delegates execution to connections-service, returns records.
    """
    from services.database_manager.database_helpers import fetch_one
    from clients.connections_service_client import get_connections_service_client

    profile = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        body.profile_id,
        current_user.user_id,
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    connector = await fetch_one(
        "SELECT id, definition, connector_type FROM data_source_connectors WHERE id = $1",
        body.connector_id,
    )
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")

    source = await fetch_one(
        "SELECT credentials_encrypted, config_overrides FROM agent_data_sources "
        "WHERE agent_profile_id = $1 AND connector_id = $2 AND is_enabled = true",
        body.profile_id,
        body.connector_id,
    )
    credentials = {}
    if source:
        creds = source.get("credentials_encrypted")
        if isinstance(creds, dict):
            credentials = creds
        overrides = source.get("config_overrides") or {}
        if isinstance(overrides, dict) and overrides.get("api_key"):
            credentials.setdefault("api_key", overrides["api_key"])

    definition = connector.get("definition") or {}
    if isinstance(definition, str):
        try:
            definition = json.loads(definition)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid connector definition")

    client = await get_connections_service_client()
    result = await client.execute_connector_endpoint(
        definition=definition,
        credentials=credentials,
        endpoint_id=body.endpoint_id,
        params=body.params,
        connector_type=connector.get("connector_type"),
    )
    return JSONResponse(content=result)


# ---------- Agent Profiles CRUD ----------

def _row_to_profile(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert DB row to API-friendly dict (UUID and JSONB as native types)."""
    if not row:
        return {}
    ownership = row.get("ownership") or "owned"
    if row.get("is_builtin"):
        ownership = "builtin"
    return {
        "id": str(row["id"]),
        "user_id": row["user_id"],
        "name": row["name"],
        "handle": row["handle"],
        "description": row.get("description"),
        "is_active": row.get("is_active", True),
        "model_preference": row.get("model_preference"),
        "model_source": row.get("model_source"),
        "model_provider_type": row.get("model_provider_type"),
        "max_research_rounds": row.get("max_research_rounds", 3),
        "system_prompt_additions": row.get("system_prompt_additions"),
        "knowledge_config": _ensure_json_obj(row.get("knowledge_config"), {}),
        "default_playbook_id": str(row["default_playbook_id"]) if row.get("default_playbook_id") else None,
        "journal_config": _ensure_json_obj(row.get("journal_config"), {}),
        "team_config": _ensure_json_obj(row.get("team_config"), {}),
        "watch_config": _ensure_json_obj(row.get("watch_config"), {}),
        "prompt_history_enabled": row.get("chat_history_enabled", False),
        "chat_history_lookback": row.get("chat_history_lookback", 10),
        "summary_threshold_tokens": row.get("summary_threshold_tokens", 5000),
        "summary_keep_messages": row.get("summary_keep_messages", 10),
        "persona_mode": row.get("persona_mode") or "none",
        "persona_id": str(row["persona_id"]) if row.get("persona_id") else None,
        "include_user_context": row.get("include_user_context", False),
        "include_datetime_context": row.get("include_datetime_context", True),
        "include_user_facts": row.get("include_user_facts", False),
        "include_facts_categories": row.get("include_facts_categories") or [],
        "use_themed_memory": row.get("use_themed_memory", True),
        "include_agent_memory": row.get("include_agent_memory", False),
        "auto_routable": row.get("auto_routable", False),
        "chat_visible": row.get("chat_visible", True),
        "category": row.get("category"),
        "data_workspace_config": _ensure_json_obj(row.get("data_workspace_config"), {}),
        "allowed_connections": _ensure_json_obj(row.get("allowed_connections"), []),
        "default_run_context": row.get("default_run_context") or "interactive",
        "default_approval_policy": row.get("default_approval_policy") or "require",
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
        "is_locked": row.get("is_locked", False),
        "is_builtin": row.get("is_builtin", False),
        "ownership": ownership,
        "owner_username": row.get("owner_username"),
        "owner_display_name": row.get("owner_display_name"),
    }


def _derive_profile_status(is_active: bool, last_execution_status: Optional[str]) -> str:
    """Derive UI status from is_active and last execution. Returns active, paused, draft, or error."""
    if not is_active:
        return "draft" if not last_execution_status else "paused"
    return "error" if last_execution_status == "failed" else "active"


@router.get("/handles")
async def list_handles(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> JSONResponse:
    """Lightweight list of agent and team handles for @mention autocomplete (handle, name, type).

    Agents include owned profiles and profiles shared with the user (non-transitive shares only).
    Each agent row includes id, ownership, and owner display fields for UI disambiguation.
    """
    from services.database_manager.database_helpers import fetch_all

    owned_rows = await fetch_all(
        """
        SELECT p.id, p.handle, p.name, p.user_id::text AS owner_user_id,
               u.username AS owner_username,
               COALESCE(NULLIF(TRIM(u.display_name), ''), u.username) AS owner_display_name,
               'owned'::text AS ownership
        FROM agent_profiles p
        JOIN users u ON u.user_id = p.user_id
        WHERE p.user_id = $1
          AND p.is_active = true
          AND p.handle IS NOT NULL AND TRIM(p.handle) <> ''
          AND COALESCE(p.chat_visible, true) = true
        """,
        current_user.user_id,
    )
    shared_rows = await fetch_all(
        """
        SELECT p.id, p.handle, p.name, p.user_id::text AS owner_user_id,
               u.username AS owner_username,
               COALESCE(NULLIF(TRIM(u.display_name), ''), u.username) AS owner_display_name,
               'shared'::text AS ownership
        FROM agent_profiles p
        JOIN agent_artifact_shares s
          ON s.artifact_type = 'agent_profile'
         AND s.artifact_id = p.id
         AND s.shared_with_user_id = $1
         AND COALESCE(s.is_transitive, false) = false
        JOIN users u ON u.user_id = p.user_id
        WHERE p.user_id <> $1
          AND p.is_active = true
          AND p.handle IS NOT NULL AND TRIM(p.handle) <> ''
          AND COALESCE(p.chat_visible, true) = true
        """,
        current_user.user_id,
    )
    agent_rows = list(owned_rows or []) + list(shared_rows or [])
    agent_rows.sort(
        key=lambda r: (
            (r.get("handle") or "").lower(),
            0 if (r.get("ownership") or "") == "owned" else 1,
            (r.get("owner_display_name") or r.get("owner_username") or "").lower(),
        )
    )
    out = [
        {
            "handle": r["handle"],
            "name": r.get("name"),
            "type": "agent",
            "id": str(r["id"]),
            "ownership": r.get("ownership") or "owned",
            "owner_user_id": r.get("owner_user_id"),
            "owner_username": r.get("owner_username"),
            "owner_display_name": r.get("owner_display_name"),
        }
        for r in agent_rows
    ]
    line_rows = await fetch_all(
        "SELECT id, handle, name FROM agent_lines WHERE user_id = $1 AND handle IS NOT NULL AND handle != '' ORDER BY name",
        current_user.user_id,
    )
    out.extend([{"handle": r["handle"], "name": r["name"], "type": "line", "id": str(r["id"])} for r in line_rows])
    return JSONResponse(content=out)


@router.get("/profiles")
async def list_profiles(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> JSONResponse:
    """List agent profiles for the current user. Includes derived status (active, paused, draft, error)."""
    from services.model_source_resolver import try_soft_retarget
    out = await agent_factory_service.list_profiles(current_user.user_id)
    for p in out:
        p["status"] = _derive_profile_status(
            p.get("is_active", True),
            p.get("last_execution_status"),
        )
        if p.get("model_preference"):
            try:
                retarget = await try_soft_retarget(current_user.user_id, p["model_preference"])
                p["model_source_meta"] = {
                    "source": p.get("model_source"),
                    "provider_type": p.get("model_provider_type"),
                    "retargeted": retarget.get("retargeted", False),
                    "available": retarget.get("available", True),
                }
            except Exception:
                p["model_source_meta"] = {"source": p.get("model_source"), "provider_type": p.get("model_provider_type"), "retargeted": False, "available": False}
    return JSONResponse(content=out)


@router.get("/sidebar-categories")
async def list_sidebar_categories(
    section: Optional[str] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """List sidebar categories for the current user, optionally filtered by section."""
    out = await agent_factory_service.list_sidebar_categories(
        current_user.user_id, section=section
    )
    return JSONResponse(content=out)


@router.post("/sidebar-categories")
async def create_sidebar_category(
    body: SidebarCategoryCreate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Create a sidebar category (folder) for a section."""
    try:
        out = await agent_factory_service.create_sidebar_category(
            current_user.user_id,
            body.section,
            body.name,
        )
        return JSONResponse(content=out, status_code=201)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/sidebar-categories/{category_id}")
async def update_sidebar_category(
    category_id: str = Path(..., description="Sidebar category UUID"),
    body: SidebarCategoryUpdate = ...,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Update a sidebar category (name and/or sort_order)."""
    try:
        data = body.model_dump(exclude_unset=True)
        out = await agent_factory_service.update_sidebar_category(
            current_user.user_id, category_id, data
        )
        return JSONResponse(content=out)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/sidebar-categories/{category_id}")
async def delete_sidebar_category(
    category_id: str = Path(..., description="Sidebar category UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Response:
    """Delete a sidebar category. Items keep their category until moved."""
    try:
        await agent_factory_service.delete_sidebar_category(
            current_user.user_id, category_id
        )
        return Response(status_code=204)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/profiles")
async def create_profile(
    body: AgentProfileCreate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Create an agent profile."""
    try:
        out = await agent_factory_service.create_profile(
            current_user.user_id,
            body.model_dump(),
        )
        pid = out.get("id") if isinstance(out, dict) else None
        if pid:
            await _notify_agent_handles_for_profile(str(pid))
        return JSONResponse(content=out, status_code=201)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/profiles/{profile_id}")
async def get_profile(
    profile_id: str = Path(..., description="Agent profile UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Get a single agent profile by ID."""
    from services.database_manager.database_helpers import fetch_one
    from services.model_source_resolver import try_soft_retarget
    row = await fetch_one(
        """SELECT p.*,
                  CASE WHEN p.user_id = $2 THEN 'owned' ELSE 'shared' END AS ownership,
                  u_owner.username AS owner_username,
                  u_owner.display_name AS owner_display_name
           FROM agent_profiles p
           LEFT JOIN users u_owner ON u_owner.user_id = p.user_id
           WHERE p.id = $1
             AND (p.user_id = $2
                  OR EXISTS (
                      SELECT 1 FROM agent_artifact_shares _sh
                      WHERE _sh.artifact_type = 'agent_profile'
                        AND _sh.artifact_id = p.id
                        AND _sh.shared_with_user_id = $2
                  ))""",
        profile_id,
        current_user.user_id,
        rls_context={"user_id": current_user.user_id, "user_role": "user"},
    )
    if not row:
        raise HTTPException(status_code=404, detail="Profile not found")
    content = _row_to_profile(row)
    content["budget"] = await agent_factory_service.get_profile_budget(profile_id, current_user.user_id)
    if content.get("model_preference"):
        try:
            retarget = await try_soft_retarget(current_user.user_id, content["model_preference"])
            content["model_source_meta"] = {
                "source": content.get("model_source"),
                "provider_type": content.get("model_provider_type"),
                "retargeted": retarget.get("retargeted", False),
                "available": retarget.get("available", True),
            }
        except Exception:
            content["model_source_meta"] = {"source": content.get("model_source"), "provider_type": content.get("model_provider_type"), "retargeted": False, "available": False}
    return JSONResponse(content=content)


@router.put("/profiles/{profile_id}")
async def update_profile(
    profile_id: str = Path(..., description="Agent profile UUID"),
    body: AgentProfileUpdate = ...,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Update an agent profile."""
    updates = body.model_dump(exclude_unset=True)
    if not updates:
        from services.database_manager.database_helpers import fetch_one
        row = await fetch_one(
            "SELECT * FROM agent_profiles WHERE id = $1 AND user_id = $2",
            profile_id,
            current_user.user_id,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Profile not found")
        return JSONResponse(content=_row_to_profile(row))
    try:
        out = await agent_factory_service.update_profile(
            current_user.user_id,
            profile_id,
            updates,
        )
        await _notify_agent_handles_for_profile(profile_id)
        return JSONResponse(content=out)
    except ValueError as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)


@router.post("/profiles/{profile_id}/pause")
async def pause_profile(
    profile_id: str = Path(..., description="Agent profile UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Pause an agent profile (set is_active = false). Stops all scheduled runs until resumed."""
    from services.database_manager.database_helpers import execute, fetch_one
    row = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Profile not found")
    await execute(
        "UPDATE agent_profiles SET is_active = false, updated_at = NOW() WHERE id = $1 AND user_id = $2",
        profile_id,
        current_user.user_id,
    )
    row = await fetch_one("SELECT * FROM agent_profiles WHERE id = $1", profile_id)
    await _notify_agent_handles_for_profile(profile_id)
    return JSONResponse(content=_row_to_profile(row))


@router.post("/profiles/{profile_id}/resume")
async def resume_profile(
    profile_id: str = Path(..., description="Agent profile UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Resume an agent profile (set is_active = true). Scheduled runs will run again."""
    from services.database_manager.database_helpers import execute, fetch_one
    row = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Profile not found")
    await execute(
        "UPDATE agent_profiles SET is_active = true, updated_at = NOW() WHERE id = $1 AND user_id = $2",
        profile_id,
        current_user.user_id,
    )
    row = await fetch_one("SELECT * FROM agent_profiles WHERE id = $1", profile_id)
    await _notify_agent_handles_for_profile(profile_id)
    return JSONResponse(content=_row_to_profile(row))


class AgentBudgetUpdate(BaseModel):
    """Update agent monthly budget (null limit = unlimited)."""
    monthly_limit_usd: Optional[float] = Field(None, ge=0, description="Monthly spend limit in USD; null = unlimited")
    warning_threshold_pct: int = Field(default=80, ge=1, le=100)
    enforce_hard_limit: bool = True


@router.get("/profiles/{profile_id}/budget")
async def get_profile_budget(
    profile_id: str = Path(..., description="Agent profile UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Get budget for an agent profile."""
    budget = await agent_factory_service.get_profile_budget(profile_id, current_user.user_id)
    return JSONResponse(content=budget if budget is not None else {})


@router.put("/profiles/{profile_id}/budget")
async def update_profile_budget(
    profile_id: str = Path(..., description="Agent profile UUID"),
    body: AgentBudgetUpdate = ...,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Set or update monthly budget for an agent. Omit monthly_limit_usd or set null for unlimited."""
    try:
        out = await agent_factory_service.set_profile_budget(
            profile_id,
            current_user.user_id,
            monthly_limit_usd=body.monthly_limit_usd,
            warning_threshold_pct=body.warning_threshold_pct,
            enforce_hard_limit=body.enforce_hard_limit,
        )
        return JSONResponse(content=out)
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/profiles/{profile_id}/memory")
async def get_profile_memory(
    profile_id: str = Path(..., description="Agent profile UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """List agent memory key/value for the profile (read-only)."""
    from services.database_manager.database_helpers import fetch_all, fetch_one
    profile_row = await fetch_one("SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2", uuid.UUID(profile_id), current_user.user_id)
    if not profile_row:
        raise HTTPException(status_code=404, detail="Profile not found")
    rows = await fetch_all(
        """SELECT memory_key, memory_value, updated_at FROM agent_memory
           WHERE agent_profile_id = $1 AND user_id = $2 AND (expires_at IS NULL OR expires_at > NOW())
           ORDER BY memory_key""",
        uuid.UUID(profile_id),
        current_user.user_id,
    )
    out = [
        {
            "key": r["memory_key"],
            "value": r["memory_value"],
            "updated_at": r["updated_at"].isoformat() if r.get("updated_at") else None,
        }
        for r in rows
    ]
    return JSONResponse(content=out)


@router.delete("/profiles/{profile_id}/memory")
async def clear_profile_memory(
    profile_id: str = Path(..., description="Agent profile UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Clear all agent memory for the profile."""
    from services.database_manager.database_helpers import execute, fetch_one
    row = await fetch_one("SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2", uuid.UUID(profile_id), current_user.user_id)
    if not row:
        raise HTTPException(status_code=404, detail="Profile not found")
    await execute(
        "DELETE FROM agent_memory WHERE agent_profile_id = $1 AND user_id = $2",
        uuid.UUID(profile_id),
        current_user.user_id,
    )
    return JSONResponse(content={"cleared": True})


@router.delete("/profiles/{profile_id}")
async def delete_profile(
    profile_id: str = Path(..., description="Agent profile UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Delete an agent profile (cascades to data sources and skills). Blocked when profile is locked."""
    from services.database_manager.database_helpers import execute, fetch_one
    row = await fetch_one(
        "SELECT id, is_locked, is_builtin FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Profile not found")
    if row.get("is_builtin"):
        raise HTTPException(status_code=403, detail="Built-in profile cannot be deleted")
    if row.get("is_locked"):
        raise HTTPException(status_code=403, detail="Profile is locked; unlock to delete")
    from services.settings_service import settings_service

    await settings_service.clear_default_chat_agent_profile_if_matches(
        current_user.user_id, str(profile_id)
    )
    await _notify_agent_handles_for_profile(profile_id)
    await execute("DELETE FROM agent_profiles WHERE id = $1 AND user_id = $2", profile_id, current_user.user_id)
    return JSONResponse(content={"deleted": True, "id": profile_id})


@router.post("/profiles/{profile_id}/reset-defaults")
async def reset_profile_defaults(
    profile_id: str = Path(..., description="Agent profile UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Reset built-in agent profile to seed defaults (name, playbook, persona, context flags). Only allowed for is_builtin profiles."""
    try:
        row = await agent_factory_service.reset_builtin_profile_defaults(
            current_user.user_id,
            profile_id,
        )
        await _notify_agent_handles_for_profile(profile_id)
        return JSONResponse(content=row)
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=403, detail=str(e))


@router.get("/profiles/{profile_id}/export")
async def export_profile(
    profile_id: str = Path(..., description="Agent profile UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Export a profile as JSON (for backup/import elsewhere)."""
    from services.database_manager.database_helpers import fetch_one
    row = await fetch_one(
        "SELECT * FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Profile not found")
    content = _row_to_profile(row)
    content["budget"] = await agent_factory_service.get_profile_budget(profile_id, current_user.user_id)
    return JSONResponse(content=content)


def _build_agent_bundle_dict(
    profile_row: Dict[str, Any],
    playbook_row: Optional[Dict[str, Any]],
    data_source_rows: List[Dict[str, Any]],
    persona_row: Optional[Dict[str, Any]] = None,
    skills: Optional[List[Dict[str, Any]]] = None,
    budget: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the agent bundle dict for YAML export (no ids, no credentials; full profile fields for round-trip)."""
    default_playbook_id = profile_row.get("default_playbook_id")
    default_playbook_name = None
    if playbook_row and default_playbook_id and str(playbook_row.get("id")) == str(default_playbook_id):
        default_playbook_name = playbook_row.get("name") or "imported-playbook"

    agent: Dict[str, Any] = {
        "name": profile_row.get("name"),
        "handle": profile_row.get("handle"),
        "description": profile_row.get("description"),
        "is_active": profile_row.get("is_active", True),
        "model_preference": profile_row.get("model_preference"),
        "model_source": profile_row.get("model_source"),
        "model_provider_type": profile_row.get("model_provider_type"),
        "max_research_rounds": profile_row.get("max_research_rounds", 3),
        "system_prompt_additions": profile_row.get("system_prompt_additions"),
        "knowledge_config": _ensure_json_obj(profile_row.get("knowledge_config"), {}),
        "default_playbook": default_playbook_name,
        "default_run_context": profile_row.get("default_run_context") or "interactive",
        "default_approval_policy": profile_row.get("default_approval_policy") or "require",
        "journal_config": _ensure_json_obj(profile_row.get("journal_config"), {}),
        "team_config": _ensure_json_obj(profile_row.get("team_config"), {}),
        "watch_config": _ensure_json_obj(profile_row.get("watch_config"), {}),
        "prompt_history_enabled": profile_row.get("chat_history_enabled", False),
        "chat_history_lookback": profile_row.get("chat_history_lookback", 10),
        "summary_threshold_tokens": profile_row.get("summary_threshold_tokens", 5000),
        "summary_keep_messages": profile_row.get("summary_keep_messages", 10),
        "persona_mode": profile_row.get("persona_mode") or "none",
        "include_user_context": profile_row.get("include_user_context", False),
        "include_datetime_context": profile_row.get("include_datetime_context", True),
        "include_user_facts": profile_row.get("include_user_facts", False),
        "include_facts_categories": _ensure_json_obj(profile_row.get("include_facts_categories"), []),
        "use_themed_memory": profile_row.get("use_themed_memory", True),
        "include_agent_memory": profile_row.get("include_agent_memory", False),
        "auto_routable": profile_row.get("auto_routable", False),
        "chat_visible": profile_row.get("chat_visible", True),
        "category": profile_row.get("category"),
        "data_workspace_config": _ensure_json_obj(profile_row.get("data_workspace_config"), {}),
        "allowed_connections": _ensure_json_obj(profile_row.get("allowed_connections"), []),
    }
    if persona_row:
        agent["persona"] = {
            "name": persona_row.get("name"),
            "is_builtin": persona_row.get("is_builtin", False),
        }
    if budget:
        agent["budget"] = dict(budget)


    playbook: Optional[Dict[str, Any]] = None
    if playbook_row:
        playbook = {
            "name": playbook_row.get("name"),
            "description": playbook_row.get("description"),
            "version": playbook_row.get("version") or "1.0",
            "definition": _ensure_json_obj(playbook_row.get("definition"), {}),
            "triggers": _ensure_json_obj(playbook_row.get("triggers"), []),
            "category": playbook_row.get("category"),
            "tags": list(playbook_row["tags"]) if playbook_row.get("tags") else [],
            "required_connectors": list(playbook_row["required_connectors"]) if playbook_row.get("required_connectors") else [],
        }

    data_sources: List[Dict[str, Any]] = []
    for r in data_source_rows:
        definition = r.get("connector_definition")
        if isinstance(definition, str):
            try:
                definition = json.loads(definition)
            except json.JSONDecodeError:
                definition = {}
        elif definition is None:
            definition = {}
        data_sources.append({
            "name": r.get("connector_name") or r.get("name"),
            "connector_type": r.get("connector_type", "rest"),
            "description": r.get("description"),
            "version": r.get("version") or "1.0",
            "definition": definition,
            "icon": r.get("icon"),
            "category": r.get("category"),
            "tags": list(r["tags"]) if r.get("tags") else [],
            "binding": {
                "config_overrides": _ensure_json_obj(r.get("config_overrides"), {}),
                "permissions": _ensure_json_obj(r.get("permissions"), {}),
                "is_enabled": r.get("is_enabled", True),
            },
        })

    out: Dict[str, Any] = {
        "bastion_agent_bundle": {
            "version": "2",
            "exported_at": datetime.now(timezone.utc).isoformat(),
        },
        "agent": agent,
        "playbook": playbook,
        "data_sources": data_sources,
    }
    if skills is not None:
        out["skills"] = skills
    return out


@router.get("/profiles/{profile_id}/export-bundle")
async def export_profile_bundle(
    profile_id: str = Path(..., description="Agent profile UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Response:
    """Export agent profile with playbook, data sources, and referenced skills as YAML bundle."""
    from services.database_manager.database_helpers import fetch_one, fetch_all
    profile_row = await fetch_one(
        "SELECT * FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        current_user.user_id,
    )
    if not profile_row:
        raise HTTPException(status_code=404, detail="Profile not found")

    persona_row = None
    persona_id = profile_row.get("persona_id")
    if persona_id:
        persona_row = await fetch_one(
            "SELECT id, name, is_builtin FROM personas WHERE id = $1",
            persona_id,
        )
        if persona_row:
            persona_row = dict(persona_row)

    default_playbook_id = profile_row.get("default_playbook_id")
    playbook_row = None
    if default_playbook_id:
        playbook_row = await fetch_one(
            "SELECT * FROM custom_playbooks WHERE id = $1 AND (user_id = $2 OR is_template = true)",
            default_playbook_id,
            current_user.user_id,
        )
        if playbook_row:
            playbook_row = dict(playbook_row)

    rows = await fetch_all(
        """
        SELECT dsc.id, dsc.name AS connector_name, dsc.description, dsc.connector_type,
               dsc.version, dsc.definition AS connector_definition, dsc.icon, dsc.category, dsc.tags,
               ads.config_overrides, ads.permissions, ads.is_enabled
        FROM agent_data_sources ads
        JOIN data_source_connectors dsc ON dsc.id = ads.connector_id
        WHERE ads.agent_profile_id = $1
        ORDER BY ads.created_at
        """,
        profile_id,
    )
    data_source_rows = [dict(r) for r in rows]

    skills_export: List[Dict[str, Any]] = []
    if playbook_row:
        definition = playbook_row.get("definition") or {}
        if isinstance(definition, str):
            try:
                definition = json.loads(definition)
            except json.JSONDecodeError:
                definition = {}
        steps = definition.get("steps") or []
        skill_ids_in_playbook: List[str] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            for sid in (step.get("skill_ids") or step.get("skills") or []):
                if sid and isinstance(sid, str) and sid not in skill_ids_in_playbook:
                    skill_ids_in_playbook.append(sid)
        if skill_ids_in_playbook:
            skills_list = await agent_skills_service.get_skills_by_ids(skill_ids_in_playbook)
            for s in skills_list:
                if s.get("is_builtin"):
                    skills_export.append({
                        "id": s.get("id"),
                        "slug": s.get("slug"),
                        "is_builtin": True,
                    })
                else:
                    skills_export.append({
                        "id": s.get("id"),
                        "name": s.get("name"),
                        "slug": s.get("slug"),
                        "description": s.get("description"),
                        "category": s.get("category"),
                        "procedure": s.get("procedure") or "",
                        "required_tools": s.get("required_tools") or [],
                        "required_connection_types": s.get("required_connection_types") or [],
                        "optional_tools": s.get("optional_tools") or [],
                        "inputs_schema": _ensure_json_obj(s.get("inputs_schema"), {}),
                        "outputs_schema": _ensure_json_obj(s.get("outputs_schema"), {}),
                        "examples": _ensure_json_obj(s.get("examples"), []),
                        "tags": list(s.get("tags") or []),
                    })

    budget_export: Optional[Dict[str, Any]] = None
    try:
        bud = await agent_factory_service.get_profile_budget(str(profile_row["id"]), current_user.user_id)
        if bud and bud.get("monthly_limit_usd") is not None:
            budget_export = {
                "monthly_limit_usd": float(bud["monthly_limit_usd"]),
                "warning_threshold_pct": int(bud.get("warning_threshold_pct") or 80),
                "enforce_hard_limit": bool(bud.get("enforce_hard_limit", True)),
            }
    except Exception as e:
        logger.debug("Agent bundle export: budget omitted: %s", e)

    bundle = _build_agent_bundle_dict(
        profile_row, playbook_row, data_source_rows,
        persona_row=persona_row,
        skills=skills_export if skills_export else None,
        budget=budget_export,
    )
    yaml_str = yaml.dump(bundle, default_flow_style=False, allow_unicode=True, sort_keys=False)
    handle = (profile_row.get("handle") or profile_row.get("name") or "agent")[:80]
    safe_handle = "".join(c if c.isalnum() or c in "-_" else "_" for c in handle) or "agent"
    filename = f"{safe_handle}.yaml"
    return Response(
        content=yaml_str,
        media_type="application/x-yaml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/profiles/import-bundle")
async def import_profile_bundle(
    body: Dict[str, Any],
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Import an agent bundle from YAML (creates connectors, playbook, profile, and bindings)."""
    from services.database_manager.database_helpers import fetch_one, execute, fetch_value
    yaml_str = body.get("yaml")
    if not yaml_str or not isinstance(yaml_str, str):
        raise HTTPException(status_code=400, detail="Request body must include 'yaml' string")
    try:
        bundle = yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")
    if not isinstance(bundle, dict):
        raise HTTPException(status_code=400, detail="YAML must be a mapping (agent bundle)")
    agent_data = bundle.get("agent")
    if not agent_data or not isinstance(agent_data, dict):
        raise HTTPException(status_code=400, detail="Bundle must include 'agent'")
    playbook_data = bundle.get("playbook")
    data_sources_data = bundle.get("data_sources")
    if not isinstance(data_sources_data, list):
        data_sources_data = []

    connector_ids: List[str] = []
    for ds in data_sources_data:
        if not isinstance(ds, dict):
            continue
        name = (ds.get("name") or "Imported connector")[:255]
        base_name = name
        suffix = 0
        while True:
            candidate = f"{base_name}_{suffix}" if suffix else base_name
            existing = await fetch_one(
                "SELECT id FROM data_source_connectors WHERE user_id = $1 AND name = $2",
                current_user.user_id,
                candidate,
            )
            if not existing:
                name = candidate
                break
            suffix += 1
        definition = ds.get("definition") or {}
        if isinstance(definition, str):
            try:
                definition = json.loads(definition)
            except json.JSONDecodeError:
                definition = {}
        tags = ds.get("tags")
        if not isinstance(tags, list):
            tags = []
        await execute(
            """
            INSERT INTO data_source_connectors (
                user_id, name, description, connector_type, version, definition,
                is_template, requires_auth, auth_fields, icon, category, tags
            ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, false, $7, $8::jsonb, $9, $10, $11)
            """,
            current_user.user_id,
            name,
            ds.get("description"),
            ds.get("connector_type", "rest"),
            ds.get("version", "1.0"),
            json.dumps(definition),
            ds.get("requires_auth", False),
            json.dumps(ds.get("auth_fields") or []),
            ds.get("icon"),
            ds.get("category"),
            tags,
        )
        row = await fetch_one(
            "SELECT id FROM data_source_connectors WHERE user_id = $1 AND name = $2 ORDER BY created_at DESC LIMIT 1",
            current_user.user_id,
            name,
        )
        if row:
            connector_ids.append(str(row["id"]))

    skill_id_map: Dict[str, str] = {}
    skills_data = bundle.get("skills")
    if isinstance(skills_data, list):
        for item in skills_data:
            if not isinstance(item, dict):
                continue
            old_id = (item.get("id") or "").strip() if item.get("id") else None
            slug = (item.get("slug") or "").strip()
            if item.get("is_builtin") and slug:
                existing = await fetch_one(
                    "SELECT id FROM agent_skills WHERE slug = $1 AND is_builtin = true LIMIT 1",
                    slug,
                )
                if existing and old_id:
                    skill_id_map[old_id] = str(existing["id"])
            elif item.get("procedure") is not None and slug:
                try:
                    created = await agent_skills_service.create_skill(
                        current_user.user_id,
                        name=item.get("name") or slug,
                        slug=slug,
                        procedure=item.get("procedure", ""),
                        required_tools=item.get("required_tools"),
                        required_connection_types=item.get("required_connection_types"),
                        optional_tools=item.get("optional_tools"),
                        description=item.get("description"),
                        category=item.get("category"),
                        inputs_schema=item.get("inputs_schema"),
                        outputs_schema=item.get("outputs_schema"),
                        examples=item.get("examples"),
                        tags=item.get("tags"),
                    )
                    if old_id:
                        skill_id_map[old_id] = str(created.get("id", ""))
                except ValueError:
                    existing = await agent_skills_service.get_skill_by_slug(slug, current_user.user_id)
                    if existing and old_id:
                        skill_id_map[old_id] = str(existing.get("id", ""))

    playbook_id: Optional[str] = None
    if playbook_data and isinstance(playbook_data, dict):
        name = (playbook_data.get("name") or "Imported playbook")[:255]
        base_name = name
        suffix = 0
        while True:
            candidate = f"{base_name}_{suffix}" if suffix else base_name
            existing = await fetch_one(
                "SELECT id FROM custom_playbooks WHERE user_id = $1 AND name = $2",
                current_user.user_id,
                candidate,
            )
            if not existing:
                name = candidate
                break
            suffix += 1
        definition = playbook_data.get("definition") or {}
        if isinstance(definition, str):
            try:
                definition = json.loads(definition)
            except json.JSONDecodeError:
                definition = {}
        triggers = playbook_data.get("triggers") or []
        tags = playbook_data.get("tags")
        required_connectors = playbook_data.get("required_connectors")
        if not isinstance(tags, list):
            tags = []
        if not isinstance(required_connectors, list):
            required_connectors = []
        if skill_id_map and isinstance(definition, dict):
            definition = json.loads(json.dumps(definition))
            steps = definition.get("steps") or []
            for step in steps:
                if not isinstance(step, dict):
                    continue
                for key in ("skill_ids", "skills"):
                    if key not in step:
                        continue
                    orig = step[key]
                    if not isinstance(orig, list):
                        continue
                    new_ids = []
                    for sid in orig:
                        s = (sid or "").strip() if isinstance(sid, str) else None
                        if not s:
                            continue
                        mapped = skill_id_map.get(s)
                        if mapped and mapped not in new_ids:
                            new_ids.append(mapped)
                    step[key] = new_ids
        playbook_id = await fetch_value(
            """
            INSERT INTO custom_playbooks (
                user_id, name, description, version, definition, triggers,
                is_template, category, tags, required_connectors
            ) VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, false, $7, $8, $9)
            RETURNING id
            """,
            current_user.user_id,
            name,
            playbook_data.get("description"),
            playbook_data.get("version", "1.0"),
            json.dumps(definition),
            json.dumps(triggers),
            playbook_data.get("category"),
            tags,
            required_connectors,
        )
        playbook_id = str(playbook_id)

    handle = (agent_data.get("handle") or "imported").strip() or "imported"
    base_handle = handle[:90]
    suffix = 0
    while True:
        candidate = f"{base_handle}_{suffix}" if suffix else base_handle
        existing = await fetch_one(
            "SELECT id FROM agent_profiles WHERE user_id = $1 AND handle = $2",
            current_user.user_id,
            candidate,
        )
        if not existing:
            handle = candidate
            break
        suffix += 1
    name = (agent_data.get("name") or "Imported profile")[:255]
    persona_id_val: Optional[str] = None
    persona_block = agent_data.get("persona")
    if isinstance(persona_block, dict) and persona_block.get("name"):
        persona_name = (persona_block.get("name") or "").strip()[:255]
        if persona_name:
            persona_lookup = await fetch_one(
                """
                SELECT id FROM personas
                WHERE (user_id = $1 OR is_builtin = true) AND name = $2
                LIMIT 1
                """,
                current_user.user_id,
                persona_name,
            )
            if persona_lookup:
                persona_id_val = str(persona_lookup["id"])

    persona_mode = (agent_data.get("persona_mode") or "none").strip() or "none"
    if persona_id_val:
        persona_mode = "specific"
    elif persona_mode == "specific":
        persona_mode = "none"
        persona_id_val = None

    create_payload: Dict[str, Any] = {
        "name": name,
        "handle": handle,
        "description": agent_data.get("description"),
        "is_active": agent_data.get("is_active", True),
        "model_preference": agent_data.get("model_preference"),
        "model_source": agent_data.get("model_source"),
        "model_provider_type": agent_data.get("model_provider_type"),
        "max_research_rounds": agent_data.get("max_research_rounds", 3),
        "system_prompt_additions": agent_data.get("system_prompt_additions"),
        "knowledge_config": _ensure_json_obj(agent_data.get("knowledge_config"), {}),
        "default_playbook_id": playbook_id,
        "default_run_context": agent_data.get("default_run_context") or "interactive",
        "default_approval_policy": agent_data.get("default_approval_policy") or "require",
        "journal_config": _ensure_json_obj(agent_data.get("journal_config"), {}),
        "team_config": _ensure_json_obj(agent_data.get("team_config"), {}),
        "watch_config": _ensure_json_obj(agent_data.get("watch_config"), {}),
        "prompt_history_enabled": agent_data.get(
            "prompt_history_enabled", agent_data.get("chat_history_enabled", False)
        ),
        "chat_history_lookback": agent_data.get("chat_history_lookback", 10),
        "summary_threshold_tokens": agent_data.get("summary_threshold_tokens", 5000),
        "summary_keep_messages": agent_data.get("summary_keep_messages", 10),
        "persona_mode": persona_mode,
        "persona_id": persona_id_val,
        "include_user_context": agent_data.get("include_user_context", False),
        "include_datetime_context": agent_data.get("include_datetime_context", True),
        "include_user_facts": agent_data.get("include_user_facts", False),
        "include_facts_categories": _ensure_json_obj(agent_data.get("include_facts_categories"), []),
        "use_themed_memory": agent_data.get("use_themed_memory", True),
        "include_agent_memory": agent_data.get("include_agent_memory", False),
        "auto_routable": agent_data.get("auto_routable", False),
        "chat_visible": agent_data.get("chat_visible", True),
        "category": agent_data.get("category"),
        "data_workspace_config": _ensure_json_obj(agent_data.get("data_workspace_config"), {}),
        "allowed_connections": _ensure_json_obj(agent_data.get("allowed_connections"), []),
    }
    try:
        created = await agent_factory_service.create_profile(current_user.user_id, create_payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    profile_id = str(created["id"])

    bdata = agent_data.get("budget")
    if isinstance(bdata, dict) and bdata.get("monthly_limit_usd") is not None:
        try:
            await agent_factory_service.set_profile_budget(
                profile_id,
                current_user.user_id,
                monthly_limit_usd=float(bdata["monthly_limit_usd"]),
                warning_threshold_pct=int(bdata.get("warning_threshold_pct") or 80),
                enforce_hard_limit=bool(bdata.get("enforce_hard_limit", True)),
            )
        except (ValueError, TypeError) as e:
            logger.warning("Import bundle: skipped budget restore: %s", e)

    bindings = []
    for i, ds in enumerate(data_sources_data):
        if not isinstance(ds, dict) or i >= len(connector_ids):
            continue
        binding = ds.get("binding")
        if not isinstance(binding, dict):
            binding = {}
        await execute(
            """
            INSERT INTO agent_data_sources (
                agent_profile_id, connector_id, credentials_encrypted,
                config_overrides, permissions, is_enabled
            ) VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6)
            """,
            profile_id,
            connector_ids[i],
            None,
            json.dumps(binding.get("config_overrides") or {}),
            json.dumps(binding.get("permissions") or {}),
            binding.get("is_enabled", True),
        )

    row = await fetch_one("SELECT * FROM agent_profiles WHERE id = $1", profile_id)
    return JSONResponse(content=_row_to_profile(row), status_code=201)


@router.post("/profiles/import")
async def import_profile(
    body: Dict[str, Any],
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Import a profile from JSON (creates new profile for current user)."""
    from services.database_manager.database_helpers import fetch_one

    handle = (body.get("handle") or "imported").strip() or "imported"
    base_handle = handle[:90]
    suffix = 0
    while True:
        candidate = f"{base_handle}_{suffix}" if suffix else base_handle
        existing = await fetch_one(
            "SELECT id FROM agent_profiles WHERE user_id = $1 AND handle = $2",
            current_user.user_id,
            candidate,
        )
        if not existing:
            handle = candidate
            break
        suffix += 1

    skip_keys = {
        "id",
        "user_id",
        "created_at",
        "updated_at",
        "is_locked",
        "is_builtin",
        "last_execution_status",
        "status",
        "model_source_meta",
        "budget",
        "output_config",
    }
    payload: Dict[str, Any] = {}
    for k, v in body.items():
        if k in skip_keys:
            continue
        payload[k] = v
    payload["handle"] = handle
    oc = body.get("output_config")
    if isinstance(oc, dict) and payload.get("default_playbook_id") is None:
        payload["default_playbook_id"] = oc.get("default_playbook_id")
    if not (payload.get("name") or "").strip():
        payload["name"] = (body.get("name") or handle or "Imported profile")[:255]

    try:
        created = await agent_factory_service.create_profile(current_user.user_id, payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    profile_id = str(created["id"])
    b = body.get("budget")
    if isinstance(b, dict) and b.get("monthly_limit_usd") is not None:
        try:
            await agent_factory_service.set_profile_budget(
                profile_id,
                current_user.user_id,
                monthly_limit_usd=float(b["monthly_limit_usd"]),
                warning_threshold_pct=int(b.get("warning_threshold_pct") or 80),
                enforce_hard_limit=bool(b.get("enforce_hard_limit", True)),
            )
        except (ValueError, TypeError):
            pass

    row = await fetch_one("SELECT * FROM agent_profiles WHERE id = $1", profile_id)
    return JSONResponse(content=_row_to_profile(row), status_code=201)


@router.get("/profiles/{profile_id}/executions")
async def list_profile_executions(
    profile_id: str = Path(..., description="Agent profile UUID"),
    limit: int = 20,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """List recent execution log entries for an agent profile."""
    from services.database_manager.database_helpers import fetch_all, fetch_one
    _rls = _api_user_rls(current_user.user_id)
    profile = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        current_user.user_id,
        rls_context=_rls,
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    rows = await fetch_all(
        """
        SELECT id, agent_profile_id, user_id, query, playbook_id, started_at, completed_at,
               duration_ms, status, error_details, metadata, tokens_input, tokens_output, cost_usd, model_used
        FROM agent_execution_log
        WHERE agent_profile_id = $1 AND user_id = $2
        ORDER BY started_at DESC
        LIMIT $3
        """,
        profile_id,
        current_user.user_id,
        min(limit, 100),
        rls_context=_rls,
    )
    out = []
    for r in rows:
        meta = r.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        out.append({
            "id": str(r["id"]),
            "agent_profile_id": str(r["agent_profile_id"]),
            "query": (r.get("query") or "")[:500],
            "playbook_id": str(r["playbook_id"]) if r.get("playbook_id") else None,
            "started_at": r["started_at"].isoformat() if r.get("started_at") else None,
            "completed_at": r["completed_at"].isoformat() if r.get("completed_at") else None,
            "duration_ms": r.get("duration_ms"),
            "status": r.get("status"),
            "error_details": (r.get("error_details") or "")[:1000],
            "steps_completed": meta.get("steps_completed"),
            "steps_total": meta.get("steps_total"),
            "tokens_input": r.get("tokens_input"),
            "tokens_output": r.get("tokens_output"),
            "cost_usd": float(r["cost_usd"]) if r.get("cost_usd") is not None else None,
            "model_used": r.get("model_used"),
        })
    return JSONResponse(content=out)


@router.get("/profiles/{profile_id}/executions/{execution_id}")
async def get_execution(
    profile_id: str = Path(..., description="Agent profile UUID"),
    execution_id: str = Path(..., description="Execution log UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Get a single execution log entry (for detail viewer)."""
    from services.database_manager.database_helpers import fetch_one, fetch_all
    _rls = _api_user_rls(current_user.user_id)
    profile = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        current_user.user_id,
        rls_context=_rls,
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    row = await fetch_one(
        """
        SELECT id, agent_profile_id, user_id, query, playbook_id, started_at, completed_at,
               duration_ms, status, error_details, metadata, output_destinations, connectors_called,
               tokens_input, tokens_output, cost_usd, model_used
        FROM agent_execution_log
        WHERE id = $1 AND agent_profile_id = $2 AND user_id = $3
        """,
        execution_id,
        profile_id,
        current_user.user_id,
        rls_context=_rls,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Execution not found")
    meta = row.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {}
    steps_rows = await fetch_all(
        """
        SELECT step_index, step_name, step_type, action_name, status,
               started_at, completed_at, duration_ms, inputs_json, outputs_json, error_details, tool_call_trace
        FROM agent_execution_steps
        WHERE execution_id = $1
        ORDER BY step_index
        """,
        execution_id,
        rls_context=_rls,
    )
    steps = []
    for s in (steps_rows or []):
        inputs = s.get("inputs_json") or {}
        outputs = s.get("outputs_json") or {}
        tool_call_trace = s.get("tool_call_trace")
        if isinstance(inputs, str):
            try:
                inputs = json.loads(inputs)
            except json.JSONDecodeError:
                inputs = {}
        if isinstance(outputs, str):
            try:
                outputs = json.loads(outputs)
            except json.JSONDecodeError:
                outputs = {}
        if isinstance(tool_call_trace, str):
            try:
                tool_call_trace = json.loads(tool_call_trace)
            except json.JSONDecodeError:
                tool_call_trace = []
        if not isinstance(tool_call_trace, list):
            tool_call_trace = []
        steps.append({
            "step_index": s.get("step_index"),
            "step_name": s.get("step_name"),
            "step_type": s.get("step_type"),
            "action_name": s.get("action_name"),
            "status": s.get("status"),
            "started_at": s["started_at"].isoformat() if s.get("started_at") else None,
            "completed_at": s["completed_at"].isoformat() if s.get("completed_at") else None,
            "duration_ms": s.get("duration_ms"),
            "inputs": inputs,
            "outputs": outputs,
            "error_details": s.get("error_details"),
            "tool_call_trace": tool_call_trace,
        })
    return JSONResponse(content={
        "id": str(row["id"]),
        "agent_profile_id": str(row["agent_profile_id"]),
        "user_id": row["user_id"],
        "query": row.get("query") or "",
        "playbook_id": str(row["playbook_id"]) if row.get("playbook_id") else None,
        "started_at": row["started_at"].isoformat() if row.get("started_at") else None,
        "completed_at": row["completed_at"].isoformat() if row.get("completed_at") else None,
        "duration_ms": row.get("duration_ms"),
        "status": row.get("status"),
        "error_details": row.get("error_details"),
        "metadata": meta,
        "output_destinations": row.get("output_destinations") or [],
        "connectors_called": row.get("connectors_called") or [],
        "tokens_input": row.get("tokens_input"),
        "tokens_output": row.get("tokens_output"),
        "cost_usd": float(row["cost_usd"]) if row.get("cost_usd") is not None else None,
        "model_used": row.get("model_used"),
        "steps": steps,
    })


@router.delete("/profiles/{profile_id}/executions/{execution_id}")
async def delete_execution(
    profile_id: str = Path(..., description="Agent profile UUID"),
    execution_id: str = Path(..., description="Execution log UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Delete a single execution (and its steps/discoveries via CASCADE)."""
    from services.database_manager.database_helpers import fetch_one, execute
    _rls = _api_user_rls(current_user.user_id)
    profile = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        current_user.user_id,
        rls_context=_rls,
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    row = await fetch_one(
        "SELECT id FROM agent_execution_log WHERE id = $1 AND agent_profile_id = $2 AND user_id = $3",
        execution_id,
        profile_id,
        current_user.user_id,
        rls_context=_rls,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Execution not found")
    await execute(
        "DELETE FROM agent_execution_log WHERE id = $1 AND agent_profile_id = $2 AND user_id = $3",
        execution_id,
        profile_id,
        current_user.user_id,
        rls_context=_rls,
    )
    return JSONResponse(content={"deleted": True, "id": execution_id})


@router.delete("/profiles/{profile_id}/executions")
async def clear_executions(
    profile_id: str = Path(..., description="Agent profile UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Delete all executions for a profile (journal clear). Cascades to agent_execution_steps and agent_discoveries."""
    from services.database_manager.database_helpers import fetch_one, execute
    _rls = _api_user_rls(current_user.user_id)
    profile = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        current_user.user_id,
        rls_context=_rls,
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    await execute(
        "DELETE FROM agent_execution_log WHERE agent_profile_id = $1 AND user_id = $2",
        profile_id,
        current_user.user_id,
        rls_context=_rls,
    )
    return JSONResponse(content={"deleted": True, "profile_id": profile_id})


# ---------- Dashboard (fleet status, cost summary, activity feed) ----------

@router.get("/dashboard/fleet-status")
async def dashboard_fleet_status(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """All agents with schedule info, last execution, budget status for operations dashboard."""
    from services.database_manager.database_helpers import fetch_all
    user_id = current_user.user_id
    _rls = _api_user_rls(user_id)
    rows = await fetch_all(
        """
        SELECT p.id, p.name, p.handle, p.is_active, p.model_preference,
               e.status AS last_execution_status, e.started_at AS last_execution_at, e.duration_ms AS last_duration_ms, e.cost_usd AS last_cost_usd,
               s.id AS schedule_id, s.is_active AS schedule_active, s.next_run_at, s.last_run_at, s.consecutive_failures, s.max_consecutive_failures,
               b.monthly_limit_usd, b.current_period_start, b.current_period_spend_usd, b.warning_threshold_pct, b.enforce_hard_limit
        FROM agent_profiles p
        LEFT JOIN LATERAL (
            SELECT status, started_at, duration_ms, cost_usd FROM agent_execution_log
            WHERE agent_profile_id = p.id AND user_id = p.user_id
            ORDER BY started_at DESC LIMIT 1
        ) e ON true
        LEFT JOIN LATERAL (
            SELECT id, is_active, next_run_at, last_run_at, consecutive_failures, max_consecutive_failures FROM agent_schedules
            WHERE agent_schedules.agent_profile_id = p.id
            ORDER BY next_run_at ASC NULLS LAST LIMIT 1
        ) s ON true
        LEFT JOIN agent_budgets b ON b.agent_profile_id = p.id
        WHERE p.user_id = $1
        ORDER BY p.updated_at DESC
        """,
        user_id,
        rls_context=_rls,
    )
    out = []
    for r in rows:
        spend = float(r.get("current_period_spend_usd") or 0)
        limit = r.get("monthly_limit_usd")
        over_limit = limit is not None and limit > 0 and spend >= float(limit)
        out.append({
            "id": str(r["id"]),
            "name": r.get("name") or r.get("handle") or "Unnamed",
            "handle": r.get("handle"),
            "is_active": r.get("is_active", True),
            "model_preference": r.get("model_preference"),
            "last_execution_status": r.get("last_execution_status"),
            "last_execution_at": r["last_execution_at"].isoformat() if r.get("last_execution_at") else None,
            "last_duration_ms": r.get("last_duration_ms"),
            "last_cost_usd": float(r["last_cost_usd"]) if r.get("last_cost_usd") is not None else None,
            "schedule_id": str(r["schedule_id"]) if r.get("schedule_id") else None,
            "schedule_active": r.get("schedule_active"),
            "next_run_at": r["next_run_at"].isoformat() if r.get("next_run_at") else None,
            "last_run_at": r["last_run_at"].isoformat() if r.get("last_run_at") else None,
            "consecutive_failures": r.get("consecutive_failures", 0),
            "max_consecutive_failures": r.get("max_consecutive_failures"),
            "budget": {
                "monthly_limit_usd": float(r["monthly_limit_usd"]) if r.get("monthly_limit_usd") is not None else None,
                "current_period_spend_usd": spend,
                "current_period_start": r["current_period_start"].isoformat() if r.get("current_period_start") else None,
                "warning_threshold_pct": r.get("warning_threshold_pct", 80),
                "enforce_hard_limit": r.get("enforce_hard_limit", True),
                "over_limit": over_limit,
            } if r.get("monthly_limit_usd") is not None else None,
        })
    return JSONResponse(content=out)


@router.get("/dashboard/cost-summary")
async def dashboard_cost_summary(
    period: str = "month",
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Aggregate spend by agent for the current period (from agent_budgets)."""
    from services.database_manager.database_helpers import fetch_all
    from datetime import date
    user_id = current_user.user_id
    today = date.today()
    period_start = today.replace(day=1)
    _rls = _api_user_rls(user_id)
    rows = await fetch_all(
        """
        SELECT b.agent_profile_id, p.name, p.handle,
               b.current_period_start, b.current_period_spend_usd, b.monthly_limit_usd
        FROM agent_budgets b
        JOIN agent_profiles p ON p.id = b.agent_profile_id AND p.user_id = b.user_id
        WHERE b.user_id = $1
        """,
        user_id,
        rls_context=_rls,
    )
    by_agent = []
    total = 0
    for r in rows:
        start = r.get("current_period_start")
        if start and (getattr(start, "year", None) != today.year or getattr(start, "month", None) != today.month):
            continue
        spend = float(r.get("current_period_spend_usd") or 0)
        total += spend
        by_agent.append({
            "agent_profile_id": str(r["agent_profile_id"]),
            "name": r.get("name") or r.get("handle") or "Unnamed",
            "handle": r.get("handle"),
            "spend_usd": spend,
            "limit_usd": float(r["monthly_limit_usd"]) if r.get("monthly_limit_usd") is not None else None,
        })
    by_agent.sort(key=lambda x: -x["spend_usd"])
    return JSONResponse(content={
        "period": period,
        "period_start": period_start.isoformat(),
        "total_spend_usd": round(total, 4),
        "by_agent": by_agent,
    })


@router.get("/dashboard/activity-feed")
async def dashboard_activity_feed(
    limit: int = 50,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Recent execution events across all agents for the dashboard activity feed."""
    from services.database_manager.database_helpers import fetch_all
    user_id = current_user.user_id
    _rls = _api_user_rls(user_id)
    rows = await fetch_all(
        """
        SELECT e.id, e.agent_profile_id, e.query, e.status, e.started_at, e.completed_at, e.duration_ms, e.error_details, e.cost_usd, e.trigger_type,
               p.name AS agent_name, p.handle AS agent_handle
        FROM agent_execution_log e
        JOIN agent_profiles p ON p.id = e.agent_profile_id AND p.user_id = e.user_id
        WHERE e.user_id = $1
        ORDER BY e.started_at DESC
        LIMIT $2
        """,
        user_id,
        min(limit, 100),
        rls_context=_rls,
    )
    out = []
    for r in rows:
        out.append({
            "id": str(r["id"]),
            "agent_profile_id": str(r["agent_profile_id"]),
            "agent_name": r.get("agent_name") or r.get("agent_handle") or "Unnamed",
            "agent_handle": r.get("agent_handle"),
            "query": (r.get("query") or "")[:200],
            "status": r.get("status"),
            "started_at": r["started_at"].isoformat() if r.get("started_at") else None,
            "completed_at": r["completed_at"].isoformat() if r.get("completed_at") else None,
            "duration_ms": r.get("duration_ms"),
            "error_details": (r.get("error_details") or "")[:500] if r.get("error_details") else None,
            "cost_usd": float(r["cost_usd"]) if r.get("cost_usd") is not None else None,
            "trigger_type": r.get("trigger_type") or "manual",
        })
    return JSONResponse(content=out)


# ---------- Approval queue (background approval for scheduled agents) ----------

@router.get("/approvals/pending")
async def list_pending_approvals(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """List pending approval queue entries for the current user."""
    from services.database_manager.database_helpers import fetch_all
    user_id = current_user.user_id
    _rls = _api_user_rls(user_id)
    rows = await fetch_all(
        """
        SELECT a.id, a.agent_profile_id, a.execution_id, a.step_name, a.prompt, a.preview_data, a.governance_type, a.status, a.created_at,
               p.name AS agent_name, p.handle AS agent_handle
        FROM agent_approval_queue a
        LEFT JOIN agent_profiles p ON p.id = a.agent_profile_id
        WHERE a.user_id = $1 AND a.status = 'pending'
        ORDER BY a.created_at DESC
        """,
        user_id,
        rls_context=_rls,
    )
    out = []
    for r in rows:
        out.append({
            "id": str(r["id"]),
            "agent_profile_id": str(r["agent_profile_id"]) if r.get("agent_profile_id") else None,
            "execution_id": str(r["execution_id"]) if r.get("execution_id") else None,
            "step_name": r.get("step_name", ""),
            "prompt": r.get("prompt", ""),
            "preview_data": r.get("preview_data") or {},
            "governance_type": r.get("governance_type") or "playbook_step",
            "status": r.get("status", "pending"),
            "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
            "agent_name": r.get("agent_name") or r.get("agent_handle") or "Agent",
        })
    return JSONResponse(content=out)


class RespondApprovalBody(BaseModel):
    """Approve or reject a parked approval."""
    approved: bool = Field(..., description="True to approve and resume, false to reject")


@router.post("/approvals/{approval_id}/respond")
async def respond_approval(
    approval_id: str,
    body: RespondApprovalBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Mark approval as approved or rejected and, if approved, enqueue resume task."""
    from services.database_manager.database_helpers import fetch_one, execute
    from datetime import datetime, timezone
    user_id = current_user.user_id
    row = await fetch_one(
        "SELECT id, status, playbook_config, thread_id, user_id, agent_profile_id FROM agent_approval_queue WHERE id = $1 AND user_id = $2",
        uuid.UUID(approval_id),
        user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Approval not found")
    if row["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"Approval already {row['status']}")
    now = datetime.now(timezone.utc)
    new_status = "approved" if body.approved else "rejected"
    await execute(
        "UPDATE agent_approval_queue SET status = $1, responded_at = $2 WHERE id = $3 AND user_id = $4",
        new_status,
        now,
        uuid.UUID(approval_id),
        user_id,
    )
    if body.approved:
        from services.celery_tasks.agent_tasks import resume_approved_agent
        resume_approved_agent.delay(approval_id, user_id)
    return JSONResponse(content={"ok": True, "status": new_status})


class NotifyExecutionEventBody(BaseModel):
    """Internal: notify execution event for dashboard live feed."""
    user_id: str = Field(..., description="User to notify")
    subtype: str = Field(..., description="execution_started, execution_completed, execution_failed, budget_warning, budget_exceeded, heartbeat_failed, team_budget_exceeded, team_escalations")
    execution_id: Optional[str] = Field(None, description="Execution log ID")
    agent_profile_id: Optional[str] = Field(None)
    agent_name: Optional[str] = Field(None)
    status: Optional[str] = Field(None)
    duration_ms: Optional[int] = Field(None)
    cost_usd: Optional[float] = Field(None)
    error_details: Optional[str] = Field(None)
    trigger_type: Optional[str] = Field(None, description="manual, scheduled")
    query: Optional[str] = Field(None)
    team_id: Optional[str] = Field(None, description="Team event: team UUID")
    team_name: Optional[str] = Field(None, description="Team event: team name")
    message: Optional[str] = Field(None, description="Short message for team events")


@router.post("/internal/notify-execution-event")
async def notify_execution_event(body: NotifyExecutionEventBody) -> JSONResponse:
    """Internal: send WebSocket notification for execution/dashboard events. Called from Celery or tools service."""
    try:
        from datetime import datetime, timezone
        from services.notification_router import route_notification

        payload = {
            "type": "agent_notification",
            "subtype": body.subtype,
            "execution_id": body.execution_id,
            "agent_profile_id": body.agent_profile_id,
            "agent_name": body.agent_name or "Agent",
            "status": body.status,
            "duration_ms": body.duration_ms,
            "cost_usd": body.cost_usd,
            "error_details": (body.error_details or "")[:500] if body.error_details else None,
            "trigger_type": body.trigger_type or "manual",
            "query": (body.query or "")[:200] if body.query else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if body.team_id is not None:
            payload["team_id"] = body.team_id
        if body.team_name is not None:
            payload["team_name"] = body.team_name
        if body.message is not None:
            payload["message"] = (body.message or "")[:500]
        an = body.agent_name or "Agent"
        st = body.subtype or "execution_event"
        payload["title"] = f"{an}: {st.replace('_', ' ')}"
        payload["preview"] = (
            (body.message or body.query or body.error_details or "")[:200]
            if (body.message or body.query or body.error_details)
            else ""
        )
        await route_notification(
            body.user_id,
            str(body.subtype or "execution_event"),
            payload,
            originating_surface_id=None,
        )
    except Exception as e:
        logger.warning("notify_execution_event WebSocket send failed: %s", e)
    return JSONResponse(content={"ok": True})


class NotifyLineTimelineBody(BaseModel):
    """Internal: notify line timeline subscribers (new message or execution_status)."""
    line_id: str = Field(..., description="Agent line UUID")
    payload: Dict[str, Any] = Field(..., description="Payload to broadcast (type, message, status, etc.)")


@router.post("/internal/notify-line-timeline")
async def notify_line_timeline(body: NotifyLineTimelineBody) -> JSONResponse:
    """Internal: send WebSocket line timeline update. Called from Celery or tools service."""
    try:
        from utils.websocket_manager import get_websocket_manager
        ws_manager = get_websocket_manager()
        if ws_manager:
            await ws_manager.send_line_timeline_update(body.line_id, body.payload)
    except Exception as e:
        logger.warning("notify_line_timeline WebSocket send failed: %s", e)
    return JSONResponse(content={"ok": True})


class NotifySchedulePausedBody(BaseModel):
    """Internal: notify user that a schedule was auto-paused (circuit breaker)."""
    user_id: str = Field(..., description="User to notify")
    agent_name: Optional[str] = Field(None, description="Agent profile name")
    consecutive: int = Field(..., description="Consecutive failure count")
    last_error: Optional[str] = Field(None, description="Last error message")


@router.post("/internal/notify-schedule-paused")
async def notify_schedule_paused(body: NotifySchedulePausedBody) -> JSONResponse:
    """Internal: send WebSocket notification when a schedule is auto-paused. Called from Celery."""
    try:
        from datetime import datetime, timezone
        from services.notification_router import route_notification

        agent_name = body.agent_name or "Scheduled Agent"
        preview = f"Paused after {body.consecutive} failures: {(body.last_error or '')[:100]}"
        await route_notification(
            body.user_id,
            "schedule_paused",
            {
                "type": "agent_notification",
                "subtype": "schedule_paused",
                "agent_name": agent_name,
                "title": f"{agent_name} schedule paused",
                "preview": preview,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            originating_surface_id=None,
        )
    except Exception as e:
        logger.warning("notify_schedule_paused WebSocket send failed: %s", e)
    return JSONResponse(content={"ok": True})


# ---------- Schedules ----------

def _compute_next_run_at(
    schedule_type: str,
    cron_expression: Optional[str],
    interval_seconds: Optional[int],
    timezone_str: str = "UTC",
):
    from datetime import datetime, timezone
    try:
        from croniter import croniter
    except ImportError:
        return None
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        ZoneInfo = None
    tz = timezone.utc
    if timezone_str and timezone_str != "UTC" and ZoneInfo is not None:
        try:
            tz = ZoneInfo(timezone_str)
        except Exception:
            pass
    now = datetime.now(timezone.utc)
    if schedule_type == "cron" and cron_expression:
        try:
            now_in_tz = now.astimezone(tz)
            now_naive = now_in_tz.replace(tzinfo=None)
            it = croniter(cron_expression, now_naive)
            next_naive = it.get_next(datetime)
            next_in_tz = next_naive.replace(tzinfo=tz)
            next_utc = next_in_tz.astimezone(timezone.utc)
            return next_utc
        except Exception:
            return None
    if schedule_type == "interval" and interval_seconds:
        from datetime import timedelta
        return now + timedelta(seconds=interval_seconds)
    return None


def _row_to_schedule(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {}
    return {
        "id": str(row["id"]),
        "agent_profile_id": str(row["agent_profile_id"]),
        "user_id": row.get("user_id"),
        "schedule_type": row.get("schedule_type"),
        "cron_expression": row.get("cron_expression"),
        "interval_seconds": row.get("interval_seconds"),
        "timezone": row.get("timezone") or "UTC",
        "is_active": row.get("is_active", True),
        "next_run_at": row["next_run_at"].isoformat() if row.get("next_run_at") else None,
        "last_run_at": row["last_run_at"].isoformat() if row.get("last_run_at") else None,
        "last_status": row.get("last_status"),
        "run_count": row.get("run_count", 0),
        "consecutive_failures": row.get("consecutive_failures", 0),
        "max_consecutive_failures": row.get("max_consecutive_failures", 5),
        "timeout_seconds": row.get("timeout_seconds", 300),
        "input_context": row.get("input_context") or {},
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
    }


@router.get("/profiles/{profile_id}/schedules")
async def list_schedules(
    profile_id: str = Path(..., description="Agent profile UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """List schedules for an agent profile."""
    from services.database_manager.database_helpers import fetch_all, fetch_one
    profile = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        current_user.user_id,
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    rows = await fetch_all(
        "SELECT * FROM agent_schedules WHERE agent_profile_id = $1 ORDER BY created_at",
        profile_id,
    )
    return JSONResponse(content=[_row_to_schedule(r) for r in rows])


@router.post("/profiles/{profile_id}/schedules")
async def create_schedule(
    profile_id: str = Path(..., description="Agent profile UUID"),
    body: AgentScheduleCreate = ...,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Create a schedule for an agent profile."""
    try:
        out = await agent_factory_service.create_schedule(
            current_user.user_id,
            profile_id,
            body.model_dump(),
        )
        return JSONResponse(content=out, status_code=201)
    except ValueError as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)


@router.get("/schedules/{schedule_id}")
async def get_schedule(
    schedule_id: str = Path(..., description="Schedule UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Get a single schedule by ID."""
    from services.database_manager.database_helpers import fetch_one
    row = await fetch_one(
        "SELECT * FROM agent_schedules WHERE id = $1 AND user_id = $2",
        schedule_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return JSONResponse(content=_row_to_schedule(row))


@router.put("/schedules/{schedule_id}")
async def update_schedule(
    schedule_id: str = Path(..., description="Schedule UUID"),
    body: AgentScheduleUpdate = ...,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Update a schedule."""
    from services.database_manager.database_helpers import fetch_one, execute
    row = await fetch_one(
        "SELECT * FROM agent_schedules WHERE id = $1 AND user_id = $2",
        schedule_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Schedule not found")
    updates = []
    args = []
    idx = 1
    if body.schedule_type is not None:
        updates.append(f"schedule_type = ${idx}")
        args.append(body.schedule_type)
        idx += 1
    if body.cron_expression is not None:
        updates.append(f"cron_expression = ${idx}")
        args.append(body.cron_expression)
        idx += 1
    if body.interval_seconds is not None:
        updates.append(f"interval_seconds = ${idx}")
        args.append(body.interval_seconds)
        idx += 1
    if body.timezone is not None:
        updates.append(f"timezone = ${idx}")
        args.append(body.timezone)
        idx += 1
    if body.timeout_seconds is not None:
        updates.append(f"timeout_seconds = ${idx}")
        args.append(body.timeout_seconds)
        idx += 1
    if body.max_consecutive_failures is not None:
        updates.append(f"max_consecutive_failures = ${idx}")
        args.append(body.max_consecutive_failures)
        idx += 1
    if body.input_context is not None:
        updates.append(f"input_context = ${idx}::jsonb")
        args.append(json.dumps(body.input_context))
        idx += 1
    if body.is_active is not None:
        updates.append(f"is_active = ${idx}")
        args.append(body.is_active)
        idx += 1
    if not updates:
        return JSONResponse(content=_row_to_schedule(row))
    updates.append("updated_at = NOW()")
    args.extend([schedule_id, current_user.user_id])
    await execute(
        f"UPDATE agent_schedules SET {', '.join(updates)} WHERE id = ${idx + 1} AND user_id = ${idx + 2}",
        *args,
    )
    row = await fetch_one("SELECT * FROM agent_schedules WHERE id = $1", schedule_id)
    return JSONResponse(content=_row_to_schedule(row))


@router.delete("/schedules/{schedule_id}")
async def delete_schedule(
    schedule_id: str = Path(..., description="Schedule UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Delete a schedule."""
    from services.database_manager.database_helpers import execute, fetch_one
    row = await fetch_one(
        "SELECT id FROM agent_schedules WHERE id = $1 AND user_id = $2",
        schedule_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Schedule not found")
    await execute("DELETE FROM agent_schedules WHERE id = $1 AND user_id = $2", schedule_id, current_user.user_id)
    return JSONResponse(content={"deleted": True, "id": schedule_id})


@router.post("/schedules/{schedule_id}/pause")
async def pause_schedule(
    schedule_id: str = Path(..., description="Schedule UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Pause a schedule (set is_active = false)."""
    from services.database_manager.database_helpers import execute, fetch_one
    row = await fetch_one(
        "SELECT id FROM agent_schedules WHERE id = $1 AND user_id = $2",
        schedule_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Schedule not found")
    await execute(
        "UPDATE agent_schedules SET is_active = false, updated_at = NOW() WHERE id = $1 AND user_id = $2",
        schedule_id,
        current_user.user_id,
    )
    row = await fetch_one("SELECT * FROM agent_schedules WHERE id = $1", schedule_id)
    return JSONResponse(content=_row_to_schedule(row))


@router.post("/schedules/{schedule_id}/resume")
async def resume_schedule(
    schedule_id: str = Path(..., description="Schedule UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Resume a schedule (set is_active = true, reset consecutive_failures, compute next_run_at)."""
    from services.database_manager.database_helpers import execute, fetch_one
    row = await fetch_one(
        "SELECT * FROM agent_schedules WHERE id = $1 AND user_id = $2",
        schedule_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Schedule not found")
    next_run_at = _compute_next_run_at(
        row["schedule_type"],
        row.get("cron_expression"),
        row.get("interval_seconds"),
        row.get("timezone") or "UTC",
    )
    await execute(
        """
        UPDATE agent_schedules
        SET is_active = true, consecutive_failures = 0, next_run_at = $1, updated_at = NOW()
        WHERE id = $2 AND user_id = $3
        """,
        next_run_at,
        schedule_id,
        current_user.user_id,
    )
    row = await fetch_one("SELECT * FROM agent_schedules WHERE id = $1", schedule_id)
    return JSONResponse(content=_row_to_schedule(row))


# ---------- Agent Data Sources (nested under profile) ----------

def _row_to_data_source(
    row: Dict[str, Any],
    connector_name: Optional[str] = None,
    connector_definition: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not row:
        return {}
    out = {
        "id": str(row["id"]),
        "agent_profile_id": str(row["agent_profile_id"]),
        "connector_id": str(row["connector_id"]),
        "credentials_encrypted": row.get("credentials_encrypted"),
        "config_overrides": row.get("config_overrides") or {},
        "permissions": row.get("permissions") or {},
        "is_enabled": row.get("is_enabled", True),
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
    }
    if connector_name is not None:
        out["connector_name"] = connector_name
    if connector_definition is not None:
        endpoints = connector_definition.get("endpoints") or {}
        out["connector_endpoints"] = list(endpoints.keys()) if isinstance(endpoints, dict) else []
    return out


@router.get("/profiles/{profile_id}/data-sources")
async def list_data_sources(
    profile_id: str = Path(..., description="Agent profile UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """List data source bindings for an agent profile."""
    from services.database_manager.database_helpers import fetch_all, fetch_one
    profile = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        current_user.user_id,
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    rows = await fetch_all(
        """
        SELECT ads.*, dsc.name AS connector_name, dsc.definition AS connector_definition
        FROM agent_data_sources ads
        LEFT JOIN data_source_connectors dsc ON dsc.id = ads.connector_id
        WHERE ads.agent_profile_id = $1 ORDER BY ads.created_at
        """,
        profile_id,
    )
    result = []
    for r in rows:
        definition = r.get("connector_definition")
        if isinstance(definition, str):
            try:
                definition = json.loads(definition)
            except json.JSONDecodeError:
                definition = {}
        result.append(_row_to_data_source(r, r.get("connector_name"), definition))
    return JSONResponse(content=result)


@router.post("/profiles/{profile_id}/data-sources-from-template")
async def create_data_source_from_template(
    profile_id: str = Path(..., description="Agent profile UUID"),
    body: DataSourceFromTemplateCreate = ...,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Create a new connector from a template and attach it to the profile."""
    from services.database_manager.database_helpers import fetch_one, execute
    from services.connector_templates import CONNECTOR_TEMPLATES

    profile = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        current_user.user_id,
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    template = next((t for t in CONNECTOR_TEMPLATES if t.get("name") == body.template_name), None)
    if not template:
        raise HTTPException(status_code=400, detail=f"Unknown template: {body.template_name}")

    definition = template.get("definition") or {}
    await execute(
        """
        INSERT INTO data_source_connectors (
            user_id, name, description, connector_type, version, definition,
            is_template, requires_auth, auth_fields, icon, category, tags
        ) VALUES ($1, $2, $3, $4, $5, $6, false, $7, $8, $9, $10, $11)
        """,
        current_user.user_id,
        template.get("name"),
        template.get("description"),
        template.get("connector_type", "rest"),
        "1.0",
        json.dumps(definition),
        template.get("requires_auth", False),
        json.dumps(template.get("auth_fields") or []),
        template.get("icon"),
        template.get("category"),
        json.dumps([]),
    )
    connector_row = await fetch_one(
        "SELECT id FROM data_source_connectors WHERE user_id = $1 AND name = $2 ORDER BY created_at DESC LIMIT 1",
        current_user.user_id,
        template.get("name"),
    )
    if not connector_row:
        raise HTTPException(status_code=500, detail="Failed to create connector")

    connector_id = str(connector_row["id"])
    await execute(
        """
        INSERT INTO agent_data_sources (
            agent_profile_id, connector_id, credentials_encrypted,
            config_overrides, permissions, is_enabled
        ) VALUES ($1, $2, $3, $4, $5, $6)
        """,
        profile_id,
        connector_id,
        None,
        json.dumps({}),
        json.dumps({}),
        True,
    )
    row = await fetch_one(
        "SELECT * FROM agent_data_sources WHERE agent_profile_id = $1 AND connector_id = $2 ORDER BY created_at DESC LIMIT 1",
        profile_id,
        connector_id,
    )
    out = _row_to_data_source(row)
    out["connector_name"] = template.get("name")
    return JSONResponse(content=out, status_code=201)


@router.post("/profiles/{profile_id}/data-sources")
async def create_data_source(
    profile_id: str = Path(..., description="Agent profile UUID"),
    body: AgentDataSourceCreate = ...,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Add a data source binding to an agent profile."""
    try:
        out = await agent_factory_service.create_data_source_binding(
            current_user.user_id,
            profile_id,
            body.model_dump(),
        )
        return JSONResponse(content=out, status_code=201)
    except ValueError as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)


@router.get("/profiles/{profile_id}/data-sources/{source_id}")
async def get_data_source(
    profile_id: str = Path(..., description="Agent profile UUID"),
    source_id: str = Path(..., description="Data source binding UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Get a single data source binding."""
    from services.database_manager.database_helpers import fetch_one
    profile = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        current_user.user_id,
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    row = await fetch_one(
        "SELECT * FROM agent_data_sources WHERE id = $1 AND agent_profile_id = $2",
        source_id,
        profile_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Data source not found")
    return JSONResponse(content=_row_to_data_source(row))


@router.put("/profiles/{profile_id}/data-sources/{source_id}")
async def update_data_source(
    profile_id: str = Path(..., description="Agent profile UUID"),
    source_id: str = Path(..., description="Data source binding UUID"),
    body: AgentDataSourceUpdate = ...,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Update a data source binding."""
    from services.database_manager.database_helpers import fetch_one, execute
    profile = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        current_user.user_id,
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    row = await fetch_one(
        "SELECT * FROM agent_data_sources WHERE id = $1 AND agent_profile_id = $2",
        source_id,
        profile_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Data source not found")
    updates = body.model_dump(exclude_unset=True)
    if not updates:
        return JSONResponse(content=_row_to_data_source(row))
    set_clauses = []
    args = []
    idx = 1
    json_fields = ("credentials_encrypted", "config_overrides", "permissions")
    for k, v in updates.items():
        set_clauses.append(f"{k} = ${idx}")
        args.append(json.dumps(v) if k in json_fields else v)
        idx += 1
    args.extend([source_id, profile_id])
    await execute(
        f"UPDATE agent_data_sources SET {', '.join(set_clauses)} WHERE id = ${idx + 1} AND agent_profile_id = ${idx + 2}",
        *args,
    )
    row = await fetch_one("SELECT * FROM agent_data_sources WHERE id = $1", source_id)
    return JSONResponse(content=_row_to_data_source(row))


@router.delete("/profiles/{profile_id}/data-sources/{source_id}")
async def delete_data_source(
    profile_id: str = Path(..., description="Agent profile UUID"),
    source_id: str = Path(..., description="Data source binding UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Remove a data source binding from an agent profile."""
    from services.database_manager.database_helpers import fetch_one, execute
    profile = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        current_user.user_id,
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    row = await fetch_one(
        "SELECT id FROM agent_data_sources WHERE id = $1 AND agent_profile_id = $2",
        source_id,
        profile_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Data source not found")
    await execute(
        "DELETE FROM agent_data_sources WHERE id = $1 AND agent_profile_id = $2",
        source_id,
        profile_id,
    )
    return JSONResponse(content={"deleted": True, "id": source_id})


# ---------- Plugin configs (per-profile plugin credentials) ----------

class PluginConfigUpdate(BaseModel):
    """Upsert one plugin config for a profile."""
    plugin_name: str = Field(..., min_length=1, max_length=100)
    credentials_encrypted: Optional[Dict[str, Any]] = None
    is_enabled: bool = True


class PluginConfigsBulkUpdate(BaseModel):
    """Bulk upsert plugin configs for a profile."""
    configs: List[PluginConfigUpdate] = Field(default_factory=list)


def _row_to_plugin_config(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {}
    return {
        "id": str(row["id"]),
        "agent_profile_id": str(row["agent_profile_id"]),
        "plugin_name": row["plugin_name"],
        "has_credentials": bool(row.get("credentials_encrypted")),
        "is_enabled": row.get("is_enabled", True),
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
    }


@router.get("/profiles/{profile_id}/plugin-configs")
async def list_plugin_configs(
    profile_id: str = Path(..., description="Agent profile UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """List plugin configs for an agent profile (no raw credentials)."""
    from services.database_manager.database_helpers import fetch_all, fetch_one
    profile = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        current_user.user_id,
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    try:
        rows = await fetch_all(
            "SELECT id, agent_profile_id, plugin_name, credentials_encrypted, is_enabled, created_at, updated_at "
            "FROM agent_plugin_configs WHERE agent_profile_id = $1 ORDER BY plugin_name",
            profile_id,
        )
    except Exception as e:
        if "does not exist" in str(e).lower() or "relation" in str(e).lower():
            return JSONResponse(content=[])
        raise
    return JSONResponse(content=[_row_to_plugin_config(r) for r in rows])


@router.put("/profiles/{profile_id}/plugin-configs")
async def upsert_plugin_configs(
    profile_id: str = Path(..., description="Agent profile UUID"),
    body: PluginConfigsBulkUpdate = ...,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Upsert plugin configs for an agent profile."""
    from services.database_manager.database_helpers import fetch_one, execute, fetch_all
    profile = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        current_user.user_id,
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    try:
        for cfg in body.configs:
            await execute(
                """
                INSERT INTO agent_plugin_configs (agent_profile_id, plugin_name, credentials_encrypted, is_enabled, updated_at)
                VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (agent_profile_id, plugin_name)
                DO UPDATE SET credentials_encrypted = COALESCE(EXCLUDED.credentials_encrypted, agent_plugin_configs.credentials_encrypted),
                    is_enabled = EXCLUDED.is_enabled,
                    updated_at = NOW()
                """,
                profile_id,
                cfg.plugin_name,
                json.dumps(cfg.credentials_encrypted) if cfg.credentials_encrypted is not None else None,
                cfg.is_enabled,
            )
        rows = await fetch_all(
            "SELECT id, agent_profile_id, plugin_name, credentials_encrypted, is_enabled, created_at, updated_at "
            "FROM agent_plugin_configs WHERE agent_profile_id = $1 ORDER BY plugin_name",
            profile_id,
        )
    except Exception as e:
        if "does not exist" in str(e).lower() or "relation" in str(e).lower():
            raise HTTPException(status_code=503, detail="Plugin configs table not migrated yet. Run migration 044.")
        raise
    return JSONResponse(content=[_row_to_plugin_config(r) for r in rows])


# ---------- Custom Playbooks CRUD ----------

def _ensure_json_obj(val, fallback=None):
    """Safely decode a value that might be double-encoded JSON."""
    if fallback is None:
        fallback = {}
    if val is None:
        return fallback
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return fallback


def _row_to_playbook(row: Dict[str, Any]) -> Dict[str, Any]:
    if not row:
        return {}
    return {
        "id": str(row["id"]),
        "user_id": row.get("user_id"),
        "name": row["name"],
        "description": row.get("description"),
        "version": row.get("version") or "1.0",
        "definition": _ensure_json_obj(row.get("definition"), {}),
        "triggers": _ensure_json_obj(row.get("triggers"), []),
        "is_template": row.get("is_template", False),
        "category": row.get("category"),
        "tags": list(row["tags"]) if row.get("tags") else [],
        "required_connectors": list(row["required_connectors"]) if row.get("required_connectors") else [],
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
        "is_locked": row.get("is_locked", False),
        "is_builtin": row.get("is_builtin", False),
    }


@router.get("/playbooks")
async def list_playbooks(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> JSONResponse:
    """List playbooks owned by the user or templates."""
    out = await agent_factory_service.list_playbooks(current_user.user_id)
    return JSONResponse(content=out)


@router.post("/playbooks")
async def create_playbook(
    body: CustomPlaybookCreate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Create a custom playbook."""
    out = await agent_factory_service.create_playbook(
        current_user.user_id,
        body.model_dump(),
    )
    return JSONResponse(content=out, status_code=201)


@router.get("/playbooks/{playbook_id}")
async def get_playbook(
    playbook_id: str = Path(..., description="Playbook UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Get a single playbook by ID (user's, template, or shared)."""
    from services.database_manager.database_helpers import fetch_one
    row = await fetch_one(
        """SELECT pb.*,
                  CASE
                      WHEN pb.user_id = $2 THEN 'owned'
                      WHEN pb.is_template = true THEN 'template'
                      ELSE 'shared'
                  END AS ownership,
                  u_owner.username AS owner_username,
                  u_owner.display_name AS owner_display_name
           FROM custom_playbooks pb
           LEFT JOIN users u_owner ON u_owner.user_id = pb.user_id
           WHERE pb.id = $1
             AND (pb.user_id = $2
                  OR pb.is_template = true
                  OR EXISTS (
                      SELECT 1 FROM agent_artifact_shares _sh
                      WHERE _sh.artifact_type = 'playbook'
                        AND _sh.artifact_id = pb.id
                        AND _sh.shared_with_user_id = $2
                  ))""",
        playbook_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Playbook not found")
    return JSONResponse(content=_row_to_playbook(row))


@router.get("/playbooks/{playbook_id}/usage")
async def get_playbook_usage(
    playbook_id: str = Path(..., description="Playbook UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Return which agent profiles use this playbook as default (for usage warning)."""
    from services.database_manager.database_helpers import fetch_all
    rows = await fetch_all(
        """
        SELECT id, name, handle
        FROM agent_profiles
        WHERE user_id = $1 AND default_playbook_id = $2::uuid
        ORDER BY name
        """,
        current_user.user_id,
        playbook_id,
    )
    result = [{"id": str(r["id"]), "name": r.get("name", ""), "handle": r.get("handle", "")} for r in rows]
    return JSONResponse(content=result)


@router.put("/playbooks/{playbook_id}")
async def update_playbook(
    playbook_id: str = Path(..., description="Playbook UUID"),
    body: CustomPlaybookUpdate = ...,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Update a playbook (only owner; templates are read-only)."""
    from services.database_manager.database_helpers import fetch_one, execute
    row = await fetch_one(
        "SELECT * FROM custom_playbooks WHERE id = $1 AND user_id = $2",
        playbook_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Playbook not found")
    if row.get("is_template"):
        raise HTTPException(status_code=403, detail="Cannot update template playbook")
    updates = body.model_dump(exclude_unset=True)
    if not updates:
        return JSONResponse(content=_row_to_playbook(row))
    if row.get("is_builtin"):
        raise HTTPException(status_code=403, detail="Built-in playbook cannot be updated")
    if row.get("is_locked") and set(updates.keys()) != {"is_locked"}:
        raise HTTPException(status_code=403, detail="Playbook is locked; only lock toggle is allowed")

    playbook_remediation_msgs: List[str] = []
    playbook_remediation_steps: List[str] = []
    if "definition" in updates:
        defn = updates["definition"]
        old_def = row.get("definition")
        if isinstance(old_def, str):
            try:
                old_def = json.loads(old_def) if old_def else {}
            except (json.JSONDecodeError, TypeError):
                old_def = {}
        if not isinstance(old_def, dict):
            old_def = {}
        if isinstance(defn, dict) and defn.get("steps") and old_def:
            from services import agent_factory_service
            agent_factory_service.merge_playbook_definition_steps(old_def, defn)
        if isinstance(defn, dict):
            from services import agent_factory_service

            defn, playbook_remediation_steps, playbook_remediation_msgs = (
                await agent_factory_service.validate_and_remediate_playbook_models_for_user(
                    current_user.user_id, defn
                )
            )
            updates["definition"] = defn
        def_str = json.dumps(defn)
        step_count = len(defn.get("steps", [])) if isinstance(defn, dict) else -1
        top_keys = list(defn.keys())[:10] if isinstance(defn, dict) else type(defn).__name__
        logger.info("update_playbook %s: def_bytes=%d steps=%d keys=%s", playbook_id, len(def_str), step_count, top_keys)
        # Large definitions are normal (long prompts, many steps). Preview is for debug only — does not block save.
        if len(def_str) > 5000:
            logger.debug(
                "update_playbook %s: definition size=%d bytes, first 500 chars: %s",
                playbook_id,
                len(def_str),
                def_str[:500],
            )
        if old_def is not None and old_def != {}:
            from services.database_manager.database_helpers import fetch_value
            next_ver = await fetch_value(
                "SELECT COALESCE(MAX(version_number), 0) + 1 FROM playbook_versions WHERE playbook_id = $1",
                playbook_id,
            )
            next_ver = int(next_ver) if next_ver is not None else 1
            await execute(
                """
                INSERT INTO playbook_versions (playbook_id, version_number, definition, created_by)
                VALUES ($1, $2, $3::jsonb, $4)
                """,
                playbook_id,
                next_ver,
                json.dumps(old_def) if isinstance(old_def, (dict, list)) else old_def,
                current_user.user_id,
            )

    set_clauses = []
    args = []
    idx = 1
    jsonb_fields = ("definition", "triggers")
    array_fields = ("tags", "required_connectors")
    for k, v in updates.items():
        if k in jsonb_fields:
            set_clauses.append(f"{k} = ${idx}::jsonb")
            args.append(json.dumps(v) if isinstance(v, (dict, list)) else v)
        elif k in array_fields:
            set_clauses.append(f"{k} = ${idx}")
            args.append(v)
        else:
            set_clauses.append(f"{k} = ${idx}")
            args.append(v)
        idx += 1
    set_clauses.append("updated_at = NOW()")
    args.extend([playbook_id, current_user.user_id])
    await execute(
        f"UPDATE custom_playbooks SET {', '.join(set_clauses)} WHERE id = ${idx} AND user_id = ${idx + 1}",
        *args,
    )
    row = await fetch_one("SELECT * FROM custom_playbooks WHERE id = $1", playbook_id)
    if playbook_remediation_msgs:
        await agent_factory_service.notify_playbook_model_remediation(
            current_user.user_id, playbook_id, playbook_remediation_steps, playbook_remediation_msgs
        )
    if "definition" in updates:
        try:
            from services.agent_artifact_sharing_service import refresh_transitive_shares
            await refresh_transitive_shares("playbook", playbook_id, current_user.user_id)
        except Exception as e:
            logger.warning("Failed to refresh transitive shares for playbook %s: %s", playbook_id, e)
    return JSONResponse(content=_row_to_playbook(row))


@router.delete("/playbooks/{playbook_id}")
async def delete_playbook(
    playbook_id: str = Path(..., description="Playbook UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Delete a playbook (only owner; templates are read-only). Blocked when playbook is locked."""
    from services.database_manager.database_helpers import fetch_one, execute
    row = await fetch_one(
        "SELECT id, is_template, is_locked, is_builtin FROM custom_playbooks WHERE id = $1 AND user_id = $2",
        playbook_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Playbook not found")
    if row.get("is_template"):
        raise HTTPException(status_code=403, detail="Cannot delete template playbook")
    if row.get("is_builtin"):
        raise HTTPException(status_code=403, detail="Built-in playbook cannot be deleted")
    if row.get("is_locked"):
        raise HTTPException(status_code=403, detail="Playbook is locked; unlock to delete")
    await execute("DELETE FROM custom_playbooks WHERE id = $1 AND user_id = $2", playbook_id, current_user.user_id)
    return JSONResponse(content={"deleted": True, "id": playbook_id})


@router.get("/playbooks/{playbook_id}/export")
async def export_playbook(
    playbook_id: str = Path(..., description="Playbook UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Export a playbook as JSON (for backup/import elsewhere)."""
    from services.database_manager.database_helpers import fetch_one
    row = await fetch_one(
        "SELECT * FROM custom_playbooks WHERE id = $1 AND (user_id = $2 OR is_template = true)",
        playbook_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Playbook not found")
    return JSONResponse(content=_row_to_playbook(row))


@router.get("/playbooks/{playbook_id}/versions")
async def list_playbook_versions(
    playbook_id: str = Path(..., description="Playbook UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """List version history for a playbook (id, version_number, label, created_at, created_by)."""
    from services.database_manager.database_helpers import fetch_one, fetch_all
    playbook = await fetch_one(
        "SELECT id FROM custom_playbooks WHERE id = $1 AND user_id = $2",
        playbook_id,
        current_user.user_id,
    )
    if not playbook:
        raise HTTPException(status_code=404, detail="Playbook not found")
    rows = await fetch_all(
        """
        SELECT id, version_number, label, created_at, created_by
        FROM playbook_versions
        WHERE playbook_id = $1
        ORDER BY version_number DESC
        """,
        playbook_id,
    )
    out = [
        {
            "id": str(r["id"]),
            "version_number": r.get("version_number"),
            "label": r.get("label"),
            "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
            "created_by": r.get("created_by"),
        }
        for r in (rows or [])
    ]
    return JSONResponse(content=out)


@router.get("/playbooks/{playbook_id}/versions/{version_id}")
async def get_playbook_version(
    playbook_id: str = Path(..., description="Playbook UUID"),
    version_id: str = Path(..., description="Version UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Get a single version with full definition."""
    from services.database_manager.database_helpers import fetch_one
    playbook = await fetch_one(
        "SELECT id FROM custom_playbooks WHERE id = $1 AND user_id = $2",
        playbook_id,
        current_user.user_id,
    )
    if not playbook:
        raise HTTPException(status_code=404, detail="Playbook not found")
    row = await fetch_one(
        "SELECT id, version_number, label, definition, description, created_at, created_by FROM playbook_versions WHERE id = $1 AND playbook_id = $2",
        version_id,
        playbook_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Version not found")
    defn = row.get("definition")
    if isinstance(defn, str):
        try:
            defn = json.loads(defn)
        except json.JSONDecodeError:
            defn = {}
    return JSONResponse(content={
        "id": str(row["id"]),
        "version_number": row.get("version_number"),
        "label": row.get("label"),
        "definition": defn,
        "description": row.get("description"),
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "created_by": row.get("created_by"),
    })


@router.post("/playbooks/{playbook_id}/restore/{version_id}")
async def restore_playbook_version(
    playbook_id: str = Path(..., description="Playbook UUID"),
    version_id: str = Path(..., description="Version UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Snapshot current definition to version history, then overwrite playbook with the version's definition."""
    from services.database_manager.database_helpers import fetch_one, execute, fetch_value
    row = await fetch_one(
        "SELECT * FROM custom_playbooks WHERE id = $1 AND user_id = $2",
        playbook_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Playbook not found")
    if row.get("is_template"):
        raise HTTPException(status_code=403, detail="Cannot update template playbook")
    if row.get("is_locked"):
        raise HTTPException(status_code=403, detail="Playbook is locked; unlock to restore")
    ver_row = await fetch_one(
        "SELECT definition FROM playbook_versions WHERE id = $1 AND playbook_id = $2",
        version_id,
        playbook_id,
    )
    if not ver_row:
        raise HTTPException(status_code=404, detail="Version not found")
    old_def = row.get("definition")
    if old_def is not None:
        next_ver = await fetch_value(
            "SELECT COALESCE(MAX(version_number), 0) + 1 FROM playbook_versions WHERE playbook_id = $1",
            playbook_id,
        )
        next_ver = int(next_ver) if next_ver is not None else 1
        await execute(
            """
            INSERT INTO playbook_versions (playbook_id, version_number, definition, created_by)
            VALUES ($1, $2, $3::jsonb, $4)
            """,
            playbook_id,
            next_ver,
            json.dumps(old_def) if isinstance(old_def, (dict, list)) else old_def,
            current_user.user_id,
        )
    new_def = ver_row.get("definition")
    if isinstance(new_def, str):
        try:
            new_def = json.loads(new_def)
        except json.JSONDecodeError:
            new_def = {}
    restore_msgs: List[str] = []
    restore_steps: List[str] = []
    if isinstance(new_def, dict):
        new_def, restore_steps, restore_msgs = await agent_factory_service.validate_and_remediate_playbook_models_for_user(
            current_user.user_id, new_def
        )
    await execute(
        "UPDATE custom_playbooks SET definition = $1::jsonb, updated_at = NOW() WHERE id = $2 AND user_id = $3",
        json.dumps(new_def) if isinstance(new_def, (dict, list)) else new_def,
        playbook_id,
        current_user.user_id,
    )
    row = await fetch_one("SELECT * FROM custom_playbooks WHERE id = $1", playbook_id)
    if restore_msgs:
        await agent_factory_service.notify_playbook_model_remediation(
            current_user.user_id, playbook_id, restore_steps, restore_msgs
        )
    return JSONResponse(content=_row_to_playbook(row))


@router.post("/playbooks/{playbook_id}/clone")
async def clone_playbook(
    playbook_id: str = Path(..., description="Playbook UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Create a new playbook with the same definition, named '{original name} (Copy)'.
    Uses deep-clone path (with skill remapping) for shared or template playbooks
    to produce fully independent artifacts.
    """
    from services.database_manager.database_helpers import fetch_one, fetch_value
    from services import agent_artifact_clone_service

    row = await fetch_one(
        """SELECT * FROM custom_playbooks
           WHERE id = $1
             AND (user_id = $2
                  OR is_template = true
                  OR EXISTS (
                      SELECT 1 FROM agent_artifact_shares _sh
                      WHERE _sh.artifact_type = 'playbook'
                        AND _sh.artifact_id = custom_playbooks.id
                        AND _sh.shared_with_user_id = $2
                  ))""",
        playbook_id,
        current_user.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Playbook not found")

    new_id, _ = await agent_artifact_clone_service.deep_clone_playbook(
        playbook_id, current_user.user_id
    )
    if not new_id:
        raise HTTPException(status_code=500, detail="Clone failed")

    new_row = await fetch_one("SELECT * FROM custom_playbooks WHERE id = $1", new_id)
    return JSONResponse(content=_row_to_playbook(new_row), status_code=201)


@router.post("/playbooks/import")
async def import_playbook(
    body: Dict[str, Any],
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Import a playbook from JSON (creates new playbook for current user)."""
    from services.database_manager.database_helpers import fetch_one

    name = (body.get("name") or "Imported playbook")[:255]
    base_name = name
    suffix = 0
    while True:
        candidate = f"{base_name}_{suffix}" if suffix else base_name
        existing = await fetch_one(
            "SELECT id FROM custom_playbooks WHERE user_id = $1 AND name = $2",
            current_user.user_id,
            candidate,
        )
        if not existing:
            name = candidate
            break
        suffix += 1

    definition = body.get("definition") or {}
    if not isinstance(definition, dict):
        definition = {}

    triggers = body.get("triggers") or []
    if not isinstance(triggers, list):
        triggers = []

    tags = body.get("tags") or []
    required_connectors = body.get("required_connectors") or []
    if not isinstance(tags, list):
        tags = []
    if not isinstance(required_connectors, list):
        required_connectors = []

    try:
        out = await agent_factory_service.create_playbook(
            current_user.user_id,
            {
                "name": name,
                "description": body.get("description"),
                "version": body.get("version") or "1.0",
                "definition": definition,
                "triggers": triggers,
                "category": body.get("category"),
                "tags": tags,
                "required_connectors": required_connectors,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=out, status_code=201)


# ---------- Skills ----------

@router.get("/skills/export")
async def export_skills(
    ids: Optional[str] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Response:
    """Export skills as YAML bundle. Optional query 'ids' = comma-separated UUIDs; if omitted, export all user skills."""
    if ids:
        skill_ids = [s.strip() for s in ids.split(",") if s.strip()]
        skills_list = await agent_skills_service.get_skills_by_ids(skill_ids)
        skills_list = [s for s in skills_list if not s.get("is_builtin")]
    else:
        skills_list = await agent_skills_service.list_skills(
            current_user.user_id, include_builtin=False,
        )
    skills_export: List[Dict[str, Any]] = []
    for s in skills_list:
        if s.get("is_builtin"):
            continue
        skills_export.append({
            "name": s.get("name"),
            "slug": s.get("slug"),
            "description": s.get("description"),
            "category": s.get("category"),
            "procedure": s.get("procedure") or "",
            "required_tools": s.get("required_tools") or [],
            "optional_tools": s.get("optional_tools") or [],
            "inputs_schema": _ensure_json_obj(s.get("inputs_schema"), {}),
            "outputs_schema": _ensure_json_obj(s.get("outputs_schema"), {}),
            "examples": _ensure_json_obj(s.get("examples"), []),
            "tags": list(s.get("tags") or []),
        })
    bundle = {
        "bastion_skills_bundle": {
            "version": "1",
            "exported_at": datetime.now(timezone.utc).isoformat(),
        },
        "skills": skills_export,
    }
    yaml_str = yaml.dump(bundle, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return Response(
        content=yaml_str,
        media_type="application/x-yaml",
        headers={"Content-Disposition": 'attachment; filename="skills.yaml"'},
    )


@router.post("/skills/import")
async def import_skills(
    body: Dict[str, Any],
    overwrite: bool = False,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Import skills from YAML. Body: { yaml: "..." }. Skip if slug exists unless overwrite=true (then update)."""
    yaml_str = body.get("yaml")
    if not yaml_str or not isinstance(yaml_str, str):
        raise HTTPException(status_code=400, detail="Request body must include 'yaml' string")
    try:
        bundle = yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")
    if not isinstance(bundle, dict):
        raise HTTPException(status_code=400, detail="YAML must be a mapping")
    skills_data = bundle.get("skills")
    if not isinstance(skills_data, list):
        raise HTTPException(status_code=400, detail="Bundle must include 'skills' list")
    created: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for item in skills_data:
        if not isinstance(item, dict):
            continue
        slug = (item.get("slug") or "").strip()
        if not slug:
            skipped.append({"slug": None, "reason": "missing slug"})
            continue
        existing = await agent_skills_service.get_skill_by_slug(slug, current_user.user_id)
        if existing and existing.get("is_builtin"):
            skipped.append({"slug": slug, "reason": "cannot overwrite built-in"})
            continue
        if existing and not existing.get("is_builtin"):
            if not overwrite:
                skipped.append({"slug": slug, "reason": "slug already exists"})
                continue
            try:
                out = await agent_skills_service.update_skill(
                    str(existing["id"]),
                    current_user.user_id,
                    procedure=item.get("procedure"),
                    required_tools=item.get("required_tools"),
                    optional_tools=item.get("optional_tools"),
                    name=item.get("name"),
                    description=item.get("description"),
                    category=item.get("category"),
                    inputs_schema=item.get("inputs_schema"),
                    outputs_schema=item.get("outputs_schema"),
                    examples=item.get("examples"),
                    tags=item.get("tags"),
                )
                created.append({"id": out.get("id"), "slug": slug, "name": out.get("name"), "updated": True})
            except ValueError as e:
                skipped.append({"slug": slug, "reason": str(e)})
            continue
        try:
            out = await agent_skills_service.create_skill(
                current_user.user_id,
                name=item.get("name") or slug,
                slug=slug,
                procedure=item.get("procedure", ""),
                required_tools=item.get("required_tools"),
                optional_tools=item.get("optional_tools"),
                description=item.get("description"),
                category=item.get("category"),
                inputs_schema=item.get("inputs_schema"),
                outputs_schema=item.get("outputs_schema"),
                examples=item.get("examples"),
                tags=item.get("tags"),
            )
            created.append({"id": out.get("id"), "slug": slug, "name": out.get("name")})
        except ValueError as e:
            skipped.append({"slug": slug, "reason": str(e)})
    return JSONResponse(content={"created": created, "skipped": skipped})


@router.get("/skills")
async def list_skills(
    category: Optional[str] = None,
    include_builtin: bool = True,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """List user skills and optionally built-in skills."""
    out = await agent_skills_service.list_skills(
        current_user.user_id,
        category=category,
        include_builtin=include_builtin,
    )
    return JSONResponse(content=out)


@router.get("/skills/metrics/summary")
async def get_skills_metrics_summary(
    limit: int = Query(20, ge=1, le=100, description="Max results per category"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Return top-used and lowest-success-rate skills summary."""
    summary = await agent_skills_service.get_skills_metrics_summary(limit)
    return JSONResponse(content=summary)


# ---------------------------------------------------------------------------
# Skill promotion / demotion recommendations (before /skills/{skill_id} to avoid shadowing)
# ---------------------------------------------------------------------------


@router.get("/skills/recommendations")
async def list_skill_recommendations(
    status: str = Query("pending", description="pending, applied, or dismissed"),
    limit: int = Query(50, ge=1, le=200),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """List skill promotion/demotion recommendations."""
    recs = await agent_skills_service.list_promotion_recommendations(status=status, limit=limit)
    return JSONResponse(content=recs)


@router.post("/skills/recommendations/{rec_id}/apply")
async def apply_skill_recommendation(
    rec_id: int = Path(..., description="Recommendation ID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Apply a pending recommendation (promote or demote the skill)."""
    try:
        result = await agent_skills_service.apply_recommendation(rec_id, current_user.user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=result)


@router.post("/skills/recommendations/{rec_id}/dismiss")
async def dismiss_skill_recommendation(
    rec_id: int = Path(..., description="Recommendation ID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Response:
    """Dismiss a pending recommendation without action."""
    try:
        await agent_skills_service.dismiss_recommendation(rec_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return Response(status_code=204)


@router.get("/skills/{skill_id}")
async def get_skill(
    skill_id: str = Path(..., description="Skill UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Get a single skill by ID."""
    skill = await agent_skills_service.get_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    if not skill.get("is_builtin") and skill.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=404, detail="Skill not found")
    return JSONResponse(content=skill)


@router.post("/skills")
async def create_skill(
    body: SkillCreate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Create a user skill."""
    try:
        out = await agent_skills_service.create_skill(
            current_user.user_id,
            name=body.name,
            slug=body.slug,
            procedure=body.procedure,
            required_tools=body.required_tools,
            required_connection_types=body.required_connection_types,
            optional_tools=body.optional_tools,
            description=body.description,
            category=body.category,
            inputs_schema=body.inputs_schema,
            outputs_schema=body.outputs_schema,
            examples=body.examples,
            tags=body.tags,
            is_core=body.is_core,
            depends_on=body.depends_on,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=out, status_code=201)


@router.put("/skills/{skill_id}")
async def update_skill(
    skill_id: str = Path(..., description="Skill UUID"),
    body: SkillUpdate = ...,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Update a user skill (creates new version)."""
    try:
        out = await agent_skills_service.update_skill(
            skill_id,
            current_user.user_id,
            procedure=body.procedure,
            required_tools=body.required_tools,
            required_connection_types=body.required_connection_types,
            optional_tools=body.optional_tools,
            name=body.name,
            description=body.description,
            category=body.category,
            inputs_schema=body.inputs_schema,
            outputs_schema=body.outputs_schema,
            examples=body.examples,
            tags=body.tags,
            improvement_rationale=body.improvement_rationale,
            evidence_metadata=body.evidence_metadata,
            is_core=body.is_core,
            depends_on=body.depends_on,
            as_candidate=body.as_candidate,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=out)


@router.delete("/skills/{skill_id}")
async def delete_skill(
    skill_id: str = Path(..., description="Skill UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Response:
    """Delete a user skill (built-in skills cannot be deleted)."""
    try:
        await agent_skills_service.delete_skill(skill_id, current_user.user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return Response(status_code=204)


@router.get("/skills/{skill_id}/versions")
async def list_skill_versions(
    skill_id: str = Path(..., description="Skill UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """List version history for a skill."""
    out = await agent_skills_service.list_skill_versions(skill_id, current_user.user_id)
    if not out:
        raise HTTPException(status_code=404, detail="Skill not found")
    return JSONResponse(content=out)


@router.post("/skills/{skill_id}/revert/{version_id}")
async def revert_skill(
    skill_id: str = Path(..., description="Skill UUID"),
    version_id: str = Path(..., description="Version UUID to revert to"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Revert skill to a previous version (creates new version with old content)."""
    try:
        out = await agent_skills_service.revert_skill_to_version(
            skill_id, version_id, current_user.user_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=out)


# ---------------------------------------------------------------------------
# Skill execution metrics
# ---------------------------------------------------------------------------


@router.get("/skills/{skill_id}/metrics")
async def get_skill_metrics(
    skill_id: str = Path(..., description="Skill UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Return aggregated execution metrics for a single skill."""
    metrics = await agent_skills_service.get_skill_metrics(skill_id)
    return JSONResponse(content=metrics)


# ---------------------------------------------------------------------------
# Skill candidate versioning
# ---------------------------------------------------------------------------


@router.get("/skills/{skill_id}/candidate")
async def get_skill_candidate(
    skill_id: str = Path(..., description="Active skill UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Return the candidate version of a skill, if any."""
    skill = await agent_skills_service.get_skill(skill_id, current_user.user_id)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    slug = skill.get("slug") or ""
    candidate = await agent_skills_service.get_candidate_for_slug(
        slug, user_id=current_user.user_id
    )
    if not candidate:
        return JSONResponse(content={"has_candidate": False, "candidate": None})
    return JSONResponse(content={"has_candidate": True, "candidate": candidate})


@router.post("/skills/{skill_id}/promote")
async def promote_skill_candidate(
    skill_id: str = Path(..., description="Candidate skill UUID to promote"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Promote a candidate skill version to active."""
    try:
        out = await agent_skills_service.promote_candidate(
            skill_id, current_user.user_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=out)


@router.post("/skills/{skill_id}/reject")
async def reject_skill_candidate(
    skill_id: str = Path(..., description="Candidate skill UUID to reject"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Response:
    """Reject and delete a candidate skill version."""
    try:
        await agent_skills_service.reject_candidate(
            skill_id, current_user.user_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return Response(status_code=204)


class CandidateWeightBody(BaseModel):
    weight: int = Field(..., ge=0, le=100, description="Traffic weight 0-100")


@router.patch("/skills/{skill_id}/candidate-weight")
async def set_skill_candidate_weight(
    skill_id: str = Path(..., description="Candidate skill UUID"),
    body: CandidateWeightBody = ...,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Update the traffic weight for a candidate skill version."""
    try:
        out = await agent_skills_service.set_candidate_weight(
            skill_id, body.weight, current_user.user_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=out)


# ---------------------------------------------------------------------------
# Artifact Sharing
# ---------------------------------------------------------------------------


class ArtifactShareBody(BaseModel):
    artifact_type: str = Field(..., description="agent_profile, playbook, or skill")
    artifact_id: str = Field(..., description="UUID of the artifact")
    shared_with_user_id: str = Field(..., description="Target user's user_id")


@router.post("/shares")
async def create_share(
    body: ArtifactShareBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Share an artifact with another user (cascades to dependencies)."""
    from services import agent_artifact_sharing_service
    try:
        result = await agent_artifact_sharing_service.share_artifact(
            artifact_type=body.artifact_type,
            artifact_id=body.artifact_id,
            owner_user_id=current_user.user_id,
            target_user_id=body.shared_with_user_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if result is None:
        raise HTTPException(status_code=404, detail="Artifact not found or not owned by you")
    try:
        from utils.websocket_manager import get_websocket_manager
        from services.database_manager.database_helpers import fetch_one
        artifact_name = ""
        table_map = {"agent_profile": "agent_profiles", "playbook": "custom_playbooks", "skill": "agent_skills"}
        table = table_map.get(body.artifact_type)
        if table:
            name_row = await fetch_one(f"SELECT name FROM {table} WHERE id = $1", body.artifact_id)
            artifact_name = name_row.get("name", "") if name_row else ""
        ws = get_websocket_manager()
        await ws.send_to_session({
            "type": "agent_notification",
            "subtype": "artifact_shared",
            "artifact_type": body.artifact_type,
            "artifact_id": body.artifact_id,
            "artifact_name": artifact_name,
            "shared_by": current_user.username if hasattr(current_user, "username") else current_user.user_id,
        }, body.shared_with_user_id)
    except Exception as e:
        logger.warning("WebSocket share notification failed: %s", e)
    if body.artifact_type == "agent_profile" and body.artifact_id:
        try:
            await _notify_agent_handles_for_profile(str(body.artifact_id))
        except Exception as e:
            logger.warning("agent_handles_changed after share failed: %s", e)
    return JSONResponse(content=result, status_code=201)


@router.delete("/shares/{share_id}")
async def revoke_share(
    share_id: str = Path(..., description="Share UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Revoke a share (owner only). Transitive child shares are auto-deleted."""
    from services import agent_artifact_sharing_service
    from services.database_manager.database_helpers import fetch_one

    pre = await fetch_one(
        """
        SELECT artifact_type, artifact_id::text AS artifact_id
        FROM agent_artifact_shares
        WHERE id = $1::uuid AND owner_user_id = $2
        """,
        share_id,
        current_user.user_id,
    )
    await agent_artifact_sharing_service.revoke_share(share_id, current_user.user_id)
    if pre and pre.get("artifact_type") == "agent_profile" and pre.get("artifact_id"):
        await _notify_agent_handles_for_profile(pre["artifact_id"])
    return JSONResponse(content={"revoked": True})


@router.get("/shares/mine")
async def list_my_shares(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """List artifacts I have shared with others (non-transitive only)."""
    from services import agent_artifact_sharing_service
    shares = await agent_artifact_sharing_service.list_shares_by_owner(current_user.user_id)
    return JSONResponse(content=shares)


@router.get("/shares/with-me")
async def list_shared_with_me(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """List artifacts shared with me (non-transitive only)."""
    from services import agent_artifact_sharing_service
    shares = await agent_artifact_sharing_service.list_shares_with_user(current_user.user_id)
    return JSONResponse(content=shares)


@router.get("/shares/artifact/{artifact_type}/{artifact_id}")
async def list_artifact_shares(
    artifact_type: str = Path(..., description="agent_profile, playbook, or skill"),
    artifact_id: str = Path(..., description="Artifact UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """List all direct recipients of a specific artifact."""
    from services import agent_artifact_sharing_service
    shares = await agent_artifact_sharing_service.list_shares_for_artifact(artifact_type, artifact_id)
    return JSONResponse(content=shares)


@router.post("/shares/{share_id}/copy-to-mine")
async def copy_shared_to_mine(
    share_id: str = Path(..., description="Share UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> JSONResponse:
    """Deep-clone a shared artifact into my workspace, then remove the share."""
    from services.database_manager.database_helpers import fetch_one, execute
    from services import agent_artifact_clone_service

    share_row = await fetch_one(
        "SELECT * FROM agent_artifact_shares WHERE id = $1 AND shared_with_user_id = $2",
        share_id,
        current_user.user_id,
    )
    if not share_row:
        raise HTTPException(status_code=404, detail="Share not found")

    artifact_type = share_row["artifact_type"]
    artifact_id = str(share_row["artifact_id"])
    result: Dict[str, Any] = {"artifact_type": artifact_type}

    if artifact_type == "skill":
        new_id, _ = await agent_artifact_clone_service.deep_clone_skill(
            artifact_id, current_user.user_id
        )
        result["new_skill_id"] = new_id

    elif artifact_type == "playbook":
        new_id, _ = await agent_artifact_clone_service.deep_clone_playbook(
            artifact_id, current_user.user_id
        )
        result["new_playbook_id"] = new_id

    elif artifact_type == "agent_profile":
        created = await agent_artifact_clone_service.deep_clone_agent_profile(
            artifact_id, current_user.user_id
        )
        result["new_profile"] = created

    else:
        raise HTTPException(status_code=400, detail=f"Unknown artifact type: {artifact_type}")

    parent_share = await fetch_one(
        """SELECT id FROM agent_artifact_shares
           WHERE artifact_type = $1 AND artifact_id = $2
             AND shared_with_user_id = $3 AND is_transitive = false""",
        artifact_type,
        artifact_id,
        current_user.user_id,
    )
    if parent_share:
        await execute(
            "DELETE FROM agent_artifact_shares WHERE id = $1",
            parent_share["id"],
        )

    if artifact_type == "agent_profile":
        await _notify_agent_handles_for_profile(artifact_id)

    return JSONResponse(content=result, status_code=201)
