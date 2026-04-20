"""gRPC handlers for Control Pane operations."""

import json
import logging

import grpc
from protos import tool_service_pb2
from utils.grpc_rls import grpc_user_rls as _grpc_rls

logger = logging.getLogger(__name__)


class ControlPaneHandlersMixin:
    """Mixin providing Control Pane gRPC handlers.

    Mixed into ToolServiceImplementation; provides handlers for listing playbooks,
    agent profiles, schedules, data sources, and control pane CRUD.
    """

    async def ListPlaybooks(
        self,
        request: tool_service_pb2.ListPlaybooksRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListPlaybooksResponse:
        """List playbooks owned by the user or templates."""
        try:
            from services import agent_factory_service
            user_id = request.user_id or "system"
            playbooks = await agent_factory_service.list_playbooks(user_id)
            result = []
            for p in playbooks:
                definition = p.get("definition") or {}
                if not isinstance(definition, dict):
                    definition = {}
                steps = definition.get("steps") or []
                triggers = p.get("triggers") or []
                result.append({
                    "id": p.get("id"),
                    "name": p.get("name", ""),
                    "description": p.get("description"),
                    "step_count": len(steps) if isinstance(steps, list) else 0,
                    "is_template": p.get("is_template", False),
                    "category": p.get("category"),
                    "tags": list(p.get("tags") or []),
                    "is_locked": p.get("is_locked", False),
                    "run_context": definition.get("run_context") or "background",
                    "has_triggers": len(triggers) > 0,
                })
            formatted = f"Found {len(result)} playbook(s)."
            return tool_service_pb2.ListPlaybooksResponse(
                success=True,
                playbooks_json=json.dumps(result),
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("ListPlaybooks failed")
            return tool_service_pb2.ListPlaybooksResponse(success=False, error=str(e))

    async def ListAgentProfiles(
        self,
        request: tool_service_pb2.ListAgentProfilesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListAgentProfilesResponse:
        """List agent profiles for the user with derived status."""
        try:
            from services import agent_factory_service
            user_id = request.user_id or "system"
            profiles = await agent_factory_service.list_profiles(user_id)
            for p in profiles:
                is_active = p.get("is_active", True)
                last_status = p.get("last_execution_status")
                p["status"] = "draft" if (not is_active and not last_status) else "paused" if not is_active else ("error" if last_status == "failed" else "active")
            formatted = f"Found {len(profiles)} profile(s)."
            return tool_service_pb2.ListAgentProfilesResponse(
                success=True,
                profiles_json=json.dumps(profiles),
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("ListAgentProfiles failed")
            return tool_service_pb2.ListAgentProfilesResponse(success=False, error=str(e))

    async def ListAgentSchedules(
        self,
        request: tool_service_pb2.ListAgentSchedulesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListAgentSchedulesResponse:
        """List schedules for an agent profile (user must own the profile)."""
        try:
            from services.database_manager.database_helpers import fetch_all, fetch_one
            user_id = request.user_id or "system"
            ctx = _grpc_rls(user_id)
            agent_id = request.agent_id or ""
            if not agent_id:
                return tool_service_pb2.ListAgentSchedulesResponse(
                    success=False, error="agent_id required"
                )
            profile = await fetch_one(
                "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
                agent_id,
                user_id,
                rls_context=ctx,
            )
            if not profile:
                return tool_service_pb2.ListAgentSchedulesResponse(
                    success=False, error="Profile not found"
                )
            rows = await fetch_all(
                "SELECT * FROM agent_schedules WHERE agent_profile_id = $1 ORDER BY created_at",
                agent_id,
                rls_context=ctx,
            )
            result = []
            for r in rows:
                result.append({
                    "id": str(r["id"]),
                    "agent_profile_id": str(r["agent_profile_id"]),
                    "schedule_type": r.get("schedule_type"),
                    "cron_expression": r.get("cron_expression"),
                    "interval_seconds": r.get("interval_seconds"),
                    "timezone": r.get("timezone") or "UTC",
                    "is_active": r.get("is_active", True),
                    "next_run_at": r["next_run_at"].isoformat() if r.get("next_run_at") else None,
                    "last_run_at": r["last_run_at"].isoformat() if r.get("last_run_at") else None,
                    "last_status": r.get("last_status"),
                    "run_count": r.get("run_count", 0),
                })
            parts = [f"Found {len(result)} schedule(s) for agent {agent_id}:"]
            for s in result:
                stype = s.get("schedule_type", "?")
                active = "active" if s.get("is_active") else "paused"
                parts.append(f"  - {s['id']}: {stype} ({active})")
            return tool_service_pb2.ListAgentSchedulesResponse(
                success=True,
                schedules_json=json.dumps(result),
                formatted="\n".join(parts),
            )
        except Exception as e:
            logger.exception("ListAgentSchedules failed")
            return tool_service_pb2.ListAgentSchedulesResponse(success=False, error=str(e))

    async def ListAgentDataSources(
        self,
        request: tool_service_pb2.ListAgentDataSourcesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListAgentDataSourcesResponse:
        """List data source bindings for an agent profile (user must own the profile)."""
        try:
            from services.database_manager.database_helpers import fetch_all, fetch_one
            user_id = request.user_id or "system"
            ctx = _grpc_rls(user_id)
            agent_id = request.agent_id or ""
            if not agent_id:
                return tool_service_pb2.ListAgentDataSourcesResponse(
                    success=False, error="agent_id required"
                )
            profile = await fetch_one(
                "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
                agent_id,
                user_id,
                rls_context=ctx,
            )
            if not profile:
                return tool_service_pb2.ListAgentDataSourcesResponse(
                    success=False, error="Profile not found"
                )
            rows = await fetch_all(
                """
                SELECT ads.id AS binding_id, ads.connector_id, ads.config_overrides, ads.is_enabled,
                       dsc.name AS connector_name, dsc.connector_type, dsc.definition
                FROM agent_data_sources ads
                LEFT JOIN data_source_connectors dsc ON dsc.id = ads.connector_id
                WHERE ads.agent_profile_id = $1
                ORDER BY ads.created_at
                """,
                agent_id,
                rls_context=ctx,
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
                    "binding_id": str(r["binding_id"]),
                    "connector_id": str(r["connector_id"]),
                    "connector_name": r.get("connector_name", ""),
                    "connector_type": r.get("connector_type", "rest"),
                    "endpoint_count": endpoint_count,
                    "is_enabled": r.get("is_enabled", True),
                    "config_overrides": r.get("config_overrides") or {},
                })
            parts = [f"Found {len(result)} data source binding(s) for agent {agent_id}:"]
            for b in result:
                parts.append(f"  - {b['connector_name']} (connector: {b['connector_id']}, enabled: {b['is_enabled']})")
            return tool_service_pb2.ListAgentDataSourcesResponse(
                success=True,
                bindings_json=json.dumps(result),
                formatted="\n".join(parts),
            )
        except Exception as e:
            logger.exception("ListAgentDataSources failed")
            return tool_service_pb2.ListAgentDataSourcesResponse(success=False, error=str(e))

    async def CreateControlPane(
        self,
        request: tool_service_pb2.CreateControlPaneRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateControlPaneResponse:
        """Create a control pane wired to a data connector."""
        try:
            from services.database_manager.database_helpers import fetch_one, fetch_value
            user_id = request.user_id or "system"
            name = request.name or "Control Pane"
            icon = request.icon or "Tune"
            connector_id = request.connector_id or ""
            if not connector_id:
                return tool_service_pb2.CreateControlPaneResponse(
                    success=False, error="connector_id required"
                )
            conn = await fetch_one(
                "SELECT id FROM data_source_connectors WHERE id = $1 AND (user_id = $2 OR is_template = true)",
                connector_id,
                user_id,
            )
            if not conn:
                return tool_service_pb2.CreateControlPaneResponse(
                    success=False, error="Connector not found"
                )
            credentials = {}
            if request.credentials_encrypted_json:
                try:
                    credentials = json.loads(request.credentials_encrypted_json)
                except json.JSONDecodeError:
                    pass
            controls = []
            if request.controls_json:
                try:
                    controls = json.loads(request.controls_json)
                except json.JSONDecodeError:
                    pass
            if not isinstance(controls, list):
                controls = []
            refresh_interval = getattr(request, "refresh_interval", 0) or 0
            new_id = await fetch_value(
                """
                INSERT INTO user_control_panes
                (user_id, name, icon, connector_id, credentials_encrypted, connection_id, controls, is_visible, sort_order, refresh_interval)
                VALUES ($1, $2, $3, $4::uuid, $5::jsonb, $6, $7::jsonb, $8, $9, $10)
                RETURNING id
                """,
                user_id,
                name,
                icon,
                connector_id,
                json.dumps(credentials),
                request.connection_id if request.connection_id else None,
                json.dumps(controls),
                request.is_visible,
                request.sort_order,
                refresh_interval,
            )
            pane_id = str(new_id)
            formatted = f"Created control pane: {name} (ID: {pane_id})"
            try:
                from utils.websocket_manager import get_websocket_manager
                ws_manager = get_websocket_manager()
                if ws_manager:
                    await ws_manager.send_to_session(
                        {"type": "control_pane_updated", "subtype": "pane_created", "pane_id": pane_id},
                        user_id,
                    )
            except Exception as ws_err:
                logger.warning("CreateControlPane WebSocket send failed: %s", ws_err)
            return tool_service_pb2.CreateControlPaneResponse(
                success=True,
                pane_id=pane_id,
                name=name,
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("CreateControlPane failed")
            return tool_service_pb2.CreateControlPaneResponse(success=False, error=str(e))

    async def UpdateControlPane(
        self,
        request: tool_service_pb2.UpdateControlPaneRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateControlPaneResponse:
        """Update a control pane (partial update)."""
        try:
            from services.database_manager.database_helpers import fetch_one, execute
            user_id = request.user_id or "system"
            pane_id = request.pane_id or ""
            if not pane_id:
                return tool_service_pb2.UpdateControlPaneResponse(
                    success=False, error="pane_id required"
                )
            existing = await fetch_one(
                "SELECT id FROM user_control_panes WHERE id = $1 AND user_id = $2",
                pane_id,
                user_id,
            )
            if not existing:
                return tool_service_pb2.UpdateControlPaneResponse(
                    success=False, error="Control pane not found"
                )
            if request.HasField("connector_id") and request.connector_id:
                conn = await fetch_one(
                    "SELECT id FROM data_source_connectors WHERE id = $1 AND (user_id = $2 OR is_template = true)",
                    request.connector_id,
                    user_id,
                )
                if not conn:
                    return tool_service_pb2.UpdateControlPaneResponse(
                        success=False, error="Connector not found"
                    )
            updates = []
            args = []
            idx = 1
            if request.HasField("name"):
                updates.append(f"name = ${idx}")
                args.append(request.name)
                idx += 1
            if request.HasField("icon"):
                updates.append(f"icon = ${idx}")
                args.append(request.icon)
                idx += 1
            if request.HasField("connector_id"):
                updates.append(f"connector_id = ${idx}::uuid")
                args.append(request.connector_id)
                idx += 1
            if request.HasField("credentials_encrypted_json"):
                updates.append(f"credentials_encrypted = ${idx}::jsonb")
                args.append(request.credentials_encrypted_json)
                idx += 1
            if request.HasField("connection_id"):
                updates.append(f"connection_id = ${idx}")
                args.append(request.connection_id)
                idx += 1
            if request.HasField("controls_json"):
                updates.append(f"controls = ${idx}::jsonb")
                args.append(request.controls_json)
                idx += 1
            if request.HasField("is_visible"):
                updates.append(f"is_visible = ${idx}")
                args.append(request.is_visible)
                idx += 1
            if request.HasField("sort_order"):
                updates.append(f"sort_order = ${idx}")
                args.append(request.sort_order)
                idx += 1
            if request.HasField("refresh_interval"):
                updates.append(f"refresh_interval = ${idx}")
                args.append(request.refresh_interval)
                idx += 1
            if not updates:
                try:
                    from utils.websocket_manager import get_websocket_manager
                    ws_manager = get_websocket_manager()
                    if ws_manager:
                        await ws_manager.send_to_session(
                            {"type": "control_pane_updated", "subtype": "pane_updated", "pane_id": pane_id},
                            user_id,
                        )
                except Exception as ws_err:
                    logger.warning("UpdateControlPane WebSocket send failed: %s", ws_err)
                return tool_service_pb2.UpdateControlPaneResponse(
                    success=True, formatted="No updates applied"
                )
            updates.append("updated_at = NOW()")
            args.extend([pane_id, user_id])
            await execute(
                f"UPDATE user_control_panes SET {', '.join(updates)} WHERE id = ${idx}::uuid AND user_id = ${idx + 1}",
                *args,
            )
            try:
                from utils.websocket_manager import get_websocket_manager
                ws_manager = get_websocket_manager()
                if ws_manager:
                    await ws_manager.send_to_session(
                        {"type": "control_pane_updated", "subtype": "pane_updated", "pane_id": pane_id},
                        user_id,
                    )
            except Exception as ws_err:
                logger.warning("UpdateControlPane WebSocket send failed: %s", ws_err)
            return tool_service_pb2.UpdateControlPaneResponse(
                success=True,
                formatted=f"Updated control pane {pane_id}",
            )
        except Exception as e:
            logger.exception("UpdateControlPane failed")
            return tool_service_pb2.UpdateControlPaneResponse(success=False, error=str(e))

    async def DeleteControlPane(
        self,
        request: tool_service_pb2.DeleteControlPaneRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.DeleteControlPaneResponse:
        """Delete a control pane."""
        try:
            from services.database_manager.database_helpers import execute
            user_id = request.user_id or "system"
            pane_id = request.pane_id or ""
            if not pane_id:
                return tool_service_pb2.DeleteControlPaneResponse(
                    success=False, error="pane_id required"
                )
            await execute(
                "DELETE FROM user_control_panes WHERE id = $1::uuid AND user_id = $2",
                pane_id,
                user_id,
            )
            try:
                from utils.websocket_manager import get_websocket_manager
                ws_manager = get_websocket_manager()
                if ws_manager:
                    await ws_manager.send_to_session(
                        {"type": "control_pane_updated", "subtype": "pane_deleted", "pane_id": pane_id},
                        user_id,
                    )
            except Exception as ws_err:
                logger.warning("DeleteControlPane WebSocket send failed: %s", ws_err)
            return tool_service_pb2.DeleteControlPaneResponse(
                success=True,
                formatted=f"Deleted control pane {pane_id}",
            )
        except Exception as e:
            logger.exception("DeleteControlPane failed")
            return tool_service_pb2.DeleteControlPaneResponse(success=False, error=str(e))

    async def ExecuteControlPaneAction(
        self,
        request: tool_service_pb2.ExecuteControlPaneActionRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ExecuteControlPaneActionResponse:
        """Execute a connector endpoint through a saved control pane (same as REST execute)."""
        try:
            from services.database_manager.database_helpers import fetch_one
            from clients.connections_service_client import get_connections_service_client

            user_id = request.user_id or "system"
            pane_id = request.pane_id or ""
            endpoint_id = request.endpoint_id or ""
            if not pane_id or not endpoint_id:
                return tool_service_pb2.ExecuteControlPaneActionResponse(
                    success=False,
                    error="pane_id and endpoint_id required",
                )

            pane = await fetch_one(
                """
                SELECT id, pane_type, connector_id, credentials_encrypted, connection_id
                FROM user_control_panes WHERE id = $1 AND user_id = $2
                """,
                pane_id,
                user_id,
            )
            if not pane:
                return tool_service_pb2.ExecuteControlPaneActionResponse(
                    success=False,
                    error="Control pane not found",
                )

            if (pane.get("pane_type") or "connector") == "artifact":
                return tool_service_pb2.ExecuteControlPaneActionResponse(
                    success=False,
                    error="Artifact control panes do not support connector execution",
                )

            if not pane.get("connector_id"):
                return tool_service_pb2.ExecuteControlPaneActionResponse(
                    success=False,
                    error="Connector not configured for this pane",
                )

            connector = await fetch_one(
                "SELECT id, definition, connector_type FROM data_source_connectors WHERE id = $1",
                pane["connector_id"],
            )
            if not connector:
                return tool_service_pb2.ExecuteControlPaneActionResponse(
                    success=False,
                    error="Connector not found",
                )

            definition = connector.get("definition") or {}
            if isinstance(definition, str):
                try:
                    definition = json.loads(definition)
                except json.JSONDecodeError:
                    return tool_service_pb2.ExecuteControlPaneActionResponse(
                        success=False,
                        error="Invalid connector definition",
                    )

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
                    rls_context={"user_id": user_id},
                )
                if not oauth_token:
                    return tool_service_pb2.ExecuteControlPaneActionResponse(
                        success=False,
                        error="Could not obtain token for the selected connection",
                    )

            params = {}
            if request.params_json:
                try:
                    params = json.loads(request.params_json)
                except json.JSONDecodeError:
                    return tool_service_pb2.ExecuteControlPaneActionResponse(
                        success=False,
                        error="Invalid params_json",
                    )

            client = await get_connections_service_client()
            result = await client.execute_connector_endpoint(
                definition=definition,
                credentials=credentials,
                endpoint_id=endpoint_id,
                params=params,
                max_pages=1,
                oauth_token=oauth_token,
                raw_response=True,
                connector_type=connector.get("connector_type"),
            )

            if result.get("error"):
                return tool_service_pb2.ExecuteControlPaneActionResponse(
                    success=False,
                    raw_response_json="{}",
                    records_json="[]",
                    count=0,
                    formatted=result.get("error", ""),
                    error=result.get("error"),
                )

            records = result.get("records", [])
            raw_response = result.get("raw_response")
            formatted = result.get("formatted", "")

            return tool_service_pb2.ExecuteControlPaneActionResponse(
                success=True,
                raw_response_json=json.dumps(raw_response) if raw_response is not None else "{}",
                records_json=json.dumps(records),
                count=len(records),
                formatted=formatted,
            )
        except json.JSONDecodeError as e:
            return tool_service_pb2.ExecuteControlPaneActionResponse(
                success=False,
                error=f"Invalid JSON: {e}",
            )
        except Exception as e:
            logger.exception("ExecuteControlPaneAction failed")
            return tool_service_pb2.ExecuteControlPaneActionResponse(success=False, error=str(e))

