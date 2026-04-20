"""gRPC handlers for Agent Factory CRUD meta-tools."""

import json
import logging
import uuid

import grpc
from protos import tool_service_pb2
from utils.grpc_rls import grpc_user_rls as _grpc_rls

logger = logging.getLogger(__name__)


class AgentFactoryCrudHandlersMixin:
    """Mixin providing Agent Factory CRUD gRPC handlers.

    Mixed into ToolServiceImplementation; provides handlers for creating/updating/deleting
    agent profiles, playbooks, schedules, data source bindings, and LLM model listing.
    """

    # ===== Agent Factory meta-tools =====

    async def CreateAgentProfile(
        self,
        request: tool_service_pb2.CreateAgentProfileRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateAgentProfileResponse:
        """Create an agent profile via Agent Factory service."""
        try:
            from services import agent_factory_service
            user_id = request.user_id or "system"
            data = {
                "name": request.name or "",
                "handle": request.handle or "",
                "description": request.description if request.description else None,
                "model_preference": request.model_preference if request.model_preference else None,
                "system_prompt_additions": request.system_prompt_additions if request.system_prompt_additions else None,
                "persona_mode": "default" if (request.persona_enabled if request.HasField("persona_enabled") else False) else "none",
                "persona_id": None,
                "include_user_context": False,
                "include_user_facts": False,
                "include_facts_categories": [],
                "auto_routable": request.auto_routable if request.HasField("auto_routable") else False,
                "prompt_history_enabled": request.chat_history_enabled if request.HasField("chat_history_enabled") else False,
                "chat_visible": request.chat_visible if request.HasField("chat_visible") else True,
                "is_active": request.is_active if request.HasField("is_active") else False,
            }
            profile = await agent_factory_service.create_profile(user_id, data)
            h = profile.get("handle") or ""
            formatted = f"Created agent profile: {profile.get('name', '')} ({'@' + h if h else 'schedule/Run-only'}) — ID: {profile.get('id', '')}"
            try:
                from utils.websocket_manager import get_websocket_manager
                ws_manager = get_websocket_manager()
                if ws_manager:
                    await ws_manager.send_to_session(
                        {
                            "type": "agent_factory_updated",
                            "subtype": "profile_created",
                            "agent_id": profile.get("id", ""),
                        },
                        user_id,
                    )
            except Exception as ws_err:
                logger.warning("CreateAgentProfile WebSocket send failed: %s", ws_err)
            return tool_service_pb2.CreateAgentProfileResponse(
                success=True,
                agent_id=profile.get("id", ""),
                name=profile.get("name", ""),
                handle=profile.get("handle", ""),
                formatted=formatted,
            )
        except ValueError as e:
            return tool_service_pb2.CreateAgentProfileResponse(
                success=False, agent_id="", name="", handle="", formatted="", error=str(e)
            )
        except Exception as e:
            logger.exception("CreateAgentProfile failed")
            return tool_service_pb2.CreateAgentProfileResponse(
                success=False, agent_id="", name="", handle="", formatted="", error=str(e)
            )

    async def SetAgentProfileStatus(
        self,
        request: tool_service_pb2.SetAgentProfileStatusRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SetAgentProfileStatusResponse:
        """Update an agent profile's is_active (pause or activate). Separate capability from creating agents."""
        try:
            from services import agent_factory_service
            user_id = request.user_id or "system"
            await agent_factory_service.update_profile(
                user_id,
                request.agent_id,
                {"is_active": request.is_active},
            )
            status = "active" if request.is_active else "paused"
            formatted = f"Agent profile {request.agent_id} set to {status}."
            return tool_service_pb2.SetAgentProfileStatusResponse(
                success=True,
                is_active=request.is_active,
                formatted=formatted,
            )
        except ValueError as e:
            return tool_service_pb2.SetAgentProfileStatusResponse(
                success=False, is_active=False, formatted="", error=str(e)
            )
        except Exception as e:
            logger.exception("SetAgentProfileStatus failed")
            return tool_service_pb2.SetAgentProfileStatusResponse(
                success=False, is_active=False, formatted="", error=str(e)
            )

    async def CreatePlaybook(
        self,
        request: tool_service_pb2.CreatePlaybookRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreatePlaybookResponse:
        """Create a custom playbook via Agent Factory service."""
        try:
            from services import agent_factory_service
            user_id = request.user_id or "system"
            definition = {}
            if request.definition_json:
                try:
                    definition = json.loads(request.definition_json)
                except json.JSONDecodeError:
                    return tool_service_pb2.CreatePlaybookResponse(
                        success=False, playbook_id="", name="", step_count=0,
                        formatted="", error="Invalid definition_json"
                    )
            if request.run_context:
                definition["run_context"] = request.run_context
            data = {
                "name": request.name or "Unnamed",
                "description": request.description if request.description else None,
                "definition": definition,
                "category": request.category if request.category else None,
                "tags": list(request.tags) if request.tags else [],
            }
            warnings = agent_factory_service.validate_playbook_definition(definition)
            playbook = await agent_factory_service.create_playbook(user_id, data)
            steps = (playbook.get("definition") or {}).get("steps") or []
            step_count = len(steps) if isinstance(steps, list) else 0
            formatted = f"Created playbook: {playbook.get('name', '')} ({step_count} steps) — ID: {playbook.get('id', '')}"
            if warnings:
                formatted += "\nValidation warnings: " + "; ".join(warnings[:5])
            try:
                from utils.websocket_manager import get_websocket_manager
                ws_manager = get_websocket_manager()
                if ws_manager:
                    await ws_manager.send_to_session(
                        {"type": "agent_factory_updated", "subtype": "playbook_created", "playbook_id": playbook.get("id", "")},
                        user_id,
                    )
            except Exception as ws_err:
                logger.warning("CreatePlaybook WebSocket send failed: %s", ws_err)
            try:
                from utils.websocket_manager import get_websocket_manager
                ws = get_websocket_manager()
                if ws:
                    await ws.send_to_session(
                        {"type": "agent_factory_updated", "subtype": "playbook_created", "playbook_id": playbook.get("id", "")},
                        user_id,
                    )
            except Exception as ws_err:
                logger.warning("CreatePlaybook WebSocket send failed: %s", ws_err)
            return tool_service_pb2.CreatePlaybookResponse(
                success=True,
                playbook_id=playbook.get("id", ""),
                name=playbook.get("name", ""),
                step_count=step_count,
                validation_warnings=warnings,
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("CreatePlaybook failed")
            return tool_service_pb2.CreatePlaybookResponse(
                success=False, playbook_id="", name="", step_count=0,
                formatted="", error=str(e)
            )

    async def AssignPlaybookToAgent(
        self,
        request: tool_service_pb2.AssignPlaybookToAgentRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.AssignPlaybookToAgentResponse:
        """Assign a playbook to an agent profile (set default_playbook_id)."""
        try:
            from services import agent_factory_service
            user_id = request.user_id or "system"
            await agent_factory_service.update_profile(
                user_id,
                request.agent_id,
                {"default_playbook_id": request.playbook_id},
            )
            formatted = f"Assigned playbook {request.playbook_id} to agent {request.agent_id}."
            return tool_service_pb2.AssignPlaybookToAgentResponse(
                success=True,
                formatted=formatted,
            )
        except ValueError as e:
            return tool_service_pb2.AssignPlaybookToAgentResponse(
                success=False, formatted="", error=str(e)
            )
        except Exception as e:
            logger.exception("AssignPlaybookToAgent failed")
            return tool_service_pb2.AssignPlaybookToAgentResponse(
                success=False, formatted="", error=str(e)
            )

    async def CreateAgentSchedule(
        self,
        request: tool_service_pb2.CreateAgentScheduleRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateAgentScheduleResponse:
        """Create a schedule for an agent profile."""
        try:
            from services import agent_factory_service
            user_id = request.user_id or "system"
            data = {
                "schedule_type": request.schedule_type or "cron",
                "cron_expression": request.cron_expression if request.cron_expression else None,
                "interval_seconds": request.interval_seconds if request.interval_seconds else None,
                "timezone": request.timezone if request.timezone else "UTC",
                "is_active": request.is_active if request.HasField("is_active") else False,
                "input_context": {},
            }
            if request.input_context_json:
                try:
                    data["input_context"] = json.loads(request.input_context_json)
                except json.JSONDecodeError:
                    pass
            schedule = await agent_factory_service.create_schedule(
                user_id,
                request.agent_id,
                data,
            )
            next_run = schedule.get("next_run_at") or ""
            is_active = schedule.get("is_active", False)
            formatted = f"Created schedule for agent {request.agent_id} — next run: {next_run}, active: {is_active}"
            return tool_service_pb2.CreateAgentScheduleResponse(
                success=True,
                schedule_id=schedule.get("id", ""),
                next_run_at=next_run,
                is_active=is_active,
                formatted=formatted,
            )
        except ValueError as e:
            return tool_service_pb2.CreateAgentScheduleResponse(
                success=False, schedule_id="", next_run_at="", is_active=False,
                formatted="", error=str(e)
            )
        except Exception as e:
            logger.exception("CreateAgentSchedule failed")
            return tool_service_pb2.CreateAgentScheduleResponse(
                success=False, schedule_id="", next_run_at="", is_active=False,
                formatted="", error=str(e)
            )

    async def BindDataSourceToAgent(
        self,
        request: tool_service_pb2.BindDataSourceToAgentRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.BindDataSourceToAgentResponse:
        """Bind a data source connector to an agent profile."""
        try:
            from services import agent_factory_service
            user_id = request.user_id or "system"
            data = {
                "connector_id": request.connector_id,
                "config_overrides": {},
                "permissions": {},
                "is_enabled": True,
            }
            if request.config_overrides_json:
                try:
                    data["config_overrides"] = json.loads(request.config_overrides_json)
                except json.JSONDecodeError:
                    pass
            if request.permissions_json:
                try:
                    data["permissions"] = json.loads(request.permissions_json)
                except json.JSONDecodeError:
                    pass
            binding = await agent_factory_service.create_data_source_binding(
                user_id,
                request.agent_id,
                data,
            )
            formatted = f"Bound connector {request.connector_id} to agent {request.agent_id} — binding ID: {binding.get('id', '')}"
            return tool_service_pb2.BindDataSourceToAgentResponse(
                success=True,
                binding_id=binding.get("id", ""),
                formatted=formatted,
            )
        except ValueError as e:
            return tool_service_pb2.BindDataSourceToAgentResponse(
                success=False, binding_id="", formatted="", error=str(e)
            )
        except Exception as e:
            logger.exception("BindDataSourceToAgent failed")
            return tool_service_pb2.BindDataSourceToAgentResponse(
                success=False, binding_id="", formatted="", error=str(e)
            )

    async def ListAvailableLlmModels(
        self,
        request: tool_service_pb2.ListAvailableLlmModelsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListAvailableLlmModelsResponse:
        """Return the list of LLM models available to the user (for Agent Factory model_preference)."""
        try:
            from services.model_source_resolver import get_available_models as resolver_get_available_models
            user_id = request.user_id or "system"
            models = await resolver_get_available_models(user_id)
            proto_models = []
            for m in models:
                mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None) or ""
                name = getattr(m, "name", None) or (m.get("name") if isinstance(m, dict) else None) or mid
                prov = getattr(m, "provider", None) or (m.get("provider") if isinstance(m, dict) else "") or ""
                proto_models.append(
                    tool_service_pb2.LlmModelInfo(
                        model_id=str(mid),
                        display_name=str(name),
                        provider=str(prov),
                    )
                )
            lines = [f"Available models ({len(models)}):"] if models else ["No models configured for this user."]
            for m in models:
                mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
                name = getattr(m, "name", None) or (m.get("name") if isinstance(m, dict) else None)
                prov = getattr(m, "provider", None) or (m.get("provider") if isinstance(m, dict) else "")
                lines.append(f"- {mid} ({name or mid}) [{prov}]")
            formatted = "\n".join(lines)
            return tool_service_pb2.ListAvailableLlmModelsResponse(
                success=True,
                models=proto_models,
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("ListAvailableLlmModels failed")
            return tool_service_pb2.ListAvailableLlmModelsResponse(
                success=False,
                models=[],
                formatted="",
                error=str(e),
            )

    async def UpdateAgentProfile(
        self,
        request: tool_service_pb2.UpdateAgentProfileRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateAgentProfileResponse:
        """Update an agent profile. Lock enforced in service (only is_active/is_locked when locked)."""
        try:
            from services import agent_factory_service
            from utils.websocket_manager import get_websocket_manager

            user_id = request.user_id or "system"
            agent_id = (request.agent_id or "").strip()
            if not agent_id:
                return tool_service_pb2.UpdateAgentProfileResponse(
                    success=False, agent_id="", name="", formatted="", error="agent_id required"
                )
            updates = {}
            if request.updates_json:
                try:
                    updates = json.loads(request.updates_json)
                except json.JSONDecodeError:
                    return tool_service_pb2.UpdateAgentProfileResponse(
                        success=False, agent_id=agent_id, name="", formatted="", error="Invalid updates_json"
                    )
            if not isinstance(updates, dict):
                return tool_service_pb2.UpdateAgentProfileResponse(
                    success=False, agent_id=agent_id, name="", formatted="", error="updates_json must be a JSON object"
                )
            profile = await agent_factory_service.update_profile(user_id, agent_id, updates)
            formatted = f"Updated agent profile: {profile.get('name', '')} (ID: {agent_id})"
            try:
                ws_manager = get_websocket_manager()
                if ws_manager:
                    await ws_manager.send_to_session(
                        {"type": "agent_factory_updated", "subtype": "profile_updated", "agent_id": agent_id},
                        user_id,
                    )
            except Exception as ws_err:
                logger.warning("UpdateAgentProfile WebSocket send failed: %s", ws_err)
            return tool_service_pb2.UpdateAgentProfileResponse(
                success=True,
                agent_id=agent_id,
                name=profile.get("name", ""),
                formatted=formatted,
            )
        except ValueError as e:
            return tool_service_pb2.UpdateAgentProfileResponse(
                success=False, agent_id=request.agent_id or "", name="", formatted="", error=str(e)
            )
        except Exception as e:
            logger.exception("UpdateAgentProfile failed")
            return tool_service_pb2.UpdateAgentProfileResponse(
                success=False, agent_id=request.agent_id or "", name="", formatted="", error=str(e)
            )

    async def DeleteAgentProfile(
        self,
        request: tool_service_pb2.DeleteAgentProfileRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.DeleteAgentProfileResponse:
        """Delete an agent profile. Blocked when profile is locked."""
        try:
            from services.database_manager.database_helpers import fetch_one, execute
            from utils.websocket_manager import get_websocket_manager

            user_id = request.user_id or "system"
            ctx = _grpc_rls(user_id)
            agent_id = (request.agent_id or "").strip()
            if not agent_id:
                return tool_service_pb2.DeleteAgentProfileResponse(
                    success=False, formatted="", error="agent_id required"
                )
            row = await fetch_one(
                "SELECT id, is_locked FROM agent_profiles WHERE id = $1 AND user_id = $2",
                agent_id,
                user_id,
                rls_context=ctx,
            )
            if not row:
                return tool_service_pb2.DeleteAgentProfileResponse(
                    success=False, formatted="", error="Profile not found"
                )
            if row.get("is_locked"):
                return tool_service_pb2.DeleteAgentProfileResponse(
                    success=False, formatted="", error="Profile is locked; unlock to delete"
                )
            await execute("DELETE FROM agent_profiles WHERE id = $1 AND user_id = $2", agent_id, user_id, rls_context=ctx)
            formatted = f"Deleted agent profile {agent_id}."
            try:
                ws_manager = get_websocket_manager()
                if ws_manager:
                    await ws_manager.send_to_session(
                        {"type": "agent_factory_updated", "subtype": "profile_deleted", "agent_id": agent_id},
                        user_id,
                    )
            except Exception as ws_err:
                logger.warning("DeleteAgentProfile WebSocket send failed: %s", ws_err)
            return tool_service_pb2.DeleteAgentProfileResponse(success=True, formatted=formatted)
        except Exception as e:
            logger.exception("DeleteAgentProfile failed")
            return tool_service_pb2.DeleteAgentProfileResponse(success=False, formatted="", error=str(e))

    async def UpdatePlaybook(
        self,
        request: tool_service_pb2.UpdatePlaybookRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdatePlaybookResponse:
        """Update a playbook. Lock: only is_locked toggle allowed when locked. Templates are read-only."""
        try:
            import uuid
            from services import agent_factory_service
            from services.database_manager.database_helpers import fetch_one, execute
            from utils.websocket_manager import get_websocket_manager

            user_id = request.user_id or "system"
            ctx = _grpc_rls(user_id)
            playbook_id = (request.playbook_id or "").strip()
            if not playbook_id:
                return tool_service_pb2.UpdatePlaybookResponse(
                    success=False, playbook_id="", name="", step_count=0,
                    validation_warnings=[], formatted="", error="playbook_id required"
                )
            # If identifier is not a UUID, resolve by playbook name or slug (e.g. "morning-intelligence-briefing")
            try:
                uuid.UUID(playbook_id)
            except (ValueError, TypeError):
                slug_normalized = playbook_id.replace("_", "-").lower()
                resolved = await fetch_one(
                    """SELECT id FROM custom_playbooks
                       WHERE user_id = $1 AND (
                         name = $2
                         OR LOWER(REGEXP_REPLACE(TRIM(name), '\\s+', '-')) = $3
                       )
                       LIMIT 1""",
                    user_id,
                    playbook_id,
                    slug_normalized,
                    rls_context=ctx,
                )
                if not resolved:
                    return tool_service_pb2.UpdatePlaybookResponse(
                        success=False, playbook_id=playbook_id, name="", step_count=0,
                        validation_warnings=[], formatted="", error="Playbook not found"
                    )
                playbook_id = str(resolved["id"])
            row = await fetch_one(
                "SELECT * FROM custom_playbooks WHERE id = $1 AND user_id = $2",
                playbook_id,
                user_id,
                rls_context=ctx,
            )
            if not row:
                return tool_service_pb2.UpdatePlaybookResponse(
                    success=False, playbook_id=playbook_id, name="", step_count=0,
                    validation_warnings=[], formatted="", error="Playbook not found"
                )
            if row.get("is_template"):
                return tool_service_pb2.UpdatePlaybookResponse(
                    success=False, playbook_id=playbook_id, name="", step_count=0,
                    validation_warnings=[], formatted="", error="Cannot update template playbook"
                )
            updates = {}
            if request.updates_json:
                try:
                    updates = json.loads(request.updates_json)
                except json.JSONDecodeError:
                    return tool_service_pb2.UpdatePlaybookResponse(
                        success=False, playbook_id=playbook_id, name="", step_count=0,
                        validation_warnings=[], formatted="", error="Invalid updates_json"
                    )
            if not isinstance(updates, dict):
                return tool_service_pb2.UpdatePlaybookResponse(
                    success=False, playbook_id=playbook_id, name="", step_count=0,
                    validation_warnings=[], formatted="", error="updates_json must be a JSON object"
                )
            allowed_keys = {"name", "description", "version", "definition", "triggers", "is_template", "category", "tags", "required_connectors", "is_locked"}
            updates = {k: v for k, v in updates.items() if k in allowed_keys}
            if not updates:
                pb = agent_factory_service._row_to_playbook(row)
                step_count = len((pb.get("definition") or {}).get("steps") or [])
                return tool_service_pb2.UpdatePlaybookResponse(
                    success=True,
                    playbook_id=playbook_id,
                    name=pb.get("name", ""),
                    step_count=step_count,
                    formatted=f"Playbook unchanged: {pb.get('name', '')} (ID: {playbook_id})",
                )
            if row.get("is_locked") and set(updates.keys()) != {"is_locked"}:
                return tool_service_pb2.UpdatePlaybookResponse(
                    success=False, playbook_id=playbook_id, name="", step_count=0,
                    validation_warnings=[], formatted="", error="Playbook is locked; only lock toggle is allowed"
                )
            playbook_remediation_msgs: list = []
            playbook_remediation_steps: list = []
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
                    agent_factory_service.merge_playbook_definition_steps(old_def, defn)
                if isinstance(defn, dict):
                    defn, playbook_remediation_steps, playbook_remediation_msgs = (
                        await agent_factory_service.validate_and_remediate_playbook_models_for_user(
                            user_id, defn
                        )
                    )
                    updates["definition"] = defn
            warnings = []
            if "definition" in updates:
                warnings = agent_factory_service.validate_playbook_definition(updates.get("definition") or {})
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
            args.extend([playbook_id, user_id])
            await execute(
                f"UPDATE custom_playbooks SET {', '.join(set_clauses)} WHERE id = ${idx}::uuid AND user_id = ${idx + 1}",
                *args,
                rls_context=ctx,
            )
            row = await fetch_one("SELECT * FROM custom_playbooks WHERE id = $1", playbook_id, rls_context=ctx)
            pb = agent_factory_service._row_to_playbook(row)
            step_count = len((pb.get("definition") or {}).get("steps") or [])
            formatted = f"Updated playbook: {pb.get('name', '')} ({step_count} steps) — ID: {playbook_id}"
            if warnings:
                formatted += "\nValidation warnings: " + "; ".join(warnings[:5])
            if playbook_remediation_msgs:
                await agent_factory_service.notify_playbook_model_remediation(
                    user_id, playbook_id, playbook_remediation_steps, playbook_remediation_msgs
                )
            try:
                ws_manager = get_websocket_manager()
                if ws_manager:
                    await ws_manager.send_to_session(
                        {"type": "agent_factory_updated", "subtype": "playbook_updated", "playbook_id": playbook_id},
                        user_id,
                    )
            except Exception as ws_err:
                logger.warning("UpdatePlaybook WebSocket send failed: %s", ws_err)
            return tool_service_pb2.UpdatePlaybookResponse(
                success=True,
                playbook_id=playbook_id,
                name=pb.get("name", ""),
                step_count=step_count,
                validation_warnings=warnings[:10],
                formatted=formatted,
            )
        except Exception as e:
            logger.exception("UpdatePlaybook failed")
            return tool_service_pb2.UpdatePlaybookResponse(
                success=False, playbook_id=request.playbook_id or "", name="", step_count=0,
                validation_warnings=[], formatted="", error=str(e)
            )

    async def DeletePlaybook(
        self,
        request: tool_service_pb2.DeletePlaybookRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.DeletePlaybookResponse:
        """Delete a playbook. Blocked when locked or when playbook is a template."""
        try:
            import uuid
            from services.database_manager.database_helpers import fetch_one, execute
            from utils.websocket_manager import get_websocket_manager

            user_id = request.user_id or "system"
            ctx = _grpc_rls(user_id)
            playbook_id = (request.playbook_id or "").strip()
            if not playbook_id:
                return tool_service_pb2.DeletePlaybookResponse(success=False, formatted="", error="playbook_id required")
            # If identifier is not a UUID, resolve by playbook name or slug
            try:
                uuid.UUID(playbook_id)
            except (ValueError, TypeError):
                slug_normalized = playbook_id.replace("_", "-").lower()
                resolved = await fetch_one(
                    """SELECT id FROM custom_playbooks
                       WHERE user_id = $1 AND (
                         name = $2
                         OR LOWER(REGEXP_REPLACE(TRIM(name), '\\s+', '-')) = $3
                       )
                       LIMIT 1""",
                    user_id,
                    playbook_id,
                    slug_normalized,
                    rls_context=ctx,
                )
                if not resolved:
                    return tool_service_pb2.DeletePlaybookResponse(
                        success=False, formatted="", error="Playbook not found"
                    )
                playbook_id = str(resolved["id"])
            row = await fetch_one(
                "SELECT id, is_template, is_locked FROM custom_playbooks WHERE id = $1 AND user_id = $2",
                playbook_id,
                user_id,
                rls_context=ctx,
            )
            if not row:
                return tool_service_pb2.DeletePlaybookResponse(
                    success=False, formatted="", error="Playbook not found"
                )
            if row.get("is_template"):
                return tool_service_pb2.DeletePlaybookResponse(
                    success=False, formatted="", error="Cannot delete template playbook"
                )
            if row.get("is_locked"):
                return tool_service_pb2.DeletePlaybookResponse(
                    success=False, formatted="", error="Playbook is locked; unlock to delete"
                )
            await execute("DELETE FROM custom_playbooks WHERE id = $1 AND user_id = $2", playbook_id, user_id, rls_context=ctx)
            formatted = f"Deleted playbook {playbook_id}."
            try:
                ws_manager = get_websocket_manager()
                if ws_manager:
                    await ws_manager.send_to_session(
                        {"type": "agent_factory_updated", "subtype": "playbook_deleted", "playbook_id": playbook_id},
                        user_id,
                    )
            except Exception as ws_err:
                logger.warning("DeletePlaybook WebSocket send failed: %s", ws_err)
            return tool_service_pb2.DeletePlaybookResponse(success=True, formatted=formatted)
        except Exception as e:
            logger.exception("DeletePlaybook failed")
            return tool_service_pb2.DeletePlaybookResponse(success=False, formatted="", error=str(e))
