"""gRPC handlers for Agent Runtime operations (execution, memory, approvals)."""

import json
import logging
import uuid
from datetime import datetime, timezone

import grpc
from protos import tool_service_pb2
from services.grpc_handlers._utils import json_default
from utils.grpc_rls import grpc_user_rls as _grpc_rls

logger = logging.getLogger(__name__)


class AgentRuntimeHandlersMixin:
    """Mixin providing Agent Runtime gRPC handlers.

    Mixed into ToolServiceImplementation; provides handlers for execution logging,
    approval parking, agent memory CRUD, and run history.
    """

    async def LogAgentExecution(
        self,
        request: tool_service_pb2.LogAgentExecutionRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.LogAgentExecutionResponse:
        """Insert a row into agent_execution_log for custom agent run; update agent_budgets spend."""
        try:
            from decimal import Decimal
            from services.database_manager.database_helpers import fetch_value, execute, fetch_one
            user_id = request.user_id or "system"
            ctx = _grpc_rls(user_id)
            profile_id = request.profile_id or None
            playbook_id = request.playbook_id or None
            if not profile_id:
                return tool_service_pb2.LogAgentExecutionResponse(
                    success=False, execution_id="", error="profile_id required"
                )
            _now = datetime.now(timezone.utc)
            started_at = request.started_at or _now.isoformat()
            completed_at = request.completed_at or _now.isoformat()
            if isinstance(started_at, str):
                started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            if isinstance(completed_at, str):
                completed_at = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            metadata = {}
            if request.metadata_json:
                try:
                    metadata = json.loads(request.metadata_json)
                except json.JSONDecodeError:
                    pass
            metadata["steps_completed"] = request.steps_completed or 0
            metadata["steps_total"] = request.steps_total or 0

            steps_data = []
            if getattr(request, "steps_json", None):
                try:
                    steps_data = json.loads(request.steps_json)
                except (json.JSONDecodeError, TypeError):
                    pass
            tokens_input = 0
            tokens_output = 0
            for s in (steps_data if isinstance(steps_data, list) else []):
                if isinstance(s, dict):
                    tokens_input += int(s.get("input_tokens") or 0)
                    tokens_output += int(s.get("output_tokens") or 0)

            model_used = (metadata.get("model_used") or "")[:255] if metadata.get("model_used") else None
            cost_usd = Decimal("0")
            if model_used and (tokens_input or tokens_output):
                try:
                    from services.service_container import get_service_container
                    container = await get_service_container()
                    if container.chat_service and hasattr(container.chat_service, "get_available_models"):
                        models = await container.chat_service.get_available_models()
                        for m in (models or []):
                            mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
                            if mid == model_used:
                                inc = (getattr(m, "input_cost", None) or (m.get("input_cost") if isinstance(m, dict) else None)) or 0
                                outc = (getattr(m, "output_cost", None) or (m.get("output_cost") if isinstance(m, dict) else None)) or 0
                                cost_usd = Decimal(str(inc)) * tokens_input + Decimal(str(outc)) * tokens_output
                                break
                        if cost_usd == Decimal("0"):
                            logger.warning(
                                "Cost lookup failed: model_used='%s' not found in %d available models",
                                model_used, len(models or []),
                            )
                except Exception as cost_err:
                    logger.debug("Resolve execution cost failed: %s", cost_err)

            execution_id = await fetch_value(
                """
                INSERT INTO agent_execution_log (
                    agent_profile_id, user_id, query, playbook_id,
                    started_at, completed_at, duration_ms, status,
                    error_details, metadata, tokens_input, tokens_output, cost_usd, model_used
                ) VALUES ($1, $2, $3, $4, $5::timestamptz, $6::timestamptz, $7, $8, $9, $10, $11, $12, $13::numeric, $14)
                RETURNING id
                """,
                uuid.UUID(profile_id) if profile_id else None,
                user_id,
                request.query or "",
                uuid.UUID(playbook_id) if playbook_id else None,
                started_at,
                completed_at,
                request.duration_ms or 0,
                request.status or "completed",
                request.error_details or None,
                json.dumps(metadata),
                tokens_input,
                tokens_output,
                float(cost_usd),
                model_used,
                rls_context=ctx,
            )
            exec_uuid = uuid.UUID(str(execution_id)) if execution_id else None
            if execution_id and isinstance(steps_data, list) and steps_data:
                from services.database_manager.database_helpers import execute
                for s in steps_data:
                    if not isinstance(s, dict):
                        continue
                    try:
                        started_ts = s.get("started_at")
                        completed_ts = s.get("completed_at")
                        if isinstance(started_ts, str):
                            started_ts = datetime.fromisoformat(started_ts.replace("Z", "+00:00"))
                        if isinstance(completed_ts, str):
                            completed_ts = datetime.fromisoformat(completed_ts.replace("Z", "+00:00"))
                        await execute(
                            """
                            INSERT INTO agent_execution_steps (
                                execution_id, step_index, step_name, step_type, action_name,
                                status, started_at, completed_at, duration_ms,
                                inputs_json, outputs_json, error_details, tool_call_trace
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7::timestamptz, $8::timestamptz, $9, $10::jsonb, $11::jsonb, $12, $13::jsonb)
                            """,
                            exec_uuid,
                            int(s.get("step_index", 0)),
                            (s.get("step_name") or "")[:255],
                            (s.get("step_type") or "tool")[:50],
                            (s.get("action_name") or "")[:255] if s.get("action_name") else None,
                            (s.get("status") or "completed")[:50],
                            started_ts if started_ts else None,
                            completed_ts if completed_ts else None,
                            s.get("duration_ms"),
                            json.dumps(s.get("inputs_snapshot") or {}),
                            json.dumps(s.get("outputs_snapshot") or {}),
                            (s.get("error_details") or "")[:65535] if s.get("error_details") else None,
                            json.dumps(s.get("tool_call_trace") if s.get("tool_call_trace") is not None else []),
                            rls_context=ctx,
                        )
                    except Exception as step_err:
                        logger.warning("Insert agent_execution_step failed: %s", step_err)

            discoveries = metadata.get("discoveries")
            if execution_id and isinstance(discoveries, list) and discoveries:
                from services.database_manager.database_helpers import execute
                for d in discoveries:
                    if not isinstance(d, dict):
                        continue
                    try:
                        await execute(
                            """
                            INSERT INTO agent_discoveries (
                                execution_id, user_id, discovery_type, entity_name, entity_type,
                                entity_neo4j_id, relationship_type, related_entity_name,
                                source_connector, source_endpoint, confidence, details
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb)
                            """,
                            exec_uuid,
                            user_id,
                            (d.get("discovery_type") or "entity")[:50],
                            (d.get("entity_name") or "")[:500] if d.get("entity_name") else None,
                            (d.get("entity_type") or "")[:50] if d.get("entity_type") else None,
                            (d.get("entity_neo4j_id") or "")[:255] if d.get("entity_neo4j_id") else None,
                            (d.get("relationship_type") or "")[:100] if d.get("relationship_type") else None,
                            (d.get("related_entity_name") or "")[:500] if d.get("related_entity_name") else None,
                            (d.get("source_connector") or "")[:255] if d.get("source_connector") else None,
                            (d.get("source_endpoint") or "")[:255] if d.get("source_endpoint") else None,
                            float(d["confidence"]) if d.get("confidence") is not None else None,
                            json.dumps(d.get("details") or {}),
                            rls_context=ctx,
                        )
                    except Exception as ins_err:
                        logger.warning("Insert agent_discovery failed: %s", ins_err)

            if execution_id and profile_id and float(cost_usd) > 0:
                try:
                    from datetime import date
                    today = date.today()
                    period_start = today.replace(day=1)
                    row = await fetch_one(
                        "SELECT id, current_period_start, current_period_spend_usd FROM agent_budgets WHERE agent_profile_id = $1",
                        uuid.UUID(profile_id),
                        rls_context=ctx,
                    )
                    if row:
                        existing_start = row.get("current_period_start")
                        if existing_start and (getattr(existing_start, "year", None) != today.year or getattr(existing_start, "month", None) != today.month):
                            await execute(
                                "UPDATE agent_budgets SET current_period_start = $1, current_period_spend_usd = 0, updated_at = NOW() WHERE agent_profile_id = $2",
                                period_start,
                                uuid.UUID(profile_id),
                                rls_context=ctx,
                            )
                        await execute(
                            "UPDATE agent_budgets SET current_period_spend_usd = current_period_spend_usd + $1::numeric, updated_at = NOW() WHERE agent_profile_id = $2",
                            float(cost_usd),
                            uuid.UUID(profile_id),
                            rls_context=ctx,
                        )
                        budget_after = await fetch_one(
                            "SELECT monthly_limit_usd, current_period_spend_usd, warning_threshold_pct, enforce_hard_limit FROM agent_budgets WHERE agent_profile_id = $1",
                            uuid.UUID(profile_id),
                            rls_context=ctx,
                        )
                        if budget_after and budget_after.get("monthly_limit_usd") is not None:
                            limit_usd = float(budget_after["monthly_limit_usd"])
                            spend_usd = float(budget_after.get("current_period_spend_usd") or 0)
                            pct = int(budget_after.get("warning_threshold_pct") or 80)
                            enforce = bool(budget_after.get("enforce_hard_limit") is not False)
                            try:
                                from services.notification_router import route_notification

                                if user_id:
                                    if enforce and spend_usd >= limit_usd:
                                        await route_notification(
                                            user_id,
                                            "budget_exceeded",
                                            {
                                                "type": "agent_notification",
                                                "subtype": "budget_exceeded",
                                                "agent_profile_id": profile_id,
                                                "agent_name": None,
                                                "title": "Budget exceeded",
                                                "preview": f"Spend ${spend_usd:.2f} of ${limit_usd:.2f} limit",
                                                "spend_usd": spend_usd,
                                                "limit_usd": limit_usd,
                                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                            },
                                            originating_surface_id=None,
                                        )
                                    elif spend_usd >= limit_usd * (pct / 100.0):
                                        await route_notification(
                                            user_id,
                                            "budget_warning",
                                            {
                                                "type": "agent_notification",
                                                "subtype": "budget_warning",
                                                "agent_profile_id": profile_id,
                                                "agent_name": None,
                                                "title": "Budget warning",
                                                "preview": f"Spend ${spend_usd:.2f} approaching ${limit_usd:.2f} limit",
                                                "spend_usd": spend_usd,
                                                "limit_usd": limit_usd,
                                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                            },
                                            originating_surface_id=None,
                                        )
                            except Exception as ws_err:
                                logger.debug("Budget WebSocket notify failed: %s", ws_err)
                except Exception as budget_err:
                    logger.warning("Update agent_budgets spend failed: %s", budget_err)

            if execution_id and user_id:
                try:
                    from services.notification_router import route_notification

                    profile_row = await fetch_one(
                        "SELECT name, handle FROM agent_profiles WHERE id = $1",
                        uuid.UUID(profile_id),
                        rls_context=ctx,
                    )
                    agent_name = (profile_row.get("name") or profile_row.get("handle") or "Agent") if profile_row else "Agent"
                    subtype = "execution_completed" if request.status == "completed" else "execution_failed"
                    qprev = (request.query or "")[:200] if request.query else ""
                    await route_notification(
                        user_id,
                        subtype,
                        {
                            "type": "agent_notification",
                            "subtype": subtype,
                            "execution_id": str(execution_id),
                            "agent_profile_id": profile_id,
                            "agent_name": agent_name,
                            "title": f"{agent_name}: {subtype.replace('_', ' ')}",
                            "preview": qprev or (request.error_details or "")[:200] or "",
                            "status": request.status or "completed",
                            "duration_ms": request.duration_ms,
                            "cost_usd": float(cost_usd) if cost_usd is not None else None,
                            "error_details": (request.error_details or "")[:500] if request.error_details else None,
                            "trigger_type": metadata.get("trigger_type", "manual"),
                            "query": qprev or None,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                        originating_surface_id=None,
                    )
                except Exception as ws_err:
                    logger.debug("LogAgentExecution WebSocket notify failed: %s", ws_err)

            return tool_service_pb2.LogAgentExecutionResponse(
                success=True,
                execution_id=str(execution_id) if execution_id else "",
            )
        except Exception as e:
            logger.exception("LogAgentExecution failed")
            return tool_service_pb2.LogAgentExecutionResponse(
                success=False, execution_id="", error=str(e)
            )

    async def ParkApproval(
        self,
        request: tool_service_pb2.ParkApprovalRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ParkApprovalResponse:
        """Insert a row into agent_approval_queue for background/scheduled approval. Called from orchestrator."""
        try:
            from services.database_manager.database_helpers import execute, fetch_value
            user_id = request.user_id or "system"
            ctx = _grpc_rls(user_id)
            agent_profile_id = request.agent_profile_id or None
            execution_id = request.execution_id or None
            step_name = (request.step_name or "approval")[:255]
            prompt = request.prompt or "Approve to continue?"
            preview_data_json = request.preview_data_json or "{}"
            thread_id = (request.thread_id or "")[:500]
            checkpoint_ns = (request.checkpoint_ns or "")[:255]
            playbook_config_json = request.playbook_config_json or "{}"
            governance_type = (request.governance_type or "playbook_step")[:50]
            if not user_id or not step_name:
                return tool_service_pb2.ParkApprovalResponse(
                    success=False, approval_id="", error="user_id and step_name required"
                )
            profile_uuid = uuid.UUID(agent_profile_id) if agent_profile_id else None
            exec_uuid = uuid.UUID(execution_id) if execution_id else None
            approval_id = await fetch_value(
                """INSERT INTO agent_approval_queue
                   (user_id, agent_profile_id, execution_id, step_name, prompt, preview_data, governance_type, thread_id, checkpoint_ns, playbook_config, status)
                   VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10::jsonb, 'pending')
                   RETURNING id""",
                user_id,
                profile_uuid,
                exec_uuid,
                step_name,
                prompt[:10000],
                preview_data_json,
                governance_type,
                thread_id,
                checkpoint_ns,
                playbook_config_json,
                rls_context=ctx,
            )
            if not approval_id:
                return tool_service_pb2.ParkApprovalResponse(
                    success=False, approval_id="", error="insert failed"
                )
            try:
                if governance_type == "shell_command_approval" and user_id:
                    from services.celery_tasks.agent_tasks import notify_shell_approval

                    notify_shell_approval.delay(
                        user_id,
                        str(approval_id),
                        preview_data_json or "{}",
                        (prompt or "")[:10000],
                        step_name,
                        agent_profile_id or "",
                        execution_id or "",
                    )
                else:
                    from services.notification_router import route_notification

                    if user_id:
                        pr = (prompt or "")[:500]
                        await route_notification(
                            user_id,
                            "approval_required",
                            {
                                "type": "agent_notification",
                                "subtype": "approval_required",
                                "approval_id": str(approval_id),
                                "agent_profile_id": agent_profile_id,
                                "execution_id": execution_id,
                                "step_name": step_name,
                                "prompt": pr,
                                "title": "Approval required",
                                "preview": pr[:200],
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                            originating_surface_id=None,
                        )
            except Exception as ws_err:
                logger.debug("ParkApproval WebSocket notify failed: %s", ws_err)
            return tool_service_pb2.ParkApprovalResponse(
                success=True,
                approval_id=str(approval_id),
            )
        except Exception as e:
            logger.exception("ParkApproval failed")
            return tool_service_pb2.ParkApprovalResponse(
                success=False, approval_id="", error=str(e)
            )

    async def GetAgentMemory(
        self,
        request: tool_service_pb2.GetAgentMemoryRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetAgentMemoryResponse:
        """Read a single agent memory key. Returns value as JSON string."""
        try:
            from services.database_manager.database_helpers import fetch_one
            user_id = request.user_id or "system"
            ctx = _grpc_rls(user_id)
            profile_id = request.agent_profile_id or None
            memory_key = (request.memory_key or "")[:500]
            if not profile_id or not memory_key:
                return tool_service_pb2.GetAgentMemoryResponse(
                    success=False, memory_value_json="", error="agent_profile_id and memory_key required"
                )
            row = await fetch_one(
                """SELECT memory_value FROM agent_memory
                   WHERE agent_profile_id = $1 AND user_id = $2 AND memory_key = $3
                   AND (expires_at IS NULL OR expires_at > NOW())""",
                uuid.UUID(profile_id),
                user_id,
                memory_key,
                rls_context=ctx,
            )
            if not row:
                return tool_service_pb2.GetAgentMemoryResponse(
                    success=True, memory_value_json=""
                )
            val = row.get("memory_value")
            return tool_service_pb2.GetAgentMemoryResponse(
                success=True,
                memory_value_json=json.dumps(val, default=json_default) if val is not None else "",
            )
        except Exception as e:
            logger.exception("GetAgentMemory failed")
            return tool_service_pb2.GetAgentMemoryResponse(
                success=False, memory_value_json="", error=str(e)
            )

    async def SetAgentMemory(
        self,
        request: tool_service_pb2.SetAgentMemoryRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SetAgentMemoryResponse:
        """Write or overwrite an agent memory key."""
        try:
            from services.database_manager.database_helpers import execute, fetch_one
            user_id = request.user_id or "system"
            ctx = _grpc_rls(user_id)
            profile_id = request.agent_profile_id or None
            memory_key = (request.memory_key or "")[:500]
            memory_value_json = request.memory_value_json or "{}"
            memory_type = (request.memory_type or "kv")[:50]
            expires_at = request.expires_at if request.expires_at else None
            if not profile_id or not memory_key:
                return tool_service_pb2.SetAgentMemoryResponse(
                    success=False, error="agent_profile_id and memory_key required"
                )
            try:
                json.loads(memory_value_json)
            except (json.JSONDecodeError, TypeError):
                return tool_service_pb2.SetAgentMemoryResponse(
                    success=False, error="Invalid memory_value_json"
                )
            if expires_at:
                await execute(
                    """INSERT INTO agent_memory (agent_profile_id, user_id, memory_key, memory_value, memory_type, updated_at, expires_at)
                       VALUES ($1, $2, $3, $4::jsonb, $5, NOW(), $6::timestamptz)
                       ON CONFLICT (agent_profile_id, memory_key)
                       DO UPDATE SET memory_value = EXCLUDED.memory_value, memory_type = EXCLUDED.memory_type,
                                     updated_at = NOW(), expires_at = EXCLUDED.expires_at""",
                    uuid.UUID(profile_id),
                    user_id,
                    memory_key,
                    memory_value_json,
                    memory_type,
                    expires_at,
                    rls_context=ctx,
                )
            else:
                await execute(
                    """INSERT INTO agent_memory (agent_profile_id, user_id, memory_key, memory_value, memory_type, updated_at)
                       VALUES ($1, $2, $3, $4::jsonb, $5, NOW())
                       ON CONFLICT (agent_profile_id, memory_key)
                       DO UPDATE SET memory_value = EXCLUDED.memory_value, memory_type = EXCLUDED.memory_type, updated_at = NOW()""",
                    uuid.UUID(profile_id),
                    user_id,
                    memory_key,
                    memory_value_json,
                    memory_type,
                    rls_context=ctx,
                )
            return tool_service_pb2.SetAgentMemoryResponse(success=True)
        except Exception as e:
            logger.exception("SetAgentMemory failed")
            return tool_service_pb2.SetAgentMemoryResponse(success=False, error=str(e))

    async def ListAgentMemories(
        self,
        request: tool_service_pb2.ListAgentMemoriesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListAgentMemoriesResponse:
        """List memory keys for an agent, optionally filtered by prefix."""
        try:
            from services.database_manager.database_helpers import fetch_all
            user_id = request.user_id or "system"
            ctx = _grpc_rls(user_id)
            profile_id = request.agent_profile_id or None
            key_prefix = (request.key_prefix or "").strip() or None
            if not profile_id:
                return tool_service_pb2.ListAgentMemoriesResponse(
                    success=False, memory_keys=[], error="agent_profile_id required"
                )
            if key_prefix:
                rows = await fetch_all(
                    """SELECT memory_key FROM agent_memory
                       WHERE agent_profile_id = $1 AND user_id = $2 AND memory_key LIKE $3
                       AND (expires_at IS NULL OR expires_at > NOW())
                       ORDER BY memory_key""",
                    uuid.UUID(profile_id),
                    user_id,
                    key_prefix + "%",
                    rls_context=ctx,
                )
            else:
                rows = await fetch_all(
                    """SELECT memory_key FROM agent_memory
                       WHERE agent_profile_id = $1 AND user_id = $2
                       AND (expires_at IS NULL OR expires_at > NOW())
                       ORDER BY memory_key""",
                    uuid.UUID(profile_id),
                    user_id,
                    rls_context=ctx,
                )
            keys = [r["memory_key"] for r in rows]
            return tool_service_pb2.ListAgentMemoriesResponse(
                success=True,
                memory_keys=keys,
            )
        except Exception as e:
            logger.exception("ListAgentMemories failed")
            return tool_service_pb2.ListAgentMemoriesResponse(
                success=False, memory_keys=[], error=str(e)
            )

    async def DeleteAgentMemory(
        self,
        request: tool_service_pb2.DeleteAgentMemoryRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.DeleteAgentMemoryResponse:
        """Delete an agent memory key."""
        try:
            from services.database_manager.database_helpers import execute
            user_id = request.user_id or "system"
            ctx = _grpc_rls(user_id)
            profile_id = request.agent_profile_id or None
            memory_key = (request.memory_key or "")[:500]
            if not profile_id or not memory_key:
                return tool_service_pb2.DeleteAgentMemoryResponse(
                    success=False, error="agent_profile_id and memory_key required"
                )
            await execute(
                "DELETE FROM agent_memory WHERE agent_profile_id = $1 AND user_id = $2 AND memory_key = $3",
                uuid.UUID(profile_id),
                user_id,
                memory_key,
                rls_context=ctx,
            )
            return tool_service_pb2.DeleteAgentMemoryResponse(success=True)
        except Exception as e:
            logger.exception("DeleteAgentMemory failed")
            return tool_service_pb2.DeleteAgentMemoryResponse(success=False, error=str(e))

    async def AppendAgentMemory(
        self,
        request: tool_service_pb2.AppendAgentMemoryRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.AppendAgentMemoryResponse:
        """Append an entry to a log-type memory (JSON array)."""
        try:
            from services.database_manager.database_helpers import fetch_one, execute
            user_id = request.user_id or "system"
            ctx = _grpc_rls(user_id)
            profile_id = request.agent_profile_id or None
            memory_key = (request.memory_key or "")[:500]
            entry_json = request.entry_json or "{}"
            if not profile_id or not memory_key:
                return tool_service_pb2.AppendAgentMemoryResponse(
                    success=False, error="agent_profile_id and memory_key required"
                )
            try:
                entry = json.loads(entry_json)
            except (json.JSONDecodeError, TypeError):
                return tool_service_pb2.AppendAgentMemoryResponse(
                    success=False, error="Invalid entry_json"
                )
            row = await fetch_one(
                "SELECT memory_value, memory_type FROM agent_memory WHERE agent_profile_id = $1 AND user_id = $2 AND memory_key = $3",
                uuid.UUID(profile_id),
                user_id,
                memory_key,
                rls_context=ctx,
            )
            if row:
                current = row.get("memory_value")
                if not isinstance(current, list):
                    current = [current] if current is not None else []
                current.append(entry)
                await execute(
                    "UPDATE agent_memory SET memory_value = $1::jsonb, updated_at = NOW() WHERE agent_profile_id = $2 AND user_id = $3 AND memory_key = $4",
                    json.dumps(current),
                    uuid.UUID(profile_id),
                    user_id,
                    memory_key,
                    rls_context=ctx,
                )
            else:
                await execute(
                    """INSERT INTO agent_memory (agent_profile_id, user_id, memory_key, memory_value, memory_type, updated_at)
                       VALUES ($1, $2, $3, $4::jsonb, 'log', NOW())""",
                    uuid.UUID(profile_id),
                    user_id,
                    memory_key,
                    json.dumps([entry]),
                    rls_context=ctx,
                )
            return tool_service_pb2.AppendAgentMemoryResponse(success=True)
        except Exception as e:
            logger.exception("AppendAgentMemory failed")
            return tool_service_pb2.AppendAgentMemoryResponse(success=False, error=str(e))

    async def GetAgentRunHistory(
        self,
        request: tool_service_pb2.GetAgentRunHistoryRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetAgentRunHistoryResponse:
        """Query agent_execution_log for a user's agent run history (access-controlled by user_id)."""
        try:
            from services.database_manager.database_helpers import fetch_all

            user_id = request.user_id or "system"
            ctx = _grpc_rls(user_id)
            profile_id = request.agent_profile_id if request.HasField("agent_profile_id") and request.agent_profile_id else None
            limit = 10
            if request.HasField("limit") and request.limit > 0:
                limit = min(int(request.limit), 50)
            status_filter = request.status if request.HasField("status") and request.status else None
            start_date = request.start_date if request.HasField("start_date") and request.start_date else None
            end_date = request.end_date if request.HasField("end_date") and request.end_date else None

            conditions = ["ael.user_id = $1"]
            params = [user_id]
            n = 2
            if profile_id:
                conditions.append(f"ael.agent_profile_id = ${n}")
                params.append(uuid.UUID(profile_id))
                n += 1
            if status_filter:
                conditions.append(f"ael.status = ${n}")
                params.append(status_filter)
                n += 1
            if start_date:
                conditions.append(f"ael.started_at >= ${n}::date")
                params.append(start_date)
                n += 1
            if end_date:
                conditions.append(f"ael.started_at < (${n}::date + interval '1 day')")
                params.append(end_date)
                n += 1
            params.append(limit)
            where_clause = " AND ".join(conditions)
            q = f"""
                SELECT ael.id, ael.query, ael.status, ael.started_at, ael.duration_ms,
                       ael.connectors_called, ael.entities_discovered, ael.error_details, ael.metadata,
                       ap.name AS agent_name
                FROM agent_execution_log ael
                LEFT JOIN agent_profiles ap ON ap.id = ael.agent_profile_id AND ap.user_id = ael.user_id
                WHERE {where_clause}
                ORDER BY ael.started_at DESC
                LIMIT ${n}
            """
            rows = await fetch_all(q, *params, rls_context=ctx)
            runs = []
            agent_name_out = ""
            if profile_id and rows:
                agent_name_out = (rows[0].get("agent_name") or "").strip()
            for r in rows:
                meta = r.get("metadata") or {}
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except json.JSONDecodeError:
                        meta = {}
                steps_completed = meta.get("steps_completed", 0) or 0
                steps_total = meta.get("steps_total", 0) or 0
                conn_called = r.get("connectors_called")
                if isinstance(conn_called, list):
                    connectors_list = [str(x) for x in conn_called]
                elif isinstance(conn_called, str):
                    try:
                        connectors_list = list(json.loads(conn_called)) if conn_called else []
                    except json.JSONDecodeError:
                        connectors_list = []
                else:
                    connectors_list = []
                started = r.get("started_at")
                started_at_str = started.isoformat() if hasattr(started, "isoformat") else str(started or "")
                runs.append(
                    tool_service_pb2.AgentRunRecord(
                        execution_id=str(r["id"]),
                        agent_name=(r.get("agent_name") or "").strip(),
                        query=(r.get("query") or "")[:500],
                        status=(r.get("status") or "").strip(),
                        started_at=started_at_str,
                        duration_ms=int(r["duration_ms"]) if r.get("duration_ms") is not None else None,
                        connectors_called=connectors_list,
                        entities_discovered=int(r.get("entities_discovered") or 0),
                        error_details=(r.get("error_details") or "").strip() or None,
                        steps_completed=int(steps_completed),
                        steps_total=int(steps_total),
                    )
                )
            return tool_service_pb2.GetAgentRunHistoryResponse(
                success=True,
                runs=runs,
                total=len(runs),
                agent_name=agent_name_out,
            )
        except Exception as e:
            logger.exception("GetAgentRunHistory failed")
            return tool_service_pb2.GetAgentRunHistoryResponse(
                success=False,
                runs=[],
                total=0,
                agent_name="",
                error=str(e),
            )

