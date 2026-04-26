"""gRPC handlers for per-user shell command policy rules and shell approval grant/consume."""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import grpc
from protos import tool_service_pb2
from utils.grpc_rls import grpc_user_rls as _grpc_rls

logger = logging.getLogger(__name__)

_VALID_MATCH = frozenset({"prefix", "contains", "glob"})
_VALID_ACTION = frozenset({"allow", "deny", "require_approval"})


class ShellPolicyHandlersMixin:
    """Mixin: user_shell_policy CRUD + agent_approval_queue shell grant/consume."""

    async def GetUserShellPolicy(
        self,
        request: tool_service_pb2.GetUserShellPolicyRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetUserShellPolicyResponse:
        try:
            from services.database_manager.database_helpers import fetch_all

            user_id = (request.user_id or "").strip()
            if not user_id:
                return tool_service_pb2.GetUserShellPolicyResponse(
                    success=False, rules_json="[]", error="user_id required"
                )
            ctx = _grpc_rls(user_id)
            rows = await fetch_all(
                """
                SELECT id::text AS id, pattern, match_mode, action,
                       scope_workspace_id::text AS scope_workspace_id,
                       label, priority
                FROM user_shell_policy
                WHERE user_id = $1
                ORDER BY priority ASC NULLS LAST, created_at ASC
                """,
                user_id,
                rls_context=ctx,
            )
            rules: List[Dict[str, Any]] = []
            for r in rows or []:
                rules.append(
                    {
                        "id": r.get("id"),
                        "pattern": r.get("pattern") or "",
                        "match_mode": r.get("match_mode") or "prefix",
                        "action": r.get("action") or "allow",
                        "scope_workspace_id": r.get("scope_workspace_id"),
                        "label": r.get("label"),
                        "priority": r.get("priority"),
                    }
                )
            return tool_service_pb2.GetUserShellPolicyResponse(
                success=True,
                rules_json=json.dumps(rules),
                error="",
            )
        except Exception as e:
            logger.exception("GetUserShellPolicy failed")
            return tool_service_pb2.GetUserShellPolicyResponse(
                success=False, rules_json="[]", error=str(e)
            )

    async def UpsertShellPolicyRule(
        self,
        request: tool_service_pb2.UpsertShellPolicyRuleRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpsertShellPolicyRuleResponse:
        try:
            from services.database_manager.database_helpers import fetch_one, fetch_value

            user_id = (request.user_id or "").strip()
            pattern = (request.pattern or "").strip()
            match_mode = (request.match_mode or "prefix").strip().lower()
            action = (request.action or "").strip().lower()
            scope_ws = (request.scope_workspace_id or "").strip() or None
            label = (request.label or "").strip() or None
            priority = int(request.priority) if request.priority else 50

            if not user_id or not pattern:
                return tool_service_pb2.UpsertShellPolicyRuleResponse(
                    success=False, rule_id="", error="user_id and pattern required"
                )
            if match_mode not in _VALID_MATCH:
                return tool_service_pb2.UpsertShellPolicyRuleResponse(
                    success=False, rule_id="", error=f"invalid match_mode: {match_mode}"
                )
            if action not in _VALID_ACTION:
                return tool_service_pb2.UpsertShellPolicyRuleResponse(
                    success=False, rule_id="", error=f"invalid action: {action}"
                )

            ctx = _grpc_rls(user_id)
            scope_uuid: Optional[uuid.UUID] = None
            if scope_ws:
                ok = await fetch_one(
                    "SELECT id FROM code_workspaces WHERE id = $1::uuid AND user_id = $2",
                    scope_ws,
                    user_id,
                    rls_context=ctx,
                )
                if not ok:
                    return tool_service_pb2.UpsertShellPolicyRuleResponse(
                        success=False, rule_id="", error="scope_workspace_id not found for user"
                    )
                scope_uuid = uuid.UUID(scope_ws)

            rule_id = (request.rule_id or "").strip()
            if rule_id:
                try:
                    rid = uuid.UUID(rule_id)
                except ValueError:
                    return tool_service_pb2.UpsertShellPolicyRuleResponse(
                        success=False, rule_id="", error="invalid rule_id"
                    )
                updated_id = await fetch_value(
                    """
                    UPDATE user_shell_policy
                    SET pattern = $1, match_mode = $2, action = $3,
                        scope_workspace_id = $4, label = $5, priority = $6
                    WHERE id = $7::uuid AND user_id = $8
                    RETURNING id::text
                    """,
                    pattern,
                    match_mode,
                    action,
                    scope_uuid,
                    label,
                    priority,
                    rid,
                    user_id,
                    rls_context=ctx,
                )
                if not updated_id:
                    return tool_service_pb2.UpsertShellPolicyRuleResponse(
                        success=False, rule_id="", error="update failed or rule not found"
                    )
                return tool_service_pb2.UpsertShellPolicyRuleResponse(
                    success=True, rule_id=str(updated_id), error=""
                )

            new_id = await fetch_value(
                """
                INSERT INTO user_shell_policy
                    (user_id, pattern, match_mode, action, scope_workspace_id, label, priority)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id::text
                """,
                user_id,
                pattern,
                match_mode,
                action,
                scope_uuid,
                label,
                priority,
                rls_context=ctx,
            )
            if not new_id:
                return tool_service_pb2.UpsertShellPolicyRuleResponse(
                    success=False, rule_id="", error="insert failed"
                )
            return tool_service_pb2.UpsertShellPolicyRuleResponse(
                success=True, rule_id=str(new_id), error=""
            )
        except Exception as e:
            logger.exception("UpsertShellPolicyRule failed")
            return tool_service_pb2.UpsertShellPolicyRuleResponse(
                success=False, rule_id="", error=str(e)
            )

    async def DeleteShellPolicyRule(
        self,
        request: tool_service_pb2.DeleteShellPolicyRuleRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.DeleteShellPolicyRuleResponse:
        try:
            from services.database_manager.database_helpers import execute

            user_id = (request.user_id or "").strip()
            rid = (request.rule_id or "").strip()
            if not user_id or not rid:
                return tool_service_pb2.DeleteShellPolicyRuleResponse(
                    success=False, error="user_id and rule_id required"
                )
            try:
                uuid.UUID(rid)
            except ValueError:
                return tool_service_pb2.DeleteShellPolicyRuleResponse(
                    success=False, error="invalid rule_id"
                )
            ctx = _grpc_rls(user_id)

            await execute(
                "DELETE FROM user_shell_policy WHERE id = $1::uuid AND user_id = $2",
                rid,
                user_id,
                rls_context=ctx,
            )
            return tool_service_pb2.DeleteShellPolicyRuleResponse(success=True, error="")
        except Exception as e:
            logger.exception("DeleteShellPolicyRule failed")
            return tool_service_pb2.DeleteShellPolicyRuleResponse(success=False, error=str(e))

    async def GrantAndConsumeShellApproval(
        self,
        request: tool_service_pb2.GrantAndConsumeShellApprovalRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GrantAndConsumeShellApprovalResponse:
        """pending -> approved (consume=false); approved -> consumed (consume=true)."""
        try:
            from services.database_manager.database_helpers import fetch_value

            user_id = (request.user_id or "").strip()
            if not user_id:
                return tool_service_pb2.GrantAndConsumeShellApprovalResponse(
                    success=False, granted_or_consumed=False, error="user_id required"
                )
            ctx = _grpc_rls(user_id)
            consume = bool(request.consume)
            approval_id = (request.approval_id or "").strip()
            command = (request.command or "").strip()

            now = datetime.now(timezone.utc)

            if not consume:
                if not approval_id:
                    return tool_service_pb2.GrantAndConsumeShellApprovalResponse(
                        success=False, granted_or_consumed=False, error="approval_id required when consume=false"
                    )
                try:
                    aid = uuid.UUID(approval_id)
                except ValueError:
                    return tool_service_pb2.GrantAndConsumeShellApprovalResponse(
                        success=False, granted_or_consumed=False, error="invalid approval_id"
                    )
                updated = await fetch_value(
                    """
                    UPDATE agent_approval_queue
                    SET status = 'approved', responded_at = $1
                    WHERE id = $2::uuid AND user_id = $3
                      AND governance_type = 'shell_command_approval'
                      AND status = 'pending'
                    RETURNING id::text
                    """,
                    now,
                    aid,
                    user_id,
                    rls_context=ctx,
                )
                ok = bool(updated)
                return tool_service_pb2.GrantAndConsumeShellApprovalResponse(
                    success=True,
                    granted_or_consumed=ok,
                    error="" if ok else "no pending shell approval found for id",
                )

            # consume=true
            if approval_id:
                try:
                    aid = uuid.UUID(approval_id)
                except ValueError:
                    return tool_service_pb2.GrantAndConsumeShellApprovalResponse(
                        success=False, granted_or_consumed=False, error="invalid approval_id"
                    )
                updated = await fetch_value(
                    """
                    UPDATE agent_approval_queue
                    SET status = 'consumed', responded_at = $1
                    WHERE id = $2::uuid AND user_id = $3
                      AND governance_type = 'shell_command_approval'
                      AND status = 'approved'
                    RETURNING id::text
                    """,
                    now,
                    aid,
                    user_id,
                    rls_context=ctx,
                )
                ok = bool(updated)
                return tool_service_pb2.GrantAndConsumeShellApprovalResponse(
                    success=True,
                    granted_or_consumed=ok,
                    error="" if ok else "no approved shell approval for id",
                )

            if not command:
                return tool_service_pb2.GrantAndConsumeShellApprovalResponse(
                    success=False, granted_or_consumed=False, error="command required for consume without approval_id"
                )

            updated = await fetch_value(
                """
                WITH picked AS (
                    SELECT id FROM agent_approval_queue
                    WHERE user_id = $1
                      AND governance_type = 'shell_command_approval'
                      AND preview_data->>'command' = $2
                      AND status = 'approved'
                      AND created_at > NOW() - INTERVAL '1 hour'
                    ORDER BY created_at DESC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE agent_approval_queue a
                SET status = 'consumed', responded_at = $3
                FROM picked
                WHERE a.id = picked.id
                RETURNING a.id::text
                """,
                user_id,
                command,
                now,
                rls_context=ctx,
            )
            ok = bool(updated)
            return tool_service_pb2.GrantAndConsumeShellApprovalResponse(
                success=True,
                granted_or_consumed=ok,
                error="" if ok else "no approved shell approval for command",
            )
        except Exception as e:
            logger.exception("GrantAndConsumeShellApproval failed")
            return tool_service_pb2.GrantAndConsumeShellApprovalResponse(
                success=False, granted_or_consumed=False, error=str(e)
            )
