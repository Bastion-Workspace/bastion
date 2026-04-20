"""gRPC handler for per-execution step traces (agent_execution_steps)."""

import logging

import grpc
from protos import tool_service_pb2

logger = logging.getLogger(__name__)


def _empty_trace_response(**kwargs) -> tool_service_pb2.GetExecutionTraceResponse:
    base = dict(
        success=False,
        error="",
        execution_id="",
        agent_name="",
        query="",
        status="",
        started_at="",
        completed_at="",
        model_used="",
        error_details="",
        steps=[],
    )
    base.update(kwargs)
    return tool_service_pb2.GetExecutionTraceResponse(**base)


class AgentExecutionTraceHandlersMixin:
    """Mixin: GetExecutionTrace for Agent Factory execution detail."""

    async def GetExecutionTrace(
        self,
        request: tool_service_pb2.GetExecutionTraceRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetExecutionTraceResponse:
        """Return one execution log row and its steps for the owning user."""
        try:
            from tools_service.services.agent_execution_trace_ops import get_execution_trace_payload
            from utils.grpc_rls import grpc_user_rls as _grpc_rls

            user_id = request.user_id or "system"
            ctx = _grpc_rls(user_id)
            exec_id = (request.execution_id or "").strip()

            include_io = True
            if request.HasField("include_io"):
                include_io = request.include_io
            include_tool_calls = False
            if request.HasField("include_tool_calls"):
                include_tool_calls = request.include_tool_calls

            payload = await get_execution_trace_payload(
                user_id,
                exec_id,
                include_io=include_io,
                include_tool_calls=include_tool_calls,
                rls_context=ctx,
            )

            if not payload.get("success"):
                err = payload.get("error") or "unknown error"
                eid = payload.get("execution_id") or ""
                return _empty_trace_response(error=err, execution_id=eid)

            row = payload["execution"]
            steps_data = payload.get("steps") or []

            steps_pb = []
            for s in steps_data:
                rec = tool_service_pb2.ExecutionStepRecord(
                    step_index=int(s.get("step_index") or 0),
                    step_name=(s.get("step_name") or "")[:255],
                    step_type=(s.get("step_type") or "")[:50],
                    action_name=(s.get("action_name") or "")[:255],
                    status=(s.get("status") or "")[:50],
                    started_at=s.get("started_at") or "",
                    completed_at=s.get("completed_at") or "",
                    inputs_json=s.get("inputs_json") or "",
                    outputs_json=s.get("outputs_json") or "",
                    error_details=s.get("error_details") or "",
                    tool_call_trace_json=s.get("tool_call_trace_json") or "",
                    input_tokens=int(s.get("input_tokens") or 0),
                    output_tokens=int(s.get("output_tokens") or 0),
                )
                dm = s.get("duration_ms")
                if dm is not None:
                    rec.duration_ms = int(dm)
                steps_pb.append(rec)

            ed = row.get("error_details") or ""
            resp = tool_service_pb2.GetExecutionTraceResponse(
                success=True,
                execution_id=str(row.get("id") or ""),
                agent_name=(row.get("agent_name") or "").strip(),
                query=row.get("query") or "",
                status=(row.get("status") or "").strip(),
                started_at=row.get("started_at_iso") or "",
                completed_at=row.get("completed_at_iso") or "",
                tokens_input=int(row.get("tokens_input") or 0),
                tokens_output=int(row.get("tokens_output") or 0),
                cost_usd=row.get("cost_usd_str") or "",
                model_used=(row.get("model_used") or "")[:255],
                error_details=(ed[:2000] if ed else ""),
                steps=steps_pb,
                error="",
            )
            dm = row.get("duration_ms")
            if dm is not None:
                resp.duration_ms = int(dm)
            return resp
        except Exception as e:
            logger.exception("GetExecutionTrace failed")
            return _empty_trace_response(error=str(e))
