"""Agent execution log + steps fetch for GetExecutionTrace (no gRPC/protobuf)."""

import json
import uuid
from typing import Any, Dict, List, Tuple

_IO_JSON_CAP = 1500
_TOOL_TRACE_JSON_CAP = 8000


def json_trunc(value: Any, max_chars: int) -> str:
    """Serialize value to JSON and truncate to max_chars (UTF-8 safe by char count)."""
    if value is None:
        return "{}"
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            value = parsed
        except json.JSONDecodeError:
            raw = value
            if len(raw) <= max_chars:
                return raw
            return raw[:max_chars] + "..."
    try:
        s = json.dumps(value, default=str)
    except TypeError:
        s = str(value)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "..."


def step_tokens_from_outputs(outputs: Any) -> Tuple[int, int]:
    """Best-effort per-step token counts from persisted outputs_json."""
    if not isinstance(outputs, dict):
        return 0, 0
    inp = outputs.get("input_tokens")
    outp = outputs.get("output_tokens")
    tu = outputs.get("_token_usage")
    if isinstance(tu, dict):
        if inp is None:
            inp = tu.get("input_tokens")
        if outp is None:
            outp = tu.get("output_tokens")
    return int(inp or 0), int(outp or 0)


def _iso(dt: Any) -> str:
    if dt is None:
        return ""
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    return str(dt)


async def get_execution_trace_payload(
    user_id: str,
    execution_id: str,
    *,
    include_io: bool,
    include_tool_calls: bool,
    rls_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Load execution row and steps. Returns dict:
      success: bool
      error: str (empty when success)
      execution: dict | None (id as str uuid, plus row fields)
      steps: list[dict] (protobuf-friendly primitives)
    """
    from services.database_manager.database_helpers import fetch_all, fetch_one

    exec_id = (execution_id or "").strip()
    if not exec_id:
        return {"success": False, "error": "execution_id required", "execution": None, "steps": []}

    try:
        exec_uuid = uuid.UUID(exec_id)
    except ValueError:
        return {"success": False, "error": "invalid execution_id", "execution": None, "steps": []}

    row = await fetch_one(
        """
        SELECT ael.id, ael.query, ael.status, ael.started_at, ael.completed_at,
               ael.duration_ms, ael.error_details, ael.tokens_input, ael.tokens_output,
               ael.cost_usd, ael.model_used, ap.name AS agent_name
        FROM agent_execution_log ael
        LEFT JOIN agent_profiles ap ON ap.id = ael.agent_profile_id AND ap.user_id = ael.user_id
        WHERE ael.id = $1 AND ael.user_id = $2
        """,
        exec_uuid,
        user_id,
        rls_context=rls_context,
    )
    if not row:
        return {
            "success": False,
            "error": "execution not found",
            "execution": None,
            "steps": [],
            "execution_id": str(exec_uuid),
        }

    steps_rows = await fetch_all(
        """
        SELECT step_index, step_name, step_type, action_name, status,
               started_at, completed_at, duration_ms, inputs_json, outputs_json,
               error_details, tool_call_trace
        FROM agent_execution_steps
        WHERE execution_id = $1
        ORDER BY step_index
        """,
        exec_uuid,
        rls_context=rls_context,
    )

    steps_out: List[Dict[str, Any]] = []
    for s in steps_rows or []:
        inputs_raw = s.get("inputs_json") or {}
        outputs_raw = s.get("outputs_json") or {}
        if isinstance(inputs_raw, str):
            try:
                inputs_raw = json.loads(inputs_raw)
            except json.JSONDecodeError:
                inputs_raw = {}
        if isinstance(outputs_raw, str):
            try:
                outputs_raw = json.loads(outputs_raw)
            except json.JSONDecodeError:
                outputs_raw = {}

        tool_trace = s.get("tool_call_trace")
        if isinstance(tool_trace, str):
            try:
                tool_trace = json.loads(tool_trace)
            except json.JSONDecodeError:
                tool_trace = []
        if not isinstance(tool_trace, list):
            tool_trace = []

        in_tok, out_tok = step_tokens_from_outputs(outputs_raw)

        if include_io:
            inputs_str = json_trunc(inputs_raw, _IO_JSON_CAP)
            outputs_str = json_trunc(outputs_raw, _IO_JSON_CAP)
        else:
            inputs_str = ""
            outputs_str = ""

        if include_tool_calls:
            trace_str = json_trunc(tool_trace, _TOOL_TRACE_JSON_CAP)
        else:
            trace_str = ""

        started = s.get("started_at")
        completed = s.get("completed_at")
        err = s.get("error_details") or ""
        steps_out.append(
            {
                "step_index": int(s.get("step_index") or 0),
                "step_name": (s.get("step_name") or "")[:255],
                "step_type": (s.get("step_type") or "")[:50],
                "action_name": (s.get("action_name") or "")[:255],
                "status": (s.get("status") or "")[:50],
                "started_at": _iso(started),
                "completed_at": _iso(completed),
                "inputs_json": inputs_str,
                "outputs_json": outputs_str,
                "error_details": (err[:2000] if err else ""),
                "tool_call_trace_json": trace_str,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "duration_ms": s.get("duration_ms"),
            }
        )

    ex = dict(row)
    ex["id"] = str(ex["id"])
    ex["started_at_iso"] = _iso(row.get("started_at"))
    ex["completed_at_iso"] = _iso(row.get("completed_at"))
    cost = row.get("cost_usd")
    ex["cost_usd_str"] = str(cost) if cost is not None else ""

    return {"success": True, "error": "", "execution": ex, "steps": steps_out}
