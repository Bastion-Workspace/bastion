"""
Optional live UI progress for Agent Factory playbook runs (chat "chewing on it" subtitle).

UnifiedDispatcher registers an asyncio.Queue under playbook_progress_registry_token (string)
on metadata. Playbook nodes and deep-agent phases push short string maps consumed as
streaming status chunks. The queue must not live on checkpointed graph state (not msgpack-serializable).
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# In-process registry: token -> Queue. Token is the only progress handle stored on LangGraph metadata.
_playbook_progress_queues: Dict[str, asyncio.Queue] = {}


def register_playbook_progress_queue(token: str, queue: asyncio.Queue) -> None:
    _playbook_progress_queues[str(token)] = queue


def unregister_playbook_progress_queue(token: str) -> None:
    _playbook_progress_queues.pop(str(token), None)


def resolve_playbook_progress_queue(metadata: Optional[Dict[str, Any]]) -> Optional[asyncio.Queue]:
    """Resolve queue from checkpoint-safe metadata (token) or direct queue (tests)."""
    if not isinstance(metadata, dict):
        return None
    q = metadata.get("playbook_progress_queue")
    if q is not None:
        return q  # type: ignore[return-value]
    tok = metadata.get("playbook_progress_registry_token")
    if tok is not None:
        return _playbook_progress_queues.get(str(tok))
    return None


def _coerce_metadata_strings(payload: Dict[str, Any]) -> Dict[str, str]:
    """ChatChunk.metadata is map<string,string>; truncate very long values."""
    out: Dict[str, str] = {}
    for k, v in payload.items():
        if v is None:
            continue
        if isinstance(v, (dict, list)):
            s = json.dumps(v, ensure_ascii=False)
        else:
            s = str(v)
        s = s.strip()
        if not s:
            continue
        out[str(k)] = s[:6000]
    return out


def format_activity_detail_line(meta: Dict[str, str]) -> str:
    """Single-line status for SSE message + chat subtitle."""
    step_key = (meta.get("playbook_activity_step") or meta.get("playbook_step_key") or "").strip()
    stype = (meta.get("playbook_activity_type") or meta.get("playbook_step_type") or "").strip()
    phase = (meta.get("deep_phase_name") or "").strip()
    ptype = (meta.get("deep_phase_type") or "").strip()
    plan = (meta.get("deep_phases_plan") or "").strip()

    parts: list[str] = []
    if step_key:
        parts.append(f"Step: {step_key}")
    if stype and stype != step_key:
        parts.append(f"({stype})")
    if phase:
        parts.append("Phase: " + phase + (f" ({ptype})" if ptype else ""))
    elif plan:
        parts.append(f"Phases: {plan}")
    if not parts:
        return "Working…"
    return " · ".join(parts)


async def emit_playbook_ui_progress(metadata: Optional[Dict[str, Any]], payload: Dict[str, Any]) -> None:
    """Push one progress event to the chat stream (non-fatal if queue missing or slow)."""
    if not isinstance(metadata, dict):
        return
    q = resolve_playbook_progress_queue(metadata)
    if q is None:
        return
    merged = {**payload}
    # Deep-agent phase events only send phase fields; keep step context from metadata.
    if metadata.get("playbook_activity_step") and not merged.get("playbook_activity_step"):
        merged["playbook_activity_step"] = metadata["playbook_activity_step"]
    if metadata.get("playbook_activity_type") and not merged.get("playbook_activity_type"):
        merged["playbook_activity_type"] = metadata["playbook_activity_type"]
    step_key = (merged.get("playbook_activity_step") or merged.get("playbook_step_key") or "").strip()
    stype = (merged.get("playbook_activity_type") or merged.get("playbook_step_type") or "").strip()
    if step_key:
        metadata["playbook_activity_step"] = step_key
    if stype:
        metadata["playbook_activity_type"] = stype

    md = _coerce_metadata_strings(merged)
    detail = format_activity_detail_line(md)
    md["activity_detail"] = detail
    md["message"] = detail
    try:
        await asyncio.wait_for(q.put(md), timeout=2.0)
    except asyncio.TimeoutError:
        logger.debug("playbook_progress_queue blocked; dropped progress event")
