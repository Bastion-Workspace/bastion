"""Best-effort WebSocket notifications for Agent Lines (tasks, goals)."""

import logging

logger = logging.getLogger(__name__)


async def notify_line_event(line_id: str, event_type: str, data: dict = None):
    """Best-effort WebSocket push to line timeline subscribers."""
    try:
        import os
        import httpx
        base = os.getenv("BACKEND_URL", "http://backend:8000")
        payload = {"type": event_type}
        if data:
            payload.update(data)
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{base}/api/agent-factory/internal/notify-line-timeline",
                json={"line_id": line_id, "payload": payload},
                timeout=3.0,
            )
    except Exception as e:
        logger.debug("WebSocket line event emit skipped: %s", e)
