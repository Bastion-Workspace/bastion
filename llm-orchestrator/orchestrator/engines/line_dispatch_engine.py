"""
Line dispatch engine: run the agent line CEO from chat with full line briefing.

Triggered when backend sets metadata line_dispatch_mode and ceo_profile_id.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LineDispatchEngine:
    """Invokes the line CEO via CustomAgentRunner with chat-framed briefing."""

    async def process(
        self,
        query: str,
        metadata: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Any]] = None,
        cancellation_token: Optional[Any] = None,
    ) -> Dict[str, Any]:
        meta = dict(metadata or {})
        ceo_id = meta.get("ceo_profile_id")
        if not ceo_id:
            return {
                "formatted": "Line dispatch error: no CEO profile id in context.",
                "response": "Line dispatch error: no CEO profile id in context.",
                "task_status": "error",
            }

        briefing = (meta.get("line_dispatch_briefing") or meta.get("team_chat_context") or "").strip()
        rules = (meta.get("line_dispatch_tool_rules") or "").strip()

        composed_query = (
            "You are the CEO (root) of this agent line. The user sent this message from **chat** "
            "(interactive conversation, not a scheduled heartbeat). Respond clearly in chat; use tools "
            "to delegate to your team when work should be done by others.\n\n"
            "LINE CHAT — DELEGATION:\n"
            "- Use create_task_for_agent when work can continue asynchronously.\n"
            "- Use send_to_agent(..., wait_for_response=True) when you need a worker's output in this turn.\n"
            "- line_id is in your tool context; use UUIDs from the briefing for tool calls.\n\n"
        )
        if rules:
            composed_query += f"{rules}\n\n"
        composed_query += f"USER MESSAGE (address this):\n{query or '(empty)'}\n\n"
        if briefing:
            composed_query += f"LINE STATUS AND CONTEXT:\n{briefing}\n"

        runner_metadata = {
            **meta,
            "agent_profile_id": str(ceo_id),
            "trigger_type": "line_chat_dispatch",
        }
        line_id = meta.get("line_id") or meta.get("team_id") or meta.get("team_context_id")
        if line_id:
            runner_metadata["line_id"] = str(line_id)
            runner_metadata["team_id"] = str(line_id)

        from orchestrator.agents.custom_agent_runner import CustomAgentRunner

        runner = CustomAgentRunner()
        try:
            return await runner.process(
                query=composed_query,
                metadata=runner_metadata,
                messages=messages or [],
                cancellation_token=cancellation_token,
            )
        except Exception as e:
            logger.exception("Line dispatch CEO run failed: %s", e)
            return {
                "formatted": f"Line dispatch failed: {e}",
                "response": f"Line dispatch failed: {e}",
                "task_status": "error",
            }
