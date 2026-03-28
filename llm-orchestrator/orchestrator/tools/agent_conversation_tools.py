"""
Agent Conversation Tools - multi-turn conversations between 2+ agents.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.tools.agent_invocation_tools import invoke_agent_tool
from orchestrator.utils.action_io_registry import register_action
from orchestrator.utils.line_context import line_id_from_metadata

logger = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _is_uuid(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and _UUID_RE.match(s.strip()))


def _parse_moderator_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON { action, message } from moderator agent response. Supports raw JSON or markdown code block."""
    if not text or not text.strip():
        return None
    raw = text.strip()
    if raw.startswith("```"):
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "action" in data:
            action = data.get("action", "continue")
            if action not in ("continue", "redirect", "summarize_and_continue", "conclude"):
                action = "continue"
            return {"action": action, "message": (data.get("message") or "").strip()}
    except json.JSONDecodeError:
        pass
    return None


async def _is_conversation_halted(
    client: Any,
    user_id: str,
    initiator_agent_id: str,
    conversation_id: str,
) -> bool:
    """Check if a halt flag was set for this conversation."""
    try:
        val = await client.get_agent_memory(
            user_id=user_id,
            agent_profile_id=initiator_agent_id,
            memory_key=f"conversation_halt:{conversation_id}",
        )
        return isinstance(val, dict) and val.get("halted") is True
    except Exception:
        return False


class StartAgentConversationInputs(BaseModel):
    """Required inputs for start_agent_conversation."""
    participants: List[str] = Field(description="List of @handles or agent IDs (min 2)")
    seed_message: str = Field(description="Initial message to start the conversation")


class StartAgentConversationParams(BaseModel):
    """Optional configuration for start_agent_conversation."""
    max_turns: int = Field(default=10, description="Maximum number of back-and-forth turns")
    termination_condition: Optional[str] = Field(default=None, description="Optional condition description for early stop")
    output_destinations: Optional[List[str]] = Field(default=None, description="Where to send final output")
    team_id: Optional[str] = Field(default=None, description="Agent line UUID for timeline (same as line_id in context)")
    moderator: Optional[str] = Field(default=None, description="@handle of moderator agent; when set, moderator is invoked every moderator_frequency turns with structured control (continue/redirect/summarize_and_continue/conclude)")
    moderator_frequency: int = Field(default=3, ge=1, le=20, description="Invoke moderator after this many turns when moderator is set")


class StartAgentConversationOutputs(BaseModel):
    """Outputs from start_agent_conversation."""
    formatted: str = Field(description="Human-readable summary")
    conversation_id: str = Field(description="Root message id for the thread")
    status: str = Field(description="completed, halted, or error")


async def start_agent_conversation_tool(
    participants: List[str],
    seed_message: str,
    max_turns: int = 10,
    termination_condition: Optional[str] = None,
    output_destinations: Optional[List[str]] = None,
    team_id: Optional[str] = None,
    moderator: Optional[str] = None,
    moderator_frequency: int = 3,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Start a multi-turn conversation between 2+ agents. Creates a root message, runs turns
    in a loop, persists each turn to the team timeline, checks for halt and termination.
    When moderator is set, the moderator agent is invoked every moderator_frequency turns
    and returns a structured action: continue, redirect (new seed), summarize_and_continue,
    or conclude (end with moderator summary). Without a moderator, the conversation runs to max_turns.
    """
    if not participants or len(participants) < 2:
        return {
            "formatted": "At least 2 participants are required.",
            "conversation_id": "",
            "status": "error",
        }
    metadata = _pipeline_metadata or {}
    from_agent_id = metadata.get("agent_profile_id")
    resolved_team_id = team_id or line_id_from_metadata(metadata)
    if resolved_team_id and not _is_uuid(resolved_team_id):
        return {
            "formatted": "line_id must be the agent line UUID, not the line name. It is automatically provided from context.",
            "conversation_id": "",
            "status": "error",
        }
    client = await get_backend_tool_client()

    participant_infos: List[Dict[str, str]] = []
    for handle in participants:
        h = handle.strip()
        if not h:
            continue
        try:
            resolved = await client.resolve_agent_handle(handle=h, user_id=user_id)
            if not resolved or not resolved.get("found"):
                return {
                    "formatted": f"Could not resolve participant @{h}.",
                    "conversation_id": "",
                    "status": "error",
                }
            participant_infos.append({
                "handle": h,
                "profile_id": resolved.get("agent_profile_id", ""),
                "name": resolved.get("agent_name") or h,
            })
        except Exception as e:
            logger.warning("start_agent_conversation: resolve %s failed: %s", h, e)
            return {
                "formatted": f"Could not resolve participant @{h}: {e}",
                "conversation_id": "",
                "status": "error",
            }
    if len(participant_infos) < 2:
        return {
            "formatted": "At least 2 resolvable participants are required.",
            "conversation_id": "",
            "status": "error",
        }

    moderator_profile_id: Optional[str] = None
    moderator_handle: Optional[str] = None
    if moderator and moderator.strip():
        try:
            resolved = await client.resolve_agent_handle(handle=moderator.strip(), user_id=user_id)
            if resolved and resolved.get("found"):
                moderator_profile_id = resolved.get("agent_profile_id")
                moderator_handle = moderator.strip()
        except Exception as e:
            logger.warning("start_agent_conversation: resolve moderator %s failed: %s", moderator, e)

    root_message_id = ""
    if resolved_team_id and from_agent_id:
        try:
            result = await client.create_agent_message(
                user_id=user_id,
                team_id=resolved_team_id,
                from_agent_id=from_agent_id,
                to_agent_id=None,
                message_type="system",
                content=f"Conversation started: {seed_message[:100]}..." + ("..." if len(seed_message) > 100 else ""),
                metadata={"participants": [p["handle"] for p in participant_infos], "seed_message": seed_message},
                parent_message_id=None,
            )
            if result.get("success") and result.get("message", {}).get("id"):
                root_message_id = result["message"]["id"]
            elif result.get("message_id"):
                root_message_id = result["message_id"]
        except Exception as e:
            logger.warning("start_agent_conversation: create root message failed: %s", e)

    thread_lines: List[str] = [f"Seed: {seed_message}"]
    current_seed = seed_message
    turns_completed = 0
    final_status = "completed"
    termination_note = ""
    moderator_summary = ""

    for turn in range(max_turns):
        if root_message_id and from_agent_id:
            halted = await _is_conversation_halted(client, user_id, from_agent_id, root_message_id)
            if halted:
                final_status = "halted"
                termination_note = " (halt requested)"
                break

        idx = turn % len(participant_infos)
        info = participant_infos[idx]
        handle = info["handle"]
        profile_id = info["profile_id"]
        name = info["name"]

        thread_context = "\n\n".join(thread_lines)
        prompt = (
            "You are in a multi-agent conversation. Current direction: "
            f"{current_seed[:200]}{'...' if len(current_seed) > 200 else ''}\n\n"
            "Here is the conversation so far:\n\n"
            f"{thread_context}\n\n"
            "Respond in character. Keep your response concise. "
        )
        if termination_condition:
            prompt += f"Stop when: {termination_condition}\n\n"
        prompt += "Your response:"

        try:
            inv = await invoke_agent_tool(
                input_content=prompt,
                agent_handle=handle,
                agent_profile_id=profile_id,
                timeout_seconds=min(300, max_turns * 45),
                user_id=user_id,
                _pipeline_metadata=_pipeline_metadata,
            )
        except Exception as e:
            logger.exception("start_agent_conversation turn %s failed: %s", turn, e)
            final_status = "error"
            termination_note = f" (error on turn {turn + 1}: {e})"
            break

        resp_text = (inv.get("agent_response") or inv.get("formatted") or "").strip()
        if inv.get("status") == "error" and not resp_text:
            resp_text = "Error during response."
        thread_lines.append(f"{name}: {resp_text}")

        if resolved_team_id and root_message_id and profile_id:
            try:
                await client.create_agent_message(
                    user_id=user_id,
                    team_id=resolved_team_id,
                    from_agent_id=profile_id,
                    to_agent_id=None,
                    message_type="response",
                    content=resp_text,
                    metadata={"turn": turn + 1, "participant": handle},
                    parent_message_id=root_message_id,
                )
            except Exception as e:
                logger.debug("start_agent_conversation: persist turn message failed: %s", e)

        turns_completed += 1

        if moderator_profile_id and moderator_handle and turns_completed > 0 and turns_completed % moderator_frequency == 0:
            mod_context = "\n\n".join(thread_lines)
            mod_prompt = (
                "You are the moderator of this multi-agent conversation. Here is the conversation so far:\n\n"
                f"{mod_context}\n\n"
                "Respond with ONLY a JSON object (no other text) with two keys: "
                '"action" (one of: continue, redirect, summarize_and_continue, conclude) and '
                '"message" (string). '
                "continue = let conversation continue as-is. "
                "redirect = replace the direction for the next round; put the new direction in message. "
                "summarize_and_continue = add a brief summary in message; the conversation will continue with that summary injected. "
                "conclude = end the conversation; put the final summary in message."
            )
            try:
                mod_inv = await invoke_agent_tool(
                    input_content=mod_prompt,
                    agent_handle=moderator_handle,
                    agent_profile_id=moderator_profile_id,
                    timeout_seconds=60,
                    user_id=user_id,
                    _pipeline_metadata=_pipeline_metadata,
                )
                mod_text = (mod_inv.get("agent_response") or mod_inv.get("formatted") or "").strip()
                mod_result = _parse_moderator_response(mod_text)
                if mod_result:
                    action = mod_result.get("action", "continue")
                    msg = mod_result.get("message") or ""
                    if action == "conclude":
                        moderator_summary = msg
                        termination_note = " (moderator concluded)"
                        break
                    if action == "redirect" and msg:
                        current_seed = msg
                        thread_lines.append(f"Moderator (redirect): {msg}")
                    elif action == "summarize_and_continue" and msg:
                        thread_lines.append(f"Moderator (summary): {msg}")
            except Exception as e:
                logger.debug("start_agent_conversation: moderator invocation failed: %s", e)

    summary = (
        f"Conversation finished after {turns_completed} turn(s); status={final_status}{termination_note}. "
        f"Root message_id={root_message_id or 'n/a'}."
    )
    if moderator_summary:
        summary = moderator_summary + "\n\n" + summary
    return {
        "formatted": summary,
        "conversation_id": root_message_id,
        "status": final_status,
    }


class HaltAgentConversationInputs(BaseModel):
    """Required inputs for halt_agent_conversation."""
    conversation_id: str = Field(description="Root message id or conversation identifier to halt")


class HaltAgentConversationOutputs(BaseModel):
    """Outputs from halt_agent_conversation."""
    formatted: str = Field(description="Human-readable summary")
    success: bool = Field(description="Whether halt was recorded")
    turns_completed: int = Field(default=0, description="Turns completed before halt")
    final_state: str = Field(default="", description="State description if available")


async def halt_agent_conversation_tool(
    conversation_id: str,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Stop a running agent-to-agent conversation by setting a halt flag in agent memory.
    """
    metadata = _pipeline_metadata or {}
    agent_profile_id = metadata.get("agent_profile_id")
    if not agent_profile_id:
        return {
            "formatted": "No agent context; cannot halt conversation.",
            "success": False,
            "turns_completed": 0,
            "final_state": "",
        }
    try:
        client = await get_backend_tool_client()
        await client.set_agent_memory(
            user_id=user_id,
            agent_profile_id=agent_profile_id,
            memory_key=f"conversation_halt:{conversation_id}",
            memory_value_json=json.dumps({"halted": True, "conversation_id": conversation_id}),
            memory_type="kv",
        )
        return {
            "formatted": f"Halt requested for conversation {conversation_id}.",
            "success": True,
            "turns_completed": 0,
            "final_state": "halt_requested",
        }
    except Exception as e:
        logger.warning("halt_agent_conversation failed: %s", e)
        return {
            "formatted": f"Failed to set halt flag: {e}",
            "success": False,
            "turns_completed": 0,
            "final_state": "",
        }


register_action(
    name="start_agent_conversation",
    category="agent_communication",
    description="Start a multi-turn conversation between two or more agents",
    inputs_model=StartAgentConversationInputs,
    params_model=StartAgentConversationParams,
    outputs_model=StartAgentConversationOutputs,
    tool_function=start_agent_conversation_tool,
)
register_action(
    name="halt_agent_conversation",
    category="agent_communication",
    description="Stop a running agent-to-agent conversation",
    inputs_model=HaltAgentConversationInputs,
    params_model=None,
    outputs_model=HaltAgentConversationOutputs,
    tool_function=halt_agent_conversation_tool,
)
