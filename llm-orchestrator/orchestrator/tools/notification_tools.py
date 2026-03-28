"""
Notification Tools - Agent-initiated multi-channel notifications.

Provides notify_user_tool (high-level, multi-output) and send_channel_message_tool
(low-level, single channel). Always creates an in-app conversation as the ledger;
external channels (Telegram, Discord, email) are optional delivery mechanisms.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action
from orchestrator.utils.tool_type_models import NotificationResult

logger = logging.getLogger(__name__)

try:
    from orchestrator.tools.email_tools import send_email_tool as _send_email_tool
except Exception:
    _send_email_tool = None

EXTERNAL_CHANNELS = {"telegram", "discord", "slack", "email"}
ALL_CHANNELS = {"in_app", "telegram", "discord", "slack", "email"}


def _detect_html_body(body: str) -> bool:
    """Return True if body appears to be an HTML document."""
    if not body or not body.strip():
        return False
    stripped = body.strip().lower()
    return stripped.startswith("<!doctype") or stripped.startswith("<html")


# ── I/O Models: send_channel_message ─────────────────────────────────────────


class SendChannelMessageInputs(BaseModel):
    """Required inputs for send_channel_message."""
    message: str = Field(default="", description="Message content to send (wire from e.g. {get_weather.formatted})")


class SendChannelMessageParams(BaseModel):
    """Optional configuration for send_channel_message."""
    channel: str = Field(
        default="telegram",
        description="Messaging channel: in_app, telegram, discord, slack, email",
    )
    connection_id: str = Field(
        default="",
        description="Specific connection ID; empty = default for provider",
    )
    format: str = Field(
        default="markdown",
        description=(
            "Message format: markdown, plain, html. "
            "For email channel: set to 'html' when the message body is HTML (e.g. <!DOCTYPE html>...). "
            "HTML is also auto-detected if the body starts with <!DOCTYPE or <html."
        ),
    )
    recipient_chat_id: str = Field(
        default="",
        description=(
            "Optional. Where to send the message. If empty, uses the last chat that messaged the bot. "
            "Telegram: use numeric chat_id (for DMs the user must have messaged the bot first), or @channelname for public channels (bot must be admin). "
            "Discord: use numeric channel ID. Note: you cannot send to a private Telegram user by @username; use their numeric id or leave empty after they message the bot."
        ),
    )
    to_email: str = Field(
        default="",
        description="For email channel: recipient address. Leave empty to use the user's email from notification preferences.",
    )
    from_source: str = Field(
        default="system",
        description="For email channel: 'system' = Bastion SMTP (default), 'user' = user's own email connection (Gmail/Microsoft)",
    )
    subject: str = Field(
        default="",
        description="For email channel: subject line. Leave empty to auto-generate from message.",
    )
    agent_name: str = Field(
        default="",
        description="Agent attribution name for in-app notification display (optional; can be auto-injected from profile).",
    )


class SendChannelMessageOutputs(BaseModel):
    """Outputs for send_channel_message."""
    success: bool = Field(description="Whether delivery succeeded")
    message_id: str = Field(default="", description="Provider message ID")
    channel: str = Field(default="", description="Channel that was used")
    error: Optional[str] = Field(default=None, description="Error if failed")
    formatted: str = Field(description="Human-readable summary")


# ── I/O Models: notify_user ──────────────────────────────────────────────────


class NotifyUserInputs(BaseModel):
    """Required inputs for notify_user."""
    message: str = Field(description="Message content to send to the user")


class NotifyUserParams(BaseModel):
    """Optional configuration for notify_user."""
    title: str = Field(
        default="",
        description="Conversation title (auto-generated from message if empty)",
    )
    channels: List[str] = Field(
        default=["default"],
        description=(
            "Delivery channels: 'default' (use user preferences), "
            "'in_app', 'telegram', 'discord', 'slack', 'email', or a list of multiple"
        ),
    )
    urgency: str = Field(
        default="normal",
        description="Urgency level: low, normal, high",
    )
    conversation_id: str = Field(
        default="",
        description="Append to existing conversation, or empty for new",
    )
    agent_name: str = Field(
        default="",
        description="Agent attribution name for display",
    )
    agent_profile_id: str = Field(
        default="",
        description="Agent Factory profile ID (if applicable)",
    )
    email_from: str = Field(
        default="system",
        description="For email channel: 'system' = Bastion SMTP (default), 'user' = user's email connection",
    )


class ChannelDeliveryResult(BaseModel):
    """Result of delivering to a single channel."""
    channel: str = Field(description="Channel name")
    success: bool = Field(description="Whether delivery succeeded")
    message_id: Optional[str] = Field(default=None, description="Message ID")
    error: Optional[str] = Field(default=None, description="Error if failed")


class NotifyUserOutputs(BaseModel):
    """Outputs for notify_user."""
    conversation_id: str = Field(description="In-app conversation ID (always created)")
    message_id: str = Field(description="In-app message ID")
    deliveries: List[ChannelDeliveryResult] = Field(
        default_factory=list,
        description="Per-channel delivery results",
    )
    all_succeeded: bool = Field(description="True if all channels delivered")
    formatted: str = Field(description="Human-readable summary")


# ── I/O Models: schedule_reminder ─────────────────────────────────────────────


class ScheduleReminderInputs(BaseModel):
    """Required inputs for schedule_reminder."""
    message: str = Field(description="Message content to send when the reminder fires")


class ScheduleReminderParams(BaseModel):
    """Optional configuration for schedule_reminder."""
    delay_seconds: int = Field(
        default=3600,
        description="Delay in seconds before sending the reminder (default: 1 hour)",
    )
    channels: List[str] = Field(
        default=["default"],
        description="Delivery channels when reminder fires: default, in_app, telegram, discord, email",
    )
    user_id: str = Field(default="system", description="Target user ID")


class ScheduleReminderOutputs(BaseModel):
    """Outputs for schedule_reminder."""
    success: bool = Field(description="Whether the reminder was scheduled")
    reminder_id: Optional[str] = Field(default=None, description="Server-assigned reminder ID if scheduled")
    scheduled_at: Optional[str] = Field(default=None, description="ISO 8601 time when reminder will fire")
    error: Optional[str] = Field(default=None, description="Error message if scheduling failed")
    formatted: str = Field(description="Human-readable summary")


# ── Tool Functions ───────────────────────────────────────────────────────────


async def send_channel_message_tool(
    message: str = "",
    user_id: str = "system",
    channel: str = "telegram",
    connection_id: str = "",
    format: str = "markdown",
    channel_preference: str = "",
    recipient_chat_id: str = "",
    to_email: str = "",
    from_source: str = "system",
    subject: str = "",
    agent_name: str = "",
) -> Dict[str, Any]:
    """Send a message through a specific messaging channel.

    CHANNEL USAGE:

      Email:     channel="email", to_email="user@example.com", subject="Subject Line", message="body"
                 Optional: from_source="user" to send from user's own email (default: "system" = Bastion SMTP)
                 HTML email: set format="html" when message is HTML, or start body with <!DOCTYPE html> for auto-detection

      Telegram:  channel="telegram", message="body"
                 Optional: recipient_chat_id="123456" (default: last user who messaged the bot)

      Discord:   channel="discord", message="body"
                 Optional: recipient_chat_id="channel_id" (default: bot's default channel)

      In-app:    channel="in_app", message="body"

    Args:
        message: Message content (required, max 4000 chars).
        channel: One of: email, telegram, discord, in_app. Default: telegram.
        to_email: Recipient email address (email channel only; empty = user's notification preferences).
        subject: Email subject line (email channel only; empty = auto-generated from message).
        from_source: Email sender: "system" (Bastion SMTP, default) or "user" (user's own email connection).
        recipient_chat_id: Telegram chat_id or Discord channel ID (optional; not used for email/in_app).
        format: Message format: markdown, plain, html. Default: markdown.
        connection_id: Specific provider connection; empty = default for that provider.
    """
    message = (message or "").strip()
    if not message:
        return {
            "success": False,
            "message_id": "",
            "channel": channel_preference or channel,
            "error": "Message content is required. Wire the 'message' input to an upstream step (e.g. {get_weather.formatted}).",
            "formatted": "Send failed: message is empty. In step config, set message to e.g. {get_weather.formatted}.",
        }
    provider = channel_preference or channel
    if provider not in ALL_CHANNELS:
        return {
            "success": False,
            "message_id": "",
            "channel": provider,
            "error": f"Unsupported channel: {provider}. Supported: {', '.join(sorted(ALL_CHANNELS))}",
            "formatted": f"Failed: unsupported channel '{provider}'",
        }

    if provider == "in_app":
        try:
            client = await get_backend_tool_client()
            result = await client.create_agent_conversation(
                user_id=user_id,
                message=message[:4000],
                agent_name=agent_name or "",
                agent_profile_id="",
                title="",
                conversation_id="",
            )
            success = result.get("success", False)
            convo_id = result.get("conversation_id", "")
            msg_id = result.get("message_id", "")
            err = result.get("error") if not success else None
            if success:
                formatted = f"In-app message delivered (conversation: {convo_id})"
                if msg_id:
                    formatted += f" (message ID: {msg_id})"
            else:
                formatted = f"Failed to create in-app message: {err or 'unknown error'}"
            return {
                "success": success,
                "message_id": msg_id,
                "channel": "in_app",
                "error": err,
                "formatted": formatted,
            }
        except Exception as e:
            logger.exception("create_agent_conversation failed: %s", e)
            return {
                "success": False,
                "message_id": "",
                "channel": "in_app",
                "error": str(e),
                "formatted": f"Failed to create in-app message: {e}",
            }

    if provider == "email":
        if _send_email_tool is None:
            return {
                "success": False,
                "message_id": "",
                "channel": "email",
                "error": "Email tools not available",
                "formatted": "Failed: email tools not available",
            }
        recipient = (to_email or "").strip()
        if not recipient:
            try:
                client = await get_backend_tool_client()
                prefs = await client.get_user_notification_preferences(user_id)
                recipient = (prefs.get("email_address") or "").strip()
            except Exception:
                pass
        if not recipient:
            return {
                "success": False,
                "message_id": "",
                "channel": "email",
                "error": "No recipient. Set to_email or configure email in user notification preferences.",
                "formatted": "Failed: no email recipient (set to_email or user notification preferences)",
            }
        email_subject = (subject or "").strip()
        if not email_subject:
            email_subject = message[:60] + ("..." if len(message) > 60 else "")
        source = (from_source or "system").strip().lower() or "system"
        # Detect HTML: explicit format param or body content detection
        is_html = (format or "").strip().lower() == "html" or _detect_html_body(message)
        email_result = await _send_email_tool(
            user_id=user_id,
            to=[recipient],
            subject=email_subject,
            body=message,
            confirmed=True,
            from_source=source,
            body_is_html=is_html,
        )
        success = isinstance(email_result, dict) and email_result.get("success", False)
        msg_id = email_result.get("message_id", "") if isinstance(email_result, dict) else ""
        err = email_result.get("error") if isinstance(email_result, dict) and not success else None
        if success:
            formatted = f"Email delivered to {recipient}"
            if msg_id:
                formatted += f" (ID: {msg_id})"
        else:
            formatted = f"Failed to send email: {err or 'unknown error'}"
        return {
            "success": success,
            "message_id": str(msg_id) if msg_id else "",
            "channel": "email",
            "error": err,
            "formatted": formatted,
        }

    client = await get_backend_tool_client()
    result = await client.send_outbound_message(
        user_id=user_id,
        provider=provider,
        connection_id=connection_id,
        message=message[:4000],
        format=format,
        recipient_chat_id=recipient_chat_id or "",
    )

    success = result.get("success", False)
    error = result.get("error")
    msg_id = result.get("message_id", "")

    if success:
        formatted = f"Message delivered via {provider}"
        if msg_id:
            formatted += f" (ID: {msg_id})"
    else:
        formatted = f"Failed to send via {provider}: {error or 'unknown error'}"

    return {
        "success": success,
        "message_id": msg_id,
        "channel": provider,
        "error": error,
        "formatted": formatted,
    }


async def _resolve_channels(
    channels: List[str],
    user_id: str,
) -> List[str]:
    """Resolve 'default' to user's configured channels; deduplicate and validate."""
    resolved: List[str] = []

    if "default" in channels:
        try:
            client = await get_backend_tool_client()
            prefs = await client.get_user_notification_preferences(user_id)
            default_channels = prefs.get("default_channels", ["in_app"])
            resolved.extend(default_channels)
        except Exception:
            logger.warning("Failed to load notification preferences for %s; falling back to in_app", user_id)
            resolved.append("in_app")

    for ch in channels:
        if ch != "default":
            resolved.append(ch)

    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for ch in resolved:
        if ch not in seen:
            seen.add(ch)
            unique.append(ch)

    # Ensure in_app is always included
    if "in_app" not in unique:
        unique.insert(0, "in_app")

    return unique


async def notify_user_tool(
    message: str,
    user_id: str = "system",
    title: str = "",
    channels: List[str] = None,
    urgency: str = "normal",
    conversation_id: str = "",
    agent_name: str = "",
    agent_profile_id: str = "",
    email_from: str = "system",
) -> Dict[str, Any]:
    """Send a proactive notification to the user via one or more channels.

    Always creates an in-app conversation record (the ledger). Then delivers
    to each configured external channel in parallel. Returns per-channel
    delivery results.

    Args:
        message: Message content to send.
        user_id: Target user ID.
        title: Conversation title (auto-generated if empty).
        channels: Delivery channels list (default: ["default"]).
        urgency: low, normal, high.
        conversation_id: Append to existing conversation, or empty for new.
        agent_name: Agent attribution name.
        agent_profile_id: Agent Factory profile ID.
    """
    if channels is None:
        channels = ["default"]

    resolved_channels = await _resolve_channels(channels, user_id)

    # Step 1: Always create/append in-app conversation (the ledger)
    client = await get_backend_tool_client()
    convo_result = await client.create_agent_conversation(
        user_id=user_id,
        message=message,
        agent_name=agent_name,
        agent_profile_id=agent_profile_id,
        title=title,
        conversation_id=conversation_id,
    )

    convo_id = convo_result.get("conversation_id", "")
    msg_id = convo_result.get("message_id", "")
    convo_success = convo_result.get("success", False)

    deliveries: List[Dict[str, Any]] = []

    # in_app is already done via conversation creation
    deliveries.append({
        "channel": "in_app",
        "success": convo_success,
        "message_id": msg_id,
        "error": convo_result.get("error") if not convo_success else None,
    })

    # Step 2: Fan out to external channels in parallel
    external = [ch for ch in resolved_channels if ch in EXTERNAL_CHANNELS]

    async def _deliver_external(ch: str) -> Dict[str, Any]:
        result = await send_channel_message_tool(
            message=message,
            user_id=user_id,
            channel=ch,
            format="markdown",
        )
        return {
            "channel": ch,
            "success": result.get("success", False),
            "message_id": result.get("message_id"),
            "error": result.get("error"),
        }

    if external:
        external_results = await asyncio.gather(
            *[_deliver_external(ch) for ch in external],
            return_exceptions=True,
        )
        for i, res in enumerate(external_results):
            if isinstance(res, Exception):
                deliveries.append({
                    "channel": external[i],
                    "success": False,
                    "message_id": None,
                    "error": str(res),
                })
            else:
                deliveries.append(res)

    # Step 3: Handle email channel (requires user's email from backend)
    if "email" in resolved_channels:
        try:
            if _send_email_tool is None:
                deliveries.append({
                    "channel": "email",
                    "success": False,
                    "message_id": None,
                    "error": "Email tools not available",
                })
            else:
                user_email = ""
                try:
                    prefs = await client.get_user_notification_preferences(user_id)
                    user_email = prefs.get("email_address", "")
                except Exception:
                    pass

                if not user_email:
                    deliveries.append({
                        "channel": "email",
                        "success": False,
                        "message_id": None,
                        "error": "No email address configured in user notification preferences",
                    })
                else:
                    email_title = title or (message[:60] + "..." if len(message) > 60 else message)
                    email_result = await _send_email_tool(
                        user_id=user_id,
                        to=[user_email],
                        subject=f"[{agent_name or 'Agent'}] {email_title}",
                        body=message,
                        confirmed=True,  # Bypass HITL for proactive notifications
                        from_source=(email_from or "system").strip().lower() or "system",
                    )
                    email_success = isinstance(email_result, dict) and email_result.get("success", False)
                    deliveries.append({
                        "channel": "email",
                        "success": email_success,
                        "message_id": email_result.get("message_id") if isinstance(email_result, dict) else None,
                        "error": email_result.get("error") if isinstance(email_result, dict) and not email_success else None,
                    })
        except Exception as e:
            logger.warning("Email notification failed: %s", e)
            deliveries.append({
                "channel": "email",
                "success": False,
                "message_id": None,
                "error": str(e),
            })

    all_succeeded = all(d["success"] for d in deliveries)

    # Build formatted summary
    parts = []
    for d in deliveries:
        status = "delivered" if d["success"] else f"failed ({d.get('error', 'unknown')})"
        parts.append(f"  {d['channel']}: {status}")
    summary_lines = "\n".join(parts)
    formatted = f"Notification sent to {len(deliveries)} channel(s):\n{summary_lines}"
    if convo_id:
        formatted += f"\nConversation: {convo_id}"

    return {
        "conversation_id": convo_id,
        "message_id": msg_id,
        "deliveries": deliveries,
        "all_succeeded": all_succeeded,
        "formatted": formatted,
    }


async def schedule_reminder_tool(
    message: str,
    user_id: str = "system",
    delay_seconds: int = 3600,
    channels: List[str] = None,
) -> Dict[str, Any]:
    """Schedule a time-delayed notification to the user.

    When the delay elapses, the user receives the message via the configured
    channels (same as notify_user). Requires backend support for scheduled
    reminders; if not available, returns success=False with an explanatory message.

    Args:
        message: Message content to send when the reminder fires.
        user_id: Target user ID.
        delay_seconds: Seconds until the reminder fires (default 3600 = 1 hour).
        channels: Delivery channels when reminder fires (default: ["default"]).
    """
    if channels is None:
        channels = ["default"]
    try:
        client = await get_backend_tool_client()
        if hasattr(client, "schedule_reminder") and callable(getattr(client, "schedule_reminder")):
            result = await client.schedule_reminder(
                user_id=user_id,
                message=message[:4000],
                delay_seconds=delay_seconds,
                channels=channels,
            )
            success = result.get("success", False)
            reminder_id = result.get("reminder_id")
            scheduled_at = result.get("scheduled_at", "")
            err = result.get("error")
            if success:
                formatted = f"Reminder scheduled for {scheduled_at or 'later'} (ID: {reminder_id or 'n/a'})."
            else:
                formatted = f"Scheduling failed: {err or 'unknown error'}."
            return {
                "success": success,
                "reminder_id": reminder_id,
                "scheduled_at": scheduled_at,
                "error": err,
                "formatted": formatted,
            }
    except Exception as e:
        logger.warning("schedule_reminder not implemented by backend: %s", e)
    from datetime import datetime, timezone
    from datetime import timedelta
    # Stub: no backend support yet; return structured result so playbooks can wire the step
    at = (datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)).isoformat()
    formatted = (
        f"Scheduled reminders require backend support (not yet implemented). "
        f"Would have sent in {delay_seconds}s at {at}. Use notify_user for immediate delivery."
    )
    return {
        "success": False,
        "reminder_id": None,
        "scheduled_at": at,
        "error": "Backend schedule_reminder not implemented",
        "formatted": formatted,
    }


# ── Tool Lists and Registry ─────────────────────────────────────────────────

NOTIFICATION_TOOLS = [notify_user_tool, send_channel_message_tool, schedule_reminder_tool]

register_action(
    name="notify_user",
    category="notifications",
    description="Send proactive message to user via one or more channels (always creates in-app conversation)",
    inputs_model=NotifyUserInputs,
    params_model=NotifyUserParams,
    outputs_model=NotifyUserOutputs,
    tool_function=notify_user_tool,
    retriable=False,
)

register_action(
    name="send_channel_message",
    category="notifications",
    description="Send message via a specific channel (in-app conversation, Telegram, Discord, Email)",
    inputs_model=SendChannelMessageInputs,
    params_model=SendChannelMessageParams,
    outputs_model=SendChannelMessageOutputs,
    tool_function=send_channel_message_tool,
    retriable=False,
)

register_action(
    name="schedule_reminder",
    category="notifications",
    description="Schedule a time-delayed notification (requires backend support)",
    inputs_model=ScheduleReminderInputs,
    params_model=ScheduleReminderParams,
    outputs_model=ScheduleReminderOutputs,
    tool_function=schedule_reminder_tool,
    retriable=False,
)
