"""
Email Tools - LangGraph tools for email operations via connections-service.
Returns formatted strings for LLM consumption.
"""

import logging
from typing import Any, Dict, List, Optional

from clients.connections_service_client import get_connections_service_client

logger = logging.getLogger(__name__)

_email_tools_instance: Optional[Any] = None


async def _get_email_client():
    global _email_tools_instance
    if _email_tools_instance is None:
        _email_tools_instance = await get_connections_service_client()
    return _email_tools_instance


def _format_emails(messages: List[Dict[str, Any]], max_preview: int = 200) -> str:
    lines = []
    for i, m in enumerate(messages, 1):
        preview = (m.get("body_preview") or "")[:max_preview]
        if len(m.get("body_preview") or "") > max_preview:
            preview += "..."
        msg_id = m.get("id") or m.get("message_id") or ""
        conv_id = m.get("conversation_id") or ""
        block = (
            f"{i}. message_id: {msg_id}\n"
            f"   From: {m.get('from_name') or m.get('from_address')} <{m.get('from_address')}>\n"
            f"   Subject: {m.get('subject')}\n"
            f"   Date: {m.get('received_datetime')}\n"
            f"   Read: {m.get('is_read')}\n"
            f"   Preview: {preview}"
        )
        if conv_id:
            block += f"\n   conversation_id: {conv_id}"
        lines.append(block)
    if lines:
        lines.append("")
        lines.append("To read an email in full, call read_email with the message_id above (not the list number).")
    return "\n".join(lines) if lines else "No emails found."


async def read_recent_emails(
    user_id: str,
    folder: str = "inbox",
    count: int = 10,
    unread_only: bool = False,
    connection_id: Optional[int] = None,
) -> str:
    """Read recent emails from the user's connected account. Returns a formatted summary for the LLM."""
    try:
        client = await _get_email_client()
        result = await client.get_emails(
            user_id=user_id,
            connection_id=connection_id,
            folder_id=folder,
            top=count,
            skip=0,
            unread_only=unread_only,
        )
        if result.get("error") and not result.get("messages"):
            return f"Error: {result.get('error', 'Failed to fetch emails')}. Ensure an email connection is configured in Settings."
        messages = result.get("messages", [])
        return _format_emails(messages)
    except Exception as e:
        logger.exception("read_recent_emails failed: %s", e)
        return f"Error reading emails: {e}"


async def search_emails(
    user_id: str,
    query: str,
    from_address: Optional[str] = None,
    date_range: Optional[Dict[str, str]] = None,
    top: int = 20,
    connection_id: Optional[int] = None,
) -> str:
    """Search emails with optional filters. Returns a formatted summary for the LLM."""
    try:
        client = await _get_email_client()
        start_date = date_range.get("start") if date_range else None
        end_date = date_range.get("end") if date_range else None
        result = await client.search_emails(
            user_id=user_id,
            query=query,
            top=top,
            from_address=from_address,
            start_date=start_date,
            end_date=end_date,
            connection_id=connection_id,
        )
        if result.get("error") and not result.get("messages"):
            return f"Error: {result.get('error', 'Search failed')}. Ensure an email connection is configured."
        messages = result.get("messages", [])
        return _format_emails(messages)
    except Exception as e:
        logger.exception("search_emails failed: %s", e)
        return f"Error searching emails: {e}"


async def get_email_thread(
    user_id: str,
    conversation_id: str,
    connection_id: Optional[int] = None,
) -> str:
    """Get all messages in an email thread. Returns formatted content for the LLM."""
    try:
        client = await _get_email_client()
        result = await client.get_email_thread(
            user_id=user_id,
            conversation_id=conversation_id,
            connection_id=connection_id,
        )
        if result.get("error") and not result.get("messages"):
            return f"Error: {result.get('error')}. Ensure an email connection is configured."
        messages = result.get("messages", [])
        if not messages:
            return "No messages in this thread."
        lines = []
        for m in messages:
            msg_id = m.get("id") or m.get("message_id") or ""
            block = (
                f"message_id: {msg_id}\n"
                f"From: {m.get('from_name')} <{m.get('from_address')}>\n"
                f"Date: {m.get('received_datetime')}\n"
                f"Body: {m.get('body_content') or m.get('body_preview')}\n"
            )
            lines.append(block)
        return "\n---\n".join(lines)
    except Exception as e:
        logger.exception("get_email_thread failed: %s", e)
        return f"Error loading thread: {e}"


def _body_looks_like_html(body: str) -> bool:
    """Return True if body appears to be an HTML document."""
    if not body or not body.strip():
        return False
    stripped = body.strip().lower()
    return stripped.startswith("<!doctype") or stripped.startswith("<html")


async def send_email(
    user_id: str,
    to: List[str],
    subject: str,
    body: str,
    cc: Optional[List[str]] = None,
    from_source: str = "user",
    connection_id: Optional[int] = None,
    body_is_html: bool = False,
) -> str:
    """Send an email. from_source: 'system' = Bastion SMTP; 'user' = user's email connection. Returns success or error message."""
    try:
        is_html = body_is_html or _body_looks_like_html(body)
        if (from_source or "").strip().lower() == "system":
            from services.external_connections_service import external_connections_service
            from services.email_service import email_service

            rls_context = {"user_id": "", "user_role": "admin"}
            smtp_config = await external_connections_service.get_smtp_settings_for_sending(rls_context=rls_context)
            if not smtp_config:
                return "Failed to send email: System SMTP is not configured. Ask an admin to set up SMTP in Settings."
            to_str = ", ".join(to) if to else ""
            if not to_str:
                return "Failed to send email: No recipients specified."
            cc_list = list(cc) if cc else None
            if is_html:
                ok = await email_service.send_email(
                    to_email=to_str,
                    subject=subject,
                    body_text="",
                    body_html=body,
                    cc=cc_list,
                    smtp_config=smtp_config,
                )
            else:
                ok = await email_service.send_email(
                    to_email=to_str,
                    subject=subject,
                    body_text=body,
                    cc=cc_list,
                    smtp_config=smtp_config,
                )
            if ok:
                return f"Email sent successfully to {', '.join(to)} (from system)."
            return "Failed to send email via system SMTP."
        client = await _get_email_client()
        result = await client.send_email(
            user_id=user_id,
            to_recipients=to,
            subject=subject,
            body=body,
            cc_recipients=cc,
            connection_id=connection_id,
            body_is_html=is_html,
        )
        if result.get("success"):
            return f"Email sent successfully to {', '.join(to)}."
        return f"Failed to send email: {result.get('error', 'Unknown error')}"
    except Exception as e:
        logger.exception("send_email failed: %s", e)
        return f"Error sending email: {e}"


async def reply_to_email(
    user_id: str,
    message_id: str,
    body: str,
    reply_all: bool = False,
    connection_id: Optional[int] = None,
) -> str:
    """Reply to an email. Returns success or error message. Requires user approval (HITL) before sending."""
    try:
        client = await _get_email_client()
        result = await client.reply_to_email(
            user_id=user_id,
            message_id=message_id,
            body=body,
            reply_all=reply_all,
            connection_id=connection_id,
        )
        if result.get("success"):
            return "Reply sent successfully."
        return f"Failed to send reply: {result.get('error', 'Unknown error')}"
    except Exception as e:
        logger.exception("reply_to_email failed: %s", e)
        return f"Error sending reply: {e}"


async def mark_email_as_read(user_id: str, message_id: str) -> str:
    """Mark an email as read."""
    try:
        client = await _get_email_client()
        result = await client.update_email(
            user_id=user_id,
            message_id=message_id,
            is_read=True,
        )
        if result.get("success"):
            return "Email marked as read."
        return f"Failed: {result.get('error', 'Unknown error')}"
    except Exception as e:
        logger.exception("mark_email_as_read failed: %s", e)
        return f"Error: {e}"


async def get_email_statistics(
    user_id: str,
    connection_id: Optional[int] = None,
) -> str:
    """Get email statistics (total and unread count) for the user's inbox."""
    try:
        client = await _get_email_client()
        result = await client.get_email_statistics(
            user_id=user_id,
            folder_id="inbox",
            connection_id=connection_id,
        )
        if result.get("error") and "total_count" not in result:
            return f"Error: {result.get('error', 'Failed to get statistics')}"
        total = result.get("total_count", 0)
        unread = result.get("unread_count", 0)
        return f"Inbox: {total} total emails, {unread} unread."
    except Exception as e:
        logger.exception("get_email_statistics failed: %s", e)
        return f"Error: {e}"


async def summarize_unread_emails(user_id: str) -> str:
    """Fetch unread emails and return a short summary for the LLM."""
    try:
        client = await _get_email_client()
        result = await client.get_emails(
            user_id=user_id,
            folder_id="inbox",
            top=20,
            unread_only=True,
        )
        if result.get("error") and not result.get("messages"):
            return f"Error: {result.get('error')}. Ensure an email connection is configured."
        messages = result.get("messages", [])
        if not messages:
            return "No unread emails."
        return _format_emails(messages, max_preview=150)
    except Exception as e:
        logger.exception("summarize_unread_emails failed: %s", e)
        return f"Error: {e}"
