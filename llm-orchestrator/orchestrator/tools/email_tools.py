"""
Email Tools - Email operations via backend gRPC
"""

import logging
from typing import List, Union

from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)


def _to_list(value: Union[List[str], str]) -> List[str]:
    """Accept comma-separated string or list for 'to' / 'cc'."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    return []


async def get_emails_tool(
    user_id: str = "system",
    folder: str = "inbox",
    top: int = 10,
    unread_only: bool = False,
) -> str:
    """
    Get emails from a folder.

    Args:
        user_id: User ID (injected by engine if omitted).
        folder: Folder name, e.g. inbox, sent, drafts.
        top: Maximum number of emails to return.
        unread_only: If true, return only unread emails.

    Returns:
        Formatted list of emails.
    """
    try:
        logger.info("get_emails: folder=%s top=%s", folder, top)
        client = await get_backend_tool_client()
        return await client.get_emails(
            user_id=user_id,
            folder=folder,
            top=top,
            skip=0,
            unread_only=unread_only,
        )
    except Exception as e:
        logger.error("get_emails_tool error: %s", e)
        return f"Error: {str(e)}"


async def search_emails_tool(
    user_id: str = "system",
    query: str = "",
    top: int = 20,
) -> str:
    """
    Search emails by query.

    Args:
        user_id: User ID (injected by engine if omitted).
        query: Search query (subject, body, sender).
        top: Maximum number of results.

    Returns:
        Formatted search results.
    """
    try:
        logger.info("search_emails: query=%s", query[:80])
        client = await get_backend_tool_client()
        return await client.search_emails(
            user_id=user_id,
            query=query,
            top=top,
            from_address="",
        )
    except Exception as e:
        logger.error("search_emails_tool error: %s", e)
        return f"Error: {str(e)}"


async def get_email_thread_tool(
    user_id: str = "system",
    conversation_id: str = "",
) -> str:
    """
    Get a full email thread by conversation ID.

    Args:
        user_id: User ID (injected by engine if omitted).
        conversation_id: Thread/conversation ID from a previous email list.

    Returns:
        Formatted thread.
    """
    try:
        logger.info("get_email_thread: conversation_id=%s", conversation_id[:50] if conversation_id else "")
        client = await get_backend_tool_client()
        return await client.get_email_thread(
            user_id=user_id,
            conversation_id=conversation_id,
        )
    except Exception as e:
        logger.error("get_email_thread_tool error: %s", e)
        return f"Error: {str(e)}"


async def get_email_statistics_tool(user_id: str = "system") -> str:
    """
    Get email statistics (inbox total, unread count).

    Args:
        user_id: User ID (injected by engine if omitted).

    Returns:
        Formatted statistics.
    """
    try:
        logger.info("get_email_statistics")
        client = await get_backend_tool_client()
        return await client.get_email_statistics(user_id=user_id)
    except Exception as e:
        logger.error("get_email_statistics_tool error: %s", e)
        return f"Error: {str(e)}"


async def send_email_tool(
    user_id: str = "system",
    to: Union[List[str], str] = None,
    subject: str = "",
    body: str = "",
    confirmed: bool = False,
) -> str:
    """
    Send an email. Call with confirmed=False first to show a draft; after user approves, call with confirmed=True to send.

    Args:
        user_id: User ID (injected by engine if omitted).
        to: Recipient(s) - comma-separated string or list.
        subject: Subject line.
        body: Email body (plain text).
        confirmed: If False, return draft preview and ask user to approve. If True, send the email.

    Returns:
        Draft preview (when confirmed=False) or success/error message (when confirmed=True).
    """
    try:
        to_list = _to_list(to or [])
        if not to_list:
            return "Error: at least one recipient (to) is required."
        if not confirmed:
            preview = (
                "[DRAFT - not sent yet]\n"
                f"To: {', '.join(to_list)}\n"
                f"Subject: {subject}\n\n"
                f"Body:\n{body[:2000]}{'...' if len(body) > 2000 else ''}\n\n"
                "Reply with 'yes', 'send', or 'approve' to send this email. Otherwise say no to cancel."
            )
            return preview
        logger.info("send_email: to=%s subject=%s", to_list, subject[:50])
        client = await get_backend_tool_client()
        return await client.send_email(
            user_id=user_id,
            to=to_list,
            subject=subject,
            body=body,
            cc=None,
        )
    except Exception as e:
        logger.error("send_email_tool error: %s", e)
        return f"Error: {str(e)}"


async def reply_to_email_tool(
    user_id: str = "system",
    message_id: str = "",
    body: str = "",
    reply_all: bool = False,
    confirmed: bool = False,
) -> str:
    """
    Reply to an email by message ID. Call with confirmed=False first to show a draft; after user approves, call with confirmed=True to send.

    Args:
        user_id: User ID (injected by engine if omitted).
        message_id: ID of the message to reply to (from thread/list).
        body: Reply body (plain text).
        reply_all: If true, reply to all recipients.
        confirmed: If False, return draft preview and ask user to approve. If True, send the reply.

    Returns:
        Draft preview (when confirmed=False) or success/error message (when confirmed=True).
    """
    try:
        if not message_id:
            return "Error: message_id is required."
        if not confirmed:
            preview = (
                "[DRAFT - not sent yet]\n"
                f"Replying to message_id: {message_id[:50]}...\n"
                f"Reply all: {reply_all}\n\n"
                f"Body:\n{body[:2000]}{'...' if len(body) > 2000 else ''}\n\n"
                "Reply with 'yes', 'send', or 'approve' to send this reply. Otherwise say no to cancel."
            )
            return preview
        logger.info("reply_to_email: message_id=%s", message_id[:50])
        client = await get_backend_tool_client()
        return await client.reply_to_email(
            user_id=user_id,
            message_id=message_id,
            body=body,
            reply_all=reply_all,
        )
    except Exception as e:
        logger.error("reply_to_email_tool error: %s", e)
        return f"Error: {str(e)}"
