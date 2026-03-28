"""
Email Tools - Email operations via backend gRPC
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action
from orchestrator.utils.tool_type_models import EmailRef

logger = logging.getLogger(__name__)


class EmailOutputs(BaseModel):
    """Legacy minimal outputs; prefer specific output models below."""
    formatted: str = Field(description="Human-readable result")
    content: str = Field(default="", description="Raw or structured content")


class GetEmailsOutputs(BaseModel):
    """Outputs for get_emails_tool."""
    emails: List[EmailRef] = Field(default_factory=list, description="List of email refs")
    count: int = Field(description="Number of emails returned")
    folder: str = Field(description="Folder that was queried")
    formatted: str = Field(description="Human-readable result")
    content: str = Field(default="", description="Raw or structured content")


class SearchEmailsOutputs(BaseModel):
    """Outputs for search_emails_tool."""
    emails: List[EmailRef] = Field(default_factory=list, description="Matching emails")
    count: int = Field(description="Number of results")
    query_used: str = Field(description="Query that was executed")
    formatted: str = Field(description="Human-readable result")
    content: str = Field(default="", description="Raw or structured content")


class SendEmailOutputs(BaseModel):
    """Outputs for send_email_tool."""
    success: bool = Field(description="Whether send succeeded")
    message_id: Optional[str] = Field(default=None, description="Provider message ID if sent")
    formatted: str = Field(description="Human-readable result")
    content: str = Field(default="", description="Raw or structured content")


class GetEmailThreadOutputs(BaseModel):
    """Outputs for get_email_thread_tool."""
    thread_id: str = Field(default="", description="Conversation/thread ID")
    messages: List[EmailRef] = Field(default_factory=list, description="Messages in thread")
    count: int = Field(default=0, description="Number of messages")
    formatted: str = Field(description="Human-readable result")
    content: str = Field(default="", description="Raw or structured content")


class GetEmailStatisticsOutputs(BaseModel):
    """Outputs for get_email_statistics_tool."""
    total_messages: Optional[int] = Field(default=None, description="Total message count")
    unread_count: Optional[int] = Field(default=None, description="Unread message count")
    stats: Dict[str, Any] = Field(default_factory=dict, description="Raw stats from backend")
    formatted: str = Field(description="Human-readable result")
    content: str = Field(default="", description="Raw or structured content")


class ReplyToEmailOutputs(BaseModel):
    """Outputs for reply_to_email_tool."""
    success: bool = Field(description="Whether reply was sent")
    message_id: Optional[str] = Field(default=None, description="New message ID if sent")
    formatted: str = Field(description="Human-readable result")
    content: str = Field(default="", description="Raw or structured content")


def _to_list(value: Union[List[str], str]) -> List[str]:
    """Accept comma-separated string or list for 'to' / 'cc'."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    return []


def _parse_folders_from_result(result: str) -> List[Dict[str, Any]]:
    """Parse backend folder list string into list of folder dicts. Format: '- name (id=id, unread=N)'."""
    folders = []
    if not result or not isinstance(result, str):
        return folders
    pattern = re.compile(r"-\s*(.+?)\s*\(id=([^,)]+),\s*unread=(\d+)\)")
    for line in result.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if m:
            name, folder_id, unread = m.group(1).strip(), m.group(2).strip(), int(m.group(3))
            folders.append({"id": folder_id, "name": name, "unread_count": unread, "total_count": 0})
    return folders


def _parse_emails_from_result(result: Any) -> Tuple[List[Dict[str, Any]], str]:
    """Parse backend result into list of email dicts and display text."""
    if isinstance(result, str):
        return [], result
    items = result.get("emails") or result.get("items") or result.get("messages") or []
    if not items:
        text = result.get("formatted") or result.get("content") or str(result)
        return [], text
    emails = []
    for m in items:
        emails.append({
            "message_id": m.get("id") or m.get("message_id") or "",
            "subject": m.get("subject") or "",
            "from_address": m.get("from") or m.get("from_address") or "",
            "to_addresses": m.get("to") or m.get("to_addresses") or "",
            "date": m.get("date") or m.get("received_at"),
            "snippet": m.get("snippet") or m.get("body_preview") or "",
            "thread_id": m.get("conversation_id") or m.get("thread_id"),
        })
    text = result.get("formatted") or result.get("content") or str(result)
    return emails, text


def _stamp_emails_with_account(emails_list: List[Dict[str, Any]], connection_id: Optional[int]) -> None:
    """Stamp each email with account_connection_id for reply/move routing."""
    for email in emails_list:
        if connection_id is not None:
            email["account_connection_id"] = connection_id
        else:
            email["account_connection_id"] = None
        email["account_label"] = None


async def get_emails_tool(
    user_id: str = "system",
    folder: str = "inbox",
    top: int = 10,
    unread_only: bool = False,
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Get emails from a folder. Returns dict with emails, count, folder, formatted, content."""
    try:
        logger.info("get_emails: folder=%s top=%s", folder, top)
        client = await get_backend_tool_client()
        result = await client.get_emails(
            user_id=user_id,
            folder=folder,
            top=top,
            skip=0,
            unread_only=unread_only,
            connection_id=connection_id,
        )
        emails_list, text = _parse_emails_from_result(result)
        _stamp_emails_with_account(emails_list, connection_id)
        return {
            "emails": emails_list,
            "count": len(emails_list),
            "folder": folder,
            "formatted": text,
            "content": text,
        }
    except Exception as e:
        logger.error("get_emails_tool error: %s", e)
        err = str(e)
        return {"emails": [], "count": 0, "folder": folder, "formatted": f"Error: {err}", "content": err}


async def search_emails_tool(
    user_id: str = "system",
    query: str = "",
    top: int = 20,
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Search emails by query. Returns dict with emails, count, query_used, formatted, content."""
    try:
        logger.info("search_emails: query=%s", query[:80])
        client = await get_backend_tool_client()
        result = await client.search_emails(
            user_id=user_id,
            query=query,
            top=top,
            from_address="",
            connection_id=connection_id,
        )
        emails_list, text = _parse_emails_from_result(result)
        _stamp_emails_with_account(emails_list, connection_id)
        return {
            "emails": emails_list,
            "count": len(emails_list),
            "query_used": query,
            "formatted": text,
            "content": text,
        }
    except Exception as e:
        logger.error("search_emails_tool error: %s", e)
        err = str(e)
        return {"emails": [], "count": 0, "query_used": query, "formatted": f"Error: {err}", "content": err}


async def get_email_thread_tool(
    user_id: str = "system",
    conversation_id: str = "",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Get a full email thread by conversation ID. Returns dict with thread_id, messages, count, formatted, content."""
    try:
        logger.info("get_email_thread: conversation_id=%s", conversation_id[:50] if conversation_id else "")
        client = await get_backend_tool_client()
        result = await client.get_email_thread(
            user_id=user_id,
            conversation_id=conversation_id,
            connection_id=connection_id,
        )
        text = result if isinstance(result, str) else str(result)
        messages_list, _ = _parse_emails_from_result(result) if isinstance(result, dict) else ([], text)
        return {
            "thread_id": conversation_id,
            "messages": messages_list,
            "count": len(messages_list),
            "formatted": text,
            "content": text,
        }
    except Exception as e:
        logger.error("get_email_thread_tool error: %s", e)
        err = str(e)
        return {"thread_id": conversation_id, "messages": [], "count": 0, "formatted": f"Error: {err}", "content": err}


async def read_email_tool(
    user_id: str = "system",
    message_id: str = "",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Get a single email by message ID (full content). Returns dict with message_id, formatted, content."""
    try:
        if not message_id:
            msg = "Error: message_id is required."
            return {"message_id": "", "formatted": msg, "content": msg}
        logger.info("read_email: message_id=%s", message_id[:50] if message_id else "")
        client = await get_backend_tool_client()
        result = await client.get_email_by_id(
            user_id=user_id,
            message_id=message_id,
            connection_id=connection_id,
        )
        text = result if isinstance(result, str) else str(result)
        return {"message_id": message_id, "formatted": text, "content": text}
    except Exception as e:
        logger.error("read_email_tool error: %s", e)
        err = str(e)
        return {"message_id": message_id, "formatted": f"Error: {err}", "content": err}


async def move_email_tool(
    user_id: str = "system",
    message_id: str = "",
    destination_folder_id: str = "",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Move an email to a different folder. Returns dict with success, formatted, content."""
    try:
        if not message_id or not destination_folder_id:
            msg = "Error: message_id and destination_folder_id are required."
            return {"success": False, "formatted": msg, "content": msg}
        logger.info("move_email: message_id=%s dest=%s", message_id[:50] if message_id else "", destination_folder_id)
        client = await get_backend_tool_client()
        result = await client.move_email(
            user_id=user_id,
            message_id=message_id,
            destination_folder_id=destination_folder_id,
            connection_id=connection_id,
        )
        text = result if isinstance(result, str) else str(result)
        success = not (text.startswith("Error") or text.startswith("Failed"))
        return {"success": success, "formatted": text, "content": text}
    except Exception as e:
        logger.error("move_email_tool error: %s", e)
        err = str(e)
        return {"success": False, "formatted": f"Error: {err}", "content": err}


async def get_email_folders_tool(
    user_id: str = "system",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    """List mailbox folders (inbox, sent, drafts, custom). Returns dict with folders, formatted, content."""
    try:
        logger.info("get_email_folders")
        client = await get_backend_tool_client()
        result = await client.get_email_folders(
            user_id=user_id,
            connection_id=connection_id,
        )
        text = result if isinstance(result, str) else str(result)
        folders_list = _parse_folders_from_result(text)
        folder_refs = [
            EmailFolderRef(
                id=f.get("id", ""),
                name=f.get("name", ""),
                unread_count=f.get("unread_count", 0),
                total_count=f.get("total_count", 0),
            )
            for f in folders_list
        ]
        return {
            "folders": [f.model_dump() for f in folder_refs],
            "formatted": text,
            "content": text,
        }
    except Exception as e:
        logger.error("get_email_folders_tool error: %s", e)
        err = str(e)
        return {"folders": [], "formatted": f"Error: {err}", "content": err}


async def update_email_tool(
    user_id: str = "system",
    message_id: str = "",
    is_read: Optional[bool] = None,
    importance: Optional[str] = None,
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Update an email (mark read/unread, set importance). Returns dict with success, formatted, content."""
    try:
        if not message_id:
            msg = "Error: message_id is required."
            return {"success": False, "formatted": msg, "content": msg}
        if is_read is None and not importance:
            msg = "Error: at least one of is_read or importance is required."
            return {"success": False, "formatted": msg, "content": msg}
        logger.info("update_email: message_id=%s is_read=%s importance=%s", message_id[:50] if message_id else "", is_read, importance)
        client = await get_backend_tool_client()
        result = await client.update_email(
            user_id=user_id,
            message_id=message_id,
            is_read=is_read,
            importance=importance,
            connection_id=connection_id,
        )
        text = result if isinstance(result, str) else str(result)
        success = not (text.startswith("Error") or text.startswith("Failed"))
        return {"success": success, "formatted": text, "content": text}
    except Exception as e:
        logger.error("update_email_tool error: %s", e)
        err = str(e)
        return {"success": False, "formatted": f"Error: {err}", "content": err}


async def create_draft_tool(
    user_id: str = "system",
    to: Union[List[str], str] = None,
    subject: str = "",
    body: str = "",
    cc: Optional[Union[List[str], str]] = None,
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a draft email (do not send). Returns dict with success, message_id, formatted, content."""
    try:
        to_list = _to_list(to or [])
        if not to_list:
            msg = "Error: at least one recipient (to) is required."
            return {"success": False, "message_id": None, "formatted": msg, "content": msg}
        cc_list = _to_list(cc) if cc else []
        logger.info("create_draft: to=%s subject=%s", to_list, subject[:50] if subject else "")
        client = await get_backend_tool_client()
        result = await client.create_draft(
            user_id=user_id,
            to=to_list,
            subject=subject,
            body=body,
            cc=cc_list if cc_list else None,
            connection_id=connection_id,
        )
        text = result if isinstance(result, str) else str(result)
        success = not (text.startswith("Error") or text.startswith("Failed"))
        message_id = None
        if success and "ID:" in text:
            m = re.search(r"ID:\s*([A-Za-z0-9_-]+)", text)
            if m:
                message_id = m.group(1)
        return {"success": success, "message_id": message_id, "formatted": text, "content": text}
    except Exception as e:
        logger.error("create_draft_tool error: %s", e)
        err = str(e)
        return {"success": False, "message_id": None, "formatted": f"Error: {err}", "content": err}


async def get_email_statistics_tool(
    user_id: str = "system",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Get email statistics. Returns dict with total_messages, unread_count, stats, formatted, content."""
    try:
        logger.info("get_email_statistics")
        client = await get_backend_tool_client()
        result = await client.get_email_statistics(user_id=user_id, connection_id=connection_id)
        text = result if isinstance(result, str) else str(result)
        stats = result if isinstance(result, dict) else {}
        return {
            "total_messages": stats.get("total_messages") or stats.get("total"),
            "unread_count": stats.get("unread_count") or stats.get("unread"),
            "stats": stats if isinstance(result, dict) else {},
            "formatted": text,
            "content": text,
        }
    except Exception as e:
        logger.error("get_email_statistics_tool error: %s", e)
        err = str(e)
        return {"total_messages": None, "unread_count": None, "stats": {}, "formatted": f"Error: {err}", "content": err}


async def send_email_tool(
    user_id: str = "system",
    to: Union[List[str], str] = None,
    subject: str = "",
    body: str = "",
    confirmed: bool = False,
    from_source: str = "user",
    connection_id: Optional[int] = None,
    body_is_html: bool = False,
) -> Dict[str, Any]:
    """Send an email. from_source: 'system' = Bastion SMTP, 'user' = user's email connection. Returns dict with success, message_id, formatted, content."""
    try:
        to_list = _to_list(to or [])
        if not to_list:
            msg = "Error: at least one recipient (to) is required."
            return {"success": False, "message_id": None, "formatted": msg, "content": msg}
        if not confirmed:
            preview = (
                "[DRAFT - not sent yet]\n"
                f"To: {', '.join(to_list)}\n"
                f"Subject: {subject}\n\n"
                f"Body:\n{body[:2000]}{'...' if len(body) > 2000 else ''}\n\n"
                "Reply with 'yes', 'send', or 'approve' to send this email. Otherwise say no to cancel."
            )
            return {"success": False, "message_id": None, "formatted": preview, "content": preview}
        logger.info("send_email: to=%s subject=%s from_source=%s", to_list, subject[:50], from_source)
        client = await get_backend_tool_client()
        result = await client.send_email(
            user_id=user_id,
            to=to_list,
            subject=subject,
            body=body,
            cc=None,
            from_source=(from_source or "user").strip().lower() or "user",
            connection_id=connection_id,
            body_is_html=body_is_html,
        )
        if isinstance(result, dict):
            text = result.get("formatted", result.get("content", str(result)))
            success = result.get("success", True)
            message_id = result.get("message_id") or result.get("id")
        else:
            text = str(result)
            success = not (
                text.startswith("Error") or text.startswith("Failed") or text.startswith("Send failed")
            )
            message_id = None
        return {"success": success, "message_id": message_id, "formatted": text, "content": text}
    except Exception as e:
        logger.error("send_email_tool error: %s", e)
        err = str(e)
        return {"success": False, "message_id": None, "formatted": f"Error: {err}", "content": err}


async def reply_to_email_tool(
    user_id: str = "system",
    message_id: str = "",
    body: str = "",
    reply_all: bool = False,
    confirmed: bool = False,
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Reply to an email. Returns dict with formatted and content."""
    try:
        if not message_id:
            msg = "Error: message_id is required."
            return {"success": False, "message_id": None, "formatted": msg, "content": msg}
        if not confirmed:
            preview = (
                "[DRAFT - not sent yet]\n"
                f"Replying to message_id: {message_id[:50]}...\n"
                f"Reply all: {reply_all}\n\n"
                f"Body:\n{body[:2000]}{'...' if len(body) > 2000 else ''}\n\n"
                "Reply with 'yes', 'send', or 'approve' to send this reply. Otherwise say no to cancel."
            )
            return {"success": False, "message_id": None, "formatted": preview, "content": preview}
        logger.info("reply_to_email: message_id=%s", message_id[:50])
        client = await get_backend_tool_client()
        result = await client.reply_to_email(
            user_id=user_id,
            message_id=message_id,
            body=body,
            reply_all=reply_all,
            connection_id=connection_id,
        )
        text = result if isinstance(result, str) else str(result)
        success = isinstance(result, dict) and result.get("success", True)
        new_id = (result.get("message_id") or result.get("id")) if isinstance(result, dict) else None
        return {"success": success, "message_id": new_id, "formatted": text, "content": text}
    except Exception as e:
        logger.error("reply_to_email_tool error: %s", e)
        err = str(e)
        return {"success": False, "message_id": None, "formatted": f"Error: {err}", "content": err}


class GetEmailsInputs(BaseModel):
    folder: str = Field(default="inbox", description="inbox, sent, drafts")
    top: int = Field(default=10, description="Max emails to return")
    unread_only: bool = Field(default=False, description="Only unread")


class SearchEmailsInputs(BaseModel):
    query: str = Field(description="Search query")
    top: int = Field(default=20, description="Max results")


class GetEmailThreadInputs(BaseModel):
    conversation_id: str = Field(description="Thread ID from email list")


class ReadEmailInputs(BaseModel):
    message_id: str = Field(
        description="The message_id string from get_emails or search_emails output (the long ID, not the list number 1, 2, 3)"
    )


class ReadEmailOutputs(BaseModel):
    message_id: str = Field(description="The message ID that was read")
    formatted: str = Field(description="Human-readable full email content")
    content: str = Field(default="", description="Raw or structured content")


class MoveEmailInputs(BaseModel):
    message_id: str = Field(description="Email message ID to move")
    destination_folder_id: str = Field(description="Target folder ID (from get_email_folders)")


class MoveEmailOutputs(BaseModel):
    success: bool = Field(description="Whether move succeeded")
    formatted: str = Field(description="Human-readable result")
    content: str = Field(default="", description="Raw or structured content")


class EmailFolderRef(BaseModel):
    """Reference to an email folder (inbox, sent, custom)."""
    id: str = Field(description="Folder ID for move_email etc.")
    name: str = Field(description="Display name")
    unread_count: int = Field(default=0, description="Unread message count")
    total_count: int = Field(default=0, description="Total message count")


class GetEmailFoldersInputs(BaseModel):
    pass


class GetEmailFoldersOutputs(BaseModel):
    folders: List[EmailFolderRef] = Field(default_factory=list, description="List of mailbox folders")
    formatted: str = Field(description="Human-readable list of folders")
    content: str = Field(default="", description="Raw or structured content")


class UpdateEmailInputs(BaseModel):
    message_id: str = Field(description="Email message ID to update")
    is_read: Optional[bool] = Field(default=None, description="Set read state (true/false); omit to leave unchanged")
    importance: Optional[str] = Field(default=None, description="Set importance: low, normal, high; omit to leave unchanged")


class UpdateEmailOutputs(BaseModel):
    success: bool = Field(description="Whether update succeeded")
    formatted: str = Field(description="Human-readable result")
    content: str = Field(default="", description="Raw or structured content")


class CreateDraftInputs(BaseModel):
    to: str = Field(description="Recipients (comma-separated)")
    subject: str = Field(description="Subject")
    body: str = Field(description="Body")


class CreateDraftOutputs(BaseModel):
    success: bool = Field(description="Whether draft was created")
    message_id: Optional[str] = Field(default=None, description="Draft message ID if created")
    formatted: str = Field(description="Human-readable result")
    content: str = Field(default="", description="Raw or structured content")


class SendEmailInputs(BaseModel):
    to: str = Field(description="Recipients (comma-separated)")
    subject: str = Field(description="Subject")
    body: str = Field(description="Body (plain text or HTML)")
    confirmed: bool = Field(default=False, description="True to send, False for draft")
    from_source: str = Field(default="user", description="'system' = Bastion SMTP, 'user' = user's email connection")
    body_is_html: bool = Field(default=False, description="Set True when body is HTML; auto-detected if body starts with <!DOCTYPE or <html")


class ReplyToEmailInputs(BaseModel):
    message_id: str = Field(description="Message ID to reply to")
    body: str = Field(description="Reply body")
    reply_all: bool = Field(default=False, description="Reply to all")
    confirmed: bool = Field(default=False, description="True to send")


class GetEmailStatisticsInputs(BaseModel):
    pass


register_action(name="list_emails", category="email", description="Get emails from a folder", inputs_model=GetEmailsInputs, outputs_model=GetEmailsOutputs, tool_function=get_emails_tool)
register_action(name="search_emails", category="email", description="Search emails", inputs_model=SearchEmailsInputs, outputs_model=SearchEmailsOutputs, tool_function=search_emails_tool)
register_action(name="get_email_thread", category="email", description="Get full email thread", inputs_model=GetEmailThreadInputs, outputs_model=GetEmailThreadOutputs, tool_function=get_email_thread_tool)
register_action(name="read_email", category="email", description="Get a single email by message ID (full content)", inputs_model=ReadEmailInputs, outputs_model=ReadEmailOutputs, tool_function=read_email_tool)
register_action(name="move_email", category="email", description="Move an email to a different folder", inputs_model=MoveEmailInputs, outputs_model=MoveEmailOutputs, tool_function=move_email_tool, retriable=False)
register_action(name="list_email_folders", category="email", description="List mailbox folders (inbox, sent, drafts, custom)", inputs_model=GetEmailFoldersInputs, outputs_model=GetEmailFoldersOutputs, tool_function=get_email_folders_tool)
register_action(name="update_email", category="email", description="Update an email (mark read/unread, set importance)", inputs_model=UpdateEmailInputs, outputs_model=UpdateEmailOutputs, tool_function=update_email_tool, retriable=False)
register_action(name="create_draft", category="email", description="Create a draft email (do not send)", inputs_model=CreateDraftInputs, outputs_model=CreateDraftOutputs, tool_function=create_draft_tool, retriable=False)
register_action(name="get_email_statistics", category="email", description="Get email statistics", inputs_model=GetEmailStatisticsInputs, outputs_model=GetEmailStatisticsOutputs, tool_function=get_email_statistics_tool)
register_action(name="send_email", category="email", description="Send an email", inputs_model=SendEmailInputs, outputs_model=SendEmailOutputs, tool_function=send_email_tool, retriable=False)
register_action(name="reply_to_email", category="email", description="Reply to an email", inputs_model=ReplyToEmailInputs, outputs_model=ReplyToEmailOutputs, tool_function=reply_to_email_tool, retriable=False)
