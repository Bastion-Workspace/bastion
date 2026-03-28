"""
Shared Pydantic models for tool I/O contracts.

Reused across document, web, and file tools so output models reference
the same types. Import from here in tool modules and in action_io_registry usage.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class DocumentRef(BaseModel):
    """Standard document reference — reused by document/search tools."""

    document_id: str = Field(description="Document ID")
    title: str = Field(description="Document title")
    filename: str = Field(description="Filename")
    file_type: str = Field(description="File extension or type")
    folder_path: str = Field(description="Folder path")
    relevance_score: float = Field(default=0.0, description="Search relevance score")
    content_preview: str = Field(default="", description="Short content preview")


class WebResult(BaseModel):
    """Single web search result — reused by web tools."""

    title: str = Field(description="Page title")
    url: str = Field(description="URL")
    snippet: str = Field(default="", description="Snippet or summary")


class FileRef(BaseModel):
    """Standard file reference — reused by file/folder tools."""

    document_id: str = Field(description="Document or file ID")
    title: str = Field(description="Title or name")
    file_type: str = Field(description="File type/extension")
    folder_path: str = Field(description="Folder path")
    scope: str = Field(default="my_docs", description="Scope: my_docs, team_docs, global_docs")
    modified_at: Optional[str] = Field(default=None, description="Last modified ISO 8601")


class TodoItem(BaseModel):
    """Standard todo/task reference — reused by org-capture and task tools."""

    todo_id: str = Field(description="Todo or task ID")
    title: str = Field(description="Title or headline")
    state: str = Field(description="State: todo, done, etc.")
    priority: Optional[str] = Field(default=None, description="Priority if set")
    deadline: Optional[str] = Field(default=None, description="Deadline ISO 8601 if set")
    tags: List[str] = Field(default_factory=list, description="Tags")
    file_path: str = Field(default="", description="Path to org file")


class PlanStep(BaseModel):
    """Single step in a self-managed LLM plan — reused by planning tools."""

    step_id: str = Field(description="Step identifier (e.g. step_1)")
    title: str = Field(description="What this step aims to accomplish")
    status: str = Field(
        default="pending",
        description="pending, in_progress, complete, or skipped",
    )
    result_summary: str = Field(
        default="",
        description="What was found or accomplished",
    )


class AccountRef(BaseModel):
    """Account reference — connection_id, provider, type, label for email/calendar/contacts."""

    connection_id: int = Field(description="Connection ID — pass to email/calendar/contact tools")
    provider: str = Field(description="Provider: microsoft, imap_smtp, caldav, org")
    type: str = Field(description="Type: email, calendar, contacts")
    label: str = Field(description="Human-readable label, e.g. Work (john@acme.com)")
    address: str = Field(default="", description="Email address or username")


class EmailRef(BaseModel):
    """Standard email reference — reused by email tools."""

    message_id: str = Field(description="Email message ID")
    subject: str = Field(description="Subject line")
    from_address: str = Field(description="From address")
    to_addresses: str = Field(default="", description="To addresses")
    date: Optional[str] = Field(default=None, description="Date ISO 8601")
    snippet: str = Field(default="", description="Short preview")
    thread_id: Optional[str] = Field(default=None, description="Thread ID if applicable")
    account_connection_id: Optional[int] = Field(
        default=None,
        description="Connection ID this email came from — use for replies/moves",
    )
    account_label: Optional[str] = Field(
        default=None,
        description="Account label, e.g. Work (john@acme.com)",
    )


class CalendarRef(BaseModel):
    """Standard calendar reference — reused by calendar tools."""

    id: str = Field(description="Calendar ID")
    name: str = Field(description="Calendar name")
    color: str = Field(default="", description="Hex or provider color")
    is_default: bool = Field(default=False, description="Whether this is the default calendar")
    can_edit: bool = Field(default=True, description="Whether the user can edit this calendar")


class CalendarEventRef(BaseModel):
    """Standard calendar event reference — reused by calendar tools."""

    event_id: str = Field(description="Event ID")
    subject: str = Field(description="Event subject")
    start_datetime: str = Field(description="Start ISO 8601")
    end_datetime: str = Field(description="End ISO 8601")
    location: str = Field(default="", description="Location")
    body_preview: str = Field(default="", description="Short body preview")
    organizer_email: str = Field(default="", description="Organizer email")
    organizer_name: str = Field(default="", description="Organizer name")
    is_all_day: bool = Field(default=False, description="Whether the event is all-day")
    recurrence: str = Field(default="", description="Recurrence description if any")
    calendar_id: str = Field(default="", description="Calendar ID")
    web_link: str = Field(default="", description="Web link to open event")
    account_connection_id: Optional[int] = Field(
        default=None,
        description="Connection ID this event came from",
    )
    account_label: Optional[str] = Field(
        default=None,
        description="Account label, e.g. Work Calendar",
    )


class ContactRef(BaseModel):
    """Standard contact reference — reused by contact tools (O365, org)."""

    contact_id: str = Field(description="Contact ID")
    display_name: str = Field(description="Display name")
    given_name: str = Field(default="", description="Given name")
    surname: str = Field(default="", description="Surname")
    email_addresses: List[str] = Field(default_factory=list, description="Email addresses")
    phone_numbers: List[str] = Field(default_factory=list, description="Phone numbers")
    company_name: str = Field(default="", description="Company name")
    job_title: str = Field(default="", description="Job title")
    birthday: Optional[str] = Field(default=None, description="Birthday ISO 8601")
    source: str = Field(default="microsoft", description="Source: microsoft or org")


class NotificationResult(BaseModel):
    """Standard notification delivery result for single-channel delivery (e.g. send_channel_message).
    For per-channel results inside notify_user_tool, notification_tools uses ChannelDeliveryResult
    (channel: str required, same other fields)."""

    success: bool = Field(description="Whether delivery succeeded")
    message_id: Optional[str] = Field(default=None, description="Provider message ID if applicable")
    channel: Optional[str] = Field(default=None, description="Channel used")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ImageRef(BaseModel):
    """Standard image reference — reused by image search/generation tools."""

    url: str = Field(description="Image URL or data URI")
    alt_text: Optional[str] = Field(default=None, description="Alt text or caption")
    image_type: str = Field(default="image", description="Type: image, thumbnail, etc.")
    metadata: Optional[dict] = Field(default=None, description="Title, date, series, author, etc.")


class SlackChannel(BaseModel):
    """Slack channel — reused by Slack plugin tools."""

    channel_id: str = Field(description="Slack channel ID")
    name: str = Field(description="Channel name")
    is_private: bool = Field(default=False, description="Whether the channel is private")
    num_members: int = Field(default=0, description="Number of members")
    topic: str = Field(default="", description="Channel topic")
    purpose: str = Field(default="", description="Channel purpose")


class SlackMessage(BaseModel):
    """Slack message — reused by Slack plugin tools."""

    ts: str = Field(description="Message timestamp (Slack ts)")
    user: str = Field(description="User ID of sender")
    text: str = Field(description="Message text")
    thread_ts: Optional[str] = Field(default=None, description="Thread timestamp if in thread")


class SlackUser(BaseModel):
    """Slack user — reused by Slack plugin tools."""

    user_id: str = Field(description="Slack user ID")
    name: str = Field(description="Username")
    real_name: str = Field(default="", description="Real name")
    display_name: str = Field(default="", description="Display name")
    is_bot: bool = Field(default=False, description="Whether the user is a bot")


class SlackSearchResult(BaseModel):
    """Slack search result — reused by slack_search_messages."""

    ts: str = Field(description="Message timestamp")
    channel_id: str = Field(description="Channel ID")
    channel_name: str = Field(default="", description="Channel name")
    user: str = Field(description="User ID")
    text: str = Field(description="Message text")
    permalink: Optional[str] = Field(default=None, description="Link to message")
