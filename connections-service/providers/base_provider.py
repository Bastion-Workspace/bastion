"""
Base provider interface for external connections (email, calendar, etc.).
All providers must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseProvider(ABC):
    """Abstract base class for external connection providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. 'microsoft', 'gmail')."""
        pass

    @abstractmethod
    async def get_emails(
        self,
        access_token: str,
        folder_id: str = "inbox",
        top: int = 50,
        skip: int = 0,
        filter_expr: Optional[str] = None,
        unread_only: bool = False,
    ) -> Dict[str, Any]:
        """Return list of email messages and total_count. Keys: messages (list), total_count (int), error (optional)."""
        pass

    @abstractmethod
    async def get_email_by_id(self, access_token: str, message_id: str) -> Dict[str, Any]:
        """Return single message or error. Keys: message (dict or None), error (optional)."""
        pass

    @abstractmethod
    async def get_email_thread(
        self, access_token: str, conversation_id: str
    ) -> Dict[str, Any]:
        """Return messages in thread. Keys: messages (list), error (optional)."""
        pass

    @abstractmethod
    async def search_emails(
        self,
        access_token: str,
        query: str,
        top: int = 50,
        from_address: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return search results. Keys: messages (list), error (optional)."""
        pass

    @abstractmethod
    async def send_email(
        self,
        access_token: str,
        to_recipients: List[str],
        subject: str,
        body: str,
        cc_recipients: Optional[List[str]] = None,
        bcc_recipients: Optional[List[str]] = None,
        body_is_html: bool = False,
    ) -> Dict[str, Any]:
        """Send email. Keys: success (bool), message_id (optional), error (optional)."""
        pass

    @abstractmethod
    async def reply_to_email(
        self,
        access_token: str,
        message_id: str,
        body: str,
        reply_all: bool = False,
        body_is_html: bool = False,
    ) -> Dict[str, Any]:
        """Reply to message. Keys: success (bool), message_id (optional), error (optional)."""
        pass

    @abstractmethod
    async def update_email(
        self,
        access_token: str,
        message_id: str,
        is_read: Optional[bool] = None,
        importance: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update message. Keys: success (bool), error (optional)."""
        pass

    @abstractmethod
    async def move_email(
        self,
        access_token: str,
        message_id: str,
        destination_folder_id: str,
    ) -> Dict[str, Any]:
        """Move message. Keys: success (bool), error (optional)."""
        pass

    @abstractmethod
    async def delete_email(self, access_token: str, message_id: str) -> Dict[str, Any]:
        """Delete message. Keys: success (bool), error (optional)."""
        pass

    @abstractmethod
    async def get_folders(self, access_token: str) -> Dict[str, Any]:
        """Return mailbox folders. Keys: folders (list of dicts with id, name, parent_id, unread_count, total_count), error (optional)."""
        pass

    @abstractmethod
    async def sync_folder(
        self,
        access_token: str,
        folder_id: str,
        delta_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Incremental sync. Keys: added (list), updated (list), deleted_ids (list), next_delta_token (str), error (optional)."""
        pass

    @abstractmethod
    async def get_email_statistics(
        self, access_token: str, folder_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return counts. Keys: total_count (int), unread_count (int), error (optional)."""
        pass
