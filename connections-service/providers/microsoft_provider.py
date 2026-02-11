"""
Microsoft Graph provider for email operations.
"""

import logging
from typing import Any, Dict, List, Optional

import httpx

from config.settings import settings
from providers.base_provider import BaseProvider

logger = logging.getLogger(__name__)

GRAPH_BASE = settings.MICROSOFT_GRAPH_BASE.rstrip("/")
TIMEOUT = settings.GRAPH_REQUEST_TIMEOUT


def _message_to_dict(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Graph API message to our normalized shape."""
    from_addr = msg.get("from", {}).get("emailAddress", {})
    to_list = msg.get("toRecipients", [])
    cc_list = msg.get("ccRecipients", [])
    body_obj = msg.get("body", {}) or {}
    return {
        "id": msg.get("id", ""),
        "conversation_id": msg.get("conversationId", ""),
        "subject": msg.get("subject", ""),
        "from_address": from_addr.get("address", ""),
        "from_name": from_addr.get("name", ""),
        "to_addresses": [r.get("emailAddress", {}).get("address", "") for r in to_list],
        "cc_addresses": [r.get("emailAddress", {}).get("address", "") for r in cc_list],
        "received_datetime": msg.get("receivedDateTime", ""),
        "is_read": msg.get("isRead", False),
        "has_attachments": msg.get("hasAttachments", False),
        "importance": msg.get("importance", "normal"),
        "body_preview": msg.get("bodyPreview", ""),
        "body_content": body_obj.get("content", ""),
    }


def _folder_to_dict(f: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Graph mailFolder to our shape."""
    return {
        "id": f.get("id", ""),
        "name": f.get("displayName", ""),
        "parent_id": f.get("parentFolderId", ""),
        "unread_count": f.get("unreadItemCount", 0),
        "total_count": f.get("totalItemCount", 0),
    }


class MicrosoftGraphProvider(BaseProvider):
    """Microsoft Graph API implementation for email."""

    @property
    def name(self) -> str:
        return "microsoft"

    def _headers(self, access_token: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

    async def _get(
        self, access_token: str, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        url = f"{GRAPH_BASE}{path}"
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(url, headers=self._headers(access_token), params=params)
            resp.raise_for_status()
            return resp.json()

    async def _post(
        self, access_token: str, path: str, json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        url = f"{GRAPH_BASE}{path}"
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(url, headers=self._headers(access_token), json=json or {})
            if resp.status_code in (200, 201, 202, 204):
                if resp.content:
                    return resp.json()
                return {}
            resp.raise_for_status()
            return resp.json() if resp.content else {}

    async def _patch(
        self, access_token: str, path: str, json: Dict[str, Any]
    ) -> None:
        url = f"{GRAPH_BASE}{path}"
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.patch(url, headers=self._headers(access_token), json=json)
            resp.raise_for_status()

    async def _delete(self, access_token: str, path: str) -> None:
        url = f"{GRAPH_BASE}{path}"
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.delete(url, headers=self._headers(access_token))
            resp.raise_for_status()

    async def get_emails(
        self,
        access_token: str,
        folder_id: str = "inbox",
        top: int = 50,
        skip: int = 0,
        filter_expr: Optional[str] = None,
        unread_only: bool = False,
    ) -> Dict[str, Any]:
        try:
            path = f"/me/mailFolders/{folder_id}/messages"
            params = {"$top": min(top, 1000), "$skip": skip}
            if unread_only:
                params["$filter"] = "isRead eq false"
            elif filter_expr:
                params["$filter"] = filter_expr
            params["$select"] = "id,conversationId,subject,from,toRecipients,ccRecipients,receivedDateTime,isRead,hasAttachments,importance,bodyPreview"
            data = await self._get(access_token, path, params)
            value = data.get("value", [])
            return {
                "messages": [_message_to_dict(m) for m in value],
                "total_count": len(value),
            }
        except httpx.HTTPStatusError as e:
            logger.exception("Graph get_emails failed: %s", e)
            return {"messages": [], "total_count": 0, "error": str(e)}
        except Exception as e:
            logger.exception("Graph get_emails error: %s", e)
            return {"messages": [], "total_count": 0, "error": str(e)}

    async def get_email_by_id(self, access_token: str, message_id: str) -> Dict[str, Any]:
        try:
            path = f"/me/messages/{message_id}"
            data = await self._get(access_token, path)
            return {"message": _message_to_dict(data)}
        except httpx.HTTPStatusError as e:
            logger.exception("Graph get_email_by_id failed: %s", e)
            return {"message": None, "error": str(e)}
        except Exception as e:
            logger.exception("Graph get_email_by_id error: %s", e)
            return {"message": None, "error": str(e)}

    async def get_email_thread(
        self, access_token: str, conversation_id: str
    ) -> Dict[str, Any]:
        try:
            path = "/me/messages"
            params = {
                "$filter": f"conversationId eq '{conversation_id}'",
                "$orderby": "receivedDateTime asc",
                "$top": 100,
            }
            data = await self._get(access_token, path, params)
            value = data.get("value", [])
            return {"messages": [_message_to_dict(m) for m in value]}
        except Exception as e:
            logger.exception("Graph get_email_thread error: %s", e)
            return {"messages": [], "error": str(e)}

    async def search_emails(
        self,
        access_token: str,
        query: str,
        top: int = 50,
        from_address: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            path = "/me/messages"
            params = {"$top": min(top, 1000), "$search": f'"{query}"'}
            filters = []
            if from_address:
                filters.append(f"from/emailAddress/address eq '{from_address}'")
            if start_date:
                filters.append(f"receivedDateTime ge {start_date}")
            if end_date:
                filters.append(f"receivedDateTime le {end_date}")
            if filters:
                params["$filter"] = " and ".join(filters)
            data = await self._get(access_token, path, params)
            value = data.get("value", [])
            return {"messages": [_message_to_dict(m) for m in value]}
        except Exception as e:
            logger.exception("Graph search_emails error: %s", e)
            return {"messages": [], "error": str(e)}

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
        try:
            to_list = [{"emailAddress": {"address": a}} for a in to_recipients]
            cc_list = [{"emailAddress": {"address": a}} for a in (cc_recipients or [])]
            bcc_list = [{"emailAddress": {"address": a}} for a in (bcc_recipients or [])]
            payload = {
                "message": {
                    "subject": subject,
                    "body": {"contentType": "HTML" if body_is_html else "Text", "content": body},
                    "toRecipients": to_list,
                    "ccRecipients": cc_list,
                    "bccRecipients": bcc_list,
                },
                "saveToSentItems": True,
            }
            await self._post(access_token, "/me/sendMail", json=payload)
            return {"success": True, "message_id": ""}
        except Exception as e:
            logger.exception("Graph send_email error: %s", e)
            return {"success": False, "error": str(e)}

    async def reply_to_email(
        self,
        access_token: str,
        message_id: str,
        body: str,
        reply_all: bool = False,
        body_is_html: bool = False,
    ) -> Dict[str, Any]:
        try:
            path = f"/me/messages/{message_id}/{'replyAll' if reply_all else 'reply'}"
            await self._post(access_token, path, json={"comment": body})
            return {"success": True, "message_id": message_id}
        except Exception as e:
            logger.exception("Graph reply_to_email error: %s", e)
            return {"success": False, "error": str(e)}

    async def update_email(
        self,
        access_token: str,
        message_id: str,
        is_read: Optional[bool] = None,
        importance: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            payload = {}
            if is_read is not None:
                payload["isRead"] = is_read
            if importance is not None:
                payload["importance"] = importance
            if payload:
                await self._patch(access_token, f"/me/messages/{message_id}", payload)
            return {"success": True}
        except Exception as e:
            logger.exception("Graph update_email error: %s", e)
            return {"success": False, "error": str(e)}

    async def move_email(
        self,
        access_token: str,
        message_id: str,
        destination_folder_id: str,
    ) -> Dict[str, Any]:
        try:
            await self._post(
                access_token,
                f"/me/messages/{message_id}/move",
                json={"destinationId": destination_folder_id},
            )
            return {"success": True}
        except Exception as e:
            logger.exception("Graph move_email error: %s", e)
            return {"success": False, "error": str(e)}

    async def delete_email(self, access_token: str, message_id: str) -> Dict[str, Any]:
        try:
            await self._delete(access_token, f"/me/messages/{message_id}")
            return {"success": True}
        except Exception as e:
            logger.exception("Graph delete_email error: %s", e)
            return {"success": False, "error": str(e)}

    async def get_folders(self, access_token: str) -> Dict[str, Any]:
        try:
            data = await self._get(access_token, "/me/mailFolders")
            value = data.get("value", [])
            return {"folders": [_folder_to_dict(f) for f in value]}
        except Exception as e:
            logger.exception("Graph get_folders error: %s", e)
            return {"folders": [], "error": str(e)}

    async def sync_folder(
        self,
        access_token: str,
        folder_id: str,
        delta_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            path = f"/me/mailFolders/{folder_id}/messages/delta"
            params = {}
            if delta_token:
                params["$deltatoken"] = delta_token
            data = await self._get(access_token, path, params)
            value = data.get("value", [])
            added = []
            updated = []
            deleted_ids = []
            for m in value:
                if "@removed" in m:
                    deleted_ids.append(m.get("id", ""))
                else:
                    msg = _message_to_dict(m)
                    if m.get("id"):
                        updated.append(msg)
                    else:
                        added.append(msg)
            next_link = data.get("@odata.nextLink", "")
            delta_link = data.get("@odata.deltaLink", "")
            next_delta = delta_link.split("$deltatoken=")[-1] if delta_link else ""
            return {
                "added": added,
                "updated": updated,
                "deleted_ids": deleted_ids,
                "next_delta_token": next_delta,
            }
        except Exception as e:
            logger.exception("Graph sync_folder error: %s", e)
            return {"added": [], "updated": [], "deleted_ids": [], "next_delta_token": "", "error": str(e)}

    async def get_email_statistics(
        self, access_token: str, folder_id: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            fid = folder_id or "inbox"
            path = f"/me/mailFolders/{fid}"
            params = {"$select": "totalItemCount,unreadItemCount"}
            data = await self._get(access_token, path, params)
            return {
                "total_count": data.get("totalItemCount", 0),
                "unread_count": data.get("unreadItemCount", 0),
            }
        except Exception as e:
            logger.exception("Graph get_email_statistics error: %s", e)
            return {"total_count": 0, "unread_count": 0, "error": str(e)}
