"""
IMAP/SMTP provider for generic email (Gmail, Fastmail, self-hosted, etc.).
access_token is JSON: {imap_host, imap_port, imap_ssl, smtp_host, smtp_port, smtp_tls, username, imap_password, smtp_password}.
"""

import json
import logging
from email import policy
from email.parser import BytesParser
from typing import Any, Dict, List, Optional

from providers.base_provider import BaseProvider

logger = logging.getLogger(__name__)

_CALENDAR_MSG = "IMAP/SMTP provider does not support calendar operations."


def _parse_credentials(access_token: str) -> Dict[str, Any]:
    try:
        data = json.loads(access_token)
        if not isinstance(data, dict):
            raise ValueError("credentials must be a JSON object")
        return data
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("IMAP/SMTP invalid credentials JSON: %s", e)
        raise ValueError("Invalid IMAP/SMTP credentials") from e


def _parse_fetch_rfc822(fetch_data: list) -> List[Dict[str, Any]]:
    """Extract RFC822 message bodies from aioimaplib fetch response. Literals are full message bytes (lines[1], etc.)."""
    messages = []
    i = 0
    while i < len(fetch_data):
        line = fetch_data[i]
        if not isinstance(line, bytes):
            i += 1
            continue
        if line.startswith(b"* ") and b"FETCH" in line and b"RFC822" in line:
            i += 1
            if i < len(fetch_data) and isinstance(fetch_data[i], bytes):
                blob = fetch_data[i]
                if blob.startswith(b"From ") or b"Received:" in blob[:500]:
                    try:
                        msg = BytesParser(policy=policy.default).parsebytes(blob)
                        messages.append(_message_to_dict(msg))
                    except Exception as e:
                        logger.debug("Parse one message: %s", e)
                i += 1
            continue
        if line.startswith(b"From ") and len(line) > 20:
            try:
                msg = BytesParser(policy=policy.default).parsebytes(line)
                messages.append(_message_to_dict(msg))
            except Exception as e:
                logger.debug("Parse one message: %s", e)
        i += 1
    return messages


def _message_to_dict(msg) -> Dict[str, Any]:
    """Build normalized message dict from email.message.Message."""
    mid = str(msg.get("Message-ID", "") or "")
    refs = msg.get("References", "") or ""
    conv_id = str(msg.get("In-Reply-To", "") or (refs.split()[-1] if refs else ""))
    subject = str(msg.get("Subject", ""))
    from_hdr = msg.get("From") or ""
    if hasattr(from_hdr, "addresses") and from_hdr.addresses:
        from_addr = from_hdr.addresses[0].addr_spec
        from_name = (from_hdr.addresses[0].display_name or "") or ""
    else:
        from_addr = str(from_hdr)
        from_name = ""
    to_hdr = msg.get("To") or []
    if not isinstance(to_hdr, list):
        to_hdr = [to_hdr] if to_hdr else []
    to_addresses = []
    for t in to_hdr:
        if hasattr(t, "addresses"):
            to_addresses.extend(getattr(a, "addr_spec", str(a)) for a in t.addresses)
        else:
            to_addresses.append(str(t))
    cc_hdr = msg.get("Cc") or []
    if not isinstance(cc_hdr, list):
        cc_hdr = [cc_hdr] if cc_hdr else []
    cc_addresses = []
    for c in cc_hdr:
        if hasattr(c, "addresses"):
            cc_addresses.extend(getattr(a, "addr_spec", str(a)) for a in c.addresses)
        else:
            cc_addresses.append(str(c))
    date = msg.get("Date")
    received_datetime = ""
    if date and hasattr(date, "as_datetime"):
        try:
            received_datetime = date.as_datetime().isoformat()
        except Exception:
            received_datetime = str(date)
    else:
        received_datetime = str(date or "")
    body = msg.get_body(preferencelist=("plain", "html")) if hasattr(msg, "get_body") else None
    body_content = body.get_content() if body else ""
    body_preview = (body_content[:500] + "…") if len(body_content) > 500 else body_content
    return {
        "id": mid or conv_id or "(no-id)",
        "conversation_id": conv_id,
        "subject": subject,
        "from_address": from_addr,
        "from_name": from_name,
        "to_addresses": to_addresses,
        "cc_addresses": cc_addresses,
        "received_datetime": received_datetime,
        "is_read": False,
        "has_attachments": bool(msg.get_content_maintype() == "multipart" and any(p.get_content_disposition() == "attachment" for p in msg.walk())),
        "importance": "normal",
        "body_preview": body_preview,
        "body_content": body_content,
    }


class ImapSmtpProvider(BaseProvider):
    """IMAP/SMTP email provider. Calendar methods raise NotImplementedError."""

    @property
    def name(self) -> str:
        return "imap_smtp"

    def _creds(self, access_token: str) -> Dict[str, Any]:
        return _parse_credentials(access_token)

    async def _imap_connect(self, access_token: str):
        creds = self._creds(access_token)
        host = creds.get("imap_host", "")
        port = int(creds.get("imap_port", 993))
        use_ssl = bool(creds.get("imap_ssl", True))
        user = creds.get("username", "")
        password = creds.get("imap_password", "")
        if not host or not user or not password:
            raise ValueError("imap_host, username, and imap_password required")
        try:
            from aioimaplib import aioimaplib
        except ImportError:
            raise RuntimeError("aioimaplib is not installed; add it to connections-service requirements")
        if use_ssl:
            client = aioimaplib.IMAP4_SSL(host=host, port=port)
        else:
            client = aioimaplib.IMAP4(host=host, port=port)
        await client.wait_hello_from_server()
        resp, _ = await client.login(user, password)
        if resp != "OK":
            raise ValueError(f"IMAP login failed: {resp}")
        return client

    async def get_emails(
        self,
        access_token: str,
        folder_id: str = "inbox",
        top: int = 50,
        skip: int = 0,
        filter_expr: Optional[str] = None,
        unread_only: bool = False,
    ) -> Dict[str, Any]:
        client = None
        try:
            client = await self._imap_connect(access_token)
            mailbox = "INBOX" if folder_id.lower() == "inbox" else folder_id
            resp, data = await client.select(mailbox)
            if resp != "OK":
                return {"messages": [], "total_count": 0, "error": f"Select failed: {resp}"}
            total = int(data[0]) if data else 0
            if total == 0:
                await client.logout()
                return {"messages": [], "total_count": 0}
            if unread_only:
                resp2, uids = await client.uid_search("UNSEEN", charset="UTF-8")
            else:
                resp2, uids = await client.uid_search("ALL", charset="UTF-8")
            if resp2 != "OK" or not uids or not uids[0]:
                await client.logout()
                return {"messages": [], "total_count": 0}
            uid_list = uids[0].decode().split()
            uid_list.reverse()
            start = min(skip, len(uid_list))
            end = min(skip + top, len(uid_list))
            subset = uid_list[start:end]
            if not subset:
                await client.logout()
                return {"messages": [], "total_count": total}
            uid_set = ",".join(subset)
            resp3, fetch_data = await client.uid("fetch", uid_set, "(RFC822)")
            await client.logout()
            if resp3 != "OK":
                return {"messages": [], "total_count": total, "error": "Fetch failed"}
            messages = _parse_fetch_rfc822(fetch_data)
            return {"messages": messages, "total_count": total}
        except Exception as e:
            logger.exception("IMAP get_emails: %s", e)
            if client:
                try:
                    await client.logout()
                except Exception:
                    pass
            return {"messages": [], "total_count": 0, "error": str(e)}

    async def get_email_by_id(self, access_token: str, message_id: str) -> Dict[str, Any]:
        client = None
        try:
            client = await self._imap_connect(access_token)
            resp, uids = await client.uid_search("HEADER Message-ID " + message_id.replace('"', '\\"'), charset="UTF-8")
            if resp != "OK" or not uids or not uids[0]:
                await client.logout()
                return {"message": None, "error": "Message not found"}
            uid = uids[0].decode().strip().split()[-1]
            resp2, data = await client.uid("fetch", uid, "(RFC822)")
            await client.logout()
            if resp2 != "OK" or not data:
                return {"message": None, "error": "Fetch failed"}
            for line in data:
                if isinstance(line, bytes) and line.startswith(b"From ") or b"RFC822" in line:
                    continue
                if isinstance(line, bytes):
                    try:
                        msg = BytesParser(policy=policy.default).parsebytes(line)
                        return {"message": _message_to_dict(msg)}
                    except Exception:
                        pass
            return {"message": None, "error": "Parse failed"}
        except Exception as e:
            logger.exception("IMAP get_email_by_id: %s", e)
            if client:
                try:
                    await client.logout()
                except Exception:
                    pass
            return {"message": None, "error": str(e)}

    async def get_email_thread(
        self, access_token: str, conversation_id: str
    ) -> Dict[str, Any]:
        client = None
        try:
            client = await self._imap_connect(access_token)
            resp, uids = await client.uid_search("HEADER References " + conversation_id.replace('"', '\\"'), charset="UTF-8")
            if resp != "OK":
                resp, uids = await client.uid_search("HEADER In-Reply-To " + conversation_id.replace('"', '\\"'), charset="UTF-8")
            if resp != "OK" or not uids or not uids[0]:
                await client.logout()
                return {"messages": []}
            uid_set = uids[0].decode().strip().replace(" ", ",")
            resp2, data = await client.uid("fetch", uid_set, "(RFC822)")
            await client.logout()
            messages = []
            if resp2 == "OK" and data:
                for line in data:
                    if isinstance(line, bytes) and (line.startswith(b"From ") or b"RFC822" in line):
                        continue
                    if isinstance(line, bytes):
                        try:
                            msg = BytesParser(policy=policy.default).parsebytes(line)
                            messages.append(_message_to_dict(msg))
                        except Exception:
                            pass
            return {"messages": messages}
        except Exception as e:
            logger.exception("IMAP get_email_thread: %s", e)
            if client:
                try:
                    await client.logout()
                except Exception:
                    pass
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
        client = None
        try:
            client = await self._imap_connect(access_token)
            criteria = []
            if query:
                criteria.append("TEXT")
                criteria.append(query.replace('"', '\\"'))
            if from_address:
                criteria.extend(["FROM", from_address.replace('"', '\\"')])
            if start_date:
                criteria.extend(["SINCE", start_date[:10].replace('"', '\\"')])
            if end_date:
                criteria.extend(["BEFORE", end_date[:10].replace('"', '\\"')])
            if not criteria:
                criteria = ["ALL"]
            resp, uids = await client.uid_search(*criteria, charset="UTF-8")
            if resp != "OK" or not uids or not uids[0]:
                await client.logout()
                return {"messages": []}
            uid_list = uids[0].decode().split()
            uid_list.reverse()
            subset = uid_list[:top]
            uid_set = ",".join(subset)
            resp2, data = await client.uid("fetch", uid_set, "(RFC822)")
            await client.logout()
            messages = []
            if resp2 == "OK" and data:
                for line in data:
                    if isinstance(line, bytes) and (line.startswith(b"From ") or b"RFC822" in line):
                        continue
                    if isinstance(line, bytes):
                        try:
                            msg = BytesParser(policy=policy.default).parsebytes(line)
                            messages.append(_message_to_dict(msg))
                        except Exception:
                            pass
            return {"messages": messages}
        except Exception as e:
            logger.exception("IMAP search_emails: %s", e)
            if client:
                try:
                    await client.logout()
                except Exception:
                    pass
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
        creds = self._creds(access_token)
        try:
            import aiosmtplib
            from email.message import EmailMessage
        except ImportError:
            return {"success": False, "error": "aiosmtplib is not installed"}
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = creds.get("username", "")
        msg["To"] = ", ".join(to_recipients)
        if cc_recipients:
            msg["Cc"] = ", ".join(cc_recipients)
        if bcc_recipients:
            msg["Bcc"] = ", ".join(bcc_recipients)
        if body_is_html:
            msg.set_content(body, subtype="html")
        else:
            msg.set_content(body)
        host = creds.get("smtp_host", "")
        port = int(creds.get("smtp_port", 587))
        use_tls = bool(creds.get("smtp_tls", True))
        user = creds.get("username", "")
        password = creds.get("smtp_password", "")
        if not host or not user or not password:
            return {"success": False, "error": "smtp_host, username, smtp_password required"}
        try:
            if use_tls and port == 465:
                await aiosmtplib.send(
                    msg,
                    hostname=host,
                    port=port,
                    username=user,
                    password=password,
                    use_tls=True,
                )
            else:
                await aiosmtplib.send(
                    msg,
                    hostname=host,
                    port=port,
                    username=user,
                    password=password,
                    start_tls=use_tls,
                )
            return {"success": True}
        except Exception as e:
            logger.exception("SMTP send_email: %s", e)
            return {"success": False, "error": str(e)}

    async def create_draft(
        self,
        access_token: str,
        to_recipients: List[str],
        subject: str,
        body: str,
        cc_recipients: Optional[List[str]] = None,
        bcc_recipients: Optional[List[str]] = None,
        body_is_html: bool = False,
    ) -> Dict[str, Any]:
        return {"success": False, "error": "IMAP/SMTP provider does not support create_draft"}

    async def reply_to_email(
        self,
        access_token: str,
        message_id: str,
        body: str,
        reply_all: bool = False,
        body_is_html: bool = False,
    ) -> Dict[str, Any]:
        result = await self.get_email_by_id(access_token, message_id)
        if result.get("error") or not result.get("message"):
            return {"success": False, "error": result.get("error", "Message not found")}
        msg = result["message"]
        to_list = [msg.get("from_address", "")]
        if reply_all:
            to_list.extend(msg.get("cc_addresses", []))
        subject = msg.get("subject", "")
        if not subject.startswith("Re:"):
            subject = "Re: " + subject
        return await self.send_email(
            access_token, to_list, subject, body,
            cc_recipients=msg.get("cc_addresses") if reply_all else None,
            body_is_html=body_is_html,
        )

    async def update_email(
        self,
        access_token: str,
        message_id: str,
        is_read: Optional[bool] = None,
        importance: Optional[str] = None,
    ) -> Dict[str, Any]:
        client = None
        try:
            client = await self._imap_connect(access_token)
            resp, uids = await client.uid_search("HEADER Message-ID " + message_id.replace('"', '\\"'), charset="UTF-8")
            if resp != "OK" or not uids or not uids[0]:
                await client.logout()
                return {"success": False, "error": "Message not found"}
            uid = uids[0].decode().strip().split()[-1]
            if is_read is True:
                await client.uid("store", uid, "+FLAGS", "(\Seen)")
            elif is_read is False:
                await client.uid("store", uid, "-FLAGS", "(\Seen)")
            await client.logout()
            return {"success": True}
        except Exception as e:
            logger.exception("IMAP update_email: %s", e)
            if client:
                try:
                    await client.logout()
                except Exception:
                    pass
            return {"success": False, "error": str(e)}

    async def move_email(
        self,
        access_token: str,
        message_id: str,
        destination_folder_id: str,
    ) -> Dict[str, Any]:
        client = None
        try:
            client = await self._imap_connect(access_token)
            resp, uids = await client.uid_search("HEADER Message-ID " + message_id.replace('"', '\\"'), charset="UTF-8")
            if resp != "OK" or not uids or not uids[0]:
                await client.logout()
                return {"success": False, "error": "Message not found"}
            uid = uids[0].decode().strip().split()[-1]
            dest = "INBOX" if destination_folder_id.lower() == "inbox" else destination_folder_id
            await client.uid("move", uid, dest)
            await client.logout()
            return {"success": True}
        except Exception as e:
            logger.exception("IMAP move_email: %s", e)
            if client:
                try:
                    await client.logout()
                except Exception:
                    pass
            return {"success": False, "error": str(e)}

    async def delete_email(self, access_token: str, message_id: str) -> Dict[str, Any]:
        client = None
        try:
            client = await self._imap_connect(access_token)
            resp, uids = await client.uid_search("HEADER Message-ID " + message_id.replace('"', '\\"'), charset="UTF-8")
            if resp != "OK" or not uids or not uids[0]:
                await client.logout()
                return {"success": False, "error": "Message not found"}
            uid = uids[0].decode().strip().split()[-1]
            await client.uid("store", uid, "+FLAGS", "(\Deleted)")
            await client.expunge()
            await client.logout()
            return {"success": True}
        except Exception as e:
            logger.exception("IMAP delete_email: %s", e)
            if client:
                try:
                    await client.logout()
                except Exception:
                    pass
            return {"success": False, "error": str(e)}

    async def get_folders(self, access_token: str) -> Dict[str, Any]:
        client = None
        try:
            client = await self._imap_connect(access_token)
            resp, data = await client.list()
            await client.logout()
            if resp != "OK":
                return {"folders": [], "error": "LIST failed"}
            folders = []
            for line in data or []:
                if not isinstance(line, bytes):
                    continue
                parts = line.decode().split(None, 2)
                if len(parts) >= 3:
                    name = parts[2].strip(' "').split('"')[-1]
                    folders.append({
                        "id": name,
                        "name": name,
                        "parent_id": "",
                        "unread_count": 0,
                        "total_count": 0,
                    })
            return {"folders": folders}
        except Exception as e:
            logger.exception("IMAP get_folders: %s", e)
            if client:
                try:
                    await client.logout()
                except Exception:
                    pass
            return {"folders": [], "error": str(e)}

    async def sync_folder(
        self,
        access_token: str,
        folder_id: str,
        delta_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = await self.get_emails(access_token, folder_id=folder_id, top=100)
        return {
            "added": result.get("messages", []),
            "updated": [],
            "deleted_ids": [],
            "next_delta_token": "",
            "error": result.get("error"),
        }

    async def get_email_statistics(
        self, access_token: str, folder_id: Optional[str] = None
    ) -> Dict[str, Any]:
        client = None
        try:
            client = await self._imap_connect(access_token)
            mailbox = "INBOX" if not folder_id or folder_id.lower() == "inbox" else folder_id
            resp, data = await client.select(mailbox)
            if resp != "OK" or not data:
                await client.logout()
                return {"total_count": 0, "unread_count": 0, "error": "Select failed"}
            total = int(data[0])
            resp2, uids = await client.uid_search("UNSEEN", charset="UTF-8")
            unread = len(uids[0].decode().split()) if resp2 == "OK" and uids and uids[0] else 0
            await client.logout()
            return {"total_count": total, "unread_count": unread}
        except Exception as e:
            logger.exception("IMAP get_email_statistics: %s", e)
            if client:
                try:
                    await client.logout()
                except Exception:
                    pass
            return {"total_count": 0, "unread_count": 0, "error": str(e)}

    async def list_calendars(self, access_token: str) -> Dict[str, Any]:
        return {"calendars": [], "error": _CALENDAR_MSG}

    async def get_calendar_events(
        self,
        access_token: str,
        calendar_id: str = "",
        start_datetime: str = "",
        end_datetime: str = "",
        top: int = 50,
    ) -> Dict[str, Any]:
        return {"events": [], "total_count": 0, "error": _CALENDAR_MSG}

    async def get_event_by_id(self, access_token: str, event_id: str) -> Dict[str, Any]:
        return {"event": None, "error": _CALENDAR_MSG}

    async def create_event(
        self,
        access_token: str,
        subject: str,
        start_datetime: str,
        end_datetime: str,
        location: str = "",
        body: str = "",
        body_is_html: bool = False,
        attendee_emails: Optional[List[str]] = None,
        is_all_day: bool = False,
        calendar_id: str = "",
    ) -> Dict[str, Any]:
        return {"success": False, "error": _CALENDAR_MSG}

    async def update_event(
        self,
        access_token: str,
        event_id: str,
        subject: Optional[str] = None,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
        location: Optional[str] = None,
        body: Optional[str] = None,
        body_is_html: bool = False,
        attendee_emails: Optional[List[str]] = None,
        is_all_day: Optional[bool] = None,
    ) -> Dict[str, Any]:
        return {"success": False, "error": _CALENDAR_MSG}

    async def delete_event(self, access_token: str, event_id: str) -> Dict[str, Any]:
        return {"success": False, "error": _CALENDAR_MSG}
