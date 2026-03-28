"""
Twilio SMS provider for outbound SMS. No inbound listener; start() validates credentials only.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from providers.base_messaging_provider import (
    BaseMessagingProvider,
    MessageCallback,
    OutboundImage,
)

logger = logging.getLogger(__name__)

# Max body length to avoid excessive segments (Twilio auto-segments at 160 chars)
SMS_BODY_MAX_LEN = 1600


def _strip_formatting(text: str) -> str:
    """Strip markdown/HTML to plain text for SMS."""
    if not text:
        return ""
    # Remove markdown bold/italic/code
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove markdown links but keep URL
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 \2", text)
    # Strip remaining HTML-like tags
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


class SMSProvider(BaseMessagingProvider):
    """Twilio SMS outbound-only provider. No persistent listener."""

    def __init__(self) -> None:
        self._connection_id: Optional[str] = None
        self._auth_token: Optional[str] = None
        self._account_sid: Optional[str] = None
        self._from_number: Optional[str] = None
        self._running = False
        self._client: Any = None

    @property
    def name(self) -> str:
        return "sms"

    async def start(
        self,
        bot_token: str,
        config: Dict[str, Any],
        message_callback: MessageCallback,
        connection_id: str,
    ) -> None:
        self._connection_id = connection_id
        self._auth_token = bot_token
        self._account_sid = (config or {}).get("account_sid") or ""
        self._from_number = (config or {}).get("from_number") or ""
        try:
            from twilio.rest import Client

            self._client = Client(self._account_sid, self._auth_token)
            self._running = True
            logger.info("SMS provider started for connection_id=%s from=%s", connection_id, self._from_number)
        except Exception as e:
            logger.exception("SMS start failed: %s", e)
            raise

    async def stop(self) -> None:
        self._running = False
        self._client = None
        self._auth_token = None
        self._account_sid = None
        self._from_number = None
        logger.info("SMS provider stopped for connection_id=%s", self._connection_id)

    async def send_message(
        self,
        chat_id: str,
        text: str,
        images: Optional[List[OutboundImage]] = None,
    ) -> Dict[str, Any]:
        if not self._client or not self._running or not self._from_number:
            return {"success": False, "error": "SMS provider not running or missing from number"}
        body = _strip_formatting(text or "")
        if len(body) > SMS_BODY_MAX_LEN:
            body = body[:SMS_BODY_MAX_LEN - 3] + "..."
        try:
            params: Dict[str, Any] = {
                "to": chat_id,
                "from_": self._from_number,
                "body": body,
            }
            if images and len(images) > 0:
                # MMS: Twilio accepts media_url (public URLs). We have bytes; skip MMS for initial implementation.
                logger.debug("SMS provider: images not supported for MMS in this version, sending text only")
            message = self._client.messages.create(**params)
            return {"success": True, "message_id": message.sid}
        except Exception as e:
            err_msg = str(e)
            logger.exception("SMS send_message failed: %s", e)
            return {"success": False, "error": err_msg}

    async def get_bot_info(self, bot_token: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        account_sid = (config or {}).get("account_sid") if config else None
        from_number = (config or {}).get("from_number") if config else None
        if not account_sid or not from_number:
            return {"username": "", "error": "SMS requires config with account_sid and from_number"}
        try:
            from twilio.rest import Client

            client = Client(account_sid, bot_token)
            client.api.accounts(account_sid).fetch()
            return {"username": from_number}
        except Exception as e:
            logger.warning("SMS get_bot_info failed: %s", e)
            return {"username": "", "error": str(e)}
