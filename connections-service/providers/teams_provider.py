"""
Microsoft Teams bot provider via Azure Bot Framework (inbound HTTPS webhook).

Inbound activities are POSTed by Microsoft; outbound replies use the Bot Framework REST API.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import urllib.parse
from typing import Any, Callable, Dict, List, Optional

import httpx
import jwt
from jwt import PyJWKClient

from providers.base_messaging_provider import (
    BaseMessagingProvider,
    InboundMessage,
    MessageCallback,
    OutboundImage,
)

logger = logging.getLogger(__name__)

# Bot Framework JWT issuer and JWKS (see Azure Bot authentication docs).
BOT_FRAMEWORK_ISSUER = "https://api.botframework.com"
BOT_FRAMEWORK_JWKS_URI = "https://login.botframework.com/v1/.well-known/keys"

CHAT_ID_SEP = "||"

_instances: Dict[str, "TeamsProvider"] = {}

_jwks_client: Optional[PyJWKClient] = None


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(BOT_FRAMEWORK_JWKS_URI)
    return _jwks_client


def get_teams_instance(connection_id: str) -> Optional["TeamsProvider"]:
    """Lookup a running Teams provider by Bastion external_connections id."""
    return _instances.get(str(connection_id))


def _register_instance(connection_id: str, instance: "TeamsProvider") -> None:
    _instances[str(connection_id)] = instance


def _unregister_instance(connection_id: str) -> None:
    _instances.pop(str(connection_id), None)


def _strip_teams_mentions(text: str) -> str:
    if not text:
        return ""
    out = re.sub(r"<at>.*?</at>\s*", "", text)
    return out.strip()


class TeamsProvider(BaseMessagingProvider):
    """Teams channel / DM bot using Bot Framework REST API."""

    def __init__(self) -> None:
        self._app_id: str = ""
        self._app_password: str = ""
        self._tenant_id: str = "common"
        self._connection_id: Optional[str] = None
        self._message_callback: Optional[MessageCallback] = None
        self._access_token: str = ""
        self._access_token_expires_at: float = 0.0
        self._running = False

    @property
    def name(self) -> str:
        return "teams"

    async def get_bot_info(
        self, bot_token: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate Microsoft App ID + secret by obtaining a Bot Framework token."""
        cfg = dict(config or {})
        password = (cfg.get("app_password") or "").strip()
        tenant = (cfg.get("tenant_id") or "common").strip() or "common"
        app_id = (bot_token or "").strip()
        if not app_id:
            return {"username": "", "error": "app_id (Microsoft App ID) is required"}
        if not password:
            return {"username": "", "error": "app_password is required in config"}
        token_url = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    token_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": app_id,
                        "client_secret": password,
                        "scope": "https://api.botframework.com/.default",
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
            if resp.status_code != 200:
                detail = resp.text[:500]
                return {
                    "username": "",
                    "error": f"Token request failed ({resp.status_code}): {detail}",
                }
            return {"username": app_id}
        except Exception as e:
            logger.warning("Teams get_bot_info failed: %s", e)
            return {"username": "", "error": str(e)}

    async def start(
        self,
        bot_token: str,
        config: Dict[str, Any],
        message_callback: MessageCallback,
        connection_id: str,
    ) -> None:
        cfg = dict(config or {})
        password = (cfg.get("app_password") or "").strip()
        if not password:
            raise ValueError("Teams bot requires app_password in config (Microsoft client secret)")
        self._app_id = (bot_token or "").strip()
        self._app_password = password
        self._tenant_id = (cfg.get("tenant_id") or "common").strip() or "common"
        self._connection_id = str(connection_id)
        self._message_callback = message_callback
        self._running = True
        _register_instance(connection_id, self)
        logger.info("Teams bot registered connection_id=%s", connection_id)

    async def stop(self) -> None:
        self._running = False
        if self._connection_id:
            _unregister_instance(self._connection_id)
        self._message_callback = None
        self._access_token = ""
        self._access_token_expires_at = 0.0
        logger.info("Teams bot stopped connection_id=%s", self._connection_id)

    def _verify_jwt(self, auth_header: str) -> None:
        """Validate Bot Framework Bearer JWT (blocking; call via asyncio.to_thread)."""
        if not auth_header or not auth_header.startswith("Bearer "):
            raise ValueError("missing_or_invalid_authorization")
        token = auth_header[7:].strip()
        if not token:
            raise ValueError("empty_bearer_token")
        try:
            jwks = _get_jwks_client()
            signing_key = jwks.get_signing_key_from_jwt(token)
            jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self._app_id,
                issuer=BOT_FRAMEWORK_ISSUER,
                options={"verify_exp": True},
            )
        except jwt.exceptions.InvalidTokenError as e:
            raise ValueError(f"invalid_bot_framework_jwt: {e}") from e
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"jwt_validation_failed: {e}") from e

    async def handle_activity(self, body: Dict[str, Any], auth_header: str) -> None:
        """
        Entry point from HTTP webhook: validate JWT, acknowledge quickly, process async.
        Only handles type=message; other activity types are ignored.
        """
        if not self._running or not self._message_callback:
            return
        await asyncio.to_thread(self._verify_jwt, auth_header)
        atype = (body.get("type") or "").strip().lower()
        if atype != "message":
            return
        asyncio.create_task(self._process_message_activity(body))

    async def _process_message_activity(self, body: Dict[str, Any]) -> None:
        try:
            text = _strip_teams_mentions(body.get("text") or "")
            if not text:
                return
            service_url = (body.get("serviceUrl") or "").strip().rstrip("/")
            conv = body.get("conversation") or {}
            conversation_id = (conv.get("id") or "").strip()
            if not service_url or not conversation_id:
                logger.warning("Teams message missing serviceUrl or conversation.id")
                return
            frm = body.get("from") or {}
            sender_id = str(frm.get("id") or "")
            sender_name = str(frm.get("name") or sender_id or "unknown")
            chat_id = f"{service_url}{CHAT_ID_SEP}{conversation_id}"
            tenant_id = ""
            ch_data = body.get("channelData") or {}
            if isinstance(ch_data, dict):
                tenant_id = str(ch_data.get("tenant", {}).get("id") or "")
            chat_title = conversation_id
            if isinstance(conv.get("name"), str) and conv["name"]:
                chat_title = conv["name"]
            elif tenant_id:
                chat_title = f"{conversation_id[:16]}…"

            teams_chat_type = str(conv.get("conversationType") or "")
            inbound = InboundMessage(
                sender_id=sender_id,
                sender_name=sender_name,
                chat_id=chat_id,
                text=text,
                images=[],
                platform="teams",
                timestamp=body.get("timestamp"),
                connection_id=self._connection_id,
                chat_title=chat_title,
                chat_username=None,
                chat_type=teams_chat_type or None,
                audio=None,
            )
            cb = self._message_callback
            if cb:
                await cb(inbound)
        except Exception as e:
            logger.exception("Teams _process_message_activity failed: %s", e)

    async def _get_access_token(self) -> str:
        now = time.time()
        if self._access_token and self._access_token_expires_at > now + 120:
            return self._access_token
        token_url = f"https://login.microsoftonline.com/{self._tenant_id}/oauth2/v2.0/token"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._app_id,
                    "client_secret": self._app_password,
                    "scope": "https://api.botframework.com/.default",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        resp.raise_for_status()
        data = resp.json()
        self._access_token = data.get("access_token") or ""
        expires_in = int(data.get("expires_in") or 3600)
        self._access_token_expires_at = now + expires_in
        return self._access_token

    async def send_message(
        self,
        chat_id: str,
        text: str,
        images: Optional[List[OutboundImage]] = None,
    ) -> Dict[str, Any]:
        if not self._running:
            return {"success": False, "error": "Bot not running"}
        if CHAT_ID_SEP not in chat_id:
            return {"success": False, "error": "Invalid Teams chat_id encoding"}
        service_url, conversation_id = chat_id.split(CHAT_ID_SEP, 1)
        service_url = service_url.rstrip("/")
        if not service_url or not conversation_id:
            return {"success": False, "error": "Invalid Teams chat_id parts"}
        path_conv = urllib.parse.quote(conversation_id, safe="")
        url = f"{service_url}/v3/conversations/{path_conv}/activities"
        try:
            token = await self._get_access_token()
        except Exception as e:
            logger.exception("Teams token for send_message failed: %s", e)
            return {"success": False, "error": str(e)}

        plain = (text or "")[:28000]
        chunks: List[str] = []
        step = 1800
        for i in range(0, len(plain), step):
            chunks.append(plain[i : i + step])
        if not chunks:
            chunks = [""]

        message_ts = ""
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                for chunk in chunks:
                    payload: Dict[str, Any] = {
                        "type": "message",
                        "text": chunk,
                        "from": {"id": self._app_id, "name": "Bot"},
                        "conversation": {"id": conversation_id},
                    }
                    resp = await client.post(url, json=payload, headers=headers)
                    if resp.status_code not in (200, 201, 202):
                        err = resp.text[:500]
                        return {
                            "success": False,
                            "error": f"Send failed ({resp.status_code}): {err}",
                        }
                    try:
                        body = resp.json()
                        if isinstance(body, dict) and body.get("id"):
                            message_ts = str(body.get("id"))
                    except Exception:
                        pass
                if images:
                    logger.info(
                        "Teams send_message: image attachments not fully supported; skipped %s image(s)",
                        len(images),
                    )
            return {"success": True, "message_id": message_ts}
        except Exception as e:
            logger.exception("Teams send_message failed: %s", e)
            return {"success": False, "error": str(e)}

    async def send_typing_indicator(self, chat_id: str) -> None:
        return


# Alias for webhook module (plan naming)
def get_instance(connection_id: str) -> Optional[TeamsProvider]:
    return get_teams_instance(connection_id)
