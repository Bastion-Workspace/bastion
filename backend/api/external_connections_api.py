"""
External Connections API - OAuth and connection management for email, calendar, etc.
"""

import logging
import secrets
from urllib.parse import urlencode
from typing import List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from config import settings
from models.api_models import AuthenticatedUserResponse
from services.auth_service import auth_service
from services.external_connections_service import external_connections_service
from utils.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["external-connections"])

OAUTH_STATE_TTL = 600
MICROSOFT_AUTHORIZE_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize"
MICROSOFT_TOKEN_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
MICROSOFT_SCOPES = "openid profile email Mail.Read Mail.ReadWrite Mail.Send MailboxSettings.Read User.Read"


class AuthorizeResponse(BaseModel):
    url: str
    state: str


class TelegramConnectRequest(BaseModel):
    bot_token: str


class DiscordConnectRequest(BaseModel):
    bot_token: str
    guild_ids: Optional[List[str]] = None


async def _get_redis():
    """Return Redis client from auth service if available."""
    if auth_service.redis_client:
        return auth_service.redis_client
    return None


async def _store_oauth_state(state: str, user_id: str) -> None:
    redis = await _get_redis()
    if redis:
        await redis.setex(f"oauth:state:{state}", OAUTH_STATE_TTL, user_id)
    else:
        logger.warning("Redis not available; OAuth state cannot be stored (use Redis for production)")


async def _get_oauth_state(state: str) -> str | None:
    redis = await _get_redis()
    if not redis:
        return None
    data = await redis.get(f"oauth:state:{state}")
    if data:
        await redis.delete(f"oauth:state:{state}")
        return data.decode() if isinstance(data, bytes) else data
    return None


@router.get("/oauth/microsoft/authorize", response_model=AuthorizeResponse)
async def microsoft_authorize(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Return Microsoft OAuth authorization URL and state for the current user."""
    redirect_uri = settings.effective_microsoft_redirect_uri
    if not settings.MICROSOFT_CLIENT_ID or not redirect_uri:
        raise HTTPException(
            status_code=503,
            detail="Microsoft OAuth is not configured. Set MICROSOFT_CLIENT_ID and MICROSOFT_CLIENT_SECRET in the server environment. Redirect URI is derived from SITE_URL when MICROSOFT_REDIRECT_URI is not set.",
        )
    state = secrets.token_urlsafe(32)
    await _store_oauth_state(state, current_user.user_id)
    tenant = settings.MICROSOFT_TENANT_ID or "common"
    params = {
        "client_id": settings.MICROSOFT_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "response_mode": "query",
        "scope": MICROSOFT_SCOPES,
        "state": state,
    }
    url = MICROSOFT_AUTHORIZE_URL.format(tenant=tenant) + "?" + urlencode(params)
    return AuthorizeResponse(url=url, state=state)


@router.get("/oauth/microsoft/callback")
async def microsoft_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
):
    """Handle Microsoft OAuth callback: exchange code for tokens and store connection."""
    if error:
        logger.warning("Microsoft OAuth error: %s - %s", error, error_description or "")
        return RedirectResponse(
            url=f"{settings.SITE_URL}/settings?connections=error&message={error_description or error}",
            status_code=302,
        )
    if not code or not state:
        return RedirectResponse(
            url=f"{settings.SITE_URL}/settings?connections=error&message=missing_code_or_state",
            status_code=302,
        )
    user_id = await _get_oauth_state(state)
    if not user_id:
        logger.warning("Invalid or expired OAuth state")
        return RedirectResponse(
            url=f"{settings.SITE_URL}/settings?connections=error&message=invalid_state",
            status_code=302,
        )
    import httpx
    tenant = settings.MICROSOFT_TENANT_ID or "common"
    token_url = MICROSOFT_TOKEN_URL.format(tenant=tenant)
    redirect_uri = settings.effective_microsoft_redirect_uri
    data = {
        "client_id": settings.MICROSOFT_CLIENT_ID,
        "client_secret": settings.MICROSOFT_CLIENT_SECRET,
        "code": code,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(token_url, data=data, timeout=15.0)
            resp.raise_for_status()
            body = resp.json()
    except Exception as e:
        logger.exception("Microsoft token exchange failed: %s", e)
        return RedirectResponse(
            url=f"{settings.SITE_URL}/settings?connections=error&message=token_exchange_failed",
            status_code=302,
        )
    access_token = body.get("access_token")
    refresh_token = body.get("refresh_token")
    expires_in = int(body.get("expires_in", 3600))
    scopes = (body.get("scope") or "").split()
    if not access_token or not refresh_token:
        return RedirectResponse(
            url=f"{settings.SITE_URL}/settings?connections=error&message=no_tokens",
            status_code=302,
        )
    try:
        async with httpx.AsyncClient() as client:
            user_resp = await client.get(
                "https://graph.microsoft.com/v1.0/me",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10.0,
            )
            user_resp.raise_for_status()
            me = user_resp.json()
    except Exception as e:
        logger.warning("Failed to fetch Microsoft user info: %s", e)
        me = {}
    account_identifier = (me.get("mail") or me.get("userPrincipalName") or "").strip()
    display_name = (me.get("displayName") or "").strip()
    provider_metadata = {"tenant_id": tenant}
    await external_connections_service.store_connection(
        user_id=user_id,
        provider="microsoft",
        connection_type="email",
        account_identifier=account_identifier or "unknown",
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires_in,
        scopes=scopes,
        display_name=display_name or None,
        provider_metadata=provider_metadata,
    )
    return RedirectResponse(
        url=f"{settings.SITE_URL}/settings?connections=success",
        status_code=302,
    )


@router.get("/connections")
async def list_connections(
    provider: str | None = None,
    connection_type: str | None = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """List external connections for the current user."""
    connections = await external_connections_service.get_user_connections(
        current_user.user_id,
        provider=provider,
        connection_type=connection_type,
        active_only=True,
    )
    return {"connections": connections}


@router.delete("/connections/{connection_id:int}")
async def revoke_connection(
    connection_id: int,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Revoke an external connection. User can only revoke their own."""
    conn = await external_connections_service.get_connection_by_id(connection_id)
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found")
    if conn.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not allowed to revoke this connection")
    if conn.get("connection_type") == "chat_bot":
        try:
            from clients.connections_service_client import get_connections_service_client
            client = await get_connections_service_client()
            await client.unregister_bot(connection_id)
        except Exception as e:
            logger.warning("UnregisterBot failed on revoke: %s", e)
    await external_connections_service.revoke_connection(connection_id)
    return {"ok": True}


@router.get("/connections/{connection_id:int}/bot-status")
async def get_bot_status(
    connection_id: int,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Get status of a messaging bot (Telegram/Discord) listener."""
    conn = await external_connections_service.get_connection_by_id(connection_id)
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found")
    if conn.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not allowed to view this connection")
    if conn.get("connection_type") != "chat_bot":
        return {"status": "n/a", "bot_username": "", "error": None}
    try:
        from clients.connections_service_client import get_connections_service_client
        client = await get_connections_service_client()
        result = await client.get_bot_status(connection_id)
        return result
    except Exception as e:
        logger.warning("GetBotStatus failed: %s", e)
        return {"status": "error", "bot_username": "", "error": str(e)}


@router.post("/connections/{connection_id:int}/refresh")
async def refresh_connection(
    connection_id: int,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Force token refresh for a connection."""
    conn = await external_connections_service.get_connection_by_id(connection_id)
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found")
    if conn.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not allowed to refresh this connection")
    ok = await external_connections_service.refresh_access_token(connection_id)
    return {"ok": ok}


async def _validate_telegram_token(bot_token: str) -> str:
    """Validate Telegram bot token and return bot username."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"https://api.telegram.org/bot{bot_token}/getMe",
        )
        resp.raise_for_status()
        data = resp.json()
    if not data.get("ok"):
        raise ValueError(data.get("description", "Invalid token"))
    result = data.get("result", {})
    return result.get("username", "") or ""


async def _validate_discord_token(bot_token: str) -> str:
    """Validate Discord bot token and return bot username."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            "https://discord.com/api/v10/users/@me",
            headers={"Authorization": f"Bot {bot_token}"},
        )
        resp.raise_for_status()
        data = resp.json()
    return data.get("username", "") or ""


@router.post("/connections/telegram")
async def connect_telegram_bot(
    body: TelegramConnectRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Register a Telegram bot for the current user."""
    try:
        bot_username = await _validate_telegram_token(body.bot_token)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Telegram token: {e}")
    conn_id = await external_connections_service.store_connection(
        user_id=current_user.user_id,
        provider="telegram",
        connection_type="chat_bot",
        account_identifier=bot_username or "telegram_bot",
        access_token=body.bot_token,
        refresh_token=body.bot_token,
        expires_in=365 * 24 * 3600,
        scopes=["bot"],
        display_name=bot_username or None,
        provider_metadata={},
    )
    token = await external_connections_service.get_valid_access_token(conn_id)
    if token:
        from clients.connections_service_client import get_connections_service_client
        client = await get_connections_service_client()
        result = await client.register_bot(
            connection_id=conn_id,
            user_id=current_user.user_id,
            provider="telegram",
            bot_token=token,
            display_name=bot_username or "",
        )
        if not result.get("success") and result.get("error"):
            logger.warning("Connections-service RegisterBot failed: %s", result.get("error"))
    return {"connection_id": conn_id, "bot_username": bot_username}


@router.post("/connections/discord")
async def connect_discord_bot(
    body: DiscordConnectRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Register a Discord bot for the current user."""
    try:
        bot_username = await _validate_discord_token(body.bot_token)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Discord token: {e}")
    config = {}
    if body.guild_ids:
        config["guild_ids"] = ",".join(body.guild_ids)
    conn_id = await external_connections_service.store_connection(
        user_id=current_user.user_id,
        provider="discord",
        connection_type="chat_bot",
        account_identifier=bot_username or "discord_bot",
        access_token=body.bot_token,
        refresh_token=body.bot_token,
        expires_in=365 * 24 * 3600,
        scopes=["bot"],
        display_name=bot_username or None,
        provider_metadata=config,
    )
    token = await external_connections_service.get_valid_access_token(conn_id)
    if token:
        from clients.connections_service_client import get_connections_service_client
        client = await get_connections_service_client()
        result = await client.register_bot(
            connection_id=conn_id,
            user_id=current_user.user_id,
            provider="discord",
            bot_token=token,
            display_name=bot_username or "",
            config=config,
        )
        if not result.get("success") and result.get("error"):
            logger.warning("Connections-service RegisterBot failed: %s", result.get("error"))
    return {"connection_id": conn_id, "bot_username": bot_username}
