"""
External Connections API - OAuth and connection management for email, calendar, etc.
"""

import json
import logging
import secrets
from urllib.parse import urlencode
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from config import settings
from models.api_models import AuthenticatedUserResponse
from services.auth_service import auth_service
from services.external_connections_service import external_connections_service
from services.m365_oauth_utils import (
    M365_ALL_SERVICE_KEYS,
    build_m365_scope_string,
    missing_scopes_for_services,
    normalize_m365_services,
    parse_services_query,
)
from utils.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["external-connections"])

OAUTH_STATE_TTL = 600
MICROSOFT_AUTHORIZE_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize"
MICROSOFT_TOKEN_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
# Long-lived placeholder expiry when GitHub does not return expires_in (token valid until revoked)
GITHUB_TOKEN_EXPIRES_SECONDS = 10 * 365 * 24 * 3600


class AuthorizeResponse(BaseModel):
    url: str
    state: str


class TelegramConnectRequest(BaseModel):
    bot_token: str


class DiscordConnectRequest(BaseModel):
    bot_token: str
    guild_ids: Optional[List[str]] = None


class SlackConnectRequest(BaseModel):
    bot_token: str
    app_token: str  # xapp-... for Socket Mode


class TeamsConnectRequest(BaseModel):
    """Azure Bot / Microsoft Teams: App ID + client secret (Bot Framework token scope)."""

    app_id: str
    app_password: str
    tenant_id: str = "common"


class SMSConnectRequest(BaseModel):
    account_sid: str
    auth_token: str
    from_number: str


class ImapSmtpConnectRequest(BaseModel):
    imap_host: str
    imap_port: int = 993
    imap_ssl: bool = True
    smtp_host: str
    smtp_port: int = 587
    smtp_tls: bool = True
    username: str
    imap_password: str
    smtp_password: str
    display_name: Optional[str] = None


class CalDAVConnectRequest(BaseModel):
    url: str
    username: str
    password: str
    display_name: Optional[str] = None


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


async def _store_microsoft_oauth_state(state: str, payload: Dict[str, Any]) -> None:
    redis = await _get_redis()
    if redis:
        await redis.setex(f"oauth:ms:{state}", OAUTH_STATE_TTL, json.dumps(payload))
    else:
        logger.warning("Redis not available; Microsoft OAuth state cannot be stored (use Redis for production)")


async def _get_microsoft_oauth_state(state: str) -> Optional[Dict[str, Any]]:
    redis = await _get_redis()
    if not redis:
        return None
    data = await redis.get(f"oauth:ms:{state}")
    if not data:
        return None
    await redis.delete(f"oauth:ms:{state}")
    text = data.decode() if isinstance(data, bytes) else data
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _microsoft_authorize_url_for_user(
    *,
    user_id: str,
    services: List[str],
    connection_id: Optional[int] = None,
) -> tuple[str, str, Dict[str, Any]]:
    """Build authorize URL, opaque state, and Redis payload (store payload under state)."""
    redirect_uri = settings.effective_microsoft_redirect_uri
    tenant = settings.MICROSOFT_TENANT_ID or "common"
    state = secrets.token_urlsafe(32)
    svc_norm = normalize_m365_services(services)
    payload: Dict[str, Any] = {"user_id": user_id, "services": svc_norm}
    if connection_id is not None:
        payload["connection_id"] = int(connection_id)
    scope_str = build_m365_scope_string(svc_norm)
    params: Dict[str, Any] = {
        "client_id": settings.MICROSOFT_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "response_mode": "query",
        "scope": scope_str,
        "state": state,
    }
    if connection_id is not None:
        params["prompt"] = "consent"
    url = MICROSOFT_AUTHORIZE_URL.format(tenant=tenant) + "?" + urlencode(params)
    return url, state, payload


@router.get("/oauth/microsoft/authorize", response_model=AuthorizeResponse)
async def microsoft_authorize(
    services: Optional[str] = None,
    connection_id: Optional[int] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Return Microsoft OAuth authorization URL and state; optional services=comma list and connection_id for incremental consent."""
    redirect_uri = settings.effective_microsoft_redirect_uri
    if not settings.MICROSOFT_CLIENT_ID or not redirect_uri:
        raise HTTPException(
            status_code=503,
            detail="Microsoft OAuth is not configured. Set MICROSOFT_CLIENT_ID and MICROSOFT_CLIENT_SECRET in the server environment. Redirect URI is derived from SITE_URL when MICROSOFT_REDIRECT_URI is not set.",
        )
    svc_list = parse_services_query(services)
    if connection_id is not None:
        conn = await external_connections_service.get_connection_by_id(
            connection_id, rls_context={"user_id": current_user.user_id}
        )
        if not conn or conn.get("user_id") != current_user.user_id:
            raise HTTPException(status_code=404, detail="Connection not found")
        if conn.get("provider") != "microsoft" or conn.get("connection_type") != "email":
            raise HTTPException(status_code=400, detail="Not a Microsoft 365 email connection")
    url, state, payload = _microsoft_authorize_url_for_user(
        user_id=current_user.user_id,
        services=svc_list,
        connection_id=connection_id,
    )
    await _store_microsoft_oauth_state(state, payload)
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
    ms_payload = await _get_microsoft_oauth_state(state)
    if not ms_payload or not ms_payload.get("user_id"):
        logger.warning("Invalid or expired Microsoft OAuth state")
        return RedirectResponse(
            url=f"{settings.SITE_URL}/settings?connections=error&message=invalid_state",
            status_code=302,
        )
    user_id = str(ms_payload["user_id"])
    raw_svcs = ms_payload.get("services")
    if isinstance(raw_svcs, list):
        svc_list = normalize_m365_services([str(x) for x in raw_svcs])
    else:
        svc_list = normalize_m365_services([])
    existing_connection_id = ms_payload.get("connection_id")
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
    provider_metadata: Dict[str, Any] = {"tenant_id": tenant, "enabled_services": svc_list}
    if existing_connection_id is not None:
        try:
            cid = int(existing_connection_id)
        except (TypeError, ValueError):
            cid = None
        if cid is not None:
            conn = await external_connections_service.get_connection_by_id(
                cid, rls_context={"user_id": user_id}
            )
            if not conn or conn.get("user_id") != user_id:
                return RedirectResponse(
                    url=f"{settings.SITE_URL}/settings?connections=error&message=connection_mismatch",
                    status_code=302,
                )
            ok = await external_connections_service.update_oauth_tokens_and_metadata(
                cid,
                access_token,
                refresh_token,
                expires_in,
                scopes,
                provider_metadata_updates=provider_metadata,
                rls_context={"user_id": user_id},
            )
            if not ok:
                return RedirectResponse(
                    url=f"{settings.SITE_URL}/settings?connections=error&message=update_failed",
                    status_code=302,
                )
            return RedirectResponse(
                url=f"{settings.SITE_URL}/settings?connections=success",
                status_code=302,
            )
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


@router.get("/oauth/github/authorize", response_model=AuthorizeResponse)
async def github_authorize(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Return GitHub OAuth authorization URL and state for the current user."""
    redirect_uri = settings.effective_github_redirect_uri
    if not settings.GITHUB_CLIENT_ID or not settings.GITHUB_CLIENT_SECRET or not redirect_uri:
        raise HTTPException(
            status_code=503,
            detail="GitHub OAuth is not configured. Set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET. Redirect URI is derived from SITE_URL when GITHUB_REDIRECT_URI is not set.",
        )
    state = secrets.token_urlsafe(32)
    await _store_oauth_state(state, current_user.user_id)
    params = {
        "client_id": settings.GITHUB_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": settings.GITHUB_SCOPES,
        "state": state,
    }
    url = GITHUB_AUTHORIZE_URL + "?" + urlencode(params)
    return AuthorizeResponse(url=url, state=state)


@router.get("/oauth/github/callback")
async def github_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
):
    """Handle GitHub OAuth callback: exchange code for token and store connection."""
    if error:
        logger.warning("GitHub OAuth error: %s - %s", error, error_description or "")
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
        logger.warning("Invalid or expired GitHub OAuth state")
        return RedirectResponse(
            url=f"{settings.SITE_URL}/settings?connections=error&message=invalid_state",
            status_code=302,
        )
    redirect_uri = settings.effective_github_redirect_uri
    token_body = {
        "client_id": settings.GITHUB_CLIENT_ID,
        "client_secret": settings.GITHUB_CLIENT_SECRET,
        "code": code,
        "redirect_uri": redirect_uri,
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                GITHUB_TOKEN_URL,
                data=token_body,
                headers={"Accept": "application/json"},
                timeout=15.0,
            )
            resp.raise_for_status()
            body = resp.json()
    except Exception as e:
        logger.exception("GitHub token exchange failed: %s", e)
        return RedirectResponse(
            url=f"{settings.SITE_URL}/settings?connections=error&message=token_exchange_failed",
            status_code=302,
        )
    access_token = body.get("access_token")
    if not access_token:
        err_msg = body.get("error_description") or body.get("error") or "no_access_token"
        return RedirectResponse(
            url=f"{settings.SITE_URL}/settings?connections=error&message={err_msg}",
            status_code=302,
        )
    refresh_token = body.get("refresh_token") or access_token
    expires_in = int(body.get("expires_in") or GITHUB_TOKEN_EXPIRES_SECONDS)
    scope_str = body.get("scope") or settings.GITHUB_SCOPES
    scopes = scope_str.split() if isinstance(scope_str, str) else list(scope_str or [])
    try:
        async with httpx.AsyncClient() as client:
            user_resp = await client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/vnd.github+json",
                },
                timeout=10.0,
            )
            user_resp.raise_for_status()
            gh_user = user_resp.json()
    except Exception as e:
        logger.warning("Failed to fetch GitHub user info: %s", e)
        gh_user = {}
    login = (gh_user.get("login") or "").strip()
    display_name = (gh_user.get("name") or login or "").strip()
    account_identifier = login or "github_user"
    provider_metadata = {"github_user_id": gh_user.get("id")}
    await external_connections_service.store_connection(
        user_id=user_id,
        provider="github",
        connection_type="code_platform",
        account_identifier=account_identifier,
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


def _enrich_connection_for_api(row: Dict[str, Any]) -> Dict[str, Any]:
    from services.provider_capability_registry import resolve_capability_keys_for_row

    out = dict(row)
    if out.get("provider") == "microsoft" and out.get("connection_type") == "email":
        out["enabled_services"] = external_connections_service.get_enabled_services_from_metadata(
            out.get("provider_metadata")
        )
        meta = external_connections_service._parse_provider_metadata(out.get("provider_metadata"))
        out["devops_organization"] = meta.get("devops_organization") or ""
    out["capabilities"] = resolve_capability_keys_for_row(
        str(out.get("provider") or ""),
        str(out.get("connection_type") or ""),
        out.get("provider_metadata"),
    )
    return out


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
    return {"connections": [_enrich_connection_for_api(c) for c in connections]}


class MicrosoftUpdateServicesRequest(BaseModel):
    services: List[str]


@router.get("/connections/{connection_id:int}/microsoft/services")
async def get_microsoft_connection_services(
    connection_id: int,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Return enabled M365 services, all toggleable keys, and granted OAuth scopes."""
    conn = await external_connections_service.get_connection_by_id(
        connection_id, rls_context={"user_id": current_user.user_id}
    )
    if not conn or conn.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=404, detail="Connection not found")
    if conn.get("provider") != "microsoft" or conn.get("connection_type") != "email":
        raise HTTPException(status_code=400, detail="Not a Microsoft 365 email connection")
    scopes = conn.get("scopes") or []
    if hasattr(scopes, "tolist"):
        scopes = list(scopes)
    elif not isinstance(scopes, list):
        scopes = list(scopes) if scopes else []
    return {
        "enabled_services": external_connections_service.get_enabled_services_from_metadata(
            conn.get("provider_metadata")
        ),
        "available_services": list(M365_ALL_SERVICE_KEYS),
        "granted_scopes": [str(s) for s in scopes],
    }


@router.post("/connections/{connection_id:int}/microsoft/update-services")
async def microsoft_update_services(
    connection_id: int,
    body: MicrosoftUpdateServicesRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Update enabled M365 services; triggers re-authorization if new OAuth scopes are required."""
    conn = await external_connections_service.get_connection_by_id(
        connection_id, rls_context={"user_id": current_user.user_id}
    )
    if not conn or conn.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=404, detail="Connection not found")
    if conn.get("provider") != "microsoft" or conn.get("connection_type") != "email":
        raise HTTPException(status_code=400, detail="Not a Microsoft 365 email connection")
    normalized = normalize_m365_services(body.services)
    raw_scopes = conn.get("scopes") or []
    if not isinstance(raw_scopes, list):
        raw_scopes = list(raw_scopes) if raw_scopes else []
    granted = [str(s) for s in raw_scopes]
    missing = missing_scopes_for_services(normalized, granted)
    if missing:
        url, state, payload = _microsoft_authorize_url_for_user(
            user_id=current_user.user_id,
            services=normalized,
            connection_id=connection_id,
        )
        await _store_microsoft_oauth_state(state, payload)
        return {
            "reauth_required": True,
            "missing_scopes": missing,
            "authorize_url": url,
            "state": state,
            "enabled_services": normalized,
        }
    ok = await external_connections_service.update_enabled_services_only(
        connection_id, normalized, current_user.user_id, rls_context={"user_id": current_user.user_id}
    )
    if not ok:
        raise HTTPException(status_code=400, detail="Failed to update services")
    return {
        "reauth_required": False,
        "enabled_services": normalized,
        "missing_scopes": [],
    }


class DevopsConfigRequest(BaseModel):
    organization: str


@router.post("/connections/{connection_id:int}/microsoft/devops-config")
async def microsoft_devops_config(
    connection_id: int,
    body: DevopsConfigRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Set the Azure DevOps organization name on a Microsoft connection."""
    conn = await external_connections_service.get_connection_by_id(
        connection_id, rls_context={"user_id": current_user.user_id}
    )
    if not conn or conn.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=404, detail="Connection not found")
    if conn.get("provider") != "microsoft" or conn.get("connection_type") != "email":
        raise HTTPException(status_code=400, detail="Not a Microsoft 365 email connection")
    meta = external_connections_service._parse_provider_metadata(conn.get("provider_metadata"))
    org_name = (body.organization or "").strip()
    meta["devops_organization"] = org_name
    await external_connections_service.update_provider_metadata(
        connection_id, meta, rls_context={"user_id": current_user.user_id}
    )
    return {"success": True, "devops_organization": org_name}


class ConnectionLockUpdate(BaseModel):
    """Body for PATCH /connections/{connection_id} - lock toggle only."""
    is_locked: bool


@router.patch("/connections/{connection_id:int}")
async def update_connection_lock(
    connection_id: int,
    body: ConnectionLockUpdate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Set lock state on an external connection. Owner only; only is_locked and updated_at are updated."""
    conn = await external_connections_service.get_connection_by_id(connection_id)
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found")
    if conn.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not allowed to update this connection")
    updated = await external_connections_service.set_connection_lock(
        connection_id, current_user.user_id, body.is_locked,
        rls_context={"user_id": current_user.user_id},
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Connection not found")
    conn_after = await external_connections_service.get_connection_by_id(connection_id)
    return {"connection": conn_after}


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
    if conn.get("is_locked"):
        raise HTTPException(status_code=403, detail="Connection is locked; unlock to revoke")
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


@router.get("/connections/{connection_id:int}/known-chats")
async def get_known_chats(
    connection_id: int,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Get known chats for a messaging bot connection (chats that have messaged the bot). Used for recipient dropdown."""
    conn = await external_connections_service.get_connection_by_id(connection_id)
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found")
    if conn.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not allowed to view this connection")
    metadata = external_connections_service._parse_provider_metadata(conn.get("provider_metadata"))
    known_chats = metadata.get("known_chats") or []
    if not isinstance(known_chats, list):
        known_chats = []
    return {"known_chats": [c for c in known_chats if isinstance(c, dict) and c.get("chat_id")]}


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
    ok = await external_connections_service.refresh_access_token(
        connection_id, rls_context={"user_id": current_user.user_id}
    )
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


async def _validate_slack_token(bot_token: str) -> str:
    """Validate Slack bot token via auth.test and return bot user name."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            "https://slack.com/api/auth.test",
            headers={"Authorization": f"Bearer {bot_token}"},
        )
        resp.raise_for_status()
        data = resp.json()
    if not data.get("ok"):
        raise ValueError(data.get("error", "Invalid token"))
    return data.get("user", "") or ""


async def _validate_teams_credentials(app_id: str, app_password: str, tenant_id: str) -> None:
    """Validate Microsoft App ID + secret by requesting a Bot Framework access token."""
    tid = (tenant_id or "common").strip() or "common"
    token_url = f"https://login.microsoftonline.com/{tid}/oauth2/v2.0/token"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": app_id.strip(),
                "client_secret": app_password.strip(),
                "scope": "https://api.botframework.com/.default",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if resp.status_code != 200:
        raise ValueError(resp.text[:500] or f"HTTP {resp.status_code}")


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


@router.post("/connections/slack")
async def connect_slack_bot(
    body: SlackConnectRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Register a Slack bot for the current user (Socket Mode: bot_token + app_token)."""
    try:
        bot_username = await _validate_slack_token(body.bot_token)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Slack token: {e}")
    config = {"app_token": body.app_token}
    conn_id = await external_connections_service.store_connection(
        user_id=current_user.user_id,
        provider="slack",
        connection_type="chat_bot",
        account_identifier=bot_username or "slack_bot",
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
            provider="slack",
            bot_token=token,
            display_name=bot_username or "",
            config=config,
        )
        if not result.get("success") and result.get("error"):
            logger.warning("Connections-service RegisterBot failed: %s", result.get("error"))
    return {"connection_id": conn_id, "bot_username": bot_username}


@router.post("/connections/teams")
async def connect_teams_bot(
    body: TeamsConnectRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Register a Microsoft Teams bot (Azure Bot Framework App ID + client secret)."""
    app_id = (body.app_id or "").strip()
    app_password = (body.app_password or "").strip()
    tenant_id = (body.tenant_id or "common").strip() or "common"
    if not app_id or not app_password:
        raise HTTPException(status_code=400, detail="app_id and app_password are required")
    try:
        await _validate_teams_credentials(app_id, app_password, tenant_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Teams / Bot Framework credentials: {e}")
    config = {"app_password": app_password, "tenant_id": tenant_id}
    conn_id = await external_connections_service.store_connection(
        user_id=current_user.user_id,
        provider="teams",
        connection_type="chat_bot",
        account_identifier=app_id,
        access_token=app_id,
        refresh_token=app_id,
        expires_in=365 * 24 * 3600,
        scopes=["bot"],
        display_name=app_id,
        provider_metadata=config,
    )
    token = await external_connections_service.get_valid_access_token(conn_id)
    if token:
        from clients.connections_service_client import get_connections_service_client

        client = await get_connections_service_client()
        result = await client.register_bot(
            connection_id=conn_id,
            user_id=current_user.user_id,
            provider="teams",
            bot_token=token,
            display_name=app_id,
            config=config,
        )
        if not result.get("success") and result.get("error"):
            logger.warning("Connections-service RegisterBot failed: %s", result.get("error"))
    base = (settings.SITE_URL or "").rstrip("/")
    webhook_path = f"/teams/webhook/{conn_id}"
    webhook_url = f"{base}{webhook_path}" if base else webhook_path
    return {"connection_id": conn_id, "bot_username": app_id, "webhook_url": webhook_url}


@router.post("/connections/sms")
async def connect_sms(
    body: SMSConnectRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Register Twilio SMS for the current user (outbound-only). Validation is done by connections-service via RegisterBot."""
    config = {"account_sid": body.account_sid.strip(), "from_number": body.from_number.strip()}
    conn_id = await external_connections_service.store_connection(
        user_id=current_user.user_id,
        provider="sms",
        connection_type="chat_bot",
        account_identifier=body.from_number.strip(),
        access_token=body.auth_token.strip(),
        refresh_token=body.auth_token.strip(),
        expires_in=365 * 24 * 3600,
        scopes=["sms"],
        display_name=body.from_number.strip() or None,
        provider_metadata=config,
    )
    token = await external_connections_service.get_valid_access_token(conn_id)
    if token:
        from clients.connections_service_client import get_connections_service_client
        client = await get_connections_service_client()
        result = await client.register_bot(
            connection_id=conn_id,
            user_id=current_user.user_id,
            provider="sms",
            bot_token=token,
            display_name=body.from_number.strip() or "",
            config=config,
        )
        if not result.get("success") and result.get("error"):
            raise HTTPException(status_code=400, detail=result.get("error", "SMS connection validation failed"))
    return {"connection_id": conn_id, "bot_username": body.from_number.strip()}


@router.post("/connections/imap-smtp")
async def connect_imap_smtp(
    body: ImapSmtpConnectRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Register an IMAP/SMTP email account for the current user."""
    conn_id = await external_connections_service.store_imap_smtp_connection(
        user_id=current_user.user_id,
        imap_host=body.imap_host,
        imap_port=body.imap_port,
        imap_ssl=body.imap_ssl,
        smtp_host=body.smtp_host,
        smtp_port=body.smtp_port,
        smtp_tls=body.smtp_tls,
        username=body.username,
        imap_password=body.imap_password,
        smtp_password=body.smtp_password,
        display_name=body.display_name,
    )
    return {"connection_id": conn_id}


@router.post("/connections/caldav")
async def connect_caldav(
    body: CalDAVConnectRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Register a CalDAV calendar for the current user."""
    conn_id = await external_connections_service.store_caldav_connection(
        user_id=current_user.user_id,
        url=body.url,
        username=body.username,
        password=body.password,
        display_name=body.display_name,
    )
    return {"connection_id": conn_id}


