"""
External Connections Service - OAuth token storage and refresh for external providers.
Handles encryption, storage, and token refresh for Microsoft, Gmail, etc.
"""

import base64
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from config import settings
from services.database_manager.database_helpers import (
    execute,
    fetch_all,
    fetch_one,
    fetch_value,
)

logger = logging.getLogger(__name__)

MICROSOFT_TOKEN_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"


class ExternalConnectionsService:
    """Service for storing and refreshing OAuth tokens for external connections."""

    def __init__(self) -> None:
        self._fernet: Optional[Fernet] = None
        self._encryption_initialized = False

    def _initialize_encryption(self) -> None:
        if self._encryption_initialized:
            return
        master_key_str = getattr(settings, "SECRET_KEY", "") or ""
        if not master_key_str:
            logger.warning("SECRET_KEY not set; using temporary key for external connections")
            key = Fernet.generate_key()
            self._fernet = Fernet(key)
        else:
            try:
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b"external_connections_salt",
                    iterations=100000,
                    backend=default_backend(),
                )
                derived = base64.urlsafe_b64encode(kdf.derive(master_key_str.encode()))
                self._fernet = Fernet(derived)
            except Exception as e:
                logger.error("Failed to initialize external connections encryption: %s", e)
                key = Fernet.generate_key()
                self._fernet = Fernet(key)
        self._encryption_initialized = True

    def _encrypt_token(self, token: str) -> str:
        self._initialize_encryption()
        assert self._fernet is not None
        encrypted = self._fernet.encrypt(token.encode("utf-8"))
        return base64.b64encode(encrypted).decode("utf-8")

    def _decrypt_token(self, encrypted_token: str) -> str:
        self._initialize_encryption()
        assert self._fernet is not None
        raw = base64.b64decode(encrypted_token.encode("utf-8"))
        return self._fernet.decrypt(raw).decode("utf-8")

    async def store_connection(
        self,
        user_id: str,
        provider: str,
        connection_type: str,
        account_identifier: str,
        access_token: str,
        refresh_token: str,
        expires_in: int,
        scopes: List[str],
        display_name: Optional[str] = None,
        provider_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Store a new external connection with encrypted tokens."""
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        enc_access = self._encrypt_token(access_token)
        enc_refresh = self._encrypt_token(refresh_token)
        metadata_json = provider_metadata or {}
        metadata_str = json.dumps(metadata_json)
        conn_id = await fetch_value(
            """
            INSERT INTO external_connections (
                user_id, provider, connection_type, account_identifier, display_name,
                encrypted_access_token, encrypted_refresh_token, token_expires_at,
                scopes, provider_metadata, created_at, updated_at, is_active, connection_status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW(), NOW(), true, 'active')
            ON CONFLICT (user_id, provider, connection_type, account_identifier)
            DO UPDATE SET
                encrypted_access_token = EXCLUDED.encrypted_access_token,
                encrypted_refresh_token = EXCLUDED.encrypted_refresh_token,
                token_expires_at = EXCLUDED.token_expires_at,
                scopes = EXCLUDED.scopes,
                provider_metadata = EXCLUDED.provider_metadata,
                display_name = EXCLUDED.display_name,
                updated_at = NOW(),
                is_active = true,
                connection_status = 'active'
            RETURNING id
            """,
            user_id,
            provider,
            connection_type,
            account_identifier,
            display_name,
            enc_access,
            enc_refresh,
            expires_at,
            scopes,
            metadata_str,
        )
        logger.info("Stored connection id=%s for user=%s provider=%s", conn_id, user_id, provider)
        return int(conn_id)

    async def get_connection_by_id(
        self,
        connection_id: int,
        rls_context: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get connection row by id (without decrypted tokens). Pass rls_context for admin access."""
        row = await fetch_one(
            """
            SELECT id, user_id, provider, connection_type, account_identifier, display_name,
                   token_expires_at, scopes, provider_metadata, created_at, updated_at,
                   last_sync_at, is_active, connection_status
            FROM external_connections
            WHERE id = $1 AND is_active = true
            """,
            connection_id,
            rls_context=rls_context,
        )
        return dict(row) if row else None

    async def get_valid_access_token(
        self, connection_id: int, rls_context: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Return a valid access token, refreshing if necessary."""
        row = await fetch_one(
            """
            SELECT id, encrypted_access_token, encrypted_refresh_token, token_expires_at,
                   provider, provider_metadata
            FROM external_connections
            WHERE id = $1 AND is_active = true
            """,
            connection_id,
            rls_context=rls_context,
        )
        if not row:
            return None
        expires_at = row["token_expires_at"]
        if expires_at is not None and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        buffer_seconds = 300
        now_utc = datetime.now(timezone.utc)
        if expires_at and (expires_at - timedelta(seconds=buffer_seconds)) > now_utc:
            return self._decrypt_token(row["encrypted_access_token"])
        refreshed = await self.refresh_access_token(connection_id)
        if not refreshed:
            return None
        row2 = await fetch_one(
            """
            SELECT encrypted_access_token FROM external_connections WHERE id = $1
            """,
            connection_id,
        )
        return self._decrypt_token(row2["encrypted_access_token"]) if row2 else None

    async def refresh_access_token(self, connection_id: int) -> bool:
        """Refresh access token using refresh_token. Returns True on success."""
        row = await fetch_one(
            """
            SELECT id, encrypted_refresh_token, provider, provider_metadata
            FROM external_connections
            WHERE id = $1 AND is_active = true
            """,
            connection_id,
        )
        if not row:
            return False
        provider = row["provider"]
        refresh_token = self._decrypt_token(row["encrypted_refresh_token"])
        metadata = row["provider_metadata"] or {}
        if provider == "microsoft":
            tenant = metadata.get("tenant_id") or getattr(settings, "MICROSOFT_TENANT_ID", "common")
            client_id = getattr(settings, "MICROSOFT_CLIENT_ID", "")
            client_secret = getattr(settings, "MICROSOFT_CLIENT_SECRET", "")
            if not client_id or not client_secret:
                logger.error("Microsoft OAuth credentials not configured")
                return False
            url = MICROSOFT_TOKEN_URL.format(tenant=tenant)
            data = {
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            }
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(url, data=data, timeout=15.0)
                    resp.raise_for_status()
                    body = resp.json()
            except Exception as e:
                logger.exception("Microsoft token refresh failed for connection_id=%s: %s", connection_id, e)
                return False
            access_token = body.get("access_token")
            new_refresh = body.get("refresh_token") or refresh_token
            expires_in = int(body.get("expires_in", 3600))
            await self._update_tokens(connection_id, access_token, new_refresh, expires_in)
            return True
        logger.warning("Token refresh not implemented for provider=%s", provider)
        return False

    async def _update_tokens(
        self,
        connection_id: int,
        access_token: str,
        refresh_token: str,
        expires_in: int,
    ) -> None:
        """Update stored tokens after refresh."""
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        enc_access = self._encrypt_token(access_token)
        enc_refresh = self._encrypt_token(refresh_token)
        await execute(
            """
            UPDATE external_connections
            SET encrypted_access_token = $1, encrypted_refresh_token = $2,
                token_expires_at = $3, updated_at = NOW()
            WHERE id = $4
            """,
            enc_access,
            enc_refresh,
            expires_at,
            connection_id,
        )

    async def revoke_connection(self, connection_id: int) -> bool:
        """Mark connection as inactive (soft revoke)."""
        await execute(
            """
            UPDATE external_connections
            SET is_active = false, connection_status = 'revoked', updated_at = NOW()
            WHERE id = $1
            """,
            connection_id,
        )
        logger.info("Revoked connection id=%s", connection_id)
        return True

    async def get_user_connections(
        self,
        user_id: str,
        provider: Optional[str] = None,
        connection_type: Optional[str] = None,
        active_only: bool = True,
        rls_context: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """List connections for a user. Tokens are not included."""
        conditions = ["user_id = $1"]
        args: List[Any] = [user_id]
        if active_only:
            conditions.append("is_active = true")
        if provider:
            conditions.append("provider = $2")
            args.append(provider)
        if connection_type:
            idx = len(args) + 1
            conditions.append(f"connection_type = ${idx}")
            args.append(connection_type)
        where = " AND ".join(conditions)
        rows = await fetch_all(
            f"""
            SELECT id, user_id, provider, connection_type, account_identifier, display_name,
                   token_expires_at, scopes, provider_metadata, created_at, updated_at,
                   last_sync_at, is_active, connection_status
            FROM external_connections
            WHERE {where}
            ORDER BY provider, connection_type, account_identifier
            """,
            *args,
            rls_context=rls_context,
        )
        return [dict(r) for r in rows]

    SYSTEM_EMAIL_KEY = "system_email_connection_id"

    async def get_system_email_connection_id(
        self, rls_context: Optional[Dict[str, str]] = None
    ) -> Optional[int]:
        """Get the designated system email connection id (admin-only table)."""
        ctx = rls_context or {"user_id": "", "user_role": "admin"}
        row = await fetch_value(
            "SELECT value FROM system_settings WHERE key = $1",
            self.SYSTEM_EMAIL_KEY,
            rls_context=ctx,
        )
        if not row or row == "":
            return None
        try:
            return int(row)
        except (ValueError, TypeError):
            return None

    async def set_system_email_connection_id(
        self,
        connection_id: Optional[int],
        rls_context: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set or clear the designated system email connection id (admin-only)."""
        ctx = rls_context or {"user_id": "", "user_role": "admin"}
        value = str(connection_id) if connection_id is not None else ""
        await execute(
            """
            INSERT INTO system_settings (key, value) VALUES ($1, $2)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """,
            self.SYSTEM_EMAIL_KEY,
            value,
            rls_context=ctx,
        )

    SMTP_KEYS = (
        "smtp_enabled",
        "smtp_host",
        "smtp_port",
        "smtp_user",
        "smtp_password_encrypted",
        "smtp_use_tls",
        "smtp_from_email",
        "smtp_from_name",
    )

    async def get_smtp_settings(
        self, rls_context: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Get SMTP settings from system_settings (admin-only). Password never returned; password_set indicates if stored."""
        ctx = rls_context or {"user_id": "", "user_role": "admin"}
        rows = await fetch_all(
            "SELECT key, value FROM system_settings WHERE key = ANY($1::text[])",
            list(self.SMTP_KEYS),
            rls_context=ctx,
        )
        by_key = {r["key"]: r["value"] for r in rows} if rows else {}
        port = by_key.get("smtp_port", "")
        try:
            port_int = int(port) if port else 587
        except (ValueError, TypeError):
            port_int = 587
        return {
            "enabled": (by_key.get("smtp_enabled") or "").lower() == "true",
            "host": by_key.get("smtp_host") or "",
            "port": port_int,
            "user": by_key.get("smtp_user") or "",
            "from_email": by_key.get("smtp_from_email") or "",
            "from_name": by_key.get("smtp_from_name") or "",
            "use_tls": (by_key.get("smtp_use_tls") or "true").lower() == "true",
            "password_set": bool(by_key.get("smtp_password_encrypted")),
        }

    async def get_smtp_settings_for_sending(
        self, rls_context: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get full SMTP config for sending (includes decrypted password). Returns None if not configured or disabled."""
        ctx = rls_context or {"user_id": "", "user_role": "admin"}
        rows = await fetch_all(
            "SELECT key, value FROM system_settings WHERE key = ANY($1::text[])",
            list(self.SMTP_KEYS),
            rls_context=ctx,
        )
        by_key = {r["key"]: r["value"] for r in rows} if rows else {}
        if (by_key.get("smtp_enabled") or "").lower() != "true":
            return None
        host = (by_key.get("smtp_host") or "").strip()
        if not host:
            return None
        port = by_key.get("smtp_port", "")
        try:
            port_int = int(port) if port else 587
        except (ValueError, TypeError):
            port_int = 587
        enc = by_key.get("smtp_password_encrypted") or ""
        try:
            password = self._decrypt_token(enc) if enc else ""
        except Exception:
            logger.warning("Failed to decrypt stored SMTP password; SMTP config skipped")
            return None
        return {
            "host": host,
            "port": port_int,
            "user": by_key.get("smtp_user") or "",
            "password": password,
            "use_tls": (by_key.get("smtp_use_tls") or "true").lower() == "true",
            "from_email": by_key.get("smtp_from_email") or "",
            "from_name": by_key.get("smtp_from_name") or "",
        }

    async def set_smtp_settings(
        self,
        enabled: bool,
        host: str = "",
        port: int = 587,
        user: str = "",
        password: Optional[str] = None,
        use_tls: bool = True,
        from_email: str = "",
        from_name: str = "",
        rls_context: Optional[Dict[str, str]] = None,
    ) -> None:
        """Store SMTP settings in system_settings (admin-only). Password is encrypted at rest
        using Fernet (SECRET_KEY); never stored in plaintext. Omit password to keep existing."""
        ctx = rls_context or {"user_id": "", "user_role": "admin"}
        pairs = [
            ("smtp_enabled", "true" if enabled else "false"),
            ("smtp_host", (host or "").strip()),
            ("smtp_port", str(port)),
            ("smtp_user", (user or "").strip()),
            ("smtp_use_tls", "true" if use_tls else "false"),
            ("smtp_from_email", (from_email or "").strip()),
            ("smtp_from_name", (from_name or "").strip()),
        ]
        if password is not None:
            pairs.append(
                ("smtp_password_encrypted", self._encrypt_token(password) if password else "")
            )
        for key, value in pairs:
            await execute(
                """
                INSERT INTO system_settings (key, value) VALUES ($1, $2)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                key,
                value,
                rls_context=ctx,
            )


external_connections_service = ExternalConnectionsService()
