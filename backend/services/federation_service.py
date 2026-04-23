"""
Federation Phase 1: instance keypair, peer pairing, connectivity probe, outbox.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from nacl.exceptions import BadSignatureError
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from config import settings
from services.database_manager.database_helpers import (
    execute,
    execute_transaction,
    fetch_all,
    fetch_one,
    fetch_value,
)
from utils import federation_crypto

logger = logging.getLogger(__name__)

ADMIN_RLS = {"user_id": "", "user_role": "admin"}

KEY_PRIVATE_ENC = "federation_private_key_encrypted"
KEY_PUBLIC = "federation_public_key"

BFP_VERSION = "1"


def _normalize_instance_url(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if not u:
        return ""
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    parsed = urllib.parse.urlparse(u)
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def _canonical_json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


class FederationService:
    """Singleton-style federation operations."""

    def __init__(self) -> None:
        self._fernet: Optional[Fernet] = None
        self._inbound_rate_timestamps: Dict[str, deque] = {}

    def _fernet_cipher(self) -> Fernet:
        if self._fernet is not None:
            return self._fernet
        master = (settings.SECRET_KEY or "").encode("utf-8")
        if not master:
            raise ValueError("SECRET_KEY is required for federation key encryption")
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"federation_instance_key_salt_v1",
            iterations=100_000,
            backend=default_backend(),
        )
        key = __import__("base64").urlsafe_b64encode(kdf.derive(master))
        self._fernet = Fernet(key)
        return self._fernet

    async def _get_setting(self, key: str, rls: Optional[Dict[str, str]] = None) -> Optional[str]:
        ctx = rls or ADMIN_RLS
        row = await fetch_value(
            "SELECT value FROM system_settings WHERE key = $1",
            key,
            rls_context=ctx,
        )
        if row is None or row == "":
            return None
        return str(row)

    async def _set_setting(self, key: str, value: str, rls: Optional[Dict[str, str]] = None) -> None:
        ctx = rls or ADMIN_RLS
        await execute(
            """
            INSERT INTO system_settings (key, value) VALUES ($1, $2)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """,
            key,
            value,
            rls_context=ctx,
        )

    async def has_keypair(self, rls: Optional[Dict[str, str]] = None) -> bool:
        pub = await self._get_setting(KEY_PUBLIC, rls)
        return bool(pub)

    async def get_public_key_b64(self, rls: Optional[Dict[str, str]] = None) -> Optional[str]:
        return await self._get_setting(KEY_PUBLIC, rls)

    async def _get_private_key_b64(self, rls: Optional[Dict[str, str]] = None) -> Optional[str]:
        enc = await self._get_setting(KEY_PRIVATE_ENC, rls)
        if not enc:
            return None
        try:
            raw = self._fernet_cipher().decrypt(enc.encode("utf-8"))
            return raw.decode("utf-8")
        except (InvalidToken, ValueError, TypeError) as e:
            logger.error("Failed to decrypt federation private key: %s", e)
            return None

    async def get_private_key_b64_for_signing(self, rls: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Return decrypted Ed25519 private key seed (base64) for outbound signing."""
        return await self._get_private_key_b64(rls)

    async def initialize_keypair(self, rls: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Generate and persist Ed25519 keypair if missing. Returns public key (always) and whether created."""
        ctx = rls or ADMIN_RLS
        existing_pub = await self._get_setting(KEY_PUBLIC, ctx)
        if existing_pub:
            return {"public_key": existing_pub, "created": False}
        priv, pub = federation_crypto.generate_keypair()
        enc = self._fernet_cipher().encrypt(priv.encode("utf-8")).decode("utf-8")
        await self._set_setting(KEY_PRIVATE_ENC, enc, ctx)
        await self._set_setting(KEY_PUBLIC, pub, ctx)
        return {"public_key": pub, "created": True}

    async def regenerate_keypair(self, rls: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        ctx = rls or ADMIN_RLS
        priv, pub = federation_crypto.generate_keypair()
        enc = self._fernet_cipher().encrypt(priv.encode("utf-8")).decode("utf-8")
        await self._set_setting(KEY_PRIVATE_ENC, enc, ctx)
        await self._set_setting(KEY_PUBLIC, pub, ctx)
        return {"public_key": pub, "created": True}

    def _display_name(self) -> str:
        name = (settings.FEDERATION_DISPLAY_NAME or "").strip()
        if name:
            return name
        parsed = urllib.parse.urlparse(settings.SITE_URL)
        return parsed.netloc or "bastion"

    async def get_instance_identity(self, rls: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        pub = await self.get_public_key_b64(rls)
        if not pub:
            return None
        return {
            "instance_url": _normalize_instance_url(settings.SITE_URL),
            "public_key": pub,
            "display_name": self._display_name(),
            "bfp_version": BFP_VERSION,
        }

    async def fetch_remote_identity(self, peer_url: str) -> Dict[str, Any]:
        base = _normalize_instance_url(peer_url)
        url = f"{base}/api/federation/identity"
        timeout = float(settings.FEDERATION_HTTP_TIMEOUT)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.json()

    async def connectivity_probe(self, peer_base_url: str) -> str:
        """
        Return 'bidirectional' if peer identity endpoint is reachable,
        'asymmetric_listener' if not (caller will hold outbox for that peer).
        """
        base = _normalize_instance_url(peer_base_url)
        url = f"{base}/api/federation/identity"
        timeout = float(settings.FEDERATION_CONNECTIVITY_PROBE_TIMEOUT)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(url)
                if r.status_code == 200:
                    return "bidirectional"
        except Exception as e:
            logger.info("Federation connectivity probe failed for %s: %s", base, e)
        return "asymmetric_listener"

    async def initiate_pairing(
        self,
        peer_url: str,
        admin_user_id: str,
        rls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        ctx = rls or ADMIN_RLS
        await self.initialize_keypair(ctx)
        priv = await self._get_private_key_b64(ctx)
        if not priv:
            raise ValueError("Federation keypair not available")
        remote = await self.fetch_remote_identity(peer_url)
        remote_base = _normalize_instance_url(peer_url)
        remote_pub = remote.get("public_key") or ""
        if not remote_pub:
            raise ValueError("Remote instance did not return a public key")
        dup = await fetch_one(
            "SELECT peer_id FROM federation_peers WHERE peer_url = $1",
            remote_base,
            rls_context=ctx,
        )
        if dup:
            raise ValueError("A peer with this URL already exists")
        await execute(
            """
            INSERT INTO federation_peers (
                peer_url, peer_public_key, display_name, status, connectivity_mode,
                allowed_scopes, initiated_by, is_inbound, metadata
            ) VALUES ($1, $2, $3, 'pending', 'bidirectional', ARRAY['messaging']::TEXT[], $4, FALSE, '{}'::jsonb)
            """,
            remote_base,
            remote_pub,
            remote.get("display_name") or remote_base,
            admin_user_id,
            rls_context=ctx,
        )
        local_identity = await self.get_instance_identity(ctx)
        payload = {
            "bfp_version": BFP_VERSION,
            "event_type": "peer_request",
            "from_instance": local_identity["instance_url"],
            "from_public_key": local_identity["public_key"],
            "from_display_name": local_identity["display_name"],
        }
        body_bytes = _canonical_json_bytes(payload)
        sig = federation_crypto.sign_payload(priv, body_bytes)
        out_url = f"{remote_base}/api/federation/peer-request"
        async with httpx.AsyncClient(timeout=float(settings.FEDERATION_HTTP_TIMEOUT)) as client:
            r = await client.post(
                out_url,
                content=body_bytes,
                headers={
                    "Content-Type": "application/json",
                    "X-Bastion-Signature": sig,
                    "X-Bastion-Instance": local_identity["instance_url"],
                },
            )
            r.raise_for_status()
        return {"ok": True, "peer_url": remote_base}

    async def receive_signed_peer_message(
        self,
        raw_body: bytes,
        signature_b64: Optional[str],
        from_instance_header: Optional[str],
    ) -> Dict[str, Any]:
        """
        Handle POST /api/federation/peer-request from a remote instance (signed body, no JWT).
        """
        ctx = ADMIN_RLS
        if not signature_b64:
            raise ValueError("Missing X-Bastion-Signature header")
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e
        event_type = payload.get("event_type")
        from_public_key = payload.get("from_public_key") or ""
        from_instance = _normalize_instance_url(
            payload.get("from_instance") or from_instance_header or ""
        )
        if not from_public_key or not from_instance:
            raise ValueError("from_public_key and from_instance are required")
        try:
            federation_crypto.verify_signature(from_public_key, raw_body, signature_b64)
        except BadSignatureError as e:
            raise ValueError("Invalid federation signature") from e

        if event_type == "peer_request":
            return await self._handle_inbound_peer_request(payload, ctx)
        if event_type == "peer_accept":
            return await self._handle_peer_accept(payload, ctx)
        raise ValueError(f"Unsupported event_type: {event_type}")

    async def _handle_inbound_peer_request(
        self, payload: Dict[str, Any], ctx: Dict[str, str]
    ) -> Dict[str, Any]:
        from_instance = _normalize_instance_url(payload.get("from_instance") or "")
        from_public_key = payload.get("from_public_key") or ""
        mode = await self.connectivity_probe(from_instance)
        dup = await fetch_one(
            "SELECT peer_id FROM federation_peers WHERE peer_url = $1",
            from_instance,
            rls_context=ctx,
        )
        if dup:
            return {"ok": True, "duplicate": True}
        await execute(
            """
            INSERT INTO federation_peers (
                peer_url, peer_public_key, display_name, status, connectivity_mode,
                allowed_scopes, initiated_by, is_inbound, metadata
            ) VALUES ($1, $2, $3, 'pending', $4, ARRAY['messaging']::TEXT[], NULL, TRUE, '{}'::jsonb)
            """,
            from_instance,
            from_public_key,
            payload.get("from_display_name") or from_instance,
            mode,
            rls_context=ctx,
        )
        return {"ok": True, "connectivity_mode": mode}

    async def _handle_peer_accept(self, payload: Dict[str, Any], ctx: Dict[str, str]) -> Dict[str, Any]:
        from_instance = _normalize_instance_url(payload.get("from_instance") or "")
        from_public_key = payload.get("from_public_key") or ""
        peer_mode = payload.get("peer_connectivity_mode") or "bidirectional"
        local_mode = (
            "asymmetric_caller" if peer_mode == "asymmetric_listener" else "bidirectional"
        )
        row = await fetch_one(
            """
            SELECT peer_id, peer_public_key, status FROM federation_peers
            WHERE peer_url = $1 AND is_inbound = FALSE AND status = 'pending'
            """,
            from_instance,
            rls_context=ctx,
        )
        if not row:
            raise ValueError("No matching outbound peer request for this acceptance")
        if row.get("peer_public_key") != from_public_key:
            raise ValueError("Public key mismatch for peer acceptance")
        await execute(
            """
            UPDATE federation_peers
            SET status = 'active', activated_at = NOW(), connectivity_mode = $2, metadata = metadata
            WHERE peer_id = $1
            """,
            row["peer_id"],
            local_mode,
            rls_context=ctx,
        )
        return {"ok": True, "peer_id": str(row["peer_id"])}

    async def get_peer_by_id(
        self, peer_id: str, rls: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        ctx = rls or ADMIN_RLS
        row = await fetch_one(
            """
            SELECT peer_id, peer_url, peer_public_key, status, connectivity_mode
            FROM federation_peers WHERE peer_id = $1::uuid
            """,
            peer_id,
            rls_context=ctx,
        )
        return dict(row) if row else None

    async def list_peers(self, rls: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        ctx = rls or ADMIN_RLS
        rows = await fetch_all(
            """
            SELECT p.peer_id, p.peer_url, p.peer_public_key, p.display_name, p.status, p.connectivity_mode,
                   p.allowed_scopes, p.initiated_by, p.is_inbound, p.created_at, p.activated_at, p.metadata,
                   COALESCE(oc.cnt, 0)::bigint AS outbox_pending_count
            FROM federation_peers p
            LEFT JOIN (
                SELECT peer_id, COUNT(*)::bigint AS cnt
                FROM federation_outbox
                WHERE picked_up_at IS NULL
                GROUP BY peer_id
            ) oc ON oc.peer_id = p.peer_id
            ORDER BY p.created_at DESC
            """,
            rls_context=ctx,
        )
        return [dict(r) for r in rows]

    async def verify_signed_peer_messaging_scope(
        self,
        raw_body: bytes,
        signature_b64: Optional[str],
        from_instance_header: Optional[str],
        rls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Verify Ed25519 signature and that caller is an active peer with messaging scope."""
        ctx = rls or ADMIN_RLS
        if not signature_b64:
            raise ValueError("Missing X-Bastion-Signature header")
        caller = _normalize_instance_url(from_instance_header or "")
        if not caller:
            raise ValueError("Missing X-Bastion-Instance header")
        row = await fetch_one(
            """
            SELECT peer_id, peer_url, peer_public_key, status, allowed_scopes
            FROM federation_peers
            WHERE peer_url = $1 AND status = 'active'
            """,
            caller,
            rls_context=ctx,
        )
        if not row:
            raise ValueError("Unknown or inactive peer for this instance URL")
        try:
            federation_crypto.verify_signature(
                row["peer_public_key"], raw_body, signature_b64
            )
        except BadSignatureError as e:
            raise ValueError("Invalid federation signature") from e
        scopes = row.get("allowed_scopes") or []
        if isinstance(scopes, str):
            scopes = [scopes]
        if "messaging" not in list(scopes):
            raise ValueError("Peer does not have messaging scope")
        return dict(row)

    async def resolve_discoverable_user(
        self, address: str, rls: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Lookup local user by federated address; only if federation_discoverable and active."""
        ctx = rls or ADMIN_RLS
        s = (address or "").strip()
        if "@" not in s:
            return {"found": False}
        username, _host = s.rsplit("@", 1)
        username = (username or "").strip()
        if not username:
            return {"found": False}
        row = await fetch_one(
            """
            SELECT username, display_name, avatar_url
            FROM users
            WHERE lower(username) = lower($1)
              AND federation_discoverable = TRUE
              AND is_active = TRUE
            """,
            username,
            rls_context=ctx,
        )
        if not row:
            return {"found": False}
        return {
            "found": True,
            "username": row["username"],
            "display_name": row.get("display_name"),
            "avatar_url": row.get("avatar_url"),
        }

    async def list_discoverable_users_directory(
        self,
        search: Optional[str],
        limit: int,
        offset: int,
        rls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        ctx = rls or ADMIN_RLS
        lim = max(1, min(int(limit or 50), 200))
        off = max(0, int(offset or 0))
        term = (search or "").strip().lower()
        if term:
            like = f"%{term}%"
            rows = await fetch_all(
                """
                SELECT username, display_name, avatar_url
                FROM users
                WHERE federation_discoverable = TRUE AND is_active = TRUE
                  AND (
                    lower(username) LIKE $1
                    OR lower(COALESCE(display_name, '')) LIKE $1
                  )
                ORDER BY username ASC
                LIMIT $2 OFFSET $3
                """,
                like,
                lim,
                off,
                rls_context=ctx,
            )
            total = await fetch_value(
                """
                SELECT COUNT(*)::bigint FROM users
                WHERE federation_discoverable = TRUE AND is_active = TRUE
                  AND (
                    lower(username) LIKE $1
                    OR lower(COALESCE(display_name, '')) LIKE $1
                  )
                """,
                like,
                rls_context=ctx,
            )
        else:
            rows = await fetch_all(
                """
                SELECT username, display_name, avatar_url
                FROM users
                WHERE federation_discoverable = TRUE AND is_active = TRUE
                ORDER BY username ASC
                LIMIT $1 OFFSET $2
                """,
                lim,
                off,
                rls_context=ctx,
            )
            total = await fetch_value(
                """
                SELECT COUNT(*)::bigint FROM users
                WHERE federation_discoverable = TRUE AND is_active = TRUE
                """,
                rls_context=ctx,
            )
        users = [
            {
                "username": r["username"],
                "display_name": r.get("display_name"),
                "avatar_url": r.get("avatar_url"),
            }
            for r in rows
        ]
        return {"users": users, "total": int(total or 0), "limit": lim, "offset": off}

    async def find_active_peer_by_address_host(
        self, host: str, rls: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Match remote host (e.g. bastion-b.com) to an active peer by peer_url netloc."""
        ctx = rls or ADMIN_RLS
        want = (host or "").strip().lower().rstrip(".")
        if not want:
            return None
        rows = await fetch_all(
            """
            SELECT peer_id, peer_url, peer_public_key, display_name, status, connectivity_mode
            FROM federation_peers
            WHERE status = 'active'
            """,
            rls_context=ctx,
        )
        for r in rows:
            nu = urllib.parse.urlparse((r.get("peer_url") or "").strip()).netloc.lower().rstrip(".")
            if nu == want:
                return dict(r)
        return None

    async def update_peer_last_sync_at(
        self, peer_id: str, rls: Optional[Dict[str, str]] = None
    ) -> None:
        ctx = rls or ADMIN_RLS
        ts = datetime.now(timezone.utc).isoformat()
        patch = json.dumps({"last_federation_sync_at": ts})
        await execute(
            """
            UPDATE federation_peers
            SET metadata = COALESCE(metadata, '{}'::jsonb) || $2::jsonb
            WHERE peer_id = $1::uuid
            """,
            peer_id,
            patch,
            rls_context=ctx,
        )

    async def prune_outbox(self, rls: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Remove stale outbox rows by age and per-peer cap (oldest first beyond cap)."""
        ctx = rls or ADMIN_RLS
        max_age = int(getattr(settings, "FEDERATION_OUTBOX_MAX_AGE_HOURS", 72) or 72)
        max_per = int(getattr(settings, "FEDERATION_OUTBOX_MAX_PER_PEER", 5000) or 5000)
        age_delta = timedelta(hours=max_age)
        row_age = await fetch_one(
            """
            WITH del AS (
                DELETE FROM federation_outbox
                WHERE created_at < NOW() - $1::interval
                RETURNING outbox_id
            )
            SELECT COUNT(*)::bigint AS n FROM del
            """,
            age_delta,
            rls_context=ctx,
        )
        da = int((row_age or {}).get("n") or 0)
        row_cap = await fetch_one(
            """
            WITH ranked AS (
                SELECT outbox_id, peer_id,
                       ROW_NUMBER() OVER (PARTITION BY peer_id ORDER BY created_at DESC) AS rn
                FROM federation_outbox
            ),
            del AS (
                DELETE FROM federation_outbox o
                USING ranked r
                WHERE o.outbox_id = r.outbox_id AND r.rn > $1
                RETURNING o.outbox_id
            )
            SELECT COUNT(*)::bigint AS n FROM del
            """,
            max_per,
            rls_context=ctx,
        )
        dc = int((row_cap or {}).get("n") or 0)
        if da or dc:
            logger.warning(
                "Federation outbox pruned: %s rows by max age (%sh), %s rows by per-peer cap (%s)",
                da,
                max_age,
                dc,
                max_per,
            )
        return {"deleted_by_age": da, "deleted_by_cap": dc}

    async def signed_post_json_to_peer(
        self,
        peer_base_url: str,
        path: str,
        body: Dict[str, Any],
        rls: Optional[Dict[str, str]] = None,
    ) -> httpx.Response:
        """POST canonical JSON to a peer with local instance signature headers."""
        ctx = rls or ADMIN_RLS
        await self.initialize_keypair(ctx)
        priv = await self._get_private_key_b64(ctx)
        if not priv:
            raise ValueError("Federation keypair not available")
        ident = await self.get_instance_identity(ctx)
        if not ident:
            raise ValueError("Federation identity not available")
        body_bytes = _canonical_json_bytes(body)
        sig = federation_crypto.sign_payload(priv, body_bytes)
        base = _normalize_instance_url(peer_base_url)
        url = f"{base}{path if path.startswith('/') else '/' + path}"
        async with httpx.AsyncClient(timeout=float(settings.FEDERATION_HTTP_TIMEOUT)) as client:
            return await client.post(
                url,
                content=body_bytes,
                headers={
                    "Content-Type": "application/json",
                    "X-Bastion-Signature": sig,
                    "X-Bastion-Instance": ident["instance_url"],
                },
            )

    async def update_peer_status(
        self,
        peer_id: str,
        new_status: str,
        rls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        ctx = rls or ADMIN_RLS
        allowed = {"active", "suspended", "revoked"}
        if new_status not in allowed:
            raise ValueError(f"status must be one of {allowed}")
        row = await fetch_one(
            "SELECT * FROM federation_peers WHERE peer_id = $1::uuid",
            peer_id,
            rls_context=ctx,
        )
        if not row:
            raise ValueError("Peer not found")
        if new_status == "active":
            if not row.get("is_inbound"):
                raise ValueError("Only inbound pending peers can be approved this way")
            if row.get("status") != "pending":
                raise ValueError("Peer is not pending approval")
            await self._approve_inbound_peer_row(dict(row), ctx)
            return {"ok": True, "peer_id": peer_id, "status": "active"}
        await execute(
            """
            UPDATE federation_peers SET status = $2 WHERE peer_id = $1::uuid
            """,
            peer_id,
            new_status,
            rls_context=ctx,
        )
        await self.log_federation_audit(
            action=f"federation.peer_status.{new_status}",
            record_id=str(peer_id),
            new_values={"peer_id": str(peer_id), "status": new_status},
            rls=ctx,
        )
        return {"ok": True, "peer_id": peer_id, "status": new_status}

    async def delete_revoked_peer(
        self,
        peer_id: str,
        admin_user_id: str,
        rls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Permanently remove a revoked peer row so peer_url can be re-used.
        Clears chat_messages.federated_sender_id for federated users of this peer
        so CASCADE deletes are not blocked by FK.
        """
        ctx = rls or ADMIN_RLS
        removed: Dict[str, str] = {}

        async def _tx(conn: Any) -> None:
            uid = "" if admin_user_id is None else str(admin_user_id)
            await conn.execute("SELECT set_config('app.current_user_id', $1, false)", uid)
            await conn.execute("SELECT set_config('app.current_user_role', $1, false)", "admin")

            row = await conn.fetchrow(
                """
                SELECT peer_id, status, peer_url FROM federation_peers
                WHERE peer_id = $1::uuid FOR UPDATE
                """,
                peer_id,
            )
            if not row:
                raise ValueError("Peer not found")
            if (row.get("status") or "") != "revoked":
                raise ValueError("Only revoked peers can be removed")

            await conn.execute(
                """
                UPDATE chat_messages SET federated_sender_id = NULL
                WHERE federated_sender_id IN (
                    SELECT federated_user_id FROM federated_users WHERE peer_id = $1::uuid
                )
                """,
                peer_id,
            )
            deleted = await conn.fetchrow(
                """
                DELETE FROM federation_peers
                WHERE peer_id = $1::uuid AND status = 'revoked'
                RETURNING peer_url
                """,
                peer_id,
            )
            if not deleted:
                raise ValueError("Peer could not be removed")
            removed["peer_url"] = str(deleted.get("peer_url") or "")

        try:
            await execute_transaction([_tx])
        except Exception as e:
            prefix = "Database transaction failed: "
            msg = str(e)
            if msg.startswith(prefix):
                inner = msg[len(prefix) :].strip()
                if inner in (
                    "Peer not found",
                    "Only revoked peers can be removed",
                    "Peer could not be removed",
                ):
                    raise ValueError(inner) from e
            raise

        await self.log_federation_audit(
            action="federation.peer_removed",
            record_id=str(peer_id),
            new_values={
                "peer_id": str(peer_id),
                "peer_url": removed.get("peer_url", ""),
            },
            user_id=str(admin_user_id) if admin_user_id else None,
            rls=ctx,
        )
        return {"ok": True, "peer_id": str(peer_id), "peer_url": removed.get("peer_url", "")}

    def check_inbound_federation_rate_limit(self, peer_url_key: str) -> Tuple[bool, int]:
        """
        Sliding-window per-peer rate limit for signed inbound federation HTTP.
        Returns (allowed, retry_after_seconds). When not allowed, does not record a hit.
        """
        lim = int(getattr(settings, "FEDERATION_RATE_LIMIT_PER_PEER_PER_MINUTE", 120) or 120)
        if lim <= 0:
            return True, 0
        key = _normalize_instance_url(peer_url_key) or peer_url_key
        now = time.monotonic()
        window = 60.0
        dq = self._inbound_rate_timestamps.setdefault(key, deque())
        while dq and (now - dq[0]) > window:
            dq.popleft()
        if len(dq) >= lim:
            retry_after = int(max(1.0, window - (now - dq[0])) + 0.999)
            return False, retry_after
        dq.append(now)
        return True, 0

    async def log_federation_audit(
        self,
        action: str,
        record_id: Optional[str] = None,
        new_values: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        rls: Optional[Dict[str, str]] = None,
    ) -> None:
        """Append a row to audit_log for federation operations."""
        ctx = rls or ADMIN_RLS
        try:
            nv = json.dumps(new_values or {})
            await execute(
                """
                INSERT INTO audit_log (user_id, action, table_name, record_id, new_values)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                """,
                user_id,
                action[:100],
                "federation",
                (record_id or "")[:255] if record_id else None,
                nv,
                rls_context=ctx,
            )
        except Exception as e:
            logger.warning("Federation audit log insert failed: %s", e)

    def build_messaging_attachment_federation_token(self, attachment_id: str) -> str:
        """HMAC time-limited token for peer attachment fetch (no JWT dependency)."""
        ttl = int(getattr(settings, "FEDERATION_ATTACHMENT_TOKEN_TTL_SECONDS", 900) or 900)
        exp = int(time.time()) + max(60, ttl)
        secret = (settings.SECRET_KEY or settings.JWT_SECRET_KEY or "").encode("utf-8")
        msg = f"{attachment_id}|{exp}".encode("utf-8")
        sig = hmac.new(secret, msg, hashlib.sha256).digest()
        tok = base64.urlsafe_b64encode(msg + b"|" + sig).decode("ascii").rstrip("=")
        return tok

    def verify_messaging_attachment_federation_token(
        self, attachment_id: str, token: str
    ) -> bool:
        secret = (settings.SECRET_KEY or settings.JWT_SECRET_KEY or "").encode("utf-8")
        if not token:
            return False
        pad = "=" * (-len(token) % 4)
        try:
            raw = base64.urlsafe_b64decode(token + pad)
        except Exception:
            return False
        parts = raw.rsplit(b"|", 1)
        if len(parts) != 2:
            return False
        body, sig = parts
        expect = hmac.new(secret, body, hashlib.sha256).digest()
        if not hmac.compare_digest(expect, sig):
            return False
        try:
            aid, exp_s = body.decode("utf-8").split("|", 1)
        except ValueError:
            return False
        if aid != attachment_id:
            return False
        try:
            exp = int(exp_s)
        except ValueError:
            return False
        if int(time.time()) > exp:
            return False
        return True

    async def list_federated_users(self, rls: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        ctx = rls or ADMIN_RLS
        rows = await fetch_all(
            """
            SELECT fu.federated_user_id, fu.peer_id, fu.federated_address, fu.display_name,
                   fu.is_blocked, fu.last_seen_at, fu.presence_status, fu.presence_updated_at,
                   fp.peer_url, fp.display_name AS peer_display_name
            FROM federated_users fu
            JOIN federation_peers fp ON fp.peer_id = fu.peer_id
            ORDER BY fu.created_at DESC
            LIMIT 500
            """,
            rls_context=ctx,
        )
        return [dict(r) for r in rows]

    async def set_federated_user_blocked(
        self,
        federated_user_id: str,
        blocked: bool,
        admin_user_id: str,
        rls: Optional[Dict[str, str]] = None,
    ) -> bool:
        ctx = rls or ADMIN_RLS
        await execute(
            """
            UPDATE federated_users SET is_blocked = $2 WHERE federated_user_id = $1::uuid
            """,
            federated_user_id,
            blocked,
            rls_context=ctx,
        )
        await self.log_federation_audit(
            action="federation.federated_user.blocked" if blocked else "federation.federated_user.unblocked",
            record_id=str(federated_user_id),
            new_values={"federated_user_id": str(federated_user_id), "blocked": blocked},
            user_id=str(admin_user_id),
            rls=ctx,
        )
        return True

    async def _approve_inbound_peer_row(self, row: Dict[str, Any], ctx: Dict[str, str]) -> None:
        peer_id = row["peer_id"]
        peer_url = row["peer_url"]
        mode = row.get("connectivity_mode") or "bidirectional"
        priv = await self._get_private_key_b64(ctx)
        if not priv:
            raise ValueError("Local federation private key missing")
        local_identity = await self.get_instance_identity(ctx)
        payload = {
            "bfp_version": BFP_VERSION,
            "event_type": "peer_accept",
            "from_instance": local_identity["instance_url"],
            "from_public_key": local_identity["public_key"],
            "from_display_name": local_identity["display_name"],
            "peer_connectivity_mode": mode,
            "accepts_instance": peer_url,
        }
        body_bytes = _canonical_json_bytes(payload)
        sig = federation_crypto.sign_payload(priv, body_bytes)
        if mode == "asymmetric_listener":
            await execute(
                """
                UPDATE federation_peers
                SET status = 'active', activated_at = NOW() WHERE peer_id = $1
                """,
                peer_id,
                rls_context=ctx,
            )
            await self.enqueue_outbox(
                peer_id, "peer_accept", json.loads(body_bytes.decode("utf-8")), ctx
            )
        else:
            out_url = f"{_normalize_instance_url(peer_url)}/api/federation/peer-request"
            async with httpx.AsyncClient(timeout=float(settings.FEDERATION_HTTP_TIMEOUT)) as client:
                r = await client.post(
                    out_url,
                    content=body_bytes,
                    headers={
                        "Content-Type": "application/json",
                        "X-Bastion-Signature": sig,
                        "X-Bastion-Instance": local_identity["instance_url"],
                    },
                )
                r.raise_for_status()
            await execute(
                """
                UPDATE federation_peers
                SET status = 'active', activated_at = NOW() WHERE peer_id = $1
                """,
                peer_id,
                rls_context=ctx,
            )

    async def enqueue_outbox(
        self,
        peer_id: Any,
        event_type: str,
        payload: Dict[str, Any],
        rls: Optional[Dict[str, str]] = None,
    ) -> None:
        ctx = rls or ADMIN_RLS
        await execute(
            """
            INSERT INTO federation_outbox (peer_id, event_type, payload)
            VALUES ($1::uuid, $2, $3::jsonb)
            """,
            str(peer_id),
            event_type,
            json.dumps(payload),
            rls_context=ctx,
        )

    async def drain_outbox_for_peer_url(
        self,
        caller_instance_url: str,
        raw_body: bytes,
        signature_b64: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        ctx = ADMIN_RLS
        caller = _normalize_instance_url(caller_instance_url)
        row = await fetch_one(
            "SELECT peer_id, peer_public_key FROM federation_peers WHERE peer_url = $1 AND status = 'active'",
            caller,
            rls_context=ctx,
        )
        if not row:
            raise ValueError("Unknown or inactive peer for this instance URL")
        try:
            federation_crypto.verify_signature(
                row["peer_public_key"], raw_body, signature_b64
            )
        except BadSignatureError as e:
            raise ValueError("Invalid federation signature") from e
        try:
            body_obj = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            body_obj = {}
        since = body_obj.get("since_outbox_id")
        lim = min(int(body_obj.get("limit") or limit), 200)
        if since:
            rows = await fetch_all(
                """
                SELECT outbox_id, peer_id, event_type, payload, created_at
                FROM federation_outbox
                WHERE peer_id = $1::uuid AND picked_up_at IS NULL AND created_at > (
                    SELECT created_at FROM federation_outbox WHERE outbox_id = $2::uuid
                )
                ORDER BY created_at ASC
                LIMIT $3
                """,
                str(row["peer_id"]),
                since,
                lim,
                rls_context=ctx,
            )
        else:
            rows = await fetch_all(
                """
                SELECT outbox_id, peer_id, event_type, payload, created_at
                FROM federation_outbox
                WHERE peer_id = $1::uuid AND picked_up_at IS NULL
                ORDER BY created_at ASC
                LIMIT $2
                """,
                str(row["peer_id"]),
                lim,
                rls_context=ctx,
            )
        return [dict(r) for r in rows]

    async def ack_outbox(
        self,
        caller_instance_url: str,
        raw_body: bytes,
        signature_b64: str,
    ) -> None:
        ctx = ADMIN_RLS
        caller = _normalize_instance_url(caller_instance_url)
        row = await fetch_one(
            "SELECT peer_id, peer_public_key FROM federation_peers WHERE peer_url = $1 AND status = 'active'",
            caller,
            rls_context=ctx,
        )
        if not row:
            raise ValueError("Unknown or inactive peer for this instance URL")
        try:
            federation_crypto.verify_signature(
                row["peer_public_key"], raw_body, signature_b64
            )
        except BadSignatureError as e:
            raise ValueError("Invalid federation signature") from e
        body_obj = json.loads(raw_body.decode("utf-8"))
        ids = body_obj.get("outbox_ids") or []
        if not ids:
            return
        await execute(
            """
            UPDATE federation_outbox SET picked_up_at = NOW()
            WHERE outbox_id = ANY($1::uuid[]) AND peer_id = $2::uuid
            """,
            ids,
            str(row["peer_id"]),
            rls_context=ctx,
        )

    async def sync_pull_for_local_instance(self, rls: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        For outbound pending or asymmetric_caller active peers, pull remote outbox and apply peer_accept.
        """
        ctx = rls or ADMIN_RLS
        priv = await self._get_private_key_b64(ctx)
        if not priv:
            return {"pulled": 0, "message": "No local keypair"}
        local_identity = await self.get_instance_identity(ctx)
        if not local_identity:
            return {"pulled": 0}
        candidates = await fetch_all(
            """
            SELECT peer_id, peer_url, peer_public_key, status, connectivity_mode, is_inbound
            FROM federation_peers
            WHERE peer_url IS NOT NULL
              AND (
                (is_inbound = FALSE AND status = 'pending')
                OR (status = 'active' AND connectivity_mode = 'asymmetric_caller')
              )
            """,
            rls_context=ctx,
        )
        pulled = 0
        for c in candidates:
            peer_url = c["peer_url"]
            drain_body = _canonical_json_bytes({"limit": 50})
            sig = federation_crypto.sign_payload(priv, drain_body)
            drain_url = f"{_normalize_instance_url(peer_url)}/api/federation/outbox/drain"
            try:
                async with httpx.AsyncClient(timeout=float(settings.FEDERATION_HTTP_TIMEOUT)) as client:
                    r = await client.post(
                        drain_url,
                        content=drain_body,
                        headers={
                            "Content-Type": "application/json",
                            "X-Bastion-Signature": sig,
                            "X-Bastion-Instance": local_identity["instance_url"],
                        },
                    )
                    if r.status_code != 200:
                        continue
                    await self.update_peer_last_sync_at(str(c["peer_id"]), ctx)
                    events = r.json().get("events") or []
                    outbox_ids: List[str] = []
                    for ev in events:
                        pl = ev.get("payload") or {}
                        if isinstance(pl, str):
                            pl = json.loads(pl)
                        et = ev.get("event_type")
                        if et == "peer_accept":
                            fk = pl.get("from_public_key") or ""
                            if fk != (c.get("peer_public_key") or ""):
                                logger.warning(
                                    "peer_accept public key mismatch for peer %s", peer_url
                                )
                                continue
                            await self._apply_peer_accept_from_payload(pl, ctx)
                            pulled += 1
                            outbox_ids.append(str(ev["outbox_id"]))
                        elif et in (
                            "message",
                            "room_invite",
                            "room_invite_accept",
                            "attachment",
                            "reaction",
                            "read_receipt",
                            "presence_batch",
                        ):
                            from services.federation_message_service import (
                                federation_message_service,
                            )

                            try:
                                await federation_message_service.apply_outbox_message_payload(
                                    pl, rls=ctx
                                )
                                pulled += 1
                                outbox_ids.append(str(ev["outbox_id"]))
                            except Exception as ex:
                                logger.warning(
                                    "Federation outbox event %s apply failed: %s", et, ex
                                )
                    if outbox_ids:
                        ack_body = _canonical_json_bytes({"outbox_ids": outbox_ids})
                        ack_sig = federation_crypto.sign_payload(priv, ack_body)
                        ack_url = f"{_normalize_instance_url(peer_url)}/api/federation/outbox/ack"
                        async with httpx.AsyncClient(timeout=float(settings.FEDERATION_HTTP_TIMEOUT)) as client2:
                            await client2.post(
                                ack_url,
                                content=ack_body,
                                headers={
                                    "Content-Type": "application/json",
                                    "X-Bastion-Signature": ack_sig,
                                    "X-Bastion-Instance": local_identity["instance_url"],
                                },
                            )
            except Exception as e:
                logger.warning("Federation sync pull failed for %s: %s", peer_url, e)
        prune_info: Dict[str, Any] = {}
        try:
            prune_info = await self.prune_outbox(ctx)
        except Exception as pe:
            logger.warning("Federation outbox prune failed: %s", pe)
        return {"pulled": pulled, "prune": prune_info}

    async def _apply_peer_accept_from_payload(self, payload: Dict[str, Any], ctx: Dict[str, str]) -> None:
        """Apply peer_accept payload delivered via outbox; caller must validate public key when needed."""
        from_instance = _normalize_instance_url(payload.get("from_instance") or "")
        from_public_key = payload.get("from_public_key") or ""
        peer_mode = payload.get("peer_connectivity_mode") or "bidirectional"
        local_mode = (
            "asymmetric_caller" if peer_mode == "asymmetric_listener" else "bidirectional"
        )
        row = await fetch_one(
            """
            SELECT peer_id, peer_public_key FROM federation_peers
            WHERE peer_url = $1 AND is_inbound = FALSE AND status = 'pending'
            """,
            from_instance,
            rls_context=ctx,
        )
        if not row or (row.get("peer_public_key") or "") != from_public_key:
            return
        await execute(
            """
            UPDATE federation_peers
            SET status = 'active', activated_at = NOW(), connectivity_mode = $2
            WHERE peer_id = $1
            """,
            row["peer_id"],
            local_mode,
            rls_context=ctx,
        )

    async def sync_federation_presence_outbound(
        self, rls: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Push presence snapshots for local users in federated rooms (respects user privacy)."""
        from services.federation_message_service import federation_message_service

        ctx = rls or ADMIN_RLS
        if not getattr(settings, "FEDERATION_ENABLED", False):
            return {"skipped": True}
        peers = await fetch_all(
            "SELECT peer_id FROM federation_peers WHERE status = 'active'",
            rls_context=ctx,
        )
        batches = 0
        for pr in peers:
            pid = str(pr["peer_id"])
            rows = await fetch_all(
                """
                SELECT DISTINCT u.user_id::text AS uid, u.display_name, u.username,
                       up.status::text AS st, up.last_seen_at
                FROM room_participants rp
                JOIN chat_rooms r ON r.room_id = rp.room_id
                JOIN users u ON u.user_id = rp.user_id
                JOIN user_presence up ON up.user_id = u.user_id
                WHERE r.room_type = 'federated'
                  AND (r.federation_metadata->>'peer_id')::uuid = $1::uuid
                  AND COALESCE(u.federation_share_presence, true) = true
                """,
                pid,
                rls_context=ctx,
            )
            entries: List[Dict[str, Any]] = []
            for r2 in rows:
                uid = r2.get("uid")
                if not uid:
                    continue
                addr = await federation_message_service.federated_address_for_local_user(
                    str(uid), ctx
                )
                ls = r2.get("last_seen_at")
                ls_out = ls.isoformat() if hasattr(ls, "isoformat") else str(ls or "")
                entries.append(
                    {
                        "user_address": addr,
                        "display_name": r2.get("display_name") or r2.get("username"),
                        "status": str(r2.get("st") or "offline"),
                        "last_seen_at": ls_out,
                    }
                )
            if entries:
                await federation_message_service.deliver_outbound_presence_batch(
                    pid, entries, ctx
                )
                batches += 1
        return {"presence_batches": batches}


federation_service = FederationService()

normalize_instance_url = _normalize_instance_url
