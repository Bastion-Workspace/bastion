"""
Federation Phase 2: federated rooms, cross-instance message delivery, room invites.
"""

from __future__ import annotations

import base64
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import httpx
from nacl.exceptions import BadSignatureError

from config import settings
from services.database_manager.database_helpers import execute, fetch_one
from services.federation_service import federation_service
from services.messaging.encryption_service import encryption_service
from services.messaging.messaging_attachment_service import messaging_attachment_service
from utils import federation_crypto
from utils.websocket_manager import get_websocket_manager

logger = logging.getLogger(__name__)

ADMIN_RLS = {"user_id": "", "user_role": "admin"}
BFP_VERSION = "1"


class PeerFederationInactiveError(Exception):
    """Peer exists but federation is suspended or revoked (reject signed ingest)."""

    def __init__(self, status: str):
        self.status = status or "unknown"
        super().__init__(f"Peer federation not active: {self.status}")


def _normalize_instance_url(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if not u:
        return ""
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    from urllib.parse import urlparse

    parsed = urlparse(u)
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def _canonical_json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _site_host_for_address() -> str:
    from urllib.parse import urlparse

    p = urlparse(settings.SITE_URL or "")
    return p.netloc or "localhost"


def _parse_federated_address(addr: str) -> Tuple[str, str]:
    """Return (username, host_part) from user@host."""
    s = (addr or "").strip()
    if "@" not in s:
        return s, ""
    user, host = s.rsplit("@", 1)
    return user.strip(), host.strip()


class FederationMessageService:
    async def get_or_create_federated_user(
        self,
        peer_id: str,
        remote_user_id: str,
        federated_address: str,
        display_name: Optional[str] = None,
        avatar_url: Optional[str] = None,
        rls: Optional[Dict[str, str]] = None,
    ) -> str:
        ctx = rls or ADMIN_RLS
        row = await fetch_one(
            """
            INSERT INTO federated_users (
                peer_id, remote_user_id, federated_address, display_name, avatar_url, last_seen_at
            ) VALUES ($1::uuid, $2, $3, $4, $5, NOW())
            ON CONFLICT (peer_id, remote_user_id) DO UPDATE SET
                federated_address = EXCLUDED.federated_address,
                display_name = COALESCE(EXCLUDED.display_name, federated_users.display_name),
                avatar_url = COALESCE(EXCLUDED.avatar_url, federated_users.avatar_url),
                last_seen_at = NOW()
            RETURNING federated_user_id
            """,
            peer_id,
            remote_user_id,
            federated_address,
            display_name,
            avatar_url,
            rls_context=ctx,
        )
        return str(row["federated_user_id"]) if row else ""

    async def federated_address_for_local_user(self, user_id: str, rls: Optional[Dict[str, str]] = None) -> str:
        ctx = rls or ADMIN_RLS
        row = await fetch_one(
            "SELECT username FROM users WHERE user_id = $1",
            user_id,
            rls_context=ctx,
        )
        user = ((row or {}).get("username") or "user").strip()
        return f"{user}@{_site_host_for_address()}"

    async def create_federated_room(
        self,
        creator_user_id: str,
        peer_id: str,
        room_name: Optional[str],
        rls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        ctx = rls or ADMIN_RLS
        # federation_peers is admin-RLS-only; peer lookup always uses admin context
        peer = await fetch_one(
            """
            SELECT peer_id, peer_url, status FROM federation_peers
            WHERE peer_id = $1::uuid AND status = 'active'
            """,
            peer_id,
            rls_context=ADMIN_RLS,
        )
        if not peer:
            raise ValueError("Peer not found or not active")
        room_id = str(uuid.uuid4())
        meta = {"peer_id": str(peer["peer_id"]), "remote_room_id": None}
        disp = (room_name or "").strip() or "Federated chat"
        await execute(
            """
            INSERT INTO chat_rooms (room_id, room_name, room_type, created_by, federation_metadata)
            VALUES ($1::uuid, $2, 'federated', $3, $4::jsonb)
            """,
            room_id,
            disp,
            creator_user_id,
            json.dumps(meta),
            rls_context=ctx,
        )
        await execute(
            """
            INSERT INTO room_participants (room_id, user_id) VALUES ($1::uuid, $2)
            ON CONFLICT DO NOTHING
            """,
            room_id,
            creator_user_id,
            rls_context=ctx,
        )
        if encryption_service.is_encryption_enabled():
            room_key = encryption_service.derive_room_key(room_id)
            encrypted_key = encryption_service.encrypt_room_key(room_key)
            if encrypted_key:
                await execute(
                    """
                    INSERT INTO room_encryption_keys (room_id, encrypted_key)
                    VALUES ($1::uuid, $2)
                    ON CONFLICT DO NOTHING
                    """,
                    room_id,
                    encrypted_key,
                    rls_context=ctx,
                )
        parts = await fetch_one(
            """
            SELECT COALESCE(
                json_agg(json_build_object(
                    'user_id', u.user_id, 'username', u.username,
                    'display_name', u.display_name, 'avatar_url', u.avatar_url
                )), '[]'::json
            ) AS participants
            FROM room_participants rp
            JOIN users u ON rp.user_id = u.user_id
            WHERE rp.room_id = $1::uuid
            """,
            room_id,
            rls_context=ctx,
        )
        plist = parts["participants"] if parts else []
        if isinstance(plist, str):
            plist = json.loads(plist)
        return {
            "room_id": room_id,
            "room_name": disp,
            "room_type": "federated",
            "created_by": creator_user_id,
            "participant_ids": [creator_user_id],
            "participants": plist,
            "display_name": disp,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "federation_metadata": meta,
        }

    async def update_room_remote_id(
        self, local_room_id: str, remote_room_id: str, rls: Optional[Dict[str, str]] = None
    ) -> None:
        ctx = rls or ADMIN_RLS
        patch = json.dumps({"remote_room_id": remote_room_id})
        await execute(
            """
            UPDATE chat_rooms
            SET federation_metadata = COALESCE(federation_metadata, '{}'::jsonb) || $2::jsonb
            WHERE room_id = $1::uuid
            """,
            local_room_id,
            patch,
            rls_context=ctx,
        )

    async def _get_peer_row_for_room(self, room_id: str, ctx: Dict[str, str]) -> Optional[Dict[str, Any]]:
        row = await fetch_one(
            """
            SELECT r.room_id, r.room_type, r.federation_metadata,
                   p.peer_id, p.peer_url, p.peer_public_key, p.connectivity_mode, p.status
            FROM chat_rooms r
            JOIN federation_peers p ON p.peer_id = (r.federation_metadata->>'peer_id')::uuid
            WHERE r.room_id = $1::uuid
            """,
            room_id,
            rls_context=ctx,
        )
        return dict(row) if row else None

    async def deliver_outbound_message(
        self, room_id: str, message_dict: Dict[str, Any], rls: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        ctx = rls or ADMIN_RLS
        if not settings.FEDERATION_ENABLED:
            return {"ok": False, "reason": "federation_disabled"}
        row = await self._get_peer_row_for_room(room_id, ctx)
        if not row or (row.get("room_type") or "") != "federated":
            return {"ok": False, "reason": "not_federated_room"}
        meta = row.get("federation_metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        remote_room = meta.get("remote_room_id")
        if not remote_room:
            logger.warning("Federation deliver skipped: remote_room_id not set for room %s", room_id)
            return {"ok": False, "reason": "no_remote_room_id"}
        if (row.get("status") or "") != "active":
            return {"ok": False, "reason": "peer_suspended"}

        priv = await federation_service.get_private_key_b64_for_signing(ADMIN_RLS)
        if not priv:
            return {"ok": False, "reason": "no_private_key"}
        ident = await federation_service.get_instance_identity(ADMIN_RLS)
        if not ident:
            return {"ok": False, "reason": "no_identity"}

        sender_id = message_dict.get("sender_id") or ""
        from_user = await self.federated_address_for_local_user(str(sender_id), ctx)
        from_dn = message_dict.get("display_name") or message_dict.get("username") or from_user

        envelope: Dict[str, Any] = {
            "bfp_version": BFP_VERSION,
            "event_type": "federation_message",
            "from_instance": ident["instance_url"],
            "from_user": from_user,
            "from_user_display_name": from_dn,
            "room_id": str(remote_room),
            "message_id": str(message_dict.get("message_id") or ""),
            "sent_at": message_dict.get("created_at") or datetime.now(timezone.utc).isoformat(),
            "content": message_dict.get("content") or "",
            "message_type": message_dict.get("message_type") or "text",
        }
        wire = _canonical_json_bytes(envelope)
        sig = federation_crypto.sign_payload(priv, wire)
        peer_url = _normalize_instance_url(row["peer_url"] or "")
        mode = row.get("connectivity_mode") or "bidirectional"

        out_payload = {
            "bfp_wire_b64": base64.b64encode(wire).decode("ascii"),
            "signature": sig,
            "x_bastion_instance": ident["instance_url"],
        }

        if mode == "asymmetric_listener":
            await federation_service.enqueue_outbox(str(row["peer_id"]), "message", out_payload, ADMIN_RLS)
            return {"ok": True, "channel": "outbox"}

        url = f"{peer_url}/api/federation/message"
        try:
            async with httpx.AsyncClient(timeout=float(settings.FEDERATION_HTTP_TIMEOUT)) as client:
                r = await client.post(
                    url,
                    content=wire,
                    headers={
                        "Content-Type": "application/json",
                        "X-Bastion-Signature": sig,
                        "X-Bastion-Instance": ident["instance_url"],
                    },
                )
                if r.status_code >= 400:
                    logger.warning("Federation message POST failed %s: %s", r.status_code, r.text[:500])
                    return {"ok": False, "reason": "http_error", "status": r.status_code}
        except Exception as e:
            logger.warning("Federation message POST error: %s", e)
            return {"ok": False, "reason": str(e)}
        return {"ok": True, "channel": "http"}

    def _verify_and_parse_envelope(
        self, raw_body: bytes, signature_b64: Optional[str], from_instance_header: Optional[str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not signature_b64:
            raise ValueError("Missing X-Bastion-Signature header")
        payload = json.loads(raw_body.decode("utf-8"))
        from_inst = _normalize_instance_url(
            str(payload.get("from_instance") or from_instance_header or "")
        )
        if not from_inst:
            raise ValueError("Missing from_instance")
        return payload, {"from_instance": from_inst}

    async def _get_peer_by_url_any(self, from_instance: str, ctx: Dict[str, str]) -> Dict[str, Any]:
        row = await fetch_one(
            """
            SELECT peer_id, peer_url, peer_public_key, status, connectivity_mode
            FROM federation_peers
            WHERE peer_url = $1
            """,
            from_instance,
            rls_context=ADMIN_RLS,
        )
        if not row:
            raise ValueError("Unknown or inactive peer instance")
        return dict(row)

    async def _verify_peer_signature_and_require_active(
        self,
        from_instance: str,
        raw_body: bytes,
        signature_b64: Optional[str],
        ctx: Dict[str, str],
    ) -> Dict[str, Any]:
        peer = await self._get_peer_by_url_any(from_instance, ctx)
        try:
            federation_crypto.verify_signature(
                peer["peer_public_key"], raw_body, signature_b64 or ""
            )
        except BadSignatureError as e:
            raise ValueError("Invalid federation signature") from e
        if (peer.get("status") or "") != "active":
            raise PeerFederationInactiveError(str(peer.get("status") or ""))
        return peer

    async def ingest_inbound_message(
        self,
        raw_body: bytes,
        signature_b64: Optional[str],
        from_instance_header: Optional[str],
        rls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        ctx = rls or ADMIN_RLS
        payload, meta = self._verify_and_parse_envelope(raw_body, signature_b64, from_instance_header)
        peer = await self._verify_peer_signature_and_require_active(
            meta["from_instance"], raw_body, signature_b64, ctx
        )

        if payload.get("event_type") != "federation_message":
            raise ValueError("Invalid event_type for message endpoint")

        room_id = str(payload.get("room_id") or "")
        if not room_id:
            raise ValueError("room_id required")
        room_chk = await fetch_one(
            "SELECT room_id, room_type FROM chat_rooms WHERE room_id = $1::uuid",
            room_id,
            rls_context=ctx,
        )
        if not room_chk or (room_chk.get("room_type") or "") != "federated":
            raise ValueError("Unknown federated room")

        remote_msg = str(payload.get("message_id") or "")
        dup = await fetch_one(
            """
            SELECT 1 FROM chat_messages
            WHERE room_id = $1::uuid
              AND deleted_at IS NULL
              AND COALESCE(metadata->>'federation_remote_message_id','') = $2
              AND $2 != ''
            LIMIT 1
            """,
            room_id,
            remote_msg,
            rls_context=ctx,
        )
        if dup:
            return {"ok": True, "duplicate": True}

        from_addr = str(payload.get("from_user") or "")
        if from_addr and await self._federated_user_is_blocked(str(peer["peer_id"]), from_addr, ctx):
            return {"ok": True, "ignored": "blocked"}

        fu = await self.get_or_create_federated_user(
            str(peer["peer_id"]),
            from_addr or "unknown",
            from_addr,
            display_name=payload.get("from_user_display_name"),
            avatar_url=None,
            rls=ctx,
        )
        blk = await fetch_one(
            "SELECT is_blocked FROM federated_users WHERE federated_user_id = $1::uuid",
            fu,
            rls_context=ctx,
        )
        if blk and blk.get("is_blocked"):
            return {"ok": True, "ignored": "blocked"}

        content = str(payload.get("content") or "")
        enc = encryption_service.encrypt_message(content)
        message_id = str(uuid.uuid4())
        meta_json = json.dumps({"federation_remote_message_id": remote_msg})

        mt = str(payload.get("message_type") or "text")
        if mt not in ("text", "ai_share", "system"):
            mt = "text"
        await execute(
            """
            INSERT INTO chat_messages (
                message_id, room_id, sender_id, federated_sender_id,
                message_content, message_type, metadata
            ) VALUES ($1::uuid, $2::uuid, NULL, $3::uuid, $4, $5::message_type_enum, $6::jsonb)
            """,
            message_id,
            room_id,
            fu,
            enc,
            mt,
            meta_json,
            rls_context=ctx,
        )
        await execute(
            "UPDATE chat_rooms SET last_message_at = NOW() WHERE room_id = $1::uuid",
            room_id,
            rls_context=ctx,
        )

        fu_row = await fetch_one(
            """
            SELECT federated_address, display_name, avatar_url
            FROM federated_users WHERE federated_user_id = $1::uuid
            """,
            fu,
            rls_context=ctx,
        )
        fd = dict(fu_row) if fu_row else {}
        created_iso = datetime.now(timezone.utc).isoformat()
        ws_message = {
            "message_id": message_id,
            "room_id": room_id,
            "sender_id": None,
            "federated_sender_id": fu,
            "federated_address": fd.get("federated_address"),
            "content": content,
            "message_type": mt,
            "metadata": json.loads(meta_json) if meta_json else {},
            "reply_to_message_id": None,
            "created_at": created_iso,
            "username": None,
            "display_name": fd.get("display_name") or fd.get("federated_address"),
            "avatar_url": fd.get("avatar_url"),
            "is_federated": True,
        }

        ws = get_websocket_manager()
        await ws.broadcast_to_room(
            room_id,
            {"type": "new_message", "message": ws_message},
            exclude_user_id=None,
        )
        await federation_service.log_federation_audit(
            action="federation.inbound.message",
            record_id=str(message_id),
            new_values={
                "room_id": room_id,
                "peer_id": str(peer["peer_id"]),
                "from_user": str(payload.get("from_user") or ""),
            },
            rls=ctx,
        )
        return {"ok": True}

    async def send_room_invite(
        self,
        peer_id: str,
        local_room_id: str,
        target_local_username: str,
        inviter_user_id: str,
        room_name: Optional[str],
        rls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        ctx = rls or ADMIN_RLS
        priv = await federation_service.get_private_key_b64_for_signing(ADMIN_RLS)
        if not priv:
            raise ValueError("Federation keypair not available")
        ident = await federation_service.get_instance_identity(ADMIN_RLS)
        if not ident:
            raise ValueError("Federation identity not available")
        peer = await fetch_one(
            """
            SELECT peer_id, peer_url, connectivity_mode, status
            FROM federation_peers WHERE peer_id = $1::uuid
            """,
            peer_id,
            rls_context=ADMIN_RLS,
        )
        if not peer or (peer.get("status") or "") != "active":
            raise ValueError("Peer not active")
        from_user = await self.federated_address_for_local_user(inviter_user_id, ctx)
        inv_row = await fetch_one(
            "SELECT display_name, username FROM users WHERE user_id = $1",
            inviter_user_id,
            rls_context=ctx,
        )
        from_dn = (inv_row or {}).get("display_name") or (inv_row or {}).get("username") or from_user
        envelope: Dict[str, Any] = {
            "bfp_version": BFP_VERSION,
            "event_type": "room_invite",
            "from_instance": ident["instance_url"],
            "from_user": from_user,
            "from_user_display_name": from_dn,
            "inviter_room_id": local_room_id,
            "room_name": room_name or "",
            "target_local_username": target_local_username.strip(),
            "inviter_user_id": str(inviter_user_id),
        }
        wire = _canonical_json_bytes(envelope)
        sig = federation_crypto.sign_payload(priv, wire)
        peer_url = _normalize_instance_url(peer["peer_url"] or "")
        mode = peer.get("connectivity_mode") or "bidirectional"
        out_payload = {
            "bfp_wire_b64": base64.b64encode(wire).decode("ascii"),
            "signature": sig,
            "x_bastion_instance": ident["instance_url"],
        }

        if mode == "asymmetric_listener":
            await federation_service.enqueue_outbox(str(peer["peer_id"]), "room_invite", out_payload, ADMIN_RLS)
            return {"ok": True, "channel": "outbox", "local_room_id": None}

        url = f"{peer_url}/api/federation/room-invite"
        async with httpx.AsyncClient(timeout=float(settings.FEDERATION_HTTP_TIMEOUT)) as client:
            r = await client.post(
                url,
                content=wire,
                headers={
                    "Content-Type": "application/json",
                    "X-Bastion-Signature": sig,
                    "X-Bastion-Instance": ident["instance_url"],
                },
            )
            if r.status_code >= 400:
                raise ValueError(f"room-invite failed: HTTP {r.status_code} {r.text[:300]}")
            body = r.json()
            remote_rid = body.get("local_room_id")
            if remote_rid:
                await self.update_room_remote_id(local_room_id, str(remote_rid), ctx)
            return {"ok": True, "channel": "http", "remote_room_id": remote_rid}

    async def handle_room_invite(
        self,
        raw_body: bytes,
        signature_b64: Optional[str],
        from_instance_header: Optional[str],
        rls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        ctx = rls or ADMIN_RLS
        payload, meta = self._verify_and_parse_envelope(raw_body, signature_b64, from_instance_header)
        peer = await self._verify_peer_signature_and_require_active(
            meta["from_instance"], raw_body, signature_b64, ctx
        )
        if payload.get("event_type") != "room_invite":
            raise ValueError("Invalid event_type")

        target_username = (payload.get("target_local_username") or "").strip()
        if not target_username:
            raise ValueError("target_local_username required")
        user_row = await fetch_one(
            "SELECT user_id, username, display_name FROM users WHERE lower(username) = lower($1)",
            target_username,
            rls_context=ctx,
        )
        if not user_row:
            raise ValueError(f"No local user with username: {target_username}")
        uid = str(user_row["user_id"])

        await self.get_or_create_federated_user(
            str(peer["peer_id"]),
            str(payload.get("from_user") or payload.get("inviter_user_id") or "inviter"),
            str(payload.get("from_user") or ""),
            display_name=payload.get("from_user_display_name"),
            rls=ctx,
        )

        inviter_room = str(payload.get("inviter_room_id") or "")
        room_name = (payload.get("room_name") or "").strip() or "Federated chat"
        new_room_id = str(uuid.uuid4())
        meta = {"peer_id": str(peer["peer_id"]), "remote_room_id": inviter_room}
        await execute(
            """
            INSERT INTO chat_rooms (room_id, room_name, room_type, created_by, federation_metadata)
            VALUES ($1::uuid, $2, 'federated', $3, $4::jsonb)
            """,
            new_room_id,
            room_name,
            uid,
            json.dumps(meta),
            rls_context=ctx,
        )
        await execute(
            "INSERT INTO room_participants (room_id, user_id) VALUES ($1::uuid, $2) ON CONFLICT DO NOTHING",
            new_room_id,
            uid,
            rls_context=ctx,
        )
        if encryption_service.is_encryption_enabled():
            room_key = encryption_service.derive_room_key(new_room_id)
            encrypted_key = encryption_service.encrypt_room_key(room_key)
            if encrypted_key:
                await execute(
                    """
                    INSERT INTO room_encryption_keys (room_id, encrypted_key)
                    VALUES ($1::uuid, $2)
                    ON CONFLICT DO NOTHING
                    """,
                    new_room_id,
                    encrypted_key,
                    rls_context=ctx,
                )

        room_payload = await self._room_ws_payload(new_room_id, uid, ctx)
        ws = get_websocket_manager()
        await ws.broadcast_to_users([uid], {"type": "new_room", "room": room_payload})

        try:
            await self._push_room_invite_accept(
                meta["from_instance"], inviter_room, new_room_id, ctx
            )
        except Exception as e:
            logger.warning("room_invite_accept notify failed (metadata may stay incomplete): %s", e)

        return {"ok": True, "local_room_id": new_room_id}

    async def _push_room_invite_accept(
        self,
        inviter_instance_url: str,
        inviter_room_id: str,
        acceptor_room_id: str,
        ctx: Dict[str, str],
    ) -> None:
        """Tell the inviter instance to record remote_room_id on their federated room."""
        inviter_base = _normalize_instance_url(inviter_instance_url)
        peer = await fetch_one(
            """
            SELECT peer_id, peer_url, peer_public_key, connectivity_mode, status
            FROM federation_peers
            WHERE peer_url = $1 AND status = 'active'
            """,
            inviter_base,
            rls_context=ADMIN_RLS,
        )
        if not peer:
            logger.warning("No active peer row for inviter %s; skip invite-accept push", inviter_base)
            return
        priv = await federation_service.get_private_key_b64_for_signing(ADMIN_RLS)
        if not priv:
            return
        ident = await federation_service.get_instance_identity(ADMIN_RLS)
        if not ident:
            return
        envelope: Dict[str, Any] = {
            "bfp_version": BFP_VERSION,
            "event_type": "room_invite_accept",
            "from_instance": ident["instance_url"],
            "inviter_room_id": inviter_room_id,
            "acceptor_room_id": acceptor_room_id,
        }
        wire = _canonical_json_bytes(envelope)
        sig = federation_crypto.sign_payload(priv, wire)
        mode = peer.get("connectivity_mode") or "bidirectional"
        out_payload = {
            "bfp_wire_b64": base64.b64encode(wire).decode("ascii"),
            "signature": sig,
            "x_bastion_instance": ident["instance_url"],
        }
        if mode == "asymmetric_listener":
            await federation_service.enqueue_outbox(str(peer["peer_id"]), "room_invite_accept", out_payload, ADMIN_RLS)
            return
        url = f"{inviter_base}/api/federation/room-invite-accept"
        async with httpx.AsyncClient(timeout=float(settings.FEDERATION_HTTP_TIMEOUT)) as client:
            r = await client.post(
                url,
                content=wire,
                headers={
                    "Content-Type": "application/json",
                    "X-Bastion-Signature": sig,
                    "X-Bastion-Instance": ident["instance_url"],
                },
            )
            if r.status_code >= 400:
                logger.warning("room_invite_accept POST failed: %s %s", r.status_code, r.text[:200])

    async def handle_room_invite_accept(
        self,
        raw_body: bytes,
        signature_b64: Optional[str],
        from_instance_header: Optional[str],
        rls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        ctx = rls or ADMIN_RLS
        payload, meta = self._verify_and_parse_envelope(raw_body, signature_b64, from_instance_header)
        peer = await self._verify_peer_signature_and_require_active(
            meta["from_instance"], raw_body, signature_b64, ctx
        )
        if payload.get("event_type") != "room_invite_accept":
            raise ValueError("Invalid event_type")
        inviter_room = str(payload.get("inviter_room_id") or "")
        acceptor_room = str(payload.get("acceptor_room_id") or "")
        if not inviter_room or not acceptor_room:
            raise ValueError("inviter_room_id and acceptor_room_id required")
        exists = await fetch_one(
            "SELECT room_id FROM chat_rooms WHERE room_id = $1::uuid AND room_type = 'federated'",
            inviter_room,
            rls_context=ctx,
        )
        if not exists:
            raise ValueError("inviter_room not found or not federated")
        await self.update_room_remote_id(inviter_room, acceptor_room, ctx)
        return {"ok": True}

    async def _room_ws_payload(self, room_id: str, for_user_id: str, ctx: Dict[str, str]) -> Dict[str, Any]:
        """Shape compatible with frontend new_room handler."""
        row = await fetch_one(
            """
            SELECT room_id, room_name, room_type, created_by, created_at, last_message_at, federation_metadata
            FROM chat_rooms WHERE room_id = $1::uuid
            """,
            room_id,
            rls_context=ctx,
        )
        if not row:
            return {"room_id": room_id}
        r = dict(row)
        parts = await fetch_one(
            """
            SELECT COALESCE(
                json_agg(json_build_object(
                    'user_id', u.user_id, 'username', u.username,
                    'display_name', u.display_name, 'avatar_url', u.avatar_url
                )), '[]'::json
            ) AS participants
            FROM room_participants rp
            JOIN users u ON rp.user_id = u.user_id
            WHERE rp.room_id = $1::uuid
            """,
            room_id,
            rls_context=ctx,
        )
        plist = parts["participants"] if parts else []
        if isinstance(plist, str):
            plist = json.loads(plist)
        disp = r.get("room_name") or "Federated chat"
        if r.get("room_type") == "federated":
            disp = r.get("room_name") or disp
        return {
            "room_id": str(r["room_id"]),
            "room_name": r.get("room_name"),
            "room_type": r.get("room_type"),
            "created_by": r.get("created_by"),
            "participant_ids": [p["user_id"] for p in plist],
            "participants": plist,
            "display_name": disp,
            "created_at": r["created_at"].isoformat() if hasattr(r.get("created_at"), "isoformat") else str(r.get("created_at")),
            "last_message_at": r["last_message_at"].isoformat()
            if hasattr(r.get("last_message_at"), "isoformat")
            else str(r.get("last_message_at")),
            "message_count": 0,
            "unread_count": 0,
            "notification_settings": {},
            "federation_metadata": r.get("federation_metadata"),
        }

    async def apply_outbox_message_payload(self, payload: Dict[str, Any], rls: Optional[Dict[str, str]] = None) -> None:
        """Apply a message or room_invite stored in outbox (already authenticated via drain)."""
        ctx = rls or ADMIN_RLS
        b64w = payload.get("bfp_wire_b64") or ""
        if not b64w:
            raise ValueError("outbox payload missing bfp_wire_b64")
        sig = payload.get("signature") or ""
        inst = payload.get("x_bastion_instance")
        raw = base64.b64decode(b64w)
        pl = json.loads(raw.decode("utf-8"))
        et = pl.get("event_type")
        if et == "federation_message":
            await self.ingest_inbound_message(raw, sig, inst, rls=ctx)
        elif et == "room_invite":
            await self.handle_room_invite(raw, sig, inst, rls=ctx)
        elif et == "room_invite_accept":
            await self.handle_room_invite_accept(raw, sig, inst, rls=ctx)
        elif et == "attachment":
            await self.ingest_inbound_attachment_event(raw, sig, inst, rls=ctx)
        elif et == "reaction":
            await self.ingest_inbound_reaction(raw, sig, inst, rls=ctx)
        elif et == "read_receipt":
            await self.ingest_inbound_read_receipt(raw, sig, inst, rls=ctx)
        elif et == "presence_batch":
            await self.ingest_inbound_presence_batch(raw, sig, inst, rls=ctx)
        else:
            logger.warning("Unknown outbox BFP event_type: %s", et)

    async def _federated_user_is_blocked(
        self, peer_id: str, federated_address: str, ctx: Dict[str, str]
    ) -> bool:
        row = await fetch_one(
            """
            SELECT is_blocked FROM federated_users
            WHERE peer_id = $1::uuid AND federated_address = $2
            """,
            peer_id,
            federated_address,
            rls_context=ctx,
        )
        return bool(row and row.get("is_blocked"))

    async def _resolve_local_message_id_for_wire(
        self, room_id: str, message_wire_id: str, ctx: Dict[str, str]
    ) -> Optional[str]:
        row = await fetch_one(
            """
            SELECT message_id::text AS mid
            FROM chat_messages
            WHERE room_id = $1::uuid AND deleted_at IS NULL
              AND (
                message_id::text = $2
                OR COALESCE(metadata->>'federation_remote_message_id','') = $2
              )
            LIMIT 1
            """,
            room_id,
            message_wire_id,
            rls_context=ctx,
        )
        return str(row["mid"]) if row else None

    async def _sign_deliver_bfp(
        self,
        room_id: str,
        envelope: Dict[str, Any],
        http_path: str,
        outbox_outer_type: str,
        rls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        ctx = rls or ADMIN_RLS
        row = await self._get_peer_row_for_room(room_id, ctx)
        if not row or (row.get("room_type") or "") != "federated":
            return {"ok": False, "reason": "not_federated_room"}
        if (row.get("status") or "") != "active":
            return {"ok": False, "reason": "peer_suspended"}
        meta = row.get("federation_metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        remote_room = meta.get("remote_room_id")
        if not remote_room:
            return {"ok": False, "reason": "no_remote_room_id"}
        priv = await federation_service.get_private_key_b64_for_signing(ADMIN_RLS)
        if not priv:
            return {"ok": False, "reason": "no_private_key"}
        ident = await federation_service.get_instance_identity(ADMIN_RLS)
        if not ident:
            return {"ok": False, "reason": "no_identity"}
        envelope.setdefault("bfp_version", BFP_VERSION)
        envelope.setdefault("from_instance", ident["instance_url"])
        wire = _canonical_json_bytes(envelope)
        sig = federation_crypto.sign_payload(priv, wire)
        peer_url = _normalize_instance_url(row["peer_url"] or "")
        mode = row.get("connectivity_mode") or "bidirectional"
        out_payload = {
            "bfp_wire_b64": base64.b64encode(wire).decode("ascii"),
            "signature": sig,
            "x_bastion_instance": ident["instance_url"],
        }
        if mode == "asymmetric_listener":
            await federation_service.enqueue_outbox(
                str(row["peer_id"]), outbox_outer_type, out_payload, ADMIN_RLS
            )
            return {"ok": True, "channel": "outbox"}
        url = f"{peer_url}{http_path}"
        try:
            async with httpx.AsyncClient(timeout=float(settings.FEDERATION_HTTP_TIMEOUT)) as client:
                r = await client.post(
                    url,
                    content=wire,
                    headers={
                        "Content-Type": "application/json",
                        "X-Bastion-Signature": sig,
                        "X-Bastion-Instance": ident["instance_url"],
                    },
                )
                if r.status_code >= 400:
                    logger.warning("Federation POST %s failed %s", http_path, r.status_code)
                    return {"ok": False, "reason": "http_error", "status": r.status_code}
        except Exception as e:
            logger.warning("Federation POST %s error: %s", http_path, e)
            return {"ok": False, "reason": str(e)}
        return {"ok": True, "channel": "http"}

    async def deliver_outbound_attachment(
        self,
        room_id: str,
        attachment_row: Dict[str, Any],
        message_wire_id: str,
        uploader_user_id: str,
        rls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        ctx = rls or ADMIN_RLS
        row = await self._get_peer_row_for_room(room_id, ctx)
        if not row:
            return {"ok": False, "reason": "not_federated_room"}
        meta = row.get("federation_metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        remote_room = meta.get("remote_room_id")
        if not remote_room:
            return {"ok": False, "reason": "no_remote_room_id"}
        aid = str(attachment_row.get("attachment_id") or "")
        ident = await federation_service.get_instance_identity(ADMIN_RLS)
        if not ident:
            return {"ok": False, "reason": "no_identity"}
        tok = federation_service.build_messaging_attachment_federation_token(aid)
        from urllib.parse import quote

        fetch_url = (
            f"{ident['instance_url'].rstrip('/')}/api/federation/attachments/{aid}/file"
            f"?token={quote(tok, safe='')}"
        )
        from_user = await self.federated_address_for_local_user(str(uploader_user_id), ctx)
        inv = await fetch_one(
            "SELECT display_name, username FROM users WHERE user_id = $1",
            uploader_user_id,
            rls_context=ctx,
        )
        from_dn = (inv or {}).get("display_name") or (inv or {}).get("username") or from_user
        envelope = {
            "event_type": "attachment",
            "from_instance": ident["instance_url"],
            "from_user": from_user,
            "from_user_display_name": from_dn,
            "room_id": str(remote_room),
            "message_wire_id": str(message_wire_id),
            "attachment_id": aid,
            "filename": str(attachment_row.get("filename") or "file"),
            "mime_type": str(attachment_row.get("mime_type") or "application/octet-stream"),
            "file_size": int(attachment_row.get("file_size") or 0),
            "fetch_url": fetch_url,
        }
        return await self._sign_deliver_bfp(
            room_id, envelope, "/api/federation/attachment-event", "attachment", ctx
        )

    async def ingest_inbound_attachment_event(
        self,
        raw_body: bytes,
        signature_b64: Optional[str],
        from_instance_header: Optional[str],
        rls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        ctx = rls or ADMIN_RLS
        payload, meta = self._verify_and_parse_envelope(raw_body, signature_b64, from_instance_header)
        peer = await self._verify_peer_signature_and_require_active(
            meta["from_instance"], raw_body, signature_b64, ctx
        )
        if payload.get("event_type") != "attachment":
            raise ValueError("Invalid event_type")
        from_addr = str(payload.get("from_user") or "")
        if from_addr and await self._federated_user_is_blocked(
            str(peer["peer_id"]), from_addr, ctx
        ):
            return {"ok": True, "ignored": "blocked"}
        room_id = str(payload.get("room_id") or "")
        mw = str(payload.get("message_wire_id") or "")
        if not room_id or not mw:
            raise ValueError("room_id and message_wire_id required")
        local_mid = await self._resolve_local_message_id_for_wire(room_id, mw, ctx)
        if not local_mid:
            raise ValueError("Message not found for attachment")
        fetch_url = str(payload.get("fetch_url") or "")
        if not fetch_url:
            raise ValueError("fetch_url required")
        await messaging_attachment_service.initialize()
        await messaging_attachment_service.import_from_peer_url(
            room_id=room_id,
            message_id=local_mid,
            filename=str(payload.get("filename") or "attachment"),
            mime_type=str(payload.get("mime_type") or "application/octet-stream"),
            source_url=fetch_url,
        )
        await federation_service.log_federation_audit(
            action="federation.inbound.attachment",
            record_id=room_id,
            new_values={"message_id": local_mid, "peer_id": str(peer["peer_id"])},
            rls=ctx,
        )
        ws = get_websocket_manager()
        atts = await messaging_attachment_service.list_attachments_for_message_admin(local_mid)
        await ws.broadcast_to_room(
            room_id,
            {"type": "attachment_added", "message_id": local_mid, "attachments": atts},
            exclude_user_id=None,
        )
        return {"ok": True}

    async def deliver_outbound_reaction(
        self,
        room_id: str,
        action: str,
        message_wire_id: str,
        emoji: str,
        local_user_id: str,
        reaction_id: Optional[str],
        rls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        ctx = rls or ADMIN_RLS
        from_user = await self.federated_address_for_local_user(str(local_user_id), ctx)
        inv = await fetch_one(
            "SELECT display_name, username FROM users WHERE user_id = $1",
            local_user_id,
            rls_context=ctx,
        )
        from_dn = (inv or {}).get("display_name") or (inv or {}).get("username") or from_user
        envelope = {
            "event_type": "reaction",
            "action": action,
            "room_id": "",
            "message_wire_id": str(message_wire_id),
            "emoji": emoji,
            "from_user": from_user,
            "from_user_display_name": from_dn,
            "reaction_id": reaction_id or "",
        }
        row = await self._get_peer_row_for_room(room_id, ctx)
        if not row:
            return {"ok": False, "reason": "not_federated_room"}
        meta = row.get("federation_metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        remote_room = meta.get("remote_room_id")
        if not remote_room:
            return {"ok": False, "reason": "no_remote_room_id"}
        envelope["room_id"] = str(remote_room)
        return await self._sign_deliver_bfp(
            room_id, envelope, "/api/federation/reaction-event", "reaction", ctx
        )

    async def ingest_inbound_reaction(
        self,
        raw_body: bytes,
        signature_b64: Optional[str],
        from_instance_header: Optional[str],
        rls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        ctx = rls or ADMIN_RLS
        payload, meta = self._verify_and_parse_envelope(raw_body, signature_b64, from_instance_header)
        peer = await self._verify_peer_signature_and_require_active(
            meta["from_instance"], raw_body, signature_b64, ctx
        )
        if payload.get("event_type") != "reaction":
            raise ValueError("Invalid event_type")
        from_user = str(payload.get("from_user") or "")
        if from_user and await self._federated_user_is_blocked(
            str(peer["peer_id"]), from_user, ctx
        ):
            return {"ok": True, "ignored": "blocked"}
        room_id = str(payload.get("room_id") or "")
        mw = str(payload.get("message_wire_id") or "")
        emoji = str(payload.get("emoji") or "")
        action = str(payload.get("action") or "add")
        if not room_id or not mw or not emoji:
            raise ValueError("room_id, message_wire_id, emoji required")
        local_mid = await self._resolve_local_message_id_for_wire(room_id, mw, ctx)
        if not local_mid:
            raise ValueError("Message not found for reaction")
        fu = await self.get_or_create_federated_user(
            str(peer["peer_id"]),
            from_user or "unknown",
            from_user,
            display_name=payload.get("from_user_display_name"),
            rls=ctx,
        )
        blk = await fetch_one(
            "SELECT is_blocked FROM federated_users WHERE federated_user_id = $1::uuid",
            fu,
            rls_context=ctx,
        )
        if blk and blk.get("is_blocked"):
            return {"ok": True, "ignored": "blocked"}
        if action == "remove":
            await execute(
                """
                DELETE FROM message_reactions
                WHERE message_id = $1::uuid AND federated_user_id = $2::uuid AND emoji = $3
                """,
                local_mid,
                fu,
                emoji,
                rls_context=ctx,
            )
        else:
            rid = str(uuid.uuid4())
            await execute(
                """
                INSERT INTO message_reactions (reaction_id, message_id, user_id, emoji, federated_user_id)
                SELECT $1::uuid, $2::uuid, NULL, $3, $4::uuid
                WHERE NOT EXISTS (
                    SELECT 1 FROM message_reactions
                    WHERE message_id = $2::uuid AND federated_user_id = $4::uuid AND emoji = $3
                )
                """,
                rid,
                local_mid,
                emoji,
                fu,
                rls_context=ctx,
            )
        ws = get_websocket_manager()
        await ws.broadcast_to_room(
            room_id,
            {
                "type": "reaction_update",
                "room_id": room_id,
                "message_id": local_mid,
                "emoji": emoji,
                "action": action,
                "federated_user_id": fu,
            },
            exclude_user_id=None,
        )
        return {"ok": True}

    async def deliver_outbound_read_receipt(
        self,
        room_id: str,
        user_id: str,
        last_read_iso: str,
        rls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        ctx = rls or ADMIN_RLS
        from_user = await self.federated_address_for_local_user(str(user_id), ctx)
        row = await self._get_peer_row_for_room(room_id, ctx)
        if not row:
            return {"ok": False, "reason": "not_federated_room"}
        meta = row.get("federation_metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        remote_room = meta.get("remote_room_id")
        if not remote_room:
            return {"ok": False, "reason": "no_remote_room_id"}
        envelope = {
            "event_type": "read_receipt",
            "room_id": str(remote_room),
            "user_address": from_user,
            "last_read_at": last_read_iso,
        }
        return await self._sign_deliver_bfp(
            room_id, envelope, "/api/federation/read-receipt-event", "read_receipt", ctx
        )

    async def ingest_inbound_read_receipt(
        self,
        raw_body: bytes,
        signature_b64: Optional[str],
        from_instance_header: Optional[str],
        rls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        ctx = rls or ADMIN_RLS
        payload, meta = self._verify_and_parse_envelope(raw_body, signature_b64, from_instance_header)
        peer = await self._verify_peer_signature_and_require_active(
            meta["from_instance"], raw_body, signature_b64, ctx
        )
        if payload.get("event_type") != "read_receipt":
            raise ValueError("Invalid event_type")
        addr = str(payload.get("user_address") or "")
        if addr and await self._federated_user_is_blocked(str(peer["peer_id"]), addr, ctx):
            return {"ok": True, "ignored": "blocked"}
        room_id = str(payload.get("room_id") or "")
        if not room_id:
            raise ValueError("room_id required")
        last_read = str(payload.get("last_read_at") or "")
        ws = get_websocket_manager()
        await ws.broadcast_to_room(
            room_id,
            {
                "type": "federated_read_receipt",
                "room_id": room_id,
                "user_address": addr,
                "last_read_at": last_read,
            },
            exclude_user_id=None,
        )
        return {"ok": True}

    async def deliver_outbound_presence_batch(
        self, peer_id: str, entries: List[Dict[str, Any]], rls: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        ctx = rls or ADMIN_RLS
        if not entries:
            return {"ok": True, "skipped": True}
        priv = await federation_service.get_private_key_b64_for_signing(ADMIN_RLS)
        if not priv:
            return {"ok": False, "reason": "no_private_key"}
        ident = await federation_service.get_instance_identity(ADMIN_RLS)
        if not ident:
            return {"ok": False, "reason": "no_identity"}
        peer = await fetch_one(
            """
            SELECT peer_id, peer_url, connectivity_mode, status
            FROM federation_peers WHERE peer_id = $1::uuid AND status = 'active'
            """,
            peer_id,
            rls_context=ADMIN_RLS,
        )
        if not peer:
            return {"ok": False, "reason": "no_peer"}
        envelope = {
            "bfp_version": BFP_VERSION,
            "event_type": "presence_batch",
            "from_instance": ident["instance_url"],
            "entries": entries,
        }
        wire = _canonical_json_bytes(envelope)
        sig = federation_crypto.sign_payload(priv, wire)
        peer_url = _normalize_instance_url(peer["peer_url"] or "")
        mode = peer.get("connectivity_mode") or "bidirectional"
        out_payload = {
            "bfp_wire_b64": base64.b64encode(wire).decode("ascii"),
            "signature": sig,
            "x_bastion_instance": ident["instance_url"],
        }
        if mode == "asymmetric_listener":
            await federation_service.enqueue_outbox(str(peer["peer_id"]), "presence_batch", out_payload, ADMIN_RLS)
            return {"ok": True, "channel": "outbox"}
        url = f"{peer_url}/api/federation/presence-event"
        try:
            async with httpx.AsyncClient(timeout=float(settings.FEDERATION_HTTP_TIMEOUT)) as client:
                r = await client.post(
                    url,
                    content=wire,
                    headers={
                        "Content-Type": "application/json",
                        "X-Bastion-Signature": sig,
                        "X-Bastion-Instance": ident["instance_url"],
                    },
                )
                if r.status_code >= 400:
                    return {"ok": False, "reason": "http_error", "status": r.status_code}
        except Exception as e:
            return {"ok": False, "reason": str(e)}
        return {"ok": True, "channel": "http"}

    async def ingest_inbound_presence_batch(
        self,
        raw_body: bytes,
        signature_b64: Optional[str],
        from_instance_header: Optional[str],
        rls: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        ctx = rls or ADMIN_RLS
        payload, meta = self._verify_and_parse_envelope(raw_body, signature_b64, from_instance_header)
        peer = await self._verify_peer_signature_and_require_active(
            meta["from_instance"], raw_body, signature_b64, ctx
        )
        if payload.get("event_type") != "presence_batch":
            raise ValueError("Invalid event_type")
        entries = payload.get("entries") or []
        if not isinstance(entries, list):
            entries = []
        ws = get_websocket_manager()
        out_entries: List[Dict[str, Any]] = []
        for ent in entries:
            if not isinstance(ent, dict):
                continue
            addr = str(ent.get("user_address") or "")
            if not addr:
                continue
            if await self._federated_user_is_blocked(str(peer["peer_id"]), addr, ctx):
                continue
            st = str(ent.get("status") or "offline")
            ls = str(ent.get("last_seen_at") or datetime.now(timezone.utc).isoformat())
            await self.get_or_create_federated_user(
                str(peer["peer_id"]),
                addr,
                addr,
                display_name=ent.get("display_name"),
                rls=ctx,
            )
            await execute(
                """
                UPDATE federated_users
                SET presence_status = $3,
                    presence_updated_at = NOW(),
                    last_seen_at = NOW()
                WHERE peer_id = $1::uuid AND federated_address = $2
                """,
                str(peer["peer_id"]),
                addr,
                st,
                rls_context=ctx,
            )
            out_entries.append(
                {
                    "peer_id": str(peer["peer_id"]),
                    "user_address": addr,
                    "status": st,
                    "last_seen_at": ls,
                }
            )
        if out_entries:
            urows = await fetch_all(
                """
                SELECT DISTINCT rp.user_id::text AS uid
                FROM room_participants rp
                JOIN chat_rooms r ON r.room_id = rp.room_id
                WHERE r.room_type = 'federated'
                  AND (r.federation_metadata->>'peer_id')::uuid = $1::uuid
                """,
                str(peer["peer_id"]),
                rls_context=ctx,
            )
            uids = [str(r["uid"]) for r in urows if r.get("uid")]
            if uids:
                await ws.broadcast_to_users(
                    uids,
                    {"type": "federated_presence_batch", "entries": out_entries},
                )
        return {"ok": True}


federation_message_service = FederationMessageService()
