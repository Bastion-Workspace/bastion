"""
Federation API — Phase 1: identity, pairing, outbox (admin + signed peer requests).
"""

from __future__ import annotations

import json
import logging
from urllib.parse import urlparse

from typing import Annotated, Any, Dict, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import settings
from models.api_models import AuthenticatedUserResponse
from services.database_manager.database_helpers import fetch_one
from services.federation_message_service import (
    PeerFederationInactiveError,
    federation_message_service,
)
from pathlib import Path

from services.federation_service import federation_service, normalize_instance_url
from services.messaging.messaging_attachment_service import messaging_attachment_service
from utils.auth_middleware import get_current_user, get_current_user_optional, require_admin

logger = logging.getLogger(__name__)

router = APIRouter(tags=["federation"])


def _require_federation() -> None:
    if not settings.FEDERATION_ENABLED:
        raise HTTPException(status_code=404, detail="Federation is not enabled")


def _admin_rls(user: AuthenticatedUserResponse) -> Dict[str, str]:
    return {"user_id": str(user.user_id), "user_role": "admin"}


def _admin_rls_static() -> Dict[str, str]:
    """RLS context for federation background reads (no HTTP user)."""
    return {"user_id": "", "user_role": "admin"}


async def _federation_inbound_rate_limit_or_raise(request: Request, raw_body: bytes) -> None:
    try:
        jo = json.loads(raw_body.decode("utf-8") or "{}")
    except Exception:
        jo = {}
    key = request.headers.get("X-Bastion-Instance") or jo.get("from_instance") or ""
    key = normalize_instance_url(str(key)) if key else "unknown"
    ok, retry_after = federation_service.check_inbound_federation_rate_limit(key)
    if not ok:
        await federation_service.log_federation_audit(
            action="federation.rate_limited",
            record_id=key,
            new_values={"retry_after": retry_after},
            rls=_admin_rls_static(),
        )
        raise HTTPException(
            status_code=429,
            detail="Federation inbound rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )


def _serialize_peer_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for k in ("peer_id",):
        if out.get(k) is not None:
            out[k] = str(out[k])
    for k in ("created_at", "activated_at"):
        if out.get(k) is not None:
            out[k] = out[k].isoformat() if hasattr(out[k], "isoformat") else str(out[k])
    md = out.get("metadata")
    if isinstance(md, str):
        try:
            md = json.loads(md)
        except json.JSONDecodeError:
            md = {}
    if isinstance(md, dict):
        out["last_sync_at"] = md.get("last_federation_sync_at")
    else:
        out["last_sync_at"] = None
    if "outbox_pending_count" in out and out["outbox_pending_count"] is not None:
        out["outbox_pending_count"] = int(out["outbox_pending_count"])
    return out


def _serialize_outbox_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    if out.get("outbox_id") is not None:
        out["outbox_id"] = str(out["outbox_id"])
    if out.get("peer_id") is not None:
        out["peer_id"] = str(out["peer_id"])
    if out.get("created_at") is not None:
        out["created_at"] = (
            out["created_at"].isoformat()
            if hasattr(out["created_at"], "isoformat")
            else str(out["created_at"])
        )
    pl = out.get("payload")
    if isinstance(pl, str):
        try:
            out["payload"] = json.loads(pl)
        except json.JSONDecodeError:
            pass
    return out


class InitiatePairingBody(BaseModel):
    peer_url: str = Field(..., min_length=4, description="Remote instance base URL")


class ProbePeerBody(BaseModel):
    peer_url: str = Field(..., min_length=4)


class PatchPeerBody(BaseModel):
    status: Literal["active", "suspended", "revoked"]


class CreateFederatedRoomBody(BaseModel):
    peer_id: str = Field(..., min_length=8, description="Active federation peer UUID")
    remote_user_address: str = Field(
        ...,
        min_length=3,
        description="Remote username@host (host should match peer URL)",
    )
    room_name: Optional[str] = Field(None, max_length=255)


class FederatedDmCreateBody(BaseModel):
    remote_user_address: str = Field(
        ...,
        min_length=3,
        description="Remote username@host (host must match an active peer URL)",
    )


@router.get("/api/federation/identity")
async def get_federation_identity():
    _require_federation()
    ident = await federation_service.get_instance_identity()
    if not ident:
        raise HTTPException(status_code=404, detail="Federation identity not initialized")
    return ident


@router.post("/api/federation/identity/initialize")
async def initialize_federation_identity(
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
):
    _require_federation()
    rls = _admin_rls(current_user)
    result = await federation_service.initialize_keypair(rls)
    return result


@router.post("/api/federation/identity/regenerate")
async def regenerate_federation_identity(
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
):
    _require_federation()
    rls = _admin_rls(current_user)
    result = await federation_service.regenerate_keypair(rls)
    return result


@router.post("/api/federation/peers/probe")
async def probe_remote_peer(
    body: ProbePeerBody,
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
):
    _require_federation()
    try:
        remote = await federation_service.fetch_remote_identity(body.peer_url)
        return {"ok": True, "remote": remote}
    except Exception as e:
        logger.warning("Federation probe failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Could not reach remote instance: {e}") from e


@router.post("/api/federation/peer-request")
async def federation_peer_request(
    request: Request,
    current_user: Annotated[
        Optional[AuthenticatedUserResponse], Depends(get_current_user_optional)
    ],
):
    _require_federation()
    raw = await request.body()
    sig = request.headers.get("X-Bastion-Signature")
    from_hdr = request.headers.get("X-Bastion-Instance")

    if sig:
        try:
            result = await federation_service.receive_signed_peer_message(
                raw, sig, from_hdr
            )
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception("Federation peer-request handling failed")
            raise HTTPException(status_code=500, detail=str(e)) from e

    if current_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    if getattr(current_user, "role", None) != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")

    try:
        body = InitiatePairingBody.model_validate_json(raw.decode("utf-8") or "{}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}") from e

    rls = _admin_rls(current_user)
    try:
        await federation_service.initialize_keypair(rls)
        return await federation_service.initiate_pairing(
            body.peer_url, str(current_user.user_id), rls
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Federation initiate pairing failed")
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.get("/api/federation/peers")
async def list_federation_peers(
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
):
    _require_federation()
    rls = _admin_rls(current_user)
    rows = await federation_service.list_peers(rls)
    return {"peers": [_serialize_peer_row(r) for r in rows]}


@router.patch("/api/federation/peers/{peer_id}")
async def patch_federation_peer(
    peer_id: str,
    body: PatchPeerBody,
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
):
    _require_federation()
    rls = _admin_rls(current_user)
    try:
        return await federation_service.update_peer_status(peer_id, body.status, rls)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.delete("/api/federation/peers/{peer_id}")
async def delete_federation_peer(
    peer_id: str,
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
):
    """Permanently remove a revoked peer so its URL can be paired again."""
    _require_federation()
    rls = _admin_rls(current_user)
    try:
        return await federation_service.delete_revoked_peer(
            peer_id, str(current_user.user_id), rls
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/api/federation/outbox/drain")
async def federation_outbox_drain(request: Request):
    _require_federation()
    raw = await request.body()
    sig = request.headers.get("X-Bastion-Signature")
    inst = request.headers.get("X-Bastion-Instance")
    if not sig or not inst:
        raise HTTPException(
            status_code=400, detail="X-Bastion-Signature and X-Bastion-Instance required"
        )
    try:
        rows = await federation_service.drain_outbox_for_peer_url(inst, raw, sig)
        return {"events": [_serialize_outbox_row(r) for r in rows]}
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e


@router.post("/api/federation/outbox/ack")
async def federation_outbox_ack(request: Request):
    _require_federation()
    raw = await request.body()
    sig = request.headers.get("X-Bastion-Signature")
    inst = request.headers.get("X-Bastion-Instance")
    if not sig or not inst:
        raise HTTPException(
            status_code=400, detail="X-Bastion-Signature and X-Bastion-Instance required"
        )
    try:
        await federation_service.ack_outbox(inst, raw, sig)
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e


@router.post("/api/federation/sync")
async def federation_sync_outbox(
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
):
    _require_federation()
    rls = _admin_rls(current_user)
    return await federation_service.sync_pull_for_local_instance(rls)


@router.post("/api/federation/message")
async def federation_inbound_message(request: Request):
    _require_federation()
    raw = await request.body()
    await _federation_inbound_rate_limit_or_raise(request, raw)
    sig = request.headers.get("X-Bastion-Signature")
    from_hdr = request.headers.get("X-Bastion-Instance")
    try:
        return await federation_message_service.ingest_inbound_message(raw, sig, from_hdr)
    except PeerFederationInactiveError as e:
        raise HTTPException(
            status_code=403,
            detail=f"Federation with this peer is not active ({e.status})",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Federation message ingest failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/federation/room-invite")
async def federation_room_invite(request: Request):
    _require_federation()
    raw = await request.body()
    await _federation_inbound_rate_limit_or_raise(request, raw)
    sig = request.headers.get("X-Bastion-Signature")
    from_hdr = request.headers.get("X-Bastion-Instance")
    try:
        return await federation_message_service.handle_room_invite(raw, sig, from_hdr)
    except PeerFederationInactiveError as e:
        raise HTTPException(
            status_code=403,
            detail=f"Federation with this peer is not active ({e.status})",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Federation room-invite failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/federation/room-invite-accept")
async def federation_room_invite_accept(request: Request):
    _require_federation()
    raw = await request.body()
    await _federation_inbound_rate_limit_or_raise(request, raw)
    sig = request.headers.get("X-Bastion-Signature")
    from_hdr = request.headers.get("X-Bastion-Instance")
    try:
        return await federation_message_service.handle_room_invite_accept(raw, sig, from_hdr)
    except PeerFederationInactiveError as e:
        raise HTTPException(
            status_code=403,
            detail=f"Federation with this peer is not active ({e.status})",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Federation room-invite-accept failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/federation/rooms")
async def federation_create_federated_room(
    body: CreateFederatedRoomBody,
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
):
    _require_federation()
    rls = _admin_rls(current_user)
    addr = (body.remote_user_address or "").strip()
    if "@" not in addr:
        raise HTTPException(status_code=400, detail="remote_user_address must be user@host")
    remote_username, remote_host = addr.rsplit("@", 1)
    remote_username = remote_username.strip()
    remote_host = remote_host.strip().lower()
    if not remote_username or not remote_host:
        raise HTTPException(status_code=400, detail="Invalid remote_user_address")

    peer_row = await federation_service.get_peer_by_id(body.peer_id, rls)
    if not peer_row:
        raise HTTPException(status_code=404, detail="Peer not found")
    if (peer_row.get("status") or "") != "active":
        raise HTTPException(status_code=400, detail="Peer must be active")
    peer_netloc = urlparse(peer_row["peer_url"] or "").netloc.lower()
    if peer_netloc and remote_host != peer_netloc:
        raise HTTPException(
            status_code=400,
            detail=f"Address host must match peer URL host ({peer_netloc})",
        )

    try:
        room = await federation_message_service.create_federated_room(
            creator_user_id=str(current_user.user_id),
            peer_id=body.peer_id,
            room_name=body.room_name,
            rls=rls,
        )
        await federation_message_service.send_room_invite(
            peer_id=body.peer_id,
            local_room_id=str(room["room_id"]),
            target_local_username=remote_username,
            inviter_user_id=str(current_user.user_id),
            room_name=body.room_name or room.get("room_name"),
            rls=rls,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Federation create federated room failed")
        raise HTTPException(status_code=502, detail=str(e)) from e

    return {"ok": True, "room": room}


@router.post("/api/federation/users/resolve")
async def federation_users_resolve_signed(request: Request):
    """Signed peer-to-peer: resolve a discoverable local user by federated address."""
    _require_federation()
    raw = await request.body()
    sig = request.headers.get("X-Bastion-Signature")
    inst = request.headers.get("X-Bastion-Instance")
    try:
        await federation_service.verify_signed_peer_messaging_scope(raw, sig, inst)
        body = json.loads(raw.decode("utf-8") or "{}")
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
    address = (body.get("address") or "").strip()
    if not address:
        raise HTTPException(status_code=400, detail="address is required")
    return await federation_service.resolve_discoverable_user(address, rls=_admin_rls_static())


@router.post("/api/federation/users/directory")
async def federation_users_directory_signed(request: Request):
    """Signed peer-to-peer: paginated directory of discoverable local users."""
    _require_federation()
    raw = await request.body()
    sig = request.headers.get("X-Bastion-Signature")
    inst = request.headers.get("X-Bastion-Instance")
    try:
        await federation_service.verify_signed_peer_messaging_scope(raw, sig, inst)
        body = json.loads(raw.decode("utf-8") or "{}")
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
    search = body.get("search")
    limit = int(body.get("limit") or 50)
    offset = int(body.get("offset") or 0)
    return await federation_service.list_discoverable_users_directory(
        search, limit, offset, rls=_admin_rls_static()
    )


@router.get("/api/federation/users/resolve-remote")
async def federation_users_resolve_remote_proxy(
    address: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """JWT-authenticated proxy: resolve a remote user via a signed request to the peer instance."""
    _require_federation()
    addr = (address or "").strip()
    if "@" not in addr:
        raise HTTPException(status_code=400, detail="address must be user@host")
    _, remote_host = addr.rsplit("@", 1)
    remote_host = remote_host.strip().lower()
    peer = await federation_service.find_active_peer_by_address_host(
        remote_host, rls=_admin_rls_static()
    )
    if not peer:
        raise HTTPException(
            status_code=400,
            detail=f"No active federation peer for {remote_host}",
        )
    peer_url = peer.get("peer_url") or ""
    try:
        r = await federation_service.signed_post_json_to_peer(
            peer_url,
            "/api/federation/users/resolve",
            {"address": addr},
            rls=_admin_rls_static(),
        )
    except Exception as e:
        logger.warning("resolve-remote peer request failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e)) from e
    if r.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Peer returned HTTP {r.status_code}: {r.text[:300]}",
        )
    try:
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Invalid peer response: {e}") from e


@router.post("/api/federation/rooms/dm")
async def federation_create_user_dm_room(
    body: FederatedDmCreateBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Create a federated DM room as the current user (non-admin); peer inferred from address host."""
    _require_federation()
    addr = (body.remote_user_address or "").strip()
    if "@" not in addr:
        raise HTTPException(status_code=400, detail="remote_user_address must be user@host")
    remote_username, remote_host = addr.rsplit("@", 1)
    remote_username = remote_username.strip()
    remote_host = remote_host.strip().lower()
    if not remote_username or not remote_host:
        raise HTTPException(status_code=400, detail="Invalid remote_user_address")
    peer_row = await federation_service.find_active_peer_by_address_host(
        remote_host, rls=_admin_rls_static()
    )
    if not peer_row:
        raise HTTPException(
            status_code=400,
            detail=f"No active federation peer for {remote_host}",
        )
    peer_netloc = urlparse(peer_row["peer_url"] or "").netloc.lower()
    if peer_netloc and remote_host.rstrip(".") != peer_netloc.rstrip("."):
        raise HTTPException(
            status_code=400,
            detail=f"Address host must match peer URL host ({peer_netloc})",
        )
    peer_id = str(peer_row["peer_id"])
    user_rls = {
        "user_id": str(current_user.user_id),
        "user_role": getattr(current_user, "role", None) or "user",
    }
    try:
        room = await federation_message_service.create_federated_room(
            creator_user_id=str(current_user.user_id),
            peer_id=peer_id,
            room_name=None,
            rls=user_rls,
        )
        await federation_message_service.send_room_invite(
            peer_id=peer_id,
            local_room_id=str(room["room_id"]),
            target_local_username=remote_username,
            inviter_user_id=str(current_user.user_id),
            room_name=room.get("room_name"),
            rls=user_rls,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Federation user DM room failed")
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"ok": True, "room": room}


@router.get("/api/federation/rooms/{room_id}/status")
async def federation_room_federation_status(
    room_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Participant-only: federation peer status for a room (for UI banners)."""
    _require_federation()
    user_rls = {
        "user_id": str(current_user.user_id),
        "user_role": getattr(current_user, "role", None) or "user",
    }
    row = await fetch_one(
        """
        SELECT r.room_type, r.federation_metadata,
               p.status AS federation_peer_status,
               p.peer_url AS federation_peer_url,
               p.display_name AS federation_peer_display_name
        FROM chat_rooms r
        JOIN room_participants rp ON rp.room_id = r.room_id AND rp.user_id = $2
        LEFT JOIN federation_peers p
          ON r.room_type = 'federated'
         AND p.peer_id = (r.federation_metadata->>'peer_id')::uuid
        WHERE r.room_id = $1::uuid
        """,
        room_id,
        str(current_user.user_id),
        rls_context=user_rls,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Room not found")
    if (row.get("room_type") or "") != "federated":
        return {"federated": False}
    meta = row.get("federation_metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {}
    return {
        "federated": True,
        "federation_metadata": meta if isinstance(meta, dict) else {},
        "federation_peer_status": row.get("federation_peer_status"),
        "federation_peer_url": row.get("federation_peer_url"),
        "federation_peer_display_name": row.get("federation_peer_display_name"),
    }


@router.get("/api/federation/attachments/{attachment_id}/file")
async def federation_serve_messaging_attachment(
    attachment_id: str,
    token: str = Query(..., description="HMAC time-limited federation token"),
):
    """Peer fetch of a messaging attachment (token auth, no JWT)."""
    _require_federation()
    if not federation_service.verify_messaging_attachment_federation_token(
        attachment_id, token
    ):
        raise HTTPException(status_code=403, detail="Invalid or expired attachment token")
    await messaging_attachment_service.initialize()
    row = await messaging_attachment_service.get_attachment_row_admin(attachment_id)
    if not row:
        raise HTTPException(status_code=404, detail="Attachment not found")
    await federation_service.log_federation_audit(
        action="federation.attachment.fetch",
        record_id=str(attachment_id),
        new_values={"room_id": str(row.get("room_id") or "")},
        rls=_admin_rls_static(),
    )
    fp = Path(str(row["file_path"]))
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Attachment file missing")
    return FileResponse(
        path=str(fp),
        media_type=str(row.get("mime_type") or "application/octet-stream"),
        filename=str(row.get("filename") or "attachment"),
    )


@router.post("/api/federation/attachment-event")
async def federation_attachment_event(request: Request):
    _require_federation()
    raw = await request.body()
    await _federation_inbound_rate_limit_or_raise(request, raw)
    sig = request.headers.get("X-Bastion-Signature")
    from_hdr = request.headers.get("X-Bastion-Instance")
    try:
        return await federation_message_service.ingest_inbound_attachment_event(
            raw, sig, from_hdr
        )
    except PeerFederationInactiveError as e:
        raise HTTPException(
            status_code=403,
            detail=f"Federation with this peer is not active ({e.status})",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Federation attachment-event failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/federation/reaction-event")
async def federation_reaction_event(request: Request):
    _require_federation()
    raw = await request.body()
    await _federation_inbound_rate_limit_or_raise(request, raw)
    sig = request.headers.get("X-Bastion-Signature")
    from_hdr = request.headers.get("X-Bastion-Instance")
    try:
        return await federation_message_service.ingest_inbound_reaction(raw, sig, from_hdr)
    except PeerFederationInactiveError as e:
        raise HTTPException(
            status_code=403,
            detail=f"Federation with this peer is not active ({e.status})",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Federation reaction-event failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/federation/read-receipt-event")
async def federation_read_receipt_event(request: Request):
    _require_federation()
    raw = await request.body()
    await _federation_inbound_rate_limit_or_raise(request, raw)
    sig = request.headers.get("X-Bastion-Signature")
    from_hdr = request.headers.get("X-Bastion-Instance")
    try:
        return await federation_message_service.ingest_inbound_read_receipt(
            raw, sig, from_hdr
        )
    except PeerFederationInactiveError as e:
        raise HTTPException(
            status_code=403,
            detail=f"Federation with this peer is not active ({e.status})",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Federation read-receipt-event failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/federation/presence-event")
async def federation_presence_event(request: Request):
    _require_federation()
    raw = await request.body()
    await _federation_inbound_rate_limit_or_raise(request, raw)
    sig = request.headers.get("X-Bastion-Signature")
    from_hdr = request.headers.get("X-Bastion-Instance")
    try:
        return await federation_message_service.ingest_inbound_presence_batch(
            raw, sig, from_hdr
        )
    except PeerFederationInactiveError as e:
        raise HTTPException(
            status_code=403,
            detail=f"Federation with this peer is not active ({e.status})",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Federation presence-event failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/federation/federated-users")
async def federation_list_federated_users(
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
):
    _require_federation()
    rows = await federation_service.list_federated_users(_admin_rls(current_user))
    return {"users": rows}


@router.post("/api/federation/federated-users/{federated_user_id}/block")
async def federation_block_federated_user(
    federated_user_id: str,
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
):
    _require_federation()
    await federation_service.set_federated_user_blocked(
        federated_user_id, True, str(current_user.user_id), _admin_rls(current_user)
    )
    return {"ok": True, "blocked": True}


@router.post("/api/federation/federated-users/{federated_user_id}/unblock")
async def federation_unblock_federated_user(
    federated_user_id: str,
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
):
    _require_federation()
    await federation_service.set_federated_user_blocked(
        federated_user_id, False, str(current_user.user_id), _admin_rls(current_user)
    )
    return {"ok": True, "blocked": False}
