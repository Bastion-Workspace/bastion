"""
Ebooks: OPDS catalog proxy, reader settings, recently opened metadata, KoSync proxy.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from models.api_models import AuthenticatedUserResponse
from models.ebooks_models import (
    EbooksSettingsResponse,
    EbooksSettingsUpdate,
    KosyncAuthTestRequest,
    KosyncProgressPut,
    KosyncRegisterRequest,
    OpdsCatalogEntry,
    OpdsCatalogEntryResponse,
    OpdsFetchRequest,
)
from services.kosync_client_service import (
    kosync_authorize,
    kosync_get_progress,
    kosync_healthcheck,
    kosync_put_progress,
    kosync_register,
    md5_hex_password,
)
from services.opds_fetch_service import fetch_opds_resource
from services.opds_url_validator import assert_http_catalog_url, normalize_catalog_root_url
from services.user_settings_kv_service import get_user_setting, set_user_setting
from utils.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ebooks"])

KEY_CATALOGS = "ebooks_opds_catalogs"
KEY_READER_PREFS = "ebooks_reader_prefs"
KEY_RECENT = "ebooks_recently_opened"
KEY_KOSYNC = "ebooks_kosync"


def _json_default_list() -> list:
    return []


def _json_default_dict() -> dict:
    return {}


async def _get_json_setting(user_id: str, key: str, default: Any) -> Any:
    raw = await get_user_setting(user_id, key)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


async def _set_json_setting(user_id: str, key: str, value: Any) -> None:
    await set_user_setting(user_id, key, json.dumps(value), data_type="json")


def _catalog_by_id(catalogs: List[dict], catalog_id: str) -> Optional[dict]:
    for c in catalogs:
        if str(c.get("id")) == str(catalog_id):
            return c
    return None


def _catalog_to_response_row(c: OpdsCatalogEntry) -> OpdsCatalogEntryResponse:
    b64 = c.http_basic_b64
    configured = bool(b64 and str(b64).strip())
    return OpdsCatalogEntryResponse(
        id=c.id,
        title=c.title,
        root_url=c.root_url,
        verify_ssl=c.verify_ssl,
        http_basic_configured=configured,
    )


@router.get("/api/ebooks/settings", response_model=EbooksSettingsResponse)
async def get_ebooks_settings(current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    catalogs_raw = await _get_json_setting(current_user.user_id, KEY_CATALOGS, _json_default_list())
    catalogs: List[OpdsCatalogEntryResponse] = []
    for c in catalogs_raw:
        try:
            catalogs.append(_catalog_to_response_row(OpdsCatalogEntry(**c)))
        except Exception:
            continue
    reader_prefs = await _get_json_setting(current_user.user_id, KEY_READER_PREFS, _json_default_dict())
    recent = await _get_json_setting(current_user.user_id, KEY_RECENT, _json_default_list())
    ks = await _get_json_setting(current_user.user_id, KEY_KOSYNC, _json_default_dict())
    kosync_out: Dict[str, Any] = {
        "configured": bool(ks.get("base_url") and ks.get("username") and ks.get("userkey")),
        "base_url": ks.get("base_url") or "",
        "username": ks.get("username") or "",
        "verify_ssl": ks.get("verify_ssl", True),
    }
    return EbooksSettingsResponse(
        catalogs=catalogs,
        reader_prefs=reader_prefs if isinstance(reader_prefs, dict) else {},
        recently_opened=recent if isinstance(recent, list) else [],
        kosync=kosync_out,
    )


@router.put("/api/ebooks/settings", response_model=EbooksSettingsResponse)
async def put_ebooks_settings(
    body: EbooksSettingsUpdate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    if body.catalogs is not None:
        existing_raw = await _get_json_setting(current_user.user_id, KEY_CATALOGS, _json_default_list())
        existing_by_id: Dict[str, dict] = {str(x.get("id")): x for x in existing_raw if x.get("id") is not None}
        normalized_entries: List[OpdsCatalogEntry] = []
        for c in body.catalogs:
            normalized = c.model_copy(update={"root_url": normalize_catalog_root_url(c.root_url)})
            try:
                assert_http_catalog_url(normalized.root_url)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            cid = str(normalized.id)
            old = existing_by_id.get(cid)
            inc_b64 = normalized.http_basic_b64
            if inc_b64 is not None and str(inc_b64).strip():
                merged = normalized
            elif inc_b64 is not None and str(inc_b64).strip() == "":
                merged = normalized.model_copy(update={"http_basic_b64": None})
            else:
                prev = (old or {}).get("http_basic_b64")
                merged = (
                    normalized.model_copy(update={"http_basic_b64": prev})
                    if prev and str(prev).strip()
                    else normalized.model_copy(update={"http_basic_b64": None})
                )
            normalized_entries.append(merged)
        await _set_json_setting(
            current_user.user_id,
            KEY_CATALOGS,
            [c.model_dump() for c in normalized_entries],
        )
    if body.reader_prefs is not None:
        await _set_json_setting(current_user.user_id, KEY_READER_PREFS, body.reader_prefs)
    if body.recently_opened is not None:
        await _set_json_setting(current_user.user_id, KEY_RECENT, body.recently_opened)
    return await get_ebooks_settings(current_user)


class KosyncSettingsBody(BaseModel):
    base_url: str = ""
    username: str = ""
    password: Optional[str] = Field(None, description="If set, replaces stored userkey (MD5)")
    verify_ssl: bool = True


@router.put("/api/ebooks/kosync/settings", response_model=EbooksSettingsResponse)
async def put_kosync_settings(
    body: KosyncSettingsBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    existing = await _get_json_setting(current_user.user_id, KEY_KOSYNC, _json_default_dict())
    merged = {
        "base_url": body.base_url.strip(),
        "username": body.username.strip(),
        "verify_ssl": body.verify_ssl,
        "userkey": existing.get("userkey") or "",
    }
    if body.password:
        merged["userkey"] = md5_hex_password(body.password)
    await _set_json_setting(current_user.user_id, KEY_KOSYNC, merged)
    return await get_ebooks_settings(current_user)


@router.post("/api/ebooks/opds/fetch")
async def post_opds_fetch(
    body: OpdsFetchRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    catalogs_raw = await _get_json_setting(current_user.user_id, KEY_CATALOGS, _json_default_list())
    cat = _catalog_by_id(catalogs_raw, body.catalog_id)
    if not cat:
        raise HTTPException(status_code=404, detail="Unknown catalog_id")
    catalog_root = normalize_catalog_root_url(cat.get("root_url") or "")
    try:
        assert_http_catalog_url(catalog_root)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    http_basic = cat.get("http_basic_b64")
    verify_ssl = bool(cat.get("verify_ssl", True))
    try:
        mode, payload = await fetch_opds_resource(
            catalog_root=catalog_root,
            url=body.url.strip(),
            want=body.want,
            http_basic=http_basic,
            verify_ssl=verify_ssl,
        )
        if mode == "octet" and isinstance(payload, tuple) and len(payload) == 3:
            raw_bytes, _fetched_url, media_type = payload
            return Response(
                content=raw_bytes,
                media_type=media_type or "application/octet-stream",
                headers={"Cache-Control": "private, max-age=60"},
            )
        if isinstance(payload, dict):
            return payload
        raise HTTPException(status_code=500, detail="Unexpected OPDS fetch payload")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("OPDS fetch failed: %s", e)
        raise HTTPException(status_code=502, detail="Upstream fetch failed") from e


@router.post("/api/ebooks/kosync/test")
async def post_kosync_test(
    body: KosyncAuthTestRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    _ = current_user
    userkey = md5_hex_password(body.password)
    status, resp = await kosync_authorize(body.base_url, body.username, userkey, body.verify_ssl)
    ok = status == 200
    return {"ok": ok, "status": status, "body": resp}


@router.post("/api/ebooks/kosync/register")
async def post_kosync_register(
    body: KosyncRegisterRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    ks = await _get_json_setting(current_user.user_id, KEY_KOSYNC, _json_default_dict())
    if body.base_url and body.base_url.strip():
        ks["base_url"] = body.base_url.strip()
    ks["verify_ssl"] = body.verify_ssl
    await _set_json_setting(current_user.user_id, KEY_KOSYNC, ks)
    base = (ks.get("base_url") or "").strip()
    verify = bool(ks.get("verify_ssl", True))
    if not base:
        raise HTTPException(status_code=400, detail="KoSync base URL is required")
    userkey = md5_hex_password(body.password)
    status, resp = await kosync_register(base, body.username, body.password, verify)
    if status not in (201, 200):
        raise HTTPException(status_code=400, detail={"status": status, "body": resp})
    ks["username"] = body.username.strip()
    ks["userkey"] = userkey
    await _set_json_setting(current_user.user_id, KEY_KOSYNC, ks)
    return {"ok": True, "username": ks["username"]}


@router.get("/api/ebooks/kosync/progress/{document}")
async def get_kosync_progress(
    document: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    if not document or len(document) > 64 or any(c not in "0123456789abcdefABCDEF" for c in document):
        raise HTTPException(status_code=400, detail="Invalid document digest")
    ks = await _get_json_setting(current_user.user_id, KEY_KOSYNC, _json_default_dict())
    base = (ks.get("base_url") or "").strip()
    user = (ks.get("username") or "").strip()
    key = (ks.get("userkey") or "").strip()
    verify = bool(ks.get("verify_ssl", True))
    if not (base and user and key):
        raise HTTPException(status_code=400, detail="KoSync is not configured")
    status, body = await kosync_get_progress(base, user, key, document, verify)
    if status == 401:
        raise HTTPException(status_code=401, detail="KoSync unauthorized")
    if status != 200:
        raise HTTPException(status_code=502, detail={"status": status, "body": body})
    return body


@router.put("/api/ebooks/kosync/progress")
async def put_kosync_progress(
    body: KosyncProgressPut,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    ks = await _get_json_setting(current_user.user_id, KEY_KOSYNC, _json_default_dict())
    base = (ks.get("base_url") or "").strip()
    user = (ks.get("username") or "").strip()
    key = (ks.get("userkey") or "").strip()
    verify = bool(ks.get("verify_ssl", True))
    if not (base and user and key):
        raise HTTPException(status_code=400, detail="KoSync is not configured")
    payload = {
        "document": body.document,
        "progress": body.progress,
        "percentage": body.percentage,
        "device": body.device,
        "device_id": body.device_id or "bastion-web",
    }
    status, resp = await kosync_put_progress(base, user, key, payload, verify)
    if status == 401:
        raise HTTPException(status_code=401, detail="KoSync unauthorized")
    if status not in (200, 202):
        raise HTTPException(status_code=502, detail={"status": status, "body": resp})
    return resp


@router.get("/api/ebooks/kosync/health")
async def get_kosync_health_proxy(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    ks = await _get_json_setting(current_user.user_id, KEY_KOSYNC, _json_default_dict())
    base = (ks.get("base_url") or "").strip()
    verify = bool(ks.get("verify_ssl", True))
    if not base:
        raise HTTPException(status_code=400, detail="KoSync base URL not configured")
    ok, msg = await kosync_healthcheck(base, verify)
    return {"ok": ok, "detail": msg}
