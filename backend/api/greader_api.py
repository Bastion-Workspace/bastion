"""
Google Reader compatible API for mobile/desktop RSS clients.
Base URL: /api/greader (e.g. https://host/api/greader/accounts/ClientLogin).
"""

from __future__ import annotations

import hmac
import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from config import settings
from models.api_models import AuthenticatedUserResponse
from tools_service.services.rss_service import get_rss_service
from services.auth_service import auth_service
from api.greader_helpers import (
    GOOGLE_READING_LIST,
    TAG_READ,
    TAG_STARRED,
    article_to_greader_item,
    article_to_greader_item_ref,
    greader_user_label_stream_id,
    normalize_stream_id,
    parse_form_pairs,
    parse_item_id,
    rss_category_display_label,
    stream_params,
    system_tag_list,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["GReader"])


def _extract_greader_auth_header(request: Request) -> Optional[str]:
    auth = request.headers.get("Authorization") or request.headers.get("authorization") or ""
    auth = auth.strip()
    low = auth.lower()
    if low.startswith("googlelogin auth="):
        return auth.split("=", 1)[1].strip()
    return None


async def _greader_user_optional(request: Request) -> Optional[AuthenticatedUserResponse]:
    token = _extract_greader_auth_header(request)
    if not token:
        return None
    return await auth_service.verify_greader_auth_token(token)


async def require_greader_user(request: Request) -> AuthenticatedUserResponse:
    user = await _greader_user_optional(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


def _verify_write_token(request: Request, t_param: Optional[str]) -> None:
    """When T is sent, it must match the session token derived from the auth header."""
    auth_raw = _extract_greader_auth_header(request)
    if not auth_raw:
        raise HTTPException(status_code=401, detail="Unauthorized")
    t_val = (t_param or request.query_params.get("T") or "").strip()
    if not t_val:
        return
    expected = auth_service.greader_session_token_value(auth_raw)
    if not hmac.compare_digest(t_val, expected):
        raise HTTPException(status_code=401, detail="Invalid session token")


async def _client_login_form(request: Request) -> tuple[str, str]:
    email = ""
    passwd = ""
    if request.query_params.get("Email"):
        email = request.query_params.get("Email") or ""
        passwd = request.query_params.get("Passwd") or ""
    body = await request.body()
    if body:
        form = parse_form_pairs(body)
        email = (form.get("Email") or form.get("email") or [email or ""])[0]
        passwd = (form.get("Passwd") or form.get("passwd") or [passwd or ""])[0]
    return email.strip(), passwd


@router.post("/accounts/ClientLogin")
async def greader_client_login(request: Request):
    email, passwd = await _client_login_form(request)
    if not email or not passwd:
        raise HTTPException(status_code=403, detail="Bad authentication")
    token = await auth_service.authenticate_greader_client_login(email, passwd)
    if not token:
        raise HTTPException(status_code=403, detail="Bad authentication")
    lines = [
        f"SID={token}",
        "LSID=null",
        f"Auth={token}",
        "expires_in=604800",
    ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")


@router.get("/reader/api/0/token")
async def greader_token(request: Request):
    user = await require_greader_user(request)
    auth_raw = _extract_greader_auth_header(request)
    assert auth_raw
    val = auth_service.greader_session_token_value(auth_raw)
    logger.info(
        "GReader token user=%s token_len=%s",
        user.user_id,
        len(val) if val else 0,
    )
    return PlainTextResponse(val, media_type="text/plain")


@router.get("/reader/api/0/user-info")
async def greader_user_info(request: Request, output: str = "json"):
    user = await require_greader_user(request)
    return {
        "userId": user.user_id,
        "userName": user.username,
        "userProfileId": user.user_id,
        "userEmail": user.email or user.username,
    }


@router.get("/reader/api/0/subscription/list")
async def greader_subscription_list(request: Request, output: str = "json"):
    user = await require_greader_user(request)
    rss = await get_rss_service()
    feeds = await rss.get_user_feeds(user.user_id, is_admin=False)
    subs = []
    for f in feeds:
        raw = (f.category or "").strip()
        cat = [
            {
                "id": greader_user_label_stream_id(raw),
                "label": rss_category_display_label(raw),
            }
        ]
        subs.append(
            {
                "id": f"feed/{f.feed_id}",
                "title": f.feed_name,
                "categories": cat,
                "url": f.feed_url,
                "htmlUrl": f.feed_url,
                "iconUrl": "",
            }
        )
    return {"subscriptions": subs}


@router.get("/reader/api/0/tag/list")
async def greader_tag_list(request: Request, output: str = "json"):
    user = await require_greader_user(request)
    rss = await get_rss_service()
    feeds = await rss.get_user_feeds(user.user_id, is_admin=False)
    tags: List[Dict[str, str]] = list(system_tag_list())
    seen = {t["id"] for t in tags}
    for f in feeds:
        raw = (f.category or "").strip()
        tid = greader_user_label_stream_id(raw)
        label = rss_category_display_label(raw)
        if tid not in seen:
            tags.append({"id": tid, "label": label})
            seen.add(tid)
    return {"tags": tags}


@router.get("/reader/api/0/unread-count")
async def greader_unread_count(
    request: Request,
    output: str = "json",
    include_all: str = Query("", alias="all"),
):
    user = await require_greader_user(request)
    rss = await get_rss_service()
    counts = await rss.get_unread_count(user.user_id)
    feeds = await rss.get_user_feeds(user.user_id, is_admin=False)
    include_zero = include_all == "1"
    feed_ids = {f.feed_id for f in feeds} | set(counts.keys())
    unread_list: List[Dict[str, Any]] = []
    total = 0
    for fid in sorted(feed_ids):
        c = int(counts.get(fid, 0))
        if not include_zero and c == 0:
            continue
        total += c
        unread_list.append(
            {
                "id": f"feed/{fid}",
                "count": c,
                "newestItemTimestampUsec": "0",
            }
        )
    # Many clients only look for this aggregate id (per-feed rows are not enough).
    unreadcounts_out: List[Dict[str, Any]] = [
        {
            "id": GOOGLE_READING_LIST,
            "count": total,
            "newestItemTimestampUsec": "0",
        }
    ]
    unreadcounts_out.extend(unread_list)
    logger.info(
        "GReader unread-count user=%s max=%s feeds_with_rows=%s per_feed_entries=%s "
        "aggregate_reading_list_count=%s include_all_zero=%s",
        user.user_id,
        total,
        len(counts),
        len(unread_list),
        total,
        include_zero,
    )
    return {"max": total, "unreadcounts": unreadcounts_out}


def _continuation_offset(c: Optional[str]) -> int:
    if not c:
        return 0
    try:
        return int(c)
    except ValueError:
        return 0


def _parse_ts(ts: Optional[str]) -> Optional[int]:
    if not ts:
        return None
    try:
        v = int(ts)
        if v > 10_000_000_000_000:
            return v // 1_000_000
        if v > 10_000_000_000:
            return v // 1000
        return v
    except ValueError:
        return None


async def _feed_map(rss, feed_ids: set) -> Dict[str, Any]:
    out = {}
    for fid in feed_ids:
        f = await rss.get_feed(fid)
        out[fid] = f
    return out


def _greader_it_merged_flags(
    it: Optional[str],
    starred_only: bool,
    read_stream_only: bool,
) -> tuple[bool, bool]:
    """
    Merge stream-derived flags with include-target `it` (Google Reader / FreshRSS).

    Clients such as Read You (FreshRSS mode) send e.g.
    s=reading-list&it=user/-/state/com.google/read to list only read items.
    """
    it_s = (it or "").strip()
    merged_starred = starred_only or ("com.google/starred" in it_s)
    merged_read_only = read_stream_only or ("com.google/read" in it_s)
    return merged_starred, merged_read_only


async def _stream_articles(
    request: Request,
    stream_id: str,
    n: int = 20,
    c: Optional[str] = None,
    ot: Optional[str] = None,
    xt: Optional[str] = None,
    it: Optional[str] = None,
) -> Dict[str, Any]:
    user = await require_greader_user(request)
    rss = await get_rss_service()
    sid = normalize_stream_id(stream_id)
    feed_id, starred_only, read_stream_only = stream_params(sid)
    starred_only, read_stream_only = _greader_it_merged_flags(it, starred_only, read_stream_only)
    exclude_read = bool(xt and "com.google/read" in xt)
    offset = _continuation_offset(c)
    limit = min(max(n, 1), 500)
    # Do not pass GReader `ot` into SQL. Clients often send a stale sync watermark on the
    # first request; we interpreted it as "published < ot", which hides all newer articles.
    articles = await rss.get_articles_paginated(
        user.user_id,
        feed_id=feed_id,
        starred_only=starred_only,
        limit=limit,
        offset=offset,
        older_than_ts=None,
        exclude_read=exclude_read,
        include_only_read=read_stream_only,
    )
    skipped_gid = sum(1 for a in articles if a.greader_id is None)
    fids = {a.feed_id for a in articles}
    fmap = await _feed_map(rss, fids)
    items: List[Dict[str, Any]] = []
    for a in articles:
        it_json = article_to_greader_item(a, fmap.get(a.feed_id))
        if it_json:
            items.append(it_json)
    logger.info(
        "GReader stream/contents user=%s stream=%s feed_id=%s starred_only=%s read_stream_only=%s "
        "exclude_read=%s offset=%s limit=%s client_ot=%s client_xt=%s client_it=%s db_rows=%s "
        "items_out=%s skipped_null_greader_id=%s",
        user.user_id,
        sid,
        feed_id,
        starred_only,
        read_stream_only,
        exclude_read,
        offset,
        limit,
        ot,
        xt,
        it,
        len(articles),
        len(items),
        skipped_gid,
    )
    next_c = str(offset + len(items)) if len(items) >= limit else None
    payload: Dict[str, Any] = {
        "id": sid,
        "updated": int(time.time()),
        "items": items,
        "direction": "ltr",
    }
    if next_c:
        payload["continuation"] = next_c
    return payload


@router.api_route(
    "/reader/api/0/stream/contents/{stream_id:path}",
    methods=["GET", "POST"],
)
async def greader_stream_contents(
    request: Request,
    stream_id: str,
    n: int = 20,
    c: Optional[str] = None,
    ot: Optional[str] = None,
    xt: Optional[str] = None,
    it: Optional[str] = None,
):
    if request.method == "POST":
        body = await request.body()
        form = parse_form_pairs(body)
        if form.get("n"):
            try:
                n = int(form["n"][0])
            except (ValueError, IndexError):
                pass
        if form.get("c"):
            c = form["c"][0]
        if form.get("ot"):
            ot = form["ot"][0]
        if form.get("xt"):
            xt = form["xt"][0]
        if form.get("it"):
            it = form["it"][0]
    return await _stream_articles(request, stream_id, n=n, c=c, ot=ot, xt=xt, it=it)


@router.get("/reader/api/0/stream/items/ids")
async def greader_stream_items_ids(
    request: Request,
    s: str,
    n: int = 20,
    c: Optional[str] = None,
    ot: Optional[str] = None,
    xt: Optional[str] = None,
    it: Optional[str] = None,
    r: Optional[str] = None,
    output: str = "json",
):
    user = await require_greader_user(request)
    rss = await get_rss_service()
    sid = normalize_stream_id(s)
    feed_id, starred_only, read_stream_only = stream_params(sid)
    starred_only, read_stream_only = _greader_it_merged_flags(it, starred_only, read_stream_only)
    exclude_read = bool(xt and "com.google/read" in xt)
    offset = _continuation_offset(c)
    limit = min(max(n, 1), 10000)
    articles = await rss.get_articles_paginated(
        user.user_id,
        feed_id=feed_id,
        starred_only=starred_only,
        limit=limit,
        offset=offset,
        older_than_ts=None,
        exclude_read=exclude_read,
        include_only_read=read_stream_only,
    )
    refs = []
    for a in articles:
        ref = article_to_greader_item_ref(a)
        if ref:
            refs.append(ref)
    skipped_gid = len(articles) - len(refs)
    logger.info(
        "GReader stream/items/ids user=%s stream=%s feed_id=%s starred_only=%s read_stream_only=%s "
        "exclude_read=%s offset=%s limit=%s client_ot=%s client_xt=%s client_it=%s db_rows=%s "
        "item_refs=%s skipped_null_greader_id=%s",
        user.user_id,
        sid,
        feed_id,
        starred_only,
        read_stream_only,
        exclude_read,
        offset,
        limit,
        ot,
        xt,
        it,
        len(articles),
        len(refs),
        skipped_gid,
    )
    next_c = str(offset + len(refs)) if len(refs) >= limit else None
    payload: Dict[str, Any] = {"itemRefs": refs}
    if next_c:
        payload["continuation"] = next_c
    return payload


@router.post("/reader/api/0/stream/items/contents")
async def greader_stream_items_contents(request: Request, output: str = "json"):
    user = await require_greader_user(request)
    body = await request.body()
    form = parse_form_pairs(body)
    raw_ids = form.get("i") or []
    gids: List[int] = []
    parse_failures = 0
    for raw in raw_ids:
        gid = parse_item_id(raw)
        if gid is not None:
            gids.append(gid)
        elif (raw or "").strip():
            parse_failures += 1
    rss = await get_rss_service()
    articles = await rss.get_articles_by_greader_ids(gids, user.user_id)
    fids = {a.feed_id for a in articles}
    fmap = await _feed_map(rss, fids)
    items = []
    for a in articles:
        it_json = article_to_greader_item(a, fmap.get(a.feed_id))
        if it_json:
            items.append(it_json)
    skipped_gid = len(articles) - len(items)
    logger.info(
        "GReader stream/items/contents user=%s raw_id_fields=%s parsed_greader_ids=%s "
        "parse_failures_nonempty=%s db_matched_rows=%s items_out=%s skipped_null_greader_id=%s",
        user.user_id,
        len(raw_ids),
        len(gids),
        parse_failures,
        len(articles),
        len(items),
        skipped_gid,
    )
    return {
        "id": GOOGLE_READING_LIST,
        "updated": int(time.time()),
        "items": items,
        "direction": "ltr",
    }


@router.post("/reader/api/0/edit-tag")
async def greader_edit_tag(request: Request):
    user = await require_greader_user(request)
    body = await request.body()
    logger.info(
        "GReader edit-tag user=%s body_len=%d",
        user.user_id,
        len(body),
    )
    form = parse_form_pairs(body)
    t_param = (form.get("T") or [None])[0]
    _verify_write_token(request, t_param)
    add_raw = (form.get("a") or [None])[0]
    rem_raw = (form.get("r") or [None])[0]
    add_tag = unquote(add_raw) if add_raw else None
    rem_tag = unquote(rem_raw) if rem_raw else None
    raw_ids = form.get("i") or []
    logger.info(
        "GReader edit-tag parsed add_tag=%s rem_tag=%s id_count=%d",
        add_tag,
        rem_tag,
        len(raw_ids),
    )
    rss = await get_rss_service()
    skipped_parse = 0
    skipped_missing = 0
    applied = 0
    for raw in raw_ids:
        gid = parse_item_id(raw)
        if gid is None:
            skipped_parse += 1
            continue
        art = await rss.get_article_by_greader_id(gid, user.user_id)
        if not art:
            skipped_missing += 1
            continue
        if add_tag == TAG_READ:
            await rss.mark_article_read(art.article_id, user.user_id)
            applied += 1
        elif rem_tag == TAG_READ:
            await rss.mark_article_unread(art.article_id, user.user_id)
            applied += 1
        elif add_tag == TAG_STARRED:
            await rss.set_article_starred(art.article_id, user.user_id, True)
            applied += 1
        elif rem_tag == TAG_STARRED:
            await rss.set_article_starred(art.article_id, user.user_id, False)
            applied += 1
    logger.info(
        "GReader edit-tag done user=%s applied=%d skipped_parse=%d skipped_missing_article=%d",
        user.user_id,
        applied,
        skipped_parse,
        skipped_missing,
    )
    return PlainTextResponse("OK", media_type="text/plain")


@router.post("/reader/api/0/mark-all-as-read")
async def greader_mark_all_as_read(request: Request):
    user = await require_greader_user(request)
    body = await request.body()
    logger.info(
        "GReader mark-all-as-read user=%s body_len=%d",
        user.user_id,
        len(body),
    )
    form = parse_form_pairs(body)
    t_param = (form.get("T") or [None])[0]
    _verify_write_token(request, t_param)
    s = (form.get("s") or [""])[0]
    ts_raw = (form.get("ts") or [None])[0]
    before = _parse_ts(ts_raw)
    rss = await get_rss_service()
    sid = normalize_stream_id(s)
    branch = "unknown"
    if sid == GOOGLE_READING_LIST or (
        "reading-list" in sid and "com.google" in sid
    ):
        await rss.mark_all_user_articles_read(user.user_id)
        branch = "all_reading_list"
    elif sid.startswith("feed/"):
        fid = sid[5:]
        await rss.mark_all_feed_read(fid, user.user_id, before_epoch=before)
        branch = f"feed:{fid}"
    else:
        feed_id, _, _ = stream_params(sid)
        if feed_id:
            await rss.mark_all_feed_read(feed_id, user.user_id, before_epoch=before)
            branch = f"stream_feed:{feed_id}"
        else:
            await rss.mark_all_user_articles_read(user.user_id)
            branch = "all_fallback"
    logger.info(
        "GReader mark-all-as-read done user=%s stream=%s ts=%s before_epoch=%s branch=%s",
        user.user_id,
        sid,
        ts_raw,
        before,
        branch,
    )
    return PlainTextResponse("OK", media_type="text/plain")


@router.post("/{catchall:path}", include_in_schema=False)
async def greader_unhandled_post(catchall: str, request: Request):
    """Log POSTs under /api/greader that did not match a defined route."""
    logger.warning(
        "GReader unhandled POST path=%s client=%s",
        catchall,
        request.client.host if request.client else None,
    )
    raise HTTPException(status_code=404, detail="Not found")
