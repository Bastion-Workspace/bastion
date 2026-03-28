"""Google Reader API helpers (parsing, item JSON)."""

from __future__ import annotations

import calendar
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote

from tools_service.models.rss_models import RSSArticle, RSSFeed

GOOGLE_READING_LIST = "user/-/state/com.google/reading-list"
GOOGLE_STARRED = "user/-/state/com.google/starred"
GOOGLE_READ = "user/-/state/com.google/read"
TAG_READ = "user/-/state/com.google/read"
TAG_STARRED = "user/-/state/com.google/starred"
TAG_READING_LIST = "user/-/state/com.google/reading-list"


def rss_category_canonical_id_segment(raw: str) -> str:
    """Lowercase segment for user/-/label/... ids so mixed DB casing maps to one folder."""
    s = (raw or "").strip().lower()
    return s if s else "uncategorized"


def rss_category_display_label(raw: str) -> str:
    """
    Folder title aligned with FileTreeSidebar.groupRssFeedsByCategory (title case per word).
    """
    s = (raw or "").strip()
    if not s:
        return "Uncategorized"
    key = s.lower()
    if key == "uncategorized":
        return "Uncategorized"
    parts = re.split(r"[\s_]+", s)
    return " ".join(
        (p[0].upper() + p[1:].lower()) if p else "" for p in parts if p
    )


def greader_user_label_stream_id(raw: str) -> str:
    """Stable stream id for a user category label."""
    return f"user/-/label/{rss_category_canonical_id_segment(raw)}"


def parse_item_id(raw: str) -> Optional[int]:
    """Parse GReader item id (long hex, short hex, or decimal) to integer greader_id."""
    if not raw:
        return None
    raw = raw.strip()
    prefix = "tag:google.com,2005:reader/item/"
    if raw.startswith(prefix):
        hex_id = raw[len(prefix) :]
        try:
            return int(hex_id, 16)
        except ValueError:
            return None
    if len(raw) == 16 and all(c in "0123456789abcdefABCDEF" for c in raw):
        try:
            return int(raw, 16)
        except ValueError:
            return None
    try:
        return int(raw, 10)
    except ValueError:
        return None


def format_long_item_id(greader_id: int) -> str:
    return f"tag:google.com,2005:reader/item/{greader_id:016x}"


def normalize_stream_id(stream_id: str) -> str:
    return unquote(stream_id or "").strip()


def _epoch_seconds(article: RSSArticle) -> int:
    dt = article.published_date or article.created_at
    if dt is None:
        return int(datetime.now(timezone.utc).timestamp())
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return calendar.timegm(dt.utctimetuple())
        return int(dt.timestamp())
    return int(datetime.now(timezone.utc).timestamp())


def article_to_greader_item_ref(article: RSSArticle) -> Optional[Dict[str, Any]]:
    """
    One entry for /stream/items/ids itemRefs (see google-reader-api ApiStreamItemsIds).
    """
    if article.greader_id is None:
        return None
    gid = int(article.greader_id)
    ts = _epoch_seconds(article)
    ts_us = str(ts * 1_000_000)
    return {
        "id": str(gid),
        "timestampUsec": ts_us,
        "directStreamIds": [f"feed/{article.feed_id}"],
    }


def article_to_greader_item(
    article: RSSArticle,
    feed: Optional[RSSFeed],
) -> Optional[Dict[str, Any]]:
    """Single item object for stream/contents JSON."""
    if article.greader_id is None:
        return None
    gid = int(article.greader_id)
    ts = _epoch_seconds(article)
    ts_ms = str(ts * 1000)
    ts_us = str(ts * 1_000_000)
    link = article.link or ""
    title = article.title or ""
    body = article.full_content_html or article.description or article.full_content or ""
    categories: List[str] = [TAG_READING_LIST]
    if article.is_read:
        categories.append(TAG_READ)
    if getattr(article, "is_starred", False):
        categories.append(TAG_STARRED)
    feed_title = feed.feed_name if feed else article.feed_id
    feed_url = feed.feed_url if feed else ""
    return {
        "id": format_long_item_id(gid),
        "title": title,
        "published": ts,
        "crawlTimeMsec": ts_ms,
        "timestampUsec": ts_us,
        "updated": ts,
        "alternate": [{"href": link, "type": "text/html"}],
        "canonical": [{"href": link, "type": "text/html"}],
        "summary": {"content": body, "direction": "ltr"},
        "content": {"content": body, "direction": "ltr"},
        "categories": categories,
        "origin": {
            "streamId": f"feed/{article.feed_id}",
            "title": feed_title,
            "htmlUrl": feed_url or link,
        },
    }


def parse_form_pairs(body: bytes) -> Dict[str, List[str]]:
    """application/x-www-form-urlencoded with repeated keys (e.g. multiple i=)."""
    if not body:
        return {}
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        return {}
    return parse_qs(text, keep_blank_values=True)


def stream_params(
    stream_id: str,
) -> Tuple[Optional[str], bool, bool]:
    """
    Returns (feed_id or None for all-user, starred_only, read_stream_only).

    When read_stream_only is True (s=.../com.google/read), only articles with
    is_read=true must be returned. Otherwise mobile clients mark every id as read.
    """
    sid = normalize_stream_id(stream_id)
    if sid.startswith("feed/"):
        return (sid[5:], False, False)
    if sid == GOOGLE_STARRED or sid.endswith("/starred"):
        return (None, True, False)
    if sid == GOOGLE_READ or sid.endswith("/state/com.google/read"):
        return (None, False, True)
    if "reading-list" in sid or sid == GOOGLE_READING_LIST:
        return (None, False, False)
    return (None, False, False)


def system_tag_list() -> List[Dict[str, str]]:
    return [
        {"id": TAG_STARRED},
        {"id": TAG_READ},
        {"id": TAG_READING_LIST},
        {"id": "user/-/state/com.google/kept-unread"},
    ]
