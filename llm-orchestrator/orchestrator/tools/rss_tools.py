"""
RSS Tools - RSS feed management via backend gRPC
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)

def _truncate_url(url: str, max_len: int = 72) -> str:
    u = (url or "").strip()
    if len(u) <= max_len:
        return u
    return u[: max_len - 3] + "..."


def _format_feeds_for_llm(feeds: List[Dict[str, Any]], total_count: Optional[int] = None) -> str:
    """Group feeds by category (same as Documents RSS sidebar folders) for agent-friendly scanning."""
    if not feeds:
        return "No feeds found."
    count = total_count if total_count is not None else len(feeds)
    parts = [
        f"RSS feeds ({count} total). Each feed's category matches its folder in the Documents RSS sidebar.",
        "Use feed_id with get_rss_articles for headlines; use search_rss for keyword search; use list_starred_rss_articles_tool for all starred items across feeds.",
    ]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in feeds:
        if not isinstance(f, dict):
            continue
        cat = (f.get("category") or "").strip() or "Uncategorized"
        grouped[cat].append(f)
    category_order = sorted(
        grouped.keys(),
        key=lambda k: (k == "Uncategorized", k.lower()),
    )
    for cat in category_order:
        parts.append(f"--- {cat} ---")
        bucket = sorted(
            grouped[cat],
            key=lambda x: ((x.get("feed_name") or "") or "").lower(),
        )
        for f in bucket:
            name = f.get("feed_name", "?")
            fid = f.get("feed_id", "")
            url = _truncate_url(f.get("feed_url") or "")
            scope = "global" if f.get("is_global") else "user"
            ac = f.get("article_count")
            ac_s = f"{int(ac)} articles" if isinstance(ac, int) else "articles: ?"
            un = f.get("unread_count")
            un_s = f" | unread: {int(un)}" if isinstance(un, int) else ""
            polled = f.get("last_polled")
            polled_s = f" | last_polled: {polled}" if polled else ""
            parts.append(f"  • {name} | {ac_s}{un_s} | scope={scope} | id={fid}{polled_s}")
            if url:
                parts.append(f"      url: {url}")
    return "\n".join(parts)


def _format_rss_result(data: Dict[str, Any], default_msg: str = "Done.") -> str:
    """Format RSS result dict into a readable string for the LLM."""
    if not data:
        return default_msg
    if not data.get("success", True):
        return data.get("error", "Operation failed.")
    parts = []
    if "message" in data and data["message"]:
        parts.append(data["message"])
    if "feed_id" in data and data["feed_id"]:
        parts.append(f"Feed ID: {data['feed_id']}")
    if "task_id" in data and data["task_id"]:
        parts.append(f"Task ID: {data['task_id']}")
    if "feeds" in data:
        feeds = data["feeds"]
        if not isinstance(feeds, list):
            feeds = []
        count = data.get("count", len(feeds))
        parts.append(_format_feeds_for_llm(feeds, total_count=count))
    return "\n".join(parts) if parts else default_msg


class AddRssFeedInputs(BaseModel):
    feed_url: str = Field(description="URL of the RSS feed")


class RssOutputs(BaseModel):
    """Legacy minimal outputs; prefer specific output models below."""
    formatted: str = Field(description="Human-readable summary")
    success: bool = True
    error: Optional[str] = None


class AddRssFeedOutputs(BaseModel):
    """Outputs for add_rss_feed_tool."""
    success: bool = Field(description="Whether the feed was added")
    feed_id: Optional[str] = Field(default=None, description="Feed ID if created")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class RefreshRssFeedOutputs(BaseModel):
    """Outputs for refresh_rss_feed_tool."""
    success: bool = Field(description="Whether the refresh was triggered")
    new_items: List[Dict[str, Any]] = Field(default_factory=list, description="New items fetched (if returned by backend)")
    count: int = Field(default=0, description="Number of new items or count from backend")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class ListRssFeedsInputs(BaseModel):
    scope: str = Field(
        default="user",
        description="user (my feeds) or global (shared feeds); call both to see all sources",
    )


class ListRssFeedsOutputs(BaseModel):
    """Outputs for list_rss_feeds_tool."""
    feeds: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Feed dicts: feed_id, feed_name, feed_url, category (UI folder), is_global, article_count, unread_count, last_polled",
    )
    count: int = Field(description="Number of feeds returned")
    success: bool = Field(description="Whether the call succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def add_rss_feed_tool(
    feed_url: str = "",
    user_id: str = "system",
    feed_name: str = "",
    category: str = "",
    is_global: bool = False,
) -> Dict[str, Any]:
    """Add an RSS feed. Returns structured dict with formatted."""
    try:
        if not feed_url:
            msg = "Error: feed_url is required."
            return {"success": False, "error": msg, "formatted": msg}
        logger.info("add_rss_feed: url=%s name=%s", feed_url[:80], feed_name[:50] if feed_name else "")
        client = await get_backend_tool_client()
        result = await client.add_rss_feed(
            user_id=user_id,
            feed_url=feed_url,
            feed_name=feed_name,
            category=category,
            is_global=is_global,
        )
        out = dict(result) if isinstance(result, dict) else {}
        out["formatted"] = _format_rss_result(result, "Feed added.")
        out["success"] = out.get("success", "error" not in out)
        out["error"] = out.get("error") if not out.get("success", True) else None
        out.setdefault("feed_id", None)
        return out
    except Exception as e:
        logger.error("add_rss_feed_tool error: %s", e)
        return {"success": False, "feed_id": None, "error": str(e), "formatted": f"Error: {str(e)}"}


async def list_rss_feeds_tool(
    user_id: str = "system",
    scope: str = "user",
) -> Dict[str, Any]:
    """List RSS feeds. Returns structured dict with feeds, count, success, formatted."""
    try:
        logger.info("list_rss_feeds: scope=%s", scope)
        client = await get_backend_tool_client()
        result = await client.list_rss_feeds(user_id=user_id, scope=scope)
        if not isinstance(result, dict):
            return {"feeds": [], "count": 0, "success": False, "error": "Invalid response", "formatted": "No feeds found."}
        feeds = result.get("feeds", [])
        count = result.get("count", len(feeds))
        success = result.get("success", True) and "error" not in result
        error = result.get("error") if not success else None
        formatted = _format_rss_result(result, "No feeds found.")
        return {"feeds": feeds, "count": count, "success": success, "error": error, "formatted": formatted}
    except Exception as e:
        logger.error("list_rss_feeds_tool error: %s", e)
        err = str(e)
        return {"feeds": [], "count": 0, "success": False, "error": err, "formatted": f"Error: {err}"}


async def refresh_rss_feed_tool(
    user_id: str = "system",
    feed_name: str = "",
    feed_id: str = "",
) -> Dict[str, Any]:
    """Trigger a refresh for an RSS feed. Returns structured dict with formatted."""
    try:
        if not feed_name and not feed_id:
            msg = "Error: feed_name or feed_id is required."
            return {"success": False, "new_items": [], "count": 0, "error": msg, "formatted": msg}
        logger.info("refresh_rss_feed: name=%s id=%s", feed_name[:50] if feed_name else "", feed_id[:50] if feed_id else "")
        client = await get_backend_tool_client()
        result = await client.refresh_rss_feed(
            user_id=user_id,
            feed_name=feed_name,
            feed_id=feed_id,
        )
        out = dict(result) if isinstance(result, dict) else {}
        out["formatted"] = _format_rss_result(result, "Refresh triggered.")
        out["success"] = out.get("success", "error" not in out)
        out["error"] = out.get("error") if not out.get("success", True) else None
        out.setdefault("new_items", out.get("items", []))
        out.setdefault("count", len(out.get("new_items", [])))
        return out
    except Exception as e:
        logger.error("refresh_rss_feed_tool error: %s", e)
        return {"success": False, "new_items": [], "count": 0, "error": str(e), "formatted": f"Error: {str(e)}"}


class RefreshRssFeedInputs(BaseModel):
    feed_name: str = Field(default="", description="Display name of the feed")
    feed_id: str = Field(default="", description="Feed ID")


# ----- Get RSS Articles -----

class GetRssArticlesInputs(BaseModel):
    feed_id: str = Field(description="Feed ID (from list_rss_feeds_tool)")


class GetRssArticlesParams(BaseModel):
    limit: int = Field(default=20, description="Max articles to return")
    unread_only: bool = Field(
        default=False,
        description="If true, only articles not marked read (for current user)",
    )
    starred_only: bool = Field(
        default=False,
        description="If true, only starred articles; combine with unread_only for AND semantics",
    )


def _format_article_digest_lines(articles: List[Dict[str, Any]]) -> List[str]:
    """Short per-article lines for LLM formatted output."""
    lines: List[str] = []
    for i, a in enumerate(articles, 1):
        if not isinstance(a, dict):
            continue
        title = ((a.get("title") or "") or "")[:120]
        aid = a.get("article_id") or ""
        pub = a.get("published_at") or a.get("created_at") or ""
        state = []
        if a.get("is_read"):
            state.append("read")
        else:
            state.append("unread")
        if a.get("is_starred"):
            state.append("starred")
        if a.get("is_imported"):
            state.append("imported")
        fid = a.get("feed_id") or ""
        fn = (a.get("feed_name") or "")[:60]
        line1 = f"  {i}. {title}"
        line2 = f"      id={aid} | {', '.join(state)} | published={pub}"
        line3_parts = []
        if fid:
            line3_parts.append(f"feed_id={fid}")
        if fn:
            line3_parts.append(f"feed={fn}")
        lines.append(line1)
        lines.append(line2)
        if line3_parts:
            lines.append(f"      {' | '.join(line3_parts)}")
    return lines


class GetRssArticlesOutputs(BaseModel):
    articles: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Articles: article_id, title, content, url, published_at, feed_id, feed_name, is_read, is_starred, is_imported, created_at",
    )
    count: int = Field(description="Number of articles returned")
    feed_id: str = Field(description="Feed ID that was queried")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


# ----- Search RSS -----

class SearchRssInputs(BaseModel):
    query: str = Field(
        description="Search query across RSS article content and feed names"
    )


class SearchRssParams(BaseModel):
    limit: int = Field(default=20, description="Max articles to return")
    unread_only: bool = Field(default=False, description="Only unread articles")
    starred_only: bool = Field(default=False, description="Only starred articles")


class SearchRssOutputs(BaseModel):
    articles: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Matching articles: article_id, title, content, url, published_at, feed_id, feed_name, is_read, is_starred, is_imported, created_at",
    )
    count: int = Field(description="Number of articles found")
    query_used: str = Field(description="Query that was executed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class ListStarredRssArticlesInputs(BaseModel):
    pass


class ListStarredRssArticlesParams(BaseModel):
    limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Max starred articles to return (server caps at 500)",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Skip this many starred articles (newest first) for pagination",
    )


class ListStarredRssArticlesOutputs(BaseModel):
    articles: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Starred articles: article_id, title, content, url, published_at, feed_id, feed_name, is_read, is_starred, is_imported, created_at",
    )
    count: int = Field(description="Number of articles returned in this page")
    limit: int = Field(description="Limit requested")
    offset: int = Field(description="Offset requested")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def get_rss_articles_tool(
    feed_id: str,
    user_id: str = "system",
    limit: int = 20,
    unread_only: bool = False,
    starred_only: bool = False,
) -> Dict[str, Any]:
    """Retrieve articles from a specific RSS feed. Use list_rss_feeds_tool to get feed_id."""
    try:
        if not feed_id or not feed_id.strip():
            return {
                "articles": [],
                "count": 0,
                "feed_id": "",
                "formatted": "Error: feed_id is required.",
            }
        logger.info(
            "get_rss_articles: feed_id=%s limit=%s unread_only=%s starred_only=%s",
            feed_id[:32],
            limit,
            unread_only,
            starred_only,
        )
        client = await get_backend_tool_client()
        result = await client.get_rss_articles(
            feed_id=feed_id.strip(),
            user_id=user_id,
            limit=limit,
            unread_only=unread_only,
            starred_only=starred_only,
        )
        articles = result.get("articles", [])
        err = result.get("error")
        if err:
            return {
                "articles": [],
                "count": 0,
                "feed_id": feed_id,
                "formatted": f"Error: {err}",
            }
        filter_note = []
        if unread_only:
            filter_note.append("unread_only")
        if starred_only:
            filter_note.append("starred_only")
        fn = f" ({', '.join(filter_note)})" if filter_note else ""
        lines = [f"Feed {feed_id}: {len(articles)} article(s){fn}."]
        lines.extend(_format_article_digest_lines(articles))
        return {
            "articles": articles,
            "count": len(articles),
            "feed_id": feed_id,
            "formatted": "\n".join(lines),
        }
    except Exception as e:
        logger.error("get_rss_articles_tool error: %s", e)
        return {
            "articles": [],
            "count": 0,
            "feed_id": feed_id,
            "formatted": f"Error: {str(e)}",
        }


async def search_rss_tool(
    query: str,
    user_id: str = "system",
    limit: int = 20,
    unread_only: bool = False,
    starred_only: bool = False,
) -> Dict[str, Any]:
    """Search across all RSS article titles and content. Use after listing feeds to find relevant articles."""
    try:
        if not query or not query.strip():
            return {
                "articles": [],
                "count": 0,
                "query_used": "",
                "formatted": "Error: query is required.",
            }
        logger.info(
            "search_rss: query=%s limit=%s unread_only=%s starred_only=%s",
            query[:80],
            limit,
            unread_only,
            starred_only,
        )
        client = await get_backend_tool_client()
        result = await client.search_rss(
            query=query.strip(),
            user_id=user_id,
            limit=limit,
            unread_only=unread_only,
            starred_only=starred_only,
        )
        articles = result.get("articles", [])
        err = result.get("error")
        query_used = result.get("query_used", query)
        if err:
            return {
                "articles": [],
                "count": 0,
                "query_used": query_used,
                "formatted": f"Error: {err}",
            }
        lines = [f"Found {len(articles)} article(s) for '{query_used}'."]
        lines.extend(_format_article_digest_lines(articles))
        return {
            "articles": articles,
            "count": len(articles),
            "query_used": query_used,
            "formatted": "\n".join(lines),
        }
    except Exception as e:
        logger.error("search_rss_tool error: %s", e)
        return {
            "articles": [],
            "count": 0,
            "query_used": query,
            "formatted": f"Error: {str(e)}",
        }


async def list_starred_rss_articles_tool(
    user_id: str = "system",
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """List all starred RSS articles for the user across every feed (newest first)."""
    try:
        logger.info(
            "list_starred_rss_articles: limit=%s offset=%s",
            limit,
            offset,
        )
        client = await get_backend_tool_client()
        result = await client.list_starred_rss_articles(
            user_id=user_id,
            limit=limit,
            offset=offset,
        )
        articles = result.get("articles", [])
        err = result.get("error")
        lim = int(result.get("limit", limit))
        off = int(result.get("offset", offset))
        if err:
            return {
                "articles": [],
                "count": 0,
                "limit": lim,
                "offset": off,
                "formatted": f"Error: {err}",
            }
        lines = [
            f"Starred articles (all feeds): {len(articles)} in this page "
            f"(limit={lim}, offset={off}; use offset for older stars).",
        ]
        lines.extend(_format_article_digest_lines(articles))
        return {
            "articles": articles,
            "count": len(articles),
            "limit": lim,
            "offset": off,
            "formatted": "\n".join(lines),
        }
    except Exception as e:
        logger.error("list_starred_rss_articles_tool error: %s", e)
        return {
            "articles": [],
            "count": 0,
            "limit": limit,
            "offset": offset,
            "formatted": f"Error: {str(e)}",
        }


# ----- Delete RSS feed -----

class DeleteRssFeedInputs(BaseModel):
    feed_id: str = Field(default="", description="Feed ID from list_rss_feeds")
    feed_name: str = Field(default="", description="Feed display name if ID unknown")


class DeleteRssFeedOutputs(BaseModel):
    success: bool = Field(description="Whether deletion succeeded")
    feed_id: Optional[str] = Field(default=None, description="Deleted feed ID if known")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def delete_rss_feed_tool(
    user_id: str = "system",
    feed_id: str = "",
    feed_name: str = "",
) -> Dict[str, Any]:
    """Remove an RSS feed."""
    try:
        if not feed_id and not feed_name:
            msg = "Error: feed_id or feed_name is required."
            return {"success": False, "feed_id": None, "error": msg, "formatted": msg}
        client = await get_backend_tool_client()
        result = await client.delete_rss_feed(
            user_id=user_id,
            feed_name=feed_name or "",
            feed_id=feed_id or "",
        )
        if not result.get("success"):
            err = result.get("error", "Failed")
            return {"success": False, "feed_id": None, "error": err, "formatted": f"Error: {err}"}
        fid = result.get("feed_id") or feed_id or ""
        msg = result.get("message", "Feed deleted.")
        return {
            "success": True,
            "feed_id": fid or None,
            "error": None,
            "formatted": msg,
        }
    except Exception as e:
        logger.error("delete_rss_feed_tool error: %s", e)
        return {"success": False, "feed_id": None, "error": str(e), "formatted": f"Error: {str(e)}"}


# ----- Mark article read -----

class MarkArticleReadInputs(BaseModel):
    article_id: str = Field(description="RSS article_id from list/search")


class MarkArticleReadOutputs(BaseModel):
    success: bool = Field(description="Whether the update succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def mark_article_read_tool(
    article_id: str,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Mark one RSS article as read."""
    try:
        aid = (article_id or "").strip()
        if not aid:
            msg = "Error: article_id is required."
            return {"success": False, "error": msg, "formatted": msg}
        client = await get_backend_tool_client()
        result = await client.mark_article_read(user_id=user_id, article_id=aid)
        if not result.get("success"):
            err = result.get("error", "Failed")
            return {"success": False, "error": err, "formatted": f"Error: {err}"}
        return {
            "success": True,
            "error": None,
            "formatted": result.get("message", "Article marked as read."),
        }
    except Exception as e:
        logger.error("mark_article_read_tool error: %s", e)
        return {"success": False, "error": str(e), "formatted": f"Error: {str(e)}"}


class SetArticleStarredInputs(BaseModel):
    article_id: str = Field(description="RSS article_id from list/search")
    starred: bool = Field(description="True to star, false to unstar")


async def mark_article_unread_tool(
    article_id: str,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Mark one RSS article as unread."""
    try:
        aid = (article_id or "").strip()
        if not aid:
            msg = "Error: article_id is required."
            return {"success": False, "error": msg, "formatted": msg}
        client = await get_backend_tool_client()
        result = await client.mark_article_unread(user_id=user_id, article_id=aid)
        if not result.get("success"):
            err = result.get("error", "Failed")
            return {"success": False, "error": err, "formatted": f"Error: {err}"}
        return {
            "success": True,
            "error": None,
            "formatted": result.get("message", "Article marked as unread."),
        }
    except Exception as e:
        logger.error("mark_article_unread_tool error: %s", e)
        return {"success": False, "error": str(e), "formatted": f"Error: {str(e)}"}


async def set_article_starred_tool(
    article_id: str,
    starred: bool,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Set starred flag on an RSS article."""
    try:
        aid = (article_id or "").strip()
        if not aid:
            msg = "Error: article_id is required."
            return {"success": False, "error": msg, "formatted": msg}
        client = await get_backend_tool_client()
        result = await client.set_article_starred(
            user_id=user_id, article_id=aid, starred=starred
        )
        if not result.get("success"):
            err = result.get("error", "Failed")
            return {"success": False, "error": err, "formatted": f"Error: {err}"}
        return {
            "success": True,
            "error": None,
            "formatted": result.get("message", "Starred state updated."),
        }
    except Exception as e:
        logger.error("set_article_starred_tool error: %s", e)
        return {"success": False, "error": str(e), "formatted": f"Error: {str(e)}"}


# ----- Unread counts -----

class GetUnreadCountsInputs(BaseModel):
    pass


class GetUnreadCountsOutputs(BaseModel):
    counts: Dict[str, int] = Field(default_factory=dict, description="feed_id -> unread count")
    success: bool = Field(description="Whether the call succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def get_unread_counts_tool(user_id: str = "system") -> Dict[str, Any]:
    """Unread article counts per feed for the user."""
    try:
        client = await get_backend_tool_client()
        result = await client.get_unread_counts(user_id=user_id)
        if not result.get("success"):
            err = result.get("error", "Failed")
            return {"counts": {}, "success": False, "error": err, "formatted": f"Error: {err}"}
        counts = result.get("counts") or {}
        lines = [f"Unread counts for {len(counts)} feed(s):"]
        for fid, n in sorted(counts.items(), key=lambda kv: (kv[0] or "")):
            lines.append(f"  {fid}: {n}")
        return {
            "counts": counts,
            "success": True,
            "error": None,
            "formatted": "\n".join(lines) if counts else "No unread articles.",
        }
    except Exception as e:
        logger.error("get_unread_counts_tool error: %s", e)
        return {"counts": {}, "success": False, "error": str(e), "formatted": f"Error: {str(e)}"}


# ----- Toggle feed active -----

class ToggleFeedActiveInputs(BaseModel):
    feed_id: str = Field(description="Feed ID from list_rss_feeds")
    is_active: bool = Field(description="True to resume polling, False to pause")


class ToggleFeedActiveOutputs(BaseModel):
    success: bool = Field(description="Whether the update succeeded")
    feed_id: str = Field(default="", description="Feed ID affected")
    is_active: bool = Field(default=True, description="New active flag")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def toggle_feed_active_tool(
    feed_id: str,
    is_active: bool,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Pause or resume automatic polling for a feed."""
    try:
        fid = (feed_id or "").strip()
        if not fid:
            msg = "Error: feed_id is required."
            return {
                "success": False,
                "feed_id": "",
                "is_active": is_active,
                "error": msg,
                "formatted": msg,
            }
        client = await get_backend_tool_client()
        result = await client.toggle_feed_active(
            user_id=user_id, feed_id=fid, is_active=is_active
        )
        if not result.get("success"):
            err = result.get("error", "Failed")
            return {
                "success": False,
                "feed_id": fid,
                "is_active": is_active,
                "error": err,
                "formatted": f"Error: {err}",
            }
        state = "enabled" if result.get("is_active", is_active) else "paused"
        return {
            "success": True,
            "feed_id": result.get("feed_id", fid),
            "is_active": bool(result.get("is_active", is_active)),
            "error": None,
            "formatted": f"Feed polling {state}: {result.get('feed_id', fid)}",
        }
    except Exception as e:
        logger.error("toggle_feed_active_tool error: %s", e)
        return {
            "success": False,
            "feed_id": feed_id or "",
            "is_active": is_active,
            "error": str(e),
            "formatted": f"Error: {str(e)}",
        }


register_action(name="add_rss_feed", category="rss", description="Add an RSS feed", inputs_model=AddRssFeedInputs, outputs_model=AddRssFeedOutputs, tool_function=add_rss_feed_tool)
register_action(
    name="list_rss_articles",
    category="rss",
    description="Retrieve articles from a specific RSS feed (list_rss_feeds for feed_id). Returns is_read, is_starred, is_imported, published_at, feed_id; optional unread_only/starred_only filters.",
    short_description="Retrieve articles from an RSS feed",
    inputs_model=GetRssArticlesInputs,
    params_model=GetRssArticlesParams,
    outputs_model=GetRssArticlesOutputs,
    tool_function=get_rss_articles_tool,
)
register_action(
    name="search_rss",
    category="rss",
    description="Search RSS article titles and content; optional unread_only/starred_only; results include feed_id, feed_name, read/star/import flags",
    inputs_model=SearchRssInputs,
    params_model=SearchRssParams,
    outputs_model=SearchRssOutputs,
    tool_function=search_rss_tool,
)
register_action(
    name="list_starred_rss_articles",
    category="rss",
    description="List all starred RSS articles across every feed for the user (newest first); use limit/offset to paginate",
    short_description="List starred RSS articles across all feeds",
    inputs_model=ListStarredRssArticlesInputs,
    params_model=ListStarredRssArticlesParams,
    outputs_model=ListStarredRssArticlesOutputs,
    tool_function=list_starred_rss_articles_tool,
)
register_action(
    name="list_rss_feeds",
    category="rss",
    description="List RSS feeds by category (Documents RSS sidebar). Includes feed_id, article_count, unread_count, scope, last_polled.",
    inputs_model=ListRssFeedsInputs,
    outputs_model=ListRssFeedsOutputs,
    tool_function=list_rss_feeds_tool,
)
register_action(
    name="refresh_rss_feed",
    category="rss",
    description="Refresh an RSS feed",
    inputs_model=RefreshRssFeedInputs,
    outputs_model=RefreshRssFeedOutputs,
    tool_function=refresh_rss_feed_tool,
)
register_action(
    name="delete_rss_feed",
    category="rss",
    description="Delete an RSS feed by feed_id or feed_name",
    inputs_model=DeleteRssFeedInputs,
    outputs_model=DeleteRssFeedOutputs,
    tool_function=delete_rss_feed_tool,
)
register_action(
    name="mark_article_read",
    category="rss",
    description="Mark an RSS article as read",
    inputs_model=MarkArticleReadInputs,
    outputs_model=MarkArticleReadOutputs,
    tool_function=mark_article_read_tool,
)
register_action(
    name="mark_article_unread",
    category="rss",
    description="Mark an RSS article as unread",
    inputs_model=MarkArticleReadInputs,
    outputs_model=MarkArticleReadOutputs,
    tool_function=mark_article_unread_tool,
)
register_action(
    name="set_article_starred",
    category="rss",
    description="Star or unstar an RSS article",
    inputs_model=SetArticleStarredInputs,
    outputs_model=MarkArticleReadOutputs,
    tool_function=set_article_starred_tool,
)
register_action(
    name="get_unread_counts",
    category="rss",
    description="Get per-feed unread RSS article counts",
    inputs_model=GetUnreadCountsInputs,
    outputs_model=GetUnreadCountsOutputs,
    tool_function=get_unread_counts_tool,
)
register_action(
    name="toggle_feed_active",
    category="rss",
    description="Pause or resume automatic polling for an RSS feed",
    inputs_model=ToggleFeedActiveInputs,
    outputs_model=ToggleFeedActiveOutputs,
    tool_function=toggle_feed_active_tool,
)
