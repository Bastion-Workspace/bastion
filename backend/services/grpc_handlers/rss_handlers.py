"""gRPC handlers for RSS operations."""

import logging
from typing import Any

import grpc
from protos import tool_service_pb2

logger = logging.getLogger(__name__)


def _rss_article_to_pb(art: Any, feed_name_fallback: str = "") -> tool_service_pb2.RSSArticle:
    """Map tools_service RSSArticle (or compatible) to gRPC RSSArticle."""
    content = (getattr(art, "description", None) or "")[:5000]
    full = getattr(art, "full_content", None) or ""
    if full and len(full) > len(content):
        content = full[:5000]
    fn = feed_name_fallback or (getattr(art, "feed_name", None) or "")
    pub = getattr(art, "published_date", None)
    pd = pub.isoformat() if pub else ""
    cr = getattr(art, "created_at", None)
    ca = cr.isoformat() if cr else ""
    return tool_service_pb2.RSSArticle(
        article_id=getattr(art, "article_id", "") or "",
        title=getattr(art, "title", "") or "",
        content=content,
        url=getattr(art, "link", "") or "",
        published_at=pd,
        feed_id=getattr(art, "feed_id", "") or "",
        feed_name=fn,
        is_read=bool(getattr(art, "is_read", False)),
        is_starred=bool(getattr(art, "is_starred", False)),
        is_imported=bool(getattr(art, "is_processed", False)),
        created_at=ca,
    )


class RssHandlersMixin:
    """Mixin providing RSS gRPC handlers.

    Mixed into ToolServiceImplementation; accesses self._get_search_service(),
    self._get_document_repo(), etc. via standard Python MRO.
    """

    # ===== RSS Operations =====
    
    async def SearchRSSFeeds(
        self,
        request: tool_service_pb2.RSSSearchRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.RSSSearchResponse:
        """Search RSS feeds and articles by query (title, description, content)."""
        try:
            logger.info(f"SearchRSSFeeds: user={request.user_id}, query={request.query[:80]}")
            from tools_service.services.rss_service import get_rss_service

            rss_service = await get_rss_service()
            limit = request.limit or 20
            articles = await rss_service.search_articles(
                user_id=request.user_id or "system",
                query=request.query or "",
                limit=limit,
                unread_only=bool(request.unread_only),
                starred_only=bool(request.starred_only),
            )
            response = tool_service_pb2.RSSSearchResponse()
            for art in articles:
                response.articles.append(_rss_article_to_pb(art))
            logger.info(f"SearchRSSFeeds: Found {len(response.articles)} articles")
            return response
        except Exception as e:
            logger.error(f"SearchRSSFeeds error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"RSS search failed: {str(e)}")

    async def GetRSSArticles(
        self,
        request: tool_service_pb2.RSSArticlesRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.RSSArticlesResponse:
        """Get articles from a specific RSS feed."""
        try:
            logger.info(f"GetRSSArticles: feed_id={request.feed_id}")
            from tools_service.services.rss_service import get_rss_service

            rss_service = await get_rss_service()
            feed = await rss_service.get_feed(request.feed_id)
            if not feed:
                return tool_service_pb2.RSSArticlesResponse()
            if feed.user_id is not None and feed.user_id != request.user_id:
                return tool_service_pb2.RSSArticlesResponse()
            limit = request.limit or 20
            uid = request.user_id or "system"
            unread_only = bool(request.unread_only)
            starred_only = bool(request.starred_only)
            if unread_only or starred_only:
                articles = await rss_service.get_feed_articles_filtered(
                    feed_id=request.feed_id,
                    user_id=uid,
                    limit=limit,
                    unread_only=unread_only,
                    starred_only=starred_only,
                )
            else:
                articles = await rss_service.get_feed_articles(
                    feed_id=request.feed_id,
                    user_id=uid,
                    limit=limit,
                )
            feed_name = feed.feed_name or ""
            response = tool_service_pb2.RSSArticlesResponse()
            for art in articles:
                response.articles.append(_rss_article_to_pb(art, feed_name_fallback=feed_name))
            logger.info(f"GetRSSArticles: Returned {len(response.articles)} articles")
            return response
        except Exception as e:
            logger.error(f"GetRSSArticles error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Get articles failed: {str(e)}")

    async def ListStarredRSSArticles(
        self,
        request: tool_service_pb2.ListStarredRSSArticlesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListStarredRSSArticlesResponse:
        """List starred RSS articles for the user across all feeds."""
        try:
            uid = (request.user_id or "").strip()
            if not uid:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "user_id is required"
                )
            from tools_service.services.rss_service import get_rss_service

            rss_service = await get_rss_service()
            limit = int(request.limit) if request.limit else 50
            limit = max(1, min(limit, 500))
            offset = max(0, int(request.offset))
            articles = await rss_service.get_starred_articles(
                user_id=uid, limit=limit, offset=offset
            )
            response = tool_service_pb2.ListStarredRSSArticlesResponse()
            for art in articles:
                response.articles.append(_rss_article_to_pb(art))
            logger.info(
                "ListStarredRSSArticles: user=%s limit=%s offset=%s count=%s",
                uid,
                limit,
                offset,
                len(response.articles),
            )
            return response
        except (grpc.RpcError, grpc._cython.cygrpc.AbortError):
            raise
        except Exception as e:
            logger.error("ListStarredRSSArticles error: %s", e)
            await context.abort(
                grpc.StatusCode.INTERNAL, f"List starred articles failed: {str(e)}"
            )
    
    # ===== RSS Management Operations =====
    
    async def AddRSSFeed(
        self,
        request: tool_service_pb2.AddRSSFeedRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.AddRSSFeedResponse:
        """Add a new RSS feed"""
        try:
            logger.info(f"AddRSSFeed: user={request.user_id}, url={request.feed_url}, is_global={request.is_global}")
            
            from services.auth_service import auth_service
            from tools_service.models.rss_models import RSSFeedCreate
            from tools_service.services.rss_service import get_rss_service
            
            rss_service = await get_rss_service()
            
            # Check permissions for global feeds
            if request.is_global:
                user_info = await auth_service.get_user_by_id(request.user_id)
                if not user_info or user_info.role != "admin":
                    return tool_service_pb2.AddRSSFeedResponse(
                        success=False,
                        error="Only admin users can add global RSS feeds"
                    )
            
            # Create RSS feed data
            feed_data = RSSFeedCreate(
                feed_url=request.feed_url,
                feed_name=request.feed_name,
                user_id=request.user_id if not request.is_global else None,  # None for global
                category=request.category or "general",
                tags=["rss", "imported"],
                check_interval=3600  # Default 1 hour
            )
            
            # Add the feed
            new_feed = await rss_service.create_feed(feed_data)
            
            logger.info(f"AddRSSFeed: Successfully added feed {new_feed.feed_id}")
            
            return tool_service_pb2.AddRSSFeedResponse(
                success=True,
                feed_id=new_feed.feed_id,
                feed_name=new_feed.feed_name,
                message=f"Successfully added {'global' if request.is_global else 'user'} RSS feed: {new_feed.feed_name}"
            )
            
        except Exception as e:
            logger.error(f"AddRSSFeed error: {e}")
            return tool_service_pb2.AddRSSFeedResponse(
                success=False,
                error=f"Failed to add RSS feed: {str(e)}"
            )
    
    async def ListRSSFeeds(
        self,
        request: tool_service_pb2.ListRSSFeedsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ListRSSFeedsResponse:
        """List RSS feeds"""
        try:
            logger.info(f"ListRSSFeeds: user={request.user_id}, scope={request.scope}")
            
            from services.auth_service import auth_service
            from tools_service.services.rss_service import get_rss_service
            
            rss_service = await get_rss_service()
            
            # Determine if user is admin for global feed access
            is_admin = False
            if request.scope == "global":
                user_info = await auth_service.get_user_by_id(request.user_id)
                is_admin = user_info and user_info.role == "admin"
            
            # Get feeds based on scope
            feeds = await rss_service.get_user_feeds(request.user_id, is_admin=is_admin)
            
            # Convert to proto response
            response = tool_service_pb2.ListRSSFeedsResponse(
                success=True,
                count=len(feeds)
            )
            counts_map = await rss_service.get_unread_count(request.user_id or "system")

            for feed in feeds:
                # Get article count for this feed
                from services.database_manager.database_helpers import fetch_value
                try:
                    article_count = await fetch_value(
                        "SELECT COUNT(*) FROM rss_articles WHERE feed_id = $1",
                        feed.feed_id
                    ) or 0
                except Exception:
                    article_count = 0
                last_chk = getattr(feed, "last_check", None) or getattr(
                    feed, "last_poll_date", None
                )
                last_polled_s = last_chk.isoformat() if last_chk else ""

                feed_details = tool_service_pb2.RSSFeedDetails(
                    feed_id=feed.feed_id,
                    feed_name=feed.feed_name,
                    feed_url=feed.feed_url,
                    category=feed.category or "general",
                    is_global=(feed.user_id is None),
                    last_polled=last_polled_s,
                    article_count=int(article_count),
                    unread_count=int(counts_map.get(feed.feed_id, 0)),
                )
                response.feeds.append(feed_details)
            
            logger.info(f"ListRSSFeeds: Found {len(feeds)} feeds")
            return response
            
        except Exception as e:
            logger.error(f"ListRSSFeeds error: {e}")
            return tool_service_pb2.ListRSSFeedsResponse(
                success=False,
                error=f"Failed to list RSS feeds: {str(e)}"
            )
    
    async def RefreshRSSFeed(
        self,
        request: tool_service_pb2.RefreshRSSFeedRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.RefreshRSSFeedResponse:
        """Refresh a specific RSS feed"""
        try:
            logger.info(f"RefreshRSSFeed: user={request.user_id}, feed_name={request.feed_name}, feed_id={request.feed_id}")
            
            from services.celery_tasks.rss_tasks import poll_rss_feeds_task
            from tools_service.services.rss_service import get_rss_service
            
            rss_service = await get_rss_service()
            
            # Find the feed by ID or name
            target_feed = None
            if request.feed_id:
                target_feed = await rss_service.get_feed(request.feed_id)
            else:
                # Find by name
                feeds = await rss_service.get_user_feeds(request.user_id, is_admin=True)
                for feed in feeds:
                    if feed.feed_name.lower() == request.feed_name.lower():
                        target_feed = feed
                        break
            
            if not target_feed:
                return tool_service_pb2.RefreshRSSFeedResponse(
                    success=False,
                    error=f"RSS feed '{request.feed_name or request.feed_id}' not found"
                )
            
            # Trigger refresh via Celery
            task = poll_rss_feeds_task.delay(
                user_id=request.user_id,
                feed_ids=[target_feed.feed_id],
                force_poll=True
            )
            
            logger.info(f"RefreshRSSFeed: Triggered refresh task {task.id} for feed {target_feed.feed_id}")
            
            return tool_service_pb2.RefreshRSSFeedResponse(
                success=True,
                feed_id=target_feed.feed_id,
                feed_name=target_feed.feed_name,
                task_id=task.id,
                message=f"Refresh initiated for RSS feed: {target_feed.feed_name}"
            )
            
        except Exception as e:
            logger.error(f"RefreshRSSFeed error: {e}")
            return tool_service_pb2.RefreshRSSFeedResponse(
                success=False,
                error=f"Failed to refresh RSS feed: {str(e)}"
            )
    
    async def DeleteRSSFeed(
        self,
        request: tool_service_pb2.DeleteRSSFeedRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DeleteRSSFeedResponse:
        """Delete an RSS feed"""
        try:
            logger.info(f"DeleteRSSFeed: user={request.user_id}, feed_name={request.feed_name}, feed_id={request.feed_id}")
            
            from tools_service.services.rss_service import get_rss_service
            
            rss_service = await get_rss_service()
            
            # Find the feed by ID or name
            target_feed = None
            if request.feed_id:
                target_feed = await rss_service.get_feed(request.feed_id)
            else:
                # Find by name
                feeds = await rss_service.get_user_feeds(request.user_id, is_admin=True)
                for feed in feeds:
                    if feed.feed_name.lower() == request.feed_name.lower():
                        target_feed = feed
                        break
            
            if not target_feed:
                return tool_service_pb2.DeleteRSSFeedResponse(
                    success=False,
                    error=f"RSS feed '{request.feed_name or request.feed_id}' not found"
                )
            
            # Check permission - only feed owner or admin can delete
            # For now, we trust the user_id passed from orchestrator
            
            # Delete the feed
            await rss_service.delete_feed(target_feed.feed_id, request.user_id, is_admin=False)
            
            logger.info(f"DeleteRSSFeed: Successfully deleted feed {target_feed.feed_id}")
            
            return tool_service_pb2.DeleteRSSFeedResponse(
                success=True,
                feed_id=target_feed.feed_id,
                message=f"Successfully deleted RSS feed: {target_feed.feed_name}"
            )
            
        except Exception as e:
            logger.error(f"DeleteRSSFeed error: {e}")
            return tool_service_pb2.DeleteRSSFeedResponse(
                success=False,
                error=f"Failed to delete RSS feed: {str(e)}"
            )

    async def MarkArticleRead(
        self,
        request: tool_service_pb2.MarkArticleReadRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.MarkArticleReadResponse:
        """Mark an RSS article as read for the requesting user."""
        try:
            uid = request.user_id or "system"
            aid = (request.article_id or "").strip()
            if not aid:
                return tool_service_pb2.MarkArticleReadResponse(
                    success=False,
                    message="",
                    error="article_id is required",
                )
            from tools_service.services.rss_service import get_rss_service

            rss_service = await get_rss_service()
            ok = await rss_service.mark_article_read(aid, uid)
            if not ok:
                return tool_service_pb2.MarkArticleReadResponse(
                    success=False,
                    message="",
                    error="Failed to mark article read",
                )
            return tool_service_pb2.MarkArticleReadResponse(
                success=True,
                message="Article marked as read",
            )
        except Exception as e:
            logger.error("MarkArticleRead error: %s", e)
            return tool_service_pb2.MarkArticleReadResponse(
                success=False,
                message="",
                error=str(e),
            )

    async def MarkArticleUnread(
        self,
        request: tool_service_pb2.MarkArticleUnreadRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.MarkArticleUnreadResponse:
        """Mark an RSS article as unread for the requesting user."""
        try:
            uid = request.user_id or "system"
            aid = (request.article_id or "").strip()
            if not aid:
                return tool_service_pb2.MarkArticleUnreadResponse(
                    success=False,
                    message="",
                    error="article_id is required",
                )
            from tools_service.services.rss_service import get_rss_service

            rss_service = await get_rss_service()
            ok = await rss_service.mark_article_unread(aid, uid)
            if not ok:
                return tool_service_pb2.MarkArticleUnreadResponse(
                    success=False,
                    message="",
                    error="Failed to mark article unread",
                )
            return tool_service_pb2.MarkArticleUnreadResponse(
                success=True,
                message="Article marked as unread",
            )
        except Exception as e:
            logger.error("MarkArticleUnread error: %s", e)
            return tool_service_pb2.MarkArticleUnreadResponse(
                success=False,
                message="",
                error=str(e),
            )

    async def SetArticleStarred(
        self,
        request: tool_service_pb2.SetArticleStarredRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SetArticleStarredResponse:
        """Set RSS article starred flag for the requesting user."""
        try:
            uid = request.user_id or "system"
            aid = (request.article_id or "").strip()
            if not aid:
                return tool_service_pb2.SetArticleStarredResponse(
                    success=False,
                    message="",
                    error="article_id is required",
                )
            from tools_service.services.rss_service import get_rss_service

            rss_service = await get_rss_service()
            ok = await rss_service.set_article_starred(aid, uid, request.starred)
            if not ok:
                return tool_service_pb2.SetArticleStarredResponse(
                    success=False,
                    message="",
                    error="Failed to update starred state",
                )
            state = "starred" if request.starred else "unstarred"
            return tool_service_pb2.SetArticleStarredResponse(
                success=True,
                message=f"Article {state}",
            )
        except Exception as e:
            logger.error("SetArticleStarred error: %s", e)
            return tool_service_pb2.SetArticleStarredResponse(
                success=False,
                message="",
                error=str(e),
            )

    async def GetUnreadCounts(
        self,
        request: tool_service_pb2.GetUnreadCountsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.GetUnreadCountsResponse:
        """Per-feed unread article counts for the user."""
        try:
            uid = request.user_id or "system"
            from tools_service.services.rss_service import get_rss_service

            rss_service = await get_rss_service()
            counts_map = await rss_service.get_unread_count(uid)
            response = tool_service_pb2.GetUnreadCountsResponse(success=True)
            for feed_id, cnt in (counts_map or {}).items():
                response.counts.append(
                    tool_service_pb2.UnreadCountEntry(
                        feed_id=feed_id,
                        count=int(cnt),
                    )
                )
            return response
        except Exception as e:
            logger.error("GetUnreadCounts error: %s", e)
            return tool_service_pb2.GetUnreadCountsResponse(
                success=False,
                error=str(e),
            )

    async def ToggleFeedActive(
        self,
        request: tool_service_pb2.ToggleFeedActiveRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ToggleFeedActiveResponse:
        """Enable or disable polling for an RSS feed."""
        try:
            uid = request.user_id or "system"
            fid = (request.feed_id or "").strip()
            if not fid:
                return tool_service_pb2.ToggleFeedActiveResponse(
                    success=False,
                    feed_id="",
                    is_active=request.is_active,
                    message="",
                    error="feed_id is required",
                )
            from services.auth_service import auth_service
            from tools_service.services.rss_service import get_rss_service

            rss_service = await get_rss_service()
            user_info = await auth_service.get_user_by_id(uid)
            is_admin = bool(user_info and user_info.role == "admin")
            ok = await rss_service.toggle_feed_active(
                fid, uid, request.is_active, is_admin=is_admin
            )
            if not ok:
                return tool_service_pb2.ToggleFeedActiveResponse(
                    success=False,
                    feed_id=fid,
                    is_active=request.is_active,
                    message="",
                    error="Not allowed or feed not found",
                )
            return tool_service_pb2.ToggleFeedActiveResponse(
                success=True,
                feed_id=fid,
                is_active=request.is_active,
                message="Feed active state updated",
            )
        except Exception as e:
            logger.error("ToggleFeedActive error: %s", e)
            return tool_service_pb2.ToggleFeedActiveResponse(
                success=False,
                feed_id=request.feed_id or "",
                is_active=request.is_active,
                message="",
                error=str(e),
            )

