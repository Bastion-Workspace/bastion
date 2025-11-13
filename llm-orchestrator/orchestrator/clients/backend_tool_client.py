"""
Backend Tool Client for LLM Orchestrator
Connects to backend Tool Service for web search and other data access operations
"""

import logging
import os
import json
from typing import Optional, List, Dict, Any
import grpc

from protos import tool_service_pb2, tool_service_pb2_grpc

logger = logging.getLogger(__name__)


class BackendToolClient:
    """Client for backend gRPC Tool Service - web search and data access"""
    
    def __init__(self):
        self.host = os.getenv("BACKEND_TOOL_SERVICE_HOST", "backend")
        self.port = os.getenv("BACKEND_TOOL_SERVICE_PORT", "50052")
        self.address = f"{self.host}:{self.port}"
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[tool_service_pb2_grpc.ToolServiceStub] = None
        logger.info(f"BackendToolClient configured for {self.address}")
    
    async def _ensure_connected(self):
        """Ensure gRPC channel is connected"""
        if self._channel is None:
            self._channel = grpc.aio.insecure_channel(self.address)
            self._stub = tool_service_pb2_grpc.ToolServiceStub(self._channel)
            logger.info(f"✅ Connected to backend tool service at {self.address}")
    
    async def search_web(
        self,
        query: str,
        max_results: int = 10,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Search the web using backend tool service
        
        Returns dict with search results
        """
        try:
            await self._ensure_connected()
            
            # Build request
            request = tool_service_pb2.WebSearchRequest(
                query=query,
                max_results=max_results,
                user_id=user_id
            )
            
            logger.info(f"🔍 Calling backend SearchWeb: query={query[:100]}...")
            
            # Call gRPC method
            response = await self._stub.SearchWeb(request)
            
            # Convert proto response to dict
            results = []
            for result in response.results:
                results.append({
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "relevance_score": result.relevance_score
                })
            
            logger.info(f"✅ Found {len(results)} search results")
            
            return {
                "success": True,
                "results": results,
                "count": len(results)
            }
            
        except Exception as e:
            logger.error(f"❌ SearchWeb gRPC call failed: {e}")
            return {
                "success": False,
                "error": f"Web search failed: {str(e)}",
                "results": []
            }
    
    async def search_documents(
        self,
        query: str,
        limit: int = 20,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """Search documents using backend tool service"""
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.SearchRequest(
                user_id=user_id,
                query=query,
                limit=limit
            )
            
            logger.info(f"📄 Calling backend SearchDocuments: query={query[:100]}...")
            
            response = await self._stub.SearchDocuments(request)
            
            results = []
            for result in response.results:
                results.append({
                    "document_id": result.document_id,
                    "title": result.title,
                    "filename": result.filename,
                    "content_preview": result.content_preview,
                    "relevance_score": result.relevance_score,
                    "metadata": dict(result.metadata)
                })
            
            logger.info(f"✅ Found {len(results)} documents")
            
            return {
                "success": True,
                "results": results,
                "total_count": response.total_count
            }
            
        except Exception as e:
            logger.error(f"❌ SearchDocuments gRPC call failed: {e}")
            return {
                "success": False,
                "error": f"Document search failed: {str(e)}",
                "results": []
            }
    
    async def add_rss_feed(
        self,
        user_id: str,
        feed_url: str,
        feed_name: str,
        category: str,
        is_global: bool = False
    ) -> Dict[str, Any]:
        """Add RSS feed via backend tool service"""
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.AddRSSFeedRequest(
                user_id=user_id,
                feed_url=feed_url,
                feed_name=feed_name,
                category=category,
                is_global=is_global
            )
            
            logger.info(f"📰 Calling backend AddRSSFeed: url={feed_url}, name={feed_name}...")
            
            response = await self._stub.AddRSSFeed(request)
            
            return {
                "success": response.success,
                "feed_id": response.feed_id,
                "feed_name": response.feed_name,
                "message": response.message,
                "error": response.error if response.error else None
            }
            
        except Exception as e:
            logger.error(f"❌ AddRSSFeed gRPC call failed: {e}")
            return {
                "success": False,
                "error": f"Add RSS feed failed: {str(e)}"
            }
    
    async def list_rss_feeds(
        self,
        user_id: str,
        scope: str = "user"
    ) -> Dict[str, Any]:
        """List RSS feeds via backend tool service"""
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.ListRSSFeedsRequest(
                user_id=user_id,
                scope=scope
            )
            
            logger.info(f"📰 Calling backend ListRSSFeeds: user={user_id}, scope={scope}...")
            
            response = await self._stub.ListRSSFeeds(request)
            
            feeds = []
            for feed in response.feeds:
                feeds.append({
                    "feed_id": feed.feed_id,
                    "feed_name": feed.feed_name,
                    "feed_url": feed.feed_url,
                    "category": feed.category,
                    "is_global": feed.is_global,
                    "last_polled": feed.last_polled,
                    "article_count": feed.article_count
                })
            
            logger.info(f"✅ Found {len(feeds)} RSS feeds")
            
            return {
                "success": response.success,
                "feeds": feeds,
                "count": response.count,
                "error": response.error if response.error else None
            }
            
        except Exception as e:
            logger.error(f"❌ ListRSSFeeds gRPC call failed: {e}")
            return {
                "success": False,
                "error": f"List RSS feeds failed: {str(e)}",
                "feeds": []
            }
    
    async def refresh_rss_feed(
        self,
        user_id: str,
        feed_name: str = "",
        feed_id: str = ""
    ) -> Dict[str, Any]:
        """Refresh RSS feed via backend tool service"""
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.RefreshRSSFeedRequest(
                user_id=user_id,
                feed_name=feed_name,
                feed_id=feed_id
            )
            
            logger.info(f"📰 Calling backend RefreshRSSFeed: feed_name={feed_name}, feed_id={feed_id}...")
            
            response = await self._stub.RefreshRSSFeed(request)
            
            return {
                "success": response.success,
                "feed_id": response.feed_id,
                "feed_name": response.feed_name,
                "task_id": response.task_id,
                "message": response.message,
                "error": response.error if response.error else None
            }
            
        except Exception as e:
            logger.error(f"❌ RefreshRSSFeed gRPC call failed: {e}")
            return {
                "success": False,
                "error": f"Refresh RSS feed failed: {str(e)}"
            }
    
    async def delete_rss_feed(
        self,
        user_id: str,
        feed_name: str = "",
        feed_id: str = ""
    ) -> Dict[str, Any]:
        """Delete RSS feed via backend tool service"""
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.DeleteRSSFeedRequest(
                user_id=user_id,
                feed_name=feed_name,
                feed_id=feed_id
            )
            
            logger.info(f"📰 Calling backend DeleteRSSFeed: feed_name={feed_name}, feed_id={feed_id}...")
            
            response = await self._stub.DeleteRSSFeed(request)
            
            return {
                "success": response.success,
                "feed_id": response.feed_id,
                "message": response.message,
                "error": response.error if response.error else None
            }
            
        except Exception as e:
            logger.error(f"❌ DeleteRSSFeed gRPC call failed: {e}")
            return {
                "success": False,
                "error": f"Delete RSS feed failed: {str(e)}"
            }
    
    async def list_org_inbox_items(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """List org inbox items via backend tool service"""
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.ListOrgInboxItemsRequest(
                user_id=user_id
            )
            
            logger.info(f"🗂️ Calling backend ListOrgInboxItems: user={user_id}...")
            
            response = await self._stub.ListOrgInboxItems(request)
            
            items = []
            for item in response.items:
                items.append({
                    "line_index": item.line_index,
                    "text": item.text,
                    "item_type": item.item_type,
                    "todo_state": item.todo_state,
                    "tags": list(item.tags),
                    "is_done": item.is_done
                })
            
            return {
                "success": response.success,
                "path": response.path,
                "items": items,
                "error": response.error if response.error else None
            }
            
        except Exception as e:
            logger.error(f"❌ ListOrgInboxItems gRPC call failed: {e}")
            return {
                "success": False,
                "error": f"List org inbox items failed: {str(e)}",
                "items": []
            }
    
    async def add_org_inbox_item(
        self,
        user_id: str,
        text: str,
        kind: str = "todo",
        schedule: str = None,
        repeater: str = None,
        tags: List[str] = None,
        contact_properties: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Add org inbox item via backend tool service"""
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.AddOrgInboxItemRequest(
                user_id=user_id,
                text=text,
                kind=kind
            )
            
            if schedule:
                request.schedule = schedule
            if repeater:
                request.repeater = repeater
            if tags:
                request.tags.extend(tags)
            if contact_properties:
                for key, value in contact_properties.items():
                    request.contact_properties[key] = value
            
            logger.info(f"🗂️ Calling backend AddOrgInboxItem: text={text[:50]}...")
            
            response = await self._stub.AddOrgInboxItem(request)
            
            return {
                "success": response.success,
                "line_index": response.line_index,
                "message": response.message,
                "error": response.error if response.error else None
            }
            
        except Exception as e:
            logger.error(f"❌ AddOrgInboxItem gRPC call failed: {e}")
            return {
                "success": False,
                "error": f"Add org inbox item failed: {str(e)}"
            }
    
    async def toggle_org_inbox_item(
        self,
        user_id: str,
        line_index: int
    ) -> Dict[str, Any]:
        """Toggle org inbox item via backend tool service"""
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.ToggleOrgInboxItemRequest(
                user_id=user_id,
                line_index=line_index
            )
            
            logger.info(f"🗂️ Calling backend ToggleOrgInboxItem: line={line_index}...")
            
            response = await self._stub.ToggleOrgInboxItem(request)
            
            return {
                "success": response.success,
                "updated_index": response.updated_index,
                "new_line": response.new_line,
                "error": response.error if response.error else None
            }
            
        except Exception as e:
            logger.error(f"❌ ToggleOrgInboxItem gRPC call failed: {e}")
            return {
                "success": False,
                "error": f"Toggle org inbox item failed: {str(e)}"
            }
    
    async def update_org_inbox_item(
        self,
        user_id: str,
        line_index: int,
        new_text: str
    ) -> Dict[str, Any]:
        """Update org inbox item via backend tool service"""
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.UpdateOrgInboxItemRequest(
                user_id=user_id,
                line_index=line_index,
                new_text=new_text
            )
            
            logger.info(f"🗂️ Calling backend UpdateOrgInboxItem: line={line_index}...")
            
            response = await self._stub.UpdateOrgInboxItem(request)
            
            return {
                "success": response.success,
                "updated_index": response.updated_index,
                "new_line": response.new_line,
                "error": response.error if response.error else None
            }
            
        except Exception as e:
            logger.error(f"❌ UpdateOrgInboxItem gRPC call failed: {e}")
            return {
                "success": False,
                "error": f"Update org inbox item failed: {str(e)}"
            }
    
    async def set_org_inbox_schedule(
        self,
        user_id: str,
        line_index: int,
        scheduled: str,
        repeater: str = None
    ) -> Dict[str, Any]:
        """Set org inbox schedule via backend tool service"""
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.SetOrgInboxScheduleRequest(
                user_id=user_id,
                line_index=line_index,
                scheduled=scheduled
            )
            
            if repeater:
                request.repeater = repeater
            
            logger.info(f"🗂️ Calling backend SetOrgInboxSchedule: line={line_index}...")
            
            response = await self._stub.SetOrgInboxSchedule(request)
            
            return {
                "success": response.success,
                "updated_index": response.updated_index,
                "scheduled_line": response.scheduled_line,
                "error": response.error if response.error else None
            }
            
        except Exception as e:
            logger.error(f"❌ SetOrgInboxSchedule gRPC call failed: {e}")
            return {
                "success": False,
                "error": f"Set org inbox schedule failed: {str(e)}"
            }
    
    async def apply_org_inbox_tags(
        self,
        user_id: str,
        line_index: int,
        tags: List[str]
    ) -> Dict[str, Any]:
        """Apply tags to org inbox item via backend tool service"""
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.ApplyOrgInboxTagsRequest(
                user_id=user_id,
                line_index=line_index
            )
            request.tags.extend(tags)
            
            logger.info(f"🗂️ Calling backend ApplyOrgInboxTags: line={line_index}, tags={tags}...")
            
            response = await self._stub.ApplyOrgInboxTags(request)
            
            return {
                "success": response.success,
                "applied_tags": list(response.applied_tags),
                "error": response.error if response.error else None
            }
            
        except Exception as e:
            logger.error(f"❌ ApplyOrgInboxTags gRPC call failed: {e}")
            return {
                "success": False,
                "error": f"Apply org inbox tags failed: {str(e)}"
            }
    
    async def archive_org_inbox_done(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Archive DONE items from org inbox via backend tool service"""
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.ArchiveOrgInboxDoneRequest(
                user_id=user_id
            )
            
            logger.info(f"🗂️ Calling backend ArchiveOrgInboxDone: user={user_id}...")
            
            response = await self._stub.ArchiveOrgInboxDone(request)
            
            return {
                "success": response.success,
                "archived_count": response.archived_count,
                "message": response.message,
                "error": response.error if response.error else None
            }
            
        except Exception as e:
            logger.error(f"❌ ArchiveOrgInboxDone gRPC call failed: {e}")
            return {
                "success": False,
                "error": f"Archive org inbox done failed: {str(e)}"
            }
    
    async def append_org_inbox_text(
        self,
        user_id: str,
        text: str
    ) -> Dict[str, Any]:
        """Append raw org-mode text to inbox via backend tool service"""
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.AppendOrgInboxTextRequest(
                user_id=user_id,
                text=text
            )
            
            logger.info(f"🗂️ Calling backend AppendOrgInboxText: user={user_id}...")
            
            response = await self._stub.AppendOrgInboxText(request)
            
            return {
                "success": response.success,
                "message": response.message,
                "error": response.error if response.error else None
            }
            
        except Exception as e:
            logger.error(f"❌ AppendOrgInboxText gRPC call failed: {e}")
            return {
                "success": False,
                "error": f"Append org inbox text failed: {str(e)}"
            }
    
    async def get_org_inbox_path(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Get org inbox path via backend tool service"""
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.GetOrgInboxPathRequest(
                user_id=user_id
            )
            
            logger.info(f"🗂️ Calling backend GetOrgInboxPath: user={user_id}...")
            
            response = await self._stub.GetOrgInboxPath(request)
            
            return {
                "success": response.success,
                "path": response.path,
                "error": response.error if response.error else None
            }
            
        except Exception as e:
            logger.error(f"❌ GetOrgInboxPath gRPC call failed: {e}")
            return {
                "success": False,
                "error": f"Get org inbox path failed: {str(e)}"
            }
    
    async def close(self):
        """Close gRPC channel"""
        if self._channel:
            await self._channel.close()
            logger.info("Closed backend tool service connection")


# Global instance
_backend_tool_client: Optional[BackendToolClient] = None


async def get_backend_tool_client() -> BackendToolClient:
    """Get global backend tool client instance"""
    global _backend_tool_client
    if _backend_tool_client is None:
        _backend_tool_client = BackendToolClient()
    return _backend_tool_client

