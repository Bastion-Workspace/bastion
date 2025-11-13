"""
gRPC Tool Service - Backend Data Access for LLM Orchestrator
Provides document, RSS, entity, weather, and org-mode data via gRPC
"""

import logging
from typing import Optional
import asyncio
import json

import grpc
from protos import tool_service_pb2, tool_service_pb2_grpc

# Import repositories and services directly (safe - no circular dependencies)
from repositories.document_repository import DocumentRepository
from services.direct_search_service import DirectSearchService
from services.embedding_service_wrapper import get_embedding_service

logger = logging.getLogger(__name__)


class ToolServiceImplementation(tool_service_pb2_grpc.ToolServiceServicer):
    """
    gRPC Tool Service Implementation
    
    Provides data access methods for the LLM Orchestrator service.
    Uses repositories directly for Phase 2 (services via container in Phase 3).
    """
    
    def __init__(self):
        logger.info("Initializing gRPC Tool Service...")
        # Use direct search service for document operations
        self._search_service: Optional[DirectSearchService] = None
        self._document_repo: Optional[DocumentRepository] = None
        self._embedding_manager = None  # EmbeddingServiceWrapper
    
    async def _get_search_service(self) -> DirectSearchService:
        """Lazy initialization of search service"""
        if not self._search_service:
            self._search_service = DirectSearchService()
        return self._search_service
    
    async def _get_embedding_manager(self):
        """Lazy initialization of embedding service wrapper"""
        if not self._embedding_manager:
            self._embedding_manager = await get_embedding_service()
        return self._embedding_manager
    
    def _get_document_repo(self) -> DocumentRepository:
        """Lazy initialization of document repository"""
        if not self._document_repo:
            self._document_repo = DocumentRepository()
        return self._document_repo
    
    # ===== Document Operations =====
    
    async def SearchDocuments(
        self,
        request: tool_service_pb2.SearchRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.SearchResponse:
        """Search documents by query using direct search with optional tag/category filtering"""
        try:
            logger.info(f"SearchDocuments: user={request.user_id}, query={request.query[:100]}")
            
            # Parse filters for tags and categories
            tags = []
            categories = []
            for filter_str in request.filters:
                if filter_str.startswith("tag:"):
                    tags.append(filter_str[4:])
                elif filter_str.startswith("category:"):
                    categories.append(filter_str[9:])
            
            if tags or categories:
                logger.info(f"SearchDocuments: Filtering by tags={tags}, categories={categories}")
            
            # Get search service
            search_service = await self._get_search_service()
            
            # Perform direct search with optional tag/category filtering
            search_result = await search_service.search_documents(
                query=request.query,
                limit=request.limit or 10,
                similarity_threshold=0.3,  # Lowered from 0.7 for better recall
                user_id=request.user_id if request.user_id and request.user_id != "system" else None,
                tags=tags if tags else None,
                categories=categories if categories else None
            )
            
            if not search_result.get("success"):
                logger.warning(f"SearchDocuments: Search failed - {search_result.get('error')}")
                return tool_service_pb2.SearchResponse(total_count=0)
            
            results = search_result.get("results", [])
            
            # Convert to proto response
            response = tool_service_pb2.SearchResponse(
                total_count=len(results)
            )
            
            for result in results:
                # DirectSearchService returns nested structure with document metadata
                document_metadata = result.get('document', {})
                doc_result = tool_service_pb2.DocumentResult(
                    document_id=str(document_metadata.get('document_id', '')),
                    title=document_metadata.get('title', document_metadata.get('filename', '')),
                    filename=document_metadata.get('filename', ''),
                    content_preview=result.get('text', '')[:500],  # Limit preview
                    relevance_score=float(result.get('similarity_score', 0.0))
                )
                response.results.append(doc_result)
            
            logger.info(f"SearchDocuments: Found {len(results)} results")
            return response
            
        except Exception as e:
            logger.error(f"SearchDocuments error: {e}")
            import traceback
            traceback.print_exc()
            await context.abort(grpc.StatusCode.INTERNAL, f"Search failed: {str(e)}")
    
    async def GetDocument(
        self,
        request: tool_service_pb2.DocumentRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DocumentResponse:
        """Get document metadata"""
        try:
            logger.info(f"GetDocument: doc_id={request.document_id}, user={request.user_id}")
            
            doc_repo = self._get_document_repo()
            doc = await doc_repo.get_document_by_id(document_id=request.document_id)
            
            if not doc:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Document not found")
            
            response = tool_service_pb2.DocumentResponse(
                document_id=str(doc.get('document_id', '')),
                title=doc.get('title', ''),
                filename=doc.get('filename', ''),
                content_type=doc.get('content_type', 'text/plain')
            )
            
            return response
            
        except Exception as e:
            logger.error(f"GetDocument error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Get document failed: {str(e)}")
    
    async def GetDocumentContent(
        self,
        request: tool_service_pb2.DocumentRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DocumentContentResponse:
        """Get document full content"""
        try:
            logger.info(f"GetDocumentContent: doc_id={request.document_id}")
            
            doc_repo = self._get_document_repo()
            doc = await doc_repo.get_document_by_id(document_id=request.document_id)
            
            if not doc:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Document not found")
            
            response = tool_service_pb2.DocumentContentResponse(
                document_id=str(doc.get('document_id', '')),
                content=doc.get('content', ''),
                format='text'
            )
            
            return response
            
        except Exception as e:
            logger.error(f"GetDocumentContent error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Get content failed: {str(e)}")
    
    # ===== RSS Operations =====
    
    async def SearchRSSFeeds(
        self,
        request: tool_service_pb2.RSSSearchRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.RSSSearchResponse:
        """Search RSS feeds and articles"""
        try:
            logger.info(f"SearchRSSFeeds: user={request.user_id}, query={request.query}")
            
            # Placeholder implementation - Phase 2 will wire up real RSS service
            response = tool_service_pb2.RSSSearchResponse()
            logger.info(f"SearchRSSFeeds: Returning placeholder response")
            return response
            
        except Exception as e:
            logger.error(f"SearchRSSFeeds error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"RSS search failed: {str(e)}")
    
    async def GetRSSArticles(
        self,
        request: tool_service_pb2.RSSArticlesRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.RSSArticlesResponse:
        """Get articles from RSS feed"""
        try:
            logger.info(f"GetRSSArticles: feed_id={request.feed_id}")
            
            # Placeholder implementation
            response = tool_service_pb2.RSSArticlesResponse()
            return response
            
        except Exception as e:
            logger.error(f"GetRSSArticles error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Get articles failed: {str(e)}")
    
    # ===== RSS Management Operations =====
    
    async def AddRSSFeed(
        self,
        request: tool_service_pb2.AddRSSFeedRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.AddRSSFeedResponse:
        """Add a new RSS feed"""
        try:
            logger.info(f"AddRSSFeed: user={request.user_id}, url={request.feed_url}, is_global={request.is_global}")
            
            from services.service_container import get_service_container
            from services.auth_service import get_auth_service
            from models.rss_models import RSSFeedCreate
            
            service_container = await get_service_container()
            rss_service = service_container.rss_service
            auth_service = await get_auth_service()
            
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
            
            from services.service_container import get_service_container
            from services.auth_service import get_auth_service
            
            service_container = await get_service_container()
            rss_service = service_container.rss_service
            auth_service = await get_auth_service()
            
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
            
            for feed in feeds:
                # Get article count for this feed
                from services.database_manager.database_helpers import fetch_value
                try:
                    article_count = await fetch_value(
                        "SELECT COUNT(*) FROM rss_articles WHERE feed_id = $1",
                        feed.feed_id
                    ) or 0
                except:
                    article_count = 0
                
                feed_details = tool_service_pb2.RSSFeedDetails(
                    feed_id=feed.feed_id,
                    feed_name=feed.feed_name,
                    feed_url=feed.feed_url,
                    category=feed.category or "general",
                    is_global=(feed.user_id is None),
                    last_polled=feed.last_poll_date.isoformat() if feed.last_poll_date else "",
                    article_count=int(article_count)
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
            
            from services.service_container import get_service_container
            from services.celery_tasks.rss_tasks import poll_rss_feeds_task
            
            service_container = await get_service_container()
            rss_service = service_container.rss_service
            
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
            
            from services.service_container import get_service_container
            
            service_container = await get_service_container()
            rss_service = service_container.rss_service
            
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
            await rss_service.delete_feed(target_feed.feed_id)
            
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
    
    # ===== Entity Operations =====
    
    async def SearchEntities(
        self,
        request: tool_service_pb2.EntitySearchRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.EntitySearchResponse:
        """Search entities"""
        try:
            logger.info(f"SearchEntities: query={request.query}")
            
            # Placeholder implementation
            response = tool_service_pb2.EntitySearchResponse()
            return response
            
        except Exception as e:
            logger.error(f"SearchEntities error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Entity search failed: {str(e)}")
    
    async def GetEntity(
        self,
        request: tool_service_pb2.EntityRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.EntityResponse:
        """Get entity details"""
        try:
            logger.info(f"GetEntity: entity_id={request.entity_id}")
            
            # Placeholder implementation
            entity = tool_service_pb2.Entity(
                entity_id=request.entity_id,
                entity_type="unknown",
                name="Placeholder"
            )
            response = tool_service_pb2.EntityResponse(entity=entity)
            return response
            
        except Exception as e:
            logger.error(f"GetEntity error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Get entity failed: {str(e)}")
    
    # ===== Weather Operations =====
    
    async def GetWeatherData(
        self,
        request: tool_service_pb2.WeatherRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.WeatherResponse:
        """Get weather data"""
        try:
            logger.info(f"GetWeatherData: location={request.location}")
            
            # Placeholder implementation
            response = tool_service_pb2.WeatherResponse(
                location=request.location,
                current_conditions="Placeholder weather data"
            )
            return response
            
        except Exception as e:
            logger.error(f"GetWeatherData error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Weather data failed: {str(e)}")
    
    # ===== Image Generation Operations =====
    
    async def GenerateImage(
        self,
        request: tool_service_pb2.ImageGenerationRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ImageGenerationResponse:
        """Generate images using OpenRouter image models"""
        try:
            logger.info(f"🎨 GenerateImage: prompt={request.prompt[:100]}...")
            
            # Get image generation service
            from services.image_generation_service import get_image_generation_service
            image_service = await get_image_generation_service()
            
            # Call image generation service
            result = await image_service.generate_images(
                prompt=request.prompt,
                size=request.size if request.size else "1024x1024",
                fmt=request.format if request.format else "png",
                seed=request.seed if request.HasField("seed") else None,
                num_images=request.num_images if request.num_images else 1,
                negative_prompt=request.negative_prompt if request.HasField("negative_prompt") else None
            )
            
            # Convert result to proto response
            if result.get("success"):
                images = []
                for img in result.get("images", []):
                    images.append(tool_service_pb2.GeneratedImage(
                        filename=img.get("filename", ""),
                        path=img.get("path", ""),
                        url=img.get("url", ""),
                        width=img.get("width", 1024),
                        height=img.get("height", 1024),
                        format=img.get("format", "png")
                    ))
                
                response = tool_service_pb2.ImageGenerationResponse(
                    success=True,
                    model=result.get("model", ""),
                    prompt=result.get("prompt", request.prompt),
                    size=result.get("size", "1024x1024"),
                    format=result.get("format", "png"),
                    images=images
                )
                logger.info(f"✅ Generated {len(images)} image(s) successfully")
                return response
            else:
                # Error occurred
                error_msg = result.get("error", "Unknown error")
                logger.error(f"❌ Image generation failed: {error_msg}")
                response = tool_service_pb2.ImageGenerationResponse(
                    success=False,
                    error=error_msg
                )
                return response
            
        except Exception as e:
            logger.error(f"❌ GenerateImage error: {e}")
            response = tool_service_pb2.ImageGenerationResponse(
                success=False,
                error=str(e)
            )
            return response
    
    # ===== Org-mode Operations =====
    
    async def SearchOrgFiles(
        self,
        request: tool_service_pb2.OrgSearchRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.OrgSearchResponse:
        """Search org-mode files"""
        try:
            logger.info(f"SearchOrgFiles: query={request.query}")
            
            # Placeholder implementation
            response = tool_service_pb2.OrgSearchResponse()
            return response
            
        except Exception as e:
            logger.error(f"SearchOrgFiles error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Org search failed: {str(e)}")
    
    async def GetOrgInboxItems(
        self,
        request: tool_service_pb2.OrgInboxRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.OrgInboxResponse:
        """Get org-mode inbox items"""
        try:
            logger.info(f"GetOrgInboxItems: user={request.user_id}")
            
            # Placeholder implementation
            response = tool_service_pb2.OrgInboxResponse()
            return response
            
        except Exception as e:
            logger.error(f"GetOrgInboxItems error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Get inbox items failed: {str(e)}")
    
    # ===== Org Inbox Management Operations =====
    
    async def ListOrgInboxItems(
        self,
        request: tool_service_pb2.ListOrgInboxItemsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ListOrgInboxItemsResponse:
        """List all org inbox items for user"""
        try:
            logger.info(f"ListOrgInboxItems: user={request.user_id}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_list_items, org_inbox_path
            
            # Get inbox path
            path = await org_inbox_path(request.user_id)
            
            # List items
            listing = await org_inbox_list_items(request.user_id)
            
            # Convert to proto response
            response = tool_service_pb2.ListOrgInboxItemsResponse(
                success=True,
                path=path
            )
            
            for item in listing.get("items", []):
                item_details = tool_service_pb2.OrgInboxItemDetails(
                    line_index=item.get("line_index", 0),
                    text=item.get("text", ""),
                    item_type=item.get("item_type", "plain"),
                    todo_state=item.get("todo_state", ""),
                    tags=item.get("tags", []),
                    is_done=item.get("is_done", False)
                )
                response.items.append(item_details)
            
            logger.info(f"ListOrgInboxItems: Found {len(response.items)} items")
            return response
            
        except Exception as e:
            logger.error(f"❌ ListOrgInboxItems error: {e}")
            return tool_service_pb2.ListOrgInboxItemsResponse(
                success=False,
                error=str(e)
            )
    
    async def AddOrgInboxItem(
        self,
        request: tool_service_pb2.AddOrgInboxItemRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.AddOrgInboxItemResponse:
        """Add new item to org inbox"""
        try:
            logger.info(f"AddOrgInboxItem: user={request.user_id}, kind={request.kind}, text={request.text[:50]}...")
            
            from services.langgraph_tools.org_inbox_tools import (
                org_inbox_add_item,
                org_inbox_append_text,
                org_inbox_list_items,
                org_inbox_set_schedule_and_repeater,
                org_inbox_apply_tags
            )
            
            # Handle different kinds of entries
            if request.kind == "contact":
                # Build contact entry with PROPERTIES drawer
                headline = f"* {request.text}"
                org_entry = f"{headline}\n"
                
                if request.contact_properties:
                    org_entry += ":PROPERTIES:\n"
                    for key, value in request.contact_properties.items():
                        if value:
                            org_entry += f":{key}: {value}\n"
                    org_entry += ":END:\n"
                
                result = await org_inbox_append_text(org_entry, request.user_id)
                line_index = None  # Will determine after listing
                
            elif request.schedule or request.kind == "event":
                # Build a proper org-mode entry with schedule
                org_type = "TODO" if request.kind == "todo" else ""
                headline = f"* {org_type} {request.text}".strip()
                org_entry = f"{headline}\n"
                result = await org_inbox_append_text(org_entry, request.user_id)
                
                # Get the line index of the newly added item
                listing = await org_inbox_list_items(request.user_id)
                items = listing.get("items", [])
                line_index = items[-1].get("line_index") if items else None
                
                # Set schedule if provided
                if line_index is not None and request.schedule:
                    await org_inbox_set_schedule_and_repeater(
                        line_index=line_index,
                        scheduled=request.schedule,
                        repeater=request.repeater if request.repeater else None,
                        user_id=request.user_id
                    )
            else:
                # Regular todo or checkbox
                kind = "todo" if request.kind != "checkbox" else "checkbox"
                result = await org_inbox_add_item(text=request.text, kind=kind, user_id=request.user_id)
                line_index = result.get("line_index")
            
            # Apply tags if provided
            if line_index is not None and request.tags:
                await org_inbox_apply_tags(line_index=line_index, tags=list(request.tags), user_id=request.user_id)
            elif line_index is None and request.tags:
                # Best effort: get last item's index
                listing = await org_inbox_list_items(request.user_id)
                items = listing.get("items", [])
                if items:
                    line_index = items[-1].get("line_index")
                    if line_index is not None:
                        await org_inbox_apply_tags(line_index=line_index, tags=list(request.tags), user_id=request.user_id)
            
            logger.info(f"✅ AddOrgInboxItem: Added item successfully")
            return tool_service_pb2.AddOrgInboxItemResponse(
                success=True,
                line_index=line_index if line_index is not None else 0,
                message=f"Added '{request.text}' to inbox.org"
            )
            
        except Exception as e:
            logger.error(f"❌ AddOrgInboxItem error: {e}")
            return tool_service_pb2.AddOrgInboxItemResponse(
                success=False,
                error=str(e)
            )
    
    async def ToggleOrgInboxItem(
        self,
        request: tool_service_pb2.ToggleOrgInboxItemRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ToggleOrgInboxItemResponse:
        """Toggle DONE status of org inbox item"""
        try:
            logger.info(f"ToggleOrgInboxItem: user={request.user_id}, line={request.line_index}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_toggle_done
            
            result = await org_inbox_toggle_done(line_index=request.line_index, user_id=request.user_id)
            
            if result.get("error"):
                return tool_service_pb2.ToggleOrgInboxItemResponse(
                    success=False,
                    error=result.get("error")
                )
            
            return tool_service_pb2.ToggleOrgInboxItemResponse(
                success=True,
                updated_index=result.get("updated_index", request.line_index),
                new_line=result.get("new_line", "")
            )
            
        except Exception as e:
            logger.error(f"❌ ToggleOrgInboxItem error: {e}")
            return tool_service_pb2.ToggleOrgInboxItemResponse(
                success=False,
                error=str(e)
            )
    
    async def UpdateOrgInboxItem(
        self,
        request: tool_service_pb2.UpdateOrgInboxItemRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.UpdateOrgInboxItemResponse:
        """Update org inbox item text"""
        try:
            logger.info(f"UpdateOrgInboxItem: user={request.user_id}, line={request.line_index}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_update_line
            
            result = await org_inbox_update_line(
                line_index=request.line_index,
                new_text=request.new_text,
                user_id=request.user_id
            )
            
            if result.get("error"):
                return tool_service_pb2.UpdateOrgInboxItemResponse(
                    success=False,
                    error=result.get("error")
                )
            
            return tool_service_pb2.UpdateOrgInboxItemResponse(
                success=True,
                updated_index=result.get("updated_index", request.line_index),
                new_line=result.get("new_line", "")
            )
            
        except Exception as e:
            logger.error(f"❌ UpdateOrgInboxItem error: {e}")
            return tool_service_pb2.UpdateOrgInboxItemResponse(
                success=False,
                error=str(e)
            )
    
    async def SetOrgInboxSchedule(
        self,
        request: tool_service_pb2.SetOrgInboxScheduleRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.SetOrgInboxScheduleResponse:
        """Set schedule and repeater for org inbox item"""
        try:
            logger.info(f"SetOrgInboxSchedule: user={request.user_id}, line={request.line_index}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_set_schedule_and_repeater
            
            result = await org_inbox_set_schedule_and_repeater(
                line_index=request.line_index,
                scheduled=request.scheduled,
                repeater=request.repeater if request.repeater else None,
                user_id=request.user_id
            )
            
            if result.get("error"):
                return tool_service_pb2.SetOrgInboxScheduleResponse(
                    success=False,
                    error=result.get("error")
                )
            
            return tool_service_pb2.SetOrgInboxScheduleResponse(
                success=True,
                updated_index=result.get("updated_index", request.line_index),
                scheduled_line=result.get("scheduled_line", "")
            )
            
        except Exception as e:
            logger.error(f"❌ SetOrgInboxSchedule error: {e}")
            return tool_service_pb2.SetOrgInboxScheduleResponse(
                success=False,
                error=str(e)
            )
    
    async def ApplyOrgInboxTags(
        self,
        request: tool_service_pb2.ApplyOrgInboxTagsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ApplyOrgInboxTagsResponse:
        """Apply tags to org inbox item"""
        try:
            logger.info(f"ApplyOrgInboxTags: user={request.user_id}, line={request.line_index}, tags={list(request.tags)}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_apply_tags
            
            result = await org_inbox_apply_tags(
                line_index=request.line_index,
                tags=list(request.tags),
                user_id=request.user_id
            )
            
            if result.get("error"):
                return tool_service_pb2.ApplyOrgInboxTagsResponse(
                    success=False,
                    error=result.get("error")
                )
            
            return tool_service_pb2.ApplyOrgInboxTagsResponse(
                success=True,
                applied_tags=list(request.tags)
            )
            
        except Exception as e:
            logger.error(f"❌ ApplyOrgInboxTags error: {e}")
            return tool_service_pb2.ApplyOrgInboxTagsResponse(
                success=False,
                error=str(e)
            )
    
    async def ArchiveOrgInboxDone(
        self,
        request: tool_service_pb2.ArchiveOrgInboxDoneRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ArchiveOrgInboxDoneResponse:
        """Archive all DONE items from org inbox"""
        try:
            logger.info(f"ArchiveOrgInboxDone: user={request.user_id}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_archive_done
            
            result = await org_inbox_archive_done(request.user_id)
            
            if result.get("error"):
                return tool_service_pb2.ArchiveOrgInboxDoneResponse(
                    success=False,
                    error=result.get("error")
                )
            
            archived_count = result.get("archived_count", 0)
            
            return tool_service_pb2.ArchiveOrgInboxDoneResponse(
                success=True,
                archived_count=archived_count,
                message=f"Archived {archived_count} DONE items"
            )
            
        except Exception as e:
            logger.error(f"❌ ArchiveOrgInboxDone error: {e}")
            return tool_service_pb2.ArchiveOrgInboxDoneResponse(
                success=False,
                error=str(e)
            )
    
    async def AppendOrgInboxText(
        self,
        request: tool_service_pb2.AppendOrgInboxTextRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.AppendOrgInboxTextResponse:
        """Append raw org-mode text to inbox"""
        try:
            logger.info(f"AppendOrgInboxText: user={request.user_id}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_append_text
            
            result = await org_inbox_append_text(request.text, request.user_id)
            
            if result.get("error"):
                return tool_service_pb2.AppendOrgInboxTextResponse(
                    success=False,
                    error=result.get("error")
                )
            
            return tool_service_pb2.AppendOrgInboxTextResponse(
                success=True,
                message="Text appended to inbox.org"
            )
            
        except Exception as e:
            logger.error(f"❌ AppendOrgInboxText error: {e}")
            return tool_service_pb2.AppendOrgInboxTextResponse(
                success=False,
                error=str(e)
            )
    
    async def GetOrgInboxPath(
        self,
        request: tool_service_pb2.GetOrgInboxPathRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.GetOrgInboxPathResponse:
        """Get path to user's inbox.org file"""
        try:
            logger.info(f"GetOrgInboxPath: user={request.user_id}")
            
            from services.langgraph_tools.org_inbox_tools import org_inbox_path
            
            path = await org_inbox_path(request.user_id)
            
            return tool_service_pb2.GetOrgInboxPathResponse(
                success=True,
                path=path
            )
            
        except Exception as e:
            logger.error(f"❌ GetOrgInboxPath error: {e}")
            return tool_service_pb2.GetOrgInboxPathResponse(
                success=False,
                error=str(e)
            )
    
    # ===== Web Operations =====
    
    async def SearchWeb(
        self,
        request: tool_service_pb2.WebSearchRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.WebSearchResponse:
        """Search the web"""
        try:
            logger.info(f"SearchWeb: query={request.query}")
            
            # Import web search tool
            from services.langgraph_tools.web_content_tools import search_web
            
            # Execute search
            results = await search_web(query=request.query, num_results=request.max_results or 15)
            
            # Parse results (returns list of dicts)
            response = tool_service_pb2.WebSearchResponse()
            
            if isinstance(results, list):
                for result in results[:request.max_results or 15]:
                    web_result = tool_service_pb2.WebSearchResult(
                        title=result.get('title', ''),
                        url=result.get('url', ''),
                        snippet=result.get('snippet', ''),
                        relevance_score=float(result.get('relevance_score', 0.0))
                    )
                    response.results.append(web_result)
            
            logger.info(f"SearchWeb: Found {len(response.results)} results")
            return response
            
        except Exception as e:
            logger.error(f"SearchWeb error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Web search failed: {str(e)}")
    
    async def CrawlWebContent(
        self,
        request: tool_service_pb2.WebCrawlRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.WebCrawlResponse:
        """Crawl web content from URLs"""
        try:
            urls = list(request.urls) if request.urls else ([request.url] if request.url else [])
            logger.info(f"CrawlWebContent: {len(urls)} URLs")
            
            # Import crawl tool
            from services.langgraph_tools.web_content_tools import crawl_web_content
            
            # Execute crawl
            result = await crawl_web_content(url=request.url if request.url else None, urls=list(request.urls) if request.urls else None)
            
            response = tool_service_pb2.WebCrawlResponse()
            
            # Parse result
            if isinstance(result, dict) and 'results' in result:
                for item in result['results']:
                    crawl_result = tool_service_pb2.WebCrawlResult(
                        url=item.get('url', ''),
                        title=item.get('title', ''),
                        content=item.get('content', ''),
                        metadata={}  # WebCrawlResult doesn't have html field
                    )
                    response.results.append(crawl_result)
            
            logger.info(f"CrawlWebContent: Crawled {len(response.results)} URLs")
            return response
            
        except Exception as e:
            logger.error(f"CrawlWebContent error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Web crawl failed: {str(e)}")
    
    async def SearchAndCrawl(
        self,
        request: tool_service_pb2.SearchAndCrawlRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.SearchAndCrawlResponse:
        """Combined search and crawl operation"""
        try:
            logger.info(f"SearchAndCrawl: query={request.query}")
            
            # Import tool
            from services.langgraph_tools.web_content_tools import search_and_crawl
            
            # Execute combined operation
            result = await search_and_crawl(query=request.query, max_results=request.num_results or 10)
            
            response = tool_service_pb2.SearchAndCrawlResponse()
            
            # Parse results - tool returns dict with search_results and crawled_content
            if isinstance(result, dict):
                # Add search results
                if 'search_results' in result and isinstance(result['search_results'], list):
                    for item in result['search_results']:
                        search_result = tool_service_pb2.WebSearchResult(
                            title=item.get('title', ''),
                            url=item.get('url', ''),
                            snippet=item.get('snippet', ''),
                            relevance_score=float(item.get('relevance_score', 0.0))
                        )
                        response.search_results.append(search_result)
                
                # Add crawl results
                if 'crawled_content' in result and isinstance(result['crawled_content'], list):
                    for item in result['crawled_content']:
                        crawl_result = tool_service_pb2.WebCrawlResult(
                            url=item.get('url', ''),
                            title=item.get('title', ''),
                            content=item.get('content', ''),
                            metadata={}  # WebCrawlResult doesn't have html field, use metadata instead
                        )
                        response.crawl_results.append(crawl_result)
            
            logger.info(f"SearchAndCrawl: {len(response.search_results)} search results, {len(response.crawl_results)} crawled")
            return response
            
        except Exception as e:
            logger.error(f"SearchAndCrawl error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Search and crawl failed: {str(e)}")
    
    # ===== Query Enhancement =====
    
    async def ExpandQuery(
        self,
        request: tool_service_pb2.QueryExpansionRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.QueryExpansionResponse:
        """Expand query with variations"""
        try:
            logger.info(f"ExpandQuery: query={request.query}")
            
            # Import expansion tool
            from services.langgraph_tools.query_expansion_tool import expand_query
            
            # Execute expansion - returns JSON string
            result_json = await expand_query(
                original_query=request.query, 
                num_expansions=request.num_variations or 3
            )
            result = json.loads(result_json)
            
            # Parse result
            response = tool_service_pb2.QueryExpansionResponse(
                original_query=request.query,
                expansion_count=0
            )
            
            if isinstance(result, dict):
                response.original_query = result.get('original_query', request.query)
                response.expanded_queries.extend(result.get('expanded_queries', []))
                response.key_entities.extend(result.get('key_entities', []))
                response.expansion_count = len(response.expanded_queries)
            
            logger.info(f"ExpandQuery: Generated {response.expansion_count} variations")
            return response
            
        except Exception as e:
            logger.error(f"ExpandQuery error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Query expansion failed: {str(e)}")
    
    # ===== Conversation Cache =====
    
    async def SearchConversationCache(
        self,
        request: tool_service_pb2.CacheSearchRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.CacheSearchResponse:
        """Search conversation cache for previous research"""
        try:
            logger.info(f"SearchConversationCache: query={request.query}")
            
            # Import cache tool
            from services.langgraph_tools.unified_search_tools import search_conversation_cache
            
            # Execute cache search
            result = await search_conversation_cache(
                query=request.query,
                conversation_id=request.conversation_id if request.conversation_id else None,
                freshness_hours=request.freshness_hours or 24
            )
            
            response = tool_service_pb2.CacheSearchResponse(cache_hit=False)
            
            # Parse result
            if isinstance(result, dict) and result.get('cache_hit'):
                response.cache_hit = True
                entries = result.get('entries', [])
                for entry in entries:
                    cache_entry = tool_service_pb2.CacheEntry(
                        content=entry.get('content', ''),
                        timestamp=entry.get('timestamp', ''),
                        agent_name=entry.get('agent_name', ''),
                        relevance_score=float(entry.get('relevance_score', 0.0))
                    )
                    response.entries.append(cache_entry)
            
            logger.info(f"SearchConversationCache: Cache hit={response.cache_hit}, {len(response.entries)} entries")
            return response
            
        except Exception as e:
            logger.error(f"SearchConversationCache error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Cache search failed: {str(e)}")


async def serve_tool_service(port: int = 50052):
    """
    Start the gRPC tool service server
    
    Runs alongside the main FastAPI server to provide data access
    for the LLM orchestrator service.
    """
    try:
        # Import health checking inside function (lesson learned!)
        from grpc_health.v1 import health, health_pb2, health_pb2_grpc
        
        logger.info(f"Starting gRPC Tool Service on port {port}...")
        
        # Create gRPC server with increased message size limits
        # Default is 4MB, increase to 100MB for large document search responses
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100 MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100 MB
        ]
        server = grpc.aio.server(options=options)
        
        # Register tool service
        tool_service = ToolServiceImplementation()
        tool_service_pb2_grpc.add_ToolServiceServicer_to_server(tool_service, server)
        
        # Register health checking
        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
        health_servicer.set(
            "tool_service.ToolService",
            health_pb2.HealthCheckResponse.SERVING
        )
        
        # Bind to port
        server.add_insecure_port(f'[::]:{port}')
        
        # Start server
        await server.start()
        logger.info(f"✅ gRPC Tool Service listening on port {port}")
        
        # Wait for termination
        await server.wait_for_termination()
        
    except Exception as e:
        logger.error(f"❌ gRPC Tool Service failed to start: {e}")
        raise

