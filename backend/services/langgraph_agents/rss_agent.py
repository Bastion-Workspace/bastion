"""
RSS Agent for Chat Interactions
LangGraph agent for RSS feed management through natural language commands
"""

import logging
import re
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

from services.langgraph_agents.base_agent import BaseAgent
from models.agent_response_models import RSSManagementResult, RSSMetadataRequest
from models.rss_models import RSSFeedCreate
import uuid

logger = logging.getLogger(__name__)


class RSSAgent(BaseAgent):
    """
    RSS Agent for chat-based RSS feed management
    
    **BULLY!** This agent handles RSS feed operations through natural language commands!
    """
    
    def __init__(self):
        super().__init__("rss_agent")
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for RSS management"""
        return """
You are the RSS Management Agent, responsible for handling RSS feed operations through natural language commands.

**Your Mission:**
- Parse RSS feed management commands from user messages
- Add new RSS feeds to the system with proper metadata validation
- List and manage existing RSS feeds
- Handle both user-specific and global RSS feeds
- Provide clear feedback on RSS operations

**CRITICAL METADATA REQUIREMENTS:**
- **Title and Category are MANDATORY** for all RSS feeds
- If title or category is missing, request them from the user
- Never proceed with incomplete metadata
- Provide helpful suggestions based on URL analysis

**STRUCTURED OUTPUT REQUIRED:**
You MUST respond with valid JSON matching this schema:
{
    "task_status": "complete|incomplete|permission_required|error",
    "response": "Your natural language response",
    "metadata": {
        "operation_type": "add_feed|list_feeds|refresh_feed|delete_feed|metadata_request",
        "feeds_processed": 0,
        "feeds_added": 0,
        "feeds_listed": 0,
        "errors": []
    },
    "rss_operations": [
        {
            "operation": "add_feed|list_feeds|refresh_feed|delete_feed|metadata_request",
            "feed_url": "string (for add operations)",
            "feed_name": "string (for add operations)",
            "category": "string (for add operations)",
            "scope": "user|global",
            "status": "success|error|pending|metadata_required",
            "error_message": "string (if error)",
            "missing_metadata": ["title", "category"] (if metadata_required)
        }
    ]
}

**METADATA REQUEST PATTERN:**
When metadata is missing, use this structured format:
{
    "task_status": "incomplete",
    "response": "I need some additional information to add this RSS feed. Please provide:",
    "missing_metadata": ["title", "category"],
    "feed_url": "https://example.com/feed",
    "suggested_title": "Example Feed",
    "suggested_category": "technology",
    "available_categories": ["technology", "science", "news", "business", "politics", "entertainment", "sports", "health", "education", "other"],
    "operation_id": "unique_operation_id"
}

**RSS Command Patterns:**
- "Add RSS feed: <url>" - Add new RSS feed (will request metadata if missing)
- "Add global RSS feed: <url>" - Add global RSS feed (will request metadata if missing)
- "List my RSS feeds" - List user's RSS feeds
- "List global RSS feeds" - List global RSS feeds
- "Refresh RSS feed <name>" - Refresh specific feed
- "Delete RSS feed <name>" - Delete specific feed

**RSS Processing Rules:**
1. Always validate RSS feed URLs before adding
2. **REQUIRE title and category for all feeds**
3. Extract meaningful suggestions from URLs when possible
4. Distinguish between user-specific and global feeds
5. Provide clear feedback on operation success/failure
6. Handle permission requirements for global operations
7. **ADMIN ONLY**: Global RSS feeds can only be added by admin users
8. **USER FEEDS**: Regular users can only add user-specific feeds

**By George!** Follow these practices and your RSS management will be as efficient as a cavalry charge!
"""
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process RSS management commands from chat
        
        **Trust busting for unstructured RSS commands!** We use structured outputs only!
        """
        try:
            logger.info(f"📰 RSS AGENT: Starting RSS management processing")
            
            # Extract user message from state
            messages = state.get("messages", [])
            if not messages:
                return self._create_response(
                    task_status="error",
                    response="No user message found for RSS processing",
                    metadata={"errors": ["No user message provided"]}
                )
            
            # Get the latest user message
            latest_message = messages[-1]
            user_message = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
            user_id = state.get("user_id")
            
            logger.info(f"📰 RSS AGENT: Processing message: {user_message[:100]}...")
            
            # Check if this is a metadata response for pending RSS operations
            shared_memory = state.get("shared_memory", {})
            rss_metadata_request = shared_memory.get("rss_metadata_request", {})
            pending_operations = rss_metadata_request.get("pending_operations", [])
            
            if pending_operations:
                # This is a metadata response - update operations with user's metadata
                updated_operations = await self._update_operations_with_metadata(
                    pending_operations, user_message
                )
                rss_operations = updated_operations
            else:
                # This is a new RSS command - parse it
                rss_operations = await self._parse_rss_commands(user_message, user_id)
            
            if not rss_operations:
                return self._create_response(
                    task_status="complete",
                    response="No RSS operations detected in your message. Try commands like 'Add RSS feed: <url>' or 'List my RSS feeds'.",
                    metadata={"operation_type": "none", "feeds_processed": 0}
                )
            
            # Execute RSS operations
            results = await self._execute_rss_operations(rss_operations, user_id)
            
            # Create structured response
            response = self._create_rss_response(results, user_message)
            
            logger.info(f"✅ RSS AGENT: Completed RSS management processing")
            return response
            
        except Exception as e:
            logger.error(f"❌ RSS AGENT ERROR: {e}")
            return self._create_response(
                task_status="error",
                response=f"RSS management failed: {str(e)}",
                metadata={"errors": [str(e)]}
            )
    
    async def _parse_rss_commands(self, user_message: str, user_id: str) -> List[Dict[str, Any]]:
        """Parse RSS commands from user message"""
        operations = []
        user_message_lower = user_message.lower()
        
        # Pattern for adding RSS feeds with metadata
        add_patterns = [
            r'add\s+(?:global\s+)?rss\s+feed:?\s*(https?://[^\s]+)(?:\s+as\s+"([^"]+)"\s+in\s+(\w+))?',
            r'add\s+(?:global\s+)?rss\s+feed\s+(https?://[^\s]+)(?:\s+as\s+"([^"]+)"\s+in\s+(\w+))?',
            r'add\s+(?:global\s+)?feed:?\s*(https?://[^\s]+)(?:\s+as\s+"([^"]+)"\s+in\s+(\w+))?',
        ]
        
        for pattern in add_patterns:
            matches = re.finditer(pattern, user_message_lower)
            for match in matches:
                feed_url = match.group(1)
                is_global = "global" in match.group(0)
                
                # Extract title and category if provided
                provided_title = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
                provided_category = match.group(3) if len(match.groups()) > 2 and match.group(3) else None
                
                # Generate suggested title from URL
                suggested_title = self._extract_feed_name_from_url(feed_url)
                suggested_category = self._suggest_category_from_url(feed_url)
                
                # Determine if metadata is complete
                has_title = provided_title is not None
                has_category = provided_category is not None
                
                operation = {
                    "operation": "add_feed",
                    "feed_url": feed_url,
                    "feed_name": provided_title or suggested_title,
                    "category": provided_category or suggested_category,
                    "scope": "global" if is_global else "user",
                    "status": "pending" if (has_title and has_category) else "metadata_required",
                    "operation_id": str(uuid.uuid4()),
                    "suggested_title": suggested_title,
                    "suggested_category": suggested_category,
                    "missing_metadata": []
                }
                
                # Track missing metadata
                if not has_title:
                    operation["missing_metadata"].append("title")
                if not has_category:
                    operation["missing_metadata"].append("category")
                
                operations.append(operation)
        
        # Pattern for listing RSS feeds
        if "list" in user_message_lower and "rss" in user_message_lower:
            if "global" in user_message_lower:
                operations.append({
                    "operation": "list_feeds",
                    "scope": "global",
                    "status": "pending"
                })
            else:
                operations.append({
                    "operation": "list_feeds",
                    "scope": "user",
                    "status": "pending"
                })
        
        # Pattern for refreshing RSS feeds
        refresh_pattern = r'refresh\s+(?:rss\s+)?feed\s+([^\s]+)'
        refresh_matches = re.finditer(refresh_pattern, user_message_lower)
        for match in refresh_matches:
            feed_name = match.group(1)
            operations.append({
                "operation": "refresh_feed",
                "feed_name": feed_name,
                "scope": "user",  # Default to user scope
                "status": "pending"
            })
        
        return operations
    
    def _extract_feed_name_from_url(self, url: str) -> str:
        """Extract a meaningful feed name from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
            
            # Common RSS feed patterns
            if 'rss' in url.lower():
                return f"RSS Feed from {domain}"
            elif 'feed' in url.lower():
                return f"Feed from {domain}"
            elif 'blog' in url.lower():
                return f"Blog from {domain}"
            elif 'news' in url.lower():
                return f"News from {domain}"
            else:
                return f"Feed from {domain}"
        except:
            return "RSS Feed"
    
    def _suggest_category_from_url(self, url: str) -> str:
        """Suggest category based on URL analysis"""
        url_lower = url.lower()
        
        # Technology patterns
        if any(tech in url_lower for tech in ['tech', 'technology', 'programming', 'software', 'github', 'stackoverflow']):
            return "technology"
        
        # Science patterns
        if any(science in url_lower for science in ['science', 'research', 'nature', 'scientific', 'arxiv']):
            return "science"
        
        # News patterns
        if any(news in url_lower for news in ['news', 'bbc', 'cnn', 'reuters', 'ap', 'breaking']):
            return "news"
        
        # Business patterns
        if any(business in url_lower for business in ['business', 'finance', 'economy', 'market', 'forbes', 'wsj']):
            return "business"
        
        # Politics patterns
        if any(politics in url_lower for politics in ['politics', 'government', 'policy', 'congress', 'senate']):
            return "politics"
        
        # Entertainment patterns
        if any(entertainment in url_lower for entertainment in ['entertainment', 'movie', 'music', 'celebrity', 'hollywood']):
            return "entertainment"
        
        # Sports patterns
        if any(sports in url_lower for sports in ['sports', 'football', 'basketball', 'baseball', 'soccer', 'nfl', 'nba']):
            return "sports"
        
        # Health patterns
        if any(health in url_lower for health in ['health', 'medical', 'medicine', 'fitness', 'wellness']):
            return "health"
        
        # Education patterns
        if any(education in url_lower for education in ['education', 'university', 'college', 'school', 'academic']):
            return "education"
        
        # Default to general
        return "other"
    
    async def _execute_rss_operations(self, operations: List[Dict[str, Any]], user_id: str) -> List[Dict[str, Any]]:
        """Execute RSS operations"""
        results = []
        
        for operation in operations:
            try:
                if operation["operation"] == "add_feed":
                    if operation["status"] == "metadata_required":
                        result = await self._create_metadata_request(operation, user_id)
                    else:
                        result = await self._add_rss_feed(operation, user_id)
                elif operation["operation"] == "list_feeds":
                    result = await self._list_rss_feeds(operation, user_id)
                elif operation["operation"] == "refresh_feed":
                    result = await self._refresh_rss_feed(operation, user_id)
                else:
                    result = {
                        **operation,
                        "status": "error",
                        "error_message": f"Unknown operation: {operation['operation']}"
                    }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"❌ RSS AGENT ERROR: Failed to execute operation {operation}: {e}")
                results.append({
                    **operation,
                    "status": "error",
                    "error_message": str(e)
                })
        
        return results
    
    async def _create_metadata_request(self, operation: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Create a metadata request for incomplete RSS feed operations"""
        try:
            missing_metadata = operation.get("missing_metadata", [])
            feed_url = operation["feed_url"]
            suggested_title = operation.get("suggested_title", "RSS Feed")
            suggested_category = operation.get("suggested_category", "other")
            
            # Build request message
            request_parts = []
            if "title" in missing_metadata:
                request_parts.append(f"a title (suggested: '{suggested_title}')")
            if "category" in missing_metadata:
                request_parts.append(f"a category (suggested: '{suggested_category}')")
            
            request_message = f"I need {', '.join(request_parts)} to add this RSS feed."
            
            # Available categories
            available_categories = [
                "technology", "science", "news", "business", "politics", 
                "entertainment", "sports", "health", "education", "other"
            ]
            
            return {
                **operation,
                "status": "metadata_required",
                "response": request_message,
                "missing_metadata": missing_metadata,
                "suggested_title": suggested_title,
                "suggested_category": suggested_category,
                "available_categories": available_categories,
                "operation_id": operation["operation_id"]
            }
            
        except Exception as e:
            logger.error(f"❌ RSS AGENT ERROR: Failed to create metadata request: {e}")
            return {
                **operation,
                "status": "error",
                "error_message": str(e)
            }
    
    async def _add_rss_feed(self, operation: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Add a new RSS feed"""
        try:
            from services.service_container import get_service_container
            from services.auth_service import get_auth_service
            
            service_container = await get_service_container()
            rss_service = service_container.rss_service
            auth_service = await get_auth_service()
            
            # Check permissions for global feeds
            if operation["scope"] == "global":
                # Get user info to check admin status
                user_info = await auth_service.get_user_by_id(user_id)
                if not user_info or user_info.role != "admin":
                    return {
                        **operation,
                        "status": "error",
                        "error_message": "Only admin users can add global RSS feeds. Please contact an administrator or add this as a user-specific feed."
                    }
            
            # Create RSS feed data
            feed_data = RSSFeedCreate(
                feed_url=operation["feed_url"],
                feed_name=operation["feed_name"],
                user_id=user_id if operation["scope"] == "user" else None,  # None for global
                category="general",
                tags=["rss", "imported"],
                check_interval=3600
            )
            
            # Add the feed
            new_feed = await rss_service.create_feed(feed_data)
            
            return {
                **operation,
                "status": "success",
                "feed_id": new_feed.feed_id,
                "message": f"Successfully added {operation['scope']} RSS feed: {new_feed.feed_name}"
            }
            
        except Exception as e:
            logger.error(f"❌ RSS AGENT ERROR: Failed to add RSS feed: {e}")
            return {
                **operation,
                "status": "error",
                "error_message": str(e)
            }
    
    async def _list_rss_feeds(self, operation: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """List RSS feeds"""
        try:
            from services.service_container import get_service_container
            service_container = await get_service_container()
            rss_service = service_container.rss_service
            
            # Get feeds based on scope
            if operation["scope"] == "global":
                feeds = await rss_service.get_user_feeds(user_id, is_admin=True)  # Admin can see global
            else:
                feeds = await rss_service.get_user_feeds(user_id, is_admin=False)
            
            feed_list = []
            for feed in feeds:
                feed_list.append({
                    "feed_id": feed.feed_id,
                    "feed_name": feed.feed_name,
                    "feed_url": feed.feed_url,
                    "category": feed.category,
                    "is_global": feed.user_id is None
                })
            
            return {
                **operation,
                "status": "success",
                "feeds": feed_list,
                "count": len(feed_list),
                "message": f"Found {len(feed_list)} {operation['scope']} RSS feeds"
            }
            
        except Exception as e:
            logger.error(f"❌ RSS AGENT ERROR: Failed to list RSS feeds: {e}")
            return {
                **operation,
                "status": "error",
                "error_message": str(e)
            }
    
    async def _refresh_rss_feed(self, operation: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Refresh a specific RSS feed"""
        try:
            from services.service_container import get_service_container
            from services.celery_tasks.rss_tasks import poll_rss_feeds_task
            
            service_container = await get_service_container()
            rss_service = service_container.rss_service
            
            # Find the feed by name
            feeds = await rss_service.get_user_feeds(user_id, is_admin=True)
            target_feed = None
            
            for feed in feeds:
                if feed.feed_name.lower() == operation["feed_name"].lower():
                    target_feed = feed
                    break
            
            if not target_feed:
                return {
                    **operation,
                    "status": "error",
                    "error_message": f"RSS feed '{operation['feed_name']}' not found"
                }
            
            # Trigger refresh
            task = poll_rss_feeds_task.delay(
                user_id=user_id,
                feed_ids=[target_feed.feed_id],
                force_poll=True
            )
            
            return {
                **operation,
                "status": "success",
                "feed_id": target_feed.feed_id,
                "task_id": task.id,
                "message": f"Refresh initiated for RSS feed: {target_feed.feed_name}"
            }
            
        except Exception as e:
            logger.error(f"❌ RSS AGENT ERROR: Failed to refresh RSS feed: {e}")
            return {
                **operation,
                "status": "error",
                "error_message": str(e)
            }
    
    def _create_rss_response(self, results: List[Dict[str, Any]], user_message: str) -> Dict[str, Any]:
        """Create structured RSS response from operation results"""
        try:
            # Count different types of results
            feeds_processed = len(results)
            feeds_added = sum(1 for r in results if r.get("status") == "success" and r.get("operation") == "add_feed")
            feeds_listed = sum(1 for r in results if r.get("operation") == "list_feeds")
            errors = [r.get("error_message") for r in results if r.get("status") == "error"]
            
            # Determine overall task status
            has_metadata_required = any(r.get("status") == "metadata_required" for r in results)
            has_errors = any(r.get("status") == "error" for r in results)
            has_success = any(r.get("status") == "success" for r in results)
            
            if has_metadata_required:
                task_status = "incomplete"
                response = "I need additional information to complete your RSS feed request."
            elif has_errors and not has_success:
                task_status = "error"
                response = f"RSS operation failed: {'; '.join(errors)}"
            elif has_success:
                task_status = "complete"
                response = self._build_success_response(results)
            else:
                task_status = "complete"
                response = "RSS operation completed."
            
            # Determine operation type
            operation_types = [r.get("operation") for r in results]
            if "add_feed" in operation_types:
                operation_type = "add_feed"
            elif "list_feeds" in operation_types:
                operation_type = "list_feeds"
            elif "refresh_feed" in operation_types:
                operation_type = "refresh_feed"
            elif "delete_feed" in operation_types:
                operation_type = "delete_feed"
            else:
                operation_type = "none"
            
            return {
                "task_status": task_status,
                "response": response,
                "metadata": {
                    "operation_type": operation_type,
                    "feeds_processed": feeds_processed,
                    "feeds_added": feeds_added,
                    "feeds_listed": feeds_listed,
                    "errors": errors
                },
                "rss_operations": results
            }
            
        except Exception as e:
            logger.error(f"❌ RSS AGENT ERROR: Failed to create response: {e}")
            return {
                "task_status": "error",
                "response": f"Failed to process RSS response: {str(e)}",
                "metadata": {"errors": [str(e)]},
                "rss_operations": []
            }
    
    def _build_success_response(self, results: List[Dict[str, Any]]) -> str:
        """Build success response message from RSS operation results"""
        response_parts = []
        
        for result in results:
            if result.get("status") == "success":
                if result.get("operation") == "add_feed":
                    response_parts.append(f"✅ {result.get('message', 'RSS feed added successfully')}")
                elif result.get("operation") == "list_feeds":
                    response_parts.append(f"📰 {result.get('message', 'RSS feeds listed')}")
                    if "feeds" in result:
                        for feed in result["feeds"][:5]:  # Show first 5 feeds
                            scope_indicator = "🌐" if feed.get("is_global") else "👤"
                            response_parts.append(f"  {scope_indicator} {feed.get('feed_name')} ({feed.get('category')})")
                        if len(result["feeds"]) > 5:
                            response_parts.append(f"  ... and {len(result['feeds']) - 5} more feeds")
                elif result.get("operation") == "refresh_feed":
                    response_parts.append(f"🔄 {result.get('message', 'RSS feed refreshed')}")
        
        if not response_parts:
            response_parts.append("RSS operation completed successfully.")
        
        return "\n".join(response_parts)
    
    async def _update_operations_with_metadata(self, pending_operations: List[Dict[str, Any]], user_message: str) -> List[Dict[str, Any]]:
        """Update RSS operations with metadata from user response"""
        try:
            updated_operations = []
            user_message_lower = user_message.lower()
            
            for operation in pending_operations:
                updated_operation = operation.copy()
                missing_metadata = operation.get("missing_metadata", [])
                
                # Parse title from user response
                if "title" in missing_metadata:
                    title = self._extract_title_from_response(user_message, user_message_lower, operation)
                    if title:
                        updated_operation["feed_name"] = title
                        updated_operation["missing_metadata"].remove("title")
                
                # Parse category from user response
                if "category" in missing_metadata:
                    category = self._extract_category_from_response(user_message, user_message_lower, operation)
                    if category:
                        updated_operation["category"] = category
                        updated_operation["missing_metadata"].remove("category")
                
                # Update status based on remaining missing metadata
                if not updated_operation["missing_metadata"]:
                    updated_operation["status"] = "pending"
                else:
                    updated_operation["status"] = "metadata_required"
                
                updated_operations.append(updated_operation)
            
            return updated_operations
            
        except Exception as e:
            logger.error(f"❌ RSS AGENT ERROR: Failed to update operations with metadata: {e}")
            return pending_operations
    
    def _extract_title_from_response(self, user_message: str, user_message_lower: str, operation: Dict[str, Any]) -> Optional[str]:
        """Extract title from user's metadata response"""
        try:
            # Common title patterns
            title_patterns = [
                r'title:\s*["\']([^"\']+)["\']',
                r'title:\s*([^\n,]+)',
                r'call\s+it\s+["\']([^"\']+)["\']',
                r'name\s+it\s+["\']([^"\']+)["\']',
                r'as\s+["\']([^"\']+)["\']',
                r'["\']([^"\']+)["\']\s+in\s+category',
                r'["\']([^"\']+)["\']\s+for\s+the\s+title'
            ]
            
            for pattern in title_patterns:
                match = re.search(pattern, user_message_lower)
                if match:
                    return match.group(1).strip()
            
            # Check for simple confirmations of suggested titles
            suggested_title = operation.get("suggested_title")
            if suggested_title:
                confirmation_words = ["yes", "ok", "sure", "fine", "good", "use that", "that's fine"]
                if any(word in user_message_lower for word in confirmation_words):
                    return suggested_title
            
            return None
            
        except Exception as e:
            logger.error(f"❌ RSS AGENT ERROR: Failed to extract title: {e}")
            return None
    
    def _extract_category_from_response(self, user_message: str, user_message_lower: str, operation: Dict[str, Any]) -> Optional[str]:
        """Extract category from user's metadata response"""
        try:
            # Common category patterns
            category_patterns = [
                r'category:\s*(\w+)',
                r'in\s+category\s+(\w+)',
                r'put\s+it\s+in\s+(\w+)',
                r'categorize\s+as\s+(\w+)',
                r'["\']([^"\']+)["\']\s+in\s+category'
            ]
            
            for pattern in category_patterns:
                match = re.search(pattern, user_message_lower)
                if match:
                    category = match.group(1).strip().lower()
                    # Validate against available categories
                    available_categories = [
                        "technology", "science", "news", "business", "politics", 
                        "entertainment", "sports", "health", "education", "other"
                    ]
                    if category in available_categories:
                        return category
            
            # Check for simple confirmations of suggested categories
            suggested_category = operation.get("suggested_category")
            if suggested_category:
                confirmation_words = ["yes", "ok", "sure", "fine", "good", "use that", "that's fine"]
                if any(word in user_message_lower for word in confirmation_words):
                    return suggested_category
            
            return None
            
        except Exception as e:
            logger.error(f"❌ RSS AGENT ERROR: Failed to extract category: {e}")
            return None
