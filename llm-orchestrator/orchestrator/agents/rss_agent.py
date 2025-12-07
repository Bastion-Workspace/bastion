"""
RSS Agent for Chat Interactions
LangGraph agent for RSS feed management through natural language commands
"""

import logging
import re
import uuid
from typing import Dict, Any, List, Optional, TypedDict
from urllib.parse import urlparse
from langchain_core.messages import AIMessage

from langgraph.graph import StateGraph, END
from orchestrator.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RSSState(TypedDict):
    """State for RSS agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    user_message: str
    pending_operations: List[Dict[str, Any]]
    rss_operations: List[Dict[str, Any]]
    execution_results: List[Dict[str, Any]]
    response: Dict[str, Any]
    task_status: str
    error: str


class RSSAgent(BaseAgent):
    """
    RSS Agent for chat-based RSS feed management
    
    Handles RSS feed operations through natural language commands
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("rss_agent")
        self._grpc_client = None
        logger.info("📰 RSS Agent ready!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for RSS agent"""
        workflow = StateGraph(RSSState)
        
        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("parse_or_update_operations", self._parse_or_update_operations_node)
        workflow.add_node("execute_operations", self._execute_operations_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Linear flow: prepare_context -> parse_or_update_operations -> execute_operations -> format_response -> END
        workflow.add_edge("prepare_context", "parse_or_update_operations")
        workflow.add_edge("parse_or_update_operations", "execute_operations")
        workflow.add_edge("execute_operations", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    async def _get_grpc_client(self):
        """Get or create gRPC client for backend tools"""
        if self._grpc_client is None:
            from orchestrator.clients.backend_tool_client import get_backend_tool_client
            self._grpc_client = await get_backend_tool_client()
        return self._grpc_client
    
    async def _prepare_context_node(self, state: RSSState) -> Dict[str, Any]:
        """Prepare context: extract message and check for pending operations"""
        try:
            logger.info("📋 Preparing context for RSS management...")
            
            messages = state.get("messages", [])
            if not messages:
                return {
                    "error": "No user message found for RSS processing",
                    "task_status": "error"
                }
            
            # Get the latest user message
            latest_message = messages[-1]
            user_message = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
            
            # Check if this is a metadata response for pending RSS operations
            shared_memory = state.get("shared_memory", {})
            rss_metadata_request = shared_memory.get("rss_metadata_request", {})
            pending_operations = rss_metadata_request.get("pending_operations", [])
            
            logger.info(f"📰 RSS Agent: Processing message: {user_message[:100]}...")
            
            return {
                "user_message": user_message,
                "pending_operations": pending_operations or []
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare context: {e}")
            return {
                "user_message": "",
                "pending_operations": [],
                "error": str(e),
                "task_status": "error"
            }
    
    async def _parse_or_update_operations_node(self, state: RSSState) -> Dict[str, Any]:
        """Parse RSS commands or update operations with metadata"""
        try:
            logger.info("📝 Parsing or updating RSS operations...")
            
            user_message = state.get("user_message", "")
            user_id = state.get("user_id", "")
            pending_operations = state.get("pending_operations", [])
            
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
                return {
                    "rss_operations": [],
                    "task_status": "complete"
                }
            
            return {
                "rss_operations": rss_operations
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to parse/update operations: {e}")
            return {
                "rss_operations": [],
                "error": str(e),
                "task_status": "error"
            }
    
    async def _execute_operations_node(self, state: RSSState) -> Dict[str, Any]:
        """Execute RSS operations"""
        try:
            logger.info("⚙️ Executing RSS operations...")
            
            rss_operations = state.get("rss_operations", [])
            user_id = state.get("user_id", "")
            
            if not rss_operations:
                return {
                    "execution_results": [],
                    "task_status": "complete"
                }
            
            # Execute RSS operations
            results = await self._execute_rss_operations(rss_operations, user_id)
            
            return {
                "execution_results": results
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to execute operations: {e}")
            return {
                "execution_results": [],
                "error": str(e),
                "task_status": "error"
            }
    
    async def _format_response_node(self, state: RSSState) -> Dict[str, Any]:
        """Format response from execution results"""
        try:
            logger.info("📝 Formatting RSS response...")
            
            rss_operations = state.get("rss_operations", [])
            execution_results = state.get("execution_results", [])
            user_message = state.get("user_message", "")
            
            if not rss_operations:
                response = self._create_response(
                    success=True,
                    response="No RSS operations detected in your message. Try commands like 'Add RSS feed: <url>' or 'List my RSS feeds'."
                )
            else:
            # Create structured response
                response = self._create_rss_response(execution_results, user_message)
            
            logger.info("✅ RSS Agent: Completed RSS management processing")
            
            return {
                "response": response,
                "task_status": "complete"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to format response: {e}")
            return {
                "response": self._create_error_result(f"RSS management failed: {str(e)}"),
                "task_status": "error",
                "error": str(e)
            }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """
        Process RSS management commands using LangGraph workflow
        
        Args:
            query: User query string
            metadata: Optional metadata dictionary (persona, editor context, etc.)
            messages: Optional conversation history
            
        Returns:
            Dictionary with RSS management response and metadata
        """
        try:
            logger.info(f"📰 RSS Agent: Starting RSS management processing: {query[:100]}...")
            
            # Extract user_id and shared_memory from metadata
            metadata = metadata or {}
            user_id = metadata.get("user_id", "system")
            shared_memory = metadata.get("shared_memory", {}) or {}
            
            # Prepare new messages (current query)
            new_messages = self._prepare_messages_with_query(messages, query)
            
            # Get workflow to access checkpoint
            workflow = await self._get_workflow()
            config = self._get_checkpoint_config(metadata)
            
            # Load and merge checkpointed messages to preserve conversation history
            conversation_messages = await self._load_and_merge_checkpoint_messages(
                workflow, config, new_messages
            )
            
            # Load shared_memory from checkpoint if available
            checkpoint_state = await workflow.aget_state(config)
            existing_shared_memory = {}
            if checkpoint_state and checkpoint_state.values:
                existing_shared_memory = checkpoint_state.values.get("shared_memory", {})
            
            # Merge with any shared_memory from incoming metadata
            shared_memory_merged = shared_memory.copy()
            shared_memory_merged.update(existing_shared_memory)
            
            # Build initial state for LangGraph workflow
            initial_state: RSSState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory_merged,
                "user_message": "",
                "pending_operations": [],
                "rss_operations": [],
                "execution_results": [],
                "response": {},
                "task_status": "",
                "error": ""
            }
            
            # Invoke LangGraph workflow with checkpointing
            final_state = await workflow.ainvoke(initial_state, config=config)
            
            # Return response from final state
            return final_state.get("response", {
                "messages": [AIMessage(content="RSS management failed")],
                "agent_results": {
                    "agent_type": self.agent_type,
                    "is_complete": False
                },
                "is_complete": False
            })
            
        except Exception as e:
            logger.error(f"❌ RSS Agent ERROR: {e}")
            return self._create_error_result(f"RSS management failed: {str(e)}")
    
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
                "scope": "user",
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
        """Execute RSS operations via gRPC"""
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
                logger.error(f"❌ RSS Agent ERROR: Failed to execute operation {operation}: {e}")
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
            logger.error(f"❌ RSS Agent ERROR: Failed to create metadata request: {e}")
            return {
                **operation,
                "status": "error",
                "error_message": str(e)
            }
    
    async def _add_rss_feed(self, operation: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Add a new RSS feed via gRPC"""
        try:
            grpc_client = await self._get_grpc_client()
            
            result = await grpc_client.add_rss_feed(
                user_id=user_id,
                feed_url=operation["feed_url"],
                feed_name=operation["feed_name"],
                category=operation["category"],
                is_global=(operation["scope"] == "global")
            )
            
            if result["success"]:
                return {
                    **operation,
                    "status": "success",
                    "feed_id": result["feed_id"],
                    "message": result["message"]
                }
            else:
                return {
                    **operation,
                    "status": "error",
                    "error_message": result.get("error", "Unknown error")
                }
            
        except Exception as e:
            logger.error(f"❌ RSS Agent ERROR: Failed to add RSS feed: {e}")
            return {
                **operation,
                "status": "error",
                "error_message": str(e)
            }
    
    async def _list_rss_feeds(self, operation: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """List RSS feeds via gRPC"""
        try:
            grpc_client = await self._get_grpc_client()
            
            result = await grpc_client.list_rss_feeds(
                user_id=user_id,
                scope=operation["scope"]
            )
            
            if result["success"]:
                return {
                    **operation,
                    "status": "success",
                    "feeds": result["feeds"],
                    "count": result["count"],
                    "message": f"Found {result['count']} {operation['scope']} RSS feeds"
                }
            else:
                return {
                    **operation,
                    "status": "error",
                    "error_message": result.get("error", "Unknown error")
                }
            
        except Exception as e:
            logger.error(f"❌ RSS Agent ERROR: Failed to list RSS feeds: {e}")
            return {
                **operation,
                "status": "error",
                "error_message": str(e)
            }
    
    async def _refresh_rss_feed(self, operation: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Refresh a specific RSS feed via gRPC"""
        try:
            grpc_client = await self._get_grpc_client()
            
            result = await grpc_client.refresh_rss_feed(
                user_id=user_id,
                feed_name=operation["feed_name"]
            )
            
            if result["success"]:
                return {
                    **operation,
                    "status": "success",
                    "feed_id": result["feed_id"],
                    "task_id": result["task_id"],
                    "message": result["message"]
                }
            else:
                return {
                    **operation,
                    "status": "error",
                    "error_message": result.get("error", "Unknown error")
                }
            
        except Exception as e:
            logger.error(f"❌ RSS Agent ERROR: Failed to refresh RSS feed: {e}")
            return {
                **operation,
                "status": "error",
                "error_message": str(e)
            }
    
    def _create_rss_response(self, results: List[Dict[str, Any]], user_message: str) -> Dict[str, Any]:
        """Create structured RSS response from operation results"""
        try:
            # Count different types of results
            feeds_added = sum(1 for r in results if r.get("status") == "success" and r.get("operation") == "add_feed")
            feeds_listed = sum(1 for r in results if r.get("operation") == "list_feeds")
            errors = [r.get("error_message") for r in results if r.get("status") == "error"]
            
            # Determine overall task status
            has_metadata_required = any(r.get("status") == "metadata_required" for r in results)
            has_errors = any(r.get("status") == "error" for r in results)
            has_success = any(r.get("status") == "success" for r in results)
            
            if has_metadata_required:
                response = "I need additional information to complete your RSS feed request."
            elif has_errors and not has_success:
                response = f"RSS operation failed: {'; '.join(errors)}"
            elif has_success:
                response = self._build_success_response(results)
            else:
                response = "RSS operation completed."
            
            return self._create_response(
                success=not (has_errors and not has_success),
                response=response,
                rss_operations=results
            )
            
        except Exception as e:
            logger.error(f"❌ RSS Agent ERROR: Failed to create response: {e}")
            return self._create_error_result(f"Failed to process RSS response: {str(e)}")
    
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
            logger.error(f"❌ RSS Agent ERROR: Failed to update operations with metadata: {e}")
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
            logger.error(f"❌ RSS Agent ERROR: Failed to extract title: {e}")
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
            logger.error(f"❌ RSS Agent ERROR: Failed to extract category: {e}")
            return None
    
    def _create_response(self, success: bool, response: str, rss_operations: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create standardized response"""
        return {
            "messages": [AIMessage(content=response)],
            "agent_results": {
                "agent_type": "rss_agent",
                "success": success,
                "rss_operations": rss_operations or [],
                "is_complete": True
            },
            "is_complete": True
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        logger.error(f"❌ RSS Agent error: {error_message}")
        return {
            "messages": [AIMessage(content=f"RSS management failed: {error_message}")],
            "agent_results": {
                "agent_type": "rss_agent",
                "success": False,
                "error": error_message,
                "is_complete": True
            },
            "is_complete": True
        }


# Singleton instance
_rss_agent_instance = None


def get_rss_agent() -> RSSAgent:
    """Get global RSS agent instance"""
    global _rss_agent_instance
    if _rss_agent_instance is None:
        _rss_agent_instance = RSSAgent()
    return _rss_agent_instance

