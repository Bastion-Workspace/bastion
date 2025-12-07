"""
Website Crawler Agent for LLM Orchestrator
Specialized agent for recursive website crawling, content extraction, and vectorization
"""

import logging
import re
from typing import Dict, Any, List, Optional, TypedDict
from urllib.parse import urlparse

from langgraph.graph import StateGraph, END
from .base_agent import BaseAgent, TaskStatus
from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)


class WebsiteCrawlerState(TypedDict):
    """State for website crawler agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    url_to_crawl: str
    crawl_result: Dict[str, Any]
    response: Dict[str, Any]
    task_status: str
    error: str


class WebsiteCrawlerAgent(BaseAgent):
    """
    Agent specialized in recursive website crawling and content ingestion.
    
    Capabilities:
    - Recursive crawling with configurable depth
    - Internal link preservation
    - Content extraction and vectorization
    - Progress reporting
    - Automatic folder organization
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("website_crawler_agent")
        
        self.max_pages_default = 500
        self.max_depth_default = 10
        
        logger.info("Website Crawler Agent initialized")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for website crawler agent"""
        workflow = StateGraph(WebsiteCrawlerState)
        
        # Add nodes
        workflow.add_node("extract_url", self._extract_url_node)
        workflow.add_node("execute_crawl", self._execute_crawl_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("extract_url")
        
        # Linear flow: extract_url -> execute_crawl -> format_response -> END
        workflow.add_edge("extract_url", "execute_crawl")
        workflow.add_edge("execute_crawl", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for website crawling"""
        return """You are the Website Crawler Agent - an expert in recursively crawling and ingesting entire websites!

**MISSION**: Recursively crawl websites, extract all content (HTML, images, documents), and store them in the knowledge base for future reference.

**CAPABILITIES**:
1. **Recursive Crawling**: Follow internal links up to specified depth
2. **Content Extraction**: Extract HTML pages, images, PDFs, and other documents
3. **Vectorization**: Store all content in searchable vector database
4. **Organization**: Automatically organize content by website domain

**STRUCTURED OUTPUT REQUIRED**:
You MUST respond with valid JSON matching this exact schema:

{
  "task_status": "complete|incomplete|error",
  "response": "Your formatted natural language response with crawl summary",
  "website_url": "https://example.com",
  "pages_crawled": 42,
  "pages_stored": 40,
  "pages_failed": 2,
  "max_depth_reached": 5,
  "crawl_session_id": "abc123",
  "elapsed_time_seconds": 123.45,
  "metadata": {
    "base_domain": "example.com",
    "html_pages": 35,
    "images_downloaded": 5,
    "images_stored": 5,
    "documents_downloaded": 2,
    "documents_stored": 2
  }
}

**RESPONSE GUIDELINES**:
- Provide clear summary of crawl results
- Include statistics: pages crawled, stored, failed
- Mention where content can be found (Documents page)
- Use markdown formatting for better readability
- Be enthusiastic about successful crawls!

**IMPORTANT**: 
- Extract URL from user message
- If no URL found, ask user to provide one
- Default to 500 pages max, 10 depth max
- Report any errors clearly
"""
    
    async def _extract_url_node(self, state: WebsiteCrawlerState) -> Dict[str, Any]:
        """Extract URL from query"""
        try:
            logger.info("🔍 Extracting URL from query...")
            
            query = state.get("query", "")
            url_to_crawl = self._extract_url_from_message(query)
            
            if not url_to_crawl:
                return {
                    "url_to_crawl": "",
                    "response": self._create_no_url_response(),
                    "task_status": "incomplete"
                }
            
            logger.info(f"Crawling URL: {url_to_crawl}")
            
            return {
                "url_to_crawl": url_to_crawl
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to extract URL: {e}")
            return {
                "url_to_crawl": "",
                "error": str(e),
                "task_status": "error"
            }
    
    async def _execute_crawl_node(self, state: WebsiteCrawlerState) -> Dict[str, Any]:
        """Execute website crawl"""
        try:
            logger.info("🕷️ Executing website crawl...")
            
            url_to_crawl = state.get("url_to_crawl", "")
            user_id = state.get("user_id", "")
            
            if not url_to_crawl:
                return {
                    "crawl_result": {},
                    "task_status": "incomplete"
                }
            
            # Get backend tool client
            tool_client = await get_backend_tool_client()
            
            # Execute recursive crawl via gRPC
            crawl_result = await tool_client.crawl_website_recursive(
                start_url=url_to_crawl,
                max_pages=self.max_pages_default,
                max_depth=self.max_depth_default,
                user_id=user_id
            )
            
            return {
                "crawl_result": crawl_result
            }
            
        except Exception as e:
            logger.error(f"❌ Crawl execution failed: {e}")
            return {
                "crawl_result": {"success": False, "error": str(e)},
                "error": str(e),
                "task_status": "error"
            }
    
    async def _format_response_node(self, state: WebsiteCrawlerState) -> Dict[str, Any]:
        """Format response from crawl results"""
        try:
            logger.info("📝 Formatting crawl response...")
            
            url_to_crawl = state.get("url_to_crawl", "")
            crawl_result = state.get("crawl_result", {})
            
            if not url_to_crawl:
                # Already handled in extract_url_node
                response = state.get("response")
                return {
                    "response": response,
                    "task_status": "incomplete"
                }
            
            if not crawl_result.get("success"):
                response = self._create_crawl_failure_response(crawl_result, url_to_crawl)
                return {
                    "response": response,
                    "task_status": "error"
                }
            
            # Build success response
            response = self._create_success_response(crawl_result)
            
            logger.info("✅ Website crawl completed successfully")
            
            return {
                "response": response,
                "task_status": "complete"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to format response: {e}")
            return {
                "response": self._create_error_response(str(e)),
                "task_status": "error",
                "error": str(e)
            }
    
    async def process(
        self,
        query: str = None,
        user_id: str = None,
        conversation_history: List[Dict[str, str]] = None,
        state: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process website crawling request using LangGraph workflow
        
        Supports both legacy signature (query, user_id, conversation_history) 
        and new state dict signature for compatibility
        
        Args:
            query: User query containing URL to crawl (legacy)
            user_id: User ID (legacy)
            conversation_history: Previous conversation messages (legacy)
            state: Dictionary with messages, shared_memory, user_id, etc. (new)
            
        Returns:
            Dict with structured response and task status
        """
        try:
            # Support both legacy and new signatures
            if state is not None:
                # New state dict signature
                messages = state.get("messages", [])
                if messages:
                    latest_message = messages[-1]
                    query = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
                else:
                    query = state.get("query", "")
                user_id = state.get("user_id", "system")
            else:
                # Legacy signature - convert to state dict format
                if query is None:
                    query = ""
                if user_id is None:
                    user_id = "system"
                state = {
                    "messages": [],
                    "user_id": user_id,
                    "shared_memory": {}
                }
            
            logger.info(f"Website Crawler Agent processing: {query[:80]}...")
            
            metadata = state.get("metadata", {})
            shared_memory = state.get("shared_memory", {})
            
            # Get workflow (lazy initialization with checkpointer)
            workflow = await self._get_workflow()
            
            # Get checkpoint config (handles thread_id from conversation_id/user_id)
            config = self._get_checkpoint_config(metadata)
            
            # Prepare new messages (current query)
            messages = state.get("messages", [])
            new_messages = self._prepare_messages_with_query(messages, query)
            
            # Load and merge checkpointed messages to preserve conversation history
            conversation_messages = await self._load_and_merge_checkpoint_messages(
                workflow, config, new_messages
            )
            
            # Load shared_memory from checkpoint if available
            checkpoint_state = await workflow.aget_state(config)
            existing_shared_memory = {}
            if checkpoint_state and checkpoint_state.values:
                existing_shared_memory = checkpoint_state.values.get("shared_memory", {})
            
            # Merge with any shared_memory from metadata
            shared_memory_merged = shared_memory.copy()
            shared_memory_merged.update(existing_shared_memory)
            
            # Build initial state for LangGraph workflow
            initial_state: WebsiteCrawlerState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory_merged,
                "url_to_crawl": "",
                "crawl_result": {},
                "response": {},
                "task_status": "",
                "error": ""
            }
            
            # Invoke LangGraph workflow with checkpointing
            final_state = await workflow.ainvoke(initial_state, config=config)
            
            # Return response from final state
            return final_state.get("response", {
                "task_status": TaskStatus.ERROR.value,
                "response": "Website crawl failed",
                "website_url": None,
                "pages_crawled": 0,
                "pages_stored": 0,
                "pages_failed": 0,
                "max_depth_reached": 0,
                "crawl_session_id": None,
                "elapsed_time_seconds": 0,
                "metadata": {"error": "unknown"}
            })
            
        except Exception as e:
            logger.error(f"Website Crawler Agent failed: {e}")
            return self._create_error_response(str(e))
    
    def _extract_url_from_message(self, message: str) -> Optional[str]:
        """Extract URL from user message"""
        # URL pattern
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        
        matches = re.findall(url_pattern, message)
        
        if matches:
            return matches[0]
        
        return None
    
    def _create_success_response(self, crawl_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create successful crawl response"""
        from urllib.parse import urlparse
        
        elapsed_time = crawl_result.get("elapsed_time_seconds", 0)
        total_items = crawl_result.get("total_items_crawled", 0)
        html_pages = crawl_result.get("html_pages_crawled", 0)
        images = crawl_result.get("images_downloaded", 0)
        documents = crawl_result.get("documents_downloaded", 0)
        max_depth = crawl_result.get("max_depth_reached", 0)
        failed_items = crawl_result.get("total_items_failed", 0)
        
        # Extract website name for folder path
        parsed_url = urlparse(crawl_result.get("start_url", ""))
        website_name = parsed_url.netloc.replace("www.", "")
        
        response_text = f"""Successfully crawled and ingested the website with all its media!

**Crawl Summary:**
- 🌐 Website: {crawl_result.get("start_url", "")}
- 📄 HTML Pages: {html_pages} crawled and stored
- 📸 Images: {images} downloaded and stored
- 📎 Documents: {documents} downloaded and stored (PDFs, etc.)
- 💾 Total Items: {total_items} crawled
- 📊 Maximum Depth: {max_depth} levels
- ⏱️ Time Taken: {elapsed_time:.1f} seconds

The website content, images, and documents have been vectorized and are now searchable!

**Where to Find Your Content:**
- 📂 **Browseable markdown files**: Navigate to **Documents** page → **Web Sources** folder → **Scraped** → **{website_name}**
- 🔍 **Searchable content**: All pages are in your vector database and can be queried through chat
- 🖼️ **Binary files**: Images and PDFs saved locally for direct access

**Features:**
- HTML pages converted to markdown for easy reading
- Searchable via semantic search in chat
- Images and documents cataloged with metadata
- Internal links between pages preserved"""
        
        if failed_items > 0:
            response_text += f"\n\n⚠️ Note: {failed_items} items failed to download."
        
        return {
            "task_status": TaskStatus.COMPLETE.value,
            "response": response_text,
            "website_url": crawl_result.get("start_url"),
            "pages_crawled": total_items,
            "pages_stored": total_items - failed_items,
            "pages_failed": failed_items,
            "max_depth_reached": max_depth,
            "crawl_session_id": crawl_result.get("crawl_session_id"),
            "elapsed_time_seconds": elapsed_time,
            "metadata": {
                "base_domain": crawl_result.get("base_domain", ""),
                "html_pages": html_pages,
                "images_downloaded": images,
                "images_stored": images,
                "documents_downloaded": documents,
                "documents_stored": documents
            }
        }
    
    def _create_no_url_response(self) -> Dict[str, Any]:
        """Create response when no URL found"""
        response_text = """I couldn't find a website URL in your request.

Please provide a website URL to crawl, for example:
- "Crawl this website: https://example.com"
- "Capture the entire site at https://example.com/docs"

I'll recursively crawl up to 500 pages and 10 levels deep, extracting all content and making it searchable!"""
        
        return {
            "task_status": TaskStatus.INCOMPLETE.value,
            "response": response_text,
            "website_url": None,
            "pages_crawled": 0,
            "pages_stored": 0,
            "pages_failed": 0,
            "max_depth_reached": 0,
            "crawl_session_id": None,
            "elapsed_time_seconds": 0,
            "metadata": {"error": "no_url_found"}
        }
    
    def _create_crawl_failure_response(
        self,
        crawl_result: Dict[str, Any],
        url: str
    ) -> Dict[str, Any]:
        """Create response for crawl failure"""
        error_msg = crawl_result.get("error", "Unknown error")
        
        response_text = f"""The website crawl encountered an error:

**Error:** {error_msg}

This could be due to:
- Website blocking automated access
- Network connectivity issues
- Invalid URL format
- Website requiring authentication

Please check the URL and try again, or contact the site administrator if the issue persists."""
        
        return {
            "task_status": TaskStatus.ERROR.value,
            "response": response_text,
            "website_url": url,
            "pages_crawled": 0,
            "pages_stored": 0,
            "pages_failed": 0,
            "max_depth_reached": 0,
            "crawl_session_id": None,
            "elapsed_time_seconds": 0,
            "metadata": {"error": error_msg}
        }
    
    def _create_error_response(self, error: str) -> Dict[str, Any]:
        """Create response for agent-level error"""
        response_text = f"""The Website Crawler Agent encountered an error:

**Error:** {error}

Please try again or contact support if the issue persists."""
        
        return {
            "task_status": TaskStatus.ERROR.value,
            "response": response_text,
            "website_url": None,
            "pages_crawled": 0,
            "pages_stored": 0,
            "pages_failed": 0,
            "max_depth_reached": 0,
            "crawl_session_id": None,
            "elapsed_time_seconds": 0,
            "metadata": {"agent_error": error}
        }


def get_website_crawler_agent() -> WebsiteCrawlerAgent:
    """Get WebsiteCrawlerAgent instance"""
    return WebsiteCrawlerAgent()




