"""
Website Crawler Agent
Specialized agent for recursive website crawling, content extraction, and vectorization
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

from services.langgraph_agents.base_agent import BaseAgent
from models.agent_response_models import WebsiteCrawlerResponse

logger = logging.getLogger(__name__)


class WebsiteCrawlerAgent(BaseAgent):
    """
    Agent specialized in recursive website crawling and content ingestion.
    
    **BULLY!** This agent charges forward to capture entire websites!
    
    Capabilities:
    - Recursive crawling with configurable depth
    - Internal link preservation
    - Content extraction and vectorization
    - Progress reporting
    - Automatic folder organization
    """
    
    def __init__(self):
        super().__init__("website_crawler_agent")
        
        self.max_pages_default = 500
        self.max_depth_default = 10
        self.rate_limit = 2.0  # seconds between requests
        
        logger.info("🕷️ Website Crawler Agent initialized")
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process website crawling request
        
        Expected input in state:
        - messages: List of conversation messages
        - shared_memory: Shared context including user_id
        
        Adds to state:
        - agent_results: Structured crawl results
        - is_complete: Task completion status
        """
        try:
            logger.info("🕷️ Website Crawler Agent processing request")
            
            # Extract request details
            latest_message = self._get_latest_user_message(state)
            user_id = state.get("shared_memory", {}).get("user_id")
            
            # Extract URL from request
            url_to_crawl = self._extract_url_from_message(latest_message)
            
            if not url_to_crawl:
                return await self._handle_no_url_found(state)
            
            logger.info(f"🌐 Crawling URL: {url_to_crawl}")
            
            # Import crawler tools
            from services.langgraph_tools.website_crawler_tools import WebsiteCrawlerTools
            
            crawler = WebsiteCrawlerTools()
            
            # Execute recursive crawl
            crawl_result = await crawler.crawl_website_recursive(
                start_url=url_to_crawl,
                max_pages=self.max_pages_default,
                max_depth=self.max_depth_default,
                user_id=user_id
            )
            
            if not crawl_result.get("success"):
                return await self._handle_crawl_failure(state, crawl_result)
            
            # Store crawled content
            storage_result = await self._store_crawled_website(crawl_result, user_id)
            
            # Build response
            return await self._build_success_response(state, crawl_result, storage_result)
            
        except Exception as e:
            logger.error(f"❌ Website Crawler Agent failed: {e}")
            return await self._handle_agent_error(state, str(e))
    
    def _extract_url_from_message(self, message: str) -> Optional[str]:
        """Extract URL from user message"""
        import re
        
        # URL pattern
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        
        matches = re.findall(url_pattern, message)
        
        if matches:
            return matches[0]
        
        return None
    
    async def _store_crawled_website(
        self,
        crawl_result: Dict[str, Any],
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Store crawled website content as documents"""
        try:
            logger.info("📥 Storing crawled website content")
            
            from services.document_service_v2 import DocumentService
            from urllib.parse import urlparse
            
            # Initialize document service
            doc_service = DocumentService()
            await doc_service.initialize()
            
            # Extract website name from URL
            parsed_url = urlparse(crawl_result["start_url"])
            website_name = parsed_url.netloc.replace("www.", "")
            
            crawled_pages = crawl_result.get("crawled_pages", [])
            stored_count = 0
            failed_count = 0
            images_stored = 0
            documents_stored = 0
            
            from pathlib import Path
            from config import settings
            
            for page in crawled_pages:
                try:
                    # Generate document ID
                    doc_id = hashlib.md5(page["url"].encode()).hexdigest()[:16]
                    content_type = page.get("content_type", "html")
                    
                    # Prepare common metadata
                    base_metadata = {
                        "category": "web_crawl",
                        "source_url": page["url"],
                        "site_root": crawl_result["base_domain"],
                        "crawl_session_id": crawl_result["crawl_session_id"],
                        "depth": page["depth"],
                        "parent_url": page.get("parent_url"),
                        "crawl_date": page["crawl_time"],
                        "website_name": website_name,
                        "content_type": content_type
                    }
                    
                    success = False
                    
                    if content_type == "html":
                        # Store HTML page as markdown text document
                        metadata = {
                            **base_metadata,
                            "title": page.get("metadata", {}).get("title", page["url"]),
                            "internal_links": page.get("internal_links", []),
                            "image_links": page.get("image_links", []),
                            "document_links": page.get("document_links", []),
                            **page.get("metadata", {})
                        }
                        
                        path_part = urlparse(page["url"]).path.strip("/") or "index"
                        filename = f"{website_name}_{path_part.replace('/', '_')}.md"
                        content = page["markdown_content"]
                        page_title = page.get("metadata", {}).get("title", page["url"])
                        
                        # Store in vector database for search
                        success = await doc_service.store_text_document(
                            doc_id=doc_id,
                            content=content,
                            metadata=metadata,
                            filename=filename,
                            user_id=user_id,
                            collection_type="user" if user_id else "global"
                        )
                        
                        # ALSO create browseable markdown file using FileManager
                        if success:
                            try:
                                from services.file_manager.agent_helpers import place_web_content
                                await place_web_content(
                                    content=content,
                                    title=page_title,
                                    url=page["url"],
                                    domain=website_name,
                                    user_id=user_id,
                                    tags=["web-crawl", website_name],
                                    description=f"Crawled from {page['url']}"
                                )
                                logger.info(f"✅ Created browseable file for: {page_title}")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to create browseable file for {page['url']}: {e}")
                        
                    elif content_type == "image":
                        # Store image binary file
                        binary_content = page.get("binary_content")
                        filename = page.get("filename", "image")
                        
                        if binary_content:
                            # Save image to uploads directory
                            upload_dir = Path(settings.UPLOAD_DIR) / "web_sources" / "images" / website_name
                            upload_dir.mkdir(parents=True, exist_ok=True)
                            
                            safe_filename = filename.replace("/", "_").replace("\\", "_")
                            file_path = upload_dir / f"{doc_id}_{safe_filename}"
                            
                            with open(file_path, 'wb') as f:
                                f.write(binary_content)
                            
                            logger.info(f"📸 Saved image: {file_path}")
                            
                            # Create metadata entry
                            metadata = {
                                **base_metadata,
                                "title": filename,
                                "file_path": str(file_path),
                                "mime_type": page.get("mime_type"),
                                "size_bytes": page.get("size_bytes", 0)
                            }
                            
                            # Store as text document with reference to image
                            content = f"Image from {page['url']}\n\nLocal path: {file_path}\n\nSource: {website_name}"
                            
                            success = await doc_service.store_text_document(
                                doc_id=doc_id,
                                content=content,
                                metadata=metadata,
                                filename=safe_filename,
                                user_id=user_id,
                                collection_type="user" if user_id else "global"
                            )
                            
                            if success:
                                images_stored += 1
                        
                    elif content_type == "document":
                        # Store document binary file (PDF, DOC, etc.)
                        binary_content = page.get("binary_content")
                        filename = page.get("filename", "document")
                        
                        if binary_content:
                            # Save document to uploads directory
                            upload_dir = Path(settings.UPLOAD_DIR) / "web_sources" / "documents" / website_name
                            upload_dir.mkdir(parents=True, exist_ok=True)
                            
                            safe_filename = filename.replace("/", "_").replace("\\", "_")
                            file_path = upload_dir / f"{doc_id}_{safe_filename}"
                            
                            with open(file_path, 'wb') as f:
                                f.write(binary_content)
                            
                            logger.info(f"📄 Saved document: {file_path}")
                            
                            # Create metadata entry
                            metadata = {
                                **base_metadata,
                                "title": filename,
                                "file_path": str(file_path),
                                "mime_type": page.get("mime_type"),
                                "size_bytes": page.get("size_bytes", 0)
                            }
                            
                            # Store as text document with reference to file
                            content = f"Document from {page['url']}\n\nLocal path: {file_path}\n\nFilename: {filename}\n\nSource: {website_name}"
                            
                            success = await doc_service.store_text_document(
                                doc_id=doc_id,
                                content=content,
                                metadata=metadata,
                                filename=safe_filename,
                                user_id=user_id,
                                collection_type="user" if user_id else "global"
                            )
                            
                            if success:
                                documents_stored += 1
                    
                    if success:
                        stored_count += 1
                    else:
                        failed_count += 1
                        logger.warning(f"⚠️ Failed to store item: {page['url']}")
                    
                except Exception as e:
                    logger.error(f"❌ Error storing item {page.get('url', 'unknown')}: {e}")
                    failed_count += 1
            
            logger.info(f"✅ Stored {stored_count}/{len(crawled_pages)} items ({images_stored} images, {documents_stored} documents)")
            
            return {
                "success": True,
                "stored_count": stored_count,
                "failed_count": failed_count,
                "total_items": len(crawled_pages),
                "images_stored": images_stored,
                "documents_stored": documents_stored
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to store crawled website: {e}")
            return {
                "success": False,
                "error": str(e),
                "stored_count": 0
            }
    
    async def _build_success_response(
        self,
        state: Dict[str, Any],
        crawl_result: Dict[str, Any],
        storage_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build successful response"""
        from urllib.parse import urlparse
        
        elapsed_time = crawl_result.get("elapsed_time_seconds", 0)
        total_items = crawl_result.get("total_items_crawled", 0)
        html_pages = crawl_result.get("html_pages_crawled", 0)
        images = crawl_result.get("images_downloaded", 0)
        documents = crawl_result.get("documents_downloaded", 0)
        items_stored = storage_result.get("stored_count", 0)
        images_stored = storage_result.get("images_stored", 0)
        documents_stored = storage_result.get("documents_stored", 0)
        max_depth = crawl_result.get("max_depth_reached", 0)
        
        # Extract website name for folder path
        parsed_url = urlparse(crawl_result["start_url"])
        website_name = parsed_url.netloc.replace("www.", "")
        
        response_text = f"""**BULLY!** Successfully crawled and ingested the website with all its media!

**Crawl Summary:**
- 🌐 Website: {crawl_result["start_url"]}
- 📄 HTML Pages: {html_pages} crawled, {html_pages} stored
- 📸 Images: {images} downloaded, {images_stored} stored
- 📎 Documents: {documents} downloaded (PDFs, etc.), {documents_stored} stored
- 💾 Total Items: {total_items} crawled, {items_stored} stored
- 📊 Maximum Depth: {max_depth} levels
- ⏱️ Time Taken: {elapsed_time:.1f} seconds

**By George!** The website content, images, and documents have been vectorized and are now searchable! 

**Where to Find Your Content:**
- 📂 **Browseable markdown files**: Navigate to **Documents** page → **Web Sources** folder → **Scraped** → **{website_name}**
- 🔍 **Searchable content**: All pages are in your vector database and can be queried through chat
- 🖼️ **Binary files**: Images and PDFs saved locally for direct access

**Features:**
- HTML pages converted to markdown for easy reading
- Searchable via semantic search in chat
- Images and documents cataloged with metadata
- Internal links between pages preserved"""
        
        if crawl_result.get("total_items_failed", 0) > 0:
            response_text += f"\n\n⚠️ Note: {crawl_result['total_items_failed']} items failed to download."
        
        # Create structured response
        structured_response = WebsiteCrawlerResponse(
            task_status="complete",
            response=response_text,
            website_url=crawl_result["start_url"],
            pages_crawled=total_items,
            pages_stored=items_stored,
            pages_failed=crawl_result.get("total_items_failed", 0),
            max_depth_reached=max_depth,
            crawl_session_id=crawl_result["crawl_session_id"],
            elapsed_time_seconds=elapsed_time,
            metadata={
                "base_domain": crawl_result["base_domain"],
                "html_pages": html_pages,
                "images_downloaded": images,
                "images_stored": images_stored,
                "documents_downloaded": documents,
                "documents_stored": documents_stored,
                "storage_result": storage_result
            }
        )
        
        # Update state
        state["agent_results"] = {
            "structured_response": structured_response.dict(),
            "response_text": response_text,
            "task_status": "complete"
        }
        
        state["is_complete"] = True
        
        # Add assistant message
        from langchain_core.messages import AIMessage
        state["messages"].append(AIMessage(content=response_text))
        
        logger.info("✅ Website Crawler Agent completed successfully")
        
        return state
    
    async def _handle_no_url_found(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle case where no URL was found in message"""
        
        response_text = """**By George!** I couldn't find a website URL in your request.

Please provide a website URL to crawl, for example:
- "Crawl this website: https://example.com"
- "Capture the entire site at https://example.com/docs"

I'll recursively crawl up to 500 pages and 10 levels deep, extracting all content and making it searchable!"""
        
        structured_response = WebsiteCrawlerResponse(
            task_status="incomplete",
            response=response_text,
            website_url=None,
            pages_crawled=0,
            pages_stored=0,
            pages_failed=0,
            max_depth_reached=0,
            crawl_session_id=None,
            elapsed_time_seconds=0,
            metadata={"error": "no_url_found"}
        )
        
        state["agent_results"] = {
            "structured_response": structured_response.dict(),
            "response_text": response_text,
            "task_status": "incomplete"
        }
        
        state["is_complete"] = True
        
        from langchain_core.messages import AIMessage
        state["messages"].append(AIMessage(content=response_text))
        
        return state
    
    async def _handle_crawl_failure(
        self,
        state: Dict[str, Any],
        crawl_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle crawl failure"""
        
        error_msg = crawl_result.get("error", "Unknown error")
        
        response_text = f"""**Blast it!** The website crawl encountered an error:

**Error:** {error_msg}

This could be due to:
- Website blocking automated access
- Network connectivity issues
- Invalid URL format
- Website requiring authentication

Please check the URL and try again, or contact the site administrator if the issue persists."""
        
        structured_response = WebsiteCrawlerResponse(
            task_status="error",
            response=response_text,
            website_url=crawl_result.get("start_url"),
            pages_crawled=0,
            pages_stored=0,
            pages_failed=0,
            max_depth_reached=0,
            crawl_session_id=None,
            elapsed_time_seconds=0,
            metadata={"error": error_msg}
        )
        
        state["agent_results"] = {
            "structured_response": structured_response.dict(),
            "response_text": response_text,
            "task_status": "error"
        }
        
        state["is_complete"] = True
        
        from langchain_core.messages import AIMessage
        state["messages"].append(AIMessage(content=response_text))
        
        return state
    
    async def _handle_agent_error(self, state: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Handle agent-level error"""
        
        response_text = f"""**Confound it!** The Website Crawler Agent encountered an error:

**Error:** {error}

Please try again or contact support if the issue persists."""
        
        structured_response = WebsiteCrawlerResponse(
            task_status="error",
            response=response_text,
            website_url=None,
            pages_crawled=0,
            pages_stored=0,
            pages_failed=0,
            max_depth_reached=0,
            crawl_session_id=None,
            elapsed_time_seconds=0,
            metadata={"agent_error": error}
        )
        
        state["agent_results"] = {
            "structured_response": structured_response.dict(),
            "response_text": response_text,
            "task_status": "error"
        }
        
        state["is_complete"] = True
        
        from langchain_core.messages import AIMessage
        state["messages"].append(AIMessage(content=response_text))
        
        return state

