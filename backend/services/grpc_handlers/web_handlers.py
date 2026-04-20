"""gRPC handlers for Web search, crawling, and browser automation operations."""

import logging
from typing import Any, Dict, Optional

import grpc
from protos import tool_service_pb2

logger = logging.getLogger(__name__)


class WebHandlersMixin:
    """Mixin providing Web gRPC handlers.

    Mixed into ToolServiceImplementation; accesses self methods
    via standard Python MRO.
    """

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
            search_response = await search_web(query=request.query, limit=request.max_results or 15)
            
            # Parse results - search_web returns a dict with "results" key containing list
            response = tool_service_pb2.WebSearchResponse()
            
            # Extract results list from response dict
            if isinstance(search_response, dict) and search_response.get("success"):
                results_list = search_response.get("results", [])
                if isinstance(results_list, list):
                    for result in results_list[:request.max_results or 15]:
                        web_result = tool_service_pb2.WebSearchResult(
                            title=result.get('title', ''),
                            url=result.get('url', ''),
                            snippet=result.get('snippet', ''),
                            relevance_score=float(result.get('relevance_score', 0.0))
                        )
                        response.results.append(web_result)
            elif isinstance(search_response, list):
                # Fallback: if it's already a list (legacy format)
                for result in search_response[:request.max_results or 15]:
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
        """Crawl web content from URLs, with optional pagination."""
        try:
            # Import crawl tool
            from services.langgraph_tools.crawl4ai_web_tools import crawl_web_content

            paginate = request.paginate if request.HasField("paginate") else False
            max_pages = request.max_pages if request.HasField("max_pages") else 10
            pagination_param = request.pagination_param if request.HasField("pagination_param") else None
            start_page = request.start_page if request.HasField("start_page") else 0
            next_page_css_selector = request.next_page_css_selector if request.HasField("next_page_css_selector") else None
            css_selector = request.css_selector if request.HasField("css_selector") else None
            max_urls = request.max_urls if request.HasField("max_urls") and request.max_urls > 0 else 5

            kwargs = {
                "url": request.url if request.url else None,
                "urls": list(request.urls) if request.urls else None,
                "user_id": request.user_id or "system",
                "css_selector": css_selector,
                "paginate": paginate,
                "max_pages": max_pages,
                "pagination_param": pagination_param,
                "start_page": start_page,
                "next_page_css_selector": next_page_css_selector,
                "max_urls": max_urls,
            }
            result = await crawl_web_content(**kwargs)

            response = tool_service_pb2.WebCrawlResponse()

            if isinstance(result, dict) and "results" in result:
                for item in result["results"]:
                    if not item.get("success"):
                        continue

                    metadata = item.get("metadata", {})
                    title = metadata.get("title", "") if isinstance(metadata, dict) else ""
                    content = item.get("full_content", "") or item.get("content", "")
                    html = item.get("html", "")

                    crawl_result = tool_service_pb2.WebCrawlResult(
                        url=item.get("url", ""),
                        title=title,
                        content=content,
                        html=html
                    )
                    if isinstance(metadata, dict):
                        for key, value in metadata.items():
                            crawl_result.metadata[str(key)] = str(value)
                    for img in item.get("images", [])[:20]:
                        crawl_result.images.append(img)
                    for link in item.get("links", [])[:50]:
                        crawl_result.links.append(link)

                    response.results.append(crawl_result)

            logger.info(f"CrawlWebContent: Crawled {len(response.results)} URLs")
            return response

        except Exception as e:
            logger.error(f"CrawlWebContent error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Web crawl failed: {str(e)}")
    
    async def CrawlWebsiteRecursive(
        self,
        request: tool_service_pb2.RecursiveWebsiteCrawlRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.RecursiveWebsiteCrawlResponse:
        """Recursively crawl entire website"""
        try:
            logger.info(f"CrawlWebsiteRecursive: {request.start_url}, max_pages={request.max_pages}, max_depth={request.max_depth}")
            
            # Import recursive crawler tool
            from services.langgraph_tools.website_crawler_tools import WebsiteCrawlerTools
            
            crawler = WebsiteCrawlerTools()
            
            # Execute recursive crawl
            crawl_result = await crawler.crawl_website_recursive(
                start_url=request.start_url,
                max_pages=request.max_pages if request.max_pages > 0 else 500,
                max_depth=request.max_depth if request.max_depth > 0 else 10,
                user_id=request.user_id if request.user_id else None
            )
            
            # Store crawled content (same as backend agent does)
            if crawl_result.get("success"):
                try:
                    storage_result = await self._store_crawled_website(crawl_result, request.user_id if request.user_id else None)
                    logger.info(f"CrawlWebsiteRecursive: Stored {storage_result.get('stored_count', 0)} items")
                except Exception as e:
                    logger.warning(f"CrawlWebsiteRecursive: Storage failed: {e}, but crawl succeeded")
            
            # Build response
            response = tool_service_pb2.RecursiveWebsiteCrawlResponse()
            
            if crawl_result.get("success"):
                response.success = True
                response.start_url = crawl_result.get("start_url", "")
                response.base_domain = crawl_result.get("base_domain", "")
                response.crawl_session_id = crawl_result.get("crawl_session_id", "")
                response.total_items_crawled = crawl_result.get("total_items_crawled", 0)
                response.html_pages_crawled = crawl_result.get("html_pages_crawled", 0)
                response.images_downloaded = crawl_result.get("images_downloaded", 0)
                response.documents_downloaded = crawl_result.get("documents_downloaded", 0)
                response.total_items_failed = crawl_result.get("total_items_failed", 0)
                response.max_depth_reached = crawl_result.get("max_depth_reached", 0)
                response.elapsed_time_seconds = crawl_result.get("elapsed_time_seconds", 0.0)
                
                # Add crawled pages
                crawled_pages = crawl_result.get("crawled_pages", [])
                for page in crawled_pages:
                    crawled_page = tool_service_pb2.CrawledPage()
                    crawled_page.url = page.get("url", "")
                    crawled_page.content_type = page.get("content_type", "html")
                    crawled_page.markdown_content = page.get("markdown_content", "")
                    crawled_page.html_content = page.get("html_content", "")
                    
                    # Add metadata
                    if page.get("metadata"):
                        for key, value in page["metadata"].items():
                            crawled_page.metadata[str(key)] = str(value)
                    
                    # Add links
                    crawled_page.internal_links.extend(page.get("internal_links", []))
                    crawled_page.image_links.extend(page.get("image_links", []))
                    crawled_page.document_links.extend(page.get("document_links", []))
                    
                    crawled_page.depth = page.get("depth", 0)
                    if page.get("parent_url"):
                        crawled_page.parent_url = page["parent_url"]
                    crawled_page.crawl_time = page.get("crawl_time", "")
                    
                    # Add binary content for images/documents
                    if page.get("binary_content"):
                        crawled_page.binary_content = page["binary_content"]
                    if page.get("filename"):
                        crawled_page.filename = page["filename"]
                    if page.get("mime_type"):
                        crawled_page.mime_type = page["mime_type"]
                    if page.get("size_bytes"):
                        crawled_page.size_bytes = page["size_bytes"]
                    
                    response.crawled_pages.append(crawled_page)
            else:
                response.success = False
                error_msg = crawl_result.get("error", "Unknown error")
                response.error = error_msg
            
            logger.info(f"CrawlWebsiteRecursive: Success={response.success}, Pages={response.total_items_crawled}")
            return response
            
        except Exception as e:
            logger.error(f"CrawlWebsiteRecursive error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Recursive website crawl failed: {str(e)}")
    
    async def CrawlSite(
        self,
        request: tool_service_pb2.DomainCrawlRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DomainCrawlResponse:
        """Domain-scoped crawl starting from seed URL, filtering by query criteria"""
        try:
            logger.info(f"CrawlSite: {request.seed_url}, query={request.query_criteria}, max_pages={request.max_pages}, max_depth={request.max_depth}")
            
            # Import domain-scoped crawler tool
            from services.langgraph_tools.crawl4ai_web_tools import Crawl4AIWebTools
            
            crawler = Crawl4AIWebTools()
            
            # Execute domain-scoped crawl
            crawl_result = await crawler.crawl_site(
                seed_url=request.seed_url,
                query_criteria=request.query_criteria,
                max_pages=request.max_pages if request.max_pages > 0 else 50,
                max_depth=request.max_depth if request.max_depth > 0 else 2,
                allowed_path_prefix=request.allowed_path_prefix if request.allowed_path_prefix else None,
                include_pdfs=request.include_pdfs,
                user_id=request.user_id if request.user_id else None
            )
            
            # Build response
            response = tool_service_pb2.DomainCrawlResponse()
            
            if crawl_result.get("success"):
                response.success = True
                response.domain = crawl_result.get("domain", "")
                response.successful_crawls = crawl_result.get("successful_crawls", 0)
                response.urls_considered = crawl_result.get("urls_considered", 0)
                
                # Add crawl results
                results = crawl_result.get("results", [])
                for item in results:
                    result = tool_service_pb2.DomainCrawlResult()
                    result.url = item.get("url", "")
                    result.title = ((item.get("metadata") or {}).get("title") or "No title").strip()
                    result.full_content = item.get("full_content", "")
                    result.relevance_score = item.get("relevance_score", 0.0)
                    result.success = item.get("success", False)
                    
                    # Add metadata
                    if item.get("metadata"):
                        for key, value in item["metadata"].items():
                            result.metadata[str(key)] = str(value)
                    
                    response.results.append(result)
            else:
                response.success = False
                response.error = crawl_result.get("error", "Unknown error")
            
            return response
            
        except Exception as e:
            logger.error(f"CrawlSite error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Domain crawl failed: {str(e)}")

    async def BrowserRun(
        self,
        request: tool_service_pb2.BrowserRunToolRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserRunToolResponse:
        """Run Playwright browser session: steps then final action (download, click, extract, screenshot)."""
        try:
            from services.langgraph_tools.playwright_browser_tools import browser_run
            steps = []
            for s in request.steps:
                steps.append({
                    "action": s.action or "",
                    "selector": s.selector if s.HasField("selector") else None,
                    "value": s.value if s.HasField("value") else None,
                    "wait_for": s.wait_for if s.HasField("wait_for") else None,
                    "timeout_seconds": s.timeout_seconds if s.HasField("timeout_seconds") else None,
                    "url": s.url if s.HasField("url") else None,
                })
            result = await browser_run(
                user_id=request.user_id or "system",
                url=request.url or "",
                final_action_type=request.final_action_type or "download",
                final_selector=request.final_selector or "",
                folder_path=request.folder_path or "",
                steps=steps if steps else None,
                connection_id=request.connection_id if request.HasField("connection_id") and request.connection_id else None,
                tags=list(request.tags) if request.tags else None,
                title=request.title if request.HasField("title") and request.title else None,
                goal=request.goal if request.HasField("goal") and request.goal else None,
            )
            response = tool_service_pb2.BrowserRunToolResponse()
            response.success = result.get("success", False)
            if result.get("error"):
                response.error = result["error"]
            if result.get("document_id") is not None:
                response.document_id = result["document_id"]
            if result.get("filename") is not None:
                response.filename = result["filename"]
            if result.get("file_size_bytes") is not None:
                response.file_size_bytes = result["file_size_bytes"]
            if result.get("extracted_text") is not None:
                response.extracted_text = result["extracted_text"]
            if result.get("message") is not None:
                response.message = result["message"]
            if result.get("images_markdown") is not None:
                response.images_markdown = result["images_markdown"]
            return response
        except Exception as e:
            logger.error(f"BrowserRun error: {e}")
            response = tool_service_pb2.BrowserRunToolResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserDownload(
        self,
        request: tool_service_pb2.BrowserDownloadToolRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserDownloadToolResponse:
        """Run Playwright browser session: optional steps, trigger download, save file to user folder. Delegates to BrowserRun with final_action_type=download."""
        try:
            from services.langgraph_tools.playwright_browser_tools import browser_run
            steps = []
            for s in request.steps:
                steps.append({
                    "action": s.action or "",
                    "selector": s.selector if s.HasField("selector") else None,
                    "value": s.value if s.HasField("value") else None,
                    "wait_for": s.wait_for if s.HasField("wait_for") else None,
                    "timeout_seconds": s.timeout_seconds if s.HasField("timeout_seconds") else None,
                    "url": s.url if s.HasField("url") else None,
                })
            result = await browser_run(
                user_id=request.user_id or "system",
                url=request.url or "",
                final_action_type="download",
                final_selector=request.download_selector or "",
                folder_path=request.folder_path or "Downloads",
                steps=steps if steps else None,
                connection_id=request.connection_id if request.HasField("connection_id") and request.connection_id else None,
                tags=list(request.tags) if request.tags else None,
                title=request.title if request.HasField("title") and request.title else None,
                goal=request.goal if request.HasField("goal") and request.goal else None,
            )
            response = tool_service_pb2.BrowserDownloadToolResponse()
            response.success = result.get("success", False)
            response.document_id = result.get("document_id", "")
            response.filename = result.get("filename", "")
            response.file_size_bytes = result.get("file_size_bytes", 0)
            if result.get("error"):
                response.error = result["error"]
            return response
        except Exception as e:
            logger.error(f"BrowserDownload error: {e}")
            response = tool_service_pb2.BrowserDownloadToolResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserOpenSession(
        self,
        request: tool_service_pb2.BrowserOpenSessionRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserOpenSessionResponse:
        """Create browser session; restore saved state for user/site if available."""
        try:
            from clients.crawl_service_client import get_crawl_service_client
            from services.browser_session_state_service import get_browser_session_state_service
            user_id = request.user_id or "system"
            site_domain = request.site_domain or ""
            state_svc = get_browser_session_state_service()
            state_json = await state_svc.load_session_state(user_id, site_domain) if site_domain else None
            client = await get_crawl_service_client()
            session_id = await client.browser_create_session(
                timeout_seconds=request.timeout_seconds or 30,
                storage_state_json=state_json,
            )
            response = tool_service_pb2.BrowserOpenSessionResponse()
            if session_id:
                response.success = True
                response.session_id = session_id
            else:
                response.success = False
                response.error = "Failed to create browser session"
            return response
        except Exception as e:
            logger.error(f"BrowserOpenSession error: {e}")
            response = tool_service_pb2.BrowserOpenSessionResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserNavigate(
        self,
        request: tool_service_pb2.BrowserNavigateRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserNavigateResponse:
        try:
            from clients.crawl_service_client import get_crawl_service_client
            client = await get_crawl_service_client()
            result = await client.browser_execute_action(
                request.session_id, "navigate", url=request.url or ""
            )
            response = tool_service_pb2.BrowserNavigateResponse()
            response.success = result.get("success", False)
            response.current_url = request.url or ""
            if result.get("error"):
                response.error = result["error"]
            return response
        except Exception as e:
            logger.error(f"BrowserNavigate error: {e}")
            response = tool_service_pb2.BrowserNavigateResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserClick(
        self,
        request: tool_service_pb2.BrowserClickRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserClickResponse:
        try:
            from clients.crawl_service_client import get_crawl_service_client
            client = await get_crawl_service_client()
            result = await client.browser_execute_action(
                request.session_id, "click", selector=request.selector or ""
            )
            response = tool_service_pb2.BrowserClickResponse()
            response.success = result.get("success", False)
            if result.get("error"):
                response.error = result["error"]
            return response
        except Exception as e:
            logger.error(f"BrowserClick error: {e}")
            response = tool_service_pb2.BrowserClickResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserFill(
        self,
        request: tool_service_pb2.BrowserFillRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserFillResponse:
        try:
            from clients.crawl_service_client import get_crawl_service_client
            client = await get_crawl_service_client()
            result = await client.browser_execute_action(
                request.session_id, "fill",
                selector=request.selector or "",
                value=request.value or "",
            )
            response = tool_service_pb2.BrowserFillResponse()
            response.success = result.get("success", False)
            if result.get("error"):
                response.error = result["error"]
            return response
        except Exception as e:
            logger.error(f"BrowserFill error: {e}")
            response = tool_service_pb2.BrowserFillResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserWait(
        self,
        request: tool_service_pb2.BrowserWaitRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserWaitResponse:
        try:
            from clients.crawl_service_client import get_crawl_service_client
            client = await get_crawl_service_client()
            result = await client.browser_execute_action(
                request.session_id, "wait",
                wait_selector=request.selector if request.HasField("selector") and request.selector else None,
                wait_timeout_seconds=request.timeout_seconds if request.HasField("timeout_seconds") else None,
            )
            response = tool_service_pb2.BrowserWaitResponse()
            response.success = result.get("success", False)
            response.found = result.get("success", False)
            if result.get("error"):
                response.error = result["error"]
            return response
        except Exception as e:
            logger.error(f"BrowserWait error: {e}")
            response = tool_service_pb2.BrowserWaitResponse()
            response.success = False
            response.found = False
            response.error = str(e)
            return response

    async def BrowserScroll(
        self,
        request: tool_service_pb2.BrowserScrollRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserScrollResponse:
        try:
            from clients.crawl_service_client import get_crawl_service_client
            client = await get_crawl_service_client()
            result = await client.browser_execute_action(
                request.session_id,
                "scroll",
                direction=request.direction if request.direction else "down",
                amount_pixels=request.amount_pixels if request.amount_pixels > 0 else 800,
            )
            response = tool_service_pb2.BrowserScrollResponse()
            response.success = result.get("success", False)
            if result.get("error"):
                response.error = result["error"]
            return response
        except Exception as e:
            logger.error(f"BrowserScroll error: {e}")
            response = tool_service_pb2.BrowserScrollResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserExtract(
        self,
        request: tool_service_pb2.BrowserExtractRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserExtractResponse:
        try:
            from clients.crawl_service_client import get_crawl_service_client
            client = await get_crawl_service_client()
            result = await client.browser_execute_action(
                request.session_id, "extract", selector=request.selector or ""
            )
            response = tool_service_pb2.BrowserExtractResponse()
            response.success = result.get("success", False)
            if result.get("extracted_content") is not None:
                response.extracted_text = result["extracted_content"]
            if result.get("error"):
                response.error = result["error"]
            return response
        except Exception as e:
            logger.error(f"BrowserExtract error: {e}")
            response = tool_service_pb2.BrowserExtractResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserInspect(
        self,
        request: tool_service_pb2.BrowserInspectRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserInspectResponse:
        try:
            from clients.crawl_service_client import get_crawl_service_client
            client = await get_crawl_service_client()
            result = await client.browser_inspect_page(request.session_id)
            response = tool_service_pb2.BrowserInspectResponse()
            response.success = result.get("success", False)
            if result.get("page_structure"):
                response.page_structure = result["page_structure"]
            if result.get("error"):
                response.error = result["error"]
            return response
        except Exception as e:
            logger.error(f"BrowserInspect error: {e}")
            response = tool_service_pb2.BrowserInspectResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserScreenshot(
        self,
        request: tool_service_pb2.BrowserScreenshotRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserScreenshotResponse:
        try:
            import base64
            from clients.crawl_service_client import get_crawl_service_client
            from services.langgraph_tools.file_creation_tools import create_user_file
            client = await get_crawl_service_client()
            result = await client.browser_execute_action(
                request.session_id, "screenshot"
            )
            response = tool_service_pb2.BrowserScreenshotResponse()
            if not result.get("success"):
                response.success = False
                response.error = result.get("error", "Screenshot failed")
                return response
            png_bytes = result.get("screenshot_png") or b""
            if not png_bytes:
                response.success = False
                response.error = "Screenshot produced no image"
                return response
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            response.images_markdown = f"![Screenshot](data:image/png;base64,{b64})"
            response.success = True
            if request.folder_path:
                filename = request.title if request.HasField("title") and request.title else f"screenshot_{int(__import__('time').time())}.png"
                create_result = await create_user_file(
                    filename=filename,
                    content="",
                    folder_path=request.folder_path,
                    title=request.title if request.HasField("title") and request.title else filename,
                    tags=list(request.tags) if request.tags else [],
                    user_id=request.user_id or "system",
                    content_bytes=png_bytes,
                )
                if create_result.get("success"):
                    response.document_id = create_result.get("document_id", "")
                    response.filename = create_result.get("filename", filename)
                    response.file_size_bytes = len(png_bytes)
            return response
        except Exception as e:
            logger.error(f"BrowserScreenshot error: {e}")
            response = tool_service_pb2.BrowserScreenshotResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserDownloadFile(
        self,
        request: tool_service_pb2.BrowserDownloadFileRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserDownloadFileResponse:
        try:
            from clients.crawl_service_client import get_crawl_service_client
            from services.langgraph_tools.file_creation_tools import create_user_file
            client = await get_crawl_service_client()
            download_result = await client.browser_download_file(
                session_id=request.session_id,
                trigger_selector=request.selector or "",
                timeout_seconds=60,
            )
            response = tool_service_pb2.BrowserDownloadFileResponse()
            if not download_result.get("success"):
                response.success = False
                response.error = download_result.get("error", "Download failed")
                return response
            file_content = download_result.get("file_content") or b""
            filename = download_result.get("filename") or "download"
            if not file_content:
                response.success = False
                response.error = "Download produced no content"
                return response
            create_result = await create_user_file(
                filename=filename,
                content="",
                folder_path=request.folder_path or "Downloads",
                title=request.title if request.HasField("title") and request.title else filename,
                tags=list(request.tags) if request.tags else [],
                user_id=request.user_id or "system",
                content_bytes=file_content,
            )
            response.success = create_result.get("success", False)
            if create_result.get("document_id"):
                response.document_id = create_result["document_id"]
            if create_result.get("filename"):
                response.filename = create_result["filename"]
            response.file_size_bytes = len(file_content)
            if create_result.get("error"):
                response.error = create_result["error"]
            return response
        except Exception as e:
            logger.error(f"BrowserDownloadFile error: {e}")
            response = tool_service_pb2.BrowserDownloadFileResponse()
            response.success = False
            response.error = str(e)
            return response

    async def BrowserCloseSession(
        self,
        request: tool_service_pb2.BrowserCloseSessionRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.BrowserCloseSessionResponse:
        try:
            from clients.crawl_service_client import get_crawl_service_client
            from services.browser_session_state_service import get_browser_session_state_service
            client = await get_crawl_service_client()
            session_saved = False
            if request.save_state and request.site_domain:
                state_json = await client.browser_save_session_state(request.session_id)
                if state_json:
                    state_svc = get_browser_session_state_service()
                    session_saved = await state_svc.save_session_state(
                        request.user_id or "system",
                        request.site_domain,
                        state_json,
                    )
            await client.browser_destroy_session(request.session_id)
            response = tool_service_pb2.BrowserCloseSessionResponse()
            response.success = True
            response.session_saved = session_saved
            return response
        except Exception as e:
            logger.error(f"BrowserCloseSession error: {e}")
            response = tool_service_pb2.BrowserCloseSessionResponse()
            response.success = False
            response.session_saved = False
            response.error = str(e)
            return response

    async def _store_crawled_website(
        self,
        crawl_result: Dict[str, Any],
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Store crawled website content as documents (same logic as backend agent)"""
        try:
            logger.info("Storing crawled website content")
            
            from services.service_container import get_service_container
            from urllib.parse import urlparse
            import hashlib

            container = await get_service_container()
            doc_service = container.document_service
            
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
                                logger.info(f"Created browseable file for: {page_title}")
                            except Exception as e:
                                logger.warning(f"Failed to create browseable file for {page['url']}: {e}")
                        
                    elif content_type == "image":
                        # Store image binary file
                        binary_content = page.get("binary_content")
                        filename = page.get("filename", "image")
                        
                        if binary_content:
                            # Save crawled image to web_sources volume
                            upload_dir = Path(settings.WEB_SOURCES_ROOT) / "images" / website_name
                            upload_dir.mkdir(parents=True, exist_ok=True)
                            
                            safe_filename = filename.replace("/", "_").replace("\\", "_")
                            file_path = upload_dir / f"{doc_id}_{safe_filename}"
                            
                            with open(file_path, 'wb') as f:
                                f.write(binary_content)
                            
                            logger.info(f"Saved image: {file_path}")
                            
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
                            # Save crawled document binary to web_sources volume
                            upload_dir = Path(settings.WEB_SOURCES_ROOT) / "documents" / website_name
                            upload_dir.mkdir(parents=True, exist_ok=True)
                            
                            safe_filename = filename.replace("/", "_").replace("\\", "_")
                            file_path = upload_dir / f"{doc_id}_{safe_filename}"
                            
                            with open(file_path, 'wb') as f:
                                f.write(binary_content)
                            
                            logger.info(f"Saved document: {file_path}")
                            
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
                        logger.warning(f"Failed to store item: {page['url']}")
                    
                except Exception as e:
                    logger.error(f"Error storing item {page.get('url', 'unknown')}: {e}")
                    failed_count += 1
            
            logger.info(f"Stored {stored_count}/{len(crawled_pages)} items ({images_stored} images, {documents_stored} documents)")
            
            return {
                "success": True,
                "stored_count": stored_count,
                "failed_count": failed_count,
                "total_items": len(crawled_pages),
                "images_stored": images_stored,
                "documents_stored": documents_stored
            }
            
        except Exception as e:
            logger.error(f"Failed to store crawled website: {e}")
            return {
                "success": False,
                "error": str(e),
                "stored_count": 0
            }
