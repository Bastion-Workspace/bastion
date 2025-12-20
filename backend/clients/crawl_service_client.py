"""
Crawl4AI Service gRPC Client

Provides client interface to the Crawl4AI Service for web crawling.
"""

import grpc
import logging
from typing import List, Dict, Any, Optional

from config import get_settings
from protos import crawl_service_pb2, crawl_service_pb2_grpc

logger = logging.getLogger(__name__)


class CrawlServiceClient:
    """Client for interacting with the Crawl4AI Service via gRPC"""
    
    def __init__(self, service_host: Optional[str] = None, service_port: Optional[int] = None):
        """
        Initialize Crawl4AI Service client
        
        Args:
            service_host: gRPC service host (default: from config)
            service_port: gRPC service port (default: from config)
        """
        self.settings = get_settings()
        self.service_host = service_host or getattr(self.settings, 'CRAWL4AI_SERVICE_HOST', 'crawl4ai-service')
        self.service_port = service_port or getattr(self.settings, 'CRAWL4AI_SERVICE_PORT', 50055)
        self.service_url = f"{self.service_host}:{self.service_port}"
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[crawl_service_pb2_grpc.CrawlServiceStub] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the gRPC channel and stub"""
        if self._initialized:
            return
        
        try:
            logger.info(f"Connecting to Crawl4AI Service at {self.service_url}")
            
            # Create insecure channel with increased message size limits
            # Default is 4MB, increase to 100MB for large crawl responses
            options = [
                ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100 MB
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100 MB
            ]
            self.channel = grpc.aio.insecure_channel(self.service_url, options=options)
            self.stub = crawl_service_pb2_grpc.CrawlServiceStub(self.channel)
            
            # Test connection
            health_request = crawl_service_pb2.HealthCheckRequest()
            response = await self.stub.HealthCheck(health_request, timeout=5.0)
            
            if response.status == "healthy":
                logger.info(f"✅ Connected to Crawl4AI Service v{response.service_version}")
                logger.info(f"   Crawl4AI Available: {response.crawl4ai_available}")
                logger.info(f"   Browser Available: {response.browser_available}")
                self._initialized = True
            else:
                logger.warning(f"⚠️ Crawl4AI Service health check returned: {response.status}")
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to Crawl4AI Service: {e}")
            raise
    
    async def close(self):
        """Close the gRPC channel"""
        if self.channel:
            await self.channel.close()
            self._initialized = False
            logger.info("Crawl4AI Service client closed")
    
    async def crawl(
        self,
        url: str,
        extraction_strategy: str = "markdown",
        chunking_strategy: str = "RegexChunking",
        css_selector: Optional[str] = None,
        llm_question: Optional[str] = None,
        max_content_length: Optional[int] = None,
        include_links: bool = True,
        include_metadata: bool = True,
        timeout_seconds: Optional[int] = None,
        virtual_scroll: bool = False,
        scroll_delay: float = 1.0,
        use_fit_markdown: bool = False,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Crawl a single URL
        
        Args:
            url: URL to crawl
            extraction_strategy: Extraction strategy ("markdown", "text", "html", "llm_extraction")
            chunking_strategy: Chunking strategy
            css_selector: Optional CSS selector for content extraction
            llm_question: Optional question for LLM extraction
            max_content_length: Maximum content length
            include_links: Include links in response
            include_metadata: Include metadata in response
            timeout_seconds: Timeout in seconds
            virtual_scroll: Enable virtual scroll for infinite-scroll pages
            scroll_delay: Delay between scrolls
            use_fit_markdown: Use fit markdown for LLM optimization
            user_id: Optional user ID
            
        Returns:
            Dictionary with crawl results
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            request = crawl_service_pb2.CrawlRequest(
                url=url,
                extraction_strategy=extraction_strategy,
                chunking_strategy=chunking_strategy,
                css_selector=css_selector or "",
                llm_question=llm_question or "",
                max_content_length=max_content_length or 0,
                include_links=include_links,
                include_metadata=include_metadata,
                timeout_seconds=timeout_seconds or 0,
                virtual_scroll=virtual_scroll,
                scroll_delay=scroll_delay,
                use_fit_markdown=use_fit_markdown,
                user_id=user_id or ""
            )
            
            response = await self.stub.Crawl(request, timeout=timeout_seconds or 120.0)
            
            return {
                "success": response.success,
                "url": response.url,
                "title": response.title,
                "content": response.content,
                "markdown": response.markdown,
                "html": response.html,
                "metadata": dict(response.metadata) if response.metadata else {},
                "links": list(response.links) if response.links else [],
                "images": list(response.images) if response.images else [],
                "content_length": response.content_length,
                "fetch_time_seconds": response.fetch_time_seconds,
                "status_code": response.status_code,
                "error": response.error if response.error else None,
                "extracted_content": response.extracted_content if response.extracted_content else None
            }
            
        except grpc.RpcError as e:
            logger.error(f"❌ gRPC error crawling {url}: {e.code()}: {e.details()}")
            raise
        except Exception as e:
            logger.error(f"❌ Error crawling {url}: {e}")
            raise
    
    async def crawl_many(
        self,
        urls: List[str],
        extraction_strategy: str = "markdown",
        chunking_strategy: str = "RegexChunking",
        max_concurrent: int = 5,
        rate_limit_seconds: float = 1.0,
        css_selector: Optional[str] = None,
        llm_question: Optional[str] = None,
        max_content_length: Optional[int] = None,
        include_links: bool = True,
        include_metadata: bool = True,
        timeout_seconds: Optional[int] = None,
        virtual_scroll: bool = False,
        scroll_delay: float = 1.0,
        use_fit_markdown: bool = False,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Crawl multiple URLs in parallel
        
        Args:
            urls: List of URLs to crawl
            extraction_strategy: Extraction strategy
            chunking_strategy: Chunking strategy
            max_concurrent: Maximum concurrent crawls
            rate_limit_seconds: Delay between requests
            css_selector: Optional CSS selector
            llm_question: Optional question for LLM extraction
            max_content_length: Maximum content length
            include_links: Include links in response
            include_metadata: Include metadata in response
            timeout_seconds: Timeout in seconds
            virtual_scroll: Enable virtual scroll
            scroll_delay: Delay between scrolls
            use_fit_markdown: Use fit markdown
            user_id: Optional user ID
            
        Returns:
            Dictionary with crawl results
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            request = crawl_service_pb2.CrawlManyRequest(
                urls=urls,
                extraction_strategy=extraction_strategy,
                chunking_strategy=chunking_strategy,
                max_concurrent=max_concurrent,
                rate_limit_seconds=rate_limit_seconds,
                css_selector=css_selector or "",
                llm_question=llm_question or "",
                max_content_length=max_content_length or 0,
                include_links=include_links,
                include_metadata=include_metadata,
                timeout_seconds=timeout_seconds or 0,
                virtual_scroll=virtual_scroll,
                scroll_delay=scroll_delay,
                use_fit_markdown=use_fit_markdown,
                user_id=user_id or ""
            )
            
            response = await self.stub.CrawlMany(request, timeout=300.0)
            
            # Convert results to dictionaries
            results = []
            for result in response.results:
                results.append({
                    "success": result.success,
                    "url": result.url,
                    "title": result.title,
                    "content": result.content,
                    "markdown": result.markdown,
                    "html": result.html,
                    "metadata": dict(result.metadata) if result.metadata else {},
                    "links": list(result.links) if result.links else [],
                    "images": list(result.images) if result.images else [],
                    "content_length": result.content_length,
                    "fetch_time_seconds": result.fetch_time_seconds,
                    "status_code": result.status_code,
                    "error": result.error if result.error else None,
                    "extracted_content": result.extracted_content if result.extracted_content else None
                })
            
            return {
                "success": response.success,
                "results": results,
                "urls_requested": response.urls_requested,
                "successful_crawls": response.successful_crawls,
                "failed_crawls": response.failed_crawls,
                "total_content_length": response.total_content_length,
                "total_time_seconds": response.total_time_seconds,
                "error": response.error if response.error else None
            }
            
        except grpc.RpcError as e:
            logger.error(f"❌ gRPC error crawling many URLs: {e.code()}: {e.details()}")
            raise
        except Exception as e:
            logger.error(f"❌ Error crawling many URLs: {e}")
            raise
    
    async def adaptive_crawl(
        self,
        seed_url: str,
        query: str,
        max_depth: int = 3,
        max_pages: int = 50,
        stop_when_satisfied: bool = True,
        extraction_strategy: str = "markdown",
        use_fit_markdown: bool = False,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Adaptive intelligent crawl
        
        Args:
            seed_url: Starting URL
            query: Query to find information
            max_depth: Maximum crawl depth
            max_pages: Maximum pages to crawl
            stop_when_satisfied: Stop when enough info gathered
            extraction_strategy: Extraction strategy
            use_fit_markdown: Use fit markdown
            user_id: Optional user ID
            
        Returns:
            Dictionary with crawl results
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            request = crawl_service_pb2.AdaptiveCrawlRequest(
                seed_url=seed_url,
                query=query,
                max_depth=max_depth,
                max_pages=max_pages,
                stop_when_satisfied=stop_when_satisfied,
                extraction_strategy=extraction_strategy,
                use_fit_markdown=use_fit_markdown,
                user_id=user_id or ""
            )
            
            response = await self.stub.AdaptiveCrawl(request, timeout=300.0)
            
            return {
                "success": response.success,
                "url": response.url,
                "title": response.title,
                "content": response.content,
                "markdown": response.markdown,
                "html": response.html,
                "metadata": dict(response.metadata) if response.metadata else {},
                "links": list(response.links) if response.links else [],
                "images": list(response.images) if response.images else [],
                "content_length": response.content_length,
                "fetch_time_seconds": response.fetch_time_seconds,
                "status_code": response.status_code,
                "error": response.error if response.error else None,
                "extracted_content": response.extracted_content if response.extracted_content else None
            }
            
        except grpc.RpcError as e:
            logger.error(f"❌ gRPC error in adaptive crawl: {e.code()}: {e.details()}")
            raise
        except Exception as e:
            logger.error(f"❌ Error in adaptive crawl: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check service health
        
        Returns:
            Dictionary with health status
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            request = crawl_service_pb2.HealthCheckRequest()
            response = await self.stub.HealthCheck(request, timeout=5.0)
            
            return {
                "status": response.status,
                "crawl4ai_available": response.crawl4ai_available,
                "browser_available": response.browser_available,
                "service_version": response.service_version,
                "details": dict(response.details) if response.details else {}
            }
            
        except grpc.RpcError as e:
            logger.error(f"❌ gRPC error checking health: {e.code()}: {e.details()}")
            raise
        except Exception as e:
            logger.error(f"❌ Error checking health: {e}")
            raise


# Singleton instance
_crawl_service_client: Optional[CrawlServiceClient] = None

async def get_crawl_service_client() -> CrawlServiceClient:
    """Get or create singleton Crawl4AI Service client"""
    global _crawl_service_client
    
    if _crawl_service_client is None:
        _crawl_service_client = CrawlServiceClient()
        await _crawl_service_client.initialize()
    
    return _crawl_service_client







